"""Shared private helpers for ``etl.bench`` (stdlib + numpy + etl only)."""
from __future__ import annotations

import math
import time

import numpy as np

from etl import core
from ._torch import require_torch, torch_available
from .examples import get_example, list_examples

__all__ = [
    "resolve_examples",
    "resolve_torch_mode",
    "best_time_ms",
    "flatten_outputs",
]


def resolve_examples(examples):
    """Normalize the ``examples`` argument to a list of registered names.

    ``None`` → all registered examples; a ``str`` → a single name; any
    iterable of names → a list. Unknown names raise
    :class:`~etl.bench.examples.UnknownExampleError` (a ``ValueError``)
    listing the available names.
    """
    if examples is None:
        return list(list_examples())
    if isinstance(examples, str):
        names = [examples]
    else:
        names = list(examples)
    for name in names:
        get_example(name)  # validates; raises UnknownExampleError if unknown
    return names


def resolve_torch_mode(use_torch):
    """Resolve the ``use_torch`` argument to ``(mode, enabled, available)``.

    ``mode`` is ``"auto"`` | ``"enabled"`` | ``"disabled"``; ``enabled`` says
    whether torch references will actually run; ``available`` says whether
    torch imports. ``use_torch=None`` → ``"auto"`` (enabled iff torch
    imports). ``use_torch=True`` → ``"enabled"`` and raises a clear
    ``ImportError`` with the ``pip install etl[bench]`` hint when torch is
    unavailable. ``use_torch=False`` → ``"disabled"``.
    """
    if use_torch is True:
        require_torch()  # raises a clear ImportError when torch is absent
        return "enabled", True, True
    if use_torch is False:
        return "disabled", False, torch_available()
    available = torch_available()
    return ("enabled" if available else "disabled"), available, available


def best_time_ms(fn, warmup: int, repeats: int) -> float:
    """Time ``fn``: ``warmup`` untimed runs, then ``repeats`` timed runs.

    Returns the BEST (minimum) run time in milliseconds — best-of-N, so the
    reported number reflects the fastest observed run (least noise), not the
    mean.
    """
    for _ in range(warmup):
        fn()
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def flatten_outputs(outputs):
    """Flatten structured outputs to a list of numpy ndarrays.

    Handles ``etl.Tensor`` (→ ``.numpy()``), ``numpy.ndarray``, tuples /
    lists / namedtuples (recursed), and dicts (key order). Anything else
    raises ``TypeError`` — explicit, never silently dropped.
    """
    if isinstance(outputs, core.Tensor):
        return [outputs.numpy()]
    if isinstance(outputs, np.ndarray):
        return [outputs]
    if isinstance(outputs, (tuple, list)):
        flat = []
        for item in outputs:
            flat.extend(flatten_outputs(item))
        return flat
    if isinstance(outputs, dict):
        flat = []
        for key in sorted(outputs):
            flat.extend(flatten_outputs(outputs[key]))
        return flat
    raise TypeError(
        f"cannot interpret {type(outputs).__name__} as tensor outputs "
        "(expected etl.Tensor / numpy.ndarray / tuple / list / dict)"
    )
