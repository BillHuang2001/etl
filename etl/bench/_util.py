"""Shared private helpers for ``etl.bench`` (stdlib + numpy + etl only).

Runner contract (binding) — see :func:`stage_example` for routing:

``Example.runner(backend: str, device: core.Device, opts: dict) ->
callable(inputs) -> outputs`` is an optional runner FACTORY for multi-run
end-to-end procedures (e.g. a Python-level training loop). It receives the
resolved backend name, the resolved :class:`~etl.core.Device`, and the
resolved backend options dict (with the device-derived ``target_backends``
default and the ``opt_level`` harness default (``"O3"`` for compiler
backends; env-aware, explicit always wins) already injected, exactly what
``stage_example``'s single-run path passes to ``etl.build``), and returns a
RUN-CALLABLE taking the SAME inputs
list the single-run path uses (``example.generate_inputs(seed)``, a list of
numpy arrays) and returning the same outputs structure as ``numpy_ref``
(single ndarray or tuple). The runner builds its executables ONCE (its own
staging: ``@etl.defn`` graph → ``etl.build(..., backend=backend,
device=device, **opts)``; TransformCallable/Graph-builder →
``graph(*specs)`` → ``etl.lower`` → ``etl.compile`` → ``etl.load``) and runs
them N times internally. Runner bodies MUST NEVER call
:func:`stage_example` — ``stage_example`` routes examples with a ``runner``
straight to the runner factory, so a runner calling it would recurse
infinitely.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

import etl
from etl import backends, core
from ._torch import require_torch, torch_available
from .examples import expand_names, list_examples

__all__ = [
    "resolve_examples",
    "stage_example",
    "resolve_torch_mode",
    "best_time_ms",
    "flatten_outputs",
    "resolve_device",
    "format_device",
    "resolve_backend",
    "resolve_backend_options",
    "resolve_torch_device",
]


def resolve_device(device=None) -> core.Device:
    """Normalize a device argument to a :class:`~etl.core.Device`.

    ``None`` → ``core.Device("cpu", 0)``. A ``core.Device`` is validated
    (kind must be ``"cpu"`` or ``"cuda"``) and returned as-is. A string
    parses as ``"KIND[:INDEX]"`` (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:3"``);
    the kind must be ``"cpu"`` or ``"cuda"`` and the index a non-negative
    integer. Anything else raises ``TypeError``; a bad kind or index raises
    ``ValueError`` with a clear message.
    """
    if device is None:
        return core.Device("cpu", 0)
    if isinstance(device, core.Device):
        if device.kind not in ("cpu", "cuda"):
            raise ValueError(
                f"unsupported device kind {device.kind!r}: supported kinds "
                "are 'cpu' and 'cuda'"
            )
        return device
    if not isinstance(device, str):
        raise TypeError(
            f"device must be None, a core.Device, or a 'KIND[:INDEX]' string "
            f"(e.g. 'cpu', 'cuda:3'), got {type(device).__name__}"
        )
    kind, separator, index_part = device.partition(":")
    if separator:
        if ":" in index_part or not index_part:
            raise ValueError(
                f"invalid device {device!r}: expected 'KIND[:INDEX]' "
                "(e.g. 'cpu', 'cuda', 'cuda:3')"
            )
        try:
            index = int(index_part)
        except ValueError:
            raise ValueError(
                f"invalid device {device!r}: index {index_part!r} is not an "
                "integer"
            ) from None
        if index < 0:
            raise ValueError(
                f"invalid device {device!r}: index must be a non-negative "
                "integer"
            )
    else:
        index = 0
    if kind not in ("cpu", "cuda"):
        raise ValueError(
            f"unsupported device kind {kind!r} in {device!r}: supported "
            "kinds are 'cpu' and 'cuda'"
        )
    return core.Device(kind, index)


def format_device(device) -> str:
    """Format a device as a string: ``"cpu"`` for ``Device("cpu", 0)``,
    otherwise ``"KIND:INDEX"`` (e.g. ``"cuda:3"``)."""
    if device.kind == "cpu" and device.index == 0:
        return "cpu"
    return f"{device.kind}:{device.index}"


def resolve_backend(backend) -> str:
    """Validate a backend name against the etl backend registry.

    Must be a non-empty string (else ``TypeError``).
    ``etl.backends.get`` validates registration and auto-activates optional
    compiler adapters; its ``core.BackendError`` for unknown names or
    missing adapter dependencies propagates unchanged. Returns the backend
    name.
    """
    if not isinstance(backend, str) or not backend:
        raise TypeError(f"backend must be a non-empty string, got {backend!r}")
    backends.get(backend)  # validates; raises core.BackendError
    return backend


def resolve_backend_options(backend, device, backend_options) -> dict:
    """Resolve backend compile options for a chosen backend/device.

    Returns a NEW dict ``{**backend_options}``. For every non-numpy backend
    (numpy is the only interpreter backend; all others are compiler
    backends) without an explicit ``target_backends`` option, the
    device-derived default is injected: ``["cuda"]`` for a cuda device,
    ``["llvm-cpu"]`` otherwise. An explicit option always wins — never
    overridden.

    For every non-numpy backend, when ``opt_level`` is absent from the
    options AND the ``ETL_OPT_LEVEL`` env var is unset/blank, the harness
    default ``"O3"`` is injected (high optimization for compiler backends).
    An explicit ``opt_level`` always wins; ``ETL_OPT_LEVEL`` wins over the
    harness default (injection is skipped when the env var is set, so the
    pipeline's env machinery applies it at compile). numpy is never
    affected — it is the reference interpreter.
    """
    options = dict(backend_options or {})
    if backend != "numpy":
        if "target_backends" not in options:
            options["target_backends"] = (
                ["cuda"] if device.kind == "cuda" else ["llvm-cpu"]
            )
        if (
            "opt_level" not in options
            and not (os.environ.get("ETL_OPT_LEVEL") or "").strip()
        ):
            options["opt_level"] = "O3"
    return options


def resolve_torch_device(device, torch_mod):
    """Resolve the torch device for an etl device, or ``None``.

    Returns ``None`` when the device is not cuda, torch has no CUDA support
    (``cuda.is_available()`` False), or the device index is beyond
    ``torch.cuda.device_count()`` — torch references then run on CPU
    (``device=None``), exactly today's behavior. Receives the already
    imported torch module — never imports torch itself (torch optionality
    is binding).
    """
    if device.kind != "cuda":
        return None
    if not torch_mod.cuda.is_available():
        return None
    if device.index >= torch_mod.cuda.device_count():
        return None
    return torch_mod.device(f"cuda:{device.index}")


def resolve_examples(examples):
    """Normalize the ``examples`` argument to a list of registered names.

    ``None`` → all registered examples; a ``str`` → a single entry; any
    iterable → a list. Each entry is then expanded via
    :func:`~etl.bench.examples.expand_names`: a category name (e.g. ``'grad'``,
    ``'large'``) expands to all its example names (registry order); an exact
    example name is kept as-is; a tag name expands to all examples carrying
    that tag. Unknown names raise
    :class:`~etl.bench.examples.UnknownExampleError` (a ``ValueError``)
    listing the available names.
    """
    if examples is None:
        return list(list_examples())
    if isinstance(examples, str):
        names = [examples]
    else:
        names = list(examples)
    return expand_names(names)


def stage_example(example, backend, device, opts) -> callable:
    """Stage an example's graph and return a RUN-CALLABLE ``run(inputs)``.

    The returned callable takes the SAME inputs list the single-run path
    uses (``example.generate_inputs(seed)``, a list of numpy arrays) and
    returns the same outputs structure as ``example.numpy_ref`` (single
    ndarray or tuple). Routing (documented):

    - ``example.runner`` set → ``return example.runner(backend, device, opts)``
      (the runner factory builds its own executables ONCE — see the runner
      contract in this module's docstring; a runner must NEVER call
      ``stage_example``, that would recurse infinitely).
    - ``@etl.defn`` graphs → ``etl.build(..., backend=backend,
      device=device, **opts)`` then ``lambda inputs: etl.run(executable,
      *inputs)``.
    - Transform-produced graphs (``etl.grad``/``etl.vmap`` TransformCallables
      — ``example.graph`` lacks the ``__etl_defn__`` marker) are materialized
      with ``example.graph(*example.specs) -> Graph`` and staged through the
      explicit pipeline ``etl.lower`` → ``etl.compile`` → ``etl.load``
      (options go to BOTH lower and compile, exactly like build does), then
      the same run lambda.
    """
    if example.runner is not None:
        return example.runner(backend, device, opts)
    if getattr(example.graph, "__etl_defn__", False):
        executable = etl.build(
            example.graph, *example.specs,
            backend=backend, device=device, **opts
        )
    else:
        graph = example.graph(*example.specs)
        lp = etl.lower(graph, backend=backend, **opts)
        ca = etl.compile(lp, **opts)
        executable = etl.load(ca, device=device)
    return lambda inputs: etl.run(executable, *inputs)


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
