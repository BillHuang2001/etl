"""Conformance checking: etl graphs vs numpy (and optional torch) references.

Each selected example is staged through the explicit pipeline
(``etl.build`` + ``etl.run`` on the default numpy backend — or a chosen
``backend``/``device`` with ``backend_options`` passthrough) with inputs
generated from a seeded RNG, then compared elementwise against the example's
pure-numpy reference — and, when torch is available/requested, against its
torch reference. Staging goes through :func:`~etl.bench._util.stage_example`,
which returns a RUN-CALLABLE ``run(inputs)`` per example: ``@etl.defn``
graphs are routed through ``etl.build``, transform-produced graphs
(``etl.grad``/``etl.vmap`` TransformCallables) through the explicit
``etl.lower`` → ``etl.compile`` → ``etl.load`` pipeline, and examples with
an ``Example.runner`` set (e2e multi-run procedures) through the runner
factory (see the runner contract in ``etl.bench._util``'s docstring).

Per-example execution failures are recorded in the report (``error`` field,
``overall_pass`` False) rather than aborting the whole run — nothing is
silently swallowed, every failure is visible in the report. Argument errors
(unknown backend, bad device) raise up front instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._torch import require_torch
from ._util import (
    flatten_outputs,
    format_device,
    resolve_backend,
    resolve_backend_options,
    resolve_device,
    resolve_examples,
    resolve_torch_device,
    resolve_torch_mode,
    stage_example,
)
from .examples import get_example
from .report import ConformanceReport, ExampleResult

__all__ = ["conformance"]


@dataclass
class _Comparison:
    max_abs_error: float
    max_rel_error: float
    passed: bool


def _compare(actual, expected, rtol: float, atol: float,
             tolerance: Optional[float]) -> _Comparison:
    """Elementwise comparison of flattened output lists (float64).

    Rule (documented): with ``tolerance=None`` the check passes iff every
    element satisfies ``|a - b| <= atol + rtol * |b|`` (numpy-allclose-style,
    computed in float64). With a numeric ``tolerance`` the check passes iff
    the max absolute error is ``<= tolerance``. NaN anywhere fails the check
    and yields NaN error values. ``max_rel_error`` is the max of
    ``|a - b| / |b|`` over elements with ``|b| > 0`` — 0 when the expected
    output is all zeros and the match is exact, ``inf`` when the match
    differs there.

    Raises:
        ValueError: output count or shape mismatch between etl and the
            reference (explicit, never broadcast silently).
    """
    if len(actual) != len(expected):
        raise ValueError(
            f"output count mismatch: etl produced {len(actual)} tensor(s), "
            f"reference produced {len(expected)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    has_nan = False
    ok = True
    for index, (a, b) in enumerate(zip(actual, expected)):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(
                f"output {index}: shape mismatch — etl produced {a.shape}, "
                f"reference produced {b.shape}"
            )
        diff = np.abs(a - b)
        if np.isnan(diff).any():
            has_nan = True
            continue
        abs_err = float(diff.max())
        max_abs = max(max_abs, abs_err)
        nonzero = b != 0
        if nonzero.any():
            rel_err = float(np.max(diff[nonzero] / np.abs(b[nonzero])))
        elif abs_err == 0.0:
            rel_err = 0.0
        else:
            rel_err = float("inf")
        max_rel = max(max_rel, rel_err)
        if tolerance is None:
            ok = ok and bool(np.all(diff <= atol + rtol * np.abs(b)))
        else:
            ok = ok and abs_err <= tolerance
    if has_nan:
        return _Comparison(float("nan"), float("nan"), False)
    return _Comparison(max_abs, max_rel, ok)


def conformance(examples=None, *, use_torch=None, tolerance=None,
                rtol=1e-5, atol=1e-5, seed=0, backend="numpy", device=None,
                **backend_options) -> ConformanceReport:
    """Run conformance checks for the selected examples.

    Args:
        examples: ``None`` (all registered examples), a single name or
            category name, or an iterable of names and/or category names
            (categories expand to their examples).
        use_torch: ``None`` = auto (torch comparisons run iff ``import torch``
            succeeds); ``True`` = require torch — raises a clear
            ``ImportError`` mentioning ``pip install etl[bench]`` when torch
            is unavailable; ``False`` = numpy-only.
        tolerance: optional absolute-error pass threshold (``None`` = the
            default ``rtol``/``atol`` allclose-style rule; see
            :func:`_compare`). Per-example overrides: when an example sets
            ``example.tolerance`` (not ``None``) it wins; the three
            tolerances are resolved independently.
        rtol: relative tolerance for the default rule (per-example
            ``example.rtol`` overrides it when set).
        atol: absolute tolerance for the default rule (per-example
            ``example.atol`` overrides it when set).
        seed: RNG seed for generated inputs (``numpy.random.default_rng``).
        backend: etl backend name to stage the graphs on (default
            ``"numpy"``; e.g. ``"iree"`` — validated up front through
            ``etl.backends.get``; unknown names / missing adapter deps raise
            ``core.BackendError``).
        device: device to run on — ``None`` (``Device("cpu", 0)``), a
            ``core.Device``, or a ``"KIND[:INDEX]"`` string (e.g. ``"cpu"``,
            ``"cuda"``, ``"cuda:3"``). The report records the formatted
            device.
        **backend_options: extra options passed through to ``etl.build``
            (compile options for compiler backends). For any non-numpy
            backend without an explicit ``target_backends`` option, the
            device-derived default is injected: ``["cuda"]`` on a cuda
            device, ``["llvm-cpu"]`` otherwise (an explicit option always
            wins).

    Per-example tolerance resolution (documented): each example's EFFECTIVE
    ``rtol``/``atol``/``tolerance`` is the example's own value when it is not
    ``None``, else the corresponding global argument — the three are resolved
    independently. Effective values are recorded on each
    :class:`ExampleResult` and used for BOTH the numpy and the torch
    comparison.

    Examples with ``Example.runner`` set (e2e multi-run procedures) are
    executed through the runner path: ``stage_example`` returns the runner
    factory's run-callable and ``run_fn(inputs)`` runs the whole multi-run
    procedure once (see the runner contract in ``etl.bench._util``'s
    docstring).

    Returns:
        :class:`ConformanceReport` with one :class:`ExampleResult` per
        example (``max_abs_error``, ``max_rel_error``, numpy/torch
        pass flags, effective per-example tolerances, etl run time in ms, or
        an ``error`` string).

    Raises:
        TypeError: invalid ``backend``/``device`` argument types.
        ValueError: unsupported device kind or malformed device string.
        BackendError: unknown backend name or missing adapter dependency
            (raised up front, before any example runs).
        ImportError: ``use_torch=True`` without torch (clear hint, never a
            raw ``ModuleNotFoundError``).
        UnknownExampleError: unknown example name or category (lists
            available names).
    """
    if tolerance is not None and not isinstance(tolerance, (int, float)):
        raise TypeError(
            f"tolerance must be a number or None, got {type(tolerance).__name__}"
        )
    dev = resolve_device(device)
    backend = resolve_backend(backend)
    opts = resolve_backend_options(backend, dev, backend_options)
    names = resolve_examples(examples)
    mode, enabled, available = resolve_torch_mode(use_torch)
    torch_device = None
    if enabled:
        torch_device = resolve_torch_device(dev, require_torch())
    results: List[ExampleResult] = []
    for name in names:
        example = get_example(name)
        # Effective per-example tolerances: the example's own value when set,
        # else the global argument (each resolved independently).
        e_rtol = example.rtol if example.rtol is not None else rtol
        e_atol = example.atol if example.atol is not None else atol
        e_tolerance = (
            example.tolerance if example.tolerance is not None else tolerance
        )
        try:
            inputs = example.generate_inputs(seed)
            run_fn = stage_example(example, backend, dev, opts)
            start = time.perf_counter()
            actual = flatten_outputs(run_fn(inputs))
            etl_ms = (time.perf_counter() - start) * 1000.0
            expected = flatten_outputs(example.numpy_ref(inputs))
            comparison = _compare(actual, expected, e_rtol, e_atol, e_tolerance)
            torch_pass = None
            if enabled:
                if example.torch_ref is None:
                    raise ValueError(
                        f"example {name!r} has no torch reference"
                    )
                torch_expected = flatten_outputs(
                    example.torch_ref(inputs, device=torch_device)
                )
                torch_comparison = _compare(
                    actual, torch_expected, e_rtol, e_atol, e_tolerance
                )
                torch_pass = torch_comparison.passed
            results.append(
                ExampleResult(
                    name=name,
                    description=example.description,
                    max_abs_error=comparison.max_abs_error,
                    max_rel_error=comparison.max_rel_error,
                    numpy_pass=comparison.passed,
                    torch_pass=torch_pass,
                    etl_ms=etl_ms,
                    rtol=e_rtol,
                    atol=e_atol,
                    tolerance=e_tolerance,
                )
            )
        except Exception as exc:  # record per-example failures in the report
            results.append(
                ExampleResult(
                    name=name, error=f"{type(exc).__name__}: {exc}"
                )
            )
    return ConformanceReport(
        results=results,
        use_torch=mode,
        torch_available=available,
        rtol=rtol,
        atol=atol,
        tolerance=tolerance,
        seed=seed,
        backend=backend,
        device=format_device(dev),
    )
