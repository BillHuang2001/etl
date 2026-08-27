"""Benchmarking: wall-clock timing of etl graphs vs numpy (and optional
torch) references.

Reported run times are BEST-of-``repeats`` milliseconds after ``warmup``
untimed runs (best-of-N) — the fastest observed run, which is the least
noise-prone summary for small CPU workloads. Speedup ratios are
``reference_ms / etl_ms`` (``> 1`` means etl is faster).

The etl graph is staged on the default numpy backend — or a chosen
``backend``/``device`` with ``backend_options`` passthrough.
"""
from __future__ import annotations

from typing import List

import etl
from ._torch import require_torch
from ._util import (
    best_time_ms,
    format_device,
    resolve_backend,
    resolve_backend_options,
    resolve_device,
    resolve_examples,
    resolve_torch_device,
    resolve_torch_mode,
)
from .examples import get_example
from .report import BenchmarkReport, ExampleResult

__all__ = ["benchmark"]


def benchmark(examples=None, *, use_torch=None, repeats=20, warmup=2,
              seed=0, backend="numpy", device=None,
              **backend_options) -> BenchmarkReport:
    """Benchmark the selected examples.

    For each example: generate inputs, stage the etl graph once through the
    explicit pipeline (``etl.build``), then time — best-of-``repeats`` runs
    after ``warmup`` untimed runs — the etl graph on the chosen backend (by
    default the numpy backend), the pure-numpy reference, and (when torch is
    available / requested) the torch reference.

    Args:
        examples: ``None`` (all registered examples), a single name, or an
            iterable of names.
        use_torch: ``None`` = auto (torch runs iff ``import torch``
            succeeds); ``True`` = require torch — raises a clear
            ``ImportError`` mentioning ``pip install etl[bench]`` when torch
            is unavailable; ``False`` = numpy-only.
        repeats: number of timed runs per implementation (best-of-N
            reported); must be >= 1.
        warmup: number of untimed runs before timing; must be >= 0.
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

    Returns:
        :class:`BenchmarkReport` with one :class:`ExampleResult` per example
        (``etl_ms``, ``numpy_ms``, optional ``torch_ms``, speedup ratios, or
        an ``error`` string).

    Raises:
        ValueError: invalid ``repeats``/``warmup``; unsupported device kind
            or malformed device string.
        TypeError: invalid ``backend``/``device`` argument types.
        BackendError: unknown backend name or missing adapter dependency
            (raised up front, before any example runs).
        ImportError: ``use_torch=True`` without torch (clear hint, never a
            raw ``ModuleNotFoundError``).
        UnknownExampleError: unknown example name (lists available names).
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")
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
        try:
            inputs = example.generate_inputs(seed)
            executable = etl.build(
                example.graph, *example.specs,
                backend=backend, device=dev, **opts
            )
            etl_ms = best_time_ms(
                lambda: etl.run(executable, *inputs), warmup, repeats
            )
            numpy_ms = best_time_ms(
                lambda: example.numpy_ref(inputs), warmup, repeats
            )
            torch_ms = None
            if enabled:
                if example.torch_ref is None:
                    raise ValueError(
                        f"example {name!r} has no torch reference"
                    )
                torch_ms = best_time_ms(
                    lambda: example.torch_ref(inputs, device=torch_device),
                    warmup, repeats,
                )
            speedup_vs_numpy = (
                numpy_ms / etl_ms if numpy_ms is not None and etl_ms > 0 else None
            )
            speedup_vs_torch = (
                torch_ms / etl_ms if torch_ms is not None and etl_ms > 0 else None
            )
            results.append(
                ExampleResult(
                    name=name,
                    description=example.description,
                    etl_ms=etl_ms,
                    numpy_ms=numpy_ms,
                    torch_ms=torch_ms,
                    speedup_vs_numpy=speedup_vs_numpy,
                    speedup_vs_torch=speedup_vs_torch,
                )
            )
        except Exception as exc:  # record per-example failures in the report
            results.append(
                ExampleResult(
                    name=name, error=f"{type(exc).__name__}: {exc}"
                )
            )
    return BenchmarkReport(
        results=results,
        repeats=repeats,
        warmup=warmup,
        use_torch=mode,
        torch_available=available,
        seed=seed,
        backend=backend,
        device=format_device(dev),
    )
