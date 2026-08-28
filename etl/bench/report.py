"""Report dataclasses and table printing for ``etl.bench``.

All fields are plain machine-readable values (``float``/``bool``/``str``/
``None``); use :meth:`to_dict` / :meth:`to_json` for structured output, or
``str(report)`` / :func:`print_report` for the human-readable table.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional

__all__ = [
    "ExampleResult",
    "ConformanceReport",
    "BenchmarkReport",
    "print_report",
]


@dataclass
class ExampleResult:
    """Per-example outcome.

    Conformance fields: ``max_abs_error``, ``max_rel_error``, ``numpy_pass``,
    ``torch_pass`` (``None`` when torch comparisons were skipped or the
    example has no torch reference), and the EFFECTIVE per-example
    tolerances used for the comparisons — ``rtol``/``atol``/``tolerance``
    (:func:`etl.bench.conformance` records the resolved effective values on
    every result; benchmark results leave them ``None``). Benchmark fields:
    run times in milliseconds per implementation plus speedup ratios vs the
    numpy reference (``numpy_ms / etl_ms``) and vs the torch reference
    (``torch_ms / etl_ms``). ``error`` records any execution failure
    (``"ExceptionType: message"``); on error all other fields stay ``None``.
    """

    name: str
    description: str = ""
    max_abs_error: Optional[float] = None
    max_rel_error: Optional[float] = None
    numpy_pass: Optional[bool] = None
    torch_pass: Optional[bool] = None
    etl_ms: Optional[float] = None
    numpy_ms: Optional[float] = None
    torch_ms: Optional[float] = None
    speedup_vs_numpy: Optional[float] = None
    speedup_vs_torch: Optional[float] = None
    error: Optional[str] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    tolerance: Optional[float] = None


@dataclass
class ConformanceReport:
    """Aggregate conformance result.

    ``use_torch`` is the resolved mode: ``"auto"`` | ``"enabled"`` |
    ``"disabled"``. ``overall_pass`` is True iff every example ran without
    error and passed every executed comparison (numpy always; torch only when
    it ran). ``backend`` is the etl backend the graphs ran on (default
    ``"numpy"``); ``device`` is the formatted device string (``"cpu"`` or
    e.g. ``"cuda:3"``).
    """

    results: List[ExampleResult]
    use_torch: str = "auto"
    torch_available: bool = False
    rtol: float = 1e-5
    atol: float = 1e-5
    tolerance: Optional[float] = None
    seed: int = 0
    backend: str = "numpy"
    device: str = "cpu"

    @property
    def overall_pass(self) -> bool:
        for result in self.results:
            if result.error is not None:
                return False
            if result.numpy_pass is not True:
                return False
            if result.torch_pass is not None and result.torch_pass is not True:
                return False
        return True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["overall_pass"] = self.overall_pass
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __str__(self) -> str:
        return _format_conformance(self)


@dataclass
class BenchmarkReport:
    """Aggregate benchmark result.

    Run times are best-of-``repeats`` milliseconds after ``warmup`` untimed
    runs (best-of-N; see :func:`etl.bench.benchmark`). ``use_torch`` is the
    resolved mode (``"auto"`` | ``"enabled"`` | ``"disabled"``). ``backend``
    is the etl backend the graphs ran on (default ``"numpy"``); ``device``
    is the formatted device string (``"cpu"`` or e.g. ``"cuda:3"``).
    """

    results: List[ExampleResult]
    repeats: int = 20
    warmup: int = 2
    use_torch: str = "auto"
    torch_available: bool = False
    seed: int = 0
    backend: str = "numpy"
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __str__(self) -> str:
        return _format_benchmark(self)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _torch_label(use_torch: str, available: bool) -> str:
    if use_torch == "disabled":
        return "disabled"
    if available:
        return use_torch
    return f"{use_torch} (torch unavailable)"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}"


def _ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _verdict(value: Optional[bool]) -> str:
    if value is None:
        return "-"
    return "PASS" if value else "FAIL"


def _table(header: List[str], rows: List[List[str]]) -> str:
    widths = [len(head) for head in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(head.ljust(widths[i]) for i, head in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        )
    return "\n".join(lines)


def _short_error(message: Optional[str]) -> str:
    if not message:
        return ""
    return message if len(message) <= 80 else message[:77] + "..."


def _backend_device_prefix(report) -> str:
    """Header prefix naming a non-default backend/device (default output is
    byte-identical to the numpy/cpu-only format: empty prefix)."""
    if report.backend == "numpy" and report.device == "cpu":
        return ""
    return f"backend={report.backend} device={report.device}, "


def _tol_cell(result: ExampleResult, report: ConformanceReport) -> str:
    """Compact per-example tolerance cell for the conformance table.

    ``"-"`` when the example used the global defaults (its effective
    tolerances equal the report's); ``"abs 0.0001"`` for a max-abs-error
    override; ``"rtol 0.001 atol 0.001"`` for rtol/atol overrides.
    """
    if result.error is not None:
        return "-"  # errored results carry no tolerances
    if (
        result.rtol == report.rtol
        and result.atol == report.atol
        and result.tolerance == report.tolerance
    ):
        return "-"
    if result.tolerance is not None:
        return f"abs {result.tolerance:g}"
    return f"rtol {result.rtol:g} atol {result.atol:g}"


def _format_conformance(report: ConformanceReport) -> str:
    header = [
        "example",
        "max_abs_error",
        "max_rel_error",
        "numpy",
        "torch",
        "tol",
        "etl_ms",
        "error",
    ]
    rows = []
    for result in report.results:
        rows.append(
            [
                result.name,
                _num(result.max_abs_error),
                _num(result.max_rel_error),
                _verdict(result.numpy_pass),
                _verdict(result.torch_pass),
                _tol_cell(result, report),
                _ms(result.etl_ms),
                _short_error(result.error),
            ]
        )
    tolerance = (
        f" tolerance={report.tolerance:g}" if report.tolerance is not None else ""
    )
    lines = [
        "etl.bench conformance — "
        f"{_backend_device_prefix(report)}"
        f"torch={_torch_label(report.use_torch, report.torch_available)}, "
        f"rtol={report.rtol:g} atol={report.atol:g}{tolerance}, seed={report.seed}",
        _table(header, rows),
    ]
    checked = [r for r in report.results if r.error is None]
    passed = sum(
        1
        for r in checked
        if r.numpy_pass is True
        and (r.torch_pass is None or r.torch_pass is True)
    )
    lines.append(
        f"overall: {'PASS' if report.overall_pass else 'FAIL'} "
        f"({passed}/{len(checked)} checked)"
    )
    return "\n".join(lines)


def _format_benchmark(report: BenchmarkReport) -> str:
    header = [
        "example",
        "etl_ms",
        "numpy_ms",
        "torch_ms",
        "speedup_vs_numpy",
        "speedup_vs_torch",
        "error",
    ]
    rows = []
    for result in report.results:
        rows.append(
            [
                result.name,
                _ms(result.etl_ms),
                _ms(result.numpy_ms),
                _ms(result.torch_ms),
                _num(result.speedup_vs_numpy),
                _num(result.speedup_vs_torch),
                _short_error(result.error),
            ]
        )
    lines = [
        "etl.bench benchmark — "
        f"{_backend_device_prefix(report)}"
        f"best-of-{report.repeats} ms after {report.warmup} warmup run(s), "
        f"torch={_torch_label(report.use_torch, report.torch_available)}, "
        f"seed={report.seed}",
        _table(header, rows),
    ]
    return "\n".join(lines)


def print_report(report) -> None:
    """Print the human-readable table of a report.

    Args:
        report: a :class:`ConformanceReport` or :class:`BenchmarkReport`.

    Raises:
        TypeError: any other object.
    """
    if isinstance(report, (ConformanceReport, BenchmarkReport)):
        print(str(report))
    else:
        raise TypeError(
            f"print_report expects a ConformanceReport or BenchmarkReport, "
            f"got {type(report).__name__}"
        )
