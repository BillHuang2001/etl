"""etl.bench benchmark contract.

``benchmark(...)`` times each selected example through the explicit etl
pipeline vs its pure-numpy reference (and optionally torch): ``warmup``
untimed runs followed by best-of-``repeats`` wall-clock milliseconds per
implementation, with speedup ratios ``reference_ms / etl_ms``. Per-example
failures are recorded in the report (``error`` field), never swallowed.
``BenchmarkReport`` exposes plain values plus ``to_dict()``/``to_json()``
round-tripping.
"""
from __future__ import annotations

import json

from etl.bench import BenchmarkReport, benchmark


def test_benchmark_selected_examples_numpy_only():
    report = benchmark(
        examples=["matmul", "softmax"], use_torch=False, repeats=3, warmup=1
    )
    assert isinstance(report, BenchmarkReport)
    assert report.repeats == 3
    assert report.warmup == 1
    assert report.use_torch == "disabled"
    # Results keep the requested order.
    assert [result.name for result in report.results] == ["matmul", "softmax"]
    for result in report.results:
        assert result.error is None
        assert result.etl_ms > 0
        assert result.numpy_ms > 0
        assert result.speedup_vs_numpy > 0
        assert result.torch_ms is None
        assert result.speedup_vs_torch is None


def test_benchmark_report_serialization_roundtrip():
    report = benchmark(
        examples=["matmul"], use_torch=False, repeats=3, warmup=1
    )
    data = report.to_dict()
    assert data["repeats"] == 3
    assert data["warmup"] == 1
    assert data["use_torch"] == "disabled"
    assert json.loads(report.to_json()) == data


def test_benchmark_report_str_smoke():
    report = benchmark(
        examples=["matmul"], use_torch=False, repeats=3, warmup=1
    )
    text = str(report)
    assert "etl_ms" in text
    assert "matmul" in text
