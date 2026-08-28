"""etl.bench conformance contract.

``conformance(...)`` stages every selected example through the explicit etl
pipeline (``etl.build`` + ``etl.run`` on the default numpy backend) and
compares against the pure-numpy reference (always) and the torch reference
(when enabled). Per-example failures are RECORDED in the report (``error``
field, ``overall_pass`` False) — never swallowed. torch optionality is
binding: ``use_torch=True`` without torch raises a clear ``ImportError``
mentioning ``pip install etl[bench]``; ``use_torch=None`` (auto) silently
skips torch comparisons when torch is absent (``torch_pass`` None,
``use_torch`` "disabled").
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from etl.bench import (
    ConformanceReport,
    ExampleResult,
    conformance,
    print_report,
)

TORCH_ABSENT = importlib.util.find_spec("torch") is None
requires_torch_absent = pytest.mark.skipif(
    not TORCH_ABSENT, reason="requires torch absent"
)

# Documented ExampleResult fields (serialized inside report.to_dict()).
EXAMPLE_RESULT_FIELDS = {
    "name",
    "description",
    "max_abs_error",
    "max_rel_error",
    "numpy_pass",
    "torch_pass",
    "etl_ms",
    "numpy_ms",
    "torch_ms",
    "speedup_vs_numpy",
    "speedup_vs_torch",
    "error",
    # effective per-example tolerances (added with the per-example override
    # contract; None on errored/benchmark results)
    "rtol",
    "atol",
    "tolerance",
}


def test_conformance_numpy_only_all_examples():
    report = conformance(use_torch=False)
    assert isinstance(report, ConformanceReport)
    assert len(report.results) == 26
    assert report.use_torch == "disabled"
    assert report.overall_pass is True
    for result in report.results:
        assert isinstance(result, ExampleResult)
        assert result.error is None
        assert result.numpy_pass is True
        assert result.torch_pass is None
        # Far below float32 noise across the registry: micro/vectorize are
        # exact (shared kernels), grad ~1e-5, the large category's conv2d_large
        # measures ~1.1e-4 (fp32 accumulation order; covered by its atol
        # override).
        assert result.max_abs_error < 1e-3


def test_conformance_to_dict_to_json_roundtrip():
    report = conformance(use_torch=False)
    data = report.to_dict()
    assert "overall_pass" in data
    assert data["overall_pass"] is True
    assert json.loads(report.to_json()) == data
    # ExampleResult dicts inside the report carry the documented fields.
    assert len(data["results"]) == 26
    for result_dict in data["results"]:
        assert set(result_dict) == EXAMPLE_RESULT_FIELDS


def test_str_report_and_print_report(capsys):
    report = conformance(use_torch=False)
    text = str(report)
    assert "matmul" in text
    assert "attention" in text
    assert "overall" in text
    print_report(report)
    captured = capsys.readouterr()
    assert "matmul" in captured.out
    assert "overall" in captured.out
    with pytest.raises(TypeError):
        print_report(object())


def test_conformance_single_name_and_tolerance():
    report = conformance("matmul", use_torch=False)
    assert len(report.results) == 1
    assert report.results[0].name == "matmul"
    assert report.overall_pass is True

    report = conformance(["matmul"], tolerance=1e-3, use_torch=False)
    assert report.tolerance == 1e-3
    assert report.overall_pass is True
    assert report.results[0].numpy_pass is True
    assert report.results[0].max_abs_error <= 1e-3


def test_conformance_per_example_tolerance_resolution():
    # mlp's per-example tolerance=1e-4 override is recorded on its result
    # (effective value = the example's when set, else the global argument).
    report = conformance(["mlp"], use_torch=False)
    assert report.tolerance is None  # global default
    (result,) = report.results
    assert result.tolerance == 1e-4
    assert result.rtol == 1e-5  # no rtol/atol override on mlp
    assert result.atol == 1e-5
    assert result.numpy_pass is True

    # matmul has no overrides: effective values fall back to the globals
    # (tolerance None — the allclose-style rule is used).
    report = conformance(["matmul"], use_torch=False)
    (result,) = report.results
    assert result.tolerance is None
    assert result.rtol == 1e-5
    assert result.atol == 1e-5
    assert result.numpy_pass is True

    # grad examples carry rtol=atol=1e-3 overrides.
    report = conformance(["grad_mlp"], use_torch=False)
    (result,) = report.results
    assert result.rtol == 1e-3
    assert result.atol == 1e-3
    assert result.tolerance is None


def test_conformance_grad_example_smoke():
    # grad examples stage transform-produced graphs (etl.grad TransformCallables)
    # through the explicit lower/compile/load pipeline — exercise that path.
    report = conformance(["grad_mlp"], use_torch=False)
    assert report.overall_pass is True
    (result,) = report.results
    assert result.error is None
    assert result.numpy_pass is True
    assert result.max_abs_error < 1e-3


@requires_torch_absent
def test_conformance_use_torch_true_without_torch_raises_import_error():
    with pytest.raises(ImportError) as excinfo:
        conformance(use_torch=True)
    assert type(excinfo.value) is ImportError  # never a raw ModuleNotFoundError
    assert "pip install etl[bench]" in str(excinfo.value)


@requires_torch_absent
def test_conformance_auto_without_torch_skips_torch():
    report = conformance(use_torch=None)
    assert report.use_torch == "disabled"
    assert report.torch_available is False
    assert report.overall_pass is True
    for result in report.results:
        assert result.error is None
        assert result.numpy_pass is True
        assert result.torch_pass is None


def test_conformance_with_torch_enabled_all_examples():
    torch = pytest.importorskip("torch")  # noqa: F841  (torch must be present)
    report = conformance(use_torch=True)
    assert report.use_torch == "enabled"
    assert report.torch_available is True
    assert len(report.results) == 26
    assert report.overall_pass is True
    for result in report.results:
        # torch references must pass too; results are recorded, never
        # swallowed.
        assert result.error is None
        assert result.numpy_pass is True
        assert result.torch_pass is True
