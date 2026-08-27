"""etl.bench repeated torch-mode calls regression pin.

In a torch-present environment, the SECOND torch-mode call in the same
process used to crash with ``UnboundLocalError``: ``require_torch()``
(``etl.bench._torch``, used by ``conformance(use_torch=True)`` and
``benchmark(use_torch=True)`` via ``resolve_torch_mode``) only bound the
local ``torch`` name on its first (probing) invocation; on the second call
the cached probe skipped the binding and ``return torch`` raised
``UnboundLocalError``. The fix caches the imported module and returns it on
subsequent calls.

This environment has NO torch installed, so every test is gated on torch
presence via ``pytest.importorskip("torch")`` (first line of each function)
and skips cleanly here — but the assertions are the executable spec for
torch-present runs: repeated torch-mode calls must all succeed.
"""
from __future__ import annotations

import pytest

from etl.bench import benchmark, conformance


def test_conformance_torch_mode_twice_in_row():
    torch = pytest.importorskip("torch")  # noqa: F841  (torch must be present)
    first = conformance(["matmul"], use_torch=True)
    second = conformance(["matmul"], use_torch=True)
    for report in (first, second):
        assert report.use_torch == "enabled"
        assert report.torch_available is True
        assert report.overall_pass is True
        (result,) = report.results
        assert result.name == "matmul"
        assert result.error is None
        assert result.numpy_pass is True
        assert result.torch_pass is True


def test_benchmark_torch_mode_after_conformance():
    torch = pytest.importorskip("torch")  # noqa: F841  (torch must be present)
    # A torch-mode conformance run first, so this process has already gone
    # through require_torch()'s probing path.
    conformance(["matmul"], use_torch=True)
    report = benchmark(["matmul"], use_torch=True, repeats=3, warmup=1)
    assert report.use_torch == "enabled"
    assert report.torch_available is True
    (result,) = report.results
    assert result.name == "matmul"
    assert result.error is None
    assert result.torch_ms is not None and result.torch_ms > 0
    assert result.speedup_vs_torch is not None and result.speedup_vs_torch > 0


def test_require_torch_twice_returns_same_module():
    torch = pytest.importorskip("torch")  # noqa: F841  (torch must be present)
    from etl.bench import _torch  # exact regression site (private helper)

    first = _torch.require_torch()
    second = _torch.require_torch()
    assert first is torch
    assert second is torch
    assert first is second
