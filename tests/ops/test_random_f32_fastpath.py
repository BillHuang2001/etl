"""etl.random random_normal — the f32 Box–Muller fast path pins (numpy backend).

The numpy kernel takes the DOCUMENTED f32 fast path when the op's output
dtype is float32 (``etl/backends/numpy/kernels/random.py`` — mirrors the
stablehlo exporter's f32 semantics EXACTLY): the u1/u2 words are the SAME
words 2i/2i+1 as the f64 chain, each word converts to float32 FIRST (the
53-/32-bit fraction rounds to f32's 24-bit mantissa), the ``2^-53``/``2^-32``
scaling is an exact power-of-two f32 multiply, and the Box–Muller tail
(``sqrt(-2 log u1) * cos(2π u2)`` plus the mean/std combine) runs entirely
in float32. f64/f16 outputs and every other op keep the exact f64 math.

Pinned here (all three algorithms — the f32 fast path is algorithm-agnostic):

1. Determinism: the same key + operands produce BIT-IDENTICAL f32 values
   across separate evaluate calls.
2. Deviation budget: the f32 output vs the SAME graph run with ``dtype=
   float64`` and cast to f32 stays within the suite's documented budget
   ``rtol=1e-4``/``atol=1e-5`` — with a huge margin: measured over 2^16
   samples (mean 0 / std 1) the max deviation is ~3-5.5e-6
   (threefry2x32 largest), mean_abs ~1e-7 — f32 mantissa rounding plus f32
   transcendental ulps, as documented in the kernel docstring.
3. f64 outputs stay bit-identical across runs (the exporter's
   bit-exactness reference).
4. f16 outputs (which keep the exact f64 path, casting at the end) are
   finite, sane, and bit-identical across runs.

Backends: numpy interpreter only (this file asserts nothing about compiler
backends — the iree-llvm-cpu side of the fast path is pinned in
``tests/backends/``).
"""
import numpy as np
import pytest

import etl

from tests.ops.conftest import run_numpy

ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")
#: 2^16 samples: big enough for the deviation tail to show (measured max
#: deviation at this size: ~3-5.5e-6), small enough for the numpy backend
#: to stay fast.
SIZE = 1 << 16
#: Documented deviation budget for the f32 fast path vs the f64 chain
#: (10x+ headroom over the measured ~5.5e-6 max).
F32_BUDGET = dict(rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_f32_normal_bit_identical_across_evaluate_calls(alg):
    """Same key + same operands => bit-identical f32 normal draws across
    separate evaluate calls (the fast path is deterministic)."""

    @etl.defn
    def f(key):
        return etl.random.normal(key, (SIZE,), dtype=etl.float32)

    key = etl.random.key(7, alg)
    a = run_numpy(f, key)
    b = run_numpy(f, key)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


@pytest.mark.parametrize("alg", ALGORITHMS)
@pytest.mark.parametrize(
    "mean,std",
    [(0.0, 1.0), (1.5, 2.0)],
    ids=["mean0_std1", "mean1.5_std2"],
)
def test_f32_normal_within_f64_chain_budget(alg, mean, std):
    """The f32 fast path deviates from the exact f64 chain only by f32
    rounding (mantissa + transcendental ulps, measured max ~5.5e-6 at 2^16
    samples) — well inside the documented budget. The offset config also
    exercises the f32 mean/std rounding (mean/std convert to f32 first)."""

    @etl.defn
    def f32(key):
        return etl.random.normal(key, (SIZE,), mean=mean, std=std, dtype=etl.float32)

    @etl.defn
    def f64(key):
        return etl.random.normal(key, (SIZE,), mean=mean, std=std, dtype=etl.float64)

    key = etl.random.key(7, alg)
    fast = run_numpy(f32, key)
    chain = run_numpy(f64, key).astype(np.float32)
    assert np.allclose(fast, chain, **F32_BUDGET)
    # Margin guard: the real deviation is ~1e-6-scale — never near the
    # budget (regression tripwire if a future change drifts the fast path).
    assert np.abs(fast - chain).max() < 1e-4


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_f64_normal_bit_identical_across_runs(alg):
    """f64 outputs keep the exact f64 math — bit-identical across separate
    evaluate calls (the StableHLO export's bit-exactness reference)."""

    @etl.defn
    def f(key):
        return etl.random.normal(key, (SIZE,), dtype=etl.float64)

    key = etl.random.key(7, alg)
    a = run_numpy(f, key)
    b = run_numpy(f, key)
    assert a.dtype == np.float64
    assert np.array_equal(a, b)


@pytest.mark.parametrize("alg", ALGORITHMS)
def test_f16_normal_finite_sane_and_deterministic(alg):
    """f16 outputs keep the exact f64 path (casting at the end) — the
    values are finite, sane for a standard normal, and bit-identical across
    separate evaluate calls."""

    @etl.defn
    def f(key):
        return etl.random.normal(key, (SIZE,), dtype=etl.float16)

    key = etl.random.key(7, alg)
    a = run_numpy(f, key)
    b = run_numpy(f, key)
    assert a.dtype == np.float16
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    # |z| over 2^16 standard normals stays well under 10 (measured max
    # ~4.5); a broken f64/f16 path would blow this immediately.
    assert np.abs(a).max() < 10.0
