"""iree-llvm-cpu parity for the f32 random_normal fast path — all three
algorithms on the ADAPTER-DEFAULT lowering (round-1 cross-backend finding).

Round-1 added the numpy-side f32 Box–Muller fast path mirroring the
stablehlo exporter's f32 semantics EXACTLY (``random.py`` /
``random_export.py``). With BOTH sides on the fast path, an iree-llvm-cpu
f32 ``random_normal`` run vs ``etl.evaluate`` (numpy reference) is NOT
bit-exact — the residual is pure libm ``log``/``cos`` drift between numpy
and the iree CPU runtime, measured max_abs ~5e-7 at 2^16 samples (mean 0 /
std 1, all three algorithms) — and uniform/randint/permutation stay EXACT.

This file extends ``test_iree_random_algorithms_parity.py``:
``test_inline_bit_exact`` (uniform/randint/permutation EXACT) and
``test_normal_f32_tolerance_and_determinism`` cover threefry2x32 and
philox4x32_10 only (the inline-forced path there). Here every algorithm —
INCLUDING splitmix64, which that file does not cover at all and which
``test_iree_emitters_parity.py`` pins at one seed only — runs on the
ADAPTER-DEFAULT lowering (no ``rng_bit_generator`` override: splitmix64 is
always the inline i64 expansion, threefry uses the native THREE_FRY
capability default) across several keys, with a measured-margin guard
(max_abs < 1e-5, 20x headroom over the ~5e-7 measured drift) so a future
fast-path drift trips loudly instead of hiding inside the budget.
"""
import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (same conventions as test_iree_random_algorithms_parity.py)
# ---------------------------------------------------------------------------
ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")
KEY_SPECS = {
    "splitmix64": etl.TensorSpec((), etl.int64),
    "threefry2x32": etl.TensorSpec((2,), etl.int32),
    "philox4x32_10": etl.TensorSpec((4,), etl.int32),
}
KEYS = {
    "splitmix64": (
        np.array(0, dtype=np.int64),
        np.array(-1, dtype=np.int64),
        etl.random.key(42).numpy(),
    ),
    "threefry2x32": (
        np.array([0, 0], dtype=np.int32),
        np.array([-1, -1], dtype=np.int32),
        etl.random.key(42, algorithm="threefry2x32").numpy(),
    ),
    "philox4x32_10": (
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([-1, -1, -1, -1], dtype=np.int32),
        etl.random.key(42, algorithm="philox4x32_10").numpy(),
    ),
}
#: Documented f32 fast-path budget (same as the parity file's
#: RANDOM_NORMAL_TOL): measured max drift ~5e-7 (pure libm log/cos drift —
#: both sides run the SAME f32 fast-path semantics now).
RANDOM_NORMAL_TOL = dict(rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# helpers (_np/_assert_exact/_assert_close mirror the parity file)
# ---------------------------------------------------------------------------
def _np(v):
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = _np(got), _np(want)
    assert g.shape == w.shape
    assert np.array_equal(g, w), f"{g} != {w}"


def _assert_close(got, want, rtol=1e-5, atol=1e-5):
    g, w = _np(got), _np(want)
    assert g.shape == w.shape
    assert np.allclose(g, w, rtol=rtol, atol=atol), f"{g} != {w}"


# ---------------------------------------------------------------------------
# sampling graphs (each takes the algorithm's key as its only input)
# ---------------------------------------------------------------------------
@etl.defn
def _uniform(k):
    return etl.random.uniform(k, (4, 5), dtype=etl.float32)


@etl.defn
def _randint(k):
    return etl.random.randint(k, (8,), low=0, high=100)


@etl.defn
def _perm(k):
    return etl.random.permutation(k, 5)


@etl.defn
def _normal_f32(k):
    return etl.random.normal(k, (4, 5), dtype=etl.float32)


# ---------------------------------------------------------------------------
# 1. uniform/randint/permutation — EXACT vs numpy on the adapter-default
#    lowering (splitmix64 inline i64 expansion; threefry native THREE_FRY;
#    philox inline i32 expansion), every key through one executable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
@pytest.mark.parametrize("case_id,fn", [
    ("uniform", _uniform),
    ("randint", _randint),
    ("permutation", _perm),
], ids=["uniform", "randint", "permutation"])
def test_value_ops_bit_exact_adapter_default(algorithm, case_id, fn):
    """uniform/randint/permutation stay BIT-EXACT vs ``etl.evaluate`` on
    iree-llvm-cpu — the fast path only touches random_normal."""
    exe = etl.build(fn, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(fn, key)
        got = etl.run(exe, key)
        _assert_exact(got, want)
        _assert_exact(etl.run(exe, key), got)  # same exe, second call


# ---------------------------------------------------------------------------
# 2. f32 normal — within the documented budget vs numpy (NOT exact: both
#    sides run the f32 fast path; the residual is libm log/cos drift,
#    measured max_abs ~5e-7) AND bit-identical across two separate runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_normal_f32_tolerance_and_determinism(algorithm):
    exe = etl.build(_normal_f32, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(_normal_f32, key)
        g1 = etl.run(exe, key)
        g2 = etl.run(exe, key)
        assert _np(g1).dtype == np.float32
        _assert_close(g1, want, **RANDOM_NORMAL_TOL)
        _assert_exact(g1, g2)  # the hard same-key determinism contract
        # Measured-margin guard: the real iree-vs-numpy drift is ~5e-7
        # (libm log/cos), so a max deviation at the 1e-5 scale means a
        # fast-path semantic drift, not libm noise — trip loudly.
        assert np.abs(_np(g1) - _np(want)).max() < 1e-5
