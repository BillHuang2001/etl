"""iree-llvm-cpu round-trip parity for the v1 eigh/diag StableHLO emitters.

Both are multi-op compositions — no mnemonic StableHLO op exists (StableHLO
1.0 removed ``eigh``/``qr``/``svd``, and there is no diag op), see
``etl/backends/stablehlo/ops.py``:

* ``eigh`` — a while-loop cyclic-Jacobi symmetric eigensolver: ONE
  ``stablehlo.while`` loop carrying ``(a, v, p, q, k)`` applies the ADAPTIVE
  sweep schedule (dim-based for fp32 — n ≤ 5 → 3, n = 6 → 4, 7 ≤ n ≤ 15 → 5,
  16 ≤ n ≤ 30 → 6, 31 ≤ n ≤ 50 → 7; f64 always 8 — of the (p, q) rotation
  pairs, with dynamic (p, q) indexing via gathers (never ``stablehlo.slice``
  — its start indices must be static) and constants/iota hoisted outside the
  loop. The emitted text is O(1) in n — the unrolled predecessor's
  O(n²·sweeps) text is gone, so dim ≥ 25 compiles — while the runtime stays
  O(n⁴) (the rotations still execute sequentially). A stable pair-sort then
  orders the final diagonal ASCENDING + a gather reorders the V columns.
* ``diag`` — rank-1 → iota-EQ mask + select (the diagonal matrix); rank-2 →
  flatten reshape + constant-index single-axis gather (the main diagonal).

Parity vs numpy: ``diag`` is EXACT (``np.array_equal``). ``eigh`` is
fp32-tolerance, NOT bit-exact — LAPACK (numpy) vs the Jacobi composition
converge to the fp32 floor, the eigenvectors of (near-)degenerate eigenspaces
are basis-dependent, so tests check the mathematical contract — ascending
``w``, reconstruction ``A ≈ V diag(w) Vᵀ``, column orthogonality
``VᵀV ≈ I`` — and ``w`` vs numpy within fp32 tolerance, never elementwise v.

Measured max diffs (seed 7, iree-llvm-cpu, fp32): 3x3 rec 1.2e-6 / orth
1.8e-7 / w-vs-numpy 9.5e-7; 10x10 (CMAES size) rec 1.7e-5 / orth 6e-7 /
w-vs-numpy 1.5e-5; batched (2, 3, 3) rec 9.5e-7; f64 3x3 rec + w-vs-numpy
2.7e-15; int32 3x3 (upcast f64) rec 4.9e-15; dim 25 rec 1.7e-3 / orth
8.7e-6 — the fp32 tolerance band across sizes is rec 1.2e-6..1.7e-3,
orth 1.8e-7..8.7e-6.

Build+run times: the while composition is O(1) emitted text, so the build is
~1.6-2.3 s at ANY dim (3x3 f32 ~1.94 s build / 0.005 s run; 10x10 ~1.6 s /
0.035 s; dim 25 ~2.26 s / 0.37 s; dim 50 ~1.79 s / 3.6 s — the run grows
O(n⁴) with the rotation count). CPU only (llvm-cpu — well-shaped carries,
no iree-cuda while fragility, and no slices; keep these on llvm-cpu).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (fixed seed — same convention as test_iree_emitters_parity.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)


def _sym(n, scale=0.0, dtype=np.float32):
    """A symmetric (n, n) matrix with well-separated eigenvalues: X @ X.T
    symmetrized, plus ``scale`` on the diagonal for conditioning."""
    x = _RNG.standard_normal((n, n))
    m = x @ x.T
    m = (m + m.T) / 2
    return (m + np.eye(n) * scale).astype(dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _check_eigh_contract(A, w, v, rtol=1e-3, atol=1e-3):
    """The eigh mathematical contract: ascending w, A ≈ V diag(w) Vᵀ,
    VᵀV ≈ I (batch-aware einsum — never elementwise v)."""
    w, v = np.asarray(w), np.asarray(v)
    batch, n = A.shape[:-2], A.shape[-1]
    assert w.shape == batch + (n,)
    assert v.shape == A.shape
    # StableHLO sort output is monotone in the comparator order: diff >= 0.
    assert np.all(np.diff(w, axis=-1) >= 0), f"w not ascending: {w}"
    rec = np.einsum("...ij,...j,...kj->...ik", v, w, v)
    orth = np.einsum("...ij,...ik->...jk", v, v)
    eye = np.broadcast_to(np.eye(n, dtype=v.dtype), orth.shape)
    assert np.allclose(rec, A, rtol=rtol, atol=atol), (
        f"A ≈ V diag(w) Vᵀ failed: max abs diff "
        f"{np.max(np.abs(rec - A))}"
    )
    assert np.allclose(orth, eye, rtol=rtol, atol=atol), (
        f"VᵀV ≈ I failed: max abs diff {np.max(np.abs(orth - eye))}"
    )


def _run_eigh(fn, A, spec):
    """Build on iree-llvm-cpu, run, and return the (w, v) ndarrays."""
    want = etl.evaluate(fn, A)
    exe = etl.build(fn, spec, backend="iree")
    got = etl.run(exe, A)
    return _np(got[0]), _np(got[1]), _np(want[0]), _np(want[1])


# ---------------------------------------------------------------------------
# 1. diag — EXACT vs numpy (both directions)
# ---------------------------------------------------------------------------


def _diag(x):
    return etl.diag(x)


DIAG_CASES = [
    ("diag_rank1", _diag, etl.TensorSpec((5,), etl.float32),
     _RNG.standard_normal(5).astype(np.float32)),
    ("diag_rank2_square", _diag, etl.TensorSpec((3, 3), etl.float32),
     _RNG.standard_normal((3, 3)).astype(np.float32)),
    ("diag_rank2_rect", _diag, etl.TensorSpec((4, 6), etl.float32),
     _RNG.standard_normal((4, 6)).astype(np.float32)),
]


@pytest.mark.parametrize(
    "fn,spec,arg",
    [(fn, spec, arg) for _, fn, spec, arg in DIAG_CASES],
    ids=[case_id for case_id, _, _, _ in DIAG_CASES],
)
def test_diag_iree_exact(fn, spec, arg):
    want = etl.evaluate(fn, arg)
    got = etl.run(etl.build(fn, spec, backend="iree"), arg)
    g, w = _np(got), _np(want)
    assert g.shape == w.shape
    assert np.array_equal(g, w), f"{g} != {w}"


# ---------------------------------------------------------------------------
# 2. eigh — fp32-tolerance contract (ascending w, reconstruction,
#    orthogonality) + w vs numpy
# ---------------------------------------------------------------------------


def _eigh(x):
    w, v = etl.eigh(x)
    return w, v


def test_eigh_3x3_f32_contract():
    A = _sym(3)
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((3, 3), etl.float32))
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw, ww, rtol=1e-3, atol=1e-3), f"w vs numpy: {gw} != {ww}"


def test_eigh_10x10_f32_contract():
    # CMAES population-covariance size; well-conditioned (diagonal +10).
    A = _sym(10, scale=10.0)
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((10, 10), etl.float32))
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw, ww, rtol=1e-3, atol=1e-3), (
        f"w vs numpy: max abs diff {np.max(np.abs(gw - ww))}"
    )


def test_eigh_batched_2x3x3_f32_contract():
    A = np.stack([_sym(3), _sym(3)])
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((2, 3, 3), etl.float32))
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw, ww, rtol=1e-3, atol=1e-3), (
        f"w vs numpy: max abs diff {np.max(np.abs(gw - ww))}"
    )


def test_eigh_f64_3x3_contract():
    # f64 passes through — the Jacobi composition is near-exact (LAPACK-level
    # agreement measured at 2.7e-15).
    A = _sym(3, dtype=np.float64)
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((3, 3), etl.float64))
    _check_eigh_contract(A, gw, gv, rtol=1e-12, atol=1e-12)
    assert np.allclose(gw, ww, rtol=1e-12, atol=1e-12), (
        f"w vs numpy: max abs diff {np.max(np.abs(gw - ww))}"
    )


def test_eigh_int32_upcast_f64_contract():
    # int → float64 upcast (numpy linalg rule): the composition runs in f64,
    # so the integer spectrum is recovered exactly.
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.int32)
    want = etl.evaluate(_eigh, A)
    exe = etl.build(_eigh, etl.TensorSpec((3, 3), etl.int32), backend="iree")
    got = etl.run(exe, A)
    gw, gv = _np(got[0]), _np(got[1])
    assert gw.dtype == np.float64 and gv.dtype == np.float64
    _check_eigh_contract(A.astype(np.float64), gw, gv, rtol=1e-12, atol=1e-12)
    assert np.allclose(gw, _np(want[0]), rtol=1e-12, atol=1e-12), (
        f"w vs numpy: {gw} != {_np(want[0])}"
    )


def test_eigh_dim25_f32_contract():
    # while-loop emission sanity at a larger dim: the emitted text is O(1)
    # in n, so the build stays ~2.3 s (single build) and only the run grows
    # O(n^4). fp32 tolerance band measured at dim 25: rec 1.7e-3 / orth
    # 8.7e-6 — still inside the 1e-3 contract.
    A = _sym(25, scale=10.0)
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((25, 25), etl.float32))
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw, ww, rtol=1e-3, atol=1e-3), (
        f"w vs numpy: max abs diff {np.max(np.abs(gw - ww))}"
    )
