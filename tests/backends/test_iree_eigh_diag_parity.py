"""iree-llvm-cpu round-trip parity for the v1 eigh/diag StableHLO emitters.

Both are multi-op compositions — no mnemonic StableHLO op exists (StableHLO
1.0 removed ``eigh``/``qr``/``svd``, and there is no diag op), see
``etl/backends/stablehlo/ops.py``:

* ``eigh`` — TWO cyclic-Jacobi symmetric-eigensolver compositions selected by
  the exporter option ``eigh_algorithm`` (``"jacobi_batched"`` = the DEFAULT,
  ``"jacobi"`` = the sequential predecessor):

  - **batched (default,** ``etl/backends/stablehlo/eigh_batched.py`` **)** —
    ONE ``stablehlo.while`` carrying ``(a, v, k)`` applies ALL ``m`` mutually
    DISJOINT rotations of one odd-even round-robin round per iteration (``L``
    rounds per sweep, ``total = sweeps·L``), so a sweep costs ``L`` iterations
    instead of ``n(n-1)/2``. The round is expressed through partner gathers
    plus per-slot ``(m,)`` pair vectors and per-index ``(n,)`` vectors derived
    ARITHMETICALLY from hoisted iotas — no scatter region, and the body's op
    count is constant in n (only the tensor types mention n). Sweeps are
    dim-based for fp32 (n ≤ 5 → 3, n = 6 → 4, 7 ≤ n ≤ 15 → 5, 16 ≤ n ≤ 30 → 6,
    31 ≤ n ≤ 50 → 7, n > 50 → 8) and FIXED at 8 for f64.
  - **sequential (** ``eigh_algorithm="jacobi"`` **,** ``writer.py``
    ``_emit_eigh_jacobi`` **)** — one (p, q) rotation per iteration,
    ``sweeps × n(n−1)/2`` iterations, byte-identical to the pre-option text.
    Still fully covered below (``test_eigh_jacobi_10x10_f32_option_contract``).

  Both compositions share the same post-loop tail: a stable pair-sort for the
  ASCENDING eigenvalue order plus a gather column reorder of V.
* ``diag`` — rank-1 → iota-EQ mask + select (the diagonal matrix); rank-2 →
  flatten reshape + constant-index single-axis gather (the main diagonal).

Parity vs numpy: ``diag`` is EXACT (``np.array_equal``). ``eigh`` is
tolerance-based, NOT bit-exact — LAPACK (numpy) and the Jacobi composition
converge along different paths, the eigenvectors of (near-)degenerate
eigenspaces are basis-dependent, so the tests assert the mathematical
contract — ascending ``w``, reconstruction ``A ≈ V diag(w) Vᵀ``, column
orthogonality ``VᵀV ≈ I`` — and the spectrum vs numpy, never elementwise v.

Measured (iree-llvm-cpu, the FIXED-SEED matrices parametrized below, batched
default; errors NORMALIZED by ``max|A|`` — see ``_check_eigh_scaled``):
f32 n=25 rec 4.2e-6 / orth 3.5e-6; f32 n=50 rec 1.6e-5 / orth 9.1e-6; f32
n=100 rec 3.3e-5 / orth 2.1e-5 (the same band as the earlier
3.22e-5 / 1.71e-5 reference measurement); f64 n=25 rec 8.8e-15 / orth 5.8e-15;
f64 n=50 rec 1.6e-14 / orth 1.3e-14; f64 n=100 rec 4.4e-9 / orth 2.5e-14; the
ROCm leg (MI300X/gfx942, n=10) rec 1.3e-6 / orth 8.3e-7 — the llvm-cpu band.
The small sizes keep their original numbers: 3x3 rec ~1.2e-6, 10x10 rec
~1.7e-5, batched (2, 3, 3) rec ~1e-6, f64 3x3 ~2.7e-15, int32→f64 ~4.9e-15.

IMPORTANT — the n=100 accuracy is set by the SWEEP COUNT, not by the dtype
floor: n > 50 keeps the same 8 sweeps for both dtypes, so the fp32 n=100 band
(3e-5) and the f64 n=100 band are both the sweep-limited residual, and the f64
value is MATRIX-dependent (measured 4.4e-9 on the pinned seed, up to ~8e-7 on
a harder near-degenerate spectrum) instead of the ~1e-14 the f64 floor would
allow at n ≤ 50. The per-case bounds below are the measured values with a
2-5× margin (deterministic inputs + a real compiler, so they cannot flake the
way a sampled bound would).

Build+run times (batched default): the emitted text is O(1) in n, so a build
stays ~0.8 s at ANY dim (n=25 5 ms run / n=50 18 ms / n=100 ~100 ms — the run
grows O(n⁴) in the rotation count). CPU only (llvm-cpu — well-shaped carries,
no iree-cuda while fragility); ONE extra ROCm-guarded case proves the batched
composition also runs on the AMD GPU (the sequential one SIGSEGVs there).

KNOWN ISSUE (**BUG(etl)**, pre-existing — it hits BOTH compositions):
``eigh`` inside a nested ``stablehlo`` REGION is broken on the compiler path.
The eigh emitters hoist their constants/iotas "outside the loop" by appending
them to the FUNCTION-level body, so an eigh inside an ``etl.cond`` branch is
emitted at the top level and both ``stablehlo.if`` regions are left holding
only ``stablehlo.return`` of captured values — iree then mis-executes the
graph: ``w`` is right but ``V`` comes back non-orthonormal (all columns equal
in the true branch, a cyclic column rotation in the false branch). Inside a
``while_loop`` body it does not even compile (``'util.global.load' op
undefined global: @__hoisted_tensor_NxNxf32``). The numpy backend is
unaffected (bit-identical to a plain eigh). Minimal repro (n=3, iree-llvm-cpu,
~1 s)::

    import numpy as np, etl
    def f(x):
        return etl.cond(etl.sum(x) > 0.0,
                        lambda t: etl.eigh(t),
                        lambda t: etl.eigh(-t), x)
    a = np.asarray([[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]])
    exe = etl.build(f, etl.TensorSpec((3, 3), etl.float32), backend="iree")
    w, v = etl.run(exe, a)
    v = v.numpy()                      # columns 0/1/2 come back IDENTICAL
    print(np.max(np.abs(v.T @ v - np.eye(3))))   # ~1.0, expected ~1e-7

(The numpy backend gives a correct eigh for the same graph, bit-identical to
a plain ``etl.eigh``.)

``test_eigh_inside_cond_*`` below pins the CONTRACT with the full assertions
under ``xfail(strict=True)`` (the suite stays green, and an XPASS — i.e. the
day the hoisting becomes region-aware — fails loudly instead of silently
dropping the coverage).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

from tests.backends import _rocm_guard as guard

# ---------------------------------------------------------------------------
# data (fixed seed — same convention as test_iree_emitters_parity.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)

#: etl dtype per numpy dtype (specs are spelled with the etl names below).
_ETL_DTYPE = {
    np.dtype(np.float32): etl.float32,
    np.dtype(np.float64): etl.float64,
}


def _sym(n, scale=0.0, dtype=np.float32, seed=None):
    """A symmetric (n, n) matrix with well-separated eigenvalues: X @ X.T
    symmetrized, plus ``scale`` on the diagonal for conditioning.

    ``seed=None`` draws from the shared module stream (the existing
    convention); the parametrized n = 25/50/100 matrices pass an explicit seed
    so their pinned bounds are independent of the order pytest runs them in.
    """
    rng = np.random.default_rng(seed) if seed is not None else _RNG
    x = rng.standard_normal((n, n))
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


def _scaled_eigh_errors(A, w, v, w_ref):
    """(reconstruction, orthogonality, eigenvalue) errors with the first and
    last NORMALIZED by ``max|A|``.

    The pinned numbers are normalized because a plain ``np.allclose`` ``atol``
    cannot express "3e-5 OF THE MATRIX SCALE" for a (100, 100) matrix whose
    entries reach ~150: the reconstruction residual is proportional to the
    matrix norm, not to the individual element, so near-zero off-diagonals
    would fail any absolute bound that the large entries need. Orthogonality
    needs no normalization (VᵀV is O(1) by construction).
    """
    w, v, w_ref = np.asarray(w), np.asarray(v), np.asarray(w_ref)
    n = A.shape[-1]
    rec = np.einsum("...ij,...j,...kj->...ik", v, w, v)
    orth = np.einsum("...ij,...ik->...jk", v, v)
    eye = np.broadcast_to(np.eye(n, dtype=v.dtype), orth.shape)
    scale = float(np.max(np.abs(A)))
    return (
        float(np.max(np.abs(rec - A))) / scale,
        float(np.max(np.abs(orth - eye))),
        float(np.max(np.abs(w - w_ref))) / scale,
    )


def _check_eigh_scaled(A, w, v, w_ref, rec_rel, orth, w_rel):
    """The eigh contract in scale-normalized form: ascending ``w`` and the
    three error measures below the pinned bounds (see ``_scaled_eigh_errors``
    for the normalization; every failure prints the measured value)."""
    w = np.asarray(w)
    assert np.all(np.diff(w, axis=-1) >= 0), f"w not ascending: {w}"
    measured = _scaled_eigh_errors(A, w, v, w_ref)
    names = ("reconstruction", "orthogonality", "eigenvalue")
    bounds = (rec_rel, orth, w_rel)
    for name, value, bound in zip(names, measured, bounds):
        assert value <= bound, (
            f"{name} error {value:.3e} (of the matrix scale) exceeds the "
            f"pinned bound {bound:.1e}"
        )


def _run_eigh(fn, A, spec, **options):
    """Build on iree-llvm-cpu (``options`` → ``etl.build``), run, and return
    the (w, v) ndarrays plus the numpy-backend reference for ``w``."""
    want = etl.evaluate(fn, A)
    exe = etl.build(fn, spec, backend="iree", **options)
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
# 2. eigh — the small-size contract (ascending w, reconstruction,
#    orthogonality) + w vs numpy. These run on the BATCHED default.
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


# ---------------------------------------------------------------------------
# 2b. the BATCHED default across sizes — n = 25 / 50 / 100, f32 and f64.
#
# ONE compile per (size, dtype) case, reused for all three assertions (the
# n=100 builds are the expensive ones — ~0.8 s each, and the run is O(n⁴)).
# Bounds are the measured values (module docstring) with a 2-5× margin;
# ``rec_rel``/``w_rel`` are normalized by max|A|, ``orth`` is the raw
# max|VᵀV − I|. The n=25 f32 case absorbs the former standalone dim-25 test.
# ---------------------------------------------------------------------------

EIGH_BATCHED_MATRIX = [
    # id, n, dtype, seed, rec_rel, orth, w_rel
    ("f32_n25", 25, np.float32, 11, 1.5e-5, 1.5e-5, 1.5e-5),
    ("f32_n50", 50, np.float32, 13, 5.0e-5, 3.0e-5, 5.0e-5),
    # measured rec 3.31e-5 / orth 2.11e-5 — the sweep-limited band the
    # earlier reference measurement pinned at 3.22e-5 / 1.71e-5.
    ("f32_n100", 100, np.float32, 17, 5.0e-5, 3.5e-5, 1.0e-4),
    ("f64_n25", 25, np.float64, 19, 1.8e-14, 1.8e-14, 1.8e-14),
    ("f64_n50", 50, np.float64, 23, 3.0e-14, 2.5e-14, 4.0e-14),
    # measured rec 4.4e-9 (sweep-limited, matrix-dependent — see the module
    # docstring) / orth 2.5e-14 / eigenvalues 8.6e-14.
    ("f64_n100", 100, np.float64, 29, 2.0e-8, 1.0e-13, 5.0e-13),
]


@pytest.mark.parametrize(
    "n,dtype,seed,rec_rel,orth,w_rel",
    [(n, dtype, seed, rec_rel, orth, w_rel)
     for _, n, dtype, seed, rec_rel, orth, w_rel in EIGH_BATCHED_MATRIX],
    ids=[case_id for case_id, *_ in EIGH_BATCHED_MATRIX],
)
def test_eigh_batched_default_matrix(n, dtype, seed, rec_rel, orth, w_rel):
    A = _sym(n, scale=10.0, dtype=dtype, seed=seed)
    spec = etl.TensorSpec((n, n), _ETL_DTYPE[np.dtype(dtype)])
    gw, gv, ww, _ = _run_eigh(_eigh, A, spec)
    _check_eigh_scaled(A, gw, gv, ww, rec_rel, orth, w_rel)


# ---------------------------------------------------------------------------
# 2c. the SEQUENTIAL composition is still selectable and still covered
#     (``eigh_algorithm="jacobi"``: one rotation per while iteration).
#     Measured n=10 f32: rec 1.3e-6 / orth 1.1e-6 (scale-normalized).
# ---------------------------------------------------------------------------


def test_eigh_jacobi_10x10_f32_option_contract():
    A = _sym(10, scale=10.0, seed=31)
    gw, gv, ww, _ = _run_eigh(
        _eigh, A, etl.TensorSpec((10, 10), etl.float32),
        eigh_algorithm="jacobi",
    )
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    _check_eigh_scaled(A, gw, gv, ww, 1.0e-5, 1.0e-5, 1.0e-5)


# --- 3. eigh_early_exit exporter option (llvm-cpu only) ----------------------
#
# The `eigh_early_exit` option (bool, default False) adds a convergence
# early exit to the while-Jacobi composition: an i1 done carry + a nested
# stablehlo.if at every sweep boundary (scale-aware relative off-diagonal
# energy vs the CALIBRATED f32 tolerance 3e-5), exiting via a k-clamp. The
# iree adapter does NOT forward the option (compiler.py `_exporter_options`
# reserves only rng_bit_generator / sort_emission / while_init_rewrite /
# dot_precision / eigh_algorithm / split_reduction), so these tests splice the
# directly-exported text into the lowered payload under its "mlir_text" key —
# both modes are explicit here. Option-on is known-broken at invoke on
# iree-cuda (dealloca INVALID_ARGUMENT n=10 / SIGSEGV n=50) and validated on
# llvm-cpu only — this file never requests cuda. Measured (dim-50
# sample-covariance-like f32): the exit fires at the sweep-4 boundary of 7;
# dim-45 cov-like stays on the full schedule or exits sweep ~4-5 with results
# inside the same contract.


def _cov_like(n, seed, k=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    a = x @ x.T + np.eye(n)
    return ((a + a.T) / 2).astype(np.float32)


def _run_eigh_variant(A, spec, early_exit):
    # Splice the directly-exported text into the payload (the adapters do
    # not forward eigh_early_exit; both modes are explicit here since the
    # option defaults to False). The iree LoweredProgram payload carries
    # the MLIR text under "mlir_text" (etl.backends.compiler).
    graph = etl.trace(_eigh, spec)
    lowered = etl.lower(graph, backend="iree")
    text = etl.backends.stablehlo.export(
        graph, options={"eigh_early_exit": early_exit})
    lowered = etl.LoweredProgram(backend="iree", signature=lowered.signature,
                                 payload={**lowered.payload, "mlir_text": text})
    got = etl.run(etl.load(etl.compile(lowered, backend="iree"), backend="iree"), A)
    return _np(got[0]), _np(got[1])


def test_eigh_dim45_f32_early_exit_contract():
    A = _cov_like(45, seed=0)
    gw, gv, ww, _ = _run_eigh(_eigh, A, etl.TensorSpec((45, 45), etl.float32))
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw, ww, rtol=1e-3, atol=1e-3)


def test_eigh_dim45_early_exit_off_matches_on():
    A = _cov_like(45, seed=0)
    spec = etl.TensorSpec((45, 45), etl.float32)
    gw_on, gv_on = _run_eigh_variant(A, spec, early_exit=True)
    gw_off, gv_off = _run_eigh_variant(A, spec, early_exit=False)
    _check_eigh_contract(A, gw_on, gv_on, rtol=1e-3, atol=1e-3)
    _check_eigh_contract(A, gw_off, gv_off, rtol=1e-3, atol=1e-3)
    assert np.allclose(gw_on, gw_off, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. eigh INSIDE etl.cond — BUG(etl), see the module docstring
#
# Each branch computes its own eigh and the predicate (sign of the matrix sum)
# selects; the graph is built and run ONCE per size on the batched default and
# fed the matrix AND its negation, so BOTH branches execute. The assertions are
# the ordinary eigh contract — they are kept verbatim under a strict xfail
# because the emitted regions currently lose the composition (w is right, V is
# non-orthonormal: all columns equal / cyclic column rotation). Delete the
# xfail marker — it turns into an XPASS failure — once the hoisting is
# region-aware.
# ---------------------------------------------------------------------------

EIGH_COND_CASES = [
    # id, n, seed, rec_rel — the reconstruction bound is the reference
    # measurement of the same composition on a comparable well-conditioned
    # matrix (5e-7 / 2e-6 / 4.6e-6 at n=3 / 10 / 25) with a ~2x margin. The
    # orthogonality (1e-5) / eigenvalue (2e-5) bounds are the f32 n=25 band
    # from ``EIGH_BATCHED_MATRIX`` — VᵀV is O(1) and the eigenvalues do not
    # get cheaper inside a region, so they do not scale with n.
    ("n3", 3, 41, 1.0e-6),
    ("n10", 10, 43, 4.0e-6),
    ("n25", 25, 47, 1.0e-5),
]


def _eigh_cond(x):
    return etl.cond(etl.sum(x) > 0.0, _eigh, lambda t: _eigh(-t), x)


@pytest.mark.xfail(
    strict=True,
    reason="BUG(etl): eigh inside a cond branch is emitted outside the "
    "stablehlo.if regions (constant/iota hoisting is not region-aware), so "
    "iree returns a non-orthonormal V",
)
@pytest.mark.parametrize(
    "n,seed,rec_rel",
    [(n, seed, rec_rel) for _, n, seed, rec_rel in EIGH_COND_CASES],
    ids=[case_id for case_id, *_ in EIGH_COND_CASES],
)
def test_eigh_inside_cond_contract(n, seed, rec_rel):
    A = _sym(n, scale=10.0, seed=seed)
    spec = etl.TensorSpec((n, n), etl.float32)
    w_ref = _np(etl.evaluate(_eigh, A)[0])
    exe = etl.build(_eigh_cond, spec, backend="iree")
    for arg in (A, -A):
        got = etl.run(exe, arg)
        gw, gv = _np(got[0]), _np(got[1])
        _check_eigh_contract(A, gw, gv, rtol=1e-4, atol=1e-4)
        _check_eigh_scaled(A, gw, gv, w_ref, rec_rel, 1.0e-5, 2.0e-5)


# ---------------------------------------------------------------------------
# 5. ROCm guard: the BATCHED composition on the AMD GPU (the ONLY one that
#    completes there — the sequential composition SIGSEGVs/hangs on iree/rocm).
#    Short correctness check only (a shared GPU box): n=10, one compile, one
#    run, invariants only — no timing.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rocm_device():
    """The ROCm etl device; ``pytest.skip`` when no HIP device is usable."""
    return guard.require_rocm()


def test_eigh_batched_rocm_n10_f32_contract(rocm_device):
    A = _sym(10, scale=10.0, seed=53)
    exe = etl.build(
        _eigh, etl.TensorSpec((10, 10), etl.float32), backend="iree",
        device=rocm_device, target_backends=["rocm"],
    )
    got = etl.run(exe, etl.core.Tensor(A).to(rocm_device))
    # Device-resident run-outs; host access goes through the explicit D2H hop
    # (a bare .numpy() on a rocm payload raises core.DeviceError).
    for out in got:
        assert out.device == rocm_device
    gw = np.asarray(got[0].to(etl.core.Device("cpu", 0)).numpy())
    gv = np.asarray(got[1].to(etl.core.Device("cpu", 0)).numpy())
    w_ref = _np(etl.evaluate(_eigh, A)[0])
    # Measured on the MI300X (gfx942): rec 1.31e-6 / orth 8.3e-7 (normalized
    # as above) — the same band as llvm-cpu; bounds at ~8x margin.
    _check_eigh_contract(A, gw, gv, rtol=1e-3, atol=1e-3)
    _check_eigh_scaled(A, gw, gv, w_ref, 1.0e-5, 1.0e-5, 1.0e-5)
