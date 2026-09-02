"""iree parity for the v1 StableHLO emitters added in the GPU-export line.

Covered here (all newly v1 — see ``etl/backends/stablehlo/ops.py``):

* gather / sort / argsort / argmax / argmin / tile / scatter,
* the SplitMix64 random family (``random_key_mix`` / ``random_uniform`` /
  ``random_normal`` / ``random_randint`` / ``random_permutation`` —
  uniform/randint/permutation/key_mix are bit-exact vs the numpy kernels;
  ``random_normal`` (f32 out) takes the DOCUMENTED f32 Box–Muller fast path
  on compiler backends — deterministic (same key ⇒ bit-identical across
  runs, asserted below) but ~1e-6 vs the numpy f64 math, so the parity
  checks are tolerance-based for normal and the split_n chain),
* the shift primitives (``bitwise_left_shift`` / ``bitwise_right_shift``),
* ``etl.scan`` (desugars at trace time into while + gather + scatter — all
  v1, so scan graphs compile and run on iree-llvm-cpu).

Every graph is built with ``etl.build(..., backend="iree")`` (llvm-cpu),
run, and compared against ``etl.evaluate`` (the numpy reference backend) —
EXACT (``np.array_equal``) for integer / data-movement ops, plus explicit
``np.*`` references where the semantics are unambiguous (sort, argsort
stable/descending, argmax/argmin, tile, scatter, gather). Test data mirrors
the repo-root ``probe_new_emitters.py`` with the same fixed seed
(``np.random.default_rng(7)``).

The iree-cuda smoke set (device 5, ``target_backends=["cuda"]``) skips when
the IREE cuda HAL driver or the GPU is unavailable. The smokes follow the
explicit device-placement model: each run input is moved to
``Device("cuda", 5)`` via the explicit ``core.Tensor(...).to(...)``
transfer before ``etl.run`` (no implicit host→device staging happens at
the run boundary — a cuda executable rejects host inputs), and the cuda
outputs are read back through the explicit ``.to(core.Device("cpu", 0))``
transfer (a non-cpu payload's ``.numpy()`` raises ``DeviceError``).
While-loop graphs are deliberately excluded from cuda (upstream iree cuda
HAL crashes on while — documented in ``etl/bench`` Known Issues; ``scan``
is therefore cpu-only here).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (fixed seed — same draws as probe_new_emitters.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)

_P, _D, _K = 8, 5, 4
POP = _RNG.standard_normal((_P, _D)).astype(np.float32)
IDX0 = np.array([3, 1, 6, 1], dtype=np.int64)
IDX1 = np.array([2, 0, 4], dtype=np.int64)
NEG = np.array([-1, 2], dtype=np.int64)

X1 = np.array([5.0, 1.0, 5.0, 3.0, 2.0], dtype=np.float32)  # ties at 5.0
X2 = _RNG.standard_normal((3, 6)).astype(np.float32)
X1B = np.array([True, False, True, False], dtype=np.bool_)

AM = np.array([3.0, 1.0, 3.0, 2.0], dtype=np.float32)  # tie at 3.0
A2D = _RNG.standard_normal((3, 7)).astype(np.float32)

T = np.arange(6, dtype=np.float32).reshape(2, 3)

SC = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
SI = np.array([4, 1, 3], dtype=np.int64)
SU = np.array([7.0, 8.0, 9.0], dtype=np.float32)
MAT = np.arange(12, dtype=np.float32).reshape(3, 4)
MUPD = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
IDX2 = np.array([2, 0], dtype=np.int64)
UPD2 = np.array(
    [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], dtype=np.float32
)
ICOL = np.array([1, 3], dtype=np.int64)
UCOL = np.array(
    [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32
)

A64 = np.array([1, -8, 255], dtype=np.int64)
U64 = np.array([1, 2 ** 63, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays.

    Device-aware readback (explicit device placement): a NON-cpu tensor
    (an iree-cuda run output) is first moved to the host via the explicit
    ``t.to(core.Device('cpu', 0))`` transfer — ``.numpy()`` on a
    non-cpu-kind payload raises ``core.DeviceError``. Cpu tensors (the
    numpy backend and iree-llvm-cpu) keep the plain ``.numpy()`` path —
    byte-identical to the pre-placement model.
    """
    if isinstance(v, etl.Tensor):
        if v.device.kind != "cpu":
            v = v.to(etl.core.Device("cpu", 0))
        return np.asarray(v.numpy())
    return np.asarray(v)


def _place_inputs(args, device):
    """Explicitly place host inputs on ``device`` before ``etl.run``.

    Run-boundary device contract (``core.Tensor.to`` / pipeline ``run``):
    no implicit host→device transfer happens at the run boundary, so a
    cuda executable rejects cpu/host input tensors (including raw numpy
    arrays, which are cpu:0) with ``core.DeviceError``. Each ndarray input
    is wrapped via ``core.Tensor`` and moved with the explicit
    ``.to(device)``; ``core.Tensor`` inputs pass through ``.to(device)``.
    Cpu devices (``device is None`` or kind ``"cpu"`` — the llvm-cpu
    runs) are a NO-OP: the host arrays are fed exactly as before.
    """
    if device is None or device.kind == "cpu":
        return args
    placed = []
    for value in args:
        if isinstance(value, etl.Tensor):
            placed.append(value.to(device))
        elif isinstance(value, np.ndarray):
            placed.append(etl.core.Tensor(value).to(device))
        else:
            raise TypeError(
                "cannot place a cuda run input of type "
                f"{type(value).__qualname__}: expected a numpy ndarray or "
                "an etl.Tensor"
            )
    return tuple(placed)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _assert_close(got, want, rtol=1e-5, atol=1e-5):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.allclose(gp, wp, rtol=rtol, atol=atol), f"{gp} != {wp}"


def _parity(fn, *specs, args, exact=True, ref=None, **iree_kwargs):
    """Build on iree, run, and compare vs ``etl.evaluate`` (numpy reference).

    ``ref``: optional explicit numpy reference (an ndarray) asserted against
    BOTH sides; ``None`` means the numpy backend output is the reference.

    Device-aware (explicit device placement): when ``iree_kwargs`` carry a
    non-cpu ``device`` (the cuda smokes), every run input is explicitly
    placed on that device first via ``_place_inputs`` (the run boundary
    never transfers implicitly — a cuda executable rejects host inputs)
    and the run outputs are read back through the explicit ``.to(cpu)``
    host transfer inside ``_np`` (a non-cpu payload's ``.numpy()`` raises
    ``DeviceError``). Cpu devices (``device`` absent — the llvm-cpu runs)
    feed the host arrays as-is: byte-identical to the pre-placement model.
    """
    want = etl.evaluate(fn, *args)
    if ref is not None:
        _assert_exact(want, ref)
    exe = etl.build(fn, *specs, backend="iree", **iree_kwargs)
    got = etl.run(exe, *_place_inputs(args, iree_kwargs.get("device")))
    if exact:
        _assert_exact(got, want)
    else:
        _assert_close(got, want)


def _require_cuda():
    """Skip when the IREE cuda HAL driver or GPU 5 is unavailable."""
    import iree.runtime as rt

    try:
        # etl Device("cuda", 5) maps to iree device_id 6 (1-based ids).
        rt.get_driver("cuda").create_device(device_id=6)
    except Exception as exc:  # noqa: BLE001 — any driver/device failure skips
        pytest.skip(f"IREE cuda HAL driver or GPU 5 unavailable: {exc}")


# ---------------------------------------------------------------------------
# 1. gather (axis 0/1, static + runtime indices, negative, 0-d index)
# ---------------------------------------------------------------------------


def _g_axis0(x, i):
    return etl.gather(x, i, axis=0)


def _g_axis1(x, i):
    return etl.gather(x, i, axis=1)


def _g_axis0_static(x):
    idx = etl.constant(etl.core.Tensor(IDX0.copy()))
    return etl.gather(x, idx, axis=0)


def _g_axis1_static(x):
    idx = etl.constant(etl.core.Tensor(IDX1.copy()))
    return etl.gather(x, idx, axis=1)


GATHER_CASES = [
    ("gather_axis0_rt", _g_axis0,
     (etl.TensorSpec((_P, _D), etl.float32), etl.TensorSpec((_K,), etl.int64)),
     (POP, IDX0), np.take(POP, IDX0, axis=0)),
    ("gather_axis0_static", _g_axis0_static,
     (etl.TensorSpec((_P, _D), etl.float32),), (POP,),
     np.take(POP, IDX0, axis=0)),
    ("gather_axis1_rt", _g_axis1,
     (etl.TensorSpec((_P, _D), etl.float32), etl.TensorSpec((3,), etl.int64)),
     (POP, IDX1), np.take(POP, IDX1, axis=1)),
    ("gather_axis1_static", _g_axis1_static,
     (etl.TensorSpec((_P, _D), etl.float32),), (POP,),
     np.take(POP, IDX1, axis=1)),
    ("gather_negative_indices", _g_axis0,
     (etl.TensorSpec((_P, _D), etl.float32), etl.TensorSpec((2,), etl.int64)),
     (POP, NEG), np.take(POP, NEG, axis=0)),
    ("gather_0d_index", _g_axis0,
     (etl.TensorSpec((_P, _D), etl.float32), etl.TensorSpec((), etl.int64)),
     (POP, np.array(3, dtype=np.int64)), np.take(POP, np.array(3), axis=0)),
]


@pytest.mark.parametrize(
    "fn,specs,args,ref",
    [(fn, specs, args, ref) for _, fn, specs, args, ref in GATHER_CASES],
    ids=[c[0] for c in GATHER_CASES],
)
def test_gather_iree_parity(fn, specs, args, ref):
    _parity(fn, *specs, args=args, exact=True, ref=ref)


# ---------------------------------------------------------------------------
# 2. sort / argsort / argmax / argmin
# ---------------------------------------------------------------------------


def _sort1(x):
    return etl.sort(x, axis=0)


def _sort1d_desc(x):
    return etl.sort(x, axis=0, descending=True)


def _sort2(x):
    return etl.sort(x, axis=1)


def _argsort1(x):
    return etl.argsort(x, axis=0)


def _argsort1d_desc(x):
    return etl.argsort(x, axis=0, descending=True)


def _argsort2(x):
    return etl.argsort(x, axis=1)


def _argsort_bool(x):
    return etl.argsort(x, axis=0)


def _argmax1(x):
    return etl.argmax(x, axis=0)


def _argmin1(x):
    return etl.argmin(x, axis=0)


def _argmax_kd(x):
    return etl.argmax(x, axis=1, keepdims=True)


def _argmax_flat(x):
    return etl.argmax(x)


SORT_ARGSORT_ARGMINMAX_CASES = [
    ("sort_1d", _sort1, (etl.TensorSpec((5,), etl.float32),), (X1,),
     np.sort(X1)),
    ("sort_1d_descending", _sort1d_desc, (etl.TensorSpec((5,), etl.float32),),
     (X1,), np.sort(X1)[::-1]),
    ("sort_2d_axis1", _sort2, (etl.TensorSpec((3, 6), etl.float32),), (X2,),
     np.sort(X2, axis=1)),
    ("argsort_1d_stable_ties", _argsort1, (etl.TensorSpec((5,), etl.float32),),
     (X1,), np.argsort(X1, kind="stable")),
    # descending ties break in REVERSED index order — matches the numpy
    # backend's np.argsort(x)[::-1] convention (probe-validated exact)
    ("argsort_1d_descending", _argsort1d_desc,
     (etl.TensorSpec((5,), etl.float32),), (X1,), np.argsort(X1)[::-1]),
    ("argsort_2d_axis1", _argsort2, (etl.TensorSpec((3, 6), etl.float32),),
     (X2,), np.argsort(X2, axis=1, kind="stable")),
    ("argsort_bool", _argsort_bool, (etl.TensorSpec((4,), np.bool_),), (X1B,),
     np.argsort(X1B, kind="stable")),
    ("argmax_1d", _argmax1, (etl.TensorSpec((4,), etl.float32),), (AM,),
     np.argmax(AM)),
    ("argmin_1d", _argmin1, (etl.TensorSpec((4,), etl.float32),), (AM,),
     np.argmin(AM)),
    ("argmax_2d_axis1_keepdims", _argmax_kd,
     (etl.TensorSpec((3, 7), etl.float32),), (A2D,),
     np.argmax(A2D, axis=1, keepdims=True)),
    ("argmax_axis_none_flat", _argmax_flat,
     (etl.TensorSpec((3, 7), etl.float32),), (A2D,), np.argmax(A2D)),
]


@pytest.mark.parametrize(
    "fn,specs,args,ref",
    [(fn, specs, args, ref) for _, fn, specs, args, ref in
     SORT_ARGSORT_ARGMINMAX_CASES],
    ids=[c[0] for c in SORT_ARGSORT_ARGMINMAX_CASES],
)
def test_sort_argsort_argminmax_iree_parity(fn, specs, args, ref):
    _parity(fn, *specs, args=args, exact=True, ref=ref)


# ---------------------------------------------------------------------------
# 3. tile (right-align, multi-dim, rank-promote) / scatter (1-D, row 0-d,
# multi-row, axis=1)
# ---------------------------------------------------------------------------


def _tile1(x):
    return etl.tile(x, (2,))


def _tile2(x):
    return etl.tile(x, (2, 2))


def _tile3(x):
    return etl.tile(x, (2, 1))


def _tile4(x):
    return etl.tile(x, (2, 2, 1))


def _scatter1(x, i, u):
    return etl.scatter(x, i, u, axis=0)


def _scatter_row_0d(x, i, u):
    return etl.scatter(x, i, u, axis=0)


def _scatter_row_multi(x, i, u):
    return etl.scatter(x, i, u, axis=0)


def _scatter_axis1(x, i, u):
    return etl.scatter(x, i, u, axis=1)


def _scatter_1d_ref():
    ref = SC.copy()
    ref[SI] = SU
    return ref


def _scatter_row_0d_ref():
    ref = MAT.copy()
    ref[1] = MUPD
    return ref


def _scatter_row_multi_ref():
    ref = MAT.copy()
    ref[IDX2] = UPD2
    return ref


def _scatter_axis1_ref():
    ref = MAT.copy()
    ref[:, ICOL] = UCOL
    return ref


TILE_SCATTER_CASES = [
    ("tile_rightalign", _tile1, (etl.TensorSpec((2, 3), etl.float32),), (T,),
     np.tile(T, (2,))),
    ("tile_2x2", _tile2, (etl.TensorSpec((2, 3), etl.float32),), (T,),
     np.tile(T, (2, 2))),
    ("tile_2x1", _tile3, (etl.TensorSpec((2, 3), etl.float32),), (T,),
     np.tile(T, (2, 1))),
    ("tile_rank_promote", _tile4, (etl.TensorSpec((2, 3), etl.float32),),
     (T,), np.tile(T, (2, 2, 1))),
    ("scatter_1d", _scatter1,
     (etl.TensorSpec((5,), etl.float32), etl.TensorSpec((3,), etl.int64),
      etl.TensorSpec((3,), etl.float32)),
     (SC, SI, SU), _scatter_1d_ref()),
    ("scatter_row_0d_index", _scatter_row_0d,
     (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((), etl.int64),
      etl.TensorSpec((4,), etl.float32)),
     (MAT, np.array(1, dtype=np.int64), MUPD), _scatter_row_0d_ref()),
    ("scatter_row_multi", _scatter_row_multi,
     (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((2,), etl.int64),
      etl.TensorSpec((2, 4), etl.float32)),
     (MAT, IDX2, UPD2), _scatter_row_multi_ref()),
    ("scatter_axis1", _scatter_axis1,
     (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((2,), etl.int64),
      etl.TensorSpec((3, 2), etl.float32)),
     (MAT, ICOL, UCOL), _scatter_axis1_ref()),
]


@pytest.mark.parametrize(
    "fn,specs,args,ref",
    [(fn, specs, args, ref) for _, fn, specs, args, ref in TILE_SCATTER_CASES],
    ids=[c[0] for c in TILE_SCATTER_CASES],
)
def test_tile_scatter_iree_parity(fn, specs, args, ref):
    _parity(fn, *specs, args=args, exact=True, ref=ref)


# ---------------------------------------------------------------------------
# 4. scan (trace-time desugar into while + gather + scatter) and the shift
# primitives
# ---------------------------------------------------------------------------


def _fn_scan(x):
    def step(carry, elem):
        return etl.add(carry, elem), etl.multiply(carry, elem)

    return etl.scan(step, 0.0, x)[1]


def _scan_ref(x):
    carry = np.float32(0.0)
    out = []
    for elem in x:
        out.append(carry * elem)
        carry = carry + elem
    return np.asarray(out, dtype=np.float32)


def _shift_left(x, s):
    return etl.bitwise_left_shift(x, s)


def _shift_right(x, s):
    return etl.bitwise_right_shift(x, s)


def _shift_right_u64(x, s):
    # Logical shift on u64 via the mask trick (probe-validated bit-exact vs
    # np.right_shift on the raw u64 view).
    one = etl.constant(etl.core.Tensor(np.array([1], dtype=np.int64)))
    mask = etl.subtract(etl.bitwise_left_shift(one, etl.subtract(64, s)), 1)
    return etl.bitwise_and(etl.bitwise_right_shift(x, s), mask)


SCAN_SHIFT_CASES = [
    ("scan_desugared_while_gather_scatter", _fn_scan,
     (etl.TensorSpec((4,), etl.float32),),
     (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),), None),
    ("shift_left_i64", _shift_left,
     (etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64)),
     (A64, np.array(3, dtype=np.int64)), np.left_shift(A64, 3)),
    ("shift_right_i64_arithmetic", _shift_right,
     (etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64)),
     (A64, np.array(3, dtype=np.int64)), np.right_shift(A64, 3)),
    ("shift_right_u64_logical_mask", _shift_right_u64,
     (etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64)),
     (U64.view(np.int64), np.array(3, dtype=np.int64)),
     np.right_shift(U64, 3).view(np.int64)),
]


@pytest.mark.parametrize(
    "fn,specs,args,ref",
    [(fn, specs, args, ref) for _, fn, specs, args, ref in SCAN_SHIFT_CASES],
    ids=[c[0] for c in SCAN_SHIFT_CASES],
)
def test_scan_and_shifts_iree_parity(fn, specs, args, ref):
    if ref is None:  # the scan case: explicit cumulative-product reference
        ref = _scan_ref(args[0])
    _parity(fn, *specs, args=args, exact=True, ref=ref)


# ---------------------------------------------------------------------------
# 5. random — bit-exact SplitMix64 parity vs the numpy kernels
# ---------------------------------------------------------------------------


def _rkey():
    k = etl.random.key(42)
    return etl.random.split(k)


def _runiform(k):
    return etl.random.uniform(k, (4, 5), low=-1.0, high=3.0)


def _rnormal(k):
    return etl.random.normal(k, (4, 5), mean=2.0, std=0.5)


def _rrandint(k):
    return etl.random.randint(k, (4, 5), low=-10, high=10)


def _rperm(k):
    return etl.random.permutation(k, 12)


def _rperm64(k):
    return etl.random.permutation(k, 12, dtype=etl.int64)


def _rchain(k):
    ks = etl.random.split_n(k, 3)
    return (
        etl.random.uniform(ks[0], (4, 3))
        + etl.random.normal(ks[1], (4, 3))
        + etl.random.randint(ks[2], (4, 3), low=0, high=5)
    )


SEED = np.array(42, dtype=np.int64)
KEY_SPEC = etl.TensorSpec((), etl.int64)

# (id, fn, specs, args, exact) — the numpy kernel IS the reference for
# uniform/randint/permutation/key_mix (the emitter must reproduce them
# bit-for-bit); normal is the documented f32 fast-path deviation (see
# etl/backends/stablehlo/random_export.py) → explicit tolerance, and its
# determinism (same key ⇒ bit-identical draws) is asserted in
# test_random_iree_determinism.
RANDOM_CASES = [
    ("random_key_mix_split", _rkey, (), (), True),
    ("random_uniform", _runiform, (KEY_SPEC,), (SEED,), True),
    ("random_normal", _rnormal, (KEY_SPEC,), (SEED,), False),
    ("random_randint", _rrandint, (KEY_SPEC,), (SEED,), True),
    ("random_permutation_i32", _rperm, (KEY_SPEC,), (SEED,), True),
    ("random_permutation_i64", _rperm64, (KEY_SPEC,), (SEED,), True),
    ("random_split_n_chain", _rchain, (KEY_SPEC,), (SEED,), False),
]

#: Documented tolerance for the f32 normal fast path (measured ~1e-6; 10x
#: headroom over the f32 rounding budget).
RANDOM_NORMAL_TOL = dict(rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "case_id,fn,specs,args,exact",
    [(c[0], c[1], c[2], c[3], c[4]) for c in RANDOM_CASES],
    ids=[c[0] for c in RANDOM_CASES],
)
def test_random_iree_parity(case_id, fn, specs, args, exact):
    want = etl.evaluate(fn, *args)
    exe = etl.build(fn, *specs, backend="iree")
    got = etl.run(exe, *args)
    if exact:
        _assert_exact(got, want)
    else:
        _assert_close(got, want, **RANDOM_NORMAL_TOL)
    if "permutation" in case_id:
        got = _np(etl.evaluate(fn, *args))
        assert sorted(got.tolist()) == list(range(12))  # a real permutation


@pytest.mark.parametrize(
    "case_id,fn,specs,args",
    [(c[0], c[1], c[2], c[3]) for c in RANDOM_CASES],
    ids=[c[0] for c in RANDOM_CASES],
)
def test_random_iree_determinism(case_id, fn, specs, args):
    """The hard etl.random contract on compiler backends: same key +
    operands ⇒ BIT-IDENTICAL draws across runs (the values may deviate from
    the numpy reference for the f32 normal fast path, but never across
    iree runs)."""
    exe = etl.build(fn, *specs, backend="iree")
    g1 = etl.run(exe, *args)
    g2 = etl.run(exe, *args)
    _assert_exact(g1, g2)


# ---------------------------------------------------------------------------
# 6. iree-cuda smoke set (device 5; while-loop graphs deliberately excluded
# — upstream iree cuda HAL crashes on while). Explicit device placement:
# _parity moves every host input to Device("cuda", 5) via
# core.Tensor(...).to(...) before etl.run (the run boundary never
# transfers implicitly — host inputs to a cuda executable raise
# DeviceError) and reads the cuda outputs back through the explicit
# .to(core.Device("cpu", 0)) host transfer in _np.
# ---------------------------------------------------------------------------

CUDA_CASES = [
    ("cuda_gather_axis1", _g_axis1,
     (etl.TensorSpec((_P, _D), etl.float32), etl.TensorSpec((3,), etl.int64)),
     (POP, IDX1), np.take(POP, IDX1, axis=1)),
    ("cuda_sort_2d_axis1", _sort2, (etl.TensorSpec((3, 6), etl.float32),),
     (X2,), np.sort(X2, axis=1)),
    ("cuda_random_uniform", _runiform, (KEY_SPEC,), (SEED,), None),
    ("cuda_tile_2x1", _tile3, (etl.TensorSpec((2, 3), etl.float32),), (T,),
     np.tile(T, (2, 1))),
    ("cuda_scatter_row_0d", _scatter_row_0d,
     (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((), etl.int64),
      etl.TensorSpec((4,), etl.float32)),
     (MAT, np.array(1, dtype=np.int64), MUPD), _scatter_row_0d_ref()),
]


@pytest.mark.parametrize(
    "fn,specs,args,ref",
    [(fn, specs, args, ref) for _, fn, specs, args, ref in CUDA_CASES],
    ids=[c[0] for c in CUDA_CASES],
)
def test_iree_cuda_smoke(fn, specs, args, ref):
    _require_cuda()
    _parity(
        fn, *specs, args=args, exact=True, ref=ref,
        device=etl.core.Device("cuda", 5), target_backends=["cuda"],
    )
