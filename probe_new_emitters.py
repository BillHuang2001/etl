"""Probe: exercise all new StableHLO emissions through iree (llvm-cpu) vs numpy."""
import sys
import numpy as np
import etl

OUT = {}
FAIL = []


def _to_np(v):
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def check(name, got, want, exact=False, tol=1e-5):
    # Pytree-aware: tuple outputs (e.g. random.split's key pair) compare
    # leaf-by-leaf; Tensor leaves convert to ndarrays.
    g = etl.tree_map(_to_np, got)
    w = etl.tree_map(_to_np, want)
    g_leaves, w_leaves = etl.tree_leaves(g), etl.tree_leaves(w)
    if len(g_leaves) != len(w_leaves):
        FAIL.append(f"{name}: leaf count {len(g_leaves)} != {len(w_leaves)}")
        return
    ok = True
    maxdiff = 0.0
    for gp, wp in zip(g_leaves, w_leaves):
        if gp.shape != wp.shape:
            FAIL.append(f"{name}: shape {gp.shape} != {wp.shape}")
            return
        if exact:
            leaf_ok = np.array_equal(gp, wp)
        else:
            leaf_ok = np.allclose(gp, wp, rtol=tol, atol=tol)
        maxdiff = max(
            maxdiff, float(np.max(np.abs(gp.astype(np.float64) - wp.astype(np.float64))))
        )
        ok = ok and leaf_ok
    msg = f"{name}: {'EXACT' if exact else 'allclose'} {'OK' if ok else 'FAIL'} maxdiff={maxdiff}"
    print(msg)
    if not ok:
        FAIL.append(msg)


def run_iree(fn, *specs):
    exe = etl.build(fn, *specs, backend="iree")
    return exe


def run_cuda(fn, *specs):
    exe = etl.build(fn, *specs, backend="iree", device=etl.core.Device("cuda", 5), target_backends=["cuda"])
    return exe


def cmp(name, fn, *specs, args=(), exact=False, tol=1e-5, backend="iree"):
    try:
        exe = run_iree(fn, *specs) if backend == "iree" else run_cuda(fn, *specs)
        want = etl.evaluate(fn, *args)
        got = etl.run(exe, *args)
        check(name, got.numpy() if isinstance(got, etl.Tensor) else got,
              want.numpy() if isinstance(want, etl.Tensor) else want, exact=exact, tol=tol)
    except Exception as e:
        import traceback
        traceback.print_exc()
        FAIL.append(f"{name}: EXCEPTION {e!r}")


rng = np.random.default_rng(7)

# ---------------- gather ----------------
P, D, K = 8, 5, 4
pop = rng.standard_normal((P, D)).astype(np.float32)
idx0 = np.array([3, 1, 6, 1], dtype=np.int64)
idx1 = np.array([2, 0, 4], dtype=np.int64)   # axis=1 indices
neg = np.array([-1, 2], dtype=np.int64)


def g0(x, i):
    return etl.gather(x, i, axis=0)


def g1(x, i):
    return etl.gather(x, i, axis=1)


def g1s(x):
    return etl.gather(x, etl.constant(etl.core.Tensor(np.array([2, 0, 4], dtype=np.int64))), axis=1)


def g0s(x):
    return etl.gather(x, etl.constant(etl.core.Tensor(np.array([3, 1, 6, 1], dtype=np.int64))), axis=0)


cmp("gather_axis0_rt", g0, etl.TensorSpec((P, D), etl.float32), etl.TensorSpec((K,), etl.int64), args=(pop, idx0), exact=True)
cmp("gather_axis0_static", g0s, etl.TensorSpec((P, D), etl.float32), args=(pop,), exact=True)
cmp("gather_axis1_rt", g1, etl.TensorSpec((P, D), etl.float32), etl.TensorSpec((3,), etl.int64), args=(pop, idx1), exact=True)
cmp("gather_axis1_static", g1s, etl.TensorSpec((P, D), etl.float32), args=(pop,), exact=True)
cmp("gather_neg", g0, etl.TensorSpec((P, D), etl.float32), etl.TensorSpec((2,), etl.int64), args=(pop, neg), exact=True)


def g_scalar(x, i):
    return etl.gather(x, i, axis=0)


cmp("gather_0d", g_scalar, etl.TensorSpec((P, D), etl.float32), etl.TensorSpec((), etl.int64), args=(pop, np.array(3, dtype=np.int64)), exact=True)

# ---------------- sort / argsort ----------------
x1 = np.array([5.0, 1.0, 5.0, 3.0, 2.0], dtype=np.float32)
x2 = rng.standard_normal((3, 6)).astype(np.float32)
x1b = np.array([True, False, True, False], dtype=np.bool_)


def s1(x):
    return etl.sort(x, axis=0)


def s1d(x):
    return etl.sort(x, axis=0, descending=True)


def s2(x):
    return etl.sort(x, axis=1)


def a1(x):
    return etl.argsort(x, axis=0)


def a1d(x):
    return etl.argsort(x, axis=0, descending=True)


def a2(x):
    return etl.argsort(x, axis=1)


def sb(x):
    return etl.argsort(x, axis=0)


cmp("sort_1d", s1, etl.TensorSpec((5,), etl.float32), args=(x1,), exact=True)
cmp("sort_1d_desc", s1d, etl.TensorSpec((5,), etl.float32), args=(x1,), exact=True)
cmp("sort_2d_axis1", s2, etl.TensorSpec((3, 6), etl.float32), args=(x2,), exact=True)
cmp("argsort_1d_stable", a1, etl.TensorSpec((5,), etl.float32), args=(x1,), exact=True)
cmp("argsort_1d_desc", a1d, etl.TensorSpec((5,), etl.float32), args=(x1,), exact=True)
cmp("argsort_2d_axis1", a2, etl.TensorSpec((3, 6), etl.float32), args=(x2,), exact=True)
cmp("argsort_bool", sb, etl.TensorSpec((4,), np.bool_), args=(x1b,), exact=True)

# ---------------- argmax / argmin ----------------
am = np.array([3.0, 1.0, 3.0, 2.0], dtype=np.float32)
a2d = rng.standard_normal((3, 7)).astype(np.float32)


def mx(x):
    return etl.argmax(x, axis=0)


def mn(x):
    return etl.argmin(x, axis=0)


def mxk(x):
    return etl.argmax(x, axis=1, keepdims=True)


def mxflat(x):
    return etl.argmax(x)


cmp("argmax_1d", mx, etl.TensorSpec((4,), etl.float32), args=(am,), exact=True)
cmp("argmin_1d", mn, etl.TensorSpec((4,), etl.float32), args=(am,), exact=True)
cmp("argmax_2d_keepdims", mxk, etl.TensorSpec((3, 7), etl.float32), args=(a2d,), exact=True)
cmp("argmax_flat", mxflat, etl.TensorSpec((3, 7), etl.float32), args=(a2d,), exact=True)


def mx0(x):
    return etl.argmax(x, axis=0)


cmp("argmax_2d_axis0", mx0, etl.TensorSpec((3, 7), etl.float32), args=(a2d,), exact=True)

# ---------------- tile ----------------
t = np.arange(6, dtype=np.float32).reshape(2, 3)


def t1(x):
    return etl.tile(x, (2,))


def t2(x):
    return etl.tile(x, (2, 2))


def t3(x):
    return etl.tile(x, (2, 1))


def t4(x):
    return etl.tile(x, (2, 2, 1))


cmp("tile_rightalign", t1, etl.TensorSpec((2, 3), etl.float32), args=(t,), exact=True)
cmp("tile_2x2", t2, etl.TensorSpec((2, 3), etl.float32), args=(t,), exact=True)
cmp("tile_2x1", t3, etl.TensorSpec((2, 3), etl.float32), args=(t,), exact=True)
cmp("tile_promote", t4, etl.TensorSpec((2, 3), etl.float32), args=(t,), exact=True)

# ---------------- scatter ----------------
sc = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
si = np.array([4, 1, 3], dtype=np.int64)
su = np.array([7.0, 8.0, 9.0], dtype=np.float32)
mat = np.arange(12, dtype=np.float32).reshape(3, 4)
mupd = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)


def sc1(x, i, u):
    return etl.scatter(x, i, u, axis=0)


def sc2(x, i, u):
    return etl.scatter(x, i, u, axis=0)


cmp("scatter_1d", sc1, etl.TensorSpec((5,), etl.float32), etl.TensorSpec((3,), etl.int64), etl.TensorSpec((3,), etl.float32), args=(sc, si, su), exact=True)


def scrow(x, i, u):
    return etl.scatter(x, i, u, axis=0)


cmp("scatter_row_0d", scrow, etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((), etl.int64), etl.TensorSpec((4,), etl.float32), args=(mat, np.array(1, dtype=np.int64), mupd), exact=True)


def scrow2(x, i, u):
    return etl.scatter(x, i, u, axis=0)


idx2 = np.array([2, 0], dtype=np.int64)
upd2 = np.array([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], dtype=np.float32)
cmp("scatter_row_multi", scrow2, etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((2,), etl.int64), etl.TensorSpec((2, 4), etl.float32), args=(mat, idx2, upd2), exact=True)


def sccol(x, i, u):
    return etl.scatter(x, i, u, axis=1)


icol = np.array([1, 3], dtype=np.int64)
# Frontend rule: updates = x.shape[:axis] + indices.shape + x.shape[axis+1:]
# = (3,) + (2,) + () = (3, 2).
ucol = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
cmp("scatter_axis1_col", sccol, etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((2,), etl.int64), etl.TensorSpec((3, 2), etl.float32), args=(mat, icol, ucol), exact=True)

# ---------------- shifts ----------------
a64 = np.array([1, -8, 255], dtype=np.int64)
u64 = np.array([1, 2**63, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)


def shl(x, s):
    return etl.bitwise_left_shift(x, s)


def shr(x, s):
    return etl.bitwise_right_shift(x, s)


def shr_u64(x, s):
    one = etl.constant(etl.core.Tensor(np.array([1], dtype=np.int64)))
    mask = etl.subtract(etl.bitwise_left_shift(one, etl.subtract(64, s)), 1)
    return etl.bitwise_and(etl.bitwise_right_shift(x, s), mask)


cmp("shift_left_i64", shl, etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64), args=(a64, np.array(3, dtype=np.int64)), exact=True)
cmp("shift_right_i64_arith", shr, etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64), args=(a64, np.array(3, dtype=np.int64)), exact=True)
cmp("shift_right_u64_logical", shr_u64, etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64),
    args=(u64.view(np.int64), np.array(3, dtype=np.int64)), exact=True)
# The mask trick must equal numpy's native uint64 shift, bit for bit.
try:
    exe = run_iree(shr_u64, etl.TensorSpec((3,), etl.int64), etl.TensorSpec((), etl.int64))
    got_u64 = etl.run(exe, u64.view(np.int64), np.array(3, dtype=np.int64)).numpy().view(np.uint64)
    check("shift_right_u64_numpy_parity", got_u64, np.right_shift(u64, 3), exact=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    FAIL.append(f"shift_right_u64_numpy_parity: EXCEPTION {e!r}")

# ---------------- random (SplitMix64 parity, BIT-EXACT) ----------------
def rkey():
    k = etl.random.key(42)
    return etl.random.split(k)


def runiform(k):
    return etl.random.uniform(k, (4, 5), low=-1.0, high=3.0)


def rnormal(k):
    return etl.random.normal(k, (4, 5), mean=2.0, std=0.5)


def rrandint(k):
    return etl.random.randint(k, (4, 5), low=-10, high=10)


def rperm(k):
    return etl.random.permutation(k, 12)


def rperm64(k):
    return etl.random.permutation(k, 12, dtype=etl.int64)


cmp("random_key_mix", rkey, args=(), exact=True)
cmp("random_uniform", runiform, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_normal", rnormal, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_randint", rrandint, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_permutation", rperm, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=True)
cmp("random_permutation_i64", rperm64, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=True)

# split_n + uniform chained (PSO-style stream)
def rchain(k):
    ks = etl.random.split_n(k, 3)
    return etl.random.uniform(ks[0], (4, 3)) + etl.random.normal(ks[1], (4, 3)) + etl.random.randint(ks[2], (4, 3), low=0, high=5)


cmp("random_chain", rchain, etl.TensorSpec((), etl.int64), args=(np.array(42, dtype=np.int64),), exact=False, tol=1e-5)

print()
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(" -", f)
    sys.exit(1)
print("ALL PROBES PASSED")
