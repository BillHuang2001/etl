"""Independent manager-level spot-check of the 16 sparse kernels (NOT committed).
Fresh random data, own harness; covers the objective's required families."""
import sys
import numpy as np

import etl
from etl import core
from etl.trace import current_builder

PASS = FAIL = 0


def check(label, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {label} {extra}")


def sym(value):
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape, location=None
    )


def mk(name, operands, attributes):
    b = current_builder()
    op = b.create(name, operands=tuple(o.value for o in operands), attributes=attributes)
    return tuple(sym(v) for v in op.results)


def run_graph(fn, *specs):
    graph = etl.trace(fn, *specs)
    lowered = etl.lower(graph, backend="numpy")
    artifact = etl.compile(lowered)
    return etl.load(artifact)


def run(exe, *args):
    out = []
    for spec, a in zip(exe.signature.input_specs, args):
        if isinstance(a, np.ndarray) and a.dtype != spec.dtype:
            a = a.astype(spec.dtype)
        out.append(a)
    return etl.run(exe, *out)


def ref_sparse(dense):
    coords = np.nonzero(dense)
    return np.stack(coords, axis=-1).astype(np.int64), dense[coords]


def ref_to_dense(idx, vals, shape, dtype):
    out = np.zeros(shape, dtype=dtype)
    np.add.at(out, tuple(idx.T), vals)
    return out


rng = np.random.default_rng(42)

# ---------------------------------------------------------------- roundtrip
print("== from_dense -> to_dense (unbatched, random) ==")
dense = (rng.random((4, 5)) > 0.5).astype(np.float32)
spec = etl.TensorSpec(dense.shape, etl.float32)


def fn(d):
    idx, vals = mk("sparse_from_dense", (d,), {"dense_shape": (4, 5), "dtype": "float32"})
    (out,) = mk("sparse_to_dense", (idx, vals), {"dense_shape": (4, 5), "dtype": "float32"})
    return out


check("roundtrip", np.array_equal(run(run_graph(fn, spec), dense).numpy(), dense))

# ------------------------------------------------- negate / add / multiply
print("== negate / add / multiply (unbatched) ==")
a = (rng.random((3, 4)) > 0.4).astype(np.float32)
b = (rng.random((3, 4)) > 0.6).astype(np.float32)
spec34 = etl.TensorSpec((3, 4), etl.float32)


def fn_neg(d):
    idx, vals = mk("sparse_from_dense", (d,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_negate", (idx, vals), {"dense_shape": (3, 4), "dtype": "float32"})


ridx, rvals = ref_sparse(a)
nidx, nvals = run(run_graph(fn_neg, spec34), a)
check("negate idx", np.array_equal(nidx.numpy(), ridx))
check("negate vals", np.array_equal(nvals.numpy(), -rvals))


def fn_add(x, y):
    ia, va = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ib, vb = mk("sparse_from_dense", (y,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_add", (ia, va, ib, vb), {"dense_shape": (3, 4), "dtype": "float32"})


out = run(run_graph(fn_add, spec34, spec34), a, b)
check("add dense", np.array_equal(ref_to_dense(out[0].numpy(), out[1].numpy(), (3, 4), np.float32), a + b))


def fn_mul(x, y):
    ia, va = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ib, vb = mk("sparse_from_dense", (y,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_multiply", (ia, va, ib, vb), {"dense_shape": (3, 4), "dtype": "float32"})


out = run(run_graph(fn_mul, spec34, spec34), a, b)
check("mul dense", np.array_equal(ref_to_dense(out[0].numpy(), out[1].numpy(), (3, 4), np.float32), a * b))

# ---------------------------------------------------------------- batched add
print("== add (batched B=3) ==")
ab = (rng.random((3, 2, 3)) > 0.5).astype(np.float32)
bb = (rng.random((3, 2, 3)) > 0.5).astype(np.float32)
spec_b = etl.TensorSpec(ab.shape, etl.float32)


def fn_add_b(x, y):
    ia, va = mk("sparse_from_dense", (x,), {"dense_shape": (2, 3), "dtype": "float32"})
    ib, vb = mk("sparse_from_dense", (y,), {"dense_shape": (2, 3), "dtype": "float32"})
    return mk("sparse_add", (ia, va, ib, vb), {"dense_shape": (2, 3), "dtype": "float32"})


o_idx, o_vals = run(run_graph(fn_add_b, spec_b, spec_b), ab, bb)
dense_out = np.zeros((3, 2, 3), np.float32)
for e in range(3):
    dense_out[e] = ref_to_dense(o_idx.numpy()[e], o_vals.numpy()[e], (2, 3), np.float32)
check("batched add", np.array_equal(dense_out, ab + bb))

# ---------------------------------------------------------- multiply_dense
print("== multiply_dense (unbatched + batched-broadcast) ==")
d2 = (rng.random((3, 4)) > 0.5).astype(np.float32)
fac = rng.random((3, 4)).astype(np.float32)
spec_f = etl.TensorSpec(fac.shape, etl.float32)


def fn_md(d, f):
    idx, vals = mk("sparse_from_dense", (d,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_multiply_dense", (idx, vals, f), {"dense_shape": (3, 4), "dtype": "float32"})


mi, mv = run(run_graph(fn_md, spec, spec_f), d2, fac)
check("multiply_dense", np.array_equal(mv.numpy(), d2[d2 != 0] * fac[d2 != 0]))

# batched sparse, unbatched dense (broadcast)
d3 = (rng.random((2, 3, 4)) > 0.5).astype(np.float32)
spec3 = etl.TensorSpec(d3.shape, etl.float32)


def fn_md_b(d, f):
    idx, vals = mk("sparse_from_dense", (d,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_multiply_dense", (idx, vals, f), {"dense_shape": (3, 4), "dtype": "float32"})


_, mv = run(run_graph(fn_md_b, spec3, spec_f), d3, fac)
ref = (d3 * fac)[d3 != 0]
check("multiply_dense batched", np.array_equal(mv.numpy(), ref))

# ---------------------------------------------------------------- reduce_sum
print("== reduce_sum variants ==")
d4 = (rng.random((2, 3)) > 0.3).astype(np.float32)
spec4 = etl.TensorSpec(d4.shape, etl.float32)

for axes, keepdims, label in [((0,), False, "ax0"), ((1,), False, "ax1"), ((0, 1), False, "all"),
                              ((0,), True, "ax0-kd"), ((1,), True, "ax1-kd")]:
    def fn_r(x, axes=axes, keepdims=keepdims):
        idx, vals = mk("sparse_from_dense", (x,), {"dense_shape": (2, 3), "dtype": "float32"})
        (out,) = mk("sparse_reduce_sum", (idx, vals),
                    {"dense_shape": (2, 3), "dtype": "float32", "axes": axes, "keepdims": keepdims})
        return out

    got = run(run_graph(fn_r, spec4), d4).numpy()
    exp = np.sum(d4, axis=tuple(axes), keepdims=keepdims)
    check(f"reduce {label}", np.array_equal(got, exp), f"got {got} exp {exp}")

# --------------------------------------------------------- transpose/reshape
print("== transpose / reshape ==")
d5 = (rng.random((2, 3, 4)) > 0.5).astype(np.float32)
spec5 = etl.TensorSpec(d5.shape, etl.float32)


def fn_t(x):
    idx, vals = mk("sparse_from_dense", (x,), {"dense_shape": (2, 3, 4), "dtype": "float32"})
    return mk("sparse_transpose", (idx, vals),
              {"dense_shape": (4, 2, 3), "dtype": "float32", "perm": (2, 0, 1)})


ti, tv = run(run_graph(fn_t, spec5), d5)
t_dense = ref_to_dense(ti.numpy(), tv.numpy(), (4, 2, 3), np.float32)
check("transpose", np.array_equal(t_dense, d5.transpose(2, 0, 1)))


def fn_rsp(x):
    idx, vals = mk("sparse_from_dense", (x,), {"dense_shape": (2, 3, 4), "dtype": "float32"})
    return mk("sparse_reshape", (idx, vals),
              {"dense_shape": (6, 4), "dtype": "float32", "old_shape": (2, 3, 4)})


ri, rv = run(run_graph(fn_rsp, spec5), d5)
r_dense = ref_to_dense(ri.numpy(), rv.numpy(), (6, 4), np.float32)
check("reshape", np.array_equal(r_dense, d5.reshape(6, 4)))

# ------------------------------------------------------------- csr / csc
print("== coo<->csr / coo<->csc roundtrips ==")
d6 = (rng.random((3, 4)) > 0.5).astype(np.float32)
spec6 = etl.TensorSpec(d6.shape, etl.float32)


def fn_csr(x):
    idx, vals = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ip, ci, cv = mk("sparse_coo_to_csr", (idx, vals), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_csr_to_coo", (ip, ci, cv), {"dense_shape": (3, 4), "dtype": "float32"})


i2, v2 = run(run_graph(fn_csr, spec6), d6)
check("csr roundtrip", np.array_equal(ref_to_dense(i2.numpy(), v2.numpy(), (3, 4), np.float32), d6))


def fn_csc(x):
    idx, vals = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ip, ci, cv = mk("sparse_coo_to_csc", (idx, vals), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_csc_to_coo", (ip, ci, cv), {"dense_shape": (3, 4), "dtype": "float32"})


i2, v2 = run(run_graph(fn_csc, spec6), d6)
check("csc roundtrip", np.array_equal(ref_to_dense(i2.numpy(), v2.numpy(), (3, 4), np.float32), d6))

# -------------------------------------------------------------------- dots
print("== sparse@dense / dense@sparse ==")
sm = (rng.random((3, 4)) > 0.5).astype(np.float32)
dn = rng.random((4, 2)).astype(np.float32)
spec_dn = etl.TensorSpec(dn.shape, etl.float32)


def fn_sd(s, dd):
    idx, vals = mk("sparse_from_dense", (s,), {"dense_shape": (3, 4), "dtype": "float32"})
    (out,) = mk("sparse_dot_dense", (idx, vals, dd), {"dense_shape": (3, 4), "dtype": "float32"})
    return out


check("sparse@dense", np.allclose(run(run_graph(fn_sd, spec, spec_dn), sm, dn).numpy(), sm @ dn, atol=1e-6))


def fn_ds(dd, s):
    idx, vals = mk("sparse_from_dense", (s,), {"dense_shape": (3, 4), "dtype": "float32"})
    (out,) = mk("dense_dot_sparse", (dd, idx, vals), {"dense_shape": (3, 4), "dtype": "float32"})
    return out


check("dense@sparse", np.allclose(run(run_graph(fn_ds, spec_dn, spec), dn, sm).numpy(), dn @ sm, atol=1e-6))

# batched dots
smb = (rng.random((2, 3, 4)) > 0.5).astype(np.float32)
dnb = rng.random((2, 4, 2)).astype(np.float32)
spec_dnb = etl.TensorSpec(dnb.shape, etl.float32)
spec_smb = etl.TensorSpec(smb.shape, etl.float32)


def fn_sd_b(s, dd):
    idx, vals = mk("sparse_from_dense", (s,), {"dense_shape": (3, 4), "dtype": "float32"})
    (out,) = mk("sparse_dot_dense", (idx, vals, dd), {"dense_shape": (3, 4), "dtype": "float32"})
    return out


check("batched sparse@dense", np.allclose(run(run_graph(fn_sd_b, spec_smb, spec_dnb), smb, dnb).numpy(),
                                           np.einsum("bij,bjk->bik", smb, dnb), atol=1e-6))

# ----------------------------------------------------------- error paths
print("== canonical-form violation errors ==")


def expect_err(fn, specs_and_args, label):
    specs, args = specs_and_args
    try:
        run(run_graph(fn, *specs), *args)
        check(label, False, "no error raised")
    except core.ETLError as e:
        check(label, "kernel for op" in str(e), str(e)[:90])


bad_idx = np.array([[1, 0], [0, 1]], np.int64)  # unsorted (0,1) after (1,0)
bad_vals = np.array([1.0, 2.0], np.float32)
spec_idx = etl.TensorSpec((2, 2), etl.int64)
spec_vals = etl.TensorSpec((2,), etl.float32)


def fn_to_dense(i, v):
    (out,) = mk("sparse_to_dense", (i, v), {"dense_shape": (3, 3), "dtype": "float32"})
    return out


expect_err(fn_to_dense, ([spec_idx, spec_vals], [bad_idx, bad_vals]), label="unsorted rejected")

oor_idx = np.array([[0, 0], [5, 1]], np.int64)  # row 5 out of range
expect_err(fn_to_dense, ([spec_idx, spec_vals], [oor_idx, bad_vals]), label="out-of-range rejected")

dup_idx = np.array([[0, 0], [0, 0]], np.int64)
expect_err(fn_to_dense, ([spec_idx, spec_vals], [dup_idx, bad_vals]), label="duplicate rejected")

spec_ip = etl.TensorSpec((5,), etl.int64)
spec_ci = etl.TensorSpec((3,), etl.int64)
bad_ip = np.array([0, 1, 1, 1, 0], np.int64)  # non-monotone


def fn_csr2coo(ip, ci, v):
    return mk("sparse_csr_to_coo", (ip, ci, v), {"dense_shape": (4, 4), "dtype": "float32"})


expect_err(fn_csr2coo, ([spec_ip, spec_ci, spec_vals], [bad_ip, np.array([0, 1, 2], np.int64), bad_vals]),
           label="non-monotone indptr rejected")

print(f"===== {PASS} passed, {FAIL} failed =====")
sys.exit(1 if FAIL else 0)
