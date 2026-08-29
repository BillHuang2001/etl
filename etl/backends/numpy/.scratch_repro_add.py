"""Reproduce the exact spot-check failure (scratch, not committed)."""
import numpy as np
import etl
from etl import core
from etl.trace import current_builder

rng = np.random.default_rng(42)
a = (rng.random((3, 4)) > 0.4).astype(np.float32)
b = (rng.random((3, 4)) > 0.6).astype(np.float32)


def sym(value):
    return core.SymbolicTensor(value=value, dtype=value.type.dtype, shape=value.type.shape, location=None)


def mk(name, operands, attributes):
    op = current_builder().create(name, operands=tuple(o.value for o in operands), attributes=attributes)
    return tuple(sym(v) for v in op.results)


def run_graph(fn, *specs):
    graph = etl.trace(fn, *specs)
    lowered = etl.lower(graph, backend="numpy")
    return etl.load(etl.compile(lowered))


def run(exe, *args):
    out = []
    for spec, x in zip(exe.signature.input_specs, args):
        if isinstance(x, np.ndarray) and x.dtype != spec.dtype:
            x = x.astype(spec.dtype)
        out.append(x)
    return etl.run(exe, *out)


def fn_add(x, y):
    ia, va = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ib, vb = mk("sparse_from_dense", (y,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_add", (ia, va, ib, vb), {"dense_shape": (3, 4), "dtype": "float32"})


s = etl.TensorSpec((3, 4), etl.float32)
exe = run_graph(fn_add, s, s)
print("signature.input_specs type:", type(exe.signature.input_specs), "len:", len(exe.signature.input_specs))
print("zip pairs:", [(i, type(x).__name__) for i, x in enumerate(zip(exe.signature.input_specs, (a, b)))])
out = run(exe, a, b)
print("ADDED OK — out idx:\n", out[0].numpy())
print("out vals:", out[1].numpy())
