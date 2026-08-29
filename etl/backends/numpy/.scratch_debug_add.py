"""Debug the sparse_add duplicate-rows error (scratch, not committed)."""
import numpy as np
import etl
from etl import core
from etl.trace import current_builder

rng = np.random.default_rng(42)
a = (rng.random((3, 4)) > 0.4).astype(np.float32)
b = (rng.random((3, 4)) > 0.6).astype(np.float32)
print("a:\n", a)
print("b:\n", b)


def sym(value):
    return core.SymbolicTensor(value=value, dtype=value.type.dtype, shape=value.type.shape, location=None)


def mk(name, operands, attributes):
    op = current_builder().create(name, operands=tuple(o.value for o in operands), attributes=attributes)
    return tuple(sym(v) for v in op.results)


def fn_add(x, y):
    ia, va = mk("sparse_from_dense", (x,), {"dense_shape": (3, 4), "dtype": "float32"})
    ib, vb = mk("sparse_from_dense", (y,), {"dense_shape": (3, 4), "dtype": "float32"})
    return mk("sparse_add", (ia, va, ib, vb), {"dense_shape": (3, 4), "dtype": "float32"})


s = etl.TensorSpec((3, 4), etl.float32)
graph = etl.trace(fn_add, s, s)
print("graph input_specs type:", type(graph.input_specs))
leaves = etl.tree_leaves(graph.input_specs)
print("num input leaves:", len(leaves))
print("--- run (distinct spec instances) ---")
s2 = etl.TensorSpec((3, 4), etl.float32)
graph2 = etl.trace(fn_add, s, s2)
print("num input leaves graph2:", len(etl.tree_leaves(graph2.input_specs)))

exe = etl.load(etl.compile(etl.lower(graph2, backend="numpy")))
out = etl.run(exe, a, b)
print("add out idx:\n", out[0].numpy())
print("add out vals:", out[1].numpy())
ref = a + b
print("ref dense:\n", ref)
