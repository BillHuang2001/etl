"""Backend tests for ``etl.sparse``: the numpy interpreter path (capabilities +
explicit-pipeline parity) and the compiler-backend deferrals (all 16 sparse
ops are numpy-backend-only in v1 — every compiler adapter rejects them at
LOWER time with the capability-drift ``BackendError``, and the StableHLO
exporter names them).

Contract under test: ``../etl/sparse/CONTEXT.md`` (Known issues) and
``../etl/backends/CONTEXT.md``. The capability drift check runs in
``CompilerBackend.lower`` (``../etl/backends/compiler.py``) BEFORE any
compiler is invoked, so ``etl.lower(graph, backend="iree"|"tvm")`` rejects a
sparse graph at lower time; ``lower`` never compiles, so a dense-only graph
lowers to a ``LoweredProgram`` on ``"iree"`` without any compiler run.

NOTE on the conftest helpers: ``tests.sparse.conftest.csr_spec`` /
``csc_spec`` currently omit the ``format=`` keyword (so they raise
``ShapeError`` — a test-helper bug, not an etl bug). This file therefore
defines corrected ``csr_spec`` / ``csc_spec`` helpers at module level;
``coo_spec`` and the concrete examples from conftest are used as-is.
"""

import re

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse

from tests.sparse.conftest import (
    coo_example,
    coo_spec,
    csr_example,
    csc_example,
    dense_example,
    materialize,
    run_graph,
)

# ---------------------------------------------------------------------------
# Per-file helpers (see the module docstring re: conftest's csr/csc specs).
# ---------------------------------------------------------------------------


def csr_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csr_example()` (indptr static (4,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((4,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csr",
    )


def csc_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csc_example()` (indptr static (5,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((5,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csc",
    )


#: The 16 numpy-backend-only sparse ops (exact names, per
#: ``etl.backends.stablehlo.ops.DEFERRED_OPS``).
SPARSE_OPS = (
    "sparse_from_dense",
    "sparse_to_dense",
    "sparse_coo_to_csr",
    "sparse_csr_to_coo",
    "sparse_coo_to_csc",
    "sparse_csc_to_coo",
    "sparse_negate",
    "sparse_add",
    "sparse_multiply",
    "sparse_multiply_dense",
    "sparse_reduce_sum",
    "sparse_transpose",
    "sparse_reshape",
    "sparse_concatenate",
    "sparse_dot_dense",
    "dense_dot_sparse",
)


def _graph_for(op_name):
    """A tiny graph whose entry block contains exactly the named sparse op."""
    if op_name == "sparse_from_dense":
        return etl.trace(
            lambda d: sparse.from_dense(d), etl.TensorSpec((3, 4), etl.float32)
        )
    if op_name == "sparse_to_dense":
        return etl.trace(lambda x: sparse.to_dense(x), coo_spec())
    if op_name == "sparse_coo_to_csr":
        return etl.trace(lambda x: sparse.to_csr(x), coo_spec())
    if op_name == "sparse_csr_to_coo":
        return etl.trace(lambda x: sparse.to_coo(x), csr_spec())
    if op_name == "sparse_coo_to_csc":
        return etl.trace(lambda x: sparse.to_csc(x), coo_spec())
    if op_name == "sparse_csc_to_coo":
        return etl.trace(lambda x: sparse.to_coo(x), csc_spec())
    if op_name == "sparse_negate":
        return etl.trace(lambda x: sparse.negate(x), coo_spec())
    if op_name == "sparse_add":
        return etl.trace(lambda a, b: sparse.add(a, b), coo_spec(), coo_spec())
    if op_name == "sparse_multiply":
        return etl.trace(lambda a, b: sparse.multiply(a, b), coo_spec(), coo_spec())
    if op_name == "sparse_multiply_dense":
        return etl.trace(
            lambda a, d: sparse.multiply_dense(a, d),
            coo_spec(),
            etl.TensorSpec((3, 4), etl.float32),
        )
    if op_name == "sparse_reduce_sum":
        return etl.trace(lambda x: sparse.sum(x, axes=0), coo_spec())
    if op_name == "sparse_transpose":
        return etl.trace(lambda x: sparse.transpose(x), coo_spec())
    if op_name == "sparse_reshape":
        return etl.trace(lambda x: sparse.reshape(x, (12,)), coo_spec())
    if op_name == "sparse_concatenate":
        return etl.trace(
            lambda a, b: sparse.concatenate([a, b], axis=0), coo_spec(), coo_spec()
        )
    if op_name == "sparse_dot_dense":
        return etl.trace(
            lambda x, d: sparse.matmul(x, d),
            coo_spec(),
            etl.TensorSpec((4, 5), etl.float32),
        )
    if op_name == "dense_dot_sparse":
        return etl.trace(
            lambda d, x: sparse.matmul(d, x),
            etl.TensorSpec((5, 3), etl.float32),
            coo_spec(),
        )
    raise ValueError(f"unknown sparse op {op_name!r}")


@pytest.fixture(scope="module")
def sparse_graphs():
    """All 16 single-op sparse graphs, built once per module."""
    return {name: _graph_for(name) for name in SPARSE_OPS}


# ---------------------------------------------------------------------------
# 1. numpy backend: capability + explicit-pipeline parity
# ---------------------------------------------------------------------------


def test_numpy_backend_declares_sparse_capability():
    backend = etl.backends.get("numpy")
    assert backend.name == "numpy"
    assert backend.capabilities.sparse_ops is True


def test_numpy_backend_dtype_capabilities_cover_sparse_adjacent():
    caps = etl.backends.get("numpy").capabilities
    assert caps.supports_dtype(np.dtype("float32"))
    assert caps.supports_dtype(np.dtype("int64"))  # indices/indptr dtype


def test_explicit_pipeline_parity_add_coo_csr():
    d = dense_example()
    graph = etl.trace(
        lambda a, b: sparse.add(a, b), coo_spec(), csr_spec()
    )
    out = run_graph(graph, coo_example(), csr_example())
    np.testing.assert_allclose(materialize(out), d + d, rtol=1e-6, atol=1e-6)


def test_explicit_pipeline_parity_matmul_sparse_dot_dense():
    d = dense_example()
    dense = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    graph = etl.trace(
        lambda x, y: sparse.matmul(x, y),
        coo_spec(),
        etl.TensorSpec((4, 5), etl.float32),
    )
    out = run_graph(graph, coo_example(), dense)
    np.testing.assert_allclose(materialize(out), d @ dense, rtol=1e-6, atol=1e-6)


def test_explicit_pipeline_parity_reduce_sum_keepdims():
    d = dense_example()
    graph = etl.trace(
        lambda x: sparse.reduce_sum(x, axes=(0,), keepdims=True), coo_spec()
    )
    out = run_graph(graph, coo_example())
    np.testing.assert_allclose(
        materialize(out), d.sum(axis=0, keepdims=True), rtol=1e-6, atol=1e-6
    )


def test_explicit_pipeline_parity_to_dense_csr():
    d = dense_example()
    graph = etl.trace(lambda x: sparse.to_dense(x), csr_spec())
    out = run_graph(graph, csr_example())
    np.testing.assert_allclose(materialize(out), d, rtol=1e-6, atol=1e-6)


def test_lower_defaults_to_numpy():
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    lowered = etl.lower(graph)  # backend=None -> numpy
    assert lowered.backend == "numpy"


# ---------------------------------------------------------------------------
# 2. Compiler-backend deferrals: all 16 sparse ops are numpy-backend-only
# ---------------------------------------------------------------------------

# The drift rejection is a LOWER-time capability check (no compiler is
# invoked). All 16 ops x iree stay within the time budget; tvm is exercised
# on a representative subset (the drift path is shared — CompilerBackend.lower
# — so per-op coverage on one adapter plus subset coverage on the other is
# sufficient; xla raises a plugin-required error instead and is not used).
_TVM_REPRESENTATIVES = ("sparse_add", "sparse_to_dense", "sparse_dot_dense")


@pytest.mark.parametrize("op_name", SPARSE_OPS)
def test_sparse_op_defers_on_iree(sparse_graphs, op_name):
    graph = sparse_graphs[op_name]
    with pytest.raises(core.BackendError) as excinfo:
        etl.lower(graph, backend="iree")
    msg = str(excinfo.value)
    assert re.search("capability drift", msg)
    assert op_name in msg
    assert "etl.sparse.to_dense" in msg


@pytest.mark.parametrize("op_name", _TVM_REPRESENTATIVES)
def test_sparse_op_defers_on_tvm(sparse_graphs, op_name):
    graph = sparse_graphs[op_name]
    with pytest.raises(core.BackendError) as excinfo:
        etl.lower(graph, backend="tvm")
    msg = str(excinfo.value)
    # Env-dependent adapter-missing error (tvm/jaxlib not installed) skips,
    # mirroring the pattern in tests/backends/test_external_call_iree.py —
    # the capability-drift contract is what's under test.
    if "capability drift" not in msg:
        pytest.skip(f"tvm adapter not available in this env: {msg}")
    assert re.search("capability drift", msg)
    assert op_name in msg
    assert "etl.sparse.to_dense" in msg


@pytest.mark.parametrize("op_name", SPARSE_OPS)
def test_sparse_op_defers_in_stablehlo_export(sparse_graphs, op_name):
    graph = sparse_graphs[op_name]
    with pytest.raises(core.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph)
    assert op_name in str(excinfo.value)


def test_dense_graph_lowers_on_compiler_backend_without_compiler(sparse_graphs):
    # Positive control: a dense-only graph passes the capability pre-check —
    # lower never invokes a compiler, so it succeeds even though no compiler
    # has run (and would also succeed if the compiler packages were absent).
    del sparse_graphs

    def f(x):
        return etl.add(x, x)

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    lowered = etl.lower(graph, backend="iree")
    assert isinstance(lowered, etl.backends.LoweredProgram)
    assert lowered.backend == "iree"
