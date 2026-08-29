"""Shared construction helpers for the etl.sparse test suite.

The suite mirrors the `etl.sparse` contract (`../../etl/sparse/CONTEXT.md` —
sibling, read-only). Helpers here are minimal and stable; per-file extras
belong in the test modules themselves. Import package-qualified:
`from tests.sparse.conftest import ...` (never `from conftest import ...`).
"""

import numpy as np

import etl
from etl import sparse

# ---------------------------------------------------------------------------
# Canonical example: the SAME 3x4 dense matrix in all three formats.
# ---------------------------------------------------------------------------


def dense_example():
    """The dense reference of the canonical examples (float32, 3x4)."""
    return np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 2.0, 0.0],
         [3.0, 0.0, 0.0, 4.0]],
        dtype=np.float32,
    )


def coo_example():
    """A small canonical COO (numpy leaves, float32 values)."""
    return sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (3, 4),
    )


def csr_example():
    """The same data in canonical CSR form."""
    return sparse.csr(
        np.array([0, 1, 2, 4], dtype=np.int64),
        np.array([1, 2, 0, 3], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (3, 4),
    )


def csc_example():
    """The same data in canonical CSC form (column-major ordering)."""
    return sparse.csc(
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        np.array([2, 0, 1, 2], dtype=np.int64),
        np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32),
        (3, 4),
    )


# ---------------------------------------------------------------------------
# Matching SparseTensorSpecs (unbatched, trace-time inputs).
# ---------------------------------------------------------------------------


def coo_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `coo_example()`."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
    )


def csr_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csr_example()` (indptr static (4,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((4,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
    )


def csc_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csc_example()` (indptr static (5,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((5,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
    )


# ---------------------------------------------------------------------------
# Execution helpers.
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Explicit staging: graph -> lower -> compile -> load -> run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def materialize(value):
    """Sparse -> `.to_dense()` ndarray; `etl.Tensor` -> `.numpy()`; recurse
    into tuples/lists. Lets tests compare graph outputs against pure-numpy
    references regardless of the result's phase."""
    if sparse.is_sparse(value):
        return value.to_dense()
    if isinstance(value, etl.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return type(value)(materialize(v) for v in value)
    return value


def eval_dense(fn, *args):
    """`etl.evaluate(fn, *args)` (concrete sparse args derive
    SparseTensorSpecs automatically), then `materialize` the result."""
    return materialize(etl.evaluate(fn, *args))


def batched_coo_example():
    """A 2-element batched COO input for vectorize/vmap runs.

    Leaves are (B, nnz, ndim) / (B, nnz) with dense_shape UNCHANGED (3, 4) —
    the input-side batched convention (batch lives in the flat leaves only).
    Built via `SparseTensor.from_parts`: the validating constructors reject
    batched leaves (rank-2 canonical form only) in v1.
    """
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return sparse.SparseTensor.from_parts(
        etl.core.from_numpy(np.stack([idx, idx], axis=0)),
        etl.core.from_numpy(np.stack([vals, vals * 2.0], axis=0)),
        dense_shape=(3, 4),
        format="coo",
    )
