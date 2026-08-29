"""Computation-op tests for ``etl.sparse``: graph evaluation vs pure-numpy
references of the DENSE matrices, mixed-format auto-conversion, from_dense /
to_dense roundtrips, empty-nnz edge cases, trace-time static error checks, and
IR-contract spot checks (exact emitted op names).

Contract under test: ``../etl/sparse/CONTEXT.md`` (sibling, read-only). Every
computation op is evaluated through the explicit pipeline
(``etl.evaluate`` — concrete sparse args derive ``SparseTensorSpec``s
automatically) and compared against the dense reference with a tight
tolerance. Static violations are asserted at TRACE time with the exact
wording from ``../etl/sparse/ops.py``.

NOTE on the conftest helpers: ``tests.sparse.conftest.csr_spec`` /
``csc_spec`` currently omit the ``format=`` keyword (so they raise
``ShapeError`` — a test-helper bug, not an etl bug). This file therefore
defines corrected ``csr_spec`` / ``csc_spec`` helpers at module level;
``coo_spec`` and the concrete examples from conftest are used as-is.
"""

import numpy as np
import pytest

import etl
from etl import sparse
from etl import core

from tests.sparse.conftest import (
    csc_example,
    coo_example,
    coo_spec,
    csr_example,
    dense_example,
    eval_dense,
    materialize,
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


def empty_coo(shape=(3, 4), dtype=np.float32):
    """An nnz=0 COO with the given dense shape (canonical by vacuity)."""
    return sparse.coo(
        np.zeros((0, len(shape)), dtype=np.int64),
        np.zeros((0,), dtype=dtype),
        shape,
    )


def _ops_of(graph):
    """The IR op names of a traced graph's entry block (terminator excluded)."""
    return [
        op.name
        for op in graph.module.functions[0].entry_block.ops
        if not op.is_terminator
    ]


# ---------------------------------------------------------------------------
# 1. add / subtract (union merge, overlapping entries summed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("coo", "csr"),
        ("csr", "csc"),
        ("coo", "coo"),
        ("csr", "csr"),
        ("csc", "csc"),
    ],
)
def test_add_union_merge_matches_dense_sum(a, b):
    d = dense_example()
    left = {"coo": coo_example, "csr": csr_example, "csc": csc_example}[a]()
    right = {"coo": coo_example, "csr": csr_example, "csc": csc_example}[b]()
    got = eval_dense(lambda x=left, y=right: sparse.add(x, y), left, right)
    np.testing.assert_allclose(got, d + d, rtol=1e-6, atol=1e-6)


def test_add_overlapping_entries_are_summed():
    # Both operands store a value at (2, 3): 4.0 + 4.0 = 8.0.
    d = dense_example()
    got = eval_dense(lambda a, b: sparse.add(a, b), coo_example(), csr_example())
    expected = d + d
    assert got[2, 3] == 8.0
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_add_result_is_coo():
    result = etl.evaluate(lambda a, b: sparse.add(a, b), coo_example(), csr_example())
    assert sparse.is_sparse(result)
    assert result.format == "coo"


def test_subtract_is_add_of_negated_second_operand():
    d = dense_example()
    got = eval_dense(lambda a, b: sparse.subtract(a, b), coo_example(), csr_example())
    np.testing.assert_allclose(got, d - d, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. multiply (intersection merge)
# ---------------------------------------------------------------------------


def test_multiply_overlapping_supports():
    d = dense_example()
    got = eval_dense(lambda a, b: sparse.multiply(a, b), coo_example(), csr_example())
    np.testing.assert_allclose(got, d * d, rtol=1e-6, atol=1e-6)


def test_multiply_disjoint_supports_is_empty():
    # Supports {(0,1),(1,2),(2,0),(2,3)} vs {(0,0),(1,1)}: no overlap.
    d = dense_example()
    disjoint = sparse.coo(
        np.array([[0, 0], [1, 1]], dtype=np.int64),
        np.array([7.0, 8.0], dtype=np.float32),
        (3, 4),
    )
    result = etl.evaluate(lambda a, b: sparse.multiply(a, b), coo_example(), disjoint)
    assert sparse.is_sparse(result)
    assert result.indices.shape[0] == 0  # nnz-0 result
    np.testing.assert_allclose(
        materialize(result), np.zeros_like(d), rtol=1e-6, atol=1e-6
    )


def test_multiply_result_is_coo():
    result = etl.evaluate(
        lambda a, b: sparse.multiply(a, b), coo_example(), csr_example()
    )
    assert sparse.is_sparse(result)
    assert result.format == "coo"


# ---------------------------------------------------------------------------
# 3. multiply_dense (sparse structure preserved, values scaled)
# ---------------------------------------------------------------------------


def test_multiply_dense_scales_values_at_sparse_positions():
    d = dense_example()
    dense = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    got = eval_dense(lambda a, x: sparse.multiply_dense(a, x), coo_example(), dense)
    np.testing.assert_allclose(got, d * dense, rtol=1e-6, atol=1e-6)


def test_multiply_dense_traces_with_dense_tensor_spec():
    graph = etl.trace(
        lambda a, x: sparse.multiply_dense(a, x),
        coo_spec(),
        etl.TensorSpec((3, 4), etl.float32),
    )
    assert "sparse_multiply_dense" in _ops_of(graph)


# ---------------------------------------------------------------------------
# 4. negate
# ---------------------------------------------------------------------------


def test_negate():
    d = dense_example()
    got = eval_dense(lambda x: sparse.negate(x), coo_example())
    np.testing.assert_allclose(got, -d, rtol=1e-6, atol=1e-6)


def test_negate_keeps_sparsity_structure():
    d = dense_example()
    result = etl.evaluate(lambda x: sparse.negate(x), coo_example())
    assert sparse.is_sparse(result)
    assert result.format == "coo"
    assert result.indices.shape[0] == 4  # same nnz as the input
    np.testing.assert_allclose(materialize(result), -d, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. sum / reduce_sum -> DENSE result
# ---------------------------------------------------------------------------


def test_sum_result_is_dense():
    result = etl.evaluate(lambda x: sparse.sum(x), coo_example())
    assert isinstance(result, etl.Tensor)


@pytest.mark.parametrize(
    "axes,keepdims",
    [
        (None, False),
        (0, False),
        (1, False),
        (-1, False),
        (0, True),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_sum_axes_variants(axes, keepdims):
    d = dense_example()
    got = eval_dense(lambda x: sparse.sum(x, axes=axes, keepdims=keepdims), coo_example())
    np.testing.assert_allclose(got, d.sum(axis=axes, keepdims=keepdims), rtol=1e-6, atol=1e-6)


def test_reduce_sum_is_an_alias_of_sum():
    d = dense_example()
    got = eval_dense(lambda x: sparse.reduce_sum(x, axes=1), coo_example())
    np.testing.assert_allclose(got, d.sum(axis=1), rtol=1e-6, atol=1e-6)


def test_sum_bigger_shape():
    # A denser (8, 16) case: sums equal the dense reference.
    rng = np.random.default_rng(0)
    d = rng.standard_normal((8, 16)).astype(np.float32)
    d[np.abs(d) < 0.5] = 0.0  # sparsify
    idx = np.stack(np.nonzero(d), axis=1).astype(np.int64)
    vals = d[np.nonzero(d)]
    x = sparse.coo(idx, vals, (8, 16))
    got = eval_dense(lambda a: sparse.sum(a, axes=0), x)
    np.testing.assert_allclose(got, d.sum(axis=0), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. transpose
# ---------------------------------------------------------------------------


def test_transpose_default_reverses_axes():
    d = dense_example()
    result = etl.evaluate(lambda x: sparse.transpose(x), coo_example())
    assert sparse.is_sparse(result)
    assert result.format == "coo"
    assert result.dense_shape == (4, 3)
    np.testing.assert_allclose(materialize(result), d.T, rtol=1e-6, atol=1e-6)


def test_transpose_explicit_perm():
    d = dense_example()
    got = eval_dense(lambda x: sparse.transpose(x, (1, 0)), coo_example())
    np.testing.assert_allclose(got, d.T, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. reshape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("new_shape", [(12,), (2, 6), (6, 2)])
def test_reshape_element_count_equal(new_shape):
    d = dense_example()
    result = etl.evaluate(lambda x: sparse.reshape(x, new_shape), coo_example())
    assert sparse.is_sparse(result)
    assert result.format == "coo"
    assert result.dense_shape == tuple(new_shape)
    np.testing.assert_allclose(
        materialize(result), d.reshape(new_shape), rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 8. concatenate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis,n_operands", [(0, 2), (0, 3), (1, 2), (1, 3)])
def test_concatenate_axis_and_operand_count(axis, n_operands):
    d = dense_example()
    refs = [d] * n_operands
    args = [coo_example() if i % 2 == 0 else csr_example() for i in range(n_operands)]
    got = eval_dense(
        lambda *ops, ax=axis: sparse.concatenate(list(ops), axis=ax), *args
    )
    np.testing.assert_allclose(
        got, np.concatenate(refs, axis=axis), rtol=1e-6, atol=1e-6
    )


def test_concatenate_operand_extents_and_result_shape():
    # Two (3, 4) operands along axis 0: result dense_shape (6, 4) with
    # per-operand extents (3, 3).
    graph = etl.trace(
        lambda a, b: sparse.concatenate([a, b], axis=0), coo_spec(), csr_spec()
    )
    op = next(
        o for o in graph.module.functions[0].entry_block.ops if o.name == "sparse_concatenate"
    )
    assert op.attributes["dense_shape"] == (6, 4)
    assert op.attributes["operand_extents"] == (3, 3)
    assert op.attributes["axis"] == 0

    graph = etl.trace(
        lambda a, b: sparse.concatenate([a, b], axis=1), coo_spec(), csr_spec()
    )
    op = next(
        o for o in graph.module.functions[0].entry_block.ops if o.name == "sparse_concatenate"
    )
    assert op.attributes["dense_shape"] == (3, 8)
    assert op.attributes["operand_extents"] == (4, 4)


def test_concatenate_result_dense_shape_is_extent_sum():
    # Three (3, 4) COOs along axis 0.
    graph = etl.trace(
        lambda a, b, c: sparse.concatenate([a, b, c], axis=0),
        coo_spec(),
        coo_spec(),
        coo_spec(),
    )
    op = next(
        o for o in graph.module.functions[0].entry_block.ops if o.name == "sparse_concatenate"
    )
    assert op.attributes["dense_shape"] == (9, 4)
    assert op.attributes["operand_extents"] == (3, 3, 3)


# ---------------------------------------------------------------------------
# 9. matmul
# ---------------------------------------------------------------------------


def test_matmul_sparse_dot_dense():
    d = dense_example()
    dense = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    result = etl.evaluate(lambda x, y: sparse.matmul(x, y), coo_example(), dense)
    assert isinstance(result, etl.Tensor)  # DENSE result
    np.testing.assert_allclose(result.numpy(), d @ dense, rtol=1e-6, atol=1e-6)


def test_matmul_dense_dot_sparse():
    d = dense_example()
    dense = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    result = etl.evaluate(lambda x, y: sparse.matmul(x, y), dense, coo_example())
    assert isinstance(result, etl.Tensor)  # DENSE result
    np.testing.assert_allclose(result.numpy(), dense @ d, rtol=1e-6, atol=1e-6)


def test_matmul_dense_dot_dense_routes_to_etl_ops_dot():
    graph = etl.trace(
        lambda a, b: sparse.matmul(a, b),
        etl.TensorSpec((3, 4), etl.float32),
        etl.TensorSpec((4, 5), etl.float32),
    )
    ops = _ops_of(graph)
    assert "dot" in ops
    assert not any(op.startswith("sparse") or op == "dense_dot_sparse" for op in ops)


def test_matmul_dense_dot_dense_evaluates_via_dot():
    a = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    b = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    got = eval_dense(lambda x, y: sparse.matmul(x, y), a, b)
    np.testing.assert_allclose(got, a @ b, rtol=1e-6, atol=1e-6)


def test_matmul_batched_dense_traces_with_static_tolerance():
    # The frontend static check tolerates a dense (B, K, N) operand (trailing
    # dims match); the graph builds with a sparse_dot_dense op. NOTE: running
    # it with an UNBATCHED sparse operand is outside the numpy kernel's
    # contract (the dense batch must come from the sparse operand's batch
    # dims) and fails with an explicit ShapeError at run time — not exercised
    # here.
    graph = etl.trace(
        lambda x, y: sparse.matmul(x, y),
        coo_spec(),
        etl.TensorSpec((2, 4, 5), etl.float32),
    )
    assert "sparse_dot_dense" in _ops_of(graph)


# ---------------------------------------------------------------------------
# 10. Mixed-format auto-conversion (COO is the computation format)
# ---------------------------------------------------------------------------


def test_computation_ops_auto_convert_csr_csc_inputs():
    graph = etl.trace(
        lambda a, b: sparse.add(a, b), csr_spec(), csc_spec()
    )
    ops = _ops_of(graph)
    assert ops == ["sparse_csr_to_coo", "sparse_csc_to_coo", "sparse_add"]

    graph = etl.trace(
        lambda a, b: sparse.multiply(a, b), csr_spec(), csr_spec()
    )
    ops = _ops_of(graph)
    assert ops == ["sparse_csr_to_coo", "sparse_csr_to_coo", "sparse_multiply"]

    graph = etl.trace(
        lambda x, y: sparse.matmul(x, y), csr_spec(), etl.TensorSpec((4, 5), etl.float32)
    )
    assert _ops_of(graph) == ["sparse_csr_to_coo", "sparse_dot_dense"]


def test_to_dense_auto_converts_csr_csc_but_not_coo():
    assert _ops_of(etl.trace(lambda x: sparse.to_dense(x), csr_spec())) == [
        "sparse_csr_to_coo",
        "sparse_to_dense",
    ]
    assert _ops_of(etl.trace(lambda x: sparse.to_dense(x), csc_spec())) == [
        "sparse_csc_to_coo",
        "sparse_to_dense",
    ]
    assert _ops_of(etl.trace(lambda x: sparse.to_dense(x), coo_spec())) == [
        "sparse_to_dense"
    ]


def test_to_csr_to_csc_emit_coo_conversions():
    assert _ops_of(etl.trace(lambda x: sparse.to_csr(x), coo_spec())) == [
        "sparse_coo_to_csr"
    ]
    assert _ops_of(etl.trace(lambda x: sparse.to_csc(x), coo_spec())) == [
        "sparse_coo_to_csc"
    ]


def test_to_csr_on_csr_is_identity():
    assert _ops_of(etl.trace(lambda x: sparse.to_csr(x), csr_spec())) == []


def test_to_coo_identity_on_coo():
    assert _ops_of(etl.trace(lambda x: sparse.to_coo(x), coo_spec())) == []


def test_csr_csc_inputs_auto_convert_in_negate_sum_transpose_reshape():
    assert _ops_of(etl.trace(lambda x: sparse.negate(x), csc_spec())) == [
        "sparse_csc_to_coo",
        "sparse_negate",
    ]
    assert _ops_of(etl.trace(lambda x: sparse.sum(x, axes=0), csr_spec())) == [
        "sparse_csr_to_coo",
        "sparse_reduce_sum",
    ]
    assert _ops_of(etl.trace(lambda x: sparse.transpose(x), csr_spec())) == [
        "sparse_csr_to_coo",
        "sparse_transpose",
    ]
    assert _ops_of(etl.trace(lambda x: sparse.reshape(x, (12,)), csc_spec())) == [
        "sparse_csc_to_coo",
        "sparse_reshape",
    ]
    assert _ops_of(
        etl.trace(
            lambda a, x: sparse.multiply_dense(a, x),
            csr_spec(),
            etl.TensorSpec((3, 4), etl.float32),
        )
    ) == ["sparse_csr_to_coo", "sparse_multiply_dense"]


def test_csr_csc_evaluation_roundtrip():
    # Auto-converted CSR/CSC inputs evaluate correctly end-to-end.
    d = dense_example()
    got = eval_dense(lambda x: sparse.to_dense(x), csr_example())
    np.testing.assert_allclose(got, d, rtol=1e-6, atol=1e-6)
    got = eval_dense(lambda x: sparse.to_dense(x), csc_example())
    np.testing.assert_allclose(got, d, rtol=1e-6, atol=1e-6)


def test_rank2_requirement_for_conversion_to_dense():
    # A SYMBOLIC rank-3 CSR (via from_parts inside a trace) fed to to_dense
    # must raise at trace time — the CSR->COO conversion is rank-2 only.

    def f(indptr, indices, values):
        x = sparse.csr(indptr, indices, values, (2, 3, 4))
        return sparse.to_dense(x)

    with pytest.raises(core.ShapeError, match="requires a rank-2 sparse tensor"):
        etl.trace(
            f,
            etl.TensorSpec((7,), etl.int64),
            etl.TensorSpec((None,), etl.int64),
            etl.TensorSpec((None,), etl.float32),
        )


# ---------------------------------------------------------------------------
# 11. from_dense / to_dense roundtrips (concrete + symbolic)
# ---------------------------------------------------------------------------


def test_from_dense_concrete_roundtrip():
    d = dense_example()
    x = sparse.from_dense(d)
    assert sparse.is_sparse(x)
    assert x.format == "coo"
    assert x.dtype == np.dtype("float32")
    np.testing.assert_allclose(x.to_dense(), d, rtol=1e-6, atol=1e-6)


def test_from_dense_concrete_extracts_exact_nonzeros():
    d = dense_example()
    x = sparse.from_dense(d)
    np.testing.assert_array_equal(x.indices, np.array([[0, 1], [1, 2], [2, 0], [2, 3]]))
    np.testing.assert_array_equal(x.values, np.array([1.0, 2.0, 3.0, 4.0]))


def test_from_dense_symbolic_roundtrip():
    # In-graph creator: sparse_from_dense op emitted; evaluated result equals
    # the input dense.
    d = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    graph = etl.trace(
        lambda x: sparse.to_dense(sparse.from_dense(x)), etl.TensorSpec((3, 4), etl.float32)
    )
    assert _ops_of(graph) == ["sparse_from_dense", "sparse_to_dense"]
    got = eval_dense(lambda x: sparse.to_dense(sparse.from_dense(x)), d)
    np.testing.assert_allclose(got, d, rtol=1e-6, atol=1e-6)


def test_from_dense_dtype_preserved():
    d = dense_example().astype(np.float64)
    x = sparse.from_dense(d)
    assert x.dtype == np.dtype("float64")
    np.testing.assert_allclose(x.to_dense(), d, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. Empty-nnz and edge cases
# ---------------------------------------------------------------------------


def test_empty_nnz_through_computation_ops():
    d = dense_example()
    e = empty_coo()
    np.testing.assert_allclose(
        eval_dense(lambda a, b: sparse.add(a, b), e, coo_example()), d, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eval_dense(lambda x: sparse.negate(x), e), np.zeros_like(d), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eval_dense(lambda x: sparse.sum(x), e), 0.0, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eval_dense(lambda x: sparse.transpose(x), e), np.zeros((4, 3)), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eval_dense(lambda x: sparse.reshape(x, (12,)), e), np.zeros((12,)), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        eval_dense(lambda x: sparse.to_dense(x), e), np.zeros_like(d), rtol=1e-6, atol=1e-6
    )


def test_empty_intersection_multiply_is_nnz_zero():
    e = empty_coo()
    result = etl.evaluate(lambda a, b: sparse.multiply(a, b), e, coo_example())
    assert sparse.is_sparse(result)
    assert result.indices.shape[0] == 0
    np.testing.assert_allclose(
        materialize(result), np.zeros_like(dense_example()), rtol=1e-6, atol=1e-6
    )


def test_concatenate_with_empty_operand():
    d = dense_example()
    e = empty_coo()
    got = eval_dense(
        lambda a, b: sparse.concatenate([a, b], axis=0), e, coo_example()
    )
    np.testing.assert_allclose(
        got, np.concatenate([np.zeros_like(d), d], axis=0), rtol=1e-6, atol=1e-6
    )
    got = eval_dense(
        lambda a, b: sparse.concatenate([a, b], axis=1), coo_example(), e
    )
    np.testing.assert_allclose(
        got, np.concatenate([d, np.zeros_like(d)], axis=1), rtol=1e-6, atol=1e-6
    )


def test_all_empty_concatenate():
    e = empty_coo()
    result = etl.evaluate(lambda a, b: sparse.concatenate([a, b], axis=0), e, e)
    assert sparse.is_sparse(result)
    assert result.indices.shape[0] == 0
    assert result.dense_shape == (6, 4)
    np.testing.assert_allclose(
        materialize(result), np.zeros((6, 4)), rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 13. Static error checks (trace time, exact wording from etl/sparse/ops.py)
# ---------------------------------------------------------------------------


def _spec_mismatched_shape():
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(2, 4),
    )


def _spec_mismatched_dtype():
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float64),
        dense_shape=(3, 4),
    )


def _spec_mismatch_nonconcat_axis():
    # (3, 5): differs from coo_spec's (3, 4) at axis 1 — a NON-concat-axis
    # mismatch for axis=0 concatenation.
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(3, 5),
    )


def _spec_rank1():
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 1), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(4,),
    )


def _spec_rank3():
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 3), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(2, 3, 4),
    )


@pytest.mark.parametrize(
    "build,specs,exc,match",
    [
        # add / multiply: shape + dtype equality
        (
            lambda a, b: sparse.add(a, b),
            (_spec_mismatched_shape(), coo_spec()),
            core.ShapeError,
            "dense shapes must match",
        ),
        (
            lambda a, b: sparse.add(a, b),
            (_spec_mismatched_dtype(), coo_spec()),
            core.DTypeError,
            "dtypes must match",
        ),
        (
            lambda a, b: sparse.multiply(a, b),
            (_spec_mismatched_shape(), coo_spec()),
            core.ShapeError,
            "dense shapes must match",
        ),
        (
            lambda a, b: sparse.multiply(a, b),
            (_spec_mismatched_dtype(), coo_spec()),
            core.DTypeError,
            "dtypes must match",
        ),
        # concatenate
        (
            lambda: sparse.concatenate(coo_example(), axis=0),
            (),
            TypeError,
            "must be a sequence",
        ),
        (
            lambda a: sparse.concatenate([a], axis=0),
            (coo_spec(),),
            TypeError,
            "at least 2",
        ),
        (
            lambda a, b: sparse.concatenate([a, b], axis=2),
            (coo_spec(), coo_spec()),
            core.ShapeError,
            "v1 supports axis 0 or 1",
        ),
        (
            lambda a, b: sparse.concatenate([a, b], axis=1),
            (_spec_rank1(), _spec_rank1()),
            core.ShapeError,
            "out of range",
        ),
        (
            lambda a, b: sparse.concatenate([a, b], axis=0),
            (_spec_mismatched_dtype(), coo_spec()),
            core.DTypeError,
            "operand dtypes must match",
        ),
        (
            lambda a, b: sparse.concatenate([a, b], axis=0),
            (coo_spec(), _spec_mismatch_nonconcat_axis()),
            core.ShapeError,
            "must match except along axis",
        ),
        # sum / reduce_sum
        (
            lambda x: sparse.sum(x, axes=5),
            (coo_spec(),),
            core.ShapeError,
            "out of range for rank",
        ),
        (
            lambda x: sparse.sum(x, axes=(0, "a")),
            (coo_spec(),),
            core.ShapeError,
            "axes must be ints",
        ),
        (
            lambda x: sparse.sum(x, axes=()),
            (coo_spec(),),
            core.ShapeError,
            "reduce over no axes",
        ),
        (
            lambda x: sparse.sum(x, axes=0.5),
            (coo_spec(),),
            TypeError,
            "axes must be None",
        ),
        # transpose
        (
            lambda x: sparse.transpose(x, (0,)),
            (coo_spec(),),
            core.ShapeError,
            "must be a permutation",
        ),
        (
            lambda x: sparse.transpose(x, (0.0, 1.0)),
            (coo_spec(),),
            core.ShapeError,
            "must be a permutation",
        ),
        (
            lambda x: sparse.transpose(x, (1, 1)),
            (coo_spec(),),
            core.ShapeError,
            "not a permutation",
        ),
        # reshape
        (
            lambda x: sparse.reshape(x, 12.0),
            (coo_spec(),),
            core.ShapeError,
            "non-empty tuple of positive ints",
        ),
        (
            lambda x: sparse.reshape(x, ()),
            (coo_spec(),),
            core.ShapeError,
            "non-empty tuple of positive ints",
        ),
        (
            lambda x: sparse.reshape(x, (0, 12)),
            (coo_spec(),),
            core.ShapeError,
            "positive non-bool ints",
        ),
        (
            lambda x: sparse.reshape(x, (5, 5)),
            (coo_spec(),),
            core.ShapeError,
            "element count",
        ),
        (
            lambda x: sparse.reshape(x, (12,)),
            (_spec_rank1(),),
            core.ShapeError,
            "element count",
        ),
        # matmul
        (
            lambda a, b: sparse.matmul(a, b),
            (_spec_rank3(), etl.TensorSpec((3, 4, 5), etl.float32)),
            core.ShapeError,
            "sparse operand must be rank-2",
        ),
        (
            lambda a, b: sparse.matmul(a, b),
            (coo_spec(), etl.TensorSpec((4,), etl.float32)),
            core.ShapeError,
            "dense operand must have rank >= 2",
        ),
        (
            lambda a, b: sparse.matmul(a, b),
            (coo_spec(), etl.TensorSpec((3, 5), etl.float32)),
            core.ShapeError,
            "inner dims must match",
        ),
        (
            lambda a, b: sparse.matmul(a, b),
            (etl.TensorSpec((5, 2), etl.float32), coo_spec()),
            core.ShapeError,
            "inner dims must match",
        ),
        (
            lambda a, b: sparse.matmul(a, b),
            (coo_spec(), coo_spec()),
            core.TraceError,
            "v1 deferral",
        ),
        # multiply_dense
        (
            lambda a, x: sparse.multiply_dense(a, x),
            (coo_spec(), etl.TensorSpec((2, 4), etl.float32)),
            core.ShapeError,
            "must equal the sparse dense_shape",
        ),
    ],
)
def test_static_errors_at_trace_time(build, specs, exc, match):
    with pytest.raises(exc, match=match):
        etl.trace(build, *specs)


def test_reshape_with_dim_in_dense_shape_is_rejected():
    dim = core.Dim("batch", 2)

    def f(indices, values):
        x = sparse.coo(indices, values, (dim, 4))
        return sparse.reshape(x, (8,))

    with pytest.raises(core.ShapeError, match="static shapes only in v1"):
        etl.trace(
            f,
            etl.TensorSpec((None, 2), etl.int64),
            etl.TensorSpec((None,), etl.float32),
        )


def test_ops_on_concrete_values_inside_trace_raise_no_eager_mode():
    # Representative checks: the full matrix lives in test_errors.py.
    with pytest.raises(core.TraceError, match="There is no eager mode"):
        etl.trace(lambda: sparse.add(coo_example(), csr_example()))
    with pytest.raises(core.TraceError, match="There is no eager mode"):
        etl.trace(lambda: sparse.negate(coo_example()))
    with pytest.raises(core.TraceError, match="There is no eager mode"):
        etl.trace(
            lambda a: sparse.multiply_dense(a, np.ones((3, 4), dtype=np.float32)),
            coo_spec(),
        )


def test_ops_outside_trace_raise():
    with pytest.raises(core.TraceError):
        sparse.add(coo_example(), csr_example())
    with pytest.raises(core.TraceError):
        sparse.negate(coo_example())


# ---------------------------------------------------------------------------
# 14. IR-contract spot checks: exact emitted op names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,specs,expected",
    [
        (
            lambda d: sparse.from_dense(d),
            (etl.TensorSpec((3, 4), etl.float32),),
            ["sparse_from_dense"],
        ),
        (lambda x: sparse.to_dense(x), (coo_spec(),), ["sparse_to_dense"]),
        (lambda x: sparse.to_csr(x), (coo_spec(),), ["sparse_coo_to_csr"]),
        (lambda x: sparse.to_coo(x), (csr_spec(),), ["sparse_csr_to_coo"]),
        (lambda x: sparse.to_csc(x), (coo_spec(),), ["sparse_coo_to_csc"]),
        (lambda x: sparse.to_coo(x), (csc_spec(),), ["sparse_csc_to_coo"]),
        (lambda x: sparse.negate(x), (coo_spec(),), ["sparse_negate"]),
        (
            lambda a, b: sparse.add(a, b),
            (coo_spec(), coo_spec()),
            ["sparse_add"],
        ),
        (
            lambda a, b: sparse.multiply(a, b),
            (coo_spec(), coo_spec()),
            ["sparse_multiply"],
        ),
        (
            lambda a, x: sparse.multiply_dense(a, x),
            (coo_spec(), etl.TensorSpec((3, 4), etl.float32)),
            ["sparse_multiply_dense"],
        ),
        (
            lambda x: sparse.sum(x, axes=0),
            (coo_spec(),),
            ["sparse_reduce_sum"],
        ),
        (lambda x: sparse.transpose(x), (coo_spec(),), ["sparse_transpose"]),
        (lambda x: sparse.reshape(x, (12,)), (coo_spec(),), ["sparse_reshape"]),
        (
            lambda a, b: sparse.concatenate([a, b], axis=0),
            (coo_spec(), coo_spec()),
            ["sparse_concatenate"],
        ),
        (
            lambda x, d: sparse.matmul(x, d),
            (coo_spec(), etl.TensorSpec((4, 5), etl.float32)),
            ["sparse_dot_dense"],
        ),
        (
            lambda d, x: sparse.matmul(d, x),
            (etl.TensorSpec((5, 3), etl.float32), coo_spec()),
            ["dense_dot_sparse"],
        ),
    ],
)
def test_exact_ir_op_names(build, specs, expected):
    graph = etl.trace(build, *specs)
    assert _ops_of(graph) == expected
