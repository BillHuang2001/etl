"""Operand-phase errors + run-time canonical validation for ``etl.sparse``.

Pins the contract in ``etl/sparse/CONTEXT.md`` + ``etl/sparse/ops.py`` +
``etl/backends/numpy/kernels/sparse.py`` (all read-only):

- the mandated three-option ``core.TraceError`` for concrete operands inside
  a trace ("There is no eager mode" + "explicit input" / "etl.constant" /
  "etl.evaluate"), and the no-active-builder ``TraceError`` outside a trace;
- spec-phase ``TypeError``s (TensorSpec creator components, mixed
  concrete/symbolic components, converters on a spec, converters on
  non-sparse operands);
- the sparse @ sparse matmul v1 deferral;
- RUN-TIME canonical validation by the numpy kernels (validation-free
  concrete inputs fed through a traced graph raise ``core.ShapeError``
  naming the op; the values-aware stored-zero tolerance passes);
- the trace-time rank-2 requirement for CSR/CSC -> COO conversion.

Cross-cutting v1 deferrals owned by ``test_deferrals.py`` (etl.constant on
sparse, whole-sparse bind, vjp TransformErrors, compiler-backend
BackendErrors) are NOT duplicated here.
"""

import numpy as np
import pytest

import etl
from etl import core, sparse

from tests.sparse.conftest import (
    coo_example,
    coo_spec,
    dense_example,
    materialize,
    run_graph,
)

# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _csr_spec(dtype=etl.float32):
    """Local CSR spec. (The conftest `csr_spec` helper omits
    ``format="csr"``, so calling it raises ShapeError "COO requires 2 leaf
    specs, got 3" — conftest is shared and not edited; see the suite report.)"""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((4,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csr",
    )


def _symbolic_sparse_arg():
    """A symbolic-phase sparse instance, captured from a traced graph."""
    captured = {}

    def f(x):
        captured["x"] = x
        return x

    etl.trace(f, coo_spec())
    return captured["x"]


def _coo_from_parts(indices, values, dense_shape=(3, 4)):
    """Validation-free concrete COO (run-time validation is the point)."""
    return sparse.SparseTensor.from_parts(
        indices, values, dense_shape=dense_shape, format="coo"
    )


def _run_coo_todense_graph(bad):
    """Trace a coo_spec() -> to_dense graph and run it with `bad`."""
    graph = etl.trace(lambda a: sparse.to_dense(a), coo_spec())
    return run_graph(graph, bad)


def _csr_from_parts(indptr, indices, values):
    """Validation-free concrete CSR (run-time validation is the point)."""
    return sparse.CSRTensor.from_parts(
        indptr, indices, values, dense_shape=(3, 4), format="csr"
    )


def _run_csr_todense_graph(bad):
    """Trace a csr_spec() -> to_dense graph and run it with `bad`.

    Symbolic CSR inputs to `to_dense` first emit `sparse_csr_to_coo`, so the
    indptr/segment violations surface from the ``sparse_csr_to_coo`` kernel.
    """
    graph = etl.trace(lambda a: sparse.to_dense(a), _csr_spec())
    return run_graph(graph, bad)


# ---------------------------------------------------------------------------
# 1. Three-option TraceError on concrete operands inside a trace.
# ---------------------------------------------------------------------------

_THREE_OPTIONS = ("There is no eager mode", "explicit input", "etl.constant", "etl.evaluate")


def _assert_three_option_traceerror(fn, *specs):
    with pytest.raises(core.TraceError) as exc_info:
        etl.trace(fn, *specs)
    msg = str(exc_info.value)
    for fragment in _THREE_OPTIONS:
        assert fragment in msg, f"missing {fragment!r} in: {msg}"


@pytest.mark.parametrize(
    "op_call",
    [
        pytest.param(lambda a: sparse.add(a, coo_example()), id="add"),
        pytest.param(lambda a: sparse.subtract(a, coo_example()), id="subtract"),
        pytest.param(lambda a: sparse.multiply(a, coo_example()), id="multiply"),
        pytest.param(
            lambda a: sparse.multiply_dense(coo_example(), a),
            id="multiply_dense-sparse-side",
        ),
        pytest.param(lambda a: sparse.negate(coo_example()), id="negate"),
        pytest.param(lambda a: sparse.sum(coo_example()), id="sum"),
        pytest.param(lambda a: sparse.transpose(coo_example()), id="transpose"),
        pytest.param(lambda a: sparse.reshape(coo_example(), (4, 3)), id="reshape"),
        pytest.param(
            lambda a: sparse.concatenate([a, coo_example()]), id="concatenate"
        ),
        pytest.param(
            lambda a: sparse.matmul(coo_example(), a),
            id="matmul-sparse-side-concrete",
        ),
        pytest.param(
            lambda a: sparse.matmul(a, coo_example()),
            id="matmul-sparse-second-concrete",
        ),
    ],
)
def test_three_option_traceerror_concrete_sparse_inside_trace(op_call):
    def f(a):
        return op_call(a)

    _assert_three_option_traceerror(f, coo_spec())


@etl.defn
def _defn_negate_concrete(x):
    return sparse.negate(coo_example())


def test_three_option_traceerror_via_defn_marker():
    _assert_three_option_traceerror(_defn_negate_concrete, coo_spec())


@pytest.mark.parametrize(
    "op_call",
    [
        pytest.param(
            lambda a: sparse.multiply_dense(a, np.zeros((3, 4), dtype=np.float32)),
            id="multiply_dense-ndarray",
        ),
        pytest.param(
            lambda a: sparse.multiply_dense(
                a, core.from_numpy(np.zeros((3, 4), dtype=np.float32))
            ),
            id="multiply_dense-Tensor",
        ),
        pytest.param(
            lambda a: sparse.matmul(a, np.zeros((4, 2), dtype=np.float32)),
            id="matmul-dense-side-ndarray",
        ),
        pytest.param(
            lambda a: sparse.matmul(
                a, core.from_numpy(np.zeros((4, 2), dtype=np.float32))
            ),
            id="matmul-dense-side-Tensor",
        ),
        pytest.param(
            lambda a: sparse.matmul(np.zeros((2, 4), dtype=np.float32), a),
            id="matmul-dense-first-ndarray",
        ),
        pytest.param(
            lambda a: sparse.matmul(
                core.from_numpy(np.zeros((2, 4), dtype=np.float32)), a
            ),
            id="matmul-dense-first-Tensor",
        ),
    ],
)
def test_three_option_traceerror_concrete_dense_inside_trace(op_call):
    def f(a):
        return op_call(a)

    _assert_three_option_traceerror(f, coo_spec())


@pytest.mark.parametrize(
    "converter_call",
    [
        pytest.param(lambda: sparse.to_dense(coo_example()), id="to_dense"),
        pytest.param(lambda: sparse.to_csr(coo_example()), id="to_csr"),
        pytest.param(lambda: sparse.to_csc(coo_example()), id="to_csc"),
        pytest.param(lambda: sparse.to_coo(coo_example()), id="to_coo"),
    ],
)
def test_converters_concrete_inside_trace_traceerror(converter_call):
    # The converters are POLYMORPHIC: on a concrete instance they dispatch to
    # the eager layout method (CONTEXT.md), so the op itself does not raise
    # the three-option TraceError; the concrete result is then rejected by
    # the trace-output validation with the "There is no eager mode"
    # TraceError. The three-option wording belongs to the computation ops'
    # operand normalization (covered above), not to this path.
    def f(a):
        return converter_call()

    with pytest.raises(core.TraceError, match="There is no eager mode"):
        etl.trace(f, coo_spec())


@pytest.mark.parametrize(
    "op_call",
    [
        pytest.param(lambda: sparse.negate(coo_example()), id="negate"),
        pytest.param(lambda: sparse.sum(coo_example()), id="sum"),
        pytest.param(
            lambda: sparse.add(coo_example(), coo_example()), id="add"
        ),
        pytest.param(lambda: sparse.transpose(coo_example()), id="transpose"),
        pytest.param(lambda: sparse.reshape(coo_example(), (4, 3)), id="reshape"),
        pytest.param(
            lambda: sparse.concatenate([coo_example(), coo_example()]),
            id="concatenate",
        ),
    ],
)
def test_no_active_builder_outside_trace(op_call):
    with pytest.raises(core.TraceError, match="No active trace"):
        op_call()


# ---------------------------------------------------------------------------
# 2. Spec-phase TypeErrors.
# ---------------------------------------------------------------------------


def test_creator_tensorspec_components():
    idx_spec = etl.TensorSpec((None, 2), etl.int64)
    val_spec = etl.TensorSpec((None,), etl.float32)
    with pytest.raises(TypeError, match="for trace inputs use a SparseTensorSpec"):
        sparse.coo(idx_spec, val_spec, (3, 4))
    with pytest.raises(TypeError, match="for trace inputs use a SparseTensorSpec"):
        sparse.csr(etl.TensorSpec((4,), etl.int64), idx_spec, val_spec, (3, 4))
    with pytest.raises(TypeError, match="for trace inputs use a SparseTensorSpec"):
        sparse.csc(etl.TensorSpec((5,), etl.int64), idx_spec, val_spec, (3, 4))


def test_creator_mixed_concrete_symbolic_components():
    sym = _symbolic_sparse_arg()
    with pytest.raises(TypeError, match="ALL concrete"):
        sparse.coo(np.array([[0, 1]], dtype=np.int64), sym.values, (3, 4))
    with pytest.raises(TypeError, match="mixed kinds"):
        sparse.csr(
            np.array([0, 1], dtype=np.int64),
            sym.indices,
            np.array([1.0], dtype=np.float32),
            (3, 4),
        )


def test_converters_reject_spec_phase():
    spec = coo_spec()
    with pytest.raises(TypeError, match="cannot be materialized eagerly"):
        sparse.to_dense(spec)
    with pytest.raises(TypeError, match="SparseTensorSpec"):
        sparse.to_csr(spec)
    with pytest.raises(TypeError, match="SparseTensorSpec"):
        sparse.to_csc(spec)
    with pytest.raises(TypeError, match="SparseTensorSpec"):
        sparse.to_coo(spec)


def test_sparse_in_dense_operand_position_typeerror():
    # The dense operand of multiply_dense must be a DENSE tensor — passing a
    # sparse value there is a TypeError (naming the accepted kinds), not the
    # three-option TraceError (which is reserved for concrete dense/sparse
    # values in valid operand positions).
    def f(a):
        return sparse.multiply_dense(a, coo_example())

    with pytest.raises(TypeError, match="dense operands must be SymbolicTensor"):
        etl.trace(f, coo_spec())


@pytest.mark.parametrize(
    "converter", [sparse.to_dense, sparse.to_csr, sparse.to_csc, sparse.to_coo]
)
def test_converters_reject_non_sparse(converter):
    with pytest.raises(TypeError, match="expects a sparse tensor"):
        converter(np.zeros((3, 4)))
    with pytest.raises(TypeError, match="expects a sparse tensor"):
        converter(core.from_numpy(np.zeros((3, 4))))


# ---------------------------------------------------------------------------
# 3. sparse @ sparse matmul deferral.
# ---------------------------------------------------------------------------


def test_sparse_sparse_matmul_v1_deferral():
    def f(a, b):
        return sparse.matmul(a, b)

    with pytest.raises(core.TraceError) as exc_info:
        etl.trace(f, coo_spec(), coo_spec())
    msg = str(exc_info.value)
    assert "v1 deferral" in msg
    assert "to_dense" in msg


# ---------------------------------------------------------------------------
# 4. Run-time canonical validation (numpy kernels).
# ---------------------------------------------------------------------------


def test_runtime_unsorted_rows():
    with pytest.raises(core.ShapeError, match="'sparse_to_dense'.*not lex-sorted"):
        _run_coo_todense_graph(
            _coo_from_parts(
                np.array([[1, 2], [0, 1]], dtype=np.int64),
                np.array([3.0, 1.0], dtype=np.float32),
            )
        )


def test_runtime_duplicate_nonzero_rows():
    with pytest.raises(core.ShapeError, match="'sparse_to_dense'.*duplicate rows"):
        _run_coo_todense_graph(
            _coo_from_parts(
                np.array([[0, 1], [0, 1], [2, 0]], dtype=np.int64),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        )


def test_runtime_out_of_range_coords():
    with pytest.raises(
        core.ShapeError, match="'sparse_to_dense'.*out of range for sparse axis 1"
    ):
        _run_coo_todense_graph(
            _coo_from_parts(
                np.array([[0, 1], [1, 4]], dtype=np.int64),
                np.array([1.0, 2.0], dtype=np.float32),
            )
        )


def test_runtime_values_length_mismatch():
    with pytest.raises(
        core.ShapeError, match="'sparse_to_dense'.*inconsistent with indices shape"
    ):
        _run_coo_todense_graph(
            _coo_from_parts(
                np.array([[0, 1], [1, 2]], dtype=np.int64),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )
        )


def test_runtime_stored_zero_duplicate_passes():
    # Values-aware validation: a duplicate row whose pair includes a stored
    # zero is legal at run time (the duplicate only double-counts a zero).
    out = _run_coo_todense_graph(
        _coo_from_parts(
            np.array([[0, 1], [0, 1], [2, 0]], dtype=np.int64),
            np.array([0.0, 1.0, 3.0], dtype=np.float32),
        )
    )
    np.testing.assert_array_equal(
        materialize(out),
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


def test_runtime_csr_indptr_first_nonzero():
    with pytest.raises(core.ShapeError, match=r"'sparse_csr_to_coo'.*indptr\[0\] must be 0"):
        _run_csr_todense_graph(
            _csr_from_parts(
                np.array([1, 2, 4, 4], dtype=np.int64),
                np.array([1, 2, 0, 3], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        )


def test_runtime_csr_indptr_non_monotone():
    with pytest.raises(
        core.ShapeError, match="'sparse_csr_to_coo'.*not monotone non-decreasing"
    ):
        _run_csr_todense_graph(
            _csr_from_parts(
                np.array([0, 2, 1, 4], dtype=np.int64),
                np.array([1, 2, 0, 3], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        )


def test_runtime_csr_indptr_last_mismatch():
    with pytest.raises(
        core.ShapeError,
        match="'sparse_csr_to_coo'.*must equal the number of stored entries",
    ):
        _run_csr_todense_graph(
            _csr_from_parts(
                np.array([0, 1, 2, 3], dtype=np.int64),
                np.array([1, 2, 0, 3], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        )


def test_runtime_csr_columns_out_of_range():
    with pytest.raises(
        core.ShapeError, match="'sparse_csr_to_coo'.*column index out of range"
    ):
        _run_csr_todense_graph(
            _csr_from_parts(
                np.array([0, 1, 2, 4], dtype=np.int64),
                np.array([1, 2, 0, 5], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        )


def test_runtime_csr_per_row_non_increasing():
    with pytest.raises(
        core.ShapeError,
        match="'sparse_csr_to_coo'.*not strictly increasing within each row/column",
    ):
        _run_csr_todense_graph(
            _csr_from_parts(
                np.array([0, 2, 4, 4], dtype=np.int64),
                np.array([2, 1, 0, 3], dtype=np.int64),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        )


def test_runtime_csr_canonical_input_passes():
    out = _run_csr_todense_graph(
        _csr_from_parts(
            np.array([0, 1, 2, 4], dtype=np.int64),
            np.array([1, 2, 0, 3], dtype=np.int64),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
    )
    np.testing.assert_array_equal(materialize(out), dense_example())


# ---------------------------------------------------------------------------
# 5. Trace-time rank-2 requirement for CSR/CSC -> COO conversion.
# ---------------------------------------------------------------------------


def test_symbolic_rank3_csr_to_dense_shape_error():
    def f(a):
        sym = sparse.SparseTensor.from_parts(
            a.indices, a.indices, a.values, dense_shape=(2, 3, 4), format="csr"
        )
        return sparse.to_dense(sym)

    with pytest.raises(core.ShapeError, match="requires a rank-2 sparse tensor"):
        etl.trace(f, coo_spec())


def test_symbolic_rank3_csc_to_dense_shape_error():
    def f(a):
        sym = sparse.SparseTensor.from_parts(
            a.indices, a.indices, a.values, dense_shape=(2, 3, 4), format="csc"
        )
        return sparse.to_dense(sym)

    with pytest.raises(core.ShapeError, match="requires a rank-2 sparse tensor"):
        etl.trace(f, coo_spec())
