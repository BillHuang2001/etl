"""Value-model tests for ``etl.sparse`` — the three-phase sparse tensor.

Pins the contract in ``etl/sparse/CONTEXT.md`` + ``etl/sparse/value.py``
(read-only): concrete constructors and their canonical validation, integer
index-dtype normalization, ``from_dense`` extraction, ``SparseTensorSpec``
(``from_concrete`` + constructor validation), the ONE pytree registration
(children layout, polymorphic unflatten), ``is_sparse``, and the concrete
layout helpers (``to_dense`` / ``to_coo`` / ``to_csr`` / ``to_csc``).

Small shapes only; CPU only.
"""

import numpy as np
import pytest

import etl
from etl import core, sparse

from tests.sparse.conftest import (
    batched_coo_example,
    coo_example,
    csr_example,
    csc_example,
    coo_spec,
    dense_example,
)

# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _symbolic_coo():
    """A symbolic-phase COO instance, captured from a traced graph."""
    captured = {}

    def f(x):
        captured["x"] = x
        return x

    etl.trace(f, coo_spec())
    return captured["x"]


def _batched_coo_dim():
    """A batched COO built per the ``to_dense`` contract: ``dense_shape[0]``
    is a ``core.Dim`` whose extent is the batch (value.py's documented
    batched case; ``batched_coo_example()`` from conftest uses the
    input-side convention — see the BUG(etl) test below)."""
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return sparse.SparseTensor.from_parts(
        core.from_numpy(np.stack([idx, idx], axis=0)),
        core.from_numpy(np.stack([vals, vals * 2.0], axis=0)),
        dense_shape=(core.Dim(name="b"), 3, 4),
        format="coo",
    )


# ---------------------------------------------------------------------------
# 1. Concrete constructors — eager validated values.
# ---------------------------------------------------------------------------


def test_concrete_constructors_basic_properties():
    coo = coo_example()
    assert coo.format == "coo"
    assert coo.dense_shape == (3, 4)
    assert coo.dtype == np.dtype("float32")
    assert isinstance(coo.indices, np.ndarray)
    assert coo.indices.shape == (4, 2)
    assert coo.indices.dtype == np.dtype("int64")
    assert isinstance(coo.values, np.ndarray)
    assert coo.values.shape == (4,)
    assert coo.values.dtype == np.dtype("float32")

    csr = csr_example()
    assert csr.format == "csr"
    assert csr.dense_shape == (3, 4)
    assert csr.dtype == np.dtype("float32")
    assert csr.indptr.shape == (4,)
    assert csr.indptr.dtype == np.dtype("int64")
    assert csr.indices.shape == (4,)
    assert csr.indices.dtype == np.dtype("int64")
    assert csr.values.shape == (4,)

    csc = csc_example()
    assert csc.format == "csc"
    assert csc.dense_shape == (3, 4)
    assert csc.indptr.shape == (5,)
    assert csc.indptr.dtype == np.dtype("int64")


def test_coo_has_no_indptr():
    with pytest.raises(AttributeError, match="has no indptr"):
        coo_example().indptr


def test_index_dtypes_normalized_to_int64():
    s = sparse.coo(
        np.array([[0, 1]], dtype=np.int32),
        np.array([1.0], dtype=np.float64),
        (3, 4),
    )
    assert s.indices.dtype == np.dtype("int64")
    assert s.values.dtype == np.dtype("float64")  # values dtype preserved
    c = sparse.csr(
        np.array([0, 1, 1, 1], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
        (3, 4),
    )
    assert c.indptr.dtype == np.dtype("int64")
    assert c.indices.dtype == np.dtype("int64")
    assert c.values.dtype == np.dtype("float32")
    csc = sparse.csc(
        np.array([0, 1, 1, 1, 1], dtype=np.int16),
        np.array([1], dtype=np.int16),
        np.array([1.0], dtype=np.float32),
        (3, 4),
    )
    assert csc.indptr.dtype == np.dtype("int64")
    assert csc.indices.dtype == np.dtype("int64")


def test_core_tensor_leaves_accepted():
    s = sparse.coo(
        core.from_numpy(np.array([[0, 1]], dtype=np.int32)),
        core.from_numpy(np.array([1.0], dtype=np.float64)),
        (3, 4),
    )
    assert isinstance(s.indices, core.Tensor)
    assert s.indices.dtype == np.dtype("int64")  # normalized
    assert isinstance(s.values, core.Tensor)
    assert s.values.dtype == np.dtype("float64")  # preserved
    np.testing.assert_array_equal(
        s.to_dense(),
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )


# ---------------------------------------------------------------------------
# from_dense — exact np.nonzero extraction.
# ---------------------------------------------------------------------------


def test_from_dense_numpy_extraction():
    d = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    s = sparse.from_dense(d)
    assert s.format == "coo"
    assert s.dense_shape == (2, 3)
    assert s.dtype == np.dtype("float32")
    # np.nonzero is row-major -> indices are lex-sorted; values in dense dtype.
    np.testing.assert_array_equal(
        s.indices, np.array([[0, 1], [1, 2]], dtype=np.int64)
    )
    assert s.indices.dtype == np.dtype("int64")
    np.testing.assert_array_equal(s.values, np.array([1.0, 2.0], dtype=np.float32))
    assert s.values.dtype == np.dtype("float32")


def test_from_dense_lex_sorted_indices():
    # Entries deliberately scattered — np.nonzero yields row-major (lex)
    # order, so the extracted COO is canonical by construction.
    d = np.array([[0.0, 0.0, 5.0], [7.0, 0.0, 0.0], [0.0, 9.0, 0.0]], dtype=np.float64)
    s = sparse.from_dense(d)
    expected = np.stack(np.nonzero(d), axis=1).astype(np.int64)
    np.testing.assert_array_equal(s.indices, expected)
    # Lex-sorted row-major: leftmost column is the primary key.
    assert np.array_equal(
        np.lexsort(s.indices.T[::-1]), np.arange(s.indices.shape[0])
    )


def test_from_dense_empty_dense():
    s = sparse.from_dense(np.zeros((2, 3), dtype=np.float32))
    assert s.indices.shape == (0, 2)
    assert s.values.shape == (0,)
    assert s.dtype == np.dtype("float32")


def test_from_dense_core_tensor_dense():
    d = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    s = sparse.from_dense(core.from_numpy(d))
    np.testing.assert_array_equal(
        s.indices, np.array([[0, 1], [1, 1]], dtype=np.int64)
    )
    np.testing.assert_array_equal(s.values, np.array([1.0, 2.0], dtype=np.float32))


def test_from_dense_format_restriction():
    with pytest.raises(ValueError, match="supports only format='coo'"):
        sparse.from_dense(np.zeros((2, 2)), format="csr")


def test_from_dense_non_array():
    with pytest.raises(TypeError, match="expects a numpy array or core.Tensor"):
        sparse.from_dense([1, 2, 3])


# ---------------------------------------------------------------------------
# 2. SparseTensorSpec.from_concrete.
# ---------------------------------------------------------------------------


def test_from_concrete_coo():
    spec = sparse.SparseTensorSpec.from_concrete(coo_example())
    assert isinstance(spec, sparse.SparseTensorSpec)
    assert spec.format == "coo"
    assert spec.dense_shape == (3, 4)
    indices_spec, values_spec = spec._leaves
    assert indices_spec.shape == (None, 2)
    assert indices_spec.dtype == np.dtype("int64")
    assert values_spec.shape == (None,)
    assert values_spec.dtype == np.dtype("float32")


def test_from_concrete_csr():
    spec = sparse.SparseTensorSpec.from_concrete(csr_example())
    assert isinstance(spec, sparse.SparseTensorSpec)
    assert spec.format == "csr"
    assert spec.dense_shape == (3, 4)
    indptr_spec, indices_spec, values_spec = spec._leaves
    assert indptr_spec.shape == (4,)  # STATIC (rows+1,)
    assert indptr_spec.dtype == np.dtype("int64")
    assert indices_spec.shape == (None,)
    assert indices_spec.dtype == np.dtype("int64")
    assert values_spec.shape == (None,)
    assert values_spec.dtype == np.dtype("float32")


def test_from_concrete_csc():
    spec = sparse.SparseTensorSpec.from_concrete(csc_example())
    assert isinstance(spec, sparse.SparseTensorSpec)
    assert spec.format == "csc"
    assert spec.dense_shape == (3, 4)
    indptr_spec, indices_spec, values_spec = spec._leaves
    assert indptr_spec.shape == (5,)  # STATIC (cols+1,)
    assert indptr_spec.dtype == np.dtype("int64")
    assert indices_spec.shape == (None,)
    assert values_spec.shape == (None,)
    assert values_spec.dtype == np.dtype("float32")


def test_from_concrete_accepts_core_tensor_leaves():
    s = sparse.coo(
        core.from_numpy(np.array([[0, 1]], dtype=np.int64)),
        core.from_numpy(np.array([1.0], dtype=np.float32)),
        (3, 4),
    )
    spec = sparse.SparseTensorSpec.from_concrete(s)
    indices_spec, values_spec = spec._leaves
    assert indices_spec.shape == (None, 2)
    assert values_spec.dtype == np.dtype("float32")


def test_from_concrete_rejects_spec_input():
    with pytest.raises(TypeError, match="already a spec"):
        sparse.SparseTensorSpec.from_concrete(coo_spec())


def test_from_concrete_rejects_symbolic():
    with pytest.raises(TypeError, match="symbolic sparse tensor"):
        sparse.SparseTensorSpec.from_concrete(_symbolic_coo())


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.zeros((3, 4)), id="ndarray"),
        pytest.param(core.from_numpy(np.zeros((3, 4))), id="Tensor"),
        pytest.param(5, id="int"),
    ],
)
def test_from_concrete_rejects_non_sparse(bad):
    with pytest.raises(TypeError, match="expects a concrete sparse tensor"):
        sparse.SparseTensorSpec.from_concrete(bad)


# ---------------------------------------------------------------------------
# 3. Canonical-form validation errors (exact wording from value.py).
# ---------------------------------------------------------------------------

# Canonical building blocks for the error cases below.
_I4 = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
_V4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
_V2 = np.array([1.0, 2.0], dtype=np.float32)
_V1 = np.array([1.0], dtype=np.float32)
_IP = np.array([0, 1, 2, 4], dtype=np.int64)
_CI = np.array([1, 2, 0, 3], dtype=np.int64)


@pytest.mark.parametrize(
    "shape, match",
    [
        pytest.param((), "must be non-empty", id="empty"),
        pytest.param((3, 0), "entries must be positive ints", id="zero"),
        pytest.param((3, -1), "entries must be positive ints", id="negative"),
        pytest.param((3, "x"), "entries must be positive ints", id="non-int"),
        pytest.param((3, 4.0), "entries must be positive ints", id="float"),
        pytest.param(
            (3, core.Dim(name="d")),
            "may be a core.Dim",
            id="dim-not-at-0",
        ),
    ],
)
def test_dense_shape_errors(shape, match):
    with pytest.raises(core.ShapeError, match=match):
        sparse.coo(_I4, _V4, shape)


def test_dense_shape_non_iterable():
    with pytest.raises(core.ShapeError, match="must be a tuple of positive ints, got 5"):
        sparse.SparseTensor(_I4, _V4, 5, format="coo")


@pytest.mark.parametrize(
    "indices, values, match",
    [
        pytest.param(
            np.array([0, 1], dtype=np.int64), _V2, "COO indices must be 2-D",
            id="indices-rank-1",
        ),
        pytest.param(
            np.zeros((2, 4, 2), dtype=np.int64), _V2, "COO indices must be 2-D",
            id="indices-rank-3",
        ),
        pytest.param(
            np.array([[0, 1, 2]], dtype=np.int64), _V1, "matching dense_shape rank",
            id="indices-ndim-mismatch",
        ),
        pytest.param(
            _I4, np.zeros((2, 2), dtype=np.float32), "COO values must be 1-D",
            id="values-rank-2",
        ),
        pytest.param(
            _I4, _V2, "COO values length must equal nnz", id="values-length",
        ),
        pytest.param(
            np.array([[0, 1], [3, 2]], dtype=np.int64), _V2,
            "COO indices out of range for axis 0", id="out-of-range-axis-0",
        ),
        pytest.param(
            np.array([[0, 1], [1, 4]], dtype=np.int64), _V2,
            "COO indices out of range for axis 1", id="out-of-range-axis-1",
        ),
        pytest.param(
            np.array([[1, 2], [0, 1]], dtype=np.int64), _V2,
            "must be lex-sorted in row-major order", id="not-lex-sorted",
        ),
        pytest.param(
            np.array([[0, 1], [0, 1], [2, 0]], dtype=np.int64),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "must be unique", id="duplicate-rows",
        ),
    ],
)
def test_coo_canonical_errors(indices, values, match):
    with pytest.raises(core.ShapeError, match=match):
        sparse.coo(indices, values, (3, 4))


@pytest.mark.parametrize(
    "indptr, indices, values, shape, match",
    [
        pytest.param(
            _IP, _CI, _V4, (2, 3, 4),
            "CSR format requires a rank-2 dense_shape", id="rank-3-shape",
        ),
        pytest.param(
            _IP.reshape(2, 2), _CI, _V4, (3, 4),
            "indptr must be 1-D", id="indptr-rank-2",
        ),
        pytest.param(
            np.array([0, 1, 2, 4, 4], dtype=np.int64), _CI, _V4, (3, 4),
            "indptr must have shape", id="indptr-length",
        ),
        pytest.param(
            np.array([1, 2, 4, 4], dtype=np.int64), _CI, _V4, (3, 4),
            r"indptr\[0\] must be 0", id="indptr-first-nonzero",
        ),
        pytest.param(
            np.array([0, 2, 1, 4], dtype=np.int64), _CI, _V4, (3, 4),
            "indptr must be monotone non-decreasing", id="indptr-non-monotone",
        ),
        pytest.param(
            np.array([0, 1, 2, 3], dtype=np.int64), _CI, _V4, (3, 4),
            "must equal nnz", id="indptr-last-mismatch",
        ),
        pytest.param(
            _IP, _CI.reshape(2, 2), _V4, (3, 4),
            "indices must be 1-D", id="indices-rank-2",
        ),
        pytest.param(
            _IP, _CI, _V4.reshape(2, 2), (3, 4),
            "values must be 1-D", id="values-rank-2",
        ),
        pytest.param(
            _IP, _CI, _V2, (3, 4),
            "values length must equal nnz", id="values-length",
        ),
        pytest.param(
            _IP, np.array([1, 2, 0, 5], dtype=np.int64), _V4, (3, 4),
            "CSR indices must be in range", id="cols-out-of-range",
        ),
        pytest.param(
            np.array([0, 2, 4, 4], dtype=np.int64),
            np.array([2, 1, 0, 3], dtype=np.int64), _V4, (3, 4),
            "per-row indices must be sorted and strictly increasing",
            id="per-row-non-increasing",
        ),
        pytest.param(
            np.array([0, 2, 4, 4], dtype=np.int64),
            np.array([1, 1, 0, 3], dtype=np.int64), _V4, (3, 4),
            "per-row indices must be sorted and strictly increasing",
            id="per-row-equal-nonzero",
        ),
    ],
)
def test_csr_canonical_errors(indptr, indices, values, shape, match):
    with pytest.raises(core.ShapeError, match=match):
        sparse.csr(indptr, indices, values, shape)


@pytest.mark.parametrize(
    "indptr, indices, values, shape, match",
    [
        pytest.param(
            _IP, _CI, _V4, (2, 3, 4),
            "CSC format requires a rank-2 dense_shape", id="rank-3-shape",
        ),
        pytest.param(
            np.array([0, 1, 2, 4], dtype=np.int64), _CI, _V4, (3, 4),
            "indptr must have shape", id="indptr-length",
        ),
        pytest.param(
            np.array([0, 1, 2, 3, 4], dtype=np.int64),
            np.array([2, 0, 1, 3], dtype=np.int64), _V4, (3, 4),
            "CSC indices must be in range", id="rows-out-of-range",
        ),
        pytest.param(
            np.array([0, 2, 3, 4, 4], dtype=np.int64),
            np.array([2, 0, 1, 2], dtype=np.int64), _V4, (3, 4),
            "per-column indices must be sorted and strictly increasing",
            id="per-col-non-increasing",
        ),
    ],
)
def test_csc_canonical_errors(indptr, indices, values, shape, match):
    with pytest.raises(core.ShapeError, match=match):
        sparse.csc(indptr, indices, values, shape)


@pytest.mark.parametrize(
    "indices, values, match",
    [
        pytest.param(
            np.array([[0.0, 1.0]], dtype=np.float64), _V1,
            "indices must be integer-typed", id="float-indices",
        ),
        pytest.param(
            np.array([[True, False]], dtype=bool), _V1,
            "indices must be integer-typed", id="bool-indices",
        ),
        pytest.param(
            _I4, np.array([1, 2, 3, 4], dtype=object),
            "values must have a numeric dtype", id="object-values",
        ),
        pytest.param(
            _I4, np.array(["a", "b", "c", "d"]),
            "values must have a numeric dtype", id="string-values",
        ),
    ],
)
def test_dtype_errors(indices, values, match):
    with pytest.raises(core.DTypeError, match=match):
        sparse.coo(indices, values, (3, 4))


def test_non_array_leaves_dtype_error():
    with pytest.raises(
        core.DTypeError, match="indices must be an integer numpy array or core.Tensor"
    ):
        sparse.SparseTensor([[0, 1]], _V4, (3, 4), format="coo")
    with pytest.raises(
        core.DTypeError, match="values must be a numpy array or core.Tensor"
    ):
        sparse.SparseTensor(_I4, [1.0, 2.0, 3.0, 4.0], (3, 4), format="coo")


def test_base_class_format_value_errors():
    with pytest.raises(ValueError, match="not constructible through the base class"):
        sparse.SparseTensor(_I4, _V4, (3, 4), format="csr")
    with pytest.raises(ValueError, match="unknown sparse format"):
        sparse.SparseTensor(_I4, _V4, (3, 4), format="dia")
    with pytest.raises(ValueError, match="from_parts: unknown sparse format"):
        sparse.SparseTensor.from_parts(_I4, _V4, dense_shape=(3, 4), format="dia")


def test_stored_zero_duplicate_row_constructs():
    # BUG(etl): the concrete COO constructor rejects duplicate rows even when
    # one of the pair is a stored zero, but the contract
    # (etl/sparse/CONTEXT.md: "duplicate NONZERO rows error; duplicate rows
    # with one stored zero are legal") requires them to construct fine. The
    # runtime kernels ARE values-aware (see
    # tests/sparse/test_errors.py::test_runtime_stored_zero_duplicate_passes);
    # the concrete constructor is not.
    # Minimal repro:
    s = sparse.coo(
        np.array([[0, 1], [0, 1], [2, 0]], dtype=np.int64),
        np.array([0.0, 1.0, 3.0], dtype=np.float32),
        (3, 4),
    )
    assert s.indices.shape == (3, 2)
    np.testing.assert_array_equal(
        s.to_dense(),
        np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )


# ---------------------------------------------------------------------------
# 4. SparseTensorSpec constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, match",
    [
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((None, 2), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                etl.TensorSpec((None,), etl.int64),
                dense_shape=(3, 4),
            ),
            "COO requires 2 leaf specs",
            id="coo-wrong-leaf-count",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                dense_shape=(3, 4),
                format="csr",
            ),
            "CSR requires 3 leaf specs",
            id="csr-wrong-leaf-count",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                5,
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
            ),
            "indices leaf spec must be a core.TensorSpec",
            id="non-tensorspec-leaf",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4, 2), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
            ),
            "COO indices spec must have shape",
            id="coo-indices-shape",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((None, 2), etl.int64),
                etl.TensorSpec((4,), etl.float32),
                dense_shape=(3, 4),
            ),
            "COO values spec must have shape",
            id="coo-values-shape",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((None, 2), etl.float32),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
            ),
            "indices spec dtype must be int64",
            id="coo-indices-dtype",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(2, 3, 4),
                format="csr",
            ),
            "CSR requires a rank-2 dense_shape",
            id="csr-rank-3",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="csr",
            ),
            "CSR indptr spec must have shape",
            id="csr-indptr-shape",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.float32),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="csr",
            ),
            "CSR indptr spec dtype must be int64",
            id="csr-indptr-dtype",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="csr",
            ),
            "CSR indices spec dtype must be int64",
            id="csr-indices-dtype",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="csc",
            ),
            "CSC indptr spec must have shape",
            id="csc-indptr-shape",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((4,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(core.Dim(name="b"), 4),
                format="csr",
            ),
            "must be STATIC",
            id="csr-dim-at-0",
        ),
        pytest.param(
            lambda: sparse.SparseTensorSpec(
                etl.TensorSpec((None, 2), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="dia",
            ),
            "unknown sparse format",
            id="unknown-format",
        ),
    ],
)
def test_spec_constructor_validation(build, match):
    with pytest.raises((core.ShapeError, core.DTypeError, ValueError), match=match):
        build()


def test_spec_dim_at_position_zero_allowed_for_coo():
    spec = sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(core.Dim(name="b"), 4),
    )
    assert isinstance(spec.dense_shape[0], core.Dim)
    assert spec.dense_shape[1] == 4


# ---------------------------------------------------------------------------
# 5. Pytree contract — ONE registered node, polymorphic unflatten.
# ---------------------------------------------------------------------------


def test_pytree_concrete_roundtrip():
    for ex in (coo_example(), csr_example(), csc_example()):
        children, tree = core.flatten(ex)
        assert tree.type is sparse.SparseTensor
        n_leaves = 2 if ex.format == "coo" else 3
        assert tree.num_leaves == n_leaves + ex.ndim + 2
        # Children layout: [tensor leaves..., *dense_shape, dtype, format].
        assert tuple(children[n_leaves:-2]) == ex.dense_shape
        assert all(isinstance(d, int) for d in children[n_leaves:-2])
        assert isinstance(children[-2], np.dtype)
        assert children[-2] == ex.dtype
        assert children[-1] == ex.format
        assert isinstance(children[-1], str)
        # Tensor leaves come first, in canonical order.
        assert all(isinstance(c, np.ndarray) for c in children[:n_leaves])
        back = core.unflatten(children, tree)
        assert type(back) is type(ex)
        assert back.format == ex.format
        assert back.dense_shape == ex.dense_shape
        assert back.dtype == ex.dtype


def test_pytree_spec_roundtrip():
    spec = coo_spec()
    children, tree = core.flatten(spec)
    assert tree.type is sparse.SparseTensor
    assert tree.num_leaves == 6
    assert isinstance(children[0], core.TensorSpec)
    assert isinstance(children[1], core.TensorSpec)
    assert children[-1] == "coo"
    back = core.unflatten(children, tree)
    assert isinstance(back, sparse.SparseTensorSpec)
    assert back.dense_shape == (3, 4)


def test_pytree_symbolic_roundtrip():
    sym = _symbolic_coo()
    children, tree = core.flatten(sym)
    assert tree.type is sparse.SparseTensor
    assert tree.num_leaves == 6
    assert isinstance(children[0], core.SymbolicTensor)
    assert isinstance(children[1], core.SymbolicTensor)
    assert children[-1] == "coo"
    assert children[-2] == np.dtype("float32")
    back = core.unflatten(children, tree)
    assert sparse.is_sparse(back)
    assert isinstance(back.values, core.SymbolicTensor)
    assert back.format == "coo"
    assert back.dense_shape == (3, 4)


def test_pytree_unflatten_dispatch_on_first_child():
    # TensorSpec leaf -> SparseTensorSpec.
    spec_children, spec_tree = core.flatten(coo_spec())
    rebuilt = core.unflatten(spec_children, spec_tree)
    assert isinstance(rebuilt, sparse.SparseTensorSpec)

    # SymbolicTensor leaf -> symbolic instance (from_parts, no validation).
    sym_children, sym_tree = core.flatten(_symbolic_coo())
    rebuilt = core.unflatten(sym_children, sym_tree)
    assert isinstance(rebuilt.values, core.SymbolicTensor)

    # ndarray leaf -> concrete (canonical validation applies).
    arr_children, arr_tree = core.flatten(coo_example())
    rebuilt = core.unflatten(arr_children, arr_tree)
    assert isinstance(rebuilt.indices, np.ndarray)

    # core.Tensor leaf -> concrete (validated; Tensor leaves kept).
    t = sparse.coo(
        core.from_numpy(np.array([[0, 1]], dtype=np.int64)),
        core.from_numpy(np.array([1.0], dtype=np.float32)),
        (3, 4),
    )
    ten_children, ten_tree = core.flatten(t)
    rebuilt = core.unflatten(ten_children, ten_tree)
    assert isinstance(rebuilt.indices, core.Tensor)


def test_pytree_unflatten_dtype_leaf_mismatch():
    children, tree = core.flatten(coo_example())
    children[-2] = np.dtype("float64")
    with pytest.raises(core.DTypeError, match="does not match the values leaf dtype"):
        core.unflatten(children, tree)


# ---------------------------------------------------------------------------
# 6. is_sparse — every phase and variant.
# ---------------------------------------------------------------------------


def test_is_sparse():
    assert sparse.is_sparse(coo_example())
    assert sparse.is_sparse(csr_example())
    assert sparse.is_sparse(csc_example())
    assert sparse.is_sparse(coo_spec())
    assert sparse.is_sparse(_symbolic_coo())
    assert not sparse.is_sparse(np.zeros((3, 4)))
    assert not sparse.is_sparse(core.from_numpy(np.zeros((3, 4))))
    assert not sparse.is_sparse(5)
    assert not sparse.is_sparse({})
    assert not sparse.is_sparse("coo")


# ---------------------------------------------------------------------------
# 7. Concrete layout helpers.
# ---------------------------------------------------------------------------


def test_to_dense_all_formats():
    ref = dense_example()
    for ex in (coo_example(), csr_example(), csc_example()):
        out = ex.to_dense()
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, ref)


def test_to_coo():
    ref = dense_example()
    coo = coo_example()
    assert coo.to_coo() is coo  # identity on COO
    for ex in (csr_example(), csc_example()):
        converted = ex.to_coo()
        assert isinstance(converted, sparse.SparseTensor)
        assert converted.format == "coo"
        np.testing.assert_array_equal(converted.to_dense(), ref)
    # CSC -> COO is re-sorted row-major: indices == np.nonzero of the dense.
    expected = np.stack(np.nonzero(ref), axis=1).astype(np.int64)
    np.testing.assert_array_equal(csc_example().to_coo().indices, expected)


def test_to_csr():
    ref = dense_example()
    csr = csr_example()
    assert csr.to_csr() is csr  # identity on CSR
    for ex in (coo_example(), csc_example()):
        converted = ex.to_csr()
        assert isinstance(converted, sparse.CSRTensor)
        assert converted.format == "csr"
        np.testing.assert_array_equal(converted.to_dense(), ref)


def test_to_csc():
    ref = dense_example()
    csc = csc_example()
    assert csc.to_csc() is csc  # identity on CSC
    for ex in (coo_example(), csr_example()):
        converted = ex.to_csc()
        assert isinstance(converted, sparse.CSCTensor)
        assert converted.format == "csc"
        np.testing.assert_array_equal(converted.to_dense(), ref)


def test_batched_to_dense_dim_at_position_zero():
    # The documented to_dense batched case: dense_shape[0] is a Dim whose
    # extent is indices.shape[0] (the batch).
    b = _batched_coo_dim()
    out = b.to_dense()
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[0], dense_example())
    np.testing.assert_array_equal(out[1], dense_example() * 2.0)


def test_batched_concrete_to_dense_input_side_convention():
    # BUG(etl): `batched_coo_example()` follows the documented input-side
    # batched convention — batched leaves with dense_shape UNCHANGED (3, 4)
    # (conftest docstring; CONTEXT.md "Batched sparse = leading batch dim on
    # the tensor leaves with dense_shape UNCHANGED at the input I/O
    # boundary"). The concrete `to_dense` batched branch (value.py) assumes
    # dense_shape[0] is a core.Dim and CRASHES with a raw IndexError
    # ("too many indices for array: array is 2-dimensional, but 3 were
    # indexed") instead of materializing (B, 3, 4) — never an explicit error.
    # Minimal repro:
    b = batched_coo_example()
    out = b.to_dense()
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[0], dense_example())
    np.testing.assert_array_equal(out[1], dense_example() * 2.0)
