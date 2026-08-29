"""Statistics and misc math ops: var, std, median, nansum, trace, diagonal,
sort, norm — numpy semantics (etl/ops/stats.py compositions, etl/ops/linalg.py
primitives/compositions).

Conventions: numeric checks compare against numpy on the default numpy
backend; IR checks inspect the traced graph (``ops_of``); error contracts use
``etl.trace`` to capture failures at trace time.
"""
import numpy as np
import pytest

import etl

from tests.ops.conftest import ops_of, run_numpy

# ---------------------------------------------------------------------------
# var / std
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axes", [None, 0, 1, -1, (0, 1)], ids=str)
@pytest.mark.parametrize("ddof", [0, 1], ids=str)
@pytest.mark.parametrize("keepdims", [False, True], ids=str)
def test_var_matches_numpy(axes, ddof, keepdims):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = run_numpy(
        lambda t: etl.var(t, axes=axes, keepdims=keepdims, ddof=ddof), x
    )
    ref = np.var(x, axis=axes, keepdims=keepdims, ddof=ddof)
    assert got.dtype == ref.dtype
    assert np.allclose(got, ref)


def test_std_matches_numpy():
    x = np.array([1.0, 2.0, 4.0], np.float32)
    assert np.allclose(run_numpy(lambda t: etl.std(t), x), np.std(x))
    assert np.allclose(run_numpy(lambda t: etl.std(t, ddof=1), x), np.std(x, ddof=1))


def test_var_int_input_promotes_to_float64():
    x = np.array([1, 2, 3], np.int32)
    got = run_numpy(lambda t: etl.var(t), x)
    assert got.dtype == np.dtype("float64")
    assert np.allclose(got, np.var(x))


def test_var_degenerate_ddof_matches_numpy():
    """Divisor is clamped at max(n - ddof, 0) and divided raw — numpy parity
    for degenerate ddof (n=2, ddof=2 → inf, matching numpy's inf)."""
    x = np.array([1.0, 2.0], np.float32)
    for ddof in (2, 3):
        with np.errstate(invalid="ignore", divide="ignore"):
            ref = np.var(x, ddof=ddof)
        got = run_numpy(lambda t: etl.var(t, ddof=ddof), x)
        assert np.array_equal(got, ref, equal_nan=True)


def test_var_std_are_documented_compositions():
    graph = etl.trace(lambda t: etl.var(t), etl.TensorSpec((4,), etl.float32))
    names = [op.name for op in ops_of(graph)]
    assert "var" not in names  # no dedicated IR op
    assert {"subtract", "square", "reduce_mean"} <= set(names)


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [None, 0, 1, -1], ids=str)
@pytest.mark.parametrize("keepdims", [False, True], ids=str)
def test_median_matches_numpy(axis, keepdims):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = run_numpy(lambda t: etl.median(t, axis=axis, keepdims=keepdims), x)
    ref = np.median(x, axis=axis, keepdims=keepdims)
    assert got.dtype == np.dtype("float64")  # numpy median always float64
    assert np.array_equal(got, ref)


def test_median_even_count_is_mean_of_middle_two():
    got = run_numpy(lambda t: etl.median(t), np.array([1, 2, 3, 4], np.int32))
    assert got.dtype == np.dtype("float64")
    assert got == 2.5


def test_median_scalar_is_float64_identity():
    got = run_numpy(lambda t: etl.median(t), np.array(7.0, np.float32))
    assert got.dtype == np.dtype("float64")
    assert got == 7.0


def test_median_symbolic_extents_raise_shape_error():
    """v1 deferral: the slice composition needs static limit indices."""
    with pytest.raises(etl.ShapeError, match="symbolic extents"):
        etl.trace(
            lambda t: etl.median(t),
            etl.TensorSpec((etl.dim("n"), 3), etl.float32),
        )


def test_median_axis_errors():
    with pytest.raises(etl.ShapeError, match="out of range"):
        etl.trace(
            lambda t: etl.median(t, axis=2), etl.TensorSpec((2, 2), etl.float32)
        )
    with pytest.raises(TypeError, match="axis must be an int or None"):
        etl.trace(
            lambda t: etl.median(t, axis="x"), etl.TensorSpec((2, 2), etl.float32)
        )


# ---------------------------------------------------------------------------
# nansum
# ---------------------------------------------------------------------------


def test_nansum_matches_numpy():
    x = np.array([1.0, np.nan, 3.0, np.nan], np.float32)
    got = run_numpy(lambda t: etl.nansum(t), x)
    assert np.allclose(got, np.nansum(x))


def test_nansum_all_nan_is_zero():
    got = run_numpy(lambda t: etl.nansum(t), np.array([np.nan, np.nan], np.float32))
    assert got == 0.0


def test_nansum_axes_keepdims():
    x = np.array([[1.0, np.nan], [3.0, 4.0]], np.float32)
    got = run_numpy(lambda t: etl.nansum(t, axes=1, keepdims=True), x)
    assert got.shape == (2, 1)
    assert np.allclose(got, np.nansum(x, axis=1, keepdims=True))


def test_nansum_int_promotes_via_reduction():
    got = run_numpy(lambda t: etl.nansum(t), np.array([1, 2, 3], np.int32))
    assert got.dtype == np.dtype("int64")
    assert got == 6


def test_nansum_complex_works():
    x = np.array([1 + 2j, complex(np.nan, 1)], np.complex64)
    got = run_numpy(lambda t: etl.nansum(t), x)
    assert np.allclose(got, np.nansum(x))


# ---------------------------------------------------------------------------
# trace / diagonal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2], ids=str)
def test_trace_offsets(offset):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = run_numpy(lambda t: etl.ops.trace(t, offset=offset), x)
    assert np.allclose(got, np.trace(x, offset=offset))


def test_trace_3d_sums_first_two_axes_diagonal():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    got = run_numpy(lambda t: etl.ops.trace(t), x)
    assert got.shape == np.trace(x).shape
    assert np.allclose(got, np.trace(x))


def test_trace_int_input_promotes_via_reduction():
    got = run_numpy(lambda t: etl.ops.trace(t), np.array([[1, 2], [3, 4]], np.int32))
    assert got.dtype == np.dtype("int64")
    assert got == 5


def test_trace_requires_rank_2():
    with pytest.raises(etl.ShapeError, match="rank >= 2"):
        etl.trace(lambda t: etl.ops.trace(t), etl.TensorSpec((4,), etl.float32))


@pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3], ids=str)
def test_diagonal_offsets(offset):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = run_numpy(lambda t: etl.diagonal(t, offset=offset), x)
    assert np.array_equal(got, np.diagonal(x, offset=offset))


def test_diagonal_axis_pair_on_3d():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    got = run_numpy(lambda t: etl.diagonal(t, axis1=0, axis2=2), x)
    assert got.shape == np.diagonal(x, axis1=0, axis2=2).shape
    assert np.array_equal(got, np.diagonal(x, axis1=0, axis2=2))


def test_diagonal_preserves_dtype():
    got = run_numpy(
        lambda t: etl.diagonal(t), np.array([[1, 2], [3, 4]], np.int32)
    )
    assert got.dtype == np.dtype("int32")


def test_diagonal_errors():
    with pytest.raises(etl.ShapeError, match="rank >= 2"):
        etl.trace(lambda t: etl.diagonal(t), etl.TensorSpec((4,), etl.float32))
    with pytest.raises(etl.ShapeError, match="different"):
        etl.trace(
            lambda t: etl.diagonal(t, axis1=0, axis2=0),
            etl.TensorSpec((2, 2), etl.float32),
        )


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [-1, 0, 1], ids=str)
def test_sort_matches_numpy(axis):
    x = np.array([[3, 1, 2], [6, 4, 5]], np.float32)
    got = run_numpy(lambda t: etl.sort(t, axis=axis), x)
    assert np.array_equal(got, np.sort(x, axis=axis))


def test_sort_axis_none_flattens():
    x = np.array([[3, 1], [6, 4]], np.int32)
    got = run_numpy(lambda t: etl.sort(t, axis=None), x)
    assert got.shape == (4,)
    assert np.array_equal(got, np.sort(x, axis=None))


def test_sort_preserves_dtype():
    got = run_numpy(lambda t: etl.sort(t), np.array([3, 1, 2], np.int32))
    assert got.dtype == np.dtype("int32")


def test_sort_axis_out_of_range():
    with pytest.raises(etl.ShapeError, match="out of range"):
        etl.trace(
            lambda t: etl.sort(t, axis=2), etl.TensorSpec((2, 2), etl.float32)
        )


# ---------------------------------------------------------------------------
# norm (vector-norm semantics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ord", [2, 1, float("inf"), float("-inf")], ids=["2", "1", "inf", "-inf"]
)
def test_norm_matches_numpy_vector_norm(ord):
    x = np.array([3.0, 4.0], np.float32)
    got = run_numpy(lambda t: etl.norm(t, ord=ord), x)
    assert np.allclose(got, np.linalg.norm(x, ord=ord))


def test_norm_flat_vector_on_2d():
    """axis=None on rank > 1 reduces over ALL elements (flat-vector norm;
    ord=2 ≡ Frobenius) — documented divergence from numpy's MATRIX norms for
    2-D input (spectral for ord=2)."""
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = run_numpy(lambda t: etl.norm(t, ord=2), x)
    assert np.allclose(got, np.linalg.norm(x.ravel(), ord=2))
    assert np.allclose(got, np.linalg.norm(x, ord="fro"))


def test_norm_axis_tuple_and_keepdims():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    got = run_numpy(lambda t: etl.norm(t, axis=(1, 2), keepdims=True), x)
    ref = np.linalg.norm(x, axis=(1, 2), keepdims=True)
    assert got.shape == ref.shape
    assert np.allclose(got, ref)


def test_norm_ord2_int_input_promotes_to_float64():
    got = run_numpy(lambda t: etl.norm(t), np.array([3, 4], np.int32))
    assert got.dtype == np.dtype("float64")
    assert np.allclose(got, 5.0)


def test_norm_unsupported_ord_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not supported"):
        etl.trace(
            lambda t: etl.norm(t, ord=3), etl.TensorSpec((4,), etl.float32)
        )


# ---------------------------------------------------------------------------
# evox-style usage: stats ops composed inside an @etl.defn graph
# ---------------------------------------------------------------------------


def test_stats_ops_in_defn_graph():
    @etl.defn
    def stats(x):
        return (
            etl.std(x, axes=1),
            etl.median(x, axis=1),
            etl.nansum(x, axes=1),
            etl.norm(x, axes=1),
        )

    x = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], np.float32)
    got = tuple(o.numpy() for o in etl.evaluate(stats, x))
    ref = (
        np.std(x, axis=1),
        np.median(x, axis=1),
        np.nansum(x, axis=1),
        np.linalg.norm(x, axis=1),
    )
    for g, r in zip(got, ref):
        assert np.allclose(g, r, equal_nan=True)
