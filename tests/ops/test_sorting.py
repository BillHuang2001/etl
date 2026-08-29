"""pytest suite for etl sorting ops — ``sort``/``argsort`` numpy parity plus
``topk`` regression coverage.

``sort``/``argsort``: exact parity with ``np.sort``/``np.argsort`` (the numpy
kernels call numpy directly with the same kind, so equality is exact), axis /
``axis=None`` flatten / descending / stable variants, dtype preservation
(``sort`` keeps the operand dtype exactly; ``argsort`` → int64), and the
trace-time error kinds (directing TraceError outside a trace, three-option
TraceError for concrete Tensor operands, TypeError for bad static params,
ShapeError for out-of-range axes).

``topk``: regression coverage for the trace-time ``_sort_axis`` arity bug
(topk crashed with ``TypeError``) and the numpy-exact semantics of the
``topk`` composition (``sort``/``argsort`` + ``gather`` of ``0..k-1``):
values/indices agree with ``np.sort``/``np.argsort`` axis slices, dtype
rules, static ``k <= extent`` validation (``ShapeError``), symbolic extents,
and error kinds.
"""
from __future__ import annotations

import numpy as np
import pytest

import etl
from tests.ops.conftest import run_numpy


def _topk(x, k, axis=-1, largest=True):
    return etl.topk(x, k, axis=axis, largest=largest)


def test_topk_largest_basic():
    """1-D top-k: largest values and their original positions (descending)."""
    x = np.array([3.0, 1.0, 2.0])
    values, indices = run_numpy(lambda t: _topk(t, 2), x)
    assert np.array_equal(values, [3.0, 2.0])
    assert np.array_equal(indices, [0, 2])
    assert values.dtype == x.dtype
    assert indices.dtype == np.int64
    # values[i] == x[indices[i]] — indices index the ORIGINAL tensor.
    assert np.array_equal(values, x[indices])


def test_topk_smallest():
    x = np.array([3.0, 1.0, 2.0])
    values, indices = run_numpy(lambda t: _topk(t, 2, largest=False), x)
    ref = np.sort(x)[:2]
    ref_idx = np.argsort(x)[:2]
    assert np.array_equal(values, ref)
    assert np.array_equal(indices, ref_idx)
    assert np.array_equal(values, x[indices])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_topk_2d_matches_sort_argsort_slices(axis):
    """2-D top-k equals the flip-sorted np.sort/np.argsort slices along the
    axis (the documented composition semantics)."""
    x = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    k = 2
    values, indices = run_numpy(lambda t: _topk(t, k, axis=axis), x)
    ref = np.take(np.flip(np.sort(x, axis=axis), axis=axis), np.arange(k), axis=axis)
    ref_idx = np.take(
        np.flip(np.argsort(x, axis=axis), axis=axis), np.arange(k), axis=axis
    )
    assert np.array_equal(values, ref)
    assert np.array_equal(indices, ref_idx)
    assert np.array_equal(values, np.take_along_axis(x, indices, axis=axis))


def test_topk_k_equals_extent_and_zero():
    x = np.array([3.0, 1.0, 2.0])
    values, indices = run_numpy(lambda t: _topk(t, 3), x)
    assert np.array_equal(values, [3.0, 2.0, 1.0])
    assert np.array_equal(indices, [0, 2, 1])
    values, indices = run_numpy(lambda t: _topk(t, 0), x)
    assert values.shape == (0,) and indices.shape == (0,)


def test_topk_k_exceeds_static_extent_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="k=5 exceeds"):
        run_numpy(lambda t: _topk(t, 5), np.array([3.0, 1.0, 2.0]))


@pytest.mark.parametrize("axis", [2, -3])
def test_topk_bad_axis_raises_shape_error(axis):
    with pytest.raises(etl.ShapeError, match="topk"):
        run_numpy(lambda t: _topk(t, 2, axis=axis), np.array([3.0, 1.0, 2.0]))


def test_sort_bad_axis_message_unchanged():
    """The sort call site keeps its unprefixed error text (regression guard
    for the ``_sort_axis`` signature extension)."""
    with pytest.raises(etl.ShapeError) as excinfo:
        run_numpy(lambda t: etl.sort(t, axis=2), np.array([3.0, 1.0, 2.0]))
    assert "sort" not in str(excinfo.value)


def test_topk_symbolic_extent_traces_and_runs():
    """A symbolic (Dim) axis extent must trace without crashing and defer the
    k-vs-extent check to run time."""
    exe = etl.build(
        lambda t: _topk(t, 2)[0],
        etl.TensorSpec((etl.dim("n"),), etl.float32),
    )
    out = etl.run(exe, np.array([3.0, 1.0, 2.0], dtype=np.float32))
    assert np.array_equal(out.numpy(), [3.0, 2.0])


def test_topk_preserves_dtype_indices_int64():
    values, indices = run_numpy(
        lambda t: _topk(t, 2), np.array([3, 1, 2], dtype=np.int16)
    )
    assert values.dtype == np.int16
    assert indices.dtype == np.int64
    assert np.array_equal(values, [3, 2])
    assert np.array_equal(indices, [0, 2])


def test_topk_errors():
    with pytest.raises(TypeError, match="k must be a Python int"):
        run_numpy(lambda t: _topk(t, 2.0), np.array([3.0, 1.0, 2.0]))
    with pytest.raises(etl.ShapeError, match="rank >= 1"):
        run_numpy(lambda t: _topk(t, 1), np.array(3.0))


def test_topk_negative_k_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="k must be >= 0"):
        run_numpy(lambda t: _topk(t, -1), np.array([3.0, 1.0, 2.0]))


def test_topk_largest_non_bool_raises_type_error():
    with pytest.raises(TypeError, match="largest must be a bool"):
        run_numpy(lambda t: _topk(t, 1, largest="yes"), np.array([3.0, 1.0, 2.0]))


def test_topk_bad_axis_type_raises_type_error():
    with pytest.raises(TypeError, match="topk: axis must be an int"):
        run_numpy(lambda t: _topk(t, 1, axis=1.5), np.array([3.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# sort — numpy parity
# ---------------------------------------------------------------------------

def test_sort_1d_default_axis():
    x = np.array([3.0, 1.0, 2.0])
    out = run_numpy(lambda t: etl.sort(t), x)
    assert np.array_equal(out, np.sort(x))
    assert out.shape == x.shape


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_sort_2d_along_axis(axis):
    x = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    out = run_numpy(lambda t: etl.sort(t, axis=axis), x)
    assert np.array_equal(out, np.sort(x, axis=axis))


def test_sort_axis_none_flattens():
    """axis=None sorts the flattened tensor (numpy semantics) — 1-D result."""
    x = np.array([[3.0, 1.0], [2.0, 4.0]])
    out = run_numpy(lambda t: etl.sort(t, axis=None), x)
    assert out.shape == (4,)
    assert np.array_equal(out, np.sort(x, axis=None))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_sort_descending_matches_flip_of_ascending(axis):
    x = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    out = run_numpy(lambda t: etl.sort(t, axis=axis, descending=True), x)
    assert np.array_equal(out, np.flip(np.sort(x, axis=axis), axis=axis))


def test_sort_stable_matches_numpy_stable():
    x = np.array([2.0, 1.0, 1.0, 3.0])
    out = run_numpy(lambda t: etl.sort(t, stable=True), x)
    assert np.array_equal(out, np.sort(x, kind="stable"))


def test_sort_default_kind_matches_numpy_default():
    """stable=False passes numpy's default kind (quicksort) — the kernel calls
    np.sort directly, so the results are bit-identical to np.sort(x)."""
    x = np.array([3.0, 1.0, 2.0])
    out = run_numpy(lambda t: etl.sort(t, stable=False), x)
    assert np.array_equal(out, np.sort(x))


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int32, np.int64],
    ids=["float32", "float64", "int32", "int64"],
)
def test_sort_preserves_dtype_exactly(dtype):
    x = np.array([3, 1, 2], dtype=dtype)
    out = run_numpy(lambda t: etl.sort(t), x)
    assert out.dtype == x.dtype
    assert np.array_equal(out, np.sort(x))


# ---------------------------------------------------------------------------
# argsort — numpy parity
# ---------------------------------------------------------------------------

def test_argsort_1d_default_axis():
    x = np.array([3.0, 1.0, 2.0])
    indices = run_numpy(lambda t: etl.argsort(t), x)
    assert np.array_equal(indices, np.argsort(x))
    assert indices.dtype == np.int64
    # indices index the ORIGINAL tensor and recover the sorted values.
    assert np.array_equal(x[indices], np.sort(x))


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_argsort_2d_along_axis(axis):
    x = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    indices = run_numpy(lambda t: etl.argsort(t, axis=axis), x)
    assert np.array_equal(indices, np.argsort(x, axis=axis))
    assert indices.dtype == np.int64


def test_argsort_axis_none_flattens():
    x = np.array([[3.0, 1.0], [2.0, 4.0]])
    indices = run_numpy(lambda t: etl.argsort(t, axis=None), x)
    assert indices.shape == (4,)
    assert np.array_equal(indices, np.argsort(x, axis=None))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_argsort_descending_matches_flip_of_ascending(axis):
    x = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
    indices = run_numpy(lambda t: etl.argsort(t, axis=axis, descending=True), x)
    assert np.array_equal(indices, np.flip(np.argsort(x, axis=axis), axis=axis))


def test_argsort_stable_keeps_equal_elements_in_original_order():
    """Stable argsort: equal elements keep their original relative order —
    verified explicitly (indices [1, 2, 0, 3]) and against numpy's stable
    kind."""
    x = np.array([2.0, 1.0, 1.0, 3.0])
    indices = run_numpy(lambda t: etl.argsort(t, stable=True), x)
    assert np.array_equal(indices, [1, 2, 0, 3])
    assert np.array_equal(indices, np.argsort(x, kind="stable"))


def test_argsort_default_kind_matches_numpy_default():
    x = np.array([3.0, 1.0, 2.0])
    indices = run_numpy(lambda t: etl.argsort(t, stable=False), x)
    assert np.array_equal(indices, np.argsort(x))


def test_argsort_indices_int64_for_int_input():
    x = np.array([3, 1, 2], dtype=np.int32)
    indices = run_numpy(lambda t: etl.argsort(t), x)
    assert indices.dtype == np.int64
    assert np.array_equal(indices, np.argsort(x))


# ---------------------------------------------------------------------------
# sort / argsort — error paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", ["sort", "argsort"])
def test_sort_argsort_outside_trace_raises_directing_trace_error(op_name):
    with pytest.raises(etl.TraceError) as exc:
        getattr(etl, op_name)(np.array([3.0, 1.0, 2.0]))
    message = str(exc.value)
    assert message.startswith(
        "No active trace: tensor ops can only be called while tracing"
    )
    assert "etl.trace" in message
    assert "etl.evaluate" in message


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
def test_sort_argsort_reject_concrete_tensor_operand_in_trace(op_name):
    tensor = etl.tensor(np.array([3.0, 1.0, 2.0], dtype=np.float32))

    def fn(x):
        return getattr(etl, op_name)(tensor)

    with pytest.raises(etl.TraceError) as exc:
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))
    message = str(exc.value)
    assert message.startswith(
        "Concrete Tensor operands are not allowed in graph ops "
        "(etl has no eager mode)"
    )
    assert "explicit input" in message
    assert "etl.constant" in message
    assert "etl.evaluate" in message


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
@pytest.mark.parametrize("axis", [1.5, "0"], ids=["float", "str"])
def test_sort_argsort_bad_axis_type_raises_type_error(op_name, axis):
    with pytest.raises(TypeError, match="axis must be an int or None"):
        run_numpy(lambda t: getattr(etl, op_name)(t, axis=axis), np.array([3.0, 1.0, 2.0]))


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
@pytest.mark.parametrize("axis", [2, -3], ids=["too-large", "too-negative"])
def test_sort_argsort_axis_out_of_range_raises_shape_error(op_name, axis):
    with pytest.raises(etl.ShapeError, match="out of range for rank"):
        run_numpy(lambda t: getattr(etl, op_name)(t, axis=axis), np.array([3.0, 1.0, 2.0]))


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
def test_sort_argsort_bad_descending_raises_type_error(op_name):
    with pytest.raises(TypeError, match="descending must be a bool"):
        run_numpy(
            lambda t: getattr(etl, op_name)(t, descending="yes"),
            np.array([3.0, 1.0, 2.0]),
        )


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
def test_sort_argsort_bad_stable_raises_type_error(op_name):
    with pytest.raises(TypeError, match="stable must be a bool"):
        run_numpy(
            lambda t: getattr(etl, op_name)(t, stable=1),
            np.array([3.0, 1.0, 2.0]),
        )


@pytest.mark.parametrize("op_name", ["sort", "argsort"])
def test_sort_argsort_scalar_operand_raises_shape_error(op_name):
    """Rank-0 operand: axis normalization fails like numpy's AxisError."""
    with pytest.raises(etl.ShapeError, match="out of range for rank 0"):
        run_numpy(lambda t: getattr(etl, op_name)(t), np.array(3.0))
