"""pytest suite for etl sorting ops — regression coverage for ``topk``.

Focus: the trace-time ``_sort_axis`` arity bug (topk crashed with
``TypeError``) and the numpy-exact semantics of the ``topk`` composition
(``sort``/``argsort`` + ``gather`` of ``0..k-1``): values/indices agree with
``np.sort``/``np.argsort`` axis slices, dtype rules, static ``k <= extent``
validation (``ShapeError``), symbolic extents, and error kinds.
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
