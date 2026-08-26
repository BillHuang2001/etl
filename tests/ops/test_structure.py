"""Structure ops: broadcast, reshape, transpose, slice, gather, scatter,
concatenate, pad, select.

Each op is exercised for (1) static shape inference — including symbolic
``Dim``/``DimExpr`` dims — (2) its documented error contract (``ShapeError``
for static failures, ``DTypeError`` for bad dtypes, ``TypeError`` for
malformed static params; see ``etl/ops/CONTEXT.md``), and (3) numerics against
the matching numpy reference on the default numpy backend.
"""

import numpy as np
import pytest

import etl
from etl import DTypeError, ShapeError

from conftest import ops_of, run_numpy, trace_fn

#: Module-level symbolic dims for parametrize lists (Dims with equal names
#: compare equal, so tests may build their own ``etl.dim("n")`` as well).
N = etl.dim("n")


def last_result(graph):
    """The result Value of the last data-producing op of the main function."""
    for op in reversed(ops_of(graph)):
        if op.results:
            return op.results[0]
    raise AssertionError("graph produces no result")


def np_scatter_ref(x, indices, updates, axis=0):
    """numpy reference for ``scatter``: functional ``put_along_axis``.

    Mirrors the numpy backend kernel (``etl/backends/numpy/kernels/
    indexing.py::_scatter``): copy the input, normalize index rank (0-d →
    all-ones of full rank; low-rank → 1-padded around the axis), then
    ``np.put_along_axis`` (overwrite semantics: last write wins on
    duplicated indices).
    """
    out = np.array(x, copy=True)
    idx = np.asarray(indices)
    axis = int(axis)
    if axis < 0:
        axis += out.ndim
    if idx.ndim == 0:
        idx = idx.reshape((1,) * out.ndim)
    elif idx.ndim < out.ndim:
        idx = idx.reshape(
            (1,) * axis + idx.shape + (1,) * (out.ndim - axis - 1)
        )
    np.put_along_axis(out, idx, updates, axis=axis)
    return out


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------

class TestBroadcast:
    @pytest.mark.parametrize(
        "src_shape,target",
        [
            ((4,), (2, 3, 4)),  # rank-increasing broadcast
            ((1, 4), (3, 4)),  # leading 1 expands
            ((3, 1), (3, 5)),  # inner 1 expands
            ((), (2, 3)),  # scalar to full target
        ],
    )
    def test_numerics_vs_broadcast_to(self, src_shape, target):
        def f(x):
            return etl.broadcast(x, target)

        g = trace_fn(f, etl.TensorSpec(src_shape, etl.float32))
        assert last_result(g).type.shape == target
        arr = np.arange(
            1 if not src_shape else np.prod(src_shape), dtype=np.float32
        ).reshape(src_shape)
        np.testing.assert_array_equal(
            run_numpy(f, arr), np.broadcast_to(arr, target)
        )

    def test_dtype_preserved(self):
        def f(x):
            return etl.broadcast(x, (2, 3))

        g = trace_fn(f, etl.TensorSpec((3,), etl.int32))
        assert last_result(g).type.dtype == np.dtype("int32")
        out = run_numpy(f, np.array([1, 2, 3], np.int32))
        assert out.dtype == np.dtype("int32")

    def test_scalar_literal_operand(self):
        def f():
            return etl.broadcast(3, (2, 2))

        g = trace_fn(f)
        assert last_result(g).type.shape == (2, 2)
        assert last_result(g).type.dtype == np.dtype("int64")
        assert ops_of(g, "constant") and ops_of(g, "broadcast")
        np.testing.assert_array_equal(
            run_numpy(f), np.broadcast_to(3, (2, 2))
        )

    @pytest.mark.parametrize(
        "src_shape,target,match",
        [
            ((3, 4), (5, 4), "operand dim 3 cannot expand to target dim 5"),
            ((3, 4), (3, 5, 4), "operand dim 3 cannot expand to target dim 5"),
            ((2, 4), (2,), "target rank 1 < operand rank 2"),
            ((3, 4), (3, -1), "invalid shape dim -1"),
        ],
    )
    def test_static_errors(self, src_shape, target, match):
        def f(x):
            return etl.broadcast(x, target)

        with pytest.raises(ShapeError, match=match):
            trace_fn(f, etl.TensorSpec(src_shape, etl.float32))

    def test_shape_must_be_tuple(self):
        def f(x):
            return etl.broadcast(x, [3, 4])

        with pytest.raises(TypeError, match="shape must be a tuple"):
            trace_fn(f, etl.TensorSpec((3, 4), etl.float32))

    def test_symbolic_target_dims(self):
        n = etl.dim("n")

        def f(x):
            return etl.broadcast(x, (2, n, 4))

        g = trace_fn(f, etl.TensorSpec((n, 4), etl.float32))
        assert last_result(g).type.shape == (2, n, 4)

        def g1(x):
            return etl.broadcast(x, (n, 4))

        gr = trace_fn(g1, etl.TensorSpec((1, 4), etl.float32))
        assert last_result(gr).type.shape == (n, 4)


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------

class TestReshape:
    @pytest.mark.parametrize(
        "src,target",
        [((4, 6), (2, 12)), ((4, 6), (24,)), ((2, 3, 4), (6, 4)), ((), (1, 1, 1))],
    )
    def test_numerics_vs_reshape(self, src, target):
        def f(x):
            return etl.reshape(x, target)

        g = trace_fn(f, etl.TensorSpec(src, etl.float32))
        assert last_result(g).type.shape == target
        arr = np.arange(
            1 if not src else np.prod(src), dtype=np.float32
        ).reshape(src)
        np.testing.assert_array_equal(run_numpy(f, arr), np.reshape(arr, target))

    @pytest.mark.parametrize(
        "src,target,expected",
        [
            ((4, 6), (2, -1), (2, 12)),
            ((4, 6), (3, 2, -1), (3, 2, 4)),
            ((4, 6), (-1,), (24,)),
        ],
    )
    def test_minus_one_inferred(self, src, target, expected):
        def f(x):
            return etl.reshape(x, target)

        g = trace_fn(f, etl.TensorSpec(src, etl.float32))
        assert last_result(g).type.shape == expected

    def test_dim_expr_arithmetic(self):
        n = etl.dim("n")

        def f(x):
            return etl.reshape(x, (n * 2, 3))

        g = trace_fn(f, etl.TensorSpec((n, 6), etl.float32))
        assert last_result(g).type.shape == (n * 2, 3)

    @pytest.mark.parametrize(
        "target,expected",
        [
            ((N, 2, -1), (N, 2, 3)),
            ((N, -1), (N, 6)),
            ((-1, N), (6, N)),
        ],
    )
    def test_symbolic_minus_one_inferred(self, target, expected):
        n = etl.dim("n")

        def f(x):
            return etl.reshape(x, target)

        g = trace_fn(f, etl.TensorSpec((n, 6), etl.float32))
        assert last_result(g).type.shape == expected

    def test_symbolic_minus_one_dim_expr_quotient(self):
        n = etl.dim("n")

        def f(x):
            return etl.reshape(x, (5, -1))

        g = trace_fn(f, etl.TensorSpec((n, 6), etl.float32))
        # element count n*6 over known target count 5 -> DimExpr quotient.
        assert last_result(g).type.shape == (5, (6 * n) // 5)

    @pytest.mark.parametrize(
        "src,target,match",
        [
            ((4, 6), (5, -1), "not divisible by the known target count 5"),
            ((3, 4), (5, 2), "element counts differ: 12 vs 10"),
            ((4, 6), (-1, -1), "at most one -1 wildcard allowed"),
            ((4, 6), (-2, 12), "invalid shape dim -2"),
        ],
    )
    def test_static_errors(self, src, target, match):
        def f(x):
            return etl.reshape(x, target)

        with pytest.raises(ShapeError, match=match):
            trace_fn(f, etl.TensorSpec(src, etl.float32))

    def test_shape_must_be_tuple(self):
        def f(x):
            return etl.reshape(x, [2, 12])

        with pytest.raises(TypeError, match="shape must be a tuple"):
            trace_fn(f, etl.TensorSpec((4, 6), etl.float32))


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_none_reverses_axes(self):
        def f(x):
            return etl.transpose(x)

        g = trace_fn(f, etl.TensorSpec((2, 3, 4), etl.float32))
        assert last_result(g).type.shape == (4, 3, 2)
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        np.testing.assert_array_equal(run_numpy(f, arr), np.transpose(arr))

    @pytest.mark.parametrize("axes", [(1, 2, 0), (2, 0, 1), (0, 2, 1)])
    def test_explicit_permutation(self, axes):
        def f(x):
            return etl.transpose(x, axes)

        src = (2, 3, 4)
        g = trace_fn(f, etl.TensorSpec(src, etl.float32))
        assert last_result(g).type.shape == tuple(src[a] for a in axes)
        arr = np.arange(24, dtype=np.float32).reshape(src)
        np.testing.assert_array_equal(
            run_numpy(f, arr), np.transpose(arr, axes)
        )

    def test_symbolic_dims_permuted(self):
        n = etl.dim("n")

        def f(x):
            return etl.transpose(x, (2, 0, 1))

        g = trace_fn(f, etl.TensorSpec((n, 3, 4), etl.float32))
        assert last_result(g).type.shape == (4, n, 3)

    @pytest.mark.parametrize(
        "axes,match",
        [
            ((1, 0), "invalid permutation"),  # wrong length
            ((0, 1, 1), "not a permutation"),  # duplicates
            ((0, 1, 3), "not a permutation"),  # out of range
        ],
    )
    def test_invalid_permutation(self, axes, match):
        def f(x):
            return etl.transpose(x, axes)

        with pytest.raises(ShapeError, match=match):
            trace_fn(f, etl.TensorSpec((2, 3, 4), etl.float32))

    def test_axes_must_be_none_or_tuple(self):
        def f(x):
            return etl.transpose(x, 1)

        with pytest.raises(TypeError, match="axes must be None or a tuple"):
            trace_fn(f, etl.TensorSpec((2, 3, 4), etl.float32))


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------

class TestSlice:
    @pytest.mark.parametrize(
        "start,lengths,strides,ref",
        [
            (0, (4, 6), 1, (slice(0, 4), slice(0, 6))),
            ((1, 2), (2, 3), (1, 1), (slice(1, 3), slice(2, 5))),
            (0, 2, 1, (slice(0, 2), slice(0, 2))),  # scalar broadcast to axes
            ((0, 0), (4, 6), (2, 2), (slice(0, 4, 2), slice(0, 6, 2))),
            (-1, 1, 1, (slice(3, 4), slice(5, 6))),  # Nx negative starts
        ],
    )
    def test_numerics_vs_numpy_slicing(self, start, lengths, strides, ref):
        def f(x):
            return etl.slice(x, start, lengths, strides=strides)

        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        g = trace_fn(f, etl.TensorSpec((4, 6), etl.float32))
        expected = arr[ref]
        assert last_result(g).type.shape == expected.shape
        np.testing.assert_array_equal(run_numpy(f, arr), expected)

    def test_shape_inference_stride_formula(self):
        # per dim: (length + stride - 1) // stride
        def f(x):
            return etl.slice(x, 0, (4, 6), strides=(1, 2))

        g = trace_fn(f, etl.TensorSpec((4, 6), etl.float32))
        assert last_result(g).type.shape == (4, 3)

    def test_empty_result(self):
        def f(x):
            return etl.slice(x, (4, 0), (0, 6))

        g = trace_fn(f, etl.TensorSpec((4, 6), etl.float32))
        assert last_result(g).type.shape == (0, 6)

    def test_symbolic_dim_with_static_bounds(self):
        n = etl.dim("n")

        def f(x):
            return etl.slice(x, (0, 0), (2, 6))

        g = trace_fn(f, etl.TensorSpec((n, 6), etl.float32))
        # axis 0: static length 2 over symbolic n; axis 1: full slice preserved.
        assert last_result(g).type.shape == (2, 6)

    @pytest.mark.parametrize(
        "start,lengths,strides,match",
        [
            (0, -1, 1, "lengths must be non-negative ints"),
            (0, 2, 0, "strides must be positive ints"),
            (0, 2, -2, "strides must be positive ints"),
            (3, 2, 1, "limit 5 exceeds dim 4"),
            (5, 0, 1, "start 5 exceeds dim 4"),
            ((0, 1, 2), 2, 1, "expected 2 entries"),
        ],
    )
    def test_bad_params(self, start, lengths, strides, match):
        def f(x):
            return etl.slice(x, start, lengths, strides=strides)

        with pytest.raises(ShapeError, match=match):
            trace_fn(f, etl.TensorSpec((4, 6), etl.float32))

    def test_symbolic_length_not_expressible(self):
        n = etl.dim("n")

        def f(x):
            return etl.slice(x, 0, n)

        with pytest.raises(ShapeError, match="symbolic length"):
            trace_fn(f, etl.TensorSpec((4, 6), etl.float32))

    def test_negative_start_over_symbolic_dim(self):
        n = etl.dim("n")

        def f(x):
            return etl.slice(x, -1, 2)

        with pytest.raises(ShapeError, match="negative start -1 over symbolic"):
            trace_fn(f, etl.TensorSpec((n, 6), etl.float32))


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

class TestGather:
    @pytest.mark.parametrize(
        "src,idx_shape,axis,expected",
        [
            ((5, 4), (3,), 0, (3, 4)),
            ((3, 5), (2, 2), 1, (3, 2, 2)),
            ((3, 5), (2,), -1, (3, 2)),
            ((5, 4), (), 0, (4,)),
        ],
    )
    def test_shape_inference(self, src, idx_shape, axis, expected):
        def f(x, i):
            return etl.gather(x, i, axis=axis)

        g = trace_fn(
            f,
            etl.TensorSpec(src, etl.float32),
            etl.TensorSpec(idx_shape, etl.int64),
        )
        assert last_result(g).type.shape == expected

    @pytest.mark.parametrize(
        "axis,src,indices",
        [
            (0, (5, 4), np.array([4, 1, 2], dtype=np.int64)),
            (-1, (3, 5), np.array([4, 1], dtype=np.int64)),
        ],
    )
    def test_numerics_vs_take(self, axis, src, indices):
        def f(x, i):
            return etl.gather(x, i, axis=axis)

        arr = np.arange(np.prod(src), dtype=np.float32).reshape(src)
        g = trace_fn(
            f,
            etl.TensorSpec(src, etl.float32),
            etl.TensorSpec(indices.shape, etl.int64),
        )
        expected = np.take(arr, indices, axis=axis)
        assert last_result(g).type.shape == expected.shape
        np.testing.assert_array_equal(run_numpy(f, arr, indices), expected)

    def test_zero_dim_indices(self):
        def f(x, i):
            return etl.gather(x, i, axis=0)

        arr = np.arange(20, dtype=np.float32).reshape(5, 4)
        idx = np.asarray(2, dtype=np.int64)
        expected = np.take(arr, idx, axis=0)
        np.testing.assert_array_equal(run_numpy(f, arr, idx), expected)

    def test_symbolic_dims_spliced(self):
        n = etl.dim("n")

        def f(x, i):
            return etl.gather(x, i, axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((n, 5), etl.float32),
            etl.TensorSpec((3,), etl.int64),
        )
        assert last_result(g).type.shape == (3, 5)

    @pytest.mark.parametrize("idx_dtype", [etl.float32, etl.float16, etl.bool_])
    def test_indices_dtype_errors(self, idx_dtype):
        def f(x, i):
            return etl.gather(x, i)

        with pytest.raises(DTypeError, match="indices must be an integer dtype"):
            trace_fn(
                f,
                etl.TensorSpec((5, 4), etl.float32),
                etl.TensorSpec((2,), idx_dtype),
            )

    def test_int32_indices_accepted(self):
        def f(x, i):
            return etl.gather(x, i, axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((5, 4), etl.float32),
            etl.TensorSpec((3,), etl.int32),
        )
        assert last_result(g).type.shape == (3, 4)

    def test_axis_out_of_range(self):
        def f(x, i):
            return etl.gather(x, i, axis=2)

        with pytest.raises(ShapeError, match="axis 2 out of range for rank 2"):
            trace_fn(
                f,
                etl.TensorSpec((3, 5), etl.float32),
                etl.TensorSpec((2,), etl.int64),
            )

    def test_axis_must_be_int(self):
        def f(x, i):
            return etl.gather(x, i, axis=1.5)

        with pytest.raises(TypeError, match="axis must be an int"):
            trace_fn(
                f,
                etl.TensorSpec((3, 5), etl.float32),
                etl.TensorSpec((2,), etl.int64),
            )

    def test_dtype_preserved(self):
        def f(x, i):
            return etl.gather(x, i, axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((5, 4), etl.int32),
            etl.TensorSpec((3,), etl.int64),
        )
        assert last_result(g).type.dtype == np.dtype("int32")


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

class TestScatter:
    @pytest.mark.parametrize(
        "axis,src,idx,updates_shape",
        [
            (0, (5, 4), np.array([1, 3], dtype=np.int64), (2, 4)),
            (1, (3, 4), np.array([3, 0], dtype=np.int64), (3, 2)),
            (-1, (3, 4), np.array([3, 0], dtype=np.int64), (3, 2)),
        ],
    )
    def test_numerics_vs_put_along_axis(self, axis, src, idx, updates_shape):
        def f(x, i, u):
            return etl.scatter(x, i, u, axis=axis)

        arr = np.arange(np.prod(src), dtype=np.float32).reshape(src)
        updates = np.full(updates_shape, -5.0, np.float32)
        g = trace_fn(
            f,
            etl.TensorSpec(src, etl.float32),
            etl.TensorSpec(idx.shape, etl.int64),
            etl.TensorSpec(updates_shape, etl.float32),
        )
        assert last_result(g).type.shape == src
        assert last_result(g).type.dtype == np.dtype("float32")
        expected = np_scatter_ref(arr, idx, updates, axis=axis)
        np.testing.assert_array_equal(run_numpy(f, arr, idx, updates), expected)

    def test_duplicate_indices_last_write_wins(self):
        # The numpy kernel is overwrite semantics (put_along_axis): with a
        # duplicated index the LAST update wins — encode that behavior.
        def f(x, i, u):
            return etl.scatter(x, i, u, axis=0)

        arr = np.arange(15, dtype=np.float32).reshape(5, 3)
        idx = np.array([1, 1], dtype=np.int64)
        updates = np.array([[100, 101, 102], [200, 201, 202]], np.float32)
        out = run_numpy(f, arr, idx, updates)
        expected = np_scatter_ref(arr, idx, updates, axis=0)
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(out[1], updates[1])
        assert not np.array_equal(out[1], updates[0])

    def test_zero_dim_index(self):
        def f(x, i, u):
            return etl.scatter(x, i, u, axis=0)

        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx = np.asarray(1, dtype=np.int64)
        updates = np.full(4, -9.0, np.float32)
        expected = np_scatter_ref(arr, idx, updates, axis=0)
        np.testing.assert_array_equal(run_numpy(f, arr, idx, updates), expected)

    def test_int32_indices_accepted(self):
        def f(x, i, u):
            return etl.scatter(x, i, u, axis=0)

        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx = np.array([0, 2], dtype=np.int32)
        updates = np.full((2, 4), 5.0, np.float32)
        g = trace_fn(
            f,
            etl.TensorSpec((3, 4), etl.float32),
            etl.TensorSpec((2,), etl.int32),
            etl.TensorSpec((2, 4), etl.float32),
        )
        assert last_result(g).type.shape == (3, 4)
        np.testing.assert_array_equal(
            run_numpy(f, arr, idx, updates),
            np_scatter_ref(arr, idx, updates, axis=0),
        )

    def test_updates_cast_to_x_dtype(self):
        def f(x, i, u):
            return etl.scatter(x, i, u, axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((3, 4), etl.float32),
            etl.TensorSpec((2,), etl.int64),
            etl.TensorSpec((2, 4), etl.int32),
        )
        assert ops_of(g, "cast")  # updates int32 -> float32
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx = np.array([0, 2], dtype=np.int64)
        updates = np.full((2, 4), 7, np.int32)
        out = run_numpy(f, arr, idx, updates)
        assert out.dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            out, np_scatter_ref(arr, idx, updates.astype(np.float32), axis=0)
        )

    def test_symbolic_result_shape(self):
        n = etl.dim("n")

        def f(x, i, u):
            return etl.scatter(x, i, u, axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((n, 3), etl.float32),
            etl.TensorSpec((2,), etl.int64),
            etl.TensorSpec((2, 3), etl.float32),
        )
        assert last_result(g).type.shape == (n, 3)
        assert last_result(g).type.dtype == np.dtype("float32")

    @pytest.mark.parametrize(
        "axis,specs,match",
        [
            (  # updates rank matches but dims differ statically
                0,
                ((5, 3), (2,), (3, 3)),
                "updates dim 3 does not match the expected dim 2",
            ),
            (  # axis out of range
                5,
                ((5, 3), (2,), (2, 3)),
                "axis 5 out of range for rank 2",
            ),
        ],
    )
    def test_static_errors(self, axis, specs, match):
        x_shape, i_shape, u_shape = specs

        def f(x, i, u):
            return etl.scatter(x, i, u, axis=axis)

        with pytest.raises(ShapeError, match=match):
            trace_fn(
                f,
                etl.TensorSpec(x_shape, etl.float32),
                etl.TensorSpec(i_shape, etl.int64),
                etl.TensorSpec(u_shape, etl.float32),
            )

    def test_indices_dtype_error(self):
        def f(x, i, u):
            return etl.scatter(x, i, u)

        with pytest.raises(DTypeError, match="indices must be an integer dtype"):
            trace_fn(
                f,
                etl.TensorSpec((5, 3), etl.float32),
                etl.TensorSpec((2,), etl.float32),
                etl.TensorSpec((2, 3), etl.float32),
            )


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------

class TestConcatenate:
    @pytest.mark.parametrize(
        "axis,shapes,expected",
        [
            (0, ((2, 3), (4, 3)), (6, 3)),
            (1, ((2, 3), (2, 5)), (2, 8)),
            (-1, ((2, 3), (2, 5)), (2, 8)),
        ],
    )
    def test_numerics_vs_concatenate(self, axis, shapes, expected):
        def f(x, y):
            return etl.concatenate([x, y], axis=axis)

        g = trace_fn(
            f,
            etl.TensorSpec(shapes[0], etl.float32),
            etl.TensorSpec(shapes[1], etl.float32),
        )
        assert last_result(g).type.shape == expected
        a = np.zeros(shapes[0], np.float32)
        b = np.ones(shapes[1], np.float32)
        np.testing.assert_array_equal(
            run_numpy(f, a, b), np.concatenate([a, b], axis=axis)
        )

    def test_axis_dim_symbolic_sum(self):
        n, m = etl.dim("n"), etl.dim("m")

        def f(x, y):
            return etl.concatenate([x, y], axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((n, 3), etl.float32),
            etl.TensorSpec((m, 3), etl.float32),
        )
        shape = last_result(g).type.shape
        assert shape == (n + m, 3)
        assert isinstance(shape[0], etl.DimExpr) and shape[0].op == "add"

    def test_single_tensor(self):
        def f(x):
            return etl.concatenate([x], axis=0)

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        g = trace_fn(f, etl.TensorSpec((2, 3), etl.float32))
        assert last_result(g).type.shape == (2, 3)
        np.testing.assert_array_equal(run_numpy(f, arr), arr)

    @pytest.mark.parametrize(
        "dtype_a,dtype_b",
        [(np.int32, np.float32), (np.bool_, np.int32), (np.float32, np.float32)],
    )
    def test_dtype_promotion_result_type(self, dtype_a, dtype_b):
        def f(x, y):
            return etl.concatenate([x, y], axis=0)

        g = trace_fn(
            f,
            etl.TensorSpec((2, 3), dtype_a),
            etl.TensorSpec((4, 3), dtype_b),
        )
        expected_dtype = np.result_type(dtype_a, dtype_b)
        assert last_result(g).type.dtype == expected_dtype
        a = np.zeros((2, 3), dtype_a)
        b = np.ones((4, 3), dtype_b)
        out = run_numpy(f, a, b)
        assert out.dtype == expected_dtype
        np.testing.assert_array_equal(out, np.concatenate([a, b], axis=0))

    @pytest.mark.parametrize(
        "error_case,match",
        [
            ("non_axis_mismatch", "dims must agree, got 3 vs 4"),
            ("rank_mismatch", "rank mismatch 2 vs 3"),
            ("axis_out_of_range", "axis 2 out of range for rank 2"),
        ],
    )
    def test_static_errors(self, error_case, match):
        if error_case == "non_axis_mismatch":

            def f(x, y):
                return etl.concatenate([x, y], axis=0)

            specs = (
                etl.TensorSpec((2, 3), etl.float32),
                etl.TensorSpec((2, 4), etl.float32),
            )
        elif error_case == "rank_mismatch":

            def f(x, y):
                return etl.concatenate([x, y], axis=0)

            specs = (
                etl.TensorSpec((2, 3), etl.float32),
                etl.TensorSpec((4, 3, 2), etl.float32),
            )
        else:

            def f(x, y):
                return etl.concatenate([x, y], axis=2)

            specs = (
                etl.TensorSpec((2, 3), etl.float32),
                etl.TensorSpec((2, 3), etl.float32),
            )
        with pytest.raises(ShapeError, match=match):
            trace_fn(f, *specs)

    def test_empty_input(self):
        def f(x):
            return etl.concatenate([], axis=0)

        with pytest.raises(ShapeError, match="at least one tensor"):
            trace_fn(f, etl.TensorSpec((2, 3), etl.float32))

    def test_wrong_container(self):
        def f(x):
            return etl.concatenate(x, axis=0)

        with pytest.raises(TypeError, match="non-empty list/tuple"):
            trace_fn(f, etl.TensorSpec((2, 3), etl.float32))


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------

class TestPad:
    @pytest.mark.parametrize(
        "config,pad_width",
        [
            (((1, 2), (3, 0)), ((1, 2), (3, 0))),
            ((1, 2), ((1, 1), (2, 2))),  # int entries: symmetric pad
            ([(1, 2), (0, 1)], ((1, 2), (0, 1))),  # list config accepted
        ],
    )
    def test_numerics_vs_np_pad(self, config, pad_width):
        def f(x):
            return etl.pad(x, config, value=0.0)

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        g = trace_fn(f, etl.TensorSpec((2, 3), etl.float32))
        expected = np.pad(arr, pad_width, mode="constant", constant_values=0.0)
        assert last_result(g).type.shape == expected.shape
        np.testing.assert_array_equal(run_numpy(f, arr), expected)

    def test_value_casts_to_tensor_dtype_int(self):
        def f(x):
            return etl.pad(x, ((1, 0),), value=2.5)

        arr = np.array([1, 2, 3], np.int32)
        out = run_numpy(f, arr)
        assert out.dtype == np.dtype("int32")
        np.testing.assert_array_equal(
            out, np.pad(arr, (1, 0), mode="constant", constant_values=2.5)
        )

    def test_value_casts_to_tensor_dtype_float(self):
        def f(x):
            return etl.pad(x, ((1, 0),), value=7)

        arr = np.array([1.5, 2.5], np.float32)
        out = run_numpy(f, arr)
        assert out.dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            out, np.pad(arr, (1, 0), mode="constant", constant_values=7.0)
        )

    def test_symbolic_dim_arithmetic(self):
        n = etl.dim("n")

        def f(x):
            return etl.pad(x, ((1, 2), (0, 1)), value=0.0)

        g = trace_fn(f, etl.TensorSpec((n, 3), etl.float32))
        assert last_result(g).type.shape == (n + 3, 4)

    def test_zero_rank_identity(self):
        def f(x):
            return etl.pad(x, (), value=0.0)

        arr = np.asarray(3.0, np.float32)
        g = trace_fn(f, etl.TensorSpec((), etl.float32))
        assert last_result(g).type.shape == ()
        np.testing.assert_array_equal(run_numpy(f, arr), arr)

    @pytest.mark.parametrize(
        "config,value,exc,match",
        [
            (((1, 2),), 0.0, ShapeError, "must have 2 entries"),
            (((1, 2), (0, -1)), 0.0, ShapeError, "negative padding"),
            (((1, 2), (0, 1, 2)), 0.0, ShapeError, "invalid padding entry"),
            (1, 0.0, ShapeError, "config must be a tuple"),
            (((1, 2), (0, 1)), "a", TypeError, "int/float scalar"),
            (((1, 2), (0, 1)), True, TypeError, "int/float scalar"),
        ],
    )
    def test_malformed_config(self, config, value, exc, match):
        def f(x):
            return etl.pad(x, config, value=value)

        with pytest.raises(exc, match=match):
            trace_fn(f, etl.TensorSpec((2, 3), etl.float32))


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

class TestSelect:
    def test_numerics_vs_where(self):
        def f(p, a, b):
            return etl.select(p, a, b)

        pred = np.array([True, False, True, False])
        a = np.arange(4, dtype=np.float32)
        b = np.full(4, -1.0, np.float32)
        g = trace_fn(
            f,
            etl.TensorSpec((4,), etl.bool_),
            etl.TensorSpec((4,), etl.float32),
            etl.TensorSpec((4,), etl.float32),
        )
        assert last_result(g).type.shape == (4,)
        np.testing.assert_array_equal(
            run_numpy(f, pred, a, b), np.where(pred, a, b)
        )

    def test_three_operand_broadcast(self):
        def f(p, a, b):
            return etl.select(p, a, b)

        pred = np.array([[True], [False], [True], [False]])
        a = np.ones((1, 3), np.float32)
        b = np.zeros((4, 3), np.float32)
        g = trace_fn(
            f,
            etl.TensorSpec((4, 1), etl.bool_),
            etl.TensorSpec((1, 3), etl.float32),
            etl.TensorSpec((4, 3), etl.float32),
        )
        assert last_result(g).type.shape == (4, 3)
        np.testing.assert_array_equal(
            run_numpy(f, pred, a, b), np.where(pred, a, b)
        )

    def test_scalar_branches(self):
        def f(p):
            return etl.select(p, 1, 2)

        pred = np.array([True, False, True])
        g = trace_fn(f, etl.TensorSpec((3,), etl.bool_))
        assert last_result(g).type.shape == (3,)
        assert last_result(g).type.dtype == np.dtype("int64")
        np.testing.assert_array_equal(run_numpy(f, pred), np.where(pred, 1, 2))

    def test_python_bool_pred(self):
        def f(a, b):
            return etl.select(True, a, b)

        a = np.arange(4, dtype=np.float32)
        b = np.full(4, -1.0, np.float32)
        g = trace_fn(
            f,
            etl.TensorSpec((4,), etl.float32),
            etl.TensorSpec((4,), etl.float32),
        )
        assert last_result(g).type.shape == (4,)
        np.testing.assert_array_equal(
            run_numpy(f, a, b), np.where(True, a, b)
        )

    def test_all_scalar(self):
        def f():
            return etl.select(True, 1, 2)

        g = trace_fn(f)
        assert last_result(g).type.shape == ()
        assert last_result(g).type.dtype == np.dtype("int64")
        out = run_numpy(f)
        assert out.dtype == np.dtype("int64") and out.item() == 1

    def test_static_broadcast_error(self):
        def f(p, a, b):
            return etl.select(p, a, b)

        with pytest.raises(ShapeError, match="cannot broadcast incompatible"):
            trace_fn(
                f,
                etl.TensorSpec((4,), etl.bool_),
                etl.TensorSpec((4,), etl.float32),
                etl.TensorSpec((3,), etl.float32),
            )

    def test_pred_dtype_error(self):
        def f(p, a, b):
            return etl.select(p, a, b)

        with pytest.raises(DTypeError, match="pred must have bool dtype"):
            trace_fn(
                f,
                etl.TensorSpec((4,), etl.float32),
                etl.TensorSpec((4,), etl.float32),
                etl.TensorSpec((4,), etl.float32),
            )

    def test_python_int_pred_error(self):
        def f(a, b):
            return etl.select(1, a, b)

        with pytest.raises(
            DTypeError, match="bool SymbolicTensor or a Python bool scalar"
        ):
            trace_fn(
                f,
                etl.TensorSpec((3,), etl.float32),
                etl.TensorSpec((3,), etl.float32),
            )

    def test_result_dtype_tensor_plus_tensor(self):
        # result dtype = result_type(on_true, on_false)
        def f(p, a, b):
            return etl.select(p, a, b)

        g = trace_fn(
            f,
            etl.TensorSpec((4,), etl.bool_),
            etl.TensorSpec((4,), etl.int32),
            etl.TensorSpec((4,), etl.float32),
        )
        assert last_result(g).type.dtype == np.dtype("float64")

    def test_result_dtype_weak_scalar_branch(self):
        # int scalar branch weak-promotes toward a float32 branch (NEP 50)
        def f(p, a):
            return etl.select(p, 2, a)

        g = trace_fn(
            f,
            etl.TensorSpec((4,), etl.bool_),
            etl.TensorSpec((4,), etl.float32),
        )
        assert last_result(g).type.dtype == np.dtype("float32")
        const = ops_of(g, "constant")[0]
        assert const.results[0].type.dtype == np.dtype("float32")

    def test_result_dtype_scalar_against_int_tensor(self):
        # int scalar against an int32 tensor -> plain result_type (int64)
        def f(p, a):
            return etl.select(p, a, 0)

        g = trace_fn(
            f,
            etl.TensorSpec((4,), etl.bool_),
            etl.TensorSpec((4,), etl.int32),
        )
        assert last_result(g).type.dtype == np.dtype("int64")

    def test_symbolic_broadcast_dim_expr_max(self):
        # The three-operand broadcast of the broadcast contract: conflicting
        # symbolic dims combine as DimExpr("max", ...) (runtime-checked).
        n, m = etl.dim("n"), etl.dim("m")

        def f(p, a, b):
            return etl.select(p, a, b)

        g = trace_fn(
            f,
            etl.TensorSpec((n, 1, 3), etl.bool_),
            etl.TensorSpec((1, 4, 1), etl.float32),
            etl.TensorSpec((m, 1, 3), etl.float32),
        )
        shape = last_result(g).type.shape
        assert shape == (n.max(m), 4, 3)
        assert isinstance(shape[0], etl.DimExpr) and shape[0].op == "max"
