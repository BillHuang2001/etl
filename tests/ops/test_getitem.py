"""``SymbolicTensor.__getitem__`` — the registered operator handler kind
``getitem`` (``etl/ops/indexing.py::getitem``).

Strictly STATIC indexing: an ``int`` key becomes a ``gather`` op with a 0-d
index constant (dropping the axis); a contiguous ``builtins.slice`` becomes a
``slice`` op; strided slices become ``gather`` ops with numpy's exact static
index array; tuples of ints/slices combine per axis. Everything dynamic
(symbolic tensors, boolean masks, ``None``/newaxis, ellipsis, non-static
index values) raises ``TraceError``; static out-of-bounds int indices raise
``ShapeError`` at trace time.
"""

import numpy as np
import pytest

import etl
from etl import ShapeError, TraceError

from conftest import ops_of, run_numpy, trace_fn


def last_result(graph):
    """The result Value of the last data-producing op of the main function."""
    for op in reversed(ops_of(graph)):
        if op.results:
            return op.results[0]
    raise AssertionError("graph produces no result")


ARR = np.arange(20, dtype=np.float32).reshape(5, 4)


# ---------------------------------------------------------------------------
# int indices -> gather op dropping the axis
# ---------------------------------------------------------------------------

class TestIntIndex:
    @pytest.mark.parametrize("key,expected_shape", [(0, (4,)), (2, (4,))])
    def test_int_index_drops_axis(self, key, expected_shape):
        def f(x):
            return x[key]

        g = trace_fn(f, etl.TensorSpec((3, 4), etl.float32))
        gathers = ops_of(g, "gather")
        assert len(gathers) == 1
        assert gathers[0].attributes["axes"] == (0,)
        const = ops_of(g, "constant")[0]
        assert const.results[0].type.shape == ()
        assert const.attributes["value"].item() == key
        assert last_result(g).type.shape == expected_shape
        np.testing.assert_array_equal(
            run_numpy(f, ARR[:3]), ARR[:3][key]
        )

    def test_negative_int_index(self):
        def f(x):
            return x[-1]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        const = ops_of(g, "constant")[0]
        assert const.attributes["value"].item() == 4  # 5 + (-1)
        assert ops_of(g, "gather")[0].attributes["axes"] == (0,)
        assert last_result(g).type.shape == (4,)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[-1])

    def test_negative_int_index_over_symbolic_dim(self):
        n = etl.dim("n")

        def f(x):
            return x[-1]

        with pytest.raises(ShapeError, match="negative index -1 over symbolic"):
            trace_fn(f, etl.TensorSpec((n, 5), etl.float32))

    def test_out_of_bounds_int_index(self):
        def f(x):
            return x[10]

        with pytest.raises(ShapeError, match="out of range for dim 5"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_out_of_bounds_negative_int_index(self):
        def f(x):
            return x[-6]

        with pytest.raises(ShapeError, match="index -6 out of range for dim 5"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))


# ---------------------------------------------------------------------------
# slice keys -> slice op (contiguous) / gather op (strided)
# ---------------------------------------------------------------------------

class TestSliceIndex:
    def test_contiguous_slice_becomes_slice_op(self):
        def f(x):
            return x[1:3]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        slices = ops_of(g, "slice")
        assert len(slices) == 1
        s = slices[0]
        assert s.attributes["start_indices"] == (1, 0)
        assert s.attributes["limit_indices"] == (3, 4)
        assert s.attributes["strides"] == (1, 1)
        assert last_result(g).type.shape == (2, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1:3])

    def test_open_ended_bounds(self):
        def f(x):
            return x[:2]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (0, 0)
        assert s.attributes["limit_indices"] == (2, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[:2])

    def test_stop_none_uses_axis_dim(self):
        def f(x):
            return x[1:]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (1, 0)
        assert s.attributes["limit_indices"] == (5, 4)
        assert last_result(g).type.shape == (4, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1:])

    def test_out_of_bounds_bounds_clamp(self):
        def f(x):
            return x[1:10]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (1, 0)
        assert s.attributes["limit_indices"] == (5, 4)  # numpy clamps
        assert last_result(g).type.shape == (4, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1:10])

    def test_negative_bounds_count_from_end(self):
        def f(x):
            return x[-2:]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (3, 0)
        assert s.attributes["limit_indices"] == (5, 4)
        assert last_result(g).type.shape == (2, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[-2:])

    def test_negative_stop(self):
        def f(x):
            return x[1:-1]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (1, 0)
        assert s.attributes["limit_indices"] == (4, 4)
        assert last_result(g).type.shape == (3, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1:-1])

    def test_reversed_bounds_yield_empty(self):
        def f(x):
            return x[2:1]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        assert last_result(g).type.shape == (0, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[2:1])

    def test_full_axis_slice_is_noop(self):
        def f(x):
            return x[:]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        # no data op at all: the function returns the input unchanged
        assert [op.name for op in ops_of(g)] == ["return"]
        ret = ops_of(g)[0]
        assert ret.operands[0].type.shape == (5, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR)

    def test_strided_slice_becomes_gather(self):
        def f(x):
            return x[::2]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        const = ops_of(g, "constant")[0]
        np.testing.assert_array_equal(const.attributes["value"], [0, 2, 4])
        assert ops_of(g, "gather")[0].attributes["axes"] == (0,)
        assert last_result(g).type.shape == (3, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[::2])

    def test_negative_step_slice_becomes_gather(self):
        def f(x):
            return x[::-1]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        const = ops_of(g, "constant")[0]
        np.testing.assert_array_equal(const.attributes["value"], [4, 3, 2, 1, 0])
        assert last_result(g).type.shape == (5, 4)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[::-1])

    def test_zero_step_slice_malformed_param_error(self):
        # BUG(etl): getitem with slice step 0 leaks a raw ValueError
        # ('slice step cannot be zero') from numpy instead of the documented
        # TypeError for malformed static params (ops/CONTEXT.md error semantics).
        def f(x):
            return x[::0]

        with pytest.raises(TypeError):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))


# ---------------------------------------------------------------------------
# tuple keys -> per-axis combination
# ---------------------------------------------------------------------------

class TestTupleIndex:
    def test_full_axis_and_int(self):
        def f(x):
            return x[:, 2]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        gathers = ops_of(g, "gather")
        assert len(gathers) == 1
        assert gathers[0].attributes["axes"] == (1,)
        assert last_result(g).type.shape == (5,)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[:, 2])

    def test_int_and_slice(self):
        def f(x):
            return x[1, 2:4]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        gathers = ops_of(g, "gather")
        assert len(gathers) == 1 and gathers[0].attributes["axes"] == (0,)
        s = ops_of(g, "slice")[0]
        assert s.attributes["start_indices"] == (2,)
        assert s.attributes["limit_indices"] == (4,)
        assert last_result(g).type.shape == (2,)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1, 2:4])

    def test_two_ints_produce_scalar(self):
        def f(x):
            return x[0, 0]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        assert len(ops_of(g, "gather")) == 2
        assert last_result(g).type.shape == ()
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[0, 0])

    def test_strided_and_int(self):
        def f(x):
            return x[::2, 1]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        assert last_result(g).type.shape == (3,)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[::2, 1])

    def test_contiguous_and_strided(self):
        def f(x):
            return x[1:3, ::2]

        g = trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
        const = ops_of(g, "constant")[0]
        np.testing.assert_array_equal(const.attributes["value"], [0, 2])
        assert last_result(g).type.shape == (2, 2)
        np.testing.assert_array_equal(run_numpy(f, ARR), ARR[1:3, ::2])

    def test_too_many_indices(self):
        def f(x):
            return x[0, 0, 0]

        with pytest.raises(ShapeError, match="too many indices \\(3\\) for rank 2"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))


# ---------------------------------------------------------------------------
# symbolic dims
# ---------------------------------------------------------------------------

class TestSymbolicDims:
    def test_int_index_result_shape(self):
        n = etl.dim("n")

        def f(x):
            return x[1]

        g = trace_fn(f, etl.TensorSpec((n, 5), etl.float32))
        assert last_result(g).type.shape == (5,)

    def test_slice_result_shape_keeps_symbolic_dims(self):
        n = etl.dim("n")

        def f(x):
            return x[1:3]

        g = trace_fn(f, etl.TensorSpec((n, 5), etl.float32))
        assert last_result(g).type.shape == (2, 5)

        def g1(x):
            return x[2:3]

        gr = trace_fn(g1, etl.TensorSpec((n, 5), etl.float32))
        assert last_result(gr).type.shape == (1, 5)

    def test_full_axis_slice_over_symbolic_dim_not_expressible(self):
        n, m = etl.dim("n"), etl.dim("m")

        def f(x):
            return x[2:3]

        with pytest.raises(
            TraceError, match="full-axis slice over symbolic dim"
        ):
            trace_fn(f, etl.TensorSpec((n, m), etl.float32))

    def test_negative_slice_bound_over_symbolic_dim(self):
        n = etl.dim("n")

        def f(x):
            return x[-2:]

        with pytest.raises(TraceError, match="negative slice bound"):
            trace_fn(f, etl.TensorSpec((n, 5), etl.float32))

    def test_strided_slice_over_symbolic_dim(self):
        n = etl.dim("n")

        def f(x):
            return x[::2]

        with pytest.raises(TraceError, match="strided slice over symbolic dim"):
            trace_fn(f, etl.TensorSpec((n, 5), etl.float32))

    def test_dim_slice_bound_is_not_static(self):
        # etl.dim objects are NOT static slice bounds: the contract requires
        # plain ints (or None); a Dim bound raises TraceError.
        n = etl.dim("n")

        def f(x):
            return x[1:n]

        with pytest.raises(TraceError, match="slice bounds must be static ints"):
            trace_fn(f, etl.TensorSpec((n, 5), etl.float32))

    def test_dim_stop_is_not_static(self):
        n = etl.dim("n")

        def f(x):
            return x[:n]

        with pytest.raises(TraceError, match="slice bounds must be static ints"):
            trace_fn(f, etl.TensorSpec((n, 5), etl.float32))


# ---------------------------------------------------------------------------
# non-static / unsupported keys -> TraceError
# ---------------------------------------------------------------------------

class TestUnsupportedKeys:
    def test_newaxis_unsupported(self):
        def f(x):
            return x[None]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_ellipsis_unsupported(self):
        def f(x):
            return x[..., 1]

        with pytest.raises(TraceError, match="not supported"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_boolean_mask_symbolic_tensor_unsupported(self):
        def f(x):
            return x[etl.greater(x, 1.0)]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_ndarray_key_unsupported(self):
        def f(x):
            return x[np.array([True, False, True, False, True])]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_float_index_unsupported(self):
        def f(x):
            return x[1.0]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_bool_index_unsupported(self):
        def f(x):
            return x[True]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_dim_index_unsupported(self):
        def f(x):
            return x[etl.dim("k")]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))

    def test_symbolic_tensor_index_unsupported(self):
        def f(x, i):
            return x[i, 1]

        with pytest.raises(TraceError, match=r"symbolic \(runtime\) indices"):
            trace_fn(
                f,
                etl.TensorSpec((5, 4), etl.float32),
                etl.TensorSpec((), etl.int64),
            )

    def test_numpy_integer_index_unsupported_v1(self):
        # numpy scalars are NOT static Python values in v1 (see
        # etl/trace/trace.py::_is_static_value) — an np.int64 key is rejected
        # as a non-static index value.
        def f(x):
            return x[np.int64(1)]

        with pytest.raises(TraceError, match="unsupported index key"):
            trace_fn(f, etl.TensorSpec((5, 4), etl.float32))
