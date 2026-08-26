"""Tests for the ``runtime_call`` op executed by the numpy interpreter.

Contract under test (see ``etl/ops/constant.py`` and
``etl/backends/numpy/kernels/custom.py``):

- ``etl.runtime_call(callback, *operands, result=...)`` builds a
  ``runtime_call`` IR op carrying the callback as a STRING registry id. The
  numpy interpreter executes the callback synchronously at the op's position
  (a documented sync point — the op's block position is the effect ordering),
  passing the operands as raw numpy arrays. Python scalar operands are
  promoted to 0-d Constant ops at trace time and arrive as 0-d arrays with
  the natural scalar dtype.
- Outputs are validated against the declared result specs: per-output dtype
  must match exactly (``BackendError`` — no silent coercion), the declared
  shape is compared to the actual output shape (``ShapeError``), and the
  output count must match the declared count (``BackendError``).
- Artifacts carry only the callback id (callbacks are never embedded), so a
  missing registration at run time raises ``BackendError`` naming the id.
"""

import numpy as np
import pytest

import etl


def test_simple_callback_doubles_input():
    invocations = []

    def double(x):
        invocations.append(1)
        return x * 2

    def fn(x):
        return etl.runtime_call(
            double, x, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = etl.evaluate(fn, x)
    assert isinstance(out, etl.Tensor)
    assert out.dtype == np.float32
    assert out.shape == (3,)
    np.testing.assert_array_equal(out.numpy(), x * 2)

    # The callback executes synchronously at the op's position, once per run.
    assert invocations == [1]
    etl.evaluate(fn, x)
    assert len(invocations) == 2


def test_callback_receives_numpy_arrays():
    seen = []

    def record(x):
        seen.append((type(x), x.dtype, x.shape, x.copy()))
        return x

    def fn(x):
        return etl.runtime_call(
            record, x, result=etl.TensorSpec((2, 2), etl.float32)
        )

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = etl.evaluate(fn, x)
    assert len(seen) == 1
    kind, dtype, shape, value = seen[0]
    assert kind is np.ndarray
    assert dtype == np.float32
    assert shape == (2, 2)
    np.testing.assert_array_equal(value, x)
    np.testing.assert_array_equal(out.numpy(), x)


def test_multi_operand_with_python_scalar():
    seen = []

    def add_scalar(x, s):
        seen.append((type(s), s.dtype, s.shape))
        return (x + s).astype(np.float32)

    def fn(x):
        return etl.runtime_call(
            add_scalar, x, 3.0, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = etl.evaluate(fn, x)
    # Python scalar operands are promoted to 0-d Constant ops at trace time
    # and reach the callback as 0-d numpy arrays with the natural scalar
    # dtype (float -> float64).
    assert seen == [(np.ndarray, np.dtype("float64"), ())]
    np.testing.assert_array_equal(out.numpy(), x + 3.0)


def test_multi_output_tuple():
    def split(x):
        total = x.sum(keepdims=True).astype(np.float32)
        return x * 2, total

    def fn(x):
        a, b = etl.runtime_call(
            split,
            x,
            result=(
                etl.TensorSpec((3,), etl.float32),
                etl.TensorSpec((1,), etl.float32),
            ),
        )
        return a, b

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a, b = etl.evaluate(fn, x)
    np.testing.assert_array_equal(a.numpy(), x * 2)
    np.testing.assert_array_equal(b.numpy(), np.array([6.0], dtype=np.float32))


def test_result_dtype_mismatch_raises_backend_error():
    def bad(x):
        return x.astype(np.float64) * 2.0  # float64, declared result is f32

    def fn(x):
        return etl.runtime_call(
            bad, x, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(etl.BackendError, match="dtype"):
        etl.evaluate(fn, x)


def test_result_shape_mismatch_raises_shape_error():
    def bad(x):
        return np.zeros(4, dtype=np.float32)  # (4,), declared result is (3,)

    def fn(x):
        return etl.runtime_call(
            bad, x, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(etl.ShapeError, match="shape"):
        etl.evaluate(fn, x)


def test_wrong_output_count_raises_backend_error():
    def one(x):
        return x  # 1 output, 2 declared

    def fn(x):
        return etl.runtime_call(
            one,
            x,
            result=(
                etl.TensorSpec((3,), etl.float32),
                etl.TensorSpec((3,), etl.float32),
            ),
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(etl.BackendError, match="output"):
        etl.evaluate(fn, x)


def test_non_array_return_raises_backend_error():
    def bad(x):
        return "not a tensor"

    def fn(x):
        return etl.runtime_call(
            bad, x, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(etl.BackendError):
        etl.evaluate(fn, x)


def test_runtime_call_in_a_larger_graph():
    seen = []

    def double(x):
        seen.append(x.copy())
        return x * 2

    # The callback's result feeds into further computation ...
    def fn(x):
        y = etl.runtime_call(
            double, x, result=etl.TensorSpec((3,), etl.float32)
        )
        return etl.add(y, x)

    # ... and the callback can consume operands computed by other ops.
    def fn2(x):
        y = etl.multiply(x, x)
        return etl.runtime_call(
            double, y, result=etl.TensorSpec((3,), etl.float32)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = etl.evaluate(fn, x)
    np.testing.assert_array_equal(out.numpy(), x * 3)

    out2 = etl.evaluate(fn2, x)
    np.testing.assert_array_equal(out2.numpy(), x * x * 2)

    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], x)  # raw input operand
    np.testing.assert_array_equal(seen[1], x * x)  # computed operand


def test_missing_callback_registration_raises_backend_error():
    def cb(x):
        return x

    def fn(x):
        return etl.runtime_call(
            cb, x, result=etl.TensorSpec((3,), etl.float32)
        )

    graph = etl.trace(fn, etl.TensorSpec((3,), etl.float32))
    call_op = next(
        op
        for op in graph.module.get_function("main").entry_block.ops
        if op.name == "runtime_call"
    )
    # The op carries only the string registry id (never the callable itself):
    # rewiring it to an unknown id simulates loading an artifact in a process
    # where the callback was never registered.
    call_op.attributes["callback"] = "callback_registered_nowhere"

    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    executable = etl.load(artifact)
    x = etl.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(etl.BackendError, match="callback_registered_nowhere"):
        etl.run(executable, x)
