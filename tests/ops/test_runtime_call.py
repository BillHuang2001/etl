"""Contract tests for ``etl.runtime_call`` — the explicit Python escape hatch.

Assertions on the contract documented in ``etl/ops/CONTEXT.md`` (constant.py
section, ``runtime_call``), ``etl/ops/constant.py`` and
``etl/backends/numpy/CONTEXT.md``:

- builds a ``runtime_call`` IR op with effect ``callback``; the callback is
  carried as a registered-id STRING attribute (JSON-safe) and the declared
  outputs as a ``result_specs`` tuple of ``ir.ValueType`` entries;
  result dtype/shape come from the user-supplied ``TensorSpec``(s).
- the numpy backend executes the callback synchronously at the op position
  with operand arrays (single and multiple outputs; Python scalar operands
  are auto-promoted to 0-d Constant ops).
- ``TypeError`` for a non-callable callback, malformed/empty/missing result
  specs; ``core.TraceError`` for concrete-Tensor operands and use outside a
  trace. Backend REJECTION policy (stablehlo) is tested in tests/backends.
"""
import importlib

import numpy as np
import pytest

import etl
from etl import ir

from tests.ops.conftest import ops_of, run_numpy

# Module-level access to the callback registry (``etl.ops.constant`` as an
# attribute is the re-exported function; see test_constant.py).
constant_module = importlib.import_module("etl.ops.constant")


# ---------------------------------------------------------------------------
# IR construction: op name, effect, callback attr, result_specs, result types
# ---------------------------------------------------------------------------


def test_single_output_ir_contract():
    callback = lambda a: a * 2 + 1  # noqa: E731
    spec = etl.TensorSpec((2, 3), etl.float32)
    captured = {}

    def f(x):
        out = etl.runtime_call(callback, x, result=spec)
        captured["out"] = out
        return out

    graph = etl.trace(f, etl.TensorSpec((2, 3), etl.float32))
    (op,) = ops_of(graph, "runtime_call")

    # Effect annotation: backends may reject but never silently reorder it.
    assert op.name == "runtime_call"
    assert op.effect == "callback"

    # The callback attribute is a registered-id STRING (JSON-safe), which
    # resolves back to the exact callable in the ops-level registry.
    callback_id = op.attributes["callback"]
    assert isinstance(callback_id, str)
    assert constant_module._get_callback(callback_id) is callback

    # result_specs: a tuple of ir.ValueType entries exactly equal to the
    # op's result types.
    result_specs = op.attributes["result_specs"]
    assert isinstance(result_specs, tuple)
    assert len(result_specs) == 1
    assert all(isinstance(entry, ir.ValueType) for entry in result_specs)
    assert result_specs[0].dtype == np.dtype("float32")
    assert result_specs[0].shape == (2, 3)

    # The op has one result per declared spec, typed by the spec.
    (result,) = op.results
    assert result.type.dtype == np.dtype("float32")
    assert result.type.shape == (2, 3)

    # The SymbolicTensor wrapper agrees with the IR value type.
    sym = captured["out"]
    assert isinstance(sym, etl.SymbolicTensor)
    assert sym.dtype == np.dtype("float32")
    assert sym.shape == (2, 3)

    # The single operand is the traced input block argument.
    (block_arg,) = graph.module.functions[0].region.blocks[0].arguments
    assert op.operands == (block_arg,)


def test_multiple_outputs_ir_contract():
    callback = lambda a: (a + 1, a * 2)  # noqa: E731
    specs = (
        etl.TensorSpec((3,), etl.float32),
        etl.TensorSpec((3,), etl.int64),
    )
    captured = {}

    def f(x):
        out = etl.runtime_call(callback, x, result=specs)
        captured["out"] = out
        return out

    graph = etl.trace(f, etl.TensorSpec((3,), etl.float32))
    (op,) = ops_of(graph, "runtime_call")

    assert op.effect == "callback"
    assert isinstance(op.attributes["callback"], str)
    assert len(op.results) == 2
    result_specs = op.attributes["result_specs"]
    assert isinstance(result_specs, tuple)
    assert [entry.dtype for entry in result_specs] == [
        np.dtype("float32"),
        np.dtype("int64"),
    ]
    assert [entry.shape for entry in result_specs] == [(3,), (3,)]
    assert [(r.type.dtype, r.type.shape) for r in op.results] == [
        (np.dtype("float32"), (3,)),
        (np.dtype("int64"), (3,)),
    ]

    # A tuple of SymbolicTensors comes back from the call.
    syms = captured["out"]
    assert isinstance(syms, tuple)
    assert len(syms) == 2
    assert all(isinstance(s, etl.SymbolicTensor) for s in syms)
    assert [s.dtype for s in syms] == [np.dtype("float32"), np.dtype("int64")]


def test_scalar_operands_are_promoted_to_0d_constants():
    callback = lambda a, b: a * b  # noqa: E731

    def f(x):
        return etl.runtime_call(callback, x, 3, result=etl.TensorSpec((2,), etl.float64))

    graph = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    (op,) = ops_of(graph, "runtime_call")
    assert len(op.operands) == 2

    # The Python scalar operand became a 0-d int64 Constant op feeding the
    # runtime_call (transparent scalar-promotion sugar).
    (scalar_op,) = ops_of(graph, "constant")
    assert scalar_op.results[0].type.dtype == np.dtype("int64")
    assert scalar_op.results[0].type.shape == ()
    assert op.operands[1] is scalar_op.results[0]


# ---------------------------------------------------------------------------
# Execution through the numpy backend
# ---------------------------------------------------------------------------


def test_single_output_execution_matches_callback():
    def callback(a):
        return a * 2 + 1

    def f(x):
        return etl.runtime_call(callback, x, result=etl.TensorSpec((2, 2), etl.float32))

    x = np.arange(4, dtype=np.float32).reshape(2, 2)
    out = run_numpy(f, x)
    np.testing.assert_array_equal(out, callback(x))


def test_single_output_execution_with_extra_static_capture():
    # The callback may close over Python values (they specialize the graph;
    # the callback itself is not analyzed).
    scale = np.float32(3.0)

    def callback(a):
        return a * scale

    def f(x):
        return etl.runtime_call(callback, x, result=etl.TensorSpec((4,), etl.float32))

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = run_numpy(f, x)
    np.testing.assert_array_equal(out, x * scale)


def test_multiple_outputs_execution_returns_tuple():
    def callback(a):
        return a + 1, a * 2

    def f(x):
        return etl.runtime_call(
            callback,
            x,
            result=(
                etl.TensorSpec((3,), etl.float32),
                etl.TensorSpec((3,), etl.float32),
            ),
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = etl.evaluate(f, x)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(isinstance(o, etl.Tensor) for o in out)
    expected_a, expected_b = callback(x)
    np.testing.assert_array_equal(out[0].numpy(), expected_a)
    np.testing.assert_array_equal(out[1].numpy(), expected_b)


def test_scalar_operand_execution():
    def callback(a, b):
        return a * b

    def f(x):
        return etl.runtime_call(callback, x, 3, result=etl.TensorSpec((2,), etl.float64))

    out = run_numpy(f, np.array([1.0, 2.0], dtype=np.float32))
    # float32 array * int64 array -> float64 (numpy array promotion); the
    # declared spec documents the exact callback output dtype.
    assert out.dtype == np.dtype("float64")
    np.testing.assert_array_equal(out, np.array([3.0, 6.0]))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_non_callable_callback_raises_typeerror():
    def f(x):
        return etl.runtime_call(3.14, x, result=etl.TensorSpec((2,), etl.float32))

    with pytest.raises(TypeError, match="callback must be callable"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


@pytest.mark.parametrize(
    "bad_result",
    [42, 1.5, "spec", None, True],
    ids=["int", "float", "str", "none", "bool"],
)
def test_result_not_a_tensorspec_raises_typeerror(bad_result):
    def f(x):
        return etl.runtime_call(lambda a: a, x, result=bad_result)

    with pytest.raises(TypeError, match="result must be a TensorSpec"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


def test_result_tuple_with_non_spec_entry_raises_typeerror():
    def f(x):
        return etl.runtime_call(
            lambda a: a,
            x,
            result=(etl.TensorSpec((2,), etl.float32), 42),
        )

    with pytest.raises(TypeError, match="non-empty tuple/list of TensorSpecs"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


def test_result_empty_tuple_raises_typeerror():
    def f(x):
        return etl.runtime_call(lambda a: a, x, result=())

    with pytest.raises(TypeError, match="non-empty"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


def test_missing_result_spec_raises_typeerror():
    # ``result`` is a required keyword-only parameter: omitting it is a
    # Python-level TypeError raised before any tracing can begin.
    with pytest.raises(TypeError, match="result"):
        etl.runtime_call(lambda a: a)


def test_concrete_tensor_operand_raises_traceerror():
    captured = etl.tensor(np.ones(2, dtype=np.float32))

    def f(x):
        return etl.runtime_call(
            lambda a, b: a + b,
            x,
            captured,
            result=etl.TensorSpec((2,), etl.float32),
        )

    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


def test_runtime_call_outside_trace_raises_traceerror():
    with pytest.raises(etl.TraceError, match="No active trace"):
        etl.runtime_call(lambda a: a, result=etl.TensorSpec((2,), etl.float32))
