"""Creation-op tests for `etl.numpy` (enp).

`enp.zeros/ones/full/empty/arange` are GRAPH ops: inside `@etl.defn` they
build `etl.constant` IR ops (the same op kind `etl.constant` builds) and
return `SymbolicTensor`s; outside a trace they raise `TraceError` — no
concrete tensor is ever created, no eager fallback exists. Default dtype is
float32 (documented deviation from numpy's float64); `full`/`arange` with
`dtype=None` use numpy's own inference over the static value/bounds.
"""

import numpy as np
import pytest

import etl
import etl.ir
import etl.numpy as enp


# --- file-local helpers -----------------------------------------------------


def _zero_input(build):
    """Evaluate a zero-input defn whose body is ``build()``."""

    @etl.defn
    def f():
        return build()

    return etl.evaluate(f)


def _trace_capture(build):
    """Trace a zero-input defn, returning ``(graph, captured_output)``."""
    captured = []

    @etl.defn
    def f():
        out = build()
        captured.append(out)
        return out

    graph = etl.trace(f)
    return graph, captured[0]


def _constant_ops(graph):
    return [
        op
        for op in graph.module.functions[0].entry_block.ops
        if op.name == "constant"
    ]


# --- graph-op semantics (IR) ------------------------------------------------


CREATION_BUILDERS = [
    pytest.param(lambda: enp.zeros((2, 3)), id="zeros"),
    pytest.param(lambda: enp.ones(5), id="ones"),
    pytest.param(lambda: enp.full((3,), 7), id="full"),
    pytest.param(lambda: enp.empty((2, 3)), id="empty"),
    pytest.param(lambda: enp.arange(5), id="arange"),
]


@pytest.mark.parametrize("build", CREATION_BUILDERS)
def test_creation_ops_build_symbolic_constant_ir(build):
    graph, out = _trace_capture(build)
    assert isinstance(out, etl.SymbolicTensor)
    graph.verify()
    text = etl.ir.pretty_print(graph.module)
    assert "etl.constant" in text
    assert len(_constant_ops(graph)) == 1


# --- numeric equivalence via etl.evaluate ------------------------------------


def test_zeros_numeric():
    out = _zero_input(lambda: enp.zeros((2, 3)))
    assert isinstance(out, etl.Tensor)
    np.testing.assert_array_equal(out.numpy(), np.zeros((2, 3), dtype=np.float32))
    assert out.dtype == etl.float32


def test_ones_numeric():
    out = _zero_input(lambda: enp.ones(5))
    np.testing.assert_array_equal(out.numpy(), np.ones(5, dtype=np.float32))


def test_ones_dtype_float64():
    out = _zero_input(lambda: enp.ones(5, dtype=etl.float64))
    np.testing.assert_array_equal(out.numpy(), np.ones(5, dtype=np.float64))
    assert out.dtype == etl.float64


def test_full_infers_dtype_from_value():
    out = _zero_input(lambda: enp.full((3,), 7))
    np.testing.assert_array_equal(out.numpy(), np.full((3,), 7))
    # No explicit dtype arg: numpy inference over the static fill value.
    assert out.numpy().dtype == np.result_type(7)


def test_full_explicit_dtype():
    out = _zero_input(lambda: enp.full((2, 2), 2.5, dtype=etl.float32))
    np.testing.assert_array_equal(out.numpy(), np.full((2, 2), 2.5, dtype=np.float32))
    assert out.dtype == etl.float32


def test_zeros_int32():
    out = _zero_input(lambda: enp.zeros(4, dtype=etl.int32))
    np.testing.assert_array_equal(out.numpy(), np.zeros(4, dtype=np.int32))
    assert out.dtype == etl.int32


ARANGE_CASES = [
    pytest.param((5,), {}, np.arange(5), id="single"),
    pytest.param((2, 8), {}, np.arange(2, 8), id="start-stop"),
    pytest.param((2, 8, 2), {}, np.arange(2, 8, 2), id="start-stop-step"),
    pytest.param((0.0, 1.0, 0.25), {}, np.arange(0.0, 1.0, 0.25), id="floats"),
    pytest.param((5,), {"dtype": etl.float32}, np.arange(5, dtype=np.float32), id="dtype"),
    pytest.param((0,), {}, np.arange(0), id="zero-length"),
]


@pytest.mark.parametrize("args, kwargs, ref", ARANGE_CASES)
def test_arange_numeric(args, kwargs, ref):
    out = _zero_input(lambda: enp.arange(*args, **kwargs))
    arr = out.numpy()
    np.testing.assert_array_equal(arr, ref)
    assert arr.dtype == ref.dtype
    assert arr.shape == ref.shape


@pytest.mark.parametrize(
    "build, shape",
    [
        pytest.param(lambda: enp.zeros((2, 3)), (2, 3), id="zeros"),
        pytest.param(lambda: enp.ones(5), (5,), id="ones"),
        pytest.param(lambda: enp.empty(4), (4,), id="empty"),
    ],
)
def test_creation_default_dtype_is_float32(build, shape):
    out = _zero_input(build)
    assert out.dtype == etl.float32
    assert out.numpy().shape == shape


def test_empty_values_unspecified():
    out = _zero_input(lambda: enp.empty((2, 3)))
    arr = out.numpy()
    # Values are unspecified (numpy semantics) — assert shape/dtype only.
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32


def test_empty_constant_embeds_ndarray():
    graph, _ = _trace_capture(lambda: enp.empty((2, 3)))
    graph.verify()
    constants = _constant_ops(graph)
    assert len(constants) == 1
    value = constants[0].attributes["value"]
    assert isinstance(value, np.ndarray)
    assert value.shape == (2, 3)
    assert value.dtype == np.float32


# --- error paths -------------------------------------------------------------


SYMBOLIC_SHAPES = [
    pytest.param((4, etl.Dim("n", 4)), id="dim"),
    pytest.param((4, None), id="none-dim"),
    pytest.param((2 * etl.Dim("n", 4),), id="dimexpr"),
]

SHAPE_BUILDERS = [
    pytest.param(lambda shape: enp.zeros(shape), id="zeros"),
    pytest.param(lambda shape: enp.ones(shape), id="ones"),
    pytest.param(lambda shape: enp.full(shape, 1.0), id="full"),
    pytest.param(lambda shape: enp.empty(shape), id="empty"),
]


@pytest.mark.parametrize("build", SHAPE_BUILDERS)
@pytest.mark.parametrize("shape", SYMBOLIC_SHAPES)
def test_creation_symbolic_shape_raises_trace_error(build, shape):
    with pytest.raises(etl.TraceError, match="symbolic"):
        build(shape)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: enp.arange(etl.Dim("n", 5)), id="start"),
        pytest.param(lambda: enp.arange(2, etl.Dim("n", 5)), id="stop"),
    ],
)
def test_arange_symbolic_bounds_raise_trace_error(build):
    with pytest.raises(etl.TraceError, match="symbolic"):
        build()


@pytest.mark.parametrize("shape", ["abc", 2.5], ids=["str", "float"])
@pytest.mark.parametrize("build", SHAPE_BUILDERS)
def test_creation_non_int_seq_shape_raises_trace_error(build, shape):
    with pytest.raises(etl.TraceError, match="shape must be a Python int"):
        build(shape)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: enp.zeros((2,)), id="zeros"),
        pytest.param(lambda: enp.ones(2), id="ones"),
        pytest.param(lambda: enp.full((2,), 1.0), id="full"),
        pytest.param(lambda: enp.empty(2), id="empty"),
        pytest.param(lambda: enp.arange(3), id="arange"),
    ],
)
def test_creation_ops_outside_trace_raise_trace_error(build):
    # No defn / active builder: these build graph ops, so no concrete tensor
    # is created — call them directly in the test body.
    with pytest.raises(etl.TraceError, match="No active trace"):
        build()
