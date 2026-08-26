"""Contract tests for static-value snapshotting in `etl.trace`.

Static Python values (`None`/bool/int/float/complex/str/`Enum`/numpy `dtype`/
`slice` — the `_is_static_value` predicate in `etl/trace/trace.py`)
specialize the graph at TRACE time and are validated at RUN time
(`Graph.flatten_inputs`). Anything else is NOT static in v1 and must be
rejected. These tests pin specialization, run-time validation, trace-time
snapshotting, and rejection of non-static values.
"""

import dataclasses
import enum

import numpy as np
import pytest

import etl


class Mode(enum.Enum):
    A = 1
    B = 2


@dataclasses.dataclass
class Config:
    """A plain config dataclass — NOT a static value in v1."""

    lr: float


class PlainConfig:
    """A plain (non-dataclass) config object — NOT a static value in v1."""

    def __init__(self, lr):
        self.lr = lr


def cast_to(x, dt):
    return etl.cast(x, dt)


def mode_fn(x, mode):
    if mode is Mode.A:
        return etl.multiply(x, x)
    return etl.negate(x)


def slice_fn(x, s):
    # etl.slice takes Nx-style (start, lengths): a Python slice(start, stop)
    # maps to start = s.start, lengths = s.stop - s.start.
    return etl.slice(x, s.start, s.stop - s.start)


def flag_fn(x, flag):
    if flag:
        return etl.add(x, x)
    return etl.negate(x)


def identity_fn(x, cfg):
    return x


# --- dtype static arg ---------------------------------------------------------


def test_dtype_static_arg_specializes_graph(run_graph, as_numpy):
    x = np.array([1, 2, 3], dtype=np.int32)
    g32 = etl.trace(cast_to, etl.TensorSpec((3,), etl.int32), etl.float32)
    g64 = etl.trace(cast_to, etl.TensorSpec((3,), etl.int32), etl.float64)

    # the two traces are different programs
    assert g32.module.main.output_types[0].dtype != g64.module.main.output_types[0].dtype
    assert etl.ir.serialize_module(g32.module)["ops"] != etl.ir.serialize_module(g64.module)["ops"]

    out32 = as_numpy(run_graph(g32, x, etl.float32))
    out64 = as_numpy(run_graph(g64, x, etl.float64))
    assert out32.dtype == np.float32 and out64.dtype == np.float64
    np.testing.assert_array_equal(out32, x.astype(np.float32))
    np.testing.assert_array_equal(out64, x.astype(np.float64))


# --- enum static arg ----------------------------------------------------------


def test_enum_static_arg_specializes_graph(run_graph, as_numpy):
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    g_a = etl.trace(mode_fn, etl.TensorSpec((3,), etl.float32), Mode.A)
    g_b = etl.trace(mode_fn, etl.TensorSpec((3,), etl.float32), Mode.B)

    # Python `if` over the enum ran at trace time: different op sets
    ops_a = [op.name for op in g_a.module.main.entry_block.ops]
    ops_b = [op.name for op in g_b.module.main.entry_block.ops]
    assert "multiply" in ops_a and "negate" not in ops_a
    assert "negate" in ops_b and "multiply" not in ops_b

    np.testing.assert_array_equal(as_numpy(run_graph(g_a, x, Mode.A)), x * x)
    np.testing.assert_array_equal(as_numpy(run_graph(g_b, x, Mode.B)), -x)


# --- slice static arg ---------------------------------------------------------


def test_slice_static_arg_specializes_graph(run_graph, as_numpy):
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    g02 = etl.trace(slice_fn, etl.TensorSpec((4,), etl.float32), slice(0, 2))
    g13 = etl.trace(slice_fn, etl.TensorSpec((4,), etl.float32), slice(1, 3))

    attrs02 = g02.module.main.entry_block.ops[0].attributes
    attrs13 = g13.module.main.entry_block.ops[0].attributes
    assert attrs02 != attrs13
    assert attrs02["start_indices"] == (0,) and attrs02["limit_indices"] == (2,)
    assert attrs13["start_indices"] == (1,) and attrs13["limit_indices"] == (3,)

    np.testing.assert_array_equal(as_numpy(run_graph(g02, x, slice(0, 2))), x[0:2])
    np.testing.assert_array_equal(as_numpy(run_graph(g13, x, slice(1, 3))), x[1:3])


# --- bool / int specialization ------------------------------------------------


@pytest.mark.parametrize(
    "flag,expected_op",
    [(True, "add"), (False, "negate"), (1, "add"), (0, "negate")],
    ids=["True", "False", "1", "0"],
)
def test_static_bool_int_specializes_graph(flag, expected_op):
    graph = etl.trace(flag_fn, etl.TensorSpec((3,), etl.float32), flag)
    names = [op.name for op in graph.module.main.entry_block.ops]
    assert expected_op in names
    assert ("negate" if expected_op == "add" else "add") not in names


def test_static_flag_validated_at_run_time(run_graph, as_numpy):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    graph = etl.trace(flag_fn, etl.TensorSpec((3,), etl.float32), True)

    np.testing.assert_array_equal(as_numpy(run_graph(graph, x, True)), x + x)
    with pytest.raises(etl.TraceError, match="graph was specialized on"):
        run_graph(graph, x, False)


def test_flatten_inputs_rejects_changed_static_value():
    graph = etl.trace(flag_fn, etl.TensorSpec((3,), etl.float32), True)
    with pytest.raises(etl.TraceError, match="graph was specialized on"):
        graph.flatten_inputs((np.ones(3, dtype=np.float32), False))


# --- non-static spec rejection ------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [PlainConfig(0.1), np.float32(1.0), object()],
    ids=["plain-object", "numpy-scalar", "generic-object"],
)
def test_non_static_specs_are_rejected(bad):
    with pytest.raises(
        etl.TraceError, match="is neither a core.TensorSpec nor a static"
    ):
        etl.trace(identity_fn, etl.TensorSpec((2,), etl.float32), bad)


# BUG(etl): a plain dataclass config object must be REJECTED as a trace spec
# ("is neither a core.TensorSpec nor a static" — `trace._is_static_value`
# documents that "arbitrary config objects are NOT static in v1"), but
# `etl.trace` silently ACCEPTS it: `core.flatten` descends into every
# dataclass, so `Config(lr=0.1)` is exploded into a static-float leaf and the
# graph silently specializes on 0.1 as if the config object had been a plain
# float. Minimal repro:
#     etl.trace(identity_fn, etl.TensorSpec((2,), etl.float32), Config(0.1))
# builds a graph instead of raising TraceError.
def test_dataclass_config_spec_is_rejected():
    with pytest.raises(
        etl.TraceError, match="is neither a core.TensorSpec nor a static"
    ):
        etl.trace(identity_fn, etl.TensorSpec((2,), etl.float32), Config(0.1))


# --- run-time static validation via flatten_inputs ----------------------------


@pytest.mark.parametrize(
    "traced,passed",
    [
        (5, 6),  # value mismatch
        (1, True),  # kind mismatch: bool is not int
        (True, 1),  # kind mismatch: int is not bool
        (1.0, np.float32(1.0)),  # numpy scalars are NOT static in v1
        ("a", "b"),  # str value mismatch
    ],
    ids=["value-5-vs-6", "int-vs-bool", "bool-vs-int", "float-vs-np-scalar", "str-a-vs-b"],
)
def test_flatten_inputs_rejects_static_mismatch(traced, passed):
    def fn(x, s):
        return x

    graph = etl.trace(fn, etl.TensorSpec((2,), etl.float32), traced)
    with pytest.raises(etl.TraceError, match="graph was specialized on"):
        graph.flatten_inputs((np.ones(2, dtype=np.float32), passed))


def test_flatten_inputs_accepts_matching_static_value():
    def fn(x, s):
        return x

    graph = etl.trace(fn, etl.TensorSpec((2,), etl.float32), 5)
    flat = graph.flatten_inputs((np.ones(2, dtype=np.float32), 5))
    assert len(flat) == 1
    assert flat[0].dtype == np.dtype("float32")


# --- snapshotting is trace-time ------------------------------------------------


def test_closure_static_value_is_snapshotted_at_trace_time(run_graph, as_numpy):
    def make_graph():
        k = 2

        def f(x):
            return etl.multiply(x, etl.constant(etl.tensor(float(k), dtype=etl.float32)))

        graph = etl.trace(f, etl.TensorSpec((3,), etl.float32))
        k = 10  # mutated AFTER tracing — the graph must keep the baked 2
        return graph

    graph = make_graph()
    x = np.ones(3, dtype=np.float32)
    np.testing.assert_allclose(as_numpy(run_graph(graph, x)), 2.0 * x, rtol=0, atol=0)


# --- dtype equality ------------------------------------------------------------


def test_np_dtype_static_arg_validates_against_etl_dtype(run_graph, as_numpy):
    x = np.array([1, 2, 3], dtype=np.int32)
    assert etl.float32 == np.dtype("float32")

    graph = etl.trace(cast_to, etl.TensorSpec((3,), etl.int32), etl.float32)
    flat = graph.flatten_inputs((x, np.dtype("float32")))
    assert len(flat) == 1 and flat[0].dtype == np.dtype("int32")

    out = as_numpy(run_graph(graph, x, np.dtype("float32")))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x.astype(np.float32))
