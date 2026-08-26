"""Tests for the active-builder context (`current_builder` / `with_builder` /
`builder_stack`).

Contract under test (`etl/trace/builder.py` + `etl/trace/CONTEXT.md`):
- Outside any trace, `current_builder()` raises `etl.TraceError`
  ("No active trace") — this is the hook `etl.ops` uses, so op functions
  called outside a trace fail the same way.
- Inside `etl.trace(fn, ...)`, `current_builder()` returns the ONE
  `etl.ir.Builder` serving the whole trace: the same instance across ops
  in the traced fn AND inside control-flow regions (`cond`/`while_loop`
  callables) — `control_flow.py` swaps the builder's insertion point, it
  does not create new builders.
- `with_builder` (alias `builder_stack`) is a nestable LIFO context
  manager usable standalone.

NOTE (contract gap): the contract is silent about calling `etl.trace`
inside a traced function — no assertion is made for nested traces here.
"""

import numpy as np
import pytest

import etl
from etl.trace import builder_stack, current_builder, with_builder


# --- no trace active --------------------------------------------------------


def test_current_builder_raises_outside_trace():
    with pytest.raises(etl.TraceError, match="No active trace"):
        current_builder()


def test_op_outside_trace_raises_no_active_trace():
    """Ops query `current_builder()` — a SymbolicTensor from a finished
    trace cannot be used outside one."""
    captured = []

    def fn(x):
        captured.append(x)
        return etl.add(x, x)

    etl.trace(fn, etl.TensorSpec((2,), etl.float32))

    assert len(captured) == 1
    assert isinstance(captured[0], etl.SymbolicTensor)
    with pytest.raises(etl.TraceError, match="No active trace"):
        etl.add(captured[0], captured[0])


# --- active builder during a trace ------------------------------------------


def test_current_builder_inside_trace_is_stable_shared_builder():
    builders = []

    def fn(x):
        builders.append(current_builder())
        y = etl.add(x, x)
        builders.append(current_builder())
        return y

    etl.trace(fn, etl.TensorSpec((2,), etl.float32))

    assert len(builders) == 2
    assert isinstance(builders[0], etl.ir.Builder)
    # Same object on two separate ops — the contextvar stack is stable
    # for the duration of the trace.
    assert builders[0] is builders[1]


# --- manual with_builder installation ---------------------------------------


def test_with_builder_installs_and_restores():
    b = etl.ir.Builder()

    with pytest.raises(etl.TraceError, match="No active trace"):
        current_builder()

    with with_builder(b) as entered:
        assert entered is b
        assert current_builder() is b

    # Stack fully restored on exit.
    with pytest.raises(etl.TraceError, match="No active trace"):
        current_builder()


def test_with_builder_nesting_is_lifo():
    b1 = etl.ir.Builder()
    b2 = etl.ir.Builder()

    with with_builder(b1):
        assert current_builder() is b1
        with with_builder(b2):
            assert current_builder() is b2
        assert current_builder() is b1

    with pytest.raises(etl.TraceError, match="No active trace"):
        current_builder()


def test_builder_stack_is_alias_of_with_builder():
    assert builder_stack is with_builder


# --- ops route into nested control-flow regions on the SAME builder ---------


def test_while_loop_regions_use_the_same_builder_instance():
    trace_builders = []
    region_builders = []

    @etl.defn
    def f(x):
        trace_builders.append(current_builder())

        def cond_fn(state):
            region_builders.append(("cond", current_builder()))
            i, acc = state
            return etl.less(i, etl.constant(etl.tensor(3, dtype=etl.int32)))

        def body_fn(state):
            region_builders.append(("body", current_builder()))
            i, acc = state
            return (
                etl.add(i, etl.constant(etl.tensor(1, dtype=etl.int32))),
                etl.add(acc, x),
            )

        return etl.while_loop(
            cond_fn,
            body_fn,
            (
                etl.constant(etl.tensor(0, dtype=etl.int32)),
                etl.constant(etl.zeros((2,), etl.float32)),
            ),
        )

    etl.trace(f, etl.TensorSpec((2,), etl.float32))

    # cond_fn and body_fn each ran once at trace time.
    assert [where for where, _ in region_builders] == ["cond", "body"]
    for where, builder in region_builders:
        assert isinstance(builder, etl.ir.Builder)
        assert builder is trace_builders[0]


def test_while_loop_runs_end_to_end(run_graph, as_numpy):
    @etl.defn
    def f(x):
        def cond_fn(state):
            i, acc = state
            return etl.less(i, etl.constant(etl.tensor(3, dtype=etl.int32)))

        def body_fn(state):
            i, acc = state
            return (
                etl.add(i, etl.constant(etl.tensor(1, dtype=etl.int32))),
                etl.add(acc, x),
            )

        return etl.while_loop(
            cond_fn,
            body_fn,
            (
                etl.constant(etl.tensor(0, dtype=etl.int32)),
                etl.constant(etl.zeros((2,), etl.float32)),
            ),
        )

    graph = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    result = as_numpy(run_graph(graph, np.array([10.0, 20.0], dtype=np.float32)))

    i, acc = result
    assert i == 3
    np.testing.assert_allclose(acc, np.array([30.0, 60.0], dtype=np.float32))
