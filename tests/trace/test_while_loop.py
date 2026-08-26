"""Tests for `etl.while_loop` — runtime tensor control flow traced into a
`while` op.

Contract under test (see ``etl/trace/CONTEXT.md``, "Control-flow region
conventions"): `etl.while_loop(cond_fn, body_fn, init)` traces one `while` op
whose operands are the loop-carried SymbolicTensors (≥1 — an all-static init
raises) with two single-block regions: `regions[0]` = condition (its `return`
yields ONE 0-d bool) and `regions[1]` = body (its `return` yields the next
carried values, same tree as init, types constant across iterations). Each
region's entry block has one arg per carried operand. Static init leaves
specialize the loop (not carried) and must be returned unchanged by the body.
Runtime semantics are checked on the numpy backend via `run_graph`; IR layout
is inspected on the traced module.
"""

import numpy as np
import pytest

import etl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _const(value, dtype):
    """Embed a static value as a constant op (valid only inside a trace)."""
    return etl.constant(etl.tensor(value, dtype=dtype))


def _all_ops(function):
    """Yield every op of `function`'s body, descending into nested regions."""
    stack = [function.entry_block]
    emitted = set()
    while stack:
        block = stack.pop()
        for op in block.ops:
            if op.id in emitted:
                continue
            emitted.add(op.id)
            yield op
            for region in op.regions:
                stack.extend(region.blocks)


def _trace_count_loop(n, x_shape=(3,)):
    """Graph: `(counter, acc)` starting at (0, zeros); body does
    `counter += 1`, `acc += x` while `counter < n` (n is a static int that
    specializes the traced condition)."""

    def f(x):
        def cond_fn(state):
            i, acc = state
            return etl.less(i, _const(n, etl.int32))

        def body_fn(state):
            i, acc = state
            return (etl.add(i, _const(1, etl.int32)), etl.add(acc, x))

        init = (_const(0, etl.int32), etl.constant(etl.zeros(x_shape, etl.float32)))
        return etl.while_loop(cond_fn, body_fn, init)

    return etl.trace(f, etl.TensorSpec(x_shape, etl.float32))


# ---------------------------------------------------------------------------
# iteration semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3, 7])
def test_iteration_count_and_accumulated_values(run_graph, as_numpy, n):
    x = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    graph = _trace_count_loop(n, x_shape=x.shape)
    counter, acc = as_numpy(run_graph(graph, x))
    assert isinstance(counter, np.ndarray) and counter.shape == ()
    assert counter == n
    np.testing.assert_allclose(acc, n * x)


def test_structured_dict_state(run_graph, as_numpy):
    """Init may be a dict; the result keeps the same structure."""

    def f(x):
        def cond_fn(state):
            return etl.less(state["i"], _const(3, etl.int32))

        def body_fn(state):
            return {
                "i": etl.add(state["i"], _const(1, etl.int32)),
                "acc": etl.add(state["acc"], x),
            }

        init = {
            "i": _const(0, etl.int32),
            "acc": etl.constant(etl.zeros((2,), etl.float32)),
        }
        return etl.while_loop(cond_fn, body_fn, init)

    x = np.array([2.0, 4.0], dtype=np.float32)
    graph = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    result = as_numpy(run_graph(graph, x))
    assert isinstance(result, dict) and set(result) == {"i", "acc"}
    assert result["i"] == 3
    np.testing.assert_allclose(result["acc"], 3 * x)


def test_nested_tuple_state(run_graph, as_numpy):
    """Nested tuple init `(counter, (a, b))` round-trips structurally."""

    def f(x):
        def cond_fn(state):
            return etl.less(state[0], _const(2, etl.int32))

        def body_fn(state):
            counter, (a, b) = state
            return (
                etl.add(counter, _const(1, etl.int32)),
                (etl.add(a, x), etl.add(b, x)),
            )

        init = (
            _const(0, etl.int32),
            (
                etl.constant(etl.zeros((2,), etl.float32)),
                etl.constant(etl.zeros((2,), etl.float32)),
            ),
        )
        return etl.while_loop(cond_fn, body_fn, init)

    x = np.array([1.0, 2.0], dtype=np.float32)
    graph = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    counter, (a, b) = as_numpy(run_graph(graph, x))
    assert counter == 2
    np.testing.assert_allclose(a, 2 * x)
    np.testing.assert_allclose(b, 2 * x)


def test_zero_iterations_returns_init_unchanged(run_graph, as_numpy):
    """A false-at-entry condition executes the body zero times."""

    def f():
        def cond_fn(state):
            return etl.less(state[0], _const(0, etl.int32))

        def body_fn(state):
            return state

        init = (_const(7, etl.int32), etl.constant(etl.zeros((2,), etl.float32)))
        return etl.while_loop(cond_fn, body_fn, init)

    graph = etl.trace(f)
    counter, acc = as_numpy(run_graph(graph))
    assert counter == 7
    np.testing.assert_allclose(acc, np.zeros((2,), dtype=np.float32))


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------

def test_while_op_ir_structure():
    """A traced loop is exactly one `while` op: 2 single-block regions whose
    entry args bind the carried operands (n args each); the cond region
    returns ONE value, the body region n; op results = final carried values,
    feeding main's `return` terminator."""
    graph = _trace_count_loop(3)
    function = graph.module.main
    ops = list(_all_ops(function))
    while_ops = [op for op in ops if op.name == "while"]
    assert len(while_ops) == 1, "exactly one `while` op expected"
    while_op = while_ops[0]

    # 2 carried tensors: counter (int32 scalar) + acc (float32 (3,)).
    assert len(while_op.operands) == 2
    assert len(while_op.regions) == 2
    cond_region, body_region = while_op.regions
    for region in (cond_region, body_region):
        assert len(region.blocks) == 1
        args = region.entry.arguments
        assert len(args) == len(while_op.operands)
        assert args[0].type.dtype == etl.int32 and args[0].type.shape == ()
        assert args[1].type.dtype == etl.float32 and args[1].type.shape == (3,)

    # operands == the carried values (the two constants in main's block).
    constants = [op for op in function.entry_block if op.name == "constant"]
    assert len(constants) == 2
    assert {v.id for v in while_op.operands} == {op.result.id for op in constants}

    # cond region returns one value; body region returns the carried count.
    assert len(cond_region.entry.terminator.operands) == 1
    assert cond_region.entry.terminator.operands[0].type.dtype == etl.bool_
    assert len(body_region.entry.terminator.operands) == 2

    # op results = final carried values = main's return operands.
    terminator = function.entry_block.terminator
    assert terminator.name == "return"
    assert [v.id for v in terminator.operands] == [v.id for v in while_op.results]


# ---------------------------------------------------------------------------
# error cases
# ---------------------------------------------------------------------------

def _cond_vector(state):
    return etl.greater(state[0], etl.constant(etl.zeros((3,), etl.int32)))


def _cond_int32(state):
    return state[0]


def _cond_python_bool(state):
    return True


@pytest.mark.parametrize(
    ("cond_fn", "match"),
    [
        (_cond_vector, "0-d"),
        (_cond_int32, "dtype must be bool"),
        (_cond_python_bool, "must be a core.SymbolicTensor"),
    ],
    ids=["vector", "int32", "python_bool"],
)
def test_cond_fn_result_must_be_scalar_bool_symbolic(cond_fn, match):
    def f():
        def body_fn(state):
            return state

        init = (etl.constant(etl.zeros((3,), etl.int32)),)
        return etl.while_loop(cond_fn, body_fn, init)

    with pytest.raises(etl.TraceError, match=match):
        etl.trace(f)


def test_body_fn_tree_mismatch_raises():
    def f():
        def cond_fn(state):
            return _const(True, etl.bool_)

        def body_fn(state):
            return (state[0], state[0])

        init = (_const(0, etl.int32),)
        return etl.while_loop(cond_fn, body_fn, init)

    with pytest.raises(etl.TraceError, match="must match init's tree"):
        etl.trace(f)


def test_body_fn_dtype_change_raises():
    def f():
        def cond_fn(state):
            return _const(True, etl.bool_)

        def body_fn(state):
            return (_const(1.5, etl.float32),)

        init = (_const(0, etl.int32),)
        return etl.while_loop(cond_fn, body_fn, init)

    with pytest.raises(etl.DTypeError, match="dtype"):
        etl.trace(f)


def test_body_fn_shape_change_raises():
    def f():
        def cond_fn(state):
            return _const(True, etl.bool_)

        def body_fn(state):
            return (etl.constant(etl.zeros((3,), etl.float32)),)

        init = (etl.constant(etl.zeros((2,), etl.float32)),)
        return etl.while_loop(cond_fn, body_fn, init)

    with pytest.raises(etl.ShapeError, match="shape"):
        etl.trace(f)


def test_all_static_init_rejected():
    """A loop over static values only is plain Python control flow."""

    def f():
        return etl.while_loop(lambda s: True, lambda s: s, (1, 2))

    with pytest.raises(etl.TraceError, match="at least one SymbolicTensor"):
        etl.trace(f)


def test_unknown_init_leaf_rejected():
    def f():
        return etl.while_loop(lambda s: True, lambda s: s, np.array([1.0]))

    with pytest.raises(etl.TraceError, match="must be a core.SymbolicTensor"):
        etl.trace(f)


def test_static_leaf_change_rejected():
    """Static leaves are not loop-carried — the body must return them equal."""

    def f():
        def cond_fn(state):
            return etl.less(state[0], _const(3, etl.int32))

        def body_fn(state):
            return (etl.add(state[0], _const(1, etl.int32)), 99)

        init = (_const(0, etl.int32), 3)
        return etl.while_loop(cond_fn, body_fn, init)

    with pytest.raises(etl.TraceError, match="must equal init's static value"):
        etl.trace(f)


def test_static_leaf_preserved_in_result(run_graph, as_numpy):
    """An unchanged static init leaf is re-inserted into the result structure
    (it is never carried as a tensor)."""

    def f():
        def cond_fn(state):
            return etl.less(state[0], _const(3, etl.int32))

        def body_fn(state):
            return (etl.add(state[0], _const(1, etl.int32)), 3)

        init = (_const(0, etl.int32), 3)
        return etl.while_loop(cond_fn, body_fn, init)

    graph = etl.trace(f)
    counter, static_leaf = as_numpy(run_graph(graph))
    assert counter == 3
    assert static_leaf == 3 and not isinstance(static_leaf, etl.Tensor)


# ---------------------------------------------------------------------------
# full-graph integration
# ---------------------------------------------------------------------------

def test_while_loop_inside_defn_verifies_and_runs(run_graph, as_numpy):
    @etl.defn
    def f(x):
        def cond_fn(state):
            return etl.less(state[0], _const(3, etl.int32))

        def body_fn(state):
            return (
                etl.add(state[0], _const(1, etl.int32)),
                etl.add(state[1], x),
            )

        init = (_const(0, etl.int32), etl.constant(etl.zeros((2,), etl.float32)))
        return etl.while_loop(cond_fn, body_fn, init)

    graph = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    graph.verify()
    x = np.array([1.0, 2.0], dtype=np.float32)
    counter, acc = as_numpy(run_graph(graph, x))
    assert counter == 3
    np.testing.assert_allclose(acc, 3 * x)
