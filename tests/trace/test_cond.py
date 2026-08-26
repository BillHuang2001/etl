"""Tests for `etl.cond` — runtime tensor control flow traced into `if` ops.

Contract under test (see ``etl/trace/CONTEXT.md``, "Control-flow region
conventions"): `etl.cond(pred, true_fn, false_fn, *operands, **static_kwargs)`
runs the two branch callables ONCE each at trace time inside the two regions
of a single `if` op — operand 0 is the 0-d bool predicate, operands 1..n are
captured SSA tensors bound to each region's entry-block args (predicate
included, one arg per operand). Branch outputs must be structurally identical
trees of SymbolicTensors with unified dtypes/shapes. Runtime semantics are
checked on the numpy backend via the explicit staging pipeline (`run_graph`);
IR layout is inspected on the traced module.
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


def _trace_scalar_cond():
    """Graph: `x > 0 ? 2 * x : -x` (scalar float32 input)."""

    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v: etl.multiply(v, _const(2.0, etl.float32)),
            lambda v: etl.negate(v),
            x,
        )

    return etl.trace(f, etl.TensorSpec((), etl.float32))


# ---------------------------------------------------------------------------
# runtime semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "x_val",
    [
        np.array(3.0, dtype=np.float32),
        np.array(-3.0, dtype=np.float32),
        np.array(0.0, dtype=np.float32),
    ],
    ids=["positive", "negative", "zero"],
)
def test_runtime_semantics_selects_branch(run_graph, as_numpy, x_val):
    """The `if` op selects the branch matching the predicate at run time."""
    graph = _trace_scalar_cond()
    result = as_numpy(run_graph(graph, x_val))
    expected = np.where(x_val > 0, 2.0 * x_val, -x_val)
    np.testing.assert_allclose(result, expected)


def test_runtime_semantics_tuple_branches(run_graph, as_numpy):
    """Both branches may yield a (structurally identical) tuple of tensors."""

    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v: (
                etl.multiply(v, _const(2.0, etl.float32)),
                _const(10.0, etl.float32),
            ),
            lambda v: (etl.negate(v), _const(-10.0, etl.float32)),
            x,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    pos = as_numpy(run_graph(graph, np.array(3.0, dtype=np.float32)))
    assert isinstance(pos, tuple) and len(pos) == 2
    np.testing.assert_allclose(pos[0], 6.0)
    np.testing.assert_allclose(pos[1], 10.0)
    neg = as_numpy(run_graph(graph, np.array(-3.0, dtype=np.float32)))
    np.testing.assert_allclose(neg[0], 3.0)
    np.testing.assert_allclose(neg[1], -10.0)


# ---------------------------------------------------------------------------
# IR structure
# ---------------------------------------------------------------------------

def test_if_op_ir_structure():
    """A traced cond is exactly one `if` op: 2 single-block regions whose
    entry args bind ALL operands (pred at index 0, captured tensors after),
    and its results feed main's `return` terminator."""
    graph = _trace_scalar_cond()
    function = graph.module.main
    ops = list(_all_ops(function))
    if_ops = [op for op in ops if op.name == "if"]
    assert len(if_ops) == 1, "exactly one `if` op expected"
    if_op = if_ops[0]

    # Pred + 1 captured operand → 2 operands, 2 region entry args each.
    assert len(if_op.operands) == 2
    assert len(if_op.regions) == 2
    for region in if_op.regions:
        assert len(region.blocks) == 1
        args = region.entry.arguments
        assert len(args) == len(if_op.operands)
        assert args[0].type.dtype == etl.bool_  # the predicate binding
        assert args[0].type.shape == ()
        assert args[1].type.dtype == etl.float32
        assert args[1].type.shape == ()

    # Operand 0 IS the predicate (the `greater` result).
    greater = [op for op in ops if op.name == "greater"]
    assert len(greater) == 1
    assert if_op.operands[0] is greater[0].result
    # Operand 1 is the captured input (main's block argument).
    assert if_op.operands[1] is function.entry_block.arguments[0]

    # The if results are exactly main's return operands.
    terminator = function.entry_block.terminator
    assert terminator.name == "return"
    assert [v.id for v in terminator.operands] == [v.id for v in if_op.results]


# ---------------------------------------------------------------------------
# predicate validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("make_pred", "match"),
    [
        pytest.param(
            lambda v: etl.greater(v, etl.constant(etl.zeros((3,), etl.float32))),
            "0-d",
            id="vector_pred",
        ),
        pytest.param(lambda v: v, "dtype must be bool", id="int32_pred"),
        pytest.param(lambda v: True, "must be a core.SymbolicTensor", id="python_bool"),
        pytest.param(
            lambda v: np.bool_(True),
            "must be a core.SymbolicTensor",
            id="numpy_bool",
        ),
    ],
)
def test_pred_must_be_scalar_bool_symbolic(make_pred, match):
    """Non-scalar, non-bool, or non-symbolic predicates fail with TraceError."""

    def f(x):
        return etl.cond(make_pred(x), lambda v: v, lambda v: v, x)

    with pytest.raises(etl.TraceError, match=match):
        etl.trace(f, etl.TensorSpec((), etl.float32))


def test_static_python_bool_pred_rejected():
    """A static True/False predicate raises — no silent folding: runtime
    tensor control flow is only for tensor predicates."""

    def f(x):
        return etl.cond(True, lambda v: v, lambda v: etl.negate(v), x)

    with pytest.raises(
        etl.TraceError, match="only for tensor predicates"
    ):
        etl.trace(f, etl.TensorSpec((), etl.float32))


# ---------------------------------------------------------------------------
# branch unification errors
# ---------------------------------------------------------------------------

def test_branch_tree_mismatch_raises():
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(pred, lambda v: (v, v), lambda v: v, x)

    with pytest.raises(etl.TraceError, match="structurally identical"):
        etl.trace(f, etl.TensorSpec((), etl.float32))


def test_branch_dtype_mismatch_raises():
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred, lambda v: v, lambda v: _const(1, etl.int32), x
        )

    with pytest.raises(etl.DTypeError, match="dtype mismatch"):
        etl.trace(f, etl.TensorSpec((), etl.float32))


def test_branch_shape_mismatch_raises():
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v: etl.add(v, etl.constant(etl.zeros((2, 3), etl.float32))),
            lambda v: etl.add(v, etl.constant(etl.zeros((2,), etl.float32))),
            x,
        )

    with pytest.raises(etl.ShapeError, match="shape mismatch"):
        etl.trace(f, etl.TensorSpec((), etl.float32))


# ---------------------------------------------------------------------------
# operands and static kwargs
# ---------------------------------------------------------------------------

def test_static_operand_passed_to_branches_unchanged(run_graph, as_numpy):
    """Static (Python) operands specialize the branches and are passed
    positionally, unchanged, to both branch callables."""

    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v, k: etl.add(v, _const(float(k), etl.float32)),
            lambda v, k: etl.negate(v),
            x,
            3,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    assert as_numpy(run_graph(graph, np.array(1.0, dtype=np.float32))) == 4.0
    assert as_numpy(run_graph(graph, np.array(-1.0, dtype=np.float32))) == 1.0


def test_static_kwargs_passed_to_both_branches(run_graph, as_numpy):
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v, scale: etl.multiply(v, _const(scale, etl.float32)),
            lambda v, scale: etl.multiply(v, _const(scale, etl.float32)),
            x,
            scale=2.0,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    assert as_numpy(run_graph(graph, np.array(1.5, dtype=np.float32))) == 3.0
    assert as_numpy(run_graph(graph, np.array(-1.5, dtype=np.float32))) == -3.0


def test_non_static_kwarg_rejected():
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(pred, lambda v, s: v, lambda v, s: v, x, scale=x)

    with pytest.raises(etl.TraceError, match="static kwarg"):
        etl.trace(f, etl.TensorSpec((), etl.float32))


def test_non_symbolic_non_static_operand_rejected():
    """A concrete numpy array operand is neither symbolic nor static → error."""

    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v, arr: v,
            lambda v, arr: v,
            x,
            np.array([1.0], dtype=np.float32),
        )

    with pytest.raises(etl.TraceError, match="must be a core.SymbolicTensor"):
        etl.trace(f, etl.TensorSpec((), etl.float32))


# ---------------------------------------------------------------------------
# zero-operand cond + full-graph integration
# ---------------------------------------------------------------------------

def test_zero_operand_cond(run_graph, as_numpy):
    """With no captured operands the `if` op has a single operand: the pred."""

    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda: _const(1.0, etl.float32),
            lambda: _const(-1.0, etl.float32),
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    if_ops = [op for op in _all_ops(graph.module.main) if op.name == "if"]
    assert len(if_ops) == 1
    if_op = if_ops[0]
    assert len(if_op.operands) == 1
    for region in if_op.regions:
        assert len(region.entry.arguments) == 1  # pred binding only
    assert as_numpy(run_graph(graph, np.array(1.0, dtype=np.float32))) == 1.0
    assert as_numpy(run_graph(graph, np.array(-1.0, dtype=np.float32))) == -1.0


def test_cond_inside_defn_verifies_and_runs(run_graph, as_numpy):
    """cond traced inside an `@etl.defn` produces a verifiable graph that
    executes end-to-end through the explicit staging pipeline."""

    @etl.defn
    def f(x):
        pred = etl.greater(x, _const(0.0, etl.float32))
        return etl.cond(
            pred,
            lambda v: etl.multiply(v, _const(2.0, etl.float32)),
            lambda v: etl.negate(v),
            x,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    graph.verify()
    assert as_numpy(run_graph(graph, np.array(4.0, dtype=np.float32))) == 8.0
    assert as_numpy(run_graph(graph, np.array(-4.0, dtype=np.float32))) == 4.0
