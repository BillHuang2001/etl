"""SymbolicTensor operator dunders build the SAME IR as the etl.op functions.

Contract (``etl/core/symbolic.py`` + ``etl/ops/CONTEXT.md`` "Operator-handler
protocol"): every dunder dispatches through the handler registered by
``etl.ops``, and reflected calls (Python scalar on the left) route through
the SAME handler with the SymbolicTensor as first argument. Dunder sugar is
transparent — identical IR up to naming. Comparisons return bool-dtype
SymbolicTensors (never Python bools); ``bool(x)`` and ``x != y`` raise
``TraceError`` (runtime control flow must use ``etl.cond``/``etl.while_loop``/
``etl.scan``).
"""
from __future__ import annotations

import numpy as np
import pytest

import etl

from tests.ops.conftest import ops_of, run_numpy, trace_fn

# Rank-2 specs (matmul/dot needs rank >= 2; every other binary pair
# broadcasts fine on these shapes).
SPEC2 = etl.TensorSpec((2, 2), etl.float32)
SPEC1 = etl.TensorSpec((2, 3), etl.float32)


def _pretty(fn, *specs) -> str:
    """Trace ``fn`` and render its module as SSA text.

    Callers must have set ``ETL_DISABLE_LOCATIONS=1`` (monkeypatch) so both
    graphs print ``<unknown>:0:0`` and the comparison is location-independent.
    """
    return etl.ir.pretty_print(trace_fn(fn, *specs).module)


# (id, dunder expression, equivalent etl.op expression) — two inputs each.
DUNDER_PAIRS = [
    ("x + y", lambda a, b: a + b, lambda a, b: etl.add(a, b)),
    ("x - y", lambda a, b: a - b, lambda a, b: etl.subtract(a, b)),
    ("x * y", lambda a, b: a * b, lambda a, b: etl.multiply(a, b)),
    ("x / y", lambda a, b: a / b, lambda a, b: etl.divide(a, b)),
    ("x ** y", lambda a, b: a ** b, lambda a, b: etl.power(a, b)),
    ("x < y", lambda a, b: a < b, lambda a, b: etl.less(a, b)),
    ("x <= y", lambda a, b: a <= b, lambda a, b: etl.less_equal(a, b)),
    ("x > y", lambda a, b: a > b, lambda a, b: etl.greater(a, b)),
    ("x >= y", lambda a, b: a >= b, lambda a, b: etl.greater_equal(a, b)),
    ("x == y", lambda a, b: a == b, lambda a, b: etl.equal(a, b)),
    ("x @ y", lambda a, b: a @ b, lambda a, b: etl.dot(a, b)),
]


@pytest.mark.parametrize(
    "dunder,opfn",
    [(d, o) for _, d, o in DUNDER_PAIRS],
    ids=[n for n, _, _ in DUNDER_PAIRS],
)
def test_dunder_builds_same_ir_as_op_function(monkeypatch, dunder, opfn):
    """Two fns differing ONLY in dunder vs etl.op must trace to identical
    IR (full pretty-print string equality, locations disabled)."""
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")
    assert _pretty(dunder, SPEC2, SPEC2) == _pretty(opfn, SPEC2, SPEC2)


def test_neg_dunder_builds_same_ir_as_negate(monkeypatch):
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")
    assert _pretty(lambda a: -a, SPEC1) == _pretty(
        lambda a: etl.negate(a), SPEC1
    )


# Reflected dunders with a Python scalar on the LEFT. core routes the call
# through the SAME handler with the tensor as first argument, so the
# equivalent etl.op call is the one with the tensor FIRST and the scalar
# second (Python's reflected comparison swaps lt<->gt, le<->ge).
REFLECTED_PAIRS = [
    ("2 + x", lambda a: 2 + a, lambda a: etl.add(2, a)),
    ("2.5 * x", lambda a: 2.5 * a, lambda a: etl.multiply(2.5, a)),
    ("1 - x", lambda a: 1 - a, lambda a: etl.subtract(1, a)),
    ("10 / x", lambda a: 10 / a, lambda a: etl.divide(10, a)),
    ("2 ** x", lambda a: 2 ** a, lambda a: etl.power(2, a)),
    ("5 < x", lambda a: 5 < a, lambda a: etl.greater(a, 5)),
    ("5 <= x", lambda a: 5 <= a, lambda a: etl.greater_equal(a, 5)),
    ("5 > x", lambda a: 5 > a, lambda a: etl.less(a, 5)),
    ("5 >= x", lambda a: 5 >= a, lambda a: etl.less_equal(a, 5)),
    ("5 == x", lambda a: 5 == a, lambda a: etl.equal(a, 5)),
]


@pytest.mark.parametrize(
    "dunder,opfn",
    [(d, o) for _, d, o in REFLECTED_PAIRS],
    ids=[n for n, _, _ in REFLECTED_PAIRS],
)
def test_reflected_dunder_builds_same_ir_as_op_function(
    monkeypatch, dunder, opfn
):
    """The scalar auto-promotes per weak NEP-50 rules in BOTH call forms —
    the reflected dunder must produce IR identical to the explicit etl.op
    call with the same scalar."""
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")
    assert _pretty(dunder, SPEC1) == _pretty(opfn, SPEC1)


# Python scalar on the RIGHT: ordinary (non-reflected) dispatch.
SCALAR_RIGHT_PAIRS = [
    ("x < 5", lambda a: a < 5, lambda a: etl.less(a, 5)),
    ("x <= 5", lambda a: a <= 5, lambda a: etl.less_equal(a, 5)),
    ("x > 5", lambda a: a > 5, lambda a: etl.greater(a, 5)),
    ("x >= 5", lambda a: a >= 5, lambda a: etl.greater_equal(a, 5)),
    ("x == 5", lambda a: a == 5, lambda a: etl.equal(a, 5)),
]


@pytest.mark.parametrize(
    "dunder,opfn",
    [(d, o) for _, d, o in SCALAR_RIGHT_PAIRS],
    ids=[n for n, _, _ in SCALAR_RIGHT_PAIRS],
)
def test_scalar_right_comparison_builds_same_ir(monkeypatch, dunder, opfn):
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")
    assert _pretty(dunder, SPEC1) == _pretty(opfn, SPEC1)


def test_operator_handlers_match_documented_mapping():
    """The registration table must map each dispatch kind to the op function
    named in etl/ops/CONTEXT.md ("Operator-handler protocol")."""
    from etl.ops import _registration
    from etl.ops import indexing

    assert _registration.OPERATOR_HANDLERS == {
        "add": etl.add,
        "sub": etl.subtract,
        "mul": etl.multiply,
        "matmul": etl.dot,
        "truediv": etl.divide,
        "pow": etl.power,
        "neg": etl.negate,
        "lt": etl.less,
        "le": etl.less_equal,
        "gt": etl.greater,
        "ge": etl.greater_equal,
        "eq": etl.equal,
        "getitem": indexing.getitem,
    }


# ---------------------------------------------------------------------------
# comparisons build bool-dtype SymbolicTensors, never Python bools
# ---------------------------------------------------------------------------

def test_comparison_returns_symbolic_bool_not_python_bool():
    def fn(a, b):
        result = a < b
        assert isinstance(result, etl.SymbolicTensor)
        return result

    graph = etl.trace(fn, SPEC1, SPEC1)
    (op,) = ops_of(graph, "less")
    assert op.results[0].type.dtype == np.dtype("bool")


@pytest.mark.parametrize(
    "fn",
    [lambda a: a == 5, lambda a: 5 == a],
    ids=["x == 5", "5 == x"],
)
def test_eq_with_python_scalar_both_directions(monkeypatch, fn):
    """``x == scalar`` and ``scalar == x`` both build an equal op with bool
    dtype (identical IR — the reflected call routes through the same handler
    with the tensor first)."""
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")
    graph = etl.trace(fn, SPEC1)
    (op,) = ops_of(graph, "equal")
    assert op.results[0].type.dtype == np.dtype("bool")
    assert etl.ir.pretty_print(graph.module) == _pretty(
        lambda a: etl.equal(a, 5), SPEC1
    )


def test_dunder_numeric_result_matches_op_function():
    """Dunder and op function must also agree NUMERICALLY at run time."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    np.testing.assert_allclose(
        run_numpy(lambda x, y: x + y, a, b),
        run_numpy(lambda x, y: etl.add(x, y), a, b),
    )
    np.testing.assert_allclose(
        run_numpy(lambda x, y: x @ y, a, b),
        run_numpy(lambda x, y: etl.dot(x, y), a, b),
    )


# ---------------------------------------------------------------------------
# bool() / != : TraceError, never silent graph values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        lambda a: bool(a),
        lambda a: bool(etl.sum(a)),
        lambda a: not a,
    ],
    ids=["bool(x)", "bool(sum(x))", "not x"],
)
def test_bool_of_symbolic_raises_trace_error(expr):
    with pytest.raises(
        etl.TraceError, match=r"cannot be used as a Python boolean"
    ):
        etl.trace(expr, SPEC1)


def test_ne_raises_via_eq_inversion_not_silent_graph_value():
    """``x != y`` must NOT silently return a graph value. SymbolicTensor
    defines no ``__ne__`` (verified below), so Python falls back to
    ``not (x == y)``: the ``==`` builds an equal-op SymbolicTensor, then
    ``not`` coerces it via ``__bool__``, which raises TraceError."""
    assert etl.SymbolicTensor.__dict__.get("__ne__") is None

    def fn(a, b):
        return a != b

    with pytest.raises(
        etl.TraceError, match=r"cannot be used as a Python boolean"
    ):
        etl.trace(fn, SPEC1, SPEC1)
