"""Tests for etl symbolic shape dimensions: ``Dim``, ``DimExpr``, ``dim()``.

Authoritative source: ``etl/core/dim.py`` (contract summarized in
``etl/core/CONTEXT.md``). Dims and dim-expressions are pure ASTs: arithmetic
dunders only build ``DimExpr`` nodes (ops are named ``add/sub/mul/floordiv/mod/
min/max``), ``min``/``max`` are *methods*, ``DimExpr.evaluate(dim_sizes)`` does
explicit substitution (bindings take precedence over known sizes) with Python
int semantics, ``==`` is structural, ordering comparisons resolve using known
sizes only ("constraint-free") and raise ``ShapeError`` when unresolved, and
``bool()`` raises ``ShapeError``.
"""

import operator

import pytest

import etl
import etl.core
from etl import Dim, DimExpr, ShapeError, dim


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_dim_fields():
    symbolic = Dim("n")
    assert symbolic.name == "n"
    assert symbolic.size is None

    known = Dim("m", 3)
    assert known.name == "m"
    assert known.size == 3


def test_dimexpr_direct_construction_stores_operands():
    a = Dim("a")
    expr = DimExpr("add", a, 3)
    assert expr.op == "add"
    assert expr.left == a
    assert expr.right == 3


def test_dimexpr_rejects_unknown_op():
    with pytest.raises(ValueError):
        DimExpr("bogus", Dim("a"), 1)


@pytest.mark.parametrize("bad", [2.5, "x", None, [1]], ids=repr)
def test_dimexpr_rejects_invalid_operands(bad):
    with pytest.raises(TypeError):
        DimExpr("add", Dim("a"), bad)
    with pytest.raises(TypeError):
        DimExpr("add", bad, Dim("a"))


# ---------------------------------------------------------------------------
# Arithmetic builds expression trees (pure construction, no evaluation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("expr", "op", "left", "right"),
    [
        (Dim("a", 1) + Dim("b", 2), "add", Dim("a", 1), Dim("b", 2)),
        (Dim("a") - 3, "sub", Dim("a"), 3),
        (Dim("a") * 2, "mul", Dim("a"), 2),
        (Dim("a") // 2, "floordiv", Dim("a"), 2),
        (Dim("a") % 2, "mod", Dim("a"), 2),
        (2 + Dim("a"), "add", 2, Dim("a")),
        (2 - Dim("a"), "sub", 2, Dim("a")),
        (3 * Dim("a"), "mul", 3, Dim("a")),
        (10 // Dim("a"), "floordiv", 10, Dim("a")),
        (7 % Dim("a"), "mod", 7, Dim("a")),
        (Dim("a").min(2), "min", Dim("a"), 2),
        (Dim("a").max(Dim("b", 5)), "max", Dim("a"), Dim("b", 5)),
    ],
    ids=["a+b", "a-3", "a*2", "a//2", "a%2", "2+a", "2-a", "3*a",
         "10//a", "7%a", "a.min(2)", "a.max(b)"],
)
def test_arithmetic_builds_expression_trees(expr, op, left, right):
    assert isinstance(expr, DimExpr)
    assert expr.op == op
    assert expr.left == left
    assert expr.right == right


def test_chained_arithmetic_nests():
    a, b, c = Dim("a"), Dim("b"), Dim("c")
    expr = (a + b) * c
    assert expr.op == "mul"
    assert expr.left == DimExpr("add", a, b)
    assert expr.right == c


@pytest.mark.parametrize("bad", [2.5, "x", None, []], ids=repr)
def test_arithmetic_rejects_invalid_operand_type(bad):
    with pytest.raises(TypeError):
        Dim("a") + bad
    with pytest.raises(TypeError):
        bad - Dim("a")


# ---------------------------------------------------------------------------
# evaluate(dim_sizes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (Dim("m", 3) + Dim("n", 4), 7),
        (Dim("m", 6) - Dim("n", 4), 2),
        (Dim("m", 6) * Dim("n", 4), 24),
        (Dim("m", 7) // Dim("n", 2), 3),
        (Dim("m", 7) % Dim("n", 3), 1),
        (Dim("m", 5).min(Dim("n", 3)), 3),
        (Dim("m", 5).max(Dim("n", 3)), 5),
        (2 - Dim("m", 5), -3),
        (10 // Dim("m", 3), 3),
        (7 % Dim("m", 4), 3),
        (dim(5) + dim(7), 12),
    ],
    ids=["3+4", "6-4", "6*4", "7//2", "7%3", "min", "max", "2-5",
         "10//3", "7%4", "dim(5)+dim(7)"],
)
def test_evaluate_known_sizes_only(expr, expected):
    assert expr.evaluate() == expected


def test_evaluate_with_bindings():
    assert (Dim("a") + Dim("b")).evaluate({"a": 10, "b": 20}) == 30


def test_bindings_take_precedence_over_known_sizes():
    # Binding "n"=7 wins over the dim's own known size 5.
    assert (Dim("n", 5) + 0).evaluate({"n": 7}) == 7


def test_evaluate_missing_binding_raises_shape_error():
    expr = Dim("m") + 1
    with pytest.raises(ShapeError):
        expr.evaluate()
    with pytest.raises(ShapeError):
        expr.evaluate({})
    with pytest.raises(ShapeError):
        expr.evaluate({"other": 4})


def test_evaluate_non_integer_binding_raises_shape_error():
    with pytest.raises(ShapeError):
        (Dim("m") + 1).evaluate({"m": 2.5})


@pytest.mark.parametrize(
    "expr", [10 // Dim("n", 0), 7 % Dim("n", 0)], ids=["div0", "mod0"]
)
def test_evaluate_divmod_by_zero_raises_shape_error(expr):
    with pytest.raises(ShapeError):
        expr.evaluate()


def test_complex_expression_tree_matches_python_semantics():
    a, b, c = Dim("a", 2), Dim("b", 3), Dim("c", 4)
    assert ((a + b) * c // 2 % 3).evaluate() == ((2 + 3) * 4 // 2 % 3)
    assert (a.max(b * 3) - (c // 2).min(b)).evaluate() == max(2, 3 * 3) - min(4 // 2, 3)


def test_complex_expression_tree_with_bindings():
    expr = (Dim("a") + Dim("b")) * Dim("c") // 2 % 3
    assert expr.evaluate({"a": 2, "b": 3, "c": 4}) == ((2 + 3) * 4 // 2 % 3)


# ---------------------------------------------------------------------------
# Structural equality (== is NOT value-based)
# ---------------------------------------------------------------------------

def test_structural_equality_same_structure():
    assert Dim("a", 1) == Dim("a", 1)
    assert Dim("a") == Dim("a")
    assert (Dim("a", 1) + Dim("b", 2)) == (Dim("a", 1) + Dim("b", 2))


def test_structural_equality_different_structure():
    assert Dim("a", 1) != Dim("a")
    assert Dim("a", 1) != Dim("a", 2)
    assert Dim("a") != Dim("b")
    # Equal numeric value (3), different structure → not equal.
    assert (Dim("a", 1) + Dim("b", 2)) != Dim("a", 3)
    # Not commutative, different op, different right operand.
    assert (Dim("a", 1) + Dim("b", 2)) != (Dim("b", 2) + Dim("a", 1))
    assert (Dim("a", 1) + Dim("b", 2)) != (Dim("a", 1) - Dim("b", 2))
    assert (Dim("a", 1) + Dim("b", 2)) != (Dim("a", 1) + Dim("b", 3))
    # Never equal to a plain int (structural comparison only).
    assert (Dim("a", 1) + 1) != 2
    assert Dim("a", 1) != 1


# ---------------------------------------------------------------------------
# Ordering comparisons: constraint-free (known sizes only) → bool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("lhs", "rhs", "cmp", "expected"),
    [
        (Dim("m", 3), Dim("n", 4), operator.lt, True),
        (Dim("m", 3), Dim("n", 3), operator.le, True),
        (Dim("m", 3), Dim("n", 4), operator.gt, False),
        (Dim("m", 3), 3, operator.ge, True),
        (Dim("m", 3), 2, operator.lt, False),
        (2, Dim("m", 3), operator.le, True),
        (Dim("m", 2) + Dim("n", 3), 6, operator.lt, True),
        (Dim("m", 2) * Dim("n", 3), 6, operator.ge, True),
    ],
    ids=["3<4", "3<=3", "3>4", "3>=3", "3<2", "2<=3", "(2+3)<6",
         "(2*3)>=6"],
)
def test_ordering_comparisons_with_known_sizes(lhs, rhs, cmp, expected):
    assert cmp(lhs, rhs) is expected


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (Dim("m"), Dim("n", 4)),
        (Dim("m", 3), Dim("n")),
        (Dim("m") + 1, 5),
        (Dim("m", 3), Dim("n") + 1),
    ],
    ids=["sym-vs-known", "known-vs-sym", "symexpr-vs-int", "known-vs-symexpr"],
)
def test_ordering_unresolved_raises_shape_error(lhs, rhs):
    with pytest.raises(ShapeError):
        lhs < rhs
    with pytest.raises(ShapeError):
        lhs <= rhs


def test_ordering_rejects_invalid_operand_type():
    with pytest.raises(TypeError):
        Dim("m", 3) < 2.5


# ---------------------------------------------------------------------------
# bool() never guesses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [Dim("m"), Dim("m", 5), Dim("a") + Dim("b"), dim(3)],
    ids=["symbolic-dim", "known-dim", "expr", "dim(3)"],
)
def test_bool_raises_shape_error(value):
    with pytest.raises(ShapeError):
        bool(value)
    with pytest.raises(ShapeError):
        not value


# ---------------------------------------------------------------------------
# dim()
# ---------------------------------------------------------------------------

def test_dim_from_int_has_deterministic_name_and_known_size():
    d = dim(5)
    assert d == Dim("dim_5", 5)
    assert d.name == "dim_5"
    assert d.size == 5
    # Deterministic: calling twice yields the same name/size.
    assert dim(5) == dim(5)
    assert dim(5).name == dim(5).name == "dim_5"
    # Known size ⇒ exact evaluation.
    assert (dim(5) + dim(7)).evaluate() == 12


def test_dim_passthrough_returns_the_same_object():
    symbolic = Dim("m")
    known = Dim("m", 3)
    assert dim(symbolic) is symbolic
    assert dim(known) is known


def test_dim_from_str_is_symbolic():
    d = dim("n")
    assert d == Dim("n")
    assert d.size is None


@pytest.mark.parametrize("bad", [None, 1.5, [3], object()], ids=repr)
def test_dim_invalid_inputs_raise_type_error(bad):
    with pytest.raises(TypeError):
        dim(bad)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

def test_importable_from_etl_and_etl_core():
    from etl import Dim as D1, DimExpr as E1, ShapeError as S1, dim as f1
    from etl.core import Dim as D2, DimExpr as E2, ShapeError as S2, dim as f2

    assert D1 is D2
    assert E1 is E2
    assert S1 is S2
    assert f1 is f2
    # Also reachable as attributes of the package modules.
    assert etl.Dim is etl.core.Dim
    assert etl.dim is etl.core.dim
