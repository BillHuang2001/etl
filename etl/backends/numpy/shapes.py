"""Runtime shape evaluation for the numpy interpreter.

Design (binding, parent CONTEXT.md): shape inference REUSE, not duplication.
Runtime shapes are computed from concrete input shapes using the SAME
ops-level inference rules with symbolic dims bound to concrete values. This
module evaluates ``core.Dim`` / ``core.DimExpr`` shape expressions against
name->int concrete-dim bindings; the backend carries NO second copy of shape
rules.

Free symbolic dims (names absent from ``bindings``) at run time raise
``core.ShapeError`` — the interpreter never guesses a dimension.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Union

from etl.core import Dim, DimExpr, ShapeError

__all__ = ["evaluate_dim_expr", "evaluate_shape"]

#: What one shape entry may be: a concrete int, a symbolic Dim, or a DimExpr.
DimEntry = Union[DimExpr, Dim, int]


def _resolve_dim(expr: DimEntry, bindings: Mapping[str, int]) -> int:
    """Recursive evaluation of a dim expression (no negativity check).

    ``int`` -> itself; ``Dim(name)`` -> ``bindings[name]`` if present, else
    the Dim's own known ``size``, else ``ShapeError`` naming the dim;
    ``DimExpr`` -> integer arithmetic over the recursively evaluated operands
    (``floordiv`` -> ``//``); division/modulo by zero -> ``ShapeError``.

    Nested sub-expressions may legally be negative (e.g. ``B - 2`` inside
    ``max(B - 2, 0)``) — the final-result negativity check lives in the public
    :func:`evaluate_dim_expr`.
    """
    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    if isinstance(expr, Dim):
        if expr.name in bindings:
            value = bindings[expr.name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ShapeError(
                    f"Binding for dimension {expr.name!r} must be an int, "
                    f"got {value!r}"
                )
            return value
        if expr.size is not None:
            return expr.size
        raise ShapeError(
            f"Cannot evaluate dimension {expr.name!r}: it has no runtime "
            "binding and no known size"
        )
    if isinstance(expr, DimExpr):
        left = _resolve_dim(expr.left, bindings)
        right = _resolve_dim(expr.right, bindings)
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "floordiv":
            if right == 0:
                raise ShapeError(f"Division by zero in DimExpr {expr!r}")
            return left // right
        if expr.op == "mod":
            if right == 0:
                raise ShapeError(f"Modulo by zero in DimExpr {expr!r}")
            return left % right
        if expr.op == "min":
            return min(left, right)
        if expr.op == "max":
            return max(left, right)
        raise ValueError(f"Unknown DimExpr op: {expr.op!r}")  # unreachable
    raise TypeError(
        f"Expected a dimension entry of type int, Dim or DimExpr, got "
        f"{type(expr).__name__}: {expr!r}"
    )


def evaluate_dim_expr(expr: DimExpr | Dim | int, bindings: Mapping[str, int]) -> int:
    """Evaluate a single dimension expression against name->int bindings.

    Rules:
    - ``int`` -> itself (concrete dims pass through).
    - ``core.Dim(name)`` -> ``bindings[name]``; a name missing from
      ``bindings`` uses the Dim's own known ``size``; otherwise it is a FREE
      symbolic dim => ``core.ShapeError`` naming it.
    - ``core.DimExpr`` -> integer semantics over the recursively bound
      operands (``floordiv`` -> ``//``); division/modulo by zero =>
      ``core.ShapeError``.
    - A NEGATIVE final result => ``core.ShapeError`` (dimensions are sizes).

    Symbolic-dim binding logic lives ONLY here — runtime shapes are computed
    by evaluating the ops-level ``DimExpr`` output against the concrete input
    dims recorded for the entry function's block args.
    """
    result = _resolve_dim(expr, bindings)
    if result < 0:
        raise ShapeError(
            f"Dimension expression {expr!r} evaluated to a negative size "
            f"{result}"
        )
    return result


def evaluate_shape(
    shape: Sequence[DimExpr | Dim | int], bindings: Mapping[str, int]
) -> tuple[int, ...]:
    """Evaluate a whole shape (per-entry ``evaluate_dim_expr``) to concrete ints.

    Entries may be ints, ``core.Dim``, or ``core.DimExpr`` — ``None`` is NOT
    accepted here: callers resolve runtime-dynamic (``None``) dims from
    concrete runtime data first (the interpreter does this when validating
    results against op result types). Free symbolic dims =>
    ``core.ShapeError``. Used by the interpreter to derive runtime output
    shapes (reusing ops-level inference rules — no second copy of shape rules
    in the backend).
    """
    result = []
    for dim in shape:
        if dim is None:
            raise ShapeError(
                "evaluate_shape does not accept None (runtime-dynamic) dims — "
                "resolve them from concrete runtime data before evaluating"
            )
        result.append(evaluate_dim_expr(dim, bindings))
    return tuple(result)
