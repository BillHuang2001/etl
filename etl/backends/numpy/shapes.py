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

from typing import Mapping, Sequence

from etl.core import Dim, DimExpr, ShapeError

__all__ = ["evaluate_dim_expr", "evaluate_shape"]


def evaluate_dim_expr(expr: DimExpr | Dim | int, bindings: Mapping[str, int]) -> int:
    """Evaluate a single dimension expression against name->int bindings.

    Rules (implement in the implementation phase):
    - ``int`` -> itself (concrete dims pass through).
    - ``core.Dim(name)`` -> ``bindings[name]``; a name missing from
      ``bindings`` is a FREE symbolic dim => ``core.ShapeError`` naming it.
    - ``core.DimExpr`` -> evaluate the DimExpr arithmetic
      (``+ - * // % min max``) with nested Dims/DimExprs recursively bound to
      ints; the result must be a non-negative int (``core.ShapeError``
      otherwise).

    Symbolic-dim binding logic lives ONLY here — runtime shapes are computed
    by evaluating the ops-level ``DimExpr`` output against the concrete input
    dims recorded for the entry function's block args.
    """
    raise NotImplementedError(
        "architecture stub: implement DimExpr evaluation in the implementation phase"
    )


def evaluate_shape(
    shape: Sequence[DimExpr | Dim | int], bindings: Mapping[str, int]
) -> tuple[int, ...]:
    """Evaluate a whole shape (per-entry ``evaluate_dim_expr``) to concrete ints.

    Entries may be ints, ``core.Dim``, or ``core.DimExpr``. Free symbolic dims
    => ``core.ShapeError``. Used by the interpreter to derive runtime output
    shapes (reusing ops-level inference rules — no second copy of shape rules
    in the backend).
    """
    raise NotImplementedError(
        "architecture stub: implement per-entry shape evaluation in the implementation phase"
    )
