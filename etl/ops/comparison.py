"""Comparison, logical, and selection ops.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- Comparison ops (``equal`` ... ``greater_equal``) produce bool results and
  broadcast their inputs via ``_utils.broadcast_shapes``.
- Logical ops (``logical_and``/``logical_or``/``logical_not``) require bool
  operands (``core.DTypeError`` otherwise), mirroring numpy.
- ``select`` broadcasts all three inputs together and promotes the dtype of
  the two branches.
"""
from __future__ import annotations

from etl import core
from etl import ir

from . import _utils

__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "select",
]


def equal(x, y) -> "core.SymbolicTensor":
    """Elementwise equality (``x == y``) → bool.

    Registered as the ``SymbolicTensor.__eq__`` operator handler (kind
    ``eq``). Result dtype is ``bool`` regardless of input dtypes; shape =
    broadcast of inputs.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand; both
            operands are Python scalars.
        core.ShapeError: static broadcast incompatibility.
    """
    raise NotImplementedError


def not_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise inequality (``x != y``) → bool. Rules identical to
    :func:`equal`."""
    raise NotImplementedError


def less(x, y) -> "core.SymbolicTensor":
    """Elementwise strict less-than (``x < y``) → bool.

    Registered as the ``SymbolicTensor.__lt__`` operator handler (kind
    ``lt``). Rules identical to :func:`equal`.
    """
    raise NotImplementedError


def less_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise less-or-equal (``x <= y``) → bool.

    Registered as the ``SymbolicTensor.__le__`` operator handler (kind
    ``le``). Rules identical to :func:`equal`.
    """
    raise NotImplementedError


def greater(x, y) -> "core.SymbolicTensor":
    """Elementwise strict greater-than (``x > y``) → bool.

    Registered as the ``SymbolicTensor.__gt__`` operator handler (kind
    ``gt``). Rules identical to :func:`equal`.
    """
    raise NotImplementedError


def greater_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise greater-or-equal (``x >= y``) → bool.

    Registered as the ``SymbolicTensor.__ge__`` operator handler (kind
    ``ge``). Rules identical to :func:`equal`.
    """
    raise NotImplementedError


def logical_and(x, y) -> "core.SymbolicTensor":
    """Elementwise logical AND (``x & y`` over bools) → bool.

    Both operands must be bool (``core.DTypeError`` otherwise). Shape =
    broadcast of inputs.
    """
    raise NotImplementedError


def logical_or(x, y) -> "core.SymbolicTensor":
    """Elementwise logical OR (``x | y`` over bools) → bool.

    Both operands must be bool (``core.DTypeError`` otherwise). Shape =
    broadcast of inputs.
    """
    raise NotImplementedError


def logical_not(x) -> "core.SymbolicTensor":
    """Elementwise logical NOT (``~x`` over bools) → bool.

    The operand must be bool (``core.DTypeError`` otherwise). Shape
    preserved.
    """
    raise NotImplementedError


def select(pred, on_true, on_false) -> "core.SymbolicTensor":
    """Elementwise ternary selection: ``on_true`` where ``pred`` is truthy,
    ``on_false`` elsewhere (numpy ``where`` semantics, SSA — both branches
    are always evaluated).

    Args:
        pred: bool ``SymbolicTensor`` or Python bool scalar.
        on_true: ``SymbolicTensor`` or Python scalar.
        on_false: ``SymbolicTensor`` or Python scalar.

    Returns:
        ``SymbolicTensor`` with ``shape = broadcast_shapes(pred.shape,
        on_true.shape, on_false.shape)`` and ``dtype =
        promote_dtypes(on_true.dtype, on_false.dtype)``.

    Raises:
        core.DTypeError: ``pred`` is not bool.
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: static broadcast incompatibility.
    """
    raise NotImplementedError
