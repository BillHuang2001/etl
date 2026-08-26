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

Implementation note: dtype/shape inference is delegated to the canonical
``etl.ir`` op registry (``infer_compare`` / ``infer_select`` /
``infer_elementwise_binary``) — result types are READ BACK from the inferred
``op.result.type``; this module keeps no parallel inference table.
"""
from __future__ import annotations

from etl import core

from . import _utils

__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "select",
]

# --- private construction helpers -------------------------------------------

#: Exact Python scalar kinds accepted as operands (the same set
#: ``_utils.as_operand`` accepts; numpy scalars are deliberately NOT included).
_SCALAR_KINDS = (bool, int, float, complex)


def _is_scalar(x) -> bool:
    """True when ``x`` is an exact Python scalar (bool/int/float/complex)."""
    return type(x) in _SCALAR_KINDS


def _wrap(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result in a SymbolicTensor, reading the dtype and
    shape back from the IR's inferred result type (never computed here)."""
    return core.SymbolicTensor(
        value=op.result,
        dtype=op.result.type.dtype,
        shape=op.result.type.shape,
        location=loc,
    )


def _binary_operands(op_name, x, y, loc):
    """Normalize the two operands of a binary op per the unified rules.

    ``SymbolicTensor`` operands pass through unchanged. A Python scalar is
    promoted to a 0-d Constant whose dtype is weakly pre-promoted against the
    OTHER (symbolic) operand's dtype — NEP 50 semantics, see
    ``_utils.as_operand``.

    Raises:
        core.TraceError: both operands are Python scalars (etl has no eager
            mode), or a concrete ``Tensor`` appears (canonical message via
            ``as_operand``).
        TypeError: unsupported operand kind (via ``as_operand``).
    """
    if isinstance(x, core.SymbolicTensor):
        return x, _utils.as_operand(y, dtype_hint=x.dtype, location=loc)
    if isinstance(y, core.SymbolicTensor):
        return _utils.as_operand(x, dtype_hint=y.dtype, location=loc), y
    if _is_scalar(x) and _is_scalar(y):
        raise core.TraceError(
            f"{op_name}: at least one operand must be a SymbolicTensor, got "
            "two Python scalars. etl has no eager mode — trace a graph with "
            "etl.trace or @etl.defn, or build and run one with etl.evaluate."
        )
    # Neither operand is symbolic and they are not both scalars: at least one
    # side is a concrete Tensor or an unsupported kind. Probe the non-scalar
    # side first so no spurious Constant op is built on the error path;
    # as_operand raises the canonical TraceError/TypeError.
    if not _is_scalar(x):
        _utils.as_operand(x, dtype_hint=None, location=loc)  # always raises
    _utils.as_operand(y, dtype_hint=None, location=loc)  # always raises
    raise AssertionError(  # pragma: no cover — as_operand raises above
        "unreachable: as_operand raises for non-symbolic, non-scalar operands"
    )


def _emit_binary(builder, op_name, xt, yt, loc) -> "core.SymbolicTensor":
    """Build the two-operand op and wrap its inferred result."""
    op = builder.create(op_name, operands=(xt.value, yt.value), location=loc)
    return _wrap(op, loc)


def _binary(op_name, x, y) -> "core.SymbolicTensor":
    """Shared body of the comparison ops (bool results, broadcast shapes)."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands(op_name, x, y, loc)
    return _emit_binary(builder, op_name, xt, yt, loc)


def _require_bool(op_name, *tensors) -> None:
    """Raise ``core.DTypeError`` when any tensor does not have bool dtype."""
    for t in tensors:
        if t.dtype.kind != "b":
            raise core.DTypeError(
                f"{op_name}: operands must have bool dtype, got {t.dtype}"
            )


# --- comparisons (bool results) ------------------------------------------------


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
    return _binary("equal", x, y)


def not_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise inequality (``x != y``) → bool. Rules identical to
    :func:`equal`."""
    return _binary("not_equal", x, y)


def less(x, y) -> "core.SymbolicTensor":
    """Elementwise strict less-than (``x < y``) → bool.

    Registered as the ``SymbolicTensor.__lt__`` operator handler (kind
    ``lt``). Rules identical to :func:`equal`.
    """
    return _binary("less", x, y)


def less_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise less-or-equal (``x <= y``) → bool.

    Registered as the ``SymbolicTensor.__le__`` operator handler (kind
    ``le``). Rules identical to :func:`equal`.
    """
    return _binary("less_equal", x, y)


def greater(x, y) -> "core.SymbolicTensor":
    """Elementwise strict greater-than (``x > y``) → bool.

    Registered as the ``SymbolicTensor.__gt__`` operator handler (kind
    ``gt``). Rules identical to :func:`equal`.
    """
    return _binary("greater", x, y)


def greater_equal(x, y) -> "core.SymbolicTensor":
    """Elementwise greater-or-equal (``x >= y``) → bool.

    Registered as the ``SymbolicTensor.__ge__`` operator handler (kind
    ``ge``). Rules identical to :func:`equal`.
    """
    return _binary("greater_equal", x, y)


# --- logical (bool operands only) ----------------------------------------------


def logical_and(x, y) -> "core.SymbolicTensor":
    """Elementwise logical AND (``x & y`` over bools) → bool.

    Both operands must be bool (``core.DTypeError`` otherwise). Shape =
    broadcast of inputs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("logical_and", x, y, loc)
    _require_bool("logical_and", xt, yt)
    return _emit_binary(builder, "logical_and", xt, yt, loc)


def logical_or(x, y) -> "core.SymbolicTensor":
    """Elementwise logical OR (``x | y`` over bools) → bool.

    Both operands must be bool (``core.DTypeError`` otherwise). Shape =
    broadcast of inputs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("logical_or", x, y, loc)
    _require_bool("logical_or", xt, yt)
    return _emit_binary(builder, "logical_or", xt, yt, loc)


def logical_not(x) -> "core.SymbolicTensor":
    """Elementwise logical NOT (``~x`` over bools) → bool.

    The operand must be bool (``core.DTypeError`` otherwise). Shape
    preserved.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt = _utils.as_operand(x, dtype_hint=None, location=loc)
    _require_bool("logical_not", xt)
    op = builder.create("logical_not", operands=(xt.value,), location=loc)
    return _wrap(op, loc)


# --- selection -----------------------------------------------------------------


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    # pred: bool SymbolicTensor or Python bool scalar — anything else fails.
    if isinstance(pred, core.SymbolicTensor):
        pred_t = pred
        if pred_t.dtype.kind != "b":
            raise core.DTypeError(
                f"select: pred must have bool dtype, got {pred_t.dtype}"
            )
    else:
        pred_t = _utils.as_operand(pred, dtype_hint=None, location=loc)
        if pred_t.dtype.kind != "b":
            raise core.DTypeError(
                "select: pred must be a bool SymbolicTensor or a Python bool "
                f"scalar, got {type(pred).__name__} (dtype {pred_t.dtype})"
            )
    # Branch operands with mutual weak-promotion hints: a Python scalar
    # branch promotes down toward the other branch's dtype (NEP 50).
    if isinstance(on_true, core.SymbolicTensor):
        on_t = on_true
        on_f = _utils.as_operand(on_false, dtype_hint=on_true.dtype, location=loc)
    elif isinstance(on_false, core.SymbolicTensor):
        on_f = on_false
        on_t = _utils.as_operand(on_true, dtype_hint=on_false.dtype, location=loc)
    else:
        on_t = _utils.as_operand(on_true, dtype_hint=None, location=loc)
        on_f = _utils.as_operand(on_false, dtype_hint=None, location=loc)
    op = builder.create(
        "select", operands=(pred_t.value, on_t.value, on_f.value), location=loc
    )
    return _wrap(op, loc)
