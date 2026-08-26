"""Reduction ops and the user-facing reduction sugar.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``axes`` (reduce_*) / ``axis`` (argmax/argmin) are STATIC: ``None`` = all
  axes, int, or tuple of ints; normalized via ``_utils.normalize_axes``.
- Output shape via ``_utils.reduced_shape`` (symbolic dims preserved for
  unreduced axes).
- ``reduce_mean`` follows numpy: integer/bool input → ``float64`` result;
  float keeps dtype.
- ``argmax``/``argmin`` produce ``int64`` index tensors (numpy convention).
- The sugar functions (``sum``/``max``/``min``/``mean``/``prod``) are
  DOCUMENTED SHORTHAND ONLY — their exact expansion onto ``reduce_*`` is
  stated in each docstring; they add no hidden semantics.
"""
from __future__ import annotations

from etl import core

from . import _utils

__all__ = [
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod",
    "sum", "max", "min", "mean", "prod", "argmax", "argmin",
]

#: ``reduce_op`` attribute value for each reduce_* op name (IR contract).
_REDUCE_KINDS = {
    "reduce_sum": "sum",
    "reduce_max": "max",
    "reduce_min": "min",
    "reduce_mean": "mean",
    "reduce_prod": "prod",
}


def _wrap(op, location) -> "core.SymbolicTensor":
    """Wrap an op's single IR result value in a ``SymbolicTensor``.

    The result dtype/shape are READ BACK from the IR value type (the opdef
    shape_fn applies the numpy dtype rules per ``reduce_op``).
    """
    result_type = op.result.type
    return core.SymbolicTensor(
        value=op.result,
        dtype=result_type.dtype,
        shape=result_type.shape,
        location=location,
    )


def _reduce(x, axes, keepdims, op_name: str) -> "core.SymbolicTensor":
    """Shared implementation of the reduce_* family (see each docstring).

    Axes are normalized frontend-side: ``None`` → all axes; an explicit empty
    ``()`` on a rank ≥ 1 tensor is rejected (the IR treats an empty ``axes``
    tuple as ALL axes — passing ``()`` through would silently change the
    semantics); a rank-0 input passes ``()`` (scalar identity semantics, with
    the reduction's dtype rule applied by the IR).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(
            f"{op_name}: keepdims must be a bool, got {keepdims!r}"
        )
    rank = len(x.shape)
    if axes is None:
        normalized = tuple(range(rank))
    elif axes == ():
        if rank >= 1:
            raise core.ShapeError(
                f"{op_name}: reducing no axes (axes=()) is not supported; "
                "pass None to reduce all axes"
            )
        normalized = ()
    else:
        # TypeError for malformed axes specs; ShapeError for out-of-range.
        normalized = _utils.normalize_axes(axes, rank)
    op = builder.create(
        op_name,
        operands=(x.value,),
        attributes={
            "axes": normalized,
            "keepdims": keepdims,
            "reduce_op": _REDUCE_KINDS[op_name],
        },
        location=loc,
    )
    expected_shape = _utils.reduced_shape(x.shape, normalized, keepdims)
    inferred_shape = op.result.type.shape
    if tuple(expected_shape) != tuple(inferred_shape):
        raise core.ShapeError(
            f"{op_name}: IR-inferred shape {inferred_shape!r} does not match "
            f"the frontend reduction shape {expected_shape!r}"
        )
    return _wrap(op, loc)


def _arg_reduce(x, axis, keepdims, op_name: str) -> "core.SymbolicTensor":
    """Shared implementation of argmax/argmin (see each docstring).

    ``axis=None`` flattens (IR semantics: scalar index, or all-1 shape with
    ``keepdims``); an int axis is normalized frontend-side (negative shifted)
    and range-checked. Result dtype is int64 (applied by the IR shape_fn).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(
            f"{op_name}: keepdims must be a bool, got {keepdims!r}"
        )
    rank = len(x.shape)
    if axis is not None:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(
                f"{op_name}: axis must be None or an int, got {axis!r}"
            )
        original = axis
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            raise core.ShapeError(
                f"{op_name}: axis {original} out of range for rank {rank}"
            )
    op = builder.create(
        op_name,
        operands=(x.value,),
        attributes={"axis": axis, "keepdims": keepdims},
        location=loc,
    )
    return _wrap(op, loc)


def reduce_sum(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Sum of elements over the given axes.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axes: ``None`` (all axes → scalar), int, or tuple of ints.
        keepdims: Keep reduced axes as extent-1 dims.

    Returns:
        ``SymbolicTensor`` with reduced shape; dtype = numpy ``sum`` result
        dtype (bool → int64, else preserved; see numpy).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range.
    """
    return _reduce(x, axes, keepdims, "reduce_sum")


def reduce_max(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Maximum of elements over the given axes.

    Dtype preserved; shape reduced. Args as in :func:`reduce_sum`.
    """
    return _reduce(x, axes, keepdims, "reduce_max")


def reduce_min(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Minimum of elements over the given axes.

    Dtype preserved; shape reduced. Args as in :func:`reduce_sum`.
    """
    return _reduce(x, axes, keepdims, "reduce_min")


def reduce_mean(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Arithmetic mean of elements over the given axes.

    Dtype rule (numpy): integer/bool input → ``float64`` result; float input
    keeps its dtype. Shape reduced. Args as in :func:`reduce_sum`.
    """
    return _reduce(x, axes, keepdims, "reduce_mean")


def reduce_prod(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Product of elements over the given axes.

    Dtype rule (numpy): bool input → int64, else preserved. Shape reduced.
    Args as in :func:`reduce_sum`.
    """
    return _reduce(x, axes, keepdims, "reduce_prod")


def sum(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: ``sum(x, axes, keepdims)`` is EXACTLY
    ``reduce_sum(x, axes=axes, keepdims=keepdims)`` — no other semantics.
    (Shadows the Python builtin inside this module on purpose.)"""
    return reduce_sum(x, axes=axes, keepdims=keepdims)


def max(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_max(x, axes=axes,
    keepdims=keepdims)``. (Shadows the Python builtin inside this module on
    purpose.)"""
    return reduce_max(x, axes=axes, keepdims=keepdims)


def min(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_min(x, axes=axes,
    keepdims=keepdims)``. (Shadows the Python builtin inside this module on
    purpose.)"""
    return reduce_min(x, axes=axes, keepdims=keepdims)


def mean(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_mean(x, axes=axes,
    keepdims=keepdims)``. Integer/bool input → float64 (see
    :func:`reduce_mean`)."""
    return reduce_mean(x, axes=axes, keepdims=keepdims)


def prod(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_prod(x, axes=axes,
    keepdims=keepdims)``."""
    return reduce_prod(x, axes=axes, keepdims=keepdims)


def argmax(x, axis=None, keepdims=False) -> "core.SymbolicTensor":
    """Index of the maximum along an axis (numpy ``argmax`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: ``None`` (flatten-and-reduce over all axes, yielding a scalar
            index — consistent with ``reduce_*``'s ``axes=None``), or an int.
        keepdims: Keep the reduced axis as extent 1.

    Returns:
        ``SymbolicTensor`` of dtype ``int64``; reduced shape.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range.
    """
    return _arg_reduce(x, axis, keepdims, "argmax")


def argmin(x, axis=None, keepdims=False) -> "core.SymbolicTensor":
    """Index of the minimum along an axis (numpy ``argmin`` semantics).

    Dtype ``int64``; rules identical to :func:`argmax`.
    """
    return _arg_reduce(x, axis, keepdims, "argmin")
