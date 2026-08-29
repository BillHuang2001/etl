"""Statistics ops: var, std, median, nansum — numpy semantics as documented
compositions over ordinary ops (no dedicated IR ops, no hidden semantics).

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Exact expansions:
- ``var``: ``mean(square(x - mean(x, keepdims=True)))`` for ``ddof=0``;
  ``reduce_sum(...) / (n - ddof)`` otherwise (``n`` = reduced element count,
  static when all reduced dims are static, else computed in-graph).
- ``std``: ``sqrt(var(...))``.
- ``median``: cast to ``float64`` (numpy always returns float64), ``sort``,
  take the middle one-or-two elements via ``slice``, ``reduce_mean``.
- ``nansum``: ``reduce_sum(select(x != x, 0, x))`` (``x != x`` is the
  NaN test).
"""
from __future__ import annotations

from etl import core

from . import _utils
from .comparison import not_equal, select
from .elementwise import cast, divide, maximum, square, sqrt, subtract
from .indexing import broadcast, reshape, slice as etl_slice
from .linalg import sort
from .reductions import mean, reduce_mean, reduce_sum

__all__ = ["var", "std", "median", "nansum"]


def _reduced_count(shape, axes) -> int:
    """Product of the extents of the given (normalized) axes.

    Returns a plain ``int`` when every reduced extent is static, else a
    ``DimExpr``/``Dim`` (the caller then builds the count in-graph).
    """
    count = 1
    for axis, extent in enumerate(shape):
        if axis in axes:
            count = count * extent
    return count


def var(x, axes=None, keepdims=False, ddof=0) -> "core.SymbolicTensor":
    """Variance (numpy ``var`` semantics).
    Documented composition: ``m = mean(x, axes, keepdims=True)``;
    ``sqdev = square(x - m)``; ``ddof=0`` → ``mean(sqdev, axes, keepdims)``;
    ``ddof != 0`` → ``reduce_sum(sqdev, axes, keepdims) / (n - ddof)`` with
    ``n`` = product of the reduced extents (static when all reduced dims are
    static; computed in-graph otherwise).
    Args:
        x: ``SymbolicTensor``.
        axes: None (all axes), int, or tuple of ints — the axes to reduce.
        keepdims: bool; keep reduced axes as extent 1.
        ddof: int; delta degrees of freedom (divisor ``n - ddof``).
    Returns:
        ``SymbolicTensor`` with the reduced shape. Dtype per numpy: integer
        input → ``float64``; float keeps its dtype.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range; ``axes=()`` on rank >= 1.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(f"var: keepdims must be a bool, got {keepdims!r}")
    if not isinstance(ddof, int) or isinstance(ddof, bool):
        raise TypeError(f"var: ddof must be an int, got {ddof!r}")
    rank = len(x_sym.shape)
    normalized = _utils.normalize_axes(axes, rank)
    centered = mean(x_sym, axes=normalized, keepdims=True)
    sqdev = square(subtract(x_sym, centered))
    if ddof == 0:
        return mean(sqdev, axes=normalized, keepdims=keepdims)
    count = _reduced_count(x_sym.shape, normalized)
    total = reduce_sum(sqdev, axes=normalized, keepdims=keepdims)
    if isinstance(count, int):
        # numpy clamps the divisor at 0 (max(n - ddof, 0)) and divides raw:
        # 0/0 → NaN, positive/0 → +inf — match exactly.
        return divide(total, max(count - ddof, 0))
    # Symbolic reduced extents: build the divisor in-graph so the result
    # stays exact under runtime dims. The divisor is cast to the total's
    # dtype (the in-graph count is int; dividing a float32 total by an int
    # tensor would promote to float64, unlike numpy) and clamped at 0 per
    # numpy.
    ones = _utils.as_operand(1, location=loc)
    count_op = reduce_sum(broadcast(ones, x_sym.shape), axes=normalized)
    divisor = maximum(cast(subtract(count_op, ddof), total.dtype), 0)
    return divide(total, divisor)


def std(x, axes=None, keepdims=False, ddof=0) -> "core.SymbolicTensor":
    """Standard deviation (numpy ``std`` semantics).
    Documented composition: ``sqrt(var(x, axes, keepdims, ddof))``.
    Args:
        x: ``SymbolicTensor``.
        axes: None (all axes), int, or tuple of ints — the axes to reduce.
        keepdims: bool; keep reduced axes as extent 1.
        ddof: int; delta degrees of freedom (divisor ``n - ddof``).
    Returns:
        ``SymbolicTensor`` with the reduced shape. Dtype per numpy: integer
        input → ``float64``; float keeps its dtype.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range; ``axes=()`` on rank >= 1.
    """
    return sqrt(var(x, axes=axes, keepdims=keepdims, ddof=ddof))


def _median_along_axis(x, axis, keepdims) -> "core.SymbolicTensor":
    """Median along one axis of a rank >= 1 tensor.

    v1 scope: every extent must be STATIC (the ``slice`` composition needs
    static limit indices on every axis; the middle index also needs a static
    axis extent) — symbolic shapes raise an explicit ``ShapeError``. The
    middle one-or-two sorted elements are sliced and averaged (float64 by
    construction — the caller cast to float64)."""
    if not all(isinstance(d, int) for d in x.shape):
        raise core.ShapeError(
            f"median: symbolic extents {x.shape!r} are not supported in v1 "
            "(the slice composition requires static limit indices); use "
            "static shapes"
        )
    n = x.shape[axis]
    rank = len(x.shape)
    sorted_x = sort(x, axis=axis)
    start = [0] * rank
    lengths = list(x.shape)
    if n % 2 == 1:
        start[axis] = n // 2
        lengths[axis] = 1
    else:
        start[axis] = n // 2 - 1
        lengths[axis] = 2
    middle = etl_slice(sorted_x, tuple(start), tuple(lengths))
    return reduce_mean(middle, axes=(axis,), keepdims=keepdims)


def median(x, axis=None, keepdims=False) -> "core.SymbolicTensor":
    """Median (numpy ``median`` semantics: always returns ``float64``).
    Documented composition: cast to ``float64``, ``sort`` along the axis,
    ``slice`` the middle one-or-two elements, ``reduce_mean``. ``axis=None``
    flattens first (numpy semantics).
    Args:
        x: ``SymbolicTensor``.
        axis: None (flatten), or int — the axis to reduce.
        keepdims: bool; keep the reduced axis as extent 1.
    Returns:
        ``SymbolicTensor`` of dtype ``float64`` with the reduced shape.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range; non-static axis extent.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(f"median: keepdims must be a bool, got {keepdims!r}")
    rank = len(x_sym.shape)
    if rank == 0:
        # Median of a scalar is the scalar itself (float64 per numpy).
        return cast(x_sym, core.float64)
    x64 = cast(x_sym, core.float64)
    if axis is None:
        flat = reshape(x64, (-1,))
        result = _median_along_axis(flat, 0, keepdims=False)
        if keepdims:
            result = reshape(result, (1,) * rank)
        return result
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"median: axis must be an int or None, got {axis!r}")
    if not -rank <= axis < rank:
        raise core.ShapeError(
            f"median: axis {axis!r} out of range for rank {rank}"
        )
    return _median_along_axis(x64, axis % rank, keepdims)


def nansum(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Sum treating NaN as 0 (numpy ``nansum`` semantics).
    Documented composition: ``reduce_sum(select(x != x, 0, x), axes,
    keepdims)`` — ``x != x`` is the NaN test (works for float and complex).
    Args:
        x: ``SymbolicTensor``.
        axes: None (all axes), int, or tuple of ints — the axes to reduce.
        keepdims: bool; keep reduced axes as extent 1.
    Returns:
        ``SymbolicTensor`` with the reduced shape; dtype per numpy
        ``nansum`` (integer input → ``int64``, float keeps its dtype).
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range; ``axes=()`` on rank >= 1.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(f"nansum: keepdims must be a bool, got {keepdims!r}")
    isnan = not_equal(x_sym, x_sym)
    cleaned = select(isnan, 0, x_sym)
    return reduce_sum(cleaned, axes=axes, keepdims=keepdims)
