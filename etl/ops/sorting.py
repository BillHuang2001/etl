"""Sorting frontend: ``sort``, ``argsort``, ``topk`` (graph ops).

Semantics follow numpy (``sort``/``argsort``); ``topk`` is a pure composition
over ``sort``/``argsort`` + ``gather`` of a static ``0..k-1`` index constant
(no dedicated IR op — see the design note in this node's CONTEXT.md):

- ``topk`` values/indices are the first ``k`` entries of the axis-sorted
  tensor (descending for ``largest=True``, ascending otherwise) — for a
  static axis extent ``k <= extent`` is validated at trace time
  (``ShapeError``); for a symbolic extent the runtime ``gather`` kernel
  raises an explicit ``ShapeError`` when ``k`` exceeds the extent (numpy
  ``take`` raises ``IndexError`` for out-of-bounds indices, converted by the
  kernel — never silent truncation).
- ``sort``/``argsort`` with ``axis=None`` flatten the operand first
  (``reshape(-1)`` + axis-0 sort), matching ``np.sort``/``np.argsort``.
- ``descending`` is implemented kernel-side as a flip of the ascending
  result along the axis; ``stable`` selects numpy's stable sort kind.

Transform coverage: no vjp/batching rules (documented ``TransformError``),
following the random-op pattern.
"""
from __future__ import annotations

import numpy as np

from etl import core

from . import _utils
from . import indexing as _indexing

__all__ = ["sort", "argsort", "topk"]


def _wrap(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result, reading dtype/shape back from the IR."""
    result = op.result
    return core.SymbolicTensor(
        value=result,
        dtype=result.type.dtype,
        shape=result.type.shape,
        location=loc,
    )


def _sort_axis(axis, rank: int) -> int:
    """Normalize a single sort axis; ``None`` is handled by the caller
    (flatten composition). Scalar operands fail like numpy's AxisError."""
    return _utils.normalize_axes(axis, rank)[0]


def _emit_sort(builder, x, axis, descending, stable, op_name: str, loc):
    """Shared IR emission for ``sort``/``argsort`` (identical attributes)."""
    op = builder.create(
        op_name,
        operands=(x.value,),
        attributes={"axis": axis, "descending": descending, "stable": stable},
        location=loc,
    )
    return _wrap(op, loc)


def _sort_common(x, axis, descending, stable, op_name: str):
    """Common body of ``sort``/``argsort``."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if not isinstance(descending, bool):
        raise TypeError(f"{op_name}: descending must be a bool, got {descending!r}")
    if not isinstance(stable, bool):
        raise TypeError(f"{op_name}: stable must be a bool, got {stable!r}")
    if axis is None:
        # numpy: flatten then sort along axis 0 (1-D result).
        flat = _indexing.reshape(x, (-1,))
        return _emit_sort(builder, flat, 0, descending, stable, op_name, loc)
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(
            f"{op_name}: axis must be an int or None, got {axis!r}"
        )
    axis_norm = _sort_axis(axis, len(x.shape))
    return _emit_sort(builder, x, axis_norm, descending, stable, op_name, loc)


def sort(x, axis=-1, descending=False, stable=False) -> "core.SymbolicTensor":
    """Sort the tensor along an axis (numpy ``sort`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: int; axis to sort along (``None`` = flatten first, 1-D
            result). Default -1 (last axis).
        descending: bool; sort in descending order (implemented as a flip of
            the ascending result along the axis).
        stable: bool; use numpy's stable sort kind.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x`` (1-D when
        ``axis=None``).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``axis``/``descending``/``stable`` of the wrong kind.
        core.ShapeError: axis out of range (scalar operands included, like
            numpy's ``AxisError``); negative ``reps``.
    """
    return _sort_common(x, axis, descending, stable, "sort")


def argsort(x, axis=-1, descending=False, stable=False) -> "core.SymbolicTensor":
    """Indices that would sort the tensor along an axis (numpy ``argsort``
    semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: int; axis to sort along (``None`` = flatten first, 1-D
            result). Default -1 (last axis).
        descending: bool; sort in descending order (implemented as a flip of
            the ascending indices along the axis).
        stable: bool; use numpy's stable sort kind.

    Returns:
        ``SymbolicTensor`` of int64 indices with the same shape as ``x``
        (1-D when ``axis=None``).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``axis``/``descending``/``stable`` of the wrong kind.
        core.ShapeError: axis out of range (scalar operands included, like
            numpy's ``AxisError``).
    """
    return _sort_common(x, axis, descending, stable, "argsort")


def topk(x, k, axis=-1, largest=True) -> tuple["core.SymbolicTensor", "core.SymbolicTensor"]:
    """Top-k values and their indices along an axis.

    Composition: ``sort``/``argsort`` (descending for ``largest=True``,
    ascending otherwise) followed by a ``gather`` of the static index range
    ``0..k-1`` along the axis — no dedicated IR op. ``k`` is a static
    non-negative int; ``k <=`` the static axis extent is validated at trace
    time (``ShapeError``); a symbolic axis extent defers the check to run
    time, where the gather kernel raises an explicit ``ShapeError`` when
    ``k`` exceeds the extent (never silent truncation).

    Args:
        x: ``SymbolicTensor`` of rank >= 1.
        k: int; number of entries to keep (``0 <= k <=`` axis extent).
        axis: int; axis along which to pick (default -1, last axis).
        largest: bool; pick the largest (True) or smallest (False) ``k``.

    Returns:
        ``(values, indices)`` — ``values`` has ``x``'s dtype and shape with
        the axis extent replaced by ``k``; ``indices`` has the same shape
        with int64 dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``k`` is not a Python int; ``axis``/``largest`` of the
            wrong kind.
        core.ShapeError: ``k < 0``; ``k`` exceeds a static axis extent; axis
            out of range.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"topk: k must be a Python int, got {k!r}")
    if k < 0:
        raise core.ShapeError(f"topk: k must be >= 0, got {k}")
    if not isinstance(largest, bool):
        raise TypeError(f"topk: largest must be a bool, got {largest!r}")
    rank = len(x.shape)
    if rank == 0:
        raise core.ShapeError("topk: input must have rank >= 1, got rank 0")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"topk: axis must be an int, got {axis!r}")
    axis_norm = _sort_axis(axis, rank, "topk")
    extent = x.shape[axis_norm]
    if isinstance(extent, int) and k > extent:
        raise core.ShapeError(
            f"topk: k={k} exceeds the axis extent {extent} along axis "
            f"{axis_norm}"
        )
    values = _emit_sort(builder, x, axis_norm, largest, False, "sort", loc)
    indices = _emit_sort(builder, x, axis_norm, largest, False, "argsort", loc)
    pick = np.arange(k, dtype=np.int64)
    op = builder.create("constant", attributes={"value": pick}, location=loc)
    pick_t = core.SymbolicTensor(
        value=op.result, dtype=np.dtype("int64"), shape=(k,), location=loc
    )
    return (
        _indexing.gather(values, pick_t, axis=axis_norm),
        _indexing.gather(indices, pick_t, axis=axis_norm),
    )
