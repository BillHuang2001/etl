"""Sorting kernels: sort, argsort (numpy reference implementations).

Semantics (each mirrors the IR inference hook / frontend contract):

- ``sort``: ``np.sort`` along the axis with ``kind="stable"`` when the
  ``stable`` attribute is set, else the default quicksort. ``descending`` is
  executed as a flip of the ascending result along the axis (composition
  semantics, documented in ``etl/ops/sorting.py``). The operand dtype is
  preserved exactly (never cast). ``np.sort`` raises ``AxisError`` (a
  ``ValueError`` subclass) for out-of-range axes / rank-0 operands —
  re-raised as ``core.ShapeError`` naming the op.
- ``argsort``: ``np.argsort`` with the same attribute handling; the numpy
  ``intp`` index dtype is cast to the op-declared ``int64`` when the platform
  disagrees (explicit dtype match, never a silent promotion).

Error behavior (binding): unsupported dtypes (object/str/void) =>
``core.BackendError`` naming the op; shape/index problems =>
``core.ShapeError``. The interpreter validates every result against the op's
declared result types afterwards.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]

#: numpy dtype kinds these kernels move: everything except object/str/void.
_SUPPORTED_KINDS = frozenset("biufc")


def _check_data_dtype(arr: np.ndarray, op_name: str) -> None:
    """Reject object/str/void arrays — the interpreter never sorts them."""
    if arr.dtype.kind not in _SUPPORTED_KINDS:
        raise core.BackendError(
            f"{op_name}: unsupported dtype {arr.dtype} — the numpy "
            "interpreter does not sort object/str/void arrays"
        )


def _shape_error(op_name: str, exc: Exception) -> core.ShapeError:
    """Re-raise a numpy shape/index failure as a ``ShapeError`` naming the op."""
    return core.ShapeError(f"{op_name}: {exc}")


def _sort_kind(stable: bool) -> str | None:
    """numpy sort kind: ``"stable"`` when requested, else the default."""
    return "stable" if stable else None


def _sort(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``sort``: numpy sort along ``axis``; ``descending`` = flip of the
    ascending result along the axis."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "sort")
    axis = op.attributes["axis"]
    try:
        result = np.sort(arr, axis=axis, kind=_sort_kind(op.attributes["stable"]))
        if op.attributes["descending"]:
            result = np.flip(result, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("sort", exc) from exc
    return core.Tensor(result)


def _argsort(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``argsort``: numpy argsort along ``axis`` (int64 result); descending
    = flip of the ascending indices along the axis."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "argsort")
    axis = op.attributes["axis"]
    try:
        result = np.argsort(
            arr, axis=axis, kind=_sort_kind(op.attributes["stable"])
        )
        if op.attributes["descending"]:
            result = np.flip(result, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("argsort", exc) from exc
    if result.dtype != np.dtype("int64"):
        result = np.asarray(result, dtype=np.int64)
    return core.Tensor(result)


def register_kernels(table: dict) -> None:
    """Register this module's sorting kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["sort"] = _sort
    table["argsort"] = _argsort
