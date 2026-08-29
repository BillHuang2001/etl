"""Structural / creation kernels: tile, flip, roll, diag, nan_to_num.

Semantics (each mirrors the IR inference hook / frontend contract — numpy
reference implementations):

- ``tile``: ``np.tile(arr, reps)`` (numpy's leading-dims promotion and
  rep-shortening rules exactly; the IR ``infer_tile`` declared the same
  shape rule). ``reps`` arrives as a tuple of non-negative ints.
- ``flip``: ``np.flip(arr, axes)`` with ``axes`` None (all axes), an int, or
  a tuple of ints (negative axes supported natively). Out-of-range axes —
  numpy ``AxisError`` (a ``ValueError`` subclass) — are re-raised as
  ``core.ShapeError``.
- ``roll``: ``np.roll(arr, shift, axis)``; the frontend normalized ``shift``
  to a per-axis tuple (a multi-entry tuple with ``axis=None`` folds to its
  sum — numpy's flattened-roll semantics) and ``axis`` to None/int/tuple.
- ``diag``: ``np.diag(arr)`` — rank-1 → diagonal matrix, rank-2 → main
  diagonal; dtype preserved in both directions (numpy keeps the input dtype).
- ``nan_to_num``: ``np.nan_to_num(arr, nan, posinf, neginf)`` with the
  scalar replacements read from the attrs; ``None`` (the attr default) keeps
  the numpy default behavior (dtype max/min finite). Only the corresponding
  infinity is replaced per argument — numpy semantics.

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
    """Reject object/str/void arrays — the interpreter never moves them."""
    if arr.dtype.kind not in _SUPPORTED_KINDS:
        raise core.BackendError(
            f"{op_name}: unsupported dtype {arr.dtype} — the numpy "
            "interpreter does not move object/str/void arrays"
        )


def _shape_error(op_name: str, exc: Exception) -> core.ShapeError:
    """Re-raise a numpy shape/index failure as a ``ShapeError`` naming the op."""
    return core.ShapeError(f"{op_name}: {exc}")


def _tile(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``tile``: numpy tile semantics (exact)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "tile")
    reps = op.attributes["reps"]
    try:
        result = np.tile(arr, tuple(reps))
    except (ValueError, IndexError) as exc:
        raise _shape_error("tile", exc) from exc
    return core.Tensor(result)


def _flip(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``flip``: numpy flip semantics; ``axes`` None = all axes."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "flip")
    axes = op.attributes.get("axes")
    try:
        result = np.flip(arr, axis=axes)
    except (ValueError, IndexError) as exc:
        raise _shape_error("flip", exc) from exc
    return core.Tensor(result)


def _roll(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``roll``: numpy roll semantics; ``shift`` is a per-axis tuple (the
    frontend folded multi-entry shifts with ``axis=None`` to their sum)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "roll")
    shift = tuple(op.attributes["shift"])
    axis = op.attributes.get("axis")
    if axis is None or isinstance(axis, int):
        # Frontend guarantees a single-entry shift for these cases.
        amount = shift[0]
    else:
        amount = shift
    try:
        result = np.roll(arr, amount, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("roll", exc) from exc
    return core.Tensor(result)


def _diag(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``diag``: numpy diag — rank-1 → (n, n) matrix; rank-2 → main
    diagonal. dtype preserved both directions."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "diag")
    if arr.ndim not in (1, 2):
        raise core.ShapeError(
            f"diag: input must be 1- or 2-d, got rank {arr.ndim}"
        )
    try:
        result = np.diag(arr)
    except (ValueError, IndexError) as exc:
        raise _shape_error("diag", exc) from exc
    return core.Tensor(result)


def _nan_to_num(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``nan_to_num``: numpy nan_to_num with the scalar replacements from
    the attrs (``None`` = the numpy default per infinity)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "nan_to_num")
    nan = op.attributes.get("nan", 0.0)
    posinf = op.attributes.get("posinf")
    neginf = op.attributes.get("neginf")
    try:
        result = np.nan_to_num(arr, nan=nan, posinf=posinf, neginf=neginf)
    except (ValueError, TypeError, IndexError) as exc:
        raise _shape_error("nan_to_num", exc) from exc
    return core.Tensor(result)


def register_kernels(table: dict) -> None:
    """Register this module's structural kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["tile"] = _tile
    table["flip"] = _flip
    table["roll"] = _roll
    table["diag"] = _diag
    table["nan_to_num"] = _nan_to_num
