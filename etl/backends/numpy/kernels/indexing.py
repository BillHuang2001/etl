"""Indexing / data-movement kernels: reshape, transpose, slice, gather,
scatter, concatenate, pad, tril, triu, cumsum.

All ops are ``pure`` data movement on ``core.Tensor`` (numpy arrays in v1) —
they produce NEW SSA results and never mutate operands (``scatter`` copies its
input before writing). Attribute conventions follow the IR op defs
(``etl/ir/op_defs/structure.py`` / ``linalg.py`` / ``reduction.py``) and the
frontend semantics in ``etl/ops/indexing.py`` (the single source of semantic
truth):

- ``reshape.shape`` may contain a single ``-1`` wildcard plus ``Dim``/
  ``DimExpr`` entries; symbolic entries are evaluated against the execution's
  dim bindings (``ctx.bindings`` via ``../shapes.py`` — shape-rule reuse,
  never a second copy). A ``None`` (runtime-dynamic) entry cannot be decided
  at run time => ``core.ShapeError`` (the interpreter never guesses a dim).
- ``slice`` attrs are the per-dim ``start_indices`` / ``limit_indices`` /
  ``strides`` tuples (strides ``None`` = all ones per the opdef default);
  they map 1:1 onto Python ``slice`` objects. Bounds/negativity were
  validated statically by the frontend wherever expressible; numpy clamps
  exactly like the ops-level contract documents.
- ``gather`` is numpy ``take`` along ONE axis (the ``etl.gather`` /
  ``etl.scan`` semantics: ``x.shape[:axis] + indices.shape + x.shape[axis+1:]``).
  A multi-axis ``axes`` tuple has no numpy equivalent for a single shared
  index tensor => ``core.BackendError`` (capability-drift safety net, never a
  silent fallback). 0-d index tensors (the ``etl.scan`` counter pattern) are
  handled natively by ``np.take``.
- ``scatter`` is numpy ``put_along_axis`` semantics on a COPY of the input
  (functional update). 0-d index tensors — the ``etl.scan`` stack-update
  pattern (``scatter(stack, counter, updates, axis=0)`` with a 0-d counter) —
  are normalized to a full-rank all-ones index array before dispatch.
- ``concatenate`` casts operands to the op-declared promoted result dtype
  (computed by the IR ``infer_concatenate`` via ``np.result_type``) — this is
  the DEFINED semantics, not kernel-side coercion.
- ``pad`` maps the IR ``padding_config`` (per-dim ``(lo, hi)`` pairs, plain
  ints = symmetric) onto ``np.pad(..., mode="constant")``.
- ``cumsum`` accumulates with ``dtype=operand.dtype``: the op contract
  preserves the operand dtype (the frontend pre-casts bool -> int64), while
  numpy >= 2.0 upcasts integer accumulation (int32 -> int64) — forcing the
  operand dtype executes the op exactly as defined.

Error behavior (binding): unsupported dtypes (object/str/void) =>
``core.BackendError`` naming the op; shape/index problems =>
``core.ShapeError`` (raw numpy ``ValueError``/``IndexError`` are re-raised as
``ShapeError`` with the op name — never swallowed). The interpreter validates
every result against the op's declared result types afterwards (dtype exactly,
symbolic dims via ``ctx.bindings``, ``None`` dims unchecked).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

from ..shapes import evaluate_dim_expr

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


def _check_index_dtype(arr: np.ndarray, op_name: str) -> None:
    """Gather/scatter indices must be integer tensors (ops-level contract)."""
    if arr.dtype.kind not in "iu":
        raise core.BackendError(
            f"{op_name}: indices must be an integer dtype (int32/int64), "
            f"got {arr.dtype}"
        )


def _shape_error(op_name: str, exc: Exception) -> core.ShapeError:
    """Re-raise a numpy shape/index failure as a ``ShapeError`` naming the op."""
    return core.ShapeError(f"{op_name}: {exc}")


def _normalize_axis(axis: Any, rank: int, op_name: str) -> int:
    """Normalize a (possibly negative) axis into ``range(rank)``."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise core.ShapeError(f"{op_name}: axis must be an int, got {axis!r}")
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        raise core.ShapeError(f"{op_name}: axis {axis} out of range for rank {rank}")
    return axis


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _reshape(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``reshape``: evaluate the target shape against ``ctx.bindings``
    (``-1`` wildcard passes through to numpy; ``None`` entries cannot be
    decided at run time => ``ShapeError``) then ``np.reshape``."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "reshape")
    target = []
    for dim in op.attributes["shape"]:
        if dim is None:
            raise core.ShapeError(
                "reshape: target shape contains a None (runtime-dynamic) dim "
                "— the interpreter never guesses a dimension"
            )
        if isinstance(dim, int) and not isinstance(dim, bool) and dim == -1:
            target.append(-1)  # single wildcard, inferred by numpy
            continue
        target.append(evaluate_dim_expr(dim, ctx.bindings))
    try:
        result = np.reshape(arr, tuple(target))
    except (ValueError, IndexError) as exc:
        raise _shape_error("reshape", exc) from exc
    return core.Tensor(result)


def _transpose(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``transpose``: numpy ``transpose`` semantics; ``permutation=None`` =
    full reversal (the opdef default)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "transpose")
    perm = op.attributes.get("permutation")
    try:
        result = np.transpose(arr, axes=perm)
    except (ValueError, IndexError) as exc:
        raise _shape_error("transpose", exc) from exc
    return core.Tensor(result)


def _slice(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``slice``: per-dim ``slice(start, limit, stride)`` tuples.

    Attribute convention (``ir/op_defs/structure.py`` + ``infer_slice``):
    ``start_indices`` defaults to zeros, ``limit_indices=None`` means the full
    axis, ``strides=None`` means all ones; entries are static ints. The
    frontend (``ops.slice``) always emits explicit non-negative
    start/limit ints; numpy's own clamping matches the ops-level contract for
    any boundary cases the static checks deferred to run time.
    """
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "slice")
    rank = arr.ndim
    starts = op.attributes.get("start_indices")
    limits = op.attributes.get("limit_indices")
    strides = op.attributes.get("strides")
    if starts is None:
        starts = (0,) * rank
    else:
        starts = tuple(starts)
    if limits is None:
        limits = (None,) * rank
    else:
        limits = tuple(limits)
    if strides is None:
        strides = (1,) * rank
    else:
        strides = tuple(strides)
    for name, seq in (
        ("start_indices", starts),
        ("limit_indices", limits),
        ("strides", strides),
    ):
        if len(seq) != rank:
            raise core.ShapeError(
                f"slice: {name} has {len(seq)} entries for rank {rank}"
            )
    slices = tuple(
        slice(start, limit, stride)
        for start, limit, stride in zip(starts, limits, strides)
    )
    # np.asarray normalizes a 0-d result (numpy returns a scalar for
    # ``arr[()]``) to the 0-d ndarray core.Tensor requires.
    return core.Tensor(np.asarray(arr[slices]))


def _gather(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``gather``: numpy ``take`` along the single gathered axis.

    Result shape is ``x.shape[:axis] + indices.shape + x.shape[axis+1:]`` (the
    IR ``infer_gather`` single-axis case — exactly ``np.take``). 0-d index
    tensors (the ``etl.scan`` step pattern: ``gather(xs, counter, axes=(0,))``
    with a 0-d int counter) are handled natively. A multi-axis ``axes`` tuple
    has no numpy equivalent for one shared index tensor => ``BackendError``
    (never a silent fallback).
    """
    x, indices_t = operands
    arr = x.numpy()
    indices = indices_t.numpy()
    _check_data_dtype(arr, "gather")
    _check_index_dtype(indices, "gather")
    axes = op.attributes.get("axes", (0,))
    if isinstance(axes, int):
        axes = (axes,)
    normalized = sorted({_normalize_axis(a, arr.ndim, "gather") for a in axes})
    if len(normalized) != 1:
        raise core.BackendError(
            "gather: multi-axis gather is not supported by the numpy "
            f"interpreter (axes {tuple(normalized)}) — a single shared index "
            "tensor has no numpy equivalent for several gathered axes"
        )
    axis = normalized[0]
    try:
        result = np.take(arr, indices, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("gather", exc) from exc
    # np.asarray normalizes a scalar result (np.take with a 0-d index — the
    # etl.scan step pattern) to the 0-d ndarray core.Tensor requires.
    return core.Tensor(np.asarray(result))


def _scatter(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``scatter``: functional numpy ``put_along_axis`` semantics.

    ``x`` is copied first — the op never mutates its operand. Index tensors
    with fewer dims than ``x`` are normalized to full rank (1-dim padding
    around the axis); 0-d indices — the ``etl.scan`` stack-update pattern
    (``scatter(stack, counter, updates, axis=0)`` with a 0-d counter) —
    reshape to all-ones, letting ``put_along_axis`` broadcast the kept dims
    against the updates exactly as the ops-level contract defines.
    """
    x, indices_t, updates_t = operands
    arr = x.numpy()
    indices = indices_t.numpy()
    updates = updates_t.numpy()
    _check_data_dtype(arr, "scatter")
    _check_index_dtype(indices, "scatter")
    rank = arr.ndim
    axis = _normalize_axis(op.attributes.get("axis", 0), rank, "scatter")
    out = np.array(arr, copy=True)  # functional update: x is never mutated
    if indices.ndim == 0:
        indices = indices.reshape((1,) * rank)
    elif indices.ndim < rank:
        indices = indices.reshape(
            (1,) * axis + indices.shape + (1,) * (rank - axis - 1)
        )
    try:
        np.put_along_axis(out, indices, updates, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("scatter", exc) from exc
    return core.Tensor(out)


def _concatenate(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``concatenate``: join along ``axis``; operands are cast to the
    op-declared promoted result dtype (``infer_concatenate`` computed it via
    ``np.result_type`` — executing the defined promotion, not inventing one)."""
    arrays = []
    for tensor in operands:
        arr = tensor.numpy()
        _check_data_dtype(arr, "concatenate")
        arrays.append(arr)
    axis = op.attributes["axis"]
    result_dtype = op.results[0].type.dtype
    arrays = [
        arr if arr.dtype == result_dtype else np.asarray(arr, dtype=result_dtype)
        for arr in arrays
    ]
    try:
        result = np.concatenate(arrays, axis=axis)
    except (ValueError, IndexError) as exc:
        raise _shape_error("concatenate", exc) from exc
    return core.Tensor(result)


def _pad(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``pad``: constant-value padding per the IR ``padding_config``.

    Each entry is a ``(lo, hi)`` pair (a plain int = symmetric pad, per the
    opdef/inference convention); maps onto ``np.pad(mode="constant")``. A
    rank-0 tensor has no axes to pad => returned unchanged (numpy's ``pad``
    rejects empty ``pad_width``).
    """
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "pad")
    rank = arr.ndim
    config = op.attributes["padding_config"]
    value = op.attributes.get("value")
    if value is None:
        value = 0
    if rank == 0:
        return core.Tensor(arr)  # no axes: identity (np.pad rejects 0-d)
    pad_width = []
    for entry in config:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            lo, hi = entry
        elif isinstance(entry, int) and not isinstance(entry, bool):
            lo = hi = entry  # symmetric pad, normalized to a pair
        else:
            raise core.BackendError(
                f"pad: invalid padding entry {entry!r} (expected an int or "
                "a (lo, hi) pair) — verify should have rejected this"
            )
        if lo < 0 or hi < 0:
            raise core.ShapeError(f"pad: negative padding ({lo}, {hi})")
        pad_width.append((lo, hi))
    try:
        result = np.pad(arr, pad_width, mode="constant", constant_values=value)
    except (ValueError, IndexError) as exc:
        raise _shape_error("pad", exc) from exc
    return core.Tensor(result)


def _tril(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``tril``: lower triangle of the last two dims (numpy ``tril``)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "tril")
    if arr.ndim < 2:
        raise core.ShapeError(
            f"tril: input must have rank >= 2, got rank {arr.ndim}"
        )
    k = op.attributes.get("k", 0)
    return core.Tensor(np.tril(arr, k=k))


def _triu(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``triu``: upper triangle of the last two dims (numpy ``triu``)."""
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "triu")
    if arr.ndim < 2:
        raise core.ShapeError(
            f"triu: input must have rank >= 2, got rank {arr.ndim}"
        )
    k = op.attributes.get("k", 0)
    return core.Tensor(np.triu(arr, k=k))


def _cumsum(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``cumsum``: cumulative sum along ``axis`` (optionally reversed).

    The op contract preserves the operand dtype (the frontend pre-casts bool
    inputs to int64); numpy >= 2.0 upcasts integer accumulation (int32 ->
    int64), so accumulate with ``dtype=arr.dtype`` to execute the op exactly
    as defined. ``reverse`` accumulates from the end toward the start (flip ->
    cumsum -> flip). A rank-0 tensor is its own cumsum (numpy's scalar
    ``cumsum`` would produce a spurious size-1 axis).
    """
    (x,) = operands
    arr = x.numpy()
    _check_data_dtype(arr, "cumsum")
    axis = op.attributes["axis"]
    reverse = bool(op.attributes.get("reverse", False))
    if arr.ndim == 0:
        return core.Tensor(np.array(arr, copy=True))
    if reverse:
        arr = np.flip(arr, axis=axis)
    try:
        result = np.cumsum(arr, axis=axis, dtype=arr.dtype)
    except (ValueError, IndexError) as exc:
        raise _shape_error("cumsum", exc) from exc
    if reverse:
        result = np.flip(result, axis=axis)
    return core.Tensor(result)


def register_kernels(table: dict) -> None:
    """Register this module's indexing kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["reshape"] = _reshape
    table["transpose"] = _transpose
    table["slice"] = _slice
    table["gather"] = _gather
    table["scatter"] = _scatter
    table["concatenate"] = _concatenate
    table["pad"] = _pad
    table["tril"] = _tril
    table["triu"] = _triu
    table["cumsum"] = _cumsum
