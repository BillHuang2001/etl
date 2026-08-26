"""Collective kernels: dist collective ops dispatched through the executor hook.

Covers all eight dist collective ops:

- The six communication collectives (1 operand each) funnel through the
  pluggable ``CollectiveExecutor`` hook
  (``etl.dist.context.get_collective_executor``): all_reduce, all_gather,
  reduce_scatter, all_to_all, broadcast_collective, collective_permute.
- ``rank`` / ``world_size`` graph scalars (0 operands) resolve from the
  per-execution ``RankContext`` (``ctx.rank_context``) — they are graph
  values, NEVER constant-folded at lower/compile time.

Design notes (binding, parent CONTEXT.md):

- The interpreter dispatches ALL collective ops through the pluggable
  ``CollectiveExecutor`` hook — it has NO collective semantics of its own.
  Default: ``SingleRankCollectiveExecutor`` — identity semantics on one
  rank. Tests simulate multi-rank in-process by installing a custom
  executor via ``dist.context.set_collective_executor``. An unset hook
  raises ``core.BackendError`` (propagates as-is — no fallback).
- ``group`` / ``mapping`` / reduce-``op`` attributes are graph-constant
  objects carried by the ir.Op; the kernels pass the ``group`` NAME STRING
  attr through as-is — they never reconstruct ``dist.Group`` objects.
  Local-tensor shape semantics follow the frontend (e.g. 4-rank all_gather
  axis=0: ``[256,1024] -> [1024,1024]``); the interpreter validates each
  result against the op's declared result types afterwards.
- Known protocol limitations (documented, never silently worked around):
  * ``reduce_scatter`` — the ``CollectiveExecutor`` protocol has NO
    ``reduce_op`` parameter, so the op's ``reduce_op`` attr is NOT
    forwarded to the executor.
  * ``all_to_all`` — the protocol takes a SINGLE ``axis``; only the op's
    ``split_axis`` is forwarded, ``concat_axis`` is not (v1 protocol
    limitation).
  * ``broadcast_collective`` — ``dist`` does not record ``src_rank`` on
    the op (ir-side gap); ``None`` is passed through for the world group.
- Defensive normalization: a ``core.Tensor`` result passes through as-is;
  a raw numpy ndarray (tolerated for custom simulators) is wrapped via
  ``core.Tensor(arr)``.
- ``rank`` / ``world_size`` resolve from ``ctx.rank_context`` (the
  per-execution ``RankContext``), NOT from the executor hook — the
  executor is only involved in tensor communication.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]


def _via_executor(operands: tuple, method: str, **kwargs: Any) -> core.Tensor:
    """Dispatch one communication collective through the executor hook.

    Body-level import of ``etl.dist.context`` (the canonical slot; dist
    never imports backends, so it is acyclic — body-level is safest).

    ``get_collective_executor()`` raises ``core.BackendError`` when no
    backend installed an executor — propagated as-is (never a fallback).
    The result is normalized: ``core.Tensor`` passes through; a raw numpy
    ndarray (tolerated for custom simulators) is wrapped in
    ``core.Tensor``. Anything else is returned untouched so the
    interpreter's output validation surfaces the contract violation.
    """
    from etl.dist import context as dist_context

    executor = dist_context.get_collective_executor()
    result = getattr(executor, method)(operands[0], **kwargs)
    if isinstance(result, core.Tensor):
        return result
    if isinstance(result, np.ndarray):
        return core.Tensor(result)
    return result


def _all_reduce(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``all_reduce``: reduce across the group with ``op`` (shape-preserving)."""
    attrs = op.attributes
    return _via_executor(
        operands,
        "all_reduce",
        group=attrs["group"],
        op=attrs["reduce_op"],
    )


def _all_gather(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``all_gather``: concatenate the group's tensors along ``axis``."""
    attrs = op.attributes
    return _via_executor(
        operands,
        "all_gather",
        axis=attrs["axis"],
        group=attrs["group"],
    )


def _reduce_scatter(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``reduce_scatter``: reduce across the group, scatter along ``axis``.

    Known protocol limitation: the ``CollectiveExecutor`` protocol has NO
    ``reduce_op`` parameter, so the op's ``reduce_op`` attr is NOT
    forwarded — the executor decides the reduction itself.
    """
    attrs = op.attributes
    return _via_executor(
        operands,
        "reduce_scatter",
        axis=attrs["axis"],
        group=attrs["group"],
    )


def _all_to_all(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``all_to_all``: exchange slices along ``split_axis``.

    Known protocol limitation: the executor protocol takes a SINGLE axis,
    so only ``split_axis`` is forwarded; ``concat_axis`` is not (v1
    protocol limitation).
    """
    attrs = op.attributes
    return _via_executor(
        operands,
        "all_to_all",
        axis=attrs["split_axis"],
        group=attrs["group"],
    )


def _broadcast_collective(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``broadcast_collective``: send ``src_rank``'s tensor to the group.

    ``src_rank`` may be absent/``None`` for the world group (``dist`` does
    not record it on the op — ir-side gap); it is passed through as-is.
    """
    attrs = op.attributes
    return _via_executor(
        operands,
        "broadcast",
        src_rank=attrs.get("src_rank"),
        group=attrs["group"],
    )


def _collective_permute(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``collective_permute``: pairwise send/recv per ``source_target_pairs``."""
    attrs = op.attributes
    return _via_executor(
        operands,
        "collective_permute",
        mapping=attrs["source_target_pairs"],
        group=attrs["group"],
    )


def _rank(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``rank``: scalar 0-d int64 — this run's rank from the execution context.

    Resolved from ``ctx.rank_context`` at run time, never constant-folded
    (the op is a graph value that may differ per executing rank).
    """
    return core.Tensor(np.asarray(ctx.rank_context.rank, dtype=np.int64))


def _world_size(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``world_size``: scalar 0-d int64 — this run's world size.

    Resolved from ``ctx.rank_context`` at run time, never constant-folded.
    """
    return core.Tensor(np.asarray(ctx.rank_context.world_size, dtype=np.int64))


def register_kernels(table: dict) -> None:
    """Register this module's collective kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["all_reduce"] = _all_reduce
    table["all_gather"] = _all_gather
    table["reduce_scatter"] = _reduce_scatter
    table["all_to_all"] = _all_to_all
    table["broadcast_collective"] = _broadcast_collective
    table["collective_permute"] = _collective_permute
    table["rank"] = _rank
    table["world_size"] = _world_size
