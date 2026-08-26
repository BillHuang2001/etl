"""Collective kernels: dist collective ops dispatched through the hook.

Covers these ops (implement in the implementation phase):
all_reduce, all_gather, reduce_scatter, all_to_all, broadcast,
collective_permute.

Design notes (binding, parent CONTEXT.md):
- The interpreter dispatches ALL collective ops through the pluggable
  ``CollectiveExecutor`` hook (``../collectives.py``:
  ``get_collective_executor()``) — it has NO collective semantics of its own.
- Default: ``SingleRankCollectiveExecutor`` — identity semantics on one rank.
  Tests simulate multi-rank in-process by installing a custom executor via
  ``set_collective_executor``.
- ``group`` / ``mapping`` / reduce-``op`` attributes are graph-constant
  objects carried by the ir.Op; local-tensor shape semantics follow the
  frontend (e.g. 4-rank all_gather axis=0: ``[256,1024] -> [1024,1024]``).
- ``dist.rank()`` / ``dist.world_size()`` graph scalars resolve via the
  executor hook as well (v1: simulation values).
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's collective kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.

    Not yet implemented (follow-up agent): registers nothing so
    ``register_all()`` assembles; the category fills in per-op kernels here.
    """
    return None
