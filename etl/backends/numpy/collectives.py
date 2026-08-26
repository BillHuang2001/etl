"""Collective execution hook for the numpy interpreter.

Design (binding, parent CONTEXT.md): the interpreter dispatches ALL ``etl.dist``
collective ops through the pluggable ``CollectiveExecutor`` hook — it never
hard-codes collective semantics. The default executor is
``SingleRankCollectiveExecutor``: identity semantics on a single rank (v1 runs
single-process). Tests simulate multi-rank in-process by installing a custom
executor via ``set_collective_executor``; the installed executor is shared
module state (reset in test teardown).

Group semantics are simulation-only in v1: ``group`` / ``mapping`` parameters
are graph-constant objects (the ``dist.Group`` descriptor / permutation
mapping), not runtime tensors.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from etl.core import Tensor

__all__ = [
    "CollectiveExecutor",
    "SingleRankCollectiveExecutor",
    "set_collective_executor",
    "get_collective_executor",
]


@runtime_checkable
class CollectiveExecutor(Protocol):
    """Pluggable collective execution strategy.

    Implementations receive local tensors and the graph-constant collective
    descriptors; they return the post-collective LOCAL tensor. The numpy
    interpreter has no collective semantics of its own — everything funnels
    through this protocol.
    """

    def all_reduce(self, tensor: Tensor, group: Any, op: Any) -> Tensor:
        """Reduce ``tensor`` across the group with ``op`` (graph-constant)."""
        ...

    def all_gather(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        """Concatenate the group's tensors along ``axis`` (each rank keeps the full result)."""
        ...

    def reduce_scatter(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        """Reduce across the group, then scatter the result along ``axis`` (each rank keeps its slice)."""
        ...

    def all_to_all(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        """Exchange slices along ``axis`` between ranks."""
        ...

    def broadcast(self, tensor: Tensor, src_rank: int, group: Any) -> Tensor:
        """Send ``src_rank``'s tensor to every rank in the group."""
        ...

    def collective_permute(self, tensor: Tensor, mapping: Any, group: Any) -> Tensor:
        """Send tensors between rank pairs per the graph-constant ``mapping``."""
        ...


class SingleRankCollectiveExecutor:
    """DEFAULT executor: identity semantics on a single rank.

    With one rank there is nothing to reduce/gather/scatter/exchange: the
    input tensor already IS the post-collective result, so every collective
    returns the tensor unchanged.
    """

    def all_reduce(self, tensor: Tensor, group: Any, op: Any) -> Tensor:
        return tensor

    def all_gather(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        return tensor

    def reduce_scatter(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        return tensor

    def all_to_all(self, tensor: Tensor, axis: int, group: Any) -> Tensor:
        return tensor

    def broadcast(self, tensor: Tensor, src_rank: int, group: Any) -> Tensor:
        return tensor

    def collective_permute(self, tensor: Tensor, mapping: Any, group: Any) -> Tensor:
        return tensor


#: Currently installed executor. Default: single-rank identity simulation.
_executor: CollectiveExecutor = SingleRankCollectiveExecutor()


def set_collective_executor(executor: CollectiveExecutor) -> None:
    """Install the collective execution strategy (tests: multi-rank simulation).

    Raises ``TypeError`` for objects not structurally satisfying the
    ``CollectiveExecutor`` protocol (runtime-checkable).
    """
    if not isinstance(executor, CollectiveExecutor):
        raise TypeError(
            f"expected a CollectiveExecutor, got {type(executor).__name__}"
        )
    global _executor
    _executor = executor


def get_collective_executor() -> CollectiveExecutor:
    """Return the installed executor (the single-rank default if unset)."""
    return _executor
