"""Collective execution for the numpy interpreter.

Coordination (binding): ``etl/dist/context.py`` is the CANONICAL home of the
pluggable collective-executor hook — ``dist.context.set_collective_executor``
/ ``get_collective_executor`` with protocol
``dist.context.CollectiveExecutor``. The numpy backend installs its default
executor into that slot at import time (see ``numpy/__init__.py``); the
interpreter's collective kernels (``kernels/collective.py``) dispatch ALL
``etl.dist`` collective ops through
``dist.context.get_collective_executor()`` — the interpreter has no
collective semantics of its own.

This module keeps only the default executor implementation:

- ``SingleRankCollectiveExecutor`` — identity semantics on a single rank
  (v1 runs single-process): with one rank there is nothing to
  reduce/gather/scatter/exchange, so every collective returns the tensor
  unchanged. Its six methods structurally satisfy
  ``dist.context.CollectiveExecutor``.
- ``CollectiveExecutor`` — re-exported alias of the canonical
  ``dist.context.CollectiveExecutor`` protocol.

Tests simulate multi-rank in-process by installing a custom executor via
``etl.dist.context.set_collective_executor`` (process-wide slot — reset in
test teardown).

Group semantics are simulation-only in v1: ``group`` / ``mapping`` parameters
are graph-constant objects (the ``dist.Group`` descriptor / permutation
mapping), not runtime tensors.
"""
from __future__ import annotations

from typing import Any

from etl.core import Tensor
from etl.dist.context import CollectiveExecutor

__all__ = ["CollectiveExecutor", "SingleRankCollectiveExecutor"]


class SingleRankCollectiveExecutor:
    """DEFAULT executor: identity semantics on a single rank.

    With one rank there is nothing to reduce/gather/scatter/exchange: the
    input tensor already IS the post-collective result, so every collective
    returns the tensor unchanged. Structurally satisfies
    ``dist.context.CollectiveExecutor`` (runtime-checkable).
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
