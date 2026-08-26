"""Execution-context scalars (rank/world_size) and the collective-executor hook.

``rank()`` / ``world_size()`` build scalar int64 graph ops (effect ``read``,
attr ``group="world"``) resolved from the runtime execution context by
backends — they are graph values, not Python values, so they can depend on
the executing rank. Backends MUST NOT constant-fold them at lower/compile
time.

``set_collective_executor`` installs the process-wide executor used to run
``collective``-effect ops. ``dist`` itself performs NO concrete computation
(no eager numerical duplication): the default single-process identity
executor (rank 0 / world size 1) is provided by the numpy reference backend
and installed into this slot at backend import time (its implementation
phase). Until then the slot starts UNSET and ``get_collective_executor``
raises ``BackendError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from etl import core
from etl.trace import current_builder

from etl.dist._op_utils import _get_location, _wrap_result
from etl.dist.group import Group

__all__ = [
    "RankContext",
    "CollectiveExecutor",
    "rank",
    "world_size",
    "set_collective_executor",
    "get_collective_executor",
]


@dataclass(frozen=True)
class RankContext:
    """Runtime rank context passed to collective executors.

    Attributes:
        rank: Current rank index in ``[0, world_size)``.
        world_size: Total number of ranks (>= 1).
    """

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise ValueError(f"rank must be a non-negative int, got {self.rank!r}")
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool) or self.world_size < 1:
            raise ValueError(f"world_size must be a positive int, got {self.world_size!r}")
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank {self.rank} out of range for world_size {self.world_size}"
            )


@runtime_checkable
class CollectiveExecutor(Protocol):
    """Pluggable collective executor: one method per collective op.

    The protocol mirrors exactly what the numpy reference backend's
    ``SingleRankCollectiveExecutor`` (``etl/backends/numpy/collectives.py``)
    implements, so it is structurally accepted by dist's slot. ``group`` /
    ``mapping`` / ``op`` parameters are graph-constant objects (the
    ``dist.Group`` descriptor / permutation mapping / reduction kind), not
    runtime tensors. Implementations receive the local concrete tensor and
    return the post-collective LOCAL tensor; they may coordinate across
    ranks in-process (multi-rank simulation in tests) or delegate to a real
    transport.

    :class:`RankContext` (the runtime rank context) is exported for
    multi-rank simulations but is NOT part of the v1 protocol parameters.
    """

    def all_reduce(self, tensor: "core.Tensor", group: Group, op: Any) -> "core.Tensor":
        """Reduce ``tensor`` across the group with ``op`` (graph-constant)."""
        ...

    def all_gather(self, tensor: "core.Tensor", axis: int, group: Group) -> "core.Tensor":
        """Concatenate the group's tensors along ``axis`` (each rank keeps the full result)."""
        ...

    def reduce_scatter(self, tensor: "core.Tensor", axis: int, group: Group) -> "core.Tensor":
        """Reduce across the group, then scatter the result along ``axis`` (each rank keeps its slice)."""
        ...

    def all_to_all(self, tensor: "core.Tensor", axis: int, group: Group) -> "core.Tensor":
        """Exchange slices along ``axis`` between ranks."""
        ...

    def broadcast(self, tensor: "core.Tensor", src_rank: int, group: Group) -> "core.Tensor":
        """Send ``src_rank``'s tensor to every rank in the group."""
        ...

    def collective_permute(self, tensor: "core.Tensor", mapping: Any, group: Group) -> "core.Tensor":
        """Send tensors between rank pairs per the graph-constant ``mapping``."""
        ...


def _build_scalar(op_name: str) -> "core.SymbolicTensor":
    """Build the 0-operand execution-context scalar op ``op_name``.

    Shared by :func:`rank` and :func:`world_size`. The op's registered
    ``shape_fn`` (``infer_scalar_int64``) yields a scalar ``()`` int64
    result; dist passes attrs ``{"group": "world"}`` (the public API takes
    no group argument).

    Raises:
        TraceError: no active trace (``current_builder()`` raises it).
    """
    location = _get_location()
    builder = current_builder()
    op = builder.create(
        op_name,
        operands=(),
        attributes={"group": "world"},
        location=location,
    )
    return _wrap_result(op.results[0], location)


def rank() -> "core.SymbolicTensor":
    """Current rank of the runtime execution context, as a scalar int64 graph value.

    Graph-time only: builds an IR op into the active builder (``TraceError``
    outside a trace). Backends resolve the value from their execution
    context at run time (reference backend: 0) and must NOT constant-fold it
    at lower/compile time.

    Returns:
        SymbolicTensor, shape ``()``, dtype int64.

    Raises:
        TraceError: no active trace.

    IR: one ``rank`` op; effect kind ``read``; attr ``group="world"``;
    0 operands, 1 result.
    """
    return _build_scalar("rank")


def world_size() -> "core.SymbolicTensor":
    """World size of the runtime execution context, scalar int64 graph value.

    Same contract as :func:`rank`; the reference backend resolves 1.

    Returns:
        SymbolicTensor, shape ``()``, dtype int64.

    Raises:
        TraceError: no active trace.

    IR: one ``world_size`` op; effect kind ``read``; attr ``group="world"``;
    0 operands, 1 result.
    """
    return _build_scalar("world_size")


#: Installed collective executor, or ``None`` when unset. Starts UNSET:
#: ``import etl`` must not install anything — the numpy backend installs its
#: default single-process identity executor into this slot in its
#: implementation phase (see ``etl/backends/numpy``).
_executor: Optional[CollectiveExecutor] = None


def set_collective_executor(executor: Optional[CollectiveExecutor]) -> None:
    """Install (or clear, with ``None``) the process-wide collective executor.

    The numpy reference backend will install its default single-process
    identity executor (rank 0 / world size 1 — every collective returns its
    input unchanged) at import time. Tests install custom executors to
    simulate multi-rank in-process semantics (see CONTEXT.md, test
    strategy).

    Args:
        executor: Object structurally satisfying the
            :class:`CollectiveExecutor` protocol (runtime-checkable — it
            must implement the six per-op methods), or ``None`` to reset
            to "unset".

    Raises:
        TypeError: ``executor`` is not ``None`` and does not structurally
            satisfy the protocol.
    """
    global _executor
    if executor is None:
        _executor = None
        return
    if not isinstance(executor, CollectiveExecutor):
        raise TypeError(
            "expected a CollectiveExecutor implementing the six per-op "
            "methods (all_reduce, all_gather, reduce_scatter, all_to_all, "
            "broadcast, collective_permute), got "
            f"{type(executor).__name__}"
        )
    _executor = executor


def get_collective_executor() -> CollectiveExecutor:
    """Return the installed collective executor.

    Raises:
        BackendError: no executor installed — load a backend that provides
            one (e.g. the numpy backend) first.
    """
    if _executor is None:
        raise core.BackendError(
            "no collective executor registered; load a backend that "
            "provides one (e.g. numpy_backend)"
        )
    return _executor
