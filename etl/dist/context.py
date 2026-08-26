"""Execution-context scalars (rank/world_size) and the collective-executor hook.

``rank()`` / ``world_size()`` build scalar int32 graph ops resolved from the
runtime execution context by backends — they are graph values, not Python
values, so they can depend on the executing rank.

``set_collective_executor`` installs the process-wide executor used to run
``collective``-effect ops. ``dist`` itself performs NO concrete computation
(no eager numerical duplication): the default single-process identity
executor (rank 0 / world size 1) is provided by the numpy reference backend
and installed at backend import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

from etl import core, trace

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


class CollectiveExecutor(Protocol):
    """Protocol for pluggable collective executors (see CONTEXT.md).

    Receives the op kind, the local concrete tensor, the static group, the
    op parameters and the runtime rank context; returns the local result
    tensor. Executors may coordinate across ranks in-process (multi-rank
    simulation) or delegate to a real transport.
    """

    def __call__(
        self,
        op_kind: str,
        tensor: "core.Tensor",
        group: Group,
        params: Mapping[str, object],
        rank_context: RankContext,
    ) -> "core.Tensor":
        ...


def rank() -> "core.SymbolicTensor":
    """Current rank of the runtime execution context, as a scalar int32 graph value.

    Graph-time only: builds an IR op into the active builder (``TraceError``
    outside a trace). Backends resolve the value from their execution
    context at run time (reference backend: 0) and must NOT constant-fold it
    at lower/compile time.

    Returns:
        SymbolicTensor, shape ``()``, dtype int32.

    Raises:
        TraceError: no active trace.

    IR: one ``dist.rank`` op; effect kind ``pure``; 0 operands, 1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


def world_size() -> "core.SymbolicTensor":
    """World size of the runtime execution context, scalar int32 graph value.

    Same contract as :func:`rank`; the reference backend resolves 1.

    Returns:
        SymbolicTensor, shape ``()``, dtype int32.

    Raises:
        TraceError: no active trace.

    IR: one ``dist.world_size`` op; effect kind ``pure``; 0 operands,
    1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


def set_collective_executor(executor: Optional[CollectiveExecutor]) -> None:
    """Install (or clear, with ``None``) the process-wide collective executor.

    The numpy reference backend installs its default single-process identity
    executor (rank 0 / world size 1 — every collective returns its input
    unchanged) at import time. Tests install custom executors to simulate
    multi-rank in-process semantics (see CONTEXT.md, test strategy).

    Args:
        executor: Callable matching the :class:`CollectiveExecutor`
            protocol, or ``None`` to reset to "unset".
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


def get_collective_executor() -> CollectiveExecutor:
    """Return the installed collective executor.

    Raises:
        BackendError: no executor installed — load a backend that provides
            one (e.g. the numpy backend) first.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)
