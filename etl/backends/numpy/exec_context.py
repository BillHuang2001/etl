"""Runtime rank context for the numpy interpreter.

The interpreter resolves ``etl.dist.rank()`` / ``etl.dist.world_size()`` graph
scalars from a per-execution ``RankContext`` (they are graph values, never
constant-folded at lower/compile time — see ``etl/dist/CONTEXT.md``).

This module provides a thread-local hook:

- ``set_rank_context(ctx)`` — install (or clear, with ``None``) the rank
  context for the current thread. ``NumpyExecutable.run(..., rank_context=...)``
  uses this to override the default for multi-rank in-process simulation.
- ``get_rank_context()`` — the thread-local value, or the default
  ``RankContext(rank=0, world_size=1)`` when none is set (single-process
  semantics).

``RankContext`` is the canonical ``etl.dist.context.RankContext`` (a frozen
dataclass validating ``0 <= rank < world_size``). Importing it at top level
is acyclic: ``etl.dist`` never imports ``etl.backends``.
"""
from __future__ import annotations

import threading
from typing import Optional

from etl.dist.context import RankContext

__all__ = ["set_rank_context", "get_rank_context"]

#: Default execution context: single-process, rank 0 of 1.
_DEFAULT_RANK_CONTEXT = RankContext(rank=0, world_size=1)

#: Thread-local rank-context slot (unset by default -> the default above).
_local = threading.local()


def set_rank_context(ctx: Optional[RankContext]) -> None:
    """Install (or clear, with ``None``) the rank context for this thread.

    Args:
        ctx: A ``RankContext``, or ``None`` to reset to the default.

    Raises:
        TypeError: ``ctx`` is not ``None`` and not a ``RankContext``.
    """
    if ctx is not None and not isinstance(ctx, RankContext):
        raise TypeError(
            f"expected a RankContext or None, got {type(ctx).__name__}"
        )
    _local.rank_context = ctx


def get_rank_context() -> RankContext:
    """The thread-local rank context, or the default rank 0 / world size 1."""
    ctx = getattr(_local, "rank_context", None)
    return ctx if ctx is not None else _DEFAULT_RANK_CONTEXT
