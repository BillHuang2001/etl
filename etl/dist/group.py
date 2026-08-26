"""Communication groups for explicit collectives.

A :class:`Group` is a *static Python value* in etl's value model (root
CONTEXT.md, "Python/static value" row): it is resolved at trace time, it
specializes the graph, and it is recorded in IR op attributes by name +
ranks so graphs serialize without carrying Python objects.

Groups never hold tensors, never create sharding and never imply implicit
communication — they only name the set of ranks a collective operates over.
Explicit data-preparation helpers (``split_tensor``, ``replicate_tensor``)
are NOT here; they live in ``etl/core`` as concrete eager utilities.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

__all__ = ["Group", "group", "WORLD_GROUP"]


def _validate_ranks(ranks: Iterable[int]) -> Tuple[int, ...]:
    """Coerce and validate a ranks iterable into a normalized tuple.

    Raises:
        ValueError: empty collection, non-int or negative entry (bool is
            rejected — it is not a rank), or duplicate entries.
    """
    normalized = tuple(ranks)
    if len(normalized) == 0:
        raise ValueError("ranks must not be empty")
    for entry in normalized:
        if not isinstance(entry, int) or isinstance(entry, bool) or entry < 0:
            raise ValueError(f"ranks must be unique non-negative ints, got entry {entry!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"ranks must be unique, got {normalized!r}")
    return normalized


class Group:
    """A named set of ranks that explicit collectives operate over.

    Attributes:
        name: Unique human-readable name (recorded in op attributes).
        ranks: Tuple of participating ranks, or ``None`` for the world group
            (all ranks of the runtime execution context — membership is only
            known at run time).
        backend: Optional backend binding for the group (``None`` = resolved
            by the executing backend).

    Groups are immutable, hashable static values: treat them as frozen after
    construction. Equality covers ``(name, ranks, backend)``.
    """

    __slots__ = ("_name", "_ranks", "_backend")

    def __init__(self, name: str, ranks: Optional[Tuple[int, ...]], backend: Any = None):
        if not isinstance(name, str) or not name:
            raise ValueError(f"group name must be a non-empty str, got {name!r}")
        self._name = name
        self._ranks = None if ranks is None else _validate_ranks(ranks)
        self._backend = backend

    @property
    def name(self) -> str:
        """Group name (serialized into op attributes)."""
        return self._name

    @property
    def ranks(self) -> Optional[Tuple[int, ...]]:
        """Participating ranks, or ``None`` for the world group."""
        return self._ranks

    @property
    def backend(self) -> Any:
        """Optional backend binding (``None`` when unresolved)."""
        return self._backend

    @property
    def is_world(self) -> bool:
        """True if this is the world group (runtime-resolved membership)."""
        return self._ranks is None

    def size(self, world_size: Optional[int] = None) -> Optional[int]:
        """Number of ranks in the group.

        Explicit groups: always known. World group: requires the runtime
        ``world_size``; returns ``None`` when unresolved.

        Raises:
            ValueError: invalid ``world_size`` (not a positive int).
        """
        if self._ranks is not None:
            return len(self._ranks)
        if world_size is None:
            return None
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
            raise ValueError(f"world_size must be a positive int, got {world_size!r}")
        return world_size

    def __contains__(self, rank: int) -> bool:
        """Membership test. The world group contains every non-negative rank."""
        if self._ranks is not None:
            return rank in self._ranks
        return isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Group):
            return NotImplemented
        return (
            self._name == other._name
            and self._ranks == other._ranks
            and self._backend == other._backend
        )

    def __hash__(self) -> int:
        return hash((self._name, self._ranks, self._backend))

    def __repr__(self) -> str:
        return (
            f"Group(name={self._name!r}, ranks={self._ranks!r}, "
            f"backend={self._backend!r})"
        )


def group(name: str, ranks: Tuple[int, ...], backend: Any = None) -> Group:
    """Create an explicit communication :class:`Group`.

    Args:
        name: Non-empty unique name for the group.
        ranks: Tuple of participating ranks (non-empty, unique,
            non-negative ints).
        backend: Optional backend binding; ``None`` lets the executing
            backend resolve the transport.

    Returns:
        A new (static, immutable) Group.

    Raises:
        ValueError: empty name; empty, duplicated, negative or non-int
            rank entries.
    """
    return Group(name, ranks, backend)


#: The implicit world group: all ranks of the runtime execution context.
#: Used as the default ``group=None`` of every collective. Its exact
#: membership (world_size) is only known at run time.
WORLD_GROUP: Group = Group("world", None)
