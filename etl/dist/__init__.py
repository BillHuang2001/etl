"""etl.dist — explicit distributed collectives (SPMD, local-tensor model).

Public surface (full semantics in this directory's CONTEXT.md):

- ``Group`` / ``group()`` — static named groups of ranks (``WORLD_GROUP``
  is the default ``group=None`` of every collective).
- ``all_reduce``, ``all_gather``, ``reduce_scatter``, ``all_to_all``,
  ``broadcast``, ``collective_permute`` — graph-time collectives building
  ``collective``-effect IR ops with local-tensor shapes.
- ``rank()`` / ``world_size()`` — scalar int32 graph values resolved from
  the runtime execution context by backends.
- ``set_collective_executor`` / ``get_collective_executor`` — pluggable
  executor hook; the default single-process identity executor is provided
  by the numpy reference backend.
"""

from etl.dist.group import Group, group, WORLD_GROUP
from etl.dist.collectives import (
    all_reduce,
    all_gather,
    reduce_scatter,
    all_to_all,
    broadcast,
    collective_permute,
)
from etl.dist.context import (
    RankContext,
    rank,
    world_size,
    set_collective_executor,
    get_collective_executor,
)

__all__ = [
    "Group",
    "group",
    "WORLD_GROUP",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "collective_permute",
    "RankContext",
    "rank",
    "world_size",
    "set_collective_executor",
    "get_collective_executor",
]
