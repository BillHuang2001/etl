"""Internal shared metadata types for graph transformations.

Not part of the public API — public names are re-exported only from
`etl/transforms/__init__.py`. Submodules of this package import these freely;
code outside `etl/transforms` must not.

See `./CONTEXT.md` ("Axis metadata (MappedAxes)") for the binding design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from etl import ir


@dataclass(frozen=True)
class MappedAxes:
    """Batching metadata for one SSA value in a transformed graph.

    `vectorize` semantics: a value "is mapped along" the axes listed here.
    Mapped axes are ALWAYS a contiguous tuple of **leading** indices:
    `(0,)` after one vectorization, `(0, 1)` after nesting two, `()` when the
    value is unmapped. A value of shape `(a, b, c)` with axes `(0,)` has batch
    extent `a` (repeated per row of the unvectorized computation) and per-row
    dims `b, c`.
    """

    axes: Tuple[int, ...] = ()

    @property
    def count(self) -> int:
        """The batching rank — number of leading mapped axes."""
        return len(self.axes)

    @property
    def is_mapped(self) -> bool:
        """True when the value carries at least one mapped axis."""
        return bool(self.axes)


#: Canonical metadata for unmapped values.
UNMAPPED: MappedAxes = MappedAxes()


class ValueEnv:
    """Per-value batching metadata for a graph being transformed.

    Maps transformed SSA values to their `MappedAxes` (keyed by value
    identity). The vectorize machinery owns one env per transformation and
    seeds it from the function inputs; batching rules receive operand
    metadata as arguments and never touch the env directly.
    """

    def __init__(self) -> None:
        self._env: Dict[int, MappedAxes] = {}

    def set(self, value: ir.Value, axes: MappedAxes) -> None:
        self._env[id(value)] = axes

    def get(self, value: ir.Value) -> MappedAxes:
        return self._env.get(id(value), UNMAPPED)

    def update(self, values: Tuple[ir.Value, ...], axes: Tuple[MappedAxes, ...]) -> None:
        for value, ax in zip(values, axes):
            self.set(value, ax)

    def __contains__(self, value: ir.Value) -> bool:
        return id(value) in self._env
