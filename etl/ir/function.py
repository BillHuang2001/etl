"""Functions: named, typed entry points holding one region.

In v1 every function's region contains exactly one block whose arguments are
the function inputs (enforced by ``verify``). The output signature is read off
the ``return`` terminator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .region import Region
from .types import ValueType

if TYPE_CHECKING:
    from .block import Block
    from .module import Module


@dataclass
class Function:
    """A named function: typed signature + one region.

    Attributes:
        name: Function name (unique within the ``Module``; "main" is the v1
            convention for traced graphs).
        input_types: Parameter types; the region's entry-block arguments must
            match exactly (verified).
        region: The function body region (single-block in v1).
        metadata: Free-form JSON-able annotations (e.g. static-value notes).
        parent: The owning ``Module``, or None.
    """

    name: str
    input_types: tuple[ValueType, ...]
    region: Region
    metadata: dict[str, Any] = field(default_factory=dict)
    parent: Optional["Module"] = None

    def __post_init__(self) -> None:
        self.region.parent = self

    @property
    def entry_block(self) -> "Block":
        """The function body's (single) block."""
        return self.region.entry

    @property
    def output_types(self) -> tuple[ValueType, ...]:
        """The output signature, read off the ``return`` terminator.

        Raises:
            ValueError: If the entry block has no terminator.
        """
        terminator = self.entry_block.terminator
        if terminator is None:
            raise ValueError(f"function '{self.name}' has no terminator")
        return tuple(value.type for value in terminator.operands)
