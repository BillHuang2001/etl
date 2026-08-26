"""Regions: ordered lists of blocks owned by a function or an op.

Regions are the nesting mechanism of EvoXIR: a function body is a region;
`if`/`while` ops own nested regions whose entry-block arguments are bound to
the op's operands. v1 regions are single-block (see ``verify``); multi-block
regions are reserved for future control-flow forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .block import Block
    from .function import Function
    from .op import Op


@dataclass
class Region:
    """An ordered list of blocks.

    Attributes:
        blocks: The region's blocks; ``blocks[0]`` is the entry block.
        parent: The owning ``Function`` or ``Op``, or None.
    """

    blocks: list[Block] = field(default_factory=list)
    parent: Optional[Union["Function", "Op"]] = None

    @property
    def entry(self) -> "Block":
        """The entry block (first block).

        Raises:
            ValueError: If the region has no blocks.
        """
        if not self.blocks:
            raise ValueError("region has no blocks")
        return self.blocks[0]

    @property
    def single_block(self) -> bool:
        """True if the region holds exactly one block (always true in v1)."""
        return len(self.blocks) == 1

    def append_block(self, block: "Block") -> "Block":
        """Append ``block`` and wire its parent pointer. Returns block."""
        block.parent = self
        self.blocks.append(block)
        return block

    def insert_block(self, index: int, block: "Block") -> "Block":
        """Insert ``block`` at ``index`` and wire its parent. Returns block."""
        block.parent = self
        self.blocks.insert(index, block)
        return block
