"""Basic blocks: an ordered op list with SSA block arguments.

v1 (single-block regions): every region holds exactly one block and its last
op is the terminator (``return``). Multi-block regions are reserved for future
control-flow forms — ``Region`` and ``verify`` are designed to allow them.

Op order IS program order for effectful ops (see CONTEXT.md, "Effect model").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from .op import Op
    from .region import Region
    from .value import Value


@dataclass
class Block:
    """A basic block.

    Attributes:
        arguments: Block argument ``Value``s (function inputs in v1; region
            entry args for nested control-flow ops).
        ops: The ordered op list; the last op must be the terminator (v1).
        parent: The owning ``Region`` (wired on insertion), or None.
    """

    arguments: tuple[Value, ...] = ()
    ops: list[Op] = field(default_factory=list)
    parent: Optional[Region] = None

    @property
    def terminator(self) -> Optional[Op]:
        """The last op if it is a terminator, else None."""
        if not self.ops:
            return None
        last = self.ops[-1]
        return last if last.is_terminator else None

    def append(self, op: "Op") -> "Op":
        """Append ``op`` at the end and wire its parent pointer. Returns op."""
        op.parent = self
        self.ops.append(op)
        return op

    def insert(self, index: int, op: "Op") -> "Op":
        """Insert ``op`` at ``index`` and wire its parent pointer. Returns op."""
        op.parent = self
        self.ops.insert(index, op)
        return op

    def erase(self, op: "Op") -> None:
        """Remove ``op`` from this block and clear its parent pointer.

        The caller is responsible for keeping the IR valid (uses of the op's
        results must be handled); ``verify`` reports dangling references.

        Raises:
            ValueError: If ``op`` is not in this block.
        """
        try:
            self.ops.remove(op)
        except ValueError:
            raise ValueError(f"op '{op.name}' is not in this block") from None
        op.parent = None

    def __iter__(self) -> Iterator["Op"]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)
