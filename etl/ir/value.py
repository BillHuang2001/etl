"""SSA values.

A ``Value`` is a typed SSA reference inside a ``Function``:

* an *op result* — produced by exactly one op (``owner`` is the defining
  ``Op``, ``index`` is the result index), or
* a *block argument* — an input of a ``Block`` (``owner`` is the block,
  ``index`` is the argument index).

Values carry NO data: no storage, no device, no ``numpy()``, no DLPack. The
frontend ``SymbolicTensor`` (in ``etl.core``) wraps a ``Value``; the IR itself
is pure structure. SSA identity is ``id``, which is module-unique (assigned by
the ``Builder`` from the owning ``Module``'s counters so serialization is
stable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from .types import ValueType

if TYPE_CHECKING:  # deferred: op.py/block.py import value.py
    from .block import Block
    from .op import Op


@dataclass
class Use:
    """One use of a Value: ``owner.operands[operand_index] is value``.

    Attributes:
        owner: The op using the value.
        operand_index: Index into ``owner.operands`` of the used value.
    """

    owner: "Op"
    operand_index: int

    @property
    def value(self) -> "Value":
        """The value used by this use."""
        return self.owner.operands[self.operand_index]


@dataclass
class Value:
    """An SSA value: a typed reference to a future runtime tensor in a graph.

    Attributes:
        id: Module-unique SSA identity (assigned by the ``Builder``).
        type: ``ValueType`` (dtype + symbolic shape).
        owner: The defining ``Op`` (op result) or the parent ``Block``
            (block argument).
        index: Result index (op result) or argument index (block argument).
        uses: All ``Use``s of this value; kept in sync by the ``Builder`` and
            by ``replace_all_uses_with``.
    """

    id: int
    type: ValueType
    owner: Union["Op", "Block"]
    index: int
    uses: list[Use] = field(default_factory=list)

    @property
    def is_block_arg(self) -> bool:
        """True if this value is a block argument (it has no defining op)."""
        from .block import Block  # deferred import: op.py/block.py import value.py

        return isinstance(self.owner, Block)

    @property
    def is_op_result(self) -> bool:
        """True if this value is produced by an op."""
        return not self.is_block_arg

    @property
    def defining_op(self) -> Union["Op", None]:
        """The op producing this value, or None for block arguments."""
        if self.is_block_arg:
            return None
        return self.owner  # type: ignore[return-value]

    def add_use(self, use: Use) -> None:
        """Record one use of this value (idempotent)."""
        if use not in self.uses:
            self.uses.append(use)

    def remove_use(self, use: Use) -> None:
        """Forget one use of this value.

        Raises:
            ValueError: If ``use`` is not recorded on this value.
        """
        try:
            self.uses.remove(use)
        except ValueError:
            raise ValueError(f"use {use} is not recorded on value %{self.id}") from None

    def replace_all_uses_with(self, new: "Value") -> None:
        """Rewrite every use of this value to ``new`` (classic SSA RAUW).

        Mutates each using op's operand list and updates ``new.uses``. This
        value's use list is drained — it ends up unused.
        """
        for use in list(self.uses):
            use.owner._replace_operand(use.operand_index, new)
        self.uses.clear()
