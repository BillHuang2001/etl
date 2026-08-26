"""Operations: named IR instructions with operands, attributes, regions, results.

An ``Op`` is a typed, effect-annotated operation over SSA values. Its
*contract* (operand arity, attribute schema, effect kind, shape inference,
region structure) lives in its ``OpDef`` in the registry (``op_defs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .location import Location
from .value import Use, Value

if TYPE_CHECKING:
    from .block import Block
    from .op_defs import OpDef
    from .region import Region


@dataclass
class Op:
    """One instruction in a ``Block``.

    Attributes:
        name: Registered op name (see ``opdef``); unknown names fail in
            ``verify`` and in ``opdef`` lookups.
        id: Module-unique op id (assigned by the ``Builder``; stable for
            serialization).
        operands: Consumed ``Value``s.
        attributes: Static (JSON-able) parameters specializing the op; schema
            declared in the ``OpDef``.
        regions: Nested bodies for control-flow ops (`if` holds true/false,
            `while` holds cond/body); entry-block arguments are bound to the
            op's operands (see CONTEXT.md, "Nested regions").
        results: Produced ``Value``s — built by the ``Builder``, which assigns
            ids and infers (or takes explicit) result types.
        location: Optional source position; never affects semantics.
        parent: The owning ``Block`` (wired on insertion), or None.
    """

    name: str
    id: int
    operands: tuple[Value, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    regions: tuple[Region, ...] = ()
    results: tuple[Value, ...] = ()
    location: Optional[Location] = None
    parent: Optional[Block] = None

    @property
    def opdef(self) -> "OpDef":
        """The registered ``OpDef`` for this op's name."""
        from .op_defs import opdef

        return opdef(self.name)

    @property
    def effect(self) -> str:
        """The declared effect kind (one of ``EFFECT_KINDS``)."""
        return self.opdef.effect

    @property
    def is_terminator(self) -> bool:
        """True if this op terminates its block (``return`` in v1)."""
        return self.opdef.is_terminator

    @property
    def result(self) -> "Value":
        """The single result of this op.

        Raises:
            ValueError: If the op does not have exactly one result.
        """
        if len(self.results) != 1:
            raise ValueError(
                f"op '{self.name}' has {len(self.results)} results, expected exactly 1"
            )
        return self.results[0]

    def _replace_operand(self, index: int, new: "Value") -> None:
        """Internal RAUW support used by ``Value.replace_all_uses_with``.

        Swaps ``operands[index]`` to ``new`` and moves the corresponding
        ``Use`` record from the old value to the new one.
        """
        old = self.operands[index]
        operands = list(self.operands)
        operands[index] = new
        self.operands = tuple(operands)
        for use in old.uses:
            if use.owner is self and use.operand_index == index:
                old.uses.remove(use)
                break
        new.add_use(Use(self, index))
