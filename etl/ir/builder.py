"""The Builder — EvoXIR's op-construction API.

A Builder emits ops into a ``Module`` via an insertion-point stack. The
frontend ``ops`` module obtains the active builder from
``trace.current_builder()`` and calls ``emit``/``create``; ALL IR mutation
funnels through this class so invariants (id assignment, parent wiring,
operand use bookkeeping, result-type inference) are enforced in exactly one
place.

ARCHITECTURE PHASE: all methods are ``NotImplementedError`` stubs. Docstrings
state the exact expected semantics for the Phase 2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .block import Block
from .function import Function
from .location import Location
from .module import Module
from .op import Op
from .region import Region
from .types import ValueType
from .value import Value


@dataclass
class InsertionPoint:
    """Internal: where the builder currently appends ops.

    Attributes:
        block: The target block.
        position: Index into ``block.ops`` where new ops are inserted.
    """

    block: Block
    position: int


class Builder:
    """Constructs IR inside a ``Module``.

    Contract for every emission method: resolve the op's ``OpDef`` (unknown
    name raises immediately); check arity, attribute schema, and region count
    (cheap failures raise ``VerificationError`` early — full validation is
    still ``verify``'s job); assign module-unique ids from the owning module's
    counters; infer result types via ``OpDef.shape_fn`` unless explicit
    ``result_types`` are given (``shape_fn=None`` means op-specific resolution
    or mandatory explicit types — see ``inference.py``); create result
    ``Value``s, wire operand ``Use``s, set parent pointers.
    """

    def __init__(self, module: Optional[Module] = None) -> None:
        """Create a builder; ``module`` may be None until ``build_module``."""
        self.module = module
        self._insertion_stack: list[InsertionPoint] = []

    # --- insertion-point management ------------------------------------------

    @property
    def current_block(self) -> Block:
        """The block new ops are inserted into.

        Raises:
            RuntimeError: If no insertion point has been set.
        """
        raise NotImplementedError("Builder.current_block: Phase 2 (implementation)")

    @property
    def current_region(self) -> Region:
        """The region containing the current insertion point."""
        raise NotImplementedError("Builder.current_region: Phase 2 (implementation)")

    def set_insertion_point(self, target: Block | Region) -> None:
        """Set the insertion point: Block -> its start; Region -> its entry
        block's start. Replaces the current insertion point."""
        raise NotImplementedError(
            "Builder.set_insertion_point: Phase 2 (implementation)"
        )

    def push_region(self, region: Region) -> None:
        """Push ``region``'s entry block onto the insertion-point stack
        (for building nested if/while bodies); ``pop_region`` restores."""
        raise NotImplementedError("Builder.push_region: Phase 2 (implementation)")

    def pop_region(self) -> Region:
        """Pop the insertion-point stack, returning the region left behind."""
        raise NotImplementedError("Builder.pop_region: Phase 2 (implementation)")

    # --- module/function/region construction -----------------------------------

    def build_module(self, name: str = "main", metadata: dict[str, Any] | None = None) -> Module:
        """Create a new ``Module``, attach it to this builder, and return it."""
        raise NotImplementedError("Builder.build_module: Phase 2 (implementation)")

    def build_function(
        self,
        name: str,
        input_types: tuple[ValueType, ...],
        metadata: dict[str, Any] | None = None,
    ) -> Function:
        """Create a function with a single-block region whose block arguments
        are fresh ``Value``s of ``input_types``; set the insertion point to its
        block. The caller must emit a terminator."""
        raise NotImplementedError("Builder.build_function: Phase 2 (implementation)")

    def build_region(self, input_types: tuple[ValueType, ...] = ()) -> Region:
        """Create a detached single-block region whose entry-block arguments
        are fresh ``Value``s of ``input_types`` — the body of an upcoming
        `if`/`while` op (pass it via ``create(..., regions=(...))``)."""
        raise NotImplementedError("Builder.build_region: Phase 2 (implementation)")

    def insert_block(self, region: Region, position: int | None = None) -> Block:
        """Create a new empty block in ``region`` at ``position`` (None =
        append) and return it. Does not change the insertion point."""
        raise NotImplementedError("Builder.insert_block: Phase 2 (implementation)")

    # --- op emission -----------------------------------------------------------

    def create(
        self,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        result_types: tuple[ValueType, ...] | None = None,
        location: Location | None = None,
        regions: tuple[Region, ...] = (),
    ) -> Op:
        """Create an op at the current insertion point and return it.

        Result types: ``result_types`` if given, else ``OpDef.shape_fn``; if
        neither can resolve them (``shape_fn=None``, op-specific resolution
        unavailable) raise ``VerificationError`` demanding explicit types.
        Result ``Value``s get fresh ids from the module counters and their
        ``owner``/``index`` set to this op. Operand ``Use``s and region parent
        pointers are wired. ``attributes`` is validated against the op's
        attribute schema (required keys present, types tagged correctly).
        """
        raise NotImplementedError("Builder.create: Phase 2 (implementation)")

    def emit(
        self,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        result_type: ValueType | None = None,
        location: Location | None = None,
        regions: tuple[Region, ...] = (),
    ) -> Value:
        """Single-result convenience: ``create(...)`` then ``op.result``.

        Raises:
            ValueError: If the op does not have exactly one result.
        """
        raise NotImplementedError("Builder.emit: Phase 2 (implementation)")

    def set_terminator(
        self,
        block: Block,
        op_name: str,
        operands: tuple[Value, ...] = (),
        attributes: dict[str, Any] | None = None,
        location: Location | None = None,
    ) -> Op:
        """Append a terminator op (``return``) to ``block``.

        Raises:
            VerificationError: If ``op_name`` is not a terminator, ``block``
                already has one, or the terminator is not last.
        """
        raise NotImplementedError("Builder.set_terminator: Phase 2 (implementation)")
