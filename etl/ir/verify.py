"""Module verification.

``verify(module)`` is the IR contract checker: it validates structural
invariants, type agreement, attribute schemas, SSA well-formedness, and v1
restrictions, raising ``VerificationError`` (owned by ``etl.core``,
re-exported by this package) with a source-location-annotated message on the
FIRST violation (no silent recovery, no partial reporting in v1).

ARCHITECTURE PHASE: the body is a ``NotImplementedError`` stub. The docstring
lists the complete invariant set so the Phase 2 implementation has no
guesswork.
"""

from __future__ import annotations

from etl.core import VerificationError  # owned by core; re-exported by etl.ir

from .module import Module

__all__ = ["verify", "VerificationError"]


def verify(module: Module) -> None:
    """Validate ``module`` against all EvoXIR invariants.

    Raises:
        VerificationError: On the first violation, with source location when
            one exists.

    Checked invariants (binding):

    *Module level*
    - ``module.version`` == ``IR_FORMAT_VERSION``.
    - At least one function; function names unique; each function's ``parent``
      is the module.

    *Function level*
    - The function region holds exactly one block (v1 restriction;
      multi-block regions are reserved for future versions).
    - Entry-block argument count/types match ``function.input_types`` exactly.
    - The entry block ends with a terminator, which is the ``return`` op and
      is the LAST op (nothing after a terminator).
    - ``function.output_types`` (return operands) is consistent.

    *Region/block level (function regions AND nested op regions)*
    - Every region has at least one block; each block's ``parent`` is its
      region; each op's ``parent`` is its block.
    - Nested regions of an op: count matches the op's ``OpDef.regions``;
      entry-block argument types match the op's operand types (v1 binding
      convention).

    *Op level*
    - ``op.name`` is registered (unknown name fails).
    - Operand count within the ``OpDef`` arity; region count matches; result
      count within the declared ``result_count`` (when not None).
    - Attributes match the schema: no unknown keys, all required keys present,
      each value's type matches its ``AttrSpec`` tag.
    - Results are ``Value``s owned by this op, with module-unique ids.
    - Result types agree with ``OpDef.shape_fn(input_types, attributes)`` when
      ``shape_fn`` is not None (ops with ``shape_fn=None`` must record
      consistent types from op-specific resolution).

    *Value/SSA level*
    - Value ids unique across the module.
    - Operands are defined before use: block arguments of an enclosing block,
      or results of ops earlier in the same or an enclosing block
      (SSA dominance). No use of a value from an unrelated region.
    - Use bookkeeping is consistent: every ``Use`` recorded on a value
      actually refers to that value at that operand index, and every operand
      of every op has a matching recorded ``Use``.

    *Effect ordering*
    - Verification does NOT reorder anything; it only checks the structural
      invariants above. Effectful-op ordering is positional by construction
      (see CONTEXT.md, "Effect model").
    """
    raise NotImplementedError("verify: Phase 2 (implementation)")
