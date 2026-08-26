"""Block-call inlining support: splice a traced portable decomposition into a
target block.

``NumpyBackend.lower`` expands ``block_call`` ops whose block has a portable
(``etl.defn``) implementation but no registered numpy impl. The portable is
traced into its own single-function module; this module re-creates its entry
block's ops inside the TARGET module with FRESH ids, remapping the traced
entry-block arguments to the ``block_call``'s operands and the traced
``return`` terminator's operands to the ``block_call``'s results
(RAUW, then the ``block_call`` is erased).

Bookkeeping invariants kept here (``ir.verify`` enforces them):

- fresh op/value ids from the target ``module``'s counters (module-unique),
- ``Value.owner`` / ``index`` wired on every cloned result,
- ``Use`` records added for every cloned op operand, and the erased
  ``block_call``'s ``Use`` records removed from its operands,
- result types of the portable's ``return`` operands must be compatible with
  the ``block_call`` result types (dtype exactly; symbolic shape compatible)
  — otherwise ``BackendError`` (never silent semantic drift).

v1 restriction: portable decompositions must be FLAT (no nested regions).
"""
from __future__ import annotations

from typing import Any, List, Tuple

from etl import core
from etl import ir

from . import shapes

__all__ = ["clone_ops_into"]


def _dim_compatible(a: Any, b: Any) -> bool:
    """True if two declared dim entries are compatible (may co-occur).

    ``None`` matches anything (runtime-dynamic); ints must be equal; ``Dim``
    unifies by name (sizes must agree when both known); ``DimExpr``
    structurally; mixed symbolic/concrete forms are compatible when both
    sides evaluate with known sizes only and agree.
    """
    if a is None or b is None:
        return True
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, core.Dim) and isinstance(b, core.Dim):
        if a.name != b.name:
            return False
        if a.size is not None and b.size is not None and a.size != b.size:
            return False
        return True
    if isinstance(a, core.DimExpr) and isinstance(b, core.DimExpr):
        return a == b  # structural equality (op/left/right)
    # Mixed forms (Dim/DimExpr vs int): evaluate with known sizes only.
    try:
        return shapes.evaluate_dim_expr(a, {}) == shapes.evaluate_dim_expr(b, {})
    except (core.ShapeError, TypeError):
        return False


def _check_output_types(
    portable_type: ir.ValueType, declared_type: ir.ValueType
) -> None:
    """The portable's traced output types must agree with the declared ones."""
    if portable_type.dtype != declared_type.dtype:
        raise core.BackendError(
            f"portable decomposition produces dtype {portable_type.dtype}, "
            f"but the block_call declares {declared_type.dtype}"
        )
    if len(portable_type.shape) != len(declared_type.shape) or not all(
        _dim_compatible(p, d)
        for p, d in zip(portable_type.shape, declared_type.shape)
    ):
        raise core.BackendError(
            f"portable decomposition produces shape "
            f"{portable_type.shape!r}, but the block_call declares "
            f"{declared_type.shape!r}"
        )


def drop_op_uses(op: ir.Op) -> None:
    """Remove ``op``'s ``Use`` records from its operand values.

    Call before erasing an op: ``ir.verify`` checks that every recorded
    ``Use`` refers to an op present in the module, so erased ops must not
    leave records behind.
    """
    for i, operand in enumerate(op.operands):
        for use in list(operand.uses):
            if use.owner is op and use.operand_index == i:
                operand.remove_use(use)


def clone_ops_into(
    target_block: ir.Block,
    index: int,
    source_entry_block: ir.Block,
    operand_values: Tuple[ir.Value, ...],
    result_values: Tuple[ir.Value, ...],
    module: ir.Module,
) -> None:
    """Splice the traced portable entry block into ``target_block`` at ``index``.

    - ``source_entry_block`` arguments map positionally to ``operand_values``
      (the ``block_call`` operands — target-module values).
    - The traced ``return`` terminator's operands map positionally to
      ``result_values`` (the ``block_call`` results): after cloning,
      ``result_values[i].replace_all_uses_with(cloned value)`` rewires the
      downstream users; the ``return`` op itself is NOT inserted.
    - All re-created ops/results get FRESH ids from ``module``'s counters;
      attributes are shallow-copied (graph-constant payloads are shared by
      design — ``constant`` kernels return them without copying).
    - Nested regions in the portable => ``core.BackendError`` (v1 portables
      are flat).

    The caller erases the ``block_call`` op afterwards (this function
    rewires its users but leaves the op in the block — see
    ``NumpyBackend.lower``).
    """
    if len(source_entry_block.arguments) != len(operand_values):
        raise core.BackendError(
            f"portable decomposition has {len(source_entry_block.arguments)} "
            f"input(s), but the block_call has {len(operand_values)} operand(s)"
        )
    terminator = source_entry_block.terminator
    if terminator is None or terminator.name != "return":
        raise core.BackendError(
            "portable decomposition entry block has no 'return' terminator"
        )
    if len(terminator.operands) != len(result_values):
        raise core.BackendError(
            f"portable decomposition produces {len(terminator.operands)} "
            f"result(s), but the block_call declares {len(result_values)}"
        )

    value_map: dict[int, ir.Value] = {
        argument.id: target
        for argument, target in zip(source_entry_block.arguments, operand_values)
    }

    for op in source_entry_block.ops:
        if op.regions:
            raise core.BackendError(
                "portable decompositions must be flat in v1 (no nested "
                f"regions), but the decomposition contains op '{op.name}' "
                "with nested regions"
            )
        if op.name == "return":
            # The terminator is not inserted; its operands map to the
            # block_call results (validated + RAUW'd below).
            for source, target in zip(op.operands, result_values):
                _check_output_types(source.type, target.type)
            continue
        new_op = ir.Op(
            name=op.name,
            id=module.new_op_id(),
            operands=tuple(value_map[source.id] for source in op.operands),
            attributes=dict(op.attributes),
            location=op.location,
        )
        new_op.results = tuple(
            ir.Value(
                id=module.new_value_id(),
                type=source.type,
                owner=new_op,
                index=i,
            )
            for i, source in enumerate(op.results)
        )
        for i, source in enumerate(op.results):
            value_map[source.id] = new_op.results[i]
        for i, operand in enumerate(new_op.operands):
            operand.add_use(ir.Use(new_op, i))
        target_block.insert(index, new_op)
        index += 1

    # Rewire downstream users of the block_call results to the cloned values
    # that now compute them, then drop the block_call's own Use records so
    # verify's use bookkeeping stays consistent after the caller erases it.
    for source, target in zip(terminator.operands, result_values):
        target.replace_all_uses_with(value_map[source.id])
