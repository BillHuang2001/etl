"""Shared block-call inlining machinery for backends.

Extracted from ``numpy/inline.py`` and ``numpy/__init__.py`` so compiler
backends (``CompilerBackend``, see ``compiler.py``) reuse the EXACT same
expansion logic as the numpy reference interpreter. Contains:

- ``iter_block_ops`` / ``iter_ops`` — regions-first, bottom-up op walks
  (previously module-private helpers in ``numpy/__init__.py``);
- ``clone_ops_into`` / ``drop_op_uses`` / ``_dim_compatible`` /
  ``_check_output_types`` — the portable-splicing bookkeeping (previously
  ``numpy/inline.py``); the invariants documented there remain binding:
  fresh op/value ids from the target module's counters, ``Value.owner`` /
  ``index`` wired on every cloned result, ``Use`` records added/removed
  correctly, result types of the portable's ``return`` operands compatible
  with the ``block_call`` result types (dtype exactly; symbolic shape
  compatible) — otherwise ``core.BackendError`` (never silent semantic
  drift). v1 restriction: portable decompositions must be FLAT (no nested
  regions).
- ``inline_portables`` — the shared fixpoint driver: inline every
  ``block_call`` whose block has a portable (``etl.defn``) decomposition
  (and, when ``keep_backend_impls`` is set, no impl registered for that
  backend).

Import acyclicity (binding, see ``../CONTEXT.md``): top-level imports
restricted to ``etl.core`` / ``etl.ir``. ``etl.block`` / ``etl.trace`` are
imported INSIDE function bodies, and ``etl.backends.numpy.shapes`` (the
dim-expression evaluator) is imported lazily inside ``_dim_compatible`` so
this module never triggers the numpy subpackage at import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Tuple

from etl import core
from etl import ir

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.ir import Module

__all__ = [
    "iter_block_ops",
    "iter_ops",
    "clone_ops_into",
    "drop_op_uses",
    "inline_portables",
]


def iter_block_ops(block: ir.Block) -> Iterator[ir.Op]:
    """Yield every op of a block, nested-region ops FIRST (bottom-up)."""
    for op in block.ops:
        for region in op.regions:
            for nested in region.blocks:
                yield from iter_block_ops(nested)
        yield op


def iter_ops(module: "Module") -> Iterator[ir.Op]:
    """Yield every op of every function of the module, regions-first.

    Bottom-up (nested regions before the op owning them) so inlining can
    walk and rewrite at any nesting depth.
    """
    for function in module.functions:
        for block in function.region.blocks:
            yield from iter_block_ops(block)


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
    # Lazy import (function-scoped): ``etl.backends.numpy.shapes`` is the
    # canonical dim-expression evaluator; importing it at module top level
    # would drag the whole numpy subpackage into every importer of this
    # shared module — no cycle, but unnecessary coupling.
    from etl.backends.numpy import shapes

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
    ``inline_portables``).
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


def inline_portables(module: "Module", keep_backend_impls: str | None = None) -> int:
    """Fixpoint: inline every block_call whose block has a portable
    decomposition (and, when keep_backend_impls is set, no impl registered
    for that backend). Returns the number of expansions performed.

    A portable trace may itself emit block calls, so inlining repeats until
    no expandable ``block_call`` remains (cap: 1000 expansions =>
    ``core.BackendError`` "did not converge ... (recursive portable?)").

    Per-block_call resolution (``etl.block.registry``, imported lazily):

    - unknown block => ``core.BackendError`` ("declare it first with
      etl.block(...)");
    - ``keep_backend_impls`` set AND an impl registered for that backend =>
      the op is KEPT (op id remembered so the fixpoint does not revisit it;
      e.g. the numpy interpreter dispatches it at run time);
    - else the portable (``etl.defn``) implementation is traced with the
      operand types as specs (the op's non-empty dict ``static_args`` attr
      re-specializes the trace as keyword arguments) and its entry block is
      spliced in place of the op (``clone_ops_into``);
    - neither impl nor portable => ``core.BackendError`` naming the block —
      with ``keep_backend_impls`` set: "... has neither a portable
      decomposition nor a registered <backend> impl ..."; with
      ``keep_backend_impls=None`` (compiler adapters): "... has no portable
      decomposition — compiler backends require BlockOp.portable(...)".
    """
    kept: set[int] = set()
    expansions = 0
    while True:
        target = next(
            (
                op
                for op in iter_ops(module)
                if op.name == "block_call" and op.id not in kept
            ),
            None,
        )
        if target is None:
            return expansions
        expansions += 1
        if expansions > 1000:
            raise core.BackendError(
                "block_call portable decomposition did not converge "
                "after 1000 expansions (recursive portable?)"
            )
        if not _expand_block_call(target, module, keep_backend_impls):
            kept.add(target.id)  # has a backend impl — keep, don't revisit


def _expand_block_call(
    op: ir.Op, module: "Module", keep_backend_impls: str | None
) -> bool:
    """Inline ONE block_call op via its portable decomposition.

    Returns True if the op was inlined; False if it was KEPT (a backend
    impl is registered and ``keep_backend_impls`` is set — the backend's
    block_call dispatch handles it at run time).
    """
    from etl.block import registry as block_registry
    from etl.block.errors import BlockError
    from etl.trace import trace

    block_name = op.attributes.get("block_name")
    try:
        block_registry.get_block(block_name)
    except BlockError as exc:
        raise core.BackendError(
            f"cannot lower block_call: unknown block {block_name!r} — "
            "declare it first with etl.block(...)"
        ) from exc
    if keep_backend_impls is not None:
        impl = block_registry.get_impl(block_name, keep_backend_impls)
        if impl is not None:
            return False  # keep the op — the block_call kernel dispatches it at run time
    portable = block_registry.get_portable(block_name)
    if portable is None:
        if keep_backend_impls is None:
            raise core.BackendError(
                f"block {block_name!r} has no portable decomposition — "
                "compiler backends require BlockOp.portable(...)"
            )
        raise core.BackendError(
            f"block {block_name!r} has neither a portable decomposition "
            f"nor a registered {keep_backend_impls} impl — register one via "
            f"BlockOp.portable(...) or BlockOp.impl({keep_backend_impls!r})"
        )
    specs = tuple(
        core.TensorSpec(shape=value.type.shape, dtype=value.type.dtype)
        for value in op.operands
    )
    static_args = op.attributes.get("static_args", ())
    if isinstance(static_args, dict) and static_args:
        # trace() binds positional specs only; re-specialize the portable
        # with the op's static kwargs through a thin wrapper (static
        # values specialize the traced graph — identical to passing them
        # at the block_call site).
        underlying = getattr(portable, "fn", portable)

        def bound(*args: Any) -> Any:
            return underlying(*args, **static_args)

        traced = trace(bound, *specs)
    elif static_args in ((), None, {}):
        traced = trace(portable, *specs)
    else:
        raise core.BackendError(
            f"block_call static_args must be a dict (or empty), got "
            f"{type(static_args).__name__}"
        )
    source_block = traced.module.main.entry_block
    target_block = op.parent
    clone_ops_into(
        target_block=target_block,
        index=target_block.ops.index(op),
        source_entry_block=source_block,
        operand_values=tuple(op.operands),
        result_values=tuple(op.results),
        module=module,
    )
    drop_op_uses(op)
    target_block.erase(op)
    return True
