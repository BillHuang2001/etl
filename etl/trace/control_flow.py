"""Runtime tensor control flow: `cond`, `while_loop`, `scan`.

These trace Python callables into IR regions of ordinary `if`/`while` ops —
backends (the numpy interpreter) need NO special control-flow runtime
support. Region ops are built through `ir.opdef(...)` + the `ir.Builder`
directly; this module NEVER imports `etl.ops` (ops imports trace for the
active-builder hook — the DAG stays acyclic).

REGION CONVENTIONS (the implemented `etl.ir` reality — binding; see
`./etl/ir/op_defs/control.py`, `./etl/ir/builder.py`, `./etl/ir/verify.py`):

* `if` op  (registry name "if"; effects: none; `regions` = 2)
  - arity (1, None): operand 0 IS the boolean predicate; the remaining
    operands are SSA values captured from the enclosing function.
  - The v1 binding convention requires EVERY region's entry block to have
    ONE argument per op operand with IDENTICAL count and types — for `if`
    ALL operands, including the predicate, bind. `regions[0]` = "then",
    `regions[1]` = "else"; each has exactly one block.
  - Branch callables receive the REGION BLOCK ARGS (wrapped as
    `core.SymbolicTensor`) at the captured-operand positions — never the
    enclosing values directly. Static operands/kwargs are passed unchanged.
  - Each region block's terminator is a `return` op yielding the branch's
    outputs (n `core.SymbolicTensor`s). Both branches must yield the same
    pytree and unify to the same result dtype/shape; the `if` op is created
    with EXPLICIT result types (tree/arity mismatch → `core.TraceError`,
    dtype mismatch → `core.DTypeError`, shape mismatch → `core.ShapeError`
    — never a silent fallback).
  - op results: n values = the selected branch's outputs.
* `while` op (registry name "while"; effects: none; `regions` = 2)
  - arity (1, None): operands = n initial loop-carried SSA values (at least
    one); `shape_fn` = infer_identity → results mirror the operand types
    (no explicit result_types needed).
  - `regions[0]` = condition, `regions[1]` = body. Each region has one block
    with n block args bound to the operands (the loop-carried values).
  - condition region terminator: `return` of ONE 0-d bool value — anything
    else → `core.TraceError`.
  - body region terminator: `return` of n next-iteration carried values.
  - op results: n final carried values.
* `return` op: terminator-only (no results, no regions); operands = yielded
  values. Emitted via `builder._return_terminator` (`Builder.set_terminator`)
  so it is APPENDED as the block's last op (a terminator is never emitted
  via `create` at the insertion point).

Region building (the `ir.Builder` workflow): ONE Builder instance serves the
whole trace. `builder.build_region(input_types)` creates a DETACHED
single-block region whose entry args are fresh `Value`s;
`builder.create("if"/"while", operands=..., regions=(region,), ...)` wires
the region parents. To run a callable inside a region, use `_run_in_region`
(defined below): it pushes the region onto the builder's insertion-point
stack, installs the builder as the active-builder context for
`current_builder()` (the SAME builder), runs the callable, lets the caller's
`after` callback emit the region's `return` via `_return_terminator`, and
pops the region in a `finally` so the stack stays consistent on error.

Static values: `*operands` / `init` / kwargs may contain static Python values
(per the static-value predicate in `./_tree.py`); they specialize the regions
(Python semantics — evaluated once at trace time, NOT per iteration) and are
passed to the callables unchanged. Static values are never loop-carried or
op results.

Local pytree note: `core.flatten` treats etl-module dataclasses
(`SymbolicTensor`, `Tensor`, `Device`, ir structures, ...) as LEAVES (the
module check in `core.tree._flatten_into`); only user-defined dataclasses
act as pytree containers. This module walks pytrees via the SHARED walker in
`./_tree.py` with the `_cf_leaf_spec` policy (`_LEAF_TYPES` → plain leaves;
everything else → walker default), so the leaf conventions stay explicit —
`SymbolicTensor`, `core.Tensor`, `core` value objects, and `ir` structural
dataclasses are LEAVES — while mirroring `core.TreeSpec`'s container
conventions (tuple/list/dict — keys sorted — namedtuple/dataclass), so
`core.unflatten` can rebuild the trees. Leaf nodes use `TreeSpec(type=None)`,
which `core.unflatten` treats as a plain leaf. `_flatten` consults the
core pytree registry FIRST (an MRO walk over `core.tree._PYTREE_NODE_REGISTRY`,
same as `core.flatten` and the tracer), so types registered via
`core.register_pytree_node` (e.g. a sparse tensor) flatten through their
registered `flatten_fn` and their children recurse normally; rebuild via
`core.unflatten` then uses the registered `unflatten_fn` (the recorded
`TreeSpec.type` is the registered base type). Registered nodes flow through
`cond`/`while_loop` generically — NEVER via an import of the registering
module: `cond` captures a registered-node operand's tensor leaves and passes
its static leaves through, `while_loop` carries registered nodes via the
ordinary static-leaf machinery, and both accept static leaves INSIDE
registered nodes in their outputs/returns (sparse tensors work end-to-end on
the numpy backend).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from etl import core
from etl import ir

from ._tree import (
    _flatten as _flatten_shared,
    _is_static_value,
    _registered_pytree_base,
    _to_symbolic,
)
from .builder import _return_terminator, current_builder, with_builder

__all__ = ["cond", "scan", "while_loop"]

_BOOL = np.dtype("bool")

#: Objects that must be treated as LEAVES by the local pytree walk (etl
#: value types — ir SSA structures and core value objects — are never
#: containers; the explicit list keeps the leaf convention independent of
#: the etl-module check alone).
_LEAF_TYPES = (
    core.SymbolicTensor,
    core.Tensor,
    core.TensorSpec,
    core.Dim,
    core.DimExpr,
    core.Device,
    ir.Value,
    ir.Use,
    ir.Op,
    ir.Block,
    ir.Region,
    ir.Function,
    ir.Module,
)


def _cf_leaf_spec(obj: Any) -> "Optional[Tuple[Any, core.TreeSpec]]":
    """The control-flow leaf policy for the shared walker (`./_tree.py`).

    `_LEAF_TYPES` instances (SymbolicTensor and every other non-container etl
    value) become plain leaves (`TreeSpec(type=None)`); everything else
    defers to the walker's default leaf (``None`` policy result → container
    descent or plain leaf). The returned pair is ``(leaf_to_record,
    TreeSpec)`` — the walker appends the object itself.
    """
    if isinstance(obj, _LEAF_TYPES):
        return (obj, core.TreeSpec(type=None))
    return None


def _flatten(obj: Any) -> tuple:
    """`(leaves, treespec)` — the local leaf variant of `core.flatten`
    (shared walker + `_cf_leaf_spec`; see the module docstring)."""
    return _flatten_shared(obj, _cf_leaf_spec)


def _is_registered_node(obj: Any) -> bool:
    """True iff `obj`'s type (or an MRO base) is registered via
    `core.register_pytree_node` — the same registry walk `_flatten` uses, so
    such operands flatten through their registered flatten_fn (e.g. a
    symbolic sparse tensor)."""
    return _registered_pytree_base(type(obj)) is not None


def _check_pred(pred: Any, where: str) -> None:
    """Validate a control-flow predicate: a 0-d bool `SymbolicTensor`."""
    if not isinstance(pred, core.SymbolicTensor):
        raise core.TraceError(
            f"{where}: pred must be a core.SymbolicTensor (0-d bool), got "
            f"{type(pred).__name__} — runtime tensor control flow is only "
            "for tensor predicates; use plain Python if/while for static "
            "values"
        )
    if pred.dtype != _BOOL:
        raise core.TraceError(
            f"{where}: pred dtype must be bool, got {pred.dtype.name}"
        )
    if pred.shape != ():
        raise core.TraceError(
            f"{where}: pred must be 0-d (scalar), got shape {pred.shape!r}"
        )


def _leaf_registered_flags(tree: "core.TreeSpec") -> List[bool]:
    """Per-leaf boolean (in `tree`'s leaf order): is the leaf nested inside a
    registered pytree node (e.g. the static `dense_shape`/`dtype`/`format`
    leaves of a sparse tensor)?

    A node spec whose `type` is found in the core pytree registry marks ALL
    its descendant leaves as inside; leaves of plain containers
    (`tuple`/`list`/`dict`/namedtuple/dataclass nodes) and top-level leaves
    are NOT inside.
    """
    flags: List[bool] = []

    def walk(spec: "core.TreeSpec", inside: bool) -> None:
        if not spec.children:
            if spec.num_leaves:  # a leaf (childless containers contribute none)
                flags.append(inside)
            return
        node_inside = inside or _registered_pytree_base(spec.type) is not None
        for child in spec.children:
            walk(child, node_inside)

    walk(tree, False)
    return flags


def _rebuild_carried(
    leaves: List[Any], tree: "core.TreeSpec", entry_args: tuple
) -> Any:
    """Reconstruct the loop-carried structure per `tree` with the region's
    entry block args (wrapped as `SymbolicTensor`) at the symbolic positions
    and the original static leaves unchanged."""
    arg_iter = iter(entry_args)
    rebuilt = []
    for leaf in leaves:
        if isinstance(leaf, core.SymbolicTensor):
            rebuilt.append(_to_symbolic(next(arg_iter)))
        else:
            rebuilt.append(leaf)
    return core.unflatten(rebuilt, tree)


def _run_in_region(
    builder: "ir.Builder",
    region: "ir.Region",
    fn: Any,
    *args: Any,
    after: Any = None,
    **kwargs: Any,
) -> Any:
    """Run `fn(*args, **kwargs)` inside `region` (single-Builder workflow).

    Pushes the region's entry block onto the builder's insertion-point stack
    (creating the block first if the region is empty) and installs the
    builder as the active builder for `current_builder()` (the SAME builder —
    control flow never creates a second builder). After `fn` returns,
    `after(result)` runs while still inside the region — callbacks emit the
    region's `return` terminator, which must be appended at the region's
    insertion point. The region is popped in a `finally`, so the
    insertion-point stack stays consistent on error.

    Returns `after(result)` when `after` is given, else `fn`'s result.
    """
    if not region.blocks:
        builder.insert_block(region)
    builder.push_region(region)
    try:
        with with_builder(builder):
            result = fn(*args, **kwargs)
        if after is not None:
            return after(result)
        return result
    finally:
        builder.pop_region()


def _classify_operands(operands: tuple) -> List[tuple]:
    """Classify every `cond` operand ONCE per call (flatten memoization).

    Returns one plan per operand, consumed by both the operand validation
    and `_run_branch`:

    - ``("symbolic", st)`` — a captured `SymbolicTensor`;
    - ``("static", value)`` — a static Python value (specialization);
    - ``("container", leaves, tree)`` — a pytree container (registered node
      or plain tuple/namedtuple/list/dict/dataclass): its tensor leaves are
      captured as `if`-op operands, its static leaves pass through.

    Raises the same `core.TraceError`s the inline checks used to (a container
    operand with no leaves — e.g. ``()`` — is an error, matching the old
    behavior). Before this helper, a container operand was flattened ~5
    times per `cond` call (validation + operand list + once per branch);
    now it is flattened exactly once.
    """
    plans: List[tuple] = []
    for i, operand in enumerate(operands):
        if isinstance(operand, core.SymbolicTensor):
            plans.append(("symbolic", operand))
        elif _is_static_value(operand):
            plans.append(("static", operand))
        else:
            leaves, tree = _flatten(operand)
            if not tree.children:
                raise core.TraceError(
                    f"etl.cond: operand {i} must be a core.SymbolicTensor or "
                    f"a static Python value, got {type(operand).__name__}"
                )
            for j, leaf in enumerate(leaves):
                if not isinstance(leaf, core.SymbolicTensor) and not _is_static_value(
                    leaf
                ):
                    raise core.TraceError(
                        f"etl.cond: operand {i} (a pytree node) "
                        f"leaf {j} must be a core.SymbolicTensor or a static "
                        f"Python value, got {type(leaf).__name__}"
                    )
            plans.append(("container", leaves, tree))
    return plans


def _run_branch(
    builder: "ir.Builder",
    region: "ir.Region",
    branch_name: str,
    fn: Any,
    operand_plans: List[tuple],
    static_kwargs: dict,
) -> tuple:
    """Run one `if` branch callable inside `region` and emit its `return`.

    Positions the builder inside the region via `_run_in_region`, runs
    `fn(*call_args, **static_kwargs)` exactly ONCE under `with_builder`,
    validates the outputs, emits the region's `return` terminator with the
    symbolic leaf values, and restores the insertion point.

    The region's entry args bind to ALL if operands (predicate at index 0 —
    the v1 verify convention); the branch callable receives the
    captured-operand entry args (wrapped as SymbolicTensor) at the symbolic
    positions, with static operands passed unchanged. A pytree CONTAINER
    operand plan (a registered node — e.g. a symbolic sparse tensor — or a
    plain tuple/namedtuple/dict/dataclass) rebuilds per its recorded tree:
    its tensor leaves bind to entry args, its static leaves pass through
    unchanged. (Plans come from `_classify_operands` — each operand is
    flattened exactly once per `cond` call.)

    Output leaves must be `SymbolicTensor`s; a static leaf is allowed only
    when it sits INSIDE a registered pytree node (e.g. a sparse tensor's
    `dense_shape`/`dtype`/`format` leaves) — bare static output leaves
    (top-level or inside plain containers) raise `core.TraceError`.

    Returns `(output_leaves, output_tree, inside_flags)` where
    `inside_flags` = `_leaf_registered_flags(output_tree)` — computed here
    (once, per branch) and reused by `cond` for the both-static unification
    check.
    """
    entry_args = region.entry.arguments
    arg_iter = iter(entry_args[1:])  # skip the predicate binding at index 0
    call_args = []
    for plan in operand_plans:
        kind = plan[0]
        if kind == "symbolic":
            call_args.append(_to_symbolic(next(arg_iter)))
        elif kind == "container":
            _, leaves, tree = plan
            rebuilt = []
            for leaf in leaves:
                if isinstance(leaf, core.SymbolicTensor):
                    rebuilt.append(_to_symbolic(next(arg_iter)))
                else:
                    rebuilt.append(leaf)  # static leaf passed through unchanged
            call_args.append(core.unflatten(rebuilt, tree))
        else:  # "static": passed unchanged
            call_args.append(plan[1])

    def finish(result: Any) -> tuple:
        leaves, tree = _flatten(result)
        inside_flags = _leaf_registered_flags(tree)
        for i, (leaf, inside) in enumerate(zip(leaves, inside_flags)):
            if isinstance(leaf, core.SymbolicTensor):
                continue
            if inside and _is_static_value(leaf):
                continue  # static leaf inside a registered node (e.g. sparse)
            raise core.TraceError(
                f"etl.cond: {branch_name} branch output leaf {i} must be "
                f"a SymbolicTensor, got {type(leaf).__name__} — cond "
                "branches yield tensors only (static output leaves are "
                "not supported)"
            )
        _return_terminator(
            builder,
            tuple(
                leaf.value for leaf in leaves if isinstance(leaf, core.SymbolicTensor)
            ),
        )
        return leaves, tree, inside_flags

    return _run_in_region(builder, region, fn, *call_args, after=finish, **static_kwargs)


def cond(pred: "core.SymbolicTensor", true_fn: Any, false_fn: Any, *operands: Any, **static_kwargs: Any) -> Any:
    """Runtime `if` over a tensor predicate → traced `if` op.

    1. `pred` must be a `core.SymbolicTensor` of 0-d bool dtype — else
       `core.TraceError` (non-scalar or non-bool or concrete value).
    2. `static_kwargs` must be static Python values (else `core.TraceError`);
       they specialize the regions and are passed to both branches as kwargs.
       `*operands` may be `SymbolicTensor`s (captured SSA values), static
       values (specialization), or pytree CONTAINERS — REGISTERED pytree
       nodes (e.g. a symbolic sparse tensor) AND plain containers
       (tuple/namedtuple/list/dict/plain user dataclass): the container is
       flattened via `_flatten`, its tensor leaves are captured as
       if-operands, its static leaves are passed through unchanged; every
       leaf must be a `SymbolicTensor` or a static value (else
       `core.TraceError`). Operands are passed positionally to both
       branches. (Classification happens ONCE per call via
       `_classify_operands` — each container operand is flattened exactly
       once, not once per use.)
    3. Build the `if` op via `ir.opdef("if")` with two regions per the
       conventions above. Run `true_fn(*operands, **static_kwargs)` inside
       the then-region under `with_builder(...)`, `false_fn` likewise in the
       else-region. Each is called exactly ONCE at trace time. The branches
       receive the region's entry block args (the captured tensor operands
       bound as block args) — not the enclosing values directly.
    4. Flatten each branch's return value (pytree); trees must be identical
       (else `core.TraceError`). Per leaf position: both `SymbolicTensor` →
       dtype/shape unification (dtype mismatch → `core.DTypeError`, shape
       mismatch → `core.ShapeError`); both static leaves INSIDE a registered
       pytree node → must be equal across branches (else `core.TraceError`);
       one tensor + one static, or any other leaf kind → `core.TraceError`
       (bare static output leaves are not supported). Emit each region's
       `return` terminator with its TENSOR leaves; the `if` op's result
       types are the unified tensor-leaf types only.
    5. Create the `if` op with the explicit unified result types and return
       its results unflattened per the branch output tree (tensor positions
       ← if-op results, validated-equal static positions ← the leaf; single
       tensor → returned bare).

    The numpy interpreter backend executes regions by selecting the branch —
    no graph-level Python callbacks.
    """
    _check_pred(pred, "etl.cond")
    for name, value in static_kwargs.items():
        if not _is_static_value(value):
            raise core.TraceError(
                f"etl.cond: static kwarg {name!r} is not a static Python "
                f"value (got {type(value).__name__}); static kwargs "
                "specialize the branches at trace time"
            )
    operand_plans = _classify_operands(operands)

    builder = current_builder()
    if_values = [pred.value]
    for plan in operand_plans:
        if plan[0] == "symbolic":
            if_values.append(plan[1].value)
        elif plan[0] == "container":
            _, leaves, _ = plan
            if_values.extend(
                leaf.value
                for leaf in leaves
                if isinstance(leaf, core.SymbolicTensor)
            )
    if_values = tuple(if_values)
    input_types = tuple(value.type for value in if_values)
    then_region = builder.build_region(input_types)
    else_region = builder.build_region(input_types)

    then_leaves, then_tree, then_flags = _run_branch(
        builder, then_region, "true", true_fn, operand_plans, static_kwargs
    )
    else_leaves, else_tree, _ = _run_branch(
        builder, else_region, "false", false_fn, operand_plans, static_kwargs
    )

    if then_tree != else_tree:
        raise core.TraceError(
            "etl.cond: branch output trees must be structurally identical — "
            f"true: {then_tree!r}, false: {else_tree!r}"
        )
    if len(then_leaves) != len(else_leaves):  # defensive: equal trees ⇒ equal counts
        raise core.TraceError(
            f"etl.cond: branch arity mismatch — true yields "
            f"{len(then_leaves)} output(s), false yields {len(else_leaves)}"
        )
    inside_flags = then_flags
    result_types = []
    for i, (true_leaf, false_leaf) in enumerate(zip(then_leaves, else_leaves)):
        if isinstance(true_leaf, core.SymbolicTensor) and isinstance(
            false_leaf, core.SymbolicTensor
        ):
            if true_leaf.dtype != false_leaf.dtype:
                raise core.DTypeError(
                    f"etl.cond: branch output {i} dtype mismatch — true: "
                    f"{true_leaf.dtype.name}, false: {false_leaf.dtype.name}"
                )
            if true_leaf.shape != false_leaf.shape:
                raise core.ShapeError(
                    f"etl.cond: branch output {i} shape mismatch — true: "
                    f"{true_leaf.shape!r}, false: {false_leaf.shape!r}"
                )
            result_types.append(ir.ValueType(true_leaf.dtype, true_leaf.shape))
        elif isinstance(true_leaf, core.SymbolicTensor) or isinstance(
            false_leaf, core.SymbolicTensor
        ):
            raise core.TraceError(
                f"etl.cond: branch output leaf {i} kinds differ — true: "
                f"{type(true_leaf).__name__}, false: "
                f"{type(false_leaf).__name__} (a leaf must be a tensor on "
                "both branches or a static value on both)"
            )
        else:
            # Both static: allowed only INSIDE a registered pytree node
            # (validated per branch in `_run_branch`), and must be equal
            # across branches — mirroring while_loop's static-leaf semantics.
            if (
                not inside_flags[i]
                or not _is_static_value(true_leaf)
                or not _is_static_value(false_leaf)
                or type(true_leaf) is not type(false_leaf)
                or not (true_leaf == false_leaf or true_leaf is false_leaf)
            ):
                raise core.TraceError(
                    f"etl.cond: branch output leaf {i} static values must "
                    f"match across branches — true: {true_leaf!r}, false: "
                    f"{false_leaf!r} (static leaves are only supported "
                    "inside registered pytree nodes and must be equal on "
                    "both branches)"
                )

    op = builder.create(
        "if",
        operands=if_values,
        regions=(then_region, else_region),
        result_types=tuple(result_types),
    )
    result_iter = iter(op.results)
    rebuilt = []
    for leaf in then_leaves:
        if isinstance(leaf, core.SymbolicTensor):
            rebuilt.append(_to_symbolic(next(result_iter)))
        else:
            rebuilt.append(leaf)  # validated-equal static leaf re-inserted
    return core.unflatten(rebuilt, then_tree)


def _shapes_compatible(body_shape: tuple, init_shape: tuple) -> bool:
    """Positional loop-carried shape compatibility.

    Same rank required. Per position: equal entries pass; a runtime-dynamic
    `None` on ONE side passes when the other side is a symbolic
    `core.Dim`/`core.DimExpr` — e.g. the nnz dim of a sparse tensor, where
    the traced input's `Dim("_dynamic_...")` wrapper (trace.py) meets the
    sparse-op result's true `None` (ir.ValueType). The loop-carried (init)
    type wins: the while-op result types mirror the operand/IR types, which
    already carry the true `None`, and the interpreter treats `None` result
    dims as unchecked. Any other mismatch (int-vs-Dim, different ints,
    Dim-vs-Dim, rank) is incompatible — the caller raises `core.ShapeError`.
    """
    if len(body_shape) != len(init_shape):
        return False
    for body_dim, init_dim in zip(body_shape, init_shape):
        if body_dim == init_dim:
            continue
        if body_dim is None and isinstance(init_dim, (core.Dim, core.DimExpr)):
            continue
        if init_dim is None and isinstance(body_dim, (core.Dim, core.DimExpr)):
            continue
        return False
    return True


def while_loop(cond_fn: Any, body_fn: Any, init: Any) -> Any:
    """Runtime `while` over a tensor condition → traced `while` op.

    1. `init` = `SymbolicTensor` or pytree of (`SymbolicTensor` | static
       value). Static leaves specialize the regions and are NOT loop-carried.
       No symbolic leaves → `core.TraceError` (a loop over static values
       only is Python control flow).
    2. Build the `while` op via `ir.opdef("while")`: operands = the flat
       carried `SymbolicTensor`s; two regions whose blocks carry them as
       block args (conventions above).
    3. Run `cond_fn(carried)` once inside the condition region (carried
       reconstructed per init's tree, static leaves as-is). Its return must
       be a scalar 0-d bool `SymbolicTensor` — else `core.TraceError`.
       Emit the condition region's `return` with it.
    4. Run `body_fn(carried)` once inside the body region. Its return tree
       must equal init's tree exactly (tensor leaves → next carried values,
       types must stay constant across iterations — dtype mismatch →
       `core.DTypeError`, shape mismatch → `core.ShapeError`; static leaves
       must match the init's static values — else `core.TraceError`). Emit
       the body region's `return` with the flat next-carried values.
    5. Return the `while` op's results (final carried values) unflattened
       per init's tree.

    v1 note: the condition/body callables run ONCE at trace time (via
    `_run_in_region`, which keeps the insertion-point stack consistent); the
    traced regions repeat their IR at run time — no Python callbacks.
    """
    leaves, tree = _flatten(init)
    carried = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, core.SymbolicTensor):
            carried.append(leaf)
        elif _is_static_value(leaf):
            continue
        else:
            raise core.TraceError(
                f"etl.while_loop: init leaf {i} must be a core.SymbolicTensor "
                f"or a static Python value, got {type(leaf).__name__}"
            )
    if not carried:
        raise core.TraceError(
            "etl.while_loop: init must contain at least one SymbolicTensor "
            "leaf to carry around the loop — a loop over static values only "
            "is plain Python control flow, not graph control flow"
        )

    builder = current_builder()
    carried_values = tuple(leaf.value for leaf in carried)
    input_types = tuple(value.type for value in carried_values)
    cond_region = builder.build_region(input_types)
    body_region = builder.build_region(input_types)

    # --- condition region: cond_fn(carried) → ONE 0-d bool tensor ---
    cond_carried = _rebuild_carried(leaves, tree, cond_region.entry.arguments)

    def finish_cond(result: Any) -> None:
        _check_pred(result, "etl.while_loop cond_fn result")
        _return_terminator(builder, (result.value,))

    _run_in_region(builder, cond_region, cond_fn, cond_carried, after=finish_cond)

    # --- body region: body_fn(carried) → next carried values ---
    body_carried = _rebuild_carried(leaves, tree, body_region.entry.arguments)

    def finish_body(result: Any) -> None:
        body_leaves, body_tree = _flatten(result)
        if body_tree != tree:
            raise core.TraceError(
                "etl.while_loop: body_fn output tree must match init's tree "
                f"— init: {tree!r}, body: {body_tree!r}"
            )
        next_values = []
        for i, (init_leaf, body_leaf) in enumerate(zip(leaves, body_leaves)):
            if isinstance(init_leaf, core.SymbolicTensor):
                if not isinstance(body_leaf, core.SymbolicTensor):
                    raise core.TraceError(
                        f"etl.while_loop: body_fn output leaf {i} must be a "
                        f"SymbolicTensor (init has one at this position), got "
                        f"{type(body_leaf).__name__}"
                    )
                # Loop-carried types must stay constant across iterations.
                if body_leaf.dtype != init_leaf.dtype:
                    raise core.DTypeError(
                        f"etl.while_loop: body_fn output leaf {i} dtype "
                        f"{body_leaf.dtype.name} differs from the loop-carried "
                        f"dtype {init_leaf.dtype.name}"
                    )
                if not _shapes_compatible(body_leaf.shape, init_leaf.shape):
                    raise core.ShapeError(
                        f"etl.while_loop: body_fn output leaf {i} shape "
                        f"{body_leaf.shape!r} differs from the loop-carried "
                        f"shape {init_leaf.shape!r}"
                    )
                next_values.append(body_leaf.value)
            else:
                if (
                    not _is_static_value(body_leaf)
                    or type(body_leaf) is not type(init_leaf)
                    or not (body_leaf == init_leaf or body_leaf is init_leaf)
                ):
                    raise core.TraceError(
                        f"etl.while_loop: body_fn output leaf {i} must equal "
                        f"init's static value {init_leaf!r}, got {body_leaf!r} "
                        "— static leaves specialize the loop and are NOT "
                        "loop-carried"
                    )
        _return_terminator(builder, tuple(next_values))

    _run_in_region(builder, body_region, body_fn, body_carried, after=finish_body)

    op = builder.create(
        "while", operands=carried_values, regions=(cond_region, body_region)
    )
    result_iter = iter(op.results)
    rebuilt = []
    for leaf in leaves:
        if isinstance(leaf, core.SymbolicTensor):
            rebuilt.append(_to_symbolic(next(result_iter)))
        else:
            rebuilt.append(leaf)  # static leaves re-inserted unchanged
    return core.unflatten(rebuilt, tree)


def _static_leading_dim(leaf: "core.SymbolicTensor") -> int:
    """The scan length derived from a leaf's leading dim (v1: static only).

    Accepts plain ints and `Dim`/`DimExpr` that evaluate without bindings.
    Raises `core.TraceError` for symbolic dims with no known size and for
    runtime-dynamic (`None`) dims — dynamic-length scans are reserved.
    """
    dim0 = leaf.shape[0]
    if isinstance(dim0, int) and not isinstance(dim0, bool):
        return dim0
    if isinstance(dim0, core.Dim):
        if dim0.size is None:
            raise core.TraceError(
                f"etl.scan: xs leading dim {dim0!r} has no known size — "
                "symbolic/dynamic scan lengths are reserved in v1 (no silent "
                "fallback)"
            )
        return int(dim0.size)
    if isinstance(dim0, core.DimExpr):
        try:
            return int(dim0.evaluate())
        except core.ShapeError as error:
            raise core.TraceError(
                f"etl.scan: xs leading dim {dim0!r} has no known size — "
                "symbolic/dynamic scan lengths are reserved in v1 (no silent "
                "fallback)"
            ) from error
    raise core.TraceError(
        f"etl.scan: xs leading dim must be a static int, got {dim0!r} "
        "(None = runtime-dynamic; symbolic dims need a known size)"
    )


def _split_scan_result(result: Any, where: str) -> tuple:
    """Validate `f`'s `(new_carry, y_step)` pair."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise core.TraceError(
            f"etl.scan: f must return a (new_carry, y_step) pair, got "
            f"{type(result).__name__} (from the {where} call)"
        )
    return result[0], result[1]


def scan(f: Any, init: Any, xs: Any, length: Optional[int] = None) -> tuple:
    """Scan along a leading axis → `(carry, stacked_outputs)`.

    1. `xs` = `SymbolicTensor` (or pytree of them) whose leading axis is the
       scan axis. `length`: static `int` or `None` → derived from xs's
       static leading dim (must be a static int — else `core.TraceError`).
       SYMBOLIC/dynamic length → `core.TraceError` in v1 (documented:
       dynamic-scan region ops are reserved; no silent fallback). Length 0
       (empty scan) is unsupported in v1 → `core.TraceError`. An explicit
       `length` SHORTER than xs's static leading dim is a PREFIX scan: the
       scan runs only the first `length` steps and yields stacked outputs
       of shape `(length, ...)` (the shorter loop simply never touches the
       remaining elements). Only an explicit `length` LARGER than a
       statically known leading dim raises `core.TraceError`.
    2. Desugars to `while_loop`. Because the `while` op is typed
       (loop-carried value types must stay constant across iterations —
       `while_loop` enforces this), the stack accumulators are carried at
       their FULL static size `(length, ...)` instead of growing: step 0 runs
       at the ENCLOSING level (before the loop) — `(carry0, y0) = f(init,
       xs[0])` — each y0 leaf is reshaped to `(1, ...)` and `broadcast` to
       `(length, ...)` as the initial stack (row 0 is the true step-0
       value; the filler rows are overwritten by the loop). The loop then
       carries `(counter i32 0-d, carry, stacked...)` from counter = 1 and
       runs `f(carry, x_step)` ONCE inside the body region, where `x_step` =
       per-leaf `gather(xs, counter, axes=(0,))`; each y leaf is reshaped to
       `(1, ...)` and written into its stack with
       `scatter(stack, counter, step, axis=0)` (operands
       `(tensor, indices, updates)` — numpy put-along-axis semantics); the
       counter is incremented with `add`. The condition is
       `less(counter, length)`. All carried types are constant, so the final
       stacked results carry the correct `(length, ...)` type.
       (A concatenate-grown stack would change the carried type per
       iteration and yield a final result typed `(1, ...)` — unsound for a
       typed while op.)
    3. All building blocks are raw region ops via `ir.opdef(...)` +
       `Builder` (NEVER `etl.ops` — keep the import DAG acyclic): `constant`
       (np.int32 0-d payloads 0/1/length), `gather` (0-d int32 index), `add`,
       `less`, `reshape` (leading-1 step), `broadcast` (initial full-size
       stack), `scatter` (in-place step write). (The registry has no
       `expand_dims` and `slice` takes static-int attributes only — hence
       gather/reshape instead.)
    4. Returns `(final_carry, stacked_outputs)` — carry per init's tree,
       stacked outputs per y's tree (single tensor → returned bare).
    """
    xs_leaves, xs_tree = _flatten(xs)
    if not xs_leaves:
        raise core.TraceError(
            "etl.scan: xs must contain at least one SymbolicTensor leaf"
        )
    for i, leaf in enumerate(xs_leaves):
        if not isinstance(leaf, core.SymbolicTensor):
            raise core.TraceError(
                f"etl.scan: xs leaf {i} must be a core.SymbolicTensor, got "
                f"{type(leaf).__name__}"
            )
        if not leaf.shape:
            raise core.TraceError(
                f"etl.scan: xs leaf {i} must have rank >= 1 (its leading "
                f"axis is the scan axis), got shape {leaf.shape!r}"
            )

    if length is None:
        # Derive from xs's leading dims — v1: static only.
        sizes = [_static_leading_dim(leaf) for leaf in xs_leaves]
        length = sizes[0]
        if any(size != length for size in sizes[1:]):
            raise core.TraceError(
                f"etl.scan: xs leaves disagree on the leading dim: {sizes}"
            )
        if length <= 0:
            raise core.TraceError(
                f"etl.scan: xs leading dim must be >= 1, got {length} — "
                "empty scans are not supported in v1"
            )
    else:
        if not isinstance(length, int) or isinstance(length, bool):
            raise core.TraceError(
                f"etl.scan: length must be a static int or None, got {length!r}"
            )
        if length <= 0:
            raise core.TraceError(
                f"etl.scan: length must be >= 1, got {length} — empty scans "
                "are not supported in v1"
            )
        # Trace-time sanity: an explicit length must not EXCEED any
        # statically known leading dim (a shorter length is a prefix scan —
        # scan only the first `length` elements; symbolic/dynamic dims
        # defer to runtime).
        for leaf in xs_leaves:
            dim0 = leaf.shape[0]
            if isinstance(dim0, int) and not isinstance(dim0, bool) and dim0 < length:
                raise core.TraceError(
                    f"etl.scan: explicit length {length} does not match xs's "
                    f"static leading dim {dim0}"
                )

    builder = current_builder()

    def _const_i32(value: int) -> "core.SymbolicTensor":
        op = builder.create(
            "constant", attributes={"value": np.array(value, dtype=np.int32)}
        )
        return _to_symbolic(op.result)

    const_zero = _const_i32(0)
    const_one = _const_i32(1)
    const_length = _const_i32(length)

    # --- step 0 at the ENCLOSING level (its ops dominate the while op) ---
    step0_values = []
    for leaf in xs_leaves:
        gathered = builder.create(
            "gather",
            operands=(leaf.value, const_zero.value),
            attributes={"axes": (0,)},
        )
        step0_values.append(_to_symbolic(gathered.result))
    xs_step0 = core.unflatten(step0_values, xs_tree)

    with with_builder(builder):
        step0_result = f(init, xs_step0)
    carry0, y0 = _split_scan_result(step0_result, "step-0")

    init_leaves, init_tree = _flatten(init)
    carry0_leaves, carry0_tree = _flatten(carry0)
    if carry0_tree != init_tree:
        raise core.TraceError(
            "etl.scan: f's returned carry tree must match init's tree — "
            f"init: {init_tree!r}, carry: {carry0_tree!r}"
        )
    for i, leaf in enumerate(carry0_leaves):
        if not (
            isinstance(leaf, core.SymbolicTensor) or _is_static_value(leaf)
        ):
            raise core.TraceError(
                f"etl.scan: f's returned carry leaf {i} must be a "
                f"SymbolicTensor or a static Python value, got "
                f"{type(leaf).__name__}"
            )

    y0_leaves, y_tree = _flatten(y0)
    if not y0_leaves or any(
        not isinstance(leaf, core.SymbolicTensor) for leaf in y0_leaves
    ):
        raise core.TraceError(
            "etl.scan: f's y outputs must be SymbolicTensors (at least one) — "
            "a scan with no stacked tensor outputs cannot be traced"
        )

    # Initial stacked accumulators at their FULL static size: step-0
    # reshaped to (1, ...) then broadcast to (length, ...). Row 0 holds the
    # true step-0 value; rows 1..length-1 are filler, overwritten by the
    # loop's scatter. (Carried types stay constant — the while op is typed.)
    stack0_leaves = []
    for leaf in y0_leaves:
        step_dims = tuple(leaf.value.type.shape)
        reshaped = builder.create(
            "reshape",
            operands=(leaf.value,),
            attributes={"shape": (1,) + step_dims},
        )
        filled = builder.create(
            "broadcast",
            operands=(reshaped.result,),
            attributes={"shape": (length,) + step_dims},
        )
        stack0_leaves.append(_to_symbolic(filled.result))
    stack0 = core.unflatten(stack0_leaves, y_tree)

    # Carried state: (counter, carry, stacked) — all extras are
    # SymbolicTensors; static carry leaves are handled by while_loop.
    loop_init = (const_one, carry0, stack0)

    def cond_fn(state: tuple) -> "core.SymbolicTensor":
        counter = state[0]
        region_builder = current_builder()
        lt = region_builder.create(
            "less", operands=(counter.value, const_length.value)
        )
        return _to_symbolic(lt.result)

    def body_fn(state: tuple) -> tuple:
        counter, carry, stacked = state
        region_builder = current_builder()
        step_values = []
        for leaf in xs_leaves:
            gathered = region_builder.create(
                "gather",
                operands=(leaf.value, counter.value),
                attributes={"axes": (0,)},
            )
            step_values.append(_to_symbolic(gathered.result))
        x_step = core.unflatten(step_values, xs_tree)
        result = f(carry, x_step)
        new_carry, y = _split_scan_result(result, "body iteration")
        y_leaves, y_leaf_tree = _flatten(y)
        if y_leaf_tree != y_tree:
            raise core.TraceError(
                "etl.scan: f's y output tree must stay constant across "
                f"iterations — step 0: {y_tree!r}, body: {y_leaf_tree!r}"
            )
        stacked_leaves, _ = _flatten(stacked)
        appended = []
        for stack_leaf, y_leaf in zip(stacked_leaves, y_leaves):
            if not isinstance(y_leaf, core.SymbolicTensor):
                raise core.TraceError(
                    f"etl.scan: f's y leaf must be a SymbolicTensor, got "
                    f"{type(y_leaf).__name__}"
                )
            reshaped = region_builder.create(
                "reshape",
                operands=(y_leaf.value,),
                attributes={"shape": (1,) + tuple(y_leaf.value.type.shape)},
            )
            # scatter(tensor, indices, updates, axis) — numpy put-along-axis
            # semantics: write the (1, ...) step at row `counter`. The stack
            # type stays (length, ...) across iterations.
            stacked_new = region_builder.create(
                "scatter",
                operands=(stack_leaf.value, counter.value, reshaped.result),
                attributes={"axis": 0},
            )
            appended.append(_to_symbolic(stacked_new.result))
        stacked_out = core.unflatten(appended, y_tree)
        incremented = region_builder.create(
            "add", operands=(counter.value, const_one.value)
        )
        return (_to_symbolic(incremented.result), new_carry, stacked_out)

    _, final_carry, final_stacked = while_loop(cond_fn, body_fn, loop_init)
    return (final_carry, final_stacked)
