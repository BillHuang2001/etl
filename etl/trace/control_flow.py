"""Runtime tensor control flow: `cond`, `while_loop`, `scan`.

These trace Python callables into IR regions of ordinary `if`/`while` ops —
backends (the numpy interpreter) need NO special control-flow runtime
support. Region ops are built through `ir.opdef(...)` + the `ir.Builder`
directly; this module NEVER imports `etl.ops` (ops imports trace for the
active-builder hook — the DAG stays acyclic).

REGION CONVENTIONS (the implemented `etl.ir` reality — binding; the
CONTEXT.md sketch is superseded where it differs; see
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
  values. Emitted via `Builder.set_terminator` so it is APPENDED as the
  block's last op (a terminator is never emitted via `create` at the
  insertion point).

Region building (the `ir.Builder` workflow): ONE Builder instance serves the
whole trace. `builder.build_region(input_types)` creates a DETACHED
single-block region whose entry args are fresh `Value`s;
`builder.create("if"/"while", operands=..., regions=(region,), ...)` wires
the region parents. To run a callable inside a region:
`builder.push_region(region)` (insertion point → region entry), run the
callable under `with_builder(builder)` (installs the active-builder context
for `current_builder()` — the SAME builder), emit the region's `return` via
`set_terminator`, then `builder.pop_region()` (always in a `finally` so the
stack stays consistent on error).

Static values: `*operands` / `init` / kwargs may contain static Python values
(per the static-value predicate in `./trace.py`); they specialize the regions
(Python semantics — evaluated once at trace time, NOT per iteration) and are
passed to the callables unchanged. Static values are never loop-carried or
op results.

Local pytree note: `core.flatten` treats etl-module dataclasses
(`SymbolicTensor`, `Tensor`, `Device`, ir structures, ...) as LEAVES (the
module check in `core.tree._flatten_into`); only user-defined dataclasses
act as pytree containers. This module walks pytrees itself (`_flatten_tree`)
so the leaf conventions stay explicit — `SymbolicTensor`, `core.Tensor`,
`core` value objects, and `ir` structural dataclasses are LEAVES — while
mirroring `core.TreeSpec`'s container conventions (tuple/list/dict — keys
sorted — namedtuple/dataclass), so `core.unflatten` can rebuild the trees.
Leaf nodes use `TreeSpec(type=None)`, which `core.unflatten` treats as a
plain leaf.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, List, Optional

import numpy as np

from etl import core
from etl import ir

from .builder import current_builder, with_builder

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


def _is_static_value(obj: Any) -> bool:
    """True iff `obj` is a static Python value that specializes the graph.

    Local copy of the predicate in `./trace.py` (avoids coupling this module
    to the tracer's internals; keep the two in sync): `None`, bool, int,
    float, complex, str, `enum.Enum`, numpy `dtype` objects, `slice`,
    `core.Device`.
    """
    if obj is None:
        return True
    # bool must be checked before int (True is an int instance).
    if isinstance(obj, (bool, int, float, complex, str, slice, enum.Enum)):
        return True
    if isinstance(obj, np.dtype):
        return True
    if isinstance(obj, core.Device):
        return True
    return False


def _to_symbolic(value: "ir.Value") -> "core.SymbolicTensor":
    """Wrap an `ir.Value` (op result or region block arg) as a SymbolicTensor."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


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


def _flatten_tree(obj: Any, leaves: List[Any]) -> "core.TreeSpec":
    """Local pytree walk treating `SymbolicTensor` (and every other
    non-container) as a LEAF; see the module docstring "Local pytree note"."""
    if isinstance(obj, _LEAF_TYPES):
        leaves.append(obj)
        return core.TreeSpec(type=None)
    obj_type = type(obj)
    if isinstance(obj, tuple) and hasattr(obj_type, "_fields"):
        # namedtuple (checked before plain tuples, like core.TreeSpec).
        child_specs = tuple(_flatten_tree(child, leaves) for child in obj)
        return core.TreeSpec(
            type=obj_type, children=child_specs, node_data=obj_type._fields
        )
    if (
        dataclasses.is_dataclass(obj)
        and not isinstance(obj, type)
        and not obj_type.__module__.split(".")[0] == "etl"
    ):
        field_names = [field.name for field in dataclasses.fields(obj)]
        child_specs = tuple(
            _flatten_tree(getattr(obj, name), leaves) for name in field_names
        )
        return core.TreeSpec(
            type=obj_type, children=child_specs, node_data=field_names
        )
    if isinstance(obj, tuple):
        child_specs = tuple(_flatten_tree(child, leaves) for child in obj)
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, list):
        child_specs = tuple(_flatten_tree(child, leaves) for child in obj)
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, dict):
        keys = sorted(obj)  # core.TreeSpec convention: keys sorted
        child_specs = tuple(_flatten_tree(obj[key], leaves) for key in keys)
        return core.TreeSpec(type=obj_type, children=child_specs, node_data=keys)
    leaves.append(obj)
    return core.TreeSpec(type=None)


def _flatten(obj: Any) -> tuple:
    """`(leaves, treespec)` — the local-leaf variant of `core.flatten`."""
    leaves: List[Any] = []
    return leaves, _flatten_tree(obj, leaves)


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


def _run_branch(
    builder: "ir.Builder",
    region: "ir.Region",
    branch_name: str,
    fn: Any,
    operands: tuple,
    static_kwargs: dict,
) -> tuple:
    """Run one `if` branch callable inside `region` and emit its `return`.

    Positions the builder inside the region, runs
    `fn(*call_args, **static_kwargs)` exactly ONCE under `with_builder`,
    validates the outputs (all `SymbolicTensor` leaves — static leaves are
    not supported in cond branches), emits the region's `return` terminator
    with the symbolic leaf values, and restores the insertion point.

    The region's entry args bind to ALL if operands (predicate at index 0 —
    the v1 verify convention); the branch callable receives the
    captured-operand entry args (wrapped as SymbolicTensor) at the symbolic
    positions, with static operands passed unchanged.

    Returns `(output_leaves, output_tree)`.
    """
    entry_args = region.entry.arguments
    arg_iter = iter(entry_args[1:])  # skip the predicate binding at index 0
    call_args = []
    for operand in operands:
        if isinstance(operand, core.SymbolicTensor):
            call_args.append(_to_symbolic(next(arg_iter)))
        else:
            call_args.append(operand)  # static value, passed unchanged
    region_builder = _region_builder(builder, region)
    try:
        with with_builder(region_builder):
            result = fn(*call_args, **static_kwargs)
        leaves, tree = _flatten(result)
        for i, leaf in enumerate(leaves):
            if not isinstance(leaf, core.SymbolicTensor):
                raise core.TraceError(
                    f"etl.cond: {branch_name} branch output leaf {i} must be "
                    f"a SymbolicTensor, got {type(leaf).__name__} — cond "
                    "branches yield tensors only (static output leaves are "
                    "not supported)"
                )
        _return_terminator(region_builder, tuple(leaf.value for leaf in leaves))
        return leaves, tree
    finally:
        region_builder.pop_region()


def cond(pred: "core.SymbolicTensor", true_fn: Any, false_fn: Any, *operands: Any, **static_kwargs: Any) -> Any:
    """Runtime `if` over a tensor predicate → traced `if` op.

    1. `pred` must be a `core.SymbolicTensor` of 0-d bool dtype — else
       `core.TraceError` (non-scalar or non-bool or concrete value).
    2. `static_kwargs` must be static Python values (else `core.TraceError`);
       they specialize the regions and are passed to both branches as kwargs.
       `*operands` may be `SymbolicTensor`s (captured SSA values) and static
       values (specialization); passed positionally to both branches.
    3. Build the `if` op via `ir.opdef("if")` with two regions per the
       conventions above. Run `true_fn(*operands, **static_kwargs)` inside
       the then-region under `with_builder(...)`, `false_fn` likewise in the
       else-region. Each is called exactly ONCE at trace time. The branches
       receive the region's entry block args (the captured operands bound as
       block args) — not the enclosing values directly.
    4. Flatten each branch's return value (pytree); trees must be identical
       and leaves must be `SymbolicTensor` (static leaves → `TraceError`).
       Emit each region's `return` terminator with its leaves; result
       dtype/shape unification across branches (dtype mismatch →
       `core.DTypeError`, shape mismatch → `core.ShapeError`).
    5. Create the `if` op with the explicit unified result types and return
       its results unflattened per the branch output tree (single tensor →
       returned bare).

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
    for i, operand in enumerate(operands):
        if not isinstance(operand, core.SymbolicTensor) and not _is_static_value(
            operand
        ):
            raise core.TraceError(
                f"etl.cond: operand {i} must be a core.SymbolicTensor or a "
                f"static Python value, got {type(operand).__name__}"
            )

    builder = current_builder()
    if_values = (pred.value,) + tuple(
        operand.value
        for operand in operands
        if isinstance(operand, core.SymbolicTensor)
    )
    input_types = tuple(value.type for value in if_values)
    then_region = builder.build_region(input_types)
    else_region = builder.build_region(input_types)

    then_leaves, then_tree = _run_branch(
        builder, then_region, "true", true_fn, operands, static_kwargs
    )
    else_leaves, else_tree = _run_branch(
        builder, else_region, "false", false_fn, operands, static_kwargs
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
    result_types = []
    for i, (true_leaf, false_leaf) in enumerate(zip(then_leaves, else_leaves)):
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

    op = builder.create(
        "if",
        operands=if_values,
        regions=(then_region, else_region),
        result_types=tuple(result_types),
    )
    results = [_to_symbolic(result) for result in op.results]
    return core.unflatten(results, then_tree)


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

    v1 note: the condition/body callables run ONCE at trace time; the traced
    regions repeat their IR at run time — no Python callbacks.
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
    _region_builder(builder, cond_region)
    try:
        with with_builder(builder):
            cond_result = cond_fn(cond_carried)
        _check_pred(cond_result, "etl.while_loop cond_fn result")
        _return_terminator(builder, (cond_result.value,))
    finally:
        builder.pop_region()

    # --- body region: body_fn(carried) → next carried values ---
    body_carried = _rebuild_carried(leaves, tree, body_region.entry.arguments)
    _region_builder(builder, body_region)
    try:
        with with_builder(builder):
            body_result = body_fn(body_carried)
        body_leaves, body_tree = _flatten(body_result)
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
                if body_leaf.shape != init_leaf.shape:
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
    finally:
        builder.pop_region()

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

    final_counter, final_carry, final_stacked = while_loop(
        cond_fn, body_fn, loop_init
    )
    return (final_carry, final_stacked)


def _return_terminator(builder: "ir.Builder", values: Any) -> "ir.Op":
    """Build the `return` terminator op (`ir.opdef("return")`) yielding
    `values` (a flat sequence of `ir.Value`) in the builder's current block.

    Delegates to `Builder.set_terminator`, which APPENDS the op (last by
    construction) — a terminator is never emitted via `create` at the
    insertion point. Private helper for `trace`, `cond`, `while_loop`,
    `scan`.
    """
    return builder.set_terminator(builder.current_block, "return", tuple(values))


def _region_builder(enclosing: "ir.Builder", region: "ir.Region") -> "ir.Builder":
    """Position `enclosing` inside `region`'s entry block and return it.

    With the single-Builder workflow there is no separate "region builder"
    object: this pushes `region`'s entry block onto the enclosing builder's
    insertion-point stack (creating the block first if the region is empty)
    and returns the SAME builder. The caller runs the region callable under
    `with_builder(builder)` (the active-builder context for
    `current_builder()`) and MUST call `builder.pop_region()` afterwards —
    `cond`/`while_loop`/`scan` do so in a `finally` block so the stack stays
    consistent on error.
    """
    if not region.blocks:
        enclosing.insert_block(region)
    enclosing.push_region(region)
    return enclosing
