"""Public `vmap` — function-side sugar over `vectorize`.

`vmap` uses exactly the same batching machinery as `vectorize`; the
equivalence contract (binding, tested) lives in `./CONTEXT.md`:

* `vmap(graph, in_axes, out_axes=0)` ≡ `vectorize(graph, in_axes)` followed by
  an output-axis rearrangement step (ordinary transpose / size-one axis
  insertion ops) driven by `out_axes`; with `out_axes=0` both produce
  identical IR.
* `vmap(fn_or_defn, in_axes, out_axes)` returns a `TransformCallable`:
  `vmap(f)(*batched_specs)` ≡ rearrange(vectorize(etl.trace(f,
  *strip(batched_specs)), in_axes), out_axes), where `strip` removes the
  leading mapped dim from each mapped spec (dtype/device/name preserved).
  The wrapped function is traced exactly ONCE per call.
"""

from __future__ import annotations

import re

from etl import core
from etl import ir
from etl.ir.builder import InsertionPoint
from etl.trace import Graph, trace
from etl.trace.graph import _normalize_leaf_types
from etl.transforms._wrappers import TransformCallable
from etl.transforms.batching import with_batch_depth
from etl.transforms.vectorize import (
    _is_int,
    _normalize_axis_entries,
    vectorize,
)

_VMAP_CALLABLE_DOC = """\
`vmap(f)` applied to a callable/`Defn` returns a TransformCallable:
calling it with a structure of TensorSpecs (mapped inputs include the leading
batch dim) + static values returns the vectorized Graph, tracing `f` exactly
once. Equivalent to: trace `f` with the leading mapped dim stripped from each
mapped spec, then vectorize, then rearrange outputs per out_axes.

`vmap(tf)` applied to another TransformCallable (composition — nested vmap,
`vmap(grad(f))`, ...) also returns a TransformCallable: calling it invokes
the inner callable with the stripped specs (which traces its own wrapped
function exactly once) and vectorizes the returned Graph with the outer
axes, then rearranges outputs per out_axes — each level adds one leading
mapped dim."""

#: Reserved name pattern of the fresh batch dims `vectorize_graph` introduces
#: (ONE shared `Dim` per pass, named from the active pass depth — `"batch"`
#: at depth 0, `"batch_1"` at depth 1, ... — see
#: `batching._batched_input_specs`). These are the ONLY dims mapped-output
#: detection keys on (by identity); the `_\d+` suffix stays accepted for dims
#: named `batch_1` in pre-existing or user-built graphs.
_BATCH_DIM_NAME = re.compile(r"^batch(_\d+)?$")


def vmap(fn_or_graph, in_axes=0, out_axes=0):
    """Vectorize a function or graph over a leading axis (function-side sugar).

    Args:
        fn_or_graph: a `Graph` (vectorized directly), or a callable/`Defn`
            (returned as a `TransformCallable`; see below).
        in_axes: `int`, `None`, or a pytree thereof matching the inputs — the
            axis of each input that becomes the vectorized (leading) axis;
            `None` leaves an input unmapped. v1 supports entries in {None, 0};
            other ints raise `core.TransformError` (deferred).
        out_axes: `int`, `None`, or a pytree thereof matching the outputs —
            where each mapped output axis should end up. `0` keeps it leading
            (an unmapped output gets an explicitly requested size-one axis at
            0); `None` requires the output to be unmapped, otherwise
            `core.TransformError` (axis mismatch). v1: {None, 0}.

    Returns:
        * `Graph` when given a `Graph` — the vectorized graph with outputs
          rearranged per `out_axes`.
        * `TransformCallable` when given a callable/`Defn` — `vmap(f)(*specs)`
          returns the vectorized `Graph`; `f` is invoked exactly once, in
          graph-building mode. Passing concrete tensors raises `TraceError`
          (transforms never execute; build the returned graph explicitly).
        * `TransformCallable` when given another `TransformCallable`
          (composition — `vmap(vmap(f))`, `vmap(grad(f))`, ...): calling the
          result invokes the inner callable with the stripped specs (it traces
          its wrapped fn exactly once), then vectorizes the returned graph
          with the outer `in_axes` and rearranges per `out_axes`. Nested
          levels compose mechanically, each adding one leading mapped dim.

    Unsupported ops without a batching rule raise `core.TransformError`
    naming the op — never a silent fallback. Nested vmap is supported when the
    required batching rules exist (each level adds one leading mapped dim).
    """
    if isinstance(fn_or_graph, Graph):
        graph = vectorize(fn_or_graph, in_axes)
        return _rearrange_outputs(graph, out_axes)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__etl_defn__", False):
        return TransformCallable(
            build=lambda *args: _vmap_fn(fn_or_graph, args, in_axes, out_axes),
            kind="vmap",
            doc=_VMAP_CALLABLE_DOC,
        )

    raise TypeError("vmap expects an etl.Graph, a callable, or an etl.Defn")


def _vmap_fn(fn, args, in_axes, out_axes) -> Graph:
    """Build the `TransformCallable` result for one `vmap(f)(*args)` call.

    Plain callable/`Defn`: trace `fn` exactly once with unvectorized specs
    derived from `args` (strip the leading mapped dim from each mapped spec
    per `in_axes`), then vectorize and rearrange outputs per `out_axes`.

    `TransformCallable` (`vmap` of another transform — e.g. `vmap(vmap(f))`,
    `vmap(grad(f))`): composition. The inner callable already maps stripped
    specs to a `Graph`, so it is INVOKED with the stripped argument structure
    (tracing its own wrapped fn exactly once) instead of being traced here;
    the outer `in_axes` is then applied to that graph via `vectorize` — the
    inner graph's input tree has the same structure as the stripped args, so
    `in_axes` re-normalizes cleanly — followed by the usual output
    rearrangement. Depth-3+ nesting recurses through the same path (each
    nested `vmap` layer is itself a `TransformCallable`).
    """
    unvectorized = _derive_unvectorized_args(args, in_axes)
    if isinstance(fn, TransformCallable):
        # Composition (nested vmap): the inner callable runs one level deeper
        # so its vectorize pass names its batch dim `batch_1`, `batch_2`, ...
        # while the outer pass below runs at the parent depth and keeps the
        # current level's name — the numpy interpreter binds symbolic dims BY
        # NAME, so same-named dims from different nesting levels would
        # collide at run time (unequal extents raise a misleading ShapeError).
        with with_batch_depth():
            inner_graph = fn(*unvectorized)
        graph = vectorize(inner_graph, in_axes)
    else:
        graph = trace(fn, *unvectorized)
        graph = vectorize(graph, in_axes)
    return _rearrange_outputs(graph, out_axes)


def _derive_unvectorized_args(args, in_axes):
    """Given the batched specs/static values `args` and `in_axes`, return the
    underlying (unvectorized) spec structure for tracing the wrapped function:
    mapped entries drop their leading dim (dtype/device/name preserved);
    unmapped entries pass through; static values stay static.

    `in_axes` follows the same normalization rules as
    `vectorize._normalize_axes` (pytree matching `args`; bare int/None only
    with exactly one tensor spec; v1 entries {None, 0}; static leaves None).
    Concrete `core.Tensor` leaves raise `core.TraceError` (transforms never
    execute; `TransformCallable` performs the same check at its boundary).
    """
    args_leaves, args_spec = core.flatten(args)
    for index, leaf in enumerate(args_leaves):
        if isinstance(leaf, core.Tensor):
            raise core.TraceError(
                "calling an etl.vmap result with concrete Tensors is not "
                f"supported (argument leaf {index}): transforms build Graphs "
                "from TensorSpecs and never execute. Pass TensorSpecs to "
                "obtain the transformed Graph, then build/run it explicitly "
                "(etl.build / etl.run)."
            )
    # `core.flatten` records a `TensorSpec` (an etl dataclass) as a leaf VALUE
    # but classifies the leaf *spec* as an empty container, so `core.unflatten`
    # would try to rebuild it as a dataclass node — normalize the leaf types
    # first (the same trick etl/trace/graph.py uses for output trees).
    args_spec = _normalize_leaf_types(args_spec)
    tensor_positions = {
        index
        for index, leaf in enumerate(args_leaves)
        if isinstance(leaf, core.TensorSpec)
    }
    ranks = {position: args_leaves[position].rank for position in tensor_positions}

    if in_axes is None or _is_int(in_axes):
        if len(tensor_positions) != 1:
            raise core.TransformError(
                "vmap: a bare int/None in_axes applies only when the "
                "arguments have exactly ONE tensor spec; got "
                f"{len(tensor_positions)} tensor specs — pass a pytree "
                "matching the argument structure instead"
            )
        entries = [None] * len(args_leaves)
        entries[sorted(tensor_positions)[0]] = in_axes
    else:
        axes_leaves, axes_spec = core.flatten(in_axes)
        mismatch = core.first_mismatch_path(
            axes_spec, args_spec, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TransformError(
                "vmap: the in_axes pytree does not match the argument "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; in_axes must be a pytree with "
                "the same container structure as the arguments (0 maps a "
                "tensor spec's leading axis, None leaves it unmapped)"
            )
        entries = list(axes_leaves)

    normalized = _normalize_axis_entries(
        entries, tensor_positions, ranks, "vmap"
    )
    unvectorized_leaves = []
    for index, leaf in enumerate(args_leaves):
        if index in tensor_positions and normalized[index] == 0:
            spec = leaf
            unvectorized_leaves.append(
                core.TensorSpec(
                    shape=spec.shape[1:],
                    dtype=spec.dtype,
                    device=spec.device,
                    name=spec.name,
                )
            )
        else:
            unvectorized_leaves.append(leaf)
    return core.unflatten(unvectorized_leaves, args_spec)


# --- output-axis rearrangement ----------------------------------------------


def _collect_batch_dims(graph: Graph):
    """The fresh batch `Dim` objects this vectorization introduced.

    `vectorize_graph` prepends ONE fresh symbolic batch `Dim` (named from the
    active pass depth — `"batch"` at depth 0, `"batch_1"` at depth 1, ...)
    shared by all mapped input specs of the pass (unmapped specs are reused
    unchanged), so the batch dims of THIS vectorization are exactly the
    leading entries of the input specs whose names follow that reserved
    pattern. Collected as objects — nested vmap levels create their OWN fresh
    Dims with level-distinct names, and only the current level's objects must
    count (output detection compares by identity, `is`). An empty list (no
    mapped inputs) makes every output unmapped.
    """
    batch_dims = []
    for spec in graph.tensor_specs:
        if not spec.shape:
            continue
        leading = spec.shape[0]
        if isinstance(leading, core.Dim) and _BATCH_DIM_NAME.match(leading.name):
            batch_dims.append(leading)
    return batch_dims


def _is_batch_entry(entry, batch_dims) -> bool:
    """True when one shape entry IS a batch dim (identity) or a `DimExpr`
    built purely from batch dims and ints — e.g. the `max(batch, batch)` a
    symbolic broadcast of mapped operands from DIFFERENT vmap levels produces
    (distinct Dim objects never compare equal, so their broadcast stays a max
    DimExpr; within one pass the shared dim compares equal and no max arises).
    Such an entry still IS the mapped leading axis."""
    if isinstance(entry, core.Dim):
        return any(entry is dim for dim in batch_dims)
    if isinstance(entry, core.DimExpr):
        return _dims_in_batch(entry, batch_dims)
    return False


def _dims_in_batch(expr, batch_dims) -> bool:
    if isinstance(expr, core.Dim):
        return any(expr is dim for dim in batch_dims)
    if isinstance(expr, core.DimExpr):
        return _dims_in_batch(expr.left, batch_dims) and _dims_in_batch(
            expr.right, batch_dims
        )
    return True  # int factors are fine


def _leading_batch_count(shape, batch_dims) -> int:
    """The length of the leading contiguous prefix of `shape` made of batch
    dims (identity-checked). > 0 ⇒ the value is mapped."""
    count = 0
    for entry in shape:
        if _is_batch_entry(entry, batch_dims):
            count += 1
        else:
            break
    return count


def _rearrange_outputs(graph: Graph, out_axes) -> Graph:
    """Post-vectorize output-axis rearrangement, as ordinary ops added to the
    already-vectorized graph (never inside batching rules).

    `out_axes` is normalized against the graph's output tree (pytree of
    0|None; bare 0/None allowed with at most one tensor output — a no-op
    when there are none; static output leaves must be None; v1 entries
    {None, 0}). Each tensor output's
    mapped-ness is detected by identity against the fresh batch Dims in the
    input specs (see `_collect_batch_dims`). Per output:

    * `out_axes` = 0 keeps the mapped axis leading — if the output is
      unmapped, an explicitly requested size-one axis is inserted at 0 via a
      reshape op;
    * `out_axes` = None requires the output to be unmapped, otherwise
      `core.TransformError` (axis mismatch — a batch axis is never silently
      dropped).

    With `out_axes` = 0 for every mapped output, NO extra op is added and the
    result is IR-identical to plain `vectorize` — the graph is returned
    untouched (this is the vmap≡vectorize equivalence path).
    """
    function = graph.module.main
    block = function.entry_block
    terminator = block.terminator
    if terminator is None or terminator.name != "return":
        raise core.TransformError(
            "vmap: the vectorized graph's entry block has no 'return' "
            "terminator — cannot rearrange outputs"
        )
    outputs = terminator.operands  # tensor outputs, in output-leaf order

    static_positions = {record.index for record in graph.output_static_values}
    tensor_positions = set(range(graph.output_tree.num_leaves)) - static_positions
    if len(outputs) != len(tensor_positions):
        raise core.TransformError(
            "vmap: internal error — the return terminator yields "
            f"{len(outputs)} values but the output tree has "
            f"{len(tensor_positions)} tensor leaves"
        )

    if out_axes is None or _is_int(out_axes):
        # A bare entry applies to the single tensor output; with ZERO tensor
        # outputs there is nothing to rearrange and the bare form (e.g. the
        # out_axes=0 default) is a no-op. Multiple tensor outputs need a
        # pytree (a bare entry would be ambiguous).
        if len(tensor_positions) > 1:
            raise core.TransformError(
                "vmap: a bare int/None out_axes applies only when the graph "
                f"has at most ONE tensor output; got {len(tensor_positions)} "
                "tensor outputs — pass a pytree matching the output "
                "structure instead"
            )
        entries = [None] * graph.output_tree.num_leaves
        if tensor_positions:
            entries[sorted(tensor_positions)[0]] = out_axes
    else:
        axes_leaves, axes_spec = core.flatten(out_axes)
        mismatch = core.first_mismatch_path(
            axes_spec, graph.output_tree, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TransformError(
                "vmap: the out_axes pytree does not match the output "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; out_axes must be a pytree with "
                "the same container structure as the outputs (0 keeps a "
                "mapped axis leading / inserts a size-one axis for an "
                "unmapped output, None requires the output unmapped)"
            )
        entries = list(axes_leaves)
    # No rank range-check for outputs: entry 0 on a rank-0 unmapped output is
    # the legal size-one-axis insertion, not an error.
    out_entries = _normalize_axis_entries(
        entries, tensor_positions, None, "vmap"
    )
    tensor_entries = [out_entries[pos] for pos in sorted(tensor_positions)]

    batch_dims = _collect_batch_dims(graph)
    mapped = [
        _leading_batch_count(value.type.shape, batch_dims) > 0
        for value in outputs
    ]

    for index, (entry, is_mapped) in enumerate(zip(tensor_entries, mapped)):
        if entry is None and is_mapped:
            raise core.TransformError(
                f"vmap: out_axes=None at output leaf {index} requires the "
                "output to be unmapped, but it carries a mapped batch axis "
                "— a batch axis is never silently dropped"
            )
    needs_insert = [
        entry == 0 and not is_mapped
        for entry, is_mapped in zip(tensor_entries, mapped)
    ]
    if not any(needs_insert):
        return graph
    return _insert_size_one_axes(graph, outputs, needs_insert, terminator)


def _insert_size_one_axes(graph, outputs, needs_insert, old_terminator) -> Graph:
    """Insert size-one leading axes for unmapped outputs, as ordinary
    `reshape` ops spliced into the module right before the `return`
    terminator. The graph is mutated IN PLACE — its module/trees stay the
    same objects, only the IR body changes.

    Splice mechanism (Use-bookkeeping-safe):

    1. Clear the `Use` records the old terminator left on its operand values,
       then `block.erase` it (erase only clears `op.parent` — stale Use
       records would fail `verify`'s forward check).
    2. Create a fresh `ir.Builder` over the SAME module (op/value id counters
       continue) and set its insertion point to the END of the block via
       `InsertionPoint` — the public builder API (`push_region`/
       `set_insertion_point`) only offers position 0, which would put the new
       ops before their operand definitions.
    3. Emit one `reshape` per changed output through `builder.create` (the IR
       path accepts `None` dims in the shape attr, which the frontend
       `ops.reshape` rejects).
    4. Re-emit the `return` terminator with the new operand tuple via
       `set_terminator`, which wires fresh Use records.
    """
    function = graph.module.main
    block = function.entry_block
    for operand in old_terminator.operands:
        operand.uses = [
            use for use in operand.uses if use.owner is not old_terminator
        ]
    block.erase(old_terminator)

    builder = ir.Builder(graph.module)
    builder._insertion_stack.append(InsertionPoint(block, len(block.ops)))

    new_operands = []
    for value, changed in zip(outputs, needs_insert):
        if not changed:
            new_operands.append(value)
            continue
        op = builder.create(
            "reshape",
            operands=(value,),
            attributes={"shape": (1, *value.type.shape)},
            location=old_terminator.location,
        )
        new_operands.append(op.result)
    builder.set_terminator(
        block,
        "return",
        operands=tuple(new_operands),
        location=old_terminator.location,
    )
    return graph
