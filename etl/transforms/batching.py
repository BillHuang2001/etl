"""Batching: the vectorize primitive's rule registry and core algorithm.

`vectorize` is THE primitive graph transformation of this package: it rewrites
a traced `Graph` into a new `Graph` in which inputs mapped by `axes` carry
explicit leading batch dims and every op has been rewritten by its batching
rule. The result graph contains only ordinary `etl.ops` ops — backends never
need to understand vectorization (binding: root CONTEXT.md design principle 7;
`./CONTEXT.md` "The vectorize core" and "Rule-call signatures").

Rule convention: while a rule runs, the machinery has pushed its `ir.Builder`
onto the trace builder stack, so rules build replacement ops with ordinary
`etl.ops.*` functions (`trace.current_builder()` resolves to the transform
builder). Rules are pure graph builders: no Python loops over batch elements,
no silent fallbacks.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, Dict, Optional, Tuple

from etl import core
from etl import ir
from etl import ops
from etl.core import TransformError
from etl.trace import Graph, StaticValue, current_builder, with_builder
from etl.transforms._metadata import MappedAxes, UNMAPPED, ValueEnv

# Binding rule signature (see ./CONTEXT.md):
#   rule(op, operands, axes) -> (new_values, new_axes)
#   * op:          the ir.Op being vectorized (its operands/results are the
#                  ORIGINAL graph's values — read them for original types)
#   * operands:    tuple of ir.Value — the REWRITTEN operand values of the new
#                  module, aligned with `op.operands`. Rules must build their
#                  replacement ops over THESE values (never over the original
#                  op.operands) so shape/dtype inference sees the batched types.
#   * axes:        tuple of MappedAxes aligned with `operands`
#   * new_values:  tuple of ir.Value aligned with `op.results`
#   * new_axes:    tuple of MappedAxes aligned with `new_values`
BatchingRule = Callable[
    [ir.Op, Tuple[ir.Value, ...], Tuple[MappedAxes, ...]],
    Tuple[Tuple[ir.Value, ...], Tuple[MappedAxes, ...]],
]

#: Op-def-name → batching rule. Custom blocks register under `block:<block_name>`
#: (done by `BlockOp.batching_rule(fn)` in `etl/block` — transforms never
#: imports block). Builtin rules are registered by `rules.py` at import time.
batching_rules: Dict[str, BatchingRule] = {}


def register_batching_rule(op_name: str, fn: BatchingRule) -> None:
    """Register (or replace) the batching rule for `op_name`.

    Custom blocks register under the `block:<block_name>` namespace; that is
    exactly what `BlockOp.batching_rule(fn)` does in `etl/block`.
    """
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    batching_rules[op_name] = fn


def get_batching_rule(op_name: str) -> Optional[BatchingRule]:
    """The registered rule for `op_name`, or `None` (never raises)."""
    return batching_rules.get(op_name)


def require_batching_rule(op_name: str) -> BatchingRule:
    """Like `get_batching_rule`, but raises `TransformError` naming the op."""
    rule = batching_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"vectorize: no batching rule for op '{op_name}'. "
            f"Register one with register_batching_rule('{op_name}', fn) "
            f"(custom blocks: BlockOp.batching_rule); there is no silent "
            f"Python-loop fallback."
        )
    return rule


#: Pytree node type → batched-output aux remap fn. Registered pytree nodes
#: whose children include STATIC leaves (e.g. a future sparse-tensor node:
#: children = [tensor leaves..., dense_shape ints, dtype, format]) get their
#: OUTPUT static leaves remapped at the vectorize boundary — `vectorize_graph`
#: prepends a batch dim to the tensor outputs, which goes stale for node
#: metadata derived from tensor shapes (the sparse value's dense_shape must
#: gain the batch dim as a leading symbolic `Dim`). INPUT-side static values
#: are never remapped (the batch lives in the flat tensor-leaf specs only).
_BATCHED_AUX_REMAP: Dict[type, Callable] = {}


def register_batched_aux_remap(node_type: type, fn: Callable) -> None:
    """Register (or replace) the batched-output aux remap fn for `node_type`.

    At the vectorize boundary (`vectorize_graph`, step 6), every registered
    pytree node in the OUTPUT tree with static-leaf children hands its node
    spec, its direct static-leaf child values (in child order), and the
    current pass's fresh batch `Dim` to `fn`, and the returned values become
    the node's new static leaves (rebuilt as fresh plain-leaf
    `TreeSpec(type(None))` specs — the canonical normalized leaf; a
    `type=None` field would classify as a TENSOR position in the pipeline's
    `_is_tensor_leaf_spec`) appended after the non-static children. Static leaves inside
    unregistered subtrees and inside nested container children keep their
    values verbatim (only their flat index/path are recomputed) — they are
    NOT passed to `fn`.

    Args:
        node_type: The pytree node class whose output static leaves are
            remapped (looked up by MRO walk — a registered base class catches
            subclasses, exact type first).
        fn: ``fn(node: core.TreeSpec, static_values: tuple, batch_dim:
            core.Dim) -> tuple`` — returns the REMAPPED static leaf values
            (the count may change).

    Raises:
        TypeError: If ``node_type`` is not a type or ``fn`` is not callable;
            if ``node_type`` is ``object`` itself (every type's MRO ends in
            ``object``, so registering it would hijack the lookup for all
            types — the same guard as ``core.register_pytree_node``).
    """
    if node_type is object:
        raise TypeError(
            "register_batched_aux_remap: cannot register 'object' — it "
            "would hijack the MRO lookup for all types"
        )
    if not isinstance(node_type, type):
        raise TypeError(f"node_type must be a type, got {node_type!r}")
    if not callable(fn):
        raise TypeError("fn must be callable")
    _BATCHED_AUX_REMAP[node_type] = fn


def _lookup_batched_aux_remap(spec) -> Optional[Callable]:
    """The registered batched-aux remap fn for `spec`, or `None` (never raises).

    MRO walk over ``spec.type.__mro__`` mirroring
    ``core.tree._flatten_into`` (first match wins, exact type first) so a
    registered base class catches subclasses. Non-type specs (leaf specs may
    carry ``type=None``) have no registrations → ``None``.
    """
    spec_type = spec.type
    if not isinstance(spec_type, type):
        return None
    for base in spec_type.__mro__:
        fn = _BATCHED_AUX_REMAP.get(base)
        if fn is not None:
            return fn
    return None


def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an `ir.Value` as a `core.SymbolicTensor` (dtype/shape from its type)."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def block_call_pass_through_rule() -> BatchingRule:
    """Built-in batching rule for elementwise/map_over_batch block calls.

    Both policies make batch dims pass through untouched — `elementwise`
    blocks act per-element on ALL dims, `map_over_batch` blocks map over
    their leading batch dims internally — so vectorizing them is safe
    WITHOUT a registered rule. The rule therefore only

    1. aligns operands exactly like the pointwise batching rule: reshape
       each operand to `own batch dims + (1,)*(max_row_rank − row_rank) +
       per-row shape` (skipped when already aligned) so per-row dims align
       on the right and a mapped operand's batch dims can never be mistaken
       for an unmapped operand's per-row dims;
    2. prepends the common batch dims — the leading `mapped_count` shape
       slice of the operand with the most mapped axes (the first one when
       several tie) — to every ORIGINAL result type; per-row result dims
       are unchanged;
    3. rebuilds the `block_call` over the aligned operands through the
       active transform builder (`result_specs` and the op's result types
       must be the very same `ir.ValueType` tuple — `ir.verify` requires
       exact equality).

    Result axes = union of the operand mapped axes (leading-contiguous, as
    `_validate_rule_result` requires). `etl/block` may pre-register this
    factory's product for blocks declared with an `elementwise` or
    `map_over_batch` batching policy.
    """
    def rule(op, operands, axes):
        counts = [ax.count for ax in axes]
        mapped_count = max(counts) if counts else 0
        # Per-row rank = everything past the operand's own mapped batch dims.
        row_ranks = [value.type.rank - count for value, count in zip(operands, counts)]
        max_row_rank = max(row_ranks) if operands else 0
        values = []
        for value, count, row_rank in zip(operands, counts, row_ranks):
            target = (
                value.type.shape[:count]
                + (1,) * (max_row_rank - row_rank)
                + value.type.shape[count:]
            )
            if target == value.type.shape:
                out = value  # already aligned: no reshape (common equal-rank case)
            else:
                out = ops.reshape(_sym(value), target).value
            values.append(out)
        if mapped_count == 0:
            batch_dims = ()
        else:
            # The first operand with the most mapped axes supplies the batch
            # dims (guaranteed to exist: mapped_count IS some operand's count).
            batch_dims = next(
                value.type.shape[:mapped_count]
                for value, count in zip(values, counts)
                if count == mapped_count
            )
        result_types = tuple(
            ir.ValueType(
                dtype=result.type.dtype, shape=batch_dims + tuple(result.type.shape)
            )
            for result in op.results
        )
        builder = current_builder()
        new_op = builder.create(
            "block_call",
            operands=tuple(values),
            attributes={
                "block_name": op.attributes["block_name"],
                "static_args": op.attributes["static_args"],
                "result_specs": result_types,
            },
            result_types=result_types,
            location=op.location,
        )
        result_axes = MappedAxes(tuple(range(mapped_count)))
        return tuple(new_op.results), (result_axes,) * len(op.results)

    return rule


def _rule_key(op: ir.Op) -> str:
    """Registry key for `op` (block calls are keyed ``block:<block_name>``,
    external calls keyed ``external:<name>``)."""
    if op.name == "block_call":
        return f"block:{op.attributes['block_name']}"
    if op.name == "external_call":
        return f"external:{op.attributes['name']}"
    return op.name


def _resolve_batching_rule(op: ir.Op) -> BatchingRule:
    """The rule vectorizing `op`.

    Lookup: the registry first — block calls under the namespaced key
    `block:<block_name>`, external calls under `external:<name>` (exactly
    like autodiff's `_rule_name`; an explicitly registered rule always
    wins). No registered rule ⇒ `TransformError` naming the op/block/kernel
    with an adaptive registration hint — never a silent fallback.
    """
    key = _rule_key(op)
    rule = batching_rules.get(key)
    if rule is not None:
        return rule
    if key.startswith("block:"):
        block_name = op.attributes["block_name"]
        raise TransformError(
            f"vectorize: no batching rule for block '{block_name}' — "
            f"register one with register_batching_rule('{key}', fn) "
            f"(custom blocks: BlockOp.batching_rule); there is no silent "
            f"Python-loop fallback."
        )
    if key.startswith("external:"):
        name = op.attributes["name"]
        raise TransformError(
            f"vectorize: no batching rule for op 'external_call' (key "
            f"'{key}') — register one via the external-kernel handle "
            f"returned by etl.register_external_kernel('{name}', fn) "
            f"(.batching_rule/.vjp_rule/.jvp_rule) or register a portable "
            f"decomposition; there is no silent Python-loop fallback."
        )
    return require_batching_rule(op.name)


def vectorize_graph(graph: Graph, axes) -> Graph:
    """Core vectorize algorithm: rewrite a traced graph with batched inputs.

    Walks the graph's function blocks in topological order, rewrites each op
    via its batching rule, seeds input metadata from the normalized `axes`
    structure, and builds a NEW `Graph` (new module/function — the input graph
    is never mutated) whose mapped inputs/outputs carry an extra leading
    dimension. All mapped input specs share ONE fresh symbolic `Dim` (named
    from the active pass depth — `_batch_dim`: `"batch"` at depth 0,
    `"batch_1"` at depth 1, ...) as the leading dim — a single pass
    introduces exactly one batch axis; `output_tree` and `static_values` are
    preserved, except that registered output-tree nodes with static-leaf
    children (e.g. a future sparse-tensor node) get their OUTPUT static
    leaves remapped at this boundary via `register_batched_aux_remap`
    (input-side statics are never remapped). Region-bearing control-flow ops
    (`cond`/`while_loop`/`scan`) are not vectorizable in v1 and raise
    `TransformError`.

    Args:
        graph: the traced graph to rewrite.
        axes: the NORMALIZED axes pytree (produced by
            `vectorize._normalize_axes`): `int 0` or `None` at every input-
            tree leaf, matching the graph's input structure; `0` maps that
            tensor input's leading axis, `None` leaves it unmapped (mandatory
            for static leaves).

    Output specs are NOT computed here: they are implicit in the return
    values' `ir.ValueType`s (rules + type inference produce them), so mapped
    outputs automatically carry the leading batch dim.
    """
    # 1. Split the normalized axes pytree into per-tensor-input entries and
    #    guard the v1 module shape (exactly one function, one block).
    tensor_axes = _tensor_input_axes(graph, axes)
    old_function = _entry_function(graph)
    old_block = old_function.entry_block
    if len(old_block.arguments) != len(graph.tensor_specs):
        raise TransformError(
            "vectorize: the graph function has "
            f"{len(old_block.arguments)} block args but "
            f"{len(graph.tensor_specs)} tensor input specs"
        )

    # 2. New input specs: all mapped inputs share ONE fresh leading symbolic
    #    dim (`batch`); unmapped specs are reused unchanged. The one fresh
    #    `batch_dim` object is threaded into the output-aux remap (step 6) so
    #    remapped static leaves share the batch Dim identity with the tensor
    #    output/input specs.
    batch_dim = _batch_dim()
    new_specs, mapped_input_axes = _batched_input_specs(graph, tensor_axes, batch_dim)

    # 3. Build the NEW module/function (the input graph is never mutated).
    builder = ir.Builder()
    module = builder.build_module(
        name=graph.module.name, metadata=dict(graph.module.metadata)
    )
    input_types = tuple(
        ir.ValueType(spec.dtype, tuple(spec.shape)) for spec in new_specs
    )
    function = builder.build_function(
        name=old_function.name,
        input_types=input_types,
        metadata=dict(old_function.metadata),
    )
    new_block = function.entry_block

    # 4. Seed the metadata env (new block args → MappedAxes) and the old→new
    #    value remap; carry the input source locations forward.
    env = ValueEnv()
    remap: Dict[int, ir.Value] = {}
    source_locations: Dict[int, object] = {}
    for old_arg, new_arg, mapped in zip(
        old_block.arguments, new_block.arguments, mapped_input_axes
    ):
        env.set(new_arg, mapped)
        remap[old_arg.id] = new_arg
        location = graph.source_locations.get(old_arg.id)
        if location is not None:
            source_locations[new_arg.id] = location

    # 5. Rewrite the body with the transform builder active: rules build
    #    replacement ops through ordinary `etl.ops.*` functions, which
    #    resolve the builder via `trace.current_builder()`.
    with with_builder(builder):
        _rewrite_block(old_block, env, builder, remap, source_locations, graph)

    # 6. Wrap the new IR into a NEW Graph: the input/output tree skeletons
    #    and static values are reused unchanged — only the flat tensor specs
    #    and the module are new. Registered output-tree nodes with static-leaf
    #    children get their OUTPUT static leaves remapped here
    #    (`_remap_output_aux` — e.g. a sparse node's dense_shape gains the
    #    batch dim); input statics are never remapped.
    output_tree, output_static_values = _remap_output_aux(graph, batch_dim)
    return Graph(
        module,
        graph.input_specs,
        new_specs,
        output_tree,
        graph.static_values,
        output_static_values,
        source_locations,
    )


def _tensor_input_axes(graph: Graph, axes) -> Tuple[Optional[int], ...]:
    """Split the normalized axes pytree into per-tensor-input entries.

    `core.flatten` walks the axes pytree in the same pre-order as the graph's
    input tree (both sort dict keys), so leaf ``i`` corresponds to the ``i``-th
    input leaf. Static leaves (positions recorded in `graph.static_values`)
    must be `None`; the returned tuple has one entry per TENSOR input: ``0``
    (map the leading axis) or `None` (unmapped).
    """
    axes_leaves, _ = core.flatten(axes)
    total_leaves = len(graph.tensor_specs) + len(graph.static_values)
    if len(axes_leaves) != total_leaves:
        raise TransformError(
            "vectorize: the axes pytree has "
            f"{len(axes_leaves)} leaves but the graph has {total_leaves} "
            "input leaves; axes must match the graph's input structure "
            "(int 0 maps a tensor input's leading axis, None leaves it "
            "unmapped)"
        )
    static_positions = {record.index for record in graph.static_values}
    tensor_axes = []
    for index, entry in enumerate(axes_leaves):
        if index in static_positions:
            if entry is not None:
                raise TransformError(
                    f"vectorize: static input at leaf {index} must map to "
                    f"None, got {entry!r} (static values specialize the "
                    "graph and cannot be batched)"
                )
            continue
        if entry is None or (
            isinstance(entry, int) and not isinstance(entry, bool) and entry == 0
        ):
            tensor_axes.append(entry)
        else:
            raise TransformError(
                f"vectorize: mapped axis for tensor input leaf {index} must "
                f"be 0 (the leading axis) or None in v1, got {entry!r}"
            )
    return tuple(tensor_axes)


# --- active vectorize-pass depth (level-distinct batch dim names) -----------


#: Active vectorize-pass depth, contextvars-backed (the same pattern as the
#: builder stack in `etl/trace/builder.py`). `_batched_input_specs` names each
#: pass's shared batch dim from the CURRENT depth: depth 0 → `"batch"`,
#: depth 1 → `"batch_1"`, depth 2 → `"batch_2"`, ... — nested vmap levels get
#: level-distinct dim NAMES. The numpy interpreter binds symbolic dims BY
#: NAME, so same-named dims from different nesting levels would collide at
#: run time (equal extents accidentally pass; unequal extents raise a
#: misleading ShapeError on a valid graph). The depth is incremented only
#: around the invocation of a nested `TransformCallable` (vmap composition —
#: see `vmap._vmap_fn`); top-level `vectorize`/`vmap` calls always run at
#: depth 0 and keep the name `"batch"`.
_batch_depth: ContextVar[int] = ContextVar(
    "etl_transforms_batch_depth", default=0
)


class with_batch_depth:
    """Context manager running a nested transform composition one level deeper.

    `_vmap_fn` invokes a nested `TransformCallable` (e.g. `vmap(vmap(f))`)
    under this context: the inner pass then names its batch dim `batch_1`,
    while the outer pass (vectorizing the inner graph after the call returns)
    runs at the parent depth and names its dim `batch`. Nestable (LIFO) —
    depth-3+ recursion composes through the same path; the saved parent depth
    is restored on exit even on error.
    """

    def __init__(self) -> None:
        self._parent: Optional[int] = None

    def __enter__(self) -> int:
        self._parent = _batch_depth.get()
        depth = self._parent + 1
        _batch_depth.set(depth)
        return depth

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _batch_depth.set(self._parent if self._parent is not None else 0)
        return False


def _batch_dim() -> core.Dim:
    """The fresh shared batch `Dim` of the CURRENT vectorize pass.

    Named from the active pass depth (`"batch"` at depth 0, `"batch_1"` at
    depth 1, ...) so nested passes get level-distinct names — the numpy
    interpreter binds symbolic dims by name, so a same-named dim from a
    different nesting level would collide at run time.
    """
    depth = _batch_depth.get()
    return core.Dim("batch" if depth <= 0 else f"batch_{depth}")


def _batched_input_specs(
    graph: Graph,
    tensor_axes: Tuple[Optional[int], ...],
    batch_dim: Optional[core.Dim] = None,
) -> Tuple[Tuple, Tuple[MappedAxes, ...]]:
    """New flat tensor specs + per-input `MappedAxes`.

    Mapped inputs all share ONE fresh symbolic `core.Dim` (named from the
    active pass depth — `_batch_dim`; `"batch"` at depth 0, `"batch_1"` at
    depth 1, ...) as their leading dim: a single vectorize pass introduces
    exactly one batch axis, so every mapped input carries the same Dim object
    — object identity IS the logical axis. Same-level dims therefore compare
    equal in shape inference (no `max(batch, batch_1)` DimExprs in batched
    result types), and the runtime's name-keyed dim binding rejects mismatched
    batch extents WITHIN a level with a clear ShapeError. Level-distinct
    names keep nested vmap levels from colliding at run time (unequal extents
    run; equal extents keep working). Unmapped specs are reused unchanged.

    `batch_dim` may be supplied by the caller (`vectorize_graph` threads the
    SAME object into the output-aux remap so remapped static leaves share the
    batch Dim identity with the specs); when omitted it is created here.
    """
    if batch_dim is None:
        batch_dim = _batch_dim()
    new_specs = []
    mapped_axes = []
    for spec, entry in zip(graph.tensor_specs, tensor_axes):
        if entry is None:
            new_specs.append(spec)
            mapped_axes.append(UNMAPPED)
            continue
        new_specs.append(
            core.TensorSpec(
                shape=(batch_dim, *spec.shape),
                dtype=spec.dtype,
                device=spec.device,
                name=spec.name,
            )
        )
        mapped_axes.append(MappedAxes((0,)))
    return tuple(new_specs), tuple(mapped_axes)


def _remap_output_aux(
    graph: Graph, batch_dim: core.Dim
) -> Tuple["core.TreeSpec", Tuple[StaticValue, ...]]:
    """Rebuild the output tree + output static records at the vectorize boundary.

    `vectorize_graph` prepends a leading batch dim to mapped tensor outputs,
    which goes stale for registered pytree nodes whose children include
    STATIC leaves (e.g. a future sparse-tensor node: children = [tensor
    leaves..., dense_shape ints, dtype, format] — the sparse value's
    dense_shape must gain the batch dim as a leading symbolic `Dim`). For
    every output-tree container node with a registered remap fn
    (`_lookup_batched_aux_remap`), the fn receives the node spec, its direct
    static-leaf child values (in child order), and `batch_dim` — the same
    object the mapped tensor specs carry — and returns the REMAPPED static
    leaf values (count may change). INPUT-side static values are never
    remapped (the batch lives in the flat tensor-leaf specs only).

    Algorithm (two mutable leaf counters — an OLD-tree counter classifying
    leaves against the recorded static indices, since registered nodes may
    change leaf counts and old/new indices diverge, and a NEW-tree counter
    assigning fresh sequential indices to the rebuilt records):

    1. Leaf (`num_leaves == 1`, no children): classified by its OLD index —
       recorded static leaves keep their value/kind VERBATIM, only
       index/path recomputed; the leaf spec itself is unchanged. Empty
       containers (`num_leaves == 0`) consume nothing and pass through
       unchanged.
    2. Container with a registered fn: direct static-leaf children are
       collected (in child order) for the fn call; direct tensor-leaf
       children keep their specs verbatim at their original positions;
       non-leaf children recurse normally (they pass through — static leaves
       inside them keep verbatim values with recomputed indices, and are NOT
       passed to the fn). The rebuilt children = all non-static children in
       original order + one fresh plain-leaf `core.TreeSpec` (`type(None)` —
       the canonical normalized leaf; a `type=None` FIELD would classify as a
       TENSOR position in the pipeline) per returned static value; the node's
       `type`/`context`/`node_data` are preserved. One `StaticValue` record is
       emitted per new static value at its index in the NEW children tuple.
    3. Unregistered container: recurse into children; `type`/`context`/
       `node_data` preserved. Path keys mirror `_iter_leaf_paths` (dict
       nodes use the recorded sorted key from `node_data`, all other
       container kinds use positional indices).

    When NO registered container node is found, the ORIGINAL
    `(output_tree, output_static_values)` objects are returned — byte-
    identical old behavior (zero regression).
    """
    old_static = {record.index for record in graph.output_static_values}
    old_by_index = {record.index: record for record in graph.output_static_values}
    found = False

    def _child_key(spec, index):
        # Mirror `trace._iter_leaf_paths`: dict nodes use the recorded sorted
        # key from `node_data`; all other container kinds use positional
        # indices.
        if isinstance(spec.type, type) and issubclass(spec.type, dict):
            return spec.node_data[index]
        return index

    def walk(spec, prefix, old_counter, new_counter):
        """Recursive rebuild; returns ``(new_spec, records)``."""
        nonlocal found
        if spec.num_leaves == 0:
            # Empty container: consumes nothing, passes through unchanged.
            return spec, []
        if not spec.children:
            # Leaf: classify by its OLD index. Static leaves keep their
            # value/kind verbatim (only index/path recomputed); the leaf spec
            # itself is unchanged.
            old_idx = old_counter[0]
            old_counter[0] += 1
            new_idx = new_counter[0]
            new_counter[0] += 1
            if old_idx not in old_static:
                return spec, []
            old_record = old_by_index[old_idx]
            return spec, [
                StaticValue(
                    index=new_idx,
                    path=prefix,
                    value=old_record.value,
                    kind=old_record.kind,
                )
            ]
        fn = _lookup_batched_aux_remap(spec)
        if fn is None:
            # Unregistered container: recurse into children.
            new_children = []
            records = []
            for index, child in enumerate(spec.children):
                rebuilt, child_records = walk(
                    child,
                    prefix + (_child_key(spec, index),),
                    old_counter,
                    new_counter,
                )
                new_children.append(rebuilt)
                records.extend(child_records)
            return (
                core.TreeSpec(
                    type=spec.type,
                    children=tuple(new_children),
                    context=spec.context,
                    node_data=spec.node_data,
                ),
                records,
            )
        # Registered container: direct static-leaf children are collected for
        # the fn; direct tensor-leaf children keep their specs verbatim;
        # container children recurse normally.
        found = True
        static_values = []
        new_children = []
        records = []
        for index, child in enumerate(spec.children):
            if not child.children:
                if child.num_leaves == 0:
                    # Empty container child: consumes nothing, passes through.
                    new_children.append(child)
                    continue
                # Direct leaf child of the registered node.
                old_idx = old_counter[0]
                old_counter[0] += 1
                if old_idx in old_static:
                    static_values.append(old_by_index[old_idx].value)
                else:
                    new_children.append(child)
                    new_counter[0] += 1  # tensor leaf: one new leaf
                continue
            rebuilt, child_records = walk(
                child,
                prefix + (_child_key(spec, index),),
                old_counter,
                new_counter,
            )
            new_children.append(rebuilt)
            records.extend(child_records)
        remapped = fn(spec, tuple(static_values), batch_dim)
        if isinstance(remapped, str):
            raise TransformError(
                "vectorize: the batched-aux remap fn registered for node "
                f"type {spec.type.__qualname__!r} must return a non-str "
                f"sequence of static leaf values, got a str"
            )
        try:
            remapped = tuple(remapped)
        except TypeError as exc:
            raise TransformError(
                "vectorize: the batched-aux remap fn registered for node "
                f"type {spec.type.__qualname__!r} must return a sequence of "
                f"static leaf values, got {remapped!r}"
            ) from exc
        for value in remapped:
            child_pos = len(new_children)
            records.append(
                StaticValue(
                    index=new_counter[0],
                    path=prefix + (child_pos,),
                    value=value,
                    kind=type(value).__qualname__,
                )
            )
            new_counter[0] += 1
            # Plain-leaf spec for the fresh static: `type(None)` (the
            # canonical normalized leaf — NOT a `type=None` field, which the
            # pipeline's `_is_tensor_leaf_spec` treats as a TENSOR marker,
            # grad/vjp output-tree convention).
            new_children.append(core.TreeSpec(type=type(None)))
        return (
            core.TreeSpec(
                type=spec.type,
                children=tuple(new_children),
                context=spec.context,
                node_data=spec.node_data,
            ),
            records,
        )

    old_counter = [0]
    new_counter = [0]
    new_tree, records = walk(graph.output_tree, (), old_counter, new_counter)
    if not found:
        return graph.output_tree, graph.output_static_values
    return new_tree, tuple(records)


def _entry_function(graph: Graph) -> ir.Function:
    """The graph's single function (v1 guard: one function, one block)."""
    if len(graph.module.functions) != 1:
        raise TransformError(
            "vectorize: graph modules with multiple functions are not "
            "supported in v1"
        )
    function = graph.module.functions[0]
    if len(function.region.blocks) != 1:
        raise TransformError(
            "vectorize: multi-block function regions are not supported in v1"
        )
    return function


def _rewrite_block(block, env, builder, remap, source_locations, graph) -> None:
    """Rewrite one basic block into the builder's current (new) block.

    Walks the ops in program order (topological for traced graphs), invoking
    each op's batching rule and splicing the emitted replacement ops into the
    new block. The `return` terminator's operands are mapped through `remap`
    and re-emitted via `set_terminator` (the terminator is last by
    construction — v1 guarantees nothing follows it). Ops carrying nested
    regions (`cond`/`while_loop`/`scan`) are not vectorizable in v1.
    """
    new_block = builder.current_block
    terminated = False
    for op in block.ops:
        if op.is_terminator:
            if op.name != "return":
                raise TransformError(
                    f"vectorize: unsupported terminator '{op.name}' (v1 "
                    "graphs terminate with 'return')"
                )
            new_operands = tuple(
                _rewritten_operand(value, remap) for value in op.operands
            )
            builder.set_terminator(
                new_block, "return", operands=new_operands, location=op.location
            )
            terminated = True
            break  # v1: the terminator is the last op — nothing follows it.
        if op.regions:
            raise TransformError(
                f"vectorize: cannot batch op '{op.name}': region-bearing "
                "control-flow ops (cond/while_loop/scan) are not vectorizable "
                "in v1"
            )
        _rewrite_op(op, env, builder, remap, source_locations, graph)
    if not terminated:
        raise TransformError(
            "vectorize: the graph function has no 'return' terminator"
        )


def _rewrite_op(op: ir.Op, env, builder, remap, source_locations, graph):
    """Dispatch one op to its batching rule and record the results.

    The transform builder is active (pushed by `vectorize_graph`), so rules
    build replacement ops through ordinary `etl.ops.*` functions. Operands
    are passed to the rule as the REWRITTEN `ir.Value`s of the new module
    (aligned with `op.operands`; rules read the original values from
    `op.operands` when they need them) — replacement ops must be built over
    new-module values so shape/dtype inference sees the batched types.
    Afterwards the replacement ops inherit the original op's source location,
    the rule's results are recorded in the env/remap, and the new value ids
    are added to the new graph's `source_locations`.

    Returns the rule's `(new_values, new_axes)` pair.
    """
    rule = _resolve_batching_rule(op)
    rewritten_operands = tuple(
        _rewritten_operand(operand, remap) for operand in op.operands
    )
    operand_axes = tuple(_operand_axes(op, value, env) for value in rewritten_operands)
    new_block = builder.current_block
    before = {emitted.id for emitted in new_block.ops}
    new_values, new_axes = rule(op, rewritten_operands, operand_axes)
    new_values, new_axes = _validate_rule_result(op, new_values, new_axes)
    # The rule emitted into the active builder via etl.ops.*, whose location
    # capture points at the rule code — overwrite with the ORIGINAL op's
    # source location (None when the original op has none).
    for emitted in new_block.ops:
        if emitted.id not in before:
            emitted.location = op.location
    env.update(new_values, new_axes)
    for old_result, new_value in zip(op.results, new_values):
        remap[old_result.id] = new_value
        location = graph.source_locations.get(old_result.id)
        if location is None:
            location = op.location
        if location is not None:
            source_locations[new_value.id] = location
    return new_values, new_axes


def _operand_axes(op: ir.Op, value: ir.Value, env: ValueEnv) -> MappedAxes:
    """The `MappedAxes` of one rewritten operand (strict: must be recorded)."""
    if value not in env:
        raise TransformError(
            f"vectorize: internal error — rewritten operand %{value.id} of "
            f"op '{op.name}' has no recorded batching metadata"
        )
    return env.get(value)


def _rewritten_operand(value: ir.Value, remap: Dict[int, ir.Value]) -> ir.Value:
    """The rewritten value replacing `value` (mapped through `remap`)."""
    new_value = remap.get(value.id)
    if new_value is None:
        raise TransformError(
            f"vectorize: internal error — value %{value.id} has no rewritten "
            "replacement; every value must be produced by an already-rewritten "
            "op or be a function block argument"
        )
    return new_value


def _validate_rule_result(
    op: ir.Op, new_values, new_axes
) -> Tuple[Tuple[ir.Value, ...], Tuple[MappedAxes, ...]]:
    """Validate a rule's return value against the binding signature.

    Rules return `(new_values, new_axes)` — sequences aligned with
    `op.results`; `new_values` are `ir.Value`s and `new_axes` are `MappedAxes`
    whose entries must be leading-contiguous (v1). Rule bugs raise
    `TransformError` naming the op — never silently repaired.
    """
    try:
        new_values = tuple(new_values)
        new_axes = tuple(new_axes)
    except TypeError as exc:
        raise TransformError(
            f"vectorize: batching rule for op '{op.name}' must return a "
            f"(new_values, new_axes) pair of tuples; got {new_values!r}, "
            f"{new_axes!r}"
        ) from exc
    if len(new_values) != len(op.results) or len(new_axes) != len(op.results):
        raise TransformError(
            f"vectorize: batching rule for op '{op.name}' returned "
            f"{len(new_values)} values / {len(new_axes)} axes for "
            f"{len(op.results)} results — both must align with op.results"
        )
    for value in new_values:
        if not isinstance(value, ir.Value):
            raise TransformError(
                f"vectorize: batching rule for op '{op.name}' returned a "
                f"non-Value result {value!r}"
            )
    for mapped in new_axes:
        if not isinstance(mapped, MappedAxes):
            raise TransformError(
                f"vectorize: batching rule for op '{op.name}' returned "
                f"non-MappedAxes metadata {mapped!r}"
            )
        if mapped.axes != tuple(range(mapped.count)):
            raise TransformError(
                f"vectorize: batching rule for op '{op.name}' returned "
                f"MappedAxes({mapped.axes!r}) — mapped axes must be a "
                "contiguous tuple of leading indices ((0,), (0, 1), ...)"
            )
    return new_values, new_axes
