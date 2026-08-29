"""Public `vectorize` — the primitive graph transformation.

`vectorize(graph, axes)` rewrites a traced `Graph` so that inputs mapped by
`axes` carry an explicit leading batch dim and every op is rewritten by its
batching rule. The result is a single transformed `Graph` of ordinary ops —
the backend never needs to understand vectorization. `etl.vmap` is transparent
function-side sugar over this machinery (equivalence contract in
`./CONTEXT.md`).
"""

from __future__ import annotations

from etl import core
from etl.core import tree as _core_tree
from etl.trace import Graph
from etl.transforms.batching import vectorize_graph

# The registered-custom-node table of core's pytrees. `register_pytree_node`
# mutates this dict in place, so the alias stays live (same convention as
# `etl/trace/control_flow.py`).
_PYTREE_NODE_REGISTRY = _core_tree._PYTREE_NODE_REGISTRY


def vectorize(graph: Graph, axes) -> Graph:
    """Vectorize a traced graph via per-op batching rules.

    Args:
        graph: a traced `Graph` (the result of `etl.trace`). Passing a
            callable/`Defn` raises `TypeError` — for function-side sugar use
            `etl.vmap(fn, in_axes=axes)`.
        axes: the mapping of graph inputs to mapped axes: an `int`, `None`, or
            a pytree thereof matching the graph's input structure. An `int`
            maps the corresponding tensor input along that leading axis (v1:
            only axis 0); `None` leaves the input unmapped (mandatory for
            static inputs).

    Returns:
        A new `Graph` of ordinary ops whose mapped inputs/outputs carry an
        extra leading dimension (one fresh symbolic dim per pass, named from
        the active pass depth — `"batch"` at depth 0, `"batch_1"` at depth 1,
        ... — shared by all mapped inputs of the pass; see
        `batching._batched_input_specs`). Unsupported ops — no registered
        batching rule, control-flow regions in v1 — raise
        `core.TransformError` naming the op; there is never a Python-loop
        fallback. `output_tree` and `static_values` are preserved; the input
        graph is not mutated.

    Example:
        g = etl.trace(fn, TensorSpec((3, 5), etl.float32))
        batched = etl.vectorize(g, 0)          # input (3,5) -> (batch,3,5)
        batched = etl.vectorize(g, (0, None))  # first input mapped, second not
    """
    if not isinstance(graph, Graph):
        raise TypeError(
            "vectorize expects a traced etl.Graph (call etl.trace first). "
            "For function-side sugar use etl.vmap(fn, in_axes=axes)."
        )
    normalized = _normalize_axes(graph, axes)
    return vectorize_graph(graph, normalized)


# --- axes normalization (shared with vmap.py) -------------------------------


def _is_int(value) -> bool:
    """True for plain ints (bools excluded — numpy-style tag checks)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_axis_entries(entries, tensor_positions, ranks, what):
    """Validate/normalize raw axes leaves into ``None``/``0`` entries.

    Shared by `vectorize` (in_axes against a graph's input tree), `vmap`'s
    callable path (in_axes against the argument specs) and `vmap`'s output
    rearrangement (out_axes against the output tree).

    Args:
        entries: raw axes leaves, aligned with the structure's leaves.
        tensor_positions: set of flat leaf indices holding tensor
            inputs/outputs; every other leaf is static and must map to None.
        ranks: mapping tensor position -> rank for the range check, or None
            to skip it (out_axes: entry 0 on a rank-0 unmapped output is the
            legal size-one-axis-insertion case, not an error).
        what: error-message prefix ("vectorize" / "vmap").

    Returns:
        A list of normalized entries (``None`` or ``0``) aligned with
        `entries`.

    Raises:
        core.TransformError: non-None static leaf; invalid/non-zero/out-of-
            range mapped entry. v1 supports only {None, 0} — non-leading axes
            are deferred (see `./CONTEXT.md`).
    """
    normalized = [None] * len(entries)
    for index, entry in enumerate(entries):
        if index not in tensor_positions:
            if entry is not None:
                raise core.TransformError(
                    f"{what}: static leaf {index} must map to None, got "
                    f"{entry!r} (static values specialize the graph and "
                    "cannot be batched)"
                )
            continue
        if entry is None:
            continue
        if not _is_int(entry):
            raise core.TransformError(
                f"{what}: invalid axes entry {entry!r} at tensor leaf "
                f"{index}: tensor leaves map to 0 (the leading axis) or None"
            )
        rank = None if ranks is None else ranks.get(index)
        if rank is not None and not 0 <= entry < rank:
            raise core.TransformError(
                f"{what}: mapped axis {entry} is out of range for tensor "
                f"leaf {index} of rank {rank}"
            )
        if entry != 0:
            raise core.TransformError(
                f"{what}: mapped axis {entry} for tensor leaf {index} must "
                "be 0 (the leading axis) in v1 — non-leading axes are "
                "deferred (see etl/transforms/CONTEXT.md)"
            )
        normalized[index] = 0
    return normalized


def _registered_pytree_base(spec_type) -> type:
    """The first type in `spec_type`'s MRO registered in the core pytree
    registry, or None.

    Mirrors the MRO walk `core.tree._flatten_into` / `etl.trace.control_flow`
    use (registered base classes catch subclasses); a node whose `TreeSpec`
    type resolves here is a registered pytree node (e.g. a sparse-tensor
    node). Non-type spec entries (e.g. the `None` leaf type) return None.
    """
    if not isinstance(spec_type, type):
        return None
    for base in spec_type.__mro__:
        if base in _PYTREE_NODE_REGISTRY:
            return base
    return None


def _is_registered_spec(spec: "core.TreeSpec") -> bool:
    """True iff `spec` describes a registered pytree node (MRO walk)."""
    return _registered_pytree_base(spec.type) is not None


def _tree_has_registered_node(tree_spec: "core.TreeSpec") -> bool:
    """True iff `tree_spec` contains at least one registered pytree node
    (e.g. a sparse-tensor node) anywhere in its structure."""
    if _is_registered_spec(tree_spec):
        return True
    return any(_tree_has_registered_node(child) for child in tree_spec.children)


def _broadcast_registered_axes(
    axes_spec: "core.TreeSpec",
    tree_spec: "core.TreeSpec",
    axes_leaves,
    tensor_positions,
    what: str,
):
    """Walk `axes_spec` against the input/output tree `tree_spec`, expanding a
    LEAF axes entry at a registered pytree-node position across the node's
    leaves: every TENSOR leaf in the node's subtree gets the entry (0/None),
    every STATIC leaf gets None — exactly ONE axes leaf is consumed for the
    whole node. Non-registered positions follow the usual structure-match
    rules (the same checks `core.first_mismatch_path` applies).

    Returns ``(entries, mismatch_path)``: on success `entries` is the flat
    list of axes entries aligned with `tree_spec`'s leaves and
    `mismatch_path` is None; on a structure divergence at a non-registered
    position, `entries` is empty and `mismatch_path` is the first-mismatch
    pytree path (rendered by `core.format_path`). A CONTAINER axes entry at a
    registered-node position raises `core.TransformError` — v1 requires a
    single int/None mapping all the node's tensor leaves (per-leaf axes for a
    registered node are deferred).
    """
    entries = []
    axes_iter = iter(axes_leaves)
    mismatch = [None]
    leaf_pos = [0]  # flat leaf index in `tree_spec` (pre-order)

    def fill(tree_node: "core.TreeSpec", entry):
        """Emit one entry per leaf under a registered node's subtree:
        tensor leaves → `entry`, static leaves → None."""
        if tree_node.children:
            for child in tree_node.children:
                fill(child, entry)
            return
        if tree_node.num_leaves == 0:
            return  # empty container inside the node: no leaves
        entries.append(entry if leaf_pos[0] in tensor_positions else None)
        leaf_pos[0] += 1

    def walk(axes_node: "core.TreeSpec", tree_node: "core.TreeSpec", prefix):
        if _is_registered_spec(tree_node):
            # Registered node: a leaf axes entry broadcasts across the node's
            # leaves; a container entry is unsupported in v1.
            if axes_node.children or axes_node.num_leaves == 0:
                raise core.TransformError(
                    f"{what}: axes for a registered pytree node "
                    f"({tree_node.type.__name__}) must be a single int/None "
                    f"mapping all its tensor leaves in v1 — got a container "
                    f"at pytree path {core.format_path(prefix)}"
                )
            try:
                entry = next(axes_iter)
            except StopIteration:
                mismatch[0] = prefix
                return
            fill(tree_node, entry)
            return
        if not tree_node.children:
            if tree_node.num_leaves == 0:
                # Empty container node: axes must also be an empty container
                # (leaf vs empty is a mismatch, per the legacy
                # `leaf_vs_empty_is_mismatch=True` semantics).
                if axes_node.children or axes_node.num_leaves != 0:
                    mismatch[0] = prefix
                return
            # Plain leaf.
            if axes_node.children or axes_node.num_leaves == 0:
                mismatch[0] = prefix
                return
            try:
                entries.append(next(axes_iter))
            except StopIteration:
                mismatch[0] = prefix
                return
            leaf_pos[0] += 1
            return
        # Container node: axes must be a matching container (the same checks
        # `core.first_mismatch_path` applies — type, node_data, child count).
        if not axes_node.children or axes_node.num_leaves == 0:
            mismatch[0] = prefix
            return
        if (
            axes_node.type != tree_node.type
            or axes_node.node_data != tree_node.node_data
            or len(axes_node.children) != len(tree_node.children)
        ):
            mismatch[0] = prefix
            return
        keys = _core_tree._dict_keys(axes_node)
        for index, (child_axes, child_tree) in enumerate(
            zip(axes_node.children, tree_node.children)
        ):
            key = keys[index] if keys is not None else index
            walk(child_axes, child_tree, prefix + (key,))
            if mismatch[0] is not None:
                return

    walk(axes_spec, tree_spec, ())
    if mismatch[0] is not None:
        return [], mismatch[0]
    try:
        next(axes_iter)
    except StopIteration:
        pass
    else:
        # Unconsumed axes leaves: the axes pytree has more leaves than the
        # tree (only reachable through registered-node expansion, where one
        # axes leaf covers many tree leaves — a count check at a container
        # node would have caught a plain leaf-count overflow).
        return [], ()
    return entries, None


def _normalize_axes(graph: Graph, axes):
    """Validate/normalize `axes` against the graph's input structure.

    Checks: the axes pytree shape matches the graph input tree (tensor inputs
    may be int|None, static inputs must be None), int entries are in range for
    the spec's rank (rank is known at trace time), and v1 requires mapped
    entries to be 0 (leading) — otherwise `TransformError` (non-leading axes
    are deferred; see `./CONTEXT.md` v1 scope). A bare int/None is accepted
    only for graphs with exactly one tensor input, or for graphs whose input
    tree CONTAINS at least one registered pytree node (e.g. a sparse-tensor
    node — the bare entry then applies to ALL tensor leaves, statics map to
    None; per-leaf axes for a registered node are unsupported in v1). Returns
    the normalized axes: leaves None|0, structure identical to the input tree
    (a flat entry list when registered nodes are present — the registered
    node cannot be rebuilt from axes entries, and `vectorize_graph` only ever
    flattens the normalized axes again).
    """
    static_positions = {record.index for record in graph.static_values}
    tensor_positions = set(range(graph.input_specs.num_leaves)) - static_positions
    ranks = {}
    for position, spec in zip(sorted(tensor_positions), graph.tensor_specs):
        ranks[position] = len(spec.shape)
    has_registered = _tree_has_registered_node(graph.input_specs)

    if axes is None or _is_int(axes):
        if len(graph.tensor_specs) != 1 and not has_registered:
            raise core.TransformError(
                "vectorize: a bare int/None axes spec applies only when the "
                "graph has exactly ONE tensor input; got "
                f"{len(graph.tensor_specs)} tensor inputs — pass a pytree "
                "matching the graph's input structure instead"
            )
        entries = [
            axes if index in tensor_positions else None
            for index in range(graph.input_specs.num_leaves)
        ]
    else:
        axes_leaves, axes_spec = core.flatten(axes)
        if has_registered:
            entries, mismatch = _broadcast_registered_axes(
                axes_spec, graph.input_specs, axes_leaves, tensor_positions,
                "vectorize",
            )
        else:
            mismatch = core.first_mismatch_path(
                axes_spec, graph.input_specs, leaf_vs_empty_is_mismatch=True
            )
            entries = list(axes_leaves) if mismatch is None else None
        if mismatch is not None:
            raise core.TransformError(
                "vectorize: the axes pytree does not match the graph's input "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; axes must be a pytree with the "
                "same container structure as the inputs (0 maps a tensor "
                "input's leading axis, None leaves it unmapped)"
            )

    normalized = _normalize_axis_entries(
        entries, tensor_positions, ranks, "vectorize"
    )
    if has_registered:
        # Flat entries: rebuilding the normalized pytree through the input
        # tree would try to reconstruct the registered node from axes entries
        # (e.g. a SparseTensor from ints) — and `vectorize_graph` only ever
        # flattens the normalized axes again.
        return normalized
    # Rebuild the normalized pytree with the input tree's structure. Leaves
    # are reconstructed positionally, so the leaf TYPES recorded by trace
    # (`_TensorSpecLeaf`, static value types) are irrelevant here.
    return core.unflatten(normalized, graph.input_specs)
