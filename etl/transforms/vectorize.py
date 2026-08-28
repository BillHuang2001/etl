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
from etl.trace import Graph
from etl.transforms.batching import vectorize_graph


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


def _normalize_axes(graph: Graph, axes):
    """Validate/normalize `axes` against the graph's input structure.

    Checks: the axes pytree shape matches the graph input tree (tensor inputs
    may be int|None, static inputs must be None), int entries are in range for
    the spec's rank (rank is known at trace time), and v1 requires mapped
    entries to be 0 (leading) — otherwise `TransformError` (non-leading axes
    are deferred; see `./CONTEXT.md` v1 scope). A bare int/None is accepted
    only for graphs with exactly one tensor input (it applies to that input;
    static leaves map to None). Returns the normalized pytree (leaves
    None|0, structure identical to the input tree).
    """
    static_positions = {record.index for record in graph.static_values}
    tensor_positions = set(range(graph.input_specs.num_leaves)) - static_positions
    ranks = {}
    for position, spec in zip(sorted(tensor_positions), graph.tensor_specs):
        ranks[position] = len(spec.shape)

    if axes is None or _is_int(axes):
        if len(graph.tensor_specs) != 1:
            raise core.TransformError(
                "vectorize: a bare int/None axes spec applies only when the "
                "graph has exactly ONE tensor input; got "
                f"{len(graph.tensor_specs)} tensor inputs — pass a pytree "
                "matching the graph's input structure instead"
            )
        entries = [None] * graph.input_specs.num_leaves
        entries[sorted(tensor_positions)[0]] = axes
    else:
        axes_leaves, axes_spec = core.flatten(axes)
        mismatch = core.first_mismatch_path(
            axes_spec, graph.input_specs, leaf_vs_empty_is_mismatch=True
        )
        if mismatch is not None:
            raise core.TransformError(
                "vectorize: the axes pytree does not match the graph's input "
                f"structure — first mismatch at pytree path "
                f"{core.format_path(mismatch)}; axes must be a pytree with the "
                "same container structure as the inputs (0 maps a tensor "
                "input's leading axis, None leaves it unmapped)"
            )
        entries = list(axes_leaves)

    normalized = _normalize_axis_entries(
        entries, tensor_positions, ranks, "vectorize"
    )
    # Rebuild the normalized pytree with the input tree's structure. Leaves
    # are reconstructed positionally, so the leaf TYPES recorded by trace
    # (`_TensorSpecLeaf`, static value types) are irrelevant here.
    return core.unflatten(normalized, graph.input_specs)
