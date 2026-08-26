"""Public `vectorize` — the primitive graph transformation.

`vectorize(graph, axes)` rewrites a traced `Graph` so that inputs mapped by
`axes` carry an explicit leading batch dim and every op is rewritten by its
batching rule. The result is a single transformed `Graph` of ordinary ops —
the backend never needs to understand vectorization. `etl.vmap` is transparent
function-side sugar over this machinery (equivalence contract in
`./CONTEXT.md`).
"""

from __future__ import annotations

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
        extra leading dimension (fresh symbolic dim named `batch`, `batch_1`,
        ...). Unsupported ops — no registered batching rule, control-flow
        regions in v1 — raise `core.TransformError` naming the op; there is
        never a Python-loop fallback. `output_tree` and `static_values` are
        preserved; the input graph is not mutated.

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


def _normalize_axes(graph: Graph, axes):
    """Validate/normalize `axes` against the graph's input structure.

    Checks: the axes pytree shape matches the graph input tree (tensor inputs
    may be int|None, static inputs must be None), int entries are in range for
    the spec's rank (rank is known at trace time), and v1 requires mapped
    entries to be 0 (leading) — otherwise `TransformError` (non-leading axes
    are deferred; see `./CONTEXT.md` v1 scope). Returns the normalized pytree.
    """
    raise NotImplementedError(
        "_normalize_axes: implementation phase; see etl/transforms/CONTEXT.md"
    )
