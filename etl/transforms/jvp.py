"""Public `jvp` — forward-mode Jacobian-vector products.

`jvp` is graph→graph: the result is a `Graph` of ordinary ops mapping
(primal inputs, tangent inputs) → (primal outputs, tangent outputs), built by
the forward sweep in `autodiff.py` using per-op JVP rules. Semantics are
binding — see `./CONTEXT.md` "AD semantics".
"""

from __future__ import annotations

from etl.trace import Graph
from etl.transforms._wrappers import TransformCallable


def jvp(fn_or_graph, tangents):
    """Forward-mode differentiation: push tangents through the computation.

    Args:
        fn_or_graph: a `Graph`, or a callable/`Defn` (see Returns).
        tangents: a pytree of `TensorSpec` (or `None` = zero tangent) matching
            the **input** structure; each tangent's shape/dtype must match its
            primal spec. Tangents are runtime tensors, so they become explicit
            inputs of the result graph.

    Returns:
        * `Graph` when given a `Graph`: inputs = primal inputs followed by
          tangent inputs (flattened order); outputs = `(primal_outputs,
          tangent_outputs)` — a 2-tuple of trees.
        * `TransformCallable` when given a callable/`Defn`:
          `jvp(f, tangents)(*specs)` traces `f` exactly once and returns that
          same graph (the callable's specs describe the primal inputs only).
          Concrete tensors raise `TraceError` (transforms never execute).

    Ops without a JVP rule (`runtime_call`, collectives, custom blocks without
    a registered rule) raise `core.TransformError` naming the op/block —
    never a silent fallback. `stop_gradient` yields a zero tangent.
    """
    if isinstance(fn_or_graph, Graph):
        return _jvp_graph(fn_or_graph, tangents)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__etl_defn__", False):
        return TransformCallable(
            build=lambda *args: _jvp_fn(fn_or_graph, args, tangents),
            kind="jvp",
        )

    raise TypeError("jvp expects an etl.Graph, a callable, or an etl.Defn")


def _jvp_graph(graph: Graph, tangents) -> Graph:
    """Validate/normalize the tangent spec tree against the graph inputs, run
    the forward sweep, and build the result graph (primal+tangent inputs,
    (primal, tangent) outputs) (stub)."""
    raise NotImplementedError(
        "_jvp_graph: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _jvp_fn(fn, args, tangents) -> Graph:
    """`jvp(f, tangents)(*specs)` — trace `fn` once via `etl.trace(fn, *args)`,
    then apply `_jvp_graph` with the stored tangents (stub)."""
    raise NotImplementedError(
        "_jvp_fn: implementation phase; see etl/transforms/CONTEXT.md"
    )
