"""Public `vjp` — reverse-mode vector-Jacobian products.

`vjp` is graph→graph: the result is a `Graph` of ordinary ops mapping
(primal inputs, cotangent inputs) → (primal outputs, input cotangents), built
by the backward sweep in `autodiff.py`. It shares VJP rules with `grad`.
Semantics are binding — see `./CONTEXT.md` "AD semantics".
"""

from __future__ import annotations

from etl.trace import Graph
from etl.transforms._wrappers import TransformCallable


def vjp(fn_or_graph, cotangents=None):
    """Reverse-mode differentiation: pull cotangents back to the inputs.

    Args:
        fn_or_graph: a `Graph`, or a callable/`Defn` (see Returns).
        cotangents: a pytree of `TensorSpec` (or `None`) matching the
            **output** structure. `None` default = scalar-one cotangent
            `TensorSpec((), out_dtype)`, valid only when the graph has exactly
            one scalar output. Cotangents are runtime tensors, so they become
            explicit inputs of the result graph.

    Returns:
        * `Graph` when given a `Graph`: inputs = primal inputs followed by
          cotangent inputs (flattened order); outputs = `(primal_outputs,
          input_cotangents)` — input cotangents are one tensor per tensor
          input (static inputs receive none).
        * `TransformCallable` when given a callable/`Defn`:
          `vjp(f, cotangents)(*specs)` traces `f` exactly once and returns
          that same graph (the callable's specs describe the primal inputs
          only). Concrete tensors raise `TraceError` (transforms never
          execute).

    Ops without a VJP rule (`runtime_call`, collectives, custom blocks without
    a registered rule) raise `core.TransformError` naming the op/block —
    never a silent fallback. `stop_gradient` yields a zero gradient.
    """
    if isinstance(fn_or_graph, Graph):
        return _vjp_graph(fn_or_graph, cotangents)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__etl_defn__", False):
        return TransformCallable(
            build=lambda *args: _vjp_fn(fn_or_graph, args, cotangents),
            kind="vjp",
        )

    raise TypeError("vjp expects an etl.Graph, a callable, or an etl.Defn")


def _vjp_graph(graph: Graph, cotangents) -> Graph:
    """Validate/normalize the cotangent spec tree against the graph outputs
    (defaulting to a scalar-one cotangent for a single scalar output), run the
    backward sweep, and build the result graph (primal+cotangent inputs,
    (primal outputs, input cotangents)) (stub)."""
    raise NotImplementedError(
        "_vjp_graph: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _vjp_fn(fn, args, cotangents) -> Graph:
    """`vjp(f, cotangents)(*specs)` — trace `fn` once via `etl.trace(fn,
    *args)`, then apply `_vjp_graph` with the stored cotangents (stub)."""
    raise NotImplementedError(
        "_vjp_fn: implementation phase; see etl/transforms/CONTEXT.md"
    )
