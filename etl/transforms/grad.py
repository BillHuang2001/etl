"""Public `grad` — reverse-mode gradients via the shared VJP machinery.

`grad` is graph→graph: the result is a `Graph` of ordinary ops mapping the
same inputs to gradient tensors. It shares VJP rules with `vjp` (and the
backward sweep in `autodiff.py`). Semantics are binding — see `./CONTEXT.md`
"AD semantics".
"""

from __future__ import annotations

from etl.trace import Graph
from etl.transforms._wrappers import TransformCallable


def grad(fn_or_graph, argnums=None):
    """Reverse-mode gradients: inputs → gradients w.r.t. selected arguments.

    Args:
        fn_or_graph: a `Graph`, or a callable/`Defn` (see Returns).
        argnums: which inputs to differentiate: an `int`, or a tuple/list of
            ints indexing the flattened **tensor** inputs (static values are
            excluded from the numbering); `None` = all tensor inputs. Selected
            inputs must be floating/complex — `core.TransformError` otherwise.

    Graph requirements:
        exactly one output, a scalar tensor — otherwise `core.ShapeError`.

    Returns:
        * `Graph` when given a `Graph`: same inputs → one gradient tensor
          when `argnums` is an int, else a tuple of gradient tensors (one per
          selected input, in order).
        * `TransformCallable` when given a callable/`Defn`: `grad(f)(*specs)`
          traces `f` exactly once and returns that same gradient graph.
          Concrete tensors raise `TraceError` (transforms never execute).

    Ops without a VJP rule (`runtime_call`, collectives, custom blocks without
    a registered rule) raise `core.TransformError` naming the op/block —
    never a silent fallback. `stop_gradient` yields a zero gradient.
    """
    if isinstance(fn_or_graph, Graph):
        return _grad_graph(fn_or_graph, argnums)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__etl_defn__", False):
        return TransformCallable(
            build=lambda *args: _grad_fn(fn_or_graph, args, argnums),
            kind="grad",
        )

    raise TypeError("grad expects an etl.Graph, a callable, or an etl.Defn")


def _grad_graph(graph: Graph, argnums) -> Graph:
    """Validate the single-scalar-output requirement (`ShapeError`), normalize
    `argnums`, run the reverse sweep with a scalar-one cotangent of the output
    dtype, and return a graph producing only the selected input gradients
    (stub)."""
    raise NotImplementedError(
        "_grad_graph: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _grad_fn(fn, args, argnums) -> Graph:
    """`grad(f)(*specs)` — trace `fn` once via `etl.trace(fn, *args)`, then
    apply `_grad_graph` (stub)."""
    raise NotImplementedError(
        "_grad_fn: implementation phase; see etl/transforms/CONTEXT.md"
    )
