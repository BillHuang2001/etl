"""Public `vjp` — reverse-mode vector-Jacobian products.

`vjp` is graph→graph: the result is a `Graph` of ordinary ops mapping
(primal inputs, cotangent inputs) → (primal outputs, input cotangents), built
by the backward sweep in `autodiff.py`. It shares VJP rules with `grad`.
Semantics are binding — see `./CONTEXT.md` "AD semantics".
"""

from __future__ import annotations

from etl import core
from etl.trace import Graph, trace
from etl.transforms._wrappers import TransformCallable
from etl.transforms.autodiff import reverse_sweep
from etl.transforms.grad import _normalize_extra_pytree, _return_values


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
    (the `None` default becomes an explicit scalar-one cotangent spec, valid
    only for a single scalar output), run the backward sweep, and return the
    sweep graph as-is — it already has the required input/output structure:
    inputs = 2-tuple (original input tree, flat cotangent tuple tree),
    outputs = 2-tuple (primal outputs with the original output tree, flat
    input cotangents aligned with the flattened tensor inputs)."""
    outputs = _return_values(graph)
    if cotangents is None:
        # Default: scalar-one cotangent as an explicit scalar input — requires
        # exactly one tensor output, a scalar.
        if len(outputs) != 1:
            raise core.ShapeError(
                f"vjp: cotangents=None requires exactly one tensor output; "
                f"the graph returns {len(outputs)}"
            )
        output_type = outputs[0].type
        if tuple(output_type.shape) != ():
            raise core.ShapeError(
                f"vjp: cotangents=None requires a scalar output (shape ()), "
                f"got shape {tuple(output_type.shape)}"
            )
        return reverse_sweep(graph, (core.TensorSpec((), output_type.dtype),))

    # Explicit cotangent pytree: validate against the output tree (tensor
    # output positions take TensorSpec/None; static output positions must be
    # None / mirror the recorded static value; a None entry at a tensor
    # output seeds the sweep's in-graph scalar-one, so that output must be
    # scalar).
    output_specs = tuple(
        core.TensorSpec(shape=tuple(value.type.shape), dtype=value.type.dtype)
        for value in outputs
    )
    flat_cotangents = _normalize_extra_pytree(
        cotangents,
        graph.output_tree,
        graph.output_static_values,
        output_specs,
        transform="vjp",
        kind="cotangent",
        none_requires_scalar=True,
    )
    return reverse_sweep(graph, flat_cotangents)


def _vjp_fn(fn, args, cotangents) -> Graph:
    """`vjp(f, cotangents)(*specs)` — trace `fn` once via `etl.trace(fn,
    *args)`, then apply `_vjp_graph` with the stored cotangents."""
    return _vjp_graph(trace(fn, *args), cotangents)
