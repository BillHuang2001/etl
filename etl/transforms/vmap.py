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

from etl.trace import Graph
from etl.transforms._wrappers import TransformCallable
from etl.transforms.vectorize import vectorize

_VMAP_CALLABLE_DOC = """\
`vmap(f)` applied to a callable/`Defn` returns a TransformCallable:
calling it with a structure of TensorSpecs (mapped inputs include the leading
batch dim) + static values returns the vectorized Graph, tracing `f` exactly
once. Equivalent to: trace `f` with the leading mapped dim stripped from each
mapped spec, then vectorize, then rearrange outputs per out_axes."""


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
    """Trace `fn` exactly once with unvectorized specs derived from `args`
    (strip the leading mapped dim from each mapped spec per `in_axes`), then
    vectorize and rearrange outputs per `out_axes` (stub)."""
    raise NotImplementedError(
        "_vmap_fn: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _derive_unvectorized_args(args, in_axes):
    """Given the batched specs/static values `args` and `in_axes`, return the
    underlying (unvectorized) spec structure for tracing the wrapped function:
    mapped entries drop their leading dim; unmapped entries pass through;
    static values stay static (stub)."""
    raise NotImplementedError(
        "_derive_unvectorized_args: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _rearrange_outputs(graph: Graph, out_axes) -> Graph:
    """Post-vectorize output-axis rearrangement, as ordinary ops added to the
    already-vectorized graph (never inside batching rules).

    Per output: `out_axes` = 0 keeps the mapped axis leading (if the output is
    unmapped, an explicitly requested size-one axis is inserted at 0 via a
    reshape); `out_axes` = None requires the output to be unmapped, otherwise
    `core.TransformError` (axis mismatch — a batch axis is never silently
    dropped). With `out_axes` = 0 for every mapped output, NO extra op is
    added and the result is IR-identical to plain `vectorize` (stub).
    """
    raise NotImplementedError(
        "_rearrange_outputs: implementation phase; see etl/transforms/CONTEXT.md"
    )
