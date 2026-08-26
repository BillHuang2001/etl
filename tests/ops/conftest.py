"""Shared helpers for the tests/ops suite.

The ``etl`` package (repo root, read-only sibling at ``etl/``) is fully
implemented; tests here assert the op contracts in ``etl/ops/CONTEXT.md`` and
``etl/CONTEXT.md`` by (1) tracing small functions into graphs, (2) inspecting
the built IR (op names, arities, inferred dtype/shape, effects, locations),
and (3) evaluating numerically with the default numpy backend.

Graph inspection facts (stable, used across the suite):

- ``graph.module.functions[0].region.blocks[0].ops`` — the built ops in order
  (block arguments come from ``.arguments``; the last op is ``etl.return``).
- ``Op`` fields: ``.name``, ``.operands``, ``.results`` (``Value`` objects with
  ``.type.dtype`` / ``.type.shape``), ``.attributes``, ``.effect`` (singular),
  ``.location`` (``Location`` with ``.file`` / ``.line``).
- ``etl.trace(fn, *specs)`` accepts plain functions and ``@etl.defn``
  functions; specs are ``TensorSpec`` entries and/or static Python values.
- ``etl.evaluate(fn, *args)`` accepts raw numpy arrays (all args must be
  concrete tensors — static values require ``etl.trace`` + ``etl.build`` +
  ``etl.run``, passing the static value to ``run``).
"""
import etl


def trace_fn(fn, *specs) -> "etl.Graph":
    """Trace ``fn`` into a Graph (accepts plain functions and ``Defn``)."""
    return etl.trace(fn, *specs)


def ops_of(graph, name=None):
    """All ops of the graph's main function, optionally filtered by op name."""
    block = graph.module.functions[0].region.blocks[0]
    ops = block.ops
    if name is not None:
        return [op for op in ops if op.name == name]
    return list(ops)


def run_numpy(fn, *args):
    """Evaluate ``fn`` (via ``etl.evaluate``) on numpy arrays.

    Returns a numpy ``ndarray`` for single outputs or a tuple of ndarrays for
    structured (tuple) outputs.
    """
    out = etl.evaluate(fn, *args)
    if isinstance(out, etl.Tensor):
        return out.numpy()
    return tuple(o.numpy() for o in out)
