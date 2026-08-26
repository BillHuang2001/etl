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

from typing import Callable, Dict, Optional, Tuple

from etl import ir
from etl.core import TransformError
from etl.trace import Graph
from etl.transforms._metadata import MappedAxes

# Binding rule signature (see ./CONTEXT.md):
#   rule(op, operands, axes) -> (new_values, new_axes)
#   * op:          the ir.Op being vectorized
#   * operands:    tuple of ir.Value — the op's original operand values
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


def vectorize_graph(graph: Graph, axes) -> Graph:
    """Core vectorize algorithm: rewrite a traced graph with batched inputs.

    Walks the graph's function blocks in topological order, rewrites each op
    via its batching rule, seeds input metadata from the normalized `axes`
    structure, and builds a NEW `Graph` (new module/function — the input graph
    is never mutated) whose mapped inputs/outputs carry an extra leading
    dimension. Mapped input specs gain a fresh symbolic `Dim` (named `batch`,
    `batch_1`, ...) as the leading dim; `output_tree` and `static_values` are
    preserved. Region-bearing control-flow ops (`cond`/`while_loop`/`scan`)
    are not vectorizable in v1 and raise `TransformError`.
    """
    raise NotImplementedError(
        "vectorize_graph: core vectorize algorithm (implementation phase); "
        "see etl/transforms/CONTEXT.md"
    )


def _rewrite_block(block, env, builder) -> None:
    """Rewrite one basic block: seed block-arg metadata, walk ops in order,
    invoke rules, splice replacement ops into the builder (stub)."""
    raise NotImplementedError(
        "_rewrite_block: implementation phase; see etl/transforms/CONTEXT.md"
    )


def _rewrite_op(op: ir.Op, env, builder) -> Tuple[Tuple[ir.Value, ...], Tuple[MappedAxes, ...]]:
    """Dispatch one op to its batching rule (with the transform builder active)
    and record the resulting value metadata in `env` (stub)."""
    raise NotImplementedError(
        "_rewrite_op: implementation phase; see etl/transforms/CONTEXT.md"
    )
