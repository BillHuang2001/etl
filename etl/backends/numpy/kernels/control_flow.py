"""Control-flow kernels: if / while / call — recursive region execution.

Implemented now:
- if: selects a branch from the 0-d bool predicate (operand 0) and runs that
  branch region; ALL operands (predicate included) bind to the region entry
  block args per the binding REGION CONVENTIONS in
  ``etl/trace/control_flow.py``.
- while: interprets the loop by repeatedly running the condition region
  (regions[0]; must yield ONE 0-d bool) and the body region (regions[1]; must
  yield len(operands) next-carried values) — genuinely dynamic runtime
  control flow.
- call: runs another function of the module (attr ``callee``).

``scan`` needs NO kernel: it is not an IR op — ``etl.scan`` pre-lowers into
``while`` + ``gather`` + ``scatter`` at trace time (binding, parent
CONTEXT.md). The ``return`` terminator is special-cased by the interpreter
loop and is NOT dispatched — never register it.

Design notes (binding, parent CONTEXT.md):
- Genuinely dynamic runtime control flow: the interpreter evaluates the
  condition/branch regions per execution — the graph is NOT specialized per
  iteration and no region is traced at run time.
- Region execution reuses the same interpreter loop over block op order with
  an inner value environment (region block args bound per iteration); shape
  checks use ops-level inference with concrete dim bindings
  (``ctx.bindings`` — extended positionally by ``ctx.run_region``).
- No iteration limits: termination is the program's own business (the
  backend does not invent limits).
- Results are NOT validated against the op's declared types here: the
  interpreter's dispatch loop validates every kernel result (count, exact
  dtype, evaluated symbolic shape) after the kernel returns.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]


def _pred_bool(ctx: Any, op: Any, tensor: core.Tensor, where: str) -> bool:
    """Validate a 0-d bool tensor and return its Python bool value.

    Guards against malformed graphs reaching the interpreter (e.g. a
    hand-built IR where an ``if`` predicate is not 0-d bool): dtype must be
    ``core.bool_`` and rank 0 — else ``core.BackendError`` naming the op,
    never a silent coercion.
    """
    array = tensor.numpy()
    if tensor.dtype != core.bool_ or array.ndim != 0:
        raise core.BackendError(
            f"kernel for op '{op.name}': {where} must be a 0-d bool tensor, "
            f"got dtype {tensor.dtype} with shape {tuple(tensor.shape)}"
        )
    return bool(np.asarray(array).item())


def _if(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``if``: run the selected branch region and yield its outputs.

    Region conventions (binding, ``etl/trace/control_flow.py``): operand 0 is
    the boolean predicate; ``op.regions[0]`` = "then", ``op.regions[1]`` =
    "else". EVERY region entry block has one arg per op operand INCLUDING the
    predicate — so ALL operands are passed to ``run_region``. The selected
    region's ``return`` terminator operands are the branch outputs (the
    interpreter validates count/dtype/shape against ``op.results``).
    """
    choice = _pred_bool(ctx, op, operands[0], "the predicate (operand 0)")
    region = op.regions[0] if choice else op.regions[1]
    return tuple(ctx.run_region(region, list(operands)))


def _while(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``while``: interpret the loop over the carried values.

    Region conventions (binding, ``etl/trace/control_flow.py``): operands are
    the n initial loop-carried values; ``op.regions[0]`` = condition (returns
    ONE 0-d bool), ``op.regions[1]`` = body (returns n next-carried values).
    Each region entry block has n args bound to the carried values. The loop
    runs until the condition region yields ``False``; the final carried
    values are the op's results.
    """
    carried = list(operands)
    while True:
        cond_results = ctx.run_region(op.regions[0], list(carried))
        if len(cond_results) != 1:
            raise core.BackendError(
                f"kernel for op 'while': condition region produced "
                f"{len(cond_results)} result(s), expected exactly 1 "
                "(a 0-d bool)"
            )
        if not _pred_bool(ctx, op, cond_results[0], "the condition region result"):
            break
        body_results = ctx.run_region(op.regions[1], list(carried))
        if len(body_results) != len(operands):
            raise core.BackendError(
                f"kernel for op 'while': body region produced "
                f"{len(body_results)} result(s), expected {len(operands)} "
                "next-carried value(s)"
            )
        carried = list(body_results)
    return tuple(carried)


def _call(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``call``: run another function of the module (attr ``callee``).

    The callee's region entry block is bound to the call operands; its
    ``return`` terminator operands are the call's outputs (the interpreter
    validates count/dtype/shape against ``op.results``). An unknown callee
    is a malformed module — ``core.BackendError`` naming the callee, never a
    silent skip.
    """
    callee = op.attributes["callee"]
    try:
        function = ctx.module.get_function(callee)
    except KeyError:
        raise core.BackendError(
            f"kernel for op 'call': module '{ctx.module.name}' has no "
            f"function named '{callee}'"
        ) from None
    return tuple(ctx.run_region(function.region, list(operands)))


def register_kernels(table: dict) -> None:
    """Register this module's control-flow kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``. Control-flow
    kernels receive ``ctx`` (interpreter state) so they can run nested region
    blocks through the same dispatch loop.
    """
    table["if"] = _if
    table["while"] = _while
    table["call"] = _call
