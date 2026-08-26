"""Builtin batching/JVP/VJP rules for the standard op set.

The op-name → rule-category tables below are the design record of how each
builtin op vectorizes and differentiates; registration happens at import
time. Rule bodies are implementation-phase stubs (`NotImplementedError`)
except the trivial zero rules. Ops deliberately absent from every table —
`runtime_call`, collectives, control-flow ops — raise `core.TransformError`
when transformed (no silent fallback; see `./CONTEXT.md`).

Rule signatures are binding: see `./CONTEXT.md` "Rule-call signatures".
While a rule runs, the transform machinery has pushed its `ir.Builder` onto
the trace builder stack — rules build replacement ops with ordinary
`etl.ops.*` functions.
"""

from __future__ import annotations

from etl.transforms.autodiff import (
    ZeroTangent,
    register_jvp_rule,
    register_vjp_rule,
)
from etl.transforms.batching import register_batching_rule

# --- op categories (design record) --------------------------------------

# Elementwise: result axes = union of operand mapped axes; an unmapped
# operand gets leading size-one dims inserted (reshape) so shapes align.
ELEMENTWISE_OPS = (
    "add", "subtract", "multiply", "divide", "power", "remainder",
    "maximum", "minimum", "abs", "negate", "square", "sqrt", "exp", "log",
    "log1p", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "gelu", "erf",
    "sign", "cast", "select", "broadcast", "stop_gradient",
)

# Batch like elementwise; AD: outputs are bool/int and cannot backpropagate,
# so their JVP/VJP rules yield ZeroTangent (not an error).
NONDIFFERENTIABLE_OUTPUT_OPS = (
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "argmax", "argmin",
)

# Reductions: shift the reduced axis by the number of leading mapped axes
# (sum/mean/prod keep both axes mapped; max/min follow the same shift).
REDUCTION_OPS = (
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod",
    "sum", "max", "min", "mean", "prod",
)

# Structural: per-op metadata rewriting — reshape splits/merges the mapped
# dims; transpose reorders axes (mapped axes stay leading; v1: only 0/None);
# slice shifts start/stop indices; pad inserts into the pad widths;
# concatenate concatenates per batch (or along a shifted axis); gather/scatter
# adjust index/axis arguments.
STRUCTURAL_OPS = ("reshape", "transpose", "slice", "pad", "concatenate")

# Dot/conv: batched matmul/convolution with the mapped dims broadcast across
# operands; AD via transposed dot/conv rules.
DOT_OPS = ("dot", "conv")

# Custom blocks: rules arrive through the `block:<name>` namespace registered
# by `etl/block` — nothing to do here.


# --- batching rule stubs ------------------------------------------------

def _pointwise_batching(op, operands, axes):
    """Default elementwise rule: for each operand missing a leading mapped
    axis, insert leading size-one dims (reshape) so all operands broadcast;
    rebuild the op; result axes = union of operand axes (stub)."""
    raise NotImplementedError("pointwise batching rule — implementation phase")


def _reduction_batching(op, operands, axes):
    """Reduction rule: shift the reduced axis attribute by the number of
    leading mapped axes; rebuild the op; result keeps the mapped axes (stub).
    """
    raise NotImplementedError("reduction batching rule — implementation phase")


def _reshape_batching(op, operands, axes):
    """Reshape rule: reconcile old/new shapes against the mapped dims (the
    batch dims must be preserved in the new shape); rebuild with adjusted
    shape attrs (stub)."""
    raise NotImplementedError("reshape batching rule — implementation phase")


def _transpose_batching(op, operands, axes):
    """Transpose rule: permutation applies to the unvectorized axes; mapped
    leading axes stay leading (v1: mapped entries are 0/None only) (stub)."""
    raise NotImplementedError("transpose batching rule — implementation phase")


def _structural_batching(op, operands, axes):
    """Dispatcher for slice/pad/concatenate (stub; per-op rules may be split
    out if this grows)."""
    raise NotImplementedError("structural batching rule — implementation phase")


def _dot_conv_batching(op, operands, axes):
    """Dot/conv rule: emit the batched matmul/convolution with mapped dims
    broadcast across operands; result mapped axes = union (stub)."""
    raise NotImplementedError("dot/conv batching rule — implementation phase")


def _gather_scatter_batching(op, operands, axes):
    """Gather/scatter rule: adjust the axis argument by the mapped-dim count
    of the data operand and update index metadata (stub)."""
    raise NotImplementedError("gather/scatter batching rule — implementation phase")


# --- JVP/VJP rule stubs ---------------------------------------------------

def _pointwise_jvp(op, tangents):
    """Per-elementwise-op linearization (sum/product rules, select on the
    primal mask, derivative table for math ops); primals read from
    `op.operands` (stub)."""
    raise NotImplementedError("pointwise JVP rule — implementation phase")


def _pointwise_vjp(op, cotangents, primals):
    """Per-elementwise-op pullback (product with the local partials, select on
    the primal mask); returns a tuple aligned with `op.operands` (stub)."""
    raise NotImplementedError("pointwise VJP rule — implementation phase")


def _reduction_jvp(op, tangents):
    """Reduction JVP: reduce the tangent over the same axes (stub)."""
    raise NotImplementedError("reduction JVP rule — implementation phase")


def _reduction_vjp(op, cotangents, primals):
    """Reduction VJP: broadcast the cotangent back to the input shape (divide
    by the reduction count for mean/prod per the op kind) (stub)."""
    raise NotImplementedError("reduction VJP rule — implementation phase")


def _structural_jvp(op, tangents):
    """Structural JVP dispatcher (reshape/transpose/slice/pad/concatenate/
    gather/scatter applied to tangents) (stub)."""
    raise NotImplementedError("structural JVP rule — implementation phase")


def _structural_vjp(op, cotangents, primals):
    """Structural VJP dispatcher (inverse structural op applied to the
    cotangent) (stub)."""
    raise NotImplementedError("structural VJP rule — implementation phase")


def _dot_conv_jvp(op, tangents):
    """Dot/conv JVP: sum over the two product terms (stub)."""
    raise NotImplementedError("dot/conv JVP rule — implementation phase")


def _dot_conv_vjp(op, cotangents, primals):
    """Dot/conv VJP: transposed dot/conv of the cotangent with each primal
    operand (stub)."""
    raise NotImplementedError("dot/conv VJP rule — implementation phase")


# --- trivial zero rules (implemented — no algorithm) ----------------------

def _zero_jvp(op, tangents):
    """JVP rule for ops with non-differentiable outputs: zero tangents."""
    return (ZeroTangent(),) * len(op.results)


def _zero_vjp(op, cotangents, primals):
    """VJP rule for ops with non-differentiable outputs: zero gradients."""
    return (ZeroTangent(),) * len(op.operands)


def _stop_gradient_jvp(op, tangents):
    """`stop_gradient`: tangent = zero (spec: gradient is zero)."""
    return (ZeroTangent(),) * len(op.results)


def _stop_gradient_vjp(op, cotangents, primals):
    """`stop_gradient`: cotangent = zero (spec: gradient is zero)."""
    return (ZeroTangent(),) * len(op.operands)


# --- registration (table-driven; runs at import) --------------------------

def register_builtin_rules() -> None:
    """Install the builtin rules above into the public registries.

    Ops left unregistered on purpose (⇒ `TransformError` when transformed):
    `runtime_call`, collectives (`all_reduce`, `all_gather`, ...), control
    flow (`cond`, `while_loop`, `scan` — v1 defers region vectorization),
    and `block_call` (its rules arrive as `block:<name>` entries from
    `etl/block`).
    """
    # batching
    for name in ELEMENTWISE_OPS + NONDIFFERENTIABLE_OUTPUT_OPS:
        register_batching_rule(name, _pointwise_batching)
    for name in REDUCTION_OPS:
        register_batching_rule(name, _reduction_batching)
    register_batching_rule("reshape", _reshape_batching)
    register_batching_rule("transpose", _transpose_batching)
    for name in ("slice", "pad", "concatenate"):
        register_batching_rule(name, _structural_batching)
    for name in ("gather", "scatter"):
        register_batching_rule(name, _gather_scatter_batching)
    for name in DOT_OPS:
        register_batching_rule(name, _dot_conv_batching)

    # jvp
    for name in ELEMENTWISE_OPS:
        register_jvp_rule(name, _pointwise_jvp)
    for name in REDUCTION_OPS:
        register_jvp_rule(name, _reduction_jvp)
    for name in STRUCTURAL_OPS + ("gather", "scatter"):
        register_jvp_rule(name, _structural_jvp)
    for name in DOT_OPS:
        register_jvp_rule(name, _dot_conv_jvp)
    for name in NONDIFFERENTIABLE_OUTPUT_OPS:
        register_jvp_rule(name, _zero_jvp)
    register_jvp_rule("stop_gradient", _stop_gradient_jvp)

    # vjp
    for name in ELEMENTWISE_OPS:
        register_vjp_rule(name, _pointwise_vjp)
    for name in REDUCTION_OPS:
        register_vjp_rule(name, _reduction_vjp)
    for name in STRUCTURAL_OPS + ("gather", "scatter"):
        register_vjp_rule(name, _structural_vjp)
    for name in DOT_OPS:
        register_vjp_rule(name, _dot_conv_vjp)
    for name in NONDIFFERENTIABLE_OUTPUT_OPS:
        register_vjp_rule(name, _zero_vjp)
    register_vjp_rule("stop_gradient", _stop_gradient_vjp)


register_builtin_rules()
