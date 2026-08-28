"""Custom-block op conformance examples (category "op", tag "custom") —
``etl.block`` custom operations, the etl-native analog of custom kernels.

Contract summary (see ``etl/block/CONTEXT.md`` for the full binding spec):

- Declaration — factory form ``etl.block(name, inputs=[...], outputs=[...],
  ...) -> BlockOp`` (callable; builds a ``block_call`` IR op inside a
  trace) or decorator form ``@etl.block @etl.defn`` over a portable
  implementation (name = fn name; input/output specs derived from
  TensorSpec annotations).
- Implementations — ``BlockOp.impl("numpy")`` registers a per-backend
  implementation (v1: numpy; called with operand arrays plus the op's
  static-arg kwargs and returning ndarray(s)); ``BlockOp.portable(fn)``
  registers a portable decomposition (fn MUST be an ``@etl.defn``).
- Lower-time resolution (numpy backend, ``keep_backend_impls="numpy"``):
  a block with a registered numpy impl is KEPT as a ``block_call`` and
  dispatched by the interpreter at run time; a portable-only block is
  INLINED (graph->graph splice) at ``lower()`` time. Compiler backends
  require a portable (they inline with ``keep_backend_impls=None``) — but
  see the stablehlo note below.
- Transform rules — ``BlockOp.batching_rule(fn)`` /
  ``BlockOp.vjp_rule(fn)`` register into the PUBLIC ``etl.transforms``
  registries under ``block:<name>`` keys: batching ``rule(op, operands,
  axes) -> (new_values, new_axes)``; vjp ``rule(op, cotangents, primals)
  -> input_cotangents`` (entries ``ir.Value | None | ZeroTangent``). An
  explicit rule always wins; without one the portable decomposition
  fallback is used (pre-registered at declaration time when a portable
  exists); neither rule nor portable -> explicit ``TransformError`` naming
  the block (never a silent fallback).
- Static attributes exist (``attribute_schema``) but are deliberately NOT
  used here: static values reach numpy impls as JSON payload dicts
  (``{"kind": ..., "value": ...}``) and re-specialize portable traces as
  kwargs — attribute-free examples sidestep those subtleties.

Compiler-backend status (measured at dev time): ``block_call`` is a v1
StableHLO export deferral — ``etl.backends.stablehlo.export(graph)`` on a
block_call graph raises ``BackendError: stablehlo export: op 'block_call'
... is not supported in v1 — decompose it into supported ops or use a
future compiler adapter`` — and every compiler adapter (iree/xla/tvm)
declares ``custom_blocks=False``, so block graphs are rejected at
``lower()`` time with an explicit ``BackendError``. The numpy backend is
the only runnable backend for blocks in v1; all five examples pass
conformance on it (measured numbers below).

Blocks declared here (names are process-wide unique):

- ``custom_l2norm`` — row-wise L2 normalization ``y = x / ||x||`` over the
  last axis. Factory form; portable + numpy impl + EXPLICIT batching and
  vjp rules (the rules are the coverage point for the ``block:<name>``
  rule keys; the numpy impl is what actually executes on the numpy
  backend; the portable is what compiler backends would inline).
- ``custom_leaky_relu`` — leaky relu with a RUNTIME slope operand,
  ``y = maximum(x, s*x)`` (exactly leaky relu for ``0 < s < 1``).
  Decorator form; portable ONLY (no per-backend impl — exercises the
  lower-time portable inlining AND the decomposition-diff fallback). The
  portable is deliberately CONSTANT-FREE (the slope is a tensor operand,
  never a Python literal): grad through a portable-only block goes through
  the portable vjp fallback + ``_repair_portable_vjp`` local sweep, which
  (unlike the main graph sweep) currently requires a vjp rule for every
  inlined op — a literal slope (``etl.multiply(x, 0.1)``) would embed a
  ``constant`` op and raise ``TransformError: no VJP rule for op
  'constant'`` (measured at dev time; the main sweep skips constants at
  ``transforms/autodiff.py``, the repair sweep does not — core-side gap).

Measured max-abs errors (numpy backend, seed 0; strict defaults except the
grad examples at rtol=atol=1e-3 — the documented grad convention):

- custom_l2norm (forward): 0.0 (block_call dispatched to the numpy impl).
- custom_l2norm_grad: 8.8e-08 (dx) / 4.3e-08 (dw) vs fp64 central FD.
- custom_l2norm_vmap: 0.0 vs the manual per-sample-loop reference (the
  explicit batching rule recomputes the normalization with ordinary ops).
- custom_relu2 (forward): 0.0 (portable inlined at lower time).
- custom_relu2_grad: 1.5e-08 (dx) / 5.1e-06 (ds) vs fp64 central FD.
"""
import numpy as np

import etl
from etl import TensorSpec, defn
from etl.transforms.autodiff import ZeroTangent

from .._torch import require_torch
from .base import Example, _F32, fd_gradient, register_all

_DIM = 8
_BATCH = 32


def _sym(value):
    """Wrap an ir.Value as a SymbolicTensor so etl.ops can build on it."""
    return etl.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


# --- custom_l2norm: row-wise L2 normalization (factory form + all rules) -----
# y = x / sqrt(sum(x^2, last, keepdims)). Per-sample shape [4,8]: "row-wise"
# means over the LAST axis, which is exactly what makes the same block
# per-sample correct under vmap (batch axis leading). VJP: with y = x/n,
# dy/dx = (I - y y^T)/n, so the input cotangent is
# g = (ct - y * sum(ct*y, last, keepdims)) / n.

custom_l2norm = etl.block(
    "custom_l2norm",
    inputs=[TensorSpec((4, _DIM), _F32)],
    outputs=[TensorSpec((4, _DIM), _F32)],
)


@custom_l2norm.portable
@defn
def _l2norm_portable(x):
    norm = etl.sqrt(etl.sum(etl.square(x), axes=-1, keepdims=True))
    return etl.divide(x, norm)


@custom_l2norm.impl("numpy")
def _l2norm_numpy_impl(x):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / norm


@custom_l2norm.batching_rule
def _l2norm_batching_rule(op, operands, axes):
    # Batched semantics: the block normalizes the last axis, so the batched
    # computation is the same formula on the batched operand; the mapped
    # axis (0) passes through unchanged.
    (x,) = operands
    xs = _sym(x)
    norm = etl.sqrt(etl.sum(etl.square(xs), axes=-1, keepdims=True))
    y = etl.divide(xs, norm)
    return (y.value,), (axes[0],)


@custom_l2norm.vjp_rule
def _l2norm_vjp_rule(op, cotangents, primals):
    (ct,) = cotangents
    if ct is None or isinstance(ct, ZeroTangent):
        return (ZeroTangent(),)
    x = _sym(primals[0])
    y = _sym(op.results[0])
    norm = etl.sqrt(etl.sum(etl.square(x), axes=-1, keepdims=True))
    proj = etl.sum(etl.multiply(_sym(ct), y), axes=-1, keepdims=True)
    g = etl.divide(etl.subtract(_sym(ct), etl.multiply(y, proj)), norm)
    return (g.value,)


@defn
def _l2norm_fn(x):
    return custom_l2norm(x)


def _l2norm_numpy_ref(inputs):
    (x,) = inputs
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def _l2norm_torch_ref(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    xt = torch.as_tensor(x, device=device)
    return (xt / torch.sqrt(torch.sum(xt * xt, dim=-1, keepdim=True))).cpu().numpy()


def _l2norm_vmap_numpy(inputs):
    """MANUAL per-sample loop over the batch — never vmap."""
    (x,) = inputs
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = x[i] / np.sqrt(np.sum(x[i] * x[i], axis=-1, keepdims=True))
    return out


@defn
def _l2norm_loss(x, w):
    return etl.sum(etl.multiply(custom_l2norm(x), w))


def _l2norm_loss_value(inputs, frozen):
    x, w = inputs
    n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return np.sum((x / n) * w)


def _l2norm_grad_numpy(inputs):
    return tuple(fd_gradient(_l2norm_loss_value, inputs))


def _l2norm_grad_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.tensor(a, device=device, requires_grad=True) for a in inputs)
    n = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))
    loss = torch.sum((x / n) * w)
    grads = torch.autograd.grad(loss, (x, w))
    return tuple(g.cpu().numpy() for g in grads)


# --- custom_leaky_relu: leaky relu with runtime slope (decorator, portable) --
# y = maximum(x, s*x) — exactly leaky_relu(x; s) for 0 < s < 1. NO numpy
# impl: the portable is inlined at lower() time (portable-inlining path).
# NO explicit rules: grad/vmap go through the decomposition fallbacks.
# The slope is a tensor OPERAND (never a Python literal) so the portable
# contains no `constant` ops — see the module docstring ("constant-free").


@etl.block
@etl.defn
def custom_leaky_relu(
    x: TensorSpec((16,), _F32), s: TensorSpec((), _F32)
) -> TensorSpec((16,), _F32):
    return etl.maximum(x, etl.multiply(x, s))


def _relu2_inputs(seed):
    """x ~ N(0,1); slope FIXED at 0.2 (positive < 1 keeps leaky-relu
    semantics; the grad example shifts x by ±5 so both branches are always
    active and the central-difference never straddles the kink)."""
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(16).astype(np.float32),
        np.array(0.2, dtype=np.float32),
    ]


@defn
def _relu2_fn(x, s):
    return custom_leaky_relu(x, s)


def _relu2_numpy_ref(inputs):
    x, s = inputs
    return np.maximum(x, s * x)


def _relu2_torch_ref(inputs, device=None):
    torch = require_torch()
    x, s = (torch.as_tensor(t, device=device) for t in inputs)
    return torch.maximum(x, s * x).cpu().numpy()


@defn
def _relu2_loss(x, s):
    # Two shifted terms so BOTH branches are always active (x-5 < 0 < x+5
    # for x ~ N(0,1)): d/dx = s + 0.5*1 = 0.7 everywhere, and the FD never
    # straddles the x=0 kink (the leaky-relu gradient is discontinuous
    # there). The gradient wrt s = sum(x-5) comes from the negative branch.
    return etl.add(
        etl.sum(custom_leaky_relu(etl.subtract(x, 5.0), s)),
        etl.multiply(etl.sum(custom_leaky_relu(etl.add(x, 5.0), s)), 0.5),
    )


def _relu2_loss_value(inputs, frozen):
    x, s = inputs
    return np.sum(np.maximum(x - 5.0, s * (x - 5.0))) + 0.5 * np.sum(
        np.maximum(x + 5.0, s * (x + 5.0))
    )


def _relu2_grad_numpy(inputs):
    return tuple(fd_gradient(_relu2_loss_value, inputs))


def _relu2_grad_torch(inputs, device=None):
    torch = require_torch()
    x = torch.tensor(inputs[0], device=device, requires_grad=True)
    s = torch.tensor(inputs[1], device=device, requires_grad=True)
    loss = torch.sum(torch.maximum(x - 5.0, s * (x - 5.0))) + 0.5 * torch.sum(
        torch.maximum(x + 5.0, s * (x + 5.0))
    )
    grads = torch.autograd.grad(loss, (x, s))
    return tuple(g.cpu().numpy() for g in grads)


# ---------------------------------------------------------------------------
# Registry (category "op", tag "custom")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="custom_l2norm",
        description=(
            "row-wise L2 normalization via etl.block (numpy impl "
            "dispatched at run time; forward-only)"
        ),
        specs=(TensorSpec((4, _DIM), _F32),),
        graph=_l2norm_fn,
        numpy_ref=_l2norm_numpy_ref,
        torch_ref=_l2norm_torch_ref,
        category="op",
        tags=("custom",),
    ),
    Example(
        name="custom_l2norm_grad",
        description=(
            "grad of sum(l2norm(x) * w) through the block's explicit vjp "
            "rule (block:custom_l2norm key)"
        ),
        specs=(
            TensorSpec((4, _DIM), _F32),
            TensorSpec((4, _DIM), _F32),
        ),
        graph=etl.grad(_l2norm_loss, argnums=(0, 1)),
        numpy_ref=_l2norm_grad_numpy,
        torch_ref=_l2norm_grad_torch,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("custom",),
    ),
    Example(
        name="custom_l2norm_vmap",
        description=(
            "vmap over the l2norm block via its explicit batching rule "
            "(per-sample [4,8] rows over a [32,4,8] batch)"
        ),
        specs=(TensorSpec((_BATCH, 4, _DIM), _F32),),
        graph=lambda x_spec: etl.vmap(_l2norm_fn, in_axes=0)(x_spec),
        numpy_ref=_l2norm_vmap_numpy,
        torch_ref=_l2norm_torch_ref,
        category="op",
        tags=("custom",),
    ),
    Example(
        name="custom_relu2",
        description=(
            "leaky relu (maximum(x, s*x), runtime slope) via a portable-"
            "only block — inlined at lower time"
        ),
        specs=(
            TensorSpec((16,), _F32),
            TensorSpec((), _F32),
        ),
        graph=_relu2_fn,
        numpy_ref=_relu2_numpy_ref,
        torch_ref=_relu2_torch_ref,
        inputs_fn=_relu2_inputs,
        category="op",
        tags=("custom",),
    ),
    Example(
        name="custom_relu2_grad",
        description=(
            "grad of a two-branch shifted leaky-relu loss through the "
            "portable decomposition diff fallback (no explicit rules)"
        ),
        specs=(
            TensorSpec((16,), _F32),
            TensorSpec((), _F32),
        ),
        graph=etl.grad(_relu2_loss, argnums=(0, 1)),
        numpy_ref=_relu2_grad_numpy,
        torch_ref=_relu2_grad_torch,
        inputs_fn=_relu2_inputs,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("custom",),
    ),
])
