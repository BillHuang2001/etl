"""Vectorization (vmap) conformance examples (category 'vectorize').

Each example is a per-sample ``@etl.defn`` wrapped via ``etl.vmap(fn,
in_axes=..., out_axes=0)``. ``example.graph`` is a PLAIN python function
``(*batched_specs) -> Graph`` (never an ``@etl.defn``): calling it invokes
the vmap TransformCallable with the BATCHED specs (leading batch dim
included) and returns the vectorized Graph, which the harness stages
through the explicit pipeline (``lower`` → ``compile`` → ``load`` → ``run``)
— the same contract ``etl.trace`` satisfies for ``@etl.defn`` graphs.

References are MANUALLY batched numpy implementations — an explicit
per-sample loop over the batch with vectorized per-sample ops (no
``np.vectorize`` / ``np.apply_along_axis``) — so the auto-vectorization is
genuinely validated, not re-derived from numpy's own broadcasting. Torch
references use batched torch ops (lazy import via ``require_torch``).

Dev-time verification (numpy backend, recorded once at development time;
seed 0 inputs):
- All five forward examples (vmap_softmax/layernorm/linear/mlp/attention)
  match their numpy references EXACTLY (max_abs error 0.0): the vectorized
  graphs and the hand-written per-sample-loop references run the same fp32
  kernels in the same order.
- vmap_softmax cross-checked against a manually batched ``@etl.defn``
  (identical formulas, explicit leading batch dim): max_abs diff 0.0 and an
  IDENTICAL op sequence (``reduce_max, subtract, exp, reduce_sum, divide``)
  — the vmap graph contains exactly the ordinary ops a hand-batched graph
  would.
- vmap_grad_mlp vs the finite-difference reference: max_abs 1.28e-05 at a
  gradient element of magnitude 108 (fp32 backward accumulation noise), max
  relative error 6.3e-06 — passes the strict defaults (rtol=atol=1e-5)
  elementwise. No tolerance relaxation needed.
- Torch references (lazy ``require_torch`` pattern) are executed and
  validated with torch present: all six examples pass conformance against
  both references at the strict defaults (torch=enabled, rtol=atol=1e-5,
  seed 0). Measured max_abs error vs the etl numpy-backend outputs: 0.0 for
  the softmax/layernorm/linear/attention examples, 7.6e-06 for vmap_mlp,
  and 1.14e-05 for vmap_grad_mlp (gradient magnitudes up to ~108) — within
  ``atol + rtol*|b|`` elementwise. The MLP torch references use per-sample
  einsum matmuls (a plain ``x @ w1`` with x[32,16] and w1[32,16,32]
  broadcasts the matrix per batch), and vmap_grad_mlp's torch reference
  differentiates only the first five tensors, mirroring
  argnums=(0,1,2,3,4) (b2 excluded).
- iree spot check (vmap_softmax, backend='iree', device cpu): FAILS at
  compile time with a GENUINE core/backend limitation, not an example bug:
  ``vectorize`` replaces each mapped static dim with a fresh symbolic
  ``Dim("batch")`` by design (mapped-output detection keyed on those dims),
  the StableHLO exporter renders symbolic dims as unbounded dynamic ``?``,
  and iree-compile rejects ``stablehlo.reshape`` with a dynamic result dim
  (``'stablehlo.reshape' op result #0 must be statically shaped or single
  bounded dimension tensor ... got 'tensor<?x1xf32>'``) — the keepdims
  reductions and the rank/batch-aligning reshapes of every shape-changing
  vmap graph hit this. Minimal repro WITHOUT vmap: tracing the same softmax
  ``@etl.defn`` with a symbolic spec ``TensorSpec((etl.dim("B"), 16))`` and
  lowering to iree fails identically; a symbolic-spec graph without
  reshapes (``relu(sum(x, axes=-1))``) compiles and runs on iree, as does a
  reshape-free vmap graph (pure elementwise). Fix belongs in core (bound
  dims or a reshape-free keepdims lowering); the numpy backend handles all
  vmap examples exactly.
- vmap_linear uses ``in_axes=(0, None, None)`` (shared per-sample
  weight/bias — the realistic linear layer); v1 vmap supports ``None``
  entries.
- vmap_grad_mlp: batched per-sample gradients via the supported
  composition ``vmap(grad(loss))`` — ``etl.vmap(etl.grad(f, argnums),
  in_axes=0)`` maps the per-sample loss gradient over the batch.
  Plain ``etl.grad(vmap(loss)(*specs))`` (grad applied to the vmap'd graph)
  cannot work in v1: the vmap'd graph's single output is the batched
  per-sample-loss VECTOR ``[B]``, and ``etl.grad`` requires exactly one
  SCALAR output (measured: ``ShapeError: grad requires the output to be a
  scalar tensor (shape ()), got shape (DimExpr(...batch...),)``) — there is
  no mean-over-batch composition as a single transform call. ``vmap∘grad``
  is the equivalent semantics: per-sample gradients of the separable
  per-sample MSE, matching the numpy reference (per-sample finite
  differences, stacked) and the torch reference (``autograd.grad`` of the
  summed batched loss — sum over the batch keeps the loss separable, so the
  gradient is exactly the stack of per-sample gradients).
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, fd_gradient, register_all

# --- vmap softmax: per-sample softmax [16] over a [32,16] batch -------------


@defn
def _vmap_softmax_sample(x):
    m = etl.max(x, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(x, m))
    return etl.divide(e, etl.sum(e, axes=-1, keepdims=True))


def _vmap_softmax_graph(x_spec):
    return etl.vmap(_vmap_softmax_sample, in_axes=0, out_axes=0)(x_spec)


def _vmap_softmax_numpy(inputs):
    (x,) = inputs
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        e = np.exp(xi - xi.max(keepdims=True))
        out[i] = e / e.sum(keepdims=True)
    return out


def _vmap_softmax_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.softmax(torch.as_tensor(x, device=device), dim=-1).cpu().numpy()


# --- vmap layernorm: per-sample layernorm [64] over a [32,64] batch ----------


@defn
def _vmap_layernorm_sample(x):
    mean = etl.mean(x, axes=-1, keepdims=True)
    diff = etl.subtract(x, mean)
    var = etl.mean(etl.multiply(diff, diff), axes=-1, keepdims=True)
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


def _vmap_layernorm_graph(x_spec):
    return etl.vmap(_vmap_layernorm_sample, in_axes=0, out_axes=0)(x_spec)


def _vmap_layernorm_numpy(inputs):
    (x,) = inputs
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        mean = xi.mean(keepdims=True)
        diff = xi - mean
        var = (diff * diff).mean(keepdims=True)
        out[i] = diff / np.sqrt(var + 1e-5)
    return out


def _vmap_layernorm_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.nn.functional.layer_norm(
        torch.as_tensor(x, device=device), (x.shape[-1],), eps=1e-5
    ).cpu().numpy()


# --- vmap linear: relu(x @ w + b) with SHARED w/b (in_axes=(0, None, None)) --
# v1 `etl.dot` requires rank >= 2 operands (batched matmul), so the per-sample
# row vector x[16] is reshaped to [1,16] around the dot and back to [8] — a
# numerical no-op that the vectorize reshape rule maps to [32,1,16]→[32,8].


@defn
def _vmap_linear_sample(x, w, b):
    x_row = etl.reshape(x, (1, x.shape[0]))
    out = etl.add(etl.dot(x_row, w), b)
    return etl.reshape(etl.relu(out), (out.shape[1],))


def _vmap_linear_graph(x_spec, w_spec, b_spec):
    # Shared weight/bias: v1 vmap supports None in_axes entries (only the
    # leading axis of x is mapped; w/b stay unmapped and broadcast).
    return etl.vmap(_vmap_linear_sample, in_axes=(0, None, None), out_axes=0)(
        x_spec, w_spec, b_spec
    )


def _vmap_linear_numpy(inputs):
    x, w, b = inputs
    out = np.empty((x.shape[0], w.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.maximum(x[i] @ w + b, 0.0)
    return out


def _vmap_linear_torch(inputs, device=None):
    torch = require_torch()
    x, w, b = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.relu(x @ w + b).cpu().numpy()


# --- vmap mlp: per-sample 2-layer relu MLP, all-batched weights -------------
# Per-sample x[16] is rank 1 — same [1,16] reshape round-trip around the
# rank-2 dots as vmap_linear (vectorized to [32,1,16] / [32,1,32]).


@defn
def _vmap_mlp_sample(x, w1, b1, w2, b2):
    x_row = etl.reshape(x, (1, x.shape[0]))
    h = etl.relu(etl.add(etl.dot(x_row, w1), b1))
    out = etl.add(etl.dot(h, w2), b2)
    return etl.reshape(out, (out.shape[1],))


def _vmap_mlp_graph(x_spec, w1_spec, b1_spec, w2_spec, b2_spec):
    return etl.vmap(_vmap_mlp_sample, in_axes=(0, 0, 0, 0, 0), out_axes=0)(
        x_spec, w1_spec, b1_spec, w2_spec, b2_spec
    )


def _vmap_mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    out = np.empty((x.shape[0], w2.shape[2]), dtype=x.dtype)
    for i in range(x.shape[0]):
        h = np.maximum(x[i] @ w1[i] + b1[i], 0.0)
        out[i] = h @ w2[i] + b2[i]
    return out


def _vmap_mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    # Per-sample batched matmul via einsum: with x[32,16] and w1[32,16,32], a
    # plain `x @ w1` would BROADCAST the matrix per batch → (32,16,8) instead
    # of the per-sample (32,8) result. b1/b2 are the BATCHED per-sample
    # biases [32,32]/[32,8], so plain `+` broadcasts correctly.
    h = torch.relu(torch.einsum("bi,bij->bj", x, w1) + b1)
    return (torch.einsum("bj,bjk->bk", h, w2) + b2).cpu().numpy()


# --- vmap attention: per-sample single-head attention [8,16] over [16,8,16] --


@defn
def _vmap_attention_sample(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = etl.multiply(etl.dot(q, etl.transpose(k, (1, 0))), scale)
    m = etl.max(scores, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(scores, m))
    probs = etl.divide(e, etl.sum(e, axes=-1, keepdims=True))
    return etl.dot(probs, v)


def _vmap_attention_graph(q_spec, k_spec, v_spec):
    return etl.vmap(_vmap_attention_sample, in_axes=(0, 0, 0), out_axes=0)(
        q_spec, k_spec, v_spec
    )


def _vmap_attention_numpy(inputs):
    q, k, v = inputs
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        qi, ki, vi = q[i], k[i], v[i]
        scale = 1.0 / math.sqrt(qi.shape[-1])
        scores = (qi @ ki.T) * scale
        # axis=-1 is required: the per-sample scores are 2-D, and a bare
        # keepdims max/sum would reduce over ALL axes (global), not rows.
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        out[i] = probs @ vi
    return out


def _vmap_attention_torch(inputs, device=None):
    torch = require_torch()
    q, k, v = (torch.as_tensor(a, device=device) for a in inputs)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).cpu().numpy()


# --- vmap_grad_mlp: batched per-sample gradients via vmap∘grad composition --


@defn
def _vmap_grad_mlp_loss(x, y, w1, b1, w2, b2):
    """Per-sample scalar loss: MSE of a tiny 2-layer relu MLP (x[8]→[16]→[8]).

    x[8] is rank 1 — the same [1,8] reshape round-trip around the rank-2
    dots as vmap_linear/vmap_mlp (vectorized to [8,1,8] over the batch).
    """
    x_row = etl.reshape(x, (1, x.shape[0]))
    h = etl.relu(etl.add(etl.dot(x_row, w1), b1))
    pred = etl.add(etl.dot(h, w2), b2)
    pred = etl.reshape(pred, (pred.shape[1],))
    diff = etl.subtract(pred, y)
    return etl.mean(etl.multiply(diff, diff), axes=-1)


def _vmap_grad_mlp_graph(x_spec, y_spec, w1_spec, b1_spec, w2_spec, b2_spec):
    # grad-of-vmap for a batched loss would need a scalar output; the vmap'd
    # graph's output is the batched per-sample-loss vector [B] — grad
    # requires a scalar (ShapeError, see module docstring). The supported,
    # equivalent composition is vmap∘grad: per-sample gradients mapped over
    # the batch (per-sample losses are separable — the batch gradient is the
    # stack of per-sample gradients, exactly what the references compute).
    per_sample_grad = etl.vmap(
        etl.grad(_vmap_grad_mlp_loss, argnums=(0, 1, 2, 3, 4)),
        in_axes=(0, 0, 0, 0, 0, 0),
        out_axes=(0, 0, 0, 0, 0),
    )
    return per_sample_grad(x_spec, y_spec, w1_spec, b1_spec, w2_spec, b2_spec)


def _vmap_grad_mlp_loss_f64(inputs, frozen):
    """float64 per-sample loss for ``fd_gradient`` (fd discipline: computes
    in float64 regardless of the fp32 input arrays)."""
    x, y, w1, b1, w2, b2 = inputs
    h = np.maximum(x @ w1 + b1, 0.0)
    pred = h @ w2 + b2
    diff = pred - y
    return float(np.mean(diff * diff))


def _vmap_grad_mlp_numpy(inputs):
    """Per-sample finite-difference gradients, stacked over the batch
    (each sample's gradient is the gradient of that sample's own MSE — the
    per-sample losses are separable)."""
    batch = inputs[0].shape[0]
    grads = [np.zeros_like(a, dtype=np.float64) for a in inputs]
    for i in range(batch):
        sample = [a[i] for a in inputs]
        for j, grad in enumerate(fd_gradient(_vmap_grad_mlp_loss_f64, sample)):
            grads[j][i] = grad
    # Mirror the etl graph's argnums=(0, 1, 2, 3, 4): b2 (index 5) is not
    # differentiated, so only the first five gradients are returned.
    return tuple(grads[j] for j in range(5))


def _vmap_grad_mlp_torch(inputs, device=None):
    """autograd.grad of the SUMMED batched loss: sum over the batch (not the
    mean) keeps the loss separable, so the gradient equals the stack of
    per-sample gradients — the exact semantics of vmap∘grad."""
    torch = require_torch()
    x, y, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    tensors = [x, y, w1, b1, w2, b2]
    for t in tensors:
        t.requires_grad_(True)
    # Per-sample batched matmul via einsum (same broadcasting pitfall as
    # vmap_mlp: x[8,8] @ w1[8,8,16] would broadcast the matrix per batch and
    # differentiate a DIFFERENT loss).
    h = torch.relu(torch.einsum("bi,bij->bj", x, w1) + b1)
    pred = torch.einsum("bj,bjk->bk", h, w2) + b2
    per_sample = ((pred - y) ** 2).mean(dim=-1)
    loss = per_sample.sum()
    # Mirror argnums=(0, 1, 2, 3, 4): b2 (the last tensor) is not
    # differentiated.
    grads = torch.autograd.grad(loss, tensors[:5])
    return tuple(g.detach().cpu().numpy() for g in grads)


# ---------------------------------------------------------------------------
# Registry (category "vectorize")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="vmap_softmax",
        description="per-sample softmax [16] vmap'd over a [32,16] batch",
        specs=(TensorSpec((32, 16), _F32),),
        graph=_vmap_softmax_graph,
        numpy_ref=_vmap_softmax_numpy,
        torch_ref=_vmap_softmax_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_layernorm",
        description="per-sample layernorm [64] vmap'd over a [32,64] batch",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_vmap_layernorm_graph,
        numpy_ref=_vmap_layernorm_numpy,
        torch_ref=_vmap_layernorm_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_linear",
        description="per-sample relu(x @ w + b) vmap'd over x only (shared w/b)",
        specs=(
            TensorSpec((32, 16), _F32),
            TensorSpec((16, 8), _F32),
            TensorSpec((8,), _F32),
        ),
        graph=_vmap_linear_graph,
        numpy_ref=_vmap_linear_numpy,
        torch_ref=_vmap_linear_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_mlp",
        description="per-sample 2-layer relu MLP vmap'd, all-batched weights",
        specs=(
            TensorSpec((32, 16), _F32),
            TensorSpec((32, 16, 32), _F32),
            TensorSpec((32, 32), _F32),
            TensorSpec((32, 32, 8), _F32),
            TensorSpec((32, 8), _F32),
        ),
        graph=_vmap_mlp_graph,
        numpy_ref=_vmap_mlp_numpy,
        torch_ref=_vmap_mlp_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_attention",
        description="per-sample single-head attention [8,16] vmap'd over [16,8,16]",
        specs=(
            TensorSpec((16, 8, 16), _F32),
            TensorSpec((16, 8, 16), _F32),
            TensorSpec((16, 8, 16), _F32),
        ),
        graph=_vmap_attention_graph,
        numpy_ref=_vmap_attention_numpy,
        torch_ref=_vmap_attention_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_grad_mlp",
        description="per-sample MLP grads via vmap∘grad composition (batch 8)",
        specs=(
            TensorSpec((8, 8), _F32),
            TensorSpec((8, 8), _F32),
            TensorSpec((8, 8, 16), _F32),
            TensorSpec((8, 16), _F32),
            TensorSpec((8, 16, 8), _F32),
            TensorSpec((8, 8), _F32),
        ),
        graph=_vmap_grad_mlp_graph,
        numpy_ref=_vmap_grad_mlp_numpy,
        torch_ref=_vmap_grad_mlp_torch,
        category="vectorize",
    ),
])
