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

- All six examples pass the strict defaults (rtol=atol=1e-5) against BOTH
  references: the forward examples match their numpy references EXACTLY
  (max_abs error 0.0 — the vectorized graphs and the hand-written
  per-sample-loop references run the same fp32 kernels in the same order);
  vmap_grad_mlp vs its float64 finite-difference reference measures
  max_abs 5.9e-06 (fp32 backward accumulation noise — the previous
  [1,8]-row formulation measured 1.28e-05; the [4,8] formulation is
  numerically comparable and stays well within the strict defaults).
  Torch references pass at the strict defaults as well.
- vmap_softmax cross-checked against a manually batched ``@etl.defn``
  (identical formulas, explicit leading batch dim): max_abs diff 0.0 and an
  IDENTICAL op sequence (``reduce_max, subtract, exp, reduce_sum, divide``)
  — the vmap graph contains exactly the ordinary ops a hand-batched graph
  would.
- iree/llvm-cpu status (all measured at dev time on the merged StableHLO
  exporter; "documented deferral" = an explicit BackendError, never a
  silent fallback — recorded per-example like the cumsum deferral):
  - vmap_mlp: COMPILES + RUNS (all-batched rank-matched formulation with
    per-sample rank-2 rows — no reshape anywhere): max_abs_error 4.8e-06
    vs the numpy reference (numpy backend exact).
  - vmap_grad_mlp: COMPILES + RUNS (the no-size-one-dims formulation —
    see below): max_abs_error 6.9e-06 vs the fd reference.
  - vmap_linear: COMPILES + RUNS on iree (shared w/b — the exporter's
    size-1-batch squeeze fast path is gated on fully static shapes, so the
    dynamic batch dim falls through to the batched
    ``dynamic_broadcast_in_dim`` + batched ``dot_general`` path, which
    legalizes on iree): max_abs_error 9.54e-07 (cpu) / 1.43e-06 (cuda) vs
    the numpy reference; the numpy backend computes it exactly (max_abs
    0.0).
  - vmap_softmax / vmap_layernorm / vmap_attention: documented deferral —
    keepdims reductions are semantically REQUIRED here (a no-keepdims
    reformulation is wrong: numpy trailing broadcasting would divide
    per-COLUMN instead of per-row), and every keepdims reduction lowers to
    a ``stablehlo.reshape`` whose operand carries the symbolic batch dim:
    export-time BackendError ``op 'reshape' ... keepdims reshape operand
    has dynamic dims (Dim('batch'),) — dynamic shapes are not supported by
    the StableHLO compiler backends in v1``. A reshape-free symbolic graph
    compiles and runs on iree; a reshape-free keepdims lowering (or bound
    dims) belongs in core. The numpy backend handles all three exactly.
- vmap_linear uses ``in_axes=(0, None, None)`` (shared per-sample
  weight/bias — the realistic linear layer); v1 vmap supports ``None``
  entries. The per-sample input is a rank-2 row ``[1,16]`` — no per-sample
  reshape anywhere: ``etl.dot`` is batched matmul (rank >= 2), so a rank-1
  vector would need a ``[1,16]`` round-trip reshape, which the exporter
  rejects on dynamic dims.
- vmap_grad_mlp: batched per-sample gradients via the supported
  composition ``vmap(grad(loss))`` — ``etl.vmap(etl.grad(f, argnums),
  in_axes=0)`` maps the per-sample loss gradient over the batch. The
  per-sample loss operates on ``[4,8]`` matrices (NOT ``[1,8]`` rows): v1's
  reduce-vjp broadcast-back sums away size-one dims via ``reduce_sum`` + a
  dynamic ``reshape`` (``_reduce_to_operand`` in
  ``etl/transforms/rules.py``), which the exporter rejects — the ``[4,8]``
  formulation has no size-one dims, so its vectorized graph is reshape-free
  and compiles on iree.
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
# keepdims is REQUIRED (row softmax): a no-keepdims reformulation would
# divide per-COLUMN under numpy trailing broadcasting (wrong values), and
# every keepdims reduction lowers to a stablehlo.reshape on the symbolic
# batch dim — export-time BackendError on iree (documented deferral).


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
# Same keepdims-required / iree-deferral note as vmap_softmax.


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
# Per-sample x is a rank-2 row [1,16] so no per-sample reshape is needed
# (etl.dot is batched matmul, rank >= 2; a rank-1 vector would need a
# [1,16] round-trip reshape, which the exporter rejects on dynamic dims).
# iree: COMPILES + RUNS — the shared-weight (size-1-batch) dot goes through
# the exporter's batched dynamic_broadcast_in_dim + batched dot_general
# path (the size-1-batch squeeze fast path is gated on fully static shapes,
# etl/backends/stablehlo/writer.py); measured max_abs_error 9.54e-07 (cpu)
# / 1.43e-06 (cuda) vs the numpy reference (numpy backend exact).


@defn
def _vmap_linear_sample(x, w, b):
    return etl.relu(etl.add(etl.dot(x, w), b))


def _vmap_linear_graph(x_spec, w_spec, b_spec):
    # Shared weight/bias: v1 vmap supports None in_axes entries (only the
    # leading axis of x is mapped; w/b stay unmapped and broadcast).
    return etl.vmap(_vmap_linear_sample, in_axes=(0, None, None), out_axes=0)(
        x_spec, w_spec, b_spec
    )


def _vmap_linear_numpy(inputs):
    x, w, b = inputs
    out = np.empty((x.shape[0], 1, w.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.maximum(x[i] @ w + b, 0.0)
    return out


def _vmap_linear_torch(inputs, device=None):
    torch = require_torch()
    x, w, b = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.relu(x @ w + b).cpu().numpy()


# --- vmap mlp: per-sample 2-layer relu MLP, all-batched weights -------------
# All-batched per-sample rank-2 shapes (x [1,16] rows, biases [1,32]/[1,8]):
# every dot is a matched-batch dot and every elementwise op is same-rank —
# the vectorized graph is reshape-free, so it COMPILES + RUNS on iree
# (measured max_abs_error 4.8e-06 vs the numpy reference).


@defn
def _vmap_mlp_sample(x, w1, b1, w2, b2):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    return etl.add(etl.dot(h, w2), b2)


def _vmap_mlp_graph(x_spec, w1_spec, b1_spec, w2_spec, b2_spec):
    return etl.vmap(_vmap_mlp_sample, in_axes=(0, 0, 0, 0, 0), out_axes=0)(
        x_spec, w1_spec, b1_spec, w2_spec, b2_spec
    )


def _vmap_mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    out = np.empty((x.shape[0], 1, w2.shape[2]), dtype=x.dtype)
    for i in range(x.shape[0]):
        h = np.maximum(x[i] @ w1[i] + b1[i], 0.0)
        out[i] = h @ w2[i] + b2[i]
    return out


def _vmap_mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    # Plain batched matmul: x[32,1,16] @ w1[32,16,32] has MATCHED batch dims
    # (left-aligned per-pair semantics — exactly what the vectorized graph
    # computes), so no einsum is needed; b1/b2 are the batched per-sample
    # biases [32,1,32]/[32,1,8] and plain `+` broadcasts correctly.
    h = torch.relu(x @ w1 + b1)
    return (h @ w2 + b2).cpu().numpy()


# --- vmap attention: per-sample single-head attention [8,16] over [16,8,16] --
# keepdims is REQUIRED here too (row softmax over 2-D scores) — same
# iree-deferral note as vmap_softmax.


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
# The per-sample loss operates on [4,8] matrices (4 rows x 8 features) —
# deliberately NO size-one dims: v1's reduce-vjp broadcast-back sums away
# size-one dims via reduce_sum + a dynamic reshape (`_reduce_to_operand` in
# etl/transforms/rules.py), which the exporter rejects. The [4,8]
# formulation's vectorized graph is reshape-free, so it COMPILES + RUNS on
# iree (measured max_abs_error 6.9e-06 vs the fd reference; numpy backend
# 5.9e-06).


@defn
def _vmap_grad_mlp_loss(x, y, w1, b1, w2, b2):
    """Per-sample scalar loss: MSE of a tiny 2-layer relu MLP over a [4,8]
    matrix (per-sample x[4,8] -> [4,16] -> [4,8]). No size-one dims (see
    the section comment); the full reduction keeps the loss a per-sample
    SCALAR (grad requires shape ())."""
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    pred = etl.add(etl.dot(h, w2), b2)
    diff = etl.subtract(pred, y)
    return etl.mean(etl.multiply(diff, diff))


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
    # Plain batched matmul (matched batch dims — same per-pair semantics as
    # the vectorized graph; no einsum needed). Per-sample MSE = mean over
    # the two trailing dims; summed over the batch to stay separable.
    h = torch.relu(x @ w1 + b1)
    pred = h @ w2 + b2
    per_sample = ((pred - y) ** 2).mean(dim=(-1, -2))
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
        description=(
            "per-sample softmax [16] vmap'd over a [32,16] batch; "
            "iree: documented deferral (keepdims-reshape export rejection)"
        ),
        specs=(TensorSpec((32, 16), _F32),),
        graph=_vmap_softmax_graph,
        numpy_ref=_vmap_softmax_numpy,
        torch_ref=_vmap_softmax_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_layernorm",
        description=(
            "per-sample layernorm [64] vmap'd over a [32,64] batch; "
            "iree: documented deferral (keepdims-reshape export rejection)"
        ),
        specs=(TensorSpec((32, 64), _F32),),
        graph=_vmap_layernorm_graph,
        numpy_ref=_vmap_layernorm_numpy,
        torch_ref=_vmap_layernorm_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_linear",
        description=(
            "per-sample relu(x @ w + b) vmap'd over x only (shared w/b); "
            "compiles+runs on iree (cpu max_abs 9.54e-07, cuda 1.43e-06)"
        ),
        specs=(
            TensorSpec((32, 1, 16), _F32),
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
        description=(
            "per-sample 2-layer relu MLP vmap'd, all-batched weights; "
            "compiles+runs on iree (cpu), max_abs 4.8e-06"
        ),
        specs=(
            TensorSpec((32, 1, 16), _F32),
            TensorSpec((32, 16, 32), _F32),
            TensorSpec((32, 1, 32), _F32),
            TensorSpec((32, 32, 8), _F32),
            TensorSpec((32, 1, 8), _F32),
        ),
        graph=_vmap_mlp_graph,
        numpy_ref=_vmap_mlp_numpy,
        torch_ref=_vmap_mlp_torch,
        category="vectorize",
    ),
    Example(
        name="vmap_attention",
        description=(
            "per-sample single-head attention [8,16] vmap'd over [16,8,16]; "
            "iree: documented deferral (keepdims-reshape export rejection)"
        ),
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
        description=(
            "per-sample MLP grads via vmap∘grad composition (batch 8, "
            "[4,8] per-sample rows); compiles+runs on iree (cpu), "
            "max_abs 6.9e-06"
        ),
        specs=(
            TensorSpec((8, 4, 8), _F32),
            TensorSpec((8, 4, 8), _F32),
            TensorSpec((8, 8, 16), _F32),
            TensorSpec((8, 4, 16), _F32),
            TensorSpec((8, 16, 8), _F32),
            TensorSpec((8, 4, 8), _F32),
        ),
        graph=_vmap_grad_mlp_graph,
        numpy_ref=_vmap_grad_mlp_numpy,
        torch_ref=_vmap_grad_mlp_torch,
        category="vectorize",
    ),
])
