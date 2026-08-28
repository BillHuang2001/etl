"""Large-problem conformance examples (category 'large').

Six heavier examples that stress the numpy backend with production-sized
tensors: a full dummy transformer block (multi-head attention + layernorm +
MLP residual block), one unrolled N-body step (multi-output), a 1024x1024
matmul, a large NCHW conv, a wide layernorm, and a ``vmap`` over a per-sample
MLP. All stay within the v1 numpy-backend budget on CPU (single etl run well
under ~1 s each; full conformance of the category < ~60 s).

Dev-time verification on iree/llvm-cpu (measured; see the per-example
comments for exact numbers):

- matmul_1024: compile ~1 s + ~3.9 s/run; max_abs_error 1.91e-04..2.37e-04
  across 4 seeds (numpy backend exactly 0). Sized 1024 because 4096 measured
  384 s/run (~0.18 GFLOPS generic codegen; 2048 → 26 s/run) — a full default
  iree/cpu CLI run would take hours; 1024 keeps the full harness ≈5-7 min
  (budget guardrail).
- conv2d_large: max_abs_error 1.45e-04 on iree/llvm-cpu (numpy worst-over-
  seeds 1.22e-04).
- layernorm_large / nbody: PASS on iree/llvm-cpu (small fp32 noise).
- transformer: iree compile FAILS with a reported core StableHLO exporter
  bug (``stablehlo.dot_general`` batching mismatch on the rank-mismatched
  3D@2D QKV dot) — a documented per-example failure; the numpy backend
  computes it correctly (max_abs 0.0).
- vmap_mlp_large: iree compile fails with a BackendError (DimExpr broadcast
  — documented v1 limitation).

Design notes:

- Every graph mirrors the corresponding numpy reference op-for-op in f32
  (the etl numpy backend executes the same numpy kernels), so etl-vs-numpy
  errors are ~0; torch references are exact formula mirrors (never
  ``nn.TransformerEncoderLayer`` etc.) and only differ by fp32 accumulation
  order, so their error is measured and covered by the per-example tolerance.
- ``vmap_mlp_large``'s ``graph`` is an ``etl.vmap`` *TransformCallable*
  (deliberately NOT an ``@etl.defn``): the harness stages transform-produced
  graphs through the explicit ``lower``/``compile``/``load`` pipeline
  (``example.graph(*example.specs) -> Graph`` first). v1 requires a pytree
  ``in_axes`` when the wrapped fn takes more than one tensor spec (a bare
  ``in_axes=0`` only applies to exactly one tensor input), so
  ``in_axes=(0, 0, 0, 0, 0)`` is used — identical semantics to the
  ``in_axes=0`` shorthand.
- ``etl.dot`` is batched matmul only (rank >= 2), so the per-sample MLP
  promotes its vector sample to a row via ``etl.reshape``.
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, conv2d_im2col_numpy, register_all

# --- transformer (dummy transformer block) -----------------------------------


@defn
def _transformer_graph(x, wqkv, wout, w1, w2):
    """One transformer block: MHA (12 heads) + layernorm + MLP + layernorm.

    Shapes: x[1,512,768], wqkv[768,2304], wout[768,768], w1[768,3072],
    w2[3072,768] (all f32). No biases (kept out on purpose — the graph is a
    compute/ops stress test, not a realistic model).

    Pipeline: QKV = x @ wqkv -> reshape [1,512,3,12,64] -> slice the 3-axis
    into q/k/v -> per-tensor transpose to [1,12,512,64] -> scaled
    softmax(Q K^T) V (3D batched dots over [1,12,...]) -> transpose back ->
    concatenate the 12 heads along the feature axis (12 single-head slices
    along the head axis, concatenated on the last axis) -> reshape
    [1,512,768] -> out-proj -> residual + x -> layernorm (mean/var from sum
    primitives, eps 1e-5) -> relu MLP -> residual -> layernorm.
    """
    # NOTE: this rank-mismatched 3D@2D batched dot ([1,512,768] @ [768,2304])
    # fails to COMPILE on compiler backends — a reported core StableHLO
    # exporter bug (invalid ``stablehlo.dot_general``: lhs_batching_dimensions
    # = [0] vs rhs = [] — "lhs and rhs should have the same number of batching
    # dimensions"), NOT a harness problem. iree records it as a documented
    # per-example BackendError; the numpy backend computes it correctly
    # (max_abs 0.0 vs ref). Kept in the prescribed 3D@2D form.
    qkv = etl.dot(x, wqkv)  # [1,512,2304]
    qkv = etl.reshape(qkv, (1, 512, 3, 12, 64))
    q = etl.reshape(etl.slice(qkv, (0, 0, 0, 0, 0), (1, 512, 1, 12, 64)), (1, 512, 12, 64))
    k = etl.reshape(etl.slice(qkv, (0, 0, 1, 0, 0), (1, 512, 1, 12, 64)), (1, 512, 12, 64))
    v = etl.reshape(etl.slice(qkv, (0, 0, 2, 0, 0), (1, 512, 1, 12, 64)), (1, 512, 12, 64))

    scale = 1.0 / math.sqrt(64)
    q = etl.transpose(q, (0, 2, 1, 3))  # [1,12,512,64]
    kt = etl.transpose(k, (0, 2, 3, 1))  # [1,12,64,512]
    v = etl.transpose(v, (0, 2, 1, 3))  # [1,12,512,64]
    scores = etl.multiply(etl.dot(q, kt), scale)  # [1,12,512,512]
    m = etl.max(scores, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(scores, m))
    probs = etl.divide(e, etl.sum(e, axes=-1, keepdims=True))
    attn = etl.dot(probs, v)  # [1,12,512,64]
    attn = etl.transpose(attn, (0, 2, 1, 3))  # [1,512,12,64]

    # Concatenate the 12 heads along the feature axis (classic concat-heads).
    heads = [etl.slice(attn, (0, 0, h, 0), (1, 512, 1, 64)) for h in range(12)]
    attn = etl.reshape(etl.concatenate(heads, axis=3), (1, 512, 768))

    out = etl.add(etl.dot(attn, wout), x)  # out-proj + residual
    out = _layernorm_last(out, 768)
    out = etl.add(etl.dot(etl.relu(etl.dot(out, w1)), w2), out)  # MLP + residual
    return _layernorm_last(out, 768)


def _layernorm_last(x, dim):
    """Layer norm over the last axis from sum primitives (eps 1e-5)."""
    mean = etl.divide(etl.sum(x, axes=-1, keepdims=True), float(dim))
    diff = etl.subtract(x, mean)
    var = etl.divide(
        etl.sum(etl.multiply(diff, diff), axes=-1, keepdims=True), float(dim)
    )
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


def _transformer_numpy(inputs):
    x, wqkv, wout, w1, w2 = inputs
    qkv = x @ wqkv
    qkv = qkv.reshape(1, 512, 3, 12, 64)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scale = 1.0 / math.sqrt(64)
    scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * scale
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    attn = (probs @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    attn = np.concatenate([attn[:, :, h : h + 1, :] for h in range(12)], axis=3)
    attn = attn.reshape(1, 512, 768)
    out = attn @ wout + x
    out = _layernorm_numpy(out, 768)
    out = np.maximum(out @ w1, 0.0) @ w2 + out
    return _layernorm_numpy(out, 768)


def _layernorm_numpy(x, dim):
    mean = x.sum(axis=-1, keepdims=True) / float(dim)
    diff = x - mean
    var = (diff * diff).sum(axis=-1, keepdims=True) / float(dim)
    return diff / np.sqrt(var + 1e-5)


def _transformer_torch(inputs, device=None):
    torch = require_torch()
    x, wqkv, wout, w1, w2 = (torch.as_tensor(a, device=device) for a in inputs)
    qkv = x @ wqkv
    qkv = qkv.reshape(1, 512, 3, 12, 64)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scale = 1.0 / math.sqrt(64)
    scores = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * scale
    m = scores.max(dim=-1, keepdim=True).values
    e = torch.exp(scores - m)
    probs = e / e.sum(dim=-1, keepdim=True)
    attn = (probs @ v.transpose(1, 2)).transpose(1, 2)
    attn = torch.cat([attn[:, :, h : h + 1, :] for h in range(12)], dim=3)
    attn = attn.reshape(1, 512, 768)
    out = attn @ wout + x
    out = _layernorm_torch(out, 768)
    out = torch.relu(out @ w1) @ w2 + out
    return _layernorm_torch(out, 768).cpu().numpy()


def _layernorm_torch(x, dim):
    mean = x.sum(dim=-1, keepdim=True) / float(dim)
    diff = x - mean
    var = (diff * diff).sum(dim=-1, keepdim=True) / float(dim)
    return diff / (var + 1e-5).sqrt()


# --- nbody (one unrolled step, multi-output) ---------------------------------


@defn
def _nbody_graph(p, v, m):
    """One unrolled N-body step (G=1.0, dt=0.01), output ``(p, v)``.

    No ``while_loop``: exactly one step. ``diff = p[:,None,:] - p[None,:,:]``
    via ``etl.reshape`` + implicit broadcast (newaxis indexing is not
    supported), softened ``r2 + 1e-6``, ``inv_r3 = r2 ** -1.5``, pairwise
    forces, accelerations, then leapfrog-ish ``v += a*dt; p += v*dt``.
    """
    n = p.shape[0]
    diff = etl.subtract(etl.reshape(p, (n, 1, 3)), etl.reshape(p, (1, n, 3)))
    r2 = etl.add(etl.sum(etl.multiply(diff, diff), axes=-1), 1e-6)
    inv_r3 = etl.power(r2, -1.5)
    force = etl.multiply(
        etl.multiply(
            etl.multiply(
                etl.reshape(m, (n, 1, 1)), etl.reshape(m, (1, n, 1))
            ),
            diff,
        ),
        etl.reshape(inv_r3, (n, n, 1)),
    )
    accel = etl.divide(etl.sum(force, axes=-2), etl.reshape(m, (n, 1)))
    v_new = etl.add(v, etl.multiply(accel, 0.01))
    p_new = etl.add(p, etl.multiply(v_new, 0.01))
    return (p_new, v_new)


def _nbody_numpy(inputs):
    p, v, m = inputs
    n = p.shape[0]
    diff = p.reshape(n, 1, 3) - p.reshape(1, n, 3)
    r2 = (diff * diff).sum(axis=-1) + 1e-6
    inv_r3 = r2 ** -1.5
    force = (
        m.reshape(n, 1, 1)
        * m.reshape(1, n, 1)
        * diff
        * inv_r3.reshape(n, n, 1)
    )
    accel = force.sum(axis=-2) / m.reshape(n, 1)
    v_new = v + accel * 0.01
    p_new = p + v_new * 0.01
    return (p_new, v_new)


def _nbody_torch(inputs, device=None):
    torch = require_torch()
    p, v, m = (torch.as_tensor(a, device=device) for a in inputs)
    n = p.shape[0]
    diff = p.reshape(n, 1, 3) - p.reshape(1, n, 3)
    r2 = (diff * diff).sum(dim=-1) + 1e-6
    inv_r3 = r2 ** -1.5
    force = (
        m.reshape(n, 1, 1)
        * m.reshape(1, n, 1)
        * diff
        * inv_r3.reshape(n, n, 1)
    )
    accel = force.sum(dim=-2) / m.reshape(n, 1)
    v_new = v + accel * 0.01
    p_new = p + v_new * 0.01
    return (p_new.cpu().numpy(), v_new.cpu().numpy())


# --- matmul_1024 --------------------------------------------------------------


@defn
def _matmul_1024_graph(x, w):
    return etl.dot(x, w)


def _matmul_1024_numpy(inputs):
    x, w = inputs
    return x @ w


def _matmul_1024_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w).cpu().numpy()


# --- conv2d_large --------------------------------------------------------------


@defn
def _conv2d_large_graph(x, w):
    return etl.conv(x, w, strides=1, padding="VALID")


def _conv2d_large_numpy(inputs):
    x, w = inputs
    # Vectorized im2col reference from base (identical semantics to the
    # loop-based _conv2d_numpy, but practical at this size).
    return conv2d_im2col_numpy(x, w, strides=(1, 1), padding="VALID")


def _conv2d_large_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=1, padding=0).cpu().numpy()


# --- layernorm_large -----------------------------------------------------------


@defn
def _layernorm_large_graph(x):
    """Layer norm over the last axis of [4096,1024] from sum primitives."""
    mean = etl.divide(etl.sum(x, axes=-1, keepdims=True), 1024.0)
    diff = etl.subtract(x, mean)
    var = etl.divide(
        etl.sum(etl.multiply(diff, diff), axes=-1, keepdims=True), 1024.0
    )
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


def _layernorm_large_numpy(inputs):
    (x,) = inputs
    mean = x.sum(axis=-1, keepdims=True) / 1024.0
    diff = x - mean
    var = (diff * diff).sum(axis=-1, keepdims=True) / 1024.0
    return diff / np.sqrt(var + 1e-5)


def _layernorm_large_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    x = torch.as_tensor(x, device=device)
    mean = x.sum(dim=-1, keepdim=True) / 1024.0
    diff = x - mean
    var = (diff * diff).sum(dim=-1, keepdim=True) / 1024.0
    return (diff / torch.sqrt(var + 1e-5)).cpu().numpy()


# --- vmap_mlp_large ------------------------------------------------------------
# ``example.graph`` is an ``etl.vmap`` TransformCallable (NOT an ``@etl.defn``)
# — the harness routes transform-produced graphs through the explicit
# ``lower``/``compile``/``load`` pipeline. v1 requires a pytree ``in_axes``
# for multi-tensor fns (bare ``in_axes=0`` is only valid with exactly one
# tensor input), hence ``(0, 0, 0, 0, 0)`` — same semantics as ``in_axes=0``.


@defn
def _vmap_mlp_sample(x, w1, b1, w2, b2):
    """Per-sample 2-layer MLP: [64] -> relu -> [128] -> [64].

    ``etl.dot`` is batched matmul only (rank >= 2), so the vector sample is
    promoted to a row ``[1,64]`` and squeezed back — identical math to a
    matrix-vector product.
    """
    h = etl.add(etl.dot(etl.reshape(x, (1, 64)), w1), b1)  # [1,128]
    h = etl.reshape(etl.relu(h), (128,))
    y = etl.add(etl.dot(etl.reshape(h, (1, 128)), w2), b2)  # [1,64]
    return etl.reshape(y, (64,))


def _vmap_mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    # Manual batched loop (never vmap): per-sample mlp, stacked.
    out = np.stack(
        [
            np.maximum(x[i] @ w1[i] + b1[i], 0.0) @ w2[i] + b2[i]
            for i in range(x.shape[0])
        ]
    )
    return out


def _vmap_mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    h = torch.relu(torch.einsum("bi,bij->bj", x, w1) + b1)
    return (torch.einsum("bj,bjk->bk", h, w2) + b2).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "large")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="transformer",
        description=(
            "dummy transformer block: 12-head MHA + layernorm + relu MLP "
            "(x[1,512,768], no biases)"
        ),
        specs=(
            TensorSpec((1, 512, 768), _F32),
            TensorSpec((768, 2304), _F32),
            TensorSpec((768, 768), _F32),
            TensorSpec((768, 3072), _F32),
            TensorSpec((3072, 768), _F32),
        ),
        graph=_transformer_graph,
        numpy_ref=_transformer_numpy,
        torch_ref=_transformer_torch,
        category="large",
    ),
    Example(
        name="nbody",
        description=(
            "one unrolled N-body step (N=1024, G=1.0, dt=0.01), "
            "multi-output (p, v)"
        ),
        specs=(
            TensorSpec((1024, 3), _F32),
            TensorSpec((1024, 3), _F32),
            TensorSpec((1024,), _F32),
        ),
        graph=_nbody_graph,
        numpy_ref=_nbody_numpy,
        torch_ref=_nbody_torch,
        # torch ref: fp32 accumulation-order noise in the 1024-term force
        # reduction — measured max_abs_error 1.24e-05 across 10 seeds (etl vs
        # numpy ref: exactly 0 — identical kernels). atol=2e-05 covers it
        # with margin while staying ~5x stricter than the micro mlp override.
        atol=2e-5,
        category="large",
    ),
    Example(
        name="matmul_1024",
        description="[1024,1024] x [1024,1024] fp32 matmul (etl.dot)",
        specs=(
            TensorSpec((1024, 1024), _F32),
            TensorSpec((1024, 1024), _F32),
        ),
        graph=_matmul_1024_graph,
        numpy_ref=_matmul_1024_numpy,
        torch_ref=_matmul_1024_torch,
        # Sized 1024: 4096 measured 384 s/run on iree llvm-cpu (~0.18 GFLOPS
        # generic codegen; 2048 → 26 s/run) — a full default iree/cpu CLI run
        # would take hours; 1024 keeps single-run ≈3.9 s and the full harness
        # ≈5-7 min per the budget guardrail (tune down only with justification
        # — measured). numpy backend still exercises a 2 GFLOP matmul.
        # Error budget: etl vs numpy ref exactly 0 (identical np.matmul
        # kernels); torch ref noise 8.4e-05 at seed 0; iree/llvm-cpu measured
        # max_abs_error 1.91e-04..2.37e-04 across 4 seeds (fp32 accumulation-
        # order + FMA contraction noise). atol=5e-04 gives ~2.1x margin over
        # the largest measured error (2.37e-04).
        atol=5e-4,
        category="large",
    ),
    Example(
        name="conv2d_large",
        description=(
            "NCHW 2D conv, VALID padding, stride 1: "
            "x[8,64,64,64] @ w[128,64,3,3]"
        ),
        specs=(
            TensorSpec((8, 64, 64, 64), _F32),
            TensorSpec((128, 64, 3, 3), _F32),
        ),
        graph=_conv2d_large_graph,
        numpy_ref=_conv2d_large_numpy,
        torch_ref=_conv2d_large_torch,
        # fp32 accumulation-order noise: etl's conv kernel (sliding_window_view
        # + tensordot) vs the im2col einsum reference — measured max_abs_error
        # 9.2e-05..1.22e-04 vs numpy and <= 8.4e-05 vs torch across 10 seeds.
        # iree/llvm-cpu measured 1.45e-04 (only ~3% headroom at atol=1.5e-04
        # — flaky across seeds/backends); numpy worst-over-seeds 1.22e-04;
        # atol=2e-04 gives ~1.4x margin over both.
        atol=2e-4,
        category="large",
    ),
    Example(
        name="layernorm_large",
        description="row-wise layernorm on [4096,1024] from sum primitives",
        specs=(TensorSpec((4096, 1024), _F32),),
        graph=_layernorm_large_graph,
        numpy_ref=_layernorm_large_numpy,
        torch_ref=_layernorm_large_torch,
        category="large",
    ),
    Example(
        name="vmap_mlp_large",
        description=(
            "etl.vmap over a per-sample 2-layer MLP (batch 64, "
            "per-sample [64] -> relu -> 128 -> [64])"
        ),
        specs=(
            TensorSpec((64, 64), _F32),
            TensorSpec((64, 64, 128), _F32),
            TensorSpec((64, 128), _F32),
            TensorSpec((64, 128, 64), _F32),
            TensorSpec((64, 64), _F32),
        ),
        graph=etl.vmap(_vmap_mlp_sample, in_axes=(0, 0, 0, 0, 0), out_axes=0),
        numpy_ref=_vmap_mlp_numpy,
        torch_ref=_vmap_mlp_torch,
        # torch ref: fp32 accumulation-order noise in the batched matmuls —
        # measured max_abs_error <= 9.16e-05 across 10 seeds (etl vs numpy
        # ref: exactly 0 — per-sample loop). atol=1.2e-04 covers it with
        # margin.
        atol=1.2e-4,
        category="large",
    ),
])
