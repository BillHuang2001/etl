"""Block-category large examples (category "block", tag "large").

Two heavier examples that stress the numpy backend with production-sized
tensors: a full dummy transformer block (multi-head attention + layernorm +
MLP residual block) and one unrolled N-body step (multi-output). Both moved
verbatim from the former ``large.py`` (the other four former ``large``
examples — matmul_1024, conv2d_large, layernorm_large, vmap_mlp_large —
live in :mod:`etl.bench.examples.op_large`, category "op"). They keep the
"block" category because they compose whole blocks of ops into a single
graph. All stay within the v1 numpy-backend budget on CPU (single etl run
well under ~1 s each). Both also carry the ``"transformer_block"`` tag so the
CLI selector ``--examples transformer_block`` covers the whole transformer
family (the two legacy blocks + the three small block examples below).

Dev-time verification on iree/llvm-cpu (measured; see the per-example
comments for exact numbers):

- transformer: compiles+runs on iree/llvm-cpu after the merged StableHLO
  exporter fix (rank-mismatched 3D@2D dots now emit a valid
  ``stablehlo.dot_general`` via ``dynamic_broadcast_in_dim`` +
  ``get_dimension_size``); measured max_abs_error ≈7.25e-04 (fp32 fusion
  noise) covered by tolerance=1e-3; the numpy backend computes it exactly
  (max_abs 0.0).
- nbody: PASS on iree/llvm-cpu (small fp32 noise).

Design notes:

- Every graph mirrors the corresponding numpy reference op-for-op in f32
  (the etl numpy backend executes the same numpy kernels), so etl-vs-numpy
  errors are ~0; torch references are exact formula mirrors (never
  ``nn.TransformerEncoderLayer`` etc.) and only differ by fp32 accumulation
  order, so their error is measured and covered by the per-example tolerance.
- ``_layernorm_last`` (the etl-graph-side, SYMBOLIC layernorm helper) stays
  module-local; the numpy reference uses the shared
  :func:`~etl.bench.examples.base.layernorm_numpy` helper from base.
- Three small transformer-block examples (tag ``"transformer_block"``) complement
  the two large ones: ``mha_block`` (multi-head attention block — QKV
  projection, per-head scaled softmax, concat heads, out-projection),
  ``ffn_block`` (gelu MLP + residual + layernorm), and ``mha_ffn_block``
  (attention + feed-forward composed into one small transformer block with
  residuals). All three use static shapes ([2,16,32], 4 heads of dim 8),
  mirror the shared numpy helpers (``softmax_numpy`` / ``gelu_numpy`` /
  ``layernorm_numpy``) op-for-op, and pass the strict conformance defaults
  on the numpy backend (max_abs 0.0 — identical kernels); torch references
  are exact formula mirrors (fp32 accumulation-order noise only, same class
  as the micro attention/layernorm examples).
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import (
    Example,
    _F32,
    gelu_numpy,
    layernorm_numpy,
    register_all,
    softmax_numpy,
)

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
    # previously failed to COMPILE on compiler backends (invalid
    # ``stablehlo.dot_general``: lhs_batching_dimensions = [0] vs rhs = []).
    # The merged StableHLO exporter fix emits a valid ``dot_general`` for
    # rank-mismatched batched dots (``dynamic_broadcast_in_dim`` +
    # ``get_dimension_size``), so the example compiles+runs on iree; the
    # numpy backend computes it exactly (max_abs 0.0 vs ref). Kept in the
    # prescribed 3D@2D form.
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
    out = layernorm_numpy(out)  # shared base helper (mean/var, eps 1e-5)
    out = np.maximum(out @ w1, 0.0) @ w2 + out
    return layernorm_numpy(out)


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


# --- mha_block (multi-head attention block) ----------------------------------


def _mha_etl(x, wqkv):
    """Multi-head attention sub-block (etl side): QKV projection → per-head
    scaled softmax → concat heads. Returns the [2,16,32] attention output
    (no out-projection — callers add it). Shared by the ``mha_block`` and
    ``mha_ffn_block`` graphs so the two examples cannot drift apart.
    """
    qkv = etl.reshape(etl.dot(x, wqkv), (2, 16, 3, 4, 8))
    q = etl.reshape(etl.slice(qkv, (0, 0, 0, 0, 0), (2, 16, 1, 4, 8)), (2, 16, 4, 8))
    k = etl.reshape(etl.slice(qkv, (0, 0, 1, 0, 0), (2, 16, 1, 4, 8)), (2, 16, 4, 8))
    v = etl.reshape(etl.slice(qkv, (0, 0, 2, 0, 0), (2, 16, 1, 4, 8)), (2, 16, 4, 8))
    scale = 1.0 / math.sqrt(8)
    q = etl.transpose(q, (0, 2, 1, 3))  # [2,4,16,8]
    kt = etl.transpose(k, (0, 2, 3, 1))  # [2,4,8,16]
    v = etl.transpose(v, (0, 2, 1, 3))  # [2,4,16,8]
    scores = etl.multiply(etl.dot(q, kt), scale)  # [2,4,16,16]
    m = etl.max(scores, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(scores, m))
    probs = etl.divide(e, etl.sum(e, axes=-1, keepdims=True))
    attn = etl.dot(probs, v)  # [2,4,16,8]
    attn = etl.transpose(attn, (0, 2, 1, 3))  # [2,16,4,8]
    heads = [etl.slice(attn, (0, 0, h, 0), (2, 16, 1, 8)) for h in range(4)]
    return etl.reshape(etl.concatenate(heads, axis=3), (2, 16, 32))


def _mha_numpy(x, wqkv):
    """Multi-head attention sub-block (numpy side) — mirror of ``_mha_etl``,
    reusing the shared :func:`~etl.bench.examples.base.softmax_numpy`
    helper (identical max-subtract formula)."""
    qkv = (x @ wqkv).reshape(2, 16, 3, 4, 8)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scale = 1.0 / math.sqrt(8)
    scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * scale
    probs = softmax_numpy(scores)
    attn = (probs @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    attn = np.concatenate([attn[:, :, h : h + 1, :] for h in range(4)], axis=3)
    return attn.reshape(2, 16, 32)


def _mha_torch(x, wqkv, torch):
    """Multi-head attention sub-block (torch side) — exact formula mirror."""
    qkv = (x @ wqkv).reshape(2, 16, 3, 4, 8)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scale = 1.0 / math.sqrt(8)
    scores = (q.transpose(1, 2) @ k.permute(0, 2, 3, 1)) * scale
    probs = torch.softmax(scores, dim=-1)
    attn = (probs @ v.transpose(1, 2)).transpose(1, 2)
    attn = torch.cat([attn[:, :, h : h + 1, :] for h in range(4)], dim=3)
    return attn.reshape(2, 16, 32)


@defn
def _mha_block_graph(x, wqkv, wout):
    """Multi-head attention block: QKV proj, per-head scaled softmax, concat
    heads, out-projection. x[2,16,32], wqkv[32,96] (3*4*8), wout[32,32] (all
    f32)."""
    return etl.dot(_mha_etl(x, wqkv), wout)


def _mha_block_numpy(inputs):
    x, wqkv, wout = inputs
    return _mha_numpy(x, wqkv) @ wout


def _mha_block_torch(inputs, device=None):
    torch = require_torch()
    x, wqkv, wout = (torch.as_tensor(a, device=device) for a in inputs)
    return (_mha_torch(x, wqkv, torch) @ wout).cpu().numpy()


# --- ffn_block (gelu feed-forward block) -------------------------------------


@defn
def _ffn_block_graph(x, w1, w2):
    """2-layer feed-forward block: gelu(x @ w1) @ w2 + residual, then
    layernorm. x[2,16,32], w1[32,64], w2[64,32] (all f32)."""
    h = etl.gelu(etl.dot(x, w1))  # [2,16,64] (erf-form gelu)
    out = etl.add(etl.dot(h, w2), x)  # residual
    return _layernorm_last(out, 32)


def _ffn_block_numpy(inputs):
    x, w1, w2 = inputs
    out = gelu_numpy(x @ w1) @ w2 + x
    return layernorm_numpy(out)


def _ffn_block_torch(inputs, device=None):
    torch = require_torch()
    x, w1, w2 = (torch.as_tensor(a, device=device) for a in inputs)
    h = 0.5 * (x @ w1) * (1.0 + torch.erf((x @ w1) / math.sqrt(2.0)))
    out = h @ w2 + x
    return _layernorm_torch(out, 32).cpu().numpy()


# --- mha_ffn_block (attention + feed-forward, one small transformer block) ---


@defn
def _mha_ffn_block_graph(x, wqkv, wout, w1, w2):
    """Small transformer block: MHA (4 heads, d=8) + residual + layernorm,
    then gelu FFN + residual + layernorm. x[2,16,32], wqkv[32,96],
    wout[32,32], w1[32,64], w2[64,32] (all f32)."""
    h = _layernorm_last(etl.add(etl.dot(_mha_etl(x, wqkv), wout), x), 32)
    out = etl.add(etl.dot(etl.gelu(etl.dot(h, w1)), w2), h)
    return _layernorm_last(out, 32)


def _mha_ffn_block_numpy(inputs):
    x, wqkv, wout, w1, w2 = inputs
    h = layernorm_numpy(_mha_numpy(x, wqkv) @ wout + x)
    out = gelu_numpy(h @ w1) @ w2 + h
    return layernorm_numpy(out)


def _mha_ffn_block_torch(inputs, device=None):
    torch = require_torch()
    x, wqkv, wout, w1, w2 = (torch.as_tensor(a, device=device) for a in inputs)
    h = _layernorm_torch(_mha_torch(x, wqkv, torch) @ wout + x, 32)
    proj = h @ w1
    out = (0.5 * proj * (1.0 + torch.erf(proj / math.sqrt(2.0)))) @ w2 + h
    return _layernorm_torch(out, 32).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "block"; tags "large" + "transformer_block")
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
        # iree/llvm-cpu fp32 fusion noise — same class as the micro mlp
        # example's tolerance=1e-4 precedent (accumulation-order/FMA
        # contraction), larger because the transformer is much deeper:
        # measured max_abs_error ≈7.25e-04, exceeding the strict default
        # atol+rtol*|b|. tolerance=1e-3 covers it with margin; the numpy
        # backend computes it exactly (max_abs 0.0 — identical kernels).
        tolerance=1e-3,
        category="block",
        tags=("large", "transformer_block"),
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
        category="block",
        tags=("large", "transformer_block"),
    ),
    Example(
        name="mha_block",
        description=(
            "multi-head attention block: QKV proj, 4 heads (d=8) scaled "
            "softmax, concat heads, out-proj (x[2,16,32])"
        ),
        specs=(
            TensorSpec((2, 16, 32), _F32),
            TensorSpec((32, 96), _F32),
            TensorSpec((32, 32), _F32),
        ),
        graph=_mha_block_graph,
        numpy_ref=_mha_block_numpy,
        torch_ref=_mha_block_torch,
        # tolerance=1e-3: measured iree fp32 accumulation-order noise, same
        # class as the sibling ``transformer`` override — max_abs 2.37e-4
        # (iree cpu) / 2.48e-4 (iree cuda) vs the fp32 ref (~4x margin);
        # verified NOT a wrong-result bug (iree cuda vs fp64 ground truth =
        # 1.34e-4 while the numpy fp32 ref's own error vs fp64 = 3.06e-4 —
        # iree is closer to fp64). numpy backend stays exact (max_abs 0.0);
        # the torch ref is an exact formula mirror (torch.softmax vs the
        # max-subtract formula differ only by fp32 noise, same class as the
        # micro attention example, which passes strict defaults).
        tolerance=1e-3,
        category="block",
        tags=("transformer_block",),
    ),
    Example(
        name="ffn_block",
        description=(
            "2-layer feed-forward block: gelu MLP + residual + layernorm "
            "(x[2,16,32], hidden 64)"
        ),
        specs=(
            TensorSpec((2, 16, 32), _F32),
            TensorSpec((32, 64), _F32),
            TensorSpec((64, 32), _F32),
        ),
        graph=_ffn_block_graph,
        numpy_ref=_ffn_block_numpy,
        torch_ref=_ffn_block_torch,
        # numpy backend: max_abs 0.0 (identical kernels, erf-form gelu
        # mirrored by the shared gelu_numpy helper); torch ref uses the exact
        # erf form via torch.erf — fp32 noise only.
        category="block",
        tags=("transformer_block",),
    ),
    Example(
        name="mha_ffn_block",
        description=(
            "small transformer block: MHA (4 heads, d=8) + residual/layernorm "
            "+ gelu FFN + residual/layernorm (x[2,16,32])"
        ),
        specs=(
            TensorSpec((2, 16, 32), _F32),
            TensorSpec((32, 96), _F32),
            TensorSpec((32, 32), _F32),
            TensorSpec((32, 64), _F32),
            TensorSpec((64, 32), _F32),
        ),
        graph=_mha_ffn_block_graph,
        numpy_ref=_mha_ffn_block_numpy,
        torch_ref=_mha_ffn_block_torch,
        # Same noise class as mha_block/ffn_block: numpy exact (max_abs 0.0),
        # torch fp32 accumulation-order noise only.
        category="block",
        tags=("transformer_block",),
    ),
])
