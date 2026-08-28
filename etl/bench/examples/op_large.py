"""Large-problem op conformance examples (category "op", tag "large").

Four heavier op examples that stress the numpy backend with production-sized
tensors: a 1024x1024 matmul, a large NCHW conv, a wide layernorm, and a
``vmap`` over a per-sample MLP. (The ``transformer`` and ``nbody`` examples
that used to live in the former ``large.py`` moved to
:mod:`etl.bench.examples.block_transformer`, category "block".) All stay
within the v1 numpy-backend budget on CPU (single etl run well under ~1 s
each; full conformance of the category < ~60 s).

Dev-time verification on iree/llvm-cpu (measured; see the per-example
comments for exact numbers):

- matmul_1024: compile ~1 s + ~3.9 s/run; max_abs_error 1.91e-04..2.37e-04
  across 4 seeds (numpy backend exactly 0). Sized 1024 because 4096 measured
  384 s/run (~0.18 GFLOPS generic codegen; 2048 → 26 s/run) — a full default
  iree/cpu CLI run would take hours; 1024 keeps the full harness ≈5-7 min
  (budget guardrail).
- conv2d_large: max_abs_error 1.45e-04 on iree/llvm-cpu (numpy worst-over-
  seeds 1.22e-04).
- layernorm_large: PASS on iree/llvm-cpu (small fp32 noise).
- vmap_mlp_large: COMPILES + RUNS on iree/llvm-cpu after the vmap examples
  were reformulated to per-sample rank-2 rows (all-batched matched dots,
  no per-sample reshape): measured max_abs_error 9.2e-05 (fp32
  accumulation-order noise, covered by the existing atol=1.2e-4); the
  numpy backend computes it exactly (max_abs 0.0 — identical per-sample
  kernels). The previous per-sample-vector formulation (with a [1,64]
  reshape round-trip) was an iree export-time rejection (dynamic reshape).

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
  uses rank-2 rows ``[1,64]`` — no per-sample ``etl.reshape`` anywhere (a
  reshape round-trip would be rejected by the StableHLO exporter on the
  symbolic batch dim).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, conv2d_im2col_numpy, layernorm_numpy, register_all

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
    torch.backends.cudnn.allow_tf32 = False  # cuDNN TF32 (torch default) degrades fp32 conv precision ~0.036 abs on cuda; TF32 is a perf optimization, not fp32 semantics — keep the reference accurate
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
    # Shared base helper: mean/var over the last axis (eps 1e-5) — the same
    # formula as the graph's sum-primitive layernorm.
    return layernorm_numpy(x)


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
    """Per-sample 2-layer MLP over a rank-2 row: [1,64] -> relu -> [1,128]
    -> [1,64].

    ``etl.dot`` is batched matmul only (rank >= 2), and the StableHLO
    exporter rejects per-sample reshape round-trips on the symbolic batch
    dim — so the sample is a ``[1,64]`` row and every op is same-rank /
    matched-batch (no reshape anywhere; compiles+runs on iree).
    """
    h = etl.relu(etl.add(etl.dot(x, w1), b1))  # [1,128]
    return etl.add(etl.dot(h, w2), b2)  # [1,64]


def _vmap_mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    # Manual batched loop (never vmap): per-sample mlp, stacked.
    out = np.empty((x.shape[0], 1, w2.shape[2]), dtype=x.dtype)
    for i in range(x.shape[0]):
        h = np.maximum(x[i] @ w1[i] + b1[i], 0.0)
        out[i] = h @ w2[i] + b2[i]
    return out


def _vmap_mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    # Plain batched matmul: x[64,1,64] @ w1[64,64,128] has MATCHED batch
    # dims (left-aligned per-pair semantics — exactly what the vectorized
    # graph computes), so no einsum is needed.
    h = torch.relu(x @ w1 + b1)
    return (h @ w2 + b2).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "op", tag "large")
# ---------------------------------------------------------------------------

register_all([
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
        category="op",
        tags=("large",),
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
        category="op",
        tags=("large",),
    ),
    Example(
        name="layernorm_large",
        description="row-wise layernorm on [4096,1024] from sum primitives",
        specs=(TensorSpec((4096, 1024), _F32),),
        graph=_layernorm_large_graph,
        numpy_ref=_layernorm_large_numpy,
        torch_ref=_layernorm_large_torch,
        category="op",
        tags=("large",),
    ),
    Example(
        name="vmap_mlp_large",
        description=(
            "etl.vmap over a per-sample 2-layer MLP (batch 64, "
            "per-sample [1,64] -> relu -> 128 -> [1,64]); "
            "compiles+runs on iree (cpu), max_abs 9.2e-05"
        ),
        specs=(
            TensorSpec((64, 1, 64), _F32),
            TensorSpec((64, 64, 128), _F32),
            TensorSpec((64, 1, 128), _F32),
            TensorSpec((64, 128, 64), _F32),
            TensorSpec((64, 1, 64), _F32),
        ),
        graph=etl.vmap(_vmap_mlp_sample, in_axes=(0, 0, 0, 0, 0), out_axes=0),
        numpy_ref=_vmap_mlp_numpy,
        torch_ref=_vmap_mlp_torch,
        # torch ref: fp32 accumulation-order noise in the batched matmuls —
        # measured max_abs_error <= 9.16e-05 across 10 seeds (etl vs numpy
        # ref: exactly 0 — per-sample loop). iree/llvm-cpu measured 9.2e-05
        # (same noise class). atol=1.2e-04 covers it with margin.
        atol=1.2e-4,
        category="op",
        tags=("large",),
    ),
])
