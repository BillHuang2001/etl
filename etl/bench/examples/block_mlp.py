"""MLP-block conformance examples (category "block", tag "mlp").

Two whole-MLP-block graphs composed from etl primitives (dot + relu +
layernorm + residual add): a 4-layer deep MLP with a middle layernorm and a
2-layer residual MLP. The numpy references mirror the graphs op-for-op in
f32 (the etl numpy backend executes the same numpy kernels), so
etl-vs-numpy errors are ~0; the torch references are exact formula mirrors
(never ``nn.Sequential`` etc.) and only differ by fp32 accumulation order.

Design notes:

- No biases (kept out on purpose — the graph is a compute/ops stress test,
  not a realistic model; same convention as the transformer block).
- Layernorm is built from ``mean``/``sum`` primitives over the last axis
  (eps 1e-5), matching :func:`~etl.bench.examples.base.layernorm_numpy`
  and the micro layernorm example.
- Sizing: both graphs are well under a MFLOP — single numpy-backend runs
  are far under the ~1 s budget.
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, register_all


def _layernorm_last(x, dim):
    """Layer norm over the last axis (mean/var from sum primitives, eps 1e-5)."""
    mean = etl.mean(x, axes=-1, keepdims=True)
    diff = etl.subtract(x, mean)
    var = etl.mean(etl.multiply(diff, diff), axes=-1, keepdims=True)
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


# --- deep_mlp_block (4 layers + middle layernorm) ----------------------------


@defn
def _deep_mlp_block_graph(x, w1, w2, w3, w4):
    """4-layer MLP block [4,32] -> 64 -> 32 -> 16 -> 8 with relu and a
    layernorm between the 2nd and 3rd layers (over the 32-wide hidden).

    ``h1 = relu(x @ w1); h2 = relu(h1 @ w2); n = layernorm(h2);
    h3 = relu(n @ w3); out = h3 @ w4``. Shapes: x[4,32], w1[32,64],
    w2[64,32], w3[32,16], w4[16,8] (all f32).
    """
    h1 = etl.relu(etl.dot(x, w1))
    h2 = etl.relu(etl.dot(h1, w2))
    h2n = _layernorm_last(h2, 32)
    h3 = etl.relu(etl.dot(h2n, w3))
    return etl.dot(h3, w4)


def _deep_mlp_block_numpy(inputs):
    x, w1, w2, w3, w4 = inputs
    h1 = np.maximum(x @ w1, 0.0)
    h2 = np.maximum(h1 @ w2, 0.0)
    mean = h2.mean(axis=-1, keepdims=True)
    diff = h2 - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    h2n = diff / np.sqrt(var + 1e-5)
    h3 = np.maximum(h2n @ w3, 0.0)
    return h3 @ w4


def _deep_mlp_block_torch(inputs, device=None):
    torch = require_torch()
    x, w1, w2, w3, w4 = (torch.as_tensor(a, device=device) for a in inputs)
    h1 = torch.relu(x @ w1)
    h2 = torch.relu(h1 @ w2)
    h2n = torch.nn.functional.layer_norm(h2, (32,), eps=1e-5)
    h3 = torch.relu(h2n @ w3)
    return (h3 @ w4).cpu().numpy()


# --- mlp_block_residual (2 layers + residual + layernorm) --------------------


@defn
def _mlp_block_residual_graph(x, w1, w2):
    """2-layer residual MLP block [4,32] -> 32 -> 32 with layernorm.

    ``h = relu(x @ w1); y = h @ w2 + x`` (residual); ``out = layernorm(y)``.
    Shapes: x[4,32], w1[32,32], w2[32,32] (all f32). The hidden width equals
    the input width so the residual add lines up.
    """
    h = etl.relu(etl.dot(x, w1))
    y = etl.add(etl.dot(h, w2), x)
    return _layernorm_last(y, 32)


def _mlp_block_residual_numpy(inputs):
    x, w1, w2 = inputs
    h = np.maximum(x @ w1, 0.0)
    y = h @ w2 + x
    mean = y.mean(axis=-1, keepdims=True)
    diff = y - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    return diff / np.sqrt(var + 1e-5)


def _mlp_block_residual_torch(inputs, device=None):
    torch = require_torch()
    x, w1, w2 = (torch.as_tensor(a, device=device) for a in inputs)
    h = torch.relu(x @ w1)
    y = h @ w2 + x
    return torch.nn.functional.layer_norm(y, (32,), eps=1e-5).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "block", tag "mlp")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="deep_mlp_block",
        description=(
            "4-layer MLP block [4,32] -> 64 -> 32 -> 16 -> 8: relu + middle "
            "layernorm + relu"
        ),
        specs=(
            TensorSpec((4, 32), _F32),
            TensorSpec((32, 64), _F32),
            TensorSpec((64, 32), _F32),
            TensorSpec((32, 16), _F32),
            TensorSpec((16, 8), _F32),
        ),
        graph=_deep_mlp_block_graph,
        numpy_ref=_deep_mlp_block_numpy,
        torch_ref=_deep_mlp_block_torch,
        # Strict defaults: etl and the numpy ref share the same dot kernels
        # op-for-op — measured max_abs_error 0.0 (exact); the torch ref
        # differs only by fp32 accumulation order (~1e-7, the micro mlp
        # precedent).
        category="block",
        tags=("mlp",),
    ),
    Example(
        name="mlp_block_residual",
        description=(
            "2-layer residual MLP block [4,32] -> 32 -> 32: relu + residual "
            "add + layernorm"
        ),
        specs=(
            TensorSpec((4, 32), _F32),
            TensorSpec((32, 32), _F32),
            TensorSpec((32, 32), _F32),
        ),
        graph=_mlp_block_residual_graph,
        numpy_ref=_mlp_block_residual_numpy,
        torch_ref=_mlp_block_residual_torch,
        # Strict defaults: measured max_abs_error 0.0 (identical kernels).
        category="block",
        tags=("mlp",),
    ),
])
