"""Micro-benchmark examples (category 'micro') — the original 10 examples.

Moved verbatim from the former single-file ``etl/bench/examples.py``; the
:class:`Example` dataclass, registry, and shared references now live in
:mod:`etl.bench.examples.base`. Each example bundles:

- ``graph``: an ``@etl.defn`` graph taking fixed static ``TensorSpec`` inputs,
- ``specs``: the input specs (tuple of ``etl.TensorSpec``),
- ``numpy_ref``: a pure-numpy reference (same inputs → numpy output(s)),
- ``torch_ref``: an OPTIONAL factory that imports torch INSIDE the function
  body and raises the clear ``pip install etl[bench]`` hint error when torch
  is absent — merely listing examples (``import etl.bench``) never imports
  torch.

References must match the graph's output structure exactly (single ndarray or
a tuple of ndarrays for multi-output graphs).

Shapes are deliberately small so a full conformance run stays well under a
couple of seconds on the default numpy backend.
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, _conv2d_numpy, register_all

# --- matmul -----------------------------------------------------------------


@defn
def _matmul_graph(x, w):
    return etl.dot(x, w)


def _matmul_numpy(inputs):
    x, w = inputs
    return x @ w


def _matmul_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w).cpu().numpy()


# --- conv2d (VALID, stride 1) -----------------------------------------------


@defn
def _conv2d_graph(x, w):
    return etl.conv(x, w, strides=1, padding="VALID")


def _conv2d_numpy_ref(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(1, 1), padding="VALID")


def _conv2d_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=1, padding=0).cpu().numpy()


# --- conv2d (SAME padding, stride 1) ----------------------------------------


@defn
def _conv2d_same_graph(x, w):
    return etl.conv(x, w, strides=1, padding="SAME")


def _conv2d_same_numpy(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(1, 1), padding="SAME")


def _conv2d_same_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=1, padding=1).cpu().numpy()


# --- conv2d (VALID, stride 2) -----------------------------------------------


@defn
def _conv2d_stride2_graph(x, w):
    return etl.conv(x, w, strides=2, padding="VALID")


def _conv2d_stride2_numpy(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(2, 2), padding="VALID")


def _conv2d_stride2_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=2, padding=0).cpu().numpy()


# --- elementwise fusion -----------------------------------------------------


@defn
def _elementwise_fusion_graph(x, y, z):
    return etl.relu(etl.multiply(etl.add(x, y), z))


def _elementwise_fusion_numpy(inputs):
    x, y, z = inputs
    return np.maximum((x + y) * z, 0.0)


def _elementwise_fusion_torch(inputs, device=None):
    torch = require_torch()
    x, y, z = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.relu((x + y) * z).cpu().numpy()


# --- softmax (from exp/sum/max primitives) ----------------------------------


@defn
def _softmax_graph(x):
    m = etl.max(x, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(x, m))
    return etl.divide(e, etl.sum(e, axes=-1, keepdims=True))


def _softmax_numpy(inputs):
    (x,) = inputs
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.softmax(torch.as_tensor(x, device=device), dim=-1).cpu().numpy()


# --- layernorm (mean/var from sum primitives) -------------------------------


@defn
def _layernorm_graph(x):
    mean = etl.mean(x, axes=-1, keepdims=True)
    diff = etl.subtract(x, mean)
    var = etl.mean(etl.multiply(diff, diff), axes=-1, keepdims=True)
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


def _layernorm_numpy(inputs):
    (x,) = inputs
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    return diff / np.sqrt(var + 1e-5)


def _layernorm_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.nn.functional.layer_norm(
        torch.as_tensor(x, device=device), (x.shape[-1],), eps=1e-5
    ).cpu().numpy()


# --- mlp (2 linear layers + relu) -------------------------------------------


@defn
def _mlp_graph(x, w1, b1, w2, b2):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    return etl.add(etl.dot(h, w2), b2)


def _mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    h = np.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2


def _mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.as_tensor(a, device=device) for a in inputs)
    h = torch.relu(x @ w1 + b1)
    return (h @ w2 + b2).cpu().numpy()


# --- cumsum -----------------------------------------------------------------


@defn
def _cumsum_graph(x):
    return etl.cumsum(x, axis=1)


def _cumsum_numpy(inputs):
    (x,) = inputs
    return np.cumsum(x, axis=1)


def _cumsum_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.cumsum(torch.as_tensor(x, device=device), dim=1).cpu().numpy()


# --- attention (QK^T softmax V) ---------------------------------------------


@defn
def _attention_graph(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = etl.multiply(etl.dot(q, etl.transpose(k, (0, 2, 1))), scale)
    m = etl.max(scores, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(scores, m))
    probs = etl.divide(e, etl.sum(e, axes=-1, keepdims=True))
    return etl.dot(probs, v)


def _attention_numpy(inputs):
    q, k, v = inputs
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    return probs @ v


def _attention_torch(inputs, device=None):
    torch = require_torch()
    q, k, v = (torch.as_tensor(a, device=device) for a in inputs)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "op", tag "micro")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="matmul",
        description="[64,64] x [64,64] matrix multiply (etl.dot)",
        specs=(TensorSpec((64, 64), _F32), TensorSpec((64, 64), _F32)),
        graph=_matmul_graph,
        numpy_ref=_matmul_numpy,
        torch_ref=_matmul_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="conv2d",
        description="NCHW 2D conv, VALID padding, stride 1 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_graph,
        numpy_ref=_conv2d_numpy_ref,
        torch_ref=_conv2d_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="conv2d_same",
        description="NCHW 2D conv, SAME padding, stride 1 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_same_graph,
        numpy_ref=_conv2d_same_numpy,
        torch_ref=_conv2d_same_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="conv2d_stride2",
        description="NCHW 2D conv, VALID padding, stride 2 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_stride2_graph,
        numpy_ref=_conv2d_stride2_numpy,
        torch_ref=_conv2d_stride2_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="elementwise_fusion",
        description="relu((x + y) * z) elementwise fusion chain",
        specs=(
            TensorSpec((64, 64), _F32),
            TensorSpec((64, 64), _F32),
            TensorSpec((64, 64), _F32),
        ),
        graph=_elementwise_fusion_graph,
        numpy_ref=_elementwise_fusion_numpy,
        torch_ref=_elementwise_fusion_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="softmax",
        description="row-wise softmax from exp/sum/max primitives",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_softmax_graph,
        numpy_ref=_softmax_numpy,
        torch_ref=_softmax_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="layernorm",
        description="row-wise layernorm (mean/var from sum primitives, eps=1e-5)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_layernorm_graph,
        numpy_ref=_layernorm_numpy,
        torch_ref=_layernorm_torch,
        category="op",
        tags=("micro",),
    ),
    # mlp tolerance override: documented iree fp32 accumulation-order noise —
    # deterministic max_abs_error ≈3.8e-05 (iree cpu) / ≈6.1e-05 (iree cuda);
    # 1e-4 passes comfortably; IREE's fp32 output is closer to float64 ground
    # truth than the numpy fp32 reference itself, so relaxing the harness
    # tolerance is honest, not weakening.
    Example(
        name="mlp",
        description="2-layer MLP: relu(x @ w1 + b1) @ w2 + b2",
        specs=(
            TensorSpec((64, 64), _F32),
            TensorSpec((64, 64), _F32),
            TensorSpec((64,), _F32),
            TensorSpec((64, 64), _F32),
            TensorSpec((64,), _F32),
        ),
        graph=_mlp_graph,
        numpy_ref=_mlp_numpy,
        torch_ref=_mlp_torch,
        tolerance=1e-4,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="cumsum",
        description="cumulative sum along axis 1 (etl.cumsum)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_cumsum_graph,
        numpy_ref=_cumsum_numpy,
        torch_ref=_cumsum_torch,
        category="op",
        tags=("micro",),
    ),
    Example(
        name="attention",
        description="single-head attention: softmax(Q K^T / sqrt(d)) V",
        specs=(
            TensorSpec((4, 8, 16), _F32),
            TensorSpec((4, 8, 16), _F32),
            TensorSpec((4, 8, 16), _F32),
        ),
        graph=_attention_graph,
        numpy_ref=_attention_numpy,
        torch_ref=_attention_torch,
        category="op",
        tags=("micro",),
    ),
])
