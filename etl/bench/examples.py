"""Curated registry of example etl programs for conformance & benchmarking.

Each :class:`Example` bundles:

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
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import etl
from etl import TensorSpec, defn, float32
from ._torch import require_torch

__all__ = [
    "Example",
    "UnknownExampleError",
    "list_examples",
    "get_example",
    "generate_inputs",
]


class UnknownExampleError(ValueError):
    """Raised by :func:`get_example` for unknown example names."""


@dataclass(frozen=True)
class Example:
    """A registered benchmark/conformance example.

    Attributes:
        name: stable registry key.
        description: one-line human-readable description.
        specs: tuple of ``etl.TensorSpec`` (static integer shapes).
        graph: ``@etl.defn`` graph taking one symbolic tensor per spec.
        numpy_ref: ``(inputs) -> ndarray | tuple[ndarray]``; pure numpy.
        torch_ref: optional ``(inputs) -> ndarray | tuple[ndarray]`` factory
            that imports torch inside its body (never at module scope).
    """

    name: str
    description: str
    specs: tuple
    graph: Callable
    numpy_ref: Callable
    torch_ref: Optional[Callable] = None

    def generate_inputs(self, seed: int = 0):
        """Generate a list of numpy arrays matching ``specs`` (see module
        :func:`generate_inputs`)."""
        return generate_inputs(self, seed)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _conv2d_numpy(x, w, strides=(1, 1), padding="VALID"):
    """Loop-based NCHW 2D convolution reference.

    Mirrors etl's conv semantics exactly: ``"VALID"`` → no padding;
    ``"SAME"`` → TF convention — ``out = ceil(d / stride)`` and total pad
    ``(out - 1) * stride + k - d`` split as ``(total // 2, total - total // 2)``
    per spatial axis (matches ``etl/backends/numpy/kernels/linalg.py``).
    """
    n, c_in, h, win = x.shape
    c_out, _, kh, kw = w.shape
    sh, sw = strides
    if padding == "SAME":
        out_h = (h + sh - 1) // sh
        out_w = (win + sw - 1) // sw
        total_h = max((out_h - 1) * sh + kh - h, 0)
        total_w = max((out_w - 1) * sw + kw - win, 0)
        pad_h = (total_h // 2, total_h - total_h // 2)
        pad_w = (total_w // 2, total_w - total_w // 2)
    elif padding == "VALID":
        out_h = (h - kh) // sh + 1
        out_w = (win - kw) // sw + 1
        pad_h = pad_w = (0, 0)
    else:
        raise ValueError(f"unsupported padding mode {padding!r}")
    xp = np.pad(x, ((0, 0), (0, 0), pad_h, pad_w))
    out = np.zeros(
        (n, c_out, out_h, out_w), dtype=np.result_type(x.dtype, w.dtype)
    )
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            out[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, w)
    return out


# --- matmul -----------------------------------------------------------------


@defn
def _matmul_graph(x, w):
    return etl.dot(x, w)


def _matmul_numpy(inputs):
    x, w = inputs
    return x @ w


def _matmul_torch(inputs):
    torch = require_torch()
    x, w = (torch.from_numpy(a) for a in inputs)
    return (x @ w).numpy()


# --- conv2d (VALID, stride 1) -----------------------------------------------


@defn
def _conv2d_graph(x, w):
    return etl.conv(x, w, strides=1, padding="VALID")


def _conv2d_numpy_ref(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(1, 1), padding="VALID")


def _conv2d_torch(inputs):
    torch = require_torch()
    x, w = (torch.from_numpy(a) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=1, padding=0).numpy()


# --- conv2d (SAME padding, stride 1) ----------------------------------------


@defn
def _conv2d_same_graph(x, w):
    return etl.conv(x, w, strides=1, padding="SAME")


def _conv2d_same_numpy(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(1, 1), padding="SAME")


def _conv2d_same_torch(inputs):
    torch = require_torch()
    x, w = (torch.from_numpy(a) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=1, padding=1).numpy()


# --- conv2d (VALID, stride 2) -----------------------------------------------


@defn
def _conv2d_stride2_graph(x, w):
    return etl.conv(x, w, strides=2, padding="VALID")


def _conv2d_stride2_numpy(inputs):
    x, w = inputs
    return _conv2d_numpy(x, w, strides=(2, 2), padding="VALID")


def _conv2d_stride2_torch(inputs):
    torch = require_torch()
    x, w = (torch.from_numpy(a) for a in inputs)
    return torch.nn.functional.conv2d(x, w, stride=2, padding=0).numpy()


# --- elementwise fusion -----------------------------------------------------


@defn
def _elementwise_fusion_graph(x, y, z):
    return etl.relu(etl.multiply(etl.add(x, y), z))


def _elementwise_fusion_numpy(inputs):
    x, y, z = inputs
    return np.maximum((x + y) * z, 0.0)


def _elementwise_fusion_torch(inputs):
    torch = require_torch()
    x, y, z = (torch.from_numpy(a) for a in inputs)
    return torch.relu((x + y) * z).numpy()


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


def _softmax_torch(inputs):
    torch = require_torch()
    (x,) = inputs
    return torch.softmax(torch.from_numpy(x), dim=-1).numpy()


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


def _layernorm_torch(inputs):
    torch = require_torch()
    (x,) = inputs
    return torch.nn.functional.layer_norm(
        torch.from_numpy(x), (x.shape[-1],), eps=1e-5
    ).numpy()


# --- mlp (2 linear layers + relu) -------------------------------------------


@defn
def _mlp_graph(x, w1, b1, w2, b2):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    return etl.add(etl.dot(h, w2), b2)


def _mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    h = np.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2


def _mlp_torch(inputs):
    torch = require_torch()
    x, w1, b1, w2, b2 = (torch.from_numpy(a) for a in inputs)
    h = torch.relu(x @ w1 + b1)
    return (h @ w2 + b2).numpy()


# --- cumsum -----------------------------------------------------------------


@defn
def _cumsum_graph(x):
    return etl.cumsum(x, axis=1)


def _cumsum_numpy(inputs):
    (x,) = inputs
    return np.cumsum(x, axis=1)


def _cumsum_torch(inputs):
    torch = require_torch()
    (x,) = inputs
    return torch.cumsum(torch.from_numpy(x), dim=1).numpy()


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


def _attention_torch(inputs):
    torch = require_torch()
    q, k, v = (torch.from_numpy(a) for a in inputs)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).numpy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_F32 = float32

_EXAMPLES = {
    "matmul": Example(
        name="matmul",
        description="[64,64] x [64,64] matrix multiply (etl.dot)",
        specs=(TensorSpec((64, 64), _F32), TensorSpec((64, 64), _F32)),
        graph=_matmul_graph,
        numpy_ref=_matmul_numpy,
        torch_ref=_matmul_torch,
    ),
    "conv2d": Example(
        name="conv2d",
        description="NCHW 2D conv, VALID padding, stride 1 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_graph,
        numpy_ref=_conv2d_numpy_ref,
        torch_ref=_conv2d_torch,
    ),
    "conv2d_same": Example(
        name="conv2d_same",
        description="NCHW 2D conv, SAME padding, stride 1 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_same_graph,
        numpy_ref=_conv2d_same_numpy,
        torch_ref=_conv2d_same_torch,
    ),
    "conv2d_stride2": Example(
        name="conv2d_stride2",
        description="NCHW 2D conv, VALID padding, stride 2 (etl.conv)",
        specs=(TensorSpec((2, 2, 8, 8), _F32), TensorSpec((3, 2, 3, 3), _F32)),
        graph=_conv2d_stride2_graph,
        numpy_ref=_conv2d_stride2_numpy,
        torch_ref=_conv2d_stride2_torch,
    ),
    "elementwise_fusion": Example(
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
    ),
    "softmax": Example(
        name="softmax",
        description="row-wise softmax from exp/sum/max primitives",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_softmax_graph,
        numpy_ref=_softmax_numpy,
        torch_ref=_softmax_torch,
    ),
    "layernorm": Example(
        name="layernorm",
        description="row-wise layernorm (mean/var from sum primitives, eps=1e-5)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_layernorm_graph,
        numpy_ref=_layernorm_numpy,
        torch_ref=_layernorm_torch,
    ),
    "mlp": Example(
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
    ),
    "cumsum": Example(
        name="cumsum",
        description="cumulative sum along axis 1 (etl.cumsum)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_cumsum_graph,
        numpy_ref=_cumsum_numpy,
        torch_ref=_cumsum_torch,
    ),
    "attention": Example(
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
    ),
}


def list_examples():
    """Return the registered example names (registry order)."""
    return list(_EXAMPLES)


def get_example(name: str) -> Example:
    """Return the :class:`Example` registered under ``name``.

    Raises:
        UnknownExampleError: unknown name — the message lists all available
            names.
    """
    try:
        return _EXAMPLES[name]
    except KeyError:
        available = ", ".join(list_examples())
        raise UnknownExampleError(
            f"unknown example {name!r}; available examples: {available}"
        ) from None


def generate_inputs(example: Example, seed: int = 0):
    """Generate a list of numpy arrays matching ``example.specs``.

    Uses ``numpy.random.default_rng(seed)``: standard-normal draws for
    floating dtypes, small integers for integer dtypes, uniform bools for
    bool specs. Specs must have static integer shapes (``Dim``/``DimExpr``
    shapes are not supported by the harness — explicit error).
    """
    rng = np.random.default_rng(seed)
    arrays = []
    for index, spec in enumerate(example.specs):
        shape = []
        for dim in spec.shape:
            if not isinstance(dim, (int, np.integer)):
                raise ValueError(
                    f"example {example.name!r}: spec {index} has non-static "
                    f"shape dim {dim!r}; bench examples require static "
                    "integer shapes"
                )
            shape.append(int(dim))
        dtype = np.dtype(spec.dtype)
        if np.issubdtype(dtype, np.floating):
            arrays.append(rng.standard_normal(shape).astype(dtype))
        elif np.issubdtype(dtype, np.integer):
            arrays.append(rng.integers(-5, 6, size=shape, dtype=dtype))
        elif dtype == np.dtype("bool"):
            arrays.append(rng.integers(0, 2, size=shape).astype(dtype))
        else:
            raise ValueError(
                f"example {example.name!r}: unsupported spec dtype {dtype} "
                "for input generation"
            )
    return arrays
