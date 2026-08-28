"""Conv-block conformance examples (category "block", tag "conv").

Two ResNet-style conv blocks composed from etl primitives (conv + relu +
residual add + layernorm): a stride-1 basic block and a stride-2
downsampling block with a 1x1 projection skip. The numpy references are the
shared base helpers (:func:`~etl.bench.examples.base._conv2d_numpy` with the
same VALID/SAME TF semantics the etl numpy backend implements — see
``etl/backends/numpy/kernels/linalg.py``), so etl-vs-numpy errors are ~0
(identical kernels); the micro conv examples already pass the strict
conformance defaults and these blocks inherit the same exactness.

Design notes:

- NCHW throughout. ``"SAME"`` padding (TF convention — asymmetric lo/hi
  split when needed) keeps the spatial extent unchanged at stride 1 and
  halves it at stride 2, so the residual add and the 1x1 projection line up
  exactly.
- The stride-2 block's 3x3 SAME padding is ASYMMETRIC (pad ``(0, 1)`` per
  spatial axis), so it deliberately has no torch reference — torch's
  ``conv2d`` only supports symmetric padding and would compute a different
  alignment (a torch ref would need ``F.pad`` shims; kept out on purpose).
- The 1x1 projection uses ``"VALID"`` padding on purpose: 1x1 SAME at
  stride 2 on a 16-wide axis needs NEGATIVE total padding (``(8-1)*2+1-16 =
  -1``), where the shared :func:`~etl.bench.examples.base._conv2d_numpy`
  reference (which clamps the total to zero) and the etl numpy kernel (which
  CROPS ``-lo`` rows/cols off the front — ``etl/backends/numpy/kernels/
  linalg.py``) disagree by a one-pixel shift. VALID 1x1 stride 2 needs no
  padding, yields the same 8x8 extent, and matches the reference exactly.
- Layernorm is built from ``mean``/``sum`` primitives over the channel axis
  (axis 1, NCHW) — etl has no batchnorm op. The numpy reference uses the
  same formula as :func:`~etl.bench.examples.base.layernorm_numpy` but over
  ``axis=1``.
- Sizing: each graph is ~1 MFLOP — a single numpy-backend run is far under
  the ~1 s budget.
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, _conv2d_numpy, register_all


def _layernorm_axis(x, axis):
    """Layer norm over ``axis`` (mean/var from sum primitives, eps 1e-5)."""
    mean = etl.mean(x, axes=axis, keepdims=True)
    diff = etl.subtract(x, mean)
    var = etl.mean(etl.multiply(diff, diff), axes=axis, keepdims=True)
    return etl.divide(diff, etl.sqrt(etl.add(var, 1e-5)))


# --- resnet_conv_block (stride-1 basic block) --------------------------------


@defn
def _resnet_conv_block_graph(x, w1, w2):
    """ResNet basic block: SAME-3x3 conv -> relu -> SAME-3x3 conv -> residual
    add -> layernorm over channels.

    Shapes: x[2,8,16,16], w1[8,8,3,3], w2[8,8,3,3] (all f32). SAME padding at
    stride 1 keeps the spatial extent at 16x16, so the residual add with the
    input lines up; the layernorm normalizes each (n,h,w) location over the
    8 channels (axis 1) — etl has no batchnorm op.
    """
    h = etl.relu(etl.conv(x, w1, strides=1, padding="SAME"))
    y = etl.conv(h, w2, strides=1, padding="SAME")
    return _layernorm_axis(etl.add(y, x), axis=1)


def _resnet_conv_block_numpy(inputs):
    x, w1, w2 = inputs
    h = np.maximum(_conv2d_numpy(x, w1, strides=(1, 1), padding="SAME"), 0.0)
    y = _conv2d_numpy(h, w2, strides=(1, 1), padding="SAME") + x
    mean = y.mean(axis=1, keepdims=True)
    diff = y - mean
    var = (diff * diff).mean(axis=1, keepdims=True)
    return diff / np.sqrt(var + 1e-5)


def _resnet_conv_block_torch(inputs, device=None):
    torch = require_torch()
    x, w1, w2 = (torch.as_tensor(a, device=device) for a in inputs)
    # SAME (stride 1, 3x3) == symmetric padding 1 — matches the etl graph.
    h = torch.relu(torch.nn.functional.conv2d(x, w1, stride=1, padding=1))
    y = torch.nn.functional.conv2d(h, w2, stride=1, padding=1) + x
    mean = y.mean(dim=1, keepdim=True)
    diff = y - mean
    var = (diff * diff).mean(dim=1, keepdim=True)
    return (diff / (var + 1e-5).sqrt()).cpu().numpy()


# --- conv_block_stride2 (downsampling block with 1x1 projection) -------------


@defn
def _conv_block_stride2_graph(x, w1, w2, ws):
    """ResNet downsampling block: SAME-3x3 conv stride 2 -> relu -> SAME-3x3
    conv stride 1 -> residual via VALID-1x1 projection stride 2 -> relu.

    Shapes: x[2,8,16,16], w1[8,8,3,3], w2[8,8,3,3], ws[8,8,1,1] (all f32).
    SAME padding halves the spatial extent to 8x8 in the stride-2 conv leg;
    the 1x1 projection uses VALID padding (no padding needed — stride 2 on a
    16-wide axis subsamples to 8), so the two legs line up for the residual
    add. 1x1 SAME stride 2 would need negative padding (crop) and is
    deliberately avoided (see module docstring).
    """
    h = etl.relu(etl.conv(x, w1, strides=2, padding="SAME"))
    y = etl.conv(h, w2, strides=1, padding="SAME")
    proj = etl.conv(x, ws, strides=2, padding="VALID")
    return etl.relu(etl.add(y, proj))


def _conv_block_stride2_numpy(inputs):
    x, w1, w2, ws = inputs
    h = np.maximum(_conv2d_numpy(x, w1, strides=(2, 2), padding="SAME"), 0.0)
    y = _conv2d_numpy(h, w2, strides=(1, 1), padding="SAME")
    proj = _conv2d_numpy(x, ws, strides=(2, 2), padding="VALID")
    return np.maximum(y + proj, 0.0)


# ---------------------------------------------------------------------------
# Registry (category "block", tag "conv")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="resnet_conv_block",
        description=(
            "ResNet basic block: 3x3-SAME conv + relu + 3x3-SAME conv + "
            "residual add + layernorm over channels (x[2,8,16,16])"
        ),
        specs=(
            TensorSpec((2, 8, 16, 16), _F32),
            TensorSpec((8, 8, 3, 3), _F32),
            TensorSpec((8, 8, 3, 3), _F32),
        ),
        graph=_resnet_conv_block_graph,
        numpy_ref=_resnet_conv_block_numpy,
        torch_ref=_resnet_conv_block_torch,
        # Strict defaults: etl and the numpy ref execute the same conv
        # kernels op-for-op — measured max_abs_error 1.19e-06 (fp32
        # accumulation order in the convs, the micro conv2d precedent); the
        # torch ref differs only by fp32 accumulation order as well.
        category="block",
        tags=("conv",),
    ),
    Example(
        name="conv_block_stride2",
        description=(
            "ResNet downsampling block: 3x3-SAME conv stride 2 + relu + "
            "3x3-SAME conv + residual via 1x1-VALID projection stride 2 "
            "(x[2,8,16,16] -> [2,8,8,8])"
        ),
        specs=(
            TensorSpec((2, 8, 16, 16), _F32),
            TensorSpec((8, 8, 3, 3), _F32),
            TensorSpec((8, 8, 3, 3), _F32),
            TensorSpec((8, 8, 1, 1), _F32),
        ),
        graph=_conv_block_stride2_graph,
        numpy_ref=_conv_block_stride2_numpy,
        # Strict defaults: measured max_abs_error 3.05e-05 (fp32 accumulation
        # order in the stride-2 convs — the 3x3 convs pass exactly like the
        # micro conv2d examples; the residual add of ~1-magnitude values
        # accumulates the small kernel-level noise).
        # No torch ref: SAME stride-2 padding is asymmetric (pad (0,1) per
        # spatial axis), which torch's symmetric conv2d padding cannot
        # express — a torch ref would need F.pad shims (kept out on purpose).
        category="block",
        tags=("conv",),
    ),
])
