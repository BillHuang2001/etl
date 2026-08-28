"""Second vmap-batch op conformance examples (category "op", tag "vmap").

New vmap variants beyond the legacy ``vectorize`` set: per-sample
elementwise, nested vmap over two UNEQUAL batch extents (level-distinct
``batch`` / ``batch_1`` dims), a shared-weight two-input dot, and the
structural per-sample ops ``pad`` / ``concatenate``.

Design notes (the ``vectorize`` module's conventions, binding here):

- Each example is a per-sample ``@etl.defn`` wrapped via ``etl.vmap(fn,
  in_axes=..., out_axes=0)``; ``example.graph`` is a PLAIN python function
  ``(*batched_specs) -> Graph`` (never an ``@etl.defn``) — the harness
  stages it through the explicit lower/compile/load pipeline.
- v1 mapped axes must be LEADING (``in_axes`` entries in {None, 0}).
  ``out_axes=1`` is DEFERRED in v1 — measured at dev time: calling
  ``etl.vmap(fn, out_axes=1)(spec)`` raises ``TransformError: vmap: mapped
  axis 1 for tensor leaf 0 must be 0 (the leading axis) in v1 —
  non-leading axes are deferred`` — so a ``vmap_out_axes`` example is NOT
  included (it cannot trace; documented deferral, never silent).
- Per-sample rank-2 rows wherever dots appear (``etl.dot`` is batched
  matmul, rank >= 2; rank-1 vectors would need per-sample reshape
  round-trips, which the StableHLO exporter rejects on dynamic dims).
- A gather/select-based per-sample variant was considered and dropped:
  pad and concatenate cover the "structural per-sample op" niche with
  purely static shapes on every axis, so the vectorized graphs stay
  reshape-free and compile on the numpy backend exactly (all max_abs 0.0).
- References are MANUALLY batched numpy implementations — an explicit
  per-sample loop over the batch (never ``np.vectorize`` /
  ``np.apply_along_axis``) — so the auto-vectorization is genuinely
  validated, not re-derived from numpy broadcasting. Torch references use
  plain batched torch ops (lazy import via ``require_torch``).

Dev-time verification (numpy backend, seed 0 inputs): all five examples
match their manual-loop references EXACTLY (max_abs error 0.0 — the
vectorized graphs and the hand-written per-sample loops run the same fp32
kernels in the same order) and pass the strict defaults (rtol=atol=1e-5);
the torch references pass the strict defaults as well.

Nested-vmap semantics (``vmap_nested``): ``etl.vmap(etl.vmap(fn,
in_axes=(0, 0)), in_axes=(0, 0))`` composes mechanically — each level adds
one leading mapped dim, named ``batch`` / ``batch_1`` (level-distinct:
UNEQUAL extents are supported; the numpy interpreter binds symbolic dims
BY NAME, so same-named dims from different nesting levels would collide at
run time). The per-sample fn is a rank-2-row dot ``[4,8] @ [8,4] ->
[4,4]`` over ``[2,3,4,8]`` / ``[2,3,8,4]`` inputs (outer extent 2, inner
extent 3) — forward-only, so no size-one-dim concerns (the reduce-vjp
size-one-dims caveat applies to vmap∘grad, not to forward vmap).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, register_all

# --- vmap_elementwise: per-sample sigmoid [16] over a [32,16] batch ----------
# Simplest possible vmap: a pure elementwise per-sample fn. The vectorized
# graph is identical to a manually batched defn (sigmoid is per-element).


@defn
def _vmap_elementwise_sample(x):
    return etl.sigmoid(x)


def _vmap_elementwise_graph(x_spec):
    return etl.vmap(_vmap_elementwise_sample, in_axes=0, out_axes=0)(x_spec)


def _vmap_elementwise_numpy(inputs):
    (x,) = inputs
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return out


def _vmap_elementwise_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.sigmoid(torch.as_tensor(x, device=device)).cpu().numpy()


# --- vmap_nested: nested vmap over two UNEQUAL batch extents -----------------
# etl.vmap(etl.vmap(fn)) — each level adds one leading mapped dim named
# `batch` / `batch_1` (level-distinct; unequal extents 2 and 3 are fine).
# Per-sample fn: rank-2-row dot [4,8] @ [8,4] -> [4,4] (forward-only).


@defn
def _vmap_nested_sample(a, b):
    return etl.dot(a, b)


def _vmap_nested_graph(a_spec, b_spec):
    return etl.vmap(
        etl.vmap(_vmap_nested_sample, in_axes=(0, 0)), in_axes=(0, 0)
    )(a_spec, b_spec)


def _vmap_nested_numpy(inputs):
    a, b = inputs
    out = np.empty((a.shape[0], a.shape[1], a.shape[2], b.shape[3]),
                   dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = a[i, j] @ b[i, j]
    return out


def _vmap_nested_torch(inputs, device=None):
    torch = require_torch()
    a, b = (torch.as_tensor(t, device=device) for t in inputs)
    # Batched matmul: trailing two dims are the matrix dims, the [2,3] batch
    # dims broadcast — exactly what the nested-vmap'd graph computes.
    return (a @ b).cpu().numpy()


# --- vmap_shared_weights_dot: per-sample row dot with a SHARED weight --------
# in_axes=(0, None): only x is mapped; w stays unmapped and broadcasts.
# Per-sample x is a rank-2 row [1,16] (no per-sample reshape needed).


@defn
def _vmap_shared_weights_dot_sample(x, w):
    return etl.dot(x, w)


def _vmap_shared_weights_dot_graph(x_spec, w_spec):
    return etl.vmap(
        _vmap_shared_weights_dot_sample, in_axes=(0, None), out_axes=0
    )(x_spec, w_spec)


def _vmap_shared_weights_dot_numpy(inputs):
    x, w = inputs
    out = np.empty((x.shape[0], 1, w.shape[1]), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = x[i] @ w
    return out


def _vmap_shared_weights_dot_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(t, device=device) for t in inputs)
    # x [32,1,16] @ w [16,8]: torch matmul broadcasts the batch dims.
    return (x @ w).cpu().numpy()


# --- vmap_pad: per-sample constant pad [16] -> [20] over a [32,16] batch -----
# The pad config is STATIC ((0,4) on the per-sample axis), so the vectorized
# pad pads the last axis only — no dynamic shapes anywhere. (A
# gather/select-based variant was considered and dropped in favor of the
# pad/concat pair: purely static shapes on every axis.)


@defn
def _vmap_pad_sample(x):
    return etl.pad(x, ((0, 4),), 0.0)


def _vmap_pad_graph(x_spec):
    return etl.vmap(_vmap_pad_sample, in_axes=0, out_axes=0)(x_spec)


def _vmap_pad_numpy(inputs):
    (x,) = inputs
    out = np.empty((x.shape[0], x.shape[1] + 4), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.pad(x[i], (0, 4))
    return out


def _vmap_pad_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    # F.pad's (0, 4) pads the LAST dim — same semantics as the per-sample pad.
    return torch.nn.functional.pad(
        torch.as_tensor(x, device=device), (0, 4)
    ).cpu().numpy()


# --- vmap_concat: per-sample concatenate [16]+[16] -> [32] -------------------
# Two mapped inputs, concat along the per-sample axis (axis 0 of the sample,
# i.e. the last axis of the batched tensors).


@defn
def _vmap_concat_sample(a, b):
    return etl.concatenate([a, b], axis=0)


def _vmap_concat_graph(a_spec, b_spec):
    return etl.vmap(_vmap_concat_sample, in_axes=(0, 0), out_axes=0)(
        a_spec, b_spec
    )


def _vmap_concat_numpy(inputs):
    a, b = inputs
    out = np.empty((a.shape[0], a.shape[1] + b.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        out[i] = np.concatenate([a[i], b[i]], axis=0)
    return out


def _vmap_concat_torch(inputs, device=None):
    torch = require_torch()
    a, b = (torch.as_tensor(t, device=device) for t in inputs)
    return torch.cat([a, b], dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "op", tag "vmap")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="vmap_elementwise",
        description=(
            "per-sample sigmoid [16] vmap'd over a [32,16] batch; "
            "matches the manual-loop ref exactly on numpy (max_abs 0.0)"
        ),
        specs=(TensorSpec((32, 16), _F32),),
        graph=_vmap_elementwise_graph,
        numpy_ref=_vmap_elementwise_numpy,
        torch_ref=_vmap_elementwise_torch,
        category="op",
        tags=("vmap",),
    ),
    Example(
        name="vmap_nested",
        description=(
            "nested vmap (vmap∘vmap) over two UNEQUAL batch extents "
            "[2,3,...]: per-sample rank-2-row dot [4,8]@[8,4]"
        ),
        specs=(
            TensorSpec((2, 3, 4, 8), _F32),
            TensorSpec((2, 3, 8, 4), _F32),
        ),
        graph=_vmap_nested_graph,
        numpy_ref=_vmap_nested_numpy,
        torch_ref=_vmap_nested_torch,
        category="op",
        tags=("vmap",),
    ),
    Example(
        name="vmap_shared_weights_dot",
        description=(
            "per-sample row dot [1,16]@[16,8] vmap'd over x only "
            "(shared weight, in_axes=(0, None))"
        ),
        specs=(
            TensorSpec((32, 1, 16), _F32),
            TensorSpec((16, 8), _F32),
        ),
        graph=_vmap_shared_weights_dot_graph,
        numpy_ref=_vmap_shared_weights_dot_numpy,
        torch_ref=_vmap_shared_weights_dot_torch,
        category="op",
        tags=("vmap",),
    ),
    Example(
        name="vmap_pad",
        description=(
            "per-sample constant pad [16]->[20] vmap'd over a [32,16] "
            "batch (static pad config, no dynamic shapes)"
        ),
        specs=(TensorSpec((32, 16), _F32),),
        graph=_vmap_pad_graph,
        numpy_ref=_vmap_pad_numpy,
        torch_ref=_vmap_pad_torch,
        category="op",
        tags=("vmap",),
    ),
    Example(
        name="vmap_concat",
        description=(
            "per-sample concatenate [16]+[16]->[32] vmap'd over two "
            "[32,16] batches"
        ),
        specs=(
            TensorSpec((32, 16), _F32),
            TensorSpec((32, 16), _F32),
        ),
        graph=_vmap_concat_graph,
        numpy_ref=_vmap_concat_numpy,
        torch_ref=_vmap_concat_torch,
        category="op",
        tags=("vmap",),
    ),
])
