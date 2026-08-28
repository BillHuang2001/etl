"""Matmul-variant op conformance examples (category "op", tag "basic").

Six ``etl.dot`` variants — all static-shape, all passing the STRICT global
defaults on the numpy backend (measured max-abs error 0.0 for five of the
six; ``matmul_3d_2d_shared`` 1.4e-06 — fp32 accumulation-order noise between
``np.matmul``'s broadcast path and ``np.einsum``'s summation order, still
~7x below the strict ``atol=rtol=1e-5`` bound).

Op coverage (etl op -> example):
  - dot, batched matmul, matching batch dims            -> batched_3d_matmul
  - dot, batch dims broadcast over a rank-2 shared w    -> matmul_3d_2d_shared
  - dot + reshape (matvec; rank-1 rhs is a ShapeError)  -> matvec
  - dot + transpose (x @ w.T)                           -> matmul_transposed
  - dot composed twice over a shared square operand     -> matmul_square
  - dot + multiply (diagonal scaling: (x @ w) * v,
    the "x @ diag(v)" form)                             -> diagonal_matmul

Notes:
  - ``etl.dot`` requires BOTH operands rank >= 2 (batched matmul): a rank-1
    rhs raises ``core.ShapeError`` ("dot: operands must have rank >= 2 ...").
    ``matvec`` therefore lowers the vector to ``[K, 1]``, dots, and reshapes
    back to ``[M]`` — reshape is fine on the numpy backend (the reshape-free
    rule only constrains compiler-backend export of DYNAMIC shapes; all
    shapes here are static).
  - The numpy backend shares its matmul kernel with the numpy reference
    (``np.matmul`` / einsum in fp32), so no per-example tolerance overrides
    are needed — strict defaults pass exactly.
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, register_all

# --- batched_3d_matmul -------------------------------------------------------
# [B,M,K] @ [B,K,N] — batched matmul with matching batch dims.
# numpy backend: exact (max_abs 0.0).


@defn
def _bm3d_graph(x, w):
    return etl.dot(x, w)


def _bm3d_numpy(inputs):
    x, w = inputs
    return x @ w


def _bm3d_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w).cpu().numpy()


# --- matmul_3d_2d_shared -----------------------------------------------------
# [B,M,K] @ [K,N] — rank-mismatched shared-weight batched dot (the
# vmap_linear-style dot written directly): batch dims of the 3D lhs broadcast
# over the rank-2 rhs. Reference via einsum (np.matmul's broadcast path also
# matches; einsum is the explicit documented form).
# numpy backend: max_abs 1.4e-06 (accumulation-order noise, see docstring).


@defn
def _bm3d2d_graph(x, w):
    return etl.dot(x, w)


def _bm3d2d_numpy(inputs):
    x, w = inputs
    return np.einsum("bmk,kn->bmn", x, w)


def _bm3d2d_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w).cpu().numpy()


# --- matvec ------------------------------------------------------------------
# [M,K] @ [K,1] -> reshape [M]. etl.dot rejects a rank-1 rhs (ShapeError,
# documented); the vector is lifted to a column, dotted, and reshaped back.
# numpy reference: the natural x @ w (rank-1 rhs, numpy matmul semantics).
# numpy backend: exact (max_abs 0.0).


@defn
def _matvec_graph(x, w):
    col = etl.dot(x, etl.reshape(w, (w.shape[0], 1)))
    return etl.reshape(col, (x.shape[0],))


def _matvec_numpy(inputs):
    x, w = inputs
    return x @ w


def _matvec_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w).cpu().numpy()


# --- matmul_transposed -------------------------------------------------------
# x @ w.T via etl.dot(x, etl.transpose(w)) — the transpose-before-dot pattern
# used by attention's QK^T.
# numpy backend: exact (max_abs 0.0).


@defn
def _matmul_transposed_graph(x, w):
    return etl.dot(x, etl.transpose(w))


def _matmul_transposed_numpy(inputs):
    x, w = inputs
    return x @ w.T


def _matmul_transposed_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return (x @ w.T).cpu().numpy()


# --- matmul_square -----------------------------------------------------------
# (x @ w) @ x on [64,64] squares — dot composed twice with the SAME lhs
# operand reused (SSA reuse) and a square output; extends micro's single-dot
# ``matmul`` by exercising a dot-of-dot chain.
# numpy backend: exact (max_abs 0.0).


@defn
def _matmul_square_graph(x, w):
    return etl.dot(etl.dot(x, w), x)


def _matmul_square_numpy(inputs):
    x, w = inputs
    return (x @ w) @ x


def _matmul_square_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.as_tensor(a, device=device) for a in inputs)
    return ((x @ w) @ x).cpu().numpy()


# --- diagonal_matmul ---------------------------------------------------------
# (x @ w) * v — a diagonal vector broadcast over a matmul output, i.e. the
# "x @ diag(v)" column-scaling form: out[i,j] = sum_k x[i,k] w[k,j] * v[j].
# numpy backend: exact (max_abs 0.0).


@defn
def _diagonal_matmul_graph(x, w, v):
    return etl.multiply(etl.dot(x, w), v)


def _diagonal_matmul_numpy(inputs):
    x, w, v = inputs
    return (x @ w) * v


def _diagonal_matmul_torch(inputs, device=None):
    torch = require_torch()
    x, w, v = (torch.as_tensor(a, device=device) for a in inputs)
    return ((x @ w) * v).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "op", tag "basic"). Strict defaults — no per-example
# tolerance overrides (numpy backend is exact; see the module docstring).
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="batched_3d_matmul",
        description="[B,M,K] x [B,K,N] batched matmul (etl.dot, matching batch dims)",
        specs=(
            TensorSpec((4, 8, 16), _F32),
            TensorSpec((4, 16, 32), _F32),
        ),
        graph=_bm3d_graph,
        numpy_ref=_bm3d_numpy,
        torch_ref=_bm3d_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="matmul_3d_2d_shared",
        description="[B,M,K] x [K,N] shared-weight batched dot (batch broadcast)",
        specs=(
            TensorSpec((4, 8, 16), _F32),
            TensorSpec((16, 32), _F32),
        ),
        graph=_bm3d2d_graph,
        numpy_ref=_bm3d2d_numpy,
        torch_ref=_bm3d2d_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="matvec",
        description="matvec [M,K] x [K] via [K,1] dot + reshape (rank-1 rhs is a ShapeError)",
        specs=(
            TensorSpec((8, 16), _F32),
            TensorSpec((16,), _F32),
        ),
        graph=_matvec_graph,
        numpy_ref=_matvec_numpy,
        torch_ref=_matvec_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="matmul_transposed",
        description="x @ w.T via etl.dot(x, etl.transpose(w))",
        specs=(
            TensorSpec((8, 16), _F32),
            TensorSpec((32, 16), _F32),
        ),
        graph=_matmul_transposed_graph,
        numpy_ref=_matmul_transposed_numpy,
        torch_ref=_matmul_transposed_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="matmul_square",
        description="chained square matmul (x @ w) @ x on [64,64]",
        specs=(
            TensorSpec((64, 64), _F32),
            TensorSpec((64, 64), _F32),
        ),
        graph=_matmul_square_graph,
        numpy_ref=_matmul_square_numpy,
        torch_ref=_matmul_square_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="diagonal_matmul",
        description="diagonal scaling of a matmul output: (x @ w) * v (x @ diag(v))",
        specs=(
            TensorSpec((8, 16), _F32),
            TensorSpec((16, 8), _F32),
            TensorSpec((8,), _F32),
        ),
        graph=_diagonal_matmul_graph,
        numpy_ref=_diagonal_matmul_numpy,
        torch_ref=_diagonal_matmul_torch,
        category="op",
        tags=("basic",),
    ),
])
