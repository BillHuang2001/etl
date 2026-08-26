"""Linear-algebra and scan ops: dot, conv, tril, triu, cumsum, solve.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``dot``: batched matrix multiplication (Nx ``dot`` semantics — last axis of
  ``a`` against second-to-last of ``b``, numpy ``matmul`` shape rules).
- ``conv``: convolution with static configuration; output spatial dims are
  ``DimExpr`` floor-division formulas.
- ``tril``/``triu``/``cumsum``/``solve``: numpy semantics; dtypes per numpy.
"""
from __future__ import annotations

from typing import Tuple, Union

from etl import core
from etl import ir

from . import _utils

__all__ = ["dot", "conv", "tril", "triu", "cumsum", "solve"]

PaddingSpec = Union[str, int, Tuple[Tuple[int, int], ...]]


def dot(a, b) -> "core.SymbolicTensor":
    """Batched matrix multiplication (``a @ b``).

    Registered as the ``SymbolicTensor.__matmul__`` operator handler (kind
    ``matmul``).

    Args:
        a: ``SymbolicTensor`` or Python scalar, shape ``(..., m, k)``.
        b: ``SymbolicTensor`` or Python scalar, shape ``(..., k, n)``.

    Returns:
        ``SymbolicTensor`` of shape ``broadcast_shapes(batch_a, batch_b) +
        (m, n)``; dtype = ``promote_dtypes(a.dtype, b.dtype)``.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2; static ``k`` mismatch (both ``k`` dims
            concrete and unequal, neither 1).
    """
    raise NotImplementedError


def conv(x, w, strides=1, padding="VALID", input_dilation=1,
         kernel_dilation=1, feature_group_size=1,
         channels_last=False) -> "core.SymbolicTensor":
    """N-dimensional convolution (Nx ``conv`` semantics, NCHW default).

    Input ``x`` has shape ``(N, C_in, *spatial)`` (or ``(N, *spatial, C_in)``
    with ``channels_last=True``); kernel ``w`` has shape ``(C_out,
    C_in // feature_group_size, *kernel_spatial)``.

    Args:
        x: ``SymbolicTensor`` input.
        w: ``SymbolicTensor`` kernel.
        strides: int or per-spatial-dim tuple of ints.
        padding: ``"VALID"``, ``"SAME"``, or per-spatial-dim ``(before,
            after)`` pairs (static ints).
        input_dilation: int or per-spatial-dim tuple.
        kernel_dilation: int or per-spatial-dim tuple.
        feature_group_size: int; must divide ``C_in`` and ``C_out``.
        channels_last: bool; input/output channel axis last.

    Returns:
        ``SymbolicTensor`` of shape ``(N, C_out, *out_spatial)`` where each
        out dim is the ``DimExpr`` formula
        ``(d + 2*pad - kernel_dilation*(k - 1) - 1) // stride + 1``;
        dtype = ``promote_dtypes(x.dtype, w.dtype)``.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank mismatch; static channel/shape
            incompatibilities; bad group size; unknown padding mode.
    """
    raise NotImplementedError


def tril(x, k=0) -> "core.SymbolicTensor":
    """Lower triangle of the last two dims (numpy ``tril`` semantics);
    batch dims are preserved as-is.

    Args:
        x: ``SymbolicTensor`` of rank >= 2.
        k: int diagonal offset (0 = main diagonal).

    Returns:
        ``SymbolicTensor`` with entries above the ``k``-th diagonal zeroed;
        shape and dtype preserved.
    """
    raise NotImplementedError


def triu(x, k=0) -> "core.SymbolicTensor":
    """Upper triangle of the last two dims (numpy ``triu`` semantics); batch
    dims are preserved as-is.

    Args:
        x: ``SymbolicTensor`` of rank >= 2.
        k: int diagonal offset (0 = main diagonal).

    Returns:
        ``SymbolicTensor`` with entries below the ``k``-th diagonal zeroed;
        shape and dtype preserved.
    """
    raise NotImplementedError


def cumsum(x, axis=0, reverse=False) -> "core.SymbolicTensor":
    """Cumulative sum along an axis (numpy ``cumsum`` semantics plus an
    optional reverse scan direction).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: int; scan axis.
        reverse: bool; scan from the end toward the start.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x`` (numpy
        ``cumsum`` dtype rule: bool → int64, else preserved).
    """
    raise NotImplementedError


def solve(a, b) -> "core.SymbolicTensor":
    """Solve linear systems ``a x = b`` (numpy ``linalg.solve`` semantics).

    Args:
        a: ``SymbolicTensor`` of shape ``(..., n, n)``.
        b: ``SymbolicTensor`` of shape ``(..., n)`` or ``(..., n, k)``.

    Returns:
        ``SymbolicTensor`` with ``b``'s shape (with the last axis of ``a``
        consumed); dtype follows numpy (integer inputs → ``float64``, else
        ``promote_dtypes(a.dtype, b.dtype)``).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2 for ``a``; non-square static dims; static
            batch mismatch.
    """
    raise NotImplementedError
