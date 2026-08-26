"""Reduction ops and the user-facing reduction sugar.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``axes`` (reduce_*) / ``axis`` (argmax/argmin) are STATIC: ``None`` = all
  axes, int, or tuple of ints; normalized via ``_utils.normalize_axes``.
- Output shape via ``_utils.reduced_shape`` (symbolic dims preserved for
  unreduced axes).
- ``reduce_mean`` follows numpy: integer/bool input → ``float64`` result;
  float keeps dtype.
- ``argmax``/``argmin`` produce ``int64`` index tensors (numpy convention).
- The sugar functions (``sum``/``max``/``min``/``mean``/``prod``) are
  DOCUMENTED SHORTHAND ONLY — their exact expansion onto ``reduce_*`` is
  stated in each docstring; they add no hidden semantics.
"""
from __future__ import annotations

from etl import core
from etl import ir

from . import _utils

__all__ = [
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod",
    "sum", "max", "min", "mean", "prod", "argmax", "argmin",
]


def reduce_sum(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Sum of elements over the given axes.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axes: ``None`` (all axes → scalar), int, or tuple of ints.
        keepdims: Keep reduced axes as extent-1 dims.

    Returns:
        ``SymbolicTensor`` with reduced shape; dtype = numpy ``sum`` result
        dtype (bool → int64, else preserved; see numpy).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range.
    """
    raise NotImplementedError


def reduce_max(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Maximum of elements over the given axes.

    Dtype preserved; shape reduced. Args as in :func:`reduce_sum`.
    """
    raise NotImplementedError


def reduce_min(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Minimum of elements over the given axes.

    Dtype preserved; shape reduced. Args as in :func:`reduce_sum`.
    """
    raise NotImplementedError


def reduce_mean(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Arithmetic mean of elements over the given axes.

    Dtype rule (numpy): integer/bool input → ``float64`` result; float input
    keeps its dtype. Shape reduced. Args as in :func:`reduce_sum`.
    """
    raise NotImplementedError


def reduce_prod(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """Product of elements over the given axes.

    Dtype rule (numpy): bool input → int64, else preserved. Shape reduced.
    Args as in :func:`reduce_sum`.
    """
    raise NotImplementedError


def sum(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: ``sum(x, axes, keepdims)`` is EXACTLY
    ``reduce_sum(x, axes=axes, keepdims=keepdims)`` — no other semantics.
    (Shadows the Python builtin inside this module on purpose.)"""
    raise NotImplementedError


def max(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_max(x, axes=axes,
    keepdims=keepdims)``. (Shadows the Python builtin inside this module on
    purpose.)"""
    raise NotImplementedError


def min(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_min(x, axes=axes,
    keepdims=keepdims)``. (Shadows the Python builtin inside this module on
    purpose.)"""
    raise NotImplementedError


def mean(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_mean(x, axes=axes,
    keepdims=keepdims)``. Integer/bool input → float64 (see
    :func:`reduce_mean`)."""
    raise NotImplementedError


def prod(x, axes=None, keepdims=False) -> "core.SymbolicTensor":
    """User-facing sugar: EXACTLY ``reduce_prod(x, axes=axes,
    keepdims=keepdims)``."""
    raise NotImplementedError


def argmax(x, axis=None, keepdims=False) -> "core.SymbolicTensor":
    """Index of the maximum along an axis (numpy ``argmax`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: ``None`` (flatten-and-reduce over all axes, yielding a scalar
            index — consistent with ``reduce_*``'s ``axes=None``), or an int.
        keepdims: Keep the reduced axis as extent 1.

    Returns:
        ``SymbolicTensor`` of dtype ``int64``; reduced shape.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range.
    """
    raise NotImplementedError


def argmin(x, axis=None, keepdims=False) -> "core.SymbolicTensor":
    """Index of the minimum along an axis (numpy ``argmin`` semantics).

    Dtype ``int64``; rules identical to :func:`argmax`.
    """
    raise NotImplementedError
