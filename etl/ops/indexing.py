"""Indexing and shape-manipulation ops.

NOTE: this module shadows the builtin ``slice`` with its op function; use
``builtins.slice`` when a Python slice object is needed inside this module.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``slice``/``gather``/``scatter``/``concatenate``/``pad``/``transpose``/
  ``reshape``/``broadcast`` produce NEW SSA values — etl has no in-place
  tensor mutation.
- All index/shape parameters are STATIC Python values (ints, slices, tuples
  of ints, ``Dim``/``DimExpr`` where a symbolic extent is meaningful —
  e.g. ``reshape`` output dims, ``broadcast`` target dims, ``slice``
  ``lengths``). A ``SymbolicTensor`` index anywhere raises ``TraceError``.
- Output shapes are computed statically via ``DimExpr`` arithmetic; the
  numpy backend enforces exact runtime semantics.
"""
from __future__ import annotations

from typing import Tuple, Union

from etl import core
from etl import ir

from . import _utils

__all__ = [
    "broadcast", "reshape", "transpose", "slice", "gather", "scatter",
    "concatenate", "pad", "getitem",
]


def broadcast(x, shape) -> "core.SymbolicTensor":
    """Broadcast to the target shape (numpy ``broadcast_to`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        shape: Tuple of ``int``/``Dim``/``DimExpr``. Target rank may exceed
            ``x``'s rank (new dims prepended). Per aligned dim: ``x`` dim of
            ``1`` expands to the target; equal dims stay; otherwise
            ``core.ShapeError`` (static) — symbolic conflicts defer to
            ``DimExpr`` equality and runtime enforcement.

    Returns:
        ``SymbolicTensor`` of the target shape; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: static incompatibility with the target shape.
        TypeError: ``shape`` is not a tuple of shape elements.
    """
    raise NotImplementedError


def reshape(x, shape) -> "core.SymbolicTensor":
    """Reshape to the given shape.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        shape: Tuple of ``int``/``Dim``/``DimExpr``; at most ONE element may
            be ``-1`` (inferred from the element count via ``DimExpr``
            arithmetic).

    Returns:
        ``SymbolicTensor`` with the given (or inferred) shape; dtype
        preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: more than one ``-1``; static element-count
            mismatch; negative dims other than ``-1``.
    """
    raise NotImplementedError


def transpose(x, axes=None) -> "core.SymbolicTensor":
    """Permute tensor axes (numpy ``transpose`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axes: ``None`` (reverse all axes) or a permutation tuple of
            ``len(x.shape)`` ints.

    Returns:
        ``SymbolicTensor`` with axes permuted; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: ``axes`` is not a valid permutation.
    """
    raise NotImplementedError


def slice(x, start, lengths, strides=1) -> "core.SymbolicTensor":
    """Static strided slice (Nx ``slice`` semantics).

    Args:
        x: ``SymbolicTensor``.
        start: int or per-axis tuple of ints (scalar broadcast to all axes).
        lengths: int/``Dim``/``DimExpr`` or per-axis tuple thereof.
        strides: int or per-axis tuple of ints; must be positive.

    Returns:
        ``SymbolicTensor`` of shape ``lengths`` (dims as given); dtype
        preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: static out-of-bounds; negative or zero strides;
            arity mismatch.
    """
    raise NotImplementedError


def gather(x, indices, axis=0) -> "core.SymbolicTensor":
    """Gather entries along an axis (numpy ``take`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        indices: ``SymbolicTensor`` of integer dtype (int32/int64), arbitrary
            shape.
        axis: int; the axis of ``x`` being indexed.

    Returns:
        ``SymbolicTensor`` with shape ``x.shape[:axis] + indices.shape +
        x.shape[axis + 1:]``; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand;
            symbolic indices.
        core.DTypeError: ``indices`` is not an integer dtype.
        core.ShapeError: ``axis`` out of range.
    """
    raise NotImplementedError


def scatter(x, indices, updates, axis=0) -> "core.SymbolicTensor":
    """Scatter ``updates`` into a COPY of ``x`` at ``indices`` along an axis
    (numpy ``put``-along-axis / JAX ``scatter``-update semantics; ``x`` is
    never mutated).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        indices: ``SymbolicTensor`` of integer dtype.
        updates: ``SymbolicTensor`` (or Python scalar), cast to ``x.dtype``.
        axis: int; the axis being scattered along.

    Returns:
        ``SymbolicTensor`` with ``x``'s exact shape and dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.DTypeError: ``indices`` is not an integer dtype.
        core.ShapeError: ``axis`` out of range; static shape incompatibility
            between ``indices`` and ``updates``.
    """
    raise NotImplementedError


def concatenate(tensors, axis=0) -> "core.SymbolicTensor":
    """Concatenate tensors along an axis (numpy ``concatenate`` semantics).

    Args:
        tensors: Non-empty list/tuple of ``SymbolicTensor`` (or Python
            scalars, treated as 0-d) of equal rank.
        axis: int; the axis along which to join.

    Returns:
        ``SymbolicTensor`` with the ``axis`` dim equal to the ``DimExpr`` SUM
        of input axis dims; other dims unchanged. Dtype = ``promote_dtypes``
        of all inputs.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: empty input; rank mismatch; static mismatch on a
            non-axis dim; ``axis`` out of range.
    """
    raise NotImplementedError


def pad(x, config, value=0) -> "core.SymbolicTensor":
    """Pad a tensor with a constant value (Nx ``pad`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        config: Per-axis padding spec: a tuple of length ``rank`` whose
            entries are either ``int`` (symmetric pad) or ``(before, after)``
            pairs of non-negative ints.
        value: Python scalar (cast to ``x.dtype``); the fill value.

    Returns:
        ``SymbolicTensor`` with each dim ``d + before + after`` (``DimExpr``
        arithmetic); dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: malformed config; negative padding; arity mismatch.
    """
    raise NotImplementedError


def getitem(x, key) -> "core.SymbolicTensor":
    """``x[key]`` — static indexing entry point (operator handler kind
    ``getitem``, registered by ``_registration``).

    Strictly STATIC indexing, mapped onto ``slice``/``gather`` ops:

    - ``int`` index → ``slice`` op that drops the axis.
    - ``slice`` object (contiguous) → ``slice`` op.
    - tuple of ints/slices → per-axis combination of the above.
    - strided slices → ``gather`` with explicit index arrays.

    NOT supported (raises ``core.TraceError``): ``SymbolicTensor`` indices
    (runtime control flow on indexing is graph semantics — use ``etl.cond``
    explicitly), boolean masks, ``None``/newaxis, ellipsis.

    Args:
        x: ``SymbolicTensor``.
        key: int, ``builtins.slice``, or tuple of ints/slices.

    Returns:
        ``SymbolicTensor``; dtype preserved.

    Raises:
        core.TraceError: no active trace; symbolic index; unsupported key.
    """
    raise NotImplementedError
