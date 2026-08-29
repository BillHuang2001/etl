"""Linear-algebra and scan ops: dot, matmul, conv, tril, triu, cumsum,
cumprod, solve.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``dot``: batched matrix multiplication (Nx ``dot`` semantics — last axis of
  ``a`` against second-to-last of ``b``, numpy ``matmul`` shape rules; rank
  >= 2 only).
- ``matmul``: numpy ``matmul`` semantics (1-D support) — frontend
  composition over ``dot`` via rank-1 promote/squeeze reshapes; batch
  broadcasting falls out of ``dot``. ``SymbolicTensor.__matmul__`` still
  routes to ``dot``.
- ``conv``: convolution with static configuration; output spatial dims are
  ``DimExpr`` floor-division formulas.
- ``tril``/``triu``/``cumsum``/``cumprod``/``solve``: numpy semantics; dtypes
  per numpy (bool → int64 for the cumulative scans).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from etl import core

from . import _utils

__all__ = ["dot", "matmul", "conv", "tril", "triu", "cumsum", "cumprod", "solve"]

PaddingSpec = Union[str, int, Tuple[Tuple[int, int], ...]]


def _wrap_result(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result value in a ``SymbolicTensor``.

    The dtype/shape are READ BACK from the IR value's inferred type (never
    computed independently): ``ir.verify`` re-checks that the op's declared
    result type agrees with its shape-inference hook, so a frontend-side
    recomputation could only disagree with the IR.
    """
    result = op.result
    return core.SymbolicTensor(
        value=result,
        dtype=result.type.dtype,
        shape=result.type.shape,
        location=loc,
    )


def _as_binary_operands(a, b, loc):
    """Normalize two operands with mutual weak-scalar dtype hints.

    At least one operand must already be a ``SymbolicTensor``; a Python scalar
    on either side is promoted to a 0-d Constant op whose dtype follows NEP-50
    weak promotion against the *other* side (``_utils.weak_scalar_dtype``),
    e.g. ``int + float32-tensor → float32`` but ``float + float32-tensor →
    float64``. Concrete ``Tensor`` operands raise ``TraceError`` via
    ``as_operand``.
    """
    a_symbolic = isinstance(a, core.SymbolicTensor)
    b_symbolic = isinstance(b, core.SymbolicTensor)
    if a_symbolic:
        if b_symbolic:
            return a, b
        return a, _utils.as_operand(b, dtype_hint=a.dtype, location=loc)
    if b_symbolic:
        return _utils.as_operand(a, dtype_hint=b.dtype, location=loc), b
    raise core.TraceError(
        "both operands are Python scalars — at least one operand of a "
        "binary op must be a SymbolicTensor"
    )


def _broadcast_op(builder, x, target_shape, loc) -> "core.SymbolicTensor":
    """Build a ``broadcast`` op to ``target_shape`` (compensation helpers)."""
    op = builder.create(
        "broadcast",
        operands=(x.value,),
        attributes={"shape": tuple(target_shape)},
        location=loc,
    )
    return _wrap_result(op, loc)


def _transpose_op(builder, x, permutation, loc) -> "core.SymbolicTensor":
    """Build a ``transpose`` op with the given axis permutation."""
    op = builder.create(
        "transpose",
        operands=(x.value,),
        attributes={"permutation": tuple(permutation)},
        location=loc,
    )
    return _wrap_result(op, loc)


def _cast_op(builder, x, target_dtype, loc) -> "core.SymbolicTensor":
    """Build a ``cast`` op to ``target_dtype`` (compensation helpers)."""
    op = builder.create(
        "cast",
        operands=(x.value,),
        attributes={"dtype": target_dtype},
        location=loc,
    )
    return _wrap_result(op, loc)


def _per_spatial(value, name: str, n_spatial: int) -> Tuple[int, ...]:
    """Normalize an int-or-per-spatial-tuple conv parameter.

    A bare int replicates over all spatial dims; a tuple/list must have
    exactly ``n_spatial`` entries. Entry positivity is validated by the IR's
    ``infer_conv`` (``ShapeError``).

    Raises:
        TypeError: neither an int nor a tuple/list.
        core.ShapeError: tuple/list length != ``n_spatial``.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,) * n_spatial
    if isinstance(value, (tuple, list)):
        seq = tuple(value)
        if len(seq) != n_spatial:
            raise core.ShapeError(
                f"conv: {name} must have {n_spatial} entries, "
                f"got {len(seq)}"
            )
        return seq
    raise TypeError(
        f"conv: {name} must be an int or a tuple of {n_spatial} ints, "
        f"got {value!r}"
    )


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    a_sym, b_sym = _as_binary_operands(a, b, loc)
    shape_a, shape_b = a_sym.shape, b_sym.shape
    if len(shape_a) < 2 or len(shape_b) < 2:
        raise core.ShapeError(
            f"dot: operands must have rank >= 2 (batched matmul), got "
            f"ranks {len(shape_a)} and {len(shape_b)}"
        )
    # The IR's k-dim contract requires strict equality (with symbolic
    # deferral), while the ops contract allows a size-1 k dim to broadcast
    # (numpy matmul semantics). Compensate frontend-side (ir is not editable
    # from this node): expand a size-1 k dim to the other side's k dim.
    # Batch dims broadcast automatically inside the IR's infer_dot.
    ka, kb = shape_a[-1], shape_b[-2]
    if ka == 1 and kb != 1:
        a_sym = _broadcast_op(
            builder, a_sym, shape_a[:-2] + (shape_a[-2], kb), loc
        )
    elif kb == 1 and ka != 1:
        b_sym = _broadcast_op(
            builder, b_sym, shape_b[:-2] + (ka, shape_b[-1]), loc
        )
    # Both static and unequal (neither 1) → infer_dot raises ShapeError.
    op = builder.create(
        "dot", operands=(a_sym.value, b_sym.value), location=loc
    )
    return _wrap_result(op, loc)


def _reshape_op(builder, x, shape, loc) -> "core.SymbolicTensor":
    """Build a ``reshape`` op to ``shape`` (compensation helpers)."""
    op = builder.create(
        "reshape",
        operands=(x.value,),
        attributes={"shape": tuple(shape)},
        location=loc,
    )
    return _wrap_result(op, loc)


def matmul(a, b) -> "core.SymbolicTensor":
    """Matrix product with numpy ``matmul`` semantics (1-D support).

    Frontend composition over :func:`dot` (which keeps its existing
    rank >= 2 contract — ``SymbolicTensor.__matmul__`` still routes to
    ``dot``): rank-1 operands are promoted to rank-2 (``(1, k)`` /
    ``(k, 1)`` via ``reshape``), multiplied, and the result is squeezed back
    to the numpy rank-1 shape. Batch broadcasting (including a vector
    against a batched matrix) falls out of ``dot``'s IR ``infer_dot``
    contract. See the design note in this node's CONTEXT.md.

    Args:
        a: ``SymbolicTensor`` or Python scalar, shape ``(..., m, k)`` or
            ``(k,)``.
        b: ``SymbolicTensor`` or Python scalar, shape ``(..., k, n)`` or
            ``(k,)``.

    Returns:
        ``SymbolicTensor`` per numpy matmul: vector@vector → 0-d scalar;
        vector@matrix / matrix@vector → 1-D; else batched ``dot`` shape.
        Dtype = ``promote_dtypes(a.dtype, b.dtype)``.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank-0 operand (numpy raises ``ValueError``);
            static ``k`` mismatch; incompatible batch dims.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    a_sym, b_sym = _as_binary_operands(a, b, loc)
    rank_a, rank_b = len(a_sym.shape), len(b_sym.shape)
    if rank_a == 0 or rank_b == 0:
        raise core.ShapeError(
            f"matmul: operands must have rank >= 1, got ranks {rank_a} and "
            f"{rank_b} (numpy matmul does not accept scalars)"
        )
    if rank_a == 1:
        a_sym = _reshape_op(builder, a_sym, (1,) + a_sym.shape, loc)
    if rank_b == 1:
        b_sym = _reshape_op(builder, b_sym, b_sym.shape + (1,), loc)
    result = dot(a_sym, b_sym)
    # Squeeze the promoted axes back to numpy's rank-1 result shapes. The
    # promoted axis sits at position len(batch_dims) (vector@matrix) or the
    # last position (matrix@vector); slicing from both ends is batch-safe.
    if rank_a == 1 and rank_b == 1:
        return _reshape_op(builder, result, (), loc)
    if rank_a == 1:
        return _reshape_op(
            builder, result, result.shape[:-2] + (result.shape[-1],), loc
        )
    if rank_b == 1:
        return _reshape_op(builder, result, result.shape[:-1], loc)
    return result


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym, w_sym = _as_binary_operands(x, w, loc)
    if not isinstance(channels_last, bool):
        raise TypeError(
            f"conv: channels_last must be a bool, got {channels_last!r}"
        )
    if not isinstance(feature_group_size, int) or isinstance(
        feature_group_size, bool
    ):
        raise TypeError(
            f"conv: feature_group_size must be an int, got "
            f"{feature_group_size!r}"
        )
    rank = len(x_sym.shape)
    if rank < 3:
        raise core.ShapeError(
            f"conv: input rank must be >= 3 (N, C, *spatial), got {rank}"
        )
    if len(w_sym.shape) != rank:
        raise core.ShapeError(
            f"conv: input and kernel ranks differ: {rank} vs "
            f"{len(w_sym.shape)}"
        )
    n_spatial = rank - 2
    # Static feature-group check the IR does not perform: C_out must be
    # divisible by the group size (C_in divisibility and the kernel channel
    # contract are enforced by the IR's infer_conv).
    c_out = w_sym.shape[0]
    if isinstance(c_out, int) and c_out % feature_group_size != 0:
        raise core.ShapeError(
            f"conv: out_channels {c_out} not divisible by "
            f"feature_group_size {feature_group_size}"
        )
    # channels_last is frontend sugar: transpose to NCHW, convolve, transpose
    # the result back. The IR conv op itself is NCHW-only.
    if channels_last:
        to_nchw = (0, rank - 1) + tuple(range(1, rank - 1))
        x_sym = _transpose_op(builder, x_sym, to_nchw, loc)
    attributes = {
        "strides": _per_spatial(strides, "strides", n_spatial),
        "padding": padding,
        "input_dilation": _per_spatial(
            input_dilation, "input_dilation", n_spatial
        ),
        "kernel_dilation": _per_spatial(
            kernel_dilation, "kernel_dilation", n_spatial
        ),
        "feature_group_count": feature_group_size,
        "batch_group_count": 1,
    }
    op = builder.create(
        "conv",
        operands=(x_sym.value, w_sym.value),
        attributes=attributes,
        location=loc,
    )
    result = _wrap_result(op, loc)
    if channels_last:
        to_channels_last = (0,) + tuple(range(2, rank)) + (1,)
        result = _transpose_op(builder, result, to_channels_last, loc)
    return result


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"tril: k must be an int, got {k!r}")
    if len(x_sym.shape) < 2:
        raise core.ShapeError(
            f"tril: input must have rank >= 2, got rank {len(x_sym.shape)}"
        )
    op = builder.create(
        "tril", operands=(x_sym.value,), attributes={"k": k}, location=loc
    )
    return _wrap_result(op, loc)


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"triu: k must be an int, got {k!r}")
    if len(x_sym.shape) < 2:
        raise core.ShapeError(
            f"triu: input must have rank >= 2, got rank {len(x_sym.shape)}"
        )
    op = builder.create(
        "triu", operands=(x_sym.value,), attributes={"k": k}, location=loc
    )
    return _wrap_result(op, loc)


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(reverse, bool):
        raise TypeError(f"cumsum: reverse must be a bool, got {reverse!r}")
    rank = len(x_sym.shape)
    if rank == 0:
        axis_norm = 0  # scalar cumsum: no axis to validate
    else:
        axis_norm = _utils.normalize_axes(axis, rank)[0]
    # numpy cumsum dtype rule: bool → int64. The IR cumsum preserves the
    # operand dtype (infer_identity), so compensate frontend-side by casting
    # bool inputs to int64 first; all other dtypes are preserved.
    if x_sym.dtype == np.dtype("bool"):
        x_sym = _cast_op(builder, x_sym, np.dtype("int64"), loc)
    op = builder.create(
        "cumsum",
        operands=(x_sym.value,),
        attributes={"axis": axis_norm, "reverse": reverse},
        location=loc,
    )
    return _wrap_result(op, loc)


def cumprod(x, axis=0, reverse=False) -> "core.SymbolicTensor":
    """Cumulative product along an axis (numpy ``cumprod`` semantics plus an
    optional reverse scan direction, mirroring :func:`cumsum`).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axis: int; scan axis.
        reverse: bool; scan from the end toward the start.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x`` (numpy
        ``cumprod`` dtype rule: bool → int64, else preserved).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(reverse, bool):
        raise TypeError(f"cumprod: reverse must be a bool, got {reverse!r}")
    rank = len(x_sym.shape)
    if rank == 0:
        axis_norm = 0  # scalar cumprod: no axis to validate
    else:
        axis_norm = _utils.normalize_axes(axis, rank)[0]
    # numpy cumprod dtype rule: bool → int64 (mirror of cumsum). The IR
    # cumprod preserves the operand dtype (infer_identity), so cast bool
    # inputs to int64 frontend-side.
    if x_sym.dtype == np.dtype("bool"):
        x_sym = _cast_op(builder, x_sym, np.dtype("int64"), loc)
    op = builder.create(
        "cumprod",
        operands=(x_sym.value,),
        attributes={"axis": axis_norm, "reverse": reverse},
        location=loc,
    )
    return _wrap_result(op, loc)


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    a_sym, b_sym = _as_binary_operands(a, b, loc)
    # The IR's infer_solve enforces a rank >= 2, b rank >= 1, squareness,
    # batch broadcasting, and the numpy dtype rule (int/bool → float64, else
    # promotion). No attrs: left_side defaults to True.
    op = builder.create(
        "solve", operands=(a_sym.value, b_sym.value), location=loc
    )
    return _wrap_result(op, loc)
