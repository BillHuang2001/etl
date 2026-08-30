"""Linear-algebra and scan ops: dot, conv, tril, triu, cumsum, solve,
sort, diagonal, trace, norm, eigh, cholesky, qr, matrix_rank, svd,
matrix_exp.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``dot``: batched matrix multiplication (Nx ``dot`` semantics — last axis of
  ``a`` against second-to-last of ``b``, numpy ``matmul`` shape rules).
- ``conv``: convolution with static configuration; output spatial dims are
  ``DimExpr`` floor-division formulas.
- ``tril``/``triu``/``cumsum``/``solve``/``sort``/``diagonal``: numpy
  semantics; dtypes per numpy (``sort``/``diagonal`` preserve the dtype).
- ``trace``/``norm``: documented compositions over ``diagonal`` + ``reduce``
  and ``sqrt``/``abs`` + ``reduce`` respectively (no hidden semantics).
- ``eigh``/``cholesky``/``qr``/``matrix_rank``/``svd``: numpy ``linalg``
  semantics (numpy dtype rule: int/bool → float64; eigenvalues/singular
  values real at the input's precision); ``matrix_exp`` follows scipy/torch
  semantics — numpy has no ``linalg.matrix_exp`` (pure-numpy reference
  kernel). eigh/qr/svd are multi-result ops (tuple return).
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from etl import core

from . import _utils
from .elementwise import abs, sqrt, square
from .indexing import reshape
from .reductions import reduce_max, reduce_min, reduce_sum

__all__ = ["dot", "conv", "tril", "triu", "cumsum", "solve", "sort",
           "diagonal", "trace", "norm", "eigh", "cholesky", "qr",
           "matrix_rank", "svd", "matrix_exp"]

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


def sort(x, axis=-1) -> "core.SymbolicTensor":
    """Sort in ascending order along an axis (numpy ``sort`` semantics).
    Args:
        x: ``SymbolicTensor``.
        axis: int; sort axis. ``None`` sorts the flattened tensor (numpy
            semantics — implemented as reshape to 1-D, then sort along 0).
    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: axis out of range for the input rank.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if axis is None:
        x_sym = reshape(x_sym, (-1,))
        axis = 0
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"sort: axis must be an int or None, got {axis!r}")
    rank = len(x_sym.shape)
    if rank == 0 or not -rank <= axis < rank:
        raise core.ShapeError(
            f"sort: axis {axis!r} out of range for rank {rank}"
        )
    op = builder.create(
        "sort", operands=(x_sym.value,), attributes={"axis": axis},
        location=loc,
    )
    return _wrap_result(op, loc)


def diagonal(x, offset=0, axis1=0, axis2=1) -> "core.SymbolicTensor":
    """Extract the diagonal of a 2-D slice (numpy ``diagonal`` semantics).
    Args:
        x: ``SymbolicTensor`` of rank >= 2.
        offset: int diagonal offset (0 = main diagonal; positive shifts the
            diagonal up/right, negative down/left).
        axis1: int first diagonal axis.
        axis2: int second diagonal axis (must differ from ``axis1``).
    Returns:
        ``SymbolicTensor`` with the diagonal as a NEW last axis: the two
        diagonal axes are removed (remaining dims keep their order) and the
        diagonal length is appended; dtype preserved.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2; axis1 == axis2; axis out of range.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    for name, value in (("offset", offset), ("axis1", axis1), ("axis2", axis2)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"diagonal: {name} must be an int, got {value!r}"
            )
    rank = len(x_sym.shape)
    if rank < 2:
        raise core.ShapeError(
            f"diagonal: input must have rank >= 2, got rank {rank}"
        )
    if not (-rank <= axis1 < rank) or not (-rank <= axis2 < rank):
        raise core.ShapeError(
            f"diagonal: axis1/axis2 out of range for rank {rank}: "
            f"{axis1!r}/{axis2!r}"
        )
    if axis1 % rank == axis2 % rank:
        raise core.ShapeError("diagonal: axis1 and axis2 must be different")
    op = builder.create(
        "diagonal",
        operands=(x_sym.value,),
        attributes={"offset": offset, "axis1": axis1, "axis2": axis2},
        location=loc,
    )
    return _wrap_result(op, loc)


def trace(x, offset=0) -> "core.SymbolicTensor":
    """Sum of the diagonal of a 2-D slice (numpy ``trace`` semantics).
    Documented composition: ``reduce_sum(diagonal(x, offset, 0, 1),
    axes=(-1,))``.
    Args:
        x: ``SymbolicTensor`` of rank >= 2.
        offset: int diagonal offset (0 = main diagonal).
    Returns:
        ``SymbolicTensor``: scalar for rank-2 input; for higher ranks the
        (0, 1) diagonal is summed and the remaining dims are preserved.
        Dtype per numpy (integer input → ``int64`` via the reduction).
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    diag = diagonal(x_sym, offset=offset, axis1=0, axis2=1)
    return reduce_sum(diag, axes=(-1,), keepdims=False)


def norm(x, axis=None, keepdims=False, ord=2) -> "core.SymbolicTensor":
    """Vector norm (numpy ``linalg.norm`` semantics for vectors).
    Documented composition: ``ord=2`` → ``sqrt(reduce_sum(square(x)))``;
    ``ord=1`` → ``sum(abs(x))``; ``ord=inf`` → ``max(abs(x))``;
    ``ord=-inf`` → ``min(abs(x))`` — no hidden semantics.
    Args:
        x: ``SymbolicTensor``.
        axis: None (all elements), int, or tuple of ints — the axes to
            reduce over.
        keepdims: bool; keep reduced axes as extent 1.
        ord: 2 (default), 1, ``inf``, or ``-inf``. Any other ``ord`` raises
            ``NotImplementedError`` (v1 scope — numpy supports arbitrary
            p-norms, matrix norms, and ``"fro"``).
    Returns:
        ``SymbolicTensor`` — the norm along the given axes. Dtype per numpy:
        integer input → ``float64``, float keeps its dtype.
        NOTE: ``axis=None`` on a rank > 1 input reduces over ALL elements
        (flat-vector norm; ``ord=2`` ≡ Frobenius) — numpy's ``linalg.norm``
        would instead compute MATRIX norms for 2-D input (e.g. spectral for
        ``ord=2``); those are not in v1 scope. Per-axis calls match numpy
        ``vector_norm`` exactly.
    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        NotImplementedError: unsupported ``ord``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if not isinstance(keepdims, bool):
        raise TypeError(f"norm: keepdims must be a bool, got {keepdims!r}")
    if ord not in (1, 2, float("inf"), float("-inf")):
        raise NotImplementedError(
            f"norm: ord={ord!r} is not supported in v1 (supported: 2, 1, "
            "inf, -inf)"
        )
    if ord == 2:
        return sqrt(reduce_sum(square(x_sym), axes=axis, keepdims=keepdims))
    if ord == 1:
        return reduce_sum(abs(x_sym), axes=axis, keepdims=keepdims)
    if ord == float("inf"):
        return reduce_max(abs(x_sym), axes=axis, keepdims=keepdims)
    return reduce_min(abs(x_sym), axes=axis, keepdims=keepdims)


def _wrap_results(op, loc) -> tuple:
    """Wrap ALL of an op's result values in ``SymbolicTensor``s (multi-result
    ops: eigh/qr/svd). Each result's dtype/shape are read back from the IR
    value's inferred type, exactly like ``_wrap_result``."""
    return tuple(
        core.SymbolicTensor(
            value=value,
            dtype=value.type.dtype,
            shape=value.type.shape,
            location=loc,
        )
        for value in op.results
    )


def eigh(x) -> tuple:
    """Hermitian/symmetric eigendecomposition (numpy ``linalg.eigh``).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., n, n)`` (batched).

    Returns:
        ``(w, v)`` tuple of ``SymbolicTensor``s: ``w`` the ascending REAL
        eigenvalues ``(..., n)``, ``v`` the eigenvectors ``(..., n, n)``.
        Dtype rule (numpy linalg): int/bool input → float64; float32 stays
        float32; complex64 → ``w`` float32 / ``v`` complex64.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2; statically non-square last two dims.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create("eigh", operands=(x_sym.value,), location=loc)
    w, v = _wrap_results(op, loc)
    return w, v


def cholesky(x) -> "core.SymbolicTensor":
    """Lower-triangular Cholesky factor (numpy ``linalg.cholesky``).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., n, n)`` (batched), Hermitian
            positive definite.

    Returns:
        ``SymbolicTensor`` of the input's shape; dtype int/bool → float64,
        else preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2; statically non-square last two dims.
        np.linalg.LinAlgError: runtime — matrix is not positive definite
            (numpy's error surfaces as-is).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create("cholesky", operands=(x_sym.value,), location=loc)
    return _wrap_result(op, loc)


def qr(x) -> tuple:
    """QR factorization (numpy ``linalg.qr``, reduced mode).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., m, n)`` (batched; rectangular
            allowed).

    Returns:
        ``(q, r)`` tuple of ``SymbolicTensor``s: ``q (..., m, k)`` and
        ``r (..., k, n)`` with ``k = min(m, n)``. Dtype int/bool → float64,
        else preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create("qr", operands=(x_sym.value,), location=loc)
    q, r = _wrap_results(op, loc)
    return q, r


def matrix_rank(x, tol=None) -> "core.SymbolicTensor":
    """Numerical rank via SVD (numpy ``linalg.matrix_rank``).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., m, n)`` (batched).
        tol: static float threshold; ``None`` (default) = numpy's automatic
            ``max(m, n) * eps * largest-singular-value`` cutoff.

    Returns:
        ``SymbolicTensor`` int64: scalar for 2-D input, ``(...,)`` when
        batched.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2.
        TypeError: ``tol`` neither None nor a float.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    if tol is not None and not isinstance(tol, (float, int)):
        raise TypeError(
            f"matrix_rank: tol must be None or a float, got {tol!r}"
        )
    attributes = {} if tol is None else {"tol": float(tol)}
    op = builder.create(
        "matrix_rank", operands=(x_sym.value,), attributes=attributes,
        location=loc,
    )
    return _wrap_result(op, loc)


def svd(x) -> tuple:
    """Singular value decomposition (numpy ``linalg.svd``,
    ``full_matrices=False``).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., m, n)`` (batched; rectangular
            allowed).

    Returns:
        ``(u, s, vh)`` tuple of ``SymbolicTensor``s: ``u (..., m, k)``,
        ``s (..., k)``, ``vh (..., k, n)`` with ``k = min(m, n)``. ``u``/``vh``
        keep the input dtype (int/bool → float64); ``s`` is REAL at the
        input's precision (complex64 → float32, complex128 → float64).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create("svd", operands=(x_sym.value,), location=loc)
    u, s, vh = _wrap_results(op, loc)
    return u, s, vh


def matrix_exp(x) -> "core.SymbolicTensor":
    """Matrix exponential (scipy ``linalg.expm`` / torch semantics — numpy
    has no ``linalg.matrix_exp``).

    Args:
        x: ``SymbolicTensor`` of shape ``(..., n, n)`` (batched; the last two
            dims must be square).

    Returns:
        ``SymbolicTensor`` of the input's shape; dtype int/bool → float64,
        else preserved. The reference kernel is a pure-numpy
        scaling-and-squaring Taylor implementation (Higham 2005 family).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank < 2; statically non-square last two dims.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create("matrix_exp", operands=(x_sym.value,), location=loc)
    return _wrap_result(op, loc)
