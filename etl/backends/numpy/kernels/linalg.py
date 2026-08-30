"""Linear-algebra kernels: dot, conv, solve.

Semantics source of truth (READ before editing this module):

- ``etl/ops/linalg.py`` — frontend conventions: ``dot`` is batched matmul
  (numpy ``matmul`` contract; ops already emits any size-1-k broadcast
  compensation into the graph as explicit ``broadcast`` ops — the kernel
  executes the graph as-is); ``conv`` is NCHW (``channels_last`` is frontend
  sugar: the graph transposes around the op); ``solve`` follows numpy
  ``linalg.solve`` (``left_side`` defaults to True; ops never emits the attr).
- ``etl/ir/op_defs/linalg.py`` — attribute schemas (``conv.strides``,
  ``conv.padding`` = ``"VALID"``/``"SAME"``/per-spatial ``(lo, hi)`` pairs,
  ``conv.input_dilation``/``kernel_dilation``/``feature_group_count``/
  ``batch_group_count``; ``solve.left_side``).
- ``etl/ir/inference.py`` (``infer_dot``/``infer_conv``/``infer_solve``) — the
  DECLARED result types the interpreter validates kernel outputs against.
  Result dtype must match ``op.results[i].type.dtype`` EXACTLY and the
  concrete shape must match the evaluated symbolic shape — kernels never
  silently coerce.

conv semantics implemented here (exactly per ``_conv_out_dim``):

- Input ``(N, C_in, *spatial)``, kernel ``(C_out, C_in // feature_groups,
  *kernel_spatial)``, output ``(N, C_out * batch_groups, *out_spatial)``.
- Effective sizes follow numpy convolution: ``eff = (d - 1) * dil + 1`` for
  BOTH the input (input dilation inserts zeros) and the kernel (kernel
  dilation inserts zeros).
- Order of operations per the ``_conv_out_dim`` formula
  ``(eff_d + lo + hi - eff_k) // stride + 1``: the input is dilated FIRST,
  then padded with ``lo``/``hi`` zeros on the dilated input (the formula's
  padded extent is ``eff_d + lo + hi``, which only holds for pad-after-dilate).
- ``"VALID"``: ``lo = hi = 0``. ``"SAME"`` (TF convention): output dim
  ``ceil(d / stride)``, total pad ``(out - 1) * stride + eff_k - eff_d``
  split ``lo = total // 2``, ``hi = total - lo``. With ``input_dilation >
  1`` the total can be NEGATIVE — negative padding CROPS the dilated input
  so the produced shape equals the declared ``ceil(d / stride)`` exactly
  (the interpreter validates the concrete output shape against the declared
  IR result type, so the declared size is authoritative).
- Feature groups: the input channel axis is split into ``G`` contiguous
  groups of ``C_in // G`` and the kernel output axis into ``G`` groups of
  ``C_out // G``; each group is a separate cross-correlation accumulated into
  its own output channel block.
- Batch groups (``batch_group_count`` — ops always emits 1; this is the
  hand-built-IR path): the batch axis splits into ``G`` contiguous groups;
  each group is convolved with the FULL kernel (``C_out`` channels). The
  per-group results ``(N/G, C_out, *out_spatial)`` are stacked along a new
  group axis and each group-local row is replicated ``G`` times, then the
  group axis merges into the channel axis: ``(N, C_out * G, *out_spatial)``
  — the merge consistent with ``infer_conv``'s ``out_ch = kernel_C_out * G``
  and ``out_batch = N`` shape contract.

solve semantics: ``left_side=True`` (default): ``A X = B`` via
``np.linalg.solve(a, b)``; ``left_side=False``: ``X A = B`` via transposition
(``Aᵀ Xᵀ = Bᵀ`` ⇒ ``X = solve(Aᵀ, Bᵀ)ᵀ``; vector ``b`` needs no transpose).
Batched inputs use ``np.linalg.solve`` natively. numpy's linalg dtype rule
matches ``infer_solve`` except for bool/int mixed with float (numpy returns
the promoted common type, ``infer_solve`` declares float64) — when numpy
disagrees with the DECLARED result dtype, the solve is recomputed with the
operands cast to the declared dtype (the IR result type is authoritative;
this honors the declared contract, it is not silent coercion).

Design notes (binding, parent CONTEXT.md):
- Pure numpy implementations — the interpreter is the reference for the IR,
  not a performance target. Patch gather (``sliding_window_view``) +
  ``tensordot`` reproduce the semantics exactly with plain numpy, no scipy.
- Runtime shapes come from ops-level inference rules with concrete dim
  bindings; the interpreter re-validates against the declared result types.
- Unsupported dtypes (object/str/bytes/void/datetime/timedelta) raise
  ``core.BackendError`` naming the op — never silently coerced. Complex is
  supported by all three ops (numpy semantics define it).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from etl import core

__all__ = ["register_kernels"]

#: dtype kinds all three ops define (numpy semantics): bool, int, uint,
#: float, complex. Everything else (object/str/bytes/void/datetime/
#: timedelta) is rejected with ``core.BackendError`` naming the op.
_ALLOWED_KINDS = frozenset("biufc")


def _check_dtypes(op_name: str, *arrays: np.ndarray) -> None:
    """Reject operand dtypes the op does not define (never silently coerce)."""
    for arr in arrays:
        kind = arr.dtype.kind
        if kind not in _ALLOWED_KINDS:
            raise core.BackendError(
                f"op '{op_name}': unsupported dtype {arr.dtype} — expected "
                "bool/int/uint/float/complex (object/str/void/datetime "
                "dtypes have no defined semantics)"
            )


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------


def _dot(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``dot``: ``np.matmul(a, b)``.

    The ops layer already compensates a size-1 contracting dim with an
    explicit ``broadcast`` op in the graph (numpy matmul k-broadcast
    semantics), so the kernel executes the operands as-is. ``np.matmul``'s
    dtype rule equals ``infer_dot``'s (``np.result_type``) and its shape
    rules equal the declared contract (vector/vector -> scalar,
    matrix/vector -> vector, batched matmul).
    """
    a, b = operands
    aa, bb = a.numpy(), b.numpy()
    _check_dtypes("dot", aa, bb)
    # np.asarray normalizes a scalar result (vector . vector matmul) to the
    # 0-d ndarray core.Tensor requires.
    return core.Tensor(np.asarray(np.matmul(aa, bb)))


# ---------------------------------------------------------------------------
# conv attribute normalization (mirrors ir/inference.py — runtime defensive;
# trace-time ``verify`` already re-checked these against ``infer_conv``)
# ---------------------------------------------------------------------------


def _per_dim_tuple(value: Any, n_spatial: int, default: int, name: str) -> tuple:
    """Normalize an int-or-per-spatial-tuple conv parameter.

    ``None`` (the op-def default) means all-``default``; a bare int
    replicates over all spatial dims; a tuple/list must have exactly
    ``n_spatial`` positive int entries.
    """
    if value is None:
        return (default,) * n_spatial
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return (int(value),) * n_spatial
    if isinstance(value, (tuple, list)):
        seq = tuple(value)
        if len(seq) != n_spatial:
            raise core.ShapeError(
                f"conv: {name} must have {n_spatial} entries, got {len(seq)}"
            )
        out = []
        for v in seq:
            if not isinstance(v, (int, np.integer)) or isinstance(
                v, (bool, np.bool_)
            ):
                raise core.ShapeError(
                    f"conv: {name} entries must be positive ints, got {v!r}"
                )
            out.append(int(v))
        return tuple(out)
    raise core.ShapeError(
        f"conv: {name} must be an int or a tuple of {n_spatial} ints, "
        f"got {value!r}"
    )


def _normalize_padding(padding: Any, n_spatial: int) -> tuple:
    """Normalize the raw ``padding`` attribute to per-spatial-dim entries.

    Each entry is either the string ``"SAME"`` or an int ``(lo, hi)`` pair
    (``"VALID"`` becomes ``(0, 0)``). Accepts the same raw forms the IR's
    ``_conv_padding`` accepts (and that ops passes through): the mode strings,
    a bare symmetric int, or a tuple of ``n_spatial`` entries each an int or
    a ``(lo, hi)`` pair.
    """
    if isinstance(padding, str):
        if padding not in ("VALID", "SAME"):
            raise core.ShapeError(f"conv: unknown padding mode {padding!r}")
        return ((0, 0) if padding == "VALID" else padding,) * n_spatial
    if isinstance(padding, (int, np.integer)) and not isinstance(
        padding, (bool, np.bool_)
    ):
        if padding < 0:
            raise core.ShapeError("conv: negative padding")
        return ((int(padding), int(padding)),) * n_spatial
    if isinstance(padding, (tuple, list)):
        pads = tuple(padding)
        if len(pads) != n_spatial:
            raise core.ShapeError(
                f"conv: expected {n_spatial} padding entries, got {len(pads)}"
            )
        out = []
        for p in pads:
            if isinstance(p, (int, np.integer)) and not isinstance(
                p, (bool, np.bool_)
            ):
                if p < 0:
                    raise core.ShapeError("conv: negative padding")
                out.append((int(p), int(p)))
            elif (
                isinstance(p, (tuple, list))
                and len(p) == 2
                and all(
                    isinstance(v, (int, np.integer))
                    and not isinstance(v, (bool, np.bool_))
                    for v in p
                )
            ):
                lo, hi = int(p[0]), int(p[1])
                if lo < 0 or hi < 0:
                    raise core.ShapeError("conv: negative padding")
                out.append((lo, hi))
            else:
                raise core.ShapeError(f"conv: invalid padding entry {p!r}")
        return tuple(out)
    raise core.ShapeError(f"conv: invalid padding {padding!r}")


def _dilate_spatial(arr: np.ndarray, rates: tuple) -> np.ndarray:
    """Insert ``rate - 1`` zeros between elements along the TRAILING dims.

    ``arr`` has shape ``(..., *spatial)`` with ``len(rates)`` spatial dims;
    the result has effective spatial sizes ``(d - 1) * rate + 1`` (numpy
    convolution's effective-kernel formula, applied to either the input or
    the kernel).
    """
    n_spatial = len(rates)
    if n_spatial == 0:  # not reachable for conv (rank >= 3) — clarity guard
        return arr
    new_shape = list(arr.shape[:-n_spatial])
    for i in range(n_spatial):
        new_shape.append((arr.shape[-n_spatial + i] - 1) * rates[i] + 1)
    out = np.zeros(new_shape, dtype=arr.dtype)
    out[(slice(None),) * (arr.ndim - n_spatial) + tuple(
        slice(None, None, rates[i]) for i in range(n_spatial)
    )] = arr
    return out


# ---------------------------------------------------------------------------
# conv
# ---------------------------------------------------------------------------


def _conv_group(
    x_p: np.ndarray,
    w_d: np.ndarray,
    strides: tuple,
    eff_k: tuple,
    out_spatial: tuple,
    n_spatial: int,
    feature_groups: int,
    gin: int,
    gout: int,
    result_dtype: np.dtype,
) -> np.ndarray:
    """One conv over a (padded, dilated) input: gather patches, tensordot.

    ``x_p``: ``(N, C_in, *padded_spatial)``; ``w_d``: ``(C_out, gin,
    *eff_kernel)``. Returns ``(N, C_out, *out_spatial)``.
    """
    patches = sliding_window_view(
        x_p, window_shape=eff_k, axis=tuple(range(2, 2 + n_spatial))
    )  # (N, C_in, D1..Dn, K1..Kn)
    slicer = (slice(None), slice(None)) + tuple(
        slice(None, None, s) for s in strides
    ) + (slice(None),) * n_spatial
    patches = patches[slicer]  # (N, C_in, O1..On, K1..Kn)
    out = np.empty((x_p.shape[0], feature_groups * gout) + tuple(out_spatial),
                   dtype=result_dtype)
    # Contract (C_in-group, kernel spatial) of the patch with (kernel
    # in-channels, kernel spatial) of the kernel slice.
    axes_x = [1] + [2 + n_spatial + i for i in range(n_spatial)]
    axes_w = [1] + [2 + i for i in range(n_spatial)]
    to_nchw = (0, 1 + n_spatial) + tuple(range(1, 1 + n_spatial))
    for g in range(feature_groups):
        xg = patches[:, g * gin : (g + 1) * gin]
        wg = w_d[g * gout : (g + 1) * gout]
        prod = np.tensordot(xg, wg, axes=(axes_x, axes_w))
        # tensordot order: (N, O1..On, gout) -> (N, gout, O1..On)
        out[:, g * gout : (g + 1) * gout] = prod.transpose(to_nchw)
    return out


def _conv(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``conv``: N-dimensional cross-correlation, NCHW (see module docstring).

    Layout: input ``(N, C_in, *spatial)``, kernel ``(C_out, C_in //
    feature_group_count, *kernel_spatial)``; ``channels_last`` never reaches
    the IR op (frontend sugar — the graph transposes around it).
    """
    x, w = operands
    xa, wa = x.numpy(), w.numpy()
    _check_dtypes("conv", xa, wa)
    attrs = op.attributes
    n_spatial = xa.ndim - 2

    strides = _per_dim_tuple(attrs.get("strides"), n_spatial, 1, "strides")
    in_dil = _per_dim_tuple(
        attrs.get("input_dilation"), n_spatial, 1, "input_dilation"
    )
    k_dil = _per_dim_tuple(
        attrs.get("kernel_dilation"), n_spatial, 1, "kernel_dilation"
    )
    for name, vals in (
        ("strides", strides),
        ("input_dilation", in_dil),
        ("kernel_dilation", k_dil),
    ):
        for v in vals:
            if v <= 0:
                raise core.ShapeError(
                    f"conv: {name} entries must be positive ints, got {v!r}"
                )
    pads = _normalize_padding(attrs.get("padding", "VALID"), n_spatial)
    feature_groups = attrs.get("feature_group_count", 1)
    batch_groups = attrs.get("batch_group_count", 1)
    if (
        not isinstance(feature_groups, int)
        or isinstance(feature_groups, bool)
        or feature_groups <= 0
    ):
        raise core.ShapeError(
            f"conv: feature_group_count must be a positive int, got "
            f"{feature_groups!r}"
        )
    if (
        not isinstance(batch_groups, int)
        or isinstance(batch_groups, bool)
        or batch_groups <= 0
    ):
        raise core.ShapeError(
            f"conv: batch_group_count must be a positive int, got "
            f"{batch_groups!r}"
        )
    n, c_in = int(xa.shape[0]), int(xa.shape[1])
    c_out = int(wa.shape[0])
    if c_in % feature_groups != 0:
        raise core.ShapeError(
            f"conv: in_channels {c_in} not divisible by "
            f"feature_group_count {feature_groups}"
        )
    if c_out % feature_groups != 0:
        raise core.ShapeError(
            f"conv: out_channels {c_out} not divisible by "
            f"feature_group_count {feature_groups}"
        )
    gin, gout = c_in // feature_groups, c_out // feature_groups
    if wa.shape[1] != gin:
        raise core.ShapeError(
            f"conv: kernel in-channels {wa.shape[1]} != in_channels {c_in} "
            f"// feature_group_count {feature_groups}"
        )
    if n % batch_groups != 0:
        raise core.ShapeError(
            f"conv: batch dim {n} not divisible by "
            f"batch_group_count {batch_groups}"
        )

    # 1. Dilate input and kernel (numpy effective sizes).
    x_d = _dilate_spatial(xa, in_dil)  # (N, C_in, *eff_d)
    w_d = _dilate_spatial(wa, k_dil)  # (C_out, gin, *eff_k)
    eff_k = tuple(int(v) for v in w_d.shape[2:])

    # 2. Pad the DILATED input (the _conv_out_dim formula's order: dilation
    #    first, then lo/hi on the dilated tensor). SAME padding may require a
    #    NEGATIVE amount with input dilation — cropping keeps the produced
    #    shape equal to the declared ceil(d / stride) (see module docstring).
    pad_pairs = []
    out_spatial = []
    for i in range(n_spatial):
        d = int(xa.shape[2 + i])
        eff_d = int(x_d.shape[2 + i])
        if pads[i] == "SAME":
            out_i = (d + strides[i] - 1) // strides[i]
            total = (out_i - 1) * strides[i] + eff_k[i] - eff_d
            pad_pairs.append((total // 2, total - total // 2))
        else:
            pad_pairs.append(pads[i])
        out_i = (eff_d + pad_pairs[i][0] + pad_pairs[i][1] - eff_k[i]) // strides[i] + 1
        if out_i <= 0:
            raise core.ShapeError(
                f"conv: spatial dim {i} output size {out_i} <= 0 "
                "(effective kernel larger than the padded input?)"
            )
        out_spatial.append(out_i)
    x_p = np.pad(
        x_d,
        ((0, 0), (0, 0)) + tuple((max(lo, 0), max(hi, 0)) for lo, hi in pad_pairs),
    )
    if any(lo < 0 or hi < 0 for lo, hi in pad_pairs):
        x_p = x_p[(slice(None), slice(None)) + tuple(
            slice(-lo if lo < 0 else None, hi if hi < 0 else None)
            for lo, hi in pad_pairs
        )]

    # 3. Convolve per batch group; merge per the infer_conv shape contract.
    result_dtype = np.result_type(xa.dtype, wa.dtype)
    per_group_n = n // batch_groups
    group_results = [
        _conv_group(
            x_p[g * per_group_n : (g + 1) * per_group_n],
            w_d,
            strides,
            eff_k,
            tuple(out_spatial),
            n_spatial,
            feature_groups,
            gin,
            gout,
            result_dtype,
        )
        for g in range(batch_groups)
    ]
    if batch_groups == 1:
        return core.Tensor(group_results[0])
    # (N/G, G, C_out, *out_spatial): replicate each group-local row G times,
    # then merge the group axis into the channel axis — the only merge
    # consistent with infer_conv's out_ch = kernel_C_out * G, out_batch = N.
    stacked = np.stack(group_results, axis=1)
    out = np.repeat(stacked, batch_groups, axis=0).reshape(
        (n, c_out * batch_groups) + tuple(out_spatial)
    )
    return core.Tensor(out)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def _solve_impl(aa: np.ndarray, bb: np.ndarray, left_side: bool) -> np.ndarray:
    """``np.linalg.solve`` with the IR's ``left_side`` transposition rule."""
    if left_side:
        return np.linalg.solve(aa, bb)
    # X A = B  =>  Aᵀ Xᵀ = Bᵀ  =>  X = solve(Aᵀ, Bᵀ)ᵀ. Vector b needs no
    # transpose (X A = b  =>  Aᵀ X = b).
    aa_t = np.swapaxes(aa, -1, -2)
    if bb.ndim == 1:
        return np.linalg.solve(aa_t, bb)
    x_t = np.linalg.solve(aa_t, np.swapaxes(bb, -1, -2))
    return np.swapaxes(x_t, -1, -2)


def _solve(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``solve``: linear systems per numpy ``linalg.solve``.

    ``left_side=True`` (default): solve ``A X = B``; ``left_side=False``:
    solve ``X A = B`` (via transposition — see ``_solve_impl``). Batched
    inputs (``a`` ``(..., n, n)``, ``b`` ``(..., n)``/``(..., n, k)``) use
    ``np.linalg.solve`` natively.

    numpy's linalg dtype rule matches ``infer_solve`` except when bool/int
    operands mix with float (numpy returns the promoted common type while
    ``infer_solve`` declares float64). When numpy's result dtype disagrees
    with the DECLARED IR result dtype, recompute with the operands cast to
    the declared dtype — the IR result type is authoritative (this honors the
    declared contract; it is not silent coercion).
    """
    a, b = operands
    aa, bb = a.numpy(), b.numpy()
    _check_dtypes("solve", aa, bb)
    left_side = op.attributes.get("left_side", True)
    declared = op.results[0].type.dtype
    result = _solve_impl(aa, bb, left_side)
    if result.dtype != declared:
        result = _solve_impl(
            aa.astype(declared, copy=False), bb.astype(declared, copy=False),
            left_side,
        )
    return core.Tensor(result)


def _diagonal(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``diagonal``: extract the (``axis1``, ``axis2``) diagonal with an
    ``offset`` (numpy ``np.diagonal`` semantics; dtype preserved)."""
    (x,) = operands
    x_arr = x.numpy()
    _check_dtypes("diagonal", x_arr)
    attrs = op.attributes
    return core.Tensor(
        np.diagonal(x_arr, offset=attrs["offset"], axis1=attrs["axis1"],
                    axis2=attrs["axis2"])
    )


# ---------------------------------------------------------------------------
# factorizations: eigh / cholesky / qr / matrix_rank / svd / matrix_exp
# (numpy linalg semantics; declared IR dtypes are authoritative — the
# interpreters re-validate per-result dtype/shape exactly).
# ---------------------------------------------------------------------------


def _upcast_linalg(x: np.ndarray) -> np.ndarray:
    """numpy linalg dtype rule: int/bool operands compute in float64."""
    if x.dtype.kind in "biu":
        return x.astype(np.float64, copy=False)
    return x


def _eigh(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``eigh``: Hermitian/symmetric eigendecomposition per numpy
    ``linalg.eigh`` (ascending real ``w``; batched natively). numpy already
    returns the declared dtypes (int/bool upcast to float64; complex64 → w
    float32, v complex64), so no recast is needed."""
    (x,) = operands
    x_arr = _upcast_linalg(x.numpy())
    _check_dtypes("eigh", x_arr)
    w, v = np.linalg.eigh(x_arr)
    return core.Tensor(w), core.Tensor(v)


def _cholesky(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``cholesky``: lower-triangular factor per numpy ``linalg.cholesky``
    (batched; non-PD input surfaces numpy's ``LinAlgError``)."""
    (x,) = operands
    x_arr = _upcast_linalg(x.numpy())
    _check_dtypes("cholesky", x_arr)
    return core.Tensor(np.linalg.cholesky(x_arr))


def _qr(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``qr``: reduced QR per numpy ``linalg.qr`` (``full_matrices=False``
    default; batched; rectangular inputs allowed)."""
    (x,) = operands
    x_arr = _upcast_linalg(x.numpy())
    _check_dtypes("qr", x_arr)
    q, r = np.linalg.qr(x_arr)
    return core.Tensor(q), core.Tensor(r)


def _matrix_rank(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``matrix_rank``: SVD-based numerical rank per numpy
    ``linalg.matrix_rank`` — int64 count per batch element. The static
    ``tol`` attribute (None = numpy auto) is forwarded as-is; numpy's output
    is already int64 (declared dtype)."""
    (x,) = operands
    x_arr = x.numpy()
    _check_dtypes("matrix_rank", x_arr)
    tol = op.attributes.get("tol")
    if tol is None:
        result = np.linalg.matrix_rank(x_arr)
    else:
        result = np.linalg.matrix_rank(x_arr, tol=tol)
    return core.Tensor(np.asarray(result, dtype=np.int64))


def _svd(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``svd``: singular value decomposition per numpy ``linalg.svd`` with
    ``full_matrices=False`` (batched; rectangular inputs allowed; ``s`` is
    real at the input's precision — matches the declared result dtypes)."""
    (x,) = operands
    x_arr = _upcast_linalg(x.numpy())
    _check_dtypes("svd", x_arr)
    u, s, vh = np.linalg.svd(x_arr, full_matrices=False)
    return core.Tensor(u), core.Tensor(s), core.Tensor(vh)


def _matrix_exp_impl(a: np.ndarray) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring + Taylor series (Higham
    2005 family, the scipy ``expm`` algorithm): scale ``A`` to 1-norm <= 1,
    sum ``exp(B) = sum B^k / k!`` to term 25 (truncation < 1e-25), then
    square ``s`` times. Pure numpy — numpy has no ``linalg.matrix_exp``.

    Computed in float64/complex128 and cast back to the input dtype for
    single precision (max accuracy; deviation from scipy's in-dtype
    computation is far below fp32 epsilon)."""
    work = a.astype(
        np.complex128 if a.dtype.kind == "c" else np.float64, copy=False
    )
    n = work.shape[-1]
    eye = np.broadcast_to(np.eye(n, dtype=work.dtype), work.shape)
    # 1-norm per batch element; uniform scaling over the whole batch.
    norms = np.abs(work).sum(axis=-2).max(axis=-1)
    s = int(np.ceil(np.log2(max(float(norms.max(initial=0.0)), 1.0)))) if work.size else 0
    b = work * (0.5 ** s)
    # Horner-style series: exp(B) = sum_{k=0}^{25} B^k / k!.
    term = eye.copy()
    result = term
    for k in range(1, 26):
        term = term @ b / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result.astype(a.dtype, copy=False)


def _matrix_exp(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``matrix_exp``: matrix exponential (scipy/torch semantics — numpy has
    no ``linalg.matrix_exp``). Square last two dims (validated at trace
    time); batch supported; dtype preserved with int/bool → float64."""
    (x,) = operands
    x_arr = x.numpy()
    _check_dtypes("matrix_exp", x_arr)
    work = _upcast_linalg(x_arr)
    return core.Tensor(_matrix_exp_impl(work))


def register_kernels(table: dict) -> None:
    """Register this module's linalg kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["dot"] = _dot
    table["conv"] = _conv
    table["solve"] = _solve
    table["diagonal"] = _diagonal
    table["eigh"] = _eigh
    table["cholesky"] = _cholesky
    table["qr"] = _qr
    table["matrix_rank"] = _matrix_rank
    table["svd"] = _svd
    table["matrix_exp"] = _matrix_exp
