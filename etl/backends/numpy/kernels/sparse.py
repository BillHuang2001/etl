"""Sparse kernels: the 16 sparse IR ops (category ``"sparse"``, all ``pure``).

A sparse value is an ``(indices, values)`` pair: ``indices`` is an int64
tensor of shape ``(B..., nnz, ndim)`` — leading batch dims, a runtime-dynamic
``nnz`` dim, one coordinate row per stored entry — and ``values`` holds the
stored values with shape ``(B..., nnz)``. The unbatched dense shape lives in
the op's ``dense_shape`` attribute (int entries; a SYMBOLIC first dim is
allowed after vectorize — its extent is the batch, see ``_axis_bounds``) and
the sparse value dtype in ``dtype``. **Canonical COO** = lex-sorted unique
in-range rows; ops that assume canonical input validate it at run time (see
"Canonical-form runtime validation" below) — never silently. The uniqueness
check is VALUES-AWARE: a duplicate row whose duplicate pair includes a
stored zero (at least one of the two values is 0) is tolerated — stored
zeros are semantically inert for every consumer — while duplicate NONZERO
rows (a genuine double-count hazard) still raise.

Implemented kernels (op name — semantics; shapes follow the ``ir.inference``
``infer_sparse_*`` hooks):

1. ``sparse_from_dense(dense) -> (indices, values)`` — ``np.nonzero`` +
   extraction. Batched: per batch element, padded to the common
   ``nnz_max = max(nnz_e)``; padding rows are the LEX-MAX row
   ``(dense_shape - 1)`` with stored-zero values — sortedness is preserved
   (uniqueness holds unless the lex-max row is itself a stored entry, and
   such a stored-zero duplicate is legal: canonical validation is
   values-aware, see below), and stored zeros are inert for every
   accumulator (``to_dense``/``reduce_sum``/``add``/``multiply``).
   Unbatched: unpadded.
2. ``sparse_to_dense(indices, values) -> dense`` — zeros +
   ``np.add.at`` scatter (duplicate rows accumulate). Result dtype =
   ``core.dtype(attributes["dtype"])``; shape = batch + evaluated
   ``dense_shape``.
3. ``sparse_coo_to_csr(indices, values) -> (indptr, indices, values)`` —
   rank-2 only; indptr from row counts (``np.bincount``); NO reorder — the
   COO is validated row-major-sorted first. Result column indices =
   ``indices[:, 1]``; the values array passes through.
4. ``sparse_csr_to_coo(indptr, indices, values) -> (indices, values)`` —
   rank-2; expand indptr via ``np.repeat``; validates indptr (monotone,
   ``indptr[0] == 0``, ``indptr[-1] == nnz``), columns in-range, per-row
   strictly increasing; NO reorder.
5. ``sparse_coo_to_csc(indices, values) -> (indptr, indices, values)`` —
   rank-2; canonical-validates the COO, then reorders column-major
   (``np.lexsort((row, col))``). The indptr is the STANDARD CSC column
   pointer array of length ``cols+1`` (``infer_sparse_coo_to_csc`` declares
   batch + (dense_shape[1] + 1,)): ``indptr[0] == 0``, monotone
   non-decreasing, ``indptr[-1] == nnz``; both unbatched and batched
   (per-batch element).
6. ``sparse_csc_to_coo(indptr, indices, values) -> (indices, values)`` —
   rank-2; validates the CSC triple (indptr length ``cols+1`` standard —
   the legacy padded ``rows+1`` convention is also still accepted
   defensively; the tail must equal nnz, strictly; else ``ShapeError``),
   expands, then re-sorts back to row-major (``np.lexsort((col, row))``).
7. ``sparse_negate(indices, values) -> (indices, -values)`` — structure
   preserved; no canonical assumption.
8. ``sparse_add(ia, va, ib, vb) -> (indices, values)`` — union merge over
   sorted rows: equal rows -> summed (one row kept), otherwise both rows kept
   in order. Both operands canonical-validated; matching value dtypes
   required (``DTypeError``). Batched results are padded to the common max
   nnz with lex-max stored-zero rows (see module docstring, batched note).
9. ``sparse_multiply(ia, va, ib, vb) -> (indices, values)`` — intersection
   merge: equal rows -> product, unequal -> skipped. Same validation as
   ``sparse_add``.
10. ``sparse_multiply_dense(indices, values, dense) -> (indices, values)`` —
    structure preserved: ``values * dense[rows]`` per batch element (dense
    may carry the batch dims or be unbatched — broadcast). Result values
    dtype = ``np.result_type(values.dtype, dense.dtype)``. Sparse operand
    canonical-validated (the gather must be in-range).
11. ``sparse_reduce_sum(indices, values) -> dense`` — build the dense via
    ``np.add.at`` then ``np.sum(axis=axes, keepdims=keepdims)``; all-axes fast
    path = ``values.sum()``. ``axes`` refer to the unbatched sparse axes
    (normalized: int -> tuple, list -> tuple; an EMPTY tuple = NO reduction —
    identity dense, matching ``infer_sparse_reduce_sum``, which keeps every
    dim for empty ``axes``). Result dtype
    = ``core.dtype(attributes["dtype"])`` — the op DECLARES it, so the
    computed sum is converted to it (accumulation happens in the declared
    dtype via the zeros buffer / final astype; float identity). In-range
    checked (out-of-range would otherwise surface as a raw numpy
    ``IndexError``); sortedness/duplicates are NOT assumed (stored zeros from
    batched padding are legal here).
12. ``sparse_transpose(indices, values)`` attrs {dense_shape = RESULT, perm}
    — ``new_indices = indices[:, perm]`` per batch element, re-sorted
    lexicographically, values permuted. Input canonical-validated against
    the inverse-permuted shape.
13. ``sparse_reshape(indices, values)`` attrs {dense_shape = RESULT,
    old_shape} — linearize via ``np.ravel_multi_index``, re-map via
    ``np.unravel_index``, re-sort. STATIC shapes only: evaluation failure or
    a runtime-dynamic (None) dim => ``ShapeError`` ("dynamic"). Input
    canonical-validated; an element-count mismatch (linear index beyond the
    new element count) => ``ShapeError``.
14. ``sparse_concatenate(ia, va, ib, vb, ...)`` attrs {dense_shape = RESULT,
    axis, **operand_extents**} — REQUIRED ``operand_extents`` attribute
    (per-operand ``dense_shape[axis]`` extents; MISSING =>
    ``BackendError`` naming the op — note the opdef amendment declaring it is
    pending upstream, the kernel still requires it). Axis coordinates are
    offset by the prefix sum of the extents; indices/values concatenated and
    RE-SORTED (disjoint axis ranges do not imply global lex order unless
    ``axis == 0``). Each operand canonical-validated against its own extent
    (axis coordinate < its extent; other axes vs the shared result shape);
    the extent sum must equal ``dense_shape[axis]``. Values are promoted to
    ``np.result_type`` over all operands (the inference-declared dtype).
15. ``sparse_dot_dense(indices, values, dense)`` attrs {dense_shape = (M, K)
    of the sparse} -> dense (M, N): ``np.add.at`` over the row index with
    ``values[:, None] * dense[rows, :]`` contributions (np.add.at REQUIRED —
    duplicate row targets accumulate). Batched per batch element; dense may
    carry the batch dims. Result = batch + (M, N); dtype =
    ``np.result_type(values.dtype, dense.dtype)``. Sparse canonical-
    validated; dense's inner dim must equal K (``ShapeError``).
16. ``dense_dot_sparse(dense, indices, values)`` attrs {dense_shape = (K, N)
    of the sparse} -> dense (M, N): ``np.add.at`` over the column index with
    ``dense[:, rows] * values`` contributions. Same batching/validation rules
    as ``sparse_dot_dense``.

Batched handling (binding): every kernel handles leading batch dims on the
sparse operands by LOOPING over the batch (v1). ``nnz`` is a SINGLE shared
dim, so per-batch results with element-varying nnz (``from_dense``,
``add``/``multiply`` merges) are padded to the common max nnz with the
lex-max row and stored-zero values (see the per-op notes; this is the only
way to represent them in one ndarray — downstream accumulators treat stored
zeros as inert, and downstream canonical VALIDATION is values-aware: a
duplicate row whose duplicate pair includes the stored zero is accepted).
Rank-2-only ops (the csr/csc conversions and the dot variants) enforce
``len(dense_shape) == 2`` with ``core.ShapeError``; batch dims still pass
through.

Canonical-form runtime validation (binding): ops that ASSUME canonical input
(``add``, ``multiply``, ``multiply_dense``, ``transpose``, ``reshape``,
``concatenate``, ``coo_to_csr``, ``coo_to_csc``, ``csr_to_coo``,
``csc_to_coo``, the dot variants, ``to_dense``) validate: rows lex-sorted
and unique — VALUES-AWARE: an adjacent duplicate row (COO) or duplicate
per-row/per-column segment entry (CSR/CSC) raises ONLY when BOTH values of
the duplicate pair are nonzero (a genuine double-count hazard); a duplicate
pair that includes a stored zero is tolerated, which is what makes batched
``sparse_from_dense``'s documented stored-zero padding (lex-max rows) legal
— all coordinates in-range wrt the (evaluated) ``dense_shape``, and (for
the csr/csc conversions) indptr monotone with ``indptr[0] == 0`` and
``indptr[-1] == nnz`` plus strictly-increasing per-row/per-column entries —
raising an explicit ``core.ShapeError`` naming the op and the violation,
NEVER silent. Checks are vectorized numpy. NOTE (vectorize case): when
``dense_shape[0]`` is a SYMBOLIC ``core.Dim`` whose extent is the batch,
the in-range bound for sparse axis 0 is the batch extent (the Dim evaluates
to it when bound; otherwise the batch extent of the indices array is used);
all other axes evaluate normally.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]

_INT64 = np.dtype("int64")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _batch_count(batch: tuple) -> int:
    """Number of batch elements (1 when the batch is empty)."""
    return math.prod(batch) if batch else 1


def _evaluate_dense_shape(ctx: Any, op_name: str, dense_shape: Any) -> tuple:
    """Evaluate the ``dense_shape`` attribute to concrete ints.

    Every entry is evaluated via ``ctx.evaluate_shape`` (ints pass through,
    ``Dim``/``DimExpr`` resolve against the runtime bindings). Failure =>
    ``ShapeError`` naming the op and the offending shape. ``None``
    (runtime-dynamic) entries are rejected by ``evaluate_shape`` itself.
    """
    try:
        return tuple(ctx.evaluate_shape(tuple(dense_shape)))
    except core.ShapeError as exc:
        raise core.ShapeError(
            f"kernel for op '{op_name}': cannot evaluate dense_shape "
            f"{tuple(dense_shape)!r}: {exc}"
        ) from exc


def _evaluate_static_shape(ctx: Any, op_name: str, shape: Any, what: str) -> tuple:
    """Evaluate a shape attribute that must be FULLY static.

    A runtime-dynamic (``None``) entry or an unresolvable symbolic dim =>
    ``ShapeError`` with a "dynamic" hint (used by ``sparse_reshape`` and
    ``sparse_concatenate``, which need concrete extents).
    """
    shape = tuple(shape)
    try:
        return tuple(ctx.evaluate_shape(shape))
    except core.ShapeError as exc:
        if any(d is None for d in shape):
            raise core.ShapeError(
                f"kernel for op '{op_name}': {what} {shape!r} contains a "
                "runtime-dynamic (None) dim — this op requires static shapes"
            ) from exc
        raise core.ShapeError(
            f"kernel for op '{op_name}': cannot evaluate {what} {shape!r}: {exc}"
        ) from exc


def _axis_bounds(ctx: Any, op_name: str, dense_shape: tuple, batch: tuple) -> list:
    """Per-axis in-range bounds for canonical validation.

    ``dense_shape`` is the RAW attribute (ints / ``Dim`` / ``DimExpr``). A
    SYMBOLIC first dim (``core.Dim`` at axis 0 — the vectorize case) is
    evaluated when bound; when unbound its extent is the BATCH: the bound
    falls back to the batch extent. Every other entry must evaluate.
    """
    bounds = []
    for axis, entry in enumerate(dense_shape):
        if axis == 0 and isinstance(entry, core.Dim):
            try:
                bounds.append(ctx.evaluate_shape((entry,))[0])
            except core.ShapeError:
                if batch:
                    bounds.append(_batch_count(batch))
                else:
                    raise core.ShapeError(
                        f"kernel for op '{op_name}': cannot evaluate the "
                        f"symbolic first dim {entry!r} of dense_shape "
                        f"{tuple(dense_shape)!r} and the sparse operand has "
                        "no batch dims to derive its extent from"
                    ) from None
            continue
        if isinstance(entry, int) and not isinstance(entry, bool):
            bounds.append(entry)
            continue
        try:
            bounds.append(ctx.evaluate_shape((entry,))[0])
        except core.ShapeError as exc:
            raise core.ShapeError(
                f"kernel for op '{op_name}': cannot evaluate dense_shape "
                f"entry {entry!r}: {exc}"
            ) from exc
    return bounds


def _check_pair(op_name: str, indices: np.ndarray, values: np.ndarray) -> None:
    """Validate the (indices, values) pair structure: indices ``(B..., nnz,
    ndim)`` with ``ndim >= 1``, values ``(B..., nnz)`` with the same batch
    dims and nnz. Rank-0 sparse tensors are rejected explicitly."""
    if indices.ndim < 2 or indices.shape[-1] < 1:
        raise core.ShapeError(
            f"kernel for op '{op_name}': sparse indices must have shape "
            f"(batch..., nnz, ndim) with ndim >= 1, got {tuple(indices.shape)}"
        )
    if (
        values.ndim != indices.ndim - 1
        or tuple(values.shape[:-1]) != tuple(indices.shape[:-2])
        or values.shape[-1] != indices.shape[-2]
    ):
        raise core.ShapeError(
            f"kernel for op '{op_name}': values shape {tuple(values.shape)} is "
            f"inconsistent with indices shape {tuple(indices.shape)} (expected "
            "(batch..., nnz) with the same batch dims)"
        )


def _check_sorted_unique(idx: np.ndarray, vals: np.ndarray, op_name: str) -> None:
    """Validate one element's 2-D indices as lex-sorted with unique rows.

    O(n) vectorized adjacent-row comparison (no full sort). VALUES-AWARE
    uniqueness: an adjacent duplicate row raises ``ShapeError`` only when
    BOTH values of the duplicate pair are nonzero (a genuine double-count
    hazard); a duplicate pair where at least one value is 0 (a stored zero,
    e.g. batched ``sparse_from_dense`` padding) is tolerated — stored zeros
    are semantically inert for every consumer. Raises ``ShapeError`` naming
    the op and the offending rows.
    """
    nnz, _ndim = idx.shape
    if nnz < 2:
        return
    nxt = idx[1:]
    prv = idx[:-1]
    dup = np.all(nxt == prv, axis=1)
    if np.any(dup):
        both_nonzero = (vals[1:] != 0) & (vals[:-1] != 0)
        bad = dup & both_nonzero
        if np.any(bad):
            i = int(np.argmax(bad))
            raise core.ShapeError(
                f"kernel for op '{op_name}': indices contain duplicate rows "
                "(canonical COO requires sorted unique rows) — row "
                f"{prv[i].tolist()} appears more than once"
            )
    diff = nxt != prv
    first = np.argmax(diff, axis=1)  # first differing coordinate per pair
    rows_nxt = nxt[np.arange(nnz - 1), first]
    rows_prv = prv[np.arange(nnz - 1), first]
    bad = rows_nxt < rows_prv
    if np.any(bad):
        i = int(np.argmax(bad))
        raise core.ShapeError(
            f"kernel for op '{op_name}': indices are not lex-sorted (canonical "
            "COO requires sorted unique rows) — rows "
            f"{prv[i].tolist()} and {nxt[i].tolist()} are out of order"
        )


def _check_in_range(idx: np.ndarray, bounds: list, op_name: str) -> None:
    """Validate one element's 2-D indices against per-axis bounds."""
    for axis, bound in enumerate(bounds):
        coords = idx[:, axis]
        if coords.size == 0:
            continue
        mn = int(coords.min())
        mx = int(coords.max())
        if mn < 0 or mx >= bound:
            raise core.ShapeError(
                f"kernel for op '{op_name}': index out of range for sparse "
                f"axis {axis}: coordinates span [{mn}, {mx}] but must lie in "
                f"[0, {bound}) per dense_shape {bounds!r}"
            )


def _validate_canonical(
    ctx: Any,
    op_name: str,
    indices: np.ndarray,
    values: np.ndarray,
    dense_shape: tuple,
    check_sorted: bool = True,
) -> None:
    """Validate a COO-form (indices, values) pair as CANONICAL.

    Checks (all ``core.ShapeError`` naming the op, never silent): the pair
    structure, ``indices.shape[-1] == len(dense_shape)``, per batch element
    lex-sorted values-aware-unique rows (unless ``check_sorted=False`` —
    used by ``sparse_reduce_sum``, which only needs in-range; see
    ``_check_sorted_unique`` for the stored-zero tolerance), and all
    coordinates in-range wrt the evaluated ``dense_shape`` bounds (see
    ``_axis_bounds`` for the vectorize symbolic-first-dim rule).
    """
    _check_pair(op_name, indices, values)
    rank = len(dense_shape)
    if indices.shape[-1] != rank:
        raise core.ShapeError(
            f"kernel for op '{op_name}': sparse rank {indices.shape[-1]} does "
            f"not match dense_shape rank {rank} ({tuple(dense_shape)!r})"
        )
    batch = indices.shape[:-2]
    n = _batch_count(batch)
    flat = indices.reshape((n,) + indices.shape[-2:])
    flat_vals = values.reshape((n,) + values.shape[-1:])
    bounds = _axis_bounds(ctx, op_name, tuple(dense_shape), batch)
    for b in range(n):
        idx = flat[b]
        if check_sorted:
            _check_sorted_unique(idx, flat_vals[b], op_name)
        _check_in_range(idx, bounds, op_name)


def _validate_indptr(op_name: str, indptr: np.ndarray, nnz: int, n_buckets: int) -> None:
    """Validate a CSR/CSC indptr: 1-D of length ``n_buckets + 1``,
    ``indptr[0] == 0``, monotone non-decreasing, ``indptr[-1] == nnz``."""
    if indptr.ndim != 1 or indptr.shape[0] != n_buckets + 1:
        raise core.ShapeError(
            f"kernel for op '{op_name}': indptr must be 1-D of length "
            f"{n_buckets + 1} (n_buckets + 1), got shape {tuple(indptr.shape)}"
        )
    if int(indptr[0]) != 0:
        raise core.ShapeError(
            f"kernel for op '{op_name}': indptr[0] must be 0, got {int(indptr[0])}"
        )
    if np.any(indptr[1:] < indptr[:-1]):
        raise core.ShapeError(
            f"kernel for op '{op_name}': indptr is not monotone non-decreasing"
        )
    if int(indptr[-1]) != nnz:
        raise core.ShapeError(
            f"kernel for op '{op_name}': indptr[-1] ({int(indptr[-1])}) must "
            f"equal the number of stored entries ({nnz})"
        )


def _check_segment_sorted(
    op_name: str, seg: np.ndarray, seg_vals: np.ndarray, bound: int, what: str
) -> None:
    """Validate one row/column segment of a CSR/CSC indices array: entries
    in ``[0, bound)`` and strictly increasing (VALUES-AWARE: an adjacent
    equal pair is tolerated when at least one of the two values is 0 — a
    stored zero, e.g. from batched padding — while an equal pair with both
    values nonzero and any strictly-decreasing entry still raise)."""
    if seg.size == 0:
        return
    mn = int(seg.min())
    mx = int(seg.max())
    if mn < 0 or mx >= bound:
        raise core.ShapeError(
            f"kernel for op '{op_name}': {what} index out of range: entries "
            f"span [{mn}, {mx}] but must lie in [0, {bound})"
        )
    if np.any(seg[1:] < seg[:-1]):
        raise core.ShapeError(
            f"kernel for op '{op_name}': {what} entries are not strictly "
            "increasing within each row/column (CSR/CSC requires sorted "
            "unique entries)"
        )
    eq = seg[1:] == seg[:-1]
    if np.any(eq):
        both_nonzero = (seg_vals[1:] != 0) & (seg_vals[:-1] != 0)
        if np.any(eq & both_nonzero):
            raise core.ShapeError(
                f"kernel for op '{op_name}': {what} entries are not strictly "
                "increasing within each row/column (CSR/CSC requires sorted "
                "unique entries)"
            )


def _row_keys(idx2d: np.ndarray, dense_shape: tuple) -> np.ndarray:
    """Scalar row keys (lex order == sorted order) for 2-D coordinate rows.

    Uses ``np.ravel_multi_index`` over the evaluated dense shape (1-D rows
    are their own keys). Canonical (sorted) rows yield sorted keys.
    """
    if idx2d.shape[1] == 1:
        return idx2d[:, 0]
    return np.ravel_multi_index(idx2d.T, dense_shape).astype(_INT64)


def _unstack_keys(keys: np.ndarray, dense_shape: tuple) -> np.ndarray:
    """Coordinate rows ``(nnz, rank)`` for scalar row keys (lex-sorted)."""
    if keys.size == 0:
        return np.zeros((0, len(dense_shape)), _INT64)
    return np.stack(np.unravel_index(keys, dense_shape), axis=-1).astype(_INT64)


def _lexsort_rows(idx2d: np.ndarray) -> np.ndarray:
    """Permutation sorting 2-D rows lexicographically (axis 0 primary)."""
    return np.lexsort(tuple(idx2d[:, j] for j in reversed(range(idx2d.shape[1]))))


def _pad_row(dense_shape: tuple) -> tuple:
    """The lex-max row of a dense shape — padding row for stored zeros."""
    return tuple(d - 1 for d in dense_shape)


def _dense_elements(dense: np.ndarray, batch: tuple, dense_rank: int, op_name: str) -> list:
    """Split a dense operand per batch element.

    The dense may carry the sparse batch dims or be unbatched (broadcast to
    every element). Incompatible ranks or batch extents => ``ShapeError`` —
    never a silent misalignment.
    """
    n = _batch_count(batch)
    if dense.ndim == dense_rank:
        return [dense] * n
    if dense.ndim == dense_rank + len(batch):
        if tuple(dense.shape[: len(batch)]) != tuple(batch):
            raise core.ShapeError(
                f"kernel for op '{op_name}': dense operand batch dims "
                f"{tuple(dense.shape[: len(batch)])} do not match the sparse "
                f"batch dims {tuple(batch)}"
            )
        flat = dense.reshape((n,) + dense.shape[-dense_rank:])
        return [flat[b] for b in range(n)]
    raise core.ShapeError(
        f"kernel for op '{op_name}': dense operand rank {dense.ndim} is "
        f"incompatible with sparse rank {dense_rank} and batch rank "
        f"{len(batch)} (expected rank {dense_rank} or {dense_rank + len(batch)})"
    )


def _merge_precheck(
    ctx: Any,
    op_name: str,
    ia: np.ndarray,
    va: np.ndarray,
    ib: np.ndarray,
    vb: np.ndarray,
    dense_shape: tuple,
) -> None:
    """Shared validation for the sparse merge ops (add / multiply): both
    operands canonical-validated against the SAME ``dense_shape`` attr,
    matching batch dims, matching value dtypes."""
    _validate_canonical(ctx, op_name, ia, va, dense_shape)
    _validate_canonical(ctx, op_name, ib, vb, dense_shape)
    if ia.shape[:-2] != ib.shape[:-2]:
        raise core.ShapeError(
            f"kernel for op '{op_name}': operand batch dims differ — "
            f"{tuple(ia.shape[:-2])} vs {tuple(ib.shape[:-2])}"
        )
    if va.dtype != vb.dtype:
        raise core.DTypeError(
            f"kernel for op '{op_name}': operand value dtypes differ "
            f"({va.dtype} vs {vb.dtype}) — the merge requires matching dtypes"
        )


def _rank2_shape(ctx: Any, op_name: str, dense_shape_attr: Any) -> tuple:
    """Evaluate the ``dense_shape`` attribute and enforce rank 2 (the csr/csc
    conversions and the dot variants are rank-2-only)."""
    dense_shape = _evaluate_static_shape(ctx, op_name, dense_shape_attr, "dense_shape")
    if len(dense_shape) != 2:
        raise core.ShapeError(
            f"kernel for op '{op_name}': sparse operand must be rank-2, got "
            f"dense_shape {dense_shape!r}"
        )
    return dense_shape


# ---------------------------------------------------------------------------
# Structure conversions: from_dense / to_dense / coo<->csr / coo<->csc
# ---------------------------------------------------------------------------


def _from_dense(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_from_dense``: np.nonzero + extraction; batched results are
    padded to the common max nnz with lex-max stored-zero rows."""
    name = op.name
    dense = operands[0].numpy()
    dense_shape_attr = tuple(op.attributes["dense_shape"])
    rank = len(dense_shape_attr)
    if rank == 0:
        raise core.ShapeError(
            f"kernel for op '{name}': rank-0 sparse tensors are not supported"
        )
    if dense.ndim < rank:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand rank {dense.ndim} is "
            f"smaller than the sparse rank {rank}"
        )
    dense_shape = _evaluate_dense_shape(ctx, name, dense_shape_attr)
    if tuple(dense.shape[-rank:]) != dense_shape:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand trailing shape "
            f"{tuple(dense.shape[-rank:])} does not match the declared "
            f"dense_shape {dense_shape!r}"
        )
    batch = dense.shape[: dense.ndim - rank]
    n = _batch_count(batch)
    dense_flat = dense.reshape((n,) + dense.shape[-rank:])
    pad = _pad_row(dense_shape)
    nnz_max = 0
    per = []
    for b in range(n):
        de = dense_flat[b]
        coords = np.nonzero(de)
        idx = np.stack(coords, axis=-1).astype(_INT64)
        vals = de[coords]
        per.append((idx, vals))
        nnz_max = max(nnz_max, int(idx.shape[0]))
    out_idx = np.zeros((n, nnz_max, rank), _INT64)
    out_vals = np.zeros((n, nnz_max), dtype=dense.dtype)
    for b in range(n):
        idx, vals = per[b]
        m = int(idx.shape[0])
        out_idx[b, :m] = idx
        out_vals[b, :m] = vals
        if m < nnz_max:
            out_idx[b, m:] = pad  # stored zeros (values already 0)
    out_idx = out_idx.reshape(batch + (nnz_max, rank))
    out_vals = out_vals.reshape(batch + (nnz_max,))
    return core.Tensor(out_idx), core.Tensor(out_vals)


def _to_dense(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``sparse_to_dense``: zeros + np.add.at scatter (canonical input
    assumed — duplicate rows would silently double-count)."""
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    dense_shape = _evaluate_dense_shape(ctx, name, op.attributes["dense_shape"])
    _validate_canonical(ctx, name, idx, vals, dense_shape)
    out_dtype = core.dtype(op.attributes["dtype"])
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    out = np.zeros(batch + dense_shape, dtype=out_dtype)
    out_flat = out.reshape((n,) + dense_shape)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    for b in range(n):
        np.add.at(out_flat[b], tuple(idx_flat[b].T), vals_flat[b])
    return core.Tensor(out)


def _coo_to_csr(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_coo_to_csr`` (rank-2): indptr from row counts; the COO is
    validated row-major-sorted, so NO reorder happens."""
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    rows, _cols = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    _validate_canonical(ctx, name, idx, vals, (rows, _cols))
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_indptr = np.zeros((n, rows + 1), _INT64)
    out_indices = np.empty((n, idx.shape[-2]), _INT64)
    out_vals = np.empty((n, vals.shape[-1]), dtype=vals.dtype)
    for b in range(n):
        counts = np.bincount(idx_flat[b][:, 0], minlength=rows).astype(_INT64)
        out_indptr[b, 1:] = np.cumsum(counts)
        out_indices[b] = idx_flat[b][:, 1]
        out_vals[b] = vals_flat[b]
    return (
        core.Tensor(out_indptr.reshape(batch + (rows + 1,))),
        core.Tensor(out_indices.reshape(batch + (idx.shape[-2],))),
        core.Tensor(out_vals.reshape(vals.shape)),
    )


def _csr_to_coo(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_csr_to_coo`` (rank-2): expand indptr via np.repeat; the CSR
    triple is validated (indptr monotone, columns in-range + strictly
    increasing per row); NO reorder."""
    name = op.name
    indptr, indices, values = operands
    ip_a, idx_a, vals = indptr.numpy(), indices.numpy(), values.numpy()
    rows, cols = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    batch = ip_a.shape[:-1]  # inference: batch dims come from the indptr
    n = _batch_count(batch)
    ip_flat = ip_a.reshape((n, rows + 1))
    idx_flat = idx_a.reshape((n,) + idx_a.shape[-1:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_idx = np.empty((n, idx_a.shape[-1], 2), _INT64)
    out_vals = np.empty((n, vals.shape[-1]), dtype=vals.dtype)
    for b in range(n):
        ip = ip_flat[b]
        ci = idx_flat[b]
        _validate_indptr(name, ip, ci.shape[0], rows)
        for r in range(rows):
            _check_segment_sorted(
                name,
                ci[ip[r]: ip[r + 1]],
                vals_flat[b][ip[r]: ip[r + 1]],
                cols,
                "column",
            )
        row_ids = np.repeat(np.arange(rows, dtype=_INT64), np.diff(ip))
        out_idx[b] = np.stack([row_ids, ci], axis=-1)
        out_vals[b] = vals_flat[b]
    return (
        core.Tensor(out_idx.reshape(batch + (idx_a.shape[-1], 2))),
        core.Tensor(out_vals.reshape(vals.shape)),
    )


def _coo_to_csc(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_coo_to_csc`` (rank-2): canonical-validate, then reorder to
    column-major (lexsort by (col, row)); indptr over columns.

    Shape contract (binding): the indptr is the STANDARD CSC column-pointer
    array of length ``cols + 1`` — ``infer_sparse_coo_to_csc`` declares
    batch + (dense_shape[1] + 1,) — with ``indptr[0] == 0``, monotone
    non-decreasing, ``indptr[-1] == nnz``. Both the unbatched and the
    batched paths emit one such indptr per batch element (``(B, cols+1)``).
    """
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    rows, cols = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    _validate_canonical(ctx, name, idx, vals, (rows, cols))
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_indptr = np.zeros((n, cols + 1), _INT64)
    out_indices = np.empty((n, idx.shape[-2]), _INT64)
    out_vals = np.empty((n, vals.shape[-1]), dtype=vals.dtype)
    for b in range(n):
        row = idx_flat[b][:, 0]
        col = idx_flat[b][:, 1]
        order = np.lexsort((row, col))  # primary: col, secondary: row
        counts = np.bincount(col[order], minlength=cols).astype(_INT64)
        out_indptr[b, 1:] = np.cumsum(counts)
        out_indices[b] = row[order]
        out_vals[b] = vals_flat[b][order]
    return (
        core.Tensor(out_indptr.reshape(batch + (cols + 1,))),
        core.Tensor(out_indices.reshape(batch + (idx.shape[-2],))),
        core.Tensor(out_vals.reshape(vals.shape)),
    )


def _csc_to_coo(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_csc_to_coo`` (rank-2): validate the CSC triple, expand, then
    RE-SORT back to row-major (lexsort by (row, col)).

    The indptr is the STANDARD CSC column-pointer convention of length
    ``cols+1`` (both the concrete ``etl.sparse.csc`` path and the graph
    ``sparse_coo_to_csc`` op emit it). For backward compatibility the
    legacy padded ``rows+1`` convention (true column pointers in
    ``ip[0:cols+1]`` plus a strictly-validated monotone ``nnz`` tail) is
    still accepted. Anything else => ``ShapeError``.
    """
    name = op.name
    indptr, indices, values = operands
    ip_a, idx_a, vals = indptr.numpy(), indices.numpy(), values.numpy()
    rows, cols = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    batch = ip_a.shape[:-1]
    n = _batch_count(batch)
    ip_flat = ip_a.reshape((n, ip_a.shape[-1]))
    idx_flat = idx_a.reshape((n,) + idx_a.shape[-1:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_idx = np.empty((n, idx_a.shape[-1], 2), _INT64)
    out_vals = np.empty((n, vals.shape[-1]), dtype=vals.dtype)
    for b in range(n):
        ip = ip_flat[b]
        ri = idx_flat[b]
        nnz = ri.shape[0]
        if ip.shape[0] == cols + 1:
            ip_eff = ip
        elif ip.shape[0] == rows + 1:
            if cols + 1 > ip.shape[0]:
                raise core.ShapeError(
                    f"kernel for op '{name}': indptr length {ip.shape[0]} "
                    f"(rows+1) cannot hold {cols + 1} column pointers"
                )
            _validate_indptr(name, ip, nnz, rows)  # full-length sanity
            ip_eff = ip[: cols + 1]
            if not np.all(ip[cols + 1 :] == nnz):
                raise core.ShapeError(
                    f"kernel for op '{name}': padded indptr tail "
                    f"{ip[cols + 1 :].tolist()} must equal nnz={nnz} (the "
                    "csc column-pointer padding convention)"
                )
        else:
            raise core.ShapeError(
                f"kernel for op '{name}': indptr length {ip.shape[0]} must be "
                f"cols+1 = {cols + 1} (standard) or rows+1 = {rows + 1} "
                "(padded csc convention)"
            )
        _validate_indptr(name, ip_eff, nnz, cols)
        for c in range(cols):
            _check_segment_sorted(
                name,
                ri[ip_eff[c]: ip_eff[c + 1]],
                vals_flat[b][ip_eff[c]: ip_eff[c + 1]],
                rows,
                "row",
            )
        col_ids = np.repeat(np.arange(cols, dtype=_INT64), np.diff(ip_eff))
        two_d = np.stack([ri, col_ids], axis=-1)  # (col, row) pairs
        order = np.lexsort((col_ids, ri))  # primary: row, secondary: col
        out_idx[b] = two_d[order]
        out_vals[b] = vals_flat[b][order]
    return (
        core.Tensor(out_idx.reshape(batch + (idx_a.shape[-1], 2))),
        core.Tensor(out_vals.reshape(vals.shape)),
    )


# ---------------------------------------------------------------------------
# Arithmetic: negate / add / multiply / multiply_dense
# ---------------------------------------------------------------------------


def _negate(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_negate``: structure preserved, values negated."""
    del ctx
    indices, values = operands
    return core.Tensor(indices.numpy()), core.Tensor(-values.numpy())


def _union_merge(
    i1: np.ndarray, v1: np.ndarray, i2: np.ndarray, v2: np.ndarray, dense_shape: tuple
) -> tuple:
    """Union merge of two canonical (sorted-unique) COO elements.

    Concatenate row keys, sort, group by key, sum per group — the result is
    lex-sorted with unique rows by construction. Returns (indices, values).
    """
    k1 = _row_keys(i1, dense_shape)
    k2 = _row_keys(i2, dense_shape)
    if k1.size == 0 and k2.size == 0:
        return np.zeros((0, i1.shape[1]), _INT64), v1[:0]
    keys = np.concatenate((k1, k2))
    vals = np.concatenate((v1, v2))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    vals = vals[order]
    new_keys, starts = np.unique(keys, return_index=True)
    sums = np.add.reduceat(vals, starts)
    return _unstack_keys(new_keys, dense_shape), sums


def _intersection_merge(
    i1: np.ndarray, v1: np.ndarray, i2: np.ndarray, v2: np.ndarray, dense_shape: tuple
) -> tuple:
    """Intersection merge of two canonical (sorted-unique) COO elements:
    equal rows -> product; unequal rows are skipped. Result is canonical."""
    k1 = _row_keys(i1, dense_shape)
    k2 = _row_keys(i2, dense_shape)
    common, i1c, i2c = np.intersect1d(k1, k2, return_indices=True)
    return _unstack_keys(common, dense_shape), v1[i1c] * v2[i2c]


def _merge(
    ctx: Any, op: Any, operands: tuple, merge_fn: Any, op_label: str
) -> tuple:
    """Shared driver for ``sparse_add`` / ``sparse_multiply``: validate both
    operands canonical (same dense_shape attr, matching batch + dtype), merge
    per batch element, pad batched results to the common max nnz with
    lex-max stored-zero rows."""
    name = op.name
    ia, va, ib, vb = operands
    ia_, va_, ib_, vb_ = ia.numpy(), va.numpy(), ib.numpy(), vb.numpy()
    dense_shape_attr = tuple(op.attributes["dense_shape"])
    _merge_precheck(ctx, name, ia_, va_, ib_, vb_, dense_shape_attr)
    dense_shape = _evaluate_dense_shape(ctx, name, dense_shape_attr)
    rank = len(dense_shape)
    pad = _pad_row(dense_shape)
    batch = ia_.shape[:-2]
    n = _batch_count(batch)
    ia_flat = ia_.reshape((n,) + ia_.shape[-2:])
    va_flat = va_.reshape((n,) + va_.shape[-1:])
    ib_flat = ib_.reshape((n,) + ib_.shape[-2:])
    vb_flat = vb_.reshape((n,) + vb_.shape[-1:])
    per = []
    nnz_max = 0
    for b in range(n):
        new_idx, new_vals = merge_fn(
            ia_flat[b], va_flat[b], ib_flat[b], vb_flat[b], dense_shape
        )
        per.append((new_idx, new_vals))
        nnz_max = max(nnz_max, int(new_idx.shape[0]))
    out_idx = np.zeros((n, nnz_max, rank), _INT64)
    out_vals = np.zeros((n, nnz_max), dtype=va_.dtype)
    for b in range(n):
        new_idx, new_vals = per[b]
        m = int(new_idx.shape[0])
        out_idx[b, :m] = new_idx
        out_vals[b, :m] = new_vals
        if m < nnz_max:
            out_idx[b, m:] = pad  # stored zeros (values already 0)
    return (
        core.Tensor(out_idx.reshape(batch + (nnz_max, rank))),
        core.Tensor(out_vals.reshape(batch + (nnz_max,))),
    )


def _add(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_add``: union merge (equal rows summed)."""
    return _merge(ctx, op, operands, _union_merge, "add")


def _multiply(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_multiply``: intersection merge (equal rows multiplied)."""
    return _merge(ctx, op, operands, _intersection_merge, "multiply")


def _multiply_dense(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_multiply_dense``: structure preserved; values multiplied by
    the gathered dense entries (dense may carry the batch dims or not)."""
    name = op.name
    indices, values, dense = operands
    idx, vals, dens = indices.numpy(), values.numpy(), dense.numpy()
    dense_shape_attr = tuple(op.attributes["dense_shape"])
    rank = len(dense_shape_attr)
    _validate_canonical(ctx, name, idx, vals, dense_shape_attr)
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    dense_elems = _dense_elements(dens, batch, rank, name)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_dtype = np.result_type(vals.dtype, dens.dtype)
    out_vals = np.empty((n, vals.shape[-1]), dtype=out_dtype)
    for b in range(n):
        gathered = dense_elems[b][tuple(idx_flat[b].T)]
        out_vals[b] = vals_flat[b] * gathered
    return core.Tensor(idx), core.Tensor(out_vals.reshape(vals.shape))


# ---------------------------------------------------------------------------
# reduce / transpose / reshape / concatenate
# ---------------------------------------------------------------------------


def _reduce_sum(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``sparse_reduce_sum``: dense accumulation + numpy reduction.

    ``axes`` refer to the unbatched sparse axes (int or tuple/list; an EMPTY
    tuple = NO reduction — identity dense, matching ``infer_sparse_reduce_sum``
    which keeps every dim for empty ``axes``; the frontend must emit the full
    axis tuple for an all-axes reduction). In-range is checked (sortedness/
    duplicates are NOT assumed — stored zeros from batched padding are
    legal). The result dtype is the op-DECLARED ``dtype`` attribute:
    accumulation happens in that dtype (the zeros buffer) and the final
    per-element result is converted to it.
    """
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    dense_shape_attr = tuple(op.attributes["dense_shape"])
    rank = len(dense_shape_attr)
    if rank == 0:
        raise core.ShapeError(
            f"kernel for op '{name}': rank-0 sparse tensors are not supported"
        )
    # in-range only — see module docstring for why sortedness is not assumed
    _validate_canonical(ctx, name, idx, vals, dense_shape_attr, check_sorted=False)
    dense_shape = _evaluate_dense_shape(ctx, name, dense_shape_attr)
    out_dtype = core.dtype(op.attributes["dtype"])
    axes = op.attributes.get("axes", ())
    if isinstance(axes, int) and not isinstance(axes, bool):
        axes = (axes,)
    axes = tuple(axes)
    axes_norm = tuple(sorted({a % rank for a in axes}))
    keepdims = bool(op.attributes.get("keepdims", False))
    reduced_set = set(axes_norm)
    out_shape_elem = tuple(
        1 if (i in reduced_set and keepdims) else d
        for i, d in enumerate(dense_shape)
        if (i not in reduced_set) or keepdims
    )
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out = np.empty((n,) + out_shape_elem, dtype=out_dtype)
    all_axes = axes_norm == tuple(range(rank))
    for b in range(n):
        if all_axes:
            res = np.asarray(vals_flat[b].sum())
            if keepdims:
                res = res.reshape((1,) * rank)
        else:
            de = np.zeros(dense_shape, dtype=out_dtype)
            np.add.at(de, tuple(idx_flat[b].T), vals_flat[b])
            res = np.asarray(np.sum(de, axis=axes_norm, keepdims=keepdims))
        out[b] = res.astype(out_dtype, copy=False)
    return core.Tensor(out.reshape(batch + out_shape_elem))


def _transpose(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_transpose``: permute coordinates (per batch element),
    re-sort lexicographically, permute values. The ``dense_shape`` attribute
    is the RESULT shape; the input is canonical-validated against the
    inverse-permuted shape."""
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    result_shape_attr = tuple(op.attributes["dense_shape"])
    perm = tuple(op.attributes["perm"])
    rank = len(result_shape_attr)
    if len(perm) != rank or sorted(perm) != list(range(rank)):
        raise core.ShapeError(
            f"kernel for op '{name}': perm {perm!r} is not a permutation of "
            f"{rank} axes"
        )
    inv_perm = [0] * rank
    for pos, p in enumerate(perm):
        inv_perm[p] = pos
    input_shape_attr = tuple(result_shape_attr[inv_perm[a]] for a in range(rank))
    _validate_canonical(ctx, name, idx, vals, input_shape_attr)
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out_idx = np.empty_like(idx_flat)
    out_vals = np.empty_like(vals_flat)
    for b in range(n):
        new = idx_flat[b][:, perm]
        order = _lexsort_rows(new)
        out_idx[b] = new[order]
        out_vals[b] = vals_flat[b][order]
    return core.Tensor(out_idx.reshape(idx.shape)), core.Tensor(out_vals.reshape(vals.shape))


def _reshape(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_reshape``: linearize with ``old_shape``, re-map with the
    result ``dense_shape``, re-sort. Static shapes only; an element-count
    mismatch is an explicit ``ShapeError`` (never silent garbage)."""
    name = op.name
    indices, values = operands
    idx, vals = indices.numpy(), values.numpy()
    old_shape = _evaluate_static_shape(ctx, name, op.attributes["old_shape"], "old_shape")
    new_shape = _evaluate_static_shape(ctx, name, op.attributes["dense_shape"], "dense_shape")
    if len(old_shape) == 0 or len(new_shape) == 0:
        raise core.ShapeError(
            f"kernel for op '{name}': rank-0 sparse tensors are not supported"
        )
    _validate_canonical(ctx, name, idx, vals, old_shape)
    total_new = math.prod(new_shape)
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    nnz = idx.shape[-2]
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    # NOTE: the result rank differs from the input rank — allocate from the
    # NEW shape (np.empty_like(idx_flat) would silently broadcast a rank-1
    # coordinate row into a rank-2 buffer via numpy's trailing-dim rules).
    out_idx = np.empty((n, nnz, len(new_shape)), _INT64)
    out_vals = np.empty((n, nnz), dtype=vals.dtype)
    for b in range(n):
        linear = np.ravel_multi_index(idx_flat[b].T, old_shape)
        if linear.size and (int(linear.min()) < 0 or int(linear.max()) >= total_new):
            raise core.ShapeError(
                f"kernel for op '{name}': reshape element-count mismatch — "
                f"old_shape {old_shape} and new shape {new_shape} disagree (a "
                f"linear index {int(linear.max())} exceeds the new element "
                f"count {total_new})"
            )
        new = np.stack(np.unravel_index(linear, new_shape), axis=-1).astype(_INT64)
        order = _lexsort_rows(new)
        out_idx[b] = new[order]
        out_vals[b] = vals_flat[b][order]
    return (
        core.Tensor(out_idx.reshape(batch + (nnz, len(new_shape)))),
        core.Tensor(out_vals.reshape(vals.shape)),
    )


def _concatenate(ctx: Any, op: Any, operands: tuple) -> tuple:
    """``sparse_concatenate`` (variadic (indices, values) pairs).

    REQUIRED ``operand_extents`` attribute: per-operand ``dense_shape[axis]``
    extents (the opdef amendment declaring it is pending upstream — this
    kernel still requires it; MISSING => ``BackendError`` naming the op).
    Axis coordinates are offset by the prefix sum of the extents; the
    concatenated result is RE-SORTED (disjoint axis ranges do not imply
    global lex order unless ``axis == 0``).
    """
    name = op.name
    if len(operands) < 4 or len(operands) % 2 != 0:
        raise core.BackendError(
            f"kernel for op '{name}': expected an even number of operands "
            f">= 4 (ia, va, ib, vb, ...), got {len(operands)}"
        )
    if "operand_extents" not in op.attributes:
        raise core.BackendError(
            f"kernel for op '{name}': missing required attribute "
            "'operand_extents' (per-operand dense_shape[axis] extents) — "
            "sparse_concatenate cannot be executed without it"
        )
    axis = int(op.attributes.get("axis", 0))
    result_shape = _evaluate_static_shape(ctx, name, op.attributes["dense_shape"], "dense_shape")
    rank = len(result_shape)
    if not 0 <= axis < rank:
        raise core.ShapeError(
            f"kernel for op '{name}': axis {axis} out of range for rank {rank}"
        )
    extents_attr = op.attributes["operand_extents"]
    n_operands = len(operands) // 2
    if not isinstance(extents_attr, (tuple, list)) or len(extents_attr) != n_operands:
        raise core.BackendError(
            f"kernel for op '{name}': operand_extents must be a list of "
            f"{n_operands} int(s), got {extents_attr!r}"
        )
    extents = tuple(
        e
        if isinstance(e, int) and not isinstance(e, bool)
        else _evaluate_static_shape(ctx, name, (e,), "operand_extents")[0]
        for e in extents_attr
    )
    if sum(extents) != result_shape[axis]:
        raise core.ShapeError(
            f"kernel for op '{name}': sum of operand_extents {extents} does "
            f"not match the result dense_shape[{axis}] = {result_shape[axis]}"
        )
    pairs = []
    batch = None
    for k in range(n_operands):
        ia_t, va_t = operands[2 * k].numpy(), operands[2 * k + 1].numpy()
        if batch is None:
            batch = ia_t.shape[:-2]
        elif ia_t.shape[:-2] != batch:
            raise core.ShapeError(
                f"kernel for op '{name}': operand {k} batch dims "
                f"{tuple(ia_t.shape[:-2])} do not match operand 0's "
                f"{tuple(batch)}"
            )
        shape_k = tuple(
            result_shape[a] if a != axis else extents[k] for a in range(rank)
        )
        _validate_canonical(ctx, name, ia_t, va_t, shape_k)
        pairs.append((ia_t, va_t))
    out_dtype = np.result_type(*[va_t.dtype for _, va_t in pairs])
    nnz_total = sum(int(ia_t.shape[-2]) for ia_t, _ in pairs)
    n = _batch_count(batch)
    # offsets[k] = sum(extents[:k]) — the leading 0 makes the last operand's
    # offset in range (cumsum(extents[:-1]) alone is one short)
    offsets = (0,) + tuple(np.cumsum(extents[:-1]))
    out_idx = np.empty((n, nnz_total, rank), _INT64)
    out_vals = np.empty((n, nnz_total), dtype=out_dtype)
    for b in range(n):
        parts_i = []
        parts_v = []
        for k, (ia_t, va_t) in enumerate(pairs):
            i2 = ia_t.reshape((n,) + ia_t.shape[-2:])[b].copy()
            v2 = va_t.reshape((n,) + va_t.shape[-1:])[b]
            i2[:, axis] += offsets[k]
            parts_i.append(i2)
            parts_v.append(v2.astype(out_dtype, copy=False))
        merged_i = np.concatenate(parts_i, axis=0)
        merged_v = np.concatenate(parts_v, axis=0)
        order = _lexsort_rows(merged_i)
        out_idx[b] = merged_i[order]
        out_vals[b] = merged_v[order]
    return (
        core.Tensor(out_idx.reshape(batch + (nnz_total, rank))),
        core.Tensor(out_vals.reshape(batch + (nnz_total,))),
    )


# ---------------------------------------------------------------------------
# sparse @ dense / dense @ sparse
# ---------------------------------------------------------------------------


def _dot_dense(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``sparse_dot_dense``: rank-2 sparse (M, K) x dense (..., K, N) matmul.

    ``np.add.at`` over the row index (duplicate row targets accumulate).
    The sparse operand is canonical-validated; the dense's inner dim must
    equal K.
    """
    name = op.name
    indices, values, dense = operands
    idx, vals, dens = indices.numpy(), values.numpy(), dense.numpy()
    dense_shape = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    m, k = dense_shape
    _validate_canonical(ctx, name, idx, vals, dense_shape)
    if dens.ndim < 2:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand must have rank >= 2, got "
            f"rank {dens.ndim}"
        )
    if dens.shape[-2] != k:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand inner dim {dens.shape[-2]} "
            f"does not match the sparse K dim {k} (dense_shape {dense_shape!r})"
        )
    n_cols = dens.shape[-1]
    out_dtype = np.result_type(vals.dtype, dens.dtype)
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    dense_elems = _dense_elements(dens, batch, 2, name)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out = np.zeros((n, m, n_cols), dtype=out_dtype)
    for b in range(n):
        de = dense_elems[b]  # (K, N)
        rows = idx_flat[b][:, 0]
        cols = idx_flat[b][:, 1]
        contrib = vals_flat[b][:, None] * de[cols]
        np.add.at(out[b], rows, contrib)
    return core.Tensor(out.reshape(batch + (m, n_cols)))


def _dense_dot_sparse(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``dense_dot_sparse``: dense (..., M, K) x rank-2 sparse (K, N) matmul.

    ``np.add.at`` over the column index (duplicate column targets
    accumulate). The sparse operand is canonical-validated; the dense's
    inner dim must equal K.
    """
    name = op.name
    dense, indices, values = operands
    dens, idx, vals = dense.numpy(), indices.numpy(), values.numpy()
    dense_shape = _rank2_shape(ctx, name, op.attributes["dense_shape"])
    k, n_cols = dense_shape
    _validate_canonical(ctx, name, idx, vals, dense_shape)
    if dens.ndim < 2:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand must have rank >= 2, got "
            f"rank {dens.ndim}"
        )
    if dens.shape[-1] != k:
        raise core.ShapeError(
            f"kernel for op '{name}': dense operand inner dim {dens.shape[-1]} "
            f"does not match the sparse K dim {k} (dense_shape {dense_shape!r})"
        )
    m = dens.shape[-2]
    out_dtype = np.result_type(vals.dtype, dens.dtype)
    batch = idx.shape[:-2]
    n = _batch_count(batch)
    dense_elems = _dense_elements(dens, batch, 2, name)
    idx_flat = idx.reshape((n,) + idx.shape[-2:])
    vals_flat = vals.reshape((n,) + vals.shape[-1:])
    out = np.zeros((n, m, n_cols), dtype=out_dtype)
    for b in range(n):
        de = dense_elems[b]  # (M, K)
        rows = idx_flat[b][:, 0]
        cols = idx_flat[b][:, 1]
        contrib = de[:, rows] * vals_flat[b][None, :]
        np.add.at(out[b], (slice(None), cols), contrib)
    return core.Tensor(out.reshape(batch + (m, n_cols)))


def register_kernels(table: dict) -> None:
    """Register this module's sparse kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["sparse_from_dense"] = _from_dense
    table["sparse_to_dense"] = _to_dense
    table["sparse_coo_to_csr"] = _coo_to_csr
    table["sparse_csr_to_coo"] = _csr_to_coo
    table["sparse_coo_to_csc"] = _coo_to_csc
    table["sparse_csc_to_coo"] = _csc_to_coo
    table["sparse_negate"] = _negate
    table["sparse_add"] = _add
    table["sparse_multiply"] = _multiply
    table["sparse_multiply_dense"] = _multiply_dense
    table["sparse_reduce_sum"] = _reduce_sum
    table["sparse_transpose"] = _transpose
    table["sparse_reshape"] = _reshape
    table["sparse_concatenate"] = _concatenate
    table["sparse_dot_dense"] = _dot_dense
    table["dense_dot_sparse"] = _dense_dot_sparse
