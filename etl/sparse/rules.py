"""Sparse batching rules + batched-output aux remap + VJP rules + bilinear
JVP rules.

**Batching** (vectorize/vmap support): one shared batching rule serves all 16
sparse ops. Sparse ops are
**batch-transparent**: their leading batch dims flow from the input operand
types through the op's inference hook (`etl/ir/inference.py`) into every
result type, and their attributes (`dense_shape`, `dtype`, `axes`, `perm`,
`old_shape`, `axis`, `operand_extents`, ...) describe the UNBATCHED sparse
value and stay **per-element** — they must never gain the batch dim (inference
derives the batch from the input types, so prepending would DOUBLE it in the
result shapes; the OUTPUT-tree `dense_shape` gains the batch dim via the
batched-aux remap instead).

The rule therefore rebuilds the SAME op over the already-rewritten (batched)
operands with the SAME attributes and lets inference produce the batched
result types; every result is mapped along the full leading batch
(`MappedAxes(range(mapped_count))`).

**VJP** (grad/vjp support): one rule per sparse op, registered under the exact
IR op names. All rules follow the `etl.transforms.autodiff` contract —
``rule(op, cotangents, primals)`` where ``cotangents`` align with
``op.results``, ``primals`` are the op's operand values, and the return tuple
aligns with ``op.operands`` (``None`` / ``ZeroTangent`` = zero gradient). The
rule receives a PROXY op (original attributes/location, transformed
operand/result values) and builds IR over ``primals`` / ``proxy.results``
with the frontend ``etl.ops`` / ``etl.sparse.ops`` — never re-traces.

**JVP** (jvp support): explicit rules for the FOUR bilinear sparse ops
(``sparse_multiply``, ``sparse_multiply_dense``, ``sparse_dot_dense``,
``dense_dot_sparse`` — the ops whose output depends on TWO differentiable
operands, so their JVP is a two-term product rule); everything else derives
its JVP from its VJP rule via the adjoint (double-vjp) trick in
`etl.transforms.autodiff._jvp_from_vjp`. JVP rules follow the
``rule(op, tangents) -> output_tangents`` contract (tangents align with
``op.operands``, returns align with ``op.results``; sparse results return
``(ZeroTangent, values_tangent)``, dense results ``(dense_tangent,)``).
``sparse_multiply``'s rule is a GATHER-based product rule: the intersection
merge (`np.intersect1d`) reorders both inputs' rows into the merged order, so
each input's tangent and primal values are gathered to the merged rows
(``_row_lookup``) before multiplying — the naive elementwise
``add(multiply(ta, vb), multiply(va, tb))`` would pair the WRONG rows.

Sparse vjp semantics (binding): the structure (indices) is never
differentiated — every rule returns ``ZeroTangent`` for index operands. Value
cotangents are pulled back with ordinary dense ops:

- dense-side gathers/scatters at the index rows use a FLAT row-major index
  (``indices @ strides``) because the numpy backend's ``gather``/``scatter``
  kernels are single-axis only;
- the merged-index row lookup (``sparse_add`` / ``sparse_multiply`` /
  ``sparse_transpose``) broadcasts row-equality of the input rows against the
  merged rows + ``argmax`` + ``where`` (``_row_lookup`` in
  ``etl/sparse/_utils.py``) — O(nnz²), acceptable;
- ``sparse_add``'s values cotangent is the merged cotangent gathered at each
  input row's merged position (union — every row survives);
- ``sparse_multiply``'s values cotangent is the merged cotangent WEIGHTED by
  the other operand's value at the matched row (``d(sum vm)/dv_a = v_b``),
  then gathered at each input row's merged position and masked where the row
  is absent from the intersection;
- ``sparse_coo_to_csr`` / ``sparse_csr_to_coo`` / ``sparse_reshape`` pass the
  values cotangent through 1:1 (no reorder — COO lex-sorted IS row-major);
- ``sparse_coo_to_csc`` / ``sparse_csc_to_coo`` / ``sparse_concatenate`` are
  explicit ``core.TransformError`` deferrals (they reorder / split rows, which
  needs a sort-based un-permutation / dynamic slicing — never silent);
- batched sparse operands (indices rank > 2) are a v1 gap in the
  flat-index/lookup rules → explicit ``core.TransformError`` (the passthrough
  rules are batch-agnostic).

Guards (never silent):

- If NO operand is mapped the op is rebuilt unchanged and every result is
  UNMAPPED (results must still be rebuilt: `op.results` are the OLD module's
  values, and the machinery remaps results by value identity).
- Every sparse pair-lead operand (the operand whose type supplies the result
  batch dims — position 0 of each (indices, values) pair; position 1 for
  ``dense_dot_sparse``'s (dense, indices, values) triple) must carry the FULL
  mapped count when anything is mapped, otherwise the result batch would
  silently lose axes → `core.TransformError`. Unmapped DENSE operands (e.g. a
  shared weight in `sparse_multiply_dense` / `sparse_dot_dense`) pass through
  as-is — the batch comes from the mapped sparse side.

Import contract: this module imports `etl.core`, `etl.ir`, `etl.ops`,
`etl.trace` (active-builder hook — the same hook `etl/sparse/ops.py` uses),
the public `etl.transforms` registration names, the internal
`MappedAxes`/`UNMAPPED` metadata (exactly like `etl/transforms/rules.py`),
`ZeroTangent` from `etl.transforms.autodiff`, and `etl.sparse.value` /
`etl.sparse._utils`. The JVP rules build replacement ops with the
`etl.sparse.ops` FRONTEND (imported as `sparse_ops` — safe: `etl/sparse/
__init__.py` imports `ops` before `rules`, so the module is fully loaded
whenever this file is imported). Never backends/pipeline/persist.
"""
from __future__ import annotations

import math

import numpy as np

from etl import core
from etl import ir
from etl import ops
from etl.trace import current_builder
from etl.transforms import register_batched_aux_remap, register_batching_rule
from etl.transforms.autodiff import ZeroTangent, register_jvp_rule, register_vjp_rule
from etl.transforms._metadata import MappedAxes, UNMAPPED

from etl.sparse import ops as sparse_ops
from etl.sparse._utils import _raw_reshape, _row_lookup
from etl.sparse.value import SparseTensor

#: The 16 sparse ops, keyed by their exact IR op name (note the last one
#: registers under the exact name ``"dense_dot_sparse"`` — not sparse_-prefixed).
_SPARSE_OPS = (
    "sparse_from_dense",      # dense -> (indices, values)
    "sparse_to_dense",        # (indices, values) -> dense
    "sparse_coo_to_csr",      # (indices, values) -> (indptr, indices, values)
    "sparse_csr_to_coo",      # (indptr, indices, values) -> (indices, values)
    "sparse_coo_to_csc",      # (indices, values) -> (indptr, indices, values)
    "sparse_csc_to_coo",      # (indptr, indices, values) -> (indices, values)
    "sparse_negate",          # (indices, values) -> (indices, values)
    "sparse_add",             # (ia, va, ib, vb) -> (indices, values)
    "sparse_multiply",        # (ia, va, ib, vb) -> (indices, values)
    "sparse_multiply_dense",  # (indices, values, dense) -> (indices, values)
    "sparse_reduce_sum",      # (indices, values) -> dense
    "sparse_transpose",       # (indices, values) -> (indices, values)
    "sparse_reshape",         # (indices, values) -> (indices, values)
    "sparse_concatenate",     # (ia, va, ib, vb, ...) -> (indices, values)
    "sparse_dot_dense",       # (indices, values, dense) -> dense
    "dense_dot_sparse",       # (dense, indices, values) -> dense
)

#: Per-op tuple of sparse PAIR-LEAD operand positions — the operands whose
#: types supply the result batch dims (per the inference hooks in
#: `etl/ir/inference.py`). Everything is position 0 (the indices operand of
#: each (indices, values) pair) except ``dense_dot_sparse``, whose sparse
#: operand sits at position 1.
_PAIR_LEADS = {
    "sparse_add": (0, 2),
    "sparse_multiply": (0, 2),
    "sparse_concatenate": None,  # every even position (computed per op)
    "dense_dot_sparse": (1,),
}


def _pair_leads(op_name: str, n_operands: int) -> tuple:
    """The sparse pair-lead operand positions of `op_name` (0-indexed)."""
    leads = _PAIR_LEADS.get(op_name)
    if leads is not None:
        return leads
    if op_name == "sparse_concatenate":
        return tuple(range(0, n_operands, 2))
    return (0,)


def _sparse_batching(op: ir.Op, operands, axes):
    """Shared batching rule for the 16 sparse ops.

    Rebuilds `op` over the already-rewritten (batched) `operands` with the
    SAME name and SAME attributes (never batch-prefixed — inference derives
    the result batch dims from the input types). Result types come from
    inference on the batched operand types; every result maps along the full
    leading batch.
    """
    counts = tuple(ax.count for ax in axes)
    mapped_count = max(counts) if counts else 0
    if mapped_count:
        leads = _pair_leads(op.name, len(operands))
        for index in leads:
            if counts[index] != mapped_count:
                raise core.TransformError(
                    f"vectorize: cannot batch op '{op.name}': sparse operand "
                    f"{index} carries {counts[index]} mapped axes but the op "
                    f"maps {mapped_count} (the result batch dims always come "
                    "from the sparse pair-lead operand types; an unmapped "
                    "sparse operand cannot be broadcast across the batch in "
                    "v1)"
                )
    builder = current_builder()
    rebuilt = builder.create(
        op.name,
        operands=tuple(operands),
        attributes=dict(op.attributes),
        location=op.location,
    )
    if mapped_count == 0:
        return tuple(rebuilt.results), (UNMAPPED,) * len(op.results)
    result_axes = MappedAxes(tuple(range(mapped_count)))
    return tuple(rebuilt.results), (result_axes,) * len(op.results)


def _sparse_aux_remap(node, static_values, batch_dim):
    """Batched-output aux remap for the sparse pytree node.

    At the vectorize boundary (`vectorize_graph` step 6) a mapped sparse
    OUTPUT gains its leading batch dim on the flat tensor leaves
    (indices/values); the node's static leaves — `(*dense_shape, dtype,
    format)`, the direct static-leaf children in child order — must gain the
    SAME batch `Dim` at dense_shape position 0 so the runtime rebuilds a
    concrete sparse tensor whose dense_shape is `(batch_dim, *dense_shape)`.
    """
    dense_shape = static_values[:-2]
    dtype_leaf, format_leaf = static_values[-2:]
    return (batch_dim, *dense_shape, dtype_leaf, format_leaf)


# --- VJP rules (all 16 sparse ops) -----------------------------------------
#
# Contract (etl/transforms/autodiff.py): rule(op, cotangents, primals) ->
# tuple aligned with op.operands; cotangents align with op.results; entries
# are ir.Value | None | ZeroTangent. Every rule short-circuits structurally
# zero cotangents BEFORE building gather/scatter ops (the `_ok` pattern —
# mirror of `etl/transforms/rules.py`). The structure (indices) is never
# differentiated: index operands always get ZeroTangent.

def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an ir.Value as a SymbolicTensor (dtype/shape from its type)."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def _ok(tangent):
    """The tangent as an ir.Value, or None when structurally zero."""
    if tangent is None or isinstance(tangent, ZeroTangent):
        return None
    return tangent


def _require_unbatched(value: ir.Value, op_name: str, expected_rank: int) -> None:
    """Reject batched sparse operands in the flat-index/lookup AD rules.

    The dense-side gather/scatter construction below flattens the sparse
    dense_shape into one row-major index space; with leading batch dims that
    flattening would silently mix batch elements (per-batch offsets are
    runtime values) — an explicit v1 gap, never a silent wrong gradient.
    """
    rank = value.type.rank
    if rank != expected_rank:
        raise core.TransformError(
            f"grad/vjp/jvp: differentiation of '{op_name}' is implemented for "
            f"unbatched sparse operands only (got rank {rank}, expected "
            f"{expected_rank}) — batched sparse differentiation is a v1 gap"
        )


def _flat_strides(dense_shape: tuple) -> tuple:
    """Row-major flattening strides of a fully static dense shape.

    Raises:
        core.TransformError: a symbolic/None dim (the flat-index construction
            needs static strides — v1 gap).
    """
    strides = []
    acc = 1
    for dim in reversed(dense_shape):
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise core.TransformError(
                f"grad/vjp: cannot flatten a sparse dense_shape with symbolic "
                f"dim {dim!r} — the vjp needs fully static dense shapes (v1 "
                "gap)"
            )
        strides.append(acc)
        acc *= dim
    return tuple(reversed(strides))


def _flat_indices(indices: "core.SymbolicTensor", dense_shape: tuple, location) -> "core.SymbolicTensor":
    """(nnz,) flat row-major positions of an (nnz, ndim) indices tensor.

    ``flat = sum_k indices[:, k] * stride_k`` with the static row-major
    strides of `dense_shape` (the numpy backend's ``gather``/``scatter``
    kernels are single-axis only, so the multi-axis index rows are folded
    into one flat index space).
    """
    strides = _flat_strides(dense_shape)
    scaled = ops.multiply(
        indices, ops.constant(core.tensor(np.asarray(strides, dtype=np.int64)))
    )
    return ops.reduce_sum(scaled, axes=(1,))


def _flat_gather(dense: ir.Value, flat_indices, dense_shape: tuple, location) -> "core.SymbolicTensor":
    """Gather a dense tensor's entries at the flat index positions (nnz,)."""
    flat = _raw_reshape(dense, (-1,), location)
    return ops.gather(flat, flat_indices, axis=0)


def _flat_scatter_into_dense(updates, flat_indices, dense_shape: tuple, dtype, location) -> "core.SymbolicTensor":
    """Scatter (nnz,) updates at (nnz,) flat positions into a dense tensor.

    The result is a dense-shaped zero tensor of `dtype` with the updates
    accumulated at the flat index positions (reshaped back to `dense_shape`,
    which is fully static here).
    """
    zeros = ops.constant(
        core.tensor(np.zeros((math.prod(dense_shape),), dtype=dtype))
    )
    scattered = ops.scatter(zeros, flat_indices, updates, axis=0)
    return ops.reshape(scattered, dense_shape)


def _col(indices: "core.SymbolicTensor", index: int, location) -> "core.SymbolicTensor":
    """The `index`-th coordinate column of an (nnz, ndim) indices tensor:
    (nnz,) — a 0-d constant-index gather drops the coord axis."""
    return ops.gather(
        indices,
        ops.constant(core.tensor(np.asarray(index, dtype=np.int64))),
        axis=1,
    )


def _gather_at_rows(cot_merged: ir.Value, input_indices: ir.Value, merged_indices: ir.Value, ndim: int, location) -> ir.Value:
    """Input-values cotangent: the merged cotangent gathered at each input
    row's position in the merged indices (0 where the row is absent).

    Empty-merge safety: the merged cotangent is padded with one zero row so
    the gather never indexes an empty axis (the numpy ``take`` kernel raises
    on a length-0 axis with non-empty indices — the empty intersection of
    ``sparse_multiply`` or an ``nnz=0`` operand). The padded slot is only
    ever read where the mask is False (absent rows -> position 0), and
    ``select`` zeroes those entries — real-row values are unchanged.
    """
    positions, mask = _row_lookup(input_indices, merged_indices, ndim, location)
    cot = _sym(cot_merged)
    dummy = ops.constant(core.tensor(np.zeros((1,), dtype=cot.dtype)))
    padded = ops.concatenate([cot, dummy], axis=0)
    gathered = ops.gather(padded, positions, axis=0)
    return ops.select(mask, gathered, 0).value


def _expand_reduced(v: ir.Value, dense_shape: tuple, axes: tuple, keepdims: bool, location) -> "core.SymbolicTensor":
    """Expand a reduced cotangent back to the full (static) dense shape.

    Mirrors `etl/transforms/rules.py._expand_reduced`: insert size-1 dims at
    the reduced axes (when not kept), then broadcast to the dense shape.
    """
    if keepdims:
        return ops.broadcast(_sym(v), dense_shape)
    target = tuple(1 if i in axes else d for i, d in enumerate(dense_shape))
    reshaped = ops.reshape(_sym(v), target)
    if tuple(reshaped.shape) == tuple(dense_shape):
        return reshaped
    return ops.broadcast(reshaped, dense_shape)


def _vjp_sparse_to_dense(op, cotangents, primals):
    """(indices, values) -> dense: wrt values = the dense cotangent gathered
    at the index rows (flat-index gather); indices non-differentiable."""
    cot = _ok(cotangents[0])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    _require_unbatched(primals[0], op.name, 2)
    dense_shape = tuple(op.attributes["dense_shape"])
    flat = _flat_indices(_sym(primals[0]), dense_shape, op.location)
    g_values = _flat_gather(cot, flat, dense_shape, op.location)
    return (ZeroTangent(), g_values.value)


def _vjp_sparse_from_dense(op, cotangents, primals):
    """dense -> (indices, values): wrt dense = the values cotangent scattered
    at the index rows (flat-index scatter). The indices are the op's FIRST
    RESULT (``proxy.results[0]``) — the op has a single dense operand."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(),)
    dense = primals[0]
    dense_shape = tuple(op.attributes["dense_shape"])
    _require_unbatched(dense, op.name, len(dense_shape))
    flat = _flat_indices(_sym(op.results[0]), dense_shape, op.location)
    g_dense = _flat_scatter_into_dense(
        _sym(cot), flat, dense_shape, dense.type.dtype, op.location
    )
    return (g_dense.value,)


def _vjp_sparse_negate(op, cotangents, primals):
    """(indices, values) -> (indices, -values): wrt values = negate(cot)."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    return (ZeroTangent(), ops.negate(_sym(cot)).value)


def _vjp_sparse_add(op, cotangents, primals):
    """sparse_add (ia, va, ib, vb) -> (im, vm): union merge — each input's
    values cotangent = the merged cotangent gathered at that input's rows'
    positions in the merged indices (every input row survives the union, so
    the mask is all-true). O(nnz^2) row lookup."""
    cot_merged = _ok(cotangents[1])
    if cot_merged is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
    ia, _va, ib, _vb = primals
    _require_unbatched(ia, op.name, 2)
    _require_unbatched(ib, op.name, 2)
    ndim = len(op.attributes["dense_shape"])
    merged = op.results[0]
    g_a = _gather_at_rows(cot_merged, ia, merged, ndim, op.location)
    g_b = _gather_at_rows(cot_merged, ib, merged, ndim, op.location)
    return (ZeroTangent(), g_a, ZeroTangent(), g_b)


def _vjp_sparse_multiply(op, cotangents, primals):
    """sparse_multiply (ia, va, ib, vb) -> (im, vm): intersection merge — each
    input's values cotangent is the merged cotangent WEIGHTED by the OTHER
    operand's value at the matched row (d(sum vm)/dv_a = v_b), gathered at
    the input row's position in the merged indices and masked where the row
    is absent from the intersection:
        g_a[i] = cot[pos_a[i]] * vb[merged_to_b[pos_a[i]]]
    O(nnz^2) row lookups."""
    cot_merged = _ok(cotangents[1])
    if cot_merged is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
    ia, va, ib, vb = primals
    _require_unbatched(ia, op.name, 2)
    _require_unbatched(ib, op.name, 2)
    ndim = len(op.attributes["dense_shape"])
    merged = op.results[0]
    # each merged row's position in the inputs' index lists (every merged row
    # is present in both inputs — the merge is an intersection)
    merged_to_a, _ = _row_lookup(merged, ia, ndim, op.location)
    merged_to_b, _ = _row_lookup(merged, ib, ndim, op.location)
    weighted_a = ops.multiply(
        _sym(cot_merged), ops.gather(_sym(vb), merged_to_b, axis=0)
    )
    weighted_b = ops.multiply(
        _sym(cot_merged), ops.gather(_sym(va), merged_to_a, axis=0)
    )
    g_a = _gather_at_rows(weighted_a.value, ia, merged, ndim, op.location)
    g_b = _gather_at_rows(weighted_b.value, ib, merged, ndim, op.location)
    return (ZeroTangent(), g_a, ZeroTangent(), g_b)


def _vjp_sparse_multiply_dense(op, cotangents, primals):
    """(indices, values, dense) -> (indices, values):
    wrt values = cot_values * gather(dense at the index rows);
    wrt dense = scatter(cot_values * values at the index rows into zeros)."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent())
    indices, values, dense = primals
    dense_shape = tuple(op.attributes["dense_shape"])
    _require_unbatched(indices, op.name, 2)
    _require_unbatched(dense, op.name, len(dense_shape))
    flat = _flat_indices(_sym(indices), dense_shape, op.location)
    g_values = ops.multiply(_sym(cot), _flat_gather(dense, flat, dense_shape, op.location))
    g_dense = _flat_scatter_into_dense(
        ops.multiply(_sym(cot), _sym(values)),
        flat,
        dense_shape,
        dense.type.dtype,
        op.location,
    )
    return (ZeroTangent(), g_values.value, g_dense.value)


def _vjp_sparse_reduce_sum(op, cotangents, primals):
    """(indices, values) -> dense: wrt values = the dense cotangent expanded
    back to the full dense shape, gathered at the index rows."""
    cot = _ok(cotangents[0])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    _require_unbatched(primals[0], op.name, 2)
    dense_shape = tuple(op.attributes["dense_shape"])
    full = _expand_reduced(
        cot,
        dense_shape,
        tuple(op.attributes["axes"]),
        bool(op.attributes["keepdims"]),
        op.location,
    )
    flat = _flat_indices(_sym(primals[0]), dense_shape, op.location)
    g_values = _flat_gather(full.value, flat, dense_shape, op.location)
    return (ZeroTangent(), g_values.value)


def _vjp_sparse_transpose(op, cotangents, primals):
    """(indices, values) -> (indices, values) with rows permuted by `perm`:
    each input row's cotangent = the result cotangent at that row's position
    in the result indices (the permuted-row lookup)."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    _require_unbatched(primals[0], op.name, 2)
    ndim = len(op.attributes["dense_shape"])
    perm = tuple(op.attributes["perm"])
    permuted = ops.gather(
        _sym(primals[0]),
        ops.constant(core.tensor(np.asarray(perm, dtype=np.int64))),
        axis=-1,
    )
    g_values = _gather_at_rows(cot, permuted.value, op.results[0], ndim, op.location)
    return (ZeroTangent(), g_values)


def _vjp_sparse_reshape(op, cotangents, primals):
    """(indices, values) -> (indices, values): 1:1 row mapping — the values
    cotangent passes through unchanged."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    return (ZeroTangent(), cot)


def _vjp_sparse_coo_to_csr(op, cotangents, primals):
    """(indices, values) -> (indptr, indices, values): no reorder — COO
    lex-sorted IS row-major, so the values cotangent (the THIRD result)
    passes through 1:1."""
    cot = _ok(cotangents[2])
    if cot is None:
        return (ZeroTangent(), ZeroTangent())
    return (ZeroTangent(), cot)


def _vjp_sparse_csr_to_coo(op, cotangents, primals):
    """(indptr, indices, values) -> (indices, values): no reorder (CSR
    expands in row-major order), so the values cotangent passes through 1:1."""
    cot = _ok(cotangents[1])
    if cot is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent())
    return (ZeroTangent(), ZeroTangent(), cot)


def _vjp_deferral(op_name: str, reason: str):
    """Explicit v1-deferral rule: raising ``core.TransformError`` naming the
    op — never a silent fallback (mirrors the conv-vjp precedent)."""

    def rule(op, cotangents, primals):
        raise core.TransformError(
            f"grad/vjp: vjp of {op_name} is a v1 deferral ({reason}) — "
            f"{op_name} cannot be differentiated in v1; densify with "
            "etl.sparse.to_dense before differentiating"
        )

    return rule


def _one_hot_rows(idx_col: "core.SymbolicTensor", k: int, dtype, location) -> "core.SymbolicTensor":
    """One-hot row-selection matrix ``(k, nnz)`` with ``A[k, i] = [idx[i] == k]``.

    Built from ordinary dense ops (``arange`` constant + broadcast ``equal`` +
    ``cast``) in the cotangent dtype. The sparse vjp rules use it to scatter-
    ACCUMULATE dense gradients: ``g_dense = A @ U`` is a dense matmul, which
    accumulates the per-nonzero contributions of rows sharing one target —
    the numpy backend's single-axis ``scatter`` kernel (``put_along_axis``
    overwrite semantics) cannot accumulate, so a direct scatter would silently
    drop all but the last duplicate target.
    """
    arange_k = ops.constant(
        core.tensor(np.arange(k, dtype=np.int64))  # (k,)
    )
    rows = _raw_reshape(arange_k.value, (k, 1), location)
    cols = _raw_reshape(idx_col.value, (1, -1), location)
    return ops.cast(ops.equal(rows, cols), dtype)  # (k, nnz) in cotangent dtype


def _vjp_sparse_dot_dense(op, cotangents, primals):
    """(indices, values, dense) -> dense (M, N), sparse (M, K) x dense (K, N):
    wrt values[i] = sum_n cot[m_i, n] * dense[k_i, n];
    wrt dense (K, N) = A @ U with A[k, i] = [k_i == k] (one-hot) and
    U[i, :] = v_i * cot[m_i, :] — the matmul accumulates the contributions of
    nonzeros sharing a row k (single-axis scatter cannot)."""
    cot = _ok(cotangents[0])
    if cot is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent())
    indices, values, dense = primals
    _require_unbatched(indices, op.name, 2)
    dense_shape = tuple(op.attributes["dense_shape"])  # (M, K)
    k = dense_shape[1]
    if not isinstance(k, int) or isinstance(k, bool):
        raise core.TransformError(
            f"grad/vjp: vjp of 'sparse_dot_dense' needs a static inner dim K "
            f"(dense_shape[1] = {k!r}) to build the accumulation matrix — "
            "symbolic inner dims are a v1 gap"
        )
    idx_m = _col(_sym(indices), 0, op.location)
    idx_k = _col(_sym(indices), 1, op.location)
    g_values = ops.reduce_sum(
        ops.multiply(
            ops.gather(_sym(cot), idx_m, axis=0),
            ops.gather(_sym(dense), idx_k, axis=0),
        ),
        axes=(1,),
    )
    one_hot = _one_hot_rows(idx_k, k, cot.type.dtype, op.location)  # (K, nnz)
    updates = ops.multiply(
        _raw_reshape(values, (-1, 1), op.location),  # (nnz, 1)
        ops.gather(_sym(cot), idx_m, axis=0),        # (nnz, N)
    )
    g_dense = ops.dot(one_hot, updates)  # (K, N)
    return (ZeroTangent(), g_values.value, g_dense.value)


def _vjp_dense_dot_sparse(op, cotangents, primals):
    """(dense, indices, values) -> dense (M, N), dense (M, K) x sparse
    (K, N): with index rows (k_i, n_i) —
    wrt values[i] = sum_m cot[m, n_i] * dense[m, k_i];
    wrt dense (M, K) = V @ A^T with A[k, i] = [k_i == k] (one-hot) and
    V[:, i] = cot[:, n_i] * v_i — the matmul accumulates the contributions of
    nonzeros sharing a column k (single-axis scatter cannot)."""
    cot = _ok(cotangents[0])
    if cot is None:
        return (ZeroTangent(), ZeroTangent(), ZeroTangent())
    dense, indices, values = primals
    _require_unbatched(indices, op.name, 2)
    dense_shape = tuple(op.attributes["dense_shape"])  # (K, N)
    k = dense_shape[0]
    if not isinstance(k, int) or isinstance(k, bool):
        raise core.TransformError(
            f"grad/vjp: vjp of 'dense_dot_sparse' needs a static inner dim K "
            f"(dense_shape[0] = {k!r}) to build the accumulation matrix — "
            "symbolic inner dims are a v1 gap"
        )
    idx_k = _col(_sym(indices), 0, op.location)
    idx_n = _col(_sym(indices), 1, op.location)
    g_values = ops.reduce_sum(
        ops.multiply(
            ops.gather(_sym(cot), idx_n, axis=1),   # (M, nnz)
            ops.gather(_sym(dense), idx_k, axis=1),  # (M, nnz)
        ),
        axes=(0,),
    )
    one_hot = _one_hot_rows(idx_k, k, cot.type.dtype, op.location)  # (K, nnz)
    updates = ops.multiply(
        ops.gather(_sym(cot), idx_n, axis=1),        # (M, nnz)
        _raw_reshape(values, (1, -1), op.location),  # (1, nnz)
    )
    g_dense = ops.dot(updates, ops.transpose(one_hot))  # (M, K)
    return (g_dense.value, ZeroTangent(), g_values.value)


# --- JVP rules (the four bilinear sparse ops) --------------------------------
#
# Contract (etl/transforms/autodiff.py): rule(op, tangents) -> output_tangents;
# tangents align with op.operands (None/ZeroTangent = zero); returns align
# with op.results — sparse results return (ZeroTangent, values_tangent)
# (structure is never differentiated), dense results (dense_tangent,). All
# four outputs depend on TWO differentiable operands, so the rules are
# two-term product rules built with the etl.sparse frontend ops (they build
# IR into the active builder over the proxy's operand/tangent values). Every
# other sparse op's JVP derives from its VJP rule via the adjoint
# (double-vjp) trick — sound because the vjp rules are linear in the
# cotangent and their emitted ops all carry registered vjp rules.

def _sparse_coo_of(indices: ir.Value, values: ir.Value, dense_shape: tuple) -> "SparseTensor":
    """A symbolic COO sparse over the given ir.Values (no validation — the
    leaves are graph values; used to feed the etl.sparse frontend ops)."""
    return SparseTensor.from_parts(
        _sym(indices), _sym(values), dense_shape=dense_shape, format="coo"
    )


def _jvp_sparse_multiply(op, tangents):
    """(ia, va, ib, vb) -> (im, vm): the intersection merge
    (``np.intersect1d``) reorders both inputs' rows into the merged order, so
    the product rule needs each input's tangent AND primal values gathered to
    the merged rows:
        t_vm = gather(ta, pos_a) * gather(vb, pos_b)
             + gather(va, pos_a) * gather(tb, pos_b)
    with pos_a/pos_b = the merged rows' positions in the inputs' index lists
    (`_row_lookup`, the same O(nnz²) machinery as the vjp merge rule). The
    naive elementwise ``add(multiply(ta, vb), multiply(va, tb))`` would pair
    the WRONG rows (the merged order matches neither input's order)."""
    ta, tb = _ok(tangents[1]), _ok(tangents[3])
    if ta is None and tb is None:
        return (ZeroTangent(), ZeroTangent())
    ia, va, ib, vb = op.operands
    _require_unbatched(ia, op.name, 2)
    _require_unbatched(ib, op.name, 2)
    ndim = len(op.attributes["dense_shape"])
    merged = op.results[0]
    pos_a, _ = _row_lookup(merged, ia, ndim, op.location)
    pos_b, _ = _row_lookup(merged, ib, ndim, op.location)
    terms = []
    if ta is not None:
        terms.append(
            ops.multiply(
                ops.gather(_sym(ta), pos_a, axis=0),
                ops.gather(_sym(vb), pos_b, axis=0),
            )
        )
    if tb is not None:
        terms.append(
            ops.multiply(
                ops.gather(_sym(va), pos_a, axis=0),
                ops.gather(_sym(tb), pos_b, axis=0),
            )
        )
    if len(terms) == 1:
        return (ZeroTangent(), terms[0].value)
    return (ZeroTangent(), ops.add(terms[0], terms[1]).value)


def _jvp_sparse_multiply_dense(op, tangents):
    """(indices, values, dense) -> (indices, values): structure preserved and
    the rows stay aligned, so the product rule is elementwise:
        t_values = multiply_dense(indices, ta, dense)
                 + multiply_dense(indices, va, tdense)."""
    ta, td = _ok(tangents[1]), _ok(tangents[2])
    if ta is None and td is None:
        return (ZeroTangent(), ZeroTangent())
    indices, va, dense = op.operands
    dense_shape = tuple(op.attributes["dense_shape"])
    terms = []
    if ta is not None:
        terms.append(
            sparse_ops.multiply_dense(
                _sparse_coo_of(indices, ta, dense_shape), _sym(dense)
            )
        )
    if td is not None:
        terms.append(
            sparse_ops.multiply_dense(
                _sparse_coo_of(indices, va, dense_shape), _sym(td)
            )
        )
    if len(terms) == 1:
        return (ZeroTangent(), terms[0].values.value)
    return (ZeroTangent(), ops.add(terms[0].values, terms[1].values).value)


def _jvp_sparse_dot_dense(op, tangents):
    """(indices, values, dense) -> dense: bilinear in (values, dense):
        t_out = sparse_dot_dense(indices, ta, dense)
              + sparse_dot_dense(indices, va, tdense)."""
    ta, td = _ok(tangents[1]), _ok(tangents[2])
    if ta is None and td is None:
        return (ZeroTangent(),)
    indices, va, dense = op.operands
    dense_shape = tuple(op.attributes["dense_shape"])
    terms = []
    if ta is not None:
        terms.append(
            sparse_ops.matmul(_sparse_coo_of(indices, ta, dense_shape), _sym(dense))
        )
    if td is not None:
        terms.append(
            sparse_ops.matmul(_sparse_coo_of(indices, va, dense_shape), _sym(td))
        )
    if len(terms) == 1:
        return (terms[0].value,)
    return (ops.add(terms[0], terms[1]).value,)


def _jvp_dense_dot_sparse(op, tangents):
    """(dense, indices, values) -> dense: bilinear in (dense, values):
        t_out = dense_dot_sparse(dense, indices, ta)
              + dense_dot_sparse(tdense, indices, va)."""
    td, tv = _ok(tangents[0]), _ok(tangents[2])
    if td is None and tv is None:
        return (ZeroTangent(),)
    dense, indices, va = op.operands
    dense_shape = tuple(op.attributes["dense_shape"])
    terms = []
    if tv is not None:
        terms.append(
            sparse_ops.matmul(_sym(dense), _sparse_coo_of(indices, tv, dense_shape))
        )
    if td is not None:
        terms.append(
            sparse_ops.matmul(_sym(td), _sparse_coo_of(indices, va, dense_shape))
        )
    if len(terms) == 1:
        return (terms[0].value,)
    return (ops.add(terms[0], terms[1]).value,)


def _register_sparse_rules() -> None:
    """Install the 16 sparse batching rules, the sparse aux remap, the 16
    sparse VJP rules (three of which are explicit deferrals), and the 4
    explicit bilinear JVP rules (everything else derives its JVP from its VJP
    rule via the adjoint trick).

    Runs at import time (this module is imported at the end of
    `etl/sparse/__init__.py`), so `import etl.sparse` registers everything.
    """
    for name in _SPARSE_OPS:
        register_batching_rule(name, _sparse_batching)
    register_batched_aux_remap(SparseTensor, _sparse_aux_remap)
    register_vjp_rule("sparse_to_dense", _vjp_sparse_to_dense)
    register_vjp_rule("sparse_from_dense", _vjp_sparse_from_dense)
    register_vjp_rule("sparse_negate", _vjp_sparse_negate)
    register_vjp_rule("sparse_add", _vjp_sparse_add)
    register_vjp_rule("sparse_multiply", _vjp_sparse_multiply)
    register_vjp_rule("sparse_multiply_dense", _vjp_sparse_multiply_dense)
    register_vjp_rule("sparse_reduce_sum", _vjp_sparse_reduce_sum)
    register_vjp_rule("sparse_transpose", _vjp_sparse_transpose)
    register_vjp_rule("sparse_reshape", _vjp_sparse_reshape)
    register_vjp_rule("sparse_coo_to_csr", _vjp_sparse_coo_to_csr)
    register_vjp_rule("sparse_csr_to_coo", _vjp_sparse_csr_to_coo)
    register_vjp_rule(
        "sparse_coo_to_csc",
        _vjp_deferral(
            "sparse_coo_to_csc",
            "reorders values column-major; needs a sort-based un-permutation",
        ),
    )
    register_vjp_rule(
        "sparse_csc_to_coo",
        _vjp_deferral(
            "sparse_csc_to_coo",
            "re-sorts values back to row-major; needs a sort-based "
            "un-permutation",
        ),
    )
    register_vjp_rule(
        "sparse_concatenate",
        _vjp_deferral(
            "sparse_concatenate",
            "needs dynamic slicing of the merged cotangent per operand",
        ),
    )
    register_vjp_rule("sparse_dot_dense", _vjp_sparse_dot_dense)
    register_vjp_rule("dense_dot_sparse", _vjp_dense_dot_sparse)
    register_jvp_rule("sparse_multiply", _jvp_sparse_multiply)
    register_jvp_rule("sparse_multiply_dense", _jvp_sparse_multiply_dense)
    register_jvp_rule("sparse_dot_dense", _jvp_sparse_dot_dense)
    register_jvp_rule("dense_dot_sparse", _jvp_dense_dot_sparse)


_register_sparse_rules()
