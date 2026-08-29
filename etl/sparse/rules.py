"""Sparse batching rules + batched-output aux remap (vectorize/vmap support).

One shared batching rule serves all 16 sparse ops. Sparse ops are
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

Import contract: this module imports `etl.core`, `etl.ir`, `etl.trace`
(active-builder hook — the same hook `etl/sparse/ops.py` uses), the public
`etl.transforms` registration names, the internal `MappedAxes`/`UNMAPPED`
metadata (exactly like `etl/transforms/rules.py`), and `etl.sparse.value`
(NOT `etl.sparse` — this module is imported BY `etl/sparse/__init__.py`).
Never backends/pipeline/persist.
"""
from __future__ import annotations

from etl import core
from etl import ir
from etl.trace import current_builder
from etl.transforms import register_batched_aux_remap, register_batching_rule
from etl.transforms._metadata import MappedAxes, UNMAPPED

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


def _register_sparse_rules() -> None:
    """Install the 16 sparse batching rules + the sparse aux remap.

    Runs at import time (this module is imported at the end of
    `etl/sparse/__init__.py`), so `import etl.sparse` registers everything.
    """
    for name in _SPARSE_OPS:
        register_batching_rule(name, _sparse_batching)
    register_batched_aux_remap(SparseTensor, _sparse_aux_remap)


_register_sparse_rules()
