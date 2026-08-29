"""Sparse op defs: explicit sparse-tensor structure conversions and
sparse/dense arithmetic — all ``pure``.

A sparse value is an (indices, values) pair: ``indices`` is an int64 tensor
of shape ``(B..., nnz, ndim)`` — leading batch dims, a runtime-dynamic ``nnz``
dim, and one coordinate row per non-zero — and ``values`` holds the non-zero
values with shape ``(B..., nnz)``. The unbatched dense shape of the sparse
tensor is recorded in the ``dense_shape`` attribute and the sparse value
dtype in ``dtype`` (declared on every op below). Rank is always statically
known; ``nnz`` is runtime-dynamic (``None``). Leading batch dims are
propagated from input operand types (never from the attributes).
"""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import (
    infer_sparse_add,
    infer_sparse_concatenate,
    infer_sparse_coo_to_csr,
    infer_sparse_coo_to_csc,
    infer_sparse_csr_to_coo,
    infer_sparse_csc_to_coo,
    infer_sparse_dense_dot_sparse,
    infer_sparse_dot_dense,
    infer_sparse_from_dense,
    infer_sparse_multiply,
    infer_sparse_multiply_dense,
    infer_sparse_negate,
    infer_sparse_reduce_sum,
    infer_sparse_reshape,
    infer_sparse_to_dense,
    infer_sparse_transpose,
)
from . import (
    ATTR_BOOL,
    ATTR_DTYPE,
    ATTR_INT,
    ATTR_INTS,
    ATTR_SHAPE,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "sparse"

#: Attribute pair shared by every sparse op: the unbatched dense shape and
#: the sparse value dtype.
_DENSE_SHAPE_DTYPE = (
    AttrSpec(
        name="dense_shape",
        type=ATTR_SHAPE,
        description="Unbatched dense shape of the sparse tensor (static "
        "ints; a symbolic first dim is allowed after vectorize).",
    ),
    AttrSpec(
        name="dtype",
        type=ATTR_DTYPE,
        description="Sparse value dtype (numpy dtype name).",
    ),
)


def _register_sparse() -> None:
    register_opdef(
        OpDef(
            name="sparse_from_dense",
            category=_CATEGORY,
            description="Convert a dense tensor to a sparse (indices, values) "
            "pair in COO format; nnz is runtime-dynamic.",
            arity=1,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_from_dense,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_to_dense",
            category=_CATEGORY,
            description="Materialize the dense tensor described by the sparse "
            "(indices, values) pair; zeros elsewhere.",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_to_dense,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_coo_to_csr",
            category=_CATEGORY,
            description="Convert a rank-2 COO pair to CSR (indptr, indices, "
            "values); the COO is lex-sorted, i.e. already row-major, so no "
            "reorder happens.",
            arity=2,
            result_count=3,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_coo_to_csr,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_csr_to_coo",
            category=_CATEGORY,
            description="Convert a rank-2 CSR (indptr, indices, values) "
            "triple back to a COO (indices, values) pair.",
            arity=3,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_csr_to_coo,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_coo_to_csc",
            category=_CATEGORY,
            description="Convert a rank-2 COO pair to CSC (indptr, indices, "
            "values), reordering row-major to column-major.",
            arity=2,
            result_count=3,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_coo_to_csc,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_csc_to_coo",
            category=_CATEGORY,
            description="Convert a rank-2 CSC (indptr, indices, values) "
            "triple back to a COO (indices, values) pair, reordering "
            "column-major to row-major.",
            arity=3,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_csc_to_coo,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_negate",
            category=_CATEGORY,
            description="Negate the sparse values; the sparsity structure "
            "(indices) is preserved.",
            arity=2,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_negate,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_add",
            category=_CATEGORY,
            description="Sparse + sparse with a union merge: indices that "
            "appear in either operand, values summed where they overlap.",
            arity=4,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_add,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_multiply",
            category=_CATEGORY,
            description="Sparse * sparse with an intersection merge: indices "
            "that appear in both operands, values multiplied.",
            arity=4,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_multiply,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_multiply_dense",
            category=_CATEGORY,
            description="Sparse * dense elementwise: the sparsity structure "
            "is preserved, values are multiplied by the dense tensor.",
            arity=3,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_multiply_dense,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_reduce_sum",
            category=_CATEGORY,
            description="Sum the sparse values over the unbatched sparse "
            "axes, producing a dense tensor; batch dims are never reduced.",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE
            + (
                AttrSpec(
                    name="axes",
                    type=ATTR_INTS,
                    description="Sparse (unbatched) axes to reduce over.",
                ),
                AttrSpec(
                    name="keepdims",
                    type=ATTR_BOOL,
                    description="Keep reduced axes as size-1 dims.",
                ),
            ),
            shape_fn=infer_sparse_reduce_sum,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_transpose",
            category=_CATEGORY,
            description="Permute the sparse axes; the result dense shape is "
            "recorded in ``dense_shape``.",
            arity=2,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE
            + (
                AttrSpec(
                    name="perm",
                    type=ATTR_INTS,
                    description="Target axis order (a permutation of the "
                    "sparse axes).",
                ),
            ),
            shape_fn=infer_sparse_transpose,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_reshape",
            category=_CATEGORY,
            description="Reshape the sparse tensor to a new dense shape; the "
            "element count must agree (symbolically).",
            arity=2,
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE
            + (
                AttrSpec(
                    name="old_shape",
                    type=ATTR_SHAPE,
                    description="Dense shape of the sparse operand.",
                ),
            ),
            shape_fn=infer_sparse_reshape,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_concatenate",
            category=_CATEGORY,
            description="Concatenate sparse tensors along one sparse axis "
            "(variadic (indices, values) operand pairs).",
            arity=(4, None),
            result_count=2,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE
            + (
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=0,
                    description="Sparse (unbatched) axis to concatenate along.",
                ),
                AttrSpec(
                    name="operand_extents",
                    type=ATTR_INTS,
                    description="dense_shape[axis] per operand, in order (the "
                    "kernel offsets each operand's coordinate by the prefix "
                    "sum).",
                ),
            ),
            shape_fn=infer_sparse_concatenate,
        )
    )
    register_opdef(
        OpDef(
            name="sparse_dot_dense",
            category=_CATEGORY,
            description="Rank-2 sparse (M, K) x dense (..., K, N) matmul; "
            "result is a dense tensor (batch dims from the sparse operand).",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_dot_dense,
        )
    )
    register_opdef(
        OpDef(
            name="dense_dot_sparse",
            category=_CATEGORY,
            description="Dense (..., M, K) x rank-2 sparse (K, N) matmul; "
            "result is a dense tensor (batch dims from the sparse operand).",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=_DENSE_SHAPE_DTYPE,
            shape_fn=infer_sparse_dense_dot_sparse,
        )
    )


_register_sparse()
