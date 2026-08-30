"""Linear-algebra op defs: dot, conv, tril, triu, solve, sort, diagonal,
eigh, cholesky, qr, matrix_rank, svd, matrix_exp — all ``pure``."""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import (
    infer_cholesky,
    infer_conv,
    infer_diagonal,
    infer_dot,
    infer_eigh,
    infer_identity,
    infer_matrix_exp,
    infer_matrix_rank,
    infer_qr,
    infer_solve,
    infer_svd,
)
from . import (
    ATTR_ANY,
    ATTR_BOOL,
    ATTR_INT,
    ATTR_INTS,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "linalg"


def _register_linalg() -> None:
    register_opdef(
        OpDef(
            name="dot",
            category=_CATEGORY,
            description="Generalized dot product (numpy matmul contract: "
            "vector/matrix/batched).",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_dot,
        )
    )
    register_opdef(
        OpDef(
            name="conv",
            category=_CATEGORY,
            description="N-dimensional convolution (cross-correlation, numpy-style).",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="strides",
                    type=ATTR_INTS,
                    default=None,
                    description="Per-spatial-dim stride (None = all ones).",
                ),
                AttrSpec(
                    name="padding",
                    type=ATTR_ANY,
                    default="VALID",
                    description="'VALID' | 'SAME' | per-spatial-dim (lo, hi) pairs.",
                ),
                AttrSpec(
                    name="input_dilation",
                    type=ATTR_INTS,
                    default=None,
                    description="Input dilation (None = all ones).",
                ),
                AttrSpec(
                    name="kernel_dilation",
                    type=ATTR_INTS,
                    default=None,
                    description="Kernel dilation (None = all ones).",
                ),
                AttrSpec(
                    name="feature_group_count",
                    type=ATTR_INT,
                    default=1,
                    description="Number of feature groups (grouped conv).",
                ),
                AttrSpec(
                    name="batch_group_count",
                    type=ATTR_INT,
                    default=1,
                    description="Number of batch groups (batch-grouped conv).",
                ),
            ),
            shape_fn=infer_conv,
        )
    )
    register_opdef(
        OpDef(
            name="tril",
            category=_CATEGORY,
            description="Lower triangle of the last two dims (shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="k",
                    type=ATTR_INT,
                    default=0,
                    description="Diagonal offset (numpy convention).",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="triu",
            category=_CATEGORY,
            description="Upper triangle of the last two dims (shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="k",
                    type=ATTR_INT,
                    default=0,
                    description="Diagonal offset (numpy convention).",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="sort",
            category=_CATEGORY,
            description="Sort along an axis in ascending order (numpy semantics).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=-1,
                    description="Sort axis (numpy convention).",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="diagonal",
            category=_CATEGORY,
            description="Extract the (axis1, axis2) diagonal with an offset "
            "(numpy semantics); the diagonal becomes a new last axis.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="offset",
                    type=ATTR_INT,
                    default=0,
                    description="Diagonal offset (numpy convention).",
                ),
                AttrSpec(
                    name="axis1",
                    type=ATTR_INT,
                    default=0,
                    description="First diagonal axis.",
                ),
                AttrSpec(
                    name="axis2",
                    type=ATTR_INT,
                    default=1,
                    description="Second diagonal axis.",
                ),
            ),
            shape_fn=infer_diagonal,
        )
    )
    register_opdef(
        OpDef(
            name="solve",
            category=_CATEGORY,
            description="Solve a x = b (or x a = b with left_side=False) for x.",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="left_side",
                    type=ATTR_BOOL,
                    default=True,
                    description="True: solve a x = b; False: solve x a = b.",
                ),
            ),
            shape_fn=infer_solve,
        )
    )
    register_opdef(
        OpDef(
            name="eigh",
            category=_CATEGORY,
            description="Hermitian/symmetric eigendecomposition (numpy "
            "linalg.eigh): ascending real eigenvalues w and eigenvectors v.",
            arity=1,
            result_count=2,
            effect=EFFECT_PURE,
            shape_fn=infer_eigh,
        )
    )
    register_opdef(
        OpDef(
            name="cholesky",
            category=_CATEGORY,
            description="Lower-triangular Cholesky factor (numpy "
            "linalg.cholesky); non-PD input is a runtime error.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_cholesky,
        )
    )
    register_opdef(
        OpDef(
            name="qr",
            category=_CATEGORY,
            description="QR factorization, numpy reduced mode "
            "(full_matrices=False): q (m, k), r (k, n) with k = min(m, n).",
            arity=1,
            result_count=2,
            effect=EFFECT_PURE,
            shape_fn=infer_qr,
        )
    )
    register_opdef(
        OpDef(
            name="matrix_rank",
            category=_CATEGORY,
            description="Numerical rank via SVD (numpy linalg.matrix_rank): "
            "int64 count per batch element; tol is the static threshold "
            "(None = numpy auto).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="tol",
                    type=ATTR_ANY,
                    default=None,
                    description="Rank threshold; None = numpy's automatic "
                    "max(m, n) * eps * largest-singular-value cutoff.",
                ),
            ),
            shape_fn=infer_matrix_rank,
        )
    )
    register_opdef(
        OpDef(
            name="svd",
            category=_CATEGORY,
            description="Singular value decomposition, numpy "
            "full_matrices=False: u (m, k), s (k,), vh (k, n), k = min(m, n); "
            "s is real at the input's precision.",
            arity=1,
            result_count=3,
            effect=EFFECT_PURE,
            shape_fn=infer_svd,
        )
    )
    register_opdef(
        OpDef(
            name="matrix_exp",
            category=_CATEGORY,
            description="Matrix exponential (scipy/torch linalg semantics — "
            "numpy has no matrix_exp): square matrices over the last two "
            "dims, batch supported, dtype preserved (int/bool -> float64).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_matrix_exp,
        )
    )


_register_linalg()
