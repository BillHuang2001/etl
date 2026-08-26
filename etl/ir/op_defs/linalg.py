"""Linear-algebra op defs: dot, conv, tril, triu, solve — all ``pure``."""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import infer_conv, infer_dot, infer_identity, infer_solve
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


_register_linalg()
