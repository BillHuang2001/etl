"""Structure op defs: select, broadcast, reshape, transpose, slice, gather,
scatter, concatenate, pad — all ``pure``."""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import (
    infer_broadcast_to,
    infer_concatenate,
    infer_gather,
    infer_pad,
    infer_reshape,
    infer_scatter,
    infer_select,
    infer_slice,
    infer_transpose,
)
from . import (
    ATTR_FLOAT,
    ATTR_INT,
    ATTR_INTS,
    ATTR_NESTED_INTS,
    ATTR_SHAPE,
    ATTR_STR,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "structure"


def _register_structure() -> None:
    register_opdef(
        OpDef(
            name="select",
            category=_CATEGORY,
            description="Elementwise conditional: select(pred, on_true, on_false); "
            "pred is boolean, all three operands broadcast.",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_select,
        )
    )
    register_opdef(
        OpDef(
            name="broadcast",
            category=_CATEGORY,
            description="Broadcast the operand to the target shape (leading and "
            "size-1 dims may expand; size-1 dims may grow).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="shape",
                    type=ATTR_SHAPE,
                    description="Target shape (tuple of int/DimExpr/None).",
                ),
            ),
            shape_fn=infer_broadcast_to,
        )
    )
    register_opdef(
        OpDef(
            name="reshape",
            category=_CATEGORY,
            description="Reshape the operand to the target shape; element count "
            "must agree (symbolically).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="shape",
                    type=ATTR_SHAPE,
                    description="Target shape; may contain a single -1 wildcard.",
                ),
            ),
            shape_fn=infer_reshape,
        )
    )
    register_opdef(
        OpDef(
            name="transpose",
            category=_CATEGORY,
            description="Permute the operand's dimensions.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="permutation",
                    type=ATTR_INTS,
                    default=None,
                    description="Target axis order (numpy convention: None = "
                    "full reversal).",
                ),
            ),
            shape_fn=infer_transpose,
        )
    )
    register_opdef(
        OpDef(
            name="slice",
            category=_CATEGORY,
            description="Extract a sub-tensor per numpy slice semantics.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="start_indices",
                    type=ATTR_INTS,
                    description="Per-dim start index (numpy semantics).",
                ),
                AttrSpec(
                    name="limit_indices",
                    type=ATTR_INTS,
                    description="Per-dim limit index (exclusive).",
                ),
                AttrSpec(
                    name="strides",
                    type=ATTR_INTS,
                    default=None,
                    description="Per-dim stride (None = all ones).",
                ),
            ),
            shape_fn=infer_slice,
        )
    )
    register_opdef(
        OpDef(
            name="gather",
            category=_CATEGORY,
            description="Gather slices of the tensor at index positions.",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axes",
                    type=ATTR_INTS,
                    default=(0,),
                    description="Tensor axes gathered over.",
                ),
            ),
            shape_fn=infer_gather,
        )
    )
    register_opdef(
        OpDef(
            name="scatter",
            category=_CATEGORY,
            description="Scatter updates into a tensor at index positions "
            "(functional: returns a new tensor).",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="mode",
                    type=ATTR_STR,
                    default="update",
                    description="Combining mode: 'update' | 'add' | 'max' | 'min'.",
                ),
                AttrSpec(
                    name="axes",
                    type=ATTR_INTS,
                    default=(0,),
                    description="Tensor axes scattered over.",
                ),
            ),
            shape_fn=infer_scatter,
        )
    )
    register_opdef(
        OpDef(
            name="concatenate",
            category=_CATEGORY,
            description="Concatenate two or more tensors along one axis.",
            arity=(1, None),
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    description="Axis along which to concatenate.",
                ),
            ),
            shape_fn=infer_concatenate,
        )
    )
    register_opdef(
        OpDef(
            name="pad",
            category=_CATEGORY,
            description="Pad the operand per the padding configuration.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="padding_config",
                    type=ATTR_NESTED_INTS,
                    description="((lo_0, hi_0), ..., (lo_n, hi_n)) per-dim padding.",
                ),
                AttrSpec(
                    name="value",
                    type=ATTR_FLOAT,
                    default=0.0,
                    description="Fill value for padded positions.",
                ),
            ),
            shape_fn=infer_pad,
        )
    )


_register_structure()
