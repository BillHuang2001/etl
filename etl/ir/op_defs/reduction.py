"""Reduction op defs: reduce_* family, argmax/argmin, cumsum — all ``pure``."""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import infer_arg_reduction, infer_identity, infer_reduction
from . import ATTR_BOOL, ATTR_INT, ATTR_INTS, ATTR_STR, AttrSpec, OpDef, register_opdef

_CATEGORY = "reduction"

_REDUCE_OPS = {
    "reduce_sum": "Reduce over axes by summation.",
    "reduce_max": "Reduce over axes by maximum.",
    "reduce_min": "Reduce over axes by minimum.",
    "reduce_mean": "Reduce over axes by mean (promotes to float).",
    "reduce_prod": "Reduce over axes by product.",
}

_REDUCE_ATTRS = (
    AttrSpec(
        name="axes",
        type=ATTR_INTS,
        description="Axes to reduce over (empty = all axes).",
    ),
    AttrSpec(
        name="keepdims",
        type=ATTR_BOOL,
        default=False,
        description="Keep reduced axes as size-1 dims.",
    ),
    AttrSpec(
        name="reduce_op",
        type=ATTR_STR,
        description="Reduction kind: 'sum' | 'max' | 'min' | 'mean' | 'prod' "
        "(drives the result-dtype rule in infer_reduction).",
    ),
)


def _register_reductions() -> None:
    for name, description in _REDUCE_OPS.items():
        register_opdef(
            OpDef(
                name=name,
                category=_CATEGORY,
                description=description,
                arity=1,
                result_count=1,
                effect=EFFECT_PURE,
                attributes=_REDUCE_ATTRS,
                shape_fn=infer_reduction,
            )
        )
    for name in ("argmax", "argmin"):
        register_opdef(
            OpDef(
                name=name,
                category=_CATEGORY,
                description=(
                    "Index of the maximum value over the axis."
                    if name == "argmax"
                    else "Index of the minimum value over the axis."
                ),
                arity=1,
                result_count=1,
                effect=EFFECT_PURE,
                attributes=(
                    AttrSpec(
                        name="axis",
                        type=ATTR_INT,
                        default=None,
                        description="Axis to reduce (None = flatten, scalar result).",
                    ),
                    AttrSpec(
                        name="keepdims",
                        type=ATTR_BOOL,
                        default=False,
                        description="Keep the reduced axis as a size-1 dim.",
                    ),
                ),
                shape_fn=infer_arg_reduction,
            )
        )
    register_opdef(
        OpDef(
            name="cumsum",
            category=_CATEGORY,
            description="Cumulative sum along an axis (shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    description="Axis along which to accumulate.",
                ),
                AttrSpec(
                    name="reverse",
                    type=ATTR_BOOL,
                    default=False,
                    description="Accumulate in reverse order.",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="cumprod",
            category=_CATEGORY,
            description="Cumulative product along an axis (shape-preserving; "
            "bool operands are cast to int64 frontend-side, mirroring "
            "cumsum).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    description="Axis along which to accumulate.",
                ),
                AttrSpec(
                    name="reverse",
                    type=ATTR_BOOL,
                    default=False,
                    description="Accumulate in reverse order.",
                ),
            ),
            shape_fn=infer_identity,
        )
    )


_register_reductions()
