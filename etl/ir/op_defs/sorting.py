"""Sorting op defs: ``sort`` and ``argsort`` — all ``pure``.

The numpy interpreter backend is the reference implementation; compiler
backends defer with an explicit ``BackendError`` naming the op (no stablehlo
writer mapping — see ``etl/backends/CONTEXT.md`` Known Issues). Higher-level
compositions built on these (``etl.topk``) live in ``etl/ops/sorting.py`` and
introduce no IR ops of their own.
"""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import infer_argsort, infer_identity
from . import ATTR_BOOL, ATTR_INT, AttrSpec, OpDef, register_opdef

_CATEGORY = "sorting"


def _register_sorting() -> None:
    register_opdef(
        OpDef(
            name="sort",
            category=_CATEGORY,
            description="Sort the operand along an axis (numpy sort semantics; "
            "dtype preserved). descending flips the sorted result along the "
            "axis (composition semantics, implemented in the kernel); stable "
            "selects numpy's stable sort kind.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=-1,
                    description="Axis to sort along (numpy convention; -1 = "
                    "last axis).",
                ),
                AttrSpec(
                    name="descending",
                    type=ATTR_BOOL,
                    default=False,
                    description="Sort in descending order (implemented as a "
                    "flip of the ascending result along the axis).",
                ),
                AttrSpec(
                    name="stable",
                    type=ATTR_BOOL,
                    default=False,
                    description="Use numpy's stable sort kind.",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="argsort",
            category=_CATEGORY,
            description="Indices that would sort the operand along an axis "
            "(numpy argsort semantics); result dtype int64.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=-1,
                    description="Axis to sort along (numpy convention; -1 = "
                    "last axis).",
                ),
                AttrSpec(
                    name="descending",
                    type=ATTR_BOOL,
                    default=False,
                    description="Sort in descending order (implemented as a "
                    "flip of the ascending indices along the axis).",
                ),
                AttrSpec(
                    name="stable",
                    type=ATTR_BOOL,
                    default=False,
                    description="Use numpy's stable sort kind.",
                ),
            ),
            shape_fn=infer_argsort,
        )
    )


_register_sorting()
