"""Collective op defs: explicit communication ops + rank/world_size scalars.

All communication collectives carry effect ``collective`` (functional results,
but device-synchronization side effects; see CONTEXT.md effect model).
``rank``/``world_size`` observe process configuration and carry effect
``read``. Every collective names its ``group``; ops whose result shape depends
on it also record ``group_size`` (needed for static shape inference).

Note: the IR name is ``broadcast_collective`` — the frontend shape op
``broadcast`` already owns that name.
"""

from __future__ import annotations

from ..effects import EFFECT_COLLECTIVE, EFFECT_READ
from ..inference import (
    infer_all_gather,
    infer_all_to_all,
    infer_identity,
    infer_reduce_scatter,
    infer_scalar_int64,
)
from . import (
    ATTR_INT,
    ATTR_NESTED_INTS,
    ATTR_STR,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "collective"


def _register_collectives() -> None:
    register_opdef(
        OpDef(
            name="all_reduce",
            category=_CATEGORY,
            description="Reduce the tensor across the group and broadcast the "
            "result to every rank (shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(
                    name="reduce_op",
                    type=ATTR_STR,
                    description="Reduction kind: 'sum' | 'max' | 'min' | 'prod'.",
                ),
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="all_gather",
            category=_CATEGORY,
            description="Gather slices of every rank's tensor along an axis "
            "(axis dim grows by group_size).",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=0,
                    description="Axis along which to gather.",
                ),
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                AttrSpec(
                    name="group_size",
                    type=ATTR_INT,
                    description="Number of ranks in the group (for shape inference).",
                ),
            ),
            shape_fn=infer_all_gather,
        )
    )
    register_opdef(
        OpDef(
            name="reduce_scatter",
            category=_CATEGORY,
            description="Reduce the tensor across the group and scatter slices "
            "to each rank along an axis (axis dim shrinks by group_size).",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(
                    name="reduce_op",
                    type=ATTR_STR,
                    description="Reduction kind: 'sum' | 'max' | 'min' | 'prod'.",
                ),
                AttrSpec(
                    name="axis",
                    type=ATTR_INT,
                    default=0,
                    description="Axis along which to scatter.",
                ),
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                AttrSpec(
                    name="group_size",
                    type=ATTR_INT,
                    description="Number of ranks in the group (for shape inference).",
                ),
            ),
            shape_fn=infer_reduce_scatter,
        )
    )
    register_opdef(
        OpDef(
            name="all_to_all",
            category=_CATEGORY,
            description="Exchange data between every pair of ranks: split along "
            "split_axis, concat along concat_axis.",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(
                    name="split_axis",
                    type=ATTR_INT,
                    description="Axis along which each rank splits its tensor.",
                ),
                AttrSpec(
                    name="concat_axis",
                    type=ATTR_INT,
                    description="Axis along which each rank concatenates received "
                    "slices.",
                ),
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                AttrSpec(
                    name="group_size",
                    type=ATTR_INT,
                    description="Number of ranks in the group (for shape inference).",
                ),
            ),
            shape_fn=infer_all_to_all,
        )
    )
    register_opdef(
        OpDef(
            name="broadcast_collective",
            category=_CATEGORY,
            description="Broadcast one rank's tensor to the whole group "
            "(shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                AttrSpec(
                    name="group_size",
                    type=ATTR_INT,
                    description="Number of ranks in the group.",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="collective_permute",
            category=_CATEGORY,
            description="Send slices to specified ranks and receive from others "
            "(shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_COLLECTIVE,
            attributes=(
                AttrSpec(
                    name="source_target_pairs",
                    type=ATTR_NESTED_INTS,
                    description="((source, target), ...) rank pairs.",
                ),
                AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                AttrSpec(
                    name="group_size",
                    type=ATTR_INT,
                    description="Number of ranks in the group.",
                ),
            ),
            shape_fn=infer_identity,
        )
    )
    for name, description in (
        ("rank", "Scalar int64: this process's rank in the group."),
        ("world_size", "Scalar int64: number of ranks in the group."),
    ):
        register_opdef(
            OpDef(
                name=name,
                category=_CATEGORY,
                description=description,
                arity=0,
                result_count=1,
                effect=EFFECT_READ,
                attributes=(
                    AttrSpec(name="group", type=ATTR_STR, description="Group name."),
                ),
                shape_fn=infer_scalar_int64,
            )
        )


_register_collectives()
