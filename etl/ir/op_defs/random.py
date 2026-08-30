"""Random op defs: key-based deterministic RNG (numpy-backend-only in v1).

All random ops are ``pure`` and deterministic functions of their key operand
(a rank-0 int64 tensor): the same key + same operands yield bit-identical
values, repeatable across runs. Stream derivation (SplitMix64 with per-op
salts) is defined in ``etl/ops/random.py`` and the numpy kernels; backends
never re-derive it. Compiler backends defer with an explicit ``BackendError``
(no stablehlo writer — see ``etl/backends/CONTEXT.md`` Known Issues): the
numpy interpreter is the reference implementation.
"""
from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import (
    infer_random_key_mix,
    infer_random_multinomial,
    infer_random_normal,
    infer_random_permutation,
    infer_random_randint,
    infer_random_uniform,
)
from . import ATTR_DTYPE, ATTR_INT, ATTR_SHAPE, AttrSpec, OpDef, register_opdef

_CATEGORY = "random"


def _register_random() -> None:
    register_opdef(
        OpDef(
            name="random_key_mix",
            category=_CATEGORY,
            description="Deterministic 64-bit key derivation (SplitMix64 mix of "
            "key ^ salt): the internal building block behind "
            "etl.random.split / split_n — one key in, one derived key out.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="salt",
                    type=ATTR_INT,
                    description="64-bit salt selecting the derivation "
                    "(split uses 0 and the SplitMix64 golden gamma).",
                ),
            ),
            shape_fn=infer_random_key_mix,
        )
    )
    register_opdef(
        OpDef(
            name="random_uniform",
            category=_CATEGORY,
            description="Draw uniform values in [low, high) from the key's "
            "stream; low/high broadcast against the shape attribute.",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="shape",
                    type=ATTR_SHAPE,
                    description="Requested output shape (int/Dim/DimExpr "
                    "entries; broadcast with the low/high operand shapes).",
                ),
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Output dtype (floating).",
                ),
            ),
            shape_fn=infer_random_uniform,
        )
    )
    register_opdef(
        OpDef(
            name="random_normal",
            category=_CATEGORY,
            description="Draw normal values (mean, std broadcast against the "
            "shape attribute) from the key's stream.",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="shape",
                    type=ATTR_SHAPE,
                    description="Requested output shape (int/Dim/DimExpr "
                    "entries; broadcast with the mean/std operand shapes).",
                ),
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Output dtype (floating).",
                ),
            ),
            shape_fn=infer_random_normal,
        )
    )
    register_opdef(
        OpDef(
            name="random_randint",
            category=_CATEGORY,
            description="Draw integer values in [low, high) (high exclusive) "
            "from the key's stream; low/high broadcast against shape.",
            arity=3,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="shape",
                    type=ATTR_SHAPE,
                    description="Requested output shape (int/Dim/DimExpr "
                    "entries; broadcast with the low/high operand shapes).",
                ),
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Output dtype (integer).",
                ),
            ),
            shape_fn=infer_random_randint,
        )
    )
    register_opdef(
        OpDef(
            name="random_permutation",
            category=_CATEGORY,
            description="Draw a uniformly random permutation of 0..n-1 from "
            "the key's stream; n is a runtime rank-0 int64 operand (static "
            "ints are promoted to constants at trace time).",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="n",
                    type=ATTR_INT,
                    default=None,
                    description="Static population size when n was a Python "
                    "int at trace time (None when n is a runtime rank-0 "
                    "operand → runtime-dynamic result length).",
                ),
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Output dtype (integer).",
                ),
            ),
            shape_fn=infer_random_permutation,
        )
    )
    register_opdef(
        OpDef(
            name="random_multinomial",
            category=_CATEGORY,
            description="Draw num_samples indices from the 1-D probability "
            "distribution input (non-negative, sums to 1; np.random.choice "
            "semantics) from the key's stream.",
            arity=2,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="num_samples",
                    type=ATTR_INT,
                    description="Number of draws (static, non-negative).",
                ),
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Output dtype (integer; int32 by default).",
                ),
            ),
            shape_fn=infer_random_multinomial,
        )
    )


_register_random()
