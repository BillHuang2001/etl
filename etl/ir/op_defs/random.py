"""Random op defs: key-based functional RNG (multi-algorithm, v1).

All random ops are ``pure`` and deterministic functions of their key operand:
the same key + same operands yield bit-identical values, repeatable across
runs. Each op carries an ``algorithm`` attribute selecting the key
representation and stream derivation — see ``ALGORITHMS`` below (the SINGLE
source of truth imported by ``etl/ops/random.py`` and the backend writers;
never duplicate these names elsewhere). Stream derivation (SplitMix64 with
per-op salts) is defined in ``etl/ops/random.py`` and the numpy kernels;
backends never re-derive it. 5 of the 6 ops (all but ``random_multinomial``)
export as v1 StableHLO — inline SplitMix64 expansion in
``etl/backends/stablehlo/random_export.py``; ``random_multinomial`` and any
unimplemented algorithm defer with an explicit ``BackendError``. The numpy
interpreter is the reference implementation.
"""
from __future__ import annotations

import numpy as np

from etl.core import int32, int64

from ..effects import EFFECT_PURE
from ..inference import (
    infer_random_key_mix,
    infer_random_multinomial,
    infer_random_normal,
    infer_random_permutation,
    infer_random_randint,
    infer_random_uniform,
)
from . import (
    ATTR_DTYPE,
    ATTR_INT,
    ATTR_SHAPE,
    ATTR_STR,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "random"

# ---------------------------------------------------------------------------
# Canonical algorithm names (binding — the SINGLE source of truth)
# ---------------------------------------------------------------------------

#: Canonical RNG algorithm names, in canonical order. splitmix64 (the v1
#: default) uses a rank-0 int64 64-bit state; threefry2x32 / philox4x32_10
#: use 2/4-word int32 counter-based states. `ops` and the backend writers
#: import these constants rather than duplicating the literals.
ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")

#: Default algorithm — the v1 behavior (rank-0 int64 SplitMix64 key).
DEFAULT_ALGORITHM = ALGORITHMS[0]

#: Key type (static shape, dtype) per canonical algorithm.
_ALGORITHM_KEY_TYPES = {
    "splitmix64": ((), int64),
    "threefry2x32": ((2,), int32),
    "philox4x32_10": ((4,), int32),
}


def validate_algorithm(name: str) -> str:
    """Validate an RNG algorithm name against the canonical set.

    Accepts the canonical strings (plain strings only) and returns the name
    unchanged.

    Raises:
        TypeError: If ``name`` is not a plain string.
        ValueError: If ``name`` is not one of the canonical algorithm names
            (the message lists the accepted names).
    """
    if not isinstance(name, str):
        raise TypeError(
            f"random algorithm must be a string, got "
            f"{type(name).__name__} ({name!r}); expected one of {ALGORITHMS}"
        )
    if name not in ALGORITHMS:
        raise ValueError(
            f"unknown random algorithm {name!r}; expected one of "
            f"{', '.join(ALGORITHMS)}"
        )
    return name


def algorithm_key_type(name: str) -> tuple[tuple[int, ...], np.dtype]:
    """The key type ``(shape, dtype)`` for a canonical algorithm name.

    splitmix64 → ((), int64); threefry2x32 → ((2,), int32);
    philox4x32_10 → ((4,), int32). Validates the name first.
    """
    return _ALGORITHM_KEY_TYPES[validate_algorithm(name)]


#: Shared ``algorithm`` attribute for every random op (default splitmix64).
_ALGORITHM_ATTR = AttrSpec(
    name="algorithm",
    type=ATTR_STR,
    default=DEFAULT_ALGORITHM,
    description="Canonical RNG algorithm selecting the key representation "
    "and stream derivation: splitmix64 (default), threefry2x32, or "
    "philox4x32_10.",
)


def _register_random() -> None:
    register_opdef(
        OpDef(
            name="random_key_mix",
            category=_CATEGORY,
            description="Deterministic key derivation (SplitMix64 mix of "
            "key ^ salt): the internal building block behind "
            "etl.random.split / split_n — one key in, one derived key out.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                _ALGORITHM_ATTR,
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
                _ALGORITHM_ATTR,
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
                _ALGORITHM_ATTR,
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
                _ALGORITHM_ATTR,
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
                _ALGORITHM_ATTR,
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
                _ALGORITHM_ATTR,
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
