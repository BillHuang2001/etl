"""Elementwise op defs: pointwise numerics, logic, bitwise, cast, comparisons.

All elementwise ops are ``pure``. Binary ops broadcast their operands
(``infer_elementwise_binary``); unary ops preserve shape/dtype
(``infer_elementwise_unary``); comparisons produce bool
(``infer_compare``); ``cast`` changes only the dtype (``infer_cast``).
"""

from __future__ import annotations

from ..effects import EFFECT_PURE
from ..inference import (
    infer_abs,
    infer_cast,
    infer_compare,
    infer_elementwise_binary,
    infer_elementwise_unary,
)
from . import ATTR_DTYPE, AttrSpec, OpDef, register_opdef

_CATEGORY = "elementwise"

_BINARY_OPS = {
    "add": "Elementwise addition (broadcasting).",
    "subtract": "Elementwise subtraction (broadcasting).",
    "multiply": "Elementwise multiplication (broadcasting).",
    "divide": "Elementwise division (broadcasting).",
    "power": "Elementwise exponentiation (broadcasting).",
    "remainder": "Elementwise remainder (numpy semantics; broadcasting).",
    "maximum": "Elementwise maximum (broadcasting).",
    "minimum": "Elementwise minimum (broadcasting).",
    "logical_and": "Elementwise boolean AND (broadcasting).",
    "logical_or": "Elementwise boolean OR (broadcasting).",
    "bitwise_and": "Elementwise bitwise AND (broadcasting).",
    "bitwise_or": "Elementwise bitwise OR (broadcasting).",
    "bitwise_xor": "Elementwise bitwise XOR (broadcasting).",
}

_UNARY_OPS = {
    "negate": "Elementwise negation.",
    "square": "Elementwise square.",
    "sqrt": "Elementwise square root.",
    "exp": "Elementwise natural exponential.",
    "log": "Elementwise natural logarithm.",
    "log1p": "Elementwise log(1 + x) (stable for small x).",
    "sin": "Elementwise sine.",
    "cos": "Elementwise cosine.",
    "tan": "Elementwise tangent.",
    "acos": "Elementwise arccosine.",
    "tanh": "Elementwise hyperbolic tangent.",
    "floor": "Elementwise floor (numpy semantics; int input unchanged).",
    "ceil": "Elementwise ceiling (numpy semantics; int input unchanged).",
    "round": "Elementwise round-half-to-even (numpy semantics; int input unchanged).",
    "sigmoid": "Elementwise logistic sigmoid 1/(1 + exp(-x)).",
    "relu": "Elementwise rectified linear unit max(x, 0).",
    "gelu": "Elementwise Gaussian error linear unit.",
    "erf": "Elementwise Gauss error function.",
    "sign": "Elementwise sign function.",
    "logical_not": "Elementwise boolean NOT.",
}

_COMPARISON_OPS = {
    "equal": "Elementwise equality test; result dtype bool.",
    "not_equal": "Elementwise inequality test; result dtype bool.",
    "less": "Elementwise less-than; result dtype bool.",
    "less_equal": "Elementwise less-or-equal; result dtype bool.",
    "greater": "Elementwise greater-than; result dtype bool.",
    "greater_equal": "Elementwise greater-or-equal; result dtype bool.",
}


def _register_binary() -> None:
    for name, description in _BINARY_OPS.items():
        register_opdef(
            OpDef(
                name=name,
                category=_CATEGORY,
                description=description,
                arity=2,
                result_count=1,
                effect=EFFECT_PURE,
                shape_fn=infer_elementwise_binary,
            )
        )


def _register_unary() -> None:
    for name, description in _UNARY_OPS.items():
        register_opdef(
            OpDef(
                name=name,
                category=_CATEGORY,
                description=description,
                arity=1,
                result_count=1,
                effect=EFFECT_PURE,
                shape_fn=infer_elementwise_unary,
            )
        )


def _register_abs() -> None:
    register_opdef(
        OpDef(
            name="abs",
            category=_CATEGORY,
            description="Elementwise absolute value (numpy magnitude semantics).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_abs,
        )
    )


def _register_cast() -> None:
    register_opdef(
        OpDef(
            name="cast",
            category=_CATEGORY,
            description="Reinterpret elementwise as another dtype; shape preserved.",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="dtype",
                    type=ATTR_DTYPE,
                    description="Target dtype (numpy dtype name).",
                ),
            ),
            shape_fn=infer_cast,
        )
    )


def _register_comparisons() -> None:
    for name, description in _COMPARISON_OPS.items():
        register_opdef(
            OpDef(
                name=name,
                category="comparison",
                description=description,
                arity=2,
                result_count=1,
                effect=EFFECT_PURE,
                shape_fn=infer_compare,
            )
        )


_register_binary()
_register_unary()
_register_abs()
_register_cast()
_register_comparisons()
