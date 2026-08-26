"""EvoXIR op definitions for the frontend dialect (declarative data).

Ownership decision: the generic SSA machinery and the op REGISTRY live in
``etl.ir``; the frontend dialect — WHICH ops exist and what they declare —
is defined here in ``etl.ops``, because ``ir`` must not import ``ops``
(layering: ``core ← ir ← ops``). At implementation time, ``register_all()``
registers every entry below into ``ir``'s registry (via the registration API
documented in ``etl/ir/CONTEXT.md``) so ``Builder.create(name, ...)`` can
validate operands/attrs/effects.

Effect policy (binding): all frontend ops are ``pure`` except
``runtime_call`` (``callback``). EvoXIR ops are functional SSA dataflow —
``scatter``/``constant``/``pad`` produce NEW values and mutate nothing, so
they carry no ``write`` effect. The ``write``/``read``/``collective`` effect
kinds are reserved for other layers (e.g. ``etl.dist`` collectives).

Each entry mirrors a public op function's signature exactly; keep them in
sync (test: ``tests/ops/test_opdefs.py`` asserts 1:1 coverage with
``etl.ops.__all__``).
"""
from __future__ import annotations

import dataclasses
from typing import Tuple

from etl import ir

__all__ = ["OpDef", "OPDEFS", "EFFECT_PURE", "EFFECT_CALLBACK", "register_all"]

#: Effect kinds — must match the kind names declared in ``etl.ir``.
EFFECT_PURE = "pure"
EFFECT_CALLBACK = "callback"


@dataclasses.dataclass(frozen=True)
class OpDef:
    """Declarative definition of one frontend op.

    Attributes:
        name: IR op name (equals the public function name).
        operands: Named operand slots (variadic ops use the special name
            ``"*operands"``).
        results: Named result slots (``("out",)`` unless a more specific name
            is meaningful).
        attrs: Attribute names — one per static function parameter (e.g.
            ``cast``: ``("dtype",)``).
        effects: Effect kinds attached to the op (``pure`` / ``callback``).
    """

    name: str
    operands: Tuple[str, ...]
    results: Tuple[str, ...]
    attrs: Tuple[str, ...]
    effects: Tuple[str, ...]


def _op(name, operands, attrs=(), effects=(EFFECT_PURE,),
        results=("out",)) -> OpDef:
    """Compact table constructor."""
    return OpDef(name, tuple(operands), tuple(results), tuple(attrs),
                 tuple(effects))


# fmt: off
OPDEFS: Tuple[OpDef, ...] = (
    # --- elementwise: binary (x, y) -------------------------------------
    _op("add",        ("x", "y")),
    _op("subtract",   ("x", "y")),
    _op("multiply",   ("x", "y")),
    _op("divide",     ("x", "y")),
    _op("power",      ("x", "y")),
    _op("remainder",  ("x", "y")),
    _op("maximum",    ("x", "y")),
    _op("minimum",    ("x", "y")),
    _op("bitwise_and",  ("x", "y")),
    _op("bitwise_or",   ("x", "y")),
    _op("bitwise_xor",  ("x", "y")),
    # --- elementwise: unary (x) -----------------------------------------
    _op("abs",     ("x",)),
    _op("negate",  ("x",)),
    _op("square",  ("x",)),
    _op("sqrt",    ("x",)),
    _op("exp",     ("x",)),
    _op("log",     ("x",)),
    _op("log1p",   ("x",)),
    _op("sin",     ("x",)),
    _op("cos",     ("x",)),
    _op("tan",     ("x",)),
    _op("tanh",    ("x",)),
    _op("sigmoid", ("x",)),
    _op("relu",    ("x",)),
    _op("gelu",    ("x",)),
    _op("erf",     ("x",)),
    _op("sign",    ("x",)),
    # --- elementwise: cast ----------------------------------------------
    _op("cast", ("x",), attrs=("dtype",)),
    # --- comparison / logical / selection --------------------------------
    _op("equal",        ("x", "y")),
    _op("not_equal",    ("x", "y")),
    _op("less",         ("x", "y")),
    _op("less_equal",   ("x", "y")),
    _op("greater",      ("x", "y")),
    _op("greater_equal", ("x", "y")),
    _op("logical_and",  ("x", "y")),
    _op("logical_or",   ("x", "y")),
    _op("logical_not",  ("x",)),
    _op("select",       ("pred", "on_true", "on_false")),
    # --- indexing / shape manipulation ------------------------------------
    _op("broadcast",  ("x",), attrs=("shape",)),
    _op("reshape",    ("x",), attrs=("shape",)),
    _op("transpose",  ("x",), attrs=("axes",)),
    _op("slice",      ("x",), attrs=("start", "lengths", "strides")),
    _op("gather",     ("x", "indices"), attrs=("axis",)),
    _op("scatter",    ("x", "indices", "updates"), attrs=("axis",)),
    _op("concatenate", ("*operands",), attrs=("axis",)),
    _op("pad",        ("x",), attrs=("config", "value")),
    # --- reductions -------------------------------------------------------
    _op("reduce_sum",  ("x",), attrs=("axes", "keepdims")),
    _op("reduce_max",  ("x",), attrs=("axes", "keepdims")),
    _op("reduce_min",  ("x",), attrs=("axes", "keepdims")),
    _op("reduce_mean", ("x",), attrs=("axes", "keepdims")),
    _op("reduce_prod", ("x",), attrs=("axes", "keepdims")),
    _op("argmax",      ("x",), attrs=("axis", "keepdims"), results=("indices",)),
    _op("argmin",      ("x",), attrs=("axis", "keepdims"), results=("indices",)),
    # --- linalg ------------------------------------------------------------
    _op("dot",    ("a", "b"), results=("dot",)),
    _op("conv",   ("x", "w"), attrs=("strides", "padding", "input_dilation",
                                      "kernel_dilation", "feature_group_size",
                                      "channels_last")),
    _op("tril",   ("x",), attrs=("k",)),
    _op("triu",   ("x",), attrs=("k",)),
    _op("cumsum", ("x",), attrs=("axis", "reverse")),
    _op("solve",  ("a", "b")),
    # --- constants / escape hatches -----------------------------------------
    _op("constant",   (), attrs=("value",)),
    _op("runtime_call", ("*operands",), attrs=("callback", "result"),
        effects=(EFFECT_CALLBACK,)),
    _op("stop_gradient", ("x",)),
)
# fmt: on


def register_all() -> None:
    """Register every :data:`OPDEFS` entry into ``etl.ir``'s op registry.

    Called once at ``etl.ops`` import time during the implementation phase
    (once ``etl.ir`` provides its registration API — see
    ``etl/ir/CONTEXT.md``). Registration must be idempotent (safe to call
    again in tests).
    """
    raise NotImplementedError
