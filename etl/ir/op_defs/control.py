"""Control op defs: constant, stop_gradient, if, while, call, runtime_call,
block_call, and the ``return`` terminator.

Effects: ``constant``/``stop_gradient``/``if``/``while``/``call``/``return``
are ``pure`` at the IR level (see CONTEXT.md for the ``call`` caveat);
``runtime_call`` is ``callback``; ``block_call`` defaults to ``read`` (the
declared block's actual effects are consulted by verification/backends).
"""

from __future__ import annotations

from ..effects import EFFECT_CALLBACK, EFFECT_PURE, EFFECT_READ
from ..inference import infer_identity
from . import (
    ATTR_ANY,
    ATTR_NDARRAY,
    ATTR_STR,
    AttrSpec,
    OpDef,
    register_opdef,
)

_CATEGORY = "control"


def _register_control() -> None:
    register_opdef(
        OpDef(
            name="constant",
            category=_CATEGORY,
            description="Embed a concrete tensor payload (only legal source of "
            "captured tensor data; large payloads warn at trace level).",
            arity=0,
            result_count=1,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="value",
                    type=ATTR_NDARRAY,
                    description="The numpy array payload; result type derives "
                    "from its dtype/shape.",
                ),
            ),
            # shape_fn=None: the Builder derives the result type from the payload.
        )
    )
    register_opdef(
        OpDef(
            name="stop_gradient",
            category=_CATEGORY,
            description="Identity that stops derivative propagation "
            "(shape-preserving).",
            arity=1,
            result_count=1,
            effect=EFFECT_PURE,
            shape_fn=infer_identity,
        )
    )
    register_opdef(
        OpDef(
            name="if",
            category=_CATEGORY,
            description="Conditional: operand 0 is the boolean predicate; regions "
            "(true, false) are bound to the other operands via entry-block args; "
            "branch terminators return values of matching types.",
            arity=(1, None),
            result_count=None,
            effect=EFFECT_PURE,
            regions=2,
            # shape_fn=None: result types come from the region terminators or
            # explicit result_types.
        )
    )
    register_opdef(
        OpDef(
            name="while",
            category=_CATEGORY,
            description="Loop: operands are the loop-carried initial values; "
            "regions (cond, body) are bound to them via entry-block args; "
            "results are the final carried values (types = operand types).",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_PURE,
            shape_fn=infer_identity,
            regions=2,
        )
    )
    register_opdef(
        OpDef(
            name="call",
            category=_CATEGORY,
            description="Call another function of the module (callee by name). "
            "Not emitted by v1 single-function graphs; see CONTEXT.md.",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_PURE,
            attributes=(
                AttrSpec(
                    name="callee",
                    type=ATTR_STR,
                    description="Name of the target function in the module.",
                ),
            ),
            # shape_fn=None: result types come from the callee signature.
        )
    )
    register_opdef(
        OpDef(
            name="runtime_call",
            category=_CATEGORY,
            description="Invoke a Python callback at runtime (explicit escape "
            "hatch; effect 'callback': never reordered/duplicated/eliminated).",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_CALLBACK,
            attributes=(
                AttrSpec(
                    name="callback",
                    type=ATTR_STR,
                    description="Registered callback identifier.",
                ),
                AttrSpec(
                    name="result_specs",
                    type=ATTR_ANY,
                    description="Declared result ValueTypes (serialized "
                    "structurally).",
                ),
            ),
        )
    )
    register_opdef(
        OpDef(
            name="block_call",
            category=_CATEGORY,
            description="Invoke a custom block (etl.block); static_args "
            "specialize the op. Actual effects come from the block declaration.",
            arity=(0, None),
            result_count=None,
            effect=EFFECT_READ,
            attributes=(
                AttrSpec(
                    name="block_name",
                    type=ATTR_STR,
                    description="Registered block name.",
                ),
                AttrSpec(
                    name="static_args",
                    type=ATTR_ANY,
                    default=(),
                    description="JSON-able static arguments (specialize the op).",
                ),
            ),
        )
    )
    register_opdef(
        OpDef(
            name="return",
            category="terminator",
            description="Region terminator: yields the region's result values. "
            "Added to the canonical set because verification requires a "
            "terminator; the frontend never names it directly.",
            arity=(0, None),
            result_count=0,
            effect=EFFECT_PURE,
            is_terminator=True,
        )
    )


_register_control()
