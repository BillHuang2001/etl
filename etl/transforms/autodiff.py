"""Shared automatic-differentiation machinery.

Holds the JVP/VJP rule registries, the `ZeroTangent` sentinel, and the
forward/backward graph sweeps that `jvp`, `vjp`, and `grad` build upon.
All AD transforms are graph→graph: the result graphs contain only ordinary
`etl.ops` ops (backends need no autodiff runtime support). Rule signatures are
binding — see `./CONTEXT.md` "Rule-call signatures" and "AD semantics".
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from etl import ir
from etl.core import TransformError
from etl.trace import Graph

# Binding rule signatures (see ./CONTEXT.md):
#   jvp: rule(op, tangents) -> output_tangents
#     tangents: tuple of ir.Value | None | ZeroTangent aligned with
#     `op.operands` (None == ZeroTangent == zero tangent); primal values are
#     available as `op.operands` / `op.results`. Returns a tuple aligned with
#     `op.results`.
#   vjp: rule(op, cotangents, primals) -> input_cotangents
#     cotangents: tuple of ir.Value | None | ZeroTangent aligned with
#     `op.results`; primals: tuple of the op's operand values. Returns a tuple
#     of ir.Value | None | ZeroTangent aligned with `op.operands` (None /
#     ZeroTangent = zero gradient).
JVPRule = Callable[[ir.Op, Tuple[Optional[ir.Value], ...]], Tuple[ir.Value, ...]]
VJPRule = Callable[
    [ir.Op, Tuple[Optional[ir.Value], ...], Tuple[ir.Value, ...]],
    Tuple[Optional[ir.Value], ...],
]

#: Op-def-name → rule. Custom blocks register under `block:<block_name>` (done
#: by `BlockOp.jvp_rule/vjp_rule(fn)` in `etl/block`). Builtin rules are
#: registered by `rules.py` at import time.
jvp_rules: Dict[str, JVPRule] = {}
vjp_rules: Dict[str, VJPRule] = {}


class ZeroTangent:
    """Sentinel for an all-zero tangent/cotangent that need not materialize.

    Rules return this for undifferentiated values (e.g. `stop_gradient`,
    boolean/int-producing ops). The machinery materializes a zeros op only
    when a real tensor is required (e.g. cotangent accumulation), keeping
    transformed graphs free of dead zero ops.
    """

    __slots__ = ()
    _instance: Optional["ZeroTangent"] = None

    def __new__(cls) -> "ZeroTangent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ZeroTangent"


def register_jvp_rule(op_name: str, fn: JVPRule) -> None:
    """Register (or replace) the JVP rule for `op_name`.

    Custom blocks register under the `block:<block_name>` namespace; that is
    exactly what `BlockOp.jvp_rule(fn)` does in `etl/block`.
    """
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    jvp_rules[op_name] = fn


def register_vjp_rule(op_name: str, fn: VJPRule) -> None:
    """Register (or replace) the VJP rule for `op_name`.

    Custom blocks register under the `block:<block_name>` namespace; that is
    exactly what `BlockOp.vjp_rule(fn)` does in `etl/block`.
    """
    if not isinstance(op_name, str) or not op_name:
        raise ValueError("op_name must be a non-empty string")
    vjp_rules[op_name] = fn


def get_jvp_rule(op_name: str) -> Optional[JVPRule]:
    """The registered JVP rule for `op_name`, or `None` (never raises)."""
    return jvp_rules.get(op_name)


def get_vjp_rule(op_name: str) -> Optional[VJPRule]:
    """The registered VJP rule for `op_name`, or `None` (never raises)."""
    return vjp_rules.get(op_name)


def require_jvp_rule(op_name: str) -> JVPRule:
    """Like `get_jvp_rule`, but raises `TransformError` naming the op."""
    rule = jvp_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"jvp: no JVP rule for op '{op_name}'. Register one with "
            f"register_jvp_rule('{op_name}', fn) (custom blocks: "
            f"BlockOp.jvp_rule); there is no silent fallback."
        )
    return rule


def require_vjp_rule(op_name: str) -> VJPRule:
    """Like `get_vjp_rule`, but raises `TransformError` naming the op."""
    rule = vjp_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"grad/vjp: no VJP rule for op '{op_name}'. Register one with "
            f"register_vjp_rule('{op_name}', fn) (custom blocks: "
            f"BlockOp.vjp_rule); there is no silent fallback."
        )
    return rule


def forward_sweep(graph: Graph, tangent_specs) -> Graph:
    """Core forward-mode sweep (used by `jvp`).

    Walks the graph topologically, applies each op's JVP rule to the tangent
    values (seeded from `tangent_specs`), and returns a new graph whose inputs
    are primal inputs + tangent inputs and whose outputs are (primal_outputs,
    tangent_outputs) (stub).
    """
    raise NotImplementedError(
        "forward_sweep: implementation phase; see etl/transforms/CONTEXT.md"
    )


def reverse_sweep(graph: Graph, cotangent_specs) -> Graph:
    """Core backward sweep (used by `vjp` and `grad`).

    One reverse topological pass applying each op's VJP rule, seeded from
    `cotangent_specs`; cotangents of multiply-used values accumulate via
    `add`; `ZeroTangent` entries materialize only when needed. Returns a new
    graph whose inputs are primal inputs + cotangent inputs and whose outputs
    are (primal_outputs, input_cotangents) (stub).
    """
    raise NotImplementedError(
        "reverse_sweep: implementation phase; see etl/transforms/CONTEXT.md"
    )
