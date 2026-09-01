"""Shared automatic-differentiation machinery.

Holds the JVP/VJP rule registries, the `ZeroTangent` sentinel, and the
forward/backward graph sweeps that `jvp`, `vjp`, and `grad` build upon.
All AD transforms are graph→graph: the result graphs contain only ordinary
`etl.ops` ops (backends need no autodiff runtime support). Rule signatures are
binding — see `./CONTEXT.md` "Rule-call signatures" and "AD semantics".
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from etl import core, ir, ops
from etl.core import ShapeError, TransformError
from etl.trace import Graph, StaticValue, with_builder

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
#
# Both signatures receive a PROXY `op` (see the "PROXY-OP CONVENTION" comment
# on the sweep machinery below): a fresh `ir.Op` never inserted into a block,
# carrying the original op's attributes/location but the transformed graph's
# operand/result values.
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
    """Like `get_jvp_rule`, but raises `TransformError` naming the op.

    Note: the sweep does not require a registered JVP rule when a VJP rule
    exists — it derives the JVP from the VJP rule via the adjoint
    (double-vjp) trick (see `_jvp_from_vjp`), so `register_jvp_rule` is only
    required when NEITHER rule is registered.
    """
    rule = jvp_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"jvp: no JVP rule for {_rule_display(op_name)} (and no VJP rule "
            f"to derive one from). "
            f"{_rule_hint(op_name, 'register_jvp_rule', 'jvp_rule').capitalize()}; "
            f"there is no silent fallback."
        )
    return rule


def require_jvp_or_vjp_rule(op_name: str) -> Optional[JVPRule]:
    """Like `require_jvp_rule`, but satisfied by a registered VJP rule too.

    When only a VJP rule exists, the sweep derives the JVP from it via the
    adjoint (double-vjp) trick (`_jvp_from_vjp`). Raises `TransformError`
    naming the op only when NEITHER rule is registered.
    """
    if op_name in jvp_rules or op_name in vjp_rules:
        return jvp_rules.get(op_name)
    if op_name.startswith("external:"):
        name = op_name[len("external:"):]
        hint = (
            f"Register one via the external-kernel handle returned by "
            f"etl.register_external_kernel('{name}', fn) "
            f"(.batching_rule/.vjp_rule/.jvp_rule) or register a portable "
            f"decomposition"
        )
    elif op_name.startswith("block:"):
        hint = (
            f"Register one with register_jvp_rule('{op_name}', fn) or "
            f"register_vjp_rule('{op_name}', fn) (custom blocks: "
            f"BlockOp.jvp_rule/vjp_rule)"
        )
    else:
        hint = (
            f"Register one with register_jvp_rule('{op_name}', fn) or "
            f"register_vjp_rule('{op_name}', fn)"
        )
    raise TransformError(
        f"jvp: no JVP rule for {_rule_display(op_name)} (and no VJP rule to "
        f"derive one from). {hint}; there is no silent fallback."
    )


def require_vjp_rule(op_name: str) -> VJPRule:
    """Like `get_vjp_rule`, but raises `TransformError` naming the op."""
    rule = vjp_rules.get(op_name)
    if rule is None:
        raise TransformError(
            f"grad/vjp: no VJP rule for {_rule_display(op_name)}. "
            f"{_rule_hint(op_name, 'register_vjp_rule', 'vjp_rule').capitalize()}; "
            f"there is no silent fallback."
        )
    return rule


# ---------------------------------------------------------------------------
# Sweep machinery (forward/reverse) — one combined rebuild + dual pass.
#
# PROXY-OP CONVENTION (binding for `rules.py` — it codes against this):
# a rule is invoked as `rule(proxy, ...)` where `proxy` is a fresh `ir.Op`
# dataclass instance that is NEVER inserted into any block. It carries the
# original op's `name`/`id`/`attributes` (a shallow-copied dict; leaf
# objects such as constant arrays are shared, read-only) and `location`, but
# its `operands` and `results` are the RECREATED SSA values of the
# transformed graph (same types as the originals). `op.opdef`/`op.effect`/
# `op.result` resolve normally (by name). Rules must treat the proxy as
# read-only and must NOT follow `value.owner`/`value.defining_op` off proxy
# values — those pointers still reference the ORIGINAL op.
# ---------------------------------------------------------------------------


def _rule_name(op: ir.Op) -> str:
    """Registry key for `op` (block calls are keyed ``block:<block_name>``,
    external calls keyed ``external:<name>``)."""
    if op.name == "block_call":
        return f"block:{op.attributes['block_name']}"
    if op.name == "external_call":
        return f"external:{op.attributes['name']}"
    return op.name


def _rule_display(key: str) -> str:
    """Message display for a missing rule keyed `key`.

    External-call keys name the `external_call` op (UX/tests pin the op
    name) with the key in parentheses; block keys and plain op names keep
    the key itself (which names the block / the op).
    """
    if key.startswith("external:"):
        return f"op 'external_call' (key '{key}')"
    return f"op '{key}'"


def _rule_hint(key: str, register_fn: str, block_method: str) -> str:
    """Registration-advice clause for a missing rule keyed `key`.

    Adaptive to the namespaced key prefixes: an `external:<name>` key points
    at the external-kernel handle (`etl.register_external_kernel`) and a
    `block:<name>` key at the `BlockOp` decorators; plain op-name keys keep
    the generic registry hint (`register_fn`).
    """
    if key.startswith("external:"):
        name = key[len("external:"):]
        return (
            f"register one via the external-kernel handle returned by "
            f"etl.register_external_kernel('{name}', fn) "
            f"(.batching_rule/.vjp_rule/.jvp_rule) or register a portable "
            f"decomposition"
        )
    if key.startswith("block:"):
        return (
            f"register one with {register_fn}('{key}', fn) "
            f"(custom blocks: BlockOp.{block_method})"
        )
    return f"register one with {register_fn}('{key}', fn)"


def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an `ir.Value` as a `core.SymbolicTensor` (dtype/shape from its type)."""
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def _is_real(tangent) -> bool:
    """True iff `tangent` is a real `ir.Value` (not None/ZeroTangent)."""
    return isinstance(tangent, ir.Value)
def _zeros(value: ir.Value) -> ir.Value:
    """Emit an all-zeros tensor with `value`'s shape/dtype (one `multiply`,
    plus a `cast` only when promotion would change the dtype).

    The integer scalar 0 keeps weak promotion dtype-exact for float/complex
    operands (the differentiable case); integer/bool primals would promote
    (e.g. int32 → int64), so those get an explicit cast back — a zero
    tangent/cotangent must carry the primal's dtype exactly.
    """
    zero = ops.multiply(_sym(value), 0)
    if zero.dtype != value.type.dtype:
        return ops.cast(zero, value.type.dtype).value
    return zero.value


def _materialize(tangent, primal_value: ir.Value) -> ir.Value:
    """Return `tangent` as an `ir.Value`; None/ZeroTangent → zeros of the
    primal's shape/dtype (a required-materialization point: graph outputs)."""
    if _is_real(tangent):
        return tangent
    return _zeros(primal_value)


def _accumulate(env: Dict[int, object], value: ir.Value, cotangent) -> None:
    """Accumulate `cotangent` (a real `ir.Value`) into `env` under `value`:
    a None/ZeroTangent slot is replaced; a real existing cotangent is added
    via `etl.ops.add` (cotangents of multiply-used values accumulate)."""
    existing = env.get(id(value))
    if existing is None or isinstance(existing, ZeroTangent):
        env[id(value)] = cotangent
    else:
        env[id(value)] = ops.add(_sym(existing), _sym(cotangent)).value


def _combine(entries) -> Optional[ir.Value]:
    """Combine accumulated cotangent entries: `None` for no entries, the
    single entry as-is, or the add-reduced `ir.Value` of multiple entries
    (built with `etl.ops.add` — the canonical cotangent-accumulation op)."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    total = _sym(entries[0])
    for other in entries[1:]:
        total = ops.add(total, _sym(other))
    return total.value


def _jvp_from_vjp(proxy: ir.Op, tangents, builder: ir.Builder):
    """Derive the JVP of `proxy` from its registered VJP rule (adjoint trick).

    The VJP rule builds the linear map L: c ↦ Jᵀc. Instantiate it with
    c = the op's (rebuilt) primal results — any well-typed c works, since the
    adjoint of L is independent of the concrete c — then apply L's transpose
    (J) to the tangents with a SECOND reverse sweep over the ops L emitted
    (the rule's emissions). Cotangents only propagate along paths that reach
    a ct slot (a `proxy.results` value): paths to primals are second-order
    terms and are dropped, which also keeps the graph free of dead
    second-order ops (the decomposition's own vjp rules never run).

    Returns a tuple aligned with `proxy.results` of
    `ir.Value | None | ZeroTangent` — the output tangents J·v.
    """
    if all(t is None or isinstance(t, ZeroTangent) for t in tangents):
        return (ZeroTangent(),) * len(proxy.results)

    key = _rule_name(proxy)
    vjp_rule = require_vjp_rule(key)
    before = {op.id for op in builder.current_block.ops}
    l_outputs = vjp_rule(proxy, tuple(proxy.results), proxy.operands)
    emitted = [op for op in builder.current_block.ops if op.id not in before]
    if not isinstance(l_outputs, (tuple, list)):
        raise TransformError(
            f"vjp rule for op '{key}' did not return a tuple of cotangents, "
            f"got {type(l_outputs).__name__}"
        )
    if len(l_outputs) != len(proxy.operands):
        raise TransformError(
            f"vjp rule for op '{key}' returned {len(l_outputs)} cotangents, "
            f"expected {len(proxy.operands)} (one per operand)"
        )
    for entry in l_outputs:
        if entry is not None and not isinstance(entry, (ir.Value, ZeroTangent)):
            raise TransformError(
                f"vjp rule for op '{key}' returned an invalid cotangent entry "
                f"{entry!r}: entries must be ir.Value | None | ZeroTangent"
            )

    # Second reverse sweep = the adjoint application J·v (transpose of L):
    # seed the tangents on L's outputs, then sweep back. Keep cotangents only
    # on the ct slots and on values downstream of them (the only paths that
    # reach J·v); every other operand's cotangent is a second-order term.
    ct_ids = {result.id for result in proxy.results}
    on_ct_path = set(ct_ids)
    for op in emitted:
        if any(operand.id in on_ct_path for operand in op.operands):
            on_ct_path.update(result.id for result in op.results)

    acc2: Dict[int, list] = {}
    for u, v in zip(l_outputs, tangents):
        if _is_real(u) and _is_real(v):
            acc2.setdefault(u.id, []).append(v)
    for op in reversed(emitted):
        if op.name == "constant":
            continue  # zero operands — nothing to propagate backward
        if not any(acc2.get(result.id) for result in op.results):
            continue  # no seeded result — off the ct path (second-order)
        per_result = tuple(_combine(acc2.get(result.id)) for result in op.results)
        vjp_fn = require_vjp_rule(_rule_name(op))
        in_cts = vjp_fn(op, per_result, op.operands)
        for operand, in_ct in zip(op.operands, in_cts):
            if _is_real(in_ct) and operand.id in on_ct_path:
                acc2.setdefault(operand.id, []).append(in_ct)

    # Output tangents: accumulated cotangents on the ct slots (the primal
    # results); missing -> structurally zero.
    out_tangents = []
    for result in proxy.results:
        entries = acc2.get(result.id)
        out_tangents.append(_combine(entries) if entries else ZeroTangent())
    return tuple(out_tangents)


def _validate_extra_specs(graph: Graph, extra_specs, expected: int, kind: str):
    """Normalize the flat extra-spec tuple (TensorSpec/None) for a sweep."""
    target = "inputs" if kind == "jvp" else "outputs"
    if not isinstance(extra_specs, (tuple, list)):
        raise TransformError(
            f"{kind}: expected a flat tuple of core.TensorSpec/None entries "
            f"aligned with the graph's flattened tensor {target}, got "
            f"{type(extra_specs).__name__}"
        )
    extra_specs = tuple(extra_specs)
    if len(extra_specs) != expected:
        raise TransformError(
            f"{kind}: got {len(extra_specs)} tangent/cotangent specs for "
            f"{expected} flattened tensor {target} — must align one-to-one"
        )
    for spec in extra_specs:
        if spec is not None and not isinstance(spec, core.TensorSpec):
            raise TransformError(
                f"{kind}: each entry must be a core.TensorSpec (an explicit "
                f"tangent/cotangent input) or None (zero), got {spec!r} "
                f"(type {type(spec).__name__})"
            )
    return extra_specs


def _sweep(graph: Graph, extra_specs, mode: str) -> Graph:
    """Combined forward-rebuild + dual-propagation machinery.

    `mode` is ``"jvp"`` (tangents pushed forward while rebuilding) or
    ``"vjp"`` (full forward rebuild, then one reverse pass). Both produce a
    NEW module/function: primal inputs + extra block args (one per non-None
    spec); outputs = (primal outputs, tangent/cotangent outputs).
    """
    # 1. Normalize/validate the flat extra-spec tuple.
    src_block = graph.module.main.entry_block
    terminator = src_block.terminator
    expected = len(graph.tensor_specs) if mode == "jvp" else len(terminator.operands)
    extra_specs = _validate_extra_specs(graph, extra_specs, expected, mode)

    # 2. New builder/module/function: primal inputs + extra block args.
    builder = ir.Builder()
    module = builder.build_module(name="main")
    primal_types = tuple(
        ir.ValueType(spec.dtype, tuple(spec.shape)) for spec in graph.tensor_specs
    )
    extra_types = tuple(
        ir.ValueType(spec.dtype, tuple(spec.shape))
        for spec in extra_specs
        if spec is not None
    )
    builder.build_function(name="main", input_types=primal_types + extra_types)
    block_args = builder.current_block.arguments
    primal_args = block_args[: len(primal_types)]
    extra_args = block_args[len(primal_types):]

    # 3. Seed the per-value envs: primal args map to themselves; extra args
    #    map to their dual seeds (None entries → ZeroTangent). For vjp the
    #    extra args are cotangents of the OUTPUTS and are seeded after the
    #    rebuild (the new output values do not exist yet).
    value_env: Dict[int, ir.Value] = {}
    dual_env: Dict[int, object] = {}
    for src_arg, new_arg in zip(src_block.arguments, primal_args):
        value_env[id(src_arg)] = new_arg
    extra_pos = 0
    if mode == "jvp":
        for index, spec in enumerate(extra_specs):
            if spec is None:
                dual_env[id(primal_args[index])] = ZeroTangent()
            else:
                dual_env[id(primal_args[index])] = extra_args[extra_pos]
                extra_pos += 1
    source_locations: Dict[int, object] = {}
    for src_arg, new_arg in zip(src_block.arguments, primal_args):
        location = graph.source_locations.get(src_arg.id)
        if location is not None:
            source_locations[new_arg.id] = location

    # 4. Forward rebuild: walk the original ops in order; REQUIRE the rule
    #    first (raises TransformError for rule-less ops such as runtime_call,
    #    collectives, control flow — correct, never a silent fallback), then
    #    recreate the op in the new builder with remapped operands and
    #    explicit result types (== the original results' types). The one op
    #    with NO rule by construction is `constant`: its data is fixed, so
    #    its output tangent is structurally ZeroTangent and its VJP (zero
    #    operands) propagates nothing — handled by the machinery, not the
    #    registries.
    recorded = []  # (proxy_op, rule_name) in emission order
    with with_builder(builder):
        for op in src_block.ops:
            if op.is_terminator:
                continue
            name = _rule_name(op)
            is_constant = op.name == "constant"
            if not is_constant:
                if mode == "jvp":
                    require_jvp_or_vjp_rule(name)
                else:
                    require_vjp_rule(name)
            new_operands = tuple(value_env[id(operand)] for operand in op.operands)
            new_op = builder.create(
                op.name,
                operands=new_operands,
                attributes=dict(op.attributes),
                result_types=tuple(result.type for result in op.results),
                location=op.location,
            )
            for src_result, new_result in zip(op.results, new_op.results):
                value_env[id(src_result)] = new_result
            proxy = ir.Op(
                name=op.name,
                id=op.id,
                operands=new_op.operands,
                attributes=dict(op.attributes),
                regions=(),
                results=new_op.results,
                location=op.location,
            )
            if mode == "jvp":
                if is_constant:
                    for result in new_op.results:
                        dual_env[id(result)] = ZeroTangent()
                else:
                    tangents = tuple(
                        dual_env.get(id(operand), ZeroTangent())
                        for operand in new_operands
                    )
                    jvp_rule = jvp_rules.get(name)
                    if jvp_rule is not None:
                        out_tangents = jvp_rule(proxy, tangents)
                    else:
                        # No JVP rule: derive it from the VJP rule via the
                        # adjoint (double-vjp) trick.
                        out_tangents = _jvp_from_vjp(proxy, tangents, builder)
                    if not isinstance(out_tangents, (tuple, list)):
                        raise TransformError(
                            f"jvp rule for op '{name}' did not return a tuple of "
                            f"tangents, got {type(out_tangents).__name__}"
                        )
                    if len(out_tangents) != len(new_op.results):
                        raise TransformError(
                            f"jvp rule for op '{name}' returned "
                            f"{len(out_tangents)} tangents, expected "
                            f"{len(new_op.results)} (one per result)"
                        )
                    for result, tangent in zip(new_op.results, out_tangents):
                        if tangent is not None and not isinstance(
                            tangent, (ir.Value, ZeroTangent)
                        ):
                            raise TransformError(
                                f"jvp rule for op '{name}' returned an invalid "
                                f"tangent entry {tangent!r}: entries must be "
                                f"ir.Value | None | ZeroTangent"
                            )
                        dual_env[id(result)] = (
                            tangent if _is_real(tangent) else ZeroTangent()
                        )
            recorded.append((proxy, name))

        new_primal_outputs = tuple(
            value_env[id(value)] for value in terminator.operands
        )

        # 5. Reverse sweep (vjp): seed output cotangents, then walk the
        #    recorded ops in REVERSE, accumulating input cotangents.
        if mode == "vjp":
            extra_pos = 0
            for src_result, spec in zip(terminator.operands, extra_specs):
                new_result = value_env[id(src_result)]
                if spec is None:
                    # In-graph scalar-ONE seed of the output's dtype (valid
                    # only for scalar outputs — `grad`/`vjp` validate that
                    # before calling; the check here is defensive).
                    if src_result.type.rank != 0:
                        raise ShapeError(
                            f"reverse_sweep: cannot seed a scalar-one "
                            f"cotangent for a non-scalar output of rank "
                            f"{src_result.type.rank} (shape "
                            f"{tuple(src_result.type.shape)}) — only scalar "
                            f"outputs support the None → ones cotangent "
                            f"default"
                        )
                    one = ops.constant(
                        core.tensor(np.ones((), dtype=src_result.type.dtype))
                    )
                    dual_env[id(new_result)] = one.value
                else:
                    dual_env[id(new_result)] = extra_args[extra_pos]
                    extra_pos += 1
            for proxy, name in reversed(recorded):
                if proxy.name == "constant":
                    continue  # zero operands — nothing to propagate backward
                cotangents = tuple(
                    dual_env.get(id(result), ZeroTangent())
                    for result in proxy.results
                )
                in_cotangents = vjp_rules[name](proxy, cotangents, proxy.operands)
                if not isinstance(in_cotangents, (tuple, list)):
                    raise TransformError(
                        f"vjp rule for op '{name}' did not return a tuple of "
                        f"cotangents, got {type(in_cotangents).__name__}"
                    )
                if len(in_cotangents) != len(proxy.operands):
                    raise TransformError(
                        f"vjp rule for op '{name}' returned "
                        f"{len(in_cotangents)} cotangents, expected "
                        f"{len(proxy.operands)} (one per operand)"
                    )
                for operand, cotangent in zip(proxy.operands, in_cotangents):
                    if cotangent is None or isinstance(cotangent, ZeroTangent):
                        continue
                    if not _is_real(cotangent):
                        raise TransformError(
                            f"vjp rule for op '{name}' returned an invalid "
                            f"cotangent entry {cotangent!r}: entries must be "
                            f"ir.Value | None | ZeroTangent"
                        )
                    _accumulate(dual_env, operand, cotangent)

        # 6. Materialize the extra outputs (required for every tensor slot)
        #    and terminate.
        if mode == "jvp":
            extra_outputs = tuple(
                _materialize(dual_env.get(id(value)), value)
                for value in new_primal_outputs
            )
        else:
            extra_outputs = tuple(
                _materialize(dual_env.get(id(arg)), arg) for arg in primal_args
            )
        builder.set_terminator(
            builder.current_block,
            "return",
            operands=new_primal_outputs + extra_outputs,
        )

    # 7. Result Graph: 2-tuple input tree (original tree, flat extra tree);
    #    tensor_specs = primal specs + non-None extra specs; None entries are
    #    static leaves recorded as StaticValue records (flatten_inputs
    #    requires leaf count == tensor specs + static records). 2-tuple
    #    output tree (original output tree, flat extra outputs); the original
    #    static/output-static records stay valid (their tree is the first
    #    child of the new tree, so flat indices are unchanged).
    _, extra_input_tree = core.flatten(extra_specs)
    input_tree = core.TreeSpec(
        type=tuple, children=(graph.input_specs, extra_input_tree)
    )
    tensor_specs = tuple(graph.tensor_specs) + tuple(
        spec for spec in extra_specs if spec is not None
    )
    static_values = list(graph.static_values)
    base = graph.input_specs.num_leaves
    for index, spec in enumerate(extra_specs):
        if spec is None:
            static_values.append(
                StaticValue(
                    index=base + index,
                    path=(1, index),
                    value=None,
                    kind=type(None).__qualname__,
                )
            )
    _, extra_output_tree = core.flatten(tuple(_sym(value) for value in extra_outputs))
    output_tree = core.TreeSpec(
        type=tuple, children=(graph.output_tree, extra_output_tree)
    )
    return Graph(
        module,
        input_tree,
        tensor_specs,
        output_tree,
        tuple(static_values),
        graph.output_static_values,
        source_locations,
    )


def forward_sweep(graph: Graph, tangent_specs) -> Graph:
    """Core forward-mode sweep (used by `jvp`).

    `tangent_specs` is a flat tuple aligned with `graph.tensor_specs` (the
    flattened primal tensor inputs): a `core.TensorSpec` entry becomes an
    extra block arg (the tangent input); `None` seeds a `ZeroTangent` (no
    block arg). Walks the graph's entry block in order, applies each op's JVP
    rule (via the proxy-op convention above), and returns a new `Graph`:
    inputs = 2-tuple `(original input tree, flat tuple tree of tangent
    specs)`; outputs = 2-tuple `(primal outputs with the original output
    tree, flat tuple of tangent tensors aligned with the flattened tensor
    outputs)` — ZeroTangent entries materialize as zeros. The result contains
    the full primal computation (rebuilt in the new module) plus the tangent
    ops. Ops without a JVP rule raise `core.TransformError` naming the
    op/block.
    """
    return _sweep(graph, tangent_specs, "jvp")


def reverse_sweep(graph: Graph, cotangent_specs) -> Graph:
    """Core backward sweep (used by `vjp` and `grad`).

    `cotangent_specs` is a flat tuple aligned with the flattened tensor
    outputs: a `core.TensorSpec` entry becomes an extra block arg (the
    cotangent input); `None` seeds an in-graph scalar-ONE constant of that
    output's dtype (valid only for scalar outputs — `grad`/`vjp` validate
    before calling). Rebuilds the full primal computation in a new module,
    then walks the recreated ops in REVERSE applying each op's VJP rule;
    cotangents of multiply-used values accumulate via `add`. Returns a new
    `Graph`: inputs = 2-tuple `(original input tree, flat tuple tree of
    cotangent block args)`; outputs = 2-tuple `(primal outputs with the
    original output tree, flat tuple of input cotangents aligned with the
    flattened tensor inputs — one tensor per tensor input)`. Ops without a
    VJP rule raise `core.TransformError` naming the op/block.
    """
    return _sweep(graph, cotangent_specs, "vjp")
