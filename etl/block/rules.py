"""Bridges from BlockOp rule registration into the etl.transforms registries.

Rules are keyed `block:<name>` so transforms only ever looks up namespaced
keys for a `block_call` op and never needs to import etl.block — keeping the
import graph acyclic (transforms sits above ops; block sits beside it).

Imports of `etl.transforms` (and `etl.core` / `etl.ops` / `etl.trace`) happen
inside function bodies (lazily).

Rule callback contracts (owned by etl.transforms, binding):
- batching rule: `rule(op, operands, axes) -> (new_values, new_axes)` —
  `operands`: tuple of `ir.Value` (the op's operand values); `axes`: tuple of
  `MappedAxes` aligned with `operands` (contiguous leading mapped-axis
  indices); `new_values`: tuple of `ir.Value` aligned with `op.results`;
  `new_axes`: tuple of `MappedAxes` aligned with `new_values`.
- vjp rule: `rule(op, cotangents, primals) -> input_cotangents` —
  `cotangents` aligned with `op.results`; `primals` = the op's operand
  values; entries of both may be `ir.Value | None | ZeroTangent`; returns a
  tuple aligned with `op.operands` (None/ZeroTangent = zero gradient).
- jvp rule: `rule(op, tangents) -> output_tangents` — `tangents` aligned
  with `op.operands`; returns a tuple aligned with `op.results` (stored in
  transforms.jvp_rules — see the coordination note in CONTEXT.md).

The fallback wrappers inline the portable decomposition: batching re-traces
the portable over the (batched) operand values, which makes batching safe
automatically on every backend; vjp re-traces the portable and runs a LOCAL
reverse sweep over only the inlined ops (nested block_calls inside the
decomposition resolve via their own `block:<name>` keys).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

from . import registry
from .errors import BlockError

__all__ = [
    "register_batching_rule",
    "register_jvp_rule",
    "register_portable_batching_fallback",
    "register_portable_diff_fallback",
    "register_vjp_rule",
]


def _validate(name: Any, fn: Any) -> None:
    if not isinstance(name, str) or not name:
        raise BlockError(f"block name must be a non-empty string, got {name!r}")
    if not callable(fn):
        raise BlockError(f"rule must be callable, got {fn!r}")


def register_batching_rule(name: str, fn: Callable) -> None:
    """Register a vmap/vectorize batching rule under `block:<name>`."""
    _validate(name, fn)
    from etl import transforms

    transforms.batching_rules[f"block:{name}"] = fn


def register_vjp_rule(name: str, fn: Callable) -> None:
    """Register a reverse-mode derivative rule under `block:<name>`."""
    _validate(name, fn)
    from etl import transforms

    transforms.vjp_rules[f"block:{name}"] = fn


def register_jvp_rule(name: str, fn: Callable) -> None:
    """Register a forward-mode derivative rule under `block:<name>`."""
    _validate(name, fn)
    from etl import transforms

    transforms.jvp_rules[f"block:{name}"] = fn


def register_portable_batching_fallback(name: str) -> None:
    """Install the portable decomposition as the batching rule for `block:<name>`.

    Called by decl at declaration time when a portable implementation exists
    and no explicit batching policy was given (the resolved policy is
    BATCHING_RULE). Transforms then finds an entry and never has to know
    about block's registry: the fallback inlines the portable graph
    (decomposition) over the batched operand values and returns the
    decomposition's outputs as the replacement values.
    """
    if not isinstance(name, str) or not name:
        raise BlockError(f"block name must be a non-empty string, got {name!r}")
    from etl import transforms

    transforms.batching_rules[f"block:{name}"] = _portable_batching_rule(name)


def register_portable_diff_fallback(name: str) -> None:
    """Install the portable decomposition as the derivative fallback (vjp rule).

    Called by decl whenever a portable implementation exists, so grad/jvp/vjp
    on the block_call can inline the decomposition and differentiate the
    ordinary ops. jvp is derived from the vjp rule by transforms.
    """
    if not isinstance(name, str) or not name:
        raise BlockError(f"block name must be a non-empty string, got {name!r}")
    from etl import transforms

    transforms.vjp_rules[f"block:{name}"] = _portable_vjp_rule(name)


# ---------------------------------------------------------------------------
# Fallback helpers (shared by the portable-decomposition rules)
# ---------------------------------------------------------------------------


def _flatten_outputs(name: str, node: Any, out: list) -> None:
    """Flatten a portable's return value into symbolic tensors.

    Mirrors decl's `_collect_specs` order (tuple/list items, dict values) so
    the flattened outputs align with the declared `output_specs`.

    Raises:
        BlockError: A non-symbolic return value (portables must produce
            tensor outputs).
    """
    from etl import core

    if isinstance(node, core.SymbolicTensor):
        out.append(node)
        return
    if isinstance(node, (tuple, list)):
        for item in node:
            _flatten_outputs(name, item, out)
        return
    if isinstance(node, dict):
        for item in node.values():
            _flatten_outputs(name, item, out)
        return
    raise BlockError(
        f"portable implementation for block '{name}' must return symbolic "
        f"tensors (SymbolicTensor or a tuple/list/dict thereof), got "
        f"{type(node).__name__}"
    )


def _inline_portable(
    name: str, portable: Callable, operand_values: Tuple[Any, ...]
) -> Tuple[Any, ...]:
    """Trace the portable defn over `operand_values` in the active builder.

    The ops the function builds land in the currently active builder —
    during a transform rule that is the transform's builder (guaranteed by
    the transforms contract). Returns the decomposition's outputs as a tuple
    of SymbolicTensors (a single SymbolicTensor output normalizes to a
    1-tuple).

    Raises:
        BlockError: The portable returned a non-symbolic value (portables
            must produce tensor outputs).
    """
    from etl import core

    inner = getattr(portable, "fn", portable)
    syms = tuple(
        core.SymbolicTensor(value=v, dtype=v.type.dtype, shape=v.type.shape)
        for v in operand_values
    )
    outputs: list = []
    _flatten_outputs(name, inner(*syms), outputs)
    return tuple(outputs)


def _accumulate(entries: Any) -> Any:
    """Cotangent accumulation: None for no entries, the single entry as-is,
    or the add-reduced `ir.Value` of multiple entries (built with
    `etl.ops.add` into the active transform builder — the canonical
    cotangent-accumulation op)."""
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    from etl import core, ops

    wrapped = [
        core.SymbolicTensor(value=v, dtype=v.type.dtype, shape=v.type.shape)
        for v in entries
    ]
    total = wrapped[0]
    for other in wrapped[1:]:
        total = ops.add(total, other)
    return total.value


def _zero(cotangent: Any) -> bool:
    """True when `cotangent` means "zero gradient" (None or ZeroTangent)."""
    if cotangent is None:
        return True
    from etl.transforms.autodiff import ZeroTangent

    return isinstance(cotangent, ZeroTangent)


# ---------------------------------------------------------------------------
# Portable-decomposition fallback rules
# ---------------------------------------------------------------------------


def _portable_batching_rule(name: str) -> Callable:
    """Fallback batching rule: inline the portable decomposition (batched).

    At rule time this traces `registry.get_portable(name)` over the (already
    batched) operand values: the decomposition is polymorphic in leading
    batch dims (elementwise-style), so shape inference threads the batch
    dims through and the outputs are the batched replacements for the
    block_call's results. Per-value `MappedAxes` metadata is tracked through
    the inlined ops IN ORDER (seeded from the operands; each op's result
    axes = the union — the longest contiguous leading tuple — of its operand
    axes; constant ops with no operands are unmapped). The rule returns
    `(output_values, result_axes)`; transforms' vectorize consumes the
    returned values directly.

    Raises:
        TransformError: No portable decomposition is registered (vectorize/
            vmap never guesses).
    """

    def rule(op: Any, operands: Any, axes: Any) -> Any:
        from etl import core
        from etl.trace import current_builder
        from etl.transforms._metadata import MappedAxes

        portable = registry.get_portable(name)
        if portable is None:
            raise core.TransformError(
                f"vectorize/vmap: no batching rule and no portable "
                f"decomposition for block '{name}' — vectorize/vmap never "
                "guesses; register an explicit rule (BlockOp.batching_rule) "
                "or a portable implementation"
            )
        builder = current_builder()
        before = len(builder.current_block.ops)
        outputs = _inline_portable(name, portable, tuple(operands))
        inlined_ops = builder.current_block.ops[before:]

        # Track MappedAxes through the decomposition in creation order.
        env = {value.id: ax for value, ax in zip(operands, axes)}
        for inlined in inlined_ops:
            op_axes = [
                env.get(operand.id, MappedAxes())
                for operand in inlined.operands
            ]
            # Union of contiguous leading axis tuples = the longest one.
            result_axes = (
                max(op_axes, key=lambda m: len(m.axes))
                if op_axes
                else MappedAxes()
            )
            for result in inlined.results:
                env[result.id] = result_axes

        result_axes = tuple(
            env.get(out.value.id, MappedAxes()) for out in outputs
        )
        return (tuple(out.value for out in outputs), result_axes)

    return rule


def _portable_vjp_rule(name: str) -> Callable:
    """Fallback vjp rule: inline the portable decomposition, differentiate it.

    At rule time this traces `registry.get_portable(name)` into ordinary ops
    (replacing the block_call) and runs a LOCAL reverse sweep over ONLY the
    inlined ops (reverse creation order), so the derivative is computed over
    the decomposition without needing a block_call vjp rule. Cotangents
    accumulate per value; nested block_calls inside the decomposition resolve
    through their own `block:<name>` keys via the public registries
    (transforms.autodiff). All-zero cotangents short-circuit to zero
    cotangents without inlining.

    Raises:
        TransformError: No portable decomposition is registered, or an op in
            the decomposition has no vjp rule (canonical `require_vjp_rule`
            message) — never a silent fallback.
    """

    def rule(op: Any, cotangents: Any, primals: Any) -> Any:
        from etl import core
        from etl.trace import current_builder
        from etl.transforms.autodiff import ZeroTangent, require_vjp_rule

        portable = registry.get_portable(name)
        if portable is None:
            raise core.TransformError(
                f"grad/vjp: not differentiable: no vjp rule and no portable "
                f"decomposition for block '{name}' — register an explicit "
                "rule (BlockOp.vjp_rule) or a portable implementation; "
                "there is no silent fallback"
            )

        # All-zero cotangents: nothing to backpropagate — no inlining.
        if all(_zero(ct) for ct in cotangents):
            return (ZeroTangent(),) * len(op.operands)

        builder = current_builder()
        before = len(builder.current_block.ops)
        _inline_portable(name, portable, tuple(primals))
        inlined_ops = builder.current_block.ops[before:]

        # Seed cotangents from the block's results (skip None/ZeroTangent).
        acc: dict = {}
        for result_value, ct in zip(op.results, cotangents):
            if not _zero(ct):
                acc.setdefault(result_value.id, []).append(ct)

        # Local reverse sweep over ONLY the inlined ops.
        for inlined in reversed(inlined_ops):
            if not any(acc.get(result.id) for result in inlined.results):
                continue  # dead for backprop: no result carries a cotangent
            per_result = tuple(
                _accumulate(acc.get(result.id)) for result in inlined.results
            )
            key = inlined.name
            if key == "block_call":
                key = f"block:{inlined.attributes['block_name']}"
            # Public registry lookup: canonical TransformError when absent.
            vjp_fn = require_vjp_rule(key)
            input_cotangents = vjp_fn(inlined, per_result, inlined.operands)
            for operand_value, in_ct in zip(inlined.operands, input_cotangents):
                if not _zero(in_ct):
                    acc.setdefault(operand_value.id, []).append(in_ct)

        # Finalize: aligned with op.operands (empty -> ZeroTangent).
        result = []
        for primal in op.operands:
            entries = acc.get(primal.id, [])
            if not entries:
                result.append(ZeroTangent())
            else:
                result.append(_accumulate(entries))
        return tuple(result)

    return rule
