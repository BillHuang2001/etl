"""Bridges from BlockOp rule registration into the etl.transforms registries.

Rules are keyed `block:<name>` so transforms only ever looks up namespaced
keys for a `block_call` op and never needs to import etl.block — keeping the
import graph acyclic (transforms sits above ops; block sits beside it).

Imports of `etl.transforms` happen inside function bodies (lazily). The
fallback wrappers (portable decomposition inlining) are Phase 2: their
bodies raise NotImplementedError.

Rule callback contracts (owned by etl.transforms):
- batching rule: fn(*symbolic_args, batch_meta) -> transformed args and
  output batch axes, or a replacement Graph; `batch_meta` carries the mapped
  axis metadata supplied by vectorize/vmap.
- vjp rule: fn(cotangents..., *primal_args) -> cotangents for the primal args
  (per transforms' vjp_rules contract).
- jvp rule: fn(tangents..., *primal_args) -> output tangents (stored in
  transforms.jvp_rules — see the coordination note in CONTEXT.md).
"""

from __future__ import annotations

from typing import Any, Callable

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
    (decomposition) and lets transforms batch the resulting ordinary ops.
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


def _portable_batching_rule(name: str) -> Callable:
    """Fallback batching rule: inline the portable decomposition (Phase 2).

    At rule time this traces `registry.get_portable(name)` into ordinary ops
    (replacing the block_call), then batches the resulting graph — the
    decomposition makes batching safe automatically, on every backend.
    """

    def rule(*symbolic_args: Any, batch_meta: Any) -> Any:
        raise NotImplementedError("block: portable-decomposition batching fallback (Phase 2)")

    return rule


def _portable_vjp_rule(name: str) -> Callable:
    """Fallback vjp rule: inline the portable decomposition (Phase 2).

    At rule time this traces `registry.get_portable(name)` into ordinary ops
    (replacing the block_call), then differentiates the resulting graph.
    """

    def rule(*args: Any) -> Any:
        raise NotImplementedError("block: portable-decomposition derivative fallback (Phase 2)")

    return rule
