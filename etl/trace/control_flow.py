"""Runtime tensor control flow: `cond`, `while_loop`, `scan`.

These trace Python callables into IR regions of ordinary `if`/`while` ops —
backends (the numpy interpreter) need NO special control-flow runtime
support. Region ops are built through `ir.opdef(...)` + the region `Builder`
directly; this module NEVER imports `etl.ops` (ops imports trace for the
active-builder hook — the DAG stays acyclic).

REGION CONVENTIONS (binding — coordinate with `./etl/ir`):

* `if` op  (registry name "if"; effects: none; `regions` = 2 regions)
  - operands: none — branch bodies reference SSA values captured from the
    enclosing function (values dominate their uses).
  - `regions[0]` = "then", `regions[1]` = "else"; each has exactly one block
    with NO block args.
  - Each region block's terminator is a `return` op yielding the branch's
    outputs (n `core.SymbolicTensor`s). Both branches must yield the same n
    and unify to the same result dtype/shape (DimExpr unification);
    mismatch → `core.TraceError` (never a silent fallback).
  - op results: n values = the selected branch's outputs.
* `while` op (registry name "while"; effects: none; `regions` = 2 regions)
  - operands: n initial carried SSA values.
  - `regions[0]` = condition, `regions[1]` = body. Each region has one block
    with n block args — the loop-carried values (types match the op's
    operand types). "Block args carry loop-carried values."
  - condition region terminator: `return` of ONE scalar (0-d, bool dtype)
    SymbolicTensor — anything else → `core.TraceError`.
  - body region terminator: `return` of n next-iteration carried values.
  - op results: n final carried values.
* `return` op: terminator-only (no results, no regions); operands = yielded
  values; valid only as the last op of a block.

Region building: obtain a region builder from the enclosing active builder
(assumed API: `ir.Builder` exposes a way to build inside a region's block,
e.g. `current_builder().region_builder(region)` or equivalent — coordinate
with the ir architect), create the region's block, then run the user callable
under `with_builder(region_builder)`.

Static values: `*operands` / `init` / kwargs may contain static Python values
(per the static-value predicate in `./trace.py`); they specialize the regions
(Python semantics — evaluated once at trace time, NOT per iteration) and are
passed to the callables unchanged. Static values are never loop-carried or
op results.
"""

from __future__ import annotations

from typing import Any, Optional

from etl import core
from etl import ir

__all__ = ["cond", "scan", "while_loop"]


def cond(pred: "core.SymbolicTensor", true_fn: Any, false_fn: Any, *operands: Any, **static_kwargs: Any) -> Any:
    """Runtime `if` over a tensor predicate → traced `if` op.

    Contract (implement in Phase 2):

    1. `pred` must be a `core.SymbolicTensor` of 0-d bool dtype — else
       `core.TraceError` (non-scalar or non-bool or concrete value).
    2. `static_kwargs` must be static Python values (else `core.TraceError`);
       they specialize the regions and are passed to both branches as kwargs.
       `*operands` may be `SymbolicTensor`s (captured SSA values) and static
       values (specialization); passed positionally to both branches.
    3. Build the `if` op via `ir.opdef("if")` with two regions per the
       conventions above. Run `true_fn(*operands, **static_kwargs)` inside
       the then-region under `with_builder(...)`, `false_fn` likewise in the
       else-region. Each is called exactly ONCE at trace time.
    4. Flatten each branch's return value (pytree); trees must be identical
       and leaves must be `SymbolicTensor` (static leaves → `TraceError`).
       Emit each region's `return` terminator with its leaves; result
       dtype/shape unification across branches (mismatch → `TraceError`).
    5. Return the `if` op's results unflattened per the branch output tree
       (single tensor → returned bare).

    The numpy interpreter backend executes regions by selecting the branch —
    no graph-level Python callbacks.
    """
    raise NotImplementedError(
        "etl.trace.cond: Phase 2 implementation — see docstring contract and "
        "./CONTEXT.md region conventions."
    )


def while_loop(cond_fn: Any, body_fn: Any, init: Any) -> Any:
    """Runtime `while` over a tensor condition → traced `while` op.

    Contract (implement in Phase 2):

    1. `init` = `SymbolicTensor` or pytree of (`SymbolicTensor` | static
       value). Static leaves specialize the regions and are NOT loop-carried.
    2. Build the `while` op via `ir.opdef("while")`: operands = the flat
       carried `SymbolicTensor`s; two regions whose blocks carry them as
       block args (conventions above).
    3. Run `cond_fn(carried)` once inside the condition region (carried
       reconstructed per init's tree, static leaves as-is). Its return must
       be a scalar 0-d bool `SymbolicTensor` — else `core.TraceError`.
       Emit the condition region's `return` with it.
    4. Run `body_fn(carried)` once inside the body region. Its return tree
       must equal init's tree exactly (tensor leaves → next carried values;
       static leaves must match the init's static values — else
       `core.TraceError`). Emit the body region's `return` with the flat
       next-carried values.
    5. Return the `while` op's results (final carried values) unflattened
       per init's tree.

    v1 note: the condition/body callables run ONCE at trace time; the traced
    regions repeat their IR at run time — no Python callbacks.
    """
    raise NotImplementedError(
        "etl.trace.while_loop: Phase 2 implementation — see docstring "
        "contract and ./CONTEXT.md region conventions."
    )


def scan(f: Any, init: Any, xs: Any, length: Optional[int] = None) -> tuple:
    """Scan along a leading axis → `(carry, stacked_outputs)`.

    Contract (implement in Phase 2):

    1. `xs` = `SymbolicTensor` (or pytree of them) whose leading axis is the
       scan axis. `length`: static `int` or `None` → derived from xs's
       static leading dim (must be a static int — else `core.TraceError`).
       SYMBOLIC length → `core.TraceError` in v1 (documented: dynamic-scan
       region ops are reserved; no silent fallback).
    2. Desugars to `while_loop`: carried = (counter i32 0-d scalar,
       carried init..., stacked outputs...). `f(carry, x_step)` is invoked
       ONCE inside the body region; `x_step` = per-leaf slice of xs at the
       counter. Returns `(new_carry, y_step)`; each y leaf is stacked along a
       new leading axis of size `length`.
    3. All building blocks are raw region ops via `ir.opdef(...)` +
       `Builder` (NEVER `etl.ops` — keep the import DAG acyclic). Required
       registry op defs (coordination with the ir architect): `add` (counter
       increment), `less` (counter < length condition), `slice` (dynamic
       index step), `expand_dims` + `concatenate` (stacking).
    4. Returns `(final_carry, stacked_outputs)` — carry per init's tree,
       stacked outputs per y's tree (single tensor → returned bare).
    """
    raise NotImplementedError(
        "etl.trace.scan: Phase 2 implementation — see docstring contract and "
        "./CONTEXT.md (v1: static length only; symbolic length raises "
        "TraceError)."
    )


def _return_terminator(builder: "ir.Builder", values: Any) -> "ir.Op":
    """Build the `return` terminator op (`ir.opdef("return")`) with `values`
    (flat list of `ir.Value`) in the given block via `builder`. Private
    helper for `trace`, `cond`, `while_loop`, `scan` — Phase 2."""
    raise NotImplementedError(
        "etl.trace._return_terminator: Phase 2 implementation — build a "
        "return op via ir.opdef('return') on the given builder."
    )


def _region_builder(enclosing: "ir.Builder", region: "ir.Region") -> "ir.Builder":
    """Return a builder positioned inside `region`'s (single) block, creating
    the block if needed. Private helper — Phase 2 (coordinate with the ir
    architect's Builder API for regions/block args)."""
    raise NotImplementedError(
        "etl.trace._region_builder: Phase 2 implementation — coordinate with "
        "the ir.Builder region API (see ./CONTEXT.md)."
    )
