# etl/block — custom operations (`etl.block`)

## Intent

The declaration, registration, and lowering contract for user-defined
operations ("blocks"): the `etl.block` factory, `BlockOp`, the block/impl
registries, and the bridges that publish batching/derivative rules into
`etl.transforms` under namespaced keys. A `block_call` op is a placeholder in
the graph; how it actually computes is resolved at lower/transform time from
the resolution chain (portable decomposition → backend impl → compiler-native
op → future: vendor library / custom call). Backends never need to
understand blocks as first-class concepts beyond the op itself.

Binding parent contracts: root `../CONTEXT.md` (principles, error strategy,
import acyclicity) and `../CONTEXT.md` (public API + the "block" cross-module
bullet). This file details what those leave open for this node.

## API Surface

Re-exported from `__init__.py` (and into the `etl` namespace by the package
root, per `etl/CONTEXT.md`):

```python
block(name=None, inputs=None, outputs=None, attributes=None,
      effects="pure", batching=None, portable=None, *, result=None) -> BlockOp
```

Two usage forms:

**(a) Factory form** — `name` is a string; returns a registered, callable
`BlockOp` (flash_attention-style):

```python
flash_attention = etl.block(
    "flash_attention",
    inputs=[etl.TensorSpec((256, 1024), etl.float32, name="q"),
            etl.TensorSpec((256, 1024), etl.float32, name="k"),
            etl.TensorSpec((256, 1024), etl.float32, name="v")],
    outputs=[etl.TensorSpec((256, 1024), etl.float32)],
    attributes={"scale": float, "causal": bool},
    batching="opaque_batched",
)
o = flash_attention(q, k, v, scale=0.5, causal=True)
```

**(b) Decorator form** — `@etl.block` (bare or with keyword args) over an
`@etl.defn` function providing the portable implementation; name = function
name; input specs derived from TensorSpec annotations/defaults, output specs
from the return annotation (TensorSpec or tuple/list/dict pytree thereof) or
declared via `outputs=` / `result=`. A pending decorator object is returned
when no name and no function are given:

```python
@etl.block(outputs=[etl.TensorSpec((), etl.float32)])
@etl.defn
def swish(x: etl.TensorSpec((), etl.float32)):
    return etl.sigmoid(x) * x
```

**`BlockOp`** attrs: `name`, `input_specs`, `output_specs`, `attribute_schema`
(alias `attributes`, per the etl/CONTEXT.md bullet), `effects`, `batching_policy`,
plus `has_portable` / `get_impl(backend_name)`.
Methods: `__call__(*operands, **attributes)` (builds a `block_call` op),
`portable(fn)`, `impl(backend_name)` (decorator), `batching_rule(fn)`,
`jvp_rule(fn)`, `vjp_rule(fn)` (decorators). All decorators return `fn`.

**`get_block(name) -> BlockOp`** — registry accessor.

Static attribute declaration: `attributes={"scale": float, "causal": bool}`
(bare type = required) or `attributes={"eps": 1e-5}` (default instance =
optional, type inferred). At call time, keyword args must name declared
attributes; positional static values bind to the next unfilled attribute in
schema order. Python values in block calls are ALWAYS static (no scalar→tensor
auto-promotion; promote scalars at the ops level instead).

**Status:** implemented — `BlockOp.__call__` builds `block_call` ops, the
fallback rule wrappers inline portable decompositions, and `_ensure_ir_opdef`
verifies ir's canonical def. `decl.py`/`registry.py`/`errors.py` complete.

## BlockCall IR op layout (coordinate with etl/ir)

The CANONICAL op def lives in `etl/ir/op_defs/control.py` (ir owns op defs).
`_ensure_ir_opdef()` (in `op.py`) only VERIFIES it lazily before the first op
is built (missing def or mismatched attr schema → `BlockError` pointing at
`etl/ir/op_defs/control.py`) — block never registers a conflicting def.

Real schema (as of the ir implementation):

- name `"block_call"`; operands: variadic ir Values in `input_specs` order;
  results: one per `output_spec`, resolved by the Builder from the
  `result_specs` attr (ir.verify requires the ValueType entries to equal the
  op's result types exactly).
- attrs: `block_name` (str), `static_args` (ATTR_ANY, default `()`), and
  `result_specs` (ATTR_ANY). There are NO `effects`/`batching_policy` attrs
  and the op's effect is fixed at `read` (optional ir-owned enhancements —
  see "Remaining coordination with siblings" below): the declared block's
  effects/policy do not ride on the op, so consumers must consult
  `get_block(block_name)` for them.
- `static_args` is a dict attribute-name → `{"kind": ..., "value": ...}` —
  a plain JSON-safe payload, never a `StaticValue` object (ir's ATTR_ANY
  serialization round-trips it; verified via save/load). `StaticValue` (in
  `decl.py`) is the tagged encoder with kinds
  `none|bool|int|float|complex|str|slice|dtype|enum`; anything else is a
  `BlockError` at call time — never silent pickling. Optional attributes
  left unset are NOT recorded (their defaults are fixed by the schema).

## Batching policy semantics

| policy | semantics (when NO explicit rule is registered) |
|---|---|
| `elementwise` | op acts per-element on all dims (relu-like); batch dims pass through — safe without a rule |
| `batching_rule` | driven by a registered rule, or by the default fallback: inline the portable decomposition |
| `broadcast_batch` | op broadcasts batch dims among operands; needs a rule/decomposition |
| `map_over_batch` | op maps over its leading batch dims internally; batch dims pass through untouched — safe without a rule |
| `opaque_batched` | op consumes batch dims opaquely (flash_attention); further vectorization → TransformError unless an explicit rule exists |
| `unsupported` | no safe path; vectorize/vmap raise TransformError unless an explicit rule exists (never guess) |

Default resolution (`batching=None`): portable exists → `batching_rule`
(decl pre-registers the decomposition as the namespaced batching rule so
transforms always finds an entry); else `unsupported`. A registered explicit
rule always wins over the policy. An explicit policy suppresses the automatic
decomposition fallback.

**Safe-policy dispatch (implemented):** for `elementwise` and
`map_over_batch`, decl pre-registers transforms' built-in pass-through rule
(`transforms.batching.block_call_pass_through_rule()` — rebuilds the
`block_call` over pointwise-aligned batched operands) under `block:<name>` at
declaration time. That is what makes those policies "safe without a user
rule". An explicit user rule registered afterwards overwrites the entry (dict
assignment): `BlockOp.batching_rule(fn)` → `register_batching_rule` → last
registration wins. Nothing is pre-registered for `opaque_batched`,
`unsupported`, `broadcast_batch`, or an explicit `batching_rule` policy — for
those, vectorize/vmap raise `TransformError` naming the block unless an
explicit rule (or, for the default-resolved `batching_rule`, the
decomposition fallback) exists.

## Resolution policy (lowering a block_call)

Per backend capabilities, in order — first available wins, none → explicit
`BackendError`/`TransformError` (never silent):

1. **Compiler-native op** — backend's `capabilities.custom_blocks` lists the
   block name → keep the op for the compiler to consume natively (IREE/XLA
   custom ops; future).
2. **Registered backend impl** — `impl(backend_name)` entry (v1: `"numpy"`;
   the registry is backend-neutral so future names register identically).
3. **Portable decomposition** — trace the registered `etl.defn` and inline
   the resulting ordinary-op graph; works on every backend.
4. (Future documented tiers: vendor-library dispatch, custom-call registration.)

Derivatives (grad/jvp/vjp): explicit rule → use it; else portable
decomposition (inline → differentiate) → else TransformError ("not
differentiable"). Batching (vectorize/vmap): explicit rule → use it; else
policy-based pass-through (elementwise/map_over_batch) → else the registered
decomposition fallback → else TransformError.

## Impl-registry contract

`registry.py` owns `_BLOCKS` (name → BlockOp), `_IMPLS` ((name, backend_name)
→ callable), `_PORTABLES` (name → defn). `get_impl` returns None when absent
(callers decide). Portable impls MUST be `@etl.defn` functions (validated via
the `__etl_defn__` marker; traced lazily). Declaration is registration:
duplicate names → `BlockError` (reuse `get_block(name)` to add impls/rules).
v1 numpy impls are callables the numpy interpreter backend invokes when it
meets the block_call — the exact call signature is finalized with
`etl/backends` at implementation time.

## Rule bridge to transforms

`rules.py` publishes into `etl.transforms` registries under `block:<name>`
(import of transforms is lazy, inside function bodies — import acyclicity):

- batching rules → `transforms.batching_rules`
- vjp rules → `transforms.vjp_rules` (per etl/CONTEXT.md)
- jvp rules → `transforms.jvp_rules` (transforms provides this dict; the
  portable diff fallback registers ONLY the vjp rule and transforms derives
  jvp from vjp when no jvp rule exists)
- fallbacks (registered by decl when a portable exists): batching fallback
  only when the resolved policy is `batching_rule`; derivative fallback
  always. The fallback wrappers inline the decomposition.

Frozen transforms rule signatures (binding — see `../transforms/CONTEXT.md`):

- batching: `rule(op, operands, axes) -> (new_values, new_axes)` —
  `operands`: tuple of ir.Value; `axes`: aligned `MappedAxes` (from
  `transforms._metadata`); `new_values` aligned with `op.results`;
  `new_axes` aligned with `new_values`.
- vjp: `rule(op, cotangents, primals) -> input_cotangents` — `cotangents`
  aligned with `op.results`, `primals` = `op.operands`; entries may be
  `ir.Value | None | ZeroTangent` (`transforms.autodiff`); returns aligned
  with `op.operands`.
- jvp: `rule(op, tangents) -> output_tangents`.

Fallback semantics (implemented in `rules.py`):

- **Batching fallback** traces the portable over the (batched) operand
  values in the active transform builder, tracks `MappedAxes` through the
  inlined ops in creation order (result axes = union of operand axes — the
  longest contiguous leading tuple; constant ops are unmapped), and returns
  the decomposition's outputs + their axes. Union semantics = the
  decomposition is polymorphic in leading batch dims (elementwise-style);
  transforms' vectorize consumes the returned values directly. Missing
  portable → `TransformError` (never guesses).
- **VJP fallback** short-circuits to all-`ZeroTangent` when every cotangent
  is None/ZeroTangent; otherwise traces the portable over the primals and
  runs a LOCAL reverse sweep over ONLY the inlined ops (reverse creation
  order): per-result cotangents accumulate (`etl.ops.add`), rules are looked
  up from the PUBLIC registries (`transforms.autodiff.require_vjp_rule`),
  and nested `block_call`s resolve via their own `block:<name>` keys.

## Error behavior

- `BlockError(ETLError)` — declaration/call misuse: duplicate/unknown names,
  malformed specs/schemas/policies/effects, undeclared or wrongly typed
  static values, non-defn portables. Defined in `errors.py`; the root error
  list does not name it, so core may later re-export it.
- `TraceError` — call outside a trace; concrete `Tensor` operand (no eager mode).
- `ShapeError` / `DTypeError` — operand mismatch against `input_specs`.
- `TransformError` — missing batching/derivative rule (raised by transforms).

## Constraints

- Imports: `etl.core`/`etl.ir` at module level are OK; ALL imports of
  `etl.ops`, `etl.trace`, `etl.transforms` are LAZY (inside function bodies).
  Import the trace submodule via the direct form (`from etl.trace import
  current_builder`) — `from etl import trace` yields the trace *function*
  attribute (shadowed in `etl/__init__.py`), not the module.
- Files < ~1000 lines; split along the boundaries below.
- No `etl/__init__.py` in this node (root-owned).

## Test strategy

pytest suite lives in `../tests/block/` (sibling — read-only from here;
escalate test writes to root). Planned coverage:
- declaration: factory + decorator forms, spec derivation from annotations,
  duplicate/invalid names, invalid effects/policies, attribute schema
  normalization (type vs default), required-vs-optional attributes;
- call semantics: block_call op construction + attrs, operand dtype/shape
  errors, static specialization, TraceError outside a trace;
- examples: flash_attention declaration (`opaque_batched`), swish portable
  decorator (decomposition fallbacks registered under `block:swish`);
- rules: keys land in transforms registries; fallback registration only for
  the default policy; explicit policy suppresses it;
- serialization: StaticValue round-trips (enum/dtype/slice), static attrs in
  saved artifacts, get_block errors.

## Routing table

| File | Area |
|---|---|
| `./decl.py` | `block()` factory (both forms), name/spec/schema/effects/policy validation, StaticValue encoding |
| `./op.py` | `BlockOp` class, BLOCK_CALL_OPDEF (informational mirror), `_ensure_ir_opdef` (lazy ir verification), `_get_location` |
| `./registry.py` | `get_block`, BlockOp/impl/portable registries |
| `./rules.py` | bridges into transforms rule registries (batching/vjp/jvp + safe-policy pass-through) + decomposition fallback wrappers (batching/vjp) |
| `./errors.py` | `BlockError` |

## Notes for agents

- transforms NEVER imports block: it sees only `batching_rules`/
  `vjp_rules`/`jvp_rules` entries under `block:<name>` (fallbacks are
  pre-registered by decl). Keep that boundary intact.
- The decorator form requires `@etl.defn` (portable impls are traced graphs).
- `attribute_schema` and `attributes` are the same thing (contract name vs
  objective name) — keep both properties in sync.
- `StaticValue.decode` for `enum` resolves the longest importable prefix of
  the dotted path as the module and walks the remainder as attributes
  (naive `rpartition(".")` mis-splits `module.qualname.NAME`).
- `BlockOp.__call__` positional statics bind in SCHEMA order; a positional
  whose next schema-ordered slot was already supplied by keyword raises
  `BlockError` (no double-fill).
- Remaining coordination with siblings: ir could optionally add a
  `batching_policy` attr to the `block_call` opdef (transforms supports an
  attribute channel; the rule channel already covers in-process dispatch) and
  op-level `effects` (currently fixed at `read`); backends need the numpy
  impl signature for `block_call` lowering; persist must encode `static_args`
  payloads through its envelope codec (currently handled by ir's generic
  attribute encoding).
