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

**Status:** architecture stubs complete (this phase). All algorithmic bodies
(`BlockOp.__call__`, `_ensure_ir_opdef`, fallback rule wrappers) raise
NotImplementedError — implementation is delegated to the Manager phase.

## BlockCall IR op layout (coordinate with etl/ir)

`BLOCK_CALL_OPDEF` (in `op.py`) is registered into ir's op registry lazily by
`_ensure_ir_opdef()` (Phase 2; exact ir hook finalized at integration):

- name `"block_call"`; operands: variadic ir Values in `input_specs` order;
  results: one per `output_spec`, typed from dtype + dims (DimExpr dims stay
  symbolic, None dims are runtime-dynamic).
- attrs: `block_name` (str), `static` (dict attribute-name → `StaticValue`
  payload), `effects` (ir effect kind), `batching_policy` (str). The op's
  effect = the `effects` attr.
- `static` values specialize the op (like static values specialize a traced
  graph): they participate in op identity, cache keys, and serialization.
  `StaticValue` (in `decl.py`) is a JSON-safe tagged encoding with kinds
  `none|bool|int|float|complex|str|slice|dtype|enum`; anything else is a
  `BlockError` at call time — never silent pickling.
- `batching_policy` rides on the op so transforms can handle elementwise /
  map_over_batch ops without importing block or consulting its registry.

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
- jvp rules → `transforms.jvp_rules` (coordination item: transforms must
  provide this dict, or derive jvp from vjp when absent)
- fallbacks (registered by decl when a portable exists): batching fallback
  only when the resolved policy is `batching_rule`; derivative fallback
  always. The fallback wrappers inline the decomposition (Phase 2).

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
- Files < ~1000 lines; split along the boundaries below.
- No algorithms in this phase: bodies raise NotImplementedError with the full
  semantics in docstrings.
- No `etl/__init__.py` in this phase (root-owned).

## Test strategy

pytest suite lives in `../tests/block/` (sibling — read-only from here;
escalate test writes to root). Planned coverage:
- declaration: factory + decorator forms, spec derivation from annotations,
  duplicate/invalid names, invalid effects/policies, attribute schema
  normalization (type vs default), required-vs-optional attributes;
- call semantics (Phase 2): block_call op construction + attrs, operand
  dtype/shape errors, static specialization, TraceError outside a trace;
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
| `./op.py` | `BlockOp` class, BLOCK_CALL_OPDEF, `_ensure_ir_opdef` (lazy ir registration) |
| `./registry.py` | `get_block`, BlockOp/impl/portable registries |
| `./rules.py` | bridges into transforms rule registries + decomposition fallback wrappers |
| `./errors.py` | `BlockError` |

## Notes for agents

- transforms NEVER imports block: it sees only `batching_rules`/
  `vjp_rules`/`jvp_rules` entries under `block:<name>` (fallbacks are
  pre-registered by decl). Keep that boundary intact.
- The decorator form requires `@etl.defn` (portable impls are traced graphs).
- `attribute_schema` and `attributes` are the same thing (contract name vs
  objective name) — keep both properties in sync.
- Phase 2 coordination: ir's op-def registration hook, transforms' rule
  callback signatures + `jvp_rules` dict, backends' numpy impl signature,
  persist's JSON encoding of StaticValue payloads.
