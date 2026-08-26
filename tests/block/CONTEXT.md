# tests/block — custom operations (`etl.block`) test suite

## Intent

pytest suite for the `etl.block` subsystem (sibling `../etl/block/` — read-only from here; read its CONTEXT.md for the contract being tested): block declaration (factory + decorator forms), call semantics (block_call IR construction, static specialization, operand errors), backend impl registration + execution, portable decomposition inlining, batching/derivative rule registration + fallbacks, and the BlockError hierarchy.

## Structure

| File | Area | Tests |
|---|---|---|
| `test_declaration.py` | factory/decorator forms, BlockOp attrs, attribute-schema normalization, effects/policy validation, duplicate registration, get_block | 30 |
| `test_call.py` | block_call op construction, static-arg payload encoding, operand binding rules, dtype/shape errors, TraceError paths, multi-output | 30 |
| `test_impl.py` | `impl("numpy")` registration, execution via etl.evaluate, static-args kwargs payload, multi-output impls, missing-impl BackendError | 7 |
| `test_errors.py` | ETLError hierarchy, lower-time verification errors (unknown block, output-spec mismatch, non-tensor portable return) | 21 |
| `test_portable.py` | portable inlining numerics, block_call removal at lower, spec derivation, non-defn portable errors | 9 |
| `test_rules.py` | batching/jvp/vjp rule registration + fallbacks (vmap/grad/jvp), TransformError paths, non-callable rules | 13 (6 green, 7 BUG-marked) |

## Constraints

- CPU only, small shapes (batch ≤ 16, dims ≤ 8), fast (<2s per file, whole dir ~0.8s).
- Never modify `../etl/` — bugs found are kept as failing tests with a `# BUG(etl): ...` comment above the assertion; the parent (root) delegates fixes.
- Run from repo root: `python3 -m pytest -q tests/block` (root conftest.py adds the repo to sys.path).

## Known Issues

Tests that fail because of REAL `etl` bugs (each carries a `# BUG(etl)` comment; do NOT "fix" by weakening the assertion — the parent coordinates fixes):

1. **vmap/vectorize never dispatches `block_call` to the `block:<name>` rule namespace.** `etl/transforms/batching.py::_rewrite_op` calls `require_batching_rule(op.name)` (raw `"block_call"`), but `etl.block` registers batching rules under `block:<name>` (per both CONTEXT.md files). vmap on ANY block raises `TransformError("no batching rule for op 'block_call'")` even with explicit rules/portable fallbacks. Affects `test_vmap_via_portable_fallback`, `test_explicit_batching_rule_registers_and_wins_over_unsupported_policy`, `test_explicit_rule_overrides_portable_decomposition`, `test_vmap_without_rule_raises_transform_error_naming_the_block`. (Contrast: `autodiff._rule_name` maps `block:<name>` correctly — AD lookup is fine.)
2. **Portable VJP fallback returns all-zero gradients.** `etl/block/rules.py::_portable_vjp_rule` seeds incoming cotangents on the block_call's result ids, but the inlined decomposition produces NEW values — the local reverse sweep never sees the seed. grad-via-portable returns zeros instead of the derivative. Affects `test_grad_via_portable_decomposition`.
3. **jvp is never derived from a vjp rule.** `autodiff.require_jvp_rule` consults only `jvp_rules`, though `etl/block/CONTEXT.md`/`op.py` bind that transforms derives jvp from vjp when no jvp rule exists. jvp of a portable-only block raises `TransformError("no JVP rule for op 'block:rule_portgrad'")`. Affects `test_jvp_derived_from_portable_vjp_fallback`.
4. **Batching policy is not honored by transforms.** The policy table (elementwise/map_over_batch safe without a rule) has no implementation in `etl/transforms` (which never imports block) and nothing pre-registers a pass-through rule. Affects `test_elementwise_policy_passes_batch_dims_through`.

Minimal repros (run from repo root):

```python
import numpy as np, etl
# Bug 1
b = etl.block("b1", inputs=[etl.TensorSpec((4,), etl.float32)],
              outputs=[etl.TensorSpec((4,), etl.float32)])
@b.batching_rule
def r(op, operands, axes): ...
@etl.defn
def f(x): return b(x)
etl.vmap(f)(etl.TensorSpec((8, 4), etl.float32))   # TransformError: no batching rule for op 'block_call'
# Bug 2
@etl.block
@etl.defn
def swish(x: etl.TensorSpec((4,), etl.float32)) -> etl.TensorSpec((4,), etl.float32):
    return etl.sigmoid(x) * x
@etl.defn
def loss(x): return etl.sum(swish(x))
g = etl.grad(loss)(etl.TensorSpec((4,), etl.float32))
etl.run(etl.load(etl.compile(etl.lower(g))), np.array([0.5, 1., 2., 3.], dtype=np.float32))  # all zeros
# Bug 3: etl.jvp(loss, ...) on the same swish block -> TransformError "no JVP rule for op 'block:swish'"
# Bug 4
ew = etl.block("ew", inputs=[etl.TensorSpec((4,), etl.float32)],
               outputs=[etl.TensorSpec((4,), etl.float32)], batching="elementwise")
@ew.impl("numpy")
def _ew(x): return x
@etl.defn
def fe(x): return ew(x)
etl.vmap(fe)(etl.TensorSpec((8, 4), etl.float32))  # TransformError instead of pass-through
```

## Notes for agents

- **Global registries**: block names and `block:<name>` rule keys live in process-wide registries. Every declared block uses a unique per-file prefix (`decl_`, `call_`, `impl_`, `err_`, `port_`, `rule_`) to avoid cross-file collisions when pytest runs the whole dir in one process.
- **No `from __future__ import annotations`** in test modules that declare blocks via the decorator form — with it, `inspect.signature` returns unevaluated string annotations and spec derivation breaks.
- Declare blocks at MODULE scope (declaration is registration); rule/impl decorators return the function.
- **Numpy impl contract**: at run time the interpreter calls `impl(*[t.numpy() for t in operands], **static_args)` where `static_args` values are JSON-safe payload dicts `{"kind": ..., "value": ...}` (NOT decoded Python values). Tests that need runnable rule-only blocks register a trivial `@blk.impl("numpy")`.
- vmap cannot map axis 0 of a rank-0 input in v1 — use rank-1 specs batched to `(B, D)`.
- `etl.evaluate`/`etl.run` return `etl.Tensor` leaves → compare via `.numpy()` + `np.allclose`.
- Graph inspection: `graph.module.main.entry_block.ops` (each `Op` has `.name`, `.attributes`, `.operands`, `.results`, `.effect`); lowered payload: `etl.ir.deserialize_module(lowered.payload)`.
- `etl.block` at package top level is the FACTORY function (shadows the submodule) — import internals via `from etl.block import ...` / `from etl.block.errors import BlockError`.
