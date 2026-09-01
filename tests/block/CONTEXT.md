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
| `test_rules.py` | batching/jvp/vjp rule registration + fallbacks (vmap/grad/jvp), TransformError paths, non-callable rules | 13 |
| `test_portable_vjp_regression.py` | regression for the fixed portable-decomposition vjp fallback (commit 1a88219: post-inline seeding): correct NONZERO input cotangents via grad/vjp on portable-only blocks (two-input `x*w`, matmul, nested chain), non-ones cotangents, all-zero short-circuit, mixed multi-output cotangents, jvp derived from the vjp fallback, output-count guard; + 1 BUG(etl) (constant op inside a portable fails the local sweep) | 11 |

## Constraints

- CPU only, small shapes (batch ≤ 16, dims ≤ 8), fast (<2s per file, whole dir ~0.8s).
- Never modify `../etl/` — bugs found are kept as failing tests with a `# BUG(etl): ...` comment above the assertion; the parent (root) delegates fixes.
- Run from repo root: `python3 -m pytest -q tests/block` (root conftest.py adds the repo to sys.path).

## Notes for agents

- **Global registries**: block names and `block:<name>` rule keys live in process-wide registries. Every declared block uses a unique per-file prefix (`decl_`, `call_`, `impl_`, `err_`, `port_`, `rule_`) to avoid cross-file collisions when pytest runs the whole dir in one process.
- **No `from __future__ import annotations`** in test modules that declare blocks via the decorator form — with it, `inspect.signature` returns unevaluated string annotations and spec derivation breaks.
- Declare blocks at MODULE scope (declaration is registration); rule/impl decorators return the function.
- **Numpy impl contract**: at run time the interpreter calls `impl(*[t.numpy() for t in operands], **static_args)` where `static_args` values are JSON-safe payload dicts `{"kind": ..., "value": ...}` (NOT decoded Python values). Tests that need runnable rule-only blocks register a trivial `@blk.impl("numpy")`.
- vmap cannot map axis 0 of a rank-0 input in v1 — use rank-1 specs batched to `(B, D)`.
- `etl.evaluate`/`etl.run` return `etl.Tensor` leaves → compare via `.numpy()` + `np.allclose`.
- Graph inspection: `graph.module.main.entry_block.ops` (each `Op` has `.name`, `.attributes`, `.operands`, `.results`, `.effect`); lowered payload: `etl.ir.deserialize_module(lowered.payload)`.
- `etl.block` at package top level is the FACTORY function (shadows the submodule) — import internals via `from etl.block import ...` / `from etl.block.errors import BlockError`.
