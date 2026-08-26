# tests/ir — EvoXIR IR-layer test suite

## Intent

pytest suite validating the `etl.ir` package (sibling — the contract lives in
`../etl/ir/CONTEXT.md`, read it before touching these tests). These tests are
the executable spec of the IR layer: the op registry (canonical 75-op v1 set),
`verify()` invariants, serialization round-trips/integrity, `pretty_print`
golden output, the `Builder`, shape-inference hooks, and the SSA data model.
728 tests, all fast (whole file set runs <2s).

## Test files

| File | Covers |
|---|---|
| `test_registry.py` | 75-op canonical set (incl. `return` terminator and `broadcast_collective` distinct from shape-op `broadcast`), per-op category/effect/arity/attr-schema/shape_fn, registry API (opdef/has_opdef/op_names/all_opdefs/register_opdef, KeyError/ValueError paths) |
| `test_verify.py` | valid modules + one test per violation class from `verify()`'s docstring: module/function/region/op/SSA/use-bookkeeping/shape_fn-agreement/attr-schema violations, location-annotated messages, TypeError on non-Module |
| `test_serialize.py` | payload schema, multi-feature round-trips (symbolic dims, constants as base64 npy, all effect kinds, if/while regions, multi-function + `call`), original id preservation, fast-forwarded counters, tamper detection (sha256), version/format rejection (PersistenceError) |
| `test_pretty_print.py` | exact-string goldens: SSA headers, %N renumbering, attribute rendering (sorted keys, `?` for None, ndarray summaries), locations, multi-arg header wrapping, nested `^bbN` region labels |
| `test_builder.py` | Builder contracts: eager arity/attr/region validation, result-type resolution order (explicit → shape_fn → op-specific), attribute normalization (dtype→name string, lists→tuples, nullable-int rule), insertion-point stack, set_terminator rules, per-module id counters |
| `test_inference.py` | all 23 shape-inference hooks: static + symbolic (`Dim`/`DimExpr`) + `None` dims, numpy dtype promotion/reduction rules, ShapeError/ValueError cases, world-group collectives (group_size=None → None dims) |
| `test_model.py` | structural attrs of Module/Function/Region/Block/Op/Value/Use/ValueType/Location/effects, use-def chains, RAUW (`replace_all_uses_with`), Dim/DimExpr interop |

## Known etl bugs

None found — every test passes against the current implementation; the documented
contracts in `../etl/ir/CONTEXT.md` (including the Known Issues section) hold.
Known-Issue behaviors are encoded as tests with `# CURRENT contract` comments:
- `divide` uses `infer_elementwise_binary` (int/int stays int) and unary math ops
  are dtype-preserving — the ops-layer true-division/float64-promotion contract
  is NOT yet implemented in the IR registry.
- `cumsum` uses `infer_identity` (bool stays bool).
- `pad.padding_config` (ATTR_NESTED_INTS) rejects bare int entries via the
  Builder, though `infer_pad` the hook accepts them.
- `slice.limit_indices` is schema-required (the None-limit branch of
  `infer_slice` is unreachable via the Builder).

## Notes for agents

- **Registry is global — never leave test-only OpDefs registered.** Two tests
  in `test_verify.py` need fake ops (`_test_halt`, `_test_no_rule`); they are
  registered by the `test_only_opdefs` fixture, which restores the registry to
  its exact pre-fixture state in a `finally` (pops from `etl.ir.op_defs._REGISTRY`).
  Import-time registration broke `test_registry.py`'s canonical-75 assertions
  when the two files ran in one pytest process.
- Test modules must NOT import each other's helpers; keep fixtures/helpers local
  to each file (avoids cross-file coupling and ordering hazards).
- The dominant pattern for `test_verify.py` violations: build a valid module via
  the `Builder`, mutate ONE thing in place (bypasses the Builder's eager checks),
  assert `verify` fails with a specific message, restore, assert it passes again.
- The package is READ-ONLY from here: if a test finds a genuine contract
  violation in `etl`, keep the failing test with a `# BUG(etl): ...` comment and
  escalate the fix to the parent (do not weaken the test).
- Run: `python3 -m pytest -q tests/ir` from the repo root (root `conftest.py`
  puts the repo root on sys.path). CPU only, no GPU/network.
