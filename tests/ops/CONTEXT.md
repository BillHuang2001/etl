# tests/ops — frontend tensor-operation contract tests

## Intent

pytest suite asserting the op contracts in `etl/ops/CONTEXT.md` and `etl/ops/constant.py`: each test traces a small function via `etl.trace`, inspects the built IR (op name, effect, attributes, result `ValueType`s, SymbolicTensor fields), evaluates numerically via `etl.evaluate`/`etl.run` (numpy backend), and checks the documented error paths (`TraceError`/`TypeError`, no silent fallback).

## Files

| File | Covers |
|---|---|
| `test_constant.py` | `etl.constant`: pure `constant` IR op, dtype/shape preservation, trace-time data snapshot (source mutation after tracing does not leak), `ETL_LARGE_CONSTANT_BYTES` warning threshold (default 1 MiB; patch the module attribute directly — `etl.ops.constant` as an attribute is the re-exported function, so resolve the module via `importlib.import_module("etl.ops.constant")`), TraceError paths, registration-hook wiring |
| `test_runtime_call.py` | `etl.runtime_call`: effect `callback`, callback carried as a registered-id STRING (resolves via the ops-level registry), `result_specs` as a tuple of `ir.ValueType`s, single/multi-output execution, scalar-operand promotion to 0-d constants, TypeError/TraceError paths. Backend rejection policy is NOT tested here (see `../backends/`) |

`conftest.py` provides `trace_fn`, `ops_of(graph, name=None)` and `run_numpy(fn, *args)` (see its docstring for the stable IR-inspection facts).

## Notes for agents

- Keep tests small/fast (CPU only, shapes ≤ ~256×256) and per-op focused; use `pytest.warns`/`recwarn` for the warning threshold and `pytest.raises(..., match=...)` for error paths.
- A real etl bug that contradicts the documented contract must NOT be papered over: keep the failing test with `# BUG(etl): <one-line description>` and report a minimal repro.
