# tests/ops — frontend tensor-operation contract tests

## Intent

pytest suite asserting the op contracts in `../../etl/ops/CONTEXT.md` and `../../etl/CONTEXT.md` (siblings — read-only). Each test traces a small function via `etl.trace` (plain fn or `@etl.defn`), inspects the built IR (op name, effect, attributes, result `ValueType`s, SymbolicTensor dtype/shape, locations), evaluates numerically via `etl.evaluate` (numpy backend), and checks the documented error paths (`TraceError`/`ShapeError`/`DTypeError`/`TypeError`, no silent fallback).

`conftest.py` provides shared helpers (no per-file duplication): `trace_fn(fn, *specs)`, `ops_of(graph, name=None)`, `run_numpy(fn, *args)`. Its docstring holds the stable IR-inspection facts (`graph.module.functions[0].region.blocks[0].ops`; `Op` fields `.name/.operands/.results/.attributes/.effect/.location`).

## Files

| File | Covers |
|---|---|
| `test_elementwise.py` | 37 elementwise + comparison ops: broadcasting (incl. symbolic `DimExpr` rules, static ShapeError), dtype inference (np.result_type; NEP-50 weak scalar promotion; unary math int→float64; true division; comparisons→bool; logical/bitwise dtype guards; cast), IR construction (op name, effect pure, location capture), scalar promotion to 0-d constants, numerics vs numpy |
| `test_linalg.py` | dot (batched matmul, batch broadcast, k=1 broadcast op, ShapeError), conv (DimExpr formula + numpy reference numerics, channels_last sugar), solve, tril/triu, cumsum (bool→int64, axis/reverse), argmax/argmin (int64), stop_gradient (identity, pure) |
| `test_reductions.py` | reduce_sum/max/min/mean/prod + sugar sum/max/min/mean/prod (same IR): axes None/int/tuple/negative, keepdims, dtype rules (mean→float64, sum/prod bool→int64), numerics vs numpy, axis errors |
| `test_structure.py` | broadcast, reshape (DimExpr + one `-1`), transpose, slice (Nx-style start/lengths/strides), gather, scatter (last-write-wins, value cast), concatenate (DimExpr axis sum, result_type promotion), pad ((lo,hi) config), select (3-way broadcast) |
| `test_getitem.py` | `SymbolicTensor.__getitem__`: int → gather, contiguous slices → slice op, strided → gather (arange), tuples, negative indices, full-axis slices, static OOB → ShapeError; non-static keys (None/newaxis, ellipsis, masks, symbolic, numpy scalars) → TraceError |
| `test_constant.py` | `etl.constant`: pure Constant op, dtype/shape preserved, trace-time DATA SNAPSHOT (post-trace mutation does not leak), `ETL_LARGE_CONSTANT_BYTES` warning (default 1048576; patch the module attribute — resolve via `importlib.import_module("etl.ops.constant")` since `etl.ops.constant` is the re-exported function), TraceError paths, core hook wiring |
| `test_runtime_call.py` | `etl.runtime_call`: effect `callback`, callback attr = registered-id STRING, `result_specs` = tuple of `ir.ValueType`s equal to result types, single/multi-output numpy-backend execution, scalar-operand promotion, TypeError/TraceError paths. Backend-rejection policy lives in `../backends/` |
| `test_errors.py` | per-op table (pins the pre-batch 67-op set — STALE: `etl.ops.__all__` is 82 names, so `test_op_table_covers_all_public_ops` FAILS until `OP_CALLS` is extended with the 15 new ops; test update pending at the ops owner): no-trace → TraceError, concrete-Tensor operand → TraceError (three-option message), wrong dtype → DTypeError, static broadcast mismatch → ShapeError, both-scalars → TraceError, unsupported operand kinds → TypeError |
| `test_dunders.py` | dunder ≡ op-function transparency: identical `ir.pretty_print` (with `ETL_DISABLE_LOCATIONS=1`) for `+ - * / ** < <= > >= == @`, unary `-`, reflected scalar-left forms; `bool(x)`/`x != y` raise TraceError |

## Notes for agents

- Keep tests small/fast (CPU only, shapes ≤ ~256×256); `pytest.raises(..., match=...)`, `pytest.warns`/`recwarn`, heavy parametrization.
- A real etl bug contradicting the documented contract must NOT be papered over: keep the test plain-failing with `# BUG(etl): <one-line description>` and report a minimal repro; the parent delegates etl fixes (this dir is read-only w.r.t. etl).
- Contract divergences that turned out to be *documented* behavior are tested as such (e.g. `ops.dot` is rank-≥2 only; numpy-scalar and `Dim`-valued getitem keys are non-static → TraceError; float scalar + int tensor → float64 per NEP-50/result_type, NOT float32).
- Locations: for pretty_print equality, set `ETL_DISABLE_LOCATIONS=1`; for location assertions, user-frame calls in plain fns and `@etl.defn` both capture real `file:line` in `Op.location`.
