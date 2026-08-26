# etl/backends/stablehlo — StableHLO MLIR export utility (v1: export-only)

## Intent

Turn verified EvoXIR (`etl.ir`) programs into **StableHLO MLIR text** so external compilers can take over compilation: `iree-compile model.mlir -o model.vmfb`. This module is an **export utility ONLY** in v1 — it is NOT a compilable `Backend`, it is NOT registered in the backend registry, and it never lowers/compiles/loads/runs anything. IREE/XLA/TVM adapters are documented future integration points, not implemented here.

Binding contracts: `../../../CONTEXT.md` (staging pipeline, error strategy, non-goals) and `../../CONTEXT.md` (backends package; the "StableHLO exporter — v1 scope" section defines the mapping, deferred list, and type map).

## API Surface

`etl.backends.stablehlo.export(graph_or_module) -> str` — the ONLY public name (`__all__ = ["export"]`; nothing else is re-exported).

- Accepts a `trace.Graph` (duck-typed — uses its `.module`) or an `etl.ir.Module` directly; anything else ⇒ `TypeError`.
- Calls `module.verify()` FIRST — failures surface as `core.VerificationError`; never emits MLIR from invalid IR.
- Returns StableHLO MLIR text (`str`) — compiler input for external tools.
- Unsupported or deferred op ⇒ `core.BackendError` NAMING THE OP, message suggests decomposition or a future adapter. Never silently skips an op, never partial output.
- Export-only: not registered as a Backend; performs no lower/compile/load/run.
- **Architecture phase**: `export()` and all `Writer` methods are stubs raising `NotImplementedError`; only `ops.py` (mapping data + trivial helpers) is implemented. Behavior is implemented by `subagent_manager`.

## v1 mapping (data lives in `./ops.py`; binding table from `../../CONTEXT.md`)

Direct mnemonics (etl op → StableHLO, emitted with `stablehlo.` prefix):

| Table | etl op(s) | StableHLO |
|---|---|---|
| ELEMENTWISE_MAP | add/subtract/multiply/divide/power/remainder/maximum/minimum | `stablehlo.add/subtract/multiply/divide/power/remainder/maximum/minimum` |
| ELEMENTWISE_MAP | abs/negate/sqrt/sign | `stablehlo.abs/negate/sqrt/sign` |
| ELEMENTWISE_MAP | exp/log/log1p/sin/cos/tan/tanh/sigmoid/erf | `stablehlo.exponential/log/log_plus_one/sine/cosine/tan/tanh/logistic/erf` |
| ELEMENTWISE_MAP | bitwise_and/or/xor; logical_and/or/not | `stablehlo.and/or/xor/not` |
| ELEMENTWISE_MAP | cast | `stablehlo.convert` |
| COMPARISON_MAP | equal/not_equal/less/less_equal/greater/greater_equal | `stablehlo.compare` + `comparison_direction` attr (EQ/NE/LT/LE/GT/GE) |
| SHAPE_MAP | select/broadcast/reshape/transpose/slice/concatenate/pad | `stablehlo.select/broadcast_in_dim/reshape/transpose/slice/concatenate/pad` |
| SHAPE_MAP | reduce_sum/max/min/prod | `stablehlo.reduce` (reduction kind comes from the op's attrs) |
| SHAPE_MAP | argmax/argmin | `stablehlo.argmax/argmin` |
| SHAPE_MAP | dot/conv | `stablehlo.dot_general` / `stablehlo.convolution` |
| CONSTANT_MAP | constant | `stablehlo.constant` (dense elements attr) |
| CONTROL_FLOW_MAP | cond/while_loop | `stablehlo.if` / `stablehlo.while` |
| COLLECTIVE_MAP | all_reduce/all_gather/reduce_scatter/all_to_all/broadcast/collective_permute | `stablehlo.all_reduce/all_gather/reduce_scatter/all_to_all/collective_broadcast/collective_permute` |

Decompositions (`DECOMPOSITIONS` — writer emits ordinary sub-ops, no direct mnemonic):
- `square` → `multiply(x, x)`
- `relu` → `maximum(x, 0)`
- `gelu` → erf-based: `0.5*x*(1+erf(x/sqrt(2)))`
- `stop_gradient` → identity passthrough (emit operand directly)
- `reduce_mean` → reduce-sum then divide by element count

Deferred in v1 (`DEFERRED_OPS` ⇒ `core.BackendError` naming the op): `gather`, `scatter`, `scan`, `runtime_call`, `block_call`, `rank`, `world_size` (dist graph scalars), complex-number elementwise beyond cast.

Type map (`DTYPE_MAP`, keys are numpy dtype objects): float16→`f16`, float32→`f32`, float64→`f64`, int8→`i8`, int16→`i16`, int32→`i32`, int64→`i64`, uint8→`ui8`, uint16→`ui16`, uint32→`ui32`, uint64→`ui64`, bool→`i1`, complex64→`complex<f32>`, complex128→`complex<f64>`.

Helpers (trivial, implemented): `lookup_mapping(op_name)` (first hit across tables, search order ELEMENTWISE→SHAPE→CONSTANT→CONTROL_FLOW→COLLECTIVE→COMPARISON→DECOMPOSITIONS; KeyError if absent), `status(op_name)` → `"v1"|"decompose"|"deferred"` (unmapped names count as deferred), `is_supported(op_name)`, `mlir_dtype(dtype)` (normalizes via `numpy.dtype`).

## Design decisions

- **Verify before export**: `export()` runs `module.verify()` first so emission never works from invalid IR.
- **Symbolic dims render as `?`**: int dims render literally; `Dim`/`DimExpr`/`None` render as `?` (e.g. `tensor<?xNxf32>`); rank is always concrete in etl so emitted rank is exact.
- **Error style**: deferred or unmapped op ⇒ `core.BackendError` naming the op and suggesting decomposition or a future adapter — never silent skip, never partial output.
- **Data/code separation**: mapping tables live in `ops.py` (auditable against the StableHLO spec without touching emission code); `writer.py` contains only emission logic.
- **Mnemonic verification**: the exact StableHLO mnemonics must be verified against the StableHLO spec at implementation time (note recorded in `ops.py`).
- **Program order emission**: ops emit in block order — program order IS effect order (write/read/collective/callback semantics preserved).

## Constraints (binding)

- Top-level imports restricted to `etl.core` and `etl.ir`; NEVER import `etl.pipeline` (cycle). `trace.Graph` is duck-typed via `.module` (no `etl.trace` import). numpy is allowed (DTYPE_MAP keys). `writer.py` uses TYPE_CHECKING imports for `etl.ir` annotations.
- Export-only: never registers with the backend registry; no lower/compile/load/run.
- Files < ~1000 lines; `writer.py` skeleton < ~300 lines; `ops.py` is data-only.
- CPU-neutral: MLIR text only, no device interaction.

## Test strategy

`../../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root):
- Golden-text exports for the v1 table: elementwise, comparisons (direction attr), reduce, dot/conv, if/while, collectives.
- Symbolic-dims rendering: `tensor<?xNxf32>` etc.
- Decomposition emission: square/relu/gelu/stop_gradient/reduce_mean.
- Deferred ops (gather/scatter/scan/runtime_call/block_call/rank/world_size) ⇒ `BackendError` naming the op; unknown op ⇒ same.
- `verify()` failure surfaces `VerificationError`; non-Graph/non-Module input ⇒ `TypeError`.
- CPU only, pytest, numpy-only deps.

## Routing table

| Path | Area |
|---|---|
| `./ops.py` | v1 mapping tables (data) + lookup/status helpers — the auditable mapping source of truth |
| `./writer.py` | `Writer` skeleton — StableHLO MLIR text emission (stubbed) |
| `./__init__.py` | `export()` — the only public entry point (stubbed) |

## Notes for agents

- **Architecture phase**: behavioral bodies raise `NotImplementedError`; only `ops.py` data + trivial helpers are implemented. Implementation is delegated to `subagent_manager` at this node by the parent orchestrator.
- **`broadcast` name collision**: the etl op name `broadcast` appears in BOTH `SHAPE_MAP` (`stablehlo.broadcast_in_dim` — data movement) and `COLLECTIVE_MAP` (`stablehlo.collective_broadcast` — dist collective). `lookup_mapping("broadcast")` resolves to the SHAPE_MAP entry; the writer must disambiguate by the op's effect kind (collective effect ⇒ collective mnemonic). Confirm the IR op name used by `dist.broadcast` with the `dist` owner during implementation.
- `../../` = `etl/backends/`, `../../../` = repo root (the tests path above is correct from this node).
