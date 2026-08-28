# etl/backends/stablehlo — StableHLO MLIR export utility (v1: export-only)

## Intent

Turn verified EvoXIR (`etl.ir`) programs into **StableHLO MLIR text** so external compilers can take over compilation: `iree-compile model.mlir -o model.vmfb`. This module is an **export utility ONLY** in v1 — it is NOT a compilable `Backend`, it is NOT registered in the backend registry, and it never lowers/compiles/loads/runs anything. The compiler adapters that consume this export — `"iree"`, `"xla"` (via PJRT), `"tvm"` — are IMPLEMENTED in `../adapters/` (see `../adapters/CONTEXT.md`).

Binding contracts: `../../../CONTEXT.md` (staging pipeline, error strategy, non-goals) and `../../CONTEXT.md` (backends package; the "StableHLO exporter — v1 scope" section defines the mapping, deferred list, and type map).

## API Surface

`etl.backends.stablehlo.export(graph_or_module) -> str` — the ONLY public name (`__all__ = ["export"]`; nothing else is re-exported).

- Accepts a `trace.Graph` (duck-typed — uses its `.module`) or an `etl.ir.Module` directly; anything else ⇒ `TypeError`.
- Calls `module.verify()` FIRST — failures surface as `core.VerificationError`; never emits MLIR from invalid IR.
- Returns StableHLO MLIR text (`str`) — compiler input for external tools.
- Unsupported or deferred op ⇒ `core.BackendError` NAMING THE OP, message suggests decomposition or a future adapter. Never silently skips an op, never partial output.
- Export-only: not registered as a Backend; performs no lower/compile/load/run.

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

**Dynamic-dim deferrals (v1, validated through iree):** every rejection names the op, the shape, the offending dims, and contains "dynamic" — raised at export/`lower()` time, never invalid MLIR: `reshape` with ANY dynamic dim (incl. the keepdims reshapes inside the reduce/reduce_mean emitters), `conv` with any dynamic dim in x/w/result, `slice` / `pad` with dynamic dims (iree-compile ACCEPTS the MLIR but the runtime ABORTs at every concrete size), `reduce_mean` reducing over a dynamic dim (element count not statically known), and `dot` batch structure that cannot be emitted (no shape source for the required dynamic broadcast / unprovable symbolic batch merge).

Type map (`DTYPE_MAP`, keys are numpy dtype objects): float16→`f16`, float32→`f32`, float64→`f64`, int8→`i8`, int16→`i16`, int32→`i32`, int64→`i64`, uint8→`ui8`, uint16→`ui16`, uint32→`ui32`, uint64→`ui64`, bool→`i1`, complex64→`complex<f32>`, complex128→`complex<f64>`.

Helpers (trivial, implemented): `lookup_mapping(op_name)` (first hit across tables, search order ELEMENTWISE→SHAPE→CONSTANT→CONTROL_FLOW→COLLECTIVE→COMPARISON→DECOMPOSITIONS; KeyError if absent), `status(op_name)` → `"v1"|"decompose"|"deferred"` (unmapped names count as deferred), `is_supported(op_name)`, `mlir_dtype(dtype)` (normalizes via `numpy.dtype`).

## Design decisions

- **Verify before export**: `export()` runs `module.verify()` first so emission never works from invalid IR.
- **Batched dot emission matches numpy `matmul`** (`_emit_dot`, ~line 1041 — 5-path dispatch): the frontend guarantees rank >= 2 (ops); batch dims broadcast per `infer_dot` (equal or size-1). Before emitting `stablehlo.dot_general` each operand's batch dims are aligned to the matmul-broadcast target batch: matched batch structure emits directly (byte-identical to the pre-fix output); a plain-matrix rhs with fully static shapes emits a non-batched dot_general (lhs free dims = batch); static targets use `stablehlo.broadcast_in_dim`; symbolic targets use `dynamic_broadcast_in_dim` whose output-dimensions vector is built per-dim (`get_dimension_size` on the batch-dims source + static constants — the other operand is NOT a valid full-shape source when its non-batch dims differ); size-1 batch dims broadcast/squeeze to the other side; an unprovable symbolic merge raises `BackendError` containing "dynamic". Result shape is always exactly the declared `infer_dot` result. Rank-1 operands (IR-level only — `infer_dot` allows them) emit valid non-batched dot_general (v·v → scalar, M·v, v·M), verified through iree.
- **v1 dynamic-dims policy (per-op, validated through iree)** — ALLOWED: elementwise (incl. the `_equalize_operand`/`_scalar_constant_for` dynamic-broadcast paths), cast, select, compare, bitwise/logical, reductions over dynamic dims (no keepdims reshape), `reduce_mean` with dynamic NON-reduced dims (dynamic-broadcast divisor), transpose, concatenate, if/while, constant, dot_general with matched batch structure, and the dynamic broadcast path. REJECTED with a clear `core.BackendError` (names op + shape + dims + "dynamic"): reshape (any dynamic dim), conv (any dynamic dim), slice/pad (dynamic dims — iree runtime ABORT), `reduce_mean` over dynamic reduced dims, broadcast with a dynamic result and no full-shape source, dot with unemittable batch structure. Elementwise dynamic shapes (e.g. `x*2+1` on `TensorSpec((dim("B"),), f32)`) compile+run correctly on iree and tvm at multiple sizes — pinned by the adapter contract tests. iree/tvm declare `dynamic_shapes=True` but support is per-op partial — the exporter is the gate.
- **Symbolic dims render as `?`**: int dims render literally; `Dim`/`DimExpr`/`None` render as `?` (e.g. `tensor<?xNxf32>`); rank is always concrete in etl so emitted rank is exact.
- **Error style**: deferred or unmapped op ⇒ `core.BackendError` naming the op and suggesting decomposition or a future adapter — never silent skip, never partial output.
- **Data/code separation**: mapping tables live in `ops.py` (auditable against the StableHLO spec without touching emission code); `writer.py` contains only emission logic.
- **Mnemonic verification**: the exact StableHLO mnemonics must be verified against the StableHLO spec at implementation time (note recorded in `ops.py`).
- **Program order emission**: ops emit in block order — program order IS effect order (write/read/collective/callback semantics preserved).

## Constraints (binding)

- Top-level imports restricted to `etl.core` and `etl.ir`; NEVER import `etl.pipeline` (cycle). `trace.Graph` is duck-typed via `.module` (no `etl.trace` import). numpy is allowed (DTYPE_MAP keys). `writer.py` uses TYPE_CHECKING imports for `etl.ir` annotations.
- Export-only: never registers with the backend registry; no lower/compile/load/run.
- Files < ~1000 lines (`writer.py` is the known exception — the single emission file, ~1800 lines; splitting is deferred, see Notes for agents); `ops.py` is data-only.
- CPU-neutral: MLIR text only, no device interaction.

## Test strategy

`../../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root):
- Golden-text exports for the v1 table: elementwise, comparisons (direction attr), reduce, dot/conv, if/while, collectives.
- Symbolic-dims rendering: `tensor<?xNxf32>` etc.
- Decomposition emission: square/relu/gelu/stop_gradient/reduce_mean.
- Deferred ops (gather/scatter/scan/runtime_call/block_call/rank/world_size) ⇒ `BackendError` naming the op; unknown op ⇒ same.
- Dynamic-dims rejection contract (pending — to be added by root): symbolic reshape/conv/slice/pad and dynamic-reduced-axis reduce_mean ⇒ `BackendError` naming op/dims/"dynamic"; positive: reduce_mean over a static axis with dynamic non-reduced dims exports; batched-dot goldens for rank-3@rank-2 (aligned batching dims after broadcast), rhs-higher-rank, size-1 batch squeeze, matched multi-batch (byte-identical), symbolic-batch dynamic-broadcast emission, and unprovable symbolic merge ⇒ `BackendError`; adapter-level: softmax-style symbolic reshape graph must raise from `etl.lower(..., backend='iree'|'tvm')` before any compiler invocation.
- `verify()` failure surfaces `VerificationError`; non-Graph/non-Module input ⇒ `TypeError`.
- CPU only, pytest, numpy-only deps.

## Routing table

| Path | Area |
|---|---|
| `./ops.py` | v1 mapping tables (data) + lookup/status helpers — the auditable mapping source of truth |
| `./writer.py` | StableHLO MLIR text emission (`Writer`, ~1800 lines incl. the dynamic-broadcast emission path, batched-dot batch alignment, and the per-op dynamic-dims validation; consumes `ops.py` data only) |
| `./__init__.py` | `export()` — the only public entry point |

## Notes for agents

- **`broadcast` name collision — resolved**: the IR op names do NOT collide — the dist collective's IR op name is `broadcast_collective` (COLLECTIVE_MAP → `stablehlo.collective_broadcast`, emitted in `writer.py`'s `_emit_collective`, ~line 821), while the shape op keeps the IR name `broadcast` (SHAPE_MAP → `stablehlo.broadcast_in_dim`, emitted in `_emit_broadcast`). Dispatch is purely by IR op name — no effect-kind disambiguation is needed at emission time. Golden-tested in `../../../tests/backends/test_stablehlo.py` (collective export asserts `stablehlo.collective_broadcast`).
- `../../` = `etl/backends/`, `../../../` = repo root (the tests path above is correct from this node).
