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
| ELEMENTWISE_MAP | exp/log/log1p/sin/cos/tan/tanh/sigmoid | `stablehlo.exponential/log/log_plus_one/sine/cosine/tan/tanh/logistic` |
| ELEMENTWISE_MAP | bitwise_and/or/xor; logical_and/or/not | `stablehlo.and/or/xor/not` |
| ELEMENTWISE_MAP | bitwise_left_shift / bitwise_right_shift | `stablehlo.shift_left` / `shift_right_arithmetic` (the writer picks `shift_right_logical` for unsigned results) |
| ELEMENTWISE_MAP | cast | `stablehlo.convert` |
| COMPARISON_MAP | equal/not_equal/less/less_equal/greater/greater_equal | `stablehlo.compare` + `comparison_direction` attr (EQ/NE/LT/LE/GT/GE) |
| SHAPE_MAP | select/broadcast/reshape/transpose/slice/concatenate/pad | `stablehlo.select/broadcast_in_dim/reshape/transpose/slice/concatenate/pad` |
| SHAPE_MAP | reduce_sum/max/min/prod | `stablehlo.reduce` (reduction kind comes from the op's attrs) |
| SHAPE_MAP | dot/conv | `stablehlo.dot_general` / `stablehlo.convolution` |
| CONSTANT_MAP | constant | `stablehlo.constant` (dense elements attr) |
| CONTROL_FLOW_MAP | cond/while_loop | `stablehlo.if` / `stablehlo.while` |
| COLLECTIVE_MAP | all_reduce/all_gather/reduce_scatter/all_to_all/broadcast/collective_permute | `stablehlo.all_reduce/all_gather/reduce_scatter/all_to_all/collective_broadcast/collective_permute` |

Dedicated emitters (`SPECIAL_EMITTERS` — multi-op StableHLO compositions with dedicated writer routines; `status()` reports "v1"):
- `gather` → `stablehlo.gather` (single-axis, exact numpy `take` semantics; full-rank index reshaping)
- `scatter` → `stablehlo.scatter` with an `update_computation` region (numpy `put_along_axis` semantics; replacement, not accumulation)
- `sort`/`argsort` → `stablehlo.sort` with an LT comparator region (argsort sorts a (value, iota) pair; descending = `stablehlo.reverse` after — the numpy composition)
- `argmin`/`argmax` → two-operand `stablehlo.reduce` over (value, iota) with an index tie-break comparator (first occurrence on ties, matching `np.argmax`/`np.argmin`)
- `tile` → reshape + `broadcast_in_dim` + reshape decomposition
- `eigh` → an unrolled cyclic-Jacobi symmetric eigensolver (8 sweeps of the (p, q) rotation pairs — slice/iota/compare/select/elementwise, no while-loops) + a stable pair-sort for the ASCENDING eigenvalue order + a gather column reorder of V. StableHLO 1.0 removed `stablehlo.eigh`/`qr`/`svd` (iree 3.11 ships only `cholesky`), so the LAPACK-based numpy kernel has no mnemonic counterpart; dtype rule mirrors the numpy kernel (int/bool → f64; f32/f64 pass; complex/f16 → explicit `BackendError`); dynamic dims → explicit `BackendError`. Parity is fp32-tolerance, NOT bit-exact (measured 3x3: w 9.5e-7; 10x10: w 1.5e-5 / reconstruction 1.7e-5; f64 near-exact 2.7e-15; int32 exact at f64 floor) — tests check ascending w + `A ≈ V diag(w) Vᵀ` + `VᵀV ≈ I`, never elementwise v (eigenvectors of degenerate eigenspaces are basis-dependent). 10x10 build+run ~22-24 s (the unrolled composition grows with n²).
- `diag` → rank-2: flatten reshape + constant-index single-axis gather (the main diagonal); rank-1: iota-EQ mask + select (the diagonal matrix). Dtype preserved both directions incl. complex (the mask compares iotas, never data); EXACT vs numpy. No StableHLO diag op exists.
- `random_key_mix`/`random_uniform`/`random_normal`/`random_randint`/`random_permutation` → inline SplitMix64 i64 subgraph expansion (`./random_export.py`; bit-exact vs the numpy kernels by two's-complement equivalence — uniform/randint/permutation EXACT, normal 1 ulp; NEVER `stablehlo.rng`, whose implementation-defined algorithm would break the same-key ⇒ same-values determinism contract)
- **`scan` has no IR OpDef**: `etl.scan` desugars at trace time into while + gather + scatter (all v1 now), so scan graphs compile and run on iree-llvm-cpu.

Decompositions (`DECOMPOSITIONS` — writer emits ordinary sub-ops, no direct mnemonic):
- `square` → `multiply(x, x)`
- `relu` → `maximum(x, 0)`
- `stop_gradient` → identity passthrough (emit operand directly)
- `reduce_mean` → reduce-sum then divide by element count

Deferred in v1 (`DEFERRED_OPS` ⇒ `core.BackendError` naming the op): `runtime_call`, `block_call`, `rank`, `world_size` (dist graph scalars), `erf`/`gelu` (no `stablehlo.erf` exists — CHLO only; gelu's erf-based decomposition needs erf, so it is deferred together with it, no silent approximation), `diagonal`, `cumsum`, `solve`, the linalg factorizations `cholesky`/`qr`/`svd` (StableHLO counterparts exist but are not wired in v1 — `eigh` IS v1 via the unrolled cyclic-Jacobi composition in `_emit_eigh`) and `matrix_rank`/`matrix_exp` (need SVD-cutoff / Padé decompositions), `random_multinomial` (cumulative-search decomposition not wired), the 16 `sparse_*`/`dense_dot_sparse` ops (numpy-backend-only; densify via `etl.sparse.to_dense`), complex-number elementwise beyond cast. Unmapped op names (`call`, `tril`, `triu`, …) also count as deferred via `status()`.

**Dynamic-dim deferrals (v1, validated through iree):** every rejection names the op, the shape, the offending dims, and contains "dynamic" — raised at export/`lower()` time, never invalid MLIR: `reshape` with ANY dynamic dim (incl. the keepdims reshapes inside the reduce/reduce_mean emitters), `conv` with any dynamic dim in x/w/result, `slice` / `pad` with dynamic dims (iree-compile ACCEPTS the MLIR but the runtime ABORTs at every concrete size), `reduce_mean` reducing over a dynamic dim (element count not statically known), `eigh`/`diag` with any dynamic dim (the compositions need static shapes for the unrolled rotation constants / constant index offsets), and `dot` batch structure that cannot be emitted (no shape source for the required dynamic broadcast / unprovable symbolic batch merge).

Type map (`DTYPE_MAP`, keys are numpy dtype objects): float16→`f16`, float32→`f32`, float64→`f64`, int8→`i8`, int16→`i16`, int32→`i32`, int64→`i64`, uint8→`ui8`, uint16→`ui16`, uint32→`ui32`, uint64→`ui64`, bool→`i1`, complex64→`complex<f32>`, complex128→`complex<f64>`.

Helpers (trivial, implemented): `lookup_mapping(op_name)` (first hit across tables, search order ELEMENTWISE→SHAPE→CONSTANT→CONTROL_FLOW→COLLECTIVE→COMPARISON→DECOMPOSITIONS; KeyError if absent), `status(op_name)` → `"v1"|"decompose"|"deferred"` (unmapped names count as deferred), `is_supported(op_name)`, `mlir_dtype(dtype)` (normalizes via `numpy.dtype`).

## Design decisions

- **Verify before export**: `export()` runs `module.verify()` first so emission never works from invalid IR.
- **Batched dot emission matches numpy `matmul`** (`_emit_dot`, ~line 1041 — 5-path dispatch): the frontend guarantees rank >= 2 (ops); batch dims broadcast per `infer_dot` (equal or size-1). Before emitting `stablehlo.dot_general` each operand's batch dims are aligned to the matmul-broadcast target batch: matched batch structure emits directly (byte-identical to the pre-fix output); a plain-matrix rhs with fully static shapes emits a non-batched dot_general (lhs free dims = batch); static targets use `stablehlo.broadcast_in_dim`; symbolic targets use `dynamic_broadcast_in_dim` whose output-dimensions vector is built per-dim (`get_dimension_size` on the batch-dims source + static constants — the other operand is NOT a valid full-shape source when its non-batch dims differ); a provably all-size-1 rhs batch is squeezed to a plain matrix for a non-batched dot_general ONLY when BOTH operands are fully static (same dynamic_reshape legalization gate as the plain-matrix fast path — dynamic shapes take the batched dynamic-broadcast path instead); an unprovable symbolic merge raises `BackendError` containing "dynamic". Result shape is always exactly the declared `infer_dot` result. Rank-1 operands (IR-level only — `infer_dot` allows them) emit valid non-batched dot_general (v·v → scalar, M·v, v·M), verified through iree.
- **v1 dynamic-dims policy (per-op, validated through iree)** — ALLOWED: elementwise (incl. the `_equalize_operand`/`_scalar_constant_for` dynamic-broadcast paths), cast, select, compare, bitwise/logical, reductions over dynamic dims (no keepdims reshape), `reduce_mean` with dynamic NON-reduced dims (dynamic-broadcast divisor), transpose, concatenate, if/while, constant, dot_general with matched batch structure, and the dynamic broadcast path. REJECTED with a clear `core.BackendError` (names op + shape + dims + "dynamic"): reshape (any dynamic dim), conv (any dynamic dim), slice/pad (dynamic dims — iree runtime ABORT), `reduce_mean` over dynamic reduced dims, broadcast with a dynamic result and no full-shape source, dot with unemittable batch structure. Elementwise dynamic shapes (e.g. `x*2+1` on `TensorSpec((dim("B"),), f32)`) compile+run correctly on iree and tvm at multiple sizes — pinned by the adapter contract tests. iree/tvm declare `dynamic_shapes=True` but support is per-op partial — the exporter is the gate.
- **Symbolic dims render as `?`**: int dims render literally; `Dim`/`DimExpr`/`None` render as `?` (e.g. `tensor<?xNxf32>`); rank is always concrete in etl so emitted rank is exact.
- **Error style**: deferred or unmapped op ⇒ `core.BackendError` naming the op and suggesting decomposition or a future adapter — never silent skip, never partial output.
- **Data/code separation**: mapping tables live in `ops.py` (auditable against the StableHLO spec without touching emission code); `writer.py` contains only emission logic.
- **Mnemonic verification**: the exact StableHLO mnemonics must be verified against the StableHLO spec at implementation time (note recorded in `ops.py`).
- **Program order emission**: ops emit in block order — program order IS effect order (write/read/collective/callback semantics preserved).

## Constraints (binding)

- Top-level imports restricted to `etl.core` and `etl.ir`; NEVER import `etl.pipeline` (cycle). `trace.Graph` is duck-typed via `.module` (no `etl.trace` import). numpy is allowed (DTYPE_MAP keys). `writer.py` uses TYPE_CHECKING imports for `etl.ir` annotations.
- Export-only: never registers with the backend registry; no lower/compile/load/run.
- Files < ~1000 lines (`writer.py` is the known exception — the single emission file, ~2700 lines; splitting is deferred, see Notes for agents); `ops.py` is data-only.
- CPU-neutral: MLIR text only, no device interaction.

## Test strategy

`../../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root):
- Golden-text exports for the v1 table: elementwise, comparisons (direction attr), reduce, dot/conv, if/while, collectives.
- Symbolic-dims rendering: `tensor<?xNxf32>` etc.
- Decomposition emission: square/relu/stop_gradient/reduce_mean.
- Deferred ops (runtime_call/block_call/rank/world_size/erf/gelu/diagonal/cumsum/solve/cholesky/qr/matrix_rank/svd/matrix_exp/random_multinomial + the 16 sparse ops) ⇒ `BackendError` naming the op; unknown op ⇒ same.
- `../../../tests/backends/test_iree_emitters_parity.py` (41 tests, iree-llvm-cpu): parity vs numpy for gather/sort/argsort/argmin/argmax/tile/scatter, bit-exact random draws (uniform/randint/permutation EXACT, normal 1 ulp), scan-through-while graphs (scan desugars to while+gather+scatter at trace time), and the shift primitives (4 parity cases incl. u64-logical via i64-bits+masking); plus 5 iree-cuda smoke tests on `Device("cuda", 5)` (gather/sort/random_uniform/tile/scatter) — GPU-guarded (`pytest.skip` without the cuda HAL driver/GPU).
- `../../../tests/backends/test_iree_eigh_diag_parity.py` (8 tests, iree-llvm-cpu): diag EXACT vs numpy (rank-1 / square / rect), eigh contract checks (ascending w, `A ≈ V diag(w) Vᵀ`, `VᵀV ≈ I`) + w vs numpy within fp32 tolerance — 3x3 f32, 10x10 f32 (CMAES size, ~22-24 s build), batched (2,3,3) f32, f64 3x3 (1e-12), int32→f64 upcast (1e-12); never elementwise v.
- Dynamic-dims rejection contract (pending — to be added by root): symbolic reshape/conv/slice/pad and dynamic-reduced-axis reduce_mean ⇒ `BackendError` naming op/dims/"dynamic"; positive: reduce_mean over a static axis with dynamic non-reduced dims exports; batched-dot goldens for rank-3@rank-2 (aligned batching dims after broadcast), rhs-higher-rank, size-1 batch squeeze, matched multi-batch (byte-identical), symbolic-batch dynamic-broadcast emission, and unprovable symbolic merge ⇒ `BackendError`; adapter-level: softmax-style symbolic reshape graph must raise from `etl.lower(..., backend='iree'|'tvm')` before any compiler invocation.
- `verify()` failure surfaces `VerificationError`; non-Graph/non-Module input ⇒ `TypeError`.
- CPU only, pytest, numpy-only deps.

## Routing table

| Path | Area |
|---|---|
| `./ops.py` | v1 mapping tables (data) + lookup/status helpers — the auditable mapping source of truth |
| `./writer.py` | StableHLO MLIR text emission (`Writer`, ~2700 lines incl. the dynamic-broadcast emission path, batched-dot batch alignment, the per-op dynamic-dims validation, and the gather/scatter/sort/argsort/arg_reduce/tile emitter routines; consumes `ops.py` data only) |
| `./__init__.py` | `export()` — the only public entry point |
| `./random_export.py` | SplitMix64 random-op expansion (inline i64 subgraphs for the 5 v1 random ops) |

## Notes for agents

- **`broadcast` name collision — resolved**: the IR op names do NOT collide — the dist collective's IR op name is `broadcast_collective` (COLLECTIVE_MAP → `stablehlo.collective_broadcast`, emitted in `writer.py`'s `_emit_collective`, ~line 821), while the shape op keeps the IR name `broadcast` (SHAPE_MAP → `stablehlo.broadcast_in_dim`, emitted in `_emit_broadcast`). Dispatch is purely by IR op name — no effect-kind disambiguation is needed at emission time. Golden-tested in `../../../tests/backends/test_stablehlo.py` (collective export asserts `stablehlo.collective_broadcast`).
- `../../` = `etl/backends/`, `../../../` = repo root (the tests path above is correct from this node).
- `_emit_dot`'s 5-path dispatch (rank-1 operands / matched batch / non-batched fast paths — plain-matrix rhs AND the size-1-batch squeeze — each gated on FULLY static shapes: iree 3.11 cannot legalize the `dynamic_reshape` its import inserts for non-batched dynamic dot_general, so dynamic shapes fall through to the batched dynamic-broadcast path, which legalizes fine / static or dynamic batch broadcast / unprovable symbolic merge ⇒ BackendError). All `_dot*` helpers live between `_emit_dot` and `_emit_conv`.
- Dynamic-dims rejection helper: `Writer._reject_dynamic_dims(op, shape, what, op_name=None)` (~line 652) — used by the reshape/conv/slice/pad emitters and the reduce keepdims paths; message template always contains "dynamic" and the offending dims. `slice`/`pad` were added to the reject list EMPIRICALLY: iree-compile accepts the MLIR but the runtime ABORTs (hal.fence.await) at every concrete size.
- `writer.py` is ~2700 lines and legitimately long: it is the single emission file (mapping data lives in `ops.py`; splitting the writer itself is a possible future refactor, not a correctness issue).
- **`stablehlo.sort` tie order is NOT guaranteed to match numpy's stable argsort**: the (value, iota) comparator matches numpy for ascending order, but the descending path (`stablehlo.reverse` after sorting) reverses ties too — the parity tests pin the descending-argsort tie-break as `np.argsort(x)[::-1]`.
- **iree's declared dtypes exclude uint64** (the function-signature capability check): u64 values must be constructed INSIDE the graph via i64 bits + masking (as the shift u64-logical parity test does) — u64 tensors cannot be graph inputs/outputs on iree.
- **iree-cuda while-loop fragility is shape-specific (upstream compiler/runtime)**: `while_fib` segfaults and `while_cond_combo` hits a `hal.device.queue.dealloca` ABORT on iree-cuda (both pass on llvm-cpu), while an ISOLATED NSGA2-`non_dominate_rank`-shaped while_loop (3 carries incl. an input-derived vector carry, `etl.select` on carried vectors) compiles and runs bit-exact on BOTH llvm-cpu and iree-cuda (maxdiff 0 / 1.1e-16). However, the REAL NSGA2 environmental-selection defn graphs (nsga2.py `_make_tell` / `_make_init_tell` — `non_dominate_rank`'s while_loop combined with gather/scatter/sort) make `iree-compile` itself SEGFAULT (CompilerToolError "Error code: -11", SIGSEGV) on BOTH llvm-cpu and cuda — measured INTERMITTENT but graph-specific: the ACTUAL `non_dominate_rank` while_loop (input-derived (n,n) bool dominance matrix + intra-body reduction over it + bool→int32 cast chains) crashes ALONE ~9/10 on llvm-cpu and ~3/5 on cuda with identical input+flags (the hand-shaped proxy loop with `etl.select` on carried vectors — what earlier CONTEXT called "the isolated loop" — is NOT the real graph and does compile fine); the pass dump stops after `VerifyLoweringToAsyncPass` (stream/async lowering), `--mlir-disable-threading` does not help, and every tested pop/dim config crashes (pop 32/128, n_obj 2/3) — an upstream iree 3.11 compiler bug, NOT a clean deferred-op `BackendError`. Deterministic companions (both targets): `'util.global.load' op undefined global: @__hoisted_*` for constant tensors used inside while bodies, and — cuda only — `failed to bufferize op` on 2-operand `stablehlo.sort` (the argsort emission) whenever the sorted axis ≥ 32. NSGA2 GPU is therefore blocked on the tell/init_tell graphs (see `../adapters/CONTEXT.md` Known Issues 8f–h); the etl-side workaround for benchmarking is a hybrid design (GPU evaluate + numpy-backend tell) or re-expressing the ranking without while.
