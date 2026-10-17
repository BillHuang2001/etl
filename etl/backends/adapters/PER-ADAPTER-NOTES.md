# etl/backends/adapters — per-adapter reference notes

Companion to `./CONTEXT.md` in the same directory.
Everything here is REFERENCE material — measured numbers, machinery detail, probe narratives and host recipes — moved out of `CONTEXT.md` so that file stays readable in full.
Nothing here is a second contract: the binding rules, per-adapter option surfaces, capability matrices, constraints and the numbered Known Issues live in `CONTEXT.md`.
Known-Issue sections here use the SAME numbers as `CONTEXT.md` (`## Known Issue 3`, `## Known Issue 9`, `## Known Issue 10`, `## Known Issue 13`) so cross-references stay stable.

## IREE — host→device staging tiers (byte-level machinery)

Scope: the `_staged_input` machinery in `iree.py`. It serves ONLY (a) host-mode `external_call` boundaries and (b) the explicit-placement provider `upload_tensor`. Plain GPU runs never stage: the run boundary rejects device-mismatched inputs.

Buffer lifetime and bounds:
- Staging buffers are cached per `(input index, shape, dtype)` so no per-call device allocation happens — each cuda SEGMENT executable keeps one staging set per segment.
- The cache is bounded by `_STAGED_INPUT_CACHE_MAX = 64` and cleared wholesale on overflow (correctness unaffected — only the next allocation).
- `upload_tensor` uses a FRESH one-shot dict (no cache) and returns a destination owned by the returned tensor.
- Inputs larger than `_STAGED_INPUT_MAX_BYTES = 16 * 1024 * 1024`, inputs with an unmappable dtype, non-GPU paths, and any fast-path failure fall back to `rt.asdevicearray`.

The two tiers, split at `_PINNED_DIRECT_MAX_BYTES = 4 * 1024` (`4 KB`):
- **Tier 1 — `nbytes <= 4 KB`** (the DE rank-0 key is the motivating case): one persistent `DEVICE_LOCAL | HOST_VISIBLE` (`DEFAULT | MAPPING`) buffer, mapped once, then a single `np.copyto` into the mapped view per call — no `queue_copy`, no semaphore, no wait.
- **Tier 2 — `nbytes > 4 KB`**: one persistent `HOST_LOCAL | DEVICE_VISIBLE` (`TRANSFER_SOURCE | MAPPING`) source buffer plus a per-call async `queue_copy` into a persistent `DEVICE_LOCAL` staging buffer.

Why the cutoff is 4 KB and not larger: host writes into `DEVICE_LOCAL | HOST_VISIBLE` mapped memory are serialized by the driver against device reads, so the pinned tier stalls on larger uploads — measured 0.5–1.2 ms at 160 KB — while the true crossover between the tiers sits around 16–32 KB, and below ~4 KB the pinned tier is both faster than the DMA tier and immune to its ordering race.

**The tier-2 semaphore wait is LOAD-BEARING (do not remove):** the cuda HAL does not reliably stream-order `queue_copy` ahead of invoke dispatches — un-fenced copies exposed ~0.4 % garbage results. Each call therefore waits on a FRESH semaphore (reusing one ABORTS), query-first: `_wait_copy_semaphore` calls `HalSemaphore.query()` and skips the fence wait (~0.02 ms) when the copy is already done. Runtimes without `query` fall back to the plain fence wait.

Each cuda SEGMENT executable also keeps a 4 KB `DEVICE_LOCAL` `_pool_anchor`, which warms the CLASSIC allocator pool for users who opt into `--cuda_async_allocations=false`. Plain `IreeExecutable`s allocate no anchor (the default async allocator needs none).

## IREE — device-resident vs staging-dominated call costs

Same-device loop (state stays on the device; each step's outputs are fed back as the next inputs), iree 3.11 / RTX A6000 class hardware:
- Per-call budget on cuda = invoke floor ~0.09 ms + wrapper ~0.05 ms.
- A fused 4096×50 f32 step (the DE/PSO shape, ~8 iree dispatches) measures ~0.146 ms/call at steady state — at or above torch-eager parity (~0.4–1.0 ms/step).
- The loop performs ZERO `rt.asdevicearray` calls and ZERO host round-trips from the first iteration on.

Anti-pattern for comparison: re-staging the ~800 KB state with an explicit per-step `.to(device)` costs ~1.05–2.8 ms/step — measured ~23× the loop cost (pinned RELATIVELY in `tests/backends/test_iree_same_device_loop.py`).
Reading results back per step is separately expensive: a `.to(cpu)` of an 800 KB buffer costs ~0.13–1.4 ms under contention.

## ROCm/HIP — measured placement and staging costs

MI300X gfx942 / iree-base 3.11.0, best-of-20 with 3 warmups:
- `Tensor.to(Device("rocm", 0))` ≈ 0.052 ms for 4 KB and ≈ 0.43 ms for 4 MB (thread-local cached HAL device + thread-local reusable host source buffer + one `queue_copy`).
- A per-call matmul place+run+readback ≈ 0.28 ms; a device-resident run alone ≈ 0.065 ms.
- `driver.create_device(device_id=1)` on the hip driver costs ~46 ms per call and does not warm up — which is exactly why rocm placement holds a thread-local cached device instead of creating one per call. (Cuda placement keeps the fresh-per-call device; the hip `create_device` cost, not correctness, drives the difference.)

## ROCm/HIP — validated parity inventory

Bit-exact vs the numpy backend on a real MI300X:
- `cond` across the probed shapes.
- `sort`/`argsort` at n = 64/128 with the default `sort_emission="auto"` — no bufferization failure (the cuda-only bug does not appear).
- splitmix64 `uniform`/`randint`.
- threefry2x32 `uniform` through the NATIVE `stablehlo.rng_bit_generator` THREE_FRY path.
- philox4x32_10 `uniform` through the inline expansion.
- host-mode `external_call` dispatch.
- matmul+tanh (1.8e-6 vs an fp64 reference).
f32 `random.normal` differs from numpy by ≤ 4.6e-7 — the documented Box–Muller fast-path budget, matching cuda.

Not covered by tests: rocm × `external_call` segment staging, direct thread-locality of the staging caches (pinned only indirectly), and a real-rocm `upload_tensor` on the `etl.io` worker thread (`prefetch`).

## XLA — device-resident per-call hot path (design + A/B medians)

Design intent: a `run()` on an accelerator executable re-derives nothing that is static for a loaded executable.
- `XlaExecutable` builds ONE `_XlaRunPlan` — input/output arity, per-leaf `(dtype, shape)` validation contracts from the signature, and output metadata from the artifact's recorded static output shapes — then validates through it in one tight loop with the same exception types, messages and order. The CPU host-staging branch is unaffected.
- In the driver, `xla_util._LoadedExecutable` caches `num_outputs` (one FFI per executable) and reuses per-thread ctypes scaffolding — the input buffer-pointer array, output pointer array, `PJRT_ExecuteOptions`, and `PJRT_LoadedExecutable_Execute_Args` — keyed `(executable ptr, num_args, num_outputs)` in `threading.local`, bounded by `_EXECUTE_SCAFFOLDING_MAX`. Thread-local (not a shared cache) keeps concurrent runs of one executable from different threads safe. A call therefore writes only the input pointers and the execute device before the FFI.
- Output buffers are created with the recorded static metadata PREFILLED (`execute(output_meta=...)` → `_Buffer(shape=..., dtype=...)`), replacing two plugin metadata FFIs per output (`PJRT_Buffer_Dimensions` + `PJRT_Buffer_ElementType`, ~3.4 us). Prefilled metadata is never trusted blindly: `_Buffer._verify_metadata()` runs the real queries and raises a loud `core.BackendError` on a declared-vs-actual mismatch; every output is verified on the FIRST device-resident run of an executable (afterwards the static contract is trusted — XLA executable output types are static), and `_Buffer.to_host()` re-verifies on every explicit readback.
- A run's whole output set shares ONE reference on the process-global client and ONE `weakref.finalize` (`_XlaDeviceRunGroup` → `xla_util.release_payloads`) instead of a refcount acquire + finalizer per output. `release_payload` stays the single-buffer wrapper used by `upload_tensor` and external payloads, and the public 4-arg `XlaDevicePayload` constructor keeps its own reference/finalizer.

A/B medians (MI300X VF / ROCm 7.1, interleaved A/B of a re-deriving run path vs the planned path, medians of 60 calls after 20 warmup; shared GPU, so absolute values are inflated):
- 2-in/2-out `(1024,50)` f32 graph: 66–70 us → 40–46 us.
- 8-in/8-out evox-step-shaped graph (five `(1000,50)`/`(1000,)` f32 + `(50,)` f32 + rank-0 i64 + rank-0 i32): 133–143 us → 80–87 us (≈ −31 % … −44 %).

The residual is dominated by the plugin's own `PJRT_LoadedExecutable_Execute` FFI (~32 us for 2 buffers, ~59 us for 8) plus GPU dispatch — not adapter overhead.
Caller-side monitor reads are separate: `.to(cpu)` of a `(1000,50)` f32 output costs ~120 us (rank-0 ~100 us), and the pipeline's `run`/pytree validation above the adapter adds its own fixed cost.
Real-GPU device-resident loop, host-staging comparison: ~0.36 ms/call in the loop vs 4.18–10.14 ms/call with host staging.

## XLA — real-plugin validation narrative (Known Issue 3)

Validation surface:
- A throwaway fake plugin (compiled at test time from `tests/backends/_fake_pjrt_plugin.c`) pins the ABI: `sizeof(PJRT_Api)` == the ctypes 1144 plus field offsets, compile/execute/copy/serialize round-trip, error propagation, the version gate, and the missing-plugin message.
- The REAL `jax_cuda12_pjrt` 0.10.2 CUDA plugin and the REAL `jax-rocm7-pjrt` 0.11.2 ROCm plugin (`platform_name="rocm"`) are validated by the GPU probe `bench_logs/probe_xla_gpu_validation.py`, which passes **26/26**: rank-0 staging, PSO/DE/OpenES-shaped fused steps bit-exact except the documented fp32 budgets, threefry2x32 + philox4x32_10 RNG bit-exact through the INLINE expansions, and a non-dominate-rank while_loop bit-exact.
- Module-level plugin guard: `tests/backends/test_adapter_xla.py` SKIPS at import when no plugin is configured (collects 0 tests in a plugin-absent env); with a plugin present the in-env xla assertion counts are `test_xla_device_resident.py` 9 (fake plugin) and `test_pjrt_ctypes_plugin.py` 10 (C fake plugin, built at test time).

Residual limits: the artifact stays device-agnostic (`target="cpu"`, device chosen at `load()`); native `rng_bit_generator` is off by design (below); CPU staging is naive while the device-resident path is the optimized one (pinned by `tests/backends/test_xla_device_resident.py`, 9 tests).

Why native `rng_bit_generator` is impossible for etl's algorithms on XLA:
1. The exporter's u32-word, key-first state layout fails `PJRT_Client_Compile` against XLA's spec-strict u64-state importer (`Binary op add with different element types: u32[1] and u64[]`).
2. Even with a spec-compliant state, XLA's `THREE_FRY` is threefry2x64 and its `PHILOX` a 2-word philox4x32-10 — different ciphers from etl's threefry2x32 and 4-word philox4x32_10 — so bit-exactness against the numpy reference is unreachable.
The bit-exact inline expansions are therefore the xla default; the per-call `rng_bit_generator` lower option still forces the native path for experiments.

ROCm-plugin command-buffer defect: the rocm plugin SIGSEGVs inside `PJRT_Executable_Execute` for XLA-capture-eligible graphs under the default command-buffer kinds (it bites the evox SO step graphs). The failure is shape/dispatch-count specific, not universal — small 2–8-buffer micro-graphs run with or without the workaround at the same cost (43.1 / 91.2 us without vs 40.4–45.7 / 79.5–86.8 us with; run-to-run noise). The adapter deliberately does not mutate `XLA_FLAGS`.

## XLA — environment recipes

All of these are CALLER-SUPPLIED environment recipes (tests, probes and benchmarks export them themselves); the adapter never rewrites the environment.

CUDA plugin:
- `ETL_PJRT_PLUGIN=/path/to/xla_plugins/jax_cuda12_pjrt/xla_cuda_plugin.so`.
- cuDNN ≥ 9.8 on the loader path — `LD_LIBRARY_PATH` (or `LD_PRELOAD`) must expose it or plugin load fails.
- `XLA_FLAGS` must point at the XLA libdevice data dir.
- `CUDA_VISIBLE_DEVICES` selects the GPU; `ptxas` must be on `PATH` (the adapter adds a ptxas hint to compile failures).

ROCm plugin:
- `ETL_PJRT_PLUGIN=…/jax_plugins/xla_rocm7/xla_rocm_plugin.so`.
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (empty kind list ⇒ HIP command graphs disabled) for the capture-eligible-graph SIGSEGV.
- `HIP_VISIBLE_DEVICES` on a shared multi-GPU box.
- The device kind served is `rocm`/`hip`; `Device("cuda", N)` against this plugin is rejected rather than mislabelled.

Exit behaviour: any process that created a PJRT client ABORTS at interpreter exit AFTER all work completes (SIGABRT 134 / SIGSEGV 139; `PJRT_Client_Destroy` does not help). Read stdout, then `os._exit(0)` for a clean exit status.

Downstream gotcha worth knowing: a backend spec written as `("xla", None)` runs host-staged; on this stack the intended device-resident spec is `("xla", "cuda:0")` / `("xla", "rocm:0")`.

## Known Issue 9 — opt-level and cuda codegen

Both limits are upstream iree 3.11 codegen and reproduce on raw `iree-compile` of etl-dumped MLIR, so they are out of etl's scope.
- eigh: at `--iree-opt-level=O2`/`O3` on cuda the cyclic-Jacobi composition returns WRONG eigenvalues — measured w_err ~1.3e1 vs LAPACK for both the bare eigh and the CMAES fused tell (the tell runs without crashing, just wrong). O0/O1/default are correct on cuda and llvm-cpu is correct at every level.
- NSGA2 tell graph: fails to COMPILE on cuda at O2/O3 (upstream fusion+hoisting on the crowding-distance scatter); default/O0/O1 work.
Practical guidance: leave `opt_level` unset (or ≤ O1) for cuda eigh / NSGA2-tell work; `etl.bench`'s O3 default applies to benchmarks, not to these two graph families.

## Known Issue 10 — IREE version matrix for while-loop graphs

Symptom: `stablehlo.while` with ≥3 loop-carried values plus a `select` in the body (the NSGA2/MOEAD tell shapes) makes `iree-compile` SIGSEGV (exit -11, shell rc 245), surfaced by the adapter as `core.BackendError: iree-compile failed to compile the StableHLO program for the iree backend (targets: …): … Error code: -11`.
Symbolize with `LLVM_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-18`, which shows `SpecializeEncodingsPass::runOnOperation` → `IREE::Stream::AffinityAnalysis::run()` → `DFX::Solver::updateElement` → `getOrCreateElementFor<ValueConsumerAffinityPVS>` → `ValueConsumerAffinityPVS::updateValue` → `Explorer::walkTransitiveUses` → `updateFromUse` → repeat — unbounded recursion through the while back-edge's cyclic use-chain. The pass runs even with `--iree-opt-data-tiling=false`.

Nondeterminism (the reason a single green compile means nothing): measured on 3.11.0rc20260316 on llvm-cpu AND rocm with byte-identical input MLIR, the same `.mlir` file crashes ~85–100 % of runs and compiles ~0–15 % (observed ok-rates 3/20, 1/6, 1/8, 0/20 across batches; single runs flip either way). Any while-graph probe or test must repeat or be treated as flaky.

Levers swept on an NSGA2-shaped 4-carry while graph (each with repeat counts, all still crash most runs):
- `opt_level` O0 / O1 / O2 / O3.
- `--iree-stream-affinity-solver-max-iterations=1` and `=64`.
- `--iree-stream-annotate-input-affinities`.
- `--mlir-disable-threading`.
- `--iree-execution-model=host-only` and `inline-dynamic`.
- `--iree-opt-data-tiling=false`; `--iree-dispatch-creation-data-tiling=false --iree-global-opt-data-tiling=false`.
- The exporter option `while_init_rewrite=False`.
- A larger process stack (`ulimit -s 1GB`, `unlimited`) — the recursion consumes any stack.
Rewriting the body's `select` as cast-based arithmetic masking does NOT dodge it either.

What works:
- Remove the `while`: the same math unrolled at trace time compiles 5/5. A plain 3–6-carry elementwise+reduce+select while (no reshape-broadcast / multi-index body) also compiles reliably.
- A bounded retry loop: 5–13 attempts of ~0.3 s each recovers, and the recovered artifact loads, runs on llvm-cpu and matches the numpy backend bit-exactly (all outputs max_abs_diff 0.0). Retry is a smoke/CI workaround, not a fix.

Per-version verdicts: 3.9.0 compiles the NSGA2/MOEAD-class tell graphs 12/12 clean with full numpy parity (user-pinnable for MO compile work) but its cuda runtime crashes ~9/10 on SO step graphs (Known Issue 11) and it still SIGSEGVs on NSGA3-class graphs. 3.12.0rc fixes the compiler but its runtime is unusable for these graphs. Cross-version loading is not a workaround (the bytecode version is rejected).
Conclusion: MO/while-heavy graphs are upstream-blocked on 3.10/3.11 and out of scope — use the numpy backend or the xla route; the `iree`/`compilers` extras pin `>=3.11.0,<3.12` for both compiler and runtime, which must match.

## Known Issue 13 — rank-0-result reductions on IREE GPU

Diagnosis (MI300X/gfx942, iree 3.11, a 10000×100 evox SO-shaped step `@etl.defn`):
- `concatenate(prev_best (1,), fitness (10000,)) → (10001,)` + `iota` + `etl.argmin` (2-operand reduce, LT/EQ tie-break comparator) + `etl.min` (1-operand reduce) + one i1 reduce (NaN guard) are each lowered by `LLVMGPUBaseLowering` with `workgroup_size = [1,1,1]`, `linalg.fill`, `lowering_config tile_sizes = [[]]` — ONE workgroup, ONE thread, sequential loop.
- Every data-parallel sibling dispatch in the same program gets `LLVMGPUTileAndFuse workgroup_size = [64,1,1]` (`workgroup=[64]`, `thread=[1]`).
- `rocprofv3 --kernel-trace` per-call cost on the real DE step (18 dispatches / 16 executables): argmin-reduce **890.7 µs**, min-reduce **692.4 µs**, i1-reduce **182.2 µs** = **1765 of 1858 µs (95 %)**; ALL 10000×100 data-parallel work (elementwise + gather + reduce-over-dim1) is ~61 µs.
- The cost is independent of pop×dim, so at 1000×50 it hides inside the ~0.13 ms launch floor and at 10000×100 it dominates — the crossover users observe.

Subgraph attribution at 10000×100 (same etl graph, device-resident loop), iree-rocm vs xla-rocm:
- `argmin_1d` 0.9386 vs 0.0618 (15.2×).
- `reduce_min_1d` 0.7608 vs 0.0597 (12.7×).
- Full RNG+eval+select step WITHOUT the monitor reductions: 0.1518 vs 0.1128 (1.35×).
- RNG-only, 1M uniform: 0.0769 vs 0.0642 (1.20×).
⇒ the RNG and the data-parallel math are at parity; the whole gap is the three scalar reductions.

The two-stage rewrite (numerically exact: min/argmin are associative with the emitter's first-occurrence tie-break, and padding with +inf / `n` never wins) measured at etl level:
- `redmin_1d` 0.7608 → `redmin_2stage` (exact factor 73×137) **0.0957**; generic pad-to-`64×157` **0.1067**.
- `argmin_1d` 0.9386 → `argmin_2stage` (row min → row argmin → gather row → combine index) **0.1162**.
- A keepdims reshape to `(n,1)` does NOT help (0.744 ms) — the split must yield ≥2 non-unit factors.
- The same rewrite also helps llvm-cpu (`reduce_min_1d` 0.771 → 0.085), so it is target-agnostic and never a regression.
Shipped form: the `split_reduction` exporter option (default `True`), landing in the StableHLO emitters for `argmin`/`argmax`/`reduce_min`/`reduce_max`/`reduce_*` with a rank-0 result over a large static dim; measured 7.1–8.1× on those ops and 2.7–3.5× end-to-end on population-scale SO steps, e.g. an evox fitness monitor of pop 10000 × dim 100 at 13.53 → 0.746 ms per call on iree/rocm (18.1×). Related: `split_reduction_threshold`/`split_reduction_non_exact` are writer-only knobs, not plumbed through the adapters.

Compiler-flag sweep — no substitute exists. At both the isolated-reduce MLIR level and the real step-graph level, all of these stay within ±2 % of the 0.744 ms isolated reduce:
- `opt_level=O3` (1.818 vs 1.767).
- `--iree-codegen-llvmgpu-use-reduction-vector-distribution` and `-use-vector-distribution` (1.772).
- `--iree-dispatch-creation-split-argmax-threshold=0` and `-split-argmax-tile-size=1024` (1.772).
- `--iree-codegen-enable-default-tuning-specs=false` and `--iree-rocm-waves-per-eu=4` (1.762).
- `--iree-codegen-llvmgpu-vectorize-pipeline`, `--iree-dispatch-creation-enable-split-reduction`, `--iree-llvmgpu-set-workgroup-distribution-along=x`.

RNG is not implicated: on rocm the inline SplitMix64 i64 expansion (`random_uniform` 1M f32, 0.0769 ms) is 1.9× FASTER than the native `stablehlo.rng_bit_generator` THREE_FRY path (0.1433 ms), so i64 emulation is not a bottleneck — the cuda-era "threefry wins at small sizes" result does not carry over to rocm bulk draws.
Host-side per-step overhead is small: the real DE 10000×100 loop measures 2.30 ms/step with `full_fit_history` on and 2.15 ms with it off (the ~0.15 ms delta is the per-step `to(cpu).numpy()` sync), vs 1.94 ms for the bare module ⇒ ~0.2–0.4 ms of wrapper/invoke.

## Why iree's native THREE_FRY is the default (`rng_bit_generator` capability)

`{"threefry2x32"}` is measured, not assumed. On iree 3.11 the native `stablehlo.rng_bit_generator` with `RNG_ALG_THREE_FRY` is bit-exact vs the numpy reference on llvm-cpu, cuda and rocm with the key-first `[k0, k1, ctr0, ctr1]` state layout (verified across 5 keys including `(0,0)` and `(0xFFFFFFFF,0xFFFFFFFF)`, and with odd element counts via truncated flat `tensor<5xui32>` outputs), and it is ~1.6–2.1× faster than the inline expansion overall: cuda randint 2^24 3.84 ms native vs 8.16 ms inline, cuda uniform 4.27 vs 7.64, llvm-cpu uniform 2^24 100 vs 164. Full benchmark matrix: `etl/bench/rng_bench_results.md` (`python3 -m etl.bench.rng_bench`).

## Design decision — cuda async allocator A/B

Why etl forces no `--cuda_async_allocations` value: forcing `false` (the classic pool) measures ~2–3× slower per step on the fused DE/PSO step graphs at 4096×50 — ~2.2–3.5 ms/step vs ~1.16 ms/step with iree's own stream-ordered default — and async-ON wins or ties every other measured cell.
The failure mode a forced classic-pool flag triggers is a full device sync on every per-call result alloc/free: 9.45 ms vs 0.10 ms per call.
Opt-in path for users who need the classic pool: `iree_runtime_args=["--cuda_async_allocations=false"]` or `IREE_PY_RUNTIME_FLAGS`; `upload_tensor` honors the same `ETL_IREE_RUNTIME_ARGS`.

## `sort_emission` — why iree defaults to `auto`

On iree-cuda the multi-operand `stablehlo.sort` the pair emission produces cannot be bufferized once the sorted-axis extent reaches 32; independently, the 1-operand `stablehlo.sort` the VALUES-only emission produces lowers to a single-threaded O(n²) kernel.
The count composition is bit-exact vs numpy on cuda and ~120–134× faster for those argsorts, is bit-exact on llvm-cpu as well, and emits no `stablehlo.sort` at all — hence `IreeBackend.default_sort_emission = "auto"` (= count iff the sorted extent ≥ 32; the shared `CompilerBackend` default stays `"pair"`, so xla/tvm never see the change).
Failure boundary measured on the pair emission: sorted extents 32/48/64/96/128 fail, ≤ 24 pass; the 1-operand `sort` and llvm-cpu are unaffected.

## `eigh_algorithm` — why `jacobi_batched` is the default

`"jacobi_batched"` applies all mutually disjoint rotations of one odd-even round-robin round per `stablehlo.while` iteration instead of one (p,q) rotation per iteration: measured 65–74× faster at n=100 and 4–5.7× at n=10 (n=100: xla/rocm 1181 ms → 18.2 ms, iree/llvm-cpu 9532 ms → 128 ms on the MI300X host), and it avoids an iree-llvm-cpu compile crash at n ∈ {4, 8, 12, 16}.
`"jacobi"` stays selectable and emits the single-pair-per-iteration text.

`dot_precision` context: XLA/ROCm lowers fp32 `stablehlo.dot_general` to a TF32-class (10-bit-mantissa) tensor-core path by default — a 1024² fp32 matmul deviates from an fp64 reference by max-abs 0.1228. `dot_precision="HIGHEST"` recovers true fp32 (2.03e-4); `"HIGH"` does not disable TF32 on this XLA version, and neither a plugin flag nor an `XLA_FLAGS` route exists.

## Diagnostic recipes

iree scheduling/module diagnostics:
- `--iree-scheduling-dump-statistics-file=- --iree-scheduling-dump-statistics-format=json` — dispatch/executable/copy counts.
- `--iree-hal-dump-executable-configurations-to=-` — per-dispatch `workgroup_size` + `lowering_config` (this is what exposes the single-workgroup reductions of Known Issue 13).
- `iree-benchmark-module --device=hip … --benchmark_repetitions=20` — module-only time; rank-0 scalars pass as `--input=i64=1`.
- `rocprofv3 --kernel-trace -f csv -o <out>` — per-kernel µs.

iree-compile crash triage: `LLVM_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-18` (Known Issue 10's stack).
xla: the probe harness `bench_logs/probe_xla_gpu_validation.py` is the end-to-end validation entry point (26 checks) and documents the env recipes it needs.

## Validation inventory

- The `etl.bench` conformance/benchmark matrix (98 examples) exercises these adapters on numpy/cpu, iree/llvm-cpu, iree/rocm and xla/rocm; numbers in `etl/bench/rocm_mi300x_results.md` and `etl/bench/de_vs_torch_results.md`.
- Adapter-specific numbers referenced from `CONTEXT.md` (staging tiers, hot path, version matrix, reduction diagnosis) are reproduced above with their measurement conditions.
- The pytest note for this repo: `-q` (the configured `addopts`) suppresses the final "N passed" line — use `-v` or `-o addopts=` when a count is needed.
