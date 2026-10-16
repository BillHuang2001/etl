---
name: etl-compiler-adapter-recipes
description: Procedural run recipes, host<->device staging machinery details, version matrices, and measured performance/validation references for the etl compiler adapters (iree/xla/tvm) under etl/backends/adapters. Use when configuring a real GPU run (XLA PJRT plugin, iree cuda/rocm), debugging staging / hot-path performance, or needing the full validation narratives. Companion to etl/backends/adapters/CONTEXT.md.
---

# etl compiler-adapter recipes & reference

Long procedural recipes and measured references for the iree/xla/tvm adapters — kept out of
`etl/backends/adapters/CONTEXT.md` to stay under the per-file size limit. The CONTEXT.md holds the
current-state contract; this skill holds the bulky how-to and the numbers behind it.

## XLA run environment (real PJRT plugins)

### CUDA plugin (`jax_cuda12_pjrt` 0.10.2, `xla_cuda_plugin.so`)

```
ETL_PJRT_PLUGIN=<venv>/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
LD_PRELOAD=/…/cudnn-xla/lib/libcudnn.so.9     # set at process start (see below)
XLA_FLAGS=--xla_gpu_cuda_data_dir=<dir with nvvm/libdevice/libdevice.10.bc>
CUDA_VISIBLE_DEVICES=<free gpu>               # adapter executes on the client's addressable device 0
PATH=<...>                                     # must contain ptxas
```

- **cuDNN:** the plugin is compiled against cuDNN 9.8.0 but its `DT_RPATH` points at the venv's pip cuDNN
  9.1.0; RPATH beats `LD_LIBRARY_PATH` for the plugin's own `dlopen("libcudnn.so.9")`, so only `LD_PRELOAD`
  (set at process start) beats RPATH. With the venv's 9.1.0 every compile fails `RET_CHECK dnn_support != nullptr`.
  (`LD_LIBRARY_PATH=<cu12 cuDNN 9.10.2.21 lib dir>` also works, but `LD_PRELOAD` is the reliable form on 0.10.2.)
- **libdevice:** graphs using libdevice routines (e.g. `random.normal`'s Box–Muller log/sqrt/cos) fail
  `PJRT_Client_Compile` without `XLA_FLAGS=--xla_gpu_cuda_data_dir=…` when no system CUDA toolkit is installed;
  the adapter appends a libdevice hint to such errors.
- Artifact stays device-agnostic (`target="cpu"`); the device is chosen at `load()` (`Device("cuda", N)`
  resolves the client's addressable device N with explicit out-of-range/unknown-kind errors).

### ROCm plugin (`jax-rocm7-pjrt` 0.11.2, `jax_plugins/xla_rocm7/xla_rocm_plugin.so`)

```
ETL_PJRT_PLUGIN=<venv>/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
```
The wheel resolves libhsa/librocm via its own `DT_RPATH` — no `LD_LIBRARY_PATH`; no `XLA_FLAGS`/libdevice needed.
`platform_name="rocm"`; `Device("rocm", N)` / `Device("hip", N)` build+load+run device-resident, `Device("cuda", N)`
is honestly rejected. Host-staged path (`device` omitted) unchanged (fp32 parity ~1e-7).

### Plugin exit-time abort (defect — do not fix)

Any process that created a PJRT client ABORTS at interpreter exit (`malloc_consolidate(): invalid chunk size` /
`double free or corruption`, SIGABRT 134, sometimes SIGSEGV 139) AFTER all work succeeds; explicit
`PJRT_Client_Destroy` does not prevent it. Probes on these plugins must read STDOUT rather than the process
exit code — `os._exit(0)` after `sys.stdout.flush()` exits cleanly.

## XLA staging / device-resident reference

### Rank-0 staging (0.10.2 plugin — implemented)

`buffer_from_host` on a rank-0 array stages with `dims=NULL` + `num_dims=0` (the header-contract form) —
verified on the real 0.10.2 plugin: shape `()` round-trips through `PJRT_Buffer_Dimensions` for ALL dtypes.
Root cause of a former `(1,)` symptom: `np.ascontiguousarray` has a hardcoded `ndmin=1` (numpy#5300) and
silently promoted 0-d host arrays to `(1,)` BEFORE the rank-0 branch was reached. Fix: staging uses `np.asarray`
(0-d-preserving) + contiguity only for `ndim ≥ 1`, and passes `dims=NULL` for rank 0. RE-PROBED matrix on
0.10.2: NULL+0 → `()`, zeroed non-NULL `(0,)`+0 → `()`, garbage non-NULL+0 → `()`, `(1,)`+1 → `(1,)`
(the hypothesized "non-NULL dims + num_dims=0 → (1,)" quirk does NOT reproduce on 0.10.2). The fake test plugin
still emulates the quirk as a driver-regression guard (`test_xla_device_resident.py::test_rank0_staging_scalar_shape`).
evox states carry rank-0 leaves (`key` () i64, `generation` () i32) — this fix is REQUIRED for the evox
device-resident loop.

### Device-resident loop measurements (0.10.2 CUDA plugin, RTX A6000)

- 10-leaf PSO-shaped state (key/pop/vel/fit/pbest/pbest_fit/gbest/gbest_fit/gen/mon, 3×0.2 MB leaves):
  **~0.36 ms/call at BOTH 1000×50 and 10000×100** (destroy-of-previous-outputs dominates; per-call is
  size-insensitive up to 4 MB buffers). A 4-output PSO step 0.19–0.24 ms/call; plain `add` (1000,50)
  0.11–0.12 ms/call; DE-shaped 4096×50 (RNG + 3 gathers + select) 0.165 ms/call (torch DE harness:
  0.405 ms/step — etl at/below torch). First call one-time ~2.9–4.3 ms.
- **Same state with `device=None` (host-staging path): 4.18–4.50 ms/call @1000×50 and 10.14 ms/call
  @10000×100** — per-buffer `BufferFromHostBuffer` med 0.167 ms (0.2 MB) / 0.689 ms (4 MB) / 0.134–0.140 ms
  (≤4 KB); `ToHostBuffer` 0.187 ms (0.2 MB) / 0.349 ms (0.82 MB).
- Per-call budget (device-resident, 10-output state): blocking `PJRT_LoadedExecutable_Execute`
  0.087–0.121 ms (NULL-events ⇒ the plugin waits in-call — the irreducible serial cost); payload wrap
  ~6–8 µs/output; destroy of previous outputs ~0.03 ms (deferred form; immediate destroy of 4 fresh outputs
  ~0.106 ms); pipeline validation ~0.02–0.09 ms.
- Async execute (`device_complete_events` set) dispatch 0.061 + `PJRT_Event_Await` 0.053 ms; a lag-2
  deferred-destroy async loop is CORRECT (outputs bit-match) but NOT faster in steady state
  (0.219 vs 0.152 ms raw / 0.360 etl.run) — for a sequential same-device loop the await is irreducible;
  do NOT re-investigate.
- Hazards: an input buffer MUST outlive its consuming kernel — destroying it right after an async dispatch
  SEGFAULTs the plugin (exit 139; destroy only after awaiting the consuming dispatch's event).
  `PJRT_Event_IsReady` is in `_API_FIELD_NAMES` but has NO ctypes struct/CFUNCTYPE (unusable —
  AttributeError/TypeError). `PJRT_ExecuteOptions` in the vendored revision has `non_donatable_input_indices`
  but NO `untied_allocation` field.
- **Contention:** with NULL-events blocking execute, another tenant's kernel queue on the same GPU is exposed
  per call (under a torch matmul hammer: p90 18.4 ms, max 862 ms vs quiet med 0.112 ms).
- **Deployment gotcha:** the evox harness (`evox/benchmarks/etl_vs_torch/bench_common.py::etl_backend_spec`)
  maps `etl-xla-cuda` → `("xla", None)` behind a stale note claiming the adapter rejects non-cpu devices at
  load — false since the device-resident implementation. `device=None` re-stages all state leaves from host
  every step ⇒ exactly the 4–16 ms/step class recorded in its BENCHMARK_RESULTS.md. Fix: `("xla", "cuda:0")`
  — expect 12–28× (≈0.36 ms/step).

## iree host→device staging machinery (detail)

Serves ONLY host-mode external-kernel boundaries and the placement provider — plain cuda runs never stage.
Tier split at `_PINNED_DIRECT_MAX_BYTES = 4 * 1024` (measured crossover ~16–32 KB).

- **Pinned direct-write tier (nbytes ≤ 4 KB):** a persistent `DEVICE_LOCAL | HOST_VISIBLE` (`DEFAULT | MAPPING`)
  buffer mapped once; each call is one `np.copyto` into the mapped view (~0.02–0.03 ms idle); dispatches read it
  directly — no `queue_copy`, no semaphore, no wait, no post-invoke degradation (measured staging-after-invoke
  median 0.041/0.071 ms vs DMA 0.216/0.480 ms, p90 0.044 vs 0.54+). Cache entry `("pin", (device_array, mapped_view))`.
- **DMA tier (nbytes > 4 KB):** `np.copyto` into the mapped view of a persistent `HOST_LOCAL | DEVICE_VISIBLE`
  (`TRANSFER_SOURCE | MAPPING`) source buffer, then a per-call async `queue_copy` (DMA) into a persistent
  `DEVICE_LOCAL` staging buffer. The copy's semaphore wait is LOAD-BEARING: the iree cuda HAL does NOT reliably
  stream-order `queue_copy` ahead of invoke dispatches (mixed device+host alternating stress exposed ~0.4%
  garbage results with an un-fenced copy). Each call waits on a FRESH semaphore (reusing a waited
  event-semaphore ABORTS the runtime — `event_semaphore.c:354` on second use). The wait is QUERY-FIRST
  (`_wait_copy_semaphore`): `HalSemaphore.query()` (cheap host read, no kernel sync) skips the heavyweight
  `HalFence.wait()` (~0.02 ms fixed) when the copy already completed; the full fence wait runs when pending
  (query-then-wait is race-free — semaphore values are monotonic; runtimes without `query` fall back to the
  plain wait). Hit rate at 4096×50 (800 KB) ~0% (PCIe copy 0.1–0.3 ms ≫ the ~20 µs to reach the wait); fires
  for SMALL DMA-tier inputs (≲ 32 KB) on short queues, costs ~0.4 µs on a miss. Cache entry
  `("dma", (hl_buf, hl_view, staging_da))`.
- Fallbacks to `rt.asdevicearray` (unchanged semantics): oversized inputs (>16 MB), unmappable dtypes,
  non-cuda/rocm fast paths, cache overflow (bounded 64 entries).
- **`_pool_anchor`:** each cuda SEGMENT executable retains a 4 KB DEVICE_LOCAL buffer (classic-pool warmth for
  users who opt into `--cuda_async_allocations=false`); plain `IreeExecutable`s allocate none.
- **DMA-tier cost right after an invoke:** ~0.16–0.48 ms (the fresh-semaphore wait absorbs the queue's
  leftover latency); the wait cannot be dropped — which is why ≤4 KB inputs use the pinned tier instead.
- Fallback path (`rt.asdevicearray`, no staging cache): stable at ~0.9–1.0 ms.

## iree ROCm fast host→device staging path (detail)

`_staged_input` picks a per-kind fast path from the NORMALIZED `device_kind`: the cuda two-tier pinned/DMA
machinery, or the rocm host-local source + `queue_copy` machinery (both waits LOAD-BEARING). Segment
executables are built with the normalized kind, so a rocm host-mode external-call boundary stages
host↔device through the rocm fast path and warns once. `rt.asdevicearray` is the fallback for
oversized/unmappable inputs and any fast-path failure.

rocm placement (`upload_tensor`):
- reuses ONE THREAD-LOCAL HAL device per `(index, runtime args)` — `_acquire_cached_rocm_device`.
- stages through a THREAD-LOCAL reusable `HOST_LOCAL|DEVICE_VISIBLE` (`TRANSFER_SOURCE|MAPPING`) source
  buffer keyed `(shape, dtype)` — `_rocm_source_buffer`, identity-checked owning device, bounded
  `_ROCM_SOURCE_CACHE_MAX = 16`; `np.copyto` into the source view, then a per-call `queue_copy` into the
  destination with a FRESH semaphore + query-first wait (`_rocm_staged_input`).
- Thread-locality is REQUIRED, not an optimization: HAL devices are thread-affine (see the thread-affinity
  gotcha below; `etl/io.py::_bootstrap_worker_context` — the `etl.io` worker calls `upload_tensor` for
  `prefetch`), so a process-global cache would hand the worker a foreign device; cached devices stay
  referenced for process lifetime.
- One-shot `upload_tensor` returns a FRESH destination (the returned tensor owns its buffer — no later
  upload shares it); segment executables keep a persistent destination per `(index, shape, dtype)` exactly
  like the cuda tiers.

Measured (MI300X gfx942 / iree-base 3.11.0, best-of-20, warmup-3): `driver.create_device(device_id=1)`
~46 ms/call and does NOT warm up; `Tensor.to(Device("rocm",0))` 4 KB 36.9→0.052 ms / 4 MB 43.1→0.427 ms;
per-call matmul place+run+readback 89.4→0.280 ms; device-resident run-only 0.064→0.065 ms (unchanged).
CUDA keeps its fresh-per-call device and its own tiers UNCHANGED (gated on the normalized kind being rocm;
cuda may share the same `create_device` cost but that is unvalidated and deliberately left alone).

## iree version matrix & cuda-runtime history (upstream — do not re-investigate)

- Extras pin `iree-base-compiler>=3.11.0,<3.12` + `iree-base-runtime>=3.11.0,<3.12`; compiler and runtime
  must MATCH. Rationale lives in the pyproject comment. 3.11.x is the healthy release for the path that
  matters now (SO algorithms running DEVICE-RESIDENT on cuda with many sequential `etl.run` calls: crash
  repro 4/4 clean, evox SO harness smoke 6 cases / 0 crashes).
- `iree-base-compiler` 3.10.0 and 3.11.0 SIGSEGV (exit -11) in `IREE::Stream::AffinityAnalysis::run()`
  (unbounded `Explorer::walkTransitiveUses → updateFromUse → updateElement → updateValue` recursion) on
  while-heavy MO graphs (≥3 loop-carried values + `select` in a body; SO graphs compile fine). No etl
  control dodges it (`sort_emission`, `while_init_rewrite`, `opt_level`, `iree_compile_args` all tested).
- 3.9.0 compiles NSGA2/MOEAD-class tell graphs 12/12 clean with full numpy parity (user-pinnable for
  compiler-only MO work) — BUT its cuda RUNTIME SIGSEGVs ~9/10 on SO step graphs at the first `etl.run`
  (any scale; error-path use-after-free/race in `iree._runtime_libs._runtime*.so`), and it still SIGSEGVs on
  NSGA3-class graphs (`SpecializeEncodingsPass` / AffinityAnalysis recursion; no exporter-side dodge works —
  select→arith, carry reduction, per-while ablation and computed-init all fail; a second stacked bug emits
  `stream.timepoint.await` awaits before an `async.execute`'s definition). NSGA3-class graphs CANNOT compile
  on iree 3.9.0 regardless of StableHLO formulation.
- 3.12.0rc fixes the compiler but its runtime is NOT usable for these graphs: `local-task` fails at invoke
  (`transient_buffer.c:80 FAILED_PRECONDITION … hal.fence.await`) and `local-sync` fails MOEAD
  (`ref is null` at `hal.buffer_view.create`). Cross-version loading is not a workaround (3.9.0 bytecode v15
  rejected by the 3.11 verifier / 3.12 runtime).
- MO/while-heavy compile class stays upstream-blocked → use the numpy backend or the xla route.
- Legacy releases: also validated on 20241104.1068 (old `iree-compiler`/`iree-runtime` names).

## Adapter gotchas / environment (detail)

- **Mixed-install trap:** old `iree-compiler`/`iree-runtime` and new `iree-base-*` share the SAME `iree`
  namespace and must NOT coexist. A mixed/residual install produced `ValueError: Error creating vm module …
  required module features [Ch] are not available in this runtime configuration` at load (even for a trivial
  graph); a stale pair installed AFTER the base pair silently shadows 3.11.0 inside `iree/` (e.g.
  `iree/compiler/version.py` reporting 20241104.1068 while `importlib.metadata` reports both dists). Fix:
  purge ALL `iree*` dists (incl. residue dirs like `~ree_runtime-*.dist-info`, stale `.egg-info`) and
  reinstall only the base pair; check `iree.compiler.version.VERSION` after any install. `load()` wraps the
  verifier `ValueError` into an actionable `BackendError` (original as `__cause__`).
- **nanobind refleak noise:** `iree-base-runtime` 3.x prints "nanobind: leaked N instances/functions/keep_alive
  records …" to stderr at interpreter shutdown — upstream diagnostics, harmless, not a test failure.
- **CUDA HAL thread-affinity:** any allocating HAL op from a thread with no prior CUDA activity fails with
  `CUDA_ERROR_INVALID_CONTEXT` (`to_host()`, `asdevicearray`, module invoke), even when the HAL device was
  created by another thread — the allocator requires a CUDA context CURRENT on the calling thread
  (primary-context-per-thread; iree does not push its stored context per call). The failure is thread-local
  and non-destructive. One-time per-thread mitigation: the thread first creates its own cuda HAL device on
  the same GPU (`rt.get_driver("cuda").create_device(device_id=idx+1)`, unused) — this makes the shared
  primary context current, after which all ops on the main thread's device work from the worker. Context-current
  state is per-thread (a second context-free thread still fails). No iree-python API makes a context current
  directly. llvm-cpu (local-task) has no such constraint (foreign-thread use is clean). This is why the rocm
  placement caches and the `etl.io` worker bootstrap (`_bootstrap_worker_context`) are thread-local.
- **GPU 7** of the 8× RTX A6000 host reports 2689 uncorrected ECC errors (`CUDA_ERROR_ECC_UNCORRECTABLE` at
  HAL-device creation) — exclude it; GPU 7 is also unusable for the XLA CUDA plugin
  (`PJRT_Client_Create` → "no supported devices found for platform CUDA").

## iree measured per-call performance (iree 3.11.0, RTX A6000 — box noise moves absolutes ±3×)

- DE-shaped fused graph at 4096×50 f32 (~419 stablehlo ops, ~8 dispatches): raw `entry()` invoke floor
  **~0.089–0.093 ms** (host-staged and device buffers identical); full same-device `etl.run` loop
  (device-resident outputs fed back) **~0.146 ms min / ~0.15 med / 0.16 p90 quiet** — per-call etl wrapper
  ~0.05 ms (structure walk + plan lookup ~0.009, per-input validation sub-µs, output wrap + unflatten ~0.01,
  invoke-call python ~0.02). Output readback of the 800 KB state (D2H) **~0.13 ms quiet / 1.2–1.4 ms noisy**.
- **Hot path, trivial `x+1` graph (4096,10) f32:** full `etl.run` pass-through ~0.09–0.17 ms = wrapper
  (`_prepare_flat_inputs` + `_unflatten_outputs`, plan-cached) ~0.018 + `be.run` non-invoke (validation +
  buffer resolution + output wrap + validation) ~0.031 + synchronous `entry()` invoke ~0.06–0.10 ms (the floor).
- `entry()` is fully synchronous with the whole device queue: entry timed after queued 32/64/256/1024 MB D2D
  copies scales 0.18/0.28/1.17/3.52 ms (linear ≈ 230–330 GB/s); back-to-back entries show the second absorbs
  the first's remaining GPU time.
- **Stability (500-run phases, fresh process):** pure pass-through shows NO drift (0.091→0.092). Historic drift
  modes — (a) outputs retained then freed (~80 MB): subsequent dropped runs degrade 0.22→0.94 ms with
  100–400 ms synchronous-trim spikes; (b) `_pool_anchor` removal: host-staging runs degrade 0.36→0.86 ms
  (applies today only to cuda SEGMENT executables). `gc.collect()` every 100 runs does not help. Nanobind exit
  leak counts are CONSTANT (5 instances / 4 keep_alive / 5 types / 39 functions) regardless of run count.
  Intermittent 0.3–0.8 s per-run stalls on the shared box hit BOTH entry and python sides — external contention,
  not etl/iree.
- Small-graph reference (Sphere(4096,10) sum-of-squares, N=300): cuda device pass-through ~0.13 ms,
  llvm-cpu host ~0.13 ms.

## Test strategy — validation narratives

Committed adapter tests live in the SIBLING `tests/backends/` (test-related writes escalate to the repo root):
- Options-override behavior pinned by `tests/backends/test_options_override.py` (38) + `tests/test_pipeline_options.py` (28):
  validate_options union semantics + per-stage listings, iree default-override/deny-list/runtime-args suppression
  + llvm-cpu E2E (incl. reserved `sort_emission`/`while_init_rewrite` lower options), xla plugin_path-at-load +
  compile-options bytes (fake plugin), tvm target/pass_configs version-adaptive, numpy documented-ignore, env
  precedence/laziness/malformed-loud + stub full-path.
- xla device-resident staging semantics regression-pinned by `tests/backends/test_xla_device_resident.py`
  (fake plugin): zero-staging same-device loop (spy counters on `BufferFromHostBuffer`/`ToHostBuffer`),
  run-boundary host-input `DeviceError`, load-device validation, payload/client lifecycle incl. the
  stale-finalizer cache-identity guard.
- iree adapter E2E: `tests/backends/test_iree_emitters_parity.py`, `test_iree_random_algorithms_parity.py`,
  `test_iree_eigh_diag_parity.py`, `test_iree_device_resident.py`, `test_iree_same_device_loop.py`,
  `test_iree_hotpath_validation.py`, `test_iree_rocm_device_resident.py` (rocm-guarded via `_rocm_guard.py`),
  `test_external_call_iree.py`, `test_external_batched_iree.py`, `test_build_cache_iree.py`.
- Fake-plugin driver tests (throwaway scripts in `$TMPDIR`, not committed): ABI sizeof/offset checks, client
  create, compile of real etl StableHLO text, stage/execute/copy, serialize+deserialize round-trip,
  injected-error ⇒ BackendError, version gate, missing-plugin message.
- TVM: numpy parity (matmul+relu+reduce_sum static; dynamic-shape graph with scalar broadcast at sizes 3 and 7),
  artifact save/load round-trip, jax-blocked run (`sys.modules["jax"] = None` ⇒ full pipeline still works).
- Full suite status snapshot (this machine, iree 3.11.0 + tvm 0.26 + jaxlib installed, torch absent): see the
  repo root `CONTEXT.md` for the current counts. NOTE: with pytest 9.1.1, `-q` (the pyproject `addopts`)
  suppresses the final "N passed" line — run with `-v` or `-o addopts=` to see it.
- Under the current explicit-placement semantics, GPU tests written for the OLD host-input semantics fail with
  the intentional `DeviceError` pin category (cuda host-input smokes) — a root-level follow-up converts them to
  explicit placement.
