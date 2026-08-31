# XLA GPU probe: real PJRT plugin E2E + iree-vs-XLA on the fused DE/PSO 4096×50 graphs

Date: 2026-09-01 (~01:10–01:25). GPU: RTX A6000 #3 (free at run time, re-checked
via nvidia-smi; 6 tenant GPUs busy). Driver 535.261.03 / CUDA 12.2; plugin built
for CUDA 12.3 (compiles+executes fine).

Answers the user's "try out different compiler backends, see if it's iree's
problem" for the evox-refactor fused DE/PSO 4096×50 tell graphs.

## Setup (exact recipe)

- **etl**: scratch checkout `etl_763a415` at commit `763a415` ("xla adapter: real
  PJRT plugin fixes" — GetPjrtApi casing fallback, minimal CompileOptionsProto,
  ptxas guidance) + one local patch in `xla_util.buffer_from_host` (dims=NULL for
  rank-0; did NOT fix the plugin quirk, see below). All probes run with
  `sys.path.insert(0, etl_763a415)` — the etl-bench venv's editable etl is
  broken (points at a deleted worker dir).
- **plugin**: `xla_cuda_plugin.so` from the `jax_cuda12_pjrt` 0.4.38 wheel
  (exports `GetPjrtApi`). Env `ETL_PJRT_PLUGIN=<so>` (needed at backend-registry
  activation — `plugin_path` alone fails `check_available()`), plus
  `plugin_path` at `etl.compile`.
- **ptxas**: `nvidia-cuda-nvcc-cu12` 12.3.52 unpacked wheel `bin/` on PATH
  (required at compile time).
- **CUDA libs**: `LD_LIBRARY_PATH` = 12 `nvidia/<pkg>/lib` dirs of the
  etl-bench venv's torch wheels (without it: "DNN library initialization
  failed" at `PJRT_Client_Compile`). Must be a hardcoded shell literal — must
  exist before python starts.
- **GPU selection**: xla uses `client.addressable_devices()[0]` →
  `CUDA_VISIBLE_DEVICES=<gpu>`. iree device ids are 1-based
  (`create_device(device_id=gpu+1)`), no CUDA_VISIBLE_DEVICES.
- **Clock state matters**: idle clocks (210/405 MHz) distort staging-heavy
  timing. A tiny spinner (256×256 fp32 matmul every 5 ms, ~1% duty) holds the
  card at 1800/7600 MHz with ~0% sampled utilization (`bench_logs/_spinner.py`;
  nvidia-smi clock lock requires root). Both probes were re-run under the
  spinner; primary numbers below are the boosted-clock runs.
- **Graphs**: phase-A StableHLO dumps
  `evox-refactor/.genesis/workers/worker_T1_A1/benchmarks/results/{de,pso}_4096x50_fused.mlir`
  (read-only; predate the split-reduce fix — PSO still carries the single-stage
  argmin, ideal for a same-text compiler-vs-compiler comparison). The dumps'
  rank-0 inputs (DE arg3 i64 key; PSO arg6 f32 gbest_fit, arg7 i64 key) were
  promoted to `tensor<1xT>` + leading `stablehlo.reshape` (**value-identical
  surgery**) because the 0.4.38 plugin stages rank-0 host buffers as rank-1
  `[1]` (see Blocker). Both compilers run the SAME surg'd text
  (`*_key1.mlir`/`*_key1.vmfb` in this dir); the original vmfb runs as a
  cross-check (iree handles rank-0 natively).

## E2E: xla adapter + real plugin WORKS

DE and PSO both compile (0.60–1.32 s / 0.26–0.77 s across runs — LLVM
fingerprint-reinit variance) and execute on the GPU: 6/10 outputs, all finite,
correct shapes; full StableHLO ingested (incl. the single-stage argmin and the
inline splitmix64 RNG expansion) — **no missing-op ingestion errors**, static
shapes accepted (capabilities `dynamic_shapes=False`).

### Blocker found (rank-0 input staging, plugin-side)
Exact error at run:
```
BackendError: the xla PJRT plugin failed at PJRT_LoadedExecutable_Execute:
Executable expected shape s64[] for argument 3 but got incompatible shape
s64[1]{0}
```
Plugin evidence: `client.buffer_from_host(np.array(42, np.int64)).to_host().shape`
→ `(1,)` (rank-1 [1] buffer) even with dims=NULL/num_dims=0 — the header
contract permits NULL+0. The local dims=NULL patch in `xla_util` did not help.
Workaround: mlir surgery (value-identical). If rank-0 inputs are needed on
xla, the plugin is the blocker, not the adapter.

## Parity evidence (seed 0, uniform(-5.12,5.12) f32, key=42)

xla vs iree, identical surg'd mlir text (real inputs):

| graph | bit-exact outputs | non-exact (reduce-family, fp32 accumulation order) |
|---|---|---|
| DE | 4/6 (outs 0,1,2,3) | outs 4,5: max_abs 2.44e-4 |
| PSO | 7/10 (outs 0,2,3,4,5,6) | outs 1,8,9: max_abs 3.05e-4 |

All-zeros inputs: **16/16 outputs bit-exact** (RNG from key=0 bit-exact on both).

xla vs numpy backend (evox re-trace of the same fused algorithms,
`StdWorkflow(fused=True)`, init key 41 / step key 42, pre-step state fed to the
xla executable):

| graph | state leaves | result |
|---|---|---|
| DE | pop, fit, trial_pop | outs 0,2 **bit-exact**; out 1 (fit re-eval reduce over 204800 f32): max_abs 1.22e-4, **max_rel 2.2e-7**; outs 3–5 = workflow extras (finite, no state leaf) |
| PSO | 7 leaves | outs 0,2,3,4,5,6 **bit-exact**; out 1: max_abs 1.22e-4, max_rel 2.3e-7; outs 7–9 extras (finite) |

Elementwise + RNG outputs are bit-exact as expected. The only non-exact outputs
are the big fp32 fit sums; relative error ~2–3e-7 (two orders below fp32
accumulation noise) — max_abs 1.2–3.1e-4 on ~400–1000-magnitude sums exceeds
the nominal 1e-4 abs budget only in absolute terms on that single output; this
is exactly accumulation-order noise, not a semantic difference.

## Timing (4096×50 fused; boosted clocks; medians)

| cell | DE ms/step | PSO ms/step | how |
|---|---|---|---|
| **XLA total** (per-call: BufferFromHostBuffer ×N + execute + to_host ×M) | **7.38 / 7.89** (2×300) | **14.79 / 15.41** (2×300) | this probe |
| XLA execute-only (pre-staged buffers, 300 calls, per-call output close) | 0.175 | 0.181 | this probe |
| IREE raw invoke, same surg'd text (vm.main + host numpy, 2×300) | 5.37 / 6.14 | 10.11 / 11.79 | this probe |
| IREE original vmfb (phase-A artifact, rank-0 key) | 6.28 | 6.05 | this probe (cross-check) |
| **IREE etl.run reference** (phase-B: etl's cached-staging fast path) | **~1.0–1.3** (~823 sps) | **~1.0–1.25** (~679 sps) | objective/final JSONs |
| IREE device-only reference | 0.062 | 0.067 | phase-B |
| Torch reference | ~0.97–1.04 (~1036 sps) | ~1.14 (~875 sps) | phase-B |

Compile times: **XLA 0.6–1.3 s (DE) / 0.3–0.8 s (PSO)**; iree-compile 2.4 s (DE)
/ 1.3 s (PSO). XLA compiles the same text ~3× faster than iree-compile.

### Component breakdown — where the xla per-call time goes (plugin microbench)
- `PJRT_Client_BufferFromHostBuffer`, 4096×50 f32 (1.64 MB): **5.56 ms/call**
  (≈0.3 GB/s — not bandwidth, per-call staging/sync overhead); 16 KB: 0.12 ms.
- `to_host`, 1.64 MB: **3.15 ms/call** (≈0.5 GB/s).
- Per-buffer fixed cost ≈ 0.75–0.8 ms (DE 10 buffers ≈ 7.5 ms total, PSO 18
  buffers ≈ 15 ms total — matches the measured totals).

## Verdict: not iree's codegen, not the graph — it's host staging, and the xla
## adapter currently has the worst staging path

1. **The graph is tiny on device.** Same mlir text: iree device-only 0.062/0.067
   ms; xla execute-only ~0.18 ms (incl. per-call python/driver overhead + output
   buffer close). Codegen is a non-issue on both compilers.
2. **The naive per-call staging path is the whole story**: 2–15 ms/call on both
   compilers (noisy, interference-prone; see caveats), dominated by host↔device
   copies. XLA's 0.4.38 plugin is the worst single term (5.56 ms per 1.64 MB
   upload); the xla adapter at 763a415 re-stages every input every call — it has
   **no cached-staging fast path**.
3. **The 1-ms-class etl.run reference is the iree adapter's cached-staging
   fast path** (host inputs → cached staging buffers; CONTEXT.md "per-call
   pass-through ~0.13 ms"). Same-graph compiler-vs-compiler on the naive path:
   xla ≈ 1.3–1.5× iree (7.4–7.9 vs 5.4–6.1 DE; 14.8–15.4 vs 10.1–11.8 PSO) —
   a runtime-plumbing gap, not codegen.
4. **Bottom line**: the remaining gap vs pure kernel time is NOT iree's problem
   (its device time is 0.062 ms) and NOT the graph's (kernel is 0.06–0.2 ms);
   it is host staging + per-call overhead. The actionable fix is giving the xla
   adapter the same cached-staging/device-resident treatment the iree adapter
   has — with it, xla per-call should land near its execute-only ~0.2 ms,
   beating both iree etl.run (~1 ms) and torch (~1 ms). Without it, xla is the
   slowest per-call path measured here.

## Adapter limits & gotchas (future work)
- Static-shape gate: OK for these graphs (all static). Dynamic shapes remain
  unsupported by design.
- Full StableHLO ingested — no missing-op errors (incl. argmin, gather,
  scatter, inline splitmix64 RNG, reductions).
- Host-copy per call is the cost: measured above. Cached staging is the missing
  adapter feature (compare adapters/iree.py).
- Rank-0 input staging quirk is plugin-side (0.4.38): rank-0 host buffers
  become `[1]` device buffers; adapter-side workaround is graph surgery (or a
  reshape+copy in `buffer_from_host` for rank-0 — the dims=NULL attempt did not
  help).
- `ETL_PJRT_PLUGIN` env needed at registry activation; `plugin_path` alone
  fails; `LD_LIBRARY_PATH` (venv nvidia wheels) required at compile; ptxas on
  PATH.
- GPU selection via `CUDA_VISIBLE_DEVICES` (addressable_devices()[0]); iree ids
  are 1-based.
- Timing caveats: naive-staging numbers are noisy (5–12 ms spread across
  runs/conditions — bimodal, tenant interference on the shared node; min ~2.6–
  6.5 ms). The fresh surg'd PSO vmfb measured ~2× the phase-A artifact on the
  naive path (10–12 vs 6 ms) in the same run — unexplained, but both sit in the
  noisy naive-staging class and the cached-path reference is unaffected. Use the
  medians across both batches; the structural conclusion (staging ≫ kernel) is
  robust.

## Files
- `_common.py`, `probe_xla_gpu.py`, `probe_iree_same_mlir.py`, `probe_parity.py`,
  `_spinner.py` — scripts (commit `8c6ecfc` + this commit).
- `results_xla.json`, `results_iree.json`, `results_parity.json` — machine
  numbers. `results_xla/`, `results_iree/` — npz parity evidence (regenerable).
- `{de,pso}_4096x50_fused_key1.mlir` + `.vmfb` — surg'd same-text artifacts.
- `clocks_*.log`, `spinner_*.log` — clock/boost evidence.
- Probe etl: `/mnt/hdd_pool/bchuang/tmp_pjrt_probe/etl_763a415` (commit
  763a415 + local dims=NULL patch). Plugin/audit scratch:
  `/mnt/hdd_pool/bchuang/tmp_pjrt_probe/`.
