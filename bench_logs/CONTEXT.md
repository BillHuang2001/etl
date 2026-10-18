# bench_logs — probe scratch area

One-off compiler/backend probe scripts, notes, and result artifacts for
experiments on this machine.
Historical artifacts remain committed here: the 763a415-era xla-vs-iree fused
4096×50 probes (`probe_xla_gpu.py`, `probe_iree_same_mlir.py`, `probe_parity.py`,
`de_4096x50_fused_key1.mlir`/`.vmfb`, `clocks_*.log`, `_common.py`) — their
conclusions (host-staging era, adapter 763a415 without cached staging) are
superseded by the device-resident finding below and by
`./etl/backends/adapters/CONTEXT.md`.

## `hf_goldens/` — torch/transformers logits oracle for `etl_hf` (durable reference fixtures)

`./hf_goldens/` holds **ground-truth HuggingFace logits** for the five checkpoints `etl_hf` is
validated against, produced by a real `transformers`/PyTorch forward pass. Each fixture is
`<slug>.npz` (`logits` f32 `(1,S,V)` + `input_ids` i32 `(1,S)`), `<slug>.meta.json` (model id,
resolved `model_type`, config summary, versions, exact id/forward recipes, `logits_absmax`/`mean`/
`std`, tie/untie observations, non-parameter buffers to skip, flags), `<slug>.keys.txt` (full
on-disk weight key listing) and `oracle_report.json` (run summary + verification rows). Full
contract, regeneration recipe and key-mapping gotchas: `./hf_goldens/README.md`.

Generator: `./probe_hf_goldens.py`. It imports torch+transformers, so it **must** run in a
throwaway `$TMPDIR` venv — never the repo `.venv`, whose `pytest.importorskip("torch")` guards
would flip to real runs. Reading the fixtures needs numpy only (no torch in tests).

Each `<slug>.meta.json` must carry `model` (the HF repo id — the key the `etl_hf` consumer reads)
alongside the legacy `model_id` (same value), plus `slug`, `seq_len`, `vocab_size`, `dtype`,
`input_ids_recipe`; the generator emits `model` first, immediately before `model_id`. Contract check
(numpy only, no torch): `uv run python bench_logs/verify_hf_goldens_meta.py` — reloads all five
metas + npz files and asserts `model == model_id`, the required keys, preserved key ordering /
`json.dumps(indent=2)` formatting, and the logits/ids shapes, dtypes and `input_ids` recipe.
Pinned checkpoints: `hf-internal-testing/tiny-random-LlamaForCausalLM` (`tiny_random_llama`, S=8,
safetensors, silu, untied), `hf-internal-testing/tiny-random-MistralForCausalLM`
(`tiny_random_mistral`, S=8, legacy torch `.bin`, **GQA 2:1**, `hidden_act: "gelu"` = exact
erf-GELU ⇒ not etl-exportable), `sshleifer/tiny-gpt2` (`tiny_gpt2`, S=8, legacy `.bin`, n_embd=2),
`distilgpt2` (S=32, safetensors, `transformer.`-prefixed keys), `gpt2` (S=32, safetensors,
**prefix-free** keys) — so the GPT-2 family's inconsistent `transformer.` prefix, the
`attn.bias`/`attn.masked_bias` mask buffers (skip them; 4 MB f32 per layer in the real models,
`uint8` in tiny-gpt2) and tied-vs-untied `lm_head` behaviour (tiny-gpt2 ships an *equal copy*;
distilgpt2/gpt2 omit it; the Llama/Mistral tinies ship a *different* matrix) are all pinned.
All five are finite, non-constant, non-degenerate (absmax 0.13–139); a ~1e-44 `tiny_gpt2` result
is therefore a real etl/iree bug, not checkpoint degeneracy.

## Finding (f2f50a7): evox_etl DE/Rastrigin 10000×100 on xla-cuda is at torch parity — the 10–25× "gap" was per-step host staging, not the graph

Real evox_etl StdWorkflow steps (DE/PSO × Rastrigin/Sphere, pop 10000, dim 100,
+ EvalMonitor, seed 42) measured on GPU 1 with the xla-cuda device-resident path
(`backend="xla"`, `device=etl.core.Device("cuda", 0)`, CUDA_VISIBLE_DEVICES
remap, venv recipe below): DE/Rastrigin 0.76–1.03 ms/step (median ~0.86),
PSO/Rastrigin ~0.81–0.86, DE/Sphere ~0.64–0.66.
torch-cuda eager references (evox torch StdWorkflow, same cases): DE/Rastrigin
0.99, PSO/Rastrigin 1.19 ms/step.
So the compiled DE step is AT or BELOW torch eager parity; generation advances
correctly in-graph (verified 0→6 across warmup steps).
The 11.6–28 ms/step numbers quoted in the probe request come from the evox
benchmark harness results (`/mnt/local-ssd/bchuang/evox/benchmarks/etl_vs_torch/
results/so_etl-xla-cuda.json`): DE/Rastrigin/10000x100 = 11.598 ms/step,
DE/Sphere 11.176, PSO/Rastrigin 15.205, PSO/Sphere 17.0 — PSO was equally slow
there, and every algorithm shows a ~5 ms/step floor even at 100×10.
The harness stages host↔device EVERY step by design: `bench_common.
etl_backend_spec` maps `etl-xla-cuda` → `("xla", None)` (host-staged —
`device=None`, written before the device-resident xla path) and its own notes
say the adapters "stage inputs from host buffers and copy outputs back every
step, so their ms/step includes device↔host transfer" (BENCHMARK_RESULTS.md
Setup).
Quantified on current code with the SAME DE step graph: restaged-host execution
(state moved H2D before each step, D2H after) = 11.2–16.6 ms/step; a pure
H2D+D2H round trip of the state alone (10 leaves, 12.1 MB) = 9.3–10.0 ms/step;
device-resident baseline = 0.76–1.03 ms/step.
Verdict: the gap is a MEASUREMENT-PATH artifact (per-step PCIe staging of the
~12 MB state), not (a) an evox_etl graph artifact, (b) an etl exporter/emission
artifact, or (c) inherent xla behavior.
Kernel health of the DE graph (XLA compile dump, sm_8.6 after-optimizations):
10 static-shape args, 17 fusions / 14 PTX kernels, ONE in-graph `conditional`
(the gen==0 init branch — only the taken branch executes per call), no
sort/scatter/while/rng/dynamic-shape markers, largest buffers f32[10000,100]
(4 MB); the only gathers are DE's 3 parent row-gathers of shape (10000,1,100).
The same 4096×50-era xla host-staging floor (~5.6 ms/call, plugin
BufferFromHostBuffer) explains the harness's ~5 ms/step floor at tiny scales.
Scratch evidence (regenerable, machine-local): `/tmp/deprobe/` — `de_pso_xla.py`
(graph/time/dump/torch commands), `stage_cost.py` (device-resident vs restaged
A/B), `run_xla.sh` (env wrapper), `graph_{de,pso}_{rastrigin,sphere}[_nomon].json
[.mlir]` (CPU-side StableHLO exports + op-class/shape stats), `time_*.json`
(xla + torch timings), `xla_dump/` (HLO text, thunk metadata, PTX).

## iree 3.11.0 SO-cuda validation (current state — Defect A resolution)

iree-cuda device-resident SO benchmarking is wired on 3.11.0 (shared venv:
`iree-base-compiler`/`iree-base-runtime` 3.11.0, matched). Crash repro (the
3.9.0-class init-finalization race — fresh process, 2+ consecutive
device-resident `etl.run` sequences at 1000x50 / 10000x100 on iree-cuda, one
process): PASSED 4/4 fresh processes (probe `probe_iree_so_cuda.py` in this dir;
logs in $TMPDIR: `probe_run3.log`-`probe_run5.log`, `probe_final_verify.log`) —
PSO 10000x100 ~0.88 ms/step, DE 10000x100 ~0.72, 1000x50 ~0.23–0.27 ms/step,
exit 0. evox harness smoke (SO suite, ONE process, 6 cases, 0 crashes):
PSO/Sphere 10000x100 3.42 ms/step, PSO/Rastrigin 1000x50 0.96, DE/Rastrigin
10000x100 1.90, DE/Ackley 100x10 0.72, OpenES/Sphere 1000x50 0.60 — parity ok
on all; ANOMALY (not a crash, open follow-up): CMAES/Sphere 10000x100 =
5628.78 ms/step, parity rel 0.185 ok=False. iree compiled artifacts are
version-specific — clear stale caches after any iree upgrade.

## Working env recipe (xla-cuda on this machine)

`ETL_PJRT_PLUGIN` = evox venv's `lib/python3.11/site-packages/jax_plugins/
xla_cuda12/xla_cuda_plugin.so`; `XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/
bchuang/xla_cuda_data`; `LD_PRELOAD=/mnt/local-ssd/bchuang/cudnn-xla/lib/
libcudnn.so.9` (cuDNN ≥9.8; RPATH beats LD_LIBRARY_PATH); prepend that dir to
`LD_LIBRARY_PATH`; `CUDA_VISIBLE_DEVICES=<free gpu>` (in-process device id 0);
`PYTHONPATH=/mnt/local-ssd/bchuang/evox/src:/mnt/local-ssd/bchuang/evox`;
interpreter = `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (the editable etl
install currently resolves to the Defect-fix worker worktree
`/mnt/local-ssd/bchuang/etl/.genesis/workers/worker_T1_A2/etl` — assert
`etl.__file__` at runtime).
ptxas on PATH. Scan `nvidia-smi` for a free GPU first (GPU 7 is excluded:
2689 uncorrected ECC errors — `CUDA_ERROR_ECC_UNCORRECTABLE` at HAL-device
creation). iree-cuda runs use 3.11.0 — crash-repro clean, see below. For HLO dumps add `--xla_dump_to=$TMPDIR/xla_dump
--xla_dump_hlo_as_text` to XLA_FLAGS (compile-time dumps land even if the run
fails). `graph`-command outputs live on CPU (numpy backend) — no GPU needed.

## Finding (MI300X / ROCm 7.1 / jax-rocm7-pjrt 0.11.2): etl's xla adapter runs on ROCm device-resident; the plugin corrupts the heap at process exit; TVM cannot run
Full report: `./rocm_mi300x_findings.md`; conformance + performance matrix for
the 98-example registry: `./etl/bench/rocm_mi300x_results.md` (raw harness JSON
in `./etl/bench/rocm_mi300x_data/`); probes in this dir:
`probe_rocm_raw_pjrt.py`, `probe_rocm_device_info.py`,
`probe_rocm_teardown_minimal.py`, `probe_rocm_etl_pipeline.py`,
`probe_rocm_tvm.py` — set `ETL_PJRT_PLUGIN` to the venv's
`jax_plugins/xla_rocm7/xla_rocm_plugin.so`.
On an AMD Instinct MI300X VF (`gfx942`, ROCm 7.1.0, no NVIDIA GPU;
`uv pip install jax-rocm7-pjrt` = 0.11.2, jaxlib stays 0.10.2) the plugin passes
etl's vendored-header ABI gate (plugin `PJRT_Api.struct_size` = 1144 = the
driver's `sizeof(PJRT_Api)`; API minor 115 vs the vendored 114 — appended
fields only, no rejection/offset mismatch) and the raw driver probe is **7/7
PASS** deterministically (discovery, dlopen/`GetPjrtApi`, initialize, client,
`platform_name='rocm'`, compile, bit-exact execute). `"rocm"` (alias `"hip"`) is
a first-class accelerator device kind for the xla adapter: the kind is validated
against the plugin's own `platform_name` (`xla.py` `_PLATFORM_KINDS`), a mismatch
(e.g. `Device("cuda", 0)` on a ROCm plugin) is rejected rather than mislabelled,
and the device-transfer provider is registered for every non-cpu kind
(`"cuda"`/`"rocm"`/`"hip"`). So `etl.build(..., backend="xla")` + `etl.run` run
with numpy parity host-staged and **device-resident on `Device("rocm", 0)`**,
`Tensor.to(Device("rocm", N))` placement works, and
`pytest tests/backends/test_adapter_xla.py` passes 10/10. The one runtime defect
is **the plugin corrupts the heap at process exit** — any process
that created a PJRT client aborts with `malloc_consolidate(): invalid chunk
size` (SIGABRT 134, sometimes SIGSEGV 139) AFTER all work succeeds (no-client
runs exit 0; `probe_rocm_teardown_minimal.py` reproduces it with **pure ctypes
importing no etl at all**, and explicit `PJRT_Client_Destroy` does not help,
while `os._exit(0)` is always clean) — so it is plugin/runtime-side, not the
driver. TVM: the apache-tvm 0.26.0 wheel is `USE_ROCM=OFF` — it compiles for
`rocm`/`gfx942` but `tvm.runtime.device("rocm",0).exist` is False and the VM run
fails with `RuntimeError: ROCm runtime is not linked into this build; cannot
launch kernels.` (`rocm_fallback_module.cc:59`); etl's tvm adapter is
additionally CPU-only by design (`adapters/tvm.py:304`).

## `probe_hf_perf.py` + `hf_perf_results.md` + `hf_perf_data/` — whole-forward perf/parity matrix (numpy / iree-llvm-cpu / iree-rocm)

Measures the `etl_hf` full-model forward as ONE graph across three legs per checkpoint:
`numpy` (interpreter, cpu:0), `iree-llvm-cpu` (`target_backends=["llvm-cpu"]`,
`--iree-llvmcpu-target-cpu=host`, `opt_level="O3"`), and `iree-rocm` (device-resident on
`Device("rocm", 0)` — `place_params` ONCE, then a same-device `etl.run` loop, never per-call
`.to(device)`; the per-call restage is measured separately as the documented anti-pattern).
Models: `tiny_llama` (1.03M params / 22 graph inputs), `tiny_gpt2` (103K / 37), `distilgpt2`
(81.9M / 101), `gpt2` (124.4M / 197). Per leg it records param/weight-input/graph-input counts,
IR ops + histogram, StableHLO bytes/constants, trace/export/lower/compile/load ms, first-call ms,
warm best/median ms, tokens/s, d2h and restage variants, and parity (max-abs vs the stored torch
goldens in `./hf_goldens/` AND vs the numpy backend's logits, cached as
`hf_perf_data/_numpy_logits_<model>.npz`).

Run (each leg is its own subprocess, so one crash cannot kill the matrix; results are written per
leg and the report is regenerated from the JSONs, never hand-edited):

```
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py              # full matrix (~36 s) + report
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --report     # re-render from existing JSONs
```

Checkpoints are never downloaded: it resolves `$ETL_HF_MODELS_DIR`, else
`$TMPDIR/etl_hf_models/<slug>` (shared with the `etl_hf` test helper + the goldens generator).

Headline measured results (JSONs in `./hf_perf_data/`): warm best per call — `tiny_llama`
numpy 1.05 / llvm-cpu 1.91 / **rocm 0.18 ms** (45,558 tok/s); `tiny_gpt2` 0.98 / 0.49 / **0.19**;
`distilgpt2` 14.01 / 266.39 / **3.02 ms** (10,600 tok/s); `gpt2` numpy 28.67 ms (compiled rows
blocked). GPU vs numpy 4.6–6.0×, GPU vs llvm-cpu up to 88× (`distilgpt2`). On CPU numpy/OpenBLAS
is competitive-to-faster (numpy 1.82× / 19.0× faster on `tiny_llama` / `distilgpt2`; llvm-cpu
2.02× faster only on `tiny_gpt2`) — the compiled path's value is the GPU. Per-call restaging costs
10.3× / 12.3× / 32.5× the same-device loop; adding a d2h readback costs 0.64 / 0.86 / 5.23 ms.
Parity vs torch goldens: `tiny_llama` ≤8.9e-8, `tiny_gpt2` bit-exact (0.0), `distilgpt2` 2.21e-04
(absmax 82.1), `gpt2` 1.53e-04 (absmax 138.7); vs the numpy backend ≤2.29e-04.

Two measured blockers (reported, never worked around here):
1. **`iree-compile` for rocm rejects `opt_level="O3"`/`"O2"` on `distilgpt2`**
   (`'func.func' op failed to distribute`, at the LayerNorm variance `stablehlo.reduce`
   `tensor<1x32x768xf32>` → `tensor<1x32xf32>`); `O1`/`O0` compile, so that row is measured at
   `opt_level="O1"` and recorded in `build.opt_level_sweep` / `build.opt_level_used`. The probe
   sweeps requested → O3 → O2 → O1 → O0 per leg.
2. **`gpt2` (197 graph inputs) cannot run compiled**: IREE's Python runtime SIGSEGVs above ~161
   consumed inputs. A deliberate one-shot subprocess attempt recorded **exit code `-11` with no
   Python exception** (`hf_perf_data/gpt2__iree-llvm-cpu.crash.json`), so the compiled rows are
   marked *pending weight packing*; `etl_hf` v1 exposes no `pack_weights` option (probed via
   `inspect.signature` on `build_model`, `HFModel.build`, `HFModel.graph_fn`). Those rows are NOT
   extrapolated. The oversize crash probe runs on cpu-kind targets only — never provoke an IREE
   SIGSEGV with a live GPU context.
3. **`etl_hf.run_model` gap (etl_hf, not `bench_logs`)**: it converts `input_ids` to a host
   ndarray, which a non-cpu executable rejects at the run boundary (`core.DeviceError`). The
   device-resident rows therefore drive `etl.run(exe, placed_params, placed_ids)` directly.

Also measured: `place_params` cold cost is much higher than steady state (distilgpt2 197 ms cold,
~106 ms steady) — first use builds iree's thread-local HAL device + staging buffer. Reproducibility
is checked by re-running `--sanity-rows` (default `tiny_llama:{numpy,iree-llvm-cpu,iree-rocm}`);
warm-best spread was ±1.7 / ±4.6 / ±9.5 % with parity identical across runs.

## Known gotchas

- MI300X/ROCm runs of anything touching a PJRT client abort at process exit
  (see the finding above) — read the PASS output, not the exit code; probe
  scripts that must exit cleanly use `os._exit(0)` or `--hard-exit`.
- Device-transfer provider registration is per-backend and order-independent:
  each adapter registers its `upload_tensor` under its own `(kind, backend)`
  per-backend slot, `etl.backends.registry.get` records the preferred transfer
  backend, and `Tensor.to` resolves the preferred slot first, then the flat
  DEFAULT slot (`etl.backends`' lazy iree thunk), then `DeviceError`.
- The run boundary enforces device-resident inputs for non-cpu executables
  (cuda/rocm/hip — `core.DeviceError`, no implicit staging) — the same-device
  loop (feed each run's device `Tensor` outputs back as next inputs) is the
  required fast iteration pattern; host restaging costs ~0.8 ms per MB of state
  round trip.
- Bash `for` loops + `set --` argument passing silently produced misnamed
  outputs in probe runs — use explicit per-invocation args or a Python driver.
