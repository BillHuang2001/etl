# bench_logs — probe scratch area

One-off compiler/backend probe scripts, notes, and result artifacts for
experiments on this machine.
The **current** HuggingFace benchmark is `./hf_bench.py` → `./hf_bench_results.md` (see its own
section below); `./probe_hf_perf.py` is the older, narrower matrix kept as a working historical
probe.
`./hf_bench_vulkan.py` (+ `./vulkan_findings.md`) is the **IREE Vulkan / lavapipe** datapoint: a
benchmark-only runtime shim that widens the IREE adapter's eight Vulkan string/dialect gates
(monkeypatch, no `./etl/` or `./etl_hf/` edit) so `etl_hf`'s public `build`/`run` path executes a
real checkpoint's whole forward on the software Vulkan device — see its own section below.
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

## `hf_bench.py` — the current benchmark: `etl_hf` vs PyTorch on iree / xla / numpy

`./hf_bench.py` + `./hf_bench_lib.py` measure a real HuggingFace checkpoint's whole-model forward
through `etl_hf` (ONE etl graph, every weight a graph input) on the compiler backends and against
PyTorch baselines. Environment provisioning is `./hf_bench_env.py`; raw records live in
`./hf_bench_data/`; the rendered report is `./hf_bench_results.md` (generated, never hand-edited).
`./probe_hf_perf.py` below is the older narrower matrix — new measurements belong here.

Legs: `numpy-cpu` (interpreter reference), `iree-llvmcpu`, `iree-rocm`, `xla-rocm`, `torch-cpu`,
`torch-rocm` (fp32 eager `.eval()`/`no_grad()`, `torch.cuda.synchronize()` around each call), plus
the opt-in `xla-rocm-fp32` (`dot_precision="HIGHEST"`), `torchcompile-cpu`, `torchcompile-rocm`.
`--phase cpu` = `numpy-cpu,iree-llvmcpu,torch-cpu`; `--phase gpu` = `iree-rocm,xla-rocm,torch-rocm`.
The phases are run SEQUENTIALLY — the MI300X is shared, so never two GPU legs at once.
One subprocess per `(model, seq_len, leg)`, with `--leg-timeout`; the child writes its own JSON and
`.logits.npz` then `os._exit(0)`, so the xla/ROCm teardown abort cannot lose a record.
A leg that cannot run is a RECORD (`*.error.json`) with its verbatim error/returncode/signal, never
an omission: a `—` cell in the report always carries a reason. The record keeps the error text in
full; the report's failure sections (6b, 7) quote only its first 3 non-empty lines, each cut at 200
characters, plus a one-line pointer to that cell's JSON record (`errors[0]`/`notes[0]`) — a compiler
diagnostic is tens of KB of repeated target-triple/CPU-feature flags.

Device legs use the documented **device-resident same-device loop** (params + ids placed once,
each run's device outputs fed back) and record `checks.asdevicearray_calls`, which must be `0` after
warmup; `+d2h` is a separate column (one explicit `Tensor.to(Device('cpu',0))` per call) and the
per-call restage anti-pattern is measured once under `--anti-pattern` for reference only.
Compiled legs always use the library defaults from `etl_hf.pipeline.resolve_build_options`
(`opt_level="O1"` for an all-ROCm target set; `"O3"` + `--iree-llvmcpu-target-cpu=host` /
`--iree-opt-data-tiling` for llvm-cpu) — the harness never hand-picks compiler flags.
`--pack auto` packs only where required or beneficial (`gpt2`'s 197 unpacked inputs exceed the
~161-input IREE runtime limit, and packing collapses ~100 device staging leaves for the 124M
models); the counts are read from `pack_plan(...).specs`, never hard-coded.
Parity is recomputed by `--report` from every leg's saved `.logits.npz` **while those dumps are on
disk**, so leg order does not matter; a golden fixture whose `seq_len` differs from the cell renders
`—` with the reason (the tiny goldens are S=8, `distilgpt2`/`gpt2` S=32), never a number.
`--report` is a pure function of the JSON records + `parity_digest.json` + `hf_bench_env.json` (no
torch, no iree, no measurement) and reproduces byte-identically.

**Data policy — records committed, logits dumps not.** `./hf_bench_data/` commits the 51 JSON leg
records, the env artifact and one small `parity_digest.json` (~26 KB: the parity numbers
`--report` would recompute from the dumps, per cell and reference). The 46 `*.logits.npz` dumps are
**derived bulk** (~393 MB) and are `.gitignore`d — a bench run re-creates them (the driver then
refreshes the digest automatically; `--parity-digest` does it by hand). `--report` therefore
resolves every parity pair from the first available source: the dumps (recomputed, exactly like a
fresh run) → the parity block the leg's own record stored at measure time → the committed digest
(only needed for the 31 pairs whose reference leg was measured *after* that cell, where the
measure-time block legitimately says "no logits on disk") → `—` with the reason. The report header
states which sources were used, and the rendered parity table is identical either way (verified by
diffing the dumps-present and dumps-absent renders) — so the directory is ~700 KB instead of ~393 MB
without losing a number.

Run it (etl/etl_hf come from the checkout via `PYTHONPATH`, never installed into the bench venv):

```
$TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench_env.py          # provision + 17 capability checks → hf_bench_env.json
PYTHONPATH=$PWD ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase cpu --seq-lens 32 \
  --out bench_logs/hf_bench_data --warmup 2 --repeat 5
PYTHONPATH=$PWD ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models ETL_PJRT_PLUGIN=$TMPDIR/pjrt/xla_rocm_plugin.so \
  XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase gpu --seq-lens 32 \
  --out bench_logs/hf_bench_data --warmup 2 --repeat 5 --anti-pattern
$TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --report --out bench_logs/hf_bench_data
```

The throwaway venv (`$TMPDIR/hfbench-venv`) is mandatory: torch must NEVER enter the repo `.venv`
(it flips the etl suite's `importorskip("torch")` guards), and the ROCm torch wheel comes only from
`--index-url https://download.pytorch.org/whl/rocm7.1` **with** `--index-strategy unsafe-best-match`
(without it uv reports "No solution found" for `torch==2.13.0+rocm7.1`).
`hf_bench_env.py` is idempotent (`--verify-only`, `--force`) and records every version, install
command and the required env vars in `hf_bench_env.json`, which section 1 of the report renders.

Measured state of the committed matrix (51 cells: 5 models × seq_len 32, `distilgpt2`/`gpt2` also at
128; best-of-5 warm, batch 1; full tables and every ratio in `./hf_bench_results.md`):
- **xla/rocm is the fastest leg on every measured model** — 7.18×–9.99× torch-rocm (`gpt2` s128
  0.736 ms vs 5.286; `gpt2` s32 0.694 vs 5.277; `distilgpt2` s128 0.399 vs 3.074) and 21.6×–35.6×
  torch-cpu on the real checkpoints.
- **iree/rocm beats torch-rocm in 5 of 6 cells** (9.13× `tiny_llama`, 6.80× `tiny_gpt2`, 2.58×
  `gpt2` s32) and loses exactly one: `gpt2` s128 (7.276 ms vs 5.286 = 0.73×).
- On CPU the numpy interpreter (OpenBLAS) beats the compiled llvm-cpu path on large matmuls
  (`gpt2` 28.9 ms vs 403 ms) and torch-cpu beats both on the real models (14.99 / 8.824 ms).
- **xla/rocm's default fp32 dots are TF32-class** — rel-to-absmax 1.5e-3–2.6e-3 and argmax agreement
  0.9688 on real `gpt2`, vs 1.5e-6–2.9e-6 and argmax 1.0 from the opt-in `xla-rocm-fp32` leg
  (`dot_precision="HIGHEST"`) at +3–29% latency. Always report the pair, never conflate them.
- xla builds cost 8.4–22.4 s (1.2–7.6× iree) plus a one-time ~6 s first `place_params`.

Facts about the current matrix worth knowing before re-running it:
- `tiny_mistral` is exact-erf GELU, so every COMPILED leg of it fails with an explicit
  `HFUnsupportedError` from the activation gate — the numpy and torch legs run normally.
- `iree/llvm-cpu` cannot compile `distilgpt2`/`gpt2` at `seq_len=128` (works at 32):
  `'func.func' op exceeded stack allocation limit of 32768 bytes for function` — an upstream
  iree-compile limit, independent of packing and of any harness option.
- Inductor/Triton caches must not live under `$HOME` (a read-only mount here): the harness points
  `TRITON_HOME`/`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` at `$TMPDIR/hf_bench_cache` for the
  `torchcompile-*` legs.
- `./hf_bench_data/` commits the JSON records + `parity_digest.json` only: a bench run may write one
  `.logits.npz` per ok leg (output-sized, ~393 MB total) but that derived bulk is gitignored and
  regenerated on demand — order-independent parity recomputation comes from the dumps when a run
  left them, else from the records' own parity blocks plus the digest.
- The environment record's `host.repo_root`/`host.tmpdir` are the worktree/scratch paths at
  measurement time, recorded as provenance only.

## `probe_hf_perf.py` + `hf_perf_results.md` + `hf_perf_data/` — whole-forward perf/parity matrix (numpy / iree-llvm-cpu / iree-rocm × unpacked / packed)
Measures the `etl_hf` full-model forward as ONE graph across **six legs per checkpoint** — three
backend paths each measured **unpacked** (one graph input per weight) and **packed**
(`etl_hf.build(forward_fn(cfg, pack_weights=True), cfg)` + `etl_hf.pack_params(pack_plan(...))`,
key suffix `-packed`):
`numpy` (interpreter), `iree-llvm-cpu` (host inputs, library default `opt_level="O3"` +
`--iree-llvmcpu-target-cpu=host`/`--iree-opt-data-tiling`), `iree-rocm` (device-resident on
`Device("rocm", 0)` — the per-call restage anti-pattern is measured separately). All compiled
rows are built with the **library defaults from `etl_hf.pipeline.resolve_build_options`** (the
probe no longer hard-codes options; the all-ROCm default is `opt_level="O1"`). The probe drives
the **functional** `etl_hf` API — `load_hf` → `forward_fn`/`param_specs` → `build`/`run` with the
explicit `etl.lower`/`compile`/`load` stage timings in between (a leg records the API generation
it was measured with in `model.api`; records written by the retired `HFModel` object API predate
that field and the report labels them legacy).
Models (goldens in `./hf_goldens/`): `tiny_llama`, `tiny_gpt2`, `distilgpt2`, `gpt2`;
checkpoints resolve from `$ETL_HF_MODELS_DIR` else `$TMPDIR/etl_hf_models/<slug>` (never downloaded).

Per leg it records graph size (param/weight-input/graph-input counts, IR ops + histogram,
StableHLO bytes/constants, trace/export ms, `pack_params_ms`), lower/compile/load +
`public_build`, `opt_level_default`/`opt_level_used`/`opt_level_sweep`, first/warm
(best/median/mean) call, tokens/s, `+d2h`, restage, `run.api` (which public API drove it), and
parity vs the torch goldens AND vs the *unpacked* numpy reference (packed numpy rows are
bit-identical: `0.00`).

Run (one subprocess per leg, so a crash cannot kill the matrix; results are written per leg and
the report is regenerated from the JSONs, never hand-edited):

```
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py                     # full 24-leg matrix (~110 s)
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --opt-level-evidence # + re-attempt rejected opt levels
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --report            # markdown only
# a CPU-only verification subset, written OUTSIDE the committed matrix:
uv run --isolated --extra iree python bench_logs/probe_hf_perf.py \
    --models tiny_llama --backends numpy,numpy-packed,iree-llvm-cpu,iree-llvm-cpu-packed \
    --no-sanity --no-crash-probe --data-dir $TMPDIR/hfperf/data --report-out $TMPDIR/hfperf/report.md
```

`--data-dir` redirects the per-leg JSON records (and therefore what `--report` renders) and
`--report-out` redirects the markdown, so a subset run never overwrites the committed
`hf_perf_data/` + `hf_perf_results.md` history. Legs need the checkout root on `sys.path` to
import `etl_hf` (a sibling top-level package, not part of the `etl` distribution) — the probe
prepends its own repository root before importing, so the documented command works from a plain
`etl` install without an editable `etl_hf`.

Headline (best warm ms/call, `hf_perf_data/*.json`): unpacked — `tiny_llama` numpy 1.14 /
llvm-cpu 2.10 / rocm 0.20, `tiny_gpt2` 0.97 / 0.63 / 0.21, `distilgpt2` 17.72 / 258.14 / 1.24;
packed — `tiny_llama` 1.11 / 1.75 / 0.17, `tiny_gpt2` 1.02 / 0.48 / 0.16, `distilgpt2` 16.95 /
291.68 / 1.11, **`gpt2` 28.42 / 397.56 / 7.07** (4.02× numpy; 56× llvm-cpu). Parity vs the
torch goldens: ≤8.9e-8 (tiny), 2.2e-4 (distilgpt2), 3.05e-4 (gpt2 packed, absmax 138.7).

Facts pinned by this matrix (all measured, nothing extrapolated):
1. **The ~161-consumed-input cliff is real for UNPACKED graphs only** — unpacked `gpt2` (197
   inputs) is refused up front and the deliberate isolated attempt dies with exit `-11` and no
   Python exception (`gpt2__iree-llvm-cpu.crash.json`); the packed graph declares 4 inputs
   (`gpt2`) / 4–5 (others) and its compiled rows are measured normally.
2. **Packing does not change numerics** — the packed numpy rows are bit-identical to the
   unpacked numpy reference on all four checkpoints; packed IR op counts are higher only because
   of the in-graph static `etl.slice` unpacking (`gpt2` 870 → 1,066).
3. **ROCm `opt_level` default is `O1` for a reason, measured** — `--opt-level-evidence`
   re-attempts the higher levels after a successful build and `O3`/`O2` both fail with
   `'func.func' op failed to distribute` (upstream `iree-compile` ROCm codegen on the
   LayerNorm-shaped reduce) for `distilgpt2` and `gpt2`, unpacked and packed. Evidence lives in
   each leg's `build.opt_level_sweep`.
4. **Device rows use the public API** — `etl_hf.run(exe, place_params(...),
   place_ids(...))`; `run.api` records it. No raw `etl.run` short-circuit is needed (the ids
   tensor reaches `run` as a metadata-validated `etl.Tensor`, never copied to the host).

## `hf_bench_vulkan.py` + `vulkan_findings.md` — `etl_hf` public build/run on IREE Vulkan (lavapipe)

`./hf_bench_vulkan.py` is a **benchmark-only runtime shim** (never a library change; nothing under
`./etl/`, `./etl_hf/`, `./tests/` knows it exists). `install()` monkeypatch-widens exactly eight gates
in the live etl modules — compile-target whitelist (`iree.py:1071`), artifact label (`:460`),
`_GPU_DEVICE_KINDS` (`:486`), the vulkan device-id resolution (`:536`), the CUDA staging tier
(`:1817`), the load device-kind whitelist + `expected_target` (`:1326`/`:1346`) and the transfer-provider
registration (`:2907`), plus `pipeline.py:405` device→`["vulkan-spirv"]` inference — and exposes
`patched_sites()` / `device_info()` / `--check` (self-check, no model). Deliberately not patched:
`core/device.py devices()`, `iree.py:1417 _load_segments`, `etl_hf/pipeline.py:193`, `etl.io`.

Device recipe (both files document it): Mesa **lavapipe** (software, device name
`lavapipe (LLVM 20.1.2, 256 bits)`, apiVersion 1.4.318, Mesa 25.2.8, vendorID 0x10005) reached
through a byte-patched `libvulkan_lvp.so` + ICD json — IREE 3.11 hard-denylists `llvmpipe*` names, so
`VK_ICD_FILENAMES` must point at `$TMPDIR/vk-lvp/lavapipe_icd.json` (also prevents the loader from
dlopening the wedged radeon ICD). Its `device_id` is a per-process UUID hash, never `index+1`.
Throwaway venv `$TMPDIR/vk-venv` (iree-base-compiler/runtime 3.11 + numpy, CPython 3.14.8) with
`etl`/`etl_hf` from the checkout via `PYTHONPATH`; `os._exit(0)` avoids the py3.14 nanobind exit dump.

Measured (`tiny_llama` s32, packed, warmup 2 + best-of-5, two processes): build 954–994 ms, first call
9.8 ms warm-process / 655 ms cold, **best warm 1.53–1.79 ms/call**, +d2h 3.29–3.49 ms,
`asdevicearray` **0**; parity vs the numpy oracle and the torch golden max_abs **8.94e-08**
(rel-to-absmax 2.1e-07 / 2.6e-07), argmax agreement **1.0**, bit-identical across runs. Lavapipe is
CPU software — a Vulkan-path correctness/perf datapoint, never a GPU number.

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
