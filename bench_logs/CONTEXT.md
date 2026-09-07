# bench_logs — probe scratch area

One-off compiler/backend probe scripts, notes, and result artifacts for
experiments on this machine.
Historical artifacts remain committed here: the 763a415-era xla-vs-iree fused
4096×50 probes (`probe_xla_gpu.py`, `probe_iree_same_mlir.py`, `probe_parity.py`,
`de_4096x50_fused_key1.mlir`/`.vmfb`, `clocks_*.log`, `_common.py`) — their
conclusions (host-staging era, adapter 763a415 without cached staging) are
superseded by the device-resident finding below and by
`./etl/backends/adapters/CONTEXT.md`.

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
etl_backend_spec` maps `etl-xla-cuda` → `("xla", None)` and its own notes say
the adapters "stage inputs from host buffers and copy outputs back every step,
so their ms/step includes device↔host transfer" (BENCHMARK_RESULTS.md Setup;
bench_common.py comment: xla "rejects non-cpu devices at load … device spec is
therefore None" — that note predates the device-resident xla path).
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

## Working env recipe (xla-cuda on this machine)

`ETL_PJRT_PLUGIN` = evox venv's `lib/python3.11/site-packages/jax_plugins/
xla_cuda12/xla_cuda_plugin.so`; `XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/
bchuang/xla_cuda_data`; `LD_PRELOAD=/mnt/local-ssd/bchuang/cudnn-xla/lib/
libcudnn.so.9` (cuDNN ≥9.8; RPATH beats LD_LIBRARY_PATH); prepend that dir to
`LD_LIBRARY_PATH`; `CUDA_VISIBLE_DEVICES=<free gpu>` (in-process device id 0);
`PYTHONPATH=/mnt/local-ssd/bchuang/evox/src:/mnt/local-ssd/bchuang/evox`;
interpreter = `/mnt/local-ssd/bchuang/evox/.venv/bin/python` (editable etl must
resolve to `/mnt/local-ssd/bchuang/etl/etl` — assert `etl.__file__` at runtime).
ptxas on PATH. Scan `nvidia-smi` for a free GPU first (GPU 7 is unusable for
xla; iree-cuda 3.9.0 segfaults on a second in-process run — skip iree if it
crashes). For HLO dumps add `--xla_dump_to=$TMPDIR/xla_dump
--xla_dump_hlo_as_text` to XLA_FLAGS (compile-time dumps land even if the run
fails). `graph`-command outputs live on CPU (numpy backend) — no GPU needed.

## Known gotchas

- Device-transfer provider registration is per-backend and order-independent
  (fixed in 55a1d2c): each adapter registers its `upload_tensor` under its own
  `(kind, backend)` per-backend slot, `etl.backends.registry.get` records the
  preferred transfer backend, and `Tensor.to` resolves the preferred slot
  first, then the flat DEFAULT slot (`etl.backends`' lazy iree thunk), then
  `DeviceError`. The old re-registration workaround is unnecessary.
- The run boundary enforces device-resident inputs for cuda executables
  (`core.DeviceError`, no implicit staging) — the same-device loop (feed each
  run's device `Tensor` outputs back as next inputs) is the required fast
  iteration pattern; host restaging costs ~0.8 ms per MB of state round trip.
- Bash `for` loops + `set --` argument passing silently produced misnamed
  outputs in probe runs — use explicit per-invocation args or a Python driver.
