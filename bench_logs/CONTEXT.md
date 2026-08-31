# bench_logs — probe scratch area

Probe scripts, notes, and results for one-off experiments (compiler/backend
validation, benchmarking, hardware probes). Artifacts are committed here as
"probe work" so they survive and can be re-run; each probe typically adds a
`probe_*.py` script, a `*_notes.md` summary, and result JSONs.

## Current probe: xla-vs-iree on the fused DE/PSO 4096×50 graphs (real PJRT plugin)

See `xla_gpu_notes.md` for the full report (committed with this CONTEXT.md).

- **Verdict**: both compilers' kernels are ~0.06–0.2 ms on the same mlir text;
  the per-call cost is host staging. The xla adapter (763a415) lacks the iree
  adapter's cached-staging fast path and its per-call floor is the 0.4.38
  plugin's BufferFromHostBuffer (~5.6 ms per 1.64 MB). Not iree's problem, not
  the graph's.
- **Scripts**: `probe_xla_gpu.py` (compile+run+time xla), `probe_iree_same_mlir.py`
  (iree on the identical surg'd text), `probe_parity.py` (xla-vs-iree-vs-numpy),
  `_spinner.py` (clock-boost keeper — short kernels leave the GPU at idle
  clocks which distorts staging timing), `_common.py` (shared graph/spec tables
  + rank-0-input mlir surgery for the plugin's rank-0 staging quirk).
- **Run env** (see notes): `ETL_PJRT_PLUGIN`, ptxas on PATH, `LD_LIBRARY_PATH`
  = 12 venv nvidia lib dirs (hardcoded shell literal), `CUDA_VISIBLE_DEVICES`
  for xla / `--gpu N` (device_id = N+1) for iree, light spinner for stable
  clocks, `ulimit -n 65536`.
- **etl under test**: `/mnt/hdd_pool/bchuang/tmp_pjrt_probe/etl_763a415`
  (commit 763a415 + local dims=NULL patch in xla_util.buffer_from_host that did
  NOT fix the rank-0 plugin quirk). Venv: `/mnt/hdd_pool/bchuang/venvs/etl-bench`
  (its editable etl is broken — always sys.path.insert the clone).
- **Known gotcha**: `subprocess.run([...], shell=True)` with a leading
  `ulimit ... &&` silently no-ops the rest of the command (rc=0, nothing runs);
  call executables directly — the parent shell's fd limit is inherited anyway.
  iree 3.11 runtime: `rt.load_vm_module(mod, config=rt.Config(device=dev))`
  (no `device=` kwarg).
- `results_xla/` and `results_iree/` npz are regenerable from the scripts
  (~30 s each); JSONs are the committed numbers.
