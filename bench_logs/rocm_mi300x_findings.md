# etl XLA (PJRT) + TVM on AMD Instinct MI300X (ROCm 7.1.0, gfx942) — probe findings

Date: 2026-10-16. Machine: AMD Instinct MI300X VF (`gfx942:sramecc+:xnack-`),
ROCm 7.1.0 (`/opt/rocm`), no NVIDIA GPU (`nvidia-smi` absent). Python 3.14.8
(uv venv, `uv sync --all-extras`): numpy 2.5.2, jaxlib 0.10.2, apache-tvm
0.26.0, torch 2.13.0+cu130, iree-base-* 3.11.0, pytest 9.1.1.

Conformance + performance results for the whole `etl.bench` registry across
numpy/cpu, iree/llvm-cpu, iree/rocm:0 and xla/rocm:0:
`./etl/bench/rocm_mi300x_results.md` (raw harness JSON in
`./etl/bench/rocm_mi300x_data/`).

## TL;DR

| Question | Answer |
|---|---|
| Does etl's XLA/PJRT driver work against a ROCm plugin? | **YES** — raw driver probe 7/7 PASS (load, ABI gate, client, compile, execute bit-exact) |
| Does the full etl `xla` pipeline run on ROCm? | **YES** — `etl.build(..., backend="xla")` + `etl.run` give numpy-parity results, host-staged *and* device-resident |
| Is the ROCm plugin wired as a first-class device kind? | **YES** — `Device("rocm", 0)` (alias `"hip"`) is a first-class accelerator kind, validated device-resident on the real MI300X; the kind is checked against the plugin's `platform_name` (`xla.py` `_PLATFORM_KINDS`), so a `"cuda"` label on a ROCm plugin is rejected rather than mislabelled |
| Can TVM RUN on ROCm here? | **NO** — the installed wheel is built `USE_ROCM=OFF`; it *compiles* for gfx942 but cannot launch kernels (verbatim error below) |
| Any blocker? | **YES (one):** the ROCm PJRT plugin corrupts the heap at **process exit** — every process that created a PJRT client aborts with `malloc_consolidate(): invalid chunk size` (or SIGSEGV) *after* all work has completed successfully |

## 1. Plugin: `jax-rocm7-pjrt` 0.11.2

`uv pip install --dry-run jax-rocm7-pjrt` → `Resolved 1 package` / no extra
deps (verified), then `uv pip install jax-rocm7-pjrt` → `jax-rocm7-pjrt==0.11.2`.
**jaxlib stayed at 0.10.2** (unaffected — the plugin package does not depend on
jaxlib or jax).

Exact plugin path (absolute):

```
/root/etl/.genesis/workers/worker_T1_A2/.venv/lib/python3.14/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
```

`import jax_plugins.xla_rocm7` does **not** work (its `__init__.py` imports
`jax._src.hardware_utils`; the `jax` package is absent) — irrelevant, since
etl drives the `.so` directly through ctypes. `ldd` resolves all 37 deps with
**0 "not found"** (ROCm is at `/opt/rocm`, RPATH does the work — **no
`LD_LIBRARY_PATH` change is needed**). The exported C symbol is
`GetPjrtApi` (lowercase "t") — etl's casing fallback (`_pjrt_c_api.py:793-798`)
handles it.

## 2. Raw PJRT driver probe — `probe_rocm_raw_pjrt.py` — **7/7 PASS**

```
ETL_PJRT_PLUGIN=<so> ./.venv/bin/python bench_logs/probe_rocm_raw_pjrt.py
```

```
[PASS] 1.  plugin discovery (_find_plugin_path)
[PASS] 1b. dlopen + GetPjRtApi + verify_api (raw)
           plugin abi major/minor  = 0/115
           plugin struct_size      = 1144 bytes
           driver requires        >= 608 bytes (through PJRT_Buffer_ToHostBuffer)
           driver PJRT_Api sizeof  = 1144
           vendored header         = major 0 / minor 114 @ 70fe66213b73 (2026-07-15)
[PASS] 2.  xla_util._load_plugin (ABI gate + PJRT_Plugin_Initialize)
[PASS] 3.  PJRT_Client_Create (xla_util.acquire_client)
[PASS] 4.  platform info + device inventory
           platform_name    = 'rocm'
           platform_version = 'rocm 70125424'
           num addressable devices = 1
[PASS] 5.  PJRT_Client_Compile (trivial StableHLO add)   num_outputs = 1
[PASS] 6.  execute (buffer_from_host + Execute + ToHostBuffer)
           device output = [11.0, 22.0, 33.0, 44.0]   numpy expected = same   bit-exact: True
== RAW PJRT PROBE: 7/7 PASS ==
malloc_consolidate(): invalid chunk size         <- see §4
```

**No version-gate rejection, no ABI/offset mismatch.** The vendored header is
PJRT API minor **114** (commit `70fe66213b73`); the plugin reports minor **115**
and `struct_size = 1144` — *exactly* the driver's own `sizeof(PJRT_Api)`. The
gate passes because fields are only ever appended (major stays 0) and the
driver only reads through `PJRT_Buffer_ToHostBuffer` (needs ≥ 608 bytes).
Deterministic: 3/3 identical runs.

## 3. Device inventory — `probe_rocm_device_info.py` (5/5 rounds, deterministic)

Reading the untyped `PJRT_Device_*` / `PJRT_DeviceDescription_*` slots (the
vendored `PJRT_Api` types only the fields etl calls, so the probe casts them
itself — driver untouched):

```
[round 1..5] platform_name='rocm' platform_version='rocm 70125424' ndev=1
  device[0] kind='AMD Instinct MI300X VF' debug_string='rocm:0'
```

Corroborated by the plugin's own XLA log line:

```
service.cc:154] XLA service ... initialized for platform ROCM ...
service.cc:170]   StreamExecutor [0]: AMD Instinct MI300X VF, AMDGPU ISA version:
                  gfx942:sramecc+:xnack- (Driver: 7.1.25424; Runtime: 7.1.25424;
                  Toolkit: 7.14.60850; DNN: 1.3.0)
gpu_helpers.cc:162] XLA backend allocating 143.77GiB (154367164416 bytes) on device 0 for BFCAllocator.
```

So: **yes, it is ROCm** (`platform_name = "rocm"`, device debug string
`"rocm:0"`). The plugin grabs a 143.77 GiB BFC pool at client creation (the
same one-client-per-process concern etl's refcounted shared client already
handles for CUDA).

## 4. BLOCKER: the plugin corrupts the heap at process exit

Everything above succeeds, and *then* the process aborts **at exit**:

```
malloc_consolidate(): invalid chunk size
```
(SIGABRT, exit 134 — sometimes SIGSEGV 139 at exit instead).

### Isolation matrix — all runs on this machine

| # | What runs | Exit | Notes |
|---|---|---|---|
| A1 | `probe_rocm_raw_pjrt.py --stop-after=load` (dlopen + Initialize, **no client**) | **0** | clean |
| A2 | `--stop-after=client`, normal exit | **134** ×5/5 | abort at exit |
| A3 | `--stop-after=client --hard-exit` (`os._exit(0)`) | **0** ×3/3 | no exit handlers run |
| A4 | full probe, normal exit | **134** ×3/3 | all 7 steps PASS first |
| B1 | `probe_rocm_teardown_minimal.py` (**pure ctypes, imports no etl at all**), normal exit | **134** | ← attribution |
| B2 | same `--destroy` (explicit, error-free `PJRT_Client_Destroy`) + normal exit | **134** | clean destroy does not help |
| B3 | same `--hard-exit` | **0** | |
| C1 | `pytest tests/backends/test_adapter_xla.py` (with the plugin) | **134** | **10 passed** first |

### Attribution — it is the PLUGIN, not etl

`probe_rocm_teardown_minimal.py` reproduces the abort with a **hand-rolled
ctypes struct and no etl code imported whatsoever** (only the first 12
`PJRT_Api` slots: `PJRT_Plugin_Initialize` = 3, `PJRT_Client_Create` = 10,
`PJRT_Client_Destroy` = 11). So it cannot be etl's driver, its handle wrappers,
or any Python object finalizer: **creating a PJRT client with this plugin
poisons process teardown**. `os._exit(0)` (which skips both Python
finalization and glibc `exit()` handlers / C++ static destructors) is always
clean, so the corruption is detected during exit handling, not during any PJRT
call (every PJRT call returns `err=None`).

Explicit `PJRT_Client_Destroy` does **not** avoid it (B2), so it is not a
"forgot to destroy" problem. Most likely the plugin's own static destructors /
the ROCm+HIP runtime teardown after XLA's global state — it is
environment/plugin-level, not fixable from etl's side. Not investigated
further (out of scope): no workaround is known from this probe.

**Practical consequence:** the xla adapter is *functionally* usable on ROCm
(graphs compile and run with numpy parity, and the existing
`tests/backends/test_adapter_xla.py` suite passes 10/10 there), but any
process that used it exits with SIGABRT/SIGSEGV *after* doing its work. A host
application must tolerate that (or `os._exit` deliberately).

## 5. Full etl pipeline on ROCm — `probe_rocm_etl_pipeline.py`

Graph: `etl.sum(etl.sin(x) * y, axes=(1,)) + etl.sum(x, axes=(1,))`,
`x`,`y`: `float32[4,3]`, compared against the numpy reference
(`rtol=atol=1e-6`).

`"rocm"` (alias `"hip"`) is a first-class accelerator kind: the adapter derives
the acceptable kinds from the plugin's own `platform_name`
(`xla.py` `_PLATFORM_KINDS`), so a kind the platform does not serve is rejected
rather than mislabelled.

| Case | `device=` | Result |
|---|---|---|
| A | omitted → `Device("cpu", 0)` (host-staged execution on the plugin's device 0) | **PASS** — `etl.build(fn, spec, spec, backend="xla", plugin_path=<so>)` + `etl.run` → `allclose: True` |
| B | `core.Device("rocm", 0)` | **PASS** — device-resident: `Tensor.to(Device("rocm",0))` placement, outputs stay device-resident, `.to(cpu)` + `.numpy()` round trip → `allclose: True` |
| B | `core.Device("hip", 0)` | **PASS** — same; `"hip"` is an accepted alias of the canonical `"rocm"` kind |
| C | `core.Device("cuda", 0)` | **REJECTED** — `BackendError`: the plugin reports platform `'rocm'`, which serves `"rocm"`/`"hip"`, not `"cuda"` — an honest rejection (never mislabel where the data lives) |

### Current state

The device kind is validated against the plugin's platform: after
`check_available()`, the adapter queries the plugin's `platform_info()` and
accepts only a kind the platform serves (`rocm`/`hip` on a ROCm plugin, `cuda`
on a CUDA plugin, `cpu` always the host-staged path), rejecting a mismatch
instead of mislabelling. `upload_tensor` is registered per non-cpu kind
(`register_backend_device_transfer_provider` for `"cuda"`, `"rocm"` and
`"hip"`), so explicit `Tensor.to(Device("rocm", N))` placement works. The
device-resident run path
(`device_resident = self.device is not None and self.device.kind != "cpu"`) is
kind-agnostic. The whole `etl.bench` matrix ran on this hardware with
`Device("rocm", 0)` — numpy/cpu 98/98 PASS, iree/llvm-cpu 75 PASS,
iree/rocm:0 70 PASS, xla/rocm:0 59 PASS, zero numeric failures beyond the
documented deferrals — see `./etl/bench/rocm_mi300x_results.md` (raw JSON in
`./etl/bench/rocm_mi300x_data/`).

Also note: the placement provider calls `_load_plugin()` **with no options**, so
`Tensor.to(Device(...))` discovers the plugin only via `ETL_PJRT_PLUGIN` /
well-known paths — `plugin_path` is a compile/load option only.

## 6. TVM on ROCm — `probe_rocm_tvm.py` — **cannot run**

```
[INFO] 1. wheel build config: USE_ROCM=OFF; USE_LLVM=ON; USE_CUDA=ON; ROCM_PATH=<unset>
[INFO] 2. tvm.contrib.rocm = False; tvm.contrib.hip = False
[FAIL] 3. tvm.runtime.device('rocm', 0)   -> device(type='rocm', index=0); .exist = False
[PASS] 4. tvm.target.Target('rocm')       -> mcpu=gfx942, mtriple=amdgcn-amd-amdhsa-hcc
[PASS] 5. build_vm_executable(target='rocm')  -> VMExecutable built (codegen+link for gfx942 OK)
[FAIL] 6. relax VM run on device rocm:0   -> RuntimeError (verbatim):
   "ROCm runtime is not linked into this build; cannot launch kernels. Re-link
    with USE_ROCM=ON or load this module in a ROCm-equipped environment via
    tvm.runtime.load_module."
   (raised from tvm/target/rocm_fallback_module.cc:59,
    tvm::target::ROCmFallbackModuleNode::GetFunction, via
    tvm.runtime.vm.VirtualMachine._setup_device)
```

Conclusion: **apache-tvm 0.26.0 (the PyPI wheel) can COMPILE for
`rocm`/`gfx942` but has NO ROCm device runtime and cannot execute.** It is
`USE_ROCM=OFF`, ships no `tvm.contrib.rocm`/`tvm.contrib.hip`, and
`tvm.runtime.device("rocm", 0).exist` is `False`. Building the device module
succeeds (LLVM AMDGPU codegen + link), but the module is a
`ROCmFallbackModule` that refuses to launch. Running TVM on this machine
therefore needs a **from-source TVM built with `USE_ROCM=ON`** (or a
ROCm-equipped environment used via `tvm.runtime.load_module`), which is
entirely outside etl.

Independently, etl's own tvm adapter is **CPU-only by design** in v1:
`etl/backends/adapters/tvm.py:304` rejects any `device.kind != "cpu"` with
`BackendError: the tvm backend v1 supports CPU devices only`. So even with a
ROCm-enabled TVM, etl's tvm adapter would need a device-kind/execution change
of its own (unlike xla, the tvm adapter's load path is CPU-hardcoded and its
default `tvm_target` is `"llvm"`).

## 7. Files in this directory

| File | Purpose |
|---|---|
| `probe_rocm_raw_pjrt.py` | raw PJRT viability through etl's driver (7 steps); `--stop-after=load\|client\|compile\|execute`, `--hard-exit` for the teardown A/B |
| `probe_rocm_device_info.py` | platform + device inventory (kind / debug string); `--repeat=N`; ends in `os._exit(0)` on purpose |
| `probe_rocm_teardown_minimal.py` | **pure-ctypes, imports no etl** — attributes the exit crash to the plugin; `--destroy`, `--hard-exit` |
| `probe_rocm_etl_pipeline.py` | full `etl.build`/`etl.run` on ROCm — host-staged + device-resident on `Device("rocm"\|"hip", 0)` — and the plugin-platform kind check (`Device("cuda",0)` on a ROCm plugin rejected) |
| `probe_rocm_tvm.py` | TVM/ROCm build-vs-run verdict with verbatim errors |

Reproduce (all from the repo root, `ETL_PJRT_PLUGIN` set to the path in §1):

```bash
SO=$PWD/.venv/lib/python3.14/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
ETL_PJRT_PLUGIN=$SO ./.venv/bin/python bench_logs/probe_rocm_raw_pjrt.py
ETL_PJRT_PLUGIN=$SO ./.venv/bin/python bench_logs/probe_rocm_device_info.py --repeat=5
./.venv/bin/python bench_logs/probe_rocm_teardown_minimal.py $SO            # -> 134
./.venv/bin/python bench_logs/probe_rocm_teardown_minimal.py $SO --hard-exit # -> 0
ETL_PJRT_PLUGIN=$SO ./.venv/bin/python bench_logs/probe_rocm_etl_pipeline.py
./.venv/bin/python bench_logs/probe_rocm_tvm.py
```

Expected exit codes today: every command except the `--hard-exit` / `--stop-after=load`
ones ends in 134 (or 139) **after** printing all of its PASS lines — that is
finding §4, not a probe failure.

## 8. Evidence log (raw, machine-local)

Full stdout of every run in this probe (XLA's own `I…` log lines filtered):

```
raw probe, full, normal exit      : 7/7 PASS, then malloc_consolidate -> 134   (3/3 runs)
raw probe, --stop-after=load      : 2 PASS, exit 0
raw probe, --stop-after=client    : 3 PASS, then malloc_consolidate -> 134     (5/5 runs)
raw probe, --stop-after=client --hard-exit : exit 0                            (3/3 runs)
device info, --repeat=5           : 5 identical clean rounds, exit 0
minimal pure-ctypes, normal exit  : err=None everywhere, then malloc_consolidate -> 134
minimal pure-ctypes, --destroy    : PJRT_Client_Destroy err=None, then abort -> 134
minimal pure-ctypes, --hard-exit  : exit 0
etl pipeline                      : A PASS, B device-resident PASS (rocm/hip), C REJECTED (cuda on rocm plugin), then abort -> 134
pytest tests/backends/test_adapter_xla.py : 10 passed, then abort -> 134
tvm probe                         : 2 PASS / 3 FAIL + 1 SKIPPED, exit 0
```
