# etl_hf on IREE Vulkan (lavapipe): public build/run path via a runtime shim

**Status: WORKS.** `etl_hf`'s unmodified public API — `load_hf` → `forward_fn` →
`pack_plan`/`pack_params` → `build(backend="iree", device=Device("vulkan", 0))` →
`place_params`/`place_ids` → `run` in a warm same-device loop — runs a real
checkpoint's whole forward pass as ONE graph on IREE 3.11's **Vulkan** HAL, with
the shim in [`./hf_bench_vulkan.py`](./hf_bench_vulkan.py) installed. Nothing under
`./etl/`, `./etl_hf/` or `./tests/` was modified: the shim widens the adapter's
Vulkan gates at runtime (monkeypatch), so this is benchmark-only evidence, never a
library change.

Measured on `tiny_llama` (hf-internal `tiny-random-LlamaForCausalLM`), `seq_len=32`,
batch 1, packed weights, warmup 2 + best-of-5. Two independent processes:

| metric | run 1 | run 2 |
|---|---|---|
| build (trace → lower → compile → load, incl. `iree-compile`) | **993.7 ms** | **954.3 ms** |
| first call | **655.1 ms** (cold process: SPIR-V module instantiation + lavapipe cold pages) | **9.8 ms** |
| **best warm ms/call** | **1.530 ms** (median 1.567, mean 1.572, stdev 0.040) | **1.791 ms** (median 1.804, mean 1.829, stdev 0.054) |
| +d2h (one explicit `Tensor.to(Device("cpu",0))` per call) | 3.289 ms best (median 3.922) | 3.485 ms best (median 3.953) |
| `asdevicearray` calls in the warm loop | **0** | **0** |
| host→device placement of the 4 packed leaves (once) | 2.218 ms (`place_ids` 0.040 ms) | 2.301 ms |
| parity vs the pure-numpy oracle (`etl_hf.reference_logits`) | max_abs **8.94e-08**, rel-to-absmax 2.14e-07, argmax **1.0000** | identical |
| parity vs the committed torch/transformers golden (`hf_goldens/tiny_random_llama.npz`, 1×8) | max_abs **8.94e-08**, rel-to-absmax 2.56e-07, argmax **1.0000** | identical |

The two runs' logits are **bit-identical** (same `max_abs`/`mean_abs`), so the Vulkan
path is deterministic; the first-call number varies with process warmth (655 ms cold
vs 9.8 ms on the second run) while the steady-state cost is ~1.5–1.8 ms/call.

fp32 accumulation order differs from numpy's ⇒ **not** bit-exact; ~1e-7 absolute at
absmax 0.42 (0.35 for the s8 cell) with **full argmax agreement** is the expected
fp32 result, and it is the same order of magnitude every other etl backend shows on
this checkpoint. (The golden fixture is `seq_len=8`, so the s32 cell cannot be
compared to it directly; the s8 comparison above was measured with a second build at
`seq_len=8` — 930.2 ms — which is apples-to-apples.)

Logits stayed on the device in an `IreeDevicePayload` (`out.device` =
`Device(kind='vulkan', index=0)`, shape `[1, 32, 32000]`, float32); exactly one
explicit `.to(cpu)` produced the host array used for the parity numbers.

## Device actually used

| property | value |
|---|---|
| IREE-visible device name | `lavapipe (LLVM 20.1.2, 256 bits)` |
| device apiVersion | **1.4.318** (`0x0040413e`) |
| driverVersion | Mesa **25.2.8** (`0x06402008`) |
| vendorID | `0x10005` |
| deviceType | `CPU` (software rasterizer — **not** a GPU) |
| VkPhysicalDevice path / UUID | `6d657361-3235-2e32-2e38-2d3075627500` |
| IREE `device_id` | a **per-process UUID hash** (observed 941009984, 588459008, 1052880608, 1066058208, 752717504, 270783856 in different processes) — never `index + 1` |

Two facts about the host matter:

* **IREE 3.11 hard-denylists lavapipe by name.** `vulkan_driver.cc`
  (`iree_hal_vulkan_is_device_visible`) returns false when `strstr(deviceName,
  "llvmpipe") == deviceName`, and lavapipe's stock name starts with `llvmpipe`, so
  IREE enumerates **0** Vulkan devices with no flag/env override. The workaround is a
  byte-patched copy of `libvulkan_lvp.so` whose `deviceName` string says `lavapipe`
  (`$TMPDIR/vk-lvp/lavapipe_renamed.so` + `lavapipe_icd.json`).
* **Always set `VK_ICD_FILENAMES` to that ICD json.** It pins the loader to the
  renamed lavapipe ICD and therefore stops it from dlopening the **radeon** ICD,
  which *blocks* on this host (the amdgpu DRM device is wedged; HIP reports no
  device). Never touch `/dev/dri`, ROCm or HIP.

```bash
export VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json
export ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=<repo root>              # etl and etl_hf from the checkout, never installed
$TMPDIR/vk-venv/bin/python bench_logs/hf_bench_vulkan.py --check     # self-check, no model
```

`$TMPDIR/vk-venv` is a throwaway venv holding **only** `iree-base-compiler` +
`iree-base-runtime` 3.11 + numpy (CPython 3.14.8). `iree.runtime` on 3.14 can emit a
nanobind teardown dump at interpreter exit; anything that must exit cleanly flushes
and calls `os._exit(0)` (the `--check` path does).

## What blocked the public path before the shim (verbatim)

Reproduced in this tree with the shim **not** installed (`etl` from the checkout,
`etl_hf.load_hf` → `forward_fn`):

```
Device ctor OK: Device(kind='vulkan', index=0)

etl_hf.build(fwd, cfg, backend='iree', device=Device('vulkan',0))
  -> etl.core.errors.BackendError: the iree backend supports cpu, cuda and rocm/hip
     devices only, got device kind 'vulkan'

etl_hf.build(..., device=dev, target_backends=['vulkan-spirv'])
  -> etl.core.errors.BackendError: the iree backend does not support compile target
     'vulkan-spirv' — supported targets: cuda, llvm-cpu, rocm

etl.core.devices('vulkan')
  -> etl.core.errors.DeviceError: Unknown device kind 'vulkan'; recognized kinds are
     'cpu', 'cuda' and 'rocm' (alias 'hip').

etl.from_numpy(zeros).to(Device('vulkan',0))
  -> etl.core.errors.DeviceError: Tensor.to cannot place data on
     Device(kind='vulkan', index=0): no device-transfer provider is registered for
     device kind 'vulkan'. The etl iree backend provides cuda placement — activate it
     (etl.backends.get('iree')) or register a provider via
     core.register_device_transfer_provider.
```

So three independent *string* whitelists (compile targets, load device kinds,
transfer providers) plus a device→target inference gap and a device-id convention
block the path. No public flag or env var reaches them; `target_backends` is the only
entry point and the raw `--iree-hal-target-backends` flag escape hatch is explicitly
denied by the adapter.

## The eight patched sites (exactly these, nothing else)

`etl/backends/adapters/iree.py`

| site | library behaviour | shim |
|---|---|---|
| `:1071` `IreeBackend._resolve_compile_options` | `supported_targets = {"llvm-cpu","cuda","rocm"}` rejects `vulkan-spirv` | **wrapped** — a vulkan-only target list is resolved by the shim (the library's own `opt_level` → `--iree-opt-level=O{N}` + `iree_compile_args` resolution), a mixed list re-attaches `vulkan-spirv` to the library result, unknown targets delegate |
| `:460` `_artifact_target` | target→label map; `vulkan-spirv` falls through to a `"+".join` label | **wrapped** → label `"vulkan"` |
| `:486` `_GPU_DEVICE_KINDS` | `frozenset({"cuda","rocm"})` selects the `rt.Config(device=…)` load path and the non-host input branch | module global **widened** with `"vulkan"` |
| `:536` `_acquire_runtime_device` | `device_id = device.index + 1` (CUDA/ROCm 1-based ordinal) | **wrapped** — vulkan creates the device from the **queried** `query_available_devices()[index]` info dict and caches one HAL device per `(index, runtime args)` so uploads and the module load share it |
| `:1817` `_staged_input` | non-ROCm branch is the CUDA pinned/DMA staging tier | **wrapped** → `None` for vulkan (plain `rt.asdevicearray`; the CUDA ordering guarantee is unverified on lavapipe) |
| `:1326` / `:1346` `IreeBackend.load` | device-kind whitelist `("cpu","cuda","rocm","hip")`; `expected_target = {…}[device_kind]` raises `KeyError` | **wrapped** — for a vulkan device the wrapper replays the adapter's own load steps (`check_available` / `validate_options` / `_apply_iree_runtime_args` / `base64.b64decode` / `_acquire_runtime_device` / `_load_vm_module` / `IreeExecutable`) with those two gates widened; every other device delegates unchanged |
| `:2907` transfer-provider registration | `for kind in ("cuda","rocm","hip")` | the adapter's own `upload_tensor` **registered** for `"vulkan"` in both the per-backend `("vulkan","iree")` slot and the flat default slot |

`etl/pipeline.py`

| site | library behaviour | shim |
|---|---|---|
| `:405` `_resolve_backend_device` | device→`target_backends` inference covers `cuda`/`rocm`/`hip`/`cpu`; a vulkan device compiles `llvm-cpu` and then fails the load-time target check | **wrapped** — an iree backend on a vulkan device infers `target_backends=["vulkan-spirv"]` when neither the option nor `ETL_TARGET_BACKENDS` is set |

Deliberately **not** patched (verified unnecessary for this path):
`core/device.py:122 devices()` (callers construct `Device("vulkan", 0)` directly — the
constructor has no kind whitelist); `iree.py:1417 _load_segments` (external-kernel
split path — an HF forward graph has no `external_call`); `etl_hf/pipeline.py:193
_targets_from_device` (it only picks etl_hf's `opt_level` default; compile targets
come from `target_backends`); `core/tensor.py` (provider resolution is
kind-agnostic); `etl.io` (unused here); `etl.bench._util` (not used here).

`install()` is idempotent, must run after `etl` is importable and before the
executable is built, and returns `{site: "what changed"}` (also `patched_sites()`).
`device_info()` reports the enumerated device as evidence.

## Reproduce the end-to-end proof

```bash
export VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json
export ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD
$TMPDIR/vk-venv/bin/python bench_logs/hf_bench_vulkan.py --check   # shim + device evidence
```

and, for the measured cell above (a scratch driver; the committed artifact is the
shim, not a benchmark harness):

```python
import sys; sys.path[:0] = [".", "bench_logs"]
import hf_bench_vulkan; hf_bench_vulkan.install()
import etl, etl_hf

loaded = etl_hf.load_hf("$ETL_HF_MODELS_DIR/tiny_llama")
cfg, params = loaded.config, loaded.params
fwd = etl_hf.forward_fn(cfg, seq_len=32, pack_weights=True)
dev = etl.Device("vulkan", 0)
exe = etl_hf.build(fwd, cfg, backend="iree", device=dev)          # ~0.95-1.0 s
plan = etl_hf.pack_plan(etl_hf.param_specs(cfg))
placed = etl_hf.place_params(etl_hf.pack_params(plan, params), dev)  # ~2.2 ms, once
ids = etl_hf.place_ids(input_ids, dev)
out = etl_hf.run(exe, placed, ids)   # 9.8-655 ms first call, then ~1.5-1.8 ms/call,
                                     # 0 uploads in the warm loop
host = out.to(etl.Device("cpu", 0)).numpy()        # the one explicit D2H copy
```

## Notes / caveats

* **lavapipe is a software Vulkan device on the CPU.** These numbers are a
  Vulkan-path correctness/perf datapoint, not GPU numbers: 1.53 ms/call for a 124M-class
  *tiny* model is a plausible software-rasterizer cost, and it moves no work to any
  accelerator. The interesting result is that the whole **public** staging pipeline
  (trace → lower → `iree-compile --iree-hal-target-backends=vulkan-spirv` → load →
  device-resident run) is device-kind-generic apart from those eight string/dialect
  gates.
* Device-resident semantics hold unchanged on vulkan: host inputs are rejected at the
  run boundary, results stay on the device, and the documented same-device loop costs
  **0** `asdevicearray` calls (measured with `hf_bench_lib.AsDeviceArrayCounter`).
* Vulkan `device_id` is a queried UUID hash, so `Device(kind, index)` must map through
  `query_available_devices()[index]` — the single behaviour that is not a string
  whitelist and the one a future first-class etl vulkan kind would have to model.
* `iree-base-compiler` and `iree-base-runtime` must be the **same** 3.11 series;
  compiled artifacts are version-specific (clear any stale build cache after an
  upgrade).
