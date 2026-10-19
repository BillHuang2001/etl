# IREE Vulkan on this host: no AMD-GPU Vulkan path exists; lavapipe (software) runs, and `etl_hf` runs on it

> **Status.** `vulkan` is a **first-class `etl` device kind**, so the library work this document
> originally asked for (§3) is implemented and the benchmark legs run the **unmodified public API**.
> There is no runtime shim: [`./hf_bench_vulkan.py`](./hf_bench_vulkan.py) holds device/provenance
> helpers (`VULKAN_KIND`/`VULKAN_TARGET`, `device_info`, `physical_device_properties`) plus a `--check`
> environment self-test, and nothing patches etl. The host verdict (§1), the device recipe (renamed
> lavapipe ICD, throwaway venv) and the measurements (§2, §4) are the record; §3 keeps the shim-era
> patch list as history.

## Verdict

**The AMD Instinct MI300X VF of this host cannot execute Vulkan, under any driver or configuration.**
Two independent Vulkan drivers refuse the device at *their own* policy/table level, and the amdgpu DRM
device is additionally wedged in a stuck reset, so nothing can initialize it even when a driver is
patched to accept it (§1). This is a host/driver fact, not an etl or IREE limitation.

**IREE's Vulkan backend does work here — on lavapipe, Mesa's *software* Vulkan implementation
(`deviceType = CPU`).** The whole `etl_hf` pipeline (trace → lower → `iree-compile
--iree-hal-target-backends=vulkan-spirv` → load → device-resident run) executes a real checkpoint's
forward pass as ONE graph with **zero** `asdevicearray` calls in the warm loop, bit-comparable to the
numpy interpreter (§2). IREE 3.11 hides lavapipe behind a hard-coded device-name denylist, so the only
host-side setup is a **renamed lavapipe ICD** pinned through `VK_ICD_FILENAMES` (§1d); `Device("vulkan",
0)` itself is an ordinary etl device — `etl.pipeline` infers `target_backends=["vulkan-spirv"]`, the
iree adapter maps the etl index onto IREE's own enumeration and load/run/staging/transfer treat it as
a device-resident kind, and `Tensor.to(Device("vulkan", N))` has a registered transfer provider (§3).

**Seven HF benchmark cells are measured on the Vulkan path under two legs** — two cells run on the
library defaults (`opt_level="O3"`, `vulkan_target` unset), six on the one hand-picked option
`opt_level="O1"` (`tiny_mistral` fails the exact-`erf`-GELU activation gate in both, so 14 Vulkan
cells in total: **8 ok, 6 not ok**) (§4, and section 5c of
[`./hf_bench_results.md`](./hf_bench_results.md)). Every number is a **software-Vulkan / CPU**
number: the same work runs on the CPU through a Vulkan driver, and on the 124M-class models it is far
slower than the numpy interpreter. No cell moves work to the GPU.

---

## 1. Feasibility: the AMD GPU has no Vulkan path on this host

### 1a. Mesa RADV refuses CDNA3 by policy

`radeon_icd.json` → `libvulkan_radeon.so` (Mesa 25.2.8) reports **0 devices**, so
`vkEnumeratePhysicalDevices` fails with `VK_ERROR_INITIALIZATION_FAILED (-3)`. The loader debug log
names the cause:

```
ERROR: setup_loader_term_phys_devs: Failed to detect any valid GPUs in the current config
```

The silent drop is a hard policy gate in `src/amd/vulkan/radv_physical_device.c:2122`
(`radv_is_gpu_supported()`, called from `radv_physical_device_try_create()`):

```c
/* AMD CDNA isn't supported. */
if (info->gfx_level == GFX9 && !info->has_graphics)
   return false;
```

The MI300X VF matches it exactly (direct libdrm probe + RADV's own `RADV_DEBUG=startup` dump):

| value | source | result |
|---|---|---|
| `family_id = 141` (`AMDGPU_FAMILY_AI`), `device_id = 0x74b5`, `chip_external_rev = 71` (0x47) | `amdgpu_query_gpu_info` | `identify_chip()` → `CHIP_GFX940`, name `GFX940` |
| `amdgpu_query_hw_ip_info(type=GFX).available_rings = 0x0` | ibdrm | `ac_gpu_info.c:557-560` `else continue;` → `ip[AMD_IP_GFX].num_queues = 0` |
| `has_graphics = num_queues > 0` | `ac_gpu_info.c:993` | `0` |
| `gfx_level = GFX9`, `has_graphics = 0` | — | `radv_is_gpu_supported()` → `VK_ERROR_INCOMPATIBLE_DRIVER` |

Verbatim (stock ICD, `RADV_DEBUG=startup` — the only debug flag that prints at this point):

```
radv: info: Created an instance.
radv: info: Found device '/dev/dri/renderD129'.
radv: info: device 'GFX940' is not supported by RADV.
vkCreateInstance -> 0
count-call -> -3 n= 0
```

The same code exists on mesa `main`, so no Mesa release changes this: RADV does not support CDNA
compute parts. There is no environment variable or loader setting that overrides it.

**It is bypassable in the binary (proven), which is how the rest was diagnosed.**
`patched/libvulkan_radeon_patched.so` is the stock `.so` with the six bytes of the
"unsupported" branch at file offset `0x137047` (`0f8410feffff` → `909090909090`) rewritten; with that
ICD the device enumerates and IREE 3.11's Vulkan HAL sees it:

```
count-call -> 0 n= 1
apiVersion=1.4.318 driverVersion=104865800 vendor=0x1002 device=0x74b5 type=2
name=AMD Instinct MI300X VF (RADV GFX940)
```

RADV's own dump of the patched device warns that the surface/addrlib path is untrustworthy
(`num_rb = 0`, `enabled_rb_mask = 0x0`, `GB_ADDR_CONFIG = 0x0` → mis-tiled pipes), so only
buffer/compute work would be sane on it.

### 1b. AMDVLK has no CDNA device path at all

The second AMD Vulkan driver was installed side-by-side without root
(`AMDVLK 2025.Q2.1`, `.deb` extracted with `dpkg-deb -x` into `$TMPDIR/vk-amdvlk/root`, ICD json
pointed at by `VK_ICD_FILENAMES`, no `LD_LIBRARY_PATH` needed). It also reports **0 devices**:

* PAL has no `AMDGPU_FAMILY_AI` / CDNA case in `src/core/device.cpp:212-249`
  `DetermineGpuIpLevels()` → `Result::Unsupported`.
* `src/core/os/amdgpu/amdgpuPlatform.cpp` `Platform::ReQueryDevices()` then deletes the GPU entry
  silently, so `vkCreateInstance` returns `VK_ERROR_INCOMPATIBLE_DRIVER (-9)` through the loader.
* IREE surfaces it as `vulkan_driver.cc:251 FAILED_PRECONDITION; VK_ERROR_INCOMPATIBLE_DRIVER;
  vkCreateInstance: invalid instance configuration`.

So the two independent AMD Vulkan drivers both refuse CDNA3: RADV by explicit policy, AMDVLK because
its PAL backend never learned the IP family. **No AMD-GPU Vulkan driver exists for this part.**

### 1c. Even a patched RADV cannot initialize: the amdgpu DRM device is wedged

With the patched ICD the next step dies in libdrm, before RADV gets a device:

```
_amdgpu_device_initialize: amdgpu_query_gpu_info_init failed
```

The chain (libdrm 2.4.125 + the **DKMS** amdgpu 6.16.6 module; the kernel itself is `6.8.0-142-generic`):

* `amdgpu/amdgpu_gpu_info.c:215-218` reads `AMDGPU_INFO_READ_MMR_REG` (`0x263e`, `gb_addr_cfg`)
  because `family_id (141) >= AMDGPU_FAMILY_AI` skips the per-SE path.
* `drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c:812-824`:
  `if (!down_read_trylock(&adev->reset_domain->sem)) return -ENOENT;` — the reset-domain **write**
  lock is held, i.e. a GPU reset/recovery is in progress and never completes.
* libdrm then takes its cleanup arm `close(dev->fd)` (`amdgpu_device.c:267-269`), and **`close()`
  never returns** — a 5 s `SIGALRM` fires inside `close(2)` in the minimal probe.

Minimal reproducer (`mmr.c`, one ioctl per line; same on `/dev/dri/renderD129` and `/dev/dri/card1`):

```
open /dev/dri/renderD129 -> 3
ACCEL_WORKING       query=0x00 -> r=0 errno=0 (Success)
DEV_INFO            query=0x16 -> r=0 errno=0 (Success)
READ_MMR_REG 0x263e query=0x15 -> r=-1 errno=2 (No such file or directory)
READ_MMR_REG 0x0    query=0x15 -> r=-1 errno=2 (No such file or directory)
MEMORY              query=0x19 -> r=0 errno=0 (Success)
VRAM_GTT            query=0x14 -> r=0 errno=0 (Success)
TIMESTAMP           query=0x05 -> r=0 errno=0 (Success)
closing fd (may block)...        <- SIGALRM(5s) fired INSIDE close(2)
```

Corroboration that this is device state, not a Mesa/libdrm bug:

* the **ROCm/KFD path is dead too** — `hipGetDeviceCount -> 100 cnt=0`
  (`hipErrorNoDevice`); `iree.runtime.get_driver("hip")` → `hip_driver.c:66 FAILED_PRECONDITION;
  HIP driver error 'hipErrorNoDevice' (100)`;
* sysfs-only readers still answer (`rocm-smi --showmeminfo vram` → 205 822 885 888 B total), and
  `amdgpu_fence_info` shows every ring's "last signaled == last emitted" — no hung ring is pending,
  the reset lock is simply never released;
* the state is persistent (re-checked over ~30 min) and `dmesg` is unreadable in this container
  (`Operation not permitted`), so the kernel-side reset cause cannot be quoted.

The DRM nodes open O_RDWR fine, which is why probes "look" healthy until the first mm-register read.
**A VF/VM-level reset (or a fresh instance) is required to get the GPU back; nothing in userspace can.**
All DRM/ROCm state was left untouched from here on.

### 1d. The only usable Vulkan device is lavapipe — which IREE 3.11 hides by name

Auditing every ICD manifest in `/usr/share/vulkan/icd.d/` (intel, nouveau, asahi, gfxstream, virtio,
radeon, lvp): only **lavapipe** yields a physical device:

| property | value |
|---|---|
| device name | `llvmpipe (LLVM 20.1.2, 256 bits)` |
| apiVersion | `1.4.318` |
| driverVersion | Mesa `25.2.8` |
| vendorID | `0x10005` |
| deviceType | `CPU` (**software** rasterizer) |
| `maxComputeSharedMemorySize` | `32768` |
| `maxComputeWorkGroupInvocations` | `1024` |
| `maxComputeWorkGroupSize` | `[1024, 1024, 1024]` |
| `maxComputeWorkGroupCount` | `[65535, 65535, 65535]` |
| `maxStorageBufferRange` | `134217728` (128 MiB) |
| extensions / features | 164 device extensions incl. `VK_KHR_buffer_device_address`, `VK_KHR_timeline_semaphore` v2, `VK_KHR_8bit/16bit_storage`; all core features true (`shaderInt64`, `shaderFloat64`, …) |

With the stock ICD, `iree.runtime.get_driver("vulkan").query_available_devices()` returns `[]`
— while the Vulkan loader's own debug log shows it enumerated that device. The reason is a
**hard-coded device-name denylist** in IREE 3.11's `vulkan_driver.cc`
(`iree_hal_vulkan_is_device_visible()`):

```c
if (strstr(deviceName, "llvmpipe") == deviceName) return false;
```

lavapipe's stock name starts with `llvmpipe`, so IREE filters it out with no flag/env override.
The workaround is a **copy** of `libvulkan_lvp.so` whose name string reads `lavapipe` instead
(`$TMPDIR/vk-lvp/lavapipe_renamed.so` + `lavapipe_icd.json`, byte patch at offset 4814272), selected
via `VK_ICD_FILENAMES`. A side benefit: pinning the ICD also stops the loader from `dlopen`ing the
**radeon** ICD, whose wedged device makes creation attempts hang.

### 1e. The IREE compiler side works

`iree-compile --iree-hal-target-backends=vulkan-spirv` compiles both a trivial StableHLO module and
real etl exports (including a two-512×512-matmul + tanh + reduce graph) for this target, and the
resulting `.vmfb` runs on the lavapipe HAL device. Flag spellings, verified from
`iree-compile --help` (3.11):

* `--iree-hal-target-backends=vulkan-spirv` (the target backend),
* `--iree-vulkan-target=<profile>` (the SPIR-V environment: `gfx*`, `sm_*`, `rdna3`, `valhall4`,
  `ampere`, `adreno`, `rx7900xtx`, `rtx4090`, …). There is **no** `--iree-hal-vulkan-target`.
  A bogus profile value crashes the compiler with an MLIR stack dump rather than a clean diagnostic.

Without `--iree-vulkan-target`, codegen uses the default profile `vp_android_baseline_2022`, whose
budget is the binding constraint for large graphs (§4c):

```
#iree_gpu.target<arch = "vp_android_baseline_2022", features = "spirv:v1.3,cap:Shader",
  wgp = <compute = fp32|int32, storage = b32, subgroup = none, subgroup_size_choices = [64],
  max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
  max_workgroup_memory_bytes = 16384, max_workgroup_counts = [65535, 65535, 65535]>>
```

Raising it is possible via a real profile (`adreno`/`valhall*` 32768 B, `rdna3`/`rdna4`/`gfx942`/
`gfx1100` 65536 B, `sm_80`/`ampere` 166912 B) — but that is *not* what the benchmark legs do (they
keep the library default, i.e. no profile flag), and pushing the limit merely moves the failure to
the vector-dialect legalization (`vector.step`/`vector.extract`) rather than producing a running module.

### 1f. Verdict on the AMD GPU

| route | outcome |
|---|---|
| RADV (`radeon_icd.json`) | 0 devices — `radv_is_gpu_supported()` rejects CDNA3 by policy |
| RADV, binary-patched | enumerates, then fails init — the amdgpu DRM device is in a stuck reset (`READ_MMR_REG` → `ENOENT`, `close()` blocks) |
| AMDVLK 2025.Q2.1 (side-by-side) | 0 devices — PAL has no `FAMILY_AI`/CDNA IP path |
| ROCm/HIP (for reference) | dead — `hipErrorNoDevice`; the KFD path shares the wedged device |
| lavapipe (**software**) | works — the device all numbers below were measured on |

So: **no GPU Vulkan execution is possible on this host**, and the honest way to satisfy the Vulkan
mandate is the software device, which is what the committed `iree-vulkan` legs measure.

---

## 2. Execution through the public `etl_hf` API (proof)

`etl_hf`'s unmodified public API — `load_hf` → `forward_fn` → `pack_plan`/`pack_params` →
`build(backend="iree", device=Device("vulkan", 0))` → `place_params`/`place_ids` → `run` in a warm
same-device loop — runs a real checkpoint's whole forward as ONE graph on IREE's Vulkan HAL, with no
option, monkeypatch or harness helper involved. The committed `iree-vulkan` record for `tiny_llama`
(`hf-internal-testing/tiny-random-LlamaForCausalLM`), `seq_len=32`, batch 1, unpacked weights,
warmup 2 + best-of-5, one subprocess per cell:

| metric | value |
|---|---|
| build (trace → lower → compile → load, incl. `iree-compile`) | **913.7 ms** |
| first call | **7.720 ms** |
| **best warm ms/call** | **1.617 ms** (median 1.637, stdev 0.019) |
| throughput | 19 787 tok/s |
| +d2h (one explicit `Tensor.to(Device("cpu",0))` per call) | 1.555 ms best |
| `asdevicearray` calls in the warm loop | **0** (`zero_asdevicearray_after_warmup: true`) |
| host→device placement of the unpacked weight leaves (once) | 2.819 ms for 21 leaves (`place_ids` 0.059 ms) |
| parity vs the `numpy-cpu` leg's logits (the committed reference) | max_abs **8.94e-08**, rel-to-absmax 2.14e-07, argmax **1.0000** |

fp32 accumulation order differs from numpy's ⇒ **not** bit-exact; ~1e-7 absolute at absmax 0.42 with
full argmax agreement is the expected fp32 result and matches what every other etl backend shows on
this checkpoint. (The torch/transformers golden fixture is `seq_len=8`, so this `seq_len=32` cell has
no golden comparison — the report renders `—` with that reason and points at the `numpy-cpu`/torch
legs instead.)

Logits stay on the device in an `IreeDevicePayload` (`out.device` = `Device(kind='vulkan', index=0)`,
shape `[1, 32, 32000]`, float32); exactly one explicit `.to(cpu)` produced the host array used for
parity. Device-resident semantics hold unchanged: host inputs are rejected at the run boundary,
results are never auto-copied, and the documented same-device loop costs **0** `asdevicearray` calls
(measured with `hf_bench_lib.AsDeviceArrayCounter`).

The one Vulkan-specific device behaviour the kind models: the IREE `vulkan` `device_id` is a
**queried per-process UUID hash** (observed 941009984, 588459008, 1052880608, 998819584, … in
different processes), never `index + 1` as on CUDA/ROCm — so `Device(kind, index)` maps through
`query_available_devices()[index]` (the adapter's `_acquire_vulkan_device`) and one HAL device is
cached per `(index, runtime args)` so uploads and the module load share it.

---

## 3. The `vulkan` device kind: what it needed, and where the library provides it

`vulkan` is a first-class `etl` device kind; §3a/§3b are the **history** of what a benchmark-only
shim had to widen before the kind existed, and §3c maps each of those capabilities onto the library
code that owns it now. Library-level documentation: [`../etl/backends/adapters/CONTEXT.md`](../etl/backends/adapters/CONTEXT.md)
(the iree adapter's Vulkan section: device acquisition, load gate, staging tier, transfer slots, the
`vulkan_target` compile option) and the skills
[`.agents/skills/etl-vulkan-device-recipe.md`](../.agents/skills/etl-vulkan-device-recipe.md) /
[`.agents/skills/etl-compiler-adapter-recipes.md`](../.agents/skills/etl-compiler-adapter-recipes.md).

### 3a. What the library raised before the kind existed (historical, verbatim)

Reproduced at the time with `etl_hf.load_hf` → `forward_fn`:

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

Four independent *string* whitelists (compile targets, load device kinds, transfer providers, device
enumeration) plus a device→target inference gap and a device-id convention blocked the path. Each is
now part of the kind (§3c); the raw `--iree-hal-target-backends` escape hatch remains deliberately
denied by the adapter (`BackendError: the iree backend denies the iree-compile flag
'--iree-hal-target-backends=llvm-cpu': use the 'target_backends' compile option instead`), because the
target is derived from the device.

### 3b. What the retired shim patched (historical)

The retired shim ran `install()` (idempotent, after `etl` was importable and before the executable was
built) and returned `{site: "what changed"}` (also `patched_sites()`), with `device_info()` as
enumerated-device evidence. None of that exists in the current module, which holds the two
device/provenance helpers and `--check` only. The sites the shim had to widen, kept as the historical
record:

`etl/backends/adapters/iree.py`

| site (at shim time) | library behaviour then | shim |
|---|---|---|
| `:1071` `IreeBackend._resolve_compile_options` | `supported_targets = {"llvm-cpu","cuda","rocm"}` rejected `vulkan-spirv` (`:1074-1079` message) | **wrapped** — a vulkan-only target list was resolved by the shim (the library's own `opt_level` → `--iree-opt-level=O{N}` + `iree_compile_args` resolution), a mixed list re-attached `vulkan-spirv`, unknown targets delegated |
| `:460` `_artifact_target` | target→label map; `vulkan-spirv` fell through to a `"+".join` label | **wrapped** → label `"vulkan"` |
| `:486` `_GPU_DEVICE_KINDS` | `frozenset({"cuda","rocm"})` selected the `rt.Config(device=…)` load path and the non-host input branch (also used at `:532`, `:607`, `:2231`, `:2820`) | module global **widened** with `"vulkan"` |
| `:536` `_acquire_runtime_device` | `device_id = device.index + 1` (CUDA/ROCm 1-based ordinal) | **wrapped** — vulkan created the device from the **queried** `query_available_devices()[index]` info dict and cached one HAL device per `(index, runtime args)` |
| `:1817` `_staged_input` | non-ROCm branch was the CUDA pinned/DMA staging tier | **wrapped** → `None` for vulkan (plain `rt.asdevicearray`; the CUDA ordering guarantee is unverified on lavapipe) |
| `:1326` / `:1346` `IreeBackend.load` | device-kind whitelist `("cpu","cuda","rocm","hip")`; `expected_target = {…}[device_kind]` raised `KeyError` | **wrapped** — for a vulkan device the wrapper replayed the adapter's own load steps (`check_available` / `validate_options` / `_apply_iree_runtime_args` / `base64.b64decode` / `_acquire_runtime_device` / `_load_vm_module` / `IreeExecutable`) with those two gates widened; every other device delegated unchanged |
| `:2759` `upload_tensor` (with the `:2818-2820` kind gate) | placement provider for CUDA/ROCm/HIP only | the adapter's own `upload_tensor` **registered** for `"vulkan"` in both the per-backend `("vulkan","iree")` slot and the flat default slot |
| `:2907-2908` `register()` | `for kind in ("cuda","rocm","hip")` | same as above — the shim performed the registration for `"vulkan"` |

`etl/pipeline.py`

| site (at shim time) | library behaviour then | shim |
|---|---|---|
| `:360-412` `_resolve_backend_device` | device→`target_backends` inference covered `cuda`/`rocm`/`hip`/`cpu` (`:405-411`); a vulkan device compiled `llvm-cpu` and then failed the load-time target check | **wrapped** — an iree backend on a vulkan device inferred `target_backends=["vulkan-spirv"]` when neither the option nor `ETL_TARGET_BACKENDS` was set |

### 3c. Where each capability lives in the library now

Every item below is library behaviour, exercised by the HF legs through the public API only:

| capability | library owner |
|---|---|
| device→`target_backends` inference (`vulkan` → `["vulkan-spirv"]`), and NO target flag unless the `vulkan_target` option is set | `etl/pipeline.py` `_resolve_backend_device`; the etl_hf-side twin is `etl_hf/pipeline.py::_targets_from_device` |
| compile-target whitelist (`{"llvm-cpu","cuda","rocm","vulkan-spirv"}`), artifact label `"vulkan"`, and the `vulkan_target` → `--iree-vulkan-target=<profile>` option | `etl/backends/adapters/iree.py` `IreeBackend._resolve_compile_options` / `_artifact_target` / the options table |
| device-resident semantics: `_GPU_DEVICE_KINDS` includes `vulkan`, the load gate accepts it, the artifact↔device table maps `vulkan-spirv` ↔ `vulkan`, and `_staged_input` returns `None` (no CUDA pinned/DMA tier on lavapipe) | `etl/backends/adapters/iree.py` (`_GPU_DEVICE_KINDS`, `IreeBackend.load`, `_expected_target_for_kind`, `_staged_input`), shared with `_load_segments` (external-kernel split path) |
| device acquisition by **queried device info** (`query_available_devices()[index]`, one cached HAL device per `(index, runtime args)`), not the CUDA/ROCm `index + 1` ordinal | `etl/backends/adapters/iree.py` `_acquire_vulkan_device` / `_acquire_runtime_device` |
| `Tensor.to(Device("vulkan", N))` — the adapter's `upload_tensor` in the per-backend `("vulkan","iree")` slot AND the flat DEFAULT slot (the one documented exception: nothing else serves a vulkan device) | `etl/backends/adapters/iree.py` `register()`, `etl/core/device.py` provider resolution |
| `etl.devices("vulkan")` enumeration through `iree.runtime.get_driver("vulkan")` (lazy iree import) | `etl/core/device.py` `devices()` / `_vulkan_devices()` |
| `etl.io.prefetch`/`sink` on a vulkan device (driver name `vulkan`, device-id mapping, the worker-thread context bootstrap) | `etl/io.py` |
| `python -m etl.bench --device vulkan:0` accepted, treated as a device-residency kind and never labelled GPU hardware | `etl/bench/_util.py` `_SUPPORTED_DEVICE_KINDS` |

Unchanged and deliberately so: `core/tensor.py` (provider resolution is kind-agnostic),
`etl/backends/compiler.py`, and the StableHLO exporter — the exported StableHLO compiles for
`vulkan-spirv` as-is.

### 3d. The two constraints that are host/compiler side, not etl side

1. **The default SPIR-V profile is the binding constraint for large graphs.** `vulkan-spirv` without
   `--iree-vulkan-target` uses `vp_android_baseline_2022` (16 KiB workgroup memory, 128 threads),
   which cannot compile the 124M-class HF graphs at `opt_level="O3"` (§4c). The `vulkan_target` compile
   option exists for that (a real-GPU / larger profile); the benchmark legs deliberately do not set it.
2. **IREE 3.11's `llvmpipe` name denylist** (`iree_hal_vulkan_is_device_visible()` in
   `vulkan_driver.cc`) hides software devices, so lavapipe is only enumerable through the renamed ICD
   (§1d) — a runtime-side request, not an etl one.

---

## 4. Benchmark: `iree-vulkan` and `iree-vulkan-o1` (lavapipe, software Vulkan on the CPU)

Both legs run the **unmodified public API** — `etl_hf.build(fwd, cfg, backend="iree", device=Device("vulkan", 0))`
and then `etl_hf.place_params`/`place_ids` + `etl_hf.run` in the device-resident same-device loop. A
leg passes **no** `target_backends` option: `etl.pipeline` infers `target_backends=["vulkan-spirv"]`
from the device. The `vulkan_target` compile option is **not** set, so codegen keeps IREE's default
SPIR-V profile `vp_android_baseline_2022` (`max_workgroup_memory_bytes = 16384`). The two legs are
separate on purpose: `iree-vulkan` uses the library defaults (`opt_level` as resolved by
`etl_hf.pipeline.resolve_build_options` — `O3` for a non-ROCm target set), `iree-vulkan-o1` passes
the one hand-picked option `opt_level="O1"`, and each record marks which
(`compile_options_hand_picked` / `compile_options_default` / `compile_options_note`).

Both legs are integrated in the harness ([`./hf_bench.py`](./hf_bench.py), [`./hf_bench_lib.py`](./hf_bench_lib.py)),
one subprocess per `(model, seq_len, leg)`, records in [`./hf_bench_data/`](./hf_bench_data/)
(`*__iree-vulkan.json`, `*__iree-vulkan-o1.json`), rendered in **section 5c** of
[`./hf_bench_results.md`](./hf_bench_results.md).

The committed matrix is **65 cells (54 ok, 11 not ok) over 11 legs**; the Vulkan subset is **14 cells
(8 ok, 6 not ok)** — 7 model/seq cells × 2 legs, with `tiny_mistral` failing on both. Every ok cell
records `asdevicearray_calls == 0` / `zero_asdevicearray_after_warmup: true`, logits on
`Device(kind='vulkan', index=0)`, `vulkan.public_api: true`, the inferred target backend and the
`compile_options_*` pair.

Run them with `--phase vulkan`, which requires the environment to name the ICD:

```bash
export ETL_VULKAN_ICD=$TMPDIR/vk-lvp/lavapipe_icd.json      # never hard-coded in the repo
export ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH=$PWD $TMPDIR/vk-venv/bin/python bench_logs/hf_bench.py \
  --phase vulkan --seq-lens 32,128 --out bench_logs/hf_bench_data --warmup 2 --repeat 5
```

`child_env` forwards `ETL_VULKAN_ICD` to the leg process as `VK_ICD_FILENAMES`; a vulkan leg
requested without it writes a record naming the variable instead of a silently empty device list.
Each record carries a `vulkan` provenance block — device name / apiVersion / driverVersion (read from
`VkPhysicalDeviceProperties` via `hf_bench_vulkan.physical_device_properties()`), vendorID,
`deviceType`, IREE device id/path, the ICD used, `public_api: true`, `etl_device`
(`Device(kind='vulkan', index=0)`), the inferred `target_backends`, `compile_options_default`
(`opt_level` default + the default SPIR-V profile) and `compile_options_hand_picked` (plus whether
`vulkan_target` was used), and an explicit `"note": "lavapipe = software Vulkan on CPU; not a GPU
measurement"`. The report's GPU columns are **reused records** measured on the real GPU before it
stopped accepting work — reference data, never a comparison with the software leg.

### 4a. `iree-vulkan` (library defaults, `opt_level="O3"`, `vulkan_target` unset): 2 of 7 cells run

| model | seq | weights | first ms | best ms | median ms | +d2h best | `asdevicearray` | ×numpy-cpu | ×iree-llvmcpu | ×torch-cpu |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny_llama | 32 | unpacked | 7.720 | **1.617** | 1.637 | 1.555 | 0 | 0.95× | 1.32× | 0.56× |
| tiny_gpt2 | 32 | unpacked | 5.609 | **1.744** | 1.767 | 2.292 | 0 | 0.69× | 1.04× | 0.48× |
| tiny_mistral | 32 | unpacked | — | — | — | — | n/a | — | — | — |
| distilgpt2 | 32 | packed | — | — | — | — | n/a | — | — | — |
| distilgpt2 | 128 | packed | — | — | — | — | n/a | — | — | — |
| gpt2 | 32 | packed | — | — | — | — | n/a | — | — | — |
| gpt2 | 128 | packed | — | — | — | — | n/a | — | — | — |

Build totals 913.74 ms (`tiny_llama`) and 695.01 ms (`tiny_gpt2`); throughput 19 787.29 and
18 349.68 tok/s. The `×` columns are CPU-vs-CPU (that leg's best ÷ this leg's best, so >1 means
Vulkan is faster): lavapipe sits at numpy-interpreter level (0.95× / 0.69× numpy — the interpreter is
the faster of the two on `tiny_gpt2`) and 4–32 % faster than `iree-llvmcpu` (1.32× / 1.04×), which on
a software rasterizer is the expected outcome and means these cells carry no GPU information.

The five non-running cells are recorded verbatim, never omitted:

* `tiny_mistral` s32 — `HFUnsupportedError: hidden_act='gelu' needs etl.erf, which the StableHLO
  exporter defers in v1, so it cannot run on backend 'iree'` (the same activation gate every compiled
  leg of this checkpoint hits; the numpy and torch legs run normally).
* the four 124M-class cells — `iree-compile` diagnostics, see §4c.

Parity vs the numpy interpreter: `tiny_llama` s32 max_abs **8.94e-08** / rel-to-absmax 2.14e-07 /
argmax 1.0; `tiny_gpt2` s32 max_abs **2.24e-08** / rel 1.73e-07 / argmax 1.0.

### 4b. `iree-vulkan-o1` (hand-picked `opt_level="O1"`): 6 of 7 cells run

| model | seq | weights | first ms | best ms | median ms | +d2h best | `asdevicearray` | ×numpy-cpu | ×iree-llvmcpu | ×torch-cpu |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny_llama | 32 | unpacked | 7.225 | **1.698** | 1.712 | 1.520 | 0 | 0.90× | 1.25× | 0.54× |
| tiny_gpt2 | 32 | unpacked | 6.073 | **1.900** | 1.936 | 2.274 | 0 | 0.64× | 0.95× | 0.44× |
| tiny_mistral | 32 | unpacked | — | — | — | — | n/a | — | — | — |
| distilgpt2 | 32 | packed | 189.931 | **139.455** | 142.725 | 1.905 | 0 | 0.11× | 1.82× | 0.06× |
| distilgpt2 | 128 | packed | 11 528.31 | **11 477.25** | 11 495.23 | 7.221 | 0 | 0.01× | — | 0.00× |
| gpt2 | 32 | packed | 249.409 | **194.588** | 206.498 | 2.013 | 0 | 0.15× | 2.07× | 0.08× |
| gpt2 | 128 | packed | 11 617.58 | **11 559.80** | 11 594.93 | 6.930 | 0 | 0.01× | — | 0.00× |

Build totals: 910.61 / 695.10 / 1 842.97 / 1 806.63 / 3 179.42 / 3 110.12 ms for the six built cells;
throughput 18 843.48 / 16 841.22 / 229.46 / 11.15 / 164.45 / 11.07 tok/s in the same order.
`tiny_mistral` fails on the erf-GELU activation gate in this leg too, so the leg runs 6 of the 7 cells.

Parity vs the numpy interpreter (all with **argmax agreement 1.0**), plus the torch/transformers
golden where a fixture of the cell's `seq_len` exists:

| model | seq | max_abs | rel-to-absmax | max_abs vs golden |
|---|---|---|---|---|
| tiny_llama | 32 | 8.94e-08 | 2.14e-07 | — (golden fixture is `1x8`) |
| tiny_gpt2 | 32 | 3.30e-04 | 2.56e-03 | — (no `seq_len=32` golden) |
| distilgpt2 | 32 | 2.441e-04 | 2.97e-06 | 2.44e-04 |
| distilgpt2 | 128 | 2.594e-04 | 3.16e-06 | — |
| gpt2 | 32 | 3.662e-04 | 2.64e-06 | 4.27e-04 |
| gpt2 | 128 | 4.196e-04 | 3.02e-06 | — |

The real models' absolute deviation grows with logit magnitude (absmax ~139 for `gpt2`), while
relative-to-absmax stays at the fp32 accumulation-order floor (~3e-06) — the same order every other
etl backend shows, with **no** SPIR-V fast-math divergence and no argmax disagreement.

**A `1e-4`-magnitude Vulkan parity value is fp32 accumulation-order tiling variance**, not a precision
or code-generation change: lavapipe splits reductions and element-wise work across threads, so the
summation order — and therefore the rounding — can differ, while `argmax` agreement stays 1.0. The
`tiny_gpt2` s32 `iree-vulkan-o1` cell is **bimodal across processes**: 7 replicate processes with an
identical command give 5 runs at `max_abs = 6.258e-07` and 2 runs at `3.304e-04` (bit-stable within a
run, `argmax_agreement = 1.0` in both modes, the divergence covering ~15 % of the vocabulary and
growing along the sequence). The committed record is one sample of that distribution. The same model's
`iree-vulkan` (defaults) cell is stable at 2.234e-08, and `distilgpt2`/`gpt2` reproduce bit-identically
across replicates at both seq lens — the records hold one number per cell, so the replicate spread is
documented here rather than in a table.

Reading the numbers honestly: under `O1` the 124M-class graphs *run* but are slow — 139–195 ms at
`seq_len=32` (0.11–0.15× the numpy interpreter, 1.82–2.07× `iree-llvmcpu`) and ~11.5 s at `seq_len=128`
(~0.01× numpy). The `seq_len=128` cliff is a codegen-quality problem, not a device limit: `O1` leaves
the 50257-wide vocabulary matmul and the packed gather/copy loops unoptimized, so a software rasterizer
executes them as scalar SPIR-V work. These are recorded as measured, with the hand-picked level
labelled in every record and in the report; they are **not** comparable with the `iree-vulkan`
defaults cells as a compile-setting-free measurement.

### 4c. Why the four 124M-class cells fail under the library-default compile level (verbatim)

`distilgpt2`/`gpt2` at `seq_len=32` (locations `<stdin>:155:12` and `<stdin>:251:12`) — the default
target profile's 16 KiB workgroup-memory budget:

```
<stdin>:155:12: error: function 'main_dispatch_0_reduction_32x768_f32' uses 196864 bytes of
shared memory; exceeded the limit of 16384 bytes
    %154 = "stablehlo.reduce"(%149, %150) ({
```

(the LayerNorm-shaped reduction materialized a 32×768 fp32 staging buffer ≈ 96 KiB per workgroup).
Passing a larger-profile `--iree-vulkan-target` (e.g. `gfx942`, 64 KiB) moves the wall: the reduction
then legalizes and compilation stops later in vector-dialect lowering.

`distilgpt2`/`gpt2` at `seq_len=128` (locations `<stdin>:124:12` and `<stdin>:220:12`) — the same
graphs instead die in legalization, before the memory budget matters:

```
<stdin>:220:12: error: failed to legalize operation 'vector.from_elements' that was explicitly
marked illegal: %1201 = "vector.from_elements"(%562, %564, %566, %568) : (i64,i64,i64,i64)
-> vector<4xi64>
    %217 = "stablehlo.gather"(%61, %216) {…} : (tensor<1024x768xf32>, tensor<1x128x1xi64>)
    -> tensor<1x128x768xf32>
```

`opt_level="O1"` avoids both (the aggressive vectorization passes that create the illegal form run
above `O1`); `O2`/`O3` both fail identically. So the leg split is a compiler limitation of
`vulkan-spirv` codegen on this graph shape, not a lavapipe constraint.

### 4d. Low-level runtime probe (512×512 matmul + tanh + variance; historical, before the public-API proof)

Before the public-API proof, the same Vulkan path was driven through `iree.runtime` directly on an
etl-exported graph (`etl.backends.stablehlo.export` → `iree-compile --iree-hal-target-backends=vulkan-spirv`
→ load on the lavapipe HAL device), which isolates the runtime from the etl adapter:

| leg | best ms/call | parity vs f64 numpy |
|---|---|---|
| lavapipe `vulkan-spirv`, host-staged inputs | 1.74 ms | max_abs 2.96e-04 |
| lavapipe `vulkan-spirv`, device-resident | 2.66 ms | max_abs 2.96e-04 |
| `iree-llvmcpu` (`--iree-llvmcpu-target-cpu=host`) | 4.09 ms | max_abs 2.96e-04 |
| `iree-llvmcpu` (generic CPU) | 243 ms | max_abs 2.96e-04 |
| numpy interpreter | 4.25 ms | — |

---

## 5. Notes and caveats

* **lavapipe is a software Vulkan device on the CPU.** Every `iree-vulkan`/`iree-vulkan-o1` number is
  a Vulkan-path correctness/performance datapoint, never a GPU number: ~1.6–1.9 ms/call for a tiny
  model is a plausible software-rasterizer cost, and 139 ms – 11.6 s for the 124M checkpoints shows
  the cost of unoptimized scalar SPIR-V. The architectural result is that the whole **public** staging
  pipeline is device-kind-generic: `vulkan` is a first-class device kind serviced entirely by the
  library (§3c), with no benchmark-side patch involved.
* **The GPU columns in the report are reused, not re-measured** — `iree-rocm`, `xla-rocm`,
  `xla-rocm-fp32`, `torch-rocm` (and the CPU legs `numpy-cpu`/`iree-llvmcpu`/`torch-cpu`) are the
  committed records from before the AMD GPU stopped accepting work. No ratio is formed between a GPU
  column and a software-Vulkan column.
* **ROCm/HIP has not recovered** — `iree.runtime.get_driver("hip").query_available_devices()` still
  fails with `hip_driver.c:66 FAILED_PRECONDITION; HIP driver error 'hipErrorNoDevice' (100): no
  ROCm-capable device is detected`, i.e. the same wedged device as §1c, so no GPU column can be
  re-measured. No DRM/kernel recovery is attempted.
* **Determinism**: within one process a cell's logits are bit-stable across warm iterations; across
  processes the fp32 accumulation order can differ on this threaded software device — `tiny_gpt2` s32
  `iree-vulkan-o1` is bimodal (§4b), every other cell reproduces bit-identically.
* **`iree-base-compiler` and `iree-base-runtime` must be the same 3.11 series** (validated pair
  `3.11.0`); compiled artifacts are version-specific, so clear stale build caches after an upgrade.
* **Python 3.14 + nanobind**: `iree.runtime` can emit a teardown dump at interpreter exit; anything
  that must exit cleanly flushes and calls `os._exit(0)` (`hf_bench_vulkan.py --check` does).
* **Never touch `/dev/dri`, ROCm or HIP on this host** — the wedged device makes `close()` and
  HIP init hang; wrap any DRM probe in `timeout`. Pinning `VK_ICD_FILENAMES` to the lavapipe ICD is
  also what keeps the loader away from the radeon ICD.
* The renamed lavapipe ICD, the patched RADV `.so`, AMDVLK and the venv are all **machine-local
  scratch** under `$TMPDIR` (never repo dependencies, never hard-coded paths in the repo).

## 6. Reproduce

```bash
export VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json
export ETL_HF_MODELS_DIR=$TMPDIR/etl_hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$PWD                       # etl and etl_hf from the checkout, never installed
$TMPDIR/vk-venv/bin/python bench_logs/hf_bench_vulkan.py --check      # device + public-path evidence, no model
```

`$TMPDIR/vk-venv` is a throwaway venv holding only `iree-base-compiler` + `iree-base-runtime` 3.11 +
numpy (CPython 3.14.8); `etl`/`etl_hf` come from the checkout via `PYTHONPATH`.

The end-to-end proof (also the shape of the harness's etl legs):

```python
import etl, etl_hf

loaded = etl_hf.load_hf("$ETL_HF_MODELS_DIR/tiny_llama")
cfg, params = loaded.config, loaded.params
fwd = etl_hf.forward_fn(cfg, seq_len=32, pack_weights=True)
dev = etl.Device("vulkan", 0)
exe = etl_hf.build(fwd, cfg, backend="iree", device=dev)             # ~0.91 s
plan = etl_hf.pack_plan(etl_hf.param_specs(cfg))
placed = etl_hf.place_params(etl_hf.pack_params(plan, params), dev)  # ~1.3-2.8 ms, once
ids = etl_hf.place_ids(input_ids, dev)
out = etl_hf.run(exe, placed, ids)    # 7.7 ms first call, then ~1.6-1.7 ms/call,
                                      # 0 asdevicearray calls in the warm loop
host = out.to(etl.Device("cpu", 0)).numpy()        # the one explicit D2H copy
```

**The in-package conformance harness accepts the kind too.** `PYTHONPATH=$PWD
VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json python -m etl.bench --backend iree --device vulkan:0
--examples matmul,op_gather_scatter,elementwise_fusion --repeats 3` prints `overall: PASS (3/3 checked)`
and exits 0 — `matmul` max_abs 3.81e-06 / max_rel 1.61e-04, `op_gather_scatter` and
`elementwise_fusion` bit-exact (0 / 0); benchmark best-of-3 0.201 / 0.238 / 0.212 ms vs numpy
0.004 / 0.003 / 0.005 ms (lavapipe launch/overhead on a software device — not a GPU number). No
deferral and no harness flag: `etl.bench` lists `vulkan` in its supported device kinds. The full
98-example matrix was not run on it.
