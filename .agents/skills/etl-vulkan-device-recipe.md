---
name: etl-vulkan-device-recipe
description: Machine-local recipe for validating etl's first-class "vulkan" device kind (IREE Vulkan/SPIR-V HAL) — throwaway 3.11 iree venv, the lavapipe software-device denylist workaround (renamed ICD), target-profile compile notes, and the run/env verification probes. Use when implementing or testing anything on Device("vulkan", N), or when a vulkan build/run fails on a machine with no usable Vulkan GPU. Companion to etl/backends/adapters/CONTEXT.md (Known Issue 16).
---
# etl "vulkan" device recipe (machine-local)

## What the kind is
- `"vulkan"` = **IREE's Vulkan/SPIR-V HAL**, served by the **iree adapter only** (`numpy`/`xla`/`tvm` reject it with an explicit `BackendError`).
- Public path: `etl.build(fn, *specs, backend="iree", device=etl.Device("vulkan", 0))` / `etl.lower(g, backend="iree")`; compile target spelling `vulkan-spirv` (inferred from the device, or `target_backends=["vulkan-spirv"]`); artifact label `"vulkan"`; a vulkan device only runs a vulkan-spirv artifact (mismatch = explicit error, never a silent recompile).
- Transfers are explicit: `t.to(Device("vulkan", N))` upload (iree placement provider), `t.to(Device("cpu", 0))` download. Outputs stay device-resident; `.numpy()` on them raises. Same-device inputs pass through with zero host round-trips.
- Enumerate with `etl.devices("vulkan")` (`[]`, never raises, when iree/loader/device is absent). `etl.devices()` deliberately excludes vulkan.

## Environment (throwaway venv OUTSIDE the repo; run etl from the checkout)
```bash
uv venv $TMPDIR/vk-venv --python 3.14
uv pip install --python $TMPDIR/vk-venv/bin/python \
  "iree-base-compiler>=3.11,<3.12" "iree-base-runtime>=3.11,<3.12" numpy pytest
export PYTHONPATH=$PWD        # etl from the checkout — never install etl, never a venv inside the repo
```
`iree-base-compiler` and `iree-base-runtime` MUST be the same series (compiled artifacts are version-specific).

## Software-device gotcha (no usable GPU Vulkan)
IREE 3.11 hard-codes a device-NAME denylist (`strstr(name,"llvmpipe")==name` → hidden) with **no env override**, and Mesa lavapipe's name starts with `llvmpipe`. Workaround, machine-local scratch:
1. Copy lavapipe's `libvulkan_lvp.so` + its ICD json into `$TMPDIR/vk-lvp/`.
2. Byte-patch the device-name string `llvmpipe` → `lavapipe` in the copied `.so` (same length, so a plain in-place patch works).
3. Point the ICD json's `library_path` at the patched `.so`; export `VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json`.
**ALWAYS export it** — unset, the loader may dlopen a radeon ICD and HANG; wrap any Vulkan/DRM probe in `timeout`.
lavapipe is `deviceType=CPU`: provenance is "software Vulkan on the CPU", never a GPU number.

## Verify
```bash
VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json PYTHONPATH=$PWD \
  $TMPDIR/vk-venv/bin/python -c 'import etl; print(etl.devices("vulkan"))'   # must be non-empty
VK_ICD_FILENAMES=$TMPDIR/vk-lvp/lavapipe_icd.json \
  $TMPDIR/vk-venv/bin/python -c 'import iree.runtime as rt; print(rt.get_driver("vulkan").query_available_devices())'
```
End-to-end proof: build → run → outputs are device-resident `IreeDevicePayload` tensors on `Device("vulkan", 0)`; read back with `.to(cpu).numpy()` and compare against the numpy backend.

## Compile notes
- `--iree-hal-target-backends=vulkan-spirv`; there is **no** `--iree-hal-vulkan-target`. The optional SPIR-V profile is `--iree-vulkan-target=<profile>` = etl's `vulkan_target` compile option (env `ETL_IREE_VULKAN_TARGET`).
- Default profile `vp_android_baseline_2022` caps workgroup memory at 16384 B, so large graphs fail to compile until a bigger profile is passed (`rdna3`/`gfx*` 64 KiB, `adreno`/`valhall*` 32 KiB, `sm_80` 166912 B) — and the failure then moves to vector-dialect legalization. A bogus profile value crashes `iree-compile` with an MLIR dump, not a clean diagnostic.

## Run time
`Device("vulkan", N)` maps through `query_available_devices()[N]` — vulkan HAL device ids are queried per-process UUID hashes, NOT the cuda/rocm `index + 1` ordinal. One HAL device is cached per `(index, runtime_args)` in a thread-local cache shared by `load()` and `Tensor.to` (HAL devices are thread-affine).

## This box's host verdict
The AMD MI300X cannot run Vulkan at all: RADV policy-rejects CDNA3, AMDVLK has no CDNA path, and the amdgpu DRM device is wedged in a stuck reset (which also takes hip/ROCm and cuda down with it). So vulkan work here = lavapipe software path only; full evidence in `bench_logs/vulkan_findings.md`.

## Pointers
Adapter known issues (16 = vulkan denylist / hang / profile budget): `etl/backends/adapters/CONTEXT.md`. Binding public contract: `etl/CONTEXT.md` → "Device kinds (binding)". Longer CPU/GPU run recipes: skill `etl-compiler-adapter-recipes`.
