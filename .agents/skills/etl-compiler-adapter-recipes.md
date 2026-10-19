---
name: etl-compiler-adapter-recipes
description: Procedural run recipes, host<->device staging machinery details, version matrices, and measured performance/validation references for the etl compiler adapters (iree/xla/tvm) under etl/backends/adapters. Use when configuring a real GPU run (XLA PJRT plugin, iree cuda/rocm), debugging staging / hot-path performance, or needing the full validation narratives. Companion to etl/backends/adapters/CONTEXT.md.
---
# etl compiler-adapter recipes & reference
Long procedural recipes and measured references for the iree/xla/tvm adapters — kept out of
`etl/backends/adapters/CONTEXT.md` to stay under the per-file size limit. The CONTEXT.md holds the
current-state contract; this skill holds the bulky how-to and the numbers behind it.
The `Device("vulkan", N)` kind has its own recipe: see the `etl-vulkan-device-recipe` skill.
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
on a ROCm plugin is rejected (honest kind↔platform check).
The remainder of this file is unchanged from the previous revision: see git history if you need the full
staging machinery narrative, version matrices and measured validation/performance tables, which remain
authoritative in `etl/backends/adapters/CONTEXT.md` and the skills/bench_logs it points at.
