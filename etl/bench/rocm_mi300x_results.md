# etl.bench on AMD Instinct MI300X (ROCm 7.1) — conformance & performance report

Full `etl.bench` conformance + benchmark matrix for the 98-example registry on one AMD Instinct
MI300X VF, across four backend/device combinations, plus the TVM-on-ROCm probe.

Run date/host: single-tenant MI300X VF (the only GPU on the machine), seed 0, `repeats=20`,
`warmup=2`, default tolerances (`rtol=atol=1e-5`, per-example overrides where the registry sets
them — see `./CONTEXT.md`). Device always explicit: `rocm:0`.

- **GPU**: AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-`; ROCm 7.1.0 (driver/runtime
  7.1.25424, toolkit 7.14.60850, MIOpen DNN 1.3.0). PJRT reports 143.77 GiB HBM for its BFC
  allocator.
- **Env**: `.venv/bin/python` = Python 3.14.8, numpy 2.5.2, scipy 1.18.1, etl 0.1.0,
  iree-base(-runtime) 3.11.0, `jax-rocm7-pjrt` 0.11.2 (the PJRT plugin used by the xla backend),
  apache-tvm 0.26.0, torch 2.13.0+cu130, triton 3.7.1.

## Caveats that qualify every number below

1. **The torch numbers are CPU baselines, not GPU numbers.** `torch` here is a CUDA build with no
   CUDA device present (`torch.version.hip is None`, `torch.cuda.is_available() is False`), so the
   harness's `resolve_torch_device` degrades every torch reference to CPU. In the timing tables a
   column headed `torch-cpu` is a single-process CPU torch run measured on this host — it is *not*
   the MI300X. Use it only as a "what a mature CPU framework does" yardstick; it says nothing about
   torch-on-ROCm, which was not available for this run.
2. **`iree/rocm` per-call timings are dominated by host→device staging, not by device compute.**
   The harness places every input leaf with an explicit `Tensor.to(device)` transfer before each
   timed `run`. On `rocm` that transfer is the expensive part (~34 ms fixed + ~0.7–1.0 ms/MB — see
   "Root cause 1"). Results that look absurd for a 192 GB/s-class accelerator (e.g. a [64,64] matmul
   at 87 ms) are *staging* numbers. The device-resident run cost is measured separately below.
3. **The xla run was driven without torch comparisons** (`use_torch=False`): the corrected xla
   wrapper is conformance-first and the PJRT plugin's teardown abort makes long benchmark+torch runs
   awkward. The xla conformance numbers are complete; xla benchmark numbers are present for all 98
   examples (deferrals recorded as errors).
4. **Machine API level**: only the *lowest* ROCm call layer is in play for iree (the HIP HAL
   driver); the xla path uses a real gfx942 PJRT plugin. Results are specific to ROCm 7.1 + iree
   3.11 + `jax_plugins/xla_rocm7` 0.11.2.

## Per-backend conformance summary

| backend | device | conf examples | numeric PASS | numeric FAIL | deferral/error | crash (excluded) | lower+compile+load total | build n | bench examples |
|---|---|---|---|---|---|---|---|---|---|
| numpy / cpu | `cpu` | 98 | 98 | 0 | 0 | 0 | 1.1 s | 98 | 98 |
|---|---|---|---|---|---|---|---|---|---|
| iree / llvm-cpu | `cpu` | 98 | 75 | 0 | 23 | 0 | 20.1 s | 75 | 98 |
| iree / rocm:0 | `rocm:0` | 94 | 65 | 0 | 29 | 4 | 23.9 s | 66 | 94 |
| xla / rocm:0 | `rocm:0` | 98 | 46 | 13 | 39 | 0 | 6.0 s | 59 | 98 |

- `lower+compile+load total` is the sum over examples whose staging succeeded (deferred examples
  never reach a build, so they contribute no build time); see the build section.
- `iree / rocm:0` measured 94 of 98 examples: **4 examples were excluded because the process died
  mid-example** (SIGSEGV, `rc=-11`) and had to be skipped to protect the remaining results — see
  "Exclusions" below. Nothing was silently dropped.
- **Zero `torch_pass=False` anywhere**: every executed torch comparison (62 numpy / 46 llvm-cpu /
  45 rocm) passed. `torch_pass=None` on the rest = no torch reference for that example, or the
  example never got that far.

### Notable precision

- **numpy / cpu** — bit-exact to the pure-numpy references on every one of the 98 examples (as
  expected: shared kernels).
- **iree / llvm-cpu** — 0 numeric failures in 75 executed examples. Spot: `transformer` max_abs
  7.2e-4 (covered by its `tolerance=1e-3`), `matmul_1024` 2.06e-4 (its `atol=5e-4`), `attention`
  2.4e-7, `layernorm` 9.5e-7-class.
- **iree / rocm:0** — 0 numeric failures in 65 executed examples, and **25 of the 65 are bit-exact
  (max_abs exactly 0.0)**, including `matmul`, all three `cond_*`, `while_cumsum`,
  `while_cond_combo`, `cond_while_pipeline`, `kmeans_iter`, `op_*` elementwise families. Larger:
  `matmul_1024` max_abs 2.14e-4 (same fp32 accumulation class as llvm-cpu's 2.06e-4 — ROCm is
  fp32-accurate, no TF32), `transformer` 8.4e-4, `conv2d_large` 1.4e-4, `nbody` 7.6e-6,
  `layernorm_large` 9.5e-7, `attention` 2.4e-7. **The one thing that matters: iree-on-ROCm is
  numerically correct for every example it manages to run.**
- **xla / rocm:0** — **13 numeric FAILs, all dot/accumulation-heavy**, and all traced to a single
  cause: XLA's ROCm backend runs fp32 `dot` in **TF32** (10-bit mantissa). See "Root cause 2"; the
  failure magnitudes match a TF32 simulation of the same matmul *exactly* (0.193 vs 0.193 at
  n=1024, 0.0528 vs 0.0528 at n=256, 0.0158 vs 0.0158 at n=64). The 13: `transformer` (0.465),
  `matmul_square` (0.379), `mlp` (0.283), `matmul_1024` (0.123), `mha_block` (0.074),
  `matmul` (0.026), `grad_mlp` (0.020), `deep_mlp_block` (0.015), `batched_3d_matmul` (0.011),
  `xla_batched_dot` (0.011), `e2e_train_mlp` (0.0063), `gru_cell` (0.002), `attention` (0.0013).
  The other 46 xla passes include 26 bit-exact ones. XLA-on-ROCm therefore runs and is fast, but is
  **not fp32-accurate by default** — do not use it where the reference is a true fp32 result.

## Timing matrix — best-of-20 `etl_ms`, one column per backend

`ERR` = explicit deferral/error (recorded, never silent); `CRASH` = process death, excluded.
`torch-cpu` = CPU baseline (caveat 1). Ratio columns: `<1×` means ROCm is *slower*.

| example | numpy/cpu | iree/llvm-cpu | iree/rocm:0 | xla/rocm:0 | torch-cpu | rocm vs llvm-cpu | rocm vs numpy | rocm vs torch-cpu |
|---|---|---|---|---|---|---|---|---|
| matmul | 0.0237 | 0.136 | 87.4 | 0.272 | 0.0103 | 0.00155× | 0.000272× | 0.000118× |
|---|---|---|---|---|---|---|---|---|
| conv2d | 0.0993 | 0.0572 | 83.1 | ERR | 0.52 | 0.000688× | 0.00119× | 0.00626× |
| conv2d_same | 0.114 | 0.077 | 86 | ERR | 0.401 | 0.000895× | 0.00132× | 0.00466× |
| conv2d_stride2 | 0.108 | 0.0503 | 84.2 | ERR | 0.0424 | 0.000598× | 0.00128× | 0.000504× |
| elementwise_fusion | 0.0408 | 0.0616 | 138 | 0.366 | 0.0106 | 0.000447× | 0.000296× | 7.73e-05× |
| softmax | 0.0523 | 0.137 | 36.9 | 0.186 | 0.00763 | 0.00373× | 0.00142× | 0.000207× |
| layernorm | 0.0683 | 0.0469 | 37.5 | 0.207 | 0.00952 | 0.00125× | 0.00182× | 0.000254× |
| mlp | 0.0644 | 0.249 | 261 | 0.548 | 0.0293 | 0.000957× | 0.000247× | 0.000112× |
| cumsum | 0.0205 | ERR | ERR | ERR | 0.00503 | - | - | - |
| attention | 0.0816 | 0.0568 | 138 | 0.386 | 0.0327 | 0.000412× | 0.000591× | 0.000237× |
| grad_mlp | 0.194 | 0.0837 | 289 | 0.709 | 0.119 | 0.00029× | 0.000673× | 0.000413× |
| grad_mix | 2.38 | 0.082 | 189 | 0.503 | 1.29 | 0.000434× | 0.0126× | 0.00684× |
| grad_stopgrad | 0.114 | 0.0378 | **CRASH** | 0.4 | 0.0636 | - | - | - |
| grad_structural | 0.355 | ERR | ERR | ERR | 0.178 | - | - | - |
| vmap_softmax | 0.044 | ERR | ERR | ERR | 0.00745 | - | - | - |
| vmap_layernorm | 0.0765 | ERR | ERR | ERR | 0.00939 | - | - | - |
| vmap_linear | 0.0563 | 0.0583 | 139 | ERR | 0.0107 | 0.00042× | 0.000405× | 7.67e-05× |
| vmap_mlp | 0.057 | 0.103 | 245 | ERR | 0.0306 | 0.000421× | 0.000233× | 0.000125× |
| vmap_attention | 0.1 | ERR | ERR | ERR | 0.035 | - | - | - |
| vmap_grad_mlp | 0.18 | 0.132 | 291 | ERR | 0.191 | 0.000452× | 0.000619× | 0.000657× |
| matmul_1024 | 1.29 | 3.84e+03 | 91.9 | 0.795 | 1.33 | 41.7× | 0.014× | 0.0145× |
| conv2d_large | 42.8 | 1.16e+03 | 84.9 | ERR | 2.48 | 13.7× | 0.504× | 0.0292× |
| layernorm_large | 7.26 | 6.79 | 52.2 | 1.09 | 0.582 | 0.13× | 0.139× | 0.0112× |
| vmap_mlp_large | 0.209 | 1.59 | 243 | ERR | 0.0422 | 0.00654× | 0.000861× | 0.000173× |
| op_erf | 0.144 | ERR | ERR | ERR | 0.00494 | - | - | - |
| op_gelu | 0.142 | ERR | ERR | ERR | 0.0149 | - | - | - |
| op_tril_triu | 0.032 | ERR | ERR | ERR | 0.0165 | - | - | - |
| op_argmax_argmin | 0.0236 | 0.0386 | 40.6 | 0.217 | 0.00739 | 0.000951× | 0.000581× | 0.000182× |
| op_bitwise | 0.0371 | 0.0448 | 89 | 0.324 | 0.0096 | 0.000504× | 0.000417× | 0.000108× |
| op_logical | 0.0281 | 0.0437 | 84.1 | 0.298 | 0.00847 | 0.000519× | 0.000335× | 0.000101× |
| op_compare | 0.0389 | 0.0452 | 83.8 | 0.327 | 0.0096 | 0.000539× | 0.000464× | 0.000115× |
| op_power_remainder | 0.0347 | 0.0439 | 84.9 | 0.319 | 0.00824 | 0.000517× | 0.000409× | 9.71e-05× |
| op_gather_scatter | 0.0426 | 0.0502 | **CRASH** | 0.441 | - | - | - | - |
| op_pad_slice | 0.0496 | 0.0339 | 35.4 | 0.235 | 0.01 | 0.000957× | 0.0014× | 0.000283× |
| op_stop_gradient | 0.0105 | 0.0182 | **CRASH** | 0.205 | - | - | - | - |
| op_select | 0.0323 | 0.0251 | 40.2 | 0.185 | 0.00799 | 0.000623× | 0.000803× | 0.000199× |
| op_broadcast | 0.0164 | 0.0254 | 37.8 | 0.187 | 0.00313 | 0.00067× | 0.000434× | 8.27e-05× |
| batched_3d_matmul | 0.0194 | 0.0375 | 86.6 | 0.26 | 0.0137 | 0.000433× | 0.000224× | 0.000158× |
| matmul_3d_2d_shared | 0.0214 | 0.055 | 81.2 | 0.304 | 0.007 | 0.000678× | 0.000264× | 8.62e-05× |
| matvec | 0.0293 | 0.0474 | 79.2 | 0.274 | 0.00498 | 0.000598× | 0.00037× | 6.28e-05× |
| matmul_transposed | 0.0232 | 0.0511 | 81.5 | 0.269 | 0.0068 | 0.000626× | 0.000285× | 8.34e-05× |
| matmul_square | 0.0308 | 0.224 | 81.3 | 0.267 | 0.0223 | 0.00275× | 0.000379× | 0.000275× |
| diagonal_matmul | 0.0318 | 0.0537 | 135 | 0.444 | 0.00816 | 0.000398× | 0.000236× | 6.06e-05× |
| cond_basic | 0.0302 | 0.0648 | 87.2 | 0.316 | - | 0.000743× | 0.000346× | - |
| cond_multi_output | 0.0483 | 0.0765 | 137 | 0.46 | - | 0.000557× | 0.000352× | - |
| cond_nested | 0.0372 | 0.0852 | 138 | 0.411 | - | 0.000619× | 0.00027× | - |
| while_cumsum | 0.164 | 0.202 | 39.8 | 0.277 | - | 0.00509× | 0.00413× | - |
| while_fib | 0.158 | 0.189 | **CRASH** | 0.143 | - | - | - | - |
| while_cond_combo | 0.231 | 0.193 | 37.9 | 0.277 | - | 0.0051× | 0.00609× | - |
| while_poly | 0.176 | 0.102 | ERR | 0.357 | - | - | - | - |
| cond_while_pipeline | 0.184 | 0.203 | 83.4 | 0.415 | - | 0.00243× | 0.00221× | - |
| grad_erf | 0.096 | ERR | ERR | ERR | - | - | - | - |
| grad_gelu | 0.136 | ERR | ERR | ERR | - | - | - | - |
| grad_cumsum | 0.0722 | ERR | ERR | ERR | - | - | - | - |
| grad_tril | 0.0639 | ERR | ERR | ERR | - | - | - | - |
| grad_triu | 0.0583 | ERR | ERR | ERR | - | - | - | - |
| grad_power_frac | 0.114 | 0.0327 | 37.2 | 0.176 | - | 0.000879× | 0.00305× | - |
| grad_bitwise_zero | 0.146 | 0.029 | 43.4 | 0.187 | - | 0.000669× | 0.00336× | - |
| vmap_elementwise | 0.0187 | 0.0424 | 42.9 | ERR | 0.00359 | 0.000987× | 0.000435× | 8.37e-05× |
| vmap_nested | 0.0215 | 0.038 | 80.4 | ERR | 0.0083 | 0.000473× | 0.000267× | 0.000103× |
| vmap_shared_weights_dot | 0.0322 | 0.0392 | 82.9 | ERR | 0.00615 | 0.000472× | 0.000389× | 7.42e-05× |
| vmap_pad | 0.0356 | ERR | ERR | ERR | 0.00462 | - | - | - |
| vmap_concat | 0.0166 | 0.0369 | 84 | ERR | 0.00452 | 0.00044× | 0.000198× | 5.38e-05× |
| custom_l2norm | 0.0229 | ERR | ERR | ERR | 0.00733 | - | - | - |
| custom_l2norm_grad | 0.126 | ERR | ERR | ERR | 0.0865 | - | - | - |
| custom_l2norm_vmap | 0.0431 | ERR | ERR | ERR | 0.00993 | - | - | - |
| custom_relu2 | 0.0278 | ERR | ERR | ERR | 0.00612 | - | - | - |
| custom_relu2_grad | 0.256 | ERR | ERR | ERR | 0.123 | - | - | - |
| op_external | 0.0395 | 0.1 | 89 | ERR | 0.01 | 0.00112× | 0.000444× | 0.000113× |
| xla_elementwise_chain | 0.0611 | 0.0489 | 84.4 | 0.265 | - | 0.00058× | 0.000724× | - |
| xla_softmax | 0.0457 | 0.0911 | 37.5 | 0.188 | - | 0.00243× | 0.00122× | - |
| xla_mlp | 0.0548 | 0.0883 | 238 | 0.493 | - | 0.000371× | 0.00023× | - |
| xla_batched_dot | 0.0326 | 0.0631 | 134 | 0.363 | - | 0.000471× | 0.000244× | - |
| xla_reductions | 0.0607 | 0.0616 | 36.3 | 0.205 | - | 0.0017× | 0.00167× | - |
| transformer | 26.9 | 1.04e+04 | 259 | 2.25 | 6.26 | 40.2× | 0.104× | 0.0242× |
| nbody | 43.4 | 43.2 | 139 | 0.391 | 1.02 | 0.312× | 0.313× | 0.00735× |
| mha_block | 0.19 | 0.24 | 140 | 0.358 | 0.0806 | 0.00171× | 0.00136× | 0.000575× |
| ffn_block | 0.25 | ERR | ERR | ERR | 0.0469 | - | - | - |
| mha_ffn_block | 0.464 | ERR | ERR | ERR | 0.154 | - | - | - |
| lstm_cell | 0.116 | 0.104 | 288 | 0.631 | 0.0339 | 0.000361× | 0.000401× | 0.000118× |
| gru_cell | 0.114 | 0.105 | 387 | 0.743 | 0.0355 | 0.000273× | 0.000296× | 9.19e-05× |
| resnet_conv_block | 0.384 | 0.563 | 143 | ERR | 0.154 | 0.00394× | 0.00268× | 0.00108× |
| conv_block_stride2 | 0.332 | 0.201 | 199 | ERR | - | 0.00101× | 0.00167× | - |
| deep_mlp_block | 0.109 | 0.0937 | 247 | 0.51 | 0.025 | 0.000379× | 0.00044× | 0.000101× |
| mlp_block_residual | 0.078 | 0.0811 | 148 | 0.364 | 0.018 | 0.000547× | 0.000526× | 0.000121× |
| pso_step | 0.0723 | 0.0729 | 200 | 0.441 | - | 0.000364× | 0.000362× | - |
| kmeans_iter | 0.427 | 0.0923 | 90.2 | 0.306 | - | 0.00102× | 0.00473× | - |
| als_step | 0.0454 | ERR | ERR | ERR | - | - | - | - |
| cg_step | 0.112 | 0.0628 | 197 | 0.458 | - | 0.000319× | 0.000569× | - |
| power_iter | 0.0533 | 0.0402 | 90.1 | 0.255 | - | 0.000446× | 0.000591× | - |
| softmax_reg_step | 0.142 | 0.0707 | 197 | 0.563 | - | 0.000358× | 0.000718× | - |
| e2e_train_mlp | 2.5 | 2.2 | 2.3e+03 | 15.3 | - | 0.000959× | 0.00109× | - |
| e2e_train_convnet | 1.96 | 1.82 | ERR | ERR | - | - | - | - |
| e2e_train_transformer | 41.5 | ERR | ERR | ERR | - | - | - | - |
| e2e_infer_transformer | 1.1 | 0.406 | ERR | 0.822 | - | - | - | - |
| e2e_pso_optimize | 1.39 | 0.444 | ERR | 0.914 | - | - | - | - |
| e2e_kmeans | 1.01 | 0.162 | ERR | 0.373 | - | - | - | - |
| e2e_power_iteration | 0.681 | 0.243 | ERR | 0.428 | - | - | - | - |

### ROCm vs llvm-cpu, aggregate

- **65 comparable cells** (examples that ran on both). ROCm is faster in **3**, slower in **62**.
- The 3 ROCm wins are exactly the examples whose llvm-cpu time is dominated by *compute*:
  `matmul_1024` **41.7×**, `transformer` **40.2×**, `conv2d_large` **13.7×** — i.e. ROCm wins by
  making the 3.8–10.4-second generic-CPU codegen problem disappear.
- Everything else loses to llvm-cpu because ROCm's per-call staging floor (~35–90 ms) dwarfs the
  graph's own runtime: `grad_mlp` 289 ms vs 0.084 ms (0.00029×), `gru_cell` 387 vs 0.105, `mlp`
  261 vs 0.249, `vmap_mlp` 245 vs 0.103, `nbody` 139 vs 43 (0.31× — the one large example where
  ROCm's compute win is not enough to claw back staging).
- **xla/rocm vs iree/rocm is not close**: xla is faster in **50/50** comparable cells, by 12×–520×
  (`gru_cell` 0.74 ms vs 387 ms, `mlp` 0.55 vs 261, `pso_step` 0.44 vs 200). Both run on the same
  GPU, so the entire gap is the ~35 ms-per-upload iree staging path (Root cause 1) plus iree's
  dispatch overhead. XLA's PjRt upload path is ~128× cheaper for the same buffer.

## Headline numbers

| example | numpy/cpu | iree/llvm-cpu | iree/rocm:0 | xla/rocm:0 | torch-cpu | rocm vs llvm-cpu | rocm vs numpy | rocm vs torch-cpu | rocm max_abs | xla max_abs |
|---|---|---|---|---|---|---|---|---|---|---|
| `matmul_1024` | 1.29 | 3.84e+03 | 91.9 | 0.795 | 1.33 | 41.7× | 0.014× | 0.0145× | 0.000214 | 0.123 |
|---|---|---|---|---|---|---|---|---|---|---|
| `matmul_square` | 0.0308 | 0.224 | 81.3 | 0.267 | 0.0223 | 0.00275× | 0.000379× | 0.000275× | 0 | 0.379 |
| `transformer` | 26.9 | 1.04e+04 | 259 | 2.25 | 6.26 | 40.2× | 0.104× | 0.0242× | 0.000841 | 0.465 |
| `attention` | 0.0816 | 0.0568 | 138 | 0.386 | 0.0327 | 0.000412× | 0.000591× | 0.000237× | 2.38e-07 | 0.00125 |
| `conv2d_large` | 42.8 | 1.16e+03 | 84.9 | - | 2.48 | 13.7× | 0.504× | 0.0292× | 0.000137 | - |
| `resnet_conv_block` | 0.384 | 0.563 | 143 | - | 0.154 | 0.00394× | 0.00268× | 0.00108× | 1.19e-06 | - |
| `nbody` | 43.4 | 43.2 | 139 | 0.391 | 1.02 | 0.312× | 0.313× | 0.00735× | 7.63e-06 | 5.72e-06 |
| `layernorm_large` | 7.26 | 6.79 | 52.2 | 1.09 | 0.582 | 0.13× | 0.139× | 0.0112× | 9.54e-07 | 9.54e-07 |
| `e2e_train_mlp` | 2.5 | 2.2 | 2.3e+03 | 15.3 | - | 0.000959× | 0.00109× | - | 4.43e-07 | 0.00631 |
| `e2e_train_transformer` | 41.5 | - | - | - | - | - | - | - | - | - |
| `e2e_infer_transformer` | 1.1 | 0.406 | - | 0.822 | - | - | - | - | - | 0 |
| `e2e_pso_optimize` | 1.39 | 0.444 | - | 0.914 | - | - | - | - | - | 4.77e-07 |
| `e2e_kmeans` | 1.01 | 0.162 | - | 0.373 | - | - | - | - | - | 0 |
| `e2e_power_iteration` | 0.681 | 0.243 | - | 0.428 | - | - | - | - | - | 5.96e-08 |
| `mha_block` | 0.19 | 0.24 | 140 | 0.358 | 0.0806 | 0.00171× | 0.00136× | 0.000575× | 7.63e-06 | 0.0745 |
| `matmul` | 0.0237 | 0.136 | 87.4 | 0.272 | 0.0103 | 0.00155× | 0.000272× | 0.000118× | 0 | 0.0258 |

One-line commentary (ROCm vs llvm-cpu, ROCm vs torch-cpu):

- **`matmul_1024`** — **the ROCm showcase: 91.9 ms vs 3836 ms on llvm-cpu (41.7×)**. It is still
  ~0.3 ms of compute wrapped in one ~90 ms upload of the two 4 MB operands (2 staged leaves → the
  ~80 ms staging class). vs torch-cpu (1.33 ms): 0.0145× — torch on CPU does the whole 2 GFLOP in
  1.3 ms; ROCm pays the staging toll twice. On a *device-resident* loop this example would be
  sub-millisecond (see Root cause 1).
- **`matmul_square`** — 81.3 ms vs 0.224 ms llvm-cpu (0.0028×): a tiny [64,64] matmul whose llvm-cpu
  time is ~0.2 ms is pure staging on ROCm. The canonical illustration of the harness's per-run
  transfer cost.
- **`transformer`** — **258 ms vs 10.4 s on llvm-cpu (40.2×)** and 2.25 ms under xla/rocm. A real
  workload where the CPU codegen is so slow that even with staging the GPU wins by 40×. vs torch-cpu
  6.26 ms → 0.024×.
- **`attention`** — 138 ms vs 0.057 ms llvm-cpu (0.0004×). Correct (2.4e-7) but the [4,8,16] graph is
  microseconds; 100 % staging.
- **`conv2d_large`** — **84.9 ms vs 1163 ms llvm-cpu (13.7×)**; vs torch-cpu 2.48 ms → 0.029×.
  Interesting: ROCm *also* beats the numpy interpreter by 2× (42.8 ms) — one of the few examples
  where the GPU's compute advantage survives the staging overhead. XLA/ROCm **cannot run conv**
  (MIOpen config failure — see exclusions).
- **`resnet_conv_block`** — 143 ms vs 0.563 ms llvm-cpu (0.0039×). Compound: 66 ms of staging plus
  a conv that llvm-cpu does in well under a millisecond. Deferred on xla (MIOpen).
- **`nbody`** — 139 ms vs 43.2 ms llvm-cpu (0.31×) and vs torch-cpu 1.02 ms (0.0074×). N=1024,
  fully unrolled: the largest pure-elementwise graph in the suite; ROCm's compute win is real but
  ~43 ms of it is staging. xla runs it in 0.39 ms.
- **`layernorm_large`** — 52.2 ms vs 6.79 ms (0.13×), torch-cpu 0.58 ms. [4096,1024] with two staged
  leaves; xla 1.09 ms.
- **`e2e_train_mlp`** — 2297 ms per 10-iteration training run vs 2.2 ms llvm-cpu (0.00096×) and
  15.3 ms under xla/rocm. **The runner path is where per-run staging compounds 10×**: each of the
  10 iterations re-stages its inputs, so ROCm pays ~10 × (staging + dispatch) for a graph whose
  total compute is ~2 ms. This is the clearest argument for a device-resident loop on ROCm.
- **`e2e_train_transformer`** — **cannot run on any compiler backend here**: it defers at export
  (`gelu` is a StableHLO v1 deferral) on iree/llvm-cpu, iree/rocm and xla alike; numpy runs it in
  41.5 ms. Not a ROCm issue — a backend-coverage issue.
- **`e2e_infer_transformer`** — runs on llvm-cpu (0.406 ms) and xla/rocm (0.822 ms, bit-exact), but
  **is one of the six iree/rocm-only failures** (`iree-compile` error for targets: rocm). Notable
  inversion: the *while_loop-based* inference graph is exactly the shape iree-rocm struggles with.
- **`e2e_pso_optimize` / `e2e_kmeans` / `e2e_power_iteration`** — ROCm **excluded/errored**
  (two `hipErrorIllegalAddress`, one `hal.fence.await` abort). All three run on llvm-cpu
  (0.44 / 0.16 / 0.24 ms) and on xla/rocm (0.91 / 0.37 / 0.43 ms). On this GPU the in-graph
  iterative-optimization family must go through xla or the numpy/llvm-cpu path.
- **`mha_block` / `matmul`** — small-graph staging victims: 140 ms vs 0.24 ms (0.0017×) and 87 ms vs
  0.136 ms. Both bit-exact on ROCm; both 2 orders slower than xla (0.358 / 0.272 ms).

## Build (lower + compile + load) wall-times

Compile cost matters on ROCm: `iree/rocm` has the highest median build in the matrix (302 ms) and a
harder tail (1561 ms) than llvm-cpu (225 / 987 ms). xla/rocm is the *cheapest* overall (51 ms median,
6.0 s total) because the PJRT plugin JIT-compiles quickly and reuses its HSACO path, but its worst
case (`grad_mix`, 1686 ms) is the largest single build of all three compiler backends.

| backend | builds | min | median | p90 | max | total |
|---|---|---|---|---|---|---|
| numpy / cpu | 98 | 1 | 4 | 16 | 271 | 1.1 s |
|---|---|---|---|---|---|---|
| iree / llvm-cpu | 75 | 96 | 225 | 377 | 987 | 20.1 s |
| iree / rocm:0 | 66 | 235 | 302 | 486 | 1561 | 23.9 s |
| xla / rocm:0 | 59 | 23 | 51 | 144 | 1686 | 6.0 s |

Slowest 8 lower+compile+load builds per compiler backend (ms):

| backend | slowest examples |
|---|---|
| iree / llvm-cpu | `grad_mix` 987, `e2e_train_convnet` 685, `e2e_train_mlp` 533, `transformer` 494, `cond_while_pipeline` 434, `while_cond_combo` 433, `mha_block` 408, `while_cumsum` 397 |
|---|---|
| iree / rocm:0 | `grad_mix` 1561, `transformer` 1214, `e2e_train_mlp` 738, `conv2d_large` 552, `cond_while_pipeline` 499, `softmax_reg_step` 487, `while_cond_combo` 487, `conv_block_stride2` 486 |
| xla / rocm:0 | `grad_mix` 1686, `kmeans_iter` 301, `transformer` 283, `e2e_train_mlp` 197, `e2e_infer_transformer` 184, `grad_mlp` 165, `e2e_pso_optimize` 144, `mha_block` 140 |

Stage wall-times for a full run of the 98-example registry (conformance, then benchmark):

| run | conformance wall | benchmark wall | notes |
|---|---|---|---|
| numpy / cpu | 14.1 s | 111.4 s | includes 62 CPU-torch reference timings |
|---|---|---|---|
| iree / llvm-cpu | 52.8 s | 484.2 s | benchmark wall dominated by the multi-second llvm-cpu examples |
| iree / rocm:0 | 64.1 s | 356.4 s | benchmark wall ~ half llvm-cpu's (compute is faster, staging dominates) |
| xla / rocm:0 | 195.2 s | 11.6 s | long conformance wall = per-example cold PJRT compiles in chunked subprocesses; benchmark reuses warm executables within a chunk |

The xla conformance wall is inflated by the chunked-subprocess methodology (each chunk re-loads the
482 MB plugin and re-initialises the ROCm service); it is not a measure of per-example compile cost —
use the build table above for that.

## Deferrals, errors and exclusions — exactly what was dropped, per backend

Everything below is an **explicit, recorded error** (an `ExampleResult.error`) — never a silent
skip. Error text is reproduced as recorded (truncated where the harness truncates).

**numpy / cpu** — 0 distinct error(s), 0 example(s) deferred/failed.

- *(no deferrals/errors)*

**iree / llvm-cpu** — 11 distinct error(s), 23 example(s) deferred/failed.

- `stablehlo export: op 'gelu' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `capability drift: the iree backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'cumsum' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `cumsum`, `grad_cumsum`
- `stablehlo export: op 'solve' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `grad_structural`, `als_step`
- `stablehlo export: op 'reshape' at driver.py:77 keepdims reshape operand has dynamic dims (Dim('batch'),) (offending dims (Dim('batch'),)) — dynamic sh` — 2: `vmap_softmax`, `vmap_layernorm`
- `stablehlo export: op 'erf' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_tril_triu`, `grad_tril`
- `stablehlo export: op 'reshape' at driver.py:77 keepdims reshape operand has dynamic dims (Dim('batch'), 8) (offending dims (Dim('batch'),)) — dynamic ` — 1: `vmap_attention`
- `stablehlo export: op 'triu' at driver.py:77 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 1: `grad_triu`
- `stablehlo export: op 'pad' at driver.py:77 operand shape has dynamic dims (Dim('batch'), 16) (offending dims (Dim('batch'),)) — dynamic shapes are not` — 1: `vmap_pad`
- `stablehlo export: op 'reshape' at driver.py:77 keepdims reshape operand has dynamic dims (Dim('batch'), 4) (offending dims (Dim('batch'),)) — dynamic ` — 1: `custom_l2norm_vmap`

**iree / rocm:0** — 16 distinct error(s), 29 example(s) deferred/failed, 4 crashed (excluded).

- `stablehlo export: op 'gelu' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `capability drift: the iree backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'reshape' at driver.py:79 keepdims reshape operand has dynamic dims (Dim('batch'),) (offending dims (Dim('batch'),)) — dynamic sh` — 2: `vmap_softmax`, `vmap_layernorm`
- `stablehlo export: op 'erf' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_tril_triu`, `grad_tril`
- `stablehlo export: op 'cumsum' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `grad_cumsum`, `cumsum`
- `stablehlo export: op 'solve' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `als_step`, `grad_structural`
- `Error querying devices: c/runtime/src/iree/hal/drivers/hip/hip_driver.c:208: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700): an illegal mem` — 2: `e2e_kmeans`, `e2e_power_iteration`
- `stablehlo export: op 'reshape' at driver.py:79 keepdims reshape operand has dynamic dims (Dim('batch'), 8) (offending dims (Dim('batch'),)) — dynamic ` — 1: `vmap_attention`
- `Error invoking function: c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:786: ABORTED; the semaphore was aborted; while invoking native function ` — 1: `while_poly`
- `stablehlo export: op 'triu' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 1: `grad_triu`
- `stablehlo export: op 'pad' at driver.py:79 operand shape has dynamic dims (Dim('batch'), 16) (offending dims (Dim('batch'),)) — dynamic shapes are not` — 1: `vmap_pad`
- `stablehlo export: op 'reshape' at driver.py:79 keepdims reshape operand has dynamic dims (Dim('batch'), 4) (offending dims (Dim('batch'),)) — dynamic ` — 1: `custom_l2norm_vmap`
- `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm):
Error invoking IREE compiler tool iree-compile
Error code: ` — 1: `softmax_reg_step`
- `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm):
Error invoking IREE compiler tool iree-compile
Error code: ` — 1: `e2e_infer_transformer`
- `Error invoking function: c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:770: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700): an illeg` — 1: `e2e_pso_optimize`

| crashed example | rc | phase | evidence |
|---|---|---|---|
| `op_gather_scatter` | -11 | see exclusions section | `/tmp/genesis/task_9049baaa4ad84718/out/C_logs/chunk_0006.json.log` |
|---|---|---|---|
| `op_stop_gradient` | -11 | see exclusions section | `/tmp/genesis/task_9049baaa4ad84718/out/C_logs/chunk_0009.json.log` |
| `while_fib` | -11 | see exclusions section | `/tmp/genesis/task_9049baaa4ad84718/out/C_logs/chunk_0015.json.log` |
| `grad_stopgrad` | -11 | see exclusions section | `/tmp/genesis/task_9049baaa4ad84718/out/C_logs_resume/chunk_0005.json.log` |

**xla / rocm:0** — 12 distinct error(s), 39 example(s) deferred/failed.

- `capability drift: the xla backend cannot execute dynamic shapes (function signature has symbolic/runtime-dynamic dimensions)` — 13: `vmap_softmax`, `vmap_layernorm`, `vmap_linear`, `vmap_mlp`, `vmap_attention`, `vmap_grad_mlp`, `vmap_mlp_large`, `vmap_elementwise`, `vmap_nested`, `vmap_shared_weights_dot`, `vmap_pad`, `vmap_concat`, `custom_l2norm_vmap`
- `stablehlo export: op 'gelu' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `the xla PJRT plugin failed at PJRT_Client_Compile: All backends failed to find supported configs:
 - INTERNAL: GetMIOpenConvolveAlgorithms failure` — 4: `conv2d`, `conv2d_same`, `conv2d_stride2`, `conv2d_large`
- `capability drift: the xla backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'cumsum' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `cumsum`, `grad_cumsum`
- `stablehlo export: op 'solve' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `grad_structural`, `als_step`
- `stablehlo export: op 'erf' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 2: `op_tril_triu`, `grad_tril`
- `the xla PJRT plugin failed at PJRT_Client_Compile: Failed to get configs for: 2 out of 5 instructions. See logs for all failures. Example failure: 
Al` — 2: `resnet_conv_block`, `e2e_train_convnet`
- `stablehlo export: op 'triu' at driver.py:79 is not supported in v1 — decompose it into supported ops or use a future compiler adapter` — 1: `grad_triu`
- `capability drift: the xla backend cannot execute external_call op 'bench_row_softmax' — adapter host-dispatch for external kernels is not yet wired fo` — 1: `op_external`
- `the xla PJRT plugin failed at PJRT_Client_Compile: Failed to get configs for: 3 out of 6 instructions. See logs for all failures. Example failure: 
Al` — 1: `conv_block_stride2`

### iree / rocm:0 crash exclusions (the 4 process deaths)

These four examples killed the worker process with SIGSEGV (`rc=-11`) during the iree/rocm matrix
run. The run was chunked per-example with bisection, so each death cost exactly one example; the
crash was then confirmed by re-running the example **alone in conformance-only mode**.

| example | rc | where it died | evidence | isolation re-run |
|---|---|---|---|---|
| `while_fib` | -11 | `run` → `iree/runtime/function.py:147 _invoke` → `iree.py:1597` → `etl/pipeline.py:1290` → `_util.py:310` (C stack through `/opt/rocm-7.1.0/lib/libamdhip64.so`) | `out/C_logs/chunk_0015.json.log` (rc=-11, 0-byte stdout); fault-handler stack `out/C_repro/fh_while_fib.log` | **still segfaults** in isolation (conformance-only) — deterministic |
|---|---|---|---|---|
| `op_gather_scatter` | -11 | `_util.py:344 best_time_ms` — i.e. **during the benchmark phase**, after conformance had already passed; C stack in `iree/_runtime_libs/_runtime.abi3.so` at `_Py_Dealloc` | `out/C_logs/chunk_0006.json.log`; `out/C_repro/fh_op_gather_scatter.log` | **PASSES** conf-only: max_abs **0.0**, etl_ms 185.5 (`out/C_repro/repro_op_gather_scatter.json`) |
| `op_stop_gradient` | -11 | same benchmark-phase pattern | `out/C_logs/chunk_0009.json.log` | **PASSES** conf-only: max_abs **0.0**, etl_ms 75.7 (`repro_op_stop_gradient.json`) |
| `grad_stopgrad` | -11 | same benchmark-phase pattern | `out/C_logs_resume/chunk_0005.json.log` | **PASSES** conf-only: max_abs **1.82e-12**, etl_ms 130.6 (`repro_grad_stopgrad.json`) |

Reading: `while_fib` is a genuine deterministic segfault of the iree ROCm HAL on a while-loop
graph (it also takes the process down on iree-cuda — see `./CONTEXT.md` Known Issues; it passes on
llvm-cpu). The other three die in the **benchmark** phase inside iree's runtime teardown/peek path
(`best_time_ms` → `_Py_Dealloc`), i.e. repeated invoke+free cycles of the same executable, not the
graph itself — they are numerically correct and complete conformance. Two of them
(`op_gather_scatter`, `op_stop_gradient`) are the only non-control-flow members of the set, which
points at the iree ROCm HAL's buffer deallocation rather than at graph shape.

**Excluded per backend, in one line each**: numpy/cpu — none. iree/llvm-cpu — none. iree/rocm:0 —
`while_fib` (deterministic segfault) plus `op_gather_scatter` / `op_stop_gradient` / `grad_stopgrad`
(segfault during the benchmark phase; each verified numerically correct in isolation). xla/rocm:0 —
none (the plugin aborts only at *interpreter teardown*, after all work is on disk; see below).

### iree/rocm-only failures that pass on llvm-cpu

Six examples run correctly on llvm-cpu but fail on the ROCm target — the delta between the 23
llvm-cpu deferrals and the 29 rocm ones:

| example | failure | evidence line |
|---|---|---|
| `softmax_reg_step` | `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm)` | iree-compile error |
|---|---|---|
| `e2e_infer_transformer` | same iree-compile failure (the scatter/while decoding graph) | iree-compile error |
| `while_poly` | `c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:786: ABORTED; the semaphore was aborted; ... hal.fence.await` | runtime invoke |
| `e2e_pso_optimize` | `event_semaphore.c:770: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700)` | runtime invoke |
| `e2e_kmeans` | `hip_driver.c:208: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700): an illegal memory access` ("Error querying devices") | runtime |
| `e2e_power_iteration` | same `hipErrorIllegalAddress` | runtime |

This is the family the pre-run ROCm validation flagged: **iree-rocm `while_loop` graphs whose carry
comes from a graph constant and/or whose condition reads a float carry**. The probe here is
partial: `while_cond_combo` and `while_cumsum` (constant-carry, float-condition) **both ran fine on
rocm** (37.9 ms / 39.8 ms, both bit-exact), and `cond_while_pipeline` too (83.4 ms, bit-exact), so
the constant-carry/float-condition shape is *not* sufficient on its own. What fails is the
optimisation-loop subset (`while_poly`, `e2e_pso_optimize`, `e2e_kmeans`, `e2e_power_iteration`) plus
the scatter-bearing `e2e_infer_transformer` — consistent with bufferization/HAL binding of carried
mutable buffers rather than with the condition type.

### xla / rocm:0 — plugin behaviour and failure classes

The xla backend was run with `ETL_PJRT_PLUGIN` pointing at
`.venv/lib/python3.14/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so` (482 MB). At startup
the plugin logs:

```
XLA service ... initialized for platform ROCM (this does not guarantee that XLA will be used)
  StreamExecutor [0]: AMD Instinct MI300X VF, AMDGPU ISA version: gfx942:sramecc+:xnack-
    (Driver: 7.1.25424; Runtime: 7.1.25424; Toolkit: 7.14.60850; DNN: 1.3.0)
Using BFC allocator.  XLA backend allocating 143.77GiB (154367164416 bytes) on device 0
Found GPU target config and a PjRtClient with matching configuration. Performing a JIT compilation.
amdgpu_backend.cc:163] TF_XLA_HSACO_CACHE_DIR is not set: HSACO file cache is disabled!
```

and **aborts the process at interpreter teardown** (SIGABRT, `rc=134`) *after* all work has
succeeded. Every xla run must therefore be read from its flushed payload and never from its exit
code (`ETL_BENCH_HARD_EXIT=1` in the driver calls `os._exit(0)` after the JSON is on disk).

Failure classes (39 examples, from the error section above):

- **Dynamic shapes (13)** — every `vmap_*` example including `vmap_mlp_large` and
  `custom_l2norm_vmap`: `capability drift: the xla backend cannot execute dynamic shapes (function
  signature has symbolic/runtime-dynamic dimensions)`. The xla adapter is static-shape-only, so the
  whole vectorize family is out — this is the largest single class and is inherent to the adapter,
  not to the plugin.
- **MIOpen convolution (7)** — `conv2d`, `conv2d_same`, `conv2d_stride2`, `conv2d_large`
  (`GetMIOpenConvolveAlgorithms failure`), `resnet_conv_block` + `e2e_train_convnet`
  (`Failed to get configs for: 2 out of 5 instructions`), `conv_block_stride2` (`3 out of 6`).
  The plugin's ROCm build cannot find a working MIOpen conv algorithm for these shapes → **no conv
  runs on xla/rocm at all** here.
- **Shared export deferrals (19)** — `gelu` ×5, `block_call` ×4, `cumsum` ×2, `solve` ×2, `erf` ×2,
  `tril` ×2, `triu` ×1: identical to the other compiler backends (StableHLO v1 export gaps).
- **`op_external` (1)** — `adapter host-dispatch for external kernels is not yet wired` for xla
  (works on iree host-dispatch and numpy).

## Root cause 1 — iree/rocm per-call latency is ~100 % host→device staging

`probe2.py` (best-of-30, warmup 3) decomposes one `matmul` call:

| measurement | llvm-cpu | rocm:0 |
|---|---|---|
| full harness pattern (place + run + readback) | **0.100 ms** | **82.961 ms** |
|---|---|---|
| placement only (2 × `etl.tensor(x).to(rocm:0)`) | — | **80.106 ms** |
| readback only (D2H, `.to(cpu)`) | — | **0.066 ms** |
| run only, device-resident inputs **and** outputs | — | **0.058 ms** |
| tiny add graph, device-resident run only | — | **0.055 ms** |
| tiny add graph, full place + run + readback | — | **34.704 ms** |

So the "83 ms matmul" is ~80 ms of upload + ~0.06 ms of actual execution. `probe3.py` gives the
upload cost model (`etl.tensor(x).to(rocm:0)`):

| buffer | upload | marginal |
|---|---|---|
| 0.004 MB | 35.96 ms | (600 ms/MB — fixed-cost regime) |
|---|---|---|
| 0.25 MB | 36.19 ms | 0.9 ms/MB |
| 4 MB | 40.07 ms | 1.0 ms/MB |
| 16 MB | 48.45 ms | 0.7 ms/MB |
| 64 MB | 82.66 ms | 0.7 ms/MB |

→ **~33–35 ms fixed per upload + ~0.7–1.0 ms/MB**, and **D2H readback is free**
(0.001–0.002 ms even at 64 MB — iree's device→host path is a lazy host map, not a copy). This also
retires the earlier "23 %/page-fault stall" hypothesis: the measured difference between iree's
`--interleave` and `--contiguous` host-buffer modes is only 0.292 vs 0.340 ms at 64 MB.

That model explains the entire ROCm timing column: the observed per-example times cluster in bands
determined by the *number of staged input leaves* (1 leaf ≈ 35–45 ms, 2 ≈ 80–90 ms, 3 ≈ 134–140 ms,
4+ ≈ 190–290 ms) plus a small dispatch term — not by graph size.

**The iree upload provider is the cost, not the GPU.** `probe6.py` measures the same 4 MB upload
under two providers on the same device:

| provider for kind `rocm` | 4 MB upload |
|---|---|
| iree thunk (flat default slot, before any xla activation) | 37.47 ms |
|---|---|
| … and again after `etl.backends.get("iree")` | 37.58 ms |
| **xla PjRt provider (after building one XLA executable)** | **0.293 ms** |

A ~128× difference. Mechanism (measured): `etl/backends/__init__.py` installs the flat default
transfer provider for kind `rocm` (→ iree's `upload_tensor`), while activating the xla adapter
records xla as the preferred provider under its own backend slot; `Tensor.to` resolves the recorded
backend slot first, so **a process that has built an XLA executable gets xla's fast provider for
subsequent iree runs too**. `probe2.py` shows this leaking into results: with xla initialised first,
the iree `matmul` error vs fp64 became 6.037e-4 — *identical* to xla's — because the operands took
the xla upload path. In the headline matrix the processes are clean (iree alone), which is why
iree/rocm `matmul_1024` max_abs is 2.14e-4 (fp32-accurate) and xla's is 0.123 (TF32).

**Practical consequence**: any ROCm workload that re-stages inputs per call pays ~35 ms/call
regardless of graph size. The documented remedy already exists in the library — the **same-device
loop** (`docs`/`etl/CONTEXT.md`: build once, feed each run's device outputs back as the next run's
inputs). Measured here: a device-resident `matmul` run costs **0.057 ms** and a 0-input tiny-add run
**0.055 ms**. The harness cannot exploit that (it must compare against host references and times
`place + run + readback`), which is why the benchmark column understates ROCm; the `e2e_train_mlp`
runner result (2297 ms for 10 iterations) is the clearest example of the compounding.

## Root cause 2 — XLA/ROCm fp32 `dot` runs in TF32 (13 numeric failures)

`probe4.py` compares each backend's `[n,n]×[n,n]` fp32 matmul against an fp64 ground truth, alongside
simulations that round both operands to TF32 (10-bit mantissa) and bf16 (7-bit):

| n | \|truth\|max | sim fp32 | sim **TF32** | sim bf16 | iree/rocm | **xla/rocm** |
|---|---|---|---|---|---|---|---|---|
| 1024 | 293.7 | 0 (rel 0) | 0.193 (rel 6.6e-4) | 1.54 (rel 5.2e-3) | 0.000574 (rel 2.0e-6) | **0.193 (rel 6.6e-4)** |
|---|---|---|---|---|---|---|
| 256 | 80.7 | 0 | 0.0528 (rel 6.5e-4) | 0.43 | 7.33e-5 (rel 9.1e-7) | **0.0528 (rel 6.5e-4)** |
| 64 | 22.2 | 0 | 0.0158 (rel 7.1e-4) | 0.131 | 7.43e-6 (rel 3.4e-7) | **0.0158 (rel 7.1e-4)** |

XLA's error matches the TF32 operand-truncation simulation to all printed digits at every size,
while iree-on-ROCm stays at ordinary fp32 accumulation error. Conclusion: **the XLA ROCm backend
lowers fp32 `dot` to a TF32 tensor-core path by default**, and there is no `precision_config`
anywhere in etl to request otherwise (`rg precision_config etl/` finds nothing — only
`torch.backends.cudnn.allow_tf32 = False` in `etl/bench/examples/op_large.py`). There is also no
XLA flag to disable it on this plugin: no `xla_gpu*` TF32 flag string exists in the plugin binary
and `XLA_FLAGS=--xla_gpu_enable_tf32=false` fails with `Unknown flag in XLA_FLAGS`.

Therefore the 13 xla numeric failures are a **backend precision mode**, not an etl bug and not a
plugin bug: they are the correct output of a TF32 computation. Two consequences:
(a) the xla/rocm timing column is partly a TF32-timing column (the ~0.8 ms `matmul_1024` is TF32
tensor-core speed, not fp32); (b) anyone running xla on ROCm must either accept TF32 accuracy or
have etl emit an explicit precision request — which today it cannot.

## Task E — TVM cannot target ROCm

`apache-tvm` 0.26.0 is installed, but its wheel is built with `USE_ROCM=OFF`:
`tvm.runtime.device("rocm", 0).exist` → **False**. TVM is therefore unusable on this machine
(no ROCm runtime support, and the LLVM-CPU target only duplicates `iree/llvm-cpu`). No TVM matrix was
run. Record-only.

## Harness note found during this work (needs an owner outside this node)

Driving the xla run through the **public CLI/API as-is** produces 98/98 identical errors:

```
BackendError: the xla backend does not recognize the lower option 'target_backends'
— known options by stage: lower: rng_bit_generator; compile: opt_level, plugin_path,
  xla_compile_options; load: plugin_path; run: (none)
```

Cause: `etl/bench/_util.py::resolve_backend_options` injects a device-derived `target_backends`
for **every** non-numpy backend, but `target_backends` is an iree-only concept; the xla adapter
declares it in no stage. The correct gate is the iree family (backend name startswith `"iree"`).
The xla results in this report were produced by a standalone wrapper that bypasses that injection
(conformance/benchmark called with the resolved options minus `target_backends`) — i.e. the numbers
are real, the CLI path is broken for xla. `tests/bench/test_backend_options.py` currently *pins*
the buggy behaviour (it parametrises over backends and asserts `target_backends == ["rocm"]` for
"every compiler backend"), so the fix and the test must change together. Reported to the parent
agent; `_util.py` is inside this node but the pinned test lives in the read-only `tests/` tree.

## How to reproduce

```bash
# one-time environment (Python 3.14)
uv sync --all-extras
uv pip install jax-rocm7-pjrt            # for the xla run only; no other deps

# A — numpy reference baseline (CPU)
./.venv/bin/python -m etl.bench --backend numpy --device cpu --torch

# B — iree on llvm-cpu
./.venv/bin/python -m etl.bench --backend iree --device cpu --torch

# C — iree on the MI300X  (device is ALWAYS explicit; target_backends is derived as ["rocm"])
./.venv/bin/python -m etl.bench --backend iree --device rocm:0 --torch

# D — xla on the MI300X (standalone wrapper; the plugin aborts at interpreter exit,
#     so read the printed report, never the exit code)
ETL_PJRT_PLUGIN=$PWD/.venv/lib/python3.14/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so \
  ./.venv/bin/python - <<'PY'
import etl, os
from etl.bench import conformance, benchmark
rep = conformance(None, use_torch=False, backend="xla", device="rocm:0",
                  opt_level="O3", plugin_path=os.environ["ETL_PJRT_PLUGIN"])
print(rep, flush=True)
print(benchmark(None, use_torch=False, backend="xla", device="rocm:0",
                opt_level="O3", plugin_path=os.environ["ETL_PJRT_PLUGIN"]), flush=True)
os._exit(0)   # the ROCm PJRT plugin SIGABRTs at teardown AFTER the work succeeded
PY

# E — TVM probe (expected: no ROCm support)
./.venv/bin/python -c "import tvm; print(tvm.runtime.device('rocm',0).exist)"   # -> False
```

**Crash isolation (required for iree/rocm).** Do not run the ROCm iree matrix as one process: four
examples SIGSEGV the interpreter, and a mid-run death loses everything after it. The matrix in this
report was produced by a per-example chunk driver (run N examples, bisect the chunk if it dies, record
`rc` + stderr, resume) — conceptually:

```bash
for ex in $(./.venv/bin/python -c "import etl.bench as b; print(' '.join(b.list_examples()))"); do
  ./.venv/bin/python -m etl.bench --backend iree --device rocm:0 --conformance --no-benchmark \
      --examples "$ex" >> run.log 2>&1 || echo "EXCLUDED $ex rc=$?" >> exclusions.txt
done
```

Exclusions recorded in this report: iree/rocm `while_fib` (deterministic), plus
`op_gather_scatter`, `op_stop_gradient`, `grad_stopgrad` (benchmark-phase deaths; each verified
correct in isolation).

## Conclusions

1. **iree-on-ROCm is numerically correct** — 65/65 executed examples pass, 25 bit-exact — but its
   per-call economics on this setup are **~35 ms + ~0.8 ms/MB of input staging**, so only
   compute-dominated graphs beat llvm-cpu (3 of 65: `matmul_1024` 41.7×, `transformer` 40.2×,
   `conv2d_large` 13.7×) and every small graph is 100–1000× slower than llvm-cpu. Device-resident
   execution (one ~0.06 ms run per call) is the only workable pattern; ROCm is a poor fit for the
   harness's per-call `place + run + readback` pattern.
2. **xla-on-ROCm is ~128× cheaper to feed and correspondingly faster** (50/50 comparable cells,
   12–520×), but runs fp32 matmul in **TF32**, which produces 13 numeric failures against fp32
   references and makes its matmul timings tensor-core timings. It also cannot run conv (MIOpen) or
   any dynamic-shape/vmap graph.
3. **Neither compiler backend is a drop-in for the other on ROCm**: iree covers control flow,
   gather/scatter, conv and external kernels at fp32 accuracy but is staging-bound and crashes on
   five in-graph-iterative-optimization examples; xla covers exactly those optimization loops at
   ~400× the speed but is static-shape-only, conv-less and TF32-inaccurate.
4. **The numpy backend is the portability floor**: all 98 examples, bit-exact, no exclusions.
5. Three defects worth escalating: the iree-rocm while-loop/optimization-loop runtime failures and
   the `while_fib`/benchmark-phase SIGSEGVs (iree ROCm HAL); the ~128× iree upload-provider gap
   versus PjRt; and XLA's unconditional TF32 fp32 `dot` (etl emits no precision request). The
   harness-side `target_backends` injection bug blocks the public xla path entirely.
