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

Raw harness output for every table below is committed under `./rocm_mi300x_data/`
(`A_numpy_cpu.json`, `B_iree_cpu.json`, `C_iree_rocm.json`, `D_xla_conf_highest.json`,
`D_xla_bench_tf32.json`, `D_xla_bench_highest.json`, plus the staging probe); the tables are
regenerated from those JSONs by `./rocm_mi300x_data/gen_fragments.py`.

## Caveats that qualify every number below

1. **The torch numbers are CPU baselines, not GPU numbers.** `torch` here is a CUDA build with no
   CUDA device present (`torch.version.hip is None`, `torch.cuda.is_available() is False`), so the
   harness's `resolve_torch_device` degrades every torch reference to CPU. In the timing tables a
   column headed `torch-cpu` is a single-process CPU torch run measured on this host — it is *not*
   the MI300X. Use it only as a "what a mature CPU framework does" yardstick; it says nothing about
   torch-on-ROCm, which was not available for this run.
2. **The timed `iree/rocm` column is the harness's per-call pattern** (`place_input` → `run` →
   readback), i.e. a host tensor is explicitly placed on the device, one `run` is timed, and the
   result is read back for comparison. The host↔device staging path is cheap on this branch
   (~0.05 ms fixed + ~0.11 ms/MB per staged leaf — see "Root cause 1"), so this column is close to
   the device-resident cost; a stateful loop that keeps tensors on the device (build once, feed the
   previous run's device outputs back as inputs) is cheaper still.
3. **The xla timings are DEFAULT TF32 numbers** unless a row says otherwise: this XLA/ROCm plugin
   lowers fp32 `dot` to a TF32 tensor-core path by default. The `## Root cause 2` section reports
   the same cells under `dot_precision="HIGHEST"` (true fp32) so the speed/precision tradeoff is
   visible. The accuracy column of the xla timing matrix is a static-shape, TF32 column.
4. **The xla plugin SIGABRTs at interpreter teardown** *after* all work has completed. Every xla
   run is read from its flushed stdout/JSON, never from its exit code; the matrix driver calls
   `os._exit(0)` after the JSON is on disk (see "How to reproduce").
5. **Machine API level**: only the *lowest* ROCm call layer is in play for iree (the HIP HAL
   driver); the xla path uses a real gfx942 PJRT plugin. Results are specific to ROCm 7.1 + iree
   3.11 + `jax_plugins/xla_rocm7` 0.11.2.

## Per-backend conformance summary

| backend | device | conf examples | numeric PASS | numeric FAIL | deferral/error | crash (excluded) | lower+compile+load total | build n | bench examples |
|---|---|---|---|---|---|---|---|---|---|
| numpy / cpu | `cpu` | 98 | 98 | 0 | 0 | 0 | 1.1 s | 98 | 98 |
| iree / llvm-cpu | `cpu` | 98 | 75 | 0 | 23 | 0 | 18.8 s | 75 | 98 |
| iree / rocm:0 | `rocm:0` | 97 | 70 | 0 | 27 | 1 | 25.5 s | 71 | 97 |
| xla / rocm:0 | `rocm:0` | 98 | 59 | 0 | 39 | 0 | 93.3 s | 59 | 98 |

- The xla conformance row is the `dot_precision="HIGHEST"` run (true fp32); the "build n" column
  uses the TF32 benchmark run's 59 successful builds.
- `lower+compile+load total` is the sum over examples whose staging succeeded (deferred examples
  never reach a build, so they contribute no build time); see the build section.
- `iree / rocm:0` recorded 97 of the 98 registry examples: **`while_fib` kills the worker process**
  (SIGSEGV) and is excluded so the remaining results survive — see "Exclusions" below. Nothing was
  silently dropped.
- **Zero `torch_pass=False` anywhere**: every executed torch comparison passed (62 numpy / 46
  llvm-cpu / 46 rocm). `torch_pass=None` on the rest = no torch reference for that example, or the
  example never got that far.

### Notable precision

- **numpy / cpu** — 98/98 PASS against the pure-numpy references, 70 of them exactly bit-exact
  (`max_abs == 0.0`); the non-zero remainder is the fp64-finite-difference grad/vmap/e2e family
  (`grad_mix` 3.4e-6, `vmap_grad_mlp` 5.9e-6, `conv2d_large` 1.07e-4 — shared-kernel accumulation
  order, same class as before).
- **iree / llvm-cpu** — 0 numeric failures in 75 executed examples, 27 bit-exact. Spot:
  `transformer` max_abs 7.2e-4 (covered by its `tolerance=1e-3`), `matmul_1024` 2.06e-4 (its
  `atol=5e-4`), `attention` 2.4e-7, `mha_block` 2.4e-4.
- **iree / rocm:0** — 0 numeric failures in 70 executed examples, and **28 of the 70 are bit-exact
  (`max_abs` exactly 0.0)**, including `matmul`, the `cond_*` family, `while_cumsum`,
  `while_cond_combo`, `cond_while_pipeline`, `op_*` elementwise families. Larger: `matmul_1024`
  max_abs 2.14e-4 (same fp32 accumulation class as llvm-cpu's 2.06e-4 — ROCm is fp32-accurate, no
  TF32), `transformer` 8.4e-4, `conv2d_large` 1.4e-4, `nbody` 7.6e-6, `mha_block` 7.6e-6,
  `layernorm_large` 9.5e-7, `attention` 2.4e-7. **iree-on-ROCm is numerically correct for every
  example it manages to run.**
- **xla / rocm:0** — with `dot_precision="HIGHEST"`, **59/59 executed examples PASS, 0 numeric
  FAILures, 29 bit-exact**. The 13 `dot`-heavy examples that fail under the DEFAULT TF32 lowering
  all pass under true fp32 (see "Root cause 2"). `transformer`
  keeps a thin margin: max_abs 8.684e-4 against its `tolerance=1e-3` (1.15×) — ordinary fp32
  accumulation-order noise, not a TF32 artifact, and the same class iree shows (8.4e-4).

## Timing matrix — best-of-20 `etl_ms`, one column per backend

`ERR` = explicit deferral/error (recorded, never silent); `CRASH` would mark a process death (none
in the cells below — the one crash is `while_fib` on iree/rocm, shown as `ERR` because no JSON row
exists for it). `torch-cpu` = CPU baseline (caveat 1) as measured in the numpy/cpu run — each run
times its own torch reference and the values vary run-to-run (e.g. `transformer` 5.56 ms here vs
8.50 ms in the iree/rocm run, 2.26× behind iree/rocm's 3.76 ms there). xla values are the DEFAULT
**TF32** path.

| example | numpy/cpu | iree/llvm-cpu | iree/rocm:0 | xla/rocm:0 (TF32) | torch-cpu |
|---|---|---|---|---|---|
| matmul | 0.01992 | 0.1217 | 0.2028 | 0.284 | 0.0107 |
| conv2d | 0.08676 | 0.04723 | 0.1897 | ERR | 0.4791 |
| conv2d_same | 0.1037 | 0.04536 | 0.208 | ERR | 0.3843 |
| conv2d_stride2 | 0.09806 | 0.04599 | 0.1777 | ERR | 0.03948 |
| elementwise_fusion | 0.04055 | 0.05625 | 0.2761 | 0.3507 | 0.009031 |
| softmax | 0.04875 | 0.1354 | 0.1174 | 0.1786 | 0.006954 |
| layernorm | 0.06475 | 0.02514 | 0.119 | 0.1731 | 0.008802 |
| mlp | 0.06125 | 0.221 | 0.3686 | 0.5179 | 0.02913 |
| cumsum | 0.02205 | ERR | ERR | ERR | 0.004785 |
| attention | 0.06999 | 0.05175 | 0.2601 | 0.3762 | 0.03075 |
| grad_mlp | 0.1732 | 0.08459 | 0.4688 | 0.6593 | 0.1289 |
| grad_mix | 2.307 | 0.06827 | 0.3926 | 0.4656 | 1.068 |
| grad_stopgrad | 0.1017 | 0.03393 | 0.1779 | 0.2851 | 0.05623 |
| grad_structural | 0.3239 | ERR | ERR | ERR | 0.1759 |
| vmap_softmax | 0.04356 | ERR | ERR | ERR | 0.007229 |
| vmap_layernorm | 0.07271 | ERR | ERR | ERR | 0.009637 |
| vmap_linear | 0.05739 | 0.03724 | 0.2562 | ERR | 0.01204 |
| vmap_mlp | 0.05335 | 0.06135 | 0.3962 | ERR | 0.02796 |
| vmap_attention | 0.09096 | ERR | ERR | ERR | 0.0323 |
| vmap_grad_mlp | 0.1807 | 0.07938 | 0.4576 | ERR | 0.1683 |
| matmul_1024 | 1.252 | 4.02e+03 | 1.01 | 0.7706 | 4.851 |
| conv2d_large | 40.72 | 1.12e+03 | 1.021 | ERR | 2.107 |
| layernorm_large | 5.899 | 6.453 | 1.526 | 1.15 | 0.4599 |
| vmap_mlp_large | 0.2026 | 1.332 | 1.137 | ERR | 0.03537 |
| op_erf | 0.1271 | ERR | ERR | ERR | 0.004617 |
| op_gelu | 0.1243 | ERR | ERR | ERR | 0.01294 |
| op_tril_triu | 0.03621 | ERR | ERR | ERR | 0.01449 |
| op_argmax_argmin | 0.02566 | 0.04347 | 0.142 | 0.2501 | 0.00813 |
| op_bitwise | 0.03912 | 0.04015 | 0.1868 | 0.3642 | 0.01088 |
| op_logical | 0.03011 | 0.03737 | 0.1895 | 0.4142 | 0.009113 |
| op_compare | 0.03817 | 0.03915 | 0.1912 | 0.3543 | 0.01007 |
| op_power_remainder | 0.0353 | 0.03804 | 0.1858 | 0.3374 | 0.008704 |
| op_gather_scatter | 0.04075 | 0.04106 | 0.2504 | 0.3762 | - |
| op_pad_slice | 0.04769 | 0.03079 | 0.133 | 0.1886 | 0.009027 |
| op_stop_gradient | 0.009341 | 0.01559 | 0.07432 | 0.1655 | - |
| op_select | 0.02712 | 0.02401 | 0.1202 | 0.1878 | 0.007197 |
| op_broadcast | 0.01516 | 0.02356 | 0.1206 | 0.1689 | 0.00278 |
| batched_3d_matmul | 0.01796 | 0.03371 | 0.191 | 0.268 | 0.01162 |
| matmul_3d_2d_shared | 0.01847 | 0.03351 | 0.1746 | 0.2667 | 0.006717 |
| matvec | 0.02923 | 0.03214 | 0.1853 | 0.2726 | 0.005223 |
| matmul_transposed | 0.02382 | 0.05475 | 0.1903 | 0.2895 | 0.006595 |
| matmul_square | 0.02948 | 0.2047 | 0.1926 | 0.2941 | 0.01682 |
| diagonal_matmul | 0.02935 | 0.03528 | 0.2325 | 0.3218 | 0.007183 |
| cond_basic | 0.02822 | 0.03559 | 0.3988 | 0.2535 | - |
| cond_multi_output | 0.04745 | 0.05456 | 0.4798 | 0.3621 | - |
| cond_nested | 0.03518 | 0.05039 | 0.5843 | 0.3299 | - |
| while_cumsum | 0.1491 | 0.09159 | 0.4997 | 0.2468 | - |
| while_fib | 0.1377 | 0.08647 | ERR | 0.132 | - |
| while_cond_combo | 0.2647 | 0.1576 | 1.072 | 0.2516 | - |
| while_poly | 0.1765 | 0.1028 | ERR | 0.3636 | - |
| cond_while_pipeline | 0.1572 | 0.2057 | 0.6585 | 0.4079 | - |
| grad_erf | 0.08181 | ERR | ERR | ERR | - |
| grad_gelu | 0.124 | ERR | ERR | ERR | - |
| grad_cumsum | 0.06168 | ERR | ERR | ERR | - |
| grad_tril | 0.04973 | ERR | ERR | ERR | - |
| grad_triu | 0.04753 | ERR | ERR | ERR | - |
| grad_power_frac | 0.09637 | 0.03012 | 0.1178 | 0.1803 | - |
| grad_bitwise_zero | 0.1145 | 0.02516 | 0.117 | 0.1646 | - |
| vmap_elementwise | 0.01602 | 0.03578 | 0.1195 | ERR | 0.00315 |
| vmap_nested | 0.01865 | 0.03364 | 0.1974 | ERR | 0.008657 |
| vmap_shared_weights_dot | 0.03152 | 0.03198 | 0.1927 | ERR | 0.005692 |
| vmap_pad | 0.03579 | ERR | ERR | ERR | 0.006167 |
| vmap_concat | 0.02264 | 0.04396 | 0.179 | ERR | 0.005795 |
| custom_l2norm | 0.02934 | ERR | ERR | ERR | 0.01062 |
| custom_l2norm_grad | 0.119 | ERR | ERR | ERR | 0.07134 |
| custom_l2norm_vmap | 0.03454 | ERR | ERR | ERR | 0.00962 |
| custom_relu2 | 0.02314 | ERR | ERR | ERR | 0.005127 |
| custom_relu2_grad | 0.2285 | ERR | ERR | ERR | 0.1223 |
| op_external | 0.03751 | 0.07262 | 0.4527 | ERR | 0.01018 |
| xla_elementwise_chain | 0.05854 | 0.04625 | 0.1997 | 0.2736 | - |
| xla_softmax | 0.04481 | 0.07999 | 0.1323 | 0.1783 | - |
| xla_mlp | 0.05158 | 0.08432 | 0.3585 | 0.4815 | - |
| xla_batched_dot | 0.03131 | 0.06311 | 0.234 | 0.396 | - |
| xla_reductions | 0.05748 | 0.06508 | 0.1518 | 0.1829 | - |
| transformer | 25.57 | 1.05e+04 | 3.758 | 2.183 | 5.561 |
| nbody | 40.97 | 42.93 | 0.3106 | 0.373 | 1.005 |
| mha_block | 0.1888 | 0.2428 | 0.2967 | 0.4025 | 0.07359 |
| ffn_block | 0.2522 | ERR | ERR | ERR | 0.04307 |
| mha_ffn_block | 0.4644 | ERR | ERR | ERR | 0.1256 |
| lstm_cell | 0.1052 | 0.09869 | 0.4489 | 0.6343 | 0.02854 |
| gru_cell | 0.1144 | 0.06116 | 0.5638 | 0.7283 | 0.03317 |
| resnet_conv_block | 0.3809 | 0.4108 | 0.3039 | ERR | 0.1362 |
| conv_block_stride2 | 0.3093 | 0.2281 | 0.3374 | ERR | - |
| deep_mlp_block | 0.09418 | 0.09288 | 0.3832 | 0.5098 | 0.02649 |
| mlp_block_residual | 0.0793 | 0.04834 | 0.2606 | 0.3533 | 0.01858 |
| pso_step | 0.07827 | 0.04902 | 0.2988 | 0.4357 | - |
| kmeans_iter | 0.4374 | 0.05329 | 0.2392 | 0.295 | - |
| als_step | 0.04799 | ERR | ERR | ERR | - |
| cg_step | 0.1026 | 0.06285 | 0.3432 | 0.4504 | - |
| power_iter | 0.047 | 0.04302 | 0.2266 | 0.2652 | - |
| softmax_reg_step | 0.1469 | 0.06802 | ERR | 0.4534 | - |
| e2e_train_mlp | 2.546 | 1.627 | 7.805 | 11.87 | - |
| e2e_train_convnet | 1.999 | 1.308 | 4.797 | ERR | - |
| e2e_train_transformer | 42.5 | ERR | ERR | ERR | - |
| e2e_infer_transformer | 1.109 | 0.2196 | ERR | 0.7109 | - |
| e2e_pso_optimize | 1.516 | 0.1994 | ERR | 0.7402 | - |
| e2e_kmeans | 0.9178 | 0.1729 | 0.7006 | 0.3551 | - |
| e2e_power_iteration | 0.6648 | 0.245 | 0.9813 | 0.4138 | - |

### ROCm vs llvm-cpu, aggregate

- **70 comparable cells** (examples whose staging succeeded on both). ROCm is faster in **9**,
  slower in 61.
- The 9 ROCm wins are the examples whose llvm-cpu time is dominated by *compute* (or by an
  llvm-cpu path that is disproportionately slow): `matmul_1024` **3.98e+03×**
  (1.01 ms vs 4022 ms), `transformer` **2.79e+03×** (3.76 vs 10483), `conv2d_large` **1.1e+03×**
  (1.02 vs 1125), `nbody` **138×** (0.31 vs 42.9), `layernorm_large` **4.23×**, `resnet_conv_block`
  **1.35×**, `vmap_mlp_large` **1.17×**, `softmax` **1.15×**, `matmul_square` **1.06×**.
- Everything else loses to llvm-cpu because the ROCm per-call floor (~0.12–0.6 ms, set by transfer
  + dispatch) dwarfs a sub-0.1 ms llvm-cpu graph: `attention` 0.260 vs 0.0518 ms (0.199×),
  `mha_block` 0.297 vs 0.243 (0.819×), `e2e_pso_optimize` and `e2e_infer_transformer` are the
  reverse direction (llvm-cpu is slower at 0.199/0.220 vs ROCm ERR — see exclusions).
- **ROCm also beats xla/rocm(TF32)** in 43 of the 54 comparable cells; xla is faster in only 11
  (`while_cond_combo` 4.26×, `e2e_power_iteration` 2.37×, `while_cumsum` 2.02×, `e2e_kmeans` 1.97×,
  `cond_nested` 1.77×, `transformer` 1.72×, `cond_while_pipeline` 1.61×, `cond_basic` 1.57×,
  `layernorm_large` 1.33×, `cond_multi_output` 1.33×, `matmul_1024` 1.31×). Both run on the same
  GPU; the xla column is bounded by its own launch/host floor (~0.13–0.27 ms/call), so it wins only
  where the extra TF32 compute edge dominates.

### xla / rocm:0 — the `dot_precision="HIGHEST"` (true fp32) timings

17 `dot`-heavy examples benchmarked with `dot_precision="HIGHEST"`, next to their DEFAULT (TF32)
times from the matrix above. `H/TF32` ≈ 1 means the precision fix is free on these sizes.

| example | TF32 ms | HIGHEST ms | H/TF32 |
|---|---|---|---|
| matmul | 0.2840 | 0.2771 | 0.98 |
| matmul_square | 0.2941 | 0.2627 | 0.89 |
| matmul_1024 | 0.7706 | 0.7633 | 0.99 |
| matmul_3d_2d_shared | 0.2667 | 0.2654 | 0.99 |
| matmul_transposed | 0.2895 | 0.2646 | 0.91 |
| batched_3d_matmul | 0.2680 | 0.2716 | 1.01 |
| diagonal_matmul | 0.3218 | 0.4347 | 1.35 |
| matvec | 0.2726 | 0.3217 | 1.18 |
| mlp | 0.5179 | 0.5202 | 1.00 |
| deep_mlp_block | 0.5098 | 0.4971 | 0.98 |
| transformer | 2.1833 | 2.2390 | 1.03 |
| mha_block | 0.4025 | 0.4339 | 1.08 |
| attention | 0.3762 | 0.3734 | 0.99 |
| layernorm_large | 1.1499 | 1.0814 | 0.94 |
| grad_mlp | 0.6593 | 0.7054 | 1.07 |
| e2e_train_mlp | 11.8728 | 15.4888 | 1.30 |
| mlp_block_residual | 0.3533 | 0.3568 | 1.01 |

Median `H/TF32` = 1.00 (range 0.89–1.35): on these sizes xla/rocm is launch/host-overhead bound
(floor ~0.13–0.27 ms/call) and never saturates the TF32 tensor cores, so requesting true fp32 costs
essentially nothing.

## Headline numbers

Best-of-20 `etl_ms` and `build_ms` (lower + compile + load) per backend for representative tasks.
`defer` = the backend refused the graph with an explicit `BackendError` (never a silent skip);
`-` = the task never got that far (crashed upstream/predecessor). The xla build column carries
per-chunk plugin-init inflation for the first example of each chunk (see the build section).

| example | numpy/cpu ms | numpy build | iree/llvm-cpu ms | iree-cpu build | iree/rocm:0 ms | iree-rocm build | xla/rocm:0(TF32) ms | xla build |
|---|---|---|---|---|---|---|---|---|
| matmul_1024 | 1.252 | 1.08 | 4.02e+03 | 183 | 1.01 | 480 | 0.7706 | 7.73e+03 |
| matmul_square | 0.0295 | 1.14 | 0.2047 | 195 | 0.1926 | 330 | 0.2941 | 2.59e+03 |
| transformer | 25.57 | 21.8 | 1.05e+04 | 477 | 3.758 | 1.23e+03 | 2.183 | 4.25e+03 |
| attention | 0.0700 | 4.00 | 0.0518 | 248 | 0.2601 | 312 | 0.3762 | 1.03e+04 |
| conv2d_large | 40.72 | 0.93 | 1.12e+03 | 206 | 1.021 | 558 | defer | - |
| resnet_conv_block | 0.3809 | 4.72 | 0.4108 | 274 | 0.3039 | 438 | defer | - |
| nbody | 40.97 | 8.81 | 42.93 | 264 | 0.3106 | 342 | 0.373 | 244 |
| layernorm_large | 5.899 | 4.82 | 6.453 | 230 | 1.526 | 277 | 1.15 | 73.9 |
| e2e_train_mlp | 2.546 | 18.8 | 1.627 | 518 | 7.805 | 684 | 11.87 | 1.65e+03 |
| e2e_train_convnet | 1.999 | 14.9 | 1.308 | 644 | 4.797 | 1.08e+03 | defer | - |
| e2e_train_transformer | 42.5 | 277 | defer | - | defer | - | defer | - |
| e2e_infer_transformer | 1.109 | 16.9 | 0.2196 | 311 | defer | - | 0.7109 | 1.21e+03 |
| e2e_pso_optimize | 1.516 | 14.6 | 0.1994 | 295 | defer | - | 0.7402 | 112 |
| e2e_kmeans | 0.9178 | 12.5 | 0.1729 | 265 | 0.7006 | 367 | 0.3551 | 8.33e+03 |
| e2e_power_iteration | 0.6648 | 4.74 | 0.245 | 220 | 0.9813 | 272 | 0.4138 | 59.6 |
| mha_block | 0.1888 | 11.0 | 0.2428 | 405 | 0.2967 | 397 | 0.4025 | 1.04e+03 |

Cross-backend speedup ratios (ratio = reference `etl_ms` / backend `etl_ms`; **> 1 means the
accelerator did the task faster than the reference**). The reference is the numpy/cpu backend:

| example | numpy/cpu ÷ iree/llvm-cpu | numpy/cpu ÷ iree/rocm:0 | numpy/cpu ÷ xla/rocm:0(TF32) | iree/llvm-cpu ÷ iree/rocm:0 |
|---|---|---|---|---|
| matmul_1024 | 0.000311× | 1.24× | 1.62× | 3.98e+03× |
| matmul_square | 0.144× | 0.153× | 0.100× | 1.06× |
| transformer | 0.00244× | 6.80× | 11.7× | 2.79e+03× |
| attention | 1.35× | 0.269× | 0.186× | 0.199× |
| conv2d_large | 0.0362× | 39.9× | - | 1.10e+03× |
| resnet_conv_block | 0.927× | 1.25× | - | 1.35× |
| nbody | 0.954× | 132× | 110× | 138× |
| layernorm_large | 0.914× | 3.86× | 5.13× | 4.23× |
| e2e_train_mlp | 1.56× | 0.326× | 0.214× | 0.209× |
| e2e_train_convnet | 1.53× | 0.417× | - | 0.273× |
| e2e_train_transformer | - | - | - | - |
| e2e_infer_transformer | 5.05× | - | 1.56× | - |
| e2e_pso_optimize | 7.60× | - | 2.05× | - |
| e2e_kmeans | 5.31× | 1.31× | 2.58× | 0.247× |
| e2e_power_iteration | 2.71× | 0.677× | 1.61× | 0.250× |
| mha_block | 0.777× | 0.636× | 0.469× | 0.819× |

`conv2d_large` is the standout in the other direction: against the *pure-numpy im2col reference*
(the harness's `speedup_vs_numpy`) iree/rocm is **4032×** (4116 ms → 1.02 ms); against the etl
numpy *backend* it is 39.9×; against llvm-cpu 1.1e+03×.

Stage wall-times for a full run of the 98-example registry (conformance, then benchmark):

| run | conformance wall | benchmark wall | notes |
|---|---|---|---|
| numpy / cpu | 13.5 s | 103.5 s | includes 62 CPU-torch reference timings |
| iree / llvm-cpu | 50.7 s | 474.4 s | benchmark wall dominated by the multi-second llvm-cpu examples |
| iree / rocm:0 | 123.9 s | 124.1 s | conformance wall includes the `while_poly` teardown hang |
| xla / rocm:0 | 172.0 s (HIGHEST) | 183.7 s (TF32) | long walls = per-example cold PJRT compiles in chunked subprocesses; the HIGHEST benchmark subset alone was 91.6 s |

The xla conformance wall is inflated by the chunked-subprocess methodology (each chunk re-loads the
482 MB plugin and re-initialises the ROCm service); it is not a measure of per-example compile cost
— use the build table below for that.

## Build (lower + compile + load) wall-times

| backend | builds | min | median | p90 | max | total |
|---|---|---|---|---|---|---|
| numpy / cpu | 98 | 0.66 | 3.4 | 16.7 | 277 | 1.1 s |
| iree / llvm-cpu | 75 | 89 | 211 | 392 | 887 | 18.8 s |
| iree / rocm:0 | 71 | 167 | 286 | 473 | 1561 | 25.5 s |
| xla / rocm:0 | 59 | 17 | 90 | 5.87e+03 | 1.03e+04 | 93.3 s |

`iree / rocm:0` has the highest *median* build in the matrix (286 ms) and a 1561 ms worst case
(`grad_mix`); llvm-cpu is a notch cheaper (211 ms median / 887 ms max). **xla's per-build cost is
low (90 ms median) but its totals are inflated by chunk restarts**: each chunked subprocess pays
~5–10 s of one-time plugin/ROCm initialisation, charged to the first successfully-built example in
that chunk, which is why the p90/max columns run to ~10 s rather than ~150 ms. Use the median for
per-build comparisons and the wall-times table for run-level cost.

Slowest 8 lower+compile+load builds per compiler backend (ms):

| backend | slowest examples |
|---|---|
| iree / llvm-cpu | `grad_mix` 887, `e2e_train_convnet` 644, `e2e_train_mlp` 518, `transformer` 477, `cond_while_pipeline` 416, `while_cond_combo` 411, `mha_block` 405, `while_cumsum` 392 |
| iree / rocm:0 | `grad_mix` 1556, `transformer` 1231, `e2e_train_convnet` 1078, `e2e_train_mlp` 684, `conv2d_large` 558, `cond_while_pipeline` 489, `matmul_1024` 480, `while_cond_combo` 473 |
| xla / rocm:0 | first-of-chunk plugin restarts dominate the tail: `attention` 10297, `e2e_kmeans` 8330, `matmul_transposed` 8197, `gru_cell` 8138, `xla_batched_dot` 7793, `matmul_1024` 7734, `softmax_reg_step` 5870, `cg_step` 5833 |

## Deferrals, errors and exclusions — exactly what was dropped, per backend

Everything below is an **explicit, recorded error** (an `ExampleResult.error`) — never a silent
skip. Error text is reproduced as recorded (truncated where the harness truncates).

**numpy / cpu** — 0 distinct error(s), 0 example(s) deferred/failed.

- *(no deferrals/errors)*

**iree / llvm-cpu** — 11 distinct error(s), 23 example(s) deferred/failed.

- `stablehlo export: op 'gelu' ... is not supported in v1` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `capability drift: the iree backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'cumsum' ... is not supported in v1` — 2: `cumsum`, `grad_cumsum`
- `stablehlo export: op 'solve' ... is not supported in v1` — 2: `als_step`, `grad_structural`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'),)` — 2: `vmap_layernorm`, `vmap_softmax`
- `stablehlo export: op 'erf' ... is not supported in v1` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' ... is not supported in v1` — 2: `op_tril_triu`, `grad_tril`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'), 8)` — 1: `vmap_attention`
- `stablehlo export: op 'triu' ... is not supported in v1` — 1: `grad_triu`
- `stablehlo export: op 'pad' ... operand shape has dynamic dims (Dim('batch'), 16)` — 1: `vmap_pad`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'), 4)` — 1: `custom_l2norm_vmap`

**iree / rocm:0** — 15 distinct error(s), 27 example(s) deferred/failed, 1 crashed (excluded).

- `stablehlo export: op 'gelu' ... is not supported in v1` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `capability drift: the iree backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'cumsum' ... is not supported in v1` — 2: `cumsum`, `grad_cumsum`
- `stablehlo export: op 'solve' ... is not supported in v1` — 2: `als_step`, `grad_structural`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'),)` — 2: `vmap_layernorm`, `vmap_softmax`
- `stablehlo export: op 'erf' ... is not supported in v1` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' ... is not supported in v1` — 2: `op_tril_triu`, `grad_tril`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'), 8)` — 1: `vmap_attention`
- `Error invoking function: c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:786: ABORTED; ... hal.fence.await` — 1: `while_poly`
- `stablehlo export: op 'triu' ... is not supported in v1` — 1: `grad_triu`
- `stablehlo export: op 'pad' ... operand shape has dynamic dims (Dim('batch'), 16)` — 1: `vmap_pad`
- `stablehlo export: op 'reshape' ... keepdims reshape operand has dynamic dims (Dim('batch'), 4)` — 1: `custom_l2norm_vmap`
- `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm)` — 1: `softmax_reg_step`
- `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm)` — 1: `e2e_infer_transformer`
- `Error invoking function: c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:770: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700)` — 1: `e2e_pso_optimize`

The 27 error rows decompose as: **19 shared StableHLO v1 export deferrals** (gelu 5, reshape/keepdims
4, cumsum 2, solve 2, erf 2, tril 2, triu 1, pad 1) + **4** `block_call` capability drift + **2**
`iree-compile ... targets: rocm` failures + **2** hip `RuntimeError`s (`while_poly`,
`e2e_pso_optimize`).

| crashed example | rc | phase | evidence |
|---|---|---|---|
| `while_fib` | -11 | `run` (SIGSEGV, zero-byte log) | `C_iree_rocm.json` `crashes[0]` |

**xla / rocm:0** — 12 distinct error(s), 39 example(s) deferred/failed.

- `capability drift: the xla backend cannot execute dynamic shapes (function signature has symbolic/runtime-dynamic dimensions)` — 13: `vmap_softmax`, `vmap_layernorm`, `vmap_linear`, `vmap_mlp`, `vmap_attention`, `vmap_grad_mlp`, `vmap_mlp_large`, `vmap_elementwise`, `vmap_nested`, `vmap_shared_weights_dot`, `vmap_pad`, `vmap_concat`, `custom_l2norm_vmap`
- `stablehlo export: op 'gelu' ... is not supported in v1` — 5: `op_gelu`, `grad_gelu`, `ffn_block`, `mha_ffn_block`, `e2e_train_transformer`
- `the xla PJRT plugin failed at PJRT_Client_Compile: All backends failed to find supported configs: - INTERNAL: GetMIOpenConvolveAlgorithms failure` — 4: `conv2d`, `conv2d_same`, `conv2d_stride2`, `conv2d_large`
- `capability drift: the xla backend cannot execute custom block ops (block_call)` — 4: `custom_l2norm`, `custom_l2norm_grad`, `custom_relu2`, `custom_relu2_grad`
- `stablehlo export: op 'cumsum' ... is not supported in v1` — 2: `cumsum`, `grad_cumsum`
- `stablehlo export: op 'solve' ... is not supported in v1` — 2: `grad_structural`, `als_step`
- `stablehlo export: op 'erf' ... is not supported in v1` — 2: `op_erf`, `grad_erf`
- `stablehlo export: op 'tril' ... is not supported in v1` — 2: `op_tril_triu`, `grad_tril`
- `the xla PJRT plugin failed at PJRT_Client_Compile: Failed to get configs for: 2 out of 5 instructions` — 2: `resnet_conv_block`, `e2e_train_convnet`
- `stablehlo export: op 'triu' ... is not supported in v1` — 1: `grad_triu`
- `capability drift: the xla backend cannot execute external_call op 'bench_row_softmax' — adapter host-dispatch for external kernels is not yet wired` — 1: `op_external`
- `the xla PJRT plugin failed at PJRT_Client_Compile: Failed to get configs for: 3 out of 6 instructions` — 1: `conv_block_stride2`

### iree / rocm:0 crash exclusions (1 process death)

Exactly one example kills the worker process on iree/rocm:0. The matrix runs each example in a
chunked subprocess with bisection, so the death costs only that example; the crash is then
confirmed by re-running the example alone in conformance-only mode.

| example | rc | where it died | evidence |
|---|---|---|---|
| `while_fib` | -11 | `run` → `iree/runtime/function.py:147 _invoke` → C stack through `libamdhip64.so`; deterministic (~1.45 s), zero-byte log, no JSON produced | `C_iree_rocm.json` `crashes[0]` |

`while_fib` (the only zero-spec, constant-only example) is a genuine segfault of the iree ROCm HAL
on a while-loop graph: it also takes the process down on iree/cuda and passes on llvm-cpu
(0.086 ms, max_abs 0.0). One further rocm example **hangs at interpreter teardown** after its work
is done — `while_poly` records its hip `RuntimeError` in the JSON and finishes the chunk, then never
exits (recorded in the JSON's `nonzero_rc`, `rc=999`, not in `crashes`).

**Excluded per backend, in one line each**: numpy/cpu — none. iree/llvm-cpu — none. iree/rocm:0 —
`while_fib` (deterministic segfault). xla/rocm:0 — none (the plugin aborts only at *interpreter
teardown*, after all work is on disk; see below).

### iree/rocm-only failures that pass on llvm-cpu

Four examples run correctly on llvm-cpu but fail on the ROCm target — the delta between the 23
llvm-cpu deferrals and the 27 rocm ones:

| example | failure | evidence line |
|---|---|---|
| `softmax_reg_step` | `iree-compile failed to compile the StableHLO program for the iree backend (targets: rocm)` | iree-compile error |
| `e2e_infer_transformer` | same iree-compile failure (the scatter/while decoding graph) | iree-compile error |
| `while_poly` | `c/runtime/src/iree/hal/drivers/hip/event_semaphore.c:786: ABORTED; ... hal.fence.await` | runtime invoke |
| `e2e_pso_optimize` | `event_semaphore.c:770: INTERNAL; HIP driver error 'hipErrorIllegalAddress' (700)` | runtime invoke |

`while_cumsum`, `while_cond_combo` and `cond_while_pipeline` (constant-carry / float-condition
while loops) all run fine on rocm (0.50 / 1.07 / 0.66 ms, three of them non-error), so the
constant-carry/float-condition shape alone is NOT sufficient to trigger the abort — what fails is
the optimisation-loop subset (`while_poly`, `e2e_pso_optimize`) plus the scatter-bearing decoding
graph, consistent with HAL bufferization/binding of carried mutable buffers rather than the
condition type.

These examples complete and PASS on iree/rocm:0 and are **not** exclusions: `op_gather_scatter`
(0.250 ms), `op_stop_gradient` (0.074 ms), `grad_stopgrad` (0.178 ms), `e2e_kmeans` (0.701 ms) and
`e2e_power_iteration` (0.981 ms) all ran to completion on iree/rocm:0.

### xla / rocm:0 — plugin behaviour and failure classes

The xla backend runs with `ETL_PJRT_PLUGIN` pointing at
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
code (the matrix driver calls `os._exit(0)` after the JSON is on disk).

Failure classes (39 examples, from the error section above):

- **Dynamic shapes (13)** — every `vmap_*` example including `vmap_mlp_large` and
  `custom_l2norm_vmap`: `capability drift: the xla backend cannot execute dynamic shapes`. The xla
  adapter is static-shape-only, so the whole vectorize family is out — this is the largest single
  class and is inherent to the adapter, not to the plugin.
- **MIOpen convolution (7)** — `conv2d`, `conv2d_same`, `conv2d_stride2`, `conv2d_large`
  (`GetMIOpenConvolveAlgorithms failure`), `resnet_conv_block` + `e2e_train_convnet`
  (`2 out of 5 instructions`), `conv_block_stride2` (`3 out of 6`). The plugin's ROCm build cannot
  find a working MIOpen conv algorithm for these shapes → **no conv runs on xla/rocm at all** here.
- **Shared export deferrals (19)** — `gelu` ×5, `block_call` ×4, `cumsum` ×2, `solve` ×2, `erf` ×2,
  `tril` ×2, `triu` ×1: identical to the other compiler backends (StableHLO v1 export gaps).
- **`op_external` (1)** — `adapter host-dispatch for external kernels is not yet wired` for xla
  (works on iree host-dispatch and numpy).

## Root cause 1 — iree/rocm per-call latency is now near the launch floor

The iree ROCm upload path uses a thread-local cached HAL device plus a reusable source buffer and
`queue_copy` — not a fresh `rt.get_driver("hip").create_device(...)` per upload. `probe_rocm_staging2.py`
(best-of-N, warmup) measures the result on the `matmul` example:

| measurement | best ms | median ms |
|---|---|---|
| host→device upload `etl.tensor(x).to(Device("rocm",0))`, 4 KB | 0.0520 | 0.0543 |
| … `.to(rocm)` alone, 4 KB | 0.0497 | 0.0511 |
| host→device upload `etl.tensor(x).to(Device("rocm",0))`, 4 MB | 0.4261 | 0.4385 |
| … `.to(rocm)` alone, 4 MB | 0.4221 | 0.4261 |
| full harness pattern `place + run` (b) | 0.1775 | 0.1817 |
| full harness pattern `place + run + readback` (b') | 0.2671 | 0.2781 |
| placement only (2 × `etl.tensor(x).to(rocm:0)`) | 0.1013 | 0.1033 |
| readback only (D2H `.to(cpu)`, 1 leaf) | 0.0693 | 0.0747 |
| run only, device-resident in **and** out (c) | 0.0584 | 0.0609 |

The 4 KB and 4 MB round-trips are bit-exact; the probe's correctness check reports `max_abs=0` vs
the numpy reference.

Reading: host→device upload is now **~0.05 ms fixed plus ~0.11 ms/MB** (4 MB → 0.43 ms), with no
large fixed tax. The harness's per-call pattern (b) costs 0.178 ms, of which ~0.10 ms is the two
explicit placements and the device-resident run itself is 0.058 ms. That is what the fresh timing
matrix shows: `iree/rocm` small graphs land at ~0.12–0.6 ms — a transfer + dispatch + launch
overhead term — while compute-dominated graphs win outright (`matmul_1024` 1.01 ms,
`transformer` 3.76 ms, `conv2d_large` 1.02 ms).

**Iteration pattern for stateful loops:** a loop that keeps state on the device — build the graph
once, feed each run's device `Tensor` outputs back as the next run's inputs (zero host round-trips,
run-only cost ~0.058 ms here) — avoids even the ~0.17 ms per-call `place + run + readback` pattern.
An explicit per-call `.to(device)` restaging pays the placement/readback terms above on every
iteration; that is why the runner-based `e2e_train_*` examples (one re-stage per iteration, 10
iterations) are the slowest `iree/rocm` cells in the matrix (`e2e_train_mlp` 7.81 ms). This is an
iteration-pattern choice, not a fixed per-call tax.

## Root cause 2 — XLA/ROCm fp32 `dot` runs in TF32 (and the `dot_precision` remedy)

By default the XLA ROCm backend lowers fp32 `stablehlo.dot_general` to a **TF32** tensor-core path
(10-bit mantissa). A simulation cross-check pinned this: an `[n,n]×[n,n]` fp32 matmul's error vs an
fp64 ground truth matches a TF32 operand-truncation simulation to all printed digits (0.193 vs
0.193 at n=1024, 0.0528 vs 0.0528 at n=256, 0.0158 vs 0.0158 at n=64), while iree-on-ROCm stays at
ordinary fp32 accumulation error. This produced **13 numeric conformance FAILs**.

**The remedy is now in the public option surface:** the `lower` option `dot_precision`
(`"DEFAULT"` | `"HIGH"` | `"HIGHEST"`, case-insensitive) makes the shared StableHLO exporter stamp
`precision_config = [#stablehlo<precision HIGHEST>, …]` on every `dot_general`. `"HIGHEST"` forces
true fp32; `"DEFAULT"` emits no attribute (byte-identical to before). With
`dot_precision="HIGHEST"`, **all 13 TF32-affected examples PASS**:

| example | TF32 `max_abs` | HIGHEST `max_abs` |
|---|---|---|
| matmul | 2.575e-2 | 0.0 |
| matmul_square | 3.790e-1 | 0.0 |
| matmul_1024 | 1.228e-1 | 2.060e-4 |
| batched_3d_matmul | 1.085e-2 | 0.0 |
| xla_batched_dot | 1.085e-2 | 1.907e-6 |
| mlp | 2.828e-1 | 1.526e-5 |
| deep_mlp_block | 1.494e-2 | 3.815e-6 |
| grad_mlp | 1.995e-2 | 3.340e-6 |
| attention | 1.250e-3 | 2.384e-7 |
| mha_block | 7.448e-2 | 1.144e-5 |
| gru_cell | 2.012e-3 | 1.192e-7 |
| transformer | 4.648e-1 | 8.684e-4 |
| e2e_train_mlp | 6.306e-3 | 4.431e-7 |

Each HIGHEST number is now ordinary fp32 accumulation error. `transformer` keeps a thin ~1.15×
margin at its `tolerance=1e-3` (8.684e-4) — the same fp32 accumulation-order/fusion noise class
that iree shows (8.4e-4), **not** a TF32 artifact and not removed by `dot_precision`. **No
per-example tolerance was changed.**

**The precision fix is essentially free on these sizes.** The `dot`-heavy subset re-measured under
`dot_precision="HIGHEST"` shows a median HIGHEST/TF32 ratio of ≈ 1.00 (range 0.89–1.35): per-cell
ms TF32 → HIGHEST are matmul 0.2840 → 0.2771, matmul_square 0.2941 → 0.2627, matmul_1024 0.7706 →
0.7633, matmul_transposed 0.2895 → 0.2646, batched_3d_matmul 0.2680 → 0.2716,
matmul_3d_2d_shared 0.2667 → 0.2654, diagonal_matmul 0.3218 → 0.4347, matvec 0.2726 → 0.3217,
mlp 0.5179 → 0.5202, deep_mlp_block 0.5098 → 0.4971, mlp_block_residual 0.3533 → 0.3568,
grad_mlp 0.6593 → 0.7054, transformer 2.1833 → 2.2390, mha_block 0.4025 → 0.4339,
attention 0.3762 → 0.3734, layernorm_large 1.1499 → 1.0814, e2e_train_mlp 11.8728 → 15.4888. The
reason is that xla/rocm here is launch/host-overhead bound (floor ~0.13–0.27 ms/call) and never
saturates the TF32 tensor cores.

**Recommendation:** use `dot_precision="HIGHEST"` as the guidance default for xla/rocm — the
default costs 13 numeric failures for no measurable speedup. iree and tvm parse-and-ignore the
option (results bit-identical to `"DEFAULT"`), and the default behaviour is unchanged for anyone who
does not set it.

## Task E — TVM cannot target ROCm

`apache-tvm` 0.26.0 is installed, but its wheel is built with `USE_ROCM=OFF`:
`tvm.runtime.device("rocm", 0).exist` → **False**. TVM is therefore compile-only on this machine
(no ROCm runtime support, and the LLVM-CPU target only duplicates `iree/llvm-cpu`). No TVM matrix
was run. Record-only.

## Harness note

The device-derived `target_backends` injection is gated correctly now: `etl/bench/_util.py`
`resolve_backend_options` injects `target_backends` **only** for the iree backend family (backend
name starts with `"iree"`), so `--backend xla` and `--backend tvm` run **directly** through the
normal harness/CLI with no wrapper. The xla numbers in this report come from plain
conformance/benchmark calls; `--no-conformance` / `--conformance` and `--backend-option` work as
documented. The remaining xla operational caveat is the plugin's **teardown SIGABRT** (Root cause
2 / caveat 4) — a separate plugin issue, not this one: read the flushed stdout/JSON, not the exit
code. The sibling test pin `../tests/bench/test_backend_options.py` was refreshed to the
iree-family-only contract.

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
#     etl Device("rocm", N) maps to iree HAL device_id = N + 1
./.venv/bin/python -m etl.bench --backend iree --device rocm:0 --torch

# D — xla on the MI300X: no wrapper needed (target_backends is iree-only now).
#     The plugin SIGABRTs at interpreter teardown AFTER the work succeeds, so read the
#     printed report / JSON, never the exit code; the matrix driver runs each chunk in a
#     subprocess that calls os._exit(0) once the JSON is on disk (ETL_BENCH_HARD_EXIT=1).
export ETL_PJRT_PLUGIN=$PWD/.venv/lib/python3.14/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so
# conformance at true fp32 (this is the 59/59 row)
ETL_BENCH_HARD_EXIT=1 ./.venv/bin/python -m etl.bench --backend xla --device rocm:0 \
    --conformance --no-benchmark --backend-option dot_precision=HIGHEST
# the DEFAULT (TF32) benchmark
ETL_BENCH_HARD_EXIT=1 ./.venv/bin/python -m etl.bench --backend xla --device rocm:0 \
    --benchmark --no-conformance
# the dot-heavy HIGHEST benchmark subset
ETL_BENCH_HARD_EXIT=1 ./.venv/bin/python -m etl.bench --backend xla --device rocm:0 \
    --benchmark --no-conformance --backend-option dot_precision=HIGHEST \
    --examples matmul,matmul_square,matmul_1024,matmul_3d_2d_shared,matmul_transposed,\
batched_3d_matmul,diagonal_matmul,matvec,mlp,deep_mlp_block,transformer,mha_block,attention,\
layernorm_large,grad_mlp,e2e_train_mlp,mlp_block_residual

# E — TVM probe (expected: no ROCm support)
./.venv/bin/python -c "import tvm; print(tvm.runtime.device('rocm',0).exist)"   # -> False
```

**Crash isolation (required for iree/rocm).** Do not run the ROCm iree matrix as one process:
`while_fib` SIGSEGVs the interpreter and a mid-run death loses everything after it. The matrix in
this report was produced by a per-example chunk driver (run N examples, bisect the chunk if it
dies, record `rc` + stderr, resume) — conceptually:

```bash
for ex in $(./.venv/bin/python -c "import etl.bench as b; print(' '.join(b.list_examples()))"); do
  ./.venv/bin/python -m etl.bench --backend iree --device rocm:0 --conformance --no-benchmark \
      --examples "$ex" >> run.log 2>&1 || echo "EXCLUDED $ex rc=$?" >> exclusions.txt
done
```

Exclusions recorded in this report: iree/rocm `while_fib` (deterministic segfault; `while_poly`
additionally hangs at teardown after recording its error).

## Conclusions

1. **iree-on-ROCm is numerically correct and competitive** — 70/70 executed examples pass, 28
   bit-exact, and its per-call floor is now ~0.12–0.6 ms (host→device upload ~0.05 ms fixed +
   ~0.11 ms/MB). It beats llvm-cpu in 9 of 70 comparable cells, and those 9 include every
   compute-dominated graph (`matmul_1024` 3.98e+03×, `transformer` 2.79e+03×, `conv2d_large`
   1.1e+03×, `nbody` 138×, `layernorm_large` 4.23×). Every remaining loss is a small graph whose
   llvm-cpu time is already below the ROCm launch floor. For stateful loops the device-resident
   iteration pattern (feed device outputs back as the next inputs) removes even that floor.
2. **xla-on-ROCm is fast, but its default TF32 fp32 `dot` is inexact; `dot_precision="HIGHEST"`
   fixes that at no measurable cost** — the 13 fp32 numeric failures collapse to 0 (59/59 pass, 29
   bit-exact) with a median HIGHEST/TF32 timing ratio of 1.00. Use `dot_precision="HIGHEST"` as the
   xla/rocm default. xla remains static-shape-only and cannot run conv (MIOpen) or `external_call`.
3. **The two compiler backends are now complementary, not one-sided**: iree/rocm is the better
   per-call choice (it beats xla in 43 of 54 comparable cells), covers control flow, conv, gather/
   scatter and external kernels at fp32 accuracy; xla holds a 1.3–4.3× edge on 11 compute-bound
   cells (matmul_1024, the `cond_*`/in-graph inference loops, `layernorm_large`, `transformer`) but
   only with `dot_precision="HIGHEST"` for accuracy.
4. **The numpy backend is the portability floor**: all 98 examples, 0 deferrals, 70 bit-exact, no
   exclusions.
5. **Three upstream items remain, none of them harness bugs**: the iree ROCm HAL's `while_fib`
   segfault (deterministic; also on iree/cuda, fine on llvm-cpu) and the `while_poly` / rocm-only
   optimisation-loop compile/runtime failures; the general `iree-compile ... targets: rocm`
   failures for `softmax_reg_step` and `e2e_infer_transformer`; and XLA/ROCm's TF32 fp32 `dot`
   default (now controllable via `dot_precision`, in the library). On the harness side, the
   `target_backends` injection is iree-family-only, so `--backend xla` / `--backend tvm` run
   directly through the public CLI with no wrapper (see "Harness note").
