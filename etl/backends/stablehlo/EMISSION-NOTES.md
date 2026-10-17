# etl/backends/stablehlo — emission measurement log & probe evidence

## Intent

Evidence file behind `./CONTEXT.md`.

`CONTEXT.md` is the CONTRACT (what the exporter emits, which options exist, which deferrals are explicit); this file holds the measurements, probe matrices, golden char/line counts and upstream-bug evidence that justify those rules.

Nothing here is a behavioral contract — read `./CONTEXT.md` first, and `../../CONTEXT.md` ("StableHLO exporter — v1 scope") / `../CONTEXT.md` (backend options override) for the binding tables.

Rules of thumb when you change the encoder and re-measure: replace the affected numbers here (never annotate them with "was X, now Y" — git is the change log), and keep the contract sentence in `CONTEXT.md` accurate rather than moving numbers back into it.

Host for all `iree`, `xla` and numpy-backend numbers below unless stated otherwise: 128-core shared host (contention-affected); GPU numbers from an AMD Instinct MI300X (gfx942, ROCm 7.1.0) and an RTX A6000 (CUDA, driver 535.247.01).

## 1. `eigh` — iree-llvm-cpu, sequential vs batched (`eigh_algorithm`)

Exec medians of 5 interleaved repeats in one process, f32, ms: n=10 10.878 → 2.687, n=50 714.978 → 24.450, n=100 9531.579 → 128.030 (4.0× / 29.2× / 74.4× in favour of batched).

Sequential body (88 ops per rotation) fuses to ~5 HAL dispatches + 10 device copies + 15 barriers per rotation (~25 serialized command submissions), so its cost is overhead-dominated: n=10 ≈ 31–116 µs/rotation, n=50 ≈ 162–270 µs/rotation — exactly linear in the rotation count at ~116 µs/rotation, i.e. no fixed per-call overhead.

The fp32 off-diagonal-energy floor is ~1e-6 relative (reduce rounding over n² elements), so `eigh_early_exit` tolerances below ~1e-6 can never fire in fp32.

Build (trace + lower + compile + load), seconds: 0.655 / 0.660 / 0.806 sequential vs 0.735 / 0.763 / 0.921 batched at n = 10 / 50 / 100 (+12–16 %); compile time is dim-independent.

MLIR text: 13558 / 26041 / 64354 chars sequential vs 14735 / 27304 / 65599 batched at n = 10 / 50 / 100, at 132 vs 148 lines (even n) / 139 lines (odd n).

Only the hoisted O(n²) `eye` constant grows: 575 chars at n=10, 12,655 at n=50, a single 50,257-char line at n=100 (plus, with the early exit, the i≠j mask).

Batched golden markers (default options): 14735 / 16560 / 27304 / 65599 chars at n = 10 / 25 / 50 / 100; 5× `stablehlo.remainder`, 6× `stablehlo.transpose`, NO `stablehlo.if` / `stablehlo.reduce` / `stablehlo.slice`; the while carries `(tensor<nxnxf32>, tensor<nxnxf32>, tensor<i64>)`; constants include the round count `dense<L>` (9 / 25 / 49 / 99) and the loop total `dense<45>` / `dense<150>` / `dense<343>` / `dense<792>`.

`eigh_early_exit=True` markers at n=10: 176 lines / 17127 chars, +1 `stablehlo.if`, +2 `stablehlo.reduce`, +1 `stablehlo.remainder`; text growth +31 lines / +2.5 KB (n=10) / +14.6 KB (n=50, the hoisted (n,n) f32 mask dominates), ~+23 ops, compile +10–22 %.

Round-reduction table (why batched is ~n/2 fewer iterations) — the sweep kernel `sweeps·L` vs `sweeps·n(n-1)/2`:

| n | sweeps (f32) | sequential rotations/sweep → total | `L` | `m` | batched rounds/sweep → total | reduction |
|---|---|---|---|---|---|---|
| 2 | 3 | 1 → 3 | 1 | 1 | 1 → 3 | 1× |
| 3 | 3 | 3 → 9 | 3 | 2 | 3 → 9 | 1× |
| 10 | 5 | 45 → 225 | 9 | 5 | 9 → 45 | 5× |
| 25 | 6 | 300 → 1800 | 25 | 13 | 25 → 150 | 12× |
| 50 | 7 | 1225 → 8575 | 49 | 25 | 49 → 343 | 25× |
| 100 | 8 | 4950 → 39600 | 99 | 50 | 99 → 792 | 50× |

f64 keeps the sequential 8-sweep count in both compositions; a BYE slot costs one iteration without a real rotation.

Accuracy, f32, sequential / batched (relative reconstruction `‖A0 − V diag(w) Vᵀ‖∞ / ‖A0‖∞`, orthogonality `‖VᵀV − I‖∞`, `werr` vs `numpy.linalg.eigvalsh`):

| n | rec | orth | werr |
|---|---|---|---|
| 3 | 6.09e-07 / 2.75e-07 | 3.17e-07 / 1.40e-07 | 3.26e-07 / 7.89e-08 |
| 10 | 1.31e-06 / 1.74e-06 | 8.83e-07 / 9.23e-07 | 5.96e-07 / 6.42e-07 |
| 25 | 4.85e-06 / 8.12e-06 | 3.06e-06 / 4.01e-06 | 2.63e-06 / 3.69e-06 |
| 50 | 1.34e-05 / 1.41e-05 | 8.72e-06 / 7.81e-06 | 6.14e-06 / 7.74e-06 |
| 100 | 2.87e-05 / 2.93e-05 | 1.797e-05 / 1.831e-05 | 1.33e-05 / 1.61e-05 |

f64 rec: n=3 8.06e-16 / 3.45e-16, n=10 2.72e-15 / 3.83e-15, n=25 9.73e-15 / 8.83e-15, n=50 1.46e-14 / 1.80e-14 (all orth ≤ 1.3e-14, werr ≤ 1.3e-14).

A batched (2,3,3) f32 input: batched rec 2.75e-07 / orth 2.45e-07 (sequential 6.09e-07 / 3.17e-07); int32 3x3 → f64: both at the f64 floor (rec ≤ 3.9e-16, batched werr exactly 0).

Column-wise `abs(v)` (sign-free) matches `numpy.linalg.eigh` at the fp32 floor for n=3 and a diagonally dominant n=10 in both modes (~6e-08–2.4e-06), so the eigenvector pairing agrees.

Summary: batched is at least as accurate as sequential within a ≤2× noise band at every size (worst ratios 2.000× orth at int32 n=3 — 1–2 ulp of f64 — and 1.674× rec at f32 n=25, both ≪ 1e-3), strictly better at the tiny sizes, at f64 n=25 and on the diagonally dominant n=10; residuals are noise-like with no systematic trend at larger n.

## 2. `eigh` — iree/rocm (AMD MI300X, `Device("rocm", 0)`)

The batched composition is the only route that completes: the sequential one SEGFAULTs at invoke for n=10 (exit 139, libamdhip64 frames under `iree/runtime/function.py::_invoke`) and hangs >180 s at n=50 (the documented HIP-HAL while-loop unreliability), so there is no sequential iree/rocm baseline.

Batched f32 exec medians of 7 interleaved runs: n=10 4.926 ms, n=25 15.419 ms, n=50 36.893 ms, n=100 83.356 ms (rec 2.26e-06 / 4.51e-06 / 1.25e-05 / 3.09e-05, orth 1.29e-06 / 2.80e-06 / 6.54e-06 / 1.62e-05); build 0.73–1.04 s per shape.

## 3. `eigh` — xla/rocm (A/B in ONE process, f32, randomized symmetrized `standard_normal`, jax_rocm7_pjrt 0.11.2, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, medians of 7 interleaved `etl.run`)

Both compositions run here, and batched wins at every size: n=10 1.156 vs 6.648 ms (5.7×), n=25 3.593 vs 49.793 ms (13.9×), n=50 7.834 vs 246.695 ms (31.5×), n=100 18.178 vs 1181.178 ms (65×); best-of-7 within ~2 % of the median.

Builds (trace + lower + compile + load) 0.11–0.65 s per shape (xla compile 0.11–0.18 s) plus a one-off per-process first-execute cost of up to ~1.8 s that is NOT part of the steady-state medians.

Accuracy, batched / sequential (ascending w in all eight arms): rec 4.05e-07 / 9.76e-07 (n=10), 1.63e-06 / 1.18e-06 (n=25), 2.89e-06 / 2.64e-06 (n=50), 4.98e-06 / 4.15e-06 (n=100); orth 5.24e-07 / 7.39e-07, 1.38e-06 / 9.55e-07, 1.82e-06 / 1.59e-06, 2.74e-06 / 2.73e-06; max-abs w error vs LAPACK 1.19e-06 / 2.14e-06, 1.05e-05 / 6.12e-06, 2.54e-05 / 2.51e-05, 6.04e-05 / 5.18e-05 — a ≤1.4× noise band, batched never worse than 1.37× on rec and better on orth at n=50/100.

CMA-ES reading: one n=10 eigh (the evox CMA-ES covariance factorisation) costs 1.156 ms/step on xla/rocm batched (6.648 ms sequential) and 4.926 ms/step on iree/rocm batched — ~15 % of the ~7.5 ms xla-rocm 100×10 benchmark step and ~65 % of the iree/rocm step.

## 4. Rejected: round-unrolling the batched while body (`eigh_unroll`)

Measured and rejected — do not repeat.

Variant: a parameterized U-rounds-per-iteration body (round index `k*U + u` reduced mod `L`, loop bound `ceil(total/U)`, the tail padded by REPEATING schedule rounds — a repeated (p, q) rotation is an ordinary similarity transform, so padding is valid and results stay bit-identical).

It failed the ≥3× keep gate: at U=8 (medians of 5 interleaved A/B arms in one process, f32) it gains only 1.77× / 1.32× at n=50 / n=100 on iree/llvm-cpu and 2.02× / 1.74× on iree/rocm (best 2.38×, llvm-cpu n=25), because the composition already sits at the per-round COMPUTE floor (~0.057 ms/round on rocm, saturating at U=4) rather than at removable per-iteration overhead.

Exec medians (ms, U = 1/4/8) — llvm-cpu: n=10 1.399/0.825/0.683, n=25 8.554/4.584/3.588, n=50 22.609/14.764/12.767, n=100 106.349/86.908/80.830; rocm: n=10 4.514/3.012/2.775, n=25 16.820/9.220/8.690, n=50 39.432/20.110/19.550, n=100 81.807/49.645/47.024.

Accuracy is U-invariant (bitwise identical for even n and all f64; f32 n=100 rec 3.220e-05 / orth 1.714e-05 stays inside the ≲3e-5 bound) and the only cost is size: build +16 %/+47 % and MLIR text ×1.43/×2.00 at n=100.

## 5. `eigh_early_exit` — calibration and measurements

Tolerances calibrated on llvm-cpu: f32 3e-5, f64 1e-13; the parity suite stays green at every looser candidate, 3e-5 keeps dim25/batched-3x3 on the FULL schedule, 1e-4 rejected. Calibration home: `tests/backends/test_iree_eigh_diag_parity.py`.

llvm-cpu 3.11 (noisy shared host, within-session interleaved A/B; sequential composition unless noted):

- dim50 sample-covariance-like f32 exits at the sweep-4 boundary of 7 (executed ~4900/8575 rotations; median ~14 % time savings, best-of-N ratios 0.63–0.99).
- dim10 exits sweep 4 of 5 but NET-LOSES ~1.5–1.9× eigh time (machinery overhead beats the one saved sweep — ~2–4 % CMAES-dim10 step regression).
- near-diagonal / ill-conditioned dim50 exits after sweep 1 (~75–86 % savings, correct — w bitwise-identical to the full schedule on the ill-conditioned family, rec within the 3e-5 relative-Frobenius guarantee).
- well-spread random-symmetric dim50 exits only at sweep ~5.5–6 (marginal).
- Net loss in BOTH compositions (exec medians, option off → on: n=10 sequential 11.10 → 19.25 ms; n=45 sequential 554.69 → 719.89 ms; n=10 batched 2.94 → 3.95 ms; n=45 batched 21.74 → 29.10 ms) with accuracy unchanged.

iree-cuda 3.11.0 (RTX A6000 probe), SEQUENTIAL composition: option-on crashes deterministically under BOTH allocators (async default AND `--cuda_async_allocations=false`) — bare eigh n=10 fails with `hal.device.queue.dealloca` `INVALID_ARGUMENT` (`memory_pools.c:297`, `CUDA_ERROR_INVALID_VALUE`), n=50 SIGSEGVs the process; option-off runs clean at n=10/n=50 (3/3 each). llvm-cpu is green in both modes.

Design rule derived from probing: a loop-carried i1 flowing through not/and in a while COND region SEGVs iree 3.11's AffinityAnalysis (0/10 at 3x3 f32 with adapter flags), while the body-side `k := total` clamp is 10/10 clean — hence the cond region stays the plain `k < total` compare in both modes.

## 6. `_eigh_take`'s transpose dance — required by an upstream iree vectorization bug

Emitting the batched composition's partner take as a gather with a NON-leading `collapsed_slice_dims` / `start_index_map` (e.g. `[1]`) makes iree-compile fail at EVERY size with `<'vector.transfer_read' op inferred mask type ('vector<4xi1>') and mask operand type ('vector<1x4xi1>') don't match` on the gather op, while the same op compiles standalone with a function-argument index — a body-context vectorization bug in iree, not a StableHLO shape error.

`_eigh_take` therefore transposes the target axis to the front, gathers with `collapsed_slice_dims=[0]` / `slice_sizes=[1, n] + batch`, and transposes back — the same static-shape permutation the frontend `gather` emitter uses for `axis != 0`.

## 7. Split (two-stage) reduction — root cause and measured validation

Root cause: IREE's GPU backend lowers any `stablehlo.reduce` whose RESULT is rank-0 to ONE workgroup of ONE thread.

Measured on a real evox DE step graph (AMD MI300X): 3 rank-0 reduces = 1765 µs of 1858 µs total, all data-parallel work 61 µs; 14 compiler-flag variants and `opt_level="O3"` moved it ≤0.3 %, so the emitter is the only lever.

Evox-shaped fitness monitor (pop 10000 × dim 100: `sum(pop,1)` → rank-0 `min`/`argmin`/`max`/`argmax` + an i1 `max(isnan(pop))` guard), per-call medians of 9: iree/rocm 13.53 → 0.746 ms (18.1×), iree/llvm-cpu 2.69 → 2.74 ms (0.98×), xla/rocm 0.620 → 0.635 ms (0.98× — XLA has no rank-0 pathology).

Per-op iree/rocm: rank-0 `min`/`max`/`argmin`/`argmax` over `(10001,)` 4.1–4.9×; over `(10000,100)` axis 0 1.5× (`min`/`max`) and 4.1–4.3× (arg reduces); axis 1 (10000 non-reduced elements → above the small-result gate) does not split and stays ~1.0×.

llvm-cpu 0.80–0.93× on `(10001,)` (the two extra dispatches exceed a cheap CPU reduce) but 1.5–1.6× on `(10000,100)` axis 0; xla/rocm 0.97–1.07× everywhere (noise).

Parity matrix, 770 cases (shapes `(10001,)`, `(10000,100)` ax0/ax1, `(7,)`, `(1024,)`, `(1025,)`, `(10007,)`; f32/f64/int32/bool; min/max/argmin/argmax/sum/prod/mean; plain/NaN/±inf/all-equal/all-zero/tie-heavy data): 770/770 single-stage ≡ split bit-identical on xla/rocm, 760/770 on iree/rocm, 0 build/run errors on either, and no row where the split is wrong and the single stage right. (The 760/770 figure predates the NaN-aware arg comparator; arg indices now match numpy exactly on both backends.)

`split_reduction_non_exact=True` float `sum`/`mean` stays within 4.5e-5 (iree/rocm, worst) / 2.7e-5 (xla/rocm) relative — inside a 1e-4 budget in all 12 spot cases, and sometimes MORE accurate than the single-stage emission (3.7e-7 vs 1.6e-6 on `(10001,)` f32).

R/C geometry is insensitive: forcing `64×157` / `128×79` / `100×101` / `50×201` / `256×40` against the implemented `73×137` for `(10001,)` f32 `min` on iree/rocm spans 0.167–0.188 ms (best alternative = 1.01× the implemented pick, within noise), all bit-exact.

`split_reduction=False` byte-identity was verified on 53 graphs against the single-stage text.

## 8. Promote-before-reduce — probe detail

60/60 probe rows agree that numpy's accumulator dtype is fixed by the OPERAND dtype and the declared IR result dtype IS that dtype.

Float paths: 31/31 float graphs byte-identical in an 86-graph matrix.

iree-llvm-cpu parity after the promotion: 1020 checks with only the 8 pre-existing float16 accumulation-order failures (f16 MLIR byte-identical).

Unsigned end-to-end validated on the **tvm** adapter (iree's capability gate rejects every uint dtype): 14/14 OK with promotion vs 12/14 failing without.

Consequence recorded in the contract: int/bool `mean` at reduced extents ≥ 1024 does not take the split path (accumulator f64, and float `sum` is not split by default) — the `bool_split_mean_2000` export marker count is 0 (an i1 split would be numerically wrong anyway).

## 9. Upstream iree 3.11 ROCm miscompile: signed narrow-int `minimum` at extent ≡ 0 (mod 64)

A single-stage `stablehlo.reduce` with `minimum` over `i8`/`i16`/`i32` whose reduce EXTENT is a multiple of 64 returns `-1` instead of the minimum.

Measured extents WRONG: 64/128/192/256/320/384/448 (every power of two ≥ 64); CORRECT: 60/61/62/63/65/66/96/100/112/120/127/129/160/200/224/288/352.

Unaffected: `i64` min, `maximum` (all dtypes), the UNSIGNED narrow ints, `sum`/`prod`, f16/f32/f64/i1; llvm-cpu is clean at every extent.

`i8`/`ui8` reduce graphs fail to COMPILE on ROCm at all (independent).

`split_reduce.py`'s guard rule 7 (`UNSAFE_MIN_EXTENT_MULTIPLE`) keeps BOTH stage extents off that class (`128 x 79` → `127 x 79`), so a split never introduces the miscompile — and the split incidentally fixes the single-stage failure at `(1024,)` int32 `min` (`32×32` is safe). Do not remove the guard; do not generalize it to float/`i64`/unsigned/`maximum`/`sum`/`prod`.

## 10. numpy reference edge cases the emission cannot express

`np.min`/`np.max` of an EMPTY reduction raise `ValueError` ("zero-size array to reduction operation minimum which has no identity") while the emitted `stablehlo.reduce` returns the init (`∓inf` / `iinfo.min|max`); `np.sum`/`np.prod` of empty give the identity (`0.0`/`1.0`) and match.

`np.sum(int32)→int64`, `np.sum(bool)→int64`, `np.prod(int32)→int64`, `np.mean(int)→float64`, `np.sum(float16)→float16` — the first three follow the promote-before-reduce rule, and `reduce_mean` always converts + divides.

## 11. arg-reduce NaN awareness — validation matrix

1920/1920 EXACT vs the numpy backend on iree-llvm-cpu: f32/f64 × plain/nan_first/nan_mid/nan_last/nan_multi/nan_all/nan_inf/nan_ties/dup_extrema; i32/i64/bool × plain/dup_extrema; shapes `(10001,)`, `(10007,)` padded, `(7,)`, `(1000,)`, `(10000,100)`, `(100,10000)` × axes none/0/1 × keepdims × split on/off.

The first attempt encoded the equal-class term as `NOT(a NE c)`; two NaNs are NE, which produced a 304/1920 failure class. Never express it that way.

NaN-aware form cost: +7 combiner ops (8 region instructions / 7 value ops → 15 / 14); f32 `(10001,)` argmin export 20 → 27 single-stage lines and 38 → 52 split. int/bool keys emit byte-identical text.

Split geometry is unaffected by NaN awareness: the `// etl split reduction: pivot axis 0, 10001 -> 73 x 137` marker is unchanged, the pad VALUE stays ±inf (never NaN) and the pad INDEX stays the `>= N` sentinel, so a pad can never win.

Integer tie-breaks are order-independent first-occurrence (`argmax([5,1,5,1])=0`, `argmin([5,1,5,1])=1`).

## 12. fp32 `sum` re-association evidence

A 2-stage sum over extent 2048 is bit-identical to the single stage at unit scale but differs at scale 1e3/1e6 — and is closer to the fp64 truth (relerr 1.17e-6 vs 2.72e-6, 1.04e-7 vs 6.29e-7).

Relevant only if a bit-exact `sum` pin exists at extent ≥ 1024 (none does).

## 13. Reduce emission-site inventory (writer line numbers as of the current HEAD; they drift — re-grep `"stablehlo.reduce"` before relying on them)

Six real `"stablehlo.reduce"` emission sites:

| Site | Current line | Role |
|---|---|---|
| `_emit_reduce_core` | 1403 | shared by `_emit_reduce` (1254: IR `reduce_sum/max/min/prod`) and `_emit_reduce_mean` (1320) |
| `_emit_arg_reduce` | 3431 | the argmin/argmax composite reduce |
| `_emit_count_reduce` | 2453 | count-argsort rank/position reduce |
| `_emit_count_reduce_min` | 2473 | count-argsort inverse-permutation / sorted-values reduce |
| `_emit_eigh_reduce` | 3074 | eigh convergence energy |
| `_emit_while_zero_init` | 4396 | the `while_init_rewrite` computed-zeros chain |

`_emit_reduce_region` (1451) is NOT one of them — it builds only the region of the `all_reduce`/`reduce_scatter` collectives (`_emit_collective`, 4516).

The non-frontend sites can cross the same "reduced extent ≥ `split_reduction_threshold`" gate: count-argsort reduces over the k-axis (extent = sorted-axis size) and the eigh energy reduce over n² elements (2025 at n=45).

## 14. Region costs (measured)

Count-in-region: +37 lines / +2.7–2.9 KB text for a region-nested sort.

while-body `eigh`: run 2.706 → 2.762 ms at n=10 (+2.1 %) and 9.808 → 10.170 ms at n=25 (+3.7 %), build +1.4 %/+2.3 %.

Select fallback (`_emit_if_select`): the `cond` costs the SUM of both branches — cond/2×bare-eigh = 0.978–1.049 (n=10) / 0.997–1.004 (n=25) median, i.e. exactly 2× a bare `eigh` (cond/bare = 1.96–2.10) with no extra per-`select` cost; build +7–13 % (bare 703.8–765.0 ms vs cond 770.7–814.7 ms).

`_select_inline_depth` visibility check: byte-identical across 12 exported graphs; only the two while-bearing-cond exports change (2 sorts → 0 sorts, +74 lines at n=10).

Root cause of the while-in-if prohibition: iree 3.11's `local-task` driver mis-executes a `stablehlo.while` nested in a `stablehlo.if` — the SAME compiled module is 6/6 correct under `local-sync` and returns zeros / non-orthonormal vectors under `local-task`. The implicit-capture-mis-binding hypothesis is refuted (capture-free variants are worse, not better).

## 15. One-operand `stablehlo.sort` (values sort) codegen pathology on iree-cuda

Probe: iree 3.9.0, RTX A6000, driver 535.247.01.

iree-cuda lowers a 1-operand `stablehlo.sort` to a SINGLE-THREADED insertion/bubble-sort kernel (PTX `.maxntid 1,1,1` scalar global-memory swap loop, verified via `--iree-hal-dump-executable-binaries-to`) with ~O(n²) wall time: 0.45 / 10.9 / 33.8 / 136 / 844 / 3394 ms at n = 101 / 501 / 1001 / 2001 / 5001 / 10001 (powers of two equally bad; `opt_level` O0–O3 no effect — a lowering decision).

This owns the evox SO (PSO/DE/OpenES/CMA-ES) step on iree-cuda: the EvalMonitor topk (k=1) sorts the (pop+1,) fitness vector — 35–43 ms/step at 1000x50 and ~3.48 s/step at 10000x100 (= sort(10001) 3.39 s); MO graphs are unaffected (no topk → NSGA2 stays 4–6 ms).

llvm-cpu's 1-operand sort is also superlinear (2.4 / 35 / 150 ms at 1001 / 5001 / 10001) and the count-argsort adds 241 ms @10001 (the documented 400–700 ms/step llvm-cpu cost).

The COUNT machinery (used for argsort ≥ 32) is fast on cuda: 0.29 ms @1001 / 56 ms @10001 — routing VALUES sorts through it measures 0.25 ms @1001 (134×) / 28.2 ms @10001 (120×) on cuda, bit-exact vs numpy.

llvm-cpu count values-composition ≈ the count-argsort cost (~1.6× the single-threaded sort at 10001), which is why the extent-gated routing is left to the strict `sort_emission="count"` mode and the iree cuda target.

Complementary ops-level shortcut (`etl.topk` k==1 → `argmin`/`argmax` + gather): bit-exact vs numpy, 0.13 ms @1001 / 0.75 ms @10001 / 4.4 ms @100001 — O(n), ideal for the evox monitor k=1 (argsort-stable first-min index == `np.argmin`).

Caveats: the count composition is O(n²) parallel — beyond n≈100k prefer argmin-style short-cuts or an upstream iree sort-codegen fix; bit-exact only for finite keys (NaN: `stablehlo` LT is false → ranks collide — fitness vectors are finite in practice).

## 16. While-loop AffinityAnalysis crash probes (`while_init_rewrite` evidence)

Symptom: iree-compile crashes with `CompilerToolError "Error code: -11"` (SIGSEGV) on `non_dominate_rank` while loops (the evox NSGA2 tell graph).

gdb backtrace: infinite recursion in `mlir::iree_compiler::Explorer::walkTransitiveUses` ⇄ `IREE::Stream::ValueConsumerAffinityPVS::updateValue` ⇄ `DFX::Solver::updateElement`/`updateFromUse`, inside `IREE::Stream::AffinityAnalysis::run()` (right after the `VerifyLoweringToAsyncPass` marker).

Baseline rates: `non_dominate_rank` 8/10 crash on llvm-cpu (n=8 AND n=32 — size-independent), 13/13 crash on cuda for EVERY while graph tested; the full NSGA2 tell graph 10/10 crash llvm-cpu.

Capture hypothesis REFUTED: capture-free variants crash MORE (all-captures-removed 10/10, partial-capture 6–7/10 vs baseline 8/10).

Two independent triggers, isolated with 15 micro-probes (each 10×, n=32, llvm-cpu):

- **T1 = a dense rank≥1 constant as a while-loop INIT operand** (const-init 0–1/10 OK, w8_T1 = safe body + const init 0/10, w13 1/10, w16 2/10, v1 0/10).
- **T2 = a `broadcast_in_dim` whose operand is a CONVERT of a loop-carried value, consumed directly by a non-convert op** (w6/w7/w9 0/10 with all-computed inits).

SAFE shapes: `broadcast(carried-directly)→select` (P10 10/10); `broadcast(carried)→convert→select` (w8 10/10) — the convert must sit between the broadcast and its consumer, not between the carried value and the broadcast; a same-dtype convert is canonicalized away and still crashes; broadcasts of CAPTURED CONSTANTS are safe (w12 10/10); scalar-only whiles and vector pass-through/subtract-only bodies are safe (w1–w5 10/10).

Init-producer kind matters: reduce/convert-derived inits are CLEAN (P10 `%26` 10/10, P15 10/10); dense-const (v0 0/10), broadcast-of-scalar-const (P16 → folds to dense const → crash) and subtract-derived inits (P11 `x−x` folds→const, P12b non-foldable `subtract(reduce, transpose-reduce)` 1/10 — CSE/fold defeats all subtract forms) all crash.

`--iree-stream-affinity-solver-max-iterations=1` converts the SEGV into a clean deterministic `failed to solve for affinity analysis` (0 crashes in 8) — it proves solver non-convergence and is a fail-loud partial workaround, NOT a working compile. `--mlir-disable-threading`, `--iree-llvmcpu-target-cpu=generic`, `--iree-llvmcpu-link-embedded=false` do NOT help.

`while_init_rewrite` validation: v0(n=32), v0(n=128) and the full NSGA2 tell graph export the reduce-first init chain and compile 10/10 on llvm-cpu with ZERO SEGVs (also 3/3 clean on cuda, bit-exact); v0 runs bit-exact vs the numpy reference; v5 rank/dis/offspring bit-exact + np-ranks-of-selected True (pop/fit same-row-SET modulo the tie-order caveat); small-graph smoke (exact-shape/iota/count sources) 10/10.

Source-preference evidence: an exact-shape rank-1-derived source SEGVs 0/10 deterministically on the REAL loop, and the `iota` fallback 0/10 at scale — only the reduce path (an input-derived bool/int value one axis larger, reduced along the extra axis) is validated safe at real-graph scale.

Companion deterministic bug: any constant tensor referenced from a while body/cond fails compile 5/5 on llvm-cpu AND cuda with `error: 'util.global.load' op undefined global: @__hoisted_tensor_i32` — iree's constant-hoisting/global-fusion is broken for constants materialized inside the body after fusion, even when the constant is loop-carried.

Repro artifacts (read-only reference): `/tmp/nsga2_repro.py` (verbatim `non_dominate_rank` + builders v0–v16), `/tmp/nsga2_unrolled_fix.py` (unrolled workaround), `/tmp/validate_final.py` (validation driver), `/tmp/crash_v0_n32.txt`, `/tmp/crash_v0_printir.txt`; MLIR scratch `v0_n32*.mlir`, `mlir_w*.mlir`, `build_fix_variants.py`.

Async-allocator A/B session (iree 3.11; the adapter forces no process-global runtime flag — iree's own `--cuda_async_allocations=true` default stands; scratch `/tmp/ws_c_async_probe/`): the runtime iree-cuda while crash modes are STREAM-ORDERED-ALLOCATOR-SPECIFIC.

- The NSGA2-shaped front-peel loop (`non_dominate_rank`: input-derived (n,) i64 count, (n,) i32 rank whose rank-1 zero-const init the rewrite replaces, plus an i32 scalar; body: select on carried vectors + per-column reduce over the (n,n) input-derived i1 dominance matrix; data-dependent `max(count) >= 0` exit) compiles (~0.9 s) and runs BIT-EXACT under the DEFAULT async allocator at n=64 AND n=128 (25+ runs; also clean under classic, ~2× slower). With `while_init_rewrite=False` the cuda COMPILE SEGVs (-11, AffinityAnalysis class, allocator-independent).
- `while_fib` (scalar-only carries, rank-0 const inits, fixed-iteration exit) SEGVs at INVOKE under async only (3× repro incl. O3) and runs clean under classic (default + O3) — the "exclude while_fib from iree/cuda" guidance is really "exclude while_fib from ASYNC-allocator iree/cuda".
- `while_cond_combo` with `while_init_rewrite=False` ABORTs at invoke under async (`event_semaphore.c:354 ... hal.device.queue.dealloca`) and runs clean under classic; with the default rewrite ON it is clean under BOTH allocators.

A future exporter/runtime workaround should target the async allocator's device-side dealloca path for scalar/const-domain while loops — NOT the NSGA2 loop shape.

Unrolled fixed-point workaround (loops that can use neither exporter option): re-express `non_dominate_rank` as `steps` static front-extraction iterations (`front = count==0; rank = where(front, r, rank); count -= front_dominators + front; r += 1`) with no while op — n=128 llvm-cpu compile clean (2.4 s), run 3.3 ms bit-exact; cuda compile 3/3, run 9.9 ms bit-exact; full unrolled tell: numpy bit-exact on all fields, llvm-cpu rank/dis bit-exact + pop/fit same-selected-set.

Recommended NSGA2-on-GPU route instead: the xla adapter with a real PJRT plugin — the full tell graph compiles and runs with rank/offspring bit-exact (tell `etl.run` ~2.0–2.3 ms vs numpy 6.35 ms; exec-only 0.32 ms); env recipe in `../adapters/CONTEXT.md` Known Issue 3. Hybrid option: GPU `evaluate` + numpy-backend `tell`.

## 17. Loop-carried-count crash observation (upstream iree AffinityAnalysis) — the NSGA3 N-carry class, with candidate workarounds (none implemented)

This is the evidence section behind the "Loop-carried-count crash observation" kept as a current fact in `./CONTEXT.md`.
It is the CARRY-COUNT axis of the while-loop crash family: the carrier is `_emit_while`'s operand list, and the exporter applies no carry-count gate (see the facts in `./CONTEXT.md`).
Scope note: measured on iree 3.9.0rc20251125 — the T1/T2 triggers in §16 were measured on iree 3.11 and are a DIFFERENT (init-value / broadcast-topology) axis.

Measured on iree 3.9.0rc20251125: the NSGA3 tell graph (5 whiles: 4/4/5/5/7 carries, selects in every body) SIGSEGVs; `--iree-stream-affinity-solver-max-iterations=1` converts it to a clean `failed to solve for affinity analysis`.

The 4-carry front-peel while (w1) is the unique necessary while in the full module (removing it alone kills the SEGV; w1 standalone compiles — module context is part of the trigger), and rewriting its select to arithmetic does NOT dodge it.

Synthetic probes: 3/4-carry whiles + 1 select compile; a 7-carry + 2-select while SEGVs standalone; 3-carry whiles (NSGA2) compile fine in full modules.

The exporter is carry-count-agnostic: `_emit_while` (writer.py:4258, operand emission at 4330-ish) emits every operand positionally with no count gate.

Candidate workarounds — all OPT-IN, unvalidated, must be A/B'd:

- (a) Arithmetic rewrite of `select` inside while BODIES only — safe for i1/i/u results via the `f + (t-f)*c` form (no in-body constants); must SKIP float results (NaN/±inf in the unselected branch propagates, breaking `_emit_eigh`'s NaN-guard selects). Touch points: `_emit_select` (writer.py:992), nesting test via `op.parent.parent.parent` owner-name == "while".
- (b) Carry packing in `_emit_while`: flatten+concat same-dtype-group carries (ints→i64, floats→f64, bools→i1) into 1-D tensors, unpack via static-offset slice+reshape in cond/body prologues, repack at the body terminator; requires fully-static carry shapes (`slice`/`reshape` reject dynamic dims), applies the init rewrite first, bit-lossless (upcast/round-trip exact). Graph-level packing (carry count is 1:1 with the init's symbolic leaves; no carry-rewrite machinery exists in trace/transforms) belongs upstream in `etl/transforms` or `CompilerBackend.lower` before export — the writer accepts packed whiles unchanged.

Exporter-option plumbing recipe (for any new exporter option): (1) Writer flag + normalizer in `Writer.__init__`; (2) `export()` docstring (`./__init__.py`); (3) `_exporter_options` entry in `../compiler.py` (+ an optional `default_<option>` ClassVar); (4) each consuming adapter's `KNOWN_OPTIONS["lower"]`.

## 18. `constant` emission path (line references)

The IR `constant` op dispatches through `Writer.render_constant` (writer.py:662), which reads the full payload from `op.attributes["value"]` and inlines it via `Writer._constant_text` (writer.py:4852) — a recursive `dense<[...]>` string join with no size threshold, no chunking, no streaming, no separate-file serialization, and no path that turns a large constant into a runtime function input.

The whole module text is built in memory (`Writer.write`, writer.py:565). The only value-dependent constant handling in the writer is the `while_init_rewrite` zero-init rewrite (`_emit_while_zero_init`, writer.py:4396) — size-independent.

`etl.constant`'s large-constant warning (`ETL_LARGE_CONSTANT_BYTES`) lives in `etl/ops`, outside this node.

## 19. `_emit_dot` helper layout

`_emit_dot` is at writer.py:3641; all `_dot*` helpers live between `_emit_dot` and `_emit_conv`.

Its 5-path dispatch: rank-1 operands / matched batch / non-batched fast paths — plain-matrix rhs AND the size-1-batch squeeze — each fast path gated on FULLY static shapes (iree 3.11 cannot legalize the `dynamic_reshape` its import inserts for non-batched dynamic `dot_general`, so dynamic shapes fall through to the batched dynamic-broadcast path) / static or dynamic batch broadcast / unprovable symbolic merge ⇒ `BackendError`.

## 20. Environment gotcha — stale `etl` from the main checkout

On the AMD validation host, `/usr/bin/python3` resolves `import etl` to the MAIN checkout (`/mnt/hdd_pool/bchuang/etl`, pip editable install `__editable__.etl-0.1.0.pth`) whenever the script's directory is not the worktree (scripts run from `/tmp` — cwd is only on `sys.path` for `-c`/stdin), and that checkout may lag the worktree.

Any validation script must `sys.path.insert(0, "<worktree path>")` before `import etl`; a `/tmp` script that skips it silently validates the stale checkout (observed: it reproduced the upstream cuda sort bufferization failure with no `sort_emission` present in the writer at all — a usable accidental baseline, but misleading if unnoticed).
