# RNG algorithm benchmark — results & default-algorithm decision

Benchmark harness: `etl/bench/rng_bench.py` (run via `python3 -m etl.bench.rng_bench`).

## Environment

- Python 3.11.2, numpy 2.4.6, iree-base-compiler/runtime 3.11.0, no torch.
- CPU: host (numpy interpreter backend) + iree `llvm-cpu`. GPU: iree `cuda` on a free RTX A6000 (`Device("cuda", 5)`; nvidia-smi free-GPU scan).
- Methodology per config (op × size × algorithm × target × path): compile once (`compile_ms`), warmup 2, best-of-7 timed `run()` calls (device-resident timing on cuda — `.numpy()` excluded), bit-exactness verified against the numpy-backend run of the same graph + key, same-key two-run determinism checked.
- Seed 0; sizes 2^20 / 2^22 / 2^24 elements (uniform/normal f32, randint i32); split = `split_n(key, 512)` with an accumulation formulation (single output; the cuda HAL 16-binding export limit excludes split there).
- Exclusions: `random_permutation` (iree-cuda `stablehlo.sort` bufferization failure), `random_multinomial` (v1 compiler-backend deferral), native PHILOX on iree (expected legalization failure — recorded, never a crash).
- `normal` f32 on compiler targets uses the documented Box–Muller f32 fast path: bitwise differs from the f64 numpy kernel within budget (rtol 1e-4, atol 1e-4·max(1,size/2^20)); uniform/randint/split are bitwise EXACT vs numpy on every successful config.

## Geometric-mean best-of-N run time per algorithm (best LEGAL path per cell: inline for splitmix64; min(inline, native) for the ciphers)

| target | cells | splitmix64 | threefry2x32 | philox4x32_10 | fastest |
|---|---|---|---|---|---|
| numpy (CPU) | 10/10 | **116.27 ms** | 426.41 ms | 186.74 ms | splitmix64 |
| iree llvm-cpu | 10/10 | **20.40 ms** | 41.75 ms | 43.27 ms | splitmix64 |
| iree cuda | 9/9 | 1.67 ms | **1.63 ms** | 2.25 ms | threefry (tie) |

Per-cell fastest (algorithm, path):

- **numpy**: splitmix64 wins 8/10 cells (philox edges it on uniform 2^20 24.17 vs 25.24 ms and normal 2^22 397.7 vs 399.3 ms — noise-level).
- **llvm-cpu**: splitmix64 inline wins ALL 10 cells (e.g. uniform 2^24: 53.3 vs 100.5 ms native-threefry; randint 2^24: 67.8 vs 128.6; normal 2^24: 624 vs 802).
- **cuda**: threefry (native) wins all three 2^20 cells (uniform 0.52 vs 1.36 ms; normal 0.55 vs 1.42; randint 0.33 vs 1.45); splitmix64 inline wins ALL 2^22 and 2^24 cells (uniform 2^24: 2.16 vs 4.27 native-threefry; randint 2^24: 2.50 vs 3.84; normal 2^24: 2.22 vs 8.23). The crossover is at ~2-4M elements: threefry has lower fixed overhead, splitmix64 a lower marginal cost per element.

## Native `stablehlo.rng_bit_generator` vs inline expansion (threefry2x32)

| target | native vs inline | example (16M elements) |
|---|---|---|
| iree cuda | **1.8–2.1× faster** | randint 3.84 vs 8.16 ms; uniform 4.27 vs 7.64; normal 8.23 vs 15.30 |
| iree llvm-cpu | faster at bulk, mixed at small | uniform 2^24: 100.5 vs 163.8 ms; randint 2^22: 29.9 vs 47.8; randint 2^20: 16.3 vs 15.7 (native slower); split: 2.56 vs 0.73 (native much slower) |
| compile time | native ~700–950 ms vs inline ~960–2600 ms | — |

Native THREE_FRY is bit-exact vs numpy (and == inline == numpy) on both llvm-cpu and cuda. Native PHILOX fails iree legalization on BOTH targets — philox runs inline on iree (bit-exact). splitmix64 has no native form anywhere (always inline — and still the fastest).

## Decision: splitmix64 remains the default algorithm

`etl.random.key(seed)` keeps `algorithm="splitmix64"`:

1. **CPU (both targets): decisively fastest** — 1.6–3.7× geomean over the ciphers, every llvm-cpu cell.
2. **GPU: statistically tied with threefry** (1.67 vs 1.63 ms geomean), and splitmix64 wins every ≥4M-element cell — the bulk-RNG regime of typical training/sampling batches. threefry only wins the 1M-element cells (lower fixed overhead), where absolute times are sub-ms anyway.
3. **Lowest compile times** (381–643 ms llvm-cpu vs 580–2600 ms for the cipher expansions/native paths) — matters for JIT-heavy workloads.
4. **No native-primitive dependency** — works on every backend and target with zero legalization risk.
5. **Backward compatible** — bit-identical to the pre-framework default stream; no test/user churn.

**When to choose another algorithm explicitly**: `threefry2x32` for small-sample GPU workloads and native-primitive backends (iree now emits native THREE_FRY for it by default — ~2× faster on cuda); `philox4x32_10` for XLA-native deployments (native PHILOX is v1 export-only on xla until validated against a real PJRT plugin; on iree it runs the bit-exact inline path).

## Follow-on change driven by this benchmark

`Capabilities.rng_bit_generator` is now a per-algorithm set (was a single bool): iree declares `{"threefry2x32"}` — native THREE_FRY is the iree default for threefry (bit-exact, 1.6–2.1× faster on cuda); xla declares both (by design, re-validate with a real plugin); tvm none. The reserved per-call `rng_bit_generator` option (bool or collection of names) overrides the capability per `lower()`/`build()`/`evaluate()` call (inline pinning, e.g. `rng_bit_generator=frozenset()`).

## Reproduce

```
python3 -m etl.bench.rng_bench                          # full matrix (CPU + GPU auto-free-scan)
python3 -m etl.bench.rng_bench --targets numpy,llvm-cpu # CPU only
python3 -m etl.bench.rng_bench --targets cuda           # GPU only (auto free-device scan)
python3 -m etl.bench.rng_bench --quick                  # smoke: 2^20/2^22, halved repeats
```
