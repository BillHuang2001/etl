# Can etl run a FULL HuggingFace transformer forward pass as ONE compiled graph?

**VERDICT: YES.** Four real HF checkpoints (Llama-family, Mistral-GQA, GPT-2 tiny, distilgpt2,
gpt2-124M) were each hand-built as a **single `@etl.defn` graph** (weights = explicit graph
inputs), validated on the numpy interpreter backend against an independent pure-numpy reference,
then **exported to StableHLO and compiled+run on `iree`/`llvm-cpu` — the WHOLE graph, one
compile, no splitting, no `external_call`, no fallback.**

* **Zero ops blocked export** for RMSNorm/LayerNorm + RoPE + GQA attention + SwiGLU/GELU-tanh +
  lm_head at static shapes. The ONE thing that defers is **exact erf-GELU** (`erf`, hence `gelu`,
  which are documented v1 export deferrals) — hit by `tiny-random-Mistral`'s `hidden_act: "gelu"`.
  GPT-2's `gelu_new` is the tanh approximation, so **GPT-2 needs no erf at all** (no error
  introduced: `gelu_new` *is* the tanh form; the erf-form cost measured at 5.1e-07 max-abs on
  tiny-mistral logits).
* **Compiling a 124M GPT-2 takes ~2 s and 1.5 GB RSS.** A 12-layer/124M model is entirely
  practical to compile on llvm-cpu. Running it fast on CPU is the problem (below), not compiling.
* **Numerics: ~1e-7 (tiny models) to ~1e-3 abs (fp32 logits of magnitude 120-240, i.e. ~3e-6
  relative)** across backends — see the table.

Probe code (all in this scratch dir, throwaway): `probe_hf_download.py` (urllib download +
safetensors reader), `probe_hf_torchbin.py` (torch zip **and legacy** `.bin` reader, numpy only),
`hf_probe_lib.py` (weight trees, graph builders, numpy references), `probe_hf_feasibility.py`
(driver), raw results in `./hf_results/`.

## Measured (this host: 20 cores, uv venv, Python 3.14, numpy 2.5.2, iree-base 3.11.0)

iree numbers use `iree_compile_args=["--iree-llvmcpu-target-cpu=host", "--iree-opt-data-tiling"]`
passed to `etl.compile` (see pitfall #1 — with etl's default `--iree-llvmcpu-target-cpu=generic`
the same graphs are **12-19× slower**).

| model | weights | params* | graph inputs | IR ops | StableHLO | trace | iree compile | numpy-backend run | iree/llvm-cpu run | max-abs vs ref (numpy / iree) |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-internal-testing/tiny-random-LlamaForCausalLM (s=8) | safetensors | 1.0M | 22 | 170 | 301 lines / 29 KB | 0.04 s | 1.2 s | 1.1 ms | 1.4 ms | 7.5e-08 / 8.9e-08 |
| hf-internal-testing/tiny-random-MistralForCausalLM (s=8, **GQA 4q/2kv**) | torch .bin (legacy→zip ok) | 2.1M | 22 | 194 | 343 lines / 32 KB | 0.04 s | 1.3 s | 1.4 ms | 4.9 ms | 0.0 / 1.5e-07 (tanh-GELU leg) |
| sshleifer/tiny-gpt2 (s=8, n_embd=2!) | torch .bin (**legacy** format) | 0.2M | 30 | 168 | 372 lines / 33 KB | 0.04 s | 1.0 s | 2.9 ms | 0.5 ms | 0.0 / 2.2e-44 (denormal logits) |
| distilgpt2 (s=32) | safetensors 353 MB | 120.5M* | 78 | 444 | 952 lines / 87 KB | 0.09 s | 1.6 s | 14.0 ms | 351 ms | 1.7e-04 / 4.1e-04 |
| **gpt2 124M** (s=32) | safetensors 548 MB | 163.0M* | 150 | 858 | 1822 lines / 169 KB | 0.18 s | **2.0 s** | 26.5 ms | 420 ms | 5.1e-04 / 6.9e-04 |
| gpt2 124M (s=128) | " | " | 150 | **858 (s-independent)** | 1822 lines / 172 KB | 0.19 s | 2.1 s | 219 ms | 586 ms | 2.6e-04 / 9.1e-04 |
| gpt2 124M (s=32, **default compile flags**) | " | " | 150 | 858 | 169 KB | 0.17 s | 1.3 s | 32.0 ms | **7997 ms** | 5.1e-04 / 3.5e-03 |

\* param counts double-count the tied `wte`/`lm_head` copy (gpt2 = 124M real, distilgpt2 = 82M real).
Peak RSS for gpt2/124M incl. weights + artifact ≈ 1.5 GB; distilgpt2 ≈ 1.2 GB.

**Parity**: numpy-backend vs the independent numpy reference (per-head loops, no etl) and iree vs
that same reference. For the 124M models the logits reach |max| ≈ 120-240, so 5e-4 abs ≈ **2e-6
relative** — fp32 accumulation-order noise, and iree is often no further from the fp32 reference
than the numpy backend is. `argmax` agreement is not asserted (many near-ties in a tiny-random LM);
`tiny-gpt2` logits are ~1e-44 (that checkpoint is degenerate) so its parity numbers are vacuous.

## The real cliff: iree/llvm-cpu `dot` throughput, not graph size

`etl.compile` is cheap and scales fine (858-op graph → 2 s). **Execution** on llvm-cpu is what
hurts, and it is entirely the **matmul codegen**:

* iree's adapter default `--iree-llvmcpu-target-cpu=generic` makes LLVM emit **`vector<1xf32>`
  (fully scalar) code** — measured 0.5-3.5 GF/s for `dot` (all shapes, `--iree-opt-data-tiling`,
  `--iree-llvmcpu-enable-ukernels=all`, `-O3` all useless *at that target*). gpt2-124M forward:
  **8.0 s**.
* `--iree-llvmcpu-target-cpu=host` restores AVX-512 vectorization (`vector<16xf32>`):
  18-25 GF/s (44 GF/s with `--iree-opt-data-tiling`) → gpt2-124M forward **420 ms** (19× faster).
* numpy's OpenBLAS on the same 20 cores: **1000-1800 GF/s** for these shapes, i.e. the numpy
  interpreter backend (26.5 ms) is **~16× faster than the compiled graph** (420 ms) on CPU, and
  **250× faster** than the compiled graph with default flags.

⇒ For CPU-only transformer inference, **the numpy backend is the fast path**; iree/llvm-cpu is the
cross-check / portability path. The compiled path's value is GPU (`iree` cuda/rocm, `xla`), which
this host does not have (no CUDA, no PJRT plugin installed) — **not measured here**.

## What was built (and what each part proves)

* **Llama family** (`make_llama_graph`): `rms_norm`, **RoPE computed in-graph** from
  `arange`+`power`+`cos`/`sin` with the HF `rotate_half` convention (no precomputed tables needed
  — sin/cos export fine), **GQA** via explicit `broadcast`-repeat of K/V heads (validated on the
  real 4q/2kv Mistral config), explicit batched `etl.dot` attention
  (`dot(q, transpose(k))` → scale → `+mask` → softmax → `dot(p, v)`), SwiGLU (`silu` =
  `x*sigmoid(x)`) or GELU, untied/tied `lm_head`.
* **GPT-2** (`make_gpt2_graph`): LayerNorm, `gather`-based learned positional embeddings, Conv1D
  weights used as-is (`(in, out)`, no transpose), `gelu_new` tanh approximation, tied lm_head,
  `strided slice` of the fused QKV `c_attn` output.
* **Causal mask built in-graph** (`arange` → `less_equal` → `cast` → `(1-m)*-1e4`) for both
  families — exercises compare/convert export and keeps the graph self-contained. The constant is
  numerically irrelevant (`exp(-1e4 - max)` underflows to exactly 0 in f32, same as HF's
  `finfo.min` for Llama and `-1e4` for GPT-2).
* Static shapes only (batch 1, fixed seq len) — no dynamic dims anywhere, so the exporter's
  dynamic-shape restrictions never trigger.
* Weights as **explicit graph inputs** (`spec_tree(params)` → nested dict/list of `TensorSpec`),
  never `etl.constant` (a 124M model embedded as StableHLO constants would be a ~500 MB MLIR
  literal).

## Pitfalls / gotchas for a real `etl_hf` package (in priority order)

1. **`iree_compile_args` belongs to `compile()`, not `lower()`.** Passing it to `etl.lower(...)`
   is accepted and silently has no effect — the first run of this probe "proved" llvm-cpu was
   300× slower than numpy when in fact the perf flag simply never reached iree-compile. Always
   pass it (and `opt_level`) at `etl.compile(low, backend="iree", iree_compile_args=[...])`.
2. **llvm-cpu needs `--iree-llvmcpu-target-cpu=host`** (etl's default is `generic` = scalar code,
   12-19× penalty) — plus `--iree-opt-data-tiling` for another ~1.5-2× on matmul. An `etl_hf`
   adapter should set these by default (with an explicit opt-out), or document them loudly.
3. **erf-GELU cannot be exported** (`erf`, `gelu` are v1 deferrals). Only `gelu`/`gelu_pytorch_tanh`
   (tanh) is exportable; an exact-erf model either runs on numpy only or accepts the tanh
   approximation (measured delta 5e-7 max-abs on tiny-mistral). Note `hidden_act: "gelu"` in a
   config means **exact erf** in HF.
4. **Everything else a transformer needs already exports**: `dot` (batched + broadcast, incl. GQA
   head-repeat), `reshape`/`transpose`/`slice`/`concatenate`/`broadcast`/`pad`, `gather` (embedding
   + positional lookup), reductions (`reduce_mean/max/sum` with `axes=-1` and `keepdims`),
   `divide`/`sqrt`/`exp`/`tanh`/`sigmoid`/`cos`/`sin`/`power`/`cast`, comparisons + `select`.
   There is **no `etl.softmax`** — write `exp(x-max)/sum(exp(x-max))` (2 ops, no helper exists;
   this is the single most repeated idiom and belongs in an `etl_hf.ops` module).
5. **Long-context RoPE is an in-graph `power`/`cos`/`sin` accuracy question**, not a shape question:
   the in-graph table is computed in f32 like HF, so parity holds, but for 4k+ contexts precompute
   the cos/sin tables in f64 host-side and pass them as graph inputs (the objective's suggestion)
   if bit-level agreement with a specific HF version matters.
6. **Ops count is seq-len independent** (858 ops at s=32 and s=128) — only constant payloads and
   the attention matmul sizes grow. Compile time therefore does NOT scale with context; run time
   does (219 → 586 ms from s=32 → s=128 on iree-llvm-cpu; 26 → 220 ms on numpy).
7. **Weights as graph inputs cost a full re-stage per call** (150 input leaves, 548 MB for 124M).
   For repeated runs `etl.bind` only helps as sugar (same staging); the real answer on GPU is the
   same-device loop (device-resident inputs), and on CPU it is the numpy backend. Do NOT embed
   weights with `etl.constant`.
8. **GGUF/HF key normalization**: the `gpt2` repo's `model.safetensors` drops the `transformer.`
   prefix on block keys while older exports keep it (`hf_probe_lib._normalize_gpt2_keys`).
   Llama-family linears are `(out, in)` and must be transposed host-side; GPT-2 Conv1D weights are
   `(in, out)` and must NOT be. Tied vs untied `lm_head` must be read from the config
   (`tie_word_embeddings`, or presence of `lm_head.weight`).
9. **torch `.bin` is readable with numpy alone** — but there are TWO formats: the zip format
   (torch ≥1.6: `*/data.pkl` + `*/data/<key>` blobs) **and the legacy one** (`sshleifer/tiny-gpt2`:
   magic `0x1950a86a20f9469cfc6c` + protocol 1001 + sysinfo + state-dict pickle + storage-key-list
   pickle + raw storage bytes in that key order). `probe_hf_torchbin.py` handles both (~180 lines,
   no torch); if `etl_hf` wants `pytorch_model.bin` support, copy that file.
10. iree-python prints `nanobind: leaked ...` reference-count noise at interpreter exit; harmless.

## Recommendation: an `etl_hf` integration package

Layout (mirroring etl's own layering, one concern per module — all of it OUTSIDE the etl core):

```
etl_hf/
  config.py        # HF config.json -> frozen static dataclass (hidden/n_heads/n_kv/hd/eps/theta/
                   # rope_scaling/hidden_act/tie_word_embeddings/max_positions). Pure static;
                   # every field specializes the graph. Reject anything not in the v1 matrix.
  weights.py       # safetensors + torch(zip|legacy) readers (copy probe_hf_torchbin.py), a
                   # canonical WeightTree (name -> ndarray), per-family KEY MAPS
                   # (llama vs gpt2/gpt2-conv1d), + transpose-at-load and tied-head resolution.
                   # NO etl import: keep it numpy-only so it is testable offline.
  models/
    llama.py       # make_llama_graph(cfg, seq_len) -> @etl.defn  (RMSNorm, RoPE, GQA, SwiGLU)
    gpt2.py        # make_gpt2_graph(cfg, seq_len) -> @etl.defn  (LayerNorm, learned pos, gelu_new)
  ops.py           # softmax, rms_norm, layer_norm, silu, gelu_tanh, rope_tables, apply_rope,
                   # causal_mask, grouped_heads  — the ONE place the repeated idioms live
                   # (etl deliberately has no softmax/silu/etc.)
  spec.py          # WeightTree -> TensorSpec pytree (spec_tree) + `fp` alias tables
  build.py         # the documented shorthand: spec + trace + lower + compile + load, with the
                   # backend DEFAULT MATRIX (numpy: nothing; iree/llvm-cpu: target-cpu=host +
                   # data-tiling + opt_level; iree/cuda, xla) — i.e. where pitfalls #1/#2 are
                   # encoded once instead of in every user notebook. Wrap etl.build_cached for
                   # (cfg, weights-id) memoization.
  reference.py     # the pure-numpy per-head reference forwards -> the parity oracle used by tests
  __init__.py      # forward(...) / build(...) / load_hf(dir) public surface
tests/             # golden parity: numpy backend vs reference.py (1e-6), iree-llvm-cpu vs numpy
                   # (1e-5 rel), StableHLO export golden (op count / line count pin) per family,
                   # config-matrix tests (GQA 1..n, untied head, gelu vs silu vs erf-deferral).
```

Where the value is / isn't:
* **CPU**: use `backend="numpy"` — it is ~16× faster than iree-llvm-cpu on real models and needs
  no toolchain. Keep iree-llvm-cpu as the numeric cross-check in CI.
* **GPU** (`iree` cuda/rocm, `xla`): the whole point of the compiled path. Nothing here was
  measured on a GPU — **that is the top open item** for this integration (the graphs are
  size-appropriate: 858 ops, 169 KB of StableHLO, one dot per projection).
* **Sequence length is baked in** (static shapes): one graph per (cfg, seq_len) — use
  `etl.build_cached` keyed on (cfg digest, weights id, seq_len) to avoid re-compiles in
  notebooks. A kv-cache/incremental-decode mode is a *separate* graph per step size (static shapes
  only; dynamic dims would hit the exporter's reshape/slice restrictions).
* **What is still missing v1-wise**: exact erf-GELU (deferral), any attention variant needing
  dynamic shapes or `pad`-with-non-constant modes (fine — we don't use it), and
  `sdpa`/flash-attention-style kernels (would need `external_call` + a registered kernel; the iree
  adapter's host-dispatch split would break the "one graph" property, so keep plain attention).

**Honest caveats.** (a) Parity is against MY OWN numpy reference (independent code — per-head loops,
no etl — but the same author), not against HF/torch: no torch, transformers, or safetensors lib is
installed here and none was used. The first thing `etl_hf` should add is a torch golden test
(`pytest.importorskip("torch")`, compare `logits` for a fixed input id tensor). (b) This ran on
**iree/llvm-cpu only** — no CUDA device, no `ETL_PJRT_PLUGIN`, so the xla/cuda/rocm legs are
unverified for these graphs. (c) `tie_word_embeddings`/transpose/`hidden_act` mappings are the
places a silent model bug would hide — the reference forward is the guard.
