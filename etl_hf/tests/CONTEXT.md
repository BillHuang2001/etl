# etl_hf/tests — colocated pytest suite for the `etl_hf` package

## Intent

Colocated tests for `etl_hf`: a separate top-level distribution, so they live here rather than under `./tests/`, which mirrors `etl` only (`pyproject.toml` `testpaths = ["tests", "etl_hf/tests"]`, package list `include = ["etl*"]`).
The suite pins the **external** contracts — config parsing, checkpoint reading, canonicalisation, graph assembly, whole-graph export, compiled parity, parity against an independent oracle — plus the **functional shape** of the API: a loading half (`etl_hf.load`), a compute-function half (`etl_hf.program`), a staging half (`etl_hf.pipeline`), joined only by ordinary values (`ModelConfig`, a host params dict, an `@etl.defn` forward, an `Executable`).
Everything is offline and numpy-only by default (no torch, no compiler), so the contract verifies on a bare `uv run --with pytest` install; only the guarded checkpoint readers touch the network.

## Counts (default env: torch/transformers and iree absent, checkpoint + golden caches present)

**675 collected in 15 modules — 647 passed, 28 skipped**; with the `iree` extra **666 passed, 9 skipped**; with the extra and `ETL_HF_BIG=1` **672 passed, 3 skipped** (torch absent only — the ROCm device-resident and real-size legs really run on this GPU host).
Skip groups: torch/transformers absent (3, `test_torch_oracle.py`), `iree` absent (19), `ETL_HF_BIG` unset (6).

| Module | Collected | Area |
|---|---|---|
| `test_build.py` | 94 | `pipeline` + `program`: build/run/predict, option matrix, memoisation, placement, `load_hf` |
| `test_config.py` | 70 | HF `config.json` normalisation + explicit errors |
| `test_export.py` | 11 | whole-graph StableHLO export (numpy only) |
| `test_families.py` | 33 | `FAMILIES` registry, `resolve_family`, routing |
| `test_gpt2.py` | 67 | GPT-2 parity + canonicalisation |
| `test_iree_llvm_cpu.py` | 7 | compiled parity on iree/llvm-cpu (1 ungated, 6 need iree) |
| `test_iree_rocm.py` | 11 | device-resident forward on iree/ROCm (1 ungated, 8 need a GPU, 2 also `ETL_HF_BIG`) |
| `test_llama.py` | 43 | Llama/Mistral parity + canonicalisation |
| `test_ops.py` | 83 | `ops` / `models._common` graph idioms vs numpy |
| `test_pack.py` | 38 | packed weights: plans, bit-exact equivalence, export, llvm-cpu + big-gpt2 |
| `test_public_api.py` | 88 | public surface, functional invariants, import hygiene |
| `test_real_checkpoints.py` | 10 | real downloaded checkpoints end-to-end (7 run, 3 big) |
| `test_reference.py` | 59 | the pure-numpy oracle primitives |
| `test_weights.py` | 50 | checkpoint readers + `load_checkpoint` |
| `test_torch_oracle.py` | 11 | optional torch/transformers oracle (8 torch-free, 3 need torch) |

## What each module pins

- `test_config.py` — per-family `config.json` parsing (llama/mistral/qwen2 routing, GQA/MQA, derived vs explicit `head_dim`, RoPE/eps defaults, GPT-2 `n_*` keys incl. `n_inner: null` → `4·n_embd`), `from_dict`/`from_json`/`from_dir` agreement, `digest` stability, frozen-ness, and the config error table (`hidden_size % heads`, `heads % kv_heads`, bad `head_dim`, non-bool tie, `rope_scaling`, non-dict payload).
- `test_weights.py` — the readers on locally written files: safetensors round-trip + corrupt header/offset/shape/dtype table; torch **zip** (negative-stride views, empty tensors, bf16/f16 storages, `{"state_dict": …}` unwrapping, unknown storage type, arbitrary pickled globals rejected); **legacy** (`_LEGACY_MAGIC` + `_LEGACY_PROTOCOL` 1001, per-storage int64 element count, trailer-key order); then `load_checkpoint` candidate order, shard errors, float32 widening, parameter counting.
  The torch format is exercised **without torch**: a `pickle.Pickler` subclass emits real storage persistent ids and `torch._utils` globals through fake `torch` modules registered in `sys.modules` for the pickle's duration only.
- `test_reference.py` — the oracle against hand-written float64 formulas (biased-variance LayerNorm, mean-free RMSNorm, SiLU/tanh-/erf-GELU, the exact additive mask, RoPE `rotate_half`, attention vs a brute-force triple loop incl. GQA), `MASK_NEG` underflowing `exp` to 0 in f32, and proof the module imports no `etl` graph code.
- `test_ops.py` — every composition (`softmax`, `layer_norm`, `rms_norm`, activations, `causal_mask`, RoPE helpers, `split_heads`/`merge_heads`/`repeat_kv`, `_common.*`) built as a real graph through `etl.evaluate` and compared to expectations computed **in the test file** (no `reference` import), pinning graph semantics rather than a reference against itself.
- `test_llama.py` / `test_gpt2.py` — parity of `run(build(forward_fn(cfg, seq_len=…, batch_size=…), cfg), params, ids)` vs `reference_logits(cfg, params, ids)` (GQA/MQA/MHA, batch 2, tied/untied, activation variants, explicit `head_dim`, `rtol=1e-5`/`atol=1e-6`, non-collapse guard on the oracle), and canonicalisation exactness (key set, `(in_features, out_features)` orientation, values, tie resolution, ignored buffers, one explicit error per failure mode).
- `test_families.py` — registry consistency rather than numerics: keys ↔ `FamilySpec.name`, read-only/frozen specs, spec callables being the family modules' own functions, `model_types` agreeing with `FAMILY_BY_MODEL_TYPE`, the exact `__all__` (incl. `resolve_family`), unknown-family errors naming the supported set, and a subprocess pin that the registry pulls in no optional dependency.
- `test_build.py` — the staging modules end-to-end on synthetic configs/params and a hand-written safetensors directory: `build`+`run` parity vs `reference_logits`; the `pipeline.resolve_build_options` matrix (llvm-cpu flags, `opt_level="O3"`, the all-ROCm `"O1"` rule, "explicit wins", `ETL_OPT_LEVEL`/`ETL_BACKEND`/`ETL_TARGET_BACKENDS`, caller dict never mutated, numpy unchanged); memoisation (`forward_fn` identity per `(family, cfg.digest, seq_len, batch_size, pack_weights)`, `_GRAPH_CACHE_MAX` bound, one `Executable` per specialisation, private cache isolating); the **static specialisation** (`seq_len`/`batch_size` belong to `forward_fn`; junk rejected there, a disagreeing value handed to `build` rejected instead of mismatching the ids spec); the pre-trace erf-activation gate; input handling (integral ids → int32, `DTypeError`/`ShapeError` else, an `etl.Tensor` of ids forwarded verbatim, etl pytree errors untouched); `place_params`/`place_ids` as one visible `Tensor.to` per leaf; `predict` host-facing, refusing device params with the device recipe named (metadata-only `_FakeDevicePayload` raising if `to_host()` runs); `load_hf`/`LoadedModel`.
- `test_public_api.py` — `import etl_hf` in a fresh **subprocess** (numpy + `etl` only), `__all__` completeness/reachability, docstrings naming every parameter, star-import equality, submodule attributes, the `etl.core.ETLError` hierarchy, and the functional invariants: `HFModel`/`build_model`/`run_model` exist nowhere, `etl_hf.build` is a **function** of `etl_hf.pipeline` while `etl_hf/build.py` is unimportable, `load_hf` returns a `LoadedModel` NamedTuple (no methods, no `seq_len`), `load_config` + `load_params` reproduce `load_hf` exactly, `forward_fn` is memoised and captures **no** parameters (`len(param_specs) + 1` graph inputs), the staging stages are distinct functions, `run` returns a device-placed `Tensor`, `reference_logits` agrees with a numpy forward.
- `test_export.py` — the "one graph" property at IR level with numpy only: each family module's `graph(cfg, seq_len, batch)` traced against `param_specs(cfg)` + an int32 ids spec, verified, exported, and checked for the whole forward's constructs (attention dots, reductions, softmax exp, in-graph RoPE `cos`/`sin`, RMSNorm `sqrt`, SiLU logistic, causal compare, embedding gather); "weights are inputs" pinned structurally (`tensor_specs == params + 1`, a `stablehlo.constant` ceiling, an O(`seq_len`) largest-literal bound); plus determinism, config sensitivity, static batch extent. Its `_count_ops` is imported by `test_real_checkpoints.py`.
- `test_iree_llvm_cpu.py` — that property behaviourally: `run(build(forward_fn(…), cfg, backend="iree"), params, ids)` parity vs `reference_logits` *and* the numpy interpreter, bit-identical reruns, one-program structure (single entry function, no `external_call`/`runtime_call`), and two plumbing pitfalls — `iree_compile_args` must reach `etl.compile` (spied on `etl.pipeline.compile`, never timings) and llvm-cpu needs `--iree-llvmcpu-target-cpu=host` (the pure-options test is ungated).
- `test_pack.py` — shape-driven `(dtype, rank, shape[1:])` bucketing (input counts per family/tie state, every canonical key covered once, prefix-sum offsets), plan determinism + rejection table, `pack_params(plan, params)` → in-graph static-slice round trip bit-identical per key, packed-vs-unpacked logits bit-identical on numpy and within `1e-6` on iree/llvm-cpu, unpacked staying the default, `run` mismatch errors both ways, "no baked weights" export bounds, `predict(loaded, ids, pack_weights=True)`, the pack→place leaf-count reduction, the `_pack_plan_for` memo, and the real-size 3-input plan under `ETL_HF_BIG=1`.
- `test_iree_rocm.py` — the GPU leg on this host's MI300X (gfx942, ROCm 7.1) through `load_hf` → `forward_fn` → `build(..., device=Device("rocm", 0))` → `run(exe, place_params(…), place_ids(…))` → ONE explicit `.to(cpu)`: the all-ROCm `"O1"` default vs llvm-cpu `"O3"`, 1-based HAL `device_id`, device-resident parity vs numpy **and** the goldens (relative-to-absmax), host params/ids rejected at the run boundary, the same-device loop (bit-identical, `asdevicearray` spy = 0), packed-vs-unpacked device parity, `dot_precision="HIGHEST"` as a true-fp32 no-op, and the big `distilgpt2`/124M `gpt2` packed cases.
- `test_real_checkpoints.py` — real checkpoints from disk (`require_model` → `load_hf` → `forward_fn`/`build`/`run` → `reference_logits`; loading knows nothing about `seq_len`): real config values, canonical key/param counts, the tied contract (`tiny_gpt2` 36 keys / 102,714 params), `inputs == keys + 1`, traced op count, export size with no compiler, logit parity, explicit float32 placement, a loudly-failing mutated config — incl. legacy `sshleifer/tiny-gpt2` and an exact-erf `tiny-random-Mistral` that is numpy-backend-only; plus the `ETL_HF_BIG`+iree packed-124M payoff (3 `packed.*` inputs, 12 bit-identical runs).
- `test_torch_oracle.py` — the optional **external** oracle (real `transformers` + `torch`, CPU) that catches a canonicalisation bug both etl_hf implementations would share: golden-fixture parity, in-process HF-forward parity (1e-4 abs for tiny, 1e-3 abs **and** 1e-5 relative for real-size), strict fixture validation, `test_etl_hf_never_imports_torch`.
  Its oracle is lazy (`_oracle_available()` probes `find_spec` for the `skipif`, `_oracle()` imports inside a test), so the eight torch-free cases always run.
  Golden→checkpoint resolution is **strict**: only `DownloadUnavailable` may skip; an unmappable golden fails naming the slug and its `.meta.json`.

## How to run

`pytest` is not in the base uv env: `uv run --with pytest python -m pytest etl_hf/tests -q` (add `--extra iree` for the compiled path; `uv run --isolated --extra iree …` for GPU work — the base env has no iree, so `etl.devices()` shows only cpu).
Take authoritative counts from `--junit-xml`: iree's nanobind teardown can abort at exit and swallow the `-q` summary; also export `PYTHONUNBUFFERED=1` on a GPU host so buffered diagnostics survive a redirect.
Never assert timings (the GPU is shared): measured numbers live in module-level `MEASURED` dicts or assertion messages (`test_export.py`, `test_iree_*.py`, `test_real_checkpoints.py`).
Use `$TMPDIR` (never `/tmp`) for downloads and scratch files, and never write into `./bench_logs/` or `./etl/`.

## Network and environment gates

Network access never fails a run: `hf_download.require_model` turns `DownloadUnavailable` into `pytest.skip` (`test_real_checkpoints.py`, `test_torch_oracle.py`, `test_pack.py`, `test_iree_rocm.py`), while `test_weights.py`'s own `_download` helper returns `None` and the test skips — an offline machine reports skips, never failures or hangs.
Checkpoints cache under `$ETL_HF_MODELS_DIR` if set, else `$TMPDIR/etl_hf_models`; a cached file is trusted only when non-empty and, for `*.safetensors`, when its 8-byte header length parses and fits in the file.
`ETL_HF_BIG=1` re-enables 6 real-size cases across `test_real_checkpoints.py`, `test_iree_rocm.py`, `test_pack.py`, each also guarding `/proc/meminfo` `MemAvailable` (the packed 124M path peaks near 1.7 GB RSS plus device buffers).
`@requires_iree` (module-level `find_spec` probe) gates the compiling tests; `test_iree_rocm.py` adds `@requires_rocm`, composing it with `etl.devices()` so a CPU host's skip reason separates "iree not installed" from "no rocm device".
No test requires torch: the compiled and conformance paths are numpy-only, and every optional dependency is imported lazily inside a function or probed in a subprocess.

## Golden fixtures

`../bench_logs/hf_goldens/` (resolved from the repo root, never the cwd) is the read-only contract with a separate torch-golden harness; it carries five fixtures (`tiny_random_llama`, `tiny_random_mistral`, `tiny_gpt2`, `distilgpt2`, `gpt2`) and nothing here writes it.
Per model: `<slug>.npz` with `input_ids` (int32, `(batch, seq_len)`) and `logits` (float32, `(batch, seq_len, vocab)` or `(batch*seq_len, vocab)`), plus `<slug>.meta.json` with an HF repo id under `model` or `model_id` (`model` wins) and informational `revision`/`dtype`.
Reading uses `allow_pickle=False`; the meta repo id wins over the slug mapping, and the stored logits must be float32.
An absent directory skips the parity test with a contract restatement; a malformed fixture fails `test_golden_fixture_format`, an unmappable one fails `test_golden_slugs_resolve`.
Measured parity (numpy interpreter vs stored HF logits): `tiny_gpt2` bit-exact, `tiny_random_llama` 8.9e-8, `tiny_random_mistral` 1.2e-7, `distilgpt2` 1.07e-4 (1.3e-6 of 82.1 absmax), `gpt2` 1.53e-4 (1.1e-6 of 138.7).

## `hf_download.py` contract

Stdlib-only test utility (`pytest` imported only inside `require_model`), runnable standalone from the CLI; the `etl_hf` library never imports it.
Exports `HF_BASE`, `MODELS` (the tiny `hf-internal-testing` repos, `sshleifer/tiny-gpt2`, `distilgpt2`, `openai-community/gpt2`, plus the golden aliases `tiny_random_llama`/`tiny_random_mistral`), `ALIASES`, `BIG_MODELS`, `DEFAULT_SLUGS` (the three tiny checkpoints), `DownloadUnavailable`, `models_dir`, `fetch`, `download_model`, `is_available`, `require_model`, `ensure_models`, `main`.
Aliases resolve to the canonical repo id **and** cache dir (import-time invariant); `fetch` is atomic (`<dest>.part` + `os.replace`, capped backoff, HTTP 4xx except 429 fails fast) and always raises `DownloadUnavailable` naming the URL.
`download_model` fetches `config.json` then `model.safetensors` → its index+shards → `pytorch_model.bin` → its index+shards, raising one error listing every candidate; `is_available` never touches the network.
CLI: `python etl_hf/tests/hf_download.py [--slugs a,b] [--root DIR] [--check]`, one line per model, non-zero exit on failure.

## Known Issues / quirks

`sshleifer/tiny-gpt2` is an ordinary oracle case — its legacy `pytorch_model.bin` reads correctly (33 tensors matching `torch.load`, `max |logit| ≈ 0.1285`, 2.2e-8 vs HuggingFace at `seq_len=8`); a shifted parse still yields well-shaped tensors that agree with etl_hf's own numpy reference, so only that external comparison (or the exact-consumption check in `test_weights.py`) catches it.
The `~1e-44` "degenerate checkpoint" figure in `../bench_logs/hf_feasibility_findings.md` describes such a shifted parse, not the checkpoint.
ROCm: `Device("rocm", 0)` maps to HAL `device_id = 1` (1-based; `create_device(0)` raises); fp32 `dot` is true fp32 on iree/ROCm; the first `.to(cpu)` readback and the first `place_params` leaf each pay a one-time driver init that must never be charged to a per-call loop.
`tiny-random-Mistral` uses exact `erf`-GELU, a documented StableHLO v1 deferral, so it is numpy-backend-only.
`test_public_api.py` enumerates the whole surface, so it is the first file to update when the public API changes — it fails loudly on drift rather than tolerating stale names.

## Routing Table

| Path | Area |
|---|---|
| `./test_build.py` | `pipeline` + `program`: `build`/`run`/`predict`/placement, option matrix, memoisation, static specialisation, `load_hf`. |
| `./test_config.py` | `ModelConfig` parsing per family + the config error table. |
| `./test_export.py` | Whole-graph export (numpy only), no baked weights, op/line bounds; `_count_ops` shared with `test_real_checkpoints.py`. |
| `./test_families.py` | `FAMILIES` registry consistency, `__all__`, `resolve_family`, routing. |
| `./test_gpt2.py` | GPT-2 parity vs `reference_logits` + canonicalisation (fused QKV, prefixes, tie). |
| `./test_iree_llvm_cpu.py` | Compiled parity + structural pins on iree/llvm-cpu; compile-stage option plumbing. |
| `./test_iree_rocm.py` | Device-resident forward on iree/ROCm (GPU-guarded): goldens, same-device loop, packed parity, big pins. |
| `./test_llama.py` | Llama/Mistral parity + canonicalisation (GQA/MQA, RoPE, tie). |
| `./test_ops.py` | `ops` / `models._common` idioms vs in-file expectations. |
| `./test_pack.py` | Packed weights: plans, bit-exact equivalence, export bounds, llvm-cpu + big-gpt2. |
| `./test_public_api.py` | Exports, docstrings, submodule reachability, functional invariants, import hygiene. |
| `./test_real_checkpoints.py` | Real downloaded checkpoints end-to-end (network- and `ETL_HF_BIG`-gated). |
| `./test_reference.py` | The independent pure-numpy oracle primitives. |
| `./test_torch_oracle.py` | Optional torch/transformers oracle + golden-fixture contract. |
| `./hf_download.py` | Guarded downloader/cache helper + slug tables + CLI (stdlib only). |
| `../` | The package under test (`config`, `weights`, `ops`, `reference`, `families`, `load`, `program`, `pipeline`, `pack`). |
| `../../etl/` | The tensor library it builds on (read-only from here). |
