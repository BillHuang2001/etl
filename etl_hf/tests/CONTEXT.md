# etl_hf/tests — colocated pytest suite for the `etl_hf` package

## Intent

Colocated tests for `etl_hf`: a separate top-level distribution, so they live here rather than under `./tests/`, which mirrors `etl` only (`pyproject.toml` `testpaths = ["tests", "etl_hf/tests"]`, package list `include = ["etl*"]`, `etl_hf.tests` excluded).
The suite pins the **external** contracts — config parsing, checkpoint reading, canonicalisation, graph assembly, whole-graph export, compiled parity, parity against an independent oracle — plus the **functional shape** of the API: `etl_hf.load` (checkpoint → config + params), `etl_hf.program` (memoised forward, specs, oracle), `etl_hf.pipeline` (build/run/predict/placement), joined only by ordinary values.
Everything is offline and numpy-only by default (no torch, no compiler), so the contract verifies on a bare `uv run --with pytest` install; only the guarded checkpoint readers touch the network.

## Counts (default env: torch/transformers and iree absent, checkpoint + golden caches present)

**675 collected in 15 modules — 647 passed, 28 skipped**; with the `iree` extra **666 passed, 9 skipped**; with the extra and `ETL_HF_BIG=1` **672 passed, 3 skipped** (torch absent only — the ROCm device-resident and real-size legs really run on this GPU host).
Skip groups: torch/transformers absent (3), `iree` absent (19), `ETL_HF_BIG` unset (6).

| Module | Collected | Area |
|---|---|---|
| `test_build.py` | 94 | `pipeline` + `program` staging |
| `test_config.py` | 70 | `ModelConfig` parsing |
| `test_export.py` | 11 | whole-graph StableHLO export |
| `test_families.py` | 33 | family registry |
| `test_gpt2.py` | 67 | GPT-2 parity/canonicalisation |
| `test_iree_llvm_cpu.py` | 7 | iree/llvm-cpu parity |
| `test_iree_rocm.py` | 11 | iree/ROCm device-resident |
| `test_llama.py` | 43 | Llama/Mistral parity/canonicalisation |
| `test_ops.py` | 83 | `ops` graph idioms |
| `test_pack.py` | 38 | packed weights |
| `test_public_api.py` | 88 | public surface + functional invariants |
| `test_real_checkpoints.py` | 10 | real checkpoints end-to-end |
| `test_reference.py` | 59 | the numpy oracle |
| `test_weights.py` | 50 | checkpoint readers |
| `test_torch_oracle.py` | 11 | optional torch oracle |

## What each module pins

- `test_config.py` — per-family `config.json` normalisation (llama/mistral/qwen2 routing, GQA/MQA, derived vs explicit `head_dim`, RoPE/eps defaults, GPT-2 `n_*` keys incl. `n_inner: null` → `4·n_embd`), `from_dict`/`from_json`/`from_dir` agreement, `digest` stability, frozen-ness, and the config error table.
- `test_weights.py` — the readers on locally written files: safetensors round-trip + corrupt header/offset/shape/dtype table, torch **zip** (negative-stride views, storages, `{"state_dict": …}`, unknown storage type, arbitrary pickled globals rejected), **legacy** (`_LEGACY_MAGIC`/`_LEGACY_PROTOCOL`, per-storage int64 count, trailer-key order), then `load_checkpoint` candidate order, shard errors, float32 widening, counting.
  Nothing imports torch: fake `torch` modules go into `sys.modules` only for the duration of a pickle.
- `test_reference.py` — the oracle against hand-written float64 formulas (LayerNorm variance, RMSNorm, the three GELUs, the exact causal mask, RoPE `rotate_half`, attention vs a triple loop incl. GQA), `MASK_NEG` underflow, and proof the module imports no `etl` graph code.
- `test_ops.py` — every composition built as a real graph through `etl.evaluate` and compared to expectations computed **in the test file** (no `reference` import), pinning semantics rather than a reference against itself.
- `test_llama.py` / `test_gpt2.py` — parity of `run(build(forward_fn(cfg, seq_len=…, batch_size=…), cfg), params, ids)` vs `reference_logits(cfg, params, ids)` across head layouts/tie/activations, plus canonicalisation exactness (key set, `(in_features, out_features)` orientation, values, one explicit error per failure mode).
- `test_families.py` — registry consistency: keys ↔ names, frozen/read-only, spec callables are the family modules' own, `model_types` ↔ `FAMILY_BY_MODEL_TYPE`, the exact `__all__` (incl. `resolve_family`), unknown-family reasons, subprocess import hygiene.
- `test_build.py` — the staging modules end-to-end (synthetic configs/params + a hand-written safetensors dir): parity vs `reference_logits`; the `resolve_build_options` matrix (llvm-cpu flags, `O3`, all-ROCm `O1`, explicit/env precedence, caller dict untouched); memoisation (`forward_fn` per `(family, cfg.digest, seq_len, batch_size, pack_weights)`, one `Executable` per specialisation, private cache isolating); **static specialisation** (`seq_len`/`batch_size` belong to `forward_fn`; `build` rejects a disagreement instead of mismatching the ids spec); the pre-trace activation gate; input handling (ids → int32, `Tensor` ids verbatim, pytree errors untouched); one visible `Tensor.to` per `place_params`/`place_ids` leaf; `predict` host-facing, refusing device params via the device recipe (metadata-only fake payload); `load_hf`/`LoadedModel`.
- `test_public_api.py` — `import etl_hf` in a subprocess (numpy + `etl` only), `__all__`/docstrings/star-import/submodules, the `ETLError` hierarchy, and the functional invariants: no `HFModel`/`build_model`/`run_model` anywhere, `etl_hf.build` is `pipeline.build` (no `build.py`), `load_hf` is a `LoadedModel` NamedTuple with no methods and no `seq_len`, `load_config` + `load_params` reproduce it, `forward_fn` is memoised with no captured parameters (`len(param_specs) + 1` inputs), the stages are distinct functions, `run` returns a device-placed `Tensor`.
- `test_export.py` — the "one graph" property at IR level with numpy only: `graph(cfg, seq_len, batch)` traced against `param_specs(cfg)`, verified, exported, and checked for the whole forward's constructs; weights-as-inputs pinned structurally (`tensor_specs == params + 1`, a `stablehlo.constant` ceiling, an O(`seq_len`) literal bound); determinism, config sensitivity, static batch extent. Its `_count_ops` is reused by `test_real_checkpoints.py`.
- `test_iree_llvm_cpu.py` — that property behaviourally on iree/llvm-cpu: parity vs `reference_logits` *and* the numpy interpreter, bit-identical reruns, one-program structure, and the two plumbing pitfalls — `iree_compile_args` must reach `etl.compile` (spied on `etl.pipeline.compile`, never timed) and llvm-cpu needs `--iree-llvmcpu-target-cpu=host` (the options-only test is ungated).
- `test_pack.py` — shape-driven `(dtype, rank, shape[1:])` bucketing (coverage, prefix-sum offsets, determinism, rejection), `pack_params(plan, params)` → static-slice round trip bit-identical per key, packed-vs-unpacked logits bit-identical on numpy and within `1e-6` on iree, unpacked staying the default, mismatch errors both ways, no baked weights, `predict(…, pack_weights=True)`, the `_pack_plan_for` memo, the big 3-input plan.
- `test_iree_rocm.py` — the GPU leg (this host's MI300X, gfx942, ROCm 7.1) via `load_hf` → `forward_fn` → `build(..., device=Device("rocm", 0))` → `run(exe, place_params(…), place_ids(…))` → ONE `.to(cpu)`: the all-ROCm `"O1"` default vs llvm-cpu `"O3"`, the 1-based HAL `device_id`, device-resident parity vs numpy **and** the goldens (relative-to-absmax), host inputs rejected at the run boundary, the same-device loop (`asdevicearray` spy = 0, bit-identical), packed device parity, `dot_precision` as a true-fp32 no-op, the big `distilgpt2`/124M `gpt2` cases.
- `test_real_checkpoints.py` — real checkpoints from disk (`require_model` → `load_hf` → `forward_fn`/`build`/`run` → `reference_logits`; loading knows nothing about `seq_len`): config values, key/param counts, the tied contract, `inputs == keys + 1`, op count, export size with no compiler, parity, float32 placement, a mutated config failing loudly, legacy `tiny-gpt2`, numpy-only erf-GELU Mistral, plus the big packed-124M payoff.
- `test_torch_oracle.py` — the **external** oracle (real torch/transformers, CPU) that catches a bug both etl_hf implementations would share, behind a lazy `find_spec` gate so its 8 torch-free cases always run: golden parities, in-process HF parity (1e-4 abs tiny, 1e-3 abs / 1e-5 relative real-size), strict fixture validation, `test_etl_hf_never_imports_torch`; an unmappable golden fails rather than skipping silently.

## How to run

`pytest` is not in the base uv env: `uv run --with pytest python -m pytest etl_hf/tests -q` (add `--extra iree` for the compiled path; `uv run --isolated --extra iree …` for GPU work — the base env has no iree, so `etl.devices()` shows only cpu).
Take authoritative counts from `--junit-xml`: iree's nanobind teardown can abort at exit and swallow the `-q` summary; also export `PYTHONUNBUFFERED=1` on a GPU host so buffered diagnostics survive a redirect.
Never assert timings (the GPU is shared): measured numbers live in module-level `MEASURED` dicts or assertion messages.
Use `$TMPDIR` (never `/tmp`) for scratch files, and never write into `./bench_logs/` or `./etl/`.

## Network and environment gates

Network access never fails a run: `hf_download.require_model` turns `DownloadUnavailable` into `pytest.skip` (`test_real_checkpoints.py`, `test_torch_oracle.py`, `test_pack.py`, `test_iree_rocm.py`), while `test_weights.py`'s own `_download` returns `None` and the test skips — offline means skips, never failures or hangs.
Checkpoints cache under `$ETL_HF_MODELS_DIR` if set, else `$TMPDIR/etl_hf_models`; a cached file is trusted only when non-empty and, for `*.safetensors`, when its 8-byte header length parses and fits in the file.
`ETL_HF_BIG=1` re-enables 6 real-size cases (`test_real_checkpoints.py`, `test_iree_rocm.py`, `test_pack.py`), each also guarding `/proc/meminfo` `MemAvailable` (the packed 124M path peaks near 1.7 GB RSS plus device buffers).
`@requires_iree` (module-level `find_spec` probe) gates the compiling tests; `test_iree_rocm.py` adds `@requires_rocm`, composing it with `etl.devices()` so a CPU host's skip reason separates "iree not installed" from "no rocm device".
No test requires torch: the compiled and conformance paths are numpy-only, and every optional dependency is imported lazily inside a function or probed in a subprocess.

## Golden fixtures

`../bench_logs/hf_goldens/` (resolved from the repo root, never the cwd) is the read-only contract with a separate torch-golden harness; it carries five fixtures (`tiny_random_llama`, `tiny_random_mistral`, `tiny_gpt2`, `distilgpt2`, `gpt2`) and nothing here writes it.
Per model: `<slug>.npz` with `input_ids` (int32, `(batch, seq_len)`) and `logits` (float32, `(batch, seq_len, vocab)` or `(batch*seq_len, vocab)`), plus `<slug>.meta.json` with an HF repo id under `model` or `model_id` (`model` wins) and informational `revision`/`dtype`.
Reading uses `allow_pickle=False`; the meta repo id wins over the slug mapping, and the stored logits must be float32.
An absent directory skips the parity test with a contract restatement; a malformed fixture fails `test_golden_fixture_format`, an unmappable one fails `test_golden_slugs_resolve`.
Measured parity (numpy interpreter vs stored HF logits): `tiny_gpt2` bit-exact, `tiny_random_llama` 8.9e-8, `tiny_random_mistral` 1.2e-7, `distilgpt2` 1.07e-4, `gpt2` 1.53e-4 (1.1e-6 relative to absmax).

## `hf_download.py` contract

Stdlib-only test utility (`pytest` imported only inside `require_model`), runnable standalone from the CLI; the `etl_hf` library never imports it.
Exports `HF_BASE`, `MODELS` (tiny `hf-internal-testing` repos, `sshleifer/tiny-gpt2`, `distilgpt2`, `openai-community/gpt2`, plus the golden aliases `tiny_random_llama`/`tiny_random_mistral`), `ALIASES`, `BIG_MODELS`, `DEFAULT_SLUGS`, `DownloadUnavailable`, `models_dir`, `fetch`, `download_model`, `is_available`, `require_model`, `ensure_models`, `main`.
Aliases resolve to the canonical repo id **and** cache dir (import-time invariant); `fetch` is atomic (`<dest>.part` + `os.replace`, capped backoff, HTTP 4xx except 429 fails fast) and raises `DownloadUnavailable` naming the URL.
`download_model` tries `config.json` then `model.safetensors` → its index+shards → `pytorch_model.bin` → its index+shards, raising one error listing every candidate; `is_available` never touches the network.
CLI: `python etl_hf/tests/hf_download.py [--slugs a,b] [--root DIR] [--check]`, one line per model, non-zero exit on failure.

## Known Issues / quirks

`sshleifer/tiny-gpt2` is an ordinary oracle case — its legacy `pytorch_model.bin` reads correctly (33 tensors matching `torch.load`, `max |logit| ≈ 0.1285`, 2.2e-8 vs HuggingFace); a shifted parse still yields well-shaped tensors that agree with etl_hf's own numpy reference, so only that external comparison (or `test_weights.py`'s exact-consumption check) catches it.
The `~1e-44` "degenerate checkpoint" figure in `../bench_logs/hf_feasibility_findings.md` describes such a shifted parse, not the checkpoint.
ROCm: `Device("rocm", 0)` maps to HAL `device_id = 1` (1-based; `create_device(0)` raises); fp32 `dot` is true fp32 on iree/ROCm; the first `.to(cpu)` readback and first `place_params` leaf each pay a one-time driver init never to be charged to a per-call loop.
`tiny-random-Mistral` uses exact `erf`-GELU, a StableHLO v1 deferral, so it is numpy-backend-only.
`test_public_api.py` enumerates the whole surface, so it is the first file to update when the public API changes — it fails loudly on drift instead of tolerating stale names.

## Routing Table

| Path | Area |
|---|---|
| `./test_build.py` | `pipeline` + `program`: `build`/`run`/`predict`/placement, options, memoisation, static specialisation, `load_hf`. |
| `./test_config.py` | `ModelConfig` parsing + the config error table. |
| `./test_export.py` | Whole-graph export (numpy only), no baked weights; `_count_ops` shared with `test_real_checkpoints.py`. |
| `./test_families.py` | `FAMILIES` registry, `__all__`, `resolve_family`, routing. |
| `./test_gpt2.py` | GPT-2 parity + canonicalisation (fused QKV, prefixes, tie). |
| `./test_iree_llvm_cpu.py` | Compiled parity + structural pins on iree/llvm-cpu; compile-stage plumbing. |
| `./test_iree_rocm.py` | Device-resident forward on iree/ROCm (GPU-guarded): goldens, same-device loop, packed parity, big pins. |
| `./test_llama.py` | Llama/Mistral parity + canonicalisation (GQA/MQA, RoPE, tie). |
| `./test_ops.py` | `ops` / `models._common` idioms vs in-file expectations. |
| `./test_pack.py` | Packed weights: plans, bit-exact equivalence, export bounds, iree pins. |
| `./test_public_api.py` | Exports, docstrings, submodules, functional invariants, import hygiene. |
| `./test_real_checkpoints.py` | Real downloaded checkpoints end-to-end (network- and big-gated). |
| `./test_reference.py` | The independent pure-numpy oracle primitives. |
| `./test_torch_oracle.py` | Optional torch/transformers oracle + golden-fixture contract. |
| `./hf_download.py` | Guarded downloader/cache helper + slug tables + CLI (stdlib only). |
| `../` | The package under test (`config`, `weights`, `ops`, `reference`, `families`, `load`, `program`, `pipeline`, `pack`). |
| `../../etl/` | The tensor library it builds on (read-only from here). |
