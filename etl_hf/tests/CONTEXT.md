# etl_hf/tests — colocated pytest suite for the `etl_hf` package

## Intent

Colocated test suite for `etl_hf`: the package is a separate top-level distribution, so its tests live here rather than under `./tests/`, which mirrors the `etl` package only (`pyproject.toml` `testpaths = ["tests", "etl_hf/tests"]`).
The suite's job is to pin the **external** contracts: config parsing, checkpoint reading, canonicalisation, graph assembly, whole-graph export, compiled parity, and parity against an independent oracle.
Everything is numpy-based by default — no torch, no compiler — so the contract can be verified on a bare `uv run --with pytest` install; the only network-touching tests are the guarded checkpoint readers, which skip when a fetch fails.
Counts below are what `python -m pytest etl_hf/tests -q --collect-only` reports in the default environment (torch/transformers absent, network and the checkpoint cache available): **533 collected in 14 modules** — `526 passed, 7 skipped` with the `iree` extra (`uv run --extra iree --with pytest`), `519 passed, 14 skipped` without it.

| Module | Collected | Area |
|---|---|---|
| `test_build.py` | 73 | `etl_hf.build` — build/run/placement/load, default-option matrix, memoisation |
| `test_config.py` | 70 | `etl_hf.config` — HF `config.json` normalisation and explicit errors |
| `test_export.py` | 11 | whole-graph StableHLO export for both families (numpy only) |
| `test_families.py` | 32 | `etl_hf.families` — registry consistency, routing |
| `test_gpt2.py` | 59 | GPT-2 parity + canonicalisation |
| `test_iree_llvm_cpu.py` | 7 | compiled whole-graph parity on `iree`/llvm-cpu (1 ungated, 6 need iree) |
| `test_llama.py` | 37 | Llama/Mistral parity + canonicalisation |
| `test_ops.py` | 83 | `etl_hf.ops` / `models._common` graph idioms vs numpy |
| `test_public_api.py` | 37 | `etl_hf/__init__.py` public surface, import hygiene |
| `test_real_checkpoints.py` | 9 | real downloaded checkpoints end-to-end (7 run, 2 `ETL_HF_BIG`-gated) |
| `test_reference.py` | 59 | the pure-numpy oracle primitives |
| `test_weights.py` | 42 | checkpoint readers + `load_checkpoint` |
| `test_torch_oracle.py` | 6 | optional torch/transformers golden oracle (3 need torch, 3 are torch-free) |

## Test Strategy

`test_config.py` pins per-family `config.json` parsing: `llama`/`mistral`/`qwen2` routing, GQA and MQA head counts, `num_key_value_heads` defaulting to `num_attention_heads`, derived vs explicit `head_dim`, RoPE/`rms_norm_eps` defaults, GPT-2 `n_*` key normalisation (`n_inner: null` → `4·n_embd`, `activation_function` falling back to `hidden_act`, tie defaulting true) — plus `from_dict`/`from_json`/`from_dir` agreement, digest stability and order-independence, frozen-ness, `raw` excluded from equality/hash, and an explicit-error table (missing/non-positive/non-integer fields, `hidden_size % heads`, `heads % kv_heads`, non-positive eps, bad `head_dim`, non-bool tie, `rope_scaling` rejection, non-dict payload).
`test_weights.py` drives the readers from locally written files: safetensors round-trip bit-exactness (dtypes, scalars, empty shapes, metadata skipping, writable arrays, corrupt-header/offset/shape/dtype validation), the torch **zip** reader (header, offsets and strides incl. negative-stride views, empty tensors, bf16/f16 storages, `_rebuild_parameter` and `{"state_dict": ...}` unwrapping, missing header tolerated, bad protocol, missing storage blob, unknown storage type, arbitrary pickled globals rejected, non-tensor entries rejected), and the **legacy** reader (magic number `_LEGACY_MAGIC` + protocol `_LEGACY_PROTOCOL` = 1001, storages recovered from the pickle's persistent ids — storage type, key, element `numel` — with the raw byte blocks appended in trailer-key order, truncated/unknown-key/non-checkpoint rejection), then `load_checkpoint`'s candidate order (`model.safetensors` → sharded safetensors index → `pytorch_model.bin` → sharded bin index), safetensors preference, direct file paths, shard/index error paths, the float32-widening policy and `state_dict` parameter counting.
The whole torch-format surface is exercised **without torch**: a `pickle` `Pickler` subclass emits real torch storage persistent ids and `torch._utils._rebuild_tensor_v2` globals through fake `torch`/`torch._utils` modules registered in `sys.modules` for the duration of the pickle and always removed again.
`test_reference.py` validates the oracle itself against hand-written float64 formulas (softmax incl. axis 0 and extreme magnitudes, biased-variance LayerNorm, mean-free RMSNorm, SiLU/tanh-GELU/erf-GELU/ReLU, the activation table, the exact additive causal-mask pattern, RoPE tables and `rotate_half`, attention vs a brute-force triple loop incl. GQA and mask broadcasting, `param_count`), asserts `MASK_NEG` really underflows `exp` to exactly 0 in f32, and pins that this module imports **no** `etl` graph code (so the oracle cannot echo the implementation it validates).
`test_ops.py` builds every composition (`softmax`, `layer_norm`, `rms_norm`, activations, `causal_mask`, `rope_cos_sin`, `rope_tables`, `apply_rope`, `split_heads`, `merge_heads`, `repeat_kv`, `_common.linear`/`attention`/`gated_mlp`/`two_layer_mlp`) as a real graph, runs it on the numpy interpreter via `etl.evaluate`, and compares against an expectation computed **in the test file** (no `etl_hf.reference` import) — which pins graph *semantics*, not a reference against itself.
`test_llama.py` / `test_gpt2.py` pin each family twice: numeric parity of the built-and-run graph against that family's `reference` (default config, GQA/MQA/MHA variants, batch 2, layer counts, sequence lengths, tied and untied heads, activation variants, explicit `head_dim`, at `rtol=1e-5`/`atol=1e-6` plus a non-collapse guard on the oracle), and canonicalisation exactness (key set, orientation, values, tie resolution, ignored buffers, and one explicit error per failure mode: unexpected key, missing key, bias on llama, wrong shape, wrong fused-QKV width, position-table mismatch, foreign-family config).
`test_families.py` is about consistency rather than numerics: registry keys ↔ `FamilySpec.name`, read-only mapping, frozen specs, spec callables being the family modules' own functions, `model_types` agreeing with `etl_hf.config.FAMILY_BY_MODEL_TYPE` and with each module's `MODEL_TYPES`, and unknown families/`model_type`s failing with an `HFUnsupportedError` that lists the supported set.
`test_build.py` covers the build/run layer end-to-end from a synthetic HF checkpoint directory written with a local safetensors writer: compiled-pipeline parity per family, the documented default-option matrix (`llvm-cpu` flags, `opt_level="O3"` only for compiled backends, "explicit wins", the caller's dict never mutated, `ETL_BACKEND`/`ETL_TARGET_BACKENDS` env defaults), memoisation identity for equal keys and distinct executables for differing `seq_len`/`batch_size`/options/config/backend/private cache, the pre-trace erf-activation gate, input handling (integral ids → int32, `DTypeError`/`ShapeError` otherwise), `place_params` as one explicit `Tensor.to` per leaf, and `load_hf`/`HFModel` behaviour including the `min(max_position_embeddings, 32)` default `seq_len`.
`test_public_api.py` pins `import etl_hf` in a fresh **subprocess** (no torch/transformers/safetensors/iree pulled in, numpy + `etl` only), `__all__` completeness/duplicate-freedom/reachability, docstrings and documented arguments, star-import equality, submodule attributes, the error hierarchy under `etl.core.ETLError`, and registry reachability.
`test_export.py` proves the "one graph" property at the IR level with numpy only: each family's graph is traced against its own `param_specs` + an int32 ids spec, verified, and exported through `etl.backends.stablehlo.export`, then the text is checked for the whole forward's constructs (attention dots, reductions, softmax exp, in-graph RoPE `cos`/`sin`, RMSNorm `sqrt`/`divide`, SiLU logistic, causal compare, embedding gather) with deliberately **loose lower-bound** op/line counts recorded in a module-level `MEASURED` table; the "weights are inputs, not payloads" rule is pinned structurally (`tensor_specs == params + 1`, a `stablehlo.constant` count ceiling, and an O(`seq_len`) bound on the largest dense literal).
`test_iree_llvm_cpu.py` proves the same property behaviourally on `iree`/`llvm-cpu` for both families and a real `tiny_llama` checkpoint: parity vs the family reference *and* vs the numpy interpreter, bit-identical reruns, the one-program structure (one compiled entry function, no `external_call`/`runtime_call` ops), and the two measured plumbing pitfalls — `iree_compile_args` must reach `etl.compile` (pinned by spying on `etl.pipeline.compile`, never by timing) and llvm-cpu needs `--iree-llvmcpu-target-cpu=host`.
`test_real_checkpoints.py` is the guarded real-checkpoint suite (`require_model` download → `load_hf` → whole-model forward → `reference`): real config values, canonical key/parameter counts, `inputs == keys + 1` (weights are always explicit inputs), traced op count, export size with zero compiler required, and logit parity — including `sshleifer/tiny-gpt2` (legacy-format checkpoint, ordinary ~0.13 logits) and an exact-erf `tiny-random-Mistral` that runs numpy-backend-only.
`test_torch_oracle.py` is the optional **external** oracle (real `transformers` + `torch` on CPU) that can catch a canonicalisation bug both etl_hf implementations would share: golden-fixture parity (`bench_logs/hf_goldens/`), in-process HF-forward parity (tolerances 1e-4 abs for tiny checkpoints, 1e-3 abs **and** 1e-5 relative for real-size ones), strict validation of the fixture format, and `test_etl_hf_never_imports_torch`. Its oracle is gated lazily (`_oracle_available()` probes `importlib.util.find_spec` for the `skipif`, `_oracle()` imports and caches the pair on first use inside a test), so its three torch-free cases (golden-fixture parity, the fixture-format check, the import-hygiene check) collect and run without torch.
`hf_download.py` is imported by `test_real_checkpoints.py` and `test_torch_oracle.py`; `test_weights.py` keeps its own tiny `_download` helper (same `$TMPDIR/etl_hf_models` cache, no `ETL_HF_MODELS_DIR` support) so its three real-checkpoint reader tests stay self-contained.

## Network and environment gates

Network access is guarded two ways and never fails a run: `hf_download.require_model` converts any `DownloadUnavailable` into `pytest.skip` (used by `test_real_checkpoints.py` and `test_torch_oracle.py`), while `test_weights.py`'s local `_download` helper returns `None` on any failure and the test skips with a message.
An offline machine therefore reports every checkpoint test as skipped, never failed and never hung.
The CLI's default slug set (`DEFAULT_SLUGS`) is the three tiny checkpoints (`tiny_llama`, `tiny_mistral`, `tiny_gpt2`); the real-size ones (`distilgpt2` ≈ 0.34 GB, `gpt2` ≈ 0.52 GB) are gated behind `ETL_HF_BIG=1`, and the 124M GPT-2 case is additionally skipped when the host lacks memory headroom.
Checkpoints are cached under `$ETL_HF_MODELS_DIR` when set, otherwise `$TMPDIR/etl_hf_models` (`tempfile.gettempdir()` honours `$TMPDIR`); a cached file is trusted only when it is non-empty and, for `*.safetensors`, when its 8-byte header length parses and fits inside the file.
iree tests are marked with a module-level `@requires_iree` probe, so the 6 compiling tests skip cleanly without the extra (`uv run --extra iree --with pytest`) while the pure options-resolution test (`test_default_iree_llvm_cpu_compile_options`) always runs; none of them needs a GPU.
`test_torch_oracle.py` is the only module whose contract involves torch, and it imports `torch`/`transformers` **nowhere at module scope**: `_oracle_available()` (an `importlib.util.find_spec` probe) drives its `skipif` gates and `_oracle()` imports + caches the pair inside a test, so without them the 3 oracle cases skip and the module's torch-free cases still run.
No test requires torch: the compiled/conformance path is numpy-only, and every other optional dependency is either imported lazily inside a function or detected in a subprocess (`test_public_api.py`, `test_weights.py`, `test_build.py`).

## `hf_download.py` contract

Stdlib-only module (imports `argparse`/`json`/`os`/`sys`/`tempfile`/`time`/`urllib`/`pathlib`; `pytest` only inside `require_model`) so it imports and runs standalone from the CLI.
Exports `HF_BASE` (`https://huggingface.co`), `MODELS` (`tiny_llama` → `hf-internal-testing/tiny-random-LlamaForCausalLM`, `tiny_mistral` → `hf-internal-testing/tiny-random-MistralForCausalLM`, `tiny_gpt2` → `sshleifer/tiny-gpt2`, `distilgpt2` → `distilgpt2`, `gpt2` → `openai-community/gpt2`), `BIG_MODELS` (`{"distilgpt2", "gpt2"}`), `DownloadUnavailable`, `models_dir`, `fetch`, `download_model`, `is_available`, `require_model`, `ensure_models` and `main`.
`fetch(url, dest, *, timeout=120.0, retries=2)` is atomic — the body streams to `<dest>.part` and is `os.replace`-d into place, retrying transient failures with capped exponential backoff while HTTP 4xx (except 429) fails immediately — and every failure raises `DownloadUnavailable` naming the URL and the status/reason, including an empty body.
`download_model(slug, *, root=None, revision="main")` accepts a `MODELS` key or a raw `org/name` repo id, fetches `config.json` first, then the weight candidates in order (`model.safetensors`, `model.safetensors.index.json` + its `weight_map` shards, `pytorch_model.bin`, `pytorch_model.bin.index.json` + its shards), and raises one error listing every candidate when none works — a half-downloaded directory is never returned as usable.
Cache directories are named from the slug/repo id with `/` replaced by `__`; `is_available` checks the cache without touching the network; `ensure_models` collects failures into a single error and never returns a partial dict.
CLI: `python etl_hf/tests/hf_download.py [--slugs a,b] [--root DIR] [--check]`, printing one line per model (directory, weight-file count, size in MB) and exiting non-zero on any failure.
This module is a **test utility** — the `etl_hf` library never imports it.

## Golden fixtures

`bench_logs/hf_goldens/` (resolved from the repository root, never from the process cwd) is the read-only contract between this suite and a separate torch-golden harness; it does not exist in the repository today and nothing here writes it.
One pair per model: `<slug>.npz` with the required arrays `input_ids` (int32, `(batch, seq_len)`) and `logits` (float32, either `(batch, seq_len, vocab_size)` or `(batch*seq_len, vocab_size)`), and `<slug>.meta.json` carrying at least `{"model": "<hf repo id>"}` plus informational `revision`/`dtype` keys.
Reading uses `allow_pickle=False`; `model` wins over the slug mapping (otherwise the slug must be a `MODELS` key so the checkpoint can be located), and `dtype` is informational only — the stored logits must still be float32.
An absent directory makes the golden-parity test skip with a message restating the contract; a present but malformed fixture is reported strictly by `test_golden_fixture_format`, which is how the contract is enforced on the producer.

## Known Issues / quirks

`pytest` is not in the base uv environment: run `uv run --with pytest python -m pytest etl_hf/tests -q`, and use `uv run --extra iree …` to exercise the compiled path.
iree's nanobind teardown can abort the interpreter at exit and swallow pytest's `-q` summary line, so take authoritative counts from `--junit-xml` rather than from the last printed line.
Use `$TMPDIR` (never `/tmp`) for downloaded checkpoints and scratch files, and never let a test write into `./bench_logs/` or `./etl/` — both are out of scope for this package.
`sshleifer/tiny-gpt2` (slug `tiny_gpt2`) is an ordinary in-process oracle case: its legacy `pytorch_model.bin` layout is read correctly (the trailer's per-storage int64 element count is consumed), so its 33 tensors match `torch.load` and its logits are ordinary (`max |logit| ≈ 0.1285`) — measured vs HuggingFace at `seq_len=8`: max-abs 2.2e-8, argmax identical at every non-tie position. That case *is* the legacy reader's end-to-end check against an external oracle — a shifted parse yields correctly-shaped tensors that still agree with `etl_hf`'s own numpy reference, so only an external comparison can catch it. `tiny-random-Mistral` uses exact `erf`-GELU and is therefore numpy-backend-only.
The compiled-path tests must never assert timings (they would be flaky); measured compile/run numbers land in the module-level `MEASURED` dicts (`test_export.py`, `test_iree_llvm_cpu.py`) and in assertion messages instead.

## Routing Table

| Path | Area |
|---|---|
| `./test_build.py` | `etl_hf.build`: `build_model`/`run_model`/`place_params`/`load_hf`/`HFModel`, option matrix, memoisation. |
| `./test_config.py` | `ModelConfig` parsing per family + the config error table. |
| `./test_export.py` | Whole-graph StableHLO export (numpy only), no baked weights (bounded literal sizes), op-count/line lower bounds. |
| `./test_families.py` | `FAMILIES` registry consistency and `model_type` routing. |
| `./test_gpt2.py` | GPT-2 graph parity vs reference + canonicalisation (fused QKV, prefix forms, tie). |
| `./test_iree_llvm_cpu.py` | Compiled whole-graph parity and structural pins on `iree`/llvm-cpu. |
| `./test_llama.py` | Llama/Mistral graph parity vs reference + canonicalisation. |
| `./test_ops.py` | `etl_hf.ops` / `_common` graph idioms vs expectations computed in-file. |
| `./test_public_api.py` | Package exports, docstrings, submodule reachability, import hygiene (subprocess). |
| `./test_real_checkpoints.py` | Real downloaded checkpoints end-to-end (network- and `ETL_HF_BIG`-gated). |
| `./test_reference.py` | The independent pure-numpy oracle primitives. |
| `./test_torch_oracle.py` | Optional torch/transformers external oracle + golden-fixture contract (6 collected: 3 need torch, 3 are torch-free). |
| `./hf_download.py` | Guarded checkpoint downloader/cache helper + CLI (test utility, stdlib only). |
| `../` | The package under test (`etl_hf`: config, weights, ops, reference, families, build, models). |
| `../../etl/` | The tensor library the package builds on (read-only from here). |
