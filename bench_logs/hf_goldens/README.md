# hf_goldens — ground-truth HuggingFace logits (torch/transformers oracle) for `etl_hf`

Durable reference fixtures for validating the `etl_hf` package: **for each checkpoint, the exact
`logits` a real `transformers`/PyTorch forward pass produces** on a fixed, deterministic
`input_ids`, plus everything needed to reproduce it (config, versions, weight-file layout, tensor
key listing).

Torch/transformers are **never** imported by the test suite to read these files — the `.npz` is
plain numpy. Only the **regeneration** script needs torch, and it must run in a throwaway venv
(see the CRITICAL note below).

```
bench_logs/hf_goldens/
├── README.md                 # this file
├── probe_hf_goldens.py       # (in ./bench_logs/) the generator — needs torch+transformers
├── verify_hf_goldens_meta.py # (in ./bench_logs/) numpy-only meta/npz contract check
├── oracle_report.json        # machine-readable summary of the last generation run
├── <slug>.npz                # logits (f32 (1,S,V)) + input_ids (i32 (1,S))
├── <slug>.meta.json          # model id, config summary, versions, recipes, sanity numbers,
│                             # tie/untie observations, non-parameter buffers to skip
└── <slug>.keys.txt           # FULL on-disk weight tensor listing (name, shape, dtype)
```

## CRITICAL: never install torch into the repo `.venv`

The repo `.venv` is shared by the etl test suite, which uses `pytest.importorskip("torch")` (and
`importorskip("transformers")`) as **skip guards** for the optional-torch interop paths. Installing
torch there would silently flip dozens of skips into real runs. Regeneration therefore uses a
separate throwaway venv under `$TMPDIR`:

```bash
export TMPDIR=${TMPDIR:-/tmp}

uv venv "$TMPDIR/hforacle-venv" --python 3.12          # torch cp312 wheels are the safe bet
uv pip install --python "$TMPDIR/hforacle-venv/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python "$TMPDIR/hforacle-venv/bin/python" transformers safetensors numpy

# HF_HOME keeps ~4 GB of checkpoints out of the repo/home; HF_HUB_DISABLE_XET avoids an
# `.../xet-read-token/...` 404 seen on this network (the plain-HTTP path works).
export HF_HOME="$TMPDIR/hf_oracle_cache" HF_HUB_DISABLE_XET=1
"$TMPDIR/hforacle-venv/bin/python" bench_logs/probe_hf_goldens.py
```

Run it from the repo root. The script is idempotent: it re-downloads from the HF cache and
overwrites the fixtures in place; it ends with a verification pass that reloads **every** `.npz`
and asserts shape/dtype/finiteness/non-constancy, and it exits non-zero if any model failed (a
single-model failure is caught, recorded in `oracle_report.json`, and does not abort the run).

Versions used for the committed fixtures: **transformers 5.20.1, torch 2.14.1+cpu, numpy 2.5.4,
Python 3.12.3**. Bumping transformers/torch may shift the last bits of the logits (different fused
kernels / eager-vs-sdpa paths); regenerate and re-tune tolerances if you do.

## What the fixtures are (exact recipe)

For every model, with `torch.set_grad_enabled(False)`, `model.eval()`, **float32**, **CPU**, and a
plain **forward pass (never `generate`)**:

| step | exact code |
|---|---|
| load | `AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager")` then `.eval()` |
| input ids | `input_ids[0, i] = (i * 7 + 3) % vocab_size`, `i = 0 .. S-1` — no RNG, so it is trivially reproducible outside torch |
| forward | `out = model(input_ids=input_ids); logits = out.logits` (torch int64 ids; `logits` f32, shape `(1, S, vocab_size)`) |

`S = 8` for the three tiny checkpoints, `S = 32` for `distilgpt2` / `gpt2`. The saved `input_ids`
is `int32` (cast to int64 for torch). The loaded logits are saved **unmodified** (no cast of the
values, only dtype assertion f32).

Each model is run twice and the two forwards must be **bit-identical** (`second_forward_maxdiff:
0.0` in every meta) — the oracle is a pure function, and that property is asserted by the generator.

## The fixtures

| slug | model id | `model_type` | S | logits shape | `hidden_act` | weights on disk | #keys | absmax | mean | std | npz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tiny_random_llama` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | `llama` | 8 | (1, 8, 32000) | `silu` | `model.safetensors` | 21 | 0.3488 | 1.57e-04 | 7.99e-02 | 0.95 MB |
| `tiny_random_mistral` | `hf-internal-testing/tiny-random-MistralForCausalLM` | `mistral` | 8 | (1, 8, 32000) | **`gelu` (exact erf!)** | `pytorch_model.bin` | 21 | 0.5609 | 1.77e-04 | 1.13e-01 | 0.95 MB |
| `tiny_gpt2` | `sshleifer/tiny-gpt2` | `gpt2` | 8 | (1, 8, 50257) | `gelu_new` | `pytorch_model.bin` | 33 | 0.1289 | -1.30e-05 | 2.74e-02 | 1.5 MB |
| `distilgpt2` | `distilgpt2` | `gpt2` | 32 | (1, 32, 50257) | `gelu_new` | `model.safetensors` (+ legacy `pytorch_model.bin`) | 82 | 82.11 | -58.54 | 8.65 | 5.2 MB |
| `gpt2` | `gpt2` | `gpt2` | 32 | (1, 32, 50257) | `gelu_new` | `model.safetensors` (+ legacy `pytorch_model.bin`) | 160 | 138.72 | -102.01 | 18.61 | 5.1 MB |

All five are **finite, non-constant, and non-degenerate** (`degenerate_flag: false` everywhere).
The largest `.npz` is 5.2 MB (well under the 30 MB cap).

Config essentials (full detail in each `.meta.json`):

| slug | vocab | hidden | layers | heads | kv heads | head_dim | intermediate | eps | max_pos | tie_word_embeddings | rope_theta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `tiny_random_llama` | 32000 | 16 | 2 | 4 | 4 | 4 | 64 | rms 1e-6 | 2048 | **false (untied)** | 10000.0 |
| `tiny_random_mistral` | 32000 | 32 | 2 | 4 | **2 (GQA 2:1)** | 8 | 37 | rms 1e-6 | 512 | **false (untied)** | 10000.0 |
| `tiny_gpt2` | 50257 | **2** | 2 | 2 | 2 | 1 | 8 (=4·n_embd) | ln 1e-5 | 1024 | true (tied) | n/a |
| `distilgpt2` | 50257 | 768 | 6 | 12 | 12 | 64 | 3072 (=4·768) | ln 1e-5 | 1024 | true (tied) | n/a |
| `gpt2` | 50257 | 768 | 12 | 12 | 12 | 64 | 3072 (=4·768) | ln 1e-5 | 1024 | true (tied) | n/a |

Notes baked into the metadata because they are easy to get wrong:

* **`hidden_act: "gelu"` means EXACT erf-GELU** (HF's `gelu` = `erf` form) — `tiny_random_mistral`
  is the only such checkpoint here, and erf-GELU is **not exportable by etl v1** (`erf`/`gelu` are
  documented StableHLO deferrals). Implementers must either run that checkpoint on the numpy
  backend only or document the tanh-GELU approximation error (measured ~5e-7 max-abs on these
  logits in `../hf_feasibility_findings.md`). `gelu_new` (all three GPT-2s) **is** the tanh form and
  needs no `erf` at all.
* **GPT-2 has no `intermediate_size` in `config.json`** — it is hard-coded to `4 * n_embd`
  (`n_inner: null`). Recorded as `intermediate_size_source: "4 * n_embd (GPT-2 hard-codes n_inner = None)"`.
* **`rope_theta` is not top-level in the Llama/Mistral tiny configs** — it lives inside
  `rope_scaling.rope_theta` (with `rope_type: "default"`). Both metas resolve it to `10000.0`.
* `tiny_gpt2` is `n_embd = 2` with `head_dim = 1`, i.e. numerically absurd but valid; its logits are
  **ordinary magnitude (absmax 0.129)**, NOT denormal. The ~1e-44 outputs previously observed on
  `sshleifer/tiny-gpt2` come from the *compiled* (iree) leg of the old probe, so a ~1e-44 etl result
  is a real bug, not checkpoint degeneracy — this oracle is the reference that distinguishes them.

## Required `<slug>.meta.json` keys

The `etl_hf` golden-fixture consumer reads a fixed contract out of each `<slug>.meta.json`. These
keys are **required** (regeneration must keep emitting them):

| key | type | meaning |
|---|---|---|
| `model` | str | the HuggingFace repo id, e.g. `sshleifer/tiny-gpt2` — **the key the consumer reads** |
| `model_id` | str | legacy spelling of the same repo id; **must equal `model`** (kept for backward compatibility) |
| `slug` | str | the fixture slug = the file stem (`<slug>.npz`, `<slug>.meta.json`, `<slug>.keys.txt`) |
| `seq_len` | int | `S`: the sequence length of the fixed `input_ids` (`8` for the tinies, `32` for `distilgpt2`/`gpt2`) |
| `vocab_size` | int | vocabulary size — the last dim of `logits` (`(1, S, vocab_size)`) |
| `dtype` | str | `"float32"` — the dtype the model was run and the logits were saved in |
| `input_ids_recipe` | str | the exact recipe that rebuilds the saved ids: `input_ids[0, i] = (i * 7 + 3) % vocab_size` |

Everything else in the meta (`config_summary`, `raw_config`, `versions`, `from_pretrained_kwargs`,
`logits_*` sanity numbers, `flags`, `tie_observation`, `skip_keys_non_parameter`, `weights`, `npz`)
is descriptive/regeneration provenance — useful when diagnosing a mismatch, not part of the
consumer's minimum contract.

## Weight-file / key-name observations (the `etl_hf` key-mapping gotchas)

Full per-model listings are in `<slug>.keys.txt` (name, shape, dtype — every key, not a sample);
`<slug>.meta.json → weights.keys_head/keys_tail` carries the first/last 12.

1. **The GPT-2 `transformer.` prefix is INCONSISTENT across checkpoints in the same family.**
   * `gpt2` (recent export, `model.safetensors`): keys are **prefix-free** — `h.0.attn.c_attn.weight`,
     `ln_f.weight`, `wte.weight`, `wpe.weight` (160 keys).
   * `distilgpt2` (`model.safetensors`) and `sshleifer/tiny-gpt2` (`pytorch_model.bin`): keys keep
     `transformer.` — `transformer.h.0.attn.c_attn.weight`, …, `transformer.wte.weight` (82 / 33 keys).
   A loader must normalize (strip optional `transformer.`) rather than match one spelling.
2. **Non-parameter buffers must be skipped.** Every GPT-2 file contains a causal-mask buffer
   `…attn.bias` of shape `(1, 1, 1024, 1024)` — **f32, 4 MB per layer** (48 MB total in `gpt2`) in
   the two real-size checkpoints, but **`uint8`** in `sshleifer/tiny-gpt2`. `sshleifer/tiny-gpt2`
   additionally carries the legacy scalar `…attn.masked_bias` (shape `()`, `-1e4`, 0-d). transformers
   v5 reports `attn.masked_bias` as UNEXPECTED on load (harmless). Do **not** confuse `attn.bias`
   with `attn.c_attn.bias` — the latter is the real fused-QKV bias parameter `(3·n_embd,)`.
   The skip lists are enumerated per model in `meta → skip_keys_non_parameter`.
3. **Tied vs untied heads, from disk:**
   | slug | `lm_head.weight` on disk | config `tie_word_embeddings` | runtime `lm_head ≡ embed` |
   |---|---|---|---|
   | `tiny_random_llama` | yes (32000, 16) | false | **False — a genuinely different matrix** |
   | `tiny_random_mistral` | yes (32000, 32) | false | **False — different** |
   | `tiny_gpt2` | yes (50257, 2) | true | True (a redundant **copy** of `wte`) |
   | `distilgpt2` | no | true | True (implicit tie) |
   | `gpt2` | no | true | True (implicit tie) |
   So "presence of `lm_head.weight`" alone is not a tie signal: `tiny_gpt2` ships an *equal copy*
   while `distilgpt2`/`gpt2` omit it; the Llama/Mistral tinies ship a *different* matrix
   (`tie_word_embeddings: false`). Honour the flag and, if `lm_head.weight` is present, use it.
4. Linear weight orientation differs by family: Llama/Mistral projections are `(out, in)` and need a
   host-side transpose; GPT-2 `Conv1D` weights are `(in, out)` (`c_attn.weight` = (768, 2304)) and
   must **not** be transposed.
5. None of these five repos is sharded (no `*.index.json`, no `model-0000N-of-0000M.safetensors`);
   `distilgpt2`/`gpt2` offer both `model.safetensors` (preferred) and a legacy `pytorch_model.bin`,
   plus flax/tf/rust artifacts that `etl_hf` ignores. Other files present in the snapshot dirs
   (tokenizers, `.tflite`, `.mlmodel`, README, `.gitattributes`) are listed verbatim in
   `meta → weights.model_dir_files`.

## How tests consume these fixtures

```python
import json, numpy as np, pathlib
G = pathlib.Path(__file__).parents[N] / "bench_logs" / "hf_goldens"   # or an env override

z = np.load(G / "tiny_random_llama.npz")
logits, ids = z["logits"], z["input_ids"]        # (1,8,32000) f32, (1,8) i32
meta = json.loads((G / "tiny_random_llama.meta.json").read_text())
assert logits.shape == (1, meta["seq_len"], meta["vocab_size"])
# rebuild the same ids for the etl side:
same = ((np.arange(meta["seq_len"]) * 7 + 3) % meta["vocab_size"])[None, :].astype(np.int32)
assert np.array_equal(same, ids)
```

No torch is needed for this. Compare `etl_hf` forward logits against `logits` with an **fp32
accumulation-order tolerance**, not bit-equality: ~1e-5 abs for the tiny models (absmax ≈ 0.13–0.56)
and ~1e-3 abs / ~1e-5 relative for `distilgpt2`/`gpt2` (absmax 82 / 139 — the same regime the
earlier probe measured, 1.7e-4–5.1e-4 abs ≈ 2e-6 relative vs an independent numpy reference).
Do **not** assert elementwise `argmax` agreement on the tiny random LMs — near-ties are common; use
`logits_argmax_last_pos` from the meta only as a smoke signal.

`oracle_report.json` is the machine-readable record of the last run: per-model shape/format/absmax /
flags plus the verification rows (including `npz_bytes` and `degenerate_flag`) and the tool versions.

## Regeneration history / environment gotchas

* HF **Xet** downloads 404'd on this network (`Client error '404 Not Found' for url
  'https://huggingface.co/api/models/<repo>/xet-read-token/<hash>'`) for `distilgpt2` and `gpt2`;
  setting `HF_HUB_DISABLE_XET=1` makes `snapshot_download` use the plain-HTTP path and works. The
  generator sets it by default.
* `snapshot_download` pulls **all** files in these repos (~2.2 GB for `gpt2` incl. flax/tf/rust);
  `HF_HOME` under `$TMPDIR` keeps that out of the repo and out of `$HOME`.
* transformers v5 renamed `torch_dtype=` → `dtype=`; the generator tries `dtype`, then
  `torch_dtype`, then drops `attn_implementation`, recording which spelling worked in
  `meta → from_pretrained_kwargs` and the effective implementation in
  `meta → attn_implementation_effective` (all five end up `eager`).
* The generator writes only inside `bench_logs/hf_goldens/`; it imports torch/transformers and
  therefore must never run under the repo `.venv` or under pytest.
