# etl_hf — HuggingFace model loading & compiled execution on etl

## Intent

Load a HuggingFace checkpoint (its `config.json` **and** its weights from disk) and run the **whole model
forward pass as ONE etl graph** — trace → lower → compile → load → run — with the same explicit semantics as
`etl` itself: no hidden tracing, no hidden weight embedding, no hidden device movement.

`etl_hf` is a **separate top-level package** (`etl_hf`, sibling of `etl`), not a submodule of `etl`: etl core's
non-goals forbid NN modules / model zoos / loaders, so model-level code lives here and only *depends on* the
public etl API. It adds **no hard dependency** — `numpy` is still the only requirement (`safetensors` and
`pytorch_model.bin` files are parsed with `numpy` + `struct` + `json` + `zipfile`; `torch`/`transformers`//
`iree` are never imported at module scope).

Feasibility is established (`./bench_logs/hf_feasibility_findings.md`): tiny Llama / Mistral / GPT-2
checkpoints and a real `gpt2` (124M, 858 IR ops, 150 inputs) all built as a single stagged graph, compiled
whole on `iree`/`llvm-cpu` in ~2 s, and matched an independent numpy reference at ~1e-7…1e-4 abs.

## Why a separate package

- etl core stays minimal: `etl`'s CONTEXT non-goals exclude NN layers / model loaders / model zoos.
- The HF integration needs host-side semantics etl deliberately does not own (config schema knowledge,
  checkpoint key naming, weight transposes, tied embeddings, RoPE tables).
- Nothing here changes etl's public API or its import DAG; `etl_hf` sits above `etl`.

## API Surface (binding contract)

Import-time constraint: `import etl_hf` must work with **numpy only** (`etl` is a hard dep; no torch /
safetensors / transformers / iree at module scope — optional ones are imported lazily inside functions).

### `etl_hf/config.py`

```python
@dataclass(frozen=True)
class ModelConfig:
    model_type: str              # raw HF "model_type", e.g. "llama" | "mistral" | "qwen2" | "gpt2"
    family: str                  # etl_hf family key: "llama" | "gpt2"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int     # == num_attention_heads when the model has no GQA
    intermediate_size: int
    hidden_act: str              # "silu" | "gelu_new" | "gelu" | "relu" | ...
    layer_norm_eps: float        # eps for RMSNorm and LayerNorm alike
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_theta: float | None     # None for absolute-position families (GPT-2)
    head_dim: int                # explicit HF override when present, else hidden_size // num_attention_heads
    raw: dict                    # the raw parsed JSON (for diagnostics / unknown fields)

    @staticmethod
    def from_json(path) -> "ModelConfig"
    @staticmethod
    def from_dir(model_dir) -> "ModelConfig"
```

`ModelConfig` values are **static graph specialisation values** (per etl's value model): one config = one graph.
Unknown HF fields are ignored; missing required fields raise an explicit `etl_hf.HFConfigError` (subclass of
`ETLError` — see Constraints) naming the field and file.

### `etl_hf/weights.py` — checkpoint readers + canonicalisation (no `etl` import)

```python
read_safetensors(path) -> dict[str, np.ndarray]        # 8-byte LE header len + JSON header + raw buffer
read_torch_bin(path)  -> dict[str, np.ndarray]         # BOTH formats: legacy pickle (proto 1001) and zip
load_checkpoint(model_dir) -> dict[str, np.ndarray]    # *.safetensors / sharded index json / *.bin
```

- Readers return **`float32`-preserving** arrays; bf16/f16 are widened to float32 (documented), f64 kept.
- Arrays are normal writable `ndarray`s (never pickled objects, never `allow_pickle`).
- `pytorch_model.bin` reading must not require torch (restricted unpickler that only rebuilds tensors).

Canonical weight tree (family-specific mapping lives with the family, applied by `canonicalize_weights(cfg, raw)`):

- **Orientation is `(in_features, out_features)`** for every `*.weight` matrix that the graphs feed to
  `etl.dot(x, W)`. Llama linears are stored `(out, in)` in HF → transposed at load; GPT-2 `Conv1D` is already
  `(in, out)` → NOT transposed.
- Canonical names: `embed.weight`, `pos_embed.weight` (absolute-position families only), `final_norm.weight`
  (+ `.bias` where the family has one), `lm_head.weight`, `layers.{i}.attn_norm.weight`,
  `layers.{i}.attn.{q,k,v,o}.weight` (+ `.bias`), `layers.{i}.mlp.{gate,up,down}.weight` (gated families) or
  `layers.{i}.mlp.{fc1,fc2}.weight` (+ `.bias`), `layers.{i}.mlp_norm.weight` (pre-LN families).
- Tied embeddings: when `cfg.tie_word_embeddings` and `lm_head.weight` is absent, it aliases `embed.weight`
  (documented, never silently invented when the checkpoint HAS an lm_head).

### `etl_hf/ops.py` — the missing transformer idioms (compositions of public etl ops)

`softmax(x, axis=-1)`, `layer_norm(x, weight, bias, eps)`, `rms_norm(x, weight, eps)`, `silu(x)`,
`gelu_tanh(x)`, `gelu_erf(x)` (numpy-backend only — `erf` is a v1 StableHLO deferral), `causal_mask(seq_len)`
(composed from comparisons/`select`, never a full `(S,S)` `etl.constant`), `rope_tables(cfg, seq_len)` (host
`(S, head_dim)` cos/sin tables — computed with numpy, passed to the graph), `apply_rope(x, cos, sin)`,
`split_heads(x, cfg)`, `merge_heads(x)`, `repeat_kv(x, n_rep)`.

These are **graph→graph compositions only** — no new IR ops, no eager numerical path in etl.

### `etl_hf/models/<family>.py` — one family per module

```python
param_specs(cfg) -> dict[str, etl.TensorSpec]                    # canonical names, float32, static shapes
graph(cfg, seq_len, batch_size=1) -> Callable[[params, input_ids], logits]   # @etl.defn, static shapes
reference(cfg, params, input_ids) -> np.ndarray                  # independent pure-numpy forward
canonicalize_weights(cfg, raw) -> dict[str, np.ndarray]          # HF key map + transposes + tie resolution
```

- Graph signature: `defn(params: dict[str, SymbolicTensor], input_ids: SymbolicTensor) -> logits` with
  `input_ids` `(batch, seq_len)` int32 and `logits` `(batch, seq_len, vocab_size)` float32.
- **Weights are always explicit graph inputs** (`param_specs`), never `etl.constant`.
- Static shapes only (`batch`, `seq_len` baked in) — symbolic `reshape`/`reduce(keepdims)` defer on the
  compiler backends, so variable length is "one graph per `seq_len`", memoised.
- RoPE / masks are host-precomputed or composed from ops — never giant constant payloads.
- Initial families: **`llama`** (RMSNorm, RoPE, GQA, SwiGLU; also covers `mistral`/`qwen2` configs) and
  **`gpt2`** (LayerNorm, learned positions, `gelu_new`, tied head; covers `gpt2`/`distilgpt2`).
  An exact-`erf` GELU config (`hidden_act: "gelu"`) must raise an explicit, actionable error on compiler
  backends (documented numpy-backend-only path) — never a silent tanh substitute.

### `etl_hf/build.py` — build / compile / run

```python
build_model(cfg, seq_len, *, batch_size=1, backend=None, device=None, cache=None, **options) -> etl.Executable
run_model(executable, params, input_ids) -> etl.Tensor
place_params(params, device) -> dict[str, etl.Tensor]            # explicit Tensor.to per leaf
load_hf(model_dir, *, seq_len=None, family=None, batch_size=1) -> HFModel
class HFModel: .config .params(host np) .param_specs() .graph_fn .forward(params=None, input_ids=None)
               (numpy/eager-free: builds+runs through the compiled pipeline) .build(**kw) .reference(...)
```

- `backend=None` resolves the documented default matrix: numpy interpreter for cpu, `iree` with
  `target_backends=["llvm-cpu"]` **plus `--iree-llvmcpu-target-cpu=host` in `iree_compile_args`** (the
  default `generic` target is 12–19× slower) and `opt_level="O3"` for compiled backends.
- `iree_compile_args` must reach `etl.compile` (passing them to `etl.lower` is silently ineffective).
- Builds are memoised with `etl.build_cached` keyed on config digest + weight identity + `seq_len`/`batch`.
- Device-resident runs: weights pre-placed with `place_params` (host inputs to a non-cpu executable raise
  `DeviceError` by design); same-device loop is the documented iteration pattern.

## Constraints

- **No new hard dependency.** `numpy` + `etl` only. Optional `torch`/`transformers` appear solely in tests
  (`pytest.importorskip`) as golden oracles; `iree` only behind skip guards.
- **Never modify `./etl/`** (sibling node — read-only, escalate core gaps to the parent agent). Gaps are
  worked around here and documented (`rsqrt` → `1/sqrt`; `erf` → numpy-backend-only).
- **Explicit errors only**: all `etl_hf` errors derive from `etl.core.ETLError` (or `ValueError` for pure
  host-side argument errors) with actionable messages naming the file/field/tensor — never a silent fallback,
  never a silent dtype/positional swap.
- **Static-shape-first**: dynamic dims are not a v1 goal; a mismatched `seq_len` is an explicit error, not a
  recompile surprise.
- **Files stay small** (~1000 lines baseline). Module split as above; no duplicate family logic — shared
  attention/norm/MLP pieces live in `ops.py` or a shared `models/_common.py`.
- **Tests are colocated** in `./etl_hf/tests/` (this package is a separate distribution-level package; the
  repo's `./tests/` tree mirrors `./etl/` only). `pyproject.toml` `testpaths` includes `etl_hf/tests`.

## Routing Table

| Path | Area |
|---|---|
| `./etl_hf/models/` | Per-family graph builders (llama, gpt2) + family registry. |
| `./etl_hf/tests/` | pytest suite for this package (numpy parity, export, optional iree/real-checkpoint gates). |

Cross-references (read-only): `../etl/` → the tensor library this package builds on; `../bench_logs/` →
HF feasibility spike, probe scripts and measured results.

## Known Issues / Notes for Agents

- The exporter defers `erf`, `gelu`, `tril`, `triu`, `cumsum`, `cumprod`, `flip`, `roll` — do not use them in a
  compiled graph; compose from comparisons/`select`/`add`.
- `etl` has no `rsqrt`, `where` (top-level), `softmax`, `layer_norm`, `rms_norm`, `silu`, `iota`, or
  dynamic `slice`; see `ops.py`.
- Mixed-dtype `dot` does not auto-cast (unlike elementwise); cast explicitly.
- On CPU the numpy interpreter backend is *faster* than the compiled graph for large matmuls (OpenBLAS); the
  compiled path's value is GPU. Report both, never conflate them.
- `etl.constant` on a weight inlines the payload into MLIR text — never do it.
