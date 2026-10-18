# etl_hf — HuggingFace model loading & compiled execution on etl

## Intent

Load a HuggingFace checkpoint (its `config.json` **and** its weights from disk) and run the **whole model forward pass as ONE etl graph** — trace → lower → compile → load → run — with the same explicit semantics as `etl` itself: no hidden tracing, no hidden weight embedding, no hidden device movement.
`etl_hf` is a **separate top-level package** (`etl_hf`, sibling of `etl`), not a submodule of `etl`: etl core's non-goals forbid NN modules / model zoos / loaders, so model-level code lives here and only *depends on* the public etl API.
It adds **no hard dependency** — `etl` + `numpy` are the only requirements (`safetensors` and `pytorch_model.bin` files are parsed with `numpy` + `struct` + `json` + `zipfile`/`pickle`; `torch`/`transformers`/`safetensors`/`iree` are never imported at module scope).
Two model families are implemented: **`llama`** (HF `llama`/`mistral`/`qwen2`) and **`gpt2`** (HF `gpt2`, e.g. `gpt2`/`distilgpt2`), each as one `@etl.defn` forward whose weights are ordinary graph inputs.
Parity is pinned against an **independent pure-numpy oracle** (`etl_hf/reference.py`, no `etl` import) and — when `torch`+`transformers` are installed — against the real HuggingFace implementation (`tests/test_torch_oracle.py`).
Whole-graph StableHLO export is pinned for both families without any compiler (`tests/test_export.py`), and the compiled `iree`/`llvm-cpu` path is pinned when `iree` is installed (`tests/test_iree_llvm_cpu.py`).
Test suite size (default environment: torch/transformers and iree absent, checkpoints and goldens cached): **553 collected — 541 passed, 12 skipped**; with the `iree` extra (`uv run --extra iree --with pytest`) **548 passed, 5 skipped**. The skips are torch/transformers absent (3 in-process `tests/test_torch_oracle.py` parity cases), `iree` absent (7 without the extra: the six `tests/test_iree_llvm_cpu.py` tests plus one backend-distinctness test in `tests/test_build.py`), and the opt-in `ETL_HF_BIG=1` real-size checkpoints (2); see `Status` under Constraints. Nothing optional is imported at `tests/test_torch_oracle.py`'s module scope: its oracle is a lazy `(torch, transformers)` pair behind a `find_spec` `skipif` probe, so that file's torch-free cases run in every environment.

## Why a separate package

- etl core stays minimal: `etl`'s CONTEXT non-goals exclude NN layers / model loaders / model zoos.
- The HF integration needs host-side semantics etl deliberately does not own (config schema knowledge, checkpoint key naming, weight transposes, tied embeddings, RoPE/mask conventions).
- Nothing here changes etl's public API or its import DAG; `etl_hf` sits above `etl` and imports only public etl entry points (`etl.trace`, `etl.build_cached`, `etl.run`, `etl.TensorSpec`, `etl.core`, `etl.backends.stablehlo` in tests).

## API Surface

Import-time constraint: `import etl_hf` requires **only** `numpy` + `etl`; no torch / transformers / safetensors / iree at module scope (pinned in a subprocess by `tests/test_public_api.py`).

Top-level `etl_hf/__init__.py` re-exports `HFModel`, `ModelConfig`, `build_model`, `load_hf`, `place_params`, `run_model`, `FAMILIES`, `FamilySpec`, `family_for_model_type`, `get_family`, `HFConfigError`, `HFError`, `HFUnsupportedError`, `HFWeightError`, the submodules `build`/`config`/`errors`/`families`/`models`/`ops`/`reference`/`weights`, and `__version__ = "0.1.0"`.

### `etl_hf/errors.py` — the error tree (imports only `etl.core`)

```python
class HFError(etl.core.ETLError)      # catch-all for the whole integration
class HFConfigError(HFError)          # malformed/unsupported config.json: missing/invalid field,
                                      # non-divisible head counts, seq_len > max_position_embeddings,
                                      # family override disagreeing with the config, bad seq_len/batch
class HFWeightError(HFError)          # checkpoint reading, key-mapping and tensor-shape problems:
                                      # unreadable file, missing/unexpected/duplicate key, wrong shape,
                                      # inconsistent tied head, params tree not matching param_specs
class HFUnsupportedError(HFError)     # valid config+checkpoint outside the v1 matrix: unknown family,
                                      # unknown activation, erf-GELU on a compiled backend, llama biases,
                                      # rope_scaling, symbolically-shaped operands in ops.py
```

Pure host-side argument errors (a non-positive `n_heads` to `ops.split_heads`, a non-numeric `input_ids` in `reference.py`) raise `ValueError`/`TypeError`, never an `etl` error class.

### `etl_hf/config.py` — HF `config.json` → frozen static `ModelConfig`

```python
FAMILY_BY_MODEL_TYPE: dict[str, str] = {"llama": "llama", "mistral": "llama",
                                        "qwen2": "llama", "gpt2": "gpt2"}

@dataclass(frozen=True)
class ModelConfig:
    model_type: str                  # raw HF "model_type"
    family: str                      # etl_hf family key: "llama" | "gpt2"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int         # == num_attention_heads when the model has no GQA/MQA
    intermediate_size: int
    hidden_act: str                  # raw HF name, lower-cased/stripped ("SiLU" -> "silu"); NOT validated here
    layer_norm_eps: float            # eps for RMSNorm and LayerNorm alike
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_theta: float | None         # None for the absolute-position family (GPT-2)
    head_dim: int                    # explicit HF override, else hidden_size // num_attention_heads
    raw: dict = field(default_factory=dict, compare=False, repr=False)   # parsed JSON, diagnostics only

    @staticmethod
    def from_json(path) -> "ModelConfig"
    @staticmethod
    def from_dir(model_dir) -> "ModelConfig"                     # <dir>/config.json
    @classmethod
    def from_dict(cls, raw, *, source: str = "<dict>") -> "ModelConfig"
    @property
    def digest(self) -> str          # 16-hex sha256 over every graph-specialising field, `raw` excluded
```

Per-family normalisation: llama reads `rms_norm_eps` (default `1e-6`), `hidden_act` default `"silu"`, `rope_theta` default `10000.0`, `num_key_value_heads` default `num_attention_heads`, `tie_word_embeddings` default `False`, `head_dim` from an explicit override else derived.
GPT-2 normalises its compact keys (`n_embd`/`n_head`/`n_layer`/`n_vocab`/`n_positions`/`n_inner`/`activation_function`/`layer_norm_epsilon`, with canonical-name fallbacks), defaults `n_inner` to `4 * hidden_size` (HF writes `null`), activation to `"gelu_new"`, eps to `1e-5`, `tie_word_embeddings` to `True`, and pins `rope_theta`/`head_dim` to `None`.
Validation: `hidden_size % num_attention_heads == 0` and `num_attention_heads % num_key_value_heads == 0` (else `HFConfigError` naming the file); non-positive/non-int sizes are errors; an unsupported `model_type` lists the supported set; `rope_scaling` (unless `None`/`{}`/`[]`) raises `HFUnsupportedError` instead of silently diverging from HF.
`ModelConfig` values are **static graph specialisation values** (etl's value model): one config = one graph, and `digest` is directly usable as a cache key (two configs differing only in harmless extra HF fields digest equal).

### `etl_hf/weights.py` — checkpoint readers (numpy + stdlib only, no key mapping, no transposition)

```python
read_safetensors(path) -> dict[str, np.ndarray]          # 8-byte LE u64 header length + JSON header + raw buffer
read_torch_bin(path) -> dict[str, np.ndarray]            # zip (torch >= 1.6) OR legacy format, decided by content
load_checkpoint(model_dir_or_file) -> dict[str, np.ndarray]
widen_to_float32(arr: np.ndarray) -> np.ndarray          # float16 -> float32, everything else unchanged
state_dict_parameter_count(state_dict: dict[str, np.ndarray]) -> int
```

- **safetensors**: one file; `__metadata__` skipped; every entry's `dtype`/`shape`/`data_offsets` validated against the buffer (`HFWeightError` naming the tensor); sharded checkpoints are merged by `load_checkpoint` through the index `weight_map`.
- **torch**: zip format (`*/data.pkl` + `*/data/<storage-key>` blobs) vs legacy format (`magic 0x1950A86A20F9469CFC6C` → protocol `1001` → `sys_info` dict → pickled state dict → pickled storage-key list → per key, an int64 element count followed by `numel × itemsize` raw bytes) — chosen by `zipfile.is_zipfile`, never by name; tensor views/offsets/strides (negative strides included) are resolved, `{"state_dict": {...}}` wrappers are unwrapped.
- **Security boundary**: `pytorch_model.bin` reading unpickles data, so `_RestrictedUnpickler` allows only `collections.OrderedDict`, the three `torch._utils` rebuild helpers, `torch.<X>Storage` classes from its dtype table, inert placeholders for `torch.Size`/`device`/`dtype`/`Tensor`, the `copyreg` reconstructors and the builtin base classes; any other global raises `HFWeightError` instead of executing code (no `allow_pickle`, no torch import).
- **Dtype policy**: `float32` preserved, `float16` and `bfloat16` widened to `float32` (bfloat16 via the `u16 << 16 -> f32` bit trick, since numpy has no bfloat16), `float64`/integer/bool kept; every returned array is a normal, writable, C-contiguous `ndarray`.
- **Directory candidate order** in `load_checkpoint`: `model.safetensors` → `pytorch_model.bin` → `model.safetensors.index.json` → `pytorch_model.bin.index.json` (an absent checkpoint error lists every candidate); shard names must be relative paths inside the model directory, a missing shard, a tensor listed in the index but absent from its shard, or a tensor present in two shards all raise `HFWeightError`; a file argument is dispatched by suffix (`.safetensors` / `.bin`/`.pt`/`.pth`/`.ckpt`/`.pkl`) and otherwise by content sniffing.
- Returned names are the **on-disk** checkpoint keys — the `(in_features, out_features)` orientation and canonical naming happen in the per-family canonicalisers, never here.

### `etl_hf/ops.py` — the transformer idioms `etl` does not ship (graph→graph compositions)

```python
MASK_NEG: float = -1e4                                              # additive mask constant (=0 weight after softmax)
softmax(x, axis=-1)                                                 # exp(x - max) / sum(exp(x - max)), keepdims broadcast
layer_norm(x, weight, bias, eps)                                    # biased variance over the LAST axis (torch.nn.LayerNorm)
rms_norm(x, weight, eps)                                            # weight * x * (1/sqrt(mean(x^2, -1, keepdims) + eps))
silu(x)                                                             # x * sigmoid(x)
gelu_tanh(x)                                                        # gelu_new / gelu_pytorch_tanh (tanh approximation)
gelu_erf(x)                                                         # exact erf-GELU — numpy interpreter only
relu                                                                # private impl, reachable via ACTIVATIONS["relu"]
ACTIVATIONS: dict[str, Callable]                                    # silu, swish, gelu_new, gelu_pytorch_tanh, gelu, erf, relu
EXPORT_UNSAFE_ACTIVATIONS: frozenset[str] = frozenset({"gelu", "erf", "gelu_erf"})
resolve_activation(name)                                            # HF name -> builder; HFUnsupportedError lists supported names
activation_is_export_safe(name)                                     # False exactly for the erf-based names
causal_mask(seq_len)                                                # in-graph (S, S) f32 additive mask, 0 on/below the diagonal
rope_cos_sin(seq_len, head_dim, theta) -> (cos, sin)                # in-graph (S, head_dim) f32 tables (HF rotate_half layout)
rope_tables(seq_len, head_dim, theta) -> (cos, sin)                 # host numpy tables — oracle/reporting only
apply_rope(x, cos, sin)                                             # x*cos + rotate_half(x)*sin on (b, heads, s, head_dim)
split_heads(x, n_heads)                                             # (b, s, fused) -> (b, n_heads, s, fused/n_heads)
merge_heads(x)                                                      # exact inverse -> (b, s, n_heads*head_dim)
repeat_kv(x, n_rep)                                                 # GQA/MQA expansion via insert-axis + broadcast + reshape
```

Every helper is a composition of public `etl` ops on `SymbolicTensor` values: no new IR op and no eager numerical path (a concrete `Tensor` argument surfaces the underlying `etl.TraceError`).
Every helper is **static-shape only** — where an extent is needed it is read from the operand's static shape, and a symbolic `Dim`/`DimExpr` raises `HFUnsupportedError` naming the axis (`ops._static_dim` is the single enforcement point).
Dtypes are preserved; nothing here casts implicitly.
Export safety: `gelu_erf` uses `etl.erf`, which the StableHLO exporter defers in v1, so graphs built with it raise `etl.BackendError` on every compiler backend while still tracing/compiling/running on the numpy interpreter — `EXPORT_UNSAFE_ACTIVATIONS`/`activation_is_export_safe` exist so the build layer refuses the compiled path up front.

### `etl_hf/reference.py` — the independent pure-numpy oracle

```python
MASK_NEG, ACTIVATIONS
softmax, layer_norm, rms_norm, silu, gelu_tanh, gelu_erf, relu, resolve_activation,
rope_tables, apply_rope, additive_causal_mask, attention, param_count
```

Deliberately written twice: these primitives are plain numpy in a *different formulation* from the graphs (per-head Python loops in `attention`, hand-written mean/variance, `math.erf`), so a parity failure points at exactly one side.
Hard rules: numpy + stdlib only (the one `etl_hf` import is `HFUnsupportedError`), every numeric input normalised to `float32` and every return value `float32`, explicit `ValueError`/`TypeError` on malformed shapes/dtypes/static values, and `param_count(tree)` for checkpoint accounting.

### `etl_hf/families.py` — the family registry (the one place families are enumerated)

```python
@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str                            # etl_hf family key
    model_types: tuple[str, ...]         # every HF model_type it covers, sorted
    canonicalize_weights: Callable       # (cfg, raw) -> dict[str, np.ndarray]
    param_specs: Callable                # (cfg) -> dict[str, etl.TensorSpec]
    graph: Callable                      # (cfg, seq_len, batch_size=1) -> @etl.defn forward
    reference: Callable                  # (cfg, params, input_ids) -> np.ndarray

FAMILIES: MappingProxyType[str, FamilySpec]          # {"llama": ..., "gpt2": ...}, read-only
get_family(name) -> FamilySpec                       # by family key; HFUnsupportedError otherwise
family_for_model_type(model_type) -> FamilySpec      # delegates to config.FAMILY_BY_MODEL_TYPE
```

Two invariants are checked at **import** time (`RuntimeError`): every `FamilySpec.name` equals its `FAMILIES` key and the module's `FAMILY`, and a module-declared `MODEL_TYPES` must agree with the config map derived values.
Adding a family is one module plus one `FAMILIES` entry — `build` and the public API never hard-code a family; `models/__init__.py` deliberately does not re-export the family modules (import acyclicity).

### `etl_hf/models/_common.py` — shared block assembly (both families use it verbatim)

```python
linear(x, weight, bias=None)                                  # etl.dot(x, weight) [+ bias]; weight is (in, out)
attention(q, k, v, mask, scale)                               # scores->+mask->softmax->dot(v)->merge_heads
gated_mlp(x, w_gate, w_up, w_down, act)                       # act(dot(x, gate)) * dot(x, up), then down
two_layer_mlp(x, w1, b1, w2, b2, act)                         # linear(act(linear(x, w1, b1)), w2, b2)
```

`attention` validates rank-4 head tensors, identical `k`/`v` shapes and matching `(batch, heads, head_dim)` (`HFUnsupportedError`); GQA expansion is the caller's explicit `repeat_kv` step.

### `etl_hf/models/llama.py` and `etl_hf/models/gpt2.py` — one family per module

```python
FAMILY: str                                                    # "llama" / "gpt2"
MODEL_TYPES: tuple[str, ...]                                   # llama only: ("llama", "mistral", "qwen2")
canonicalize_weights(cfg, raw) -> dict[str, np.ndarray]
param_specs(cfg) -> dict[str, etl.TensorSpec]                  # float32, static shapes, canonical order
graph(cfg, seq_len, batch_size=1) -> Callable                  # @etl.defn forward(params, input_ids) -> logits
reference(cfg, params, input_ids) -> np.ndarray                # float32 logits, independent numpy forward
```

Graph contract (both families): `params` is the flat canonical dict (exactly the `param_specs` keys), `input_ids` is `(batch_size, seq_len)` `etl.int32`, `logits` is `(batch_size, seq_len, vocab_size)` float32; all shapes static (one graph per config/`seq_len`/`batch_size`); every weight is a graph **input**, never `etl.constant`.
**Orientation rule — `(in_features, out_features)`** for every weight the graphs feed to `etl.dot(x, W)`: llama linears are stored `(out, in)` in HF and are transposed at load; GPT-2 `Conv1D` weights are already `(in, out)` and are **not** transposed; `lm_head.weight` is `(hidden, vocab)` in both families (HF stores `(vocab, hidden)` — transposed at load).
**Tied embeddings — one graph input, never two:** `tie_word_embeddings=True` means the output head IS `embed.weight`, so both `param_specs(cfg)` and `canonicalize_weights(cfg, raw)` **omit** `lm_head.weight` and the graph feeds the single `params["embed.weight"]` `SymbolicTensor` to the token gather and — as `etl.transpose(embed)` — to the final projection. A tied tree therefore carries exactly one embedding array and one key fewer than the untied shape (`gpt2` 196 keys / 124 439 808 params, `distilgpt2` 100 / 81 912 576, `tiny_gpt2` 36 / 102 714; untied `tiny_llama` 21 / 1 032 272).
**llama canonical keys (`2 + 9·L + 1` untied, `2 + 9·L` tied)**: `embed.weight` ← `model.embed_tokens.weight` (verbatim `(vocab, hidden)`); per layer `attn_norm.weight` ← `input_layernorm.weight`, `attn.{q,k,v,o}.weight` ← `self_attn.{q,k,v,o}_proj.weight`.T, `mlp_norm.weight` ← `post_attention_layernorm.weight`, `mlp.{gate,up,down}.weight` ← `mlp.{gate,up,down}_proj.weight`.T; then `final_norm.weight` ← `model.norm.weight` and, untied only, `lm_head.weight` ← `lm_head.weight`.T.
**gpt2 canonical keys (`16·L + 5` untied, `16·L + 4` tied)**: `embed.weight` ← `wte.weight`, `pos_embed.weight` ← `wpe.weight`; per layer `attn_norm.{weight,bias}` ← `ln_1.*`, `attn.{q,k,v}.{weight,bias}` ← the column blocks `[0:H]`, `[H:2H]`, `[2H:3H]` of the fused `attn.c_attn.{weight,bias}` (biases split along their single axis), `attn.o.{weight,bias}` ← `attn.c_proj.*`, `mlp_norm.{weight,bias}` ← `ln_2.*`, `mlp.fc1.{weight,bias}` ← `mlp.c_fc.*`, `mlp.fc2.{weight,bias}` ← `mlp.c_proj.*`; then `final_norm.{weight,bias}` ← `ln_f.*` and, untied only, `lm_head.weight` ← `lm_head.weight`.T.
Llama attention dims derive from `cfg.head_dim` (`n_heads*head_dim` / `n_kv_heads*head_dim`), never from an assumed `hidden_size // num_attention_heads`; GPT-2 has no per-head override (`head_dim` is always `hidden_size // num_attention_heads`).
Both canonicalisers resolve tied heads explicitly: a tied config whose checkpoint ships a `lm_head.weight` must agree with the embedding — it is verified (`np.allclose`), consumed and dropped, so no duplicate array and no unexpected-key error; a disagreeing tied head is an `HFWeightError` (never "pick one"), and an untied config without `lm_head.weight` is an `HFWeightError` listing the remedy.
Unexpected/duplicate keys are errors, never silently dropped; each family has one documented ignore list — llama `model.rotary_emb.inv_freq`, GPT-2 `h.{i}.attn.bias` / `h.{i}.attn.masked_bias` (derived causal buffers that older exports persisted).
Llama additionally rejects **any** `*.bias` tensor with `HFUnsupportedError` (v1 llama builds are bias-free); GPT-2's state dict may carry the optional `transformer.` prefix in any mix (duplicates rejected).
Both graphs compose the causal mask (`ops.causal_mask`) and RoPE tables (`ops.rope_cos_sin`, llama only) **in-graph** and read every weight from `params` by its flat canonical key.

### `etl_hf/build.py` — build / run / placement / load

```python
DEFAULT_LLVM_CPU_COMPILE_ARGS: tuple[str, ...] = ("--iree-llvmcpu-target-cpu=host",
                                                 "--iree-opt-data-tiling")
DEFAULT_SEQ_LEN: int = 32
resolve_build_options(backend, device, options) -> tuple[str, dict]
build_model(cfg, seq_len, *, batch_size=1, backend=None, device=None, cache=None,
            family=None, **options) -> etl.Executable
run_model(executable, params, input_ids) -> etl.Tensor
place_params(params, device) -> dict[str, etl.Tensor]
load_hf(model_dir, *, seq_len=None, family=None, batch_size=1) -> HFModel

class HFModel:
    def __init__(self, config, params, *, seq_len, batch_size=1, model_dir=None, family=None)
    .config .family .seq_len .batch_size .model_dir .family_spec      # attributes / property
    def params(self) -> dict[str, np.ndarray]                          # canonical host tree (fresh dict)
    def param_specs(self) -> dict[str, etl.TensorSpec]
    def graph_fn(self, *, seq_len=None, batch_size=None) -> Callable   # the @etl.defn forward
    def build(self, **kwargs) -> etl.Executable
    def forward(self, params=None, input_ids=None, **kwargs) -> np.ndarray   # host float32 logits
    def reference(self, params=None, input_ids=None) -> np.ndarray
```

Resolved default backend/option matrix (the ONE implementation is `resolve_build_options`; precedence **explicit argument > environment > etl_hf default**):

| Situation | Resolution |
|---|---|
| `backend` explicit | used verbatim (a backend name or an etl backend instance with `.name`) |
| `backend=None`, `ETL_BACKEND` set | that value |
| `backend=None`, device `None`/cpu | `"numpy"` (the interpreter) |
| `backend=None`, device cuda/rocm/other | `"iree"` (the compiled path) |
| any non-numpy backend | `opt_level="O3"` injected unless passed (`DEFAULT_OPT_LEVEL`) |
| `"iree"` whose effective targets are exactly `["llvm-cpu"]` | `iree_compile_args=list(DEFAULT_LLVM_CPU_COMPILE_ARGS)` injected unless passed |
| `"numpy"` | options returned as an unchanged copy — nothing injected |

- Effective IREE targets: `options["target_backends"]`, else `ETL_TARGET_BACKENDS`, else derived from the device (`None`/cpu → `["llvm-cpu"]`, cuda → `["cuda"]`, rocm/hip → `["rocm"]`, otherwise `[device.kind]`); nothing is injected for cuda/rocm (the flags are llvm-cpu-specific).
- `iree_compile_args` is a **compile-stage** option: `build_model` calls `etl.build_cached`, which forwards options to both `lower` and `compile`, so the flags reach `etl.compile` structurally (passing them to `etl.lower` alone is silently ineffective).
- `build_model` validates inputs in this order: `ModelConfig`/positive `seq_len`/`batch_size`/`seq_len <= cfg.max_position_embeddings` → family resolution (`HFConfigError` on disagreement) → `resolve_build_options` → `_check_activation` (before any tracing) → the memoised forward → `etl.build_cached(forward, spec.param_specs(cfg), etl.TensorSpec((batch_size, seq_len), etl.int32), backend=..., device=..., cache=..., **resolved_options)`.
- `run_model` validates then calls `etl.run`: `input_ids` must be 2-D and match the executable's recorded ids spec (`etl.ShapeError` naming expected vs got) and any integral dtype is accepted, cast to `int32` via `np.ascontiguousarray` (the one documented host-side coercion); non-integral dtypes raise `etl.DTypeError`; `params` is passed through **untouched** (etl validates keys/shapes/dtypes and reports the first mismatching pytree path); the returned logits are an `etl.Tensor` on the executable's device.
- `place_params` performs one explicit `etl.core.from_numpy(array).to(device)` per leaf (rejecting non-string keys and non-numeric leaves with `TypeError`) — the documented preparation for a device-resident run.
- `load_hf` takes a checkpoint **directory** (`<dir>/config.json` plus weights found by `weights.load_checkpoint`), resolves the family, canonicalises, and keeps **only** the canonical tree (`del raw`, so a checkpoint is never held twice); `seq_len=None` means `min(cfg.max_position_embeddings, DEFAULT_SEQ_LEN)`; `seq_len`/`batch_size` are validated and stored, nothing is compiled until `build`/`forward`.
- `HFModel` validates its params against `param_specs` exactly at construction (`HFWeightError` listing missing/extra keys, then per-key shapes) and stores a copy of the mapping.
- `HFModel.forward` = `self.build(**kwargs)` → `run_model` → one explicit `Tensor.to(Device("cpu", 0))` when the output device kind is not `cpu` → `np.asarray(..., np.float32)`; `input_ids=None` means all-zero `int32` ids of the model's static shape.
- `HFModel.reference` runs the family's pure-numpy oracle on the same params/ids (default: the model's own weights, all-zero ids).

## Design Decisions

- **In-graph RoPE, not host tables.** `ops.rope_cos_sin` builds the `(seq_len, head_dim)` cos/sin tables inside the graph from `arange` + `power` + `cos`/`sin` (all StableHLO-exportable), which keeps the graph signature at exactly `(params, input_ids)` — no extra positional input a caller could forget, nothing to keep in sync when the sequence length changes — keeps the emitted IR op count independent of `seq_len` (pinned by `tests/test_export.py`, which counts IR ops for the same graph at two sequence lengths), and avoids an `etl.constant` payload (which inlines bytes into MLIR).
- `ops.rope_tables` (host numpy, same formula) exists for the pure-numpy oracle and reporting only; it is **not** used by the forward graph — do not "optimise" the graph by feeding it as an input.
- **Flat canonical param dict** (`dict[str, ndarray]` / `dict[str, TensorSpec]`), not a nested pytree: keys are matched to HF keys one-to-one by the canonicaliser (nearest-miss suggestions via `difflib`), and the graph reads them by explicit key strings.
- **Weights are deliberately excluded from the build-cache key.** Weights are ordinary graph inputs validated at the run boundary and cannot change the traced program, so the cache key is `(family, cfg.digest, seq_len, batch_size)` layered on top of etl's own `(id(fn), specs tree, backend/device, canonical options)` key: `build_model` keeps a bounded 64-entry LRU of `@etl.defn` forwards keyed exactly on that tuple (built outside the lock, first-insert-wins), which makes `id(fn)` stable so two identical `build_model` calls return the **same** `Executable` object instead of recompiling; keying on weights would force a pointless recompile per weights object / per checkpoint load.
- **The activation export gate runs before any tracing** (`_check_activation`): an erf-GELU config on a compiled backend fails immediately with `HFUnsupportedError` naming the activation, the backend and both remedies (use `backend="numpy"`, or a `gelu_new` config — measured max |gelu_tanh − gelu_erf| = 4.7e-4, peaking near |x| ≈ 2.7) instead of failing deep inside lowering; the numpy interpreter accepts every supported activation.
- **Legacy pickles are a security boundary**: `weights._RestrictedUnpickler` rebuilds tensors from storages and nothing else, so a hostile `pytorch_model.bin` cannot execute code; the allowlist is deliberately explicit and every rejection names the offending `module.name`.
- **`HFModel.forward` returns host `np.ndarray`** and performs exactly one documented `.to(Device("cpu", 0))` on a non-cpu executable — the single hidden-free transfer in the package, confined to the host-facing convenience helper; `run_model` itself returns device-resident `etl.Tensor`s and never copies.
- **`run_model`'s id-dtype coercion is the only host-side type coercion**: integral token ids are cast to `int32` (etl's run boundary is dtype-strict, and ids are integral by definition); params are never filtered, renamed or cast.
- **Static shapes are baked in** rather than symbolic: symbolic reshape/reduce-with-dynamic-dims paths are the exporter's per-op limitation, and "one graph per `seq_len`/`batch_size`" is the documented, memoised contract — a mismatched length is an explicit error, never a recompile surprise.
- **HF layout knowledge lives only in the family modules** (`canonicalize_weights`): readers stay format-only, so a mapping bug is testable one layer at a time and a new family is one module plus one registry entry.
- **Two-formulation parity**: the graphs use `etl` ops, the oracle uses plain numpy with different formulations (per-head loops, tripled-loop attention in the tests), so a shared bug would have to be written twice in two different shapes of code to pass.

## Constraints

- **No new hard dependency.** `numpy` + `etl` only; `torch`/`transformers` appear solely in tests behind `pytest.importorskip` as external golden oracles, `iree` only behind `pytest.mark.skipif` availability guards, and no `safetensors`/`huggingface_hub` package is required (the test downloader uses `urllib`).
- **Never modify `./etl/`** (sibling node — read-only; escalate core gaps to the parent agent).
- Gaps are composed around here and documented: `rsqrt` → `1/sqrt`, `erf` → numpy-backend-only GELU.
- **Explicit errors only**: everything derives from `etl.core.ETLError` via `HFError` (or `ValueError`/`TypeError` for pure host-side argument errors), messages name the file/field/tensor/key, and there is never a silent fallback, dtype swap or positional guess.
- **Static-shape-first**: dynamic dims are not a goal; a mismatched `seq_len`/`batch_size` is an explicit `HFConfigError`/`ShapeError`.
- **Files stay small** (~1000 lines baseline, one concern per module); no duplicate family logic — shared attention/norm/MLP pieces live in `ops.py` / `models/_common.py`.
- **Tests are colocated** in `./etl_hf/tests/` (this is a separate distribution-level package; the repo's `./tests/` tree mirrors `./etl/` only); `pyproject.toml` `testpaths = ["tests", "etl_hf/tests"]` and its package list excludes `etl_hf.tests`.
- **Status**: implemented and tested — 553 collected / 541 passed / 12 skipped without the `iree` extra and 548 passed / 5 skipped with it (torch absent, `ETL_HF_BIG=1` and — without the extra — iree are the skip groups), zero failures; the whole forward of both families traces, verifies, exports to StableHLO and runs on the numpy interpreter, and the same graph compiles and runs as one IREE program on `llvm-cpu` when `iree` is installed (measured on this CPU-only host, best-of-5 warm: `tiny_llama` `seq_len=32` build 1.5 s, forward 3.3 ms, logits 1.0e-7 vs numpy; `tiny_gpt2` build 1.1 s, forward 3.7 ms, 2.2e-8; `distilgpt2` `seq_len=16` build 1.9 s, forward 346 ms, 2.1e-4 — against 6.0 / 1.9 / 12.1 ms for the numpy interpreter, whose OpenBLAS matmuls win on CPU; both graphs are ~170 IR ops for the tiny checkpoints and 450/870 for `distilgpt2`/`gpt2`). The legacy `pytorch_model.bin` path is validated end-to-end against the real HuggingFace implementation (`tests/test_torch_oracle.py`, torch 2.14.1+cpu / transformers 5.20.1: `tiny_llama` 8.9e-8, `tiny_mistral` 1.8e-7 and `sshleifer/tiny-gpt2` 2.2e-8 max-abs at `seq_len=8`, argmax identical at every non-tie position).

## Known Issues

- **Exact-erf GELU is numpy-backend-only.** `hidden_act: "gelu"`/`"erf"` builds `ops.gelu_erf` (which uses `etl.erf`, a v1 StableHLO export deferral): the graph traces and runs on the numpy interpreter and raises an explicit `etl.BackendError` on any compiler backend, with `HFUnsupportedError` raised even earlier by `build_model`'s activation gate; there is never a silent tanh substitute.
- **Static shapes only** — one graph per `(config, seq_len, batch_size)`; a mismatched `seq_len` beyond `max_position_embeddings` is `HFConfigError`, and a mismatched `input_ids` shape is `etl.ShapeError` telling you to build another length rather than reshape.
- **The llama family is bias-free**: any `*.bias` key in a checkpoint raises `HFUnsupportedError` (v1 builds `attention_bias=False`/`mlp_bias=False` models only).
- **Per-family key allowlists**: llama ignores `model.rotary_emb.inv_freq`; GPT-2 ignores `h.{i}.attn.bias` / `h.{i}.attn.masked_bias`. Everything else must map onto the canonical scheme, or it is an `HFWeightError` listing the offending keys.
- **`rope_scaling` is unsupported** (`HFUnsupportedError` from `ModelConfig.from_dict` unless the value is `None`/`{}`/`[]`): a scaled-RoPE model would silently diverge from HuggingFace, so it is rejected rather than approximated.
- **No KV cache, no incremental decode, no sampling, no training**: the package runs a full-sequence forward pass; every call re-runs the whole graph over `seq_len` tokens.
- **`load_hf` requires a directory**: it reads `<dir>/config.json` before the weights, so passing a single checkpoint file path fails with `HFConfigError` naming the missing `config.json` (single-file loading is `weights.load_checkpoint`'s job).
- **numpy-on-CPU vs the compiled graph**: on CPU the numpy interpreter can beat the compiled graph for large matmuls (OpenBLAS), while the compiled path's value is GPU — report both, never conflate them.
- **IREE/llvm-cpu `_invoke` limit on wide GPT-2 graphs**: a GPT-2-shaped graph of ≳10 layers (≳150 graph arguments — every weight is an explicit input) compiles fine and then SIGSEGVs inside `iree.runtime.function._invoke`; the boundary is between 9 layers (149 inputs, OK) and 10 layers (165 inputs, crash), it is independent of parameter count (a 12-layer hidden-96 graph with 1.2 M params crashes too) and is unaffected by `opt_level="O0"` or dropping `--iree-opt-data-tiling`. The 124M `gpt2` checkpoint therefore compiles (2.2 s, 136 KB vmfb, 869 IR ops) but cannot be invoked on iree/llvm-cpu, while `distilgpt2` (6 layers, 101 inputs) runs; use the numpy interpreter for the real-size GPT-2 — this is an upstream runtime limit, not an `etl_hf` behaviour.
- **A tied head is one array, so a tied tree is the model's true parameter count**: `tie_word_embeddings=True` models (`gpt2` 124 439 808, `distilgpt2` 81 912 576, `tiny_gpt2` 102 714) omit `lm_head.weight` and reuse `embed.weight` transposed in-graph; a checkpoint that ships its tied head copy is verified against the embedding and dropped, never kept as a second graph input. Peak RSS at `seq_len=32` is still ~1.1 GB for the 124M `gpt2` because the raw checkpoint dict and the canonical tree coexist while canonicalising (`load_hf` drops the raw dict afterwards).
- **`etl` ships no `rsqrt`, top-level `where`, `softmax`, `layer_norm`, `rms_norm`, `silu` or `iota`**, and its `slice` takes static start/limit tuples (no data-dependent slice): all of these are composed in `ops.py`.
- **Legacy `pytorch_model.bin` layout and `sshleifer/tiny-gpt2`**: the legacy on-disk format writes an **int64 element count before each storage's bytes** (after the pickled state dict and the storage-key list) and stores the storages in trailer key order; consuming `numel × itemsize` bytes per storage **without** that prefix shifts every storage, so `weights._read_torch_legacy` must read the count first (verified on a real legacy file: the trailing bytes exceed the summed storage sizes by exactly 8 per storage).
  `sshleifer/tiny-gpt2` (test slug `tiny_gpt2`, `n_embd=2`) has **ordinary** logits — measured `max |logit| ≈ 0.1285` when its storages are consumed with the prefix — so the `~1e-44` "degenerate checkpoint" figure recorded in `../bench_logs/hf_feasibility_findings.md` describes a shifted parse, not the checkpoint (that file is a read-only sibling; this is a fact about the layout, not a change log).
  A shifted parse still yields correctly-shaped, plausible-looking tensors, and graph-vs-`reference` parity cannot detect it (both sides read the same loaded arrays) — validate a legacy `.bin` checkpoint's tensors against an external oracle (`tests/test_torch_oracle.py`'s torch/transformers goldens) or assert the reader consumes the file exactly.
  The tiny-gpt2 legacy-bin case is covered by the network-guarded `tests/test_real_checkpoints.py`.

## Routing Table

| Path / module | Area |
|---|---|
| `./etl_hf/models/` | Per-family graph builders (`llama.py`, `gpt2.py`) plus the shared block assembly `_common.py`; `models/__init__.py` re-exports nothing (import acyclicity). |
| `./etl_hf/tests/` | pytest suite for this package (parity, export, iree/real-checkpoint gates); `tests/hf_download.py` is the guarded checkpoint downloader used by the network-dependent tests. |
| `./etl_hf/config.py` | HF `config.json` → `ModelConfig`, `digest`, `FAMILY_BY_MODEL_TYPE` (config parsing and family mapping). |
| `./etl_hf/weights.py` | Checkpoint reading: safetensors, torch zip/legacy, sharded indices, dtype policy, restricted unpickler (no key mapping). |
| `./etl_hf/ops.py` | Graph idioms (norms/activations/softmax/mask/RoPE/head layout) + `ACTIVATIONS`/`EXPORT_UNSAFE_ACTIVATIONS`. |
| `./etl_hf/reference.py` | Independent pure-numpy oracle primitives (no `etl` import) — the thing graphs are validated against. |
| `./etl_hf/families.py` | `FamilySpec` + the `FAMILIES` registry, `get_family`, `family_for_model_type`. |
| `./etl_hf/build.py` | `build_model`/`run_model`/`place_params`/`load_hf`/`HFModel` + the default backend/option matrix (`resolve_build_options`). |
| `./etl_hf/errors.py` | The `HFError` hierarchy (rooted at `etl.core.ETLError`). |
| `./etl_hf/tests/hf_download.py` | Stdlib-only guarded downloader: `MODELS`/`BIG_MODELS`/`DEFAULT_SLUGS`, `download_model`, `require_model` (→ `pytest.skip` offline), cache root `$ETL_HF_MODELS_DIR` or `$TMPDIR/etl_hf_models`. |

Cross-references (read-only): `../etl/` → the tensor library this package builds on (public API contract in `../etl/CONTEXT.md`); `../bench_logs/` → the HF feasibility spike, probe scripts and measured results (incl. `../bench_logs/hf_goldens/`, the torch-golden fixtures read by `tests/test_torch_oracle.py`).

## Notes for Agents

- The StableHLO exporter defers `erf`, `gelu`, `tril`, `triu`, `cumsum`, `cumprod`, `flip`, `roll` (all verified to raise `BackendError`) — do not put them in a compiled graph; compose from comparisons/`select`/arithmetic instead, as `ops.py` does.
- `etl.dot` does not auto-cast mixed dtypes (unlike elementwise ops): pass weights of the same dtype as the activations (every `etl_hf` spec is float32).
- Never `etl.constant` a weight — it inlines the payload into the emitted MLIR text; weights are graph inputs (`param_specs`) by contract.
- Export needs `etl.trace(defn, *specs)` first: `stablehlo.export` on a `Defn` raises `TypeError` ("expected a trace.Graph or an etl.ir.Module"), so the sequence is `specs = param_specs(cfg)` → `etl.trace(graph(cfg, seq_len, batch), specs, etl.TensorSpec((batch, seq_len), etl.int32))` → `graph.verify()` → `stablehlo.export(graph)`.
- `param_specs` is a **function** (`param_specs(cfg)`), not a mapping — the same holds for `HFModel.param_specs()`, `graph_fn()`, `params()`.
- `pytest` is not in the base uv env: run `uv run --with pytest python -m pytest etl_hf/tests` (add `--extra iree` for the compiled path).
- The `iree` adapter's teardown can swallow pytest's `-q` summary line: use `--junit-xml` (or read the per-test output) when counting results.
- Use `$TMPDIR` for checkpoints, never `/tmp` (`hf_download` defaults to `$TMPDIR/etl_hf_models`, overridable with `$ETL_HF_MODELS_DIR`); real-size checkpoints are opt-in with `ETL_HF_BIG=1`.
- Never modify `./etl/` (sibling node, read-only) — a missing etl primitive is composed here in `ops.py` and the gap documented above.
