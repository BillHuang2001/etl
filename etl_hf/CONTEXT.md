# etl_hf — HuggingFace checkpoints as one etl graph

## Intent

Load a HuggingFace checkpoint (`config.json` **and** its weights from disk) and run the **whole model forward pass as ONE etl graph** — trace → lower → compile → load → run — with etl's explicit semantics applied to a whole model: no hidden tracing, no hidden weight embedding, no hidden device movement.
`etl_hf` is a **separate top-level package**, never a submodule of `etl`: etl core's non-goals forbid NN modules / loaders / model zoos, so model-level code lives here and only *depends on* the public etl API.
The API is **functional** and split along etl's own value model into three layers that never blur:

| Layer | Module | Question it answers | Surface |
|---|---|---|---|
| **loading** | `load.py` | *what is this checkpoint?* | `load_config`, `load_params`, `load_hf` → `LoadedModel` (plain immutable data), `validate_params` |
| **the compute function** | `program.py` | *what does it compute?* | `forward_fn` (the memoised `@etl.defn` forward, capturing **no** parameters), `param_specs`, `reference_logits`, `resolve_seq_len`, `DEFAULT_SEQ_LEN` |
| **staging** | `pipeline.py` | *where and how does it run?* | `build`, `run`, `predict`, `place_params`, `place_ids`, `resolve_build_options`, the `DEFAULT_*` option constants |

Nothing here is a class with behaviour: `LoadedModel` is a `typing.NamedTuple` of field access only, and every operation is a module-level function taking its inputs as arguments.
Two model families are implemented — **`llama`** (HF `llama`/`mistral`/`qwen2`) and **`gpt2`** (HF `gpt2`, e.g. `gpt2`/`distilgpt2`) — each as one `@etl.defn` forward whose weights are ordinary graph inputs.
Parity is pinned against an **independent pure-numpy oracle** (`reference.py`, no `etl` import) and, when `torch`+`transformers` are installed, against the real HuggingFace implementation (`tests/test_torch_oracle.py`).
Whole-graph StableHLO export is pinned without any compiler (`tests/test_export.py`); the compiled `iree` paths are pinned when `iree` is installed (`tests/test_iree_llvm_cpu.py`, `tests/test_iree_rocm.py`), and the compiled `xla` path is validated by the benchmark harness (`../bench_logs/hf_bench.py` → `../bench_logs/hf_bench_results.md`), not by a gated test.

## Package boundary & distribution status

- **Hard dependency: `numpy` + `etl`, nothing else.** `safetensors` and `pytorch_model.bin` files are parsed with `numpy` + `struct` + `json` + `zipfile`/`pickle`; `torch`/`transformers`/`safetensors`/`iree` are **never** imported at module scope (pinned in a subprocess by `tests/test_public_api.py`).
- **Not shipped in the `etl` distribution — a source-only top-level package used from this checkout.** `pyproject.toml`'s `[tool.setuptools.packages.find]` uses `include = ["etl", "etl.*"]`, which cannot match `etl_hf` / `etl_hf.*` (the `etl.` prefix is literal), so neither a wheel nor an sdist of `etl` contains `etl_hf` and nothing installs it for you: `pip install etl` gives you etl only, and `import etl_hf` works from the repo checkout (or any tree where `etl_hf/` sits beside `etl/` on `sys.path`). The colocated tests stay excluded as belt-and-braces (`exclude = ["etl_hf.tests", "etl_hf.tests.*"]` — never shipped), and `testpaths = ["tests", "etl_hf/tests"]` covers both suites in the checkout. `../tests/test_packaging.py` pins the boundary (find config vs. real packages, numpy-only runtime deps, a built wheel's RECORD).
- **Every optional capability is an optional extra, never a requirement**: `pip install etl[hf]` pulls ONLY the `hf` extra — `torch` + `transformers`, the reference oracles used to validate and benchmark `etl_hf` against the real HuggingFace implementation (never a runtime requirement; the checkpoint readers are numpy + stdlib) — and `pip install etl[iree]` (or `etl[tvm]`, or a user-provided PJRT plugin `.so` for `xla`) enables the compiled path. `etl_hf` itself needs nothing beyond `numpy` + `etl`.
- Nothing here changes etl's public API or import DAG; `etl_hf` imports only public etl entry points (`etl.trace`, `etl.build_cached`, `etl.run`, `etl.TensorSpec`, `etl.core`, `etl.backends.stablehlo` in tests).
- **Import layering inside the package**: `errors`/`_validate` bottom, then `config` → `weights` → `ops`/`reference` → `models/*` → `families` → `pack` → `load` → `program` → `pipeline` → `__init__`. `models/__init__.py` re-exports nothing (each family module is discovered by `families`), and `models/**` never imports `pipeline`/`load`.

## API Surface

Top-level `etl_hf/__init__.py` re-exports exactly: `LoadedModel`, `ModelConfig`, `load_config`, `load_hf`, `load_params`, `validate_params`, `forward_fn`, `param_specs`, `reference_logits`, `build`, `place_ids`, `place_params`, `predict`, `run`, `PackPlan`, `pack_params`, `pack_plan`, `FAMILIES`, `FamilySpec`, `family_for_model_type`, `get_family`, `HFConfigError`, `HFError`, `HFUnsupportedError`, `HFWeightError`, the submodules `config`/`families`/`load`/`ops`/`pack`/`pipeline`/`program`/`reference`/`weights`, and `__version__ = "0.1.0"`.
`etl_hf.build` is the **function** `build` — there is no `build` submodule and no `HFModel`; `etl_hf.models` is reachable as a subpackage but is deliberately not in `__all__`.

### `etl_hf/load.py` — checkpoint → static config + host params (the data half)

```python
class LoadedModel(NamedTuple):      # immutable, field access only — NO methods, NO seq_len/batch_size
    config: ModelConfig
    params: dict[str, np.ndarray]   # canonical (in_features, out_features) tree, exactly param_specs' keys
    family: str                     # "llama" | "gpt2"
    model_dir: Path | None          # where it came from, or None

def load_config(model_dir, *, family: str | None = None) -> ModelConfig
def load_params(model_dir, config: ModelConfig, *, family: str | None = None) -> dict[str, np.ndarray]
def load_hf(model_dir, *, family: str | None = None) -> LoadedModel      # == load_config + load_params
def validate_params(config, params: Mapping[str, np.ndarray], *, family=None) -> dict[str, np.ndarray]
```

- `load_hf` is the documented one-call composition of the other two and **nothing more**: no compilation, no `seq_len`/`batch_size` (those are *build*-time specialisation values owned by `forward_fn`), no caching of a graph.
- `load_config` requires a **directory**: it reads `<dir>/config.json` before any weight is touched, so a single checkpoint *file* path fails with `HFConfigError` naming the missing `config.json` (file-level loading is `weights.load_checkpoint`'s job).
- `load_params` reads the checkpoint (`weights.load_checkpoint`), canonicalises it for the family (orientation, fused-QKV split, tied head), `del`s the raw HF state dict as soon as the map is built (a checkpoint is never held twice), and validates the result with `validate_params` — a mis-shaped canonicalisation raises here, not deep inside a run.
- `validate_params` is the documented pre-flight check for a params mapping that did **not** come from `load_params` (hand-edited tree, converted checkpoint); keys and shapes must match `param_specs` exactly.
- `LoadedModel` is the **only** thing "a loaded model" means: config plus params. Every call that needs a graph takes `loaded.config` (and `loaded.params`) explicitly — nothing is implicit in the record.

### `etl_hf/program.py` — the compute function (specs, forward, oracle)

```python
DEFAULT_SEQ_LEN: int = 32
def resolve_seq_len(config: ModelConfig, seq_len: int | None = None) -> int      # min(max_position_embeddings, 32)
def forward_fn(config, *, seq_len=None, batch_size=None, family=None,
               pack_weights=None) -> Callable                                  # the memoised @etl.defn
def param_specs(config, *, family=None, pack_weights=None) -> dict[str, etl.TensorSpec]
def reference_logits(config, params, input_ids, *, family=None) -> np.ndarray
```

- `forward_fn` returns the **whole model pass** — `(params, input_ids) -> logits`, every weight an ordinary graph **input**, all shapes static. It captures **no parameters** and knows nothing about backends, devices or compilation: build it here, stage it with `build`.
- **Memoised** on `(family, config.digest, seq_len, batch_size, pack_weights)` in a bounded 64-entry LRU (built outside the lock, first-insert-wins): identical calls return the **same object**, so `id(fwd)` — the identity component of `etl.build_cached`'s key — is stable, and two identical `build` calls return the identical `Executable`.
- The returned object carries its specialisation as `fwd._etl_hf_specialisation = {"family", "digest", "seq_len", "batch_size", "pack_weights"}`; `build`/`predict` read their `seq_len`/`batch_size`/`pack_weights` defaults from it and reject an explicit value that disagrees (`HFConfigError` naming `etl_hf.forward_fn`) rather than tracing to a mismatched ids spec.
- `param_specs` is the specs tree `build` traces with and `run` validates against; `pack_weights=True` returns the packed inputs (`PackPlan.specs`), default `False`/`None` the canonical per-weight tree.
- `reference_logits` is the parity oracle — same numbers from a *deliberately different* formulation, both inputs explicit (nothing defaults to a checkpoint's own weights).
- `seq_len=None` everywhere means `resolve_seq_len(config, None)` = `min(config.max_position_embeddings, DEFAULT_SEQ_LEN)`; `batch_size=None` means 1.

### `etl_hf/pipeline.py` — staging (build / run / predict / placement)

```python
DEFAULT_LLVM_CPU_COMPILE_ARGS: tuple[str, ...] = ("--iree-llvmcpu-target-cpu=host", "--iree-opt-data-tiling")
DEFAULT_OPT_LEVEL: str = "O3"
DEFAULT_ROCM_OPT_LEVEL: str = "O1"
def resolve_build_options(backend, device, options: Mapping) -> tuple[str, dict]
def build(fwd, config, *, seq_len=None, batch_size=None, backend=None, device=None, cache=None,
          pack_weights=None, family=None, **options) -> etl.Executable
def run(executable, params: Mapping[str, np.ndarray], input_ids) -> etl.Tensor
def place_params(params: Mapping[str, np.ndarray], device: etl.core.Device) -> dict[str, etl.Tensor]
def place_ids(input_ids, device: etl.core.Device) -> etl.Tensor
def predict(loaded, input_ids=None, *, params=None, seq_len=None, batch_size=None, backend=None,
            device=None, cache=None, pack_weights=None, family=None, **options) -> np.ndarray
```

- **`build`** is the explicit build stage: `etl.build_cached(forward_fn(...), param_specs(config), TensorSpec((batch_size, seq_len), int32), backend=..., device=..., cache=..., **resolved_options)` — trace → lower → compile → load, memoised so an identical specialisation returns the **same** `Executable`. Weight identity is deliberately **not** in the key.
- `build` validates in this order: config / positive `seq_len` / `batch_size` / `seq_len <= max_position_embeddings` → family resolution → `resolve_build_options` → **activation gate before any tracing** → the memoised forward → the cache call. A `pack_weights`/`seq_len` disagreement with `fwd` is an `HFConfigError`.
- **`run`** is validation + `etl.run`: `input_ids` may be a host integral array (2-D, matching the executable's ids spec; the **one** documented host-side coercion is the cast to `int32`, non-integral → `etl.DTypeError`) or an `etl.Tensor`, validated from **metadata only** and forwarded **verbatim** (no `np.asarray`, no cast, no device move). `params` is passed through **untouched**; the only exception is a packed-vs-canonical mix, which raises `HFWeightError` naming `pack_weights` and the `pack_params` remedy. Outputs are `etl.Tensor`s on the executable's device — never copied to the host.
- **`predict`** is the ONE host-facing shorthand: `build(forward_fn(loaded.config, ...), loaded.config, ...)` → `run` → one explicit `Tensor.to(Device("cpu", 0))` when the output is non-cpu → `np.asarray(..., np.float32)`. Non-cpu `params` are rejected up front with `etl.DeviceError` naming the device-resident recipe; `params=None` means `loaded.params`, `input_ids=None` all-zero ids of the static shape, and a canonical tree under `pack_weights=True` is packed once on the host before the run.
- **`place_params`/`place_ids`** are the explicit placement primitives: one `Tensor.to(device)` per leaf (host ids cast to `int32`; a `Tensor` is never re-typed, only moved). Place **once** and keep the placed tree; the device-resident recipe is `build(..., device=d)` + `place_params` + `place_ids` + `run`, which stages nothing per call and never round-trips to the host.
- **`resolve_build_options`** is the ONE implementation of the default matrix; precedence is binding: **explicit argument > environment > etl_hf default**.

| Situation | Resolution |
|---|---|
| `backend` explicit | used verbatim (a name, or an etl backend instance with `.name`) |
| `backend=None`, `ETL_BACKEND` set | that value (honoured, not fought — etl itself would apply it) |
| `backend=None`, device `None`/cpu | `"numpy"` (the interpreter) |
| `backend=None`, device cuda/rocm/vulkan/other | `"iree"` (the compiled path) |
| any non-numpy backend whose effective IREE targets are **all** `rocm`/`hip` | `opt_level="O1"` injected unless passed (`DEFAULT_ROCM_OPT_LEVEL`) — O2/O3 hit an upstream `iree-compile` `'func.func' op failed to distribute` on gpt2-class LayerNorm-reduce shapes, while O0/O1/default emit a byte-identical vmfb |
| any other non-numpy backend | `opt_level="O3"` injected unless passed (`DEFAULT_OPT_LEVEL`) |
| `opt_level` passed explicitly, or `ETL_OPT_LEVEL` set | nothing injected |
| `"iree"` whose effective targets are exactly `["llvm-cpu"]` | `iree_compile_args=list(DEFAULT_LLVM_CPU_COMPILE_ARGS)` injected unless passed |
| `"numpy"` | options returned as an unchanged copy — nothing injected |

- Effective IREE targets: `options["target_backends"]`, else `ETL_TARGET_BACKENDS`, else derived from the device (`None`/cpu → `["llvm-cpu"]`, cuda → `["cuda"]`, rocm/hip → `["rocm"]`, **vulkan → `["vulkan-spirv"]`** — a device *kind* is not a target-backend name, so the bare `vulkan` would be an unknown-target error at compile time — otherwise `[device.kind]`); the llvm-cpu flags are injected only for that exact target.
- `iree_compile_args` is a **compile-stage** option: `build` forwards options to both `lower` and `compile` through `etl.build_cached`, so the flags reach `etl.compile` structurally (passing them to `etl.lower` alone is silently ineffective).
- **`vulkan` is an accepted device kind for `build(...)`**: `build(fwd, config, device=Device("vulkan", 0))` resolves to backend `"iree"` with the `vulkan-spirv` compile target, the `O3` `opt_level` default (it is not an all-ROCm target set) and **no** injected `iree_compile_args` — the same shape as cuda/rocm; placement and the device-resident same-device loop are the ordinary `place_params`/`place_ids`/`run` recipe, and `predict` still makes the one explicit `Tensor.to(Device("cpu", 0))`. Practical caveat (measured on the only Vulkan device this host can reach — lavapipe, i.e. software Vulkan on a CPU): the default IREE Vulkan target profile `vp_android_baseline_2022` allows only **16384 bytes** of workgroup shared memory, so at the default `opt_level="O3"` the 124M-class graphs do not compile (`function 'main_dispatch_0_reduction_32x768_f32' uses 196864 bytes of shared memory; exceeded the limit of 16384 bytes` at `seq_len=32`; `failed to legalize operation 'vector.from_elements'` at `seq_len=128`) — the measured remedy is an explicit **`opt_level="O1"`**, which compiles all of them (tiny checkpoints compile with the defaults). A larger-profile `--iree-vulkan-target=<profile>` in `iree_compile_args` (e.g. `gfx942`, 64 KiB) lifts the shared-memory limit but `O3` then still stops later in vector-dialect lowering. etl_hf injects nothing for either knob: both are explicit caller options. Host/driver details and the full measured matrix: [`../bench_logs/vulkan_findings.md`](../bench_logs/vulkan_findings.md).

### `etl_hf/config.py` — HF `config.json` → frozen static `ModelConfig`

```python
FAMILY_BY_MODEL_TYPE: dict[str, str] = {"llama": "llama", "mistral": "llama", "qwen2": "llama", "gpt2": "gpt2"}
@dataclass(frozen=True)
class ModelConfig:
    model_type, family, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
    num_key_value_heads, intermediate_size, hidden_act, layer_norm_eps, max_position_embeddings,
    tie_word_embeddings, rope_theta, head_dim, raw                       # `raw`: parsed JSON, diagnostics only
    @staticmethod
    def from_json(path) -> "ModelConfig"
    @staticmethod
    def from_dir(model_dir) -> "ModelConfig"
    @classmethod
    def from_dict(cls, raw, *, source: str = "<dict>") -> "ModelConfig"
    @property
    def digest(self) -> str          # 16-hex sha256 over every graph-specialising field, `raw` excluded
```

Llama reads `rms_norm_eps` (1e-6), `hidden_act` (`"silu"`), `rope_theta` (10000.0), `num_key_value_heads` (= heads), `tie_word_embeddings` (False) and an explicit `head_dim` override else `hidden_size // num_attention_heads`.
GPT-2 normalises its compact keys (`n_embd`/`n_head`/`n_layer`/`n_vocab`/`n_positions`/`n_inner`/`activation_function`/`layer_norm_epsilon`, with canonical fallbacks), defaults `n_inner` to `4 × hidden_size` (HF writes `null`), activation to `"gelu_new"`, eps to 1e-5, ties embeddings, and pins `rope_theta`/`head_dim` to `None`.
Validation raises `HFConfigError` for head-count divisibility, non-positive/non-int sizes and unknown `model_type` (listing the supported set), and `HFUnsupportedError` for non-empty `rope_scaling` (a scaled-RoPE model would silently diverge from HF, so it is refused rather than approximated).
`ModelConfig` values are **static graph specialisation values**: one config = one graph, and `digest` is directly usable as a cache key (configs differing only in harmless extra HF fields digest equal).

### `etl_hf/weights.py` — checkpoint readers (numpy + stdlib only, no key mapping)

```python
read_safetensors(path) / read_torch_bin(path) / load_checkpoint(model_dir_or_file) -> dict[str, np.ndarray]
widen_to_float32(arr) -> np.ndarray
state_dict_parameter_count(state_dict) -> int
```

- **safetensors**: 8-byte LE u64 header length + JSON header + raw buffer; `__metadata__` skipped; dtype/shape/offsets validated against the buffer; sharded checkpoints merged through the index `weight_map`.
- **torch**: zip format (`*/data.pkl` + `*/data/<storage-key>`) vs legacy format (magic → protocol 1001 → `sys_info` → pickled state dict → storage-key list → per key an **int64 element count followed by raw bytes**), chosen by `zipfile.is_zipfile`, never by name; views/offsets/strides (negative included) resolved; `{"state_dict": {...}}` unwrapped.
- **Security boundary**: `_RestrictedUnpickler` allows only `collections.OrderedDict`, the three `torch._utils` rebuild helpers, storage classes from its dtype table, inert placeholders and `copyreg` reconstructors — any other global raises `HFWeightError` instead of executing code (no `allow_pickle`, no torch import).
- **Dtype policy**: float32 preserved, float16/bfloat16 widened to float32 (bfloat16 via `u16 << 16`), float64/integer/bool kept; every array is writable and C-contiguous.
- **Directory candidate order**: `model.safetensors` → `pytorch_model.bin` → `model.safetensors.index.json` → `pytorch_model.bin.index.json` (an absent checkpoint lists every candidate); a file argument is dispatched by suffix, else by content sniffing.
- Returned names are the **on-disk** keys — orientation and canonical naming happen in the family canonicalisers, never here.

### `etl_hf/ops.py` — the transformer idioms `etl` does not ship (graph→graph compositions)

```python
softmax, layer_norm, rms_norm, silu, gelu_tanh, gelu_erf, resolve_activation, activation_is_export_safe,
ACTIVATIONS, EXPORT_UNSAFE_ACTIVATIONS, MASK_NEG,
causal_mask(seq_len), rope_cos_sin(seq_len, head_dim, theta), rope_tables(...), apply_rope(x, cos, sin),
split_heads(x, n_heads), merge_heads(x), repeat_kv(x, n_rep)
```

Every helper is a composition of public etl ops on `SymbolicTensor`s (no new IR op, no eager path — a concrete `Tensor` surfaces the underlying `etl.TraceError`), **static-shape only** (`_static_dim` is the single enforcement point; a symbolic `Dim`/`DimExpr` raises `HFUnsupportedError` naming the axis), and never casts implicitly.
`EXPORT_UNSAFE_ACTIVATIONS`/`activation_is_export_safe` exist so the pack/build layer can refuse the compiled path up front for erf-based GELU; `rope_tables` is host numpy for the oracle only and must never feed the graph.

### `etl_hf/reference.py` — the independent pure-numpy oracle

`softmax`, `layer_norm`, `rms_norm`, `silu`, `gelu_tanh`, `gelu_erf`, `relu`, `resolve_activation`, `rope_tables`, `apply_rope`, `additive_causal_mask`, `attention`, `param_count`, `MASK_NEG`, `ACTIVATIONS`.
Deliberately written twice in a *different* formulation (per-head Python loops in `attention`, hand-written mean/variance, `math.erf`), so a parity failure points at exactly one side.
Hard rules: numpy + stdlib only (the single `etl_hf` import is `HFUnsupportedError`), every numeric input/return normalised to `float32`, explicit `ValueError`/`TypeError` on malformed shapes/dtypes.

### `etl_hf/families.py` — the family registry (the one place families are enumerated)

```python
@dataclass(frozen=True, slots=True)
class FamilySpec:
    name: str                      # "llama" | "gpt2"
    model_types: tuple[str, ...]   # every HF model_type it covers, sorted
    canonicalize_weights: Callable # (cfg, raw) -> dict[str, np.ndarray]
    param_specs: Callable          # (cfg) -> dict[str, etl.TensorSpec]
    graph: Callable                # (cfg, seq_len, batch_size=1, *, pack_plan=None) -> @etl.defn forward
    reference: Callable            # (cfg, params, input_ids) -> np.ndarray
FAMILIES: MappingProxyType[str, FamilySpec]
get_family(name) -> FamilySpec                       # HFUnsupportedError otherwise
family_for_model_type(model_type) -> FamilySpec
resolve_family(config, family=None) -> FamilySpec    # the ONE "config + optional override" resolver
```

`resolve_family` is the single place the disagreement rule lives — `load`, `program` and `pipeline` all go through it, so it can never drift between the loading half and the compute half.
Two invariants are checked at **import** time (`RuntimeError`): every `FamilySpec.name` equals its `FAMILIES` key and its module's `FAMILY`, and a module-declared `MODEL_TYPES` must agree with the config map.
Adding a family is one module plus one `FAMILIES` entry — `load`/`program`/`pipeline` never hard-code a family.

### `etl_hf/models/_common.py`, `llama.py`, `gpt2.py` — one family per module

```python
# _common.py (shared verbatim by both families)
linear(x, weight, bias=None), attention(q, k, v, mask, scale),
gated_mlp(x, w_gate, w_up, w_down, act), two_layer_mlp(x, w1, b1, w2, b2, act)
# per family
FAMILY: str; MODEL_TYPES: tuple[str, ...]
canonicalize_weights(cfg, raw), param_specs(cfg), graph(cfg, seq_len, batch_size=1, *, pack_plan=None), reference(cfg, params, ids)
```

Graph contract (both families): `params` is the flat canonical dict (exactly the `param_specs` keys), `input_ids` is `(batch_size, seq_len)` `int32`, `logits` is `(batch_size, seq_len, vocab_size)` float32; all shapes static; every weight a graph **input**, never `etl.constant`.
**Orientation rule — `(in_features, out_features)`** for every weight fed to `etl.dot(x, W)`: llama linears are `(out, in)` in HF and transposed at load; GPT-2 `Conv1D` weights are already `(in, out)` and are **not** transposed; `lm_head.weight` is `(hidden, vocab)` in both (HF stores `(vocab, hidden)`).
**Tied embeddings — one graph input, never two**: `tie_word_embeddings=True` omits `lm_head.weight` from both `param_specs` and `canonicalize_weights`, and the graph feeds `params["embed.weight"]` to the token gather and — as `etl.transpose(embed)` — to the final projection. A tied tree carries one key fewer (`gpt2` 196 keys / 124 439 808 params, `distilgpt2` 100 / 81 912 576, `tiny_gpt2` 36 / 102 714; untied `tiny_llama` 21 / 1 032 272).
**llama canonical keys (`2 + 9·L + 1` untied, `2 + 9·L` tied)**: `embed.weight` ← `model.embed_tokens.weight`; per layer `attn_norm.weight` ← `input_layernorm.weight`, `attn.{q,k,v,o}.weight` ← `self_attn.*_proj.weight`.T, `mlp_norm.weight` ← `post_attention_layernorm.weight`, `mlp.{gate,up,down}.weight` ← `mlp.*_proj.weight`.T; then `final_norm.weight` ← `model.norm.weight` and, untied only, `lm_head.weight` ← `lm_head.weight`.T.
**gpt2 canonical keys (`16·L + 5` untied, `16·L + 4` tied)**: `embed.weight` ← `wte.weight`, `pos_embed.weight` ← `wpe.weight`; per layer `attn_norm.{weight,bias}` ← `ln_1.*`, `attn.{q,k,v}.{weight,bias}` ← column blocks `[0:H]`/`[H:2H]`/`[2H:3H]` of the fused `attn.c_attn.*`, `attn.o.*` ← `attn.c_proj.*`, `mlp_norm.*` ← `ln_2.*`, `mlp.fc1.*` ← `mlp.c_fc.*`, `mlp.fc2.*` ← `mlp.c_proj.*`; then `final_norm.*` ← `ln_f.*` and, untied only, `lm_head.weight` ← `lm_head.weight`.T.
Llama attention dims derive from `cfg.head_dim`; GPT-2 always uses `hidden_size // num_attention_heads`.
Both canonicalisers resolve tied heads explicitly: a tied config shipping `lm_head.weight` is verified (`np.allclose`), consumed and dropped; a disagreeing tied head is `HFWeightError`, and an untied config without it is `HFWeightError` listing the remedy.
Unexpected/duplicate keys are errors, never silently dropped; each family has one documented ignore list — llama `model.rotary_emb.inv_freq`, GPT-2 `h.{i}.attn.bias` / `h.{i}.attn.masked_bias`.
Llama rejects **any** `*.bias` tensor (`HFUnsupportedError`, v1 builds are bias-free); GPT-2 tolerates a `transformer.` prefix in any mix (duplicates rejected).
Both graphs compose the causal mask and (llama) the RoPE tables **in-graph** and read every weight from `params` by its flat canonical key.

### `etl_hf/pack.py` — opt-in packed weights (few graph inputs, static in-graph slices)

```python
PACKED_PREFIX: str = "packed."
@dataclass(frozen=True)
class PackEntry:  input: str; start: tuple[int, ...]; shape: tuple[int, ...]
@dataclass(frozen=True)
class PackPlan:
    specs: dict[str, etl.TensorSpec]   # the PACKED params inputs (3-5 for these families)
    entries: dict[str, PackEntry]      # canonical key -> entry
    def pack(self, params) -> dict[str, np.ndarray]
    def unpack(self, packed) -> dict[str, SymbolicTensor]
def pack_plan(specs: Mapping[str, etl.TensorSpec], *, prefix=PACKED_PREFIX) -> PackPlan
def pack_params(plan: PackPlan, params) -> dict[str, np.ndarray]
def is_packed_tree(tree) -> bool
```

Generic over any `param_specs` tree — **no family-specific code**: buckets are `(dtype, rank, shape[1:])` groups concatenated along axis 0, ordered by their minimum canonical key, keys sorted inside a bucket, offsets = prefix sums, so a plan is deterministic for equal spec trees.
Rank-0 scalars get their own bucket and unpack through `reshape(slice(...), ())` (a static slice cannot yield rank 0).
Measured packed input counts: llama untied **5** / tied **4**, gpt2 untied **4** / tied **3**, real 124M `gpt2` **3** against 196 canonical keys; the packed element count equals the canonical total exactly (lossless and bit-identical). Counts are shape-driven, not role-driven — read them from `pack_plan(...).specs`, never assume a number.
Families opt in with one line: `graph(cfg, seq_len, batch_size, *, pack_plan=None)` and, as the forward's **first** statement, `params = pack_plan.unpack(params)`.
Pure graph→graph composition: **no new IR ops** (one `etl.slice` per canonical key) and never an `etl.constant` of weights. Explicit errors: symbolic/dynamic dims → `HFUnsupportedError`; missing/extra keys or shape/dtype mismatches → `HFWeightError` naming the key.

### `etl_hf/errors.py` and `etl_hf/_validate.py`

```python
class HFError(etl.core.ETLError)      # catch-all for the whole integration
class HFConfigError(HFError)          # malformed/unsupported config.json, bad seq_len/batch_size, family override disagreement, forward specialisation mismatch
class HFWeightError(HFError)          # checkpoint reading, key mapping, tensor shapes, tied-head disagreement, packed-vs-canonical mix
class HFUnsupportedError(HFError)     # outside the v1 matrix: unknown family, unknown activation, erf-GELU on a compiled backend, llama biases, rope_scaling, symbolic dims
```

Pure host-side argument errors (a non-positive `n_heads` to `ops.split_heads`, a non-numeric `input_ids` in `reference.py`) raise `ValueError`/`TypeError`, never an etl error class.
`_validate.py` holds the internal argument guards (`positive_int`, `require_config`, `require_bool`, `require_device`) shared by `load`/`program`/`pipeline`, so the same message shape is used everywhere.

## Design Decisions

- **Functional, not OOP.** Behaviour lives in module functions that take their inputs as arguments; `LoadedModel` is a `NamedTuple` of field access only. There is no object that bundles config + params + a graph, so no call can silently reuse a stale `seq_len`, a stale backend or a stale placed tree — every piece is passed explicitly, exactly like `etl`'s own pipeline-object model.
- **The compute function captures no parameters.** `forward_fn(config, seq_len=..., batch_size=..., pack_weights=...)` is pure in its static specialisation; weights are graph inputs (`param_specs`). This is why the build cache can exclude weight identity, why one graph serves every checkpoint with the same config, and why a test can assert that no ndarray appears in any closure or default of the returned forward.
- **`seq_len`/`batch_size`/`pack_weights` are *build*-time specialisation values**, not load-time ones: `LoadedModel` carries no `seq_len`, and `forward_fn` stamps its choice on the returned object (`_etl_hf_specialisation`) so `build`/`predict` derive the ids spec from the graph that actually exists instead of guessing — an explicit disagreement is an `HFConfigError`, never a silently mismatched spec.
- **Memoisation on both layers.** `forward_fn` memoises the graph (64-entry LRU on `(family, digest, seq_len, batch_size, pack_weights)`), which keeps `id(fwd)` stable so `etl.build_cached` returns the **same** `Executable` for an identical `build` call; weight identity is excluded because weights cannot change the traced program.
- **Static shapes are baked in**, not symbolic: symbolic reshape/reduce-with-dynamic-dims paths are the exporter's per-op limitation, and "one graph per `(config, seq_len, batch_size)`" is the documented, memoised contract — a mismatched length is an explicit error, never a recompile surprise.
- **In-graph RoPE and mask, not host tables.** `ops.rope_cos_sin` builds the cos/sin tables from `arange` + `power` + `cos`/`sin` (all StableHLO-exportable), keeping the graph signature at exactly `(params, input_ids)`, the emitted IR op count independent of `seq_len` (pinned by `tests/test_export.py`), and the MLIR free of inlined payload bytes. `ops.rope_tables` is oracle-only.
- **Flat canonical param dict** (`dict[str, ndarray]` / `dict[str, TensorSpec]`), not a nested pytree: keys map one-to-one onto HF keys (nearest-miss suggestions via `difflib`) and the graph reads them by explicit key strings.
- **The activation export gate runs before any tracing** (`_check_activation`): an erf-GELU config on a compiled backend fails immediately with `HFUnsupportedError` naming the activation, the backend and both remedies (`backend="numpy"`, or a `gelu_new` config — measured max |gelu_tanh − gelu_erf| = 4.7e-4 near |x| ≈ 2.7) instead of failing deep inside lowering; the numpy interpreter accepts every supported activation.
- **`run`'s id-dtype coercion is the only host-side type coercion.** Integral token ids are cast to `int32` (etl's run boundary is dtype-strict, ids are integral by definition); params are never filtered, renamed or cast, and an `etl.Tensor` of ids is forwarded verbatim — which is exactly what makes the device-resident recipe reachable.
- **`predict` is the only place a host transfer happens**, and it is one documented `Tensor.to(Device("cpu", 0))` on the *output* of a non-cpu executable; it refuses non-cpu params with the device-resident recipe rather than copying or dying inside `np.asarray`.
- **Device-resident is explicit on both sides**: `place_params` (weights) + `place_ids` (ids) then `run`; the run boundary rejects host inputs for a non-cpu executable by design.
- **Packed weights are an explicit, opt-in composition, never a silent default.** The limitation it removes is IREE's Python-runtime ~161-*consumed*-input cliff on llvm-cpu; bucketing by `(dtype, rank, shape[1:])` beats a hand-written per-role scheme (3 inputs for the 124M `gpt2` instead of ~12), costs no family code, and keeps every weight a graph input. Unpacked stays default because packing is not free (~475 MB of concatenated host arrays for the 124M model).
- **Pack once on the host, then place — never per call.** `pack_params` runs at build/params time and `place_params` then stages 3-5 leaves instead of ~200, so the same-device loop keeps its zero-host-round-trip property.
- **`opt_level` is per-target** — `"O3"` generally, `"O1"` when every effective IREE target is ROCm — because O2/O3 trip an upstream `iree-compile` failure on these LayerNorm-reduce shapes while O0/O1/default emit a byte-identical vmfb; encoding it as a default (rather than a workaround users must know) is what makes `build(..., backend="iree", device=Device("rocm", 0))` work for real-size checkpoints, and an explicit `opt_level`/`ETL_OPT_LEVEL` still wins.
- **Legacy pickles are a security boundary** (`weights._RestrictedUnpickler`): an explicit allowlist, every rejection naming the offending `module.name`.
- **HF layout knowledge lives only in the family modules** (`canonicalize_weights`): readers stay format-only, so a mapping bug is testable one layer at a time and a new family is one module plus one registry entry.
- **Two-formulation parity**: the graphs use etl ops, the oracle uses plain numpy in different shapes of code, so a shared bug would have to be written twice to pass.

## Constraints

- **No new hard dependency**; `torch`/`transformers` appear only in tests behind availability guards as external golden oracles, `iree` only behind `skipif` guards, and no `safetensors`/`huggingface_hub` package is required.
- **Never modify `./etl/`** (sibling node — read-only; escalate core gaps to the parent agent). Gaps are composed around here and documented: `rsqrt` → `1/sqrt`, `erf` → numpy-backend-only GELU.
- **Explicit errors only**: everything derives from `etl.core.ETLError` via `HFError` (or `ValueError`/`TypeError` for pure host-side argument errors), messages name the file/field/tensor/key, and there is never a silent fallback, dtype swap or positional guess.
- **Static-shape-first**: dynamic dims are not a goal; a mismatched `seq_len`/`batch_size` is an explicit `HFConfigError`/`etl.ShapeError`.
- **Files stay small** (~1000 lines baseline, one concern per module); no duplicate family logic — shared attention/norm/MLP pieces live in `ops.py` / `models/_common.py`. `pipeline.py` sits at ~1110 lines: one concern (the verbatim option matrix plus `build`/`run`/`predict`/placement), each function documented in place.
- **Tests are colocated** in `./etl_hf/tests/`; `pyproject.toml` `testpaths = ["tests", "etl_hf/tests"]` and its package list excludes `etl_hf.tests`.
- **Status (measured)**: **675 collected — 647 passed / 28 skipped** in the base env (`uv run --with pytest python -m pytest etl_hf/tests`), **666 passed / 9 skipped** with `--extra iree`, **672 passed / 3 skipped** with the extra plus `ETL_HF_BIG=1`, and `tests/test_iree_rocm.py` **11/11 passed** on the real AMD MI300X — zero failures in every leg. Skips are host/dependency guards only: torch/transformers absent (3), `ETL_HF_BIG` unset (3+), `iree` absent (19 in the base env).
- **Status (capability)**: both families trace, verify, export to StableHLO and run on the numpy interpreter, and the same graph compiles and runs on **every validated compiler adapter** — one IREE program on `llvm-cpu` and — packed where needed — device-resident on ROCm, plus device-resident **XLA/ROCm** via a user-provided PJRT plugin. Adapter coverage for the whole forward: `numpy` (host), `iree` (llvm-cpu + ROCm), `xla` (ROCm — the first `etl_hf` runs on XLA anywhere, measured by the benchmark harness, not a gated test); **`tvm` is UNVALIDATED for `etl_hf`** (no whole-model run has been recorded on the TVM adapter — its option handling is unit-tested only).
  Measured best-of-5 warm, llvm-cpu: `tiny_llama` `seq_len=32` build 1.5 s / 3.3 ms per call (1.0e-7 vs numpy), `tiny_gpt2` 1.1 s / 3.7 ms (2.2e-8), `distilgpt2` 1.9 s / 346 ms (2.1e-4) — against 6.0 / 1.9 / 12.1 ms for the numpy interpreter, whose OpenBLAS matmuls win on CPU.
  On ROCm/MI300X (`Device("rocm", 0)`, `target_backends=["rocm"]`): `tiny_llama` build 1.15 s / 0.21 ms warm (5.2× the interpreter), `tiny_gpt2` 0.99 s / 0.25 ms (5.0×, bit-exact), `distilgpt2` packed 3.99 s / 1.12 ms (12.4×), 124M `gpt2` packed 7.22 s / 2.75 ms (~13×), 2.4e-6 relative-to-absmax, 10/10 bit-identical runs.
  Parity against the real HuggingFace implementation (`tests/test_torch_oracle.py`): `tiny_llama` 8.9e-8, `tiny_mistral` 1.8e-7, `sshleifer/tiny-gpt2` 2.2e-8 max-abs at `seq_len=8`, argmax identical at every non-tie position; the torch-free golden cases over `../bench_logs/hf_goldens/` are `tiny_gpt2` bit-exact, `tiny_random_llama` 8.9e-8, `tiny_random_mistral` 1.2e-7, `distilgpt2` 1.07e-4, `gpt2` 1.53e-4 (1.1e-6 of a 138.7 absmax), with strict golden→checkpoint resolution.
- **Status (vs PyTorch, `../bench_logs/hf_bench_results.md`)**: measured on an AMD Instinct MI300X VF (gfx942, ROCm 7.1) over five checkpoints (`tiny_llama`, `tiny_mistral`, `tiny_gpt2`, `distilgpt2`, `gpt2` 124M), `seq_len` 32 plus 128 for the two real ones, batch 1, the whole forward as ONE graph in the device-resident same-device loop (`asdevicearray` = 0). **`xla`/rocm beats torch-rocm in 6/6 measured cells, 7.18×–9.99×** (21.6×–35.6× vs torch-cpu on the real models) and **`iree`/rocm in 5/6, 0.73×–9.13×** — the single loss is `gpt2` at `seq_len=128` (7.276 ms vs 5.286). XLA's default fp32 `dot` is TF32-class (rel-to-absmax 1.5e-3–2.6e-3, argmax as low as 0.969 on `gpt2`), so the opt-in `dot_precision="HIGHEST"` leg is the bit-accurate one (rel-to-absmax 1.5e-6–2.9e-6, argmax agreement 1.0, for 3%–29% more latency). On CPU the numpy/OpenBLAS interpreter still wins the large matmuls (`gpt2` s32: 28.9 ms interpreted vs 403 ms on llvm-cpu). Legs that cannot run fail explicitly, never silently: `tiny_mistral` on `iree`/`xla` (its `gelu` needs `etl.erf`, a StableHLO export v1 deferral) and `distilgpt2`/`gpt2` at `seq_len=128` on iree-llvm-cpu (upstream `iree-compile` stack-allocation limit); there is no XLA CPU plugin on that host, so the xla legs are ROCm-only. Harness: `../bench_logs/hf_bench.py` (+ `hf_bench_env.py`, records in `hf_bench_data/`).

## Known Issues

- **Exact-erf GELU is numpy-backend-only**: `hidden_act: "gelu"`/`"erf"` builds `ops.gelu_erf` (which uses `etl.erf`, a v1 StableHLO export deferral), so the graph runs on the numpy interpreter and raises an explicit `etl.BackendError` on any compiler backend — with `HFUnsupportedError` from the activation gate even earlier, and never a silent tanh substitute.
- **Static shapes only** — one graph per `(config, seq_len, batch_size)`; `seq_len > max_position_embeddings` is `HFConfigError` and a mismatched `input_ids` shape is `etl.ShapeError` telling you to build another length rather than reshape.
- **The llama family is bias-free**: any `*.bias` key raises `HFUnsupportedError` (v1 covers `attention_bias=False`/`mlp_bias=False` models only).
- **Per-family key allowlists**: llama ignores `model.rotary_emb.inv_freq`; GPT-2 ignores `h.{i}.attn.bias` / `h.{i}.attn.masked_bias`. Everything else must map onto the canonical scheme or it is an `HFWeightError` listing the offending keys.
- **`rope_scaling` is unsupported** (`HFUnsupportedError` unless `None`/`{}`/`[]`).
- **No KV cache, no incremental decode, no sampling, no training**: every call re-runs the whole graph over `seq_len` tokens.
- **`load_config`/`load_hf` require a directory**: `<dir>/config.json` is read before any weights, so a single checkpoint *file* path fails with `HFConfigError` naming the missing `config.json` (single-file reading is `weights.load_checkpoint`'s job).
- **numpy-on-CPU vs the compiled graph**: on CPU the interpreter can beat the compiled graph for large matmuls (OpenBLAS) while the compiled path's value is GPU — report both, never conflate them. The opposite holds on this host's GPU (5× tiny, 12-14× `distilgpt2`, ~13× the 124M `gpt2`), and `distilgpt2` on llvm-cpu is 266 ms compiled vs 14.8 ms interpreted.
- **Device-resident run costs**: the FIRST `place_params` leaf in a process pays a one-time HIP driver init (~0.08-0.29 s), after which a leaf costs ~6e-5 s (tiny) to ~0.10 s (124M-sized); re-placing per call instead of keeping the placed tree costs ~9× the same-device loop (2.04 vs 0.222 ms/call for `tiny_llama`). The first `.to(Device("cpu", 0))` readback pays ~1-17 ms once per process, then ~0.75-1.3 ms per 2 MiB; the first call after build is only 1.1-3× warm.
- **Packing is not free, just cheap**: the packed 124M `gpt2` tree is ~475 MB of fresh concatenated host arrays (peak RSS ~1.7 GB on the packed llvm-cpu path vs ~1.1 GB unpacked-numpy, since canonical and packed trees briefly coexist) and the interpreter pays ~0-2% extra latency for the in-graph slices (logits stay bit-identical). Its payoff is the input count (197 → 4) and the staging leaf count (196 → 3).
- **ROCm `opt_level` O2/O3 fails to compile gpt2-class graphs** (upstream `'func.func' op failed to distribute` on the LayerNorm mean reduce; every probed hidden × heads combination compiles at O1 with a byte-identical vmfb to O0/default). The `"O1"` default for all-ROCm targets is what keeps real-size builds working; an explicit `opt_level`/`ETL_OPT_LEVEL` overrides it and owns the risk.
- **ROCm fp32 `dot` is true fp32**: `dot_precision="HIGHEST"` (needed against xla/ROCm's TF32 default) is a no-op on iree/ROCm — pinned by bit-identical logits. ROCm parity is fp32 accumulation-order noise (`tiny_gpt2` bit-exact, `tiny_llama` 8.9e-8, real-size GPT-2 ~2-3e-6 **relative to absmax**), so real-size comparisons must be relative-to-absmax, never element-wise.
- **The IREE Python-runtime ~161-consumed-input `_invoke` cliff exists on llvm-cpu only — and `pack_weights=True` removes it**: a graph that **consumes** more than ~161 distinct inputs compiles, loads and then SIGSEGVs at the first run (exit 139, no traceback, `iree/runtime/function.py` `_invoke`). The trigger is input *consumption* (not declared-argument count, op/dispatch count, tensor size or parameter count; bracketed at 161 for a chained-matmul graph and 162 for a fan-out-reduce one; neither `opt_level="O0"` nor dropping `--iree-opt-data-tiling` helps). The 124M `gpt2` (`seq_len=32`, 197 inputs) traces, lowers, compiles (870 IR ops) and loads, then crashes — truncated to 9 layers (149 inputs) it runs while 10 layers (165) already crashes; `pack_weights=True` brings it to 3 weight inputs + ids = 4 and it runs (~405 ms/call median on llvm-cpu, 13 consecutive bit-identical runs, 2.5e-6 relative vs the interpreter). **ROCm has no such cliff** (the unpacked 197-input graph runs at 2.19 ms/call at `seq_len=16`, bit-exact), so packing is optional there and its value is the 196 → 3 staging-leaf reduction. `distilgpt2` (101 inputs) is unaffected at `max_abs` 3.2e-4 (2.8e-6 of an 112.5 absmax), argmax identical, 230 ms/call on llvm-cpu.
- **A tied head is one array, so a tied tree is the model's true parameter count**: a tied checkpoint that ships its `lm_head.weight` copy is verified against the embedding and dropped, never kept as a second graph input. Retained RSS for `gpt2` at `seq_len=32` is the canonical tree itself (475 MiB), while the *peak* is ~1.06 GiB because the raw safetensors dict and the canonical tree coexist during canonicalisation (`load_params` drops the raw dict afterwards).
- **Vulkan (`device=Device("vulkan", 0)` → `iree` + `vulkan-spirv`) is compile-limited on real-size models**: IREE's default Vulkan target profile `vp_android_baseline_2022` allows only **16384 bytes** of workgroup shared memory, so at the default `opt_level="O3"` the 124M-class graphs fail to compile (`function 'main_dispatch_0_reduction_32x768_f32' uses 196864 bytes of shared memory; exceeded the limit of 16384 bytes`, and `failed to legalize operation 'vector.from_elements'` at `seq_len=128`) — the measured remedy is an explicit `opt_level="O1"`, and a larger-profile `--iree-vulkan-target=<profile>` in `iree_compile_args` lifts the shared-memory limit but not the later vector-lowering failure at `O3`. Tiny checkpoints compile with the defaults. Both knobs are explicit caller options — etl_hf injects neither, and nothing else about `build` changes for this kind. Host/driver reality and the measured software-Vulkan (lavapipe) matrix: [`../bench_logs/vulkan_findings.md`](../bench_logs/vulkan_findings.md).
- **`etl` ships no `rsqrt`, top-level `where`, `softmax`, `layer_norm`, `rms_norm`, `silu` or `iota`**, and its `slice` takes static start/limit tuples (no data-dependent slice): all composed in `ops.py`.
- **Legacy `pytorch_model.bin` layout**: the legacy format writes an **int64 element count before each storage's bytes** (after the pickled state dict and the storage-key list) and stores storages in trailer key order; consuming `numel × itemsize` bytes per storage **without** that prefix shifts every storage, so `weights._read_torch_legacy` must read the count first (verified on a real legacy file: trailing bytes exceed the summed storage sizes by exactly 8 per storage). A shifted parse still yields correctly-shaped, plausible tensors and graph-vs-`reference` parity cannot detect it (both sides read the same arrays) — validate a legacy `.bin` against an external oracle or assert the reader consumes the file exactly. `sshleifer/tiny-gpt2` has ordinary logits (`max |logit| ≈ 0.1285` with the prefix), so the `~1e-44` "degenerate checkpoint" figure in `../bench_logs/hf_feasibility_findings.md` describes a shifted parse, not the checkpoint (that file is a read-only sibling; this is a fact about the layout, not a change log).

## Routing Table

| Path / module | Area |
|---|---|
| `./etl_hf/load.py` | Loading half: `LoadedModel`, `load_config`, `load_params`, `load_hf`, `validate_params`. |
| `./etl_hf/program.py` | Compute half: `forward_fn` (+ memoisation/specialisation stamp), `param_specs`, `reference_logits`, `resolve_seq_len`, `DEFAULT_SEQ_LEN`. |
| `./etl_hf/pipeline.py` | Staging: `build`, `run`, `predict`, `place_params`, `place_ids`, `resolve_build_options`, the `DEFAULT_*` option constants and the default backend/option matrix. |
| `./etl_hf/config.py` | HF `config.json` → `ModelConfig`, `digest`, `FAMILY_BY_MODEL_TYPE`. |
| `./etl_hf/weights.py` | Checkpoint reading: safetensors, torch zip/legacy, sharded indices, dtype policy, restricted unpickler (no key mapping). |
| `./etl_hf/ops.py` | Graph idioms (norms/activations/softmax/mask/RoPE/head layout) + `ACTIVATIONS`/`EXPORT_UNSAFE_ACTIVATIONS`. |
| `./etl_hf/reference.py` | Independent pure-numpy oracle primitives (no `etl` import) — the thing graphs are validated against. |
| `./etl_hf/families.py` | `FamilySpec`, the `FAMILIES` registry, `get_family`, `family_for_model_type`, `resolve_family`. |
| `./etl_hf/models/` | Per-family graph builders (`llama.py`, `gpt2.py`) plus the shared block assembly `_common.py`; `models/__init__.py` re-exports nothing (import acyclicity). |
| `./etl_hf/pack.py` | Opt-in packed weights: `PackPlan`/`pack_plan`/`pack_params`/`is_packed_tree` — bucket the canonical `param_specs` by `(dtype, rank, shape[1:])`, concatenate along axis 0 on the host, unpack inside the graph with static `etl.slice`. |
| `./etl_hf/errors.py` | The `HFError` hierarchy (rooted at `etl.core.ETLError`). |
| `./etl_hf/_validate.py` | Internal argument guards shared by `load`/`program`/`pipeline`. |
| `./etl_hf/tests/` | pytest suite (parity, export, iree/llvm-cpu + iree/ROCm device-resident gates, packed-weight pins, real-checkpoint gates, and the functional-API design pins in `test_public_api.py`); `tests/hf_download.py` is the guarded checkpoint downloader used by the network-dependent tests. |

Cross-references (read-only): `../etl/` → the tensor language this package builds on (public API contract in `../etl/CONTEXT.md`); `../bench_logs/` → the HF feasibility spike, probe scripts and measured results — notably the current HF-vs-PyTorch benchmark harness `../bench_logs/hf_bench.py` (+ `hf_bench_env.py`, records in `../bench_logs/hf_bench_data/`) and its report `../bench_logs/hf_bench_results.md`, the older `../bench_logs/hf_perf_results.md`, and `../bench_logs/hf_goldens/`, the torch-golden fixtures read by `tests/test_torch_oracle.py`.

## Notes for Agents

- **The functional surface is the contract.** No re-introduction of a loaded-model class with methods, no `HFModel`, no `build_model`/`run_model`, and no `seq_len` on the loaded record: config data (`load_*`), compute function (`forward_fn`), and staging (`build`/`run`/`predict`) stay separate layers.
- The StableHLO exporter defers `erf`, `gelu`, `tril`, `triu`, `cumsum`, `cumprod`, `flip`, `roll` (all verified to raise `BackendError`) — do not put them in a compiled graph; compose from comparisons/`select`/arithmetic instead, as `ops.py` does.
- `etl.dot` does not auto-cast mixed dtypes (unlike elementwise ops): pass weights of the same dtype as the activations (every `etl_hf` spec is float32).
- Never `etl.constant` a weight — it inlines the payload into the emitted MLIR text; weights are graph inputs (`param_specs`) by contract.
- Export needs an explicit trace first: `fwd = forward_fn(config, seq_len=s)` → `etl.trace(fwd, param_specs(config), etl.TensorSpec((b, s), etl.int32))` → `graph.verify()` → `stablehlo.export(graph)` (`stablehlo.export` on a `Defn` raises `TypeError`).
- `param_specs` is a **function** (`param_specs(config)`), never a mapping.
- `pytest` is not in the base uv env: run `uv run --with pytest python -m pytest etl_hf/tests` (add `--extra iree` for the compiled path). The `iree` adapter's nanobind teardown can swallow the `-q` summary line — always count with `--junit-xml`.
- `uv run --extra iree` **mutates the shared venv**: after such a run, `uv sync` restores the documented base env (otherwise the "iree-absent" skips silently disappear).
- Use `$TMPDIR` for checkpoints, never `/tmp` (`hf_download` defaults to `$TMPDIR/etl_hf_models`, overridable with `$ETL_HF_MODELS_DIR`); real-size checkpoints are opt-in with `ETL_HF_BIG=1`, which also satisfies `require_model` for locally cached checkpoints offline.
- Packed weights are opt-in and generic: `build(forward_fn(cfg, pack_weights=True), cfg, pack_weights=True)` + `pack_params(pack_plan(param_specs(cfg)), params)` (or `predict(..., pack_weights=True)` which packs a canonical tree once on the host); pack on the **host first**, then `place_params`, then `run`. Never re-concatenate per call, and never assume a fixed packed count — read it from `pack_plan(...).specs`.
- GPU work needs `uv run --isolated --extra iree …` (the base env has no iree, so `etl.devices()` shows only cpu); the ROCm device is `Device("rocm", 0)` (HAL `device_id` is 1-based), the adapter auto-detects the `gfxNNN` target, and the ROCm cases are GPU-guarded so the suite stays green on a CPU-only host. Run the ROCm module with `PYTHONUNBUFFERED=1`.
- Real-size (`gpt2`/`distilgpt2`) work is memory-hungry (packed 124M ≈ 1.7 GB peak, canonical tree ≈ 475 MB): guard such cases on available memory and keep GPU runs short (this host's GPU is often shared).
