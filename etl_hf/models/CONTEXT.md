# etl_hf/models — per-family graph builders

## Intent

One module per HuggingFace model family: each module owns the HF-key → canonical-weight map, the `etl.TensorSpec` tree, the whole-model `@etl.defn` forward graph, and an independent pure-numpy reference forward used as the parity oracle.
Every family module exposes the same pinned four-entry interface — `param_specs(cfg)`, `graph(cfg, seq_len, batch_size=1, *, pack_plan=None)`, `reference(cfg, params, input_ids)`, `canonicalize_weights(cfg, raw)` — plus `FAMILY` (the `etl_hf` family key wired into `ModelConfig.family`) and, when the architecture covers several HF `model_type`s, a `MODEL_TYPES` tuple.
`./etl_hf/models/__init__.py` imports and re-exports nothing (`__all__ = []`); family modules are reached by qualified path (`etl_hf.models.llama`, `etl_hf.models.gpt2`), and their imports stay in-layer: `etl`, `etl_hf.ops` (graph idioms), `etl_hf.models._common` (block wiring), `etl_hf.reference` (oracle primitives), `etl_hf.config` (`ModelConfig`), `etl_hf.errors` — never `etl_hf.families`/`program`/`load`/`pipeline`.
The call direction runs one-way into this node: `etl_hf.families` imports both family modules to build the `FAMILIES` registry, and the resulting `FamilySpec` is what `etl_hf.program` (`forward_fn`/`param_specs`/`reference_logits`) and `etl_hf.load` (`load_params` → `canonicalize_weights`) call — no consumer is imported from here, and all shapes are static (`seq_len`/`batch_size` are graph specialisation values), all weights are graph **inputs** (never `etl.constant`), and every canonicalisation/validation failure is an explicit `HFConfigError` / `HFWeightError` / `HFUnsupportedError`.

## Shared assembly (`_common.py`)

`linear(x, weight, bias=None)` is `etl.dot(x, weight)` plus an optional broadcast bias: `x` is `(..., in_features)`, `weight` is `(in_features, out_features)`, `bias` is `(out_features,)`, result `(..., out_features)` in `x`'s dtype (no mixed-dtype cast — `etl.dot` does not auto-cast).
`attention(q, k, v, mask, scale)` takes rank-4 head tensors `q` `(b, n_heads, s, head_dim)` and `k`/`v` `(b, n_heads, s_kv, head_dim)` **already repeated** by `etl_hf.ops.repeat_kv`, and returns the fused `(b, s, n_heads·head_dim)` layout the output projection consumes: `scores = dot(q, transpose(k, (0,1,3,2))) · scale (+ mask)` → `ops.softmax` → `dot(probs, v)` → `ops.merge_heads`; `mask` is a plain additive array broadcastable to `(b, n_heads, s, s_kv)` (normally the `(s, s)` table from `ops.causal_mask`, `None` = unmasked) and `scale` is conventionally `head_dim ** -0.5`.
GQA/MQA stays outside that helper so `repeat_kv` is an explicit, compiler-visible op rather than a hidden reshape; the helper raises `HFUnsupportedError` for non-rank-4 operands, differing `k`/`v` shapes, mismatched `(batch, heads)`/`head_dim`, or a caller who forgot `repeat_kv`.
`gated_mlp(x, w_gate, w_up, w_down, act)` is the bias-free SwiGLU MLP `act(dot(x, w_gate)) * dot(x, w_up)` then `dot(·, w_down)`; `two_layer_mlp(x, w1, b1, w2, b2, act)` is the GPT-2 MLP `linear(act(linear(x, w1, b1)), w2, b2)` (`None` biases allowed), with `w1`/gate/up `(hidden, intermediate)` and `w2`/down `(intermediate, hidden)`.
All four are `SymbolicTensor` graph builders (assembly only, no numerics of their own): concrete tensors surface as the underlying `etl.TraceError`, extents come from static shapes (a symbolic dim raises `HFUnsupportedError` naming the axis in `etl_hf.ops`), and `act` is always a builder from `etl_hf.ops.resolve_activation` (`silu`, `gelu_tanh`, `gelu_erf`, `relu`), resolved eagerly by each family's `graph()` so an unknown name fails before tracing.

## Canonical weight keys

Canonical names are flat `dict[str, np.ndarray]` keys (`params` is never nested), shared verbatim between `param_specs` (specs) and `canonicalize_weights` (loaded arrays).

**llama** (`etl_hf/models/llama.py`, `9·L + 3` keys = `2 + 9·L + 1` untied, `9·L + 2` tied — the head is `embed.weight`):

| Canonical key | Shape | HF source (transposed at load where marked) |
|---|---|---|
| `embed.weight` | `(vocab, hidden)` | `model.embed_tokens.weight` (verbatim) |
| `layers.{i}.attn_norm.weight` | `(hidden,)` | `model.layers.{i}.input_layernorm.weight` |
| `layers.{i}.attn.q.weight` | `(hidden, n_heads·head_dim)` | `...self_attn.q_proj.weight`, transposed |
| `layers.{i}.attn.k.weight` | `(hidden, n_kv_heads·head_dim)` | `...self_attn.k_proj.weight`, transposed |
| `layers.{i}.attn.v.weight` | `(hidden, n_kv_heads·head_dim)` | `...self_attn.v_proj.weight`, transposed |
| `layers.{i}.attn.o.weight` | `(n_heads·head_dim, hidden)` | `...self_attn.o_proj.weight`, transposed |
| `layers.{i}.mlp_norm.weight` | `(hidden,)` | `...post_attention_layernorm.weight` |
| `layers.{i}.mlp.gate.weight` | `(hidden, intermediate)` | `...mlp.gate_proj.weight`, transposed |
| `layers.{i}.mlp.up.weight` | `(hidden, intermediate)` | `...mlp.up_proj.weight`, transposed |
| `layers.{i}.mlp.down.weight` | `(intermediate, hidden)` | `...mlp.down_proj.weight`, transposed |
| `final_norm.weight` | `(hidden,)` | `model.norm.weight` |
| `lm_head.weight` | `(hidden, vocab)` | `lm_head.weight`, transposed — **untied only** (a tied config has no such key) |

The llama builds are bias-free: any `*.bias` key raises `HFUnsupportedError` (naming the keys and the `attention_bias=False` / `mlp_bias=False` remedy), and `model.rotary_emb.inv_freq` is the one deliberately ignored key (a stale buffer; the forward recomputes frequencies).

**gpt2** (`etl_hf/models/gpt2.py`, `16·L + 5` keys = `2 + 16·L + 3` untied, `16·L + 4` tied — the head is `embed.weight`):

| Canonical key | Shape | HF source (base key, `transformer.` prefix stripped) |
|---|---|---|
| `embed.weight` | `(vocab, hidden)` | `wte.weight` |
| `pos_embed.weight` | `(max_position_embeddings, hidden)` | `wpe.weight` |
| `layers.{i}.attn_norm.{weight,bias}` | `(hidden,)` | `h.{i}.ln_1.{weight,bias}` |
| `layers.{i}.attn.{q,k,v}.{weight,bias}` | `(hidden, hidden)` / `(hidden,)` | `h.{i}.attn.c_attn.{weight,bias}` split at load |
| `layers.{i}.attn.o.{weight,bias}` | `(hidden, hidden)` / `(hidden,)` | `h.{i}.attn.c_proj.{weight,bias}` |
| `layers.{i}.mlp_norm.{weight,bias}` | `(hidden,)` | `h.{i}.ln_2.{weight,bias}` |
| `layers.{i}.mlp.fc1.{weight,bias}` | `(hidden, intermediate)` / `(intermediate,)` | `h.{i}.mlp.c_fc.{weight,bias}` |
| `layers.{i}.mlp.fc2.{weight,bias}` | `(intermediate, hidden)` / `(hidden,)` | `h.{i}.mlp.c_proj.{weight,bias}` |
| `final_norm.{weight,bias}` | `(hidden,)` | `ln_f.{weight,bias}` |
| `lm_head.weight` | `(hidden, vocab)` | `lm_head.weight` transposed — **untied only** (a tied config has no such key) |

The fused `h.{i}.attn.c_attn` `(hidden, 3·hidden)` tensor and its `(3·hidden,)` bias are split **at load time** into the `q`, `k`, `v` column blocks `[0:H]`, `[H:2H]`, `[2H:3H]` (HF's `query, key, value` order), so the graph never slices a fused matrix.
Both `transformer.`-prefixed and bare key spellings are accepted (mixing allowed); two keys normalising onto the same canonical slot raise `HFWeightError` instead of last-one-wins, and the derived causal buffers `h.{i}.attn.bias` / `h.{i}.attn.masked_bias` are the only ignored keys.

## Orientation rule

Every `*.weight` matrix fed to `etl.dot(x, W)` — i.e. every `linear` operand — is `(in_features, out_features)`; Llama/Mistral/Qwen2 linears are stored `(out, in)` by HF and are transposed at load (`np.ascontiguousarray(np.transpose(...))`), while GPT-2's `Conv1D` layers already compute `x @ W + b` with `(in, out)` weights and are **not** transposed (GPT-2's `lm_head` is an `nn.Linear` `(vocab, hidden)` and **is**).
Tied embeddings are **one array, not two**: with `tie_word_embeddings=True` the head IS `embed.weight`, so `param_specs`/`canonicalize_weights` emit no `lm_head.weight` key and the graph feeds the one embedding input to the token gather and (as `etl.transpose(embed)`) to the final projection.
A tied checkpoint that ships its own `lm_head.weight` must agree with the embedding — it is verified, consumed and dropped; a disagreement raises `HFWeightError` (never silently picks one), and an untied config with no `lm_head` raises `HFWeightError`.
Every canonicalised array is `float32`, C-contiguous and writable (`int`/`float` inputs accepted, `bool`/complex rejected).

## Graph conventions

`graph(cfg, seq_len, batch_size=1, *, pack_plan=None)` returns a fresh `@etl.defn` callable each call with the positional signature `(params, input_ids) -> logits`; `input_ids` is `(batch_size, seq_len)` `etl.int32` and `logits` is `(batch_size, seq_len, vocab_size)` `float32`.
The optional `pack_plan` (an `etl_hf.pack.PackPlan`) is the ONE hook the opt-in packed-weights mode needs: when it is not `None` the forward's first statement is `params = pack_plan.unpack(params)`, mapping the few packed graph inputs back onto the canonical keys with static `etl.slice`s, and the rest of the body is unchanged (identical code in both families — never a second forward implementation).
Weights are always read from `params` by their canonical flat key — never `etl.constant`, so the exported program's text stays O(1) in the weight sizes.
Layout is pre-LN residual — `x += attn(norm(x))`, then `h = norm(x)`, `x += mlp(h)`, then a final norm and the `lm_head` projection — and normalisation is family-specific: llama uses RMSNorm (`etl_hf.ops.rms_norm`, no mean subtraction, no bias, `layer_norm_eps` = HF `rms_norm_eps`), GPT-2 uses HF `nn.LayerNorm` (`etl_hf.ops.layer_norm`, **biased**/divide-by-N variance plus eps inside the sqrt, affine weight and bias after).
Positions: llama computes RoPE cos/sin **in-graph** (`etl_hf.ops.rope_cos_sin` from `arange` + `power` + `cos`/`sin`) and applies HF `rotate_half` via `etl_hf.ops.apply_rope` to `q` and `k` only; GPT-2 uses a learned position table (`etl.arange` → `etl.gather(pos_embed, ...)` → `etl.broadcast` over the batch axis) added to the token embeddings.
Masking: both families apply an **additive causal mask** — `0.0` on and below the diagonal (`j <= i`, a query attends to itself and the past) and `MASK_NEG` (`-1e4`, i.e. `etl_hf.ops.MASK_NEG` / `etl_hf.reference.MASK_NEG`) strictly above it, added to the scaled scores before the softmax so masked-out weights are exactly `0.0` in fp32; the mask is built from `arange`/`less_equal`/`cast` (`etl_hf.ops.causal_mask`), never a constant payload.
Head layout: `etl_hf.ops.split_heads` `(b, s, n·hd) → (b, n, s, hd)` and `merge_heads` back; GQA via `etl_hf.ops.repeat_kv` with `n_rep = num_attention_heads // num_key_value_heads` (head `i` serves query heads `i·n_rep … i·n_rep + n_rep − 1`, i.e. `np.repeat`, never interleaved; `n_rep == 1` is the identity).
`seq_len` is bounded by `cfg.max_position_embeddings` and rejected beyond it (`HFConfigError`): RoPE has no positions past it and GPT-2's learned table cannot extrapolate.
The `reference` functions compose **only** `etl_hf.reference` primitives (no `etl` call on that path) in a deliberately different formulation — per-head Python loops, hand-written mean/variance formulas, `math.erf`, plain `matmul` in the canonical `(in, out)` orientation — so a wrong transpose or head split shows up as a numeric difference rather than a shared bug.

## Per-family notes

**llama** (`FAMILY = "llama"`, `MODEL_TYPES = ("llama", "mistral", "qwen2")`) covers every Llama-shaped checkpoint: RMSNorm, RoPE, GQA/MQA and a SwiGLU MLP.
The config fields that matter here: `num_key_value_heads` (defaults to `num_attention_heads`), `rope_theta` (default `10000.0`), `tie_word_embeddings` (default `False`), `layer_norm_eps` (HF `rms_norm_eps`) and an optional explicit `head_dim` override, honoured verbatim — q/k/v/o shapes derive from `num_attention_heads · head_dim` / `num_key_value_heads · head_dim`, never from an assumed `hidden_size // num_attention_heads`, and RoPE requires `head_dim` to be even.
Any non-empty `rope_scaling` (YaRN / linear / dynamic NTK) is rejected at **config parse** time (`etl_hf/config.py`, `HFUnsupportedError`) because a scaled-RoPE model would silently diverge from HF.
An unexpected (unconsumed) checkpoint key, a shape that disagrees with the config, a missing required key (with difflib near-miss suggestions) and a foreign-family `ModelConfig` are all `HFWeightError` / `HFConfigError`.

**gpt2** (`FAMILY = "gpt2"`, registry `model_types` derived from `etl_hf.config.FAMILY_BY_MODEL_TYPE` = `("gpt2",)`, which is what `openai-community/gpt2` and `distilgpt2` both declare) covers LayerNorm + learned positions + `gelu_new` + tied head.
Its compact HF keys (`n_embd`/`n_head`/`n_layer`/`n_positions`/`n_vocab`/`n_inner`/`layer_norm_epsilon`/`activation_function`) are normalised in `etl_hf/config.py`, which owns that mapping and its defaults (`n_inner` `null` → `4·n_embd`, eps `1e-5`, `hidden_act` `gelu_new`, `tie_word_embeddings` `True`).
GPT-2 has neither GQA nor MQA (`num_key_value_heads == num_attention_heads`), no RoPE (`rope_theta is None`) and no per-head override (`head_dim` always `hidden_size // num_attention_heads`).
`canonicalize_weights` ends with a completeness assertion against its own shape table (`_assert_complete`), so an internal key-set drift fails loudly instead of handing the tracer a surprise.

## Notes for Agents

The two `canonicalize_weights` implementations are the **only** place HF checkpoint key naming is known — llama's per-layer table is `_LAYER_RECORDS` (`(canonical suffix, HF sub-key, transpose?)`) plus three explicit `take` calls, gpt2's is the `_take` call sequence inside `canonicalize_weights`; never add HF key knowledge elsewhere.
`param_specs` is a **function** of the config (per-config shapes), never a constant mapping, and its keys/shapes are exactly what `canonicalize_weights` emits — `etl_hf/tests` pins that equality for both families.
`_expected_shapes` (llama) and `_canonical_shapes` (gpt2) are the single source of truth for `param_specs` **and** load-time shape validation; add a parameter by editing that table only.
Adding a family = one new module here + one `FamilySpec` entry in `etl_hf/families.py`; that registry enforces two import-time invariants — each `FamilySpec.name` equals the module's `FAMILY` and its `FAMILIES` key, and a declared `MODEL_TYPES` matches the `model_type`s that `etl_hf.config.FAMILY_BY_MODEL_TYPE` maps onto that family (declaring it is optional: llama declares it, gpt2 does not, and `__all__` is not the interface — gpt2's omits `FAMILY` while the registry reads the attribute directly).
Export-unsafe activations are gated before tracing: `graph()` resolves the activation eagerly (unknown name → `HFUnsupportedError`) and the compiled-path refusal for `gelu`/`erf` lives in `etl_hf.pipeline.build` (`_check_activation`) via `etl_hf.ops.activation_is_export_safe` (`EXPORT_UNSAFE_ACTIVATIONS`), never a silent tanh substitute.
Never `etl.constant` a weight here — the tests bound the inlined data instead (`test_graph_has_no_constant_payloads`: ≤ 64 constant elements, a `stablehlo.constant` ceiling and an O(`seq_len`) largest-dense-literal bound in the export suite, plus a `seq_len`-independent op count).
Checkpoint tensors are validated (presence, dtype, shape) before the graph ever sees them, so a mismatched checkpoint fails at load, not inside lowering; dict insertion order carries no graph meaning (`core.flatten` sorts dict keys), the canonical order only keeps specs and loaded arrays reported identically.

## Routing Table

| Path | Area |
|---|---|
| `./__init__.py` | Package marker only (`__all__ = []`) — no imports, no re-exports. |
| `./_common.py` | Shared block assembly: `linear`, `attention`, `gated_mlp`, `two_layer_mlp` (no family-specific code). |
| `./llama.py` | Llama/Mistral/Qwen2 family: RMSNorm + RoPE + GQA + SwiGLU; `9·L + 3` bias-free weights (`9·L + 2` when tied). |
| `./gpt2.py` | GPT-2 family: LayerNorm + learned positions + `gelu_new` + tied head; `16·L + 5` weights (`16·L + 4` when tied), fused QKV split at load. |
| `../families.py` | `FAMILIES`/`get_family`/`family_for_model_type`/`resolve_family` — imports both family modules and wraps their four entry points in `FamilySpec`s. |
| `../program.py` / `../load.py` | `forward_fn`/`param_specs`/`reference_logits`/`resolve_seq_len` and `load_config`/`load_params`/`load_hf`/`validate_params` — call `graph`/`param_specs`/`reference`/`canonicalize_weights` through the family spec. |
| `../pipeline.py` | `build`/`run`/`predict`/`place_params`/`place_ids` + the resolved default-option matrix; owns the pre-trace activation export gate. |
| `../tests/` | Family suites: `test_llama.py`/`test_gpt2.py` (canonicalisation, parity, no-constant pins), `test_families.py`, `test_export.py`, `test_pack.py`, `test_public_api.py`. |
| `../../etl/` | The tensor library whose public ops and `@etl.defn` this package composes (read-only from here). |
