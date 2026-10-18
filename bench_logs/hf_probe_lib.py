"""hf_probe_lib — tiny HF transformer forward passes as ONE etl graph, plus a
pure-numpy reference. Scratch probe support code (bench_logs is throwaway).

Two families, both at TINY scale, static shapes only:

* **Llama family** (``LlamaForCausalLM`` / ``MistralForCausalLM`` configs):
  RMSNorm, RoPE computed IN-GRAPH from ``arange``+``power``+``cos``/``sin``,
  GQA-correct multi-head causal attention with explicit batched ``etl.dot``,
  SwiGLU (or GELU) MLP, optional tied lm_head.
* **GPT-2** (``GPT2LMHeadModel``): LayerNorm, learned positional embeddings,
  ``gelu_new`` (the tanh approximation GPT-2 itself uses — NO erf needed),
  Conv1D-style weights (already ``(in, out)``, no transpose), tied lm_head.

Weights are ordinary graph INPUTS (a nested dict/list pytree of ``TensorSpec``)
— never ``etl.constant`` embeddings — so the same graph runs against any
weights and the graph stays small.

Every etl op used here is StableHLO-exportable in v1 *except* exact erf-GELU
(``erf``/``gelu`` are documented v1 export deferrals) — see
``HIDDEN_ACT`` below and the findings note.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import etl
from probe_hf_download import read_safetensors
from probe_hf_torchbin import read_torch_bin

# The additive causal-mask constant. Any value <= -104 makes exp() underflow to
# exactly 0 in f32, so the choice is numerically irrelevant — HF's GPT-2 uses
# -1e4 (checkpoint buffer `attn.masked_bias`), HF's Llama uses finfo.min.
MASK_NEG = -1e4


# ---------------------------------------------------------------------------
# checkpoint loading (no hub libs: config.json + safetensors / torch .bin)
# ---------------------------------------------------------------------------
def load_checkpoint(model_dir: str | Path) -> tuple[dict, dict[str, np.ndarray], str]:
    """Read ``config.json`` + weights from a local model dir. Returns (cfg, sd, fmt)."""
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text())
    st = d / "model.safetensors"
    if st.exists():
        return cfg, read_safetensors(st), "safetensors"
    for name in ("pytorch_model.bin", "pytorch_model.bin.index.json"):
        cand = d / name
        if cand.exists() and cand.suffix == ".bin":
            return cfg, read_torch_bin(cand), "torch-bin"
    raise FileNotFoundError(f"{d}: no model.safetensors / pytorch_model.bin")


# ---------------------------------------------------------------------------
# weight-tree builders: HF state dict -> compact graph-input pytree
# ---------------------------------------------------------------------------
def llama_params(cfg: dict, sd: dict[str, np.ndarray]) -> dict:
    """HF Llama/Mistral state dict -> nested param pytree (linear weights as (in, out))."""
    layers = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        layers.append(
            {
                "ln1": sd[p + "input_layernorm.weight"].astype(np.float32),
                "wq": np.ascontiguousarray(sd[p + "self_attn.q_proj.weight"].T.astype(np.float32)),
                "wk": np.ascontiguousarray(sd[p + "self_attn.k_proj.weight"].T.astype(np.float32)),
                "wv": np.ascontiguousarray(sd[p + "self_attn.v_proj.weight"].T.astype(np.float32)),
                "wo": np.ascontiguousarray(sd[p + "self_attn.o_proj.weight"].T.astype(np.float32)),
                "ln2": sd[p + "post_attention_layernorm.weight"].astype(np.float32),
                "wgate": np.ascontiguousarray(sd[p + "mlp.gate_proj.weight"].T.astype(np.float32)),
                "wup": np.ascontiguousarray(sd[p + "mlp.up_proj.weight"].T.astype(np.float32)),
                "wdown": np.ascontiguousarray(sd[p + "mlp.down_proj.weight"].T.astype(np.float32)),
            }
        )
    head = sd.get("lm_head.weight")
    if head is None or cfg.get("tie_word_embeddings", False):
        lm = sd["model.embed_tokens.weight"].T  # tied
    else:
        lm = head.T
    return {
        "embed": sd["model.embed_tokens.weight"].astype(np.float32),
        "layers": layers,
        "norm": sd["model.norm.weight"].astype(np.float32),
        "lm_head": np.ascontiguousarray(lm.astype(np.float32)),
    }


def _normalize_gpt2_keys(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Canonicalize GPT-2 state-dict keys to the ``transformer.``-prefixed form.

    Older HF exports (sshleifer/tiny-gpt2, distilgpt2) prefix every key with
    ``transformer.``; the ``gpt2`` repo's ``model.safetensors`` drops that prefix
    for the block/embedding keys (``h.0.attn.c_attn.weight``, ``wte.weight``).
    """
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        out[k if k.startswith("transformer.") else f"transformer.{k}"] = v
    return out


def gpt2_params(cfg: dict, sd: dict[str, np.ndarray]) -> dict:
    """HF GPT-2 state dict -> nested param pytree (Conv1D weights kept as (in, out))."""
    sd = _normalize_gpt2_keys(sd)
    blocks = []
    for i in range(cfg["n_layer"]):
        p = f"transformer.h.{i}."
        blocks.append(
            {
                "ln1_w": sd[p + "ln_1.weight"].astype(np.float32),
                "ln1_b": sd[p + "ln_1.bias"].astype(np.float32),
                "attn_w": sd[p + "attn.c_attn.weight"].astype(np.float32),
                "attn_b": sd[p + "attn.c_attn.bias"].astype(np.float32),
                "proj_w": sd[p + "attn.c_proj.weight"].astype(np.float32),
                "proj_b": sd[p + "attn.c_proj.bias"].astype(np.float32),
                "ln2_w": sd[p + "ln_2.weight"].astype(np.float32),
                "ln2_b": sd[p + "ln_2.bias"].astype(np.float32),
                "fc_w": sd[p + "mlp.c_fc.weight"].astype(np.float32),
                "fc_b": sd[p + "mlp.c_fc.bias"].astype(np.float32),
                "fcproj_w": sd[p + "mlp.c_proj.weight"].astype(np.float32),
                "fcproj_b": sd[p + "mlp.c_proj.bias"].astype(np.float32),
            }
        )
    wte = sd["transformer.wte.weight"].astype(np.float32)
    return {
        "wte": wte,
        "wpe": sd["transformer.wpe.weight"].astype(np.float32),
        "blocks": blocks,
        "lnf_w": sd["transformer.ln_f.weight"].astype(np.float32),
        "lnf_b": sd["transformer.ln_f.bias"].astype(np.float32),
        "lm_head": np.ascontiguousarray(wte.T),  # GPT-2 ties lm_head to wte
    }


def spec_tree(tree):
    """Mirror a nested dict/list of numpy arrays as a pytree of TensorSpecs."""
    if isinstance(tree, dict):
        return {k: spec_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [spec_tree(v) for v in tree]
    arr = np.asarray(tree)
    return etl.TensorSpec(tuple(int(d) for d in arr.shape), etl.dtype(str(arr.dtype)))


def param_count(tree) -> int:
    if isinstance(tree, dict):
        return sum(param_count(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(param_count(v) for v in tree)
    return int(np.prod(np.shape(tree)))


# ---------------------------------------------------------------------------
# etl graph building blocks
# ---------------------------------------------------------------------------
def _rsqrt(x):
    return etl.divide(1.0, etl.sqrt(x))


def rms_norm(x, w, eps):
    """HF LlamaRMSNorm: w * (x * rsqrt(mean(x^2, -1, keepdims) + eps))."""
    var = etl.reduce_mean(etl.multiply(x, x), axes=-1, keepdims=True)
    return etl.multiply(w, etl.multiply(x, _rsqrt(etl.add(var, eps))))


def layer_norm(x, w, b, eps):
    """HF nn.LayerNorm (biased variance over the last axis)."""
    mean = etl.reduce_mean(x, axes=-1, keepdims=True)
    d = etl.subtract(x, mean)
    var = etl.reduce_mean(etl.multiply(d, d), axes=-1, keepdims=True)
    return etl.add(etl.multiply(etl.multiply(d, _rsqrt(etl.add(var, eps))), w), b)


def softmax(x):
    """exp(x - max) / sum(exp(x - max)) over the last axis (no etl.softmax op)."""
    e = etl.exp(etl.subtract(x, etl.reduce_max(x, axes=-1, keepdims=True)))
    return etl.divide(e, etl.reduce_sum(e, axes=-1, keepdims=True))


def silu(x):
    return etl.multiply(x, etl.sigmoid(x))


def gelu_tanh(x):
    """gelu_new / gelu_pytorch_tanh: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))."""
    inner = etl.multiply(
        etl.add(x, etl.multiply(etl.multiply(x, etl.multiply(x, x)), 0.044715)),
        float(np.sqrt(2.0 / np.pi)),
    )
    return etl.multiply(etl.multiply(x, 0.5), etl.add(1.0, etl.tanh(inner)))


def gelu_exact(x):
    """Exact erf-GELU. NOTE: ``erf`` is a v1 StableHLO export DEFERRAL."""
    return etl.multiply(etl.multiply(x, 0.5), etl.add(1.0, etl.erf(etl.multiply(x, 0.7071067811865476))))


HIDDEN_ACT = {
    "silu": silu,
    "swish": silu,
    "gelu_new": gelu_tanh,
    "gelu_pytorch_tanh": gelu_tanh,
    "gelu": gelu_exact,  # export-deferred; tanh approx is the injected alternative
}


def causal_additive_mask(s):
    """In-graph (s, s) f32 causal mask: 0 on/below the diagonal, MASK_NEG above.

    Built from ``arange`` + compare + cast rather than passed in, so the graph is
    self-contained (also exercises compare/convert export).
    """
    idx = etl.arange(0, s, dtype=etl.int32)
    keep = etl.less_equal(etl.reshape(idx, (s, 1)), etl.reshape(idx, (1, s)))
    return etl.multiply(etl.subtract(1.0, etl.cast(keep, etl.float32)), MASK_NEG)


def rope_tables(seq, head_dim, theta):
    """In-graph RoPE cos/sin of shape (seq, head_dim) (HF rotate_half convention)."""
    half = head_dim // 2
    idx = etl.cast(etl.arange(0, head_dim, 2, dtype=etl.int32), etl.float32)
    inv_freq = etl.power(theta, etl.multiply(idx, -1.0 / head_dim))
    pos = etl.cast(etl.arange(0, seq, dtype=etl.int32), etl.float32)
    freqs = etl.multiply(etl.reshape(pos, (seq, 1)), etl.reshape(inv_freq, (1, half)))
    emb = etl.concatenate([freqs, freqs], axis=1)
    return etl.cos(emb), etl.sin(emb)


def apply_rope(x, cos, sin, b, n, s, hd):
    """x (b, n, s, hd) rotated by the HF rotate_half convention."""
    half = hd // 2
    x1 = etl.slice(x, (0, 0, 0, 0), (b, n, s, half))
    x2 = etl.slice(x, (0, 0, 0, half), (b, n, s, half))
    rot = etl.concatenate([etl.negate(x2), x1], axis=3)
    cos4 = etl.reshape(cos, (1, 1, s, hd))
    sin4 = etl.reshape(sin, (1, 1, s, hd))
    return etl.add(etl.multiply(x, cos4), etl.multiply(rot, sin4))


def grouped_heads(x, b, s, n_kv, hd, n_q):
    """(b, s, n_kv*hd) -> (b, n_q, s, hd), repeating kv heads n_q/n_kv times (GQA)."""
    x = etl.transpose(etl.reshape(x, (b, s, n_kv, hd)), (0, 2, 1, 3))
    rep = n_q // n_kv
    if rep == 1:
        return x
    x = etl.reshape(x, (b, n_kv, 1, s, hd))
    x = etl.broadcast(x, (b, n_kv, rep, s, hd))
    return etl.reshape(x, (b, n_q, s, hd))


# ---------------------------------------------------------------------------
# Llama-family graph
# ---------------------------------------------------------------------------
def make_llama_graph(cfg: dict, seq_len: int, hidden_act: str | None = None):
    """Return an ``@etl.defn`` forward ``f(params, ids) -> logits`` for a Llama/Mistral cfg."""
    H = cfg["hidden_size"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_q)
    hd = H // n_q
    n_layers = cfg["num_hidden_layers"]
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    theta = float(cfg.get("rope_theta", 10000.0))
    act_name = hidden_act or cfg.get("hidden_act", "silu")
    act = HIDDEN_ACT[act_name]
    scale = hd**-0.5
    b, s = 1, seq_len

    @etl.defn
    def forward(params, ids):
        x = etl.gather(params["embed"], ids, axis=0)  # (b, s, H)
        x = etl.reshape(x, (b, s, H))
        mask = causal_additive_mask(s)
        cos, sin = rope_tables(s, hd, theta)
        for i in range(n_layers):
            L = params["layers"][i]
            h = rms_norm(x, L["ln1"], eps)
            q = grouped_heads(etl.dot(h, L["wq"]), b, s, n_q, hd, n_q)
            k = grouped_heads(etl.dot(h, L["wk"]), b, s, n_kv, hd, n_q)
            v = grouped_heads(etl.dot(h, L["wv"]), b, s, n_kv, hd, n_q)
            q = apply_rope(q, cos, sin, b, n_q, s, hd)
            k = apply_rope(k, cos, sin, b, n_q, s, hd)
            scores = etl.dot(q, etl.transpose(k, (0, 1, 3, 2)))  # (b, n_q, s, s)
            scores = etl.multiply(scores, scale)
            scores = etl.add(scores, mask)
            probs = softmax(scores)
            ctx = etl.dot(probs, v)  # (b, n_q, s, hd)
            ctx = etl.reshape(etl.transpose(ctx, (0, 2, 1, 3)), (b, s, H))
            x = etl.add(x, etl.dot(ctx, L["wo"]))
            h = rms_norm(x, L["ln2"], eps)
            x = etl.add(x, etl.dot(etl.multiply(act(etl.dot(h, L["wgate"])), etl.dot(h, L["wup"])), L["wdown"]))
        x = rms_norm(x, params["norm"], eps)
        return etl.dot(x, params["lm_head"])  # (b, s, vocab)

    return forward


# ---------------------------------------------------------------------------
# GPT-2 graph
# ---------------------------------------------------------------------------
def make_gpt2_graph(cfg: dict, seq_len: int):
    """Return an ``@etl.defn`` forward ``f(params, ids) -> logits`` for GPT-2."""
    H = cfg["n_embd"]
    n_head = cfg["n_head"]
    hd = H // n_head
    n_layer = cfg["n_layer"]
    eps = float(cfg.get("layer_norm_epsilon", 1e-5))
    act = HIDDEN_ACT[cfg.get("activation_function", "gelu_new")]
    scale = hd**-0.5
    b, s = 1, seq_len

    @etl.defn
    def forward(params, ids):
        wte, wpe = params["wte"], params["wpe"]
        tok = etl.reshape(etl.gather(wte, ids, axis=0), (b, s, H))
        pos_idx = etl.reshape(etl.arange(0, s, dtype=etl.int32), (1, s))
        pos = etl.reshape(etl.gather(wpe, pos_idx, axis=0), (b, s, H))
        x = etl.add(tok, pos)
        mask = causal_additive_mask(s)
        for i in range(n_layer):
            L = params["blocks"][i]
            h = layer_norm(x, L["ln1_w"], L["ln1_b"], eps)
            qkv = etl.add(etl.dot(h, L["attn_w"]), L["attn_b"])  # (b, s, 3H)
            q = grouped_heads(etl.slice(qkv, (0, 0, 0), (b, s, H)), b, s, n_head, hd, n_head)
            k = grouped_heads(etl.slice(qkv, (0, 0, H), (b, s, H)), b, s, n_head, hd, n_head)
            v = grouped_heads(etl.slice(qkv, (0, 0, 2 * H), (b, s, H)), b, s, n_head, hd, n_head)
            scores = etl.dot(q, etl.transpose(k, (0, 1, 3, 2)))
            scores = etl.multiply(scores, scale)
            scores = etl.add(scores, mask)
            ctx = etl.dot(softmax(scores), v)
            ctx = etl.reshape(etl.transpose(ctx, (0, 2, 1, 3)), (b, s, H))
            x = etl.add(x, etl.add(etl.dot(ctx, L["proj_w"]), L["proj_b"]))
            h = layer_norm(x, L["ln2_w"], L["ln2_b"], eps)
            h = act(etl.add(etl.dot(h, L["fc_w"]), L["fc_b"]))
            x = etl.add(x, etl.add(etl.dot(h, L["fcproj_w"]), L["fcproj_b"]))
        x = layer_norm(x, params["lnf_w"], params["lnf_b"], eps)
        return etl.dot(x, params["lm_head"])

    return forward


# ---------------------------------------------------------------------------
# pure-numpy references (independent implementations: per-head loops, no etl)
# ---------------------------------------------------------------------------
def _ref_rms_norm(x, w, eps):
    return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps)))).astype(np.float32) * w


def _ref_layer_norm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    d = x - m
    var = (d * d).mean(-1, keepdims=True)
    return (d / np.sqrt(var + np.float32(eps))) * w + b


def _ref_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _ref_rope_tables(seq, head_dim, theta):
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.power(np.float32(theta), idx * np.float32(-1.0 / head_dim)).astype(np.float32)
    pos = np.arange(seq, dtype=np.float32)
    freqs = (pos[:, None] * inv_freq[None, :]).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _ref_rope(x, cos, sin):
    """x (s, n, hd) -> rotated, HF rotate_half convention."""
    hd = x.shape[-1]
    half = hd // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return (x * cos[:, None, :] + rot * sin[:, None, :]).astype(np.float32)


def _ref_mask(s):
    keep = np.arange(s)[:, None] <= np.arange(s)[None, :]
    return np.where(keep, np.float32(0.0), np.float32(MASK_NEG)).astype(np.float32)


def _ref_gelu_tanh(x):
    return (x * 0.5 * (1.0 + np.tanh(np.sqrt(np.float32(2.0 / np.pi)) * (x + 0.044715 * x**3)))).astype(np.float32)


def _ref_gelu_exact(x):
    from math import erf

    v = np.vectorize(erf, otypes=[np.float32])
    return (x * 0.5 * (1.0 + v(x * np.float32(0.7071067811865476)))).astype(np.float32)


_REF_ACT = {
    "silu": lambda x: (x / (1.0 + np.exp(-x))).astype(np.float32),
    "swish": lambda x: (x / (1.0 + np.exp(-x))).astype(np.float32),
    "gelu_new": _ref_gelu_tanh,
    "gelu_pytorch_tanh": _ref_gelu_tanh,
    "gelu": _ref_gelu_exact,
}


def _ref_attention(q, k, v, mask, scale, n_q, n_kv):
    """q,k,v: (s, n, hd) with n_kv kv heads. Returns (s, n_q*hd) f32."""
    s, hd = q.shape[0], q.shape[-1]
    n_kv = k.shape[1]
    rep = n_q // n_kv
    if rep > 1:
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)
    ctx = np.zeros((s, n_q, hd), dtype=np.float32)
    for h in range(n_q):
        sc = (q[:, h, :] @ k[:, h, :].T).astype(np.float32) * np.float32(scale) + mask
        ctx[:, h, :] = _ref_softmax(sc) @ v[:, h, :]
    return ctx.reshape(s, n_q * hd)


def llama_reference(cfg, params, ids, hidden_act=None):
    """Pure-numpy Llama/Mistral forward. ids: (1, s) int."""
    act = _REF_ACT[hidden_act or cfg.get("hidden_act", "silu")]
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    theta = float(cfg.get("rope_theta", 10000.0))
    n_q = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_q)
    hd = cfg["hidden_size"] // n_q
    s = ids.shape[1]
    x = params["embed"][ids[0]].astype(np.float32)  # (s, H)
    mask = _ref_mask(s)
    cos, sin = _ref_rope_tables(s, hd, theta)
    for L in params["layers"]:
        h = _ref_rms_norm(x, L["ln1"], eps)
        q = _ref_rope((h @ L["wq"]).reshape(s, n_q, hd), cos, sin)
        k = _ref_rope((h @ L["wk"]).reshape(s, n_kv, hd), cos, sin)
        v = (h @ L["wv"]).reshape(s, n_kv, hd)
        x = x + (_ref_attention(q, k, v, mask, hd**-0.5, n_q, n_kv) @ L["wo"])
        h = _ref_rms_norm(x, L["ln2"], eps)
        x = x + (act(h @ L["wgate"]) * (h @ L["wup"])) @ L["wdown"]
    x = _ref_rms_norm(x, params["norm"], eps)
    return (x @ params["lm_head"])[None].astype(np.float32)


def gpt2_reference(cfg, params, ids):
    """Pure-numpy GPT-2 forward. ids: (1, s) int."""
    act = _REF_ACT[cfg.get("activation_function", "gelu_new")]
    eps = float(cfg.get("layer_norm_epsilon", 1e-5))
    H = cfg["n_embd"]
    n_head = cfg["n_head"]
    hd = H // n_head
    s = ids.shape[1]
    x = params["wte"][ids[0]] + params["wpe"][np.arange(s)]
    x = x.astype(np.float32)
    mask = _ref_mask(s)
    for L in params["blocks"]:
        h = _ref_layer_norm(x, L["ln1_w"], L["ln1_b"], eps)
        qkv = h @ L["attn_w"] + L["attn_b"]
        q = qkv[:, :H].reshape(s, n_head, hd)
        k = qkv[:, H : 2 * H].reshape(s, n_head, hd)
        v = qkv[:, 2 * H :].reshape(s, n_head, hd)
        x = x + (_ref_attention(q, k, v, mask, hd**-0.5, n_head, n_head) @ L["proj_w"] + L["proj_b"])
        h = _ref_layer_norm(x, L["ln2_w"], L["ln2_b"], eps)
        h = act(h @ L["fc_w"] + L["fc_b"])
        x = x + (h @ L["fcproj_w"] + L["fcproj_b"])
    x = _ref_layer_norm(x, params["lnf_w"], params["lnf_b"], eps)
    return (x @ params["lm_head"])[None].astype(np.float32)


# ---------------------------------------------------------------------------
# graph introspection helpers
# ---------------------------------------------------------------------------
def count_ops(graph) -> dict[str, int]:
    """Op-name histogram of a traced Graph (walks nested regions)."""
    counts: dict[str, int] = {}

    def walk_block(blk):
        for op in blk.ops:
            counts[op.opdef.name] = counts.get(op.opdef.name, 0) + 1
            for reg in op.regions:
                for b2 in reg.blocks:
                    walk_block(b2)

    walk_block(graph.module.functions[0].entry_block)
    return counts


def graph_stats(graph, export_text: str | None = None) -> dict:
    ops = count_ops(graph)
    stats = {
        "graph_inputs": len(graph.tensor_specs),
        "ops_total": sum(ops.values()),
        "distinct_ops": len(ops),
        "op_histogram": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
    }
    if export_text is not None:
        stats["stablehlo_lines"] = len(export_text.splitlines())
        stats["stablehlo_bytes"] = len(export_text)
    return stats
