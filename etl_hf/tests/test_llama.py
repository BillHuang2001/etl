"""Tests for :mod:`etl_hf.models.llama` — the Llama/Mistral family.

Two things are pinned here:

* **Parity** — the traced-and-compiled graph (numpy interpreter backend) against
  the independent pure-numpy oracle :func:`etl_hf.models.llama.reference`, across
  GQA/MQA/MHA head layouts, batch sizes, layer counts, activations and an explicit
  ``head_dim`` override.
* **Weight canonicalisation** — the HF state dict → canonical ``(in_features,
  out_features)`` flat dict mapping, where a silent transpose/mapping bug would
  hide: exact key sets, exact orientation, exact values, tie resolution, and one
  explicit error per failure mode.

Everything here is offline and numpy-only (no torch / transformers / network):
configs are dicts, checkpoints are synthetic seeded arrays, and "weights" are
random but well-scaled so fp32 stays well-conditioned.
"""

from __future__ import annotations

import numpy as np
import pytest

import etl
from etl_hf import ops
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.models import llama

# ---------------------------------------------------------------------------
# helpers: synthetic configs, checkpoints, params
# ---------------------------------------------------------------------------
#: Weight scale for synthetic params — small enough that softmax/RMSNorm stay in
#: a well-conditioned fp32 range (the parity assertion is about *semantics*, not
#: about surviving saturation).
_SCALE = 0.3


def _llama_json(**overrides) -> dict:
    """A minimal but complete HF ``LlamaConfig``-shaped dict.

    Defaults: ``hidden_size=16``, 4 attention heads, 2 KV heads (GQA), 2 layers,
    ``intermediate_size=32``, ``vocab_size=32``, ``head_dim=4`` derived. Any
    field can be overridden by keyword.

    Args:
        **overrides: config fields to replace.

    Returns:
        dict: the raw config dict.
    """
    raw = {
        "model_type": "llama",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    raw.update(overrides)
    return raw


def _cfg(**overrides) -> ModelConfig:
    """Build a :class:`ModelConfig` from :func:`_llama_json` plus overrides."""
    return ModelConfig.from_dict(_llama_json(**overrides))


#: The config whose linears are ALL non-square, so a missing or extra transpose
#: can never accidentally pass: hidden 16, 4 query heads x head_dim 8 => q_dim 32,
#: 1 KV head x head_dim 8 => kv_dim 8, intermediate 32, vocab 32.
_NON_SQUARE_JSON = {
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "intermediate_size": 32,
    "vocab_size": 32,
    "num_hidden_layers": 2,
}


def _hf_state_dict(cfg: ModelConfig, *, seed: int = 0, head: str = "present") -> dict:
    """Synthetic HF-style state dict for ``cfg`` (linears in HF ``(out, in)``).

    Args:
        cfg: the llama config the checkpoint must match.
        seed: RNG seed (deterministic synthetic weights).
        head: ``"present"`` (a random, independent ``lm_head.weight``),
            ``"tied"`` (``lm_head.weight`` is exactly ``embed_tokens.weight``) or
            ``"absent"`` (no ``lm_head.weight`` key at all).

    Returns:
        dict: ``str -> float32 ndarray`` in HF orientation.
    """
    rng = np.random.default_rng(seed)
    hidden = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size

    def draw(shape):
        return (rng.standard_normal(shape) * _SCALE).astype(np.float32)

    state = {"model.embed_tokens.weight": draw((vocab, hidden))}
    for index in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{index}."
        state[prefix + "input_layernorm.weight"] = draw((hidden,))
        state[prefix + "self_attn.q_proj.weight"] = draw((q_dim, hidden))
        state[prefix + "self_attn.k_proj.weight"] = draw((kv_dim, hidden))
        state[prefix + "self_attn.v_proj.weight"] = draw((kv_dim, hidden))
        state[prefix + "self_attn.o_proj.weight"] = draw((hidden, q_dim))
        state[prefix + "post_attention_layernorm.weight"] = draw((hidden,))
        state[prefix + "mlp.gate_proj.weight"] = draw((inter, hidden))
        state[prefix + "mlp.up_proj.weight"] = draw((inter, hidden))
        state[prefix + "mlp.down_proj.weight"] = draw((hidden, inter))
    state["model.norm.weight"] = draw((hidden,))
    if head == "present":
        state["lm_head.weight"] = draw((vocab, hidden))
    elif head == "tied":
        state["lm_head.weight"] = state["model.embed_tokens.weight"].copy()
    elif head != "absent":
        raise AssertionError(f"unknown head mode {head!r}")
    return state


def _params(cfg: ModelConfig, *, seed: int = 0) -> dict:
    """Synthetic canonical params: seeded well-scaled float32 arrays per spec."""
    rng = np.random.default_rng(seed)
    return {
        key: (rng.standard_normal(spec.shape) * _SCALE).astype(np.float32)
        for key, spec in llama.param_specs(cfg).items()
    }


def _ids(cfg: ModelConfig, batch: int, seq: int, *, seed: int = 1) -> np.ndarray:
    """Random ``(batch, seq)`` int32 token ids inside the vocabulary."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)


def _specs(cfg: ModelConfig, batch: int, seq: int) -> tuple[dict, etl.TensorSpec]:
    """The trace/build argument tree: (param specs, input_ids spec)."""
    return llama.param_specs(cfg), etl.TensorSpec(shape=(batch, seq), dtype=etl.int32)


def _traced(cfg: ModelConfig, seq: int, batch: int = 1):
    """Trace ``llama.graph(cfg, seq, batch)``; return ``(defn, Graph)``."""
    defn = llama.graph(cfg, seq, batch_size=batch)
    param_spec_tree, ids_spec = _specs(cfg, batch, seq)
    return defn, etl.trace(defn, param_spec_tree, ids_spec)


def _run(cfg: ModelConfig, seq: int, batch: int, params: dict, ids: np.ndarray) -> np.ndarray:
    """Build the graph on the numpy backend, run it, and return numpy logits."""
    defn = llama.graph(cfg, seq, batch_size=batch)
    executable = etl.build(defn, *_specs(cfg, batch, seq))
    return etl.run(executable, params, ids).numpy()


def _assert_parity(
    cfg: ModelConfig, seq: int, batch: int = 1, *, seed: int = 0
) -> float:
    """Run graph vs reference and assert parity; return the measured max-abs.

    Args:
        cfg: llama config.
        seq: static sequence length.
        batch: static batch size.
        seed: synthetic-weight seed.

    Returns:
        float: the measured maximum absolute difference (reported by callers).
    """
    params = _params(cfg, seed=seed)
    ids = _ids(cfg, batch, seq)
    got = _run(cfg, seq, batch, params, ids)
    want = llama.reference(cfg, params, ids)
    assert got.shape == want.shape == (batch, seq, cfg.vocab_size)
    assert got.dtype == np.float32 and want.dtype == np.float32
    max_abs = float(np.max(np.abs(got - want)))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6), (
        f"parity failure: max |graph - reference| = {max_abs:g} "
        f"(batch={batch}, seq={seq}, heads={cfg.num_attention_heads}/"
        f"{cfg.num_key_value_heads}, layers={cfg.num_hidden_layers}, "
        f"act={cfg.hidden_act}, head_dim={cfg.head_dim})"
    )
    # A graph that collapsed to a constant would pass allclose only against a
    # also-collapsed reference; the oracle is plain numpy, so this guards both.
    assert float(np.std(want)) > 0.0
    return max_abs


# ---------------------------------------------------------------------------
# 1. parity: the default small config
# ---------------------------------------------------------------------------
def test_parity_default_config():
    """hidden 16 / 4 q-heads / 2 kv-heads / 2 layers / seq 4 / batch 1."""
    cfg = _cfg()
    assert (cfg.head_dim, cfg.intermediate_size, cfg.vocab_size) == (4, 32, 32)
    max_abs = _assert_parity(cfg, seq=4)
    assert max_abs < 1e-6


# ---------------------------------------------------------------------------
# 2. GQA variants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_key_value_heads", [4, 2, 1])
def test_parity_gqa_variants(num_key_value_heads):
    """MHA (4 kv heads), GQA (2) and MQA (1) all match the oracle."""
    cfg = _cfg(num_key_value_heads=num_key_value_heads)
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
    assert _assert_parity(cfg, seq=4) < 1e-6


# ---------------------------------------------------------------------------
# 3. batch size and layer count
# ---------------------------------------------------------------------------
def test_parity_batch_two():
    """A batch of 2 independent sequences matches the oracle."""
    assert _assert_parity(_cfg(), seq=3, batch=2) < 1e-6


@pytest.mark.parametrize("num_hidden_layers", [1, 3])
def test_parity_layer_counts(num_hidden_layers):
    """One and three transformer blocks (block indexing/stacking) match."""
    cfg = _cfg(num_hidden_layers=num_hidden_layers)
    assert _assert_parity(cfg, seq=4) < 1e-6


# ---------------------------------------------------------------------------
# 4. activations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_act", ["silu", "gelu_new", "gelu", "relu"])
def test_parity_activations(hidden_act):
    """Every supported SwiGLU/MLP activation matches the oracle.

    ``"gelu"`` is the exact erf form: it runs on the numpy interpreter backend but
    is a StableHLO export deferral, so ``activation_is_export_safe`` must say so.
    """
    cfg = _cfg(hidden_act=hidden_act)
    assert _assert_parity(cfg, seq=4) < 1e-6
    assert ops.activation_is_export_safe(hidden_act) is (hidden_act != "gelu")


# ---------------------------------------------------------------------------
# 5. explicit head_dim override (head_dim != hidden // heads)
# ---------------------------------------------------------------------------
def test_parity_explicit_head_dim_override():
    """head_dim=8 with hidden 16 / 4 heads (q is (16, 32), o is (32, 16))."""
    cfg = _cfg(hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    assert cfg.head_dim == 8 != cfg.hidden_size // cfg.num_attention_heads
    specs = llama.param_specs(cfg)
    assert specs["layers.0.attn.q.weight"].shape == (16, 32)
    assert specs["layers.0.attn.k.weight"].shape == (16, 16)
    assert specs["layers.0.attn.o.weight"].shape == (32, 16)
    assert _assert_parity(cfg, seq=4) < 1e-6


# ---------------------------------------------------------------------------
# 6. canonicalize_weights
# ---------------------------------------------------------------------------
def test_canonicalize_keys_orientation_and_values():
    """Canonical key set, ``(in, out)`` orientation and exact values (non-square)."""
    cfg = _cfg(**_NON_SQUARE_JSON)
    state = _hf_state_dict(cfg, head="present")
    params = llama.canonicalize_weights(cfg, state)

    assert len(params) == 2 + cfg.num_hidden_layers * 9 + 1  # embed + layers + norm + head
    assert set(params) == set(llama.param_specs(cfg))
    # Non-transposed: taken verbatim.
    assert np.array_equal(params["embed.weight"], state["model.embed_tokens.weight"])
    assert np.array_equal(params["final_norm.weight"], state["model.norm.weight"])
    assert np.array_equal(
        params["layers.1.attn_norm.weight"], state["model.layers.1.input_layernorm.weight"]
    )
    assert np.array_equal(
        params["layers.0.mlp_norm.weight"],
        state["model.layers.0.post_attention_layernorm.weight"],
    )
    # Transposed: exact transpose of the HF tensor, and non-square so a missing
    # transpose cannot pass.
    transposed = {
        "layers.0.attn.q.weight": "self_attn.q_proj.weight",
        "layers.0.attn.k.weight": "self_attn.k_proj.weight",
        "layers.1.attn.v.weight": "self_attn.v_proj.weight",
        "layers.1.attn.o.weight": "self_attn.o_proj.weight",
        "layers.0.mlp.gate.weight": "mlp.gate_proj.weight",
        "layers.1.mlp.up.weight": "mlp.up_proj.weight",
        "layers.1.mlp.down.weight": "mlp.down_proj.weight",
    }
    for canonical, hf_suffix in transposed.items():
        index = canonical.split(".")[1]
        raw = state[f"model.layers.{index}.{hf_suffix}"]
        assert params[canonical].shape == (raw.shape[1], raw.shape[0]) != raw.shape
        assert np.array_equal(params[canonical], raw.T), canonical
    assert np.array_equal(params["lm_head.weight"], state["lm_head.weight"].T)

    # Shape check against the specs, and float32/contiguous/writable.
    for key, spec in llama.param_specs(cfg).items():
        array = params[key]
        assert array.shape == spec.shape, key
        assert array.dtype == np.float32, key
        assert array.flags["C_CONTIGUOUS"] and array.flags["WRITEABLE"], key

def test_canonicalize_ignores_rotary_inv_freq():
    """``model.rotary_emb.inv_freq`` is a documented, ignored buffer."""
    cfg = _cfg()
    state = _hf_state_dict(cfg)
    state["model.rotary_emb.inv_freq"] = np.arange(cfg.head_dim // 2, dtype=np.float32)
    params = llama.canonicalize_weights(cfg, state)
    assert set(params) == set(llama.param_specs(cfg))


def test_canonicalize_tied_head_absent_derives_from_embedding():
    """tied + no ``lm_head.weight`` → ``embed.weight.T``."""
    cfg = _cfg(tie_word_embeddings=True)
    state = _hf_state_dict(cfg, head="absent")
    params = llama.canonicalize_weights(cfg, state)
    assert np.array_equal(
        params["lm_head.weight"], state["model.embed_tokens.weight"].T
    )


def test_canonicalize_tied_head_present_and_consistent():
    """tied + a consistent ``lm_head.weight`` → accepted (transposed)."""
    cfg = _cfg(tie_word_embeddings=True)
    state = _hf_state_dict(cfg, head="tied")
    params = llama.canonicalize_weights(cfg, state)
    assert np.array_equal(params["lm_head.weight"], state["lm_head.weight"].T)


def test_canonicalize_tied_head_present_but_inconsistent():
    """tied + a disagreeing ``lm_head.weight`` → HFWeightError (never a silent pick)."""
    cfg = _cfg(tie_word_embeddings=True)
    state = _hf_state_dict(cfg, head="tied")
    state["lm_head.weight"] = state["lm_head.weight"] + np.float32(1e-3)
    with pytest.raises(HFWeightError, match="tie_word_embeddings"):
        llama.canonicalize_weights(cfg, state)


def test_canonicalize_untied_uses_lm_head():
    """untied + present ``lm_head.weight`` → its transpose (not the embedding)."""
    cfg = _cfg(tie_word_embeddings=False)
    state = _hf_state_dict(cfg, head="present")
    params = llama.canonicalize_weights(cfg, state)
    assert np.array_equal(params["lm_head.weight"], state["lm_head.weight"].T)
    assert not np.allclose(params["lm_head.weight"], params["embed.weight"].T)


def test_canonicalize_untied_missing_lm_head():
    """untied + no ``lm_head.weight`` → HFWeightError naming the key."""
    cfg = _cfg(tie_word_embeddings=False)
    state = _hf_state_dict(cfg, head="absent")
    with pytest.raises(HFWeightError, match="lm_head.weight"):
        llama.canonicalize_weights(cfg, state)


def test_canonicalize_unexpected_key_raises():
    """Any key that maps nowhere is an error (sorted list in the message)."""
    cfg = _cfg()
    state = _hf_state_dict(cfg)
    state["model.layers.0.extra_scale"] = np.ones(cfg.hidden_size, dtype=np.float32)
    state["aaa_unrelated"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(HFWeightError) as info:
        llama.canonicalize_weights(cfg, state)
    message = str(info.value)
    assert "aaa_unrelated" in message and "extra_scale" in message
    assert message.index("aaa_unrelated") < message.index("model.layers.0.extra_scale")


def test_canonicalize_bias_raises():
    """Llama v1 is bias-free: a ``*.bias`` tensor is unsupported, explicitly."""
    cfg = _cfg()
    state = _hf_state_dict(cfg)
    state["model.layers.0.self_attn.q_proj.bias"] = np.zeros(4 * cfg.head_dim, np.float32)
    state["model.layers.1.mlp.down_proj.bias"] = np.zeros(cfg.hidden_size, np.float32)
    with pytest.raises(HFUnsupportedError) as info:
        llama.canonicalize_weights(cfg, state)
    message = str(info.value)
    assert "bias" in message
    assert "self_attn.q_proj.bias" in message and "mlp.down_proj.bias" in message
    assert message.index("self_attn.q_proj.bias") < message.index("mlp.down_proj.bias")


def test_canonicalize_wrong_shape_raises():
    """A shape that disagrees with the config is an explicit error naming the key."""
    cfg = _cfg(**_NON_SQUARE_JSON)
    state = _hf_state_dict(cfg)
    # q_proj must be (q_dim=32, hidden=16); hand it an already-transposed-looking
    # (16, 32) instead, which the loader's transpose turns into (32, 16).
    state["model.layers.0.self_attn.q_proj.weight"] = np.zeros((16, 32), np.float32)
    with pytest.raises(HFWeightError) as info:
        llama.canonicalize_weights(cfg, state)
    message = str(info.value)
    assert "self_attn.q_proj.weight" in message
    assert "(32, 16)" in message and "requires (16, 32)" in message
    # A norm scale of the wrong length is rejected too.
    state = _hf_state_dict(cfg)
    state["model.norm.weight"] = np.zeros(cfg.hidden_size + 1, np.float32)
    with pytest.raises(HFWeightError, match="requires \\(16,\\)"):
        llama.canonicalize_weights(cfg, state)


def test_canonicalize_missing_key_raises():
    """A missing required key names the key and suggests near misses."""
    cfg = _cfg()
    state = _hf_state_dict(cfg)
    del state["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(HFWeightError) as info:
        llama.canonicalize_weights(cfg, state)
    message = str(info.value)
    assert "model.layers.1.mlp.up_proj.weight" in message
    assert "layers.1.mlp.up.weight" in message  # the canonical target it feeds
    assert "self_attn" in message or "mlp" in message  # near-miss suggestions


def test_canonicalize_rejects_foreign_family():
    """A gpt2 config is refused (the caller must pick the right family module)."""
    with pytest.raises(HFConfigError, match="family"):
        llama.canonicalize_weights(_gpt2_cfg(), {})


def _gpt2_cfg() -> ModelConfig:
    """A minimal GPT-2 config (used only to prove family rejection)."""
    return ModelConfig.from_dict(
        {
            "model_type": "gpt2",
            "n_embd": 16,
            "n_head": 4,
            "n_layer": 1,
            "n_positions": 8,
            "n_vocab": 32,
        }
    )


# ---------------------------------------------------------------------------
# 7. param_specs
# ---------------------------------------------------------------------------
def test_param_specs_match_canonicalized_weights():
    """Same keys, same shapes, all float32 — the two halves of the contract."""
    cfg = _cfg(**_NON_SQUARE_JSON)
    specs = llama.param_specs(cfg)
    params = llama.canonicalize_weights(cfg, _hf_state_dict(cfg))
    assert list(specs) == list(params)
    for key, spec in specs.items():
        assert spec.dtype == etl.float32, key
        assert spec.shape == params[key].shape, key
    assert specs["layers.0.attn.k.weight"].shape == (16, 8)  # 1 kv head x head_dim 8
    assert specs["lm_head.weight"].shape == (16, 32)


# ---------------------------------------------------------------------------
# 8. graph shape/dtype and "no baked-in weights"
# ---------------------------------------------------------------------------
def test_graph_output_shape_and_dtype():
    """The graph yields ``(batch, seq, vocab)`` float32 and holds no constants."""
    cfg = _cfg()
    defn, traced = _traced(cfg, seq=4, batch=2)
    # One tensor input per canonical param, plus input_ids: the weights are graph
    # INPUTS — none of them was embedded as an etl.constant payload.
    assert len(traced.tensor_specs) == len(llama.param_specs(cfg)) + 1
    assert traced.tensor_specs[-1].dtype == etl.int32
    assert traced.tensor_specs[-1].shape == (2, 4)

    ids = _ids(cfg, 2, 4)
    out = etl.run(etl.build(defn, *_specs(cfg, 2, 4)), _params(cfg), ids)
    assert isinstance(out, etl.Tensor)
    assert out.shape == (2, 4, cfg.vocab_size)
    assert out.dtype == etl.float32


def test_graph_has_no_constant_payloads():
    """No weight is inlined: the only ``constant`` ops are tiny structural ones.

    ``etl.constant`` on a weight would inline the whole payload into the graph
    (and then into MLIR text), so the RoPE tables and the causal mask are composed
    from ``arange`` + arithmetic instead. ``arange`` itself is an IR constant, so
    the assertion is on *size*: the constants total a handful of elements while
    the parameters total thousands.
    """
    cfg = _cfg()
    _, traced = _traced(cfg, seq=4)
    constant_elements = 0
    for function in traced.module.functions:
        for block in function.region.blocks:
            for op in block.ops:
                if op.name != "constant":
                    continue
                shape = tuple(op.results[0].type.shape)
                constant_elements += int(np.prod(shape, dtype=np.int64))
    param_elements = sum(
        int(np.prod(spec.shape, dtype=np.int64)) for spec in llama.param_specs(cfg).values()
    )
    assert param_elements > 1000
    assert constant_elements <= 64, (
        f"the graph inlines {constant_elements} constant elements — weights must stay "
        "graph inputs (never etl.constant)"
    )


def test_causal_direction():
    """Position t cannot see token t+1 (HF causal mask, not the anti-causal inverse)."""
    cfg = _cfg()
    params = _params(cfg)
    ids = _ids(cfg, 1, 4)
    tampered = ids.copy()
    tampered[0, -1] = (int(tampered[0, -1]) + 1) % cfg.vocab_size

    base = _run(cfg, 4, 1, params, ids)
    moved = _run(cfg, 4, 1, params, tampered)
    # Everything before the last position must be bit-identical ...
    assert np.array_equal(base[:, :-1], moved[:, :-1])
    # ... and the last position must actually depend on its own token.
    assert not np.allclose(base[:, -1], moved[:, -1])

    # The oracle follows the same convention (a shared inversion would show up as
    # a graph/reference mismatch above; this pins the oracle independently).
    ref_base = llama.reference(cfg, params, ids)
    ref_moved = llama.reference(cfg, params, tampered)
    assert np.array_equal(ref_base[:, :-1], ref_moved[:, :-1])
    assert not np.allclose(ref_base[:, -1], ref_moved[:, -1])


# ---------------------------------------------------------------------------
# 9. explicit errors from graph()
# ---------------------------------------------------------------------------
def test_graph_rejects_seq_beyond_max_positions():
    """``seq_len > max_position_embeddings`` names the limit and the config field."""
    cfg = _cfg(max_position_embeddings=8)
    with pytest.raises(HFConfigError) as info:
        llama.graph(cfg, 9)
    message = str(info.value)
    assert "9" in message and "8" in message and "max_position_embeddings" in message
    # At the limit it is fine.
    assert llama.graph(cfg, 8) is not None


def test_graph_rejects_zero_seq_len():
    """``seq_len=0`` is a config error, not a silently empty graph."""
    with pytest.raises(HFConfigError, match="seq_len"):
        llama.graph(_cfg(), 0)


def test_graph_rejects_zero_batch_size():
    """``batch_size=0`` is a config error."""
    with pytest.raises(HFConfigError, match="batch_size"):
        llama.graph(_cfg(), 4, batch_size=0)


def test_graph_rejects_foreign_family():
    """A gpt2 config cannot be built by the llama module."""
    with pytest.raises(HFConfigError, match="family"):
        llama.graph(_gpt2_cfg(), 4)


def test_param_specs_rejects_foreign_family():
    """``param_specs`` refuses a gpt2 config too."""
    with pytest.raises(HFConfigError, match="family"):
        llama.param_specs(_gpt2_cfg())


# ---------------------------------------------------------------------------
# 10. determinism
# ---------------------------------------------------------------------------
def test_end_to_end_determinism():
    """Two runs of one executable on the same params are bit-identical."""
    cfg = _cfg()
    params = _params(cfg)
    ids = _ids(cfg, 1, 4)
    executable = etl.build(llama.graph(cfg, 4, batch_size=1), *_specs(cfg, 1, 4))
    first = etl.run(executable, params, ids).numpy()
    second = etl.run(executable, params, ids).numpy()
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# 11. the reference forward's own validation
# ---------------------------------------------------------------------------
def test_reference_rejects_missing_or_misshaped_params():
    """The oracle validates params exactly like the specs (never guesses)."""
    cfg = _cfg()
    params = _params(cfg)
    del params["layers.1.attn.o.weight"]
    with pytest.raises(HFWeightError, match="layers.1.attn.o.weight"):
        llama.reference(cfg, params, _ids(cfg, 1, 3))

    params = _params(cfg)
    params["layers.0.mlp.down.weight"] = np.zeros((16, 32), np.float32)
    with pytest.raises(HFWeightError, match="layers.0.mlp.down.weight"):
        llama.reference(cfg, params, _ids(cfg, 1, 3))


def test_reference_rejects_bad_input_ids():
    """``input_ids`` must be a 2-D in-vocabulary integer array."""
    cfg = _cfg()
    params = _params(cfg)
    with pytest.raises(HFConfigError, match="2-D"):
        llama.reference(cfg, params, _ids(cfg, 1, 3).reshape(3))
    with pytest.raises(ValueError, match="integers"):
        llama.reference(cfg, params, _ids(cfg, 1, 3).astype(np.float32))
    out_of_vocab = _ids(cfg, 1, 3)
    out_of_vocab[0, 0] = cfg.vocab_size
    with pytest.raises(ValueError, match="vocabulary"):
        llama.reference(cfg, params, out_of_vocab)
    too_long = _ids(cfg, 1, cfg.max_position_embeddings + 1)
    with pytest.raises(HFConfigError, match="max_position_embeddings"):
        llama.reference(cfg, params, too_long)


def test_graph_returns_a_fresh_defn_each_call():
    """``graph()`` builds a new ``@etl.defn`` per call (nothing shared/global)."""
    cfg = _cfg()
    first = llama.graph(cfg, 4)
    second = llama.graph(cfg, 4)
    assert first is not second
    assert getattr(first, "__etl_defn__", False)
