"""Tests for :mod:`etl_hf.config` — HF ``config.json`` normalisation.

Everything here is offline: configs are plain dicts written to ``tmp_path``.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from etl_hf.config import FAMILY_BY_MODEL_TYPE, ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError

LLAMA_JSON = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 176,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}

GPT2_JSON = {
    "architectures": ["GPT2LMHeadModel"],
    "model_type": "gpt2",
    "n_embd": 32,
    "n_head": 4,
    "n_layer": 2,
    "n_positions": 16,
    "n_vocab": 1000,
    "n_inner": 128,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": True,
}


def write_config(tmp_path, payload, name="config.json"):
    """Write ``payload`` (a dict, or raw text) as ``config.json`` and return the path."""
    path = tmp_path / name
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# family resolution and normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2"])
def test_llama_family_model_types(model_type):
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "model_type": model_type})
    assert cfg.family == "llama"
    assert cfg.model_type == model_type
    assert cfg.hidden_size == 64
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 8
    assert cfg.rope_theta == 10000.0
    assert cfg.layer_norm_eps == 1e-6
    assert cfg.tie_word_embeddings is False


def test_family_map_covers_every_supported_model_type():
    assert FAMILY_BY_MODEL_TYPE["llama"] == "llama"
    assert FAMILY_BY_MODEL_TYPE["mistral"] == "llama"
    assert FAMILY_BY_MODEL_TYPE["qwen2"] == "llama"
    assert FAMILY_BY_MODEL_TYPE["gpt2"] == "gpt2"


@pytest.mark.parametrize("num_key_value_heads,expected", [(8, 8), (4, 4), (2, 2), (1, 1)])
def test_gqa_and_mqa_head_counts(num_key_value_heads, expected):
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "num_key_value_heads": num_key_value_heads})
    assert cfg.num_key_value_heads == expected
    assert cfg.num_attention_heads == 8


def test_num_key_value_heads_defaults_to_num_attention_heads():
    raw = {k: v for k, v in LLAMA_JSON.items() if k != "num_key_value_heads"}
    cfg = ModelConfig.from_dict(raw)
    assert cfg.num_key_value_heads == cfg.num_attention_heads == 8


def test_head_dim_default_and_explicit_override():
    assert ModelConfig.from_dict(LLAMA_JSON).head_dim == 8
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "head_dim": 16})
    assert cfg.head_dim == 16
    # An explicit head_dim that is not hidden_size // num_attention_heads is honoured
    # (HF models with attention head dim != hidden/heads exist).
    assert cfg.hidden_size // cfg.num_attention_heads == 8


def test_head_dim_override_ignored_for_gpt2():
    cfg = ModelConfig.from_dict({**GPT2_JSON, "head_dim": 999})
    assert cfg.head_dim == 8


def test_llama_defaults_when_optional_fields_absent():
    minimal = {
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
    }
    cfg = ModelConfig.from_dict(minimal)
    assert cfg.hidden_act == "silu"
    assert cfg.layer_norm_eps == 1e-6
    assert cfg.rope_theta == 10000.0
    assert cfg.tie_word_embeddings is False
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 8


def test_explicit_rope_theta_is_kept():
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "rope_theta": 500000.0})
    assert cfg.rope_theta == 500000.0


def test_tie_word_embeddings_true_is_kept():
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "tie_word_embeddings": True})
    assert cfg.tie_word_embeddings is True


def test_hidden_act_is_normalised_but_not_validated():
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "hidden_act": "  SiLU "})
    assert cfg.hidden_act == "silu"
    # Not in any activation table: the family module owns that decision.
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "hidden_act": "mystery_act"})
    assert cfg.hidden_act == "mystery_act"


def test_unknown_fields_are_ignored_but_preserved_in_raw():
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "some_future_field": {"a": 1}})
    assert cfg.raw["some_future_field"] == {"a": 1}
    assert "some_future_field" not in cfg.__dict__


def test_rope_theta_none_for_gpt2():
    assert ModelConfig.from_dict(GPT2_JSON).rope_theta is None


# ---------------------------------------------------------------------------
# GPT-2 key normalisation
# ---------------------------------------------------------------------------


def test_gpt2_key_normalisation():
    cfg = ModelConfig.from_dict(GPT2_JSON)
    assert cfg.family == "gpt2"
    assert cfg.model_type == "gpt2"
    assert cfg.hidden_size == 32
    assert cfg.num_attention_heads == 4
    assert cfg.num_hidden_layers == 2
    assert cfg.max_position_embeddings == 16
    assert cfg.vocab_size == 1000
    assert cfg.intermediate_size == 128
    assert cfg.layer_norm_eps == 1e-5
    assert cfg.hidden_act == "gelu_new"
    assert cfg.tie_word_embeddings is True
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 8


def test_gpt2_new_style_aliases_are_accepted():
    legacy = {
        "model_type": "gpt2",
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "max_position_embeddings": 8,
        "vocab_size": 24,
        "layer_norm_eps": 1e-5,
        "hidden_act": "gelu_new",
    }
    cfg = ModelConfig.from_dict(legacy)
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers) == (16, 2, 1)
    assert cfg.max_position_embeddings == 8
    assert cfg.vocab_size == 24


def test_gpt2_n_inner_default_and_null():
    raw = {k: v for k, v in GPT2_JSON.items() if k != "n_inner"}
    assert ModelConfig.from_dict(raw).intermediate_size == 4 * 32
    assert ModelConfig.from_dict({**GPT2_JSON, "n_inner": None}).intermediate_size == 4 * 32


def test_gpt2_activation_function_falls_back_to_hidden_act():
    raw = {k: v for k, v in GPT2_JSON.items() if k != "activation_function"}
    assert ModelConfig.from_dict({**raw, "hidden_act": "gelu"}).hidden_act == "gelu"
    assert ModelConfig.from_dict(raw).hidden_act == "gelu_new"


def test_gpt2_tie_word_embeddings_defaults_true():
    raw = {k: v for k, v in GPT2_JSON.items() if k != "tie_word_embeddings"}
    assert ModelConfig.from_dict(raw).tie_word_embeddings is True
    assert ModelConfig.from_dict({**GPT2_JSON, "tie_word_embeddings": False}).tie_word_embeddings is False


# ---------------------------------------------------------------------------
# constructors agree
# ---------------------------------------------------------------------------


def test_from_dict_from_json_and_from_dir_agree(tmp_path):
    dict_cfg = ModelConfig.from_dict(LLAMA_JSON, source="<test>")
    json_path = write_config(tmp_path, LLAMA_JSON)
    json_cfg = ModelConfig.from_json(json_path)
    dir_cfg = ModelConfig.from_dir(tmp_path)
    assert dict_cfg == json_cfg == dir_cfg
    assert dict_cfg.digest == json_cfg.digest == dir_cfg.digest
    assert json_cfg.raw == LLAMA_JSON


def test_from_dir_missing_config_names_the_directory(tmp_path):
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dir(tmp_path)
    assert str(tmp_path) in str(err.value)
    assert "config.json" in str(err.value)


def test_from_json_missing_file(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_json(missing)
    assert str(missing) in str(err.value)


def test_from_json_malformed_json(tmp_path):
    path = write_config(tmp_path, "{not json")
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_json(path)
    assert str(path) in str(err.value)
    assert "JSON" in str(err.value)


def test_from_json_rejects_non_object(tmp_path):
    path = write_config(tmp_path, [1, 2, 3])
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_json(path)
    assert "JSON object" in str(err.value)


# ---------------------------------------------------------------------------
# digest / equality
# ---------------------------------------------------------------------------


def test_digest_is_stable_and_order_independent():
    reordered = {key: LLAMA_JSON[key] for key in reversed(list(LLAMA_JSON))}
    with_extra = {**LLAMA_JSON, "unknown_extra": [1, 2, 3]}
    base = ModelConfig.from_dict(LLAMA_JSON)
    assert base.digest == ModelConfig.from_dict(reordered).digest
    assert base.digest == ModelConfig.from_dict(with_extra).digest
    assert len(base.digest) == 16
    assert all(c in "0123456789abcdef" for c in base.digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 32001),
        ("hidden_size", 128),
        ("num_hidden_layers", 3),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 2),
        ("intermediate_size", 256),
        ("hidden_act", "gelu"),
        ("max_position_embeddings", 256),
        ("tie_word_embeddings", True),
        ("rope_theta", 500000.0),
        ("head_dim", 16),
        ("rms_norm_eps", 1e-5),  # llama's name for layer_norm_eps
        ("model_type", "mistral"),
    ],
)
def test_digest_changes_with_every_graph_field(field, value):
    base = ModelConfig.from_dict(LLAMA_JSON)
    other_raw = {**LLAMA_JSON, field: value}
    other = ModelConfig.from_dict(other_raw)
    assert other.digest != base.digest


def test_raw_is_excluded_from_equality_and_hash():
    base = ModelConfig.from_dict(LLAMA_JSON)
    other = ModelConfig.from_dict({**LLAMA_JSON, "extra": 1})
    assert base == other
    assert hash(base) == hash(other)


def test_config_is_frozen():
    cfg = ModelConfig.from_dict(LLAMA_JSON)
    with pytest.raises(FrozenInstanceError):
        cfg.hidden_size = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# explicit errors
# ---------------------------------------------------------------------------


def test_unknown_model_type_lists_supported_types():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "model_type": "bloom"}, source="m.json")
    message = str(err.value)
    assert "bloom" in message
    assert "m.json" in message
    assert "gpt2" in message and "llama" in message


def test_missing_model_type():
    raw = {k: v for k, v in LLAMA_JSON.items() if k != "model_type"}
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict(raw, source="cfg.json")
    assert "model_type" in str(err.value)


@pytest.mark.parametrize(
    "field",
    ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
     "intermediate_size", "max_position_embeddings"],
)
def test_missing_required_llama_field_names_field_and_source(field, tmp_path):
    raw = {k: v for k, v in LLAMA_JSON.items() if k != field}
    path = write_config(tmp_path, raw)
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dir(tmp_path)
    message = str(err.value)
    assert field in message
    assert str(path) in message


@pytest.mark.parametrize("bad", [0, -4])
def test_non_positive_sizes_rejected(bad):
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "hidden_size": bad})
    assert "hidden_size" in str(err.value)
    assert "positive" in str(err.value)


@pytest.mark.parametrize("bad", ["64", 64.5, None, True])
def test_non_integer_sizes_rejected(bad):
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "num_hidden_layers": bad})
    assert "num_hidden_layers" in str(err.value)


def test_hidden_size_must_divide_heads():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "hidden_size": 65})
    assert "divisible" in str(err.value)


def test_heads_must_divide_kv_heads():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "num_key_value_heads": 3})
    assert "num_key_value_heads" in str(err.value)
    assert "divisible" in str(err.value)


@pytest.mark.parametrize("bad", [0.0, -1e-6])
def test_layer_norm_eps_must_be_positive(bad):
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "rms_norm_eps": bad})
    assert "rms_norm_eps" in str(err.value)


def test_invalid_head_dim_override():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "head_dim": 0})
    assert "head_dim" in str(err.value)


def test_non_boolean_tie_word_embeddings():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "tie_word_embeddings": "yes"})
    assert "tie_word_embeddings" in str(err.value)


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {"type": "linear", "factor": 4.0},
        {"type": "dynamic", "factor": 2.0},
        {"rope_type": "yarn", "factor": 1.0},
        [1, 2],
    ],
)
def test_rope_scaling_is_rejected(rope_scaling):
    with pytest.raises(HFUnsupportedError) as err:
        ModelConfig.from_dict({**LLAMA_JSON, "rope_scaling": rope_scaling})
    assert "rope_scaling" in str(err.value)
    assert "v1" in str(err.value)


@pytest.mark.parametrize("value", [None, {}])
def test_empty_rope_scaling_is_accepted(value):
    cfg = ModelConfig.from_dict({**LLAMA_JSON, "rope_scaling": value})
    assert cfg.family == "llama"


def test_non_dict_payload():
    with pytest.raises(HFConfigError) as err:
        ModelConfig.from_dict(["not", "a", "config"], source="x.json")
    assert "x.json" in str(err.value)
