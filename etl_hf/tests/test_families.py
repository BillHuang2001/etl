"""Tests for :mod:`etl_hf.families` — the family registry.

The registry is the one place families are enumerated, so these tests are about
*consistency*, not numerics: the registry must agree with
:data:`etl_hf.config.FAMILY_BY_MODEL_TYPE` (what config parsing uses), with each
family module's own ``FAMILY``/``MODEL_TYPES`` attributes, and with itself
(keys ↔ ``FamilySpec.name``). Unknown families must fail with an explicit
``HFUnsupportedError`` that names the supported set.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from etl_hf import config as config_module
from etl_hf import families, ops
from etl_hf.config import ModelConfig
from etl_hf.errors import HFUnsupportedError
from etl_hf.families import FAMILIES, FamilySpec, family_for_model_type, get_family
from etl_hf.models import gpt2 as gpt2_module
from etl_hf.models import llama as llama_module

#: The v1 family set, and the modules that implement it.
_MODULES = {llama_module.FAMILY: llama_module, gpt2_module.FAMILY: gpt2_module}

#: Minimal HF-shaped configs (only the fields the registry-relevant constructors
#: need — this file exercises routing, never numerics).
_LLAMA_JSON = {
    "model_type": "llama",
    "vocab_size": 16,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 1,
    "intermediate_size": 16,
    "max_position_embeddings": 8,
}
_GPT2_JSON = {
    "model_type": "gpt2",
    "n_vocab": 16,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 1,
    "n_inner": 16,
    "n_positions": 8,
}


# ---------------------------------------------------------------------------
# registry contents
# ---------------------------------------------------------------------------
def test_registry_has_exactly_the_v1_families():
    assert set(FAMILIES) == {"llama", "gpt2"}
    assert sorted(FAMILIES) == ["gpt2", "llama"]


def test_registry_keys_match_spec_names():
    for key, spec in FAMILIES.items():
        assert isinstance(spec, FamilySpec)
        assert key == spec.name
        assert spec.name == _MODULES[key].FAMILY


def test_registry_is_read_only():
    """A mapping-proxy registry cannot be mutated at runtime (no silent aliasing)."""
    with pytest.raises(TypeError):
        FAMILIES["new_family"] = FAMILIES["llama"]  # type: ignore[index]
    assert set(FAMILIES) == {"llama", "gpt2"}


def test_every_spec_field_is_callable_or_hashable():
    for spec in FAMILIES.values():
        assert isinstance(spec.name, str) and spec.name
        assert isinstance(spec.model_types, tuple) and spec.model_types
        assert all(isinstance(model_type, str) for model_type in spec.model_types)
        for function in (
            spec.canonicalize_weights,
            spec.param_specs,
            spec.graph,
            spec.reference,
        ):
            assert callable(function)
            hash(function)  # registry values are usable as dict/set members
        hash(spec)  # frozen + slots ⇒ hashable


def test_spec_is_frozen():
    spec = FAMILIES["llama"]
    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_spec_functions_are_the_family_module_functions():
    """The registry holds the modules' own functions — never wrappers that drift."""
    for key, module in _MODULES.items():
        spec = FAMILIES[key]
        assert spec.canonicalize_weights is module.canonicalize_weights
        assert spec.param_specs is module.param_specs
        assert spec.graph is module.graph
        assert spec.reference is module.reference


# ---------------------------------------------------------------------------
# agreement with config parsing
# ---------------------------------------------------------------------------
def test_registry_keys_agree_with_config_family_map():
    assert set(FAMILIES) == set(config_module.FAMILY_BY_MODEL_TYPE.values())


def test_model_types_agree_with_config_family_map():
    for key, spec in FAMILIES.items():
        expected = tuple(
            sorted(
                model_type
                for model_type, family in config_module.FAMILY_BY_MODEL_TYPE.items()
                if family == key
            )
        )
        assert spec.model_types == expected, key


def test_model_types_agree_with_each_module():
    """A module that declares ``MODEL_TYPES`` must agree with the registry."""
    for key, module in _MODULES.items():
        declared = getattr(module, "MODEL_TYPES", None)
        if declared is not None:
            assert tuple(sorted(declared)) == FAMILIES[key].model_types


def test_family_specs_accept_the_configs_own_family():
    """Each registry entry drives a real build path for a tiny config of its family."""
    for raw in (_LLAMA_JSON, _GPT2_JSON):
        cfg = ModelConfig.from_dict(raw)
        spec = get_family(cfg.family)
        assert spec.name == cfg.family
        assert cfg.model_type in spec.model_types
        specs = spec.param_specs(cfg)
        assert specs and all(
            isinstance(value, type(specs[next(iter(specs))]))
            for value in specs.values()
        )


def test_llama_spec_covers_mistral_and_qwen2():
    assert set(get_family("llama").model_types) == {"llama", "mistral", "qwen2"}


# ---------------------------------------------------------------------------
# the tied-weight contract at the registry level
# ---------------------------------------------------------------------------
def test_tied_specs_drop_the_head_for_every_family():
    """Both families honour ``tie_word_embeddings``: tied drops ``lm_head.weight``."""
    cases = (
        (_LLAMA_JSON, {"tie_word_embeddings": True}, 2 + 9),
        (_GPT2_JSON, {}, 16 + 4),
    )
    for raw, overrides, expected_tied in cases:
        cfg = ModelConfig.from_dict(dict(raw, **overrides))
        spec = get_family(cfg.family)
        assert cfg.tie_word_embeddings is True
        tied = spec.param_specs(cfg)
        untied_raw = dict(raw, **overrides)
        untied_raw["tie_word_embeddings"] = False
        untied_cfg = ModelConfig.from_dict(untied_raw)
        untied = spec.param_specs(untied_cfg)
        assert len(tied) == expected_tied, cfg.family
        assert len(untied) == expected_tied + 1, cfg.family
        assert set(untied) - set(tied) == {"lm_head.weight"}, cfg.family


def test_gpt2_spec_covers_only_gpt2():
    assert get_family("gpt2").model_types == ("gpt2",)


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["llama", "gpt2"])
def test_get_family_returns_the_registry_entry(name):
    assert get_family(name) is FAMILIES[name]


@pytest.mark.parametrize("name", ["bert", "", "Llama", None, 3])
def test_get_family_unknown_raises_naming_the_registry(name):
    with pytest.raises(HFUnsupportedError) as excinfo:
        get_family(name)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "unsupported model family" in message
    for supported in sorted(FAMILIES):
        assert supported in message
    assert "families.py" in message


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("llama", "llama"),
        ("mistral", "llama"),
        ("qwen2", "llama"),
        ("gpt2", "gpt2"),
    ],
)
def test_family_for_model_type_resolves_every_supported_type(model_type, expected):
    spec = family_for_model_type(model_type)
    assert spec is FAMILIES[expected]
    assert model_type in spec.model_types


@pytest.mark.parametrize("model_type", ["bert", "gpt3", "", None, 7])
def test_family_for_model_type_unknown_raises(model_type):
    with pytest.raises(HFUnsupportedError) as excinfo:
        family_for_model_type(model_type)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "unsupported HF model_type" in message
    for supported in sorted(config_module.FAMILY_BY_MODEL_TYPE):
        assert supported in message


def test_family_for_model_type_agrees_with_model_config_family():
    """Registry routing and config parsing must never disagree."""
    for model_type in config_module.FAMILY_BY_MODEL_TYPE:
        raw = dict(
            _LLAMA_JSON if model_type in ("llama", "mistral", "qwen2") else _GPT2_JSON
        )
        raw["model_type"] = model_type
        cfg = ModelConfig.from_dict(raw)
        assert family_for_model_type(model_type).name == cfg.family


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------
def test_registry_import_does_not_pull_optional_dependencies():
    """``etl_hf.families`` is numpy-only: no torch/transformers/iree/safetensors."""
    import subprocess
    import sys

    code = (
        "import sys, etl_hf.families;"
        "bad = [m for m in ('torch', 'transformers', 'safetensors') "
        "if m in sys.modules];"
        "bad += [m for m in ('iree', 'iree.compiler') if m in sys.modules];"
        "print(','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""


def test_family_specs_are_usable_keys():
    """Registry entries are hashable and compare by value (frozen dataclass)."""
    spec = FAMILIES["llama"]
    assert spec == get_family("llama")
    assert {spec: 1}[get_family("llama")] == 1
    assert len({FAMILIES["llama"], FAMILIES["gpt2"], get_family("llama")}) == 2


def test_registry_does_not_expose_helper_builders():
    """Only the public surface is exported by the module."""
    assert set(families.__all__) == {
        "FAMILIES",
        "FamilySpec",
        "family_for_model_type",
        "get_family",
        "resolve_family",
    }
    # the activation table lives in ops, not here — no duplicate source of truth
    assert not hasattr(families, "ACTIVATIONS")
    assert "ACTIVATIONS" in ops.__all__
    assert np is not None  # numpy is a hard dependency of the family modules
