"""Parity + canonicalisation tests for the GPT-2 family (``etl_hf.models.gpt2``).

Self-contained: no network required (the one real-checkpoint case is guarded by
:func:`etl_hf.tests.hf_download.require_model` and skips offline), no torch, no
locally written checkpoint files. Every case builds a small synthetic GPT-2
config + parameter set from :func:`gpt2.param_specs`, stages the real graph
through ``etl.evaluate`` on the numpy interpreter backend, and compares against
:func:`gpt2.reference` (a pure-numpy, per-head-loop formulation of the same
forward). The canonicalisation cases additionally drive the loader from a
hand-built **HF-shaped** state dict (``c_attn`` fused, ``Conv1D`` orientation,
optional ``transformer.`` prefix) so that a wrong QKV split or a stray transpose
cannot pass unnoticed.

Section 4 also pins the **tied-weight contract**: ``tie_word_embeddings=True``
means the output head IS ``embed.weight``, so there is no ``lm_head.weight``
key, exactly one embedding array, and no extra graph input.
"""

from __future__ import annotations

import numpy as np
import pytest

import etl
from etl_hf import ops
from etl_hf import reference as numpy_reference
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFWeightError
from etl_hf.load import LoadedModel, load_hf, validate_params
from etl_hf.models import gpt2
from etl_hf.pipeline import build, predict, run
from etl_hf.program import forward_fn, param_specs, reference_logits
from etl_hf.tests import hf_download

#: Base synthetic config: tiny but non-degenerate (non-square MLP, 4 heads).
BASE_CONFIG: dict = {
    "model_type": "gpt2",
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 2,
    "n_positions": 16,
    "n_vocab": 32,
    "n_inner": 32,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
    "tie_word_embeddings": True,
}

#: Type alias for the flat canonical parameter dict the family interface uses.
Params = dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# helpers: configs, synthetic parameters, graph execution
# ---------------------------------------------------------------------------
def make_config(**overrides) -> ModelConfig:
    """Build a small GPT-2 :class:`ModelConfig`, overriding raw HF keys."""
    raw = dict(BASE_CONFIG)
    raw.update(overrides)
    return ModelConfig.from_dict(raw)


def make_params(cfg: ModelConfig, seed: int = 0) -> Params:
    """Synthesise a canonical parameter dict matching ``cfg`` (seeded, small)."""
    rng = np.random.default_rng(seed)
    return {
        name: np.asarray(rng.standard_normal(spec.shape) * 0.1, dtype=np.float32)
        for name, spec in gpt2.param_specs(cfg).items()
    }


def make_ids(cfg: ModelConfig, batch: int, seq: int, seed: int = 1) -> np.ndarray:
    """Synthesise an ``(batch, seq)`` int32 token-id array inside the vocab."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)


def run_graph(cfg: ModelConfig, params: Params, ids: np.ndarray) -> np.ndarray:
    """Trace + build + run the GPT-2 graph on the numpy backend; return logits."""
    forward = gpt2.graph(cfg, int(ids.shape[1]), int(ids.shape[0]))
    return np.asarray(etl.evaluate(forward, params, ids).numpy(), dtype=np.float32)


def _loaded(cfg: ModelConfig, params: Params, family: str = "gpt2") -> LoadedModel:
    """Wrap a config + canonical params as the public :class:`LoadedModel` value.

    :class:`~etl_hf.load.LoadedModel` is plain immutable data (config + params +
    family + source directory), so a test that synthesised its own params can
    build one directly and use the host-facing :func:`etl_hf.predict` shorthand.
    """
    return LoadedModel(config=cfg, params=params, family=family, model_dir=None)


def assert_parity(
    cfg: ModelConfig,
    params: Params,
    ids: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> float:
    """Assert graph-vs-reference parity; return the measured max-abs difference."""
    got = run_graph(cfg, params, ids)
    want = gpt2.reference(cfg, params, ids)
    assert got.shape == (ids.shape[0], ids.shape[1], cfg.vocab_size)
    assert got.dtype == np.float32
    assert want.shape == got.shape and want.dtype == np.float32
    max_abs = float(np.max(np.abs(got - want)))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    return max_abs


# ---------------------------------------------------------------------------
# helpers: HF-shaped state dicts (fused Conv1D projections, HF key names)
# ---------------------------------------------------------------------------
def make_hf_state_dict(
    cfg: ModelConfig,
    seed: int = 7,
    *,
    transformer_prefix: bool = True,
    include_lm_head: bool = True,
    tie_lm_head: bool = True,
    include_attention_buffers: bool = False,
) -> dict[str, np.ndarray]:
    """Build a synthetic HF ``GPT2LMHeadModel`` state dict for ``cfg``.

    Args:
        cfg: the GPT-2 config the checkpoint is "written for".
        seed: RNG seed for every tensor.
        transformer_prefix: emit ``transformer.``-prefixed keys (older exports)
            instead of the bare block/embedding keys of the ``gpt2`` repo's
            ``model.safetensors``. ``lm_head.weight`` is never prefixed.
        include_lm_head: include the (separate) ``lm_head.weight`` tensor.
        tie_lm_head: make that tensor identical to ``wte.weight`` (a consistent
            tied checkpoint) instead of an independent matrix.
        include_attention_buffers: add the derived ``attn.bias`` (bool) and
            ``attn.masked_bias`` (scalar) buffers older exports wrote to disk.

    Returns:
        ``dict`` of HF key → float32 (or bool) array.
    """
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    vocab = cfg.vocab_size

    def key(base: str) -> str:
        """Apply the optional ``transformer.`` prefix (never to ``lm_head``)."""
        if base.startswith("lm_head.") or not transformer_prefix:
            return base
        return "transformer." + base

    def normal(shape) -> np.ndarray:
        """One seeded tensor of the given shape (``~N(0, 0.2)``: O(1) activations)."""
        return np.asarray(rng.standard_normal(shape) * 0.2, dtype=np.float32)

    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {
        key("wte.weight"): normal((vocab, hidden)),
        key("wpe.weight"): normal((cfg.max_position_embeddings, hidden)),
    }
    for index in range(cfg.num_hidden_layers):
        source = f"h.{index}."
        state[key(source + "ln_1.weight")] = normal((hidden,))
        state[key(source + "ln_1.bias")] = normal((hidden,))
        state[key(source + "attn.c_attn.weight")] = normal((hidden, 3 * hidden))
        state[key(source + "attn.c_attn.bias")] = normal((3 * hidden,))
        state[key(source + "attn.c_proj.weight")] = normal((hidden, hidden))
        state[key(source + "attn.c_proj.bias")] = normal((hidden,))
        state[key(source + "ln_2.weight")] = normal((hidden,))
        state[key(source + "ln_2.bias")] = normal((hidden,))
        state[key(source + "mlp.c_fc.weight")] = normal((hidden, intermediate))
        state[key(source + "mlp.c_fc.bias")] = normal((intermediate,))
        state[key(source + "mlp.c_proj.weight")] = normal((intermediate, hidden))
        state[key(source + "mlp.c_proj.bias")] = normal((hidden,))
        if include_attention_buffers:
            length = cfg.max_position_embeddings
            state[key(source + "attn.bias")] = np.tril(
                np.ones((1, 1, length, length), dtype=bool)
            )
            state[key(source + "attn.masked_bias")] = np.asarray(-1e4, dtype=np.float32)
    state[key("ln_f.weight")] = normal((hidden,))
    state[key("ln_f.bias")] = normal((hidden,))
    if include_lm_head:
        state["lm_head.weight"] = (
            state[key("wte.weight")].copy()
            if tie_lm_head
            else normal((vocab, hidden))
        )
    return state


def hf_tensor(state: dict[str, np.ndarray], base: str) -> np.ndarray:
    """Fetch an HF tensor by its prefix-free name (either spelling)."""
    for candidate in (base, "transformer." + base):
        if candidate in state:
            return np.asarray(state[candidate], dtype=np.float32)
    raise AssertionError(f"state dict has neither {base!r} nor {'transformer.' + base!r}")


def fused_qkv_reference(
    cfg: ModelConfig, state: dict[str, np.ndarray], ids: np.ndarray
) -> np.ndarray:
    """GPT-2 forward that keeps ``c_attn`` **fused** (HF's own layout).

    Written independently of :func:`gpt2.reference`: the QKV projection is one
    ``(hidden, 3 * hidden)`` matmul whose ``[0:H] / [H:2H] / [2H:3H]`` column
    blocks are split here, so a permuted or offset split in
    :func:`gpt2.canonicalize_weights` shows up as a numeric mismatch. Uses only
    :mod:`etl_hf.reference` primitives for the numerics.
    """
    batch, seq = int(ids.shape[0]), int(ids.shape[1])
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    eps = cfg.layer_norm_eps
    scale = head_dim**-0.5
    act = numpy_reference.resolve_activation(cfg.hidden_act)
    mask = numpy_reference.additive_causal_mask(seq)

    tokens = hf_tensor(state, "wte.weight")[ids]
    positions = hf_tensor(state, "wpe.weight")[:seq]
    x = np.asarray(tokens + positions[None, :, :], dtype=np.float32)

    def to_heads(projected: np.ndarray) -> np.ndarray:
        """``(b, s, n * hd) -> (b, n, s, hd)``."""
        return projected.reshape(batch, seq, heads, head_dim).transpose(0, 2, 1, 3)

    for index in range(cfg.num_hidden_layers):
        source = f"h.{index}."
        normed = numpy_reference.layer_norm(
            x,
            hf_tensor(state, source + "ln_1.weight"),
            hf_tensor(state, source + "ln_1.bias"),
            eps,
        )
        qkv = np.dot(
            normed, hf_tensor(state, source + "attn.c_attn.weight")
        ) + hf_tensor(state, source + "attn.c_attn.bias")
        query = qkv[..., 0:hidden]
        key = qkv[..., hidden : 2 * hidden]
        value = qkv[..., 2 * hidden : 3 * hidden]
        context = numpy_reference.attention(
            to_heads(query), to_heads(key), to_heads(value), mask, scale
        )
        x = np.asarray(
            x
            + np.dot(context, hf_tensor(state, source + "attn.c_proj.weight"))
            + hf_tensor(state, source + "attn.c_proj.bias"),
            dtype=np.float32,
        )
        normed = numpy_reference.layer_norm(
            x,
            hf_tensor(state, source + "ln_2.weight"),
            hf_tensor(state, source + "ln_2.bias"),
            eps,
        )
        activated = act(
            np.dot(normed, hf_tensor(state, source + "mlp.c_fc.weight"))
            + hf_tensor(state, source + "mlp.c_fc.bias")
        )
        x = np.asarray(
            x
            + np.dot(activated, hf_tensor(state, source + "mlp.c_proj.weight"))
            + hf_tensor(state, source + "mlp.c_proj.bias"),
            dtype=np.float32,
        )
    normed = numpy_reference.layer_norm(
        x, hf_tensor(state, "ln_f.weight"), hf_tensor(state, "ln_f.bias"), eps
    )
    return np.asarray(
        np.dot(normed, hf_tensor(state, "lm_head.weight").T), dtype=np.float32
    )


# ===========================================================================
# 1. parity (graph vs independent numpy reference)
# ===========================================================================
def test_parity_small_model():
    """The reference config (2 layers, 4 heads, seq 4, batch 1) matches tightly."""
    cfg = make_config()
    params = make_params(cfg)
    ids = make_ids(cfg, batch=1, seq=4)
    max_abs = assert_parity(cfg, params, ids)
    assert max_abs < 1e-5


@pytest.mark.parametrize("n_layer", [1, 2, 3])
def test_parity_layer_counts(n_layer):
    """1, 2 and 3 transformer blocks all match the numpy reference."""
    cfg = make_config(n_layer=n_layer)
    params = make_params(cfg, seed=n_layer)
    ids = make_ids(cfg, batch=1, seq=4)
    assert_parity(cfg, params, ids)


def test_parity_batch_two():
    """A batch of two sequences matches (per-row embedding/attention indices)."""
    cfg = make_config()
    params = make_params(cfg)
    ids = make_ids(cfg, batch=2, seq=4)
    assert_parity(cfg, params, ids)


@pytest.mark.parametrize("seq_len", [1, 4, 8])
def test_parity_seq_lengths(seq_len):
    """Sequence lengths 1/4/8 match (seq_len=1 is a single-row causal mask)."""
    cfg = make_config()
    params = make_params(cfg, seed=seq_len)
    ids = make_ids(cfg, batch=1, seq=seq_len)
    assert_parity(cfg, params, ids)


def test_parity_seq_one_batch_two():
    """``seq_len=1`` with a real batch axis (degenerate-but-valid attention)."""
    cfg = make_config()
    params = make_params(cfg)
    ids = make_ids(cfg, batch=2, seq=1)
    assert_parity(cfg, params, ids)


def test_parity_untied_head():
    """An untied config (independent ``lm_head``) matches as well."""
    cfg = make_config(tie_word_embeddings=False)
    params = make_params(cfg)
    ids = make_ids(cfg, batch=1, seq=4)
    assert_parity(cfg, params, ids)


# ===========================================================================
# 2. activation variants
# ===========================================================================
@pytest.mark.parametrize("activation", ["gelu_new", "gelu", "relu", "silu"])
def test_parity_activations(activation):
    """Every supported GPT-2 activation selection matches the numpy reference."""
    cfg = make_config(activation_function=activation)
    assert cfg.hidden_act == activation
    params = make_params(cfg)
    ids = make_ids(cfg, batch=1, seq=4)
    assert_parity(cfg, params, ids)


def test_activation_export_safety_flags():
    """Exact-erf ``gelu`` is numpy-backend-only; ``gelu_new``/``relu``/``silu`` export."""
    assert ops.activation_is_export_safe("gelu") is False
    assert ops.activation_is_export_safe("gelu_new") is True
    assert ops.activation_is_export_safe("gelu_pytorch_tanh") is True
    assert ops.activation_is_export_safe("relu") is True
    assert ops.activation_is_export_safe("silu") is True


def test_gelu_new_is_the_config_default():
    """A config without ``activation_function`` defaults to tanh GELU (``gelu_new``)."""
    raw = dict(BASE_CONFIG)
    del raw["activation_function"]
    cfg = ModelConfig.from_dict(raw)
    assert cfg.hidden_act == "gelu_new"
    assert_parity(cfg, make_params(cfg), make_ids(cfg, batch=1, seq=4))


def test_unknown_activation_is_explicit():
    """An unsupported ``activation_function`` raises instead of falling back."""
    from etl_hf.errors import HFUnsupportedError

    cfg = make_config(activation_function="quick_gelu")
    with pytest.raises(HFUnsupportedError):
        gpt2.graph(cfg, 4)


# ===========================================================================
# 3. canonicalisation from an HF state dict
# ===========================================================================
def test_canonical_keys_and_shapes():
    """``canonicalize_weights`` emits exactly the ``param_specs`` keys/shapes."""
    cfg = make_config()
    canonical = gpt2.canonicalize_weights(cfg, make_hf_state_dict(cfg))
    specs = gpt2.param_specs(cfg)
    assert set(canonical) == set(specs)
    for name, spec in specs.items():
        array = canonical[name]
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float32
        assert array.flags["C_CONTIGUOUS"]
        assert tuple(array.shape) == tuple(spec.shape)


def test_canonicalize_splits_fused_qkv():
    """``c_attn`` is split into the ``[0:H] / [H:2H] / [2H:3H]`` column blocks."""
    cfg = make_config(n_layer=1)
    hidden = cfg.hidden_size
    state = make_hf_state_dict(cfg)
    fused = np.arange(hidden * 3 * hidden, dtype=np.float32).reshape(hidden, 3 * hidden)
    fused_bias = np.arange(3 * hidden, dtype=np.float32) + 100.0
    state["transformer.h.0.attn.c_attn.weight"] = fused
    state["transformer.h.0.attn.c_attn.bias"] = fused_bias

    canonical = gpt2.canonicalize_weights(cfg, state)
    for name, start in (("q", 0), ("k", hidden), ("v", 2 * hidden)):
        np.testing.assert_array_equal(
            canonical[f"layers.0.attn.{name}.weight"],
            fused[:, start : start + hidden],
        )
        np.testing.assert_array_equal(
            canonical[f"layers.0.attn.{name}.bias"],
            fused_bias[start : start + hidden],
        )
    # The three blocks are numerically distinct, so a mis-ordered split cannot pass.
    assert not np.array_equal(
        canonical["layers.0.attn.q.weight"], canonical["layers.0.attn.v.weight"]
    )


def test_canonicalize_matches_fused_matrix_multiply():
    """A graph built from the split weights equals one using fused ``h @ c_attn``."""
    cfg = make_config(n_layer=1)
    state = make_hf_state_dict(cfg, seed=11)
    params = gpt2.canonicalize_weights(cfg, state)
    ids = make_ids(cfg, batch=1, seq=4)

    got = run_graph(cfg, params, ids)
    want = fused_qkv_reference(cfg, state, ids)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    # And the identity holds at the matmul level the graph relies on.
    normed = np.asarray(
        np.random.default_rng(3).standard_normal((2, cfg.hidden_size)), dtype=np.float32
    )
    fused = hf_tensor(state, "h.0.attn.c_attn.weight")
    projected = np.dot(normed, fused)
    hidden = cfg.hidden_size
    for name, start in (("q", 0), ("k", hidden), ("v", 2 * hidden)):
        np.testing.assert_allclose(
            np.dot(normed, params[f"layers.0.attn.{name}.weight"]),
            projected[:, start : start + hidden],
            rtol=1e-6,
            atol=1e-6,
        )


def test_canonicalize_prefix_forms_agree():
    """``transformer.``-prefixed and bare-block state dicts canonicalise identically."""
    cfg = make_config()
    prefixed = gpt2.canonicalize_weights(
        cfg, make_hf_state_dict(cfg, seed=5, transformer_prefix=True)
    )
    bare = gpt2.canonicalize_weights(
        cfg, make_hf_state_dict(cfg, seed=5, transformer_prefix=False)
    )
    assert set(prefixed) == set(bare)
    for name in prefixed:
        np.testing.assert_array_equal(prefixed[name], bare[name])


def test_canonicalize_rejects_duplicate_prefix_forms():
    """Both spellings of one key in a single dict is an explicit error."""
    cfg = make_config()
    state = make_hf_state_dict(cfg, transformer_prefix=False)
    state["transformer.wte.weight"] = state["wte.weight"].copy()
    with pytest.raises(HFWeightError, match="duplicate"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_ignores_attention_buffers():
    """The derived ``attn.bias`` / ``attn.masked_bias`` buffers are ignored."""
    cfg = make_config()
    state = make_hf_state_dict(
        cfg, transformer_prefix=True, include_attention_buffers=True
    )
    # Mix spellings: one prefixed buffer, one bare (as some exports do).
    state["h.0.attn.masked_bias"] = state.pop("transformer.h.0.attn.masked_bias")
    canonical = gpt2.canonicalize_weights(cfg, state)
    assert set(canonical) == set(gpt2.param_specs(cfg))


def test_canonicalize_tied_without_head_aliases_embedding():
    """Tied config + no ``lm_head.weight`` → no head key at all (the head IS the embedding)."""
    cfg = make_config(tie_word_embeddings=True)
    state = make_hf_state_dict(cfg, include_lm_head=False)
    canonical = gpt2.canonicalize_weights(cfg, state)
    assert "lm_head.weight" not in canonical
    assert set(canonical) == set(gpt2.param_specs(cfg))
    np.testing.assert_array_equal(
        canonical["embed.weight"], hf_tensor(state, "wte.weight")
    )
    assert canonical["embed.weight"].shape == (cfg.vocab_size, cfg.hidden_size)


def test_canonicalize_tied_with_consistent_head():
    """Tied config + a ``lm_head.weight`` equal to ``wte`` → verified, consumed, dropped."""
    cfg = make_config(tie_word_embeddings=True)
    state = make_hf_state_dict(cfg, include_lm_head=True, tie_lm_head=True)
    # No "unexpected key" error: the shipped copy was consumed, not ignored.
    canonical = gpt2.canonicalize_weights(cfg, state)
    assert "lm_head.weight" not in canonical
    assert set(canonical) == set(gpt2.param_specs(cfg))


def test_canonicalize_tied_with_inconsistent_head_raises():
    """Tied config + a ``lm_head.weight`` that disagrees with ``wte`` → error."""
    cfg = make_config(tie_word_embeddings=True)
    state = make_hf_state_dict(cfg, include_lm_head=True, tie_lm_head=False)
    with pytest.raises(HFWeightError, match="tie_word_embeddings"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_untied_head_is_transposed():
    """Untied config transposes the HF ``(vocab, hidden)`` head to ``(hidden, vocab)``."""
    cfg = make_config(tie_word_embeddings=False)
    state = make_hf_state_dict(cfg, include_lm_head=True, tie_lm_head=False)
    canonical = gpt2.canonicalize_weights(cfg, state)
    np.testing.assert_array_equal(
        canonical["lm_head.weight"], hf_tensor(state, "lm_head.weight").T
    )


def test_canonicalize_untied_without_head_raises():
    """Untied config with no ``lm_head.weight`` is an explicit error."""
    cfg = make_config(tie_word_embeddings=False)
    state = make_hf_state_dict(cfg, include_lm_head=False)
    with pytest.raises(HFWeightError, match="lm_head.weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_rejects_unexpected_key():
    """A leftover key is rejected and listed."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    state["transformer.h.0.attn.c_attn_weight"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(HFWeightError, match="c_attn_weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_rejects_missing_key():
    """A missing required key is named (in both accepted spellings)."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    del state["transformer.h.1.mlp.c_fc.weight"]
    with pytest.raises(HFWeightError, match="h.1.mlp.c_fc.weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_rejects_wrong_shape():
    """A tensor whose shape disagrees with the config is rejected."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    state["transformer.h.0.attn.c_proj.weight"] = np.zeros(
        (cfg.hidden_size + 1, cfg.hidden_size), dtype=np.float32
    )
    with pytest.raises(HFWeightError, match="c_proj.weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_rejects_wrong_qkv_width():
    """A fused QKV width other than ``3 * hidden_size`` is rejected."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    hidden = cfg.hidden_size
    state["transformer.h.0.attn.c_attn.weight"] = np.zeros(
        (hidden, 2 * hidden), dtype=np.float32
    )
    with pytest.raises(HFWeightError, match="c_attn.weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_rejects_position_table_mismatch():
    """A ``wpe`` with more/fewer rows than ``max_position_embeddings`` is rejected."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    state["transformer.wpe.weight"] = np.zeros(
        (cfg.max_position_embeddings + 1, cfg.hidden_size), dtype=np.float32
    )
    with pytest.raises(HFWeightError, match="wpe.weight"):
        gpt2.canonicalize_weights(cfg, state)


def test_canonicalize_accepts_non_float32_dtypes():
    """float64 / float16 inputs are converted to contiguous float32."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    state["transformer.wte.weight"] = np.asarray(
        state["transformer.wte.weight"], dtype=np.float64
    )
    state["transformer.ln_f.bias"] = np.asarray(
        state["transformer.ln_f.bias"], dtype=np.float16
    )
    canonical = gpt2.canonicalize_weights(cfg, state)
    assert canonical["embed.weight"].dtype == np.float32
    assert canonical["final_norm.bias"].dtype == np.float32


def test_canonicalize_does_not_mutate_input():
    """The input state dict is left untouched."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    before = {name: value.copy() for name, value in state.items()}
    gpt2.canonicalize_weights(cfg, state)
    assert set(state) == set(before)
    for name, value in before.items():
        np.testing.assert_array_equal(state[name], value)


# ===========================================================================
# 4. param_specs / graph contract
# ===========================================================================
def test_param_specs_types_and_shapes():
    """Every spec is a float32 ``TensorSpec`` with the canonical shape."""
    cfg = make_config()
    specs = gpt2.param_specs(cfg)
    assert all(isinstance(spec, etl.TensorSpec) for spec in specs.values())
    hidden = cfg.hidden_size
    assert specs["embed.weight"].shape == (cfg.vocab_size, hidden)
    assert specs["pos_embed.weight"].shape == (cfg.max_position_embeddings, hidden)
    assert specs["layers.0.attn.q.weight"].shape == (hidden, hidden)
    assert specs["layers.0.attn.q.bias"].shape == (hidden,)
    assert specs["layers.0.mlp.fc1.weight"].shape == (hidden, cfg.intermediate_size)
    assert specs["layers.0.mlp.fc2.weight"].shape == (cfg.intermediate_size, hidden)
    assert specs["layers.1.attn.o.weight"].shape == (hidden, hidden)
    assert specs["final_norm.weight"].shape == (hidden,)
    # Tied (this config): the head reuses embed.weight, so there is no head spec.
    assert "lm_head.weight" not in specs
    for spec in specs.values():
        assert spec.dtype == np.float32


def test_param_specs_untied_keeps_the_head():
    """An untied config keeps the separate ``(hidden, vocab)`` head spec."""
    cfg = make_config(tie_word_embeddings=False)
    specs = gpt2.param_specs(cfg)
    assert specs["lm_head.weight"].shape == (cfg.hidden_size, cfg.vocab_size)
    assert specs["lm_head.weight"].dtype == np.float32
    assert len(specs) == len(gpt2.param_specs(make_config())) + 1


def test_param_specs_scale_with_layer_count():
    """``16 * num_hidden_layers + 4`` tied / ``+ 5`` untied parameters (no aliases)."""
    for layers in (1, 2, 3):
        cfg = make_config(n_layer=layers)
        assert cfg.tie_word_embeddings is True
        assert len(gpt2.param_specs(cfg)) == 16 * layers + 4
        untied = make_config(n_layer=layers, tie_word_embeddings=False)
        assert len(gpt2.param_specs(untied)) == 16 * layers + 5


def test_graph_output_shape_and_dtype():
    """The graph's logits are ``(batch, seq, vocab)`` float32."""
    cfg = make_config()
    params = make_params(cfg)
    ids = make_ids(cfg, batch=2, seq=5)
    logits = run_graph(cfg, params, ids)
    assert logits.shape == (2, 5, cfg.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))


def test_graph_takes_weights_as_inputs_only():
    """The traced graph has exactly ``len(param_specs) + 1`` tensor inputs.

    This config is tied, so the output head is **not** among them: the one
    ``embed.weight`` input feeds both the gather and the projection.
    """
    cfg = make_config()
    specs = gpt2.param_specs(cfg)
    assert "lm_head.weight" not in specs
    forward = gpt2.graph(cfg, seq_len=4, batch_size=2)
    traced = etl.trace(forward, specs, etl.TensorSpec((2, 4), etl.int32))
    assert len(traced.tensor_specs) == len(specs) + 1
    assert traced.tensor_specs[-1].dtype == np.int32


def test_untied_graph_takes_the_head_as_an_input():
    """An untied config keeps the head as a normal extra graph input."""
    tied = make_config()
    untied = make_config(tie_word_embeddings=False)

    def inputs_of(cfg: ModelConfig) -> int:
        """Trace ``cfg``'s graph at one static shape and count its tensor inputs."""
        traced = etl.trace(
            gpt2.graph(cfg, 4, 1),
            gpt2.param_specs(cfg),
            etl.TensorSpec((1, 4), etl.int32),
        )
        return len(traced.tensor_specs)

    assert "lm_head.weight" in gpt2.param_specs(untied)
    assert inputs_of(untied) == inputs_of(tied) + 1


def test_graph_is_a_defn_not_callable():
    """``graph`` returns an ``@etl.defn`` graph definition, never an eager callable."""
    cfg = make_config()
    forward = gpt2.graph(cfg, seq_len=4)
    assert getattr(forward, "__etl_defn__", False) is True
    with pytest.raises(etl.TraceError):
        forward(make_params(cfg), make_ids(cfg, batch=1, seq=4))


def test_graph_accepts_canonicalized_hf_weights():
    """A graph built from real (synthetic-HF) weights matches the fused reference."""
    cfg = make_config(n_layer=2)
    state = make_hf_state_dict(cfg, seed=23)
    params = gpt2.canonicalize_weights(cfg, state)
    ids = make_ids(cfg, batch=1, seq=4)
    got = run_graph(cfg, params, ids)
    want = fused_qkv_reference(cfg, state, ids)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        want, gpt2.reference(cfg, params, ids), rtol=1e-5, atol=1e-6
    )


def test_graph_input_shape_mismatch_is_explicit():
    """Running a seq-4 graph on seq-5 ids raises instead of reshaping silently."""
    cfg = make_config()
    params = make_params(cfg)
    forward = gpt2.graph(cfg, seq_len=4)
    with pytest.raises(etl.ShapeError):
        etl.evaluate(forward, params, make_ids(cfg, batch=1, seq=5))


def test_graph_is_causal():
    """Changing a later token must not change earlier positions' logits."""
    cfg = make_config(n_layer=2)
    params = make_params(cfg, seed=4)
    ids = make_ids(cfg, batch=1, seq=6)
    perturbed = ids.copy()
    perturbed[0, -1] = (int(perturbed[0, -1]) + 1) % cfg.vocab_size
    base = run_graph(cfg, params, ids)
    other = run_graph(cfg, params, perturbed)
    np.testing.assert_allclose(base[0, :-1], other[0, :-1], rtol=1e-6, atol=1e-6)
    assert not np.allclose(base[0, -1], other[0, -1], rtol=1e-6, atol=1e-6)


# ===========================================================================
# 5. explicit errors (static-specialisation validation)
# ===========================================================================
def test_graph_rejects_seq_len_beyond_position_table():
    """``seq_len > max_position_embeddings`` is an explicit config error."""
    cfg = make_config()
    with pytest.raises(HFConfigError, match="max_position_embeddings"):
        gpt2.graph(cfg, cfg.max_position_embeddings + 1)


def test_graph_accepts_exact_position_budget():
    """``seq_len == max_position_embeddings`` is allowed (table fully used)."""
    cfg = make_config()
    params = make_params(cfg, seed=9)
    ids = make_ids(cfg, batch=1, seq=cfg.max_position_embeddings)
    assert_parity(cfg, params, ids)


@pytest.mark.parametrize("bad_seq_len", [0, -1])
def test_graph_rejects_non_positive_seq_len(bad_seq_len):
    """``seq_len <= 0`` is an explicit config error."""
    with pytest.raises(HFConfigError, match="seq_len"):
        gpt2.graph(make_config(), bad_seq_len)


@pytest.mark.parametrize("bad_batch", [0, -3])
def test_graph_rejects_non_positive_batch_size(bad_batch):
    """``batch_size <= 0`` is an explicit config error."""
    with pytest.raises(HFConfigError, match="batch_size"):
        gpt2.graph(make_config(), 4, bad_batch)


def test_graph_rejects_llama_config():
    """A llama-family config is refused by the GPT-2 module."""
    llama = ModelConfig.from_dict(
        {
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
        }
    )
    assert llama.family == "llama"
    with pytest.raises(HFConfigError, match="family"):
        gpt2.graph(llama, 4)


@pytest.mark.parametrize("entry", ["param_specs", "canonicalize_weights", "reference"])
def test_other_entry_points_reject_llama_config(entry):
    """Every entry point validates the family, not just ``graph``."""
    llama = ModelConfig.from_dict(
        {
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
        }
    )
    with pytest.raises(HFConfigError, match="family"):
        if entry == "param_specs":
            gpt2.param_specs(llama)
        elif entry == "canonicalize_weights":
            gpt2.canonicalize_weights(llama, {})
        else:
            gpt2.reference(llama, {}, np.zeros((1, 1), dtype=np.int32))


def test_reference_rejects_bad_ids():
    """``reference`` validates the ids array (rank, dtype, length) explicitly."""
    cfg = make_config()
    params = make_params(cfg)
    with pytest.raises(ValueError, match="input_ids"):
        gpt2.reference(cfg, params, np.zeros((4,), dtype=np.int32))
    with pytest.raises(ValueError, match="integer"):
        gpt2.reference(cfg, params, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(HFConfigError, match="max_position_embeddings"):
        gpt2.reference(cfg, params, np.zeros((1, cfg.max_position_embeddings + 1), dtype=np.int32))


def test_reference_rejects_param_mismatch():
    """``reference`` requires exactly the canonical parameter set (tied: no head key)."""
    cfg = make_config()
    params = make_params(cfg)
    ids = make_ids(cfg, batch=1, seq=4)
    assert "lm_head.weight" not in params
    incomplete = dict(params)
    del incomplete["embed.weight"]
    with pytest.raises(HFWeightError, match="embed.weight"):
        gpt2.reference(cfg, incomplete, ids)
    redundant = dict(params)
    redundant["lm_head.weight"] = np.ascontiguousarray(params["embed.weight"].T)
    with pytest.raises(HFWeightError, match="lm_head.weight"):
        gpt2.reference(cfg, redundant, ids)
    extended = dict(params)
    extended["extra.weight"] = np.zeros((2,), dtype=np.float32)
    with pytest.raises(HFWeightError, match="extra.weight"):
        gpt2.reference(cfg, extended, ids)
    wrong = dict(params)
    wrong["embed.weight"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(HFWeightError, match="embed.weight"):
        gpt2.reference(cfg, wrong, ids)


# ===========================================================================
# 5b. the tied-weight contract (one embedding input for gather AND output head)
# ===========================================================================
def test_tied_canonical_tree_holds_one_embedding_array():
    """No second array duplicates the embedding (the head is not materialised)."""
    cfg = make_config(n_layer=1)
    canonical = gpt2.canonicalize_weights(cfg, make_hf_state_dict(cfg))
    embed = canonical["embed.weight"]
    # Content-based (shapes collide in a tiny config), so a transposed copy of the
    # embedding under any other key fails, whatever its shape.
    copies = [
        name
        for name, array in canonical.items()
        if name != "embed.weight"
        and (np.array_equal(array, embed) or np.array_equal(array, embed.T))
    ]
    assert copies == []
    assert set(canonical) == set(gpt2.param_specs(cfg))
    assert not any("lm_head" in name for name in canonical)


def test_tied_graph_matches_the_reference_and_reuses_the_embedding():
    """The tied graph's logits equal the oracle's (head = embedding transpose)."""
    cfg = make_config(tie_word_embeddings=True)
    state = make_hf_state_dict(cfg, seed=17)
    params = gpt2.canonicalize_weights(cfg, state)
    ids = make_ids(cfg, batch=2, seq=5)
    max_abs = assert_parity(cfg, params, ids)
    assert max_abs < 1e-5, f"tied parity regressed: max-abs {max_abs:.3e}"
    # The same weights through the FUSED-HF oracle agree as well.
    np.testing.assert_allclose(
        run_graph(cfg, params, ids),
        fused_qkv_reference(cfg, state, ids),
        rtol=1e-5,
        atol=1e-6,
    )


def test_tied_and_untied_graphs_agree():
    """Tied ≡ untied-with-head=embeddingᵀ (same one embedding, same numbers)."""
    tied = make_config(n_layer=1)
    untied = make_config(n_layer=1, tie_word_embeddings=False)
    params = gpt2.canonicalize_weights(tied, make_hf_state_dict(tied, seed=19))
    with_head = dict(params)
    with_head["lm_head.weight"] = np.ascontiguousarray(params["embed.weight"].T)
    ids = make_ids(tied, batch=1, seq=4)
    np.testing.assert_allclose(
        run_graph(tied, params, ids),
        run_graph(untied, with_head, ids),
        rtol=1e-6,
        atol=1e-6,
    )


def test_untied_checkpoint_with_head_round_trips_unchanged():
    """The untied path is untouched: its head key comes from the checkpoint."""
    cfg = make_config(tie_word_embeddings=False)
    params = gpt2.canonicalize_weights(
        cfg, make_hf_state_dict(cfg, tie_lm_head=False)
    )
    assert set(params) == set(gpt2.param_specs(cfg))
    assert "lm_head.weight" in params
    assert_parity(cfg, params, make_ids(cfg, batch=1, seq=4))


def test_validate_params_accepts_a_tied_tree_and_rejects_a_redundant_head():
    """``validate_params`` mirrors the specs: the tied tree has no head, a redundant one is an error."""
    cfg = make_config(n_layer=1)
    params = gpt2.canonicalize_weights(cfg, make_hf_state_dict(cfg))
    validated = validate_params(cfg, params)
    assert set(validated) == set(param_specs(cfg)) == set(gpt2.param_specs(cfg))
    assert "lm_head.weight" not in validated

    ids = make_ids(cfg, batch=1, seq=4)
    np.testing.assert_allclose(
        predict(_loaded(cfg, validated), ids, seq_len=4),
        reference_logits(cfg, validated, ids),
        rtol=1e-5,
        atol=1e-6,
    )

    redundant = dict(params)
    redundant["lm_head.weight"] = np.ascontiguousarray(params["embed.weight"].T)
    with pytest.raises(HFWeightError, match="lm_head.weight"):
        validate_params(cfg, redundant)
    with pytest.raises(etl.ETLError, match="lm_head.weight"):
        run(build(forward_fn(cfg, seq_len=4), cfg), redundant, ids)


def test_real_tiny_gpt2_tied_checkpoint_dedupes_its_shipped_head():
    """The real ``sshleifer/tiny-gpt2`` ships ``lm_head.weight``: verified, deduped.

    The checkpoint's head is a byte-copy of ``wte.weight``, so canonicalisation
    must consume it (never leave it as an unexpected key) and drop it — the tied
    tree has one fewer key and the graph keeps numpy-interpreter parity.
    """
    loaded = load_hf(hf_download.require_model("tiny_gpt2"))
    cfg = loaded.config
    assert loaded.family == "gpt2"
    assert cfg.tie_word_embeddings is True
    params = loaded.params
    assert "lm_head.weight" not in params
    assert set(params) == set(param_specs(cfg))
    assert params["embed.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    # Loading knows no length: ``seq_len`` is a graph specialisation, so the
    # forward for this call is built explicitly (and memoised by the program).
    fwd = forward_fn(cfg, seq_len=8)
    ids = (np.arange(8, dtype=np.int32) * 7 % cfg.vocab_size).reshape(1, 8)
    got = run(build(fwd, cfg), params, ids).numpy()
    want = reference_logits(cfg, params, ids)
    max_abs = float(np.max(np.abs(got - want)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert max_abs < 1e-6, f"tiny_gpt2 tied parity: max-abs {max_abs:.3e}"

# ===========================================================================
# 6. determinism
# ===========================================================================
def test_runs_are_bit_identical():
    """Two runs of the same graph on the same inputs give bit-identical logits."""
    cfg = make_config()
    params = make_params(cfg, seed=13)
    ids = make_ids(cfg, batch=2, seq=4)
    first = run_graph(cfg, params, ids)
    second = run_graph(cfg, params, ids)
    np.testing.assert_array_equal(first, second)
    # The reference is a different formulation of the same math, so it agrees
    # numerically but not bit-for-bit (measured max-abs ~3e-8).
    np.testing.assert_allclose(
        first, gpt2.reference(cfg, params, ids), rtol=1e-5, atol=1e-6
    )


def test_canonicalize_is_deterministic():
    """Canonicalising the same state dict twice yields identical arrays."""
    cfg = make_config()
    state = make_hf_state_dict(cfg)
    first = gpt2.canonicalize_weights(cfg, state)
    second = gpt2.canonicalize_weights(cfg, state)
    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


# ===========================================================================
# 7. compiler-neutrality (StableHLO export of the whole forward)
# ===========================================================================
def test_graph_exports_to_stablehlo():
    """The whole GPT-2 forward exports as StableHLO text (no deferrals)."""
    from etl.backends import stablehlo

    cfg = make_config(n_layer=1)
    traced = etl.trace(
        gpt2.graph(cfg, seq_len=4, batch_size=2),
        gpt2.param_specs(cfg),
        etl.TensorSpec((2, 4), etl.int32),
    )
    text = stablehlo.export(traced)
    for marker in (
        "stablehlo.gather",  # embeddings + learned positions
        "stablehlo.broadcast_in_dim",  # position broadcast over the batch axis
        "stablehlo.compare",  # causal mask
        "stablehlo.tanh",  # gelu_new
        "stablehlo.exponential",  # masked softmax
        "stablehlo.dot_general",  # projections / lm_head
    ):
        assert marker in text, f"expected {marker} in the exported StableHLO"


def test_erf_gelu_export_defers_explicitly():
    """An exact-erf ``gelu`` config is numpy-backend-only (explicit BackendError)."""
    from etl.backends import stablehlo

    cfg = make_config(activation_function="gelu", n_layer=1)
    traced = etl.trace(
        gpt2.graph(cfg, seq_len=4, batch_size=1),
        gpt2.param_specs(cfg),
        etl.TensorSpec((1, 4), etl.int32),
    )
    with pytest.raises(etl.BackendError, match="erf"):
        stablehlo.export(traced)
