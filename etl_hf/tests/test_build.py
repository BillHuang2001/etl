"""Tests for the staging layer — :mod:`etl_hf.pipeline` (+ :mod:`etl_hf.program`).

The package API is **functional**: loading (:func:`etl_hf.load_hf` →
``LoadedModel``), the compute function (:func:`etl_hf.forward_fn` →
``@etl.defn``), and staging (:func:`etl_hf.build` / :func:`etl_hf.run` /
:func:`etl_hf.predict`). This file pins those three halves end-to-end.

Self-contained and offline: tiny synthetic Llama and GPT-2 configs plus seeded
synthetic parameter trees, and one **synthetic HF checkpoint directory** written
with a local safetensors writer (8-byte LE header length + JSON header + raw
buffer), so ``load_hf`` is exercised end-to-end without a download, torch, or
``safetensors``.

What is pinned here:

* **parity** — the compiled pipeline (numpy interpreter) against each family's
  independent pure-numpy reference;
* **the default-option matrix** (:func:`etl_hf.pipeline.resolve_build_options`) —
  the measured pitfalls (``opt_level`` injection including the **ROCm** ``"O1"``
  rule, and the llvm-cpu-only ``iree_compile_args`` that must reach the
  *compile* stage), including "explicit wins", "``ETL_OPT_LEVEL`` wins", "the
  caller's dict is not mutated" and the ``ETL_BACKEND``/``ETL_TARGET_BACKENDS``
  env defaults — plus the device→IREE-target map itself, including
  ``vulkan`` → ``vulkan-spirv``;
* **memoisation** — identical ``build`` calls return the identical
  ``Executable``; a different ``seq_len``/``batch_size``/option/private cache does
  not;
* **the static specialisation** — ``seq_len``/``batch_size`` belong to
  :func:`etl_hf.forward_fn` (the graph), so junk is rejected there and an
  explicit-but-disagreeing value handed to :func:`etl_hf.build` is rejected
  instead of silently building a mismatched ids spec;
* **the activation gate** — an erf-based activation is rejected on a compiled
  backend *before tracing*, with both remedies named, and needs no iree install;
* **explicit input handling** — the single documented host coercion
  (integral ids → int32), ``DTypeError``/``ShapeError`` for everything else, an
  ``etl.Tensor`` of ids forwarded to ``etl.run`` **verbatim** (the
  device-resident path), and etl's structure errors propagating untouched for a
  bad params tree;
* **explicit placement** — ``place_params``/``place_ids`` are one visible
  ``Tensor.to`` each, with their argument error tables;
* **``predict`` is host-facing** — device params are refused with the
  device-resident recipe named (never a hidden device-to-host copy);
* **``load_hf``/``LoadedModel``** — checkpoint → config + canonical params as
  plain data, the documented default ``seq_len`` living on the forward (never on
  the loaded model), and the explicit errors of both.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import etl

from etl_hf import ops
from etl_hf import pipeline as pipeline_module
from etl_hf import program as program_module
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.load import LoadedModel, load_config, load_hf, validate_params
from etl_hf.models import gpt2 as gpt2_module
from etl_hf.models import llama as llama_module
from etl_hf.pipeline import (
    DEFAULT_LLVM_CPU_COMPILE_ARGS,
    DEFAULT_ROCM_OPT_LEVEL,
    build,
    place_ids,
    place_params,
    predict,
    resolve_build_options,
    run,
)
from etl_hf.program import DEFAULT_SEQ_LEN, forward_fn, param_specs, reference_logits

#: Synthetic weight scale: small enough that RMSNorm/LayerNorm/softmax stay
#: well-conditioned in fp32 (the parity claim is about semantics, not saturation).
_SCALE = 0.1

_LLAMA_JSON: dict = {
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

_GPT2_JSON: dict = {
    "model_type": "gpt2",
    "n_vocab": 24,
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 2,
    "n_inner": 32,
    "n_positions": 16,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": True,
}

_FAMILY_MODULES = {"llama": llama_module, "gpt2": gpt2_module}

#: rtol/atol for the compiled-vs-reference parity assertions (fp32 accumulation
#: order differs between the graph and the oracle; measured max-abs is ~1e-7).
_RTOL = 1e-5
_ATOL = 1e-6


# ---------------------------------------------------------------------------
# helpers: configs, params, ids, the graph's stamped specialisation
# ---------------------------------------------------------------------------
def llama_config(**overrides) -> ModelConfig:
    """Tiny Llama config (GQA: 4 query heads, 2 KV heads), with overrides."""
    raw = dict(_LLAMA_JSON)
    raw.update(overrides)
    return ModelConfig.from_dict(raw)


def gpt2_config(**overrides) -> ModelConfig:
    """Tiny GPT-2 config (4 heads, 2 layers), with overrides."""
    raw = dict(_GPT2_JSON)
    raw.update(overrides)
    return ModelConfig.from_dict(raw)


def make_params(cfg: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded canonical params matching ``cfg``'s own family specs."""
    specs = _FAMILY_MODULES[cfg.family].param_specs(cfg)
    rng = np.random.default_rng(seed)
    return {
        name: np.asarray(rng.standard_normal(spec.shape) * _SCALE, dtype=np.float32)
        for name, spec in specs.items()
    }


def make_ids(cfg: ModelConfig, batch: int, seq: int, seed: int = 1) -> np.ndarray:
    """Seeded ``(batch, seq)`` int32 token ids inside the vocabulary."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference (the number reported in parity assertions)."""
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def specialisation(cfg: ModelConfig, **kwargs) -> dict:
    """The static specialisation stamped on ``forward_fn(cfg, **kwargs)``.

    A config (plus ``seq_len``/``batch_size``/``pack_weights``) is a static
    specialisation *value*, so the graph — not the loaded model — is where those
    numbers live. Reading them back off the forward is how the tests check that
    ``seq_len=None`` means the documented default.
    """
    forward = forward_fn(cfg, **kwargs)
    return dict(getattr(forward, program_module.SPECIALISATION_ATTR))


@pytest.fixture(autouse=True)
def _hermetic_option_env(monkeypatch):
    """Clear the process env knobs the option matrix reads (per test).

    ``ETL_OPT_LEVEL``/``ETL_TARGET_BACKENDS`` are consulted by
    :func:`resolve_build_options` (explicit > env > etl_hf default), so a
    developer shell that exports them must not change what the *defaults* are
    asserted to be. Tests that exercise the env set it themselves with the same
    ``monkeypatch``, which wins over this fixture's deletes.
    """
    monkeypatch.delenv("ETL_OPT_LEVEL", raising=False)
    monkeypatch.delenv("ETL_TARGET_BACKENDS", raising=False)


# ---------------------------------------------------------------------------
# helpers: a synthetic HF checkpoint on disk (real safetensors bytes)
# ---------------------------------------------------------------------------
def _safetensors_bytes(header: dict, payload: bytes) -> bytes:
    """Assemble a safetensors blob: 8-byte LE header length + JSON header + data."""
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)  # the real writer pads the header to 8 bytes
    return struct.pack("<Q", len(blob)) + blob + payload


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write ``tensors`` as a float32 safetensors file (no ``safetensors`` dep)."""
    header: dict = {}
    payload = bytearray()
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        begin = len(payload)
        payload += data.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(data.shape),
            "data_offsets": [begin, len(payload)],
        }
    path = Path(path)
    path.write_bytes(_safetensors_bytes(header, bytes(payload)))
    return path


def llama_hf_state_dict(cfg: ModelConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """An HF ``LlamaForCausalLM``-shaped state dict: linears stored ``(out, in)``."""
    hidden = cfg.hidden_size
    heads, head_dim = cfg.num_attention_heads, cfg.head_dim
    kv_dim = cfg.num_key_value_heads * head_dim
    intermediate, vocab = cfg.intermediate_size, cfg.vocab_size

    def weigh(*shape: int) -> np.ndarray:
        return np.asarray(rng.standard_normal(shape) * _SCALE, dtype=np.float32)

    def scale(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(1.0 + 0.05 * rng.standard_normal(shape), dtype=np.float32)

    state: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weigh(vocab, hidden),
        "model.norm.weight": scale((hidden,)),
    }
    for index in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{index}."
        state[prefix + "input_layernorm.weight"] = scale((hidden,))
        state[prefix + "self_attn.q_proj.weight"] = weigh(heads * head_dim, hidden)
        state[prefix + "self_attn.k_proj.weight"] = weigh(kv_dim, hidden)
        state[prefix + "self_attn.v_proj.weight"] = weigh(kv_dim, hidden)
        state[prefix + "self_attn.o_proj.weight"] = weigh(hidden, heads * head_dim)
        state[prefix + "post_attention_layernorm.weight"] = scale((hidden,))
        state[prefix + "mlp.gate_proj.weight"] = weigh(intermediate, hidden)
        state[prefix + "mlp.up_proj.weight"] = weigh(intermediate, hidden)
        state[prefix + "mlp.down_proj.weight"] = weigh(hidden, intermediate)
    if not cfg.tie_word_embeddings:
        state["lm_head.weight"] = weigh(vocab, hidden)
    return state


def gpt2_hf_state_dict(cfg: ModelConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """An HF ``GPT2LMHeadModel``-shaped state dict (``Conv1D`` ``(in, out)``)."""
    hidden = cfg.hidden_size
    intermediate, vocab = cfg.intermediate_size, cfg.vocab_size

    def weigh(*shape: int) -> np.ndarray:
        return np.asarray(rng.standard_normal(shape) * _SCALE, dtype=np.float32)

    def scale(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(1.0 + 0.05 * rng.standard_normal(shape), dtype=np.float32)

    def bias(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(0.05 * rng.standard_normal(shape), dtype=np.float32)

    state: dict[str, np.ndarray] = {
        "wte.weight": weigh(vocab, hidden),
        "wpe.weight": weigh(cfg.max_position_embeddings, hidden),
        "ln_f.weight": scale((hidden,)),
        "ln_f.bias": bias((hidden,)),
    }
    for index in range(cfg.num_hidden_layers):
        prefix = f"h.{index}."
        state[prefix + "ln_1.weight"] = scale((hidden,))
        state[prefix + "ln_1.bias"] = bias((hidden,))
        state[prefix + "attn.c_attn.weight"] = weigh(hidden, 3 * hidden)
        state[prefix + "attn.c_attn.bias"] = bias((3 * hidden,))
        state[prefix + "attn.c_proj.weight"] = weigh(hidden, hidden)
        state[prefix + "attn.c_proj.bias"] = bias((hidden,))
        state[prefix + "ln_2.weight"] = scale((hidden,))
        state[prefix + "ln_2.bias"] = bias((hidden,))
        state[prefix + "mlp.c_fc.weight"] = weigh(hidden, intermediate)
        state[prefix + "mlp.c_fc.bias"] = bias((intermediate,))
        state[prefix + "mlp.c_proj.weight"] = weigh(intermediate, hidden)
        state[prefix + "mlp.c_proj.bias"] = bias((hidden,))
    if not cfg.tie_word_embeddings:
        state["lm_head.weight"] = weigh(vocab, hidden)
    return state


def write_checkpoint_dir(
    tmp_path: Path, cfg: ModelConfig, *, name: str = "ckpt", seed: int = 7
) -> Path:
    """Write a minimal-but-real HF checkpoint directory for ``cfg``."""
    directory = Path(tmp_path) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(cfg.raw), encoding="utf-8")
    rng = np.random.default_rng(seed)
    state = (
        llama_hf_state_dict(cfg, rng)
        if cfg.family == "llama"
        else gpt2_hf_state_dict(cfg, rng)
    )
    write_safetensors(directory / "model.safetensors", state)
    return directory


# ---------------------------------------------------------------------------
# parity: forward_fn → build → run vs the family's numpy reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["llama", "gpt2"])
def test_build_and_run_match_the_family_reference(family):
    """The compiled pipeline must reproduce the independent numpy oracle."""
    cfg = llama_config() if family == "llama" else gpt2_config()
    params = make_params(cfg, seed=3)
    ids = make_ids(cfg, batch=2, seq=6, seed=4)
    executable = build(forward_fn(cfg, seq_len=6, batch_size=2), cfg)
    logits = run(executable, params, ids)
    expected = reference_logits(cfg, params, ids)
    assert isinstance(logits, etl.Tensor)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert logits.dtype == etl.float32
    assert logits.device == etl.core.Device("cpu", 0)
    got = logits.numpy()
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)
    assert max_abs(got, expected) <= 1e-6, (
        f"{family}: max-abs {max_abs(got, expected):.3e} exceeds the fp32 budget"
    )


def test_build_and_run_match_reference_for_untied_and_tied_variants():
    """Batch 1 + a tied GPT-2 head (lm_head derived from the embedding)."""
    cfg = gpt2_config(tie_word_embeddings=True)
    params = make_params(cfg, seed=11)
    ids = make_ids(cfg, batch=1, seq=5, seed=12)
    got = run(build(forward_fn(cfg, seq_len=5), cfg), params, ids).numpy()
    expected = gpt2_module.reference(cfg, params, ids)
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# build: the program-stage argument checks
# ---------------------------------------------------------------------------
def test_build_rejects_a_bad_family_override():
    cfg = llama_config()
    with pytest.raises(HFConfigError) as excinfo:
        build(forward_fn(cfg, seq_len=4), cfg, family="gpt2")
    message = str(excinfo.value)
    assert "llama" in message and "gpt2" in message


def test_build_rejects_unknown_family():
    cfg = llama_config()
    with pytest.raises(HFUnsupportedError) as excinfo:
        build(forward_fn(cfg, seq_len=4), cfg, family="bert")
    assert "unsupported model family" in str(excinfo.value)


def test_build_rejects_non_config():
    with pytest.raises(TypeError) as excinfo:
        build(forward_fn(llama_config(), seq_len=4), {"model_type": "llama"})  # type: ignore[arg-type]
    assert "ModelConfig" in str(excinfo.value)


def test_build_rejects_an_already_staged_object():
    """``build`` takes the graph *definition* — a traced Graph/Executable is past it."""
    cfg = llama_config()
    executable = build(forward_fn(cfg, seq_len=4), cfg)
    with pytest.raises(TypeError) as excinfo:
        build(executable, cfg)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "etl_hf.forward_fn" in message and "etl_hf.run" in message


@pytest.mark.parametrize("seq_len", [0, -1, 1.5, "4", True])
def test_bad_seq_len_is_rejected(seq_len):
    """``seq_len`` is static: junk is rejected at both stages that accept it."""
    cfg = llama_config()
    with pytest.raises(HFConfigError) as excinfo:
        forward_fn(cfg, seq_len=seq_len)  # type: ignore[arg-type]
    assert "seq_len" in str(excinfo.value)
    fwd = forward_fn(cfg, seq_len=4)
    with pytest.raises(HFConfigError) as excinfo:
        build(fwd, cfg, seq_len=seq_len)  # type: ignore[arg-type]
    assert "seq_len" in str(excinfo.value)


def test_seq_len_none_is_the_documented_default():
    """``seq_len=None`` means ``min(max_position_embeddings, 32)`` — not an error."""
    assert DEFAULT_SEQ_LEN == 32
    cfg = llama_config()
    assert specialisation(cfg)["seq_len"] == min(cfg.max_position_embeddings, 32)
    # ... and it is the same graph, so the same memo entry
    assert forward_fn(cfg, seq_len=None) is forward_fn(cfg)


@pytest.mark.parametrize("batch_size", [0, -2, 2.0])
def test_bad_batch_size_is_rejected(batch_size):
    cfg = llama_config()
    with pytest.raises(HFConfigError) as excinfo:
        forward_fn(cfg, seq_len=4, batch_size=batch_size)  # type: ignore[arg-type]
    assert "batch_size" in str(excinfo.value)
    fwd = forward_fn(cfg, seq_len=4, batch_size=1)
    with pytest.raises(HFConfigError) as excinfo:
        build(fwd, cfg, batch_size=batch_size)  # type: ignore[arg-type]
    assert "batch_size" in str(excinfo.value)


def test_batch_size_none_means_one():
    assert specialisation(llama_config(), seq_len=4)["batch_size"] == 1
    assert forward_fn(llama_config(), seq_len=4, batch_size=None) is forward_fn(
        llama_config(), seq_len=4
    )


def test_build_rejects_a_specialisation_that_disagrees_with_the_forward():
    """The forward's stamped specialisation is authoritative (never silently swapped)."""
    cfg = llama_config()
    fwd = forward_fn(cfg, seq_len=4, batch_size=1)
    with pytest.raises(HFConfigError) as excinfo:
        build(fwd, cfg, seq_len=5)
    message = str(excinfo.value)
    assert "etl_hf.build(seq_len=5)" in message
    assert "etl_hf.forward_fn" in message and "seq_len=4" in message
    with pytest.raises(HFConfigError) as excinfo:
        build(fwd, cfg, batch_size=2)
    message = str(excinfo.value)
    assert "etl_hf.build(batch_size=2)" in message and "batch_size=1" in message
    # agreeing values are fine, and land on the same memoised artifact
    assert build(fwd, cfg, seq_len=4, batch_size=1) is build(fwd, cfg)


def test_build_rejects_seq_len_beyond_max_position_embeddings():
    cfg = llama_config(max_position_embeddings=8)
    with pytest.raises(HFConfigError) as excinfo:
        forward_fn(cfg, seq_len=9)
    message = str(excinfo.value)
    assert "max_position_embeddings (8)" in message


# ---------------------------------------------------------------------------
# memoisation
# ---------------------------------------------------------------------------
def test_identical_builds_return_the_same_executable():
    cfg = llama_config()
    first = build(forward_fn(cfg, seq_len=5, batch_size=2), cfg)
    other = llama_config()  # equal config, new object
    second = build(forward_fn(other, seq_len=5, batch_size=2), other)
    assert first is second


def test_equal_configs_with_the_same_digest_share_the_graph():
    """``ModelConfig.digest`` (not object identity) keys the forward memo."""
    a = llama_config()
    b = llama_config()
    assert a is not b and a.digest == b.digest
    assert program_module.forward_fn(a, seq_len=5) is program_module.forward_fn(b, seq_len=5)


def test_explicit_numpy_backend_matches_the_default():
    cfg = llama_config()
    fwd = forward_fn(cfg, seq_len=4)
    assert build(fwd, cfg) is build(fwd, cfg, backend="numpy")


def test_different_seq_len_or_batch_size_gets_its_own_executable():
    cfg = llama_config()
    base = build(forward_fn(cfg, seq_len=4), cfg)
    assert build(forward_fn(cfg, seq_len=5), cfg) is not base
    assert build(forward_fn(cfg, seq_len=4, batch_size=2), cfg) is not base


def test_different_options_get_their_own_executable():
    """Options participate in etl's cache key (so do backend/device names)."""
    cfg = llama_config()
    fwd = forward_fn(cfg, seq_len=4)
    base = build(fwd, cfg)
    other = build(fwd, cfg, opt_level="O0")
    assert other is not base
    assert build(fwd, cfg) is base  # the first entry is still memoised


def test_different_config_gets_its_own_executable():
    cfg = llama_config()
    base = build(forward_fn(cfg, seq_len=4), cfg)
    deeper_cfg = llama_config(num_hidden_layers=1)
    deeper = build(forward_fn(deeper_cfg, seq_len=4), deeper_cfg)
    assert deeper is not base
    assert deeper is not build(forward_fn(cfg, seq_len=4), cfg)


def test_configs_differing_only_in_raw_share_the_executable():
    """``ModelConfig.digest`` ignores ``raw``, so ignorable HF fields share a graph."""
    plain = llama_config()
    annotated = llama_config(torch_dtype="float16", architectures=["LlamaForCausalLM"])
    assert annotated.digest == plain.digest
    assert build(forward_fn(annotated, seq_len=4), annotated) is build(
        forward_fn(plain, seq_len=4), plain
    )


def test_private_compile_cache_isolates():
    cfg = llama_config()
    fwd = forward_fn(cfg, seq_len=4)
    private = etl.CompileCache()
    first = build(fwd, cfg, cache=private)
    second = build(fwd, cfg, cache=private)
    assert first is second
    assert private.hits >= 1 and private.misses >= 1
    default_cached = build(fwd, cfg)
    assert default_cached is not first
    third = build(fwd, cfg, cache=private)
    assert third is first  # the private cache kept its entry


def test_different_backend_gets_its_own_executable():
    """A distinct backend name is a distinct artifact (needs a compiled backend)."""
    pytest.importorskip("iree.compiler", reason="iree is not installed")
    cfg = llama_config()
    fwd = forward_fn(cfg, seq_len=4)
    numpy_executable = build(fwd, cfg)
    iree_executable = build(fwd, cfg, backend="iree")
    assert iree_executable is not numpy_executable
    assert iree_executable.backend != numpy_executable.backend


def test_forward_memo_is_bounded():
    """The LRU of ``@etl.defn`` forwards stays bounded (documented at 64 entries)."""
    cfg = llama_config(max_position_embeddings=128)
    for seq_len in range(1, program_module._GRAPH_CACHE_MAX + 8):
        program_module.forward_fn(cfg, seq_len=seq_len)
    assert len(program_module._GRAPH_CACHE) <= program_module._GRAPH_CACHE_MAX


# ---------------------------------------------------------------------------
# the default-option matrix
# ---------------------------------------------------------------------------
def test_default_llvm_cpu_compile_args_are_the_measured_flags():
    assert DEFAULT_LLVM_CPU_COMPILE_ARGS == (
        "--iree-llvmcpu-target-cpu=host",
        "--iree-opt-data-tiling",
    )


def test_resolve_numpy_default_for_host():
    assert resolve_build_options(None, None, {}) == ("numpy", {})
    cpu = etl.core.Device("cpu", 0)
    assert resolve_build_options(None, cpu, {}) == ("numpy", {})


def test_resolve_numpy_returns_the_options_unchanged():
    options = {"target_backends": ["llvm-cpu"], "custom": 1}
    name, resolved = resolve_build_options("numpy", None, options)
    assert name == "numpy"
    assert resolved == options and resolved is not options


def test_resolve_defaults_to_iree_for_a_non_cpu_device():
    cuda = etl.core.Device("cuda", 0)
    name, resolved = resolve_build_options(None, cuda, {})
    assert name == "iree"
    assert resolved["opt_level"] == "O3"
    # cuda targets: never llvm-cpu flags
    assert "iree_compile_args" not in resolved


def test_resolve_injects_iree_llvm_cpu_flags_and_opt_level():
    name, resolved = resolve_build_options("iree", None, {})
    assert name == "iree"
    assert resolved["opt_level"] == "O3"
    assert resolved["iree_compile_args"] == list(DEFAULT_LLVM_CPU_COMPILE_ARGS)
    # a cpu device is an llvm-cpu target too
    cpu = etl.core.Device("cpu", 0)
    assert resolve_build_options("iree", cpu, {})[1] == resolved


def test_resolve_injects_llvm_cpu_flags_for_an_explicit_llvm_cpu_target():
    _, resolved = resolve_build_options("iree", None, {"target_backends": ["llvm-cpu"]})
    assert resolved["iree_compile_args"] == list(DEFAULT_LLVM_CPU_COMPILE_ARGS)


def test_resolve_never_injects_llvm_cpu_flags_for_cuda_or_rocm():
    for kind in ("cuda", "rocm", "hip"):
        device = etl.core.Device(kind, 0)
        name, resolved = resolve_build_options("iree", device, {})
        assert name == "iree"
        # ROCm gets the codegen-safe O1 default (see the O1 tests below); cuda
        # keeps O3. Neither ever receives the llvm-cpu flags.
        expected = "O1" if kind in ("rocm", "hip") else "O3"
        assert resolved["opt_level"] == expected, kind
        assert "iree_compile_args" not in resolved, kind
    _, resolved = resolve_build_options(
        "iree", None, {"target_backends": ["cuda", "rocm"]}
    )
    assert "iree_compile_args" not in resolved


def test_targets_from_device_maps_every_known_kind():
    """The device→IREE-target map: each kind has its own target spelling.

    ``vulkan`` is a device *kind*, while IREE's target backend is spelled
    ``vulkan-spirv`` — passing the bare kind through would be an unknown-target
    error at compile time.
    """
    device = etl.core.Device
    assert pipeline_module._targets_from_device(None) == ["llvm-cpu"]
    assert pipeline_module._targets_from_device(device("cpu", 0)) == ["llvm-cpu"]
    assert pipeline_module._targets_from_device(device("cuda", 0)) == ["cuda"]
    assert pipeline_module._targets_from_device(device("rocm", 0)) == ["rocm"]
    # ``hip`` is an accepted spelling of the rocm kind (canonicalised here)
    assert pipeline_module._targets_from_device(device("hip", 1)) == ["rocm"]
    # vulkan → vulkan-spirv (the only kind whose target is not its kind name)
    for index in (0, 1):
        assert pipeline_module._targets_from_device(device("vulkan", index)) == [
            "vulkan-spirv"
        ]
    # an unknown kind is still passed through verbatim — never guessed
    assert pipeline_module._targets_from_device(device("metal", 2)) == ["metal"]


def test_resolve_maps_a_vulkan_device_to_iree_and_vulkan_spirv():
    """``device=Device('vulkan', 0)`` → the compiled path, Vulkan target, no cpu flags.

    The derived target list is what decides the injected defaults, so this pins
    the observable consequence of the mapping: Vulkan keeps the ``O3`` default
    (it is not an all-ROCm target set) and never receives the llvm-cpu-only
    ``iree_compile_args``. The device-level Vulkan compile caveats (the default
    target profile's 16384-byte workgroup-memory budget, and the explicit
    ``opt_level="O1"`` the 124M-class graphs need) are the *user's* options —
    etl_hf injects nothing for them (see ``bench_logs/vulkan_findings.md``).
    """
    vulkan = etl.core.Device("vulkan", 0)
    name, resolved = resolve_build_options(None, vulkan, {})
    assert name == "iree", "a vulkan device is a non-cpu device ⇒ the compiled path"
    assert resolved == {"opt_level": "O3"}, resolved
    assert "iree_compile_args" not in resolved
    # an explicitly named iree backend behaves identically ...
    assert resolve_build_options("iree", vulkan, {})[1] == {"opt_level": "O3"}
    # ... and an explicit target/opt_level still wins (the documented remedy).
    _, explicit = resolve_build_options(
        "iree",
        vulkan,
        {"target_backends": ["vulkan-spirv"], "opt_level": "O1"},
    )
    assert explicit["opt_level"] == "O1"
    assert explicit["target_backends"] == ["vulkan-spirv"]
    # an explicit vulkan-spirv target set is not llvm-cpu ⇒ nothing injected
    # (the caller's own target_backends is of course passed through untouched)
    _, by_target = resolve_build_options("iree", None, {"target_backends": ["vulkan-spirv"]})
    assert by_target == {"target_backends": ["vulkan-spirv"], "opt_level": "O3"}, by_target
    # numpy stays option-untouched even on a vulkan device
    assert resolve_build_options("numpy", vulkan, {}) == ("numpy", {})


def test_default_rocm_opt_level_is_the_codegen_safe_one():
    """The ROCm default is ``O1`` (upstream iree fails at O2/O3 on these shapes)."""
    assert DEFAULT_ROCM_OPT_LEVEL == "O1"
    assert pipeline_module.DEFAULT_OPT_LEVEL == "O3"


def test_resolve_injects_o1_for_a_rocm_device():
    """The documented GPU path: ``device=Device('rocm', 0)`` → iree + O1."""
    for kind in ("rocm", "hip"):
        device = etl.core.Device(kind, 0)
        name, resolved = resolve_build_options(None, device, {})
        assert name == "iree"
        assert resolved["opt_level"] == DEFAULT_ROCM_OPT_LEVEL == "O1"
        assert "iree_compile_args" not in resolved
    # a rocm device with the iree backend named explicitly behaves the same
    assert resolve_build_options("iree", etl.core.Device("rocm", 1), {})[1] == {
        "opt_level": "O1"
    }


def test_resolve_injects_o1_for_explicit_rocm_targets():
    for targets in (["rocm"], ["hip"], ["rocm", "hip"], "rocm"):
        _, resolved = resolve_build_options("iree", None, {"target_backends": targets})
        assert resolved["opt_level"] == "O1", targets
        assert resolved["target_backends"] == targets  # untouched
        assert "iree_compile_args" not in resolved


def test_resolve_injects_o1_through_the_target_backends_env(monkeypatch):
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "rocm")
    _, resolved = resolve_build_options("iree", None, {})
    assert resolved["opt_level"] == "O1"
    monkeypatch.setenv("ETL_TARGET_BACKENDS", " hip ")
    _, resolved = resolve_build_options("iree", None, {})
    assert resolved["opt_level"] == "O1"
    # an explicit target list beats the env (and decides the level)
    _, resolved = resolve_build_options("iree", None, {"target_backends": ["llvm-cpu"]})
    assert resolved["opt_level"] == "O3"
    # the env's rocm targets also beat the device-derived llvm-cpu default
    _, resolved = resolve_build_options("iree", etl.core.Device("cpu", 0), {})
    assert resolved["opt_level"] == "O1"
    # ... and then the llvm-cpu flags are NOT injected (targets are not llvm-cpu)
    assert "iree_compile_args" not in resolved


def test_resolve_keeps_o3_for_every_non_rocm_target():
    cases = [
        (None, {}),  # llvm-cpu via no device
        (etl.core.Device("cpu", 0), {}),
        (etl.core.Device("cuda", 0), {}),
        (None, {"target_backends": ["llvm-cpu"]}),
        (None, {"target_backends": ["cuda"]}),
        # a mixed set containing a non-rocm target stays on O3
        (None, {"target_backends": ["rocm", "llvm-cpu"]}),
        (None, {"target_backends": ["rocm", "cuda"]}),
        (None, {"target_backends": ["gfx942"]}),
    ]
    for device, options in cases:
        _, resolved = resolve_build_options("iree", device, options)
        assert resolved["opt_level"] == "O3", (device, options)
    # non-iree compiled backends have no targets ⇒ unchanged O3 default
    for backend in ("xla", "tvm"):
        assert resolve_build_options(backend, etl.core.Device("rocm", 0), {})[1][
            "opt_level"
        ] == "O3", backend
    # numpy stays option-untouched even on a rocm device
    assert resolve_build_options("numpy", etl.core.Device("rocm", 0), {})[1] == {}


def test_resolve_explicit_opt_level_wins_over_the_rocm_default():
    rocm = etl.core.Device("rocm", 0)
    for explicit in ("O3", "O0", 2):
        _, resolved = resolve_build_options("iree", rocm, {"opt_level": explicit})
        assert resolved["opt_level"] == explicit
    _, resolved = resolve_build_options(
        "iree", None, {"target_backends": ["rocm"], "opt_level": "O2"}
    )
    assert resolved["opt_level"] == "O2"


def test_resolve_etl_opt_level_env_wins_over_the_rocm_default(monkeypatch):
    """``ETL_OPT_LEVEL`` (etl's own knob) beats etl_hf's injected default."""
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    for device in (etl.core.Device("rocm", 0), None):
        _, resolved = resolve_build_options("iree", device, {})
        assert "opt_level" not in resolved, device
    _, resolved = resolve_build_options("xla", None, {})
    assert "opt_level" not in resolved
    # a blank value is treated as unset
    monkeypatch.setenv("ETL_OPT_LEVEL", "   ")
    _, resolved = resolve_build_options("iree", etl.core.Device("rocm", 0), {})
    assert resolved["opt_level"] == "O1"
    # an explicit argument still beats the env
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    _, resolved = resolve_build_options("iree", etl.core.Device("rocm", 0), {})
    assert "opt_level" not in resolved
    _, resolved = resolve_build_options(
        "iree", etl.core.Device("rocm", 0), {"opt_level": "O3"}
    )
    assert resolved["opt_level"] == "O3"


def test_resolve_does_not_mutate_the_caller_options_for_rocm():
    options = {"target_backends": ["rocm"]}
    resolve_build_options("iree", None, options)
    assert options == {"target_backends": ["rocm"]}
    empty: dict = {}
    resolve_build_options(None, etl.core.Device("rocm", 0), empty)
    assert empty == {}


def test_resolve_honours_explicit_options():
    explicit = ["--iree-llvmcpu-target-cpu=znver4"]
    name, resolved = resolve_build_options(
        "iree", None, {"iree_compile_args": explicit, "opt_level": "O1"}
    )
    assert name == "iree"
    assert resolved["iree_compile_args"] is explicit
    assert resolved["opt_level"] == "O1"


def test_resolve_injects_opt_level_only_for_compiled_backends():
    for backend in ("xla", "tvm", "iree"):
        name, resolved = resolve_build_options(backend, None, {})
        assert name == backend
        assert resolved["opt_level"] == "O3"
    assert resolve_build_options("numpy", None, {})[1] == {}


def test_resolve_does_not_mutate_the_caller_options():
    options: dict = {}
    resolve_build_options("iree", None, options)
    assert options == {}
    options = {"target_backends": ["llvm-cpu"]}
    resolve_build_options("iree", None, options)
    assert options == {"target_backends": ["llvm-cpu"]}


def test_resolve_honours_etl_backend_env(monkeypatch):
    monkeypatch.setenv("ETL_BACKEND", "iree")
    name, resolved = resolve_build_options(None, None, {})
    assert name == "iree"
    assert resolved["opt_level"] == "O3"
    # explicit beats env
    assert resolve_build_options("numpy", None, {})[0] == "numpy"
    # blank env is treated as unset
    monkeypatch.setenv("ETL_BACKEND", "   ")
    assert resolve_build_options(None, None, {})[0] == "numpy"


def test_resolve_honours_etl_target_backends_env(monkeypatch):
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "cuda")
    _, resolved = resolve_build_options("iree", None, {})
    assert "iree_compile_args" not in resolved
    monkeypatch.setenv("ETL_TARGET_BACKENDS", " llvm-cpu , cuda ")
    _, resolved = resolve_build_options("iree", None, {})
    # a mixed target set is not "exactly llvm-cpu" ⇒ nothing injected
    assert "iree_compile_args" not in resolved
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "llvm-cpu")
    _, resolved = resolve_build_options("iree", None, {})
    assert resolved["iree_compile_args"] == list(DEFAULT_LLVM_CPU_COMPILE_ARGS)
    # explicit target_backends beats the env
    _, resolved = resolve_build_options("iree", None, {"target_backends": ["cuda"]})
    assert "iree_compile_args" not in resolved


def test_resolve_rejects_unknown_argument_shapes():
    with pytest.raises(TypeError):
        resolve_build_options(1234, None, {})
    with pytest.raises(TypeError):
        resolve_build_options("numpy", "cpu", {})


# ---------------------------------------------------------------------------
# the activation gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("activation", ["gelu", "erf"])
def test_erf_activation_is_rejected_on_a_compiled_backend(activation):
    """No iree install needed: the gate fires before any backend resolution."""
    cfg = llama_config(hidden_act=activation)
    fwd = forward_fn(cfg, seq_len=4)  # a supported (if unexportable) activation
    with pytest.raises(HFUnsupportedError) as excinfo:
        build(fwd, cfg, backend="iree")
    message = str(excinfo.value)
    assert activation in message
    assert "iree" in message
    assert "backend='numpy'" in message  # remedy 1
    assert "gelu_new" in message  # remedy 2
    assert "etl.erf" in message and "5e-4" in message  # measured tanh-vs-erf delta


def test_erf_activation_still_builds_on_the_numpy_backend():
    cfg = llama_config(hidden_act="gelu")
    params = make_params(cfg, seed=5)
    ids = make_ids(cfg, batch=1, seq=4, seed=6)
    got = run(build(forward_fn(cfg, seq_len=4), cfg), params, ids).numpy()
    expected = llama_module.reference(cfg, params, ids)
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_tanh_gelu_passes_the_gate_for_a_compiled_backend():
    """``gelu_new`` is export-safe: the gate must let it through to the backend."""
    cfg = gpt2_config(activation_function="gelu_new")
    assert ops.activation_is_export_safe(cfg.hidden_act)
    fwd = forward_fn(cfg, seq_len=4)
    try:
        build(fwd, cfg, backend="iree")
    except HFUnsupportedError:  # pragma: no cover - would be a gate bug
        pytest.fail("the activation gate rejected an export-safe activation")
    except etl.ETLError:
        pass  # iree absent: the request reached the backend layer, as intended


def test_unknown_activation_is_rejected_even_on_numpy():
    """The graph builder validates the activation eagerly (before any tracing)."""
    cfg = llama_config(hidden_act="swiglu_magic")
    with pytest.raises(HFUnsupportedError) as excinfo:
        forward_fn(cfg, seq_len=4)
    assert "swiglu_magic" in str(excinfo.value)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def llama_case():
    """A built tiny Llama executable plus its params and ids (shared, read-only)."""
    cfg = llama_config()
    params = make_params(cfg, seed=21)
    ids = make_ids(cfg, batch=2, seq=5, seed=22)
    executable = build(forward_fn(cfg, seq_len=5, batch_size=2), cfg)
    return cfg, params, ids, executable


def test_run_casts_integral_ids_to_int32(llama_case):
    cfg, params, ids, executable = llama_case
    int32 = run(executable, params, ids).numpy()
    for dtype in (np.int64, np.int16, np.uint8, np.int8):
        cast = run(executable, params, ids.astype(dtype)).numpy()
        np.testing.assert_array_equal(cast, int32)


def test_run_rejects_float_ids(llama_case):
    _, params, ids, executable = llama_case
    with pytest.raises(etl.DTypeError) as excinfo:
        run(executable, params, ids.astype(np.float32))
    message = str(excinfo.value)
    assert "float32" in message and "input_ids" in message


def test_run_rejects_a_wrong_id_shape(llama_case):
    _, params, ids, executable = llama_case
    with pytest.raises(etl.ShapeError) as excinfo:
        run(executable, params, ids[:, :3])
    message = str(excinfo.value)
    assert str(ids.shape) in message and "(2, 5)" in message


def test_run_rejects_a_non_2d_id_array(llama_case):
    _, params, ids, executable = llama_case
    with pytest.raises(etl.ShapeError) as excinfo:
        run(executable, params, ids[0])
    assert "2-D" in str(excinfo.value)


def test_run_rejects_a_bad_params_tree(llama_case):
    """A missing/extra param must surface as an etl structure error."""
    _, params, ids, executable = llama_case
    missing = {name: value for name, value in params.items() if not name.startswith("lm_head")}
    with pytest.raises(etl.ETLError) as excinfo:
        run(executable, missing, ids)
    assert "structure does not match" in str(excinfo.value)
    assert "lm_head.weight" in str(excinfo.value)
    extra = dict(params)
    extra["not_a_param"] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(etl.ETLError) as excinfo:
        run(executable, extra, ids)
    assert "structure does not match" in str(excinfo.value)


def test_run_rejects_wrong_argument_types(llama_case):
    _, params, ids, executable = llama_case
    with pytest.raises(TypeError) as excinfo:
        run("not an executable", params, ids)
    assert "etl.Executable" in str(excinfo.value)
    with pytest.raises(TypeError):
        run(executable, [params], ids)


def test_run_accepts_placed_device_params(llama_case):
    _, params, ids, executable = llama_case
    placed = place_params(params, etl.core.Device("cpu", 0))
    np.testing.assert_array_equal(
        run(executable, placed, ids).numpy(),
        run(executable, params, ids).numpy(),
    )


# ---------------------------------------------------------------------------
# run: ids as an etl.Tensor (the device-resident path)
# ---------------------------------------------------------------------------
def test_run_accepts_a_cpu_tensor_of_ids(llama_case):
    _, params, ids, executable = llama_case
    np.testing.assert_array_equal(
        run(executable, params, etl.from_numpy(ids)).numpy(),
        run(executable, params, ids).numpy(),
    )


def test_run_forwards_a_tensor_of_ids_verbatim(llama_case, monkeypatch):
    """The tensor reaches ``etl.run`` as the SAME object — no cast, no copy."""
    _, params, ids, executable = llama_case
    tensor_ids = etl.from_numpy(ids)
    seen: list = []
    real_run = etl.run

    def spy(exe, run_params, run_ids):
        seen.append((exe, run_params, run_ids))
        return real_run(exe, run_params, run_ids)

    monkeypatch.setattr(etl, "run", spy)
    logits = run(executable, params, tensor_ids)
    assert len(seen) == 1
    exe, run_params, run_ids = seen[0]
    assert exe is executable
    assert run_params is params
    assert run_ids is tensor_ids  # no np.asarray, no ascontiguousarray, no cast
    assert isinstance(run_ids, etl.Tensor)
    expected = real_run(executable, params, ids)
    np.testing.assert_array_equal(logits.numpy(), np.asarray(expected.numpy()))


def test_run_does_not_retype_a_tensor_of_ids(llama_case, monkeypatch):
    """A host array is cast to int32; a Tensor's data is never re-typed.

    So an int64 *tensor* reaches etl's dtype-strict run boundary unchanged and
    fails there, while the equivalent host array is cast and runs — which is
    exactly the documented asymmetry, pinned here.
    """
    _, params, ids, executable = llama_case
    int64_tensor = etl.from_numpy(ids.astype(np.int64))
    forwarded: list = []
    real_run = etl.run

    def spy(exe, run_params, run_ids):
        forwarded.append(run_ids)
        return real_run(exe, run_params, run_ids)

    monkeypatch.setattr(etl, "run", spy)
    with pytest.raises(etl.DTypeError) as excinfo:
        run(executable, params, int64_tensor)
    assert forwarded == [int64_tensor], "run must not cast a Tensor"
    assert "int64" in str(excinfo.value)
    # the host-array path performs the one documented coercion
    np.testing.assert_array_equal(
        run(executable, params, ids.astype(np.int64)).numpy(),
        real_run(executable, params, ids).numpy(),
    )


def test_run_host_ids_are_forwarded_as_int32(llama_case, monkeypatch):
    """The unchanged host path still coerces to a contiguous int32 array."""
    _, params, ids, executable = llama_case
    forwarded: list = []
    real_run = etl.run

    def spy(exe, run_params, run_ids):
        forwarded.append(run_ids)
        return real_run(exe, run_params, run_ids)

    monkeypatch.setattr(etl, "run", spy)
    run(executable, params, ids.astype(np.int64))
    assert len(forwarded) == 1
    forwarded_ids = forwarded[0]
    assert isinstance(forwarded_ids, np.ndarray)
    assert forwarded_ids.dtype == np.int32
    assert forwarded_ids.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(forwarded_ids, ids)


def test_run_rejects_bad_tensor_ids(llama_case):
    """The Tensor path keeps the same explicit error table as the array path."""
    _, params, ids, executable = llama_case
    with pytest.raises(etl.ShapeError) as excinfo:
        run(executable, params, etl.from_numpy(ids[0]))
    assert "2-D" in str(excinfo.value) and "etl.Tensor" in str(excinfo.value)
    with pytest.raises(etl.ShapeError) as excinfo:
        run(executable, params, etl.from_numpy(ids[:, :3]))
    message = str(excinfo.value)
    assert "(2, 3)" in message and "(2, 5)" in message and "Tensor" in message
    with pytest.raises(etl.DTypeError) as excinfo:
        run(executable, params, etl.from_numpy(ids.astype(np.float32)))
    message = str(excinfo.value)
    assert "float32" in message and "input_ids" in message


# ---------------------------------------------------------------------------
# place_params
# ---------------------------------------------------------------------------
def test_place_params_is_one_explicit_transfer_per_leaf():
    params = make_params(llama_config(), seed=31)
    device = etl.core.Device("cpu", 0)
    placed = place_params(params, device)
    assert set(placed) == set(params)
    for name, tensor in placed.items():
        assert isinstance(tensor, etl.Tensor)
        assert tensor is not params[name]  # a Tensor wrapping host data, not the array
        assert tensor.device == device
        assert tensor.shape == params[name].shape
        np.testing.assert_array_equal(np.asarray(tensor.numpy()), params[name])


def test_place_params_rejects_a_non_device():
    params = {"a": np.zeros((2,), dtype=np.float32)}
    for bad in ("cpu", 0, None, ("cpu", 0)):
        with pytest.raises(TypeError) as excinfo:
            place_params(params, bad)  # type: ignore[arg-type]
        assert "Device" in str(excinfo.value)


def test_place_params_rejects_bad_params():
    device = etl.core.Device("cpu", 0)
    with pytest.raises(TypeError):
        place_params([("a", np.zeros((2,), np.float32))], device)  # type: ignore[arg-type]
    with pytest.raises(TypeError) as excinfo:
        place_params({1: np.zeros((2,), np.float32)}, device)  # type: ignore[dict-item]
    assert "names must be strings" in str(excinfo.value)
    with pytest.raises(TypeError) as excinfo:
        place_params({"a": object()}, device)
    assert "'a'" in str(excinfo.value)


def test_place_params_accepts_non_contiguous_arrays():
    params = {"a": np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]}
    placed = place_params(params, etl.core.Device("cpu", 0))
    np.testing.assert_array_equal(placed["a"].numpy(), params["a"])


# ---------------------------------------------------------------------------
# place_ids
# ---------------------------------------------------------------------------
def test_place_ids_is_an_explicit_int32_transfer():
    ids = np.arange(6, dtype=np.int64).reshape(2, 3)
    placed = place_ids(ids, etl.core.Device("cpu", 0))
    assert isinstance(placed, etl.Tensor)
    assert placed.device == etl.core.Device("cpu", 0)
    assert placed.shape == (2, 3)
    # the same single documented id coercion run performs on a host array
    assert placed.dtype == np.int32
    np.testing.assert_array_equal(placed.numpy(), ids)


def test_place_ids_moves_an_existing_tensor_without_retyping():
    tensor = etl.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3))
    placed = place_ids(tensor, etl.core.Device("cpu", 0))
    assert placed is tensor, "moving to the tensor's own device is a no-op"
    assert placed.dtype == np.int64, "a Tensor's data is never re-typed"


def test_place_ids_rejects_bad_arguments():
    device = etl.core.Device("cpu", 0)
    with pytest.raises(etl.DTypeError) as excinfo:
        place_ids(np.zeros((2, 3), dtype=np.float32), device)
    assert "place_ids" in str(excinfo.value) and "float32" in str(excinfo.value)
    with pytest.raises(etl.DTypeError):
        place_ids(etl.from_numpy(np.zeros((2, 3), dtype=np.float64)), device)
    with pytest.raises(etl.ShapeError) as excinfo:
        place_ids(np.zeros((3,), dtype=np.int32), device)
    assert "2-D" in str(excinfo.value)
    with pytest.raises(etl.ShapeError):
        place_ids(etl.from_numpy(np.zeros((2, 3, 4), dtype=np.int32)), device)
    for bad in ("cpu", 0, None, ("cpu", 0)):
        with pytest.raises(TypeError) as excinfo:
            place_ids(np.zeros((2, 3), dtype=np.int32), bad)  # type: ignore[arg-type]
        assert "Device" in str(excinfo.value)


def test_place_ids_drives_a_run(llama_case):
    """``place_params`` + ``place_ids`` run bit-identically to the host path."""
    _, params, ids, executable = llama_case
    device = etl.core.Device("cpu", 0)
    np.testing.assert_array_equal(
        run(executable, place_params(params, device), place_ids(ids, device)).numpy(),
        run(executable, params, ids).numpy(),
    )


# ---------------------------------------------------------------------------
# load_hf / LoadedModel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["llama", "gpt2"])
def test_load_hf_reads_config_and_checkpoint(tmp_path, family):
    cfg = llama_config() if family == "llama" else gpt2_config()
    directory = write_checkpoint_dir(tmp_path, cfg)
    loaded = load_hf(directory)
    assert isinstance(loaded, LoadedModel)
    assert loaded.config.digest == cfg.digest
    assert loaded.family == family
    assert loaded.model_dir == directory
    specs = param_specs(loaded.config)
    assert set(loaded.params) == set(specs)
    for name, spec in specs.items():
        assert loaded.params[name].shape == tuple(spec.shape)
        assert loaded.params[name].dtype == np.float32
    seq_len = specialisation(loaded.config)["seq_len"]
    ids = make_ids(cfg, batch=1, seq=seq_len, seed=41)
    got = predict(loaded, input_ids=ids)
    expected = reference_logits(loaded.config, loaded.params, ids)
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)
    assert max_abs(got, expected) <= 1e-6


def test_load_hf_default_seq_len_is_min_max_positions_capped_at_32(tmp_path):
    """Loading knows no length: the default lives on the forward's specialisation."""
    short = llama_config(max_position_embeddings=12)
    loaded = load_hf(write_checkpoint_dir(tmp_path, short, name="short"))
    assert DEFAULT_SEQ_LEN == 32
    assert specialisation(loaded.config)["seq_len"] == 12
    long = llama_config(max_position_embeddings=64)
    loaded = load_hf(write_checkpoint_dir(tmp_path, long, name="long"))
    assert specialisation(loaded.config)["seq_len"] == DEFAULT_SEQ_LEN


def test_predict_honours_an_explicit_seq_len(tmp_path):
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    assert specialisation(loaded.config, seq_len=7)["seq_len"] == 7
    assert predict(loaded, seq_len=7).shape == (1, 7, cfg.vocab_size)


def test_a_bad_seq_len_is_rejected_at_the_program_stage(tmp_path):
    """``load_hf`` says nothing about length; the graph builder validates it."""
    directory = write_checkpoint_dir(tmp_path, llama_config(max_position_embeddings=8))
    cfg = load_config(directory)
    with pytest.raises(HFConfigError):
        forward_fn(cfg, seq_len=9)
    with pytest.raises(HFConfigError):
        forward_fn(cfg, seq_len=0)
    # the checkpoint itself loads fine — the limit is a graph-specialisation fact
    assert load_hf(directory).config.max_position_embeddings == 8


def test_load_hf_family_mismatch_is_explicit(tmp_path):
    directory = write_checkpoint_dir(tmp_path, llama_config())
    with pytest.raises(HFConfigError) as excinfo:
        load_hf(directory, family="gpt2")
    message = str(excinfo.value)
    assert "llama" in message and "gpt2" in message
    with pytest.raises(HFUnsupportedError):
        load_hf(directory, family="bert")


def test_load_hf_family_override_that_agrees_is_accepted(tmp_path):
    directory = write_checkpoint_dir(tmp_path, llama_config())
    assert load_hf(directory, family="llama").family == "llama"


def test_load_hf_missing_config_is_explicit(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(HFConfigError) as excinfo:
        load_hf(empty)
    assert "config.json" in str(excinfo.value)


def test_load_hf_rejects_non_path():
    with pytest.raises(TypeError) as excinfo:
        load_hf(object())  # type: ignore[arg-type]
    assert "path-like" in str(excinfo.value)


def test_build_is_memoised(tmp_path):
    loaded = load_hf(write_checkpoint_dir(tmp_path, llama_config()))
    fwd = forward_fn(loaded.config)
    assert build(fwd, loaded.config) is build(fwd, loaded.config)
    assert build(fwd, loaded.config, backend="numpy") is build(fwd, loaded.config)
    # an equal config from a second load shares the graph (and so the artifact)
    again = load_hf(write_checkpoint_dir(tmp_path, llama_config(), name="again"))
    assert build(forward_fn(again.config), again.config) is build(fwd, loaded.config)


def test_loaded_model_repr_names_family_model_type_and_dir(tmp_path):
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    text = repr(loaded)
    assert "LoadedModel(" in text
    assert "family='llama'" in text
    assert f"model_type={cfg.model_type!r}" in text
    assert str(loaded.model_dir) in text
    # the static shape is NOT loading's business any more — it lives on the graph
    spec = specialisation(loaded.config, seq_len=5)
    assert spec["seq_len"] == 5 and spec["batch_size"] == 1


def test_predict_shape_error_names_the_static_seq_len(tmp_path):
    loaded = load_hf(write_checkpoint_dir(tmp_path, llama_config()))
    with pytest.raises(etl.ShapeError) as excinfo:
        predict(loaded, input_ids=np.zeros((1, 5), dtype=np.int32), seq_len=6)
    message = str(excinfo.value)
    assert "seq_len=6" in message
    assert "etl_hf.forward_fn" in message and "seq_len=5" in message
    assert "etl_hf.predict" in message


def test_predict_defaults_are_shape_only(tmp_path):
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    logits = predict(loaded, seq_len=4)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert logits.dtype == np.float32


def test_predict_and_reference_accept_explicit_params(tmp_path):
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    other = make_params(cfg, seed=99)
    ids = make_ids(cfg, batch=1, seq=4, seed=98)
    got = predict(loaded, ids, params=other, seq_len=4)
    expected = reference_logits(loaded.config, other, ids)
    np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)
    # and it really used `other`, not the checkpoint's own weights
    assert max_abs(got, predict(loaded, ids, seq_len=4)) > 1e-5


def test_forward_fn_bakes_the_static_shape(tmp_path):
    cfg = llama_config()
    forward = forward_fn(cfg, seq_len=4, batch_size=2)
    other = forward_fn(cfg, seq_len=3)
    assert forward is not other
    assert callable(forward)
    # the built graph agrees with the forward's own static shape
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    assert predict(loaded, seq_len=4, batch_size=2).shape == (2, 4, cfg.vocab_size)


def test_validate_params_checks_the_tree():
    """The pre-flight validator replaces the old constructor's tree check."""
    cfg = llama_config()
    params = make_params(cfg, seed=13)
    with pytest.raises(HFWeightError) as excinfo:
        validate_params(cfg, {k: v for k, v in params.items() if k != "embed.weight"})
    assert "missing=" in str(excinfo.value) and "embed.weight" in str(excinfo.value)
    wrong = dict(params)
    wrong["embed.weight"] = np.zeros(
        (cfg.vocab_size + 1, cfg.hidden_size), dtype=np.float32
    )
    with pytest.raises(HFWeightError) as excinfo:
        validate_params(cfg, wrong)
    assert "expected" in str(excinfo.value)
    with pytest.raises(TypeError):
        validate_params(cfg, [("embed.weight", params["embed.weight"])])  # type: ignore[arg-type]
    # a matching tree comes back as a fresh, equal mapping
    validated = validate_params(cfg, params)
    assert validated is not params and set(validated) == set(params)


# ---------------------------------------------------------------------------
# predict is host-facing (device params are refused, never copied back)
# ---------------------------------------------------------------------------
class _FakeDevicePayload:
    """Metadata-only non-cpu payload — no data movement is ever attempted.

    A real cuda/rocm tensor needs a device *and* a transfer provider, neither of
    which exists on a CPU-only host. ``core.Tensor`` documents a duck-typed
    payload protocol (``shape``/``dtype``/``device`` + host materialization), so
    this reproduces the device-resident *shape* of the situation: the code under
    test only reads ``Tensor.device``, which is the point — the helper must
    refuse BEFORE any host copy, so ``to_host()`` asserting proves it never ran.
    """

    def __init__(self, array: np.ndarray, device) -> None:
        self.shape = tuple(array.shape)
        self.dtype = array.dtype
        self.device = device
        self._array = array

    def to_host(self) -> np.ndarray:  # pragma: no cover - must never be called
        raise AssertionError("a non-cpu payload must never be host-copied")


def test_predict_rejects_device_params_with_the_recipe(tmp_path):
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    params = make_params(cfg, seed=17)
    device = etl.core.Device("cuda", 0)
    device_params = dict(params)
    device_params["embed.weight"] = etl.core.Tensor(
        _FakeDevicePayload(params["embed.weight"], device)
    )
    ids = make_ids(cfg, batch=1, seq=4, seed=18)
    with pytest.raises(etl.DeviceError) as excinfo:
        predict(loaded, ids, params=device_params, seq_len=4)
    message = str(excinfo.value)
    assert "embed.weight" in message
    assert str(device) in message  # names where the offending leaf lives
    for named in ("etl_hf.run", "place_params", "place_ids"):
        assert named in message, named
    assert "Device('cpu', 0)" in message  # the one explicit D2H for host logits


def test_predict_accepts_cpu_tensors_and_tensor_ids(tmp_path):
    """cpu Tensors (etl's own rule: host-addressable) keep working."""
    cfg = llama_config()
    loaded = load_hf(write_checkpoint_dir(tmp_path, cfg))
    ids = make_ids(cfg, batch=1, seq=4, seed=19)
    expected = predict(loaded, ids, seq_len=4)
    device = etl.core.Device("cpu", 0)
    np.testing.assert_array_equal(
        predict(loaded, ids, params=place_params(loaded.params, device), seq_len=4),
        expected,
    )
    np.testing.assert_array_equal(
        predict(loaded, place_ids(ids, device), seq_len=4), expected
    )


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------
def test_pipeline_module_imports_without_optional_dependencies():
    """``etl_hf`` and its load/program/pipeline modules stay numpy-only."""
    import subprocess

    code = (
        "import sys, etl_hf, etl_hf.load, etl_hf.pipeline, etl_hf.program;"
        "bad = [m for m in ('torch', 'transformers', 'safetensors') "
        "if m in sys.modules];"
        "bad += [m for m in ('iree', 'iree.compiler') if m in sys.modules];"
        "print(','.join(bad))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""
