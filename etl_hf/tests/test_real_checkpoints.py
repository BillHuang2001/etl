"""Guarded REAL HuggingFace checkpoint suite: checkpoint → ONE etl graph → run.

The package contract is "load a real checkpoint **and** run the whole model
forward pass as ONE ``etl`` graph" (``./etl_hf/CONTEXT.md``).  The rest of
``etl_hf/tests`` pins that contract on synthetic configs and locally written
checkpoints; this module pins it on **real HuggingFace checkpoints read from
disk**, i.e. on the last unverified link: a real ``config.json``, its real
weights, real key names, real transposes, a real legacy ``pytorch_model.bin``, a
real GQA config and real tied heads::

    require_model (download) → load_hf (config + canonical weights)
        → forward (trace → lower → compile → load → run, the WHOLE model)
        → reference (the independent pure-numpy oracle)

Coverage per checkpoint: the real config values (asserted, never assumed), the
canonical param-key count and total parameter count, that every parameter is an
explicit graph **input** (``inputs == keys + 1`` — never an inlined
:class:`etl.constant` weight), the traced IR op count, the StableHLO export size
(zero compiler required), and finally logit parity against
:mod:`etl_hf.reference`.

Guards, in order:

* **Offline ⇒ skip.**  Every download goes through
  :mod:`etl_hf.tests.hf_download` and every test loads its checkpoint with
  :func:`~etl_hf.tests.hf_download.require_model`, which turns "cannot fetch"
  into ``pytest.skip`` — the module is green (skipped) on a machine with no
  network and never hangs.
* **Real-size checkpoints are opt-in** behind ``ETL_HF_BIG=1``
  (``distilgpt2`` ≈ 0.34 GB of weights, ``gpt2`` ≈ 0.52 GB), and the 124M GPT-2
  is additionally skipped when the host does not have the memory headroom.
* **No torch / transformers / iree in the default path**: the oracle is plain
  numpy and the default backend for a cpu device is the numpy interpreter.  The
  single exception is the opt-in packed-weights case
  (``test_real_gpt2_packed_runs_on_iree_llvm_cpu``, section 7), which needs
  ``ETL_HF_BIG=1`` **and** the optional ``iree`` extra and skips cleanly without
  either (``pip install iree-base-compiler iree-base-runtime`` / ``uv run
  --extra iree``).

``sshleifer/tiny-gpt2`` ships the **legacy** (pre-torch-1.6) ``pytorch_model.bin``
layout, so its parity claim is also the reader's end-to-end check: the legacy
trailer prefixes every storage blob with an int64 element count, and reading it
wrong shifts every storage and produces garbage weights (historically:
``~1e-44`` logits instead of the ``~0.128`` these weights actually produce).
``tiny-random-Mistral`` uses ``hidden_act: "gelu"`` (exact erf), a documented
StableHLO v1 deferral, so it runs on the **numpy backend only** — never compiled
here.

The numbers quoted in the docstrings below are what this host measured while
writing the file; every assertion message re-states the numbers it actually
compared, so a failure is self-describing.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import etl
from etl.backends import stablehlo
from etl_hf import ops as hf_ops
from etl_hf import weights
from etl_hf.build import load_hf, place_params, run_model
from etl_hf.errors import HFConfigError, HFWeightError
from etl_hf.tests import hf_download
from etl_hf.tests.test_export import _count_ops

# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

#: Real-size checkpoints are opt-in: they download hundreds of MB of weights and
#: need ~0.7–1.1 GB of RSS, which the default suite must not impose.
_BIG = pytest.mark.skipif(
    os.environ.get("ETL_HF_BIG") != "1",
    reason="real-size checkpoints are opt-in: set ETL_HF_BIG=1 "
    "(downloads distilgpt2 ≈ 0.34 GB / gpt2 ≈ 0.52 GB of weights)",
)


def _iree_available() -> bool:
    """Probe for the IREE Python bindings without importing them into ``etl_hf``.

    The probe lives here (not in :mod:`etl_hf`) so ``import etl_hf`` stays
    numpy-only: a partial/absent install must *skip* the compiled case, never
    error, and never be imported at package scope.
    """
    try:
        import iree.compiler  # noqa: F401 - presence probe only
        import iree.runtime  # noqa: F401 - presence probe only
    except Exception:  # noqa: BLE001 - a broken install must skip, not error
        return False
    return True


IREE_AVAILABLE = _iree_available()

#: The compiled pin (section 7) needs the optional ``iree`` extra.
requires_iree = pytest.mark.skipif(
    not IREE_AVAILABLE,
    reason="iree (iree.compiler / iree.runtime) is not installed; "
    "run with `uv run --extra iree --with pytest`",
)


#: Memory headroom required before the optional 124M GPT-2 is attempted: the
#: measured peak RSS is ~1.1 GB (the canonical 124.4M-parameter float32 tree plus
#: the raw checkpoint dict held while canonicalising, then the reference run), so
#: demand twice that and skip (not fail) on a smaller host.
_BIG_MODEL_MIN_MEMORY = 2 * 1024**3


# ---------------------------------------------------------------------------
# helpers — checkpoint loading, parity, parameter accounting
# ---------------------------------------------------------------------------
def _load(slug: str, **kwargs: object):
    """Load a real checkpoint by slug, skipping cleanly when it is unavailable.

    Args:
        slug: a key of :data:`etl_hf.tests.hf_download.MODELS`.
        **kwargs: forwarded to :func:`etl_hf.load_hf` (``seq_len``,
            ``batch_size``, ``family``).

    Returns:
        etl_hf.HFModel: the loaded model (nothing compiled yet).
    """
    return load_hf(hf_download.require_model(slug), **kwargs)


def _parity(got, want) -> tuple[float, float]:
    """Return ``(max_abs, max_rel_vs_scale)`` between graph logits and the reference.

    ``max_rel_vs_scale`` is the maximum absolute difference divided by the
    reference's own magnitude (``max(1e-45, max|want|)``): the only meaningful
    figure for a checkpoint whose logits are near-denormal (mis-read legacy
    checkpoints used to land there) and the honest one for the real checkpoints,
    whose logits reach |max| ≈ 0.13–147.

    Args:
        got: logits produced by the compiled/in-``etl`` forward.
        want: logits produced by the independent numpy reference.

    Returns:
        tuple: ``(max_abs, max_rel_vs_scale)`` as floats, computed in float64 so
        the subtraction itself introduces no rounding.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, (
        f"_parity needs matching shapes, got graph {got.shape} vs reference {want.shape}"
    )
    if want.size == 0:
        return 0.0, 0.0
    max_abs = float(np.max(np.abs(got - want)))
    scale = float(np.max(np.abs(want)))
    return max_abs, max_abs / max(1e-45, scale)


def _ids(batch: int, seq_len: int) -> np.ndarray:
    """Deterministic token ids ``0 .. batch*seq_len-1`` as int32 ``(batch, seq_len)``."""
    return np.arange(batch * seq_len, dtype=np.int32).reshape(batch, seq_len)


def _total_params(tree) -> int:
    """Sum of the element counts of a params (arrays) or spec (``TensorSpec``) tree."""
    return sum(int(np.prod(tuple(leaf.shape), dtype=np.int64)) for leaf in tree.values())


def _llama_param_count(cfg) -> int:
    """Parameter count of the llama family computed independently from the config.

    Per layer: ``q``/``o`` are ``(hidden, hidden)``, ``k``/``v`` are
    ``(hidden, kv_heads * head_dim)`` (GQA-aware), the two RMSNorm scales are
    ``hidden`` each, and the SwiGLU MLP is three ``hidden × intermediate``
    matrices. The token embedding is ``vocab × hidden`` and the final norm
    ``hidden``; ``lm_head`` is its own ``hidden × vocab`` matrix **only** when the
    config is untied — a tied head IS ``embed.weight`` (one graph input, used
    transposed), so it contributes nothing. Used to check the loaded checkpoint
    against arithmetic rather than against itself.

    Args:
        cfg: an :class:`etl_hf.config.ModelConfig` of the ``llama`` family.

    Returns:
        int: expected total parameter count of the canonical tree.
    """
    hidden = cfg.hidden_size
    kv_width = cfg.num_key_value_heads * cfg.head_dim
    per_layer = (
        2 * hidden * hidden  # q + o
        + 2 * hidden * kv_width  # k + v (GQA)
        + 2 * hidden  # attn_norm + mlp_norm scales
        + 3 * hidden * cfg.intermediate_size  # gate + up + down
    )
    return (
        cfg.num_hidden_layers * per_layer
        + cfg.vocab_size * hidden  # embed
        + (0 if cfg.tie_word_embeddings else hidden * cfg.vocab_size)  # untied lm_head
        + hidden  # final_norm
    )

def _gpt2_param_count(cfg) -> int:
    """Parameter count of the gpt2 family computed independently from the config.

    Per layer: four biased ``(hidden, hidden)`` attention projections, two biased
    LayerNorms (``2 * hidden`` each), and the biased MLP pair ``(hidden, inner)``
    / ``(inner, hidden)``. Plus the token embedding ``vocab × hidden``, the
    learned positional table ``positions × hidden`` and a biased final LayerNorm.
    The output head is a separate ``hidden × vocab`` matrix **only** when the
    config is untied — a tied head IS ``embed.weight`` (one graph input, used
    transposed), so it contributes nothing.

    Args:
        cfg: an :class:`etl_hf.config.ModelConfig` of the ``gpt2`` family.

    Returns:
        int: expected total parameter count of the canonical tree.
    """
    hidden = cfg.hidden_size
    inner = cfg.intermediate_size
    per_layer = (
        4 * (hidden * hidden + hidden)  # q, k, v, o weights + biases
        + 2 * (2 * hidden)  # attn_norm + mlp_norm (weight + bias)
        + (hidden * inner + inner)  # fc1 (weight + bias)
        + (inner * hidden + hidden)  # fc2 (weight + bias)
    )
    return (
        cfg.num_hidden_layers * per_layer
        + cfg.vocab_size * hidden  # embed
        + cfg.max_position_embeddings * hidden  # learned positions
        + 2 * hidden  # final_norm (weight + bias)
        + (0 if cfg.tie_word_embeddings else hidden * cfg.vocab_size)  # untied lm_head
    )


def _trace(model):
    """Trace the model's whole forward as ONE graph and verify it.

    The argument tree is the model's own ``param_specs`` (weights are explicit
    **inputs**, never ``etl.constant`` payloads) plus the int32 ``input_ids`` spec
    — the same staging :meth:`etl_hf.HFModel.build` performs.

    Args:
        model: an :class:`etl_hf.HFModel`.

    Returns:
        etl.Graph: the verified graph of the whole forward pass.
    """
    graph = etl.trace(
        model.graph_fn(),
        model.param_specs(),
        etl.TensorSpec((model.batch_size, model.seq_len), etl.int32),
    )
    graph.verify()
    return graph


def _ops_total(graph) -> int:
    """Total number of IR ops in a traced graph (nested regions included)."""
    return sum(_count_ops(graph).values())


def _describe(model, graph=None) -> str:
    """One-line description of a loaded model (plus its graph) for assertion messages.

    Args:
        model: an :class:`etl_hf.HFModel`.
        graph: optional traced graph; when given, the op and input counts are added.

    Returns:
        str: e.g. ``"tiny_llama family=llama seq_len=8 batch=1 keys=21 params=1,032,272
        ops=170 inputs=22"``.
    """
    params = model.params()
    parts = [
        f"family={model.family}",
        f"model_type={model.config.model_type}",
        f"seq_len={model.seq_len}",
        f"batch={model.batch_size}",
        f"keys={len(params)}",
        f"params={_total_params(params):,}",
    ]
    if graph is not None:
        parts += [f"ops={_ops_total(graph)}", f"inputs={len(graph.tensor_specs)}"]
    return " ".join(parts)


def _check_forward(
    model,
    input_ids,
    *,
    atol: float,
    rtol: float,
    graph=None,
    spread: bool = True,
    **build_kwargs: object,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run ``forward`` and ``reference``, assert shape/dtype/finiteness/parity.

    Args:
        model: an :class:`etl_hf.HFModel`.
        input_ids: ``(batch_size, seq_len)`` integer ids.
        atol: maximum permitted absolute difference.
        rtol: maximum permitted difference **relative to the logit scale**
            (see :func:`_parity`) — the bound that survives a denormal checkpoint.
        graph: optional traced graph, only used to enrich the message.
        spread: when True also require the reference logits to be non-constant
            (a graph collapsing to a constant would otherwise pass against a
            collapsed oracle); disable for a genuinely degenerate checkpoint.
        **build_kwargs: forwarded to :meth:`etl_hf.HFModel.forward` (``backend``…).

    Returns:
        tuple: ``(got, want, max_abs, max_rel_vs_scale)`` — the measured logits
        and their measured numbers (callers report them).
    """
    got = model.forward(input_ids=input_ids, **build_kwargs)
    want = model.reference(input_ids=input_ids)
    expected = (model.batch_size, model.seq_len, model.config.vocab_size)
    assert got.shape == want.shape == expected, (
        f"logits must be {expected} (batch, seq_len, vocab), got graph {got.shape} "
        f"vs reference {want.shape} [{_describe(model, graph)}]"
    )
    assert got.dtype == np.dtype(np.float32) and want.dtype == np.dtype(np.float32), (
        f"logits must be float32, got graph {got.dtype} vs reference {want.dtype}"
    )
    max_abs, max_rel = _parity(got, want)
    assert bool(np.isfinite(got).all()), f"graph logits are not finite [{_describe(model, graph)}]"
    assert np.isfinite(want).all(), f"reference logits are not finite [{_describe(model, graph)}]"
    if spread:
        assert float(np.std(want)) > 0.0, (
            f"the reference logits are constant — the oracle (and therefore this parity "
            f"claim) would be vacuous [{_describe(model, graph)}]"
        )
    assert max_abs <= atol and max_rel <= rtol, (
        f"parity failure vs the numpy reference: max_abs={max_abs:.3e} "
        f"(budget {atol:.1e}), max_rel_vs_scale={max_rel:.3e} (budget {rtol:.1e}) "
        f"[{_describe(model, graph)}]"
    )
    return got, want, max_abs, max_rel


def _available_memory_bytes() -> int:
    """Host memory available to this process in bytes (``0`` when undeterminable).

    A ``0`` result means "unknown", never "none": the optional big-model guard
    then proceeds instead of skipping on a non-POSIX host.
    """
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):  # pragma: no cover - non-POSIX host
        return 0


# ---------------------------------------------------------------------------
# 1. tiny-random-LlamaForCausalLM — safetensors, 2 layers, no GQA
# ---------------------------------------------------------------------------
def test_tiny_random_llama_matches_the_numpy_reference():
    """An **untied** safetensors checkpoint → 21 canonical tensors (1,032,272 params) → 3e-8."""
    model = _load("tiny_llama", seq_len=8)
    cfg = model.config
    # The real values from this checkpoint's config.json (asserted, not assumed).
    assert cfg.family == "llama"
    assert cfg.model_type == "llama"
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (2, 16)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (4, 4, 4)
    assert (cfg.intermediate_size, cfg.vocab_size) == (64, 32000)
    assert (cfg.max_position_embeddings, cfg.hidden_act) == (2048, "silu")
    assert cfg.tie_word_embeddings is False
    assert hf_ops.activation_is_export_safe(cfg.hidden_act) is True

    params = model.params()
    specs = model.param_specs()
    assert set(params) == set(specs), "params() and param_specs() must describe one tree"
    assert len(params) == 21
    # Untied: the head is its own canonical array, never an alias of the embedding.
    assert "lm_head.weight" in params and "lm_head.weight" in specs, (
        "tie_word_embeddings=False must keep a separate lm_head.weight entry"
    )
    assert params["lm_head.weight"].shape == (cfg.hidden_size, cfg.vocab_size), (
        f"the head must be (hidden, vocab) = {(cfg.hidden_size, cfg.vocab_size)}, "
        f"got {params['lm_head.weight'].shape}"
    )
    assert not np.allclose(params["lm_head.weight"], params["embed.weight"].T), (
        "an untied head is its own checkpoint tensor — it must not equal the embedding"
    )
    assert all(isinstance(value, np.ndarray) for value in params.values())
    assert {value.dtype for value in params.values()} == {np.dtype(np.float32)}
    # Independent accounting (see _llama_param_count) and the spec-tree arithmetic.
    expected = _llama_param_count(cfg)
    assert expected == 1_032_272, f"llama param arithmetic changed: {expected:,}"
    assert _total_params(params) == _total_params(specs) == expected, (
        f"parameter accounting: params={_total_params(params):,} "
        f"specs={_total_params(specs):,} arithmetic={expected:,} [{_describe(model)}]"
    )

    graph = _trace(model)
    # Every parameter is an explicit graph input; only input_ids is added.
    assert len(graph.tensor_specs) == len(params) + 1, (
        f"weights must be graph inputs (one leaf each) + one ids input: "
        f"{len(graph.tensor_specs)} inputs for {len(params)} parameters"
    )
    assert _ops_total(graph) > 150, f"suspiciously small graph [{_describe(model, graph)}]"
    text = stablehlo.export(graph)
    constants = text.count("stablehlo.constant")
    assert len(text) < 200_000 and constants <= 100, (
        f"the export must stay compact — weights are inputs, never inlined constants: "
        f"{len(text)} bytes / {len(text.splitlines())} lines / {constants} constants "
        f"[{_describe(model, graph)}]"
    )

    got, _, max_abs, _ = _check_forward(model, _ids(1, 8), atol=1e-5, rtol=1e-6, graph=graph)
    assert float(np.abs(got).max()) > 0.0, "the graph's logits are all zero"
    # Measured here: max_abs ≈ 3e-8, max_rel_vs_scale ≈ 8e-8 (fp32 round-off).
    assert max_abs < 1e-6, f"tiny_llama parity regressed: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# 2. tiny-random-MistralForCausalLM — legacy torch .bin, GQA 4q/2kv, exact-erf GELU
# ---------------------------------------------------------------------------
def test_tiny_random_mistral_legacy_bin_gqa_matches_the_numpy_reference():
    """A legacy ``pytorch_model.bin`` with real GQA (4 q / 2 kv) → parity 9e-8 (1.5e-7 rel)."""
    directory = hf_download.require_model("tiny_mistral")
    # This checkpoint predates safetensors: a pass here proves the legacy torch
    # reader works on a real file (there is no *.safetensors fallback to hide behind).
    assert not list(directory.glob("*.safetensors")), (
        f"expected a legacy-only checkpoint directory, found safetensors in {directory}"
    )
    assert (directory / "pytorch_model.bin").is_file()
    raw = weights.load_checkpoint(directory)
    assert len(raw) > 0, f"the legacy .bin reader returned no tensors from {directory}"

    model = load_hf(directory, seq_len=8)
    cfg = model.config
    assert (cfg.model_type, cfg.family) == ("mistral", "llama")
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (2, 32)
    assert cfg.num_key_value_heads < cfg.num_attention_heads, "this checkpoint is real GQA"
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (4, 2, 8)
    assert (cfg.intermediate_size, cfg.vocab_size, cfg.max_position_embeddings) == (37, 32000, 512)
    assert cfg.tie_word_embeddings is False
    # This config carries hidden_act "gelu" = exact erf, a documented StableHLO v1
    # deferral — the graph is numpy-backend-only and is never compiled here.
    assert cfg.hidden_act == "gelu"
    assert hf_ops.activation_is_export_safe(cfg.hidden_act) is False

    params = model.params()
    assert len(params) == 21
    assert "lm_head.weight" in params, "tie_word_embeddings=False keeps a separate head"
    assert {value.dtype for value in params.values()} == {np.dtype(np.float32)}
    expected = _llama_param_count(cfg)
    assert expected == 2_061_408, f"mistral param arithmetic changed: {expected:,}"
    assert _total_params(params) == expected, (
        f"parameter accounting: params={_total_params(params):,} arithmetic={expected:,} "
        f"[{_describe(model)}]"
    )
    # GQA really narrows k/v in the canonical tree.
    assert params["layers.0.attn.k.weight"].shape == (32, 16)
    assert params["layers.0.attn.q.weight"].shape == (32, 32)

    graph = _trace(model)
    assert len(graph.tensor_specs) == len(params) + 1
    assert _ops_total(graph) > 150, f"suspiciously small graph [{_describe(model, graph)}]"
    with pytest.raises(etl.BackendError) as excinfo:
        stablehlo.export(graph)
    assert "erf" in str(excinfo.value), (
        f"the erf-GELU deferral must name the offending op, got: {excinfo.value}"
    )

    _, _, max_abs, _ = _check_forward(
        model, _ids(1, 8), atol=1e-5, rtol=1e-6, graph=graph, backend="numpy"
    )
    # Measured here: max_abs ≈ 9e-8, max_rel_vs_scale ≈ 1.5e-7.
    assert max_abs < 1e-6, f"tiny_mistral parity regressed: max_abs={max_abs:.3e}"


# ---------------------------------------------------------------------------
# 3. sshleifer/tiny-gpt2 — legacy .bin (the legacy reader's end-to-end check)
# ---------------------------------------------------------------------------
def test_tiny_gpt2_legacy_checkpoint_matches_the_numpy_reference():
    """``n_embd=2`` GPT-2 read from a real **legacy** ``pytorch_model.bin`` → parity 1.5e-8.

    The checkpoint ships a ``lm_head.weight`` that is a byte-copy of its
    ``wte.weight`` and its config ties the two, so this is also the real-checkpoint
    pin for the tied contract: the shipped head is verified against the embedding,
    consumed and dropped, and the canonical tree is **36 keys / 102,714 params**
    with exactly one embedding array and no ``lm_head.weight``.

    Historical note: while the legacy reader ignored the per-storage int64
    element-count prefix, every storage was read 8·k bytes off and this
    checkpoint's logits came out ~1e-44 (and constant) instead of the ~0.128
    its weights really produce — a wrong-scale parity claim that had to be
    stated relatively. With the prefix consumed the weights are the real ones,
    so the normal absolute budget applies again.
    """
    model = _load("tiny_gpt2", seq_len=8)
    cfg = model.config
    assert (cfg.model_type, cfg.family) == ("gpt2", "gpt2")
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers) == (2, 2, 2)
    assert (cfg.vocab_size, cfg.max_position_embeddings) == (50257, 1024)
    assert cfg.hidden_act == "gelu_new"  # the tanh approximation — exportable
    assert hf_ops.activation_is_export_safe(cfg.hidden_act) is True
    assert cfg.tie_word_embeddings is True
    assert (cfg.head_dim, cfg.intermediate_size) == (1, 8)

    directory = hf_download.require_model("tiny_gpt2")
    assert not list(directory.glob("*.safetensors")), "tiny-gpt2 ships a legacy torch .bin"
    raw = weights.load_checkpoint(directory)
    assert len(raw) > 0
    # The shipped head is the tied copy: it must be identical to the embedding the
    # canonicaliser keeps (this is what makes consuming-and-dropping it safe).
    assert "lm_head.weight" in raw, "this checkpoint is expected to ship its tied head"
    embed_key = "transformer.wte.weight" if "transformer.wte.weight" in raw else "wte.weight"
    assert embed_key in raw, f"tiny-gpt2 must ship the token embedding, keys: {sorted(raw)}"
    np.testing.assert_array_equal(
        raw["lm_head.weight"],
        raw[embed_key],
    )

    params = model.params()
    specs = model.param_specs()
    assert set(params) == set(specs)
    assert len(params) == 36
    assert "lm_head.weight" not in params and "lm_head.weight" not in specs, (
        f"a tied config has no separate head: {sorted(name for name in params if 'lm_head' in name)}"
    )
    assert params["embed.weight"].shape == (cfg.vocab_size, cfg.hidden_size), (
        f"the one tied embedding must stay (vocab, hidden) = "
        f"{(cfg.vocab_size, cfg.hidden_size)}, got {params['embed.weight'].shape}"
    )
    assert {value.dtype for value in params.values()} == {np.dtype(np.float32)}
    expected = _gpt2_param_count(cfg)
    assert expected == 102_714, f"gpt2 param arithmetic changed: {expected:,}"
    assert _total_params(params) == _total_params(specs) == expected, (
        f"parameter accounting: params={_total_params(params):,} "
        f"specs={_total_params(specs):,} arithmetic={expected:,} [{_describe(model)}]"
    )

    graph = _trace(model)
    assert len(graph.tensor_specs) == len(params) + 1, (
        f"the tied head must not add a graph input: {len(graph.tensor_specs)} inputs for "
        f"{len(params)} parameters [{_describe(model, graph)}]"
    )
    # Measured here: max|ref| ≈ 0.128 (normal scale — the legacy reader now reads
    # the real weights), max_abs ≈ 1.5e-8, max_rel_vs_scale ≈ 1.2e-7. The scale is
    # pinned so a checkpoint (or a reader) that produced degenerate logits again
    # would be noticed instead of passing a relative-only claim.
    _, want, max_abs, max_rel = _check_forward(model, _ids(1, 8), atol=1e-5, rtol=1e-6, graph=graph)
    scale = float(np.abs(want).max())
    assert 0.1 < scale < 1.0, (
        f"tiny-gpt2 reference logits should be normal-scale (~0.128), got max|ref|={scale:.3e}"
    )
    assert max_abs < 1e-6, f"tiny_gpt2 absolute parity regressed: max_abs={max_abs:.3e}"
    assert max_rel <= 1e-6, f"tiny_gpt2 relative parity regressed: {max_rel:.3e}"


# ---------------------------------------------------------------------------
# 4. longer context / batch size — one static shape per graph
# ---------------------------------------------------------------------------
def test_longer_context_and_batch_size_two_build_and_match():
    """``seq_len=16`` and ``batch_size=2`` both load, build, run and match."""
    directory = hf_download.require_model("tiny_llama")

    longer = load_hf(directory, seq_len=16)
    assert (longer.seq_len, longer.batch_size) == (16, 1)
    graph = _trace(longer)
    assert len(graph.tensor_specs) == len(longer.params()) + 1
    _check_forward(longer, _ids(1, 16), atol=1e-5, rtol=1e-6, graph=graph)

    batched = load_hf(directory, seq_len=16, batch_size=2)
    assert (batched.seq_len, batched.batch_size) == (16, 2)
    assert batched.build() is not longer.build(), "batch is a static shape, hence a new graph"
    got, _, _, _ = _check_forward(batched, _ids(2, 16), atol=1e-5, rtol=1e-6)
    assert got.shape == (2, 16, longer.config.vocab_size)
    # The two rows run the same graph, so a batch of identical rows must agree.
    single = load_hf(directory, seq_len=16)
    row = _ids(1, 16)
    assert np.array_equal(
        batched.forward(input_ids=np.repeat(row, 2, axis=0))[0], single.forward(input_ids=row)[0]
    )

    limit = longer.config.max_position_embeddings
    with pytest.raises(HFConfigError) as excinfo:
        load_hf(directory, seq_len=limit + 1)
    message = str(excinfo.value)
    assert "max_position_embeddings" in message and str(limit) in message, message


# ---------------------------------------------------------------------------
# 5. static-shape policy pinned
# ---------------------------------------------------------------------------
def test_static_shape_policy_sequence_length_is_baked_into_the_graph():
    """One ``seq_len`` = one executable; the IR op count is ``seq_len``-independent."""
    directory = hf_download.require_model("tiny_llama")
    short = load_hf(directory, seq_len=4)
    long_ = load_hf(directory, seq_len=8)
    again = load_hf(directory, seq_len=8)

    executable_short = short.build()
    executable_long = long_.build()
    assert executable_short is not executable_long, (
        "a different static seq_len must produce a different executable"
    )
    assert again.build() is executable_long, (
        "the same static shape must reuse the memoised executable (no recompile)"
    )

    graph_short = _trace(short)
    graph_long = _trace(long_)
    ops_short, ops_long = _ops_total(graph_short), _ops_total(graph_long)
    assert ops_short == ops_long, (
        f"the IR op count must not depend on seq_len: seq4={ops_short} ops, "
        f"seq8={ops_long} ops [{_describe(short, graph_short)} | {_describe(long_, graph_long)}]"
    )
    assert ops_short > 100, f"suspiciously small graph [{_describe(short, graph_short)}]"

    # The baked-in length is enforced, not silently adapted.
    with pytest.raises(etl.ShapeError) as excinfo:
        short.forward(input_ids=_ids(1, 8))
    assert "seq_len=4" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. env-gated real-size checkpoints (ETL_HF_BIG=1)
# ---------------------------------------------------------------------------
@_BIG
def test_real_distilgpt2_end_to_end():
    """distilgpt2 (6 layers / 768 wide / 81,912,576 params) → parity ~1.5e-4 abs."""
    directory = hf_download.require_model("distilgpt2")
    model = load_hf(directory, seq_len=32)
    cfg = model.config
    assert (cfg.model_type, cfg.family) == ("gpt2", "gpt2")
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (6, 768)
    assert (cfg.num_attention_heads, cfg.head_dim) == (12, 64)
    assert (cfg.intermediate_size, cfg.vocab_size) == (3072, 50257)
    assert (cfg.max_position_embeddings, cfg.hidden_act) == (1024, "gelu_new")
    assert cfg.tie_word_embeddings is True
    assert hf_ops.activation_is_export_safe(cfg.hidden_act) is True

    params = model.params()
    specs = model.param_specs()
    assert set(params) == set(specs)
    assert len(params) == 100
    # Tied: the head IS the embedding, so the canonical tree is exactly the real
    # model's parameter count and carries no duplicate lm_head array.
    assert "lm_head.weight" not in params and "lm_head.weight" not in specs, (
        f"distilgpt2 is tied: no separate head, got "
        f"{sorted(name for name in params if 'lm_head' in name)}"
    )
    assert {value.dtype for value in params.values()} == {np.dtype(np.float32)}
    expected = _gpt2_param_count(cfg)
    assert expected == 81_912_576, f"distilgpt2 param arithmetic changed: {expected:,}"
    assert _total_params(params) == _total_params(specs) == expected, (
        f"parameter accounting: params={_total_params(params):,} "
        f"specs={_total_params(specs):,} arithmetic={expected:,} [{_describe(model)}]"
    )
    graph = _trace(model)
    assert len(graph.tensor_specs) == len(params) + 1, (
        f"graph inputs: {len(graph.tensor_specs)} for {len(params)} parameters "
        f"[{_describe(model, graph)}]"
    )
    assert _ops_total(graph) > 400, (
        f"a 6-layer GPT-2 should be a large graph [{_describe(model, graph)}]"
    )
    text = stablehlo.export(graph)
    assert len(text) < 400_000 and text.count("stablehlo.constant") <= 200, (
        f"export must not inline weights: {len(text)} bytes / {len(text.splitlines())} lines / "
        f"{text.count('stablehlo.constant')} constants [{_describe(model, graph)}]"
    )

    _, _, max_abs, max_rel = _check_forward(
        model, _ids(1, 32), atol=1e-3, rtol=1e-5, graph=graph, backend="numpy"
    )
    # Measured here: max_abs ≈ 1.6e-4 (≈ 2.3e-6 relative to |logits|max ≈ 68) —
    # fp32 accumulation-order noise, well inside the 1e-5 relative budget, and the
    # reason the absolute bound is 1e-3 rather than 1e-4 (1.6e-4 > 1e-4).
    assert max_rel <= 1e-5, f"distilgpt2 relative parity regressed: {max_rel:.3e}"
    start = time.perf_counter()
    model.forward(input_ids=_ids(1, 32), backend="numpy")
    elapsed = time.perf_counter() - start
    assert elapsed < 60.0, (
        f"a warm distilgpt2 forward took {elapsed:.2f} s (measured ~0.2 s here) "
        f"[{_describe(model, graph)}]"
    )


@_BIG
def test_real_gpt2_end_to_end_when_memory_allows():
    """gpt2 124M (12 layers / 124,439,808 canonical params) → ~2.1e-4 abs, best effort."""
    available = _available_memory_bytes()
    if 0 < available < _BIG_MODEL_MIN_MEMORY:
        pytest.skip(
            f"gpt2 124M needs ~1.1 GB RSS; only {available / 1024**2:.0f} MB of host memory "
            f"is available"
        )
    directory = hf_download.require_model("gpt2")
    model = load_hf(directory, seq_len=32)
    cfg = model.config
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (12, 768)
    assert (cfg.num_attention_heads, cfg.head_dim) == (12, 64)
    assert (cfg.vocab_size, cfg.hidden_act) == (50257, "gelu_new")
    assert cfg.tie_word_embeddings is True

    params = model.params()
    assert len(params) == 196
    # Tied: no duplicate head array, so the canonical tree equals the 124M model.
    assert "lm_head.weight" not in params and "lm_head.weight" not in model.param_specs(), (
        f"gpt2 is tied: no separate head, got "
        f"{sorted(name for name in params if 'lm_head' in name)}"
    )
    expected = _gpt2_param_count(cfg)
    assert expected == 124_439_808, f"gpt2 param arithmetic changed: {expected:,}"
    assert _total_params(params) == expected, (
        f"parameter accounting: params={_total_params(params):,} arithmetic={expected:,} "
        f"[{_describe(model)}]"
    )
    graph = _trace(model)
    assert len(graph.tensor_specs) == len(params) + 1
    assert _ops_total(graph) > 400, (
        f"a 12-layer GPT-2 should be a large graph [{_describe(model, graph)}]"
    )
    assert len(stablehlo.export(graph)) < 600_000, (
        f"export must not inline weights [{_describe(model, graph)}]"
    )

    _, _, max_abs, max_rel = _check_forward(
        model, _ids(1, 32), atol=1e-3, rtol=1e-5, graph=graph, backend="numpy"
    )
    # Measured here: max_abs ≈ 2.1e-4 (≈ 1.5e-6 relative to |logits|max ≈ 147).
    assert max_rel <= 1e-5, f"gpt2 relative parity regressed: {max_rel:.3e}"


# ---------------------------------------------------------------------------
# 7. packed weights on iree/llvm-cpu — the 124M GPT-2 payoff (ETL_HF_BIG=1)
# ---------------------------------------------------------------------------
_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "bench_logs" / "hf_goldens"

#: The IREE Python runtime limit packing exists for: a program that CONSUMES more
#: than this many distinct inputs SIGSEGVs on the first run (measured bracket
#: 161 works / 162 crashes — see ``./etl_hf/CONTEXT.md``).
IREE_CONSUMED_INPUT_CLIFF = 161

#: The packed 124M ``gpt2`` plan: 3 concatenated weight inputs (``packed.0``
#: ``(125009, 768)``, ``packed.1`` ``(121344,)``, ``packed.2`` ``(9216, 3072)``)
#: plus the ``input_ids`` leaf = **4** declared inputs, versus 197 unpacked.
PACKED_GPT2_INPUTS = 3
PACKED_GPT2_DECLARED = PACKED_GPT2_INPUTS + 1

#: fp32 parity budget of a compiled graph against the numpy interpreter, relative
#: to the logit scale (``max|logits|`` ≈ 139 for this checkpoint).
PACKED_RTOL = 1e-5
PACKED_ATOL = 1e-3

#: Consecutive runs the compiled packed executable must survive unchanged (the
#: unpacked graph crashes on the FIRST run — exit 139 — so repetition is the
#: interesting part).
PACKED_STABILITY_RUNS = 12

#: The packed ``iree``/llvm-cpu build peaks at ~1.7 GB RSS (canonical 124.4M f32
#: tree + its 475 MB packed copy + the compiled artifact), so demand ~3 GB of
#: available host memory and skip (never fail) on a smaller host.
_PACKED_MIN_MEMORY = 3 * 1024**3

#: Numbers measured on the host that wrote this file for the packed 124M ``gpt2``
#: on ``iree``/``llvm-cpu`` at ``seq_len=32`` — **reported, never asserted**: a
#: timing threshold would turn a busy host into a red test, and none of these is
#: a correctness property.  (``first_call``/``warm`` are best-of-12 medians; the
#: numpy interpreter stays far ahead on CPU, so packing's payoff here is
#: *runnability*, not speed — 197 → 4 inputs, at numpy-equivalent latency and
#: with the same fp32 numbers.)
MEASURED: dict[str, object] = {
    "canonical_keys": 196,
    "packed_inputs": PACKED_GPT2_INPUTS,
    "declared_inputs": PACKED_GPT2_DECLARED,
    "load_hf_s": 0.46,
    "pack_s": 0.12,
    "trace_s": 0.22,
    "lower_s": 0.12,
    "compile_s": 3.2,
    "load_s": 0.02,
    "build_s": 3.4,
    "first_call_ms": 417.0,
    "warm_median_ms": 405.0,
    "warm_best_ms": 401.0,
    "warm_stdev_ms": 3.0,
    "numpy_packed_median_ms": 29.6,
    "numpy_unpacked_median_ms": 29.4,
    "iree_vs_numpy_max_abs": 3.5e-4,
    "iree_vs_numpy_rel": 2.5e-6,
    "iree_vs_golden_max_abs": 3.1e-4,
    "iree_vs_golden_rel": 2.2e-6,
    "unpacked_control": "SIGSEGV: exit 139, empty stderr, after a successful "
    "compile + load of the 197-input program",
}


def _golden_logits(slug: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``<repo>/bench_logs/hf_goldens/<slug>.npz`` as ``(input_ids, logits)``.

    The fixtures are plain ``numpy.savez`` archives (never pickled), so they are
    read with ``allow_pickle=False`` and validated by key.  A missing or
    malformed fixture **skips** — ``bench_logs/`` is a read-only sibling area, so
    this module must not fail over an absent artifact it cannot regenerate.

    Args:
        slug: fixture name (``"gpt2"``).

    Returns:
        tuple: ``(input_ids, logits)`` — ``(batch, seq_len)`` int ids and the
        real HuggingFace ``(batch, seq_len, vocab_size)`` float32 logits.
    """
    path = _GOLDEN_DIR / f"{slug}.npz"
    if not path.is_file():
        pytest.skip(f"no golden fixture at {path} (read-only bench_logs/ area)")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "input_ids" not in archive.files or "logits" not in archive.files:
                pytest.skip(f"{path} carries no input_ids/logits arrays")
            return np.asarray(archive["input_ids"]), np.asarray(archive["logits"])
    except (OSError, ValueError, EOFError) as exc:  # pragma: no cover - broken fixture
        pytest.skip(f"{path} is not a readable numpy archive: {exc}")


@_BIG
@requires_iree
def test_real_gpt2_packed_runs_on_iree_llvm_cpu():
    """124M ``gpt2`` packed → **4** inputs instead of 197, runs 12× unchanged on iree.

    The unpacked 124M GPT-2 compiles and loads fine on ``iree``/llvm-cpu and then
    dies on the first run — IREE's Python runtime SIGSEGVs on a program that
    consumes more than ~161 distinct inputs (measured control, in its own
    subprocess: **exit 139, empty stderr**, 197 declared inputs).  Packing
    (:mod:`etl_hf.pack`, ``pack_weights=True``) is what makes the case runnable:
    the same weights arrive as 3 concatenated inputs and are sliced back in-graph.

    What this pins:

    * the packed plan really is the small number (3 weight inputs, 4 declared with
      ``input_ids``) and covers every canonical element exactly once;
    * the compiled executable **runs 12 times without crashing** and every run is
      **bit-identical** to the first (the unpacked program never reaches run #1);
    * the numbers are unchanged by packing: fp32 parity against the numpy
      interpreter's packed forward (relative-to-absmax budget
      :data:`PACKED_RTOL`) and against the stored HuggingFace golden
      (:data:`PACKED_ATOL` / :data:`PACKED_RTOL`).

    No timing assertion: the measured latencies live in :data:`MEASURED` (this
    host: build ≈ 3.4 s, first call ≈ 0.42 s, warm median ≈ 0.41 s — the numpy
    interpreter is ~14× faster at this size, so the compiled payoff is
    runnability, not speed).
    """
    available = _available_memory_bytes()
    if 0 < available < _PACKED_MIN_MEMORY:
        pytest.skip(
            f"the packed 124M gpt2 build peaks at ~1.7 GB RSS; only "
            f"{available / 1024**2:.0f} MB of host memory is available"
        )
    input_ids, want_golden = _golden_logits("gpt2")
    if input_ids.shape != (1, 32):
        pytest.skip(f"golden gpt2 ids shape {input_ids.shape} is not (1, 32)")

    model = load_hf(hf_download.require_model("gpt2"), seq_len=32)
    params = model.params()
    assert len(params) == 196, f"expected 196 canonical keys, got {len(params)}"

    # --- the packed plan: 196 canonical arrays → 3 concatenated inputs -------
    plan = model.pack_plan()
    assert len(plan.specs) == PACKED_GPT2_INPUTS, (
        f"the 124M gpt2 must pack to {PACKED_GPT2_INPUTS} weight inputs, got "
        f"{len(plan.specs)}: "
        f"{ {name: tuple(map(int, spec.shape)) for name, spec in plan.specs.items()} }"
    )
    assert list(plan.specs) == [f"packed.{i}" for i in range(PACKED_GPT2_INPUTS)], (
        f"packed inputs must be named packed.0..packed.{PACKED_GPT2_INPUTS - 1}, got "
        f"{list(plan.specs)}"
    )
    assert len(model.param_specs(pack_weights=True)) == PACKED_GPT2_INPUTS
    packed_elements = sum(
        int(np.prod(tuple(map(int, spec.shape)), dtype=np.int64))
        for spec in plan.specs.values()
    )
    canonical_elements = sum(int(array.size) for array in params.values())
    assert packed_elements == canonical_elements == 124_439_808, (
        f"packing must be lossless: packed={packed_elements:,} vs "
        f"canonical={canonical_elements:,} elements"
    )
    packed = model.pack_params(params)
    assert set(packed) == set(plan.specs), (
        f"pack_params must key by the plan's inputs, got {sorted(packed)}"
    )

    # --- numpy interpreter reference (packing is exact: no reordering) -------
    numpy_exe = model.build(backend="numpy", pack_weights=True, cache=etl.CompileCache())
    assert len(numpy_exe.signature.input_specs) == PACKED_GPT2_DECLARED
    want = run_model(numpy_exe, packed, input_ids).numpy()
    assert want.shape == want_golden.shape == (1, 32, 50257), (
        f"logits shape: numpy {want.shape} vs golden {want_golden.shape}"
    )

    # --- the compiled packed graph ------------------------------------------
    executable = model.build(backend="iree", pack_weights=True, cache=etl.CompileCache())
    declared = len(executable.signature.input_specs)
    assert declared == PACKED_GPT2_DECLARED, (
        f"the compiled packed executable must declare {PACKED_GPT2_DECLARED} inputs "
        f"({PACKED_GPT2_INPUTS} weights + ids), got {declared} "
        f"(unpacked declares {len(params) + 1})"
    )
    assert declared < IREE_CONSUMED_INPUT_CLIFF, (
        f"{declared} declared inputs is at/over the IREE runtime cliff of "
        f"{IREE_CONSUMED_INPUT_CLIFF} consumed inputs — the packed plan must keep "
        f"the program well below it"
    )

    # --- 12 consecutive runs: no SIGSEGV, bit-identical logits ---------------
    outputs = []
    for run in range(PACKED_STABILITY_RUNS):
        got = run_model(executable, packed, input_ids).numpy()
        assert got.shape == want.shape, f"run {run}: shape {got.shape} != {want.shape}"
        assert bool(np.isfinite(got).all()), f"run {run}: compiled logits are not finite"
        if outputs:
            first = outputs[0]
            assert np.array_equal(got, first), (
                f"run {run} of the packed iree/llvm-cpu executable is not bit-identical "
                f"to run 0 (max-abs {float(np.max(np.abs(got.astype(np.float64) - first.astype(np.float64)))):g} "
                f"— same inputs must give the same program output"
            )
        outputs.append(got)
    assert len(outputs) == PACKED_STABILITY_RUNS
    got = outputs[0]

    # --- parity: compiled vs the interpreter, and vs the HuggingFace golden --
    max_abs, max_rel = _parity(got, want)
    assert max_abs <= PACKED_ATOL and max_rel <= PACKED_RTOL, (
        f"packed iree/llvm-cpu logits drifted from the numpy interpreter: "
        f"max_abs={max_abs:.3e} (budget {PACKED_ATOL:.1e}), "
        f"max_rel_vs_scale={max_rel:.3e} (budget {PACKED_RTOL:.1e}) — measured "
        f"≈{MEASURED['iree_vs_numpy_max_abs']:.1e} / "
        f"{MEASURED['iree_vs_numpy_rel']:.1e} here"
    )
    golden_abs, golden_rel = _parity(got, want_golden)
    assert golden_abs <= PACKED_ATOL and golden_rel <= PACKED_RTOL, (
        f"packed iree/llvm-cpu logits drifted from the stored HuggingFace golden "
        f"({_GOLDEN_DIR / 'gpt2.npz'}): max_abs={golden_abs:.3e} (budget "
        f"{PACKED_ATOL:.1e}), max_rel_vs_scale={golden_rel:.3e} (budget "
        f"{PACKED_RTOL:.1e}) — measured ≈"
        f"{MEASURED['iree_vs_golden_max_abs']:.1e} / "
        f"{MEASURED['iree_vs_golden_rel']:.1e} here"
    )


# ---------------------------------------------------------------------------
# 8. no silent device / dtype / shape surprises
# ---------------------------------------------------------------------------
def test_params_are_float32_transfer_explicitly_and_never_guess():
    """``params`` are float32 host arrays; placement is explicit; a bad config is loud."""
    model = _load("tiny_llama", seq_len=4)
    params = model.params()
    assert all(isinstance(value, np.ndarray) for value in params.values())
    assert {value.dtype for value in params.values()} == {np.dtype(np.float32)}
    assert all(value.flags.c_contiguous for value in params.values())

    device = etl.core.Device("cpu", 0)
    placed = place_params(params, device)
    assert set(placed) == set(params)
    for name, tensor in placed.items():
        assert isinstance(tensor, etl.Tensor), f"{name} must be an etl.Tensor, got {type(tensor)}"
        assert tensor.device == device
        assert tensor.shape == params[name].shape
        np.testing.assert_array_equal(np.asarray(tensor.numpy()), params[name])
    # ... and the placed tree runs the same graph to the same logits.
    ids = _ids(1, 4)
    assert np.array_equal(
        model.forward(params=placed, input_ids=ids, backend="numpy"),
        model.forward(input_ids=ids, backend="numpy"),
    )


def test_a_mutated_config_fails_loudly(tmp_path):
    """A checkpoint whose ``config.json`` disagrees with its weights raises, never guesses."""
    source = hf_download.require_model("tiny_llama")
    pristine = tmp_path / "pristine"
    pristine.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copy2(source / name, pristine / name)
    # The copy itself must load — otherwise the failure below might be the copy's.
    assert _total_params(load_hf(pristine, seq_len=4).params()) == _total_params(
        load_hf(source, seq_len=4).params()
    )

    mutated = tmp_path / "mutated_hidden_size"
    shutil.copytree(pristine, mutated)
    raw = json.loads((mutated / "config.json").read_text(encoding="utf-8"))
    raw["hidden_size"] = 32
    (mutated / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(HFWeightError) as excinfo:
        load_hf(mutated, seq_len=4)
    message = str(excinfo.value)
    assert "embed.weight" in message, message
    assert "(32000, 16)" in message and "(32000, 32)" in message, message
    assert "hidden_size=32" in message, message

    # A structurally invalid variant is caught by the config layer instead.
    broken = tmp_path / "mutated_not_divisible"
    shutil.copytree(pristine, broken)
    raw = json.loads((broken / "config.json").read_text(encoding="utf-8"))
    raw["hidden_size"] = 15
    (broken / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(HFConfigError) as excinfo:
        load_hf(broken, seq_len=4)
    assert "hidden_size (15)" in str(excinfo.value)
