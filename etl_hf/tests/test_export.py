"""Whole-graph StableHLO export conformance for the ``etl_hf`` families.

The package contract is "**the whole model forward pass is ONE etl graph**"
(``./etl_hf/CONTEXT.md``). This suite proves that property at the **IR level**,
with numpy only and no compiler installed: each family's ``graph(cfg, seq_len,
batch_size)`` is traced against its own ``param_specs(cfg)`` (+ an int32 ids
spec), verified, and exported to StableHLO text via
``etl.backends.stablehlo.export``.

What is pinned here:

* the export *succeeds* for both families and the text contains the StableHLO
  constructs of the whole forward (attention dots, reductions, softmax exp,
  in-graph RoPE cos/sin, RMSNorm sqrt/divide, SiLU logistic, causal compare,
  embedding gather) — not just a fragment;
* the graph stays *small*: weights are graph **inputs**, never inlined
  ``stablehlo.constant`` payloads, so the export is a few tens of KB and the IR
  op count is independent of ``seq_len``;
* scaling is roughly linear in the layer count;
* exact-erf GELU stays an **explicit** compiler-backend deferral
  (``etl.BackendError`` naming ``erf``) while the very same graph still builds
  and runs on the numpy interpreter — a backend limitation, not a broken graph;
* export is deterministic (byte-identical on repeat) and config-sensitive;
* ``batch_size=2`` exports with a *static* batch extent (no ``?`` dynamic dims).

Every measurement taken while the tests run lands in :data:`MEASURED` (keyed by
a label), so a failure message can carry the actual numbers and a human (or the
next agent) can read the recorded sizes without re-running anything.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import etl
from etl.backends import stablehlo
from etl_hf import ops
from etl_hf.config import ModelConfig
from etl_hf.models import gpt2 as gpt2_mod
from etl_hf.models import llama as llama_mod

# ---------------------------------------------------------------------------
# measured numbers (filled in by ``_trace``; readable when reporting)
# ---------------------------------------------------------------------------

#: ``label -> {"ops_total", "distinct_ops", "lines", "bytes", "constants",
#: "tensor_specs", "histogram"}`` for every graph traced in this module.
MEASURED: dict[str, dict[str, object]] = {}


# ---------------------------------------------------------------------------
# tiny static configs (one config = one graph, per etl's value model)
# ---------------------------------------------------------------------------

#: 1-layer grouped-query Llama: 16 hidden / 4 heads / 2 kv heads, SwiGLU.
LLAMA_TINY: dict[str, object] = {
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 32,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

#: 4-layer Llama, wider than :data:`LLAMA_TINY` (used for the scaling pin).
LLAMA_FOUR_LAYER: dict[str, object] = {
    **LLAMA_TINY,
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 64,
}

#: 2-layer GPT-2 (gelu_new, learned positions, tied lm_head).
GPT2_TINY: dict[str, object] = {
    "model_type": "gpt2",
    "n_embd": 16,
    "n_layer": 2,
    "n_head": 4,
    "n_inner": 32,
    "n_vocab": 32,
    "n_positions": 16,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": True,
}

#: Sequence length used by the fixed-shape exports.
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _count_ops(graph: object) -> dict[str, int]:
    """Op-name histogram of a traced :class:`etl.Graph`, walking nested regions.

    Mirrors the prototype's ``count_ops`` in ``bench_logs/hf_probe_lib.py``:
    every op of the entry block counts once, and each nested region (``cond`` /
    ``while_loop`` / ``scan`` bodies) is walked recursively. These model graphs
    are straight-line, so the walk finds no nested regions — it is written
    generically so a future control-flow model cannot silently under-count.

    Args:
        graph: the traced graph (``graph.module.functions[0].entry_block``).

    Returns:
        dict: op def name -> occurrence count (empty for an empty graph).
    """
    counts: dict[str, int] = {}

    def walk_block(block: object) -> None:
        """Accumulate ``block``'s ops, recursing into their regions."""
        for op in block.ops:  # type: ignore[attr-defined]
            name = op.opdef.name
            counts[name] = counts.get(name, 0) + 1
            for region in op.regions:
                for nested in region.blocks:
                    walk_block(nested)

    walk_block(graph.module.functions[0].entry_block)  # type: ignore[attr-defined]
    return counts


def _record(label: str, graph: object, text: str) -> None:
    """Store one graph's measured sizes in :data:`MEASURED` under ``label``."""
    histogram = _count_ops(graph)
    MEASURED[label] = {
        "ops_total": sum(histogram.values()),
        "distinct_ops": len(histogram),
        "lines": len(text.splitlines()),
        "bytes": len(text),
        "constants": text.count("stablehlo.constant"),
        "tensor_specs": len(graph.tensor_specs),  # type: ignore[attr-defined]
        "histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
    }


def _summary(label: str) -> str:
    """Render :data:`MEASURED` entry ``label`` as one self-documenting line.

    Falls back to a plain marker when ``label`` was never measured (e.g. an
    export raised before the numbers were recorded) so that building a failure
    message can never mask the failure it is describing.
    """
    m = MEASURED.get(label)
    if m is None:
        return f"[{label}] <not measured>"
    return (
        f"[{label}] ops={m['ops_total']} distinct={m['distinct_ops']} "
        f"lines={m['lines']} bytes={m['bytes']} "
        f"constants={m['constants']} inputs={m['tensor_specs']}"
    )


def _trace(
    family_mod: object,
    cfg: ModelConfig,
    seq_len: int,
    batch: int = 1,
    *,
    label: str | None = None,
) -> tuple[object, str]:
    """Trace + verify + export one family graph.

    Args:
        family_mod: ``etl_hf.models.llama`` or ``etl_hf.models.gpt2`` (both
            expose ``param_specs`` / ``graph``).
        cfg: the model config (one config = one graph).
        seq_len: static sequence length.
        batch: static batch size.
        label: key under which the measured sizes are stored in
            :data:`MEASURED`; defaults to
            ``f"{family}<S{seq_len}>_B{batch}"`` (``family`` = the module's
            ``FAMILY`` constant).
    Returns:
        tuple: ``(graph, export_text)`` — the verified graph and its
        StableHLO export.

    Raises:
        etl.BackendError: the graph uses an op the StableHLO exporter defers
            (e.g. exact-erf GELU); propagated untouched.
    """
    specs = family_mod.param_specs(cfg)  # type: ignore[attr-defined]
    ids_spec = etl.TensorSpec((batch, seq_len), etl.int32)
    graph = etl.trace(
        family_mod.graph(cfg, seq_len, batch),  # type: ignore[attr-defined]
        specs,
        ids_spec,
    )
    graph.verify()
    text = stablehlo.export(graph)
    family = getattr(family_mod, "FAMILY", "family")
    _record(label or f"{family}_S{seq_len}_B{batch}", graph, text)
    return graph, text


def _synthetic_params(specs: dict[str, etl.TensorSpec], seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic synthetic weights matching ``specs`` exactly.

    Norm scales (1-D specs) are ``1.0`` so the activations stay in a sane range;
    matrices get small seeded gaussians. Nothing here is a real checkpoint —
    only the numpy-interpreter smoke path needs *some* well-shaped host data.
    """
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for key, spec in specs.items():
        shape = tuple(int(dim) for dim in spec.shape)
        if len(shape) == 1:
            params[key] = np.ones(shape, dtype=np.float32)
        else:
            params[key] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return params


def _require(text: str, required: dict[str, tuple[str, ...]], label: str) -> None:
    """Assert every entry of ``required`` is satisfied by ``text``.

    Args:
        text: the StableHLO export.
        required: construct name -> tuple of markers, ANY of which satisfies the
            entry (e.g. ``{"rmsnorm": ("stablehlo.rsqrt", "stablehlo.sqrt")}``).
        label: the :data:`MEASURED` label, used in the failure message.
    """
    missing = {name: markers for name, markers in required.items() if not any(m in text for m in markers)}
    assert not missing, (
        f"StableHLO export of {label} is missing whole-model constructs: {sorted(missing)}; "
        f"{_summary(label)}; present stablehlo ops: "
        f"{sorted(set(re.findall(r'stablehlo[.][a-z_0-9]+', text)))}"
    )


# ---------------------------------------------------------------------------
# 1. Llama exports the whole forward
# ---------------------------------------------------------------------------
def test_llama_tiny_exports_whole_forward() -> None:
    """A 1-layer Llama forward exports as one func with every model construct."""
    cfg = ModelConfig.from_dict(LLAMA_TINY)
    label = "llama_tiny_S8_B1"
    graph, text = _trace(llama_mod, cfg, SEQ_LEN, 1, label=label)

    graph.verify()
    assert text.strip().startswith("module {"), (
        f"export must be an MLIR module, got: {text[:80]!r}; {_summary(label)}"
    )
    assert "func.func @main" in text, (
        f"the whole forward must be a single exported @main function; {_summary(label)}"
    )

    _require(
        text,
        {
            # linear layers / attention score+context matmuls
            "dot": ("stablehlo.dot_general", "stablehlo.dot"),
            # softmax max/sum and RMSNorm mean reductions
            "reduce": ("stablehlo.reduce",),
            # softmax exp
            "softmax_exp": ("stablehlo.exponential",),
            # in-graph RoPE tables
            "rope_cos": ("stablehlo.cosine",),
            "rope_sin": ("stablehlo.sine",),
            # RMSNorm mean(x^2) normalisation. NOTE: the exporter emits
            # sqrt + divide; `stablehlo.rsqrt` is NEVER emitted here (see the
            # module docstring / report) — accepted either way.
            "rmsnorm": ("stablehlo.rsqrt", "stablehlo.sqrt"),
            "divide": ("stablehlo.divide",),
            # silu (SwiGLU gate)
            "silu": ("stablehlo.logistic",),
            # causal mask
            "causal_mask": ("stablehlo.compare",),
            # token embedding lookup
            "embed_gather": ("stablehlo.gather",),
        },
        label,
    )


# ---------------------------------------------------------------------------
# 2. recorded sizes as loose lower bounds
# ---------------------------------------------------------------------------
def test_recorded_sizes_are_loose_lower_bounds() -> None:
    """Op count / export length are large but not brittle; no payload inlining."""
    cfg = ModelConfig.from_dict(LLAMA_TINY)
    label = "llama_tiny_S8_B1"
    graph, _text = _trace(llama_mod, cfg, SEQ_LEN, 1, label=label)

    assert label in MEASURED, f"the measurements must be recorded; keys={sorted(MEASURED)}"
    ops_total = int(MEASURED[label]["ops_total"])
    lines = int(MEASURED[label]["lines"])
    size = int(MEASURED[label]["bytes"])

    # Lower bounds only: a whole 1-layer transformer forward cannot be a handful
    # of ops, and these bounds leave generous headroom for exporter changes.
    assert ops_total >= 60, f"whole-graph IR too small — {_summary(label)}"
    assert lines >= 100, f"whole-graph StableHLO too short — {_summary(label)}"
    # Upper bound: a tiny config cannot possibly carry weight payloads.
    assert size < 200 * 1024, (
        f"tiny export must not inline payloads (< 200 KB) — {_summary(label)}"
    )
    assert len(_count_ops(graph)) >= 10, (
        f"expected a rich op mix (not a single fused op) — {_summary(label)}"
    )


# ---------------------------------------------------------------------------
# 3. no weights are baked into the IR
# ---------------------------------------------------------------------------
def test_no_weights_baked_into_ir() -> None:
    """Weights are graph inputs: spec count and ``stablehlo.constant`` count prove it."""
    cfg = ModelConfig.from_dict(LLAMA_TINY)
    label = "llama_tiny_S8_B1"
    graph, text = _trace(llama_mod, cfg, SEQ_LEN, 1, label=label)

    specs = llama_mod.param_specs(cfg)
    assert len(graph.tensor_specs) == len(specs) + 1, (
        f"graph must take exactly the {len(specs)} weight specs + 1 ids spec, "
        f"got {len(graph.tensor_specs)} — {_summary(label)}"
    )

    constants = text.count("stablehlo.constant")
    assert constants < 40, (
        f"too many stablehlo.constant ops — weights are being inlined instead of "
        f"passed as inputs — {_summary(label)}"
    )
    # Structural constants only: the causal-mask/arange bases are O(seq_len) long,
    # while any real weight payload (embedding, projection) would be O(hidden x
    # vocab) elements. A literal-count ceiling separates the two decisively.
    literals = re.findall(r"dense<\[([^\]]*)\]>", text)
    biggest_literal = max((len(lit.split(",")) for lit in literals), default=0)
    assert biggest_literal <= 4 * SEQ_LEN, (
        f"largest dense literal has {biggest_literal} elements — only O(seq_len) "
        f"structural constants are allowed (no weight payloads) — {_summary(label)}"
    )


# ---------------------------------------------------------------------------
# 4. GPT-2 exports the whole forward
# ---------------------------------------------------------------------------
def test_gpt2_tiny_exports_whole_forward() -> None:
    """A 2-layer GPT-2 forward (gelu_new, learned positions) exports as one func."""
    cfg = ModelConfig.from_dict(GPT2_TINY)
    label = "gpt2_tiny_S8_B1"
    graph, text = _trace(gpt2_mod, cfg, SEQ_LEN, 1, label=label)

    graph.verify()
    assert "func.func @main" in text, (
        f"the whole forward must be a single exported @main function; {_summary(label)}"
    )
    _require(
        text,
        {
            "dot": ("stablehlo.dot_general", "stablehlo.dot"),
            # gelu_new = the tanh approximation (never erf on this path)
            "gelu_new_tanh": ("stablehlo.tanh",),
            # token embedding + learned position embedding lookups
            "gather": ("stablehlo.gather",),
            # causal mask compare / LayerNorm mean reductions / softmax exp
            "compare": ("stablehlo.compare",),
            "reduce": ("stablehlo.reduce",),
            "softmax_exp": ("stablehlo.exponential",),
        },
        label,
    )
    assert text.count("stablehlo.gather") >= 2, (
        f"GPT-2 has BOTH a token and a position embedding gather; "
        f"found {text.count('stablehlo.gather')} — {_summary(label)}"
    )
    # Absolute positions: GPT-2 must NOT build in-graph RoPE tables.
    assert "stablehlo.cosine" not in text and "stablehlo.sine" not in text, (
        f"GPT-2 uses learned positions — no in-graph RoPE expected — {_summary(label)}"
    )
    # Same "weights are inputs" contract as llama.
    assert len(graph.tensor_specs) == len(gpt2_mod.param_specs(cfg)) + 1, (
        f"weights must all be graph inputs — {_summary(label)}"
    )


# ---------------------------------------------------------------------------
# 5. scaling is roughly linear in the layer count
# ---------------------------------------------------------------------------
def test_llama_op_count_scales_with_layers() -> None:
    """4 layers produce at least twice the 1-layer op count, and still export."""
    one_label, four_label = "llama_tiny_S8_B1", "llama_4layer_S8_B1"
    cfg1 = ModelConfig.from_dict(LLAMA_TINY)
    _trace(llama_mod, cfg1, SEQ_LEN, 1, label=one_label)
    cfg4 = ModelConfig.from_dict(LLAMA_FOUR_LAYER)
    _trace(llama_mod, cfg4, SEQ_LEN, 1, label=four_label)

    ops1 = int(MEASURED[one_label]["ops_total"])
    ops4 = int(MEASURED[four_label]["ops_total"])
    assert ops4 > 2 * ops1, (
        f"4 layers must emit roughly 4x the per-block ops (> 2x the 1-layer count): "
        f"ops_4L={ops4} ops_1L={ops1} — {_summary(one_label)} / {_summary(four_label)}"
    )
    assert int(MEASURED[four_label]["lines"]) > int(MEASURED[one_label]["lines"]), (
        f"deeper model must export more StableHLO — "
        f"{_summary(one_label)} / {_summary(four_label)}"
    )


# ---------------------------------------------------------------------------
# 6. op count is seq-len independent
# ---------------------------------------------------------------------------
def test_llama_op_count_is_seq_len_independent() -> None:
    """The IR op count is unchanged by ``seq_len`` (only static extents change)."""
    cfg = ModelConfig.from_dict(LLAMA_TINY)
    short_label, long_label = "llama_tiny_S8_B1", "llama_tiny_S32_B1"
    _, text8 = _trace(llama_mod, cfg, 8, 1, label=short_label)
    _, text32 = _trace(llama_mod, cfg, 32, 1, label=long_label)

    assert MEASURED[short_label]["ops_total"] == MEASURED[long_label]["ops_total"], (
        f"op count must be seq-len independent: "
        f"S8 ops={MEASURED[short_label]['ops_total']} "
        f"S32 ops={MEASURED[long_label]['ops_total']} "
        f"(histograms equal={MEASURED[short_label]['histogram'] == MEASURED[long_label]['histogram']})"
    )
    assert MEASURED[short_label]["histogram"] == MEASURED[long_label]["histogram"], (
        f"the whole op histogram (not just its total) must be seq-len independent — "
        f"S8={MEASURED[short_label]['histogram']} S32={MEASURED[long_label]['histogram']}"
    )
    # Shapes DO change — proving the equality above is about structure, not a
    # no-op trace.
    assert text8 != text32, f"different seq_len must change the static shapes — {_summary(long_label)}"
    assert "tensor<1x32xi32>" in text32, (
        f"S32 ids argument must be statically (1, 32) — {_summary(long_label)}"
    )
    assert "-> tensor<1x32x32xf32>" in text32, (
        f"S32 logits must be statically (1, 32, 32) — {_summary(long_label)}"
    )


# ---------------------------------------------------------------------------
# 7. exact-erf GELU is an explicit export deferral (not a broken graph)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_mod, raw_cfg",
    [
        pytest.param(llama_mod, {**LLAMA_TINY, "hidden_act": "gelu"}, id="llama"),
        pytest.param(
            gpt2_mod,
            {**GPT2_TINY, "n_layer": 1, "activation_function": "gelu"},
            id="gpt2",
        ),
    ],
)
def test_exact_erf_gelu_is_an_explicit_export_deferral(
    family_mod: object, raw_cfg: dict[str, object]
) -> None:
    """``erf`` GELU defers on compiler backends yet runs on the numpy interpreter."""
    assert ops.activation_is_export_safe("gelu") is False, (
        f"exact-erf GELU must be flagged unsafe: EXPORT_UNSAFE_ACTIVATIONS="
        f"{sorted(ops.EXPORT_UNSAFE_ACTIVATIONS)}"
    )
    assert ops.activation_is_export_safe("silu") is True
    assert ops.activation_is_export_safe("gelu_new") is True

    cfg = ModelConfig.from_dict(raw_cfg)
    with pytest.raises(etl.BackendError) as excinfo:
        _trace(family_mod, cfg, SEQ_LEN, 1)
    message = str(excinfo.value)
    assert "erf" in message or "gelu" in message, (
        f"the deferral must NAME the offending op (erf/gelu), got: {message!r}"
    )
    # Explicit deferral, never a silent tanh substitute.
    assert "not supported in v1" in message, (
        f"the deferral must say the op is unsupported in v1, got: {message!r}"
    )

    # ... and the SAME graph builds + runs on the numpy interpreter, matching the
    # family's independent numpy reference: the limit is the backend, not the graph.
    specs = family_mod.param_specs(cfg)  # type: ignore[attr-defined]
    params = _synthetic_params(specs)
    ids = np.zeros((1, SEQ_LEN), dtype=np.int32)
    executable = etl.build(
        family_mod.graph(cfg, SEQ_LEN, 1),  # type: ignore[attr-defined]
        specs,
        etl.TensorSpec((1, SEQ_LEN), etl.int32),
    )
    result = etl.run(executable, params, ids)
    logits = result.numpy()
    vocab = int(cfg.vocab_size)
    assert logits.shape == (1, SEQ_LEN, vocab), (
        f"numpy-interpreter logits must be (1, {SEQ_LEN}, {vocab}), got {logits.shape}"
    )
    assert np.isfinite(logits).all(), "numpy-interpreter logits contain non-finite values"
    expected = family_mod.reference(cfg, params, ids)  # type: ignore[attr-defined]
    max_abs = float(np.abs(logits - expected).max())
    assert max_abs <= 1e-4, (
        f"erf-GELU graph must still be numerically correct on numpy: "
        f"max_abs={max_abs:.3e} vs the family reference"
    )


# ---------------------------------------------------------------------------
# 8. determinism / config sensitivity
# ---------------------------------------------------------------------------
def test_export_is_deterministic_and_config_sensitive() -> None:
    """Exporting the same graph twice is byte-identical; another config is not."""
    cfg = ModelConfig.from_dict(LLAMA_TINY)
    label = "llama_tiny_S8_B1"
    graph, text_a = _trace(llama_mod, cfg, SEQ_LEN, 1, label=label)
    text_b = stablehlo.export(graph)
    assert text_a == text_b, (
        f"re-exporting the same graph must be byte-identical "
        f"(len {len(text_a)} vs {len(text_b)}) — {_summary(label)}"
    )

    # Re-tracing the same static config yields the same program text too.
    _, text_c = _trace(llama_mod, cfg, SEQ_LEN, 1, label="llama_tiny_S8_B1_retrace")
    assert text_c == text_a, (
        f"tracing the same config twice must be deterministic "
        f"(len {len(text_a)} vs {len(text_c)}) — {_summary('llama_tiny_S8_B1_retrace')}"
    )

    # A different config is a different graph: the text must change.
    _, text_deep = _trace(
        llama_mod, ModelConfig.from_dict(LLAMA_FOUR_LAYER), SEQ_LEN, 1,
        label="llama_4layer_S8_B1",
    )
    assert text_deep != text_a, (
        f"a 4-layer/wider config must export a different program — "
        f"{_summary(label)} / {_summary('llama_4layer_S8_B1')}"
    )


# ---------------------------------------------------------------------------
# 9. static batch extent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "family_mod, raw_cfg, vocab",
    [
        pytest.param(llama_mod, LLAMA_TINY, 32, id="llama"),
        pytest.param(gpt2_mod, GPT2_TINY, 32, id="gpt2"),
    ],
)
def test_batch_size_two_exports_static_batch(
    family_mod: object, raw_cfg: dict[str, object], vocab: int
) -> None:
    """``batch_size=2`` bakes a concrete batch extent (no ``?`` dynamic dims)."""
    cfg = ModelConfig.from_dict(raw_cfg)
    label = f"{family_mod.FAMILY}_tiny_S{SEQ_LEN}_B2"  # type: ignore[attr-defined]
    graph, text = _trace(family_mod, cfg, SEQ_LEN, 2, label=label)

    graph.verify()
    assert f"tensor<2x{SEQ_LEN}xi32>" in text, (
        f"the ids input must be statically (2, {SEQ_LEN}) — {_summary(label)}"
    )
    assert f"-> tensor<2x{SEQ_LEN}x{vocab}xf32>" in text, (
        f"logits must be statically (2, {SEQ_LEN}, {vocab}) — {_summary(label)}"
    )
    assert "?" not in text, (
        f"static-shape-first: no dynamic (?) dims may appear in the export — "
        f"{text.count('?')} found — {_summary(label)}"
    )
    # The batch extent is a static shape value: it changes extents, never the IR
    # structure, so the op count matches the batch-1 graph of the same config.
    one_label = f"{family_mod.FAMILY}_tiny_S{SEQ_LEN}_B1"  # type: ignore[attr-defined]
    if one_label not in MEASURED:
        _trace(family_mod, cfg, SEQ_LEN, 1, label=one_label)
    assert MEASURED[label]["ops_total"] == MEASURED[one_label]["ops_total"], (
        f"batch extent must not change the IR op count — "
        f"{_summary(label)} vs {_summary(one_label)}"
    )
