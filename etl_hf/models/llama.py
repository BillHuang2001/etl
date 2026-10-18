"""The Llama / Mistral (and Qwen2) family: HF checkpoint → ONE ``etl`` graph.

A HuggingFace ``LlamaForCausalLM`` / ``MistralForCausalLM`` checkpoint is a
pre-norm transformer with **RMSNorm**, **RoPE**, **grouped-query attention** and a
**SwiGLU** MLP. This module turns such a checkpoint into a single static-shape
``@etl.defn`` forward pass — ``params, input_ids -> logits`` — plus a canonical
weight map, a :class:`etl.TensorSpec` tree for those weights, and an
*independent* pure-numpy reference forward used as the parity oracle.

Everything here is assembly of already-public pieces:

``etl_hf.ops``         the graph idioms (``rms_norm``, ``apply_rope``, …)
``etl_hf.models._common`` the block wiring (``linear``, ``attention``, ``gated_mlp``)
``etl_hf.reference``   the numpy oracle primitives (no ``etl`` import)

Static shapes only
------------------
``seq_len`` and ``batch_size`` are baked into the graph (they are static graph
specialisation values in etl's value model), so a different length is a
different graph — memoise at the build layer. ``graph()`` rejects a ``seq_len``
beyond ``cfg.max_position_embeddings`` instead of silently wrapping positions.

Two conventions this module fixes (and every docstring restates, because both are
places where a silent model bug would hide):

* **Linear weights are ``(in_features, out_features)``** — HF stores Llama
  linears as ``(out_features, in_features)``, so :func:`canonicalize_weights`
  transposes them; the graph then feeds them straight to
  ``etl_hf.models._common.linear`` (``etl.dot(x, w)``).
* **The causal mask puts ``0`` on and below the diagonal** (HF convention: a
  query attends to itself and the past), never the anti-causal inverse.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

import etl
import etl_hf.reference as _numpy_ref
from etl_hf import ops
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.models import _common

if TYPE_CHECKING:  # pragma: no cover - typing only (no runtime import)
    from etl_hf.pack import PackPlan

__all__ = [
    "FAMILY",
    "MODEL_TYPES",
    "canonicalize_weights",
    "graph",
    "param_specs",
    "reference",
]

#: ``etl_hf`` family key implemented by this module.
FAMILY = "llama"

#: HF ``model_type`` values this architecture covers (all Llama-shaped).
MODEL_TYPES = ("llama", "mistral", "qwen2")

#: Checkpoint keys that are deliberately ignored rather than reported as
#: unexpected: older HF checkpoints persist the RoPE inverse frequencies as a
#: buffer even though the forward pass recomputes them.
_IGNORED_KEYS: frozenset[str] = frozenset({"model.rotary_emb.inv_freq"})

#: Per-layer canonical suffix → ``(HF sub-key, transpose?)``. Linears are
#: transposed (HF ``(out, in)`` → canonical ``(in, out)``); norm scales are 1-D
#: and taken verbatim.
_LAYER_RECORDS: tuple[tuple[str, str, bool], ...] = (
    ("attn_norm.weight", "input_layernorm.weight", False),
    ("attn.q.weight", "self_attn.q_proj.weight", True),
    ("attn.k.weight", "self_attn.k_proj.weight", True),
    ("attn.v.weight", "self_attn.v_proj.weight", True),
    ("attn.o.weight", "self_attn.o_proj.weight", True),
    ("mlp_norm.weight", "post_attention_layernorm.weight", False),
    ("mlp.gate.weight", "mlp.gate_proj.weight", True),
    ("mlp.up.weight", "mlp.up_proj.weight", True),
    ("mlp.down.weight", "mlp.down_proj.weight", True),
)


# ---------------------------------------------------------------------------
# small internal helpers
# ---------------------------------------------------------------------------
def _require_llama(cfg: ModelConfig, what: str) -> None:
    """Validate that ``cfg`` belongs to this family.

    Args:
        cfg: the config under test.
        what: calling function name, used in the error message.

    Raises:
        HFConfigError: ``cfg`` is not a :class:`ModelConfig`, or its ``family``
            field is not ``"llama"``.
    """
    if not isinstance(cfg, ModelConfig):
        raise HFConfigError(
            f"{what}: expected an etl_hf.config.ModelConfig, got "
            f"{type(cfg).__name__}; load one with ModelConfig.from_dir(model_dir)."
        )
    if cfg.family != FAMILY:
        raise HFConfigError(
            f"{what}: this module builds the {FAMILY!r} family (HF model_type "
            f"{list(MODEL_TYPES)}), but the config's model_type is "
            f"{cfg.model_type!r} which maps to family {cfg.family!r}; use "
            f"etl_hf.models.{cfg.family} for that checkpoint."
        )


def _positive_int(value: object, what: str) -> int:
    """Return ``value`` as an ``int``, requiring a positive non-bool integer.

    Args:
        value: caller-supplied value.
        what: parameter name for the error message.

    Returns:
        int: ``value`` normalised to ``int``.

    Raises:
        HFConfigError: ``value`` is not a positive integer (a static graph
            parameter, so a config-level error rather than a Python ``TypeError``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise HFConfigError(
            f"{what} must be a positive integer, got {type(value).__name__} "
            f"({value!r})."
        )
    if int(value) <= 0:
        raise HFConfigError(f"{what} must be a positive integer, got {value!r}.")
    return int(value)


def _expected_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Canonical parameter key → expected shape, for this config.

    The single source of truth shared by :func:`param_specs` (which turns it into
    ``etl.TensorSpec`` objects) and ``canonicalize_weights`` / :func:`reference`
    (which validate against it). Linears are ``(in_features, out_features)``;
    norm scales are ``(hidden_size,)``.

    Note:
        ``attn.q/k/v/o`` shapes derive from ``cfg.head_dim`` (which may differ
        from ``hidden_size // num_attention_heads`` when HF sets it explicitly),
        never from an assumed ``hidden_size // num_attention_heads``.

    Note:
        A ``tie_word_embeddings=True`` config has **no** ``lm_head.weight`` key:
        the output head IS ``embed.weight`` (one graph input, gathered for tokens
        and transposed for the projection), so the tied tree is exactly one key
        smaller than the untied one.

    Args:
        cfg: a ``family == "llama"`` config.

    Returns:
        dict: canonical key → shape tuple, in canonical (embed → layers →
        final_norm → untied lm_head) order.
    """
    hidden = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size

    shapes: dict[str, tuple[int, ...]] = {
        "embed.weight": (vocab, hidden),
    }
    for index in range(cfg.num_hidden_layers):
        prefix = f"layers.{index}."
        shapes[f"{prefix}attn_norm.weight"] = (hidden,)
        shapes[f"{prefix}attn.q.weight"] = (hidden, q_dim)
        shapes[f"{prefix}attn.k.weight"] = (hidden, kv_dim)
        shapes[f"{prefix}attn.v.weight"] = (hidden, kv_dim)
        shapes[f"{prefix}attn.o.weight"] = (q_dim, hidden)
        shapes[f"{prefix}mlp_norm.weight"] = (hidden,)
        shapes[f"{prefix}mlp.gate.weight"] = (hidden, inter)
        shapes[f"{prefix}mlp.up.weight"] = (hidden, inter)
        shapes[f"{prefix}mlp.down.weight"] = (inter, hidden)
    shapes["final_norm.weight"] = (hidden,)
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (hidden, vocab)
    return shapes


def _as_f32(key: str, value: object) -> np.ndarray:
    """Convert one checkpoint tensor to a contiguous writable float32 array.

    Args:
        key: the checkpoint key, used in the error message.
        value: the raw object from the state dict.

    Returns:
        np.ndarray: ``float32``, C-contiguous, writable, at least 1-D.

    Raises:
        HFWeightError: ``value`` is not array-like or not real-numeric.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise HFWeightError(
            f"checkpoint tensor {key!r} is not array-like "
            f"(got {type(value).__name__}); etl_hf needs numeric tensors"
        ) from exc
    if array.dtype.kind not in "iuf":
        raise HFWeightError(
            f"checkpoint tensor {key!r} has non-real dtype {array.dtype}; etl_hf "
            "canonicalises every weight to float32"
        )
    if array.ndim == 0:
        raise HFWeightError(
            f"checkpoint tensor {key!r} is a 0-d scalar; expected a weight array"
        )
    return np.ascontiguousarray(array.astype(np.float32))


def _missing_key_error(hf_key: str, canonical: str, available: list[str]) -> HFWeightError:
    """Build the "required key absent" error, with near-miss suggestions.

    Args:
        hf_key: the absent checkpoint key.
        canonical: the canonical key it would have produced.
        available: every key present in the checkpoint.

    Returns:
        HFWeightError: the error to raise (naming key, canonical target, and a
        few near-miss keys when there are any).
    """
    near = difflib.get_close_matches(hf_key, available, n=3, cutoff=0.6)
    hint = f"; nearest available keys: {near}" if near else ""
    return HFWeightError(
        f"missing required checkpoint key {hf_key!r} (needed for canonical "
        f"weight {canonical!r}){hint}. etl_hf models.llama expects a Llama/Mistral "
        f"state dict whose keys start with 'model.' (embed_tokens / layers.N / "
        f"norm) plus an optional 'lm_head.weight' — check the checkpoint layout "
        f"against the config's num_hidden_layers."
    )


# ---------------------------------------------------------------------------
# canonical weight map
# ---------------------------------------------------------------------------
def canonicalize_weights(
    cfg: ModelConfig, raw: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Map an HF Llama/Mistral state dict onto the canonical flat key scheme.

    Canonical keys (``params`` is a FLAT ``dict[str, np.float32 ndarray]``; every
    linear weight is **``(in_features, out_features)``**, i.e. HF's ``(out, in)``
    transposed at load, and every norm scale is ``(hidden_size,)``)::

        embed.weight                       (vocab_size, hidden_size)      <- model.embed_tokens.weight (verbatim)
        layers.{i}.attn_norm.weight        (hidden_size,)                 <- model.layers.{i}.input_layernorm.weight
        layers.{i}.attn.q.weight           (hidden_size, n_heads*head_dim)     <- ...self_attn.q_proj.weight  .T
        layers.{i}.attn.k.weight           (hidden_size, n_kv_heads*head_dim)  <- ...k_proj.weight            .T
        layers.{i}.attn.v.weight           (hidden_size, n_kv_heads*head_dim)  <- ...v_proj.weight            .T
        layers.{i}.attn.o.weight           (n_heads*head_dim, hidden_size)     <- ...o_proj.weight            .T
        layers.{i}.mlp_norm.weight         (hidden_size,)                 <- ...post_attention_layernorm.weight
        layers.{i}.mlp.gate.weight         (hidden_size, intermediate)    <- ...mlp.gate_proj.weight     .T
        layers.{i}.mlp.up.weight           (hidden_size, intermediate)    <- ...mlp.up_proj.weight       .T
        layers.{i}.mlp.down.weight         (intermediate, hidden_size)    <- ...mlp.down_proj.weight     .T
        final_norm.weight                  (hidden_size,)                 <- model.norm.weight
        lm_head.weight                     (hidden_size, vocab_size)      <- lm_head.weight .T (UNTIED only)

    Weight tying: with ``tie_word_embeddings`` the output head IS
    ``embed.weight``, so a tied tree has **no** ``lm_head.weight`` entry (one key
    fewer, and the head is never a second transposed copy); a checkpoint that
    also ships ``lm_head.weight`` must agree with the embedding, is verified,
    consumed and dropped.

    Nothing is guessed: every problem is an explicit error (see Raises).

    Args:
        cfg: the ``family == "llama"`` config the checkpoint must match.
        raw: the HF state dict (``key -> array``), e.g. from
            :func:`etl_hf.weights.load_checkpoint`.

    Returns:
        dict: canonical key → ``float32`` contiguous writable array, in the order
        of :func:`param_specs` (a tied config therefore has no
        ``lm_head.weight`` entry).

    Raises:
        HFConfigError: ``cfg`` is not a llama-family :class:`ModelConfig`.
        HFWeightError: a required key is missing, an unexpected/unconsumed key is
            present, a shape does not match ``cfg``, or a tied head disagrees with
            the embedding.
        HFUnsupportedError: the checkpoint carries any ``*.bias`` tensor (v1
            llama builds are bias-free).
    """
    _require_llama(cfg, "canonicalize_weights")
    if not isinstance(raw, Mapping):
        raise HFWeightError(
            f"canonicalize_weights: expected a mapping of checkpoint keys to "
            f"arrays, got {type(raw).__name__}"
        )
    keys = list(raw.keys())
    non_str = [key for key in keys if not isinstance(key, str)]
    if non_str:
        raise HFWeightError(
            f"canonicalize_weights: checkpoint keys must be strings; got "
            f"{non_str[:3]!r} of type {type(non_str[0]).__name__}"
        )

    biased = sorted(key for key in keys if key.endswith(".bias"))
    if biased:
        raise HFUnsupportedError(
            f"unsupported checkpoint: v1 etl_hf llama builds are bias-free "
            f"(Llama/Mistral set attention_bias=False / mlp_bias=False), but this "
            f"checkpoint carries bias tensors {biased}; use a config/checkpoint with "
            f"biases disabled (attention_bias=False, mlp_bias=False)."
        )

    shapes = _expected_shapes(cfg)
    consumed: set[str] = set()
    params: dict[str, np.ndarray] = {}

    def take(hf_key: str, canonical: str, transpose: bool) -> np.ndarray:
        """Fetch + convert + shape-check one required key (records consumption)."""
        if hf_key not in raw:
            raise _missing_key_error(hf_key, canonical, keys)
        array = _as_f32(hf_key, raw[hf_key])
        if transpose:
            array = np.ascontiguousarray(np.transpose(array))
        expected = shapes[canonical]
        if array.shape != expected:
            raise HFWeightError(
                f"checkpoint tensor {hf_key!r} gives canonical {canonical!r} of "
                f"shape {array.shape}, but the config (hidden_size="
                f"{cfg.hidden_size}, num_attention_heads={cfg.num_attention_heads}, "
                f"num_key_value_heads={cfg.num_key_value_heads}, head_dim="
                f"{cfg.head_dim}, intermediate_size={cfg.intermediate_size}, "
                f"vocab_size={cfg.vocab_size}) requires {expected}"
                + (
                    " — note HF stores linear weights as (out_features, in_features),"
                    " so the loader transposes them; an already-transposed checkpoint"
                    " cannot be used verbatim"
                    if transpose
                    else ""
                )
            )
        consumed.add(hf_key)
        return array

    params["embed.weight"] = take("model.embed_tokens.weight", "embed.weight", False)
    for index in range(cfg.num_hidden_layers):
        for suffix, hf_suffix, transpose in _LAYER_RECORDS:
            canonical = f"layers.{index}.{suffix}"
            params[canonical] = take(f"model.layers.{index}.{hf_suffix}", canonical, transpose)
    params["final_norm.weight"] = take("model.norm.weight", "final_norm.weight", False)

    # lm_head: a required, separate canonical entry when untied; the embedding
    # itself when tied (`embed.weight` is one graph input and the head is its
    # transpose, so no duplicate array is built). A tied checkpoint that ALSO
    # ships an lm_head must agree with the embedding — never silently pick one of
    # the two — and is then consumed-and-dropped so the leftover check passes.
    embed_raw = params["embed.weight"]
    if cfg.tie_word_embeddings:
        if "lm_head.weight" in raw:
            head_raw = _as_f32("lm_head.weight", raw["lm_head.weight"])
            if head_raw.shape != embed_raw.shape:
                raise HFWeightError(
                    f"tie_word_embeddings is set, but 'lm_head.weight' "
                    f"{head_raw.shape} and 'model.embed_tokens.weight' "
                    f"{embed_raw.shape} have different shapes, so they cannot be "
                    f"the same tensor"
                )
            if not np.allclose(head_raw, embed_raw, atol=1e-6):
                max_abs = float(np.max(np.abs(head_raw - embed_raw)))
                raise HFWeightError(
                    "tie_word_embeddings is set, but 'lm_head.weight' and "
                    f"'model.embed_tokens.weight' differ (max |delta| = {max_abs:g}); "
                    "refusing to pick one silently — either re-export the checkpoint "
                    "or set tie_word_embeddings=False in the config"
                )
            consumed.add("lm_head.weight")
    elif "lm_head.weight" in raw:
        params["lm_head.weight"] = take("lm_head.weight", "lm_head.weight", True)
    else:
        raise HFWeightError(
            "missing required checkpoint key 'lm_head.weight' (needed for canonical "
            "weight 'lm_head.weight'): the config has tie_word_embeddings=False, so "
            "the output projection cannot be derived from the token embedding — "
            "provide the full checkpoint, or set tie_word_embeddings=True if the "
            "checkpoint really ties them"
        )

    unexpected = sorted(set(keys) - consumed - _IGNORED_KEYS)
    if unexpected:
        raise HFWeightError(
            f"unsupported checkpoint keys: {unexpected} — every key must map onto "
            f"the canonical scheme (unrecognised keys are an error so a checkpoint "
            f"variant can never be silently mis-mapped); ignored by design: "
            f"{sorted(_IGNORED_KEYS)}"
        )
    return params


# ---------------------------------------------------------------------------
# graph-input specs
# ---------------------------------------------------------------------------
def param_specs(cfg: ModelConfig) -> dict[str, etl.TensorSpec]:
    """``etl.TensorSpec`` for every canonical parameter, in canonical key order.

    The weights are **graph inputs**, never ``etl.constant`` payloads, so this
    tree is exactly what ``etl.trace`` / ``etl.build`` are given alongside the
    ``input_ids`` spec. Keys and shapes match :func:`canonicalize_weights`
    exactly (linears ``(in_features, out_features)``); every spec is
    ``etl.float32``.

    Weight tying: a ``tie_word_embeddings=True`` config has **no**
    ``lm_head.weight`` spec — the output head reuses ``embed.weight`` (one graph
    input, gathered for tokens and transposed for the projection), so the tied
    set is exactly one key smaller. An untied config keeps the separate
    ``(hidden_size, vocab_size)`` head.

    Args:
        cfg: the ``family == "llama"`` config.

    Returns:
        dict: canonical key → ``etl.TensorSpec`` (float32, static shape).

    Raises:
        HFConfigError: ``cfg`` is not a llama-family :class:`ModelConfig`.
    """
    _require_llama(cfg, "param_specs")
    return {
        key: etl.TensorSpec(shape=shape, dtype=etl.float32)
        for key, shape in _expected_shapes(cfg).items()
    }


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------
def graph(
    cfg: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    *,
    pack_plan: "PackPlan | None" = None,
) -> Callable[..., object]:
    """Build the ``@etl.defn`` Llama/Mistral forward pass for one static shape.

    The forward pass is HF's ``LlamaModel``: token-embedding gather →
    ``num_hidden_layers`` of [RMSNorm → RoPE'd grouped-query causal attention →
    residual → RMSNorm → SwiGLU MLP → residual] → final RMSNorm → ``lm_head``.
    Concretely, per block::

        h  = rms_norm(x, attn_norm)                       # RMSNorm, no mean-centering
        q  = apply_rope(split_heads(linear(h, attn.q), n_q),  cos, sin)
        k  = apply_rope(split_heads(linear(h, attn.k), n_kv), cos, sin)
        v  = split_heads(linear(h, attn.v), n_kv)
        k, v = repeat_kv(k, rep), repeat_kv(v, rep)       # GQA/MQA (no-op when rep == 1)
        x += linear(attention(q, k, v, causal_mask, head_dim ** -0.5), attn.o)
        h  = rms_norm(x, mlp_norm)
        x += gated_mlp(h, mlp.gate, mlp.up, mlp.down, act)  # act(gate) * up, then down

    Everything is static: ``seq_len``/``batch_size`` are baked in, the causal mask
    and the RoPE tables are computed *in-graph* (no ``etl.constant`` payload, no
    extra inputs), and every weight is read from ``params`` as a graph input by
    its canonical **flat** key (``params[f"layers.{i}.attn.q.weight"]``, linears
    ``(in_features, out_features)`` — see :func:`canonicalize_weights`).

    Weight tying: a tied config feeds the SAME ``params["embed.weight"]``
    ``SymbolicTensor`` to the token gather and — transposed to the
    ``(in_features, out_features)`` orientation ``etl.dot`` needs — to the output
    projection; there is no extra ``lm_head`` graph input.

    Packed weights (opt-in): when ``pack_plan`` is an
    :class:`etl_hf.pack.PackPlan` the forward's FIRST statement maps the packed
    params tree back to this canonical dict with
    ``pack_plan.unpack(params)`` (static ``etl.slice`` per canonical key — no
    dynamic slice, no new IR op), and the rest of the body is unchanged. The
    positional signature stays ``(params, input_ids)``; only ``params`` becomes
    the plan's few packed inputs (:attr:`etl_hf.pack.PackPlan.specs`) instead of
    one input per canonical weight. ``pack_plan=None`` (the default) is today's
    behaviour, unchanged.

    Args:
        cfg: the ``family == "llama"`` config (one config = one graph).
        seq_len: static sequence length; ``1 <= seq_len <=
            cfg.max_position_embeddings``.
        batch_size: static batch size (default 1).
        pack_plan: optional :class:`etl_hf.pack.PackPlan` built from
            :func:`param_specs` (``etl_hf.pack.pack_plan(param_specs(cfg))``);
            ``None`` keeps one graph input per canonical weight.

    Returns:
        A fresh ``@etl.defn`` callable with positional signature
        ``(params, input_ids)``: ``params`` is the flat canonical weight dict and
        ``input_ids`` is ``(batch_size, seq_len)`` ``etl.int32``; the result is
        ``(batch_size, seq_len, cfg.vocab_size)`` float32 logits. Trace it with
        ``etl.trace`` / build it with ``etl.build(fn, param_specs(cfg),
        TensorSpec((batch_size, seq_len), etl.int32))``.

    Raises:
        HFConfigError: ``cfg`` is not llama-family; ``seq_len`` or ``batch_size``
            is not a positive integer; or ``seq_len > cfg.max_position_embeddings``
            (the graph would index positions RoPE was never defined for).
        HFUnsupportedError: ``cfg.hidden_act`` is not a supported activation
            (raised eagerly, before tracing).
    """
    _require_llama(cfg, "graph")
    seq_len = _positive_int(seq_len, "graph(seq_len)")
    batch_size = _positive_int(batch_size, "graph(batch_size)")
    if seq_len > cfg.max_position_embeddings:
        raise HFConfigError(
            f"graph(seq_len={seq_len}) exceeds the config's "
            f"max_position_embeddings={cfg.max_position_embeddings} (config field "
            f"'max_position_embeddings'); RoPE has no positions beyond it — use a "
            f"shorter window or a config with a larger limit"
        )

    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    n_layers = cfg.num_hidden_layers
    eps = float(cfg.layer_norm_eps)
    theta = cfg.rope_theta
    n_rep = n_heads // n_kv_heads
    scale = head_dim**-0.5
    act = ops.resolve_activation(cfg.hidden_act)
    tied = cfg.tie_word_embeddings
    b, s = batch_size, seq_len

    @etl.defn
    def forward(params, input_ids):
        """Llama forward: ``(params, input_ids) -> logits`` (see :func:`graph`)."""
        if pack_plan is not None:
            # Packed weights: static slicing of the few packed graph inputs back
            # into the canonical dict — the rest of the body is unchanged.
            params = pack_plan.unpack(params)
        # (b, s) ids -> (b, s, hidden); the reshape pins the static extents.
        embed = params["embed.weight"]
        x = etl.reshape(etl.gather(embed, input_ids, axis=0), (b, s, hidden))
        mask = ops.causal_mask(s)
        cos, sin = ops.rope_cos_sin(s, head_dim, theta)
        for index in range(n_layers):
            prefix = f"layers.{index}."
            h = ops.rms_norm(x, params[prefix + "attn_norm.weight"], eps)
            q = ops.split_heads(_common.linear(h, params[prefix + "attn.q.weight"]), n_heads)
            k = ops.split_heads(_common.linear(h, params[prefix + "attn.k.weight"]), n_kv_heads)
            v = ops.split_heads(_common.linear(h, params[prefix + "attn.v.weight"]), n_kv_heads)
            q = ops.apply_rope(q, cos, sin)
            k = ops.apply_rope(k, cos, sin)
            k = ops.repeat_kv(k, n_rep)
            v = ops.repeat_kv(v, n_rep)
            context = _common.attention(q, k, v, mask, scale)
            x = etl.add(x, _common.linear(context, params[prefix + "attn.o.weight"]))
            h = ops.rms_norm(x, params[prefix + "mlp_norm.weight"], eps)
            x = etl.add(
                x,
                _common.gated_mlp(
                    h,
                    params[prefix + "mlp.gate.weight"],
                    params[prefix + "mlp.up.weight"],
                    params[prefix + "mlp.down.weight"],
                    act,
                ),
            )
        x = ops.rms_norm(x, params["final_norm.weight"], eps)
        if tied:
            # The tied head IS the embedding `(vocab, hidden)`: transposed to
            # `(in_features, out_features)` it is the output projection, so the
            # one embedding graph input is reused instead of adding a duplicate.
            return _common.linear(x, etl.transpose(embed))
        return _common.linear(x, params["lm_head.weight"])

    return forward


# ---------------------------------------------------------------------------
# the numpy reference forward (parity oracle)
# ---------------------------------------------------------------------------
def _reference_param(
    params: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    """Fetch one canonical parameter as float32, validating presence and shape.

    Args:
        params: flat canonical parameter dict (``canonicalize_weights`` output).
        key: canonical key.
        shape: expected shape.

    Returns:
        np.ndarray: ``float32`` array of ``shape``.

    Raises:
        HFWeightError: the key is absent or its shape differs.
    """
    try:
        value = params[key]
    except (KeyError, TypeError) as exc:
        raise HFWeightError(
            f"reference: missing parameter {key!r}; params must be the flat "
            f"canonical dict produced by etl_hf.models.llama.canonicalize_weights"
        ) from exc
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise HFWeightError(
            f"reference: parameter {key!r} has shape {array.shape}, expected {shape}"
        )
    return array


def reference(
    cfg: ModelConfig, params: Mapping[str, np.ndarray], input_ids: np.ndarray
) -> np.ndarray:
    """Independent pure-numpy Llama/Mistral forward — the parity oracle.

    Composed **only** from :mod:`etl_hf.reference` primitives (never ``etl``), in
    a deliberately different formulation from the graph: per-head attention
    driven by :func:`etl_hf.reference.attention`, hand-written RMSNorm, host RoPE
    tables, and plain ``matmul`` in the canonical **``(in_features,
    out_features)``** orientation (``x @ W``). Both the causal mask
    (:func:`etl_hf.reference.additive_causal_mask`, ``0`` on and below the
    diagonal) and the RoPE layout (HF ``rotate_half``) are the HF conventions.

    Args:
        cfg: the ``family == "llama"`` config.
        params: flat canonical parameter dict (``canonicalize_weights`` output);
            every required key is validated for presence and shape, and the key
            set must match :func:`param_specs` exactly. A tied config's tree has
            no ``lm_head.weight`` — the head is
            ``np.transpose(params["embed.weight"])``, exactly as in the graph.
        input_ids: ``(batch_size, seq_len)`` integer token ids, each in
            ``[0, cfg.vocab_size)``.

    Returns:
        np.ndarray: float32 logits of shape ``(batch_size, seq_len,
        cfg.vocab_size)``.

    Raises:
        HFConfigError: ``cfg`` is not llama-family, or ``input_ids`` is not a
            2-D positive-shape array within ``cfg.max_position_embeddings``.
        HFWeightError: a parameter is missing, unexpected, or mis-shaped.
        ValueError: ``input_ids`` is not integer-valued or is out of vocabulary.
    """
    _require_llama(cfg, "reference")
    ids = np.asarray(input_ids)
    if ids.ndim != 2:
        raise HFConfigError(
            f"reference: input_ids must be 2-D (batch, seq_len), got shape {ids.shape}"
        )
    if ids.dtype.kind not in "iu":
        raise ValueError(
            f"reference: input_ids must be integers, got dtype {ids.dtype}"
        )
    batch, seq = ids.shape
    if batch <= 0 or seq <= 0:
        raise HFConfigError(
            f"reference: input_ids must have a positive batch and sequence length, "
            f"got shape {ids.shape}"
        )
    if seq > cfg.max_position_embeddings:
        raise HFConfigError(
            f"reference: sequence length {seq} exceeds the config's "
            f"max_position_embeddings={cfg.max_position_embeddings}"
        )
    if ids.min() < 0 or ids.max() >= cfg.vocab_size:
        raise ValueError(
            f"reference: input_ids span [{ids.min()}, {ids.max()}], outside the "
            f"vocabulary [0, {cfg.vocab_size})"
        )
    ids = ids.astype(np.int32, copy=False)

    shapes = _expected_shapes(cfg)
    missing = sorted(set(shapes) - set(params))
    extra = sorted(set(params) - set(shapes))
    if missing or extra:
        raise HFWeightError(
            f"reference: params must be exactly the canonical llama parameter set "
            f"(missing={missing}, unexpected={extra})"
        )
    weights = {key: _reference_param(params, key, shape) for key, shape in shapes.items()}

    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    eps = float(cfg.layer_norm_eps)
    scale = head_dim**-0.5
    mask = _numpy_ref.additive_causal_mask(seq)
    cos, sin = _numpy_ref.rope_tables(seq, head_dim, cfg.rope_theta)
    act = _numpy_ref.resolve_activation(cfg.hidden_act)

    def heads(x, n_heads_of_x):
        """``(b, s, n*hd) -> (b, n, s, hd)`` by explicit reshape/transpose."""
        return x.reshape(batch, seq, n_heads_of_x, head_dim).transpose(0, 2, 1, 3)

    x = weights["embed.weight"][ids].astype(np.float32)  # (b, s, hidden)
    for index in range(cfg.num_hidden_layers):
        prefix = f"layers.{index}."
        h = _numpy_ref.rms_norm(x, weights[prefix + "attn_norm.weight"], eps)
        q = heads(h @ weights[prefix + "attn.q.weight"], n_heads)
        k = heads(h @ weights[prefix + "attn.k.weight"], n_kv_heads)
        v = heads(h @ weights[prefix + "attn.v.weight"], n_kv_heads)
        q = _numpy_ref.apply_rope(q, cos, sin)
        k = _numpy_ref.apply_rope(k, cos, sin)
        # attention() repeats the KV heads itself (heads_q // heads_kv).
        context = _numpy_ref.attention(q, k, v, mask, scale)
        x = x + context @ weights[prefix + "attn.o.weight"]
        h = _numpy_ref.rms_norm(x, weights[prefix + "mlp_norm.weight"], eps)
        gated = act(h @ weights[prefix + "mlp.gate.weight"]) * (h @ weights[prefix + "mlp.up.weight"])
        x = x + gated @ weights[prefix + "mlp.down.weight"]
    x = _numpy_ref.rms_norm(x, weights["final_norm.weight"], eps)
    head = (
        np.transpose(weights["embed.weight"])
        if cfg.tie_word_embeddings
        else weights["lm_head.weight"]
    )
    logits = np.asarray(x @ head, dtype=np.float32)
    return logits.reshape(batch, seq, cfg.vocab_size)
