"""GPT-2 family: a HuggingFace GPT-2 checkpoint as ONE compiled ``etl`` graph.

This module is the family-specific mapping between the HF ``GPT2LMHeadModel``
checkpoint layout and the ``etl_hf`` canonical parameter tree, plus the forward
graph (``etl`` ops, static shapes, weights as explicit graph inputs) and an
**independent** numpy reference (via :mod:`etl_hf.reference` primitives) the
graph is validated against.

The four public entry points are the pinned family interface (identical for
``llama``): :func:`canonicalize_weights`, :func:`param_specs`, :func:`graph`,
:func:`reference`.

GPT-2 specifics this module pins (each is a place where a silent model bug
could hide, so none of them is inferred from shapes at run time):

* **``Conv1D`` orientation.** HF's ``GPT2Conv1D`` computes ``x @ W + b`` with ``W``
  of shape ``(in_features, out_features)`` — unlike the ``nn.Linear`` layers of
  the Llama family (``(out, in)``). GPT-2 weights are therefore **never
  transposed**; only ``lm_head.weight`` (an ``nn.Linear``, ``(vocab, hidden)``)
  is transposed, and only when the config is UNTIED — a tied head is not a
  canonical key at all (see :func:`canonicalize_weights`).
* **Weight tying is one input, not two.** When ``tie_word_embeddings`` is true
  the output head *is* ``embed.weight``: the canonical tree has no
  ``lm_head.weight`` entry and the graph feeds the one embedding
  ``SymbolicTensor`` to both the token gather and the output projection.
* **Fused QKV split order.** ``attn.c_attn`` packs ``(hidden, 3 * hidden)`` and
  HF splits it ``query, key, value = c_attn(x).split(hidden, dim=2)``, i.e. the
  column blocks ``[0:H]``, ``[H:2H]``, ``[2H:3H]`` in that order. The split
  happens **at load time** here (``np.ascontiguousarray`` column blocks), so the
  graph never slices the fused matrix and the batching/diff machinery of a
  custom op never sees a fused projection.
* **LayerNorm eps placement.** Both LayerNorms add the config's
  ``layer_norm_epsilon`` to the *biased* (divide-by-N) variance before the
  square root, with the affine weight/bias applied after normalisation —
  exactly ``torch.nn.LayerNorm`` (see :func:`etl_hf.ops.layer_norm`).
* **Activation.** HF's ``activation_function`` for GPT-2 defaults to
  ``"gelu_new"`` — the *tanh* approximation
  ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))``, not exact erf-GELU. It is
  resolved through :func:`etl_hf.ops.resolve_activation`, so an exact-erf
  ``"gelu"`` config works on the numpy interpreter backend only (an explicit
  ``etl.BackendError`` on compiler backends, never a silent tanh substitute).
* **Additive causal mask.** :func:`etl_hf.ops.causal_mask` (0 on and below the
  diagonal, :data:`etl_hf.ops.MASK_NEG` above) — the HF convention.
* **Key prefixing.** Older HF exports prefix every key with ``transformer.``;
  the ``gpt2`` repo's ``model.safetensors`` drops it for block/embedding keys.
  Both spellings are accepted (see :func:`canonicalize_weights`), and a key that
  normalises onto the same canonical slot twice is an explicit error rather
  than a silent last-one-wins.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

import etl

from etl_hf import ops
from etl_hf import reference as numpy_reference
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFWeightError
from etl_hf.models._common import attention, linear, two_layer_mlp
from etl_hf.ops import resolve_activation

if TYPE_CHECKING:  # pragma: no cover - typing only (no runtime import)
    from etl_hf.pack import PackPlan

__all__ = ["canonicalize_weights", "graph", "param_specs", "reference"]

#: The single HF family key this module implements (``ModelConfig.family``).
FAMILY = "gpt2"

#: HF's optional ``transformer.`` key prefix (present in older exports only).
_PREFIX = "transformer."

#: Keys a GPT-2 state dict may carry that are *not* model parameters: the causal
#: buffers ``attn.bias`` (bool ``(1, 1, max_positions, max_positions)``) and
#: ``attn.masked_bias`` (the ``-1e4`` scalar). HF registers them non-persistent,
#: but older exports wrote them to disk; they are derived, never loaded.
_IGNORED_KEY = re.compile(r"h\.\d+\.attn\.(?:bias|masked_bias)")


# ---------------------------------------------------------------------------
# canonical shapes (the single source of truth for keys + shapes)
# ---------------------------------------------------------------------------
def _canonical_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Return ``{canonical name: shape}`` for every GPT-2 parameter.

    One dict drives both :func:`param_specs` (specs for tracing) and
    :func:`canonicalize_weights` (loaded arrays), so the two can never drift.

    Canonical orientation: every linear weight is ``(in_features,
    out_features)`` — GPT-2 ``Conv1D`` weights already are, so nothing is
    transposed here except the untied ``lm_head`` (whose HF source is
    ``(vocab, hidden)``).

    Weight tying: a ``tie_word_embeddings=True`` config has **no**
    ``lm_head.weight`` entry — the head is ``embed.weight`` itself (one graph
    input, gathered for tokens and transposed for the output projection), so the
    tied tree is exactly one key smaller.

    Args:
        cfg: a GPT-2 :class:`~etl_hf.config.ModelConfig` (validated by callers).

    Returns:
        ``dict`` from canonical parameter name to concrete shape, in canonical
        order: ``embed.weight``, ``pos_embed.weight``, then per layer the
        attention projections (``q``/``k``/``v``/``o``, each weight then bias),
        the attention LayerNorm, the MLP projections (``fc1``, ``fc2``) and the
        MLP LayerNorm, then ``final_norm.{weight,bias}`` and — untied only —
        ``lm_head.weight``.
        Dict order carries no graph meaning (``core.flatten`` sorts dict keys);
        it only keeps specs and loaded arrays reported identically.
    """
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    vocab = cfg.vocab_size
    shapes: dict[str, tuple[int, ...]] = {
        "embed.weight": (vocab, hidden),
        "pos_embed.weight": (cfg.max_position_embeddings, hidden),
    }
    for index in range(cfg.num_hidden_layers):
        prefix = f"layers.{index}."
        for name in ("q", "k", "v", "o"):
            shapes[prefix + f"attn.{name}.weight"] = (hidden, hidden)
            shapes[prefix + f"attn.{name}.bias"] = (hidden,)
        shapes[prefix + "attn_norm.weight"] = (hidden,)
        shapes[prefix + "attn_norm.bias"] = (hidden,)
        shapes[prefix + "mlp.fc1.weight"] = (hidden, intermediate)
        shapes[prefix + "mlp.fc1.bias"] = (intermediate,)
        shapes[prefix + "mlp.fc2.weight"] = (intermediate, hidden)
        shapes[prefix + "mlp.fc2.bias"] = (hidden,)
        shapes[prefix + "mlp_norm.weight"] = (hidden,)
        shapes[prefix + "mlp_norm.bias"] = (hidden,)
    shapes["final_norm.weight"] = (hidden,)
    shapes["final_norm.bias"] = (hidden,)
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (hidden, vocab)
    return shapes


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _require_family(cfg: ModelConfig, where: str) -> None:
    """Raise ``HFConfigError`` unless ``cfg`` is a GPT-2 config.

    Args:
        cfg: the config passed by the caller.
        where: name of the calling public function (used in the message).

    Raises:
        HFConfigError: ``cfg`` is not a :class:`ModelConfig` or its ``family``
            is not ``"gpt2"`` (e.g. ``"llama"`` — that family has its own
            module and a different parameter tree).
    """
    if not isinstance(cfg, ModelConfig):
        raise HFConfigError(
            f"{where}: cfg must be an etl_hf.config.ModelConfig, got "
            f"{type(cfg).__name__}"
        )
    if cfg.family != FAMILY:
        raise HFConfigError(
            f"{where}: ModelConfig.family is {cfg.family!r} but this module implements "
            f"the {FAMILY!r} family; use etl_hf.models.{cfg.family} for that config "
            "(model_type "
            f"{cfg.model_type!r})."
        )


def _check_positive_int(value: Any, what: str) -> int:
    """Return ``value`` as an ``int``, raising ``HFConfigError`` when invalid.

    Args:
        value: the caller-supplied static size.
        what: field name for the error message.

    Returns:
        int: ``value`` as a plain ``int``.

    Raises:
        HFConfigError: ``value`` is not a positive integer (bools are rejected:
            they are a static value of the wrong type, not a size).
    """
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise HFConfigError(f"{what} must be an integer, got {type(value).__name__} ({value!r})")
    if int(value) <= 0:
        raise HFConfigError(f"{what} must be a positive integer, got {value}")
    return int(value)


def _check_seq_len(cfg: ModelConfig, seq_len: Any, where: str) -> int:
    """Validate a static ``seq_len`` against the config's position budget.

    Args:
        cfg: the GPT-2 config (its ``max_position_embeddings`` bounds the learned
            position table).
        seq_len: the requested sequence length.
        where: calling context for the error message.

    Returns:
        int: ``seq_len`` as a plain ``int``.

    Raises:
        HFConfigError: ``seq_len`` is not a positive integer, or it exceeds
            ``cfg.max_position_embeddings`` (GPT-2 has a *learned* position
            table, so there is no extrapolation to fall back on).
    """
    length = _check_positive_int(seq_len, f"{where}: seq_len")
    if length > cfg.max_position_embeddings:
        raise HFConfigError(
            f"{where}: seq_len={length} exceeds ModelConfig.max_position_embeddings="
            f"{cfg.max_position_embeddings} (the learned position table length); "
            "GPT-2 cannot extrapolate past its position embeddings."
        )
    return length


def _as_f32(value: Any, key: str) -> np.ndarray:
    """Return ``value`` as a contiguous ``float32`` array, validating its dtype.

    Args:
        value: candidate tensor from a checkpoint or parameter dict.
        key: name used in the error message.

    Returns:
        ``np.ndarray``: a fresh (or converted) C-contiguous ``float32`` array.

    Raises:
        HFWeightError: ``value`` is not a real-numeric numpy-castable tensor.
    """
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise HFWeightError(
            f"{key}: expected a real-numeric tensor, got dtype {array.dtype} — "
            "convert bool/int buffers or drop them before canonicalisation"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def _check_shape(array: np.ndarray, expected: tuple[int, ...], key: str) -> None:
    """Raise ``HFWeightError`` unless ``array.shape == expected``.

    Args:
        array: the loaded tensor.
        expected: the shape the config dictates.
        key: the checkpoint key the tensor came from (named in the message).

    Raises:
        HFWeightError: shape mismatch, naming key, expected and actual shapes.
    """
    if tuple(array.shape) != tuple(expected):
        raise HFWeightError(
            f"{key}: expected shape {tuple(expected)} from the ModelConfig, got "
            f"{tuple(array.shape)} (check hidden_size / num_attention_heads / "
            "intermediate_size / vocab_size / max_position_embeddings)"
        )


# ---------------------------------------------------------------------------
# canonicalisation
# ---------------------------------------------------------------------------
def _normalize_keys(raw: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    """Strip the optional ``transformer.`` prefix from every checkpoint key.

    Args:
        raw: the HF state dict, with or without the ``transformer.`` prefix (a
            single dict may mix both spellings — the ignored causal-buffer keys
            commonly do).

    Returns:
        ``dict`` mapping the prefixed-stripped ("base") key to
        ``(original key, value)``; the original key is kept for error messages.

    Raises:
        HFWeightError: two input keys normalise onto the same base key (e.g. both
            ``wte.weight`` and ``transformer.wte.weight``) — never resolved
            silently to last-one-wins.
    """
    normalized: dict[str, tuple[str, Any]] = {}
    for key, value in raw.items():
        base = key[len(_PREFIX) :] if key.startswith(_PREFIX) else key
        previous = normalized.get(base)
        if previous is not None:
            raise HFWeightError(
                f"duplicate checkpoint key {base!r}: the state dict contains both "
                f"{previous[0]!r} and {key!r}, which normalise onto the same GPT-2 "
                "parameter; keep exactly one spelling (with or without the "
                f"'{_PREFIX}' prefix)"
            )
        normalized[base] = (key, value)
    return normalized


def _take(
    normalized: Mapping[str, tuple[str, Any]],
    consumed: set[str],
    base: str,
    expected: tuple[int, ...],
) -> np.ndarray:
    """Fetch one required checkpoint tensor by its canonical ("base") key.

    Args:
        normalized: output of :func:`_normalize_keys`.
        consumed: set of base keys already mapped; ``base`` is added on success.
        base: the prefix-stripped HF key, e.g. ``"h.0.attn.c_attn.weight"``.
        expected: the shape the config dictates for this tensor.

    Returns:
        ``np.ndarray``: the tensor as a contiguous ``float32`` array.

    Raises:
        HFWeightError: the key is missing (message names both accepted
            spellings), has a non-numeric dtype, or has the wrong shape.
    """
    entry = normalized.get(base)
    if entry is None:
        spelled = base if base.startswith("lm_head.") else _PREFIX + base
        raise HFWeightError(
            f"missing required GPT-2 checkpoint key {spelled!r} (etl_hf accepts both "
            f"{base!r} and {_PREFIX + base!r}); the checkpoint must carry every GPT-2 "
            "weight — see etl_hf/models/gpt2.py for the canonical key map"
        )
    original, value = entry
    consumed.add(base)
    array = _as_f32(value, original)
    _check_shape(array, expected, original)
    return array


def canonicalize_weights(cfg: ModelConfig, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Map a raw HF GPT-2 state dict onto the canonical parameter tree.

    Handles, all explicitly (never by guessing):

    * the optional ``transformer.`` key prefix (both spellings accepted, mixing
      allowed, duplicates rejected);
    * the fused ``attn.c_attn`` ``(hidden, 3 * hidden)`` QKV projection split into
      three ``(hidden, hidden)`` blocks in HF's ``query, key, value`` column
      order (biases split the same way along their single axis);
    * ``Conv1D`` weights kept in their ``(in, out)`` orientation (no transpose);
    * the output head: ``lm_head.weight`` (transposed) when the config is
      untied; when the config ties embeddings the head IS ``embed.weight``, so
      no ``lm_head.weight`` entry is emitted — a checkpoint that ships a copy
      must agree with ``wte.weight`` (it is verified, consumed and dropped; an
      inconsistent tied pair is an error, never a silent pick);
    * the derived causal-buffer keys (``attn.bias`` / ``attn.masked_bias``),
      which are ignored;
    * shape validation of every tensor against ``cfg``, and rejection of any
      leftover (unconsumed) key.

    Args:
        cfg: the GPT-2 config the checkpoint was written for.
        raw: ``{key: ndarray}`` as read by :mod:`etl_hf.weights` (float32 is
            preserved; bf16/f16 were already widened by the readers). The mapping
            is not mutated.

    Returns:
        ``dict`` mapping every canonical name (the exact keys of
        :func:`param_specs`) to a C-contiguous ``float32`` array of the
        corresponding spec shape; a tied head contributes no entry (the head is
        ``embed.weight``), so a tied tree has exactly that one key fewer.

    Raises:
        HFConfigError: ``cfg`` is not a GPT-2 config.
        HFWeightError: a required key is missing, a duplicate key normalises onto
            one slot, a tensor has the wrong shape/dtype, the fused QKV width is
            not ``3 * hidden_size``, a tied ``lm_head`` disagrees with
            ``wte.weight``, an untied config has no ``lm_head.weight``, or the
            state dict carries unexpected keys.
    """
    _require_family(cfg, "canonicalize_weights")
    if not isinstance(raw, Mapping):
        raise HFWeightError(
            "canonicalize_weights: raw must be a mapping of checkpoint key to array, "
            f"got {type(raw).__name__}"
        )
    normalized = _normalize_keys(raw)
    consumed: set[str] = set()
    hidden = cfg.hidden_size
    out: dict[str, np.ndarray] = {}

    out["embed.weight"] = _take(
        normalized, consumed, "wte.weight", (cfg.vocab_size, hidden)
    )
    out["pos_embed.weight"] = _take(
        normalized, consumed, "wpe.weight", (cfg.max_position_embeddings, hidden)
    )

    for index in range(cfg.num_hidden_layers):
        source = f"h.{index}."
        target = f"layers.{index}."
        out[target + "attn_norm.weight"] = _take(
            normalized, consumed, source + "ln_1.weight", (hidden,)
        )
        out[target + "attn_norm.bias"] = _take(
            normalized, consumed, source + "ln_1.bias", (hidden,)
        )
        fused_weight = _take(
            normalized, consumed, source + "attn.c_attn.weight", (hidden, 3 * hidden)
        )
        fused_bias = _take(
            normalized, consumed, source + "attn.c_attn.bias", (3 * hidden,)
        )
        # ``_take`` pinned the fused width to 3 * hidden_size; HF splits it
        # ``query, key, value = c_attn(x).split(hidden, dim=2)`` — column blocks.
        for name, start in (("q", 0), ("k", hidden), ("v", 2 * hidden)):
            out[target + f"attn.{name}.weight"] = np.ascontiguousarray(
                fused_weight[:, start : start + hidden]
            )
            out[target + f"attn.{name}.bias"] = np.ascontiguousarray(
                fused_bias[start : start + hidden]
            )
        out[target + "attn.o.weight"] = _take(
            normalized, consumed, source + "attn.c_proj.weight", (hidden, hidden)
        )
        out[target + "attn.o.bias"] = _take(
            normalized, consumed, source + "attn.c_proj.bias", (hidden,)
        )
        out[target + "mlp_norm.weight"] = _take(
            normalized, consumed, source + "ln_2.weight", (hidden,)
        )
        out[target + "mlp_norm.bias"] = _take(
            normalized, consumed, source + "ln_2.bias", (hidden,)
        )
        out[target + "mlp.fc1.weight"] = _take(
            normalized,
            consumed,
            source + "mlp.c_fc.weight",
            (hidden, cfg.intermediate_size),
        )
        out[target + "mlp.fc1.bias"] = _take(
            normalized, consumed, source + "mlp.c_fc.bias", (cfg.intermediate_size,)
        )
        out[target + "mlp.fc2.weight"] = _take(
            normalized,
            consumed,
            source + "mlp.c_proj.weight",
            (cfg.intermediate_size, hidden),
        )
        out[target + "mlp.fc2.bias"] = _take(
            normalized, consumed, source + "mlp.c_proj.bias", (hidden,)
        )

    out["final_norm.weight"] = _take(
        normalized, consumed, "ln_f.weight", (hidden,)
    )
    out["final_norm.bias"] = _take(normalized, consumed, "ln_f.bias", (hidden,))
    _resolve_lm_head(cfg, normalized, consumed, out)

    _reject_unexpected(normalized, consumed)
    _assert_complete(cfg, out)
    return out


def _resolve_lm_head(
    cfg: ModelConfig,
    normalized: Mapping[str, tuple[str, Any]],
    consumed: set[str],
    canonical: dict[str, np.ndarray],
) -> None:
    """Resolve the output head into ``canonical``, honouring weight tying.

    A tied head IS ``embed.weight`` — one canonical array, one graph input — so
    nothing is added to ``canonical`` in that case. A checkpoint that also ships
    ``lm_head.weight`` is checked against ``wte.weight`` (never a silent pick)
    and then consumed-and-dropped, so the leftover-key check does not fire and
    no second transposed copy is materialised.

    Args:
        cfg: the GPT-2 config (``tie_word_embeddings`` decides the contract).
        normalized: output of :func:`_normalize_keys`.
        consumed: set of mapped base keys (``lm_head.weight`` is added when the
            checkpoint carries one).
        canonical: the canonical dict being built; ``embed.weight`` (already
            present, ``(vocab, hidden)`` f32) is the tied head, and an untied
            ``lm_head.weight`` ``(hidden, vocab)`` is added here.

    Raises:
        HFWeightError: the config is untied but the checkpoint has no
            ``lm_head.weight``, or the config is tied and the checkpoint's
            ``lm_head.weight`` disagrees with ``wte.weight``.
    """
    entry = normalized.get("lm_head.weight")
    if not cfg.tie_word_embeddings:
        if entry is None:
            raise HFWeightError(
                "missing required GPT-2 checkpoint key 'lm_head.weight': the config "
                "sets tie_word_embeddings=False, so the output head cannot be aliased "
                "to 'transformer.wte.weight'"
            )
        array = _take(
            normalized,
            consumed,
            "lm_head.weight",
            (cfg.vocab_size, cfg.hidden_size),
        )
        canonical["lm_head.weight"] = np.ascontiguousarray(array.T)
        return
    if entry is None:
        return
    original, value = entry
    consumed.add("lm_head.weight")
    array = _as_f32(value, original)
    _check_shape(array, (cfg.vocab_size, cfg.hidden_size), original)
    if not np.allclose(array, canonical["embed.weight"], rtol=0.0, atol=1e-6):
        raise HFWeightError(
            f"{original}: the config sets tie_word_embeddings=True but this tensor "
            "differs from 'transformer.wte.weight' (they must be identical when "
            "tied); fix the checkpoint or set tie_word_embeddings=False"
        )


def _reject_unexpected(
    normalized: Mapping[str, tuple[str, Any]], consumed: set[str]
) -> None:
    """Raise ``HFWeightError`` listing every checkpoint key nothing consumed.

    Args:
        normalized: output of :func:`_normalize_keys`.
        consumed: the set of base keys the canonicalisation mapped.

    Raises:
        HFWeightError: at least one key was neither mapped nor on the ignored
            allowlist (GPT-2's derived ``attn.bias`` / ``attn.masked_bias``).
    """
    unexpected = sorted(
        original
        for base, (original, _value) in normalized.items()
        if base not in consumed and not _IGNORED_KEY.fullmatch(base)
    )
    if unexpected:
        raise HFWeightError(
            f"unexpected key(s) in the GPT-2 checkpoint: {unexpected}; etl_hf maps "
            "exactly the GPT-2 parameter set (see etl_hf/models/gpt2.py) and ignores "
            "only the derived causal buffers 'h.<i>.attn.bias' / "
            "'h.<i>.attn.masked_bias'"
        )


def _assert_complete(cfg: ModelConfig, canonical: Mapping[str, np.ndarray]) -> None:
    """Assert the canonical dict is exactly the ``param_specs`` key/shape set.

    Args:
        cfg: the GPT-2 config.
        canonical: the dict built by :func:`canonicalize_weights`.

    Raises:
        HFWeightError: an internal invariant was violated (a key is missing or
            unknown). This can only fire on a bug in this module, never on a
            differently shaped checkpoint — surfaces it loudly instead of handing
            the tracer a surprise.
    """
    expected = _canonical_shapes(cfg)
    missing = sorted(set(expected) - set(canonical))
    extra = sorted(set(canonical) - set(expected))
    if missing or extra:
        raise HFWeightError(
            "canonicalize_weights: internal key-set mismatch against param_specs "
            f"(missing={missing}, unexpected={extra}); this is an etl_hf bug"
        )
    for name, shape in expected.items():
        _check_shape(np.asarray(canonical[name]), shape, name)


# ---------------------------------------------------------------------------
# parameter specs
# ---------------------------------------------------------------------------
def param_specs(cfg: ModelConfig) -> dict[str, etl.TensorSpec]:
    """Return the ``TensorSpec`` of every GPT-2 parameter, keyed canonically.

    Weight tensors are always **explicit graph inputs** (never ``etl.constant``),
    so these specs are what the caller traces/builds with; the keys and shapes
    are exactly what :func:`canonicalize_weights` emits.

    Weight tying: when ``cfg.tie_word_embeddings`` is true there is **no**
    ``lm_head.weight`` spec — the output head reuses ``embed.weight`` (one graph
    input, gathered for tokens and transposed for the projection), so the tied
    set is exactly one key smaller. An untied config carries the separate
    ``(hidden, vocab)`` head exactly as before.

    Args:
        cfg: the GPT-2 config.

    Returns:
        ``dict`` from canonical parameter name to ``TensorSpec`` of
        ``float32`` dtype, in canonical order (``embed``, ``pos_embed``,
        ``layers.0.…``, …, ``final_norm.{weight,bias}``, then ``lm_head``
        only when untied).

    Raises:
        HFConfigError: ``cfg`` is not a GPT-2 config.
    """
    _require_family(cfg, "param_specs")
    return {
        name: etl.TensorSpec(shape, etl.float32)
        for name, shape in _canonical_shapes(cfg).items()
    }


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
def graph(
    cfg: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    *,
    pack_plan: "PackPlan | None" = None,
) -> Callable[..., Any]:
    """Build the GPT-2 forward as one ``@etl.defn`` graph.

    The graph is ``forward(params, input_ids) -> logits`` with ``params`` a dict
    keyed by the canonical names (the exact keys of :func:`param_specs`) and
    ``input_ids`` an ``(batch_size, seq_len)`` integer tensor; ``logits`` is
    ``(batch_size, seq_len, vocab_size)`` float32. All shapes are static — one
    graph per ``(cfg, seq_len, batch_size)``, the package-wide
    static-shape-first contract.

    Everything numerical goes through public ``etl`` ops and the shared
    ``etl_hf`` compositions (:mod:`etl_hf.ops` / :mod:`etl_hf.models._common`);
    the fused QKV was split at load time, so no in-graph slicing is needed.

    Weight tying: a tied config feeds the SAME ``params["embed.weight"]``
    ``SymbolicTensor`` to the token gather and — transposed to the
    ``(in_features, out_features)`` orientation ``etl.dot`` needs — to the output
    projection; there is no extra ``lm_head`` graph input.

    Packed weights (opt-in): when ``pack_plan`` is an
    :class:`etl_hf.pack.PackPlan` the forward's FIRST statement maps the packed
    params tree back to this canonical dict with
    ``pack_plan.unpack(params)`` (static ``etl.slice`` per canonical key — no
    dynamic slice, no new IR op, and the fused-QKV split still happened at load
    time), then the rest of the body is unchanged. The positional signature stays
    ``(params, input_ids)``; only ``params`` becomes the plan's few packed inputs
    (:attr:`etl_hf.pack.PackPlan.specs`) instead of one input per canonical
    weight. ``pack_plan=None`` (the default) is today's behaviour, unchanged.

    Args:
        cfg: the GPT-2 config (a static specialisation value: one config = one
            graph).
        seq_len: static sequence length; must satisfy
            ``1 <= seq_len <= cfg.max_position_embeddings``.
        batch_size: static batch size (default 1).
        pack_plan: optional :class:`etl_hf.pack.PackPlan` built from
            :func:`param_specs` (``etl_hf.pack.pack_plan(param_specs(cfg))``);
            ``None`` keeps one graph input per canonical weight.

    Returns:
        The ``@etl.defn``-marked function; pass it to ``etl.trace`` /
        ``etl.build`` / ``etl.evaluate`` (it is not callable).

    Raises:
        HFConfigError: ``cfg`` is not a GPT-2 config, ``seq_len``/``batch_size``
            are not positive integers, or ``seq_len`` exceeds
            ``cfg.max_position_embeddings``.
        HFUnsupportedError: ``cfg.hidden_act`` is not a supported activation.
    """
    _require_family(cfg, "graph")
    seq_len = _check_seq_len(cfg, seq_len, "graph")
    batch_size = _check_positive_int(batch_size, "graph: batch_size")
    act = resolve_activation(cfg.hidden_act)  # explicit error for unknown names
    hidden = cfg.hidden_size
    heads = cfg.num_attention_heads
    eps = cfg.layer_norm_eps
    scale = cfg.head_dim**-0.5
    n_layers = cfg.num_hidden_layers
    tied = cfg.tie_word_embeddings

    @etl.defn
    def gpt2_forward(params, input_ids):
        if pack_plan is not None:
            # Packed weights: static slicing of the few packed graph inputs back
            # into the canonical dict — the rest of the body is unchanged.
            params = pack_plan.unpack(params)
        embed = params["embed.weight"]
        pos_embed = params["pos_embed.weight"]
        tokens = etl.gather(embed, input_ids, axis=0)  # (batch, seq, hidden)
        positions = etl.reshape(etl.arange(0, seq_len, dtype=etl.int32), (1, seq_len))
        # gather keeps the (1, seq) index shape -> (1, seq, hidden), then the
        # explicit broadcast over the batch axis materialises (batch, seq, hidden).
        pos = etl.broadcast(
            etl.gather(pos_embed, positions, axis=0), (batch_size, seq_len, hidden)
        )
        x = etl.add(tokens, pos)
        mask = ops.causal_mask(seq_len)
        for index in range(n_layers):
            prefix = f"layers.{index}."
            normed = ops.layer_norm(
                x,
                params[prefix + "attn_norm.weight"],
                params[prefix + "attn_norm.bias"],
                eps,
            )
            query = ops.split_heads(
                linear(
                    normed,
                    params[prefix + "attn.q.weight"],
                    params[prefix + "attn.q.bias"],
                ),
                heads,
            )
            key = ops.split_heads(
                linear(
                    normed,
                    params[prefix + "attn.k.weight"],
                    params[prefix + "attn.k.bias"],
                ),
                heads,
            )
            value = ops.split_heads(
                linear(
                    normed,
                    params[prefix + "attn.v.weight"],
                    params[prefix + "attn.v.bias"],
                ),
                heads,
            )
            context = attention(query, key, value, mask, scale)
            x = etl.add(
                x,
                linear(
                    context,
                    params[prefix + "attn.o.weight"],
                    params[prefix + "attn.o.bias"],
                ),
            )
            normed = ops.layer_norm(
                x,
                params[prefix + "mlp_norm.weight"],
                params[prefix + "mlp_norm.bias"],
                eps,
            )
            mlp = two_layer_mlp(
                normed,
                params[prefix + "mlp.fc1.weight"],
                params[prefix + "mlp.fc1.bias"],
                params[prefix + "mlp.fc2.weight"],
                params[prefix + "mlp.fc2.bias"],
                act,
            )
            x = etl.add(x, mlp)
        normed = ops.layer_norm(
            x, params["final_norm.weight"], params["final_norm.bias"], eps
        )
        if tied:
            # The tied head IS the embedding `(vocab, hidden)`, so it is
            # transposed to `(hidden, vocab)` — the `(in_features,
            # out_features)` shape `linear`/`etl.dot` needs — and reused here
            # instead of adding a second, duplicate graph input.
            return linear(normed, etl.transpose(embed))
        return linear(normed, params["lm_head.weight"])

    return gpt2_forward


# ---------------------------------------------------------------------------
# independent numpy reference
# ---------------------------------------------------------------------------
def reference(
    cfg: ModelConfig, params: Mapping[str, Any], input_ids: np.ndarray
) -> np.ndarray:
    """Pure-numpy GPT-2 forward — the parity oracle for :func:`graph`.

    Written against :mod:`etl_hf.reference` primitives only (no ``etl`` import
    anywhere on this path): token/position gathering, HF LayerNorm, a per-head
    attention loop, sigmoid/tanh/erf activations. The linear algebra uses
    explicit ``numpy.dot`` calls in the canonical ``(in, out)`` orientation, so
    a wrong weight transpose or QKV split in the loader shows up as a numeric
    difference rather than a shared bug.

    Args:
        cfg: the GPT-2 config.
        params: the canonical parameter dict (``str -> ndarray``, the exact keys
            of :func:`param_specs`); extra or missing keys are an error. A tied
            config's tree has no ``lm_head.weight`` — the head is
            ``np.transpose(params["embed.weight"])``, exactly as in the graph.
        input_ids: ``(batch, seq)`` integer array of token ids.

    Returns:
        ``float32`` array ``(batch, seq, cfg.vocab_size)``.

    Raises:
        HFConfigError: ``cfg`` is not a GPT-2 config, the ids are empty, or
            ``seq`` exceeds ``cfg.max_position_embeddings``.
        HFWeightError: ``params`` is missing a canonical key, carries an unknown
            key, or a tensor has the wrong shape/dtype.
        ValueError: ``input_ids`` is not a 2-D integer array.
    """
    _require_family(cfg, "reference")
    ids = np.asarray(input_ids)
    if ids.ndim != 2:
        raise ValueError(
            f"reference: input_ids must be a (batch, seq) array, got shape {ids.shape}"
        )
    if ids.dtype.kind not in "iu":
        raise ValueError(
            f"reference: input_ids must be an integer array, got dtype {ids.dtype}"
        )
    batch, seq_len = (int(ids.shape[0]), int(ids.shape[1]))
    _check_seq_len(cfg, seq_len, "reference")
    _check_positive_int(batch, "reference: batch_size")

    shapes = _canonical_shapes(cfg)
    missing = sorted(set(shapes) - set(params))
    extra = sorted(set(params) - set(shapes))
    if missing or extra:
        raise HFWeightError(
            f"reference: params must be exactly the canonical GPT-2 parameter set "
            f"(missing={missing}, unexpected={extra})"
        )
    weights: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        array = _as_f32(params[name], name)
        _check_shape(array, shape, name)
        weights[name] = array

    act = numpy_reference.resolve_activation(cfg.hidden_act)
    eps = cfg.layer_norm_eps
    heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    scale = head_dim**-0.5

    tokens = weights["embed.weight"][ids]  # (batch, seq, hidden)
    positions = weights["pos_embed.weight"][np.arange(seq_len)]  # (seq, hidden)
    x = np.asarray(tokens + positions[None, :, :], dtype=np.float32)
    mask = numpy_reference.additive_causal_mask(seq_len)

    def heads_of(projected: np.ndarray) -> np.ndarray:
        """``(b, s, n_heads * head_dim) -> (b, n_heads, s, head_dim)``."""
        return projected.reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

    def project(hidden_states: np.ndarray, name: str) -> np.ndarray:
        """``hidden_states @ W + b`` in the canonical ``(in, out)`` orientation."""
        weight = weights[name + ".weight"]
        bias = weights[name + ".bias"]
        return np.asarray(np.dot(hidden_states, weight) + bias, dtype=np.float32)

    for index in range(cfg.num_hidden_layers):
        prefix = f"layers.{index}."
        normed = numpy_reference.layer_norm(
            x,
            weights[prefix + "attn_norm.weight"],
            weights[prefix + "attn_norm.bias"],
            eps,
        )
        context = numpy_reference.attention(
            heads_of(project(normed, prefix + "attn.q")),
            heads_of(project(normed, prefix + "attn.k")),
            heads_of(project(normed, prefix + "attn.v")),
            mask,
            scale,
        )
        x = np.asarray(x + project(context, prefix + "attn.o"), dtype=np.float32)
        normed = numpy_reference.layer_norm(
            x,
            weights[prefix + "mlp_norm.weight"],
            weights[prefix + "mlp_norm.bias"],
            eps,
        )
        activated = act(project(normed, prefix + "mlp.fc1"))
        x = np.asarray(x + project(activated, prefix + "mlp.fc2"), dtype=np.float32)

    normed = numpy_reference.layer_norm(
        x, weights["final_norm.weight"], weights["final_norm.bias"], eps
    )
    head = (
        np.transpose(weights["embed.weight"])
        if cfg.tie_word_embeddings
        else weights["lm_head.weight"]
    )
    return np.asarray(np.dot(normed, head), dtype=np.float32)
