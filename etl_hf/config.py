"""HuggingFace ``config.json`` → frozen static :class:`ModelConfig`.

A :class:`ModelConfig` is a **static graph specialisation value** in etl's
value model: one config produces exactly one graph. Every field therefore
specialises the traced program — changing ``num_attention_heads`` is a new
graph, not a new runtime argument. For that reason :attr:`ModelConfig.digest`
exists: it is a stable 16-hex-character hash over all graph-specialising
fields, usable directly as a build-cache key.

Two supported config families (see :data:`FAMILY_BY_MODEL_TYPE`):

* ``llama`` — HF ``LlamaConfig`` / ``MistralConfig`` / ``Qwen2Config``
  (RMSNorm, RoPE, GQA, SwiGLU-style MLP).
* ``gpt2`` — HF ``GPT2Config`` (LayerNorm, learned positional embeddings,
  tied head). Its compact key names (``n_embd``/``n_head``/``n_layer``/…)
  are normalised onto the canonical fields here.

Design rules enforced by this module (each is a place where a silent model bug
could hide, so none of them guesses silently):

* Unknown HF fields are **ignored** (but preserved verbatim in
  :attr:`ModelConfig.raw` for diagnostics).
* Missing required fields, non-positive/non-integer sizes, and inconsistent
  head counts raise :class:`~etl_hf.errors.HFConfigError` naming the field and
  the source file.
* ``rope_scaling`` (YaRN / linear / dynamic NTK …) is **not implemented** in
  v1; a config carrying it raises
  :class:`~etl_hf.errors.HFUnsupportedError` rather than silently diverging
  from HF.
* ``hidden_act`` is **not** validated against a supported-activation table
  here (the model family module owns that table); the raw HF name is stored,
  lower-cased and stripped (``"SiLU"`` → ``"silu"``).

Import-time constraint: numpy is *not* imported (this module is pure stdlib +
``etl.core`` via :mod:`etl_hf.errors`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from etl_hf.errors import HFConfigError, HFUnsupportedError

__all__ = ["ModelConfig", "FAMILY_BY_MODEL_TYPE"]

#: HF ``model_type`` → ``etl_hf`` family key. Adding a model family that reuses
#: an existing architecture is a one-line change here (plus its per-family key
#: parser in :meth:`ModelConfig.from_dict` when the key names differ).
FAMILY_BY_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "gpt2": "gpt2",
}

#: Sentinel distinguishing "field absent" from "field present with value None".
_MISSING = object()

#: HF defaults, kept in one place so they are auditable.
_LLAMA_EPS_DEFAULT = 1e-6
_GPT2_EPS_DEFAULT = 1e-5
_DEFAULT_ROPE_THETA = 10000.0
_GPT2_DEFAULT_ACTIVATION = "gelu_new"


@dataclass(frozen=True)
class ModelConfig:
    """The static, normalised description of a supported HuggingFace model.

    All size fields are positive ints; ``hidden_act`` is the raw (lower-cased)
    HF activation name and is deliberately *not* validated here — the family
    module decides which activations it can build (e.g. exact-``erf`` ``gelu``
    is numpy-backend-only). ``rope_theta`` is ``None`` for absolute-position
    families (GPT-2). ``head_dim`` honours an explicit HF override and
    otherwise equals ``hidden_size // num_attention_heads``.

    ``raw`` keeps the parsed JSON for diagnostics; it takes part in neither
    equality nor hashing, so two configs loaded from files that differ only in
    harmless extra HF fields compare (and digest) equal.
    """

    model_type: str
    family: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    hidden_act: str
    layer_norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_theta: float | None
    head_dim: int
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_json(path) -> "ModelConfig":
        """Load a config from a ``config.json`` file on disk.

        Raises :class:`~etl_hf.errors.HFConfigError` when the file cannot be
        read, is not valid JSON, or is not a JSON object.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HFConfigError(f"{path}: cannot read HuggingFace config file: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HFConfigError(f"{path}: invalid JSON in HuggingFace config: {exc}") from exc
        return ModelConfig.from_dict(raw, source=str(path))

    @staticmethod
    def from_dir(model_dir) -> "ModelConfig":
        """Load ``<model_dir>/config.json``.

        Raises :class:`~etl_hf.errors.HFConfigError` naming the directory when
        no ``config.json`` is present.
        """
        directory = Path(model_dir)
        config_path = directory / "config.json"
        if not config_path.is_file():
            raise HFConfigError(
                f"{directory}: no 'config.json' found (looked for {config_path})"
            )
        return ModelConfig.from_json(config_path)

    @classmethod
    def from_dict(cls, raw, *, source: str = "<dict>") -> "ModelConfig":
        """Normalise a parsed HF config object into a :class:`ModelConfig`.

        ``source`` names the originating file in every error message (the
        returned config's :attr:`raw` is a shallow copy of ``raw``).
        """
        if not isinstance(raw, dict):
            raise HFConfigError(
                f"{source}: HuggingFace config must be a JSON object (dict), "
                f"got {type(raw).__name__}"
            )
        model_type = raw.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            raise HFConfigError(
                f"{source}: config field 'model_type' is missing or is not a non-empty "
                f"string (got {model_type!r}); etl_hf needs it to pick a model family"
            )
        family = FAMILY_BY_MODEL_TYPE.get(model_type)
        if family is None:
            raise HFConfigError(
                f"{source}: unsupported model_type {model_type!r}; etl_hf v1 supports "
                f"{sorted(FAMILY_BY_MODEL_TYPE)}"
            )
        _reject_rope_scaling(raw, source)

        if family == "llama":
            values = _parse_llama(raw, source)
        else:  # pragma: no branch - FAMILY_BY_MODEL_TYPE has exactly two families
            values = _parse_gpt2(raw, source)
        values = _finish(values, source)
        return cls(model_type=model_type, family=family, raw=dict(raw), **values)

    # -- derived values -----------------------------------------------------

    @property
    def digest(self) -> str:
        """Stable 16-hex-character sha256 over every graph-specialising field.

        ``raw`` is excluded, so two configs describing the same graph digest
        identically even when loaded from different files with different extra
        HF fields or key order. Any change to a graph field changes the digest.
        """
        payload = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "raw"
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# field extraction helpers (every failure names the field and the source)
# ---------------------------------------------------------------------------


def _field_key(raw: dict, keys: tuple[str, ...]) -> str | None:
    """Return the first alias in ``keys`` that is present with a non-None value."""
    for key in keys:
        if raw.get(key) is not None:
            return key
    return None


def _int_field(
    raw: dict,
    keys: tuple[str, ...],
    source: str,
    *,
    default=_MISSING,
):
    """Extract a positive integer field, accepting any alias in ``keys``."""
    key = _field_key(raw, keys)
    if key is None:
        if default is _MISSING:
            raise HFConfigError(
                f"{source}: missing required config field {keys[0]!r} "
                f"(accepted names: {list(keys)})"
            )
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise HFConfigError(
            f"{source}: config field {key!r} must be an integer, "
            f"got {type(value).__name__} ({value!r})"
        )
    if value <= 0:
        raise HFConfigError(
            f"{source}: config field {key!r} must be a positive integer, got {value}"
        )
    return value


def _float_field(
    raw: dict,
    keys: tuple[str, ...],
    source: str,
    *,
    default=_MISSING,
):
    """Extract a strictly-positive float field (``int`` values are accepted)."""
    key = _field_key(raw, keys)
    if key is None:
        if default is _MISSING:
            raise HFConfigError(
                f"{source}: missing required config field {keys[0]!r} "
                f"(accepted names: {list(keys)})"
            )
        return float(default)
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HFConfigError(
            f"{source}: config field {key!r} must be a number, "
            f"got {type(value).__name__} ({value!r})"
        )
    value = float(value)
    if value <= 0.0:
        raise HFConfigError(
            f"{source}: config field {key!r} must be > 0, got {value!r}"
        )
    return value


def _bool_field(raw: dict, keys: tuple[str, ...], source: str, *, default: bool) -> bool:
    """Extract a boolean field, accepting any alias in ``keys``."""
    key = _field_key(raw, keys)
    if key is None:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        raise HFConfigError(
            f"{source}: config field {key!r} must be a boolean, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _str_field(
    raw: dict, keys: tuple[str, ...], source: str, *, default: str | None
) -> str:
    """Extract a non-empty string field, normalised to lower case."""
    key = _field_key(raw, keys)
    if key is None:
        if default is None:
            raise HFConfigError(
                f"{source}: missing required config field {keys[0]!r} "
                f"(accepted names: {list(keys)})"
            )
        return default
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise HFConfigError(
            f"{source}: config field {key!r} must be a non-empty string, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value.strip().lower()


def _reject_rope_scaling(raw: dict, source: str) -> None:
    """Reject RoPE scaling variants instead of silently ignoring them."""
    if "rope_scaling" not in raw:
        return
    value = raw["rope_scaling"]
    if value is None or value == {} or value == []:
        return
    raise HFUnsupportedError(
        f"{source}: 'rope_scaling' = {value!r} is not implemented in etl_hf v1; "
        f"a scaled-RoPE model would silently diverge from HuggingFace, so it is "
        f"rejected instead (use a checkpoint with a plain RoPE config, or build "
        f"the scaled tables yourself and pass them as graph inputs)"
    )


# ---------------------------------------------------------------------------
# per-family parsers
# ---------------------------------------------------------------------------


def _parse_llama(raw: dict, source: str) -> dict:
    """Normalise a ``LlamaConfig``/``MistralConfig``/``Qwen2Config`` JSON object."""
    num_attention_heads = _int_field(raw, ("num_attention_heads",), source)
    return {
        "vocab_size": _int_field(raw, ("vocab_size",), source),
        "hidden_size": _int_field(raw, ("hidden_size",), source),
        "num_hidden_layers": _int_field(raw, ("num_hidden_layers",), source),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": _int_field(
            raw, ("num_key_value_heads",), source, default=num_attention_heads
        ),
        "intermediate_size": _int_field(raw, ("intermediate_size",), source),
        "hidden_act": _str_field(raw, ("hidden_act",), source, default="silu"),
        "layer_norm_eps": _float_field(
            raw, ("rms_norm_eps",), source, default=_LLAMA_EPS_DEFAULT
        ),
        "max_position_embeddings": _int_field(
            raw, ("max_position_embeddings",), source
        ),
        "tie_word_embeddings": _bool_field(
            raw, ("tie_word_embeddings",), source, default=False
        ),
        "rope_theta": _float_field(
            raw, ("rope_theta",), source, default=_DEFAULT_ROPE_THETA
        ),
        "head_dim": _int_field(raw, ("head_dim",), source, default=None),
    }


def _parse_gpt2(raw: dict, source: str) -> dict:
    """Normalise a ``GPT2Config`` JSON object (``n_*`` keys → canonical names)."""
    hidden_size = _int_field(raw, ("n_embd", "hidden_size"), source)
    num_attention_heads = _int_field(raw, ("n_head", "num_attention_heads"), source)
    # HF writes "n_inner": null to mean "use the GPT-2 default" (4 * n_embd).
    intermediate_size = _int_field(
        raw, ("n_inner",), source, default=4 * hidden_size
    )
    return {
        "vocab_size": _int_field(raw, ("n_vocab", "vocab_size"), source),
        "hidden_size": hidden_size,
        "num_hidden_layers": _int_field(raw, ("n_layer", "num_hidden_layers"), source),
        "num_attention_heads": num_attention_heads,
        # GPT-2 has neither GQA nor MQA: every query head has its own KV head.
        "num_key_value_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "hidden_act": _str_field(
            raw,
            ("activation_function", "hidden_act"),
            source,
            default=_GPT2_DEFAULT_ACTIVATION,
        ),
        "layer_norm_eps": _float_field(
            raw, ("layer_norm_epsilon", "layer_norm_eps"), source, default=_GPT2_EPS_DEFAULT
        ),
        "max_position_embeddings": _int_field(
            raw, ("n_positions", "max_position_embeddings"), source
        ),
        "tie_word_embeddings": _bool_field(
            raw, ("tie_word_embeddings",), source, default=True
        ),
        # Absolute learned positions: no RoPE, hence no theta.
        "rope_theta": None,
        # GPT-2 has no per-head dimension override (and no GQA).
        "head_dim": None,
    }


def _finish(values: dict, source: str) -> dict:
    """Apply the cross-field checks and derive ``head_dim`` when unset."""
    hidden_size = values["hidden_size"]
    num_attention_heads = values["num_attention_heads"]
    num_key_value_heads = values["num_key_value_heads"]
    if hidden_size % num_attention_heads != 0:
        raise HFConfigError(
            f"{source}: hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise HFConfigError(
            f"{source}: num_attention_heads ({num_attention_heads}) must be divisible "
            f"by num_key_value_heads ({num_key_value_heads})"
        )
    if values["head_dim"] is None:
        values["head_dim"] = hidden_size // num_attention_heads
    return values
