"""The family registry: the ONE place model families are enumerated.

Every supported HuggingFace architecture is described by a :class:`FamilySpec`
— a frozen value object holding the family key, the HF ``model_type`` values it
covers, and the four functions a family module must implement::

    canonicalize_weights(cfg, raw) -> dict[str, np.ndarray]
    param_specs(cfg)               -> dict[str, etl.TensorSpec]
    graph(cfg, seq_len, batch_size=1, *, pack_plan=None) -> @etl.defn forward
    reference(cfg, params, input_ids) -> np.ndarray

Adding a family is **one module plus one entry in** :data:`FAMILIES` — nothing
else in the package needs to change (:mod:`etl_hf.build` and the public API read
the registry, they never hard-code a family). The two v1 families live in
:mod:`etl_hf.models.llama` and :mod:`etl_hf.models.gpt2`.

Two invariants are enforced at *import* time, so a drifted registry fails loudly
instead of silently mis-routing a checkpoint:

* every ``FamilySpec.name`` equals its key in :data:`FAMILIES` and the source
  module's ``FAMILY`` attribute;
* the per-family ``model_types`` are derived from
  :data:`etl_hf.config.FAMILY_BY_MODEL_TYPE` and must agree with a family
  module's own ``MODEL_TYPES`` tuple when the module declares one.

Import-time constraint: this module imports only the family modules (which
import ``numpy`` / ``etl`` — both hard dependencies) and
:mod:`etl_hf.config` / :mod:`etl_hf.errors`. It must never import torch,
transformers, safetensors or iree, so ``import etl_hf`` stays numpy-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

from etl_hf import config as _config
from etl_hf.errors import HFUnsupportedError
from etl_hf.models import gpt2 as _gpt2
from etl_hf.models import llama as _llama

__all__ = ["FAMILIES", "FamilySpec", "family_for_model_type", "get_family"]


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """A frozen description of one supported model family.

    Attributes:
        name: the ``etl_hf`` family key (``ModelConfig.family``), e.g.
            ``"llama"``.
        model_types: every HF ``model_type`` string this family covers, sorted
            (``("llama", "mistral", "qwen2")`` for the Llama family).
        canonicalize_weights: ``(cfg, raw) -> dict[str, np.ndarray]`` — the HF
            state dict → canonical ``(in_features, out_features)`` weight map.
        param_specs: ``(cfg) -> dict[str, etl.TensorSpec]`` — the graph-input
            spec of every canonical parameter.
        graph: ``(cfg, seq_len, batch_size=1, *, pack_plan=None) -> Callable`` —
            the ``@etl.defn`` forward pass ``(params, input_ids) -> logits``.
            ``pack_plan`` is an optional :class:`etl_hf.pack.PackPlan`: when
            given, the forward unpacks the plan's few packed params inputs back
            into the canonical dict as its first statement (``None`` keeps one
            graph input per canonical weight).
        reference: ``(cfg, params, input_ids) -> np.ndarray`` — the independent
            pure-numpy forward used as the parity oracle.
    """

    name: str
    model_types: tuple[str, ...]
    canonicalize_weights: Callable
    param_specs: Callable
    graph: Callable
    reference: Callable


def _model_types_for(module: object) -> tuple[str, ...]:
    """Derive one family module's covered ``model_type`` strings.

    The authoritative source is :data:`etl_hf.config.FAMILY_BY_MODEL_TYPE` (that
    map is what turns a parsed ``config.json`` into :attr:`ModelConfig.family`),
    so the registry can never disagree with config parsing. A module that also
    declares its own ``MODEL_TYPES`` tuple must agree with it — a mismatch is a
    programming error, reported at import time.

    Args:
        module: the family module (``etl_hf.models.llama`` / ``...gpt2``).

    Returns:
        tuple[str, ...]: the sorted HF ``model_type`` values for that family.

    Raises:
        RuntimeError: the module's ``MODEL_TYPES`` disagrees with the config map
            (never silently resolved to one of the two).
    """
    family = getattr(module, "FAMILY", None)
    derived = tuple(
        sorted(
            model_type
            for model_type, mapped in _config.FAMILY_BY_MODEL_TYPE.items()
            if mapped == family
        )
    )
    declared = getattr(module, "MODEL_TYPES", None)
    if declared is not None and tuple(sorted(declared)) != derived:
        raise RuntimeError(
            f"etl_hf family registry drift: {module.__name__}.MODEL_TYPES is "
            f"{tuple(declared)!r} but etl_hf.config.FAMILY_BY_MODEL_TYPE maps "
            f"{derived!r} onto family {family!r}; update one of the two"
        )
    return derived


def _build_family(module: object) -> FamilySpec:
    """Assemble a :class:`FamilySpec` from a family module's module-level API.

    Args:
        module: the family module exposing ``FAMILY``, ``canonicalize_weights``,
            ``param_specs``, ``graph`` and ``reference``.

    Returns:
        FamilySpec: the frozen registry entry.

    Raises:
        RuntimeError: the module is missing one of the required attributes.
    """
    family = getattr(module, "FAMILY", None)
    missing = [
        attribute
        for attribute in ("canonicalize_weights", "param_specs", "graph", "reference")
        if not callable(getattr(module, attribute, None))
    ]
    if not isinstance(family, str) or not family or missing:
        raise RuntimeError(
            f"etl_hf family registry: {module.__name__} must define a FAMILY string "
            f"and the callables {sorted(missing) if missing else '[]'} — got "
            f"FAMILY={family!r}, missing={sorted(missing)}"
        )
    return FamilySpec(
        name=family,
        model_types=_model_types_for(module),
        canonicalize_weights=module.canonicalize_weights,
        param_specs=module.param_specs,
        graph=module.graph,
        reference=module.reference,
    )


#: The registry of supported families, keyed by ``ModelConfig.family``.
#:
#: Read-only (a ``MappingProxyType``): the registry is populated once at import
#: and is the single source of truth for "which families exist".
FAMILIES: MappingProxyType[str, FamilySpec] = MappingProxyType(
    {
        spec.name: spec
        for spec in (_build_family(_llama), _build_family(_gpt2))
    }
)


def get_family(name: str) -> FamilySpec:
    """Look up a family by its ``etl_hf`` family key.

    Args:
        name: the family key, e.g. ``"llama"`` or ``"gpt2"`` (this is
            :attr:`ModelConfig.family`, *not* an HF ``model_type`` — use
            :func:`family_for_model_type` for that).

    Returns:
        FamilySpec: the registry entry for ``name``.

    Raises:
        HFUnsupportedError: ``name`` is not a registered family; the message
            lists the supported families in sorted order.
    """
    spec = FAMILIES.get(name) if isinstance(name, str) else None
    if spec is None:
        raise HFUnsupportedError(
            f"unsupported model family {name!r}; etl_hf v1 supports "
            f"{sorted(FAMILIES)} (add a family module plus one FAMILIES entry in "
            "etl_hf/families.py for anything else)."
        )
    return spec


def family_for_model_type(model_type: str) -> FamilySpec:
    """Resolve an HF ``model_type`` string (e.g. ``"mistral"``) to its family.

    Delegates the mapping to :data:`etl_hf.config.FAMILY_BY_MODEL_TYPE` — the
    same table :meth:`ModelConfig.from_dict` uses — so registry lookups and
    config parsing can never disagree.

    Args:
        model_type: the raw HF ``model_type`` from ``config.json``.

    Returns:
        FamilySpec: the family implementing that architecture (``"mistral"`` →
        the ``llama`` family).

    Raises:
        HFUnsupportedError: ``model_type`` is not supported; the message lists
            the supported ``model_type`` values in sorted order.
    """
    family = (
        _config.FAMILY_BY_MODEL_TYPE.get(model_type)
        if isinstance(model_type, str)
        else None
    )
    if family is None:
        raise HFUnsupportedError(
            f"unsupported HF model_type {model_type!r}; etl_hf v1 supports "
            f"{sorted(_config.FAMILY_BY_MODEL_TYPE)} (families: "
            f"{sorted(FAMILIES)})."
        )
    return get_family(family)
