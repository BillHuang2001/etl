"""The **loading** half: a checkpoint directory → static config + canonical params.

This module answers exactly one question — *what is this checkpoint?* — and
nothing about computation:

* :func:`load_config` reads and validates ``config.json`` into a frozen
  :class:`~etl_hf.config.ModelConfig` (a static specialisation value: one config
  = one graph);
* :func:`load_params` reads the weights (safetensors / sharded index / torch
  ``.bin`` — see :mod:`etl_hf.weights`) and canonicalises them for the family
  into a flat ``dict[str, np.ndarray]`` of host arrays, dropping the raw HF state
  dict as soon as the map is built (a checkpoint is never held twice);
* :func:`validate_params` is the public pre-flight validator (exact key set plus
  per-key shapes) for a params tree that did not come from :func:`load_params`;
* :func:`load_hf` is the documented one-call composition of the two, answering a
  :class:`LoadedModel` — a plain immutable tuple of data, with **no** methods.

Nothing here builds, traces, compiles or moves a tensor: :func:`etl_hf.build`
and :func:`etl_hf.predict` are the staging half, and they take the config/params
this module returns as ordinary values.

Import-time constraint: ``numpy``, ``etl`` and sibling ``etl_hf`` modules only.
No torch / transformers / safetensors / iree at module scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from etl_hf import weights as _weights
from etl_hf._validate import require_config
from etl_hf.config import ModelConfig
from etl_hf.errors import HFWeightError
from etl_hf.families import resolve_family

__all__ = [
    "LoadedModel",
    "load_config",
    "load_hf",
    "load_params",
    "validate_params",
]


class LoadedModel(NamedTuple):
    """A checkpoint's static config and its canonical host parameters.

    Plain data — an immutable :class:`typing.NamedTuple` with field access only.
    It deliberately carries **no** methods and no ``seq_len``: loading a
    checkpoint says nothing about how the graph will be specialised or where it
    will run, which is :func:`etl_hf.forward_fn` / :func:`etl_hf.build`'s
    business::

        loaded = etl_hf.load_hf("path/to/checkpoint")
        fwd    = etl_hf.forward_fn(loaded.config, seq_len=8)
        exe    = etl_hf.build(fwd, loaded.config)
        logits = etl_hf.run(exe, loaded.params, ids)

    A tied output head is **not** a separate canonical entry: it aliases
    ``embed.weight`` (``tie_word_embeddings=True``), so the tied tree carries one
    key fewer and no duplicated embedding array.

    Attributes:
        config: the parsed :class:`~etl_hf.config.ModelConfig`.
        params: the canonical host parameter tree — a flat mapping from the
            canonical name to a ``(in_features, out_features)``-oriented
            ``np.ndarray``, exactly the :func:`etl_hf.param_specs` keys.
        family: the resolved ``etl_hf`` family key (``"llama"`` / ``"gpt2"``).
        model_dir: the checkpoint directory this model came from, or ``None``
            when the params came from somewhere else.
    """

    config: ModelConfig
    params: dict[str, np.ndarray]
    family: str
    model_dir: Path | None


def _as_directory(model_dir: object, what: str) -> Path:
    """Validate a path-like checkpoint argument.

    Args:
        model_dir: the candidate checkpoint directory (a ``str`` or
            ``os.PathLike``).
        what: function name used in the error message.

    Returns:
        Path: ``model_dir`` as a :class:`pathlib.Path`.

    Raises:
        TypeError: ``model_dir`` is not path-like.
    """
    try:
        return Path(model_dir)
    except TypeError as exc:
        raise TypeError(
            f"{what}(model_dir=...) must be a path-like object, got "
            f"{model_dir!r} of type {type(model_dir).__name__}"
        ) from exc


def validate_params(
    config: ModelConfig,
    params: Mapping[str, np.ndarray],
    *,
    family: str | None = None,
) -> dict[str, np.ndarray]:
    """Validate a canonical params tree against a config's parameter specs.

    The documented pre-flight check for a params mapping that did **not** come
    from :func:`load_params` (a hand-edited tree, a converted checkpoint, a tree
    that went through :func:`etl_hf.pack_params` the wrong way round): keys and
    shapes must match :func:`etl_hf.param_specs` exactly, because a wrong tree
    would only fail deep inside a run.

    Args:
        config: the static model config whose specs ``params`` must satisfy.
        params: the candidate canonical params mapping (name → array).
        family: optional explicit family key; when given it must agree with
            ``config.family``.

    Returns:
        dict[str, np.ndarray]: a **new** dict of the canonical arrays (the
        caller's mapping is not retained, so later mutation cannot desynchronise
        anything).

    Raises:
        TypeError: ``params`` is not a mapping, or ``config`` is not a
            :class:`~etl_hf.config.ModelConfig`.
        HFWeightError: keys and shapes must match the family's parameter specs
            exactly — extra/missing keys or wrong shapes are always an error (no
            silent filtering or reshaping).
        HFConfigError: ``family`` disagrees with ``config.family``.
        HFUnsupportedError: ``family`` is not a registered family.
    """
    cfg = require_config(config)
    spec = resolve_family(cfg, family)
    if not isinstance(params, Mapping):
        raise TypeError(
            f"params must be a mapping of canonical name -> array, got "
            f"{type(params).__name__}"
        )
    specs = spec.param_specs(cfg)
    expected = set(specs)
    got = set(params)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise HFWeightError(
            f"params do not match the {spec.name!r} parameter specs: "
            f"missing={missing[:8]}{'…' if len(missing) > 8 else ''}, "
            f"extra={extra[:8]}{'…' if len(extra) > 8 else ''}"
        )
    canonical: dict[str, np.ndarray] = {}
    for name, tensor_spec in specs.items():
        array = np.asarray(params[name])
        if array.shape != tuple(tensor_spec.shape):
            raise HFWeightError(
                f"param {name!r} has shape {array.shape}, expected {tuple(tensor_spec.shape)}"
            )
        canonical[name] = array
    return canonical


def load_config(model_dir: object, *, family: str | None = None) -> ModelConfig:
    """Read and validate a checkpoint directory's ``config.json``.

    The static half of a checkpoint: everything that pins the graph and nothing
    that has to be allocated. ``model_dir`` must be a **directory** — it is read
    for ``config.json`` before any weights are touched.

    Args:
        model_dir: the checkpoint directory (a ``str`` or path-like).
        family: optional explicit family key; when given it must agree with the
            config's own family (never silently reinterpreted).

    Returns:
        ModelConfig: the parsed, validated config (a frozen static
        specialisation value — see :attr:`ModelConfig.digest`).

    Raises:
        TypeError: ``model_dir`` is not path-like.
        HFConfigError: ``config.json`` is missing, malformed or unsupported, or
            ``family`` disagrees with the config's own family.
        HFUnsupportedError: ``family`` is not a registered family, or the config
            uses an unsupported feature (e.g. ``rope_scaling``).
    """
    directory = _as_directory(model_dir, "load_config")
    cfg = ModelConfig.from_dir(directory)
    # Resolve (and thereby check) the family before anything reads weights: a
    # disagreeing override is a config error, not a weight-mapping surprise.
    resolve_family(cfg, family)
    return cfg


def load_params(
    model_dir: object,
    config: ModelConfig,
    *,
    family: str | None = None,
) -> dict[str, np.ndarray]:
    """Read a checkpoint's weights and canonicalise them for its family.

    Reads the checkpoint (``model.safetensors`` / sharded index /
    ``pytorch_model.bin`` — see :func:`etl_hf.weights.load_checkpoint`),
    canonicalises it for the family (``(in_features, out_features)`` linears,
    fused-QKV split, tied head resolved) and drops the raw HF state dict as soon
    as the map is built, so a checkpoint is never held twice. The result is
    validated with :func:`validate_params`, so a mis-shaped canonicalisation
    raises here rather than deep inside a run.

    Args:
        model_dir: the checkpoint directory (a ``str`` or path-like).
        config: the config read by :func:`load_config` (the canonicalisation is
            config-driven: layer counts, head counts, tying).
        family: optional explicit family key; when given it must agree with
            ``config.family``.

    Returns:
        dict[str, np.ndarray]: the canonical host params — a fresh dict, exactly
        the :func:`etl_hf.param_specs` keys, host arrays.

    Raises:
        TypeError: ``model_dir`` is not path-like, or ``config`` is not a
            :class:`~etl_hf.config.ModelConfig`.
        HFWeightError: the checkpoint cannot be read, or does not match the
            config (missing/extra/duplicate/mis-shaped keys, a disagreeing tied
            head).
        HFUnsupportedError: ``family`` is not a registered family, or the
            checkpoint/config is outside the v1 support matrix (e.g. biases on a
            llama build).
        HFConfigError: ``family`` disagrees with ``config.family``.
    """
    directory = _as_directory(model_dir, "load_params")
    cfg = require_config(config)
    spec = resolve_family(cfg, family)
    raw = _weights.load_checkpoint(directory)
    params = spec.canonicalize_weights(cfg, raw)
    # Drop the raw HF state dict: only the canonical tree is retained, so the
    # checkpoint is never held twice (and per-layer transposes are freed here).
    del raw
    return validate_params(cfg, params, family=spec.name)


def load_hf(model_dir: object, *, family: str | None = None) -> LoadedModel:
    """Load a checkpoint directory into a :class:`LoadedModel`.

    The documented one-call composition of :func:`load_config` and
    :func:`load_params`, and nothing more: no ``seq_len``, no ``batch_size``, no
    compilation. Those are *build*-time specialisation values — pass the config
    to :func:`etl_hf.forward_fn` when you are ready to stage a graph::

        loaded = etl_hf.load_hf("path/to/checkpoint")
        logits = etl_hf.predict(loaded, ids, seq_len=ids.shape[-1])

    Args:
        model_dir: the checkpoint directory (a ``str`` or path-like): it must
            contain ``config.json``; the weights are found next to it
            (:func:`etl_hf.weights.load_checkpoint`).
        family: optional explicit family key; when given it must agree with the
            config's own family (never silently reinterpreted).

    Returns:
        LoadedModel: the static config, the canonical params tree, the resolved
        family key and the directory — plain data, nothing compiled.

    Raises:
        TypeError: ``model_dir`` is not path-like.
        HFConfigError: ``config.json`` is missing/malformed/unsupported, or
            ``family`` disagrees with the config.
        HFWeightError: the checkpoint cannot be read or does not match the config
            (missing/extra/mis-shaped keys).
        HFUnsupportedError: unknown family, or a checkpoint/config outside the v1
            support matrix (e.g. biases on a llama build).
    """
    directory = _as_directory(model_dir, "load_hf")
    cfg = load_config(directory, family=family)
    params = load_params(directory, cfg, family=family)
    return LoadedModel(
        config=cfg,
        params=params,
        family=resolve_family(cfg, family).name,
        model_dir=directory,
    )
