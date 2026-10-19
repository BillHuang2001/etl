"""The **compute-function** half: a static config → the pure ``@etl.defn`` forward.

This module owns everything that describes *what the graph computes* and nothing
about *where it runs*:

* :func:`forward_fn` — the ``@etl.defn`` forward ``(params, input_ids) ->
  logits`` for one static specialisation ``(family, config, seq_len,
  batch_size, pack_weights)``, memoised so ``id(fwd)`` is stable (that is what
  makes two identical :func:`etl_hf.build` calls return the *identical*
  ``Executable``);
* :func:`param_specs` — the ``TensorSpec`` of every graph input (the canonical
  per-weight tree, or the packed bundle);
* :func:`reference_logits` — the independent pure-numpy oracle;
* :func:`resolve_seq_len` — the single ``seq_len=None`` default.

The separation is deliberate and mirrors etl's value model: a config (plus
``seq_len``/``batch_size``) is a **static specialisation value**, so it pins one
graph; the params are ordinary **graph inputs** (:func:`param_specs`) validated
at the run boundary and never ``etl.constant`` payloads. A forward therefore
captures no parameters at all — it is pure in its specialisation, which is what
lets :mod:`etl_hf.pipeline` cache compiled artifacts per specialisation while
weights change freely between runs.

Each forward returned by :func:`forward_fn` carries its specialisation on the
object itself (``fwd._etl_hf_specialisation`` — the plain-dict attribute
``"_etl_hf_specialisation"``), so :func:`etl_hf.build` can derive
``seq_len``/``batch_size``/``pack_weights`` from the forward instead of the
caller having to repeat them, and can reject an explicit value that disagrees
with the graph it was handed.

Import-time constraint: ``numpy``, ``etl`` and sibling ``etl_hf`` modules only.
No torch / transformers / safetensors / iree at module scope.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

import etl
from etl_hf import pack as _pack
from etl_hf._validate import positive_int, require_bool, require_config
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError
from etl_hf.families import FamilySpec, resolve_family
from etl_hf.pack import PackPlan

__all__ = [
    "DEFAULT_SEQ_LEN",
    "forward_fn",
    "param_specs",
    "reference_logits",
    "resolve_seq_len",
]

#: Default ``seq_len`` when the caller passes ``None``:
#: ``min(config.max_position_embeddings, 32)`` — small enough that an exploratory
#: forward pass is instant, long enough to exercise every positional path
#: (RoPE lookups, the causal mask, the final projection).
DEFAULT_SEQ_LEN: int = 32

#: Name of the plain-dict attribute :func:`forward_fn` stamps on every forward,
#: carrying its static specialisation. Read (never written) by
#: :func:`etl_hf.build`.
SPECIALISATION_ATTR: str = "_etl_hf_specialisation"

#: Bound of the module-level ``@etl.defn`` forward memo. Small on purpose: the
#: entries are static-shape graphs (one per config/seq_len/batch/packing), and
#: the compiled artifacts they lead to live in etl's own ``CompileCache``.
_GRAPH_CACHE_MAX = 64

_GRAPH_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_GRAPH_CACHE_LOCK = threading.Lock()

#: Bound of the module-level packed-plan memo (same reasoning as
#: :data:`_GRAPH_CACHE`: a plan is a pure function of a config's ``param_specs``).
_PACK_CACHE_MAX = 64

_PACK_CACHE: "OrderedDict[tuple, PackPlan]" = OrderedDict()
_PACK_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# the seq_len default
# ---------------------------------------------------------------------------
def resolve_seq_len(config: ModelConfig, seq_len: int | None = None) -> int:
    """Resolve (and validate) the static sequence length of a graph.

    ``None`` means :data:`DEFAULT_SEQ_LEN` capped by the config's positional
    range — ``min(config.max_position_embeddings, 32)`` — which is the default
    :func:`etl_hf.forward_fn` / :func:`etl_hf.build` / :func:`etl_hf.predict`
    use.

    Args:
        config: the static model config (:class:`~etl_hf.config.ModelConfig`).
        seq_len: the requested sequence length, or ``None`` for the default.

    Returns:
        int: the validated static sequence length.

    Raises:
        HFConfigError: ``seq_len`` is not a positive integer, or it exceeds
            ``config.max_position_embeddings`` (there is no position beyond
            ``max_position_embeddings - 1`` — RoPE / the learned position table
            has nothing there).
    """
    cfg = require_config(config)
    limit = cfg.max_position_embeddings
    if seq_len is None:
        return min(limit, DEFAULT_SEQ_LEN)
    seq_len = positive_int(seq_len, "seq_len")
    if seq_len > limit:
        raise HFConfigError(
            f"seq_len={seq_len} exceeds the config's max_position_embeddings "
            f"({limit}); this checkpoint has no position beyond {limit - 1} (use a "
            f"smaller seq_len, or a config with a larger max_position_embeddings)"
        )
    return seq_len


# ---------------------------------------------------------------------------
# the memoised packed plan
# ---------------------------------------------------------------------------
def _pack_plan_for(config: ModelConfig, family: str | None = None) -> PackPlan:
    """Return the memoised :class:`~etl_hf.pack.PackPlan` for one config.

    A plan is a pure function of the family's ``param_specs(config)`` (dtypes,
    ranks and trailing dims only), so it is memoised exactly like the forward
    objects: bounded LRU, built outside the lock, first-insert-wins.

    Args:
        config: the static config.
        family: optional explicit family key; ``None`` means ``config.family``.

    Returns:
        etl_hf.pack.PackPlan: the packed-input plan (one input per
        ``(dtype, rank, shape[1:])`` bucket — a handful, never one per weight).

    Raises:
        etl_hf.HFConfigError: ``family`` disagrees with ``config.family``.
        etl_hf.HFUnsupportedError: ``family`` is not a registered family, or the
            config's specs cannot be packed (symbolic dims).
    """
    spec = resolve_family(config, family)
    key = (spec.name, config.digest)
    with _PACK_CACHE_LOCK:
        cached = _PACK_CACHE.get(key)
        if cached is not None:
            _PACK_CACHE.move_to_end(key)
            return cached
    plan = _pack.pack_plan(spec.param_specs(config))
    with _PACK_CACHE_LOCK:
        existing = _PACK_CACHE.setdefault(key, plan)
        _PACK_CACHE.move_to_end(key)
        while len(_PACK_CACHE) > _PACK_CACHE_MAX:
            _PACK_CACHE.popitem(last=False)
        return existing


# ---------------------------------------------------------------------------
# the forward
# ---------------------------------------------------------------------------
def _stamp(
    forward: object,
    spec: FamilySpec,
    config: ModelConfig,
    seq_len: int,
    batch_size: int,
    pack_weights: bool,
) -> None:
    """Stamp the static specialisation onto a forward object.

    The attribute is what lets :func:`etl_hf.build` derive ``seq_len`` /
    ``batch_size`` / ``pack_weights`` from the graph it was handed (and reject an
    explicit value that disagrees) instead of silently building an executable
    whose ids spec does not match the weights tree the caller passes. A plain
    ``setattr``; an uninspectable/exotic object is left unstamped rather than
    breaking the forward.

    Args:
        forward: the ``@etl.defn`` forward returned by the family's ``graph``.
        spec: the resolved family.
        config: the static config.
        seq_len: the static sequence length baked into the graph.
        batch_size: the static batch size baked into the graph.
        pack_weights: whether the graph takes the packed params inputs.

    Returns:
        None
    """
    metadata = {
        "family": spec.name,
        "digest": config.digest,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "pack_weights": bool(pack_weights),
    }
    try:
        setattr(forward, SPECIALISATION_ATTR, metadata)
    except AttributeError:  # pragma: no cover - only an exotic, sealed object
        pass


def forward_fn(
    config: ModelConfig,
    *,
    seq_len: int | None = None,
    batch_size: int | None = None,
    family: str | None = None,
    pack_weights: bool | None = None,
) -> Callable:
    """Return the memoised ``@etl.defn`` forward for one static specialisation.

    The forward is the whole model pass — ``(params, input_ids) -> logits`` —
    with every weight an ordinary graph **input** (:func:`param_specs`) and all
    shapes static. It captures **no parameters** and knows nothing about
    backends, devices or compilation: build it here, then stage it with
    :func:`etl_hf.build`.

    Memoised on ``(family, config.digest, seq_len, batch_size, pack_weights)``,
    so identical calls return the **same** object and ``id(fwd)`` — the identity
    component of ``etl.build_cached``'s cache key — stays stable. Entries are
    built outside the lock (tracing/graph construction is expensive) and
    first-insert-wins, so concurrent callers still observe a single object per
    key.

    The returned object also carries its specialisation as
    ``fwd._etl_hf_specialisation`` (``{"family", "digest", "seq_len",
    "batch_size", "pack_weights"}``), which :func:`etl_hf.build` uses to derive
    those values when it is not told them and to reject a mismatch when it is.

    Args:
        config: the static model config (one config = one graph).
        seq_len: static sequence length, or ``None`` for
            :func:`resolve_seq_len`'s default ``min(max_position_embeddings, 32)``.
        batch_size: static batch size, or ``None`` for 1.
        family: optional explicit family key; when given it must agree with
            ``config.family``.
        pack_weights: whether the forward takes the PACKED params inputs
            (:attr:`etl_hf.pack.PackPlan.specs`, unpacked in-graph as its first
            statement) instead of one input per canonical weight; ``None`` means
            ``False``. Part of the graph's identity: a packed and an unpacked
            forward are different functions.

    Returns:
        Callable: the ``@etl.defn``-marked forward ``(params, input_ids) ->
        logits`` — not callable eagerly (etl never executes a defn); hand it to
        :func:`etl_hf.build`, or to ``etl.trace`` for a custom staging.

    Raises:
        HFConfigError: ``seq_len``/``batch_size`` is not a positive integer, or
            ``seq_len`` exceeds ``config.max_position_embeddings``.
        HFUnsupportedError: ``family`` is not a registered family, the config's
            activation is unsupported, or a packed plan cannot be built for a
            non-static config.
        TypeError: ``config`` is not a :class:`~etl_hf.config.ModelConfig`,
            ``family`` disagrees with ``config.family`` (``HFConfigError``), or
            ``pack_weights`` is not a bool.
    """
    cfg = require_config(config)
    spec = resolve_family(cfg, family)
    resolved_seq_len = resolve_seq_len(cfg, seq_len)
    resolved_batch_size = (
        1 if batch_size is None else positive_int(batch_size, "etl_hf.forward_fn(batch_size)")
    )
    resolved_packed = (
        False if pack_weights is None else require_bool(pack_weights, "etl_hf.forward_fn(pack_weights)")
    )
    key = (spec.name, cfg.digest, resolved_seq_len, resolved_batch_size, resolved_packed)
    with _GRAPH_CACHE_LOCK:
        cached = _GRAPH_CACHE.get(key)
        if cached is not None:
            _GRAPH_CACHE.move_to_end(key)
            return cached
    plan = _pack_plan_for(cfg, spec.name) if resolved_packed else None
    forward = spec.graph(cfg, resolved_seq_len, resolved_batch_size, pack_plan=plan)
    _stamp(forward, spec, cfg, resolved_seq_len, resolved_batch_size, resolved_packed)
    with _GRAPH_CACHE_LOCK:
        existing = _GRAPH_CACHE.setdefault(key, forward)
        _GRAPH_CACHE.move_to_end(key)
        while len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX:
            _GRAPH_CACHE.popitem(last=False)
        return existing


# ---------------------------------------------------------------------------
# the graph inputs / the oracle
# ---------------------------------------------------------------------------
def param_specs(
    config: ModelConfig,
    *,
    family: str | None = None,
    pack_weights: bool | None = None,
) -> dict[str, "etl.TensorSpec"]:
    """Return the ``TensorSpec`` of every **graph input** parameter.

    Weights are graph inputs, never ``etl.constant`` payloads (etl principle 6),
    so this is the specs tree :func:`etl_hf.build` traces with — and the tree
    :func:`etl_hf.run` validates a params mapping against at the run boundary.

    Args:
        config: the static model config.
        family: optional explicit family key; when given it must agree with
            ``config.family``.
        pack_weights: ``True`` returns the PACKED inputs
            (:attr:`etl_hf.pack.PackPlan.specs` — a handful of concatenated
            tensors, the tree a ``pack_weights=True`` executable takes); ``False``
            or ``None`` (default) returns the canonical per-weight tree.

    Returns:
        dict[str, etl.TensorSpec]: canonical name → float32 static-shape spec, or
        packed input name (``"packed.N"``) → its concatenated spec.

    Raises:
        HFConfigError: ``family`` disagrees with ``config.family``.
        HFUnsupportedError: ``family`` is not a registered family, or the specs
            cannot be packed (symbolic dims).
        TypeError: ``config`` is not a :class:`~etl_hf.config.ModelConfig`, or
            ``pack_weights`` is not a bool.
    """
    cfg = require_config(config)
    spec = resolve_family(cfg, family)
    packed = False if pack_weights is None else require_bool(pack_weights, "etl_hf.param_specs(pack_weights)")
    if packed:
        return dict(_pack_plan_for(cfg, spec.name).specs)
    return spec.param_specs(cfg)


def reference_logits(
    config: ModelConfig,
    params: object,
    input_ids: object,
    *,
    family: str | None = None,
) -> np.ndarray:
    """Compute the independent pure-numpy forward — the parity oracle.

    The same numbers as a compiled run, but from a *deliberately different*
    formulation (:mod:`etl_hf.reference` primitives, per-head Python loops, no
    ``etl`` involvement), which is the way to check a compiled result rather than
    re-running it. Both inputs are explicit: nothing defaults to the checkpoint's
    own weights, so a call always says which weights and which ids it compares.

    Args:
        config: the static model config.
        params: canonical flat params mapping (:func:`param_specs` keys, host
            arrays); validated by the family's reference (``HFWeightError``).
        input_ids: ``(batch_size, seq_len)`` integer token ids — a host array, or
            an :class:`etl.Tensor` whose **host** data is exposed via
            ``.numpy()`` (a cpu tensor: one visible read; a non-cpu one raises
            ``etl.DeviceError`` — never a hidden device-to-host copy).
        family: optional explicit family key; when given it must agree with
            ``config.family``.

    Returns:
        np.ndarray: float32 logits ``(batch_size, seq_len, vocab_size)``.

    Raises:
        etl.DeviceError: ``input_ids`` is an :class:`etl.Tensor` on a non-cpu
            device.
        HFWeightError: ``params`` is missing a required key or has a wrong shape.
        HFConfigError: ``family`` disagrees with ``config.family``.
        HFUnsupportedError: ``family`` is not a registered family.
        TypeError: ``config`` is not a :class:`~etl_hf.config.ModelConfig`.
    """
    cfg = require_config(config)
    spec = resolve_family(cfg, family)
    ids = input_ids
    if isinstance(ids, etl.Tensor):
        # The oracle is pure numpy: expose a cpu Tensor's host data explicitly
        # (a non-cpu one raises etl.DeviceError — never a hidden D2H transfer).
        ids = ids.numpy()
    return np.asarray(spec.reference(cfg, params, ids), dtype=np.float32)
