"""Build / run layer: config + weights → one compiled ``etl`` executable.

:func:`build_model` is the single place the package turns a
:class:`~etl_hf.config.ModelConfig` into an :class:`etl.Executable`, and it is
where the *measured* backend-default matrix lives (once):

===============================  =============================================
situation                        resolution
===============================  =============================================
``backend`` explicit             used verbatim (explicit > env > default)
``backend=None``, ``ETL_BACKEND`` set  that value is used
``backend=None``, device cpu/None      ``"numpy"`` (the numpy interpreter)
``backend=None``, device cuda/rocm…    ``"iree"`` (the compiled path)
any compiled backend             ``opt_level="O3"`` injected unless passed
``"iree"`` + all-ROCm targets   ``opt_level="O1"`` injected instead
``"iree"`` + ``["llvm-cpu"]``    ``iree_compile_args`` injected unless passed
===============================  =============================================

Every injected default is a *measured* pitfall rather than taste:

* IREE's default ``--iree-llvmcpu-target-cpu=generic`` target is 12–19× slower
  than ``host`` for these graphs, hence :data:`DEFAULT_LLVM_CPU_COMPILE_ARGS`.
* IREE's ROCm codegen fails at ``opt_level`` O2/O3 on these shapes: a
  transformer graph's LayerNorm mean is a ``stablehlo.reduce`` over the hidden
  axis, and the rocm target's translation aborts with ``'func.func' op failed
  to distribute`` (measured for gpt2-shaped graphs at hidden 128/640/768 — i.e.
  ``distilgpt2`` and the real ``gpt2``), so the documented
  ``build(backend="iree", device=Device("rocm", 0))`` recipe used to fail on
  exactly the checkpoints it is for. O0/O1 compile every probed
  hidden ∈ 64…768 × heads ∈ {2,4,8,12} combination, and O1's vmfb is
  byte-identical to O0's and to the no-flag default, so
  :data:`DEFAULT_ROCM_OPT_LEVEL` is ``"O1"`` — shape-robust, no optimisation
  left on the table. An explicit ``opt_level`` (or ``ETL_OPT_LEVEL``) still
  wins.
* ``iree_compile_args`` is a **compile-stage** option: it has to reach
  ``etl.compile`` (``etl.build``/``etl.build_cached`` forward their options to
  both ``lower`` and ``compile``). Passing it to ``etl.lower`` alone is silently
  ineffective, so this module always goes through the one-call composition.

Memoisation: the graph is a pure function of ``(family, config, seq_len,
batch_size)``, so :func:`build_model` keeps a bounded (64-entry) LRU of the
``@etl.defn`` forward objects and hands ``etl.build_cached`` a **stable**
``id(fn)`` — that is exactly what makes two identical ``build_model`` calls
return the identical ``Executable`` object instead of recompiling.

**Deliberate deviation from the earlier CONTEXT draft:** the cache key does
**not** include weight identity. Weights are ordinary graph *inputs* (etl
principle 6: large tensors are explicit inputs) validated at the run boundary,
so they cannot change the graph; the graph depends only on ``(family, config,
seq_len, batch_size, backend, device, options)``. Keying on weights would mean a
pointless recompile per weights object (and per checkpoint load) with zero
change in the compiled program.

Device-resident runs: a non-cpu executable rejects host inputs at the run
boundary by design, so callers must place **both** sides explicitly —
:func:`place_params` for the weights (one ``Tensor.to(device)`` per leaf) and
:func:`place_ids` for the token ids — and then drive the documented same-device
loop. :func:`run_model` accepts ids either as a host integer array or as an
:class:`etl.Tensor`, and forwards a ``Tensor`` **verbatim** (no host copy, no
dtype cast, no device move), so a device-resident ids tensor keeps host
round-trips at zero. :meth:`HFModel.forward` is the *host-facing* helper and
refuses device params with an error naming that recipe rather than pulling data
back to the host behind the caller's back.
Packed weights (``pack_weights=True``, opt-in): the graph takes a handful of
concatenated params inputs (:mod:`etl_hf.pack`) instead of one per canonical
weight and unpacks them in-graph with static ``etl.slice`` ops. It exists for one
measured reason — IREE's Python runtime SIGSEGVs above ~161 *consumed* inputs and
the 124M ``gpt2`` declares 197 — and it changes nothing about the arithmetic:
packing happens once on the host (:func:`etl_hf.pack.pack_params` /
:meth:`HFModel.pack_params`), never per run. Unpacked stays the default, so no
existing caller changes behaviour.

Import-time constraint: only ``numpy``, ``etl`` and sibling ``etl_hf`` modules
are imported here. Backend adapters (iree / xla / tvm) are resolved lazily by
``etl`` itself, so ``import etl_hf`` never imports them.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

import etl
from etl import core as _core
from etl_hf import families as _families
from etl_hf import ops as _ops
from etl_hf import pack as _pack
from etl_hf import weights as _weights
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.families import FamilySpec
from etl_hf.pack import PackPlan

__all__ = [
    "DEFAULT_LLVM_CPU_COMPILE_ARGS",
    "DEFAULT_ROCM_OPT_LEVEL",
    "DEFAULT_SEQ_LEN",
    "HFModel",
    "build_model",
    "load_hf",
    "place_ids",
    "place_params",
    "resolve_build_options",
    "run_model",
]

#: IREE/llvm-cpu compile flags etl_hf injects by default.
#:
#: ``--iree-llvmcpu-target-cpu=host`` is the measured fix for the default
#: ``generic`` target being 12–19× slower on these graphs;
#: ``--iree-opt-data-tiling`` enables the data-tiling passes the exporter's
#: dot/gather-heavy stablehlo benefits from. Injected only for an effective
#: ``llvm-cpu`` target and only when the caller passed no ``iree_compile_args``
#: (an explicit list always wins, and these are compile-stage flags — see the
#: module docstring).
DEFAULT_LLVM_CPU_COMPILE_ARGS: tuple[str, ...] = (
    "--iree-llvmcpu-target-cpu=host",
    "--iree-opt-data-tiling",
)

#: Default ``seq_len`` for :func:`load_hf` when the caller passes ``None``:
#: ``min(cfg.max_position_embeddings, 32)`` — small enough that an exploratory
#: forward pass is instant, long enough to exercise every positional path.
DEFAULT_SEQ_LEN: int = 32

#: The default compile optimisation level injected for compiled backends
#: (``"O0"``–``"O3"`` / int form all accepted by etl; ``"O3"`` is what the
#: package's measurements used).
DEFAULT_OPT_LEVEL: str = "O3"

#: The default ``opt_level`` injected for an **all-ROCm** IREE target set.
#:
#: ``"O1"`` rather than ``"O3"`` because of a measured upstream failure:
#: ``iree-compile`` cannot translate these graphs for the ``rocm`` HAL at O2/O3
#: — the LayerNorm mean (a ``stablehlo.reduce`` over the hidden axis) makes
#: ``--iree-opt-level=O2``/``O3`` abort with ``error: 'func.func' op failed to
#: distribute`` inside ``#hal.executable.target<"rocm", …>`` (reproduced with
#: random weights for gpt2-shaped graphs at hidden 128 / 640 / 768, i.e. the
#: ``distilgpt2`` and real ``gpt2`` shapes). O0, O1 and no flag at all compile,
#: O1's vmfb is **byte-identical** to O0's and to the no-flag default, and O1
#: compiles every probed hidden ∈ 64…768 × heads ∈ {2,4,8,12} combination. So
#: O1 is the shape-robust default for ROCm, with nothing lost to the disabled
#: higher levels. An explicit ``opt_level`` (or ``ETL_OPT_LEVEL``) still wins.
DEFAULT_ROCM_OPT_LEVEL: str = "O1"

#: The backend name whose options are returned untouched (the numpy interpreter
#: documents ignoring compile options, so etl_hf injects nothing for it).
_HOST_BACKEND: str = "numpy"

#: Bound of the module-level ``@etl.defn`` forward memo. Small on purpose: the
#: entries are static-shape graphs (one per config/seq_len/batch), and the
#: compiled artifacts they lead to live in etl's own ``CompileCache``.
_GRAPH_CACHE_MAX = 64

_GRAPH_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_GRAPH_CACHE_LOCK = threading.Lock()

#: Bound of the module-level packed-plan memo (same reasoning as
#: :data:`_GRAPH_CACHE`: a plan is a pure function of a config's ``param_specs``).
_PACK_CACHE_MAX = 64

_PACK_CACHE: "OrderedDict[tuple, PackPlan]" = OrderedDict()
_PACK_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# small validation helpers (every failure names the argument)
# ---------------------------------------------------------------------------
def _positive_int(value: object, what: str) -> int:
    """Validate a positive integer argument.

    Args:
        value: the candidate value.
        what: argument name used in the error message.

    Returns:
        int: ``value`` as an ``int``.

    Raises:
        HFConfigError: ``value`` is not a positive integer (booleans are
            rejected even though ``bool`` is an ``int`` subclass — a ``True``
            sequence length is always a bug).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFConfigError(
            f"{what} must be a positive integer, got {value!r} "
            f"(type {type(value).__name__})"
        )
    return int(value)


def _require_config(cfg: object) -> ModelConfig:
    """Validate that ``cfg`` is a :class:`ModelConfig`.

    Args:
        cfg: the candidate config.

    Returns:
        ModelConfig: ``cfg`` unchanged.

    Raises:
        TypeError: ``cfg`` is not a :class:`ModelConfig` (fail before any
            attribute access, so the message is actionable).
    """
    if not isinstance(cfg, ModelConfig):
        raise TypeError(
            "expected an etl_hf.ModelConfig, got "
            f"{type(cfg).__name__}; load one with ModelConfig.from_dir(model_dir) "
            "or ModelConfig.from_json(config_path)"
        )
    return cfg


def _require_device(device: object, what: str) -> _core.Device:
    """Validate that ``device`` is an explicit :class:`etl.core.Device`.

    Args:
        device: the candidate device.
        what: argument name used in the error message.

    Returns:
        etl.core.Device: ``device`` unchanged.

    Raises:
        TypeError: ``device`` is not a :class:`etl.core.Device`; etl_hf never
            guesses a device from a string (that would be hidden placement).
    """
    if not isinstance(device, _core.Device):
        raise TypeError(
            f"{what} must be an etl.core.Device (e.g. Device('cpu', 0) or "
            f"Device('cuda', 0)), got {device!r} of type {type(device).__name__}"
        )
    return device


def _resolve_family(cfg: ModelConfig, family: str | None) -> FamilySpec:
    """Resolve the family module for ``cfg``, honouring (and checking) an override.

    Args:
        cfg: the model config.
        family: optional explicit family key; ``None`` means ``cfg.family``.

    Returns:
        FamilySpec: the family implementation.

    Raises:
        HFConfigError: ``family`` disagrees with ``cfg.family`` (never silently
            re-interpret one config with another family's weight map).
        HFUnsupportedError: ``family`` is not a registered family.
    """
    if family is None:
        return _families.get_family(cfg.family)
    spec = _families.get_family(family)
    if spec.name != cfg.family:
        raise HFConfigError(
            f"family={spec.name!r} was requested, but this config's model_type "
            f"{cfg.model_type!r} belongs to family {cfg.family!r}; pass "
            f"family={cfg.family!r} or load a config for {spec.name!r}"
        )
    return spec


def _resolve_backend_name(backend: object) -> str | None:
    """Normalise a ``backend`` argument to a backend *name* (or ``None``).

    Args:
        backend: ``None``, a backend name, or a backend object exposing
            ``.name`` (an ``etl`` ``Backend`` instance).

    Returns:
        str | None: the backend name, or ``None`` when the caller passed none.

    Raises:
        TypeError: ``backend`` is neither ``None``, a string, nor an object with
            a string ``name``.
    """
    if backend is None:
        return None
    if isinstance(backend, str):
        return backend
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    raise TypeError(
        "backend must be None, a backend name string (e.g. 'numpy', 'iree'), or "
        f"an etl backend instance with a .name, got {backend!r} "
        f"(type {type(backend).__name__})"
    )


def _env_list(variable: str) -> list[str] | None:
    """Parse a comma-separated environment variable into a list.

    Args:
        variable: environment variable name.

    Returns:
        list[str] | None: the non-empty comma-separated items, or ``None`` when
        the variable is unset/blank.
    """
    raw = os.environ.get(variable)
    if not raw or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _targets_from_device(device: _core.Device | None) -> list[str]:
    """Derive the effective IREE compile targets from a device.

    Args:
        device: the requested device, or ``None`` (host execution).

    Returns:
        list[str]: ``["llvm-cpu"]`` for a cpu/absent device, ``["cuda"]`` for
        cuda, ``["rocm"]`` for rocm/hip (``hip`` is an accepted spelling of the
        canonical rocm kind), and ``[device.kind]`` otherwise.
    """
    if device is None:
        return ["llvm-cpu"]
    if device.kind == "cpu":
        return ["llvm-cpu"]
    if device.kind == "cuda":
        return ["cuda"]
    if device.kind in ("rocm", "hip"):
        return ["rocm"]
    return [device.kind]


def _iree_targets(
    options: Mapping[str, object], device: _core.Device | None
) -> list[str]:
    """Return the *effective* IREE compile targets (option → env → device).

    Args:
        options: the caller's options (only ``target_backends`` is read).
        device: the requested device, or ``None``.

    Returns:
        list[str]: ``options["target_backends"]`` when given (a bare string is
        accepted as a one-element set), else the ``ETL_TARGET_BACKENDS``
        environment list, else the device-derived
        :func:`_targets_from_device` default.
    """
    targets = options.get("target_backends")
    if targets is None:
        targets = _env_list("ETL_TARGET_BACKENDS")
    if targets is None:
        return _targets_from_device(device)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _env_opt_level() -> bool:
    """Whether ``ETL_OPT_LEVEL`` is set to a non-blank value.

    etl resolves that variable itself at compile time (it is a documented
    process-env knob), so when it is set etl_hf injects **nothing** and lets the
    environment win over its own default.

    Returns:
        bool: ``True`` when ``ETL_OPT_LEVEL`` is set and non-blank.
    """
    return bool((os.environ.get("ETL_OPT_LEVEL") or "").strip())


def _default_opt_level(targets: list[str] | None) -> str:
    """etl_hf's ``opt_level`` default for a compiled backend.

    Args:
        targets: the effective IREE target list, or ``None`` for a non-IREE
            backend (which has no ``target_backends`` concept).

    Returns:
        str: :data:`DEFAULT_ROCM_OPT_LEVEL` (``"O1"``) when ``targets`` is
        non-empty and **every** target is rocm/hip (the codegen-safe level for
        that HAL), else :data:`DEFAULT_OPT_LEVEL` (``"O3"``) — so llvm-cpu,
        cuda, xla, tvm and any mixed/unknown target set are unchanged.
    """
    if targets and all(kind in ("rocm", "hip") for kind in targets):
        return DEFAULT_ROCM_OPT_LEVEL
    return DEFAULT_OPT_LEVEL


# ---------------------------------------------------------------------------
# the default-option matrix
# ---------------------------------------------------------------------------
def resolve_build_options(
    backend: object,
    device: object,
    options: Mapping[str, object],
) -> tuple[str, dict]:
    """Resolve the default backend and the injected compile options.

    This is the *one* implementation of the documented default matrix; both
    :func:`build_model` and any caller that needs to know what would be built
    use it, so the defaults can never drift between call sites.

    Precedence (binding): **explicit argument > environment > etl_hf default**.

    * ``backend=None`` → ``os.environ["ETL_BACKEND"]`` when set and non-blank
      (honoured rather than fought — ``etl`` itself would apply it), else
      ``"numpy"`` for a cpu/absent device and ``"iree"`` otherwise (the compiled
      path is what a non-cpu device wants).
    * the ``numpy`` backend → ``options`` returned as an unchanged **copy** (the
      numpy interpreter documents ignoring compile options; nothing is
      injected).
    * any other backend → ``opt_level`` injected unless the caller passed it
      **and** unless ``ETL_OPT_LEVEL`` is set/non-blank (etl resolves that
      variable itself at compile time, so the environment wins over this
      default). The injected value depends on the effective IREE targets:
      ``"O1"`` (:data:`DEFAULT_ROCM_OPT_LEVEL`) when they are **all** rocm/hip —
      IREE's ROCm translation fails at O2/O3 on transformer-shaped graphs — and
      ``"O3"`` (:data:`DEFAULT_OPT_LEVEL`) for every other target (llvm-cpu,
      cuda, a mixed set, or a backend without targets such as xla/tvm).
    * backend ``"iree"`` whose *effective* target set is exactly
      ``["llvm-cpu"]`` → ``iree_compile_args=list(DEFAULT_LLVM_CPU_COMPILE_ARGS)``
      injected unless the caller passed ``iree_compile_args``. The effective
      targets are ``options["target_backends"]``, else the
      ``ETL_TARGET_BACKENDS`` environment variable, else derived from the device
      (cpu/``None`` → ``["llvm-cpu"]``, cuda → ``["cuda"]``, rocm/hip →
      ``["rocm"]``). Nothing is ever injected for cuda/rocm (those flags are
      llvm-cpu-specific).

    Args:
        backend: ``None``, a backend name, or a backend object with ``.name``.
        device: ``None`` or an :class:`etl.core.Device`; it decides the default
            backend and the derived IREE targets.
        options: caller compile options; **not mutated** (a copy is returned).

    Returns:
        tuple[str, dict]: ``(resolved_backend_name, resolved_options)``.

    Raises:
        TypeError: ``backend`` is not a name/backend instance, or ``device`` is
            neither ``None`` nor an :class:`etl.core.Device`.
    """
    name = _resolve_backend_name(backend)
    if device is not None:
        _require_device(device, "device")
    if name is None:
        raw = os.environ.get("ETL_BACKEND")
        name = raw.strip() if raw and raw.strip() else None
        if name is None:
            name = _HOST_BACKEND if device is None or device.kind == "cpu" else "iree"

    resolved = dict(options)
    if name == _HOST_BACKEND:
        return name, resolved

    targets = _iree_targets(resolved, device) if name == "iree" else None
    if "opt_level" not in resolved and not _env_opt_level():
        resolved["opt_level"] = _default_opt_level(targets)
    if targets == ["llvm-cpu"]:
        resolved.setdefault(
            "iree_compile_args", list(DEFAULT_LLVM_CPU_COMPILE_ARGS)
        )
    return name, resolved


# ---------------------------------------------------------------------------
# the memoised @etl.defn forward
# ---------------------------------------------------------------------------
def _pack_plan_for(spec: FamilySpec, cfg: ModelConfig) -> PackPlan:
    """Return the memoised :class:`~etl_hf.pack.PackPlan` for one config.

    A plan is a pure function of the family's ``param_specs(cfg)`` (dtypes, ranks
    and trailing dims only), so it is memoised exactly like the forward objects:
    bounded LRU, built outside the lock, first-insert-wins.

    Args:
        spec: the resolved family.
        cfg: the static config.

    Returns:
        etl_hf.pack.PackPlan: the packed-input plan (one input per
        ``(dtype, rank, shape[1:])`` bucket — a handful, never one per weight).
    """
    key = (spec.name, cfg.digest)
    with _PACK_CACHE_LOCK:
        cached = _PACK_CACHE.get(key)
        if cached is not None:
            _PACK_CACHE.move_to_end(key)
            return cached
    plan = _pack.pack_plan(spec.param_specs(cfg))
    with _PACK_CACHE_LOCK:
        existing = _PACK_CACHE.setdefault(key, plan)
        _PACK_CACHE.move_to_end(key)
        while len(_PACK_CACHE) > _PACK_CACHE_MAX:
            _PACK_CACHE.popitem(last=False)
        return existing


def _forward_fn(
    spec: FamilySpec,
    cfg: ModelConfig,
    seq_len: int,
    batch_size: int,
    pack_weights: bool = False,
) -> Callable:
    """Return the memoised ``@etl.defn`` forward for one static specialisation.

    The graph depends only on ``(family, config, seq_len, batch_size, packed)`` —
    never on weights or on a backend — so a bounded LRU of forward objects is
    safe and makes ``id(fn)`` (``etl.build_cached``'s identity component) stable
    across ``build_model`` calls. Entries are built outside the lock (tracing is
    expensive) and first-insert-wins, so concurrent callers still observe a
    single object per key.

    Args:
        spec: the resolved family.
        cfg: the static config.
        seq_len: static sequence length.
        batch_size: static batch size.
        pack_weights: whether the forward takes the PACKED params inputs (it then
            unpacks them in-graph as its first statement). Part of the cache key:
            a packed and an unpacked graph are different functions.

    Returns:
        Callable: the ``@etl.defn`` forward ``(params, input_ids) -> logits``.
    """
    key = (spec.name, cfg.digest, seq_len, batch_size, bool(pack_weights))
    with _GRAPH_CACHE_LOCK:
        cached = _GRAPH_CACHE.get(key)
        if cached is not None:
            _GRAPH_CACHE.move_to_end(key)
            return cached
    plan = _pack_plan_for(spec, cfg) if pack_weights else None
    forward = spec.graph(cfg, seq_len, batch_size, pack_plan=plan)
    with _GRAPH_CACHE_LOCK:
        existing = _GRAPH_CACHE.setdefault(key, forward)
        _GRAPH_CACHE.move_to_end(key)
        while len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX:
            _GRAPH_CACHE.popitem(last=False)
        return existing


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _check_activation(cfg: ModelConfig, backend_name: str) -> None:
    """Gate a config's activation against the resolved backend.

    Runs **before any tracing**, so an unusable activation costs nothing and
    never surfaces as a confusing mid-lower error.

    Args:
        cfg: the model config (its ``hidden_act`` is the HF name).
        backend_name: the resolved backend name; ``"numpy"`` accepts every
            supported activation (the interpreter runs ``etl.erf`` happily).

    Returns:
        None

    Raises:
        HFUnsupportedError: the activation name is unsupported (any backend), or
            it needs ``etl.erf`` — a v1 StableHLO export deferral — and the
            backend is a compiled one. The message names the activation, the
            backend and both remedies.
    """
    activation = cfg.hidden_act
    export_safe = _ops.activation_is_export_safe(activation)  # validates the name
    if export_safe or backend_name == _HOST_BACKEND:
        return
    raise HFUnsupportedError(
        f"hidden_act={activation!r} needs etl.erf, which the StableHLO exporter "
        f"defers in v1, so it cannot run on backend {backend_name!r}. Two "
        f"remedies: build with backend='numpy' (the exact-erf host path — same "
        f"numbers, no compilation), or use a tanh-approximate config "
        f"(hidden_act='gelu_new'), which differs from exact erf GELU by at most "
        f"~5e-4 per activation value (measured max |gelu_tanh - gelu_erf| = "
        f"4.7e-4, peaking near |x| ~ 2.7)."
    )


def _require_bool(value: object, what: str) -> bool:
    """Validate a boolean flag argument.

    Args:
        value: the candidate value.
        what: argument name used in the error message.

    Returns:
        bool: ``value`` as a ``bool``.

    Raises:
        TypeError: ``value`` is not a ``bool`` (an int/str would silently flip
            the graph's params tree shape, so it is rejected).
    """
    if not isinstance(value, bool):
        raise TypeError(
            f"{what} must be a bool (True/False), got {value!r} of type "
            f"{type(value).__name__}"
        )
    return value


def build_model(
    cfg: ModelConfig,
    seq_len: int,
    *,
    batch_size: int = 1,
    backend: object = None,
    device: object = None,
    cache: object = None,
    family: str | None = None,
    pack_weights: bool = False,
    **options: object,
) -> "etl.Executable":
    """Build (trace → lower → compile → load) one model forward executable.

    The whole forward pass becomes **one** compiled graph whose weights are
    ordinary graph inputs (:func:`FamilySpec.param_specs`) and whose
    ``input_ids`` is a ``(batch_size, seq_len)`` ``etl.int32`` input. All shapes
    are static: a different ``seq_len``/``batch_size`` is a different graph.

    Calls are memoised: the graph is a pure function of ``(family, config,
    seq_len, batch_size)``, so two identical calls return the **same**
    ``Executable`` object and no second compile happens. Backend/device/options
    participate in etl's cache key, so changing any of them is a new artifact.
    Weight identity deliberately does **not** (see the module docstring).

    Args:
        cfg: the model config — a static specialisation value (one config = one
            graph).
        seq_len: static sequence length; ``1 <= seq_len <=
            cfg.max_position_embeddings``.
        batch_size: static batch size (default 1).
        backend: ``None`` (resolve the documented default matrix, see
            :func:`resolve_build_options`), a backend name (``"numpy"``,
            ``"iree"``, ``"xla"``, ``"tvm"`` …) or an etl backend instance.
        device: ``None`` or an :class:`etl.core.Device` the executable should run
            on.
        cache: an :class:`etl.CompileCache` to memoise into, or ``None`` for the
            process-wide default cache.
        family: optional explicit family key; when given it must agree with
            ``cfg.family``.
        pack_weights: opt-in **packed weights** (default ``False``). When
            ``True`` the graph takes a handful of concatenated params inputs
            (:attr:`etl_hf.pack.PackPlan.specs`, one per ``(dtype, rank,
            shape[1:])`` bucket) instead of one input per canonical weight, and
            unpacks them in-graph with static ``etl.slice`` ops. This is what
            makes the 124M-model case runnable on IREE (its runtime SIGSEGVs
            above ~161 consumed inputs; ``gpt2`` declares 197 unpacked). Run it
            with the packed tree — ``etl_hf.pack_params(plan, params)`` or
            :meth:`HFModel.pack_params` — never with the canonical tree.
        **options: compile options forwarded to ``etl.build_cached`` (and hence
            to both ``lower`` and ``compile`` — that is what makes
            ``iree_compile_args`` effective). Explicit values always beat the
            injected defaults.

    Returns:
        etl.Executable: the loaded executable; call it with a params mapping and
        integer ``input_ids`` through :func:`run_model` / ``etl.run``. For
        ``pack_weights=True`` that mapping is the packed tree.

    Raises:
        HFConfigError: ``seq_len``/``batch_size`` is not a positive integer,
            ``seq_len`` exceeds ``cfg.max_position_embeddings``, ``family``
            disagrees with ``cfg.family``, or the backend/device argument is
            malformed.
        HFUnsupportedError: unknown ``family``, unsupported ``cfg.hidden_act``,
            an erf-based activation on a compiled backend (raised before any
            tracing), or a non-static config shape that cannot be packed.
        TypeError: ``cfg`` is not a :class:`ModelConfig`, or ``pack_weights`` is
            not a bool.
        etl.BackendError: the resolved backend is unavailable or rejects the
            graph (propagated verbatim — never a silent fallback).
    """
    cfg = _require_config(cfg)
    seq_len = _positive_int(seq_len, "build_model(seq_len)")
    batch_size = _positive_int(batch_size, "build_model(batch_size)")
    pack_weights = _require_bool(pack_weights, "build_model(pack_weights)")
    limit = cfg.max_position_embeddings
    if seq_len > limit:
        raise HFConfigError(
            f"build_model(seq_len={seq_len}) exceeds the config's "
            f"max_position_embeddings ({limit}); this checkpoint has no position "
            f"beyond {limit - 1} (use a smaller seq_len, or a config with a larger "
            f"max_position_embeddings)"
        )
    spec = _resolve_family(cfg, family)
    resolved_backend, resolved_options = resolve_build_options(
        backend, device, options
    )
    _check_activation(cfg, resolved_backend)
    forward = _forward_fn(spec, cfg, seq_len, batch_size, pack_weights)
    if pack_weights:
        params_specs: Mapping[str, "etl.TensorSpec"] = _pack_plan_for(spec, cfg).specs
    else:
        params_specs = spec.param_specs(cfg)
    return etl.build_cached(
        forward,
        params_specs,
        etl.TensorSpec((batch_size, seq_len), etl.int32),
        backend=resolved_backend,
        device=device,
        cache=cache,
        **resolved_options,
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _ids_spec(executable: "etl.Executable") -> "etl.TensorSpec | None":
    """Return the ``input_ids`` spec recorded by the executable, if available.

    The forward signature is ``(params, input_ids)``, so the ids spec is the
    record's last entry. Executables produced by :func:`build_model` always have
    it; an executable from some other graph returns ``None`` and the caller
    skips the spec-shape check rather than guessing.

    Args:
        executable: the executable to inspect.

    Returns:
        etl.TensorSpec | None: the ids spec, or ``None`` when unavailable.
    """
    signature = getattr(executable, "signature", None)
    specs = getattr(signature, "input_specs", None)
    if not specs:
        return None
    return specs[-1]


def _check_ids_tensor(
    ids: "etl.Tensor", expected: "etl.TensorSpec | None" = None
) -> "etl.Tensor":
    """Validate an :class:`etl.Tensor` of token ids from its **metadata** only.

    A tensor of ids may already live on the executable's device (the
    device-resident path), so validation never touches its data: rank, dtype and
    the static shape against the executable's ids spec are all read from
    ``Tensor.dtype``/``Tensor.shape``. The tensor is returned unchanged — no
    host copy, no dtype cast, no device movement.

    Args:
        ids: the caller's id tensor.
        expected: the executable's ``input_ids`` spec, or ``None`` when the
            executable does not record one (the shape check is then skipped
            rather than guessed).

    Returns:
        etl.Tensor: ``ids`` unchanged.

    Raises:
        etl.ShapeError: ``ids`` is not 2-D, or its shape does not match the
            executable's static ``(batch, seq_len)`` spec.
        etl.DTypeError: ``ids.dtype`` is not integral.
    """
    shape = tuple(ids.shape)
    want = None if expected is None else tuple(expected.shape)
    if len(shape) != 2:
        raise etl.ShapeError(
            f"input_ids must be a 2-D etl.Tensor (batch, seq_len), got shape "
            f"{shape} (rank {len(shape)})"
            + (f"; this executable expects {want}" if want is not None else "")
        )
    if want is not None and all(isinstance(dim, int) for dim in want) and want != shape:
        raise etl.ShapeError(
            f"input_ids Tensor shape {shape} does not match this executable's "
            f"(batch, seq_len) input spec {want} (expected {want[1]} tokens per "
            f"row for a static graph; build for another length instead of "
            f"reshaping)"
        )
    if ids.dtype.kind not in "iu":
        raise etl.DTypeError(
            f"input_ids must be an etl.Tensor of an integral dtype (token ids are "
            f"integers; a host array would be cast to int32 — a tensor's data is "
            f"never re-typed), got dtype {ids.dtype}"
        )
    return ids


def _params_keys(executable: "etl.Executable") -> tuple | None:
    """Return the params-dict keys recorded by the executable, if available.

    The forward signature is ``(params, input_ids)`` and etl flattens the params
    dict into leaves, recording the *structure* (``signature.input_tree``) beside
    the flat ``input_specs``. The params dict is the tree's first child and its
    ``node_data`` carries the sorted key list — canonical weight names, or the
    packed ``packed.N`` names for a ``pack_weights=True`` executable. An
    executable from some other graph returns ``None`` and the caller skips the
    check rather than guessing.

    Args:
        executable: the executable to inspect.

    Returns:
        tuple | None: the recorded params keys, or ``None`` when unavailable.
    """
    tree = getattr(getattr(executable, "signature", None), "input_tree", None)
    children = getattr(tree, "children", None)
    if not children:
        return None
    keys = getattr(children[0], "node_data", None)
    if not isinstance(keys, (list, tuple)) or not keys:
        return None
    return tuple(keys)


def _check_params_mode(
    executable: "etl.Executable", params: Mapping[str, np.ndarray]
) -> None:
    """Raise an actionable error when params and executable disagree on packing.

    ``etl.run`` reports a pytree structure mismatch, which says nothing about
    *why* the two trees differ. A packed executable takes a handful of
    ``packed.N`` inputs while the canonical tree has one key per weight, and
    mixing them is the one mistake this guard translates into a remedy that
    names ``pack_weights``.

    Only a genuine packed-vs-canonical mix is reported here; any other mismatch
    (a typo'd key, a wrong shape) is left to etl's own structure/shape errors.

    Args:
        executable: the executable about to run.
        params: the caller's params mapping.

    Returns:
        None

    Raises:
        HFWeightError: the executable takes packed params inputs and ``params``
            is a canonical tree, or the executable takes canonical weight inputs
            and ``params`` is a packed tree.
    """
    expected = _params_keys(executable)
    if expected is None:
        return
    expected_keys = set(expected)
    got = set(params)
    if got == expected_keys:
        return
    expected_packed = _pack.is_packed_tree(expected_keys)
    got_packed = _pack.is_packed_tree(got)
    if expected_packed and not got_packed:
        raise HFWeightError(
            f"this executable was built with pack_weights=True, so it takes "
            f"{len(expected)} PACKED params inputs "
            f"({sorted(expected_keys)[:8]}) — one concatenated input per "
            f"(dtype, rank, trailing-shape) bucket — but it was given "
            f"{len(got)} canonical weight keys ({sorted(got)[:8]}). Pack the "
            f"params first, e.g. "
            f"plan = etl_hf.pack_plan(param_specs); "
            f"packed = etl_hf.pack_params(plan, params) "
            f"(or HFModel.pack_params()), and pass the packed tree; to run the "
            f"canonical tree instead, build with pack_weights=False."
        )
    if got_packed and not expected_packed:
        raise HFWeightError(
            f"this executable was built with pack_weights=False, so it takes the "
            f"{len(expected)} canonical weight inputs "
            f"({sorted(expected_keys)[:8]}), but it was given packed keys "
            f"({sorted(got)[:8]}). Pass the canonical params tree, or rebuild "
            f"with pack_weights=True together with "
            f"etl_hf.pack_params(etl_hf.pack_plan(param_specs), params)."
        )


def _reject_device_params(params: Mapping[str, object], what: str) -> None:
    """Reject device-resident params handed to a **host-facing** helper.

    :meth:`HFModel.forward`/``reference`` answer in host ``numpy``, so they would
    have to pull a device tree back to the host — an implicit transfer this
    package never performs. Passing placed (non-cpu) ``Tensor`` leaves is
    therefore an error naming the device-resident recipe instead of a failure
    deep inside ``np.asarray``. A cpu ``Tensor`` (including an ``iree``
    llvm-cpu payload) is fine: it is host-addressable by etl's own rules.

    Args:
        params: the params mapping about to be used.
        what: helper name used in the error message.

    Returns:
        None

    Raises:
        etl.DeviceError: a leaf is an :class:`etl.Tensor` on a non-cpu device.
    """
    for name, value in params.items():
        if not isinstance(value, etl.Tensor) or value.device.kind == "cpu":
            continue
        raise etl.DeviceError(
            f"{what} answers in HOST logits, so it does not accept "
            f"device-resident params: parameter {name!r} lives on "
            f"{value.device!r}, and nothing is copied back to the host "
            f"implicitly. Use the explicit device-resident recipe instead: "
            f"device = etl.core.Device(...); "
            f"exe = model.build(backend='iree', device=device); "
            f"out = etl_hf.run_model(exe, "
            f"etl_hf.place_params(model.params(), device), "
            f"etl_hf.place_ids(input_ids, device)); "
            f"logits = out.to(etl.core.Device('cpu', 0)) "
            f"(or drive etl.run directly with the same placed inputs, feeding "
            f"each run's outputs back as the next run's inputs). To keep using "
            f"{what}, pass the host params tree (or cpu Tensors)."
        )


def run_model(
    executable: "etl.Executable",
    params: Mapping[str, np.ndarray],
    input_ids: object,
) -> "etl.Tensor":
    """Run a built model executable on one params tree and one id batch.

    Thin, explicit validation followed by ``etl.run``. ``input_ids`` may be
    either of two things:

    * a **host integer array** (any integral dtype) — the **one** documented
      host-side coercion is casting token ids to ``int32`` (ids are integral by
      definition; etl's run boundary is dtype-strict). It must be 2-D and match
      the executable's ``input_ids`` spec exactly.
    * an :class:`etl.Tensor` — validated from its **metadata** (rank 2, integral
      dtype, static shape vs the ids spec) and then forwarded to ``etl.run``
      **verbatim**: no ``np.asarray``, no host copy, no dtype cast, no device
      move. This is what makes the device-resident recipe reachable —
      ``model.build(device=Device('rocm', 0))`` plus
      ``place_params(params, device)`` and ``place_ids(ids, device)`` runs with
      zero host round-trips, and a host tensor of ids still reaches a cpu
      executable as the same object.

    ``params`` is passed through **untouched** — etl validates keys, shapes and
    dtypes and reports the first mismatching pytree path. Nothing is filtered or
    renamed here. The one exception is the packed-vs-canonical mix: feeding a
    canonical tree to a ``pack_weights=True`` executable (or the reverse) raises
    an :class:`~etl_hf.errors.HFWeightError` naming ``pack_weights`` and the
    remedy instead of a bare etl pytree error.

    Args:
        executable: an executable whose signature is ``(params, input_ids)``,
            e.g. the result of :func:`build_model`.
        params: canonical flat params mapping (:func:`FamilySpec.param_specs`
            keys) of host arrays or placed :class:`etl.Tensor` leaves — or, for a
            ``pack_weights=True`` executable, the packed tree it was built for
            (:func:`etl_hf.pack.pack_params` / :meth:`HFModel.pack_params`).
        input_ids: ``(batch_size, seq_len)`` integer token ids, as a host array
            or an :class:`etl.Tensor` (already on the executable's device for a
            device-resident run — see :func:`place_ids`).

    Returns:
        etl.Tensor: the logits, ``(batch_size, seq_len, vocab_size)`` float32,
        on the executable's device (device-resident outputs are **not** copied
        to the host — call ``.to(Device('cpu', 0))`` explicitly).

    Raises:
        TypeError: ``executable`` is not an :class:`etl.Executable`, or ``params``
            is not a mapping.
        etl_hf.HFWeightError: ``params`` is a canonical tree where the executable
            takes packed inputs (or the reverse).
        etl.DTypeError: ``input_ids`` (array or Tensor) is not integral.
        etl.ShapeError: ``input_ids`` is not 2-D or does not match the
            executable's ids spec.
    """
    if not isinstance(executable, etl.Executable):
        raise TypeError(
            "run_model expects an etl.Executable (build one with "
            f"etl_hf.build_model or HFModel.build), got {type(executable).__name__}"
        )
    if not isinstance(params, Mapping):
        raise TypeError(
            f"params must be a mapping of canonical name -> array/Tensor, got "
            f"{type(params).__name__}"
        )
    expected = _ids_spec(executable)
    if isinstance(input_ids, etl.Tensor):
        # Device-resident path: validate metadata ONLY (never touch the data)
        # and hand the very same tensor to etl.run.
        ids: object = _check_ids_tensor(input_ids, expected)
    else:
        host = np.asarray(input_ids)
        if host.ndim != 2:
            raise etl.ShapeError(
                f"input_ids must be 2-D (batch, seq_len), got shape {host.shape} "
                f"(rank {host.ndim})"
                + (
                    f"; this executable expects {tuple(expected.shape)}"
                    if expected is not None
                    else ""
                )
            )
        if expected is not None:
            want = tuple(expected.shape)
            if all(isinstance(dim, int) for dim in want) and want != host.shape:
                raise etl.ShapeError(
                    f"input_ids shape {host.shape} does not match this executable's "
                    f"(batch, seq_len) input spec {want} (expected {want[1]} tokens per "
                    f"row for a static graph; build for another length instead of "
                    f"reshaping)"
                )
        if host.dtype.kind not in "iu":
            raise etl.DTypeError(
                f"input_ids must be an integer array (any integral dtype; token ids are "
                f"cast to int32), got dtype {host.dtype}"
            )
        ids = np.ascontiguousarray(host, dtype=np.int32)
    _check_params_mode(executable, params)
    return etl.run(executable, params, ids)


# ---------------------------------------------------------------------------
# explicit device placement
# ---------------------------------------------------------------------------
def place_params(
    params: Mapping[str, np.ndarray], device: _core.Device
) -> dict[str, "etl.Tensor"]:
    """Copy every parameter onto ``device`` — explicit per-leaf placement.

    A non-cpu executable rejects host inputs at its run boundary by design
    (:class:`etl.DeviceError`), so this is the documented way to prepare weights
    for a device-resident run: one explicit ``Tensor.to(device)`` per leaf, no
    hidden move. Use the result as the ``params`` argument of
    :func:`run_model` and feed each run's outputs back as the next run's inputs
    (the documented same-device loop).

    Packed weights fit here unchanged, and the supported order is **pack on the
    host first, then place the packed tree**: ``place_params(packed, device)``
    transfers the ~4–8 packed leaves instead of one leaf per weight (~200 for
    the 124M models) — the same bytes, a fraction of the per-leaf staging calls.

    Args:
        params: canonical flat params mapping (host arrays, e.g. from
            :meth:`HFModel.params`) — or, after :meth:`HFModel.pack_params`, the
            packed tree.
        device: the target :class:`etl.core.Device` (e.g. ``Device("cuda", 0)``).

    Returns:
        dict[str, etl.Tensor]: the same keys, each value a device ``Tensor``.

    Raises:
        TypeError: ``device`` is not an :class:`etl.core.Device`, ``params`` is
            not a mapping, a key is not a string, or a value is not convertible
            to a numpy array.
    """
    device = _require_device(device, "place_params(device)")
    if not isinstance(params, Mapping):
        raise TypeError(
            f"place_params expects a mapping of canonical name -> array, got "
            f"{type(params).__name__}"
        )
    placed: dict[str, etl.Tensor] = {}
    for name, value in params.items():
        if not isinstance(name, str):
            raise TypeError(
                f"place_params: parameter names must be strings, got {name!r} of "
                f"type {type(name).__name__}"
            )
        try:
            array = np.ascontiguousarray(np.asarray(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"place_params: parameter {name!r} is not convertible to a numpy "
                f"array (got {type(value).__name__}: {exc})"
            ) from exc
        if array.dtype.kind not in "biufc":
            raise TypeError(
                f"place_params: parameter {name!r} is not a numeric array "
                f"(dtype {array.dtype}, from {type(value).__name__}); model weights "
                f"must be numeric host data"
            )
        placed[name] = _core.from_numpy(array).to(device)
    return placed


def place_ids(input_ids: object, device: _core.Device) -> "etl.Tensor":
    """Copy token ids onto ``device`` — explicit, one-step placement.

    The ids half of the device-resident recipe (its counterpart is
    :func:`place_params`): a non-cpu executable rejects *every* host input at
    the run boundary by design, ids included, so both sides must be placed
    explicitly::

        device = etl.core.Device("rocm", 0)
        exe = model.build(backend="iree", device=device)
        out = etl_hf.run_model(
            exe,
            etl_hf.place_params(model.params(), device),
            etl_hf.place_ids(ids, device),
        )
        logits = out.to(etl.core.Device("cpu", 0))   # one explicit D2H

    Host ids are cast to ``int32`` — the same single documented id coercion
    :func:`run_model` performs on a host array, so the placed tensor matches the
    graph's ``etl.int32`` ids input instead of failing dtype-strictly at the run
    boundary. An :class:`etl.Tensor` is **not** re-typed (a tensor's data is
    never silently rewritten): it is validated and moved with the explicit
    ``Tensor.to(device)``, which is a no-op when it already lives there.

    Args:
        input_ids: ``(batch_size, seq_len)`` integer token ids — a host array
            (any integral dtype) or an :class:`etl.Tensor`.
        device: the target :class:`etl.core.Device` (e.g. ``Device("cuda", 0)``).

    Returns:
        etl.Tensor: an ``int32`` ids tensor on ``device``.

    Raises:
        TypeError: ``device`` is not an :class:`etl.core.Device`.
        etl.ShapeError: ``input_ids`` is not 2-D.
        etl.DTypeError: ``input_ids`` (array or Tensor) is not integral.
    """
    device = _require_device(device, "place_ids(device)")
    if isinstance(input_ids, etl.Tensor):
        return _check_ids_tensor(input_ids).to(device)
    try:
        ids = np.ascontiguousarray(np.asarray(input_ids))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "place_ids: input_ids is not convertible to a numpy array (got "
            f"{type(input_ids).__name__}: {exc})"
        ) from exc
    if ids.ndim != 2:
        raise etl.ShapeError(
            f"place_ids: input_ids must be 2-D (batch, seq_len), got shape "
            f"{ids.shape} (rank {ids.ndim})"
        )
    if ids.dtype.kind not in "iu":
        raise etl.DTypeError(
            f"place_ids: input_ids must have an integral dtype (token ids are "
            f"integers; they are cast to int32), got dtype {ids.dtype}"
        )
    return _core.from_numpy(np.ascontiguousarray(ids, dtype=np.int32)).to(device)


# ---------------------------------------------------------------------------
# load a checkpoint directory
# ---------------------------------------------------------------------------
class HFModel:
    """A loaded HuggingFace checkpoint: config + canonical params + graph.

    The object keeps the **canonical** parameter tree (``(in_features,
    out_features)`` linears, one array per parameter) — never the raw HF state
    dict — so memory is one copy of the weights in canonical form. A tied output
    head is **not** a separate canonical entry: it aliases ``embed.weight``
    (``tie_word_embeddings=True``), so the tied tree carries one key fewer and no
    duplicated embedding array. Nothing is compiled until
    :meth:`build`/:meth:`forward` runs, and nothing moves devices unless
    :func:`place_params` is called explicitly.

    Attributes:
        config: the parsed :class:`ModelConfig` (one config = one graph).
        model_dir: the checkpoint directory (``Path``) this model came from, or
            ``None`` when it was constructed from in-memory params.
        family: the resolved ``etl_hf`` family key (``"llama"`` / ``"gpt2"``).
        seq_len: the static sequence length this model forwards at.
        batch_size: the static batch size this model forwards at.

    Packed weights are opt-in and orthogonal to the stored canonical tree: the
    model always keeps the canonical arrays, and
    :meth:`pack_params`/:meth:`build`/:meth:`forward` take
    ``pack_weights=True`` when a compiled backend cannot cope with one graph
    input per weight (see :mod:`etl_hf.pack`).
    """

    def __init__(
        self,
        config: ModelConfig,
        params: Mapping[str, np.ndarray],
        *,
        seq_len: int,
        batch_size: int = 1,
        model_dir: object = None,
        family: str | None = None,
    ) -> None:
        """Create a model from an already-canonicalised parameter tree.

        Prefer :func:`load_hf`, which reads ``config.json`` + the checkpoint and
        canonicalises the weights for you.

        Args:
            config: the parsed model config.
            params: canonical flat params mapping (the exact keys of
                ``param_specs``, ``(in_features, out_features)`` linears).
            seq_len: static sequence length; ``1 <= seq_len <=
                config.max_position_embeddings``.
            batch_size: static batch size (default 1).
            model_dir: optional checkpoint directory (kept for diagnostics and
                :meth:`__repr__`).
            family: optional explicit family key; must agree with
                ``config.family``.

        Raises:
            TypeError: ``config`` is not a :class:`ModelConfig` or ``params`` is
                not a mapping.
            HFConfigError: ``seq_len``/``batch_size`` is not a positive integer,
                ``seq_len`` exceeds ``config.max_position_embeddings``, or
                ``family`` disagrees with ``config.family``.
            HFWeightError: ``params`` does not match the family's parameter specs
                (missing/extra keys or wrong shapes) — never tolerated, since a
                wrong tree would only fail deep inside a run.
            HFUnsupportedError: the family key is not registered.
        """
        self.config: ModelConfig = _require_config(config)
        spec = _resolve_family(self.config, family)
        self._family_spec: FamilySpec = spec
        self.family: str = spec.name
        self.seq_len: int = _positive_int(seq_len, "HFModel(seq_len)")
        self.batch_size: int = _positive_int(batch_size, "HFModel(batch_size)")
        limit = self.config.max_position_embeddings
        if self.seq_len > limit:
            raise HFConfigError(
                f"HFModel(seq_len={self.seq_len}) exceeds the config's "
                f"max_position_embeddings ({limit})"
            )
        self.model_dir: Path | None = None if model_dir is None else Path(model_dir)
        self._params: dict[str, np.ndarray] = self._check_params(params)
        #: Lazily built :class:`etl_hf.pack.PackPlan` (see :meth:`pack_plan`).
        self._pack_plan_cache: PackPlan | None = None

    # -- validation ---------------------------------------------------------
    def _check_params(
        self, params: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Validate a canonical params mapping against the family's specs.

        Args:
            params: candidate canonical params.

        Returns:
            dict[str, np.ndarray]: a new dict (the caller's mapping is not
            retained, so later mutation cannot desynchronise the model).

        Raises:
            TypeError: ``params`` is not a mapping.
            HFWeightError: keys and shapes must match
                :meth:`param_specs` exactly (extra/missing keys or wrong shapes
                are always an error — no silent filtering or reshaping).
        """
        if not isinstance(params, Mapping):
            raise TypeError(
                f"params must be a mapping of canonical name -> array, got "
                f"{type(params).__name__}"
            )
        specs = self.param_specs()
        expected = set(specs)
        got = set(params)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise HFWeightError(
                f"params do not match the {self.family!r} parameter specs: "
                f"missing={missing[:8]}{'…' if len(missing) > 8 else ''}, "
                f"extra={extra[:8]}{'…' if len(extra) > 8 else ''}"
            )
        canonical: dict[str, np.ndarray] = {}
        for name, spec in specs.items():
            value = params[name]
            array = np.asarray(value)
            if array.shape != tuple(spec.shape):
                raise HFWeightError(
                    f"param {name!r} has shape {array.shape}, expected {tuple(spec.shape)}"
                )
            canonical[name] = array
        return canonical

    # -- introspection ------------------------------------------------------
    def params(self) -> dict[str, np.ndarray]:
        """Return the canonical host parameter tree.

        A tied model's tree holds ``embed.weight`` **once** (the head aliases it),
        so no array is duplicated for ``tie_word_embeddings=True``.

        Returns:
            dict[str, np.ndarray]: a fresh dict of the canonical arrays (same
            underlying buffers — pass them to :func:`place_params` for a
            device-resident run).
        """
        return dict(self._params)

    def param_specs(self, *, pack_weights: bool = False) -> dict[str, "etl.TensorSpec"]:
        """Return the ``TensorSpec`` of every **graph input** parameter.

        Args:
            pack_weights: when ``True`` return the PACKED inputs
                (:attr:`etl_hf.pack.PackPlan.specs` — a handful of concatenated
                tensors, the tree a ``pack_weights=True`` executable takes);
                ``False`` (default) returns the canonical per-weight tree.

        Returns:
            dict[str, etl.TensorSpec]: canonical name → float32 static-shape spec,
            or packed input name (``"packed.N"``) → its concatenated spec.
        """
        if _require_bool(pack_weights, "HFModel.param_specs(pack_weights)"):
            return dict(self.pack_plan().specs)
        return self._family_spec.param_specs(self.config)

    def pack_plan(self) -> PackPlan:
        """Return (and memoise) the packed-input plan of this model's weights.

        The plan is a pure function of :meth:`param_specs` (dtypes, ranks and
        trailing dims only), so it is built once per model and reused by
        :meth:`pack_params`, :meth:`param_specs` and :meth:`forward`.

        Returns:
            etl_hf.PackPlan: the plan mapping the canonical keys onto
            ``"packed.N"`` graph inputs with static slice offsets.
        """
        if self._pack_plan_cache is None:
            self._pack_plan_cache = _pack.pack_plan(
                self._family_spec.param_specs(self.config)
            )
        return self._pack_plan_cache

    def pack_params(
        self, params: Mapping[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """Concatenate canonical host params into the packed tree.

        This is the params argument a ``pack_weights=True`` executable takes: it
        is exactly ``etl_hf.pack_params(self.pack_plan(), params)``. Pack **once**
        on the host (at params/build time, never per run), then either pass the
        result to :meth:`forward` / :func:`run_model` directly or hand it to
        :func:`place_params` for a device-resident run (the supported order is
        pack → place).

        Args:
            params: canonical params mapping (default: :meth:`params`, the
                checkpoint's own weights); host arrays whose dtype/shape must
                match :meth:`param_specs`.

        Returns:
            dict[str, np.ndarray]: packed host arrays keyed by
            ``self.pack_plan().specs`` keys (``"packed.N"``), C-contiguous.

        Raises:
            TypeError: ``params`` is not a mapping.
            etl_hf.HFWeightError: a key is missing/extra, a value is not a host
                numeric array (e.g. an already-placed device ``Tensor``), or its
                dtype/shape disagrees — the message names the key.
        """
        weights = self.params() if params is None else params
        return self.pack_plan().pack(weights)

    @property
    def family_spec(self) -> FamilySpec:
        """The :class:`FamilySpec` implementing this model's architecture."""
        return self._family_spec

    def graph_fn(self, *, seq_len: int | None = None, batch_size: int | None = None) -> Callable:
        """Build (or fetch) the ``@etl.defn`` forward for one static shape.

        Args:
            seq_len: override this model's sequence length (default: its own).
            batch_size: override this model's batch size (default: its own).

        Returns:
            Callable: the ``@etl.defn`` forward ``(params, input_ids) -> logits``
            — useful for a custom ``etl.trace``/``etl.build`` staging.
        """
        return self._family_spec.graph(
            self.config,
            self.seq_len if seq_len is None else seq_len,
            self.batch_size if batch_size is None else batch_size,
        )

    # -- build / run --------------------------------------------------------
    def build(self, **kwargs: object) -> "etl.Executable":
        """Build this model's compiled executable (memoised at the build layer).

        Args:
            **kwargs: forwarded to :func:`build_model` (``backend``, ``device``,
                ``cache``, ``pack_weights``, compile options).
                ``seq_len``/``batch_size``/``family`` are fixed by this object.

        Returns:
            etl.Executable: the loaded executable for
            ``(self.config, self.seq_len, self.batch_size)``; repeated calls with
            the same arguments return the identical object. With
            ``pack_weights=True`` the executable takes :meth:`pack_params`
            output, not the canonical tree.
        """
        return build_model(
            self.config,
            self.seq_len,
            batch_size=self.batch_size,
            family=self.family,
            **kwargs,
        )

    def _default_ids(self) -> np.ndarray:
        """All-zero ``int32`` ids of this model's static shape (token 0 everywhere)."""
        return np.zeros((self.batch_size, self.seq_len), dtype=np.int32)

    def _validate_ids(self, input_ids: object) -> np.ndarray:
        """Check an explicit ``input_ids`` against this model's static shape.

        Accepts a host array or an :class:`etl.Tensor`; a Tensor is checked from
        its metadata and returned unchanged (so :meth:`forward` can hand it to
        ``etl.run`` verbatim, device-resident path included — dtype checks happen
        in :func:`run_model`).

        Args:
            input_ids: the caller's ids.

        Returns:
            np.ndarray | etl.Tensor: ``input_ids`` as given, shape-validated
            (dtype checks happen in :func:`run_model`).

        Raises:
            etl.ShapeError: the shape is not exactly
                ``(self.batch_size, self.seq_len)``; the message says how to get a
                model for another length.
        """
        ids = input_ids if isinstance(input_ids, etl.Tensor) else np.asarray(input_ids)
        shape = tuple(ids.shape)
        expected = (self.batch_size, self.seq_len)
        if shape != expected:
            raise etl.ShapeError(
                f"this HFModel was built for seq_len={self.seq_len} and "
                f"batch_size={self.batch_size} (expected input_ids shape "
                f"{expected}), got shape {shape}; shapes are static — load "
                f"another length with load_hf(model_dir, seq_len={shape[-1] if len(shape) else '?'}) "
                f"instead of reshaping"
            )
        return ids

    def forward(
        self,
        params: Mapping[str, np.ndarray] | None = None,
        input_ids: object = None,
        *,
        pack_weights: bool = False,
        **kwargs: object,
    ) -> np.ndarray:
        """Run the compiled forward pass and return host logits.

        This is the **host-facing** helper: it answers in ``numpy``, so its
        params must be host-addressable (host arrays or cpu ``Tensor``s) — a
        non-cpu params tree raises :class:`etl.DeviceError` naming the explicit
        device-resident recipe (:meth:`build` with ``device=``,
        :func:`place_params`, :func:`place_ids`, :func:`run_model`/``etl.run``),
        never a silent copy back to the host. The only move it performs is the
        one documented ``Tensor.to(Device('cpu', 0))`` on the *output* of a
        device-resident executable.

        Args:
            params: canonical params mapping (default: :meth:`params`, the
                checkpoint's own weights). Host arrays (or cpu ``Tensor``s) only
                — for a device-resident run use :meth:`build` + :func:`place_params`
                + :func:`run_model` instead. With ``pack_weights=True`` an
                already-packed tree (:meth:`pack_params` output) is used as-is; a
                canonical tree is packed once here, on the host, before the run.
            input_ids: ``(batch_size, seq_len)`` integer ids (default: all zeros
                — a shape-only smoke forward). Integral dtypes are cast to
                ``int32``; anything else raises ``etl.DTypeError``. An
                :class:`etl.Tensor` of ids is forwarded to ``etl.run`` as-is (use
                :func:`place_ids` for the device-resident path).
            pack_weights: opt-in packed weights (default ``False``). Packing
                happens once per call, on the host — nothing is re-concatenated
                per run inside etl. For repeated runs pass the packed tree
                explicitly (``params=self.pack_params()``) so it is packed once.
            **kwargs: forwarded to :meth:`build` (``backend``, ``device``,
                ``cache``, compile options).

        Returns:
            np.ndarray: float32 logits ``(batch_size, seq_len, vocab_size)`` on
            the host. A device-resident executable's output is transferred with
            one explicit ``Tensor.to(Device('cpu', 0))`` — the documented,
            visible move this host-facing helper performs.

        Raises:
            etl.ShapeError: ``input_ids`` does not match this model's static
                ``(batch_size, seq_len)``.
            etl.DTypeError: ``input_ids`` is not integral.
            etl.DeviceError: ``params`` carries a leaf on a non-cpu device (this
                helper answers in host logits; see the device-resident recipe in
                the message), a host params tree was given to a non-cpu
                executable (place it with :func:`place_params`), or the output
                could not be transferred.
            etl_hf.HFWeightError: ``params`` is a canonical tree where the
                executable takes packed inputs (or the reverse), or the tree does
                not match this model's specs.
            etl.BackendError: the resolved backend is unavailable or rejects the
                graph.
        """
        weights = self.params() if params is None else params
        if isinstance(weights, Mapping):
            _reject_device_params(weights, "HFModel.forward")
        pack_weights = _require_bool(pack_weights, "HFModel.forward(pack_weights)")
        if pack_weights and not _pack.is_packed_tree(weights):
            # Pack ONCE, on the host, before the run: the packed tree is what a
            # pack_weights=True executable takes.
            weights = self.pack_params(weights)
        ids = self._default_ids() if input_ids is None else self._validate_ids(input_ids)
        executable = self.build(pack_weights=pack_weights, **kwargs)
        logits = run_model(executable, weights, ids)
        if logits.device.kind != "cpu":
            logits = logits.to(_core.Device("cpu", 0))
        return np.asarray(logits.numpy(), dtype=np.float32)

    def reference(
        self,
        params: Mapping[str, np.ndarray] | None = None,
        input_ids: object = None,
    ) -> np.ndarray:
        """Compute the independent pure-numpy forward (the parity oracle).

        Uses the same canonical params and ids as :meth:`forward` but a
        deliberately different formulation (:mod:`etl_hf.reference`
        primitives, no ``etl`` involvement) — the way to check a compiled result
        rather than re-running it.

        Args:
            params: canonical params mapping (default: :meth:`params`).
            input_ids: ``(batch_size, seq_len)`` integer ids (default: all
                zeros, matching :meth:`forward`).

        Returns:
            np.ndarray: float32 logits ``(batch_size, seq_len, vocab_size)``.

        Raises:
            etl.ShapeError: ``input_ids`` does not match this model's static
                shape.
            etl_hf.HFWeightError: ``params`` is missing a required key or has a
                wrong shape.
        """
        weights = self.params() if params is None else params
        ids = self._default_ids() if input_ids is None else self._validate_ids(input_ids)
        if isinstance(ids, etl.Tensor):
            # The oracle is pure numpy: expose a cpu Tensor's host data
            # explicitly (a non-cpu one raises etl.DeviceError — never a hidden
            # device-to-host transfer).
            ids = ids.numpy()
        return np.asarray(
            self._family_spec.reference(self.config, weights, ids), dtype=np.float32
        )

    def __repr__(self) -> str:
        """Short, stable description (family, HF model type, static shapes)."""
        location = "-" if self.model_dir is None else str(self.model_dir)
        return (
            f"HFModel(family={self.family!r}, model_type={self.config.model_type!r}, "
            f"seq_len={self.seq_len}, batch_size={self.batch_size}, "
            f"params={len(self._params)}, model_dir={location!r})"
        )


def load_hf(
    model_dir: object,
    *,
    seq_len: int | None = None,
    family: str | None = None,
    batch_size: int = 1,
) -> HFModel:
    """Load a HuggingFace checkpoint directory into an :class:`HFModel`.

    Reads ``<model_dir>/config.json``, resolves the model family, reads the
    weights (``*.safetensors`` / sharded index / ``*.bin`` — see
    :func:`etl_hf.weights.load_checkpoint`), canonicalises them for that family
    and keeps **only** the canonical tree: the raw HF state dict is dropped as
    soon as the map is built, so a checkpoint is never held twice.

    Args:
        model_dir: the checkpoint directory (or a single checkpoint file — see
            :func:`etl_hf.weights.load_checkpoint`; a config still has to live
            next to it when a directory is given).
        seq_len: static sequence length for the graph. ``None`` (default) means
            ``min(cfg.max_position_embeddings, 32)`` — a small, instant smoke
            length that still exercises every positional path.
        family: optional explicit family key; when given it must agree with the
            config's own family (never silently reinterpreted).
        batch_size: static batch size (default 1).

    Returns:
        HFModel: the loaded model; nothing is compiled until
        :meth:`HFModel.build`/:meth:`HFModel.forward` runs.

    Raises:
        etl_hf.HFConfigError: ``config.json`` is missing/malformed/unsupported,
            ``seq_len``/``batch_size`` is invalid, ``seq_len`` exceeds
            ``max_position_embeddings``, or ``family`` disagrees with the config.
        etl_hf.HFWeightError: the checkpoint cannot be read or does not match the
            config (missing/extra/mis-shaped keys).
        etl_hf.HFUnsupportedError: unknown family, or a checkpoint/config outside
            the v1 support matrix (e.g. biases on a llama build).
        TypeError: ``model_dir`` is not path-like.
    """
    try:
        directory = Path(model_dir)
    except TypeError as exc:
        raise TypeError(
            f"load_hf(model_dir=...) must be a path-like object, got "
            f"{model_dir!r} of type {type(model_dir).__name__}"
        ) from exc
    cfg = ModelConfig.from_dir(directory)
    spec = _resolve_family(cfg, family)
    if seq_len is None:
        seq_len = min(cfg.max_position_embeddings, DEFAULT_SEQ_LEN)
    raw = _weights.load_checkpoint(directory)
    params = spec.canonicalize_weights(cfg, raw)
    # Drop the raw HF state dict: only the canonical tree is retained, so the
    # checkpoint is never held twice (and per-layer transposes are freed here).
    del raw
    return HFModel(
        cfg,
        params,
        seq_len=seq_len,
        batch_size=batch_size,
        model_dir=directory,
        family=spec.name,
    )
