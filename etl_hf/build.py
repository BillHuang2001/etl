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
``"iree"`` + ``["llvm-cpu"]``    ``iree_compile_args`` injected unless passed
===============================  =============================================

Both injected defaults are *measured* pitfalls rather than taste:

* IREE's default ``--iree-llvmcpu-target-cpu=generic`` target is 12–19× slower
  than ``host`` for these graphs, hence :data:`DEFAULT_LLVM_CPU_COMPILE_ARGS`.
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
boundary by design, so callers must place weights explicitly with
:func:`place_params` first (one ``Tensor.to(device)`` per leaf — never a hidden
move) and then drive the documented same-device loop.

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
from etl_hf import weights as _weights
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.families import FamilySpec

__all__ = [
    "DEFAULT_LLVM_CPU_COMPILE_ARGS",
    "DEFAULT_SEQ_LEN",
    "HFModel",
    "build_model",
    "load_hf",
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

#: The backend name whose options are returned untouched (the numpy interpreter
#: documents ignoring compile options, so etl_hf injects nothing for it).
_HOST_BACKEND: str = "numpy"

#: Bound of the module-level ``@etl.defn`` forward memo. Small on purpose: the
#: entries are static-shape graphs (one per config/seq_len/batch), and the
#: compiled artifacts they lead to live in etl's own ``CompileCache``.
_GRAPH_CACHE_MAX = 64

_GRAPH_CACHE: "OrderedDict[tuple, Callable]" = OrderedDict()
_GRAPH_CACHE_LOCK = threading.Lock()


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
    * any other backend → ``opt_level="O3"`` injected unless the caller passed
      ``opt_level``.
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

    resolved.setdefault("opt_level", DEFAULT_OPT_LEVEL)
    if name == "iree":
        targets = resolved.get("target_backends")
        if targets is None:
            targets = _env_list("ETL_TARGET_BACKENDS")
        if targets is None:
            targets = _targets_from_device(device)
        elif isinstance(targets, str):
            targets = [targets]
        if list(targets) == ["llvm-cpu"]:
            resolved.setdefault(
                "iree_compile_args", list(DEFAULT_LLVM_CPU_COMPILE_ARGS)
            )
    return name, resolved


# ---------------------------------------------------------------------------
# the memoised @etl.defn forward
# ---------------------------------------------------------------------------
def _forward_fn(
    spec: FamilySpec, cfg: ModelConfig, seq_len: int, batch_size: int
) -> Callable:
    """Return the memoised ``@etl.defn`` forward for one static specialisation.

    The graph depends only on ``(family, config, seq_len, batch_size)`` — never
    on weights or on a backend — so a bounded LRU of forward objects is safe and
    makes ``id(fn)`` (``etl.build_cached``'s identity component) stable across
    ``build_model`` calls. Entries are built outside the lock (tracing is
    expensive) and first-insert-wins, so concurrent callers still observe a
    single object per key.

    Args:
        spec: the resolved family.
        cfg: the static config.
        seq_len: static sequence length.
        batch_size: static batch size.

    Returns:
        Callable: the ``@etl.defn`` forward ``(params, input_ids) -> logits``.
    """
    key = (spec.name, cfg.digest, seq_len, batch_size)
    with _GRAPH_CACHE_LOCK:
        cached = _GRAPH_CACHE.get(key)
        if cached is not None:
            _GRAPH_CACHE.move_to_end(key)
            return cached
    forward = spec.graph(cfg, seq_len, batch_size)
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


def build_model(
    cfg: ModelConfig,
    seq_len: int,
    *,
    batch_size: int = 1,
    backend: object = None,
    device: object = None,
    cache: object = None,
    family: str | None = None,
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
        **options: compile options forwarded to ``etl.build_cached`` (and hence
            to both ``lower`` and ``compile`` — that is what makes
            ``iree_compile_args`` effective). Explicit values always beat the
            injected defaults.

    Returns:
        etl.Executable: the loaded executable; call it with a params mapping and
        integer ``input_ids`` through :func:`run_model` / ``etl.run``.

    Raises:
        HFConfigError: ``seq_len``/``batch_size`` is not a positive integer,
            ``seq_len`` exceeds ``cfg.max_position_embeddings``, ``family``
            disagrees with ``cfg.family``, or the backend/device argument is
            malformed.
        HFUnsupportedError: unknown ``family``, unsupported ``cfg.hidden_act``,
            or an erf-based activation on a compiled backend (raised before any
            tracing).
        TypeError: ``cfg`` is not a :class:`ModelConfig`.
        etl.BackendError: the resolved backend is unavailable or rejects the
            graph (propagated verbatim — never a silent fallback).
    """
    cfg = _require_config(cfg)
    seq_len = _positive_int(seq_len, "build_model(seq_len)")
    batch_size = _positive_int(batch_size, "build_model(batch_size)")
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
    forward = _forward_fn(spec, cfg, seq_len, batch_size)
    return etl.build_cached(
        forward,
        spec.param_specs(cfg),
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


def run_model(
    executable: "etl.Executable",
    params: Mapping[str, np.ndarray],
    input_ids: np.ndarray,
) -> "etl.Tensor":
    """Run a built model executable on one params tree and one id batch.

    Thin, explicit validation followed by ``etl.run``:

    * ``input_ids`` must be an integer array (any integral dtype) — the **one**
      documented host-side coercion is casting token ids to ``int32`` (ids are
      integral by definition; etl's run boundary is dtype-strict). Non-integral
      dtypes raise :class:`etl.DTypeError`.
    * ``input_ids`` must be 2-D and match the executable's ``input_ids`` spec
      exactly (:class:`etl.ShapeError` naming expected vs got) whenever that
      spec is available.
    * ``params`` is passed through **untouched** — etl validates keys, shapes and
      dtypes and reports the first mismatching pytree path. Nothing is filtered
      or renamed here.

    Args:
        executable: an executable whose signature is ``(params, input_ids)``,
            e.g. the result of :func:`build_model`.
        params: canonical flat params mapping (:func:`FamilySpec.param_specs`
            keys) of host arrays or placed :class:`etl.Tensor` leaves.
        input_ids: ``(batch_size, seq_len)`` integer token ids.

    Returns:
        etl.Tensor: the logits, ``(batch_size, seq_len, vocab_size)`` float32,
        on the executable's device (device-resident outputs are **not** copied
        to the host — call ``.to(Device('cpu', 0))`` explicitly).

    Raises:
        TypeError: ``executable`` is not an :class:`etl.Executable`, or ``params``
            is not a mapping.
        etl.DTypeError: ``input_ids`` is not integral.
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
    ids = np.asarray(input_ids)
    expected = _ids_spec(executable)
    if ids.ndim != 2:
        raise etl.ShapeError(
            f"input_ids must be 2-D (batch, seq_len), got shape {ids.shape} "
            f"(rank {ids.ndim})"
            + (f"; this executable expects {tuple(expected.shape)}" if expected else "")
        )
    if expected is not None:
        want = tuple(expected.shape)
        if all(isinstance(dim, int) for dim in want) and want != ids.shape:
            raise etl.ShapeError(
                f"input_ids shape {ids.shape} does not match this executable's "
                f"(batch, seq_len) input spec {want} (expected {want[1]} tokens per "
                f"row for a static graph; build for another length instead of "
                f"reshaping)"
            )
    if ids.dtype.kind not in "iu":
        raise etl.DTypeError(
            f"input_ids must be an integer array (any integral dtype; token ids are "
            f"cast to int32), got dtype {ids.dtype}"
        )
    return etl.run(
        executable, params, np.ascontiguousarray(ids, dtype=np.int32)
    )


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

    Args:
        params: canonical flat params mapping (host arrays, e.g. from
            :meth:`HFModel.params`).
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


# ---------------------------------------------------------------------------
# load a checkpoint directory
# ---------------------------------------------------------------------------
class HFModel:
    """A loaded HuggingFace checkpoint: config + canonical params + graph.

    The object keeps the **canonical** parameter tree (``(in_features,
    out_features)`` linears, tied head resolved) — never the raw HF state dict —
    so memory is one copy of the weights in canonical form. Nothing is compiled
    until :meth:`build`/:meth:`forward` runs, and nothing moves devices unless
    :func:`place_params` is called explicitly.

    Attributes:
        config: the parsed :class:`ModelConfig` (one config = one graph).
        model_dir: the checkpoint directory (``Path``) this model came from, or
            ``None`` when it was constructed from in-memory params.
        family: the resolved ``etl_hf`` family key (``"llama"`` / ``"gpt2"``).
        seq_len: the static sequence length this model forwards at.
        batch_size: the static batch size this model forwards at.
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

        Returns:
            dict[str, np.ndarray]: a fresh dict of the canonical arrays (same
            underlying buffers — pass them to :func:`place_params` for a
            device-resident run).
        """
        return dict(self._params)

    def param_specs(self) -> dict[str, "etl.TensorSpec"]:
        """Return the ``TensorSpec`` of every canonical parameter.

        Returns:
            dict[str, etl.TensorSpec]: canonical name → float32 static-shape spec
            (the weights are graph *inputs*, so this is the build-time tree).
        """
        return self._family_spec.param_specs(self.config)

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
                ``cache``, compile options). ``seq_len``/``batch_size``/
                ``family`` are fixed by this object.

        Returns:
            etl.Executable: the loaded executable for
            ``(self.config, self.seq_len, self.batch_size)``; repeated calls with
            the same arguments return the identical object.
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
        """Check an explicit ``input_ids`` array against this model's static shape.

        Args:
            input_ids: the caller's ids.

        Returns:
            np.ndarray: ``input_ids`` as an array (dtype checks happen in
            :func:`run_model`).

        Raises:
            etl.ShapeError: the shape is not exactly
                ``(self.batch_size, self.seq_len)``; the message says how to get a
                model for another length.
        """
        ids = np.asarray(input_ids)
        expected = (self.batch_size, self.seq_len)
        if ids.shape != expected:
            raise etl.ShapeError(
                f"this HFModel was built for seq_len={self.seq_len} and "
                f"batch_size={self.batch_size} (expected input_ids shape "
                f"{expected}), got shape {ids.shape}; shapes are static — load "
                f"another length with load_hf(model_dir, seq_len={ids.shape[-1] if ids.ndim else '?'}) "
                f"instead of reshaping"
            )
        return ids

    def forward(
        self,
        params: Mapping[str, np.ndarray] | None = None,
        input_ids: object = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Run the compiled forward pass and return host logits.

        Args:
            params: canonical params mapping (default: :meth:`params`, the
                checkpoint's own weights). Pass host arrays for a cpu executable
                or :func:`place_params` output for a device-resident one.
            input_ids: ``(batch_size, seq_len)`` integer ids (default: all zeros
                — a shape-only smoke forward). Integral dtypes are cast to
                ``int32``; anything else raises ``etl.DTypeError``.
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
            etl.DeviceError: a host params tree was given to a non-cpu executable
                (place it with :func:`place_params`), or the output could not be
                transferred.
            etl.BackendError: the resolved backend is unavailable or rejects the
                graph.
        """
        weights = self.params() if params is None else params
        ids = self._default_ids() if input_ids is None else self._validate_ids(input_ids)
        executable = self.build(**kwargs)
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
