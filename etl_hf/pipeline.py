"""The **staging** half: build → run → predict, plus explicit device placement.

This module takes a compute function (a ``@etl.defn`` forward from
:func:`etl_hf.forward_fn`) plus a static config and stages it through the
explicit etl pipeline, and it is where the *measured* backend-default matrix
lives (once):

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

Every injected default is a *measured* pitfall rather than taste — each one is
documented beside its constant (see :data:`DEFAULT_LLVM_CPU_COMPILE_ARGS`,
:data:`DEFAULT_ROCM_OPT_LEVEL`, and the compile-stage note on
``iree_compile_args``: it must reach ``etl.compile``, which the one-call
``etl.build_cached`` composition guarantees, while passing it to ``etl.lower``
alone is silently ineffective).

Memoisation: the graph is a pure function of ``(family, config, seq_len,
batch_size, pack_weights)`` — :func:`etl_hf.forward_fn` guarantees a stable
object per specialisation — so :func:`build` hands ``etl.build_cached`` a
**stable** ``id(fwd)``; that is exactly what makes two identical :func:`build`
calls return the identical ``Executable`` object instead of recompiling.

**Deliberate deviation from an earlier CONTEXT draft:** the cache key does
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
loop. :func:`run` accepts ids either as a host integer array or as an
:class:`etl.Tensor`, and forwards a ``Tensor`` **verbatim** (no host copy, no
dtype cast, no device move), so a device-resident ids tensor keeps host
round-trips at zero. :func:`predict` is the *host-facing* shorthand and refuses
device params with an error naming that recipe rather than pulling data back to
the host behind the caller's back.

Packed weights (``pack_weights=True``, opt-in): the graph takes a handful of
concatenated params inputs (:mod:`etl_hf.pack`) instead of one per canonical
weight and unpacks them in-graph with static ``etl.slice`` ops. It exists for one
measured reason — IREE's Python runtime SIGSEGVs above ~161 *consumed* inputs and
the 124M ``gpt2`` declares 197 — and it changes nothing about the arithmetic:
packing happens once on the host (:func:`etl_hf.pack.pack_params`), never per
run. Unpacked stays the default, so no existing caller changes behaviour.

Import-time constraint: only ``numpy``, ``etl`` and sibling ``etl_hf`` modules
are imported here. Backend adapters (iree / xla / tvm) are resolved lazily by
``etl`` itself, so ``import etl_hf`` never imports them.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import numpy as np

import etl
from etl import core as _core
from etl_hf import ops as _ops
from etl_hf import pack as _pack
from etl_hf import program as _program
from etl_hf._validate import positive_int, require_bool, require_config, require_device
from etl_hf.config import ModelConfig
from etl_hf.errors import HFConfigError, HFUnsupportedError, HFWeightError
from etl_hf.load import LoadedModel
from etl_hf.program import SPECIALISATION_ATTR, forward_fn, param_specs, resolve_seq_len

__all__ = [
    "DEFAULT_LLVM_CPU_COMPILE_ARGS",
    "DEFAULT_OPT_LEVEL",
    "DEFAULT_ROCM_OPT_LEVEL",
    "build",
    "place_ids",
    "place_params",
    "predict",
    "resolve_build_options",
    "run",
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


# ---------------------------------------------------------------------------
# backend / option resolution
# ---------------------------------------------------------------------------
def _resolve_backend_name(backend: object) -> str | None:
    """Normalise ``backend`` (``None`` / name / object with ``.name``) to a name.

    Args:
        backend: ``None``, a backend name, or an object exposing a string
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
    """Parse a comma-separated environment variable.

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
        list[str]: ``["llvm-cpu"]`` (cpu/absent), ``["cuda"]``, ``["rocm"]``
        (rocm/hip — ``hip`` is an accepted spelling), else ``[device.kind]``.
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
        list[str]: ``options["target_backends"]`` (a bare string is a
        one-element set), else ``ETL_TARGET_BACKENDS``, else
        :func:`_targets_from_device`.
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

    etl resolves that variable itself at compile time, so when it is set etl_hf
    injects **nothing** and the environment wins over its own default.

    Returns:
        bool: ``True`` when ``ETL_OPT_LEVEL`` is set and non-blank.
    """
    return bool((os.environ.get("ETL_OPT_LEVEL") or "").strip())


def _default_opt_level(targets: list[str] | None) -> str:
    """etl_hf's ``opt_level`` default for a compiled backend.

    Args:
        targets: the effective IREE targets, or ``None`` for a non-IREE backend.

    Returns:
        str: :data:`DEFAULT_ROCM_OPT_LEVEL` (``"O1"``) when every target is
        rocm/hip (the codegen-safe level for that HAL), else
        :data:`DEFAULT_OPT_LEVEL` (``"O3"``) — llvm-cpu, cuda, xla, tvm and any
        mixed/unknown target set are unchanged.
    """
    if targets and all(kind in ("rocm", "hip") for kind in targets):
        return DEFAULT_ROCM_OPT_LEVEL
    return DEFAULT_OPT_LEVEL


def resolve_build_options(
    backend: object,
    device: object,
    options: Mapping[str, object],
) -> tuple[str, dict]:
    """Resolve the default backend and the injected compile options.

    This is the *one* implementation of the documented default matrix;
    :func:`build` and any caller that needs to know what would be built use it,
    so the defaults can never drift between call sites.

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
        require_device(device, "device")
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


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _specialisation_value(
    meta: Mapping[str, object] | None,
    name: str,
    value: object,
    normalise: Callable[[object], object],
) -> object:
    """Resolve one static specialisation value against the forward's metadata.

    Args:
        meta: the forward's stamped specialisation dict, or ``None`` when the
            forward carries none (an unstamped/exotic object).
        name: the metadata field being resolved (``"seq_len"``,
            ``"batch_size"`` or ``"pack_weights"``).
        value: the caller's explicit value, or ``None`` to take the forward's.
        normalise: validator/coercer for an explicit value.

    Returns:
        object: the resolved value, or ``None`` when neither the caller nor the
        forward supplies one (the caller then applies its own default).

    Raises:
        HFConfigError: an explicit ``value`` disagrees with the forward's — one
            graph per specialisation, so a silently mismatched executable is
            never produced.
        TypeError: ``value`` is not a bool for ``pack_weights``.
    """
    if meta is None:
        return None if value is None else normalise(value)
    want = meta[name]
    if value is None:
        return want
    value = normalise(value)
    if value != want:
        raise HFConfigError(
            f"etl_hf.build({name}={value!r}) disagrees with the forward it was "
            f"given: that forward came from etl_hf.forward_fn with "
            f"{name}={want!r} (family={meta.get('family')!r}; one graph per "
            f"(config, seq_len, batch_size, pack_weights) — shapes and packing "
            f"are static). Either drop {name}= from etl_hf.build and let the "
            f"forward decide, or build a matching forward: "
            f"etl_hf.forward_fn(config, seq_len=..., batch_size=..., "
            f"pack_weights=...) with {name}={value!r}."
        )
    return want


def build(
    fwd: Callable,
    config: ModelConfig,
    *,
    seq_len: int | None = None,
    batch_size: int | None = None,
    backend: object = None,
    device: object = None,
    cache: object = None,
    pack_weights: bool | None = None,
    family: str | None = None,
    **options: object,
) -> "etl.Executable":
    """Build (trace → lower → compile → load) the executable of one forward.

    The whole forward pass becomes **one** compiled graph whose weights are
    ordinary graph inputs (:func:`etl_hf.param_specs`) and whose ``input_ids`` is
    a ``(batch_size, seq_len)`` ``etl.int32`` input. All shapes are static: a
    different ``seq_len``/``batch_size`` is a different graph.

    ``seq_len``/``batch_size``/``pack_weights`` normally come from ``fwd``
    itself: :func:`etl_hf.forward_fn` stamps its static specialisation on the
    forward, so ``build(forward_fn(config, seq_len=8), config)`` builds the id
    spec that graph actually has. An explicit value that *disagrees* with the
    forward is an error (never a silently mismatched ids spec); an omitted one is
    taken from the forward, falling back to
    :func:`etl_hf.resolve_seq_len`/1/``False`` for an unstamped forward.

    Calls are memoised: the graph is a pure function of ``(family, config,
    seq_len, batch_size, pack_weights)`` — :func:`etl_hf.forward_fn` returns the
    same object for the same specialisation — so two identical calls return the
    **same** ``Executable`` object and no second compile happens. Backend/device/
    options participate in etl's cache key, so changing any of them is a new
    artifact. Weight identity deliberately does **not** (see the module
    docstring).

    Args:
        fwd: the ``@etl.defn`` forward to stage — the object returned by
            :func:`etl_hf.forward_fn` (its specialisation is read from it).
        config: the model config — a static specialisation value (one config =
            one graph).
        seq_len: static sequence length; ``1 <= seq_len <=
            config.max_position_embeddings``. ``None`` (default) takes the
            forward's own length.
        batch_size: static batch size. ``None`` (default) takes the forward's own
            batch size (1 when it has none).
        backend: ``None`` (resolve the documented default matrix, see
            :func:`resolve_build_options`), a backend name (``"numpy"``,
            ``"iree"``, ``"xla"``, ``"tvm"`` …) or an etl backend instance.
        device: ``None`` or an :class:`etl.core.Device` the executable should run
            on.
        cache: an :class:`etl.CompileCache` to memoise into, or ``None`` for the
            process-wide default cache.
        pack_weights: opt-in **packed weights**. ``True`` means the forward takes
            a handful of concatenated params inputs
            (:attr:`etl_hf.pack.PackPlan.specs`) instead of one input per
            canonical weight; it must match the forward (``None``, the default,
            takes the forward's own mode). This is what makes the 124M-model case
            runnable on IREE (its runtime SIGSEGVs above ~161 consumed inputs;
            ``gpt2`` declares 197 unpacked). Run it with the packed tree —
            :func:`etl_hf.pack_params` — never with the canonical tree.
        family: optional explicit family key; when given it must agree with
            ``config.family``.
        **options: compile options forwarded to ``etl.build_cached`` (and hence
            to both ``lower`` and ``compile`` — that is what makes
            ``iree_compile_args`` effective). Explicit values always beat the
            injected defaults.

    Returns:
        etl.Executable: the loaded executable; call it with a params mapping and
        integer ``input_ids`` through :func:`run` / ``etl.run``. For
        ``pack_weights=True`` that mapping is the packed tree.

    Raises:
        HFConfigError: ``seq_len``/``batch_size`` is not a positive integer,
            ``seq_len`` exceeds ``config.max_position_embeddings``, ``family``
            disagrees with ``config.family``, or an explicit
            ``seq_len``/``batch_size``/``pack_weights`` disagrees with the
            forward's own specialisation.
        HFUnsupportedError: unknown ``family``, unsupported ``config.hidden_act``,
            an erf-based activation on a compiled backend (raised before any
            tracing), or a non-static config shape that cannot be packed.
        TypeError: ``config`` is not a :class:`~etl_hf.config.ModelConfig`, or
            ``fwd`` is not ``@etl.defn``-marked (build one with
            :func:`etl_hf.forward_fn`), or ``pack_weights`` is not a bool.
        etl.BackendError: the resolved backend is unavailable or rejects the
            graph (propagated verbatim — never a silent fallback).
    """
    cfg = require_config(config)
    if not getattr(fwd, "__etl_defn__", False):
        raise TypeError(
            f"etl_hf.build expects an @etl.defn-marked forward (the graph "
            f"definition), got {type(fwd).__name__}; build one with "
            f"etl_hf.forward_fn(config, seq_len=...). (An already-traced Graph or "
            f"an Executable is past this stage — use etl's explicit "
            f"lower/compile/load pipeline, or etl_hf.run for an executable.)"
        )
    meta = getattr(fwd, SPECIALISATION_ATTR, None)
    if not isinstance(meta, Mapping) or "seq_len" not in meta:
        meta = None
    resolved_seq_len = resolve_seq_len(
        cfg,
        _specialisation_value(
            meta, "seq_len", seq_len, lambda v: positive_int(v, "etl_hf.build(seq_len)")
        ),
    )
    resolved_batch_size = _specialisation_value(
        meta,
        "batch_size",
        batch_size,
        lambda v: positive_int(v, "etl_hf.build(batch_size)"),
    )
    if resolved_batch_size is None:
        resolved_batch_size = 1
    resolved_packed = _specialisation_value(
        meta,
        "pack_weights",
        pack_weights,
        lambda v: require_bool(v, "etl_hf.build(pack_weights)"),
    )
    if resolved_packed is None:
        resolved_packed = False
    resolved_backend, resolved_options = resolve_build_options(
        backend, device, options
    )
    _check_activation(cfg, resolved_backend)
    params_specs = param_specs(cfg, family=family, pack_weights=resolved_packed)
    return etl.build_cached(
        fwd,
        params_specs,
        etl.TensorSpec((resolved_batch_size, resolved_seq_len), etl.int32),
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
    record's last entry; an executable from some other graph returns ``None`` and
    the caller skips the spec-shape check rather than guessing.

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
    the static shape vs the ids spec all come from ``Tensor.dtype``/
    ``Tensor.shape``. Returned unchanged — no host copy, no cast, no move.

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

    etl records the input *structure* (``signature.input_tree``) beside the flat
    ``input_specs``; the params dict is its first child, whose ``node_data``
    carries the sorted key list — canonical weight names, or the packed
    ``packed.N`` names. ``None`` means "unknown" and the caller skips the check.

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
            f"packed = etl_hf.pack_params(etl_hf.pack_plan("
            f"etl_hf.param_specs(config)), params), and pass the packed tree; to "
            f"run the canonical tree instead, build with pack_weights=False."
        )
    if got_packed and not expected_packed:
        raise HFWeightError(
            f"this executable was built with pack_weights=False, so it takes the "
            f"{len(expected)} canonical weight inputs "
            f"({sorted(expected_keys)[:8]}), but it was given packed keys "
            f"({sorted(got)[:8]}). Pass the canonical params tree, or rebuild "
            f"with pack_weights=True together with "
            f"etl_hf.pack_params(etl_hf.pack_plan(etl_hf.param_specs(config)), "
            f"params)."
        )


def _reject_device_params(params: Mapping[str, object], what: str) -> None:
    """Reject device-resident params handed to a **host-facing** helper.

    :func:`predict` answers in host ``numpy``, so a placed tree would force an
    implicit device-to-host copy — which this package never performs. A cpu
    ``Tensor`` (including an ``iree`` llvm-cpu payload) is fine: it is
    host-addressable by etl's own rules.

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
            f"exe = etl_hf.build(etl_hf.forward_fn(loaded.config, seq_len=...), "
            f"loaded.config, backend='iree', device=device); "
            f"out = etl_hf.run(exe, "
            f"etl_hf.place_params(loaded.params, device), "
            f"etl_hf.place_ids(input_ids, device)); "
            f"logits = out.to(etl.core.Device('cpu', 0)) "
            f"(or drive etl.run directly with the same placed inputs, feeding "
            f"each run's outputs back as the next run's inputs). To keep using "
            f"{what}, pass the host params tree (or cpu Tensors)."
        )


def run(
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
      ``etl_hf.build(fwd, config, device=Device('rocm', 0))`` plus
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
            e.g. the result of :func:`build`.
        params: canonical flat params mapping (:func:`etl_hf.param_specs` keys)
            of host arrays or placed :class:`etl.Tensor` leaves — or, for a
            ``pack_weights=True`` executable, the packed tree it was built for
            (:func:`etl_hf.pack_params`).
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
            "etl_hf.run expects an etl.Executable (build one with etl_hf.build), "
            f"got {type(executable).__name__}"
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
    hidden move. Use the result as the ``params`` argument of :func:`run` and
    feed each run's outputs back as the next run's inputs (the documented
    same-device loop).

    Packed weights fit here unchanged, and the supported order is **pack on the
    host first, then place the packed tree**: ``place_params(packed, device)``
    transfers the ~4–8 packed leaves instead of one leaf per weight (~200 for
    the 124M models) — the same bytes, a fraction of the per-leaf staging calls.

    Args:
        params: canonical flat params mapping (host arrays, e.g.
            ``etl_hf.load_hf(model_dir).params``) — or, after
            :func:`etl_hf.pack_params`, the packed tree.
        device: the target :class:`etl.core.Device` (e.g. ``Device("cuda", 0)``).

    Returns:
        dict[str, etl.Tensor]: the same keys, each value a device ``Tensor``.

    Raises:
        TypeError: ``device`` is not an :class:`etl.core.Device`, ``params`` is
            not a mapping, a key is not a string, or a value is not convertible
            to a numpy array.
    """
    device = require_device(device, "place_params(device)")
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
    :func:`place_params`): a non-cpu executable rejects *every* host input at the
    run boundary by design, ids included, so both sides must be placed
    explicitly — ``etl_hf.run(exe, etl_hf.place_params(params, device),
    etl_hf.place_ids(ids, device))``.

    Host ids are cast to ``int32`` — the same single documented id coercion
    :func:`run` performs on a host array, so the placed tensor matches the
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
    device = require_device(device, "place_ids(device)")
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
# the host-facing shorthand
# ---------------------------------------------------------------------------
def predict(
    loaded: LoadedModel,
    input_ids: object = None,
    *,
    params: Mapping[str, np.ndarray] | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    backend: object = None,
    device: object = None,
    cache: object = None,
    pack_weights: bool | None = None,
    family: str | None = None,
    **options: object,
) -> np.ndarray:
    """Run the whole forward and return host logits — the one host-facing shorthand.

    A transparent composition of explicit primitives — nothing else happens
    here::

        predict(loaded, ids, seq_len=8) ~=
            exe = build(forward_fn(loaded.config, seq_len=8), loaded.config, ...)
            logits = run(exe, loaded.params, ids)
            logits.to(Device("cpu", 0)).numpy()

    It answers in ``numpy``, so its params must be host-addressable (host arrays
    or cpu ``Tensor``s) — a non-cpu params tree raises :class:`etl.DeviceError`
    naming the explicit device-resident recipe (:func:`build` with ``device=``,
    :func:`place_params`, :func:`place_ids`, :func:`run`), never a silent copy
    back to the host. The only move it performs is the one documented
    ``Tensor.to(Device('cpu', 0))`` on the *output* of a device-resident
    executable.

    Args:
        loaded: the :class:`~etl_hf.load.LoadedModel` from
            :func:`etl_hf.load_hf` (config + canonical params, plain data).
        input_ids: ``(batch_size, seq_len)`` integer ids, or ``None`` (default)
            for all-zero ids of the static shape. Integral dtypes are cast to
            ``int32`` (``etl.DTypeError`` otherwise); an :class:`etl.Tensor` is
            forwarded to :func:`run` as-is (see :func:`place_ids`). Shapes are
            static, so ids must match ``(batch_size, seq_len)`` exactly — build
            another length instead of reshaping.
        params: the params to run (default: ``loaded.params``). Host arrays (or
            cpu ``Tensor``s) only — for a device-resident run use :func:`build` +
            :func:`place_params` + :func:`run` instead. With
            ``pack_weights=True`` an already-packed tree
            (:func:`etl_hf.pack_params` output) is used as-is; a canonical tree
            is packed once here, on the host, before the run.
        seq_len: static sequence length, or ``None`` for
            :func:`etl_hf.resolve_seq_len`'s default
            ``min(max_position_embeddings, 32)``.
        batch_size: static batch size, or ``None`` for 1.
        backend: ``None`` (default matrix — see :func:`resolve_build_options`), a
            backend name, or an etl backend instance.
        device: ``None`` or an :class:`etl.core.Device` to build/run on.
        cache: an :class:`etl.CompileCache` to memoise the build into, or
            ``None`` for the process-wide default cache.
        pack_weights: opt-in packed weights (default ``False``). Packing happens
            once per call, on the host — nothing is re-concatenated per run
            inside etl. For repeated runs pass the packed tree explicitly
            (``params=etl_hf.pack_params(plan, loaded.params)``) so it is packed
            once.
        family: optional explicit family key; when given it must agree with
            ``loaded.config.family``.
        **options: compile options forwarded to :func:`build` (and hence to both
            ``lower`` and ``compile``).

    Returns:
        np.ndarray: float32 logits ``(batch_size, seq_len, vocab_size)`` on the
        host. A device-resident executable's output is transferred with one
        explicit ``Tensor.to(Device('cpu', 0))`` — the documented, visible move
        this host-facing helper performs.

    Raises:
        TypeError: ``loaded`` is not a :class:`~etl_hf.load.LoadedModel` (use
            :func:`etl_hf.load_hf`), or ``pack_weights`` is not a bool.
        etl.ShapeError: ``input_ids`` does not match the static
            ``(batch_size, seq_len)`` shape of this call.
        etl.DTypeError: ``input_ids`` is not integral.
        etl.DeviceError: ``params`` carries a leaf on a non-cpu device (this
            helper answers in host logits; see the device-resident recipe in the
            message), a host params tree was given to a non-cpu executable (place
            it with :func:`place_params`), or the output could not be
            transferred.
        etl_hf.HFWeightError: ``params`` is a canonical tree where the executable
            takes packed inputs (or the reverse), or the tree does not match the
            config's specs.
        etl.BackendError: the resolved backend is unavailable or rejects the
            graph.
    """
    if not isinstance(loaded, LoadedModel):
        raise TypeError(
            "etl_hf.predict expects an etl_hf.LoadedModel (the result of "
            f"etl_hf.load_hf(model_dir)), got {type(loaded).__name__}; for a "
            "config/params pair you assembled yourself, stage the graph "
            "explicitly: etl_hf.run(etl_hf.build(etl_hf.forward_fn(config, "
            "seq_len=...), config), params, input_ids)"
        )
    cfg = loaded.config
    resolved_packed = (
        False if pack_weights is None else require_bool(pack_weights, "etl_hf.predict(pack_weights)")
    )
    weights = loaded.params if params is None else params
    if isinstance(weights, Mapping):
        _reject_device_params(weights, "etl_hf.predict")
    if resolved_packed and not _pack.is_packed_tree(weights):
        # Pack ONCE, on the host, before the run: the packed tree is what a
        # pack_weights=True executable takes (memoised plan, deterministic).
        weights = _pack.pack_params(_program._pack_plan_for(cfg, family), weights)
    resolved_seq_len = resolve_seq_len(cfg, seq_len)
    resolved_batch_size = (
        1 if batch_size is None else positive_int(batch_size, "etl_hf.predict(batch_size)")
    )
    if input_ids is None:
        ids: object = np.zeros(
            (resolved_batch_size, resolved_seq_len), dtype=np.int32
        )
    else:
        ids = input_ids if isinstance(input_ids, etl.Tensor) else np.asarray(input_ids)
        shape = tuple(ids.shape)
        expected = (resolved_batch_size, resolved_seq_len)
        if shape != expected:
            raise etl.ShapeError(
                f"this predict call was built for seq_len={resolved_seq_len} and "
                f"batch_size={resolved_batch_size} (expected input_ids shape "
                f"{expected}), got shape {shape}; shapes are static — build "
                f"another length with etl_hf.forward_fn(config, "
                f"seq_len={shape[-1] if shape else '?'}) (and pass seq_len= to "
                f"etl_hf.predict), or load the ids you have, instead of reshaping"
            )
    fwd = forward_fn(
        cfg,
        seq_len=resolved_seq_len,
        batch_size=resolved_batch_size,
        family=family,
        pack_weights=resolved_packed,
    )
    executable = build(
        fwd,
        cfg,
        seq_len=resolved_seq_len,
        batch_size=resolved_batch_size,
        backend=backend,
        device=device,
        cache=cache,
        pack_weights=resolved_packed,
        family=family,
        **options,
    )
    logits = run(executable, weights, ids)
    if logits.device.kind != "cpu":
        logits = logits.to(_core.Device("cpu", 0))
    return np.asarray(logits.numpy(), dtype=np.float32)
