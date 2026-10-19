#!/usr/bin/env python3
"""Runtime shim: run **`etl_hf`'s public build/run path on IREE's Vulkan HAL**.

WHY THIS FILE EXISTS
====================
``etl`` (the library this repo develops) has no ``"vulkan"`` device kind and no
``"vulkan-spirv"`` IREE compile target, so ``etl_hf.build(..., backend="iree",
device=etl.Device("vulkan", 0))`` is rejected by four independent *string*
whitelists inside the IREE adapter before any IREE code runs. The Vulkan HAL
itself works on this host (Mesa **lavapipe**, the software Vulkan
implementation, reached through a byte-patched device name because IREE 3.11
hard-denylists devices whose name starts with ``llvmpipe``), and the low-level
``iree.runtime`` path has been proven to execute etl StableHLO correctly — see
``bench_logs/vulkan_findings.md`` for the probe evidence, the device recipe
(renamed ICD, throwaway venv) and the full gap list this shim closes.

This module is a **benchmark-only runtime shim**: it monkeypatch-widens exactly
those gates in the *live* etl modules at import/install time so an unmodified
``./etl/`` and ``./etl_hf/`` can be driven end-to-end (trace → lower → compile →
load → run) on lavapipe. It is **never** a library change and must never be
imported by the library, its tests, or shipped code: nothing under ``./etl/``,
``./etl_hf/`` or ``./tests/`` knows it exists. If etl ever grows a first-class
``vulkan`` kind, this whole module becomes dead code.

THE GATES THIS SHIM WIDENS (authoritative line numbers as read in this tree)
---------------------------------------------------------------------------
``etl/backends/adapters/iree.py``
  * ``:1071`` ``IreeBackend._resolve_compile_options`` —
    ``supported_targets = {"llvm-cpu", "cuda", "rocm"}`` rejects
    ``target_backends=["vulkan-spirv"]``.  → wrapped (a local set literal cannot
    be widened from outside, so the wrapper resolves the argument list for a
    vulkan-only target set itself and delegates every other case).
  * ``:460`` ``_artifact_target`` — target → artifact label map; ``vulkan-spirv``
    falls through to the ``"+".join(...)`` label.  → wrapped (label ``"vulkan"``).
  * ``:486`` ``_GPU_DEVICE_KINDS = frozenset({"cuda", "rocm"})`` — decides the
    ``rt.Config(device=...)`` load path (``:607`` ``_load_vm_module``) and the
    cuda/rocm upload fast paths.  → module global widened with ``"vulkan"``.
  * ``:536`` ``_acquire_runtime_device`` — ``device_id = device.index + 1`` is a
    CUDA/ROCm 1-based ordinal; a Vulkan ``device_id`` is a queried UUID hash, so
    ``create_device(device_id=1)`` raises ``Device id 1 not found``.  → wrapped:
    the vulkan branch enumerates the driver and creates the device from the
    queried info dict (and caches it per ``(index, runtime args)``).
  * ``:1326`` ``IreeBackend.load`` — ``device.kind not in ("cpu","cuda","rocm",
    "hip")`` rejects a vulkan device, and ``:1346`` ``expected_target = {...}
    [device_kind]`` raises a ``KeyError`` for it.  → ``load`` wrapped; for a
    vulkan device the wrapper performs the adapter's own load steps against the
    adapter's own helpers (``check_available`` / ``validate_options`` /
    ``_apply_iree_runtime_args`` / ``_acquire_runtime_device`` /
    ``_load_vm_module`` / ``IreeExecutable``) with those two gates widened;
    every other device delegates to the library method unchanged.
  * ``:2907`` ``register()`` / ``:2820`` ``upload_tensor`` — the transfer-provider
    registration loop and the placement gate cover ``cuda``/``rocm``/``hip``
    only, so ``Tensor.to(Device("vulkan", N))`` finds no provider.  → the
    adapter's own ``upload_tensor`` is registered for the ``"vulkan"`` kind
    (per-backend slot *and* flat slot).
  * ``:1817`` ``_staged_input`` — the non-ROCm branch is the **CUDA** pinned/DMA
    staging tier, whose correctness rests on a CUDA driver guarantee (a host
    write into mapped device memory is serialized against device reads). That
    guarantee is unverified on lavapipe, so for ``"vulkan"`` the shim returns
    ``None`` and ``upload_tensor`` takes its documented ``rt.asdevicearray``
    fallback — a plain, readback-safe copy.

``etl/pipeline.py``
  * ``:405`` ``_resolve_backend_device`` — device → ``target_backends``
    inference covers ``cuda``/``rocm``/``hip``/``cpu``; without a vulkan branch
    ``etl.build(..., backend="iree", device=Device("vulkan", 0))`` compiles
    ``llvm-cpu`` and then fails the load-time target/device check.  → wrapped
    (adds ``["vulkan-spirv"]`` when the option and ``ETL_TARGET_BACKENDS`` are
    both unset).

Deliberately NOT patched (verified unnecessary):
  * ``etl/core/device.py:122 devices()`` — only needed to *enumerate* vulkan
    devices; callers here construct ``Device("vulkan", 0)`` directly (whose
    constructor has no kind whitelist).
  * ``etl/backends/adapters/iree.py:1417`` (``_load_segments``) — the
    external-kernel split path; a HuggingFace forward graph has no
    ``external_call``. A vulkan load of a *split* artifact still delegates to
    the library method and fails its device gate explicitly.
  * ``etl_hf/pipeline.py:193 _targets_from_device`` — it only selects etl_hf's
    ``opt_level`` default (``O3`` for a vulkan target); the compile targets come
    from ``target_backends`` / the pipeline inference above.
  * ``etl/core/tensor.py`` — provider resolution is kind-agnostic; it only needs
    a provider registered for the kind (above).
  * ``etl.io`` — not used by this path.

INSTALL SEMANTICS
-----------------
``install()`` is idempotent and returns the record of what it patched
(``{site: "what changed"}``, also available later as ``patched_sites()``).
It must run **before** the executable is built (it patches ``IreeBackend.load``
and the pipeline inference, both resolved per call) and after etl is importable;
it lazily imports ``etl.backends`` and activates the IREE adapter once.

The environment recipe is the caller's job (see ``bench_logs/vulkan_findings.md``):
``VK_ICD_FILENAMES`` must point at the renamed-ICD json (otherwise the loader
dlopens the radeon ICD, which blocks on this host) and ``iree-base-runtime`` /
``iree-base-compiler`` 3.11 must be installed. ``device_info()`` reports what the
driver actually enumerated.

USAGE
-----
    import hf_bench_vulkan                      # this file, from ./bench_logs
    hf_bench_vulkan.install()
    import etl, etl_hf

    loaded = etl_hf.load_hf(ckpt)
    cfg    = loaded.config
    fwd    = etl_hf.forward_fn(cfg, seq_len=32, pack_weights=True)
    dev    = etl.Device("vulkan", 0)
    # target_backends=["vulkan-spirv"] is inferred from the device (patched site 8)
    exe    = etl_hf.build(fwd, cfg, backend="iree", device=dev)
    plan   = etl_hf.pack_plan(etl_hf.param_specs(cfg))     # pack on the HOST first
    params = etl_hf.place_params(etl_hf.pack_params(plan, loaded.params), dev)
    ids    = etl_hf.place_ids(input_ids, dev)              # both sides placed
    out    = etl_hf.run(exe, params, ids)                  # device-resident logits
    host   = out.to(etl.Device("cpu", 0)).numpy()          # the one explicit D2H copy

``python bench_logs/hf_bench_vulkan.py --check`` installs the shim and prints
the patch table plus the enumerated Vulkan device (a self-check, no model).

Caveats that are *not* the shim's to hide: lavapipe is a **software** Vulkan
device on the CPU (no hardware acceleration — treat its timings as a Vulkan-path
correctness/perf datapoint, not as GPU numbers), and ``iree.runtime`` on
CPython 3.14 can emit a nanobind teardown dump at interpreter exit; scripts that
must exit cleanly flush and call ``os._exit(0)`` (see the ``--check`` path).
"""
from __future__ import annotations

import base64
import os
import sys
import threading
from typing import Any

__all__ = [
    "install",
    "is_installed",
    "patched_sites",
    "device_info",
    "VULKAN_KIND",
    "VULKAN_TARGET",
]

#: The etl/``core.Device`` kind and the IREE compile target this shim is about.
VULKAN_KIND = "vulkan"
VULKAN_TARGET = "vulkan-spirv"

#: The compile targets the *library* accepts today — mirrors the set literal at
#: ``etl/backends/adapters/iree.py:1071``. Used only to split a requested target
#: list into "the library handles this" vs "the shim must resolve it"; if the
#: library ever grows targets, more requests simply delegate to it unchanged.
_LIBRARY_COMPILE_TARGETS = frozenset({"llvm-cpu", "cuda", "rocm"})

#: Targets the shim itself accepts on top of the library's set.
_SHIM_EXTRA_COMPILE_TARGETS = frozenset({VULKAN_TARGET})

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_PATCHED: dict[str, str] = {}

#: One cached IREE HAL device per ``(kind, index, runtime args)``. Sharing the
#: device between ``Tensor.to`` uploads and the executable's module load is what
#: makes the same-device pass-through work with zero host round-trips (and it
#: avoids paying lavapipe device creation per uploaded parameter leaf).
_VULKAN_DEVICES: dict[tuple, Any] = {}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def is_installed() -> bool:
    """Whether :func:`install` has already widened the gates in this process."""
    return _INSTALLED


def patched_sites() -> dict[str, str]:
    """Return a copy of the ``{site: description}`` record of the patches."""
    return dict(_PATCHED)


def install() -> dict[str, str]:
    """Widen the IREE adapter's Vulkan gates in the live etl modules.

    Idempotent: the second and later calls return the same record without
    patching again. Returns ``{site: description}`` (``site`` is
    ``"<file>:<line> <symbol>"``, matching the docstring's gap list).

    Raises:
        Whatever the lazy ``etl.backends`` / IREE import raises (e.g. a
        ``BackendError`` naming ``pip install etl[iree]`` when the iree packages
        are absent) — never swallowed.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return dict(_PATCHED)
        adapter = _activate_adapter()
        _install_compile_target_whitelist(adapter)
        _install_artifact_target(adapter)
        _install_gpu_device_kinds(adapter)
        _install_runtime_device(adapter)
        _install_staged_input(adapter)
        _install_load(adapter)
        _install_transfer_providers(adapter)
        _install_pipeline_target_inference()
        _INSTALLED = True
        return dict(_PATCHED)


def device_info() -> dict[str, Any]:
    """Return what the IREE Vulkan driver actually enumerates.

    ``{"count": n, "devices": [info dicts], "name": …, "device_id": …,
    "path": …, "dump_device_info": …}`` — the evidence a report needs. Creates
    (and caches) the IREE Vulkan *driver* only; it does not create a device.

    Raises:
        ImportError: ``iree.runtime`` is unavailable.
    """
    import iree.runtime as rt

    driver = rt.get_driver(VULKAN_KIND)
    infos = driver.query_available_devices()
    record: dict[str, Any] = {"count": len(infos), "devices": list(infos)}
    if infos:
        first = infos[0]
        record["name"] = first.get("name")
        record["device_id"] = first.get("device_id")
        record["path"] = first.get("path")
    try:
        # ``dump_device_info`` PRINTS the device properties to the process stdout
        # and returns ``None`` — capture stdout at the file-descriptor level to
        # read it (Python's ``contextlib.redirect_stdout`` does not see it).
        returned = driver.dump_device_info(0)
        record["dump_device_info"] = (
            f"<printed to the process stdout; returned {returned!r}>"
        )
    except Exception as exc:  # noqa: BLE001 - evidence only, never load-bearing
        record["dump_device_info"] = f"<dump_device_info failed: {type(exc).__name__}: {exc}>"
    return record


# ---------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------
def _activate_adapter() -> Any:
    """Import and activate the IREE adapter (registers its cuda/rocm/hip slots)."""
    import etl.backends  # noqa: F401  (registers the optional adapters)
    from etl.backends.adapters import iree as adapter

    adapter.register()
    return adapter


# ---------------------------------------------------------------------------
# patches: compile target whitelist (iree.py:1071) + artifact label (:460)
# ---------------------------------------------------------------------------
def _install_compile_target_whitelist(adapter: Any) -> None:
    """Accept ``target_backends=["vulkan-spirv", ...]`` in ``_resolve_compile_options``.

    The supported-target check reads a LOCAL set literal, so it cannot be
    widened by rebinding a module global. The wrapper therefore hands the
    library's own resolver a target list it accepts (the subset it knows) and
    re-attaches the vulkan target to the returned tuple; for a vulkan-ONLY
    request it resolves the argument list itself — exactly the library's
    ``opt_level`` + ``iree_compile_args`` resolution, which adds per-target
    defaults only for ``llvm-cpu``/``rocm`` (neither is present here). Unknown
    targets still go through the library resolver, so its error is unchanged.
    """
    from etl import core

    original = adapter.IreeBackend._resolve_compile_options

    def _resolve_compile_options(self: Any, options: dict | None):
        raw_targets = (options or {}).get("target_backends", ["llvm-cpu"])
        if (
            not isinstance(raw_targets, (list, tuple))
            or not raw_targets
            or not all(isinstance(target, str) for target in raw_targets)
        ):
            return original(self, options)  # the library's own validation error
        extra = [
            target
            for target in raw_targets
            if target not in _LIBRARY_COMPILE_TARGETS
        ]
        if not extra or any(
            target not in _SHIM_EXTRA_COMPILE_TARGETS for target in extra
        ):
            return original(self, options)  # unknown target -> the library's error
        kept = [
            target
            for target in raw_targets
            if target in _LIBRARY_COMPILE_TARGETS
        ]
        if kept:
            proxy = dict(options or {})
            proxy["target_backends"] = kept
            _, extra_args = original(self, proxy)
        else:
            extra_args = _vulkan_only_compile_args(adapter, core, raw_targets, options)
        return tuple(raw_targets), extra_args

    _resolve_compile_options.__doc__ = (
        "shim wrapper over IreeBackend._resolve_compile_options "
        "(etl/backends/adapters/iree.py:1071) that accepts 'vulkan-spirv'."
    )
    adapter.IreeBackend._resolve_compile_options = _resolve_compile_options
    _PATCHED["etl/backends/adapters/iree.py:1071 IreeBackend._resolve_compile_options"] = (
        "wrapped: accepts the extra compile target 'vulkan-spirv'"
    )


def _vulkan_only_compile_args(
    adapter: Any, core: Any, targets: tuple | list, options: dict | None
) -> list[str]:
    """Resolve iree-compile args for a vulkan-only target list.

    Mirrors the option handling of ``IreeBackend._resolve_compile_options``
    (``iree_compile_args`` type check, ``opt_level`` → ``--iree-opt-level=O{N}``)
    and then uses the library's own ``_resolve_iree_compile_args`` with the
    vulkan target list — which contributes no per-target defaults.
    """
    from etl.backends.options import normalize_opt_level

    raw_extra = (options or {}).get("iree_compile_args")
    if raw_extra is None:
        user_args: list[str] = []
    elif isinstance(raw_extra, (list, tuple)) and all(
        isinstance(flag, str) for flag in raw_extra
    ):
        user_args = list(raw_extra)
    else:
        raise core.BackendError(
            f"the iree 'iree_compile_args' compile option must be a list/tuple "
            f"of flag strings, got {raw_extra!r}"
        )
    raw_opt_level = (options or {}).get("opt_level")
    if raw_opt_level is None:
        opt_level_args: tuple[str, ...] = ()
    else:
        opt_level_args = (f"--iree-opt-level=O{normalize_opt_level(raw_opt_level)}",)
    return adapter._resolve_iree_compile_args(
        tuple(targets), user_args, opt_level_args=opt_level_args
    )


def _install_artifact_target(adapter: Any) -> None:
    """Label a vulkan-spirv artifact ``"vulkan"`` (iree.py:460 ``_artifact_target``)."""
    original = adapter._artifact_target

    def _artifact_target(target_backends: tuple[str, ...]) -> str:
        if set(target_backends) == {VULKAN_TARGET}:
            return VULKAN_KIND
        return original(target_backends)

    adapter._artifact_target = _artifact_target
    _PATCHED["etl/backends/adapters/iree.py:460 _artifact_target"] = (
        "wrapped: {'vulkan-spirv'} -> artifact label 'vulkan'"
    )


# ---------------------------------------------------------------------------
# patches: device kinds, runtime device, staging
# ---------------------------------------------------------------------------
def _install_gpu_device_kinds(adapter: Any) -> None:
    """Add ``"vulkan"`` to ``_GPU_DEVICE_KINDS`` (iree.py:486).

    This set selects the ``rt.Config(device=...)`` module-load path (the one a
    non-``local-task`` HAL device needs) and the non-host branches of the
    per-input buffer build — the vulkan device must take both.
    """
    adapter._GPU_DEVICE_KINDS = frozenset(
        set(adapter._GPU_DEVICE_KINDS) | {VULKAN_KIND}
    )
    _PATCHED["etl/backends/adapters/iree.py:486 _GPU_DEVICE_KINDS"] = (
        "widened: frozenset({'cuda','rocm'}) -> {'cuda','rocm','vulkan'} "
        "(Config-path module load + non-host input branch)"
    )


def _install_runtime_device(adapter: Any) -> None:
    """Resolve the IREE HAL device for a vulkan ``core.Device`` (iree.py:536).

    The library's ``device_id = device.index + 1`` is a CUDA/ROCm 1-based
    ordinal; IREE Vulkan device ids are queried UUID hashes, and
    ``create_device(device_id=1)`` raises ``Device id 1 not found``. The wrapper
    enumerates the driver and creates the device from the queried info dict of
    ``device.index``, caching one device per ``(index, runtime args)`` so the
    uploads (``Tensor.to``) and the module load share ONE HAL device.
    """
    from etl import core

    original = adapter._acquire_runtime_device

    def _acquire_runtime_device(
        device_kind: str, device: Any, runtime_args: list[str]
    ) -> Any:
        if device_kind != VULKAN_KIND:
            return original(device_kind, device, runtime_args)
        import iree.runtime as rt

        index = 0 if device is None else device.index
        key = (VULKAN_KIND, index, tuple(runtime_args or ()))
        cached = _VULKAN_DEVICES.get(key)
        if cached is not None:
            return cached
        driver = rt.get_driver(VULKAN_KIND)
        infos = driver.query_available_devices()
        if not infos:
            raise core.BackendError(
                f"IREE's vulkan driver enumerated no devices (requested "
                f"{VULKAN_KIND} index {index}). On this host lavapipe is hidden "
                "by IREE 3.11's device-name denylist unless VK_ICD_FILENAMES "
                "points at the renamed ICD (see bench_logs/vulkan_findings.md)."
            )
        if index >= len(infos) or index < 0:
            raise core.BackendError(
                f"IREE's vulkan driver enumerated {len(infos)} device(s); "
                f"index {index} does not exist (vulkan device ids are queried "
                f"UUID hashes, not ordinals): "
                f"{[info.get('name') for info in infos]}"
            )
        # ``create_device`` takes the queried info dict (its ``device_id`` is
        # the UUID hash the HAL expects); the index+1 ordinal form the cuda/
        # rocm branch uses does not exist for vulkan.
        runtime_device = driver.create_device(infos[index])
        _VULKAN_DEVICES[key] = runtime_device
        return runtime_device

    adapter._acquire_runtime_device = _acquire_runtime_device
    _PATCHED["etl/backends/adapters/iree.py:536 _acquire_runtime_device"] = (
        "wrapped: vulkan branch creates the device from the QUERIED info dict "
        "(no index+1 ordinal) and caches one HAL device per (index, runtime args)"
    )


def _install_staged_input(adapter: Any) -> None:
    """Skip the CUDA pinned/DMA staging tier for vulkan (iree.py:1817).

    The non-ROCm branch of ``_staged_input`` is the CUDA fast path: its
    correctness rests on the CUDA driver serializing a host write into mapped
    device memory against device reads of that memory. That guarantee is not
    established for lavapipe, so returning ``None`` here makes ``upload_tensor``
    take its documented ``rt.asdevicearray`` fallback — a plain, readback-safe
    host→device copy (same bytes, no ordering assumption).
    """
    original = adapter._staged_input

    def _staged_input(
        staged_inputs: dict | None,
        runtime_device: Any,
        index: int,
        tensor: Any,
        device_kind: str = "cuda",
    ) -> Any:
        if device_kind == VULKAN_KIND:
            return None
        return original(staged_inputs, runtime_device, index, tensor, device_kind)

    adapter._staged_input = _staged_input
    _PATCHED["etl/backends/adapters/iree.py:1817 _staged_input"] = (
        "wrapped: vulkan -> None (upload_tensor's rt.asdevicearray fallback; "
        "the CUDA pinned/DMA ordering guarantee does not hold on lavapipe)"
    )


# ---------------------------------------------------------------------------
# patch: load device-kind whitelist + expected-target map (iree.py:1326, :1346)
# ---------------------------------------------------------------------------
def _install_load(adapter: Any) -> None:
    """Let ``IreeBackend.load`` accept a vulkan device (iree.py:1326/1346).

    ``load``'s device-kind whitelist and its ``expected_target`` map are LOCAL
    literals, so the method cannot be widened from outside by wrapping a helper.
    The wrapper keeps the library method for every other device and, for a
    vulkan device, performs the same load steps against the adapter's own
    helpers with the two gates widened (see :func:`_load_vulkan`).
    """
    original = adapter.IreeBackend.load

    def load(self: Any, artifact: Any, device: Any = None, options: dict | None = None):
        from etl import core

        if not (isinstance(device, core.Device) and device.kind == VULKAN_KIND):
            return original(self, artifact, device, options)
        return _load_vulkan(adapter, core, self, artifact, device, options, original)

    load.__doc__ = (
        "shim wrapper over IreeBackend.load (etl/backends/adapters/iree.py:1211): "
        "a vulkan device takes an equivalent load path with the device-kind "
        "whitelist (:1326) and the expected-target map (:1346) widened; every "
        "other device delegates to the library method unchanged."
    )
    adapter.IreeBackend.load = load
    _PATCHED["etl/backends/adapters/iree.py:1326/1346 IreeBackend.load"] = (
        "wrapped: vulkan device accepted (kind whitelist) and mapped to the "
        "'vulkan-spirv' expected target; other kinds delegate unchanged"
    )


def _load_vulkan(
    adapter: Any,
    core: Any,
    backend: Any,
    artifact: Any,
    device: Any,
    options: dict | None,
    original_load: Any,
) -> Any:
    """The adapter's own load for a vulkan device — gates widened, nothing else.

    Mirrors ``IreeBackend.load`` step for step: artifact backend, payload
    shape, entry functions, availability, option validation, runtime args,
    artifact/device target compatibility, base64 decode, ``_acquire_runtime_device``,
    ``_load_vm_module``, ``IreeExecutable``. Payload problems delegate to the
    library method so its own error (including "corrupt" and the unsupported
    split-artifact device error) is what the caller sees.
    """
    from etl.backends.options import validate_options

    if artifact.backend != backend.name:
        raise core.PersistenceError(
            f"artifact was produced by backend {artifact.backend!r}; the "
            f"{backend.name} backend cannot load it — never silently recompile"
        )
    payload = artifact.payload
    if not (
        isinstance(payload, dict)
        and payload.get("format") == "iree-vmfb"
        and isinstance(payload.get("vmfb_base64"), str)
    ):
        # Split (external-call) artifacts and corrupt payloads: let the library
        # raise its own error. The split path has no vulkan support (it re-hits
        # the device whitelist there), which the library reports explicitly.
        return original_load(backend, artifact, device, options)
    entry_functions = tuple(payload.get("entry_functions", ()))
    if not entry_functions:
        return original_load(backend, artifact, device, options)

    backend.check_available()
    validate_options(options, backend.KNOWN_OPTIONS, backend.name, "load")
    runtime_args = _load_runtime_args(adapter, core, options)
    adapter._apply_iree_runtime_args(runtime_args)

    raw_targets = payload.get("target_backends")
    if raw_targets is None:
        target_backends = ("llvm-cpu",)
    elif isinstance(raw_targets, str):
        target_backends = (raw_targets,)
    else:
        target_backends = tuple(raw_targets)
    expected_target = VULKAN_TARGET
    if expected_target not in target_backends:
        raise core.BackendError(
            f"the {backend.name} artifact was compiled for target(s) "
            f"{', '.join(target_backends) or '(none)'} and cannot run on the "
            f"requested device {device!r} (device kind {VULKAN_KIND!r} requires "
            f"compile target {expected_target!r}) — never silently recompile"
        )

    import base64

    vmfb = base64.b64decode(payload["vmfb_base64"].encode("ascii"))
    runtime_device = adapter._acquire_runtime_device(
        VULKAN_KIND, device, runtime_args
    )
    module = adapter._load_vm_module(vmfb, VULKAN_KIND, runtime_device)
    return adapter.IreeExecutable(
        artifact=artifact,
        signature=artifact.signature,
        device=device,
        module=module,
        entry_functions=entry_functions,
        runtime_device=runtime_device,
    )


def _load_runtime_args(adapter: Any, core: Any, options: dict | None) -> list[str]:
    """Parse the ``iree_runtime_args`` load option (adapter's own rules, same message)."""
    raw_runtime = (options or {}).get("iree_runtime_args")
    if raw_runtime is None:
        return []
    if isinstance(raw_runtime, (list, tuple)) and all(
        isinstance(flag, str) for flag in raw_runtime
    ):
        return list(raw_runtime)
    raise core.BackendError(
        f"the iree 'iree_runtime_args' load option must be a list/tuple of flag "
        f"strings (e.g. ['--cuda_async_allocations=false']), got {raw_runtime!r}"
    )


# ---------------------------------------------------------------------------
# patch: transfer providers for the vulkan kind (iree.py:2907 / :2820)
# ---------------------------------------------------------------------------
def _install_transfer_providers(adapter: Any) -> None:
    """Register the adapter's ``upload_tensor`` for the ``"vulkan"`` kind.

    Both slots the library's resolution consults are filled with the adapter's
    OWN provider (no copy): the per-backend slot ``("vulkan", "iree")`` and the
    flat default slot. With ``_GPU_DEVICE_KINDS`` and ``_staged_input`` widened
    above, that provider performs a plain ``rt.asdevicearray`` upload onto the
    cached vulkan HAL device and wraps it in an ``IreeDevicePayload``.
    """
    from etl import core

    core.register_backend_device_transfer_provider(
        VULKAN_KIND, adapter.IreeBackend.name, adapter.upload_tensor
    )
    core.register_device_transfer_provider(VULKAN_KIND, adapter.upload_tensor)
    _PATCHED["etl/backends/adapters/iree.py:2907 transfer providers"] = (
        "registered the adapter's upload_tensor for kind 'vulkan' in the "
        "per-backend ('vulkan','iree') slot and the flat default slot"
    )


# ---------------------------------------------------------------------------
# patch: pipeline device -> compile-target inference (pipeline.py:405)
# ---------------------------------------------------------------------------
def _install_pipeline_target_inference() -> None:
    """Infer ``["vulkan-spirv"]`` from a vulkan device (pipeline.py:405).

    Without this, ``etl.build(..., backend="iree", device=Device("vulkan", 0))``
    compiles the ``llvm-cpu`` default target and then fails the load-time
    target/device compatibility check. Explicit ``target_backends`` and
    ``ETL_TARGET_BACKENDS`` keep precedence (the wrapper only fills a gap).
    """
    from etl import pipeline

    original = pipeline._resolve_backend_device

    def _resolve_backend_device(backend: Any, device: Any, options: Any):
        resolved_backend, resolved_device, resolved_options = original(
            backend, device, options
        )
        if (
            "target_backends" not in resolved_options
            and getattr(resolved_backend, "name", "").startswith("iree")
            and resolved_device is not None
            and resolved_device.kind == VULKAN_KIND
        ):
            resolved_options["target_backends"] = [VULKAN_TARGET]
        return resolved_backend, resolved_device, resolved_options

    pipeline._resolve_backend_device = _resolve_backend_device
    _PATCHED["etl/pipeline.py:405 _resolve_backend_device"] = (
        "wrapped: an iree backend on a vulkan device infers "
        "target_backends=['vulkan-spirv'] when the option and "
        "ETL_TARGET_BACKENDS are unset"
    )


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    """``python bench_logs/hf_bench_vulkan.py`` — install and print the record.

    Needs only etl + iree.runtime, no model and no checkpoint: it prints the
    patch table and the enumerated Vulkan device, i.e. the two facts every
    measurement with this shim depends on.
    """
    import json

    install()
    print(f"installed shim: {len(patched_sites())} patch site(s)")
    for site, what in sorted(patched_sites().items()):
        print(f"  - {site}\n      {what}")
    info = device_info()
    print("\nIREE vulkan driver:")
    for key in ("count", "name", "device_id", "path"):
        print(f"  {key}: {info.get(key)}")
    print(f"  devices: {json.dumps(info['devices'], default=str)}")
    print(f"  dump_device_info:\n{info['dump_device_info']}")
    print(f"\nVK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES')!r}")
    return 0


if __name__ == "__main__":
    if "--check" not in sys.argv[1:] and len(sys.argv) > 1:
        print("usage: hf_bench_vulkan.py [--check]", file=sys.stderr)
        raise SystemExit(2)
    code = _main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # dodge iree.runtime's py3.14 nanobind teardown dump
