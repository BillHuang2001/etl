"""IREE compiler adapter — ``etl.backends.adapters.iree``.

Pluggable compiler backend that consumes the shared StableHLO payload
(``CompilerBackend.lower``, see ``../compiler.py``) and compiles it with the
external IREE compiler, producing an IREE VM flatbuffer executed through the
IREE runtime on the ``local-task`` CPU driver or the ``cuda`` HAL driver —
both drivers are compiled into the runtime wheel (validated on IREE 3.11.0,
``iree-base-compiler``/``iree-base-runtime`` 3.11.0rc20260316).

Exact IREE Python APIs used (all inside function bodies — the heavy-import
rule: ``iree`` / ``iree.compiler`` / ``iree.runtime`` / ``numpy`` are NEVER
imported at module top level, so ``import etl`` stays light):

- ``iree.compiler.compile_str(text, target_backends=..., input_type=
  "stablehlo", extra_args=[...])`` — compiles the StableHLO MLIR payload
  in-process (spawns ``iree-compile``) and returns the VM flatbuffer as
  ``bytes``. The compile ``target_backends`` option selects the targets
  (default ``["llvm-cpu"]``; v1 supported set ``{"llvm-cpu", "cuda"}`` —
  anything else is rejected with ``core.BackendError`` naming it, never a
  silent fallback). ``"cuda"`` emits default sm_60 PTX, JIT'd by the CUDA
  driver to newer archs at load (validated on 3.11.0 to sm_86 / RTX A6000 —
  no arch flag needed). ``extra_args`` = etl's MINIMAL defaults — ALL
  OVERRIDABLE via the ``iree_compile_args`` compile option (a user flag
  overriding a default by name replaces it; the small documented deny list
  ``_IREE_DENIED_COMPILE_FLAGS`` guards only the genuinely required flags;
  every other flag passes through to the compiler, which validates it):
  ``--iree-input-demote-f64-to-f32=false`` (IREE demotes f64 to f32 by
  default for StableHLO input — silent dtype coercion, which etl's error
  strategy forbids; disabling it keeps f64 semantics exact) and, ONLY when
  ``llvm-cpu`` is among the requested targets,
  ``--iree-llvmcpu-target-cpu=generic`` (portable generic-CPU codegen; also
  silences IREE's generic-CPU warning — llvm-cpu-specific and meaningless for
  cuda, so not passed there) and
  ``--iree-llvmcpu-link-embedded=false`` (the default embedded
  ``-nostdlib -static`` link cannot resolve libm symbols — ``log``/``cos``/
  ``floor`` — that f64 math ops lower to; the dynamically-linked dylib
  resolves them from the system libc/libm at load time).
- ``iree.runtime.get_driver("local-task")`` / ``get_driver("cuda")`` →
  ``driver.create_default_device()`` (GPU 0 for cuda) — the RELIABLE
  device-acquisition path. The historical
  ``rt.system_setup(config=rt.Config("local-task"))`` recipe fails with
  ``TypeError: 'module' object is not callable`` because
  ``iree.runtime.system_setup`` is a MODULE in IREE 20241104.1068
  (distributed at the time as ``iree-runtime``; the current PyPI
  distribution is ``iree-base-runtime``); its function was replaced by
  ``system_api.Config``. This path was validated repeatedly (fresh
  processes and in-process loops).
- ``driver.create_device(device_id=N)`` — IREE CUDA device ids are 1-BASED:
  N maps to physical GPU index N-1 (verified empirically on 3.11.0:
  device_id=4 -> GPU 3; device_id=1 -> GPU 0); device_id=0 raises ValueError
  "Device id 0 not found" — never pass 0. etl maps ``Device("cuda", index)``
  to ``device_id=index+1`` (default index 0 -> device_id 1).
- ``iree.runtime.load_vm_flatbuffer(flatbuffer_bytes, driver="local-task")``
  — loads the VM flatbuffer into a ``BoundModule`` (CPU path); entry
  functions are resolved as attributes (``module.main``). This driver-name
  form ALWAYS creates the DEFAULT device, so it cannot select a CUDA index —
  the CUDA path instead loads via ``rt.Config(device=device)`` +
  ``rt.VmModule.copy_buffer(config.vm_instance, vmfb)`` +
  ``rt.load_vm_module(vm_module, config)``, which binds the module to a
  SPECIFIC acquired device.
- ``iree.runtime.asdevicearray(device, np_array)`` — copies a host numpy
  array into a HAL device buffer for function input (used only for the host
  input FALLBACKS — oversized inputs, unmappable dtypes, non-cuda devices;
  the cuda fast path stages host inputs through persistent host-local +
  device buffers, see ``IreeExecutable.run`` / ``_staged_input``).
- ``DeviceArray`` (``iree.runtime.DeviceArray``) — a HAL buffer-view
  handle, already resident on the device. Run outputs are wrapped into
  ``core.Tensor(IreeDevicePayload(...))`` — device-resident tensors whose
  ``.numpy()`` materializes a LAZY host copy (``DeviceArray.to_host()``) on
  demand, so the hot run() path performs no host round-trip. Same-device
  DeviceArray inputs are handed to the invoke directly (no copy).

Staging is explicit and honest: ``compile`` NEVER loads, ``load`` NEVER
re-lowers/re-compiles. Compiler failures surface as
``core.BackendError`` carrying the ``iree.compiler.CompilerToolError``
diagnostics — never silently swallowed or papered over.

Capability notes (all validated end-to-end on IREE 3.11.0 —
``iree-base-compiler``/``iree-base-runtime`` 3.11.0rc20260316 — llvm-cpu CPU
and cuda GPU; the notes below apply to BOTH targets — fp64/fp16 etc. all work
on cuda, and the v1 wgp list includes fp64):

- ``dtypes``: float16/float32/float64/int8/int16/int32/int64/bool_. NOT
  declared: uint8/uint16 (iree-compile cannot legalize unsigned-int
  ``reduce`` on these — deterministic failure), uint32/uint64
  (iree-compile 20241104 legalizes the same unsigned-reduce module
  NON-DETERMINISTICALLY — an upstream race: identical input+flags flip
  between success and "failed to legalize unresolved materialization" across
  trials, so they cannot be a reliable capability), complex64/complex128
  (the StableHLO exporter v1 defers complex computation beyond ``cast``).
- ``collectives=False``: NONE of the six ``etl.dist`` collectives can run.
  ``stablehlo.collective_broadcast`` (``etl.dist.broadcast``) cannot be
  LEGALIZED by iree-compile 20241104 ("failed to legalize operation
  'stablehlo.collective_broadcast' that was explicitly marked illegal" —
  upstream limitation); the other five compile but the local-task/local-sync
  CPU and cuda HAL drivers of the IREE runtime wheels (3.11.0 validated)
  raise "UNIMPLEMENTED; collectives not implemented" at ``hal.channel.create``
  (the Python wheels ship no communicating channel provider). With this flag
  False, the SHARED ``lower()`` rejects every collective-effect op explicitly
  with ``core.BackendError`` naming it — never a silent fallback.
- ``dynamic_shapes=True``: a symbolic-dim graph compiles and runs correctly
  with different concrete sizes. Scalar-constant broadcasts over dynamic
  shapes (e.g. ``relu`` on a dynamic tensor) are emitted as
  ``stablehlo.dynamic_broadcast_in_dim`` with a
  ``get_dimension_size``-built runtime ``output_dimensions`` chain —
  validated end-to-end on 3.11.0 at multiple concrete sizes.

Import acyclicity (binding, ``../CONTEXT.md``): top-level imports restricted
to stdlib + ``etl.core`` + the sibling modules ``compiler`` / ``registry`` /
``program`` / ``backend``. ``etl.pipeline`` is never imported.
"""
from __future__ import annotations

import base64
import os
import warnings
from typing import TYPE_CHECKING, Any

from etl import core
from etl.core import Device

from ..backend import Capabilities
from ..compiler import CompilerBackend, CompilerExecutable
from ..program import CompiledArtifact, LoweredProgram, Signature
from ..registry import register as _registry_register

# ---------------------------------------------------------------------------
# CUDA per-call fast-path configuration (measured on IREE 3.11.0, RTX A6000)
# ---------------------------------------------------------------------------
# Host inputs on cuda executables are uploaded through PERSISTENT staging
# buffers cached per (input index, shape, dtype), in TWO tiers:
#   * TINY inputs (<= _PINNED_DIRECT_MAX_BYTES, measured crossover ~16-32 KB
#     on IREE 3.11.0 / RTX A6000): the host values are memcpy'd straight
#     into the mapped view of a persistent DEVICE_LOCAL | HOST_VISIBLE
#     ("pinned") buffer that the dispatches read DIRECTLY — no queue_copy,
#     no semaphore, no wait. The driver serializes the host write against
#     device reads of the mapped memory, which is what makes the dispatch
#     see the new values (and what makes large pinned writes slow — the
#     measured 0.5-1.2 ms/call at 160 KB — hence the size cutoff). Measured
#     for the DE step's rank-0 int64 key (8 B): ~0.05-0.06 ms incl. entry,
#     vs ~0.11-0.12 ms for the DMA path, and it does NOT degrade when the
#     upload follows another invoke (the DMA path's fresh-semaphore wait
#     inflates to 0.24-0.39 ms in that pattern).
#   * LARGER inputs: the DMA tier — np.copyto into the mapped view of a
#     persistent HOST_LOCAL source buffer, then a per-call async queue_copy
#     into a persistent DEVICE_LOCAL staging buffer the dispatches read,
#     waiting on a FRESH semaphore signaled by the copy (~0.04 ms idle;
#     the copy itself is a ~0.01 ms DMA). A fresh semaphore per call is
#     REQUIRED: reusing a semaphore whose event was already waited aborts
#     the CUDA event semaphore inside the next invoke (verified:
#     event_semaphore.c:354 ABORTED on the second use). No per-call device
#     allocation in either tier.
# NOTE (measured, do not "optimize" back): the queue_copy and the invoke's
# dispatches are NOT reliably ordered on the cuda HAL — dropping the DMA
# tier's semaphore wait races the dispatch against the copy (~0.4 % of runs
# return garbage; verified by 1000-run alternating-value stress). The wait
# is load-bearing. The pinned tier avoids the race by construction (no DMA).
# Two invariants keep the DMA tier fast:
#   * the CUDA driver's ASYNC (stream-ordered) allocator is disabled via the
#     process-global runtime flag --cuda_async_allocations=false: with it
#     enabled the pool re-trims to empty whenever invoke results are freed,
#     making every subsequent invocation ~1.1 ms (vs ~0.04 ms warm);
#   * each cuda executable retains a small DEVICE-LOCAL anchor buffer
#     (_pool_anchor) that keeps the classic allocator's pool from emptying,
#     so per-call result allocation/free stays ~0.04 ms (measured: 4 KB
#     anchor suffices, independent of the result size class).
# DESIGN NOTE (measured, do not "simplify" back): the earlier fast path had
# dispatches read the host values directly from a persistent DEVICE_LOCAL |
# HOST_VISIBLE ("pinned", host-mapped DEVICE memory) buffer, memcpy'd from
# host every call. That pattern measured ~0.5-1.2 ms/call on cuda while the
# pure invoke was 0.04-0.08 ms: host writes into the host-mapped DEVICE
# memory go over PCIe write-through AND the driver serializes a host write
# against any outstanding device read of that memory (~0.3-0.6 ms stall),
# so the "pinned direct-read" idea is inherently slow. Writing into
# HOST_LOCAL memory is a plain cached memcpy (~0.01 ms for 160 KB) and the
# DMA queue_copy is ~0.01 ms, so the staged path tracks the pure invoke
# cost (measured 0.08-0.13 ms/call quiet, ~0.3 ms under PCIe contention on
# this busy 8-GPU box).
# Both invariants and the staging are pure performance configuration —
# correctness-neutral (identical buffers, copies, and synchronization as
# the default paths).
_CUDA_FLAGS_CONFIGURED = False
_STAGED_INPUT_MAX_BYTES = 16 * 1024 * 1024  # larger inputs keep asdevicearray
# Pinned direct-write tier cutoff: host writes into DEVICE_LOCAL|HOST_VISIBLE
# mapped memory are serialized by the driver against device reads, so large
# pinned uploads stall (0.5-1.2 ms at 160 KB, measured); below ~4 KB the
# pinned tier is both faster than the DMA tier AND immune to its race.
_PINNED_DIRECT_MAX_BYTES = 4 * 1024
_STAGED_INPUT_CACHE_MAX = 64  # bounded per (index, shape, dtype) cache

# ---------------------------------------------------------------------------
# Compile-flag policy (the options-override contract — see ../options.py)
# ---------------------------------------------------------------------------
# etl's compile flags are MINIMAL and every one of them is OVERRIDABLE by
# the user via the ``iree_compile_args`` compile option (a user flag whose
# NAME matches a default's name replaces the default — never both) or the
# ``ETL_IREE_COMPILE_ARGS`` env var (resolved by the pipeline, precedence:
# explicit option > env var > etl default). The design lesson behind this
# policy: a hardcoded, non-overridable iree flag once disabled iree's
# stream-ordered allocator (``--cuda_async_allocations=false``) and caused a
# ~100x per-call regression users could not fix. etl never hardcodes flags
# users cannot override.
_IREE_DEFAULT_COMPILE_ARGS = ("--iree-input-demote-f64-to-f32=false",)
# llvm-cpu-only defaults — added ONLY when "llvm-cpu" is among the requested
# targets (meaningless for cuda):
#   --iree-llvmcpu-target-cpu=generic   portable generic-CPU codegen (also
#                                       silences IREE's generic-CPU warning)
#   --iree-llvmcpu-link-embedded=false  the default embedded -nostdlib -static
#                                       link cannot resolve libm symbols
#                                       (log/cos/floor) that f64 math ops lower
#                                       to; the dynamically-linked dylib
#                                       resolves them from system libc/libm
#                                       at load
_IREE_DEFAULT_LLVMCPU_ARGS = (
    "--iree-llvmcpu-target-cpu=generic",
    "--iree-llvmcpu-link-embedded=false",
)
# Genuinely required flags — the small documented DENY list. Overriding these
# would break etl's own machinery contract with ``iree.compiler.compile_str``
# (targets/input format are passed as its parameters; the vmfb output format
# is what the adapter loads), so they are rejected with an actionable message
# — never silently overwritten, never passed through. EVERY other iree-compile
# flag passes through unvalidated (the compiler validates flag values and its
# diagnostics are surfaced as BackendError).
_IREE_DENIED_COMPILE_FLAGS = {
    "--iree-hal-target-backends": (
        "use the 'target_backends' compile option instead"
    ),
    "--iree-input-type": (
        "etl always feeds StableHLO; the input type is fixed by the adapter"
    ),
    "--iree-vm-bytecode-module-output-format": (
        "the adapter requires the default VM flatbuffer output; changing it "
        "would break artifact loading"
    ),
}


def _flag_name(flag: str) -> str:
    """The flag name part (the text before '=') — override/deny matching."""
    return flag.partition("=")[0]


def _resolve_iree_compile_args(
    target_backends: tuple[str, ...], user_args: list[str] | None
) -> list[str]:
    """Final ``extra_args`` for ``iree.compiler.compile_str``.

    = surviving etl defaults (a default whose name is overridden by a user
    flag is dropped — the user wins) + the user's flags, deny-checked.
    """
    defaults = list(_IREE_DEFAULT_COMPILE_ARGS)
    if "llvm-cpu" in target_backends:
        defaults.extend(_IREE_DEFAULT_LLVMCPU_ARGS)
    if not user_args:
        return defaults
    user_names = [_flag_name(flag) for flag in user_args]
    for flag, name in zip(user_args, user_names):
        reason = _IREE_DENIED_COMPILE_FLAGS.get(name)
        if reason is not None:
            raise core.BackendError(
                f"the {IreeBackend.name} backend denies the iree-compile "
                f"flag {flag!r}: {reason}"
            )
    return [
        default
        for default in defaults
        if _flag_name(default) not in user_names
    ] + list(user_args)


def _artifact_target(target_backends: tuple[str, ...]) -> str:
    """The ``CompiledArtifact.target`` label for a target-backends tuple."""
    if set(target_backends) == {"llvm-cpu"}:
        return "cpu"
    if set(target_backends) == {"cuda"}:
        return "cuda"
    return "+".join(target_backends)


def _runtime_dependencies() -> dict:
    """Self-describing runtime-dependency versions for CompiledArtifacts."""
    import numpy as np

    return {
        "numpy": np.__version__,
        "iree-base-compiler": _iree_distribution_version(
            "iree-base-compiler", "iree-compiler"
        ),
        "iree-base-runtime": _iree_distribution_version(
            "iree-base-runtime", "iree-runtime"
        ),
    }


def _acquire_runtime_device(
    device_kind: str, device: Device | None, runtime_args: list[str]
) -> Any:
    """Acquire the IREE runtime HAL device (shared by single/segment loads).

    CPU: the reliable path (``rt.get_driver("local-task")`` +
    ``create_default_device()``). CUDA: ``rt.get_driver("cuda")`` +
    ``driver.create_device(device_id=index+1)`` (IREE cuda device ids are
    1-BASED: etl ``Device("cuda", index)`` maps to ``device_id = index + 1``
    — verified empirically on 3.11.0: device_id=4 -> physical GPU 3;
    device_id=0 raises ValueError "Device id 0 not found", never pass 0);
    the driver's ``ValueError`` for an out-of-range index is surfaced as
    ``core.BackendError`` naming the index.
    """
    import iree.runtime as rt

    if device_kind == "cuda":
        _configure_cuda_runtime_flags(tuple(runtime_args))
        driver = rt.get_driver("cuda")
        device_id = device.index + 1
        try:
            return driver.create_device(device_id=device_id)
        except ValueError as exc:
            raise core.BackendError(
                f"IREE could not acquire CUDA device id {device_id} "
                f"(1-based; physical GPU index {device.index}): {exc}. "
                f"IREE cuda device ids are 1-based and device_id=0 does "
                f"not exist; the requested index must be less than the "
                f"number of available GPUs (check nvidia-smi)"
            ) from exc
    driver = rt.get_driver("local-task")
    return driver.create_default_device()


def _load_vm_module(vmfb: bytes, device_kind: str, runtime_device: Any) -> Any:
    """Load one VM flatbuffer onto ``runtime_device`` (shared load path).

    CUDA: the ``rt.Config(device=...)`` + ``rt.VmModule.copy_buffer(config.
    vm_instance, vmfb)`` + ``rt.load_vm_module(vm_module, config)`` path —
    ``load_vm_flatbuffer``'s driver-name form always creates the DEFAULT
    device and cannot select a CUDA index. CPU: ``load_vm_flatbuffer(vmfb,
    driver="local-task")``; a ``ValueError`` (the runtime VM verifier
    rejecting the module's required features — the mixed-install trap, see
    ``adapters/CONTEXT.md`` Known Issue #7) is surfaced as an actionable
    ``core.BackendError`` naming the cause and remedy.
    """
    import iree.runtime as rt

    if device_kind == "cuda":
        config = rt.Config(device=runtime_device)
        vm_module = rt.VmModule.copy_buffer(config.vm_instance, vmfb)
        return rt.load_vm_module(vm_module, config)
    try:
        return rt.load_vm_flatbuffer(vmfb, driver="local-task")
    except ValueError as exc:
        # Mixed-install trap (Known Issue #7 in adapters/CONTEXT.md): the
        # legacy iree-compiler/iree-runtime distributions and the current
        # iree-base-* distributions share the SAME 'iree' Python namespace
        # and must NOT coexist; a mixed/residual install makes the runtime
        # VM verifier reject the module ("required module features [Ch] are
        # not available in this runtime configuration") with a raw ValueError
        # deep inside the runtime. Surface it as an actionable BackendError
        # instead of the cryptic raw error.
        raise core.BackendError(
            f"IREE could not load the compiled VM module: the runtime "
            f"rejected the module's required features at load ({exc}). "
            f"This is almost certainly a MIXED IREE INSTALL: the legacy "
            f"'iree-compiler'/'iree-runtime' distributions and the "
            f"current 'iree-base-compiler'/'iree-base-runtime' "
            f"distributions share the SAME 'iree' Python namespace and "
            f"must NOT coexist — a mixed or residual install (e.g. pip "
            f"uninstall leaving dist-info/egg-info residue) makes the "
            f"runtime verifier claim the module's features are "
            f"unavailable. Remedy: purge ALL 'iree*' distributions from "
            f"site-packages (including stale dist-info/egg-info residue) "
            f"and reinstall ONLY 'iree-base-compiler' and "
            f"'iree-base-runtime'."
        ) from exc


def _configure_cuda_runtime_flags(user_args: tuple[str, ...] = ()) -> None:
    """Parse the process-global CUDA allocator flag ONCE (idempotent).

    The etl default ``--cuda_async_allocations=false`` is OVERRIDABLE and
    applied ONLY when neither the user's explicit ``iree_runtime_args`` load
    option nor the ``IREE_PY_RUNTIME_FLAGS`` env var mentions
    ``cuda_async_allocations`` (env is parsed at iree import; the explicit
    option is parsed AFTER it and wins by last-wins flag semantics). Parse
    errors of the DEFAULT are ignored (best-effort performance tuning, never
    correctness) — user-provided runtime args are parsed separately with LOUD
    errors (``_apply_iree_runtime_args``).
    """
    global _CUDA_FLAGS_CONFIGURED
    if _CUDA_FLAGS_CONFIGURED:
        return
    _CUDA_FLAGS_CONFIGURED = True
    if "cuda_async_allocations" in os.environ.get("IREE_PY_RUNTIME_FLAGS", ""):
        return
    if any(
        _flag_name(flag) == "--cuda_async_allocations" for flag in user_args
    ):
        return  # the user's explicit iree_runtime_args wins
    try:
        import iree.runtime as rt

        rt.flags.parse_flags("--cuda_async_allocations=false")
    except Exception:
        pass  # flag unavailable — the anchor alone still helps


def _apply_iree_runtime_args(runtime_args: list[str] | None) -> None:
    """Parse user-provided iree runtime/loader flags (LOUD — never swallowed).

    ``iree.runtime.flags.parse_flags`` may be called multiple times per
    process (empirically verified) and parses AFTER ``IREE_PY_RUNTIME_FLAGS``
    (parsed at iree import), so the explicit option wins by last-wins flag
    semantics. A flag the runtime rejects raises ``core.BackendError`` naming
    the option, the flags, and the iree error — per etl's error strategy, a
    user's flag value is never silently dropped.
    """
    if not runtime_args:
        return
    import iree.runtime as rt

    try:
        rt.flags.parse_flags(*runtime_args)
    except Exception as exc:
        raise core.BackendError(
            f"the {IreeBackend.name} backend could not parse the "
            f"'iree_runtime_args' load option {runtime_args!r}: {exc} — the "
            "iree runtime rejected the flags; remove or fix them. Note: "
            "IREE_PY_RUNTIME_FLAGS is parsed at iree import time; explicit "
            "iree_runtime_args are parsed after it and win on conflict"
        ) from exc

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.ir import Module  # noqa: F401

__all__ = ["IreeBackend", "IreeExecutable", "iree_backend", "register"]


def _dtype_capabilities() -> frozenset:
    """The dtype capability set (core dtype constants — validated end-to-end).

    Kept as a factory so the module body performs no numpy import; ``core``
    dtype constants ARE numpy dtype objects.
    """
    return frozenset(
        {
            core.float16,
            core.float32,
            core.float64,
            core.int8,
            core.int16,
            core.int32,
            core.int64,
            core.bool_,
        }
    )


def _iree_distribution_version(*distribution_names: str) -> str:
    """Version of the first installed IREE distribution (diagnostics only).

    IREE renamed its PyPI distributions: ``iree-compiler`` →
    ``iree-base-compiler`` and ``iree-runtime`` → ``iree-base-runtime`` (the
    old names are deprecated and stale on PyPI — last released
    20241104.1068; the Python namespaces ``iree.compiler`` /
    ``iree.runtime`` are unchanged by the rename). Either distribution set
    may be installed, so probe the current names first and fall back to the
    old names for environments still on the stale packages. Returns
    ``"unknown"`` when none are found (``compile`` only reaches here after
    ``check_available`` succeeded on the Python namespaces).
    """
    import importlib.metadata as metadata

    for name in distribution_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


class IreeBackend(CompilerBackend):
    """IREE compiler backend: shared StableHLO lowering + IREE compilation.

    ``lower`` is inherited from ``CompilerBackend`` (verify -> capability
    pre-check -> portable inlining -> verify -> StableHLO export -> signature
    -> MLIR-text payload). This class adds the compiler-specific half:

    - ``check_available`` — probes the IREE packages; missing deps raise
      ``core.BackendError`` with the install hint ``pip install etl[iree]``.
    - ``compile`` — invokes ``iree.compiler.compile_str`` on the MLIR payload
      (targets from the ``target_backends`` compile option, default
      ``["llvm-cpu"]``; ``"cuda"`` supported) and records the VM flatbuffer
      (base64, JSON-safe) in the artifact.
    - ``load`` — decodes the flatbuffer, validates artifact/device target
      compatibility, acquires the runtime device (``cpu`` via the
      ``local-task`` driver; ``cuda`` via the ``cuda`` driver with the
      1-based device-id mapping), and builds an ``IreeExecutable``.

    Staging never composes: ``compile`` never loads; ``load`` never
    re-lowers/re-compiles.
    """

    name = "iree"

    #: Options-override contract (see ../options.py): the compile option
    #: ``iree_compile_args`` (arbitrary iree-compile flags — etl's minimal
    #: defaults are overridable by name collision, a small documented deny
    #: list guards the genuinely required flags) and ``target_backends``;
    #: the load option ``iree_runtime_args`` (iree runtime/loader flags,
    #: parsed loudly at load). No run options in v1 (a non-empty run options
    #: dict raises BackendError). The reserved exporter lower options
    #: ``sort_emission``/``while_init_rewrite`` are consumed by the shared
    #: CompilerBackend (see ../compiler.py) and declared here so they pass
    #: validation.
    KNOWN_OPTIONS: dict[str, frozenset[str]] = {
        "lower": frozenset(
            {"rng_bit_generator", "sort_emission", "while_init_rewrite"}
        ),
        "compile": frozenset({"target_backends", "iree_compile_args"}),
        "load": frozenset({"iree_runtime_args"}),
        "run": frozenset(),
    }
    # "auto" sort emission: the iree-cuda HAL cannot bufferize multi-operand
    # `stablehlo.sort` whenever the sorted-axis extent is >= 32 (upstream
    # iree 3.11.0 bug) — the count-based composition (bit-exact vs numpy on
    # both targets) is used for those argsorts, the two-operand sort for the
    # rest. Explicit per-call override: `lower(..., sort_emission="pair"|"count")`.
    default_sort_emission = "auto"
    capabilities = Capabilities(
        dynamic_shapes=True,
        dtypes=_dtype_capabilities(),
        collectives=False,
        runtime_calls=False,
        custom_blocks=False,
        async_collectives=False,
        # Round 2: adapter host-dispatch for ``external_call`` — graphs
        # containing the op are split at lower() time into compiled segments
        # (``CompilerBackend._lower_external_segments`` + ``external_split``)
        # and the kernel calls are dispatched on the HOST between segment
        # runs (staging via the existing device-resident Tensor paths). See
        # ``etl/CONTEXT.md`` "External kernels" and ``IreeExternalExecutable``.
        external_calls=True,
        # {"threefry2x32"} (measured): native THREE_FRY via
        # stablehlo.rng_bit_generator is verified BIT-EXACT vs numpy on
        # llvm-cpu AND cuda (iree 3.11.0) and faster than the inline
        # expansion (benchmark in etl/bench/rng_bench.py: ~1.6-2.1x, e.g.
        # cuda randint 2^24: 3.84 ms native vs 8.16 ms inline; uniform
        # 4.27 vs 7.64; llvm-cpu uniform 2^24: 100 vs 164). PHILOX is
        # excluded because iree cannot legalize RNG_ALG_PHILOX on either
        # target (see adapters/CONTEXT.md Known Issues) — philox graphs
        # use the exporter's bit-exact inline expansion.
        rng_bit_generator=frozenset({"threefry2x32"}),
    )

    @classmethod
    def check_available(cls) -> None:
        """Probe the IREE compiler/runtime packages (function-body imports).

        Raises ``core.BackendError`` with the install hint
        ``pip install etl[iree]`` when either package is missing; does
        nothing when available. Called from ``compile`` / ``load`` / the
        module ``register()`` so a missing dependency fails with an
        actionable message instead of an obscure ``ImportError`` deep inside
        the vendor API.
        """
        try:
            import iree.compiler  # noqa: F401
            import iree.runtime  # noqa: F401
        except ImportError as exc:
            raise core.BackendError(
                f"the {cls.name} backend requires the IREE Python packages "
                f"(iree-base-compiler + iree-base-runtime), which are "
                f"unavailable: {exc}. Install them with "
                f"`pip install etl[iree]`"
            ) from exc

    def compile(
        self, lowered: LoweredProgram, options: dict | None = None
    ) -> CompiledArtifact:
        """Compile a shared StableHLO ``LoweredProgram`` into an IREE artifact.

        1. Validate ``lowered.backend == "iree"`` (``core.BackendError``
           naming the producing backend otherwise — never cross-backend
           compilation).
        2. Validate the payload is the shared StableHLO record (dict with
           ``format == "stablehlo"`` and a ``mlir_text`` string; a malformed
           payload raises ``core.BackendError`` "corrupt").
        3. ``check_available``.
        4. Validate options against ``KNOWN_OPTIONS`` (unknown keys =>
           ``core.BackendError``; keys valid for other stages pass). Resolve
           the compile ``target_backends`` option (default ``["llvm-cpu"]``;
           v1 supported set ``{"llvm-cpu", "cuda"}`` — a non-list/tuple, an
           empty list, or an unknown target raises ``core.BackendError``
           naming the offending value, never a silent fallback) and the
           ``iree_compile_args`` option (arbitrary iree-compile flags —
           list/tuple of flag strings; the values are never validated by
           etl, the compiler validates them). ``iree.compiler.compile_str``
           with those targets, ``input_type="stablehlo"`` and
           ``extra_args`` = etl's MINIMAL defaults (f64 semantics preserved;
           portable generic-CPU codegen; dynamically-linked llvm-cpu link —
           see the module constants) MINUS any default a user flag overrides
           by name, PLUS the user's flags (deny-checked against the small
           documented ``_IREE_DENIED_COMPILE_FLAGS`` list — genuinely
           required flags fail with an actionable message; every other flag
           passes through). ``iree.compiler.CompilerToolError`` is re-raised
           as ``core.BackendError`` CARRYING the compiler diagnostics —
           honest, never silent.
        5. Record a self-describing ``CompiledArtifact``: JSON-safe payload
           (``format == "iree-vmfb"``, the MLIR text, the base64 VM
           flatbuffer, the entry-function names, and the ``target_backends``
           tuple — ``load`` validates artifact/device compatibility against
           it), ``required_custom_ops=()`` (the shared ``lower`` already
           inlined every portable block), and ``runtime_dependencies`` (numpy
           + IREE package versions). ``CompiledArtifact.target`` is ``"cpu"``
           for ``["llvm-cpu"]``, ``"cuda"`` for ``["cuda"]``, and the
           ``"+"``-joined targets for mixed lists.

        ``compile`` NEVER loads (no runtime is touched here).
        """
        if lowered.backend != self.name:
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the {self.name} backend — never "
                "silently re-lower"
            )
        payload = lowered.payload
        if isinstance(payload, dict) and payload.get("format") == "stablehlo-segments":
            return self._compile_segments(lowered, payload, options)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "stablehlo"
            and isinstance(payload.get("mlir_text"), str)
        ):
            raise core.BackendError(
                f"the {self.name} backend cannot compile the lowered payload "
                f"of type {type(payload).__name__} — corrupt: expected the "
                "shared StableHLO record (dict with format='stablehlo' and "
                "an 'mlir_text' string)"
            )
        self.check_available()

        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "compile")

        import numpy as np

        entry_functions = tuple(payload.get("entry_functions", ()))
        if not entry_functions or not all(
            isinstance(name, str) for name in entry_functions
        ):
            raise core.BackendError(
                f"corrupt: the {self.name} lowered payload records no "
                "entry functions"
            )
        mlir_text = payload["mlir_text"]
        target_backends, extra_args = self._resolve_compile_options(options)
        vmfb = self._compile_mlir(mlir_text, target_backends, extra_args)

        artifact_target = _artifact_target(target_backends)
        artifact_payload = {
            "format": "iree-vmfb",
            "format_version": 1,
            "mlir_text": mlir_text,
            "vmfb_base64": base64.b64encode(vmfb).decode("ascii"),
            "entry_functions": entry_functions,
            "target_backends": target_backends,
        }
        return CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target=artifact_target,
            payload=artifact_payload,
            required_custom_ops=(),
            runtime_dependencies=_runtime_dependencies(),
        )

    # -------------------------------------------------- external-call split

    def _resolve_compile_options(
        self, options: dict | None
    ) -> tuple[tuple[str, ...], list[str]]:
        """Resolve the ``target_backends`` + ``iree_compile_args`` options.

        Compile targets come from the ``target_backends`` compile option
        (default ``["llvm-cpu"]`` — unchanged CPU behavior). v1 supported
        set: ``{"llvm-cpu", "cuda"}``. ``iree_compile_args`` carries
        arbitrary iree-compile flags (list/tuple of flag strings; values are
        NEVER validated here — the compiler validates them and its
        diagnostics surface as BackendError). etl's minimal defaults are
        dropped when a user flag overrides them by name; the small
        documented deny list guards the genuinely required flags (see module
        constants). Returns ``(target_backends, extra_args)``.
        """
        raw_targets = (options or {}).get("target_backends", ["llvm-cpu"])
        if (
            not isinstance(raw_targets, (list, tuple))
            or not raw_targets
            or not all(isinstance(target, str) for target in raw_targets)
        ):
            raise core.BackendError(
                f"the {self.name} 'target_backends' compile option must be a "
                f"non-empty list/tuple of strings, got {raw_targets!r}"
            )
        supported_targets = {"llvm-cpu", "cuda"}
        for target in raw_targets:
            if target not in supported_targets:
                raise core.BackendError(
                    f"the {self.name} backend does not support compile "
                    f"target {target!r} — supported targets: "
                    f"{', '.join(sorted(supported_targets))}"
                )
        target_backends = tuple(raw_targets)
        raw_extra = (options or {}).get("iree_compile_args")
        if raw_extra is None:
            user_args: list[str] = []
        elif isinstance(raw_extra, (list, tuple)) and all(
            isinstance(flag, str) for flag in raw_extra
        ):
            user_args = list(raw_extra)
        else:
            raise core.BackendError(
                f"the {self.name} 'iree_compile_args' compile option must be "
                f"a list/tuple of flag strings (e.g. "
                f'["--iree-llvmcpu-target-cpu=native"]), got {raw_extra!r}'
            )
        extra_args = _resolve_iree_compile_args(target_backends, user_args)
        return target_backends, extra_args

    def _compile_mlir(
        self, mlir_text: str, target_backends: tuple[str, ...], extra_args: list[str]
    ) -> bytes:
        """Invoke iree-compile on one MLIR text; diagnostics -> BackendError."""
        import iree.compiler as iree_compiler

        try:
            return iree_compiler.compile_str(
                mlir_text,
                target_backends=list(target_backends),
                input_type="stablehlo",
                extra_args=extra_args,
            )
        except iree_compiler.CompilerToolError as exc:
            raise core.BackendError(
                f"iree-compile failed to compile the StableHLO program for "
                f"the {self.name} backend (targets: "
                f"{', '.join(target_backends)}):\n{exc}"
            ) from exc

    def _compile_segments(
        self, lowered: LoweredProgram, payload: dict, options: dict | None
    ) -> CompiledArtifact:
        """Compile a split external-call payload (``stablehlo-segments``).

        Every segment's MLIR text is compiled to its own VM flatbuffer with
        the SAME ``target_backends`` / ``iree_compile_args`` options as the
        single-program path; the artifact records the per-segment vmfbs plus
        the kernel-call plan (JSON-safe, so the artifact persists without
        re-lowering). The artifact format is ``iree-vmfb-segments`` — ``load``
        reconstructs an ``IreeExternalExecutable`` from it.
        """
        self.check_available()

        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "compile")

        raw_segments = payload.get("segments")
        plan = payload.get("plan")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise core.BackendError(
                f"corrupt: the {self.name} split payload records no segments"
            )
        if not isinstance(plan, dict):
            raise core.BackendError(
                f"corrupt: the {self.name} split payload records no plan"
            )
        from ..external_split import decode_plan

        try:
            decode_plan(plan)
        except core.BackendError as exc:
            raise core.BackendError(
                f"corrupt: the {self.name} split payload has a malformed "
                f"plan: {exc}"
            ) from exc

        target_backends, extra_args = self._resolve_compile_options(options)
        segments = []
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, dict) or not isinstance(
                raw.get("mlir_text"), str
            ):
                raise core.BackendError(
                    f"corrupt: the {self.name} split payload segment "
                    f"{index} is malformed (expected dict with an "
                    "'mlir_text' string)"
                )
            entry_functions = tuple(raw.get("entry_functions", ()))
            if not entry_functions or not all(
                isinstance(name, str) for name in entry_functions
            ):
                raise core.BackendError(
                    f"corrupt: the {self.name} split payload segment "
                    f"{index} records no entry functions"
                )
            vmfb = self._compile_mlir(
                raw["mlir_text"], target_backends, extra_args
            )
            segments.append(
                {
                    "vmfb_base64": base64.b64encode(vmfb).decode("ascii"),
                    "entry_functions": entry_functions,
                    "target_backends": target_backends,
                }
            )
        artifact_payload = {
            "format": "iree-vmfb-segments",
            "format_version": 1,
            "segments": segments,
            "plan": plan,
        }
        return CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target=_artifact_target(target_backends),
            payload=artifact_payload,
            required_custom_ops=(),
            runtime_dependencies=_runtime_dependencies(),
        )

    def load(
        self,
        artifact: CompiledArtifact,
        device: Device | None = None,
        options: dict | None = None,
    ) -> "IreeExecutable":
        """Reconstruct an ``IreeExecutable`` from an artifact. Never re-compiles.

        1. Validate ``artifact.backend == "iree"`` (mismatch =>
           ``core.PersistenceError`` naming both — artifacts are never
           silently reinterpreted).
        2. Validate the payload (``format == "iree-vmfb"``, base64 vmfb,
           entry functions; malformed => ``core.BackendError`` "corrupt").
        3. ``check_available``.
        4. Validate options against ``KNOWN_OPTIONS`` (unknown keys =>
           ``core.BackendError``) and parse the ``iree_runtime_args`` load
           option (list/tuple of iree runtime/loader flags, e.g.
           ``["--cuda_async_allocations=false"]``) via
           ``iree.runtime.flags.parse_flags`` — the explicit option is parsed
           AFTER ``IREE_PY_RUNTIME_FLAGS`` (parsed at iree import) and wins
           by last-wins semantics; a flag the runtime rejects raises
           ``core.BackendError`` (never silent). etl's own default
           ``--cuda_async_allocations=false`` (cuda only) is applied ONLY
           when neither the explicit option nor the env mentions it — the
           hardcoded-flag lesson: every etl default is overridable.
        5. Validate the device: ``None`` (CPU default) or a ``core.Device``
           of kind ``"cpu"`` or ``"cuda"`` — a non-``Device`` object raises
           ``core.DeviceError``; another kind raises ``core.BackendError``
           naming it.
        6. Artifact/device target compatibility — never silent: the artifact
           payload's ``target_backends`` must include the driver matching the
           requested device kind (``"llvm-cpu"`` for cpu, ``"cuda"`` for
           cuda); absent ``target_backends`` (artifacts saved before CUDA
           support) is treated as ``["llvm-cpu"]``. A mismatch raises
           ``core.BackendError`` naming both the artifact's compile targets
           and the requested device.
        7. Base64-decode the VM flatbuffer; acquire the runtime device. CPU:
           the reliable path (``rt.get_driver("local-task")`` +
           ``create_default_device()``) then ``load_vm_flatbuffer``. CUDA:
           ``rt.get_driver("cuda")`` + ``driver.create_device(device_id=
           index+1)`` (IREE cuda ids are 1-BASED: etl index N -> device_id
           N+1; the driver's ``ValueError`` for an out-of-range index is
           surfaced as ``core.BackendError`` naming the index) then the
           ``rt.Config(device=...)`` + ``rt.VmModule.copy_buffer(config.
           vm_instance, vmfb)`` + ``rt.load_vm_module(vm_module, config)``
           path (the ``load_vm_flatbuffer`` driver-name form always creates
           the DEFAULT device and cannot select a CUDA index). Then resolve
           the v1 entry function (``"main"``, else the single recorded
           entry). A ``ValueError`` from the CPU ``load_vm_flatbuffer`` (the
           runtime VM verifier rejecting the module's required features — see
           ``adapters/CONTEXT.md`` Known Issue #7, the mixed-install trap) is
           surfaced as an actionable ``core.BackendError`` naming the cause
           and remedy, with the original exception as ``__cause__``.

        NEVER re-traces / re-lowers / re-compiles — load-time mismatches fail
        clearly (root error strategy).
        """
        if artifact.backend != self.name:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                f"the {self.name} backend cannot load it — never silently "
                "recompile"
            )
        payload = artifact.payload
        if isinstance(payload, dict) and payload.get("format") == "iree-vmfb-segments":
            return self._load_segments(artifact, payload, device, options)
        if not (
            isinstance(payload, dict)
            and payload.get("format") == "iree-vmfb"
            and isinstance(payload.get("vmfb_base64"), str)
        ):
            raise core.BackendError(
                f"corrupt: the {self.name} artifact payload must be the "
                "iree-vmfb record (dict with format='iree-vmfb' and a "
                "'vmfb_base64' string)"
            )
        entry_functions = tuple(payload.get("entry_functions", ()))
        if not entry_functions:
            raise core.BackendError(
                f"corrupt: the {self.name} artifact records no entry "
                "functions"
            )
        self.check_available()
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "load")
        # ``iree_runtime_args`` — iree runtime/loader flags (list/tuple of
        # flag strings; values are NEVER validated by etl — the runtime
        # validates them and its rejection surfaces as BackendError below).
        raw_runtime = (options or {}).get("iree_runtime_args")
        if raw_runtime is None:
            runtime_args: list[str] = []
        elif isinstance(raw_runtime, (list, tuple)) and all(
            isinstance(flag, str) for flag in raw_runtime
        ):
            runtime_args = list(raw_runtime)
        else:
            raise core.BackendError(
                f"the {self.name} 'iree_runtime_args' load option must be a "
                f"list/tuple of flag strings (e.g. "
                f'["--cuda_async_allocations=false"]), got {raw_runtime!r}'
            )
        _apply_iree_runtime_args(runtime_args)
        device_kind = "cpu"  # device=None = default CPU device
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got "
                    f"{type(device).__name__}"
                )
            if device.kind not in ("cpu", "cuda"):
                raise core.BackendError(
                    f"the {self.name} backend supports cpu and cuda devices "
                    f"only, got device kind {device.kind!r}"
                )
            device_kind = device.kind

        # Artifact/device target compatibility — never silent: the artifact's
        # compile targets must include the driver matching the requested
        # device kind. Absent target_backends (artifacts saved before CUDA
        # support) is treated as ["llvm-cpu"].
        raw_targets = payload.get("target_backends")
        if raw_targets is None:
            target_backends = ("llvm-cpu",)
        elif isinstance(raw_targets, str):
            target_backends = (raw_targets,)
        else:
            target_backends = tuple(raw_targets)
        expected_target = "llvm-cpu" if device_kind == "cpu" else "cuda"
        if expected_target not in target_backends:
            raise core.BackendError(
                f"the {self.name} artifact was compiled for target(s) "
                f"{', '.join(target_backends) or '(none)'} and cannot run on "
                f"the requested device {device!r} (device kind "
                f"{device_kind!r} requires compile target {expected_target!r})"
                f" — never silently recompile"
            )

        import iree.runtime as rt

        vmfb = base64.b64decode(payload["vmfb_base64"].encode("ascii"))
        runtime_device = _acquire_runtime_device(device_kind, device, runtime_args)
        module = _load_vm_module(vmfb, device_kind, runtime_device)
        return IreeExecutable(
            artifact=artifact,
            signature=artifact.signature,
            device=device,
            module=module,
            entry_functions=entry_functions,
            runtime_device=runtime_device,
        )

    def _load_segments(
        self,
        artifact: CompiledArtifact,
        payload: dict,
        device: Device | None,
        options: dict | None,
    ) -> "IreeExternalExecutable":
        """Reconstruct an ``IreeExternalExecutable`` from a split artifact.

        The artifact records one VM flatbuffer per segment plus the kernel-
        call plan (``format == "iree-vmfb-segments"``). The device is
        acquired ONCE and every segment module is loaded onto it — segment
        modules are ordinary compiled functions (no ``external_call``
        inside), so the same load path as the single-program case applies
        per segment. The plan is decoded and validated here (malformed =>
        ``core.BackendError`` "corrupt"); the executable interprets it at
        run time.
        """
        self.check_available()
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "load")
        raw_runtime = (options or {}).get("iree_runtime_args")
        if raw_runtime is None:
            runtime_args: list[str] = []
        elif isinstance(raw_runtime, (list, tuple)) and all(
            isinstance(flag, str) for flag in raw_runtime
        ):
            runtime_args = list(raw_runtime)
        else:
            raise core.BackendError(
                f"the {self.name} 'iree_runtime_args' load option must be a "
                f"list/tuple of flag strings (e.g. "
                f'["--cuda_async_allocations=false"]), got {raw_runtime!r}'
            )
        _apply_iree_runtime_args(runtime_args)
        device_kind = "cpu"  # device=None = default CPU device
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got "
                    f"{type(device).__name__}"
                )
            if device.kind not in ("cpu", "cuda"):
                raise core.BackendError(
                    f"the {self.name} backend supports cpu and cuda devices "
                    f"only, got device kind {device.kind!r}"
                )
            device_kind = device.kind
        expected_target = "llvm-cpu" if device_kind == "cpu" else "cuda"

        raw_segments = payload.get("segments")
        plan = payload.get("plan")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise core.BackendError(
                f"corrupt: the {self.name} split artifact records no segments"
            )
        if not isinstance(plan, dict):
            raise core.BackendError(
                f"corrupt: the {self.name} split artifact records no plan"
            )
        from ..external_split import decode_plan

        try:
            decoded_plan = decode_plan(plan)
        except core.BackendError as exc:
            raise core.BackendError(
                f"corrupt: the {self.name} split artifact has a malformed "
                f"plan: {exc}"
            ) from exc

        runtime_device = _acquire_runtime_device(device_kind, device, runtime_args)
        segment_execs: list[_IreeSegmentExecutable] = []
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, dict) or not isinstance(
                raw.get("vmfb_base64"), str
            ):
                raise core.BackendError(
                    f"corrupt: the {self.name} split artifact segment "
                    f"{index} is malformed (expected dict with a "
                    "'vmfb_base64' string)"
                )
            entry_functions = tuple(raw.get("entry_functions", ()))
            if not entry_functions:
                raise core.BackendError(
                    f"corrupt: the {self.name} split artifact segment "
                    f"{index} records no entry functions"
                )
            # Per-segment artifact/device target compatibility (same rule as
            # the single-program path).
            raw_targets = raw.get("target_backends")
            if raw_targets is None:
                segment_targets = ("llvm-cpu",)
            elif isinstance(raw_targets, str):
                segment_targets = (raw_targets,)
            else:
                segment_targets = tuple(raw_targets)
            if expected_target not in segment_targets:
                raise core.BackendError(
                    f"the {self.name} split artifact segment {index} was "
                    f"compiled for target(s) "
                    f"{', '.join(segment_targets) or '(none)'} and cannot "
                    f"run on the requested device {device!r} (device kind "
                    f"{device_kind!r} requires compile target "
                    f"{expected_target!r}) — never silently recompile"
                )
            vmfb = base64.b64decode(raw["vmfb_base64"].encode("ascii"))
            module = _load_vm_module(vmfb, device_kind, runtime_device)
            core_device = device if device is not None else Device("cpu", 0)
            input_specs = tuple(
                core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
                for value_type in decoded_plan[index]["input_specs"]
            )
            segment_execs.append(
                _IreeSegmentExecutable(
                    module=module,
                    runtime_device=runtime_device,
                    core_device=core_device,
                    input_specs=input_specs,
                    cuda=(device_kind == "cuda"),
                )
            )
        return IreeExternalExecutable(
            artifact=artifact,
            signature=artifact.signature,
            device=device,
            segments=segment_execs,
            plan=decoded_plan,
            runtime_device=runtime_device,
        )


class IreeExecutable(CompilerExecutable):
    """IREE runtime executable (satisfies the backend ``Executable`` protocol).

    Runs the compiled VM module on the acquired IREE HAL device (``local-task``
    CPU or ``cuda`` GPU):

    1. Validate flat inputs against the recorded ``Signature``: count
       (``core.BackendError``), per-input dtype (``core.BackendError``),
       per-input shape — rank and STATIC dims must match
       (``core.ShapeError``); symbolic dims pass through (IREE executes
       runtime-dynamic shapes — validated).
    2. Per input: a device-resident tensor whose payload is an IREE
       ``DeviceArray`` on THIS executable's device (kind+index) is passed
       to the invoke directly — no host round-trip, no copy. Any other
       input (numpy-backed host tensor, payload on a different device) is
       staged on cuda via ``_staged_input`` (persistent host-local source +
       per-call DMA queue_copy into a persistent device staging buffer) or
       copied into a HAL buffer via ``rt.asdevicearray`` (fallbacks) exactly
       as before.
    3. Invoke the entry function; handle single/multiple results (the
       invoker returns the value or a tuple).
    4. Wrap each result in ``core.Tensor(IreeDevicePayload(...))`` — a
       DEVICE-RESIDENT tensor: ``.data`` is the payload (the DeviceArray
       already on the device), ``.device`` is this executable's
       ``core.Device`` (``Device("cpu", 0)`` for the default CPU device),
       and ``.numpy()`` performs a LAZY host copy (``DeviceArray.to_host()``)
       on demand — the run() hot path never touches host memory. This holds
       for both cuda AND llvm-cpu executables (on ``local-task`` the to_host
       mapping is zero-copy for mappable buffers).
    5. Validate output count + dtype against the signature output specs.

    ``save`` / ``load`` are inherited from ``CompilerExecutable``: ``save``
    persists the underlying ``CompiledArtifact`` (device handles are NEVER
    serialized — reconstruction is explicit); ``load`` reads the artifact,
    validates its backend, and routes reconstruction through the registry
    (which auto-activates this adapter lazily).
    """

    backend_name = "iree"

    # Run-stage option validation table — mirrors ``IreeBackend.KNOWN_OPTIONS``
    # (single source of truth; the class attribute is defined here so the
    # executable validates independently of the backend instance).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = IreeBackend.KNOWN_OPTIONS

    def __init__(
        self,
        artifact: CompiledArtifact,
        signature: Signature | None,
        device: core.Device | None,
        module: Any,
        entry_functions: tuple[str, ...],
        runtime_device: Any,
    ) -> None:
        super().__init__(
            artifact=artifact,
            signature=signature,
            device=device,
            native_module=module,
            entry_functions=entry_functions,
        )
        self.runtime_device = runtime_device
        # Per-call invariants cached at load time (never re-derived per run):
        # the core.Device label of this executable (output tensors + the
        # same-device pass-through rule), the flat spec tuples, and the
        # resolved entry function (resolution constructs a new
        # ``FunctionInvoker`` — ABI JSON parse + argument packer — on every
        # BoundModule attribute access, so it must happen once).
        self._core_device: Device = (
            device if device is not None else Device("cpu", 0)
        )
        self._input_specs = (
            tuple(signature.input_specs) if signature is not None else ()
        )
        self._output_specs = (
            tuple(signature.output_specs) if signature is not None else ()
        )
        self._entry: Any = None
        # CUDA per-call fast path (see module-level notes): a retained
        # device-local anchor buffer keeps the allocator pool warm so per-call
        # result allocation/free stays ~0.04 ms; the staged-input cache holds
        # persistent host-local source + device staging buffers per
        # (index, shape, dtype).
        self._pool_anchor: Any = None
        self._staged_inputs: dict | None = None
        if device is not None and device.kind == "cuda":
            import iree.runtime as rt

            self._staged_inputs = {}
            try:
                self._pool_anchor = runtime_device.allocator.allocate_buffer(
                    rt.MemoryType.DEVICE_LOCAL, rt.BufferUsage.DEFAULT, 4096
                )
            except Exception:
                self._pool_anchor = None  # best-effort perf anchor, never fatal

    # ------------------------------------------------------------------ run

    def run(
        self,
        flat_input_tensors: list[core.Tensor],
        options: dict | None = None,
    ) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        See the class docstring for the exact validation/invocation model.
        Runtime IREE errors propagate as-is (they are explicit hardware/runtime
        failures, never silently swallowed); etl-level mismatches raise
        ``core.BackendError`` / ``core.ShapeError``.

        ``options``: per-run options, validated against ``KNOWN_OPTIONS`` —
        the iree run stage has NO known options in v1, so any non-empty
        options dict raises ``core.BackendError`` (loud; never silently
        swallowed). Runtime/loader flags are load-time options
        (``iree_runtime_args``), not per-run.
        """
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.backend_name, "run")

        input_specs = self._input_specs
        output_specs = self._output_specs
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"program expects {len(input_specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        for i, tensor in enumerate(flat_input_tensors):
            if not isinstance(tensor, core.Tensor):
                raise core.BackendError(
                    f"input {i} must be a core.Tensor, got "
                    f"{type(tensor).__name__}"
                )
            spec = input_specs[i]
            if tensor.dtype != spec.dtype:
                raise core.BackendError(
                    f"input {i}: expected dtype {spec.dtype}, got "
                    f"{tensor.dtype} — never silently coerce"
                )
            _validate_input_shape(i, spec.shape, tensor.shape)

        buffers = _build_buffers(
            flat_input_tensors,
            input_specs,
            self.runtime_device,
            self._core_device,
            self._staged_inputs,
        )
        entry = self._entry_function()
        results = entry(*buffers)
        if results is None:
            results = ()
        elif not isinstance(results, tuple):
            results = (results,)
        # Device-resident outputs: wrap the returned DeviceArrays (already on
        # the device) in core.Tensor with the lazy IreeDevicePayload — the
        # D2H copy happens only if/when .numpy() is called.
        core_device = self._core_device
        outputs = [
            core.Tensor(IreeDevicePayload(result, core_device))
            for result in results
        ]

        if self.signature is not None:
            if len(outputs) != len(output_specs):
                raise core.BackendError(
                    f"program produced {len(outputs)} output tensor(s), "
                    f"expected {len(output_specs)}"
                )
            for i, (tensor, spec) in enumerate(zip(outputs, output_specs)):
                if tensor.dtype != spec.dtype:
                    raise core.BackendError(
                        f"output {i}: expected dtype {spec.dtype}, got "
                        f"{tensor.dtype} — never silently coerce"
                    )
        return outputs

    # ------------------------------------------------------------- internals

    def _entry_function(self) -> Any:
        """Resolve the v1 entry function: ``"main"``, else the single one.

        Resolved ONCE and cached: resolution is a ``BoundModule`` attribute
        access that constructs a fresh ``FunctionInvoker`` (ABI JSON parse +
        argument packer) every time — per-call re-resolution would redo that
        work on every run.

        Raises:
            core.BackendError: no ``"main"`` function and not exactly one
                recorded entry function.
        """
        if self._entry is not None:
            return self._entry
        if "main" in self._entry_functions:
            entry = getattr(self.native_module, "main")
        elif len(self._entry_functions) == 1:
            entry = getattr(self.native_module, self._entry_functions[0])
        else:
            raise core.BackendError(
                f"the {self.backend_name} executable has "
                f"{len(self._entry_functions)} entry functions and no 'main' "
                "entry function"
            )
        self._entry = entry
        return entry

def _staged_input(
    staged_inputs: dict | None,
    runtime_device: Any,
    index: int,
    tensor: core.Tensor,
) -> Any | None:
    """Upload a host tensor through persistent staging buffers.

    CUDA fast path replacing ``rt.asdevicearray`` for host inputs, in
    two tiers (both fully synchronous — nothing is left racing):

    * TINY inputs (nbytes <= ``_PINNED_DIRECT_MAX_BYTES``): the values
      are memcpy'd (``np.copyto``) into the mapped view of a persistent
      DEVICE_LOCAL | HOST_VISIBLE ("pinned") buffer that the dispatches
      read DIRECTLY — no queue_copy, no semaphore, no wait. The driver
      serializes the host write against device reads of the mapped
      memory, so the dispatch always sees the new values; that same
      serialization is what makes large pinned uploads slow (hence the
      cutoff — measured 0.5-1.2 ms at 160 KB vs ~0.05-0.06 ms at 8 B
      including the invoke, and the pinned tier does NOT degrade when
      the upload follows another invoke, unlike the DMA tier's wait).
    * LARGER inputs: the values are memcpy'd into the mapped view of a
      persistent HOST_LOCAL (plain host memory) source buffer, then a
      per-call ``queue_copy`` (DMA) transfers them into a persistent
      DEVICE_LOCAL staging buffer the dispatches read, waiting on a
      FRESH semaphore signaled by the copy before returning. The fresh
      semaphore is REQUIRED (reusing a waited event-semaphore aborts
      the cuda HAL, event_semaphore.c:354, on the second use) and the
      wait is REQUIRED for correctness: the copy and the invoke's
      dispatches are NOT reliably ordered on the cuda HAL — dropping
      the wait races the dispatch against the copy (~0.4 % of runs
      return garbage under alternating-value stress).

    The DeviceArray over the staging/pinned buffer is built ONCE per
    (input index, shape, dtype) and cached in ``staged_inputs`` (per
    executable); every call re-copies the current host values into the
    same mapped view (runs are synchronous, so reuse cannot race).

    Why not use the pinned tier for everything? Measured on IREE
    3.11.0 / RTX A6000: host writes into host-mapped DEVICE memory go
    over PCIe write-through AND the driver serializes a host write
    against outstanding device reads of that memory (~0.3-0.6 ms stall
    per call at 160 KB), while HOST_LOCAL writes are a plain cached
    memcpy (~0.01 ms for 160 KB) and the DMA queue_copy is ~0.01 ms —
    the DMA tier tracks the pure invoke cost at scale.

    Falls back to ``None`` (caller uses ``asdevicearray``) for
    non-cuda executables (``staged_inputs is None``), oversized inputs,
    unmappable dtypes, and a full cache — semantics identical either way.
    """
    if staged_inputs is None:
        return None
    import numpy as np
    import iree.runtime as rt
    from iree.runtime.array_interop import map_dtype_to_element_type

    dtype = tensor.dtype
    shape = tuple(tensor.shape)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes > _STAGED_INPUT_MAX_BYTES:
        return None
    element_type = map_dtype_to_element_type(dtype)
    if element_type is None:
        return None
    key = (index, shape, dtype)
    cached = staged_inputs.get(key)
    if cached is not None:
        kind, payload = cached
        if kind == "pin":
            da, view = payload
            np.copyto(view, tensor.numpy())
            return da
        hl_buf, hl_view, staging_da = payload
        np.copyto(hl_view, tensor.numpy())
        sem = runtime_device.create_semaphore(0)
        runtime_device.queue_copy(
            hl_buf,
            staging_da._buffer_view.get_buffer(),
            rt.HalFence.create_at(sem, 0),
            rt.HalFence.create_at(sem, 1),
        )
        rt.HalFence.create_at(sem, 1).wait()
        return staging_da
    if len(staged_inputs) >= _STAGED_INPUT_CACHE_MAX:
        staged_inputs.clear()  # bounded cache; correctness unaffected
    if nbytes <= _PINNED_DIRECT_MAX_BYTES:
        # Pinned direct-read tier: dispatch reads the host-mapped
        # DEVICE_LOCAL buffer directly; the driver's write/read
        # serialization orders the host write before the dispatch.
        buf = runtime_device.allocator.allocate_buffer(
            rt.MemoryType.DEVICE_LOCAL | rt.MemoryType.HOST_VISIBLE,
            rt.BufferUsage.DEFAULT | rt.BufferUsage.MAPPING,
            nbytes,
        )
        view = buf.map().asarray(shape, np.dtype(dtype))
        da = rt.DeviceArray(
            runtime_device,
            rt.HalBufferView(buf, shape, element_type),
            implicit_host_transfer=False,
            override_dtype=dtype,
        )
        np.copyto(view, tensor.numpy())
        staged_inputs[key] = ("pin", (da, view))
        return da
    # DMA tier: host-local source (plain host memory — cached writes) +
    # persistent device staging buffer (dispatch storage).
    hl_buf = runtime_device.allocator.allocate_buffer(
        rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE,
        rt.BufferUsage.TRANSFER_SOURCE | rt.BufferUsage.MAPPING,
        nbytes,
    )
    hl_view = hl_buf.map().asarray(shape, np.dtype(dtype))
    stg_buf = runtime_device.allocator.allocate_buffer(
        rt.MemoryType.DEVICE_LOCAL, rt.BufferUsage.DEFAULT, nbytes
    )
    staging_da = rt.DeviceArray(
        runtime_device,
        rt.HalBufferView(stg_buf, shape, element_type),
        implicit_host_transfer=False,
        override_dtype=dtype,
    )
    np.copyto(hl_view, tensor.numpy())
    sem = runtime_device.create_semaphore(0)
    runtime_device.queue_copy(
        hl_buf,
        staging_da._buffer_view.get_buffer(),
        rt.HalFence.create_at(sem, 0),
        rt.HalFence.create_at(sem, 1),
    )
    rt.HalFence.create_at(sem, 1).wait()
    staged_inputs[key] = ("dma", (hl_buf, hl_view, staging_da))
    return staging_da


def _build_buffers(
    flat_input_tensors: list[core.Tensor],
    input_specs: tuple,
    runtime_device: Any,
    core_device: Device,
    staged_inputs: dict | None,
) -> list:
    """Build the HAL buffers for one invoke from validated input tensors.

    Shared by ``IreeExecutable.run`` and the per-segment executor: an input
    tensor already living on THIS device (payload-backed with a matching
    ``core.Device`` — our own ``IreeDevicePayload`` (run outputs of this or
    another iree executable, unwrapped to the underlying DeviceArray) or a
    raw iree ``DeviceArray``) is handed to the invoke as-is — no host
    round-trip, no re-upload; the caller's validation loop already proved
    dtype/shape, so the payload is guaranteed correct. Anything else
    (numpy-backed host input, payload on a different device) takes the
    staged upload on cuda (``_staged_input``), else the classic
    ``asdevicearray`` H2D copy path.
    """
    import numpy as np
    import iree.runtime as rt

    buffers = []
    for i, tensor in enumerate(flat_input_tensors):
        data = tensor.data
        if isinstance(data, IreeDevicePayload) and tensor.device == core_device:
            buffers.append(data.device_array)
        elif (
            not isinstance(data, np.ndarray)
            and isinstance(data, rt.DeviceArray)
            and tensor.device == core_device
        ):
            buffers.append(data)
        else:
            staged = _staged_input(staged_inputs, runtime_device, i, tensor)
            if staged is not None:
                buffers.append(staged)
            else:
                buffers.append(rt.asdevicearray(runtime_device, tensor.numpy()))
    return buffers


class _IreeSegmentExecutable:
    """One compiled segment of an external-call graph (internal).

    A segment is an ordinary compiled function (no ``external_call``
    inside) with its own static input spec list. ``run`` mirrors
    ``IreeExecutable.run``'s validation + buffer-building + invoke + lazy
    device-resident output wrapping (shared helpers), against the segment's
    own input specs.
    """

    def __init__(
        self,
        module: Any,
        runtime_device: Any,
        core_device: Device,
        input_specs: tuple,
        cuda: bool,
    ) -> None:
        self.module = module
        self.runtime_device = runtime_device
        self.core_device = core_device
        self.input_specs = input_specs
        self._entry: Any = None
        # CUDA per-call fast path: the same persistent staging machinery as
        # IreeExecutable (one cache per segment — inputs repeat per run).
        self._staged_inputs: dict | None = None
        self._pool_anchor: Any = None
        if cuda:
            import iree.runtime as rt

            self._staged_inputs = {}
            try:
                self._pool_anchor = runtime_device.allocator.allocate_buffer(
                    rt.MemoryType.DEVICE_LOCAL, rt.BufferUsage.DEFAULT, 4096
                )
            except Exception:
                self._pool_anchor = None  # best-effort perf anchor, never fatal

    def _entry_function(self) -> Any:
        """Resolve the segment's "main" entry function (cached once)."""
        if self._entry is None:
            self._entry = getattr(self.module, "main")
        return self._entry

    def run(self, flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]:
        """Validate + invoke the segment; outputs stay DEVICE-RESIDENT."""
        input_specs = self.input_specs
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"external-call segment expects {len(input_specs)} input "
                f"tensor(s), got {len(flat_input_tensors)}"
            )
        for i, tensor in enumerate(flat_input_tensors):
            if not isinstance(tensor, core.Tensor):
                raise core.BackendError(
                    f"external-call segment input {i} must be a core.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            spec = input_specs[i]
            if tensor.dtype != spec.dtype:
                raise core.BackendError(
                    f"external-call segment input {i}: expected dtype "
                    f"{spec.dtype}, got {tensor.dtype} — never silently "
                    "coerce"
                )
            _validate_input_shape(i, spec.shape, tensor.shape)
        buffers = _build_buffers(
            flat_input_tensors,
            input_specs,
            self.runtime_device,
            self.core_device,
            self._staged_inputs,
        )
        results = self._entry_function()(*buffers)
        if results is None:
            results = ()
        elif not isinstance(results, tuple):
            results = (results,)
        return [
            core.Tensor(IreeDevicePayload(result, self.core_device))
            for result in results
        ]


class IreeExternalExecutable(CompilerExecutable):
    """IREE executable for a SPLIT external-call graph (round 2+ dispatch).

    Runs the graph's compiled segments strictly in plan order on the IREE
    HAL device; at each ``external_call`` boundary the kernel's operand
    tensors (the DEVICE-RESIDENT outputs of the producing segment — never
    pre-staged) are dispatched through the SAME name-keyed registry the
    numpy backend uses (``etl.external``; an unregistered name raises
    ``core.BackendError`` pointing at ``etl.register_external_kernel``).
    Two kernel modes:

    - HOST mode (default registration, ``device_resident=False``): the
      dispatcher stages the operands to host numpy arrays (lazy
      ``.numpy()``), calls the kernel, and its validated host results are
      staged BACK into the following segment's inputs (the existing
      host-input staging machinery uploads them).
    - DEVICE mode (``device_resident=True`` registration): the dispatcher
      passes the device tensors DIRECTLY (zero host staging on the way in);
      the kernel may return ``core.Tensor`` s, duck-typed device payloads
      (e.g. ``IreeDevicePayload``), raw iree ``DeviceArray`` s (wrapped with
      the run's ``core.Device``), numpy arrays, or tuple/list mixes of them;
      results are validated METADATA-ONLY (count/dtype/shape — never a
      forced host copy) and occupy the plan's result slots, where the next
      segment's same-device pass-through consumes device results with ZERO
      host round-trips (host-array results are staged back by the segment
      input machinery).

    Carried plain values that are NOT kernel operands stay device-resident
    across segment runs — zero extra host round-trips for them.

    Execution model / determinism:
    - Segments execute strictly sequentially in plan order; the ``callback``
      effect anchors of the original graph are preserved (a kernel call is
      never reordered against the compiled ops around it).
    - Kernels are assumed PURE (same inputs -> same outputs): the same
      inputs + the same registered kernels produce the same results.
    - A ``UserWarning`` is emitted once per executable when the FIRST
      boundary that actually stages tensors host<->device executes (a
      host-mode kernel, or device-mode results that are host-backed) —
      fully device-resident boundaries never warn; the message notes that
      ``device_resident=True`` registrations skip per-boundary host staging
      (see ``run``).
    - The graph's structured I/O contract (the recorded ``Signature``) is
      unchanged: ``run`` validates flat inputs against it and returns the
      graph's flat outputs (the final segment's module outputs).
    """

    backend_name = "iree"

    # Run-stage option validation table — mirrors ``IreeBackend.KNOWN_OPTIONS``
    # (the iree run stage has NO known options in v1).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = IreeBackend.KNOWN_OPTIONS

    def __init__(
        self,
        artifact: CompiledArtifact,
        signature: Signature | None,
        device: core.Device | None,
        segments: list[_IreeSegmentExecutable],
        plan: list[dict],
        runtime_device: Any,
    ) -> None:
        super().__init__(
            artifact=artifact,
            signature=signature,
            device=device,
            native_module=None,  # segments are held per-segment
            entry_functions=(),
        )
        self.runtime_device = runtime_device
        self._core_device: Device = (
            device if device is not None else Device("cpu", 0)
        )
        self._segments = tuple(segments)
        self._plan = tuple(plan)
        self._input_specs = (
            tuple(signature.input_specs) if signature is not None else ()
        )
        self._output_specs = (
            tuple(signature.output_specs) if signature is not None else ()
        )
        if not plan:
            raise core.BackendError(
                "corrupt: the external-call plan records no segments"
            )
        self._final_module_outputs = plan[-1]["module_outputs"]
        self._staging_warned = False  # once-per-executable staging warning

    # ------------------------------------------------------------------ run

    def run(
        self,
        flat_input_tensors: list[core.Tensor],
        options: dict | None = None,
    ) -> list[core.Tensor]:
        """Execute the split graph: per-segment device runs + kernel dispatch.

        1. Validate flat inputs against the recorded ``Signature`` (the
           original graph's I/O contract — count/dtype/shape, same rules as
           ``IreeExecutable.run``).
        2. For each segment in plan order: build its flat input list from
           (graph inputs) ∪ (carried values from earlier segment outputs —
           device-resident tensors pass through with no host round-trip) and
           run it on the device. If the segment ends in an ``external_call``:
           dispatch the registered kernel with the segment's device-resident
           operand tensors — a HOST-mode kernel (``device_resident=False``)
           receives host numpy arrays (staged here), a DEVICE-mode kernel
           (``device_resident=True``) receives the device tensors DIRECTLY
           and may return device tensors / duck payloads (wrapped via
           ``_wrap_device_result``) / host arrays; outputs are validated
           METADATA-ONLY against the declared result specs — count/dtype/
           shape, never silent coercion, never a forced host copy — and
           stored at the plan's result slots. Missing kernel =>
           ``core.BackendError`` naming the kernel and pointing at
           ``register_external_kernel``. The once-per-executable
           "host-dispatch" staging ``UserWarning`` fires ONLY when a
           boundary actually stages tensors host<->device (a host-mode
           kernel, or device-mode results that are host-backed numpy —
           ``device_resident=True`` registrations with fully device-resident
           results never warn).
        3. Return the final segment's module outputs (the graph's outputs),
           validated against the signature output specs.
        """
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.backend_name, "run")
        from ..external_split import dispatch_external_kernel  # lazy

        input_specs = self._input_specs
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"program expects {len(input_specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        for i, tensor in enumerate(flat_input_tensors):
            if not isinstance(tensor, core.Tensor):
                raise core.BackendError(
                    f"input {i} must be a core.Tensor, got "
                    f"{type(tensor).__name__}"
                )
            spec = input_specs[i]
            if tensor.dtype != spec.dtype:
                raise core.BackendError(
                    f"input {i}: expected dtype {spec.dtype}, got "
                    f"{tensor.dtype} — never silently coerce"
                )
            _validate_input_shape(i, spec.shape, tensor.shape)

        # segment_outputs[j] = the run-time output slot list of segment j:
        # slots [0, module_outputs) are the module's return values
        # (device-resident Tensors); slots [module_outputs, ...) are the
        # kernel results (device-resident for device-mode kernels, host
        # Tensors for host-mode kernels — staged back by the next segment's
        # input machinery at dispatch time).
        segment_outputs: list[list[core.Tensor]] = []
        for segment_index, (segment, plan_segment) in enumerate(
            zip(self._segments, self._plan)
        ):
            seg_inputs = []
            for entry in plan_segment["inputs"]:
                kind = entry["kind"]
                if kind == "graph":
                    seg_inputs.append(flat_input_tensors[entry["index"]])
                elif kind in ("carry", "kernel"):
                    seg_inputs.append(
                        segment_outputs[entry["segment"]][entry["output"]]
                    )
                else:
                    raise core.BackendError(
                        f"corrupt: external-call plan segment "
                        f"{segment_index} has an unknown input kind "
                        f"{kind!r}"
                    )
            outputs = segment.run(seg_inputs)
            out_slots: list[core.Tensor] = list(outputs)
            call = plan_segment.get("call")
            if call is not None:
                name = call["name"]
                label = f"op 'external_call' (kernel {name!r})"
                operand_tensors = [
                    outputs[slot]
                    for slot in call["operand_outputs"]
                ]
                results, staged = dispatch_external_kernel(
                    name,
                    operand_tensors,
                    call["result_specs"],
                    label,
                    wrap_device_result=self._wrap_device_result,
                )
                if staged and not self._staging_warned:
                    self._staging_warned = True
                    warnings.warn(
                        "iree host-dispatch: this external_call boundary "
                        "staged tensors device -> host -> device "
                        "(per-boundary host round-trips, v1 mechanism; "
                        "device_resident=True registrations skip "
                        "per-boundary host staging). Lower-time "
                        "restrictions: no control-flow bodies around calls, "
                        "static result dims, at least one tensor operand, "
                        "kernels assumed pure",
                        UserWarning,
                        stacklevel=2,
                    )
                out_slots.extend([None] * len(call["result_specs"]))
                for slot, tensor in zip(call["result_outputs"], results):
                    out_slots[slot] = tensor
            segment_outputs.append(out_slots)

        final = segment_outputs[-1]
        outputs = final[: self._final_module_outputs]
        if self.signature is not None:
            if len(outputs) != len(self._output_specs):
                raise core.BackendError(
                    f"program produced {len(outputs)} output tensor(s), "
                    f"expected {len(self._output_specs)}"
                )
            for i, (tensor, spec) in enumerate(zip(outputs, self._output_specs)):
                if tensor.dtype != spec.dtype:
                    raise core.BackendError(
                        f"output {i}: expected dtype {spec.dtype}, got "
                        f"{tensor.dtype} — never silently coerce"
                    )
        return outputs

    def _wrap_device_result(self, entry: Any) -> core.Tensor:
        """Wrap one raw DEVICE-kernel result entry into a ``core.Tensor``.

        Passed to ``dispatch_external_kernel`` as ``wrap_device_result``: a
        raw iree runtime ``DeviceArray`` is wrapped in
        ``core.Tensor(IreeDevicePayload(entry, self._core_device))`` — the
        result lives on THIS executable's device (``DeviceArray`` handles
        returned by device kernels are HAL buffers on the run's device);
        any other entry (a duck-typed device payload carrying its own
        ``core.Device``) wraps as ``core.Tensor(entry)`` directly. Never
        materializes a host copy. ``iree.runtime`` is imported lazily
        inside the function body (import acyclicity).
        """
        import iree.runtime as rt

        if isinstance(entry, rt.DeviceArray):
            return core.Tensor(IreeDevicePayload(entry, self._core_device))
        return core.Tensor(entry)


def _validate_input_shape(index: int, declared: tuple[Any, ...], actual: tuple[int, ...]) -> None:
    """Walk a declared input shape against the concrete shape.

    Rank must match exactly (``core.ShapeError``). Per-dim: a static ``int``
    dim must equal the runtime size (``core.ShapeError`` naming input/dim);
    symbolic entries (``Dim``/``DimExpr``) pass through — IREE executes
    runtime-dynamic shapes (validated) — and ``None`` (runtime-dynamic,
    unchecked) is skipped, mirroring the interpreter's unchecked-dim rule.
    """
    if len(actual) != len(declared):
        raise core.ShapeError(
            f"input {index}: rank mismatch — declared rank {len(declared)} "
            f"vs runtime rank {len(actual)}"
        )
    for dim, (want, got) in enumerate(zip(declared, actual)):
        if want is None:
            continue  # runtime-dynamic dim: unchecked by design
        if isinstance(want, int) and want != got:
            raise core.ShapeError(
                f"input {index}: shape mismatch at dim {dim} — expected "
                f"{want}, got runtime size {got}"
            )


class IreeDevicePayload:
    """Public duck-typed device payload for ``core.Tensor`` (the payload protocol).

    Wraps an IREE runtime ``DeviceArray`` — a HAL buffer-view handle that is
    ALREADY resident on the executable's device — plus the ``core.Device`` it
    lives on. Host materialization (``to_host``) is LAZY: the D2H copy happens
    only when the wrapping tensor's ``.numpy()`` is called, so the hot
    ``IreeExecutable.run`` path never touches host memory.

    PUBLIC interop contract (device-resident external kernels): kernel authors
    may isinstance-check ``etl.backends.adapters.iree.IreeDevicePayload`` (the
    ``.data`` of the ``core.Tensor`` operands a device kernel receives) and
    use:

    - ``.device_array`` — the wrapped iree runtime ``DeviceArray`` (the HAL
      buffer handle, already on the executable's device; also gives access to
      the underlying HAL device for running further modules on the SAME
      device);
    - ``.shape`` / ``.dtype`` — metadata (tuple of ints / numpy dtype);
    - ``.to_host()`` — an EXPLICIT materialization to a host numpy array (a
      device kernel may deliberately materialize; the adapter itself NEVER
      forces host copies of device results);
    - ``.device`` — the ``core.Device`` the tensor lives on.

    A device-resident kernel runs on the executable's device and may e.g.
    compile/run additional vmfb modules on the SAME HAL device via
    ``iree.runtime`` and return their ``DeviceArray`` s (raw or wrapped).

    Payload protocol (see ``etl/core/tensor.py``): ``.shape`` (tuple of
    ints), ``.dtype`` (numpy-normalizable — DeviceArray is override-aware,
    e.g. bool results), optional ``.device`` (the core.Device), and
    ``to_host() -> np.ndarray`` tried first by ``Tensor.numpy()`` (with
    ``np.asarray`` as the fallback, via ``__array__``). On ``local-task``
    (llvm-cpu) buffers the to_host mapping is zero-copy for mappable memory;
    on cuda it is a fresh D2H copy per call.
    """

    __slots__ = ("_array", "device")

    def __init__(self, array: Any, device: Device) -> None:
        self._array = array
        self.device = device

    @property
    def device_array(self) -> Any:
        """The wrapped IREE runtime ``DeviceArray`` (the HAL buffer handle).

        Used by ``IreeExecutable.run`` for the same-device pass-through rule:
        an input tensor produced by an iree executable can be handed to the
        next invoke without a host round-trip.
        """
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    def to_host(self) -> np.ndarray:
        """Materialize a host copy of the device buffer (lazy D2H)."""
        return self._array.to_host()

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """``np.asarray`` fallback for host materialization.

        ``core.Tensor`` prefers ``to_host()``; this keeps
        ``np.asarray(payload)`` correct for direct payload users too (a
        fresh host copy per call, like ``to_host``).
        """
        return self._array.to_host().__array__(dtype)


# Backward-compat alias: the payload was private (``_IreeDevicePayload``)
# before the device-resident external-kernel round; old isinstance checks
# keep working.
_IreeDevicePayload = IreeDevicePayload


# ---------------------------------------------------------------------------
# Adapter activation: singleton + register-on-first-use entry point
# ---------------------------------------------------------------------------


iree_backend = IreeBackend()


def register() -> IreeBackend:
    """Probe IREE availability and register the singleton (idempotent).

    Called by ``registry.get("iree")`` on first use (``OPTIONAL_ADAPTERS``
    auto-activation) and when loading persisted artifacts. Raises
    ``core.BackendError`` with the hint ``pip install etl[iree]`` when the
    IREE packages are missing; re-registering the SAME instance is a no-op
    (the registry tolerates idempotent re-registration).
    """
    iree_backend.check_available()
    return _registry_register(iree_backend)
