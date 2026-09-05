"""XLA adapter via PJRT — drives a user-provided PJRT C API plugin.

This module implements the ``"xla"`` optional compiler backend for the
shared pluggable-compiler framework (``etl.backends.compiler``). It
consumes the StableHLO MLIR text produced by the shared
``CompilerBackend.lower`` and compiles it by driving a **user-provided PJRT
C API plugin** (``.so`` exporting ``GetPjRtApi``) through pure-stdlib
``ctypes``. No Python frontend and no native binding package is imported
anywhere in this adapter — the plugin ABI translation lives in
``_pjrt_c_api.py`` (vendored from the canonical OpenXLA header; see its
docstring for provenance) and the driver in ``xla_util.py``. The full flow:

- **Plugin discovery**: (a) ``options["plugin_path"]`` (the backend
  compile-options dict, e.g. ``etl.compile(lowered, backend="xla",
  plugin_path="/path/to/pjrt_c_api_cpu_plugin.so")``), (b) the
  ``ETL_PJRT_PLUGIN`` environment variable, (c) well-known paths
  (``/usr/local/lib``, ``/usr/lib``, ``$HOME/.local/lib``, ``./``).
  Missing plugin -> ``core.BackendError`` with build instructions
  (``bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin`` from OpenXLA) — the
  plugin binary is provided BY THE USER, never pip-installed.
- **Compilation**: ``PJRT_Client_Create`` (empty options) ->
  ``PJRT_Client_Compile`` with ``PJRT_Program{code=mlir_text,
  format="mlir"}`` — the plugin accepts StableHLO MLIR text directly (the
  header documents ``"mlir"`` as "MLIR module bytecode (or string)"); no
  MLIR parsing happens in this process.
- **Clients**: ONE refcounted ``PJRT_Client`` per plugin path
  process-wide (``xla_util.acquire_client``/``release_client``) — compile,
  load, and ``check_available`` all share it; a loaded executable holds one
  reference until ``close()``. Load-bearing for real GPU plugins: they grab
  a large BFC memory pool per client (~0.9 x free VRAM) and ABORT the whole
  process (not a catchable PJRT error) when a new client cannot allocate —
  a fresh client per load exhausted the GPU in multi-executable processes.
- **Buffers**: ``PJRT_Client_BufferFromHostBuffer`` (dense row-major numpy
  arrays, all 14 etl dtypes incl. complex64/128) -> execute ->
  ``PJRT_Buffer_ToHostBuffer`` -> ``core.Tensor`` exactly like the numpy
  interpreter (the CPU path). Device-resident executables (loaded on
  ``Device("cuda", N)``) pass ``XlaDevicePayload`` inputs through with
  zero host staging and return ``XlaDevicePayload`` outputs that stay on
  the device (see the ``XlaDevicePayload`` class + ``upload_tensor``).
- **Persistence**: ``PJRT_Executable_Serialize`` /
  ``PJRT_Executable_DeserializeAndLoad`` (true serialize; no load-time
  recompile).
- **Errors**: ``PJRT_Error*`` is checked on EVERY call (NULL = success);
  failures raise ``core.BackendError`` with the plugin's message text
  (``PJRT_Error_Message``) and the error is destroyed. No silent
  fallbacks.

Capability declaration (see ``xla_util.py`` for the driver contract):

- ``dtypes``: float16/32/64, int8/16/32/64, uint8/16/32/64, bool,
  complex64/128 — ALL etl dtypes map 1:1 to ``PJRT_Buffer_Type`` and are
  staged as dense host buffers (numpy's complex layout matches C).
- ``dynamic_shapes=False``: XLA dynamic shapes are limited; ``compile``
  enforces a static-shape gate naming the offending spec.
- ``collectives=False``: 5/6 etl collectives (all_reduce, all_gather,
  reduce_scatter, all_to_all, collective_permute) compile AND run with a
  single replica, but ``dist.broadcast`` (stablehlo
  ``collective-broadcast``) fails AT RUN TIME on XLA:CPU
  (``UNIMPLEMENTED: HLO opcode collective-broadcast is not supported by
  XLA:CPU ThunkEmitter``). Per the capability contract (any failure ->
  flag off), collectives are conservatively False so the shared ``lower``
  rejects collective graphs explicitly instead of crashing mid-``run``.
- ``runtime_calls=False`` / ``custom_blocks=False`` /
  ``async_collectives=False`` (the shared ``lower`` pre-check rejects
  ``runtime_call`` / ``block_call`` / collective ops explicitly, naming
  the feature).

Import discipline (binding): top-level imports limited to stdlib +
``etl.core`` + sibling framework modules + this package's ``_pjrt_c_api``
and ``xla_util`` (both stdlib-only). numpy is imported inside function
bodies. ``import etl`` / ``import etl.backends`` never import this module
(the registry auto-activates on first ``get("xla")``).
"""

from __future__ import annotations

import base64
import weakref
from typing import Any, ClassVar

from etl import core

from ..backend import Capabilities
from ..compiler import CompilerBackend, CompilerExecutable
from ..program import CompiledArtifact, LoweredProgram
from ..registry import register as _registry_register
from . import _pjrt_c_api as _pjrt
from .xla_util import (
    _StaticShapeError,
    _load_plugin,
    _resolve_static_shape,
    acquire_client,
    release_client,
    release_payload,
    set_opt_level,
)

__all__ = [
    "XlaBackend",
    "XlaExecutable",
    "XlaDevicePayload",
    "xla_backend",
    "register",
    "upload_tensor",
]

#: Payload format tag recorded into the CompiledArtifact payload.
_ARTIFACT_FORMAT = "xla-serialized-executable"


class XlaBackend(CompilerBackend):
    """XLA-via-PJRT backend: StableHLO MLIR -> XLA CPU executable.

    Shares the frontend half of the pipeline with every compiler adapter
    (``CompilerBackend.lower``: verify -> capability pre-check -> portable
    inlining -> StableHLO export -> Signature recording). This class adds
    the compiler-specific half: ``check_available`` (plugin probe),
    ``compile`` (plugin compile -> serialized-executable artifact),
    ``load`` (deserialize -> ``XlaExecutable``). The plugin driver and the
    ABI translation are documented in ``xla_util.py`` / ``_pjrt_c_api.py``
    — keep them in sync with the plugin's PJRT C API version.
    """

    name: ClassVar[str] = "xla"
    #: Options-override contract (see ../options.py): the compile options
    #: ``plugin_path`` (plugin discovery — existing),
    #: ``xla_compile_options`` (a serialized ``xla.CompileOptionsProto`` as
    #: bytes, passed to the plugin via PJRT_Client_Compile_Args —
    #: arbitrary compile-option fields pass through, the plugin validates
    #: them) and ``opt_level`` (the XLA optimization level, "O0".."O3" or
    #: 0..3 — normalized via ``options.normalize_opt_level`` and injected
    #: into the compile options' ``executable_build_options`` wire-format
    #: field 24 by ``xla_util.set_opt_level``; an explicit optimization
    #: level already present in ``xla_compile_options`` wins unchanged —
    #: the conflict rule, never both); ``plugin_path`` is also honored at
    #: load (plugin re-discovery for deserialization). No run options in
    #: v1 (a non-empty run options dict raises BackendError).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = {
        "lower": frozenset({"rng_bit_generator"}),
        "compile": frozenset(
            {"plugin_path", "xla_compile_options", "opt_level"}
        ),
        "load": frozenset({"plugin_path"}),
        "run": frozenset(),
    }
    capabilities: ClassVar[Capabilities] = Capabilities(
        dynamic_shapes=False,  # static-shape gate in compile()
        dtypes=frozenset(
            {
                core.float16,
                core.float32,
                core.float64,
                core.int8,
                core.int16,
                core.int32,
                core.int64,
                core.uint8,
                core.uint16,
                core.uint32,
                core.uint64,
                core.bool_,
                core.complex64,
                core.complex128,
            }
        ),
        collectives=False,  # 5/6 run single-replica; collective-broadcast
        # is UNIMPLEMENTED on XLA:CPU at RUN time -> conservative False
        # (shared lower() rejects collective graphs explicitly).
        runtime_calls=False,
        custom_blocks=False,
        async_collectives=False,
        # frozenset() — empty by design. XLA ships RngBitGenerator with
        # THREE_FRY/PHILOX, but real-plugin validation (jax_cuda12_pjrt
        # 0.10.2's xla_cuda_plugin.so) showed the native path is unusable
        # for etl's algorithms: (1) the exporter's u32-word key-first
        # state layout fails PJRT_Client_Compile on XLA's spec-strict
        # u64-state importer ("Binary op add with different element
        # types: u32[1] and u64[]"), and (2) even spec-compliant, XLA's
        # THREE_FRY is threefry2x64 / PHILOX is 2-word philox4x32-10 —
        # different ciphers than etl's threefry2x32 / 4-word
        # philox4x32_10, so bit-exactness vs the numpy reference is
        # impossible by design. The exporter's bit-exact inline
        # expansions are the default; the per-call `rng_bit_generator`
        # lower option still lets users force the native path.
        rng_bit_generator=frozenset(),
    )

    # ---------------------------------------------------------- availability

    @classmethod
    def check_available(cls) -> None:
        """Probe the PJRT plugin dependency; raise ``core.BackendError`` if absent.

        Checks (1) the vendored ctypes bindings module integrity, (2)
        plugin discovery + ``GetPjRtApi`` + the ABI version gate, and (3)
        a live ``PJRT_Client_Create``/``PJRT_Client_Destroy`` round-trip
        through the process-global shared client (``acquire_client`` /
        ``release_client`` — creates the client when no one else holds it,
        destroys it when the probe is the only user).
        Raises ``core.BackendError`` naming the missing piece and how to
        provide/build a plugin (``ETL_PJRT_PLUGIN`` /
        ``options["plugin_path"]``, ``bazel build
        //xla/pjrt/c:pjrt_c_api_cpu_plugin``) — never a bare ImportError.
        """
        _pjrt.verify_api  # bindings module integrity (import-time layout)
        if _pjrt.sizeof(_pjrt.PJRT_Api) <= 0:
            raise core.BackendError(
                "the vendored PJRT C API bindings are broken: PJRT_Api has "
                "no layout"
            )
        plugin = _load_plugin()  # discovery + GetPjRtApi + version gate
        client = acquire_client(plugin)  # live create/destroy round-trip
        release_client(plugin)

    # ---------------------------------------------------------------- compile

    def compile(
        self, lowered: LoweredProgram, options: dict | None = None
    ) -> CompiledArtifact:
        """Compile a lowered StableHLO program into an XLA CPU executable.

        1. Validate ``lowered.backend == "xla"`` and the shared payload
           format (``{"format": "stablehlo", "format_version": 1,
           "mlir_text": ...}``) — ``core.BackendError`` otherwise.
        2. **Static-shape gate**: every entry of every
           ``signature.input_specs``/``output_specs`` shape must resolve
           to a plain int (ints, known-size ``Dim``, ``DimExpr`` without
           free runtime dims). A ``None`` entry or a free symbolic dim
           raises ``core.BackendError`` naming the spec — XLA dynamic
           shapes are limited; the adapter never silently miscompiles
           them.
        3. Load the PJRT plugin (discovery order in the module docstring;
           ``options["plugin_path"]`` takes precedence) and compile the
           MLIR text directly via ``PJRT_Client_Compile`` (the plugin
           accepts StableHLO MLIR text — no MLIR parsing in-process).
        4. Serialize the loaded executable (``PJRT_Executable_Serialize``)
           into a JSON-safe payload:
           ``{"format": "xla-serialized-executable", "mlir_text": ...,
           "executable_base64": ..., "entry_functions": ...,
           "static_input_shapes": ..., "static_output_shapes": ...}``.

        Plugin/compile errors are re-raised as ``core.BackendError``
        carrying the plugin's message — never silent. ``compile`` NEVER
        loads (no executable is retained beyond the serialized bytes).
        """
        if lowered.backend != self.name:
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the xla backend — never "
                "silently re-lower"
            )
        payload = lowered.payload
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "stablehlo"
            or payload.get("format_version") != 1
            or not isinstance(payload.get("mlir_text"), str)
        ):
            raise core.BackendError(
                "the xla backend expects a LoweredProgram with the shared "
                "stablehlo payload (format='stablehlo', format_version=1, "
                f"mlir_text: str), got {payload!r}"
            )
        signature = lowered.signature
        if signature is None:
            raise core.BackendError(
                "the xla backend cannot compile a LoweredProgram without "
                "a recorded signature"
            )

        # Static-shape gate (step 2).
        static_input_shapes = self._gate_static_shapes(
            signature.input_specs, "input"
        )
        static_output_shapes = self._gate_static_shapes(
            signature.output_specs, "output"
        )

        # Options contract (step 3): unknown keys raise BackendError;
        # ``xla_compile_options`` = a serialized ``xla.CompileOptionsProto``
        # (bytes) passed to the plugin — arbitrary compile-option fields pass
        # through, the plugin validates the payload; ``opt_level``
        # ("O0".."O3"/0..3) is normalized and injected into the compile
        # options' ``executable_build_options`` (wire-format field 24) via
        # ``xla_util.set_opt_level`` — an optimization level already present
        # in ``xla_compile_options`` wins unchanged (the conflict rule,
        # never both).
        from ..options import normalize_opt_level, validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "compile")
        compile_options = (options or {}).get("xla_compile_options")
        if compile_options is None:
            # Real PJRT plugins (jax_cuda12_pjrt 0.4.38's xla_cuda_plugin.so)
            # CHECK-crash on NULL compile_options — default to the
            # probe-validated jax-default single-replica proto
            # {executable_build_options { num_replicas: 1, num_partitions: 1 }}.
            compile_options = bytes.fromhex("1a0420012801")
        if not isinstance(compile_options, bytes):
            raise core.BackendError(
                f"the {self.name} 'xla_compile_options' compile option must "
                f"be bytes (a serialized xla.CompileOptionsProto), got "
                f"{type(compile_options).__name__}"
            )

        # ``opt_level`` (optional): normalized 0..3 and injected into the
        # compile options' executable_build_options; an explicit level
        # already present in ``xla_compile_options`` wins (set_opt_level
        # returns the input unchanged — the conflict rule). Unset -> bytes
        # untouched, the default path is byte-identical. A malformed value
        # raises core.BackendError naming the option (never silent).
        opt_level = (options or {}).get("opt_level")
        if opt_level is not None:
            compile_options = set_opt_level(
                compile_options, normalize_opt_level(opt_level)
            )

        # Compile through the plugin (step 4) — errors raise BackendError.
        # The client comes from the process-global shared cache (one live
        # client per plugin path; see xla_util.acquire_client) and is
        # released (destroyed when this was the only user) in the finally.
        plugin = _load_plugin(options)
        client = acquire_client(plugin)
        try:
            loaded = client.compile(payload["mlir_text"], compile_options)
            try:
                serialized = loaded.serialize()
            finally:
                loaded.close()
            platform_name, platform_version = client.platform_info()
        finally:
            release_client(plugin)

        import numpy as np

        artifact_payload = {
            "format": _ARTIFACT_FORMAT,
            "mlir_text": payload["mlir_text"],
            "executable_base64": base64.b64encode(serialized).decode("ascii"),
            "entry_functions": tuple(payload.get("entry_functions", ())),
            # Static gate results recorded for cheap exact validation at
            # run time (no re-resolution of symbolic entries).
            "static_input_shapes": list(static_input_shapes),
            "static_output_shapes": list(static_output_shapes),
        }
        return CompiledArtifact(
            backend=self.name,
            signature=signature,
            target="cpu",
            payload=artifact_payload,
            required_custom_ops=(),
            runtime_dependencies={
                "numpy": np.__version__,
                # The PJRT C API header revision the bindings were
                # translated from (see _pjrt_c_api.py), plus the plugin's
                # self-reported platform identity.
                "pjrt_c_api": _pjrt.HEADER_COMMIT,
                "plugin": f"{platform_name} {platform_version}",
            },
        )

    @staticmethod
    def _gate_static_shapes(specs: Any, kind: str) -> list[tuple[int, ...]]:
        """Apply the static-shape gate to a spec tuple; return resolved shapes.

        ``kind`` is "input" or "output" (message wording only). Raises
        ``core.BackendError`` naming the offending spec.
        """
        shapes = []
        for i, spec in enumerate(specs or ()):
            try:
                shapes.append(
                    _resolve_static_shape(spec.shape, f"{kind} spec {i}")
                )
            except _StaticShapeError as exc:
                raise core.BackendError(
                    "the xla adapter requires fully static shapes; got "
                    f"{kind} spec {i}: {spec} — {exc}"
                ) from exc
        return shapes

    # ------------------------------------------------------------------- load

    def load(
        self,
        artifact: CompiledArtifact,
        device: core.Device | None = None,
        options: dict | None = None,
    ) -> "XlaExecutable":
        """Reconstruct an ``XlaExecutable`` from a serialized artifact.

        Validates the recorded backend (``core.PersistenceError`` naming
        both on mismatch), the plugin (``check_available``), the device
        (None or a ``core.Device``; non-``Device`` ->
        ``core.DeviceError``), and the payload format. Device model
        (explicit placement — mirroring the iree adapter):

        - ``None`` or ``Device("cpu", 0)``: a CPU executable that stages
          host inputs via ``PJRT_Client_BufferFromHostBuffer`` and copies
          outputs back via ``PJRT_Buffer_ToHostBuffer`` (the historical
          behavior, unchanged). A non-zero cpu index is a
          ``core.DeviceError`` (only ``Device('cpu', 0)`` exists).
        - ``Device("cuda", N)``: a device-resident executable on the
          client's addressable device N (``core.BackendError`` naming the
          index and the device count when out of range; other kinds are a
          ``core.BackendError`` naming the supported set). Its inputs must
          be device-resident ``XlaDevicePayload`` tensors on the same
          device (host inputs raise ``core.DeviceError`` — never staged),
          and its outputs stay on the device as ``XlaDevicePayload``
          tensors.

        The base64 executable is deserialized with
        ``PJRT_Executable_DeserializeAndLoad`` on the process-global shared
        client for the (re-discovered) plugin (one refcounted client per
        plugin path — ``acquire_client``/``release_client``; the executable
        releases its reference in ``close()``) — the ``plugin_path`` load
        option is honored for discovery (falls back to ``ETL_PJRT_PLUGIN`` /
        well-known paths). Options are validated against ``KNOWN_OPTIONS``
        (unknown keys => ``core.BackendError``). NEVER re-traces, re-lowers,
        or re-compiles; a deserialization failure (environment/ABI
        mismatch) raises ``core.PersistenceError`` — no silent
        recompilation.
        """
        if artifact.backend != self.name:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                "the xla backend cannot load it"
            )
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "load")
        self.check_available()
        if device is not None:
            if not isinstance(device, core.Device):
                raise core.DeviceError(
                    "device must be None or a core.Device, got "
                    f"{type(device).__name__}"
                )
            if device.kind == "cpu" and device.index != 0:
                raise core.DeviceError(
                    "only Device('cpu', 0) exists for kind 'cpu' — a CPU "
                    f"executable always runs on the host, got {device!r}"
                )
            if device.kind not in ("cpu", "cuda"):
                raise core.BackendError(
                    "the xla adapter supports CPU ('cpu') and GPU ('cuda') "
                    f"devices, got {device!r}"
                )
        effective_device = (
            device if device is not None else core.Device("cpu", 0)
        )
        device_index = (
            effective_device.index if effective_device.kind == "cuda" else None
        )

        payload = artifact.payload
        if (
            not isinstance(payload, dict)
            or payload.get("format") != _ARTIFACT_FORMAT
            or not isinstance(payload.get("executable_base64"), str)
        ):
            raise core.PersistenceError(
                "corrupt: the xla artifact payload must carry "
                f"format={_ARTIFACT_FORMAT!r} and a base64 "
                "executable_base64 field"
            )

        plugin = _load_plugin(options)  # plugin_path load option honored
        # One process-global client per plugin path (refcounted): the
        # executable takes a reference and releases it in close() — the
        # client is destroyed only when the LAST reference is released.
        client = acquire_client(plugin)
        try:
            # Resolve the raw PJRT_Device* for device-resident runs
            # (None for CPU — execute stays on the compile-time device).
            execute_device: int | None = None
            if device_index is not None:
                devices = client.addressable_devices()
                if device_index >= len(devices):
                    raise core.BackendError(
                        f"the xla plugin client reports {len(devices)} "
                        "addressable device(s); cannot run on "
                        f"{effective_device!r}"
                    )
                execute_device = devices[device_index]
            serialized = base64.b64decode(payload["executable_base64"])
            try:
                loaded = client.deserialize(serialized)
            except core.BackendError as exc:
                raise core.PersistenceError(
                    "failed to deserialize the XLA executable — the artifact "
                    "is incompatible with this environment/ABI/plugin: "
                    f"{exc} — never silently recompiling"
                ) from exc
        except Exception:
            release_client(plugin)  # only on failure — success hands the
            raise                     # reference to the executable
        return XlaExecutable(
            artifact=artifact,
            device=effective_device,
            native_module=loaded,
            entry_functions=tuple(payload.get("entry_functions", ())),
            client=client,
            plugin=plugin,
            execute_device=execute_device,
        )


class XlaExecutable(CompilerExecutable):
    """Run-time object for the xla backend (satisfies ``Executable``).

    ``native_module`` is the driver's ``_LoadedExecutable`` (a live
    ``PJRT_LoadedExecutable``). ``run(flat_input_tensors)`` semantics depend
    on the load device (explicit placement — mirroring the iree adapter):

    - **CPU executable** (``Device("cpu", 0)``): inputs are staged from
      numpy host buffers through ``PJRT_Client_BufferFromHostBuffer``,
      executed, and the output buffers are copied back via
      ``PJRT_Buffer_ToHostBuffer`` and wrapped as ``core.Tensor`` exactly
      like the numpy interpreter (``core.Tensor(np.asarray(...))``).
    - **Device-resident executable** (``Device("cuda", N)``): every input
      must already be a device-resident ``core.Tensor`` carrying an
      ``XlaDevicePayload`` on this executable's device (and from the same
      client) — its PJRT buffer is handed to
      ``PJRT_LoadedExecutable_Execute`` DIRECTLY with zero host staging; a
      host input raises ``core.DeviceError`` (never staged — there is no
      implicit host->device transfer; place it explicitly via
      ``t.to(Device('cuda', N))``). Outputs are wrapped as
      ``XlaDevicePayload`` tensors that STAY on the device (their
      ``.numpy()`` raises the standard ``core.DeviceError``; read them back
      via the explicit ``t.to(cpu).numpy()``).

    ``save``/``load`` are the SHARED ``CompilerExecutable``
    implementations (artifact round-trip; the executable is reconstructed
    explicitly at ``load`` — device handles are never serialized).
    """

    backend_name: ClassVar[str] = "xla"

    # Run-stage option validation table — mirrors ``XlaBackend.KNOWN_OPTIONS``
    # (single source of truth; the class attribute is defined here so the
    # executable validates independently of the backend instance).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = XlaBackend.KNOWN_OPTIONS

    def __init__(
        self,
        artifact: CompiledArtifact | None = None,
        signature: Any = None,
        device: core.Device | None = None,
        native_module: Any = None,
        entry_functions: tuple[str, ...] = (),
        client: Any = None,
        plugin: Any = None,
        execute_device: Any = None,
    ) -> None:
        super().__init__(
            artifact=artifact,
            signature=signature,
            device=device,
            native_module=native_module,
            entry_functions=entry_functions,
        )
        self._client = client
        self._plugin = plugin  # keeps the loaded plugin library alive
        # The raw PJRT_Device* this executable executes on (None = the
        # compile-time device — the CPU path, unchanged).
        self._execute_device = execute_device

    def close(self) -> None:
        """Release the plugin handles (executable first, then the client ref).

        The client is a reference on the process-global shared client for
        the plugin path: closing the executable destroys the ``PJRT_Client``
        only when NO other executable (or a live ``XlaDevicePayload``
        output) still holds a reference (``release_client``). Not part of
        the ``Executable`` protocol; call it when done with the executable
        (the process also reclaims everything at exit).
        """
        if self.native_module is not None:
            self.native_module.close()
            self.native_module = None
        if self._client is not None:
            release_client(self._plugin)
            self._client = None
            self._plugin = None

    # -------------------------------------------------------------------- run

    def run(
        self,
        flat_input_tensors: list[core.Tensor],
        options: dict | None = None,
    ) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        Validates inputs EXACTLY against ``signature.input_specs``:
        count (``BackendError``), type, dtype (``DTypeError``) and the
        static shape recorded at compile time (``ShapeError`` — the xla
        adapter's shapes are static). A CPU executable stages inputs via
        ``PJRT_Client_BufferFromHostBuffer``, executes, and copies the
        output buffers back as ``core.Tensor`` (the historical behavior).
        A device-resident executable passes ``XlaDevicePayload`` inputs
        through with ZERO host staging (any host input raises
        ``core.DeviceError`` — no implicit host->device transfer) and
        returns ``XlaDevicePayload`` outputs that stay on the device.
        Output count/dtype/shape are validated against
        ``signature.output_specs``. A runtime failure raises
        ``core.BackendError`` naming the cause — never a silent fallback.

        ``options``: per-run options, validated against ``KNOWN_OPTIONS`` —
        the xla run stage has NO known options in v1, so any non-empty
        options dict raises ``core.BackendError`` (loud; never silently
        swallowed).
        """
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.backend_name, "run")
        if self.native_module is None or self._client is None:
            raise core.BackendError(
                "this XlaExecutable has no live PJRT executable/client — "
                "construct it via the xla backend's load(artifact)"
            )
        if self.signature is None:
            raise core.BackendError(
                "this XlaExecutable has no recorded signature to validate "
                "inputs against"
            )
        input_specs = tuple(self.signature.input_specs)
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"program expects {len(input_specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        recorded_input_shapes = self._recorded_shapes(
            "static_input_shapes", len(input_specs)
        )
        expected_shapes = [
            tuple(recorded_input_shapes[i]) for i in range(len(input_specs))
        ]
        device_resident = self.device is not None and self.device.kind != "cpu"
        arrays: list[Any] = []
        for i, (tensor, spec, expected) in enumerate(
            zip(flat_input_tensors, input_specs, expected_shapes)
        ):
            if not isinstance(tensor, core.Tensor):
                raise core.BackendError(
                    f"input {i} must be a core.Tensor, got "
                    f"{type(tensor).__name__}"
                )
            if tensor.dtype != spec.dtype:
                raise core.DTypeError(
                    f"input {i}: expected dtype {spec.dtype}, got "
                    f"{tensor.dtype}"
                )
            if tuple(tensor.shape) != expected:
                raise core.ShapeError(
                    f"input {i}: expected static shape {expected}, got "
                    f"{tuple(tensor.shape)} — the xla adapter requires "
                    "exact static shapes"
                )
            if device_resident:
                self._check_device_resident_input(i, tensor)
                arrays.append(tensor.data.buffer)  # pass the PJRT buffer
            else:
                arrays.append(tensor.numpy())

        if device_resident:
            buffers = arrays  # _Buffer handles owned by their payloads
        else:
            buffers = [self._client.buffer_from_host(array) for array in arrays]
        output_buffers: list[Any] = []
        try:
            output_buffers = self.native_module.execute(
                buffers, execute_device=self._execute_device
            )
            if device_resident:
                # Outputs STAY on the device — wrap them as payloads; the
                # payloads own the buffers (never closed here).
                tensors = [
                    core.Tensor(
                        XlaDevicePayload(
                            self._plugin, self._client, buffer, self.device
                        )
                    )
                    for buffer in output_buffers
                ]
            else:
                tensors = [
                    core.Tensor(buffer.to_host()) for buffer in output_buffers
                ]
        finally:
            if not device_resident:
                # CPU path: staged inputs and copied outputs are owned here.
                for buffer in buffers:
                    buffer.close()
                for buffer in output_buffers:
                    buffer.close()

        output_specs = tuple(self.signature.output_specs)
        if len(tensors) != len(output_specs):
            raise core.BackendError(
                f"program produced {len(tensors)} output tensor(s), "
                f"expected {len(output_specs)}"
            )
        recorded_output_shapes = self._recorded_shapes(
            "static_output_shapes", len(output_specs)
        )
        for i, (tensor, spec) in enumerate(zip(tensors, output_specs)):
            if tensor.dtype != spec.dtype:
                raise core.BackendError(
                    f"output {i}: expected dtype {spec.dtype}, got "
                    f"{tensor.dtype}"
                )
            expected = tuple(recorded_output_shapes[i])
            if tuple(tensor.shape) != expected:
                raise core.BackendError(
                    f"output {i}: expected static shape {expected}, got "
                    f"{tuple(tensor.shape)}"
                )
        return tensors

    def _check_device_resident_input(self, i: int, tensor: core.Tensor) -> None:
        """Validate input ``i`` of a device-resident run (explicit placement).

        The input must be an ``XlaDevicePayload`` tensor on this
        executable's device, created by the same shared client — its PJRT
        buffer is executed directly (zero host staging). Host tensors and
        foreign payloads raise ``core.DeviceError`` with the same message
        family as the iree adapter: there is no implicit host->device
        transfer at the run boundary (the pipeline rejects host inputs
        first for ``etl.run``; this is the defensive check for direct
        backend-level ``exe.run([...])`` calls).
        """
        payload = tensor.data
        if not isinstance(payload, XlaDevicePayload):
            raise core.DeviceError(
                f"input {i} is not a device-resident tensor on this "
                f"executable's device ({self.device!r}): the xla adapter "
                "never stages host inputs for a device-resident run — "
                "there is no implicit host-to-device transfer at the run "
                "boundary. Place inputs on the device explicitly first "
                f"via t.to({self.device!r}) (the xla backend's placement "
                "provider), or load the executable on Device('cpu', 0) "
                f"for host staging. Got a tensor carrying "
                f"{type(payload).__name__}."
            )
        if payload.device != self.device:
            raise core.DeviceError(
                f"input {i} is on {payload.device!r}, but the executable "
                f"runs on {self.device!r}: no implicit device-to-device or "
                "host-to-device transfer happens at the run boundary — "
                f"move the tensor to the executable's device explicitly "
                f"via t.to({self.device!r})."
            )
        if payload.client is not self._client:
            raise core.DeviceError(
                f"input {i} carries an xla device payload owned by a "
                "different PJRT client/plugin than this executable's — a "
                "buffer from another client cannot be executed (never "
                "silently mixed)."
            )

    def _recorded_shapes(self, field: str, count: int) -> list[list[int]]:
        """Read the static-shape gate results recorded in the artifact payload.

        Falls back to re-resolving from the signature when the payload
        predates the recording (defensive; the gate guarantees static).
        """
        payload = self.artifact.payload if self.artifact is not None else None
        if isinstance(payload, dict) and field in payload:
            recorded = payload[field]
            if isinstance(recorded, list) and len(recorded) == count:
                return recorded
        if field == "static_input_shapes":
            specs = tuple(self.signature.input_specs)
        else:
            specs = tuple(self.signature.output_specs)
        shapes = []
        for i, spec in enumerate(specs):
            try:
                shapes.append(
                    list(_resolve_static_shape(spec.shape, f"spec {i}"))
                )
            except _StaticShapeError as exc:
                raise core.BackendError(
                    "this artifact predates the static-shape recording "
                    f"and spec {i} is not statically resolvable: {exc}"
                ) from exc
        return shapes


class XlaDevicePayload:
    """PUBLIC device-resident payload wrapping a live PJRT buffer.

    The ``.data`` of device-resident ``core.Tensor`` outputs of
    device-resident ``XlaExecutable`` runs and of the explicit placement
    path ``t.to(Device('cuda', N))`` (``upload_tensor``). The wrapped
    ``PJRT_Buffer`` stays alive for the payload's lifetime; the payload
    holds one reference on the process-global shared client
    (``acquire_client``) and releases it — destroying the buffer BEFORE
    the client, guarded against stale-cache ordering — when garbage
    collected. It exposes the ``core.Tensor`` device-payload protocol:

    - ``.buffer``: the driver ``_Buffer`` (the live ``PJRT_Buffer``) —
      the unit handed to ``PJRT_LoadedExecutable_Execute`` for
      same-device pass-through (zero host round-trips).
    - ``.shape`` / ``.dtype``: metadata queries (never a host copy).
    - ``.device``: the ``core.Device`` the buffer physically lives on.
    - ``.to_host()``: the EXPLICIT device-to-host copy
      (``PJRT_Buffer_ToHostBuffer``) — invoked by ``t.to(cpu)`` /
      ``.numpy()`` on a cpu-kind payload; a non-cpu payload's
      ``.numpy()`` raises the standard ``core.DeviceError`` (no implicit
      transfer).
    - ``.client``: the owning shared client (identity-checked by
      device-resident runs).

    Users may isinstance-check this public class exactly like
    ``IreeDevicePayload``. Buffers are never aliased by ``core.Tensor``
    wrappers; each run produces fresh payloads.
    """

    __slots__ = ("_buffer", "_device", "_client", "_finalizer", "__weakref__")

    def __init__(self, plugin: Any, client: Any, buffer: Any, device: core.Device) -> None:
        self._buffer = buffer
        self._device = device
        self._client = client
        # Own one shared-client reference for the buffer's lifetime: PJRT
        # buffers must outlive... in fact be destroyed BEFORE the client
        # that owns them — the guarded release_payload enforces that order.
        acquire_client(plugin)
        self._finalizer = weakref.finalize(
            self, release_payload, plugin, client, buffer
        )

    @property
    def buffer(self) -> Any:
        """The live driver ``_Buffer`` (the wrapped ``PJRT_Buffer``)."""
        return self._buffer

    @property
    def shape(self) -> tuple[int, ...]:
        """The buffer's concrete shape (metadata query — no host copy)."""
        return self._buffer.shape

    @property
    def dtype(self) -> Any:
        """The buffer's numpy dtype (metadata query — no host copy)."""
        return self._buffer.dtype

    @property
    def device(self) -> core.Device:
        """The ``core.Device`` this buffer physically lives on."""
        return self._device

    @property
    def client(self) -> Any:
        """The shared PJRT client that owns this buffer (identity check)."""
        return self._client

    def to_host(self) -> Any:
        """The EXPLICIT device-to-host copy (``PJRT_Buffer_ToHostBuffer``).

        A fresh numpy array per call — never cached (device memory may be
        updated in place by later executes).
        """
        return self._buffer.to_host()


#: The module-level singleton (registered on first use by the registry).
xla_backend = XlaBackend()


def upload_tensor(tensor: core.Tensor, device: core.Device) -> core.Tensor:
    """The core device-transfer provider for kind ``"cuda"`` (explicit placement).

    Registered by ``register()`` — overwriting whatever provider currently
    serves kind ``"cuda"`` (idempotent, last-wins — the same pattern as the
    iree adapter: in a mixed process, whichever adapter activated LAST owns
    "cuda" placement; the other backend rejects foreign payloads with a
    clear ``core.DeviceError``). The contract (``core.Tensor.to``):

    - the source must be a HOST ndarray-backed tensor on ``Device("cpu",
      0)`` — anything else raises ``core.DeviceError`` suggesting the
      explicit two-hop ``t.to(cpu).to(target)`` (no cross-device copies in
      v1);
    - the target must be a ``Device("cuda", N)`` with N below the plugin
      client's addressable-device count (``core.BackendError`` naming the
      count otherwise) — the plugin is discovered via ``ETL_PJRT_PLUGIN`` /
      well-known paths (``Tensor.to`` passes no options);
    - the upload is ONE-SHOT: ``PJRT_Client_BufferFromHostBuffer`` on the
      shared client (fresh buffer, no cache), returning a fresh
      device-resident ``core.Tensor`` wrapping an ``XlaDevicePayload``.

    This is the bootstrap for device-resident loops: place the seed state
    once with ``state.to(Device('cuda', N))``, then feed each run's
    device-resident outputs back as the next inputs — zero host
    round-trips after the first placement.
    """
    if not isinstance(tensor, core.Tensor):
        raise core.DeviceError(
            "the xla placement provider expects a core.Tensor, got "
            f"{type(tensor).__name__}"
        )
    if device.kind != "cuda":
        # Defensive: the provider is registered for kind "cuda" only.
        raise core.DeviceError(
            f"the xla placement provider places data on 'cuda' devices "
            f"only, got {device!r}"
        )
    if tensor.device != core.Device("cpu", 0):
        raise core.DeviceError(
            f"cannot place a {tensor.device!r} tensor on {device!r}: the "
            "xla placement path stages host memory only (v1 has no "
            "cross-device copies). Transfer in two explicit hops instead: "
            "t.to(core.Device('cpu', 0)) first, then .to(target)."
        )
    plugin = _load_plugin()
    client = acquire_client(plugin)
    try:
        devices = client.addressable_devices()
        if not 0 <= device.index < len(devices):
            raise core.BackendError(
                f"the xla plugin client reports {len(devices)} addressable "
                f"device(s); cannot place data on {device!r}"
            )
        buffer = client.buffer_from_host(tensor.numpy(), device_index=device.index)
    except Exception:
        release_client(plugin)
        raise
    payload = XlaDevicePayload(plugin, client, buffer, device)
    return core.Tensor(payload)


def register() -> None:
    """Probe the plugin and register the backend (idempotent).

    Called by ``etl.backends.registry.get("xla")`` on first use (and by
    persisted-artifact loads). Raises ``core.BackendError`` with an
    actionable message (plugin discovery order + how to build a plugin
    from OpenXLA) when no usable PJRT plugin is available — there is no
    pip-installable dependency; the user provides the plugin binary via
    ``options["plugin_path"]`` or the ``ETL_PJRT_PLUGIN`` environment
    variable. Does nothing observable when already registered.

    On activation it ALSO overwrites the core device-transfer provider for
    kind ``"cuda"`` with this adapter's ``upload_tensor`` (idempotent,
    last-wins — mirroring the iree adapter): ``t.to(Device('cuda', N))``
    then stages host data onto the PJRT plugin's device N through the
    shared client, returning an ``XlaDevicePayload`` tensor — the explicit
    bootstrap for device-resident xla loops.
    """
    XlaBackend.check_available()
    _registry_register(xla_backend)
    core.register_device_transfer_provider("cuda", upload_tensor)
