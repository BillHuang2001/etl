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
- **Buffers**: ``PJRT_Client_BufferFromHostBuffer`` (dense row-major numpy
  arrays, all 14 etl dtypes incl. complex64/128) -> execute ->
  ``PJRT_Buffer_ToHostBuffer`` -> ``core.Tensor`` exactly like the numpy
  interpreter.
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
)

__all__ = ["XlaBackend", "XlaExecutable", "xla_backend", "register"]

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
    #: ``plugin_path`` (plugin discovery — existing) and
    #: ``xla_compile_options`` (a serialized ``xla.CompileOptionsProto`` as
    #: bytes, passed to the plugin via PJRT_Client_Compile_Args —
    #: arbitrary compile-option fields pass through, the plugin validates
    #: them); ``plugin_path`` is also honored at load (plugin re-discovery
    #: for deserialization). No run options in v1 (a non-empty run options
    #: dict raises BackendError).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = {
        "lower": frozenset({"rng_bit_generator"}),
        "compile": frozenset({"plugin_path", "xla_compile_options"}),
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
        # {"threefry2x32", "philox4x32_10"} by design: XLA ships
        # RngBitGenerator with THREE_FRY/PHILOX (the exporter's native
        # rng_bit_generator path with the verified [key0,key1,ctr...]
        # state layout). The xla tests stay gate-skipped without a
        # user-provided PJRT plugin — bit-exactness of BOTH algorithms
        # must be re-validated against the numpy reference with a real
        # plugin (see adapters/CONTEXT.md).
        rng_bit_generator=frozenset({"threefry2x32", "philox4x32_10"}),
    )

    # ---------------------------------------------------------- availability

    @classmethod
    def check_available(cls) -> None:
        """Probe the PJRT plugin dependency; raise ``core.BackendError`` if absent.

        Checks (1) the vendored ctypes bindings module integrity, (2)
        plugin discovery + ``GetPjRtApi`` + the ABI version gate, and (3)
        a live ``PJRT_Client_Create``/``PJRT_Client_Destroy`` round-trip.
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
        client = plugin.create_client()  # live create/destroy round-trip
        client.close()

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
        # through, the plugin validates the payload.
        from ..options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "compile")
        compile_options = (options or {}).get("xla_compile_options")
        if compile_options is not None and not isinstance(compile_options, bytes):
            raise core.BackendError(
                f"the {self.name} 'xla_compile_options' compile option must "
                f"be bytes (a serialized xla.CompileOptionsProto), got "
                f"{type(compile_options).__name__}"
            )

        # Compile through the plugin (step 4) — errors raise BackendError.
        plugin = _load_plugin(options)
        client = plugin.create_client()
        try:
            loaded = client.compile(payload["mlir_text"], compile_options)
            try:
                serialized = loaded.serialize()
            finally:
                loaded.close()
            platform_name, platform_version = client.platform_info()
        finally:
            client.close()

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
        (None or a CPU ``core.Device``; non-``Device`` ->
        ``core.DeviceError``; non-cpu kind -> ``core.BackendError``), and
        the payload format. The base64 executable is deserialized with
        ``PJRT_Executable_DeserializeAndLoad`` on a fresh client from the
        (re-discovered) plugin — the ``plugin_path`` load option is honored
        for discovery (falls back to ``ETL_PJRT_PLUGIN`` / well-known
        paths). Options are validated against ``KNOWN_OPTIONS`` (unknown
        keys => ``core.BackendError``). NEVER re-traces, re-lowers, or
        re-compiles; a deserialization failure (environment/ABI mismatch)
        raises ``core.PersistenceError`` — no silent recompilation.
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
            if device.kind != "cpu":
                raise core.BackendError(
                    f"the xla adapter supports only CPU devices, got "
                    f"{device!r}"
                )
        effective_device = (
            device if device is not None else core.Device("cpu", 0)
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
        client = plugin.create_client()
        try:
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
            client.close()  # only on failure — success hands the client over
            raise
        return XlaExecutable(
            artifact=artifact,
            device=effective_device,
            native_module=loaded,
            entry_functions=tuple(payload.get("entry_functions", ())),
            client=client,
            plugin=plugin,
        )


class XlaExecutable(CompilerExecutable):
    """Run-time object for the xla backend (satisfies ``Executable``).

    ``native_module`` is the driver's ``_LoadedExecutable`` (a live
    ``PJRT_LoadedExecutable``); ``run(flat_input_tensors)`` stages numpy
    host buffers through ``PJRT_Client_BufferFromHostBuffer``, executes,
    copies the output buffers back via ``PJRT_Buffer_ToHostBuffer``, and
    wraps the results as ``core.Tensor`` exactly like the numpy interpreter
    (``core.Tensor(np.asarray(...))``). ``save``/``load`` are the SHARED
    ``CompilerExecutable`` implementations (artifact round-trip; the
    executable is reconstructed explicitly at ``load`` — device handles
    are never serialized).
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

    def close(self) -> None:
        """Release the plugin handles (executable first, then client).

        Not part of the ``Executable`` protocol; call it when done with the
        executable (the process also reclaims everything at exit).
        """
        if self.native_module is not None:
            self.native_module.close()
            self.native_module = None
        if self._client is not None:
            self._client.close()
            self._client = None

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
        adapter's shapes are static). Inputs are staged via
        ``PJRT_Client_BufferFromHostBuffer``, executed, and the output
        buffers are copied back and wrapped as ``core.Tensor`` exactly like
        the numpy interpreter. Output count/dtype/shape are validated
        against ``signature.output_specs``. A runtime failure raises
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
        arrays = []
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
            arrays.append(tensor.numpy())

        buffers = [self._client.buffer_from_host(array) for array in arrays]
        output_buffers: list[Any] = []
        try:
            output_buffers = self.native_module.execute(buffers)
            tensors = [core.Tensor(buffer.to_host()) for buffer in output_buffers]
        finally:
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


#: The module-level singleton (registered on first use by the registry).
xla_backend = XlaBackend()


def register() -> None:
    """Probe the plugin and register the backend (idempotent).

    Called by ``etl.backends.registry.get("xla")`` on first use (and by
    persisted-artifact loads). Raises ``core.BackendError`` with an
    actionable message (plugin discovery order + how to build a plugin
    from OpenXLA) when no usable PJRT plugin is available — there is no
    pip-installable dependency; the user provides the plugin binary via
    ``options["plugin_path"]`` or the ``ETL_PJRT_PLUGIN`` environment
    variable. Does nothing observable when already registered.
    """
    XlaBackend.check_available()
    _registry_register(xla_backend)
