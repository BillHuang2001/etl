"""XLA adapter via PJRT — direct jaxlib/xla_client, NOT the jax frontend.

This module implements the ``"xla"`` optional compiler backend for the
shared pluggable-compiler framework (``etl.backends.compiler``). It
consumes the StableHLO MLIR text produced by the shared
``CompilerBackend.lower`` and compiles it with the PJRT CPU client that
ships EMBEDDED in jaxlib. The jax frontend (tracing, ``jax.jit``,
``jax.core``, ``jax._src.*``) is never imported — everything goes through
the ``jaxlib.xla_client`` / ``jaxlib.mlir`` / ``jaxlib._jax`` native
bindings. The full acquisition path (plugin discovery, dialect loading,
compile entry point, buffer creation, serialization APIs — each verified
against jax 0.10.2 / jaxlib 0.10.2, CPU, numpy 2.4.6) is documented in
``xla_util.py``; a summary:

- **PJRT client**: NO standalone ``pjrt_c_api_cpu_plugin.so`` exists in
  jaxlib 0.10.2 — the CPU client is EMBEDDED in ``_xla.so`` and acquired
  via ``xc.make_cpu_client()``.
- **Compilation**: parse the MLIR text with jaxlib's MLIR bindings and
  call ``client.compile_and_load(module, executable_devices=...,
  compile_options=...)`` (the same entry point jax's own compiler uses;
  no bytecode/``mlir_module_to_xla_computation`` conversion needed).
- **Buffers**: ``client.buffer_from_pyval`` DOES NOT EXIST in 0.10.2 —
  inputs stage via ``xc.batched_device_put(aval, sharding, [arr],
  [dev], True, enable_x64=True)`` with a duck-typed aval and a patched
  single-device sharding. ``enable_x64=True`` is mandatory (the default
  x64-off state silently truncates float64/int64 to 32-bit).
- **Serialization**: ``exe.serialize() -> bytes`` and
  ``client.deserialize_executable(bytes, device_list)`` both exist and
  round-trip correctly.

Capability declaration (validated end-to-end, single CPU replica):

- ``dtypes``: float16/32/64, int8/16/32/64, uint8/16/32/64, bool,
  complex64/128 — ALL etl dtypes validated through the full
  parse->compile->execute path (elementwise add; dot additionally for
  float16/32/64/int32/int64; bool as input AND output; complex64
  multiply).
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

Import discipline (binding): NO heavy imports at module top level —
stdlib + ``etl.core`` + sibling framework modules only. numpy, jaxlib and
the MLIR bindings are imported INSIDE function bodies (``xla_util.py``
holds the jaxlib plumbing under the same rule). ``import etl`` /
``import etl.backends`` never import this module or jaxlib (the registry
auto-activates on first ``get("xla")``).
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

from etl import core

from ..backend import Capabilities
from ..compiler import CompilerBackend, CompilerExecutable
from ..program import CompiledArtifact, LoweredProgram
from ..registry import register as _registry_register
from .xla_util import (
    _StaticShapeError,
    _acquire_cpu_client,
    _import_xla_runtime,
    _make_buffer_putter,
    _parse_stablehlo_module,
    _resolve_static_shape,
    _verify_xla_api_surface,
)

__all__ = ["XlaBackend", "XlaExecutable", "xla_backend", "register"]

#: Payload format tag recorded into the CompiledArtifact payload.
_ARTIFACT_FORMAT = "xla-serialized-executable"


class XlaBackend(CompilerBackend):
    """XLA-via-PJRT backend: StableHLO MLIR -> XLA CPU executable.

    Shares the frontend half of the pipeline with every compiler adapter
    (``CompilerBackend.lower``: verify -> capability pre-check -> portable
    inlining -> StableHLO export -> Signature recording). This class adds
    the compiler-specific half: ``check_available`` (jaxlib probe),
    ``compile`` (parse -> PJRT compile -> serialized-executable artifact),
    ``load`` (deserialize -> ``XlaExecutable``). The acquisition path and
    every probed API are documented in ``xla_util.py`` — keep it updated
    when bumping jaxlib.
    """

    name: ClassVar[str] = "xla"
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
    )

    # ---------------------------------------------------------- availability

    @classmethod
    def check_available(cls) -> None:
        """Probe the jaxlib dependency; raise ``core.BackendError`` if absent.

        Checks (1) jaxlib imports, (2) the exact native API surface this
        adapter uses (``make_cpu_client``, ``batched_device_put``,
        ``compile_and_load``, ``deserialize_executable``, the MLIR dialect
        hooks), and (3) that an actual CPU PJRT client can be created.
        Raises ``core.BackendError`` with the ``pip install etl[xla]``
        hint — never a bare ImportError.
        """
        xc, _ = _import_xla_runtime()
        _verify_xla_api_surface(xc)
        _acquire_cpu_client(xc)  # definitive: a live CPU PJRT client exists

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
        3. Parse the MLIR text with jaxlib's MLIR bindings, then
           ``client.compile_and_load(module, executable_devices=...,
           compile_options=...)`` on the embedded CPU PJRT client.
        4. Serialize the loaded executable (``exe.serialize() -> bytes``)
           into a JSON-safe payload:
           ``{"format": "xla-serialized-executable", "mlir_text": ...,
           "executable_base64": ..., "entry_functions": ...,
           "static_input_shapes": ..., "static_output_shapes": ...}``.

        Conversion/compile errors are re-raised as ``core.BackendError``
        carrying the original message — never silent. ``compile`` NEVER
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

        # Dependency probe: compile itself creates a live CPU client below
        # (the definitive check — its failure raises BackendError), so only
        # the API-surface verification is needed here (no duplicate client).
        xc, jaxlib = _import_xla_runtime()
        _verify_xla_api_surface(xc)
        client = _acquire_cpu_client(xc)
        try:
            module = _parse_stablehlo_module(payload["mlir_text"])
            device_list = xc.DeviceList(tuple(client.devices()))
            compile_options = xc.CompileOptions()
            executable = client.compile_and_load(
                module,
                executable_devices=device_list,
                compile_options=compile_options,
            )
            serialized = executable.serialize()
        except core.BackendError:
            raise
        except Exception as exc:
            raise core.BackendError(
                f"the xla backend failed to compile the StableHLO program: "
                f"{exc}"
            ) from exc

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
                "jaxlib": jaxlib.__version__,
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
        self, artifact: CompiledArtifact, device: core.Device | None = None
    ) -> "XlaExecutable":
        """Reconstruct an ``XlaExecutable`` from a serialized artifact.

        Validates the recorded backend (``core.PersistenceError`` naming
        both on mismatch), the dependency, the device (None or a CPU
        ``core.Device``; non-``Device`` -> ``core.DeviceError``; non-cpu
        kind -> ``core.BackendError``), and the payload format. The
        base64 executable is deserialized with
        ``client.deserialize_executable(bytes, device_list)`` on a fresh
        embedded CPU PJRT client. NEVER re-traces, re-lowers, or
        re-compiles; a deserialization failure (environment/ABI mismatch)
        raises ``core.PersistenceError`` — no silent recompilation.
        """
        if artifact.backend != self.name:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                "the xla backend cannot load it"
            )
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

        xc, _ = _import_xla_runtime()
        client = _acquire_cpu_client(xc)
        device_list = xc.DeviceList(tuple(client.devices()))
        try:
            serialized = base64.b64decode(payload["executable_base64"])
            executable = client.deserialize_executable(serialized, device_list)
        except Exception as exc:
            raise core.PersistenceError(
                "failed to deserialize the XLA executable — the artifact "
                f"is incompatible with this environment/ABI: {exc} — never "
                "silently recompiling"
            ) from exc
        return XlaExecutable(
            artifact=artifact,
            device=effective_device,
            native_module=executable,
            entry_functions=tuple(payload.get("entry_functions", ())),
            client=client,
        )


class XlaExecutable(CompilerExecutable):
    """Run-time object for the xla backend (satisfies ``Executable``).

    ``native_module`` is the jaxlib ``LoadedExecutable``;
    ``run(flat_input_tensors)`` stages numpy host buffers through
    ``batched_device_put``, executes, and wraps the results as
    ``core.Tensor`` exactly like the numpy interpreter (``core.Tensor(
    np.asarray(result))``). ``save``/``load`` are the SHARED
    ``CompilerExecutable`` implementations (artifact round-trip; the
    executable is reconstructed explicitly at ``load`` — device handles
    are never serialized).
    """

    backend_name: ClassVar[str] = "xla"

    def __init__(
        self,
        artifact: CompiledArtifact | None = None,
        signature: Any = None,
        device: core.Device | None = None,
        native_module: Any = None,
        entry_functions: tuple[str, ...] = (),
        client: Any = None,
    ) -> None:
        super().__init__(
            artifact=artifact,
            signature=signature,
            device=device,
            native_module=native_module,
            entry_functions=entry_functions,
        )
        self._client = client
        self._put = None

    # -------------------------------------------------------------------- run

    def run(self, flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        Validates inputs EXACTLY against ``signature.input_specs``:
        count (``BackendError``), type, dtype (``DTypeError``) and the
        static shape recorded at compile time (``ShapeError`` — the xla
        adapter's shapes are static). Inputs are staged via
        ``batched_device_put`` (``enable_x64=True``), executed, and the
        output buffers are wrapped as ``core.Tensor`` exactly like the
        numpy interpreter. Output count/dtype/shape are validated against
        ``signature.output_specs``. A runtime failure raises
        ``core.BackendError`` naming the cause — never a silent fallback.
        """
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

        if self._put is None:
            self._put = _make_buffer_putter(self._client)
        buffers = [self._put(array) for array in arrays]
        try:
            outputs = self.native_module.execute(buffers)
        except Exception as exc:
            raise core.BackendError(f"XLA execution failed: {exc}") from exc

        import numpy as np

        tensors = [core.Tensor(np.asarray(buffer)) for buffer in outputs]

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
    """Probe the dependency and register the backend (idempotent).

    Called by ``etl.backends.registry.get("xla")`` on first use (and by
    persisted-artifact loads). Raises ``core.BackendError`` with the
    ``pip install etl[xla]`` hint when jaxlib is missing/incompatible;
    does nothing observable when already registered.
    """
    XlaBackend.check_available()
    _registry_register(xla_backend)
