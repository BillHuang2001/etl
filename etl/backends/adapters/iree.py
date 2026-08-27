"""IREE compiler adapter — ``etl.backends.adapters.iree``.

Pluggable compiler backend that consumes the shared StableHLO payload
(``CompilerBackend.lower``, see ``../compiler.py``) and compiles it with the
external IREE compiler, producing an IREE VM flatbuffer executed through the
IREE runtime on the ``local-task`` CPU driver.

Exact IREE Python APIs used (all inside function bodies — the heavy-import
rule: ``iree`` / ``iree.compiler`` / ``iree.runtime`` / ``numpy`` are NEVER
imported at module top level, so ``import etl`` stays light):

- ``iree.compiler.compile_str(text, target_backends=["llvm-cpu"],
  input_type="stablehlo", extra_args=[...])`` — compiles the StableHLO MLIR
  payload in-process (spawns ``iree-compile``) and returns the VM flatbuffer
  as ``bytes``. ``extra_args`` are fixed:
  ``--iree-input-demote-f64-to-f32=false`` (IREE demotes f64 to f32 by
  default for StableHLO input — silent dtype coercion, which etl's error
  strategy forbids; disabling it keeps f64 semantics exact) and
  ``--iree-llvmcpu-target-cpu=generic`` (portable generic-CPU codegen; also
  silences IREE's generic-CPU warning).
- ``iree.runtime.get_driver("local-task")`` →
  ``driver.create_default_device()`` — the RELIABLE device-acquisition path.
  The historical ``rt.system_setup(config=rt.Config("local-task"))`` recipe
  fails with ``TypeError: 'module' object is not callable`` because
  ``iree.runtime.system_setup`` is a MODULE in iree-runtime 20241104 (its
  function was replaced by ``system_api.Config``). This path was validated
  repeatedly (fresh processes and in-process loops).
- ``iree.runtime.load_vm_flatbuffer(flatbuffer_bytes, driver="local-task")``
  — loads the VM flatbuffer into a ``BoundModule``; entry functions are
  resolved as attributes (``module.main``).
- ``iree.runtime.asdevicearray(device, np_array)`` — copies a host numpy
  array into a HAL device buffer for function input.
- ``np.asarray(result)`` — converts a returned ``DeviceArray`` back to a
  host numpy array; wrapped into ``core.Tensor`` exactly like the numpy
  interpreter's kernels (``core.Tensor(np.asarray(...))``).

Staging is explicit and honest: ``compile`` NEVER loads, ``load`` NEVER
re-lowers/re-compiles. Compiler failures surface as
``core.BackendError`` carrying the ``iree.compiler.CompilerToolError``
diagnostics — never silently swallowed or papered over.

Capability notes (all validated end-to-end on iree-compiler==20241104.1068 /
iree-runtime==20241104.1068, llvm-cpu, CPU only):

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
  HAL drivers of iree-runtime 20241104 raise "UNIMPLEMENTED; collectives not
  implemented" at ``hal.channel.create`` (the Python wheels ship no
  communicating channel provider). With this flag False, the SHARED
  ``lower()`` rejects every collective-effect op explicitly with
  ``core.BackendError`` naming it — never a silent fallback.
- ``dynamic_shapes=True``: a symbolic-dim graph compiles and runs correctly
  with different concrete sizes. Scalar-constant broadcasts over dynamic
  shapes (e.g. ``relu`` on a dynamic tensor) are emitted as
  ``stablehlo.dynamic_broadcast_in_dim`` with a
  ``get_dimension_size``-built runtime ``output_dimensions`` chain —
  validated end-to-end with iree-compile/runtime 20241104.1068 at
  multiple concrete sizes.

Import acyclicity (binding, ``../CONTEXT.md``): top-level imports restricted
to stdlib + ``etl.core`` + the sibling modules ``compiler`` / ``registry`` /
``program`` / ``backend``. ``etl.pipeline`` is never imported.
"""
from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from etl import core
from etl.core import Device

from ..backend import Capabilities
from ..compiler import CompilerBackend, CompilerExecutable
from ..program import CompiledArtifact, LoweredProgram, Signature
from ..registry import register as _registry_register

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


class IreeBackend(CompilerBackend):
    """IREE compiler backend: shared StableHLO lowering + IREE compilation.

    ``lower`` is inherited from ``CompilerBackend`` (verify -> capability
    pre-check -> portable inlining -> verify -> StableHLO export -> signature
    -> MLIR-text payload). This class adds the compiler-specific half:

    - ``check_available`` — probes the IREE packages; missing deps raise
      ``core.BackendError`` with the install hint ``pip install etl[iree]``.
    - ``compile`` — invokes ``iree.compiler.compile_str`` on the MLIR payload
      and records the VM flatbuffer (base64, JSON-safe) in the artifact.
    - ``load`` — decodes the flatbuffer, acquires the ``local-task`` runtime
      device, and builds an ``IreeExecutable``.

    Staging never composes: ``compile`` never loads; ``load`` never
    re-lowers/re-compiles.
    """

    name = "iree"
    capabilities = Capabilities(
        dynamic_shapes=True,
        dtypes=_dtype_capabilities(),
        collectives=False,
        runtime_calls=False,
        custom_blocks=False,
        async_collectives=False,
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
                f"(iree-compiler + iree-runtime), which are unavailable: "
                f"{exc}. Install them with `pip install etl[iree]`"
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
        4. ``iree.compiler.compile_str`` with
           ``target_backends=["llvm-cpu"]``, ``input_type="stablehlo"`` and
           ``extra_args=["--iree-input-demote-f64-to-f32=false",
           "--iree-llvmcpu-target-cpu=generic"]`` (f64 semantics preserved;
           portable generic-CPU codegen). ``iree.compiler.CompilerToolError``
           is re-raised as ``core.BackendError`` CARRYING the compiler
           diagnostics — honest, never silent.
        5. Record a self-describing ``CompiledArtifact``: JSON-safe payload
           (``format == "iree-vmfb"``, the MLIR text, the base64 VM
           flatbuffer, the entry-function names), ``required_custom_ops=()``
           (the shared ``lower`` already inlined every portable block), and
           ``runtime_dependencies`` (numpy + IREE package versions).

        ``compile`` NEVER loads (no runtime is touched here).
        """
        if lowered.backend != self.name:
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the {self.name} backend — never "
                "silently re-lower"
            )
        payload = lowered.payload
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

        import importlib.metadata as metadata

        import iree.compiler as iree_compiler
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
        try:
            vmfb = iree_compiler.compile_str(
                mlir_text,
                target_backends=["llvm-cpu"],
                input_type="stablehlo",
                extra_args=[
                    "--iree-input-demote-f64-to-f32=false",
                    "--iree-llvmcpu-target-cpu=generic",
                ],
            )
        except iree_compiler.CompilerToolError as exc:
            raise core.BackendError(
                f"iree-compile failed to compile the StableHLO program for "
                f"the {self.name} backend:\n{exc}"
            ) from exc

        artifact_payload = {
            "format": "iree-vmfb",
            "format_version": 1,
            "mlir_text": mlir_text,
            "vmfb_base64": base64.b64encode(vmfb).decode("ascii"),
            "entry_functions": entry_functions,
        }
        return CompiledArtifact(
            backend=self.name,
            signature=lowered.signature,
            target="cpu",
            payload=artifact_payload,
            required_custom_ops=(),
            runtime_dependencies={
                "numpy": np.__version__,
                "iree-compiler": metadata.version("iree-compiler"),
                "iree-runtime": metadata.version("iree-runtime"),
            },
        )

    def load(
        self, artifact: CompiledArtifact, device: Device | None = None
    ) -> "IreeExecutable":
        """Reconstruct an ``IreeExecutable`` from an artifact. Never re-compiles.

        1. Validate ``artifact.backend == "iree"`` (mismatch =>
           ``core.PersistenceError`` naming both — artifacts are never
           silently reinterpreted).
        2. Validate the payload (``format == "iree-vmfb"``, base64 vmfb,
           entry functions; malformed => ``core.BackendError`` "corrupt").
        3. ``check_available``.
        4. Validate the device: ``None`` or a CPU ``core.Device`` — a
           non-``Device`` object raises ``core.DeviceError``; another kind
           raises ``core.BackendError`` naming it (v1 CPU only).
        5. Base64-decode the VM flatbuffer; acquire the runtime device via
           the reliable path (``rt.get_driver("local-task")`` +
           ``create_default_device()``); ``load_vm_flatbuffer``; resolve the
           v1 entry function (``"main"``, else the single recorded entry).

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
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got "
                    f"{type(device).__name__}"
                )
            if device.kind != "cpu":
                raise core.BackendError(
                    f"the {self.name} backend v1 supports CPU devices only, "
                    f"got device kind {device.kind!r}"
                )

        import iree.runtime as rt

        vmfb = base64.b64decode(payload["vmfb_base64"].encode("ascii"))
        driver = rt.get_driver("local-task")
        runtime_device = driver.create_default_device()
        module = rt.load_vm_flatbuffer(vmfb, driver="local-task")
        return IreeExecutable(
            artifact=artifact,
            signature=artifact.signature,
            device=device,
            module=module,
            entry_functions=entry_functions,
            runtime_device=runtime_device,
        )


class IreeExecutable(CompilerExecutable):
    """IREE runtime executable (satisfies the backend ``Executable`` protocol).

    Runs the compiled VM module on the ``local-task`` CPU driver:

    1. Validate flat inputs against the recorded ``Signature``: count
       (``core.BackendError``), per-input dtype (``core.BackendError``),
       per-input shape — rank and STATIC dims must match
       (``core.ShapeError``); symbolic dims pass through (IREE executes
       runtime-dynamic shapes — validated).
    2. Copy each input into a HAL buffer via ``rt.asdevicearray``.
    3. Invoke the entry function; handle single/multiple results (the
       invoker returns the value or a tuple).
    4. Convert each result via ``np.asarray`` and wrap it in a
       ``core.Tensor`` — the same construction pattern the numpy
       interpreter kernels use (``core.Tensor(np.asarray(...))``).
    5. Validate output count + dtype against the signature output specs.

    ``save`` / ``load`` are inherited from ``CompilerExecutable``: ``save``
    persists the underlying ``CompiledArtifact`` (device handles are NEVER
    serialized — reconstruction is explicit); ``load`` reads the artifact,
    validates its backend, and routes reconstruction through the registry
    (which auto-activates this adapter lazily).
    """

    backend_name = "iree"

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

    # ------------------------------------------------------------------ run

    def run(self, flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        See the class docstring for the exact validation/invocation model.
        Runtime IREE errors propagate as-is (they are explicit hardware/runtime
        failures, never silently swallowed); etl-level mismatches raise
        ``core.BackendError`` / ``core.ShapeError``.
        """
        import iree.runtime as rt
        import numpy as np

        signature = self.signature
        if signature is not None:
            input_specs = tuple(signature.input_specs)
            output_specs = tuple(signature.output_specs)
        else:
            input_specs = ()
            output_specs = ()
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

        buffers = [
            rt.asdevicearray(self.runtime_device, tensor.numpy())
            for tensor in flat_input_tensors
        ]
        entry = self._entry_function()
        results = entry(*buffers)
        if results is None:
            results = ()
        elif not isinstance(results, tuple):
            results = (results,)
        outputs = [core.Tensor(np.asarray(result)) for result in results]

        if signature is not None:
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

        Raises:
            core.BackendError: no ``"main"`` function and not exactly one
                recorded entry function.
        """
        if "main" in self._entry_functions:
            return getattr(self.native_module, "main")
        if len(self._entry_functions) == 1:
            return getattr(self.native_module, self._entry_functions[0])
        raise core.BackendError(
            f"the {self.backend_name} executable has "
            f"{len(self._entry_functions)} entry functions and no 'main' "
            "entry function"
        )


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
