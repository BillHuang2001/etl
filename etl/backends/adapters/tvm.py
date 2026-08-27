"""TVM compiler adapter: StableHLO MLIR -> Apache TVM Relax VM (CPU via LLVM).

This module implements the compiler-specific half of the pluggable-backend
seam (the shared frontend half — capability pre-check, block-portable
inlining, StableHLO export, ``Signature`` recording — lives in
``CompilerBackend.lower``, see ``../compiler.py``). Staging is explicit:

``etl.lower(graph, backend="tvm")``  -> ``LoweredProgram`` (StableHLO MLIR text)
``etl.compile(lowered)``             -> ``CompiledArtifact`` (host .so library)
``etl.load(artifact)``               -> ``TvmExecutable`` (reloads the library)
``etl.run(executable, *tensors)``    -> results (TVM VM execution)

The concrete APIs (validated on TVM 0.26.0 / tvm_ffi 0.1.13.post3 /
jaxlib 0.10.2 — see ``tvm_util.py`` for the full inventory). jaxlib is
used ONLY for its bundled LLVM MLIR python bindings, accessed exclusively
through the ``_mlir_bindings`` seam; the ``jax`` package is NOT used at
all (the vendored translator's ``jax._src`` import is satisfied by a
``sys.modules`` shim — ``tvm_util._install_jax_shim``):

- translate: ``tvm.relax.frontend.stablehlo.from_stablehlo(mlir_text)``
  (via the compatibility shim ``tvm_util.ensure_compat()``)
- build:     ``tvm.relax.vm_build.build(mod, target=tvm.target.Target("llvm"))``
- run:       ``tvm.runtime.vm.VirtualMachine(ex, tvm.runtime.cpu())["main"](...)``
- persist:   ``VMExecutable.export_library(path)`` -> base64 in the artifact;
  ``tvm.runtime.load_module(path)`` reloads WITHOUT recompiling at load
  (validated round-trip; ``Module.save_to_file`` does not exist in 0.26).

Device/target support (v1): CPU only, target ``"llvm"`` recorded as
``"cpu - llvm"``. A non-CPU device raises ``core.BackendError``.

Capabilities (each validated end-to-end — translate -> build -> run ->
numpy-backend parity on real etl graphs):

- ``dynamic_shapes=True``: symbolic dims (``tensor<?x8xf32>``) translate
  into Relax ``T.Var``-shaped tensors and run at multiple concrete sizes
  (validated with two sizes). Scalar-constant broadcasts over dynamic
  shapes (e.g. ``x * 2.0``) are handled: the writer emits
  ``stablehlo.dynamic_broadcast_in_dim`` with a
  ``get_dimension_size``-built ``output_dimensions`` chain, and the
  compat shim lowers it via ``multiply(data, full_like(source, 1))``
  (the Relax VM of 0.26 cannot codegen ``broadcast_to`` with a symbolic
  shape) — validated end-to-end at two concrete sizes.
- ``collectives=False``: the vendored translator has NO collective
  handlers — the shared ``lower()`` rejects collective-effect ops up
  front (``core.BackendError`` naming the op).
- ``runtime_calls=False`` / ``custom_blocks=False`` /
  ``async_collectives=False`` (the shared ``lower()`` pre-check rejects
  ``runtime_call`` / ``block_call`` / collective ops explicitly, naming
  the feature).
- ``dtypes``: every dtype below was run through a full etl graph with
  numpy parity. Complex64/128 are excluded (the vendored translator's
  ``_convert_data_type`` raises ``NotImplementedError`` for complex).

Op coverage is gated at compile time by a pre-check against the validated
whitelist (``tvm_util.SUPPORTED_STABLEHLO_OPS``): arithmetic, unary
math, comparisons, select, bitwise/logical, convert, broadcast/reshape/
transpose/concatenate/slice/pad, reduce (add/maximum/minimum/multiply
reducers), dot_general (matmul), constants. NOT supported (compile-time
``core.BackendError`` naming the op): control flow (``stablehlo.if``/
``while``), gather/scatter, remainder, convolution and reduce_window
(their vendored handlers hardcode NHWC/HWIO layouts — accepting them
would silently mis-compute etl's NCHW conv), multi-function modules and
multi-tensor-output functions (the vendored importer keeps only the first
output). Known Issues in ``./CONTEXT.md``.

Import rule (binding): no heavy imports at module top level — ``tvm`` and
``jaxlib`` are imported inside function bodies only (``tvm_util.py``
applies the same rule; ``numpy`` at top level is the library's single hard
runtime dependency, same as ``etl.core``); ``import etl`` never imports
the adapter or TVM.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from etl import core
from etl.core import Device

from ..backend import Capabilities
from ..compiler import CompilerBackend, CompilerExecutable
from ..program import CompiledArtifact, LoweredProgram
from ..registry import register as _registry_register
from . import tvm_util

__all__ = ["TvmBackend", "TvmExecutable", "tvm_backend", "register"]

#: Artifact payload format tag (self-describing; validated at load).
PAYLOAD_FORMAT = "tvm-vm-library"
PAYLOAD_FORMAT_VERSION = 1

#: dtypes validated end-to-end (translate -> vm build -> run -> numpy parity).
#: float16 constants are decoded via ``numpy.asarray`` in the compat shim
#: (the vendored per-element iteration crashes on them); complex dtypes are
#: not handled by the vendored translator and are excluded.
_VALIDATED_DTYPES: frozenset = frozenset(
    {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int8),
        np.dtype(np.int16),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.uint64),
        np.dtype(np.bool_),
    }
)


class TvmBackend(CompilerBackend):
    """TVM compiler adapter: consumes the shared StableHLO lowering.

    ``lower`` is inherited from ``CompilerBackend`` (verify -> capability
    pre-check -> portable inlining -> StableHLO export -> Signature ->
    MLIR payload); ``compile`` invokes the vendored StableHLO translator
    (compat-shimmed) and ``tvm.relax.vm_build.build``; ``load`` reloads
    the exported host library via ``tvm.runtime.load_module`` — never
    recompiling.
    """

    name: ClassVar[str] = "tvm"
    capabilities: ClassVar[Capabilities] = Capabilities(
        dynamic_shapes=True,
        dtypes=frozenset(_VALIDATED_DTYPES),
        collectives=False,
        runtime_calls=False,
        custom_blocks=False,
        async_collectives=False,
    )

    @classmethod
    def check_available(cls) -> None:
        """Probe TVM + its StableHLO frontend + the jaxlib MLIR bindings.

        See ``tvm_util`` — the ``jax`` package is never used (the vendored
        translator's ``jax._src`` import is satisfied by a shim). Raises
        ``core.BackendError`` with ``pip install etl[tvm]`` when the
        compiler dependency is missing or lacks the probed APIs.
        """
        tvm_util.check_available()

    # ------------------------------------------------------------------ stage

    def compile(
        self, lowered: LoweredProgram, options: dict | None = None
    ) -> CompiledArtifact:
        """LoweredProgram -> self-describing CompiledArtifact (TVM VM library).

        1. Validate ``lowered.backend == "tvm"`` (``core.BackendError``
           otherwise — never cross-backend compilation) and the shared
           StableHLO payload format.
        2. ``check_available`` + apply the compat shim.
        3. Parse the MLIR via the ``_mlir_bindings`` seam
           (``_mlir_bindings.make_ir_context``) and run the compile-time op
           gate (``tvm_util.precheck_module``) — unsupported ops /
           multi-output programs / invalid MLIR raise ``core.BackendError``
           BEFORE the translator runs.
        4. ``from_stablehlo(mlir_text)`` -> Relax module;
           ``tvm.relax.vm_build.build(mod, target=Target("llvm"))``.
           Translator/build errors re-raised as ``core.BackendError``.
        5. Persist the built executable: ``VMExecutable.export_library``
           to a temp file, base64 into the payload. ``load`` reloads the
           library — compile never loads, load never compiles.
        6. ``target="cpu - llvm"``; ``required_custom_ops=()`` (``block_call``
           graphs are rejected at lower time — ``custom_blocks=False``);
           ``runtime_dependencies`` records numpy + tvm versions
           (self-describing per the serialization contract).
        """
        if lowered.backend != "tvm":
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the tvm backend"
            )
        payload = lowered.payload
        if not isinstance(payload, dict) or payload.get("format") != "stablehlo":
            raise core.BackendError(
                "the tvm backend expects the shared stablehlo LoweredProgram "
                f"payload, got {type(payload).__name__}"
            )
        mlir_text = payload.get("mlir_text")
        if not isinstance(mlir_text, str):
            raise core.BackendError("corrupt: the stablehlo payload has no 'mlir_text'")
        entry_functions = tuple(payload.get("entry_functions") or ())
        self.check_available()
        tvm_util.ensure_compat()

        mlir_module = tvm_util.parse_stablehlo(mlir_text)
        tvm_util.precheck_module(mlir_module)
        relax_module = tvm_util.translate(mlir_text)
        executable = tvm_util.build_vm_executable(relax_module, target="llvm")
        library_base64 = tvm_util.export_library_base64(executable)

        runtime_dependencies = {
            "numpy": np.__version__,
            "tvm": tvm_util.tvm_version(),
        }
        artifact_payload = {
            "format": PAYLOAD_FORMAT,
            "format_version": PAYLOAD_FORMAT_VERSION,
            "mlir_text": mlir_text,
            "library_base64": library_base64,
            "entry_functions": entry_functions,
        }
        return CompiledArtifact(
            backend="tvm",
            signature=lowered.signature,
            target="cpu - llvm",
            payload=artifact_payload,
            required_custom_ops=(),
            runtime_dependencies=runtime_dependencies,
        )

    def load(
        self, artifact: CompiledArtifact, device: Device | None = None
    ) -> "TvmExecutable":
        """Reconstruct a TvmExecutable from an artifact. Never recompiles.

        1. Validate ``artifact.backend == "tvm"`` (``core.PersistenceError``
           otherwise).
        2. Device: ``None`` or a CPU ``Device`` — anything else
           ``core.DeviceError`` / ``core.BackendError`` (CPU only).
        3. ``check_available`` + payload/format validation + TVM version
           check against the recorded dependency (mismatch =>
           ``core.PersistenceError`` — never silently recompile).
        4. Decode the base64 host library to a temp file and reload via
           ``tvm.runtime.load_module`` -> ``VirtualMachine``.
        """
        if artifact.backend != "tvm":
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                "the tvm backend cannot load it"
            )
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got {type(device).__name__}"
                )
            if device.kind != "cpu":
                raise core.BackendError(
                    f"the tvm backend v1 supports CPU devices only, got "
                    f"device kind {device.kind!r}"
                )
        self.check_available()
        payload = artifact.payload
        if (
            not isinstance(payload, dict)
            or payload.get("format") != PAYLOAD_FORMAT
            or payload.get("format_version") != PAYLOAD_FORMAT_VERSION
        ):
            raise core.PersistenceError(
                f"corrupt: artifact payload is not a {PAYLOAD_FORMAT!r} "
                f"record — never silently recompile"
            )
        library_base64 = payload.get("library_base64")
        if not isinstance(library_base64, str):
            raise core.PersistenceError(
                "corrupt: tvm artifact payload has no 'library_base64'"
            )
        recorded_tvm = artifact.runtime_dependencies.get("tvm")
        current_tvm = tvm_util.tvm_version()
        if recorded_tvm is not None and recorded_tvm != current_tvm:
            raise core.PersistenceError(
                f"artifact requires tvm {recorded_tvm}, environment has "
                f"{current_tvm} — never silently recompile"
            )
        vm = tvm_util.load_virtual_machine(library_base64)
        entry_functions = tuple(payload.get("entry_functions") or ())
        return TvmExecutable(
            artifact=artifact,
            signature=artifact.signature,
            device=device,
            native_module=vm,
            entry_functions=entry_functions,
        )


class TvmExecutable(CompilerExecutable):
    """TVM VM runtime object (satisfies the backend ``Executable`` protocol).

    ``run`` feeds flat ``core.Tensor`` inputs through the reloaded
    ``VirtualMachine`` and returns flat ``core.Tensor`` outputs
    (constructed exactly like the numpy interpreter's kernels:
    ``core.Tensor(np.asarray(...))``). ``save``/``load`` are inherited
    from ``CompilerExecutable`` (artifact persistence + registry-routed
    reconstruction — validated round-trip).
    """

    backend_name: ClassVar[str] = "tvm"

    def run(self, flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        Validation mirrors the numpy interpreter's contract:
        (a) input count vs ``signature.input_specs`` (``BackendError``);
        (b) per-input ``core.Tensor`` type (``BackendError``) and dtype
        (``core.DTypeError`` — the same type the numpy interpreter uses
        for input dtype mismatches);
        (c) static shape dims must match exactly (``core.ShapeError``);
        symbolic dims pass through unchecked (``dynamic_shapes=True`` —
        validated: the VM accepts multiple concrete sizes);
        (d) outputs are validated against ``signature.output_specs``
        (count + dtype, ``BackendError``).
        """
        vm = self.native_module
        signature = self.signature
        input_specs = tuple(signature.input_specs) if signature is not None else ()
        if len(flat_input_tensors) != len(input_specs):
            raise core.BackendError(
                f"program expects {len(input_specs)} input tensor(s), got "
                f"{len(flat_input_tensors)}"
            )
        for i, (tensor, spec) in enumerate(zip(flat_input_tensors, input_specs)):
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
            # Shape contract mirrors the numpy interpreter's ``_bind_shape``
            # (rank must match exactly; static int dims must equal the
            # runtime dim; Dim/DimExpr/None dims pass through unchecked —
            # dynamic_shapes=True: the VM accepts multiple concrete sizes).
            declared = tuple(spec.shape)
            actual = tensor.shape
            if len(declared) != len(actual):
                raise core.ShapeError(
                    f"input {i}: rank mismatch — declared rank "
                    f"{len(declared)} vs runtime rank {len(actual)}"
                )
            for axis, expected in enumerate(declared):
                if isinstance(expected, int) and not isinstance(expected, bool):
                    if expected != actual[axis]:
                        raise core.ShapeError(
                            f"input {i}: static dimension {expected} does not "
                            f"match runtime dimension {actual[axis]}"
                        )

        tvm_inputs = [tvm_util.as_tvm_tensor(tensor.numpy()) for tensor in flat_input_tensors]
        outputs = tvm_util.invoke(vm, tvm_inputs)

        output_specs = tuple(signature.output_specs) if signature is not None else ()
        if len(outputs) != len(output_specs):
            raise core.BackendError(
                f"program produced {len(outputs)} output tensor(s), "
                f"expected {len(output_specs)}"
            )
        result: list[core.Tensor] = []
        for i, (array, spec) in enumerate(zip(outputs, output_specs)):
            if array.dtype != spec.dtype:
                raise core.BackendError(
                    f"output {i}: expected dtype {spec.dtype}, got "
                    f"{array.dtype}"
                )
            # Mirror the numpy interpreter kernels' canonical output
            # construction: core.Tensor(np.asarray(...)) on the CPU device.
            result.append(core.Tensor(np.asarray(array), device=self.device))
        return result


#: The module-level backend singleton (registered on ``register()``).
tvm_backend = TvmBackend()


def register() -> TvmBackend:
    """Probe the dependency and register the tvm backend (idempotent).

    Called by ``etl.backends.registry.get("tvm")`` on first use (the
    optional-adapter auto-activation path) and safe to call directly.
    Raises ``core.BackendError`` with the pip-install hint when TVM or
    its StableHLO frontend is unavailable.
    """
    TvmBackend.check_available()
    _registry_register(tvm_backend)
    return tvm_backend
