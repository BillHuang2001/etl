"""Shared pluggable compiler-backend framework.

``CompilerBackend`` is the common base for compiler adapters (IREE, XLA via
PJRT, TVM — implemented in ``./adapters/`` as SEPARATE modules). It shares
the frontend half of the staging pipeline across all compilers:

- the SAME block-call portable inlining every backend uses
  (``inline.py::inline_portables``);
- the SAME capability pre-check pattern as the numpy reference backend
  (``runtime_call`` / ``collective`` effects vs ``Capabilities``);
- the SAME ``Signature`` recording as ``NumpyBackend.lower``;
- a StableHLO-MLIR ``LoweredProgram`` payload — the honest capability gate:
  the exporter raises ``core.BackendError`` naming any op it cannot emit,
  so ``lower`` already tells the user whether the adapter could in
  principle consume the program.

Adapters only implement the compiler-specific half: ``check_available``
(dependency probe), ``compile`` (invoke the external compiler on the MLIR
payload) and ``load`` (build the executable from the artifact). The staging
methods NEVER compose: ``lower`` never compiles, ``compile`` never loads,
``load`` never re-lowers/re-compiles; backend/device/ABI mismatches fail
clearly (``core.BackendError`` / ``core.PersistenceError``).

Import acyclicity (binding, see ``../CONTEXT.md``): top-level imports
restricted to ``etl.core`` plus the sibling modules ``backend.py``,
``program.py``, ``registry.py``, ``inline.py``. ``etl.backends.stablehlo``
(the exporter) and ``etl.backends.numpy.interpreter`` (``entry_function``)
are imported INSIDE function bodies; never import ``etl.pipeline`` or any
adapter module at top level.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from etl import core
from etl.core import Device

from .backend import Backend, Capabilities, Executable
from .inline import inline_portables, iter_ops
from .program import CompiledArtifact, LoweredProgram, Signature

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.trace import Graph

__all__ = ["CompilerBackend", "CompilerExecutable"]


class CompilerBackend(Backend):
    """Shared base for compiler backends that consume StableHLO MLIR.

    Subclasses declare ``name`` and ``capabilities`` and implement
    ``check_available`` / ``compile`` / ``load``. ``lower`` is shared:
    verify -> capability pre-check -> portable inlining -> verify ->
    StableHLO export -> ``Signature`` recording -> MLIR-text payload.

    The lower-time contract (binding):

    - ``graph.verify()`` surfaces ``core.VerificationError`` as-is
      (before AND after inlining — defensive).
    - Capability pre-check on the LIVE op walk (``inline.iter_ops``):
      ``runtime_call`` requires ``capabilities.runtime_calls``;
      ``collective``-effect ops require ``capabilities.collectives`` —
      violations raise ``core.BackendError`` naming the op. ``block_call``
      needs NO ``custom_blocks`` flag: a compiler adapter has no per-backend
      block impls, so EVERY ``block_call`` must have a portable
      decomposition and is inlined by ``inline_portables(module,
      keep_backend_impls=None)`` — a block without one raises
      ``core.BackendError`` naming it.
    - The StableHLO export is the capability gate for ops: deferred ops
      raise ``core.BackendError`` naming them (see ``stablehlo/`` v1 scope).
    - The ``Signature`` is recorded from the Graph's LIVE attributes
      (input/output TreeSpec + per-leaf specs + static values) — passed
      down, never re-derived.
    - The payload is JSON-safe (persist-container round-trip):
      ``{"format": "stablehlo", "format_version": 1, "mlir_text": ...,
      "entry_functions": ...}``.

    How to add a new adapter (the documented pluggability seam):

    1. Subclass ``CompilerBackend``; declare ``name`` (e.g. ``"iree"``)
       and ``capabilities`` (a ``Capabilities``).
    2. Implement ``check_available`` — raise ``core.BackendError`` with a
       pip-install hint (e.g. ``pip install etl[iree]``) when the adapter's
       compiler dependency is unavailable; do nothing when available.
    3. Implement ``compile`` — validate ``lowered.backend == self.name``
       (``core.BackendError`` otherwise), invoke the external compiler on
       the MLIR payload, and produce a self-describing ``CompiledArtifact``.
       ``compile`` NEVER loads.
    4. Implement ``load`` — validate ``artifact.backend`` / device, build
       your ``CompilerExecutable`` subclass from the artifact. ``load``
       NEVER re-lowers or re-compiles; mismatches raise
       ``core.PersistenceError`` (or ``core.BackendError`` for unsupported
       devices).
    5. At the module bottom declare the singleton (e.g.
       ``iree_backend = IreeBackend()``) and a module-level ``register()``
       function that calls ``registry.register(iree_backend)``
       (idempotent). ``registry.get("iree")`` auto-imports + auto-activates
       the adapter module on first use (``registry.OPTIONAL_ADAPTERS``) —
       ``import etl`` never imports the adapter.
    6. Subclass ``CompilerExecutable`` for the run-time object: declare
       ``backend_name``, implement ``run``; ``CompilerExecutable.load``
       reconstructs it through the registry.
    """

    name: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities()

    @classmethod
    @abstractmethod
    def check_available(cls) -> None:
        """Probe the compiler dependency; raise if unavailable.

        Raises ``core.BackendError`` with a pip-install hint (e.g.
        ``pip install etl[iree]``) when the adapter's compiler dependency
        is missing in this environment; does nothing when available.
        Adapters call this from ``compile``/``load`` (or their
        ``register()``) so a missing dependency fails with an actionable
        message instead of an obscure ``ImportError`` deep inside the
        vendor API.
        """
        ...

    def lower(self, graph: "Graph", options: dict | None = None) -> LoweredProgram:
        """Shared lowering for compiler adapters: Graph -> StableHLO payload.

        1. ``graph.verify()`` — surfaces ``core.VerificationError`` as-is.
        2. Capability pre-check on the CURRENT module state (live walk via
           ``inline.iter_ops`` — the module is mutated by inlining in
           step 3): ``runtime_call`` and ``collective``-effect ops are
           checked against ``Capabilities`` here (mirrors the numpy
           backend's flag pattern). ``block_call`` requires NO flag — every
           block_call gets inlined in step 3.
        3. ``inline_portables(graph.module, keep_backend_impls=None)`` — a
           compiler adapter has NO per-backend block impls; every
           ``block_call`` MUST have a portable decomposition, else
           ``core.BackendError`` naming the block.
        4. ``graph.verify()`` again (defensive, cheap).
        5. ``stablehlo.export(graph)`` — the capability gate for ops:
           deferred ops raise ``core.BackendError`` naming them.
        6. Record the ``Signature`` EXACTLY like ``NumpyBackend.lower``
           (input/output TreeSpec + per-leaf specs + static values, from
           the Graph's LIVE attributes).
        7. Payload = JSON-safe StableHLO text record (see class docstring).

        Explicit staging: ``lower`` NEVER compiles (no compiler is invoked
        here); raising ``core.VerificationError`` / ``core.BackendError`` —
        no silent fallbacks or partial semantics.
        """
        graph.verify()  # surfaces core.VerificationError as-is

        capabilities = self.capabilities
        for op in iter_ops(graph.module):
            if op.is_terminator:
                continue  # 'return' carries no capability risk
            if op.name == "runtime_call" and not capabilities.runtime_calls:
                raise core.BackendError(
                    f"capability drift: the {self.name} backend cannot "
                    "execute runtime_call"
                )
            if op.effect == "collective" and not capabilities.collectives:
                raise core.BackendError(
                    f"capability drift: the {self.name} backend cannot "
                    f"execute collective op '{op.name}'"
                )
            # block_call: NO custom_blocks flag requirement — a compiler
            # adapter has no per-backend block impls, so step 3 inlines
            # every block_call via its portable decomposition (and raises
            # for any block without one).

        inline_portables(graph.module, keep_backend_impls=None)
        graph.verify()  # defensive post-inline verification

        from .stablehlo import export

        mlir_text = export(graph)

        from etl.backends.numpy.interpreter import entry_function

        main_fn = entry_function(graph.module)
        signature = Signature(
            input_tree=graph.input_specs,
            output_tree=graph.output_tree,
            input_specs=tuple(graph.tensor_specs),
            output_specs=tuple(
                core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
                for value_type in main_fn.output_types
            ),
            static_values=tuple(record.value for record in graph.static_values),
            output_static_values=tuple(
                record.value for record in graph.output_static_values
            ),
        )
        payload = {
            "format": "stablehlo",
            "format_version": 1,
            "mlir_text": mlir_text,
            "entry_functions": tuple(fn.name for fn in graph.module.functions),
        }
        return LoweredProgram(backend=self.name, signature=signature, payload=payload)

    @abstractmethod
    def compile(
        self, lowered: LoweredProgram, options: dict | None = None
    ) -> CompiledArtifact:
        """Turn a LoweredProgram into a self-describing CompiledArtifact.

        Adapter implementations MUST validate ``lowered.backend ==
        self.name`` (``core.BackendError`` otherwise), invoke the external
        compiler on the MLIR payload, and record required custom ops /
        runtime dependencies (self-describing per the serialization
        contract). ``compile`` NEVER loads.
        """
        ...

    @abstractmethod
    def load(self, artifact: CompiledArtifact, device: Device | None = None) -> Executable:
        """Reconstruct an adapter executable from an artifact.

        Adapter implementations validate ``artifact.backend`` / device /
        ABI (mismatch => ``core.PersistenceError`` — or
        ``core.BackendError`` for unsupported devices) and build the
        executable. ``load`` NEVER re-traces, re-lowers, or re-compiles.
        """
        ...


class CompilerExecutable(ABC):
    """Base class for compiler-adapter executables (satisfies ``Executable``).

    Adapters subclass this, declare ``backend_name`` (the adapter's
    registered name), and implement ``run``. ``save`` / ``load`` are
    shared: ``save`` persists the underlying ``CompiledArtifact`` (device
    handles and live compiler state are NEVER serialized — the executable
    is reconstructed explicitly at ``load``); ``load`` reads the artifact,
    validates its recorded backend against ``backend_name``, and routes
    reconstruction through the registry (which auto-activates the adapter
    module lazily — the adapter class is never imported at module import
    time).

    Attributes:
        functions: tuple of entry-function names (``entry_functions``).
        device: runtime device the executable was loaded for (or ``None``).
        signature: input/output contract (``Signature``; falls back to the
            artifact's recorded signature).
        artifact: the underlying ``CompiledArtifact`` — what ``save``
            persists.
        native_module: the adapter-specific compiled module handle.
    """

    backend_name: ClassVar[str] = ""

    def __init__(
        self,
        artifact: CompiledArtifact | None = None,
        signature: Signature | None = None,
        device: core.Device | None = None,
        native_module: Any = None,
        entry_functions: tuple[str, ...] = (),
    ) -> None:
        self.artifact = artifact
        if signature is None and artifact is not None:
            signature = artifact.signature
        self.signature = signature
        self.device = device
        self.native_module = native_module
        self._entry_functions = tuple(entry_functions)

    @property
    def functions(self) -> tuple[str, ...]:
        """The program's entry-function names (from ``entry_functions``)."""
        return self._entry_functions

    @abstractmethod
    def run(self, flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]:
        """Execute the compiled program on flat input tensors.

        Adapter subclasses implement this: feed the tensors through the
        native compiled module and return the flat output tensors. Raises
        ``core.BackendError`` / ``core.ShapeError`` for runtime
        mismatches — never a silent fallback.
        """
        ...

    def save(self, path: str | os.PathLike) -> None:
        """Save the underlying CompiledArtifact.

        The executable is reconstructed EXPLICITLY at ``load`` — device
        handles and live compiler state are never serialized. Delegates to
        ``CompiledArtifact.save`` (etl.persist container, lazy import).
        """
        if self.artifact is None:
            raise core.BackendError(
                f"this {type(self).__name__} has no CompiledArtifact to "
                "save — construct it via the backend's load(artifact) or "
                "pass artifact= to the constructor"
            )
        self.artifact.save(path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: Device | None = None) -> "CompilerExecutable":
        """Reconstruct an executable from a saved artifact.

        Reads the ``CompiledArtifact``, validates its recorded backend
        against ``backend_name`` (mismatch => ``core.PersistenceError``
        naming both), and delegates reconstruction to the registered
        backend — the registry auto-activates the adapter module lazily
        (the adapter class is never imported at module import time).
        NEVER silently re-compiles.
        """
        artifact = CompiledArtifact.load(path)
        if artifact.backend != cls.backend_name:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                f"{cls.backend_name} executables cannot load it"
            )
        from .registry import get

        backend = get(cls.backend_name)
        return backend.load(artifact, device)
