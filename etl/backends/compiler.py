"""Shared pluggable compiler-backend framework.

``CompilerBackend`` is the common base for compiler adapters (IREE, XLA via
PJRT, TVM — implemented in ``./adapters/`` as SEPARATE modules). It shares
the frontend half of the staging pipeline across all compilers:

- the SAME block-call portable inlining every backend uses
  (``inline.py::inline_portables``);
- the SAME capability pre-check pattern as the numpy reference backend
  (``runtime_call`` / ``collective`` effects / ``block_call`` / sparse
  ops (category "sparse") / dtypes / dynamic shapes vs ``Capabilities``
  — every rejection names the feature);
- the SAME ``Signature`` recording as ``NumpyBackend.lower``;
- a StableHLO-MLIR ``LoweredProgram`` payload — the honest capability gate:
  the exporter raises ``core.BackendError`` naming any op it cannot emit,
  so ``lower`` already tells the user whether the adapter could in
  principle consume the program.

Adapters only implement the compiler-specific half: ``check_available``
(optional dependency probe — the base class provides a no-op default),
``compile`` (invoke the external compiler on the MLIR payload) and ``load``
(build the executable from the artifact). The staging methods NEVER
compose: ``lower`` never compiles, ``compile`` never loads, ``load`` never
re-lowers/re-compiles; backend/device/ABI mismatches fail clearly
(``core.BackendError`` / ``core.PersistenceError``).

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
    ``compile`` / ``load`` (and optionally override ``check_available`` —
    concrete, no-op by default). ``lower`` is shared: verify ->
    capability pre-check -> portable inlining -> verify -> StableHLO
    export -> ``Signature`` recording -> MLIR-text payload.

    The lower-time contract (binding):

    - ``graph.verify()`` surfaces ``core.VerificationError`` as-is
      (before AND after inlining — defensive).
    - Capability pre-check on the LIVE op walk (``inline.iter_ops``),
      every rejection naming the missing feature:
      ``runtime_call`` requires ``capabilities.runtime_calls``;
      ``collective``-effect ops require ``capabilities.collectives``;
      ``block_call`` requires ``capabilities.custom_blocks``; ops in the
      ``"sparse"`` category require ``capabilities.sparse_ops``; every
      value dtype must be declared in ``capabilities.dtypes``; when
      ``capabilities.dynamic_shapes`` is False, symbolic /
      runtime-dynamic (``None``) dimensions are rejected — all raise
      ``core.BackendError`` naming the feature.
    - Portable inlining: with ``custom_blocks=True`` every remaining
      ``block_call`` is inlined by ``inline_portables(module,
      keep_backend_impls=None)`` (a compiler adapter has no per-backend
      block impls) — a block without a portable decomposition raises
      ``core.BackendError`` naming it.
    - The StableHLO export is the capability gate for ops: deferred ops
      raise ``core.BackendError`` naming them (see ``stablehlo/`` v1 scope).
    - The ``Signature`` is recorded from the Graph's LIVE attributes
      (input/output TreeSpec + per-leaf specs + static values) — passed
      down, never re-derived.
    - The payload is JSON-safe (persist-container round-trip):
      ``{"format": "stablehlo", "format_version": 1, "mlir_text": ...,
      "entry_functions": [...]}`` (``entry_functions`` is a LIST).

    How to add a new adapter (the documented pluggability seam):

    1. Subclass ``CompilerBackend``; declare ``name`` (e.g. ``"iree"``)
       and ``capabilities`` (a ``Capabilities``).
    2. Implement ``compile`` — validate ``lowered.backend == self.name``
       (``core.BackendError`` otherwise), invoke the external compiler on
       the MLIR payload, and produce a self-describing ``CompiledArtifact``.
       ``compile`` NEVER loads.
    3. Implement ``load`` — validate ``artifact.backend`` / device, build
       your ``CompilerExecutable`` subclass from the artifact. ``load``
       NEVER re-lowers or re-compiles; mismatches raise
       ``core.PersistenceError`` (or ``core.BackendError`` for unsupported
       devices).
    4. OPTIONAL: override the ``check_available`` classmethod when the
       adapter has an external compiler dependency — raise
       ``core.BackendError`` with a pip-install hint (e.g.
       ``pip install etl[iree]``) when the dependency is unavailable; do
       nothing when available. The base implementation is a concrete
       no-op (returns ``None``), so subclasses with no dependency to
       probe need not implement it.
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

    #: Known per-stage option names (the options-override contract — see
    #: ``../options.py``): each adapter OVERRIDES this class attribute with
    #: its compile/load/run sets (e.g. iree declares ``iree_compile_args`` at
    #: compile and ``iree_runtime_args`` at load). Stage methods validate the
    #: caller's options dict against the UNION of these sets before doing
    #: anything else — a key valid for no stage raises ``core.BackendError``
    #: (loud, never silent); keys valid for other stages are accepted and
    #: ignored at this stage (the build/evaluate sugar forwards one options
    #: dict to every stage).
    KNOWN_OPTIONS: dict[str, frozenset[str]] = {
        "lower": frozenset({"rng_bit_generator"}),
        "compile": frozenset(),
        "load": frozenset(),
        "run": frozenset(),
    }

    @classmethod
    def check_available(cls) -> None:
        """Probe the compiler dependency; raise if unavailable (default: no-op).

        Adapters with an external compiler dependency override this to
        probe it and raise ``core.BackendError`` with a pip-install hint
        (e.g. ``pip install etl[iree]``) when the dependency is missing in
        this environment; do nothing when available. Adapters call this
        from ``compile``/``load`` (or their ``register()``) so a missing
        dependency fails with an actionable message instead of an obscure
        ``ImportError`` deep inside the vendor API.

        The DEFAULT implementation is a concrete no-op (returns ``None``):
        subclasses without a dependency to probe (e.g. lightweight or test
        subclasses) need only implement ``compile`` / ``load``.
        """
        return None

    @staticmethod
    def _is_static_dim(dim: Any) -> bool:
        """True for concrete int dims (bool excluded — it is not a size)."""
        return isinstance(dim, int) and not isinstance(dim, bool)

    @classmethod
    def _shape_is_static(cls, shape: Any) -> bool:
        return all(cls._is_static_dim(d) for d in shape)

    def _check_value_dtype(self, dtype: Any, where: str) -> None:
        """Raise ``core.BackendError`` when ``dtype`` is not declared."""
        capabilities = self.capabilities
        if dtype in capabilities.dtypes:
            return
        import numpy as np

        raise core.BackendError(
            f"capability drift: the {self.name} backend cannot execute "
            f"dtype {np.dtype(dtype).name} ({where}) — its declared dtypes "
            f"are {sorted(d.name for d in capabilities.dtypes)}"
        )

    def _check_static_shape(self, shape: Any, where: str) -> None:
        """Raise ``core.BackendError`` when the shape has dynamic dims but
        ``capabilities.dynamic_shapes`` is False."""
        if self.capabilities.dynamic_shapes or self._shape_is_static(shape):
            return
        raise core.BackendError(
            f"capability drift: the {self.name} backend cannot execute "
            f"dynamic shapes ({where} has symbolic/runtime-dynamic "
            "dimensions)"
        )

    def lower(self, graph: "Graph", options: dict | None = None) -> LoweredProgram:
        """Shared lowering for compiler adapters: Graph -> StableHLO payload.

        1. ``graph.verify()`` — surfaces ``core.VerificationError`` as-is.
        2. Capability pre-check on the CURRENT module state (live walk via
           ``inline.iter_ops`` — the module is mutated by inlining in
           step 3): ``runtime_call`` / ``collective``-effect ops /
           ``block_call`` / sparse ops (category "sparse" vs
           ``capabilities.sparse_ops``) / dtypes / dynamic shapes are
           checked against ``Capabilities`` here (mirrors the numpy
           backend's flag pattern). Every rejection names the missing
           feature.
        3. ``inline_portables(graph.module, keep_backend_impls=None)`` — for
           adapters declaring ``custom_blocks=True`` every remaining
           ``block_call`` MUST have a portable decomposition, else
           ``core.BackendError`` naming the block (adapters with
           ``custom_blocks=False`` have already rejected ``block_call``
           at step 2).
        4. ``graph.verify()`` again (defensive, cheap).
        5. ``stablehlo.export(graph, options={"rng_bit_generator": ...})`` —
           the capability gate for ops: deferred ops raise
           ``core.BackendError`` naming them; the ``rng_bit_generator``
           exporter option selects the native ``stablehlo.rng_bit_generator``
           emission per random algorithm (threefry2x32/philox4x32_10; absent
           from the set → the exporter's bit-exact inline expansions, always
           available). The caller's ``options`` dict may carry the RESERVED
           ``rng_bit_generator`` key (a bool or a collection of algorithm
           names) which OVERRIDES the capability — otherwise the
           ``Capabilities.rng_bit_generator`` set is used. The pipeline
           ``lower``/``build``/``evaluate`` sugar forwards it; ``compile()``
           ignores unknown keys (this key is lower-only).
        6. Record the ``Signature`` EXACTLY like ``NumpyBackend.lower``
           (input/output TreeSpec + per-leaf specs + static values, from
           the Graph's LIVE attributes).
        7. Payload = JSON-safe StableHLO text record (see class docstring).

        Explicit staging: ``lower`` NEVER compiles (no compiler is invoked
        here); raising ``core.VerificationError`` / ``core.BackendError`` —
        no silent fallbacks or partial semantics.

        Options contract: ``options`` is validated against the UNION of this
        backend's ``KNOWN_OPTIONS`` sets before anything else — a key valid
        for no stage raises ``core.BackendError`` naming the known options
        (see ``../options.py``); keys valid for other stages (e.g. the
        ``compile`` option ``iree_compile_args``) are accepted and ignored
        here, so ``build``/``evaluate`` can forward one options dict.
        """
        from .options import validate_options

        validate_options(options, self.KNOWN_OPTIONS, self.name, "lower")
        graph.verify()  # surfaces core.VerificationError as-is

        capabilities = self.capabilities
        for function in graph.module.functions:
            for value_type in (*function.input_types, *function.output_types):
                self._check_value_dtype(value_type.dtype, "function signature")
                self._check_static_shape(value_type.shape, "function signature")
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
            if op.name == "block_call" and not capabilities.custom_blocks:
                raise core.BackendError(
                    f"capability drift: the {self.name} backend cannot "
                    "execute custom block ops (block_call)"
                )
            if op.opdef.category == "sparse" and not capabilities.sparse_ops:
                raise core.BackendError(
                    f"capability drift: the {self.name} backend cannot "
                    f"execute sparse op '{op.name}' — sparse ops are "
                    "numpy-backend-only in v1; densify with "
                    "etl.sparse.to_dense (or convert the sparse computation "
                    "into dense ops) before lowering to a compiler backend"
                )
            for result in op.results:
                self._check_value_dtype(result.type.dtype, f"op '{op.name}'")
                self._check_static_shape(result.type.shape, f"op '{op.name}'")

        inline_portables(graph.module, keep_backend_impls=None)
        graph.verify()  # defensive post-inline verification

        from .stablehlo import export

        # The reserved per-call ``rng_bit_generator`` key (bool or collection
        # of algorithm names) overrides the capability — the pipeline
        # lower/build/evaluate sugar forwards it; compile() ignores unknown
        # keys, so it is lower-only.
        rng_option = (options or {}).get(
            "rng_bit_generator", self.capabilities.rng_bit_generator
        )
        mlir_text = export(
            graph, options={"rng_bit_generator": rng_option}
        )

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
            "entry_functions": list(fn.name for fn in graph.module.functions),
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
