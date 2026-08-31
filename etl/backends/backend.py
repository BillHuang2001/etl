"""Backend abstraction layer: Capabilities, Backend ABC, backend Executable protocol.

Import acyclicity (binding, see ../CONTEXT.md and ../../CONTEXT.md):
- Top-level imports restricted to ``etl.core`` (and ``etl.ir`` only under TYPE_CHECKING).
- ``etl.ops`` may be imported ONLY inside function bodies (block-impl registration).
- Never import ``etl.pipeline`` (pipeline imports this package).

Backend limitations fail explicitly: raise ``core.BackendError`` — never silently
fall back to another backend or a partial implementation.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from etl.core import BackendError, Device, Tensor

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.ir import Module  # noqa: F401
    from etl.trace import Graph
    from .program import CompiledArtifact, LoweredProgram

__all__ = ["Capabilities", "Backend", "Executable"]


@dataclass(frozen=True)
class Capabilities:
    """Static capability declaration of a backend.

    Pipeline code consults these flags BEFORE handing unsupported work to a
    backend and raises ``BackendError`` proactively; backends also re-check
    defensively inside their staging methods. A ``False`` flag means the
    feature must be rejected explicitly — never emulated silently.

    Attributes:
        dynamic_shapes: supports runtime-dynamic (symbolic) dims.
        dtypes: frozenset of supported numpy dtypes.
        collectives: supports ``etl.dist`` collective ops (numpy: single-process simulation).
        runtime_calls: supports ``runtime_call`` (executing Python callbacks).
        external_calls: supports ``external_call`` ops (named external
            kernels; v1: numpy-backend-only — compiler-backend host-dispatch
            is not wired yet, see ``etl/CONTEXT.md`` "External kernels").
        custom_blocks: supports ``block_call`` ops (registered backend impls).
        async_collectives: collective execution may be asynchronous (numpy: False).
        sparse_ops: supports etl.sparse sparse-tensor ops (numpy: True).
        rng_bit_generator: frozenset of random algorithm names for which the
            backend supports bit-exact native ``stablehlo.rng_bit_generator``
            emission. Canonical names: ``"threefry2x32"`` and
            ``"philox4x32_10"`` (splitmix64 has no native form — always the
            exporter's inline expansion). The empty set = inline expansions
            only. Per-adapter status: iree ``{"threefry2x32"}`` (native
            THREE_FRY verified bit-exact on llvm-cpu AND cuda; PHILOX fails
            iree legalization on both targets), xla
            ``{"threefry2x32", "philox4x32_10"}`` by design — re-validate
            with a real PJRT plugin, tvm ``frozenset()`` (no
            rng_bit_generator support), numpy ``frozenset()`` (no StableHLO
            emission).
    """

    dynamic_shapes: bool = False
    dtypes: frozenset = field(default_factory=frozenset)
    collectives: bool = False
    runtime_calls: bool = False
    external_calls: bool = False
    custom_blocks: bool = False
    async_collectives: bool = False
    sparse_ops: bool = False
    rng_bit_generator: frozenset = field(default_factory=frozenset)

    def supports_dtype(self, dtype: Any) -> bool:
        """True iff ``dtype`` is among ``self.dtypes`` (numpy dtype equality)."""
        return any(dtype == supported for supported in self.dtypes)


@runtime_checkable
class Executable(Protocol):
    """Protocol implemented by backend-specific executables.

    A backend ``load()`` returns an object satisfying this protocol. The
    user-facing ``etl.pipeline.Executable`` wraps one of these together with
    the signature TreeSpecs to provide structured I/O; ``run`` here operates
    on FLAT lists of ``core.Tensor``.

    Persistence: ``save`` must save the underlying ``CompiledArtifact`` (or an
    equivalent self-describing form) so the executable can be reconstructed
    EXPLICITLY at ``load`` — device handles are never serialized.
    """

    functions: tuple[str, ...]
    device: Device | None

    def run(self, flat_input_tensors: list[Tensor]) -> list[Tensor]:
        """Execute the program on flat input tensors, returning flat outputs.

        Raises ``core.BackendError`` for unsupported ops/features encountered
        at run time (v1 numpy backend supports everything it compiles, so this
        is the safety net for capability drift).
        """
        ...

    def save(self, path: str | os.PathLike) -> None:
        """Serialize the executable's underlying artifact (self-describing).

        Reconstruction happens explicitly at ``load``; never serializes device
        handles or live interpreter state.
        """
        ...

    @classmethod
    def load(cls, path: str | os.PathLike, device: Device | None = None) -> "Executable":
        """Reconstruct an executable from a saved artifact.

        Mismatched backend/device/ABI or missing required custom ops raise
        ``core.PersistenceError`` — never a silent re-compile.
        """
        ...


class Backend(ABC):
    """Abstract staging backend: graph → lowered program → compiled artifact → executable.

    Subclasses declare a unique ``name`` and a ``Capabilities``. The staging
    methods are explicit and never compose: ``lower`` must not compile,
    ``compile`` must not load, ``load`` must not re-lower or re-compile
    (see the staged-sugar rules in ../../CONTEXT.md).
    """

    name: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities()

    @abstractmethod
    def lower(self, graph: "Graph", options: dict | None = None) -> "LoweredProgram":
        """Verify a Graph and produce this backend's lowered form.

        Contract (binding):
        - MUST call ``graph.verify()`` (surfacing ``core.VerificationError``).
        - MUST record the input/output ``Signature`` from the Graph (TreeSpecs,
          per-leaf specs, static values) — passed down, never re-derived.
        - The payload must be serializable/self-describing so ``save``/``load``
          round-trips without re-lowering.
        - Capability violations raise ``core.BackendError`` naming the feature.
        """
        ...

    @abstractmethod
    def compile(self, lowered: "LoweredProgram", options: dict | None = None) -> "CompiledArtifact":
        """Turn a LoweredProgram into a self-describing CompiledArtifact.

        MUST validate ``lowered.backend == self.name`` (``core.BackendError``
        otherwise) and record required custom ops / runtime dependencies.
        """
        ...

    @abstractmethod
    def load(self, artifact: "CompiledArtifact", device: Device | None = None) -> "Executable":
        """Reconstruct a backend executable from an artifact.

        NEVER silently re-traces, re-lowers, or re-compiles; backend/device
        mismatches raise ``core.PersistenceError`` (or ``core.BackendError``
        for unsupported device kinds).
        """
        ...
