"""Staged pipeline objects OWNED by backends: Signature, LoweredProgram, CompiledArtifact.

Ownership note (binding, see ../CONTEXT.md): these types belong to ``backends``,
NOT to ``pipeline``. ``pipeline`` orchestrates staging and wraps a backend
executable + signature for structured I/O; it re-exports these types through ``etl``.

Import acyclicity:
- Top-level imports restricted to ``etl.core``.
- ``etl.persist`` (container format) is imported lazily INSIDE ``save``/``load``
  bodies: persist sits ABOVE backends in the import DAG, so a top-level import
  would create a cycle.
- ``registry`` is imported lazily inside ``load`` bodies for backend-name
  validation / payload reconstruction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from etl.core import PersistenceError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .backend import Backend  # noqa: F401

__all__ = ["Signature", "LoweredProgram", "CompiledArtifact"]

#: Persisted payload type tags for the etl.persist container.
PAYLOAD_LOWERED = "etl.lowered_program"
PAYLOAD_ARTIFACT = "etl.compiled_artifact"


@dataclass(frozen=True)
class Signature:
    """Input/output contract recorded at ``lower()`` time, passed down from the Graph.

    Attributes:
        input_tree: ``core.TreeSpec`` of the structured input (flatten/unflatten).
        output_tree: ``core.TreeSpec`` of the structured output.
        input_specs: per-leaf ``core.TensorSpec`` in flattened input order.
        output_specs: per-leaf ``core.TensorSpec`` in flattened output order.
        static_values: Python values the trace specialized on; validated at run time
            (changing them means a different graph — never a hidden recompile).
    """

    input_tree: Any = None
    output_tree: Any = None
    input_specs: tuple[Any, ...] = ()
    output_specs: tuple[Any, ...] = ()
    static_values: tuple[Any, ...] = ()


@dataclass
class LoweredProgram:
    """Backend-specific lowered form of a Graph (pre-compile staging object).

    Attributes:
        backend: backend name (str) that produced this program.
        signature: input/output contract (see ``Signature``).
        payload: backend-specific, serializable, self-describing lowered form.
            numpy backend: versioned ``ir.Module`` serialization
            (``ir.serialize_module``). A hypothetical stablehlo-backed lowering
            would carry MLIR text.

    ``text()`` renders a human-readable form (numpy: serialized-IR text;
    MLIR-style payloads: the MLIR text itself).
    ``save()``/``load()`` use the ``etl.persist`` container (lazy import);
    ``load()`` validates the recorded backend name against the registry and
    delegates payload reconstruction to that backend — mismatch raises
    ``core.PersistenceError``, never a silent re-lowering.
    """

    backend: str
    signature: Signature | None = None
    payload: Any = None

    def text(self) -> str:
        """Human-readable rendering of the payload (backend-specific)."""
        raise NotImplementedError(
            "architecture stub: implement backend-specific text() rendering in the implementation phase"
        )

    def save(self, path: str | os.PathLike) -> None:
        """Serialize via the etl.persist container (lazy import).

        Metadata records: payload type tag, backend name, signature
        (TreeSpecs + per-leaf specs + static values), IR format version.
        """
        raise NotImplementedError(
            "architecture stub: implement persist-container save in the implementation phase"
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "LoweredProgram":
        """Deserialize and validate.

        Reads the container, validates the recorded backend name against the
        registry (unknown/mismatched backend => ``core.PersistenceError``), and
        delegates payload reconstruction to that backend. Never re-lowers.
        """
        raise NotImplementedError(
            "architecture stub: implement persist-container load + backend validation in the implementation phase"
        )


@dataclass
class CompiledArtifact:
    """Compiled, self-describing artifact (post-``compile`` staging object).

    Attributes:
        backend: backend name (str) that produced this artifact.
        signature: input/output contract (see ``Signature``).
        target: compilation target descriptor (numpy: ``"cpu"``).
        payload: backend-specific compiled form (numpy: the serialized
            ``ir.Module`` — the artifact IS serialized IR, there is no machine code).
        required_custom_ops: tuple of custom-op names the program depends on;
            load validates their availability (missing => ``core.PersistenceError``).
        runtime_dependencies: dict of runtime dependency name -> version required
            to execute the artifact (self-describing per serialization contract).

    ``save()``/``load()`` use the ``etl.persist`` container (lazy import);
    ``load()`` validates backend and dependencies — never silently recompiles.
    """

    backend: str
    signature: Signature | None = None
    target: str = ""
    payload: Any = None
    required_custom_ops: tuple[str, ...] = ()
    runtime_dependencies: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | os.PathLike) -> None:
        """Serialize via the etl.persist container (lazy import).

        Metadata records: payload type tag, backend name/version, target,
        signature, required custom ops, runtime dependencies, IR format version.
        """
        raise NotImplementedError(
            "architecture stub: implement persist-container save in the implementation phase"
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CompiledArtifact":
        """Deserialize and validate.

        Validates the recorded backend against the registry and required custom
        ops/runtime dependencies against the environment; mismatch raises
        ``core.PersistenceError`` — never silently recompiles.
        """
        raise NotImplementedError(
            "architecture stub: implement persist-container load + dependency validation in the implementation phase"
        )
