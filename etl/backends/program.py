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

from etl.core import BackendError, PersistenceError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .backend import Backend  # noqa: F401

__all__ = ["Signature", "LoweredProgram", "CompiledArtifact"]

#: Persisted payload type tags for the etl.persist container.
PAYLOAD_LOWERED = "etl.lowered_program"
PAYLOAD_ARTIFACT = "etl.compiled_artifact"

#: Signature encoding keys, matching ``trace.Graph.signature_info()`` exactly
#: (and the ``Signature`` field names) so formats align across artifacts.
_SIGNATURE_KEYS = (
    "input_tree",
    "output_tree",
    "input_specs",
    "output_specs",
    "static_values",
    "output_static_values",
)


def _encode_signature(signature: "Signature | None"):
    """Encode ``signature`` as a JSON-safe dict for the persist container.

    Mirrors ``trace.Graph.signature_info()`` exactly: the keys are the
    ``Signature`` field names, each value ``persist.encode_value``'d (TreeSpec,
    TensorSpec and static values all have codec entries), so the dict is
    JSON-safe as passed to ``persist.save_object``. ``None`` in, ``None`` out.
    """
    if signature is None:
        return None
    from etl import persist

    return {
        "input_tree": persist.encode_value(signature.input_tree),
        "output_tree": persist.encode_value(signature.output_tree),
        "input_specs": persist.encode_value(signature.input_specs),
        "output_specs": persist.encode_value(signature.output_specs),
        "static_values": persist.encode_value(signature.static_values),
        "output_static_values": persist.encode_value(signature.output_static_values),
    }


def _decode_signature(signature_info) -> "Signature | None":
    """Rebuild a ``Signature`` from container ``signature_info``.

    Decodes each of the encoded keys via ``persist.decode_value``; a
    missing/empty ``signature_info`` yields ``None``. A present-but-malformed
    dict raises ``PersistenceError`` naming the missing field — artifacts are
    never loaded with a silently partial signature.
    """
    if not signature_info:
        return None
    from etl import persist

    fields = {}
    for key in _SIGNATURE_KEYS:
        if key not in signature_info:
            raise PersistenceError(
                f"corrupt: signature_info is missing the {key!r} field"
            )
        fields[key] = persist.decode_value(signature_info[key])
    return Signature(**fields)


def _recorded_backend_name(backend_info) -> str:
    """Extract and validate the recorded backend name from the container header.

    Raises ``PersistenceError`` for a malformed header — never a silent
    reinterpretation of an artifact written by some other tool.
    """
    if not isinstance(backend_info, dict):
        raise PersistenceError(
            "corrupt: backend_info must be a dict, got "
            f"{type(backend_info).__name__}"
        )
    name = backend_info.get("name")
    if not isinstance(name, str) or not name:
        raise PersistenceError(
            "corrupt: backend_info has no 'name' string field"
        )
    return name


def _require_registered_backend(name: str) -> None:
    """Validate ``name`` against the backend registry (lazy import).

    An unknown backend raises ``PersistenceError`` — the artifact is NOT
    silently re-lowered/re-compiled by another backend.
    """
    from .registry import get

    try:
        get(name)
    except BackendError as exc:
        raise PersistenceError(
            f"artifact records unknown backend {name!r} — never silently "
            f"re-lowers/re-compiles"
        ) from exc


@dataclass(frozen=True)
class Signature:
    """Input/output contract recorded at ``lower()`` time, passed down from the Graph.

    Attributes:
        input_tree: ``core.TreeSpec`` of the structured input (flatten/unflatten).
        output_tree: ``core.TreeSpec`` of the structured output.
        input_specs: per-leaf ``core.TensorSpec`` in flattened input order.
        output_specs: per-leaf ``core.TensorSpec`` in flattened output order.
        static_values: Python values the trace specialized on (input static
            leaves, in pre-order leaf order); validated at run time (changing
            them means a different graph — never a hidden recompile).
        output_static_values: Python values recorded for static OUTPUT leaves
            (in pre-order leaf order), re-inserted by ``etl.run`` when
            reconstructing the structured outputs.
    """

    input_tree: Any = None
    output_tree: Any = None
    input_specs: tuple[Any, ...] = ()
    output_specs: tuple[Any, ...] = ()
    static_values: tuple[Any, ...] = ()
    output_static_values: tuple[Any, ...] = ()


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
    ``save()``/``load()`` use the ``etl.persist`` container (lazy import).
    Payloads are self-contained (serialized ``ir.Module`` dict or MLIR text),
    so ``load()`` reconstructs the payload directly from the container — no
    backend delegation. It validates the recorded backend name against the
    registry; mismatch raises ``core.PersistenceError``, never a silent
    re-lowering.
    """

    backend: str
    signature: Signature | None = None
    payload: Any = None

    def text(self) -> str:
        """Human-readable rendering of the payload (backend-specific).

        * MLIR-style payloads (``str``) are returned verbatim.
        * Compiler-backend payloads (dict with ``"format": "stablehlo"``
          and an ``"mlir_text"`` field — see ``compiler.CompilerBackend``)
          render their MLIR text.
        * Serialized ``ir.Module`` payloads (dict carrying ``version`` +
          ``module`` keys) are deserialized and pretty-printed.
        * Anything else raises ``core.BackendError`` naming the backend.
        """
        payload = self.payload
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict) and "mlir_text" in payload:
            return payload["mlir_text"]
        if isinstance(payload, dict) and "version" in payload and "module" in payload:
            from etl import ir

            module = ir.deserialize_module(payload)
            return ir.pretty_print(module)
        raise BackendError(
            f"backend {self.backend!r}: cannot render lowered payload of type "
            f"{type(payload).__name__} — expected MLIR text (str) or a "
            f"serialized ir.Module dict"
        )

    def save(self, path: str | os.PathLike) -> None:
        """Serialize via the etl.persist container (lazy import).

        Metadata records: payload type tag, backend name, signature
        (TreeSpecs + per-leaf specs + static values), IR format version.
        """
        from etl import persist

        persist.save_object(
            path,
            PAYLOAD_LOWERED,
            payload_fields={"payload": self.payload},
            backend_info={"name": self.backend},
            signature_info=_encode_signature(self.signature),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "LoweredProgram":
        """Deserialize and validate.

        Reads the container, validates the recorded backend name against the
        registry (unknown/mismatched backend => ``core.PersistenceError``),
        and rebuilds the ``Signature`` and self-contained payload directly.
        Never re-lowers.
        """
        from etl import persist

        loaded = persist.load_object(path, expected_payload_type=PAYLOAD_LOWERED)
        backend = _recorded_backend_name(loaded.backend_info)
        _require_registered_backend(backend)
        signature = _decode_signature(loaded.signature_info)
        try:
            payload = loaded.payload["payload"]
        except KeyError:
            raise PersistenceError(
                "corrupt: lowered-program payload is missing the 'payload' field"
            ) from None
        return cls(backend=backend, signature=signature, payload=payload)


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

        Metadata records: payload type tag, backend name, target, signature,
        required custom ops, runtime dependencies, IR format version.
        """
        from etl import persist

        persist.save_object(
            path,
            PAYLOAD_ARTIFACT,
            payload_fields={
                "payload": self.payload,
                "required_custom_ops": tuple(self.required_custom_ops),
                "runtime_dependencies": dict(self.runtime_dependencies),
            },
            backend_info={"name": self.backend, "target": self.target},
            signature_info=_encode_signature(self.signature),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CompiledArtifact":
        """Deserialize and validate.

        Validates the recorded backend against the registry and required
        custom ops/runtime dependencies against the environment; mismatch
        raises ``core.PersistenceError`` — never silently recompiles.
        """
        from etl import persist

        loaded = persist.load_object(path, expected_payload_type=PAYLOAD_ARTIFACT)
        backend = _recorded_backend_name(loaded.backend_info)
        _require_registered_backend(backend)
        signature = _decode_signature(loaded.signature_info)
        for field in ("payload", "required_custom_ops", "runtime_dependencies"):
            if field not in loaded.payload:
                raise PersistenceError(
                    f"corrupt: compiled-artifact payload is missing the "
                    f"{field!r} field"
                )
        payload = loaded.payload["payload"]
        target = loaded.backend_info.get("target", "")
        required_custom_ops = tuple(loaded.payload["required_custom_ops"])
        runtime_dependencies = loaded.payload["runtime_dependencies"]
        if not isinstance(runtime_dependencies, dict):
            raise PersistenceError(
                "corrupt: runtime_dependencies must be a dict, got "
                f"{type(runtime_dependencies).__name__}"
            )
        if not all(isinstance(name, str) for name in required_custom_ops):
            raise PersistenceError(
                "corrupt: required_custom_ops must be a list/tuple of strings"
            )
        _validate_runtime_dependencies(runtime_dependencies)
        _validate_required_custom_ops(required_custom_ops, backend)
        return cls(
            backend=backend,
            signature=signature,
            target=target,
            payload=payload,
            required_custom_ops=required_custom_ops,
            runtime_dependencies=dict(runtime_dependencies),
        )


def _validate_runtime_dependencies(runtime_dependencies: dict) -> None:
    """Check recorded runtime dependency versions against the environment.

    A mismatch raises ``core.PersistenceError`` — the artifact is never
    silently recompiled against different dependencies.
    """
    recorded_numpy = runtime_dependencies.get("numpy")
    if recorded_numpy is not None:
        import numpy as np

        if np.__version__ != recorded_numpy:
            raise PersistenceError(
                f"artifact requires numpy {recorded_numpy}, environment has "
                f"{np.__version__} — never silently recompile"
            )


def _validate_required_custom_ops(required_custom_ops: tuple[str, ...], backend: str) -> None:
    """Verify every required custom op resolves in this environment.

    An op must have either an implementation registered for the artifact's
    backend or a portable decomposition; anything else raises
    ``core.PersistenceError`` naming the op. If ``etl.block`` is
    unimportable (minimal deployments / parallel implementation), the check
    degrades gracefully by skipping — this is where the validation lives.
    """
    if not required_custom_ops:
        return
    try:
        from etl.block import registry as block_registry
    except ImportError:
        # etl.block unavailable: validation skipped (graceful degradation).
        return
    for op_name in required_custom_ops:
        has_impl = block_registry.get_impl(op_name, backend) is not None
        has_portable = block_registry.get_portable(op_name) is not None
        if not has_impl and not has_portable:
            raise PersistenceError(
                f"artifact requires custom op {op_name!r}, which has neither a "
                f"registered {backend!r} implementation nor a portable "
                f"decomposition — never silently recompile"
            )
