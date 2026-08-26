"""Reference numpy CPU interpreter backend (the default backend).

This package implements the staging flow (``lower`` -> ``compile`` -> ``load``)
and the interpreter execution model defined in the parent contract
(``../CONTEXT.md``, "Numpy backend design" — BINDING). In short:

- ``NumpyBackend``: stages a verified ``Graph`` into a ``LoweredProgram``
  (payload = versioned self-describing ``ir.serialize_module(graph.module)``)
  and a ``CompiledArtifact`` (target ``"cpu"``; the artifact IS serialized IR —
  there is no machine code).
- ``NumpyExecutable``: the backend executable (satisfies the ``Executable``
  protocol). Execution order = block op order (the effect ordering); shape
  inference REUSES ops-level inference rules with symbolic dims bound to
  concrete values; control flow = recursive region execution.
- ``numpy_backend``: the registered default backend instance.

Import acyclicity (binding, see ``../CONTEXT.md``): top-level imports
restricted to ``etl.core`` / ``etl.ir``; ``etl.ops`` may be imported ONLY
inside function bodies (``_register_block_impls`` — the sole allowed site);
never import ``etl.pipeline`` or ``etl.persist`` at top level.

Architecture phase: behavioral bodies raise ``NotImplementedError``; trivial
pure-type code (capability declaration, executable storage constructor,
collective hook) is live.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from etl.core import Device

from ..backend import Backend, Capabilities
from ..program import CompiledArtifact, LoweredProgram, Signature
from ..registry import register
from .collectives import (
    CollectiveExecutor,
    SingleRankCollectiveExecutor,
    get_collective_executor,
    set_collective_executor,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.core import Tensor
    from etl.ir import Module
    from etl.trace import Graph

__all__ = [
    "NumpyBackend",
    "NumpyExecutable",
    "numpy_backend",
    # collective hook (re-exported for tests / in-process multi-rank simulation)
    "CollectiveExecutor",
    "SingleRankCollectiveExecutor",
    "set_collective_executor",
    "get_collective_executor",
]


def _all_numpy_dtypes() -> frozenset:
    """All numpy dtype objects: ``numpy.sctypes`` flattening plus ``bool_``.

    Per the binding parent contract the numpy backend declares support for ALL
    numpy dtypes (``Capabilities.dtypes``). Per-op kernels validate concrete
    dtype support at run time (e.g. arithmetic on object dtypes is rejected by
    the kernel — never silently coerced).
    """
    dtypes = {np.dtype(np.bool_)}
    for dtype_list in np.sctypes.values():
        dtypes.update(np.dtype(dtype) for dtype in dtype_list)
    return frozenset(dtypes)


class NumpyBackend(Backend):
    """Default reference CPU backend: a pure-Python numpy interpreter."""

    name: ClassVar[str] = "numpy"
    capabilities: ClassVar[Capabilities] = Capabilities(
        dynamic_shapes=True,
        dtypes=_all_numpy_dtypes(),
        collectives=True,  # single-process simulation via the CollectiveExecutor hook
        runtime_calls=True,  # Python callbacks execute synchronously at the op position
        custom_blocks=True,  # registered numpy block impls
        async_collectives=False,  # simulation is synchronous
    )

    def lower(self, graph: "Graph", options: dict | None = None) -> LoweredProgram:
        """Stage a verified Graph into a LoweredProgram (serialized-IR payload).

        Binding design (implement in the implementation phase):
        1. ``graph.verify()`` — surfaces ``core.VerificationError`` as-is.
        2. Capability pre-check against ``self.capabilities`` (v1 numpy
           supports everything; the check pattern stays for capability drift).
        3. Inline ``block_call`` portable decompositions as a graph->graph
           expansion at LOWER time. A block with neither a portable
           decomposition nor a registered numpy impl -> ``core.BackendError``
           naming the block — never a silent skip.
        4. Record the ``Signature`` from the Graph (input/output TreeSpec +
           per-leaf specs + static values) — passed down, never re-derived.
        5. ``payload`` = versioned self-describing
           ``ir.serialize_module(graph.module)``.

        Raises ``core.VerificationError`` / ``core.BackendError`` — no silent
        fallbacks or partial semantics.
        """
        raise NotImplementedError(
            "architecture stub: implement the numpy staging flow in the implementation phase"
        )

    def compile(self, lowered: LoweredProgram, options: dict | None = None) -> CompiledArtifact:
        """Wrap the serialized IR into a self-describing CompiledArtifact.

        Binding design (implement in the implementation phase):
        1. Validate ``lowered.backend == "numpy"`` — mismatch =>
           ``core.BackendError`` (never cross-backend compilation).
        2. Scan the module for ``block_call`` ops to record
           ``required_custom_ops``; record ``runtime_dependencies``
           (self-describing per the serialization contract).
        3. ``target = "cpu"``; ``payload`` = the serialized ``ir.Module``.
           There is NO machine code — the artifact IS serialized IR.

        Staging methods never compose: no lowering work happens here.
        """
        raise NotImplementedError(
            "architecture stub: implement artifact assembly in the implementation phase"
        )

    def load(self, artifact: CompiledArtifact, device: Device | None = None) -> "NumpyExecutable":
        """Reconstruct a NumpyExecutable from an artifact. Never re-compiles.

        Binding design (implement in the implementation phase):
        1. Validate ``artifact.backend == "numpy"`` — mismatch =>
           ``core.PersistenceError``.
        2. Validate device: ``None`` or a CPU device — else
           ``core.BackendError`` (unsupported kind) / ``core.DeviceError``.
        3. Validate required custom ops availability — missing =>
           ``core.PersistenceError``.
        4. ``ir.deserialize_module(artifact.payload)`` and build a
           ``NumpyExecutable``.

        NEVER re-traces / re-lowers / re-compiles — load-time mismatches fail
        clearly (see the root error strategy).
        """
        raise NotImplementedError(
            "architecture stub: implement executable reconstruction in the implementation phase"
        )


def _module_function_names(module: Any) -> tuple[str, ...]:
    """Function names exposed by an ``ir.Module``.

    The exact ``ir.Module`` accessor API is finalized with the ``etl/ir``
    owners during implementation; this helper is the single touch point.
    """
    functions = getattr(module, "functions", ())
    return tuple(getattr(fn, "name", "") for fn in functions)


class NumpyExecutable:
    """Backend executable for the numpy interpreter (satisfies ``Executable``).

    Attributes:
        functions: tuple of module function names.
        device: runtime device (v1: CPU only).
        signature: input/output contract (TreeSpecs + per-leaf specs + static
            values) — used to validate inputs at ``run`` time.
        artifact: the underlying CompiledArtifact — what ``save`` persists.
    """

    def __init__(
        self,
        module: "Module",
        signature: Signature | None = None,
        artifact: CompiledArtifact | None = None,
        device: Device | None = None,
    ) -> None:
        self._module = module
        self.signature = signature
        self.artifact = artifact
        self.device = device
        self.functions = _module_function_names(module)

    def run(self, flat_input_tensors: list["Tensor"]) -> list["Tensor"]:
        """Execute the program on flat input tensors, returning flat outputs.

        Binding design (implement in the implementation phase):
        1. Bind inputs to the entry function's block args — validate count,
           dtype, and shape against the signature specs (free symbolic dims at
           run time => ``core.ShapeError``).
        2. Interpret ops in BLOCK OP ORDER — execution order = block op order
           = the effect ordering (write/read/collective/callback ops anchor
           order; pure ops keep program order for determinism).
        3. Dispatch every op through the kernels table
           (``kernels.dispatch(op_name)``); unknown op => ``core.BackendError``
           naming the op.

        Control flow (``cond``/``while_loop``/``scan``) is interpreted by
        recursively running region blocks — genuinely dynamic runtime control
        flow, never specialized per iteration. ``runtime_call`` executes the
        Python callback synchronously at the op's position (documented sync
        point — no async execution in v1).
        """
        raise NotImplementedError(
            "architecture stub: implement the interpreter loop in the implementation phase"
        )

    def save(self, path: str | os.PathLike) -> None:
        """Save the underlying CompiledArtifact.

        Binding design: the executable is reconstructed EXPLICITLY at ``load``
        — device handles and live interpreter state are never serialized.
        Delegates to ``CompiledArtifact.save`` (etl.persist container, lazy
        import).
        """
        raise NotImplementedError(
            "architecture stub: implement artifact-based save in the implementation phase"
        )

    @classmethod
    def load(cls, path: str | os.PathLike, device: Device | None = None) -> "NumpyExecutable":
        """Reconstruct an executable from a saved artifact.

        Delegates to ``NumpyBackend.load`` (backend/device/dependency
        validation; mismatch => ``core.PersistenceError``) — never a silent
        re-compile.
        """
        raise NotImplementedError(
            "architecture stub: implement artifact-based load in the implementation phase"
        )


def _register_block_impls() -> None:
    """Wire block->numpy-impl dispatch at import time.

    Block calls reach the interpreter through exactly one of two paths: a
    portable decomposition (inlined at ``lower()`` time — the interpreter
    never sees it) or a registered numpy impl (dispatched by
    ``kernels/custom.py``). This function installs the numpy impls into the
    ops-level block-impl registry so ``lower()`` can resolve them.

    Import acyclicity: ``etl.ops`` is imported LAZILY inside this body — the
    ONLY allowed location for that import in this package. The exact
    registration hook is finalized with the ``etl/ops`` and ``etl/block``
    owners during implementation (noted in ``../CONTEXT.md``).
    """
    import etl.ops  # noqa: F401  — lazy import; ops installs the block-impl hook

    raise NotImplementedError(
        "architecture stub: finalize the ops-level block-impl registration hook "
        "with the etl/ops and etl/block owners during implementation, then "
        "register numpy block impls here"
    )


numpy_backend = NumpyBackend()
register(numpy_backend)

# _register_block_impls()  # called at import time once the ops-level hook is
# finalized (implementation phase); currently stubbed to NotImplementedError.
