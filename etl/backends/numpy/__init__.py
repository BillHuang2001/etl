"""Reference numpy CPU interpreter backend (the default backend).

This package implements the staging flow (``lower`` -> ``compile`` -> ``load``)
and the interpreter execution model defined in the parent contract
(``../CONTEXT.md``, "Numpy backend design" — BINDING). In short:

- ``NumpyBackend``: stages a verified ``Graph`` into a ``LoweredProgram``
  (payload = versioned self-describing ``ir.serialize_module(graph.module)``)
  and a ``CompiledArtifact`` (target ``"cpu"``; the artifact IS serialized IR —
  there is no machine code). ``block_call`` portable decompositions are
  inlined as graph->graph expansion at LOWER time (``inline.py``); a block
  with neither a portable decomposition nor a registered numpy impl =>
  ``core.BackendError`` naming the block.
- ``NumpyExecutable``: the backend executable (satisfies the ``Executable``
  protocol), wrapping an ``Interpreter`` (``interpreter.py``).
- Execution model: execution order = block op order (the effect ordering);
  every op dispatches through the ``kernels`` table (``KernelContext``
  carries the per-execution state: symbolic-dim bindings walked from the
  spec shapes vs concrete shapes, the rank context, callback resolution);
  shape inference REUSES the ops-level inference rules (IR result types are
  evaluated against the dim bindings — no second copy of shape rules);
  control flow = recursive region execution; collectives dispatch through
  the ``etl.dist.context`` executor hook (default single-rank identity
  executor installed at import below).
- ``numpy_backend``: the registered default backend instance.

Import acyclicity (binding, see ``../CONTEXT.md``): top-level imports
restricted to ``etl.core`` / ``etl.ir`` (plus ``etl.dist.context`` for the
executor-hook installation — dist never imports backends); ``etl.ops`` /
``etl.trace`` / ``etl.block`` / ``etl.persist`` are imported ONLY inside
function bodies; never import ``etl.pipeline`` at top level.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from etl import core
from etl import ir
from etl.core import Device
from etl.dist import context as dist_context

from ..backend import Backend, Capabilities
from ..program import CompiledArtifact, LoweredProgram, Signature
from ..registry import register
from . import kernels
from .collectives import CollectiveExecutor, SingleRankCollectiveExecutor
from .interpreter import Interpreter, entry_function
from .inline import (
    clone_ops_into,  # noqa: F401  (shared machinery re-export, ../inline.py)
    drop_op_uses,  # noqa: F401  (shared machinery re-export, ../inline.py)
    inline_portables,
    iter_block_ops,  # noqa: F401  (shared machinery re-export, ../inline.py)
    iter_ops,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from etl.core import Tensor
    from etl.ir import Module
    from etl.trace import Graph

__all__ = [
    "NumpyBackend",
    "NumpyExecutable",
    "numpy_backend",
    # canonical collective protocol + the default single-rank executor
    "CollectiveExecutor",
    "SingleRankCollectiveExecutor",
]


def _all_numpy_dtypes() -> frozenset:
    """All numpy dtype objects: numeric sctypes + ``bool_``.

    Per the binding parent contract the numpy backend declares support for ALL
    numpy dtypes (``Capabilities.dtypes``). Per-op kernels validate concrete
    dtype support at run time (e.g. arithmetic on object dtypes is rejected by
    the kernel — never silently coerced).

    Note: ``numpy.sctypes`` was removed in NumPy 2.0 — the numeric type
    groups are enumerated explicitly (mirrors sctypes: int/uint/float/
    complex).
    """
    dtypes = {np.dtype(np.bool_)}
    for group in (
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128,
    ):
        dtypes.add(np.dtype(group))
    return frozenset(dtypes)


class NumpyBackend(Backend):
    """Default reference CPU backend: a pure-Python numpy interpreter.

    ``lower()`` reuses the shared block-inlining machinery
    (``../inline.py::inline_portables``, ``keep_backend_impls="numpy"``),
    shared with compiler backends (``CompilerBackend``).
    """

    name: ClassVar[str] = "numpy"
    capabilities: ClassVar[Capabilities] = Capabilities(
        dynamic_shapes=True,
        dtypes=_all_numpy_dtypes(),
        collectives=True,  # single-process simulation via the CollectiveExecutor hook
        runtime_calls=True,  # Python callbacks execute synchronously at the op position
        custom_blocks=True,  # registered numpy block impls
        async_collectives=False,  # simulation is synchronous
        sparse_ops=True,  # the ONLY backend with sparse-tensor support in v1
    )

    def lower(self, graph: "Graph", options: dict | None = None) -> LoweredProgram:
        """Stage a verified Graph into a LoweredProgram (serialized-IR payload).

        1. ``graph.verify()`` — surfaces ``core.VerificationError`` as-is.
        2. ``kernels.register_all()`` (idempotent), then a capability
           pre-check on the CURRENT module state (live walk — the module is
           mutated by inlining in step 3): capability FLAGS are checked here
           (runtime_call/block_call/collective effects vs ``Capabilities``);
           kernel-table membership is re-checked AFTER inlining (the full
           drift net — portables replace ``block_call``s first).
        3. Inline ``block_call`` portable decompositions as a graph->graph
           expansion at LOWER time via the SHARED fixpoint helper
           ``inline_portables`` (``../inline.py``, ``keep_backend_impls=
           "numpy"`` — portables may themselves emit block calls). A block
           with neither a portable decomposition nor a registered numpy
           impl -> ``core.BackendError`` naming the block — never a silent
           skip. ``graph.verify()`` again afterwards (defensive, cheap).
        4. Record the ``Signature`` from the Graph's LIVE attributes
           (input/output TreeSpec + per-leaf specs + static values) — passed
           down, never re-derived.
        5. ``payload`` = versioned self-describing
           ``ir.serialize_module(graph.module)``.

        Raises ``core.VerificationError`` / ``core.BackendError`` — no silent
        fallbacks or partial semantics.
        """
        graph.verify()  # surfaces core.VerificationError as-is
        kernels.register_all()  # idempotent
        self._check_capabilities(graph.module, check_kernels=False)
        inline_portables(graph.module, keep_backend_impls="numpy")
        self._check_capabilities(graph.module, check_kernels=True)
        graph.verify()  # defensive post-inline verification

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
        payload = ir.serialize_module(graph.module)
        return LoweredProgram(backend="numpy", signature=signature, payload=payload)

    def compile(self, lowered: LoweredProgram, options: dict | None = None) -> CompiledArtifact:
        """Wrap the serialized IR into a self-describing CompiledArtifact.

        1. Validate ``lowered.backend == "numpy"`` — mismatch =>
           ``core.BackendError`` (never cross-backend compilation).
        2. Deserialize + scan the module for ``block_call`` ops to record
           ``required_custom_ops`` (block names, sorted, deduplicated);
           record ``runtime_dependencies`` (self-describing per the
           serialization contract).
        3. ``target = "cpu"``; ``payload`` = the serialized ``ir.Module``.
           There is NO machine code — the artifact IS serialized IR.

        Staging methods never compose: no lowering work happens here.
        """
        if lowered.backend != "numpy":
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with the numpy backend"
            )
        module = ir.deserialize_module(lowered.payload)
        block_names = {
            op.attributes["block_name"]
            for op in iter_ops(module)
            if op.name == "block_call"
        }
        required_custom_ops = tuple(sorted(block_names))
        runtime_dependencies = {"numpy": np.__version__}
        return CompiledArtifact(
            backend="numpy",
            signature=lowered.signature,
            target="cpu",
            payload=lowered.payload,
            required_custom_ops=required_custom_ops,
            runtime_dependencies=runtime_dependencies,
        )

    def load(self, artifact: CompiledArtifact, device: Device | None = None) -> "NumpyExecutable":
        """Reconstruct a NumpyExecutable from an artifact. Never re-compiles.

        1. Validate ``artifact.backend == "numpy"`` — mismatch =>
           ``core.PersistenceError``.
        2. Validate device: ``None`` or a CPU device (``kind == "cpu"``) —
           else ``core.BackendError`` naming the kind (v1 CPU only);
           non-``Device`` objects => ``core.DeviceError``.
        3. Validate required custom ops availability (declared block +
           registered numpy impl — missing => ``core.PersistenceError``
           naming the block, never a silent re-lower).
        4. ``ir.deserialize_module(artifact.payload)`` and build a
           ``NumpyExecutable``.

        NEVER re-traces / re-lowers / re-compiles — load-time mismatches fail
        clearly (see the root error strategy).
        """
        if artifact.backend != "numpy":
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                "the numpy backend cannot load it"
            )
        if device is not None:
            if not isinstance(device, Device):
                raise core.DeviceError(
                    f"device must be a core.Device, got "
                    f"{type(device).__name__}"
                )
            if device.kind != "cpu":
                raise core.BackendError(
                    f"the numpy backend v1 supports CPU devices only, got "
                    f"device kind {device.kind!r}"
                )
        from etl.block import registry as block_registry
        from etl.block.errors import BlockError

        for block_name in artifact.required_custom_ops:
            try:
                block_registry.get_block(block_name)
            except BlockError as exc:
                raise core.PersistenceError(
                    f"artifact requires custom block {block_name!r}, which "
                    "is not registered in this process"
                ) from exc
            if block_registry.get_impl(block_name, "numpy") is None:
                raise core.PersistenceError(
                    f"artifact requires custom block {block_name!r}, but no "
                    "numpy impl is registered in this process"
                )
        module = ir.deserialize_module(artifact.payload)
        return NumpyExecutable(
            module=module,
            signature=artifact.signature,
            artifact=artifact,
            device=device,
        )

    # ------------------------------------------------------------- internals

    def _check_capabilities(self, module: "Module", *, check_kernels: bool) -> None:
        """Capability drift net over the module's current (live) op set.

        - every non-terminator op must have a kernel when ``check_kernels``
          (the full net runs AFTER portable inlining — the executed module
          state);
        - ``runtime_call``/``block_call``/``collective``-effect ops require
          the matching ``Capabilities`` flag (all True for numpy v1 — the
          flags make future capability changes fail explicitly).
        """
        capabilities = self.capabilities
        for op in iter_ops(module):
            if op.is_terminator:
                continue  # 'return' is special-cased in the interpreter loop
            if check_kernels and op.name not in kernels.KERNEL_TABLE:
                raise core.BackendError(
                    f"capability drift: no numpy kernel for op '{op.name}' "
                    f"(op id {op.id})"
                )
            if op.name == "runtime_call" and not capabilities.runtime_calls:
                raise core.BackendError(
                    "capability drift: the numpy backend cannot execute "
                    "runtime_call"
                )
            if op.name == "block_call" and not capabilities.custom_blocks:
                raise core.BackendError(
                    "capability drift: the numpy backend cannot execute "
                    "block_call"
                )
            if op.effect == "collective" and not capabilities.collectives:
                raise core.BackendError(
                    f"capability drift: the numpy backend cannot execute "
                    f"collective op '{op.name}'"
                )

def _module_function_names(module: Any) -> tuple[str, ...]:
    """Function names exposed by an ``ir.Module`` (the single access point)."""
    functions = getattr(module, "functions", ())
    return tuple(getattr(fn, "name", "") for fn in functions)


class NumpyExecutable:
    """Backend executable for the numpy interpreter (satisfies ``Executable``).

    Attributes:
        backend_name: ``"numpy"`` (the pipeline uses it to identify the
            backend for run-stage option/env resolution).
        functions: tuple of module function names.
        device: runtime device (v1: CPU only).
        signature: input/output contract (TreeSpecs + per-leaf specs + static
            values) — used to validate inputs at ``run`` time.
        artifact: the underlying CompiledArtifact — what ``save`` persists.
    """

    backend_name = "numpy"

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
        self._interpreter = Interpreter(module=module, signature=signature)

    def run(
        self,
        flat_input_tensors: list["Tensor"],
        options: dict | None = None,
        *,
        rank_context: Any = None,
    ) -> list["Tensor"]:
        """Execute the program on flat input tensors, returning flat outputs.

        Execution model (``interpreter.Interpreter.run``):
        1. Bind inputs to the entry function's block args — validate count,
           dtype, and shape against the signature specs; the spec-shape vs
           concrete-shape walk builds the symbolic-dim bindings (conflicting
           bindings / free symbolic dims => ``core.ShapeError``).
        2. Interpret ops in BLOCK OP ORDER — execution order = block op order
           = the effect ordering (write/read/collective/callback ops anchor
           order; pure ops keep program order for determinism).
        3. Dispatch every op through the kernels table
           (``kernels.dispatch(op_name)``); unknown op => ``core.BackendError``
           naming the op.

        Control flow (``if``/``while``/``scan``) is interpreted by
        recursively running region blocks — genuinely dynamic runtime control
        flow, never specialized per iteration. ``runtime_call`` executes the
        Python callback synchronously at the op's position (documented sync
        point — no async execution in v1).

        ``rank_context`` optionally overrides the execution context
        (``etl.dist.RankContext``) for this run — rank/world_size graph
        scalars and multi-rank collective simulations resolve from it
        (thread-local, restored afterwards). It may be passed either as the
        keyword argument or inside the ``options`` dict (key ``"rank_context"``
        — the pipeline forwards per-run kwargs through the options dict).

        Options contract: the numpy backend is the REFERENCE interpreter and
        DELIBERATELY does not validate options — cross-backend scripts pass
        compiler-option dicts (e.g. ``target_backends``) with a numpy backend
        too, so unknown keys are ignored here (documented; the compiler
        adapters validate strictly via their ``KNOWN_OPTIONS``).
        """
        if options:
            rank_context = options.get("rank_context", rank_context)
        return self._interpreter.run(flat_input_tensors, rank_context=rank_context)

    def save(self, path: str | os.PathLike) -> None:
        """Save the underlying CompiledArtifact.

        The executable is reconstructed EXPLICITLY at ``load`` — device
        handles and live interpreter state are never serialized. Delegates
        to ``CompiledArtifact.save`` (etl.persist container, lazy import).
        """
        if self.artifact is None:
            raise core.BackendError(
                "this NumpyExecutable has no CompiledArtifact to save — "
                "construct it via NumpyBackend.load(artifact) or pass "
                "artifact= to the constructor"
            )
        self.artifact.save(path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: Device | None = None) -> "NumpyExecutable":
        """Reconstruct an executable from a saved artifact.

        Delegates to ``NumpyBackend.load`` (backend/device/dependency
        validation; mismatch => ``core.PersistenceError``) — never a silent
        re-compile.
        """
        artifact = CompiledArtifact.load(path)
        return NumpyBackend().load(artifact, device)


def _register_block_impls() -> None:
    """Block-dispatch resolution point — the numpy backend registers NO
    built-in block impls.

    Block calls reach the interpreter through exactly one of two paths, both
    resolved LAZILY at ``lower()`` / ``load()`` / run time via
    ``etl.block.registry`` (no import-time wiring needed):

    - a portable decomposition — inlined at ``lower()`` time
      (``registry.get_portable``); the interpreter never sees it;
    - a registered numpy impl — ``registry.get_impl(name, "numpy")``,
      dispatched by the ``block_call`` kernel (``kernels/custom.py``).

    User blocks register their implementations via
    ``BlockOp.portable(...)`` / ``BlockOp.impl("numpy")``. This function
    exists as the documented single touch point for future built-in
    impls and deliberately does nothing.
    """
    return None


numpy_backend = NumpyBackend()
register(numpy_backend)

# Canonical import-time installation of the default single-process collective
# executor into the dist hook (coordination note A: etl/dist/context.py owns
# the slot; the numpy backend provides the default identity executor).
dist_context.set_collective_executor(SingleRankCollectiveExecutor())
