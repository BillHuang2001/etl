"""etl.pipeline — explicit execution-pipeline orchestration.

The canonical staging workflow and its transparent shorthands::

    graph = etl.trace(f, *specs)               # trace (etl/trace)
    lowered = etl.lower(graph)                 # here
    artifact = etl.compile(lowered)            # here
    executable = etl.load(artifact, device)    # here
    y = etl.run(executable, *tensors)          # here

    exe = etl.build(f, *specs)                 # ≡ trace→lower→compile→load
    y = etl.evaluate(f, *tensors)              # ≡ specs→build→run

Stage types are owned by their producing modules: `Graph` lives in
`etl.trace`; `LoweredProgram` / `CompiledArtifact` live in `etl.backends`
(they are backend products). This module owns the orchestration functions,
the user-facing `Executable` wrapper (backend executable + input/output
TreeSpec + signature so `run`/`bind` speak structured values), and
`BoundExecutable` from `etl.bind`.

Binding rules (spec §6.3): bind is pure argument-passing sugar —
conceptually `lambda x: run(executable, x, w)`. It never alters the graph,
never embeds constants, never re-specializes, never recompiles. It validates
that each binding names an existing input, and dtype/shape/device
compatibility, and that no required input is accidentally omitted.

Staging rules (spec §10): no function here silently consumes an earlier-stage
object and performs missing steps — each stage maps its documented input type
to its documented output type, raising `TypeError`/`PersistenceError`/
`BackendError` otherwise.
"""
from __future__ import annotations

__all__ = ["Executable", "BoundExecutable", "lower", "compile", "load", "run",
           "bind", "build", "evaluate"]


class Executable:
    """User-facing executable: a backend executable + structured signature.

    Wraps a backend-level executable (whose ``run`` speaks flat tensor lists)
    with the Graph's input/output TreeSpecs and static values, so ``etl.run``
    and ``etl.bind`` accept and return ordinary nested Python structures.

    Attributes:
        backend_executable: the backend executable (satisfies the
            ``etl.backends.Executable`` protocol: ``run(flat_inputs) ->
            flat_outputs``, ``.functions``, ``.device``).
        signature: ``etl.backends.Signature`` (input/output TreeSpec +
            per-leaf specs + static values).
    """

    def __init__(self, backend_executable, signature):
        self.backend_executable = backend_executable
        self.signature = signature

    @property
    def functions(self):
        """Function names exported by the loaded program."""
        raise NotImplementedError

    @property
    def device(self):
        """Device the executable is bound to."""
        raise NotImplementedError

    def save(self, path):
        """Persist the executable if the backend supports it.

        Backends that cannot serialize device handles must save the
        underlying CompiledArtifact and reconstruct on load — never pretend
        a device handle was serialized.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path, backend=None, device=None):
        """Load a persisted executable; never silently recompiles."""
        raise NotImplementedError


class BoundExecutable:
    """Result of ``etl.bind``: an executable with pre-supplied inputs.

    Also satisfies the runnable surface of ``Executable`` (so ``etl.run``
    accepts it), supplying the bound tensors before user-provided arguments.
    """

    def __init__(self, executable, bindings):
        raise NotImplementedError

    @property
    def functions(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError


def lower(graph, backend=None, **options):
    """``lower(graph) -> LoweredProgram``.

    ``backend`` defaults to ``etl.backends.numpy_backend``. The backend
    verifies the graph, records its signature (input/output TreeSpec, specs,
    static values) and produces a backend-specific lowered program.
    """
    raise NotImplementedError


def compile(lowered, backend=None, **options):
    """``compile(lowered) -> CompiledArtifact``.

    ``backend`` may be omitted (taken from the lowered program) but must
    match if given (``BackendError`` otherwise). Does NOT silently re-lower.
    """
    raise NotImplementedError


def load(artifact, backend=None, device=None):
    """``load(artifact) -> Executable``.

    ``backend`` may be omitted (taken from the artifact) but must match if
    given. ``device`` defaults to the backend's default device. Returns the
    user-facing wrapper carrying the structured signature.
    """
    raise NotImplementedError


def run(executable, *args):
    """``run(executable, *tensors) -> structured outputs``.

    Flattens inputs via the signature TreeSpec, validates dtype/shape/device
    against the recorded specs and static values (DTypeError/ShapeError/
    DeviceError), calls the backend executable with flat tensors, and
    reconstructs the structured outputs (including recorded static output
    leaves).
    """
    raise NotImplementedError


def bind(executable, **bindings):
    """``bind(executable, w=w) -> BoundExecutable`` — argument-supply sugar.

    Validates: every binding name is an existing named input; bound tensors
    are dtype/shape/device compatible with their specs; required (unbound,
    no-default) inputs remain. Returns a wrapper that supplies the bound
    values when invoked. Never alters the graph or recompiles.
    """
    raise NotImplementedError


def build(fn, *specs, backend=None, device=None, **options):
    """``build(f, *specs) -> Executable``.

    Documented shorthand for ``load(compile(lower(trace(fn, *specs))), ...)``
    — no other behavior (docstring must stay in sync with the expansion).
    """
    raise NotImplementedError


def evaluate(fn, *args, backend=None, device=None, **options):
    """``evaluate(f, *tensors) -> structured outputs``.

    Documented shorthand: derive a TensorSpec per concrete-tensor argument
    (snapshotting shape + dtype only), then build and run — no other
    behavior. Arguments that are not concrete tensors raise TypeError.
    """
    raise NotImplementedError
