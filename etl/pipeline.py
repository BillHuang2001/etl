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

Environment defaults (`build`/`evaluate` only — the explicit stage
functions keep their documented defaults): explicit kwargs always win;
unset `backend`/`device`/`target_backends` fall back to `ETL_BACKEND` /
`ETL_DEVICE` / `ETL_TARGET_BACKENDS`, read lazily at call time, with
iree-family `target_backends` inference from the resolved device — see
`build`'s docstring.

Staging rules (spec §10): no function here silently consumes an earlier-stage
object and performs missing steps — each stage maps its documented input type
to its documented output type, raising `TypeError`/`PersistenceError`/
`BackendError` otherwise.

Implementation notes (binding):
- The signature's trace-time trees record tensor positions with
  ``_TensorSpecLeaf`` / ``_SymbolicLeaf`` markers (plain, non-dataclass
  classes — ``core.flatten``/``core.unflatten`` treat them as ordinary
  leaves); static positions record the static value's own Python type.
  Transform-built trees additionally mark tensor positions with the bare
  dataclass types (``core.TensorSpec``/``core.SymbolicTensor``, e.g. jvp's
  tangent trees) or with a ``type=None`` plain-leaf spec (grad/vjp output
  trees). ``_is_tensor_leaf_spec`` unifies all of these.
- ``caution``: ``etl.trace`` (the function) shadows the submodule attribute
  in ``etl``; this module imports trace pieces via ``from etl.trace import
  ...`` (import-system resolution) and ``etl.trace.trace`` directly.
- Fast-path caching (internal, no observable behavior): ``Executable`` (and
  ``BoundExecutable``) cache a lazily built input plan
  (``_build_input_plan``: expected tree + per-leaf ``(is_tensor, path, spec,
  recorded, leaf_index)`` layout) and an output plan (``_build_output_plan``:
  once-normalized output tree + per-leaf flags). ``run`` validates structure
  and leaves in one pass over the raw arguments (``_fast_structure_walk`` —
  mirrors ``core.flatten``'s node precedence and ``first_mismatch_path``
  semantics with ``leaf_vs_empty_is_mismatch=False``); any structural
  divergence falls back to the slow path (``_prepare_flat_inputs_slow``),
  which rebuilds the runtime TreeSpec and raises the canonical
  structure-mismatch error, so messages stay byte-identical.
"""

from __future__ import annotations

import collections
import dataclasses
import os
from dataclasses import is_dataclass
from typing import Any, Iterator, Tuple

import numpy as np

from etl import backends
from etl import core
from etl.backends import CompiledArtifact, LoweredProgram
from etl.core import first_mismatch_path, format_path
from etl.core import tree as _core_tree
from etl.pipeline_options import apply_env_options
from etl.trace import Graph
from etl.trace.trace import _SymbolicLeaf, _TensorSpecLeaf

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
        # Lazy fast-path caches (built on the first ``run``): the input plan
        # precomputes the per-leaf validation layout, the output plan the
        # normalized output tree + per-leaf classification. Pure derivations
        # from the signature — no observable behavior change.
        self._input_plan = None
        self._output_plan = None

    def _get_input_plan(self):
        """Fast-path input plan (structure + per-leaf layout), built once."""
        plan = self._input_plan
        if plan is None:
            plan = _build_input_plan(self.signature, {})
            self._input_plan = plan
        return plan

    def _get_output_plan(self):
        """Fast-path output plan (normalized tree + leaf flags), built once."""
        plan = self._output_plan
        if plan is None:
            plan = _build_output_plan(self.signature)
            self._output_plan = plan
        return plan

    @property
    def backend(self):
        """Backend name the executable runs on (e.g. ``"numpy"``, ``"iree"``).

        Delegates to the backend executable's ``backend_name`` class
        attribute (declared by every in-tree backend) — used by ``etl.run``
        for run-stage option/env resolution.
        """
        return getattr(self.backend_executable, "backend_name", None)

    @property
    def functions(self):
        """Function names exported by the loaded program (delegates to the
        backend executable)."""
        return self.backend_executable.functions

    @property
    def device(self):
        """Device the executable is bound to (delegates to the backend
        executable)."""
        return self.backend_executable.device

    def save(self, path):
        """Persist the executable if the backend supports it.

        Delegates to the backend executable's ``save`` when it has one.
        Backends that cannot serialize device handles must save the
        underlying CompiledArtifact and reconstruct on load — never pretend
        a device handle was serialized: if the backend executable exposes no
        callable ``save``, this raises ``core.PersistenceError`` directing the
        user to ``artifact.save(path)`` + ``etl.load``.
        """
        saver = getattr(self.backend_executable, "save", None)
        if not callable(saver):
            raise core.PersistenceError(
                f"this backend executable ({type(self.backend_executable).__name__}) "
                "cannot be saved directly — save the underlying CompiledArtifact "
                "(artifact.save(path)) and reload it via etl.load"
            )
        saver(path)

    @classmethod
    def load(cls, path, backend=None, device=None):
        """Load a persisted executable; never silently recompiles.

        Opens the etl.persist container, requires payload type
        ``"etl.compiled_artifact"`` (a lowered program raises
        ``core.PersistenceError`` directing to compile first), resolves the
        recorded backend (an explicitly given backend — name or instance —
        must match, else ``core.PersistenceError``), then reconstructs via
        ``etl.backends.CompiledArtifact.load(path)`` + ``backend.load(...)``
        and re-wraps with the decoded signature. Never re-traces, re-lowers,
        or re-compiles.
        """
        from etl import persist

        loaded = persist.load_object(path)
        if loaded.payload_type != "etl.compiled_artifact":
            raise core.PersistenceError(
                f"not an executable artifact: {path} contains payload type "
                f"{loaded.payload_type!r} — compile the lowered program first "
                f"(etl.compile), then save/load the CompiledArtifact"
            )
        backend_info = loaded.backend_info
        if not isinstance(backend_info, dict) or not backend_info.get("name"):
            raise core.PersistenceError(
                "corrupt: saved executable records no backend name"
            )
        recorded = backend_info["name"]
        if backend is not None:
            if isinstance(backend, backends.Backend):
                given_name = backend.name
            elif isinstance(backend, str):
                given_name = backend
            else:
                raise TypeError(
                    f"backend must be a registered backend name (str) or a "
                    f"Backend instance, got {type(backend).__name__}"
                )
            if given_name != recorded:
                raise core.PersistenceError(
                    f"artifact records backend {recorded!r}; cannot load it "
                    f"with backend {given_name!r} — never silently "
                    f"recompile"
                )
        resolved = backends.get(recorded)
        device = _normalize_device(device)
        artifact = backends.CompiledArtifact.load(path)
        backend_executable = resolved.load(artifact, device)
        return cls(backend_executable, artifact.signature)


class BoundExecutable:
    """Result of ``etl.bind``: an executable with pre-supplied inputs.

    Also satisfies the runnable surface of ``Executable`` (so ``etl.run``
    accepts it), supplying the bound tensors before user-provided arguments.

    Attributes:
        executable: the wrapped ``Executable``.
        bindings: dict mapping flat-input-LEAF-INDEX -> validated
            ``core.Tensor`` (positions resolved by ``bind()``).
        bound_names: dict mapping binding name -> flat leaf index (for
            diagnostics).
    """

    def __init__(self, executable, bindings, bound_names=None):
        self.executable = executable
        self.bindings = dict(bindings)
        self.bound_names = dict(bound_names) if bound_names else {}
        self._input_plan = None  # lazy fast-path plan (reduced tree layout)

    def _get_input_plan(self):
        """Fast-path input plan over the *reduced* (unbound) input tree.

        Rebuilt when the bindings dict changes (defensive: ``bindings`` is
        fixed after ``bind``; any mutation must not reuse a stale reduced
        tree)."""
        plan = self._input_plan
        if plan is None or plan[4] != tuple(sorted(self.bindings)):
            plan = _build_input_plan(self.signature, self.bindings)
            self._input_plan = plan
        return plan

    def _get_output_plan(self):
        """The wrapped executable's output plan (shared cache)."""
        return self.executable._get_output_plan()

    @property
    def signature(self):
        """The wrapped executable's ``Signature``."""
        return self.executable.signature

    @property
    def backend_executable(self):
        """The wrapped executable's backend executable."""
        return self.executable.backend_executable

    @property
    def functions(self):
        """Function names exported by the loaded program (delegates to the
        wrapped executable)."""
        return self.executable.functions

    @property
    def device(self):
        """Device the wrapped executable is bound to."""
        return self.executable.device

    @property
    def backend(self):
        """Backend name the wrapped executable runs on (delegates)."""
        return self.executable.backend


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_backend(backend):
    """Resolve a backend argument: None -> default numpy backend; str -> a
    registered backend name; ``Backend`` instance -> as-is; anything else ->
    ``TypeError``."""
    if backend is None:
        return backends.numpy_backend
    if isinstance(backend, str):
        return backends.get(backend)
    if isinstance(backend, backends.Backend):
        return backend
    raise TypeError(
        f"backend must be a registered backend name (str) or a Backend "
        f"instance, got {type(backend).__name__}"
    )


def _normalize_device(device):
    """Normalize a device argument: None -> ``core.Device("cpu", 0)`` (v1
    default); str -> ``core.Device(name)``; ``core.Device`` -> as-is;
    anything else -> ``core.DeviceError``."""
    if device is None:
        return core.Device("cpu", 0)
    if isinstance(device, str):
        return core.Device(device)
    if isinstance(device, core.Device):
        return device
    raise core.DeviceError(
        f"device must be None, a kind string (e.g. 'cpu'), or a core.Device, "
        f"got {type(device).__name__}"
    )


def _raise_bad_env_device(var: str, value: str) -> None:
    """Canonical malformed ``ETL_DEVICE`` error: name the variable and value."""
    raise core.DeviceError(
        f"{var}={value!r} is not a valid device string: expected 'kind' or "
        f"'kind:index' (e.g. 'cpu', 'cuda:0')"
    ) from None


def _parse_env_device(var: str, value: str) -> core.Device:
    """Parse an ``ETL_DEVICE`` value ``"kind[:index]"`` into a ``core.Device``.

    ``"cpu"`` -> ``Device("cpu", 0)``, ``"cuda:0"`` -> ``Device("cuda", 0)``,
    ``"cuda:3"`` -> ``Device("cuda", 3)``. A malformed value (empty kind,
    non-integer or negative index, stray colons) raises ``core.DeviceError``
    naming the environment variable and the bad value (``from None`` — the
    raw parse error never leaks).
    """
    kind, sep, index = value.partition(":")
    kind = kind.strip()
    if not kind or (sep and not index.strip()):
        _raise_bad_env_device(var, value)
    if not sep:
        index_num = 0
    else:
        try:
            index_num = int(index)
        except ValueError:
            _raise_bad_env_device(var, value)
        if index_num < 0:
            _raise_bad_env_device(var, value)
    try:
        return core.Device(kind, index_num)
    except ValueError:
        _raise_bad_env_device(var, value)


def _resolve_backend_device(backend, device, options):
    """Resolve default backend/device/compile-options for build/evaluate.

    Explicit kwargs always win; unset ones fall back to environment
    variables read lazily at call time:

    - ``ETL_BACKEND`` — backend name (default ``"numpy"`` when unset; empty
      values are treated as unset). Unset ``backend``/``device`` are resolved
      through the same registry/device normalization as explicit arguments.
    - ``ETL_DEVICE`` — device string ``"kind[:index]"`` (e.g. ``"cpu"``,
      ``"cuda:0"``), parsed into a ``core.Device``; a malformed value raises
      ``core.DeviceError`` naming the variable and the value.
    - ``ETL_TARGET_BACKENDS`` — comma-separated ``target_backends`` compile
      option list, applied when ``target_backends`` was not passed
      explicitly (any backend; those that ignore the option are unaffected).

    When neither an explicit ``target_backends`` nor ``ETL_TARGET_BACKENDS``
    is present, an iree-family backend (resolved backend name starting with
    ``"iree"``) infers it from the resolved device: ``"cuda"`` ->
    ``["cuda"]``, ``"cpu"`` -> ``["llvm-cpu"]`` — mirroring the explicit
    probe spelling ``target_backends=["cuda"]``. The numpy default path
    (no env set) is unchanged: ``numpy_backend``, ``Device("cpu", 0)``,
    options untouched.

    Returns ``(backend, device, options)``: a resolved ``Backend`` instance,
    a resolved ``core.Device``, and a copied options dict (the caller's dict
    is never mutated).
    """
    if backend is None:
        backend = os.environ.get("ETL_BACKEND") or "numpy"
    if device is None:
        raw = os.environ.get("ETL_DEVICE")
        if raw:
            device = _parse_env_device("ETL_DEVICE", raw)
    backend = _resolve_backend(backend)
    device = _normalize_device(device)
    options = dict(options)
    if "target_backends" not in options:
        raw = os.environ.get("ETL_TARGET_BACKENDS")
        if raw and raw.strip():
            options["target_backends"] = [
                part.strip() for part in raw.split(",") if part.strip()
            ]
        elif backend.name.startswith("iree"):
            if device.kind == "cuda":
                options["target_backends"] = ["cuda"]
            elif device.kind == "cpu":
                options["target_backends"] = ["llvm-cpu"]
    return backend, device, options


def lower(graph, backend=None, **options):
    """``lower(graph) -> LoweredProgram``.

    ``backend`` defaults to ``etl.backends.numpy_backend`` (a backend name or
    ``Backend`` instance is resolved through the registry; anything else ->
    ``TypeError``). The backend verifies the graph, records its signature
    (input/output TreeSpec, specs, static values) and produces a
    backend-specific lowered program. A non-``Graph`` input raises
    ``TypeError`` — no earlier-stage object is silently consumed.

    Options contract: ``options`` are validated by the backend against its
    ``KNOWN_OPTIONS`` union (unknown keys -> ``core.BackendError``; keys
    valid for other stages pass). Per-backend option env vars
    (``ETL_*``, see ``etl.pipeline_options``) are applied for keys not
    passed explicitly — precedence: explicit kwarg > env var > etl default.
    """
    if not isinstance(graph, Graph):
        raise TypeError(
            f"lower expects an etl.Graph (from etl.trace), got "
            f"{type(graph).__name__}"
        )
    backend = _resolve_backend(backend)
    options = apply_env_options(backend.name, options, "lower")
    return backend.lower(graph, dict(options))


def compile(lowered, backend=None, **options):
    """``compile(lowered) -> CompiledArtifact``.

    ``backend`` may be omitted (taken from the lowered program) but must
    match if given (``core.BackendError`` naming both otherwise). Does NOT
    silently re-lower. A non-``LoweredProgram`` input raises ``TypeError``.

    Options contract: ``options`` are validated by the backend against its
    ``KNOWN_OPTIONS`` union (unknown keys -> ``core.BackendError``; keys
    valid for other stages pass). Per-backend option env vars (e.g.
    ``ETL_IREE_COMPILE_ARGS``, ``ETL_XLA_COMPILE_OPTIONS``,
    ``ETL_TVM_TARGET``/``ETL_TVM_PASS_CONFIGS``) are applied for keys not
    passed explicitly — precedence: explicit kwarg > env var > etl default.
    """
    if not isinstance(lowered, LoweredProgram):
        raise TypeError(
            f"compile expects an etl.backends.LoweredProgram (from etl.lower), "
            f"got {type(lowered).__name__}"
        )
    if backend is not None:
        resolved = _resolve_backend(backend)
        if resolved.name != lowered.backend:
            raise core.BackendError(
                f"cannot compile a LoweredProgram produced by backend "
                f"{lowered.backend!r} with backend {resolved.name!r} — never "
                f"silently re-lower"
            )
    else:
        resolved = backends.get(lowered.backend)
    options = apply_env_options(resolved.name, options, "compile")
    return resolved.compile(lowered, dict(options))


def load(artifact, backend=None, device=None, **options):
    """``load(artifact) -> Executable``.

    ``backend`` may be omitted (taken from the artifact) but must match if
    given (``core.PersistenceError`` naming both otherwise — never a silent
    recompile). ``device`` normalizes: None -> ``core.Device("cpu", 0)`` (the
    v1 default), str -> ``core.Device(name)``, ``core.Device`` -> as-is,
    anything else -> ``core.DeviceError``. Returns the user-facing wrapper
    carrying the structured signature (the artifact's live decoded
    signature). A non-``CompiledArtifact`` input raises ``TypeError``.

    Options contract: ``options`` are backend-specific load/loader options
    (e.g. the iree adapter's ``iree_runtime_args``, the xla adapter's
    ``plugin_path``), validated by the backend against its ``KNOWN_OPTIONS``
    union (unknown keys -> ``core.BackendError``; keys valid for other
    stages pass). Per-backend option env vars (e.g. ``ETL_IREE_RUNTIME_ARGS``)
    are applied for keys not passed explicitly — precedence: explicit kwarg
    > env var > etl default.
    """
    if not isinstance(artifact, CompiledArtifact):
        raise TypeError(
            f"load expects an etl.backends.CompiledArtifact (from "
            f"etl.compile), got {type(artifact).__name__}"
        )
    if backend is not None:
        if isinstance(backend, backends.Backend):
            given_name = backend.name
        elif isinstance(backend, str):
            given_name = backend
        else:
            raise TypeError(
                f"backend must be a registered backend name (str) or a "
                f"Backend instance, got {type(backend).__name__}"
            )
        if given_name != artifact.backend:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                f"cannot load it with backend {given_name!r} — never "
                f"silently recompile"
            )
    resolved = backends.get(artifact.backend)
    device = _normalize_device(device)
    options = apply_env_options(artifact.backend, options, "load")
    if options:
        backend_executable = resolved.load(artifact, device, options)
    else:
        backend_executable = resolved.load(artifact, device)
    return Executable(backend_executable, artifact.signature)


# ---------------------------------------------------------------------------
# Structured I/O helpers (input validation / output reconstruction)
# ---------------------------------------------------------------------------


def _format_path(path: Tuple[Any, ...]) -> str:
    """Render a pytree path readably, e.g. ``[0]['x'][1]``."""
    if not path:
        return "()"
    parts = []
    for key in path:
        if isinstance(key, str):
            parts.append(f"[{key!r}]")
        else:
            parts.append(f"[{key}]")
    return "".join(parts)


def _is_tensor_leaf_spec(leaf_spec: "core.TreeSpec") -> bool:
    """True if a leaf spec marks a TENSOR position (input or output).

    Leaf markers across the codebase (all mean "a tensor lives here"):
    - ``_TensorSpecLeaf`` / ``_SymbolicLeaf`` — trace-built trees (``etl.trace``
      records markers so ``core.flatten``/``core.unflatten`` treat the
      etl dataclasses ``TensorSpec``/``SymbolicTensor`` as leaves);
    - ``core.TensorSpec`` / ``core.SymbolicTensor`` — transform-built trees
      (jvp's tangent-spec trees via ``core.flatten``; autodiff output trees);
    - ``None`` (the ``TreeSpec.type`` FIELD is None) — the canonical
      plain-leaf spec used by grad/vjp-built output trees.
    Every other leaf type is a static position (trace records the static
    value's own Python type, e.g. ``type(None)`` for a static ``None`` leaf).
    """

    leaf_type = leaf_spec.type
    return (
        leaf_type is _TensorSpecLeaf
        or leaf_type is _SymbolicLeaf
        or leaf_type is core.TensorSpec
        or leaf_type is core.SymbolicTensor
        or leaf_type is None
    )


def _walk_leaves(spec: "core.TreeSpec", prefix=(), counter=None):
    """Yield ``(leaf_spec, flat_index, key_path)`` for every leaf of ``spec``
    in pre-order.

    Mirrors ``core.flatten``'s traversal exactly: dict children take their
    sorted key from ``node_data``; all other container kinds use positional
    indices; a childless spec with zero leaves (an empty container) yields
    nothing.
    """
    if counter is None:
        counter = [0]
    if not spec.children:
        if spec.num_leaves == 1:
            yield spec, counter[0], prefix
            counter[0] += 1
        return
    is_dict = isinstance(spec.type, type) and issubclass(spec.type, dict)
    for i, child in enumerate(spec.children):
        key = spec.node_data[i] if is_dict else i
        yield from _walk_leaves(child, prefix + (key,), counter)


def _structure_matches(spec_a: "core.TreeSpec", spec_b: "core.TreeSpec") -> bool:
    """True if two TreeSpecs describe the same container structure.

    Delegates to ``core.first_mismatch_path`` with the default
    ``leaf_vs_empty_is_mismatch=False``: container nodes must match exactly
    (same node type, same ``node_data`` — e.g. sorted dict keys / dataclass
    field names, same child count); leaves match any leaf, and a leaf vs an
    empty container is NOT a mismatch here. Leaf *types* are deliberately
    ignored: trace-time trees record marker types
    (``_TensorSpecLeaf``/``_SymbolicLeaf``/static types) while run-time trees
    record concrete types — these can never be identical. The per-leaf static
    and tensor checks below validate leaf positions/types with precise errors;
    the total leaf count is checked by the caller against the recorded specs.
    """
    return first_mismatch_path(spec_a, spec_b) is None


def _fast_structure_walk(exp: "core.TreeSpec", x, leaves: list) -> bool:
    """Single-pass structure check + leaf collection for the run fast path.

    Mirrors ``core.flatten``'s node precedence (registered pytree nodes via
    MRO, namedtuple, dataclass, tuple, list, dict, leaf) and
    ``first_mismatch_path``'s container semantics with
    ``leaf_vs_empty_is_mismatch=False``: container nodes must match exactly
    (node type, ``node_data``, child count); leaves match any leaf; a leaf
    vs an empty container is NOT a mismatch; leaf *types* are ignored.

    Runtime leaves are appended to ``leaves`` in pre-order (flatten order).
    Errors ``core.flatten`` itself raises on malformed values (e.g.
    unorderable dict keys) propagate identically and at the same point of
    the traversal. Returns ``False`` at the first structural divergence —
    the caller then falls back to the slow path, which rebuilds the runtime
    TreeSpec and raises the canonical structure-mismatch error, so error
    messages stay byte-identical.
    """
    obj_type = type(x)
    for base in obj_type.__mro__:
        registered = _core_tree._PYTREE_NODE_REGISTRY.get(base)
        if registered is not None:
            flatten_fn, _ = registered
            children, _context = flatten_fn(x)
            if not isinstance(children, (list, tuple)):
                children = tuple(children)
            base, node_data = base, None  # context is deliberately not compared
            break
    else:
        if isinstance(x, tuple) and hasattr(obj_type, "_fields"):
            base, children, node_data = obj_type, x, obj_type._fields
        elif (
            dataclasses.is_dataclass(x)
            and not isinstance(x, type)
            and not obj_type.__module__.split(".")[0] == "etl"
        ):
            fields = dataclasses.fields(x)
            children = [getattr(x, field.name) for field in fields]
            base, node_data = obj_type, [field.name for field in fields]
        elif isinstance(x, tuple):
            base, children, node_data = obj_type, x, None
        elif isinstance(x, list):
            base, children, node_data = obj_type, x, None
        elif isinstance(x, dict):
            try:
                keys = sorted(x)
            except TypeError:
                # The raw sort TypeError must never leak: wrap it with the
                # same guidance core.flatten gives.
                type_names = []
                for key in x:
                    name = type(key).__name__
                    if name not in type_names:
                        type_names.append(name)
                raise TypeError(
                    f"flatten: cannot sort dict keys of mixed types (types: {type_names}); "
                    "dict keys are sorted for deterministic tree structure — use keys of one "
                    "type or register_pytree_node"
                ) from None
            children = [x[key] for key in keys]
            if isinstance(x, collections.defaultdict):
                node_data = (x.default_factory, keys)
            else:
                node_data = keys
            base = obj_type
        else:
            # Leaf.
            if not exp.children:
                leaves.append(x)
                return True
            return False
    # Container (possibly empty).
    if not exp.children:
        return not children
    if (
        exp.type != base
        or exp.node_data != node_data
        or len(exp.children) != len(children)
    ):
        return False
    for child_exp, child_x in zip(exp.children, children):
        if not _fast_structure_walk(child_exp, child_x, leaves):
            return False
    return True


def _build_input_plan(signature, bound: dict):
    """Precompute the run-time input validation layout for one signature.

    Returns ``(expected_tree, entries, total, n_bound, bound_keys)``:

    - ``expected_tree`` — the structure run-time arguments must match: the
      full ``signature.input_tree`` for a plain executable, the bound-leaf-
      reduced tree for a ``BoundExecutable``.
    - ``entries`` — one ``(is_tensor, path, spec, recorded, leaf_index)``
      tuple per leaf of the FULL input tree in pre-order (``path`` is the
      pre-rendered pytree path string, ``spec`` the tensor spec or ``None``,
      ``recorded`` the recorded static value or ``None``, ``leaf_index`` the
      flat leaf index — bound lookups stay live via the ``bound`` dict, so
      the plan is independent of the bound values).
    - ``total`` / ``n_bound`` — feed the leaf-count error message.
    - ``bound_keys`` — the sorted bound leaf indices (freshness guard for
      ``BoundExecutable`` plans).
    """
    input_tree = signature.input_tree
    total = input_tree.num_leaves
    n_bound = len(bound)
    bound_keys = tuple(sorted(bound))
    if bound:
        expected_tree = _reduce_tree(input_tree, set(bound), [0])
    else:
        expected_tree = input_tree
    entries = []
    tensor_specs = iter(signature.input_specs)
    static_values = iter(signature.static_values)
    for leaf_spec, leaf_index, path in _walk_leaves(input_tree):
        if _is_tensor_leaf_spec(leaf_spec):
            entries.append((True, _format_path(path), next(tensor_specs), None, leaf_index))
        else:
            entries.append((False, _format_path(path), None, next(static_values), leaf_index))
    return (expected_tree, tuple(entries), total, n_bound, bound_keys)


def _build_output_plan(signature):
    """Precompute the run-time output reconstruction layout.

    Returns ``(normalized_tree, flags, tensor_count, static_count)``: the
    once-normalized output tree (``_normalize_leaf_types`` — a full
    recursive copy, previously rebuilt on every run), the per-leaf
    tensor/static flags in pre-order, and the two leaf-kind counts feeding
    the signature/output checks.
    """
    output_tree = signature.output_tree
    flags = []
    tensor_count = 0
    static_count = 0
    for leaf_spec, _, _ in _walk_leaves(output_tree):
        if _is_tensor_leaf_spec(leaf_spec):
            flags.append(True)
            tensor_count += 1
        else:
            flags.append(False)
            static_count += 1
    return (
        _normalize_leaf_types(output_tree),
        tuple(flags),
        tensor_count,
        static_count,
    )


def _subspec(spec: "core.TreeSpec", path: Tuple[Any, ...]) -> "core.TreeSpec":
    """The subtree ``core.TreeSpec`` of ``spec`` at ``path`` (keys as
    produced by ``first_mismatch_path``: positional indices for non-dict
    nodes, dict keys for dict subclasses)."""
    current = spec
    for key in path:
        node_data = current.node_data
        if node_data is None:
            index = key  # hand-built spec: path keys are positional
        elif isinstance(current.type, type) and issubclass(current.type, dict):
            keys = (
                node_data[1]
                if issubclass(current.type, collections.defaultdict)
                else node_data
            )
            index = keys.index(key)
        else:
            index = key  # positional index (tuple/list/namedtuple/dataclass)
        current = current.children[index]
    return current


def _structure_desc(spec: "core.TreeSpec") -> str:
    """Human-readable description of the node a TreeSpec describes, for
    structure-mismatch messages.

    Containers are described by kind + arity (``dict with keys ['a', 'b']``,
    ``tuple of length N``, ``list of length N``, ``namedtuple of length N``,
    ``dataclass with fields ['f1', 'f2']``); childless leaves by their type
    name (``Tensor``, ``int``, ``NoneType``); childless empty containers by
    their container description with an empty list.
    """
    spec_type = spec.type
    if isinstance(spec_type, type):
        if issubclass(spec_type, dict):
            keys = spec.node_data
            if keys is not None and issubclass(spec_type, collections.defaultdict):
                keys = keys[1]
            return f"dict with keys {keys!r}"
        if issubclass(spec_type, tuple):
            if hasattr(spec_type, "_fields"):
                return f"namedtuple of length {len(spec.children)}"
            return f"tuple of length {len(spec.children)}"
        if issubclass(spec_type, list):
            return f"list of length {len(spec.children)}"
        if (
            is_dataclass(spec_type)
            and not spec_type.__module__.split(".")[0] == "etl"
        ):
            return f"dataclass with fields {spec.node_data!r}"
        return spec_type.__name__
    return repr(spec_type)


def _structure_mismatch_message(
    lead_in: str, expected_tree: "core.TreeSpec", runtime_tree: "core.TreeSpec"
) -> str:
    """Build a run-time input structure-mismatch ``core.TraceError`` message.

    ``lead_in`` is the site-specific, byte-identical lead-in; the appended
    detail follows the shared trace/pipeline convention: ``— first mismatch at
    pytree path {path}: expected {expected_desc}, got {got_desc} (expected
    {expected_spec}, got {got_spec})`` — computed from
    ``first_mismatch_path(expected_tree, runtime_tree)`` (expected =
    ``expected_tree``, got = ``runtime_tree``).
    """
    mismatch = first_mismatch_path(expected_tree, runtime_tree)
    expected_spec = _subspec(expected_tree, mismatch)
    got_spec = _subspec(runtime_tree, mismatch)
    return (
        f"{lead_in} — first mismatch at pytree path {format_path(mismatch)}: "
        f"expected {_structure_desc(expected_spec)}, got "
        f"{_structure_desc(got_spec)} (expected {expected_spec!r}, "
        f"got {got_spec!r})"
    )


def _reduce_tree(spec: "core.TreeSpec", bound: set, counter) -> "core.TreeSpec | None":
    """Copy of ``spec`` with the leaves at flat indices in ``bound`` removed.

    Recursive rules (per the bind contract):
    - a bound leaf is dropped (returns ``None``);
    - dict nodes drop the matching ``node_data`` keys;
    - dataclass nodes drop the matching ``node_data`` field names;
    - tuple/list/namedtuple nodes drop children positionally, keeping
      ``node_data`` otherwise.
    ``counter`` threads the flat leaf index (pre-order, same traversal as
    ``core.flatten``).
    """
    if not spec.children:
        if spec.num_leaves == 1:
            index = counter[0]
            counter[0] += 1
            return None if index in bound else spec
        return spec  # empty container — kept
    is_dict = isinstance(spec.type, type) and issubclass(spec.type, dict)
    is_namedtuple = (
        isinstance(spec.type, type)
        and issubclass(spec.type, tuple)
        and hasattr(spec.type, "_fields")
    )
    is_dataclass_node = (
        isinstance(spec.type, type)
        and is_dataclass(spec.type)
        and not is_namedtuple
    )
    kept_children = []
    kept_keys = []
    for i, child in enumerate(spec.children):
        reduced = _reduce_tree(child, bound, counter)
        if reduced is not None:
            kept_children.append(reduced)
            if is_dict or is_dataclass_node:
                kept_keys.append(spec.node_data[i])
    node_data = spec.node_data
    if is_dict or is_dataclass_node:
        node_data = type(node_data)(kept_keys) if node_data is not None else node_data
    return core.TreeSpec(
        type=spec.type,
        children=tuple(kept_children),
        context=spec.context,
        node_data=node_data,
    )


def _check_shape(actual_shape, spec_shape, path) -> None:
    """Validate a concrete shape against a spec's (symbolic) shape.

    Rank must match exactly (else ``core.ShapeError``). Per dim: ``None`` is
    runtime-dynamic (unchecked); a static ``int`` must equal the actual dim;
    a ``core.Dim`` with a known size must match it; a ``core.DimExpr`` is
    evaluated without bindings — when it resolves it must match, when it
    needs bindings (named symbolic dims) it is accepted (v1 has no binding
    environment).
    """
    if len(actual_shape) != len(spec_shape):
        raise core.ShapeError(
            f"rank mismatch for input at path {path}: spec shape "
            f"{tuple(spec_shape)} has rank {len(spec_shape)}, got shape "
            f"{tuple(actual_shape)} with rank {len(actual_shape)}"
        )
    for axis, entry in enumerate(spec_shape):
        if entry is None:
            continue
        if isinstance(entry, int):
            expected = entry
        elif isinstance(entry, core.Dim):
            if entry.size is None:
                continue  # symbolic dim without a known size — accepted in v1
            expected = entry.size
        else:  # DimExpr
            try:
                expected = entry.evaluate()
            except core.ShapeError:
                continue  # symbolic expression needing bindings — accepted in v1
        if expected != actual_shape[axis]:
            raise core.ShapeError(
                f"shape mismatch for input at path {path}: spec shape "
                f"{tuple(spec_shape)} vs actual shape {tuple(actual_shape)} "
                f"(dim {axis} must be {entry!r}, got {actual_shape[axis]})"
            )


def _validate_tensor(spec: "core.TensorSpec", leaf: Any, path) -> "core.Tensor":
    """Validate one runtime tensor leaf against its ``core.TensorSpec``.

    Accepts ``core.Tensor`` as-is; wraps numpy ``ndarray`` via
    ``core.from_numpy`` (documented convenience); anything else →
    ``core.TraceError``. Checks dtype (``core.DTypeError``), shape
    (``core.ShapeError``, see ``_check_shape``) and device
    (``core.DeviceError``).
    """
    if isinstance(leaf, core.Tensor):
        tensor = leaf
    elif isinstance(leaf, np.ndarray):
        tensor = core.from_numpy(leaf)
    else:
        raise core.TraceError(
            f"input at path {path} must be a core.Tensor (or a numpy ndarray, "
            f"wrapped via core.from_numpy), got {type(leaf).__qualname__}"
        )
    if tensor.dtype != spec.dtype:
        raise core.DTypeError(
            f"dtype mismatch for input at path {path}: spec dtype {spec.dtype}, "
            f"got {tensor.dtype}"
        )
    _check_shape(tensor.shape, spec.shape, path)
    if spec.device is not None and tensor.device != spec.device:
        raise core.DeviceError(
            f"device mismatch for input at path {path}: spec device "
            f"{spec.device}, got {tensor.device}"
        )
    return tensor


def _prepare_flat_inputs_slow(signature, bound: dict, args) -> list:
    """Flatten + validate run-time inputs; return flat ``core.Tensor`` list
    in block-arg (tensor-leaf) order.

    ``bound`` maps flat leaf indices to already-validated ``core.Tensor``\s
    (empty dict for a plain ``Executable``). For bound executables the user
    arguments must match the *reduced* input tree (bound leaf positions
    removed) — anything else raises ``core.TraceError`` with both specs in
    the message. Static leaves are compared by type qualname + ``==`` value
    (``core.TraceError`` naming the pytree path and values).

    This is the fallback path: it rebuilds the runtime TreeSpec via
    ``core.flatten`` and is therefore also the source of the canonical
    structure-mismatch errors. The fast path (``_prepare_flat_inputs`` with
    a cached plan) delegates here on any structural divergence, keeping
    error messages byte-identical.
    """
    input_tree = signature.input_tree
    total = input_tree.num_leaves
    user_leaves, runtime_tree = core.flatten(tuple(args))
    if bound:
        reduced = _reduce_tree(input_tree, set(bound), [0])
        if not _structure_matches(reduced, runtime_tree):
            raise core.TraceError(
                _structure_mismatch_message(
                    "run-time input structure does not match the unbound "
                    "portion of the traced signature",
                    reduced,
                    runtime_tree,
                )
            )
    else:
        if not _structure_matches(input_tree, runtime_tree):
            raise core.TraceError(
                _structure_mismatch_message(
                    "run-time input structure does not match the traced "
                    "signature",
                    input_tree,
                    runtime_tree,
                )
            )
    expected_user = total - len(bound)
    if len(user_leaves) != expected_user:
        raise core.TraceError(
            f"input leaf count mismatch: {len(user_leaves)} run-time leaves "
            f"for {expected_user} unbound leaf positions ({total} total, "
            f"{len(bound)} bound)"
        )
    expected_total = len(signature.input_specs) + len(signature.static_values)
    if expected_total != total:
        raise core.TraceError(
            f"signature mismatch: the input tree has {total} leaves but the "
            f"signature records {len(signature.input_specs)} tensor specs and "
            f"{len(signature.static_values)} static values ({expected_total} "
            f"leaves)"
        )

    tensors = []
    user_iter = iter(user_leaves)
    tensor_specs = iter(signature.input_specs)
    static_values = iter(signature.static_values)
    for leaf_spec, leaf_index, path in _walk_leaves(input_tree):
        if _is_tensor_leaf_spec(leaf_spec):
            spec = next(tensor_specs)
            if leaf_index in bound:
                tensors.append(bound[leaf_index])
            else:
                tensors.append(
                    _validate_tensor(spec, next(user_iter), _format_path(path))
                )
        else:
            recorded = next(static_values)
            leaf = next(user_iter)
            kind = type(leaf).__qualname__
            if kind != type(recorded).__qualname__ or leaf != recorded:
                raise core.TraceError(
                    f"graph was specialized on {recorded!r} (a "
                    f"{type(recorded).__qualname__}); run-time argument "
                    f"{leaf!r} (a {kind}) at path {_format_path(path)} does "
                    f"not match"
                )
    return tensors


def _prepare_flat_inputs(signature, bound: dict, args, plan=None) -> list:
    """Flatten + validate run-time inputs; return flat ``core.Tensor`` list
    in block-arg (tensor-leaf) order.

    ``bound`` maps flat leaf indices to already-validated ``core.Tensor``\s
    (empty dict for a plain ``Executable``). For bound executables the user
    arguments must match the *reduced* input tree (bound leaf positions
    removed) — anything else raises ``core.TraceError`` with both specs in
    the message. Static leaves are compared by type qualname + ``==`` value
    (``core.TraceError`` naming the pytree path and values).

    Fast path: ``plan`` (from ``_build_input_plan``, cached on the
    executable) precomputes the per-leaf layout, so structure checking and
    leaf collection run in ONE pass over the raw arguments
    (``_fast_structure_walk``) with no runtime TreeSpec built and no
    per-leaf path re-rendering. All checks and error messages are identical
    to the slow path: flatten-style errors (unorderable dict keys) raise at
    the same traversal point, and any structural divergence falls back to
    ``_prepare_flat_inputs_slow``, which rebuilds the runtime tree and
    raises the canonical structure-mismatch error.
    """
    if plan is None:
        plan = _build_input_plan(signature, bound)
    expected_tree, entries, total, n_bound, _bound_keys = plan
    user_leaves = []
    if not _fast_structure_walk(expected_tree, args, user_leaves):
        return _prepare_flat_inputs_slow(signature, bound, args)
    expected_user = total - n_bound
    if len(user_leaves) != expected_user:
        raise core.TraceError(
            f"input leaf count mismatch: {len(user_leaves)} run-time leaves "
            f"for {expected_user} unbound leaf positions ({total} total, "
            f"{n_bound} bound)"
        )
    expected_total = len(signature.input_specs) + len(signature.static_values)
    if expected_total != total:
        raise core.TraceError(
            f"signature mismatch: the input tree has {total} leaves but the "
            f"signature records {len(signature.input_specs)} tensor specs and "
            f"{len(signature.static_values)} static values ({expected_total} "
            f"leaves)"
        )
    tensors = []
    user_iter = iter(user_leaves)
    for is_tensor, path, spec, recorded, leaf_index in entries:
        if is_tensor:
            bound_tensor = bound.get(leaf_index)
            if bound_tensor is not None:
                tensors.append(bound_tensor)
            else:
                tensors.append(_validate_tensor(spec, next(user_iter), path))
        else:
            leaf = next(user_iter)
            kind = type(leaf).__qualname__
            if kind != type(recorded).__qualname__ or leaf != recorded:
                raise core.TraceError(
                    f"graph was specialized on {recorded!r} (a "
                    f"{type(recorded).__qualname__}); run-time argument "
                    f"{leaf!r} (a {kind}) at path {path} does "
                    f"not match"
                )
    return tensors


def _normalize_leaf_types(spec: "core.TreeSpec") -> "core.TreeSpec":
    """Copy of ``spec`` with dataclass-typed *leaf* specs normalized to plain
    leaves (``type(None)``).

    ``core.unflatten`` classifies every childless dataclass-typed spec as an
    empty container, but trace-time output trees record
    ``core.SymbolicTensor`` — itself a dataclass — as leaf *types*, so
    unflattening would mis-consume leaves. Real empty dataclass *containers*
    stay distinguishable: ``core.flatten`` records ``node_data`` (the field
    names, possibly an empty list) on dataclass nodes and ``None`` on leaves.
    """
    if not spec.children:
        spec_type = spec.type
        if (
            isinstance(spec_type, type)
            and dataclasses.is_dataclass(spec_type)
            and spec.node_data is None
        ):
            return core.TreeSpec(type=type(None))
        return spec
    return core.TreeSpec(
        type=spec.type,
        children=tuple(_normalize_leaf_types(child) for child in spec.children),
        context=spec.context,
        node_data=spec.node_data,
    )


def _unflatten_outputs(signature, flat_outputs, plan=None) -> Any:
    """Wrap flat backend results as ``core.Tensor``\s and rebuild the
    structured output per ``signature.output_tree``.

    Tensor leaves (per ``_is_tensor_leaf_spec``) consume the backend outputs
    in order — ``core.Tensor`` passes through, numpy ``ndarray`` is wrapped
    via ``core.from_numpy``, anything else raises ``core.BackendError``
    naming the flat index. Static leaves consume
    ``signature.output_static_values`` in order. The tree is normalized via
    ``_normalize_leaf_types`` before ``core.unflatten``.

    Fast path: ``plan`` (from ``_build_output_plan``, cached on the
    executable) carries the once-normalized output tree and the per-leaf
    tensor/static flags, so the per-call normalization copy and leaf-kind
    counting walks are skipped. All checks and error messages are identical.
    """
    if plan is None:
        plan = _build_output_plan(signature)
    normalized_tree, flags, tensor_count, static_count = plan
    wrapped = []
    for i, element in enumerate(flat_outputs):
        if isinstance(element, core.Tensor):
            wrapped.append(element)
        elif isinstance(element, np.ndarray):
            wrapped.append(core.from_numpy(element))
        else:
            raise core.BackendError(
                f"backend produced a non-tensor output at flat index {i}: "
                f"expected core.Tensor or numpy ndarray, got "
                f"{type(element).__qualname__}"
            )
    if tensor_count != len(wrapped):
        raise core.BackendError(
            f"signature/output mismatch: the output tree has {tensor_count} "
            f"tensor leaves but the backend produced {len(wrapped)} outputs"
        )
    output_static_values = signature.output_static_values
    if static_count != len(output_static_values):
        raise core.TraceError(
            f"signature mismatch: the output tree records {static_count} "
            f"static output leaves but the signature carries "
            f"{len(output_static_values)} output static values"
        )

    leaves = []
    tensor_iter = iter(wrapped)
    static_iter = iter(output_static_values)
    for flag in flags:
        if flag:
            leaves.append(next(tensor_iter))
        else:
            leaves.append(next(static_iter))
    try:
        return core.unflatten(leaves, normalized_tree)
    except ValueError as exc:
        raise core.TraceError(
            f"output tree does not match the recorded outputs: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# run / bind
# ---------------------------------------------------------------------------


def run(executable, *args, **options):
    """``run(executable, *tensors) -> structured outputs``.

    Accepts an ``Executable`` or a ``BoundExecutable`` (anything else ->
    ``TypeError``). Flattens inputs via the signature TreeSpec, validates
    dtype/shape/device against the recorded specs and static values
    (DTypeError/ShapeError/DeviceError/TraceError), calls the backend
    executable with flat tensors, and reconstructs the structured outputs
    (including recorded static output leaves).

    Options contract: ``options`` are backend-specific per-run options,
    validated by the backend executable against its ``KNOWN_OPTIONS`` union
    (unknown keys -> ``core.BackendError``; the numpy reference backend
    documents ignoring options — e.g. ``rank_context`` may be passed here
    and is forwarded to the numpy interpreter's execution context). No
    run-stage env vars exist in v1 (the plumbing is in place).
    ``evaluate`` does not forward explicit run options in v1 — pass them to
    ``etl.run`` directly.

    Internally, the executable caches the input/output leaf layouts built
    lazily on the first call (see ``_build_input_plan`` /
    ``_build_output_plan``): repeated runs validate structure and leaves in
    a single pass with no runtime TreeSpec construction, while every error
    path (structure mismatch, leaf count, static values, per-leaf tensor
    validation) raises exactly the same errors as before.
    """
    if isinstance(executable, BoundExecutable):
        bound = executable.bindings
    elif isinstance(executable, Executable):
        bound = {}
    else:
        raise TypeError(
            f"run expects an etl.Executable (from etl.load/etl.build) or a "
            f"BoundExecutable (from etl.bind), got {type(executable).__name__}"
        )
    signature = executable.signature
    flat_tensors = _prepare_flat_inputs(
        signature, bound, args, executable._get_input_plan()
    )
    options = apply_env_options(executable.backend, options, "run")
    backend_executable = executable.backend_executable
    if options:
        flat_outputs = backend_executable.run(flat_tensors, options)
    else:
        flat_outputs = backend_executable.run(flat_tensors)
    return _unflatten_outputs(
        signature, flat_outputs, executable._get_output_plan()
    )


def bind(executable, **bindings):
    """``bind(executable, w=w) -> BoundExecutable`` — argument-supply sugar.

    Validates: every binding name is an existing named input (input specs are
    named via ``core.TensorSpec(..., name=...)`` — there is no automatic
    naming; duplicate names -> ``core.TraceError`` as ambiguous); bound
    tensors are dtype/shape/device compatible with their specs (numpy
    ndarrays wrapped via ``core.from_numpy``). Returns a wrapper that
    supplies the bound values when invoked. Never alters the graph or
    recompiles; all inputs may be bound (``run`` then takes no arguments).

    Accepts only a plain ``Executable`` — binding a ``BoundExecutable``
    raises ``TypeError`` (use ``etl.bind`` once, on the executable).
    """
    if not isinstance(executable, Executable) or isinstance(executable, BoundExecutable):
        raise TypeError(
            f"bind expects an etl.Executable (from etl.load/etl.build), got "
            f"{type(executable).__name__}"
        )
    signature = executable.signature
    name_to_entry = {}
    tensor_specs = iter(signature.input_specs)
    for leaf_spec, leaf_index, _ in _walk_leaves(signature.input_tree):
        if not _is_tensor_leaf_spec(leaf_spec):
            continue
        spec = next(tensor_specs)
        name = spec.name
        if name is None:
            continue
        if name in name_to_entry:
            raise core.TraceError(
                f"ambiguous input name {name!r}: multiple TensorSpecs carry "
                f"this name — bind cannot resolve which input is meant"
            )
        name_to_entry[name] = (leaf_index, spec)

    bound = {}
    bound_names = {}
    for name, value in bindings.items():
        entry = name_to_entry.get(name)
        if entry is None:
            if not name_to_entry:
                raise core.TraceError(
                    f"cannot bind {name!r}: the graph has no named inputs — "
                    f"declare names via core.TensorSpec(shape, dtype, name=...)"
                )
            available = ", ".join(repr(key) for key in sorted(name_to_entry))
            raise core.TraceError(
                f"cannot bind {name!r}: unknown input name; available names: "
                f"{available}"
            )
        leaf_index, spec = entry
        bound[leaf_index] = _validate_tensor(spec, value, _format_path((name,)))
        bound_names[name] = leaf_index
    return BoundExecutable(executable, bound, bound_names)


# ---------------------------------------------------------------------------
# Documented shorthands (exact compositions — no other behavior)
# ---------------------------------------------------------------------------


def build(fn, *specs, backend=None, device=None, **options):
    """``build(f, *specs) -> Executable``.

    Documented shorthand for ``load(compile(lower(trace(fn, *specs),
    backend), **options), backend, device, **options)`` — no other behavior
    (docstring must stay in sync with the expansion)::

        graph = etl.trace(fn, *specs)
        lowered = lower(graph, backend=backend, **options)
        artifact = compile(lowered, **options)
        return load(artifact, backend=backend, device=device, **options)

    ``options`` are forwarded to ``lower``, ``compile`` AND ``load``
    (stage-appropriate: each stage uses the keys it understands, validates
    them against the backend's ``KNOWN_OPTIONS`` union — a key valid for no
    stage raises ``core.BackendError`` — and ignores the rest, exactly as in
    the explicit pipeline: e.g. the iree adapter's compile reads
    ``target_backends``/``iree_compile_args`` while its load reads
    ``iree_runtime_args``).

    Default resolution (explicit kwargs always win; environment variables
    are read lazily at call time): an unset ``backend`` resolves from
    ``ETL_BACKEND`` (backend name; default ``"numpy"`` when unset), an unset
    ``device`` from ``ETL_DEVICE`` (``"kind[:index]"``, e.g. ``"cpu"`` or
    ``"cuda:0"``, parsed into a ``core.Device``; a malformed value raises
    ``core.DeviceError`` naming the variable and the value). When
    ``target_backends`` is not passed explicitly, ``ETL_TARGET_BACKENDS``
    (comma-separated list) supplies it; otherwise an iree-family backend
    infers it from the resolved device — ``"cuda"`` -> ``["cuda"]``,
    ``"cpu"`` -> ``["llvm-cpu"]``. Per-backend option env vars
    (``ETL_IREE_COMPILE_ARGS``, ``ETL_IREE_RUNTIME_ARGS``,
    ``ETL_XLA_COMPILE_OPTIONS``, ``ETL_TVM_TARGET``/``ETL_TVM_PASS_CONFIGS``
    — see ``etl.pipeline_options``) are applied per stage for option keys
    not passed explicitly. Precedence (binding): explicit kwarg > env var >
    etl default. With no environment set, the numpy defaults are unchanged.
    """
    backend, device, options = _resolve_backend_device(backend, device, options)
    from etl.trace import trace as trace_fn

    if not callable(fn) and not getattr(fn, "__etl_defn__", False):
        raise TypeError(
            f"build expects a callable or an @etl.defn function, got "
            f"{type(fn).__name__} — to stage an already-traced Graph use the "
            f"explicit lower/compile/load pipeline"
        )
    graph = trace_fn(fn, *specs)
    lowered = lower(graph, backend=backend, **options)
    artifact = compile(lowered, **options)
    return load(artifact, backend=backend, device=device)


def evaluate(fn, *args, backend=None, device=None, **options):
    """``evaluate(f, *tensors) -> structured outputs``.

    Documented shorthand: derive a TensorSpec per concrete-tensor argument
    (snapshotting shape + dtype only), then build and run — no other
    behavior. Arguments that are not concrete tensors raise TypeError;
    concrete sparse tensors (``etl.sparse`` COO/CSR/CSC) are accepted and
    each derives a ``SparseTensorSpec`` instead::

        leaves, tree = core.flatten(args)
        specs = unflatten([TensorSpec(shape, dtype) for each leaf], tree)
        exe = build(fn, *specs, backend=backend, device=device, **options)
        return run(exe, *args)
    """
    spec_leaves = []
    leaf_index = 0  # flat leaf index in core.flatten(args) — error messages

    def _specs_tree(x):
        """Structure-recursive spec derivation — mirrors
        ``core.tree._flatten_into`` (same node precedence and registry):
        tensor leaves become ``core.TensorSpec``, concrete sparse tensors
        become ``SparseTensorSpec`` via ``SparseTensorSpec.from_concrete``;
        every other leaf raises the evaluate TypeError (with its flat leaf
        index in ``core.flatten(args)``)."""
        nonlocal leaf_index
        # 1. Concrete tensor leaves (unchanged behavior).
        if isinstance(x, core.Tensor):
            leaf_index += 1
            spec_leaves.append(
                core.TensorSpec(shape=tuple(x.shape), dtype=x.dtype)
            )
            return core.TreeSpec(type=None)
        if isinstance(x, np.ndarray):
            leaf_index += 1
            tensor = core.from_numpy(x)
            spec_leaves.append(
                core.TensorSpec(shape=tuple(tensor.shape), dtype=tensor.dtype)
            )
            return core.TreeSpec(type=None)
        # 2. Concrete sparse tensors (the base class covers COO/CSR/CSC —
        #    SparseTensorSpec.from_concrete validates concreteness and raises
        #    clear TypeErrors for spec/symbolic instances). Lazy import:
        #    etl.sparse imports core/ir/trace/ops/transforms only, never
        #    pipeline, so a function-local import cannot cycle.
        from etl.sparse import SparseTensor, SparseTensorSpec

        if isinstance(x, SparseTensor):
            spec = SparseTensorSpec.from_concrete(x)
            spec_children, spec_tree = core.flatten(spec)
            spec_leaves.extend(spec_children)
            leaf_index += len(core.flatten(x)[0])
            return spec_tree
        obj_type = type(x)
        # 3. Registered custom pytree nodes (MRO walk — same registry and
        #    order as core.tree._flatten_into; the sparse check above runs
        #    first because SparseTensor is itself a registered node).
        for base in obj_type.__mro__:
            registered = _core_tree._PYTREE_NODE_REGISTRY.get(base)
            if registered is not None:
                flatten_fn, _ = registered
                children, context = flatten_fn(x)
                child_specs = tuple(_specs_tree(child) for child in children)
                return core.TreeSpec(
                    type=base, children=child_specs, context=context
                )
        # 4. namedtuple instances (checked before plain tuples).
        if isinstance(x, tuple) and hasattr(obj_type, "_fields"):
            child_specs = tuple(_specs_tree(child) for child in x)
            return core.TreeSpec(
                type=obj_type, children=child_specs, node_data=obj_type._fields
            )
        # 5. dataclass instances (never the class itself; etl's own dataclass
        #    value types are leaves, like in core.tree._flatten_into).
        if (
            is_dataclass(x)
            and not isinstance(x, type)
            and not obj_type.__module__.split(".")[0] == "etl"
        ):
            fields = dataclasses.fields(x)
            child_specs = tuple(
                _specs_tree(getattr(x, field.name)) for field in fields
            )
            return core.TreeSpec(
                type=obj_type,
                children=child_specs,
                node_data=[field.name for field in fields],
            )
        # 6. Plain containers: tuple, list, dict (keys sorted — same
        #    unorderable-keys error as core.flatten).
        if isinstance(x, tuple):
            child_specs = tuple(_specs_tree(child) for child in x)
            return core.TreeSpec(type=obj_type, children=child_specs)
        if isinstance(x, list):
            child_specs = tuple(_specs_tree(child) for child in x)
            return core.TreeSpec(type=obj_type, children=child_specs)
        if isinstance(x, dict):
            try:
                keys = sorted(x)
            except TypeError:
                # The raw sort TypeError must never leak: wrap it with the
                # same guidance core.flatten gives.
                type_names = []
                for key in x:
                    name = type(key).__name__
                    if name not in type_names:
                        type_names.append(name)
                raise TypeError(
                    f"flatten: cannot sort dict keys of mixed types (types: {type_names}); "
                    "dict keys are sorted for deterministic tree structure — use keys of one "
                    "type or register_pytree_node"
                ) from None
            child_specs = tuple(_specs_tree(x[key]) for key in keys)
            if isinstance(x, collections.defaultdict):
                node_data = (x.default_factory, keys)
            else:
                node_data = keys
            return core.TreeSpec(
                type=obj_type, children=child_specs, node_data=node_data
            )
        # 7. Any other leaf (static Python values) — traced explicitly.
        raise TypeError(
            f"evaluate: arguments that are not concrete tensors raise "
            f"TypeError — argument at flat leaf index {leaf_index} is "
            f"{obj_type.__name__}; pass core.Tensor/numpy ndarray "
            f"values (and trace static values explicitly with etl.trace)"
        )

    tree = _specs_tree(args)
    structured = core.unflatten(spec_leaves, tree)
    if not isinstance(structured, tuple):
        structured = (structured,)
    executable = build(fn, *structured, backend=backend, device=device, **options)
    return run(executable, *args)
