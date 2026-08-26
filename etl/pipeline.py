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

Implementation notes (binding):
- The signature's trace-time trees record tensor positions with
  ``_TensorSpecLeaf`` / ``_SymbolicLeaf`` markers (plain, non-dataclass
  classes — ``core.flatten``/``core.unflatten`` treat them as ordinary
  leaves); static positions record the static value's own Python type. All
  leaf classification in this module uses those markers.
- ``caution``: ``etl.trace`` (the function) shadows the submodule attribute
  in ``etl``; this module imports trace pieces via ``from etl.trace import
  ...`` (import-system resolution) and ``etl.trace.trace`` directly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import is_dataclass
from typing import Any, Iterator, Tuple

import numpy as np

from etl import backends
from etl import core
from etl.backends import CompiledArtifact, LoweredProgram
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
            resolved = _resolve_backend(backend)
            if resolved.name != recorded:
                raise core.PersistenceError(
                    f"artifact records backend {recorded!r}; cannot load it "
                    f"with backend {resolved.name!r} — never silently "
                    f"recompile"
                )
        else:
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

    @property
    def functions(self):
        """Function names exported by the loaded program (delegates to the
        wrapped executable)."""
        return self.executable.functions

    @property
    def device(self):
        """Device the wrapped executable is bound to."""
        return self.executable.device


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


def lower(graph, backend=None, **options):
    """``lower(graph) -> LoweredProgram``.

    ``backend`` defaults to ``etl.backends.numpy_backend`` (a backend name or
    ``Backend`` instance is resolved through the registry; anything else ->
    ``TypeError``). The backend verifies the graph, records its signature
    (input/output TreeSpec, specs, static values) and produces a
    backend-specific lowered program. A non-``Graph`` input raises
    ``TypeError`` — no earlier-stage object is silently consumed.
    """
    if not isinstance(graph, Graph):
        raise TypeError(
            f"lower expects an etl.Graph (from etl.trace), got "
            f"{type(graph).__name__}"
        )
    backend = _resolve_backend(backend)
    return backend.lower(graph, dict(options))


def compile(lowered, backend=None, **options):
    """``compile(lowered) -> CompiledArtifact``.

    ``backend`` may be omitted (taken from the lowered program) but must
    match if given (``core.BackendError`` naming both otherwise). Does NOT
    silently re-lower. A non-``LoweredProgram`` input raises ``TypeError``.
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
    return resolved.compile(lowered, dict(options))


def load(artifact, backend=None, device=None):
    """``load(artifact) -> Executable``.

    ``backend`` may be omitted (taken from the artifact) but must match if
    given (``core.PersistenceError`` naming both otherwise — never a silent
    recompile). ``device`` normalizes: None -> ``core.Device("cpu", 0)`` (the
    v1 default), str -> ``core.Device(name)``, ``core.Device`` -> as-is,
    anything else -> ``core.DeviceError``. Returns the user-facing wrapper
    carrying the structured signature (the artifact's live decoded
    signature). A non-``CompiledArtifact`` input raises ``TypeError``.
    """
    if not isinstance(artifact, CompiledArtifact):
        raise TypeError(
            f"load expects an etl.backends.CompiledArtifact (from "
            f"etl.compile), got {type(artifact).__name__}"
        )
    if backend is not None:
        resolved = _resolve_backend(backend)
        if resolved.name != artifact.backend:
            raise core.PersistenceError(
                f"artifact was produced by backend {artifact.backend!r}; "
                f"cannot load it with backend {resolved.name!r} — never "
                f"silently recompile"
            )
    else:
        resolved = backends.get(artifact.backend)
    device = _normalize_device(device)
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

    Container nodes must match exactly (same node type, same ``node_data`` —
    e.g. sorted dict keys / dataclass field names, same child count); leaves
    match any leaf. Leaf *types* are deliberately ignored: trace-time trees
    record marker types (``_TensorSpecLeaf``/``_SymbolicLeaf``/static types)
    while run-time trees record concrete types — these can never be
    identical. The per-leaf static and tensor checks below validate leaf
    positions/types with precise errors; the total leaf count is checked by
    the caller against the recorded specs.
    """
    if spec_a.children and spec_b.children:
        if spec_a.type != spec_b.type or spec_a.node_data != spec_b.node_data:
            return False
        return len(spec_a.children) == len(spec_b.children) and all(
            _structure_matches(child_a, child_b)
            for child_a, child_b in zip(spec_a.children, spec_b.children)
        )
    # Both childless: both leaves, or both empty containers (leaf-count
    # validation elsewhere reports an empty-container-vs-leaf mismatch).
    return bool(spec_a.children) == bool(spec_b.children)


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
    is_dataclass = (
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
            if is_dict or is_dataclass:
                kept_keys.append(spec.node_data[i])
    node_data = spec.node_data
    if is_dict or is_dataclass:
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


def _prepare_flat_inputs(signature, bound: dict, args) -> list:
    """Flatten + validate run-time inputs; return flat ``core.Tensor`` list
    in block-arg (tensor-leaf) order.

    ``bound`` maps flat leaf indices to already-validated ``core.Tensor``\s
    (empty dict for a plain ``Executable``). For bound executables the user
    arguments must match the *reduced* input tree (bound leaf positions
    removed) — anything else raises ``core.TraceError`` with both specs in
    the message. Static leaves are compared by type qualname + ``==`` value
    (``core.TraceError`` naming the pytree path and values).
    """
    input_tree = signature.input_tree
    total = input_tree.num_leaves
    user_leaves, runtime_tree = core.flatten(tuple(args))
    if bound:
        reduced = _reduce_tree(input_tree, set(bound), [0])
        if not _structure_matches(reduced, runtime_tree):
            raise core.TraceError(
                f"run-time input structure does not match the unbound portion "
                f"of the traced signature: got {runtime_tree}, expected "
                f"{reduced}"
            )
    else:
        if not _structure_matches(input_tree, runtime_tree):
            raise core.TraceError(
                f"run-time input structure does not match the traced "
                f"signature: got {runtime_tree}, expected {input_tree}"
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
        if leaf_spec.type is _TensorSpecLeaf:
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


def _unflatten_outputs(signature, flat_outputs) -> Any:
    """Wrap flat backend results as ``core.Tensor``\s and rebuild the
    structured output per ``signature.output_tree``.

    Tensor leaves (``_SymbolicLeaf`` positions) consume the backend outputs
    in order — ``core.Tensor`` passes through, numpy ``ndarray`` is wrapped
    via ``core.from_numpy``, anything else raises ``core.BackendError``
    naming the flat index. Static leaves consume
    ``signature.output_static_values`` in order. The tree is normalized via
    ``_normalize_leaf_types`` before ``core.unflatten``.
    """
    output_tree = signature.output_tree
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
    # Pre-count leaf kinds so iterator exhaustion is a clear signature error.
    static_count = 0
    tensor_count = 0
    for leaf_spec, _, _ in _walk_leaves(output_tree):
        if leaf_spec.type is _SymbolicLeaf:
            tensor_count += 1
        else:
            static_count += 1
    if tensor_count != len(wrapped):
        raise core.BackendError(
            f"signature/output mismatch: the output tree has {tensor_count} "
            f"tensor leaves but the backend produced {len(wrapped)} outputs"
        )
    if static_count != len(signature.output_static_values):
        raise core.TraceError(
            f"signature mismatch: the output tree records {static_count} "
            f"static output leaves but the signature carries "
            f"{len(signature.output_static_values)} output static values"
        )

    leaves = []
    tensor_iter = iter(wrapped)
    static_iter = iter(signature.output_static_values)
    for leaf_spec, _, _ in _walk_leaves(output_tree):
        if leaf_spec.type is _SymbolicLeaf:
            leaves.append(next(tensor_iter))
        else:
            leaves.append(next(static_iter))
    try:
        return core.unflatten(leaves, _normalize_leaf_types(output_tree))
    except ValueError as exc:
        raise core.TraceError(
            f"output tree does not match the recorded outputs: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# run / bind
# ---------------------------------------------------------------------------


def run(executable, *args):
    """``run(executable, *tensors) -> structured outputs``.

    Accepts an ``Executable`` or a ``BoundExecutable`` (anything else ->
    ``TypeError``). Flattens inputs via the signature TreeSpec, validates
    dtype/shape/device against the recorded specs and static values
    (DTypeError/ShapeError/DeviceError/TraceError), calls the backend
    executable with flat tensors, and reconstructs the structured outputs
    (including recorded static output leaves).
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
    flat_tensors = _prepare_flat_inputs(signature, bound, args)
    flat_outputs = executable.backend_executable.run(flat_tensors)
    return _unflatten_outputs(signature, flat_outputs)


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
        if leaf_spec.type is not _TensorSpecLeaf:
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
    backend), ...), backend, device)`` — no other behavior (docstring must
    stay in sync with the expansion)::

        graph = etl.trace(fn, *specs)
        lowered = lower(graph, backend=backend, **options)
        artifact = compile(lowered)
        return load(artifact, backend=backend, device=device)
    """
    from etl.trace import trace as trace_fn

    graph = trace_fn(fn, *specs)
    lowered = lower(graph, backend=backend, **options)
    artifact = compile(lowered)
    return load(artifact, backend=backend, device=device)


def evaluate(fn, *args, backend=None, device=None, **options):
    """``evaluate(f, *tensors) -> structured outputs``.

    Documented shorthand: derive a TensorSpec per concrete-tensor argument
    (snapshotting shape + dtype only), then build and run — no other
    behavior. Arguments that are not concrete tensors raise TypeError::

        leaves, tree = core.flatten(args)
        specs = unflatten([TensorSpec(shape, dtype) for each leaf], tree)
        exe = build(fn, *specs, backend=backend, device=device, **options)
        return run(exe, *args)
    """
    leaves, tree = core.flatten(args)
    spec_leaves = []
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, core.Tensor):
            tensor = leaf
        elif isinstance(leaf, np.ndarray):
            tensor = core.from_numpy(leaf)
        else:
            raise TypeError(
                f"evaluate: arguments that are not concrete tensors raise "
                f"TypeError — argument at flat leaf index {index} is "
                f"{type(leaf).__name__}; pass core.Tensor/numpy ndarray "
                f"values (and trace static values explicitly with etl.trace)"
            )
        spec_leaves.append(
            core.TensorSpec(shape=tuple(tensor.shape), dtype=tensor.dtype)
        )
    structured = core.unflatten(spec_leaves, tree)
    if not isinstance(structured, tuple):
        structured = (structured,)
    executable = build(fn, *structured, backend=backend, device=device, **options)
    return run(executable, *args)
