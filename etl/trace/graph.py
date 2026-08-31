"""`Graph` — a traced numerical program: `ir.Module` + input/output trees.

Layout of a traced `Graph` (binding contract — see `./CONTEXT.md`):

* `module`: an `ir.Module` holding one entry `ir.Function` (named "main"; the
  name is not part of the public contract). Its block args are the tensor
  inputs, in the flat order of `tensor_specs`; its terminator `return` yields
  the graph's symbolic results in the flat order of `output_tree`'s
  SymbolicTensor leaves.
* `input_specs`: `core.TreeSpec` of the traced function's argument tuple —
  each leaf is either the original `core.TensorSpec` (tensor input) or the
  original static value (graph specialization). Reconstructs the exact trace
  call structure.
* `tensor_specs`: tuple of `core.TensorSpec` for the tensor leaves only, in
  flat leaf order == function block-arg order.
* `static_values`: tuple of `StaticValue` records (flat leaf index, pytree
  path, value, type name) for static input leaves — validated at run time by
  `flatten_inputs`.
* `output_tree`: `core.TreeSpec` of the traced function's return value;
  leaves are the `core.SymbolicTensor` results (or recorded static leaves,
  see `output_static_values`).
* `output_static_values`: tuple of `StaticValue` for static output leaves,
  re-inserted by `unflatten_outputs`.
* `source_locations`: dict mapping input SymbolicTensor / `ir.Value` ids to
  the `ir.Location` captured at the `trace()` call site.

The constructor accepts a prebuilt `module` + trees so transforms
(`vectorize`/`vmap`/`grad`/...) can construct `Graph`s directly from
transformed IR.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Sequence, Tuple

import numpy as np

from etl import core
from etl import ir
from etl.core import describe_node, first_mismatch_path, format_path
from etl.core.tree import _subtree_spec_at  # shared mismatch-path subtree lookup

from ._tree import _iter_leaf_paths  # the ONE leaf-path iterator (see _tree.py)

__all__ = ["Graph", "StaticValue"]

#: Persist-container payload type tag for `.etlgraph` files.
_PAYLOAD_TYPE = "graph"


@dataclass(frozen=True)
class StaticValue:
    """Record of one static (graph-specializing) input or output leaf.

    Attributes:
        index: Flat leaf index in the input (or output) tree.
        path: Pytree key path of the leaf within the tree.
        value: The static Python value itself (None/bool/int/float/complex/
            str/Enum/dtype/slice — see the static-value predicate in
            `./trace.py`).
        kind: `type(value).__qualname__` — validated at run time so e.g. a
            recorded `1` never matches a passed `True`.
    """

    index: int
    path: Tuple[Any, ...]
    value: Any
    kind: str


def _static_record(index: int, path: Tuple[Any, ...], value: Any) -> StaticValue:
    """One `StaticValue` record for a static input/output leaf.

    The single construction site for static-leaf records (trace.py records
    inputs AND outputs with it; `kind` snapshots `type(value).__qualname__`
    so a recorded `1` never matches a run-time `True`).
    """
    return StaticValue(
        index=index, path=path, value=value, kind=type(value).__qualname__
    )


# --- persistence helpers (StaticValue/Location have no codec entries) ----------


def _encode_static_value(record: StaticValue) -> dict:
    """Encode one `StaticValue` as a plain dict.

    ``save_object`` auto-encodes the payload via the persist codec, so the
    raw fields (incl. the static value itself) round-trip automatically; the
    dict layout keeps the record self-describing.
    """
    return {
        "index": record.index,
        "path": tuple(record.path),
        "value": record.value,
        "kind": record.kind,
    }


def _decode_static_value(data: Any) -> StaticValue:
    """Rebuild a `StaticValue` from its payload dict (strict validation)."""
    if not isinstance(data, dict):
        raise core.PersistenceError(
            f"corrupt graph payload: StaticValue record must be a dict, got "
            f"{type(data).__name__}"
        )
    try:
        index = data["index"]
        path = tuple(data["path"])
        value = data["value"]
        kind = data["kind"]
    except KeyError as exc:
        raise core.PersistenceError(
            f"corrupt graph payload: StaticValue record is missing field "
            f"{exc.args[0]!r}"
        ) from exc
    if not isinstance(index, int) or isinstance(index, bool):
        raise core.PersistenceError(
            f"corrupt graph payload: StaticValue.index must be an int, got "
            f"{index!r}"
        )
    if not isinstance(kind, str):
        raise core.PersistenceError(
            f"corrupt graph payload: StaticValue.kind must be a str, got "
            f"{kind!r}"
        )
    return StaticValue(index=index, path=path, value=value, kind=kind)


def _encode_location(location: Any) -> Any:
    """Encode one `ir.Location` (or None) as a plain dict for the payload."""
    if location is None:
        return None
    return {
        "file": location.file,
        "line": location.line,
        "col": location.col,
        "code_snippet": location.code_snippet,
    }


def _decode_location(data: Any) -> Any:
    """Rebuild an `ir.Location` (or None) from its payload dict."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise core.PersistenceError(
            f"corrupt graph payload: source location must be a dict, got "
            f"{type(data).__name__}"
        )
    try:
        return ir.Location(
            file=data["file"],
            line=data["line"],
            col=data["col"],
            code_snippet=data["code_snippet"],
        )
    except (KeyError, TypeError) as exc:
        raise core.PersistenceError(
            f"corrupt graph payload: invalid source location {data!r}"
        ) from exc


# --- run-time input/output validation helpers --------------------------------


def _same_structure(spec: "core.TreeSpec", other: "core.TreeSpec") -> bool:
    """True if two TreeSpecs describe the same container structure.

    Delegates to `core.first_mismatch_path` (default flag — both-childless
    always matches): container nodes must match exactly (same node type,
    same ``node_data`` — e.g. sorted dict keys, same child count); leaves
    match any leaf. Leaf *types* are deliberately ignored: the trace-time
    tree records `core.TensorSpec`/`core.SymbolicTensor` leaf types while
    the run-time tree records concrete `core.Tensor`/ndarray/static types —
    these can never be identical. The per-leaf static and tensor checks
    below validate leaf positions/types with precise errors; the total leaf
    count is checked by the caller against the recorded specs.
    """
    return first_mismatch_path(spec, other) is None


def _check_shape(actual_shape: Tuple[int, ...], spec_shape: Tuple[Any, ...], path) -> None:
    """Validate a concrete shape against a spec's (symbolic) shape.

    Rank must match exactly (else `core.ShapeError`). Per dim: ``None`` is
    runtime-dynamic (unchecked); a static ``int`` must equal the actual dim;
    a `core.Dim` with a known size must match it; a `core.DimExpr` is
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


def _validate_tensor_leaf(spec: "core.TensorSpec", leaf: Any, path) -> "core.Tensor":
    """Validate one runtime tensor leaf against its `core.TensorSpec`.

    Accepts `core.Tensor` as-is; wraps numpy ``ndarray`` via
    `core.from_numpy` (documented convenience); anything else →
    `core.TraceError`. Checks dtype (`core.DTypeError`), shape
    (`core.ShapeError`, see `_check_shape`) and device (`core.DeviceError`).
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


def _normalize_leaf_types(spec: "core.TreeSpec") -> "core.TreeSpec":
    """Copy of ``spec`` with dataclass-typed *leaf* specs normalized to plain
    leaves (``type(None)``).

    ``core.unflatten`` classifies every childless dataclass-typed spec as an
    empty container, but trace-time output trees record
    `core.SymbolicTensor` — itself a dataclass — as leaf *types*, so
    unflattening would mis-consume leaves. Real empty dataclass *containers*
    stay distinguishable: ``core.flatten`` records ``node_data`` (the field
    names, possibly an empty list) on dataclass nodes and ``None`` on leaves.
    """
    if not spec.children:
        spec_type = spec.type
        if (
            isinstance(spec_type, type)
            and is_dataclass(spec_type)
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


class Graph:
    """A traced graph: IR module plus the input/output trees that let the
    pipeline flatten/validate/unflatten structured data around the module."""

    def __init__(
        self,
        module: "ir.Module",
        input_specs: "core.TreeSpec",
        tensor_specs: Sequence["core.TensorSpec"],
        output_tree: "core.TreeSpec",
        static_values: Sequence[StaticValue] = (),
        output_static_values: Sequence[StaticValue] = (),
        source_locations: dict = None,
    ) -> None:
        # Plain attribute assignment only (trivial). All invariants between
        # module/block args and the trees are established by `trace()` and
        # by transforms; `verify()` checks IR-level invariants.
        self.module = module
        self.input_specs = input_specs
        self.tensor_specs = tuple(tensor_specs)
        self.output_tree = output_tree
        self.static_values = tuple(static_values)
        self.output_static_values = tuple(output_static_values)
        self.source_locations = dict(source_locations) if source_locations else {}

    def __repr__(self) -> str:
        return (
            f"<Graph {len(self.tensor_specs)} tensor inputs, "
            f"{len(self.static_values)} static values>"
        )

    def print(self) -> None:
        """Pretty-print the IR module (`ir.pretty_print(self.module)`) to
        stdout."""
        print(ir.pretty_print(self.module))

    def verify(self) -> None:
        """Run `ir.verify(self.module)`; raises `core.VerificationError`.

        Deliberately NOT run automatically by `trace()` — staging stays
        explicit. `etl.build` runs it as part of its documented composition.
        """
        ir.verify(self.module)

    def save(self, path: str) -> None:
        """Persist the graph to a portable, versioned, integrity-checked
        container (`.etlgraph`).

        Imports `etl.persist` lazily inside the body (persist does not import
        trace — the DAG stays acyclic) and delegates to
        `persist.save_object` with payload type "graph" and the signature
        info from `signature_info()`. The payload carries the serialized IR
        module (`ir.serialize_module`), the input/output TreeSpecs, the
        tensor specs, the static-value records, and the source locations.
        """
        from etl import persist

        payload_fields = {
            "module": ir.serialize_module(self.module),
            "input_specs": self.input_specs,
            "tensor_specs": tuple(self.tensor_specs),
            "output_tree": self.output_tree,
            "static_values": [_encode_static_value(r) for r in self.static_values],
            "output_static_values": [
                _encode_static_value(r) for r in self.output_static_values
            ],
            "source_locations": {
                value_id: _encode_location(location)
                for value_id, location in self.source_locations.items()
            },
        }
        persist.save_object(
            path,
            _PAYLOAD_TYPE,
            payload_fields,
            signature_info=self.signature_info(),
        )

    @classmethod
    def load(cls, path: str) -> "Graph":
        """Load a graph saved by `save()`.

        Imports `etl.persist` lazily; delegates to
        `persist.load_object(path, expected_payload_type="graph")`.
        Mismatched format/type/integrity → `core.PersistenceError` — loading
        never silently re-traces or recompiles. The IR module is rebuilt via
        `ir.deserialize_module` (which re-verifies it); trees, specs, static
        records and source locations are reconstructed from the payload.
        """
        from etl import persist

        loaded = persist.load_object(path, expected_payload_type=_PAYLOAD_TYPE)
        payload = loaded.payload
        try:
            module_payload = payload["module"]
            input_specs = payload["input_specs"]
            tensor_specs = payload["tensor_specs"]
            output_tree = payload["output_tree"]
            static_data = payload["static_values"]
            output_static_data = payload["output_static_values"]
            locations_data = payload["source_locations"]
        except KeyError as exc:
            raise core.PersistenceError(
                f"corrupt graph payload: missing field {exc.args[0]!r}"
            ) from exc

        module = ir.deserialize_module(module_payload)
        if not isinstance(input_specs, core.TreeSpec) or not isinstance(
            output_tree, core.TreeSpec
        ):
            raise core.PersistenceError(
                "corrupt graph payload: input_specs/output_tree must decode "
                "to core.TreeSpec"
            )
        tensor_specs = tuple(tensor_specs)
        if not all(isinstance(spec, core.TensorSpec) for spec in tensor_specs):
            raise core.PersistenceError(
                "corrupt graph payload: tensor_specs must decode to "
                "core.TensorSpec entries"
            )
        return cls(
            module,
            input_specs,
            tensor_specs,
            output_tree,
            tuple(_decode_static_value(record) for record in static_data),
            tuple(_decode_static_value(record) for record in output_static_data),
            {
                value_id: _decode_location(location)
                for value_id, location in locations_data.items()
            },
        )

    def flatten_inputs(self, args: Sequence[Any]) -> list:
        """Flatten + validate run-time inputs against the trace-time trees.

        Contract:

        1. `args` = positional arguments matching the traced function's
           signature; each may be a nested structure (tuple/list/dict/
           namedtuple/dataclass). The pytree structure must equal
           `input_specs`'s structure — else `core.TraceError` (path in msg).
        2. Static leaves: type (`kind`) and `==` value must match the
           recorded `StaticValue` — else `core.TraceError` ("graph was
           specialized on X; run-time argument Y does not match").
        3. Tensor leaves: `core.Tensor` (or numpy `ndarray`, wrapped via
           `core.from_numpy` — documented convenience):
           - dtype != spec.dtype → `core.DTypeError`
           - shape does not unify with the spec's `Dim`/`DimExpr` shape
             (rank must match; static dims must be equal; symbolic dims bind)
             → `core.ShapeError`
           - spec.device set and != tensor.device → `core.DeviceError`
        4. Return the flat list of validated `core.Tensor`s in block-arg
           order.

        Error messages include the pytree path and spec vs. actual values.
        """
        leaves, spec = core.flatten(tuple(args))
        if not _same_structure(spec, self.input_specs):
            mismatch_path = first_mismatch_path(self.input_specs, spec)
            expected_spec = _subtree_spec_at(self.input_specs, mismatch_path)
            got_spec = _subtree_spec_at(spec, mismatch_path)
            raise core.TraceError(
                f"run-time input structure does not match the traced "
                f"signature — first mismatch at pytree path "
                f"{format_path(mismatch_path)}: expected "
                f"{describe_node(expected_spec)}, got {describe_node(got_spec)} "
                f"(expected {expected_spec!r}, got {got_spec!r})"
            )
        total = len(self.tensor_specs) + len(self.static_values)
        if len(leaves) != total:
            raise core.TraceError(
                f"input leaf count mismatch: the input tree has {len(leaves)} "
                f"leaves but the graph records {len(self.tensor_specs)} tensor "
                f"inputs and {len(self.static_values)} static values "
                f"({total} leaves)"
            )
        static_by_index = {}
        for record in self.static_values:
            if not 0 <= record.index < len(leaves):
                raise core.TraceError(
                    f"graph records a static input at leaf index "
                    f"{record.index}, but the input tree has {len(leaves)} "
                    f"leaves"
                )
            if record.index in static_by_index:
                raise core.TraceError(
                    f"graph records duplicate static input leaf index "
                    f"{record.index}"
                )
            static_by_index[record.index] = record

        tensors = []
        specs = iter(self.tensor_specs)
        for i, (path, leaf) in enumerate(
            zip(_iter_leaf_paths(spec), leaves)  # shared leaf-path iterator
        ):
            record = static_by_index.get(i)
            if record is not None:
                kind = type(leaf).__qualname__
                if kind != record.kind or leaf != record.value:
                    raise core.TraceError(
                        f"graph was specialized on {record.value!r} (a "
                        f"{record.kind}); run-time argument {leaf!r} (a "
                        f"{kind}) at path {path} does not match"
                    )
                continue
            # Non-static leaf → tensor input (counts checked above, so the
            # iterator can never run dry here).
            tensors.append(_validate_tensor_leaf(next(specs), leaf, path))
        return tensors

    def validate_inputs(self, args: Sequence[Any]) -> list:
        """Alias of `flatten_inputs` (same semantics; name used by the
        pipeline). Implemented as a call to `flatten_inputs`."""
        return self.flatten_inputs(args)

    def unflatten_outputs(self, flat_tensors: Sequence[Any]) -> Any:
        """Wrap flat results as `core.Tensor`s and rebuild the output
        structure per `output_tree`.

        Contract:

        1. Each flat element: use as-is if already a `core.Tensor`; wrap
           numpy `ndarray` via `core.from_numpy`; anything else →
           `core.BackendError`.
        2. Insert `output_static_values` back at their recorded flat indices.
           Built in ONE pass over the combined leaf positions: tensor leaves
           come from `flat_tensors` in order, static leaves from the records
           at their recorded indices (validated: in range, no duplicates).
        3. `core.unflatten(leaves, normalized_output_tree)` → the structured
           result. The tree is normalized via `_normalize_leaf_types` first:
           trace records `core.SymbolicTensor` (a dataclass) as leaf types,
           which `core.unflatten` would misclassify as empty containers.
        """
        wrapped = []
        for i, element in enumerate(flat_tensors):
            if isinstance(element, core.Tensor):
                wrapped.append(element)
            elif isinstance(element, np.ndarray):
                wrapped.append(core.from_numpy(element))
            else:
                raise core.BackendError(
                    f"backend produced a non-tensor output at flat index "
                    f"{i}: expected core.Tensor or numpy ndarray, got "
                    f"{type(element).__qualname__}"
                )
        static_by_index = {}
        for record in self.output_static_values:
            if record.index in static_by_index:
                raise core.TraceError(
                    f"graph records duplicate static output leaf index "
                    f"{record.index}"
                )
            static_by_index[record.index] = record.value
        total = len(wrapped) + len(static_by_index)
        for record in self.output_static_values:
            if not 0 <= record.index < total:
                raise core.TraceError(
                    f"graph records a static output at leaf index "
                    f"{record.index}, out of range for {len(wrapped)} tensor "
                    f"outputs"
                )
        leaves = []
        wrapped_iter = iter(wrapped)
        for i in range(total):
            if i in static_by_index:
                leaves.append(static_by_index[i])
            else:
                leaves.append(next(wrapped_iter))
        try:
            return core.unflatten(leaves, _normalize_leaf_types(self.output_tree))
        except ValueError as exc:
            raise core.TraceError(
                f"output tree does not match the recorded outputs: {exc}"
            ) from exc

    def signature_info(self) -> dict:
        """Return a JSON-safe dict describing the graph's I/O signature.

        The keys match the fields of `etl.backends.Signature`
        (`input_tree`, `output_tree`, `input_specs`, `output_specs`,
        `static_values`, `output_static_values`) so backends can construct
        their signature from it. Every value is encoded with
        `persist.encode_value` (lazy import), so the whole dict is
        JSON-safe as passed to the persistence container. `output_specs` is
        derived from the entry function's `return` terminator — its operands
        ARE the output tree's SymbolicTensor leaves in leaf order. The IR
        format version travels inside the serialized module payload itself
        (`ir.serialize_module` records and `ir.deserialize_module` validates
        it).
        """
        from etl import persist

        return {
            "input_tree": persist.encode_value(self.input_specs),
            "output_tree": persist.encode_value(self.output_tree),
            "input_specs": persist.encode_value(tuple(self.tensor_specs)),
            "output_specs": persist.encode_value(self._output_specs()),
            "static_values": persist.encode_value(
                tuple(record.value for record in self.static_values)
            ),
            "output_static_values": persist.encode_value(
                tuple(record.value for record in self.output_static_values)
            ),
        }

    def _output_specs(self) -> tuple:
        """`core.TensorSpec` for each symbolic result, in `return`-terminator
        (== output-tree leaf) order. Derived from the entry function's
        result types — the terminator operands are exactly the
        `core.SymbolicTensor` leaves of `output_tree`."""
        function = self.module.main
        return tuple(
            core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
            for value_type in function.output_types
        )
