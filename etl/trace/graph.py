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

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from etl import core
from etl import ir

__all__ = ["Graph", "StaticValue"]


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
        """Pretty-print the IR module (`ir.pretty_print(self.module)`)."""
        raise NotImplementedError(
            "etl.trace.Graph.print: Phase 2 implementation — delegate to "
            "`ir.pretty_print(self.module)` (see ./CONTEXT.md)."
        )

    def verify(self) -> None:
        """Run `ir.verify(self.module)`; raises `core.VerificationError`.

        Deliberately NOT run automatically by `trace()` — staging stays
        explicit. `etl.build` runs it as part of its documented composition.
        """
        raise NotImplementedError(
            "etl.trace.Graph.verify: Phase 2 implementation — delegate to "
            "`ir.verify(self.module)` (see ./CONTEXT.md)."
        )

    def save(self, path: str) -> None:
        """Persist the graph to a portable, versioned, integrity-checked
        container (`.etlgraph`).

        Imports `etl.persist` lazily inside the body (persist does not import
        trace — the DAG stays acyclic) and delegates to
        `persist.save_object(self, path, payload_type="graph", ...)` with
        signature info from `signature_info()`.
        """
        raise NotImplementedError(
            "etl.trace.Graph.save: Phase 2 implementation — lazy-import "
            "etl.persist and delegate to `persist.save_object` (see "
            "./CONTEXT.md)."
        )

    @classmethod
    def load(cls, path: str) -> "Graph":
        """Load a graph saved by `save()`.

        Imports `etl.persist` lazily; delegates to
        `persist.load_object(path, expected_type="graph")`. Mismatched
        format/type/integrity → `core.PersistenceError` — loading never
        silently re-traces or recompiles.
        """
        raise NotImplementedError(
            "etl.trace.Graph.load: Phase 2 implementation — lazy-import "
            "etl.persist and delegate to `persist.load_object` (see "
            "./CONTEXT.md)."
        )

    def flatten_inputs(self, args: Sequence[Any]) -> list:
        """Flatten + validate run-time inputs against the trace-time trees.

        Contract (implement in Phase 2):

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
        raise NotImplementedError(
            "etl.trace.Graph.flatten_inputs: Phase 2 implementation — see "
            "docstring contract above and ./CONTEXT.md."
        )

    def validate_inputs(self, args: Sequence[Any]) -> list:
        """Alias of `flatten_inputs` (same semantics; name used by the
        pipeline). Implement as a call to `flatten_inputs`."""
        raise NotImplementedError(
            "etl.trace.Graph.validate_inputs: Phase 2 implementation — "
            "delegate to `self.flatten_inputs(args)`."
        )

    def unflatten_outputs(self, flat_tensors: Sequence[Any]) -> Any:
        """Wrap flat results as `core.Tensor`s and rebuild the output
        structure per `output_tree`.

        Contract (implement in Phase 2):

        1. Each flat element: use as-is if already a `core.Tensor`; wrap
           numpy `ndarray` via `core.from_numpy`; anything else →
           `core.BackendError`.
        2. Insert `output_static_values` back at their recorded flat indices
           (their `path` entries).
        3. `output_tree.unflatten(leaves)` → the structured result.
        """
        raise NotImplementedError(
            "etl.trace.Graph.unflatten_outputs: Phase 2 implementation — see "
            "docstring contract above and ./CONTEXT.md."
        )

    def signature_info(self) -> dict:
        """Return a JSON-safe dict describing the graph's I/O signature for
        the persistence container (self-describing artifacts): tensor specs,
        static values, input/output trees, IR format version.

        Implement in Phase 2 (data marshaling over core/persist helpers).
        """
        raise NotImplementedError(
            "etl.trace.Graph.signature_info: Phase 2 implementation — see "
            "./CONTEXT.md serialization notes."
        )
