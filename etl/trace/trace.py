"""The tracer: `trace(fn_or_defn, *specs) -> Graph`.

Executes the user function ONCE under an active `ir.Builder`; every tensor op
encountered appends IR to the function body. Static Python values keep Python
semantics (normal `if`/loops over static values work — they specialize the
graph). Runtime tensor control flow must use `etl.cond` / `etl.while_loop` /
`etl.scan` (see `./control_flow.py`).

The step-by-step algorithm below is the BINDING CONTRACT for the Phase 2
implementation; `./CONTEXT.md` summarizes it.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import os
from typing import Any, Iterator, Tuple

import numpy as np

from etl import core
from etl import ir
from etl.core import tree as _core_tree

from .builder import with_builder
from .defn import Defn
from .graph import Graph, StaticValue

__all__ = ["trace"]

# The registered-custom-node table of core's pytrees. `register_pytree_node`
# mutates this dict in place, so the alias stays live. Trace trees honor the
# same registrations as `core.flatten`.
_PYTREE_NODE_REGISTRY = _core_tree._PYTREE_NODE_REGISTRY


class _TensorSpecLeaf:
    """Leaf marker standing for a tensor-INPUT position in a trace pytree.

    ``core.flatten`` treats every dataclass instance as a pytree container,
    so a ``TensorSpec`` leaf would be descended into (and ``core.unflatten``
    counts dataclass-typed childless specs as zero-leaf containers). Trace
    trees therefore record tensor-input positions with this plain
    (non-dataclass) marker, which ``core.flatten``/``core.unflatten`` handle
    as an ordinary leaf — keeping ``TreeSpec.num_leaves``, structural
    equality, and downstream ``core.unflatten`` coherent. The marker carries
    the original ``core.TensorSpec`` for classification/error messages.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: "core.TensorSpec") -> None:
        self.spec = spec

    def __repr__(self) -> str:
        return f"_TensorSpecLeaf({self.spec!r})"


class _SymbolicLeaf:
    """Leaf marker standing for a symbolic RESULT position in a trace pytree.

    Same rationale as ``_TensorSpecLeaf`` for ``core.SymbolicTensor`` (also a
    frozen dataclass). Carries the ``core.SymbolicTensor`` result.
    """

    __slots__ = ("symbolic",)

    def __init__(self, symbolic: "core.SymbolicTensor") -> None:
        self.symbolic = symbolic

    def __repr__(self) -> str:
        return f"_SymbolicLeaf({self.symbolic!r})"


def _flatten_trace(obj: Any) -> Tuple[list, "core.TreeSpec"]:
    """Flatten a pytree like ``core.flatten``, keeping ``TensorSpec`` and
    ``SymbolicTensor`` instances as leaves (via the markers above).

    Mirrors ``core.tree._flatten_into``'s container rules exactly (registered
    custom nodes, namedtuple, dataclass, tuple, list, dict with sorted keys);
    the only difference is the stop-set. The returned ``core.TreeSpec`` is
    fully ``core.unflatten``-compatible.
    """
    leaves: list = []
    return leaves, _flatten_trace_into(obj, leaves)


def _flatten_trace_into(obj: Any, leaves: list) -> "core.TreeSpec":
    if isinstance(obj, core.TensorSpec):
        leaves.append(_TensorSpecLeaf(obj))
        return core.TreeSpec(type=_TensorSpecLeaf)
    if isinstance(obj, core.SymbolicTensor):
        leaves.append(_SymbolicLeaf(obj))
        return core.TreeSpec(type=_SymbolicLeaf)
    obj_type = type(obj)
    # 1. Registered custom types (walk the MRO so registered base classes
    #    catch subclasses; exact type first) — same as core.tree.
    for base in obj_type.__mro__:
        registered = _PYTREE_NODE_REGISTRY.get(base)
        if registered is not None:
            flatten_fn, _ = registered
            children, context = flatten_fn(obj)
            child_specs = tuple(
                _flatten_trace_into(child, leaves) for child in children
            )
            return core.TreeSpec(type=base, children=child_specs, context=context)
    # 2. namedtuple instances (checked before plain tuples).
    if isinstance(obj, tuple) and hasattr(obj_type, "_fields"):
        child_specs = tuple(_flatten_trace_into(child, leaves) for child in obj)
        return core.TreeSpec(
            type=obj_type, children=child_specs, node_data=obj_type._fields
        )
    # 3. dataclass instances (never the class itself).
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = dataclasses.fields(obj)
        child_specs = tuple(
            _flatten_trace_into(getattr(obj, field.name), leaves) for field in fields
        )
        return core.TreeSpec(
            type=obj_type,
            children=child_specs,
            node_data=[field.name for field in fields],
        )
    # 4-6. Plain containers: tuple, list, dict (keys sorted for determinism).
    if isinstance(obj, tuple):
        child_specs = tuple(_flatten_trace_into(child, leaves) for child in obj)
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, list):
        child_specs = tuple(_flatten_trace_into(child, leaves) for child in obj)
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, dict):
        keys = sorted(obj)
        child_specs = tuple(
            _flatten_trace_into(obj[key], leaves) for key in keys
        )
        return core.TreeSpec(type=obj_type, children=child_specs, node_data=keys)
    # Leaf: anything else (None, scalars, static values, concrete tensors, ...).
    leaves.append(obj)
    return core.TreeSpec(type=obj_type)


def trace(fn_or_defn: Any, *specs: Any) -> Graph:
    """Trace `fn_or_defn` with the given input specs → `Graph`.

    ALGORITHM (binding):

    1. Unwrap: a `Defn` (or any object with `__etl_defn__`) yields its `fn`;
       plain callables are accepted as-is.
    2. Treat the `specs` tuple (the traced function's positional arguments)
       as one pytree. Flatten via `core.TreeSpec`; every leaf must be either:
         - `core.TensorSpec` → tensor input (shape may contain `Dim`/
           `DimExpr`; `None` dims = runtime-dynamic, unchecked),
         - a static Python value per `_is_static_value` → graph
           specialization,
         - anything else (incl. concrete `core.Tensor`, numpy arrays,
           `SymbolicTensor`, unknown objects) → `core.TraceError` naming the
           pytree path. Capturing a concrete tensor as an input spec is
           NEVER silently allowed.
    3. Build an `ir.Module` + entry `ir.Function` ("main") with one block arg
       per tensor leaf (arg type = (shape, dtype) from the spec). Wrap each
       block arg as `core.SymbolicTensor(value=arg, dtype=spec.dtype,
       shape=DimExpr-of-spec.shape)` and record the `trace()` call-site
       `ir.Location` per input in `source_locations`.
    4. Reconstruct the argument structure: unflatten the input tree with
       `SymbolicTensor`s at tensor positions and the original static values
       at static positions.
    5. Call `fn(*args)` exactly ONCE under `with_builder(builder)`. Normal
       Python execution: static values behave like Python (control flow over
       them specializes the graph); tensor ops build IR via
       `current_builder()`. Closure-captured concrete tensors fail inside
       ops themselves (`core.TraceError` from ops — see the ops contract);
       the tracer does not pre-scan closures.
    6. Flatten the returned outputs (pytree). Leaves must be:
         - `core.SymbolicTensor` → graph result,
         - static Python value → recorded in `output_static_values`
           (re-inserted by `Graph.unflatten_outputs`), NOT emitted as a
           result,
         - anything else (incl. `TensorSpec`, concrete `Tensor`) →
           `core.TraceError` naming the path.
       Emit the function's `return` terminator with the SymbolicTensor
       leaves (a graph may legally return zero tensors if all outputs were
       static). Branch regions use the same return-op convention as
       `control_flow.py`.
    7. Return `Graph(module, input_specs, tensor_specs, output_tree,
       static_values, output_static_values, source_locations)`. The graph is
       NOT verified automatically (staging stays explicit); `etl.build`
       runs `graph.verify()` as part of its documented composition.

    Notes: zero specs → a zero-input graph (valid). Every call produces a NEW
    `Graph` (no caching). `**kwargs` is deliberately not part of the public
    signature.
    """
    # 1. Unwrap the graph-definition marker.
    fn = _unwrap_defn(fn_or_defn)

    # 2. Flatten + classify the specs tuple (ONE pytree).
    input_tree, tensor_specs, static_values = _flatten_specs(specs)

    # 3. IR: module + entry function with one block arg per tensor input.
    builder = ir.Builder()
    module = builder.build_module(name="main")
    input_types = tuple(
        ir.ValueType(spec.dtype, tuple(spec.shape)) for spec in tensor_specs
    )
    function = builder.build_function(name="main", input_types=input_types)
    block_args = function.entry_block.arguments

    location = _trace_call_site()
    symbolics = []
    source_locations = {}
    for block_arg, spec in zip(block_args, tensor_specs):
        symbolic = _spec_to_symbolic(block_arg, spec)
        symbolics.append(symbolic)
        source_locations[symbolic.id] = location

    # 4. Reconstruct the argument structure: SymbolicTensors at tensor
    #    positions, the original static values at static positions.
    static_by_index = {static.index: static.value for static in static_values}
    arg_leaves = []
    tensor_pos = 0
    for index in range(input_tree.num_leaves):
        if index in static_by_index:
            arg_leaves.append(static_by_index[index])
        else:
            arg_leaves.append(symbolics[tensor_pos])
            tensor_pos += 1
    args = core.unflatten(arg_leaves, input_tree)

    # 5. Run the function exactly ONCE under the active builder.
    with with_builder(builder):
        outputs = fn(*args)

    # 6. Flatten + classify the outputs; emit the return terminator.
    output_leaves, output_tree = _flatten_trace(outputs)
    result_values = []
    output_static_values = []
    for index, (leaf, path) in enumerate(
        zip(output_leaves, _iter_leaf_paths(output_tree))
    ):
        if isinstance(leaf, _SymbolicLeaf):
            result_values.append(leaf.symbolic.value)
        elif isinstance(leaf, _TensorSpecLeaf):
            raise core.TraceError(
                f"Invalid trace output at pytree path {_format_path(path)} "
                f"(traced at {location}): a core.TensorSpec cannot be "
                "returned from a traced function — TensorSpecs describe "
                "future runtime tensors (inputs); return the SymbolicTensor "
                "produced by tensor ops instead."
            )
        elif _is_static_value(leaf):
            output_static_values.append(
                StaticValue(
                    index=index,
                    path=path,
                    value=leaf,
                    kind=type(leaf).__qualname__,
                )
            )
        else:
            raise core.TraceError(
                f"Invalid trace output at pytree path {_format_path(path)} "
                f"(traced at {location}): got {leaf!r} of type "
                f"{type(leaf).__name__}. Graph outputs must be "
                "core.SymbolicTensor values (built by tensor ops) or static "
                "Python values (None/bool/int/float/complex/str/Enum/dtype/"
                "slice). There is no eager mode — concrete tensors (Tensor/"
                "numpy arrays) can never be returned from a traced function."
            )
    builder.set_terminator(
        builder.current_block, "return", operands=tuple(result_values)
    )

    # 7. Wrap into a Graph — NOT verified automatically (staging explicit).
    return Graph(
        module,
        input_tree,
        tensor_specs,
        output_tree,
        static_values,
        tuple(output_static_values),
        source_locations,
    )


def _unwrap_defn(fn_or_defn: Any) -> Any:
    """Return the underlying function for a `Defn` / `__etl_defn__`-marked
    object; return plain callables unchanged."""
    if isinstance(fn_or_defn, Defn):
        return fn_or_defn.fn
    if getattr(fn_or_defn, "__etl_defn__", False):
        fn = getattr(fn_or_defn, "fn", None)
        if fn is not None:
            return fn
    return fn_or_defn


def _is_static_value(obj: Any) -> bool:
    """True iff `obj` is a static Python value that specializes the graph.

    Accepted (per the root value-model contract): `None`, bool, int, float,
    complex, str, `enum.Enum`, numpy `dtype` objects, `slice`. Everything
    else (including numpy scalars, arbitrary config objects) is NOT static in
    v1 — `trace` raises `TraceError` for it. (Future extension point:
    explicit registration of static types, e.g. config objects.)
    """
    if obj is None:
        return True
    # bool must be checked before int (True is an int instance).
    if isinstance(obj, (bool, int, float, complex, str, slice, enum.Enum)):
        return True
    if isinstance(obj, np.dtype):
        return True
    return False


def _flatten_specs(
    specs: Tuple[Any, ...],
) -> Tuple["core.TreeSpec", Tuple["core.TensorSpec", ...], Tuple[StaticValue, ...]]:
    """Flatten + classify the trace inputs.

    Contract: treat `specs` (a tuple of positional arguments) as one pytree;
    each leaf is a `TensorSpec` (tensor input), a static value (specializes),
    or invalid → `core.TraceError` with the pytree path.

    Returns `(input_tree, tensor_specs, static_values)` where
    `input_tree`'s leaves hold the original leaf objects, `tensor_specs` is
    in flat leaf order, and each `StaticValue` records (flat index, path,
    value, type name).

    Implement in Phase 2 (pure tree walking over `core.TreeSpec`).
    """
    leaves, input_tree = _flatten_trace(specs)
    tensor_specs = []
    static_values = []
    for index, (leaf, path) in enumerate(zip(leaves, _iter_leaf_paths(input_tree))):
        if isinstance(leaf, _TensorSpecLeaf):
            tensor_specs.append(leaf.spec)
        elif isinstance(leaf, _SymbolicLeaf):
            raise core.TraceError(
                f"Invalid trace input at pytree path {_format_path(path)}: a "
                "core.SymbolicTensor cannot be passed as a trace input — "
                "SymbolicTensors are graph values created while tracing. "
                "Declare tensor inputs with core.TensorSpec(shape, dtype)."
            )
        elif _is_static_value(leaf):
            static_values.append(
                StaticValue(
                    index=index,
                    path=path,
                    value=leaf,
                    kind=type(leaf).__qualname__,
                )
            )
        else:
            raise core.TraceError(
                f"Invalid trace input at pytree path {_format_path(path)}: "
                f"{leaf!r} (type {type(leaf).__name__}) is neither a "
                "core.TensorSpec nor a static Python value. Tensor inputs "
                "must be declared as TensorSpec(shape, dtype); static values "
                "may be None/bool/int/float/complex/str/Enum/dtype/slice. "
                "Concrete tensors are never silently captured — declare them "
                "as explicit inputs via TensorSpec, or embed their data "
                "explicitly with etl.constant inside the traced function."
            )
    return input_tree, tuple(tensor_specs), tuple(static_values)


def _spec_to_symbolic(
    block_arg: "ir.Value", spec: "core.TensorSpec"
) -> "core.SymbolicTensor":
    """Wrap a function block arg as `core.SymbolicTensor` with the spec's
    dtype and symbolic shape (`Dim`/`DimExpr` from `spec.shape`; `None` dims
    stay dynamic). Implement in Phase 2."""
    shape = []
    for axis, entry in enumerate(spec.shape):
        if entry is None:
            # `SymbolicTensor.shape` (owned by core) accepts only Dim/
            # DimExpr/int entries, so a runtime-dynamic `None` dim is
            # represented as a unique, unknown-size Dim: it stays dynamic
            # (resolved only at run time) and, thanks to the fresh name,
            # never accidentally unifies with user-named dims. The IR block
            # arg keeps the true `None` (ir.ValueType) — this wrapper shape
            # is for trace-level inference only.
            shape.append(core.Dim(name=f"_dynamic_{block_arg.id}_{axis}"))
        else:
            shape.append(entry)
    return core.SymbolicTensor(
        value=block_arg, dtype=spec.dtype, shape=tuple(shape)
    )


def _iter_leaf_paths(
    tree_spec: "core.TreeSpec", prefix: Tuple[Any, ...] = ()
) -> Iterator[Tuple[Any, ...]]:
    """Yield the pytree key path of every leaf in pre-order.

    Matches `core.flatten`'s leaf order exactly (one path per leaf). Path
    entries are child indices, except `dict` nodes where the recorded sorted
    key (`node_data`) is used.
    """
    if tree_spec.num_leaves == 0:
        # Empty container (or a container of empty containers): no leaves.
        return
    if not tree_spec.children:
        yield prefix  # a leaf
        return
    for index, child in enumerate(tree_spec.children):
        if isinstance(tree_spec.type, type) and issubclass(tree_spec.type, dict):
            key = tree_spec.node_data[index]
        else:
            key = index
        yield from _iter_leaf_paths(child, prefix + (key,))


def _format_path(path: Tuple[Any, ...]) -> str:
    """Render a pytree path readably, e.g. ``[0]['weights'][1]``."""
    if not path:
        return "()"
    parts = []
    for key in path:
        if isinstance(key, str):
            parts.append(f"[{key!r}]")
        else:
            parts.append(f"[{key}]")
    return "".join(parts)


def _trace_call_site() -> "ir.Location":
    """The caller's source position of `trace()` (file + line); degrades to
    `ir.Location.unknown()` when no external frame exists.

    Skips every frame inside THIS module (the helper itself, `trace()`, and
    any future internal wrappers), so the captured position is the user's
    `trace()` call site. Never raises — a missing location must not break
    tracing.
    """
    try:
        frame = inspect.currentframe()
        try:
            this_file = os.path.abspath(__file__)
            while frame is not None and os.path.abspath(
                frame.f_code.co_filename
            ) == this_file:
                frame = frame.f_back
            if frame is None:
                return ir.Location.unknown()
            return ir.Location(
                file=frame.f_code.co_filename,
                line=frame.f_lineno,
                col=0,
                code_snippet=None,
            )
        finally:
            del frame  # drop the frame reference (avoid reference cycles)
    except Exception:
        return ir.Location.unknown()
