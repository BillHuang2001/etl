"""The tracer: `trace(fn_or_defn, *specs) -> Graph`.

Executes the user function ONCE under an active `ir.Builder`; every tensor op
encountered appends IR to the function body. Static Python values keep Python
semantics (normal `if`/loops over static values work — they specialize the
graph). Runtime tensor control flow must use `etl.cond` / `etl.while_loop` /
`etl.scan` (see `./control_flow.py`).

The step-by-step algorithm in `trace()`'s docstring is the BINDING contract;
`./CONTEXT.md` summarizes it. All trace-time state lives in one
`_TraceSession` object (see below) — there is no hidden global state besides
the active-builder context (`./builder.py`), which `_TraceSession.run`
installs only for the duration of the user-function call.

`_TensorSpecLeaf` / `_SymbolicLeaf` are deliberately NOT dataclasses: they
stand in for `TensorSpec`/`SymbolicTensor` at trace-tree leaf positions so
leaf-type equality tells tensor leaves apart from static leaves while the
TreeSpec stays `core.unflatten`-compatible. They are plain marker objects
carrying the original value.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
from typing import Any, Tuple

from etl import core
from etl import ir

from ._tree import (
    _flatten,
    _format_path,  # noqa: F401  (re-exported: etl.transforms.grad imports it)
    _is_static_value,
    _iter_leaf_paths,  # noqa: F401  (re-exported: etl.transforms.grad imports it)
)
from .builder import _return_terminator, with_builder
from .defn import Defn
from .graph import Graph, StaticValue, _static_record

__all__ = ["trace"]


class _TensorSpecLeaf:
    """Leaf marker standing for a tensor-INPUT position in a trace pytree.

    Trace trees record tensor-input positions with this plain (non-dataclass)
    marker so leaf-type equality tells tensor-input leaves apart from static
    leaves — keeping ``TreeSpec.num_leaves``, structural equality, and
    downstream ``core.unflatten`` coherent. (``core.flatten`` itself now
    treats etl-module dataclasses — ``TensorSpec`` included — as single
    leaves via its module check; the marker is still load-bearing for the
    tree skeleton comparisons in ``graph.py`` and for signature encoding.)
    The marker carries the original ``core.TensorSpec`` for
    classification/error messages.

    IMPORTANT: `etl/pipeline.py` imports this class by name
    (``from etl.trace.trace import _TensorSpecLeaf``) — keep the name and
    module path stable.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: "core.TensorSpec") -> None:
        self.spec = spec

    def __repr__(self) -> str:
        return f"_TensorSpecLeaf({self.spec!r})"


class _SymbolicLeaf:
    """Leaf marker standing for a symbolic RESULT position in a trace pytree.

    Same rationale as ``_TensorSpecLeaf`` for ``core.SymbolicTensor`` (also
    a frozen dataclass, now treated as a single leaf by ``core.flatten``'s
    module check): trace trees record symbolic-result positions with this
    plain marker so leaf-type equality distinguishes them from static output
    leaves. Carries the ``core.SymbolicTensor`` result.

    IMPORTANT: `etl/pipeline.py` imports this class by name
    (``from etl.trace.trace import _SymbolicLeaf``) — keep the name and
    module path stable.
    """

    __slots__ = ("symbolic",)

    def __init__(self, symbolic: "core.SymbolicTensor") -> None:
        self.symbolic = symbolic

    def __repr__(self) -> str:
        return f"_SymbolicLeaf({self.symbolic!r})"


def _trace_leaf_spec(obj: Any) -> "Optional[Tuple[Any, core.TreeSpec]]":
    """The trace-tree leaf policy for the shared walker (`./_tree.py`).

    `TensorSpec`/`SymbolicTensor` become typed marker leaves
    (`_TensorSpecLeaf`/`_SymbolicLeaf`); every other object defers to the
    walker's default leaf (``None`` policy result → container descent or
    plain leaf). The returned pair is ``(leaf_to_record, TreeSpec)`` — the
    walker appends the marker to the leaves list.
    """
    if isinstance(obj, core.TensorSpec):
        return (_TensorSpecLeaf(obj), core.TreeSpec(type=_TensorSpecLeaf))
    if isinstance(obj, core.SymbolicTensor):
        return (_SymbolicLeaf(obj), core.TreeSpec(type=_SymbolicLeaf))
    return None


def _flatten_trace(obj: Any) -> Tuple[list, "core.TreeSpec"]:
    """Flatten a pytree like ``core.flatten``, keeping ``TensorSpec`` and
    ``SymbolicTensor`` instances as leaves (via the markers above).

    Uses the SHARED walker in ``./_tree.py`` (same container rules as
    ``core.tree._flatten_into``: registered custom nodes, namedtuple,
    dataclass, tuple, list, dict with sorted keys) with the trace leaf
    policy. Fallback (static) leaves record their own Python type as the
    ``TreeSpec.type`` (``plain_leaf_type=type``) — pipeline's
    ``_is_tensor_leaf_spec`` relies on leaf types to tell static positions
    from tensor markers. The returned ``core.TreeSpec`` is fully
    ``core.unflatten``-compatible.
    """
    return _flatten(obj, _trace_leaf_spec, plain_leaf_type=type)


@dataclasses.dataclass
class _TraceSession:
    """The complete state of one `trace()` call — the single named owner of
    trace state while the user function runs.

    Ownership rules (what lives where):

    - ``builder`` / ``module`` / ``function``: the IR under construction.
      ONE builder serves the whole trace; control flow re-positions it via
      ``push_region``/``pop_region`` and never creates new builders.
    - ``input_tree`` / ``tensor_specs`` / ``static_values``: the classified
      input contract (built by `_flatten_specs`). ``input_tree``'s leaves
      hold the original leaf objects; ``tensor_specs`` is in flat leaf order
      == function block-arg order.
    - ``symbolics``: the block-arg ``SymbolicTensor``s, one per tensor spec.
    - ``source_locations``: maps each input symbolic's value id to the
      ``trace()`` call site (``call_site``), so error messages can name the
      user's trace line.
    - ``args``: the reconstructed call arguments passed to the user function
      (SymbolicTensors at tensor positions, original static values at static
      positions) — built by `_reconstruct_args`.

    Lifecycle: ``open(specs)`` performs trace algorithm steps 1-4 (builds the
    IR skeleton + classifies inputs + reconstructs args), ``run(fn)`` is step
    5 (the single user-function call under ``with_builder``), ``finish`` is
    steps 6-7 (classifies outputs, emits the ``return`` terminator, wraps the
    ``Graph``). One session per call — traces are never cached or reused.
    """

    builder: "ir.Builder"
    module: "ir.Module"
    function: "ir.Function"
    input_tree: "core.TreeSpec"
    tensor_specs: Tuple["core.TensorSpec", ...]
    static_values: Tuple[StaticValue, ...]
    args: Tuple[Any, ...]
    symbolics: list
    source_locations: dict
    call_site: "ir.Location"

    @classmethod
    def open(cls, specs: Tuple[Any, ...]) -> "_TraceSession":
        """Trace-algorithm steps 1-4: IR skeleton + classified inputs + args.

        1. (Unwrapping the ``Defn`` marker is `trace()`'s job — this is pure
           input handling.)
        2. Classify the specs tuple as one pytree (`_flatten_specs`).
        3. Build the module + entry function with one block arg per tensor
           input; wrap each block arg as a `SymbolicTensor` and record the
           `trace()` call-site location.
        4. Reconstruct the call arguments.
        """
        builder = ir.Builder()
        module = builder.build_module(name="main")
        input_tree, tensor_specs, static_values = _flatten_specs(specs)
        input_types = tuple(
            ir.ValueType(spec.dtype, tuple(spec.shape)) for spec in tensor_specs
        )
        function = builder.build_function(name="main", input_types=input_types)
        block_args = function.entry_block.arguments

        call_site = _trace_call_site()
        symbolics = []
        source_locations = {}
        for block_arg, spec in zip(block_args, tensor_specs):
            symbolic = _spec_to_symbolic(block_arg, spec)
            symbolics.append(symbolic)
            source_locations[symbolic.id] = call_site

        args = _reconstruct_args(input_tree, tensor_specs, static_values, symbolics)
        return cls(
            builder=builder,
            module=module,
            function=function,
            input_tree=input_tree,
            tensor_specs=tensor_specs,
            static_values=static_values,
            args=args,
            symbolics=symbolics,
            source_locations=source_locations,
            call_site=call_site,
        )

    def run(self, fn: Any) -> Any:
        """Trace-algorithm step 5: call `fn(*args)` exactly ONCE.

        Normal Python execution under the active builder: static values
        behave like Python (control flow over them specializes the graph);
        tensor ops build IR via `current_builder()`. Closure-captured
        concrete tensors fail inside ops themselves (`core.TraceError` from
        ops — see the ops contract); the tracer does not pre-scan closures.
        """
        with with_builder(self.builder):
            return fn(*self.args)

    def finish(self, outputs: Any) -> Graph:
        """Trace-algorithm steps 6-7: classify outputs, emit the `return`
        terminator, wrap the `Graph`.

        NOT verified automatically (staging stays explicit); `etl.build`
        runs `graph.verify()` as part of its documented composition.
        """
        result_values, output_static_values, output_tree = _classify_outputs(
            outputs, self.call_site
        )
        _return_terminator(self.builder, result_values)
        return Graph(
            self.module,
            self.input_tree,
            self.tensor_specs,
            output_tree,
            self.static_values,
            output_static_values,
            self.source_locations,
        )


def trace(fn_or_defn: Any, *specs: Any) -> Graph:
    """Trace `fn_or_defn` with the given input specs → `Graph`.

    ALGORITHM (binding):

    1. Unwrap: a `Defn` (or any object with `__etl_defn__`) yields its `fn`;
       plain callables are accepted as-is.
    2. Treat the `specs` tuple (the traced function's positional arguments)
       as one pytree. Flatten via the local `_flatten_trace` (the shared
       `./_tree.py` walker with the trace leaf policy — mirrors
       `core.flatten`'s container rules; `TensorSpec`/`SymbolicTensor`
       become typed marker leaves). Every leaf must be either:
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

    The whole trace state lives in one `_TraceSession` (steps 2-4 = `open`,
    step 5 = `run`, steps 6-7 = `finish`).

    Notes: zero specs → a zero-input graph (valid). Every call produces a NEW
    `Graph` (no caching). `**kwargs` is deliberately not part of the public
    signature.
    """
    fn = _unwrap_defn(fn_or_defn)
    session = _TraceSession.open(specs)
    outputs = session.run(fn)
    return session.finish(outputs)


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


def _flatten_specs(
    specs: Tuple[Any, ...],
) -> Tuple["core.TreeSpec", Tuple["core.TensorSpec", ...], Tuple[StaticValue, ...]]:
    """Flatten + classify the trace inputs (trace-algorithm step 2).

    Contract: treat `specs` (a tuple of positional arguments) as one pytree;
    each leaf is a `TensorSpec` (tensor input), a static value (specializes),
    or invalid → `core.TraceError` with the pytree path.

    Returns `(input_tree, tensor_specs, static_values)` where
    `input_tree`'s leaves hold the original leaf objects, `tensor_specs` is
    in flat leaf order, and each `StaticValue` records (flat index, path,
    value, type name).
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
            static_values.append(_static_record(index, path, leaf))
        else:
            raise core.TraceError(
                f"Invalid trace input at pytree path {_format_path(path)}: "
                f"{leaf!r} (type {type(leaf).__name__}) is neither a "
                "core.TensorSpec nor a static Python value. Tensor inputs "
                "must be declared as TensorSpec(shape, dtype); static values "
                "may be None/bool/int/float/complex/str/Enum/dtype/slice/"
                "Dim/DimExpr/Device. "
                "Concrete tensors are never silently captured — declare them "
                "as explicit inputs via TensorSpec, or embed their data "
                "explicitly with etl.constant inside the traced function."
            )
    return input_tree, tuple(tensor_specs), tuple(static_values)


def _reconstruct_args(
    input_tree: "core.TreeSpec",
    tensor_specs: Tuple["core.TensorSpec", ...],
    static_values: Tuple[StaticValue, ...],
    symbolics: list,
) -> Tuple[Any, ...]:
    """Trace-algorithm step 4: rebuild the call arguments.

    Unflatten the input tree with `SymbolicTensor`s at tensor positions
    (consumed in `tensor_specs` order) and the original static values at the
    recorded static indices.
    """
    static_by_index = {static.index: static.value for static in static_values}
    arg_leaves = []
    tensor_pos = 0
    for index in range(input_tree.num_leaves):
        if index in static_by_index:
            arg_leaves.append(static_by_index[index])
        else:
            arg_leaves.append(symbolics[tensor_pos])
            tensor_pos += 1
    return core.unflatten(arg_leaves, input_tree)


def _classify_outputs(
    outputs: Any, call_site: "ir.Location"
) -> Tuple[tuple, tuple, "core.TreeSpec"]:
    """Trace-algorithm step 6: flatten + classify the traced function's
    outputs.

    Returns `(result_values, output_static_values, output_tree)`:
    `result_values` are the flat `ir.Value`s for the `return` terminator
    (SymbolicTensor leaves only), `output_static_values` are the recorded
    static leaves (re-inserted by `Graph.unflatten_outputs`), and
    `output_tree` is the output TreeSpec. Anything that is neither a
    SymbolicTensor nor a static value → `core.TraceError` naming the path.
    """
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
                f"(traced at {call_site}): a core.TensorSpec cannot be "
                "returned from a traced function — TensorSpecs describe "
                "future runtime tensors (inputs); return the SymbolicTensor "
                "produced by tensor ops instead."
            )
        elif _is_static_value(leaf):
            output_static_values.append(_static_record(index, path, leaf))
        else:
            raise core.TraceError(
                f"Invalid trace output at pytree path {_format_path(path)} "
                f"(traced at {call_site}): got {leaf!r} of type "
                f"{type(leaf).__name__}. Graph outputs must be "
                "core.SymbolicTensor values (built by tensor ops) or static "
                "Python values (None/bool/int/float/complex/str/Enum/dtype/"
                "slice). There is no eager mode — concrete tensors (Tensor/"
                "numpy arrays) can never be returned from a traced function."
            )
    return tuple(result_values), tuple(output_static_values), output_tree


def _spec_to_symbolic(
    block_arg: "ir.Value", spec: "core.TensorSpec"
) -> "core.SymbolicTensor":
    """Wrap a function block arg as `core.SymbolicTensor` with the spec's
    dtype and symbolic shape (`Dim`/`DimExpr` from `spec.shape`; `None` dims
    stay dynamic)."""
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
