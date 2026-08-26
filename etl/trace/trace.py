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

import enum
from typing import Any, Tuple

import numpy as np

from etl import core
from etl import ir

from .builder import with_builder
from .defn import Defn
from .graph import Graph, StaticValue

__all__ = ["trace"]


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
    raise NotImplementedError(
        "etl.trace.trace: Phase 2 implementation — implement the algorithm "
        "in this docstring; see also ./CONTEXT.md."
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
    raise NotImplementedError(
        "etl.trace._flatten_specs: Phase 2 implementation — flatten the "
        "specs tuple via core.TreeSpec; classify leaves per "
        "`_is_static_value`; raise core.TraceError with path on invalid "
        "leaves (see trace() docstring step 2)."
    )


def _spec_to_symbolic(
    block_arg: "ir.Value", spec: "core.TensorSpec"
) -> "core.SymbolicTensor":
    """Wrap a function block arg as `core.SymbolicTensor` with the spec's
    dtype and symbolic shape (`Dim`/`DimExpr` from `spec.shape`; `None` dims
    stay dynamic). Implement in Phase 2."""
    raise NotImplementedError(
        "etl.trace._spec_to_symbolic: Phase 2 implementation — construct "
        "core.SymbolicTensor(value=block_arg, dtype=spec.dtype, "
        "shape=spec.shape as DimExprs)."
    )
