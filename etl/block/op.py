"""`BlockOp` — the callable object produced by `etl.block(...)`.

A BlockOp carries the declaration (name, input/output specs, attribute
schema, effects, batching policy), builds a `block_call` IR op when called
inside a trace, and offers decorator methods to register implementations
and rules. The actual registration state lives in `registry.py`; the bridges
that publish rules into `etl.transforms` live in `rules.py`.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Mapping, Optional, Tuple

from . import registry
from . import rules
from .decl import AttributeField, BatchingPolicy, StaticValue, validate_static_value
from .errors import BlockError

__all__ = ["BLOCK_CALL_OPDEF", "BlockOp"]

#: Attribute layout of the `block_call` IR op — the CANONICAL op definition
#: lives in etl/ir/op_defs/control.py (ir owns op defs; block only verifies
#: it via `_ensure_ir_opdef` and never registers a conflicting def):
#:
#: - operands: variadic ir Values, in `input_specs` order
#: - results: one per `output_spec`, typed from its dtype + dims
#:   (DimExpr dims stay symbolic; None dims are runtime-dynamic)
#: - attrs: `block_name` (str), `static_args` (dict name -> JSON-safe
#:   {"kind", "value"} payload), `result_specs` (tuple of ir.ValueType)
#: - effect: fixed at `read` (the declared block's actual effects are not
#:   yet reflected on the op — ir-side gap; see block/CONTEXT.md)
BLOCK_CALL_OPDEF: dict = {
    "name": "block_call",
    "summary": "custom operation declared via etl.block",
    "operands": "variadic",
    "results": "variadic",
    "attrs": ("block_name", "static_args", "result_specs"),
    "effect": "read",
}

#: AttrSpec names `block_call` must declare (mirrors ir's schema).
_REQUIRED_OPDEF_ATTRS = frozenset({"block_name", "static_args", "result_specs"})

#: Root directory of the ``etl`` package (realpath). Frames whose filename
#: lives under this directory are internal helper frames and never produce
#: user-facing locations (see :func:`_get_location`).
_ETL_ROOT = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: Whether `_ensure_ir_opdef` has passed once (cheap idempotent check).
_OPDEF_CHECKED = False


def _ensure_ir_opdef() -> None:
    """Verify that etl.ir registered the canonical `block_call` op def.

    ir owns the op-definition registry (binding: `etl/ir/op_defs/control.py`
    declares `block_call`), so block only CHECKS the def — it never registers
    anything. Lazily imports etl.ir (inside this function body, per the
    lazy-import constraint) and is called by `BlockOp.__call__` before the
    first op is built. The check is cached at module level (idempotent).

    Raises:
        BlockError: The def is missing or its attribute schema does not
            match what this module builds against.
    """
    global _OPDEF_CHECKED
    if _OPDEF_CHECKED:
        return
    from etl import ir

    try:
        opdef = ir.opdef("block_call")
    except KeyError as exc:
        raise BlockError(
            "ir op definition 'block_call' is missing: it is owned by "
            "etl/ir/op_defs/control.py (block must not register a "
            "conflicting def) — fix the ir registry there"
        ) from exc
    attr_names = frozenset(spec.name for spec in opdef.attributes)
    missing = sorted(_REQUIRED_OPDEF_ATTRS - attr_names)
    if missing:
        raise BlockError(
            f"ir op definition 'block_call' does not match this module's "
            f"contract: missing attribute(s) {missing} (schema: "
            f"{sorted(attr_names)!r}) — fix etl/ir/op_defs/control.py"
        )
    _OPDEF_CHECKED = True


def _get_location() -> Any:
    """Capture the Python call site of the user's block call.

    Walks ``inspect.stack()``, skipping every frame whose filename lives
    inside the ``etl`` package directory (so internal helper frames never
    pollute user locations), and returns ``ir.Location(file=..., line=...,
    col=0)`` for the first external frame. Never raises — location capture
    failure degrades to ``ir.Location.unknown()`` (a missing location must
    not break tracing).
    """
    try:
        for frame in inspect.stack()[1:]:
            filename = frame.filename
            if not filename or filename.startswith("<"):
                continue  # no real source path (builtins, frozen modules)
            try:
                inside = (
                    os.path.commonpath(
                        [_ETL_ROOT, os.path.realpath(filename)]
                    )
                    == _ETL_ROOT
                )
            except ValueError:
                inside = False  # different drives (Windows) — never inside
            if inside:
                continue  # internal helper frame
            from etl import ir

            return ir.Location(file=filename, line=frame.lineno, col=0)
    except Exception:
        # Contract: location capture must never break tracing — degrade.
        pass
    from etl import ir

    return ir.Location.unknown()


def _static_payload(value: Any) -> dict:
    """Encode one static attribute value as the JSON-safe wire payload.

    ``StaticValue.encode`` rejects unsupported types (BlockError); the
    payload is a plain dict ``{"kind": ..., "value": ...}`` (never a
    StaticValue object) so ir's ATTR_ANY serialization round-trips it.
    """
    sv = StaticValue.encode(value)
    return {"kind": sv.kind, "value": sv.value}


class BlockOp:
    """A declared custom operation, callable to build `block_call` IR ops.

    Constructed only through `decl.block(...)` (the factory), which validates
    everything and registers the op. Do not instantiate directly.
    """

    __slots__ = (
        "_name",
        "_input_specs",
        "_output_specs",
        "_attribute_schema",
        "_effects",
        "_batching_policy",
    )

    def __init__(
        self,
        *,
        name: str,
        input_specs: Tuple[Any, ...],
        output_specs: Tuple[Any, ...],
        attribute_schema: Mapping[str, AttributeField],
        effects: str,
        batching_policy: BatchingPolicy,
    ) -> None:
        self._name = name
        self._input_specs = tuple(input_specs)
        self._output_specs = tuple(output_specs)
        self._attribute_schema = dict(attribute_schema)
        self._effects = effects
        self._batching_policy = (
            batching_policy
            if isinstance(batching_policy, BatchingPolicy)
            else BatchingPolicy(batching_policy)
        )

    # ------------------------------------------------------------------ attrs

    @property
    def name(self) -> str:
        """Stable operation name (globally unique, vendor namespaces via dots)."""
        return self._name

    @property
    def input_specs(self) -> Tuple[Any, ...]:
        """TensorSpecs of the symbolic tensor operands, in order."""
        return self._input_specs

    @property
    def output_specs(self) -> Tuple[Any, ...]:
        """TensorSpecs of the op results, in order."""
        return self._output_specs

    @property
    def attribute_schema(self) -> Mapping[str, AttributeField]:
        """Declared static attributes (name -> AttributeField)."""
        return self._attribute_schema

    @property
    def attributes(self) -> Mapping[str, AttributeField]:
        """Alias for `attribute_schema` (the name used in etl/CONTEXT.md)."""
        return self._attribute_schema

    @property
    def effects(self) -> str:
        """One of ir's effect kinds: pure | write | read | collective | callback."""
        return self._effects

    @property
    def batching_policy(self) -> BatchingPolicy:
        """How vectorize/vmap may treat this op when no explicit rule exists."""
        return self._batching_policy

    @property
    def has_portable(self) -> bool:
        """True if a portable (etl.defn) implementation is registered."""
        return registry.get_portable(self._name) is not None

    def get_impl(self, backend_name: str) -> Optional[Callable]:
        """The registered implementation for `backend_name`, or None."""
        return registry.get_impl(self._name, backend_name)

    # ------------------------------------------------------------- registration

    def portable(self, fn: Callable) -> Callable:
        """Register `fn` (an `etl.defn` function) as the portable implementation.

        The portable graph is inlined at lower time or by transforms (as the
        batching/derivative fallback) when a backend cannot consume the
        block_call natively. Returns `fn` (decorator semantics).
        """
        registry.register_portable(self._name, fn)
        return fn

    def impl(self, backend_name: str) -> Callable[[Callable], Callable]:
        """Decorator registering a backend-specific implementation.

        Usage::

            @flash_attention.impl("numpy")
            def numpy_flash_attention(*args):
                ...

        v1 target: "numpy" impls (consumed by the numpy interpreter backend).
        The registry is backend-neutral: IREE/XLA custom-call names can be
        registered the same way once those adapters exist. Returns `fn`.
        """
        if not isinstance(backend_name, str) or not backend_name:
            raise BlockError(
                f"backend name must be a non-empty string, got {backend_name!r}"
            )

        def decorator(fn: Callable) -> Callable:
            registry.register_impl(self._name, backend_name, fn)
            return fn

        return decorator

    def batching_rule(self, fn: Callable) -> Callable:
        """Register a vectorize/vmap batching rule under `block:<name>`.

        Rule contract (owned by etl.transforms): `fn(op, operands, axes) ->
        (new_values, new_axes)` — `operands` is the tuple of `ir.Value`
        operands, `axes` the aligned `MappedAxes` metadata; `new_values` is
        aligned with `op.results`, `new_axes` with `new_values`. Returns `fn`.
        """
        rules.register_batching_rule(self._name, fn)
        return fn

    def jvp_rule(self, fn: Callable) -> Callable:
        """Register a forward-mode derivative rule under `block:<name>`.

        Stored in `transforms.jvp_rules` (coordination with etl/transforms;
        when absent, transforms derives jvp from the vjp rule). Returns `fn`.
        """
        rules.register_jvp_rule(self._name, fn)
        return fn

    def vjp_rule(self, fn: Callable) -> Callable:
        """Register a reverse-mode derivative rule under `block:<name>`.

        Stored in `transforms.vjp_rules` per the etl/CONTEXT.md contract.
        Returns `fn`.
        """
        rules.register_vjp_rule(self._name, fn)
        return fn

    # ---------------------------------------------------------------- calling

    def __call__(self, *operands: Any, **attributes: Any) -> Any:
        """Build a `block_call` IR op for this custom operation.

        Semantics:

        1. `_ensure_ir_opdef()` then the active builder via
           `trace.current_builder()` — no active trace -> TraceError (the
           canonical message).
        2. Operand binding: positional args bind to `input_specs` slots in
           order while they are `SymbolicTensor`s. A positional arg that is
           not a SymbolicTensor is a static attribute value and binds to the
           next unfilled attribute in `attribute_schema` order (schema order,
           not call order; an attribute already given by keyword must not be
           double-filled — BlockError). Keyword args must name declared
           attributes (BlockError otherwise). A concrete `Tensor` positional
           -> TraceError (no eager mode; make it an explicit input or use
           etl.constant). Python scalars are always static values here —
           block does NOT auto-promote scalars to tensors.
        3. Static values: missing required attribute -> BlockError; wrong
           static type -> BlockError (via validate_static_value); values are
           encoded with StaticValue.encode and recorded in the op's
           `static_args` attr as JSON-safe `{"kind", "value"}` payloads (op
           identity, cache keys, serialization).
        4. Operand validation: dtype mismatch -> DTypeError; shape mismatch
           (structural DimExpr equality against input_specs[i].shape; None
           spec dims are runtime-dynamic wildcards) -> ShapeError.
        5. Op construction: one `block_call` op through the active builder,
           with attrs {block_name, static_args, result_specs} (result types
           resolve from `result_specs` ir.ValueTypes).
        6. Result: a single output_spec -> one SymbolicTensor; otherwise a
           tuple of SymbolicTensors.
        """
        _ensure_ir_opdef()
        from etl import core, ir
        from etl.trace import current_builder

        # 1. Active trace — trace.current_builder() raises the canonical
        #    core.TraceError when no builder is active (never duplicated here).
        builder = current_builder()
        loc = _get_location()

        # 2. Operand / static-attribute binding.
        schema_items = tuple(self._attribute_schema.items())
        keyword_names = frozenset(attributes)
        for key in attributes:
            if key not in self._attribute_schema:
                raise BlockError(
                    f"block '{self._name}': undeclared attribute {key!r} — "
                    f"declared attributes: {sorted(self._attribute_schema)}"
                )

        symbolic: list = []
        positional_statics: dict = {}
        for pos, arg in enumerate(operands):
            if isinstance(arg, core.SymbolicTensor):
                if len(symbolic) >= len(self._input_specs):
                    raise BlockError(
                        f"block '{self._name}': too many tensor operands — "
                        f"input_specs declares {len(self._input_specs)}, got "
                        f"a tensor at positional argument #{pos}"
                    )
                symbolic.append(arg)
                continue
            if isinstance(arg, core.Tensor):
                raise core.TraceError(
                    "Concrete Tensor operands are not allowed in graph ops "
                    "(etl has no eager mode): (1) pass the tensor as an "
                    "explicit input (TensorSpec at trace time), (2) embed "
                    "its data explicitly with etl.constant (snapshots the "
                    "data and warns for large tensors), or (3) build and run "
                    "a graph with etl.evaluate."
                )
            # Python value -> static attribute: the NEXT unfilled attribute
            # in SCHEMA order (not call order). A slot already supplied by
            # keyword must not be double-filled by a positional.
            field = None
            for attr_name, candidate in schema_items:
                if attr_name in positional_statics:
                    continue
                if attr_name in keyword_names:
                    raise BlockError(
                        f"block '{self._name}': positional argument #{pos} "
                        f"(static value {arg!r}) would bind to attribute "
                        f"'{attr_name}' which was already given by keyword"
                    )
                field = candidate
                break
            if field is None:
                raise BlockError(
                    f"block '{self._name}': too many positional arguments — "
                    f"positional #{pos} is a static value "
                    f"({type(arg).__name__}) but all declared attributes are "
                    f"already bound"
                )
            positional_statics[field.name] = arg

        if len(symbolic) != len(self._input_specs):
            raise BlockError(
                f"block '{self._name}': expected {len(self._input_specs)} "
                f"tensor operand(s) matching input_specs, got "
                f"{len(symbolic)} symbolic operand(s)"
            )

        # 3. Static values: validate, encode, and record as JSON-safe
        #    {"kind", "value"} payloads (never StaticValue objects — the
        #    payload must survive ir's ATTR_ANY serialization).
        static_payload: dict = {}
        for attr_name, field in schema_items:
            if attr_name in positional_statics:
                value = positional_statics[attr_name]
            elif attr_name in attributes:
                value = attributes[attr_name]
            elif field.required:
                raise BlockError(
                    f"block '{self._name}': missing required attribute "
                    f"'{attr_name}' ({field.type.__name__})"
                )
            else:
                continue  # optional attribute: default is fixed by the schema
            validate_static_value(field, value)
            static_payload[attr_name] = _static_payload(value)

        # 4. Operand validation against input_specs.
        for i, (spec, operand) in enumerate(zip(self._input_specs, symbolic)):
            if operand.dtype != spec.dtype:
                raise core.DTypeError(
                    f"block '{self._name}': operand {i} dtype mismatch — "
                    f"expected {spec.dtype}, got {operand.dtype}"
                )
            got_shape = tuple(operand.shape)
            if len(got_shape) != len(spec.shape):
                raise core.ShapeError(
                    f"block '{self._name}': operand {i} rank mismatch — "
                    f"expected rank {len(spec.shape)} (shape "
                    f"{spec.shape!r}), got rank {len(got_shape)} (shape "
                    f"{got_shape!r})"
                )
            for d, (expected, got) in enumerate(zip(spec.shape, got_shape)):
                if expected is None:
                    continue  # runtime-dynamic wildcard: unchecked
                if expected != got:
                    raise core.ShapeError(
                        f"block '{self._name}': operand {i} shape mismatch "
                        f"at dim {d} — expected {expected!r}, got {got!r}"
                    )

        # 5. Op construction: result types resolve from `result_specs`
        #    (ir.verify requires the ValueType entries to equal the op's
        #    result types exactly).
        result_specs = tuple(
            ir.ValueType(dtype=spec.dtype, shape=tuple(spec.shape))
            for spec in self._output_specs
        )
        op = builder.create(
            "block_call",
            operands=tuple(sym.value for sym in symbolic),
            attributes={
                "block_name": self._name,
                "static_args": static_payload,
                "result_specs": result_specs,
            },
            location=loc,
        )

        # 6. Result wrapping.
        results = tuple(
            core.SymbolicTensor(
                value=value,
                dtype=value.type.dtype,
                shape=value.type.shape,
                location=loc,
            )
            for value in op.results
        )
        return results[0] if len(self._output_specs) == 1 else results

    def __repr__(self) -> str:
        return (
            f"BlockOp(name={self._name!r}, inputs={len(self._input_specs)}, "
            f"outputs={len(self._output_specs)}, effects={self._effects!r}, "
            f"batching={self._batching_policy.value!r})"
        )
