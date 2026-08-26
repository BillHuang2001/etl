"""`BlockOp` — the callable object produced by `etl.block(...)`.

A BlockOp carries the declaration (name, input/output specs, attribute
schema, effects, batching policy), builds a `block_call` IR op when called
inside a trace, and offers decorator methods to register implementations
and rules. The actual registration state lives in `registry.py`; the bridges
that publish rules into `etl.transforms` live in `rules.py`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

from . import registry
from . import rules
from .decl import AttributeField, BatchingPolicy, StaticValue
from .errors import BlockError

__all__ = ["BLOCK_CALL_OPDEF", "BlockOp"]

#: Attribute layout of the `block_call` IR op — coordinate with etl/ir.
#: The op def is registered in ir's op registry lazily by
#: `_ensure_ir_opdef` (Phase 2), so block is usable before any op is built.
#:
#: - operands: variadic ir Values, in `input_specs` order
#: - results: one per `output_spec`, typed from its dtype + dims
#:   (DimExpr dims stay symbolic; None dims are runtime-dynamic)
#: - attrs: `block_name` (str), `static` (dict name -> StaticValue payload),
#:   `effects` (ir effect kind), `batching_policy` (str)
#: - effect: taken from the `effects` attr
BLOCK_CALL_OPDEF: dict = {
    "name": "block_call",
    "summary": "custom operation declared via etl.block",
    "operands": "variadic",
    "results": "variadic",
    "attrs": ("block_name", "static", "effects", "batching_policy"),
    "effect_attr": "effects",
}


def _ensure_ir_opdef() -> None:
    """Register `BLOCK_CALL_OPDEF` in etl.ir's op registry (Phase 2).

    Lazily imports etl.ir (inside this function body, per the lazy-import
    constraint) and registers the op def if absent. Called by
    `BlockOp.__call__` before building the first op. The exact registration
    hook is finalized together with etl.ir at implementation time.
    """
    raise NotImplementedError("block: block_call IR op-def registration (Phase 2)")


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

        Rule contract (owned by etl.transforms): fn receives the symbolic
        arguments plus the mapped-axis metadata supplied by the vectorizer,
        and returns the transformed arguments and output batch axes — or a
        replacement graph. Returns `fn`.
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

        Semantics (Phase 2 implementation — this stub raises
        NotImplementedError):

        1. No active trace -> TraceError ("block_call outside a trace").
        2. Operand binding: positional args bind to `input_specs` slots in
           order. A positional arg that is not a SymbolicTensor is a static
           attribute value and binds to the next unfilled attribute in
           `attribute_schema` order. Keyword args must name declared
           attributes (BlockError otherwise). A concrete `Tensor` operand ->
           TraceError (no eager mode; make it an explicit input or use
           etl.constant). Python scalars are always static values here —
           block does NOT auto-promote scalars to tensors.
        3. Static values: missing required attribute -> BlockError; wrong
           static type -> BlockError (via validate_static_value); values are
           encoded with StaticValue.encode and recorded in the op's `static`
           attrs (op identity, cache keys, serialization).
        4. Operand validation: dtype mismatch -> DTypeError; shape mismatch
           (DimExpr unification against input_specs[i].shape) -> ShapeError.
        5. Op construction: `_ensure_ir_opdef()` then one `block_call` op
           through the active builder (trace.current_builder()), with attrs
           {block_name, static, effects, batching_policy} and result types
           from `output_specs`.
        6. Result: a single output_spec -> one SymbolicTensor; otherwise a
           tuple of SymbolicTensors.
        """
        raise NotImplementedError("block: BlockOp.__call__ (Phase 2)")

    def __repr__(self) -> str:
        return (
            f"BlockOp(name={self._name!r}, inputs={len(self._input_specs)}, "
            f"outputs={len(self._output_specs)}, effects={self._effects!r}, "
            f"batching={self._batching_policy.value!r})"
        )
