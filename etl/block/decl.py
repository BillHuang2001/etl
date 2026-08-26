"""Declaration API for custom operations (`etl.block`).

This module owns the `block()` factory and everything needed to *declare* a
custom operation: name validation, spec normalization, the attribute schema,
effect-kind validation, batching-policy resolution, and the `StaticValue`
tag used to serialize static attribute values onto `block_call` IR ops.

`BlockOp` (call semantics, rule registration) lives in `op.py`; the block
and implementation registries live in `registry.py`; the bridges into the
`etl.transforms` rule registries live in `rules.py`. Full design: CONTEXT.md.
"""

from __future__ import annotations

import enum
import importlib
import inspect
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from etl.core import TensorSpec

from .errors import BlockError

__all__ = [
    "AttributeField",
    "BatchingPolicy",
    "StaticValue",
    "block",
    "validate_static_value",
]

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

#: The five effect kinds owned by etl.ir (see ./CONTEXT.md and the ir module);
#: validated here so declarations fail early, before any IR is built.
_EFFECT_KINDS = ("callback", "collective", "pure", "read", "write")

_MISSING = object()


class BatchingPolicy(str, enum.Enum):
    """How `vectorize`/`vmap` may treat a `block_call` op.

    The policy describes what happens when NO explicit batching rule is
    registered (a registered rule always wins). The default (batching=None)
    resolves to `BATCHING_RULE` when a portable implementation exists — block
    pre-registers the decomposition as the namespaced rule — and to
    `UNSUPPORTED` otherwise. See CONTEXT.md for full semantics.
    """

    ELEMENTWISE = "elementwise"
    """Op acts independently per element over all dims (like relu/sin).

    Batch dims pass through unchanged; vectorize/vmap is safe without a rule.
    """

    BATCHING_RULE = "batching_rule"
    """Batching driven by a registered rule (or by the default fallback:
    inlining the portable decomposition)."""

    BROADCAST_BATCH = "broadcast_batch"
    """Op broadcasts batch dims among operands (e.g. batched-matmul-style
    broadcasting); batch axes may be introduced or merged. Requires a rule
    or decomposition describing how batch dims combine."""

    MAP_OVER_BATCH = "map_over_batch"
    """Op already maps over its leading batch dims internally (batch dims are
    part of the op's own semantics). Transforms pass batch dims through
    untouched — safe without a rule."""

    OPAQUE_BATCHED = "opaque_batched"
    """Op consumes batch dims opaquely (e.g. flash_attention over the full
    batch). Vectorizing over further dims is unsafe: TransformError unless an
    explicit batching rule is registered."""

    UNSUPPORTED = "unsupported"
    """No safe automatic batching path. vectorize/vmap raise TransformError
    unless an explicit batching rule is registered (never guess)."""


@dataclass(frozen=True)
class AttributeField:
    """One declared static attribute of a BlockOp.

    Declared in `attributes=` as either a bare type (required attribute) or a
    default value (type inferred; optional). Static values specialize the
    block_call op: they are recorded as op attributes and participate in
    cache keys and serialization.
    """

    name: str
    type: type
    default: Any = _MISSING

    @property
    def required(self) -> bool:
        return self.default is _MISSING


@dataclass(frozen=True)
class StaticValue:
    """Serialized static (Python) value recorded on a `block_call` op.

    Static attribute values specialize the op exactly like static values
    specialize a traced graph: they participate in op identity, cache keys,
    and artifact serialization, so they must round-trip through JSON-safe
    payloads.

    `kind` is one of: ``none, bool, int, float, complex, str, slice, dtype,
    enum``. Anything else is rejected at call time (BlockError) — no silent
    pickling or opaque bytes.
    """

    kind: str
    value: Any

    @classmethod
    def encode(cls, v: Any) -> "StaticValue":
        if v is None:
            return cls("none", None)
        if isinstance(v, bool):
            return cls("bool", v)
        if isinstance(v, int):
            return cls("int", v)
        if isinstance(v, float):
            return cls("float", v)
        if isinstance(v, complex):
            return cls("complex", (v.real, v.imag))
        if isinstance(v, str):
            return cls("str", v)
        if isinstance(v, slice):
            return cls("slice", (v.start, v.stop, v.step))
        if isinstance(v, np.dtype):
            return cls("dtype", v.name)
        if isinstance(v, Enum):
            return cls("enum", f"{type(v).__module__}.{type(v).__qualname__}.{v.name}")
        raise BlockError(
            f"unsupported static attribute value {v!r} (type "
            f"{type(v).__name__}): static values must be None, bool, int, "
            f"float, complex, str, slice, dtype, or Enum"
        )

    def decode(self) -> Any:
        if self.kind == "none":
            return None
        if self.kind == "bool":
            return bool(self.value)
        if self.kind == "int":
            return int(self.value)
        if self.kind == "float":
            return float(self.value)
        if self.kind == "complex":
            re_, im_ = self.value
            return complex(re_, im_)
        if self.kind == "str":
            return str(self.value)
        if self.kind == "slice":
            start, stop, step = self.value
            return slice(start, stop, step)
        if self.kind == "dtype":
            return np.dtype(self.value)
        if self.kind == "enum":
            module_name, _, member_path = self.value.rpartition(".")
            obj: Any = importlib.import_module(module_name)
            for part in member_path.split("."):
                obj = getattr(obj, part)
            return obj
        raise BlockError(f"corrupt StaticValue: unknown kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Declaration helpers (validation & normalization — trivial, implemented)
# ---------------------------------------------------------------------------


def _is_defn(fn: Any) -> bool:
    """True if `fn` carries the `etl.defn` marker (a Defn object)."""
    return callable(fn) and getattr(fn, "__etl_defn__", None) is not None


def _name_of(fn: Any) -> str:
    name = getattr(fn, "__name__", None)
    if not name:
        inner = getattr(fn, "fn", None)
        name = getattr(inner, "__name__", None) if inner is not None else None
    if not name:
        raise BlockError(
            "cannot derive a block name from the decorated function — pass "
            "an explicit name to the factory form of etl.block(...)"
        )
    return name


def _validate_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise BlockError(f"block name must be a non-empty string, got {name!r}")
    if not _NAME_RE.match(name):
        raise BlockError(
            f"invalid block name {name!r}: must match [A-Za-z_][A-Za-z0-9_.]* "
            f"(dots allowed for vendor namespaces, e.g. 'myorg.flash_attention')"
        )


def _normalize_specs(specs: Any, kind: str, name: str) -> Tuple[TensorSpec, ...]:
    if specs is None:
        return ()
    if isinstance(specs, TensorSpec):
        specs = (specs,)
    if not isinstance(specs, (list, tuple)) or not all(
        isinstance(s, TensorSpec) for s in specs
    ):
        raise BlockError(
            f"block '{name}': {kind} must be a TensorSpec or a list/tuple of "
            f"TensorSpec, got {specs!r}"
        )
    return tuple(specs)


def _normalize_attribute_schema(attributes: Any) -> Dict[str, AttributeField]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise BlockError(
            f"attributes must be a mapping of name -> type-or-default, got "
            f"{attributes!r}"
        )
    schema: Dict[str, AttributeField] = {}
    for attr_name, decl in attributes.items():
        if not isinstance(attr_name, str) or not attr_name.isidentifier():
            raise BlockError(
                f"invalid attribute name {attr_name!r}: must be a valid identifier"
            )
        if isinstance(decl, type):
            schema[attr_name] = AttributeField(name=attr_name, type=decl)
        else:
            # Default value: type inferred from the value; attribute optional.
            schema[attr_name] = AttributeField(
                name=attr_name, type=type(decl), default=decl
            )
    return schema


def _normalize_effects(effects: Any) -> str:
    if not isinstance(effects, str) or effects not in _EFFECT_KINDS:
        raise BlockError(
            f"effects must be one of {', '.join(repr(k) for k in _EFFECT_KINDS)}, "
            f"got {effects!r}"
        )
    return effects


def _normalize_policy(batching: Any, has_portable: bool) -> BatchingPolicy:
    if batching is None:
        # Default: portable decomposition exists -> batch by decomposing
        # (block pre-registers the decomposition as the namespaced rule);
        # otherwise there is no safe path.
        return (
            BatchingPolicy.BATCHING_RULE if has_portable else BatchingPolicy.UNSUPPORTED
        )
    if isinstance(batching, BatchingPolicy):
        return batching
    if isinstance(batching, str):
        try:
            return BatchingPolicy(batching)
        except ValueError:
            pass
    raise BlockError(
        f"batching must be one of {', '.join(repr(p.value) for p in BatchingPolicy)}, "
        f"got {batching!r}"
    )


def validate_static_value(field: AttributeField, value: Any) -> None:
    """Trivial type check for a static attribute value.

    Called by `BlockOp.__call__` (Phase 2) before encoding. Note that
    isinstance-based checking treats `True` as an `int` — acceptable for
    static attribute typing.
    """
    if not isinstance(value, field.type):
        raise BlockError(
            f"attribute '{field.name}' expects {field.type.__name__}, got "
            f"{type(value).__name__} ({value!r})"
        )


def _collect_specs(node: Any, out: List[TensorSpec], where: str) -> None:
    """Flatten a TensorSpec or a (tuple/list/dict) pytree of TensorSpec.

    v1 flattens simple nestings here; arbitrary pytrees via core.TreeSpec are
    a Phase 2 refinement.
    """
    if isinstance(node, TensorSpec):
        out.append(node)
    elif isinstance(node, (tuple, list)):
        for item in node:
            _collect_specs(item, out, where)
    elif isinstance(node, dict):
        for item in node.values():
            _collect_specs(item, out, where)
    else:
        raise BlockError(
            f"{where}: expected a TensorSpec or a pytree of TensorSpec, got {node!r}"
        )


def _derive_input_specs(fn: Any, name: str) -> Tuple[TensorSpec, ...]:
    """Derive input_specs from a defn's TensorSpec annotations/defaults."""
    inner = getattr(fn, "fn", fn) if _is_defn(fn) else fn
    specs: List[TensorSpec] = []
    for param in inspect.signature(inner).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise BlockError(
                f"block '{name}': cannot derive input specs from a function "
                f"with *args/**kwargs — pass inputs= explicitly"
            )
        spec = None
        if isinstance(param.default, TensorSpec):
            spec = param.default
        elif isinstance(param.annotation, TensorSpec):
            spec = param.annotation
        if spec is None:
            raise BlockError(
                f"block '{name}': cannot derive an input spec for parameter "
                f"'{param.name}' — annotate it with a TensorSpec or pass "
                f"inputs= explicitly"
            )
        specs.append(spec)
    return tuple(specs)


def _derive_output_specs(fn: Any, name: str) -> Tuple[TensorSpec, ...]:
    """Derive output_specs from a defn's return annotation (TensorSpec pytree)."""
    inner = getattr(fn, "fn", fn) if _is_defn(fn) else fn
    ann = inspect.signature(inner).return_annotation
    if ann is inspect.Signature.empty or ann is None:
        raise BlockError(
            f"block '{name}': declare outputs= (or result=) — the function "
            f"has no return annotation to derive output specs from"
        )
    flat: List[TensorSpec] = []
    _collect_specs(ann, flat, f"block '{name}' return annotation")
    return tuple(flat)


def _make_block(
    name: str,
    inputs: Any,
    outputs: Any,
    attributes: Any,
    effects: Any,
    batching: Any,
    portable: Optional[Callable],
) -> Any:
    # Lazy same-package imports keep decl.py importable without cycles
    # (op.py and rules.py import decl at module level).
    from .op import BlockOp
    from .registry import register, register_portable, validate_portable
    from .rules import register_portable_batching_fallback, register_portable_diff_fallback

    _validate_name(name)
    in_specs = _normalize_specs(inputs, "inputs", name)
    out_specs = _normalize_specs(outputs, "outputs", name)
    if not in_specs and portable is None:
        raise BlockError(
            f"block '{name}': no tensor inputs and no portable implementation "
            f"— nothing to call; declare inputs= or a portable defn"
        )
    if portable is not None:
        validate_portable(name, portable)  # raises before anything is registered
    schema = _normalize_attribute_schema(attributes)
    effect = _normalize_effects(effects)
    policy = _normalize_policy(batching, has_portable=portable is not None)

    op = BlockOp(
        name=name,
        input_specs=in_specs,
        output_specs=out_specs,
        attribute_schema=schema,
        effects=effect,
        batching_policy=policy,
    )
    register(op)
    if portable is not None:
        register_portable(name, portable)
        # Derivative fallback: diffing always falls back to the decomposition.
        register_portable_diff_fallback(name)
        if batching is None:
            # Resolved policy is BATCHING_RULE: pre-register the decomposition
            # as the namespaced batching rule so transforms always finds it.
            register_portable_batching_fallback(name)
    return op


class _BlockDecorator:
    """Pending declaration returned by `block(...)` with no name.

    `@etl.block` / `@etl.block(outputs=..., ...)` produces this object; applying
    it to an `etl.defn` function completes the declaration (name = function
    name; input/output specs derived from annotations or from the declared
    `inputs=` / `outputs=` / `result=`).
    """

    def __init__(
        self,
        inputs: Any,
        outputs: Any,
        result: Any,
        attributes: Any,
        effects: Any,
        batching: Any,
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._result = result
        self._attributes = attributes
        self._effects = effects
        self._batching = batching

    def __call__(self, fn: Any) -> Any:
        if not _is_defn(fn):
            raise BlockError(
                "@etl.block(...) must decorate an etl.defn function (got "
                f"{fn!r}); or use the factory form with an explicit name"
            )
        return _decorate(
            fn,
            self._inputs,
            self._outputs,
            self._result,
            self._attributes,
            self._effects,
            self._batching,
        )

    def __repr__(self) -> str:
        return "<pending etl.block declaration (no name and no function yet)>"


def _decorate(
    fn: Any,
    inputs: Any,
    outputs: Any,
    result: Any,
    attributes: Any,
    effects: Any,
    batching: Any,
) -> Any:
    if not _is_defn(fn):
        raise BlockError(
            f"portable implementation must be an etl.defn function, got {fn!r}"
        )
    name = _name_of(fn)
    in_specs = (
        _normalize_specs(inputs, "inputs", name)
        if inputs is not None
        else _derive_input_specs(fn, name)
    )
    out_specs = outputs if outputs is not None else result
    if out_specs is None:
        out_specs = _derive_output_specs(fn, name)
    out_specs = _normalize_specs(out_specs, "outputs", name)
    return _make_block(name, in_specs, out_specs, attributes, effects, batching, fn)


def block(
    name: Any = None,
    inputs: Any = None,
    outputs: Any = None,
    attributes: Any = None,
    effects: str = "pure",
    batching: Any = None,
    portable: Optional[Callable] = None,
    *,
    result: Any = None,
) -> Any:
    """Declare a custom operation — factory form or decorator form.

    Factory form (returns a callable BlockOp)::

        flash_attention = etl.block(
            "flash_attention",
            inputs=[
                etl.TensorSpec((256, 1024), etl.float32, name="q"),
                etl.TensorSpec((256, 1024), etl.float32, name="k"),
                etl.TensorSpec((256, 1024), etl.float32, name="v"),
            ],
            outputs=[etl.TensorSpec((256, 1024), etl.float32)],
            attributes={"scale": float, "causal": bool},
            batching="opaque_batched",
        )
        o = flash_attention(q, k, v, scale=0.5, causal=True)

    Decorator form over an etl.defn providing the portable implementation
    (specs derived from TensorSpec annotations or declared)::

        @etl.block(outputs=[etl.TensorSpec((), etl.float32)])
        @etl.defn
        def swish(x: etl.TensorSpec((), etl.float32)):
            return etl.sigmoid(x) * x

    Static attributes specialize the op (recorded as op attributes, part of
    cache keys and serialization). `batching` is one of the BatchingPolicy
    values; default: the portable decomposition if one exists, else
    "unsupported". Returns the registered BlockOp (or, when no name is given
    and no function is being decorated, a pending decorator).
    """
    if callable(name):
        if portable is not None:
            raise BlockError(
                "pass either a decorated function or portable=..., not both"
            )
        return _decorate(name, inputs, outputs, result, attributes, effects, batching)
    if name is None:
        if portable is None:
            return _BlockDecorator(inputs, outputs, result, attributes, effects, batching)
        name = _name_of(portable)
    if outputs is not None and result is not None:
        raise BlockError("pass either outputs= or result=, not both")
    return _make_block(
        name,
        inputs,
        outputs if outputs is not None else result,
        attributes,
        effects,
        batching,
        portable,
    )
