"""Op definition registry — the contract of every EvoXIR op.

Every op that can appear in EvoXIR is declared here. An ``OpDef`` declares the
op's *contract*: name, category, operand arity, result count, attribute
schema, effect, shape-inference hook, region structure, terminator role.

Declaring an op in the registry does NOT mean every backend implements it —
backends reject unsupported ops explicitly via their capabilities, never
silently (see root CONTEXT.md, error strategy).

Attribute types: values must be JSON-able Python data; ``dtype`` values are
numpy dtype names (strings); ``ndarray`` values are numpy arrays (serialized
as base64 npy payloads); ``shape`` values are tuples of ints/DimExprs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..types import ValueType

# --- attribute type tags -----------------------------------------------------

ATTR_BOOL = "bool"
ATTR_INT = "int"  # int | None allowed where documented (e.g. argmax axis)
ATTR_FLOAT = "float"
ATTR_STR = "str"
ATTR_DTYPE = "dtype"  # numpy dtype name
ATTR_INTS = "ints"  # tuple/list of ints
ATTR_FLOATS = "floats"  # tuple/list of floats
ATTR_STRS = "strs"  # tuple/list of strs
ATTR_NESTED_INTS = "nested_ints"  # nested tuple of (int, ...) pairs
ATTR_SHAPE = "shape"  # tuple of int | DimExpr | None
ATTR_NDARRAY = "ndarray"  # numpy array (constant payload)
ATTR_ANY = "any"  # JSON-able (documented per op)

ATTR_TYPE_NAMES = frozenset(
    {
        ATTR_BOOL,
        ATTR_INT,
        ATTR_FLOAT,
        ATTR_STR,
        ATTR_DTYPE,
        ATTR_INTS,
        ATTR_FLOATS,
        ATTR_STRS,
        ATTR_NESTED_INTS,
        ATTR_SHAPE,
        ATTR_NDARRAY,
        ATTR_ANY,
    }
)

_NO_DEFAULT = object()


@dataclass(frozen=True)
class AttrSpec:
    """One declared attribute of an op.

    Attributes:
        name: Attribute key.
        type: One of the ``ATTR_*`` tags.
        default: Default value; if omitted the attribute is required.
        description: What the attribute controls.
    """

    name: str
    type: str
    default: Any = _NO_DEFAULT
    description: str = ""

    @property
    def required(self) -> bool:
        """True if the attribute must be present on every op instance."""
        return self.default is _NO_DEFAULT


#: Shape-inference hook: (input types, attributes) -> result types.
#: ``None`` on an OpDef means op-specific resolution by the Builder or
#: explicit ``result_types`` at the call site (see ``inference.py``).
ShapeInferenceFn = Callable[
    [tuple["ValueType", ...], dict[str, Any]], tuple["ValueType", ...]
]


@dataclass(frozen=True)
class OpDef:
    """The declared contract of one op.

    Attributes:
        name: Registered op name (registry key).
        category: Grouping for documentation/routing (elementwise, comparison,
            structure, reduction, linalg, control, terminator, collective).
        description: One-line semantics.
        arity: Exact operand count, or ``(min, max)`` with ``None`` = unbounded
            (variadic).
        result_count: Exact result count, ``(min, max)``, or ``None`` =
            unknown until built (op-specific resolution).
        effect: One of the ``EFFECT_*`` kinds (``effects.py``).
        attributes: Declared attribute schema (``AttrSpec``s).
        shape_fn: Shape-inference hook, or None (see module docstring).
        regions: Number of nested regions the op owns (0, 1, or 2).
        is_terminator: True if the op terminates its block (``return``).
    """

    name: str
    category: str
    description: str
    arity: int | tuple[int, int | None]
    result_count: int | tuple[int, int | None] | None
    effect: str
    attributes: tuple[AttrSpec, ...] = ()
    shape_fn: ShapeInferenceFn | None = None
    regions: int = 0
    is_terminator: bool = False

    def check_arity(self, count: int) -> bool:
        """True if ``count`` operands satisfy this op's arity."""
        if isinstance(self.arity, int):
            return count == self.arity
        lo, hi = self.arity
        return count >= lo and (hi is None or count <= hi)


# --- registry ----------------------------------------------------------------

_REGISTRY: dict[str, OpDef] = {}


def register_opdef(opdef: OpDef) -> OpDef:
    """Register an ``OpDef`` (idempotent per name).

    Raises:
        ValueError: If the op name is already registered.
    """
    if opdef.name in _REGISTRY:
        raise ValueError(f"op '{opdef.name}' is already registered")
    _REGISTRY[opdef.name] = opdef
    return opdef


def opdef(name: str) -> OpDef:
    """Look up the ``OpDef`` for an op name.

    Raises:
        KeyError: If no op with this name is registered.
    """
    return _REGISTRY[name]


def has_opdef(name: str) -> bool:
    """True if an op with this name is registered."""
    return name in _REGISTRY


def op_names() -> tuple[str, ...]:
    """All registered op names (sorted)."""
    return tuple(sorted(_REGISTRY))


def all_opdefs() -> tuple[OpDef, ...]:
    """All registered ``OpDef``s (sorted by name)."""
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


# Importing the category tables registers the canonical v1 op set.
from . import collective, control, elementwise, linalg, reduction, structure  # noqa: E402,F401
