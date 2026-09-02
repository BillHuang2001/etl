"""Symbolic shape dimensions: ``Dim`` and ``DimExpr``.

Shapes in etl may contain symbolic dimensions so graphs can be traced without
knowing every concrete size. ``Dim`` is a *named* symbolic dimension
(optionally with a known size); ``DimExpr`` is an expression tree over
``Dim``/``DimExpr``/``int`` operands supporting ``+ - * // % min max``.

Design: construction of expressions (the arithmetic dunders) is pure data
building — nothing is evaluated at construction time. *Evaluation*
(``DimExpr.evaluate``) is explicit; comparisons ("constraint-free evaluation
where possible") resolve both sides using known sizes only and raise
:class:`ShapeError` rather than guessing unresolved dims.

``core`` owns the type definitions only — shape inference (via ``DimExpr``
arithmetic) is ``ir``/``ops``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Optional, Union

from .errors import ShapeError

__all__ = ["Dim", "DimExpr", "dim"]

# Operand type of a DimExpr: a symbolic dim, a sub-expression, or a plain int.
_DimOperand = Union["Dim", "DimExpr", int]


def _as_dim_operand(value: Any) -> _DimOperand:
    """Validate and return a DimExpr operand (Dim/DimExpr/int)."""
    if isinstance(value, (Dim, DimExpr, int)):
        return value
    raise TypeError(
        f"Dim/DimExpr arithmetic requires int, Dim or DimExpr operands, "
        f"got {type(value).__name__}"
    )


def _resolve_operand(
    operand: _DimOperand, dim_sizes: Optional[Mapping[str, int]]
) -> int:
    """Resolve one operand of an expression to a concrete int.

    ``int`` → itself; ``DimExpr`` → recursive :meth:`DimExpr.evaluate`;
    ``Dim`` → the binding in ``dim_sizes`` (explicit bindings take
    precedence), else the Dim's own known ``size``.

    Raises:
        ShapeError: If a dimension has no known size and no binding, or a
            binding is not an integer.
    """
    if isinstance(operand, int):
        return operand
    if isinstance(operand, DimExpr):
        return operand.evaluate(dim_sizes)
    # Dim
    if dim_sizes is not None and operand.name in dim_sizes:
        size = dim_sizes[operand.name]
        if not isinstance(size, Integral):
            raise ShapeError(
                f"Binding for dimension {operand.name!r} must be an integer, "
                f"got {size!r}"
            )
        return int(size)
    if operand.size is not None:
        return operand.size
    raise ShapeError(
        f"Cannot evaluate {operand!r}: dimension {operand.name!r} has no "
        "known size and no binding in dim_sizes."
    )


def _evaluate_known(value: _DimOperand) -> int:
    """Evaluate ``value`` to an int using known sizes only (no bindings).

    Used by comparisons ("constraint-free evaluation where possible"): an
    ``int`` evaluates to itself; a ``Dim`` to its own known ``size``; a
    ``DimExpr`` recursively. An unresolved dimension raises :class:`ShapeError`
    — comparisons never guess.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, Dim):
        if value.size is None:
            raise ShapeError(
                f"Cannot compare unresolved dimension {value!r}: it has no "
                "known size (call DimExpr.evaluate(dim_sizes) to resolve it)."
            )
        return value.size
    return value.evaluate()


class _DimArithmeticMixin:
    """Arithmetic dunders shared by :class:`Dim` and :class:`DimExpr`.

    Every binary operation constructs a :class:`DimExpr` node — expression
    building is pure data construction (no evaluation happens here; call
    :meth:`DimExpr.evaluate` explicitly).

    Ordering comparisons (``< <= > >=``) are *not* expression building: they
    evaluate both sides using known sizes only and return a Python ``bool``.
    Comparing unresolved symbolic values raises :class:`ShapeError` instead of
    guessing.
    """

    __slots__ = ()

    def __add__(self, other: Any) -> "DimExpr":
        return DimExpr("add", self, _as_dim_operand(other))

    def __radd__(self, other: Any) -> "DimExpr":
        return DimExpr("add", _as_dim_operand(other), self)

    def __sub__(self, other: Any) -> "DimExpr":
        return DimExpr("sub", self, _as_dim_operand(other))

    def __rsub__(self, other: Any) -> "DimExpr":
        return DimExpr("sub", _as_dim_operand(other), self)

    def __mul__(self, other: Any) -> "DimExpr":
        return DimExpr("mul", self, _as_dim_operand(other))

    def __rmul__(self, other: Any) -> "DimExpr":
        return DimExpr("mul", _as_dim_operand(other), self)

    def __floordiv__(self, other: Any) -> "DimExpr":
        return DimExpr("floordiv", self, _as_dim_operand(other))

    def __rfloordiv__(self, other: Any) -> "DimExpr":
        return DimExpr("floordiv", _as_dim_operand(other), self)

    def __mod__(self, other: Any) -> "DimExpr":
        return DimExpr("mod", self, _as_dim_operand(other))

    def __rmod__(self, other: Any) -> "DimExpr":
        return DimExpr("mod", _as_dim_operand(other), self)

    def min(self, other: Any) -> "DimExpr":
        """Build ``min(self, other)`` as a DimExpr node (no evaluation)."""
        return DimExpr("min", self, _as_dim_operand(other))

    def max(self, other: Any) -> "DimExpr":
        """Build ``max(self, other)`` as a DimExpr node (no evaluation)."""
        return DimExpr("max", self, _as_dim_operand(other))

    # --- Ordering comparisons: "constraint-free evaluation where possible" ---
    # These are NOT expression builders: both sides are evaluated using known
    # sizes only (there is no dim_sizes mapping available to comparisons) and
    # compared as ints. An invalid operand type raises TypeError (via
    # _as_dim_operand); unresolved symbolic operands raise ShapeError rather
    # than guessing.

    def __lt__(self, other: Any) -> bool:
        operand = _as_dim_operand(other)
        return _evaluate_known(self) < _evaluate_known(operand)

    def __le__(self, other: Any) -> bool:
        operand = _as_dim_operand(other)
        return _evaluate_known(self) <= _evaluate_known(operand)

    def __gt__(self, other: Any) -> bool:
        operand = _as_dim_operand(other)
        return _evaluate_known(self) > _evaluate_known(operand)

    def __ge__(self, other: Any) -> bool:
        operand = _as_dim_operand(other)
        return _evaluate_known(self) >= _evaluate_known(operand)


@dataclass(frozen=True)
class Dim(_DimArithmeticMixin):
    """A named symbolic dimension with an optional known size.

    Two ``Dim``\\s are equal iff name and size match. A ``Dim`` is a valid
    ``DimExpr`` operand and participates in arithmetic via the shared mixin.

    Attributes:
        name: Dimension name (string). Dims with the same name unify.
        size: Optional known size; ``None`` means unknown until runtime.
    """

    name: str
    size: Optional[int] = None

    def __repr__(self) -> str:
        if self.size is None:
            return f"Dim({self.name!r})"
        return f"Dim({self.name!r}, size={self.size})"

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        raise ShapeError(
            f"Cannot use symbolic dimension {self!r} as a Python boolean; "
            "call DimExpr.evaluate(dim_sizes) to resolve it to an int."
        )


@dataclass(frozen=True)
class DimExpr(_DimArithmeticMixin):
    """An expression tree over symbolic dimensions.

    Supported binary operators (all construct nodes; nothing is evaluated at
    construction time): ``+ - * // %`` via Python operators and
    ``min(other)`` / ``max(other)`` via methods.

    Attributes:
        op: One of ``"add" | "sub" | "mul" | "floordiv" | "mod" | "min" |
            "max"``.
        left: Left operand (``Dim``/``DimExpr``/``int``).
        right: Right operand (``Dim``/``DimExpr``/``int``).

    Equality is structural (dataclass ``__eq__`` over op/left/right) — no
    evaluation is performed by ``==``.
    """

    op: str
    left: _DimOperand
    right: _DimOperand

    def __post_init__(self) -> None:
        if self.op not in ("add", "sub", "mul", "floordiv", "mod", "min", "max"):
            raise ValueError(f"Unknown DimExpr op: {self.op!r}")
        object.__setattr__(self, "left", _as_dim_operand(self.left))
        object.__setattr__(self, "right", _as_dim_operand(self.right))

    def evaluate(self, dim_sizes: Optional[Mapping[str, int]] = None) -> int:
        """Evaluate the expression to a concrete integer.

        Substitutes every ``Dim`` operand's size from ``dim_sizes`` (or from
        the ``Dim``'s own known ``size`` when present) and computes the result
        with integer semantics (``//`` for floordiv, builtins for ``min``/
        ``max``).

        Args:
            dim_sizes: Optional mapping of dim name → concrete size. A ``Dim``
                whose size is unknown here raises :class:`ShapeError`.

        Returns:
            The concrete integer value.

        Raises:
            ShapeError: If a dimension is unresolved, or arithmetic is
                invalid (e.g. division by zero, non-integer division).
        """
        left = _resolve_operand(self.left, dim_sizes)
        right = _resolve_operand(self.right, dim_sizes)
        if self.op == "add":
            return left + right
        if self.op == "sub":
            return left - right
        if self.op == "mul":
            return left * right
        if self.op == "floordiv":
            if right == 0:
                raise ShapeError(f"Division by zero in DimExpr {self!r}")
            return left // right
        if self.op == "mod":
            if right == 0:
                raise ShapeError(f"Modulo by zero in DimExpr {self!r}")
            return left % right
        if self.op == "min":
            return min(left, right)
        if self.op == "max":
            return max(left, right)
        raise ValueError(f"Unknown DimExpr op: {self.op!r}")  # unreachable

    def __bool__(self) -> bool:
        raise ShapeError(
            f"Cannot use symbolic expression {self!r} as a Python boolean; "
            "call DimExpr.evaluate(dim_sizes) to resolve it to an int."
        )

    def __repr__(self) -> str:
        return f"DimExpr({self.op!r}, left={self.left!r}, right={self.right!r})"


def dim(name_or_int: Union[str, int, Dim]) -> Dim:
    """Create a :class:`Dim` from a name, a known size, or a ``Dim``.

    Args:
        name_or_int: If a string, a named symbolic dimension with unknown
            size. If an int ``n``, an anonymous dimension named ``f"dim_{n}"``
            with known size ``n`` (deterministic name — equal ints produce
            equal dims, and the known size makes evaluation exact). If a
            ``Dim``, returned unchanged.

    Returns:
        The :class:`Dim`.

    Raises:
        TypeError: If the argument is not a str, int or Dim.
    """
    if isinstance(name_or_int, Dim):
        return name_or_int
    if isinstance(name_or_int, int):
        return Dim(name=f"dim_{name_or_int}", size=name_or_int)
    if isinstance(name_or_int, str):
        return Dim(name=name_or_int)
    raise TypeError(
        f"dim() expects a str, int or Dim, got {type(name_or_int).__name__}"
    )
