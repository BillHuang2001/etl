"""Elementwise numeric ops: arithmetic, math functions, and bitwise ops.

All functions follow the unified semantics documented in this node's
``CONTEXT.md``:

- Operands: ``SymbolicTensor`` or Python scalars (transparently promoted to
  0-d constant ops); a concrete ``Tensor`` raises ``TraceError``; calling
  outside an active trace raises ``TraceError``.
- Construction: ``builder.create(op_name, ...)`` on the active builder, with a
  call-site ``Location`` attached to the op.
- Dtype: binary ops promote via ``_utils.promote_dtypes`` (numpy semantics;
  scalars weakly per NEP 50 via ``_utils.weak_scalar_dtype``). Unary math
  functions (sqrt/exp/log/... and transcendental activations) follow numpy:
  integer/bool input → ``float64``; float input keeps its dtype. Unary
  shape/bit-pattern functions (``abs``/``negate``/``square``/``sign``)
  preserve the input dtype.
- Shape: binary ops broadcast via ``_utils.broadcast_shapes`` (symbolic dims
  via ``DimExpr.max``); unary ops preserve shape.
"""
from __future__ import annotations

import numpy as np

from etl import core
from etl import ir

from . import _utils

__all__ = [
    "add", "subtract", "multiply", "divide", "power", "remainder",
    "maximum", "minimum", "abs", "negate", "square", "sqrt", "exp", "log",
    "log1p", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "gelu", "erf",
    "sign", "cast", "bitwise_and", "bitwise_or", "bitwise_xor",
]


def add(x, y) -> "core.SymbolicTensor":
    """Elementwise addition (``x + y``).

    Registered as the ``SymbolicTensor.__add__``/``__radd__`` operator
    handler (kind ``add``).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        y: ``SymbolicTensor`` or Python scalar. At least one operand must be a
            ``SymbolicTensor``.

    Returns:
        ``SymbolicTensor`` with ``shape = broadcast_shapes(x.shape, y.shape)``
        and ``dtype = promote_dtypes(x.dtype, y.dtype)``.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand; both
            operands are Python scalars.
        core.ShapeError: static broadcast incompatibility.
        TypeError: unsupported operand kind.
    """
    raise NotImplementedError


def subtract(x, y) -> "core.SymbolicTensor":
    """Elementwise subtraction (``x - y``).

    Registered as the ``SymbolicTensor.__sub__`` operator handler (kind
    ``sub``). Dtype/shape rules identical to :func:`add`.
    """
    raise NotImplementedError


def multiply(x, y) -> "core.SymbolicTensor":
    """Elementwise multiplication (``x * y``).

    Registered as the ``SymbolicTensor.__mul__`` operator handler (kind
    ``mul``). Dtype/shape rules identical to :func:`add`.
    """
    raise NotImplementedError


def divide(x, y) -> "core.SymbolicTensor":
    """Elementwise true division (``x / y``).

    Registered as the ``SymbolicTensor.__truediv__`` operator handler (kind
    ``truediv``). Follows Python/numpy true-division semantics: integer
    operands produce a floating result (``int64 / int64 → float64``) per
    numpy promotion. Dtype = ``promote_dtypes`` then apply numpy's true-divide
    result rule; shape = broadcast.
    """
    raise NotImplementedError


def power(x, y) -> "core.SymbolicTensor":
    """Elementwise power (``x ** y``).

    Registered as the ``SymbolicTensor.__pow__`` operator handler (kind
    ``pow``). Dtype/shape rules identical to :func:`add` (numpy promotion).
    """
    raise NotImplementedError


def remainder(x, y) -> "core.SymbolicTensor":
    """Elementwise remainder of division (numpy ``remainder`` semantics:
    result has the sign of the dividend). Dtype/shape rules identical to
    :func:`add`."""
    raise NotImplementedError


def maximum(x, y) -> "core.SymbolicTensor":
    """Elementwise maximum (numpy ``maximum``, not the reduction ``max``).
    Dtype/shape rules identical to :func:`add`."""
    raise NotImplementedError


def minimum(x, y) -> "core.SymbolicTensor":
    """Elementwise minimum (numpy ``minimum``, not the reduction ``min``).
    Dtype/shape rules identical to :func:`add`."""
    raise NotImplementedError


def abs(x) -> "core.SymbolicTensor":
    """Elementwise absolute value. Shape and dtype preserved (``int → int``).
    Shorthand for ``multiply(x, sign(x))`` semantics, defined as its own op
    for backend efficiency; complex input yields the real magnitude per
    numpy."""
    raise NotImplementedError


def negate(x) -> "core.SymbolicTensor":
    """Elementwise negation (``-x``).

    Registered as the ``SymbolicTensor.__neg__`` operator handler (kind
    ``neg``). Shape and dtype preserved.
    """
    raise NotImplementedError


def square(x) -> "core.SymbolicTensor":
    """Elementwise square (``x * x``). Shape and dtype preserved (numpy
    ``square`` keeps ``int`` dtypes; no overflow promotion)."""
    raise NotImplementedError


def sqrt(x) -> "core.SymbolicTensor":
    """Elementwise square root. Integer/bool input → ``float64``; float input
    keeps its dtype. Shape preserved. Domain errors (negative input) surface
    at run time per backend semantics."""
    raise NotImplementedError


def exp(x) -> "core.SymbolicTensor":
    """Elementwise exponential. Integer/bool input → ``float64``; float input
    keeps its dtype. Shape preserved."""
    raise NotImplementedError


def log(x) -> "core.SymbolicTensor":
    """Elementwise natural logarithm. Integer/bool input → ``float64``; float
    input keeps its dtype. Shape preserved."""
    raise NotImplementedError


def log1p(x) -> "core.SymbolicTensor":
    """Elementwise ``log(1 + x)``, computed as a single op (numerically
    accurate near 0). Integer/bool input → ``float64``; float keeps dtype.
    Shape preserved."""
    raise NotImplementedError


def sin(x) -> "core.SymbolicTensor":
    """Elementwise sine. Integer/bool input → ``float64``; float keeps dtype.
    Shape preserved."""
    raise NotImplementedError


def cos(x) -> "core.SymbolicTensor":
    """Elementwise cosine. Integer/bool input → ``float64``; float keeps
    dtype. Shape preserved."""
    raise NotImplementedError


def tan(x) -> "core.SymbolicTensor":
    """Elementwise tangent. Integer/bool input → ``float64``; float keeps
    dtype. Shape preserved."""
    raise NotImplementedError


def tanh(x) -> "core.SymbolicTensor":
    """Elementwise hyperbolic tangent. Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved."""
    raise NotImplementedError


def sigmoid(x) -> "core.SymbolicTensor":
    """Elementwise logistic sigmoid ``1 / (1 + exp(-x))``. Integer/bool input
    → ``float64``; float keeps dtype. Shape preserved. Defined as its own op
    (not an expression) so the numpy backend and transforms can specialize
    its derivative."""
    raise NotImplementedError


def relu(x) -> "core.SymbolicTensor":
    """Elementwise rectified linear unit ``max(x, 0)``. Integer/bool input →
    ``float64``; float keeps dtype. Shape preserved. Defined as its own op so
    its derivative rule is exact at ``x == 0`` per backend convention."""
    raise NotImplementedError


def gelu(x) -> "core.SymbolicTensor":
    """Elementwise Gaussian Error Linear Unit (exact erf form,
    ``0.5 * x * (1 + erf(x / sqrt(2)))``). Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved. Defined as its own op for
    differentiable specialization."""
    raise NotImplementedError


def erf(x) -> "core.SymbolicTensor":
    """Elementwise Gauss error function. Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved."""
    raise NotImplementedError


def sign(x) -> "core.SymbolicTensor":
    """Elementwise sign (-1, 0, +1 per numpy; complex input returns the
    complex sign). Shape and dtype preserved."""
    raise NotImplementedError


def cast(x, dtype) -> "core.SymbolicTensor":
    """Cast to the given dtype.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        dtype: Target numpy dtype or etl dtype constant.

    Returns:
        ``SymbolicTensor`` with ``x``'s shape and exactly ``dtype``.

    Raises:
        core.DTypeError: ``dtype`` is not a valid numpy dtype.
        core.TraceError: no active trace; concrete ``Tensor`` operand.
    """
    raise NotImplementedError


def bitwise_and(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise AND (``x & y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    raise NotImplementedError


def bitwise_or(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise OR (``x | y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    raise NotImplementedError


def bitwise_xor(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise XOR (``x ^ y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    raise NotImplementedError
