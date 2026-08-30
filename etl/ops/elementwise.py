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

Implementation note: dtype/shape inference is delegated to the canonical
``etl.ir`` op registry (result types are READ BACK from the inferred
``op.result.type`` — this module keeps no parallel inference table). Where a
documented dtype rule cannot be expressed by the IR's fixed inference rules
(``divide`` true division; unary math on integral input; ``abs`` of complex),
the rule is composed transparently from explicit ``cast`` ops.
"""
from __future__ import annotations

from etl import core

from . import _utils

__all__ = [
    "add", "subtract", "multiply", "divide", "power", "remainder",
    "maximum", "minimum", "abs", "negate", "square", "sqrt", "exp", "log",
    "log1p", "sin", "cos", "tan", "tanh", "sigmoid", "relu", "gelu", "erf",
    "sign", "cast", "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_left_shift", "bitwise_right_shift",
]

# --- private construction helpers -------------------------------------------

#: Exact Python scalar kinds accepted as operands (the same set
#: ``_utils.as_operand`` accepts; numpy scalars are deliberately NOT included).
_SCALAR_KINDS = (bool, int, float, complex)

#: Numpy dtype kinds considered integral for the true-division and unary-math
#: cast rules: bool, signed int, unsigned int.
_INTEGRAL_KINDS = "biu"


def _is_scalar(x) -> bool:
    """True when ``x`` is an exact Python scalar (bool/int/float/complex)."""
    return type(x) in _SCALAR_KINDS


def _wrap(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result in a SymbolicTensor, reading the dtype and
    shape back from the IR's inferred result type (never computed here)."""
    return core.SymbolicTensor(
        value=op.result,
        dtype=op.result.type.dtype,
        shape=op.result.type.shape,
        location=loc,
    )


def _binary_operands(op_name, x, y, loc):
    """Normalize the two operands of a binary op per the unified rules.

    ``SymbolicTensor`` operands pass through unchanged. A Python scalar is
    promoted to a 0-d Constant whose dtype is weakly pre-promoted against the
    OTHER (symbolic) operand's dtype — NEP 50 semantics, see
    ``_utils.as_operand``.

    Raises:
        core.TraceError: both operands are Python scalars (etl has no eager
            mode), or a concrete ``Tensor`` appears (canonical message via
            ``as_operand``).
        TypeError: unsupported operand kind (via ``as_operand``).
    """
    if isinstance(x, core.SymbolicTensor):
        return x, _utils.as_operand(y, dtype_hint=x.dtype, location=loc)
    if isinstance(y, core.SymbolicTensor):
        return _utils.as_operand(x, dtype_hint=y.dtype, location=loc), y
    if _is_scalar(x) and _is_scalar(y):
        raise core.TraceError(
            f"{op_name}: at least one operand must be a SymbolicTensor, got "
            "two Python scalars. etl has no eager mode — trace a graph with "
            "etl.trace or @etl.defn, or build and run one with etl.evaluate."
        )
    # Neither operand is symbolic and they are not both scalars: at least one
    # side is a concrete Tensor or an unsupported kind. Probe the non-scalar
    # side first so no spurious Constant op is built on the error path;
    # as_operand raises the canonical TraceError/TypeError.
    if not _is_scalar(x):
        _utils.as_operand(x, dtype_hint=None, location=loc)  # always raises
    _utils.as_operand(y, dtype_hint=None, location=loc)  # always raises
    raise AssertionError(  # pragma: no cover — as_operand raises above
        "unreachable: as_operand raises for non-symbolic, non-scalar operands"
    )


def _emit_binary(builder, op_name, xt, yt, loc) -> "core.SymbolicTensor":
    """Build the two-operand op and wrap its inferred result."""
    op = builder.create(op_name, operands=(xt.value, yt.value), location=loc)
    return _wrap(op, loc)


def _emit_unary(builder, op_name, xt, loc) -> "core.SymbolicTensor":
    """Build the one-operand op and wrap its inferred result."""
    op = builder.create(op_name, operands=(xt.value,), location=loc)
    return _wrap(op, loc)


def _cast(builder, xt, target, loc) -> "core.SymbolicTensor":
    """Build a ``cast`` op to ``target`` and wrap its inferred result."""
    op = builder.create(
        "cast", operands=(xt.value,), attributes={"dtype": target}, location=loc
    )
    return _wrap(op, loc)


def _to_float64_if_integral(builder, xt, loc) -> "core.SymbolicTensor":
    """Cast bool/int/uint operands to float64; other kinds pass through.

    Numpy's unary-math rule (integer/bool input → float64) — composed
    explicitly because ``infer_elementwise_unary`` preserves the dtype.
    """
    if xt.dtype.kind in _INTEGRAL_KINDS:
        return _cast(builder, xt, core.float64, loc)
    return xt


def _require_kinds(op_name, tensors, kinds, expectation) -> None:
    """Raise ``core.DTypeError`` when any tensor's dtype kind is not allowed."""
    for t in tensors:
        if t.dtype.kind not in kinds:
            raise core.DTypeError(
                f"{op_name}: operands must have {expectation} dtype "
                f"(numpy kinds '{kinds}'), got {t.dtype}"
            )


def _binary(op_name, x, y) -> "core.SymbolicTensor":
    """Shared body of the plain broadcasting binary elementwise ops."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands(op_name, x, y, loc)
    return _emit_binary(builder, op_name, xt, yt, loc)


def _unary(op_name, x, *, math: bool = False) -> "core.SymbolicTensor":
    """Shared body of the unary elementwise ops.

    Args:
        math: When True, apply numpy's unary-math dtype rule (integral input
            → float64 via an explicit cast). When False the input dtype is
            preserved (unary shape/bit-pattern functions).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt = _utils.as_operand(x, dtype_hint=None, location=loc)
    if math:
        xt = _to_float64_if_integral(builder, xt, loc)
    return _emit_unary(builder, op_name, xt, loc)


# --- arithmetic ---------------------------------------------------------------


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
    return _binary("add", x, y)


def subtract(x, y) -> "core.SymbolicTensor":
    """Elementwise subtraction (``x - y``).

    Registered as the ``SymbolicTensor.__sub__`` operator handler (kind
    ``sub``). Dtype/shape rules identical to :func:`add`.
    """
    return _binary("subtract", x, y)


def multiply(x, y) -> "core.SymbolicTensor":
    """Elementwise multiplication (``x * y``).

    Registered as the ``SymbolicTensor.__mul__`` operator handler (kind
    ``mul``). Dtype/shape rules identical to :func:`add`.
    """
    return _binary("multiply", x, y)


def divide(x, y) -> "core.SymbolicTensor":
    """Elementwise true division (``x / y``).

    Registered as the ``SymbolicTensor.__truediv__`` operator handler (kind
    ``truediv``). Follows Python/numpy true-division semantics: integer
    operands produce a floating result (``int64 / int64 → float64``) per
    numpy promotion. Dtype = ``promote_dtypes`` then apply numpy's true-divide
    result rule; shape = broadcast.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("divide", x, y, loc)
    # numpy true division, expressed exactly: PROMOTE first
    # (promote_dtypes = numpy result_type, after weak scalar pre-promotion),
    # then apply the true-divide result rule — an integral promoted dtype
    # yields float64 (int64 / int64 → float64); any floating/complex promoted
    # dtype stands (int8 / float32 → float32, float16 / int8 → float16,
    # complex64 / int64 → complex128). The IR's infer_elementwise_binary
    # would keep the integral dtype, so compose the float64 rule with
    # explicit casts — transparent composition of primitives (the ir
    # registry cannot be modified from this node).
    if _utils.promote_dtypes(xt.dtype, yt.dtype).kind in _INTEGRAL_KINDS:
        xt = _cast(builder, xt, core.float64, loc)
        yt = _cast(builder, yt, core.float64, loc)
    return _emit_binary(builder, "divide", xt, yt, loc)


def power(x, y) -> "core.SymbolicTensor":
    """Elementwise power (``x ** y``).

    Registered as the ``SymbolicTensor.__pow__`` operator handler (kind
    ``pow``). Dtype/shape rules identical to :func:`add` (numpy promotion).
    """
    return _binary("power", x, y)


def remainder(x, y) -> "core.SymbolicTensor":
    """Elementwise remainder of division (numpy ``remainder`` semantics:
    result has the sign of the dividend). Dtype/shape rules identical to
    :func:`add`."""
    return _binary("remainder", x, y)


def maximum(x, y) -> "core.SymbolicTensor":
    """Elementwise maximum (numpy ``maximum``, not the reduction ``max``).
    Dtype/shape rules identical to :func:`add`."""
    return _binary("maximum", x, y)


def minimum(x, y) -> "core.SymbolicTensor":
    """Elementwise minimum (numpy ``minimum``, not the reduction ``min``).
    Dtype/shape rules identical to :func:`add`."""
    return _binary("minimum", x, y)


# --- unary shape/bit-pattern functions (dtype preserving) ---------------------


def abs(x) -> "core.SymbolicTensor":
    """Elementwise absolute value. Shape and dtype preserved (``int → int``).
    Shorthand for ``multiply(x, sign(x))`` semantics, defined as its own op
    for backend efficiency; complex input yields the real magnitude per
    numpy."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt = _utils.as_operand(x, dtype_hint=None, location=loc)
    result = _emit_unary(builder, "abs", xt, loc)
    if xt.dtype.kind == "c":
        # numpy abs of a complex value is the REAL magnitude (complex64 →
        # float32, complex128 → float64). infer_elementwise_unary preserves
        # the operand dtype, so compose the documented rule with a cast.
        target = core.float32 if xt.dtype == core.complex64 else core.float64
        result = _cast(builder, result, target, loc)
    return result


def negate(x) -> "core.SymbolicTensor":
    """Elementwise negation (``-x``).

    Registered as the ``SymbolicTensor.__neg__`` operator handler (kind
    ``neg``). Shape and dtype preserved.
    """
    return _unary("negate", x)


def square(x) -> "core.SymbolicTensor":
    """Elementwise square (``x * x``). Shape and dtype preserved (numpy
    ``square`` keeps ``int`` dtypes; no overflow promotion)."""
    return _unary("square", x)


# --- unary math functions (integral input → float64) --------------------------


def sqrt(x) -> "core.SymbolicTensor":
    """Elementwise square root. Integer/bool input → ``float64``; float input
    keeps its dtype. Shape preserved. Domain errors (negative input) surface
    at run time per backend semantics."""
    return _unary("sqrt", x, math=True)


def exp(x) -> "core.SymbolicTensor":
    """Elementwise exponential. Integer/bool input → ``float64``; float input
    keeps its dtype. Shape preserved."""
    return _unary("exp", x, math=True)


def log(x) -> "core.SymbolicTensor":
    """Elementwise natural logarithm. Integer/bool input → ``float64``; float
    input keeps its dtype. Shape preserved."""
    return _unary("log", x, math=True)


def log1p(x) -> "core.SymbolicTensor":
    """Elementwise ``log(1 + x)``, computed as a single op (numerically
    accurate near 0). Integer/bool input → ``float64``; float keeps dtype.
    Shape preserved."""
    return _unary("log1p", x, math=True)


def sin(x) -> "core.SymbolicTensor":
    """Elementwise sine. Integer/bool input → ``float64``; float keeps dtype.
    Shape preserved."""
    return _unary("sin", x, math=True)


def cos(x) -> "core.SymbolicTensor":
    """Elementwise cosine. Integer/bool input → ``float64``; float keeps
    dtype. Shape preserved."""
    return _unary("cos", x, math=True)


def tan(x) -> "core.SymbolicTensor":
    """Elementwise tangent. Integer/bool input → ``float64``; float keeps
    dtype. Shape preserved."""
    return _unary("tan", x, math=True)


def acos(x) -> "core.SymbolicTensor":
    """Elementwise arccosine. Integer/bool input → ``float64``; float keeps
    dtype. Shape preserved."""
    return _unary("acos", x, math=True)


def floor(x) -> "core.SymbolicTensor":
    """Elementwise floor (numpy semantics; integer input is unchanged).
    Shape and dtype preserved."""
    return _unary("floor", x)


def ceil(x) -> "core.SymbolicTensor":
    """Elementwise ceiling (numpy semantics; integer input is unchanged).
    Shape and dtype preserved."""
    return _unary("ceil", x)


def round(x) -> "core.SymbolicTensor":
    """Elementwise round-half-to-even (numpy ``round`` semantics; integer
    input is unchanged). Shape preserved; dtype preserved except bool input
    → ``float64`` (numpy's round ufunc promotes bool ARRAYS to float16 — a
    numpy artifact; etl follows the scalar convention ``round(True) → 1.0``
    and the unary-math rule for bool)."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt = _utils.as_operand(x, dtype_hint=None, location=loc)
    if xt.dtype.kind == "b":
        xt = _cast(builder, xt, core.float64, loc)
    return _emit_unary(builder, "round", xt, loc)


def tanh(x) -> "core.SymbolicTensor":
    """Elementwise hyperbolic tangent. Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved."""
    return _unary("tanh", x, math=True)


def sigmoid(x) -> "core.SymbolicTensor":
    """Elementwise logistic sigmoid ``1 / (1 + exp(-x))``. Integer/bool input
    → ``float64``; float keeps dtype. Shape preserved. Defined as its own op
    (not an expression) so the numpy backend and transforms can specialize
    its derivative."""
    return _unary("sigmoid", x, math=True)


def relu(x) -> "core.SymbolicTensor":
    """Elementwise rectified linear unit ``max(x, 0)``. Integer/bool input →
    ``float64``; float keeps dtype. Shape preserved. Defined as its own op so
    its derivative rule is exact at ``x == 0`` per backend convention."""
    return _unary("relu", x, math=True)


def gelu(x) -> "core.SymbolicTensor":
    """Elementwise Gaussian Error Linear Unit (exact erf form,
    ``0.5 * x * (1 + erf(x / sqrt(2)))``). Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved. Defined as its own op for
    differentiable specialization."""
    return _unary("gelu", x, math=True)


def erf(x) -> "core.SymbolicTensor":
    """Elementwise Gauss error function. Integer/bool input → ``float64``;
    float keeps dtype. Shape preserved."""
    return _unary("erf", x, math=True)


def sign(x) -> "core.SymbolicTensor":
    """Elementwise sign (-1, 0, +1 per numpy; complex input returns the
    complex sign). Shape and dtype preserved."""
    return _unary("sign", x)


# --- cast ---------------------------------------------------------------------


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
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    try:
        target = core.dtype(dtype)
    except core.DTypeError as exc:
        raise core.DTypeError(f"cast: {dtype!r} is not a valid dtype: {exc}") from exc
    xt = _utils.as_operand(x, dtype_hint=None, location=loc)
    return _cast(builder, xt, target, loc)


# --- bitwise (integer/bool operands only) -------------------------------------


def bitwise_and(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise AND (``x & y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("bitwise_and", x, y, loc)
    _require_kinds("bitwise_and", (xt, yt), _INTEGRAL_KINDS, "integer or bool")
    return _emit_binary(builder, "bitwise_and", xt, yt, loc)


def bitwise_or(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise OR (``x | y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("bitwise_or", x, y, loc)
    _require_kinds("bitwise_or", (xt, yt), _INTEGRAL_KINDS, "integer or bool")
    return _emit_binary(builder, "bitwise_or", xt, yt, loc)


def bitwise_xor(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise XOR (``x ^ y``). Both operands must have integer
    or bool dtype (``core.DTypeError`` otherwise). Dtype/shape rules
    identical to :func:`add`."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("bitwise_xor", x, y, loc)
    _require_kinds("bitwise_xor", (xt, yt), _INTEGRAL_KINDS, "integer or bool")
    return _emit_binary(builder, "bitwise_xor", xt, yt, loc)


#: Numpy dtype kinds accepted as shift operands: signed and unsigned
#: integers only. numpy 2.x shift support for bool is a promotion artifact
#: (bool arrays shift as ints) and StableHLO defines no shift on i1, so etl
#: requires explicit integer dtypes for shifts (never silent promotion).
_INTEGER_KINDS = "iu"


def bitwise_left_shift(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise left shift (``x << y``). Both operands must have
    integer dtype (``core.DTypeError`` otherwise — bool is NOT accepted;
    numpy's bool shift is a promotion artifact and StableHLO has no i1
    shift). Dtype/shape rules identical to :func:`add` (operands promote
    per ``np.result_type``; the shift amount broadcasts against ``x``)."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("bitwise_left_shift", x, y, loc)
    _require_kinds("bitwise_left_shift", (xt, yt), _INTEGER_KINDS, "integer")
    return _emit_binary(builder, "bitwise_left_shift", xt, yt, loc)


def bitwise_right_shift(x, y) -> "core.SymbolicTensor":
    """Elementwise bitwise right shift (``x >> y``). Both operands must have
    integer dtype (``core.DTypeError`` otherwise — bool is NOT accepted, see
    :func:`bitwise_left_shift`). Semantics are numpy dtype-natural: the
    shift is ARITHMETIC (sign-filling) on signed integer dtypes and LOGICAL
    (zero-filling) on unsigned dtypes — exactly what ``np.right_shift``
    does per dtype; the StableHLO writer maps signed →
    ``stablehlo.shift_right_arithmetic`` and unsigned →
    ``stablehlo.shift_right_logical`` by the promoted operand dtype.
    Dtype/shape rules identical to :func:`add`."""
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    xt, yt = _binary_operands("bitwise_right_shift", x, y, loc)
    _require_kinds("bitwise_right_shift", (xt, yt), _INTEGER_KINDS, "integer")
    return _emit_binary(builder, "bitwise_right_shift", xt, yt, loc)
