"""Elementwise kernels (arith / activations / comparisons / select / cast / broadcast).

Covers these ops:
arith: add, subtract, multiply, divide, power, remainder, maximum, minimum;
activations: abs, negate, square, sqrt, exp, log, log1p, sin, cos, tan, tanh,
sigmoid, relu, gelu, erf, sign;
bitwise/logical: bitwise_and, bitwise_or, bitwise_xor, logical_and, logical_or,
logical_not;
cast/comparisons: cast, equal, not_equal, less, less_equal, greater,
greater_equal;
structure/control: select, broadcast, stop_gradient (identity passthrough —
returns the operand unchanged).

Design notes (binding, parent CONTEXT.md):

- dtypes map 1:1 between numpy and etl; kernels reject object/string/void/
  structured dtypes with ``core.BackendError`` (naming the op) rather than
  silently coercing. No promotion beyond what ops defined.
- etl.ops already emits compensation casts (divide/unary-math pre-cast to
  float64, abs post-cast) as ordinary op graphs — the kernels execute the
  graph AS-IS, no special handling. Python scalars were already promoted to
  ``constant`` ops by the frontend, so every operand is a ``core.Tensor``.
- ``relu``/``gelu``/``square`` are frontend ops that may arrive here directly
  (the numpy backend executes what the frontend produced); the StableHLO
  exporter's decompositions are unrelated to interpreter behavior.
- The interpreter validates every result against the op's declared result
  types (dtype exact, symbolic shape via ``ctx.bindings``) afterwards —
  kernels just compute.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from etl import core

from ..shapes import evaluate_dim_expr
from .indexing import _shape_error

__all__ = ["register_kernels"]

#: Dtype kinds the numpy backend cannot compute with: object, bytes-string,
#: unicode-string, void/structured.
_UNSUPPORTED_KINDS = frozenset("OSUV")

#: Vectorized ``math.erf`` (no scipy dependency). ``frompyfunc`` produces an
#: object array; results are cast back to the operand dtype by ``_erf_array``.
_ERF_UFUNC = np.frompyfunc(math.erf, 1, 1)


def _check_supported(op_name: str, *arrays: np.ndarray) -> None:
    """Reject object/string/void/structured dtypes (never silently coerce)."""
    for array in arrays:
        if array.dtype.kind in _UNSUPPORTED_KINDS:
            raise core.BackendError(
                f"op '{op_name}': unsupported dtype {array.dtype} — the numpy "
                "backend does not support object/string/void/structured dtypes"
            )


def _erf_array(x: np.ndarray) -> np.ndarray:
    """Vectorized ``math.erf`` with the result cast back to ``x``'s dtype."""
    return _ERF_UFUNC(x).astype(x.dtype)


def _check_broadcast(op_name: str, *shapes: tuple) -> None:
    """Fail with ``core.ShapeError`` when the operand shapes cannot broadcast.

    numpy raises a raw ``ValueError`` on incompatible broadcasts; the backend
    contract requires ``core.ShapeError`` instead (the message names the op
    and embeds numpy's description, which lists every offending shape).
    """
    try:
        np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise _shape_error(f"op '{op_name}'", exc) from exc


def _binary(np_fn: Callable[..., Any]) -> Callable[..., core.Tensor]:
    """Kernel factory for two-operand broadcasting ops.

    Both operands are arrays (the frontend promoted scalars to ``constant``
    ops), so numpy's ufunc broadcasting/result-type rules apply exactly.
    Comparison/logical ufuncs return a numpy SCALAR (``np.bool_``) on 0-d
    operands — ``np.asarray`` normalizes it to the 0-d ndarray ``core.Tensor``
    requires (a no-op for real arrays; dtype never changes).
    """

    def kernel(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
        a, b = operands
        a_arr, b_arr = a.numpy(), b.numpy()
        _check_supported(op.name, a_arr, b_arr)
        _check_broadcast(op.name, a_arr.shape, b_arr.shape)
        return core.Tensor(np.asarray(np_fn(a_arr, b_arr)))

    return kernel


def _unary(np_fn: Callable[..., Any]) -> Callable[..., core.Tensor]:
    """Kernel factory for one-operand shape/dtype-preserving ops.

    ``logical_not`` returns a numpy scalar (``np.bool_``) on 0-d operands —
    ``np.asarray`` normalizes it to the 0-d ndarray ``core.Tensor`` requires
    (a no-op for real arrays; dtype never changes).
    """

    def kernel(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
        (x,) = operands
        x_arr = x.numpy()
        _check_supported(op.name, x_arr)
        return core.Tensor(np.asarray(np_fn(x_arr)))

    return kernel


def _max_min(np_fn: Callable[..., Any]) -> Callable[..., core.Tensor]:
    """Kernel factory for ``maximum``/``minimum`` (numpy has no complex order)."""

    def kernel(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
        a, b = operands
        a_arr, b_arr = a.numpy(), b.numpy()
        _check_supported(op.name, a_arr, b_arr)
        if a_arr.dtype.kind == "c" or b_arr.dtype.kind == "c":
            raise core.BackendError(
                f"op '{op.name}': complex inputs are not supported (numpy "
                "defines no ordering for complex numbers)"
            )
        _check_broadcast(op.name, a_arr.shape, b_arr.shape)
        return core.Tensor(np.asarray(np_fn(a_arr, b_arr)))

    return kernel


def _erf(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``erf``: elementwise Gauss error function via ``math.erf``."""
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    if x_arr.dtype.kind == "c":
        raise core.BackendError(
            f"op '{op.name}': complex input is not supported by the numpy "
            "backend erf kernel (math.erf is real-only)"
        )
    return core.Tensor(_erf_array(x_arr))


def _sigmoid(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``sigmoid``: 1 / (1 + exp(-x)) with dtype-stable constants."""
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    one = np.asarray(1.0, dtype=x_arr.dtype)
    return core.Tensor(one / (one + np.exp(-x_arr)))


def _relu(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``relu``: max(x, 0)."""
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    return core.Tensor(np.maximum(x_arr, 0))


def _gelu(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``gelu``: exact erf form 0.5 * x * (1 + erf(x / sqrt(2))).

    All constants are typed to the operand dtype so the result dtype is
    deterministic across numpy versions (python-scalar promotion rules
    changed with NEP 50).
    """
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    if x_arr.dtype.kind == "c":
        raise core.BackendError(
            f"op '{op.name}': complex input is not supported by the numpy "
            "backend gelu kernel (erf is real-only)"
        )
    half = np.asarray(0.5, dtype=x_arr.dtype)
    one = np.asarray(1.0, dtype=x_arr.dtype)
    sqrt_two = np.asarray(math.sqrt(2.0), dtype=x_arr.dtype)
    return core.Tensor(half * x_arr * (one + _erf_array(x_arr / sqrt_two)))


def _cast(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``cast``: astype to the target dtype (attr normalized to a name string)."""
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    target = np.dtype(op.attributes["dtype"])
    if target.kind in _UNSUPPORTED_KINDS:
        raise core.BackendError(
            f"op 'cast': unsupported target dtype {target} — the numpy "
            "backend does not support object/string/void/structured dtypes"
        )
    return core.Tensor(x_arr.astype(target))


def _select(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``select``: numpy ``where`` semantics — all three operands broadcast;
    both branches are always evaluated (SSA)."""
    pred, on_true, on_false = operands
    pred_arr, true_arr, false_arr = pred.numpy(), on_true.numpy(), on_false.numpy()
    _check_supported(op.name, pred_arr, true_arr, false_arr)
    _check_broadcast(op.name, pred_arr.shape, true_arr.shape, false_arr.shape)
    return core.Tensor(np.where(pred_arr, true_arr, false_arr))


def _broadcast(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``broadcast``: expand the operand to the target shape.

    The ``shape`` attribute (tuple of int/Dim/DimExpr/None) is evaluated
    against ``ctx.bindings`` (``shapes.evaluate_dim_expr``); a ``None``
    (runtime-dynamic) entry cannot be evaluated at run time and raises
    ``core.ShapeError`` naming the op.
    """
    (x,) = operands
    x_arr = x.numpy()
    _check_supported(op.name, x_arr)
    concrete = []
    for i, dim in enumerate(op.attributes["shape"]):
        if dim is None:
            raise core.ShapeError(
                f"op 'broadcast': target dim {i} is None (runtime-dynamic) "
                "and cannot be evaluated at run time"
            )
        concrete.append(evaluate_dim_expr(dim, ctx.bindings))
    try:
        out = np.broadcast_to(x_arr, tuple(concrete))
    except ValueError as exc:
        # A symbolic target/operand mismatch that only manifests at run time
        # (infer_broadcast_to defers symbolic conflicts) — surface cleanly.
        raise core.ShapeError(
            f"op 'broadcast': operand shape {tuple(x_arr.shape)} cannot be "
            f"broadcast to target shape {tuple(concrete)}: {exc}"
        ) from exc
    return core.Tensor(out)


def _stop_gradient(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``stop_gradient``: identity passthrough — the operand Tensor unchanged."""
    (x,) = operands
    return x


#: op name -> numpy ufunc for the plain two-operand broadcasting ops.
_BINARY_OPS = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "power": np.power,
    "remainder": np.remainder,
    "logical_and": np.logical_and,
    "logical_or": np.logical_or,
    "bitwise_and": np.bitwise_and,
    "bitwise_or": np.bitwise_or,
    "bitwise_xor": np.bitwise_xor,
    "bitwise_left_shift": np.left_shift,
    "bitwise_right_shift": np.right_shift,
    "equal": np.equal,
    "not_equal": np.not_equal,
    "less": np.less,
    "less_equal": np.less_equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
}

#: op name -> numpy ufunc for the plain one-operand ops.
_UNARY_OPS = {
    "abs": np.abs,
    "negate": np.negative,
    "square": np.square,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log1p": np.log1p,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "acos": np.arccos,
    "tanh": np.tanh,
    "sign": np.sign,
    "logical_not": np.logical_not,
}
def _rounding(np_fn: Callable[..., Any]) -> Callable[..., core.Tensor]:
    """Kernel factory for the rounding family (``floor``/``ceil``/``round``).
    Numpy defines no rounding of complex numbers (raises TypeError); the etl
    contract requires an explicit ``core.BackendError`` naming the op instead
    (matching the ``_max_min`` complex-rejection precedent).
    """
    def kernel(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
        (x,) = operands
        x_arr = x.numpy()
        _check_supported(op.name, x_arr)
        if x_arr.dtype.kind == "c":
            raise core.BackendError(
                f"op '{op.name}': complex input is not supported (numpy "
                "defines no rounding for complex numbers)"
            )
        return core.Tensor(np.asarray(np_fn(x_arr)))
    return kernel


def register_kernels(table: dict) -> None:
    """Register this module's elementwise kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    for name, np_fn in _BINARY_OPS.items():
        table[name] = _binary(np_fn)
    table["maximum"] = _max_min(np.maximum)
    table["minimum"] = _max_min(np.minimum)
    for name, np_fn in _UNARY_OPS.items():
        table[name] = _unary(np_fn)
    table["erf"] = _erf
    table["sigmoid"] = _sigmoid
    table["relu"] = _relu
    table["gelu"] = _gelu
    table["floor"] = _rounding(np.floor)
    table["ceil"] = _rounding(np.ceil)
    table["round"] = _rounding(np.round)
    table["cast"] = _cast
    table["select"] = _select
    table["broadcast"] = _broadcast
    table["stop_gradient"] = _stop_gradient
