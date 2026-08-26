"""Elementwise arithmetic/math sugar of `etl.numpy` (enp).

Every function here is 1:1 sugar over `etl.ops`: it builds exactly the same
IR (same op kind, operands, attrs) as the mapped ops call, into the active
builder. Concrete `Tensor` args raise the same `TraceError` as ops — there
is no eager numpy fallback. `clip` is a documented composition (see
CONTEXT.md). All functions are implemented pure-sugar forwards: no extra
validation, dtype logic, or error handling — ops' errors surface unchanged.
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "abs", "add", "subtract", "multiply", "divide", "power",
    "maximum", "minimum", "negative", "square", "sqrt", "exp", "log",
    "sin", "cos", "tanh", "sign", "clip", "astype",
]


def abs(x):  # noqa: A001 — shadows builtin by design (numpy.abs)
    """numpy.abs(x) → ops.abs(x)."""
    return ops.abs(x)


def add(a, b):
    """numpy.add → ops.add(a, b). No ufunc kwargs (out/where/dtype) in v1."""
    return ops.add(a, b)


def subtract(a, b):
    """numpy.subtract → ops.subtract(a, b)."""
    return ops.subtract(a, b)


def multiply(a, b):
    """numpy.multiply → ops.multiply(a, b)."""
    return ops.multiply(a, b)


def divide(a, b):
    """numpy.divide → ops.divide(a, b)."""
    return ops.divide(a, b)


def power(a, b):
    """numpy.power → ops.power(a, b)."""
    return ops.power(a, b)


def maximum(a, b):
    """numpy.maximum → ops.maximum(a, b)."""
    return ops.maximum(a, b)


def minimum(a, b):
    """numpy.minimum → ops.minimum(a, b)."""
    return ops.minimum(a, b)


def negative(x):
    """numpy.negative → ops.negate(x)."""
    return ops.negate(x)


def square(x):
    """numpy.square → ops.square(x)."""
    return ops.square(x)


def sqrt(x):
    """numpy.sqrt → ops.sqrt(x)."""
    return ops.sqrt(x)


def exp(x):
    """numpy.exp → ops.exp(x)."""
    return ops.exp(x)


def log(x):
    """numpy.log → ops.log(x)."""
    return ops.log(x)


def sin(x):
    """numpy.sin → ops.sin(x)."""
    return ops.sin(x)


def cos(x):
    """numpy.cos → ops.cos(x)."""
    return ops.cos(x)


def tanh(x):
    """numpy.tanh → ops.tanh(x)."""
    return ops.tanh(x)


def sign(x):
    """numpy.sign → ops.sign(x)."""
    return ops.sign(x)


def clip(a, a_min, a_max):
    """numpy.clip → ops.maximum(ops.minimum(a, a_max), a_min).

    None bounds skip that side (numpy semantics): a_min=None applies the
    upper bound only (minimum); a_max=None applies the lower bound only
    (maximum). Both bounds None raises ValueError (numpy parity).
    """
    if a_min is None and a_max is None:
        raise ValueError(
            "clip: at least one of a_min or a_max must be specified "
            "(both are None)"
        )
    if a_min is None:
        return ops.minimum(a, a_max)
    if a_max is None:
        return ops.maximum(a, a_min)
    return ops.maximum(ops.minimum(a, a_max), a_min)


def astype(a, dtype):
    """numpy.ndarray.astype as a function → ops.cast(a, dtype).

    numpy has no top-level astype; provided for enp ergonomics.
    """
    return ops.cast(a, dtype)
