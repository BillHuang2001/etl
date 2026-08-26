"""Elementwise arithmetic/math sugar of `etl.numpy` (enp).

Every function here is 1:1 sugar over `etl.ops`: it builds exactly the same
IR (same op kind, operands, attrs) as the mapped ops call, into the active
builder. Concrete `Tensor` args raise the same `TraceError` as ops — there
is no eager numpy fallback. `clip` is a documented composition (see
CONTEXT.md). Architecture phase: stub bodies raise NotImplementedError.
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
    raise NotImplementedError("enp.abs: architecture stub — maps to ops.abs")


def add(a, b):
    """numpy.add → ops.add(a, b). No ufunc kwargs (out/where/dtype) in v1."""
    raise NotImplementedError("enp.add: architecture stub — maps to ops.add")


def subtract(a, b):
    """numpy.subtract → ops.subtract(a, b)."""
    raise NotImplementedError(
        "enp.subtract: architecture stub — maps to ops.subtract"
    )


def multiply(a, b):
    """numpy.multiply → ops.multiply(a, b)."""
    raise NotImplementedError(
        "enp.multiply: architecture stub — maps to ops.multiply"
    )


def divide(a, b):
    """numpy.divide → ops.divide(a, b)."""
    raise NotImplementedError("enp.divide: architecture stub — maps to ops.divide")


def power(a, b):
    """numpy.power → ops.power(a, b)."""
    raise NotImplementedError("enp.power: architecture stub — maps to ops.power")


def maximum(a, b):
    """numpy.maximum → ops.maximum(a, b)."""
    raise NotImplementedError("enp.maximum: architecture stub — maps to ops.maximum")


def minimum(a, b):
    """numpy.minimum → ops.minimum(a, b)."""
    raise NotImplementedError("enp.minimum: architecture stub — maps to ops.minimum")


def negative(x):
    """numpy.negative → ops.negate(x)."""
    raise NotImplementedError("enp.negative: architecture stub — maps to ops.negate")


def square(x):
    """numpy.square → ops.square(x)."""
    raise NotImplementedError("enp.square: architecture stub — maps to ops.square")


def sqrt(x):
    """numpy.sqrt → ops.sqrt(x)."""
    raise NotImplementedError("enp.sqrt: architecture stub — maps to ops.sqrt")


def exp(x):
    """numpy.exp → ops.exp(x)."""
    raise NotImplementedError("enp.exp: architecture stub — maps to ops.exp")


def log(x):
    """numpy.log → ops.log(x)."""
    raise NotImplementedError("enp.log: architecture stub — maps to ops.log")


def sin(x):
    """numpy.sin → ops.sin(x)."""
    raise NotImplementedError("enp.sin: architecture stub — maps to ops.sin")


def cos(x):
    """numpy.cos → ops.cos(x)."""
    raise NotImplementedError("enp.cos: architecture stub — maps to ops.cos")


def tanh(x):
    """numpy.tanh → ops.tanh(x)."""
    raise NotImplementedError("enp.tanh: architecture stub — maps to ops.tanh")


def sign(x):
    """numpy.sign → ops.sign(x)."""
    raise NotImplementedError("enp.sign: architecture stub — maps to ops.sign")


def clip(a, a_min, a_max):
    """numpy.clip → ops.maximum(ops.minimum(a, a_max), a_min).

    None bounds skip that side (numpy semantics): a_min=None applies the
    maximum bound only; a_max=None applies the minimum bound only.
    """
    raise NotImplementedError(
        "enp.clip: architecture stub — maps to "
        "ops.maximum(ops.minimum(a, a_max), a_min)"
    )


def astype(a, dtype):
    """numpy.ndarray.astype as a function → ops.cast(a, dtype).

    numpy has no top-level astype; provided for enp ergonomics.
    """
    raise NotImplementedError("enp.astype: architecture stub — maps to ops.cast")
