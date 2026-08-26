"""etl.numpy — NumPy-like graph namespace (alias `enp`, registered at the
`etl` package level, not here).

Pure sugar over etl.ops: every function builds exactly the same IR as the
mapped ops calls — no new op kinds, no hidden semantics, no eager fallback.
Graph-building namespace: SymbolicTensors in, SymbolicTensors out; concrete
Tensors raise TraceError (same as ops). See CONTEXT.md for the full mapping
table, design decisions, deferrals, and documented numpy deviations.

Implementation complete: all bodies build IR per the mapping table.
"""

from __future__ import annotations

from .elementwise import (
    abs, add, subtract, multiply, divide, power, maximum, minimum,
    negative, square, sqrt, exp, log, sin, cos, tanh, sign, clip, astype,
)
from .logic import (
    equal, not_equal, less, less_equal, greater, greater_equal,
    logical_and, logical_or, logical_not, where,
)
from .shape import (
    reshape, transpose, broadcast_to, expand_dims, squeeze, concatenate,
    stack, split, pad, tril, triu,
)
from .reductions import (
    sum, mean, prod, max, min, argmax, argmin, cumsum,
)
from .creation import zeros, ones, full, empty, arange
from ._linalg import matmul, dot
from . import linalg  # submodule etl.numpy.linalg

__all__ = [
    # elementwise
    "abs", "add", "subtract", "multiply", "divide", "power",
    "maximum", "minimum", "negative", "square", "sqrt", "exp", "log",
    "sin", "cos", "tanh", "sign", "clip", "astype",
    # logic
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "where",
    # shape
    "reshape", "transpose", "broadcast_to", "expand_dims", "squeeze",
    "concatenate", "stack", "split", "pad", "tril", "triu",
    # reductions
    "sum", "mean", "prod", "max", "min", "argmax", "argmin", "cumsum",
    # creation
    "zeros", "ones", "full", "empty", "arange",
    # linalg
    "matmul", "dot", "linalg",
]
