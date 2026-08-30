"""etl.ops — frontend tensor operations.

Every numerical op that can appear in an EvoXIR graph. This module defines
the full public op surface (85 functions), the unified operand/construction/
inference semantics (see this directory's ``CONTEXT.md`` — binding), and the
``SymbolicTensor`` operator handlers registered into ``etl.core`` at import
time.

Import-time side effect: ``_registration.register_operator_handlers()``
populates ``core``'s operator-hook dict so ``SymbolicTensor`` arithmetic
(``x + y``, ``x @ y``, ``x[i]``, ...) works without import cycles. The
canonical op definitions (op names, arities, attributes, effects) live in
``etl.ir``'s registry — ``ops`` consumes them via ``ir.opdef()`` and keeps no
parallel op-definition table.

Unified semantics (binding, summarized — see CONTEXT.md for the full rules):

- Operands: ``SymbolicTensor`` or Python scalars (auto-promoted to 0-d
  constant ops). A concrete ``Tensor`` operand raises ``TraceError`` (make it
  an explicit input, use ``etl.constant``, or use ``etl.evaluate`` — there is
  no eager mode). Calling any op outside an active trace raises
  ``TraceError``.
- Construction: each call builds an IR op into the active builder
  (``trace.current_builder()``) with a call-site ``Location`` attached.
- Inference: numpy dtype promotion (scalars weakly per NEP 50); symbolic
  broadcasting via ``DimExpr.max``; static rules per category are documented
  in each function's docstring.
"""
from __future__ import annotations

# Submodules first (they populate the operator-handler mapping).
from . import _registration
from . import _utils  # noqa: F401  (shared helpers)
from . import comparison
from . import constant
from . import elementwise
from . import indexing
from . import linalg
from . import random  # noqa: F401  (etl.random frontend — re-exported via etl/random.py)
from . import reductions

# Elementwise arithmetic / math / bitwise.
from .elementwise import (  # noqa: F401
    abs, acos, add, bitwise_and, bitwise_or, bitwise_xor, cast, ceil, cos,
    divide, erf, exp, floor, gelu, log, log1p, maximum, minimum, multiply,
    negate, power, relu, remainder, round, sign, sigmoid, sin, sqrt, square,
    subtract, tan, tanh,
)

# Comparison / logical / selection.
from .comparison import (  # noqa: F401
    equal, greater, greater_equal, less, less_equal, logical_and,
    logical_not, logical_or, not_equal, select,
)

# Indexing / shape manipulation.
from .indexing import (  # noqa: F401
    broadcast, concatenate, gather, pad, reshape, scatter, slice, transpose,
)

# Reductions (reduce_* + user-facing sugar).
from .reductions import (  # noqa: F401
    argmax, argmin, max, mean, min, prod, reduce_max, reduce_mean, reduce_min,
    reduce_prod, reduce_sum, sum,
)

# Linalg.
from .linalg import (  # noqa: F401
    cholesky, conv, cumsum, diagonal, dot, eigh, matrix_exp, matrix_rank,
    norm, qr, solve, sort, svd, trace, tril, triu,
)

# Statistics (documented compositions over ordinary ops — no dedicated IR ops).
from . import stats  # noqa: F401
from .stats import median, nansum, std, var  # noqa: F401

# Constants / escape hatches.
from .constant import constant, runtime_call, stop_gradient  # noqa: F401

# Import-time side effect: wire SymbolicTensor operator handlers into core.
_registration.register_operator_handlers()

__all__ = [
    # elementwise
    "add", "subtract", "multiply", "divide", "power", "remainder",
    "maximum", "minimum", "abs", "negate", "square", "sqrt", "exp", "log",
    "log1p", "sin", "cos", "tan", "acos", "floor", "ceil", "round", "tanh",
    "sigmoid", "relu", "gelu", "erf", "sign", "cast", "bitwise_and",
    "bitwise_or", "bitwise_xor",
    # comparison / logical / selection
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "select",
    # indexing / shape manipulation
    "broadcast", "reshape", "transpose", "slice", "gather", "scatter",
    "concatenate", "pad",
    # reductions
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod",
    "sum", "max", "min", "mean", "prod", "argmax", "argmin",
    # linalg
    "dot", "conv", "tril", "triu", "cumsum", "solve", "sort", "diagonal",
    "trace", "norm", "eigh", "cholesky", "qr", "matrix_rank", "svd",
    "matrix_exp",
    # statistics
    "var", "std", "median", "nansum",
    # constants / escape hatches
    "constant", "runtime_call", "stop_gradient",
]
