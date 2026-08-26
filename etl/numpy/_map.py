"""Canonical enp → ops mapping data (architecture artifact).

Single source of truth for the enp ≡ ops equivalence contract: every 1:1 enp
function must build exactly the IR produced by its mapped ops call. Values
are "module.attr" strings so this module never imports `etl.ops` and cannot
break on contract drift (Phase 2 resolves them to callables).

Composed functions (`clip`, `stack`, `split`, `where`, `cumsum(axis=None)`)
are documented as explicit op sequences in CONTEXT.md, not here.
"""

from __future__ import annotations

__all__ = ["ELEMENTWISE", "LOGIC", "SHAPE", "REDUCTIONS", "LINALG"]

# 1:1 elementwise mappings (enp name → ops name)
ELEMENTWISE = {
    "abs": "ops.abs",
    "add": "ops.add",
    "subtract": "ops.subtract",
    "multiply": "ops.multiply",
    "divide": "ops.divide",
    "power": "ops.power",
    "maximum": "ops.maximum",
    "minimum": "ops.minimum",
    "negative": "ops.negate",
    "square": "ops.square",
    "sqrt": "ops.sqrt",
    "exp": "ops.exp",
    "log": "ops.log",
    "sin": "ops.sin",
    "cos": "ops.cos",
    "tanh": "ops.tanh",
    "sign": "ops.sign",
    "astype": "ops.cast",
}

# 1:1 comparison/logic mappings
LOGIC = {
    "equal": "ops.equal",
    "not_equal": "ops.not_equal",
    "less": "ops.less",
    "less_equal": "ops.less_equal",
    "greater": "ops.greater",
    "greater_equal": "ops.greater_equal",
    "logical_and": "ops.logical_and",
    "logical_or": "ops.logical_or",
    "logical_not": "ops.logical_not",
}

# 1:1 shape/manipulation mappings (composed entries documented in CONTEXT.md)
SHAPE = {
    "reshape": "ops.reshape",
    "transpose": "ops.transpose",
    "broadcast_to": "ops.broadcast",
    "concatenate": "ops.concatenate",
    "pad": "ops.pad",
    "tril": "ops.tril",
    "triu": "ops.triu",
}

# 1:1 reduction mappings (dtype= support composes ops.cast; axis=None expands
# to all axes at trace time — see CONTEXT.md design decisions)
REDUCTIONS = {
    "sum": "ops.sum",
    "mean": "ops.mean",
    "prod": "ops.prod",
    "max": "ops.max",
    "min": "ops.min",
    "argmax": "ops.argmax",
    "argmin": "ops.argmin",
    "cumsum": "ops.cumsum",
}

# Linear algebra mappings (dot ≡ matmul in v1 — documented deviation)
LINALG = {
    "matmul": "ops.dot",
    "dot": "ops.dot",
    "solve": "ops.solve",
}
