"""Elementwise kernels (arith / activations / comparisons / select / cast / broadcast).

Covers these ops (implement in the implementation phase):
arith: add, subtract, multiply, divide, power, remainder, maximum, minimum;
activations: abs, negate, square, sqrt, exp, log, log1p, sin, cos, tan, tanh,
sigmoid, relu, gelu, erf, sign;
bitwise/logical: bitwise_and, bitwise_or, bitwise_xor, logical_and, logical_or,
logical_not;
cast/comparisons: cast, equal, not_equal, less, less_equal, greater,
greater_equal;
structure: select, broadcast, stop_gradient (identity passthrough — returns
the operand unchanged).

Design notes (binding, parent CONTEXT.md):
- dtypes map 1:1 between numpy and etl; kernels validate supported dtype
  combinations and raise ``core.BackendError`` (naming the op) rather than
  silently coercing. No promotion beyond what ops defines.
- ``relu``/``gelu``/``square`` are frontend ops that may arrive here directly
  (the numpy backend executes what the frontend produced); the StableHLO
  exporter's decompositions are unrelated to interpreter behavior.
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's elementwise kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.

    Not yet implemented (follow-up agent): registers nothing so
    ``register_all()`` assembles; the category fills in per-op kernels here.
    """
    return None
