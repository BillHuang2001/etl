"""Operator-handler registration into ``etl.core``.

Populates the ``core`` operator-hook dict at ``etl.ops`` import time so
``SymbolicTensor.__add__``, ``__getitem__``, ... resolve to the op functions
defined in this package WITHOUT import cycles: ``core`` never imports
``ops``; ``ops`` imports ``core`` once and registers its handlers.

Handler protocol (binding — see CONTEXT.md "Operator-handler protocol"):

- Binary kinds (``add``, ``sub``, ``mul``, ``matmul``, ``truediv``, ``pow``,
  ``lt``, ``le``, ``gt``, ``ge``, ``eq``): ``handler(left, right) ->
  SymbolicTensor`` where ``left`` is the ``SymbolicTensor`` operand and
  ``right`` is a ``SymbolicTensor`` or Python scalar. ``core`` routes BOTH
  ``x + y`` and reflected ``y + x`` (scalar on the left) through the same
  handler with the tensor as first argument.
- ``neg``: ``handler(x) -> SymbolicTensor``.
- ``getitem``: ``handler(x, key) -> SymbolicTensor`` where ``key`` is a
  STATIC int/``builtins.slice``/tuple thereof; symbolic indices raise
  ``TraceError`` inside the handler.

Registration is per-kind and idempotent (last registration wins); the
mapping lives in :data:`OPERATOR_HANDLERS` so tests can assert completeness
(one entry per kind listed in ``etl/CONTEXT.md``).
"""
from __future__ import annotations

from typing import Callable, Dict

from etl import core

from . import comparison
from . import elementwise
from . import indexing
from . import linalg

__all__ = ["OPERATOR_HANDLERS", "register_operator_handlers"]

#: kind → handler. Kinds match the hook-dict names declared in the root
#: ``etl/CONTEXT.md`` contract (``add``/``sub``/``mul``/``matmul``/
#: ``getitem``/``truediv``/``pow``/``neg``/``lt``/``le``/``gt``/``ge``/
#: ``eq``).
OPERATOR_HANDLERS: Dict[str, Callable] = {
    "add": elementwise.add,
    "sub": elementwise.subtract,
    "mul": elementwise.multiply,
    "matmul": linalg.dot,
    "truediv": elementwise.divide,
    "pow": elementwise.power,
    "neg": elementwise.negate,
    "lt": comparison.less,
    "le": comparison.less_equal,
    "gt": comparison.greater,
    "ge": comparison.greater_equal,
    "eq": comparison.equal,
    "getitem": indexing.getitem,
}


def register_operator_handlers() -> None:
    """Register every entry of :data:`OPERATOR_HANDLERS` into ``core`` via
    ``core.register_operator_handlers(kind, handler)`` (one call per kind).

    Invoked once at ``etl.ops`` import time (see ``etl/ops/__init__.py``).
    Idempotent: re-calling overwrites the same entries.
    """
    for kind, handler in OPERATOR_HANDLERS.items():
        core.register_operator_handlers(kind, handler)
