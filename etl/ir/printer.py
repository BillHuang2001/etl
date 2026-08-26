"""IR text printing (debugging, logging, and human review).

Example output::

    module @"main" version 1 {
      func @main(%arg0: tensor<BxNxf32> loc("model.py":10:4),
                 %arg1: tensor<Nxf32> loc("model.py":10:18)) -> tensor<Bxf32> {
        %0 = etl.add(%arg0, %arg1) : tensor<BxNxf32> loc("model.py":12:8)
        %1 = etl.reduce_sum(%0) attributes {axes = [1]} : tensor<Bxf32> loc("model.py":13:4)
        etl.return(%1) loc("model.py":13:4")
      }
    }

Format rules: block arguments print as ``%argN``; op results print as ``%N``
(numbered sequentially per function); each op line is
``%r = name(operands) [attributes {...}] : type(s) loc(...)``; nested regions
indent two spaces and render their blocks inline under the op.

ARCHITECTURE PHASE: the body is a ``NotImplementedError`` stub.
"""

from __future__ import annotations

from .module import Module


def pretty_print(module: Module) -> str:
    """Render ``module`` as readable SSA text (format above).

    Raises:
        KeyError: If the module references an unregistered op name.
    """
    raise NotImplementedError("pretty_print: Phase 2 (implementation)")
