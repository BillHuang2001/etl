"""Reduction kernels.

Covers these ops (implement in the implementation phase):
reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_prod (explicit-axes
frontend ops) and their named sugar sum, max, min, mean, prod, plus argmax,
argmin.

Design notes (binding, parent CONTEXT.md):
- Axes and keepdims arrive as op attributes (graph constants); runtime shapes
  come from the ops-level inference rules evaluated with concrete dim
  bindings (``../shapes.py``) — the backend carries no second copy of shape
  rules.
- dtype handling is exactly what ops defines (no hidden promotion; unsupported
  combinations raise ``core.BackendError`` naming the op).
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's reduction kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    raise NotImplementedError(
        "architecture stub: implement reduction kernels + registration in the implementation phase"
    )
