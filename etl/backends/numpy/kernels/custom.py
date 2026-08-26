"""Custom-computation kernels: runtime_call and block_call dispatch.

Covers these ops (implement in the implementation phase):
- runtime_call: executes the Python callback synchronously at the op's
  position — a DOCUMENTED SYNC POINT (no async execution in v1). Callback
  outputs must match the op's declared result specs; mismatches =>
  ``core.BackendError`` (never silent coercion).
- block_call: dispatches a registered numpy block impl. Portable
  decompositions were ALREADY inlined at ``lower()`` time by
  ``NumpyBackend``, so the interpreter only ever sees blocks with a
  registered impl — an unregistered block reaching here =>
  ``core.BackendError`` naming the block (safety net, not a fallback path).

Design notes (binding, parent CONTEXT.md):
- ``runtime_call`` is the only place where Python executes "for" the graph;
  its position in block op order is part of the effect ordering.
- Block impl registration is wired at import time by
  ``NumpyBackend._register_block_impls`` (lazy ``etl.ops`` import — the sole
  allowed site for that import).
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's custom-computation kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    raise NotImplementedError(
        "architecture stub: implement runtime_call/block_call dispatch + registration in the implementation phase"
    )
