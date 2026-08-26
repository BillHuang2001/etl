"""Custom-computation kernels: constant (now), runtime_call / block_call (later).

Implemented now:
- constant: returns ``core.Tensor(op.attributes["value"])`` — the payload is
  a numpy ndarray. ``ops.constant`` already SNAPSHOT-COPIED the data at trace
  time (and deserialization produces a fresh array), so this kernel does NOT
  copy again.

To be added by the follow-up agent (docstrings kept as the binding design):
- runtime_call: executes the Python callback synchronously at the op's
  position — a DOCUMENTED SYNC POINT (no async execution in v1). Callback
  outputs must match the op's declared result specs; mismatches =>
  ``core.BackendError`` (never silent coercion). Callbacks resolve through
  ``ctx.resolve_callback`` (``etl.ops.constant._get_callback``).
- block_call: dispatches a registered numpy block impl (resolved via
  ``etl.block.registry.get_impl(name, "numpy")``). Portable decompositions
  were ALREADY inlined at ``lower()`` time by ``NumpyBackend``, so the
  interpreter only ever sees blocks with a registered impl — an unregistered
  block reaching here => ``core.BackendError`` naming the block (safety net,
  not a fallback path).

Design notes (binding, parent CONTEXT.md):
- ``runtime_call`` is the only place where Python executes "for" the graph;
  its position in block op order is part of the effect ordering.
"""
from __future__ import annotations

from typing import Any

from etl import core

__all__ = ["register_kernels"]


def _constant(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``constant``: wrap the embedded payload without copying.

    The payload was snapshot-copied by ``ops.constant`` at trace time (and
    deserialization produces a fresh array), so sharing it here is safe.
    """
    return core.Tensor(op.attributes["value"])


def register_kernels(table: dict) -> None:
    """Register this module's custom-computation kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["constant"] = _constant
