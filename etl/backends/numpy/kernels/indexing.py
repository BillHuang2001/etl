"""Indexing / data-movement kernels.

Covers these ops (implement in the implementation phase):
reshape, transpose, slice, gather, scatter, concatenate, pad.

Design notes (binding, parent CONTEXT.md):
- Slices/indices/axes are op attributes (graph constants) — the interpreter
  resolves them against runtime shapes via the ops-level inference rules
  bound to concrete dims (``../shapes.py``); free symbolic dims =>
  ``core.ShapeError``.
- These kernels are pure data movement on ``core.Tensor`` (numpy arrays in
  v1); they never invent semantics (e.g. implicit broadcasting beyond what
  ops defines).
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's indexing kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    raise NotImplementedError(
        "architecture stub: implement indexing kernels + registration in the implementation phase"
    )
