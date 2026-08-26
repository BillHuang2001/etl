"""Linear-algebra kernels.

Covers these ops (implement in the implementation phase):
dot (matmul per ops semantics — batching rules as ops defines),
conv (convolution per ops semantics — stride/padding/dilation from op
attributes, no implicit layout or group changes).

Design notes (binding, parent CONTEXT.md):
- Pure numpy implementations (``numpy.matmul``-style composition is allowed
  only if it reproduces ops-defined semantics exactly — the interpreter is
  the reference for the IR, not a performance target).
- Runtime shapes come from ops-level inference rules with concrete dim
  bindings (``../shapes.py``); dtype mismatches raise ``core.BackendError``
  naming the op.
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's linalg kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    raise NotImplementedError(
        "architecture stub: implement dot/conv kernels + registration in the implementation phase"
    )
