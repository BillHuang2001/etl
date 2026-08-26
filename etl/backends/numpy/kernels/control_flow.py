"""Control-flow kernels: cond / while_loop / scan region execution.

Covers these ops (implement in the implementation phase):
cond, while_loop, scan — interpreted by RECURSIVELY RUNNING region blocks.

Design notes (binding, parent CONTEXT.md):
- Genuinely dynamic runtime control flow: the interpreter evaluates the
  condition/branch regions per execution — the graph is NOT specialized per
  iteration and no region is traced at run time.
- Region execution reuses the same interpreter loop over block op order with
  an inner value environment (region block args bound per iteration); shape
  checks use ops-level inference with concrete dim bindings (``../shapes.py``).
- ``while_loop``/``scan`` carry explicit carry/init values; iteration limits
  are the program's own business (the backend does not invent them).
"""
from __future__ import annotations

__all__ = ["register_kernels"]


def register_kernels(table: dict) -> None:
    """Register this module's control-flow kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``. Control-flow
    kernels receive ``ctx`` (interpreter state) so they can run nested region
    blocks through the same dispatch loop.
    """
    raise NotImplementedError(
        "architecture stub: implement region execution + registration in the implementation phase"
    )
