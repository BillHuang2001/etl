"""Kernel dispatch table for the numpy interpreter.

Design (binding, parent CONTEXT.md "Numpy backend design"):

- The interpreter loop in ``NumpyExecutable.run`` dispatches EVERY op through
  this table: ``kernels.dispatch(op_name)`` — one table, op name -> kernel
  callable. Execution order is always BLOCK OP ORDER (the effect ordering);
  the table only maps op -> implementation.
- Kernel call signature (convention, finalized with the interpreter loop in
  the implementation phase): ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``
  where ``ctx`` carries interpreter state (runtime shape-dim bindings via
  ``../shapes.py``, the collective executor hook, the runtime-call registry)
  and ``op`` is the ``ir.Op`` (operands are already-computed ``core.Tensor``s).
- Per-category modules split by concern (file-size control, see parent
  CONTEXT.md): ``elementwise``, ``reductions``, ``indexing``, ``linalg``,
  ``control_flow``, ``collective``, ``custom``. Each exposes
  ``register_kernels(table)``; ``register_all()`` merges them.
- Unknown op at dispatch time => ``core.BackendError`` naming the op — the
  safety net for capability drift; never a silent skip.
- ``block_call`` ops arriving here must have a registered numpy impl: portable
  decompositions were already inlined at ``lower()`` time by ``NumpyBackend``.
"""
from __future__ import annotations

from typing import Any, Callable

from etl.core import BackendError

__all__ = ["dispatch", "register_all", "KERNEL_TABLE"]

#: op name -> kernel callable. Populated by register_all(); register_kernels
#: from each category module merges its entries in.
KERNEL_TABLE: dict[str, Callable[..., Any]] = {}


def dispatch(op_name: str) -> Callable[..., Any]:
    """Look up the kernel for an IR op name.

    Unknown op name => ``core.BackendError`` naming the op (capability-drift
    safety net; the interpreter never silently skips an op). Table population
    happens via ``register_all()`` during executable construction.
    """
    raise NotImplementedError(
        "architecture stub: implement table lookup + unknown-op BackendError in the implementation phase"
    )


def register_all() -> None:
    """Merge the per-category kernel modules into ``KERNEL_TABLE``.

    Called once at ``NumpyExecutable`` construction time (implementation
    phase). Each category module's ``register_kernels(table)`` is responsible
    for its own op-name keys; duplicate keys across modules are a contract
    violation (implementation may assert/raise ``BackendError`` on collision).
    """
    from . import (  # noqa: F401  — category modules (intra-package, acyclic)
        collective,
        control_flow,
        custom,
        elementwise,
        indexing,
        linalg,
        reductions,
    )

    for module in (
        elementwise,
        reductions,
        indexing,
        linalg,
        control_flow,
        collective,
        custom,
    ):
        module.register_kernels(KERNEL_TABLE)
