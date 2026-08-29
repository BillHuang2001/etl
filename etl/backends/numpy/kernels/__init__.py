"""Kernel dispatch table for the numpy interpreter.

Design (binding, parent CONTEXT.md "Numpy backend design"):

- The interpreter loop in ``Interpreter._run_block`` dispatches EVERY op
  through this table: ``kernels.dispatch(op_name)`` — one table, op name ->
  kernel callable. Execution order is always BLOCK OP ORDER (the effect
  ordering); the table only maps op -> implementation.
- Per-category modules split by concern (file-size control, see parent
  CONTEXT.md): ``elementwise``, ``reductions``, ``sparse``, ``indexing``,
  ``linalg``, ``control_flow``, ``collective``, ``custom``. Each exposes
  ``register_kernels(table)``; ``register_all()`` merges them (idempotent —
  repeated calls are no-ops).
- Unknown op at dispatch time => ``core.BackendError`` naming the op — the
  safety net for capability drift; never a silent skip.
- ``block_call`` ops arriving here must have a registered numpy impl: portable
  decompositions were already inlined at ``lower()`` time by ``NumpyBackend``.

THE KERNEL CONTRACT (binding for every category module):

```
kernel(ctx, op, operands) -> core.Tensor | tuple[core.Tensor, ...]
- ctx: KernelContext (etl/backends/numpy/interpreter.py) with .bindings (dim name -> int),
  .rank_context, .module, .run_region(region, arg_tensors) -> list[Tensor],
  .resolve_callback(callback_id) -> callable, .evaluate_shape(shape) -> tuple[int, ...],
  .compute_output_shapes(op, input_shapes, input_dtypes) -> list[tuple[int|None, ...]]
- op: ir.Op — read `op.attributes` (schema per etl/ir/op_defs/*.py), `op.operands`, `op.results`, `op.regions`
- operands: tuple[core.Tensor, ...] — already-computed operands in op order
- return: one core.Tensor for 1-result ops, a tuple for multi-result ops
- kernels must NEVER silently coerce dtypes or invent semantics; the interpreter
  validates outputs against op.results types (dtype must match; symbolic dims
  evaluated via ctx.bindings; None dims unchecked)
```
"""
from __future__ import annotations

from typing import Any, Callable

from etl.core import BackendError

__all__ = ["dispatch", "register_all", "KERNEL_TABLE"]

#: op name -> kernel callable. Populated by register_all(); register_kernels
#: from each category module merges its entries in.
KERNEL_TABLE: dict[str, Callable[..., Any]] = {}

#: Idempotency flag: register_all() assembles the table exactly once.
_registered = False


def dispatch(op_name: str) -> Callable[..., Any]:
    """Look up the kernel for an IR op name.

    Unknown op name => ``core.BackendError`` naming the op (capability-drift
    safety net; the interpreter never silently skips an op). Table population
    happens via ``register_all()`` (called at ``lower()`` and at interpreter
    construction — idempotent).
    """
    kernel = KERNEL_TABLE.get(op_name)
    if kernel is None:
        raise BackendError(
            f"no numpy kernel for op '{op_name}' — capability drift: the "
            "op reached the interpreter without a registered implementation"
        )
    return kernel


def register_all() -> None:
    """Merge the per-category kernel modules into ``KERNEL_TABLE``.

    Idempotent: repeated calls are no-ops (module-level ``_registered``
    flag). Each category module's ``register_kernels(table)`` is responsible
    for its own op-name keys; duplicate keys ACROSS modules are a contract
    violation and raise ``core.BackendError`` naming the collision. On
    failure the shared table is left untouched (each module fills a private
    staging dict) so a retry can succeed.
    """
    global _registered
    if _registered:
        return
    from . import (  # noqa: F401  — category modules (intra-package, acyclic)
        collective,
        control_flow,
        custom,
        elementwise,
        indexing,
        linalg,
        random,
        reductions,
        sparse,
    )

    for module in (
        elementwise,
        reductions,
        sparse,
        indexing,
        linalg,
        control_flow,
        collective,
        custom,
        random,
    ):
        staging: dict[str, Callable[..., Any]] = {}
        module.register_kernels(staging)
        collisions = sorted(set(staging) & set(KERNEL_TABLE))
        if collisions:
            raise BackendError(
                "duplicate kernel registration for op(s): "
                + ", ".join(collisions)
            )
        KERNEL_TABLE.update(staging)
    _registered = True
