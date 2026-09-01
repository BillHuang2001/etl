"""Custom-computation kernels: constant, runtime_call, block_call.

Implemented kernels:

- ``constant``: returns ``core.Tensor(op.attributes["value"])`` — the payload
  is a numpy ndarray. ``ops.constant`` already SNAPSHOT-COPIED the data at
  trace time (and deserialization produces a fresh array), so this kernel
  does NOT copy again.

- ``runtime_call``: executes the Python callback synchronously at the op's
  position — a DOCUMENTED SYNC POINT (no async execution in v1). This is the
  only place where Python executes "for" the graph; the op's position in
  block op order is part of the effect ordering. The op's ``callback``
  attribute is a STRING registry id resolved via ``ctx.resolve_callback``
  (``etl.ops.constant._get_callback``); operands are passed to the callback
  as numpy arrays (``t.numpy()``).

  PERSISTENCE CONTRACT: artifacts containing ``runtime_call`` require the
  SAME callback registrations in the process at load/run time — callbacks are
  NEVER embedded in artifacts (the op carries only the string id); a missing
  registration raises ``core.BackendError`` naming the id.

- ``external_call``: dispatches a NAMED kernel from the ``etl.external``
  registry (``etl.register_external_kernel`` — kernel-agnostic; the same
  registry serves round-2 compiler-adapter host-dispatch). The op's ``name``
  attribute is the stable user-chosen string; the registry is resolved LAZILY
  at run time (import acyclicity) through
  ``etl.external.get_external_kernel(name, "numpy")`` — the per-backend
  "numpy" slot, with automatic fallback to the default (``None``) slot —
  and an unknown name raises ``core.BackendError`` naming the kernel.
  Outputs are validated against the op's declared ``result_specs`` exactly
  like ``runtime_call``/``block_call`` (shared helpers below).

- ``block_call``: dispatches a registered numpy block impl.

  FINALIZED v1 BLOCK IMPL CALL CONVENTION::

      impl(*numpy_arrays, **static_args) -> ndarray | tuple[ndarray, ...]

  Operands are passed as numpy arrays; ``static_args`` is the op's JSON-able
  static-kwarg dict (absent/empty => {}). Portable decompositions were
  ALREADY inlined at ``lower()`` time by ``NumpyBackend``, so the interpreter
  only ever sees blocks with a registered impl — an unknown/unregistered
  block reaching here => ``core.BackendError`` naming the block (safety net,
  never a fallback path).

Shared output validation (``runtime_call``, ``block_call`` and
``external_call``): callback/impl/kernel outputs are normalized (ndarray ->
[arr]; core.Tensor -> [t]; tuple/list -> list; anything else =>
``core.BackendError`` naming the op), the count must match the declared
``result_specs``, and each output is checked against the op's declared
``result_specs``: entries may be ``ir.ValueType`` objects (the in-memory
trace form, restored after an ``ir`` serialization round-trip) OR plain
``{"dtype": ..., "shape": ...}`` dicts — both forms are handled
defensively. dtype must match exactly (``core.BackendError`` naming the op);
the declared shape is evaluated against ``ctx.bindings`` via
``ctx.evaluate_shape`` (``Dim``/``DimExpr`` dims; ``None`` dims unchecked —
runtime-dynamic) and compared to the actual output shape (mismatch =>
``core.ShapeError``). Outputs are wrapped in ``core.Tensor`` (default CPU
device). The shared implementation lives in
``etl/backends/external_validate.py`` (also used by the iree adapter's
host-dispatch path); this module keeps thin local wrappers only.
"""
from __future__ import annotations

from typing import Any

from etl import core
from etl.backends.external_validate import (
    normalize_results as _shared_normalize,
    validate_outputs as _shared_validate,
)

__all__ = ["register_kernels"]


def _constant(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``constant``: wrap the embedded payload without copying.

    The payload was snapshot-copied by ``ops.constant`` at trace time (and
    deserialization produces a fresh array), so sharing it here is safe.
    """
    return core.Tensor(op.attributes["value"])


# ---------------------------------------------------------------------------
# Output normalization + validation shared by runtime_call / block_call /
# external_call (thin wrappers over etl.backends.external_validate — the
# canonical wording lives there, shared with the iree host-dispatch path)
# ---------------------------------------------------------------------------


def _normalize_results(result: Any, label: str) -> list:
    """Normalize a callback/impl return — see ``external_validate.normalize_results``."""
    return _shared_normalize(result, label)


def _validate_outputs(
    ctx: Any, op: Any, arrays: list, declared_specs: Any, label: str | None = None
) -> list:
    """Validate callback/impl outputs — see ``external_validate.validate_outputs``.

    ``label`` names the operation in error messages (defaults to
    ``"op '<name>'"``); block_call/external_call pass a label that also
    names the block/kernel.
    """
    op_name = label if label is not None else f"op '{op.name}'"
    evaluate_shape = None
    if ctx is not None and hasattr(ctx, "evaluate_shape"):
        evaluate_shape = lambda dim: ctx.evaluate_shape((dim,))[0]
    return _shared_validate(arrays, declared_specs, op_name, evaluate_shape=evaluate_shape)


def _finish(op_name: str, outputs: list):
    """One ``core.Tensor`` for 1-result ops, a tuple for multi-result ops."""
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


# ---------------------------------------------------------------------------
# runtime_call / block_call
# ---------------------------------------------------------------------------


def _runtime_call(ctx: Any, op: Any, operands: tuple):
    """``runtime_call``: execute the registered Python callback synchronously.

    The callback is resolved from the op's string ``callback`` attribute via
    ``ctx.resolve_callback`` (unknown id => ``BackendError`` — artifacts
    require the same callback registrations in the process at run time) and
    receives the operand tensors as numpy arrays. Its return is normalized
    and validated against the declared ``result_specs`` (see module
    docstring).
    """
    callback = ctx.resolve_callback(op.attributes["callback"])
    result = callback(*[t.numpy() for t in operands])
    arrays = _normalize_results(result, f"op '{op.name}'")
    outputs = _validate_outputs(ctx, op, arrays, op.attributes["result_specs"])
    return _finish(op.name, outputs)


def _external_call(ctx: Any, op: Any, operands: tuple):
    """``external_call``: dispatch the named kernel from ``etl.external``.

    Kernel resolution goes through
    ``etl.external.get_external_kernel(name, "numpy")`` (lazy import —
    import acyclicity; the per-backend "numpy" slot, with automatic
    fallback to the default ``None`` slot): an unknown name raises
    ``core.BackendError`` naming the kernel and pointing at
    ``etl.register_external_kernel`` (the registry is never serialized —
    graphs require the same kernel registrations in the process at run
    time). The kernel receives operand tensors as numpy arrays; its return
    is normalized and validated against the declared ``result_specs``
    exactly like ``runtime_call`` (shared helpers).
    """
    from etl.external import get_external_kernel

    name = op.attributes["name"]
    kernel = get_external_kernel(name, "numpy")
    if kernel is None:
        raise core.BackendError(
            f"op 'external_call': no external kernel registered under name "
            f"{name!r} — register it with "
            "etl.register_external_kernel(name, callable) in this process "
            "before running graphs that call it (kernels are never "
            "embedded in artifacts)"
        )
    result = kernel(*[t.numpy() for t in operands])
    label = f"op 'external_call' (kernel {name!r})"
    arrays = _normalize_results(result, label)
    outputs = _validate_outputs(ctx, op, arrays, op.attributes["result_specs"], label)
    return _finish(op.name, outputs)


def _block_call(ctx: Any, op: Any, operands: tuple):
    """``block_call``: dispatch the block's registered numpy impl.

    Impl resolution goes through ``etl.block.registry`` (lazy import —
    import acyclicity): an unknown block name (``BlockError``) or a block
    with no registered numpy impl => ``core.BackendError`` naming the block.
    Portable decompositions were ALREADY inlined at ``lower()`` time, so a
    missing impl here is a safety net, never a fallback. The impl receives
    operand tensors as numpy arrays plus the op's JSON-able ``static_args``
    kwargs; its return is validated exactly like ``runtime_call``.
    """
    from etl.block import BlockError
    from etl.block import registry as block_registry

    name = op.attributes["block_name"]
    try:
        block_registry.get_block(name)
    except BlockError as exc:
        raise core.BackendError(
            f"op 'block_call': unknown block {name!r} — declare it with "
            "etl.block(...) before building graphs that call it"
        ) from exc
    impl = block_registry.get_impl(name, "numpy")
    if impl is None:
        raise core.BackendError(
            f"op 'block_call': no numpy implementation registered for block "
            f"{name!r} — portable decompositions are inlined at lower() "
            "time, so a block reaching the interpreter must have a "
            "registered backend impl (safety net, never a fallback)"
        )
    static_args = op.attributes.get("static_args") or {}
    if not isinstance(static_args, dict):
        raise core.BackendError(
            f"op 'block_call' for block {name!r}: attribute 'static_args' "
            f"must be a dict of static kwargs, got {static_args!r}"
        )
    result = impl(*[t.numpy() for t in operands], **static_args)
    label = f"op 'block_call' (block {name!r})"
    arrays = _normalize_results(result, label)
    outputs = _validate_outputs(ctx, op, arrays, op.attributes["result_specs"], label)
    return _finish(op.name, outputs)


def register_kernels(table: dict) -> None:
    """Register this module's custom-computation kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    table["constant"] = _constant
    table["runtime_call"] = _runtime_call
    table["external_call"] = _external_call
    table["block_call"] = _block_call
