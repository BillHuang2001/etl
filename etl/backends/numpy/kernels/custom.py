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
  at run time (import acyclicity) and an unknown name raises
  ``core.BackendError`` naming the kernel. Outputs are validated against the
  op's declared ``result_specs`` exactly like ``runtime_call``/``block_call``
  (shared helpers below).

- ``block_call``: dispatches a registered numpy block impl.

  FINALIZED v1 BLOCK IMPL CALL CONVENTION::

      impl(*numpy_arrays, **static_args) -> ndarray | tuple[ndarray, ...]

  Operands are passed as numpy arrays; ``static_args`` is the op's JSON-able
  static-kwarg dict (absent/empty => {}). Portable decompositions were
  ALREADY inlined at ``lower()`` time by ``NumpyBackend``, so the interpreter
  only ever sees blocks with a registered impl — an unknown/unregistered
  block reaching here => ``core.BackendError`` naming the block (safety net,
  never a fallback path).

Shared output validation (``runtime_call`` and ``block_call``): callback/impl
outputs are normalized (ndarray -> [arr]; core.Tensor -> [t]; tuple/list ->
list; anything else => ``core.BackendError`` naming the op), the count must
match ``len(op.results)``, and each output is checked against the op's
declared ``result_specs``: entries may be ``ir.ValueType`` objects (the
in-memory trace form, restored after an ``ir`` serialization round-trip) OR
plain ``{"dtype": ..., "shape": ...}`` dicts — both forms are handled
defensively. dtype must match exactly (``core.BackendError`` naming the op);
the declared shape is evaluated against ``ctx.bindings`` via
``ctx.evaluate_shape`` (``Dim``/``DimExpr`` dims; ``None`` dims unchecked —
runtime-dynamic) and compared to the actual output shape (mismatch =>
``core.ShapeError``). Outputs are wrapped in ``core.Tensor`` (default CPU
device).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]


def _constant(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``constant``: wrap the embedded payload without copying.

    The payload was snapshot-copied by ``ops.constant`` at trace time (and
    deserialization produces a fresh array), so sharing it here is safe.
    """
    return core.Tensor(op.attributes["value"])


# ---------------------------------------------------------------------------
# Output normalization + validation shared by runtime_call / block_call
# ---------------------------------------------------------------------------


def _normalize_results(result: Any, label: str) -> list:
    """Normalize a callback/impl return into a list of output entries.

    ``ndarray`` -> ``[arr]``; ``core.Tensor`` -> ``[tensor]``; tuple/list ->
    ``list(...)``. Anything else => ``core.BackendError`` naming the op (the
    interpreter never guesses how to interpret a return value).
    """
    if isinstance(result, np.ndarray) or isinstance(result, core.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    raise core.BackendError(
        f"{label}: callback returned {type(result).__name__}, expected an "
        "ndarray, a core.Tensor, or a tuple/list of them"
    )


def _as_ndarray(entry: Any, label: str, index: int) -> np.ndarray:
    """One output entry -> numpy array (``core.Tensor`` unwrapped, no copy)."""
    if isinstance(entry, core.Tensor):
        entry = entry.numpy()
    if not isinstance(entry, np.ndarray):
        raise core.BackendError(
            f"{label}: callback output {index} is "
            f"{type(entry).__name__}, expected an ndarray or a core.Tensor"
        )
    return entry


def _decode_dim_entry(dim: Any, where: str) -> Any:
    """Defensively normalize one declared-shape dim entry.

    ``None`` / ``int`` / ``core.Dim`` / ``core.DimExpr`` pass through;
    wire-encoded dim dicts (``{"int": n}``, ``{"dim": name, "size": s?}``,
    ``{"expr": {"op", "args"}}`` — the ``ir`` serialization wire forms) are
    decoded back to objects so ``ctx.evaluate_shape`` can evaluate them.
    Anything else => ``core.BackendError`` naming the op (never a guess).
    """
    if dim is None or isinstance(dim, (core.Dim, core.DimExpr)):
        return dim
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if isinstance(dim, dict):
        if "int" in dim:
            value = dim["int"]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        elif "dim" in dim:
            name = dim["dim"]
            if isinstance(name, str):
                size = dim.get("size")
                return core.Dim(name, size=size)
        elif "expr" in dim:
            expr = dim["expr"]
            if isinstance(expr, dict) and isinstance(expr.get("op"), str):
                args = expr.get("args")
                if isinstance(args, (tuple, list)) and len(args) == 2:
                    return core.DimExpr(
                        expr["op"],
                        _decode_dim_entry(args[0], where),
                        _decode_dim_entry(args[1], where),
                    )
    raise core.BackendError(
        f"{where}: cannot interpret declared shape dim {dim!r} — expected "
        "an int, a Dim, a DimExpr, or their wire encodings"
    )


def _evaluate_declared_shape(
    ctx: Any, shape: Any, where: str
) -> tuple:
    """Evaluate a declared result shape against ``ctx.bindings``.

    Per-dim evaluation via ``ctx.evaluate_shape`` (``Dim``/``DimExpr``
    entries resolve against the runtime dim bindings); ``None`` dims stay
    ``None`` — runtime-dynamic, unchecked.
    """
    evaluated = []
    for dim in tuple(shape):
        if dim is None:
            evaluated.append(None)
            continue
        evaluated.append(ctx.evaluate_shape((_decode_dim_entry(dim, where),))[0])
    return tuple(evaluated)


def _extract_spec(spec: Any, where: str) -> tuple:
    """Normalize one declared result spec to ``(np.dtype, shape_tuple)``.

    Accepts ``ir.ValueType`` objects (in-memory trace form; restored after
    an ``ir`` serialization round-trip) and plain
    ``{"dtype": ..., "shape": ...}`` dicts (hand-built attrs — defensive).
    Anything else => ``core.BackendError`` naming the op.
    """
    if isinstance(spec, dict):
        try:
            spec_dtype = core.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
        except (KeyError, TypeError, core.DTypeError) as exc:
            raise core.BackendError(
                f"{where}: result spec must be a "
                "{'dtype': ..., 'shape': ...} dict, got {spec!r}"
            ) from exc
        return spec_dtype, shape
    try:
        return spec.dtype, tuple(spec.shape)
    except AttributeError as exc:
        raise core.BackendError(
            f"{where}: cannot interpret result spec {spec!r} — expected an "
            "ir.ValueType or a {'dtype': ..., 'shape': ...} dict"
        ) from exc


def _validate_outputs(
    ctx: Any, op: Any, arrays: list, declared_specs: Any, label: str | None = None
) -> list:
    """Validate callback/impl outputs against the op's declared result_specs.

    Count must match ``len(op.results)`` and ``len(declared_specs)``
    (``BackendError`` naming the op); per output: dtype must match exactly
    (``BackendError`` — no silent coercion), the declared shape is evaluated
    against ``ctx.bindings`` (``None`` dims unchecked) and compared to the
    actual output shape (``ShapeError``). Valid outputs are wrapped in
    ``core.Tensor`` (default CPU device).

    ``label`` names the operation in error messages (defaults to
    ``"op '<name>'"``); block_call passes a label that also names the block.
    """
    op_name = label if label is not None else f"op '{op.name}'"
    if len(arrays) != len(op.results):
        raise core.BackendError(
            f"{op_name}: callback produced {len(arrays)} "
            f"output(s), expected {len(op.results)}"
        )
    if len(arrays) != len(declared_specs):
        raise core.BackendError(
            f"{op_name}: {len(arrays)} callback output(s) but "
            f"{len(declared_specs)} declared result_specs"
        )
    outputs = []
    for i, (entry, spec) in enumerate(zip(arrays, declared_specs)):
        where = f"{op_name} output {i}"
        arr = _as_ndarray(entry, op_name, i)
        spec_dtype, spec_shape = _extract_spec(spec, where)
        if arr.dtype != spec_dtype:
            raise core.BackendError(
                f"{where}: callback returned dtype {arr.dtype}, declared "
                f"result dtype {spec_dtype} — no silent dtype coercion"
            )
        expected = _evaluate_declared_shape(ctx, spec_shape, where)
        actual = tuple(int(d) for d in arr.shape)
        if len(expected) != len(actual):
            raise core.ShapeError(
                f"{where}: callback returned shape {actual} (rank "
                f"{len(actual)}), declared rank {len(expected)}"
            )
        for dim_i, (exp, act) in enumerate(zip(expected, actual)):
            if exp is not None and exp != act:
                raise core.ShapeError(
                    f"{where}: callback returned shape {actual}, declared "
                    f"shape {expected} — mismatch at dim {dim_i} "
                    f"({act} != {exp})"
                )
        outputs.append(core.Tensor(arr))
    return outputs


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

    Kernel resolution goes through ``etl.external.get_external_kernel``
    (lazy import — import acyclicity): an unknown name raises
    ``core.BackendError`` naming the kernel and pointing at
    ``etl.register_external_kernel`` (the registry is never serialized —
    graphs require the same kernel registrations in the process at run
    time). The kernel receives operand tensors as numpy arrays; its return
    is normalized and validated against the declared ``result_specs``
    exactly like ``runtime_call`` (shared helpers).
    """
    from etl.external import get_external_kernel

    name = op.attributes["name"]
    kernel = get_external_kernel(name)
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
