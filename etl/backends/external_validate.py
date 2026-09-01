"""Shared output validation for external-kernel dispatch (numpy + iree).

ONE implementation of the external-kernel output contract, imported by both
the numpy interpreter dispatch (``etl/backends/numpy/kernels/custom.py`` —
``runtime_call`` / ``block_call`` / ``external_call``) and the iree adapter
host-dispatch (``etl/backends/external_split.py`` — the static path), so
every dispatch path validates with the SAME canonical message wording.

Canonical API:

- ``normalize_results(result, label) -> list`` — ``np.ndarray`` or
  ``core.Tensor`` -> ``[x]``; tuple/list -> ``list(...)``; anything else =>
  ``core.BackendError`` naming the call (never a guess).
- ``validate_outputs(arrays, declared_specs, label, evaluate_shape=None)
  -> list[core.Tensor]`` — the count must match ``len(declared_specs)``
  (``core.BackendError``); per output: entries are unwrapped to numpy
  arrays (``core.Tensor`` -> ``.numpy()``; anything else =>
  ``core.BackendError``), specs are extracted defensively (``ir.ValueType``
  objects duck-typed via ``.dtype``/``.shape``; ``{"dtype": ..., "shape":
  ...}`` dict wire form — ``core.BackendError`` for anything else), dtype
  must match exactly (``core.BackendError`` — no silent coercion), and the
  declared shape is compared to the actual shape (``core.ShapeError``).
  ``evaluate_shape`` is an optional per-dim callable ``(dim) -> int``
  resolving symbolic dims against runtime bindings (``None`` -> static
  path, dims used as-is after decoding); ``None`` dims stay ``None``
  (runtime-dynamic, unchecked) on BOTH paths. Valid outputs are wrapped in
  ``core.Tensor`` (default CPU device).

Import acyclicity: this module imports ONLY ``etl.core`` (plus ``numpy``
and ``typing``) — never ``etl.ir`` (specs are duck-typed via attributes) —
so both consumers can import it at module level without cycles.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from etl import core

__all__ = ["normalize_results", "validate_outputs"]


def normalize_results(result: Any, label: str) -> list:
    """Normalize a callback/impl/kernel return into a list of output entries.

    ``ndarray`` -> ``[arr]``; ``core.Tensor`` -> ``[tensor]``; tuple/list ->
    ``list(...)``. Anything else => ``core.BackendError`` naming the call
    (the dispatch path never guesses how to interpret a return value).
    """
    if isinstance(result, np.ndarray) or isinstance(result, core.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    raise core.BackendError(
        f"{label}: callback returned {type(result).__name__}, expected an "
        "ndarray, a core.Tensor, or a tuple/list of them"
    )


def _decode_dim_entry(dim: Any, where: str) -> Any:
    """Defensively normalize one declared-shape dim entry.

    ``None`` / ``int`` / ``core.Dim`` / ``core.DimExpr`` pass through;
    wire-encoded dim dicts (``{"int": n}``, ``{"dim": name, "size": s?}``,
    ``{"expr": {"op", "args"}}`` — the ``ir`` serialization wire forms) are
    decoded back to objects so the runtime shape evaluator can process them.
    Anything else => ``core.BackendError`` naming the call (never a guess).
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


def _extract_spec(spec: Any, where: str) -> tuple:
    """Normalize one declared result spec to ``(np.dtype, shape_tuple)``.

    Accepts ``ir.ValueType`` objects (in-memory trace form; restored after
    an ``ir`` serialization round-trip) and plain
    ``{"dtype": ..., "shape": ...}`` dicts (hand-built attrs — defensive).
    Anything else => ``core.BackendError`` naming the call.
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


def validate_outputs(
    arrays: list,
    declared_specs: Any,
    label: str,
    evaluate_shape: Optional[Callable[[Any], int]] = None,
) -> List[core.Tensor]:
    """Validate callback outputs against the declared result specs.

    Count must match ``len(declared_specs)`` (``core.BackendError`` — the
    caller's ``ir.verify`` guarantees this equals the op's result count);
    per output: the entry is unwrapped to an ndarray (``core.Tensor`` via
    ``.numpy()``; anything else => ``core.BackendError``), dtype must match
    exactly (``core.BackendError`` — no silent coercion), and the declared
    shape is evaluated (``evaluate_shape`` per dim; ``None`` dims unchecked
    — runtime-dynamic) and compared to the actual output shape
    (``core.ShapeError``). Valid outputs are wrapped in ``core.Tensor``
    (default CPU device).

    ``label`` names the call in error messages (e.g. the op + kernel name).
    """
    if len(arrays) != len(declared_specs):
        raise core.BackendError(
            f"{label}: callback produced {len(arrays)} "
            f"output(s), expected {len(declared_specs)}"
        )
    outputs: List[core.Tensor] = []
    for i, (entry, spec) in enumerate(zip(arrays, declared_specs)):
        where = f"{label} output {i}"
        if isinstance(entry, core.Tensor):
            entry = entry.numpy()
        if not isinstance(entry, np.ndarray):
            raise core.BackendError(
                f"{label}: callback output {i} is "
                f"{type(entry).__name__}, expected an ndarray or a core.Tensor"
            )
        spec_dtype, spec_shape = _extract_spec(spec, where)
        if entry.dtype != spec_dtype:
            raise core.BackendError(
                f"{where}: callback returned dtype {entry.dtype}, declared "
                f"result dtype {spec_dtype} — no silent dtype coercion"
            )
        expected_list = []
        for dim in tuple(spec_shape):
            if dim is None:
                expected_list.append(None)
                continue
            decoded = _decode_dim_entry(dim, where)
            expected_list.append(
                evaluate_shape(decoded) if evaluate_shape is not None else decoded
            )
        expected = tuple(expected_list)
        actual = tuple(int(d) for d in entry.shape)
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
        outputs.append(core.Tensor(entry))
    return outputs
