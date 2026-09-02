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
  arrays (``core.Tensor`` -> materialized via the EXPLICIT transfer API
  ``.to(core.Device("cpu", 0)).numpy()`` — every device->host copy routes
  through ``.to()``; anything else => ``core.BackendError``), specs are
  extracted defensively (``ir.ValueType``
  objects duck-typed via ``.dtype``/``.shape``; ``{"dtype": ..., "shape":
  ...}`` dict wire form — ``core.BackendError`` for anything else), dtype
  must match exactly (``core.BackendError`` — no silent coercion), and the
  declared shape is compared to the actual shape (``core.ShapeError``).
  ``evaluate_shape`` is an optional per-dim callable ``(dim) -> int``
  resolving symbolic dims against runtime bindings (``None`` -> static
  path, dims used as-is after decoding); ``None`` dims stay ``None``
  (runtime-dynamic, unchecked) on BOTH paths. Valid outputs are wrapped in
  ``core.Tensor`` (default CPU device).
- ``normalize_device_results(result, label) -> list`` — the device-kernel
  counterpart of ``normalize_results``: ``np.ndarray`` / ``core.Tensor`` ->
  ``[x]``; payload-ish objects (both ``shape`` and ``dtype`` attributes —
  covers raw iree ``DeviceArray`` and duck payloads) -> ``[x]``;
  tuple/list -> ``list(...)``; anything else => ``core.BackendError``
  naming the call (never a guess).
- ``validate_device_outputs(entries, declared_specs, label,
  wrap_device_result=None) -> list[core.Tensor]`` — METADATA-ONLY
  validation for device-resident kernels: NEVER materializes a host copy
  (never calls ``.numpy()``/``to_host()``). Count must match
  ``len(declared_specs)`` (``core.BackendError`` — same canonical wording
  as ``validate_outputs``); per output: ``core.Tensor`` entries pass
  through, ``np.ndarray`` entries wrap in ``core.Tensor`` (host result —
  the caller stages it back), anything else goes through
  ``wrap_device_result`` when provided (else ``core.Tensor(entry)``); a
  failed wrap (``TypeError``/``core.DeviceError``) re-raises as
  ``core.BackendError`` naming the call and output index — explicit, never
  a guess. Then dtype must match exactly (``core.BackendError`` — no
  silent coercion) and the declared shape must match the actual shape
  (``core.ShapeError``) — the SAME canonical wording as
  ``validate_outputs`` via the shared ``_validate_entry_spec`` helper
  (static-int dims are guaranteed by lower-time, so no ``evaluate_shape``
  here, but wire-form dict specs still decode).

Import acyclicity: this module imports ONLY ``etl.core`` (plus ``numpy``
and ``typing``) — never ``etl.ir`` (specs are duck-typed via attributes)
and never iree (not even lazily — device entries are duck-typed via
``shape``/``dtype``) — so both consumers can import it at module level
without cycles.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import numpy as np

from etl import core

__all__ = [
    "normalize_results",
    "normalize_device_results",
    "validate_outputs",
    "validate_device_outputs",
]


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


def normalize_device_results(result: Any, label: str) -> list:
    """Normalize a DEVICE kernel return into a list of output entries.

    The device-mode counterpart of :func:`normalize_results`: ``ndarray`` /
    ``core.Tensor`` -> ``[x]``; payload-ish objects (anything with BOTH
    ``shape`` and ``dtype`` attributes — covers raw iree ``DeviceArray``
    handles and duck-typed device payloads) -> ``[x]``; tuple/list ->
    ``list(...)``. Anything else => ``core.BackendError`` naming the call
    (the dispatch path never guesses how to interpret a return value).
    """
    if isinstance(result, np.ndarray) or isinstance(result, core.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    if hasattr(result, "shape") and hasattr(result, "dtype"):
        return [result]
    raise core.BackendError(
        f"{label}: callback returned {type(result).__name__}, expected an "
        "ndarray, a core.Tensor, a device payload (shape + dtype "
        "attributes), or a tuple/list of them"
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


def _validate_entry_spec(
    dtype: np.dtype,
    shape: tuple,
    spec: Any,
    where: str,
    evaluate_shape: Optional[Callable[[Any], int]] = None,
) -> None:
    """Shared per-output check: dtype EXACT, then shape EXACT vs ``spec``.

    Used by BOTH :func:`validate_outputs` (host mode) and
    :func:`validate_device_outputs` (device mode) so the canonical message
    wording never drifts. dtype mismatch => ``core.BackendError`` ("no
    silent dtype coercion"); shape rank/dim mismatches => ``core.ShapeError``.
    ``evaluate_shape`` resolves symbolic dims against runtime bindings
    (host mode); ``None`` dims stay ``None`` (runtime-dynamic, unchecked).
    """
    spec_dtype, spec_shape = _extract_spec(spec, where)
    if dtype != spec_dtype:
        raise core.BackendError(
            f"{where}: callback returned dtype {dtype}, declared "
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
    actual = tuple(int(d) for d in shape)
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
    the EXPLICIT transfer API ``.to(core.Device("cpu", 0)).numpy()`` —
    every device->host copy routes through ``.to()``; anything else =>
    ``core.BackendError``), dtype must match exactly
    (``core.BackendError`` — no silent coercion), and the declared
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
            # Host-mode kernel results may in principle be device-resident
            # (a kernel returning a device payload tensor): materialize via
            # the EXPLICIT transfer API — every device->host copy routes
            # through .to(), never an implicit .numpy() on a non-cpu
            # payload (which raises core.DeviceError under the
            # physical-truth semantics). cpu-kind tensors pass through
            # unchanged (a failed transfer raises core.DeviceError).
            entry = entry.to(core.Device("cpu", 0)).numpy()
        if not isinstance(entry, np.ndarray):
            raise core.BackendError(
                f"{label}: callback output {i} is "
                f"{type(entry).__name__}, expected an ndarray or a core.Tensor"
            )
        _validate_entry_spec(entry.dtype, entry.shape, spec, where, evaluate_shape)
        outputs.append(core.Tensor(entry))
    return outputs


def validate_device_outputs(
    entries: list,
    declared_specs: Any,
    label: str,
    wrap_device_result: Optional[Callable[[Any], Any]] = None,
) -> List[core.Tensor]:
    """Validate DEVICE-kernel outputs against the declared result specs.

    METADATA-ONLY validation: NEVER materializes a host copy of a device
    result (never calls ``.numpy()``/``to_host()``) — the adapter itself
    never forces device->host staging. Count must match
    ``len(declared_specs)`` (``core.BackendError`` — the same canonical
    wording as :func:`validate_outputs`); per output: ``core.Tensor``
    entries pass through untouched, ``np.ndarray`` entries wrap in
    ``core.Tensor`` (a host result — the caller stages it back into the
    next segment's inputs), and any other entry goes through
    ``wrap_device_result`` when provided (else ``core.Tensor(entry)`` — a
    duck-typed device payload); a failed wrap (``TypeError`` /
    ``core.DeviceError`` raised by the wrap callback) re-raises as
    ``core.BackendError`` naming the call and output index — explicit,
    never a guess. Then dtype must match exactly (``core.BackendError`` —
    no silent coercion) and the declared shape must match the actual shape
    (``core.ShapeError``) — the SAME canonical wording as
    :func:`validate_outputs` via the shared ``_validate_entry_spec``
    helper. Static-int dims are guaranteed by lower-time (no symbolic dims
    in the plan), so there is no ``evaluate_shape`` here; wire-form dict
    specs still decode defensively.

    ``label`` names the call in error messages (e.g. the op + kernel name);
    ``wrap_device_result`` is the caller's payload-wrapping callback
    (e.g. the iree adapter wraps raw ``DeviceArray`` results with the run's
    ``core.Device``).
    """
    if len(entries) != len(declared_specs):
        raise core.BackendError(
            f"{label}: callback produced {len(entries)} "
            f"output(s), expected {len(declared_specs)}"
        )
    outputs: List[core.Tensor] = []
    for i, (entry, spec) in enumerate(zip(entries, declared_specs)):
        where = f"{label} output {i}"
        if not isinstance(entry, (core.Tensor, np.ndarray)):
            try:
                if wrap_device_result is not None:
                    entry = wrap_device_result(entry)
                else:
                    entry = core.Tensor(entry)
            except (TypeError, core.DeviceError) as exc:
                raise core.BackendError(
                    f"{label}: callback output {i} is "
                    f"{type(entry).__name__}, which is not a valid device "
                    "payload (expected a core.Tensor, an ndarray, or a "
                    "payload with shape + dtype the wrap accepts)"
                ) from exc
        if isinstance(entry, np.ndarray):
            entry = core.Tensor(entry)
        _validate_entry_spec(entry.dtype, entry.shape, spec, where)
        outputs.append(entry)
    return outputs
