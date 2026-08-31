"""External-kernel graph splitting + adapter host-dispatch (round 2).

Round 1 delivered the ``external_call`` op and the numpy-backend dispatch
(``etl.backends.numpy.kernels.custom``). This module adds the COMPILER-ADAPTER
host-dispatch path: a graph containing ``external_call`` ops is split at
``lower()`` time into an ordered list of SEGMENT GRAPHS (plain graphs with no
``external_call`` inside) plus a kernel-call PLAN. Segments run on the device;
at each ``external_call`` boundary the segment outputs are staged to host
numpy arrays, the registered kernel is dispatched, and its results are staged
back as inputs of the following segment.

Pure graph transformation: operates on ``etl.ir`` / ``trace.Graph`` structures
only — the numpy interpreter is never required here. The numpy backend keeps
its direct interpreter dispatch (round 1); compiler adapters declaring
``Capabilities.external_calls=True`` (v1: the iree adapter) consume this
module from the shared ``CompilerBackend.lower`` (see ``../compiler.py``) and
from their executables at run time.

Split model (binding — mirrors the numpy backend's block-op-order semantics):

- The entry function's block is CUT at every top-level ``external_call`` op
  (op order). Kernel segment j ends with external call j; the final segment
  holds the ops after the last call, including the ``return`` terminator.
- Segment j's INPUTS = (graph inputs used by segment j's ops) ∪ (values
  crossing boundary j-1). The crossing values are "carried" values: kernel
  results and earlier plain-op results that are still live after the
  boundary; they pass through intermediate segments untouched (a value
  defined in segment i and used in segment k > i+1 is carried across every
  boundary i..k-1).
- Segment j's MODULE OUTPUTS (the compiled function's return values) =
  call j's operand tensors (the staging boundary) followed by the boundary-j
  carry values other than the call's own results, for kernel segments; the
  final segment's module outputs are the graph outputs (the ``return``
  terminator's operands, in leaf order).
- Kernel results are NOT module outputs (the host kernel produces them):
  they are validated against the declared specs at dispatch time and occupy
  segment-output slots ``module_outputs .. module_outputs + R - 1`` (R =
  declared result count), so later segments reference them like any other
  carried value.

v1 lower-time constraints (explicit ``core.BackendError``, never silent):

- ``external_call`` inside ``cond``/``while_loop``/``scan`` bodies
  (data-dependent control flow around a host call is not splittable).
- Symbolic / runtime-dynamic result dims (staging needs static shapes).
- Zero-operand calls (no staging boundary — express the value via
  ``etl.constant`` or an explicit graph input instead).

Determinism: kernels are assumed PURE (same inputs -> same outputs); segment
execution is strictly sequential in plan order, so the graph's effect
ordering (the ``callback`` anchor) is preserved.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from etl import core
from etl import ir

__all__ = [
    "contains_external_call",
    "validate_and_split",
    "decode_plan",
    "encode_type",
    "decode_type",
    "dispatch_external_kernel",
]


# ---------------------------------------------------------------------------
# op walk (top-level block + nested regions)
# ---------------------------------------------------------------------------


def _walk_ops(block: ir.Block, visit: Callable[[ir.Op], None]) -> None:
    """Visit ``block``'s ops in order, recursing into nested regions."""
    for op in block.ops:
        visit(op)
        for region in op.regions:
            for nested_block in region.blocks:
                _walk_ops(nested_block, visit)


def contains_external_call(module: ir.Module) -> bool:
    """True iff any op in ``module`` (incl. nested regions) is ``external_call``."""
    for function in module.functions:
        found: List[ir.Op] = []
        _walk_ops(function.entry_block, lambda op: found.append(op))
        for op in found:
            if op.name == "external_call":
                return True
    return False


def _find_nested_external_call(op: ir.Op) -> Optional[ir.Op]:
    """The first ``external_call`` op nested inside ``op``'s regions, or None."""
    for region in op.regions:
        for block in region.blocks:
            nested: List[ir.Op] = []
            _walk_ops(block, lambda o: nested.append(o))
            for candidate in nested:
                if candidate.name == "external_call":
                    return candidate
    return None


def _collect_uses(op: ir.Op, into: set) -> None:
    """Collect every value id ``op`` uses, including nested-region captures."""
    for value in op.operands:
        into.add(value.id)
    for region in op.regions:
        for block in region.blocks:
            for nested in block.ops:
                _collect_uses(nested, into)


# ---------------------------------------------------------------------------
# spec codec (the standard ir wire forms — JSON-safe plan encoding)
# ---------------------------------------------------------------------------


def _encode_dim(dim: Any) -> Any:
    """Encode one shape dim: int | Dim | DimExpr | None (wire forms)."""
    if dim is None:
        return None
    if isinstance(dim, core.Dim):
        encoded: dict = {"dim": dim.name}
        if dim.size is not None:
            encoded["size"] = dim.size
        return encoded
    if isinstance(dim, core.DimExpr):
        return {
            "expr": {
                "op": dim.op,
                "args": [_encode_dim(dim.left), _encode_dim(dim.right)],
            }
        }
    if isinstance(dim, int) and not isinstance(dim, bool):
        return {"int": dim}
    raise core.BackendError(
        f"cannot encode shape dimension {dim!r} in an external-call plan"
    )


def _decode_dim(encoded: Any) -> Any:
    """Decode one wire-encoded shape dim back to int | Dim | DimExpr | None."""
    if encoded is None:
        return None
    if isinstance(encoded, dict):
        if "expr" in encoded:
            expr = encoded["expr"]
            if not isinstance(expr, dict) or not isinstance(expr.get("op"), str):
                raise core.BackendError(f"malformed DimExpr plan encoding {encoded!r}")
            args = expr.get("args")
            if not isinstance(args, list) or len(args) != 2:
                raise core.BackendError(
                    f"malformed DimExpr plan encoding {encoded!r}: 'args' must be "
                    "a 2-element list"
                )
            return core.DimExpr(expr["op"], _decode_dim(args[0]), _decode_dim(args[1]))
        if "dim" in encoded:
            name = encoded["dim"]
            if not isinstance(name, str):
                raise core.BackendError(f"malformed Dim plan encoding {encoded!r}")
            size = encoded.get("size")
            if size is not None and not (
                isinstance(size, int) and not isinstance(size, bool)
            ):
                raise core.BackendError(f"malformed Dim plan encoding {encoded!r}")
            return core.Dim(name, size=size)
        if "int" in encoded:
            value = encoded["int"]
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            raise core.BackendError(f"malformed dim plan encoding {encoded!r}")
    raise core.BackendError(f"invalid dim plan encoding {encoded!r}")


def encode_type(value_type: ir.ValueType) -> dict:
    """Encode a ``ValueType`` as a JSON-safe dict (wire forms)."""
    return {
        "dtype": value_type.dtype.name,
        "shape": [_encode_dim(dim) for dim in value_type.shape],
    }


def decode_type(encoded: dict) -> ir.ValueType:
    """Decode a wire-encoded ``ValueType`` back to an ``ir.ValueType``."""
    if not isinstance(encoded, dict) or not isinstance(encoded.get("dtype"), str):
        raise core.BackendError(f"malformed ValueType plan encoding {encoded!r}")
    try:
        dtype = core.dtype(encoded["dtype"])
    except core.DTypeError as exc:
        raise core.BackendError(
            f"malformed ValueType plan encoding {encoded!r}: {exc}"
        ) from exc
    shape = encoded.get("shape")
    if not isinstance(shape, list):
        raise core.BackendError(f"malformed ValueType plan encoding {encoded!r}")
    return ir.ValueType(dtype=dtype, shape=tuple(_decode_dim(dim) for dim in shape))


# ---------------------------------------------------------------------------
# graph splitting
# ---------------------------------------------------------------------------


def validate_and_split(graph: Any) -> Tuple[List[ir.Module], dict]:
    """Split a graph containing ``external_call`` ops into segment modules + plan.

    Lower-time entry point of the adapter host-dispatch path (called by
    ``CompilerBackend.lower`` when ``Capabilities.external_calls`` is True
    and the graph contains at least one ``external_call`` op). Enforces the
    v1 constraints (see module docstring) with explicit ``core.BackendError``
    messages and returns:

    - ``segment_modules``: one ``ir.Module`` per segment, in execution order,
      each with a single "main" function holding the segment's ops (no
      ``external_call`` inside) and a ``return`` terminator over the
      segment's outputs.
    - ``plan``: a JSON-safe dict describing the execution plan (segment
      inputs/outputs, call records with kernel names and declared result
      specs, slot plumbing). See ``decode_plan`` for the run-time form.

    Determinism: all value orderings are derived from the original block's
    op order (defining-op position, then result index), so the plan is
    stable for a given graph.
    """
    module = graph.module
    if len(module.functions) != 1:
        raise core.BackendError(
            "external-call host-dispatch supports single-function graphs in "
            "v1 (this graph has "
            f"{len(module.functions)} function(s))"
        )
    function = module.functions[0]
    block = function.entry_block
    ops = list(block.ops)  # includes the 'return' terminator as the last op
    block_args = block.arguments

    # --- v1 constraint checks (explicit BackendError, never silent) ---------
    call_indices: List[int] = []
    for index, op in enumerate(ops):
        if op.name != "external_call":
            nested = _find_nested_external_call(op)
            if nested is not None:
                raise core.BackendError(
                    f"op 'external_call' (kernel {nested.attributes['name']!r}): "
                    "adapter host-dispatch does not support external calls "
                    "inside cond/while_loop/scan bodies in v1 — "
                    "data-dependent control flow around a host call is not "
                    "splittable; use the numpy backend"
                )
            continue
        call_indices.append(index)
        name = op.attributes["name"]
        if len(op.operands) == 0:
            raise core.BackendError(
                f"op 'external_call' (kernel {name!r}): adapter host-dispatch "
                "requires at least one tensor operand in v1 — a kernel with "
                "zero operands has no staging boundary; express the value via "
                "etl.constant or an explicit graph input instead"
            )
        for spec in op.attributes["result_specs"]:
            for dim in spec.shape:
                if not (isinstance(dim, int) and not isinstance(dim, bool)):
                    raise core.BackendError(
                        f"op 'external_call' (kernel {name!r}): adapter "
                        "host-dispatch requires STATIC (integer) result dims "
                        "in v1 — staging needs concrete shapes; declared "
                        f"result dim {dim!r} is symbolic/runtime-dynamic. Use "
                        "a static TensorSpec or run the numpy backend"
                    )
    if not call_indices:  # defensive: the caller only routes here when present
        raise core.BackendError(
            "validate_and_split requires a graph with at least one "
            "external_call op"
        )

    # --- segment op ranges ------------------------------------------------
    #   kernel segment j (j < m): ops[prev+1 .. cut_j-1] (the call op at
    #     cut_j is the BOUNDARY — never cloned into a segment; its operands
    #     still count as segment uses so they land in the input/output sets)
    #   final segment:           ops[last_call+1 .. len(ops)-1] (incl. return)
    ranges: List[Tuple[int, int]] = []
    prev = -1
    for cut in call_indices:
        ranges.append((prev + 1, cut - 1))
        prev = cut
    ranges.append((prev + 1, len(ops) - 1))
    num_segments = len(ranges)
    seg_of_op: Dict[int, int] = {}
    for seg_index, (start, end) in enumerate(ranges):
        for op_index in range(start, end + 1):
            seg_of_op[op_index] = seg_index
    for seg_index, cut in enumerate(call_indices):
        seg_of_op[cut] = seg_index  # call results are "defined" at the boundary

    # --- value bookkeeping -------------------------------------------------
    # defs: value id -> (defining op index, result index)
    defs: Dict[int, Tuple[int, int]] = {}
    for op_index, op in enumerate(ops):
        for result_index, value in enumerate(op.results):
            defs[value.id] = (op_index, result_index)
    arg_index_of: Dict[int, int] = {
        value.id: i for i, value in enumerate(block_args)
    }

    # used_after[i] = value ids referenced by ops[i:] (incl. nested captures)
    used_after: List[set] = [set() for _ in range(len(ops) + 1)]
    suffix: set = set()
    for i in range(len(ops) - 1, -1, -1):
        _collect_uses(ops[i], suffix)
        used_after[i] = set(suffix)

    # boundary(j) = values defined by ops <= call_j that are used after it
    boundaries: List[set] = []
    for cut in call_indices:
        after = used_after[cut + 1] if cut + 1 < len(ops) else set()
        boundaries.append(
            {vid for vid in after if vid in defs and defs[vid][0] <= cut}
        )

    # --- per-segment input/output value lists -----------------------------
    def sort_values(value_ids: set) -> List[int]:
        """Deterministic order: (defining op index, result index)."""
        return sorted(value_ids, key=lambda vid: defs[vid])

    # Kernel results are NOT module outputs (the host kernel produces them):
    # a kernel segment's module outputs are its call operands followed by the
    # live-out values; kernel results occupy slots module_outputs + r after
    # the module outputs (appended by the executor at dispatch time).
    call_result_slots: Dict[int, Dict[int, int]] = {}
    segment_inputs: List[List[dict]] = []
    segment_outputs: List[List[int]] = []
    for seg_index, (start, end) in enumerate(ranges):
        uses = set()
        for op_index in range(start, end + 1):
            _collect_uses(ops[op_index], uses)
        if seg_index < len(call_indices):
            # the boundary call's operands are staged segment outputs even
            # though the call op itself is not part of the segment
            for value in ops[call_indices[seg_index]].operands:
                uses.add(value.id)
        # graph inputs used by this segment's ops (block-arg order)
        graph_ids = sorted(
            (vid for vid in uses if vid in arg_index_of),
            key=lambda vid: arg_index_of[vid],
        )
        # carried values crossing the previous boundary (def order)
        carry_ids = sorted(boundaries[seg_index - 1]) if seg_index > 0 else []
        entries: List[dict] = []
        for vid in graph_ids:
            entries.append({"kind": "graph", "index": arg_index_of[vid]})
        for vid in carry_ids:
            producer = seg_of_op[defs[vid][0]]
            producer_slots = call_result_slots.get(producer, {})
            if vid in producer_slots:
                slot = producer_slots[vid]
                kind = "kernel"
            else:
                slot = segment_outputs[producer].index(vid)
                kind = "carry"
            entries.append({"kind": kind, "segment": producer, "output": slot})
        segment_inputs.append(entries)

        if seg_index < len(call_indices):
            call_op = ops[call_indices[seg_index]]
            call_result_ids = {value.id for value in call_op.results}
            operand_ids = [value.id for value in call_op.operands]
            live_ids = [
                vid
                for vid in sort_values(boundaries[seg_index])
                if vid not in operand_ids and vid not in call_result_ids
            ]
            module_outputs = operand_ids + live_ids
            call_result_slots[seg_index] = {
                value.id: len(module_outputs) + result_index
                for result_index, value in enumerate(call_op.results)
            }
            segment_outputs.append(module_outputs)
        else:
            terminator = ops[end]  # the 'return' terminator
            segment_outputs.append([value.id for value in terminator.operands])

    # --- build segment modules ---------------------------------------------
    segment_modules: List[ir.Module] = []
    segments_plan: List[dict] = []
    for seg_index, ((start, end), entries, output_ids) in enumerate(
        zip(ranges, segment_inputs, segment_outputs)
    ):
        input_values: List[ir.Value] = []
        for entry in entries:
            if entry["kind"] == "graph":
                input_values.append(block_args[entry["index"]])
            elif entry["kind"] == "kernel":
                # kernel-result slot: the producing segment's call results
                producer = entry["segment"]
                producer_call = ops[call_indices[producer]]
                result_index = entry["output"] - len(segment_outputs[producer])
                input_values.append(producer_call.results[result_index])
            else:
                # the producing segment's module-output slot -> original id
                producer_outputs = segment_outputs[entry["segment"]]
                input_values.append(
                    _value_by_id(ops, block_args, producer_outputs[entry["output"]])
                )
        output_values = [
            _value_by_id(ops, block_args, vid) for vid in output_ids
        ]
        segment_modules.append(
            _build_segment_module(
                ops[start : end + 1], input_values, output_values
            )
        )
        segment_record: dict = {
            "inputs": entries,
            "input_specs": [encode_type(value.type) for value in input_values],
            "module_outputs": len(output_values),
        }
        if seg_index < len(call_indices):
            call_op = ops[call_indices[seg_index]]
            module_outputs = segment_outputs[seg_index]
            result_outputs = [
                call_result_slots[seg_index][value.id]
                for value in call_op.results
            ]
            segment_record["call"] = {
                "index": seg_index,
                "name": call_op.attributes["name"],
                "operand_outputs": [
                    module_outputs.index(value.id) for value in call_op.operands
                ],
                "result_specs": [
                    encode_type(spec) for spec in call_op.attributes["result_specs"]
                ],
                "result_outputs": result_outputs,
            }
        segments_plan.append(segment_record)

    plan = {"format_version": 1, "segments": segments_plan}
    return segment_modules, plan


def _value_by_id(
    ops: Sequence[ir.Op], block_args: Sequence[ir.Value], value_id: int
) -> ir.Value:
    """Look up an original value by id (block arg or op result)."""
    for arg in block_args:
        if arg.id == value_id:
            return arg
    for op in ops:
        for value in op.results:
            if value.id == value_id:
                return value
    raise core.BackendError(
        f"internal error: external-call plan references unknown value id "
        f"{value_id}"
    )


def _build_segment_module(
    seg_ops: Sequence[ir.Op],
    input_values: Sequence[ir.Value],
    output_values: Sequence[ir.Value],
) -> ir.Module:
    """Clone ``seg_ops`` into a fresh single-function module.

    The segment's function block args are fresh values for ``input_values``
    (in order); every original value referenced by the segment ops is
    remapped: inputs -> block args, in-segment defs -> the cloned ops'
    results, nested-region args -> fresh region args. The terminator is a
    ``return`` over the remapped ``output_values``.
    """
    builder = ir.Builder()
    module = builder.build_module(name="main")
    function = builder.build_function(
        name="main", input_types=tuple(value.type for value in input_values)
    )
    block = function.entry_block
    remap: Dict[int, ir.Value] = {
        value.id: arg for value, arg in zip(input_values, block.arguments)
    }
    for op in seg_ops:
        if op.is_terminator:
            continue  # the 'return' is rebuilt below from output_values
        _clone_op(builder, op, remap)
    builder.set_terminator(
        block, "return", operands=tuple(remap[value.id] for value in output_values)
    )
    ir.verify(module)  # defensive: the split must produce valid IR
    return module


def _clone_op(builder: ir.Builder, op: ir.Op, remap: Dict[int, ir.Value]) -> ir.Op:
    """Clone one op (with nested regions) into the builder, remapping values."""
    regions = tuple(_clone_region(builder, region, remap) for region in op.regions)
    new_op = builder.create(
        op.name,
        operands=tuple(remap[value.id] for value in op.operands),
        attributes=dict(op.attributes),
        result_types=tuple(value.type for value in op.results),
        location=op.location,
        regions=regions,
    )
    for new_value, old_value in zip(new_op.results, op.results):
        remap[old_value.id] = new_value
    return new_op


def _clone_region(
    builder: ir.Builder, region: ir.Region, remap: Dict[int, ir.Value]
) -> ir.Region:
    """Deep-clone one nested region (v1: single block) into a detached region."""
    if len(region.blocks) != 1:
        raise core.BackendError(
            "external-call splitting supports single-block regions in v1"
        )
    source = region.entry
    new_region = builder.build_region(
        tuple(arg.type for arg in source.arguments)
    )
    new_block = new_region.entry
    for old_arg, new_arg in zip(source.arguments, new_block.arguments):
        remap[old_arg.id] = new_arg
    builder.push_region(new_region)
    try:
        for op in source.ops:
            if op.is_terminator:
                continue
            _clone_op(builder, op, remap)
    finally:
        builder.pop_region()
    terminator = source.terminator
    if terminator is None:
        raise core.BackendError(
            "external-call splitting: nested region has no terminator"
        )
    builder.set_terminator(
        new_block,
        "return",
        operands=tuple(remap[value.id] for value in terminator.operands),
    )
    return new_region


# ---------------------------------------------------------------------------
# plan decoding (run-time form)
# ---------------------------------------------------------------------------


def decode_plan(plan: dict) -> List[dict]:
    """Decode a JSON-safe plan into the run-time segment list.

    Each segment record gains ``input_specs`` (tuple of ``ir.ValueType``)
    and ``call["result_specs"]`` (tuple of ``ir.ValueType``); everything
    else is copied as-is. Raises ``core.BackendError`` for malformed plans
    (artifacts are never executed with a silently partial plan).
    """
    raw_segments = plan.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise core.BackendError(
            "corrupt: external-call plan records no segments"
        )
    segments = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise core.BackendError(
                f"corrupt: external-call plan segment {index} is not a dict"
            )
        inputs = raw.get("inputs")
        raw_specs = raw.get("input_specs")
        module_outputs = raw.get("module_outputs")
        if not isinstance(inputs, list) or not isinstance(raw_specs, list):
            raise core.BackendError(
                f"corrupt: external-call plan segment {index} is missing "
                "'inputs'/'input_specs'"
            )
        if not (
            isinstance(module_outputs, int)
            and not isinstance(module_outputs, bool)
        ):
            raise core.BackendError(
                f"corrupt: external-call plan segment {index} has a bad "
                "'module_outputs'"
            )
        decoded = {
            "inputs": inputs,
            "input_specs": tuple(decode_type(spec) for spec in raw_specs),
            "module_outputs": module_outputs,
        }
        call = raw.get("call")
        if call is not None:
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                raise core.BackendError(
                    f"corrupt: external-call plan segment {index} has a bad "
                    "'call' record"
                )
            result_specs = call.get("result_specs")
            if not isinstance(result_specs, list):
                raise core.BackendError(
                    f"corrupt: external-call plan segment {index} 'call' has "
                    "no 'result_specs'"
                )
            decoded["call"] = {
                "index": call.get("index"),
                "name": call["name"],
                "operand_outputs": tuple(call.get("operand_outputs") or ()),
                "result_specs": tuple(decode_type(spec) for spec in result_specs),
            }
        segments.append(decoded)
    return segments


# ---------------------------------------------------------------------------
# run-time kernel dispatch (shared by adapter executables)
# ---------------------------------------------------------------------------


def dispatch_external_kernel(
    name: str,
    operand_arrays: Sequence[Any],
    result_types: Sequence[ir.ValueType],
    label: str,
) -> List[core.Tensor]:
    """Resolve kernel ``name``, call it with host numpy operands, validate.

    Mirrors the numpy backend's round-1 dispatch semantics (see
    ``etl/backends/numpy/kernels/custom.py``): the kernel is resolved through
    ``etl.external.get_external_kernel`` (unknown name => ``BackendError``
    naming the kernel and pointing at ``register_external_kernel``), called
    with the operand numpy arrays, and its return is normalized and validated
    against the declared ``result_types`` — count (``BackendError``), dtype
    exact (``BackendError`` — no silent coercion), shape exact
    (``ShapeError``). Validated results are returned as host ``core.Tensor``
    s (staged back to the device by the caller's next segment run).

    ``label`` names the call in error messages (e.g. the op + kernel name).
    """
    from etl.external import get_external_kernel  # lazy: import acyclicity

    kernel = get_external_kernel(name)
    if kernel is None:
        raise core.BackendError(
            f"{label}: no external kernel registered under name {name!r} — "
            "register it with etl.register_external_kernel(name, callable) "
            "in this process before running graphs that call it (kernels "
            "are never embedded in artifacts)"
        )
    result = kernel(*operand_arrays)
    arrays = _normalize_results(result, label)
    return _validate_static_outputs(arrays, result_types, label)


def _normalize_results(result: Any, label: str) -> list:
    """Normalize a kernel return into a list of output entries.

    Same contract as the numpy backend's helper: ``ndarray`` -> ``[arr]``;
    ``core.Tensor`` -> ``[tensor]``; tuple/list -> ``list(...)``. Anything
    else => ``core.BackendError`` naming the call (never a guess).
    """
    if isinstance(result, np.ndarray) or isinstance(result, core.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return list(result)
    raise core.BackendError(
        f"{label}: kernel returned {type(result).__name__}, expected an "
        "ndarray, a core.Tensor, or a tuple/list of them"
    )


def _validate_static_outputs(
    arrays: list, result_types: Sequence[ir.ValueType], label: str
) -> List[core.Tensor]:
    """Validate kernel outputs against STATIC declared specs (adapter path).

    Count must match (``BackendError``); per output: dtype must match exactly
    (``BackendError`` — no silent coercion) and the shape must equal the
    declared shape (``ShapeError``; a defensive ``None`` dim is skipped —
    lower() rejects runtime-dynamic dims, so this never triggers in v1).
    Valid outputs are wrapped in host ``core.Tensor`` s.
    """
    if len(arrays) != len(result_types):
        raise core.BackendError(
            f"{label}: kernel produced {len(arrays)} output(s), expected "
            f"{len(result_types)} (declared result specs)"
        )
    outputs: List[core.Tensor] = []
    for index, (entry, spec) in enumerate(zip(arrays, result_types)):
        where = f"{label} output {index}"
        if isinstance(entry, core.Tensor):
            entry = entry.numpy()
        if not isinstance(entry, __import__("numpy").ndarray):
            raise core.BackendError(
                f"{where}: kernel returned {type(entry).__name__}, expected "
                "an ndarray or a core.Tensor"
            )
        if entry.dtype != spec.dtype:
            raise core.BackendError(
                f"{where}: kernel returned dtype {entry.dtype}, declared "
                f"result dtype {spec.dtype} — no silent dtype coercion"
            )
        expected = tuple(spec.shape)
        actual = tuple(int(d) for d in entry.shape)
        if len(actual) != len(expected):
            raise core.ShapeError(
                f"{where}: kernel returned shape {actual} (rank "
                f"{len(actual)}), declared rank {len(expected)}"
            )
        for dim_index, (exp, act) in enumerate(zip(expected, actual)):
            if exp is not None and exp != act:
                raise core.ShapeError(
                    f"{where}: kernel returned shape {actual}, declared "
                    f"shape {expected} — mismatch at dim {dim_index} "
                    f"({act} != {exp})"
                )
        outputs.append(core.Tensor(entry))
    return outputs
