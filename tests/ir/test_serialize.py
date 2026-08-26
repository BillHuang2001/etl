"""Serialization round-trip and integrity tests for EvoXIR.

Covers the self-describing payload of ``etl.ir.serialize`` (see its module
docstring for the binding schema and encoding notes):

* payload structure and canonical sha256 integrity;
* round-trip of a multi-feature module (symbolic dims, DimExprs, dynamic dims,
  ndarray constants, all effect kinds, nested ``if`` regions, locations,
  two functions, metadata) with a structural equality walk;
* id-counter fast-forwarding after deserialization;
* tamper detection, missing keys, version/format rejection;
* serialization of invalid modules;
* multi-function ``call`` op round-trips;
* ``runtime_call``/``block_call`` ``result_specs`` normalization.

CPU only; every test builds tiny graphs (<100 ops) and stays well under 2s.
"""

import copy
import hashlib
import json
import re

import numpy as np
import pytest

from etl import ir
from etl.core import Dim, DimExpr, PersistenceError, VerificationError


# ---------------------------------------------------------------------------
# Module builders
# ---------------------------------------------------------------------------


def _simple_module() -> ir.Module:
    """One function: ``main(x) = x + x``."""
    m = ir.Module(name="simple")
    b = ir.Builder(m)
    f = b.build_function("main", (ir.ValueType(np.float32, (4,)),))
    x = f.entry_block.arguments[0]
    result = b.emit("add", (x, x))
    b.set_terminator(f.entry_block, "return", (result,))
    return m


def _multi_feature_module() -> ir.Module:
    """Two functions exercising every payload feature.

    ``main`` computes ``x + y * 2`` with symbolic inputs (Dim with known size,
    unknown Dim, DimExpr, runtime-dynamic None dim), a scalar float32 constant,
    ops of every effect kind (rank/read, all_reduce/collective,
    runtime_call/callback, block_call/read, add/pure), a broadcast with a
    symbolic DimExpr target shape, and an ``if`` with two nested regions.

    ``helper`` holds int32 (2x3) and float64 (2x2) constant payloads plus a
    reshape (shape attribute round-trip).
    """
    B = Dim("B", 8)
    N = Dim("N")
    m = ir.Module(name="m", metadata={"project": "serialize-tests", "level": 2})
    b = ir.Builder(m)

    f = b.build_function(
        "main",
        (
            ir.ValueType(np.float32, (B, N)),
            ir.ValueType(np.float32, (B, N)),
            ir.ValueType(np.int32, (DimExpr("mul", B, 2), None)),
        ),
        metadata={"kind": "entry"},
    )
    x, y, _z = f.entry_block.arguments
    loc = ir.Location("model.py", 12, 8, code_snippet="x + y * 2")
    two = b.emit(
        "constant", (), attributes={"value": np.array(2.0, dtype=np.float32)},
        location=loc,
    )
    y2 = b.emit("multiply", (y, two))
    s = b.emit("add", (x, y2), location=loc)
    b.emit("rank", (), attributes={"group": "world"})
    b.emit("all_reduce", (s,), attributes={"reduce_op": "sum", "group": "g1"})
    b.create(
        "runtime_call",
        (),
        attributes={
            "callback": "cb_scale",
            "result_specs": [ir.ValueType(np.float32, (B, N))],
        },
    )
    b.create(
        "block_call",
        (s,),
        attributes={
            "block_name": "my_block",
            "static_args": ((1, 2), "mode"),
            "result_specs": [ir.ValueType(np.float64, (N, 4))],
        },
    )
    pred = b.emit("equal", (x, y))
    b.emit("broadcast", (x,), attributes={"shape": (DimExpr("mul", B, 2), 8, N)})

    # `if` with two nested regions; entry args bind to ALL operands (pred, x).
    r_true = b.build_region((pred.type, x.type))
    b.push_region(r_true)
    xb = r_true.entry.arguments[1]
    taken = b.emit("add", (xb, xb))
    b.set_terminator(r_true.entry, "return", (taken,))
    b.pop_region()
    r_false = b.build_region((pred.type, x.type))
    b.push_region(r_false)
    fb = r_false.entry.arguments[1]
    b.set_terminator(r_false.entry, "return", (fb,))
    b.pop_region()
    iff = b.create("if", (pred, x), regions=(r_true, r_false), location=loc)
    b.set_terminator(f.entry_block, "return", (s, iff.result))

    hf = b.build_function(
        "helper",
        (ir.ValueType(np.float64, (2, 3)),),
        metadata={"kind": "helper"},
    )
    h = hf.entry_block.arguments[0]
    const_int = b.emit(
        "constant", (),
        attributes={"value": np.arange(6, dtype=np.int32).reshape(2, 3)},
    )
    const_f64 = b.emit(
        "constant", (),
        attributes={"value": np.linspace(1.0, 4.0, 4, dtype=np.float64).reshape(2, 2)},
    )
    h2 = b.emit("add", (h, h))
    hr = b.emit("reshape", (h2,), attributes={"shape": (3, 2)})
    b.set_terminator(hf.entry_block, "return", (hr, const_int))
    return m


def _call_module() -> ir.Module:
    """A module where ``main`` calls ``helper`` via the ``call`` op."""
    m = ir.Module(name="caller")
    b = ir.Builder(m)
    hf = b.build_function("helper", (ir.ValueType(np.float32, (4,)),))
    h = hf.entry_block.arguments[0]
    h2 = b.emit("add", (h, h))
    b.set_terminator(hf.entry_block, "return", (h2,))

    fm = b.build_function("main", (ir.ValueType(np.float32, (4,)),))
    x = fm.entry_block.arguments[0]
    res = b.create("call", (x,), attributes={"callee": "helper"})
    b.set_terminator(fm.entry_block, "return", (res.result,))
    return m


# ---------------------------------------------------------------------------
# Walking / comparison helpers
# ---------------------------------------------------------------------------


def _walk_ops(module: ir.Module):
    """All ops of the module, including nested region ops."""
    ops = []

    def walk_block(block):
        for op in block.ops:
            ops.append(op)
            for region in op.regions:
                for nested in region.blocks:
                    walk_block(nested)

    for function in module.functions:
        for block in function.region.blocks:
            walk_block(block)
    return ops


def _collect_ids(module: ir.Module):
    """All op/value ids used by the module (functions + nested regions)."""
    op_ids, value_ids = set(), set()

    def walk_block(block):
        value_ids.update(arg.id for arg in block.arguments)
        for op in block.ops:
            op_ids.add(op.id)
            value_ids.update(result.id for result in op.results)
            for region in op.regions:
                for nested in region.blocks:
                    walk_block(nested)

    for function in module.functions:
        for block in function.region.blocks:
            walk_block(block)
    return op_ids, value_ids


def _recompute_sha256(payload: dict) -> str:
    """The canonical sha256 serialize.py computes over the body (no 'sha256')."""
    body = {key: value for key, value in payload.items() if key != "sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_dim_equal(dim1, dim2) -> None:
    if dim1 is None or dim2 is None:
        assert dim1 is None and dim2 is None
        return
    if isinstance(dim1, int) or isinstance(dim2, int):
        assert dim1 == dim2
        return
    assert type(dim1) is type(dim2)
    if isinstance(dim1, Dim):
        assert dim1.name == dim2.name
        assert dim1.size == dim2.size
    else:  # DimExpr: structural equality (op/left/right)
        assert dim1.op == dim2.op
        _assert_dim_equal(dim1.left, dim2.left)
        _assert_dim_equal(dim1.right, dim2.right)


def _assert_type_equal(type1: ir.ValueType, type2: ir.ValueType) -> None:
    assert type1.dtype == type2.dtype
    assert len(type1.shape) == len(type2.shape)
    for dim1, dim2 in zip(type1.shape, type2.shape):
        _assert_dim_equal(dim1, dim2)


def _assert_attr_value_equal(value1, value2) -> None:
    # Strict type check: tuple/list distinction must survive the round-trip.
    assert type(value1) is type(value2), (
        f"attribute value type changed: {type(value1)} != {type(value2)}"
    )
    if isinstance(value1, np.ndarray):
        assert value1.dtype == value2.dtype
        assert value1.shape == value2.shape
        assert np.array_equal(value1, value2)
    elif isinstance(value1, ir.ValueType):
        _assert_type_equal(value1, value2)
    elif isinstance(value1, (Dim, DimExpr)):
        assert value1 == value2  # Dim: name+size; DimExpr: structural
    elif isinstance(value1, dict):
        assert value1.keys() == value2.keys()
        for key in value1:
            _assert_attr_value_equal(value1[key], value2[key])
    elif isinstance(value1, (list, tuple)):
        assert len(value1) == len(value2)
        for item1, item2 in zip(value1, value2):
            _assert_attr_value_equal(item1, item2)
    else:
        assert value1 == value2


def _assert_op_equal(op1: ir.Op, op2: ir.Op) -> None:
    assert op1.name == op2.name
    assert op1.id == op2.id
    assert [operand.id for operand in op1.operands] == [
        operand.id for operand in op2.operands
    ]
    assert len(op1.results) == len(op2.results)
    for result1, result2 in zip(op1.results, op2.results):
        assert result1.id == result2.id
        assert result1.index == result2.index
        _assert_type_equal(result1.type, result2.type)
    assert op1.attributes.keys() == op2.attributes.keys()
    for key in op1.attributes:
        if key == "result_specs":
            # The wire format normalizes the spec CONTAINER to a tuple while the
            # Builder's in-memory form is whatever sequence was passed (a list
            # here); the documented contract only fixes the ENTRY normalization
            # (each spec decodes back to a ValueType), so compare element-wise.
            specs1, specs2 = op1.attributes[key], op2.attributes[key]
            assert isinstance(specs1, (list, tuple))
            assert isinstance(specs2, (list, tuple))
            assert len(specs1) == len(specs2)
            for spec1, spec2 in zip(specs1, specs2):
                assert isinstance(spec1, ir.ValueType)
                assert isinstance(spec2, ir.ValueType)
                _assert_type_equal(spec1, spec2)
        else:
            _assert_attr_value_equal(op1.attributes[key], op2.attributes[key])
    assert op1.location == op2.location
    assert len(op1.regions) == len(op2.regions)
    for region1, region2 in zip(op1.regions, op2.regions):
        assert len(region1.blocks) == len(region2.blocks)
        for block1, block2 in zip(region1.blocks, region2.blocks):
            _assert_block_equal(block1, block2)


def _assert_block_equal(block1: ir.Block, block2: ir.Block) -> None:
    assert len(block1.arguments) == len(block2.arguments)
    for arg1, arg2 in zip(block1.arguments, block2.arguments):
        assert arg1.id == arg2.id
        assert arg1.index == arg2.index
        _assert_type_equal(arg1.type, arg2.type)
    assert len(block1.ops) == len(block2.ops)
    for op1, op2 in zip(block1.ops, block2.ops):
        _assert_op_equal(op1, op2)


def _assert_modules_equal(module1: ir.Module, module2: ir.Module) -> None:
    """Structural equality: walk both modules op-by-op (ids, types, attrs...)."""
    assert module1.name == module2.name
    assert module1.version == module2.version
    assert module1.metadata == module2.metadata
    assert [f.name for f in module1.functions] == [
        f.name for f in module2.functions
    ]
    for f1, f2 in zip(module1.functions, module2.functions):
        assert f1.metadata == f2.metadata
        assert len(f1.input_types) == len(f2.input_types)
        for type1, type2 in zip(f1.input_types, f2.input_types):
            _assert_type_equal(type1, type2)
        _assert_block_equal(f1.entry_block, f2.entry_block)


# ---------------------------------------------------------------------------
# 1. Payload structure
# ---------------------------------------------------------------------------


def test_payload_structure() -> None:
    module = _simple_module()
    payload = ir.serialize_module(module)

    assert set(payload) == {
        "format", "version", "module", "functions", "ops", "constants", "sha256",
    }
    assert payload["format"] == "etl-ir"
    assert payload["version"] == 1 == ir.IR_FORMAT_VERSION
    assert payload["module"]["name"] == "simple"
    assert re.fullmatch(r"[0-9a-f]{64}", payload["sha256"])
    assert payload["sha256"] == _recompute_sha256(payload)
    assert len(payload["functions"]) == 1

    function_entry = payload["functions"][0]
    assert set(function_entry) == {
        "name", "input_types", "output_types", "metadata", "block",
    }
    assert function_entry["name"] == "main"
    assert function_entry["input_types"] == [
        {"dtype": "float32", "shape": [{"int": 4}]}
    ]
    assert function_entry["metadata"] == {}

    # The flat ops table: int ids, str names; blocks reference it via "ref".
    assert len(payload["ops"]) == 2  # add + return
    for entry in payload["ops"]:
        assert isinstance(entry["id"], int)
        assert isinstance(entry["name"], str)
    block_ops = function_entry["block"]["ops"]
    assert block_ops and all(isinstance(entry["ref"], str) for entry in block_ops)
    ref_ids = {int(entry["ref"]) for entry in block_ops}
    table_ids = {entry["id"] for entry in payload["ops"]}
    assert ref_ids <= table_ids

    assert payload["constants"] == {}


# ---------------------------------------------------------------------------
# 2+3. Round-trip of a multi-feature module + structural equality walk
# ---------------------------------------------------------------------------


def test_round_trip_multi_feature_module() -> None:
    module = _multi_feature_module()
    payload = ir.serialize_module(module)
    rebuilt = ir.deserialize_module(payload)

    ir.verify(rebuilt)  # deserialize already verifies; still assert explicitly

    # 3. Structural equality walk: op-by-op comparison of both modules.
    _assert_modules_equal(module, rebuilt)

    # Original op/value ids are preserved exactly.
    op_ids_before, value_ids_before = _collect_ids(module)
    op_ids_after, value_ids_after = _collect_ids(rebuilt)
    assert op_ids_before == op_ids_after
    assert value_ids_before == value_ids_after

    # Two functions with preserved names, metadata, and signatures.
    assert [f.name for f in rebuilt.functions] == ["main", "helper"]
    assert rebuilt.get_function("main").metadata == {"kind": "entry"}
    assert rebuilt.get_function("helper").metadata == {"kind": "helper"}
    assert rebuilt.get_function("main").output_types == module.get_function(
        "main"
    ).output_types

    # Symbolic dims: Dim('B', 8) rebuilds with name AND size; Dim('N') has
    # no size; the DimExpr keeps its structure; the None dim stays dynamic.
    main_types = rebuilt.get_function("main").input_types
    b_dim, n_dim = main_types[0].shape
    assert isinstance(b_dim, Dim)
    assert b_dim.name == "B"
    assert b_dim.size == 8
    assert isinstance(n_dim, Dim)
    assert n_dim.name == "N"
    assert n_dim.size is None
    expr_dim, dynamic_dim = main_types[2].shape
    assert isinstance(expr_dim, DimExpr)
    assert expr_dim.op == "mul"
    assert expr_dim.left == Dim("B", 8)
    assert expr_dim.right == 2
    assert dynamic_dim is None
    helper_types = rebuilt.get_function("helper").input_types
    assert helper_types[0].shape == (2, 3)

    # Constant payloads survive bit-for-bit (int32 2x3 + float64 2x2 + f32 0-d).
    constants_before = {
        op.id: op.attributes["value"]
        for op in _walk_ops(module)
        if op.name == "constant"
    }
    constants_after = {
        op.id: op.attributes["value"]
        for op in _walk_ops(rebuilt)
        if op.name == "constant"
    }
    assert constants_before.keys() == constants_after.keys()
    assert len(constants_before) == 3
    for op_id, array in constants_before.items():
        other = constants_after[op_id]
        assert array.dtype == other.dtype
        assert array.shape == other.shape
        assert np.array_equal(array, other)

    # Effect kinds survive (they come from the registry, not the payload —
    # but the op names must round-trip to ops of the right effects).
    effects = {op.name: op.effect for op in _walk_ops(rebuilt)}
    assert effects["rank"] == ir.EFFECT_READ
    assert effects["all_reduce"] == ir.EFFECT_COLLECTIVE
    assert effects["runtime_call"] == ir.EFFECT_CALLBACK
    assert effects["block_call"] == ir.EFFECT_READ
    assert effects["add"] == ir.EFFECT_PURE
    assert effects["if"] == ir.EFFECT_PURE

    # Nested if regions survive: 2 regions, entry args bound positionally to
    # ALL operands (pred, x) with matching types; branch terminators intact.
    if_op = next(op for op in _walk_ops(rebuilt) if op.name == "if")
    assert len(if_op.regions) == 2
    for region in if_op.regions:
        args = region.entry.arguments
        assert len(args) == len(if_op.operands)
        for arg, operand in zip(args, if_op.operands):
            _assert_type_equal(arg.type, operand.type)
    assert if_op.location == ir.Location(
        "model.py", 12, 8, code_snippet="x + y * 2"
    )

    # pretty_print of the rebuilt module is identical (ids are preserved).
    assert ir.pretty_print(rebuilt) == ir.pretty_print(module)
    assert "func @main" in ir.pretty_print(rebuilt)


# ---------------------------------------------------------------------------
# 4. Builder on a deserialized module: id counters are fast-forwarded
# ---------------------------------------------------------------------------


def test_builder_after_deserialize_uses_fresh_ids() -> None:
    module = _multi_feature_module()
    rebuilt = ir.deserialize_module(ir.serialize_module(module))
    op_ids_before, value_ids_before = _collect_ids(rebuilt)

    b2 = ir.Builder(rebuilt)
    # build_function seeds the insertion-point stack (set_insertion_point
    # requires one entry); the probe function itself must get fresh ids too.
    probe = b2.build_function("probe", (ir.ValueType(np.float32, (4,)),))
    b2.set_terminator(probe.entry_block, "return", ())
    b2.set_insertion_point(rebuilt.get_function("main").entry_block)
    x = rebuilt.get_function("main").entry_block.arguments[0]
    new_op = b2.create("add", (x, x))

    fresh_op_ids = {op.id for op in probe.entry_block.ops}
    fresh_value_ids = {arg.id for arg in probe.entry_block.arguments}
    fresh_op_ids.add(new_op.id)
    fresh_value_ids.update(result.id for result in new_op.results)
    assert not fresh_op_ids & op_ids_before
    assert not fresh_value_ids & value_ids_before
    assert all(op_id > max(op_ids_before) for op_id in fresh_op_ids)
    assert all(value_id > max(value_ids_before) for value_id in fresh_value_ids)
    assert new_op.parent is rebuilt.get_function("main").entry_block
    ir.verify(rebuilt)


# ---------------------------------------------------------------------------
# 5. Tamper detection / missing keys
# ---------------------------------------------------------------------------


def _mutate_op_name(payload: dict) -> dict:
    payload["ops"][0]["name"] = "multiply"
    return payload


def _mutate_constant_payload(payload: dict) -> dict:
    key = next(iter(payload["constants"]))
    data = payload["constants"][key]["data_b64"]
    payload["constants"][key]["data_b64"] = data[:-4] + "AAAA"
    return payload


def _mutate_shape_dim(payload: dict) -> dict:
    # helper's input type is (2, 3); flip the first dim to 5.
    payload["functions"][1]["input_types"][0]["shape"][0] = {"int": 5}
    return payload


@pytest.fixture(scope="module")
def multi_payload() -> dict:
    return ir.serialize_module(_multi_feature_module())


@pytest.mark.parametrize(
    "mutate",
    [_mutate_op_name, _mutate_constant_payload, _mutate_shape_dim],
    ids=["op-name", "constant-data", "shape-dim"],
)
def test_tamper_detection(multi_payload, mutate) -> None:
    payload = copy.deepcopy(multi_payload)
    mutate(payload)
    with pytest.raises(VerificationError, match=r"integrity|sha256"):
        ir.deserialize_module(payload)


def test_dropped_sha256_rejected(multi_payload) -> None:
    payload = copy.deepcopy(multi_payload)
    del payload["sha256"]
    with pytest.raises(VerificationError, match=r"sha256"):
        ir.deserialize_module(payload)


def test_missing_functions_key_rejected(multi_payload) -> None:
    # Re-hash so the integrity check passes and the structural helper fires.
    payload = copy.deepcopy(multi_payload)
    del payload["functions"]
    payload["sha256"] = _recompute_sha256(payload)
    with pytest.raises(VerificationError, match=r"functions.*missing from payload"):
        ir.deserialize_module(payload)


def test_non_dict_payload_rejected() -> None:
    with pytest.raises(PersistenceError, match=r"dict"):
        ir.deserialize_module(["etl-ir", 1])


# ---------------------------------------------------------------------------
# 6. Version / format rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, pattern",
    [
        (lambda p: p.update({"version": 999}), r"version"),
        (lambda p: p.pop("version"), r"version"),
        (lambda p: p.update({"format": "nope"}), r"format"),
    ],
    ids=["version-999", "version-missing", "format-unknown"],
)
def test_version_format_rejection(multi_payload, mutate, pattern) -> None:
    payload = copy.deepcopy(multi_payload)
    mutate(payload)
    with pytest.raises(PersistenceError, match=pattern):
        ir.deserialize_module(payload)


# ---------------------------------------------------------------------------
# 7. Invalid modules never serialize
# ---------------------------------------------------------------------------


def test_serialize_invalid_module_raises() -> None:
    module = ir.Module(name="no-terminator")
    b = ir.Builder(module)
    f = b.build_function("main", (ir.ValueType(np.float32, (4,)),))
    x = f.entry_block.arguments[0]
    b.emit("add", (x, x))  # never terminate the entry block
    with pytest.raises(VerificationError, match=r"terminator"):
        ir.serialize_module(module)


# ---------------------------------------------------------------------------
# 8. Multi-function module + call op
# ---------------------------------------------------------------------------


def test_call_op_round_trip() -> None:
    module = _call_module()
    rebuilt = ir.deserialize_module(ir.serialize_module(module))

    ir.verify(rebuilt)
    _assert_modules_equal(module, rebuilt)

    main = rebuilt.get_function("main")
    call_op = next(op for op in main.entry_block.ops if op.name == "call")
    assert call_op.attributes["callee"] == "helper"
    # The call's results match the callee's output signature.
    assert tuple(result.type for result in call_op.results) == tuple(
        rebuilt.get_function("helper").output_types
    )
    # And the caller's own return passes them through unchanged.
    assert tuple(main.output_types) == tuple(
        rebuilt.get_function("helper").output_types
    )


# ---------------------------------------------------------------------------
# 9. runtime_call / block_call result_specs normalization
# ---------------------------------------------------------------------------


def test_result_specs_normalized_to_valuetypes() -> None:
    module = _multi_feature_module()
    rebuilt = ir.deserialize_module(ir.serialize_module(module))

    for op_name in ("runtime_call", "block_call"):
        ops = [op for op in _walk_ops(rebuilt) if op.name == op_name]
        assert ops, f"no {op_name} op in the rebuilt module"
        for op in ops:
            specs = op.attributes["result_specs"]
            assert isinstance(specs, tuple)
            assert all(isinstance(spec, ir.ValueType) for spec in specs)
            # The in-memory Builder form is a tuple of ValueTypes too.
            assert tuple(specs) == tuple(result.type for result in op.results)
    ir.verify(rebuilt)
