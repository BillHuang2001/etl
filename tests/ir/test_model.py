"""SSA data-model tests for EvoXIR (``etl.ir``).

Pins down the structural contracts of the IR building blocks —
``ValueType``, ``Location``, effect kinds, ``Module``/``Function``/
``Region``/``Block``/``Op``/``Value``/``Use`` — as documented in
``etl/ir``. The ``Builder`` is exercised here only for structure sanity
(parent wiring, id assignment, use bookkeeping); exhaustive Builder
coverage (insertion points, attribute validation, result inference)
belongs to ``test_builder.py``.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from etl import ir
from etl.core import Dim, DimExpr


def _f32_4() -> ir.ValueType:
    """f32 tensor of shape (4,)."""
    return ir.ValueType(np.dtype("float32"), (4,))


def _bare_function(name: str) -> ir.Function:
    """A detached single-block function (no terminator — for wiring tests)."""
    region = ir.Region()
    region.append_block(ir.Block())
    return ir.Function(name=name, input_types=(), region=region)


# ---------------------------------------------------------------------------
# 1. ValueType
# ---------------------------------------------------------------------------


class TestValueType:
    def test_dtype_normalized_to_numpy_dtype(self):
        vt = ir.ValueType("float32", (2, 3))
        assert vt.dtype == np.dtype("float32")
        assert isinstance(vt.dtype, np.dtype)
        # passing an actual dtype is a no-op normalization
        assert ir.ValueType(np.dtype("int64"), ()).dtype == np.dtype("int64")

    def test_shape_coerced_to_tuple(self):
        vt = ir.ValueType(np.float32, [3, 4])
        assert vt.shape == (3, 4)
        assert isinstance(vt.shape, tuple)

    def test_rank(self):
        assert ir.ValueType(np.float32, ()).rank == 0
        assert ir.ValueType(np.float32, (2,)).rank == 1
        assert ir.ValueType(np.float32, (2, 3, 4)).rank == 3

    @pytest.mark.parametrize(
        "dtype,shape,expected",
        [
            (np.float32, (3, 4), "tensor<3x4xf32>"),
            (np.float32, (), "tensor<f32>"),
            (np.float32, (None, 4), "tensor<?x4xf32>"),
            (np.int64, (2,), "tensor<2xi64>"),
            (np.bool_, (2,), "tensor<2xi1>"),
            (np.float64, (2,), "tensor<2xf64>"),
        ],
    )
    def test_str_forms(self, dtype, shape, expected):
        assert str(ir.ValueType(dtype, shape)) == expected

    def test_str_symbolic_dims(self):
        vt = ir.ValueType(np.float32, (Dim("B"), Dim("N")))
        assert str(vt) == "tensor<BxNxf32>"

    def test_str_dim_expr(self):
        vt = ir.ValueType(np.float32, (Dim("B") * 2,))
        assert str(vt) == "tensor<B * 2xf32>"

    def test_str_nested_dim_expr_is_parenthesized(self):
        vt = ir.ValueType(np.float32, (Dim("B") * 2 + 1,))
        assert str(vt) == "tensor<(B * 2) + 1xf32>"

    def test_equality_is_structural(self):
        assert ir.ValueType("float32", [2, 3]) == ir.ValueType(np.float32, (2, 3))
        assert ir.ValueType(np.float32, (2, 3)) != ir.ValueType(np.float32, (2, 4))
        assert ir.ValueType(np.float32, (2,)) != ir.ValueType(np.float64, (2,))
        assert ir.ValueType(np.float32, (2,)) != ir.ValueType(np.float32, ())
        # symbolic dims participate structurally
        assert ir.ValueType(np.float32, (Dim("B"),)) == ir.ValueType(
            np.float32, (Dim("B"),)
        )
        assert ir.ValueType(np.float32, (Dim("B"),)) != ir.ValueType(
            np.float32, (Dim("C"),)
        )

    def test_frozen(self):
        vt = ir.ValueType(np.float32, (2,))
        with pytest.raises(FrozenInstanceError):
            vt.dtype = np.float64  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            vt.shape = (3,)  # type: ignore[misc]

    def test_repr_contains_dtype_name(self):
        vt = ir.ValueType(np.float32, (2, 3))
        assert repr(vt) == "ValueType(dtype=np.dtype('float32'), shape=(2, 3))"
        assert "float32" in repr(vt)


# ---------------------------------------------------------------------------
# 2. Location
# ---------------------------------------------------------------------------


class TestLocation:
    def test_field_access(self):
        loc = ir.Location("model.py", 83, 5, "y = x + 1")
        assert loc.file == "model.py"
        assert loc.line == 83
        assert loc.col == 5
        assert loc.code_snippet == "y = x + 1"

    def test_default_snippet_is_none(self):
        assert ir.Location("model.py", 1, 1).code_snippet is None

    def test_str_format(self):
        assert str(ir.Location("model.py", 83, 5)) == "model.py:83:5"

    def test_unknown(self):
        loc = ir.Location.unknown()
        assert loc == ir.Location("<unknown>", 0, 0, None)
        assert str(loc) == "<unknown>:0:0"

    def test_frozen(self):
        loc = ir.Location("a.py", 1, 1)
        with pytest.raises(FrozenInstanceError):
            loc.line = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Effects
# ---------------------------------------------------------------------------


class TestEffects:
    def test_constants(self):
        assert ir.EFFECT_PURE == "pure"
        assert ir.EFFECT_WRITE == "write"
        assert ir.EFFECT_READ == "read"
        assert ir.EFFECT_COLLECTIVE == "collective"
        assert ir.EFFECT_CALLBACK == "callback"

    def test_effect_kinds(self):
        assert ir.EFFECT_KINDS == frozenset(
            {"pure", "write", "read", "collective", "callback"}
        )
        assert isinstance(ir.EFFECT_KINDS, frozenset)
        for kind in (
            ir.EFFECT_PURE,
            ir.EFFECT_WRITE,
            ir.EFFECT_READ,
            ir.EFFECT_COLLECTIVE,
            ir.EFFECT_CALLBACK,
        ):
            assert kind in ir.EFFECT_KINDS


# ---------------------------------------------------------------------------
# 4. Module
# ---------------------------------------------------------------------------


class TestModule:
    def test_defaults(self):
        module = ir.Module()
        assert module.name == "main"
        assert module.functions == []
        assert module.metadata == {}
        assert module.version == ir.IR_FORMAT_VERSION

    def test_add_function_wires_parent(self):
        module = ir.Module()
        function = _bare_function("f")
        assert function.parent is None
        returned = module.add_function(function)
        assert returned is function
        assert module.functions == [function]
        assert function.parent is module

    def test_get_function(self):
        module = ir.Module()
        function = module.add_function(_bare_function("f"))
        assert module.get_function("f") is function
        with pytest.raises(KeyError, match="no function named 'nope'"):
            module.get_function("nope")

    def test_main_single_function(self):
        module = ir.Module()
        function = module.add_function(_bare_function("f"))
        assert module.main is function

    def test_main_requires_exactly_one_function(self):
        with pytest.raises(ValueError, match="requires exactly one"):
            ir.Module().main
        module = ir.Module()
        module.add_function(_bare_function("f"))
        module.add_function(_bare_function("g"))
        with pytest.raises(ValueError, match="requires exactly one"):
            module.main

    def test_id_counters_are_monotonic(self):
        module = ir.Module()
        assert module.new_op_id() == 0
        assert module.new_op_id() == 1
        assert module.new_op_id() == 2
        assert module.new_value_id() == 0
        assert module.new_value_id() == 1
        assert module.new_value_id() == 2


# ---------------------------------------------------------------------------
# 5. Function
# ---------------------------------------------------------------------------


class TestFunction:
    def test_post_init_wires_region_parent(self):
        region = ir.Region()
        region.append_block(ir.Block())
        function = ir.Function(name="f", input_types=(), region=region)
        assert function.region is region
        assert region.parent is function
        assert function.parent is None

    def test_entry_block(self):
        region = ir.Region()
        block = ir.Block()
        region.append_block(block)
        function = ir.Function(name="f", input_types=(), region=region)
        assert function.entry_block is block

    def test_output_types_reads_return_operand_types(self):
        region = ir.Region()
        block = ir.Block()
        region.append_block(block)
        v1 = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        v2 = ir.Value(id=1, type=ir.ValueType(np.float64, (2, 3)), owner=block, index=1)
        block.append(ir.Op(name="return", id=0, operands=(v1, v2)))
        function = ir.Function(name="f", input_types=(v1.type, v2.type), region=region)
        assert function.output_types == (v1.type, v2.type)

    def test_output_types_requires_terminator(self):
        function = _bare_function("f")
        with pytest.raises(ValueError, match="has no terminator"):
            function.output_types

    def test_input_types_and_metadata_preserved(self):
        region = ir.Region()
        region.append_block(ir.Block())
        f32_4 = _f32_4()
        function = ir.Function(
            name="f", input_types=(f32_4,), region=region, metadata={"note": "hi"}
        )
        assert function.input_types == (f32_4,)
        assert function.metadata == {"note": "hi"}


# ---------------------------------------------------------------------------
# 6. Region
# ---------------------------------------------------------------------------


class TestRegion:
    def test_entry_requires_blocks(self):
        with pytest.raises(ValueError, match="no blocks"):
            ir.Region().entry

    def test_single_block(self):
        region = ir.Region()
        assert region.single_block is False
        region.append_block(ir.Block())
        assert region.single_block is True
        region.append_block(ir.Block())
        assert region.single_block is False

    def test_append_block_wires_parent(self):
        region = ir.Region()
        block = ir.Block()
        assert block.parent is None
        assert region.append_block(block) is block
        assert region.blocks == [block]
        assert block.parent is region
        assert region.entry is block

    def test_insert_block_wires_parent(self):
        region = ir.Region()
        first = region.append_block(ir.Block())
        second = ir.Block()
        assert region.insert_block(0, second) is second
        assert region.blocks == [second, first]
        assert second.parent is region
        assert region.entry is second

    def test_entry_block_argument_ownership(self):
        region = ir.Region()
        block = ir.Block()
        arg = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        block.arguments = (arg,)
        region.append_block(block)
        entry_arg = region.entry.arguments[0]
        assert entry_arg is arg
        assert entry_arg.owner is block
        assert entry_arg.is_block_arg


# ---------------------------------------------------------------------------
# 7. Block
# ---------------------------------------------------------------------------


class TestBlock:
    def test_terminator_none_when_empty(self):
        assert ir.Block().terminator is None

    def test_terminator_none_when_last_op_is_not_terminator(self):
        block = ir.Block()
        block.append(ir.Op(name="add", id=0))
        assert block.terminator is None

    def test_terminator_requires_last_position(self):
        block = ir.Block()
        block.append(ir.Op(name="return", id=0))
        block.append(ir.Op(name="add", id=1))
        assert block.terminator is None

    def test_terminator_is_last_op(self):
        block = ir.Block()
        block.append(ir.Op(name="add", id=0))
        ret = block.append(ir.Op(name="return", id=1))
        assert block.terminator is ret

    def test_append_wires_parent(self):
        block = ir.Block()
        op = ir.Op(name="add", id=0)
        assert op.parent is None
        assert block.append(op) is op
        assert block.ops == [op]
        assert op.parent is block

    def test_insert_wires_parent(self):
        block = ir.Block()
        block.append(ir.Op(name="add", id=0))
        op = ir.Op(name="negate", id=1)
        assert block.insert(0, op) is op
        assert block.ops[0] is op
        assert op.parent is block

    def test_erase_removes_and_clears_parent(self):
        block = ir.Block()
        op = block.append(ir.Op(name="add", id=0))
        block.erase(op)
        assert op not in block.ops
        assert op.parent is None
        assert len(block) == 0

    def test_erase_absent_op_raises(self):
        block = ir.Block()
        op = ir.Op(name="add", id=0)
        with pytest.raises(ValueError, match="is not in this block"):
            block.erase(op)

    def test_iter_and_len(self):
        block = ir.Block()
        ops = [ir.Op(name="add", id=0), ir.Op(name="negate", id=1)]
        for op in ops:
            block.append(op)
        assert list(block) == ops
        assert len(block) == 2


# ---------------------------------------------------------------------------
# 8. Op
# ---------------------------------------------------------------------------


class TestOp:
    def test_structural_fields(self):
        block = ir.Block()
        region = ir.Region()
        loc = ir.Location("model.py", 83, 5)
        op = ir.Op(
            name="add",
            id=7,
            operands=(),
            attributes={"x": 1},
            regions=(region,),
            results=(),
            location=loc,
        )
        op.parent = block
        result = ir.Value(id=9, type=_f32_4(), owner=op, index=0)
        op.results = (result,)
        assert op.name == "add"
        assert op.id == 7
        assert op.operands == ()
        assert op.attributes == {"x": 1}
        assert op.regions == (region,)
        assert op.results == (result,)
        assert op.location is loc
        assert op.parent is block

    def test_opdef_resolves(self):
        op = ir.Op(name="add", id=0)
        assert op.opdef is ir.opdef("add")
        assert op.opdef.name == "add"

    def test_opdef_unknown_name_raises(self):
        op = ir.Op(name="definitely_not_an_op", id=0)
        with pytest.raises(KeyError, match="definitely_not_an_op"):
            op.opdef

    @pytest.mark.parametrize(
        "name,effect",
        [
            ("add", "pure"),
            ("rank", "read"),
            ("all_reduce", "collective"),
            ("runtime_call", "callback"),
        ],
    )
    def test_effect_property(self, name, effect):
        op = ir.Op(name=name, id=0)
        assert op.effect == effect
        assert op.effect in ir.EFFECT_KINDS

    def test_is_terminator(self):
        assert ir.Op(name="return", id=0).is_terminator is True
        assert ir.Op(name="add", id=0).is_terminator is False

    def test_result_single(self):
        op = ir.Op(name="add", id=0)
        result = ir.Value(id=1, type=_f32_4(), owner=op, index=0)
        op.results = (result,)
        assert op.result is result

    def test_result_requires_exactly_one(self):
        with pytest.raises(ValueError, match="expected exactly 1"):
            ir.Op(name="add", id=0).result
        op = ir.Op(name="add", id=0)
        op.results = (
            ir.Value(id=1, type=_f32_4(), owner=op, index=0),
            ir.Value(id=2, type=_f32_4(), owner=op, index=1),
        )
        with pytest.raises(ValueError, match="expected exactly 1"):
            op.result


# ---------------------------------------------------------------------------
# 9. Value / Use
# ---------------------------------------------------------------------------


class TestValueUse:
    def test_block_arg_identity(self):
        block = ir.Block()
        value = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        assert value.is_block_arg is True
        assert value.is_op_result is False
        assert value.defining_op is None
        assert value.owner is block
        assert value.index == 0
        assert value.uses == []

    def test_op_result_identity(self):
        op = ir.Op(name="add", id=0)
        value = ir.Value(id=1, type=_f32_4(), owner=op, index=0)
        assert value.is_op_result is True
        assert value.is_block_arg is False
        assert value.defining_op is op
        assert value.owner is op
        assert value.index == 0

    def test_add_use_is_idempotent(self):
        block = ir.Block()
        value = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        op = ir.Op(name="add", id=0)
        value.add_use(ir.Use(op, 0))
        assert len(value.uses) == 1
        # structurally equal Use → not duplicated
        value.add_use(ir.Use(op, 0))
        assert len(value.uses) == 1
        # different operand index → recorded
        value.add_use(ir.Use(op, 1))
        assert len(value.uses) == 2

    def test_remove_use(self):
        block = ir.Block()
        value = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        use = ir.Use(ir.Op(name="add", id=0), 0)
        value.add_use(use)
        value.remove_use(use)
        assert value.uses == []

    def test_remove_unrecorded_use_raises(self):
        block = ir.Block()
        value = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        use = ir.Use(ir.Op(name="add", id=0), 0)
        with pytest.raises(ValueError, match="is not recorded on value %0"):
            value.remove_use(use)

    def test_use_value_resolves_through_owner_operands(self):
        block = ir.Block()
        arg0 = ir.Value(id=0, type=_f32_4(), owner=block, index=0)
        arg1 = ir.Value(id=1, type=_f32_4(), owner=block, index=1)
        op = ir.Op(name="multiply", id=0, operands=(arg0, arg1))
        assert ir.Use(op, 0).value is arg0
        assert ir.Use(op, 1).value is arg1
        # Use equality is structural (owner + operand index)
        assert ir.Use(op, 0) == ir.Use(op, 0)
        assert ir.Use(op, 0) != ir.Use(op, 1)


# ---------------------------------------------------------------------------
# 10. Use-def chains (built via the Builder)
# ---------------------------------------------------------------------------


def _small_add_mul_module():
    """module: main(arg0, arg1) { s = add(arg0, arg1); m = multiply(s, s); return m }"""
    module = ir.Module()
    builder = ir.Builder(module)
    function = builder.build_function("main", (_f32_4(), _f32_4()))
    arg0, arg1 = function.entry_block.arguments
    add_op = builder.create("add", operands=(arg0, arg1))
    s = add_op.result
    mul_op = builder.create("multiply", operands=(s, s))
    out = mul_op.result
    ret = builder.set_terminator(builder.current_block, "return", operands=(out,))
    return module, function, arg0, arg1, add_op, s, mul_op, out, ret


class TestUseDefChains:
    def test_operand_use_records(self):
        module, function, arg0, arg1, add_op, s, mul_op, out, ret = (
            _small_add_mul_module()
        )
        # add consumes the two block args at operand indices 0 and 1.
        assert add_op.operands == (arg0, arg1)
        assert len(arg0.uses) == 1 and arg0.uses[0].owner is add_op
        assert arg0.uses[0].operand_index == 0
        assert arg1.uses[0].owner is add_op and arg1.uses[0].operand_index == 1
        # s feeds multiply at BOTH operand positions.
        assert mul_op.operands == (s, s)
        assert sorted(u.operand_index for u in s.uses) == [0, 1]
        assert all(u.owner is mul_op for u in s.uses)
        # return consumes multiply's result.
        assert ret.operands == (out,)
        assert len(out.uses) == 1 and out.uses[0].owner is ret
        ir.verify(module)  # use bookkeeping is consistent

    def test_use_records_match_operand_slots(self):
        module, function, arg0, arg1, add_op, s, mul_op, out, ret = (
            _small_add_mul_module()
        )
        for value in (arg0, arg1, s, out):
            for use in value.uses:
                assert use.owner.operands[use.operand_index] is value

    def test_replace_all_uses_with(self):
        module, function, arg0, arg1, add_op, s, mul_op, out, ret = (
            _small_add_mul_module()
        )
        assert len(s.uses) == 2
        s.replace_all_uses_with(arg0)
        # multiply now consumes arg0 twice; s is drained.
        assert mul_op.operands == (arg0, arg0)
        assert s.uses == []
        # arg0 keeps its add use and gains the two multiply records.
        arg0_keys = sorted(
            ((u.owner, u.operand_index) for u in arg0.uses), key=lambda k: (k[0].id, k[1])
        )
        assert arg0_keys == sorted(
            [(add_op, 0), (mul_op, 0), (mul_op, 1)], key=lambda k: (k[0].id, k[1])
        )
        # same-type replacement → the module is still structurally valid.
        ir.verify(module)


# ---------------------------------------------------------------------------
# 11. Dim / DimExpr interop
# ---------------------------------------------------------------------------


class TestDimInterop:
    def test_dim_equality_includes_size(self):
        assert Dim("B", 8) == Dim("B", 8)
        assert Dim("B", 8) != Dim("B")
        assert Dim("B") == Dim("B")
        assert Dim("B") != Dim("C")
        assert Dim("B") != Dim("B", 9)

    def test_dim_expr_arithmetic(self):
        b = Dim("B")
        cases = [
            (b * 2, "mul", Dim("B"), 2),
            (2 * b, "mul", 2, Dim("B")),
            (b + 1, "add", Dim("B"), 1),
            (b - 1, "sub", Dim("B"), 1),
            (b // 2, "floordiv", Dim("B"), 2),
            (b % 3, "mod", Dim("B"), 3),
            (b.min(4), "min", Dim("B"), 4),
            (b.max(4), "max", Dim("B"), 4),
        ]
        for expr, op, left, right in cases:
            assert isinstance(expr, DimExpr)
            assert expr.op == op
            assert expr.left == left
            assert expr.right == right

    def test_dim_expr_structural_equality_and_frozen(self):
        assert Dim("B") * 2 == DimExpr("mul", Dim("B"), 2)
        assert Dim("B") * 2 != Dim("B") * 3
        expr = Dim("B") + 1
        with pytest.raises(FrozenInstanceError):
            expr.op = "mul"  # type: ignore[misc]

    def test_value_type_shape_accepts_dim_and_dim_expr(self):
        b = Dim("B")
        vt = ir.ValueType(np.float32, (b, b * 2, b + 1))
        assert vt.shape == (Dim("B"), Dim("B") * 2, Dim("B") + 1)
        assert vt.rank == 3


# ---------------------------------------------------------------------------
# 12. Builder integration (structure sanity)
# ---------------------------------------------------------------------------


class TestBuilderIntegration:
    def test_module_structure(self):
        module = ir.Module()
        builder = ir.Builder(module)
        f32_4 = _f32_4()
        function = builder.build_function("main", (f32_4, f32_4), metadata={"tag": "x"})
        # the function is registered on the module
        assert module.functions == [function]
        assert module.functions[0] is function
        assert module.get_function("main") is function
        assert function.metadata == {"tag": "x"}
        # entry block + block arguments
        entry = function.entry_block
        assert entry.parent is function.region
        assert function.input_types == (f32_4, f32_4)
        assert [arg.type for arg in entry.arguments] == [f32_4, f32_4]
        assert all(arg.owner is entry for arg in entry.arguments)
        assert all(arg.is_block_arg for arg in entry.arguments)
        # ops are wired to the entry block; results to their defining op
        add_op = builder.create("add", operands=entry.arguments)
        assert add_op.parent is entry
        assert add_op.results[0].owner is add_op
        assert add_op.results[0].index == 0
        assert add_op.results[0].is_op_result
        # value ids are module-unique and assigned in order
        values = list(entry.arguments) + [r for op in entry.ops for r in op.results]
        ids = [value.id for value in values]
        assert ids == [0, 1, 2]
        assert len(ids) == len(set(ids))

    def test_build_function_requires_value_types(self):
        builder = ir.Builder(ir.Module())
        with pytest.raises(TypeError, match="must be ValueType instances"):
            builder.build_function("main", (np.float32,))  # type: ignore[arg-type]
