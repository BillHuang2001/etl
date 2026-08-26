"""Builder (op-construction API) tests for etl's EvoXIR IR layer.

Covers every documented Builder contract in ``etl/ir/builder.py``:

* module/function/region construction wiring (parents, block args, ids,
  insertion point, metadata);
* ``create``/``emit`` basics (id assignment, operand ``Use`` records, op
  insertion order, parent pointers, locations);
* eager failure paths (no insertion point, unknown ops, arity/operand/region
  count/attribute-schema violations, ``emit`` on multi-result ops);
* attribute normalization per tag (dtype -> name string, sequences -> tuples,
  nullable ints only where the default is None, nested-ints strictness,
  float vs bool, defaults applied, non-dict rejection);
* result-type resolution order: explicit -> ``shape_fn`` -> op-specific
  (``constant``/``call``/``if``/``runtime_call``/``block_call``), with
  ``ShapeError`` propagating unchanged;
* the insertion-point stack (``set_insertion_point``, ``push_region``/
  ``pop_region``, ``current_block``/``current_region``);
* ``insert_block``/``set_terminator`` semantics;
* per-module id-counter isolation.

CPU only; no GPU/network. Fast: every test builds tiny modules.
"""

import numpy as np
import pytest

from etl import ir
from etl.core import Dim, DimExpr, ShapeError, TensorSpec, VerificationError


# ---------------------------------------------------------------------------
# helpers & fixtures
# ---------------------------------------------------------------------------

def _vt(dtype_name, shape):
    """Shortcut: a ``ValueType`` from a dtype name and shape."""
    return ir.ValueType(np.dtype(dtype_name), shape)


def _build_f32_pair(b, shape_a=(3, 1), shape_b=(1, 4)):
    """Build a function taking two f32 values; return (function, (x, y))."""
    f = b.build_function(
        "main", (_vt("float32", shape_a), _vt("float32", shape_b))
    )
    return f, f.entry_block.arguments


def _build_int32(b, shape=(2, 3)):
    """Build a function taking one int32 value; return (function, x)."""
    f = b.build_function("main", (_vt("int32", shape),))
    return f, f.entry_block.arguments[0]


def _emit_bool_const(b, shape=(2,)):
    """Emit a boolean constant usable as an ``if`` predicate."""
    return b.emit("constant", (), attributes={"value": np.ones(shape, bool)})


@pytest.fixture
def _b():
    """A fresh Builder with a fresh (empty) module attached."""
    b = ir.Builder()
    b.build_module()
    return b


@pytest.fixture
def _module(_b):
    """The module attached to the ``_b`` fixture builder."""
    return _b.module


# ---------------------------------------------------------------------------
# 1. build_module / build_function / build_region wiring
# ---------------------------------------------------------------------------

class TestBuildModuleAndFunction:
    def test_build_module_attaches_and_resets(self):
        b = ir.Builder()
        assert b.module is None
        m = b.build_module("net", metadata={"seed": 7})
        assert b.module is m
        assert m.name == "net"
        assert m.metadata == {"seed": 7}
        assert m.functions == []
        assert ir.Builder().build_module().metadata == {}
        # building a new module resets the insertion-point stack
        b.build_function("f", ())
        assert b.current_block is not None
        m2 = b.build_module()
        assert b.module is m2
        assert b.module is not m
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.current_block

    def test_build_function_wiring(self, _b, _module):
        types = (_vt("float32", (3, 1)), _vt("int32", (2,)))
        f = _b.build_function("main", types, metadata={"k": 1})
        assert f.name == "main"
        assert _module.functions == [f]
        assert f.parent is _module
        assert f.metadata == {"k": 1}
        assert f.input_types == types
        block = f.entry_block
        assert block.parent is f.region
        assert f.region.parent is f
        args = block.arguments
        assert len(args) == 2
        for i, arg in enumerate(args):
            assert isinstance(arg, ir.Value)
            assert arg.owner is block
            assert arg.index == i
            assert arg.type == types[i]
            assert arg.uses == []
        assert len({a.id for a in args}) == 2  # fresh, unique value ids
        assert _b.current_block is block
        assert _b.current_region is f.region

    def test_build_function_without_module(self):
        with pytest.raises(RuntimeError, match="no module"):
            ir.Builder().build_function("f", ())

    def test_build_function_rejects_non_valuetype_inputs(self, _b):
        with pytest.raises(TypeError, match="ValueType"):
            _b.build_function("f", (5,))

    def test_build_region_detached_with_fresh_args(self, _b):
        f = _b.build_function("f", (_vt("float32", (2,)),))
        t = _vt("float32", (2,))
        r = _b.build_region((t,))
        assert r.parent is None  # detached until attached to an op
        assert len(r.blocks) == 1
        block = r.entry
        (arg,) = block.arguments
        assert isinstance(arg, ir.Value)
        assert arg.owner is block
        assert arg.index == 0
        assert arg.type == t
        # fresh id from the module counters, distinct from the function args
        assert arg.id not in {v.id for v in f.entry_block.arguments}

    def test_build_region_defaults_and_errors(self, _b):
        assert _b.build_region().entry.arguments == ()
        with pytest.raises(RuntimeError, match="no module"):
            ir.Builder().build_region()
        with pytest.raises(TypeError, match="ValueType"):
            _b.build_region((5,))


# ---------------------------------------------------------------------------
# 2. create/emit basics: ids, uses, parent pointers, ordering
# ---------------------------------------------------------------------------

class TestCreateEmitBasics:
    def test_emit_order_ids_parent(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        ops = [
            _b.emit("add", (x, y)).owner,
            _b.emit("multiply", (x, y)).owner,
            _b.emit("subtract", (x, y)).owner,
        ]
        # inserted at the insertion point in order; the point advances
        assert f.entry_block.ops == ops
        # op ids are module-unique and increasing
        assert [op.id for op in ops] == [0, 1, 2]
        assert len({op.id for op in ops}) == 3
        for op in ops:
            assert op.parent is f.entry_block
        # result value ids continue the module counters after the arg ids
        arg_ids = {v.id for v in (x, y)}
        result_ids = [op.result.id for op in ops]
        assert result_ids == [2, 3, 4]
        assert arg_ids.isdisjoint(result_ids)

    def test_operand_use_records(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        add_op = _b.create("add", (x, y))
        assert add_op.operands == (x, y)
        assert len(x.uses) == 1
        (use,) = x.uses
        assert use.owner is add_op
        assert use.operand_index == 0
        assert use.value is x
        assert len(y.uses) == 1
        assert y.uses[0].owner is add_op
        assert y.uses[0].operand_index == 1
        # reusing a value records one Use per operand position
        mul_op = _b.create("multiply", (x, x))
        assert [(u.owner, u.operand_index) for u in x.uses] == [
            (add_op, 0),
            (mul_op, 0),
            (mul_op, 1),
        ]

    def test_result_owner_and_index(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        op = _b.create("add", (x, y))
        (result,) = op.results
        assert result.owner is op
        assert result.index == 0
        assert result.type == _vt("float32", (3, 4))

    def test_location_stored(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        loc = ir.Location("model.py", 42, 7, "z = a + b")
        op = _b.create("add", (x, y), location=loc)
        assert op.location is loc
        assert op.location is not None and op.location.file == "model.py"

    def test_insert_block_inserts_at_point(self, _b):
        # an op emitted after set_insertion_point lands in the target block
        f = _b.build_function("f", ())
        blk = _b.insert_block(f.region)
        _b.set_insertion_point(blk)
        op = _b.emit("constant", (), attributes={"value": np.zeros(1, np.float32)})
        assert op.owner.parent is blk
        assert blk.ops == [op.owner]
        assert f.entry_block.ops == []


# ---------------------------------------------------------------------------
# 3. runtime error paths
# ---------------------------------------------------------------------------

class TestNoInsertionPoint:
    def test_create_requires_insertion_point(self):
        b = ir.Builder()
        b.build_module()
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.create("constant", (), attributes={"value": np.zeros(1, np.float32)})
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.emit("constant", (), attributes={"value": np.zeros(1, np.float32)})

    def test_current_block_and_region_require_point(self):
        b = ir.Builder()
        b.build_module()
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.current_block
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.current_region

    def test_emit_requires_exactly_one_result(self, _b):
        _b.build_function("f", ())
        with pytest.raises(ValueError, match="has 0 results, expected exactly 1"):
            _b.emit("return", ())


# ---------------------------------------------------------------------------
# 4. unknown ops
# ---------------------------------------------------------------------------

class TestUnknownOp:
    def test_unknown_op_raises_keyerror(self, _b):
        with pytest.raises(KeyError, match="nope"):
            _b.create("nope")
        with pytest.raises(KeyError, match="definitely_not_an_op"):
            _b.emit("definitely_not_an_op", ())


# ---------------------------------------------------------------------------
# 5. arity & operand-kind violations
# ---------------------------------------------------------------------------

class TestArityAndOperandErrors:
    def test_arity_violations(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        with pytest.raises(VerificationError, match="expects arity 2, got 1"):
            _b.create("add", (x,))
        with pytest.raises(VerificationError, match="expects arity 2, got 3"):
            _b.create("add", (x, y, x))
        with pytest.raises(VerificationError, match="expects arity 2, got 0"):
            _b.create("add", ())
        # variadic concatenate still has a minimum of 1
        with pytest.raises(
            VerificationError, match=r"expects arity \(1, None\), got 0"
        ):
            _b.create("concatenate", (), attributes={"axis": 0})

    @pytest.mark.parametrize("bad", [3, 3.5, "x", None, [1, 2], np.float32(1.0)])
    def test_non_value_operand(self, _b, bad):
        f, (x, y) = _build_f32_pair(_b)
        with pytest.raises(VerificationError, match="not an SSA Value"):
            _b.create("add", (x, bad))
        # eager: nothing was inserted
        assert f.entry_block.ops == []

    def test_region_count_violation(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        pred = _emit_bool_const(_b)
        with pytest.raises(VerificationError, match=r"declares 2 region\(s\), got 0"):
            _b.create("if", (pred, x))


# ---------------------------------------------------------------------------
# 6. attribute schema validation & normalization
# ---------------------------------------------------------------------------

class TestAttributeSchema:
    def test_missing_required_attribute(self, _b):
        f, x = _build_int32(_b)
        with pytest.raises(
            VerificationError, match="missing required attribute 'dtype'"
        ):
            _b.create("cast", (x,))

    def test_unknown_attribute(self, _b):
        f, x = _build_int32(_b)
        with pytest.raises(VerificationError, match="unknown attribute"):
            _b.create("cast", (x,), attributes={"dtype": "float32", "surprise": 1})

    def test_attributes_must_be_dict(self, _b):
        f, x = _build_int32(_b)
        for bad in (5, "dtype", [("dtype", "float32")]):
            with pytest.raises(VerificationError, match="attributes must be a dict"):
                _b.create("transpose", (x,), attributes=bad)

    def test_attributes_none_means_empty_with_defaults(self, _b):
        f, x = _build_int32(_b)
        op = _b.create("transpose", (x,))  # permutation default None
        assert op.attributes == {"permutation": None}

    @pytest.mark.parametrize(
        "value,expected",
        [(np.float64, "float64"), ("int32", "int32"), (np.dtype("bool"), "bool")],
    )
    def test_dtype_attribute_normalization(self, _b, value, expected):
        f, x = _build_int32(_b)
        op = _b.create("cast", (x,), attributes={"dtype": value})
        assert op.attributes["dtype"] == expected

    def test_dtype_attribute_rejects_unknown(self, _b):
        f, x = _build_int32(_b)
        with pytest.raises(VerificationError, match="expected a dtype"):
            _b.create("cast", (x,), attributes={"dtype": "nonsense"})

    @pytest.mark.parametrize(
        "shape,value,expected",
        [
            ((2, 3), None, None),
            ((2, 3), [1, 0], (1, 0)),
            ((2, 3, 4), [2, 1, 0], (2, 1, 0)),
        ],
    )
    def test_transpose_permutation(self, _b, shape, value, expected):
        f, x = _build_int32(_b, shape)
        op = _b.create("transpose", (x,), attributes={"permutation": value})
        assert op.attributes["permutation"] == expected

    def test_transpose_permutation_rejects_bad_entries(self, _b):
        f, x = _build_int32(_b, (2, 3))
        with pytest.raises(VerificationError, match="expected ints, got entry True"):
            _b.create("transpose", (x,), attributes={"permutation": [True, 0]})
        with pytest.raises(VerificationError, match="expected a sequence of ints"):
            _b.create("transpose", (x,), attributes={"permutation": 3})

    def test_int_attr_none_only_where_default_is_none(self, _b):
        f, x = _build_int32(_b, (2, 3))
        # argmax.axis has default None: explicit None is accepted...
        op = _b.create("argmax", (x,), attributes={"axis": None})
        assert op.attributes["axis"] is None
        # ...and the default is applied when omitted
        op2 = _b.create("argmax", (x,))
        assert op2.attributes == {"axis": None, "keepdims": False}
        # scatter.axis has default 0 (not nullable): None is rejected
        with pytest.raises(VerificationError, match="expected an int, got None"):
            _b.create("scatter", (x, x, x), attributes={"axis": None})

    def test_bool_attr_strictness(self, _b):
        f, x = _build_int32(_b)
        with pytest.raises(VerificationError, match="expected a bool, got 1"):
            _b.create(
                "reduce_sum",
                (x,),
                attributes={"axes": (0,), "reduce_op": "sum", "keepdims": 1},
            )

    def test_pad_nested_ints(self, _b):
        f, x = _build_int32(_b, (3,))
        # bare int entries fail (known strictness: pairs required)
        with pytest.raises(VerificationError, match="expected int tuples"):
            _b.create("pad", (x,), attributes={"padding_config": [1, 1]})
        with pytest.raises(VerificationError, match="expected a sequence of int tuples"):
            _b.create("pad", (x,), attributes={"padding_config": 1})
        # pairs normalize to nested tuples; fill value default applied
        op = _b.create("pad", (x,), attributes={"padding_config": [(1, 1)]})
        assert op.attributes == {"padding_config": ((1, 1),), "value": 0.0}

    def test_reduce_sum_defaults_applied(self, _b):
        f, x = _build_int32(_b)
        op = _b.create(
            "reduce_sum", (x,), attributes={"axes": [0], "reduce_op": "sum"}
        )
        assert op.attributes == {"axes": (0,), "keepdims": False, "reduce_op": "sum"}

    def test_float_attr_accepts_int_rejects_bool(self, _b):
        f, x = _build_int32(_b, (3,))
        op = _b.create(
            "pad", (x,), attributes={"padding_config": ((1, 1),), "value": 2}
        )
        assert op.attributes["value"] == 2
        with pytest.raises(VerificationError, match="expected a float, got True"):
            _b.create(
                "pad",
                (x,),
                attributes={"padding_config": ((1, 1),), "value": True},
            )

    def test_shape_attr_entries_and_normalization(self, _b):
        f, x = _build_int32(_b, (2, 3))
        # lists normalize to tuples; symbolic dims pass through
        bdim = Dim("B")
        op = _b.create("reshape", (x,), attributes={"shape": [bdim, 3]})
        assert op.attributes["shape"] == (bdim, 3)
        with pytest.raises(VerificationError, match="invalid shape dim True"):
            _b.create("reshape", (x,), attributes={"shape": (True, 3)})

    def test_ndarray_attr_requires_ndarray(self, _b):
        with pytest.raises(VerificationError, match="expected a numpy array"):
            _b.create("constant", (), attributes={"value": [[1, 2], [3, 4]]})

    def test_attributes_stored_as_normalized_copy(self, _b):
        f, x = _build_int32(_b)
        attrs = {"axes": [0], "reduce_op": "sum"}
        op = _b.create("reduce_sum", (x,), attributes=attrs)
        attrs["axes"] = [1]  # mutating the caller's dict must not leak
        assert op.attributes["axes"] == (0,)
        assert op.attributes["keepdims"] is False


# ---------------------------------------------------------------------------
# 7. shape_fn resolution & ShapeError propagation
# ---------------------------------------------------------------------------

class TestShapeFnResolution:
    def test_broadcast_add(self, _b):
        f, (x, y) = _build_f32_pair(_b, (3, 1), (1, 4))
        v = _b.emit("add", (x, y))
        assert v.type == _vt("float32", (3, 4))

    def test_broadcast_symbolic_dims(self, _b):
        bdim = Dim("B")
        f = _b.build_function(
            "main", (_vt("float32", (bdim, 1)), _vt("float32", (1, 4)))
        )
        a, c = f.entry_block.arguments
        v = _b.emit("add", (a, c))
        assert v.type.shape == (bdim, 4)

    def test_broadcast_unequal_symbolic_dims_defer(self, _b):
        m, n = Dim("M"), Dim("N")
        f = _b.build_function(
            "main", (_vt("float32", (m, 1)), _vt("float32", (n, 4)))
        )
        a, c = f.entry_block.arguments
        v = _b.emit("add", (a, c))
        assert v.type.shape[0] == DimExpr("max", m, n)

    def test_shape_error_propagates_unwrapped(self, _b):
        f, (a, c) = _build_f32_pair(_b, (3,), (4,))
        with pytest.raises(
            ShapeError, match="cannot broadcast incompatible dims 3 and 4"
        ) as exc_info:
            _b.emit("add", (a, c))
        # NOT wrapped in a VerificationError
        assert not isinstance(exc_info.value, VerificationError)
        # eager failure: nothing was inserted
        assert f.entry_block.ops == []

    @pytest.mark.parametrize(
        "op_name,attrs,expected_dtype",
        [
            ("divide", None, "int32"),  # CURRENT contract: np.result_type
            ("sqrt", None, "int32"),  # CURRENT contract: dtype-preserving unary
            ("reduce_sum", {"axes": (1,), "reduce_op": "sum"}, "int64"),
            ("reduce_mean", {"axes": (1,), "reduce_op": "mean"}, "float64"),
            ("reduce_max", {"axes": (1,), "reduce_op": "max"}, "int32"),
            ("equal", None, "bool"),
        ],
    )
    def test_result_dtypes(self, _b, op_name, attrs, expected_dtype):
        f, x = _build_int32(_b)
        operands = (x, x) if op_name in ("divide", "equal") else (x,)
        v = _b.emit(op_name, operands, attributes=attrs)
        assert v.type.dtype == np.dtype(expected_dtype)

    def test_rank_is_scalar_int64(self, _b):
        _b.build_function("f", ())
        v = _b.emit("rank", (), attributes={"group": "world"})
        assert v.type == _vt("int64", ())


# ---------------------------------------------------------------------------
# 8. explicit result_types
# ---------------------------------------------------------------------------

class TestExplicitResultTypes:
    def test_explicit_bypasses_shape_fn(self, _b):
        # shape_fn would raise ShapeError on (3,) vs (4,) — explicit wins
        f, (a, c) = _build_f32_pair(_b, (3,), (4,))
        target = _vt("float32", (7,))
        op = _b.create("add", (a, c), result_types=(target,))
        assert op.result.type == target

    def test_explicit_overrides_successful_shape_fn(self, _b):
        f, (a, c) = _build_f32_pair(_b, (3, 1), (1, 4))
        v = _b.emit("add", (a, c), result_type=_vt("float32", (5, 5)))
        assert v.type == _vt("float32", (5, 5))

    def test_explicit_result_count_enforced(self, _b):
        f, (a, c) = _build_f32_pair(_b)
        with pytest.raises(VerificationError, match="declares 1 result"):
            _b.create(
                "add",
                (a, c),
                result_types=(_vt("float32", (2,)), _vt("float32", (2,))),
            )

    def test_explicit_non_valuetype_rejected(self, _b):
        f, (a, c) = _build_f32_pair(_b)
        with pytest.raises(VerificationError, match="result_types must be ValueType"):
            _b.create("add", (a, c), result_types=(5,))


# ---------------------------------------------------------------------------
# 9. constant: result type from the payload
# ---------------------------------------------------------------------------

class TestConstant:
    def test_result_type_from_payload(self, _b):
        _b.build_function("f", ())
        v = _b.emit(
            "constant", (), attributes={"value": np.zeros((2, 3), np.int32)}
        )
        assert v.type == _vt("int32", (2, 3))

    def test_payload_stored_verbatim(self, _b):
        _b.build_function("f", ())
        payload = np.arange(6, dtype=np.float32).reshape(2, 3)
        op = _b.create("constant", (), attributes={"value": payload})
        assert op.attributes["value"] is payload
        assert op.attributes["value"].dtype == np.dtype("float32")

    def test_missing_value_attr(self, _b):
        with pytest.raises(
            VerificationError, match="missing required attribute 'value'"
        ):
            _b.create("constant", ())

    def test_value_must_be_ndarray(self, _b):
        with pytest.raises(VerificationError, match="expected a numpy array"):
            _b.create("constant", (), attributes={"value": [[1, 2]]})


# ---------------------------------------------------------------------------
# 10. call: result types from the callee signature
# ---------------------------------------------------------------------------

class TestCall:
    @staticmethod
    def _make_helper():
        b = ir.Builder()
        b.build_module()
        helper = b.build_function(
            "helper", (_vt("float32", (2,)), _vt("float32", (2,)))
        )
        b.set_terminator(helper.entry_block, "return", helper.entry_block.arguments)
        return b

    def test_results_from_callee_signature(self):
        b = self._make_helper()
        main = b.build_function(
            "main", (_vt("float32", (2,)), _vt("float32", (2,)))
        )
        a, c = main.entry_block.arguments
        op = b.create("call", (a, c), attributes={"callee": "helper"})
        assert op.operands == (a, c)
        expected = b.module.get_function("helper").output_types
        assert [r.type for r in op.results] == list(expected)
        assert len(op.results) == 2

    def test_unknown_callee(self, _b):
        with pytest.raises(VerificationError, match="no function named 'ghost'"):
            _b.create("call", (), attributes={"callee": "ghost"})

    def test_callee_without_terminator(self, _b):
        _b.build_function("pending", ())
        with pytest.raises(VerificationError, match="has no terminator"):
            _b.create("call", (), attributes={"callee": "pending"})

    def test_call_requires_callee_attr(self, _b):
        with pytest.raises(
            VerificationError, match="missing required attribute 'callee'"
        ):
            _b.create("call", ())


# ---------------------------------------------------------------------------
# 11. runtime_call / block_call result_specs
# ---------------------------------------------------------------------------

class TestResultSpecs:
    def test_valuetype_entries(self, _b):
        _b.build_function("f", ())
        vt = _vt("float32", (2, 3))
        op = _b.create(
            "runtime_call",
            (),
            attributes={"callback": "cb", "result_specs": (vt, vt)},
        )
        assert op.results[0].type == vt
        assert op.results[1].type == vt
        assert op.results[0].owner is op
        assert op.results[0].index == 0
        assert op.results[1].index == 1

    def test_tensorspec_entries_converted(self, _b):
        _b.build_function("f", ())
        spec = TensorSpec(shape=(2,), dtype="float32")
        op = _b.create(
            "runtime_call",
            (),
            attributes={"callback": "cb", "result_specs": (spec,)},
        )
        assert op.result.type == _vt("float32", (2,))

    def test_dict_entries_converted_and_defaults_applied(self, _b):
        _b.build_function("f", ())
        op = _b.create(
            "block_call",
            (),
            attributes={
                "block_name": "blk",
                "result_specs": ({"dtype": "float32", "shape": (2,)},),
            },
        )
        assert op.result.type == _vt("float32", (2,))
        assert op.attributes["static_args"] == ()

    @pytest.mark.parametrize(
        "bad_spec,match",
        [
            ({"dtype": "nonsense", "shape": (2,)}, "must map 'dtype'/'shape'"),
            ({"dtype": "float32"}, "must map 'dtype'/'shape'"),
            ({"shape": (2,)}, "must map 'dtype'/'shape'"),
            (5, "cannot interpret result spec"),
            ("float32", "cannot interpret result spec"),
        ],
    )
    def test_bad_spec_entries(self, _b, bad_spec, match):
        with pytest.raises(VerificationError, match=match):
            _b.create(
                "runtime_call",
                (),
                attributes={"callback": "cb", "result_specs": (bad_spec,)},
            )

    def test_result_specs_must_be_sequence(self, _b):
        with pytest.raises(
            VerificationError, match="'result_specs' must be a sequence"
        ):
            _b.create(
                "runtime_call",
                (),
                attributes={"callback": "cb", "result_specs": "nope"},
            )

    def test_result_specs_required(self, _b):
        with pytest.raises(
            VerificationError, match="missing required attribute 'result_specs'"
        ):
            _b.create("runtime_call", (), attributes={"callback": "cb"})


# ---------------------------------------------------------------------------
# 12. if / while regions
# ---------------------------------------------------------------------------

class TestControlFlowRegions:
    @staticmethod
    def _make_region(b, t):
        """A detached region with one typed arg returned by its terminator."""
        r = b.build_region((t,))
        b.set_terminator(r.entry, "return", (r.entry.arguments[0],))
        return r

    def test_if_result_types_from_branches(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        pred = _emit_bool_const(_b)
        t = x.type
        r_true = self._make_region(_b, t)
        r_false = self._make_region(_b, t)
        op = _b.create("if", (pred, x), regions=(r_true, r_false))
        assert [v.type for v in op.results] == [t]
        assert op.regions == (r_true, r_false)
        assert r_true.parent is op
        assert r_false.parent is op

    def test_if_mismatched_branches(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        pred = _emit_bool_const(_b)
        r_true = self._make_region(_b, _vt("float32", (2,)))
        r_false = self._make_region(_b, _vt("int32", (2,)))
        with pytest.raises(VerificationError, match="branch result types differ"):
            _b.create("if", (pred, x), regions=(r_true, r_false))

    def test_if_region_count_mismatch(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        pred = _emit_bool_const(_b)
        with pytest.raises(VerificationError, match=r"declares 2 region\(s\), got 0"):
            _b.create("if", (pred, x))
        r = self._make_region(_b, _vt("float32", (2,)))
        with pytest.raises(VerificationError, match=r"declares 2 region\(s\), got 1"):
            _b.create("if", (pred, x), regions=(r,))

    def test_if_branch_without_terminator(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        pred = _emit_bool_const(_b)
        no_ret = _b.build_region((_vt("float32", (2,)),))  # no return op
        r_false = self._make_region(_b, _vt("float32", (2,)))
        with pytest.raises(VerificationError, match="no 'return' terminator"):
            _b.create("if", (pred, x), regions=(no_ret, r_false))

    def test_if_branch_without_entry_block(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        pred = _emit_bool_const(_b)
        empty = ir.Region(blocks=[])
        r_false = self._make_region(_b, _vt("float32", (2,)))
        with pytest.raises(VerificationError, match="no entry block"):
            _b.create("if", (pred, x), regions=(empty, r_false))

    def test_while_results_from_operand_types(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        t = x.type
        r_cond = _b.build_region((t,))
        r_body = _b.build_region((t,))
        op = _b.create("while", (x,), regions=(r_cond, r_body))
        assert [v.type for v in op.results] == [t]
        assert op.regions == (r_cond, r_body)
        assert r_cond.parent is op
        assert r_body.parent is op

    def test_while_region_count_mismatch(self, _b):
        f, (x, _) = _build_f32_pair(_b, (2,), (2,))
        with pytest.raises(VerificationError, match=r"declares 2 region\(s\), got 0"):
            _b.create("while", (x,))


# ---------------------------------------------------------------------------
# 13. insertion-point stack
# ---------------------------------------------------------------------------

class TestInsertionPointStack:
    def test_set_insertion_point_block(self):
        b = ir.Builder()
        b.build_module()
        f1 = b.build_function("f1", (_vt("float32", (2,)),))
        f2 = b.build_function("f2", (_vt("float32", (2,)),))
        a1 = f1.entry_block.arguments[0]
        a2 = f2.entry_block.arguments[0]
        b.set_insertion_point(f1.entry_block)
        op = b.emit("negate", (a1,))
        assert op.owner.parent is f1.entry_block
        assert op.owner in f1.entry_block.ops
        assert op.owner not in f2.entry_block.ops
        # and back to f2 (also exercising region targets)
        b.set_insertion_point(f2.region)
        op2 = b.emit("negate", (a2,))
        assert op2.owner.parent is f2.region.entry

    def test_set_insertion_point_region_uses_entry(self, _b):
        f = _b.build_function("f", (_vt("float32", (2,)),))
        blk = _b.insert_block(f.region)
        _b.set_insertion_point(f.region)
        assert _b.current_block is f.region.entry
        assert _b.current_block is not blk

    @pytest.mark.parametrize("bad", [5, "block", None, _vt("float32", (2,))])
    def test_set_insertion_point_rejects_other_types(self, _b, bad):
        _b.build_function("f", ())
        with pytest.raises(TypeError, match="Block or Region"):
            _b.set_insertion_point(bad)

    def test_set_insertion_point_requires_existing_point(self):
        b = ir.Builder()
        b.build_module()
        with pytest.raises(RuntimeError, match="no insertion point"):
            b.set_insertion_point(ir.Block())

    def test_push_pop_roundtrip(self, _b):
        f, (x, y) = _build_f32_pair(_b)
        r = _b.build_region((x.type,))
        _b.push_region(r)
        assert _b.current_block is r.entry
        assert _b.current_region is r
        op = _b.emit("negate", (r.entry.arguments[0],))
        assert op.owner.parent is r.entry
        assert _b.pop_region() is r
        assert _b.current_block is f.entry_block
        assert _b.current_region is f.region

    def test_pop_region_on_empty_stack(self):
        with pytest.raises(RuntimeError, match="stack is empty"):
            ir.Builder().pop_region()

    def test_current_region_requires_parented_block(self, _b):
        _b.build_function("f", ())
        det = ir.Block()
        _b.set_insertion_point(det)
        assert _b.current_block is det
        with pytest.raises(RuntimeError, match="detached"):
            _b.current_region


# ---------------------------------------------------------------------------
# 14. insert_block / set_terminator
# ---------------------------------------------------------------------------

class TestInsertBlockAndSetTerminator:
    def test_insert_block_appends_empty_block(self, _b):
        f = _b.build_function("f", ())
        blk = _b.insert_block(f.region)
        assert f.region.blocks == [f.entry_block, blk]
        assert blk.parent is f.region
        assert blk.ops == []
        assert blk.arguments == ()
        assert blk.terminator is None

    def test_insert_block_at_position(self, _b):
        f = _b.build_function("f", ())
        blk = _b.insert_block(f.region, position=0)
        assert f.region.blocks[0] is blk
        assert f.region.entry is blk

    def test_insert_block_does_not_move_point(self, _b):
        f = _b.build_function("f", ())
        entry = f.entry_block
        _b.insert_block(f.region)
        assert _b.current_block is entry

    def test_set_terminator_appends_last(self, _b):
        f = _b.build_function("f", ())
        term = _b.set_terminator(f.entry_block, "return", ())
        assert f.entry_block.ops == [term]
        assert f.entry_block.terminator is term
        assert term.name == "return"
        assert term.is_terminator
        assert term.parent is f.entry_block
        assert term.results == ()

    def test_set_terminator_ignores_insertion_point(self, _b):
        f = _b.build_function("f", ())
        blk = _b.insert_block(f.region)
        term = _b.set_terminator(blk, "return", ())
        assert blk.ops == [term]
        assert _b.current_block is f.entry_block  # stack untouched
        assert f.entry_block.ops == []

    def test_set_terminator_with_operands(self, _b):
        f = _b.build_function("f", (_vt("float32", (2,)),))
        (x,) = f.entry_block.arguments
        term = _b.set_terminator(f.entry_block, "return", (x,))
        assert term.operands == (x,)
        assert f.output_types == (x.type,)
        assert len(x.uses) == 1
        assert x.uses[0].owner is term
        assert x.uses[0].operand_index == 0

    def test_set_terminator_rejects_non_terminator(self, _b):
        f = _b.build_function("f", ())
        with pytest.raises(VerificationError, match="not a terminator"):
            _b.set_terminator(f.entry_block, "add", ())

    def test_set_terminator_rejects_duplicate(self, _b):
        f = _b.build_function("f", ())
        _b.set_terminator(f.entry_block, "return", ())
        with pytest.raises(VerificationError, match="already has a terminator"):
            _b.set_terminator(f.entry_block, "return", ())
        assert len(f.entry_block.ops) == 1


# ---------------------------------------------------------------------------
# 15. id-counter isolation between modules
# ---------------------------------------------------------------------------

class TestIdCounters:
    def test_per_module_counters(self):
        b1 = ir.Builder()
        b1.build_module("a")
        f1 = b1.build_function("f", (_vt("float32", (2,)),))
        b2 = ir.Builder()
        b2.build_module("b")
        f2 = b2.build_function("f", (_vt("float32", (2,)),))
        v1 = b1.emit("negate", (f1.entry_block.arguments[0],))
        v2 = b2.emit("negate", (f2.entry_block.arguments[0],))
        # each module counts from 0 independently
        assert f1.entry_block.arguments[0].id == 0
        assert f2.entry_block.arguments[0].id == 0
        assert v1.owner.id == 0 and v2.owner.id == 0
        assert v1.id == 1 and v2.id == 1

    def test_counters_advance_within_module(self, _b):
        f = _b.build_function("f", (_vt("float32", (2,)), _vt("float32", (2,))))
        a, c = f.entry_block.arguments
        ops = [_b.emit(name, (a, c)).owner for name in ("add", "multiply", "subtract")]
        assert [op.id for op in ops] == [0, 1, 2]
        value_ids = [a.id, c.id] + [op.result.id for op in ops]
        assert value_ids == [0, 1, 2, 3, 4]

    def test_new_module_resets_counters(self):
        b = ir.Builder()
        b.build_module()
        f = b.build_function("f", ())
        b.set_terminator(f.entry_block, "return", ())
        b.build_module()  # a fresh module -> fresh counters
        b.build_function("f", ())
        v = b.emit("constant", (), attributes={"value": np.zeros(1, np.float32)})
        assert v.owner.id == 0
        assert v.id == 0
