"""Golden-output tests for the EvoXIR IR pretty printer (`etl.ir.pretty_print`).

Every expected string here was produced by running `pretty_print` on the
corresponding hand-built module and eyeballed against the documented format
rules in `etl/ir/printer.py`:

* functions/ops/block-args print in program order; attribute keys print
  sorted; op results are renumbered ``%0, %1, ...`` per function and only ops
  *with* results consume numbers; block arguments print as ``%argN`` (argument
  index within their block);
* each op line is ``%r = name(operands) [attributes {...}] : type(s)
  loc(...)`` — ``: type`` omitted for zero-result ops, ``attributes {...}``
  only for non-empty attribute dicts;
* function headers with two or more block arguments wrap one argument per
  line, continued aligned right after ``(``; result types after ``->`` are
  omitted for zero outputs;
* nested regions render inline under the op, indented two spaces deeper, with
  nested blocks labeled ``^bbN`` (label counter shared per function).
"""

import numpy as np
import pytest

from etl import ir
from etl.core import Dim, DimExpr

F32 = np.dtype("float32")


def _vt(shape, dtype=F32):
    return ir.ValueType(dtype, shape)


def test_traced_style_module_full_output():
    """A small traced-style module: args, add, constant payload, multiply.

    Covers result renumbering (%0/%1/%2 in program order) and the constant
    op's attribute spelling ``ndarray<f32[]>`` (0-d float32 payload).
    """
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((Dim("B"), 4)), _vt((4,))))
    arg0, arg1 = f.entry_block.arguments
    s = b.emit("add", (arg0, arg1))
    c = b.create(
        "constant", (), attributes={"value": np.array(2.0, dtype=np.float32)}
    ).result
    m = b.emit("multiply", (s, c))
    b.set_terminator(f.entry_block, "return", (m,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<Bx4xf32>,
             %arg1: tensor<4xf32>) -> tensor<Bx4xf32> {
    %0 = etl.add(%arg0, %arg1) : tensor<Bx4xf32>
    %1 = etl.constant() attributes {value = ndarray<f32[]>} : tensor<f32>
    %2 = etl.multiply(%0, %1) : tensor<Bx4xf32>
    etl.return(%2)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_attribute_rendering():
    """Attribute spellings: sorted keys, lists, bools, strings, None -> ?,
    dtype names JSON-quoted."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((Dim("B"), 4)),))
    (arg0,) = f.entry_block.arguments
    b.emit(
        "reduce_sum",
        (arg0,),
        attributes={"axes": [1], "keepdims": False, "reduce_op": "sum"},
    )
    b.emit("transpose", (arg0,), attributes={"permutation": None})
    b.emit(
        "slice",
        (arg0,),
        attributes={"start_indices": [0, 1], "limit_indices": [8, 4], "strides": [1, 2]},
    )
    cast = b.emit("cast", (arg0,), attributes={"dtype": "float64"})
    b.set_terminator(f.entry_block, "return", (cast,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<Bx4xf32>) -> tensor<Bx4xf64> {
    %0 = etl.reduce_sum(%arg0) attributes {axes = [1], keepdims = False, reduce_op = "sum"} : tensor<Bxf32>
    %1 = etl.transpose(%arg0) attributes {permutation = ?} : tensor<4xBxf32>
    %2 = etl.slice(%arg0) attributes {limit_indices = [8, 4], start_indices = [0, 1], strides = [1, 2]} : tensor<8x2xf32>
    %3 = etl.cast(%arg0) attributes {dtype = "float64"} : tensor<Bx4xf64>
    etl.return(%3)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_location_rendering():
    """Op locations print as ``loc("file":line:col)``; col None is omitted."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2,)),))
    (arg0,) = f.entry_block.arguments
    a = b.emit("add", (arg0, arg0), location=ir.Location("model.py", 12, 8))
    n = b.emit("negate", (a,), location=ir.Location("model.py", 13, None))
    b.set_terminator(f.entry_block, "return", (n,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = etl.add(%arg0, %arg0) : tensor<2xf32> loc("model.py":12:8)
    %1 = etl.negate(%0) : tensor<2xf32> loc("model.py":13)
    etl.return(%1)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_multi_argument_function_header_wrapping():
    """Three block arguments render one per line, aligned right after '('."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2, 3)), _vt((3,)), _vt((3,))))
    arg0, arg1, arg2 = f.entry_block.arguments
    a = b.emit("add", (arg0, arg1))
    m = b.emit("multiply", (a, arg2))
    b.set_terminator(f.entry_block, "return", (m,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<2x3xf32>,
             %arg1: tensor<3xf32>,
             %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
    %0 = etl.add(%arg0, %arg1) : tensor<2x3xf32>
    %1 = etl.multiply(%0, %arg2) : tensor<2x3xf32>
    etl.return(%1)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_zero_output_function():
    """No result types in the header; a bare ``etl.return()`` terminator."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2,)),))
    (arg0,) = f.entry_block.arguments
    b.emit("negate", (arg0,))
    b.set_terminator(f.entry_block, "return", ())

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<2xf32>) {
    %0 = etl.negate(%arg0) : tensor<2xf32>
    etl.return()
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_nested_regions_if_op():
    """An ``if`` op with true/false regions: inline bodies, ^bbN labels
    (the label counter is shared per function, so the second region's block
    is ^bb1), ``} {`` between regions, and the closing ``}`` at op indent."""
    b = ir.Builder()
    module = b.build_module("main")
    vt_x = _vt((2,))
    f = b.build_function("main", (_vt((), dtype=np.dtype("bool")), vt_x))
    pred, x = f.entry_block.arguments
    r_true = b.build_region((vt_x,))
    b.set_terminator(r_true.entry, "return", (r_true.entry.arguments[0],))
    r_false = b.build_region((vt_x,))
    b.set_terminator(r_false.entry, "return", (r_false.entry.arguments[0],))
    op = b.create("if", (pred, x), regions=(r_true, r_false))
    b.set_terminator(f.entry_block, "return", (op.result,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<i1>,
             %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = etl.if(%arg0, %arg1) : tensor<2xf32> {
      ^bb0(%arg0: tensor<2xf32>):
        etl.return(%arg0)
    } {
      ^bb1(%arg0: tensor<2xf32>):
        etl.return(%arg0)
    }
    etl.return(%0)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_op_order_and_zero_result_ops_consume_no_number():
    """Ops print in program (emission) order; the zero-result ``return``
    terminator consumes no result number."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2,)), _vt((2,))))
    arg0, arg1 = f.entry_block.arguments
    n = b.emit("negate", (arg0,))
    a = b.emit("add", (arg0, arg1))
    m = b.emit("multiply", (n, a))
    b.set_terminator(f.entry_block, "return", (m,))

    expected = """\
module @"main" version 1 {
  func @main(%arg0: tensor<2xf32>,
             %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = etl.negate(%arg0) : tensor<2xf32>
    %1 = etl.add(%arg0, %arg1) : tensor<2xf32>
    %2 = etl.multiply(%0, %1) : tensor<2xf32>
    etl.return(%2)
  }
}
"""
    assert ir.pretty_print(module) == expected


def test_symbolic_dim_spellings():
    """ValueType dimensions: Dim names, DimExpr infix forms, parenthesized
    sub-expressions, min/max call forms, and ``?`` for None (dynamic) dims."""
    B = Dim("B")
    shape = (
        B * 2,  # mul
        B + 1,  # add
        B // 2,  # floordiv
        B % 2,  # mod
        B - 1,  # sub
        B.max(4),  # max
        DimExpr("min", 4, B),  # min (int left operand)
        (B * 2) + 1,  # parenthesized sub-expression
        None,  # runtime-dynamic
    )
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt(shape),))
    (arg0,) = f.entry_block.arguments
    r = b.emit("stop_gradient", (arg0,))
    b.set_terminator(f.entry_block, "return", (r,))

    tensor = "tensor<B * 2xB + 1xB // 2xB % 2xB - 1xmax(B, 4)xmin(4, B)x(B * 2) + 1x?xf32>"
    expected = f"""\
module @"main" version 1 {{
  func @main(%arg0: {tensor}) -> {tensor} {{
    %0 = etl.stop_gradient(%arg0) : {tensor}
    etl.return(%0)
  }}
}}
"""
    assert ir.pretty_print(module) == expected


def test_unregistered_op_name_raises_keyerror():
    """An op whose name is not in the registry fails printing loudly."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2,)),))
    (arg0,) = f.entry_block.arguments
    r = b.emit("add", (arg0, arg0))
    b.set_terminator(f.entry_block, "return", (r,))

    f.entry_block.ops[0].name = "nope"
    with pytest.raises(KeyError, match="nope"):
        ir.pretty_print(module)


def test_missing_terminator_raises_valueerror():
    """A function without a ``return`` terminator cannot be printed."""
    b = ir.Builder()
    module = b.build_module("main")
    f = b.build_function("main", (_vt((2,)),))
    (arg0,) = f.entry_block.arguments
    b.emit("add", (arg0, arg0))

    with pytest.raises(ValueError, match="has no terminator"):
        ir.pretty_print(module)


def test_determinism():
    """Repeated printing and independently rebuilt modules agree exactly."""

    def build():
        b = ir.Builder()
        module = b.build_module("main")
        f = b.build_function("main", (_vt((2,)), _vt((2,))))
        arg0, arg1 = f.entry_block.arguments
        m = b.emit("multiply", (arg0, arg1))
        b.set_terminator(f.entry_block, "return", (m,))
        return module

    m1, m2 = build(), build()
    out1, out2 = ir.pretty_print(m1), ir.pretty_print(m2)
    assert out1 == out2
    assert ir.pretty_print(m1) == out1
    assert ir.pretty_print(m2) == out2
