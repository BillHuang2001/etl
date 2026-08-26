"""Shape-inference hook tests for EvoXIR (``etl.ir.inference``).

Encodes the exact semantics documented in ``etl/ir/inference.py`` for all 23
public hooks, plus the KNOWN current contracts recorded in
``etl/ir/CONTEXT.md`` ("Known Issues"):

* ``divide`` binds ``infer_elementwise_binary`` — int/int stays int;
* unary math ops (``sqrt``, ``exp``, ...) bind ``infer_elementwise_unary`` —
  dtype-preserving;
* ``cumsum`` binds ``infer_identity`` — bool stays bool;
* ``infer_pad`` accepts bare-int padding entries at the hook level (the
  Builder's attribute schema is stricter — that is tested elsewhere).

Conventions encoded throughout (module docstring of ``inference.py``):

* broadcasting = numpy rules + symbolic dims: ``1`` yields the other side;
  structurally equal dims pass; two unequal concrete ints raise
  ``ShapeError`` NOW; ``None`` dims are unchecked and yield ``None``;
  otherwise the result is ``DimExpr("max", a, b)``;
* element counts raise only on *definite* mismatch;
* sum folds are left-associative ``DimExpr("add", ...)`` chains with static
  ints folded separately;
* dtype promotion is exactly ``np.result_type`` (so int32 + float32 ->
  float64, per numpy);
* reduction dtypes follow numpy per ``reduce_op``.

Hooks are invoked directly with ``ValueType`` operands and attribute dicts
(the ``OpDef.shape_fn`` calling convention); a few ops are also exercised
through the ``Builder``, where ``verify`` enforces shape_fn agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from etl import ir
from etl.core import Dim, DimExpr, ShapeError

INF = ir.inference
VT = ir.ValueType

#: Shared symbolic batch dimension (equality is structural: name + size).
B = Dim("B")


def run(hook, operand_specs, attrs=None):
    """Invoke an inference hook with operands built from ``(dtype, shape)`` specs."""
    operands = tuple(VT(d, s) for d, s in operand_specs)
    return getattr(INF, hook)(operands, {} if attrs is None else attrs)


# ---------------------------------------------------------------------------
# 1. infer_elementwise_binary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_spec", "b_spec", "expected_shape", "expected_dtype"),
    [
        ((np.float32, (3, 1)), (np.float32, (1, 4)), (3, 4), np.float32),
        # rank promotion (numpy): the trailing dim must match — (2,) with
        # (2, 3) is an ERROR (see the error case below)
        ((np.float32, (3,)), (np.float32, (2, 3)), (2, 3), np.float32),
        # int/int stays int — this is the hook `divide` binds (Known Issues)
        ((np.int32, (2, 2)), (np.int32, (2, 2)), (2, 2), np.int32),
        # np.result_type promotion: int32 + float32 -> float64 (numpy rule)
        ((np.int32, (2, 2)), (np.float32, (2, 2)), (2, 2), np.float64),
        ((np.int16, (2, 2)), (np.float32, (2, 2)), (2, 2), np.float32),
        ((np.bool_, (2, 2)), (np.bool_, (2, 2)), (2, 2), np.bool_),
        # None dims are unchecked and yield None
        ((np.float32, (None, 4)), (np.float32, (1, 4)), (None, 4), np.float32),
        # symbolic: B right-aligns with 4 -> max(B, 4); the 1 yields B's side
        ((np.float32, (B,)), (np.float32, (1, 4)), (1, DimExpr("max", B, 4)), np.float32),
        ((np.float32, (B,)), (np.float32, (4,)), (DimExpr("max", B, 4),), np.float32),
        # structurally equal symbolic dims pass through unchanged
        ((np.float32, (B,)), (np.float32, (B,)), (B,), np.float32),
    ],
    ids=[
        "broadcast-both",
        "rank-promotion",
        "int-int",
        "int32-float32",
        "int16-float32",
        "bool-bool",
        "none-dims",
        "symbolic-right-aligned",
        "symbolic-max",
        "symbolic-equal",
    ],
)
def test_infer_elementwise_binary(a_spec, b_spec, expected_shape, expected_dtype):
    (result,) = run("infer_elementwise_binary", (a_spec, b_spec))
    assert result.dtype == np.dtype(expected_dtype)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("operand_specs", "match"),
    [
        (((np.float32, (3,)), (np.float32, (4,))), "cannot broadcast"),
        # numpy-correct: (2,) aligns with the LAST dim 3 of (2, 3) -> error
        (((np.float32, (2,)), (np.float32, (2, 3))), "cannot broadcast"),
        (((np.float32, (3, 4)),), "expected 2 operands"),
    ],
    ids=["mismatch", "mismatch-right-aligned", "wrong-arity"],
)
def test_infer_elementwise_binary_errors(operand_specs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_elementwise_binary", operand_specs)


# ---------------------------------------------------------------------------
# 2. infer_elementwise_unary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_spec", "expected_shape"),
    [
        ((np.float32, (2, 3)), (2, 3)),
        # CURRENT contract (Known Issues): dtype-preserving, int32 stays int32
        ((np.int32, (2, 3)), (2, 3)),
        ((np.float32, (B, 2)), (B, 2)),
    ],
    ids=["static", "int32-preserved", "symbolic"],
)
def test_infer_elementwise_unary(operand_spec, expected_shape):
    (result,) = run("infer_elementwise_unary", (operand_spec,))
    assert result.dtype == np.dtype(operand_spec[0])
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("operand_specs", "match"),
    [
        ((), "expected exactly one operand, got 0"),
        (
            ((np.float32, (2, 3)), (np.float32, (2, 3))),
            "expected exactly one operand, got 2",
        ),
    ],
    ids=["zero", "two"],
)
def test_infer_elementwise_unary_errors(operand_specs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_elementwise_unary", operand_specs)


# ---------------------------------------------------------------------------
# 3. infer_cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attr", "expected_dtype"),
    [
        (np.float64, np.dtype("float64")),  # dtype object accepted
        ("int8", np.dtype("int8")),  # name string accepted
    ],
    ids=["dtype-object", "name-string"],
)
def test_infer_cast(attr, expected_dtype):
    (result,) = run("infer_cast", ((np.float32, (B, 2)),), {"dtype": attr})
    assert result.dtype == expected_dtype
    assert result.shape == (B, 2)  # shape preserved, incl. symbolic dims


# ---------------------------------------------------------------------------
# 4. infer_compare
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "expected_shape"),
    [
        ((3, 1), (1, 4), (3, 4)),
        ((B, 4), (1, 4), (B, 4)),
    ],
    ids=["broadcast", "symbolic"],
)
def test_infer_compare(a_shape, b_shape, expected_shape):
    (result,) = run("infer_compare", ((np.float32, a_shape), (np.int32, b_shape)))
    assert result.dtype == np.dtype("bool")
    assert result.shape == expected_shape


def test_infer_compare_wrong_arity():
    with pytest.raises(ShapeError, match="expected 2 operands"):
        run("infer_compare", ((np.float32, (3, 4)),))


# ---------------------------------------------------------------------------
# 5. infer_select
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred_spec", "true_spec", "false_spec", "expected_shape", "expected_dtype"),
    [
        (
            (np.bool_, (B, 1)),
            (np.float32, (1, 4)),
            (np.float32, (B, 4)),
            (B, 4),
            np.float32,
        ),
        # branch dtype = np.result_type of the two branch operands
        ((np.bool_, (3,)), (np.int32, (3,)), (np.float32, (3,)), (3,), np.float64),
        ((np.bool_, (3,)), (np.int32, (3,)), (np.int32, (3,)), (3,), np.int32),
    ],
    ids=["symbolic-broadcast", "branch-promotion", "branch-int"],
)
def test_infer_select(pred_spec, true_spec, false_spec, expected_shape, expected_dtype):
    (result,) = run("infer_select", (pred_spec, true_spec, false_spec))
    assert result.dtype == np.dtype(expected_dtype)
    assert result.shape == expected_shape


def test_infer_select_wrong_arity():
    with pytest.raises(ShapeError, match="expected 3 operands"):
        run("infer_select", ((np.bool_, (3,)), (np.float32, (3,))))


# ---------------------------------------------------------------------------
# 6. infer_broadcast_to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "target", "expected"),
    [
        ((3, 1), (3, 4), (3, 4)),
        ((1,), (3, 4), (3, 4)),  # leading dims may be added (numpy broadcast_to)
        ((3,), (2, 3), (2, 3)),
        # symbolic mismatch (B vs B * 2) is deferred to runtime
        ((B,), (B * 2,), (DimExpr("mul", B, 2),)),
        ((None,), (3, 4), (3, 4)),  # None dims unchecked
    ],
    ids=["size1-expand", "leading", "leading-2", "symbolic-deferred", "none-dim"],
)
def test_infer_broadcast_to(operand_shape, target, expected):
    (result,) = run("infer_broadcast_to", ((np.float32, operand_shape),), {"shape": target})
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("operand_shape", "target", "match"),
    [
        ((3, 4), (3,), "target rank"),
        ((3,), (4,), "cannot expand"),
        # numpy-correct: (3,) right-aligns with the target's LAST dim 4 -> error
        ((3,), (3, 4), "cannot expand"),
    ],
    ids=["rank-too-small", "static-expand", "right-aligned-expand"],
)
def test_infer_broadcast_to_errors(operand_shape, target, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_broadcast_to", ((np.float32, operand_shape),), {"shape": target})


# ---------------------------------------------------------------------------
# 7. infer_reshape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "target", "expected"),
    [
        ((2, 3), (-1, 3), (2, 3)),
        ((2, 3), (6,), (6,)),
        # symbolic wildcard: (2 * B) / 4 as a DimExpr quotient
        ((B, 2), (-1, 4), (DimExpr("floordiv", DimExpr("mul", 2, B), 4), 4)),
        ((None,), (-1,), (None,)),  # dynamic count: wildcard stays dynamic
    ],
    ids=["wildcard-static", "plain-static", "wildcard-symbolic", "wildcard-none"],
)
def test_infer_reshape(operand_shape, target, expected):
    (result,) = run("infer_reshape", ((np.float32, operand_shape),), {"shape": target})
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected
    dim = result.shape[0]
    if isinstance(dim, DimExpr):
        assert dim.evaluate({"B": 8}) == 4  # (2 * 8) / 4


@pytest.mark.parametrize(
    ("operand_shape", "target", "match"),
    [
        ((2, 3), (5,), "element counts"),
        ((2, 3), (-1, -1), "at most one -1 wildcard"),
        ((2, 3), (-1, 0), "zero-size known product"),
        ((5, 3), (-1, 4), "not divisible"),
    ],
    ids=["count-mismatch", "two-wildcards", "wildcard-zero", "wildcard-nondivisible"],
)
def test_infer_reshape_errors(operand_shape, target, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_reshape", ((np.float32, operand_shape),), {"shape": target})


# ---------------------------------------------------------------------------
# 8. infer_transpose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "permutation", "expected"),
    [
        ((2, 3, 4), (2, 0, 1), (4, 2, 3)),
        ((2, 3, 4), None, (4, 3, 2)),  # None = full reversal (numpy)
        ((B, 2, 3), (2, 0, 1), (3, B, 2)),
    ],
    ids=["perm", "reverse", "symbolic"],
)
def test_infer_transpose(operand_shape, permutation, expected):
    attrs = {} if permutation is None else {"permutation": permutation}
    (result,) = run("infer_transpose", ((np.float32, operand_shape),), attrs)
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("permutation", "match"),
    [
        ((1, 0, 0), "not a permutation"),
        ((1, 0), "invalid permutation"),
    ],
    ids=["duplicate-axis", "wrong-length"],
)
def test_infer_transpose_errors(permutation, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_transpose", ((np.float32, (2, 3, 4)),), {"permutation": permutation})


# ---------------------------------------------------------------------------
# 9. infer_slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "attrs", "expected"),
    [
        # complete slice (stride 1 over the full range) preserves symbolic dims
        (
            (B, 4),
            {"start_indices": (0, 0), "limit_indices": (B, 4), "strides": (1, 1)},
            (B, 4),
        ),
        (
            (10,),
            {"start_indices": (2,), "limit_indices": (8,), "strides": (2,)},
            (3,),  # (8 - 2 + 2 - 1) // 2
        ),
        # symbolic dim with fully static limits: pure-int arithmetic
        ((B,), {"start_indices": (2,), "limit_indices": (8,), "strides": (2,)}, (3,)),
        # limits None at the hook level: operand shape is used as the limit
        ((5,), {"start_indices": (1,), "strides": (2,)}, (2,)),
    ],
    ids=["complete-symbolic", "static", "symbolic-dim-static-limits", "limits-none"],
)
def test_infer_slice(operand_shape, attrs, expected):
    (result,) = run("infer_slice", ((np.float32, operand_shape),), attrs)
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


def test_infer_slice_symbolic_limit_gives_floordiv():
    # symbolic limits go through DimExpr arithmetic: ((B - 1 + 2) - 1) // 2
    (result,) = run(
        "infer_slice",
        ((np.float32, (B,)),),
        {"start_indices": (1,), "limit_indices": (B,), "strides": (2,)},
    )
    dim = result.shape[0]
    assert isinstance(dim, DimExpr)
    assert dim.op == "floordiv"
    assert dim.evaluate({"B": 8}) == 4


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"start_indices": (0,), "limit_indices": (11,), "strides": (1,)}, "exceeds dim"),
        ({"start_indices": (-1,), "limit_indices": (10,), "strides": (1,)}, "negative start"),
        ({"start_indices": (0,), "limit_indices": (10,), "strides": (0,)}, "positive ints"),
    ],
    ids=["limit-too-big", "negative-start", "zero-stride"],
)
def test_infer_slice_errors(attrs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_slice", ((np.float32, (10,)),), attrs)


# ---------------------------------------------------------------------------
# 10. infer_gather
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tensor_shape", "indices_shape", "axes", "expected"),
    [
        ((5, 4), (3,), (0,), (3, 4)),
        ((5, 4), (2,), (1,), (5, 2)),
        # multiple axes: removed; indices.shape inserted at the smallest axis
        ((5, 4, 3), (2, 2), (0, 1), (2, 2, 3)),
        ((5, 4), (2,), (-1,), (5, 2)),  # negative axis normalized
    ],
    ids=["axis0", "axis1", "multi-axis", "negative-axis"],
)
def test_infer_gather(tensor_shape, indices_shape, axes, expected):
    (result,) = run(
        "infer_gather",
        ((np.int32, tensor_shape), (np.float32, indices_shape)),
        {"axes": axes},
    )
    assert result.dtype == np.dtype("int32")  # result dtype = tensor dtype
    assert result.shape == expected


@pytest.mark.parametrize(
    ("operand_specs", "attrs", "match"),
    [
        (((np.int32, (5, 4)), (np.float32, (2,))), {"axes": (5,)}, "out of range"),
        (((np.int32, (5, 4)),), {"axes": (0,)}, "expected 2 operands"),
    ],
    ids=["bad-axis", "wrong-arity"],
)
def test_infer_gather_errors(operand_specs, attrs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_gather", operand_specs, attrs)


# ---------------------------------------------------------------------------
# 11. infer_scatter
# ---------------------------------------------------------------------------


def test_infer_scatter_result_is_first_operand():
    (result,) = run(
        "infer_scatter",
        ((np.int32, (B, 4)), (np.float32, (2,)), (np.float32, (2, 4))),
    )
    assert result == VT(np.int32, (B, 4))


def test_infer_scatter_wrong_arity():
    with pytest.raises(ShapeError, match="expected 3 operands"):
        run("infer_scatter", ((np.int32, (3, 4)), (np.float32, (2,))))


# ---------------------------------------------------------------------------
# 12. infer_concatenate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_specs", "axis", "expected_shape", "expected_dtype"),
    [
        (((np.float32, (3, 2)), (np.float32, (4, 2))), 0, (7, 2), np.float32),
        (((np.float32, (2, 3)), (np.float32, (2, 4))), 1, (2, 7), np.float32),
        # symbolic axis dims: left-associative add chain
        (
            ((np.float32, (B, 2)), (np.float32, (B, 2))),
            0,
            (DimExpr("add", B, B), 2),
            np.float32,
        ),
        # None dims: unchecked, yield None
        (((np.float32, (3, None)), (np.float32, (4, None))), 0, (7, None), np.float32),
        # dtype = np.result_type of the operands
        (((np.int32, (3, 2)), (np.float32, (3, 2))), 0, (6, 2), np.float64),
        (((np.int32, (3, 2)), (np.int32, (3, 2))), 0, (6, 2), np.int32),
    ],
    ids=["axis0", "axis1", "symbolic", "none-dims", "promotion", "int-int"],
)
def test_infer_concatenate(operand_specs, axis, expected_shape, expected_dtype):
    (result,) = run("infer_concatenate", operand_specs, {"axis": axis})
    assert result.dtype == np.dtype(expected_dtype)
    assert result.shape == expected_shape


def test_infer_concatenate_symbolic_sum_fold():
    # Left-associative add chain over the symbolic terms; the static 3 folds
    # into the integer part of the sum: (B + B) + 3.
    (result,) = run(
        "infer_concatenate",
        ((np.float32, (B,)), (np.float32, (3,)), (np.float32, (B,))),
        {"axis": 0},
    )
    dim = result.shape[0]
    assert dim == DimExpr("add", DimExpr("add", B, B), 3)
    assert dim.evaluate({"B": 5}) == 13


@pytest.mark.parametrize(
    ("operand_specs", "axis", "match"),
    [
        (((np.float32, (3, 2)), (np.float32, (3, 3))), 0, "must agree"),
        (((np.float32, (3, 2)), (np.float32, (3,))), 0, "rank mismatch"),
        ((), 0, "at least one operand"),
    ],
    ids=["non-axis-mismatch", "rank-mismatch", "no-operands"],
)
def test_infer_concatenate_errors(operand_specs, axis, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_concatenate", operand_specs, {"axis": axis})


# ---------------------------------------------------------------------------
# 13. infer_pad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "config", "expected"),
    [
        ((3, 4), ((1, 2), (0, 0)), (6, 4)),
        ((B,), ((1, 1),), (DimExpr("add", B, 2),)),
        # bare-int entries are accepted by the hook itself (Known Issues:
        # the Builder schema is stricter — tested in test_builder.py)
        ((3, 4), (1, (0, 0)), (5, 4)),
    ],
    ids=["static", "symbolic", "bare-int-entries"],
)
def test_infer_pad(operand_shape, config, expected):
    (result,) = run("infer_pad", ((np.float32, operand_shape),), {"padding_config": config})
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("operand_shape", "config", "match"),
    [
        ((3,), ((-1, 0),), "negative padding"),
        ((3, 4), ((1, 2),), "must have 2 entries"),
        ((3,), ("x",), "invalid padding entry"),
    ],
    ids=["negative", "wrong-length", "invalid-entry"],
)
def test_infer_pad_errors(operand_shape, config, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_pad", ((np.float32, operand_shape),), {"padding_config": config})


# ---------------------------------------------------------------------------
# 14. infer_reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_spec", "attrs", "expected_shape", "expected_dtype"),
    [
        ((np.float32, (2, 3, 4)), {"axes": (0, 2)}, (3,), np.float32),
        (
            (np.float32, (2, 3, 4)),
            {"axes": (0, 2), "keepdims": True},
            (1, 3, 1),
            np.float32,
        ),
        ((np.float32, (2, 3, 4)), {"axes": ()}, (), np.float32),  # empty = ALL axes
        ((np.float32, (2, 3, 4)), {"axes": 1}, (2, 4), np.float32),  # bare int accepted
        ((np.int32, (2, 3)), {"axes": (0,)}, (3,), np.int64),  # sum: int -> int64
        ((np.bool_, (2, 3)), {"axes": (0,)}, (3,), np.int64),  # sum: bool -> int64
        ((np.uint8, (2, 3)), {"axes": (0,)}, (3,), np.uint64),  # sum: uint -> uint64
        ((np.float32, (2, 3)), {"axes": (0,)}, (3,), np.float32),  # sum: float keeps
        (
            (np.int32, (2, 3)),
            {"axes": (0,), "reduce_op": "prod"},
            (3,),
            np.int64,  # prod: int -> int64
        ),
        (
            (np.int32, (2, 3)),
            {"axes": (0,), "reduce_op": "mean"},
            (3,),
            np.float64,  # mean: int -> float64
        ),
        (
            (np.int32, (2, 3)),
            {"axes": (0,), "reduce_op": "max"},
            (3,),
            np.int32,  # max: preserves
        ),
        (
            (np.int32, (2, 3)),
            {"axes": (0,), "reduce_op": "min"},
            (3,),
            np.int32,  # min: preserves
        ),
        ((np.float32, (B, 2, 3)), {"axes": (1, 2)}, (B,), np.float32),  # symbolic kept
    ],
    ids=[
        "axes",
        "keepdims",
        "all-axes",
        "int-axis",
        "sum-int32",
        "sum-bool",
        "sum-uint8",
        "sum-float32",
        "prod-int32",
        "mean-int32",
        "max-int32",
        "min-int32",
        "symbolic",
    ],
)
def test_infer_reduction(operand_spec, attrs, expected_shape, expected_dtype):
    attrs = {"reduce_op": "sum", **attrs}
    (result,) = run("infer_reduction", (operand_spec,), attrs)
    assert result.dtype == np.dtype(expected_dtype)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("attrs", "exc", "match"),
    [
        ({"axes": (5,)}, ShapeError, "out of range"),
        # unknown reduce_op raises ValueError, NOT ShapeError
        ({"reduce_op": "var", "axes": (0,)}, ValueError, "unknown reduce_op"),
    ],
    ids=["bad-axis", "unknown-reduce-op"],
)
def test_infer_reduction_errors(attrs, exc, match):
    attrs = {"reduce_op": "sum", **attrs}
    with pytest.raises(exc, match=match):
        run("infer_reduction", ((np.float32, (2, 3)),), attrs)


# ---------------------------------------------------------------------------
# 15. infer_arg_reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "attrs", "expected_shape"),
    [
        ((2, 3), {"axis": None}, ()),  # numpy: flatten-reduce -> scalar
        ((2, 3), {"axis": None, "keepdims": True}, (1, 1)),
        ((2, 3, 4), {"axis": 1}, (2, 4)),
        ((2, 3, 4), {"axis": -1}, (2, 3)),  # negative axis normalized
    ],
    ids=["flatten", "flatten-keepdims", "axis1", "negative-axis"],
)
def test_infer_arg_reduction(operand_shape, attrs, expected_shape):
    (result,) = run("infer_arg_reduction", ((np.float32, operand_shape),), attrs)
    assert result.dtype == np.dtype("int64")
    assert result.shape == expected_shape


def test_infer_arg_reduction_bad_axis():
    with pytest.raises(ShapeError, match="out of range"):
        run("infer_arg_reduction", ((np.float32, (2, 3)),), {"axis": 5})


# ---------------------------------------------------------------------------
# 16. infer_identity
# ---------------------------------------------------------------------------


def test_infer_identity_mirrors_operands():
    operands = (VT(np.float32, (2, 3)), VT(np.bool_, (B,)))
    assert INF.infer_identity(operands, {}) == operands
    # bool stays bool — the current cumsum contract (Known Issues)


def test_infer_identity_empty():
    assert INF.infer_identity((), {}) == ()


def test_known_issues_current_opdef_wiring():
    # CURRENT contract (etl/ir/CONTEXT.md "Known Issues"): these ops bind the
    # generic hooks; the ../ops binding contract will later replace them with
    # dedicated hooks (true division, unary-math promotion, cumsum promotion).
    assert ir.opdef("divide").shape_fn is INF.infer_elementwise_binary
    assert ir.opdef("sqrt").shape_fn is INF.infer_elementwise_unary
    assert ir.opdef("cumsum").shape_fn is INF.infer_identity


# ---------------------------------------------------------------------------
# 17. infer_dot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "expected"),
    [
        ((2, 3), (3, 4), (2, 4)),
        ((3,), (3, 4), (4,)),  # vector·matrix
        ((2, 3), (3,), (2,)),  # matrix·vector
        ((3,), (3,), ()),  # dot product
        ((5, 2, 3), (1, 3, 4), (5, 2, 4)),  # batched matmul, batch broadcast
        ((2, 3), (1, 3, 4), (1, 2, 4)),  # batch broadcast to a rank-3 operand
        ((B, 2, 3), (3, 4), (B, 2, 4)),  # symbolic batch dim
    ],
    ids=["matmul", "vec-mat", "mat-vec", "dot-product", "batched", "batch-broadcast", "symbolic"],
)
def test_infer_dot_shapes(a_shape, b_shape, expected):
    (result,) = run("infer_dot", ((np.float32, a_shape), (np.float32, b_shape)))
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


def test_infer_dot_dtype_promotion():
    (result,) = run("infer_dot", ((np.int32, (2, 3)), (np.float32, (3, 4))))
    assert result.dtype == np.dtype("float64")  # np.result_type(int32, float32)


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "match"),
    [
        ((2, 3), (4, 5), "contracting"),
        ((), (3,), "rank >= 1"),
        # numpy matmul semantics: batch dims 5 vs 2 do not broadcast
        ((5, 2, 3), (2, 3, 4), "cannot broadcast"),
    ],
    ids=["contract-mismatch", "rank-0", "batch-mismatch"],
)
def test_infer_dot_errors(a_shape, b_shape, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_dot", ((np.float32, a_shape), (np.float32, b_shape)))


def test_infer_dot_wrong_arity():
    with pytest.raises(ShapeError, match="expected 2 operands"):
        run("infer_dot", ((np.float32, (2, 3)),))


# ---------------------------------------------------------------------------
# 18. infer_conv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "attrs", "expected"),
    [
        ((1, 3, 8, 8), (4, 3, 3, 3), {}, (1, 4, 6, 6)),  # VALID default
        ((1, 3, 8, 8), (4, 3, 3, 3), {"padding": "SAME"}, (1, 4, 8, 8)),  # ceil(8/1)
        ((1, 3, 8, 8), (4, 3, 3, 3), {"strides": (2, 2)}, (1, 4, 3, 3)),
        ((1, 3, 8, 8), (4, 3, 3, 3), {"padding": ((1, 1), (1, 1))}, (1, 4, 8, 8)),
        # effective sizes: eff_d = (8 - 1) * 2 + 1 = 15, eff_k = (2 - 1) * 2 + 1 = 3
        (
            (1, 3, 8, 8),
            (4, 3, 2, 2),
            {"input_dilation": 2, "kernel_dilation": 2},
            (1, 4, 13, 13),
        ),
        # batch groups: C_out = kernel C_out * batch_group_count
        ((4, 3, 8, 8), (5, 3, 3, 3), {"batch_group_count": 2}, (4, 10, 6, 6)),
        # feature groups: C_in / 2 == kernel C_in
        ((1, 6, 8, 8), (4, 3, 3, 3), {"feature_group_count": 2}, (1, 4, 6, 6)),
    ],
    ids=["valid", "same", "strides", "pads", "dilation", "batch-groups", "feature-groups"],
)
def test_infer_conv(x_shape, w_shape, attrs, expected):
    (result,) = run("infer_conv", ((np.float32, x_shape), (np.float32, w_shape)), attrs)
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


def test_infer_conv_symbolic_spatial():
    (result,) = run("infer_conv", ((np.float32, (1, 3, B, B)), (np.float32, (4, 3, 3, 3))))
    assert result.shape[:2] == (1, 4)
    dim = result.shape[2]
    assert isinstance(dim, DimExpr)
    assert dim.evaluate({"B": 8}) == 6  # (8 - 3) // 1 + 1


@pytest.mark.parametrize(
    ("x_shape", "w_shape", "attrs", "match"),
    [
        ((1, 3, 8, 8), (4, 3, 3, 3), {"feature_group_count": 2}, "not divisible"),
        ((1, 4, 8, 8), (4, 3, 3, 3), {"feature_group_count": 2}, "kernel in-channels"),
        ((1, 3, 8, 8), (4, 3, 3), {}, "ranks differ"),
        ((3, 8), (3, 3), {}, "rank must be >= 3"),
        ((1, 3, 4, 4), (4, 3, 7, 7), {}, "negative output dim"),
        ((1, 3, 8, 8), (4, 3, 3, 3), {"strides": (0, 1)}, "positive ints"),
        ((1, 3, 8, 8), (4, 3, 3, 3), {"padding": "FOO"}, "unknown padding mode"),
        ((1, 3, 8, 8), (4, 3, 3, 3), {"padding": -1}, "negative padding"),
        ((4, 3, 8, 8), (5, 3, 3, 3), {"batch_group_count": 3}, "not divisible"),
    ],
    ids=[
        "feature-groups-indivisible",
        "feature-groups-kernel-mismatch",
        "rank-mismatch",
        "rank-too-small",
        "negative-output",
        "zero-stride",
        "unknown-padding",
        "negative-padding",
        "batch-groups-indivisible",
    ],
)
def test_infer_conv_errors(x_shape, w_shape, attrs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_conv", ((np.float32, x_shape), (np.float32, w_shape)), attrs)


# ---------------------------------------------------------------------------
# 19. infer_solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_spec", "b_spec", "expected_shape", "expected_dtype"),
    [
        # integer inputs promote to float64 exactly like numpy linalg.solve
        ((np.int32, (2, 2)), (np.int32, (2, 1)), (2, 1), np.float64),
        ((np.float32, (2, 2)), (np.float32, (2, 1)), (2, 1), np.float32),
        ((np.int32, (2, 2, 2)), (np.int32, (2, 1)), (2, 2, 1), np.float64),  # batched a
        ((np.int32, (2, 2)), (np.float32, (2, 1)), (2, 1), np.float64),  # int a promotes
    ],
    ids=["int-int", "float-float", "batched", "int-float"],
)
def test_infer_solve(a_spec, b_spec, expected_shape, expected_dtype):
    (result,) = run("infer_solve", (a_spec, b_spec))
    assert result.dtype == np.dtype(expected_dtype)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("a_shape", "b_shape", "match"),
    [
        ((2, 3), (2, 1), "must be square"),
        ((2,), (2, 1), "rank >= 2"),
        ((2, 2), (), "rank >= 1"),
        ((2, 2), (3, 1), "contracting"),
    ],
    ids=["non-square", "a-rank-1", "b-rank-0", "contract-mismatch"],
)
def test_infer_solve_errors(a_shape, b_shape, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_solve", ((np.float32, a_shape), (np.float32, b_shape)))


# ---------------------------------------------------------------------------
# 20. infer_all_gather
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "group_size", "expected"),
    [
        ((4,), 2, (8,)),
        ((B,), 2, (DimExpr("mul", B, 2),)),
        ((4,), None, (None,)),  # world group: rank count unknown at trace time
    ],
    ids=["static", "symbolic", "world-group"],
)
def test_infer_all_gather(operand_shape, group_size, expected):
    (result,) = run(
        "infer_all_gather", ((np.float32, operand_shape),), {"group_size": group_size, "axis": 0}
    )
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"group_size": 0, "axis": 0}, "positive int or None"),
        ({"group_size": 2, "axis": 1}, "out of range"),
    ],
    ids=["zero-group", "bad-axis"],
)
def test_infer_all_gather_errors(attrs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_all_gather", ((np.float32, (4,)),), attrs)


# ---------------------------------------------------------------------------
# 21. infer_reduce_scatter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operand_shape", "group_size", "expected"),
    [
        ((4,), 2, (2,)),
        ((B,), 2, (DimExpr("floordiv", B, 2),)),
        ((4,), None, (None,)),  # world group: runtime-dynamic
        ((None,), 2, (None,)),  # dynamic dim stays dynamic
    ],
    ids=["static", "symbolic", "world-group", "none-dim"],
)
def test_infer_reduce_scatter(operand_shape, group_size, expected):
    (result,) = run(
        "infer_reduce_scatter",
        ((np.float32, operand_shape),),
        {"group_size": group_size, "axis": 0},
    )
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"group_size": 2, "axis": 0}, "not divisible"),
        ({"group_size": 0, "axis": 0}, "positive int or None"),
    ],
    ids=["indivisible", "zero-group"],
)
def test_infer_reduce_scatter_errors(attrs, match):
    operand_shape = (5,) if attrs["group_size"] == 2 else (4,)
    with pytest.raises(ShapeError, match=match):
        run("infer_reduce_scatter", ((np.float32, operand_shape),), attrs)


# ---------------------------------------------------------------------------
# 22. infer_all_to_all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group_size", "split_axis", "concat_axis", "expected"),
    [
        (2, 0, 1, (1, 8)),
        (2, 0, 0, (2, 4)),  # split == concat: shape unchanged
        (None, 0, 1, (None, None)),  # world group + different axes: both dynamic
        (None, 0, 0, (2, 4)),  # world group + equal axes: unchanged
    ],
    ids=["split-concat", "equal-axes", "world-group-diff", "world-group-equal"],
)
def test_infer_all_to_all(group_size, split_axis, concat_axis, expected):
    (result,) = run(
        "infer_all_to_all",
        ((np.float32, (2, 4)),),
        {"group_size": group_size, "split_axis": split_axis, "concat_axis": concat_axis},
    )
    assert result.dtype == np.dtype("float32")
    assert result.shape == expected


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"group_size": 2, "split_axis": 0, "concat_axis": 2}, "out of range"),
        ({"group_size": 0, "split_axis": 0, "concat_axis": 1}, "positive int or None"),
    ],
    ids=["bad-axis", "zero-group"],
)
def test_infer_all_to_all_errors(attrs, match):
    with pytest.raises(ShapeError, match=match):
        run("infer_all_to_all", ((np.float32, (2, 4)),), attrs)


# ---------------------------------------------------------------------------
# 23. infer_scalar_int64
# ---------------------------------------------------------------------------


def test_infer_scalar_int64():
    for operands in (
        (),
        (VT(np.float32, (2, 3)), VT(np.int32, (B,))),
    ):
        assert INF.infer_scalar_int64(operands, {}) == (VT(np.int64, ()),)
        assert INF.infer_scalar_int64(operands, {"whatever": 1}) == (VT(np.int64, ()),)


# ---------------------------------------------------------------------------
# 24. Through-Builder integration (shape_fn agreement under verify)
# ---------------------------------------------------------------------------


def test_builder_result_types_match_hooks_and_verify():
    builder = ir.Builder()
    module = builder.build_module()
    function = builder.build_function(
        "main", (VT(np.float32, (2, 3)), VT(np.float32, (1, 3)))
    )
    x, y = function.entry_block.arguments

    z = builder.emit("add", (x, y))
    assert z.type == INF.infer_elementwise_binary((x.type, y.type), {})[0]
    assert z.type == VT(np.float32, (2, 3))

    flat = builder.emit("reshape", (z,), attributes={"shape": (-1,)})
    assert flat.type.shape == (6,)

    total = builder.emit(
        "reduce_sum", (flat,), attributes={"axes": (0,), "reduce_op": "sum"}
    )
    assert total.type == VT(np.float32, ())  # sum: float32 keeps its dtype

    builder.set_terminator(function.entry_block, "return", (total,))
    ir.verify(module)  # recomputes every shape_fn; must agree with recorded types


# ---------------------------------------------------------------------------
# 25. ValueType.__str__
# ---------------------------------------------------------------------------


def test_valuetype_str():
    assert str(VT(np.float32, (B, 4))) == "tensor<Bx4xf32>"
    assert str(VT(np.float32, (3, None))) == "tensor<3x?xf32>"  # None dim -> ?
    assert str(VT(np.float32, ())) == "tensor<f32>"
