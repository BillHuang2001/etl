"""pytest suite for etl elementwise + comparison ops.

Asserts the contracts of ``etl/ops/CONTEXT.md`` (binding) for the public
elementwise/comparison functions: broadcast shape rules (incl. symbolic
``Dim``/``DimExpr``), dtype promotion (numpy ``result_type`` for
tensor⊕tensor, NEP-50 weak scalar promotion for Python scalars), op
construction into the active trace builder (names, arities, effects,
locations), error semantics, and numerical agreement with numpy references
computed in the same promoted dtype.

``select`` is NOT covered here (it belongs to test_structure.py).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import etl
from tests.ops.conftest import ops_of, run_numpy, trace_fn

# ---------------------------------------------------------------------------
# op tables (public etl.* names; the IR registry stores the SAME plain names
# with category "elementwise"/"comparison" — verified via ops_of)
# ---------------------------------------------------------------------------

BINARY_ARITH = [
    ("add", np.add),
    ("subtract", np.subtract),
    ("multiply", np.multiply),
    ("power", np.power),
    ("remainder", np.remainder),
    ("maximum", np.maximum),
    ("minimum", np.minimum),
]

COMPARISONS = [
    ("equal", np.equal),
    ("not_equal", np.not_equal),
    ("less", np.less),
    ("less_equal", np.less_equal),
    ("greater", np.greater),
    ("greater_equal", np.greater_equal),
]

BITWISE = [
    ("bitwise_and", np.bitwise_and),
    ("bitwise_or", np.bitwise_or),
    ("bitwise_xor", np.bitwise_xor),
]

LOGICAL = [
    ("logical_and", np.logical_and),
    ("logical_or", np.logical_or),
]

UNARY_PRESERVING = ["abs", "negate", "square", "sign"]

UNARY_MATH = [
    "sqrt", "exp", "log", "log1p", "sin", "cos", "tan", "tanh",
    "sigmoid", "relu", "gelu", "erf",
]

ALL_BINARY = BINARY_ARITH + COMPARISONS + BITWISE + LOGICAL

#: dtype used in the shape/construction tests per op family (logical ops
#: require bool operands, bitwise ops require integer/bool).
def _binary_spec_dtype(op_name):
    if op_name in ("logical_and", "logical_or"):
        return etl.bool_
    if op_name.startswith("bitwise"):
        return etl.int32
    return etl.float32


def _trace_and_capture(fn, *specs):
    """Trace ``fn`` and return ``(graph, returned SymbolicTensor)``."""
    box = {}

    def wrapped(*args):
        box["out"] = fn(*args)
        return box["out"]

    graph = trace_fn(wrapped, *specs)
    return graph, box["out"]


def _promoted_ref(np_fn, a, b, out_dtype):
    """numpy reference for a binary op with BOTH operands in the result dtype."""
    return np_fn(np.asarray(a, dtype=out_dtype), np.asarray(b, dtype=out_dtype))


def _erf_vec(x):
    return np.frompyfunc(math.erf, 1, 1)(x).astype(x.dtype)


# ---------------------------------------------------------------------------
# 1. broadcasting shapes (int dims, rank mixing, 1-broadcast)
# ---------------------------------------------------------------------------

BROADCAST_CASES = [
    ((2, 3), (2, 3), (2, 3), "same-shape"),
    ((2, 1), (1, 3), (2, 3), "pairwise-1"),
    ((2, 3), (), (2, 3), "scalar-rhs"),
    ((), (3,), (3,), "scalar-lhs"),
    ((2, 3), (3,), (2, 3), "rank-mix"),
    ((2, 1, 3), (4, 3), (2, 4, 3), "rank-3"),
    ((), (), (), "two-0d-tensors"),
]


@pytest.mark.parametrize("op_name", [name for name, _ in ALL_BINARY])
@pytest.mark.parametrize(
    "shape_a,shape_b,expected", [c[:3] for c in BROADCAST_CASES],
    ids=[c[3] for c in BROADCAST_CASES],
)
def test_binary_broadcast_shapes(op_name, shape_a, shape_b, expected):
    fn = getattr(etl, op_name)
    dtype = _binary_spec_dtype(op_name)

    def traced(x, y):
        return fn(x, y)

    graph, out = _trace_and_capture(
        traced,
        etl.TensorSpec(shape_a, dtype),
        etl.TensorSpec(shape_b, dtype),
    )
    # SymbolicTensor side: the result shape is the numpy broadcast shape.
    assert out.shape == expected
    # IR side: the result Value type carries the same shape and dtype.
    ops = ops_of(graph, op_name)
    assert len(ops) == 1
    result_type = ops[0].results[0].type
    assert result_type.shape == expected
    assert result_type.dtype == out.dtype


@pytest.mark.parametrize("op_name", [name for name, _ in ALL_BINARY])
def test_binary_static_broadcast_error_at_trace_time(op_name):
    """Unequal static int dims fail at TRACE time with ShapeError (never
    deferred to run time)."""
    fn = getattr(etl, op_name)
    dtype = _binary_spec_dtype(op_name)

    def traced(x, y):
        return fn(x, y)

    with pytest.raises(etl.ShapeError, match="cannot broadcast incompatible dims 2 and 4"):
        trace_fn(
            traced,
            etl.TensorSpec((2, 3), dtype),
            etl.TensorSpec((4, 3), dtype),
        )


# ---------------------------------------------------------------------------
# 2. symbolic broadcasting (Dim / DimExpr rules)
# ---------------------------------------------------------------------------

_SYMBOLIC_CASES = [
    # (shape_a, shape_b, expected) — symbolic dims unify by name, conflicts
    # defer as DimExpr.max, 1 still broadcasts.
    ((etl.dim("n"), 3), (5, 1), (etl.DimExpr("max", etl.dim("n"), 5), 3), "dim-vs-int"),
    ((etl.dim("n"), 3), (etl.dim("n"), 3), (etl.dim("n"), 3), "same-dim-passthrough"),
    ((1, 3), (etl.dim("n"), 1), (etl.dim("n"), 3), "one-broadcasts"),
    ((etl.dim("n"), etl.dim("m")), (etl.dim("m"), 2),
     (etl.DimExpr("max", etl.dim("n"), etl.dim("m")), etl.DimExpr("max", etl.dim("m"), 2)),
     "two-symbolic"),
]


@pytest.mark.parametrize("op_name", ["add", "equal", "bitwise_and", "logical_and"])
@pytest.mark.parametrize(
    "shape_a,shape_b,expected", [c[:3] for c in _SYMBOLIC_CASES],
    ids=[c[3] for c in _SYMBOLIC_CASES],
)
def test_binary_symbolic_broadcast(op_name, shape_a, shape_b, expected):
    fn = getattr(etl, op_name)
    dtype = _binary_spec_dtype(op_name)

    def traced(x, y):
        return fn(x, y)

    graph, out = _trace_and_capture(
        traced,
        etl.TensorSpec(shape_a, dtype),
        etl.TensorSpec(shape_b, dtype),
    )
    assert out.shape == expected
    op = ops_of(graph, op_name)[0]
    assert op.results[0].type.shape == expected


@pytest.mark.parametrize("unary_name", UNARY_PRESERVING + UNARY_MATH + ["logical_not"])
def test_unary_preserves_symbolic_shape(unary_name):
    fn = getattr(etl, unary_name)
    n = etl.dim("n")
    dtype = etl.bool_ if unary_name == "logical_not" else etl.float32

    def traced(x):
        return fn(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((n, 3), dtype))
    assert out.shape == (n, 3)
    assert ops_of(graph, unary_name)[0].results[0].type.shape == (n, 3)


def test_unary_of_python_scalar_is_0d():
    graph, out = _trace_and_capture(lambda: etl.sqrt(4.0))
    assert out.shape == ()
    assert out.dtype == np.dtype("float64")


def test_symbolic_runtime_binding_happy_path():
    """A symbolic-dim graph runs once dims are bound from concrete inputs."""
    n = etl.dim("n")

    def fn(x, y):
        return etl.add(x, y)

    exe = etl.build(
        fn,
        etl.TensorSpec((n, 3), etl.float32),
        etl.TensorSpec((5, 1), etl.float32),
    )
    out = etl.run(exe, np.ones((5, 3), np.float32), np.full((5, 1), 2.0, np.float32))
    assert out.numpy().shape == (5, 3)
    assert np.all(out.numpy() == 3.0)


def test_runtime_broadcast_conflict_raises_shape_error():
    n = etl.dim("n")

    def fn(x, y):
        return etl.add(x, y)

    exe = etl.build(
        fn,
        etl.TensorSpec((n, 3), etl.float32),
        etl.TensorSpec((5, 1), etl.float32),
    )
    with pytest.raises(etl.ShapeError):
        etl.run(exe, np.ones((7, 3), np.float32), np.full((5, 1), 2.0, np.float32))


# ---------------------------------------------------------------------------
# 3. tensor ⊕ tensor dtype promotion (exactly numpy result_type)
# ---------------------------------------------------------------------------

DTYPE_PAIRS = [
    (etl.int8, etl.uint8, np.dtype("int16"), "int8+uint8->int16"),
    (etl.float32, etl.int64, np.dtype("float64"), "float32+int64->float64"),
    (etl.float16, etl.float16, np.dtype("float16"), "float16+float16->float16"),
    (etl.int32, etl.int32, np.dtype("int32"), "int32+int32->int32"),
    (etl.bool_, etl.bool_, np.dtype("bool"), "bool+bool->bool"),
    (etl.int32, etl.bool_, np.dtype("int32"), "int32+bool->int32"),
    (etl.uint8, etl.uint16, np.dtype("uint16"), "uint8+uint16->uint16"),
    (etl.int64, etl.uint64, np.dtype("float64"), "int64+uint64->float64"),
    (etl.int8, etl.float16, np.dtype("float16"), "int8+float16->float16"),
    (etl.complex64, etl.float32, np.dtype("complex64"), "complex64+float32->complex64"),
    (etl.float16, etl.float64, np.dtype("float64"), "float16+float64->float64"),
]


@pytest.mark.parametrize("op_name", ["add", "multiply", "maximum", "remainder", "power"])
@pytest.mark.parametrize(
    "dt_a,dt_b,expected", [p[:3] for p in DTYPE_PAIRS], ids=[p[3] for p in DTYPE_PAIRS]
)
def test_binary_dtype_promotion_tensor_tensor(op_name, dt_a, dt_b, expected):
    fn = getattr(etl, op_name)

    def traced(x, y):
        return fn(x, y)

    graph, out = _trace_and_capture(
        traced, etl.TensorSpec((2,), dt_a), etl.TensorSpec((2,), dt_b)
    )
    assert out.dtype == expected
    op = ops_of(graph, op_name)[0]
    assert op.results[0].type.dtype == expected
    # cross-check: the expected dtype is exactly numpy's result_type.
    assert expected == np.result_type(np.dtype(dt_a), np.dtype(dt_b))


def test_divide_true_division_dtypes():
    """divide is true division: promoted-integral operands yield float64."""
    cases = [
        (etl.int64, etl.int64, np.dtype("float64"), True),
        (etl.int8, etl.uint8, np.dtype("float64"), True),
        (etl.int16, etl.bool_, np.dtype("float64"), True),
        (etl.float16, etl.int8, np.dtype("float16"), False),
        (etl.int64, etl.float32, np.dtype("float64"), False),
        (etl.float32, etl.int64, np.dtype("float64"), False),
        (etl.complex64, etl.int64, np.dtype("complex128"), False),
        (etl.complex64, etl.float32, np.dtype("complex64"), False),
        (etl.float32, etl.float16, np.dtype("float32"), False),
    ]
    for dt_a, dt_b, expected, integral in cases:
        def traced(x, y):
            return etl.divide(x, y)

        graph, out = _trace_and_capture(
            traced, etl.TensorSpec((2,), dt_a), etl.TensorSpec((2,), dt_b)
        )
        assert out.dtype == expected
        div = ops_of(graph, "divide")
        assert len(div) == 1
        assert div[0].results[0].type.dtype == expected
        if integral:
            # transparent composition: both operands pre-cast to float64
            casts = ops_of(graph, "cast")
            assert len(casts) == 2
            assert all(c.results[0].type.dtype == np.dtype("float64") for c in casts)
        else:
            assert ops_of(graph, "cast") == []


@pytest.mark.parametrize(
    "op_name", [name for name, _ in COMPARISONS]
)
def test_comparison_result_is_bool(op_name):
    fn = getattr(etl, op_name)

    def traced(x, y):
        return fn(x, y)

    graph, out = _trace_and_capture(
        traced,
        etl.TensorSpec((2, 3), etl.float32),
        etl.TensorSpec((1, 3), etl.int32),
    )
    assert out.dtype == np.dtype("bool")
    op = ops_of(graph, op_name)[0]
    assert op.results[0].type.dtype == np.dtype("bool")
    assert op.results[0].type.shape == (2, 3)


# ---------------------------------------------------------------------------
# 4. NEP-50 weak Python-scalar promotion (verified behavior)
# ---------------------------------------------------------------------------

WEAK_SCALAR_CASES = [
    # int scalars weak toward float/complex tensors when exactly representable
    (etl.float32, 3, np.dtype("float32"), "int+float32->float32"),
    (etl.float16, 3, np.dtype("float16"), "int+float16->float16"),
    (etl.float64, 3, np.dtype("float64"), "int+float64->float64"),
    (etl.complex64, 3, np.dtype("complex64"), "int+complex64->complex64"),
    (etl.complex128, 3, np.dtype("complex128"), "int+complex128->complex128"),
    # ... else float64 / complex128 (mantissa bits 11/24/53)
    (etl.float32, 2 ** 24, np.dtype("float64"), "huge-int+float32->float64"),
    (etl.float16, 4096, np.dtype("float64"), "huge-int+float16->float64"),
    (etl.complex64, 2 ** 24, np.dtype("complex128"), "huge-int+complex64->complex128"),
    # int/bool scalars vs int/bool tensors: plain result_type (scalar = int64)
    (etl.int32, 3, np.dtype("int64"), "int+int32->int64"),
    (etl.int32, True, np.dtype("int32"), "bool+int32->int32"),
    (etl.uint8, 3, np.dtype("int64"), "int+uint8->int64"),
    (etl.bool_, 3, np.dtype("int64"), "int+bool->int64"),
    (etl.bool_, True, np.dtype("bool"), "bool+bool->bool"),
    # float scalars weak toward ALL float/complex tensors
    (etl.float32, 2.5, np.dtype("float32"), "float+float32->float32"),
    (etl.float16, 2.5, np.dtype("float16"), "float+float16->float16"),
    (etl.float64, 2.5, np.dtype("float64"), "float+float64->float64"),
    (etl.complex64, 2.5, np.dtype("complex64"), "float+complex64->complex64"),
    (etl.complex128, 2.5, np.dtype("complex128"), "float+complex128->complex128"),
    (etl.float32, True, np.dtype("float32"), "bool+float32->float32"),
    # float scalars vs int tensors promote via float64
    (etl.int32, 2.5, np.dtype("float64"), "float+int32->float64"),
    # complex scalars weak toward complex64/128 ONLY
    (etl.float32, 1j, np.dtype("complex64"), "complex+float32->complex64"),
    (etl.float16, 1j, np.dtype("complex64"), "complex+float16->complex64"),
    (etl.float64, 1j, np.dtype("complex128"), "complex+float64->complex128"),
    (etl.complex64, 1j, np.dtype("complex64"), "complex+complex64->complex64"),
    (etl.complex128, 1j, np.dtype("complex128"), "complex+complex128->complex128"),
    # complex scalars vs int tensors promote via complex128
    (etl.int32, 1j, np.dtype("complex128"), "complex+int32->complex128"),
    (etl.uint8, 1j, np.dtype("complex128"), "complex+uint8->complex128"),
]


@pytest.mark.parametrize(
    "tensor_dtype,scalar,expected", [c[:3] for c in WEAK_SCALAR_CASES],
    ids=[c[3] for c in WEAK_SCALAR_CASES],
)
def test_weak_scalar_promotion(tensor_dtype, scalar, expected):
    def traced(x):
        return etl.add(x, scalar)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), tensor_dtype))
    assert out.dtype == expected
    # the scalar became a 0-d Constant pre-promoted to the weak dtype
    consts = ops_of(graph, "constant")
    assert len(consts) == 1
    const = consts[0]
    assert const.results[0].type.dtype == expected
    assert const.results[0].type.shape == ()
    payload = const.attributes["value"]
    assert payload.shape == ()
    assert payload.dtype == expected
    # IR result dtype agrees
    assert ops_of(graph, "add")[0].results[0].type.dtype == expected
    # numerical: identical to numpy computed in the SAME promoted dtype
    a = np.array([1, 2, 3], dtype=tensor_dtype)
    got = run_numpy(traced, a)
    ref = np.add(np.asarray(a, dtype=expected), np.asarray(scalar, dtype=expected))
    assert got.dtype == expected
    if expected.kind == "b":
        assert np.array_equal(got, ref)
    else:
        assert np.allclose(got, ref)


def test_scalar_promotion_builds_constant_op():
    def traced(x):
        return etl.add(x, 3)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((2, 3), etl.float32))
    ops = ops_of(graph)
    const = ops_of(graph, "constant")[0]
    add = ops_of(graph, "add")[0]
    assert out.shape == (2, 3)
    assert out.dtype == np.dtype("float32")
    # constant: no operands, 0-d float32 payload, pure
    assert const.operands == ()
    assert const.results[0].type.dtype == np.dtype("float32")
    assert const.results[0].type.shape == ()
    assert const.effect == "pure"
    # add consumes (block arg, constant result)
    assert len(add.operands) == 2
    assert add.operands[0] is graph.module.functions[0].region.blocks[0].arguments[0]
    assert add.operands[1] is const.results[0]
    # the trailing op is the return terminator
    assert ops[-1].name == "return"
    assert ops[-1].is_terminator


def test_scalar_on_left_promotes_identically():
    def traced(x):
        return etl.add(3, x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((2,), etl.float32))
    add = ops_of(graph, "add")[0]
    assert add.operands[0] is ops_of(graph, "constant")[0].results[0]
    assert add.operands[1] is graph.module.functions[0].region.blocks[0].arguments[0]
    assert out.dtype == np.dtype("float32")


# ---------------------------------------------------------------------------
# 5. unary dtype rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unary_name", UNARY_PRESERVING)
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (etl.int8, etl.int8), (etl.uint8, etl.uint8),
        (etl.int32, etl.int32), (etl.bool_, etl.bool_),
        (etl.float16, etl.float16), (etl.float32, etl.float32),
    ],
    ids=str,
)
def test_unary_preserving_dtypes(unary_name, in_dtype, out_dtype):
    fn = getattr(etl, unary_name)

    def traced(x):
        return fn(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), in_dtype))
    assert out.dtype == np.dtype(out_dtype)
    assert ops_of(graph, unary_name)[0].results[0].type.dtype == np.dtype(out_dtype)


@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (etl.complex64, etl.float32),
        (etl.complex128, etl.float64),
    ],
    ids=str,
)
def test_abs_complex_gives_real_magnitude_dtype(in_dtype, out_dtype):
    """numpy abs of a complex value is the REAL magnitude (via a transparent
    post-cast on the abs op)."""
    def traced(x):
        return etl.abs(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), in_dtype))
    assert out.dtype == np.dtype(out_dtype)
    abs_ops = ops_of(graph, "abs")
    assert len(abs_ops) == 1
    casts = ops_of(graph, "cast")
    assert len(casts) == 1
    assert casts[0].results[0].type.dtype == np.dtype(out_dtype)


def test_sign_complex_keeps_complex_dtype():
    def traced(x):
        return etl.sign(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), etl.complex64))
    assert out.dtype == np.dtype("complex64")


@pytest.mark.parametrize("unary_name", UNARY_MATH)
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (etl.int8, etl.float64), (etl.int32, etl.float64),
        (etl.uint8, etl.float64), (etl.bool_, etl.float64),
        (etl.float16, etl.float16), (etl.float32, etl.float32),
        (etl.float64, etl.float64),
    ],
    ids=str,
)
def test_unary_math_dtypes(unary_name, in_dtype, out_dtype):
    """Unary math: integral/bool input -> float64, float keeps its dtype."""
    fn = getattr(etl, unary_name)

    def traced(x):
        return fn(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), in_dtype))
    assert out.dtype == np.dtype(out_dtype)
    op = ops_of(graph, unary_name)[0]
    assert op.results[0].type.dtype == np.dtype(out_dtype)
    if in_dtype.kind in "biu":
        casts = ops_of(graph, "cast")
        assert len(casts) == 1
        assert casts[0].results[0].type.dtype == np.dtype("float64")
    else:
        assert ops_of(graph, "cast") == []


def test_logical_not_dtypes():
    def traced(x):
        return etl.logical_not(x)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((3,), etl.bool_))
    assert out.dtype == np.dtype("bool")
    assert ops_of(graph, "logical_not")[0].results[0].type.dtype == np.dtype("bool")


# ---------------------------------------------------------------------------
# 6. cast
# ---------------------------------------------------------------------------

CAST_TARGETS = [etl.float16, etl.float64, etl.uint8, etl.bool_, etl.complex64]


@pytest.mark.parametrize("target", CAST_TARGETS, ids=str)
def test_cast_is_exact(target):
    def traced(x):
        return etl.cast(x, target)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((2, 3), etl.int32))
    assert out.dtype == np.dtype(target)
    assert out.shape == (2, 3)
    cast = ops_of(graph, "cast")
    assert len(cast) == 1
    assert cast[0].results[0].type.dtype == np.dtype(target)
    # the IR attribute stores the normalized dtype NAME string
    assert cast[0].attributes == {"dtype": np.dtype(target).name}
    # numerical: exactly numpy astype
    a = np.array([[0, 1, -2], [3, 250, -1]], dtype=np.int32)
    got = run_numpy(traced, a)
    assert got.dtype == np.dtype(target)
    assert np.array_equal(got, a.astype(target))


def test_cast_preserves_symbolic_shape():
    n = etl.dim("n")

    def traced(x):
        return etl.cast(x, etl.float16)

    graph, out = _trace_and_capture(traced, etl.TensorSpec((n, 3), etl.int32))
    assert out.shape == (n, 3)
    assert ops_of(graph, "cast")[0].results[0].type.shape == (n, 3)


def test_cast_of_python_scalar():
    def traced():
        return etl.cast(3, etl.float32)

    graph, out = _trace_and_capture(traced)
    assert out.shape == ()
    assert out.dtype == np.dtype("float32")
    assert ops_of(graph, "constant")[0].results[0].type.dtype == np.dtype("int64")


def test_cast_invalid_dtype_raises():
    def traced(x):
        return etl.cast(x, "not_a_dtype")

    with pytest.raises(etl.DTypeError, match="not a valid dtype"):
        trace_fn(traced, etl.TensorSpec((2,), etl.int32))


# ---------------------------------------------------------------------------
# 7. dtype guards: logical_* require bool, bitwise_* require int/bool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", ["logical_and", "logical_or", "logical_not"])
@pytest.mark.parametrize("bad_dtype", [etl.int32, etl.float32], ids=str)
def test_logical_ops_require_bool(op_name, bad_dtype):
    fn = getattr(etl, op_name)

    def traced(x, y=None):
        if op_name == "logical_not":
            return fn(x)
        return fn(x, y)

    with pytest.raises(etl.DTypeError, match="must have bool dtype"):
        trace_fn(
            traced,
            etl.TensorSpec((2,), bad_dtype),
            etl.TensorSpec((2,), etl.bool_),
        )


@pytest.mark.parametrize("op_name", [name for name, _ in BITWISE])
@pytest.mark.parametrize("bad_dtype", [etl.float32, etl.float16], ids=str)
def test_bitwise_ops_require_integer_or_bool(op_name, bad_dtype):
    fn = getattr(etl, op_name)

    def traced(x, y):
        return fn(x, y)

    with pytest.raises(etl.DTypeError, match="must have integer or bool"):
        trace_fn(
            traced,
            etl.TensorSpec((2,), bad_dtype),
            etl.TensorSpec((2,), etl.int32),
        )


# ---------------------------------------------------------------------------
# 8. error semantics: TraceError / TypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op_name",
    ["add", "multiply", "divide", "power", "maximum",
     "equal", "less", "logical_and", "bitwise_and", "bitwise_or"],
)
def test_both_python_scalars_raise_trace_error(op_name):
    fn = getattr(etl, op_name)

    def traced():
        return fn(1, 2)

    with pytest.raises(etl.TraceError, match="at least one operand must be a SymbolicTensor"):
        trace_fn(traced)


@pytest.mark.parametrize("call", [
    lambda: etl.add(1, 2),
    lambda: etl.sqrt(4.0),
    lambda: etl.cast(3, etl.float32),
    lambda: etl.equal(1, 2),
    lambda: etl.logical_not(True),
], ids=["add", "sqrt", "cast", "equal", "logical_not"])
def test_op_outside_trace_raises_trace_error(call):
    with pytest.raises(etl.TraceError, match="No active trace"):
        call()


def test_concrete_tensor_operand_raises_trace_error():
    def traced(x):
        return etl.add(x, etl.from_numpy(np.ones(2)))

    with pytest.raises(etl.TraceError, match="no eager mode"):
        trace_fn(traced, etl.TensorSpec((2,), etl.float32))


def test_concrete_tensor_unary_operand_raises_trace_error():
    def traced():
        return etl.abs(etl.tensor([1.0, -2.0]))

    with pytest.raises(etl.TraceError, match="no eager mode"):
        trace_fn(traced)


@pytest.mark.parametrize("bad", [
    np.ones(2),              # ndarray
    [1.0, 2.0],              # list
    np.float64(3.0),         # numpy scalar (not a Python scalar)
    np.int64(3),
    "3",
    None,
], ids=["ndarray", "list", "np-float", "np-int", "str", "None"])
def test_unsupported_operand_kind_raises_type_error(bad):
    def traced(x):
        return etl.add(x, bad)

    with pytest.raises(TypeError, match="unsupported operand type"):
        trace_fn(traced, etl.TensorSpec((2,), etl.float32))


# ---------------------------------------------------------------------------
# 9. numerical verification (numpy reference in the SAME promoted dtype)
# ---------------------------------------------------------------------------

_BINARY_NUMERIC_CASES = [
    ("add", np.add, np.array([[1, 2], [3, 4]], np.float32), np.array([[1], [1]], np.float32)),
    ("subtract", np.subtract, np.array([10, 20, 30], np.int32), np.array([3, 4, 5], np.int32)),
    ("multiply", np.multiply, np.array([1.5, -2.0], np.float16), np.array([2.0, 3.0], np.float16)),
    ("power", np.power, np.array([2.0, 3.0, 0.5], np.float32), np.array([2.0, 2.0, 3.0], np.float32)),
    ("remainder", np.remainder, np.array([10, 7, -8], np.int32), np.array([3, 2, 3], np.int32)),
    ("maximum", np.maximum, np.array([1.0, -2.0], np.float32), np.array([-1.0, 3.0], np.float32)),
    ("minimum", np.minimum, np.array([1, 2], np.int32), np.array([-1, 3], np.int32)),
    ("equal", np.equal, np.array([1.0, 2.0], np.float32), np.array([1.0, 3.0], np.float32)),
    ("not_equal", np.not_equal, np.array([1, 2], np.int32), np.array([1, 3], np.int32)),
    ("less", np.less, np.array([1.0, 2.0], np.float32), np.array([1.0, 3.0], np.float32)),
    ("less_equal", np.less_equal, np.array([1.0, 2.0], np.float32), np.array([1.0, 3.0], np.float32)),
    ("greater", np.greater, np.array([1, 2], np.int32), np.array([1, 3], np.int32)),
    ("greater_equal", np.greater_equal, np.array([1, 2], np.int32), np.array([1, 3], np.int32)),
    ("bitwise_and", np.bitwise_and, np.array([6, 5], np.int32), np.array([3, 1], np.int32)),
    ("bitwise_or", np.bitwise_or, np.array([6, 5], np.int32), np.array([3, 1], np.int32)),
    ("bitwise_xor", np.bitwise_xor, np.array([6, 5], np.int32), np.array([3, 1], np.int32)),
    ("logical_and", np.logical_and, np.array([True, False]), np.array([True, True])),
    ("logical_or", np.logical_or, np.array([True, False]), np.array([True, True])),
]


@pytest.mark.parametrize(
    "op_name,np_fn,a,b", _BINARY_NUMERIC_CASES,
    ids=[c[0] for c in _BINARY_NUMERIC_CASES],
)
def test_binary_numerics(op_name, np_fn, a, b):
    fn = getattr(etl, op_name)

    def traced(x, y):
        return fn(x, y)

    # broadcast operands to exercise the runtime path (only when possible)
    aa, bb = a, b
    try:
        expected = np_fn(aa, bb)
    except ValueError:
        return  # not broadcastable — trace-level test above covers shapes
    got = run_numpy(traced, aa, bb)
    assert got.dtype == expected.dtype
    if expected.dtype.kind == "b":
        assert np.array_equal(got, expected)
    else:
        assert np.allclose(got, expected)


def test_equal_not_equal_complex_numerics():
    a = np.array([1 + 2j, 3 + 0j], np.complex64)
    b = np.array([1 + 2j, 4 + 0j], np.complex64)

    def eq(x, y):
        return etl.equal(x, y)

    def ne(x, y):
        return etl.not_equal(x, y)

    assert np.array_equal(run_numpy(eq, a, b), np.equal(a, b))
    assert np.array_equal(run_numpy(ne, a, b), np.not_equal(a, b))


def test_divide_numerics():
    def traced(x, y):
        return etl.divide(x, y)

    x = np.array([6, 7, -8], np.int64)
    y = np.array([2, 2, 3], np.int64)
    got = run_numpy(traced, x, y)
    ref = np.divide(x.astype(np.float64), y.astype(np.float64))
    assert got.dtype == np.dtype("float64")
    assert np.allclose(got, ref)

    x16 = np.array([1.0, 3.0], np.float16)
    y8 = np.array([2, 4], np.int8)
    got16 = run_numpy(traced, x16, y8)
    assert got16.dtype == np.dtype("float16")
    assert np.allclose(got16, np.divide(x16.astype(np.float16), y8.astype(np.float16)))


_PROMOTED_NUMERIC_CASES = [
    ("add", np.add, np.array([-3, -1, 0, 2], np.int8), np.array([1, 2, 3, 250], np.uint8), np.dtype("int16")),
    ("add", np.add, np.array([1.5, 2.5], np.float32), np.array([2, 3], np.int64), np.dtype("float64")),
    ("add", np.add, np.array([1.0, 2.0], np.float16), np.array([0.5, 1.5], np.float16), np.dtype("float16")),
    ("add", np.add, np.array([True, False]), np.array([True, True]), np.dtype("bool")),
    ("multiply", np.multiply, np.array([1, 2], np.int32), np.array([True, False]), np.dtype("int32")),
    ("add", np.add, np.array([1 + 1j, 2 + 0j], np.complex64), np.array([1.0, 2.0], np.float32), np.dtype("complex64")),
    ("add", np.add, np.array([2 ** 62, 1], np.int64), np.array([1, 1], np.uint64), np.dtype("float64")),
]


@pytest.mark.parametrize(
    "op_name,np_fn,a,b,out_dtype", _PROMOTED_NUMERIC_CASES,
    ids=[f"{c[0]}:{c[4].name}" for c in _PROMOTED_NUMERIC_CASES],
)
def test_promoted_dtype_numerics(op_name, np_fn, a, b, out_dtype):
    fn = getattr(etl, op_name)

    def traced(x, y):
        return fn(x, y)

    got = run_numpy(traced, a, b)
    ref = _promoted_ref(np_fn, a, b, out_dtype)
    assert got.dtype == out_dtype
    if out_dtype.kind == "b":
        assert np.array_equal(got, ref)
    else:
        assert np.allclose(got, ref)


_UNARY_MATH_NUMERIC_CASES = [
    ("sqrt", np.sqrt),
    ("exp", np.exp),
    ("log", np.log),
    ("log1p", np.log1p),
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("tanh", np.tanh),
]


@pytest.mark.parametrize(
    "unary_name,np_fn", _UNARY_MATH_NUMERIC_CASES,
    ids=[c[0] for c in _UNARY_MATH_NUMERIC_CASES],
)
def test_unary_math_numerics(unary_name, np_fn):
    fn = getattr(etl, unary_name)
    x = np.array([0.1, 0.5, 1.0, 2.0], np.float32)

    def traced(t):
        return fn(t)

    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("float32")
    assert np.allclose(got, np_fn(x), rtol=1e-6)


@pytest.mark.parametrize(
    "unary_name,ref_fn",
    [
        ("sigmoid", lambda x: np.float32(1) / (np.float32(1) + np.exp(-x))),
        ("relu", lambda x: np.maximum(x, np.float32(0))),
        ("gelu", lambda x: np.float32(0.5) * x * (np.float32(1) + _erf_vec(x / np.float32(np.sqrt(2))))),
        ("erf", lambda x: _erf_vec(x)),
    ],
    ids=["sigmoid", "relu", "gelu", "erf"],
)
def test_activation_numerics(unary_name, ref_fn):
    fn = getattr(etl, unary_name)
    x = np.array([-2.0, -0.5, 0.0, 0.7], np.float32)

    def traced(t):
        return fn(t)

    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("float32")
    assert np.allclose(got, ref_fn(x), rtol=1e-6, atol=1e-6)


def test_unary_math_integral_input_runs_in_float64():
    def traced(x):
        return etl.sqrt(x)

    x = np.array([1, 4, 9], np.int32)
    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("float64")
    assert np.allclose(got, np.sqrt(x.astype(np.float64)))


@pytest.mark.parametrize(
    "unary_name,np_fn",
    [("abs", np.abs), ("negate", np.negative), ("square", np.square)],
    ids=["abs", "negate", "square"],
)
@pytest.mark.parametrize("dtype", [np.int32, np.float32], ids=str)
def test_unary_preserving_numerics(unary_name, np_fn, dtype):
    fn = getattr(etl, unary_name)
    x = np.array([-3, -1, 0, 2], dtype=dtype)

    def traced(t):
        return fn(t)

    got = run_numpy(traced, x)
    assert got.dtype == np.dtype(dtype)
    assert np.allclose(got, np_fn(x))


def test_sign_numerics():
    def traced(x):
        return etl.sign(x)

    x = np.array([-3.5, 0.0, 2.5], np.float32)
    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("float32")
    assert np.array_equal(got, np.sign(x))

    z = np.array([1 + 2j, -3 - 4j], np.complex64)
    gotz = run_numpy(traced, z)
    assert gotz.dtype == np.dtype("complex64")
    assert np.allclose(gotz, np.sign(z))


def test_abs_complex_numerics_give_real_magnitude():
    def traced(x):
        return etl.abs(x)

    z = np.array([3 + 4j, -1.5 + 0j], np.complex64)
    got = run_numpy(traced, z)
    assert got.dtype == np.dtype("float32")
    assert np.allclose(got, np.abs(z))


def test_logical_not_numerics():
    def traced(x):
        return etl.logical_not(x)

    x = np.array([True, False, True])
    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("bool")
    assert np.array_equal(got, np.logical_not(x))


def test_comparison_with_python_scalar():
    def traced(x):
        return etl.less(x, 3)

    x = np.array([1, 3, 5], np.int32)
    got = run_numpy(traced, x)
    assert got.dtype == np.dtype("bool")
    assert np.array_equal(got, np.less(x, 3))


# ---------------------------------------------------------------------------
# 10. IR construction contract: names, arities, effects, locations
# ---------------------------------------------------------------------------

_IR_CASES = (
    [("binary", name) for name, _ in ALL_BINARY]
    + [("divide", "divide")]
    + [("unary", name) for name in UNARY_PRESERVING]
    + [("unary", name) for name in UNARY_MATH]
    + [("unary", "logical_not")]
    + [("cast", "cast")]
)


@pytest.mark.parametrize("kind,op_name", _IR_CASES, ids=[c[1] for c in _IR_CASES])
def test_ir_construction_contract(kind, op_name):
    """Every op: right IR name, arity, pure effect, location captured, and
    the IR result type agrees with the returned SymbolicTensor."""
    fn = getattr(etl, op_name)
    if kind == "binary":
        dtype = _binary_spec_dtype(op_name)
        spec_a, spec_b = etl.TensorSpec((2, 3), dtype), etl.TensorSpec((1, 3), dtype)
        arity = 2
    elif kind == "divide":
        spec_a, spec_b = etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((1, 3), etl.float32)
        arity = 2
    elif kind == "cast":
        spec_a = etl.TensorSpec((2, 3), etl.int32)
        spec_b = None
        arity = 1
    else:
        dtype = etl.bool_ if op_name == "logical_not" else etl.float32
        spec_a = etl.TensorSpec((2, 3), dtype)
        spec_b = None
        arity = 1

    def traced(x, y=None):
        if arity == 2:
            return fn(x, y)
        if kind == "cast":
            return fn(x, etl.float16)
        return fn(x)

    graph, out = _trace_and_capture(traced, spec_a, spec_b) if arity == 2 else _trace_and_capture(traced, spec_a)
    ops = ops_of(graph, op_name)
    assert len(ops) == 1
    op = ops[0]
    assert op.name == op_name
    assert len(op.operands) == arity
    assert op.effect == "pure"
    # IR result type == SymbolicTensor metadata
    assert op.results[0].type.dtype == out.dtype
    assert op.results[0].type.shape == out.shape
    assert out.value is op.results[0]
    # attributes: {} everywhere except cast's dtype (normalized name string)
    if kind == "cast":
        assert op.attributes == {"dtype": "float16"}
    else:
        assert op.attributes == {}
    # call-site location captured from THIS test file
    assert op.location.file.endswith("test_elementwise.py")
    assert op.location.line > 0


def test_location_capture_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ETL_DISABLE_LOCATIONS", "1")

    def traced(x):
        return etl.add(x, x)

    graph, _ = _trace_and_capture(traced, etl.TensorSpec((2,), etl.float32))
    op = ops_of(graph, "add")[0]
    assert op.location.file == "<unknown>"
    assert op.location.line == 0
