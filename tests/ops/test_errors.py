"""Error semantics of the full public op surface (contract: ``etl/ops/CONTEXT.md``).

Every public op in ``etl.ops.__all__`` (67 names) is exercised in three
parametrized error categories:

(a) called OUTSIDE a trace -> ``TraceError`` with the directing message
    ("No active trace: tensor ops can only be called while tracing", naming
    ``etl.trace`` / ``etl.evaluate``);
(b) called INSIDE a trace with a concrete ``Tensor`` operand ->
    ``TraceError`` with the mandated three-option message (explicit input /
    ``etl.constant`` / ``etl.evaluate`` — no eager mode);
(c) contracted wrong-dtype / static-shape failures -> ``DTypeError`` /
    ``ShapeError`` (logical on non-bool, bitwise on float, float gather
    indices, static broadcast mismatch).

Plus: two-Python-scalar binary calls, unsupported operand kinds (TypeError),
and the contract that error messages include the captured call-site
location.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import etl

from tests.ops.conftest import ops_of

# ---------------------------------------------------------------------------
# Per-op table of MINIMAL VALID call arguments (all 67 public ops).
# Each entry: op name -> (args_builder, kwargs). ``args_builder`` receives the
# dict ``t`` of symbolic tensors below and returns the positional args of a
# valid call; ``kwargs`` are constant per op.
# ---------------------------------------------------------------------------

_CONST_TENSOR = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))
_RUNTIME_CALLBACK = np.add
_RUNTIME_RESULT = etl.TensorSpec((4,), etl.float32)

# Symbolic tensors every traced fn receives, in this exact order.
_TENSOR_SPECS = (
    ("x", (4,), etl.float32),  # general elementwise operand
    ("y", (4,), etl.float32),
    ("xi", (4,), etl.int32),  # integer operand (bitwise)
    ("xb", (4,), etl.bool_),  # bool operand (logical / select)
    ("a", (2, 2), etl.float32),  # linalg operands
    ("b", (2, 2), etl.float32),
    ("xc", (1, 1, 2), etl.float32),  # conv input (N, C, spatial)
    ("wc", (1, 1, 1), etl.float32),  # conv kernel (C_out, C_in, k)
    ("idx", (2,), etl.int32),  # gather/scatter indices
    ("u", (2,), etl.float32),  # scatter updates
)

OP_CALLS = {
    # --- elementwise arithmetic / math / bitwise (28) ---
    "add": (lambda t: (t["x"], t["x"]), {}),
    "subtract": (lambda t: (t["x"], t["x"]), {}),
    "multiply": (lambda t: (t["x"], t["x"]), {}),
    "divide": (lambda t: (t["x"], t["x"]), {}),
    "power": (lambda t: (t["x"], t["x"]), {}),
    "remainder": (lambda t: (t["x"], t["x"]), {}),
    "maximum": (lambda t: (t["x"], t["x"]), {}),
    "minimum": (lambda t: (t["x"], t["x"]), {}),
    "abs": (lambda t: (t["x"],), {}),
    "negate": (lambda t: (t["x"],), {}),
    "square": (lambda t: (t["x"],), {}),
    "sqrt": (lambda t: (t["x"],), {}),
    "exp": (lambda t: (t["x"],), {}),
    "log": (lambda t: (t["x"],), {}),
    "log1p": (lambda t: (t["x"],), {}),
    "sin": (lambda t: (t["x"],), {}),
    "cos": (lambda t: (t["x"],), {}),
    "tan": (lambda t: (t["x"],), {}),
    "tanh": (lambda t: (t["x"],), {}),
    "sigmoid": (lambda t: (t["x"],), {}),
    "relu": (lambda t: (t["x"],), {}),
    "gelu": (lambda t: (t["x"],), {}),
    "erf": (lambda t: (t["x"],), {}),
    "sign": (lambda t: (t["x"],), {}),
    "cast": (lambda t: (t["x"],), {"dtype": etl.float64}),
    "bitwise_and": (lambda t: (t["xi"], t["xi"]), {}),
    "bitwise_or": (lambda t: (t["xi"], t["xi"]), {}),
    "bitwise_xor": (lambda t: (t["xi"], t["xi"]), {}),
    # --- comparison / logical / selection (10) ---
    "equal": (lambda t: (t["x"], t["x"]), {}),
    "not_equal": (lambda t: (t["x"], t["x"]), {}),
    "less": (lambda t: (t["x"], t["x"]), {}),
    "less_equal": (lambda t: (t["x"], t["x"]), {}),
    "greater": (lambda t: (t["x"], t["x"]), {}),
    "greater_equal": (lambda t: (t["x"], t["x"]), {}),
    "logical_and": (lambda t: (t["xb"], t["xb"]), {}),
    "logical_or": (lambda t: (t["xb"], t["xb"]), {}),
    "logical_not": (lambda t: (t["xb"],), {}),
    "select": (lambda t: (t["xb"], t["x"], t["x"]), {}),
    # --- indexing / shape manipulation (8) ---
    "broadcast": (lambda t: (t["x"],), {"shape": (2, 4)}),
    "reshape": (lambda t: (t["x"],), {"shape": (2, 2)}),
    "transpose": (lambda t: (t["a"],), {"axes": (1, 0)}),
    "slice": (lambda t: (t["x"], (0,), (2,), (1,)), {}),
    "gather": (lambda t: (t["x"], t["idx"]), {}),
    "scatter": (lambda t: (t["x"], t["idx"], t["u"]), {}),
    "concatenate": (lambda t: ([t["x"], t["x"]],), {}),
    "pad": (lambda t: (t["x"],), {"config": ((1, 1),), "value": 0.0}),
    # --- reductions (12) ---
    "reduce_sum": (lambda t: (t["x"],), {}),
    "reduce_max": (lambda t: (t["x"],), {}),
    "reduce_min": (lambda t: (t["x"],), {}),
    "reduce_mean": (lambda t: (t["x"],), {}),
    "reduce_prod": (lambda t: (t["x"],), {}),
    "sum": (lambda t: (t["x"],), {}),
    "max": (lambda t: (t["x"],), {}),
    "min": (lambda t: (t["x"],), {}),
    "mean": (lambda t: (t["x"],), {}),
    "prod": (lambda t: (t["x"],), {}),
    "argmax": (lambda t: (t["x"],), {}),
    "argmin": (lambda t: (t["x"],), {}),
    # --- linalg (6) ---
    "dot": (lambda t: (t["a"], t["b"]), {}),
    "conv": (lambda t: (t["xc"], t["wc"]), {}),
    "tril": (lambda t: (t["a"],), {}),
    "triu": (lambda t: (t["a"],), {}),
    "cumsum": (lambda t: (t["x"],), {}),
    "solve": (lambda t: (t["a"], t["b"]), {}),
    # --- constants / escape hatches (3) ---
    # ``constant`` takes a concrete Tensor by design — it is excluded from
    # the concrete-Tensor-operand category but present in the no-trace one.
    "constant": (lambda t: (_CONST_TENSOR,), {}),
    "runtime_call": (
        lambda t: (_RUNTIME_CALLBACK, t["x"], t["x"]),
        {"result": _RUNTIME_RESULT},
    ),
    "stop_gradient": (lambda t: (t["x"],), {}),
}

# Plain-Python stand-ins for the table builders. Every op calls
# ``check_in_trace()`` FIRST (verified below), so outside a trace these calls
# must fail with the no-trace TraceError before any argument is inspected.
_STANDINS = {
    "x": 1.0, "y": 1.0, "xi": 1, "xb": True, "a": 1.0, "b": 1.0,
    "xc": 1.0, "wc": 1.0, "idx": 1, "u": 1.0,
}


def _specs():
    return [etl.TensorSpec(shape, dtype) for _, shape, dtype in _TENSOR_SPECS]


def _trace_call(op_name, args_builder, kwargs=None):
    """Trace a fn that calls ``op_name(*args_builder(t), **kwargs)``."""
    kwargs = kwargs or {}

    def fn(x, y, xi, xb, a, b, xc, wc, idx, u):
        t = {
            "x": x, "y": y, "xi": xi, "xb": xb, "a": a, "b": b,
            "xc": xc, "wc": wc, "idx": idx, "u": u,
        }
        return getattr(etl, op_name)(*args_builder(t), **kwargs)

    return etl.trace(fn, *_specs())


def _tensor_variant(op_name, args_builder):
    """Replace one symbolic operand of the valid call with a concrete Tensor."""
    tensor = _CONST_TENSOR

    def build(t):
        args = list(args_builder(t))
        if op_name == "concatenate":
            args[0] = [t["x"], tensor]  # Tensor inside the container
        elif op_name == "runtime_call":
            args[1] = tensor  # args[0] is the callback, args[1] the 1st operand
        else:
            args[0] = tensor
        return tuple(args)

    return build


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------

def test_op_table_covers_all_public_ops():
    """The table must exercise exactly the 67 public op names."""
    assert set(OP_CALLS) == set(etl.ops.__all__)
    assert len(etl.ops.__all__) == 67


# ---------------------------------------------------------------------------
# (a) called OUTSIDE a trace -> directing TraceError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", sorted(OP_CALLS))
def test_op_outside_trace_raises_directing_trace_error(op_name):
    args_builder, kwargs = OP_CALLS[op_name]
    with pytest.raises(etl.TraceError) as exc:
        getattr(etl, op_name)(*args_builder(_STANDINS), **kwargs)
    message = str(exc.value)
    assert message.startswith(
        "No active trace: tensor ops can only be called while tracing"
    )
    assert "etl.trace" in message
    assert "etl.evaluate" in message


def test_no_trace_message_mentions_defn():
    """Binding contract (etl/ops/CONTEXT.md, "Unified semantics" #2): the
    no-active-trace message must mention ``etl.trace`` / ``@etl.defn`` /
    ``etl.evaluate``.
    """
    with pytest.raises(etl.TraceError) as exc:
        etl.add(1.0, 1.0)
    assert "@etl.defn" in str(exc.value)


# ---------------------------------------------------------------------------
# (b) concrete Tensor operand INSIDE a trace -> mandated three-option message
# ---------------------------------------------------------------------------

# ``constant`` is the one op whose documented input IS a concrete Tensor.
_TENSOR_OPERAND_SKIP = {"constant"}


@pytest.mark.parametrize("op_name", sorted(set(OP_CALLS) - _TENSOR_OPERAND_SKIP))
def test_op_inside_trace_rejects_concrete_tensor_operand(op_name):
    args_builder, kwargs = OP_CALLS[op_name]
    with pytest.raises(etl.TraceError) as exc:
        _trace_call(op_name, _tensor_variant(op_name, args_builder), kwargs)
    message = str(exc.value)
    assert message.startswith(
        "Concrete Tensor operands are not allowed in graph ops "
        "(etl has no eager mode)"
    )
    # The mandated three options all appear.
    assert "explicit input" in message
    assert "etl.constant" in message
    assert "etl.evaluate" in message


# ---------------------------------------------------------------------------
# (c) contracted wrong-dtype / static-shape failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", ["logical_and", "logical_or", "logical_not"])
def test_logical_ops_require_bool_dtype(op_name):
    def fn(x):
        op = getattr(etl, op_name)
        return op(x) if op_name == "logical_not" else op(x, x)

    with pytest.raises(
        etl.DTypeError, match=rf"{op_name}: operands must have bool dtype"
    ):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))


@pytest.mark.parametrize("op_name", ["bitwise_and", "bitwise_or", "bitwise_xor"])
def test_bitwise_ops_reject_float_dtype(op_name):
    def fn(x):
        return getattr(etl, op_name)(x, x)

    with pytest.raises(
        etl.DTypeError,
        match=rf"{op_name}: operands must have integer or bool dtype",
    ):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))


def test_gather_rejects_float_indices():
    def fn(x, indices):
        return etl.gather(x, indices)

    with pytest.raises(
        etl.DTypeError,
        match=r"gather: indices must be an integer dtype \(int32/int64\), got float32",
    ):
        etl.trace(
            fn,
            etl.TensorSpec((4,), etl.float32),
            etl.TensorSpec((2,), etl.float32),
        )


def test_static_broadcast_mismatch_raises_shape_error():
    def fn(a, b):
        return etl.add(a, b)

    with pytest.raises(
        etl.ShapeError, match=r"cannot broadcast incompatible dims 2 and 4"
    ):
        etl.trace(
            fn,
            etl.TensorSpec((2, 3), etl.float32),
            etl.TensorSpec((4, 5), etl.float32),
        )


# ---------------------------------------------------------------------------
# call-site location in error messages
# ---------------------------------------------------------------------------

def test_op_locations_are_captured_for_plain_fn_trace():
    """Positive control: tracing a plain fn captures the real call site
    (inspect stack frames), so a graph location EXISTS for the error contract
    below."""
    def fn(a):
        return etl.add(a, a)

    graph = etl.trace(fn, etl.TensorSpec((4,), etl.float32))
    (op,) = ops_of(graph, "add")
    assert op.location.file == __file__
    assert op.location.line > 0


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain-fn", "defn"])
def test_shape_error_message_includes_call_site_location(wrapped):
    """Contract (etl/core/errors.py + root CONTEXT.md): error messages
    include the call-site location whenever a graph location exists. The
    location IS captured here (see the control test above) — for both a plain
    fn and an ``@etl.defn``-wrapped one.
    """
    def fn(a, b):
        return etl.add(a, b)

    traced = etl.defn(fn) if wrapped else fn
    with pytest.raises(
        etl.ShapeError, match=r"cannot broadcast incompatible dims 2 and 4"
    ) as exc:
        etl.trace(
            traced,
            etl.TensorSpec((2, 3), etl.float32),
            etl.TensorSpec((4, 5), etl.float32),
        )
    assert f"{os.path.basename(__file__)}:" in str(exc.value)


# ---------------------------------------------------------------------------
# two Python scalars / unsupported operand kinds
# ---------------------------------------------------------------------------

_BINARY_ELEMENTWISE = [
    "add", "subtract", "multiply", "divide", "power", "remainder",
    "maximum", "minimum",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or",
    "bitwise_and", "bitwise_or", "bitwise_xor",
]


@pytest.mark.parametrize("op_name", _BINARY_ELEMENTWISE)
def test_binary_op_with_two_python_scalars_raises_trace_error(op_name):
    def fn(x):
        return getattr(etl, op_name)(1, 2)

    with pytest.raises(
        etl.TraceError, match="at least one operand must be a SymbolicTensor"
    ):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))


@pytest.mark.parametrize("op_name", ["dot", "conv", "solve"])
def test_linalg_binary_op_with_two_python_scalars_raises(op_name):
    def fn(x):
        return getattr(etl, op_name)(1.0, 2.0)

    with pytest.raises(etl.TraceError, match="both operands are Python scalars"):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))


@pytest.mark.parametrize(
    "bad,kind",
    [
        ([1.0, 2.0], "list"),
        ("s", "str"),
        (None, "NoneType"),
        (np.array([1.0, 2.0]), "ndarray"),
        (np.float64(1.0), "float64"),
    ],
    ids=["list", "str", "None", "ndarray", "numpy-scalar"],
)
def test_unsupported_operand_kind_raises_type_error(bad, kind):
    def fn(x):
        return etl.add(x, bad)

    with pytest.raises(TypeError, match=rf"unsupported operand type {kind}"):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))


def test_unary_op_with_ndarray_raises_type_error():
    def fn(x):
        return etl.exp(np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="unsupported operand type ndarray"):
        etl.trace(fn, etl.TensorSpec((4,), etl.float32))
