"""pytest tests for CALLING a declared `etl.block` BlockOp.

Covers `block_call` IR-op construction: op layout and attributes, static
specialization, static-argument binding rules, operand dtype/shape errors,
outside-trace errors, and multi-output calls. No execution (`etl.evaluate`
is deliberately unused) — the `block_call` op is a graph placeholder whose
computing is resolved at lower time.

NOTE: the block registry is process-wide and names can never be re-declared,
so every block here has a unique name prefixed `call_` and is declared
exactly once at module level (never inside tests/parametrize).
"""

import enum

import numpy as np
import pytest

import etl
from etl.block import BlockError

F32_4 = etl.TensorSpec((4,), etl.float32)
F32_0 = etl.TensorSpec((), etl.float32)


# ---------------------------------------------------------------------------
# Global declarations (unique `call_` names, declared once).
# ---------------------------------------------------------------------------

# Main 2-input block: one required attribute (scale), one optional (eps).
call_blk = etl.block(
    "call_blk",
    inputs=[F32_4, F32_4],
    outputs=[F32_4],
    attributes={"scale": float, "eps": 1e-5},
)

# Two output specs — including a scalar output from a 1-d input.
call_multi = etl.block(
    "call_multi",
    inputs=[F32_4],
    outputs=[F32_4, etl.TensorSpec((), etl.int32)],
)

# None dims in the input/output specs: runtime-dynamic wildcards.
call_wild = etl.block(
    "call_wild",
    inputs=[etl.TensorSpec((None,), etl.float32)],
    outputs=[etl.TensorSpec((None,), etl.float32)],
)

# Single required attribute typed `object`: accepts every static-value kind
# so one block covers the whole StaticValue kind round-trip matrix.
call_kind_blk = etl.block(
    "call_kind_blk",
    inputs=[F32_0],
    outputs=[F32_0],
    attributes={"v": object},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def block_call_ops(graph):
    """All `block_call` ops in the traced graph's entry block."""
    return [op for op in graph.module.main.entry_block.ops if op.name == "block_call"]


def single_block_call(graph):
    """Assert exactly one `block_call` op exists and return it."""
    ops = block_call_ops(graph)
    assert len(ops) == 1
    return ops[0]


# ---------------------------------------------------------------------------
# 1. Op construction
# ---------------------------------------------------------------------------


def test_call_builds_one_block_call_op():
    captured = {}

    @etl.defn
    def f(x, y):
        captured["out"] = call_blk(x, y, 0.5)
        return captured["out"]

    graph = etl.trace(f, F32_4, F32_4)
    graph.verify()  # IR-level invariants hold for the block_call op

    op = single_block_call(graph)
    assert op.effect == "read"  # fixed by ir's canonical opdef
    assert set(op.attributes) == {"block_name", "static_args", "result_specs"}
    assert op.attributes["block_name"] == "call_blk"
    # Only the supplied static arg is recorded; the optional `eps` (left
    # unset) is NOT recorded — its default is fixed by the schema.
    assert op.attributes["static_args"] == {"scale": {"kind": "float", "value": 0.5}}
    result_specs = op.attributes["result_specs"]
    assert len(result_specs) == 1
    assert result_specs[0].dtype == etl.float32
    assert tuple(result_specs[0].shape) == (4,)
    assert len(op.operands) == 2
    assert len(op.results) == 1

    # The call returns ONE SymbolicTensor (single output_spec), typed from
    # the block's output spec and wrapping the op's result value.
    out = captured["out"]
    assert isinstance(out, etl.SymbolicTensor)
    assert out.dtype == etl.float32
    assert tuple(out.shape) == (4,)
    assert out.value is op.results[0]
    assert op.results[0].type.dtype == etl.float32
    assert tuple(op.results[0].type.shape) == (4,)


def test_keyword_static_args_recorded():
    captured = {}

    @etl.defn
    def f(x, y):
        captured["out"] = call_blk(x, y, scale=0.5, eps=1e-3)
        return captured["out"]

    graph = etl.trace(f, F32_4, F32_4)
    assert single_block_call(graph).attributes["static_args"] == {
        "scale": {"kind": "float", "value": 0.5},
        "eps": {"kind": "float", "value": 1e-3},
    }


# ---------------------------------------------------------------------------
# 2. Static specialization
# ---------------------------------------------------------------------------


@etl.defn
def _scaled(x, y, scale):
    return call_blk(x, y, scale)


def test_static_specialization_changes_payload():
    g05a = etl.trace(_scaled, F32_4, F32_4, 0.5)
    g10 = etl.trace(_scaled, F32_4, F32_4, 1.0)
    g05b = etl.trace(_scaled, F32_4, F32_4, 0.5)

    assert single_block_call(g05a).attributes["static_args"] == {
        "scale": {"kind": "float", "value": 0.5}
    }
    assert single_block_call(g10).attributes["static_args"] == {
        "scale": {"kind": "float", "value": 1.0}
    }
    # Same static value -> identical payload (op identity / cache keys).
    assert single_block_call(g05a).attributes["static_args"] == single_block_call(
        g05b
    ).attributes["static_args"]


# ---------------------------------------------------------------------------
# 3. Static-kind payload round-trips (StaticValue.encode formats)
# ---------------------------------------------------------------------------


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@pytest.mark.parametrize(
    "value, expected_kind, expected_value",
    [
        (None, "none", None),
        (True, "bool", True),
        (7, "int", 7),
        (-0.5, "float", -0.5),
        (1.5 - 2.5j, "complex", (1.5, -2.5)),  # complex -> (re, im) tuple
        ("relu", "str", "relu"),
        (np.dtype("float32"), "dtype", "float32"),  # dtype -> .name
        (etl.float32, "dtype", "float32"),
        (Mode.FAST, "enum", f"{Mode.__module__}.{Mode.__qualname__}.FAST"),
        (slice(1, 4, 2), "slice", (1, 4, 2)),  # slice -> (start, stop, step)
    ],
    ids=["none", "bool", "int", "float", "complex", "str", "np_dtype",
         "etl_dtype", "enum", "slice"],
)
def test_static_kind_payload_round_trip(value, expected_kind, expected_value):
    @etl.defn
    def f(x):
        return call_kind_blk(x, value)

    graph = etl.trace(f, F32_0)
    assert single_block_call(graph).attributes["static_args"] == {
        "v": {"kind": expected_kind, "value": expected_value}
    }


# ---------------------------------------------------------------------------
# 4. Static-argument binding rules
# ---------------------------------------------------------------------------


def test_positional_static_binds_in_schema_order():
    captured = {}

    @etl.defn
    def f(x, y):
        # Positional statics bind in SCHEMA order: 0.5 -> scale (first slot),
        # 1e-3 -> eps (second slot) — regardless of call order.
        captured["out"] = call_blk(x, y, 0.5, 1e-3)
        return captured["out"]

    graph = etl.trace(f, F32_4, F32_4)
    assert single_block_call(graph).attributes["static_args"] == {
        "scale": {"kind": "float", "value": 0.5},
        "eps": {"kind": "float", "value": 1e-3},
    }


@pytest.mark.parametrize(
    "call_fn, match",
    [
        # positional would bind to a slot already given by keyword
        (lambda x, y: call_blk(x, y, 0.5, scale=0.7), "already given by keyword"),
        (lambda x, y: call_blk(x, y, scale=0.5, banana=1.0), "undeclared attribute"),
        (lambda x, y: call_blk(x, y, "x"), "expects float"),
        (lambda x, y: call_blk(x, y), "missing required attribute"),
        # all slots filled, one more static positional
        (lambda x, y: call_blk(x, y, 0.5, 1e-3, 2.0), "too many positional arguments"),
    ],
    ids=["double_fill", "undeclared_kw", "wrong_type", "missing_required",
         "extra_positional"],
)
def test_static_binding_errors(call_fn, match):
    with pytest.raises(BlockError, match=match):
        etl.trace(etl.defn(call_fn), F32_4, F32_4)


# ---------------------------------------------------------------------------
# 5. Operand errors
# ---------------------------------------------------------------------------


def test_operand_dtype_mismatch_raises():
    @etl.defn
    def f(x, y):
        return call_blk(x, y, 0.5)

    with pytest.raises(etl.DTypeError, match="dtype mismatch"):
        etl.trace(f, etl.TensorSpec((4,), etl.int32), F32_4)


@pytest.mark.parametrize(
    "bad_spec, match",
    [
        (etl.TensorSpec((3, 4), etl.float32), "rank mismatch"),
        (etl.TensorSpec((5,), etl.float32), "shape mismatch"),
    ],
    ids=["rank", "dim"],
)
def test_operand_shape_mismatch_raises(bad_spec, match):
    @etl.defn
    def f(x, y):
        return call_blk(x, y, 0.5)

    with pytest.raises(etl.ShapeError, match=match):
        etl.trace(f, bad_spec, F32_4)


@pytest.mark.parametrize("size", [2, 5, 7])
def test_wildcard_none_dims_accept_any_extent(size):
    @etl.defn
    def f(x):
        return call_wild(x)

    graph = etl.trace(f, etl.TensorSpec((size,), etl.float32))
    single_block_call(graph)
    graph.verify()


def test_too_few_operands_raises():
    @etl.defn
    def f(x, y):
        return call_blk(x, 0.5)  # one symbolic operand, input_specs wants 2

    with pytest.raises(BlockError, match="expected"):
        etl.trace(f, F32_4, F32_4)


# ---------------------------------------------------------------------------
# 6. Outside trace / concrete tensors
# ---------------------------------------------------------------------------


def test_call_outside_trace_raises():
    x = np.zeros(4, dtype=np.float32)
    with pytest.raises(etl.TraceError, match="No active trace"):
        call_blk(x, x, 0.5)


@pytest.mark.parametrize(
    "make_tensor",
    [
        lambda: etl.zeros((4,), etl.float32),
        lambda: etl.tensor(np.zeros(4, dtype=np.float32)),
    ],
    ids=["zeros", "tensor"],
)
def test_concrete_tensor_operand_inside_trace_raises(make_tensor):
    @etl.defn
    def f(x):
        t = make_tensor()
        return call_blk(x, t, 0.5)

    with pytest.raises(etl.TraceError, match="Concrete Tensor"):
        etl.trace(f, F32_4)


# ---------------------------------------------------------------------------
# 7. Multi-output calls
# ---------------------------------------------------------------------------


def test_multi_output_returns_tuple():
    captured = {}

    @etl.defn
    def f(x):
        captured["out"] = call_multi(x)
        return captured["out"]

    graph = etl.trace(f, F32_4)
    graph.verify()
    op = single_block_call(graph)
    assert len(op.results) == 2

    # N output_specs -> a tuple of N SymbolicTensors with the DECLARED
    # dtypes/shapes (the scalar output is 0-d even though the input is 1-d).
    out = captured["out"]
    assert isinstance(out, tuple)
    assert len(out) == 2
    r0, r1 = out
    assert isinstance(r0, etl.SymbolicTensor)
    assert isinstance(r1, etl.SymbolicTensor)
    assert r0.dtype == etl.float32
    assert tuple(r0.shape) == (4,)
    assert r1.dtype == etl.int32
    assert tuple(r1.shape) == ()
    assert out[0].value is op.results[0]
    assert out[1].value is op.results[1]
