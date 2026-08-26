"""Graph-construction tests for the six dist collectives + rank/world_size.

Asserts the IR contract of ``etl/dist`` (sibling package, read-only — the
contract lives in ``etl/dist/CONTEXT.md``, the canonical op defs in
``etl/ir/op_defs/collective.py``):

- each collective builds exactly ONE op with effect kind ``collective``,
  1 operand / 1 result, attrs ``group`` + ``group_size`` plus per-op params;
- ``broadcast`` builds the ``broadcast_collective`` op (the shape op
  ``broadcast`` owns its name) whose ``src_rank`` attr is defaulted to 0 by
  the registry;
- ``rank()`` / ``world_size()`` are scalar int64 graph values (effect
  ``read``, attr ``group="world"``) — not Python ints;
- error matrix: eager concrete-Tensor operands, non-tensor operands,
  outside-trace calls, unknown reductions, bad axes, indivisible dims,
  bad ``src_rank``, malformed permutation pairs, non-Group groups.

These tests are graph-construction only: no execution, CPU only, small shapes.
"""

import numpy as np
import pytest

import etl
from etl import core
from etl.dist.collectives import REDUCTIONS

# --- fixtures / helpers -----------------------------------------------------

#: Explicit 4-rank group used throughout (group="data", size 4).
G4 = etl.dist.group("data", (0, 1, 2, 3))

#: Default local input shape: rank 2, dims divisible by the group size.
SHAPE = (4, 8)


def _collective_defn(fn_name, **kwargs):
    """Build a defn calling ``getattr(etl.dist, fn_name)(x, **kwargs)``."""
    collective = getattr(etl.dist, fn_name)

    @etl.defn
    def fn(x):
        return collective(x, **kwargs)

    return fn


def _trace(fn, shape=SHAPE, dtype=core.float32):
    return etl.trace(fn, core.TensorSpec(shape, dtype))


def _single_op(fn, op_name, shape=SHAPE, dtype=core.float32):
    """Trace ``fn`` and return ``(entry_block, op)`` for the single named op."""
    graph = _trace(fn, shape, dtype)
    block = graph.module.functions[0].entry_block
    matches = [o for o in block.ops if o.name == op_name]
    assert len(matches) == 1, f"expected exactly one {op_name!r} op"
    return block, matches[0]


#: (dist fn, IR op name, call kwargs, full expected attrs) per collective.
COLLECTIVE_CASES = [
    pytest.param(
        "all_reduce", "all_reduce", {"op": "sum"},
        {"reduce_op": "sum", "group": "data", "group_size": 4},
        id="all_reduce",
    ),
    pytest.param(
        "all_gather", "all_gather", {"axis": 0},
        {"axis": 0, "group": "data", "group_size": 4},
        id="all_gather",
    ),
    pytest.param(
        "reduce_scatter", "reduce_scatter", {"op": "sum", "axis": 0},
        {"reduce_op": "sum", "axis": 0, "group": "data", "group_size": 4},
        id="reduce_scatter",
    ),
    pytest.param(
        "all_to_all", "all_to_all", {"split_axis": 0, "concat_axis": 1},
        {"split_axis": 0, "concat_axis": 1, "group": "data", "group_size": 4},
        id="all_to_all",
    ),
    pytest.param(
        "broadcast", "broadcast_collective", {"src_rank": 0},
        {"group": "data", "src_rank": 0, "group_size": 4},
        id="broadcast",
    ),
    pytest.param(
        "collective_permute", "collective_permute",
        {"source_target_pairs": ((0, 1), (1, 2), (2, 3), (3, 0))},
        {"source_target_pairs": ((0, 1), (1, 2), (2, 3), (3, 0)),
         "group": "data", "group_size": 4},
        id="collective_permute",
    ),
]

DTYPES = [
    pytest.param(core.float32, id="float32"),
    pytest.param(core.int32, id="int32"),
    pytest.param(core.float64, id="float64"),
]


# --- 1. the six collectives build one collective-effect op -----------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn_name,op_name,kwargs,expected_attrs", COLLECTIVE_CASES)
def test_collective_builds_one_collective_op(
    fn_name, op_name, kwargs, expected_attrs, dtype
):
    """One op per call: effect, arity, full attrs, dtype, entry-arg operand."""
    fn = _collective_defn(fn_name, group=G4, **kwargs)
    block, op = _single_op(fn, op_name, dtype=dtype)

    assert [o.name for o in block.ops] == [op_name, "return"]
    assert op.effect == "collective"
    assert len(op.operands) == 1
    assert len(op.results) == 1
    # The operand is the traced entry-block argument (the tensor input).
    assert op.operands[0] is block.arguments[0]
    assert op.attributes == expected_attrs
    assert op.results[0].type.dtype == dtype


#: Local-shape semantics per the dist contract's worked examples.
SHAPE_CASES = [
    pytest.param("all_reduce", "all_reduce", {"op": "sum"}, (4, 8),
                 id="all_reduce-identity"),
    pytest.param("broadcast", "broadcast_collective", {"src_rank": 0}, (4, 8),
                 id="broadcast-identity"),
    pytest.param(
        "collective_permute", "collective_permute",
        {"source_target_pairs": ((0, 1), (1, 2), (2, 3), (3, 0))}, (4, 8),
        id="collective_permute-identity",
    ),
    pytest.param("all_gather", "all_gather", {"axis": 0}, (16, 8),
                 id="all_gather-axis0"),
    pytest.param("all_gather", "all_gather", {"axis": 1}, (4, 32),
                 id="all_gather-axis1"),
    pytest.param("reduce_scatter", "reduce_scatter", {"op": "sum", "axis": 0},
                 (1, 8), id="reduce_scatter-axis0"),
    pytest.param("all_to_all", "all_to_all",
                 {"split_axis": 0, "concat_axis": 1}, (1, 32),
                 id="all_to_all-01"),
]


@pytest.mark.parametrize("fn_name,op_name,kwargs,expected_shape", SHAPE_CASES)
def test_local_result_shapes(fn_name, op_name, kwargs, expected_shape):
    """Collective results carry LOCAL shapes (no global tensor type in the IR)."""
    fn = _collective_defn(fn_name, group=G4, **kwargs)
    _, op = _single_op(fn, op_name)
    assert op.results[0].type.shape == expected_shape


# --- 2. reduction kinds -----------------------------------------------------

def test_reductions_constant():
    assert REDUCTIONS == ("sum", "max", "min", "prod")


@pytest.mark.parametrize("op", ["sum", "max", "min", "prod"])
def test_all_reduce_records_reduce_op(op):
    fn = _collective_defn("all_reduce", op=op, group=G4)
    _, op_ir = _single_op(fn, "all_reduce")
    assert op_ir.attributes["reduce_op"] == op


@pytest.mark.parametrize("fn_name", ["all_reduce", "reduce_scatter"])
@pytest.mark.parametrize("op", ["mean", 3], ids=["str-mean", "int-3"])
def test_unknown_reduction_op_raises_value_error(fn_name, op):
    fn = _collective_defn(fn_name, op=op, group=G4)
    with pytest.raises(ValueError, match="unknown reduction op"):
        _trace(fn)


# --- 3. rank() / world_size() ----------------------------------------------

@pytest.mark.parametrize("fn_name", ["rank", "world_size"])
def test_execution_context_scalar_op(fn_name):
    """rank/world_size: 0-operand read-effect scalar int64 ops, group="world"."""
    scalar_fn = getattr(etl.dist, fn_name)

    @etl.defn
    def fn(x):
        return scalar_fn()

    block, op = _single_op(fn, fn_name)
    assert op.effect == "read"
    assert len(op.operands) == 0
    assert len(op.results) == 1
    assert op.attributes == {"group": "world"}
    assert op.results[0].type.shape == ()
    assert op.results[0].type.dtype == core.int64


def test_rank_world_size_return_symbolic_tensors():
    """rank()/world_size() are SymbolicTensor graph values, not Python ints."""
    captured = []

    @etl.defn
    def fn(x):
        r = etl.dist.rank()
        w = etl.dist.world_size()
        captured.extend((r, w))
        return r, w

    graph = _trace(fn)
    ops = {o.name: o for o in graph.module.functions[0].entry_block.ops}
    r, w = captured
    for scalar, name in ((r, "rank"), (w, "world_size")):
        assert isinstance(scalar, core.SymbolicTensor)
        assert not isinstance(scalar, int)
        assert scalar.shape == ()
        assert scalar.dtype == core.int64
        assert scalar.value is ops[name].results[0]


def test_rank_world_size_outside_trace_raise_trace_error():
    with pytest.raises(core.TraceError, match="No active trace"):
        etl.dist.rank()
    with pytest.raises(core.TraceError, match="No active trace"):
        etl.dist.world_size()


# --- 4. error matrix --------------------------------------------------------

#: Valid per-collective kwargs (shared by eager / outside-trace tests).
VALID_KWARGS = [
    pytest.param("all_reduce", {"op": "sum"}, id="all_reduce"),
    pytest.param("all_gather", {"axis": 0}, id="all_gather"),
    pytest.param("reduce_scatter", {"op": "sum", "axis": 0}, id="reduce_scatter"),
    pytest.param("all_to_all", {"split_axis": 0, "concat_axis": 1},
                 id="all_to_all"),
    pytest.param("broadcast", {"src_rank": 0}, id="broadcast"),
    pytest.param("collective_permute", {"source_target_pairs": ((0, 1),)},
                 id="collective_permute"),
]


@pytest.mark.parametrize("fn_name,kwargs", VALID_KWARGS)
def test_concrete_tensor_operand_raises_trace_error(fn_name, kwargs):
    """No eager mode: a concrete Tensor operand fails with a clear TraceError."""
    tensor = core.zeros((2, 2), core.float32)
    collective = getattr(etl.dist, fn_name)
    with pytest.raises(core.TraceError, match="no eager mode"):
        collective(tensor, group=G4, **kwargs)


@pytest.mark.parametrize("bad", [
    pytest.param(3, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param("x", id="str"),
    pytest.param([1, 2], id="list"),
    pytest.param(np.zeros(2), id="ndarray"),
])
def test_non_tensor_operand_raises_type_error(bad):
    with pytest.raises(TypeError, match="SymbolicTensor"):
        etl.dist.all_reduce(bad, "sum", G4)


@pytest.mark.parametrize("fn_name,kwargs", VALID_KWARGS)
def test_collective_outside_trace_raises_trace_error(fn_name, kwargs):
    """A SymbolicTensor captured during a trace is rejected after the trace."""
    captured = []

    @etl.defn
    def fn(x):
        y = etl.dist.all_reduce(x, "sum", G4)
        captured.append(y)
        return y

    _trace(fn)
    y = captured[0]
    collective = getattr(etl.dist, fn_name)
    with pytest.raises(core.TraceError, match="No active trace"):
        collective(y, group=G4, **kwargs)


AXIS_ERROR_CASES = [
    pytest.param("all_gather", {"axis": 1.5}, "expected an int axis",
                 id="all_gather-float-axis"),
    pytest.param("all_gather", {"axis": True}, "expected an int axis",
                 id="all_gather-bool-axis"),
    pytest.param("reduce_scatter", {"op": "sum", "axis": 1.5},
                 "expected an int axis", id="reduce_scatter-float-axis"),
    pytest.param("all_to_all", {"split_axis": 0, "concat_axis": True},
                 "expected an int axis", id="all_to_all-bool-axis"),
    pytest.param("all_gather", {"axis": 5}, "out of range",
                 id="all_gather-axis-oob"),
    pytest.param("reduce_scatter", {"op": "sum", "axis": 2}, "out of range",
                 id="reduce_scatter-axis-oob"),
    pytest.param("all_to_all", {"split_axis": 0, "concat_axis": 2},
                 "out of range", id="all_to_all-axis-oob"),
]


@pytest.mark.parametrize("fn_name,kwargs,match", AXIS_ERROR_CASES)
def test_axis_errors_raise_shape_error(fn_name, kwargs, match):
    """Non-int (bools rejected) and out-of-range axes raise ShapeError."""
    fn = _collective_defn(fn_name, group=G4, **kwargs)
    with pytest.raises(core.ShapeError, match=match):
        _trace(fn)


NEGATIVE_AXIS_CASES = [
    pytest.param("all_gather", {"axis": -1}, "all_gather", "axis", 1,
                 id="all_gather"),
    pytest.param("reduce_scatter", {"op": "sum", "axis": -1},
                 "reduce_scatter", "axis", 1, id="reduce_scatter"),
    pytest.param("all_to_all", {"split_axis": -2, "concat_axis": -1},
                 "all_to_all", "split_axis", 0, id="all_to_all-split"),
    pytest.param("all_to_all", {"split_axis": -2, "concat_axis": -1},
                 "all_to_all", "concat_axis", 1, id="all_to_all-concat"),
]


@pytest.mark.parametrize("fn_name,kwargs,op_name,attr,expected",
                         NEGATIVE_AXIS_CASES)
def test_negative_axes_normalized_before_recording(
    fn_name, kwargs, op_name, attr, expected
):
    """Negative axes wrap Python-style and are recorded non-negative."""
    fn = _collective_defn(fn_name, group=G4, **kwargs)
    _, op = _single_op(fn, op_name)
    assert op.attributes[attr] == expected


def test_reduce_scatter_indivisible_dim_raises_shape_error():
    fn = _collective_defn("reduce_scatter", op="sum", axis=0, group=G4)
    with pytest.raises(core.ShapeError, match="not divisible"):
        _trace(fn, shape=(5, 4))


def test_all_to_all_indivisible_dim_raises_shape_error():
    fn = _collective_defn("all_to_all", split_axis=0, concat_axis=1, group=G4)
    with pytest.raises(core.ShapeError, match="not divisible"):
        _trace(fn, shape=(5, 8))


def test_all_to_all_divisible_dim_traces():
    fn = _collective_defn("all_to_all", split_axis=0, concat_axis=1, group=G4)
    _, op = _single_op(fn, "all_to_all", shape=(8, 8))
    assert op.results[0].type.shape == (2, 32)


@pytest.mark.parametrize("src_rank", [
    pytest.param(-1, id="negative"),
    pytest.param(5, id="not-in-group"),
    pytest.param(True, id="bool"),
    pytest.param(1.5, id="float"),
])
def test_broadcast_bad_src_rank_raises_value_error(src_rank):
    fn = _collective_defn("broadcast", src_rank=src_rank, group=G4)
    with pytest.raises(ValueError, match="src_rank"):
        _trace(fn)


PERMUTE_BAD_CASES = [
    pytest.param((), id="empty"),
    pytest.param([(0, 1)], id="list-not-tuple"),
    pytest.param(((0,),), id="malformed-pair"),
    pytest.param(((0, 1), (0, 2)), id="duplicate-src"),
    pytest.param(((0, 1), (2, 1)), id="duplicate-dst"),
    pytest.param(((0, 9),), id="rank-outside-group"),
    pytest.param(((0, -1),), id="negative-rank"),
]


@pytest.mark.parametrize("pairs", PERMUTE_BAD_CASES)
def test_collective_permute_bad_pairs_raise_value_error(pairs):
    fn = _collective_defn(
        "collective_permute", source_target_pairs=pairs, group=G4
    )
    with pytest.raises(ValueError):
        _trace(fn)


@pytest.mark.parametrize("fn_name,kwargs", VALID_KWARGS)
def test_non_group_group_argument_raises_type_error(fn_name, kwargs):
    fn = _collective_defn(fn_name, group="data", **kwargs)
    with pytest.raises(TypeError, match="Group"):
        _trace(fn)


# --- 5. group=None selects the world group ---------------------------------

def test_world_group_default_attrs():
    """group=None → WORLD_GROUP: attrs group="world", group_size=None."""
    fn = _collective_defn("all_reduce", op="sum")
    _, op = _single_op(fn, "all_reduce")
    assert op.attributes == {"reduce_op": "sum", "group": "world",
                             "group_size": None}


def test_world_group_all_gather_runtime_dynamic_dim():
    """World-group all_gather: axis dim is None (runtime-dynamic)."""
    fn = _collective_defn("all_gather", axis=0)
    _, op = _single_op(fn, "all_gather")
    assert op.attributes == {"axis": 0, "group": "world", "group_size": None}
    assert op.results[0].type.shape == (None, 8)


# --- 6. broadcast builds broadcast_collective with src_rank defaulted ------

def test_broadcast_op_defaults_src_rank_to_zero():
    """The registry declares src_rank (default 0) and the builder fills it."""
    fn = _collective_defn("broadcast", group=G4)
    _, op = _single_op(fn, "broadcast_collective")
    assert op.attributes == {"group": "data", "src_rank": 0, "group_size": 4}
