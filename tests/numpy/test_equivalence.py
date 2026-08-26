"""The defining property of `etl.numpy` (enp): every enp function builds
EXACTLY the same IR as its documented `etl.ops` composition — same op
kinds, operands, attrs, and result types (see the mapping table in
`etl/numpy/CONTEXT.md`).

Each parametrized case builds two defns from identical TensorSpecs and
static values — one with enp, one with the documented ops composition —
traces both, `verify()`s both, and asserts the normalized pretty-printed
IR is character-for-character identical. The only permitted difference is
per-op ` loc(...)` tokens (call-site dependent: enp bodies live in
etl/numpy/*.py, ops bodies in this file), which are stripped per line.
"""

from __future__ import annotations

import functools
import operator

import numpy as np
import pytest

import etl
import etl.numpy as enp
from etl import ops
from tests.numpy._ir_utils import normalize_ir

F32 = etl.float32
F64 = etl.float64
I32 = etl.int32
BOOL = etl.bool_

# --- helpers ----------------------------------------------------------------


def _spec(shape, dtype=F32):
    return etl.TensorSpec(shape, dtype)


def _ir(body, *specs):
    """Trace `body` (a plain function) via etl.defn, verify, pretty-print
    into normalized SSA text (the entry func is always named `main`)."""
    graph = etl.trace(etl.defn(body), *specs)
    graph.verify()
    return normalize_ir(etl.ir.pretty_print(graph.module))


def _assert_same_ir(enp_body, ops_body, *specs):
    enp_ir = _ir(enp_body, *specs)
    ops_ir = _ir(ops_body, *specs)
    # assert a == b so pytest renders a full diff on mismatch
    assert enp_ir == ops_ir


# --- documented ops compositions (mirrors of etl/numpy internals) -----------
# These are the contract from etl/numpy/CONTEXT.md: composed enp functions
# must produce exactly this op sequence.


def _expand_dims_ops(a, axis):
    rank = len(a.shape)
    if isinstance(axis, tuple):
        axes = sorted(ax + rank + 1 if ax < 0 else ax for ax in axis)
    else:
        axes = [axis + rank + 1 if axis < 0 else axis]
    new_shape = list(a.shape)
    for ax in axes:
        new_shape.insert(ax, 1)
    return ops.reshape(a, tuple(new_shape))


def _squeeze_ops(a, axis=None):
    shape = tuple(a.shape)
    if axis is None:
        new_shape = tuple(d for d in shape if not (isinstance(d, int) and d == 1))
    else:
        new_shape = shape[:axis] + shape[axis + 1:]
    return ops.reshape(a, new_shape)


def _stack_ops(arrays, axis=0):
    arrays = list(arrays)
    rank = len(arrays[0].shape)
    if axis < 0:
        axis += rank + 1
    expanded = [
        ops.reshape(x, x.shape[:axis] + (1,) + x.shape[axis:]) for x in arrays
    ]
    return ops.concatenate(expanded, axis=axis)


def _split_ops(a, n, axis=0):
    shape = tuple(a.shape)
    rank = len(shape)
    size = shape[axis]
    boundaries = [i * (size // n) for i in range(n + 1)]
    result = []
    for i in range(len(boundaries) - 1):
        start = tuple(boundaries[i] if j == axis else 0 for j in range(rank))
        lengths = tuple(
            (boundaries[i + 1] - boundaries[i]) if j == axis else d
            for j, d in enumerate(shape)
        )
        result.append(ops.slice(a, start, lengths))
    return result


def _cumsum_flatten_ops(a):
    """cumsum(axis=None): reshape to (numel,) then cumsum(axis=0)."""
    numel = functools.reduce(operator.mul, a.shape, 1)
    return ops.cumsum(ops.reshape(a, (numel,)), axis=0)


# --- case tables ------------------------------------------------------------
# Each entry: pytest.param(enp_body, ops_body, [specs...], id=...) where the
# bodies are plain functions over SymbolicTensors (static Python values are
# captured identically on both sides).

ELEMENTWISE_CASES = [
    pytest.param(lambda x: enp.abs(x), lambda x: ops.abs(x),
                 [_spec((2, 3))], id="abs"),
    pytest.param(lambda x, y: enp.add(x, y), lambda x, y: ops.add(x, y),
                 [_spec((2, 3)), _spec((3,))], id="add"),
    pytest.param(lambda x, y: enp.subtract(x, y), lambda x, y: ops.subtract(x, y),
                 [_spec((2, 3)), _spec((3,))], id="subtract"),
    pytest.param(lambda x, y: enp.multiply(x, y), lambda x, y: ops.multiply(x, y),
                 [_spec((2, 3)), _spec((3,))], id="multiply"),
    pytest.param(lambda x, y: enp.divide(x, y), lambda x, y: ops.divide(x, y),
                 [_spec((2, 3)), _spec((3,))], id="divide"),
    pytest.param(lambda x, y: enp.power(x, y), lambda x, y: ops.power(x, y),
                 [_spec((2, 3)), _spec((3,))], id="power"),
    pytest.param(lambda x, y: enp.maximum(x, y), lambda x, y: ops.maximum(x, y),
                 [_spec((2, 3)), _spec((3,))], id="maximum"),
    pytest.param(lambda x, y: enp.minimum(x, y), lambda x, y: ops.minimum(x, y),
                 [_spec((2, 3)), _spec((3,))], id="minimum"),
    pytest.param(lambda x: enp.negative(x), lambda x: ops.negate(x),
                 [_spec((2, 3))], id="negative"),
    pytest.param(lambda x: enp.square(x), lambda x: ops.square(x),
                 [_spec((2, 3))], id="square"),
    pytest.param(lambda x: enp.sqrt(x), lambda x: ops.sqrt(x),
                 [_spec((2, 3))], id="sqrt"),
    pytest.param(lambda x: enp.exp(x), lambda x: ops.exp(x),
                 [_spec((2, 3))], id="exp"),
    pytest.param(lambda x: enp.log(x), lambda x: ops.log(x),
                 [_spec((2, 3))], id="log"),
    pytest.param(lambda x: enp.sin(x), lambda x: ops.sin(x),
                 [_spec((2, 3))], id="sin"),
    pytest.param(lambda x: enp.cos(x), lambda x: ops.cos(x),
                 [_spec((2, 3))], id="cos"),
    pytest.param(lambda x: enp.tanh(x), lambda x: ops.tanh(x),
                 [_spec((2, 3))], id="tanh"),
    pytest.param(lambda x: enp.sign(x), lambda x: ops.sign(x),
                 [_spec((2, 3))], id="sign"),
    pytest.param(lambda x: enp.astype(x, F64), lambda x: ops.cast(x, F64),
                 [_spec((2, 3))], id="astype"),
    # clip: composed ops.maximum(ops.minimum(a, a_max), a_min)
    pytest.param(lambda x: enp.clip(x, 0.1, 0.9),
                 lambda x: ops.maximum(ops.minimum(x, 0.9), 0.1),
                 [_spec((2, 3))], id="clip"),
    pytest.param(lambda x: enp.clip(x, None, 0.9),
                 lambda x: ops.maximum(x, 0.9),
                 [_spec((2, 3))], id="clip-upper-only"),
    pytest.param(lambda x: enp.clip(x, 0.1, None),
                 lambda x: ops.minimum(x, 0.1),
                 [_spec((2, 3))], id="clip-lower-only"),
]

LOGIC_CASES = [
    pytest.param(lambda x, y: enp.equal(x, y), lambda x, y: ops.equal(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="equal"),
    pytest.param(lambda x, y: enp.not_equal(x, y),
                 lambda x, y: ops.not_equal(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="not_equal"),
    pytest.param(lambda x, y: enp.less(x, y), lambda x, y: ops.less(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="less"),
    pytest.param(lambda x, y: enp.less_equal(x, y),
                 lambda x, y: ops.less_equal(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="less_equal"),
    pytest.param(lambda x, y: enp.greater(x, y), lambda x, y: ops.greater(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="greater"),
    pytest.param(lambda x, y: enp.greater_equal(x, y),
                 lambda x, y: ops.greater_equal(x, y),
                 [_spec((2, 3), I32), _spec((3,), I32)], id="greater_equal"),
    pytest.param(lambda x, y: enp.logical_and(x, y),
                 lambda x, y: ops.logical_and(x, y),
                 [_spec((2, 3), BOOL), _spec((3,), BOOL)], id="logical_and"),
    pytest.param(lambda x, y: enp.logical_or(x, y),
                 lambda x, y: ops.logical_or(x, y),
                 [_spec((2, 3), BOOL), _spec((3,), BOOL)], id="logical_or"),
    pytest.param(lambda x: enp.logical_not(x), lambda x: ops.logical_not(x),
                 [_spec((2, 3), BOOL)], id="logical_not"),
    # where(cond, x, y) → ops.select(cond, x, y)
    pytest.param(lambda c, x, y: enp.where(c, x, y),
                 lambda c, x, y: ops.select(c, x, y),
                 [_spec((2, 3), BOOL), _spec((2, 3)), _spec((2, 3))],
                 id="where"),
]

REDUCTION_CASES = [
    # sum/mean/prod/max/min: axis=None → axes=None (all axes)
    pytest.param(lambda x: enp.sum(x),
                 lambda x: ops.sum(x, axes=None, keepdims=False),
                 [_spec((2, 3))], id="sum"),
    pytest.param(lambda x: enp.sum(x, axis=1, keepdims=True),
                 lambda x: ops.sum(x, axes=1, keepdims=True),
                 [_spec((2, 3))], id="sum-axis1-keepdims"),
    # dtype≠None composes ops.cast AFTER the reduction
    pytest.param(lambda x: enp.sum(x, dtype=F64),
                 lambda x: ops.cast(ops.sum(x, axes=None, keepdims=False), F64),
                 [_spec((2, 3))], id="sum-dtype"),
    pytest.param(lambda x: enp.mean(x),
                 lambda x: ops.mean(x, axes=None, keepdims=False),
                 [_spec((2, 3))], id="mean"),
    pytest.param(lambda x: enp.mean(x, dtype=F64),
                 lambda x: ops.cast(ops.mean(x, axes=None, keepdims=False), F64),
                 [_spec((2, 3))], id="mean-dtype"),
    pytest.param(lambda x: enp.prod(x),
                 lambda x: ops.prod(x, axes=None, keepdims=False),
                 [_spec((2, 3))], id="prod"),
    pytest.param(lambda x: enp.prod(x, axis=0),
                 lambda x: ops.prod(x, axes=0, keepdims=False),
                 [_spec((2, 3))], id="prod-axis0"),
    pytest.param(lambda x: enp.max(x),
                 lambda x: ops.max(x, axes=None, keepdims=False),
                 [_spec((2, 3))], id="max"),
    pytest.param(lambda x: enp.min(x, axis=1),
                 lambda x: ops.min(x, axes=1, keepdims=False),
                 [_spec((2, 3))], id="min-axis1"),
    pytest.param(lambda x: enp.argmax(x),
                 lambda x: ops.argmax(x, axis=None, keepdims=False),
                 [_spec((2, 3))], id="argmax"),
    pytest.param(lambda x: enp.argmin(x, axis=1, keepdims=True),
                 lambda x: ops.argmin(x, axis=1, keepdims=True),
                 [_spec((2, 3))], id="argmin-axis1-keepdims"),
    pytest.param(lambda x: enp.cumsum(x, axis=1),
                 lambda x: ops.cumsum(x, axis=1),
                 [_spec((2, 3))], id="cumsum-axis1"),
    # cumsum(axis=None): flatten via reshape((numel,)) then cumsum(axis=0)
    pytest.param(lambda x: enp.cumsum(x), _cumsum_flatten_ops,
                 [_spec((2, 3))], id="cumsum-flatten"),
    pytest.param(lambda x: enp.cumsum(x, dtype=F64),
                 lambda x: ops.cast(_cumsum_flatten_ops(x), F64),
                 [_spec((2, 3))], id="cumsum-flatten-dtype"),
]

SHAPE_CASES = [
    pytest.param(lambda x: enp.reshape(x, (3, 2)),
                 lambda x: ops.reshape(x, (3, 2)),
                 [_spec((2, 3))], id="reshape"),
    pytest.param(lambda x: enp.reshape(x, (-1, 6)),
                 lambda x: ops.reshape(x, (-1, 6)),
                 [_spec((2, 3))], id="reshape-minus-one"),
    pytest.param(lambda x: enp.transpose(x),
                 lambda x: ops.transpose(x, None),
                 [_spec((2, 3))], id="transpose-default"),
    pytest.param(lambda x: enp.transpose(x, (1, 0)),
                 lambda x: ops.transpose(x, (1, 0)),
                 [_spec((2, 3))], id="transpose-axes"),
    pytest.param(lambda x: enp.broadcast_to(x, (2, 3)),
                 lambda x: ops.broadcast(x, (2, 3)),
                 [_spec((3,))], id="broadcast_to"),
    # expand_dims: reshape with size-1 dims inserted (int/tuple forms)
    pytest.param(lambda x: enp.expand_dims(x, 1),
                 lambda x: _expand_dims_ops(x, 1),
                 [_spec((2, 3))], id="expand_dims-int"),
    pytest.param(lambda x: enp.expand_dims(x, -1),
                 lambda x: _expand_dims_ops(x, -1),
                 [_spec((2, 3))], id="expand_dims-negative"),
    pytest.param(lambda x: enp.expand_dims(x, (0, 2)),
                 lambda x: _expand_dims_ops(x, (0, 2)),
                 [_spec((2, 3))], id="expand_dims-tuple"),
    # squeeze: reshape with statically-1 dims dropped
    pytest.param(lambda x: enp.squeeze(x),
                 lambda x: _squeeze_ops(x),
                 [_spec((2, 1, 3))], id="squeeze-all"),
    pytest.param(lambda x: enp.squeeze(x, axis=1),
                 lambda x: _squeeze_ops(x, axis=1),
                 [_spec((2, 1, 3))], id="squeeze-axis"),
    pytest.param(lambda x, y: enp.concatenate([x, y]),
                 lambda x, y: ops.concatenate([x, y], axis=0),
                 [_spec((2, 3)), _spec((2, 3))], id="concatenate"),
    pytest.param(lambda x, y: enp.concatenate([x, y], axis=1),
                 lambda x, y: ops.concatenate([x, y], axis=1),
                 [_spec((2, 3)), _spec((2, 3))], id="concatenate-axis1"),
    # stack: expand_dims(each, axis) + concatenate(axis)
    pytest.param(lambda x, y: enp.stack([x, y]),
                 lambda x, y: _stack_ops([x, y]),
                 [_spec((2, 3)), _spec((2, 3))], id="stack"),
    pytest.param(lambda x, y: enp.stack([x, y], axis=1),
                 lambda x, y: _stack_ops([x, y], axis=1),
                 [_spec((2, 3)), _spec((2, 3))], id="stack-axis1"),
    # split (int sections): composition of ops.slice
    pytest.param(lambda x: enp.split(x, 2),
                 lambda x: _split_ops(x, 2),
                 [_spec((4, 2))], id="split-two"),
    pytest.param(lambda x: enp.split(x, 2, axis=1),
                 lambda x: _split_ops(x, 2, axis=1),
                 [_spec((2, 4))], id="split-two-axis1"),
    # pad (constant mode): int form → per-axis symmetric config
    pytest.param(lambda x: enp.pad(x, 2),
                 lambda x: ops.pad(x, (2, 2), value=0),
                 [_spec((2, 3))], id="pad-int"),
    pytest.param(lambda x: enp.pad(x, ((1, 0), (0, 2))),
                 lambda x: ops.pad(x, ((1, 0), (0, 2)), value=0),
                 [_spec((2, 3))], id="pad-per-axis"),
    pytest.param(lambda x: enp.tril(x), lambda x: ops.tril(x, k=0),
                 [_spec((3, 3))], id="tril"),
    pytest.param(lambda x: enp.tril(x, k=1), lambda x: ops.tril(x, k=1),
                 [_spec((3, 3))], id="tril-k1"),
    pytest.param(lambda x: enp.triu(x), lambda x: ops.triu(x, k=0),
                 [_spec((3, 3))], id="triu"),
]

CREATION_CASES = [
    # Creation is graph Constant ops inside defn — same op kind as
    # etl.constant, never concrete tensors. Constant attrs print as
    # ndarray<dtype[shape]> summaries, so equal dtype/shape ⇒ equal text.
    pytest.param(lambda: enp.zeros((2, 3)),
                 lambda: ops.constant(etl.tensor(np.zeros((2, 3), dtype=np.float32))),
                 [], id="zeros"),
    pytest.param(lambda: enp.zeros(5),
                 lambda: ops.constant(etl.tensor(np.zeros(5, dtype=np.float32))),
                 [], id="zeros-int-shape"),
    pytest.param(lambda: enp.ones((2, 3)),
                 lambda: ops.constant(etl.tensor(np.ones((2, 3), dtype=np.float32))),
                 [], id="ones"),
    pytest.param(lambda: enp.ones((2, 3), dtype=F64),
                 lambda: ops.constant(etl.tensor(np.ones((2, 3), dtype=np.float64))),
                 [], id="ones-dtype"),
    # full with dtype=None → np.result_type(fill_value) inference (7 → int64)
    pytest.param(lambda: enp.full((2, 2), 7),
                 lambda: ops.constant(
                     etl.tensor(np.full((2, 2), 7, dtype=np.result_type(7)))
                 ),
                 [], id="full-inferred"),
    pytest.param(lambda: enp.full((2, 3), 0.5, dtype=F32),
                 lambda: ops.constant(
                     etl.tensor(np.full((2, 3), 0.5, dtype=np.float32))
                 ),
                 [], id="full-dtype"),
    # empty: values unspecified (numpy semantics); dtype defaults float32
    pytest.param(lambda: enp.empty((2, 3)),
                 lambda: ops.constant(etl.tensor(np.empty((2, 3), dtype=np.float32))),
                 [], id="empty"),
    pytest.param(lambda: enp.arange(6),
                 lambda: ops.constant(etl.tensor(np.arange(6))),
                 [], id="arange-stop"),
    pytest.param(lambda: enp.arange(2, 9, 3, dtype=I32),
                 lambda: ops.constant(
                     etl.tensor(np.arange(2, 9, 3, dtype=np.int32))
                 ),
                 [], id="arange-full"),
    pytest.param(lambda: enp.arange(0.0, 1.0, 0.25),
                 lambda: ops.constant(etl.tensor(np.arange(0.0, 1.0, 0.25))),
                 [], id="arange-float"),
]

LINALG_CASES = [
    pytest.param(lambda a, b: enp.matmul(a, b), lambda a, b: ops.dot(a, b),
                 [_spec((2, 3)), _spec((3, 4))], id="matmul"),
    pytest.param(lambda a, b: enp.dot(a, b), lambda a, b: ops.dot(a, b),
                 [_spec((3, 3)), _spec((3, 3))], id="dot"),
    pytest.param(lambda a, b: enp.linalg.solve(a, b),
                 lambda a, b: ops.solve(a, b),
                 [_spec((3, 3)), _spec((3,))], id="linalg-solve"),
]


# --- equivalence tests ------------------------------------------------------


@pytest.mark.parametrize("enp_body,ops_body,specs", ELEMENTWISE_CASES)
def test_elementwise_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


@pytest.mark.parametrize("enp_body,ops_body,specs", LOGIC_CASES)
def test_logic_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


@pytest.mark.parametrize("enp_body,ops_body,specs", REDUCTION_CASES)
def test_reduction_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


@pytest.mark.parametrize("enp_body,ops_body,specs", SHAPE_CASES)
def test_shape_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


@pytest.mark.parametrize("enp_body,ops_body,specs", CREATION_CASES)
def test_creation_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


@pytest.mark.parametrize("enp_body,ops_body,specs", LINALG_CASES)
def test_linalg_equivalence(enp_body, ops_body, specs):
    _assert_same_ir(enp_body, ops_body, *specs)


# --- runtime spot checks ----------------------------------------------------
# enp is sugar: it must also run — etl.evaluate on an enp defn behaves like
# the ops composition and matches numpy.


def test_evaluate_runs_enp_defn_numerically():
    """etl.evaluate runs an enp defn end-to-end; result is an etl.Tensor
    matching the numpy reference."""
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    @etl.defn
    def f(v):
        return enp.add(enp.sqrt(v), enp.ones((2, 3)))

    out = etl.evaluate(f, x)
    assert isinstance(out, etl.Tensor)
    np.testing.assert_allclose(
        out.numpy(), np.sqrt(x) + np.ones((2, 3), dtype=np.float32)
    )


def test_evaluate_python_scalar_raises_type_error():
    """evaluate arguments must be tensors/ndarrays — Python scalars raise
    TypeError (no silent conversion)."""

    @etl.defn
    def f(v):
        return enp.add(v, 1.0)

    with pytest.raises(TypeError, match="TypeError"):
        etl.evaluate(f, 1.0)


def test_enp_and_ops_defns_produce_equal_numbers():
    """Numeric spot check: the enp graph and its documented ops composition
    run to the same values."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    @etl.defn
    def f_enp(a):
        return enp.clip(a, 1.5, 3.5)

    @etl.defn
    def f_ops(a):
        return ops.maximum(ops.minimum(a, 3.5), 1.5)

    r_enp = etl.evaluate(f_enp, x)
    r_ops = etl.evaluate(f_ops, x)
    np.testing.assert_array_equal(r_enp.numpy(), r_ops.numpy())
    np.testing.assert_array_equal(r_enp.numpy(), np.clip(x, 1.5, 3.5))
