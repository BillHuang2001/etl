"""Contract tests for etl.ops.reductions.

Covers the ``reduce_*`` family (``reduce_sum`` / ``reduce_max`` /
``reduce_min`` / ``reduce_mean`` / ``reduce_prod``) and the user-facing sugar
(``sum`` / ``max`` / ``min`` / ``mean`` / ``prod`` — documented shorthand
that expands EXACTLY onto the corresponding ``reduce_*`` op). The etl package
(repo-root sibling) is fully implemented; these tests assert the contracts in
``etl/ops/reductions.py`` and ``etl/ops/CONTEXT.md``:

- ``axes`` is STATIC: None (all axes), int, or tuple of ints; negatives are
  shifted, duplicates deduped, result sorted.
- Output shape via reduced_shape: axes removed (or kept as 1 with
  ``keepdims``); reducing all axes without keepdims → scalar ``()``; symbolic
  dims on unreduced axes are preserved.
- Dtype rules (numpy): sum/prod promote bool → int64 and signed ints → int64
  / uints → uint64; floats/complex keep dtype; mean promotes integer/bool →
  float64; max/min preserve the dtype.
- Numerics equal numpy's np.sum/np.max/np.min/np.mean/np.prod.
- Sugar functions produce the SAME IR (op names, attributes, result types) as
  their reduce_* counterparts.
"""
from __future__ import annotations

import numpy as np
import pytest

import etl
from tests.ops.conftest import ops_of, run_numpy, trace_fn


REDUCE_NAMES = ["reduce_sum", "reduce_max", "reduce_min", "reduce_mean",
                "reduce_prod"]
SUGAR_PAIRS = [
    ("sum", "reduce_sum"),
    ("max", "reduce_max"),
    ("min", "reduce_min"),
    ("mean", "reduce_mean"),
    ("prod", "reduce_prod"),
]


def _op(graph, name):
    ops = ops_of(graph, name)
    assert len(ops) == 1, f"expected exactly one {name} op, got {len(ops)}"
    return ops[0]


def _trace_capturing(fn, *specs):
    """Trace ``fn`` and return ``(graph, returned_symbolic_tensor)``."""
    captured = {}

    def wrapped(*args):
        out = fn(*args)
        captured["out"] = out
        return out

    graph = etl.trace(wrapped, *specs)
    return graph, captured["out"]


# ---------------------------------------------------------------------------
# shape inference
# ---------------------------------------------------------------------------

REDUCE_SHAPE_CASES = [
    # axes, keepdims, out shape for input (2, 3, 4)
    (None, False, ()),
    (None, True, (1, 1, 1)),
    (0, False, (3, 4)),
    (-1, False, (2, 3)),
    (-2, True, (2, 1, 4)),
    ((0, 2), False, (3,)),
    ((0, 2), True, (1, 3, 1)),
    ((1,), True, (2, 1, 4)),
    ((2, 0), False, (3,)),          # normalized: sorted, deduped
    ((1, 1), False, (2, 4)),        # duplicate axes deduped (no-op)
]


@pytest.mark.parametrize("op_name", REDUCE_NAMES)
@pytest.mark.parametrize("axes,keepdims,out_shape", REDUCE_SHAPE_CASES)
def test_reduce_shape_inference(op_name, axes, keepdims, out_shape):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=axes, keepdims=keepdims)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3, 4), etl.float32))
    assert isinstance(out, etl.SymbolicTensor)
    assert tuple(out.shape) == out_shape
    assert _op(graph, op_name).results[0].type.shape == out_shape


def test_reduce_full_reduction_without_keepdims_is_scalar():
    def f(x):
        return etl.reduce_sum(x)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), etl.float32))
    assert tuple(out.shape) == ()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = run_numpy(f, x)
    assert got.shape == ()
    assert got == np.float32(x.sum())


@pytest.mark.parametrize("op_name", REDUCE_NAMES)
def test_reduce_preserves_symbolic_dims(op_name):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=(0, 2), keepdims=True)

    graph, out = _trace_capturing(
        f, etl.TensorSpec((etl.dim("n"), 3, etl.dim("m")), etl.float32)
    )
    assert tuple(out.shape) == (1, 3, 1)
    assert _op(graph, op_name).results[0].type.shape == (1, 3, 1)


def test_reduce_symbolic_axis_removed():
    """Reducing over a symbolic dim consumes it; unreduced symbolic dims stay."""
    def f(x):
        return etl.reduce_sum(x, axes=1)

    graph, out = _trace_capturing(
        f, etl.TensorSpec((etl.dim("n"), 3, etl.dim("m")), etl.float32)
    )
    shape = out.shape
    assert len(shape) == 2
    assert isinstance(shape[0], etl.Dim) and isinstance(shape[1], etl.Dim)
    assert shape[0] == etl.dim("n") and shape[1] == etl.dim("m")


def test_reduce_normalized_attrs_are_sorted_and_deduped():
    def f(x):
        return etl.reduce_sum(x, axes=(2, -1, 0, 2))

    graph = trace_fn(f, etl.TensorSpec((2, 3, 4), etl.float32))
    op = _op(graph, "reduce_sum")
    assert op.attributes["axes"] == (0, 2)
    assert op.attributes["keepdims"] is False
    assert op.attributes["reduce_op"] == "sum"


# ---------------------------------------------------------------------------
# dtype rules
# ---------------------------------------------------------------------------

REDUCE_DTYPE_CASES = [
    ("reduce_sum", etl.int8, np.int64),
    ("reduce_sum", etl.uint8, np.uint64),
    ("reduce_sum", etl.bool_, np.int64),
    ("reduce_sum", etl.float32, np.float32),
    ("reduce_sum", etl.float16, np.float16),
    ("reduce_sum", etl.complex64, np.complex64),
    ("reduce_prod", etl.int32, np.int64),
    ("reduce_prod", etl.uint16, np.uint64),
    ("reduce_prod", etl.bool_, np.int64),
    ("reduce_prod", etl.float32, np.float32),
    ("reduce_mean", etl.int32, np.float64),
    ("reduce_mean", etl.uint8, np.float64),
    ("reduce_mean", etl.bool_, np.float64),
    ("reduce_mean", etl.float32, np.float32),
    ("reduce_mean", etl.float64, np.float64),
    ("reduce_max", etl.int32, np.int32),
    ("reduce_max", etl.bool_, np.bool_),
    ("reduce_max", etl.float32, np.float32),
    ("reduce_min", etl.int16, np.int16),
    ("reduce_min", etl.float64, np.float64),
]


@pytest.mark.parametrize("op_name,in_dtype,out_dtype", REDUCE_DTYPE_CASES)
def test_reduce_dtype_rules(op_name, in_dtype, out_dtype):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=0)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), in_dtype))
    assert out.dtype == np.dtype(out_dtype)
    assert _op(graph, op_name).results[0].type.dtype == np.dtype(out_dtype)


def test_reduce_dtype_rules_match_numpy_promotion():
    """The declared rules coincide with numpy's reducer promotion for the
    supported dtype kinds (max/min preserve, floats keep, ints promote)."""
    for in_dtype, op_name, np_reducer in [
        (np.int8, "reduce_sum", np.sum),
        (np.uint8, "reduce_prod", np.prod),
        (np.int32, "reduce_mean", np.mean),
        (np.bool_, "reduce_sum", np.sum),
        (np.float32, "reduce_max", np.max),
    ]:
        x = np.ones((2, 3), dtype=in_dtype)
        got = run_numpy(lambda x: getattr(etl, op_name)(x), x)
        assert got.dtype == np_reducer(x).dtype


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name,np_reducer", [
    ("reduce_sum", np.sum),
    ("reduce_max", np.max),
    ("reduce_min", np.min),
    ("reduce_mean", np.mean),
    ("reduce_prod", np.prod),
])
@pytest.mark.parametrize("axes,keepdims", [
    (None, False),
    (0, False),
    (-1, False),
    ((0, 2), False),
    ((0, 2), True),
    (1, True),
])
def test_reduce_numerics_vs_numpy_float(op_name, np_reducer, axes, keepdims):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=axes, keepdims=keepdims)

    rng = np.random.default_rng(8)
    x = rng.uniform(0.5, 1.5, size=(2, 3, 4))  # prod stays finite & exact-ish
    got = run_numpy(f, x)
    expected = np_reducer(x, axis=axes, keepdims=keepdims)
    assert got.shape == np.asarray(expected).shape
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("op_name,np_reducer", [
    ("reduce_sum", np.sum),
    ("reduce_max", np.max),
    ("reduce_min", np.min),
    ("reduce_mean", np.mean),
    ("reduce_prod", np.prod),
])
def test_reduce_numerics_vs_numpy_int(op_name, np_reducer):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=(0, 2), keepdims=True)

    rng = np.random.default_rng(9)
    x = rng.integers(1, 4, size=(2, 3, 4)).astype(np.int32)
    got = run_numpy(f, x)
    expected = np_reducer(x, axis=(0, 2), keepdims=True)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_reduce_numerics_negative_axes_match_numpy():
    def f(x):
        return etl.reduce_sum(x, axes=-2)

    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.testing.assert_array_equal(run_numpy(f, x), np.sum(x, axis=-2))


# ---------------------------------------------------------------------------
# rank-0 inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name,in_dtype,out_dtype", [
    ("reduce_sum", etl.bool_, np.int64),
    ("reduce_sum", etl.float32, np.float32),
    ("reduce_mean", etl.int32, np.float64),
    ("reduce_max", etl.int32, np.int32),
    ("reduce_prod", etl.bool_, np.int64),
])
def test_reduce_rank0_applies_dtype_rule(op_name, in_dtype, out_dtype):
    """A scalar reduction applies the op's dtype rule and keeps shape ()
    (there are no axes to remove)."""
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axes=(), keepdims=True)

    graph, out = _trace_capturing(f, etl.TensorSpec((), in_dtype))
    assert tuple(out.shape) == ()
    assert out.dtype == np.dtype(out_dtype)
    got = run_numpy(f, np.asarray(True if in_dtype == etl.bool_ else 3, dtype=np.dtype(in_dtype)))
    assert got.shape == ()
    assert got.dtype == np.dtype(out_dtype)


# ---------------------------------------------------------------------------
# sugar equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sugar,reduce", SUGAR_PAIRS)
def test_sugar_produces_identical_ir(sugar, reduce):
    """sum/max/min/mean/prod expand EXACTLY onto reduce_* (same op name,
    attributes, result type) — no hidden semantics."""
    sugar_fn = getattr(etl, sugar)
    reduce_fn = getattr(etl, reduce)

    def f_sugar(x):
        return sugar_fn(x, axes=(0, 2), keepdims=True)

    def f_reduce(x):
        return reduce_fn(x, axes=(0, 2), keepdims=True)

    g_sugar = trace_fn(f_sugar, etl.TensorSpec((2, 3, 4), etl.float32))
    g_reduce = trace_fn(f_reduce, etl.TensorSpec((2, 3, 4), etl.float32))
    ops_sugar = [op for op in ops_of(g_sugar) if op.name != "return"]
    ops_reduce = [op for op in ops_of(g_reduce) if op.name != "return"]
    assert [op.name for op in ops_sugar] == [reduce]
    assert [op.name for op in ops_reduce] == [reduce]
    assert ops_sugar[0].attributes == ops_reduce[0].attributes
    r_sugar, r_reduce = ops_sugar[0].results[0], ops_reduce[0].results[0]
    assert r_sugar.type.dtype == r_reduce.type.dtype
    assert r_sugar.type.shape == r_reduce.type.shape == (1, 3, 1)


@pytest.mark.parametrize("sugar,reduce", SUGAR_PAIRS)
def test_sugar_and_reduce_agree_numerically(sugar, reduce):
    sugar_fn = getattr(etl, sugar)
    reduce_fn = getattr(etl, reduce)
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    got_sugar = run_numpy(lambda t: sugar_fn(t, axes=1), x)
    got_reduce = run_numpy(lambda t: reduce_fn(t, axes=1), x)
    np.testing.assert_array_equal(got_sugar, got_reduce)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", REDUCE_NAMES)
def test_reduce_axes_out_of_range(op_name):
    op = getattr(etl, op_name)
    with pytest.raises(etl.ShapeError, match="axis 3 out of range for rank 2"):
        etl.trace(lambda x: op(x, axes=3), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="axis -3 out of range for rank 2"):
        etl.trace(lambda x: op(x, axes=-3), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="axis 2 out of range for rank 2"):
        etl.trace(lambda x: op(x, axes=(0, 2)),
                  etl.TensorSpec((2, 3), etl.float32))


def test_reduce_empty_axes_rejected_on_rank1_plus():
    """axes=() would silently mean "all axes" at the IR level — the frontend
    rejects it explicitly on rank >= 1 tensors."""
    with pytest.raises(etl.ShapeError, match="reducing no axes"):
        etl.trace(lambda x: etl.reduce_sum(x, axes=()),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="reducing no axes"):
        etl.trace(lambda x: etl.reduce_mean(x, axes=()),
                  etl.TensorSpec((2, 3), etl.float32))


@pytest.mark.parametrize("bad_axes,msg", [
    (1.5, "axes must be None, an int, or a tuple of ints, got float"),
    ("x", "axes must be None, an int, or a tuple of ints, got str"),
    (True, "axes must be None, an int, or a tuple of ints, got bool"),
    ((0, 1.5), "axes must be an int or a tuple of ints, got element 1.5"),
    ([0, 1], "axes must be None, an int, or a tuple of ints, got list"),
])
def test_reduce_malformed_axes_raise_type_error(bad_axes, msg):
    with pytest.raises(TypeError, match=msg):
        etl.trace(lambda x: etl.reduce_sum(x, axes=bad_axes),
                  etl.TensorSpec((2, 3), etl.float32))


@pytest.mark.parametrize("op_name", REDUCE_NAMES)
def test_reduce_keepdims_must_be_bool(op_name):
    op = getattr(etl, op_name)
    with pytest.raises(TypeError,
                       match=f"{op_name}: keepdims must be a bool, got 1"):
        etl.trace(lambda x: op(x, axes=0, keepdims=1),
                  etl.TensorSpec((2, 3), etl.float32))


@pytest.mark.parametrize("sugar", ["sum", "max", "min", "mean", "prod"])
def test_sugar_errors_match_reduce_errors(sugar):
    sugar_fn = getattr(etl, sugar)
    with pytest.raises(etl.ShapeError, match="axis 5 out of range for rank 2"):
        etl.trace(lambda x: sugar_fn(x, axes=5), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError, match="keepdims must be a bool"):
        etl.trace(lambda x: sugar_fn(x, axes=None, keepdims=0),
                  etl.TensorSpec((2, 3), etl.float32))
