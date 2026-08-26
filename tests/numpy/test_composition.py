"""Composition-semantics tests for `etl.numpy` (enp).

The enp functions whose numpy semantics are richer than a single op
(`clip`, `stack`, `split`, `expand_dims`, `squeeze`, `where`, `pad`) must
compose the exact documented op sequences from `etl/numpy/CONTEXT.md`.
These tests check two things per function: numeric equivalence with numpy
(via `etl.evaluate`) and IR equivalence with the documented composition
(via `etl.trace` + `etl.ir.pretty_print`, locations stripped).
"""

import re

import numpy as np
import pytest

import etl
import etl.ir
import etl.numpy as enp

F32 = etl.float32


# --- file-local helpers -----------------------------------------------------


def _strip_loc(text):
    """Strip the trailing ``loc("file":line:col)`` from each pretty-print line.

    Locations record the Python call site, which necessarily differs between
    two traced defns — everything else must be identical.
    """
    return "\n".join(
        re.sub(r"\s+loc\(.*?\)\s*$", "", line) for line in text.splitlines()
    )


def _op_names(text):
    """The op-name sequence of a pretty-printed module, in program order."""
    return re.findall(r"etl\.(\w+)\(", text)


def _normalized_module(defn_fn, *specs):
    """Trace ``defn_fn`` and return its location-stripped pretty print."""
    graph = etl.trace(defn_fn, *specs)
    graph.verify()
    return _strip_loc(etl.ir.pretty_print(graph.module))


# --- clip -------------------------------------------------------------------


def _clip_defn(a_min, a_max):
    @etl.defn
    def clip_fn(x):
        return enp.clip(x, a_min, a_max)

    return clip_fn


@pytest.mark.parametrize(
    "a_min, a_max",
    [(0.0, 1.0), (-2.0, 0.5)],
    ids=["0-to-1", "neg-to-0.5"],
)
def test_clip_both_bounds_numeric(a_min, a_max):
    a = np.array([[-1.5, 0.25, 2.0], [0.5, -3.0, 0.75]], dtype=np.float32)
    got = etl.evaluate(_clip_defn(a_min, a_max), a).numpy()
    np.testing.assert_allclose(got, np.clip(a, a_min, a_max))


def test_clip_both_bounds_ir_is_max_of_min():
    # Documented composition: ops.maximum(ops.minimum(a, a_max), a_min).
    text = etl.ir.pretty_print(
        etl.trace(_clip_defn(0.0, 1.0), etl.TensorSpec((2, 3), F32)).module
    )
    names = _op_names(text)
    assert "minimum" in names and "maximum" in names
    assert names.index("minimum") < names.index("maximum")


def test_clip_upper_bound_only_is_minimum():
    # BUG(etl): clip None-bound branches inverted — enp.clip(x, None, hi)
    # builds maximum instead of minimum (and vice versa).
    a = np.array([[-1.5, 0.25, 2.0]], dtype=np.float32)
    hi = 1.0
    got = etl.evaluate(_clip_defn(None, hi), a).numpy()
    np.testing.assert_allclose(got, np.minimum(a, hi))


def test_clip_lower_bound_only_is_maximum():
    # BUG(etl): clip None-bound branches inverted — enp.clip(x, lo, None)
    # builds minimum instead of maximum (and vice versa).
    a = np.array([[-1.5, 0.25, 2.0]], dtype=np.float32)
    lo = 0.0
    got = etl.evaluate(_clip_defn(lo, None), a).numpy()
    np.testing.assert_allclose(got, np.maximum(a, lo))


def test_clip_both_none_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        enp.clip(None, None, None)


# --- stack ------------------------------------------------------------------


def _stack_defn(axis):
    @etl.defn
    def stack_fn(x, y, z):
        return enp.stack([x, y, z], axis=axis)

    return stack_fn


def _manual_stack_defn(axis):
    """The documented composition: expand_dims(each, axis) + concatenate(axis)."""

    @etl.defn
    def manual(x, y, z):
        ax = axis + 3 if axis < 0 else axis  # enp normalizes negative axes (rank-2)
        arrays = [x, y, z]
        expanded = [enp.expand_dims(t, ax) for t in arrays]
        return enp.concatenate(expanded, axis=ax)

    return manual


@pytest.mark.parametrize("axis", [0, 1], ids=["axis0", "axis1"])
def test_stack_numeric(axis):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = a + 10.0
    c = -a
    got = etl.evaluate(_stack_defn(axis), a, b, c).numpy()
    np.testing.assert_array_equal(got, np.stack([a, b, c], axis=axis))


@pytest.mark.parametrize("axis", [0, -1], ids=["axis0", "axis-1"])
def test_stack_ir_matches_documented_composition(axis):
    spec = etl.TensorSpec((2, 3), F32)
    assert _normalized_module(_stack_defn(axis), spec, spec, spec) == _normalized_module(
        _manual_stack_defn(axis), spec, spec, spec
    )


def test_stack_empty_raises_shape_error():
    @etl.defn
    def f():
        return enp.stack([])

    with pytest.raises(etl.ShapeError, match="non-empty"):
        etl.trace(f)


# --- split ------------------------------------------------------------------


def _split_defn(indices, axis):
    @etl.defn
    def split_fn(a):
        return enp.split(a, indices, axis=axis)

    return split_fn


def test_split_int_numeric_and_list_of_tensors():
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = etl.evaluate(_split_defn(2, 0), a)
    assert isinstance(out, list)
    assert all(isinstance(t, etl.Tensor) for t in out)
    for got, ref in zip(out, np.split(a, 2, axis=0)):
        np.testing.assert_array_equal(got.numpy(), ref)


def test_split_list_indices_numeric():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    out = etl.evaluate(_split_defn([2, 5], 1), a)
    refs = np.split(a, [2, 5], axis=1)
    assert [t.numpy().shape for t in out] == [r.shape for r in refs]
    for got, ref in zip(out, refs):
        np.testing.assert_array_equal(got.numpy(), ref)


def test_split_non_divisible_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="divisible"):
        etl.trace(_split_defn(3, 0), etl.TensorSpec((4, 3), F32))


@pytest.mark.parametrize(
    "indices",
    [[5, 2], [7], [-1]],
    ids=["not-increasing", "out-of-range", "negative"],
)
def test_split_bad_indices_raise_shape_error(indices):
    with pytest.raises(etl.ShapeError, match="strictly increasing"):
        etl.trace(_split_defn(indices, 0), etl.TensorSpec((4, 3), F32))


def test_split_symbolic_axis_raises_trace_error():
    spec = etl.TensorSpec((4, etl.Dim("n", 8)), F32)
    with pytest.raises(etl.TraceError, match="not a static int"):
        etl.trace(_split_defn(2, 1), spec)


# --- expand_dims ------------------------------------------------------------


def _expand_defn(axis):
    @etl.defn
    def expand_fn(a):
        return enp.expand_dims(a, axis)

    return expand_fn


@pytest.mark.parametrize("axis", [0, 1, -1], ids=["axis0", "axis1", "axis-1"])
def test_expand_dims_int_axis_numeric(axis):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = etl.evaluate(_expand_defn(axis), a).numpy()
    np.testing.assert_array_equal(got, np.expand_dims(a, axis))


def test_expand_dims_tuple_axis_numeric():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = etl.evaluate(_expand_defn((0, 2)), a).numpy()
    np.testing.assert_array_equal(got, np.expand_dims(a, (0, 2)))


def test_expand_dims_tuple_axis_ascending_numeric():
    # BUG(etl): expand_dims validates tuple axes against the ORIGINAL rank —
    # (1, 3) on a rank-2 array raises ShapeError instead of expanding
    # (numpy accepts it: insert at 1, then at 3 of the expanded shape).
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = etl.evaluate(_expand_defn((1, 3)), a).numpy()
    np.testing.assert_array_equal(got, np.expand_dims(a, (1, 3)))


def test_expand_dims_axis_out_of_range_raises():
    with pytest.raises(etl.ShapeError, match="out of range"):
        etl.trace(_expand_defn(3), etl.TensorSpec((2, 3), F32))


# --- squeeze ----------------------------------------------------------------


def _squeeze_defn(axis):
    @etl.defn
    def squeeze_fn(a):
        return enp.squeeze(a, axis)

    return squeeze_fn


@pytest.mark.parametrize("axis", [0, -1, None], ids=["axis0", "axis-1", "all"])
def test_squeeze_static_ones_numeric(axis):
    a = np.arange(3, dtype=np.float32).reshape(1, 3, 1)
    got = etl.evaluate(_squeeze_defn(axis), a).numpy()
    np.testing.assert_array_equal(got, np.squeeze(a, axis=axis))


@pytest.mark.parametrize(
    "shape, axis",
    [((1, 2), 1), ((1, etl.Dim("n", 2)), 1)],
    ids=["static-nonone", "symbolic"],
)
def test_squeeze_non_static_one_raises_trace_error(shape, axis):
    with pytest.raises(etl.TraceError, match="statically 1"):
        etl.trace(_squeeze_defn(axis), etl.TensorSpec(shape, F32))


def test_squeeze_axis_out_of_range_raises():
    with pytest.raises(etl.ShapeError, match="out of range"):
        etl.trace(_squeeze_defn(2), etl.TensorSpec((1, 2), F32))


def test_squeeze_none_keeps_symbolic_dims():
    n = etl.Dim("n", 2)
    captured = []

    @etl.defn
    def f(a):
        out = enp.squeeze(a)  # axis=None drops only statically-1 dims
        captured.append(out)
        return out

    graph = etl.trace(f, etl.TensorSpec((1, n, 1), F32))
    graph.verify()
    assert captured[0].shape == (n,)


# --- where ------------------------------------------------------------------


def _where_defn():
    @etl.defn
    def where_fn(cond, x, y):
        return enp.where(cond, x, y)

    return where_fn


def _select_defn():
    @etl.defn
    def select_fn(cond, x, y):
        return etl.select(cond, x, y)

    return select_fn


def test_where_numeric():
    cond = np.array([[True, False, True], [False, True, False]])
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = np.full((2, 3), -1.0, dtype=np.float32)
    got = etl.evaluate(_where_defn(), cond, x, y).numpy()
    np.testing.assert_array_equal(got, np.where(cond, x, y))


def test_where_ir_equals_select():
    cond_spec = etl.TensorSpec((2, 3), etl.bool_)
    val_spec = etl.TensorSpec((2, 3), F32)
    assert _normalized_module(_where_defn(), cond_spec, val_spec, val_spec) == _normalized_module(
        _select_defn(), cond_spec, val_spec, val_spec
    )


# --- pad --------------------------------------------------------------------


def _pad_defn(pad_width, **kwargs):
    @etl.defn
    def pad_fn(a):
        return enp.pad(a, pad_width, **kwargs)

    return pad_fn


@pytest.mark.parametrize(
    "a",
    [
        pytest.param(np.arange(4, dtype=np.float32), id="rank1"),
        pytest.param(np.arange(6, dtype=np.float32).reshape(2, 3), id="rank2"),
    ],
)
def test_pad_int_numeric(a):
    got = etl.evaluate(_pad_defn(1), a).numpy()
    np.testing.assert_array_equal(got, np.pad(a, 1))


def test_pad_pair_rank1_numeric():
    # BUG(etl): enp.pad(a, (1, 2)) on a rank-1 array raises ShapeError —
    # numpy accepts a single (before, after) pair for the sole axis.
    a = np.arange(4, dtype=np.float32)
    got = etl.evaluate(_pad_defn((1, 2)), a).numpy()
    np.testing.assert_array_equal(got, np.pad(a, (1, 2)))


def test_pad_nested_rank2_numeric():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    got = etl.evaluate(_pad_defn(((1, 2), (0, 1))), a).numpy()
    np.testing.assert_array_equal(got, np.pad(a, ((1, 2), (0, 1))))


def test_pad_constant_values():
    a = np.arange(4, dtype=np.float32)
    got = etl.evaluate(_pad_defn(1, constant_values=5), a).numpy()
    np.testing.assert_array_equal(got, np.pad(a, 1, constant_values=5))


def test_pad_edge_mode_not_implemented():
    with pytest.raises(NotImplementedError, match="deferred"):
        etl.trace(_pad_defn(1, mode="edge"), etl.TensorSpec((4,), F32))
