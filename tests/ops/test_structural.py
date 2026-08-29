"""pytest suite for etl structural ops.

Covers ``clamp`` (numpy-exact scalar-bound dtype behavior — np.clip
``same_kind`` pre-cast of Python-scalar bounds to ``x``'s dtype with weak
promotion fallback — value semantics ``min > max``/NaN propagation,
symbolic shapes/bounds, preserved maximum/minimum composition) and the
15-op-batch structural ops ``tile``/``stack``/``flip``/``roll``/``diag``:
numpy-exact value semantics (dedicated IR ops + numpy kernels; ``stack`` is
a ``reshape``+``concatenate`` composition), dtype rules, symbolic shapes,
and the documented error paths (``TraceError``/``ShapeError``/``ValueError``/
``TypeError``, never silent fallback).
"""
from __future__ import annotations

import numpy as np
import pytest

import etl
from tests.ops.conftest import run_numpy


def _clip_ref(x, lo, hi):
    return np.clip(x, lo, hi)


@pytest.mark.parametrize(
    "x,lo,hi",
    [
        # (dtype, bounds) — result dtype must equal np.clip's.
        (np.array([0, 2, 5], dtype=np.int32), 1, 3),
        (np.array([0, 2, 5], dtype=np.int32), 0.5, 2.5),
        (np.array([0.0, 2.0, 5.0], dtype=np.float32), 0.5, 2.5),
        (np.array([0.0, 2.0, 5.0], dtype=np.float32), 1, 3),
        (np.array([0, 2, 5]), 0.5, 2.5),
        (np.array([0.0, 2.0, 5.0], dtype=np.float16), 1, 3),
    ],
    ids=[
        "int32-int-bounds",
        "int32-float-bounds",
        "float32-float-bounds",
        "float32-int-bounds",
        "int64-float-bounds",
        "float16-int-bounds",
    ],
)
def test_clamp_dtype_and_values_match_np_clip(x, lo, hi):
    out = run_numpy(lambda t: etl.clamp(t, lo, hi), x)
    ref = _clip_ref(x, lo, hi)
    assert out.dtype == ref.dtype
    assert np.array_equal(out, ref)


def test_clamp_int32_int_bounds_stays_int32():
    """The defect: scalar int bounds must be pre-cast to x's dtype (np.clip
    behavior), not weak-promote to int64 via maximum/minimum."""
    x = np.array([0, 2, 5], dtype=np.int32)
    out = run_numpy(lambda t: etl.clamp(t, 1, 3), x)
    assert out.dtype == np.int32
    assert np.array_equal(out, [1, 2, 3])


def test_clamp_int32_float_bounds_promotes_to_float64():
    """Float bounds on an int tensor fall back to weak promotion (numpy 2.x
    np.clip result_type float64 — numpy 1.x raised TypeError)."""
    x = np.array([0, 2, 5], dtype=np.int32)
    out = run_numpy(lambda t: etl.clamp(t, 0.5, 2.5), x)
    assert out.dtype == np.float64
    assert np.array_equal(out, [0.5, 2.0, 2.5])


@pytest.mark.parametrize(
    "x",
    [np.array([1, 2, 3], dtype=np.int32), np.array([1.0, 2.0, 3.0], dtype=np.float32)],
    ids=["int", "float"],
)
def test_clamp_min_greater_than_max(x):
    """min > max returns the all-max result (np.clip semantics)."""
    out = run_numpy(lambda t: etl.clamp(t, 9, 2), x)
    ref = _clip_ref(x, 9, 2)
    assert out.dtype == ref.dtype
    assert np.array_equal(out, ref)


def test_clamp_nan_propagation():
    x = np.array([np.nan, 1.0, -1.0])
    out = run_numpy(lambda t: etl.clamp(t, 0.0, 1.0), x)
    ref = _clip_ref(x, 0.0, 1.0)
    assert out.dtype == ref.dtype
    assert np.array_equal(out, ref, equal_nan=True)
    # NaN bounds propagate too.
    out = run_numpy(lambda t: etl.clamp(t, np.nan, 1.0), np.array([0.5, 2.0]))
    ref = _clip_ref(np.array([0.5, 2.0]), np.nan, 1.0)
    assert np.array_equal(out, ref, equal_nan=True)


def test_clamp_unbounded_sides():
    x = np.array([0, 2, 5], dtype=np.int32)
    out = run_numpy(lambda t: etl.clamp(t, 1, None), x)
    assert out.dtype == np.int32
    assert np.array_equal(out, [1, 2, 5])
    out = run_numpy(lambda t: etl.clamp(t, None, 2), x)
    assert out.dtype == np.int32
    assert np.array_equal(out, [0, 2, 2])


def test_clamp_both_bounds_none_raises_type_error():
    with pytest.raises(TypeError, match="at least one"):
        run_numpy(lambda t: etl.clamp(t, None, None), np.array([1.0]))


def test_clamp_symbolic_x_with_scalar_bounds():
    """A Dim-shaped x with scalar bounds traces cleanly and runs."""
    exe = etl.build(
        lambda t: etl.clamp(t, 1, 3),
        etl.TensorSpec((etl.dim("n"),), etl.float32),
    )
    out = etl.run(exe, np.array([0.0, 2.0, 5.0], dtype=np.float32))
    assert np.array_equal(out.numpy(), [1.0, 2.0, 3.0])


def test_clamp_symbolic_bound_broadcasts():
    """Symbolic tensor bounds keep broadcasting via maximum/minimum."""
    exe = etl.build(
        lambda t, lo: etl.clamp(t, lo, 3.0),
        etl.TensorSpec((3,), etl.float32),
        etl.TensorSpec((etl.dim("n"),), etl.float32),
    )
    out = etl.run(
        exe,
        np.array([0.0, 2.0, 5.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )
    assert np.array_equal(out.numpy(), [1.0, 2.0, 3.0])


def test_clamp_stays_maximum_minimum_composition():
    """No dedicated IR op: the graph is constant(s) + maximum + minimum, so
    the vjp/batching rules of the elementwise ops keep applying."""
    graph = etl.trace(
        lambda t: etl.clamp(t, 1, 3), etl.TensorSpec((4,), etl.int32)
    )
    names = [op.name for op in graph.module.functions[0].region.blocks[0].ops]
    assert set(names) <= {"constant", "maximum", "minimum", "return"}


def test_clamp_grad_and_vmap_via_composition():
    spec = etl.TensorSpec((4,), etl.float32)
    grad_graph = etl.grad(lambda t: etl.sum(etl.clamp(t, 1, 3)))(spec)
    exe = etl.load(etl.compile(etl.lower(grad_graph)))
    out = etl.run(exe, np.array([0.0, 2.0, 5.0, 1.5], dtype=np.float32))
    grads = out[0].numpy() if isinstance(out, tuple) else out.numpy()
    assert np.array_equal(grads, [0.0, 1.0, 0.0, 1.0])

    vmap_graph = etl.vmap(lambda t: etl.clamp(t, 1, 3))(
        etl.TensorSpec((2, 3), etl.float32)
    )
    exe = etl.load(etl.compile(etl.lower(vmap_graph)))
    out = etl.run(
        exe, np.array([[0.0, 2.0, 5.0], [1.0, 4.0, 2.0]], dtype=np.float32)
    )
    arr = out.numpy() if not isinstance(out, tuple) else out[0].numpy()
    assert np.array_equal(arr, [[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])


# ---------------------------------------------------------------------------
# tile (dedicated IR op; numpy-exact np.tile kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,reps",
    [
        (np.array([1, 2, 3], dtype=np.int32), 2),
        (np.array([1, 2, 3], dtype=np.int32), (2,)),
        (np.arange(12, dtype=np.float32).reshape(3, 4), 2),
        (np.arange(12, dtype=np.float32).reshape(3, 4), (2, 3)),
    ],
    ids=["1d-int-reps", "1d-tuple-reps", "2d-int-reps", "2d-tuple-reps"],
)
def test_tile_reps_int_and_tuple_match_np_tile(x, reps):
    out = run_numpy(lambda t: etl.tile(t, reps), x)
    ref = np.tile(x, reps)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    np.testing.assert_array_equal(out, ref)


def test_tile_reps_shorter_than_rank():
    """reps shorter than the rank are pre-padded with 1s (np.tile rule):
    (2, 3) on rank 3 acts as (1, 2, 3)."""
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    out = run_numpy(lambda t: etl.tile(t, (2, 3)), x)
    ref = np.tile(x, (2, 3))
    assert out.shape == ref.shape == (2, 6, 12)
    np.testing.assert_array_equal(out, ref)


def test_tile_reps_longer_than_rank():
    """reps longer than the rank promote the operand with leading size-1
    dims (np.tile rule): rank 2 x with (2, 3, 4) acts on (1, 2, 3)."""
    x = np.arange(6, dtype=np.int32).reshape(2, 3)
    out = run_numpy(lambda t: etl.tile(t, (2, 3, 4)), x)
    ref = np.tile(x, (2, 3, 4))
    assert out.shape == ref.shape == (2, 6, 12)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("reps", [(2, 1), (1, 2)], ids=["row-noop", "col-noop"])
def test_tile_reps_with_ones_noop_dims(reps):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = run_numpy(lambda t: etl.tile(t, reps), x)
    ref = np.tile(x, reps)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


def test_tile_symbolic_dims_trace_and_run():
    """A Dim-shaped x tiles fine: reps (2,) on rank 2 pads to (1, 2), so the
    symbolic first dim is untouched."""
    exe = etl.build(
        lambda t: etl.tile(t, (2,)),
        etl.TensorSpec((etl.dim("n"), 3), etl.float32),
    )
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = etl.run(exe, x)
    assert out.numpy().shape == (4, 6)
    np.testing.assert_array_equal(out.numpy(), np.tile(x, (2,)))


def test_tile_negative_reps_raises_shape_error():
    """Negative reps are a ShapeError at trace time (numpy raises ValueError;
    documented etl deviation — ``ShapeError`` per the op contract)."""
    x = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(etl.ShapeError, match="non-negative"):
        run_numpy(lambda t: etl.tile(t, -2), x)
    with pytest.raises(etl.ShapeError, match="non-negative"):
        run_numpy(lambda t: etl.tile(t, (2, -1)), x)


def test_tile_wrong_reps_kind_raises_type_error():
    with pytest.raises(TypeError, match="int or a tuple"):
        run_numpy(lambda t: etl.tile(t, "2"), np.array([1, 2, 3]))


def test_tile_outside_trace_raises_trace_error():
    with pytest.raises(etl.TraceError):
        etl.tile(np.array([1.0, 2.0]), 2)


def test_tile_concrete_tensor_inside_trace_raises_trace_error():
    t = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(lambda: etl.tile(t, 2))


# ---------------------------------------------------------------------------
# stack (composition over reshape + concatenate; no dedicated IR op)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
def test_stack_list_and_tuple_inputs_match_np_stack(container):
    x = np.arange(6, dtype=np.int32).reshape(2, 3)
    out = run_numpy(lambda t: etl.stack(container([t, t, t])), x)
    ref = np.stack([x, x, x])
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("axis", [0, 1, -1], ids=["axis-0", "axis-1", "axis-neg1"])
def test_stack_axis_variants_2d(axis):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = run_numpy(lambda t: etl.stack([t, t], axis=axis), x)
    ref = np.stack([x, x], axis=axis)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("axis", [0, 1, 2, -1], ids=["axis-0", "axis-1", "axis-2", "axis-neg1"])
def test_stack_axis_variants_3d(axis):
    x = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    out = run_numpy(lambda t: etl.stack([t, t, t], axis=axis), x)
    ref = np.stack([x, x, x], axis=axis)
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out, ref)


def test_stack_dtype_promotion_across_mixed_inputs():
    """Mixed input dtypes promote exactly like ``np.stack`` on the installed
    numpy: NEP 50 makes int32 + float32 → float64 on numpy 2.x (etl's
    concatenate uses ``np.result_type``, the same rule)."""
    a = np.array([[1, 2]], dtype=np.int32)
    b = np.array([[0.5, 1.5]], dtype=np.float32)
    out = run_numpy(lambda x, y: etl.stack([x, y]), a, b)
    ref = np.stack([a, b])
    assert out.dtype == ref.dtype == np.float64
    np.testing.assert_array_equal(out, ref)


def test_stack_mismatched_shapes_raises_shape_error():
    a = np.zeros((2, 3))
    b = np.zeros((2, 4))
    with pytest.raises(etl.ShapeError, match="same shape"):
        run_numpy(lambda x, y: etl.stack([x, y]), a, b)


def test_stack_empty_raises_value_error():
    """Empty input is a ValueError (numpy message) — raised before any trace
    check, so it fires even outside a trace."""
    with pytest.raises(ValueError, match="at least one array"):
        etl.stack([])


def test_stack_axis_out_of_range_raises_shape_error():
    x = np.zeros((2, 3))
    with pytest.raises(etl.ShapeError, match="out of bounds"):
        run_numpy(lambda t: etl.stack([t, t], axis=3), x)
    with pytest.raises(etl.ShapeError, match="out of bounds"):
        run_numpy(lambda t: etl.stack([t, t], axis=-4), x)


def test_stack_symbolic_dims_trace_and_run():
    spec = etl.TensorSpec((etl.dim("n"), 3), etl.float32)
    exe = etl.build(lambda a, b: etl.stack([a, b], axis=0), spec, spec)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = etl.run(exe, x, x * 10)
    assert out.numpy().shape == (2, 2, 3)
    np.testing.assert_array_equal(out.numpy(), np.stack([x, x * 10], axis=0))


def test_stack_is_reshape_concatenate_composition():
    """No dedicated IR op: the graph is size-1 reshapes + one concatenate."""
    graph = etl.trace(
        lambda a, b: etl.stack([a, b], axis=1),
        etl.TensorSpec((2, 3), etl.float32),
        etl.TensorSpec((2, 3), etl.float32),
    )
    names = [op.name for op in graph.module.functions[0].region.blocks[0].ops]
    assert set(names) <= {"reshape", "concatenate", "return"}


def test_stack_concrete_tensor_inside_trace_raises_trace_error():
    t = etl.tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(lambda: etl.stack([t, t]))


# ---------------------------------------------------------------------------
# flip (dedicated IR op; numpy-exact np.flip kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axes",
    [0, 1, (0, 1), None, -1, -2],
    ids=["axis-0", "axis-1", "axes-01", "axes-none", "axis-neg1", "axis-neg2"],
)
def test_flip_axes_int_tuple_none_negative_match_np_flip(axes):
    x = np.arange(12, dtype=np.int32).reshape(3, 4)
    out = run_numpy(lambda t: etl.flip(t, axes), x)
    ref = np.flip(x, axis=axes)
    assert out.shape == ref.shape == x.shape
    assert out.dtype == ref.dtype == np.int32
    np.testing.assert_array_equal(out, ref)


def test_flip_dtype_preserved_float():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = run_numpy(lambda t: etl.flip(t, (0, 1)), x)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.flip(x, axis=(0, 1)))


def test_flip_wrong_axes_kind_raises_type_error():
    with pytest.raises(TypeError, match="int or a tuple"):
        run_numpy(lambda t: etl.flip(t, 1.5), np.array([1.0, 2.0]))


def test_flip_axis_out_of_range_raises_shape_error():
    """Out-of-range axes surface as ShapeError (the numpy AxisError —
    a ValueError subclass — is converted by the kernel at eval time)."""
    with pytest.raises(etl.ShapeError, match="out of bounds"):
        run_numpy(lambda t: etl.flip(t, 2), np.zeros((2, 3)))


def test_flip_outside_trace_raises_trace_error():
    with pytest.raises(etl.TraceError):
        etl.flip(np.array([1.0, 2.0]), 0)


def test_flip_concrete_tensor_inside_trace_raises_trace_error():
    t = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(lambda: etl.flip(t, 0))


# ---------------------------------------------------------------------------
# roll (dedicated IR op; numpy-exact np.roll kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shift,axis",
    [
        (2, 0),
        (-2, 1),
        (2, None),
        (5, 1),
        ((1, -1), (0, 1)),
        (1, (0, 1)),
    ],
    ids=[
        "int-shift-axis-0",
        "neg-shift-axis-1",
        "int-shift-flat",
        "shift-bigger-than-axis",
        "tuple-shift-tuple-axis",
        "scalar-shift-tuple-axis",
    ],
)
def test_roll_shift_axis_match_np_roll(shift, axis):
    x = np.arange(24, dtype=np.int32).reshape(4, 6)
    out = run_numpy(lambda t: etl.roll(t, shift, axis=axis), x)
    ref = np.roll(x, shift, axis=axis)
    assert out.shape == ref.shape == x.shape
    assert out.dtype == ref.dtype == np.int32
    np.testing.assert_array_equal(out, ref)


def test_roll_axis_none_multi_shift_folds_to_sum():
    """A multi-entry shift with axis=None folds to its sum (numpy's
    flattened-roll semantics): (1, 2) ≡ 3."""
    x = np.arange(24, dtype=np.int32).reshape(4, 6)
    out = run_numpy(lambda t: etl.roll(t, (1, 2), axis=None), x)
    np.testing.assert_array_equal(out, np.roll(x, 3))


def test_roll_shift_axis_length_mismatch_raises_value_error():
    """Tuple shift + tuple axis must pair up (numpy message, ValueError)."""
    with pytest.raises(ValueError, match="length of shift"):
        run_numpy(lambda t: etl.roll(t, (1, 2), axis=(0,)), np.zeros((2, 3)))


def test_roll_wrong_shift_kind_raises_type_error():
    with pytest.raises(TypeError, match="shift"):
        run_numpy(lambda t: etl.roll(t, "2", axis=0), np.array([1.0, 2.0]))


def test_roll_axis_out_of_range_raises_shape_error():
    """Out-of-range axes surface as ShapeError (the numpy AxisError is
    converted by the kernel at eval time)."""
    with pytest.raises(etl.ShapeError, match="out of bounds"):
        run_numpy(lambda t: etl.roll(t, 1, axis=5), np.zeros((2, 3)))


def test_roll_outside_trace_raises_trace_error():
    with pytest.raises(etl.TraceError):
        etl.roll(np.array([1.0, 2.0]), 1, axis=0)


def test_roll_concrete_tensor_inside_trace_raises_trace_error():
    t = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(lambda: etl.roll(t, 1, axis=0))


# ---------------------------------------------------------------------------
# diag (dedicated IR op; numpy-exact np.diag kernel)
# ---------------------------------------------------------------------------


def test_diag_1d_builds_square_matrix():
    """Rank-1 input → (n, n) matrix with the values on the diagonal and
    zeros elsewhere (np.diag(v))."""
    v = np.array([1, 2, 3], dtype=np.int32)
    out = run_numpy(lambda t: etl.diag(t), v)
    ref = np.diag(v)
    assert out.shape == ref.shape == (3, 3)
    assert out.dtype == ref.dtype == np.int32
    np.testing.assert_array_equal(out, ref)


def test_diag_2d_extracts_diagonal():
    """Rank-2 input → the main diagonal vector (np.diag(m))."""
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = run_numpy(lambda t: etl.diag(t), m)
    ref = np.diag(m)
    assert out.shape == ref.shape == (3,)
    assert out.dtype == ref.dtype == np.float32
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("shape", [(2, 5), (5, 2)], ids=["2x5", "5x2"])
def test_diag_rectangular_2d_diagonal_length_is_min(shape):
    m = np.arange(10, dtype=np.int32).reshape(*shape)
    out = run_numpy(lambda t: etl.diag(t), m)
    ref = np.diag(m)
    assert out.shape == (min(shape),)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, ref)


def test_diag_dtype_preserved_both_directions():
    v = np.array([1, 2, 3], dtype=np.int64)
    out = run_numpy(lambda t: etl.diag(t), v)
    assert out.dtype == np.int64
    m = np.arange(4, dtype=np.float64).reshape(2, 2)
    out = run_numpy(lambda t: etl.diag(t), m)
    assert out.dtype == np.float64


def test_diag_empty_1d_zero_by_zero():
    out = run_numpy(lambda t: etl.diag(t), np.array([], dtype=np.int32))
    assert out.shape == (0, 0)
    assert out.dtype == np.int32


def test_diag_rank_gt_2_raises_shape_error():
    # Deviation from numpy: np.diag raises ValueError ("Input must be 1- or
    # 2-d"); etl raises ShapeError (documented in etl/ops/structural.py).
    with pytest.raises(etl.ShapeError, match="1- or 2-d"):
        run_numpy(lambda t: etl.diag(t), np.zeros((2, 2, 2)))


def test_diag_outside_trace_raises_trace_error():
    with pytest.raises(etl.TraceError):
        etl.diag(np.array([1.0, 2.0]))


def test_diag_concrete_tensor_inside_trace_raises_trace_error():
    t = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(etl.TraceError, match="no eager mode"):
        etl.trace(lambda: etl.diag(t))
