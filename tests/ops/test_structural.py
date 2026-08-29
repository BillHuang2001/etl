"""pytest suite for etl structural ops — regression coverage for ``clamp``.

Focus: the numpy-exact scalar-bound dtype behavior of ``clamp`` (np.clip
``same_kind`` pre-cast of Python-scalar bounds to ``x``'s dtype, with weak
promotion fallback), value semantics (``min > max``, NaN propagation),
symbolic shapes/bounds, and the preserved maximum/minimum composition.
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
