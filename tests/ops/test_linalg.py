"""Contract tests for the linear-algebra / scan ops.

Covers ``etl.ops.linalg`` (``dot``, ``conv``, ``tril``, ``triu``, ``cumsum``,
``solve``, ``eigh``, ``cholesky``, ``qr``, ``matrix_rank``, ``svd``,
``matrix_exp``), ``etl.ops.reductions.argmax/argmin`` and
``etl.ops.constant.stop_gradient``. The etl package (repo-root sibling) is
fully implemented; these tests assert the per-op contracts documented in the
``etl/ops/linalg.py`` docstrings and ``etl/ops/CONTEXT.md``:

- ``dot``: batched matmul (numpy ``matmul`` shape rules); dtype =
  ``np.result_type``; static k-mismatch raises ``ShapeError`` at trace time;
  a size-1 k dim broadcasts via an EXPLICIT ``broadcast`` op (sanctioned
  composition, not a bug); operands must have rank >= 2.
- ``conv``: NCHW default (``channels_last`` is transpose sugar); each out dim
  is the DimExpr formula ``(d + 2*pad - kdil*(k-1) - 1) // stride + 1``;
  numerics vs a manual numpy cross-correlation reference.
- ``solve``: numpy ``linalg.solve`` semantics; integer/bool inputs → float64.
- ``tril``/``triu``: numpy semantics with ``k`` offset; shape/dtype preserved.
- ``cumsum``: numpy semantics plus a reverse scan ("from the end toward the
  start"); bool → int64; other dtypes preserved (numpy-2-proof).
- ``eigh``/``cholesky``/``qr``/``matrix_rank``/``svd``: numpy ``linalg``
  semantics (dtype rule int/bool → float64; eigenvalues/singular values real
  at the input's precision; batched).
- ``matrix_exp``: scipy/torch semantics (numpy has no ``linalg.matrix_exp``);
  reference kernel is pure-numpy scaling-and-squaring Taylor.
- ``argmax``/``argmin``: result dtype int64; numpy semantics.
- ``stop_gradient``: identity barrier; effect pure.
"""
from __future__ import annotations

import numpy as np
import pytest

import etl
from tests.ops.conftest import ops_of, run_numpy, trace_fn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _op(graph, name):
    """The single op of ``graph`` with the given name (fails otherwise)."""
    ops = ops_of(graph, name)
    assert len(ops) == 1, f"expected exactly one {name} op, got {len(ops)}"
    return ops[0]


def _trace_capturing(fn, *specs):
    """Trace ``fn`` and return ``(graph, returned_symbolic_tensor)``.

    The returned SymbolicTensor is captured inside the trace, so its
    ``.shape``/``.dtype`` (the frontend contract) can be asserted alongside
    the IR value type read back from the built op.
    """
    captured = {}

    def wrapped(*args):
        out = fn(*args)
        captured["out"] = out
        return out

    graph = etl.trace(wrapped, *specs)
    return graph, captured["out"]


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------

DOT_TYPE_CASES = [
    ((2, 3), (3, 4), etl.float32, etl.float32, (2, 4), np.float32),
    ((3, 4), (4, 2), etl.float64, etl.float64, (3, 2), np.float64),
    # NEP-50-agnostic np.result_type promotion (int32 ⊕ float32 → float64)
    ((2, 3), (3, 4), etl.int32, etl.float32, (2, 4), np.float64),
    ((2, 4), (4, 3), etl.float16, etl.float16, (2, 3), np.float16),
    ((1, 5), (5, 1), etl.float64, etl.float64, (1, 1), np.float64),
    ((2, 3), (3, 4), etl.int64, etl.int64, (2, 4), np.int64),
]


@pytest.mark.parametrize(
    "a_shape,b_shape,a_dtype,b_dtype,out_shape,out_dtype", DOT_TYPE_CASES
)
def test_dot_shape_and_dtype_inference(
    a_shape, b_shape, a_dtype, b_dtype, out_shape, out_dtype
):
    def f(x, w):
        return etl.dot(x, w)

    graph, out = _trace_capturing(
        f,
        etl.TensorSpec(a_shape, a_dtype),
        etl.TensorSpec(b_shape, b_dtype),
    )
    # Frontend SymbolicTensor contract.
    assert isinstance(out, etl.SymbolicTensor)
    assert tuple(out.shape) == out_shape
    assert out.dtype == np.dtype(out_dtype)
    # IR value type agrees (read back from the op, never computed twice).
    op = _op(graph, "dot")
    assert op.results[0].type.shape == out_shape
    assert op.results[0].type.dtype == np.dtype(out_dtype)
    # Documented rule: exactly np.result_type of the operand dtypes.
    assert out.dtype == np.result_type(np.dtype(a_dtype), np.dtype(b_dtype))


def test_dot_2d_numerics_vs_numpy():
    rng = np.random.default_rng(0)

    def f(x, w):
        return etl.dot(x, w)

    x = rng.standard_normal((3, 5))
    w = rng.standard_normal((5, 4))
    np.testing.assert_allclose(run_numpy(f, x, w), np.matmul(x, w))
    np.testing.assert_allclose(run_numpy(f, x, w), np.dot(x, w))

    xi = rng.integers(-3, 4, size=(3, 5)).astype(np.int32)
    wi = rng.integers(-3, 4, size=(5, 4)).astype(np.int32)
    np.testing.assert_array_equal(run_numpy(f, xi, wi), np.matmul(xi, wi))


DOT_BATCH_CASES = [
    ((2, 3, 4), (4, 5), (2, 3, 5)),
    ((2, 3, 4), (1, 4, 5), (2, 3, 5)),   # batch dims broadcast
    ((2, 1, 3, 4), (5, 4, 6), (2, 5, 3, 6)),  # (2,1,3) vs (5,) → (2,5,3)
    ((1, 3, 4), (2, 4, 5), (2, 3, 5)),
]


@pytest.mark.parametrize("a_shape,b_shape,out_shape", DOT_BATCH_CASES)
def test_dot_nd_batched_with_broadcast(a_shape, b_shape, out_shape):
    rng = np.random.default_rng(1)

    def f(x, w):
        return etl.dot(x, w)

    graph, out = _trace_capturing(
        f, etl.TensorSpec(a_shape, etl.float64), etl.TensorSpec(b_shape, etl.float64)
    )
    assert tuple(out.shape) == out_shape
    op = _op(graph, "dot")
    assert op.results[0].type.shape == out_shape

    x = rng.standard_normal(a_shape)
    w = rng.standard_normal(b_shape)
    np.testing.assert_allclose(run_numpy(f, x, w), np.matmul(x, w))


def test_dot_symbolic_batch_broadcast():
    def f(x, w):
        return etl.dot(x, w)

    graph, _ = _trace_capturing(
        f,
        etl.TensorSpec((etl.dim("b"), 3, 4), etl.float32),
        etl.TensorSpec((2, 4, 5), etl.float32),
    )
    shape = _op(graph, "dot").results[0].type.shape
    # Batch = broadcast((b,), (2,)) → the symbolic statement of numpy's rule.
    assert isinstance(shape[0], etl.DimExpr)
    assert shape[0].evaluate({"b": 1}) == 2
    assert shape[0].evaluate({"b": 7}) == 7
    assert shape[1:] == (3, 5)


def test_dot_k_mismatch_is_static_shape_error():
    def f(x, w):
        return etl.dot(x, w)

    with pytest.raises(etl.ShapeError, match="contracting dims 3 and 4 do not match"):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32),
                  etl.TensorSpec((4, 5), etl.float32))
    # Also static inside a batch.
    with pytest.raises(etl.ShapeError, match="contracting dims 4 and 6 do not match"):
        etl.trace(f, etl.TensorSpec((2, 3, 4), etl.float32),
                  etl.TensorSpec((2, 6, 5), etl.float32))


def test_dot_requires_rank_2_or_more():
    """ops.dot is batched matmul: rank < 2 on either side → ShapeError
    (vector dot is an IR-level contract, not exposed by the frontend)."""

    def f(x, w):
        return etl.dot(x, w)

    with pytest.raises(etl.ShapeError, match="operands must have rank >= 2"):
        etl.trace(f, etl.TensorSpec((3,), etl.float32),
                  etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(etl.ShapeError, match="operands must have rank >= 2"):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32),
                  etl.TensorSpec((3,), etl.float32))


@pytest.mark.parametrize("side", ["a", "b"])
def test_dot_size1_k_broadcasts_via_explicit_broadcast_op(side):
    """A size-1 k dim broadcasts (numpy matmul semantics) by inserting an
    EXPLICIT ``broadcast`` op — sanctioned composition per the ops contract."""

    def f(x, w):
        return etl.dot(x, w)

    a_shape = (2, 3, 1) if side == "a" else (2, 3)
    b_shape = (4, 5) if side == "a" else (1, 4)
    graph = trace_fn(f, etl.TensorSpec(a_shape, etl.float32),
                     etl.TensorSpec(b_shape, etl.float32))
    names = [op.name for op in ops_of(graph)]
    assert names == ["broadcast", "dot", "return"]
    bcast = ops_of(graph, "broadcast")[0]
    expected_k = b_shape[-2] if side == "a" else a_shape[-1]
    # The broadcast op expands the size-1 k dim: last dim for a, next-to-last
    # for b (target shapes (..., m, kb) and (..., ka, n)).
    k_axis = -1 if side == "a" else -2
    assert bcast.results[0].type.shape[k_axis] == expected_k
    dot_op = _op(graph, "dot")
    expected_out = (2, 4) if side == "b" else (2, 3, 5)
    assert dot_op.results[0].type.shape == expected_out

    rng = np.random.default_rng(2)
    x = rng.standard_normal(a_shape).astype(np.float32)
    w = rng.standard_normal(b_shape).astype(np.float32)
    got = run_numpy(f, x, w)
    if side == "a":
        expected = np.matmul(np.broadcast_to(x, (2, 3, 4)), w)
    else:
        expected = np.matmul(x, np.broadcast_to(w, (3, 4)))
    np.testing.assert_allclose(got, expected)


def test_dot_matmul_operator_handler():
    """``x @ w`` registers to the dot op (SymbolicTensor.__matmul__)."""

    def f(x, w):
        return x @ w

    graph, out = _trace_capturing(
        f, etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((3, 4), etl.float32)
    )
    assert tuple(out.shape) == (2, 4)
    assert _op(graph, "dot").name == "dot"
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.ones((3, 4), dtype=np.float32)
    np.testing.assert_allclose(run_numpy(f, x, w), x @ w)


def test_dot_scalar_scalar_is_trace_error():
    with pytest.raises(etl.TraceError, match="both operands are Python scalars"):
        etl.trace(lambda: etl.dot(2.0, 3.0))


# ---------------------------------------------------------------------------
# conv
# ---------------------------------------------------------------------------

def _per_spatial(value, n_spatial):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * n_spatial


def _dilate(arr, rates):
    """Insert ``rate - 1`` zeros between elements along trailing spatial dims."""
    n_spatial = len(rates)
    new_shape = list(arr.shape[:-n_spatial])
    for i in range(n_spatial):
        new_shape.append((arr.shape[-n_spatial + i] - 1) * rates[i] + 1)
    out = np.zeros(new_shape, dtype=arr.dtype)
    out[(slice(None),) * (arr.ndim - n_spatial) + tuple(
        slice(None, None, rates[i]) for i in range(n_spatial)
    )] = arr
    return out


def np_conv_ref(x, w, strides=1, padding="VALID", input_dilation=1,
                kernel_dilation=1, feature_group_size=1, channels_last=False):
    """Manual numpy cross-correlation implementing the documented etl conv
    semantics: dilate input and kernel, pad the dilated input, slide with
    strides (NCHW layout; VALID / SAME / int / per-spatial (lo, hi) pairs)."""
    if channels_last:
        x = np.moveaxis(x, -1, 1)
    n_spatial = x.ndim - 2
    strides = _per_spatial(strides, n_spatial)
    in_dil = _per_spatial(input_dilation, n_spatial)
    k_dil = _per_spatial(kernel_dilation, n_spatial)
    x_d = _dilate(x, in_dil)
    w_d = _dilate(w, k_dil)
    eff_d = x_d.shape[2:]
    eff_k = w_d.shape[2:]
    if padding == "VALID":
        pads = [(0, 0)] * n_spatial
        out_spatial = [
            (eff_d[i] - eff_k[i]) // strides[i] + 1 for i in range(n_spatial)
        ]
    elif padding == "SAME":
        out_spatial = [
            (x.shape[2 + i] + strides[i] - 1) // strides[i]
            for i in range(n_spatial)
        ]
        pads = []
        for i in range(n_spatial):
            total = (out_spatial[i] - 1) * strides[i] + eff_k[i] - eff_d[i]
            lo = total // 2
            pads.append((lo, total - lo))
    elif isinstance(padding, int):
        pads = [(padding, padding)] * n_spatial
        out_spatial = [
            (eff_d[i] + 2 * padding - eff_k[i]) // strides[i] + 1
            for i in range(n_spatial)
        ]
    else:  # per-spatial (lo, hi) pairs
        pads = [tuple(p) for p in padding]
        out_spatial = [
            (eff_d[i] + pads[i][0] + pads[i][1] - eff_k[i]) // strides[i] + 1
            for i in range(n_spatial)
        ]
    x_p = np.pad(
        x_d,
        ((0, 0), (0, 0))
        + tuple((max(lo, 0), max(hi, 0)) for lo, hi in pads),
    )
    if any(lo < 0 or hi < 0 for lo, hi in pads):
        x_p = x_p[
            (slice(None), slice(None))
            + tuple(
                slice(-lo if lo < 0 else None, hi if hi < 0 else None)
                for lo, hi in pads
            )
        ]
    n, c_in = x.shape[:2]
    c_out = w.shape[0]
    gin = c_in // feature_group_size
    gout = c_out // feature_group_size
    result_dtype = np.result_type(x.dtype, w.dtype)
    out = np.zeros((n, c_out) + tuple(out_spatial), dtype=result_dtype)
    for ni in range(n):
        for g in range(feature_group_size):
            for co in range(gout):
                for pos in np.ndindex(tuple(out_spatial)):
                    acc = np.zeros((), dtype=result_dtype)
                    for ci in range(gin):
                        for kk in np.ndindex(tuple(eff_k)):
                            idx = tuple(
                                pos[i] * strides[i] + kk[i] for i in range(n_spatial)
                            )
                            acc += x_p[ni, g * gin + ci][idx] * w_d[g * gout + co, ci][kk]
                    out[ni, g * gout + co][pos] = acc
    if channels_last:
        out = np.moveaxis(out, 1, -1)
    return out


def _conv_out_dim(d, k, stride, lo, hi, kdil):
    """The documented formula: (d + lo + hi - kdil*(k-1) - 1)//stride + 1."""
    return (d + lo + hi - kdil * (k - 1) - 1) // stride + 1


CONV_INFERENCE_CASES = [
    # x_shape, w_shape, kwargs, expected out shape (concrete)
    ((2, 3, 8, 8), (4, 3, 3, 3), {}, (2, 4, 6, 6)),                       # VALID
    ((2, 3, 6, 7), (4, 3, 3, 3), {"strides": (1, 2)}, (2, 4, 4, 3)),
    ((2, 3, 7, 7), (4, 3, 3, 3), {"kernel_dilation": 2}, (2, 4, 3, 3)),
    ((2, 3, 8, 9), (4, 3, 3, 3),
     {"strides": (1, 2), "padding": ((1, 2), (0, 1)), "kernel_dilation": (2, 1)},
     (2, 4, 7, 4)),
    ((1, 1, 4, 4), (2, 1, 3, 3), {"padding": 1}, (1, 2, 4, 4)),
    ((2, 3, 6, 6), (4, 3, 3, 3), {"strides": 2, "padding": "SAME"}, (2, 4, 3, 3)),
    ((1, 2, 4, 4), (3, 2, 2, 2), {"input_dilation": 2}, (1, 3, 6, 6)),
    ((2, 4, 5, 5), (4, 2, 3, 3), {"feature_group_size": 2}, (2, 4, 3, 3)),
]


@pytest.mark.parametrize("x_shape,w_shape,kwargs,expected", CONV_INFERENCE_CASES)
def test_conv_shape_inference_formula(x_shape, w_shape, kwargs, expected):
    def f(x, w):
        return etl.conv(x, w, **kwargs)

    graph, out = _trace_capturing(
        f, etl.TensorSpec(x_shape, etl.float32), etl.TensorSpec(w_shape, etl.float32)
    )
    assert tuple(out.shape) == expected
    op = _op(graph, "conv")
    assert op.results[0].type.shape == expected
    assert op.attributes["strides"] == _per_spatial(kwargs.get("strides", 1), 2)
    assert op.attributes["input_dilation"] == _per_spatial(
        kwargs.get("input_dilation", 1), 2
    )
    assert op.attributes["kernel_dilation"] == _per_spatial(
        kwargs.get("kernel_dilation", 1), 2
    )
    assert op.attributes["feature_group_count"] == kwargs.get("feature_group_size", 1)


def test_conv_symbolic_dims_produce_dimexpr():
    """Out dims are DimExpr formulas; bindings evaluate to the documented
    formula result."""
    def f(x, w):
        return etl.conv(x, w, strides=2, padding="SAME", kernel_dilation=2)

    graph, _ = _trace_capturing(
        f,
        etl.TensorSpec((2, 3, etl.dim("h"), etl.dim("w")), etl.float32),
        etl.TensorSpec((4, 3, 3, 3), etl.float32),
    )
    shape = _op(graph, "conv").results[0].type.shape
    assert isinstance(shape[2], etl.DimExpr) and isinstance(shape[3], etl.DimExpr)
    # SAME: out = ceil(d / stride) = (d + stride - 1) // stride.
    assert shape[2].evaluate({"h": 11, "w": 5}) == (11 + 2 - 1) // 2
    assert shape[3].evaluate({"h": 11, "w": 5}) == (5 + 2 - 1) // 2
    # VALID: out = (d - kdil*(k-1) - 1) // stride + 1.
    def g(x, w):
        return etl.conv(x, w)

    graph2, _ = _trace_capturing(
        g,
        etl.TensorSpec((2, 3, etl.dim("h"), etl.dim("w")), etl.float32),
        etl.TensorSpec((4, 3, 3, 3), etl.float32),
    )
    shape2 = _op(graph2, "conv").results[0].type.shape
    assert shape2[2].evaluate({"h": 10, "w": 7}) == _conv_out_dim(10, 3, 1, 0, 0, 1)
    assert shape2[3].evaluate({"h": 10, "w": 7}) == _conv_out_dim(7, 3, 1, 0, 0, 1)


CONV_NUMERIC_CASES = [
    ((2, 3, 5, 5), (4, 3, 3, 3), {}, "valid-s1"),
    ((1, 2, 6, 5), (3, 2, 3, 2), {"strides": (2, 1)}, "valid-strides"),
    ((2, 3, 5, 5), (4, 3, 3, 3),
     {"strides": 2, "padding": "SAME", "kernel_dilation": 2}, "same-s2-kdil2"),
    ((1, 2, 4, 6), (3, 2, 3, 3), {"padding": "SAME"}, "same-s1"),
    ((2, 3, 8, 9), (4, 3, 3, 3),
     {"strides": (1, 2), "padding": ((1, 2), (0, 1)), "kernel_dilation": (2, 1)},
     "pad-pairs"),
    ((1, 1, 4, 4), (2, 1, 3, 3), {"padding": 1}, "pad-int"),
    ((1, 2, 4, 4), (3, 2, 2, 2), {"input_dilation": 2}, "input-dilation"),
    ((2, 4, 5, 5), (4, 2, 3, 3), {"feature_group_size": 2}, "grouped"),
]


@pytest.mark.parametrize("x_shape,w_shape,kwargs,_id", CONV_NUMERIC_CASES)
def test_conv_numerics_vs_manual_reference(x_shape, w_shape, kwargs, _id):
    def f(x, w):
        return etl.conv(x, w, **kwargs)

    rng = np.random.default_rng(3)
    x = rng.integers(0, 100, size=x_shape).astype(np.float32)
    w = rng.integers(0, 100, size=w_shape).astype(np.float32)
    got = run_numpy(f, x, w)
    expected = np_conv_ref(x, w, **kwargs)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_conv_mixed_dtype_promotes_result_type():
    def f(x, w):
        return etl.conv(x, w)

    graph, _ = _trace_capturing(
        f, etl.TensorSpec((1, 2, 4, 4), etl.int32), etl.TensorSpec((3, 2, 2, 2), etl.float32)
    )
    assert _op(graph, "conv").results[0].type.dtype == np.result_type(
        np.int32, np.float32
    )
    x = np.arange(1 * 2 * 4 * 4, dtype=np.int32).reshape(1, 2, 4, 4)
    w = np.ones((3, 2, 2, 2), dtype=np.float32)
    got = run_numpy(f, x, w)
    expected = np_conv_ref(x, w)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, expected)


def test_conv_channels_last_is_transpose_sugar():
    def f_nchw(x, w):
        return etl.conv(x, w, strides=2, padding="SAME")

    def f_cl(x, w):
        return etl.conv(x, w, strides=2, padding="SAME", channels_last=True)

    graph, out = _trace_capturing(
        f_cl,
        etl.TensorSpec((2, 6, 6, 3), etl.float32),
        etl.TensorSpec((4, 3, 3, 3), etl.float32),
    )
    assert tuple(out.shape) == (2, 3, 3, 4)  # (N, *spatial, C_out)
    names = [op.name for op in ops_of(graph)]
    assert names == ["transpose", "conv", "transpose", "return"]

    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    out_nchw = run_numpy(f_nchw, x, w)
    out_cl = run_numpy(f_cl, np.moveaxis(x, 1, -1), w)
    np.testing.assert_array_equal(out_cl, np.moveaxis(out_nchw, 1, -1))


CONV_ERROR_CASES = [
    ((2, 3), (4, 3, 3), {},
     etl.ShapeError, "conv: input rank must be >= 3"),
    ((2, 3, 5, 5), (4, 3, 3), {},
     etl.ShapeError, "conv: input and kernel ranks differ: 4 vs 3"),
    ((2, 3, 5, 5), (4, 2, 3, 3), {},
     etl.ShapeError, "conv: kernel in-channels 2 != in_channels 3"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"strides": 0},
     etl.ShapeError, "strides entries must be positive ints, got 0"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"input_dilation": (2, 0)},
     etl.ShapeError, "input_dilation entries must be positive ints"),
    ((2, 4, 5, 5), (3, 2, 3, 3), {"feature_group_size": 2},
     etl.ShapeError, "conv: out_channels 3 not divisible by feature_group_size 2"),
    ((2, 3, 5, 5), (4, 1, 3, 3), {"feature_group_size": 2},
     etl.ShapeError, "conv: in_channels 3 not divisible by feature_group_count 2"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"padding": "FOO"},
     etl.ShapeError, "conv: unknown padding mode 'FOO'"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"padding": ((1, 1),)},
     etl.ShapeError, "conv: expected 2 padding entries, got 1"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"padding": ((1, -1), (0, 0))},
     etl.ShapeError, "conv: negative padding"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"strides": (1, 2, 3)},
     etl.ShapeError, "conv: strides must have 2 entries, got 3"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"channels_last": 1},
     TypeError, "conv: channels_last must be a bool, got 1"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"strides": "x"},
     TypeError, "conv: strides must be an int or a tuple of 2 ints, got 'x'"),
    ((2, 3, 5, 5), (4, 3, 3, 3), {"feature_group_size": True},
     TypeError, "conv: feature_group_size must be an int, got True"),
]


@pytest.mark.parametrize(
    "x_shape,w_shape,kwargs,exc,msg", CONV_ERROR_CASES,
    ids=[f"case-{i}" for i in range(len(CONV_ERROR_CASES))],
)
def test_conv_errors(x_shape, w_shape, kwargs, exc, msg):
    def f(x, w):
        return etl.conv(x, w, **kwargs)

    with pytest.raises(exc, match=msg):
        etl.trace(f, etl.TensorSpec(x_shape, etl.float32),
                  etl.TensorSpec(w_shape, etl.float32))


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def _well_conditioned_a(n, rng):
    """Symmetric diagonally dominant matrix — always well-conditioned."""
    a = rng.standard_normal((n, n)) * 0.1
    a = a @ a.T + np.eye(n)
    return a


SOLVE_NUMERIC_CASES = [
    (3, 1, etl.float64, False),   # single system, vector b
    (3, 2, etl.float64, False),   # matrix b (k columns)
    (2, 1, etl.float64, True),    # batched a (2,3,3), batched b (2,3)
    (2, 2, etl.float64, True),    # batched a, batched b (2,3,2)
    (3, 1, etl.float32, False),   # float32 stays float32
]


@pytest.mark.parametrize("n,k,dtype,batched", SOLVE_NUMERIC_CASES)
def test_solve_numerics_vs_numpy(n, k, dtype, batched):
    def f(a, b):
        return etl.solve(a, b)

    rng = np.random.default_rng(5)
    a = _well_conditioned_a(n, rng).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype).T if not batched else None
    if not batched:
        b = rng.standard_normal((n,) if k == 1 else (n, k)).astype(dtype)
        expected = np.linalg.solve(a, b)
    else:
        a = np.stack([_well_conditioned_a(n, rng) for _ in range(2)]).astype(dtype)
        b = rng.standard_normal((2, n) if k == 1 else (2, n, k)).astype(dtype)
        expected = np.linalg.solve(a, b)
    got = run_numpy(f, a, b)
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_solve_int_input_promotes_to_float64():
    def f(a, b):
        return etl.solve(a, b)

    graph, _ = _trace_capturing(
        f, etl.TensorSpec((3, 3), etl.int32), etl.TensorSpec((3,), etl.int32)
    )
    assert _op(graph, "solve").results[0].type.dtype == np.float64
    a = np.array([[4, 1, 0], [1, 4, 1], [0, 1, 4]], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    got = run_numpy(f, a, b)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, np.linalg.solve(a.astype(np.float64), b.astype(np.float64)))


def test_solve_mixed_int_float_promotes_to_float64():
    """Contract: ANY integer/bool operand → float64 (numpy linalg.solve rule)."""
    def f(a, b):
        return etl.solve(a, b)

    graph, _ = _trace_capturing(
        f, etl.TensorSpec((3, 3), etl.int32), etl.TensorSpec((3,), etl.float32)
    )
    assert _op(graph, "solve").results[0].type.dtype == np.float64


SOLVE_ERROR_CASES = [
    ((3, 4), (3,), etl.ShapeError, "solve: 'a' must be square, got 3 vs 4"),
    ((3, 3), (2,), etl.ShapeError, "solve: contracting dims 3 and 2 do not match"),
    ((2, 3, 3), (5, 3, 3), etl.ShapeError,
     "cannot broadcast incompatible dims 2 and 5"),
    ((3, 3), (), etl.ShapeError, "solve: 'b' must have rank >= 1"),
    ((3,), (3,), etl.ShapeError, "solve: 'a' must have rank >= 2"),
]


@pytest.mark.parametrize("a_shape,b_shape,exc,msg", SOLVE_ERROR_CASES)
def test_solve_shape_errors(a_shape, b_shape, exc, msg):
    def f(a, b):
        return etl.solve(a, b)

    with pytest.raises(exc, match=msg):
        etl.trace(f, etl.TensorSpec(a_shape, etl.float64),
                  etl.TensorSpec(b_shape, etl.float64))


# ---------------------------------------------------------------------------
# tril / triu
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", ["tril", "triu"])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_tril_triu_numerics_vs_numpy(op_name, k):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, k=k)

    rng = np.random.default_rng(6)
    x = rng.integers(-9, 10, size=(2, 3, 4)).astype(np.int32)
    np_ref = np.tril if op_name == "tril" else np.triu
    np.testing.assert_array_equal(run_numpy(f, x), np_ref(x, k=k))


@pytest.mark.parametrize("op_name", ["tril", "triu"])
def test_tril_triu_preserve_shape_and_dtype(op_name):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, k=-1)

    graph, out = _trace_capturing(
        f, etl.TensorSpec((2, 3, 4), etl.float16)
    )
    assert tuple(out.shape) == (2, 3, 4)
    assert out.dtype == np.float16
    assert _op(graph, op_name).results[0].type.shape == (2, 3, 4)

    xb = np.ones((3, 3), dtype=bool)
    np_ref = np.tril if op_name == "tril" else np.triu
    got = run_numpy(f, xb)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, np_ref(xb, k=-1))


@pytest.mark.parametrize("op_name", ["tril", "triu"])
def test_tril_triu_errors(op_name):
    op = getattr(etl, op_name)

    with pytest.raises(etl.ShapeError,
                       match=f"{op_name}: input must have rank >= 2"):
        etl.trace(lambda x: op(x), etl.TensorSpec((3,), etl.float32))
    with pytest.raises(TypeError, match=f"{op_name}: k must be an int, got 0.5"):
        etl.trace(lambda x: op(x, k=0.5), etl.TensorSpec((3, 3), etl.float32))


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", [etl.float64, etl.float32, etl.int32])
def test_cumsum_numerics_vs_numpy(axis, dtype):
    def f(x):
        return etl.cumsum(x, axis=axis)

    x = np.arange(12).reshape(3, 4).astype(dtype)
    got = run_numpy(f, x)
    # dtype is preserved exactly (numpy >= 2 upcasts int32 cumsum — the etl
    # contract explicitly accumulates in the operand dtype).
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, np.cumsum(x, axis=axis, dtype=x.dtype))


def test_cumsum_bool_promotes_to_int64():
    """Documented rule: bool → int64, implemented as an explicit pre-cast."""
    def f(x):
        return etl.cumsum(x, axis=1)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), etl.bool_))
    names = [op.name for op in ops_of(graph)]
    assert names == ["cast", "cumsum", "return"]
    assert out.dtype == np.int64
    x = np.array([[True, False, True], [False, True, True]])
    got = run_numpy(f, x)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, np.cumsum(x.astype(np.int64), axis=1))


def test_cumsum_reverse_scans_from_the_end():
    """``reverse=True`` is documented as "scan from the end toward the start",
    i.e. out[i] = sum(x[i:]) — equivalently flip(cumsum(flip(x))).
    """
    def f(x):
        return etl.cumsum(x, axis=1, reverse=True)

    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    expected = np.flip(np.cumsum(np.flip(x, axis=1), axis=1), axis=1)
    np.testing.assert_array_equal(run_numpy(f, x), expected)


def test_cumsum_scalar_is_identity():
    def f(x):
        return etl.cumsum(x, axis=0)

    got = run_numpy(f, np.array(3.5, dtype=np.float32))
    assert got.shape == ()
    assert got.dtype == np.float32
    assert got == np.float32(3.5)


def test_cumsum_errors():
    with pytest.raises(etl.ShapeError, match="axis 3 out of range for rank 2"):
        etl.trace(lambda x: etl.cumsum(x, axis=3),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError,
                       match="axes must be None, an int, or a tuple of ints"):
        etl.trace(lambda x: etl.cumsum(x, axis=1.5),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError, match="cumsum: reverse must be a bool"):
        etl.trace(lambda x: etl.cumsum(x, axis=0, reverse="yes"),
                  etl.TensorSpec((2, 3), etl.float32))


# ---------------------------------------------------------------------------
# argmax / argmin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", ["argmax", "argmin"])
@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmax_argmin_numerics_vs_numpy(op_name, axis, keepdims):
    op = getattr(etl, op_name)

    def f(x):
        return op(x, axis=axis, keepdims=keepdims)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4))
    np_ref = np.argmax if op_name == "argmax" else np.argmin
    expected = np_ref(x, axis=axis)
    if keepdims:
        if axis is None:
            expected = np.reshape(expected, (1,) * x.ndim)
        else:
            expected = np.expand_dims(expected, axis=axis)
    got = run_numpy(f, x)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, expected)


def test_argmax_argmin_result_dtype_is_int64_in_ir():
    def f(x):
        return etl.argmax(x, axis=0), etl.argmin(x, axis=0)

    graph = trace_fn(f, etl.TensorSpec((3, 4), etl.float32))
    for name in ("argmax", "argmin"):
        op = _op(graph, name)
        assert op.results[0].type.dtype == np.int64
        assert op.results[0].type.shape == (4,)


def test_argmax_ties_pick_first_index():
    def f(x):
        return etl.argmax(x, axis=0)

    x = np.array([[5.0, 1.0, 2.0], [5.0, 3.0, 4.0]])
    np.testing.assert_array_equal(run_numpy(f, x), np.argmax(x, axis=0))


@pytest.mark.parametrize("op_name", ["argmax", "argmin"])
def test_argmax_argmin_errors(op_name):
    op = getattr(etl, op_name)
    with pytest.raises(etl.ShapeError,
                       match=f"{op_name}: axis 3 out of range for rank 2"):
        etl.trace(lambda x: op(x, axis=3), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError,
                       match=f"{op_name}: axis must be None or an int, got 0.5"):
        etl.trace(lambda x: op(x, axis=0.5), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError,
                       match=f"{op_name}: keepdims must be a bool, got 1"):
        etl.trace(lambda x: op(x, axis=0, keepdims=1),
                  etl.TensorSpec((2, 3), etl.float32))


# ---------------------------------------------------------------------------
# stop_gradient
# ---------------------------------------------------------------------------

def test_stop_gradient_identity_barrier():
    def f(x):
        return etl.stop_gradient(x)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), etl.float32))
    assert tuple(out.shape) == (2, 3)
    assert out.dtype == np.float32
    op = _op(graph, "stop_gradient")
    assert op.name == "stop_gradient"
    assert op.effect == "pure"
    assert op.results[0].type.shape == (2, 3)
    assert op.results[0].type.dtype == np.float32

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(run_numpy(f, x), x)


def test_stop_gradient_passes_value_through():
    def f(x):
        y = etl.multiply(x, x)
        return etl.stop_gradient(y)

    graph = trace_fn(f, etl.TensorSpec((2, 3), etl.float32))
    assert [op.name for op in ops_of(graph)] == ["multiply", "stop_gradient", "return"]
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(run_numpy(f, x), x * x)


# ---------------------------------------------------------------------------
# eigh
# ---------------------------------------------------------------------------

def _symmetric(n, rng, dtype):
    a = rng.standard_normal((n, n))
    return ((a + a.T) / 2 + np.eye(n) * n).astype(dtype)


EIGH_CASES = [(2, np.float32), (3, np.float64), (2, np.int64), (2, np.complex64)]


@pytest.mark.parametrize("n,dtype", EIGH_CASES)
def test_eigh_numerics_vs_numpy(n, dtype):
    def f(x):
        return etl.eigh(x)

    rng = np.random.default_rng(3)
    x = _symmetric(n, rng, dtype)
    if dtype == np.complex64:
        x = x.astype(np.complex64) + 1j * rng.standard_normal((n, n)).astype(
            np.float32
        )
        x = (x + x.conj().T) / 2
    w, v = run_numpy(f, x)
    wn, vn = np.linalg.eigh(x)
    # w ascending — numpy order; v may differ by column sign/phase.
    np.testing.assert_allclose(w, wn, rtol=1e-5, atol=1e-5)
    recon = v @ np.diag(w) @ v.conj().T
    np.testing.assert_allclose(recon, x, rtol=1e-4, atol=1e-4)


def test_eigh_dtype_rules():
    def f(x):
        return etl.eigh(x)

    w64, v64 = run_numpy(f, np.eye(2, dtype=np.float64))
    assert w64.dtype == np.float64 and v64.dtype == np.float64
    w32, v32 = run_numpy(f, np.eye(2, dtype=np.float32))
    assert w32.dtype == np.float32 and v32.dtype == np.float32
    wi, vi = run_numpy(f, np.eye(2, dtype=np.int64))
    assert wi.dtype == np.float64 and vi.dtype == np.float64  # int -> float64
    wc, vc = run_numpy(f, np.eye(2, dtype=np.complex64))
    assert wc.dtype == np.float32 and vc.dtype == np.complex64


def test_eigh_batched_vs_numpy():
    def f(x):
        return etl.eigh(x)

    rng = np.random.default_rng(7)
    batch = np.stack([_symmetric(2, rng, np.float64) for _ in range(3)])
    w, v = run_numpy(f, batch)
    wn, vn = np.linalg.eigh(batch)
    assert w.shape == (3, 2) and v.shape == (3, 2, 2)
    np.testing.assert_allclose(w, wn, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(v[0] @ np.diag(w[0]) @ v[0].T, batch[0],
                               rtol=1e-10, atol=1e-10)


def test_eigh_ir_result_types():
    def f(x):
        return etl.eigh(x)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 2), etl.float32))
    assert isinstance(out, tuple) and len(out) == 2
    w, v = out
    assert tuple(w.shape) == (2,) and w.dtype == np.float32
    assert tuple(v.shape) == (2, 2) and v.dtype == np.float32
    op = _op(graph, "eigh")
    assert len(op.results) == 2
    assert op.results[0].type.shape == (2,)
    assert op.results[1].type.shape == (2, 2)


def test_eigh_non_square_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="must be square"):
        etl.trace(lambda x: etl.eigh(x), etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="rank >= 2"):
        etl.trace(lambda x: etl.eigh(x), etl.TensorSpec((2,), etl.float32))


def test_eigh_inside_defn_graph():
    @etl.defn
    def f(x):
        w, v = etl.eigh(x)
        return etl.solve(x, w)

    x = np.array([[4.0, 1.0], [1.0, 3.0]])
    out = etl.evaluate(f, x)
    wn, _ = np.linalg.eigh(x)
    np.testing.assert_allclose(out.numpy(), np.linalg.solve(x, wn),
                               rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# cholesky
# ---------------------------------------------------------------------------

CHOLESKY_CASES = [(2, np.float32), (3, np.float64), (2, np.int64)]


@pytest.mark.parametrize("n,dtype", CHOLESKY_CASES)
def test_cholesky_numerics_vs_numpy(n, dtype):
    def f(x):
        return etl.cholesky(x)

    rng = np.random.default_rng(11)
    a = rng.standard_normal((n, n))
    pd = (a @ a.T + np.eye(n) * n).astype(dtype)
    c = run_numpy(f, pd)
    cn = np.linalg.cholesky(pd.astype(np.float64) if dtype == np.int64 else pd)
    np.testing.assert_allclose(c, cn, rtol=1e-5, atol=1e-5)
    assert c.dtype == (np.float64 if dtype == np.int64 else np.dtype(dtype))


def test_cholesky_lower_triangular_reconstructs():
    def f(x):
        return etl.cholesky(x)

    rng = np.random.default_rng(13)
    a = rng.standard_normal((3, 3))
    pd = a @ a.T + np.eye(3)
    c = run_numpy(f, pd)
    assert np.allclose(c, np.tril(c))  # lower triangular
    np.testing.assert_allclose(c @ c.T, pd, rtol=1e-12, atol=1e-12)


def test_cholesky_batched_vs_numpy():
    def f(x):
        return etl.cholesky(x)

    rng = np.random.default_rng(17)
    batch = np.stack(
        [a @ a.T + np.eye(2) for a in rng.standard_normal((3, 2, 2))]
    )
    c = run_numpy(f, batch)
    cn = np.linalg.cholesky(batch)
    assert c.shape == (3, 2, 2)
    np.testing.assert_allclose(c, cn, rtol=1e-10, atol=1e-10)


def test_cholesky_non_pd_raises_linalg_error():
    def f(x):
        return etl.cholesky(x)

    with pytest.raises(np.linalg.LinAlgError):
        run_numpy(f, np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_cholesky_non_square_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="must be square"):
        etl.trace(lambda x: etl.cholesky(x),
                  etl.TensorSpec((2, 3), etl.float32))


def test_cholesky_inside_defn_graph():
    @etl.defn
    def f(x):
        c = etl.cholesky(x)
        return etl.dot(c, etl.transpose(c))

    x = np.array([[4.0, 1.0], [1.0, 3.0]])
    out = etl.evaluate(f, x)
    np.testing.assert_allclose(out.numpy(), x, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# qr
# ---------------------------------------------------------------------------

QR_CASES = [((3, 2), np.float32), ((2, 3), np.float64), ((2, 2), np.int64)]


@pytest.mark.parametrize("shape,dtype", QR_CASES)
def test_qr_numerics_vs_numpy(shape, dtype):
    def f(x):
        return etl.qr(x)

    rng = np.random.default_rng(19)
    x = rng.standard_normal(shape).astype(dtype)
    q, r = run_numpy(f, x)
    qn, rn = np.linalg.qr(x)
    assert q.shape == qn.shape and r.shape == rn.shape
    np.testing.assert_allclose(q, qn, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r, rn, rtol=1e-5, atol=1e-5)
    # reconstruction + orthonormality
    np.testing.assert_allclose(q @ r, x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), rtol=1e-5,
                               atol=1e-5)


def test_qr_reduced_mode_shapes():
    def f(x):
        return etl.qr(x)

    q, r = run_numpy(f, np.ones((4, 2), dtype=np.float64))
    assert q.shape == (4, 2) and r.shape == (2, 2)
    q, r = run_numpy(f, np.ones((2, 4), dtype=np.float64))
    assert q.shape == (2, 2) and r.shape == (2, 4)


def test_qr_batched_vs_numpy():
    def f(x):
        return etl.qr(x)

    rng = np.random.default_rng(23)
    batch = rng.standard_normal((3, 2, 2))
    q, r = run_numpy(f, batch)
    qn, rn = np.linalg.qr(batch)
    assert q.shape == (3, 2, 2) and r.shape == (3, 2, 2)
    np.testing.assert_allclose(q, qn, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r, rn, rtol=1e-10, atol=1e-10)


def test_qr_ir_result_types():
    def f(x):
        return etl.qr(x)

    graph, out = _trace_capturing(f, etl.TensorSpec((3, 2), etl.float32))
    q, r = out
    assert tuple(q.shape) == (3, 2) and tuple(r.shape) == (2, 2)
    op = _op(graph, "qr")
    assert len(op.results) == 2


def test_qr_inside_defn_graph():
    @etl.defn
    def f(x):
        q, r = etl.qr(x)
        return etl.dot(q, r)

    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = etl.evaluate(f, x)
    np.testing.assert_allclose(out.numpy(), x, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# matrix_rank
# ---------------------------------------------------------------------------

def test_matrix_rank_known_values():
    def f(x):
        return etl.matrix_rank(x)

    assert run_numpy(f, np.eye(3, dtype=np.float32)) == 3
    assert run_numpy(f, np.zeros((3, 3), dtype=np.float32)) == 0
    low = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 3.0]])
    assert run_numpy(f, low) == 2
    assert run_numpy(f, np.eye(2, dtype=np.int64)) == 2


def test_matrix_rank_dtype_is_int64():
    def f(x):
        return etl.matrix_rank(x)

    out = run_numpy(f, np.eye(2, dtype=np.float32))
    assert out.dtype == np.int64
    assert out.shape == ()  # scalar for 2-D input


def test_matrix_rank_batched():
    def f(x):
        return etl.matrix_rank(x)

    x = np.stack([np.eye(2), np.zeros((2, 2)), np.ones((2, 2))])
    out = run_numpy(f, x)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, [2, 0, 1])


def test_matrix_rank_tol():
    def f(x):
        return etl.matrix_rank(x, tol=0.9)

    small = np.diag([1.0, 0.5])
    assert run_numpy(f, small) == 1  # singular value 0.5 < tol
    def g(x):
        return etl.matrix_rank(x)
    assert run_numpy(g, small) == 2  # auto threshold keeps it


def test_matrix_rank_inside_defn_graph():
    @etl.defn
    def f(x):
        return etl.matrix_rank(x)

    x = np.eye(2, dtype=np.float32)
    out = etl.evaluate(f, x)
    assert int(out.numpy()) == 2


# ---------------------------------------------------------------------------
# svd
# ---------------------------------------------------------------------------

SVD_CASES = [((3, 2), np.float32), ((2, 3), np.float64), ((2, 2), np.int64),
             ((2, 2), np.complex64)]


@pytest.mark.parametrize("shape,dtype", SVD_CASES)
def test_svd_numerics_vs_numpy(shape, dtype):
    def f(x):
        return etl.svd(x)

    rng = np.random.default_rng(29)
    x = rng.standard_normal(shape).astype(dtype)
    if dtype == np.complex64:
        x = x + 1j * rng.standard_normal(shape).astype(np.float32)
    u, s, vh = run_numpy(f, x)
    un, sn, vhn = np.linalg.svd(x, full_matrices=False)
    assert u.shape == un.shape and s.shape == sn.shape and vh.shape == vhn.shape
    np.testing.assert_allclose(s, sn, rtol=1e-5, atol=1e-5)
    recon = u @ (s[..., :, None] * vh) if x.ndim == 2 else \
        np.matmul(u * s[..., None, :], vh)
    np.testing.assert_allclose(recon, x, rtol=1e-4, atol=1e-4)


def test_svd_dtype_rules():
    def f(x):
        return etl.svd(x)

    u32, s32, vh32 = run_numpy(f, np.eye(2, dtype=np.float32))
    assert u32.dtype == np.float32 and s32.dtype == np.float32
    uc, sc, vhc = run_numpy(f, np.eye(2, dtype=np.complex64))
    assert uc.dtype == np.complex64 and sc.dtype == np.float32
    ui, si, vhi = run_numpy(f, np.eye(2, dtype=np.int64))
    assert ui.dtype == np.float64 and si.dtype == np.float64


def test_svd_batched_vs_numpy():
    def f(x):
        return etl.svd(x)

    rng = np.random.default_rng(31)
    batch = rng.standard_normal((3, 2, 2))
    u, s, vh = run_numpy(f, batch)
    un, sn, vhn = np.linalg.svd(batch, full_matrices=False)
    assert u.shape == (3, 2, 2) and s.shape == (3, 2) and vh.shape == (3, 2, 2)
    np.testing.assert_allclose(s, sn, rtol=1e-10, atol=1e-10)


def test_svd_inside_defn_graph():
    @etl.defn
    def f(x):
        u, s, vh = etl.svd(x)
        return s

    x = np.array([[3.0, 0.0], [0.0, 1.0]])
    out = etl.evaluate(f, x)
    np.testing.assert_allclose(out.numpy(), [3.0, 1.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# matrix_exp
# ---------------------------------------------------------------------------

def _expm_series_ref(a, terms=60):
    """Independent pure-numpy reference: Taylor series of exp(A/2^s) with
    scaling-and-squaring, computed in float64 (different series order and
    scaling policy than the kernel — an oracle, not an alias)."""
    work = a.astype(np.float64 if a.dtype.kind != "c" else np.complex128)
    norm = np.abs(work).sum(axis=-2).max()
    s = max(int(np.ceil(np.log2(max(norm, 1.0)))), 0)
    b = work * 0.5 ** s
    n = b.shape[-1]
    term = np.eye(n, dtype=b.dtype)
    result = term
    for k in range(1, terms):
        term = term @ b / k
        result = result + term
    for _ in range(s):
        result = result @ result
    return result.astype(a.dtype)


def test_matrix_exp_known_values():
    def f(x):
        return etl.matrix_exp(x)

    # exp([[0,1],[1,0]]) = [[cosh(1), sinh(1)], [sinh(1), cosh(1)]]
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = run_numpy(f, x)
    expected = np.array([[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]])
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
    # exp(lambda I) = e^lambda I
    lam = 2.0
    out = run_numpy(f, np.eye(2) * lam)
    np.testing.assert_allclose(out, np.exp(lam) * np.eye(2), rtol=1e-12)
    # exp(0) = I
    np.testing.assert_allclose(run_numpy(f, np.zeros((2, 2))), np.eye(2),
                               rtol=1e-12)
    # rotation: exp([[0,-t],[t,0]]) = [[cos t, -sin t], [sin t, cos t]]
    t = 0.7
    rot = run_numpy(f, np.array([[0.0, -t], [t, 0.0]]))
    np.testing.assert_allclose(
        rot, [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]],
        rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_matrix_exp_vs_series_reference(dtype):
    def f(x):
        return etl.matrix_exp(x)

    rng = np.random.default_rng(37)
    x = rng.standard_normal((3, 3)).astype(dtype)
    if dtype == np.complex64:
        x = x + 1j * rng.standard_normal((3, 3)).astype(np.float32)
    out = run_numpy(f, x)
    ref = _expm_series_ref(x)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    assert out.dtype == np.dtype(dtype)


def test_matrix_exp_dtype_rule():
    def f(x):
        return etl.matrix_exp(x)

    out = run_numpy(f, np.eye(2, dtype=np.int64))
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.exp(1.0) * np.eye(2), rtol=1e-12)


def test_matrix_exp_batched():
    def f(x):
        return etl.matrix_exp(x)

    x = np.stack([np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2) * 2.0])
    out = run_numpy(f, x)
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(
        out[0], [[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]],
        rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out[1], np.exp(2.0) * np.eye(2), rtol=1e-12)


def test_matrix_exp_symmetric_matches_eigh_composition():
    """Symmetric matrices: exp(A) = V diag(exp(w)) V^T — the composition
    consumers rely on (e.g. CMAES-style updates)."""
    def f(x):
        return etl.matrix_exp(x)

    rng = np.random.default_rng(41)
    a = rng.standard_normal((3, 3))
    sym = a @ a.T
    out = run_numpy(f, sym)
    w, v = np.linalg.eigh(sym)
    expected = v @ np.diag(np.exp(w)) @ v.T
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)


def test_matrix_exp_non_square_raises_shape_error():
    with pytest.raises(etl.ShapeError, match="must be square"):
        etl.trace(lambda x: etl.matrix_exp(x),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="rank >= 2"):
        etl.trace(lambda x: etl.matrix_exp(x),
                  etl.TensorSpec((2,), etl.float32))


def test_matrix_exp_inside_defn_graph():
    @etl.defn
    def f(x):
        return etl.matrix_exp(x)

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = etl.evaluate(f, x)
    np.testing.assert_allclose(
        out.numpy(), [[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]],
        rtol=1e-12, atol=1e-12)
