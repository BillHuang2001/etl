"""Numerical-correctness suite for the numpy interpreter backend.

Every test traces a graph with ``@etl.defn``, executes it through the numpy
backend (``etl.evaluate`` — the documented trace→lower→compile→load→run
shorthand, see ``TestPipelineShorthand``), and compares the result against a
direct numpy reference computed with the SAME formula as the kernel.

The source of truth for exact semantics is ``etl/backends/numpy/kernels/``
(elementwise.py, reductions.py, indexing.py, linalg.py). Each reference here
re-implements the kernel formula explicitly — ``math.erf`` for ``erf``, the
erf-based ``gelu``, manual cross-correlation loops for ``conv``, the kernel's
index normalization for ``scatter`` — and never imports etl kernel code.

Conventions:
- Small shapes only, deterministic data (no RNG), CPU only, fast (<2s).
- Integer and bool results are compared with EXACT equality; floats with
  tight tolerances (float32: rtol=atol=1e-6; float64: rtol=atol=1e-12).
- Result dtypes are asserted against the reference dtype as well — dtype
  contracts (promotions, int64 argmax, float64 true-divide of ints, ...)
  are part of the semantics.

Callbacks/collectives/control-flow/stablehlo/symbolic-dims/persistence are
covered by the other files in this directory — not duplicated here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import etl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

F32_TOL = dict(rtol=1e-6, atol=1e-6)
F64_TOL = dict(rtol=1e-12, atol=1e-12)


def run_graph(fn, *args):
    """Execute a defn through the numpy backend (evaluate shorthand)."""
    return etl.evaluate(fn, *args, backend=etl.backends.numpy_backend)


def as_np(result):
    """Unwrap etl.Tensor results (or tuples thereof) into numpy values."""
    if isinstance(result, etl.Tensor):
        return result.numpy()
    return tuple(as_np(r) for r in result)


def assert_close(actual, expected):
    """Exact equality for int/bool, tight allclose for floats (dtype-aware)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (
        f"dtype mismatch: got {actual.dtype}, expected {expected.dtype}"
    )
    if actual.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    elif actual.dtype == np.float32:
        np.testing.assert_allclose(actual, expected, **F32_TOL)
    else:
        np.testing.assert_allclose(actual, expected, **F64_TOL)


def lin(shape, lo=-1.5, hi=1.5, dtype=np.float32):
    """Deterministic linearly-spaced array over [lo, hi] (no RNG)."""
    n = int(np.prod(shape)) if shape else 1
    return np.linspace(lo, hi, n).reshape(shape).astype(dtype)


def lin_pos(shape, lo=0.5, hi=3.0, dtype=np.float32):
    """Positive-valued array — safe for log/sqrt/divide denominators."""
    return lin(shape, lo, hi, dtype)


def ints(shape, lo=-3, hi=4, dtype=np.int32):
    """Deterministic small-int array (values in [lo, hi) — no overflow)."""
    n = int(np.prod(shape)) if shape else 1
    return (np.arange(n) % (hi - lo) + lo).reshape(shape).astype(dtype)


def make_binary(op_fn):
    """Factory: a two-input defn calling ``op_fn`` (for parametrization)."""

    @etl.defn
    def fn(a, b):
        return op_fn(a, b)

    return fn


def make_unary(op_fn):
    """Factory: a one-input defn calling ``op_fn`` (for parametrization)."""

    @etl.defn
    def fn(x):
        return op_fn(x)

    return fn


# --- formula references (same formulas as the kernels) -----------------------


def ref_erf(x):
    """``math.erf`` vectorized (the kernel's frompyfunc approach)."""
    return np.frompyfunc(math.erf, 1, 1)(x).astype(x.dtype)


def ref_gelu(x):
    """erf-based gelu 0.5 * x * (1 + erf(x / sqrt(2))) (kernel formula).

    Computed in float64 like the kernel's ``math.erf``, then cast back to the
    operand dtype — the kernel's computation is dtype-stable."""
    x64 = np.asarray(x, dtype=np.float64)
    return (0.5 * x64 * (1.0 + ref_erf(x64 / math.sqrt(2.0)))).astype(x.dtype)


def ref_sigmoid(x):
    """1 / (1 + exp(-x)) (kernel formula), cast back to the operand dtype."""
    x64 = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x64))).astype(x.dtype)


def _per_spatial(value, n_spatial, default, name):
    """Normalize an int-or-tuple conv parameter (kernel's ``_per_dim_tuple``)."""
    if value is None:
        return (default,) * n_spatial
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return (int(value),) * n_spatial
    if isinstance(value, (tuple, list)):
        seq = tuple(int(v) for v in value)
        if len(seq) != n_spatial:
            raise ValueError(f"{name}: expected {n_spatial} entries")
        return seq
    raise TypeError(name)


def _dilate(arr, rates):
    """Insert ``rate - 1`` zeros between elements along the trailing dims
    (kernel ``_dilate_spatial``: effective size ``(d - 1) * rate + 1``)."""
    n_spatial = len(rates)
    new_shape = list(arr.shape[:-n_spatial]) + [
        (arr.shape[-n_spatial + i] - 1) * rates[i] + 1 for i in range(n_spatial)
    ]
    out = np.zeros(new_shape, dtype=arr.dtype)
    out[(slice(None),) * (arr.ndim - n_spatial) + tuple(
        slice(None, None, rates[i]) for i in range(n_spatial)
    )] = arr
    return out


def ref_conv(x, w, strides=1, padding="VALID", input_dilation=1,
             kernel_dilation=1, feature_groups=1):
    """Manual NCHW cross-correlation implementing the exact semantics of
    ``etl/backends/numpy/kernels/linalg.py`` ``_conv``:

    - effective sizes ``(d - 1) * dil + 1`` for BOTH input and kernel
      (dilation inserts zeros),
    - the input is dilated FIRST, then padded (``lo``/``hi`` zeros on the
      dilated input; ``"SAME"``: out dim ``ceil(d / stride)``, total pad
      ``(out - 1) * stride + eff_k - eff_d`` split lo = total // 2),
    - output ``out[n, co, i...] = sum_{ci, ki...} x_p[n, ci, i*stride+ki]
      * w_d[co, ci, ki]`` per contiguous channel group.
    """
    x = np.asarray(x)
    w = np.asarray(w)
    n_spatial = x.ndim - 2
    strides = _per_spatial(strides, n_spatial, 1, "strides")
    in_dil = _per_spatial(input_dilation, n_spatial, 1, "input_dilation")
    k_dil = _per_spatial(kernel_dilation, n_spatial, 1, "kernel_dilation")
    x_d = _dilate(x, in_dil)
    w_d = _dilate(w, k_dil)
    eff_k = w_d.shape[2:]
    n, c_in = x.shape[:2]
    c_out = w.shape[0]
    gin, gout = c_in // feature_groups, c_out // feature_groups
    pad_pairs = []
    out_spatial = []
    for i in range(n_spatial):
        d = x.shape[2 + i]
        eff_d = x_d.shape[2 + i]
        if padding == "SAME":
            out_i = (d + strides[i] - 1) // strides[i]
            total = (out_i - 1) * strides[i] + eff_k[i] - eff_d
            pad_pairs.append((total // 2, total - total // 2))
        elif padding == "VALID":
            pad_pairs.append((0, 0))
        elif isinstance(padding, (tuple, list)) and len(padding) == n_spatial:
            entry = padding[i]
            if isinstance(entry, (tuple, list)):
                pad_pairs.append((int(entry[0]), int(entry[1])))
            else:
                pad_pairs.append((int(entry), int(entry)))
        else:
            raise ValueError(padding)
        out_spatial.append(
            (eff_d + pad_pairs[-1][0] + pad_pairs[-1][1] - eff_k[i])
            // strides[i] + 1
        )
    x_p = np.pad(
        x_d,
        ((0, 0), (0, 0)) + tuple((max(lo, 0), max(hi, 0)) for lo, hi in pad_pairs),
    )
    if any(lo < 0 or hi < 0 for lo, hi in pad_pairs):
        x_p = x_p[(slice(None), slice(None)) + tuple(
            slice(-lo if lo < 0 else None, hi if hi < 0 else None)
            for lo, hi in pad_pairs
        )]
    out = np.zeros(
        (n, c_out) + tuple(out_spatial),
        dtype=np.result_type(x.dtype, w.dtype),
    )
    for ni in range(n):
        for g in range(feature_groups):
            for co in range(gout):
                for idx in np.ndindex(tuple(out_spatial)):
                    acc = np.zeros((), dtype=np.result_type(x.dtype, w.dtype))
                    for ci in range(gin):
                        for ki in np.ndindex(tuple(eff_k)):
                            pos = tuple(idx[j] * strides[j] + ki[j]
                                        for j in range(n_spatial))
                            acc += x_p[ni, g * gin + ci][pos] * w_d[g * gout + co, ci][ki]
                    out[ni, g * gout + co][idx] = acc
    return out


def ref_scatter(x, indices, updates, axis):
    """Scatter reference replicating the kernel's index normalization
    (``etl/backends/numpy/kernels/indexing.py`` ``_scatter``): indices with
    fewer dims than ``x`` are padded to full rank with leading/trailing
    size-1 axes around ``axis``, then numpy ``put_along_axis`` on a copy."""
    out = np.array(x, copy=True)
    idx = np.asarray(indices)
    rank = x.ndim
    if idx.ndim == 0:
        idx = idx.reshape((1,) * rank)
    elif idx.ndim < rank:
        idx = idx.reshape((1,) * axis + idx.shape + (1,) * (rank - axis - 1))
    np.put_along_axis(out, idx, np.asarray(updates), axis=axis)
    return out


# ---------------------------------------------------------------------------
# 1. Elementwise arithmetic (two-input)
# ---------------------------------------------------------------------------

_BINARY_OPS = [
    ("add", etl.add, np.add),
    ("subtract", etl.subtract, np.subtract),
    ("multiply", etl.multiply, np.multiply),
    ("divide", etl.divide, np.divide),
    ("power", etl.power, np.power),
    ("maximum", etl.maximum, np.maximum),
    ("minimum", etl.minimum, np.minimum),
]


def _binary_data(name, dtype):
    """Domain-safe input pair per op (positive denominators for divide,
    positive bases for power)."""
    if name == "divide":
        return lin((2, 3), dtype=dtype), lin_pos((2, 3), dtype=dtype)
    if name == "power":
        return lin_pos((2, 3), dtype=dtype), lin((2, 3), -0.5, 2.0, dtype=dtype)
    return lin((2, 3), dtype=dtype), lin((2, 3), -2.0, 2.0, dtype=dtype)


class TestElementwiseBinary:
    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _BINARY_OPS,
        ids=[name for name, _, _ in _BINARY_OPS],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_same_shape(self, name, op_fn, np_fn, dtype):
        a, b = _binary_data(name, dtype)
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _BINARY_OPS,
        ids=[name for name, _, _ in _BINARY_OPS],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_broadcasting_rows_against_column(self, name, op_fn, np_fn, dtype):
        a = lin_pos((3,), dtype=dtype)
        if name == "divide":
            b = lin_pos((3, 1), dtype=dtype)  # non-zero denominators
        else:
            b = lin((3, 1), -2.0, 2.0, dtype=dtype)
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _BINARY_OPS,
        ids=[name for name, _, _ in _BINARY_OPS],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_0d_tensor_operand(self, name, op_fn, np_fn, dtype):
        a, _ = _binary_data(name, dtype)
        b = np.asarray(1.25, dtype=dtype)  # 0-d tensor operand
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _BINARY_OPS,
        ids=[name for name, _, _ in _BINARY_OPS],
    )
    def test_python_scalar_operand(self, name, op_fn, np_fn):
        # NEP-50 weak promotion: python int/float scalars keep the array
        # dtype. Scalars are baked into the defn body (evaluate only accepts
        # tensor arguments — scalars are static values at trace time).
        a = lin_pos((2, 3), dtype=np.float32)

        @etl.defn
        def fn_int(t):
            return op_fn(t, 2)

        @etl.defn
        def fn_float(t):
            return op_fn(t, 1.5)

        out_int = run_graph(fn_int, a)
        out_float = run_graph(fn_float, a)
        assert_close(as_np(out_int), np_fn(a, 2))
        assert_close(as_np(out_float), np_fn(a, 1.5))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn",
        [
            ("add", etl.add, np.add),
            ("subtract", etl.subtract, np.subtract),
            ("multiply", etl.multiply, np.multiply),
            ("maximum", etl.maximum, np.maximum),
            ("minimum", etl.minimum, np.minimum),
        ],
        ids=["add", "subtract", "multiply", "maximum", "minimum"],
    )
    def test_int32_exact(self, name, op_fn, np_fn):
        a = ints((2, 3), lo=-3, hi=4)
        b = ints((2, 3), lo=0, hi=4)
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))

    def test_remainder_int32_exact(self):
        a = ints((2, 3), lo=-7, hi=8)
        b = ints((2, 3), lo=1, hi=6)
        out = run_graph(make_binary(etl.remainder), a, b)
        assert_close(as_np(out), np.remainder(a, b))

    def test_divide_int_true_division(self):
        # true division: int64 / int64 -> float64 (numpy rule, exact here).
        a = np.array([1, 6, 7], np.int32)
        b = np.array([2, 2, 2], np.int32)
        out = run_graph(make_binary(etl.divide), a, b)
        assert_close(as_np(out), np.divide(a, b))

    def test_power_int32_exact(self):
        a = np.array([1, 2, 3, -2], np.int32)
        b = np.array([2, 3, 2, 3], np.int32)
        out = run_graph(make_binary(etl.power), a, b)
        assert_close(as_np(out), np.power(a, b))


# ---------------------------------------------------------------------------
# 1b. Elementwise unary
# ---------------------------------------------------------------------------

_UNARY_NUMPY = [
    ("abs", etl.abs, np.abs, "mixed"),
    ("negate", etl.negate, np.negative, "mixed"),
    ("square", etl.square, np.square, "mixed"),
    ("sqrt", etl.sqrt, np.sqrt, "positive"),
    ("exp", etl.exp, np.exp, "mixed"),
    ("log", etl.log, np.log, "positive"),
    ("log1p", etl.log1p, np.log1p, "gt_minus_one"),
    ("sin", etl.sin, np.sin, "mixed"),
    ("cos", etl.cos, np.cos, "mixed"),
    ("tan", etl.tan, np.tan, "bounded"),
    ("tanh", etl.tanh, np.tanh, "mixed"),
    ("sign", etl.sign, np.sign, "mixed"),
]

# unary ops with activation-style formulas (custom references)
_UNARY_FORMULA = [
    ("sigmoid", etl.sigmoid, ref_sigmoid),
    ("relu", etl.relu, lambda x: np.maximum(x, 0)),
    ("gelu", etl.gelu, ref_gelu),
    ("erf", etl.erf, ref_erf),
]


def _unary_data(domain, dtype):
    if domain == "positive":
        return lin_pos((2, 4), dtype=dtype)
    if domain == "gt_minus_one":
        return lin((2, 4), -0.5, 2.0, dtype=dtype)
    if domain == "bounded":
        return lin((2, 4), -1.0, 1.0, dtype=dtype)
    return lin((2, 4), -2.0, 2.0, dtype=dtype)


class TestElementwiseUnary:
    @pytest.mark.parametrize(
        "name,op_fn,np_fn,domain", _UNARY_NUMPY,
        ids=[name for name, _, _, _ in _UNARY_NUMPY],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_numpy_math(self, name, op_fn, np_fn, domain, dtype):
        x = _unary_data(domain, dtype)
        out = run_graph(make_unary(op_fn), x)
        assert_close(as_np(out), np_fn(x))

    @pytest.mark.parametrize(
        "name,op_fn,ref", _UNARY_FORMULA,
        ids=[name for name, _, _ in _UNARY_FORMULA],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_activation_formulas(self, name, op_fn, ref, dtype):
        # gelu/erf go through math.erf (kernel formula) — the reference
        # implements the same formula, not numpy's approximation of it.
        x = lin((2, 4), -3.0, 3.0, dtype=dtype)
        out = run_graph(make_unary(op_fn), x)
        assert_close(as_np(out), ref(x))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn",
        [
            ("abs", etl.abs, np.abs),
            ("negate", etl.negate, np.negative),
            ("square", etl.square, np.square),
            ("sign", etl.sign, np.sign),
        ],
        ids=["abs", "negate", "square", "sign"],
    )
    def test_int32_exact(self, name, op_fn, np_fn):
        x = ints((2, 3), lo=-3, hi=4)
        out = run_graph(make_unary(op_fn), x)
        assert_close(as_np(out), np_fn(x))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn",
        [
            ("sqrt", etl.sqrt, np.sqrt),
            ("exp", etl.exp, np.exp),
            ("log", etl.log, np.log),
            ("tanh", etl.tanh, np.tanh),
        ],
        ids=["sqrt", "exp", "log", "tanh"],
    )
    def test_int_input_promotes_to_float64(self, name, op_fn, np_fn):
        # unary-math dtype rule: integral input -> float64.
        x = ints((3,), lo=1, hi=5)
        out = run_graph(make_unary(op_fn), x)
        assert_close(as_np(out), np_fn(x))


# ---------------------------------------------------------------------------
# 2. Comparisons, logical, bitwise, cast
# ---------------------------------------------------------------------------

_COMPARISON_OPS = [
    ("equal", etl.equal, np.equal),
    ("not_equal", etl.not_equal, np.not_equal),
    ("less", etl.less, np.less),
    ("less_equal", etl.less_equal, np.less_equal),
    ("greater", etl.greater, np.greater),
    ("greater_equal", etl.greater_equal, np.greater_equal),
]


class TestComparisonsLogicalBitwiseCast:
    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _COMPARISON_OPS,
        ids=[name for name, _, _ in _COMPARISON_OPS],
    )
    @pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
    def test_comparisons(self, name, op_fn, np_fn, dtype):
        a = lin((2, 3), dtype=dtype) if np.dtype(dtype).kind == "f" else ints((2, 3))
        b = lin((2, 3), -1.0, 1.0, dtype=dtype) if np.dtype(dtype).kind == "f" else ints((2, 3), lo=0, hi=4)
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))  # bool outputs — exact

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _COMPARISON_OPS,
        ids=[name for name, _, _ in _COMPARISON_OPS],
    )
    def test_comparison_broadcast_and_scalar(self, name, op_fn, np_fn):
        a = lin((3,), dtype=np.float32)
        b = lin((3, 1), -1.0, 1.0, dtype=np.float32)

        @etl.defn
        def fn_scalar(t):
            return op_fn(t, 0.0)

        out_bc = run_graph(make_binary(op_fn), a, b)
        out_sc = run_graph(fn_scalar, a)
        assert_close(as_np(out_bc), np_fn(a, b))
        assert_close(as_np(out_sc), np_fn(a, 0.0))

    def test_logical_binary(self):
        a = np.array([[True, False, True], [False, True, False]])
        b = np.array([[True, True, False], [False, False, True]])
        out_and = run_graph(make_binary(etl.logical_and), a, b)
        out_or = run_graph(make_binary(etl.logical_or), a, b)
        assert_close(as_np(out_and), np.logical_and(a, b))
        assert_close(as_np(out_or), np.logical_or(a, b))

    def test_logical_not(self):
        a = np.array([[True, False], [False, True]])
        out = run_graph(make_unary(etl.logical_not), a)
        assert_close(as_np(out), np.logical_not(a))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn",
        [
            ("bitwise_and", etl.bitwise_and, np.bitwise_and),
            ("bitwise_or", etl.bitwise_or, np.bitwise_or),
            ("bitwise_xor", etl.bitwise_xor, np.bitwise_xor),
        ],
        ids=["bitwise_and", "bitwise_or", "bitwise_xor"],
    )
    def test_bitwise_int32(self, name, op_fn, np_fn):
        a = np.array([[0b1100, 0b0011], [0b1010, 0b1111]], np.int32)
        b = np.array([[0b0110, 0b1001], [0b0011, 0b0101]], np.int32)
        out = run_graph(make_binary(op_fn), a, b)
        assert_close(as_np(out), np_fn(a, b))

    @pytest.mark.parametrize(
        "src,target",
        [
            (np.float32, np.float64),
            (np.float64, np.float32),
            (np.float32, np.int32),
            (np.float64, np.int32),
            (np.int32, np.float32),
            (np.int32, np.float64),
            (np.int32, np.int64),
            (np.int64, np.int32),
        ],
        ids=[
            "f32_to_f64", "f64_to_f32", "f32_to_i32", "f64_to_i32",
            "i32_to_f32", "i32_to_f64", "i32_to_i64", "i64_to_i32",
        ],
    )
    def test_cast(self, src, target):
        if src in (np.float32, np.float64):
            x = lin((2, 3), -3.5, 3.5, dtype=src)
        else:
            x = ints((2, 3), lo=-2, hi=3, dtype=src)

        @etl.defn
        def fn(x):
            return etl.cast(x, target)

        out = run_graph(fn, x)
        # astype semantics (truncation toward zero, wraparound) — exact.
        assert_close(as_np(out), x.astype(target))


# ---------------------------------------------------------------------------
# 3. Reductions
# ---------------------------------------------------------------------------

_REDUCERS = [
    ("sum", etl.sum, np.sum),
    ("max", etl.max, np.max),
    ("min", etl.min, np.min),
    ("mean", etl.mean, np.mean),
    ("prod", etl.prod, np.prod),
]


def _reducer_input(dtype):
    if np.dtype(dtype).kind == "f":
        return lin((2, 3, 4), -2.0, 2.0, dtype=dtype)
    return ints((2, 3, 4), lo=-2, hi=3, dtype=dtype)


class TestReductions:
    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _REDUCERS, ids=[n for n, _, _ in _REDUCERS]
    )
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    @pytest.mark.parametrize("dtype", [np.float64, np.int32])
    def test_over_axis(self, name, op_fn, np_fn, axis, dtype):
        x = _reducer_input(dtype)
        out = run_graph(lambda t: op_fn(t, axes=axis), x)
        assert_close(as_np(out), np_fn(x, axis=axis))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _REDUCERS, ids=[n for n, _, _ in _REDUCERS]
    )
    @pytest.mark.parametrize("dtype", [np.float64, np.int32])
    def test_full_reduction(self, name, op_fn, np_fn, dtype):
        x = _reducer_input(dtype)
        out = run_graph(lambda t: op_fn(t), x)
        assert_close(as_np(out), np_fn(x))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _REDUCERS, ids=[n for n, _, _ in _REDUCERS]
    )
    def test_tuple_axes(self, name, op_fn, np_fn):
        x = ints((2, 3, 4), lo=-2, hi=3)
        out = run_graph(lambda t: op_fn(t, axes=(0, 2)), x)
        assert_close(as_np(out), np_fn(x, axis=(0, 2)))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn", _REDUCERS, ids=[n for n, _, _ in _REDUCERS]
    )
    def test_keepdims(self, name, op_fn, np_fn):
        x = lin((2, 3, 4), dtype=np.float32)
        out = run_graph(lambda t: op_fn(t, axes=1, keepdims=True), x)
        assert_close(as_np(out), np_fn(x, axis=1, keepdims=True))

    def test_full_reduction_keepdims(self):
        x = lin((2, 3), dtype=np.float32)
        out = run_graph(lambda t: etl.sum(t, keepdims=True), x)
        assert_close(as_np(out), np.sum(x, keepdims=True))

    def test_reduce_ops_equal_sugar(self):
        # the sugar names expand onto the reduce_* ops — same numbers.
        x = lin((2, 3, 4), dtype=np.float32)

        @etl.defn
        def fn(t):
            return (
                etl.reduce_sum(t, axes=1),
                etl.reduce_max(t, axes=1),
                etl.reduce_min(t, axes=1),
                etl.reduce_mean(t, axes=1),
                etl.reduce_prod(t, axes=1),
            )

        got = as_np(run_graph(fn, x))
        for actual, np_fn in zip(got, (np.sum, np.max, np.min, np.mean, np.prod)):
            assert_close(actual, np_fn(x, axis=1))

    def test_mean_int_is_float64(self):
        x = ints((2, 3), lo=0, hi=5)
        out = run_graph(lambda t: etl.mean(t), x)
        assert_close(as_np(out), np.mean(x))

    def test_sum_bool_is_int64(self):
        x = np.array([[True, False, True], [False, True, True]])
        out = run_graph(lambda t: etl.sum(t), x)
        assert_close(as_np(out), np.sum(x))

    @pytest.mark.parametrize(
        "name,op_fn,np_fn",
        [("argmax", etl.argmax, np.argmax), ("argmin", etl.argmin, np.argmin)],
        ids=["argmax", "argmin"],
    )
    @pytest.mark.parametrize("axis", [0, 1, 2, -1, None])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_arg_reductions(self, name, op_fn, np_fn, axis, keepdims):
        x = lin((2, 3, 4), dtype=np.float32)  # strictly increasing → no ties

        @etl.defn
        def fn(t):
            return op_fn(t, axis=axis, keepdims=keepdims)

        out = run_graph(fn, x)
        ref = np_fn(x, axis=axis)
        if keepdims:
            ref = (
                np.reshape(ref, (1,) * x.ndim)
                if axis is None
                else np.expand_dims(ref, axis=axis)
            )
        assert_close(as_np(out), ref)


# ---------------------------------------------------------------------------
# 4. Linalg
# ---------------------------------------------------------------------------

class TestLinalg:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dot_2d(self, dtype):
        a = lin((3, 4), dtype=dtype)
        b = lin((4, 2), -2.0, 2.0, dtype=dtype)
        out = run_graph(make_binary(etl.dot), a, b)
        assert_close(as_np(out), np.matmul(a, b))

    def test_dot_batched(self):
        a = lin((2, 3, 4), dtype=np.float32)
        b = lin((1, 4, 2), -2.0, 2.0, dtype=np.float32)
        out = run_graph(make_binary(etl.dot), a, b)
        assert_close(as_np(out), np.matmul(a, b))

    def test_dot_batch_broadcast(self):
        a = lin((2, 3, 4), dtype=np.float32)
        b = lin((4, 2), -2.0, 2.0, dtype=np.float32)
        out = run_graph(make_binary(etl.dot), a, b)
        assert_close(as_np(out), np.matmul(a, b))

    def test_dot_contracting_1_broadcast(self):
        # numpy matmul k-broadcast (size-1 k) — the ops layer compensates.
        a = lin((3, 1), dtype=np.float32)
        b = lin((1, 2), -2.0, 2.0, dtype=np.float32)
        out = run_graph(make_binary(etl.dot), a, b)
        assert_close(as_np(out), np.matmul(a, b))

    def test_dot_dtype_promotion(self):
        a = lin((2, 3), dtype=np.float32)
        b = lin((3, 2), dtype=np.float64)
        out = run_graph(make_binary(etl.dot), a, b)
        assert_close(as_np(out), np.matmul(a, b))

    def test_dot_rank1_raises(self):
        with pytest.raises(etl.ShapeError):
            run_graph(make_binary(etl.dot), np.ones(3, np.float32),
                      np.ones(3, np.float32))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_solve_3x3(self, dtype):
        a = np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]],
                     dtype=dtype)
        b = lin((3,), -1.0, 1.0, dtype=dtype)
        out = run_graph(make_binary(etl.solve), a, b)
        assert_close(as_np(out), np.linalg.solve(a, b))

    def test_solve_batched_rhs_one_column(self):
        # etl's solve contract: for batched ``a`` the rhs must be
        # ``(..., n, k)`` (b[-2] is the contracting dim); numpy's bare
        # ``(batch, n)`` vector form is deliberately rejected as ambiguous.
        a = np.array(
            [[[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]],
             [[3.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 4.0]]],
            dtype=np.float32,
        )
        b = lin((2, 3, 1), -1.0, 1.0, dtype=np.float32)
        out = run_graph(make_binary(etl.solve), a, b)
        assert_close(as_np(out), np.linalg.solve(a, b))

    def test_solve_batched_matrix_rhs(self):
        a = np.tile(
            np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]],
                     dtype=np.float64),
            (2, 1, 1),
        )
        b = lin((2, 3, 2), -1.0, 1.0, dtype=np.float64)
        out = run_graph(make_binary(etl.solve), a, b)
        assert_close(as_np(out), np.linalg.solve(a, b))

    def test_solve_int_promotes_to_float64(self):
        a = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 2]], np.int32)
        b = np.array([9, 8, 7], np.int32)
        out = run_graph(make_binary(etl.solve), a, b)
        assert_close(as_np(out), np.linalg.solve(a.astype(np.float64),
                                                b.astype(np.float64)))

    def test_solve_singular_raises(self):
        a = np.array([[1.0, 2.0], [2.0, 4.0]], np.float64)
        b = np.ones(2, np.float64)
        with pytest.raises(np.linalg.LinAlgError):
            run_graph(make_binary(etl.solve), a, b)

    @pytest.mark.parametrize(
        "strides,padding,expected",
        [
            (1, "VALID", (2, 1, 3, 3)),
            (1, "SAME", (2, 1, 5, 5)),
            (2, "VALID", (2, 1, 2, 2)),
            (1, ((1, 0), (0, 1)), (2, 1, 4, 4)),
        ],
        ids=["valid", "same", "stride2", "per_axis_padding"],
    )
    def test_conv_2d(self, strides, padding, expected):
        x = lin((2, 1, 5, 5), dtype=np.float32)
        w = lin((1, 1, 3, 3), -1.0, 1.0, dtype=np.float32)

        @etl.defn
        def fn(x, w):
            return etl.conv(x, w, strides=strides, padding=padding)

        out = run_graph(fn, x, w)
        got = as_np(out)
        assert got.shape == expected
        assert_close(got, ref_conv(x, w, strides=strides, padding=padding))

    def test_conv_2d_multichannel(self):
        x = lin((2, 2, 4, 4), dtype=np.float32)
        w = lin((3, 2, 2, 2), -1.0, 1.0, dtype=np.float32)
        out = run_graph(make_binary(etl.conv), x, w)
        assert_close(as_np(out), ref_conv(x, w))

    def test_conv_1d_stride2(self):
        x = lin((2, 2, 8), dtype=np.float32)
        w = lin((3, 2, 3), -1.0, 1.0, dtype=np.float32)
        out = run_graph(lambda a, b: etl.conv(a, b, strides=2), x, w)
        assert_close(as_np(out), ref_conv(x, w, strides=2))

    def test_conv_input_dilation(self):
        x = lin((1, 1, 5, 5), dtype=np.float32)
        w = lin((1, 1, 3, 3), -1.0, 1.0, dtype=np.float32)
        out = run_graph(lambda a, b: etl.conv(a, b, input_dilation=2), x, w)
        assert_close(as_np(out), ref_conv(x, w, input_dilation=2))

    def test_conv_kernel_dilation(self):
        x = lin((1, 1, 6, 6), dtype=np.float32)
        w = lin((1, 1, 3, 3), -1.0, 1.0, dtype=np.float32)
        out = run_graph(lambda a, b: etl.conv(a, b, kernel_dilation=2), x, w)
        assert_close(as_np(out), ref_conv(x, w, kernel_dilation=2))

    def test_conv_feature_groups(self):
        x = lin((2, 4, 4, 4), dtype=np.float32)
        w = lin((2, 2, 2, 2), -1.0, 1.0, dtype=np.float32)
        out = run_graph(
            lambda a, b: etl.conv(a, b, feature_group_size=2), x, w
        )
        assert_close(as_np(out), ref_conv(x, w, feature_groups=2))

    def test_conv_channels_last(self):
        # channels_last is frontend sugar: transpose around the NCHW op.
        x = lin((2, 5, 5, 1), dtype=np.float32)
        w = lin((1, 1, 3, 3), -1.0, 1.0, dtype=np.float32)
        out = run_graph(
            lambda a, b: etl.conv(a, b, channels_last=True), x, w
        )
        got = as_np(out)
        assert got.shape == (2, 3, 3, 1)  # VALID: 5x5 @ 3x3 -> 3x3
        x_nchw = np.transpose(x, (0, 3, 1, 2))
        assert_close(got, np.transpose(ref_conv(x_nchw, w), (0, 2, 3, 1)))

    @pytest.mark.parametrize("k", [0, 1, -1])
    def test_tril(self, k):
        x = lin((3, 4), dtype=np.float32)
        out = run_graph(lambda t: etl.tril(t, k=k), x)
        assert_close(as_np(out), np.tril(x, k=k))

    @pytest.mark.parametrize("k", [0, 1, -1])
    def test_triu(self, k):
        x = lin((3, 4), dtype=np.float32)
        out = run_graph(lambda t: etl.triu(t, k=k), x)
        assert_close(as_np(out), np.triu(x, k=k))

    def test_tril_batched(self):
        x = lin((2, 3, 3), dtype=np.float32)
        out = run_graph(lambda t: etl.tril(t), x)
        assert_close(as_np(out), np.tril(x))

    def test_cumsum_float(self):
        x = lin((3, 4), dtype=np.float32)
        for axis, reverse in ((0, False), (1, False), (0, True), (-1, True)):
            out = run_graph(
                lambda t, a=axis, r=reverse: etl.cumsum(t, axis=a, reverse=r), x
            )
            ref = np.cumsum(x, axis=axis)
            if reverse:
                ref = np.flip(np.cumsum(np.flip(x, axis=axis), axis=axis), axis=axis)
            assert_close(as_np(out), ref)

    def test_cumsum_int32_preserves_dtype(self):
        # op contract: cumsum keeps the operand dtype (no numpy-2.0 upcast).
        x = ints((5,), lo=-1, hi=3)
        out = run_graph(lambda t: etl.cumsum(t, axis=0), x)
        assert as_np(out).dtype == np.int32
        assert_close(as_np(out), np.cumsum(x, axis=0, dtype=x.dtype))


# ---------------------------------------------------------------------------
# 5. Structure: reshape / transpose / slice / gather / scatter / concat /
#    pad / select / broadcast
# ---------------------------------------------------------------------------

class TestStructure:
    @pytest.mark.parametrize(
        "shape",
        [(6, 4), (2, -1), (24,), (2, 3, 4)],
        ids=["6x4", "wildcard", "flatten", "identity"],
    )
    def test_reshape(self, shape):
        x = lin((2, 3, 4), dtype=np.float32)
        out = run_graph(lambda t: etl.reshape(t, shape), x)
        assert_close(as_np(out), np.reshape(x, shape))

    def test_transpose_perm(self):
        x = lin((2, 3, 4), dtype=np.float32)
        out = run_graph(lambda t: etl.transpose(t, (2, 0, 1)), x)
        assert_close(as_np(out), np.transpose(x, (2, 0, 1)))

    def test_transpose_reverse(self):
        x = lin((2, 3, 4), dtype=np.float32)
        out = run_graph(lambda t: etl.transpose(t), x)
        assert_close(as_np(out), np.transpose(x))

    @pytest.mark.parametrize(
        "start,lengths,strides",
        [
            (1, (2, 2), 1),
            ((0, 1), (2, 3), 1),
            (1, 3, 2),
            ((0, 0), (2, 2), (1, 2)),
            (-2, 2, 1),  # Nx semantics: negative start counts from the end
        ],
        ids=["basic", "per_axis", "strided", "mixed_strides", "negative_start"],
    )
    def test_slice(self, start, lengths, strides):
        # etl.slice semantics: per-axis x[start : start + length : stride]
        # (limit = start + length, NOT start + length * stride).
        x = lin((4, 5), dtype=np.float32)
        rank = x.ndim
        starts = (start,) * rank if isinstance(start, int) else start
        lens = (lengths,) * rank if isinstance(lengths, int) else lengths
        strd = (strides,) * rank if isinstance(strides, int) else strides

        @etl.defn
        def fn(t):
            return etl.slice(t, starts, lens, strd)

        out = run_graph(fn, x)
        ref_starts = tuple(
            (s + x.shape[i]) if s < 0 else s for i, s in enumerate(starts)
        )
        ref = x[tuple(
            slice(s, s + l, st)
            for s, l, st in zip(ref_starts, lens, strd)
        )]
        assert_close(as_np(out), ref)

    def test_gather_1d_indices(self):
        x = lin((4, 3), dtype=np.float32)
        idx = np.array([2, 0, 2], np.int64)
        out = run_graph(lambda t, i: etl.gather(t, i, axis=0), x, idx)
        assert_close(as_np(out), np.take(x, idx, axis=0))

    def test_gather_2d_indices(self):
        x = lin((2, 3), dtype=np.float32)
        idx = np.array([[0, 2], [1, 0]], np.int64)
        out = run_graph(lambda t, i: etl.gather(t, i, axis=1), x, idx)
        assert_close(as_np(out), np.take(x, idx, axis=1))

    def test_gather_negative_indices(self):
        x = lin((3, 2), dtype=np.float32)
        idx = np.array([-1, 0, -3], np.int64)
        out = run_graph(lambda t, i: etl.gather(t, i, axis=0), x, idx)
        assert_close(as_np(out), np.take(x, idx, axis=0))

    def test_gather_0d_index(self):
        x = lin((3, 2), dtype=np.float32)
        idx = np.array(1, np.int64)
        out = run_graph(lambda t, i: etl.gather(t, i, axis=0), x, idx)
        assert_close(as_np(out), np.take(x, idx, axis=0))

    def test_scatter_1d(self):
        x = np.zeros(5, np.float32)
        idx = np.array([0, 2], np.int64)
        upd = np.array([5.0, 7.0], np.float32)
        out = run_graph(lambda t, i, u: etl.scatter(t, i, u, axis=0), x, idx, upd)
        assert_close(as_np(out), ref_scatter(x, idx, upd, 0))

    def test_scatter_2d_axis0(self):
        x = np.zeros((3, 2), np.float32)
        idx = np.array([2, 0], np.int64)
        upd = np.array([[5.0, 7.0], [8.0, 9.0]], np.float32)
        out = run_graph(lambda t, i, u: etl.scatter(t, i, u, axis=0), x, idx, upd)
        assert_close(as_np(out), ref_scatter(x, idx, upd, 0))

    def test_scatter_2d_axis1(self):
        x = np.zeros((2, 3), np.float32)
        idx = np.array([2, 0], np.int64)
        upd = np.array([[5.0, 7.0], [8.0, 9.0]], np.float32)
        out = run_graph(lambda t, i, u: etl.scatter(t, i, u, axis=1), x, idx, upd)
        assert_close(as_np(out), ref_scatter(x, idx, upd, 1))

    def test_scatter_does_not_mutate_input(self):
        x = np.zeros((2, 3), np.float32)
        x_before = x.copy()
        idx = np.array([2, 0], np.int64)
        upd = np.array([[5.0, 7.0], [8.0, 9.0]], np.float32)
        run_graph(lambda t, i, u: etl.scatter(t, i, u, axis=1), x, idx, upd)
        np.testing.assert_array_equal(x, x_before)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_concatenate(self, axis):
        a = lin((2, 3), dtype=np.float32)
        b = (
            lin((3, 3), -2.0, -1.0, dtype=np.float32)
            if axis == 0
            else lin((2, 2), -2.0, -1.0, dtype=np.float32)
        )
        out = run_graph(
            lambda x, y: etl.concatenate((x, y), axis=axis), a, b
        )
        assert_close(as_np(out), np.concatenate((a, b), axis=axis))

    def test_concatenate_dtype_promotion(self):
        # declared promotion np.result_type(int32, float32) == float64.
        a = np.array([1, 2], np.int32)
        b = np.array([0.5, 1.5], np.float32)
        out = run_graph(lambda x, y: etl.concatenate((x, y), axis=0), a, b)
        assert_close(as_np(out), np.concatenate((a, b), axis=0))

    def test_pad_pairs(self):
        x = lin((2, 2), dtype=np.float32)
        out = run_graph(lambda t: etl.pad(t, ((1, 2), (0, 1)), value=3.0), x)
        assert_close(
            as_np(out),
            np.pad(x, ((1, 2), (0, 1)), mode="constant", constant_values=3.0),
        )

    def test_pad_symmetric_ints(self):
        x = lin((2, 3), dtype=np.float32)
        out = run_graph(lambda t: etl.pad(t, (1, 2), value=-1.0), x)
        assert_close(
            as_np(out),
            np.pad(x, ((1, 1), (2, 2)), mode="constant", constant_values=-1.0),
        )

    def test_pad_int_value(self):
        x = ints((2, 2), lo=0, hi=4)
        out = run_graph(lambda t: etl.pad(t, (1, 0), value=7), x)
        assert_close(
            as_np(out),
            np.pad(x, ((1, 1), (0, 0)), mode="constant", constant_values=7),
        )

    def test_select_broadcast(self):
        pred = np.array([[True], [False], [True]])
        on_true = lin((3, 2), dtype=np.float32)
        on_false = lin((3, 2), -2.0, -1.0, dtype=np.float32)
        out = run_graph(
            lambda p, a, b: etl.select(p, a, b), pred, on_true, on_false
        )
        assert_close(as_np(out), np.where(pred, on_true, on_false))

    def test_select_dtype_promotion(self):
        pred = np.array([True, False])
        on_true = np.array([1, 2], np.int32)
        on_false = np.array([0.5, 1.5], np.float32)
        out = run_graph(
            lambda p, a, b: etl.select(p, a, b), pred, on_true, on_false
        )
        assert_close(as_np(out), np.where(pred, on_true, on_false))

    def test_broadcast_vector_to_matrix(self):
        x = lin((3,), dtype=np.float32)
        out = run_graph(lambda t: etl.broadcast(t, (2, 3)), x)
        assert_close(as_np(out), np.broadcast_to(x, (2, 3)))

    def test_broadcast_column_to_matrix(self):
        x = lin((2, 1), dtype=np.float32)
        out = run_graph(lambda t: etl.broadcast(t, (2, 3)), x)
        assert_close(as_np(out), np.broadcast_to(x, (2, 3)))

    def test_broadcast_0d(self):
        @etl.defn
        def fn(t):
            s = etl.sum(t)  # 0-d tensor
            return etl.broadcast(s, (2, 3))

        x = lin((4,), dtype=np.float32)
        out = run_graph(fn, x)
        assert_close(as_np(out), np.broadcast_to(np.sum(x), (2, 3)))


# ---------------------------------------------------------------------------
# 6. getitem (static int / slice indexing — what ops/indexing.py supports)
# ---------------------------------------------------------------------------

class TestGetitem:
    @pytest.mark.parametrize(
        "key",
        [0, -1, slice(1, 3), slice(None, None, -1), slice(3, 0, -2),
         (0, slice(1, 3)), (slice(None), 2)],
        ids=["int", "neg_int", "slice", "rev_slice", "neg_step",
             "int_and_slice", "full_axis_and_int"],
    )
    def test_getitem(self, key):
        x = lin((2, 3, 4), dtype=np.float32)

        @etl.defn
        def fn(t):
            return t[key]

        out = run_graph(fn, x)
        assert_close(as_np(out), x[key])

    def test_getitem_1d(self):
        x = lin((10,), dtype=np.float32)

        @etl.defn
        def fn(t):
            return t[1], t[2:5], t[::3], t[-2]

        got = as_np(run_graph(fn, x))
        assert_close(got[0], x[1])
        assert_close(got[1], x[2:5])
        assert_close(got[2], x[::3])
        assert_close(got[3], x[-2])

    def test_ellipsis_not_supported(self):
        with pytest.raises(etl.TraceError):
            run_graph(lambda t: t[..., 0], lin((2, 3, 4), dtype=np.float32))

    def test_boolean_mask_not_supported(self):
        mask = np.array([True, False])

        @etl.defn
        def fn(t, m):
            return t[m]

        with pytest.raises(etl.TraceError):
            run_graph(fn, lin((2, 3), dtype=np.float32), mask)

    def test_integer_array_indexing_not_supported(self):
        idx = np.array([0, 2], np.int64)

        @etl.defn
        def fn(t, i):
            return t[i]

        with pytest.raises(etl.TraceError):
            run_graph(fn, lin((3, 3), dtype=np.float32), idx)


# ---------------------------------------------------------------------------
# 7. enp sugar (etl.numpy) — same numbers as the mapped ops / numpy
# ---------------------------------------------------------------------------

class TestEnpSugar:
    def test_enp_maps_to_same_results(self):
        import etl.numpy as enp

        @etl.defn
        def fn(x, y, a, b):
            return (
                enp.sum(x),
                enp.mean(x, axis=1),
                enp.matmul(a, b),
                enp.tril(a),
                enp.astype(x, np.float64),
                enp.reshape(x, (2, 6)),
                enp.cumsum(x, axis=0),
                enp.argmax(x, axis=1),
                enp.where(x > 0, x, -x),
                enp.clip(x, -0.5, 0.5),
                enp.maximum(x, y),
                enp.tanh(x),
            )

        x = lin((2, 6), dtype=np.float32)
        y = lin((2, 6), -1.0, 1.0, dtype=np.float32)
        a = lin((3, 3), dtype=np.float32)
        b = lin((3, 2), -1.0, 1.0, dtype=np.float32)
        got = as_np(run_graph(fn, x, y, a, b))
        refs = [
            np.sum(x),
            np.mean(x, axis=1),
            np.matmul(a, b),
            np.tril(a),
            x.astype(np.float64),
            np.reshape(x, (2, 6)),
            np.cumsum(x, axis=0),
            np.argmax(x, axis=1),
            np.where(x > 0, x, -x),
            np.clip(x, -0.5, 0.5),
            np.maximum(x, y),
            np.tanh(x),
        ]
        for actual, ref in zip(got, refs):
            assert_close(actual, ref)

    def test_enp_concatenate_and_transpose(self):
        import etl.numpy as enp

        @etl.defn
        def fn(a, b):
            return enp.concatenate([a, b], axis=0), enp.transpose(a, (1, 0))

        a = lin((2, 3), dtype=np.float32)
        b = lin((2, 3), -1.0, 0.0, dtype=np.float32)
        got = as_np(run_graph(fn, a, b))
        assert_close(got[0], np.concatenate([a, b], axis=0))
        assert_close(got[1], np.transpose(a, (1, 0)))

    def test_enp_linalg_solve(self):
        import etl.numpy as enp

        @etl.defn
        def fn(a, b):
            return enp.linalg.solve(a, b)

        a = np.array([[4.0, 1.0], [1.0, 3.0]], np.float64)
        b = np.array([2.0, 1.0], np.float64)
        out = run_graph(fn, a, b)
        assert_close(as_np(out), np.linalg.solve(a, b))


# ---------------------------------------------------------------------------
# 8. Operator overloads (SymbolicTensor handlers)
# ---------------------------------------------------------------------------

class TestOperatorOverloads:
    def test_arithmetic_overloads(self):
        @etl.defn
        def fn(x, y):
            return (x + y, x - y, x * y, x / y, x ** 2, -x, 2 + x, x * 2 + y)

        x = lin((2, 3), 0.5, 2.0, dtype=np.float32)
        y = lin((2, 3), 0.5, 1.0, dtype=np.float32)
        got = as_np(run_graph(fn, x, y))
        refs = [x + y, x - y, x * y, x / y, x ** 2, -x, 2 + x, x * 2 + y]
        for actual, ref in zip(got, refs):
            assert_close(actual, ref)

    def test_matmul_overload(self):
        @etl.defn
        def fn(a, b):
            return a @ b

        a = lin((3, 4), dtype=np.float32)
        b = lin((4, 2), -1.0, 1.0, dtype=np.float32)
        out = run_graph(fn, a, b)
        assert_close(as_np(out), a @ b)

    def test_comparison_overloads(self):
        @etl.defn
        def fn(x, y):
            return (x > y, x >= y, x < y, x <= y, x == y)

        x = lin((4,), dtype=np.float32)
        y = lin((4,), -1.0, 1.0, dtype=np.float32)
        got = as_np(run_graph(fn, x, y))
        refs = [x > y, x >= y, x < y, x <= y, x == y]
        for actual, ref in zip(got, refs):
            assert_close(actual, ref)

    def test_power_overload_scalar(self):
        @etl.defn
        def fn(x):
            return x ** 3

        x = lin((3,), -2.0, 2.0, dtype=np.float32)
        out = run_graph(fn, x)
        assert_close(as_np(out), x ** 3)

    def test_chain_of_overloads(self):
        @etl.defn
        def fn(x, w):
            return (x @ w + 1.0) * 0.5 - x

        x = lin((2, 2), dtype=np.float32)
        w = lin((2, 2), -1.0, 1.0, dtype=np.float32)
        out = run_graph(fn, x, w)
        assert_close(as_np(out), (x @ w + 1.0) * 0.5 - x)


# ---------------------------------------------------------------------------
# 9. Pipeline ground truth: evaluate returns Tensors; build/run equivalence
# ---------------------------------------------------------------------------

class TestPipelineShorthand:
    def test_evaluate_returns_tensor_with_numpy(self):
        @etl.defn
        def fn(x):
            return etl.multiply(x, 2.0)

        x = np.ones((3,), np.float32)
        out = run_graph(fn, x)
        assert isinstance(out, etl.Tensor)
        assert hasattr(out, "__dlpack__")
        assert out.dtype == np.float32
        np.testing.assert_allclose(out.numpy(), 2.0 * x, **F32_TOL)

    def test_evaluate_accepts_etl_tensors(self):
        @etl.defn
        def fn(x, y):
            return etl.add(x, y)

        x = etl.tensor(np.array([1.0, 2.0], np.float32))
        y = etl.tensor(np.array([3.0, 4.0], np.float32))
        out = run_graph(fn, x, y)
        assert isinstance(out, etl.Tensor)
        np.testing.assert_array_equal(out.numpy(), np.array([4.0, 6.0],
                                                            np.float32))

    def test_build_run_matches_evaluate(self):
        @etl.defn
        def fn(x, y):
            return etl.add(x, y)

        x = lin((2, 3), dtype=np.float32)
        y = lin((2, 3), -1.0, 1.0, dtype=np.float32)
        exe = etl.build(
            fn,
            etl.TensorSpec((2, 3), etl.float32),
            etl.TensorSpec((2, 3), etl.float32),
            backend=etl.backends.numpy_backend,
        )
        out_run = etl.run(exe, x, y)
        assert isinstance(out_run, etl.Tensor)
        np.testing.assert_allclose(out_run.numpy(), x + y, **F32_TOL)
        # same numbers as the evaluate shorthand
        out_eval = run_graph(fn, x, y)
        np.testing.assert_array_equal(out_run.numpy(), out_eval.numpy())

    def test_multi_output_returns_tuple_of_tensors(self):
        @etl.defn
        def fn(x):
            return x, etl.negate(x)

        x = lin((2, 3), dtype=np.float32)
        out = run_graph(fn, x)
        assert isinstance(out, tuple)
        assert all(isinstance(r, etl.Tensor) for r in out)
        np.testing.assert_array_equal(as_np(out)[0], x)
        np.testing.assert_array_equal(as_np(out)[1], -x)
