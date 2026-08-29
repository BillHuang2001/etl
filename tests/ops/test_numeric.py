"""Contract tests for the numeric / creation ops batch.

Covers ``etl.ops.structural.isnan``, ``nan_to_num``, ``eye``, ``linspace`` and
``etl.ops.linalg.cumprod`` (the 15-op batch; ``clamp`` lives in
``test_structural.py``). The etl package (repo-root sibling) is fully
implemented; these tests assert the per-op contracts documented in the
``etl/ops/structural.py`` / ``etl/ops/linalg.py`` docstrings:

- ``isnan``: composition over ``not_equal(x, x)`` → bool tensor (no dedicated
  IR op); complex is NaN when the real OR imaginary part is NaN; integer
  inputs are never NaN.
- ``nan_to_num``: numpy semantics with per-infinity replacement arguments
  (``None`` = the numpy default, the dtype's max/min finite); only the
  corresponding infinity is replaced per argument; exact dtype/shape
  preservation (integer inputs pass through unchanged).
- ``cumprod``: numpy ``cumprod`` semantics plus an optional reverse scan
  (mirrors ``cumsum``); bool → int64 (explicit frontend pre-cast); other
  dtypes preserved; accumulates in the operand dtype (numpy-2-proof).
- ``eye``: GRAPH creation op (Constant composition, the ``enp`` pattern) —
  NOT a concrete creator like ``etl.zeros``; default dtype float32 (the etl
  creation convention).
- ``linspace``: GRAPH creation op (Constant composition); default dtype
  float64 — a DELIBERATE deviation from the etl float32 creation convention
  for numpy exactness (documented in the op docstring); symbolic
  ``Dim``/``DimExpr`` bounds raise ``TraceError`` (deferred to v2, like
  ``enp.arange``); a SymbolicTensor bound falls through to the number check
  → ``TypeError``.
"""
from __future__ import annotations

import numpy as np
import pytest

import etl
from tests.ops.conftest import ops_of, run_numpy, trace_fn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _concrete_tensor_operand(op_call):
    """Assert the mandated three-option TraceError for a concrete Tensor
    operand passed INSIDE a trace."""
    with pytest.raises(etl.TraceError) as exc:
        trace_fn(lambda: op_call(etl.tensor(np.ones((2, 3), np.float32))))
    message = str(exc.value)
    assert message.startswith(
        "Concrete Tensor operands are not allowed in graph ops "
        "(etl has no eager mode)"
    )
    assert "explicit input" in message
    assert "etl.constant" in message
    assert "etl.evaluate" in message


# ---------------------------------------------------------------------------
# isnan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isnan_numerics_vs_numpy(dtype):
    def f(x):
        return etl.isnan(x)

    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.5], dtype=dtype)
    got = run_numpy(f, x)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, np.isnan(x))


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_isnan_int_input_is_all_false(dtype):
    """Integer tensors cannot hold NaN — every entry is False (bool dtype)."""
    def f(x):
        return etl.isnan(x)

    x = np.array([1, -2, 3, 0], dtype=dtype)
    got = run_numpy(f, x)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, np.isnan(x))
    assert not got.any()


def test_isnan_complex_nan_in_either_part_is_true():
    """Complex semantics: NaN in the real OR imaginary part → True."""
    def f(x):
        return etl.isnan(x)

    x = np.array([
        1 + 2j,             # finite
        np.nan + 1j,        # NaN in real
        1 + np.nan * 1j,    # NaN in imag
        np.inf + 0j,        # inf is not NaN
        np.nan + np.nan * 1j,
    ])
    got = run_numpy(f, x)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, np.isnan(x))


def test_isnan_2d_output_shape_and_dtype():
    def f(x):
        return etl.isnan(x)

    graph, out = _trace_capturing(
        f, etl.TensorSpec((2, 3), etl.float64))
    assert [op.name for op in ops_of(graph)] == ["not_equal", "return"]
    assert out.dtype == np.bool_
    assert out.shape == (2, 3)

    x = np.array([[1.0, np.nan, np.inf], [-np.inf, 0.0, np.nan]])
    np.testing.assert_array_equal(run_numpy(f, x), np.isnan(x))


def test_isnan_outside_trace_raises_traceerror():
    with pytest.raises(etl.TraceError):
        etl.isnan(np.array([1.0, np.nan], dtype=np.float32))


def test_isnan_concrete_tensor_operand_raises_traceerror():
    _concrete_tensor_operand(etl.isnan)


# ---------------------------------------------------------------------------
# nan_to_num
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_nan_to_num_defaults_vs_numpy(dtype):
    """Default replacements: nan→0.0, +inf→dtype max finite, -inf→dtype min
    finite (``None`` = the numpy defaults)."""
    def f(x):
        return etl.nan_to_num(x)

    x = np.array([1.5, np.nan, np.inf, -np.inf], dtype=dtype)
    got = run_numpy(f, x)
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, np.nan_to_num(x))
    # Explicit per-dtype finite bounds.
    np.testing.assert_array_equal(
        got, [np.float16(1.5) if dtype == np.float16 else 1.5,
              0, np.finfo(dtype).max, np.finfo(dtype).min])


def test_nan_to_num_explicit_nan_replacement():
    def f(x):
        return etl.nan_to_num(x, nan=1.5)

    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_equal(run_numpy(f, x), np.nan_to_num(x, nan=1.5))


def test_nan_to_num_posinf_given_leaves_neginf_alone():
    """Only the corresponding infinity is replaced: ``posinf=1`` must leave
    -inf untouched (the numpy per-argument semantics)."""
    def f(x):
        return etl.nan_to_num(x, nan=0.0, posinf=1.0)

    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    got = run_numpy(f, x)
    np.testing.assert_array_equal(got, np.nan_to_num(x, nan=0.0, posinf=1.0))
    np.testing.assert_array_equal(
        got, [1.0, 0.0, 1.0, np.finfo(np.float32).min])


def test_nan_to_num_neginf_given_leaves_posinf_alone():
    def f(x):
        return etl.nan_to_num(x, nan=0.0, neginf=-1.0)

    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    got = run_numpy(f, x)
    np.testing.assert_array_equal(got, np.nan_to_num(x, nan=0.0, neginf=-1.0))
    np.testing.assert_array_equal(
        got, [1.0, 0.0, np.finfo(np.float32).max, -1.0])


def test_nan_to_num_both_infinity_replacements():
    def f(x):
        return etl.nan_to_num(x, nan=5.0, posinf=7.0, neginf=-7.0)

    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_equal(
        run_numpy(f, x), np.nan_to_num(x, nan=5.0, posinf=7.0, neginf=-7.0))


def test_nan_to_num_int_input_passthrough():
    """Integer input has no NaN/inf values — passes through unchanged."""
    def f(x):
        return etl.nan_to_num(x)

    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    got = run_numpy(f, x)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, x)


def test_nan_to_num_2d_dtype_and_shape_preserved():
    def f(x):
        return etl.nan_to_num(x)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), etl.float32))
    assert [op.name for op in ops_of(graph)] == ["nan_to_num", "return"]
    assert out.dtype == np.float32
    assert out.shape == (2, 3)

    x = np.array([[1.0, np.nan, np.inf], [-np.inf, 2.0, 3.0]], np.float32)
    np.testing.assert_array_equal(run_numpy(f, x), np.nan_to_num(x))


@pytest.mark.parametrize("kwargs", [
    {"nan": True},
    {"nan": "x"},
    {"posinf": "x"},
    {"neginf": 1 + 2j},
])
def test_nan_to_num_invalid_replacement_type(kwargs):
    with pytest.raises(TypeError, match=r"nan_to_num: (nan|posinf|neginf) "
                                        r"must be an int/float"):
        trace_fn(lambda x: etl.nan_to_num(x, **kwargs),
                 etl.TensorSpec((3,), etl.float32))


def test_nan_to_num_outside_trace_raises_traceerror():
    with pytest.raises(etl.TraceError):
        etl.nan_to_num(np.array([1.0, np.nan], dtype=np.float32))


def test_nan_to_num_concrete_tensor_operand_raises_traceerror():
    _concrete_tensor_operand(etl.nan_to_num)


# ---------------------------------------------------------------------------
# cumprod
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("dtype", [etl.float64, etl.float32, etl.int32])
def test_cumprod_numerics_vs_numpy(axis, dtype):
    def f(x):
        return etl.cumprod(x, axis=axis)

    x = np.arange(1, 13).reshape(3, 4).astype(dtype)
    got = run_numpy(f, x)
    # dtype is preserved exactly (numpy >= 2 upcasts int32 cumprod — the etl
    # contract explicitly accumulates in the operand dtype).
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, np.cumprod(x, axis=axis, dtype=x.dtype))


def test_cumprod_int8_dtype_preserved():
    """int8 accumulation stays in int8 — small values keep results in range
    (the kernel calls np.cumprod with dtype=arr.dtype)."""
    def f(x):
        return etl.cumprod(x, axis=1)

    x = np.array([[1, 1, 2], [2, 2, 3]], dtype=np.int8)
    got = run_numpy(f, x)
    assert got.dtype == np.int8
    np.testing.assert_array_equal(got, np.cumprod(x, axis=1, dtype=x.dtype))


def test_cumprod_bool_promotes_to_int64():
    """Documented rule: bool → int64, implemented as an explicit pre-cast
    (mirror of cumsum)."""
    def f(x):
        return etl.cumprod(x, axis=1)

    graph, out = _trace_capturing(f, etl.TensorSpec((2, 3), etl.bool_))
    names = [op.name for op in ops_of(graph)]
    assert names == ["cast", "cumprod", "return"]
    assert out.dtype == np.int64
    x = np.array([[True, False, True], [False, True, True]])
    got = run_numpy(f, x)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, np.cumprod(x.astype(np.int64), axis=1))


@pytest.mark.parametrize("axis", [0, 1])
def test_cumprod_reverse_scans_from_the_end(axis):
    """``reverse=True`` is documented as "scan from the end toward the start",
    i.e. out[i] = prod(x[i:]) — equivalently flip(cumprod(flip(x)))."""
    def f(x):
        return etl.cumprod(x, axis=axis, reverse=True)

    x = np.array([[1.5, 2.0, 0.5], [3.0, -2.0, 4.0]], dtype=np.float64)
    expected = np.flip(np.cumprod(np.flip(x, axis=axis), axis=axis), axis=axis)
    np.testing.assert_array_equal(run_numpy(f, x), expected)


def test_cumprod_reverse_1d():
    def f(x):
        return etl.cumprod(x, axis=0, reverse=True)

    x = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    expected = np.flip(np.cumprod(np.flip(x, axis=0), axis=0), axis=0)
    np.testing.assert_array_equal(run_numpy(f, x), expected)
    np.testing.assert_array_equal(run_numpy(f, x), [24.0, 12.0, 4.0])


def test_cumprod_scalar_is_identity():
    def f(x):
        return etl.cumprod(x, axis=0)

    got = run_numpy(f, np.array(3.5, dtype=np.float32))
    assert got.shape == ()
    assert got.dtype == np.float32
    assert got == np.float32(3.5)


def test_cumprod_errors():
    with pytest.raises(etl.ShapeError, match="axis 3 out of range for rank 2"):
        etl.trace(lambda x: etl.cumprod(x, axis=3),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError,
                       match="axes must be None, an int, or a tuple of ints"):
        etl.trace(lambda x: etl.cumprod(x, axis=1.5),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(TypeError, match="cumprod: reverse must be a bool"):
        etl.trace(lambda x: etl.cumprod(x, axis=0, reverse="yes"),
                  etl.TensorSpec((2, 3), etl.float32))


def test_cumprod_outside_trace_raises_traceerror():
    with pytest.raises(etl.TraceError):
        etl.cumprod(np.ones((2, 3), dtype=np.float32))


def test_cumprod_concrete_tensor_operand_raises_traceerror():
    _concrete_tensor_operand(etl.cumprod)


# ---------------------------------------------------------------------------
# eye
# ---------------------------------------------------------------------------

def test_eye_default_dtype_is_float32():
    """Creation-op convention: default dtype float32 (PINNED)."""
    def f():
        return etl.eye(3)

    graph, out = _trace_capturing(f)
    assert isinstance(out, etl.SymbolicTensor)
    assert out.dtype == np.float32
    assert out.shape == (3, 3)
    assert [op.name for op in ops_of(graph)] == ["constant", "return"]

    got = run_numpy(f)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.eye(3, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.bool_,
                                   np.float64, np.float32])
def test_eye_explicit_dtypes_vs_numpy(dtype):
    def f():
        return etl.eye(4, dtype=dtype)

    got = run_numpy(f)
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, np.eye(4, dtype=dtype))


@pytest.mark.parametrize("n, m", [(2, 4), (4, 2)])
def test_eye_rectangular(n, m):
    def f():
        return etl.eye(n, m)

    got = run_numpy(f)
    assert got.shape == (n, m)
    np.testing.assert_array_equal(got, np.eye(n, m))


@pytest.mark.parametrize("n, m", [(0, None), (0, 3), (3, 0)])
def test_eye_zero_size(n, m):
    def f():
        return etl.eye(n) if m is None else etl.eye(n, m)

    got = run_numpy(f)
    expected = np.eye(n) if m is None else np.eye(n, m)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_eye_diagonal_ones_offdiagonal_zero():
    got = run_numpy(lambda: etl.eye(4, 6, dtype=np.float64))
    np.testing.assert_array_equal(np.diag(got), np.ones(4))
    mask = ~np.eye(4, 6, dtype=bool)
    np.testing.assert_array_equal(got[mask], np.zeros(mask.sum()))


def test_eye_is_graph_op_not_concrete_creator():
    """eye is NOT a concrete creator like etl.zeros — it is graph composition
    (Constant embedding) and requires an active trace."""
    with pytest.raises(etl.TraceError):
        etl.eye(3)


def test_eye_bad_static_args():
    # Non-int n/m → TypeError (checked before the trace state, so it fires
    # even outside a trace); bools are rejected as "not a plain int".
    with pytest.raises(TypeError, match=r"eye: n must be a Python int, "
                                        r"got 2\.5"):
        etl.eye(2.5)
    with pytest.raises(TypeError, match=r"eye: n must be a Python int, "
                                        r"got True"):
        etl.eye(True)
    with pytest.raises(TypeError, match=r"eye: m must be a Python int"):
        trace_fn(lambda: etl.eye(3, m=2.5))
    # Negative dimensions → ValueError (numpy parity, raised by numpy).
    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        etl.eye(-1)


# ---------------------------------------------------------------------------
# linspace
# ---------------------------------------------------------------------------

def test_linspace_default_dtype_is_float64():
    """PINNED: default dtype float64 — a DELIBERATE deviation from the etl
    float32 creation convention, for numpy exactness (documented in the op
    docstring)."""
    def f():
        return etl.linspace(0, 1, 5)

    graph, out = _trace_capturing(f)
    assert isinstance(out, etl.SymbolicTensor)
    assert out.dtype == np.float64
    assert out.shape == (5,)
    assert [op.name for op in ops_of(graph)] == ["constant", "return"]

    got = run_numpy(f)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.linspace(0, 1, 5))


@pytest.mark.parametrize("start, stop, num", [
    (0, 1, 0),       # empty
    (0, 1, 1),       # single start value
    (0, 1, 2),       # endpoints
    (0, 1, 7),
    (0, 1, 101),     # large
    (5, -5, 9),      # start > stop (descending)
    (-3.5, -0.5, 7),  # negative values
    (2, 10, 4),      # int bounds → still float64
])
def test_linspace_numerics_vs_numpy(start, stop, num):
    def f():
        return etl.linspace(start, stop, num)

    got = run_numpy(f)
    expected = np.linspace(start, stop, num)
    assert got.dtype == expected.dtype
    # float64 exactness: same algorithm as numpy — bit-identical.
    np.testing.assert_array_equal(got, expected)


def test_linspace_explicit_float32_dtype():
    def f():
        return etl.linspace(0, 1, 5, dtype=np.float32)

    got = run_numpy(f)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.linspace(0, 1, 5, dtype=np.float32))


def test_linspace_bounds_must_be_static_python_numbers():
    """Symbolic Dim/DimExpr bounds → TraceError (v2 deferral, mirrors
    ``enp.arange``); a SymbolicTensor bound falls through to the number check
    → TypeError (pinned real behavior)."""
    with pytest.raises(etl.TraceError,
                       match=r"linspace: symbolic start=Dim\('n'\) is not "
                             r"supported"):
        etl.trace(lambda: etl.linspace(etl.Dim("n"), 1.0, 3))
    with pytest.raises(
        etl.TraceError,
        match=r"linspace: symbolic start=DimExpr\('add', left=Dim\('n'\), "
              r"right=1\) is not supported",
    ):
        etl.trace(lambda: etl.linspace(etl.Dim("n") + 1, 1.0, 3))
    with pytest.raises(etl.TraceError,
                       match=r"linspace: symbolic stop=Dim\('n'\) is not "
                             r"supported"):
        etl.trace(lambda: etl.linspace(0.0, etl.Dim("n"), 3))
    with pytest.raises(TypeError,
                       match=r"linspace: start must be a Python int or "
                             r"float"):
        etl.trace(lambda x: etl.linspace(x, 1.0, 3),
                  etl.TensorSpec((), etl.float32))
    with pytest.raises(TypeError,
                       match=r"linspace: start must be a Python int or "
                             r"float, got True"):
        etl.trace(lambda: etl.linspace(True, 1.0, 3))


def test_linspace_outside_trace_raises_traceerror():
    with pytest.raises(etl.TraceError):
        etl.linspace(0, 1, 3)


@pytest.mark.parametrize("num, exc, match", [
    (5.0, TypeError, r"linspace: num must be a Python int, got 5\.0"),
    (True, TypeError, r"linspace: num must be a Python int, got True"),
    (-3, ValueError, r"Number of samples, -3, must be non-negative\."),
])
def test_linspace_bad_num(num, exc, match):
    with pytest.raises(exc, match=match):
        etl.trace(lambda: etl.linspace(0, 1, num))
