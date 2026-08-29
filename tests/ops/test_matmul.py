"""Contract tests for ``etl.matmul`` (numpy matmul semantics) and the
dot-contract guard.

Covers ``etl.ops.linalg.matmul`` — frontend composition over ``dot`` with
rank-1 promote/squeeze reshapes. The etl package (repo-root sibling) is fully
implemented; these tests assert the per-op contract documented in the
``etl/ops/linalg.py`` ``matmul`` docstring:

- ``matmul``: numpy ``matmul`` semantics — 1-D @ 1-D → 0-d scalar, 1-D @ 2-D
  / 2-D @ 1-D → 1-D, 2-D @ 2-D matrix product, batched matmul with leading
  batch broadcasting (incl. size-1 batch dims and a vector against a batched
  matrix); dtype = ``np.result_type`` of the operand dtypes.
- Values are asserted BIT-EXACT against ``np.matmul`` everywhere (including
  float32/float64): the kernel executes ``np.matmul`` on the rank-promoted
  operands and the promote/squeeze reshapes preserve the accumulation order —
  verified empirically (``max_abs`` == 0.0) on every case in this file.
- ``dot``'s rank >= 2 contract is UNCHANGED: rank-1 operands still raise
  ``ShapeError``, and ``SymbolicTensor.__matmul__`` still routes to ``dot``.
- Error contract: rank-0 operands raise ``ShapeError`` (numpy raises
  ``ValueError`` — the frontend documents ``ShapeError``); static k mismatch
  raises ``ShapeError`` at trace time; no active trace / concrete ``Tensor``
  operand raise ``TraceError`` with the directing / three-option messages.
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
    the IR value type read back from the built ops.
    """
    captured = {}

    def wrapped(*args):
        out = fn(*args)
        captured["out"] = out
        return out

    graph = etl.trace(wrapped, *specs)
    return graph, captured["out"]


# ---------------------------------------------------------------------------
# shape and dtype inference
# ---------------------------------------------------------------------------

MATMUL_SHAPE_CASES = [
    ((3,), (3,), ()),                    # 1-D @ 1-D → scalar
    ((3,), (3, 4), (4,)),                # vector @ matrix → vector
    ((2, 3), (3,), (2,)),                # matrix @ vector → vector
    ((2, 3), (3, 4), (2, 4)),            # matrix product
    ((2, 3, 4), (2, 4, 5), (2, 3, 5)),   # 3-D @ 3-D
    ((3, 4), (2, 4, 5), (2, 3, 5)),      # 2-D @ 3-D (broadcast leading batch)
    ((2, 3, 4), (4, 5), (2, 3, 5)),      # 3-D @ 2-D
    ((4,), (2, 4, 5), (2, 5)),           # vector @ batched matrix
    ((2, 4, 5), (5,), (2, 4)),           # batched matrix @ vector
    ((1, 3, 4), (2, 4, 5), (2, 3, 5)),   # size-1 batch broadcasts
    ((2, 3, 4), (1, 4, 5), (2, 3, 5)),   # size-1 batch on the other side
    ((2, 1, 3, 4), (5, 4, 6), (2, 5, 3, 6)),  # nested batch broadcast
]


@pytest.mark.parametrize("a_shape,b_shape,out_shape", MATMUL_SHAPE_CASES)
def test_matmul_shape_and_dtype_inference(a_shape, b_shape, out_shape):
    def f(x, w):
        return etl.matmul(x, w)

    graph, out = _trace_capturing(
        f,
        etl.TensorSpec(a_shape, etl.float32),
        etl.TensorSpec(b_shape, etl.float32),
    )
    # Frontend SymbolicTensor contract.
    assert isinstance(out, etl.SymbolicTensor)
    assert tuple(out.shape) == out_shape
    assert out.dtype == np.float32
    # IR value type agrees (read back from the final result op, never
    # computed twice).
    final = ops_of(graph)[-2]
    assert final.results[0].type.shape == out_shape
    assert final.results[0].type.dtype == np.float32


def test_matmul_ir_composition_is_frontend_sugar():
    """matmul is reshape + dot + reshape sugar — no matmul IR op exists."""
    def f(x, w):
        return etl.matmul(x, w)

    # 1-D @ 1-D: promote both sides, dot, squeeze back to a scalar.
    graph = trace_fn(f, etl.TensorSpec((3,), etl.float32),
                     etl.TensorSpec((3,), etl.float32))
    names = [op.name for op in ops_of(graph)]
    assert names == ["reshape", "reshape", "dot", "reshape", "return"]
    dot_op = _op(graph, "dot")
    assert dot_op.results[0].type.shape == (1, 1)

    # 2-D @ 2-D: a single dot, no reshapes.
    graph = trace_fn(f, etl.TensorSpec((2, 3), etl.float32),
                     etl.TensorSpec((3, 4), etl.float32))
    assert [op.name for op in ops_of(graph)] == ["dot", "return"]
    assert _op(graph, "dot").results[0].type.shape == (2, 4)


MATMUL_TYPE_CASES = [
    (etl.float32, etl.float32, np.float32),
    (etl.float64, etl.float64, np.float64),
    (etl.float16, etl.float16, np.float16),
    (etl.int32, etl.int32, np.int32),
    (etl.int8, etl.int32, np.int32),       # result_type(int8, int32) = int32
    (etl.float32, etl.float64, np.float64),
    (etl.int32, etl.float32, np.float64),  # int ⊕ float32 → float64
    (etl.int32, etl.float64, np.float64),
]


@pytest.mark.parametrize(
    "a_dtype,b_dtype,out_dtype", MATMUL_TYPE_CASES
)
def test_matmul_dtype_promotion(a_dtype, b_dtype, out_dtype):
    def f(x, w):
        return etl.matmul(x, w)

    graph, out = _trace_capturing(
        f,
        etl.TensorSpec((2, 3), a_dtype),
        etl.TensorSpec((3, 4), b_dtype),
    )
    assert out.dtype == np.dtype(out_dtype)
    # Documented rule: exactly np.result_type of the operand dtypes.
    assert out.dtype == np.result_type(np.dtype(a_dtype), np.dtype(b_dtype))
    final = ops_of(graph)[-2]
    assert final.results[0].type.dtype == np.dtype(out_dtype)


# ---------------------------------------------------------------------------
# numerics vs numpy (bit-exact — see module docstring)
# ---------------------------------------------------------------------------

def test_matmul_1d_dot_1d_returns_scalar():
    """1-D @ 1-D → 0-d scalar with the promoted dtype (np.dot value)."""
    def f(x, w):
        return etl.matmul(x, w)

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    got = run_numpy(f, x, w)
    assert got.shape == ()
    assert got.dtype == np.float32
    assert got == np.dot(x, w)
    np.testing.assert_array_equal(got, np.matmul(x, w))

    xi = np.array([1, 2, 3], dtype=np.int32)
    wi = np.array([4, 5, 6], dtype=np.int32)
    got_i = run_numpy(f, xi, wi)
    assert got_i.shape == ()
    assert got_i.dtype == np.int32
    assert got_i == np.dot(xi, wi)
    np.testing.assert_array_equal(got_i, np.matmul(xi, wi))


@pytest.mark.parametrize("side", ["a", "b"])
def test_matmul_rank1_with_matrix(side):
    """1-D @ 2-D / 2-D @ 1-D → 1-D (np.matmul semantics, not np.dot's extra
    dims: np.dot(a1, m2) would add a leading dim instead)."""
    def f(x, w):
        return etl.matmul(x, w)

    rng = np.random.default_rng(0)
    if side == "a":
        x = rng.standard_normal(4).astype(np.float32)
        w = rng.standard_normal((4, 5)).astype(np.float32)
        expected = np.matmul(x, w)
    else:
        x = rng.standard_normal((3, 4)).astype(np.float32)
        w = rng.standard_normal(4).astype(np.float32)
        expected = np.matmul(x, w)
    got = run_numpy(f, x, w)
    assert got.shape == expected.shape == (expected.size,)
    np.testing.assert_array_equal(got, expected)


def test_matmul_2d_numerics_vs_numpy():
    rng = np.random.default_rng(1)

    def f(x, w):
        return etl.matmul(x, w)

    x = rng.standard_normal((3, 5))
    w = rng.standard_normal((5, 4))
    np.testing.assert_array_equal(run_numpy(f, x, w), np.matmul(x, w))

    xi = rng.integers(-3, 4, size=(3, 5)).astype(np.int32)
    wi = rng.integers(-3, 4, size=(5, 4)).astype(np.int32)
    np.testing.assert_array_equal(run_numpy(f, xi, wi), np.matmul(xi, wi))


BATCH_NUMERICS_CASES = [
    ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
    ((3, 4), (2, 4, 5), (2, 3, 5)),      # 2-D @ 3-D broadcast
    ((2, 3, 4), (4, 5), (2, 3, 5)),      # 3-D @ 2-D broadcast
    ((4,), (2, 4, 5), (2, 5)),           # vector @ batched matrix
    ((2, 4, 5), (5,), (2, 4)),           # batched matrix @ vector
    ((1, 3, 4), (2, 4, 5), (2, 3, 5)),   # size-1 batch
    ((2, 3, 4), (1, 4, 5), (2, 3, 5)),   # size-1 batch (other side)
    ((2, 1, 3, 4), (5, 4, 6), (2, 5, 3, 6)),
]


@pytest.mark.parametrize("a_shape,b_shape,out_shape", BATCH_NUMERICS_CASES)
def test_matmul_batched_numerics_vs_numpy(a_shape, b_shape, out_shape):
    rng = np.random.default_rng(2)

    def f(x, w):
        return etl.matmul(x, w)

    graph, out = _trace_capturing(
        f,
        etl.TensorSpec(a_shape, etl.float32),
        etl.TensorSpec(b_shape, etl.float32),
    )
    assert tuple(out.shape) == out_shape
    final = ops_of(graph)[-2]
    assert final.results[0].type.shape == out_shape

    x = rng.standard_normal(a_shape).astype(np.float32)
    w = rng.standard_normal(b_shape).astype(np.float32)
    np.testing.assert_array_equal(run_numpy(f, x, w), np.matmul(x, w))


def test_matmul_symbolic_batch_dims_trace_and_run():
    """Symbolic batch dims trace to a DimExpr broadcast statement and the
    graph runs through the explicit pipeline with a concrete batch size."""
    def f(x, w):
        return etl.matmul(x, w)

    graph, out = _trace_capturing(
        f,
        etl.TensorSpec((etl.dim("b"), 3, 4), etl.float32),
        etl.TensorSpec((2, 4, 5), etl.float32),
    )
    shape = tuple(out.shape)
    # Batch = broadcast((b,), (2,)) → the symbolic statement of numpy's rule.
    assert isinstance(shape[0], etl.DimExpr)
    assert shape[0].evaluate({"b": 1}) == 2
    assert shape[0].evaluate({"b": 7}) == 7
    assert shape[1:] == (3, 5)

    # Run via the explicit pipeline (evaluate derives concrete specs, so the
    # symbolic-dim graph needs trace → lower → compile → load → run). The
    # runtime batch must broadcast against the concrete (2,) — numpy rejects
    # b=7 vs 2 the same way the graph's DimExpr statement predicts for
    # broadcastable pairs.
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    executable = etl.load(artifact)
    rng = np.random.default_rng(3)
    w = etl.from_numpy(rng.standard_normal((2, 4, 5)).astype(np.float32))
    for batch in (1, 2):  # size-1 broadcast and identity
        x = etl.from_numpy(rng.standard_normal((batch, 3, 4)).astype(np.float32))
        got = etl.run(executable, x, w).numpy()
        np.testing.assert_array_equal(got, np.matmul(x.numpy(), w.numpy()))


def test_matmul_mixed_dtype_values_vs_numpy():
    """Mixed-dtype numerics are bit-exact too (promotion happens inside the
    single np.matmul call, identically to the reference)."""
    rng = np.random.default_rng(4)

    def f(x, w):
        return etl.matmul(x, w)

    xi8 = rng.integers(-3, 4, size=(2, 3)).astype(np.int8)
    wi32 = rng.integers(-3, 4, size=(3, 4)).astype(np.int32)
    got = run_numpy(f, xi8, wi32)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, np.matmul(xi8, wi32))

    xf32 = rng.standard_normal((2, 3)).astype(np.float32)
    wf64 = rng.standard_normal((3, 4)).astype(np.float64)
    got = run_numpy(f, xf32, wf64)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.matmul(xf32, wf64))

    xi32 = rng.integers(-3, 4, size=(2, 3)).astype(np.int32)
    got = run_numpy(f, xi32, wf64)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.matmul(xi32, wf64))


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_matmul_rank0_operand_raises():
    """0-d operands raise ShapeError. (Deviation from numpy, which raises
    ValueError "matmul: Input operand 0 does not have enough dimensions" —
    the frontend documents ShapeError for rank-0 operands.)"""
    def f(x, w):
        return etl.matmul(x, w)

    with pytest.raises(etl.ShapeError, match="operands must have rank >= 1"):
        etl.trace(f, etl.TensorSpec((), etl.float32),
                  etl.TensorSpec((), etl.float32))
    with pytest.raises(etl.ShapeError, match="operands must have rank >= 1"):
        etl.trace(f, etl.TensorSpec((), etl.float32),
                  etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="operands must have rank >= 1"):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32),
                  etl.TensorSpec((), etl.float32))


def test_matmul_k_mismatch_is_static_shape_error():
    def f(x, w):
        return etl.matmul(x, w)

    with pytest.raises(etl.ShapeError, match="contracting dims 3 and 4 do not match"):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32),
                  etl.TensorSpec((4, 5), etl.float32))
    # Also static inside a batch, and with rank-1 promotion.
    with pytest.raises(etl.ShapeError, match="contracting dims 4 and 6 do not match"):
        etl.trace(f, etl.TensorSpec((2, 3, 4), etl.float32),
                  etl.TensorSpec((2, 6, 5), etl.float32))
    with pytest.raises(etl.ShapeError, match="contracting dims 3 and 4 do not match"):
        etl.trace(f, etl.TensorSpec((3,), etl.float32),
                  etl.TensorSpec((4, 5), etl.float32))


def test_matmul_outside_trace_raises():
    """No eager mode: matmul on concrete data outside a trace fails with the
    directing message (there is no silent eager execution)."""
    x = np.ones((2, 3), dtype=np.float32)
    w = np.ones((3, 4), dtype=np.float32)
    with pytest.raises(etl.TraceError, match="No active trace"):
        etl.matmul(x, w)


def test_matmul_concrete_tensor_operand_raises():
    """A concrete Tensor operand inside a trace raises TraceError with the
    mandated three-option message (explicit input / etl.constant /
    etl.evaluate)."""
    w = etl.tensor(np.ones((3, 4), dtype=np.float32))

    @etl.defn
    def f(x):
        return etl.matmul(x, w)

    with pytest.raises(
        etl.TraceError,
        match=r"explicit input.*etl\.constant.*etl\.evaluate",
    ):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32))


# ---------------------------------------------------------------------------
# dot-contract guards (matmul must NOT change dot's contract)
# ---------------------------------------------------------------------------

def test_dot_rank1_contract_unchanged():
    """etl.dot still requires rank >= 2 on both operands — matmul's 1-D
    support is frontend sugar, not a relaxation of dot's contract."""
    def f(x, w):
        return etl.dot(x, w)

    with pytest.raises(etl.ShapeError, match="operands must have rank >= 2"):
        etl.trace(f, etl.TensorSpec((3,), etl.float32),
                  etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(etl.ShapeError, match="operands must have rank >= 2"):
        etl.trace(f, etl.TensorSpec((2, 3), etl.float32),
                  etl.TensorSpec((3,), etl.float32))
    # And dot still works for rank >= 2 (sanity).
    graph, out = _trace_capturing(
        f, etl.TensorSpec((2, 3), etl.float32),
        etl.TensorSpec((3, 4), etl.float32)
    )
    assert tuple(out.shape) == (2, 4)
    assert _op(graph, "dot").name == "dot"


def test_matmul_operator_routes_to_dot():
    """``x @ y`` still routes to the dot op (SymbolicTensor.__matmul__) and
    works for rank >= 2 — the operator is not re-routed to matmul."""
    def f(x, w):
        return x @ w

    graph, out = _trace_capturing(
        f, etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((3, 4), etl.float32)
    )
    assert tuple(out.shape) == (2, 4)
    assert [op.name for op in ops_of(graph)] == ["dot", "return"]
    assert _op(graph, "dot").name == "dot"

    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    w = rng.standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_array_equal(run_numpy(f, x, w), np.matmul(x, w))
