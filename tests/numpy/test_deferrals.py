"""Sugar discipline of `etl.numpy` (enp): defer, never invent.

enp is a pure-sugar namespace over etl.ops (no eager numpy fallback, no
numerical kernels). These tests pin the contract that enp either forwards
calls transparently (errors surface unchanged, IR identical) or defers loudly
(NotImplementedError / ValueError / AttributeError / TypeError) — it never
silently invents semantics. See ../etl/numpy/CONTEXT.md for the mapping table
and the v1 deferral list.
"""

import numpy as np
import pytest

import etl
import etl.numpy as enp
from tests.numpy._ir_utils import normalize_ir

SPEC_2X3 = etl.TensorSpec((2, 3), etl.float32)


# --- file-local helpers -----------------------------------------------------


def _trace_text(fn, *specs):
    """Trace fn, verify the module, and return normalized pretty-printed IR."""
    graph = etl.trace(fn, *specs)
    graph.verify()
    return normalize_ir(etl.ir.pretty_print(graph.module))


# --- no eager fallback: concrete Tensors raise TraceError like ops ----------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda t: enp.add(t, t), id="add"),
        pytest.param(lambda t: enp.sqrt(t), id="sqrt"),
        pytest.param(lambda t: enp.sum(t), id="sum"),
        pytest.param(lambda t: enp.reshape(t, (2, 2)), id="reshape"),
        pytest.param(
            lambda t: enp.where(etl.tensor(np.ones((2, 2), np.bool_)), t, t),
            id="where",
        ),
        pytest.param(lambda t: enp.matmul(t, t), id="matmul"),
        pytest.param(lambda t: enp.linalg.solve(t, t), id="linalg.solve"),
    ],
)
def test_concrete_tensor_outside_trace_raises_trace_error(call):
    """enp has no eager mode: concrete Tensor args raise TraceError exactly
    like ops (the kernels live only in the backends)."""
    t = etl.tensor(np.ones((2, 2), np.float32))
    with pytest.raises(etl.TraceError, match="No active trace"):
        call(t)


def test_trace_error_message_identical_to_ops():
    """enp must not catch or reword ops' errors — same message verbatim."""
    t = etl.tensor(np.ones((2, 2), np.float32))
    with pytest.raises(etl.TraceError) as exc_enp:
        enp.add(t, t)
    with pytest.raises(etl.TraceError) as exc_ops:
        etl.ops.add(t, t)
    assert str(exc_enp.value) == str(exc_ops.value)


# --- documented deferrals ---------------------------------------------------


@pytest.mark.parametrize("mode", ["edge", "reflect", "wrap"])
def test_pad_nonconstant_modes_deferred(mode):
    """Only mode='constant' exists in v1; other numpy.pad modes are deferred
    (loud NotImplementedError, never a silent constant-pad fallback)."""

    @etl.defn
    def f(a):
        return enp.pad(a, 1, mode=mode)

    with pytest.raises(NotImplementedError, match="deferred"):
        etl.trace(f, SPEC_2X3)


def test_clip_both_bounds_none_raises_value_error():
    """numpy parity: clip with both bounds None raises ValueError."""

    @etl.defn
    def f(a):
        return enp.clip(a, None, None)

    with pytest.raises(ValueError, match="at least one"):
        etl.trace(f, SPEC_2X3)


@pytest.mark.parametrize(
    "name", ["linspace", "absolute", "var", "std", "einsum"]
)
def test_deferred_top_level_names_absent(name):
    """Documented v1 deferrals / not-provided names are genuinely absent."""
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(enp, name)


@pytest.mark.parametrize("name", ["inv", "norm", "det", "matrix_power"])
def test_deferred_linalg_names_absent(name):
    """enp.linalg is solve-only in v1; inv/norm/det/matrix_power are deferred
    (need new IR ops not in the ops contract)."""
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(enp.linalg, name)


def test_linalg_exposes_factorizations():
    """enp.linalg exposes the v1 factorization surface: solve, eigh,
    cholesky, qr, matrix_rank, svd, matrix_exp (top-level enp keeps
    matmul/dot)."""
    assert set(enp.linalg.__all__) == {
        "solve", "eigh", "cholesky", "qr", "matrix_rank", "svd", "matrix_exp",
    }
    assert getattr(enp.linalg, "inv", None) is None
    assert getattr(enp.linalg, "matmul", None) is None
    assert getattr(enp, "solve", None) is None


# --- plain forwards: no ufunc kwargs, ops' errors surface unchanged ---------


def test_add_ufunc_kwargs_not_provided():
    """numpy ufunc kwargs (out/where/dtype/...) are not provided in v1 — the
    sugar signatures are plain forwards."""

    @etl.defn
    def f(a, b):
        return enp.add(a, b, out=a)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        etl.trace(f, SPEC_2X3, SPEC_2X3)


def test_sum_ufunc_kwargs_not_provided():
    """Same for reductions: no `where=` kwarg in v1."""

    @etl.defn
    def f(a):
        return enp.sum(a, where=True)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        etl.trace(f, SPEC_2X3)


def test_reshape_numel_mismatch_surfaces_ops_error():
    """Transparent forward: ops' ShapeError surfaces unchanged through enp."""

    @etl.defn
    def f(a):
        return enp.reshape(a, (3, 7))  # 6 elements -> 21: mismatch

    with pytest.raises(etl.ShapeError, match="reshape"):
        etl.trace(f, SPEC_2X3)


# --- pure-sugar forwards build identical IR ---------------------------------


@pytest.mark.parametrize(
    "enp_fn, ops_fn, unary, op_name",
    [
        pytest.param(enp.subtract, etl.ops.subtract, False, "subtract",
                     id="subtract"),
        pytest.param(enp.negative, etl.ops.negate, True, "negate",
                     id="negative"),
    ],
)
def test_sugar_forward_builds_identical_ir(enp_fn, ops_fn, unary, op_name):
    """Smoke test that 1:1 sugar is a forward, not a reimplementation: enp and
    the mapped ops call build identical IR (same op names, operands, attrs)."""

    @etl.defn
    def f_enp(a, b):
        return enp_fn(a) if unary else enp_fn(a, b)

    @etl.defn
    def f_ops(a, b):
        return ops_fn(a) if unary else ops_fn(a, b)

    ir_enp = _trace_text(f_enp, SPEC_2X3, SPEC_2X3)
    ir_ops = _trace_text(f_ops, SPEC_2X3, SPEC_2X3)
    assert ir_enp == ir_ops
    assert f"etl.{op_name}(" in ir_enp
