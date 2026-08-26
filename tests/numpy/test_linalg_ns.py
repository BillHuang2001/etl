"""Linear-algebra sugar: enp.matmul / enp.dot (top level) and
`etl.numpy.linalg.solve`.

Contract (see ../etl/numpy/CONTEXT.md): matmul and dot are both 1:1 sugar
over ops.dot (dot is a v1 alias of matmul — numpy's 1-D inner-product
semantics are a documented deviation), and linalg.solve is sugar over
ops.solve. enp must build identical IR and produce numpy-matching numerics;
it has no numerical kernels of its own.
"""

import numpy as np
import pytest

import etl
import etl.numpy as enp
from _ir_utils import normalize_ir

SPEC_A = etl.TensorSpec((3, 3), etl.float32)
SPEC_B = etl.TensorSpec((3, 2), etl.float32)


# --- file-local helpers -----------------------------------------------------


def _trace_text(fn, *specs):
    """Trace fn, verify the module, and return normalized pretty-printed IR."""
    graph = etl.trace(fn, *specs)
    graph.verify()
    return normalize_ir(etl.ir.pretty_print(graph.module))


# --- IR equivalence ---------------------------------------------------------


def test_matmul_dot_opsdot_build_identical_ir():
    """enp.matmul and enp.dot are both transparent sugar for ops.dot: all
    three build identical IR from identical inputs."""

    @etl.defn
    def f_matmul(a, b):
        return enp.matmul(a, b)

    @etl.defn
    def f_dot(a, b):
        return enp.dot(a, b)

    @etl.defn
    def f_ops(a, b):
        return etl.ops.dot(a, b)

    ir_matmul = _trace_text(f_matmul, SPEC_A, SPEC_B)
    ir_dot = _trace_text(f_dot, SPEC_A, SPEC_B)
    ir_ops = _trace_text(f_ops, SPEC_A, SPEC_B)
    assert ir_matmul == ir_dot == ir_ops
    assert "etl.dot(" in ir_matmul


def test_solve_ops_build_identical_ir():
    """enp.linalg.solve is transparent sugar for ops.solve."""

    @etl.defn
    def f_solve(a, b):
        return enp.linalg.solve(a, b)

    @etl.defn
    def f_ops(a, b):
        return etl.ops.solve(a, b)

    ir_solve = _trace_text(f_solve, SPEC_A, SPEC_A)
    ir_ops = _trace_text(f_ops, SPEC_A, SPEC_A)
    assert ir_solve == ir_ops
    assert "etl.solve(" in ir_solve


# --- numerics via etl.evaluate ---------------------------------------------


@pytest.mark.parametrize(
    "enp_fn", [pytest.param(enp.matmul, id="matmul"),
               pytest.param(enp.dot, id="dot")]
)
def test_matmul_dot_numeric_matches_numpy(enp_fn):
    """enp.matmul / enp.dot on (3,3) x (3,2) match numpy.matmul exactly for
    float32 (same numerical kernel via the reference backend)."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3)).astype(np.float32)
    b = rng.normal(size=(3, 2)).astype(np.float32)

    @etl.defn
    def f(x, y):
        return enp_fn(x, y)

    result = etl.evaluate(f, a, b)
    assert isinstance(result, etl.Tensor)
    np.testing.assert_allclose(result.numpy(), np.matmul(a, b),
                               rtol=1e-5, atol=1e-6)
    assert result.numpy().shape == (3, 2)
    assert result.numpy().dtype == np.float32


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param(
            np.array([[3.0, 1.0], [1.0, 2.0]], np.float32),
            np.array([9.0, 8.0], np.float32),
            id="2x2-vector",
        ),
        pytest.param(
            np.array([[4.0, 1.0, 2.0],
                      [3.0, 5.0, 1.0],
                      [1.0, 1.0, 3.0]], np.float32),
            np.array([[4.0, 1.0],
                      [7.0, 2.0],
                      [3.0, 3.0]], np.float32),
            id="3x3-matrix",
        ),
    ],
)
def test_solve_numeric_matches_numpy(a, b):
    """enp.linalg.solve matches numpy.linalg.solve on well-conditioned
    float32 systems (both vector and matrix right-hand sides)."""

    @etl.defn
    def f(x, y):
        return enp.linalg.solve(x, y)

    result = etl.evaluate(f, a, b)
    assert isinstance(result, etl.Tensor)
    np.testing.assert_allclose(result.numpy(), np.linalg.solve(a, b),
                               atol=1e-5)
    assert result.numpy().shape == b.shape


# --- documented deviation: dot == matmul, even for 1-D vectors --------------


def test_dot_1d_vectors_identical_to_matmul():
    """v1 deviation (documented in etl/numpy/CONTEXT.md): enp.dot is an alias
    of enp.matmul — numpy's 1-D inner-product semantics for dot are NOT
    special-cased. ops.dot requires rank >= 2 (matmul semantics), so for 1-D
    inputs enp.dot and enp.matmul raise the identical ShapeError (np.dot on
    the same inputs would return a scalar — intentionally not matched)."""

    spec1 = etl.TensorSpec((3,), etl.float32)

    @etl.defn
    def f_dot(a, b):
        return enp.dot(a, b)

    @etl.defn
    def f_matmul(a, b):
        return enp.matmul(a, b)

    with pytest.raises(etl.ShapeError, match="rank >= 2") as exc_dot:
        etl.trace(f_dot, spec1, spec1)
    with pytest.raises(etl.ShapeError, match="rank >= 2") as exc_matmul:
        etl.trace(f_matmul, spec1, spec1)
    assert type(exc_dot.value) is type(exc_matmul.value)
    assert str(exc_dot.value) == str(exc_matmul.value)


# --- error paths ------------------------------------------------------------


def test_solve_singular_matrix_raises():
    """A singular coefficient matrix raises (numpy LinAlgError) — never
    silently returns nan/inf garbage."""

    @etl.defn
    def f(x, y):
        return enp.linalg.solve(x, y)

    a = np.array([[1.0, 2.0], [2.0, 4.0]], np.float32)
    b = np.array([3.0, 6.0], np.float32)
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        etl.evaluate(f, a, b)


@pytest.mark.parametrize(
    "a, b, match",
    [
        pytest.param(
            np.ones((2, 3), np.float32), np.ones(2, np.float32),
            "must be square", id="non-square",
        ),
        pytest.param(
            np.ones((2, 2), np.float32), np.ones(3, np.float32),
            "contracting dims", id="contracting-dim-mismatch",
        ),
        pytest.param(
            np.ones(2, np.float32), np.ones(2, np.float32),
            "rank >= 2", id="rank-1",
        ),
    ],
)
def test_solve_shape_errors(a, b, match):
    """The ShapeErrors documented in ops.solve's contract surface unchanged
    through enp.linalg.solve (non-square 'a', contracting-dim mismatch,
    rank < 2)."""

    @etl.defn
    def f(x, y):
        return enp.linalg.solve(x, y)

    with pytest.raises(etl.ShapeError, match=match):
        etl.evaluate(f, a, b)


# --- submodule wiring -------------------------------------------------------


def test_linalg_submodule_importable():
    """etl.numpy.linalg is the documented submodule; solve is reachable via
    both `import etl.numpy.linalg` and `from etl.numpy import linalg`."""
    import etl.numpy.linalg  # noqa: F401

    from etl.numpy import linalg

    assert callable(linalg.solve)
    assert linalg.solve is enp.linalg.solve
