"""Numerical + contract tests for `etl.grad` (reverse-mode gradients).

`grad` is graph→graph: every test builds the gradient graph through the
documented spec-callable convention (or from a pre-traced `Graph`), then
executes it via the explicit pipeline (lower → compile → load → run). The
binding contract under test lives in `etl/transforms/CONTEXT.md` ("AD
semantics"): exactly one scalar output, VJP-rule-driven backward sweep,
ZeroTangent rules for bool/int-producing ops, `TransformError` for ops with no
rule, `TraceError` for concrete-tensor arguments to a `TransformCallable`.
"""

import numpy as np
import pytest

import etl


def run_graph(graph, *args):
    """Explicit staging: trace-produced graph → lower → compile → load → run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def as_np(value):
    """Recursively convert `etl.Tensor` / containers to numpy values.

    `etl.Tensor` deliberately exposes `.numpy()` (not `__array__`), so plain
    `np.asarray` does not convert it.
    """
    if isinstance(value, etl.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return tuple(as_np(v) for v in value)
    return np.asarray(value)


def first(value):
    """Unwrap the 1-tuple `grad` returns for a single input with `argnums`
    spelled as `None` / a sequence (bare tensor for an int argnum)."""
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


# ---------------------------------------------------------------------------
# Elementwise quadratic: f(x) = x**2 + 3x  →  f'(x) = 2x + 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [etl.float32, etl.float64])
@pytest.mark.parametrize("x_value", [0.0, 1.0, -2.0, 3.0, 0.5, -3.5])
def test_grad_quadratic_scalar_input(dtype, x_value):
    rtol = 1e-5 if dtype == etl.float32 else 1e-7

    def f(x):
        return x * x + 3.0 * x

    graph = etl.grad(f)(etl.TensorSpec((), dtype))
    out = run_graph(graph, np.array(x_value, dtype=dtype))
    np.testing.assert_allclose(as_np(out), 2.0 * x_value + 3.0, rtol=rtol)


@pytest.mark.parametrize("dtype", [etl.float32, etl.float64])
@pytest.mark.parametrize(
    "x",
    [
        np.array([0.0, 1.0, -2.0, 3.0]),
        np.array([-0.5, 2.5, -3.0, 0.25]),
        np.array([4.0, -4.0, 1.0, -1.0]),
    ],
    ids=["mixed", "quarters", "unit-ish"],
)
def test_grad_quadratic_vector_input(dtype, x):
    rtol = 1e-5 if dtype == etl.float32 else 1e-7
    x = x.astype(dtype)

    def f(x):
        return etl.ops.sum(x * x + 3.0 * x)

    graph = etl.grad(f)(etl.TensorSpec((4,), dtype))
    out = first(as_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, 2.0 * x + 3.0, rtol=rtol)


def test_grad_defn_input():
    @etl.defn
    def f(x):
        return etl.ops.sum(x * x + 3.0 * x)

    x = np.array([0.0, 1.0, -2.0, 3.0], np.float32)
    graph = etl.grad(f)(etl.TensorSpec((4,), etl.float32))
    out = first(as_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, 2.0 * x + 3.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# argnums selection / output structure
# ---------------------------------------------------------------------------


def test_grad_multi_arg_argnums_and_structure():
    def f(x, y):
        return etl.ops.sum(x * y)

    sx = etl.TensorSpec((3,), etl.float32)
    sy = etl.TensorSpec((3,), etl.float32)
    x = np.array([1.0, 2.0, 3.0], np.float32)
    y = np.array([2.0, 3.0, 4.0], np.float32)

    # int argnum → a single bare gradient tensor.
    out = run_graph(etl.grad(f, argnums=0)(sx, sy), x, y)
    assert isinstance(out, etl.Tensor)
    np.testing.assert_allclose(as_np(out), y, rtol=1e-5)

    out = run_graph(etl.grad(f, argnums=1)(sx, sy), x, y)
    assert isinstance(out, etl.Tensor)
    np.testing.assert_allclose(as_np(out), x, rtol=1e-5)

    # Sequence argnums → a tuple of gradients (one per selected input).
    out = run_graph(etl.grad(f, argnums=(0, 1))(sx, sy), x, y)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(as_np(out)[0], y, rtol=1e-5)
    np.testing.assert_allclose(as_np(out)[1], x, rtol=1e-5)

    # None = all tensor inputs → same tuple of gradients.
    out = run_graph(etl.grad(f, argnums=None)(sx, sy), x, y)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(as_np(out)[0], y, rtol=1e-5)
    np.testing.assert_allclose(as_np(out)[1], x, rtol=1e-5)

    # A list spelling is accepted too; like None, it yields a tuple of
    # gradients (one per selected input — here a 1-tuple).
    out = run_graph(etl.grad(f, argnums=[1])(sx, sy), x, y)
    assert isinstance(out, tuple) and len(out) == 1
    np.testing.assert_allclose(as_np(out)[0], x, rtol=1e-5)


def test_grad_three_inputs_partial_argnums():
    def f(a, b, c):
        return etl.ops.sum(a * b + c * c)

    specs = (etl.TensorSpec((4,), etl.float32),) * 3
    a = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    b = np.array([0.5, -0.5, 2.0, -2.0], np.float32)
    c = np.array([1.0, -1.0, 3.0, 2.0], np.float32)

    out = run_graph(etl.grad(f, argnums=(1, 2))(*specs), a, b, c)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(as_np(out)[0], a, rtol=1e-5)  # d/db
    np.testing.assert_allclose(as_np(out)[1], 2.0 * c, rtol=1e-5)  # d/dc


def test_grad_multi_arg_finite_difference():
    def f(x, y):
        return etl.ops.sum(etl.ops.sin(x) * y * y + x * y)

    def np_f(x, y):
        return float(np.sum(np.sin(x) * y**2 + x * y))

    x0 = np.array([0.3, -0.7, 1.1, -0.4])
    y0 = np.array([0.5, 0.9, -0.2, 0.8])
    graph = etl.grad(f, argnums=(0, 1))(
        etl.TensorSpec((4,), etl.float64), etl.TensorSpec((4,), etl.float64)
    )
    got_x, got_y = as_np(run_graph(graph, x0, y0))
    eps = 1e-6
    for index, got in ((0, got_x), (1, got_y)):
        grad = np.empty_like(got)
        for i in range(got.size):
            args = [x0.copy(), y0.copy()]
            orig = args[index][i]
            args[index][i] = orig + eps
            fp = np_f(*args)
            args[index][i] = orig - eps
            fm = np_f(*args)
            grad[i] = (fp - fm) / (2.0 * eps)
        np.testing.assert_allclose(got, grad, rtol=1e-4)


def test_grad_dot_rule():
    def f(x, w):
        return etl.ops.sum(etl.ops.dot(x, w))

    x = np.array([[1.0, 2.0, 3.0]], np.float32)  # (1, 3)
    w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)  # (3, 2)
    graph = etl.grad(f, argnums=(0, 1))(
        etl.TensorSpec((1, 3), etl.float32), etl.TensorSpec((3, 2), etl.float32)
    )
    got_x, got_w = as_np(run_graph(graph, x, w))
    np.testing.assert_allclose(got_x, w.sum(axis=1)[None, :], rtol=1e-5)  # d/dx[0,i] = sum_j W[i,j]
    np.testing.assert_allclose(got_w, np.tile(x.T, (1, 2)), rtol=1e-5)  # d/dW[i,j] = x[0,i]


# ---------------------------------------------------------------------------
# Static (non-tensor) arguments are excluded from the argnums numbering
# ---------------------------------------------------------------------------


def test_grad_static_arg_numbering():
    def f(x, w, y):
        # w is a Python float → static: not a tensor input, not an argnum.
        return etl.ops.sum(x * w + y * y)

    spec = etl.TensorSpec((3,), etl.float32)
    x = np.array([1.0, 2.0, 3.0], np.float32)
    w = 2.5
    y = np.array([0.5, -1.0, 2.0], np.float32)

    # argnums=1 indexes the SECOND TENSOR input (y), skipping the static w.
    out = run_graph(etl.grad(f, argnums=1)(spec, w, spec), x, w, y)
    assert isinstance(out, etl.Tensor)
    np.testing.assert_allclose(as_np(out), 2.0 * y, rtol=1e-5)

    # A static-only argument position cannot be selected: the graph has two
    # tensor inputs, so argnums=2 is out of range.
    with pytest.raises(etl.TransformError, match="out of range"):
        etl.grad(f, argnums=2)(spec, w, spec)

    # Static values still participate in execution (recorded as static leaves).
    out = run_graph(etl.grad(f, argnums=0)(spec, w, spec), x, w, y)
    assert isinstance(out, etl.Tensor)
    np.testing.assert_allclose(as_np(out), np.full(3, w), rtol=1e-5)


# ---------------------------------------------------------------------------
# Output-shape requirements (binding: grad needs exactly one scalar output)
# ---------------------------------------------------------------------------


def test_grad_non_scalar_output_raises():
    def f(x):
        return x * x

    with pytest.raises(etl.ShapeError, match="scalar"):
        etl.grad(f)(etl.TensorSpec((3,), etl.float32))


def test_grad_multi_output_raises():
    def f(x):
        return etl.ops.sum(x * x), etl.ops.sum(x)

    with pytest.raises(etl.ShapeError, match="exactly one tensor output"):
        etl.grad(f)(etl.TensorSpec((3,), etl.float32))


# ---------------------------------------------------------------------------
# ZeroTangent rules (documented behavior, NOT an error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_builder",
    [
        pytest.param(
            lambda: (lambda x: etl.ops.cast(etl.argmax(x, axis=0), etl.float32)),
            id="argmax",
        ),
        pytest.param(
            lambda: (
                lambda x: etl.ops.sum(
                    etl.ops.cast(etl.ops.equal(x, 2.0), etl.float32)
                )
            ),
            id="equal",
        ),
        pytest.param(
            lambda: (
                lambda x: etl.ops.sum(
                    etl.ops.cast(etl.ops.less(x, 2.0), etl.float32)
                )
            ),
            id="less",
        ),
    ],
)
def test_grad_zero_tangent_rules(fn_builder):
    f = fn_builder()
    x = np.array([1.0, 2.0, 3.0], np.float32)
    graph = etl.grad(f)(etl.TensorSpec((3,), etl.float32))
    out = first(as_np(run_graph(graph, x)))
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.zeros(3, np.float32))


# ---------------------------------------------------------------------------
# sign has a zero derivative a.e. — an implemented VJP rule (not a deferral)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [etl.float32, etl.float64])
def test_grad_sign_is_all_zeros(dtype):
    def f(x):
        return etl.ops.sum(etl.sign(x))

    x = np.array([1.0, -2.0, 3.0, -0.5], dtype=dtype)
    graph = etl.grad(f)(etl.TensorSpec((4,), dtype))
    out = first(as_np(run_graph(graph, x)))
    np.testing.assert_array_equal(out, np.zeros(4, dtype=dtype))


def test_grad_mixed_sign_times_x_gradient_is_sign():
    # sum(x * sign(x)) = sum(abs(x)) → d/dx = sign(x) for x ≠ 0; verified
    # analytically and against a central difference (both in float64).
    def f(x):
        return etl.ops.sum(etl.multiply(x, etl.sign(x)))

    x = np.array([1.5, -2.0, 0.5, -3.0])  # avoid the kink at 0
    graph = etl.grad(f)(etl.TensorSpec((4,), np.float64))
    out = first(as_np(run_graph(graph, x)))
    np.testing.assert_array_equal(out, np.sign(x))

    def fnp(z):
        return float(np.sum(np.abs(z)))

    eps = 1e-6
    grad = np.empty_like(out)
    for i in range(out.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (fnp(xp) - fnp(xm)) / (2.0 * eps)
    np.testing.assert_allclose(out, grad, rtol=1e-7, atol=1e-6)


# ---------------------------------------------------------------------------
# Ops with no VJP rule raise TransformError — never a silent fallback
# ---------------------------------------------------------------------------


def test_grad_runtime_call_raises():
    def f(x):
        return etl.runtime_call(lambda a: a, x, result=etl.TensorSpec((), etl.float32))

    with pytest.raises(etl.TransformError, match="no VJP rule.*runtime_call"):
        etl.grad(f)(etl.TensorSpec((3,), etl.float32))


# ---------------------------------------------------------------------------
# Graph input vs callable input produce the same gradient graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(lambda x: etl.ops.sum(x * x + 3.0 * x), id="quadratic"),
        pytest.param(lambda x: etl.ops.sum(etl.ops.sin(x) * x), id="sin-times-x"),
    ],
)
def test_grad_graph_input_matches_callable(fn):
    spec = etl.TensorSpec((4,), etl.float32)
    x = np.array([0.2, -1.3, 0.7, 1.9], np.float32)

    from_callable = etl.grad(fn)(spec)
    from_graph = etl.grad(etl.trace(fn, spec))

    np.testing.assert_allclose(
        first(as_np(run_graph(from_graph, x))),
        first(as_np(run_graph(from_callable, x))),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argnums", [2, -1, True, "0", 1.5])
def test_grad_invalid_argnums_raise(argnums):
    def f(x, y):
        return etl.ops.sum(x * y)

    spec = etl.TensorSpec((3,), etl.float32)
    with pytest.raises(etl.TransformError):
        etl.grad(f, argnums=argnums)(spec, spec)


def test_grad_integer_input_raises():
    def f(x):
        return etl.ops.sum(etl.ops.cast(x, etl.float32))

    with pytest.raises(etl.TransformError, match="floating-point or complex"):
        etl.grad(f, argnums=0)(etl.TensorSpec((3,), etl.int32))


def test_grad_concrete_tensor_args_raise():
    def f(x):
        return etl.ops.sum(x * x)

    tf = etl.grad(f)
    with pytest.raises(etl.TraceError, match="concrete Tensors"):
        tf(etl.tensor(np.array([1.0, 2.0], np.float32)))
    with pytest.raises(etl.TraceError):
        tf(np.array([1.0, 2.0], np.float32))


# ---------------------------------------------------------------------------
# Result kinds
# ---------------------------------------------------------------------------


def test_grad_returns_graph():
    def f(x):
        return etl.ops.sum(x * x)

    tf = etl.grad(f)
    assert tf.kind == "grad"
    graph = tf(etl.TensorSpec((3,), etl.float32))
    assert isinstance(graph, etl.Graph)
