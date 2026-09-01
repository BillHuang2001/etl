"""Chain rule through composite ops — `grad` / `jvp` / `vjp` validated against
numpy central finite differences.

Every graph is produced by a graph→graph transform (never executed at
transform time), verified, and run through the explicit staging pipeline.
References are central differences computed in float64 with a step matched
to the dtype under test (see the F64/F32 settings).
"""

import numpy as np
import pytest

import etl
from etl.transforms.autodiff import ZeroTangent
from tests.transforms._fd_utils import (
    central_directional,
    central_grad,
    run_grad,
    run_jvp,
    run_vjp,
    to_np,
)

F64 = {"dtype": np.float64, "eps": 1e-6, "rtol": 1e-7, "atol": 1e-6}
F32 = {"dtype": np.float32, "eps": 1e-3, "rtol": 1e-5, "atol": 1e-6}


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_grad_dot_2d_both_args(cfg):
    """grad of sum(dot(x, y)) w.r.t. both 2D operands == finite differences."""
    dt = cfg["dtype"]
    rng = np.random.RandomState(0)
    x = rng.uniform(0.2, 1.2, (3, 4)).astype(dt)
    y = rng.uniform(0.2, 1.2, (4, 3)).astype(dt)
    spec_x = etl.TensorSpec((3, 4), dt)
    spec_y = etl.TensorSpec((4, 3), dt)

    def f(x, y):
        return etl.sum(etl.dot(x, y))

    grads = run_grad(f, (0, 1), (spec_x, spec_y), x, y)
    x64, y64 = x.astype(np.float64), y.astype(np.float64)
    fnp = lambda x, y: np.sum(x @ y)  # noqa: E731
    np.testing.assert_allclose(
        grads[0], central_grad(fnp, (x64, y64), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )
    np.testing.assert_allclose(
        grads[1], central_grad(fnp, (x64, y64), 1, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )


def test_grad_inner_product_1d_scalar():
    """1D·1D → scalar inner product. `etl.dot` is a batched matmul that
    requires rank ≥ 2 by design (asserted below), so the vector dot product
    is composed from ordinary ops (multiply + sum)."""
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(0.2, 1.2, 6)
    y = np.linspace(1.3, 0.4, 6)

    with pytest.raises(etl.ShapeError, match="rank"):
        etl.trace(lambda a, b: etl.dot(a, b), spec, spec)

    def f(x, y):
        return etl.sum(etl.multiply(x, y))

    grads = run_grad(f, (0, 1), (spec, spec), x, y)
    np.testing.assert_allclose(grads[0], y, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(grads[1], x, rtol=1e-7, atol=1e-6)
    fnp = lambda a, b: np.sum(a * b)  # noqa: E731
    np.testing.assert_allclose(
        grads[0], central_grad(fnp, (x, y), 0, 1e-6), rtol=1e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        grads[1], central_grad(fnp, (x, y), 1, 1e-6), rtol=1e-7, atol=1e-6
    )


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_grad_sigmoid_elementwise(cfg):
    dt = cfg["dtype"]
    spec = etl.TensorSpec((6,), dt)
    x = np.linspace(-1.2, 1.2, 6).astype(dt)

    def f(x):
        return etl.sum(etl.sigmoid(x))

    grad = run_grad(f, 0, (spec,), x)
    x64 = x.astype(np.float64)
    fnp = lambda z: np.sum(1.0 / (1.0 + np.exp(-z)))  # noqa: E731
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x64,), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )
    sig = 1.0 / (1.0 + np.exp(-x64))
    np.testing.assert_allclose(
        grad, sig * (1.0 - sig), rtol=cfg["rtol"], atol=cfg["atol"]
    )


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_grad_reduce_sum_square_is_2x(cfg):
    dt = cfg["dtype"]
    spec = etl.TensorSpec((6,), dt)
    x = np.linspace(0.2, 1.2, 6).astype(dt)

    def f(x):
        return etl.reduce_sum(etl.square(x))

    grad = run_grad(f, 0, (spec,), x)
    x64 = x.astype(np.float64)
    np.testing.assert_allclose(grad, 2.0 * x64, rtol=cfg["rtol"], atol=cfg["atol"])
    fnp = lambda z: np.sum(z ** 2)  # noqa: E731
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x64,), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_grad_through_reshape(cfg):
    dt = cfg["dtype"]
    spec = etl.TensorSpec((3, 4), dt)
    x = np.random.RandomState(4).uniform(0.2, 1.2, (3, 4)).astype(dt)

    def f(x):
        return etl.sum(etl.square(etl.reshape(x, (12,))))

    grad = run_grad(f, 0, (spec,), x)
    x64 = x.astype(np.float64)
    fnp = lambda z: np.sum(z.reshape(12) ** 2)  # noqa: E731
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x64,), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )
    assert grad.shape == x.shape


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_grad_through_transpose(cfg):
    dt = cfg["dtype"]
    spec = etl.TensorSpec((3, 4), dt)
    c = np.random.RandomState(5).uniform(0.2, 1.2, (4, 3)).astype(dt)
    x = np.random.RandomState(6).uniform(0.2, 1.2, (3, 4)).astype(dt)

    def f(x):
        return etl.sum(
            etl.multiply(etl.transpose(x), etl.constant(etl.tensor(c)))
        )

    grad = run_grad(f, 0, (spec,), x)
    x64 = x.astype(np.float64)
    c64 = c.astype(np.float64)
    fnp = lambda z: np.sum(z.T * c64)  # noqa: E731
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x64,), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )
    # Analytic: d/dx sum(x^T * c) = c^T (per element).
    np.testing.assert_allclose(grad, c64.T, rtol=cfg["rtol"], atol=cfg["atol"])
    assert grad.shape == x.shape


def test_grad_through_broadcast():
    spec = etl.TensorSpec((3, 1), np.float64)
    b = np.full((1, 4), 3.0)
    x = np.random.RandomState(7).uniform(0.2, 1.2, (3, 1))

    def f(x):
        return etl.sum(etl.multiply(x, etl.constant(etl.tensor(b))))

    grad = run_grad(f, 0, (spec,), x)
    fnp = lambda z: np.sum(z * b)  # noqa: E731
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x,), 0, 1e-6), rtol=1e-7, atol=1e-6
    )


def test_jvp_and_vjp_through_dot():
    """Both AD modes through sum(dot(x, y)); vjp cotangent vs ct @ J."""
    spec_x = etl.TensorSpec((3, 4), np.float64)
    spec_y = etl.TensorSpec((4, 3), np.float64)
    x = np.random.RandomState(10).uniform(0.2, 1.2, (3, 4))
    y = np.random.RandomState(11).uniform(0.2, 1.2, (4, 3))
    tx = np.random.RandomState(12).uniform(-1.0, 1.0, (3, 4))
    ty = np.random.RandomState(13).uniform(-1.0, 1.0, (4, 3))

    def f(x, y):
        return etl.sum(etl.dot(x, y))

    fnp = lambda a, b: np.sum(a @ b)  # noqa: E731

    # jvp vs the central directional derivative.
    _, primal, tangent = run_jvp(f, (spec_x, spec_y), (spec_x, spec_y), (x, y), (tx, ty))
    np.testing.assert_allclose(primal, fnp(x, y), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        tangent[0],
        central_directional(fnp, (x, y), (tx, ty), 1e-6),
        rtol=1e-7, atol=1e-6,
    )

    # vjp with an explicit scalar cotangent ct: input cotangents = ct * grad.
    ct = 1.5
    _, primal, in_cts = run_vjp(
        f, etl.TensorSpec((), np.float64), (spec_x, spec_y), (x, y),
        (np.array(ct),),
    )
    np.testing.assert_allclose(primal, fnp(x, y), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        in_cts[0], ct * central_grad(fnp, (x, y), 0, 1e-6), rtol=1e-7, atol=1e-6
    )
    np.testing.assert_allclose(
        in_cts[1], ct * central_grad(fnp, (x, y), 1, 1e-6), rtol=1e-7, atol=1e-6
    )


def test_jvp_and_vjp_through_sigmoid():
    """Vector-output sigmoid: jvp vs central direction; vjp with a vector
    cotangent vs the finite-difference ct @ J."""
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(-1.2, 1.2, 6)
    t = np.array([0.3, -0.6, 0.9, 0.4, -0.2, 0.7])
    c = np.array([1.0, -2.0, 0.5, 3.0, -0.4, 1.1])

    def f(x):
        return etl.sigmoid(x)

    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731

    _, primal, tangent = run_jvp(f, spec, (spec,), (x,), (t,))
    np.testing.assert_allclose(primal, sig(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        tangent[0], central_directional(sig, (x,), (t,), 1e-6),
        rtol=1e-7, atol=1e-6,
    )

    _, primal, in_cts = run_vjp(f, spec, (spec,), (x,), (c,))
    np.testing.assert_allclose(primal, sig(x), rtol=1e-7, atol=1e-6)
    fnp = lambda z: np.sum(c * sig(z))  # noqa: E731
    np.testing.assert_allclose(
        in_cts[0], central_grad(fnp, (x,), 0, 1e-6), rtol=1e-7, atol=1e-6
    )
    # Analytic: J is diagonal for elementwise sigmoid, so J^T c = c * sig'.
    np.testing.assert_allclose(
        in_cts[0], c * sig(x) * (1.0 - sig(x)), rtol=1e-7, atol=1e-6
    )


def test_jvp_and_vjp_through_reshape():
    spec = etl.TensorSpec((3, 4), np.float64)
    x = np.random.RandomState(20).uniform(0.2, 1.2, (3, 4))
    t = np.random.RandomState(21).uniform(-1.0, 1.0, (3, 4))

    def f(x):
        return etl.sum(etl.square(etl.reshape(x, (12,))))

    fnp = lambda z: np.sum(z.reshape(12) ** 2)  # noqa: E731

    _, primal, tangent = run_jvp(f, spec, (spec,), (x,), (t,))
    np.testing.assert_allclose(primal, fnp(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        tangent[0], central_directional(fnp, (x,), (t,), 1e-6),
        rtol=1e-7, atol=1e-6,
    )

    ct = 2.0
    _, primal, in_cts = run_vjp(
        f, etl.TensorSpec((), np.float64), (spec,), (x,), (np.array(ct),)
    )
    np.testing.assert_allclose(primal, fnp(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        in_cts[0], ct * central_grad(fnp, (x,), 0, 1e-6), rtol=1e-7, atol=1e-6
    )
    assert in_cts[0].shape == x.shape


def test_jvp_and_vjp_through_transpose():
    spec = etl.TensorSpec((3, 4), np.float64)
    c = np.random.RandomState(22).uniform(0.2, 1.2, (4, 3))
    x = np.random.RandomState(23).uniform(0.2, 1.2, (3, 4))
    t = np.random.RandomState(24).uniform(-1.0, 1.0, (3, 4))

    def f(x):
        return etl.sum(
            etl.multiply(etl.transpose(x), etl.constant(etl.tensor(c)))
        )

    fnp = lambda z: np.sum(z.T * c)  # noqa: E731

    _, primal, tangent = run_jvp(f, spec, (spec,), (x,), (t,))
    np.testing.assert_allclose(primal, fnp(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        tangent[0], central_directional(fnp, (x,), (t,), 1e-6),
        rtol=1e-7, atol=1e-6,
    )

    ct = 0.7
    _, primal, in_cts = run_vjp(
        f, etl.TensorSpec((), np.float64), (spec,), (x,), (np.array(ct),)
    )
    np.testing.assert_allclose(primal, fnp(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(
        in_cts[0], ct * central_grad(fnp, (x,), 0, 1e-6), rtol=1e-7, atol=1e-6
    )


def test_grad_stop_gradient_x_times_const_is_x():
    """f(x) = sum(x * stop_gradient(x)) differentiates like sum(x * const):
    the gradient is x — NOT the 2x that the numerical function sum(x**2)
    would give (that deviation is exactly what blocking means)."""
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(0.2, 1.2, 6)

    def f(x):
        return etl.sum(etl.multiply(x, etl.stop_gradient(x)))

    grad = run_grad(f, 0, (spec,), x)
    np.testing.assert_allclose(grad, x, rtol=1e-7, atol=1e-6)


def test_grad_stop_gradient_blocks_gradient_flow():
    """f(x) = sum(stop_gradient(x**2)) — the gradient is exactly zero."""
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(0.2, 1.2, 6)

    def f(x):
        return etl.sum(etl.stop_gradient(etl.square(x)))

    grad = run_grad(f, 0, (spec,), x)
    assert np.all(grad == 0.0)


def test_jvp_stop_gradient_gives_zero_tangent():
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(0.2, 1.2, 6)
    t = np.full(6, 0.5)

    def f(x):
        return etl.sum(etl.stop_gradient(etl.square(x)))

    _, primal, tangent = run_jvp(f, spec, (spec,), (x,), (t,))
    np.testing.assert_allclose(primal, np.sum(x ** 2), rtol=1e-7, atol=1e-6)
    assert np.asarray(tangent[0]) == 0.0


CHAIN_CASES = {
    "tanh_sigmoid": (
        lambda x: etl.sum(etl.tanh(etl.sigmoid(x))),
        lambda z: np.sum(np.tanh(1.0 / (1.0 + np.exp(-z)))),
    ),
    "exp_sin": (
        lambda x: etl.sum(etl.exp(etl.sin(x))),
        lambda z: np.sum(np.exp(np.sin(z))),
    ),
}


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", list(CHAIN_CASES))
def test_chain_multiple_elementwise_ops(name, cfg):
    """Gradient through several chained elementwise ops in one graph."""
    etl_fn, np_fn = CHAIN_CASES[name]
    dt = cfg["dtype"]
    spec = etl.TensorSpec((6,), dt)
    x = np.linspace(-0.8, 0.8, 6).astype(dt)

    grad = run_grad(etl_fn, 0, (spec,), x)
    x64 = x.astype(np.float64)
    np.testing.assert_allclose(
        grad, central_grad(np_fn, (x64,), 0, cfg["eps"]),
        rtol=cfg["rtol"], atol=cfg["atol"],
    )


def test_nondifferentiable_output_op_gives_zero_gradient():
    """argmax has a builtin ZeroTangent rule: zero gradient, not an error."""
    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(-1.0, 1.0, 6)

    def f(x):
        return etl.sum(etl.cast(etl.argmax(x), etl.float64))

    grad = run_grad(f, 0, (spec,), x)
    assert np.all(grad == 0.0)


def test_conv_vjp_is_a_deferred_transform_error():
    """conv VJP is a documented v1 deferral: TransformError, never silent."""
    spec_x = etl.TensorSpec((1, 1, 4, 4), np.float64)
    spec_w = etl.TensorSpec((1, 1, 2, 2), np.float64)

    def f(x, w):
        return etl.sum(etl.conv(x, w, padding="SAME"))

    with pytest.raises(etl.TransformError, match="conv"):
        etl.grad(f)(spec_x, spec_w)


def test_runtime_call_grad_is_a_transform_error():
    spec = etl.TensorSpec((4,), np.float64)

    def f(x):
        return etl.sum(etl.runtime_call(lambda v: v, x, result=spec))

    with pytest.raises(etl.TransformError, match="runtime_call"):
        etl.grad(f)(spec)


def test_defn_evaluate_and_grad_shorthand():
    """@etl.defn functions work with the documented evaluate shorthand and
    with the grad TransformCallable."""

    @etl.defn
    def f(x):
        return etl.sum(etl.tanh(etl.sigmoid(x)))

    spec = etl.TensorSpec((6,), np.float64)
    x = np.linspace(-0.8, 0.8, 6)
    fnp = lambda z: np.sum(np.tanh(1.0 / (1.0 + np.exp(-z))))  # noqa: E731

    np.testing.assert_allclose(to_np(etl.evaluate(f, x)), fnp(x), rtol=1e-7, atol=1e-6)
    grad = run_grad(f, 0, (spec,), x)
    np.testing.assert_allclose(
        grad, central_grad(fnp, (x,), 0, 1e-6), rtol=1e-7, atol=1e-6
    )


def test_transformed_graphs_contain_only_ordinary_ops():
    """grad/jvp/vjp results are ordinary graphs of ordinary ops — no
    autodiff op names leak into the IR."""
    spec = etl.TensorSpec((6,), np.float64)

    def f(x):
        return etl.sum(etl.tanh(etl.sigmoid(x)))

    x = np.linspace(-0.8, 0.8, 6)
    graphs = [
        etl.grad(f, 0)(spec),
        etl.jvp(f, spec)(spec),
        etl.vjp(f, etl.TensorSpec((), np.float64))(spec),
    ]
    for graph in graphs:
        graph.verify()
        op_names = {op.name for op in graph.module.main.entry_block.ops}
        assert not op_names & {"jvp", "vjp", "vectorize", "vmap"}


# ---------------------------------------------------------------------------
# external-kernel rule channel: AD (external:<name> vjp/jvp keys)
# ---------------------------------------------------------------------------
#
# `external_call` ops resolve derivative rules under the namespaced
# `external:<name>` key (mirroring `block:<name>` for block calls), registered
# through the handle returned by `etl.register_external_kernel`. Without an
# explicit rule, a registered portable decomposition (`@handle.portable`) is
# the fallback: grad/vjp inline it and run a LOCAL reverse sweep seeded on the
# decomposition's outputs (the post-inline seeding pattern); jvp derives from
# the vjp rule via the adjoint. Names are `tx_`-prefixed to stay unique in the
# process-wide registries.


def _sym(value):
    """Wrap an ir.Value as a SymbolicTensor so etl.ops can build on it."""
    return etl.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


EXT_DIM = 4


def _ext_sq(x):
    """The registered kernel: y = x*x (elementwise)."""
    return x * x


ext_sq = etl.register_external_kernel("tx_ext_sq", _ext_sq)


@ext_sq.vjp_rule
def _ext_sq_vjp(op, cotangents, primals):
    """Explicit vjp rule for y = x²: input cotangent = 2 * ct * primal."""
    (ct,) = cotangents
    if ct is None or isinstance(ct, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(etl.multiply(_sym(ct), 2), _sym(primals[0])).value,)


@etl.defn
def _ext_sq_vec(x):
    """Vector-output kernel call: y = x*x, shape (EXT_DIM,)."""
    return etl.external_call(
        "tx_ext_sq", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


@etl.defn
def _ext_sq_loss(x):
    return etl.sum(
        etl.external_call(
            "tx_ext_sq", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
        )
    )


def _ext_sq2(x):
    """Second square kernel: gets BOTH an explicit vjp and jvp rule."""
    return x * x


ext_sq2 = etl.register_external_kernel("tx_ext_sq2", _ext_sq2)


@ext_sq2.vjp_rule
def _ext_sq2_vjp(op, cotangents, primals):
    (ct,) = cotangents
    if ct is None or isinstance(ct, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(etl.multiply(_sym(ct), 2), _sym(primals[0])).value,)


@ext_sq2.jvp_rule
def _ext_sq2_jvp(op, tangents):
    """Explicit jvp rule for y = x²: output tangent = 2 * x * t."""
    (t,) = tangents
    if t is None or isinstance(t, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(etl.multiply(_sym(t), 2), _sym(op.operands[0])).value,)


@etl.defn
def _ext_sq2_vec(x):
    return etl.external_call(
        "tx_ext_sq2", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


def _ext_swish(x):
    """The registered kernel: the swish activation."""
    s = 1.0 / (1.0 + np.exp(-x))
    return s * x


def _ext_swish_deriv(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


ext_swish = etl.register_external_kernel("tx_ext_swish", _ext_swish)


@ext_swish.portable
@etl.defn
def _ext_swish_portable(
    x: etl.TensorSpec((EXT_DIM,), etl.float32),
) -> etl.TensorSpec((EXT_DIM,), etl.float32):
    return etl.sigmoid(x) * x


@etl.defn
def _ext_swish_vec(x):
    return etl.external_call(
        "tx_ext_swish", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


@etl.defn
def _ext_swish_loss(x):
    return etl.sum(
        etl.external_call(
            "tx_ext_swish", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
        )
    )


def _ext_x():
    return np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)


# --- explicit rules ----------------------------------------------------------


def test_grad_external_call_explicit_vjp_rule():
    """grad over external_call with an explicit vjp rule under
    external:<name>: d/dx sum(x²) == 2x."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    grad = run_grad(_ext_sq_loss, 0, (spec,), _ext_x())
    np.testing.assert_allclose(grad, 2 * _ext_x(), rtol=1e-5, atol=1e-6)


def test_vjp_external_call_explicit_vjp_rule():
    """vjp over the vector output with an explicit cotangent: input
    cotangent == ct * 2x."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    ct = np.array([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
    graph, primal, in_cts = run_vjp(_ext_sq_vec, spec, (spec,), (_ext_x(),), (ct,))
    np.testing.assert_allclose(primal, _ext_x() * _ext_x(), rtol=1e-6)
    np.testing.assert_allclose(in_cts[0], ct * 2 * _ext_x(), rtol=1e-5, atol=1e-6)


def test_jvp_external_call_derived_from_vjp_rule():
    """No explicit jvp rule: jvp derives from the explicit vjp rule via the
    adjoint — output tangent == 2x·dx on the vector output."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    dx = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    graph, primal, tangent = run_jvp(_ext_sq_vec, spec, (spec,), (_ext_x(),), (dx,))
    np.testing.assert_allclose(primal, _ext_x() * _ext_x(), rtol=1e-6)
    np.testing.assert_allclose(tangent[0], 2 * _ext_x() * dx, rtol=1e-5, atol=1e-6)


def test_jvp_external_call_explicit_jvp_rule():
    """An explicitly registered jvp rule (external:<name>) drives forward
    mode directly."""
    from etl import transforms

    assert transforms.jvp_rules["external:tx_ext_sq2"] is _ext_sq2_jvp
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    dx = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
    graph, primal, tangent = run_jvp(_ext_sq2_vec, spec, (spec,), (_ext_x(),), (dx,))
    np.testing.assert_allclose(primal, _ext_x() * _ext_x(), rtol=1e-6)
    np.testing.assert_allclose(tangent[0], 2 * _ext_x() * dx, rtol=1e-5, atol=1e-6)


# --- portable-decomposition fallbacks ----------------------------------------


def test_grad_external_call_portable_fallback():
    """No explicit rule: grad inlines the portable decomposition and
    differentiates it — d/dx sum(swish(x)) == swish'(x)."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    grad = run_grad(_ext_swish_loss, 0, (spec,), _ext_x())
    np.testing.assert_allclose(grad, _ext_swish_deriv(_ext_x()), rtol=1e-5, atol=1e-5)


def test_vjp_external_call_portable_fallback_real_cotangent():
    """vjp through the portable fallback with an explicit (non-scalar)
    cotangent: the reverse sweep seeds on the INLINED decomposition outputs
    (the post-inline seeding pattern) — input cotangent == swish'(x) * ct."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    ct = np.array([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
    graph, primal, in_cts = run_vjp(_ext_swish_vec, spec, (spec,), (_ext_x(),), (ct,))
    np.testing.assert_allclose(primal, _ext_swish(_ext_x()), rtol=1e-6)
    np.testing.assert_allclose(
        in_cts[0], _ext_swish_deriv(_ext_x()) * ct, rtol=1e-5, atol=1e-5
    )


# --- nested external_call inside a block portable -----------------------------


def _ext_inner(x):
    """The kernel called INSIDE the block's portable decomposition."""
    return x * x


ext_inner = etl.register_external_kernel("tx_ext_inner", _ext_inner)


@ext_inner.vjp_rule
def _ext_inner_vjp(op, cotangents, primals):
    (ct,) = cotangents
    if ct is None or isinstance(ct, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(etl.multiply(_sym(ct), 2), _sym(primals[0])).value,)


@etl.block
@etl.defn
def tx_ext_nested(
    x: etl.TensorSpec((EXT_DIM,), etl.float32),
) -> etl.TensorSpec((EXT_DIM,), etl.float32):
    return etl.external_call(
        "tx_ext_inner", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


@etl.defn
def _ext_nested_loss(x):
    return etl.sum(tx_ext_nested(x))


def test_grad_block_portable_with_nested_external_call():
    """grad through a block portable whose body contains an external_call:
    the block vjp fallback sweep resolves the nested op's OWN
    external:<name> rule — d/dx sum(x²) == 2x."""
    spec = etl.TensorSpec((EXT_DIM,), np.float32)
    grad = run_grad(_ext_nested_loss, 0, (spec,), _ext_x())
    np.testing.assert_allclose(grad, 2 * _ext_x(), rtol=1e-5, atol=1e-6)
