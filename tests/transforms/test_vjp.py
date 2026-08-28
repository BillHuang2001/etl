"""Numerical + contract tests for `etl.vjp` (reverse-mode vector-Jacobian products).

`vjp` is graph→graph: result graphs map `(primal inputs, cotangent inputs)` →
`(primal outputs, input cotangents)` and are executed through the explicit
pipeline (lower → compile → load → run). The binding contract under test lives
in `etl/transforms/CONTEXT.md` ("AD semantics"): cotangent pytree validation
against the output structure, the scalar-one default cotangent (single scalar
output only), ZeroTangent rules, and `TransformError` for ops with no rule.
"""

import numpy as np
import pytest

import etl


def run_graph(graph, *args):
    """Explicit staging: trace-produced graph → lower → compile → load → run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def as_np(value):
    """Recursively convert `etl.Tensor` / containers to numpy values."""
    if isinstance(value, etl.Tensor):
        return value.numpy()
    if isinstance(value, (tuple, list)):
        return tuple(as_np(v) for v in value)
    return np.asarray(value)


def fd_jacobian(np_fn, x0, eps=1e-6):
    """Central-difference Jacobian of a numpy callable returning an ndarray
    or a tuple of ndarrays. Rows follow the flattened outputs."""
    x0 = np.asarray(x0, dtype=np.float64)

    def flat(out):
        if isinstance(out, tuple):
            return np.concatenate([np.asarray(o, np.float64).ravel() for o in out])
        return np.asarray(out, np.float64).ravel()

    jac = np.empty((flat(np_fn(x0)).size, x0.size), dtype=np.float64)
    for i in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        jac[:, i] = (flat(np_fn(xp)) - flat(np_fn(xm))) / (2.0 * eps)
    return jac


# ---------------------------------------------------------------------------
# Default cotangent (scalar-one) equals grad
# ---------------------------------------------------------------------------


def test_vjp_default_cotangent_equals_grad():
    def f(x, y):
        return etl.ops.sum(x * y)

    spec = etl.TensorSpec((3,), etl.float32)
    x = np.array([1.0, 2.0, 3.0], np.float32)
    y = np.array([2.0, 3.0, 4.0], np.float32)

    # Default cotangent = scalar-one (an explicit scalar input of the result
    # graph — the vjp inputs are (primal tree, cotangent tree)).
    graph = etl.vjp(f)(spec, spec)
    primal_out, input_cts = run_graph(
        graph, (x, y), (np.array(1.0, np.float32),)
    )
    np.testing.assert_allclose(as_np(primal_out), np.sum(x * y), rtol=1e-5)
    assert isinstance(input_cts, tuple) and len(input_cts) == 2

    grad_out = run_graph(etl.grad(f, argnums=None)(spec, spec), x, y)
    np.testing.assert_allclose(as_np(input_cts)[0], as_np(grad_out)[0], rtol=1e-5)
    np.testing.assert_allclose(as_np(input_cts)[1], as_np(grad_out)[1], rtol=1e-5)


# ---------------------------------------------------------------------------
# Pullbacks vs cotangent @ Jacobian (finite differences)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "etl_fn,np_fn",
    [
        pytest.param(lambda x: 2.0 * x, lambda x: 2.0 * x, id="linear"),
        pytest.param(lambda x: x * x, lambda x: x**2, id="square"),
    ],
)
def test_vjp_pullback_matches_ct_jacobian(etl_fn, np_fn):
    x0 = np.array([0.2, -0.8, 1.3, 0.5])
    ct = np.array([0.7, -1.2, 0.4, 1.9])
    spec = etl.TensorSpec((4,), etl.float64)

    graph = etl.vjp(etl_fn, spec)(spec)
    primal_out, (got,) = run_graph(graph, (x0,), (ct,))

    np.testing.assert_allclose(as_np(primal_out), np_fn(x0), rtol=1e-7)
    np.testing.assert_allclose(as_np(got), ct @ fd_jacobian(np_fn, x0), rtol=1e-4)


def test_vjp_multi_output_ct_jacobian():
    def f(x):
        return x * x, 2.0 * x

    def np_f(x):
        return x**2, 2.0 * x

    x0 = np.array([0.2, -0.8, 1.3, 0.5])
    ct1 = np.array([0.7, -1.2, 0.4, 1.9])
    ct2 = np.array([-0.3, 0.6, 1.1, -0.9])
    spec = etl.TensorSpec((4,), etl.float64)

    graph = etl.vjp(f, (spec, spec))(spec)
    primal_out, (got,) = run_graph(graph, (x0,), (ct1, ct2))

    np.testing.assert_allclose(as_np(primal_out)[0], np_f(x0)[0], rtol=1e-7)
    np.testing.assert_allclose(as_np(primal_out)[1], np_f(x0)[1], rtol=1e-7)
    expected = np.concatenate([ct1, ct2]) @ fd_jacobian(np_f, x0)
    np.testing.assert_allclose(as_np(got), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Default cotangent validation (scalar output required)
# ---------------------------------------------------------------------------


def test_vjp_default_non_scalar_output_raises():
    def f(x):
        return 2.0 * x

    with pytest.raises(etl.ShapeError, match="scalar output"):
        etl.vjp(f)(etl.TensorSpec((3,), etl.float32))


def test_vjp_default_multi_output_raises():
    def f(x):
        return etl.ops.sum(x * x), etl.ops.sum(2.0 * x)

    with pytest.raises(etl.ShapeError, match="exactly one tensor output"):
        etl.vjp(f)(etl.TensorSpec((3,), etl.float32))


# ---------------------------------------------------------------------------
# Cotangent pytree validation
# ---------------------------------------------------------------------------


def test_vjp_cotangent_shape_mismatch():
    def f(x):
        return etl.ops.sum(x * x)

    with pytest.raises(etl.TransformError, match="shape"):
        etl.vjp(f, etl.TensorSpec((4,), etl.float32))(
            etl.TensorSpec((3,), etl.float32)
        )


def test_vjp_cotangent_dtype_mismatch():
    def f(x):
        return etl.ops.sum(x * x)

    with pytest.raises(etl.TransformError, match="dtype"):
        etl.vjp(f, etl.TensorSpec((), etl.float64))(
            etl.TensorSpec((3,), etl.float32)
        )


def test_vjp_cotangent_structure_mismatch():
    def f(x):
        return x * x, 2.0 * x

    # 1-tuple against a 2-output tree.
    with pytest.raises(etl.TransformError, match="structure"):
        etl.vjp(f, (etl.TensorSpec((3,), etl.float32),))(
            etl.TensorSpec((3,), etl.float32)
        )
    # The bare-spec spelling is only accepted for a single tensor output.
    with pytest.raises(etl.TransformError, match="structure"):
        etl.vjp(f, etl.TensorSpec((3,), etl.float32))(
            etl.TensorSpec((3,), etl.float32)
        )


def test_vjp_cotangent_invalid_entry_type():
    def f(x):
        return x * x, 2.0 * x

    with pytest.raises(etl.TransformError, match="TensorSpec"):
        etl.vjp(f, (etl.TensorSpec((3,), etl.float32), 1.0))(
            etl.TensorSpec((3,), etl.float32)
        )


# ---------------------------------------------------------------------------
# Zero cotangent entries (None seeds an in-graph scalar-one; scalar outputs)
# ---------------------------------------------------------------------------


def test_vjp_none_cotangent_entry_scalar_outputs():
    def f(x):
        return etl.ops.sum(x * x), etl.ops.sum(2.0 * x)

    x = np.array([1.0, 2.0, 3.0], np.float32)
    ct2 = np.array(0.5, np.float32)
    graph = etl.vjp(f, (None, etl.TensorSpec((), etl.float32)))(
        etl.TensorSpec((3,), etl.float32)
    )
    # None entries are static leaves of the cotangent tree: pass them as None.
    primal_out, (got,) = run_graph(graph, (x,), (None, ct2))

    np.testing.assert_allclose(as_np(primal_out)[0], np.sum(x * x), rtol=1e-5)
    np.testing.assert_allclose(as_np(primal_out)[1], np.sum(2.0 * x), rtol=1e-5)
    # None seeds scalar-one on output 0 → pullback = 2x + ct2 * 2.
    np.testing.assert_allclose(as_np(got), 2.0 * x + ct2 * 2.0, rtol=1e-5)


def test_vjp_none_cotangent_entry_non_scalar_raises():
    def f(x):
        return x * x, 2.0 * x

    with pytest.raises(etl.ShapeError, match="seeds a scalar-one"):
        etl.vjp(f, (None, etl.TensorSpec((3,), etl.float32)))(
            etl.TensorSpec((3,), etl.float32)
        )


# ---------------------------------------------------------------------------
# Graph input vs callable input; bare cotangent spellings
# ---------------------------------------------------------------------------


def test_vjp_graph_input_matches_callable():
    def f(x):
        return x * x

    spec = etl.TensorSpec((4,), etl.float64)
    x0 = np.array([0.2, -0.8, 1.3, 0.5])
    ct = np.array([0.7, -1.2, 0.4, 1.9])

    from_callable = etl.vjp(f, spec)(spec)
    from_graph = etl.vjp(etl.trace(f, spec), spec)

    out_c = run_graph(from_callable, (x0,), (ct,))
    out_g = run_graph(from_graph, (x0,), (ct,))
    np.testing.assert_allclose(as_np(out_c)[0], as_np(out_g)[0], rtol=1e-7)
    np.testing.assert_allclose(as_np(out_c)[1][0], as_np(out_g)[1][0], rtol=1e-7)


def test_vjp_bare_cotangent_spellings():
    def f(x):
        return etl.ops.sum(x * x)

    spec = etl.TensorSpec((3,), etl.float32)
    ct_spec = etl.TensorSpec((), etl.float32)
    x = np.array([1.0, 2.0, 3.0], np.float32)
    ct = np.array(1.5, np.float32)

    # Single tensor output: a bare TensorSpec and a 1-tuple are equivalent.
    out_bare = run_graph(etl.vjp(f, ct_spec)(spec), (x,), (ct,))
    out_tuple = run_graph(etl.vjp(f, (ct_spec,))(spec), (x,), (ct,))
    np.testing.assert_allclose(as_np(out_bare)[1][0], as_np(out_tuple)[1][0], rtol=1e-5)
    np.testing.assert_allclose(as_np(out_bare)[1][0], 2.0 * x * ct, rtol=1e-5)


# ---------------------------------------------------------------------------
# stop_gradient contributes a zero cotangent
# ---------------------------------------------------------------------------


def test_vjp_stop_gradient_zero_cotangent():
    def f(x):
        return etl.ops.sum(etl.stop_gradient(x) * x)

    x = np.array([1.0, 2.0, 3.0], np.float32)
    graph = etl.vjp(f)(etl.TensorSpec((3,), etl.float32))
    primal_out, (got,) = run_graph(graph, (x,), (np.array(1.0, np.float32),))

    np.testing.assert_allclose(as_np(primal_out), np.sum(x * x), rtol=1e-5)
    # d/dx [stop_gradient(x) * x] = stop_gradient(x) = x — the stopped path
    # contributes nothing (it is NOT 2x).
    np.testing.assert_allclose(as_np(got), x, rtol=1e-5)


# ---------------------------------------------------------------------------
# sign has a zero derivative a.e. — an implemented VJP rule (not a deferral)
# ---------------------------------------------------------------------------


def test_vjp_sign_gives_zero_cotangent():
    """sign's implemented VJP rule (derivative 0 a.e.) zeroes any cotangent:
    the input cotangent of a vector-output `sign(x)` is all zeros."""
    spec = etl.TensorSpec((4,), np.float64)
    x = np.linspace(-1.0, 1.0, 4)
    c = np.array([0.7, -1.2, 0.4, 1.9])

    graph = etl.vjp(etl.sign, spec)(spec)
    graph.verify()
    primal_out, (got,) = run_graph(graph, (x,), (c,))
    np.testing.assert_allclose(as_np(primal_out), np.sign(x), rtol=1e-7, atol=1e-6)
    np.testing.assert_array_equal(as_np(got), np.zeros(4))


# ---------------------------------------------------------------------------
# Ops with no VJP rule raise TransformError — never a silent fallback
# ---------------------------------------------------------------------------


def test_vjp_runtime_call_raises():
    def f(x):
        return etl.runtime_call(lambda a: a, x, result=etl.TensorSpec((), etl.float32))

    with pytest.raises(etl.TransformError, match="no VJP rule.*runtime_call"):
        etl.vjp(f)(etl.TensorSpec((3,), etl.float32))


# ---------------------------------------------------------------------------
# Concrete-tensor arguments / result kinds
# ---------------------------------------------------------------------------


def test_vjp_concrete_tensor_args_raise():
    def f(x):
        return etl.ops.sum(x * x)

    tf = etl.vjp(f)
    with pytest.raises(etl.TraceError, match="concrete Tensors"):
        tf(etl.tensor(np.array([1.0, 2.0], np.float32)))


def test_vjp_returns_graph():
    def f(x):
        return etl.ops.sum(x * x)

    tf = etl.vjp(f)
    assert tf.kind == "vjp"
    graph = tf(etl.TensorSpec((3,), etl.float32))
    assert isinstance(graph, etl.Graph)
    # Inputs = primal inputs followed by the (default scalar-one) cotangent.
    assert len(graph.tensor_specs) == 2
    assert tuple(graph.tensor_specs[1].shape) == ()
