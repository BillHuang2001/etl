"""Forward-mode AD (`etl.jvp`) — numerical validation against central finite
differences, tangent-argument semantics, and graph-form equivalence.

`jvp` is graph→graph: the result is an ordinary graph mapping
(primal inputs, tangent inputs) → (primal outputs, tangent outputs). Every
numerical check here runs through the explicit staging pipeline
(lower → compile → load → run); the transformed graph is always verified and
must contain only ordinary ops.
"""

import numpy as np
import pytest

import etl
from tests.transforms._fd_utils import central_directional, run_graph, run_jvp, to_np

# Reference settings: float32 references are computed in float64 with a
# coarser step (float32 rounding would dominate a 1e-6 step); float64 uses a
# tight step and tight tolerances.
F64 = {"dtype": np.float64, "eps": 1e-6, "rtol": 1e-7, "atol": 1e-6}
F32 = {"dtype": np.float32, "eps": 1e-3, "rtol": 1e-5, "atol": 1e-6}

SHAPE = (6,)
TANGENT = np.array([0.3, -0.6, 0.9, 0.4, -0.2, 0.7])

# case name -> (etl function, numpy reference)
ELEMENTWISE_CASES = {
    "square": (etl.square, lambda x: x ** 2),
    "sum_square": (lambda x: etl.sum(etl.square(x)), lambda x: np.sum(x ** 2)),
    "sin": (etl.sin, np.sin),
    "cos": (etl.cos, np.cos),
    "exp": (etl.exp, np.exp),
    "cube_sum": (lambda x: etl.sum(etl.power(x, 3)), lambda x: np.sum(x ** 3)),
}


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
@pytest.mark.parametrize("name", list(ELEMENTWISE_CASES))
def test_jvp_matches_central_difference(name, cfg):
    """jvp(f, tangent)(x) == (f(x + eps*t) - f(x - eps*t)) / (2*eps),
    with the primal output returned alongside the tangent output."""
    etl_fn, np_fn = ELEMENTWISE_CASES[name]
    dt = cfg["dtype"]
    spec = etl.TensorSpec(SHAPE, dt)
    x = np.linspace(0.2, 1.2, SHAPE[0]).astype(dt)
    t = TANGENT.astype(dt)

    graph, primal, tangent = run_jvp(etl_fn, spec, (spec,), (x,), (t,))

    # Reference computed in float64 (see F32 comment above).
    x64 = x.astype(np.float64)
    t64 = t.astype(np.float64)
    np.testing.assert_allclose(
        primal, np_fn(x64), rtol=cfg["rtol"], atol=cfg["atol"]
    )
    num_tangent = central_directional(np_fn, (x64,), (t64,), cfg["eps"])
    np.testing.assert_allclose(
        tangent[0], num_tangent, rtol=cfg["rtol"], atol=cfg["atol"]
    )

    # Tangent output shape matches the primal output shape.
    assert np.asarray(tangent[0]).shape == np.asarray(primal).shape
    # The transformed graph contains only ordinary ops.
    op_names = {op.name for op in graph.module.main.entry_block.ops}
    assert not op_names & {"jvp", "vjp", "vectorize", "vmap"}


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_jvp_two_tangents_elementwise_multiply(cfg):
    """Two tangent inputs: tangent output == b*ta + a*tb (product rule)."""
    dt = cfg["dtype"]
    spec = etl.TensorSpec((5,), dt)
    a = np.linspace(0.2, 1.2, 5).astype(dt)
    b = np.linspace(1.3, 0.4, 5).astype(dt)
    ta = np.array([0.1, -0.3, 0.5, -0.2, 0.6], dtype=dt)
    tb = np.array([-0.4, 0.7, 0.2, -0.9, 0.3], dtype=dt)

    graph, primal, tangent = run_jvp(
        etl.multiply, (spec, spec), (spec, spec), (a, b), (ta, tb)
    )

    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    ta64, tb64 = ta.astype(np.float64), tb.astype(np.float64)
    np.testing.assert_allclose(primal, a64 * b64, rtol=cfg["rtol"], atol=cfg["atol"])
    num = central_directional(
        lambda x, y: x * y, (a64, b64), (ta64, tb64), cfg["eps"]
    )
    np.testing.assert_allclose(
        tangent[0], num, rtol=cfg["rtol"], atol=cfg["atol"]
    )
    np.testing.assert_allclose(
        tangent[0], b64 * ta64 + a64 * tb64, rtol=cfg["rtol"], atol=cfg["atol"]
    )


@pytest.mark.parametrize("none_pos", [0, 1], ids=["first_none", "second_none"])
def test_jvp_none_tangent_is_zero_tangent(none_pos):
    """A `None` tangent is a zero tangent: with `f(a, b) = sg(a) * sg(b)`
    both `(None, spec)` and `(spec, None)` give an all-zeros tangent output."""
    spec = etl.TensorSpec((5,), np.float64)

    def f(a, b):
        return etl.multiply(etl.stop_gradient(a), etl.stop_gradient(b))

    tangents = [spec, spec]
    tangents[none_pos] = None
    a = np.linspace(0.2, 1.2, 5)
    b = np.linspace(1.3, 0.4, 5)
    ts = [np.full(5, 0.4), np.full(5, -0.7)]
    ts[none_pos] = None

    _, primal, tangent = run_jvp(f, tuple(tangents), (spec, spec), (a, b), tuple(ts))

    np.testing.assert_allclose(primal, a * b, rtol=1e-7, atol=1e-6)
    got = np.asarray(tangent[0])
    assert got.shape == (5,)
    assert np.all(got == 0.0)


def test_jvp_none_tangent_contributes_only_zero_to_its_input():
    """Control for the zero-tangent semantics: `f(a, b) = a * b` with
    tangent `(None, tb)` must equal the analytic derivative with `ta = 0`
    (i.e. `a * tb` — only b's tangent flows through)."""
    spec = etl.TensorSpec((5,), np.float64)

    def f(a, b):
        return etl.multiply(a, b)

    a = np.linspace(0.2, 1.2, 5)
    b = np.linspace(1.3, 0.4, 5)
    tb = np.full(5, -0.7)

    _, primal, tangent = run_jvp(f, (None, spec), (spec, spec), (a, b), (None, tb))

    np.testing.assert_allclose(primal, a * b, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(tangent[0], a * tb, rtol=1e-7, atol=1e-6)


def two_arg_fn(a, b):
    return etl.multiply(a, b)


def test_jvp_wrong_tangent_structure_dict_raises():
    spec = etl.TensorSpec((3,), np.float64)
    with pytest.raises(etl.TransformError, match="structure"):
        etl.jvp(two_arg_fn, {"a": spec, "b": spec})(spec, spec)


def test_jvp_wrong_tangent_leaf_count_raises():
    spec = etl.TensorSpec((3,), np.float64)
    with pytest.raises(etl.TransformError, match="structure"):
        etl.jvp(two_arg_fn, (spec,))(spec, spec)


def test_jvp_wrong_tangent_shape_raises():
    spec = etl.TensorSpec((3,), np.float64)
    with pytest.raises(etl.TransformError, match="shape"):
        etl.jvp(
            two_arg_fn,
            (etl.TensorSpec((4,), np.float64), etl.TensorSpec((3,), np.float64)),
        )(spec, spec)


def test_jvp_wrong_tangent_dtype_raises():
    spec = etl.TensorSpec((3,), np.float64)
    with pytest.raises(etl.TransformError, match="dtype"):
        etl.jvp(
            two_arg_fn,
            (etl.TensorSpec((3,), np.float32), etl.TensorSpec((3,), np.float64)),
        )(spec, spec)


def test_jvp_single_input_wrong_tangent_shape_raises():
    spec = etl.TensorSpec(SHAPE, np.float64)
    with pytest.raises(etl.TransformError, match="shape"):
        etl.jvp(etl.square, etl.TensorSpec((5,), np.float64))(spec)


@pytest.mark.parametrize("cfg", [F64, F32], ids=["float64", "float32"])
def test_jvp_graph_input_equals_callable_form(cfg):
    """jvp(trace(f, spec), tangent) numerically equals jvp(f, tangent)(spec)."""
    dt = cfg["dtype"]
    spec = etl.TensorSpec(SHAPE, dt)
    x = np.linspace(0.2, 1.2, SHAPE[0]).astype(dt)
    t = TANGENT.astype(dt)

    def f(x):
        return etl.sum(etl.square(x))

    g_graph = etl.jvp(etl.trace(f, spec), spec)  # bare spec for the graph form
    g_call = etl.jvp(f, (spec,))(spec)  # 1-tuple spelling via the callable
    g_graph.verify()
    g_call.verify()

    out_graph = to_np(run_graph(g_graph, (x,), (t,)))
    out_call = to_np(run_graph(g_call, (x,), (t,)))
    np.testing.assert_array_equal(out_graph[0], out_call[0])
    np.testing.assert_array_equal(out_graph[1][0], out_call[1][0])

    x64, t64 = x.astype(np.float64), t.astype(np.float64)
    num = central_directional(lambda z: np.sum(z ** 2), (x64,), (t64,), cfg["eps"])
    np.testing.assert_allclose(
        out_graph[1][0], num, rtol=cfg["rtol"], atol=cfg["atol"]
    )


def test_jvp_single_input_bare_spec_and_tuple_spellings():
    """With one tensor input, a bare TensorSpec and a 1-tuple are accepted
    and produce the same tangent output."""
    spec = etl.TensorSpec(SHAPE, np.float64)
    x = np.linspace(0.2, 1.2, SHAPE[0])
    t = TANGENT

    def f(x):
        return etl.sum(etl.square(x))

    out_bare = to_np(run_graph(etl.jvp(f, spec)(spec), (x,), (t,)))
    out_tup = to_np(run_graph(etl.jvp(f, (spec,))(spec), (x,), (t,)))
    np.testing.assert_array_equal(out_bare[0], out_tup[0])
    np.testing.assert_array_equal(out_bare[1][0], out_tup[1][0])


def test_jvp_single_input_bare_none_is_zero_tangent():
    spec = etl.TensorSpec(SHAPE, np.float64)
    x = np.linspace(0.2, 1.2, SHAPE[0])

    def f(x):
        return etl.sum(etl.square(x))

    graph = etl.jvp(f, None)(spec)
    graph.verify()
    out = to_np(run_graph(graph, (x,), (None,)))
    np.testing.assert_allclose(out[0], np.sum(x ** 2), rtol=1e-7, atol=1e-6)
    assert np.all(np.asarray(out[1][0]) == 0.0)


def test_jvp_concrete_tensor_arg_raises_trace_error():
    spec = etl.TensorSpec((4,), np.float64)
    with pytest.raises(etl.TraceError, match="concrete Tensor"):
        etl.jvp(etl.square, spec)(etl.tensor(np.ones(4, np.float64)))


def test_jvp_multi_input_primal_and_tangent_outputs():
    """Two tangents in a tuple; both primal and tangent outputs are checked
    numerically against analytic and finite-difference references."""
    spec = etl.TensorSpec((5,), np.float64)
    a = np.linspace(0.2, 1.2, 5)
    b = np.linspace(1.3, 0.4, 5)
    ta = np.full(5, 0.4)
    tb = np.full(5, -0.7)

    def f(a, b):
        return etl.multiply(a, b), etl.sum(etl.add(a, b))

    _, primal, tangent = run_jvp(f, (spec, spec), (spec, spec), (a, b), (ta, tb))

    np.testing.assert_allclose(primal[0], a * b, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(primal[1], a.sum() + b.sum(), rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(tangent[0], b * ta + a * tb, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(tangent[1], ta.sum() + tb.sum(), rtol=1e-7, atol=1e-6)

    num = central_directional(
        lambda x, y: (x * y, x.sum() + y.sum()), (a, b), (ta, tb), 1e-6
    )
    np.testing.assert_allclose(tangent[0], num[0], rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(tangent[1], num[1], rtol=1e-7, atol=1e-6)


def test_jvp_runtime_call_has_no_rule():
    spec = etl.TensorSpec((4,), np.float64)

    def f(x):
        return etl.sum(etl.runtime_call(lambda v: v, x, result=spec))

    with pytest.raises(etl.TransformError, match="runtime_call"):
        etl.jvp(f, spec)(spec)


def test_jvp_nondifferentiable_outputs_give_zero_tangent():
    """Boolean/int-producing ops have builtin ZeroTangent rules: zero
    tangent output, not an error (documented semantics)."""
    spec = etl.TensorSpec((4,), np.float64)
    x = np.linspace(-1.0, 1.0, 4)
    t = np.full(4, 0.5)

    def f_equal(x):
        return etl.equal(x, etl.constant(etl.tensor(np.zeros(4, np.float64))))

    _, _, tangent = run_jvp(f_equal, spec, (spec,), (x,), (t,))
    assert np.all(np.asarray(tangent[0]) == 0)

    def f_argmax(x):
        return etl.sum(etl.cast(etl.argmax(x), etl.float64))

    _, _, tangent = run_jvp(f_argmax, spec, (spec,), (x,), (t,))
    assert np.asarray(tangent[0]) == 0.0


def test_jvp_sign_gives_zero_tangent():
    """sign has an implemented zero-derivative JVP rule (derivative 0 a.e.):
    the pointwise tangent output is all zeros, whatever the tangent."""
    spec = etl.TensorSpec((4,), np.float64)
    x = np.linspace(-1.0, 1.0, 4)
    t = np.array([0.3, -0.6, 0.9, 0.4])

    _, primal, tangent = run_jvp(etl.sign, spec, (spec,), (x,), (t,))
    np.testing.assert_allclose(primal, np.sign(x), rtol=1e-7, atol=1e-6)
    assert np.asarray(tangent[0]).shape == x.shape
    np.testing.assert_array_equal(np.asarray(tangent[0]), np.zeros(4))


def _np_conv_valid(x, w):
    """NCHW 2D convolution with VALID padding (unit strides/dilations)."""
    x = np.asarray(x)
    w = np.asarray(w)
    _, _, kh, kw = w.shape
    out = np.zeros((1, 1, x.shape[2] - kh + 1, x.shape[3] - kw + 1))
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            out[0, 0, i, j] = np.sum(x[0, 0, i : i + kh, j : j + kw] * w[0, 0])
    return out


def test_jvp_conv_has_a_rule():
    """`conv` has a JVP rule (its VJP is a documented v1 deferral)."""
    spec_x = etl.TensorSpec((1, 1, 3, 4), np.float64)
    spec_w = etl.TensorSpec((1, 1, 2, 2), np.float64)
    x = np.random.RandomState(0).rand(1, 1, 3, 4)
    w = np.random.RandomState(1).rand(1, 1, 2, 2)
    tx = np.random.RandomState(2).rand(1, 1, 3, 4)
    tw = np.random.RandomState(3).rand(1, 1, 2, 2)

    def f(x, w):
        return etl.conv(x, w, padding="VALID")

    graph, primal, tangent = run_jvp(
        f, (spec_x, spec_w), (spec_x, spec_w), (x, w), (tx, tw)
    )
    np.testing.assert_allclose(primal, _np_conv_valid(x, w), rtol=1e-7, atol=1e-6)
    assert tangent[0].shape == primal.shape == (1, 1, 2, 3)
    num = central_directional(_np_conv_valid, (x, w), (tx, tw), 1e-6)
    np.testing.assert_allclose(tangent[0], num, rtol=1e-7, atol=1e-6)


def test_jvp_static_input_gets_no_tangent():
    """A static input position carries no tangent; the bare-spec spelling
    defaults static positions to None."""
    spec = etl.TensorSpec((4,), np.float64)
    x = np.arange(4.0)
    t = np.full(4, 0.5)

    def f(x, scale):
        return etl.multiply(x, scale)

    graph = etl.jvp(f, spec)(spec, 2.5)
    graph.verify()
    out = to_np(run_graph(graph, (x, 2.5), (t,)))
    np.testing.assert_allclose(out[0], 2.5 * x, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(out[1][0], 2.5 * t, rtol=1e-7, atol=1e-6)
