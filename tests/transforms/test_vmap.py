"""Tests for `etl.vmap` — transparent function-side sugar over `vectorize`.

Contract under test (binding): `etl/transforms/CONTEXT.md` "The vmap ⇔
vectorize equivalence contract" and "out_axes semantics (v1)". `vmap` uses
exactly the same batching machinery as `vectorize`: with `out_axes=0` both
produce identical IR; given a callable/`Defn` it returns a `TransformCallable`
that strips the leading mapped dim, traces the function exactly ONCE, then
vectorizes and rearranges outputs. Transforms never execute — execution is
explicit staging only.
"""

import numpy as np
import pytest

import etl
from etl import core
from etl import ir
from etl.transforms import TransformCallable


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run_graph(graph, *args):
    """Explicit staging pipeline: lower → compile → load → run."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def to_np(out):
    """Unwrap `etl.run` structured outputs (bare Tensor or tuple) to numpy."""
    if isinstance(out, core.Tensor):
        return out.numpy()
    if isinstance(out, (tuple, list)):
        return tuple(to_np(value) for value in out)
    return out


def iter_ops(graph):
    """All ops of a graph, including ops nested inside op regions."""
    for function in graph.module.functions:
        yield from _iter_block_ops(function.region.blocks)


def _iter_block_ops(blocks):
    for block in blocks:
        for op in block.ops:
            yield op
            for region in op.regions:
                yield from _iter_block_ops(region.blocks)


def all_op_names(graph):
    return [op.name for op in iter_ops(graph)]


def serialize(graph):
    return ir.serialize_module(graph.module)


def stacked_reference(fn, *rows):
    """Run the UNVECTORIZED fn on each row via `etl.evaluate` and stack."""
    return np.stack(
        [np.asarray(to_np(etl.evaluate(fn, *row))) for row in rows]
    )


# --- functions under test ---------------------------------------------------


def _add1(x):
    return etl.add(x, 1.0)


def _add2(x, y):
    return etl.add(x, y)


def _mul2(x, y):
    return etl.multiply(x, y)


def _row_sum(x):
    return etl.sum(x, axes=(1,))


def _mixed(x, y):
    return etl.multiply(etl.add(x, y), 2.0)


def _two_outputs(x):
    return (etl.add(x, 1.0), etl.multiply(x, 2.0))


# ---------------------------------------------------------------------------
# vmap returns a TransformCallable (fn) / Graph (Graph); never executes
# ---------------------------------------------------------------------------


def test_vmap_fn_returns_transformcallable_building_graph():
    tf = etl.vmap(_add1)
    assert isinstance(tf, TransformCallable)
    assert tf.kind == "vmap"
    assert callable(tf)

    graph = tf(etl.TensorSpec((4, 3), etl.float32))
    assert isinstance(graph, etl.Graph)  # transforms never execute
    graph.verify()


def test_vmap_defn_returns_transformcallable():
    @etl.defn
    def d(x):
        return etl.add(x, 1.0)

    tf = etl.vmap(d)
    assert tf.kind == "vmap"
    graph = tf(etl.TensorSpec((4, 3), etl.float32))
    assert isinstance(graph, etl.Graph)
    graph.verify()


def test_vmap_graph_returns_graph_directly():
    graph = etl.trace(_add1, etl.TensorSpec((3,), etl.float32))
    result = etl.vmap(graph, 0)
    assert isinstance(result, etl.Graph)


def test_vmap_rejects_concrete_tensors():
    tf = etl.vmap(_add1)
    with pytest.raises(core.TraceError, match="never execute"):
        tf(etl.zeros((4, 3)))


# ---------------------------------------------------------------------------
# vmap ≡ vectorize equivalence (binding contract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, specs, batched_specs, in_axes",
    [
        (
            _add1,
            (etl.TensorSpec((3,), etl.float32),),
            (etl.TensorSpec((4, 3), etl.float32),),
            0,
        ),
        (
            _add2,
            (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
            (etl.TensorSpec((4, 3, 4), etl.float32), etl.TensorSpec((4, 3, 4), etl.float32)),
            (0, 0),
        ),
        (
            _mixed,
            (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
            (etl.TensorSpec((4, 3, 4), etl.float32), etl.TensorSpec((4, 3, 4), etl.float32)),
            (0, 0),
        ),
        (
            # one mapped + one unmapped argument
            _add2,
            (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
            (etl.TensorSpec((4, 3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
            (0, None),
        ),
    ],
)
def test_vmap_fn_equiv_vectorize_trace(fn, specs, batched_specs, in_axes):
    """vmap(f)(*batched_specs) ≡ vectorize(trace(f, *stripped_specs), in_axes).

    The stripped specs drop the leading mapped dim (dtype/name preserved);
    same machinery ⇒ identical IR, compared via module serialization.
    """
    g_vmap = etl.vmap(fn, in_axes=in_axes)(*batched_specs)
    g_vec = etl.vectorize(etl.trace(fn, *specs), in_axes)
    assert serialize(g_vmap) == serialize(g_vec)
    assert ir.pretty_print(g_vmap.module) == ir.pretty_print(g_vec.module)


@pytest.mark.parametrize(
    "fn, specs, in_axes",
    [
        (_add1, (etl.TensorSpec((3, 4), etl.float32),), 0),
        (
            _add2,
            (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
            (0, 0),
        ),
        (
            _row_sum,
            (etl.TensorSpec((3, 4), etl.float32),),
            0,
        ),
    ],
)
def test_vmap_graph_equiv_vectorize_graph(fn, specs, in_axes):
    """vmap(graph, in_axes, out_axes=0) ≡ vectorize(graph, in_axes)."""
    graph = etl.trace(fn, *specs)
    g_vmap = etl.vmap(graph, in_axes, 0)
    g_vec = etl.vectorize(etl.trace(fn, *specs), in_axes)
    assert serialize(g_vmap) == serialize(g_vec)
    assert ir.pretty_print(g_vmap.module) == ir.pretty_print(g_vec.module)
    # out_axes=0 adds no extra op at all.
    assert all_op_names(g_vmap) == all_op_names(g_vec)


# ---------------------------------------------------------------------------
# numerical: batched graph ≡ per-row runs of the unvectorized fn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, specs, batched_specs, in_axes, tol",
    [
        (_add1, (etl.TensorSpec((3,), etl.float32),), (etl.TensorSpec((5, 3), etl.float32),), 0, 1e-5),
        (_add1, (etl.TensorSpec((3,), etl.float64),), (etl.TensorSpec((5, 3), etl.float64),), 0, 1e-7),
        (_row_sum, (etl.TensorSpec((3, 4), etl.float32),), (etl.TensorSpec((5, 3, 4), etl.float32),), 0, 1e-5),
        (_mul2, (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
         (etl.TensorSpec((5, 3, 4), etl.float32), etl.TensorSpec((5, 3, 4), etl.float32)), (0, 0), 1e-5),
    ],
)
def test_vmap_numerics(fn, specs, batched_specs, in_axes, tol):
    graph = etl.vmap(fn, in_axes=in_axes)(*batched_specs)
    rng = np.random.default_rng(0)
    arrays = [
        rng.standard_normal(spec.shape).astype(spec.dtype) for spec in batched_specs
    ]
    out = np.asarray(to_np(run_graph(graph, *arrays)))
    ref = stacked_reference(fn, *zip(*arrays))
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol)


def test_vmap_default_in_axes_and_unmapped_arg_broadcast():
    # default in_axes=0 applies to the single tensor arg.
    graph = etl.vmap(_add1)(etl.TensorSpec((4, 3), etl.float32))
    x = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x)))
    ref = stacked_reference(_add1, *zip(x))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    # in_axes=None leaves an argument unbatched: it keeps its shape and is
    # broadcast against the mapped rows.
    graph = etl.vmap(_mul2, in_axes=(0, None))(
        etl.TensorSpec((4, 3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)
    )
    xb = np.random.default_rng(2).standard_normal((4, 3, 4)).astype(np.float32)
    y = np.random.default_rng(3).standard_normal((3, 4)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, xb, y)))
    ref = np.stack([to_np(etl.evaluate(_mul2, xb[i], y)) for i in range(4)])
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# once-only tracing; fresh graph per call
# ---------------------------------------------------------------------------


def test_vmap_traces_fn_exactly_once_per_call():
    count = {"n": 0}

    def counted(x):
        count["n"] += 1
        return etl.add(x, 1.0)

    tf = etl.vmap(counted)
    g_first = tf(etl.TensorSpec((4, 3), etl.float32))
    assert count["n"] == 1

    g_second = tf(etl.TensorSpec((4, 3), etl.float32))
    assert count["n"] == 2
    assert g_second is not g_first  # fresh graph per call
    assert serialize(g_first) == serialize(g_second)


# ---------------------------------------------------------------------------
# out_axes semantics
# ---------------------------------------------------------------------------


def test_out_axes_none_requires_unmapped_output():
    # `_add1`'s output is mapped → out_axes=None must fail (a batch axis is
    # never silently dropped).
    tf = etl.vmap(_add1, out_axes=None)
    with pytest.raises(core.TransformError, match="never silently dropped"):
        tf(etl.TensorSpec((4, 3), etl.float32))


def test_out_axes_none_with_unmapped_output():
    def depends_only_on_unmapped(x, y):
        return etl.add(y, 1.0)

    tf = etl.vmap(depends_only_on_unmapped, in_axes=(0, None), out_axes=None)
    graph = tf(etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((3,), etl.float32))

    x = np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)
    y = np.random.default_rng(4).standard_normal((3,)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x, y)))
    assert out.shape == (3,)  # unchanged, no batch axis
    np.testing.assert_allclose(out, y + 1.0, rtol=1e-5, atol=1e-5)


def test_out_axes_zero_with_unmapped_output_inserts_size_one_axis():
    def depends_only_on_unmapped(x, y):
        return etl.add(y, 1.0)

    graph = etl.vmap(depends_only_on_unmapped, in_axes=(0, None))(
        etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((3,), etl.float32)
    )
    # out_axes=0 on an unmapped output: an explicitly requested size-one axis
    # is inserted at 0 via an ordinary reshape op.
    assert "reshape" in all_op_names(graph)
    graph.verify()

    x = np.random.default_rng(5).standard_normal((4, 3)).astype(np.float32)
    y = np.random.default_rng(6).standard_normal((3,)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x, y)))
    assert out.shape == (1, 3)
    np.testing.assert_allclose(out[0], y + 1.0, rtol=1e-5, atol=1e-5)


def test_out_axes_zero_with_mapped_scalar_rows_keeps_batch():
    # Per-row scalar (keepdims=True → row shape (1,)) stays mapped: the
    # batch axis is kept leading, no rearrangement op is added.
    def sum_keepdims(x):
        return etl.sum(x, keepdims=True)

    graph = etl.vmap(sum_keepdims)(etl.TensorSpec((4, 3), etl.float32))
    assert "reshape" not in all_op_names(graph)

    x = np.random.default_rng(7).standard_normal((4, 3)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x)))
    assert out.shape == (4, 1)
    np.testing.assert_allclose(out[:, 0], x.sum(axis=1), rtol=1e-5, atol=1e-5)


def test_out_axes_structure_mismatch():
    graph = etl.trace(_two_outputs, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(core.TransformError, match="does not match"):
        etl.vmap(graph, 0, (0,))


def test_bare_out_axes_with_multiple_tensor_outputs():
    graph = etl.trace(_two_outputs, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(core.TransformError, match="at most ONE tensor output"):
        etl.vmap(graph, 0, 0)


def test_out_axes_non_zero_is_deferred():
    tf = etl.vmap(_add1, out_axes=1)
    with pytest.raises(core.TransformError, match="leading axis"):
        tf(etl.TensorSpec((4, 3), etl.float32))


# ---------------------------------------------------------------------------
# in_axes validation
# ---------------------------------------------------------------------------


def test_in_axes_non_zero_is_deferred():
    tf = etl.vmap(_add1, in_axes=1)
    with pytest.raises(core.TransformError, match="leading axis"):
        tf(etl.TensorSpec((4, 3), etl.float32))


def test_bare_in_axes_with_multiple_tensor_specs():
    tf = etl.vmap(_add2)  # default in_axes=0 is bare → ambiguous with 2 args
    with pytest.raises(core.TransformError, match="exactly ONE tensor spec"):
        tf(etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((4, 3), etl.float32))


def test_in_axes_structure_mismatch():
    tf = etl.vmap(_add2, in_axes=(0,))
    with pytest.raises(core.TransformError, match="does not match"):
        tf(etl.TensorSpec((4, 3), etl.float32), etl.TensorSpec((4, 3), etl.float32))


# ---------------------------------------------------------------------------
# control-flow ops are not vectorizable in v1
# ---------------------------------------------------------------------------


def test_vmap_of_cond_raises_transformerror():
    def cond_fn(x):
        pred = etl.sum(x) > 0.0
        return etl.cond(pred, lambda: x, lambda: -x)

    tf = etl.vmap(cond_fn)
    with pytest.raises(
        core.TransformError,
        match=r"cannot batch op 'if'.*region-bearing control-flow",
    ):
        tf(etl.TensorSpec((4, 3), etl.float32))


# ---------------------------------------------------------------------------
# external-kernel rule channel: batching (external:<name> keys)
# ---------------------------------------------------------------------------
#
# `external_call` ops resolve batching rules under the namespaced
# `external:<name>` key (mirroring `block:<name>` for block calls), registered
# through the handle returned by `etl.register_external_kernel`. Without an
# explicit rule, a registered portable decomposition (`@handle.portable`) is
# the fallback: vmap/vectorize re-trace it over the batched operands. Names
# are `tx_`-prefixed to stay unique in the process-wide registries.


def _sym(value):
    """Wrap an ir.Value as a SymbolicTensor so etl.ops can build on it."""
    return etl.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


EXT_DIM = 4
EXT_BATCH = 8


def _ext_double(x):
    """The registered kernel: y = 2x (elementwise)."""
    return 2 * x


ext_double = etl.register_external_kernel("tx_ext_double", _ext_double)


@ext_double.batching_rule
def _ext_double_batch(op, operands, axes):
    """Explicit batching rule: batched rows of 2x."""
    y = etl.multiply(_sym(operands[0]), 2)
    return (y.value,), (axes[0],)


@etl.defn
def _ext_double_fn(x):
    return etl.external_call(
        "tx_ext_double", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


def _batched_ext_rows():
    return np.linspace(-1, 1, EXT_BATCH * EXT_DIM, dtype=np.float32).reshape(
        EXT_BATCH, EXT_DIM
    )


def test_vmap_external_call_explicit_batching_rule():
    """vmap over a graph containing external_call with an explicit rule
    (external:<name> key) == 2x on every row; the rule replaces the op."""
    graph = etl.vmap(_ext_double_fn)(etl.TensorSpec((EXT_BATCH, EXT_DIM), etl.float32))
    graph.verify()
    x = _batched_ext_rows()
    out = np.asarray(to_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, 2 * x, rtol=1e-6)
    assert "external_call" not in all_op_names(graph)


def test_vectorize_external_call_graph_form_explicit_rule():
    """vectorize on the traced graph (not the callable) uses the same
    external:<name> rule channel."""
    graph = etl.trace(_ext_double_fn, etl.TensorSpec((EXT_DIM,), etl.float32))
    vec = etl.vectorize(graph, 0)
    x = _batched_ext_rows()
    out = np.asarray(to_np(run_graph(vec, x)))
    np.testing.assert_allclose(out, 2 * x, rtol=1e-6)


def _ext_swish(x):
    """The registered kernel: the swish activation."""
    s = 1.0 / (1.0 + np.exp(-x))
    return s * x


ext_swish = etl.register_external_kernel("tx_ext_swish", _ext_swish)


@ext_swish.portable
@etl.defn
def _ext_swish_portable(
    x: etl.TensorSpec((EXT_DIM,), etl.float32),
) -> etl.TensorSpec((EXT_DIM,), etl.float32):
    return etl.sigmoid(x) * x


@etl.defn
def _ext_swish_fn(x):
    return etl.external_call(
        "tx_ext_swish", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


def test_vmap_external_call_portable_fallback():
    """No explicit rule: vmap falls back to inlining the registered portable
    decomposition over the batched operands — equals the per-row kernel."""
    graph = etl.vmap(_ext_swish_fn)(etl.TensorSpec((EXT_BATCH, EXT_DIM), etl.float32))
    graph.verify()
    x = _batched_ext_rows()
    out = np.asarray(to_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, _ext_swish(x), rtol=1e-6)
    assert "external_call" not in all_op_names(graph)


def _ext_win(x):
    """The registered kernel (identity; only the rule/portable semantics are
    under test here)."""
    return x


ext_win = etl.register_external_kernel("tx_ext_win", _ext_win)


@ext_win.portable
@etl.defn
def _ext_win_portable(
    x: etl.TensorSpec((EXT_DIM,), etl.float32),
) -> etl.TensorSpec((EXT_DIM,), etl.float32):
    return etl.multiply(x, 2)


@ext_win.batching_rule
def _ext_win_batch(op, operands, axes):
    """Registered AFTER the portable: must overwrite the fallback (3x)."""
    y = etl.multiply(_sym(operands[0]), 3)
    return (y.value,), (axes[0],)


@etl.defn
def _ext_win_fn(x):
    return etl.external_call(
        "tx_ext_win", x, result=etl.TensorSpec((EXT_DIM,), etl.float32)
    )


def test_external_explicit_batching_rule_wins_over_portable():
    """A rule registered AFTER the portable decomposition overwrites the
    fallback: the registry lookup returns the explicit rule and vmap uses it
    (3x) instead of the portable's 2x."""
    from etl import transforms

    assert transforms.batching_rules["external:tx_ext_win"] is _ext_win_batch
    # The portable itself computes 2x (the fallback semantics it replaced).
    x4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    port_out = run_graph(
        etl.trace(_ext_win_portable, etl.TensorSpec((EXT_DIM,), etl.float32)), x4
    )
    np.testing.assert_allclose(to_np(port_out), 2 * x4, rtol=1e-6)

    x = _batched_ext_rows()
    graph = etl.vmap(_ext_win_fn)(etl.TensorSpec((EXT_BATCH, EXT_DIM), etl.float32))
    out = np.asarray(to_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, 3 * x, rtol=1e-6)
