"""Tests for `etl.vectorize` — the batching primitive (graph → graph).

Contract under test (binding): `etl/transforms/CONTEXT.md` "The vectorize core"
and the vmap⇔vectorize equivalence contract. `vectorize` rewrites a traced
`Graph` into a NEW graph of ordinary ops whose mapped inputs/outputs carry an
explicit leading batch dim; it never executes and never mutates the input
graph. v1 axes entries are {None, 0} only; unsupported ops raise
`TransformError` naming the op.

Execution is explicit staging only (transforms never import/use
backends/pipeline): `etl.run(etl.load(etl.compile(etl.lower(graph))), *args)`.
"""

import numpy as np
import pytest

import etl
import etl.numpy as enp
from etl import core
from etl import ir


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


def _add(x, y):
    return etl.add(x, y)


def _mul(x, y):
    return etl.multiply(x, y)


def _dot(x, w):
    return etl.dot(x, w)


def _row_sum(x):
    return etl.sum(x, axes=(1,))


def _add1(x):
    return etl.add(x, 1.0)


# ---------------------------------------------------------------------------
# numerical: batched graph ≡ per-row runs of the unvectorized fn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, shapes, dtype, tol",
    [
        (_add, ((4, 3), (4, 3)), np.float32, 1e-5),
        (_add, ((4, 3), (4, 3)), np.float64, 1e-7),
        (_mul, ((4, 3), (4, 3)), np.float32, 1e-5),
        (_mul, ((4, 3), (4, 3)), np.float64, 1e-7),
        (_dot, ((4, 3), (3, 2)), np.float32, 1e-5),
        (_dot, ((4, 3), (3, 2)), np.float64, 1e-7),
        (_row_sum, ((4, 3),), np.float32, 1e-5),
        (_row_sum, ((4, 3),), np.float64, 1e-7),
    ],
)
def test_batched_graph_matches_per_row(fn, shapes, dtype, tol):
    batch = 5
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal((batch,) + shape).astype(dtype) for shape in shapes]
    specs = tuple(etl.TensorSpec(shape, dtype) for shape in shapes)
    axes = 0 if len(specs) == 1 else tuple(0 for _ in specs)

    graph = etl.vectorize(etl.trace(fn, *specs), axes)
    out = np.asarray(to_np(run_graph(graph, *arrays)))
    ref = stacked_reference(fn, *zip(*arrays))
    np.testing.assert_allclose(out, ref, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# axes forms: dict and per-argument tuple; unmapped entries broadcast
# ---------------------------------------------------------------------------


def _dict_add(args):
    return etl.add(args["a"], args["b"])


def test_dict_in_axes_with_unmapped_broadcast():
    # The traced input tree wraps each argument: root tuple → one dict child,
    # so the axes pytree must mirror that structure.
    spec = {"a": etl.TensorSpec((3, 4), etl.float32), "b": etl.TensorSpec((3, 4), etl.float32)}
    graph = etl.vectorize(etl.trace(_dict_add, spec), ({"a": 0, "b": None},))

    rng = np.random.default_rng(1)
    xa = rng.standard_normal((5, 3, 4)).astype(np.float32)
    xb = rng.standard_normal((3, 4)).astype(np.float32)  # unmapped: no batch dim
    out = np.asarray(to_np(run_graph(graph, {"a": xa, "b": xb})))
    assert out.shape == (5, 3, 4)

    ref = np.stack(
        [to_np(etl.evaluate(_dict_add, {"a": xa[i], "b": xb})) for i in range(5)]
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_tuple_in_axes_with_unmapped_arg():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    graph = etl.vectorize(etl.trace(_add, *spec), (0, None))

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 3, 4)).astype(np.float32)
    y = rng.standard_normal((3, 4)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x, y)))
    ref = np.stack([to_np(etl.evaluate(_add, x[i], y)) for i in range(5)])
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# result is an ordinary graph; input graph is not mutated
# ---------------------------------------------------------------------------


def test_result_is_ordinary_graph_without_vectorize_op():
    graph = etl.vectorize(
        etl.trace(_add, etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
        (0, 0),
    )
    assert isinstance(graph, etl.Graph)
    names = all_op_names(graph)
    assert "vectorize" not in names
    assert names[-1] == "return"
    graph.verify()  # structural/type/attribute validation passes


def test_input_graph_is_not_mutated():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    original = etl.trace(_add, *spec)
    before = serialize(original)

    batched = etl.vectorize(original, (0, 0))

    assert serialize(original) == before
    assert batched.module is not original.module
    assert batched.verify() is None


# ---------------------------------------------------------------------------
# batch dims and metadata propagation
# ---------------------------------------------------------------------------


def test_mapped_input_and_output_gain_leading_batch_dim():
    graph = etl.vectorize(etl.trace(_add1, etl.TensorSpec((3, 4), etl.float32)), 0)

    spec = graph.tensor_specs[0]
    assert len(spec.shape) == 3
    assert isinstance(spec.shape[0], core.Dim)
    assert spec.shape[0].name == "batch"
    assert tuple(spec.shape[1:]) == (3, 4)

    # The mapped output leads with the SAME batch dim (object identity).
    output = graph.module.main.entry_block.terminator.operands[0]
    assert output.type.shape[0] is spec.shape[0]
    assert tuple(output.type.shape[1:]) == (3, 4)


def test_two_mapped_inputs_output_leads_with_batch_dim_expression():
    # Two mapped inputs of the same row shape broadcast their batch dims into
    # a symbolic leading entry (max(batch, batch_1)) — still the mapped axis.
    graph = etl.vectorize(
        etl.trace(_add, etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32)),
        (0, 0),
    )
    assert graph.tensor_specs[0].shape[0].name == "batch"
    assert graph.tensor_specs[1].shape[0].name == "batch_1"

    output = graph.module.main.entry_block.terminator.operands[0]
    leading = output.type.shape[0]
    assert isinstance(leading, (core.Dim, core.DimExpr))


def test_unmapped_axes_none_keeps_input_unmapped():
    graph = etl.vectorize(etl.trace(_add1, etl.TensorSpec((3, 4), etl.float32)), None)
    assert tuple(graph.tensor_specs[0].shape) == (3, 4)  # no batch dim added

    x = np.random.default_rng(3).standard_normal((3, 4)).astype(np.float32)
    out = np.asarray(to_np(run_graph(graph, x)))
    np.testing.assert_allclose(out, np.asarray(to_np(etl.evaluate(_add1, x))), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# static values
# ---------------------------------------------------------------------------


def _scaled(x, factor):
    return etl.multiply(x, factor)


def test_static_leaf_requires_none_axes():
    graph = etl.trace(_scaled, etl.TensorSpec((3,), etl.float32), 2)
    with pytest.raises(core.TransformError, match="static leaf"):
        etl.vectorize(graph, (0, 0))


def test_static_values_and_output_tree_preserved():
    def two_outputs(x, factor):
        return (etl.multiply(x, factor), factor + 1)

    original = etl.trace(two_outputs, etl.TensorSpec((3,), etl.float32), 2)
    batched = etl.vectorize(original, (0, None))

    assert batched.static_values == original.static_values
    assert batched.output_static_values == original.output_static_values
    assert batched.output_tree == original.output_tree
    batched.verify()

    x = np.random.default_rng(4).standard_normal((5, 3)).astype(np.float32)
    out = to_np(run_graph(batched, x, 2))
    assert out[1] == 3  # static output leaf preserved
    np.testing.assert_allclose(out[0], x * 2.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


def test_vectorize_rejects_bare_callable():
    with pytest.raises(TypeError, match=r"etl\.vmap"):
        etl.vectorize(_add1, 0)


def test_non_leading_axis_is_deferred():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(core.TransformError, match="leading axis"):
        etl.vectorize(etl.trace(_add, *spec), (1, 0))


def test_axis_out_of_range():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(core.TransformError, match="out of range"):
        etl.vectorize(etl.trace(_add, *spec), (2, 0))


def test_negative_axis_is_out_of_range():
    graph = etl.trace(_add1, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(core.TransformError, match="out of range"):
        etl.vectorize(graph, -1)


def test_axes_structure_mismatch():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(core.TransformError, match="does not match"):
        etl.vectorize(etl.trace(_add, *spec), (0,))
    with pytest.raises(core.TransformError, match="does not match"):
        etl.vectorize(etl.trace(_add, *spec), {"a": 0, "b": 0})


def test_bare_axes_require_exactly_one_tensor_input():
    spec = (etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((3, 4), etl.float32))
    with pytest.raises(core.TransformError, match="exactly ONE tensor input"):
        etl.vectorize(etl.trace(_add, *spec), 0)


# ---------------------------------------------------------------------------
# control-flow ops are not vectorizable in v1 (never a silent fallback)
# ---------------------------------------------------------------------------


def _cond_fn(x):
    pred = etl.sum(x) > 0.0
    return etl.cond(pred, lambda: x, lambda: -x)


def _while_fn(x):
    def cond_fn(acc):
        return etl.sum(acc) < 1000.0

    def body_fn(acc):
        return acc + x

    return etl.while_loop(cond_fn, body_fn, enp.zeros((3,)))


def _scan_fn(x):
    def body(carry, elem):
        return carry + elem, carry

    return etl.scan(body, enp.zeros(()), x)


def test_cond_is_not_vectorizable():
    graph = etl.trace(_cond_fn, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(
        core.TransformError,
        match=r"cannot batch op 'if'.*region-bearing control-flow",
    ):
        etl.vectorize(graph, 0)


def test_while_loop_is_not_vectorizable():
    graph = etl.trace(_while_fn, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(core.TransformError, match=r"cannot batch op 'while'"):
        etl.vectorize(graph, 0)


def test_scan_is_not_vectorizable():
    # scan lowers to a step-0 `gather` in the entry block, whose documented v1
    # deferral (mapped data + unmapped indices) fires before the `while` op is
    # reached — either way a TransformError names the first unvectorizable op.
    graph = etl.trace(_scan_fn, etl.TensorSpec((3,), etl.float32))
    with pytest.raises(core.TransformError, match="cannot batch"):
        etl.vectorize(graph, 0)
