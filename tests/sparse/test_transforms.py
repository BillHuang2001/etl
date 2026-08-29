"""Batching and differentiation tests for ``etl.sparse``: vmap/vectorize
numerics vs pure-numpy dense references, pytree axes contracts (pair-lead
guard, registered-node container entries), batch-Dim static-leaf behavior,
grad/vjp/jvp numerics vs analytic references, the three vjp deferrals, and the
documented vmap∘grad gather gap.

Contract under test: ``../etl/sparse/CONTEXT.md`` ("Differentiation &
batching") + ``../etl/transforms/CONTEXT.md`` (AD semantics, registered-node
axes normalization). The supported sparse path is GRAPH-level vectorization:
trace with a ``SparseTensorSpec``, then ``etl.vmap(graph, in_axes=...)`` /
``etl.vectorize(graph, axes=...)`` (the callable path strips specs per leaf,
which produces invalid sparse leaf shapes in v1 — documented, not tested
here).

Conventions pinned in this file (verified against the current implementation):

* ``etl.vmap(graph, in_axes=0)`` on a single-sparse-input graph returns a
  Graph; the batched run takes a batched sparse (leaves with a leading batch
  dim, ``dense_shape`` UNCHANGED — ``tests.sparse.conftest.batched_coo_example``)
  and yields a concrete sparse with ``dense_shape == (Dim('batch'), 3, 4)``.
* ``etl.vectorize(graph, axes=0)`` and ``axes=(0,)`` both work; ``axes=[0]``
  (a LIST) fails with a generic axes-structure mismatch (the graph's input
  tree is a 1-tuple, and list != tuple is a generic pytree structure rule —
  not sparse-specific).
* vjp cotangents for sparse OUTPUTS are the flat per-leaf cotangents: the
  cotangent pytree mirrors the output SparseTensor node (a
  ``sparse.SparseTensorSpec`` with one ``core.TensorSpec`` per tensor leaf);
  static leaves (dense_shape/dtype/format) must mirror the recorded statics.
  The vjp result graph runs with ``(primal_tree, (indices_ct, values_ct))``.
* jvp tangent pytrees are the same flat per-leaf structure; the result graph
  runs with ``(primal_tree, (indices_t, values_t[, ...]))``.
"""

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse
from etl.transforms.autodiff import ZeroTangent, jvp_rules

from tests.sparse.conftest import (
    batched_coo_example,
    coo_example,
    coo_spec,
    csr_example,
    csr_spec,
    dense_example,
    materialize,
    run_graph,
)

# ---------------------------------------------------------------------------
# Per-file helpers
# ---------------------------------------------------------------------------


def batched_csr_example():
    """A 2-element batched CSR input (leaves (B, nnz)/(B, 4)/(B, nnz) with
    dense_shape UNCHANGED (3, 4)) — same data as `batched_coo_example`."""
    indptr = np.array([0, 1, 2, 4], dtype=np.int64)
    cols = np.array([1, 2, 0, 3], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return sparse.SparseTensor.from_parts(
        etl.core.from_numpy(np.stack([indptr, indptr], axis=0)),
        etl.core.from_numpy(np.stack([cols, cols], axis=0)),
        etl.core.from_numpy(np.stack([vals, vals * 2.0], axis=0)),
        dense_shape=(3, 4),
        format="csr",
    )


def _op_names(graph):
    """The IR op names of a traced/transformed graph's entry block."""
    return [
        op.name
        for op in graph.module.functions[0].entry_block.ops
        if not op.is_terminator
    ]


def _output_batch_dims(graph):
    """The batch Dims carried by the graph's output-side static records."""
    return [
        record.value
        for record in graph.output_static_values
        if isinstance(record.value, core.Dim)
    ]


def _dense_weights():
    """A dense (3, 4) weight matrix used by the weighted-loss grad tests."""
    w = dense_example().copy()
    w[0, 0] = 5.0
    w[1, 1] = 7.0
    w[2, 2] = 9.0
    w[2, 3] = 11.0
    return w


# ---------------------------------------------------------------------------
# 1. vmap numerics (bare int axes)
# ---------------------------------------------------------------------------


def test_vmap_dense_output_single_sparse_input():
    """vmap of a dense-output sparse fn: batch of to_dense(negate(x))."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    def f(x):
        return sparse.to_dense(sparse.negate(x))

    graph = etl.trace(f, coo_spec())
    vgraph = etl.vmap(graph, in_axes=0)
    assert isinstance(vgraph, etl.Graph)

    out = run_graph(vgraph, batched_coo_example())
    got = materialize(out)
    assert got.shape == (2, 3, 4)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_vmap_sparse_output_matches_negated_batch():
    """vmap of a sparse-output fn: the run result is concrete sparse with
    dense_shape (Dim('batch'), 3, 4) and to_dense == the negated batch."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    vgraph = etl.vmap(graph, in_axes=0)

    out = run_graph(vgraph, batched_coo_example())
    assert sparse.is_sparse(out)
    assert isinstance(out.dense_shape[0], core.Dim)
    assert out.dense_shape[0].name == "batch"
    assert out.dense_shape[1:] == (3, 4)
    np.testing.assert_allclose(out.to_dense(), ref, rtol=1e-5, atol=1e-6)


def test_vmap_csr_input_auto_conversion():
    """CSR symbolic input under vmap: the computation-format conversion
    (csr -> coo) batches transparently; the run takes a batched CSR."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    graph = etl.trace(lambda x: sparse.negate(x), csr_spec())
    vgraph = etl.vmap(graph, in_axes=0)

    out = run_graph(vgraph, batched_csr_example())
    assert sparse.is_sparse(out)
    assert out.dense_shape[1:] == (3, 4)
    np.testing.assert_allclose(out.to_dense(), ref, rtol=1e-5, atol=1e-6)


def test_vmap_two_sparse_inputs_batched_add():
    """in_axes=(0, 0) on a two-sparse-input graph: add of two batched
    operands vs the stacked dense reference."""
    d = dense_example()
    ref = np.stack([d + d, 2.0 * d + 2.0 * d])

    graph = etl.trace(lambda a, b: sparse.add(a, b), coo_spec(), coo_spec())
    vgraph = etl.vmap(graph, in_axes=(0, 0))

    out = run_graph(vgraph, batched_coo_example(), batched_coo_example())
    np.testing.assert_allclose(out.to_dense(), ref, rtol=1e-5, atol=1e-6)


def test_vmap_bare_in_axes_zero_single_sparse_tree():
    """A bare in_axes=0 works for a single-sparse-input tree (the bare entry
    broadcasts across the node's tensor leaves; statics map to None)."""
    d = dense_example()

    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    out = run_graph(etl.vmap(graph, in_axes=0), batched_coo_example())
    np.testing.assert_allclose(
        out.numpy(), np.stack([-d, -2.0 * d]), rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# 2. vectorize equivalence
# ---------------------------------------------------------------------------


def test_vectorize_equals_vmap_for_sparse_output():
    """vectorize(graph, 0) and vmap(graph, in_axes=0, out_axes=0) are the
    same machinery: identical IR and identical run results (both outputs get
    dense_shape (Dim, 3, 4))."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    vec = etl.vectorize(graph, axes=0)
    vmap_out0 = etl.vmap(graph, in_axes=0, out_axes=0)

    assert _op_names(vec) == _op_names(vmap_out0)

    got_vec = run_graph(vec, batched_coo_example())
    got_vmap = run_graph(vmap_out0, batched_coo_example())
    assert got_vec.dense_shape == got_vmap.dense_shape
    np.testing.assert_allclose(got_vec.to_dense(), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_vmap.to_dense(), ref, rtol=1e-5, atol=1e-6)


def test_vectorize_tuple_axes_spelling():
    """Both bare `axes=0` and 1-tuple `axes=(0,)` spellings vectorize a
    single-sparse-input graph (the graph's input tree is a 1-tuple wrapping
    the sparse node, so `axes=[0]` — a LIST — is a generic pytree structure
    mismatch, not sparse-specific)."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    got_bare = run_graph(etl.vectorize(graph, axes=0), batched_coo_example())
    got_tuple = run_graph(etl.vectorize(graph, axes=(0,)), batched_coo_example())
    np.testing.assert_allclose(got_bare.to_dense(), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_tuple.to_dense(), ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. pytree axes: pair-lead guard and registered-node container entries
# ---------------------------------------------------------------------------


def test_container_axes_entry_at_sparse_node_raises():
    """A container axes entry at a registered-node position is unsupported in
    v1: one int/None must map the node's tensor leaves."""
    graph = etl.trace(lambda a, b: sparse.add(a, b), coo_spec(), coo_spec())
    with pytest.raises(
        core.TransformError, match="must be a single int/None"
    ) as exc:
        etl.vmap(graph, in_axes=([0], [0]))
    assert "SparseTensor" in str(exc.value)
    assert "got a container" in str(exc.value)


def test_mixed_in_axes_unmapped_sparse_operand_raises():
    """in_axes=(0, None) on two sparse inputs: an unmapped sparse operand
    cannot broadcast across the batch (the documented pair-lead guard)."""
    graph = etl.trace(lambda a, b: sparse.add(a, b), coo_spec(), coo_spec())
    with pytest.raises(
        core.TransformError, match="cannot batch op 'sparse_add'"
    ) as exc:
        etl.vmap(graph, in_axes=(0, None))
    assert "sparse operand 2 carries 0 mapped axes but the op maps 1" in str(
        exc.value
    )


# ---------------------------------------------------------------------------
# 4. batch-Dim static-leaf checks
# ---------------------------------------------------------------------------


def test_batch_dim_appears_in_output_statics_only():
    """After vmap of a sparse-output fn, the OUTPUT-side static values carry
    the fresh batch Dim at dense_shape position 0 of the sparse node;
    INPUT-side statics stay the plain ints (3, 4) — no Dim. The run result's
    dense_shape[0] is the SAME Dim object."""
    d = dense_example()
    ref = np.stack([-d, -2.0 * d])

    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    vgraph = etl.vmap(graph, in_axes=0)

    # Output side: the sparse node's dense_shape[0] record (flat index 2 of
    # the output tree — the tuple wrapper is child 0, dense_shape[0] is the
    # third leaf) holds the fresh batch Dim named "batch".
    batch_dims = _output_batch_dims(vgraph)
    assert len(batch_dims) == 1
    assert isinstance(batch_dims[0], core.Dim)
    assert batch_dims[0].name == "batch"

    # Input side: the dense_shape leaves remain plain ints, no Dim anywhere.
    input_values = [record.value for record in vgraph.static_values]
    assert 3 in input_values and 4 in input_values
    assert not any(isinstance(v, core.Dim) for v in input_values)

    out = run_graph(vgraph, batched_coo_example())
    assert out.dense_shape[0] is batch_dims[0]
    assert out.dense_shape[1:] == (3, 4)
    np.testing.assert_allclose(out.to_dense(), ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. out_axes
# ---------------------------------------------------------------------------


def test_out_axes_none_on_mapped_sparse_output_raises():
    """out_axes=None on a mapped sparse output: the batch axis is never
    silently dropped."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with pytest.raises(
        core.TransformError, match="out_axes=None"
    ) as exc:
        etl.vmap(graph, in_axes=0, out_axes=None)
    assert "mapped batch axis" in str(exc.value)


def test_out_axes_zero_default_leaves_graph_unchanged():
    """out_axes=0 (the default) is a no-op for leading mapped outputs: the
    vmap graph is identical to the plain vectorized graph."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with_default = etl.vmap(graph, in_axes=0)
    with_explicit = etl.vmap(graph, in_axes=0, out_axes=0)
    assert _op_names(with_default) == _op_names(with_explicit)
    np.testing.assert_allclose(
        run_graph(with_default, batched_coo_example()).to_dense(),
        run_graph(with_explicit, batched_coo_example()).to_dense(),
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 6. grad numerics
# ---------------------------------------------------------------------------


def test_grad_negate_sum_values_are_minus_one():
    """loss(x) = sum(to_dense(negate(x))): the values gradient is -1 at every
    stored position (indices are never differentiated)."""

    def loss(x):
        return etl.sum(sparse.to_dense(sparse.negate(x)))

    graph = etl.grad(loss, argnums=(1,))(coo_spec())
    (got,) = run_graph(graph, coo_example())
    np.testing.assert_allclose(got.numpy(), -1.0, rtol=1e-6, atol=1e-6)


def test_grad_weighted_loss_matches_dense_weights():
    """loss(x, w) = sum(to_dense(x) * w): the values gradient at each stored
    position equals the dense weight at that position."""

    def loss(x, w):
        return etl.sum(sparse.to_dense(x) * w)

    w = _dense_weights()
    graph = etl.grad(loss, argnums=(1,))(coo_spec(), etl.TensorSpec((3, 4), etl.float32))
    (got,) = run_graph(graph, coo_example(), w)

    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])
    expected = w[idx[:, 0], idx[:, 1]]
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_grad_indices_argnum_raises():
    """argnums=(0,) selects the int64 indices leaf, which is not
    differentiable — an explicit TransformError (never a silent zero)."""

    def loss(x):
        return etl.sum(sparse.to_dense(sparse.negate(x)))

    with pytest.raises(
        core.TransformError, match="cannot be differentiated"
    ) as exc:
        etl.grad(loss, argnums=(0,))(coo_spec())
    assert "input 0 (dtype int64)" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. vjp numerics
# ---------------------------------------------------------------------------


def test_vjp_square_multiply_values_cotangent():
    """f(x) = to_dense(multiply(x, x)) with a dense cotangent C: the values
    cotangent is 2*v*C[row] at each stored row; the indices cotangent
    materializes as zeros."""
    C = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )
    graph = etl.trace(lambda x: sparse.to_dense(sparse.multiply(x, x)), coo_spec())
    vjp_graph = etl.vjp(graph, cotangents=etl.TensorSpec((3, 4), etl.float32))

    primal, cts = run_graph(vjp_graph, (coo_example(),), (C,))
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])
    v = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    expected = 2.0 * v * C[idx[:, 0], idx[:, 1]]
    np.testing.assert_allclose(cts[1].numpy(), expected, rtol=1e-5, atol=1e-6)
    # indices are never differentiated: zero input cotangent.
    assert np.all(cts[0].numpy() == 0)
    np.testing.assert_allclose(
        primal.numpy(), coo_example().to_dense() ** 2, rtol=1e-5, atol=1e-6
    )


def test_vjp_dot_dense_accumulates_shared_rows():
    """sparse@dense with TWO nonzeros sharing row 1: the dense cotangent must
    ACCUMULATE both contributions (one-hot selection + matmul — scatter
    overwrite semantics would silently drop one)."""
    idx = np.array([[0, 1], [1, 2], [1, 3], [2, 0]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = sparse.coo(idx, vals, (3, 4))
    dense = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )  # (4, 3)
    C = np.arange(9, dtype=np.float32).reshape(3, 3) / 10.0

    def f(s_, d_):
        return sparse.matmul(s_, d_)

    graph = etl.trace(f, coo_spec(), etl.TensorSpec((4, 3), etl.float32))
    vjp_graph = etl.vjp(graph, cotangents=etl.TensorSpec((3, 3), etl.float32))

    primal, cts = run_graph(vjp_graph, (s, dense), (C,))
    np.testing.assert_allclose(
        primal.numpy(), s.to_dense() @ dense, rtol=1e-5, atol=1e-6
    )
    # g_values[i] = sum_n C[m_i, n] * dense[k_i, n]
    expected_values = np.array(
        [np.sum(C[idx[i, 0], :] * dense[idx[i, 1], :]) for i in range(4)]
    )
    np.testing.assert_allclose(cts[1].numpy(), expected_values, rtol=1e-5, atol=1e-6)
    # g_dense[k, :] accumulates over every nonzero with row k (row 1 twice).
    expected_dense = np.zeros((4, 3), np.float32)
    for i in range(4):
        expected_dense[idx[i, 1]] += vals[i] * C[idx[i, 0]]
    np.testing.assert_allclose(cts[2].numpy(), expected_dense, rtol=1e-5, atol=1e-6)
    assert np.all(cts[0].numpy() == 0)


def test_vjp_dense_dot_sparse_accumulates_shared_cols():
    """dense@sparse with two nonzeros sharing a column: the dense cotangent
    accumulates both contributions (V @ A^T)."""
    idx = np.array([[0, 1], [1, 2], [1, 3], [2, 0]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = sparse.coo(idx, vals, (3, 4))
    dense = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0  # (2, 3)
    C = np.arange(8, dtype=np.float32).reshape(2, 4) / 10.0  # (2, 4)

    def f(d_, s_):
        return sparse.matmul(d_, s_)

    graph = etl.trace(f, etl.TensorSpec((2, 3), etl.float32), coo_spec())
    vjp_graph = etl.vjp(graph, cotangents=etl.TensorSpec((2, 4), etl.float32))

    primal, cts = run_graph(vjp_graph, (dense, s), (C,))
    np.testing.assert_allclose(
        primal.numpy(), dense @ s.to_dense(), rtol=1e-5, atol=1e-6
    )
    # g_values[i] = sum_m C[m, n_i] * dense[m, k_i]
    expected_values = np.array(
        [np.sum(C[:, idx[i, 1]] * dense[:, idx[i, 0]]) for i in range(4)]
    )
    np.testing.assert_allclose(cts[2].numpy(), expected_values, rtol=1e-5, atol=1e-6)
    # g_dense[m, k] accumulates over every nonzero with k_i == k.
    expected_dense = np.zeros((2, 3), np.float32)
    for i in range(4):
        expected_dense[:, idx[i, 0]] += C[:, idx[i, 1]] * vals[i]
    np.testing.assert_allclose(cts[0].numpy(), expected_dense, rtol=1e-5, atol=1e-6)


def test_vjp_sparse_output_flat_leaf_cotangents():
    """A sparse OUTPUT's vjp cotangents are the flat per-leaf cotangents: the
    cotangent pytree mirrors the output SparseTensor node (indices and values
    TensorSpecs; the static dense_shape/dtype/format leaves mirror the traced
    statics). The result graph runs with (primal_tree, (indices_ct,
    values_ct)); the values cotangent pulls back through negate and the
    input indices cotangent is zero."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    cotangents = sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(3, 4),
    )
    vjp_graph = etl.vjp(graph, cotangents=cotangents)

    values_ct = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    primal, cts = run_graph(
        vjp_graph,
        (coo_example(),),
        (np.zeros((4, 2), np.int64), values_ct),
    )
    np.testing.assert_allclose(
        primal.to_dense(), -coo_example().to_dense(), rtol=1e-5, atol=1e-6
    )
    # negate: d(-v)/dv = -1 -> values cotangent = -values_ct; indices: zero.
    np.testing.assert_allclose(cts[1].numpy(), -values_ct, rtol=1e-5, atol=1e-6)
    assert np.all(cts[0].numpy() == 0)


def test_vjp_callable_path_sparse_output_equals_graph_path():
    """The callable path `vjp(f, cotangents=<spec>)(spec)` traces once and
    yields the same sparse-output vjp graph as the graph path."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    cotangents = sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(3, 4),
    )
    from_graph = etl.vjp(graph, cotangents=cotangents)
    from_call = etl.vjp(
        lambda x: sparse.negate(x), cotangents=cotangents
    )(coo_spec())
    assert _op_names(from_graph) == _op_names(from_call)

    values_ct = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    _, cts = run_graph(
        from_call, (coo_example(),), (np.zeros((4, 2), np.int64), values_ct)
    )
    np.testing.assert_allclose(cts[1].numpy(), -values_ct, rtol=1e-5, atol=1e-6)


def test_vjp_sparse_output_cotangents_none_requires_scalar():
    """cotangents=None (scalar-one default) is invalid for a sparse output
    (two flattened tensor outputs) — an explicit ShapeError."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with pytest.raises(core.ShapeError, match="requires exactly one tensor output"):
        etl.vjp(graph)


def test_vjp_none_cotangent_at_non_scalar_output_raises():
    """A None cotangent at a non-scalar output seeds a scalar-one, which is
    rejected explicitly."""
    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    with pytest.raises(core.ShapeError, match="requires a scalar output"):
        etl.vjp(graph, cotangents=(None,))


# ---------------------------------------------------------------------------
# 8. vjp deferrals (explicit TransformError, never silent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,cotangents,op_name",
    [
        pytest.param(
            lambda a, b: sparse.to_dense(sparse.concatenate([a, b], axis=0)),
            etl.TensorSpec((6, 4), etl.float32),
            "sparse_concatenate",
            id="sparse_concatenate",
        ),
        pytest.param(
            lambda x: sparse.to_csc(x),
            sparse.SparseTensorSpec(
                etl.TensorSpec((5,), etl.int64),
                etl.TensorSpec((None,), etl.int64),
                etl.TensorSpec((None,), etl.float32),
                dense_shape=(3, 4),
                format="csc",
            ),
            "sparse_coo_to_csc",
            id="sparse_coo_to_csc",
        ),
        pytest.param(
            lambda x: sparse.to_dense(sparse.to_coo(x)),
            etl.TensorSpec((3, 4), etl.float32),
            "sparse_csc_to_coo",
            id="sparse_csc_to_coo",
        ),
    ],
)
def test_vjp_deferrals_raise_transform_error(build, cotangents, op_name):
    """The three vjp deferrals raise when the vjp graph is built (the reverse
    sweep applies the rule) — with the op name and 'v1 deferral'."""
    from tests.sparse.conftest import csc_spec

    specs = (csc_spec(),) if op_name == "sparse_csc_to_coo" else (coo_spec(),)
    if op_name == "sparse_concatenate":
        specs = (coo_spec(), coo_spec())
    graph = etl.trace(build, *specs)
    with pytest.raises(core.TransformError, match="v1 deferral") as exc:
        etl.vjp(graph, cotangents=cotangents)
    assert op_name in str(exc.value)


# ---------------------------------------------------------------------------
# 9. jvp numerics
# ---------------------------------------------------------------------------


def test_jvp_sparse_multiply_gather_product_rule():
    """sparse_multiply's jvp is a gather-based product rule: the intersection
    merge reorders rows, so the tangent is ta*vb + va*tb evaluated at the
    merged (intersection) rows."""
    a = sparse.coo(
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        np.array([1.0, 2.0, 4.0], dtype=np.float32),
        (3, 4),
    )
    b = sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
        np.array([3.0, 5.0, 7.0], dtype=np.float32),
        (3, 4),
    )
    ta = np.array([10.0, 20.0, 40.0], dtype=np.float32)
    tb = np.array([30.0, 50.0, 70.0], dtype=np.float32)

    graph = etl.trace(
        lambda x, y: sparse.multiply(x, y), coo_spec(), coo_spec()
    )
    jvp_graph = etl.jvp(graph, tangents=(coo_spec(), coo_spec()))

    primal, tout = run_graph(
        jvp_graph,
        (a, b),
        (np.zeros((3, 2), np.int64), ta, np.zeros((3, 2), np.int64), tb),
    )
    np.testing.assert_allclose(
        primal.to_dense(), a.to_dense() * b.to_dense(), rtol=1e-5, atol=1e-6
    )
    # Intersection rows: (0, 1) and (1, 2).
    expected = np.array(
        [ta[0] * b.to_dense()[0, 1] + a.to_dense()[0, 1] * tb[0],
         ta[1] * b.to_dense()[1, 2] + a.to_dense()[1, 2] * tb[1]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tout[1].numpy(), expected, rtol=1e-5, atol=1e-6)
    assert np.all(tout[0].numpy() == 0)


def test_jvp_multiply_dense_two_term_product_rule():
    """sparse_multiply_dense with tangents on values AND dense: the tangent
    values are tv*dense[pos] + v*td[pos] (structure preserved, rows stay
    aligned — an elementwise product rule)."""
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    tv = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    td = np.full((3, 4), 0.5, dtype=np.float32)

    graph = etl.trace(
        lambda s, d_: sparse.multiply_dense(s, d_),
        coo_spec(),
        etl.TensorSpec((3, 4), etl.float32),
    )
    jvp_graph = etl.jvp(
        graph, tangents=(coo_spec(), etl.TensorSpec((3, 4), etl.float32))
    )

    primal, tout = run_graph(
        jvp_graph,
        (coo_example(), dense),
        (np.zeros((4, 2), np.int64), tv, td),
    )
    np.testing.assert_allclose(
        primal.to_dense(), coo_example().to_dense() * dense, rtol=1e-5, atol=1e-6
    )
    expected = tv * dense[idx[:, 0], idx[:, 1]] + vals * td[idx[:, 0], idx[:, 1]]
    np.testing.assert_allclose(tout[1].numpy(), expected, rtol=1e-5, atol=1e-6)


def test_jvp_dot_dense_two_term_product_rule():
    """sparse@dense with tangents on values AND dense: the output tangent is
    sparse(tv)@dense + sparse(v)@td."""
    dense = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    tv = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    td = np.full((4, 3), 0.5, dtype=np.float32)

    graph = etl.trace(
        lambda s, d_: sparse.matmul(s, d_),
        coo_spec(),
        etl.TensorSpec((4, 3), etl.float32),
    )
    jvp_graph = etl.jvp(
        graph, tangents=(coo_spec(), etl.TensorSpec((4, 3), etl.float32))
    )

    primal, tout = run_graph(
        jvp_graph,
        (coo_example(), dense),
        (np.zeros((4, 2), np.int64), tv, td),
    )
    s = coo_example()
    np.testing.assert_allclose(
        primal.numpy(), s.to_dense() @ dense, rtol=1e-5, atol=1e-6
    )
    expected = sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64), tv, (3, 4)
    ).to_dense() @ dense + s.to_dense() @ td
    np.testing.assert_allclose(tout[0].numpy(), expected, rtol=1e-5, atol=1e-6)


def test_jvp_zero_tangent_run_level_numerics():
    """A zero tangent for one operand (zero arrays at run time) reduces the
    two-term product rule to the single surviving term."""
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    tv = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    graph = etl.trace(
        lambda s, d_: sparse.multiply_dense(s, d_),
        coo_spec(),
        etl.TensorSpec((3, 4), etl.float32),
    )
    jvp_graph = etl.jvp(
        graph, tangents=(coo_spec(), etl.TensorSpec((3, 4), etl.float32))
    )
    _, tout = run_graph(
        jvp_graph,
        (coo_example(), dense),
        (np.zeros((4, 2), np.int64), tv, np.zeros((3, 4), np.float32)),
    )
    expected = tv * dense[idx[:, 0], idx[:, 1]]
    np.testing.assert_allclose(tout[1].numpy(), expected, rtol=1e-5, atol=1e-6)


def test_jvp_rule_zero_tangent_short_circuit():
    """The bilinear jvp rules short-circuit structurally-zero tangents BEFORE
    building any IR: an all-ZeroTangent call returns all-zero tangents (this
    is the registered rule invoked directly — the registry is public API;
    the rule touches the builder only when a term survives)."""
    graph = etl.trace(
        lambda x, y: sparse.multiply(x, y), coo_spec(), coo_spec()
    )
    op = next(
        o for o in graph.module.functions[0].entry_block.ops
        if o.name == "sparse_multiply"
    )
    rule = jvp_rules["sparse_multiply"]
    out = rule(
        op, (ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent())
    )
    assert len(out) == 2
    assert all(isinstance(entry, ZeroTangent) for entry in out)


# ---------------------------------------------------------------------------
# 10. known limitation (documented v1 gap — NOT a bug)
# ---------------------------------------------------------------------------


def test_vmap_grad_over_sparse_defers_known_gap():
    """vectorize of a grad graph over sparse defers with the PRE-EXISTING
    transforms gap 'cannot batch op gather with dynamic index dims' — the
    grad vjp emits dense gather ops with dynamic (nnz) index dims, and
    batching dynamic-index gathers is a documented v1 deferral in
    etl/transforms (it reproduces on pure-dense gather graphs too). This is a
    known limitation, not a regression: the deferral is explicit and names
    the op (no silent fallback)."""

    def loss(x):
        return etl.sum(sparse.to_dense(sparse.negate(x)))

    grad_graph = etl.grad(loss, argnums=(1,))(coo_spec())
    with pytest.raises(
        core.TransformError, match="cannot batch op 'gather'"
    ) as exc:
        etl.vmap(grad_graph, in_axes=0)
    assert "dynamic index dims" in str(exc.value)
