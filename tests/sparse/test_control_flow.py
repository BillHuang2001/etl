"""Sparse tensors through the traced control-flow ops: `cond`, `while_loop`,
and `scan`.

Contract under test: ``../etl/sparse/CONTEXT.md`` ("Known issues / v1
deferrals": sparse loop CARRIES through scan/cond/while_loop work — trace-level
support is generic registered-pytree handling; scan over sparse `xs` and sparse
`y`-stacking are explicit TraceError deferrals) and ``../etl/trace/CONTEXT.md``
(control-flow region conventions, static-leaf snapshotting). All numerics are
checked against pure-numpy dense references with tight tolerances.

Pinned behaviors (verified against the current implementation):

* sparse values flow as loop CARRIES through cond/while_loop/scan; the
  dense_shape/dtype/format static leaves snapshot as static values (3, 4,
  float32, "coo") and stay out of the carried tensor leaves.
* `scan` over sparse `xs` defers at trace time:
  "etl.scan: xs leaf 2 must be a core.SymbolicTensor, got int" (the static
  dense_shape leaf of the flattened sparse is rejected — the flattened xs
  contains non-tensor leaves).
* `scan` whose body returns a SPARSE y (the value to stack) defers at trace
  time: "etl.scan: f's y outputs must be SymbolicTensors (at least one) — a
  scan with no stacked tensor outputs cannot be traced".
"""

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse

from tests.sparse.conftest import (
    coo_example,
    coo_spec,
    dense_example,
    materialize,
    run_graph,
)


def _const(value, dtype):
    """Embed a static value as a constant op (valid only inside a trace)."""
    return etl.constant(etl.tensor(value, dtype=dtype))


# ---------------------------------------------------------------------------
# 1. cond
# ---------------------------------------------------------------------------


def test_cond_sparse_returning_branches():
    """cond(p, negate, add(x, x)) with sparse-returning branches: the run
    result is a concrete sparse; True/False match the dense references."""

    def f(p, x):
        return etl.cond(
            p,
            lambda v: sparse.negate(v),
            lambda v: sparse.add(v, v),
            x,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.bool_), coo_spec())
    d = dense_example()

    got_true = run_graph(graph, np.array(True), coo_example())
    got_false = run_graph(graph, np.array(False), coo_example())
    assert sparse.is_sparse(got_true) and sparse.is_sparse(got_false)
    np.testing.assert_allclose(got_true.to_dense(), -d, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_false.to_dense(), 2.0 * d, rtol=1e-5, atol=1e-6)


def test_cond_dense_returning_branches():
    """cond whose branches return DENSE (to_dense inside the branches) works
    the same way and yields a dense tensor."""

    def f(p, x):
        return etl.cond(
            p,
            lambda v: sparse.to_dense(sparse.negate(v)),
            lambda v: sparse.to_dense(sparse.add(v, v)),
            x,
        )

    graph = etl.trace(f, etl.TensorSpec((), etl.bool_), coo_spec())
    d = dense_example()

    got_true = materialize(run_graph(graph, np.array(True), coo_example()))
    got_false = materialize(run_graph(graph, np.array(False), coo_example()))
    np.testing.assert_allclose(got_true, -d, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got_false, 2.0 * d, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. while_loop
# ---------------------------------------------------------------------------


def test_while_loop_sparse_carry_doubling():
    """Sparse carry through while_loop: cond on `sparse.sum(c) < threshold`
    (sum -> dense scalar), body `sparse.add(c, c)`. The canonical example sums
    to 10, so a threshold of 100 runs exactly 4 iterations (10, 20, 40, 80 <
    100; 160 is not) — the result must equal dense * 2**4."""

    def f(c):
        def cond_fn(carry):
            return etl.sparse.sum(carry) < _const(100.0, etl.float32)

        def body_fn(carry):
            return sparse.add(carry, carry)

        return etl.while_loop(cond_fn, body_fn, c)

    graph = etl.trace(f, coo_spec())
    out = run_graph(graph, coo_example())

    d = dense_example()
    np.testing.assert_allclose(out.to_dense(), d * 2 ** 4, rtol=1e-5, atol=1e-6)
    assert np.isclose(np.sum(out.to_dense()), 160.0, rtol=1e-5)


def test_while_loop_sparse_dense_tuple_carry():
    """A while loop carrying a (sparse, dense) tuple: both entries double per
    iteration and must match the dense reference after 4 iterations."""

    def f(c, w):
        def cond_fn(state):
            carry, _ = state
            return etl.sparse.sum(carry) < _const(100.0, etl.float32)

        def body_fn(state):
            carry, dense = state
            return (sparse.add(carry, carry), etl.add(dense, dense))

        return etl.while_loop(cond_fn, body_fn, (c, w))

    graph = etl.trace(f, coo_spec(), etl.TensorSpec((3, 4), etl.float32))
    carry, dense = run_graph(graph, coo_example(), dense_example())

    d = dense_example()
    np.testing.assert_allclose(carry.to_dense(), d * 2 ** 4, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dense.numpy(), d * 2 ** 4, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. scan (sparse carry + dense xs)
# ---------------------------------------------------------------------------


def test_scan_sparse_carry_dense_xs():
    """scan with a sparse carry and dense xs: body(carry, x) returns
    (add(carry, multiply_dense(carry, x)), dense y). The final carry must
    match the dense reference accumulation c_{k+1} = (1 + x_k) * c_k."""

    def f(c, xs):
        def body(carry, x):
            scaled = sparse.multiply_dense(carry, x)
            return (sparse.add(carry, scaled), etl.sum(x))

        return etl.scan(body, c, xs)

    xs = np.array([0.5, 1.0, 2.0], dtype=np.float32).reshape(3, 1, 1) * np.ones(
        (1, 3, 4), dtype=np.float32
    )
    graph = etl.trace(f, coo_spec(), etl.TensorSpec((3, 3, 4), etl.float32))
    final_carry, stacked = run_graph(graph, coo_example(), xs)

    ref = dense_example().copy()
    for step in xs:
        ref = ref + step * ref
    np.testing.assert_allclose(final_carry.to_dense(), ref, rtol=1e-5, atol=1e-6)

    # The dense y outputs stack along the scan axis as usual.
    np.testing.assert_allclose(
        stacked.numpy(), xs.sum(axis=(1, 2)), rtol=1e-5, atol=1e-6
    )


def test_scan_sparse_carry_init_is_traced_input():
    """The sparse carry must be a traced sparse input (or otherwise symbolic):
    a sparse built from graph inputs inside the traced function works as the
    scan init too."""

    def f(c, w, xs):
        acc = sparse.multiply_dense(c, w)

        def body(carry, x):
            return (sparse.add(carry, carry), etl.sum(x))

        return etl.scan(body, acc, xs)

    w = np.full((3, 4), 2.0, dtype=np.float32)
    xs = np.array([1.0, 2.0], dtype=np.float32)
    graph = etl.trace(
        f, coo_spec(), etl.TensorSpec((3, 4), etl.float32), etl.TensorSpec((2,), etl.float32)
    )
    final_carry, stacked = run_graph(graph, coo_example(), w, xs)

    d = dense_example()
    np.testing.assert_allclose(final_carry.to_dense(), 2.0 * d * 2 ** 2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(stacked.numpy(), xs, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. scan over sparse xs: v1 deferral (explicit TraceError)
# ---------------------------------------------------------------------------


def test_scan_over_sparse_xs_raises():
    """scan over sparse xs defers at trace time: the flattened xs contains the
    static dense_shape leaves, so the leaf check rejects them explicitly (the
    pinned message names the leaf kind)."""
    with pytest.raises(core.TraceError, match="must be a core.SymbolicTensor") as exc:
        etl.trace(
            lambda c, xs: etl.scan(lambda carry, x: (carry, x), c, xs),
            coo_spec(),
            coo_spec(),
        )
    assert "etl.scan: xs leaf 2" in str(exc.value)
    assert "got int" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. scan sparse y-stacking: v1 deferral (explicit TraceError)
# ---------------------------------------------------------------------------


def test_scan_sparse_y_stacking_raises():
    """scan whose body returns a SPARSE y (the value to stack) defers at trace
    time: the flattened y contains static leaves, which are not stackable
    tensor outputs — an explicit TraceError (never a silent wrong stack)."""
    with pytest.raises(core.TraceError, match="y outputs must be SymbolicTensors") as exc:
        etl.trace(
            lambda c, xs: etl.scan(
                lambda carry, x: (sparse.add(carry, carry), sparse.negate(carry)),
                c,
                xs,
            ),
            coo_spec(),
            etl.TensorSpec((3,), etl.float32),
        )
    assert "no stacked tensor outputs" in str(exc.value)


# ---------------------------------------------------------------------------
# 6. static-leaves snapshotting through control flow
# ---------------------------------------------------------------------------


def test_sparse_static_leaves_snapshot_through_while_loop():
    """Sparse carries keep their dense_shape/dtype/format static leaves: the
    while-loop graph's static values include the (3, 4) dense_shape entries
    (and the dtype/format leaves), so the runtime rebuilds a (3, 4) sparse."""

    def f(c):
        def cond_fn(carry):
            return etl.sparse.sum(carry) < _const(100.0, etl.float32)

        def body_fn(carry):
            return sparse.add(carry, carry)

        return etl.while_loop(cond_fn, body_fn, c)

    graph = etl.trace(f, coo_spec())
    statics = [record.value for record in graph.static_values]
    assert 3 in statics and 4 in statics
    assert np.dtype("float32") in statics
    assert "coo" in statics

    out = run_graph(graph, coo_example())
    assert sparse.is_sparse(out)
    assert out.dense_shape == (3, 4)
    np.testing.assert_allclose(
        out.to_dense(), dense_example() * 2 ** 4, rtol=1e-5, atol=1e-6
    )
