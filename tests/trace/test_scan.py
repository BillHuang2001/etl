"""Contract tests for `etl.scan`.

`scan` desugars to ordinary while-loop IR: a static-length loop over xs's
leading axis calling ``f(carry, x_step) -> (new_carry, y_step)``, stacking
the y steps. v1 scope (see `etl/trace/CONTEXT.md`): STATIC scan length only —
an explicit int, or derived from xs's static leading dim. Symbolic/dynamic
lengths are reserved and raise `TraceError`. These tests pin the semantics,
the desugared IR structure, and the error surface.
"""

import numpy as np
import pytest

import etl


# --- helpers -----------------------------------------------------------------


def _all_ops(module):
    """Collect every op in a module, descending into nested regions."""
    ops = []

    def walk_block(block):
        for op in block.ops:
            ops.append(op)
            for region in op.regions:
                for nested in region.blocks:
                    walk_block(nested)

    for function in module.functions:
        for block in function.region.blocks:
            walk_block(block)
    return ops


def _op_names(graph):
    return [op.name for op in _all_ops(graph.module)]


# --- scan step functions (executed under etl.scan) ----------------------------


def cumsum_step(carry, x):
    return etl.add(carry, x), etl.add(carry, x)


def cumsum_scan(xs):
    init = etl.constant(etl.tensor(0.0, dtype=etl.float32))
    return etl.scan(cumsum_step, init, xs)


def cumsum_scan_with_length(xs, length):
    init = etl.constant(etl.tensor(0.0, dtype=etl.float32))
    return etl.scan(cumsum_step, init, xs, length=length)


def running_max_step(carry, x):
    new_carry = etl.maximum(carry, x)
    return new_carry, new_carry


def running_max_scan(xs):
    init = etl.constant(etl.tensor(float("-inf"), dtype=etl.float32))
    return etl.scan(running_max_step, init, xs)


def structured_step(carry, x):
    acc_a, acc_b, tag = carry
    xa, xb = x
    new_a = etl.add(acc_a, xa)
    new_b = etl.add(acc_b, xb)
    return (new_a, new_b, tag), (new_a, new_b)


def structured_scan(xs):
    init = (
        etl.constant(etl.tensor(0.0, dtype=etl.float32)),
        etl.constant(etl.tensor(1.0, dtype=etl.float32)),
        7,  # a static leaf inside the carried tree
    )
    return etl.scan(structured_step, init, xs)


# --- semantics ----------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_scan_cumsum(n, run_graph, as_numpy):
    xs = np.arange(1, n + 1, dtype=np.float32)
    graph = etl.trace(cumsum_scan, etl.TensorSpec((n,), etl.float32))

    carry, stacked = as_numpy(run_graph(graph, xs))

    np.testing.assert_allclose(stacked, np.cumsum(xs), rtol=0, atol=0)
    np.testing.assert_allclose(carry, np.sum(xs), rtol=0, atol=0)
    assert stacked.shape == (n,)
    assert stacked.dtype == np.float32


def test_scan_running_max(run_graph, as_numpy):
    xs = np.array([3.0, -1.0, 5.0, 2.0, 8.0, 4.0, 9.0], dtype=np.float32)
    graph = etl.trace(running_max_scan, etl.TensorSpec(xs.shape, etl.float32))

    carry, stacked = as_numpy(run_graph(graph, xs))

    np.testing.assert_array_equal(stacked, np.maximum.accumulate(xs))
    np.testing.assert_equal(carry, np.max(xs))


# --- length override ----------------------------------------------------------


def test_scan_length_override_shortens_xs(run_graph, as_numpy):
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    graph = etl.trace(cumsum_scan_with_length, etl.TensorSpec((4,), etl.float32), 2)

    carry, stacked = as_numpy(run_graph(graph, xs, 2))

    np.testing.assert_allclose(stacked, np.cumsum(xs[:2]), rtol=0, atol=0)
    np.testing.assert_allclose(carry, xs[0] + xs[1], rtol=0, atol=0)
    assert stacked.shape == (2,)


def test_scan_explicit_length_equal_to_dim(run_graph, as_numpy):
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    graph = etl.trace(cumsum_scan_with_length, etl.TensorSpec((4,), etl.float32), 4)

    carry, stacked = as_numpy(run_graph(graph, xs, 4))

    np.testing.assert_allclose(stacked, np.cumsum(xs), rtol=0, atol=0)
    np.testing.assert_allclose(carry, np.sum(xs), rtol=0, atol=0)


@pytest.mark.parametrize("length", [5, 8])
def test_scan_length_larger_than_static_dim_raises(length):
    with pytest.raises(etl.TraceError, match="does not match xs's static leading dim"):
        etl.trace(cumsum_scan_with_length, etl.TensorSpec((4,), etl.float32), length)


# --- structured xs and init ---------------------------------------------------


def test_scan_structured_xs_and_init(run_graph, as_numpy):
    xa = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    xb = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    graph = etl.trace(
        structured_scan,
        (etl.TensorSpec((4,), etl.float32), etl.TensorSpec((4,), etl.float32)),
    )

    carry, stacked = as_numpy(run_graph(graph, (xa, xb)))

    # y outputs are stacked per y's tree: (cumsum(xa), init_b + cumsum(xb))
    np.testing.assert_allclose(stacked[0], np.cumsum(xa), rtol=0, atol=0)
    np.testing.assert_allclose(stacked[1], 1.0 + np.cumsum(xb), rtol=0, atol=0)
    # carries follow init's tree, static leaf passes through unchanged
    np.testing.assert_allclose(carry[0], np.sum(xa), rtol=0, atol=0)
    np.testing.assert_allclose(carry[1], 1.0 + np.sum(xb), rtol=0, atol=0)
    assert carry[2] == 7


# --- IR structure -------------------------------------------------------------


def test_scan_ir_structure(run_graph, as_numpy):
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    graph = etl.trace(cumsum_scan, etl.TensorSpec((4,), etl.float32))

    graph.verify()  # must not raise

    # the desugaring: one typed while op plus the raw region ops it uses
    names = _op_names(graph)
    assert names.count("while") == 1
    for required in ("gather", "scatter", "reshape", "broadcast", "constant", "less"):
        assert required in names, f"desugared scan must contain a {required!r} op"

    carry, stacked = as_numpy(run_graph(graph, xs))
    np.testing.assert_allclose(stacked, np.cumsum(xs), rtol=0, atol=0)


# --- errors -------------------------------------------------------------------


def test_scan_symbolic_leading_dim_raises():
    with pytest.raises(etl.TraceError, match="reserved"):
        etl.trace(cumsum_scan, etl.TensorSpec((etl.Dim("N"),), etl.float32))


def test_scan_dynamic_leading_dim_raises():
    with pytest.raises(etl.TraceError, match="reserved"):
        etl.trace(cumsum_scan, etl.TensorSpec((None,), etl.float32))


def test_scan_rank_zero_xs_raises():
    with pytest.raises(etl.TraceError, match="rank"):
        etl.trace(cumsum_scan, etl.TensorSpec((), etl.float32))


def test_scan_xs_without_tensor_leaves_raises():
    # a static-only xs (bare int) has no SymbolicTensor leaf
    with pytest.raises(etl.TraceError, match="must be a core.SymbolicTensor"):
        etl.trace(cumsum_scan, 3)
    # an empty xs tree has no leaves at all
    with pytest.raises(etl.TraceError, match="at least one SymbolicTensor"):
        etl.trace(cumsum_scan, ())


def test_scan_ndarray_xs_leaf_raises():
    with pytest.raises(
        etl.TraceError, match="is neither a core.TensorSpec nor a static"
    ):
        etl.trace(cumsum_scan, np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize("length", [0, -1])
def test_scan_nonpositive_length_raises(length):
    with pytest.raises(etl.TraceError, match="empty scans are not supported"):
        etl.trace(cumsum_scan_with_length, etl.TensorSpec((4,), etl.float32), length)


@pytest.mark.parametrize("length", [2.5, "4", True], ids=["float", "str", "bool"])
def test_scan_non_int_length_raises(length):
    with pytest.raises(etl.TraceError, match="must be a static int"):
        etl.trace(cumsum_scan_with_length, etl.TensorSpec((4,), etl.float32), length)


def test_scan_result_must_be_a_pair():
    def bad_scan(xs):
        def step(carry, x):
            return etl.add(carry, x)  # not a (new_carry, y_step) pair

        return etl.scan(step, etl.constant(etl.tensor(0.0, dtype=etl.float32)), xs)

    with pytest.raises(etl.TraceError, match=r"must return a \(new_carry, y_step\) pair"):
        etl.trace(bad_scan, etl.TensorSpec((4,), etl.float32))


def test_scan_carry_tree_must_match_init():
    def bad_scan(xs):
        def step(carry, x):
            new = etl.add(carry, x)
            return (new, new), new  # carry tree (tensor, tensor) vs init tensor

        return etl.scan(step, etl.constant(etl.tensor(0.0, dtype=etl.float32)), xs)

    with pytest.raises(etl.TraceError, match="must match init's tree"):
        etl.trace(bad_scan, etl.TensorSpec((4,), etl.float32))


def test_scan_static_only_y_raises():
    def bad_scan(xs):
        def step(carry, x):
            return etl.add(carry, x), 3  # y has no tensor leaves

        return etl.scan(step, etl.constant(etl.tensor(0.0, dtype=etl.float32)), xs)

    with pytest.raises(etl.TraceError, match="SymbolicTensors"):
        etl.trace(bad_scan, etl.TensorSpec((4,), etl.float32))


# --- equivalence with a manual loop -------------------------------------------


def test_scan_matches_manual_loop(run_graph, as_numpy):
    xs = np.array([5.0, 3.0, 1.0, 4.0, 2.0], dtype=np.float32)

    def step(carry, x):
        # non-commutative on purpose: iteration order matters
        return etl.subtract(carry, x), etl.multiply(x, carry)

    def scan_fn(xs):
        return etl.scan(step, etl.constant(etl.tensor(10.0, dtype=etl.float32)), xs)

    graph = etl.trace(scan_fn, etl.TensorSpec((5,), etl.float32))
    carry, stacked = as_numpy(run_graph(graph, xs))

    expected_carry, expected_ys = 10.0, []
    for x in xs:
        expected_ys.append(x * expected_carry)
        expected_carry = expected_carry - x
    np.testing.assert_allclose(stacked, np.asarray(expected_ys, dtype=np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(carry, expected_carry, rtol=0, atol=0)
