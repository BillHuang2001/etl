"""Control-flow execution tests for the numpy interpreter backend.

`etl.cond` / `etl.while_loop` / `etl.scan` trace Python callables into
ordinary `if` / `while` IR ops executed by the numpy interpreter with
genuinely dynamic runtime control flow (recursive region execution — no
graph-level Python callbacks and no per-iteration re-tracing).
"""

import numpy as np
import pytest

import etl
import etl.numpy as enp


def test_cond_true_and_false_branches():
    @etl.defn
    def fn(pred, x):
        return etl.cond(pred, lambda x: x * 2, lambda x: x - 1, x)

    exe = etl.build(
        fn,
        etl.TensorSpec((), etl.bool_),
        etl.TensorSpec((3,), etl.float32),
    )
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    out_true = etl.run(exe, np.array(True), x)
    np.testing.assert_array_equal(out_true.numpy(), x * 2)

    out_false = etl.run(exe, np.array(False), x)
    np.testing.assert_array_equal(out_false.numpy(), x - 1)


def test_cond_dynamic_pred_both_data_paths():
    # The predicate is computed from the input, so the branch is decided at
    # run time (not at trace time).
    @etl.defn
    def fn(x):
        pred = x[0] > 0
        return etl.cond(pred, lambda: enp.sum(x) * 2, lambda: enp.sum(x) - 1)

    exe = etl.build(fn, etl.TensorSpec((3,), etl.float32))

    out_pos = etl.run(exe, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert out_pos.numpy() == pytest.approx(12.0)

    out_neg = etl.run(exe, np.array([-1.0, 2.0, 3.0], dtype=np.float32))
    assert out_neg.numpy() == pytest.approx(3.0)


def _doubling_fn():
    @etl.defn
    def fn(x, threshold):
        def cond_fn(state):
            i, val = state
            return val < threshold

        def body_fn(state):
            i, val = state
            return (i + 1, val * 2)

        i, val = etl.while_loop(
            cond_fn, body_fn, (enp.zeros((), etl.int64), x)
        )
        return i, val

    return fn


def test_while_loop_iteration_count_and_final_value():
    fn = _doubling_fn()
    exe = etl.build(
        fn,
        etl.TensorSpec((), etl.float32),
        etl.TensorSpec((), etl.float32),
    )

    def reference(x, threshold):
        i = 0
        while x < threshold:
            x = x * 2
            i += 1
        return i, x

    for x, threshold in [(1.0, 10.0), (3.0, 20.0), (0.5, 100.0)]:
        i, val = etl.run(
            exe,
            np.array(x, dtype=np.float32),
            np.array(threshold, dtype=np.float32),
        )
        ref_i, ref_val = reference(x, threshold)
        assert i.numpy() == ref_i
        assert val.numpy() == pytest.approx(ref_val)


def test_while_loop_zero_iterations():
    # Condition immediately false: the body never runs.
    fn = _doubling_fn()
    exe = etl.build(
        fn,
        etl.TensorSpec((), etl.float32),
        etl.TensorSpec((), etl.float32),
    )

    i, val = etl.run(
        exe,
        np.array(1.0, dtype=np.float32),
        np.array(1.0, dtype=np.float32),
    )
    assert i.numpy() == 0
    assert val.numpy() == pytest.approx(1.0)


def test_while_loop_early_exit():
    # Exits after exactly one body iteration.
    fn = _doubling_fn()
    exe = etl.build(
        fn,
        etl.TensorSpec((), etl.float32),
        etl.TensorSpec((), etl.float32),
    )

    i, val = etl.run(
        exe,
        np.array(1.0, dtype=np.float32),
        np.array(2.0, dtype=np.float32),
    )
    assert i.numpy() == 1
    assert val.numpy() == pytest.approx(2.0)


def test_scan_cumulative_sum():
    @etl.defn
    def fn(xs):
        def f(carry, x):
            return carry + x, carry + x

        carry, ys = etl.scan(f, enp.zeros((), dtype=xs.dtype), xs)
        return carry, ys

    exe = etl.build(fn, etl.TensorSpec((4,), etl.float32))
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    carry, ys = etl.run(exe, xs)
    np.testing.assert_array_equal(ys.numpy(), np.cumsum(xs))
    assert carry.numpy() == pytest.approx(np.sum(xs))


def test_scan_sum_of_squares_carry_and_stacked():
    @etl.defn
    def fn(xs):
        def f(carry, x):
            y = x * x
            return carry + y, y

        carry, ys = etl.scan(f, enp.zeros((), dtype=xs.dtype), xs)
        return carry, ys

    exe = etl.build(fn, etl.TensorSpec((4,), etl.float32))
    xs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    carry, ys = etl.run(exe, xs)
    np.testing.assert_array_equal(ys.numpy(), xs * xs)
    assert carry.numpy() == pytest.approx(np.sum(xs * xs))


def test_nested_cond_inside_while_body():
    # Each body iteration picks a branch at run time.
    @etl.defn
    def fn(x):
        def cond_fn(state):
            i, val = state
            return i < 4

        def body_fn(state):
            i, val = state
            new_val = etl.cond(val > 0, lambda: val - 2, lambda: val + 1)
            return (i + 1, new_val)

        i, val = etl.while_loop(
            cond_fn, body_fn, (enp.zeros((), etl.int64), x)
        )
        return val

    exe = etl.build(fn, etl.TensorSpec((), etl.float32))

    # 5 -> 3 -> 1 -> -1 -> 0
    out = etl.run(exe, np.array(5.0, dtype=np.float32))
    assert out.numpy() == pytest.approx(0.0)

    # -3 -> -2 -> -1 -> 0 -> 1
    out = etl.run(exe, np.array(-3.0, dtype=np.float32))
    assert out.numpy() == pytest.approx(1.0)


def test_nested_while_inside_cond_branch():
    @etl.defn
    def fn(pred, x):
        def loop():
            def cond_fn(state):
                i, val = state
                return val < 8

            def body_fn(state):
                i, val = state
                return (i + 1, val * 2)

            i, val = etl.while_loop(
                cond_fn, body_fn, (enp.zeros((), etl.int64), x)
            )
            return val

        return etl.cond(pred, loop, lambda: x - 1)

    exe = etl.build(
        fn,
        etl.TensorSpec((), etl.bool_),
        etl.TensorSpec((), etl.float32),
    )

    # True branch: 1 -> 2 -> 4 -> 8 (loop stops at >= 8).
    out = etl.run(exe, np.array(True), np.array(1.0, dtype=np.float32))
    assert out.numpy() == pytest.approx(8.0)

    # False branch: no loop.
    out = etl.run(exe, np.array(False), np.array(1.0, dtype=np.float32))
    assert out.numpy() == pytest.approx(0.0)


def test_while_loop_in_symbolic_batch_graph():
    # Control flow inside a graph with a symbolic dim B: the loop iterates
    # until val >= sum(xs), so the runtime iteration count depends on the
    # concrete input bound to B.
    @etl.defn
    def fn(xs):
        target = enp.sum(xs)

        def cond_fn(state):
            i, val = state
            return val < target

        def body_fn(state):
            i, val = state
            return (i + 1, val * 2)

        i, val = etl.while_loop(
            cond_fn, body_fn, (enp.zeros((), etl.int64), enp.ones((), xs.dtype))
        )
        return i, val

    exe = etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32))

    i, val = etl.run(exe, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert i.numpy() == 3
    assert val.numpy() == pytest.approx(8.0)

    i, val = etl.run(exe, np.array([10.0], dtype=np.float32))
    assert i.numpy() == 4
    assert val.numpy() == pytest.approx(16.0)

    i, val = etl.run(exe, np.array([0.5, 0.25], dtype=np.float32))
    assert i.numpy() == 0
    assert val.numpy() == pytest.approx(1.0)


def test_scan_with_symbolic_length_raises_trace_error():
    # Documented v1 reservation: symbolic scan lengths fail explicitly at
    # trace time instead of silently falling back.
    @etl.defn
    def fn(xs):
        def f(carry, x):
            return carry + x, carry + x

        carry, ys = etl.scan(f, enp.zeros((), dtype=xs.dtype), xs)
        return carry, ys

    with pytest.raises(etl.TraceError, match="symbolic/dynamic scan lengths"):
        etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32))
