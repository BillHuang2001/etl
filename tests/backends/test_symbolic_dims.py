"""Symbolic-dimension execution tests for the numpy interpreter backend.

A single executable built from a graph with symbolic (`Dim`/`DimExpr`) or
runtime-dynamic (`None`) dims must run at multiple concrete sizes — the
backend's `dynamic_shapes` capability. Dims bind from the concrete input
shapes at run time (the graph itself is built only once).
"""

import numpy as np
import pytest

import etl
import etl.numpy as enp


def test_symbolic_batch_one_build_multiple_sizes():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    exe = etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32))

    out3 = etl.run(exe, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    # etl.run returns structured outputs wrapping core.Tensor values.
    assert isinstance(out3, etl.Tensor)
    np.testing.assert_array_equal(
        out3.numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32)
    )

    out5 = etl.run(exe, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(
        out5.numpy(), np.array([3.0, 5.0, 7.0, 9.0, 11.0], dtype=np.float32)
    )


def test_dimexpr_arithmetic_reshape_to_double_batch():
    # Dim arithmetic yields DimExpr values usable as op shapes.
    assert isinstance(etl.dim("B") * 2, etl.DimExpr)

    @etl.defn
    def fn(x):
        # x has shape (B,); concat(x, x) has shape (2B,) == (B * 2,).
        d = x.shape[0] * 2
        return enp.reshape(enp.concatenate([x, x]), (d,))

    exe = etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32))

    x3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out3 = etl.run(exe, x3)
    assert out3.numpy().shape == (6,)
    np.testing.assert_array_equal(out3.numpy(), np.concatenate([x3, x3]))

    x5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out5 = etl.run(exe, x5)
    assert out5.numpy().shape == (10,)
    np.testing.assert_array_equal(out5.numpy(), np.concatenate([x5, x5]))


def test_two_symbolic_dims_transpose():
    @etl.defn
    def fn(x):
        return enp.transpose(x)

    exe = etl.build(
        fn, etl.TensorSpec((etl.dim("B"), etl.dim("C")), etl.float32)
    )

    x34 = np.arange(12, dtype=np.float32).reshape(3, 4)
    out34 = etl.run(exe, x34)
    assert out34.numpy().shape == (4, 3)
    np.testing.assert_array_equal(out34.numpy(), x34.T)

    x52 = np.arange(10, dtype=np.float32).reshape(5, 2)
    out52 = etl.run(exe, x52)
    assert out52.numpy().shape == (2, 5)
    np.testing.assert_array_equal(out52.numpy(), x52.T)


def test_none_dim_runtime_dynamic_multiple_sizes():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    # None dims mark runtime-dynamic sizes: rank checked, size unconstrained.
    exe = etl.build(fn, etl.TensorSpec((None,), etl.float32))

    out3 = etl.run(exe, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(
        out3.numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32)
    )

    out7 = etl.run(
        exe, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        out7.numpy(), np.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype=np.float32)
    )


def test_none_dim_wrong_rank_raises_shape_error():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    exe = etl.build(fn, etl.TensorSpec((None,), etl.float32))

    # The size is free but the rank is not: a 2-d input against a 1-d
    # runtime-dynamic spec must fail explicitly.
    with pytest.raises(etl.ShapeError):
        etl.run(exe, np.ones((2, 2), dtype=np.float32))


def test_symbolic_determinism_same_input_same_result():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    exe = etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32))
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)

    first = etl.run(exe, x).numpy()
    second = etl.run(exe, x).numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.array([5.0, 7.0, 9.0], dtype=np.float32))


def test_mixed_symbolic_and_concrete_inputs():
    @etl.defn
    def fn(x, w):
        return enp.dot(x, w)

    # Symbolic batch B in one input, concrete (3, 2) in the other.
    exe = etl.build(
        fn,
        etl.TensorSpec((etl.dim("B"), 3), etl.float32),
        etl.TensorSpec((3, 2), etl.float32),
    )
    w = np.arange(6, dtype=np.float32).reshape(3, 2)

    x3 = np.arange(9, dtype=np.float32).reshape(3, 3)
    out3 = etl.run(exe, x3, w)
    assert out3.numpy().shape == (3, 2)
    np.testing.assert_array_equal(out3.numpy(), x3 @ w)

    x5 = np.arange(15, dtype=np.float32).reshape(5, 3)
    out5 = etl.run(exe, x5, w)
    assert out5.numpy().shape == (5, 2)
    np.testing.assert_array_equal(out5.numpy(), x5 @ w)


def test_mixed_symbolic_concrete_size_mismatch_raises_etl_error():
    @etl.defn
    def fn(x, w):
        return x * w

    exe = etl.build(
        fn,
        etl.TensorSpec((etl.dim("B"),), etl.float32),
        etl.TensorSpec((3,), etl.float32),
    )

    # BUG(etl): a symbolic-vs-concrete size mismatch (B=4 vs 3) escapes the
    # numpy interpreter as a raw builtins.ValueError ("operands could not be
    # broadcast together with shapes (4,) (3,)") instead of an etl.ShapeError
    # — every public error must derive from ETLError.
    with pytest.raises(etl.ShapeError):
        etl.run(
            exe,
            np.ones(4, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
