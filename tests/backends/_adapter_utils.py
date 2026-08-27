"""Shared helpers for the three optional compiler-adapter test files.

``test_adapter_iree.py``, ``test_adapter_xla.py`` and ``test_adapter_tvm.py``
exercise ONE contract — the shared ``CompilerBackend`` framework
(``etl/backends/compiler.py``) plus each adapter's real compile/load/run
through its compiler dependency — against three different dependencies. The
three files differ only in which dependency they import and which backend
name they use; the graphs, tolerances and parity checks live here so the
contract is written exactly once.

Import package-qualified (``from tests.backends import _adapter_utils``):
``tests/backends`` is a package and bare imports break pytest collection.
The underscore prefix keeps this module out of collection entirely
(precedent: ``tests/numpy/_ir_utils.py``).
"""

import numpy as np

import etl

# Cross-compiler float tolerance. IREE/XLA/TVM run the real StableHLO
# lowering/fusion pipeline, so fp32 results may differ from the numpy
# reference in the last ULP — 1e-5 relative/absolute covers compiler
# reordering while still catching genuine numerical errors (the pure-numpy
# interpreter tests use 1e-6).
FP32_RTOL = 1e-5
FP32_ATOL = 1e-5

# Fixed seed: the three adapter files draw the same deterministic inputs.
_RNG = np.random.default_rng(20241104)


def standard_normal(shape):
    """A deterministic fp32 draw from the fixed-seed RNG."""
    return _RNG.standard_normal(shape).astype(np.float32)


def matmul_relu_sum():
    """``fn(a, b) = sum(relu(dot(a, b)))`` over (4, 8) x (8, 4)."""

    def fn(a, b):
        return etl.sum(etl.relu(etl.dot(a, b)))

    return fn, (
        etl.TensorSpec((4, 8), etl.float32),
        etl.TensorSpec((8, 4), etl.float32),
    )


def matmul_relu_sum_args():
    return standard_normal((4, 8)), standard_normal((8, 4))


def reshape_broadcast_transpose():
    """``fn(x) = transpose(broadcast(reshape(x, (2, 4, 4)), (3, 2, 4, 4)), (0, 2, 1, 3))``."""

    def fn(x):
        r = etl.reshape(x, (2, 4, 4))
        b = etl.broadcast(r, (3, 2, 4, 4))
        return etl.transpose(b, (0, 2, 1, 3))

    return fn, (etl.TensorSpec((4, 8), etl.float32),)


def symbolic_scale():
    """``fn(x) = x * 2 + 1`` over a symbolic ``(B,)`` batch."""

    def fn(x):
        return x * 2.0 + 1.0

    return fn, (etl.TensorSpec((etl.dim("B"),), etl.float32),)


def _double_callback(x):
    return x * 2.0


def runtime_call_graph():
    """A verified graph carrying a ``runtime_call`` op (module-level callback)."""

    def fn(x):
        return etl.runtime_call(
            _double_callback, x, result=etl.TensorSpec((4,), etl.float32)
        )

    return etl.trace(fn, etl.TensorSpec((4,), etl.float32))


def collective_graph():
    """A verified graph carrying an explicit ``dist.all_reduce`` collective."""

    def fn(x):
        return etl.dist.all_reduce(x)

    return etl.trace(fn, etl.TensorSpec((4,), etl.float32))


def stage(backend_name, fn, specs):
    """Front half of the pipeline via backend-name resolution: trace → lower → compile."""
    graph = etl.trace(fn, *specs)
    lowered = etl.lower(graph, backend=backend_name)
    artifact = etl.compile(lowered)
    return graph, lowered, artifact


def assert_parity(fn, args, executable):
    """Run ``fn`` through ``executable`` and compare against the default
    numpy-backend evaluation at the cross-compiler fp tolerance."""
    expected = etl.evaluate(fn, *args)  # default numpy reference
    actual = etl.run(executable, *args)
    assert isinstance(actual, etl.Tensor)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=FP32_RTOL, atol=FP32_ATOL
    )
