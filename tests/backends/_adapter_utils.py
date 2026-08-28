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


def symbolic_softmax():
    """``fn(x) = softmax(x)`` over a symbolic ``(B, 16)`` batch.

    The keepdims reduces produce dynamic keepdims reshapes ((B,) -> (B, 1)),
    which the StableHLO writer must reject at lower() time with etl's own
    BackendError — never invalid MLIR that dies later with a compiler parse
    error or runtime abort (see ``Writer._reject_dynamic_dims``).
    """

    def fn(x):
        e = etl.exp(x - etl.max(x, axes=1, keepdims=True))
        return e / etl.sum(e, axes=1, keepdims=True)

    return fn, (etl.TensorSpec((etl.dim("B"), 16), etl.float32),)


def symbolic_slice():
    """``fn(x) = slice(x, [0:2, 0:4])`` over a symbolic ``(B, 4)`` batch.

    The StableHLO writer rejects dynamic slice at export time: iree 3.11.0
    parses dynamic-operand slice MLIR but the runtime ABORTS on it
    ("hal.fence.await" failure) — the rejection must never be skipped.
    """

    def fn(x):
        return etl.slice(x, (0, 0), (2, 4))

    return fn, (etl.TensorSpec((etl.dim("B"), 4), etl.float32),)


def symbolic_pad():
    """``fn(x) = pad(x, [[0,0],[1,1]])`` over a symbolic ``(B, 4)`` batch.

    Same empirical rationale as ``symbolic_slice`` (iree 3.11.0 runtime
    abort on dynamic-shape pad).
    """

    def fn(x):
        return etl.pad(x, [[0, 0], [1, 1]])

    return fn, (etl.TensorSpec((etl.dim("B"), 4), etl.float32),)


def _dot_fn(a, b):
    return etl.dot(a, b)


def batched_dot_cases():
    """The three static batched-dot emission paths pinned by the writer
    goldens (``tests/backends/test_stablehlo.py``): rank-3@rank-2
    (non-batched dot_general), rhs-higher-rank (lhs broadcast_in_dim),
    size-1 rhs batch (squeeze reshape). Returns ``{name: (fn, specs,
    args)}`` with deterministic fixed-seed fp32 args matching the specs."""
    return {
        "rank3_rank2": (
            _dot_fn,
            (etl.TensorSpec((4, 512, 768), etl.float32),
             etl.TensorSpec((768, 2304), etl.float32)),
            (standard_normal((4, 512, 768)), standard_normal((768, 2304))),
        ),
        "rhs_higher_rank": (
            _dot_fn,
            (etl.TensorSpec((512, 768), etl.float32),
             etl.TensorSpec((4, 768, 2304), etl.float32)),
            (standard_normal((512, 768)), standard_normal((4, 768, 2304))),
        ),
        "size1_rhs_batch": (
            _dot_fn,
            (etl.TensorSpec((4, 512, 768), etl.float32),
             etl.TensorSpec((1, 768, 2304), etl.float32)),
            (standard_normal((4, 512, 768)), standard_normal((1, 768, 2304))),
        ),
    }


def batched_dot_symbolic():
    """``fn(a, b) = dot(a, b)`` over a symbolic batch
    ``(B, 512, 768) @ (768, 2304)``: the rhs matrix is broadcast up to the
    runtime batch via ``dynamic_broadcast_in_dim`` + ``get_dimension_size``,
    so one compiled executable serves every concrete batch size."""

    return _dot_fn, (
        etl.TensorSpec((etl.dim("B"), 512, 768), etl.float32),
        etl.TensorSpec((768, 2304), etl.float32),
    )


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
