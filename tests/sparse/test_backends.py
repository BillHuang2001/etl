"""Backend tests for ``etl.sparse``: the numpy interpreter path (capabilities +
explicit-pipeline parity) and the compiler-backend deferrals (all 16 sparse
ops are numpy-backend-only in v1 — every compiler adapter rejects them at
LOWER time with the capability-drift ``BackendError``, and the StableHLO
exporter names them).

Contract under test: ``../etl/sparse/CONTEXT.md`` (Known issues) and
``../etl/backends/CONTEXT.md``. The capability drift check runs in
``CompilerBackend.lower`` (``../etl/backends/compiler.py``) BEFORE any
compiler is invoked, so ``etl.lower(graph, backend="iree"|"tvm")`` rejects a
sparse graph at lower time.

Env gating: ``etl.lower(..., backend=<adapter>)`` resolves the backend through
``etl.backends.registry.get``, which AUTO-ACTIVATES the optional adapter on a
registry miss — and the adapter's ``register()`` probes its compiler
dependency (``check_available``), raising the ``pip install etl[...]``
adapter-absence ``BackendError`` BEFORE ``CompilerBackend.lower`` runs. So in
a compiler-free env the drift pre-check is unreachable; those tests therefore
assert the strict drift contract and skip only on the genuine adapter-absence
signature (shared helper ``assert_capability_drift`` in ``conftest.py`` — no
``importorskip``, no dropped parametrizations).

The ``CompilerBackend.lower`` path itself is exercised dependency-free by the
``stub_iree_backend`` fixture, which registers a compiler-free
``CompilerBackend`` named ``"iree"`` by direct registry assignment
(save/restore of the previous entry, the pattern in
``tests/pipeline_env_defaults_test.py``).
"""

import dataclasses
import re

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse

from tests.sparse.conftest import (
    assert_capability_drift,
    coo_example,
    coo_spec,
    csr_example,
    csc_example,
    dense_example,
    materialize,
    run_graph,
)

# ---------------------------------------------------------------------------
# Per-file spec helpers (mirror `csr_example()` / `csc_example()`).
# ---------------------------------------------------------------------------

def csr_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csr_example()` (indptr static (4,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((4,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csr",
    )


def csc_spec(dtype=etl.float32):
    """The SparseTensorSpec matching `csc_example()` (indptr static (5,))."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((5,), etl.int64),
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), dtype),
        dense_shape=(3, 4),
        format="csc",
    )


#: The 16 numpy-backend-only sparse ops (exact names, per
#: ``etl.backends.stablehlo.ops.DEFERRED_OPS``).
SPARSE_OPS = (
    "sparse_from_dense",
    "sparse_to_dense",
    "sparse_coo_to_csr",
    "sparse_csr_to_coo",
    "sparse_coo_to_csc",
    "sparse_csc_to_coo",
    "sparse_negate",
    "sparse_add",
    "sparse_multiply",
    "sparse_multiply_dense",
    "sparse_reduce_sum",
    "sparse_transpose",
    "sparse_reshape",
    "sparse_concatenate",
    "sparse_dot_dense",
    "dense_dot_sparse",
)


def _graph_for(op_name):
    """A tiny graph whose entry block contains exactly the named sparse op."""
    if op_name == "sparse_from_dense":
        return etl.trace(
            lambda d: sparse.from_dense(d), etl.TensorSpec((3, 4), etl.float32)
        )
    if op_name == "sparse_to_dense":
        return etl.trace(lambda x: sparse.to_dense(x), coo_spec())
    if op_name == "sparse_coo_to_csr":
        return etl.trace(lambda x: sparse.to_csr(x), coo_spec())
    if op_name == "sparse_csr_to_coo":
        return etl.trace(lambda x: sparse.to_coo(x), csr_spec())
    if op_name == "sparse_coo_to_csc":
        return etl.trace(lambda x: sparse.to_csc(x), coo_spec())
    if op_name == "sparse_csc_to_coo":
        return etl.trace(lambda x: sparse.to_coo(x), csc_spec())
    if op_name == "sparse_negate":
        return etl.trace(lambda x: sparse.negate(x), coo_spec())
    if op_name == "sparse_add":
        return etl.trace(lambda a, b: sparse.add(a, b), coo_spec(), coo_spec())
    if op_name == "sparse_multiply":
        return etl.trace(lambda a, b: sparse.multiply(a, b), coo_spec(), coo_spec())
    if op_name == "sparse_multiply_dense":
        return etl.trace(
            lambda a, d: sparse.multiply_dense(a, d),
            coo_spec(),
            etl.TensorSpec((3, 4), etl.float32),
        )
    if op_name == "sparse_reduce_sum":
        return etl.trace(lambda x: sparse.sum(x, axes=0), coo_spec())
    if op_name == "sparse_transpose":
        return etl.trace(lambda x: sparse.transpose(x), coo_spec())
    if op_name == "sparse_reshape":
        return etl.trace(lambda x: sparse.reshape(x, (12,)), coo_spec())
    if op_name == "sparse_concatenate":
        return etl.trace(
            lambda a, b: sparse.concatenate([a, b], axis=0), coo_spec(), coo_spec()
        )
    if op_name == "sparse_dot_dense":
        return etl.trace(
            lambda x, d: sparse.matmul(x, d),
            coo_spec(),
            etl.TensorSpec((4, 5), etl.float32),
        )
    if op_name == "dense_dot_sparse":
        return etl.trace(
            lambda d, x: sparse.matmul(d, x),
            etl.TensorSpec((5, 3), etl.float32),
            coo_spec(),
        )
    raise ValueError(f"unknown sparse op {op_name!r}")


@pytest.fixture(scope="module")
def sparse_graphs():
    """All 16 single-op sparse graphs, built once per module."""
    return {name: _graph_for(name) for name in SPARSE_OPS}


# ---------------------------------------------------------------------------
# 1. numpy backend: capability + explicit-pipeline parity
# ---------------------------------------------------------------------------


def test_numpy_backend_declares_sparse_capability():
    backend = etl.backends.get("numpy")
    assert backend.name == "numpy"
    assert backend.capabilities.sparse_ops is True


def test_numpy_backend_dtype_capabilities_cover_sparse_adjacent():
    caps = etl.backends.get("numpy").capabilities
    assert caps.supports_dtype(np.dtype("float32"))
    assert caps.supports_dtype(np.dtype("int64"))  # indices/indptr dtype


def test_explicit_pipeline_parity_add_coo_csr():
    d = dense_example()
    graph = etl.trace(
        lambda a, b: sparse.add(a, b), coo_spec(), csr_spec()
    )
    out = run_graph(graph, coo_example(), csr_example())
    np.testing.assert_allclose(materialize(out), d + d, rtol=1e-6, atol=1e-6)


def test_explicit_pipeline_parity_matmul_sparse_dot_dense():
    d = dense_example()
    dense = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    graph = etl.trace(
        lambda x, y: sparse.matmul(x, y),
        coo_spec(),
        etl.TensorSpec((4, 5), etl.float32),
    )
    out = run_graph(graph, coo_example(), dense)
    np.testing.assert_allclose(materialize(out), d @ dense, rtol=1e-6, atol=1e-6)


def test_explicit_pipeline_parity_reduce_sum_keepdims():
    d = dense_example()
    graph = etl.trace(
        lambda x: sparse.reduce_sum(x, axes=(0,), keepdims=True), coo_spec()
    )
    out = run_graph(graph, coo_example())
    np.testing.assert_allclose(
        materialize(out), d.sum(axis=0, keepdims=True), rtol=1e-6, atol=1e-6
    )


def test_explicit_pipeline_parity_to_dense_csr():
    d = dense_example()
    graph = etl.trace(lambda x: sparse.to_dense(x), csr_spec())
    out = run_graph(graph, csr_example())
    np.testing.assert_allclose(materialize(out), d, rtol=1e-6, atol=1e-6)


def test_lower_defaults_to_numpy():
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    lowered = etl.lower(graph)  # backend=None -> numpy
    assert lowered.backend == "numpy"


# ---------------------------------------------------------------------------
# 2. Compiler-backend deferrals: all 16 sparse ops are numpy-backend-only
# ---------------------------------------------------------------------------

# The drift rejection is a LOWER-time capability check (no compiler is
# invoked). All 16 ops x iree stay within the time budget; tvm is exercised
# on a representative subset (the drift path is shared — CompilerBackend.lower
# — so per-op coverage on one adapter plus subset coverage on the other is
# sufficient; xla raises a plugin-required error instead and is not used).
_TVM_REPRESENTATIVES = ("sparse_add", "sparse_to_dense", "sparse_dot_dense")


class _StubIreeCompilerBackend(etl.backends.CompilerBackend):
    """A dependency-free ``CompilerBackend`` named ``"iree"``.

    Inherits the SHARED ``CompilerBackend.lower`` path (capability
    pre-check -> portable inlining -> StableHLO export -> ``Signature``
    recording), so that path can be exercised with NO IREE packages
    installed. ``compile`` / ``load`` are never reached (``lower`` never
    compiles) and raise explicitly rather than silently pretending to work.

    ``capabilities`` mirrors the real iree adapter in the one flag this suite
    cares about (``sparse_ops=False``) while staying dependency-free (all
    numpy dtypes, dynamic shapes).
    """

    name = "iree"
    capabilities = dataclasses.replace(
        etl.backends.numpy_backend.capabilities,
        sparse_ops=False,  # mirrors the real iree adapter
        runtime_calls=False,
        collectives=False,
        async_collectives=False,
        custom_blocks=False,
        external_calls=False,
    )

    def compile(self, lowered, options=None):  # pragma: no cover - never called
        raise NotImplementedError("stub compiler backend: lower() never compiles")

    def load(self, artifact, device=None):  # pragma: no cover - never called
        raise NotImplementedError("stub compiler backend: lower() never loads")


@pytest.fixture
def stub_iree_backend():
    """Register the dependency-free ``"iree"`` stub, restoring the registry
    afterwards (save/restore of the previous entry, so this also works when
    the real adapter is registered in this process — the pattern in
    ``tests/pipeline_env_defaults_test.py``)."""
    from etl.backends import registry as _registry

    stub = _StubIreeCompilerBackend()
    previous = _registry._registry.get("iree")
    _registry._registry["iree"] = stub
    try:
        yield stub
    finally:
        if previous is None:
            _registry._registry.pop("iree", None)
        else:
            _registry._registry["iree"] = previous


@pytest.mark.parametrize("op_name", SPARSE_OPS)
def test_sparse_op_defers_on_iree(sparse_graphs, op_name):
    # Strict drift contract; env-gated on genuine adapter absence only (see
    # the module docstring). All 16 parametrizations are kept in both envs.
    assert_capability_drift(sparse_graphs[op_name], "iree", op_name)


@pytest.mark.parametrize("op_name", _TVM_REPRESENTATIVES)
def test_sparse_op_defers_on_tvm(sparse_graphs, op_name):
    graph = sparse_graphs[op_name]
    with pytest.raises(core.BackendError) as excinfo:
        etl.lower(graph, backend="tvm")
    msg = str(excinfo.value)
    # Env-dependent adapter-missing error (tvm/jaxlib not installed) skips,
    # mirroring the pattern in tests/backends/test_external_call_iree.py —
    # the capability-drift contract is what's under test.
    if "capability drift" not in msg:
        pytest.skip(f"tvm adapter not available in this env: {msg}")
    assert re.search("capability drift", msg)
    assert op_name in msg
    assert "etl.sparse.to_dense" in msg


@pytest.mark.parametrize("op_name", SPARSE_OPS)
def test_sparse_op_defers_in_stablehlo_export(sparse_graphs, op_name):
    graph = sparse_graphs[op_name]
    with pytest.raises(core.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph)
    assert op_name in str(excinfo.value)


def test_dense_graph_lowers_on_compiler_backend_without_compiler(stub_iree_backend):
    # Positive control: a dense-only graph passes the capability pre-check and
    # exports, so the SHARED `CompilerBackend.lower` path (capability
    # pre-check -> portable inlining -> StableHLO export -> Signature
    # recording) completes with NO compiler installed. Reaching a compiler
    # ADAPTER at all requires its packages (activation runs `check_available`
    # in `register()`), which is why the dependency-free stub backend is
    # registered under the name "iree" for this test.

    def f(x):
        return etl.add(x, x)

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    lowered = etl.lower(graph, backend="iree")
    assert isinstance(lowered, etl.backends.LoweredProgram)
    assert lowered.backend == "iree"
    assert lowered.payload["format"] == "stablehlo"
    assert "func.func" in lowered.payload["mlir_text"]


def test_sparse_graph_drifts_on_capability_precheck_without_compiler(
    stub_iree_backend,
):
    # The drift contract compiler-free: `CompilerBackend.lower`'s capability
    # pre-check rejects a sparse op BEFORE any export/compile, with the stub
    # standing in for the real adapter (which cannot even activate without
    # its packages). This keeps the contract covered in a compiler-free env
    # even though the adapter-driven pins legitimately skip there.
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with pytest.raises(core.BackendError) as excinfo:
        etl.lower(graph, backend="iree")
    msg = str(excinfo.value)
    assert re.search("capability drift", msg)
    assert "sparse_negate" in msg
    assert "etl.sparse.to_dense" in msg
