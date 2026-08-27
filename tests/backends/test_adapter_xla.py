"""XLA adapter contract tests (``etl/backends/adapters/xla.py``).

Contract under test: the shared ``CompilerBackend`` framework
(``etl/backends/compiler.py``) plus the XLA adapter's REAL
lower/compile/load/run through a USER-PROVIDED PJRT C API plugin (a ``.so``
exporting ``GetPjRtApi``, configured via the ``ETL_PJRT_PLUGIN`` environment
variable or the ``plugin_path`` compile option; the adapter drives it via
ctypes — NOT the ``jax`` package). One contract, three dependencies: this
file, ``test_adapter_iree.py`` and ``test_adapter_tvm.py`` are structurally
identical and share the graph builders, tolerances and parity helper in
``tests/backends/_adapter_utils``.

The plugin is a real external artifact (build one from OpenXLA with ``bazel
build //xla/pjrt/c:pjrt_c_api_cpu_plugin``; there is no pip-installable
plugin). When none is configured the module SKIPS at import time with an
actionable message — the same contract is exercised without a real XLA
build by ``test_pjrt_ctypes_plugin.py`` against a fake plugin built from C
at test time. Set ``ETL_PJRT_PLUGIN`` (or pass ``plugin_path=...``) to run
these tests for real.

Pinned here:

* registration + lazy name resolution: ``xla.register()`` installs the
  singleton ``xla_backend`` (name ``"xla"``) and every pipeline entry point
  accepting a backend name string (``etl.lower`` / ``etl.build`` /
  ``etl.evaluate``, resolved through ``etl.backends.get``) reaches the same
  instance.
* numerical parity vs the default numpy backend (matmul+relu+sum and
  reshape/broadcast/transpose) at the cross-compiler fp32 tolerance.
* symbolic dims iff ``capabilities.dynamic_shapes`` — otherwise an explicit
  ``etl.BackendError`` NAMING the feature (never a silent fallback).
* artifact save/load round-trips at pipeline level (``etl.load``) and raw
  backend level (``backend.load(...).run([flat tensors])``); load-time
  backend-name mismatch raises ``etl.PersistenceError`` — never silently
  recompile.
* capability rejections (``runtime_call`` / collectives when undeclared)
  raise ``etl.BackendError`` NAMING the feature.
* the lowered payload is the shared StableHLO contract: ``format ==
  "stablehlo"``, ``format_version == 1``, MLIR text, entry functions.

XLA compiles for real (seconds per compile), so the 2s-per-file convention
does not apply here; the module keeps the compile count low via one shared
module-scoped artifact fixture plus one compile per distinct graph.
"""

import numpy as np
import pytest

import etl
from etl.backends.adapters import xla

from tests.backends import _adapter_utils as u

NAME = "xla"

# Real-plugin presence gate (module level): the adapter requires a
# user-provided PJRT C API plugin (.so exporting GetPjRtApi) — there is no
# pip-installable dependency. ``register()`` runs the adapter's own probe
# (bindings integrity + plugin discovery + ABI version gate + a live client
# create/destroy round-trip) and raises ``etl.BackendError`` when no plugin
# is available; skip the whole module in that case with an actionable hint.
try:
    xla.register()
except etl.BackendError as exc:
    pytest.skip(
        "the xla adapter requires a user-provided PJRT C API plugin (.so "
        "exporting GetPjRtApi); set ETL_PJRT_PLUGIN=/path/to/"
        "pjrt_c_api_cpu_plugin.so or pass plugin_path=... to the compile "
        f"options. Details: {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# module fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter_backend():
    """The registered singleton backend (``xla_backend``)."""
    xla.register()
    backend = etl.backends.get(NAME)
    assert backend.name == NAME
    return backend


@pytest.fixture(scope="module")
def adapter_artifact(adapter_backend):
    """One shared compiled matmul+relu+sum artifact for the whole module."""
    fn, specs = u.matmul_relu_sum()
    _, lowered, artifact = u.stage(NAME, fn, specs)
    assert lowered.backend == NAME
    assert artifact.backend == NAME
    return artifact


# ---------------------------------------------------------------------------
# 1. registration / singleton
# ---------------------------------------------------------------------------


def test_registered_singleton(adapter_backend):
    assert adapter_backend is xla.xla_backend
    assert etl.backends.get(NAME) is xla.xla_backend
    assert isinstance(adapter_backend.capabilities, etl.backends.Capabilities)


# ---------------------------------------------------------------------------
# 2. full pipeline + numerical parity
# ---------------------------------------------------------------------------


def test_matmul_relu_sum_parity(adapter_artifact):
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    exe = etl.load(adapter_artifact)
    u.assert_parity(fn, (a, b), exe)


def test_reshape_broadcast_transpose_parity_via_build_name(adapter_backend):
    fn, specs = u.reshape_broadcast_transpose()
    x = u.standard_normal((4, 8))
    exe = etl.build(fn, *specs, backend=NAME)
    u.assert_parity(fn, (x,), exe)


def test_evaluate_with_backend_name(adapter_backend):
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)  # default numpy reference
    actual = etl.evaluate(fn, a, b, backend=NAME)
    assert isinstance(actual, etl.Tensor)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(
        actual.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
    )


# ---------------------------------------------------------------------------
# 3. symbolic dims
# ---------------------------------------------------------------------------


def test_symbolic_dims(adapter_backend):
    fn, specs = u.symbolic_scale()
    graph = etl.trace(fn, *specs)
    graph.verify()
    if adapter_backend.capabilities.dynamic_shapes is False:
        # Capability rejection must name the feature — no silent fallback.
        with pytest.raises(etl.BackendError) as excinfo:
            etl.lower(graph, backend=NAME)
        assert "dynamic" in str(excinfo.value).lower()
        return

    numpy_exe = etl.build(fn, *specs)  # default numpy reference
    adapter_exe = etl.build(fn, *specs, backend=NAME)
    for x in (
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    ):
        expected = etl.run(numpy_exe, x)
        actual = etl.run(adapter_exe, x)
        assert actual.dtype == expected.dtype
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
        )


# ---------------------------------------------------------------------------
# 4. payload contract + name resolution
# ---------------------------------------------------------------------------


def test_lower_name_resolution_and_payload_contract(adapter_backend):
    fn, specs = u.matmul_relu_sum()
    graph = etl.trace(fn, *specs)
    graph.verify()
    lowered = etl.lower(graph, backend=NAME)
    assert lowered.backend == NAME
    payload = lowered.payload
    assert payload["format"] == "stablehlo"
    assert payload["format_version"] == 1
    mlir = payload["mlir_text"]
    assert isinstance(mlir, str) and mlir
    assert "module {" in mlir
    assert "func.func" in mlir
    assert "stablehlo." in mlir
    entry = payload["entry_functions"]
    assert isinstance(entry, list)
    assert "main" in entry
    # Direct backend invocation agrees with string-name resolution.
    assert adapter_backend.lower(graph).payload == payload


# ---------------------------------------------------------------------------
# 5. artifact persistence
# ---------------------------------------------------------------------------


def test_artifact_save_load_roundtrip(tmp_path, adapter_backend, adapter_artifact):
    path = tmp_path / "artifact.etl"
    adapter_artifact.save(path)
    loaded = etl.backends.CompiledArtifact.load(path)
    assert loaded.backend == NAME

    # Backend-level reload: raw run over flat etl.Tensor inputs.
    raw = adapter_backend.load(loaded)
    fn, specs = u.matmul_relu_sum()
    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)
    (out,) = raw.run([etl.tensor(a), etl.tensor(b)])
    np.testing.assert_allclose(
        out.numpy(), expected.numpy(), rtol=u.FP32_RTOL, atol=u.FP32_ATOL
    )

    # Pipeline-level reload runs through the structured signature.
    exe = etl.load(loaded)
    u.assert_parity(fn, (a, b), exe)


def test_backend_name_mismatch_on_load(adapter_backend, adapter_artifact):
    # An adapter artifact can never be loaded by another backend: "numpy"
    # stands in for any mismatching registered backend (always present,
    # order-independent).
    with pytest.raises(etl.PersistenceError, match="never silently recompile"):
        etl.load(adapter_artifact, backend="numpy")

    # The reverse direction: an adapter never loads another backend's
    # artifact.

    def fn(x):
        return etl.add(x, x)

    numpy_artifact = etl.compile(
        etl.lower(etl.trace(fn, etl.TensorSpec((2, 3), etl.float32)))
    )
    with pytest.raises(etl.PersistenceError):
        adapter_backend.load(numpy_artifact)


# ---------------------------------------------------------------------------
# 6. executable attributes
# ---------------------------------------------------------------------------


def test_executable_attributes(adapter_artifact):
    exe = etl.load(adapter_artifact)
    assert isinstance(exe.functions, tuple)
    assert "main" in exe.functions
    assert exe.device is None or exe.device == etl.Device("cpu", 0)


# ---------------------------------------------------------------------------
# 7. capability rejections
# ---------------------------------------------------------------------------


def test_unsupported_features_rejected_naming_feature(adapter_backend):
    cases = []
    if adapter_backend.capabilities.runtime_calls is False:
        cases.append(("runtime_call", u.runtime_call_graph()))
    if adapter_backend.capabilities.collectives is False:
        cases.append(("collective", u.collective_graph()))
    if not cases:
        pytest.skip("adapter declares runtime_calls and collectives — nothing to reject")

    for feature, graph in cases:
        graph.verify()  # negative control: valid IR — rejection is capability-only
        with pytest.raises(etl.BackendError) as excinfo:
            adapter_backend.lower(graph)
        assert feature in str(excinfo.value).lower()
