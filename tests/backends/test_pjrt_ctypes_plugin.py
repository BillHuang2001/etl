"""PJRT ctypes layer tests driven by a FAKE PJRT plugin (built at test time).

The xla adapter (``etl/backends/adapters/xla.py``) drives a USER-PROVIDED
PJRT C API plugin (``.so`` exporting ``GetPjRtApi``) through pure-stdlib
ctypes (``etl/backends/adapters/_pjrt_c_api.py`` + ``xla_util.py``). A real
XLA plugin is not pip-installable and is absent from CI, so this file builds
a small FAKE plugin (``_fake_pjrt_plugin.c``, gcc) that satisfies the same
ABI contract and exercises the whole ctypes driver + backend plumbing for
real:

* ABI/version gate: ``verify_api`` accepts the fake (struct_size == 1144,
  API major 0) and rejects deliberately broken builds (wrong struct_size,
  wrong major version) with ``etl.BackendError``.
* Full driver plumbing through the real backend: ``XlaBackend``
  lower/compile/load/run against the fake, plus artifact save/load
  round-trips. The fake parses the entry function's result types from the
  StableHLO MLIR text and returns ZERO-FILLED buffers of the declared
  shapes/dtypes — so shape/dtype/plumbing are asserted, numerical parity is
  intentionally out of scope (the fake performs no computation).
* Error reporting: an env-var-injected plugin failure raises
  ``etl.BackendError`` carrying the plugin's message text.

The fake is built at collection-adjacent runtime: when no C compiler is
available (or the build fails) the module skips with ``pytest.skip`` —
exactly like the jaxlib-gated adapter tests skip without their dependency.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import etl
from etl.backends.adapters import _pjrt_c_api as pjrt
from etl.backends.adapters import xla
from etl.backends.adapters import xla_util

from tests.backends import _adapter_utils as u

FAKE_C = Path(__file__).parent / "_fake_pjrt_plugin.c"

NAME = "xla"


# ---------------------------------------------------------------------------
# fake plugin build (gcc at test time; skip when unavailable)
# ---------------------------------------------------------------------------


def _build_plugin(tmp_path, so_name, defines=()):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler (gcc/cc) available — cannot build the "
                    "fake PJRT plugin")
    cmd = [cc, "-shared", "-fPIC", "-std=c99", "-O1", str(FAKE_C)]
    cmd += [f"-D{define}" for define in defines]
    cmd += ["-o", str(tmp_path / so_name)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"building the fake PJRT plugin failed: {proc.stderr}")
    return tmp_path / so_name


@pytest.fixture(scope="module")
def fake_plugin(tmp_path_factory):
    """The standard fake plugin (ABI-valid; zero-filled outputs)."""
    return _build_plugin(tmp_path_factory.mktemp("fake_pjrt_plugin"),
                         "fake_pjrt_plugin.so")


@pytest.fixture(scope="module")
def wrong_size_plugin(tmp_path_factory):
    """A fake whose PJRT_Api.struct_size (600) fails the ABI coverage gate."""
    return _build_plugin(
        tmp_path_factory.mktemp("fake_pjrt_plugin_wrong_size"),
        "fake_pjrt_plugin_wrong_size.so",
        defines=("ETL_FAKE_PJRT_STRUCT_SIZE=600",),
    )


@pytest.fixture(scope="module")
def wrong_major_plugin(tmp_path_factory):
    """A fake compiled against PJRT API major 1 (adapter requires major 0)."""
    return _build_plugin(
        tmp_path_factory.mktemp("fake_pjrt_plugin_wrong_major"),
        "fake_pjrt_plugin_wrong_major.so",
        defines=("ETL_FAKE_PJRT_MAJOR_VERSION=1",),
    )


# ---------------------------------------------------------------------------
# (a) ABI / version gate
# ---------------------------------------------------------------------------


def test_version_gate_passes_against_fake(fake_plugin):
    """The ctypes loader accepts the fake and the driver round-trips it."""
    library, api = pjrt.load_plugin_library(str(fake_plugin))
    assert api.struct_size == ctypes.sizeof(pjrt.PJRT_Api) == 1144
    assert api.pjrt_api_version.major_version == pjrt.PJRT_API_MAJOR
    assert api.pjrt_api_version.minor_version == pjrt.PJRT_API_MINOR
    # The driver-level probe: discovery (options path) + GetPjRtApi +
    # one-time initialize + a live client create/destroy round-trip.
    plugin = xla_util._load_plugin({"plugin_path": str(fake_plugin)})
    assert plugin.api.PJRT_Client_Compile is not None
    client = plugin.create_client()
    name, version = client.platform_info()
    assert name == "fake"
    assert version == "fake-cpu 0.0.1"
    assert len(client.addressable_devices()) == 1
    client.close()
    assert library is not None  # (kept alive by the plugin wrapper)


def test_version_gate_rejects_wrong_struct_size(wrong_size_plugin):
    """A too-small PJRT_Api struct is rejected naming struct_size."""
    with pytest.raises(etl.BackendError, match="struct_size is 600"):
        xla_util._load_plugin({"plugin_path": str(wrong_size_plugin)})


def test_version_gate_rejects_wrong_major(wrong_major_plugin):
    """A plugin compiled against another ABI family is rejected explicitly."""
    with pytest.raises(etl.BackendError, match="ABI mismatch"):
        xla_util._load_plugin({"plugin_path": str(wrong_major_plugin)})


# ---------------------------------------------------------------------------
# (b) full driver plumbing through the real backend
# ---------------------------------------------------------------------------


def _compile_fake(fn, specs, fake_plugin):
    """trace -> lower(backend="xla") -> compile with plugin_path=..."""
    graph = etl.trace(fn, *specs)
    lowered = etl.lower(graph, backend=NAME)
    return etl.compile(lowered, plugin_path=str(fake_plugin))


def test_compile_load_run_and_save_load_roundtrip(fake_plugin, tmp_path, monkeypatch):
    """XlaBackend compile/load/run + artifact save/load against the fake."""
    # Load-path plugin discovery goes through the ETL_PJRT_PLUGIN env var
    # (compile above uses the explicit plugin_path option).
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_plugin))

    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, fake_plugin)
    assert artifact.backend == NAME
    payload = artifact.payload
    assert payload["format"] == "xla-serialized-executable"
    assert isinstance(payload["executable_base64"], str)

    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)  # numpy reference (shape/dtype only)

    # Save / load round-trip at pipeline level (deserialize path).
    path = tmp_path / "artifact.etl"
    artifact.save(path)
    loaded_artifact = etl.backends.CompiledArtifact.load(path)
    exe = etl.load(loaded_artifact)
    assert "main" in exe.functions
    out = etl.run(exe, a, b)
    assert isinstance(out, etl.Tensor)
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    # The fake performs no computation: outputs are zero-filled.
    np.testing.assert_array_equal(out.numpy(), np.zeros_like(expected.numpy()))

    # Backend-level raw reload + run over flat etl.Tensor inputs.
    backend = etl.backends.get(NAME)
    raw = backend.load(artifact)
    (raw_out,) = raw.run([etl.tensor(a), etl.tensor(b)])
    np.testing.assert_array_equal(raw_out.numpy(), np.zeros_like(expected.numpy()))


def test_multi_dim_output_against_fake(fake_plugin, monkeypatch):
    """A 4-D output exercises the MLIR result-type parse beyond scalars."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_plugin))
    fn, specs = u.reshape_broadcast_transpose()
    x = u.standard_normal((4, 8))
    artifact = _compile_fake(fn, specs, fake_plugin)
    exe = etl.load(artifact)
    expected = etl.evaluate(fn, x)
    out = etl.run(exe, x)
    assert out.dtype == expected.dtype
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.numpy(), np.zeros_like(expected.numpy()))


def test_multi_output_graph_against_fake(fake_plugin, monkeypatch):
    """A tuple-returning graph exercises the parenthesized result list."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_plugin))

    def fn(x):
        return etl.add(x, x), etl.multiply(x, 2.0)

    specs = (etl.TensorSpec((2, 3), etl.float32),)
    artifact = _compile_fake(fn, specs, fake_plugin)
    exe = etl.load(artifact)
    x = u.standard_normal((2, 3))
    expected = etl.evaluate(fn, x)
    outs = etl.run(exe, x)
    assert isinstance(outs, tuple) and len(outs) == 2
    for got, want in zip(outs, expected):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        np.testing.assert_array_equal(got.numpy(), np.zeros_like(want.numpy()))


# ---------------------------------------------------------------------------
# (c) injected plugin errors -> core.BackendError with the plugin's message
# ---------------------------------------------------------------------------


def test_injected_compile_error_raises_backend_error(fake_plugin, monkeypatch):
    """A failing PJRT_Client_Compile surfaces as BackendError with the message."""
    monkeypatch.setenv("ETL_FAKE_PJRT_FAIL_STEP", "PJRT_Client_Compile")
    monkeypatch.setenv("ETL_FAKE_PJRT_FAIL_MESSAGE", "fake compile failure")
    fn, specs = u.matmul_relu_sum()
    lowered = etl.lower(etl.trace(fn, *specs), backend=NAME)
    with pytest.raises(etl.BackendError, match="fake compile failure"):
        etl.compile(lowered, plugin_path=str(fake_plugin))


def test_injected_execute_error_raises_backend_error(fake_plugin, monkeypatch):
    """A failing PJRT_LoadedExecutable_Execute surfaces at run time."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_plugin))
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, fake_plugin)
    exe = etl.load(artifact)
    monkeypatch.setenv("ETL_FAKE_PJRT_FAIL_STEP", "PJRT_LoadedExecutable_Execute")
    monkeypatch.setenv("ETL_FAKE_PJRT_FAIL_MESSAGE", "fake execute failure")
    a, b = u.matmul_relu_sum_args()
    with pytest.raises(etl.BackendError, match="fake execute failure"):
        etl.run(exe, a, b)
