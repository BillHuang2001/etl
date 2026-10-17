"""Env-var default backend/device resolution tests (ETL_BACKEND / ETL_DEVICE /
ETL_TARGET_BACKENDS).

Validates the process-wide defaulting added to ``etl.build``/``etl.evaluate``
(``etl.pipeline._resolve_backend_device``): explicit kwargs always win;
unset ``backend``/``device`` resolve from ``ETL_BACKEND`` / ``ETL_DEVICE``
(read lazily at call time); unset ``target_backends`` resolves from
``ETL_TARGET_BACKENDS`` or, for an iree-family backend, is inferred from the
resolved device (``cuda`` -> ``["cuda"]``, ``rocm``/``hip`` -> ``["rocm"]``,
``cpu`` -> ``["llvm-cpu"]``).

The env variables are read lazily at call time, so every test drives them
with ``monkeypatch`` after ``import etl`` — no module-level env snapshot.
No test imports (or triggers the import of) a real compiler adapter: the
iree-family paths use stub ``Backend`` instances (helper-level) or a
functional stub registered under the name ``"iree"`` (full-path), and
``_assert_no_new_iree_import()`` pins it — the guard's baseline is captured
PER TEST by an autouse fixture (never at module import), so an adapter import
that another module of the same process legitimately performs can never make
the guard fire, while any import triggered by the test body itself fails it.

The contract under test is ``etl/CONTEXT.md`` ("Staging & pipeline") and
``etl/pipeline.py`` docstrings; the resolution implementation lives in
``etl/pipeline.py`` (``_resolve_backend_device`` / ``_parse_env_device``).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import etl
from etl import core
from etl.pipeline import _parse_env_device, _resolve_backend_device

#: Did the real iree adapter already sit in ``sys.modules`` when the CURRENT
#: test started? Refreshed by the autouse fixture below so the guard is
#: order-independent: another module collected/run in the same process may
#: legitimately have imported the adapter already.
_iree_imported_at_test_start = "etl.backends.adapters.iree" in sys.modules


@pytest.fixture(autouse=True)
def _snapshot_adapter_import_state():
    global _iree_imported_at_test_start
    _iree_imported_at_test_start = "etl.backends.adapters.iree" in sys.modules
    yield


def _assert_no_new_iree_import():
    """This test must not be the one that imports the real iree adapter."""
    assert (
        "etl.backends.adapters.iree" in sys.modules
    ) == _iree_imported_at_test_start


# ---------------------------------------------------------------------------
# Shared graph definition
# ---------------------------------------------------------------------------


@etl.defn
def _linear(x, w, b):
    """A small linear layer: dot + bias + relu (three positional inputs)."""
    return etl.relu(etl.add(etl.dot(x, w), b))


_LINEAR_SPECS = (
    etl.TensorSpec((2, 3), etl.float32, name="x"),
    etl.TensorSpec((3, 4), etl.float32, name="w"),
    etl.TensorSpec((4,), etl.float32, name="b"),
)


def _linear_inputs():
    rng = np.random.default_rng(0)
    return (
        rng.random((2, 3)).astype("float32"),
        rng.random((3, 4)).astype("float32"),
        rng.random((4,)).astype("float32"),
    )


def _ref(x, w, b):
    return np.maximum(0.0, np.asarray(x) @ np.asarray(w) + np.asarray(b))


# ---------------------------------------------------------------------------
# Stub backends (no real adapter imports)
# ---------------------------------------------------------------------------


class _IreeStub(etl.backends.Backend):
    """Minimal Backend instance named ``"iree"`` for helper-level tests."""

    name = "iree"
    capabilities = None

    def lower(self, graph, options=None):  # pragma: no cover - never called
        raise NotImplementedError

    def compile(self, lowered, options=None):  # pragma: no cover - never called
        raise NotImplementedError

    def load(self, artifact, device=None):  # pragma: no cover - never called
        raise NotImplementedError


class _RecordingIreeStub(etl.backends.Backend):
    """Functional ``"iree"`` stub for full-path tests: delegates everything to
    the numpy backend but stamps its own name into the staged objects, and
    records the compile options it receives.

    ``load`` keeps the requested device on a minimal fake backend executable
    (asserted via ``Executable.device``); for cpu devices it delegates to the
    numpy backend so the executable is actually runnable.
    """

    name = "iree"
    capabilities = etl.backends.numpy_backend.capabilities
    recorded_compile_options: list = []

    def lower(self, graph, options=None):
        lp = etl.backends.numpy_backend.lower(graph, options)
        return etl.backends.LoweredProgram(
            backend=self.name, signature=lp.signature, payload=lp.payload
        )

    def compile(self, lowered, options=None):
        _RecordingIreeStub.recorded_compile_options.append(dict(options or {}))
        art = etl.backends.numpy_backend.compile(
            etl.backends.LoweredProgram(
                backend="numpy", signature=lowered.signature, payload=lowered.payload
            ),
            options,
        )
        return etl.backends.CompiledArtifact(
            backend=self.name,
            signature=art.signature,
            target=art.target,
            payload=art.payload,
        )

    def load(self, artifact, device=None):
        if device is not None and device.kind != "cpu":
            class _FakeExecutable:
                functions = ("main",)

                def __init__(self, device):
                    self.device = device

                def run(self, flat):  # pragma: no cover - never called
                    raise NotImplementedError

            return _FakeExecutable(device)
        art = etl.backends.CompiledArtifact(
            backend="numpy",
            signature=artifact.signature,
            target=artifact.target,
            payload=artifact.payload,
        )
        return etl.backends.numpy_backend.load(art, device)


@pytest.fixture
def iree_stub_registered():
    """Register the functional ``"iree"`` stub for a full-path test and
    restore the registry afterwards.

    The stub is installed by direct registry assignment (save/restore of the
    previous entry) so these tests also run when a real ``"iree"`` backend is
    already registered in this process (e.g. the compiler extra installed):
    the stub replaces it for the duration of the test and is never visible to
    other tests. Tests run sequentially, so no cross-test interference.
    """
    from etl.backends import registry as _registry

    stub = _RecordingIreeStub()
    _RecordingIreeStub.recorded_compile_options = []
    previous = _registry._registry.get("iree")
    _registry._registry["iree"] = stub
    try:
        yield stub
    finally:
        if previous is None:
            _registry._registry.pop("iree", None)
        else:
            _registry._registry["iree"] = previous
        _RecordingIreeStub.recorded_compile_options = []


# ---------------------------------------------------------------------------
# Defaults unchanged without env
# ---------------------------------------------------------------------------


def test_no_env_defaults_unchanged(monkeypatch):
    """No env vars -> numpy backend, cpu device, options untouched."""
    for var in ("ETL_BACKEND", "ETL_DEVICE", "ETL_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)
    backend, device, options = _resolve_backend_device(None, None, {})
    assert backend is etl.backends.numpy_backend
    assert device == core.Device("cpu", 0)
    assert options == {}
    _assert_no_new_iree_import()


def test_no_env_build_evaluate_run(monkeypatch):
    """Full path with no env: build/evaluate produce a cpu numpy executable."""
    for var in ("ETL_BACKEND", "ETL_DEVICE", "ETL_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("cpu", 0)
    x, w, b = _linear_inputs()
    np.testing.assert_allclose(etl.run(exe, x, w, b).numpy(), _ref(x, w, b))
    np.testing.assert_allclose(
        etl.evaluate(_linear, x, w, b).numpy(), _ref(x, w, b)
    )
    _assert_no_new_iree_import()


# ---------------------------------------------------------------------------
# Env honored
# ---------------------------------------------------------------------------


def test_env_backend_honored(monkeypatch):
    """ETL_BACKEND selects the backend name (numpy resolves as usual)."""
    monkeypatch.setenv("ETL_BACKEND", "numpy")
    backend, device, options = _resolve_backend_device(None, None, {})
    assert backend.name == "numpy"
    assert backend is etl.backends.numpy_backend
    _assert_no_new_iree_import()


def test_env_backend_unknown_raises(monkeypatch):
    """An unknown ETL_BACKEND name raises the registry's explicit error."""
    monkeypatch.setenv("ETL_BACKEND", "no-such-backend")
    with pytest.raises(etl.BackendError, match="no-such-backend"):
        _resolve_backend_device(None, None, {})
    _assert_no_new_iree_import()


def test_env_backend_empty_treated_as_unset(monkeypatch):
    """An empty ETL_BACKEND is treated as unset (numpy default)."""
    monkeypatch.setenv("ETL_BACKEND", "")
    backend, device, options = _resolve_backend_device(None, None, {})
    assert backend is etl.backends.numpy_backend
    _assert_no_new_iree_import()


@pytest.mark.parametrize(
    ("value", "kind", "index"),
    [
        ("cuda:0", "cuda", 0),
        ("cuda:3", "cuda", 3),
        ("cuda", "cuda", 0),
        ("cpu", "cpu", 0),
        ("cpu:0", "cpu", 0),
        ("rocm:0", "rocm", 0),
        ("rocm:3", "rocm", 3),
        ("rocm", "rocm", 0),
        ("hip:2", "hip", 2),
    ],
)
def test_env_device_parsing(monkeypatch, value, kind, index):
    """ETL_DEVICE parses ``kind[:index]`` into a core.Device.

    Any kind string is accepted, so the ROCm spellings ``"rocm"`` and the
    equivalent ``"hip"`` alias parse too — the kind is preserved verbatim
    (``Device.kind`` is never normalized; only the target-backends inference
    below maps both to the IREE ``"rocm"`` target)."""
    monkeypatch.setenv("ETL_DEVICE", value)
    backend, device, options = _resolve_backend_device(None, None, {})
    assert device == core.Device(kind, index)
    assert backend is etl.backends.numpy_backend  # device routing is independent
    _assert_no_new_iree_import()


@pytest.mark.parametrize(
    "value",
    ["cuda:x", "cuda:-1", "a:b:c", ":", "cuda:", " : 0"],
)
def test_env_device_malformed_raises(monkeypatch, value):
    """A malformed ETL_DEVICE raises DeviceError naming the var and value."""
    monkeypatch.setenv("ETL_DEVICE", value)
    with pytest.raises(etl.DeviceError) as excinfo:
        _resolve_backend_device(None, None, {})
    message = str(excinfo.value)
    assert "ETL_DEVICE" in message and value in message
    assert "kind" in message and "index" in message  # documents the format
    _assert_no_new_iree_import()


def test_env_device_parse_directly():
    """The parser accepts explicit ``kind``/``kind:index`` spellings."""
    assert _parse_env_device("ETL_DEVICE", "cuda:0") == core.Device("cuda", 0)
    assert _parse_env_device("ETL_DEVICE", "cpu") == core.Device("cpu", 0)
    assert _parse_env_device("ETL_DEVICE", " cuda : 3 ") == core.Device("cuda", 3)


def test_env_device_parse_rocm_directly():
    """``_parse_env_device`` parses the ROCm spellings ``"rocm"`` / ``"hip"``
    (``"kind[:index]"``) to a ``core.Device`` with the kind PRESERVED verbatim
    — the parser does not normalize the ``"hip"`` alias to ``"rocm"`` (both
    are exactly what ``core.Device`` accepts; only the iree target-backends
    inference maps them to the ``"rocm"`` compile target)."""
    assert _parse_env_device("ETL_DEVICE", "rocm") == core.Device("rocm", 0)
    assert _parse_env_device("ETL_DEVICE", "rocm:0") == core.Device("rocm", 0)
    assert _parse_env_device("ETL_DEVICE", "rocm:3") == core.Device("rocm", 3)
    assert _parse_env_device("ETL_DEVICE", "hip") == core.Device("hip", 0)
    assert _parse_env_device("ETL_DEVICE", "hip:2") == core.Device("hip", 2)
    # The kind is the raw string — no aliasing at the parser level (and thus
    # `hip:2` is not Device("rocm", 2)).
    assert _parse_env_device("ETL_DEVICE", "hip:2").kind == "hip"
    assert _parse_env_device("ETL_DEVICE", "hip:2") != core.Device("rocm", 2)


def test_env_device_empty_treated_as_unset(monkeypatch):
    """An empty ETL_DEVICE is treated as unset (cpu default)."""
    monkeypatch.setenv("ETL_DEVICE", "")
    backend, device, options = _resolve_backend_device(None, None, {})
    assert device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_env_device_cpu_full_path(monkeypatch):
    """Full path: ETL_DEVICE=cpu routes build/evaluate to the cpu device."""
    monkeypatch.setenv("ETL_DEVICE", "cpu")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("cpu", 0)
    x, w, b = _linear_inputs()
    np.testing.assert_allclose(
        etl.evaluate(_linear, x, w, b).numpy(), _ref(x, w, b)
    )
    _assert_no_new_iree_import()


def test_env_device_cuda_numpy_backend_fails_explicitly(monkeypatch):
    """Full path: ETL_DEVICE=cuda with the (cpu-only) numpy backend raises the
    backend's explicit BackendError — never a silent fallback."""
    monkeypatch.setenv("ETL_DEVICE", "cuda:0")
    with pytest.raises(etl.BackendError, match="CPU devices only"):
        etl.build(_linear, *_LINEAR_SPECS)
    _assert_no_new_iree_import()


# ---------------------------------------------------------------------------
# Explicit kwargs always win
# ---------------------------------------------------------------------------


def test_explicit_backend_wins_over_env(monkeypatch):
    """An explicit backend kwarg is used even when ETL_BACKEND is set (and
    even when it names something unresolvable — the env is never read)."""
    monkeypatch.setenv("ETL_BACKEND", "definitely-not-a-backend")
    monkeypatch.setenv("ETL_DEVICE", "cuda:x")
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "bogus")
    backend, device, options = _resolve_backend_device(
        "numpy", core.Device("cpu", 0), {"target_backends": ["cuda"]}
    )
    assert backend.name == "numpy"
    assert device == core.Device("cpu", 0)
    assert options == {"target_backends": ["cuda"]}
    _assert_no_new_iree_import()


def test_explicit_kwargs_win_full_path(monkeypatch):
    """Full path: garbage env + explicit kwargs -> a working cpu executable."""
    monkeypatch.setenv("ETL_BACKEND", "definitely-not-a-backend")
    monkeypatch.setenv("ETL_DEVICE", "cuda:x")
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "bogus")
    exe = etl.build(
        _linear,
        *_LINEAR_SPECS,
        backend="numpy",
        device=core.Device("cpu", 0),
        target_backends=["cuda"],
    )
    assert exe.device == core.Device("cpu", 0)
    x, w, b = _linear_inputs()
    np.testing.assert_allclose(etl.run(exe, x, w, b).numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()


# ---------------------------------------------------------------------------
# target_backends inference and ETL_TARGET_BACKENDS
# ---------------------------------------------------------------------------


def test_iree_inference_cuda():
    """iree-family backend + cuda device -> target_backends=["cuda"]."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("cuda", 0), {}
    )
    assert options == {"target_backends": ["cuda"]}
    _assert_no_new_iree_import()


def test_iree_inference_cpu():
    """iree-family backend + cpu device -> target_backends=["llvm-cpu"]."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("cpu", 0), {}
    )
    assert options == {"target_backends": ["llvm-cpu"]}
    _assert_no_new_iree_import()


def test_iree_inference_rocm():
    """iree-family backend + rocm device -> target_backends == ["rocm"] (the
    IREE HIP/ROCm target), a list of one exact string."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("rocm", 0), {}
    )
    assert options == {"target_backends": ["rocm"]}
    # Exact shape pin: a plain list of one exact lowercase string.
    assert isinstance(options["target_backends"], list)
    assert options["target_backends"] == ["rocm"]
    _assert_no_new_iree_import()


def test_iree_inference_hip_alias_normalized_to_rocm():
    """iree-family backend + ``hip`` device (the ROCm alias) -> the SAME
    target_backends == ["rocm"]: the alias is normalized to the canonical
    IREE ``"rocm"`` target string, while the resolved device keeps kind
    ``"hip"`` (the inference does not rewrite the device)."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("hip", 0), {}
    )
    assert options == {"target_backends": ["rocm"]}
    assert device == core.Device("hip", 0)
    _assert_no_new_iree_import()


def test_iree_inference_rocm_index_irrelevant():
    """The inference depends only on the device KIND: a nonzero index still
    yields target_backends == ["rocm"] (the target list carries no index)."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("rocm", 5), {}
    )
    assert options == {"target_backends": ["rocm"]}
    _assert_no_new_iree_import()


def test_rocm_inference_only_for_iree_family():
    """Non-iree backend + rocm/hip device -> no target_backends key (the
    inference is iree-family only, exactly like the cuda/cpu cases)."""
    for kind in ("rocm", "hip"):
        backend, device, options = _resolve_backend_device(
            None, core.Device(kind, 0), {}
        )
        assert options == {}
    _assert_no_new_iree_import()


def test_numpy_backend_no_inference_for_cuda():
    """Non-iree backend (numpy) + cuda device -> no target_backends key."""
    backend, device, options = _resolve_backend_device(
        None, core.Device("cuda", 0), {}
    )
    assert options == {}
    _assert_no_new_iree_import()


def test_explicit_target_backends_wins_over_inference():
    """An explicit target_backends is preserved (no inference, no env)."""
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("cuda", 0), {"target_backends": ["llvm-cpu"]}
    )
    assert options == {"target_backends": ["llvm-cpu"]}
    _assert_no_new_iree_import()


def test_options_dict_not_mutated(monkeypatch):
    """The caller's options dict is never mutated."""
    for var in ("ETL_BACKEND", "ETL_DEVICE", "ETL_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)
    options = {"custom": 1}
    _resolve_backend_device(_IreeStub(), core.Device("cuda", 0), options)
    assert options == {"custom": 1}
    _assert_no_new_iree_import()


def test_env_target_backends_parsed(monkeypatch):
    """ETL_TARGET_BACKENDS supplies the target_backends list (any backend)."""
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "cuda,llvm-cpu")
    backend, device, options = _resolve_backend_device(None, None, {})
    assert options == {"target_backends": ["cuda", "llvm-cpu"]}
    _assert_no_new_iree_import()


def test_env_target_backends_whitespace_stripped(monkeypatch):
    """Segments are stripped; empty segments are dropped."""
    monkeypatch.setenv("ETL_TARGET_BACKENDS", " cuda , , llvm-cpu ")
    backend, device, options = _resolve_backend_device(None, None, {})
    assert options == {"target_backends": ["cuda", "llvm-cpu"]}
    _assert_no_new_iree_import()


def test_env_target_backends_empty_falls_through_to_inference(monkeypatch):
    """Empty/whitespace-only ETL_TARGET_BACKENDS is treated as unset: the
    iree-family inference still applies."""
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "   ")
    backend, device, options = _resolve_backend_device(
        _IreeStub(), core.Device("cuda", 0), {}
    )
    assert options == {"target_backends": ["cuda"]}
    _assert_no_new_iree_import()


def test_explicit_target_backends_wins_over_env(monkeypatch):
    """An explicit target_backends beats ETL_TARGET_BACKENDS."""
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "cuda,llvm-cpu")
    backend, device, options = _resolve_backend_device(None, None, {"target_backends": ["cuda"]})
    assert options == {"target_backends": ["cuda"]}
    _assert_no_new_iree_import()


# ---------------------------------------------------------------------------
# Full path through a registered "iree" stub (no real adapter import)
# ---------------------------------------------------------------------------


def test_full_path_env_iree_cuda(monkeypatch, iree_stub_registered):
    """ETL_BACKEND=iree + ETL_DEVICE=cuda:3 routes the whole build: device is
    honored and target_backends=["cuda"] reaches the compile stage."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "cuda:3")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("cuda", 3)
    assert iree_stub_registered.recorded_compile_options[-1]["target_backends"] == [
        "cuda"
    ]
    _assert_no_new_iree_import()


def test_full_path_env_iree_rocm(monkeypatch, iree_stub_registered):
    """ETL_BACKEND=iree + ETL_DEVICE=rocm:3 routes the whole build: the device
    is honored (kind preserved) and target_backends == ["rocm"] reaches the
    compile stage (no real adapter import, no GPU needed)."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "rocm:3")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("rocm", 3)
    assert iree_stub_registered.recorded_compile_options[-1]["target_backends"] == [
        "rocm"
    ]
    _assert_no_new_iree_import()


def test_full_path_env_iree_hip_alias(monkeypatch, iree_stub_registered):
    """ETL_DEVICE=hip:1 (the ROCm alias) keeps kind "hip" on the resolved
    device while the inferred compile target is the canonical ["rocm"]."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "hip:1")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("hip", 1)
    assert iree_stub_registered.recorded_compile_options[-1]["target_backends"] == [
        "rocm"
    ]
    _assert_no_new_iree_import()


def test_full_path_env_iree_cpu(monkeypatch, iree_stub_registered):
    """ETL_BACKEND=iree + ETL_DEVICE=cpu: target_backends=["llvm-cpu"] and the
    executable runs (stub delegates to numpy for cpu)."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "cpu")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("cpu", 0)
    assert iree_stub_registered.recorded_compile_options[-1]["target_backends"] == [
        "llvm-cpu"
    ]
    x, w, b = _linear_inputs()
    np.testing.assert_allclose(etl.run(exe, x, w, b).numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()


def test_full_path_explicit_target_backends_forwarded(monkeypatch, iree_stub_registered):
    """An explicit target_backends is forwarded as-is (no inference)."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "cuda:3")
    exe = etl.build(
        _linear, *_LINEAR_SPECS, target_backends=["cuda", "llvm-cpu"]
    )
    assert exe.device == core.Device("cuda", 3)
    assert iree_stub_registered.recorded_compile_options[-1][
        "target_backends"
    ] == ["cuda", "llvm-cpu"]
    _assert_no_new_iree_import()


def test_full_path_env_target_backends_beats_inference(monkeypatch, iree_stub_registered):
    """ETL_TARGET_BACKENDS overrides the device-based inference."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "cpu")
    monkeypatch.setenv("ETL_TARGET_BACKENDS", "cuda,llvm-cpu")
    exe = etl.build(_linear, *_LINEAR_SPECS)
    assert exe.device == core.Device("cpu", 0)
    assert iree_stub_registered.recorded_compile_options[-1][
        "target_backends"
    ] == ["cuda", "llvm-cpu"]
    _assert_no_new_iree_import()


def test_full_path_explicit_backend_beats_env_iree(monkeypatch, iree_stub_registered):
    """ETL_BACKEND=iree + explicit backend="numpy": the explicit kwarg wins
    and the iree stub is never consulted (env not read). The device is also
    explicit, so ETL_DEVICE is not consulted either."""
    monkeypatch.setenv("ETL_BACKEND", "iree")
    monkeypatch.setenv("ETL_DEVICE", "cuda:3")
    exe = etl.build(
        _linear, *_LINEAR_SPECS, backend="numpy", device=core.Device("cpu", 0)
    )
    assert exe.device == core.Device("cpu", 0)
    assert iree_stub_registered.recorded_compile_options == []
    _assert_no_new_iree_import()
