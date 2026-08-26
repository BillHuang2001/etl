"""Backend abstraction contract tests.

Contract under test: ``../etl/backends/CONTEXT.md`` ("Backend contract") —
``Capabilities`` (frozen dataclass), the ``Backend`` ABC, the
runtime-checkable ``Executable`` protocol, the backend registry
(``register``/``get``), and the staged objects owned by backends
(``Signature``, ``LoweredProgram``, ``CompiledArtifact``).

CPU only; tiny shapes; everything imported from ``etl`` directly.
"""

import dataclasses

import numpy as np
import pytest

import etl
from etl import backends

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_backend_class(name, *, missing_method=None):
    """Minimal concrete ``Backend`` subclass (optionally missing one method).

    The staging methods return placeholder objects — the ABC contract only
    requires that ``lower``/``compile``/``load`` exist and are callable.
    """

    def lower(self, graph, options=None):
        return object()

    def compile(self, lowered, options=None):
        return object()

    def load(self, artifact, device=None):
        return object()

    namespace = {"lower": lower, "compile": compile, "load": load}
    if missing_method is not None:
        del namespace[missing_method]
    cls = type(f"FakeBackend_{name}", (backends.Backend,), namespace)
    cls.name = name
    return cls


#: Every dtype the numpy backend declares: the numeric groups (the pre-2.0
#: ``numpy.sctypes`` groups) plus ``numpy.bool_``.
NUMPY_BACKEND_DTYPES = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


@etl.defn
def _add_self(x):
    return etl.add(x, x)


@pytest.fixture(scope="module")
def staged():
    """A tiny staged pair (LoweredProgram + CompiledArtifact) via the pipeline."""
    graph = etl.trace(_add_self, etl.TensorSpec((2, 3), etl.float32))
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    return lowered, artifact


# ---------------------------------------------------------------------------
# 1. Capabilities
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_defaults_are_all_off(self):
        caps = backends.Capabilities()
        assert caps.dynamic_shapes is False
        assert caps.collectives is False
        assert caps.runtime_calls is False
        assert caps.custom_blocks is False
        assert caps.async_collectives is False
        assert caps.dtypes == frozenset()

    def test_frozen(self):
        caps = backends.Capabilities()
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.dynamic_shapes = True


# ---------------------------------------------------------------------------
# 2. numpy backend (the registered default)
# ---------------------------------------------------------------------------


class TestNumpyBackend:
    def test_is_registered_default(self):
        assert backends.get("numpy") is backends.numpy_backend
        assert isinstance(backends.numpy_backend, backends.NumpyBackend)
        assert isinstance(backends.numpy_backend, backends.Backend)

    def test_name(self):
        assert backends.numpy_backend.name == "numpy"

    def test_capability_flags(self):
        caps = backends.numpy_backend.capabilities
        assert caps.dynamic_shapes is True
        assert caps.collectives is True
        assert caps.runtime_calls is True
        assert caps.custom_blocks is True
        assert caps.async_collectives is False

    def test_dtypes_is_a_frozenset(self):
        assert isinstance(backends.numpy_backend.capabilities.dtypes, frozenset)

    @pytest.mark.parametrize(
        "nptype", NUMPY_BACKEND_DTYPES, ids=lambda t: np.dtype(t).name
    )
    def test_dtypes_membership(self, nptype):
        assert np.dtype(nptype) in backends.numpy_backend.capabilities.dtypes

    @pytest.mark.parametrize(
        "nptype", NUMPY_BACKEND_DTYPES, ids=lambda t: np.dtype(t).name
    )
    def test_supports_dtype_true_for_declared(self, nptype):
        assert (
            backends.numpy_backend.capabilities.supports_dtype(np.dtype(nptype))
            is True
        )

    def test_supports_dtype_uses_dtype_equality(self):
        # A freshly constructed dtype instance is accepted: membership is
        # numpy dtype VALUE equality, not object identity.
        caps = backends.numpy_backend.capabilities
        assert caps.supports_dtype(np.dtype("float64")) is True
        assert caps.supports_dtype(np.dtype("float32")) is True

    @pytest.mark.parametrize(
        "dtype",
        [np.dtype(object), np.dtype("S4"), np.dtype("U3")],
        ids=["object", "bytes", "str"],
    )
    def test_supports_dtype_false_for_undeclared(self, dtype):
        # Not part of the numeric/bool groups the numpy backend declares.
        assert backends.numpy_backend.capabilities.supports_dtype(dtype) is False


# ---------------------------------------------------------------------------
# 3. Backend ABC
# ---------------------------------------------------------------------------


class TestBackendABC:
    def test_base_class_declares_name_and_capabilities(self):
        assert backends.Backend.name == ""
        assert isinstance(backends.Backend.capabilities, backends.Capabilities)

    def test_abstract_instantiation_raises(self):
        with pytest.raises(TypeError, match="abstract"):
            backends.Backend()

    @pytest.mark.parametrize("missing", ["lower", "compile", "load"])
    def test_missing_staging_method_raises(self, missing):
        cls = _make_backend_class(f"fake-missing-{missing}", missing_method=missing)
        with pytest.raises(TypeError, match="abstract"):
            cls()

    def test_concrete_subclass_instantiates(self):
        backend = _make_backend_class("fake-test")()
        assert backend.name == "fake-test"
        # The ABC only requires the staging methods to exist; fake backends
        # return placeholders in these contract tests.
        assert backend.lower(object()) is not None
        assert backend.compile(object()) is not None
        assert backend.load(object()) is not None


# ---------------------------------------------------------------------------
# 4. Executable protocol (runtime-checkable)
# ---------------------------------------------------------------------------


class TestExecutableProtocol:
    def test_minimal_implementation_passes(self):
        class Minimal:
            functions = ("main",)
            device = None

            def run(self, flat_input_tensors):
                return []

            def save(self, path):
                pass

            @classmethod
            def load(cls, path, device=None):
                return cls()

        assert isinstance(Minimal(), backends.Executable)

    def test_missing_run_fails(self):
        class NoRun:
            functions = ("main",)
            device = None

            def save(self, path):
                pass

            @classmethod
            def load(cls, path, device=None):
                return cls()

        assert not isinstance(NoRun(), backends.Executable)


# ---------------------------------------------------------------------------
# 5. Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_get_roundtrip(self):
        backend = _make_backend_class("fake-reg-roundtrip")()
        assert backends.register(backend) is backend
        assert backends.get("fake-reg-roundtrip") is backend

    def test_reregistering_same_instance_is_idempotent(self):
        backend = _make_backend_class("fake-reg-idempotent")()
        assert backends.register(backend) is backend
        assert backends.register(backend) is backend
        assert backends.get("fake-reg-idempotent") is backend

    def test_duplicate_name_different_instance_raises(self):
        backends.register(_make_backend_class("fake-reg-duplicate")())
        with pytest.raises(etl.BackendError, match="already registered"):
            backends.register(_make_backend_class("fake-reg-duplicate")())

    def test_register_non_backend_raises(self):
        with pytest.raises(etl.BackendError, match="Backend instance"):
            backends.register(object())

    def test_register_empty_name_raises(self):
        with pytest.raises(etl.BackendError, match="non-empty"):
            backends.register(_make_backend_class("")())

    def test_get_unknown_name_raises(self):
        with pytest.raises(etl.BackendError, match="unknown backend") as excinfo:
            backends.get("no-such-backend-xyz")
        assert "no-such-backend-xyz" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. LoweredProgram / CompiledArtifact / Signature (owned by backends)
# ---------------------------------------------------------------------------


class TestStagedObjects:
    def test_staged_types_re_exported_from_etl(self):
        assert etl.LoweredProgram is backends.LoweredProgram
        assert etl.CompiledArtifact is backends.CompiledArtifact
        assert etl.Backend is backends.Backend
        assert etl.Capabilities is backends.Capabilities
        assert etl.numpy_backend is backends.numpy_backend

    def test_lowered_program_attrs(self, staged):
        lowered, _ = staged
        assert isinstance(lowered, backends.LoweredProgram)
        assert lowered.backend == "numpy"
        assert isinstance(lowered.signature, backends.Signature)
        assert isinstance(lowered.payload, dict)
        text = lowered.text()
        assert isinstance(text, str)
        assert text

    def test_signature_trees_and_specs(self, staged):
        lowered, _ = staged
        sig = lowered.signature
        assert isinstance(sig.input_tree, etl.TreeSpec)
        assert isinstance(sig.output_tree, etl.TreeSpec)
        assert len(sig.input_specs) == 1
        assert isinstance(sig.input_specs[0], etl.TensorSpec)
        assert sig.input_specs[0].shape == (2, 3)
        assert sig.input_specs[0].dtype == np.dtype("float32")
        assert len(sig.output_specs) == 1
        assert isinstance(sig.output_specs[0], etl.TensorSpec)
        assert sig.output_specs[0].shape == (2, 3)
        assert sig.output_specs[0].dtype == np.dtype("float32")
        assert sig.static_values == ()
        assert sig.output_static_values == ()

    def test_compiled_artifact_attrs(self, staged):
        _, artifact = staged
        assert isinstance(artifact, backends.CompiledArtifact)
        assert artifact.backend == "numpy"
        assert artifact.target == "cpu"
        assert artifact.required_custom_ops == ()
        assert artifact.runtime_dependencies["numpy"] == np.__version__
        assert isinstance(artifact.payload, dict)

    def test_artifact_signature_matches_lowered(self, staged):
        lowered, artifact = staged
        assert artifact.signature.input_tree == lowered.signature.input_tree
        assert artifact.signature.output_tree == lowered.signature.output_tree
        assert artifact.signature.input_specs == lowered.signature.input_specs
        assert artifact.signature.output_specs == lowered.signature.output_specs
        assert artifact.signature.static_values == lowered.signature.static_values
        assert (
            artifact.signature.output_static_values
            == lowered.signature.output_static_values
        )

    def test_lowered_program_save_load_roundtrip(self, staged, tmp_path):
        lowered, _ = staged
        path = tmp_path / "lowered.etl"
        lowered.save(path)
        restored = backends.LoweredProgram.load(path)
        assert isinstance(restored, backends.LoweredProgram)
        assert restored.backend == lowered.backend
        assert restored.signature == lowered.signature
        assert restored.payload == lowered.payload

    def test_compiled_artifact_save_load_roundtrip(self, staged, tmp_path):
        _, artifact = staged
        path = tmp_path / "artifact.etl"
        artifact.save(path)
        restored = backends.CompiledArtifact.load(path)
        assert isinstance(restored, backends.CompiledArtifact)
        assert restored.backend == artifact.backend
        assert restored.target == artifact.target
        assert restored.signature == artifact.signature
        assert restored.required_custom_ops == artifact.required_custom_ops
        assert restored.runtime_dependencies == artifact.runtime_dependencies

    def test_numpy_executable_satisfies_protocol(self, staged):
        _, artifact = staged
        executable = backends.numpy_backend.load(artifact)
        assert isinstance(executable, backends.NumpyExecutable)
        assert isinstance(executable, backends.Executable)
        assert isinstance(executable.functions, tuple)
        assert "main" in executable.functions
        assert executable.device is None

    def test_backend_executable_run(self, staged):
        _, artifact = staged
        executable = backends.numpy_backend.load(artifact)
        x = etl.tensor(np.ones((2, 3), dtype=np.float32))
        (y,) = executable.run([x])
        assert isinstance(y, etl.Tensor)
        np.testing.assert_allclose(y.numpy(), 2.0 * np.ones((2, 3), dtype=np.float32))

    def test_numpy_executable_save_load_roundtrip(self, staged, tmp_path):
        _, artifact = staged
        executable = backends.numpy_backend.load(artifact)
        path = tmp_path / "executable.etl"
        executable.save(path)
        restored = backends.NumpyExecutable.load(path)
        assert isinstance(restored, backends.NumpyExecutable)
        x = etl.tensor(np.ones((2, 3), dtype=np.float32))
        (y,) = restored.run([x])
        np.testing.assert_allclose(y.numpy(), 2.0 * np.ones((2, 3), dtype=np.float32))

    def test_pipeline_load_run_smoke(self, staged):
        _, artifact = staged
        executable = etl.load(artifact)
        y = etl.run(executable, np.ones((2, 3), dtype=np.float32))
        np.testing.assert_allclose(y.numpy(), 2.0 * np.ones((2, 3), dtype=np.float32))
