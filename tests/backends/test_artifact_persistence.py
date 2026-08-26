"""Save/load round-trips and failure modes for ``LoweredProgram``,
``CompiledArtifact``, and the numpy ``Executable``.

The persistence contract under test (see ``etl/persist/container.py``): every
file is a self-describing container — magic header + format version + JSON
metadata (payload type, backend info, signature info) + JSON payload + a
SHA-256 integrity hash. Loading validates everything and fails with
``etl.PersistenceError`` on any corruption or environment mismatch — an
artifact is never silently re-lowered/re-compiled (spec §10).

Key facts tested here:

* ``LoweredProgram.save/load`` and ``CompiledArtifact.save/load`` use the
  etl.persist container; the numpy payload is a serialized ``ir.Module`` dict.
* Load validates the recorded backend name against the registry and (for
  artifacts) ``required_custom_ops`` / ``runtime_dependencies`` against the
  environment.
* ``etl.load(artifact, backend=..., device=...)`` wraps the backend
  executable; the numpy backend validates the device (v1 CPU only) and never
  re-compiles.
* The numpy executable saves the underlying CompiledArtifact; the executable
  is reconstructed EXPLICITLY at load — device handles and live interpreter
  state are never serialized.
"""

import numpy as np
import pytest

import etl
from etl import backends


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fn():
    def fn(x):
        return etl.multiply(x, etl.add(x, 2.0))

    return fn


def _spec():
    return etl.TensorSpec(shape=(2, 3), dtype=etl.float32)


def _x():
    # 3 * (3 + 2) == 15 for every element.
    return np.full((2, 3), 3.0, dtype=np.float32)


def _lowered():
    return etl.lower(etl.trace(_make_fn(), _spec()))


def _artifact():
    return etl.compile(_lowered())


# ---------------------------------------------------------------------------
# 1. LoweredProgram save/load round-trip
# ---------------------------------------------------------------------------


def test_lowered_program_roundtrip(tmp_path):
    lowered = _lowered()
    path = tmp_path / "lowered.etl"
    lowered.save(path)

    loaded = backends.LoweredProgram.load(path)

    assert loaded.backend == lowered.backend == "numpy"
    # numpy payload = versioned, self-describing serialized ir.Module dict.
    assert isinstance(loaded.payload, dict)
    assert loaded.payload == lowered.payload
    assert loaded.text() == lowered.text()
    # Signature fields (specs, trees, static values) survive the round-trip.
    assert loaded.signature == lowered.signature
    assert loaded.signature.input_tree == lowered.signature.input_tree
    assert loaded.signature.output_tree == lowered.signature.output_tree
    assert loaded.signature.input_specs == lowered.signature.input_specs
    assert loaded.signature.output_specs == lowered.signature.output_specs
    assert loaded.signature.static_values == lowered.signature.static_values
    assert (
        loaded.signature.output_static_values
        == lowered.signature.output_static_values
    )


# ---------------------------------------------------------------------------
# 2. CompiledArtifact save/load round-trip
# ---------------------------------------------------------------------------


def test_compiled_artifact_roundtrip_and_run(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)

    loaded = backends.CompiledArtifact.load(path)

    assert loaded.backend == artifact.backend == "numpy"
    assert loaded.target == artifact.target == "cpu"
    assert loaded.required_custom_ops == artifact.required_custom_ops == ()
    assert loaded.runtime_dependencies == artifact.runtime_dependencies == {
        "numpy": np.__version__
    }
    assert loaded.signature == artifact.signature

    # A loaded artifact produces a runnable executable whose numerical
    # result matches a fresh build.
    exe = etl.load(loaded)
    fresh = etl.build(_make_fn(), _spec())
    out = etl.run(exe, _x())
    np.testing.assert_allclose(out.numpy(), etl.run(fresh, _x()).numpy())
    np.testing.assert_allclose(out.numpy(), 15.0)


# ---------------------------------------------------------------------------
# 3. Executable save/load round-trip (numpy)
# ---------------------------------------------------------------------------


def test_executable_save_load_roundtrip(tmp_path):
    # The numpy executable's save() writes the underlying CompiledArtifact;
    # the executable is reconstructed EXPLICITLY at load — device handles and
    # live interpreter state are never serialized.
    exe = etl.build(_make_fn(), _spec())
    path = tmp_path / "executable.etl"
    exe.save(path)

    # Pipeline round-trip: etl.Executable.load re-wraps the backend
    # executable with the decoded structured signature.
    reloaded = etl.Executable.load(path)
    assert isinstance(reloaded.backend_executable, backends.NumpyExecutable)
    np.testing.assert_allclose(
        etl.run(reloaded, _x()).numpy(), etl.run(exe, _x()).numpy()
    )
    np.testing.assert_allclose(etl.run(reloaded, _x()).numpy(), 15.0)

    # Raw backend-executable round-trip: NumpyExecutable.load speaks flat
    # tensor lists.
    raw = backends.NumpyExecutable.load(path)
    flat = raw.run([etl.tensor(_x())])
    np.testing.assert_allclose(flat[0].numpy(), 15.0)


# ---------------------------------------------------------------------------
# 4. Corrupt files
# ---------------------------------------------------------------------------


def test_corrupt_garbage_file(tmp_path):
    path = tmp_path / "garbage.etl"
    path.write_bytes(b"this is definitely not an etl persistence container")
    with pytest.raises(etl.PersistenceError, match="corrupt"):
        backends.LoweredProgram.load(path)
    with pytest.raises(etl.PersistenceError, match="corrupt"):
        backends.CompiledArtifact.load(path)


def test_corrupt_truncated_file(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)
    data = path.read_bytes()
    # Cut the valid container in half: the integrity/length checks must fail.
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(etl.PersistenceError, match="corrupt"):
        backends.CompiledArtifact.load(path)


def test_payload_type_mismatch(tmp_path):
    # A valid-format file carrying the wrong payload type tag is rejected —
    # never a silent reinterpretation of a file written for another stage.
    lowered = _lowered()
    path = tmp_path / "lowered.etl"
    lowered.save(path)
    with pytest.raises(etl.PersistenceError, match="payload_type mismatch"):
        backends.CompiledArtifact.load(path)
    # The pipeline Executable requires a compiled artifact, not a lowered
    # program (which must be compiled first).
    with pytest.raises(etl.PersistenceError, match="not an executable artifact"):
        etl.Executable.load(path)


# ---------------------------------------------------------------------------
# 5. Backend name validation
# ---------------------------------------------------------------------------


def test_unregistered_backend(tmp_path):
    # A container recording an unknown backend fails to load — the artifact
    # is never silently re-lowered/re-compiled by another backend.
    lowered = _lowered()
    fake = backends.LoweredProgram(
        backend="nonexistent-backend-xyz",
        signature=lowered.signature,
        payload={"version": 1, "module": {"name": "main"}},
    )
    path = tmp_path / "fake_backend.etl"
    fake.save(path)
    with pytest.raises(etl.PersistenceError, match="unknown backend"):
        backends.LoweredProgram.load(path)

    fake_artifact = backends.CompiledArtifact(
        backend="nonexistent-backend-xyz",
        signature=lowered.signature,
        target="cpu",
        payload={"version": 1, "module": {"name": "main"}},
    )
    artifact_path = tmp_path / "fake_backend_artifact.etl"
    fake_artifact.save(artifact_path)
    with pytest.raises(etl.PersistenceError, match="unknown backend"):
        backends.CompiledArtifact.load(artifact_path)


def test_explicit_backend_name_mismatch(tmp_path):
    # Passing an explicit backend whose name differs from the recorded one
    # fails — never a silent recompile (pipeline.load semantics).
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)
    loaded = backends.CompiledArtifact.load(path)

    with pytest.raises(etl.PersistenceError, match="never silently recompile"):
        etl.load(loaded, backend="stablehlo")
    # The matching name is accepted.
    exe = etl.load(loaded, backend="numpy")
    np.testing.assert_allclose(etl.run(exe, _x()).numpy(), 15.0)


# ---------------------------------------------------------------------------
# 6. Device validation on load
# ---------------------------------------------------------------------------


def test_device_mismatch_on_load(tmp_path):
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)
    loaded = backends.CompiledArtifact.load(path)

    # The numpy backend is CPU-only in v1: a GPU device fails explicitly.
    with pytest.raises(etl.BackendError, match="CPU devices only"):
        etl.load(loaded, device=etl.Device("gpu", 0))
    with pytest.raises(etl.BackendError, match="CPU devices only"):
        backends.NumpyExecutable.load(path, device=etl.Device("gpu", 0))
    # Non-Device objects fail device normalization before reaching a backend.
    with pytest.raises(etl.DeviceError):
        etl.load(loaded, device=object())

    # None (the default) and explicit CPU devices work.
    assert etl.load(loaded).device == etl.Device("cpu", 0)
    exe_cpu = etl.load(loaded, device=etl.Device("cpu", 0))
    np.testing.assert_allclose(etl.run(exe_cpu, _x()).numpy(), 15.0)
    exe_str = etl.load(loaded, device="cpu")
    np.testing.assert_allclose(etl.run(exe_str, _x()).numpy(), 15.0)


# ---------------------------------------------------------------------------
# 7. Runtime dependency tampering
# ---------------------------------------------------------------------------


def test_runtime_dependencies_tamper(tmp_path):
    # A recorded runtime dependency that does not match the environment must
    # fail at load — the artifact is never silently recompiled. The tampered
    # copy is a separate dataclass; the original is untouched.
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)
    loaded = backends.CompiledArtifact.load(path)

    tampered = backends.CompiledArtifact(
        backend=loaded.backend,
        signature=loaded.signature,
        target=loaded.target,
        payload=loaded.payload,
        required_custom_ops=loaded.required_custom_ops,
        runtime_dependencies={"numpy": "0.0.0"},
    )
    tampered_path = tmp_path / "tampered.etl"
    tampered.save(tampered_path)
    with pytest.raises(etl.PersistenceError, match="never silently recompile"):
        backends.CompiledArtifact.load(tampered_path)

    # The original on-disk artifact still loads.
    reloaded = backends.CompiledArtifact.load(path)
    assert reloaded.runtime_dependencies == {"numpy": np.__version__}


# ---------------------------------------------------------------------------
# 8. Self-description / re-save idempotence
# ---------------------------------------------------------------------------


def test_resave_idempotence(tmp_path):
    # Self-description: a loaded artifact exposes backend name + target as
    # attributes, and re-saving a loaded artifact round-trips again.
    artifact = _artifact()
    path = tmp_path / "artifact.etl"
    artifact.save(path)

    first = backends.CompiledArtifact.load(path)
    assert first.backend == "numpy"
    assert first.target == "cpu"

    second_path = tmp_path / "artifact_resaved.etl"
    first.save(second_path)
    second = backends.CompiledArtifact.load(second_path)
    assert second.backend == first.backend
    assert second.target == first.target
    assert second.payload == first.payload
    assert second.signature == first.signature
    assert second.runtime_dependencies == first.runtime_dependencies
    # Deterministic container encoding: the re-save is byte-identical.
    assert second_path.read_bytes() == path.read_bytes()
