"""Public-API integration smoke tests for the etl persistence layer.

Everything here goes through the root ``etl`` package exactly as a user
would consume it: ``etl.persist.*``, ``etl.FileCache`` / ``etl.Cache``, and
the public error type ``etl.core.PersistenceError``. Full pipeline-type
artifact round-trips (Graph / LoweredProgram / CompiledArtifact saves) are
covered by ``tests/pipeline_test.py`` and are deliberately not duplicated.
"""

import math
import struct

import numpy as np
import pytest

import etl
from etl.core import Dim, Device, PersistenceError, TensorSpec
from etl.persist import ETL_FORMAT_VERSION


def test_public_api_surface():
    """The persistence primitives are reachable through the root package."""
    assert callable(etl.persist.save_object)
    assert callable(etl.persist.load_object)
    assert etl.persist.FileCache is etl.FileCache
    assert etl.persist.Cache is etl.Cache
    # The public error type users catch is the same object everywhere.
    assert etl.PersistenceError is PersistenceError


def _artifact_payload():
    """A realistic artifact-like payload covering the codec surface."""
    b = Dim("B")
    c = Dim("C", size=64)
    spec = TensorSpec(
        shape=(b, 2 * c, None),
        dtype=np.float32,
        device=Device("cpu"),
        name="weights",
    )
    return {
        "w_f64": np.array([[1.0, np.nan], [np.inf, -0.5]], dtype=np.float64),
        "w_f32": np.array([1.5, -2.25, 0.0], dtype=np.float32),
        "ids_i64": np.arange(6, dtype=np.int64).reshape(2, 3),
        "spec": spec,
        "metadata": {
            "kind": "artifact",
            "steps": 3,
            "tags": ["weights", "v1"],
            "lr": math.pi,
        },
        "dims": [b, c],
        "rot": 1.0 + 2.0j,
    }


def test_save_load_roundtrip_artifact_payload(tmp_path):
    """save_object/load_object round-trip a realistic payload verbatim."""
    path = tmp_path / "artifact.etl"
    payload = _artifact_payload()

    etl.persist.save_object(path, "test.artifact", payload)
    loaded = etl.persist.load_object(path)
    assert loaded.payload_type == "test.artifact"

    got = loaded.payload
    # numpy arrays: nan-aware structural equality.
    assert np.array_equal(got["w_f64"], payload["w_f64"], equal_nan=True)
    assert np.array_equal(got["w_f32"], payload["w_f32"])
    assert np.array_equal(got["ids_i64"], payload["ids_i64"])
    assert got["rot"] == payload["rot"]

    # TensorSpec fields: symbolic shape entries compare structurally.
    spec = payload["spec"]
    decoded = got["spec"]
    assert isinstance(decoded, TensorSpec)
    assert decoded.shape == spec.shape
    assert decoded.shape[1] == 2 * Dim("C", size=64)
    assert decoded.dtype == spec.dtype == np.dtype("float32")
    assert decoded.device == Device("cpu")
    assert decoded.name == "weights"

    # Dims by name/size; metadata by plain equality.
    assert [(d.name, d.size) for d in got["dims"]] == [("B", None), ("C", 64)]
    assert got["metadata"] == payload["metadata"]


def test_filecache_smoke_through_root_package(tmp_path):
    """FileCache put/get/hit-reuse through ``etl.FileCache``."""
    key = ("artifact", "v1", {"dtype": np.dtype("float32")})
    payload = {
        "w": np.arange(6, dtype=np.float32).reshape(2, 3),
        "bias": np.array([0.1, 0.2], dtype=np.float32),
    }

    cache = etl.FileCache(tmp_path / "smoke_cache")
    cache.put(key, payload)

    # Hit without a compute_fn: the stored value comes back unchanged.
    got = cache.get(key, compute_fn=None)
    assert np.array_equal(got["w"], payload["w"])
    assert np.array_equal(got["bias"], payload["bias"])

    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {
            "w": np.zeros((2, 3), dtype=np.float32),
            "bias": np.zeros(2, dtype=np.float32),
        }

    # A fresh instance on the same directory hits without recomputing.
    fresh = etl.FileCache(tmp_path / "smoke_cache")
    hit = fresh.get(key, compute_fn=compute)
    assert calls["n"] == 0
    assert np.array_equal(hit["w"], payload["w"])

    # A genuinely new key does compute (and store) exactly once.
    miss = fresh.get(("artifact", "v2"), compute_fn=compute)
    assert calls["n"] == 1
    assert np.array_equal(miss["w"], np.zeros((2, 3), dtype=np.float32))


def test_tampered_payload_raises_persistence_error(tmp_path):
    """One flipped payload byte surfaces as the public PersistenceError."""
    path = tmp_path / "tampered.etl"
    etl.persist.save_object(path, "test.tamper", {"x": np.arange(4, dtype=np.int64)})

    data = bytearray(path.read_bytes())
    # Byte layout: magic(7) + version(4) + header_len(4) + header + payload_len(8) + payload.
    header_len = struct.unpack(">I", bytes(data[11:15]))[0]
    payload_start = 15 + header_len + 8
    data[payload_start] ^= 0x01  # single-bit flip inside the payload
    path.write_bytes(bytes(data))

    with pytest.raises(PersistenceError, match="corrupt"):
        etl.persist.load_object(path)


def test_loaded_format_version_matches_public_constant(tmp_path):
    """loaded.format_version reports the public ETL_FORMAT_VERSION."""
    path = tmp_path / "version.etl"
    etl.persist.save_object(path, "test.version", {"v": 1})
    loaded = etl.persist.load_object(path)
    assert loaded.format_version == ETL_FORMAT_VERSION
