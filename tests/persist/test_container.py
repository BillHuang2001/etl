"""Tests for etl.persist.container — the on-disk save/load container.

Covers: round-trips through the codec, the exact byte layout, integrity
(tamper/truncate/garbage detection), format versioning, payload_type
expectations, metadata round-trips, save-side validation, non-dict
payload rejection, atomic writes, and parent-directory creation.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np
import pytest

from etl.core import Device, Dim, PersistenceError, TensorSpec
from etl.persist import (
    ETL_FORMAT_VERSION,
    LoadedObject,
    load_object,
    save_object,
)

MAGIC = b"ETLPERS"
DIGEST_LEN = 64  # 32 bytes of SHA-256 as lowercase ASCII hex


# Helpers


def _rich_payload():
    """A payload exercising many codec types (arrays, specs, dims, nesting)."""
    batch = Dim("batch")
    features = Dim("features", 8)
    return {
        "arrays": {
            "weights": np.arange(12, dtype=np.float64).reshape(3, 4),
            "ints": np.array([[1, -2], [3, 4]], dtype=np.int32),
            "special": np.array([1.0, np.nan, -np.inf, np.inf]),
            "scalar": np.float32(2.5),
        },
        "specs": [
            TensorSpec(
                shape=(batch, features + 2, 3, None),
                dtype="float32",
                device=Device("cpu", 0),
                name="x",
            ),
            TensorSpec(shape=(2,), dtype="int64"),
        ],
        "dims": {"batch": batch, "expr": features * 2 + 1},
        "scalars": {
            "int": -7,
            "float": 2.5,
            "complex": 1 + 2j,
            "bool": True,
            "none": None,
            "str": "hello",
            "slice": slice(1, 10, 2),
            "dtype": np.dtype("float32"),
        },
        "nested": [{"a": [1, 2.5, -3]}, ("tuple", "of", 4), [None, [True]]],
    }


def _assert_deep_equal(actual, expected):
    """Recursive structural equality, NaN-aware for numpy arrays."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), type(actual)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), type(actual)
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            _assert_deep_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), type(actual)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_deep_equal(a, e)
        return
    assert actual == expected, f"{actual!r} != {expected!r}"


def _save_sample(path, payload_type="widgets", payload=None, backend_info=None,
                 signature_info=None):
    """Save a valid container; return the raw bytes on disk."""
    save_object(
        path, payload_type, {"value": 42} if payload is None else payload,
        backend_info=backend_info, signature_info=signature_info,
    )
    return path.read_bytes()


def _parse_layout(data: bytes):
    """Parse the container byte layout; return (header, header_bytes, payload_bytes)."""
    header_len = struct.unpack(">I", data[11:15])[0]
    header_bytes = data[15 : 15 + header_len]
    header = json.loads(header_bytes)
    payload_len = struct.unpack(">Q", data[15 + header_len : 23 + header_len])[0]
    payload_start = 23 + header_len
    payload_bytes = data[payload_start : payload_start + payload_len]
    return header, header_bytes, payload_bytes


def _flip(data: bytes, index: int) -> bytes:
    """Return ``data`` with the byte at ``index`` bit-flipped."""
    mutated = bytearray(data)
    mutated[index] ^= 0x01
    return bytes(mutated)


def _patch(data: bytes, start: int, replacement: bytes) -> bytes:
    return data[:start] + replacement + data[start + len(replacement) :]


# 1. Round-trip


def test_round_trip_rich_payload(tmp_path):
    path = tmp_path / "round_trip.etlpers"
    payload = _rich_payload()
    save_object(path, "test.payload", payload)

    loaded = load_object(path)
    assert isinstance(loaded, LoadedObject)
    assert isinstance(loaded, dict)
    assert loaded.payload_type == "test.payload"
    assert loaded.format_version == ETL_FORMAT_VERSION
    assert loaded["format_version"] == ETL_FORMAT_VERSION
    assert loaded.backend_info is None
    assert loaded.signature_info is None
    assert set(loaded.keys()) == {
        "payload",
        "payload_type",
        "backend_info",
        "signature_info",
        "format_version",
    }
    _assert_deep_equal(loaded.payload, payload)
    # Attribute and key access must return the same objects.
    assert loaded.payload is loaded["payload"]
    assert loaded.payload_type is loaded["payload_type"]


# 2. Exact byte layout


def test_file_layout(tmp_path):
    path = tmp_path / "layout.etlpers"
    backend_info = {"name": "numpy", "version": "0.1"}
    signature_info = {"inputs": [{"dtype": "float32", "shape": [2, 3]}]}
    data = _save_sample(
        path,
        payload_type="widgets",
        backend_info=backend_info,
        signature_info=signature_info,
    )

    # Magic + version + header length fields.
    assert data[:7] == MAGIC
    assert struct.unpack(">I", data[7:11])[0] == ETL_FORMAT_VERSION
    header_len = struct.unpack(">I", data[11:15])[0]

    # Header JSON: parse, exact key set, exact values.
    header_bytes = data[15 : 15 + header_len]
    header = json.loads(header_bytes)
    assert set(header.keys()) == {
        "format_version",
        "payload_type",
        "backend_info",
        "signature_info",
        "hash_algorithm",
    }
    assert header["format_version"] == ETL_FORMAT_VERSION
    assert header["payload_type"] == "widgets"
    assert header["backend_info"] == backend_info
    assert header["signature_info"] == signature_info
    assert header["hash_algorithm"] == "sha256"

    # Payload length + payload bytes.
    payload_len = struct.unpack(">Q", data[15 + header_len : 23 + header_len])[0]
    payload_bytes = data[23 + header_len : 23 + header_len + payload_len]
    assert isinstance(json.loads(payload_bytes), dict)

    # Trailing digest: 64 lowercase ASCII hex chars, SHA-256 of payload ONLY.
    digest_bytes = data[23 + header_len + payload_len :]
    assert len(digest_bytes) == DIGEST_LEN
    assert digest_bytes.isascii() and digest_bytes == digest_bytes.lower()
    assert digest_bytes == hashlib.sha256(payload_bytes).hexdigest().encode("ascii")

    # Total length: 15 + H + 8 + P + 64.
    assert len(data) == 15 + header_len + 8 + payload_len + DIGEST_LEN


# 3. Integrity: tampering, truncation, garbage


@pytest.mark.parametrize(
    "offset",
    [0, 1, 5, "middle", "last"],
    ids=["first", "second", "fifth", "middle", "last"],
)
def test_tampered_payload_byte_detected(tmp_path, offset):
    path = tmp_path / "tamper_payload.etlpers"
    data = _save_sample(path)
    _, _, payload_bytes = _parse_layout(data)
    payload_start = len(data) - DIGEST_LEN - len(payload_bytes)
    if offset == "middle":
        index = payload_start + len(payload_bytes) // 2
    elif offset == "last":
        index = payload_start + len(payload_bytes) - 1
    else:
        index = payload_start + offset
    path.write_bytes(_flip(data, index))
    with pytest.raises(PersistenceError, match="integrity hash"):
        load_object(path)


def test_tampered_header_json_detected(tmp_path):
    path = tmp_path / "tamper_header.etlpers"
    data = _save_sample(path)
    header_len = struct.unpack(">I", data[11:15])[0]
    header_start = 15

    # Flip the leading '{' of the header JSON -> invalid JSON -> "corrupt".
    path.write_bytes(_flip(data, header_start))
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(path)

    # Rewrite the hash_algorithm field (same length) -> "corrupt".
    header_bytes = data[header_start : header_start + header_len]
    tampered = _patch(data, header_start, header_bytes.replace(b'"sha256"', b'"sha257"'))
    assert len(tampered) == len(data)
    path.write_bytes(tampered)
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(path)


def test_tampered_magic_detected(tmp_path):
    path = tmp_path / "tamper_magic.etlpers"
    data = _save_sample(path)
    path.write_bytes(b"X" + data[1:])
    with pytest.raises(PersistenceError, match="bad magic header"):
        load_object(path)


def test_truncated_file_detected(tmp_path):
    path = tmp_path / "truncated.etlpers"
    data = _save_sample(path)
    header_len = struct.unpack(">I", data[11:15])[0]
    _, _, payload_bytes = _parse_layout(data)
    payload_start = len(data) - DIGEST_LEN - len(payload_bytes)
    cut_points = {
        "header-cut": 15 + header_len - 1,
        "mid-payload-length": 15 + header_len + 4,
        "mid-payload": payload_start + len(payload_bytes) // 2,
        "payload-complete-no-digest": len(data) - DIGEST_LEN,
        "one-byte-short": len(data) - 1,
    }
    for cut in cut_points.values():
        path.write_bytes(data[:cut])
        with pytest.raises(PersistenceError, match="corrupt"):
            load_object(path)


def test_appended_garbage_detected(tmp_path):
    path = tmp_path / "appended.etlpers"
    data = _save_sample(path)
    path.write_bytes(data + b"GARBAGE")
    with pytest.raises(PersistenceError, match="payload length does not match"):
        load_object(path)


def test_garbage_file_detected(tmp_path):
    path = tmp_path / "garbage.etlpers"
    path.write_bytes(os.urandom(256))
    with pytest.raises(PersistenceError):
        load_object(path)


def test_empty_file_detected(tmp_path):
    path = tmp_path / "empty.etlpers"
    path.write_bytes(b"")
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(path)


# 4. Format version


def test_newer_format_version_detected(tmp_path):
    path = tmp_path / "newer.etlpers"
    data = _save_sample(path)
    path.write_bytes(_patch(data, 7, struct.pack(">I", ETL_FORMAT_VERSION + 1)))
    with pytest.raises(PersistenceError, match="newer format"):
        load_object(path)


def test_zero_format_version_detected(tmp_path):
    path = tmp_path / "old.etlpers"
    data = _save_sample(path)
    path.write_bytes(_patch(data, 7, struct.pack(">I", 0)))
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(path)


# 5. payload_type expectations


def test_expected_payload_type_match(tmp_path):
    path = tmp_path / "widgets.etlpers"
    _save_sample(path, payload_type="widgets")
    loaded = load_object(path, expected_payload_type="widgets")
    assert loaded.payload_type == "widgets"
    assert loaded.payload == {"value": 42}


def test_expected_payload_type_mismatch(tmp_path):
    path = tmp_path / "widgets.etlpers"
    _save_sample(path, payload_type="widgets")
    with pytest.raises(PersistenceError, match="payload_type mismatch"):
        load_object(path, expected_payload_type="gadgets")


def test_expected_payload_type_none_accepts_any(tmp_path):
    path = tmp_path / "widgets.etlpers"
    _save_sample(path, payload_type="widgets")
    assert load_object(path).payload_type == "widgets"
    assert load_object(path, expected_payload_type=None).payload_type == "widgets"


# 6. Metadata round-trip


def test_backend_and_signature_info_round_trip(tmp_path):
    path = tmp_path / "meta.etlpers"
    backend_info = {
        "name": "numpy",
        "version": "0.1.0",
        "capabilities": ["cpu", "dlpack"],
        "target": {"arch": "x86_64", "features": ["sse4", "avx2"]},
    }
    signature_info = {
        "inputs": [{"dtype": "float32", "shape": [2, 3], "name": "x"}],
        "outputs": [{"dtype": "int64", "shape": [2]}],
        "static": {"lr": 0.1, "tags": ["a", "b"], "nested": {"on": True}},
    }
    _save_sample(path, backend_info=backend_info, signature_info=signature_info)
    loaded = load_object(path)
    assert loaded.backend_info == backend_info
    assert loaded.signature_info == signature_info
    assert loaded["backend_info"] == backend_info
    assert loaded["signature_info"] == signature_info


# 7. Save-side validation


@pytest.mark.parametrize(
    "bad_type",
    [123, "", None, b"bytes", 1.5],
    ids=["int", "empty", "none", "bytes", "float"],
)
def test_save_rejects_invalid_payload_type(tmp_path, bad_type):
    path = tmp_path / "bad_type.etlpers"
    with pytest.raises(PersistenceError, match="payload_type must be a non-empty string"):
        save_object(path, bad_type, {"value": 1})


def test_save_rejects_non_json_serializable_metadata(tmp_path):
    path = tmp_path / "bad_meta.etlpers"
    with pytest.raises(PersistenceError, match="JSON-serializable"):
        save_object(path, "widgets", {"value": 1}, backend_info={"tags": {"a", "b"}})
    with pytest.raises(PersistenceError, match="JSON-serializable"):
        save_object(
            path, "widgets", {"value": 1}, signature_info={"opaque": object()}
        )


def test_save_rejects_non_encodable_payload(tmp_path):
    path = tmp_path / "bad_payload.etlpers"
    with pytest.raises(PersistenceError):
        save_object(path, "widgets", {"opaque": object()})


# 8. Payload must decode to a dict


def test_non_dict_payload_fails_on_load(tmp_path):
    path = tmp_path / "list_payload.etlpers"
    # Saving a non-dict payload succeeds (it encodes fine) ...
    save_object(path, "t", [1, 2, 3])
    # ... but loading it must fail: artifacts always hold payload dicts.
    with pytest.raises(PersistenceError, match="payload must decode to a dict"):
        load_object(path)


# 9. Atomicity


def test_failed_save_leaves_no_partial_file_or_temp(tmp_path):
    path = tmp_path / "never_written.etlpers"
    with pytest.raises(PersistenceError):
        save_object(path, "", {"value": 1})
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_replace_cleans_up_temp_file(tmp_path):
    # Failure AFTER the temp file is written (dest is a directory, so
    # os.replace fails) must still unlink the temp file.
    dest = tmp_path / "sub"
    dest.mkdir()
    with pytest.raises(OSError):
        save_object(dest, "widgets", {"value": 1})
    assert list(tmp_path.glob("*.tmp")) == []


# 10. Parent directory creation


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "f.etlpers"
    payload = {"deep": [1, 2, 3]}
    save_object(path, "nested", payload)
    assert path.exists()
    loaded = load_object(path)
    assert loaded.payload_type == "nested"
    _assert_deep_equal(loaded.payload, payload)
