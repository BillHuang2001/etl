"""Error-behavior tests for the etl persistence container.

The documented contract (``etl/persist/CONTEXT.md``) says validation
failures raise ``PersistenceError``: bad magic, newer format version,
invalid header/payload JSON, integrity hash mismatch, payload_type
mismatch, unknown codec type, cyclic values, and codec shadowing.

That enumeration covers *content* validation. I/O-level failures are not
content validation: ``load_object``'s ``open()`` OSError propagates
unchanged (missing file -> ``FileNotFoundError``, directory path ->
``IsADirectoryError``, both OSErrors), so those are asserted with
``pytest.raises(OSError)``/``pytest.raises(FileNotFoundError)`` below.
"""

import hashlib
import json
import struct

import pytest

from etl.core import ETLError, PersistenceError
from etl.persist import ETL_FORMAT_VERSION, load_object, register_codec, save_object

_MAGIC = b"ETLPERS"


def _write_container(path, header_bytes, payload_bytes=None, version=ETL_FORMAT_VERSION):
    """Assemble a container file at ``path``.

    Layout: magic || uint32 version || uint32 header length || header ||
    uint64 payload length || payload || 64-char sha256 hex digest of the
    payload. With ``payload_bytes=None`` the digest field is 64 zero bytes,
    which is enough for tests that fail before the digest is ever read
    (header validation).
    """
    payload = b"" if payload_bytes is None else payload_bytes
    digest = (
        hashlib.sha256(payload).hexdigest().encode("ascii")
        if payload_bytes is not None
        else b"\x00" * 64
    )
    blob = (
        _MAGIC
        + struct.pack(">I", version)
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + struct.pack(">Q", len(payload))
        + payload
        + digest
    )
    path.write_bytes(blob)


def _valid_header_bytes(payload_type="test.payload", **overrides):
    """A fully valid header JSON; ``overrides`` replace individual fields."""
    header = {
        "format_version": ETL_FORMAT_VERSION,
        "payload_type": payload_type,
        "backend_info": None,
        "signature_info": None,
        "hash_algorithm": "sha256",
    }
    header.update(overrides)
    return json.dumps(header, sort_keys=True).encode("utf-8")


# --- error hierarchy -------------------------------------------------------


def test_persistence_error_is_etl_error():
    assert issubclass(PersistenceError, ETLError)
    assert isinstance(PersistenceError("boom"), ETLError)


# --- I/O-level failures (not content validation) ---------------------------


def test_load_missing_file(tmp_path):
    # The contract's PersistenceError enumeration is about file-content
    # validation; a missing file is an I/O failure, and open()'s
    # FileNotFoundError propagates unchanged.
    with pytest.raises(FileNotFoundError):
        load_object(tmp_path / "nope.etlpers")


def test_load_directory_as_path(tmp_path):
    # Same rationale as above: open() raises IsADirectoryError (an
    # OSError), not a PersistenceError — nothing was read, so there is no
    # content to validate.
    with pytest.raises(OSError):
        load_object(tmp_path)


# --- structural corruption -------------------------------------------------


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.etlpers"
    p.write_bytes(b"")
    with pytest.raises(PersistenceError, match="too short"):
        load_object(p)


def test_load_bad_magic(tmp_path):
    p = tmp_path / "garbage.etlpers"
    p.write_bytes(bytes(range(256)))
    with pytest.raises(PersistenceError, match="bad magic"):
        load_object(p)


def test_load_valid_magic_garbage_tail(tmp_path):
    p = tmp_path / "truncated.etlpers"
    p.write_bytes(_MAGIC + b"\x00" * 8)
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(p)


# --- format version --------------------------------------------------------


def test_load_newer_format_version(tmp_path):
    p = tmp_path / "future.etlpers"
    save_object(p, "test.payload", {"x": 1})
    blob = bytearray(p.read_bytes())
    blob[7:11] = struct.pack(">I", ETL_FORMAT_VERSION + 1)
    p.write_bytes(blob)
    with pytest.raises(PersistenceError, match="newer format"):
        load_object(p)


def test_load_unknown_older_format_version(tmp_path):
    p = tmp_path / "old.etlpers"
    save_object(p, "test.payload", {"x": 1})
    blob = bytearray(p.read_bytes())
    blob[7:11] = struct.pack(">I", 0)
    p.write_bytes(blob)
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(p)


# --- header validation -----------------------------------------------------


def test_load_invalid_header_json(tmp_path):
    p = tmp_path / "bad-header.etlpers"
    _write_container(p, b"{not json")
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(p)


def test_load_non_dict_header(tmp_path):
    p = tmp_path / "list-header.etlpers"
    _write_container(p, b"[]")  # valid JSON, but not an object
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(p)


def test_load_unsupported_hash_algorithm(tmp_path):
    p = tmp_path / "md5.etlpers"
    _write_container(p, _valid_header_bytes(hash_algorithm="md5"))
    with pytest.raises(PersistenceError, match="unsupported hash algorithm"):
        load_object(p)


# --- payload validation ----------------------------------------------------


def test_load_corrupt_payload_json(tmp_path):
    p = tmp_path / "bad-payload.etlpers"
    payload = b"definitely not json"
    # The digest must match the payload bytes, otherwise the integrity
    # check fires before the JSON parse and masks this failure.
    _write_container(p, _valid_header_bytes(), payload)
    with pytest.raises(PersistenceError, match="not valid JSON"):
        load_object(p)


def test_load_unknown_codec_type(tmp_path):
    p = tmp_path / "unknown-codec.etlpers"
    payload = json.dumps(
        {"__etl_encoded__": True, "type": "no.such.type", "data": None},
        sort_keys=True,
    ).encode("utf-8")
    _write_container(p, _valid_header_bytes(), payload)
    # load_object wraps codec decode errors with a "(corrupt)" suffix.
    with pytest.raises(PersistenceError, match="corrupt"):
        load_object(p)


def test_load_integrity_hash_mismatch(tmp_path):
    p = tmp_path / "tampered.etlpers"
    save_object(p, "test.payload", {"x": 1})
    blob = bytearray(p.read_bytes())
    header_len = struct.unpack(">I", blob[11:15])[0]
    payload_start = 15 + header_len + 8
    payload_len = struct.unpack(">Q", blob[15 + header_len : payload_start])[0]
    blob[payload_start + payload_len - 1] ^= 0xFF  # flip the last payload byte
    p.write_bytes(blob)
    with pytest.raises(PersistenceError, match="integrity hash"):
        load_object(p)


# --- codec registration ----------------------------------------------------


def test_register_codec_shadowing():
    # "int" is a built-in codec; shadowing must never be allowed.
    with pytest.raises(PersistenceError, match="already registered"):
        register_codec("int", lambda value: value, lambda data: data)
