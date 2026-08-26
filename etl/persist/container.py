"""Self-describing, integrity-checked, atomically-written save/load container.

`persist.container` is the single on-disk format behind every etl artifact
(``Graph.save``, ``LoweredProgram.save``, ``CompiledArtifact.save``,
``Executable.save``, ``FileCache``). It stores JSON metadata + an encoded
payload dict + a SHA-256 integrity hash.

Byte layout (multi-byte integers are big-endian):

    Offset     Size   Field
    0          7      Magic header: b"ETLPERS"  ("ETL" + format tag "PERS")
    7          4      Format version (uint32) — must equal ETL_FORMAT_VERSION
    11         4      Header length H (uint32)
    15         H      Header JSON (UTF-8; json.dumps(..., sort_keys=True))
    15+H       8      Payload length P (uint64)
    15+H+8     P      Payload bytes (UTF-8 JSON of the encoded payload dict)
    end        64     SHA-256 digest of the PAYLOAD bytes (ASCII hex, lowercase)

Header JSON fields::

    {
        "format_version": int,          # == ETL_FORMAT_VERSION
        "payload_type": str,            # caller-declared kind, e.g. "etl.graph"
        "backend_info": dict|None,      # backend name/version/target/ABI, JSON-safe as passed
        "signature_info": dict|None,    # input/output trees + specs + static values, JSON-safe as passed
        "hash_algorithm": "sha256",
    }

Integrity rules (binding):
  * The hash covers the payload bytes ONLY. Header corruption is caught by
    magic/version/JSON-structure validation.
  * ``backend_info`` / ``signature_info`` must already be JSON-safe when
    passed in (callers pre-serialize via ``codec.encode_value``, e.g.
    pipeline serializes TreeSpecs/specs before calling). The payload dict,
    in contrast, is encoded automatically via ``codec.encode_value``.

Atomicity: ``save_object`` writes to a temp file in the destination
directory (``tempfile.NamedTemporaryFile(dir=..., delete=False)``), fsyncs
it, then ``os.replace``s it over the destination — readers never observe a
half-written file. On any failure the temp file is unlinked.

Semantics (binding): this module only moves bytes and JSON. It has no
access to trace/backends and therefore can NEVER silently re-trace or
re-compile anything. Loading produces a plain decoded payload dict;
reconstructing live objects (e.g. backend executables with device-specific
handles) is always the caller's explicit step, and any mismatch fails with
``PersistenceError``.

Import rule (binding): may import ``etl.core`` ONLY (PersistenceError)
plus stdlib and the sibling ``.codec`` — no other etl modules.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import tempfile

from etl.core import PersistenceError

from .codec import decode_value, encode_value

ETL_FORMAT_VERSION = 1
"""Container format version. Bump on any breaking layout/codec change."""

MAGIC = b"ETLPERS"
"""Magic header: "ETL" + format tag "PERS" (7 bytes)."""

HASH_ALGORITHM = "sha256"
"""Integrity hash algorithm recorded in the header."""


class LoadedObject(dict):
    """Result of ``load_object``: a dict with both key and attribute access.

    Keys (always present): "payload", "payload_type", "format_version";
    "backend_info" and "signature_info" hold their loaded values (which may
    be None).

    This is a plain data holder — no hidden behavior, no lazy loading, no
    re-tracing.
    """

    def __init__(
        self,
        *,
        payload,
        payload_type,
        backend_info=None,
        signature_info=None,
        format_version=ETL_FORMAT_VERSION,
    ):
        super().__init__(
            payload=payload,
            payload_type=payload_type,
            backend_info=backend_info,
            signature_info=signature_info,
            format_version=format_version,
        )

    @property
    def payload(self):
        """The decoded payload dict."""
        return self["payload"]

    @property
    def payload_type(self):
        """The caller-declared payload kind string."""
        return self["payload_type"]

    @property
    def backend_info(self):
        """Backend name/version/target/ABI metadata (or None)."""
        return self["backend_info"]

    @property
    def signature_info(self):
        """Serialized input/output trees + specs + static values (or None)."""
        return self["signature_info"]

    @property
    def format_version(self):
        """Container format version the file was written with."""
        return self["format_version"]


def _compute_payload_hash(payload_bytes: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``payload_bytes`` (64 chars).

    Uses ``hashlib.sha256`` (the only supported algorithm in v1).
    """
    return hashlib.sha256(payload_bytes).hexdigest()


def _encode_header(payload_type: str, backend_info, signature_info) -> bytes:
    """Serialize the header JSON to UTF-8 bytes.

    Builds the dict described in the module docstring and returns
    ``json.dumps(header, sort_keys=True).encode("utf-8")``. Raises
    ``PersistenceError`` if ``payload_type`` is not a non-empty string or
    ``backend_info``/``signature_info`` are not JSON-serializable.
    """
    if not isinstance(payload_type, str) or not payload_type:
        raise PersistenceError(
            f"payload_type must be a non-empty string, got "
            f"{type(payload_type).__name__}: {payload_type!r}"
        )
    header = {
        "format_version": ETL_FORMAT_VERSION,
        "payload_type": payload_type,
        "backend_info": backend_info,
        "signature_info": signature_info,
        "hash_algorithm": HASH_ALGORITHM,
    }
    try:
        return json.dumps(header, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"header is not JSON-serializable: {exc}") from exc


def _decode_header(header_bytes: bytes) -> dict:
    """Parse and validate the header JSON.

    Raises ``PersistenceError`` ("corrupt") on invalid JSON, a non-dict
    result, or missing/invalid required fields: ``format_version`` (int),
    ``payload_type`` (non-empty str), ``hash_algorithm`` (== HASH_ALGORITHM).
    """
    try:
        header = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PersistenceError(f"corrupt: header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise PersistenceError(
            "corrupt: header must be a JSON object, got "
            f"{type(header).__name__}"
        )
    if not isinstance(header.get("format_version"), int):
        raise PersistenceError(
            "corrupt: header field 'format_version' is missing or not an int"
        )
    payload_type = header.get("payload_type")
    if not isinstance(payload_type, str) or not payload_type:
        raise PersistenceError(
            "corrupt: header field 'payload_type' is missing or not a "
            "non-empty string"
        )
    hash_algorithm = header.get("hash_algorithm")
    if hash_algorithm != HASH_ALGORITHM:
        raise PersistenceError(
            f"corrupt: unsupported hash algorithm {hash_algorithm!r} "
            f"(expected {HASH_ALGORITHM!r})"
        )
    # backend_info / signature_info are optional: keep whatever is present
    # (including None) — validation of their content is the caller's job.
    return header


def save_object(path, payload_type, payload_fields, backend_info=None, signature_info=None):
    """Write the standard container to ``path`` (atomic).

    Algorithm:
      1. ``encoded = codec.encode_value(payload_fields)``.
      2. payload bytes = ``json.dumps(encoded, sort_keys=True).encode("utf-8")``.
      3. header bytes = ``_encode_header(payload_type, backend_info, signature_info)``.
      4. Assemble: MAGIC || uint32be(ETL_FORMAT_VERSION) || uint32be(len(H)) ||
         header || uint64be(len(P)) || payload || ascii hexdigest(payload bytes).
      5. Atomic write: create parent directories (``os.makedirs(..., exist_ok=True)``),
         write everything to a ``NamedTemporaryFile`` in the destination
         directory (``delete=False``), flush + ``os.fsync``, then
         ``os.replace`` onto ``path``; unlink the temp file on any error.

    Raises ``PersistenceError`` for non-encodable payload values (via the
    codec) or invalid metadata.
    """
    encoded = encode_value(payload_fields)
    try:
        payload_bytes = json.dumps(encoded, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PersistenceError(
            f"encoded payload is not JSON-serializable: {exc}"
        ) from exc
    header = _encode_header(payload_type, backend_info, signature_info)
    blob = (
        MAGIC
        + struct.pack(">I", ETL_FORMAT_VERSION)
        + struct.pack(">I", len(header))
        + header
        + struct.pack(">Q", len(payload_bytes))
        + payload_bytes
        + _compute_payload_hash(payload_bytes).encode("ascii")
    )

    dest_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(dest_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=dest_dir, prefix=".etlpers-", suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(blob)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load_object(path, expected_payload_type=None):
    """Read + validate the container at ``path``; return a ``LoadedObject``.

    Algorithm (strict — any failure raises, nothing is silently repaired):
      1. Read all bytes (binary).
      2. Magic check: bytes[:7] != MAGIC -> ``PersistenceError`` ("corrupt").
      3. Version check: stored version > ETL_FORMAT_VERSION ->
         ``PersistenceError`` ("newer format"); any other version mismatch
         -> "corrupt".
      4. Parse the header via ``_decode_header`` (structure errors -> "corrupt").
      5. Read payload length + payload bytes (bounds-checked).
      6. Recompute SHA-256 over the payload bytes and compare with the
         stored digest — mismatch -> ``PersistenceError`` ("corrupt"); never
         a partial or silent load.
      7. Decode the payload via ``codec.decode_value``; the result must be
         a dict, otherwise "corrupt".
      8. If ``expected_payload_type`` is given and differs from the stored
         ``payload_type`` -> ``PersistenceError`` (type mismatch; never a
         silent reinterpretation).
      9. Return ``LoadedObject(payload=..., payload_type=..., backend_info=...,
         signature_info=..., format_version=...)``.

    Never re-traces/recompiles: this module cannot (no trace/backend
    access) and does not — the caller explicitly reconstructs any live
    object from the returned payload.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 15:
        raise PersistenceError(
            "corrupt: file is too short to contain a container header"
        )
    if data[:7] != MAGIC:
        raise PersistenceError("corrupt: bad magic header")
    version = struct.unpack(">I", data[7:11])[0]
    if version > ETL_FORMAT_VERSION:
        raise PersistenceError(
            f"newer format: file format version {version} is newer than this "
            f"library supports ({ETL_FORMAT_VERSION}); upgrade etl or re-save "
            "the artifact with the current version"
        )
    if version != ETL_FORMAT_VERSION:
        raise PersistenceError(f"corrupt: unknown format version {version}")

    try:
        header_len = struct.unpack(">I", data[11:15])[0]
    except struct.error as exc:
        raise PersistenceError("corrupt: cannot parse header length") from exc
    header_end = 15 + header_len
    if header_end > len(data):
        raise PersistenceError("corrupt: header length exceeds file size")
    header = _decode_header(data[15:header_end])

    payload_len_offset = header_end + 8
    try:
        payload_len = struct.unpack(">Q", data[header_end:payload_len_offset])[0]
    except struct.error as exc:
        raise PersistenceError("corrupt: cannot parse payload length") from exc
    payload_end = payload_len_offset + payload_len
    # Exact total length: header || payload length || payload || 64-byte
    # digest. Any extra or missing bytes mean corruption.
    if payload_end + 64 != len(data):
        raise PersistenceError(
            "corrupt: payload length does not match the file size"
        )
    payload_bytes = data[payload_len_offset:payload_end]

    try:
        stored_digest = data[payload_end:].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PersistenceError("corrupt: stored digest is not ASCII") from exc
    if not hmac.compare_digest(stored_digest, _compute_payload_hash(payload_bytes)):
        raise PersistenceError("corrupt: integrity hash mismatch")

    try:
        encoded = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PersistenceError("corrupt: payload is not valid JSON") from exc
    try:
        payload = decode_value(encoded)
    except PersistenceError as exc:
        raise PersistenceError(f"{exc} (corrupt)") from exc
    if not isinstance(payload, dict):
        raise PersistenceError(
            f"corrupt: payload must decode to a dict, got "
            f"{type(payload).__name__}"
        )

    payload_type = header["payload_type"]
    if expected_payload_type is not None and expected_payload_type != payload_type:
        raise PersistenceError(
            f"payload_type mismatch: expected {expected_payload_type!r}, "
            f"file contains {payload_type!r}"
        )

    return LoadedObject(
        payload=payload,
        payload_type=payload_type,
        backend_info=header.get("backend_info"),
        signature_info=header.get("signature_info"),
        format_version=version,
    )
