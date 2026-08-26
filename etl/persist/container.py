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
import json
import os
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
    raise NotImplementedError(
        "persist.container._compute_payload_hash: architecture stub — "
        "implementation lands in Phase 2"
    )


def _encode_header(payload_type: str, backend_info, signature_info) -> bytes:
    """Serialize the header JSON to UTF-8 bytes.

    Builds the dict described in the module docstring and returns
    ``json.dumps(header, sort_keys=True).encode("utf-8")``. Raises
    ``PersistenceError`` if ``payload_type`` is not a non-empty string or
    ``backend_info``/``signature_info`` are not JSON-serializable.
    """
    raise NotImplementedError(
        "persist.container._encode_header: architecture stub — "
        "implementation lands in Phase 2"
    )


def _decode_header(header_bytes: bytes) -> dict:
    """Parse and validate the header JSON.

    Raises ``PersistenceError`` ("corrupt") on invalid JSON, a non-dict
    result, or missing/invalid required fields: ``format_version`` (int),
    ``payload_type`` (non-empty str), ``hash_algorithm`` (== HASH_ALGORITHM).
    """
    raise NotImplementedError(
        "persist.container._decode_header: architecture stub — "
        "implementation lands in Phase 2"
    )


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
    raise NotImplementedError(
        "persist.container.save_object: architecture stub — "
        "implementation lands in Phase 2"
    )


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
    raise NotImplementedError(
        "persist.container.load_object: architecture stub — "
        "implementation lands in Phase 2"
    )
