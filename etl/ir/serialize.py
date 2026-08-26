"""IR serialization: versioned, self-describing JSON-able payloads.

``serialize_module``/``deserialize_module`` implement the core of the
``.etlgraph`` format: the payload dict returned here is what ``etl.persist``
wraps in its own container (magic header + JSON metadata + integrity hash).
The IR payload itself is already self-describing and integrity-checked —
a SHA-256 over the canonical JSON of its content — so any wrapper stays dumb.

Payload schema (``serialize_module`` result):

    {
      "format": "etl-ir",                 # discriminator
      "version": 1,                       # IR_FORMAT_VERSION
      "module": {"name": str, "metadata": {...}},
      "functions": [
        {
          "name": str,
          "input_types":  [type, ...],    # type = {"dtype": str, "shape": [dim, ...]}
          "output_types": [type, ...],
          "block": {
            "arguments": [{"id": int, "type": type}, ...],
            "ops": [{"id": int, "ref": str}, ...]     # ref into "ops" table
          }
        }, ...
      ],
      "ops": [
        {
          "id": int,
          "name": str,
          "operands": [value_id, ...],
          "results":  [{"id": int, "type": type}, ...],
          "attributes": {...},            # ndarray attrs become {"__etl_ndarray__": "k"}
          "regions":  [region, ...],      # same block encoding as functions
          "location": {"file","line","col","code_snippet"} | null
        }, ...
      ],
      "constants": {"<k>": {"dtype": str, "shape": [int, ...], "data_b64": str}},
      "sha256": "<hex digest>"
    }

Encoding notes (binding):
- ``dim``: ``{"int": n}`` | ``{"dim": "name"}`` (symbolic Dim) |
  ``{"expr": {"op": "<name>", "args": [...]}}`` (compound DimExpr) |
  ``null`` (runtime-dynamic).
- ``dtype``: numpy ``dtype.name`` string (bool/int8/.../float64/complex128).
- Constant payloads: base64 of ``np.save`` bytes — the only numpy involvement.
- The sha256 is computed over ``json.dumps(payload_without_sha256,
  sort_keys=True, separators=(",", ":"))`` (canonical form), and recomputed on
  load: mismatch raises ``VerificationError``; unknown format/version raises
  ``PersistenceError`` (both owned by ``etl.core``) — never a silent
  re-derivation.

ARCHITECTURE PHASE: bodies are ``NotImplementedError`` stubs.
"""

from __future__ import annotations

from .module import Module
from .version import IR_FORMAT_VERSION

__all__ = ["IR_FORMAT_VERSION", "serialize_module", "deserialize_module"]


def serialize_module(module: Module) -> dict:
    """Serialize ``module`` into the self-describing payload dict above.

    Requires a verified module (call ``verify`` first); serialization performs
    no semantic changes.

    Raises:
        VerificationError: If the module fails ``verify``.
    """
    raise NotImplementedError("serialize_module: Phase 2 (implementation)")


def deserialize_module(payload: dict) -> Module:
    """Rebuild a ``Module`` from a payload produced by ``serialize_module``.

    Validates ``format``/``version``, recomputes and compares the sha256,
    rebuilds the IR objects, and runs ``verify`` on the result.

    Raises:
        PersistenceError: Unknown format discriminator or version.
        VerificationError: Integrity hash mismatch or structural violation.
    """
    raise NotImplementedError("deserialize_module: Phase 2 (implementation)")
