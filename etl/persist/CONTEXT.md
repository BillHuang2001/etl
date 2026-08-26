# etl.persist — versioned/self-describing/integrity-checked container + explicit cache

## Intent

The persistence layer of etl: the single on-disk container format behind
every artifact (`Graph.save`, `LoweredProgram.save`, `CompiledArtifact.save`,
`Executable.save`, `FileCache`), a JSON-safe value codec, and an explicit
user-owned cache. It moves bytes and JSON ONLY — it contains no trace or
backend logic, so it can never (and does not) silently re-trace or
re-compile anything. Inherited principles: explicit staging, minimal
magic, explicit caching (no global cache), errors never swallowed.

## API Surface

| Name | Kind | Notes |
|---|---|---|
| `ETL_FORMAT_VERSION` | int (= 1) | container format version; also part of every cache key |
| `save_object(path, payload_type, payload_fields, backend_info=None, signature_info=None)` | fn | atomic write; payload fields auto-encoded |
| `load_object(path, expected_payload_type=None) -> LoadedObject` | fn | strict validation; never repairs silently |
| `LoadedObject` | dict subclass | keys `payload`/`payload_type`/`backend_info`/`signature_info`/`format_version` + property access |
| `encode_value(value) -> dict` / `decode_value(dict) -> value` | fns | envelope codec (see Codec below) |
| `register_codec(type_name, encoder, decoder)` | fn | documented extension point for custom types |
| `compute_key(key_components) -> str` | fn | sha256 hex key derivation, shared by cache implementations |
| `Cache` | ABC | `get`/`put`/`contains`/`clear` + `get_or_compute` alias |
| `FileCache(directory)` | class | one container file per key, sharded |

`Cache.get(key_components, compute_fn=None)`: hit → value; miss/corrupt +
compute_fn → recompute, store, return; miss/corrupt without compute_fn →
None. `get_or_compute(key_components, compute_fn)` is an alias delegating
to `get(..., compute_fn=...)` (compatibility with the package-level
contract name).

## Dependencies & import rules (binding)

- May import `etl.core` ONLY — `PersistenceError`, and (for the codec)
  `Dim`/`DimExpr`/`Device`/`TensorSpec`/`TreeSpec`; optionally `Tensor` if
  a constant-payload codec is ever registered. No other etl modules. This
  is deliberately stricter than the root contract's "core, pipeline types"
  note and guarantees no import cycle: pipeline/backends/trace call INTO
  persist with plain data (payload_type strings, pre-serialized signature
  dicts), never the reverse.
- Stdlib: json, base64, hashlib, os, tempfile, io, abc. Third-party: numpy only.
- Never imports the `etl` package root (`etl/__init__.py`) — keep import-safe.

## Container layout (container.py, binding)

Byte layout, multi-byte integers big-endian:

| Offset | Size | Field |
|---|---|---|
| 0 | 7 | Magic `b"ETLPERS"` ("ETL" + format tag "PERS") |
| 7 | 4 | Format version (uint32, == `ETL_FORMAT_VERSION`) |
| 11 | 4 | Header length H (uint32) |
| 15 | H | Header JSON UTF-8 (`sort_keys=True`) |
| 15+H | 8 | Payload length P (uint64) |
| 15+H+8 | P | Payload bytes (UTF-8 JSON of the encoded payload dict) |
| end | 64 | SHA-256 hex digest (lowercase) of the PAYLOAD bytes |

Header JSON fields: `format_version` (int), `payload_type` (str),
`backend_info` (dict|None), `signature_info` (dict|None),
`hash_algorithm` ("sha256").

Rules: the hash covers the payload bytes ONLY (header corruption is caught
by magic/version/JSON validation). `backend_info`/`signature_info` must be
JSON-safe as passed (callers pre-serialize TreeSpecs/specs via the codec);
`payload_fields` are encoded automatically. Writes are atomic: temp file in
the destination directory → fsync → `os.replace`; readers never see a
half-written file.

## Codec (codec.py)

Envelope: `{"__etl_encoded__": true, "type": <name>, "data": <json-safe>}`;
`decode_value` REQUIRES the envelope (no identity fallback). The
authoritative registry table lives in codec.py's module docstring:
`NoneType`, `bool`, `int`, `float`, `str`, `complex`, `list`, `tuple`,
`dict`, `numpy.ndarray` (base64 npy, `allow_pickle=False`), `numpy.generic`,
`numpy.dtype`, `slice`, `Dim`, `DimExpr`, `Device`, `TensorSpec`,
`TreeSpec`. Decoded numpy arrays are read-only. Unknown type / cyclic
value → `PersistenceError`. Extension: `register_codec` before any
save/load that may hit the type; shadowing registered names is rejected.
TreeSpec round-trip requires `core.TreeSpec` to be reconstructible from
(treetype name, context, node_data, children) — the codec calls the
dataclass constructor directly; treetype names must resolve (see Notes for
agents).

## Cache (cache.py)

- Key = sha256 hex of canonical JSON (`sort_keys`, compact separators) of
  `[ETL_FORMAT_VERSION, [encoded components]]` — a format bump invalidates
  all entries.
- Keying contract (binding): components MUST include every input affecting
  the value — graph bytes, frontend/IR version, static values, signatures,
  backend name+version, compiler version/options, target, custom ops,
  runtime ABI. Callers assemble the list; persist never guesses.
- FileCache layout: `<directory>/<key[:2]>/<key>.etlcache`, one
  `save_object` container per entry (`payload_type="FileCacheEntry"`,
  payload `{"value": value}`). Missing/corrupt entries: recompute when
  `compute_fn` given (atomic overwrite), else miss (None). `contains` is
  an existence check only. `clear()` wipes the whole cache directory.
- NO global cache anywhere — always an explicit user object.

## Error behavior (binding)

All failures raise `PersistenceError` (from core): bad magic, newer format
version, invalid header/payload JSON, integrity hash mismatch ("corrupt"),
payload_type mismatch, unknown codec type, cyclic value, codec shadowing.
Error messages are specific. Nothing is silently repaired or recomputed
except FileCache's documented compute_fn policy.

## Test strategy

pytest under `../tests/persist/` (sibling, read-only from here — escalate
test writes to root): container round-trips (every codec type incl.
symbolic dims), truncated/bit-flipped files → "corrupt", version bump →
"newer format", payload_type mismatch, atomicity (no partial file on
failed write), cache hit/miss/recompute-on-corrupt/clear, key determinism
+ sensitivity to every keying input. Cross-checked by
`../tests/test_spec_compliance.py` (serialization round-trips).

## Routing table

| Path | Area |
|---|---|
| `./container.py` | `save_object`/`load_object`, `LoadedObject`, magic/version/hash constants |
| `./codec.py` | envelope codec, registry, `encode_value`/`decode_value` |
| `./cache.py` | `compute_key`, `Cache` ABC, `FileCache` |
| `./__init__.py` | re-exports (this exact surface) |

## Status

Fully implemented — no stubs remain anywhere in this directory (the only
`NotImplementedError` bodies are the four intentional `Cache` ABC abstract
methods). `codec.py` (18 built-in codec pairs, envelope dispatch with
exact-type + qualified-name fallback, cycle detection, read-only decoded
arrays), `container.py` (byte-layout save/load, atomic writes, strict
validation), `cache.py` (`compute_key`, `FileCache`). Validated end-to-end
(codec→container→cache round-trips, tamper detection, version mismatch,
atomicity, recompute-on-corrupt) via inline scripts; the pytest suite in
`../tests/persist/` does not exist yet (owned by root).

## Notes for agents

- Canonical `save_object` signature is `(path, payload_type,
  payload_fields, backend_info=None, signature_info=None)`. The
  package-level bullet in `../CONTEXT.md` currently shows an older
  argument order — pipeline-type `.save()` methods must be written against
  the signature here.
- `_install_builtin_codecs()` runs unconditionally at import time
  (idempotent) — importing `etl.persist` requires `etl.core` to be
  importable.
- TreeSpec codec: the `type` field is stored as a name — `tuple`/`list`/
  `dict` short names or `"<module>.<qualname>"` resolved via importlib.
  Consequently a TreeSpec whose type is a namedtuple/dataclass/custom class
  defined inside a function (non-importable qualname) encodes fine but
  fails to DECODE with `PersistenceError` — keep pytree node types
  module-level if artifacts must round-trip.
- Decoded numpy arrays are read-only and never alias the file bytes;
  `numpy.generic` decodes back to the numpy scalar (not an array).
- FileCache corrupt entries are never propagated — treated as a miss
  (recompute + atomic overwrite when `compute_fn` is given, else `None`).
  This is the only place errors are "swallowed", and it is the documented
  cache policy, not silent magic.
