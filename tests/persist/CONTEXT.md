# tests/persist — etl persistence layer test suite

## Intent

pytest suite validating `etl.persist` (sibling — see `../../etl/persist/CONTEXT.md` for the contract under test): the value codec, the on-disk container format, error behavior, the explicit cache, and package-level integration. Tests are the executable spec of the documented contract (module docstrings in `etl/persist/*.py` + `etl/persist/CONTEXT.md`).

## Structure

| File | Covers |
|---|---|
| `test_codec.py` | `encode_value`/`decode_value` round-trips for all 18 built-in codec types, error cases, `register_codec` extension + shadowing, cycle detection, non-mutation/JSON-serializability invariants |
| `test_container.py` | `save_object`/`load_object`: exact byte layout, SHA-256 integrity (tamper/truncate/garbage), format versioning, payload_type expectations, metadata round-trips, atomicity, auto-created parent dirs |
| `test_persistence_errors.py` | `PersistenceError` hierarchy, I/O failures (missing file → `FileNotFoundError`; directory → `OSError` — the contract's PersistenceError enumeration covers content validation only), hand-crafted corrupt containers |
| `test_cache.py` | `Cache` ABC + `FileCache`: hit/miss/recompute-on-corrupt semantics, key derivation (`compute_key`), distinct entries, cross-instance persistence, clear, directory auto-creation |
| `test_integration.py` | smoke tests through the public `etl` package (artifact-like payload round-trip, `etl.Cache`/`etl.FileCache` exports) |

## Constraints

- CPU only, fast (<2s per file, suite runs in ~0.5s), all file I/O via `tmp_path`.
- Read-only access to `etl/` — never modify the package here (bug fixes belong to the parent).
- Tests assert the documented contract. When etl contradicts its own contract, keep the test FAILING with a `# BUG(etl): <description>` comment — do not weaken/skip/xfail.

## Notes for agents

- Custom classes defined inside test functions are not importable by qualname — TreeSpec codec tests only use built-in container node types (tuple/list/dict) + python leaves.
- `register_codec` mutates a global registry: codec-extension tests must pop their registration in a `finally` block (`codec_mod._CODECS.pop(name, None)`).
- Decoded numpy arrays are read-only (`writeable == False`) and fresh copies — compare values with NaN-aware equality (`np.array_equal(..., equal_nan=True)`), never strides (C-order on decode is fine).
- `pytest.raises(PersistenceError, match=...)` messages come from `etl/persist/*.py` ("corrupt: ...", "newer format", "payload_type mismatch", "already registered", "cyclic").
