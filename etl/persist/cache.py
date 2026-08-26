"""Explicit cache: deterministic key derivation + ``Cache`` interface + ``FileCache``.

There is NO global cache anywhere in etl — caches are always explicit
user-created objects (design principle 3: "Caching/binding are explicit
operations"). Callers create a ``FileCache`` and pass it explicitly to
whatever consumes it.

Import rule (binding): may import ``etl.core`` ONLY (PersistenceError)
plus stdlib and sibling persist modules.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os

from etl.core import PersistenceError

from .codec import encode_value
from .container import ETL_FORMAT_VERSION, load_object, save_object

_CACHE_PAYLOAD_TYPE = "FileCacheEntry"
"""``payload_type`` used for every FileCache container."""

_SHARD_DIR_CHARS = 2
"""Number of leading key chars used as a shard subdirectory (256 shards)."""

_CACHE_SUFFIX = ".etlcache"
"""File suffix for cache entries."""


def compute_key(key_components):
    """Derive the deterministic cache key (64-char lowercase sha256 hex).

    Algorithm:
      1. Encode every component via ``codec.encode_value`` (numpy arrays,
         specs, dims, dicts all normalize deterministically).
      2. Structure = ``[ETL_FORMAT_VERSION, [encoded component, ...]]`` —
         the format version is part of the key, so a format bump
         invalidates every old entry.
      3. Canonical JSON: ``json.dumps(structure, sort_keys=True,
         separators=(",", ":"), ensure_ascii=True)`` encoded to UTF-8.
      4. Key = ``hashlib.sha256(json_bytes).hexdigest()``.

    Keying contract (binding): ``key_components`` MUST include every input
    that affects the computed value. The canonical inputs per design doc
    21.2: graph bytes, frontend/IR version, static values, input/output
    signatures, backend (name + version), compiler version/options, target,
    required custom ops, runtime ABI. Omitting any of these makes the cache
    silently unsound — callers assemble the list explicitly; persist never
    guesses components.
    """
    raise NotImplementedError(
        "persist.cache.compute_key: architecture stub — "
        "implementation lands in Phase 2"
    )


class Cache(abc.ABC):
    """Explicit cache interface (no global state; instances are user-owned)."""

    @abc.abstractmethod
    def get(self, key_components, compute_fn=None):
        """Fetch the value stored under ``key_components``.

        ``compute_fn`` is an optional zero-arg callable. Semantics:
          * hit                          -> return the stored value
          * miss + compute_fn None       -> return None
          * miss + compute_fn given      -> ``value = compute_fn()``; store
                                            it (``put``) and return it
          * corrupt entry + compute_fn   -> recompute and atomically
                                            overwrite the corrupt entry
          * corrupt entry, no compute_fn -> treated as a miss (None)
        """
        raise NotImplementedError(
            "persist.cache.Cache.get: architecture stub — "
            "implementation lands in Phase 2"
        )

    @abc.abstractmethod
    def put(self, key_components, value):
        """Store ``value`` under the key derived from ``key_components``.

        Overwrites any existing entry atomically (temp file + rename).
        """
        raise NotImplementedError(
            "persist.cache.Cache.put: architecture stub — "
            "implementation lands in Phase 2"
        )

    @abc.abstractmethod
    def contains(self, key_components):
        """Return True if an entry exists for ``key_components``.

        Existence check only — no integrity validation (documented O(1)).
        """
        raise NotImplementedError(
            "persist.cache.Cache.contains: architecture stub — "
            "implementation lands in Phase 2"
        )

    @abc.abstractmethod
    def clear(self):
        """Remove ALL entries from this cache instance."""
        raise NotImplementedError(
            "persist.cache.Cache.clear: architecture stub — "
            "implementation lands in Phase 2"
        )

    def get_or_compute(self, key_components, compute_fn):
        """Alias for ``get(key_components, compute_fn=compute_fn)``.

        Kept for compatibility with the package-level contract name;
        ``get``'s ``compute_fn`` argument is the canonical spelling.
        """
        return self.get(key_components, compute_fn=compute_fn)


class FileCache(Cache):
    """Filesystem-backed ``Cache``: one container file per key, sharded.

    Layout: ``<directory>/<key[:2]>/<key>.etlcache`` — entries are stored
    via ``save_object`` with ``payload_type="FileCacheEntry"`` and payload
    ``{"value": value}`` (the value is encoded by the codec).

    Failure policy (explicit, no silent magic):
      * missing entries are recomputed when ``compute_fn`` is given;
      * corrupt / version-mismatched / type-mismatched entries are treated
        as misses and recomputed (atomically overwritten) when
        ``compute_fn`` is given, otherwise treated as a miss (None);
      * every write is atomic (temp file + ``os.replace``), so a crash
        never leaves a torn entry.
    """

    def __init__(self, directory):
        """Create/use ``directory`` as the cache root (``makedirs exist_ok=True``)."""
        self._directory = os.path.abspath(directory)
        os.makedirs(self._directory, exist_ok=True)

    def _path_for_key(self, key):
        """Map a 64-char hex key to ``<directory>/<key[:2]>/<key>.etlcache``."""
        shard = key[:_SHARD_DIR_CHARS]
        return os.path.join(self._directory, shard, key + _CACHE_SUFFIX)

    def get(self, key_components, compute_fn=None):
        """See ``Cache.get``.

        Implementation against ``compute_key`` + ``load_object``:
        ``key = compute_key(key_components)``; on hit,
        ``load_object(path, expected_payload_type=_CACHE_PAYLOAD_TYPE)``
        and return ``payload["value"]``; on miss or ``PersistenceError``
        from loading (corrupt/version/type), return None — or recompute via
        ``compute_fn()`` followed by ``put`` (which atomically replaces the
        bad file).
        """
        raise NotImplementedError(
            "persist.cache.FileCache.get: architecture stub — "
            "implementation lands in Phase 2"
        )

    def put(self, key_components, value):
        """``save_object(_path_for_key(compute_key(key_components)),
        _CACHE_PAYLOAD_TYPE, {"value": value})``.

        The parent shard directory is created as needed; overwrite is
        atomic (temp + ``os.replace``).
        """
        raise NotImplementedError(
            "persist.cache.FileCache.put: architecture stub — "
            "implementation lands in Phase 2"
        )

    def contains(self, key_components):
        """``os.path.isfile(_path_for_key(compute_key(key_components)))``.

        Existence check only — no integrity validation (documented O(1)).
        """
        raise NotImplementedError(
            "persist.cache.FileCache.contains: architecture stub — "
            "implementation lands in Phase 2"
        )

    def clear(self):
        """Delete every ``*.etlcache`` entry under the cache directory and
        recreate the directory empty. Not safe for concurrent use."""
        raise NotImplementedError(
            "persist.cache.FileCache.clear: architecture stub — "
            "implementation lands in Phase 2"
        )
