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
import shutil

from etl.core import PersistenceError

from .codec import ENVELOPE_TAG, encode_value
from .container import ETL_FORMAT_VERSION, load_object, save_object

_CACHE_PAYLOAD_TYPE = "FileCacheEntry"
"""``payload_type`` used for every FileCache container."""

_SHARD_DIR_CHARS = 2
"""Number of leading key chars used as a shard subdirectory (256 shards)."""

_CACHE_SUFFIX = ".etlcache"
"""File suffix for cache entries."""


def _canonical_json(value):
    """Return the canonical JSON text for ``value`` (sorted keys, compact)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonicalize_encoded(value):
    """Return a canonical copy of an encoded value (never mutates the input).

    The codec's ``"dict"`` envelope stores its items as an insertion-ordered
    list (``data["items"]``), and ``json.dumps(sort_keys=True)`` cannot
    reorder list elements — so that list is sorted here by the canonical
    JSON of each (recursively canonicalized) ``[encoded_key, encoded_value]``
    pair. All other dicts are shallow-copied with recursively canonicalized
    values (their own key order is normalized later by the sorted-JSON dump);
    lists are recursively copied; scalars pass through unchanged.
    """
    if isinstance(value, list):
        return [_canonicalize_encoded(element) for element in value]
    if isinstance(value, dict):
        if (
            value.get(ENVELOPE_TAG) is True
            and value.get("type") == "dict"
            and isinstance(value.get("data"), dict)
            and isinstance(value["data"].get("items"), list)
        ):
            items = value["data"]["items"]
            canonical_items = sorted(
                (_canonicalize_encoded(pair) for pair in items),
                key=_canonical_json,
            )
            return {ENVELOPE_TAG: True, "type": "dict", "data": {"items": canonical_items}}
        return {key: _canonicalize_encoded(val) for key, val in value.items()}
    return value


def _components_tag(key_components):
    """Return the codec-style type name of the components container.

    Type-tags the container so a bare string component list and a
    one-element sequence can never collide.
    """
    if isinstance(key_components, str):
        return "str"
    if isinstance(key_components, dict):
        return "dict"
    if isinstance(key_components, tuple):
        return "tuple"
    if isinstance(key_components, list):
        return "list"
    return "{}.{}".format(type(key_components).__module__, type(key_components).__qualname__)


def compute_key(key_components):
    """Derive the deterministic cache key (64-char lowercase sha256 hex).

    Algorithm:
      1. Materialize the component list. A bare ``str`` is treated as ONE
         component (never split into characters); a ``dict`` iterates its
         keys SORTED by the canonical JSON of their encoded form (dict
         insertion order never matters at any level); any other iterable
         is taken in order (component order is significant).
      2. Encode every component via ``codec.encode_value`` and canonicalize
         the result (numpy arrays, specs, dims, dicts all normalize
         deterministically; encoded dict-envelope item lists are sorted by
         canonical JSON so insertion order cannot leak into the key).
      3. Structure = ``[ETL_FORMAT_VERSION, container_tag, [encoded
         component, ...]]`` — ``container_tag`` type-tags the components
         container ("str"/"tuple"/"list"/"dict" or a qualified type name)
         so a bare string and a one-element sequence derive different
         keys; the format version is part of the key, so a format bump
         invalidates every old entry.
      4. Canonical JSON: ``json.dumps(structure, sort_keys=True,
         separators=(",", ":"), ensure_ascii=True)`` encoded to UTF-8.
      5. Key = ``hashlib.sha256(json_bytes).hexdigest()``.

    Keying contract (binding): ``key_components`` MUST include every input
    that affects the computed value. The canonical inputs per design doc
    21.2: graph bytes, frontend/IR version, static values, input/output
    signatures, backend (name + version), compiler version/options, target,
    required custom ops, runtime ABI. Omitting any of these makes the cache
    silently unsound — callers assemble the list explicitly; persist never
    guesses components.
    """
    if isinstance(key_components, str):
        components = [key_components]  # bare string = ONE component
    elif isinstance(key_components, dict):
        components = sorted(
            key_components, key=lambda k: _canonical_json(encode_value(k))
        )
    else:
        components = list(key_components)  # any iterable; order significant
    encoded_components = _canonicalize_encoded([encode_value(c) for c in components])
    structure = [ETL_FORMAT_VERSION, _components_tag(key_components), encoded_components]
    json_bytes = _canonical_json(structure).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


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
            "persist.cache.Cache.get: abstract method — subclasses must override"
        )

    @abc.abstractmethod
    def put(self, key_components, value):
        """Store ``value`` under the key derived from ``key_components``.

        Overwrites any existing entry atomically (temp file + rename).
        """
        raise NotImplementedError(
            "persist.cache.Cache.put: abstract method — subclasses must override"
        )

    @abc.abstractmethod
    def contains(self, key_components):
        """Return True if an entry exists for ``key_components``.

        Existence check only — no integrity validation (documented O(1)).
        """
        raise NotImplementedError(
            "persist.cache.Cache.contains: abstract method — subclasses must override"
        )

    @abc.abstractmethod
    def clear(self):
        """Remove ALL entries from this cache instance."""
        raise NotImplementedError(
            "persist.cache.Cache.clear: abstract method — subclasses must override"
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
        key = compute_key(key_components)
        path = self._path_for_key(key)
        if not os.path.isfile(path):
            # Miss: recompute only when a compute_fn is supplied.
            if compute_fn is None:
                return None
            value = compute_fn()
            self.put(key_components, value)
            return value
        try:
            loaded = load_object(path, expected_payload_type=_CACHE_PAYLOAD_TYPE)
            value = loaded.payload["value"]
        except (PersistenceError, OSError, KeyError, TypeError, ValueError):
            # Corrupt / version-mismatched / type-mismatched entry: treat
            # as a miss. Never propagate corrupt-file errors.
            if compute_fn is None:
                return None
            value = compute_fn()
            self.put(key_components, value)  # atomically replaces the bad file
            return value
        return value

    def put(self, key_components, value):
        """``save_object(_path_for_key(compute_key(key_components)),
        _CACHE_PAYLOAD_TYPE, {"value": value})``.

        The parent shard directory is created as needed; overwrite is
        atomic (temp + ``os.replace``).
        """
        save_object(
            self._path_for_key(compute_key(key_components)),
            _CACHE_PAYLOAD_TYPE,
            {"value": value},
        )

    def contains(self, key_components):
        """``os.path.isfile(_path_for_key(compute_key(key_components)))``.

        Existence check only — no integrity validation (documented O(1)).
        """
        return os.path.isfile(self._path_for_key(compute_key(key_components)))

    def clear(self):
        """Delete every ``*.etlcache`` entry under the cache directory and
        recreate the directory empty. Not safe for concurrent use."""
        shutil.rmtree(self._directory, ignore_errors=True)
        os.makedirs(self._directory, exist_ok=True)
