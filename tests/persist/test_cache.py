"""Tests for the explicit cache: ``Cache`` ABC, ``FileCache``, ``compute_key``.

Contract under test (etl/persist/cache.py):
  * No global cache — caches are explicit, user-created objects.
  * ``compute_key`` is a deterministic 64-char lowercase sha256 hex digest,
    insensitive to dict key insertion order (canonical JSON), sensitive to
    every component change.
  * ``FileCache`` stores one ``save_object`` container per key at
    ``<dir>/<key[:2]>/<key>.etlcache`` with ``payload_type="FileCacheEntry"``.
  * ``get``: hit -> value; miss/corrupt + compute_fn -> recompute+store;
    miss/corrupt without compute_fn -> None; corrupt-entry errors never
    propagate.
"""

import glob

import numpy as np
import pytest

import etl
from etl.persist import Cache, FileCache, compute_key, load_object


def counting_fn(value):
    """Return ``(counter, fn)``: fn() bumps ``counter["calls"]`` and returns value."""
    counter = {"calls": 0}

    def compute():
        counter["calls"] += 1
        return value

    return counter, compute


# ---------------------------------------------------------------------------
# Exports and ABC contract
# ---------------------------------------------------------------------------


def test_cache_exports_and_abc():
    assert etl.Cache is Cache
    assert etl.FileCache is FileCache
    with pytest.raises(TypeError):
        Cache()  # abstract: get/put/contains/clear


# ---------------------------------------------------------------------------
# get / put / get_or_compute semantics
# ---------------------------------------------------------------------------


def test_get_miss_with_compute_fn_stores_entry(tmp_path):
    cache = FileCache(tmp_path / "cache")
    key_components = ("graph", "hash123")
    counter, fn = counting_fn("computed-value")

    result = cache.get(key_components, compute_fn=fn)

    assert result == "computed-value"
    assert counter["calls"] == 1  # computed exactly once
    # Entry lives at the sharded path for the derived key.
    key = compute_key(key_components)
    entry_path = tmp_path / "cache" / key[:2] / (key + ".etlcache")
    assert entry_path.is_file()
    # It is a save_object container with payload_type "FileCacheEntry".
    loaded = load_object(str(entry_path))
    assert loaded.payload_type == "FileCacheEntry"
    assert loaded.payload == {"value": "computed-value"}


def test_get_hit_skips_compute_fn(tmp_path):
    cache = FileCache(tmp_path / "cache")
    key_components = ("graph", "hash123")
    _, first = counting_fn("first")
    assert cache.get(key_components, compute_fn=first) == "first"

    counter, second = counting_fn("second")
    assert cache.get(key_components, compute_fn=second) == "first"
    assert counter["calls"] == 0  # hit: compute_fn never called


def test_get_without_compute_fn_put_and_alias(tmp_path):
    cache = FileCache(tmp_path / "cache")

    # Unknown key without compute_fn -> None.
    assert cache.get(("missing",)) is None

    # put then get round-trips.
    cache.put(("k",), "stored")
    assert cache.get(("k",)) == "stored"

    # get_or_compute is an alias for get(compute_fn=...) — computes on miss.
    counter, fn = counting_fn("aliased")
    assert cache.get_or_compute(("alias",), fn) == "aliased"
    assert counter["calls"] == 1
    assert cache.get(("alias",)) == "aliased"  # and the result was stored


def test_distinct_keys_distinct_entries(tmp_path):
    cache = FileCache(tmp_path / "cache")
    variants = [("a", 1), ("a", 2), ("a",), "a", {"name": "a", 1: 2}]

    # Every variation derives a distinct key...
    keys = [compute_key(c) for c in variants]
    # BUG(etl): compute_key iterates key_components with a list
    # comprehension, so a bare string component ("a") is split into its
    # characters and derives the SAME key as the tuple ("a",) — the two
    # variations collide into one entry instead of two.
    assert len(set(keys)) == len(variants)

    # ...and each key stores/returns its own value.
    for i, components in enumerate(variants):
        counter, fn = counting_fn(i)
        assert cache.get(components, compute_fn=fn) == i
        assert counter["calls"] == 1
    for i, components in enumerate(variants):
        counter, fn = counting_fn(-1)
        assert cache.get(components, compute_fn=fn) == i
        assert counter["calls"] == 0
    for components in variants:
        key = compute_key(components)
        assert (tmp_path / "cache" / key[:2] / (key + ".etlcache")).is_file()


def test_dict_component_insertion_order_same_key(tmp_path):
    # Canonical JSON keying means dict insertion order must not matter.
    d1 = {"name": "a", 1: 2}
    d2 = {1: 2, "name": "a"}
    # BUG(etl): the "dict" codec keeps items in insertion order and
    # compute_key's json.dumps(sort_keys=True) cannot sort list elements,
    # so {name-first} and {1-first} derive DIFFERENT keys — violating the
    # documented canonical-key contract (cache.py compute_key docstring /
    # etl/persist/CONTEXT.md).
    assert compute_key((d1,)) == compute_key((d2,))

    cache = FileCache(tmp_path / "cache")
    cache.put((d1,), "ordered")
    assert cache.get((d2,)) == "ordered"


# ---------------------------------------------------------------------------
# compute_key: determinism, format, sensitivity
# ---------------------------------------------------------------------------


def test_compute_key_determinism_and_format():
    components = ("graph", "hash123", {"opts": [1, 2]})
    key = compute_key(components)
    assert key == compute_key(components)  # deterministic across calls
    assert len(key) == 64  # sha256 hex
    assert key == key.lower()
    assert all(c in "0123456789abcdef" for c in key)


def test_compute_key_sensitive_to_component_changes():
    base = ("a", 1)
    variations = [
        ("a", 2),  # value change
        (1, "a"),  # order change
        ("a", 1.0),  # int -> float
        ("b", 1),  # str change
        (("a", 1),),  # extra nesting
        ("a", 1, None),  # extra component
    ]
    keys = [compute_key(c) for c in [base, *variations]]
    assert len(set(keys)) == len(keys)


def test_compute_key_array_sensitivity():
    key = compute_key((np.array([1.0, 2.0, 3.0]),))
    assert compute_key((np.array([1.0, 2.0, 3.0]),)) == key  # deterministic
    assert compute_key((np.array([1.0, 2.0, 4.0]),)) != key  # element change


# ---------------------------------------------------------------------------
# Value round-trip fidelity
# ---------------------------------------------------------------------------


def test_complex_value_roundtrip_nan(tmp_path):
    cache = FileCache(tmp_path / "cache")
    value = {
        "weights": np.array([[1.0, np.nan], [np.nan, 3.0]]),
        "bias": np.array([0.5, -0.25]),
        "scalar": np.float32(2.5),
        "none_field": None,
        "meta": {"layers": [("dense", 128), ("relu", None)], "names": ["w", "b"]},
    }

    cache.put(("model", "v1"), value)
    got = cache.get(("model", "v1"))

    assert isinstance(got, dict)
    assert np.array_equal(got["weights"], value["weights"], equal_nan=True)
    assert np.array_equal(got["bias"], value["bias"])
    assert got["scalar"] == value["scalar"]
    assert got["none_field"] is None
    assert got["meta"] == value["meta"]


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------


def test_compute_fn_error_propagates_and_not_cached(tmp_path):
    cache = FileCache(tmp_path / "cache")

    def boom():
        raise ValueError("compute failed")

    with pytest.raises(ValueError, match="compute failed"):
        cache.get(("k",), compute_fn=boom)
    # The failed computation must not leave an entry behind.
    assert cache.contains(("k",)) is False
    assert cache.get(("k",)) is None


def test_corrupt_entry_treated_as_miss(tmp_path):
    directory = tmp_path / "cache"
    cache = FileCache(directory)
    cache.put(("k",), "stored")

    entries = glob.glob(str(directory / "**" / "*.etlcache"), recursive=True)
    assert len(entries) == 1
    entry_path = entries[0]
    with open(entry_path, "wb") as f:
        f.write(b"garbage bytes, not a container")

    # Corrupt entry without compute_fn -> None (errors never propagate).
    assert cache.get(("k",)) is None

    # Corrupt entry with compute_fn -> recomputed + atomically overwritten.
    counter, fn = counting_fn("recomputed")
    assert cache.get(("k",), compute_fn=fn) == "recomputed"
    assert counter["calls"] == 1
    assert cache.get(("k",)) == "recomputed"

    # The overwritten file is a loadable container again.
    loaded = load_object(entry_path)
    assert loaded.payload_type == "FileCacheEntry"
    assert loaded.payload["value"] == "recomputed"


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------


def test_persistence_across_instances(tmp_path):
    directory = tmp_path / "cache"
    cache = FileCache(directory)
    counter, fn = counting_fn("persisted")
    assert cache.get(("k",), compute_fn=fn) == "persisted"
    assert counter["calls"] == 1

    reopened = FileCache(directory)  # fresh instance, same directory
    counter2, fn2 = counting_fn("fresh")
    assert reopened.get(("k",), compute_fn=fn2) == "persisted"
    assert counter2["calls"] == 0  # hit, no recomputation


def test_contains(tmp_path):
    cache = FileCache(tmp_path / "cache")
    assert cache.contains(("k",)) is False
    cache.put(("k",), "v")
    assert cache.contains(("k",)) is True
    assert cache.contains(("other",)) is False


def test_clear(tmp_path):
    directory = tmp_path / "cache"
    cache = FileCache(directory)
    cache.put(("k",), "v")
    assert cache.contains(("k",)) is True

    cache.clear()

    assert cache.contains(("k",)) is False
    assert cache.get(("k",)) is None
    assert directory.is_dir()  # directory itself survives
    # Cache remains usable after clear.
    counter, fn = counting_fn("after-clear")
    assert cache.get(("k",), compute_fn=fn) == "after-clear"
    assert counter["calls"] == 1
    assert cache.get(("k",)) == "after-clear"


def test_directory_autocreated_nested(tmp_path):
    directory = tmp_path / "x" / "y" / "z"
    assert not directory.exists()

    cache = FileCache(directory)  # constructor creates nested paths
    assert directory.is_dir()
    counter, fn = counting_fn("nested")
    assert cache.get(("k",), compute_fn=fn) == "nested"
    assert cache.contains(("k",))
    assert cache.get(("k",)) == "nested"
