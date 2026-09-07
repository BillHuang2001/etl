"""tests/test_build_cache.py — unit coverage for the ``build_cached`` /
``CompileCache`` memoized-build sugar (``etl.build_cache``, Workstream C).

The module under test is the memoization sugar over ``etl.build``: the cache
key is computed BEFORE any tracing from (fn identity with keepalive, the full
``*specs`` tree incl. static content, resolved backend name, resolved device,
canonicalized options); env defaults ``ETL_BACKEND``/``ETL_DEVICE``/
``ETL_TARGET_BACKENDS`` resolve per call and participate. Pinned here:

  * key sensitivity (static content, TensorSpec, options, backend, fn id);
  * per-call env resolution, incl. malformed-value errors that count NO miss;
  * LRU eviction (maxsize), counters, ``clear()`` reset, direct ``Cache`` API;
  * error propagation semantics: trace-time failures re-raise and never
    cache (misses increment per attempt), unencodable static leaves raise a
    pre-cache ``TypeError`` naming the component path (no miss counted),
    ``cache=`` type checks, non-callable / Graph ``fn`` rejection;
  * zero pipeline work on hits (monkeypatched ``etl.pipeline`` /
    ``etl.trace`` spies);
  * thread-safety: concurrent distinct-key compiles and the same-key race
    (first-insert-wins, single entry);
  * static args stay in the run signature (validated at run time).

Pure numpy except the guarded iree-llvm-cpu leg in ``TestKeySensitivity``.
Every test passes its own private ``CompileCache`` — the process-global
module default is only exercised by the single opt-in test that swaps it via
monkeypatch.
"""

import sys
import threading

import numpy as np
import pytest

import etl
from etl.build_cache import CompileCache, build_cached

SPEC = etl.TensorSpec((4,), etl.float32)


@etl.defn
def _add_k(x, k):
    """x + static k — the workhorse defn for key-sensitivity tests."""
    return etl.add(x, k)


@etl.defn
def _boom(x, k):
    """Raises ZeroDivisionError at TRACE time for the static value k == 3."""
    return etl.add(x, 1 // (k - 3))


class _Weird:
    """A plain Python object the persist codec cannot encode."""


def _x():
    return np.array([0.5, 1.0, -2.0, 4.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate per-call env resolution: no ETL_* defaults leak between tests."""
    for var in ("ETL_BACKEND", "ETL_DEVICE", "ETL_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# 1. Key sensitivity
# --------------------------------------------------------------------------


class TestKeySensitivity:
    def test_identical_args_hit_identity_and_counters(self):
        cache = CompileCache()
        exe1 = build_cached(_add_k, SPEC, 3, cache=cache)
        assert cache.misses == 1 and cache.hits == 0 and len(cache) == 1
        exe2 = build_cached(_add_k, SPEC, 3, cache=cache)
        # A hit returns the SAME fully loaded Executable object ...
        assert exe2 is exe1
        # ... with zero extra cache entries and one counted hit.
        assert cache.hits == 1 and cache.misses == 1 and len(cache) == 1
        # And the cached executable is a NORMAL executable.
        got = etl.run(exe2, _x(), 3).numpy()
        np.testing.assert_allclose(got, _x() + 3, rtol=1e-6, atol=1e-6)

    def test_static_content_change_misses(self):
        cache = CompileCache()
        exe3 = build_cached(_add_k, SPEC, 3, cache=cache)
        exe4 = build_cached(_add_k, SPEC, 4, cache=cache)
        assert exe4 is not exe3
        assert cache.misses == 2 and cache.hits == 0 and len(cache) == 2
        # Static content participates: 3 and 4 specialize different graphs.
        np.testing.assert_allclose(
            etl.run(exe3, _x(), 3).numpy(), _x() + 3, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            etl.run(exe4, _x(), 4).numpy(), _x() + 4, rtol=1e-6, atol=1e-6
        )

    def test_tensor_spec_shape_change_misses(self):
        cache = CompileCache()
        spec4 = etl.TensorSpec((4,), etl.float32)
        spec8 = etl.TensorSpec((8,), etl.float32)
        exe1 = build_cached(_add_k, spec4, 3, cache=cache)
        exe2 = build_cached(_add_k, spec8, 3, cache=cache)
        assert exe2 is not exe1
        assert cache.misses == 2 and len(cache) == 2

    def test_tensor_spec_dtype_change_misses(self):
        cache = CompileCache()
        spec_f32 = etl.TensorSpec((4,), etl.float32)
        spec_f64 = etl.TensorSpec((4,), etl.float64)
        exe1 = build_cached(_add_k, spec_f32, 3, cache=cache)
        exe2 = build_cached(_add_k, spec_f64, 3, cache=cache)
        assert exe2 is not exe1
        assert cache.misses == 2 and len(cache) == 2
        got = etl.run(exe2, _x().astype(np.float64), 3).numpy()
        np.testing.assert_allclose(got, _x().astype(np.float64) + 3)

    def test_option_change_misses(self):
        # opt_level is documented-ignored by the numpy backend, but it is a
        # resolved option and MUST participate in the key.
        cache = CompileCache()
        exe1 = build_cached(_add_k, SPEC, 3, cache=cache)
        exe2 = build_cached(_add_k, SPEC, 3, cache=cache, opt_level="O0")
        exe3 = build_cached(_add_k, SPEC, 3, cache=cache, opt_level="O1")
        assert exe2 is not exe1 and exe3 is not exe2
        assert cache.misses == 3 and len(cache) == 3
        # And the same option set again is a hit.
        exe3b = build_cached(_add_k, SPEC, 3, cache=cache, opt_level="O1")
        assert exe3b is exe3
        assert cache.hits == 1 and cache.misses == 3

    def test_canonical_semantic_equality_hits(self):
        # Semantically-equal option sets (different construction order,
        # permuted flag lists) key identically.
        cache = CompileCache()
        exe1 = build_cached(
            _add_k, SPEC, 3, cache=cache, rank_context="abc", opt_level="O1"
        )
        exe2 = build_cached(
            _add_k, SPEC, 3, cache=cache, opt_level="O1", rank_context="abc"
        )
        assert exe2 is exe1
        assert cache.hits == 1 and cache.misses == 1 and len(cache) == 1
        # Permuted list-valued options canonicalize by sorting.
        exe3 = build_cached(
            _add_k, SPEC, 3, cache=cache, iree_compile_args=["--x", "--y"]
        )
        exe4 = build_cached(
            _add_k, SPEC, 3, cache=cache, iree_compile_args=["--y", "--x"]
        )
        assert exe4 is exe3
        assert cache.hits == 2 and cache.misses == 2

    def test_backend_change_misses_iree_leg(self):
        # The iree leg only runs when the iree compiler+runtime are present;
        # skips cleanly otherwise (function-local importorskip).
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")
        cache = CompileCache()
        device = etl.Device("cpu", 0)
        exe_np = build_cached(_add_k, SPEC, 3, cache=cache)
        assert exe_np.backend == "numpy"
        exe_iree = build_cached(
            _add_k, SPEC, 3, cache=cache, backend="iree", device=device
        )
        assert exe_iree.backend == "iree" and exe_iree.device == device
        # The backend name (and its inferred options) participate in the key.
        assert exe_iree is not exe_np
        assert cache.misses == 2 and len(cache) == 2
        # Identical iree call -> hit with identity.
        exe_iree2 = build_cached(
            _add_k, SPEC, 3, cache=cache, backend="iree", device=device
        )
        assert exe_iree2 is exe_iree
        assert cache.hits == 1 and cache.misses == 2
        got = etl.run(exe_iree, _x(), 3).numpy()
        np.testing.assert_allclose(got, _x() + 3, rtol=1e-5, atol=1e-5)

    def test_unknown_backend_resolution_error_counts_no_miss(self):
        # A resolution failure fires BEFORE any cache access: no miss is
        # counted and nothing is inserted.
        cache = CompileCache()
        with pytest.raises(etl.BackendError, match="unknown backend 'nope'"):
            build_cached(_add_k, SPEC, 3, cache=cache, backend="nope")
        assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0
        # A subsequent healthy call works (first real miss).
        exe = build_cached(_add_k, SPEC, 3, cache=cache)
        assert exe is not None
        assert cache.misses == 1 and len(cache) == 1


# --------------------------------------------------------------------------
# 2. Per-call env defaults (ETL_BACKEND / ETL_DEVICE)
# --------------------------------------------------------------------------


class TestEnvDefaults:
    def test_malformed_etl_backend_counts_no_miss(self, monkeypatch):
        monkeypatch.setenv("ETL_BACKEND", "nonsense")
        cache = CompileCache()
        with pytest.raises(etl.BackendError, match="unknown backend 'nonsense'"):
            build_cached(_add_k, SPEC, 3, cache=cache)
        # Resolution errors never touch the cache.
        assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0

    def test_valid_etl_backend_env_works(self, monkeypatch):
        cache = CompileCache()
        monkeypatch.setenv("ETL_BACKEND", "nonsense")
        with pytest.raises(etl.BackendError):
            build_cached(_add_k, SPEC, 3, cache=cache)
        assert cache.misses == 0
        # The env var is read lazily per call: a valid value now resolves.
        monkeypatch.setenv("ETL_BACKEND", "numpy")
        exe = build_cached(_add_k, SPEC, 3, cache=cache)
        assert exe.backend == "numpy" and exe.device == etl.Device("cpu", 0)
        assert cache.misses == 1 and len(cache) == 1

    def test_malformed_etl_device_raises_and_counts_no_miss(self, monkeypatch):
        monkeypatch.setenv("ETL_DEVICE", "cuda::9")
        cache = CompileCache()
        with pytest.raises(etl.DeviceError) as excinfo:
            build_cached(_add_k, SPEC, 3, cache=cache)
        message = str(excinfo.value)
        assert "ETL_DEVICE" in message and "cuda::9" in message
        assert "expected 'kind' or 'kind:index'" in message
        assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0

    def test_valid_etl_device_env_works(self, monkeypatch):
        monkeypatch.setenv("ETL_DEVICE", "cpu")
        cache = CompileCache()
        exe = build_cached(_add_k, SPEC, 3, cache=cache)
        assert exe.device == etl.Device("cpu", 0)
        assert cache.misses == 1 and len(cache) == 1


# --------------------------------------------------------------------------
# 3. Counters / LRU / clear / direct Cache API
# --------------------------------------------------------------------------


class TestCacheCountersAndLRU:
    def test_lru_eviction_oldest_first(self):
        cache = CompileCache(maxsize=2)
        exe1 = build_cached(_add_k, SPEC, 1, cache=cache)
        exe2 = build_cached(_add_k, SPEC, 2, cache=cache)
        exe3 = build_cached(_add_k, SPEC, 3, cache=cache)
        assert cache.misses == 3 and len(cache) == 2  # k=1 evicted
        # Rebuilding the evicted key is a miss again ...
        exe1b = build_cached(_add_k, SPEC, 1, cache=cache)
        assert exe1b is not exe1
        assert cache.misses == 4 and len(cache) == 2
        # ... and now evicts the next-oldest (k=2).
        exe2b = build_cached(_add_k, SPEC, 2, cache=cache)
        assert exe2b is not exe2
        assert cache.misses == 5 and len(cache) == 2

    def test_hit_moves_entry_to_mru_end(self):
        # A hit refreshes recency: the hit entry survives the next eviction.
        cache = CompileCache(maxsize=2)
        build_cached(_add_k, SPEC, 1, cache=cache)
        build_cached(_add_k, SPEC, 2, cache=cache)
        build_cached(_add_k, SPEC, 1, cache=cache)  # hit -> k=1 now MRU
        assert cache.hits == 1 and cache.misses == 2
        build_cached(_add_k, SPEC, 3, cache=cache)  # evicts LRU (k=2)
        assert cache.misses == 3 and len(cache) == 2
        # k=1 survived; k=2 was evicted (rebuild is a miss).
        exe1b = build_cached(_add_k, SPEC, 1, cache=cache)
        assert cache.hits == 2 and cache.misses == 3
        exe2b = build_cached(_add_k, SPEC, 2, cache=cache)
        assert exe2b is not None
        assert cache.hits == 2 and cache.misses == 4

    def test_clear_empties_and_resets_counters(self):
        cache = CompileCache()
        build_cached(_add_k, SPEC, 1, cache=cache)
        build_cached(_add_k, SPEC, 1, cache=cache)  # hit
        build_cached(_add_k, SPEC, 2, cache=cache)  # miss
        assert cache.hits == 1 and cache.misses == 2 and len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0 and cache.misses == 0
        # The cache is usable after clear().
        exe = build_cached(_add_k, SPEC, 1, cache=cache)
        assert exe is not None
        assert cache.misses == 1

    def test_direct_cache_api(self):
        cache = CompileCache()
        assert isinstance(cache, etl.Cache)
        assert cache.maxsize == 128
        # get with compute_fn on a miss: computes, stores, counts a miss.
        value = cache.get(("key", 1), compute_fn=lambda: "v1")
        assert value == "v1"
        assert cache.misses == 1 and len(cache) == 1
        # get on a hit returns the stored value and counts a hit.
        assert cache.get(("key", 1)) == "v1"
        assert cache.hits == 1 and cache.misses == 1
        # get on a miss without compute_fn returns None and counts a miss.
        assert cache.get(("key", 2)) is None
        assert cache.misses == 2 and len(cache) == 1
        # put / contains (put does not touch the counters).
        cache.put(("key", 2), "v2")
        assert cache.contains(("key", 2))
        assert not cache.contains(("key", 99))
        assert cache.hits == 1 and cache.misses == 2 and len(cache) == 2
        # get_or_compute is the Cache-ABC alias of get(..., compute_fn=...).
        assert cache.get_or_compute(("key", 3), lambda: "v3") == "v3"
        assert cache.misses == 3
        # clear resets everything.
        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0 and cache.misses == 0

    def test_maxsize_validation(self):
        assert CompileCache(2).maxsize == 2
        for bad in (0, -1, 2.5, True):
            with pytest.raises(ValueError, match="maxsize must be a positive int"):
                CompileCache(bad)

    def test_cache_none_uses_module_default(self, monkeypatch):
        # cache=None -> the process-global module default. Swap the module
        # global so the test stays isolated from the rest of the session.
        import etl.build_cache as build_cache_module

        private = CompileCache()
        monkeypatch.setattr(build_cache_module, "_DEFAULT_CACHE", private)
        exe1 = build_cached(_add_k, SPEC, 11)
        exe2 = build_cached(_add_k, SPEC, 11)
        assert exe2 is exe1
        assert private.misses == 1 and private.hits == 1 and len(private) == 1


# --------------------------------------------------------------------------
# 4. fn-identity separation
# --------------------------------------------------------------------------


class TestFnIdentity:
    def test_two_defns_with_identical_specs_are_distinct_entries(self):
        @etl.defn
        def _add_k_sibling(x, k):
            return etl.add(x, k)

        cache = CompileCache()
        exe1 = build_cached(_add_k, SPEC, 7, cache=cache)
        exe2 = build_cached(_add_k_sibling, SPEC, 7, cache=cache)
        assert exe2 is not exe1
        assert cache.misses == 2 and len(cache) == 2
        # Both are functional.
        np.testing.assert_allclose(
            etl.run(exe1, _x(), 7).numpy(), _x() + 7, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            etl.run(exe2, _x(), 7).numpy(), _x() + 7, rtol=1e-6, atol=1e-6
        )


# --------------------------------------------------------------------------
# 5. Error propagation
# --------------------------------------------------------------------------


class TestErrorPropagation:
    def test_trace_time_error_reraises_every_call_and_never_caches(self):
        cache = CompileCache()
        for _ in range(3):
            with pytest.raises(ZeroDivisionError):
                build_cached(_boom, SPEC, 3, cache=cache)
            # Each attempt is a fresh miss; nothing is ever inserted.
            assert len(cache) == 0
        assert cache.misses == 3 and cache.hits == 0
        # A healthy call afterwards works normally.
        exe = build_cached(_boom, SPEC, 5, cache=cache)
        np.testing.assert_allclose(
            etl.run(exe, _x(), 5).numpy(), _x(), rtol=1e-6, atol=1e-6
        )
        assert cache.misses == 4 and len(cache) == 1
        # And the healthy entry is cached like any other.
        exe2 = build_cached(_boom, SPEC, 5, cache=cache)
        assert exe2 is exe
        assert cache.hits == 1 and cache.misses == 4

    @pytest.mark.parametrize(
        ("leaf", "typename"),
        [
            ({1, 2}, "set"),
            (_Weird(), "_Weird"),
        ],
        ids=["set", "custom-object"],
    )
    def test_unencodable_static_spec_leaf_typeerror(self, leaf, typename):
        cache = CompileCache()
        with pytest.raises(TypeError) as excinfo:
            build_cached(_add_k, SPEC, leaf, cache=cache)
        message = str(excinfo.value)
        assert f"at specs[1] of type '{typename}'" in message
        assert "etl.persist covers" in message  # guidance suffix
        # Pre-cache TypeError: the cache is untouched, no miss counted.
        assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0

    def test_unencodable_options_leaf_typeerror(self):
        cache = CompileCache()
        with pytest.raises(TypeError) as excinfo:
            build_cached(_add_k, SPEC, 3, cache=cache, foo={_Weird(): 1})
        message = str(excinfo.value)
        assert "options['foo']" in message
        assert "of type '_Weird'" in message
        assert "etl.persist covers" in message
        assert cache.hits == 0 and cache.misses == 0 and len(cache) == 0

    def test_cache_type_checks(self):
        # cache=object() and cache=False: no cache=False escape exists —
        # plain etl.build is the documented no-cache path.
        for bad_cache in (object(), False):
            with pytest.raises(
                TypeError, match="must be a CompileCache instance"
            ) as excinfo:
                build_cached(_add_k, SPEC, 3, cache=bad_cache)
            assert "got bool" in str(excinfo.value) or "got object" in str(
                excinfo.value
            )

    def test_non_callable_fn_typeerror(self):
        with pytest.raises(TypeError, match="build_cached expects a callable"):
            build_cached(42, SPEC, 3, cache=CompileCache())

    def test_graph_fn_typeerror(self):
        # An already-traced Graph is a different pipeline stage: reject with
        # guidance toward the explicit lower/compile/load pipeline.
        graph = etl.trace(_add_k, SPEC, 3)
        with pytest.raises(TypeError) as excinfo:
            build_cached(graph, SPEC, 3, cache=CompileCache())
        message = str(excinfo.value)
        assert "got Graph" in message
        assert "explicit lower/compile/load pipeline" in message


# --------------------------------------------------------------------------
# 6. Zero pipeline work on hits (monkeypatched stage spies)
# --------------------------------------------------------------------------


class TestNoPipelineWorkOnHit:
    def test_hit_does_zero_trace_lower_compile_load(self, monkeypatch):
        cache = CompileCache()
        exe1 = build_cached(_add_k, SPEC, 3, cache=cache)
        assert cache.misses == 1

        # Spy on every pipeline stage the miss path would touch. The lazy
        # function-local imports inside build_cached resolve these module
        # attributes per call, so the spies see any re-run.
        calls = []

        def _spy(stage):
            def wrapper(*args, **kwargs):
                calls.append(stage)
                raise AssertionError(f"pipeline stage {stage!r} ran on a cache hit")

            return wrapper

        monkeypatch.setattr(sys.modules["etl.trace"], "trace", _spy("trace"))
        monkeypatch.setattr(sys.modules["etl.pipeline"], "lower", _spy("lower"))
        monkeypatch.setattr(sys.modules["etl.pipeline"], "compile", _spy("compile"))
        monkeypatch.setattr(sys.modules["etl.pipeline"], "load", _spy("load"))

        exe2 = build_cached(_add_k, SPEC, 3, cache=cache)
        assert exe2 is exe1
        assert cache.hits == 1 and cache.misses == 1 and len(cache) == 1
        assert calls == []


# --------------------------------------------------------------------------
# 7. Thread-safety smoke
# --------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_distinct_keys(self):
        cache = CompileCache()
        results = [None] * 8
        errors = []

        def work(i):
            try:
                results[i] = build_cached(_add_k, SPEC, 100 + i, cache=cache)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append((i, exc))

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert len(cache) == 8 and cache.misses == 8 and cache.hits == 0
        for i in range(8):
            got = etl.run(results[i], _x(), 100 + i).numpy()
            np.testing.assert_allclose(
                got, _x() + 100 + i, rtol=1e-6, atol=1e-6
            )

    def test_same_key_race_first_insert_wins(self):
        cache = CompileCache()
        results = [None] * 8
        errors = []

        def race(_):
            try:
                results[_] = build_cached(_add_k, SPEC, 999, cache=cache)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=race, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        # Every thread returns the SAME first-inserted executable ...
        assert len({id(r) for r in results}) == 1
        assert len(cache) == 1
        assert cache.misses >= 1
        # ... and it is fully functional.
        got = etl.run(results[0], _x(), 999).numpy()
        np.testing.assert_allclose(got, _x() + 999, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# 8. Static args stay in the run signature
# --------------------------------------------------------------------------


class TestStaticArgsAtRun:
    def test_run_accepts_and_validates_static_value(self):
        cache = CompileCache()
        exe = build_cached(_add_k, SPEC, 3, cache=cache)
        got = etl.run(exe, _x(), 3).numpy()
        np.testing.assert_allclose(got, _x() + 3, rtol=1e-6, atol=1e-6)

    def test_run_rejects_wrong_static_value(self):
        cache = CompileCache()
        exe = build_cached(_add_k, SPEC, 3, cache=cache)
        with pytest.raises(etl.TraceError) as excinfo:
            etl.run(exe, _x(), 4)
        message = str(excinfo.value)
        assert "specialized on 3" in message
        assert "path [1]" in message
