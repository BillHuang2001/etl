"""Memoized-build sugar for meta-level loops: ``build_cached`` + ``CompileCache``.

Workstream C ("True JIT"): NAS / genetic-programming-style outer loops decode
a "program" as a static Python structure and re-trace a step ``@etl.defn``
per program. ``build_cached`` memoizes that re-tracing: documented shorthand
for a cache-guarded ``etl.build`` whose exact ``trace -> lower -> compile ->
load`` composition runs AT MOST ONCE per distinct cache key — later calls
return the identical fully loaded ``Executable`` object with zero pipeline
work::

    build_cached(fn, *specs, cache=None, **opts) ~=
        cache.get(key(id(fn), specs-tree-with-static-content,
                      resolved-backend, resolved-device, canonical-options),
                  compute_fn=lambda: build(fn, *specs, **opts))

Two refinements over that sketch are binding: the key's backend/device are
the per-call ``etl.pipeline._resolve_backend_device`` snapshots (env defaults
``ETL_BACKEND``/``ETL_DEVICE``/``ETL_TARGET_BACKENDS`` are read at call time
and participate in the key), and options participate CANONICALIZED (see
``_canonicalize_options``) so semantically-equal option sets key identically.

Meta-loop contract (binding): the per-iteration "program" argument MUST be a
static Python value — its static content is exactly what the memoized key
specializes on (each distinct program = distinct graph = distinct cache
entry). A runtime tensor in the program slot is an ordinary graph input
(in-graph interpretation is a different feature): ``build_cached`` keys it
ONCE on its ``TensorSpec`` and does NOT specialize per value — documented,
no runtime warning.

Cache-key contract (binding):

  * The key is computed BEFORE any tracing, from (fn identity, the full
    ``*specs`` tree — structure AND static-leaf content —, resolved backend
    name, resolved device, canonicalized resolved options).
  * Per-call env defaults (``ETL_BACKEND``/``ETL_DEVICE``/``ETL_TARGET_BACKENDS``)
    are resolved at call time and participate in the key. Process-env compile
    knobs read by the compiler stages themselves (``ETL_OPT_LEVEL``,
    ``ETL_IREE_*``, ``ETL_XLA_*``, ``ETL_TVM_*``) are NOT part of the key —
    when changing them, clear the cache or pass a fresh ``CompileCache``
    (documented limitation).
  * In-memory only: cached values are fully loaded Executables holding device
    handles/vmfb/compiled code — never persisted; cross-process compile
    caching is out of scope.
  * There is deliberately NO ``cache=False`` escape — plain ``etl.build`` is
    the always-available no-cache path.

Errors: a program that fails to trace/lower/compile/load raises per stage
(never cached, never an eager fallback). The returned Executable is a NORMAL
executable: same-device run loops (device-resident inner loops feeding
outputs back as inputs) work unchanged.

The module-level ``_DEFAULT_CACHE`` is a process-wide memoization table shared
by every ``build_cached`` call that does not pass its own ``cache`` — pass a
private ``CompileCache`` for isolation. Importing this module installs no
global behavior (the default cache is only touched when a call opts in by
omitting ``cache``).

Import rule: module-top imports are stdlib + ``etl.persist`` (persist imports
``etl.core`` only — no cycle) + ``etl.core.PersistenceError`` (the codec's
error type, needed by the encodability walker; the same import persist.cache
uses). ``pipeline``/``trace`` are imported lazily inside ``build_cached`` —
mirrors ``etl.build``'s lazy trace import, avoids import cycles, and lets
tests monkeypatch ``etl.pipeline.lower``.
"""

import threading
from collections import OrderedDict

from etl.core import PersistenceError
from etl.persist.cache import Cache, compute_key
from etl.persist.codec import encode_value

__all__ = ["build_cached", "CompileCache"]

_GUIDANCE = (
    "etl.persist covers None/bool/int/float/str/complex/list/tuple/dict/numpy "
    "arrays/dtype/slice/Dim/DimExpr/Device/TensorSpec/TreeSpec — an Enum "
    "member or other Python object must be converted, e.g. its .value). Use "
    "plain etl.build if the call cannot be keyed."
)
"""Shared guidance suffix of every cache-key encodability TypeError (the
walker message and the defense-in-depth re-raise keep identical wording)."""


def _canonicalize_options(options):
    """Return a canonical, order-insensitive tuple form of an options dict.

    Recursively maps dicts to ``tuple(sorted((k, _canonicalize(v)) for k, v
    in d.items()))`` and lists/tuples to sorted tuples of their
    canonicalized items, so semantically-equal option sets (different
    construction orders, permuted flag lists) key identically. Scalars pass
    through. If sorting raises ``TypeError`` (incomparable mixed items), the
    original order is kept — the fallback tuple still encodes determinis-
    tically (documented; order then participates as given).
    """
    def canon(value):
        if isinstance(value, dict):
            try:
                return tuple(sorted((k, canon(v)) for k, v in value.items()))
            except TypeError:
                # Incomparable mixed keys/values: fall back to the original
                # insertion order (items stay individually canonicalized).
                return tuple((k, canon(v)) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            try:
                return tuple(sorted(canon(v) for v in value))
            except TypeError:
                # Incomparable mixed items: keep the original order.
                return tuple(canon(v) for v in value)
        return value

    return canon(options)


def _first_bad_leaf(value, path):
    """Return ``(path, leaf)`` of the first codec-unencodable leaf in
    ``value``, or ``None`` when every leaf encodes.

    Descends ONLY into plain ``list``/``tuple``/``dict`` containers — every
    other type (TensorSpec/TreeSpec/Device/Dim/np.ndarray/np.dtype, …) is a
    leaf and all are codec-covered — attempting ``codec.encode_value`` on
    each leaf (dict keys included: the codec encodes keys too). Renders
    paths like ``specs[2]``, ``specs[3]['w']``, ``options['target_backends'][1]``.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}[{key!r}]"
            try:
                encode_value(key)
            except PersistenceError:
                return item_path, key
            bad = _first_bad_leaf(item, item_path)
            if bad is not None:
                return bad
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            bad = _first_bad_leaf(item, item_path)
            if bad is not None:
                return bad
        return None
    try:
        encode_value(value)
    except PersistenceError:
        return path, value
    return None


class CompileCache(Cache):
    """In-memory LRU compile cache (a ``persist.cache.Cache`` subclass).

    Maps ``compute_key`` hex keys to ``[value, keepalive]`` entries with LRU
    eviction at ``maxsize`` entries. ``build_cached`` uses it to memoize
    fully loaded Executables; it also works directly through the standard
    ``Cache`` interface (``get``/``put``/``contains``/``clear``; the
    ``get_or_compute`` alias comes from the ABC base).

    Thread-safety: a lock guards the dict and the counters. On a miss the
    ``compute_fn()`` compile runs OUTSIDE the lock, so concurrent
    distinct-key compiles never serialize; a same-key race may duplicate a
    compile (the results are identical by construction — the key determines
    the compile) and first-insert-wins keeps the first entry. Eviction pops
    the oldest entry — dropping value+keepalive lets GC reclaim the
    Executable (``clear()`` for eager release).

    Every entry pins BOTH the value (the Executable) and an optional
    keepalive object strongly: ``build_cached`` passes the traced fn as the
    keepalive, so an ``id(fn)``-keyed entry can never alias a later object
    that reuses the fn's id after GC.
    """

    def __init__(self, maxsize: int = 128):
        """Create an LRU cache holding at most ``maxsize`` entries.

        A non-positive or non-int ``maxsize`` raises ``ValueError``.
        """
        if not isinstance(maxsize, int) or isinstance(maxsize, bool) or maxsize <= 0:
            raise ValueError(
                f"CompileCache maxsize must be a positive int, got {maxsize!r}"
            )
        self.hits = 0
        self.misses = 0
        self.maxsize = maxsize
        self._entries = OrderedDict()  # hex-key -> [value, keepalive]
        self._lock = threading.RLock()

    # -- internal lock-protected paths (shared by get/_compute so counters
    #    and LRU order stay consistent) -----------------------------------

    def _lookup(self, key):
        """Lock-protected lookup: ``(True, value)`` on a hit (counts a hit
        and moves the entry to the most-recent end) or ``(False, None)`` on
        a miss (counts a miss)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self.hits += 1
                self._entries.move_to_end(key)
                return True, entry[0]
            self.misses += 1
            return False, None

    def _store(self, key, value, keepalive):
        """Lock-protected first-insert-wins store.

        Inserts ``[value, keepalive]`` at the most-recent end and evicts LRU
        entries while over ``maxsize`` (pop oldest — dropping value+
        keepalive lets GC reclaim the Executable). If another thread
        inserted the key meanwhile, returns the EXISTING value (the
        duplicate result is discarded — documented: same-key races may
        duplicate a compile; results are identical).
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry[0]
            self._entries[key] = [value, keepalive]
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
            return value

    def _compute(self, key_components, compute_fn, keepalive):
        """Compute-or-fetch with keepalive pinning — build_cached's path.

        Hex key via ``compute_key``; on a hit returns the stored value. On
        a miss runs ``compute_fn()`` OUTSIDE the lock (exceptions propagate;
        nothing is inserted), then stores first-insert-wins and returns the
        value (the existing value if a concurrent same-key insert won).
        """
        try:
            key = compute_key(key_components)
        except Exception as exc:
            # Defense in depth: build_cached's pre-key walk should have
            # caught every unencodable leaf; anything that still slips
            # through (e.g. a cyclic container) surfaces as a TypeError
            # with the same guidance instead of a raw PersistenceError.
            if not isinstance(exc, (PersistenceError, TypeError)):
                raise
            raise TypeError(
                f"build_cached: cannot key this call ({exc}) — {_GUIDANCE}"
            ) from exc
        found, value = self._lookup(key)
        if found:
            return value
        value = compute_fn()
        return self._store(key, value, keepalive)

    # -- public Cache interface -------------------------------------------

    def get(self, key_components, compute_fn=None):
        """See ``Cache.get`` — in-memory mirror.

        On a hit: ``hits += 1``, the entry moves to the most-recent end and
        its value is returned. On a miss: ``misses += 1``; returns ``None``
        when no ``compute_fn`` is given, otherwise delegates to the private
        ``_compute`` path with keepalive=None (compute runs outside the
        lock, first-insert-wins on races).
        """
        if compute_fn is None:
            key = compute_key(key_components)
            found, value = self._lookup(key)
            return value if found else None
        return self._compute(key_components, compute_fn, None)

    def put(self, key_components, value):
        """Store ``value`` under the key derived from ``key_components``.

        Inserts or overwrites at the most-recent end (an overwrite moves
        the entry to the end) and evicts LRU entries while over ``maxsize``.
        Counters are not touched.
        """
        key = compute_key(key_components)
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = [value, None]
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def contains(self, key_components):
        """Return True if an entry exists for ``key_components`` (O(1))."""
        key = compute_key(key_components)
        with self._lock:
            return key in self._entries

    def clear(self):
        """Remove ALL entries AND reset ``hits``/``misses`` to 0.

        Documented: ``clear`` resets the counters — use it for eager
        release of cached Executables.
        """
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self):
        """Number of entries currently cached."""
        with self._lock:
            return len(self._entries)


_DEFAULT_CACHE = CompileCache()
"""Module-level default compile cache (maxsize 128) used by ``build_cached``
when no ``cache`` is passed. A process-wide memoization table: pass your own
``CompileCache`` for isolation, and plain ``etl.build`` is the
always-available no-cache escape (there is deliberately NO ``cache=False``).
"""


def build_cached(fn, *specs, backend=None, device=None, cache=None, **options):
    """``build_cached(f, *specs, cache=None) -> Executable``.

    Documented shorthand for a MEMOIZED ``build`` — no other behavior.
    The exact ``trace -> lower -> compile -> load`` composition of
    ``etl.build`` runs at most once per distinct cache key; on a hit the
    SAME fully loaded ``Executable`` object is returned with zero pipeline
    work::

        build_cached(fn, *specs, cache=None, **opts) ~=
            cache.get(key(id(fn), specs-tree-with-static-content,
                          resolved-backend, resolved-device,
                          canonical-options),
                      compute_fn=lambda: build(fn, *specs, **opts))

    ...where the miss path expands exactly like ``etl.build``::

        graph = etl.trace(fn, *specs)
        lowered = lower(graph, backend=backend, **options)
        artifact = compile(lowered, **options)
        exe = load(artifact, backend=backend, device=device)   # no options

    Semantics (binding):

    * The cache key is computed BEFORE any tracing, from (fn identity —
      ``id(fn)``; the cache entry pins the fn as its keepalive so the id
      can never alias a later object —, the full ``*specs`` tree incl.
      static content, the resolved backend name, the resolved device, and
      the canonicalized resolved options). Static content participates:
      each distinct static "program" value in the meta loop = a distinct
      graph = a distinct cache entry. A runtime tensor in the program slot
      is an ordinary graph input and is keyed ONCE on its ``TensorSpec`` —
      never specialized per value (documented; no runtime warning).
    * Defaults: unset ``backend``/``device`` resolve exactly as in
      ``etl.build`` (``ETL_BACKEND``/``ETL_DEVICE``/``ETL_TARGET_BACKENDS``
      read lazily per call; the resolved values participate in the key).
      Process-env compile knobs read by the compiler stages themselves
      (``ETL_OPT_LEVEL``, ``ETL_IREE_*``, ``ETL_XLA_*``, ``ETL_TVM_*``) are
      NOT part of the key — clear the cache or pass a fresh ``CompileCache``
      when changing them (documented limitation).
    * Options are forwarded to ``lower`` and ``compile`` exactly as in
      ``etl.build`` (``load`` gets none); the key uses a canonicalized copy
      so semantically-equal option sets key identically.
    * Errors: fn validation, backend/device resolution, cache-type checks
      and cache-key encodability all fire BEFORE any cache access; a
      program that fails to trace/lower/compile/load raises per stage —
      never cached, never an eager fallback. Only a fully loaded Executable
      is ever inserted.
    * The returned Executable is a NORMAL executable: same-device run loops
      (device-resident inner loops feeding outputs back as inputs) work
      unchanged.
    * Cached values are in-memory only (Executables hold device handles /
      vmfb / compiled code — never persisted). There is deliberately NO
      ``cache=False`` escape: plain ``etl.build`` is the no-cache path.
    """
    if not callable(fn) and not getattr(fn, "__etl_defn__", False):
        raise TypeError(
            f"build_cached expects a callable or an @etl.defn function, got "
            f"{type(fn).__name__} — to stage an already-traced Graph use the "
            f"explicit lower/compile/load pipeline"
        )

    # Lazy function-local imports: mirrors etl.build's lazy trace import,
    # avoids import cycles, and lets tests monkeypatch etl.pipeline.lower.
    from etl.pipeline import _resolve_backend_device, compile, load, lower
    from etl.trace import trace as trace_fn

    # Env defaults snapshot at call time; resolution errors (e.g. an unknown
    # backend name) propagate untouched and nothing is cached.
    resolved_backend, resolved_device, resolved_options = _resolve_backend_device(
        backend, device, options
    )

    cache = _DEFAULT_CACHE if cache is None else cache
    if not isinstance(cache, CompileCache):
        raise TypeError(
            f"build_cached: cache must be a CompileCache instance (or None "
            f"for the module-level default), got {type(cache).__name__}"
        )

    # Validate encodability with path-naming BEFORE any key computation
    # (specs first, then options): every static leaf that participates in
    # the key must be codec-encodable.
    bad = _first_bad_leaf(specs, "specs")
    if bad is None:
        bad = _first_bad_leaf(resolved_options, "options")
    if bad is not None:
        path, leaf = bad
        raise TypeError(
            f"build_cached: cannot encode the cache-key component at {path} "
            f"of type {type(leaf).__name__!r} ({_GUIDANCE}"
        )

    # Tagged, self-describing components: the *specs tuple itself is the
    # specs component (its whole tree encodes — structure AND static-leaf
    # content participate; being a tuple of positional args, order
    # participates too).
    components = (
        ("fn", id(fn)),
        ("specs", specs),
        ("backend", resolved_backend.name),
        ("device", resolved_device),
        ("options", _canonicalize_options(resolved_options)),
    )

    def _compute():
        # The exact build composition (mirror etl.build's call shapes: load
        # gets NO options). Per-stage exceptions propagate untouched —
        # never cached, never an eager fallback (binding).
        graph = trace_fn(fn, *specs)
        lowered = lower(graph, backend=resolved_backend, **resolved_options)
        artifact = compile(lowered, **resolved_options)
        return load(artifact, backend=resolved_backend, device=resolved_device)

    # keepalive=fn pins the traced callable strongly inside the entry, so
    # an id(fn)-keyed entry can never alias a later object reusing the id
    # after GC. Returns the cache's object on first-insert-wins races.
    return cache._compute(components, _compute, keepalive=fn)
