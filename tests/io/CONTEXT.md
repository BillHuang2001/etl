# tests/io — etl.io async IO test suite

## Intent

Device-free unit suite pinning the committed public surface of `etl.io` (Workstream B,
`../etl/io.py` — module imports `core` only): `prefetch(host_tensor, device)` /
`sink(device_tensor)` return `AsyncCopy` handles (`.device`, `.direction`
`"host_to_device"`/`"device_to_host"`, `.done` non-blocking, `.wait(timeout=None)`
idempotent re-raising the stored copy exception VERBATIM — never swallowed,
`.result(timeout=None)` = wait + the resulting tensor: device tensor for prefetch,
host ndarray-backed cpu:0 tensor for sink), schedule-time validation is synchronous
on the CALLING thread, worker-time failures surface only at wait/result/wait_all,
builtin `TimeoutError` on timeout leaves the handle pending and re-waitable, GC NEVER
auto-waits (no `__del__`), `wait_all(timeout=None)` drains in issue order, ONE
process-global FIFO daemon worker executes the explicit `Tensor.to` copies in issue
order, and worker-thread re-entry raises `RuntimeError`.

## Structure

| File | Covers |
|---|---|
| `test_io_worker_bootstrap.py` | 37 tests: device-free plumbing pins for the worker's non-cpu context bootstrap (fake `iree` PACKAGE **and** `iree.runtime` submodule installed via `monkeypatch.setitem(sys.modules, ...)` — only `iree.runtime` is not enough), driven through the REAL process-global worker by `io.AsyncCopy(device, direction)` + `io._submit(handle, lambda: None)` (never `prefetch`/`sink`: kinds `cuda`/`rocm`/`hip`/`vulkan` own REAL lazy transfer providers) — the driver-name mapping (`"rocm"`/`"hip"` → `"hip"`, `"cuda"` → `"cuda"`) × the 1-based `device_id = index + 1` KEYWORD, the throwaway device kept alive in `io._bootstrap_devices`; laziness (a cpu-kind handle calls `get_driver` never and leaves `_bootstrapped` empty); the `(kind, index)` dedup (`_bootstrapped`) — once per key, again per new index/kind, and `"hip"`/`"rocm"` as DISTINCT keys despite one driver; best-effort failure (`get_driver` or `create_device` raising, and the not-retried-after-failure behavior since `_bootstrapped.add(key)` precedes the try) never altering the copy's outcome. **Vulkan half (22 of the tests — additive; the cuda/rocm assertions are unchanged):** `"vulkan"` maps to the `"vulkan"` driver itself (NOT `"hip"`/`"cuda"`), and `create_device` is pinned via `_FakeRuntime.create_device_signatures` (`(driver_name, args, kwargs)`) to receive the QUERIED `query_available_devices()[index]` info dict **positionally with NO kwargs at all** (a same-run cuda-vs-vulkan contrast asserts the two call SHAPES against each other: `("cuda", (), {"device_id": 1})` vs `("vulkan", (infos[1],), {})`) — the fake info dicts carry large UUID-hash-like `device_id` values (857002912 / 1854200391) plus `path`/`name`, and the pins state that the hash inside the info dict (`args[0]["device_id"]`) identifies the device while the passed id is never an ordinal int; per-index selection over two enumerated devices; the silent-no-op bounds guard (out-of-range `(1,5)`/`(2,2)`/`(2,9)` and an EMPTY enumeration `(0,0)`/`(0,3)` ⇒ `get_driver` + query happen, zero `create_device`, zero retained devices, the copy still succeeds, the key still recorded so it is never retried; a NEGATIVE index is driven through a bare `_DuckDevice("vulkan", -1)` stand-in because `core.Device` rejects negative indices at construction — pinned with `pytest.raises(ValueError, match="non-negative")`); vulkan `(kind, index)` dedup (same key once, another index again, and `"vulkan"` vs `"cuda"` as distinct keys); best-effort swallowing of all three failure modes (`get_driver`, the query, `create_device`) including that a failed bootstrap never MASKS the copy's own `DeviceError` (surfaced verbatim, same object, via a raising thunk), is not retried for the same key, and does not block another key. |
| `test_io.py` | 37 tests: module surface/exports (incl. no iree at import — pinned in a FRESH subprocess interpreter (`_run_python_snippet`, also reused by the exit pins) so it is independent of this process's import history, since an earlier suite may already have pulled `iree.runtime` in via `etl.core.devices()`; no handle `__del__`); schedule-time validation (raw ndarray → `TypeError` naming `etl.tensor(...)`; non-`core.Device` target → `TypeError` naming `etl.Device(kind, index)`; payload-backed prefetch source → `DeviceError` with the `.to(cpu)` hint; prefetch to cpu:0 or the source device → `DeviceError` "no copy needed"; sink of host ndarray-backed → `DeviceError` "already host data"; sink of cpu-kind payload → `DeviceError` "use t.numpy()" hint; validation failures never start the worker); success flows (prefetch→sink roundtrip, numerics, `.done` transitions, wait-idempotence, `result()` identity — no re-copy); worker failures re-raised VERBATIM at wait/result/wait_all (same exception object) incl. the unregistered-kind provider `DeviceError`; wait timeouts (handle pending, later wait succeeds, `wait_all(timeout=)` likewise); wait_all snapshot excludes late submissions + first-failure-in-issue-order; FIFO completion order in BOTH directions (staggered delays, strictly increasing completion times, serialized total); GC never auto-waits (dropped handles, `gc.collect()` prompt, worker completes independently, wait_all drains GC'd handles); worker-thread guards (`RuntimeError` with pinned message prefixes for prefetch/sink/wait/wait_all from the worker, worker survives guard failures); per-backend provider slot (`register_backend_device_transfer_provider` wins over the flat slot once `_note_backend_transfer_preference` fires, mirroring what `etl.backends.registry.get` calls); interpreter-exit pins via subprocess (os._exit with a 30 s copy in flight returns instantly; normal exit best-effort-drains a finite in-flight copy — never hangs). |

## Constraints

- Device-free / CPU-only / no GPU: fake device kinds (`fio_*` — never `"cuda"`, which
  owns a lazy iree thunk), duck-typed `_DummyPayload`s, and registered
  device-transfer providers (the `tests/core/test_tensor.py` idiom) stand in for
  real devices. No real iree required: `test_io.py` reaches the worker's lazy
  bootstrap only incidentally through its non-cpu fake kinds (the import plus a
  failing `get_driver` for a fake kind are swallowed), while
  `test_io_worker_bootstrap.py` pins the bootstrap itself against a fake
  `iree`/`iree.runtime` installed in `sys.modules`.
  Full suite ~6.5 s incl. 3 subprocess spawns (the two interpreter-exit pins
  + the no-iree import pin, all via `_run_python_snippet`).
- Registries are cleaned in `finally` (`_DEVICE_TRANSFER_PROVIDERS` /
  `_BACKEND_DEVICE_TRANSFER_PROVIDERS` / `_PREFERRED_TRANSFER_BACKENDS` pops); the
  autouse `_drain_outstanding` fixture polls `io._outstanding` empty after every
  test so the process-global worker never bleeds across tests.

## Notes for agents

- **Mechanism notes (pinned by this suite):** ONE serialized FIFO daemon worker
  thread (`name="etl-io-worker"`, lazily started by the first submission; append +
  queue-put in one critical section ⇒ issue order == execution order). `atexit`
  best-effort drain joins FINITE in-flight copies at normal interpreter exit
  (os._exit skips atexit → daemon dies with the process, instant — never hangs).
  First non-cpu-kind copy triggers a one-time lazy best-effort CUDA/ROCm/Vulkan
  primary-context bootstrap (`import iree.runtime`, then
  `get_driver(driver_name).create_device(...)` — the etl kind maps to the iree HAL
  driver name (`"rocm"`/`"hip"` → `"hip"`, else the kind verbatim, so `"vulkan"` stays
  `"vulkan"`); cuda/rocm pass iree's 1-based `device_id = index + 1` KEYWORD, while
  vulkan passes the QUERIED `query_available_devices()[index]` info dict
  POSITIONALLY — Vulkan HAL ids are UUID hashes, not ordinals, and an out-of-range
  index is a silent no-op; all failures swallowed, devices kept alive
  process-globally, at most once per `(kind, index)`). v1 wording: copies are "async
  relative to the caller, no strict same-device overlap guarantee" (the single worker
  serializes everything). Buffer
  lifetime is user-controlled: the worker holds only a bound-method thunk
  (`tensor.to(device)` / `.to(cpu:0)`) — the caller must keep source buffers alive
  until `.wait()`/`.result()`.
- **Spike outcome:** iree-python 3.11 exposes no simple real async host-copy API, so
  the worker-thread mechanism stands as designed — there is NO iree fast path to
  test; the fake-provider/fake-payload machinery (`test_io.py`) plus the fake
  `iree`/`iree.runtime` modules (`test_io_worker_bootstrap.py`) are the whole suite.
- The worker's non-cpu context bootstrap is pinned only via fake `iree` modules
  installed in `sys.modules`: `monkeypatch` restores them afterwards, so
  `test_io.py`'s `test_import_pulls_no_iree` pin stays valid regardless of file
  order, and no test in this dir ever imports the real `iree` or needs a GPU.
- `io._worker_thread` stays `None` until the first successful submission — the
  schedule-time-validation class asserts validation errors never start it or leak
  handles.
- Timing margins are generous (timeout-prompt <0.3 s vs. slow-copy sleeps ≥0.4 s;
  serialization totals ≥ sum−0.05; GC prompt <0.25 s) — flakes should be treated as
  real timing regressions on slow CI.
