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
| `test_io.py` | 37 tests: module surface/exports (incl. no iree at import, no handle `__del__`); schedule-time validation (raw ndarray → `TypeError` naming `etl.tensor(...)`; non-`core.Device` target → `TypeError` naming `etl.Device(kind, index)`; payload-backed prefetch source → `DeviceError` with the `.to(cpu)` hint; prefetch to cpu:0 or the source device → `DeviceError` "no copy needed"; sink of host ndarray-backed → `DeviceError` "already host data"; sink of cpu-kind payload → `DeviceError` "use t.numpy()" hint; validation failures never start the worker); success flows (prefetch→sink roundtrip, numerics, `.done` transitions, wait-idempotence, `result()` identity — no re-copy); worker failures re-raised VERBATIM at wait/result/wait_all (same exception object) incl. the unregistered-kind provider `DeviceError`; wait timeouts (handle pending, later wait succeeds, `wait_all(timeout=)` likewise); wait_all snapshot excludes late submissions + first-failure-in-issue-order; FIFO completion order in BOTH directions (staggered delays, strictly increasing completion times, serialized total); GC never auto-waits (dropped handles, `gc.collect()` prompt, worker completes independently, wait_all drains GC'd handles); worker-thread guards (`RuntimeError` with pinned message prefixes for prefetch/sink/wait/wait_all from the worker, worker survives guard failures); per-backend provider slot (`register_backend_device_transfer_provider` wins over the flat slot once `_note_backend_transfer_preference` fires, mirroring what `etl.backends.registry.get` calls); interpreter-exit pins via subprocess (os._exit with a 30 s copy in flight returns instantly; normal exit best-effort-drains a finite in-flight copy — never hangs). |

## Constraints

- Device-free / CPU-only / no GPU: fake device kinds (`fio_*` — never `"cuda"`, which
  owns a lazy iree thunk), duck-typed `_DummyPayload`s, and registered
  device-transfer providers (the `tests/core/test_tensor.py` idiom) stand in for
  real devices. No real iree required (the worker's lazy bootstrap merely
  attempts-and-swallows a fake-driver create when iree happens to be importable).
  Full suite ~7 s incl. 2 subprocess spawns.
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
  First non-cpu-kind copy triggers a one-time lazy best-effort CUDA primary-context
  bootstrap (`import iree.runtime`, `get_driver(kind).create_device(device_id=index
  + 1)` — iree's 1-based mapping; all failures swallowed, devices kept alive
  process-globally). v1 wording: copies are "async relative to the caller, no strict
  same-device overlap guarantee" (the single worker serializes everything). Buffer
  lifetime is user-controlled: the worker holds only a bound-method thunk
  (`tensor.to(device)` / `.to(cpu:0)`) — the caller must keep source buffers alive
  until `.wait()`/`.result()`.
- **Spike outcome:** iree-python 3.11 exposes no simple real async host-copy API, so
  the worker-thread mechanism stands as designed — there is NO iree fast path to
  test; the fake-provider/fake-payload machinery above is the whole suite.
- `io._worker_thread` stays `None` until the first successful submission — the
  schedule-time-validation class asserts validation errors never start it or leak
  handles.
- Timing margins are generous (timeout-prompt <0.3 s vs. slow-copy sleeps ≥0.4 s;
  serialization totals ≥ sum−0.05; GC prompt <0.25 s) — flakes should be treated as
  real timing regressions on slow CI.
