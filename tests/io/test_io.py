"""Device-free unit tests for the ``etl.io`` module (host ↔ device async copies).

Pins the committed public surface of ``etl/io.py`` (Workstream B):
``prefetch``/``sink`` return ``AsyncCopy`` handles (``.device``/``.direction``/
``.done``/``.wait``/``.result``), schedule-time validation is synchronous on
the calling thread, runtime failures surface at ``wait()``/``result()``/
``wait_all()`` re-raised verbatim (never swallowed), wait timeouts never
cancel copies (the handle stays pending and re-waitable), GC never
auto-waits, ONE process-global FIFO daemon worker executes the explicit
``Tensor.to`` copies in issue order (both directions), and worker-thread
re-entry raises ``RuntimeError``. Interpreter-exit behavior is pinned with
subprocess tests.

No GPU / no real HAL: fake device kinds (``fio_*`` — never ``"cuda"``, which
owns a lazy iree thunk), duck-typed payloads, and registered device-transfer
providers stand in for real devices (the ``tests/core/test_tensor.py``
idiom). Slow providers/payloads make timeout and GC semantics deterministic.

Run:  python3 -m pytest tests/io/ -q
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import etl  # noqa: F401  (import-order A: parent package first)
import etl.io as io

from etl.core import (
    Device,
    DeviceError,
    Tensor,
    register_backend_device_transfer_provider,
    register_device_transfer_provider,
)
from etl.core.tensor import (
    _BACKEND_DEVICE_TRANSFER_PROVIDERS,
    _DEVICE_TRANSFER_PROVIDERS,
    _PREFERRED_TRANSFER_BACKENDS,
    _note_backend_transfer_preference,
)

# ---------------------------------------------------------------------------
# Fake device kinds + shared helpers
# ---------------------------------------------------------------------------
K_PREF = "fio_pref"        # registered flat-provider kind (prefetch target)
K_SINK = "fio_sink"        # sink-source kind — payload-only, NEVER registered
K_ABSENT = "fio_absent"    # never registered → provider DeviceError at wait()
K_WORKER = "fio_worker"    # provider re-enters etl.io (worker-guard tests)
K_BACKEND = "fio_backend"  # per-backend provider registration kind
BACKEND_NAME = "fio_test_backend"
D_PREF = Device(K_PREF, 0)
D_SINK = Device(K_SINK, 0)
D_ABSENT = Device(K_ABSENT, 0)
D_WORKER = Device(K_WORKER, 0)
D_BACKEND = Device(K_BACKEND, 0)
HOST = Device("cpu", 0)


def _host(values):
    """An ndarray-backed host tensor (Device('cpu', 0))."""
    return Tensor(np.asarray(values))


class _DummyPayload:
    """Minimal duck-typed device payload (the ``core.Tensor`` payload protocol).

    ``.shape``/``.dtype``/``.device`` plus a ``to_host()`` host-copy path
    returning a FRESH ndarray per call; optional per-payload copy delay and
    completion events ``(tag, monotonic)`` make FIFO/ordering pins
    deterministic. Counts ``to_host`` calls so tests can prove no re-copy.
    """

    def __init__(self, shape, dtype, device=None, values=None,
                 to_host_delay=0.0, events=None, tag=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = device
        self._values = (
            np.zeros(shape, self.dtype)
            if values is None
            else np.asarray(values, dtype=self.dtype)
        )
        self.to_host_calls = 0
        self._to_host_delay = to_host_delay
        self._events = events
        self._tag = tag

    def to_host(self):
        self.to_host_calls += 1
        if self._to_host_delay:
            time.sleep(self._to_host_delay)
        if self._events is not None:
            self._events.append((self._tag, time.monotonic()))
        return self._values.copy()

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class _RaisingToHostPayload:
    """A payload whose host-copy path fails (worker-time sink failure)."""

    def __init__(self, shape, dtype, device):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = device

    def to_host(self):
        raise ValueError("dl-boom")


def _place(host_tensor, device):
    """The fake provider's success result: a tensor placed on ``device``."""
    return Tensor(
        _DummyPayload(host_tensor.shape, host_tensor.dtype, device=device,
                      values=host_tensor.data),
        device=device,
    )


def _make_provider(cfg):
    """Flat device-transfer provider driven by per-test knobs in ``cfg``.

    Knobs are snapshotted AT COPY ENTRY (the worker is serialized, so
    entries never overlap): a global ``delay`` or per-host-array delays via
    ``delay_by_id`` (keyed by ``id(host_tensor.data)``), a ONE-SHOT ``fail``
    exception (self-clearing — the first copy that enters while it is set
    takes it, so a later main-thread reset cannot race a sleeping worker),
    plus ``finish_counter`` and ``completions`` [(id(data), monotonic)]
    recorders.
    """

    def provider(host_tensor, device):
        cfg.setdefault("seen", []).append(
            (host_tensor.device, device, isinstance(host_tensor.data, np.ndarray))
        )
        delay = cfg.get("delay_by_id", {}).get(
            id(host_tensor.data), cfg.get("delay", 0.0)
        )
        fail = cfg.get("fail")
        cfg["fail"] = None
        if delay:
            time.sleep(delay)
        if fail is not None:
            raise fail
        if cfg.get("finish_counter") is not None:
            cfg["finish_counter"][0] += 1
        if cfg.get("completions") is not None:
            cfg["completions"].append((id(host_tensor.data), time.monotonic()))
        return _place(host_tensor, device)

    return provider


def _register(kind, cfg):
    register_device_transfer_provider(kind, _make_provider(cfg))


def _unregister(kind):
    _DEVICE_TRANSFER_PROVIDERS.pop(kind, None)


def _wait_until(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.002)


@pytest.fixture(autouse=True)
def _drain_outstanding():
    """Every test leaves the process-global worker drained (no bleed-over).

    Polls the outstanding set empty without ever calling ``wait``/``wait_all``
    (stored failures are irrelevant — the worker removes every completed
    handle on its own).
    """
    yield
    _wait_until(
        lambda: not io._outstanding, 10.0,
        f"etl.io worker never drained its outstanding set: {io._outstanding!r}",
    )


# ---------------------------------------------------------------------------
# 1. Module surface / API shape
# ---------------------------------------------------------------------------
class TestModuleSurface:
    """Public surface of ``etl.io``. The pristine-state pins below hold only
    while no copy was scheduled yet in the process: file order guarantees
    that within this file's own run, but an earlier suite in a full-process
    run (tests/backends/ collects before tests/io/) may already have started
    the process-global worker — those pins skip then (they still run green
    in an isolated `pytest tests/io/` run)."""

    @staticmethod
    def _require_pristine_process():
        if io._worker_thread is not None:
            pytest.skip(
                "module-surface pin requires a pristine process; the etl.io "
                "worker was already started by an earlier suite in this run "
                "(the pin holds in isolated tests/io runs)"
            )

    def test_exports(self):
        assert io.__all__ == ["AsyncCopy", "prefetch", "sink", "wait_all"]
        for name in io.__all__:
            assert hasattr(io, name)

    def test_import_pulls_no_iree(self):
        # Module imports are stdlib + etl.core only; iree.runtime is imported
        # lazily inside the worker's CUDA-context bootstrap and only for
        # non-cpu-kind copies.
        self._require_pristine_process()
        assert io._worker_thread is None
        assert "iree" not in sys.modules

    def test_initial_module_state(self):
        self._require_pristine_process()
        assert io._worker_thread is None
        assert io._outstanding == []
        assert io._jobs.empty()

    def test_handles_have_no_destructor(self):
        # GC never auto-waits: the handle class defines no __del__ at all.
        assert not hasattr(io.AsyncCopy, "__del__")


# ---------------------------------------------------------------------------
# 2. Schedule-time validation (synchronous, on the calling thread)
# ---------------------------------------------------------------------------
class TestScheduleTimeValidation:
    """All of these calls are REJECTED before any worker involvement."""

    @pytest.fixture(autouse=True)
    def _worker_untouched(self):
        before = io._worker_thread
        outstanding = list(io._outstanding)
        yield
        # Validation failures must never start the worker or leak handles.
        assert io._worker_thread is before
        assert io._outstanding == outstanding

    def test_prefetch_raw_ndarray_type_error(self):
        with pytest.raises(TypeError) as ei:
            io.prefetch(np.zeros(3), D_PREF)
        msg = str(ei.value)
        assert "prefetch() expects a core.Tensor host tensor" in msg
        assert "raw numpy ndarray" in msg
        assert "etl.tensor(...)" in msg and "core.Tensor(...)" in msg

    def test_prefetch_non_tensor_type_errors(self):
        for bad, name in [("nope", "str"), (None, "NoneType"), ([1.0], "list")]:
            with pytest.raises(TypeError) as ei:
                io.prefetch(bad, D_PREF)
            msg = str(ei.value)
            assert "prefetch() expects a core.Tensor host tensor" in msg
            assert f"got {name}" in msg

    def test_prefetch_non_device_target_type_error(self):
        h = _host(np.arange(3.0))
        for bad, name in [("cuda:0", "str"), (5, "int"), (None, "NoneType")]:
            with pytest.raises(TypeError) as ei:
                io.prefetch(h, bad)
            msg = str(ei.value)
            assert "prefetch() expects a core.Device target" in msg
            assert f"got {name}" in msg
            assert "etl.Device(kind, index)" in msg

    def test_prefetch_payload_backed_source_device_error(self):
        # A device-payload source (any kind) is not host data → DeviceError
        # with the explicit .to(cpu) remedy (R3: no implicit transfer).
        pt = Tensor(_DummyPayload((2, 3), "float32", device=D_PREF,
                                  values=np.arange(6.0).reshape(2, 3)))
        with pytest.raises(DeviceError) as ei:
            io.prefetch(pt, Device("cuda", 0))  # target irrelevant: source check
        msg = str(ei.value)
        assert "prefetch" in msg and ".to(cpu)" in msg
        # Same for a cpu-kind PAYLOAD source (data is not ndarray-backed).
        cpt = Tensor(_DummyPayload((2,), "float32", device=HOST, values=[1.0, 2.0]))
        with pytest.raises(DeviceError) as ei:
            io.prefetch(cpt, D_PREF)
        assert ".to(cpu)" in str(ei.value)

    def test_prefetch_cpu_or_same_device_no_copy_needed(self):
        h = _host(np.arange(3.0))
        with pytest.raises(DeviceError) as ei:
            io.prefetch(h, HOST)
        assert "no copy needed" in str(ei.value)
        with pytest.raises(DeviceError) as ei:
            io.prefetch(h, h.device)
        assert "no copy needed" in str(ei.value)

    def test_sink_non_tensor_type_errors(self):
        for bad, name in [("nope", "str"), (None, "NoneType")]:
            with pytest.raises(TypeError) as ei:
                io.sink(bad)
            msg = str(ei.value)
            assert "sink() expects a core.Tensor device tensor" in msg
            assert f"got {name}" in msg

    def test_sink_raw_ndarray_type_error(self):
        with pytest.raises(TypeError) as ei:
            io.sink(np.zeros(3))
        msg = str(ei.value)
        assert "sink() expects a core.Tensor device tensor" in msg
        assert "raw numpy ndarray" in msg
        assert "etl.tensor(...)" in msg and "core.Tensor(...)" in msg

    def test_sink_host_ndarray_backed_device_error(self):
        h = _host(np.arange(3.0))
        with pytest.raises(DeviceError) as ei:
            io.sink(h)
        msg = str(ei.value)
        assert "already host data" in msg and "no download needed" in msg

    def test_sink_cpu_kind_payload_device_error(self):
        # cpu-kind payload on cpu:0: data reads from host memory → the
        # synchronous .numpy() path is the remedy, never an async download.
        cpt = Tensor(_DummyPayload((2,), "float32", device=HOST, values=[1.0, 2.0]))
        with pytest.raises(DeviceError) as ei:
            io.sink(cpt)
        assert ".numpy()" in str(ei.value)


# ---------------------------------------------------------------------------
# 3. Successful prefetch / sink flows — handle API + numerics (group 1 & 7)
# ---------------------------------------------------------------------------
class TestHandleApiAndSuccessFlows:
    """First scheduled copies in the process (the worker lazily starts here;
    the non-cpu kind also triggers-and-swallows the lazy CUDA bootstrap)."""

    def test_prefetch_handle_done_transitions_and_sink_roundtrip(self):
        cfg = {"seen": [], "delay": 0.3}
        _register(K_PREF, cfg)
        try:
            vals = np.arange(12.0).reshape(3, 4)
            src = _host(vals)
            up = io.prefetch(src, D_PREF)
            # Handle API shape + .done transitions (group 1).
            assert isinstance(up, io.AsyncCopy)
            assert up.direction == "host_to_device"
            assert up.device == D_PREF
            assert isinstance(up.done, bool)
            assert not up.done  # slow provider: nothing completed this fast
            assert up.wait() is None
            assert up.done
            # .done stays True on repeated polls (plain Event.is_set).
            assert up.done
            r = up.result()
            assert isinstance(r, Tensor) and r.device == D_PREF
            assert r.shape == (3, 4)
            assert np.array_equal(r.data.to_host(), vals)
            # wait() idempotent + result() identity — no re-copy (group 7).
            assert up.result() is r
            assert up.wait() is None and up.wait() is None
            # Provider received the ndarray-backed HOST tensor + target device.
            seen = cfg["seen"]
            assert seen and seen[0][0] == HOST
            assert seen[0][1] == D_PREF
            assert seen[0][2]  # host_tensor.data is a real ndarray

            # Sink the device tensor back to host.
            cfg["delay"] = 0.0
            to_host_before = r.data.to_host_calls  # 1 (manual check above)
            s = io.sink(r)
            assert isinstance(s, io.AsyncCopy)
            assert s.direction == "device_to_host"
            assert s.device == D_PREF  # the SOURCE tensor's device
            back = s.result()
            assert isinstance(back, Tensor)
            assert isinstance(back.data, np.ndarray)
            assert back.device == HOST
            assert np.array_equal(back.data, vals)
            assert r.data.to_host_calls == to_host_before + 1  # exactly once
            # result() twice → same object, no second host copy (group 7).
            assert s.result() is back
            assert r.data.to_host_calls == to_host_before + 1
            assert s.wait() is None  # idempotent wait on a done handle
            assert s.done
            assert io._outstanding == []  # worker drains its own bookkeeping
        finally:
            _unregister(K_PREF)

    def test_prefetch_to_unregistered_kind_is_a_runtime_not_schedule_failure(self):
        # Provider absence is deliberately NOT checked by prefetch() — async
        # IO schedules fine and the copy fails on the worker (group 2 last
        # bullet: pin the error surfacing at wait()).
        h = io.prefetch(_host(np.arange(4.0)), D_ABSENT)
        assert isinstance(h, io.AsyncCopy)
        assert not h.done

    def test_sink_needs_no_provider(self):
        # The device → host path goes through the payload's to_host(), never
        # the device-transfer provider registry: a sink from an UNREGISTERED
        # kind succeeds (positive control for the unregistered-kind flow).
        t = Tensor(_DummyPayload((3,), "float32", device=D_SINK,
                                 values=[1.0, 2.0, 3.0]))
        back = io.sink(t).result()
        assert isinstance(back.data, np.ndarray) and back.device == HOST
        assert np.array_equal(back.data, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 4. Runtime failures surface at wait()/result(), verbatim (group 3)
# ---------------------------------------------------------------------------
class TestRuntimeFailuresSurfaceAtWait:
    """Worker-time failures are stored on the handle and RE-RAISED on every
    wait()/result() — same exception object, never wrapped, never swallowed;
    the scheduling call itself never raises."""

    def test_unregistered_kind_fails_at_wait_with_provider_device_error(self):
        h = io.prefetch(_host(np.arange(4.0)), D_ABSENT)
        with pytest.raises(DeviceError) as ei:
            h.wait()
        msg = str(ei.value)
        assert "Tensor.to cannot place data on" in msg
        assert "no device-transfer provider is registered" in msg
        assert K_ABSENT in msg
        # Re-raised verbatim on every call (same object).
        with pytest.raises(DeviceError) as ei2:
            h.wait()
        assert ei2.value is ei.value
        with pytest.raises(DeviceError) as ei3:
            h.result()
        assert ei3.value is ei.value
        assert h.done  # .done True after a FAILURE too

    def test_provider_failure_re_raised_verbatim_at_wait_and_result(self):
        cfg = {"fail": ValueError("boom-copy"), "delay": 0.05}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(2.0)), D_PREF)  # schedule: no raise
            with pytest.raises(ValueError) as ei:
                h.wait()
            assert "boom-copy" in str(ei.value)
            with pytest.raises(ValueError) as ei2:
                h.wait()
            assert ei2.value is ei.value  # same stored exception object
            with pytest.raises(ValueError) as ei3:
                h.result()
            assert ei3.value is ei.value
            assert h.done
        finally:
            _unregister(K_PREF)

    def test_sink_to_host_failure_surfaces_at_wait(self):
        # The device → host copy calls payload.to_host() on the worker; a
        # payload failure is stored and re-raised at wait() — never wrapped.
        t = Tensor(_RaisingToHostPayload((3,), "float32", device=D_SINK))
        s = io.sink(t)
        assert s.direction == "device_to_host" and s.device == D_SINK
        with pytest.raises(ValueError, match="dl-boom"):
            s.wait()
        with pytest.raises(ValueError, match="dl-boom"):
            s.result()
        assert s.done


# ---------------------------------------------------------------------------
# 5. Timeouts (group 4): builtin TimeoutError; handle stays pending, re-waitable
# ---------------------------------------------------------------------------
class TestWaitTimeouts:

    def test_wait_timeout_handle_stays_pending_and_rewaitable(self):
        cfg = {"delay": 0.4}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(4.0)), D_PREF)
            t0 = time.monotonic()
            with pytest.raises(TimeoutError) as ei:
                h.wait(0.05)
            assert time.monotonic() - t0 < 0.3  # prompt: no blocking until done
            msg = str(ei.value)
            assert "timed out after 0.05s" in msg
            assert "never cancel" in msg
            assert not h.done  # the copy is still running
            r = h.result()  # later wait succeeds once the slow copy finishes
            assert h.done and r is not None
            assert h.wait() is None  # idempotent after completion
        finally:
            _unregister(K_PREF)

    def test_result_timeout_then_success(self):
        cfg = {"delay": 0.4}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(2.0)), D_PREF)
            with pytest.raises(TimeoutError):
                h.result(0.05)
            assert not h.done
            r = h.result()
            assert r is not None and h.done
        finally:
            _unregister(K_PREF)

    def test_timeout_fires_before_a_slow_failure_then_reraises(self):
        cfg = {"fail": RuntimeError("boom-copy"), "delay": 0.3}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(2.0)), D_PREF)
            with pytest.raises(TimeoutError):
                h.wait(0.02)  # timeout fires while the copy is still sleeping
            with pytest.raises(RuntimeError) as ei:
                h.wait()  # ...later the stored failure surfaces
            assert "boom-copy" in str(ei.value)
            with pytest.raises(RuntimeError):
                h.wait()  # re-raised again — never swallowed
            with pytest.raises(RuntimeError):
                h.result()
            assert h.done
        finally:
            _unregister(K_PREF)

    def test_wait_all_timeout_then_retry_drains(self):
        cfg = {"delay": 0.4}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_PREF)
            t0 = time.monotonic()
            with pytest.raises(TimeoutError) as ei:
                io.wait_all(0.05)
            assert time.monotonic() - t0 < 0.3
            assert "wait_all() timed out" in str(ei.value)
            assert not h.done  # pending copies stay pending
            assert io.wait_all() is None  # retry drains
            assert h.done
            assert io.wait_all() is None  # empty outstanding set → None
        finally:
            _unregister(K_PREF)


# ---------------------------------------------------------------------------
# 6. wait_all semantics: snapshot consistency + first-failure re-raise
# ---------------------------------------------------------------------------
class TestWaitAll:

    def test_snapshot_excludes_later_submissions(self):
        cfg = {"delay": 0.7}
        _register(K_PREF, cfg)
        try:
            h_a = io.prefetch(_host(np.arange(1.0)), D_PREF)
            holder = {}

            def submit_late():
                time.sleep(0.25)  # lands while wait_all() is blocking on h_a
                holder["h_b"] = io.prefetch(_host(np.arange(2.0)), D_PREF)

            threading.Thread(target=submit_late, daemon=True).start()
            io.wait_all()  # snapshots [h_a] only
            h_b = holder["h_b"]
            assert h_a.done
            assert not h_b.done  # submitted mid-wait_all → NOT in the snapshot
            h_b.wait()  # drain the late copy
            assert h_b.done
        finally:
            _unregister(K_PREF)

    def test_wait_all_re_raises_first_failure_in_issue_order(self):
        cfg = {"seen": [], "fail": RuntimeError("fail-early"), "delay": 0.15}
        _register(K_PREF, cfg)
        try:
            n_seen = len(cfg["seen"])
            fa = io.prefetch(_host(np.arange(1.0)), D_PREF)  # takes the flag
            _wait_until(lambda: len(cfg["seen"]) > n_seen, 2.0,
                        "worker never entered fa's copy")
            cfg["fail"] = None  # already consumed at fa's entry — guard only
            cfg["delay"] = 0.05
            fb = io.prefetch(_host(np.arange(1.0)), D_PREF)  # succeeds
            with pytest.raises(RuntimeError) as ei:
                io.wait_all()
            assert "fail-early" in str(ei.value)
            assert fa.done
            fb.wait()  # fb completes right after fa fails (serialized worker)
            assert fb.done and fb.result() is not None
        finally:
            _unregister(K_PREF)


# ---------------------------------------------------------------------------
# 7. FIFO ordering determinism (group 6) — completion order == issue order
# ---------------------------------------------------------------------------
class TestFifoOrdering:

    def test_sink_copies_complete_in_issue_order(self):
        # Staggered payload host-copy delays: the serialized worker MUST
        # complete s1 before s2 before s3 despite the reversed delays.
        events = []
        t1 = Tensor(_DummyPayload((4,), "float32", device=D_SINK,
                                  values=[1.0, 2.0, 3.0, 4.0],
                                  to_host_delay=0.2, events=events, tag="s1"))
        t2 = Tensor(_DummyPayload((4,), "float32", device=D_SINK,
                                  values=[5.0, 6.0, 7.0, 8.0],
                                  to_host_delay=0.12, events=events, tag="s2"))
        t3 = Tensor(_DummyPayload((4,), "float32", device=D_SINK,
                                  values=[9.0, 10.0, 11.0, 12.0],
                                  to_host_delay=0.05, events=events, tag="s3"))
        a = io.sink(t1)
        b = io.sink(t2)
        c = io.sink(t3)
        t0 = time.monotonic()
        assert io.wait_all() is None
        elapsed = time.monotonic() - t0
        assert [ev[0] for ev in events] == ["s1", "s2", "s3"], str(events)
        # Strictly serialized: completion times strictly increase and the
        # total covers the sum of the staggered delays.
        assert all(events[i][1] < events[i + 1][1]
                   for i in range(len(events) - 1))
        assert elapsed >= 0.2 + 0.12 + 0.05 - 0.05, f"elapsed {elapsed:.3f}s"
        # Results correct in FIFO order.
        assert np.array_equal(a.result().data, [1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(b.result().data, [5.0, 6.0, 7.0, 8.0])
        assert np.array_equal(c.result().data, [9.0, 10.0, 11.0, 12.0])

    def test_prefetch_copies_complete_in_issue_order(self):
        # Same pin for host → device, via per-host-array provider delays.
        arrs = [np.arange(1.0), np.arange(2.0), np.arange(3.0)]
        cfg = {
            "delay_by_id": {id(arrs[0]): 0.15, id(arrs[1]): 0.1,
                            id(arrs[2]): 0.05},
            "completions": [],
        }
        _register(K_PREF, cfg)
        try:
            issue = [id(a) for a in arrs]
            hs = [io.prefetch(_host(a), D_PREF) for a in arrs]
            t0 = time.monotonic()
            assert io.wait_all() is None
            elapsed = time.monotonic() - t0
            comps = cfg["completions"]
            assert [c[0] for c in comps] == issue, str(comps)
            assert all(comps[i][1] < comps[i + 1][1]
                       for i in range(len(comps) - 1))
            assert elapsed >= 0.15 + 0.1 + 0.05 - 0.05, f"elapsed {elapsed:.3f}s"
            for h, a in zip(hs, arrs):
                assert np.array_equal(h.result().data.to_host(), a)
        finally:
            _unregister(K_PREF)


# ---------------------------------------------------------------------------
# 8. GC never auto-waits (group 5a)
# ---------------------------------------------------------------------------
class TestGcNeverAutoWaits:

    def test_dropped_handle_gc_collect_never_blocks_or_raises(self):
        counter = [0]
        cfg = {"finish_counter": counter, "delay": 0.4}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_PREF)
            del h
            t0 = time.monotonic()
            gc.collect()  # must return promptly: no blocking destructor
            assert time.monotonic() - t0 < 0.25
            # The worker completes the copy independently of any handle refs.
            _wait_until(lambda: counter[0] == 1, 3.0,
                        "worker never finished the dropped copy")
            assert io._outstanding == []  # worker removed it from the set
        finally:
            _unregister(K_PREF)

    def test_wait_all_drains_a_gc_d_pending_handle(self):
        counter = [0]
        cfg = {"finish_counter": counter, "delay": 0.25}
        _register(K_PREF, cfg)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_PREF)
            del h
            gc.collect()
            assert io.wait_all() is None  # drains even without user refs
            assert counter[0] == 1
        finally:
            _unregister(K_PREF)

    def test_multiple_dropped_handles_complete_independently(self):
        counter = [0]
        cfg = {"finish_counter": counter, "delay": 0.2}
        _register(K_PREF, cfg)
        try:
            hs = [io.prefetch(_host(np.arange(i + 1.0)), D_PREF) for i in range(3)]
            del hs
            gc.collect()  # no blocking, no exception
            _wait_until(lambda: counter[0] == 3, 3.0,
                        "worker never finished the dropped copies")
            assert io._outstanding == []
        finally:
            _unregister(K_PREF)


# ---------------------------------------------------------------------------
# 9. Worker-thread guards (group 8): RuntimeError with the pinned prefix
# ---------------------------------------------------------------------------
class TestWorkerThreadGuards:
    """Re-entering etl.io from the worker thread would deadlock on its own
    queue — every entry point raises RuntimeError naming the worker thread."""

    def test_submit_from_worker_raises_runtime_error(self):
        def prefetch_provider(host_tensor, device):  # runs ON the worker
            io.prefetch(host_tensor, device)  # deadlock guard fires here
            return host_tensor  # pragma: no cover

        def sink_provider(host_tensor, device):  # runs ON the worker
            io.sink(np.zeros(3))  # guard fires BEFORE any arg validation
            return host_tensor  # pragma: no cover

        register_device_transfer_provider(K_WORKER, prefetch_provider)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_WORKER)
            with pytest.raises(RuntimeError) as ei:
                h.wait()
            assert str(ei.value).startswith(
                "prefetch() cannot run on the etl.io worker thread"
            )
            assert h.done
        finally:
            _DEVICE_TRANSFER_PROVIDERS.pop(K_WORKER, None)

        register_device_transfer_provider(K_WORKER, sink_provider)
        try:
            h2 = io.prefetch(_host(np.arange(1.0)), D_WORKER)
            with pytest.raises(RuntimeError) as ei:
                h2.wait()
            assert str(ei.value).startswith(
                "sink() cannot run on the etl.io worker thread"
            )
            assert h2.done
        finally:
            _DEVICE_TRANSFER_PROVIDERS.pop(K_WORKER, None)

    def test_wait_from_worker_raises_runtime_error(self):
        holder = {}

        def provider(host_tensor, device):  # runs ON the worker
            deadline = time.monotonic() + 5.0
            while holder.get("h") is None and time.monotonic() < deadline:
                time.sleep(0.001)
            h = holder.get("h")
            if h is None:
                raise RuntimeError("provider never saw the handle")
            h.wait()  # this copy is still pending → deadlock guard fires
            return host_tensor  # pragma: no cover

        register_device_transfer_provider(K_WORKER, provider)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_WORKER)
            holder["h"] = h  # provider busy-waits until this is set
            with pytest.raises(RuntimeError) as ei:
                h.wait()
            assert str(ei.value).startswith(
                "AsyncCopy.wait() would deadlock on the etl.io worker thread"
            )
            assert h.done
        finally:
            _DEVICE_TRANSFER_PROVIDERS.pop(K_WORKER, None)

    def test_wait_all_from_worker_raises_runtime_error(self):
        def provider(host_tensor, device):  # runs ON the worker
            io.wait_all()  # non-empty snapshot (this copy) → guard fires
            return host_tensor  # pragma: no cover

        register_device_transfer_provider(K_WORKER, provider)
        try:
            h = io.prefetch(_host(np.arange(1.0)), D_WORKER)
            with pytest.raises(RuntimeError) as ei:
                h.wait()
            assert str(ei.value).startswith(
                "wait_all() cannot run on the etl.io worker thread"
            )
            assert h.done
        finally:
            _DEVICE_TRANSFER_PROVIDERS.pop(K_WORKER, None)

    def test_worker_survives_guard_failures(self):
        # Guard failures are stored per-handle; the worker keeps processing.
        _register(K_PREF, {})
        try:
            assert io.prefetch(_host(np.arange(1.0)), D_PREF).wait() is None
        finally:
            _unregister(K_PREF)


# ---------------------------------------------------------------------------
# 10. Per-backend provider registration (register_backend_device_transfer_provider)
# ---------------------------------------------------------------------------
class TestPerBackendProviderSlot:
    """``Tensor.to`` resolves the PREFERRED backend's per-backend slot before
    the flat default slot — pinned end-to-end through ``etl.io.prefetch``."""

    def test_prefetch_uses_preferred_backend_slot_over_flat(self):
        seen = []

        def per_backend_provider(host_tensor, device):
            seen.append((host_tensor.device, device))
            return _place(host_tensor, device)

        def flat_provider(host_tensor, device):  # pragma: no cover — must NOT run
            raise AssertionError("the flat provider must not be consulted")

        register_backend_device_transfer_provider(
            K_BACKEND, BACKEND_NAME, per_backend_provider
        )
        register_device_transfer_provider(K_BACKEND, flat_provider)
        try:
            # The preference is recorded ONLY via the registry's internal
            # note hook (what etl.backends.registry.get calls on lookup).
            _note_backend_transfer_preference(BACKEND_NAME)
            r = io.prefetch(_host(np.arange(6.0)), D_BACKEND).result()
            assert r.device == D_BACKEND
            assert np.array_equal(r.data.to_host(), np.arange(6.0))
            assert len(seen) == 1
            assert seen[0][0] == HOST and seen[0][1] == D_BACKEND
        finally:
            _BACKEND_DEVICE_TRANSFER_PROVIDERS.pop(
                (K_BACKEND, BACKEND_NAME), None
            )
            _PREFERRED_TRANSFER_BACKENDS.pop(K_BACKEND, None)
            _DEVICE_TRANSFER_PROVIDERS.pop(K_BACKEND, None)


# ---------------------------------------------------------------------------
# 11. Interpreter exit never hangs (group 5b; subprocess pins)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Child 1: a 30 s copy is in flight; os._exit(0) must return immediately
# (skips atexit — the daemon worker dies with the process, nothing is
# joined). Exit code 0 well before the copy would have finished proves the
# process never waited on the worker.
_CHILD_OS_EXIT = """
import os
import threading
import time

import numpy as np

import etl  # noqa: F401
import etl.io as io
from etl.core import Device, Tensor, register_device_transfer_provider

STARTED = threading.Event()


def provider(host_tensor, device):
    STARTED.set()
    time.sleep(30.0)  # far longer than any test window
    return host_tensor


register_device_transfer_provider("fio_exit", provider)
io.prefetch(Tensor(np.arange(3.0)), Device("fio_exit", 0))
if not STARTED.wait(timeout=10.0):
    os._exit(3)  # the worker never started the copy
os._exit(0)  # instant: the daemon worker dies with the process
"""

# Child 2: a FINITE (0.5 s) copy is in flight when the script ends normally.
# Normal interpreter exit runs the atexit hook, which best-effort drains the
# in-flight copy (join) — so the process must exit 0 promptly and without
# error. (Do NOT assert that normal exit with an infinite copy is instant:
# the atexit drain joins finite in-flight copies by design.)
_CHILD_NORMAL_EXIT = """
import os
import threading
import time

import numpy as np

import etl  # noqa: F401
import etl.io as io
from etl.core import Device, Tensor, register_device_transfer_provider

STARTED = threading.Event()


def provider(host_tensor, device):
    STARTED.set()
    time.sleep(0.5)
    return host_tensor


register_device_transfer_provider("fio_exit", provider)
io.prefetch(Tensor(np.arange(3.0)), Device("fio_exit", 0))
if not STARTED.wait(timeout=10.0):
    os._exit(3)
# fall off the end while the copy is still in flight
"""


class TestInterpreterExitNeverHangs:

    def _run_child(self, code, timeout):
        env = dict(os.environ)
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return proc, time.monotonic() - t0

    def test_os_exit_with_inflight_copy_never_blocks(self):
        # The daemon worker must never keep the process alive: exiting with
        # os._exit(0) while a 30 s copy runs returns in ~seconds (the child
        # would otherwise be killed by the 15 s run timeout at the earliest).
        proc, elapsed = self._run_child(_CHILD_OS_EXIT, timeout=15)
        assert proc.returncode == 0, proc.stderr
        assert elapsed < 12, f"child took {elapsed:.1f}s to exit"

    def test_normal_exit_drains_a_finite_inflight_copy(self):
        # Normal exit with a finite in-flight copy: the atexit best-effort
        # drain joins it and the process exits 0 promptly — no hang, no error.
        proc, elapsed = self._run_child(_CHILD_NORMAL_EXIT, timeout=15)
        assert proc.returncode == 0, proc.stderr
        assert elapsed < 10, f"child took {elapsed:.1f}s to exit"
