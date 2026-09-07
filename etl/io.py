"""etl.io — explicit host-data async IO sugar.

Schedule explicit ``Tensor.to`` copies between host (ndarray-backed,
``Device("cpu", 0)``) tensors and device tensors so they execute off the
calling thread::

    upload = etl.io.prefetch(host_tensor, etl.Device("cuda", 0))  # host → device
    ...
    dev_tensor = upload.result()           # wait + the device tensor

    download = etl.io.sink(dev_tensor)     # device → host
    host_tensor = download.result()        # wait + the ndarray-backed host tensor

There is no hidden staging: these are the same explicit transfers
``Tensor.to`` performs synchronously, merely executed on a background
worker. Schedule-time validation is synchronous on the calling thread;
runtime failures (a missing device-transfer provider, a ``DeviceError``
from the copy itself, …) surface at ``wait()``/``result()``/``wait_all()``.

Mechanism (spike outcome)
-------------------------
iree-python 3.11 exposes no simple async host-copy API — only raw HAL
command-buffer/fence machinery — so the worker-thread design stands: ONE
process-global FIFO worker thread (a ``queue.Queue``, lazily started at the
first submission) executes every scheduled copy strictly in issue order
(host → device through the registered device-transfer provider, device →
host through the payload host-copy path). FIFO order keeps completion
deterministic in both directions with no multi-threaded HAL contention.
Copies are *async relative to the caller; no strict same-device overlap
guarantee* — true overlap needs runtime async-engine support (a v1
limitation; the FIFO worker deliberately serializes). The worker never
dies and never raises: per-copy failures are stored on the handle and
re-raised by every ``wait()``/``result()``/``wait_all()`` call; a wait
timeout only releases the caller — it never cancels the copy. GC never
auto-waits: handles have no blocking destructor; the worker completes the
copy independently (completion removes the handle from the module's
outstanding set and releases the worker's job references, so buffers are
held only while a user handle keeps them alive).

CUDA worker bootstrap (decision)
--------------------------------
Concurrent use of an llvm-cpu HAL device from a worker thread is safe, but
on CUDA a fresh thread's FIRST allocating HAL op fails with
``CUDA_ERROR_INVALID_CONTEXT``: iree does not push its stored context per
call, so the CUDA primary context must be made current on the worker
thread. The spike-verified mitigation is that the thread creates its own
throwaway cuda HAL device on the same GPU, which makes the shared primary
context current there. DECISION: this module performs that bootstrap
itself — lazily, best-effort, at most once per worker per ``(kind, index)``,
immediately before the first scheduled copy whose involved device kind is
not ``"cpu"`` — using the iree adapter's exact 1-based HAL device-id
mapping (``device_id = device.index + 1``). Rationale: the iree adapter's
upload (host → device) provider is already foreign-thread-safe — it
creates its own HAL device on the calling thread per call, establishing
the context as a side effect — but the download (device → host) path calls
``payload.to_host()`` on a main-thread-created device with no worker-side
device creation and would fail; and either direction may be scheduled
first. The bootstrap runs only when a copy touches a non-``"cpu"``-kind
device (so numpy-only environments never import iree), swallows every
failure (a failed bootstrap must never mask the copy's own error, which
surfaces at ``wait()``), and keeps its throwaway devices alive for the
worker's lifetime.

Numpy-only workloads
--------------------
``prefetch()``/``sink()`` schedule fine, but a host → device copy needs a
registered device-transfer provider for the target kind (``etl.backends``
installs one for cuda; see ``core.register_device_transfer_provider``).
Without one the copy fails at ``wait()``/``result()`` with the provider
``DeviceError`` — async IO is for real device ↔ host boundaries.

Threading policy
----------------
The worker is a single daemon thread: interpreter exit can never block on
it. An ``atexit``-registered sentinel drains pending copies and stops the
worker cleanly when atexit handlers run before interpreter shutdown; when
they do not (CPython may run ``threading._shutdown`` first), the daemon
worker simply dies with the interpreter — pending copies are abandoned,
never waited on. Submitting from the worker thread itself — or blocking
there on a still-pending copy — raises ``RuntimeError`` (it would deadlock
on its own queue). Module-level imports are stdlib + ``etl.core`` only;
``iree.runtime`` is imported lazily inside the bootstrap function (etl's
lazy-optional-dependency discipline — ``import etl.io`` stays light).
"""
from __future__ import annotations

import atexit
import queue
import threading
import time

from . import core

__all__ = ["AsyncCopy", "prefetch", "sink", "wait_all"]

_HOST = core.Device("cpu", 0)

# --- process-global FIFO worker state ---------------------------------------
_jobs = queue.Queue()  # FIFO of (AsyncCopy, thunk) jobs; _SENTINEL stops the worker
_worker_thread = None  # type: threading.Thread | None
_state_lock = threading.Lock()  # guards worker start + the outstanding set
_outstanding = []  # pending AsyncCopy handles, in issue order
_bootstrapped = set()  # worker-side: (kind, index) contexts already bootstrapped
_bootstrap_devices = []  # keep the worker's throwaway HAL devices alive
_SENTINEL = object()


def _reject_worker_submission(name):
    """Raise RuntimeError when a scheduling call runs on the worker thread."""
    if threading.current_thread() is _worker_thread:
        raise RuntimeError(
            f"{name}() cannot run on the etl.io worker thread — the worker "
            "executes queued copies and waiting on one would deadlock (only "
            "the worker can complete pending copies); submit from the "
            "calling thread instead"
        )


def _raise_not_tensor(fn_name, role, obj):
    """Raise the canonical builtin TypeError for a non-``core.Tensor`` arg."""
    import numpy as np  # lazy: error paths only (keeps module imports light)

    if isinstance(obj, np.ndarray):
        raise TypeError(
            f"{fn_name}() expects a core.Tensor {role}, got a raw numpy "
            "ndarray — wrap it with etl.tensor(...) (or core.Tensor(...)) first"
        )
    raise TypeError(
        f"{fn_name}() expects a core.Tensor {role}, got {type(obj).__name__}"
    )


def prefetch(host_tensor, device):
    """Schedule a host → device upload; returns an :class:`AsyncCopy`.

    ``host_tensor`` must be a ``core.Tensor`` whose data is ndarray-backed
    host memory on ``Device("cpu", 0)`` and ``device`` a ``core.Device`` —
    both validated synchronously on the calling thread. The copy itself
    (``host_tensor.to(device)``) runs on the worker thread; the handle's
    ``.device`` is the target device. Provider absence is NOT checked here —
    it is a runtime failure surfacing at ``wait()``/``result()``.
    """
    _reject_worker_submission("prefetch")
    if not isinstance(host_tensor, core.Tensor):
        _raise_not_tensor("prefetch", "host tensor", host_tensor)
    if not isinstance(device, core.Device):
        raise TypeError(
            f"prefetch() expects a core.Device target, got {type(device).__name__}"
            " — construct one with etl.Device(kind, index)"
        )
    import numpy as np  # sys.modules hit; etl.core already imports numpy

    data = host_tensor.data
    if host_tensor.device != _HOST or not isinstance(data, np.ndarray):
        raise core.DeviceError(
            "prefetch() copies host → device and needs ndarray-backed host "
            f"data, but the source tensor is backed by {type(data).__name__} "
            f"on {host_tensor.device!r}. Materialize host data explicitly "
            "first: t.to(core.Device('cpu', 0)) (t.to(cpu))"
        )
    if device == _HOST or device == host_tensor.device:
        raise core.DeviceError(
            f"prefetch() from {host_tensor.device!r} to {device!r}: no copy "
            "needed — host data is already at the target"
        )
    handle = AsyncCopy(device, "host_to_device")
    _submit(handle, lambda: host_tensor.to(device))
    return handle


def sink(device_tensor):
    """Schedule a device → host download; returns an :class:`AsyncCopy`.

    ``device_tensor`` must be a ``core.Tensor`` resident on a non-cpu device
    (validated synchronously on the calling thread). The copy itself
    (``device_tensor.to(core.Device("cpu", 0))``) runs on the worker thread;
    the handle's ``.device`` is the SOURCE tensor's device.
    """
    _reject_worker_submission("sink")
    if not isinstance(device_tensor, core.Tensor):
        _raise_not_tensor("sink", "device tensor", device_tensor)
    if device_tensor.device == _HOST:
        import numpy as np  # lazy: error path only

        if isinstance(device_tensor.data, np.ndarray):
            raise core.DeviceError(
                "sink() downloads device → host data, but the source tensor "
                "is already host data (ndarray-backed on Device('cpu', 0)) "
                "— no download needed"
            )
        raise core.DeviceError(
            "sink() downloads device → host data, but the source tensor sits "
            "on Device('cpu', 0) with a cpu-kind payload whose data reads "
            "directly from host memory — use t.numpy() instead of an async "
            "download"
        )
    handle = AsyncCopy(device_tensor.device, "device_to_host")
    _submit(handle, lambda: device_tensor.to(_HOST))
    return handle


class AsyncCopy:
    """A scheduled host ↔ device copy (returned by ``prefetch()``/``sink()``).

    Construction is internal — ``prefetch()``/``sink()`` are the only
    creators. Read-only public attrs: ``.device`` (the involved
    ``core.Device`` — the copy target for ``"host_to_device"``, the source
    for ``"device_to_host"``) and ``.direction`` (exactly one of those two
    strings). ``.done`` is a non-blocking completion poll (True after
    success OR failure). ``.wait(timeout=None)`` blocks until the copy
    finishes and returns ``None``; it RE-RAISES the copy's stored exception
    on every call (failures are never swallowed) and raises builtin
    ``TimeoutError`` when ``timeout`` seconds pass while the copy stays
    pending (the handle may be waited again; a later wait re-raises the
    exception if the copy subsequently failed). ``.result(timeout=None)``
    is ``wait()`` plus the copy's result tensor (prefetch: the device
    tensor; sink: the host ndarray-backed tensor). There is no destructor
    behavior: dropping a handle never blocks, joins, or raises — the worker
    completes the copy independently.
    """

    __slots__ = ("_device", "_direction", "_event", "_result", "_error")

    def __init__(self, device, direction):
        if direction not in ("host_to_device", "device_to_host"):
            raise ValueError(f"unknown copy direction: {direction!r}")
        self._device = device
        self._direction = direction
        self._event = threading.Event()
        self._result = None
        self._error = None

    @property
    def device(self):
        """The involved core.Device (copy target for prefetch, source for sink)."""
        return self._device

    @property
    def direction(self):
        """Exactly "host_to_device" (prefetch) or "device_to_host" (sink)."""
        return self._direction

    @property
    def done(self):
        """Non-blocking: True once the worker finished (success or failure)."""
        return self._event.is_set()

    def wait(self, timeout=None):
        """Block until the copy finishes; see the class docstring."""
        if not self._event.is_set() and threading.current_thread() is _worker_thread:
            raise RuntimeError(
                "AsyncCopy.wait() would deadlock on the etl.io worker "
                "thread: this copy is still pending and only the worker can "
                "complete it — wait from the submitting thread"
            )
        if not self._event.wait(timeout):
            raise TimeoutError(
                f"timed out after {timeout}s waiting for the {self.direction} "
                f"copy involving {self.device!r} — the copy is still pending "
                "(wait timeouts never cancel copies; wait again to retry)"
            )
        if self._error is not None:
            raise self._error
        return None

    def result(self, timeout=None):
        """``wait()`` + the copy's result tensor (see the class docstring)."""
        self.wait(timeout)
        return self._result

    # --- internal completion (worker side only) -----------------------------
    def _succeed(self, result):
        self._result = result
        self._event.set()

    def _fail(self, error):
        self._error = error
        self._event.set()


def _bootstrap_worker_context(device):
    """One-time, best-effort CUDA-context bootstrap for the worker thread.

    A fresh thread's first allocating HAL op on a CUDA device created by
    another thread fails with ``CUDA_ERROR_INVALID_CONTEXT`` (iree does not
    push its stored context per call). Creating a throwaway HAL device on
    the same GPU from the worker makes the shared primary context current
    there. Runs at most once per worker per ``(kind, index)``, only for
    non-``"cpu"`` kinds; every failure is swallowed — this must never mask
    the copy's own error, which surfaces at ``wait()``/``result()``.
    """
    key = (device.kind, device.index)
    if key in _bootstrapped:
        return
    _bootstrapped.add(key)
    try:
        import iree.runtime as rt  # lazy: only non-cpu-kind copies reach here

        # The iree adapter's exact 1-based HAL device-id mapping
        # (etl/backends/adapters/iree.py, _acquire_runtime_device).
        dev = rt.get_driver(device.kind).create_device(device_id=device.index + 1)
        _bootstrap_devices.append(dev)  # keep alive: the context stays current
    except Exception:
        pass  # best-effort only — the copy's own error still surfaces


def _worker_main():
    """Execute queued copies strictly in issue order; never dies, never raises."""
    while True:
        job = _jobs.get()
        if job is _SENTINEL:
            return
        handle, thunk = job
        try:
            if handle.device.kind != "cpu":
                _bootstrap_worker_context(handle.device)
            result = thunk()
        except Exception as error:  # stored on the handle; surfaces at wait()
            handle._fail(error)
        else:
            handle._succeed(result)
        finally:
            with _state_lock:
                try:
                    _outstanding.remove(handle)
                except ValueError:
                    pass


def _ensure_worker():
    """Start the process-global worker on first submission (lock-guarded)."""
    global _worker_thread
    with _state_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            thread = threading.Thread(
                target=_worker_main, name="etl-io-worker", daemon=True
            )
            _worker_thread = thread
            thread.start()
    return _worker_thread


def _submit(handle, thunk):
    """Enqueue a copy job behind all previously submitted ones.

    The outstanding-set append and the queue put happen in ONE critical
    section so the queue's execution order always equals the issue order
    recorded in ``_outstanding`` (``wait_all`` snapshots and FIFO completion
    stay consistent even under concurrent submission from several threads).
    """
    _ensure_worker()
    with _state_lock:
        _outstanding.append(handle)
        _jobs.put((handle, thunk))


def wait_all(timeout=None):
    """Wait for every copy outstanding at call time (issue order) to finish.

    Snapshots the outstanding set at call time — copies submitted during a
    ``wait_all`` are NOT part of its snapshot. Returns ``None`` once all
    complete; raises builtin ``TimeoutError`` when ``timeout`` seconds pass
    first (handles stay pending — call ``wait_all`` again); re-raises the
    FIRST stored copy failure encountered in issue order (never swallowed).
    Convenience for tests and shutdown.
    """
    with _state_lock:
        snapshot = list(_outstanding)
    if snapshot and threading.current_thread() is _worker_thread:
        raise RuntimeError(
            "wait_all() cannot run on the etl.io worker thread — it would "
            "block on copies only the worker can complete; call it from the "
            "submitting thread"
        )
    deadline = None if timeout is None else time.monotonic() + timeout
    for handle in snapshot:
        if handle.done:
            # Completed (success or failure) — surface any stored failure now.
            handle.wait(None)
            continue
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(
                f"wait_all() timed out after {timeout}s with copies still "
                "pending — timeouts never cancel copies; call wait_all() again"
            )
        try:
            handle.wait(remaining)
        except TimeoutError:
            if handle.done:
                raise  # a stored failure that happens to be a TimeoutError
            raise TimeoutError(
                f"wait_all() timed out after {timeout}s waiting for the "
                f"{handle.direction} copy involving {handle.device!r} — the "
                "copy is still pending (timeouts never cancel copies; "
                "wait_all() again to retry)"
            ) from None
    return None


def _shutdown_worker():
    """atexit hook: drain in-flight copies, then stop the worker cleanly."""
    global _worker_thread
    thread = _worker_thread
    if thread is None or not thread.is_alive():
        return
    try:
        _jobs.put(_SENTINEL)
        thread.join()
    except Exception:
        pass  # never break interpreter exit
    _worker_thread = None


atexit.register(_shutdown_worker)
