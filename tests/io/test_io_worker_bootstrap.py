"""Device-free plumbing pins for the ``etl.io`` worker's non-cpu bootstrap.

``etl/io.py::_bootstrap_worker_context(device)`` is the worker thread's
one-time, best-effort CUDA/ROCm primary-context bootstrap: for a copy whose
handle device is not ``Device("cpu", ...)`` it lazily imports
``iree.runtime``, maps the etl device kind to an iree HAL driver name
(``"rocm"``/``"hip"`` -> ``"hip"``, everything else verbatim), creates a
throwaway HAL device with iree's 1-based ``device_id = index + 1`` mapping and
keeps it alive in ``io._bootstrap_devices`` — swallowing every failure so the
copy's own error still surfaces at ``wait()``/``result()``. The dedup set
``io._bootstrapped`` holds ``(kind, index)`` keys and is appended BEFORE the
import, so a bootstrap runs at most once per worker per key even when it
failed.

**Device-free setup.** No GPU and no real iree ever run here. ``import
iree.runtime as rt`` is served from a fake ``iree`` PACKAGE **and** a fake
``iree.runtime`` submodule installed via ``monkeypatch.setitem(sys.modules,
...)`` — installing only ``"iree.runtime"`` is not enough, because the import
statement still imports the real top-level ``iree`` package (which IS
installed in this env). ``monkeypatch`` removes/restores both entries after
each test, so nothing leaks into the rest of the run (in particular
``test_io.py::TestModuleSurface::test_import_pulls_no_iree``, which runs
before this file, stays valid). The fakes carry a real ``__spec__`` so an
incidental ``importlib.util.find_spec("iree")`` during the window does not
raise.

**Driving a real worker copy.** ``io.prefetch``/``io.sink`` cannot be used as
the driver: ``etl.backends`` registers REAL lazy device-transfer providers for
kinds ``"cuda"``/``"rocm"``/``"hip"`` (plus per-backend slots once the iree
adapter activates), so a prefetch would attempt a real GPU upload. Instead the
worker's own copy-execution path is driven directly with a harmless no-op
thunk — ``io.AsyncCopy(device, direction)`` + ``io._submit(handle, lambda:
None)`` — which still runs ``_worker_main``'s bootstrap gate before the thunk,
and ``handle.wait()`` still drains the process-global worker. ``Device``
accepts any non-empty kind string, so ``Device("hip", 0)`` is constructible.

Run:  python3 -m pytest tests/io/test_io_worker_bootstrap.py -q
"""
from __future__ import annotations

import importlib.machinery
import sys
import time
import types

import pytest

import etl  # noqa: F401  (import-order A: parent package first)
import etl.io as io

from etl.core import Device


# ---------------------------------------------------------------------------
# Fake iree.runtime + recorders
# ---------------------------------------------------------------------------
class _FakeDevice:
    """The throwaway HAL device ``create_device`` hands back."""

    def __init__(self, driver_name, device_id):
        self.driver_name = driver_name
        self.device_id = device_id


class _FakeDriver:
    """Fake HAL driver: records ``create_device`` calls (name, device_id)."""

    def __init__(self, name, runtime):
        self.name = name
        self._runtime = runtime

    def create_device(self, device_id=None, **kwargs):
        if self._runtime.raise_on_create_device is not None:
            raise self._runtime.raise_on_create_device
        self._runtime.create_device_calls.append((self.name, device_id))
        return _FakeDevice(self.name, device_id)


class _FakeRuntime:
    """Recorder standing in for the ``iree.runtime`` module.

    ``get_driver(name)`` records the requested HAL driver name and returns a
    per-name fake driver (cached, so repeated lookups share one driver whose
    ``create_device`` calls accumulate). ``raise_on_get_driver`` /
    ``raise_on_create_device`` inject the two failure modes the best-effort
    bootstrap must swallow.
    """

    def __init__(self):
        self.driver_calls = []
        self.create_device_calls = []
        self.drivers = {}
        self.raise_on_get_driver = None
        self.raise_on_create_device = None

    def get_driver(self, name):
        self.driver_calls.append(name)
        if self.raise_on_get_driver is not None:
            raise self.raise_on_get_driver
        return self.drivers.setdefault(name, _FakeDriver(name, self))

    @property
    def device_ids(self):
        return [device_id for _, device_id in self.create_device_calls]


def _install_fake_iree(monkeypatch, runtime):
    """Install fake ``iree`` + ``iree.runtime`` modules into ``sys.modules``.

    BOTH are required: with only ``"iree.runtime"`` present, ``import
    iree.runtime as rt`` still imports the real top-level ``iree`` package.
    """
    fake_rt = types.ModuleType("iree.runtime")
    fake_rt.get_driver = runtime.get_driver
    fake_pkg = types.ModuleType("iree")
    fake_pkg.runtime = fake_rt
    for module, name in ((fake_pkg, "iree"), (fake_rt, "iree.runtime")):
        # A real spec keeps an incidental find_spec("iree") from raising
        # ValueError('iree.__spec__ is None') while the fakes are installed.
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    monkeypatch.setitem(sys.modules, "iree", fake_pkg)
    monkeypatch.setitem(sys.modules, "iree.runtime", fake_rt)
    return fake_rt


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
def _wait_until(predicate, timeout, message):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.002)


@pytest.fixture
def fake_iree(monkeypatch):
    """Fake ``iree.runtime`` + per-test isolation of the worker bootstrap state.

    ``_bootstrapped`` / ``_bootstrap_devices`` are process globals: an earlier
    suite in a full run may already hold ``(kind, index)`` keys and real
    throwaway devices, so both are swapped for fresh containers for the test
    and restored by ``monkeypatch`` afterwards.
    """
    runtime = _FakeRuntime()
    _install_fake_iree(monkeypatch, runtime)
    monkeypatch.setattr(io, "_bootstrapped", set())
    monkeypatch.setattr(io, "_bootstrap_devices", [])
    return runtime


@pytest.fixture(autouse=True)
def _drain_outstanding():
    """Leave the process-global worker drained (no bleed-over into other tests)."""
    yield
    _wait_until(
        lambda: not io._outstanding, 10.0,
        f"etl.io worker never drained its outstanding set: {io._outstanding!r}",
    )


def _drive_worker_copy(device, direction="host_to_device"):
    """Run ONE real worker copy with a harmless no-op thunk.

    Exercises ``_worker_main``'s non-cpu bootstrap gate via the real
    process-global worker without ever touching a device or a provider (see
    the module docstring for why ``prefetch``/``sink`` cannot be used here).
    """
    handle = io.AsyncCopy(device, direction)
    io._submit(handle, lambda: None)
    return handle


# ---------------------------------------------------------------------------
# 1. Kind -> iree HAL driver name + 1-based device-id mapping
# ---------------------------------------------------------------------------
class TestDriverNameAndDeviceIdMapping:
    """``get_driver`` gets the mapped driver name; ``create_device`` the
    1-based HAL device id (the iree adapter's exact mapping)."""

    @pytest.mark.parametrize(
        ("kind", "expected_driver"),
        [("rocm", "hip"), ("hip", "hip"), ("cuda", "cuda")],
    )
    @pytest.mark.parametrize("index", [0, 2])
    def test_kind_maps_to_driver_and_index_to_1_based_device_id(
        self, fake_iree, kind, index, expected_driver
    ):
        handle = _drive_worker_copy(Device(kind, index))
        assert handle.wait() is None

        assert fake_iree.driver_calls == [expected_driver]
        assert fake_iree.create_device_calls == [(expected_driver, index + 1)]
        # The throwaway device is kept alive process-globally.
        assert [(d.driver_name, d.device_id) for d in io._bootstrap_devices] == [
            (expected_driver, index + 1)
        ]


# ---------------------------------------------------------------------------
# 2. Laziness: cpu-kind copies never reach iree
# ---------------------------------------------------------------------------
class TestCpuKindCopiesSkipTheBootstrap:
    """``_worker_main``'s ``handle.device.kind != "cpu"`` gate keeps the
    bootstrap (and hence any iree import) off for cpu-kind copies."""

    def test_cpu_kind_copy_never_asks_for_a_driver(self, fake_iree):
        # The only reachable cpu-kind handle: prefetch rejects a cpu target
        # and sink's handle device is always the non-cpu source device, so the
        # gate is pinned by driving the copy path directly.
        handle = _drive_worker_copy(Device("cpu", 0), direction="device_to_host")
        assert handle.wait() is None

        assert fake_iree.driver_calls == []
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []
        assert io._bootstrapped == set()


# ---------------------------------------------------------------------------
# 3. At most one bootstrap per worker per (kind, index)
# ---------------------------------------------------------------------------
class TestBootstrapOncePerKindAndIndex:
    """``io._bootstrapped`` dedups on the literal ``(kind, index)`` key."""

    def test_repeated_copies_for_the_same_key_bootstrap_once(self, fake_iree):
        first = _drive_worker_copy(Device("rocm", 0))
        second = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip"]
        assert fake_iree.device_ids == [1]
        assert len(io._bootstrap_devices) == 1
        assert io._bootstrapped == {("rocm", 0)}

    def test_a_different_index_of_the_same_kind_bootstraps_again(self, fake_iree):
        first = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None
        second = _drive_worker_copy(Device("rocm", 2))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip", "hip"]
        assert fake_iree.device_ids == [1, 3]
        assert len(io._bootstrap_devices) == 2
        assert io._bootstrapped == {("rocm", 0), ("rocm", 2)}

    def test_hip_and_rocm_are_distinct_keys_despite_one_driver(self, fake_iree):
        # The dedup key is the literal kind string, not the resolved driver:
        # "hip" and "rocm" both resolve to the "hip" driver but bootstrap
        # separately (each context is created once per key).
        first = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None
        second = _drive_worker_copy(Device("hip", 0))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip", "hip"]
        assert fake_iree.device_ids == [1, 1]
        assert io._bootstrapped == {("rocm", 0), ("hip", 0)}

    def test_a_different_kind_bootstraps_again(self, fake_iree):
        first = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None
        second = _drive_worker_copy(Device("cuda", 1))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip", "cuda"]
        assert fake_iree.device_ids == [1, 2]


# ---------------------------------------------------------------------------
# 4. Best-effort: a failing bootstrap never alters the copy's outcome
# ---------------------------------------------------------------------------
class TestBootstrapFailuresAreSwallowed:
    """Every bootstrap failure is swallowed — the copy still succeeds and no
    throwaway device is retained."""

    def test_failing_driver_lookup_never_fails_the_copy(self, fake_iree):
        fake_iree.raise_on_get_driver = RuntimeError("no hip driver")
        handle = _drive_worker_copy(Device("rocm", 0))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["hip"]
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []

    def test_failing_device_creation_never_fails_the_copy(self, fake_iree):
        fake_iree.raise_on_create_device = RuntimeError("no ROCm runtime")
        handle = _drive_worker_copy(Device("cuda", 0))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["cuda"]
        assert io._bootstrap_devices == []

    def test_a_failed_bootstrap_is_not_retried_for_the_same_key(self, fake_iree):
        # _bootstrapped.add(key) happens BEFORE the import/try, so a failed
        # bootstrap is not re-attempted for a later copy on the same key.
        fake_iree.raise_on_get_driver = RuntimeError("no hip driver")
        first = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None
        second = _drive_worker_copy(Device("rocm", 0))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip"]
        assert io._bootstrapped == {("rocm", 0)}
        assert io._bootstrap_devices == []

    def test_a_failed_bootstrap_does_not_block_another_key(self, fake_iree):
        fake_iree.raise_on_get_driver = RuntimeError("no hip driver")
        first = _drive_worker_copy(Device("rocm", 0))
        assert first.wait() is None

        fake_iree.raise_on_get_driver = None
        second = _drive_worker_copy(Device("cuda", 0))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["hip", "cuda"]
        assert fake_iree.device_ids == [1]
        assert len(io._bootstrap_devices) == 1
