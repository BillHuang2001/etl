"""Device-free plumbing pins for the ``etl.io`` worker's non-cpu bootstrap.

``etl/io.py::_bootstrap_worker_context(device)`` is the worker thread's
one-time, best-effort CUDA/ROCm/Vulkan primary-context bootstrap: for a copy
whose handle device is not ``Device("cpu", ...)`` it lazily imports
``iree.runtime``, maps the etl device kind to an iree HAL driver name
(``"rocm"``/``"hip"`` -> ``"hip"``, everything else verbatim, so
``"vulkan"`` keeps ``"vulkan"``), creates a throwaway HAL device and keeps it
alive in ``io._bootstrap_devices`` — swallowing every failure so the copy's
own error still surfaces at ``wait()``/``result()``. The device ADDRESSING
follows the iree adapter exactly: cuda/rocm use the 1-based ORDINAL
``device_id = index + 1`` (the ``create_device`` keyword), while vulkan maps
through the per-process query — ``query_available_devices()[index]``, whose
info dict is passed POSITIONALLY to ``create_device`` — because Vulkan HAL ids
are UUID HASHES, not ordinals; an out-of-range (or negative) index is a silent
no-op there, since this is a best-effort context warm-up and never a
validation point. The dedup set ``io._bootstrapped`` holds ``(kind, index)``
keys and is appended BEFORE the import, so a bootstrap runs at most once per
worker per key even when it failed.

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
kinds ``"cuda"``/``"rocm"``/``"hip"``/``"vulkan"`` (plus per-backend slots once
the iree adapter activates), so a prefetch would attempt a real GPU upload.
Instead the worker's own copy-execution path is driven directly with a
harmless no-op thunk — ``io.AsyncCopy(device, direction)`` +
``io._submit(handle, lambda: None)`` — which still runs ``_worker_main``'s
bootstrap gate before the thunk,
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

from etl.core import Device, DeviceError


# ---------------------------------------------------------------------------
# Fake iree.runtime + recorders
# ---------------------------------------------------------------------------
class _FakeDevice:
    """The throwaway HAL device ``create_device`` hands back.

    ``device_id`` is whatever the bootstrap passed: the 1-based ORDINAL int
    for cuda/rocm, the QUERIED info DICT for vulkan.
    """

    def __init__(self, driver_name, device_id):
        self.driver_name = driver_name
        self.device_id = device_id


class _FakeDriver:
    """Fake HAL driver: records the query and every ``create_device`` call.

    ``create_device`` accepts BOTH shapes the real HAL driver takes — the
    cuda/rocm ordinal as the ``device_id=`` KEYWORD and the Vulkan info dict
    POSITIONALLY. Both are recorded twice: the full call
    (``runtime.create_device_signatures``: ``(driver_name, args, kwargs)``)
    and the normalized id (``runtime.create_device_calls``:
    ``(driver_name, device_id)``), so a test can pin which shape was used.
    """

    def __init__(self, name, runtime):
        self.name = name
        self._runtime = runtime

    def query_available_devices(self):
        self._runtime.query_calls.append(self.name)
        if self._runtime.raise_on_query_available_devices is not None:
            raise self._runtime.raise_on_query_available_devices
        return list(self._runtime.available_device_infos)

    def create_device(self, *args, **kwargs):
        if self._runtime.raise_on_create_device is not None:
            raise self._runtime.raise_on_create_device
        self._runtime.create_device_signatures.append((self.name, args, kwargs))
        device_id = args[0] if args else kwargs.get("device_id")
        self._runtime.create_device_calls.append((self.name, device_id))
        return _FakeDevice(self.name, device_id)


class _FakeRuntime:
    """Recorder standing in for the ``iree.runtime`` module.

    ``get_driver(name)`` records the requested HAL driver name and returns a
    per-name fake driver (cached, so repeated lookups share one driver whose
    ``create_device`` calls accumulate). ``raise_on_get_driver`` /
    ``raise_on_query_available_devices`` / ``raise_on_create_device`` inject
    the three failure modes the best-effort bootstrap must swallow.
    ``available_device_infos`` is what ``query_available_devices()`` reports
    (only the vulkan branch ever queries).
    """

    def __init__(self):
        self.driver_calls = []
        self.query_calls = []
        self.create_device_calls = []
        self.create_device_signatures = []
        self.drivers = {}
        self.available_device_infos = []
        self.raise_on_get_driver = None
        self.raise_on_query_available_devices = None
        self.raise_on_create_device = None

    def get_driver(self, name):
        self.driver_calls.append(name)
        if self.raise_on_get_driver is not None:
            raise self.raise_on_get_driver
        return self.drivers.setdefault(name, _FakeDriver(name, self))

    @property
    def device_ids(self):
        return [device_id for _, device_id in self.create_device_calls]


def _vulkan_infos():
    """Two fake ``query_available_devices()`` info dicts.

    The ``device_id`` values are large UUID-HASH-like integers, deliberately
    NOT the ordinal ``index + 1`` the cuda/rocm path uses — that is exactly
    what the vulkan branch must hand to ``create_device``.
    """
    return (
        {
            "device_id": 857002912,
            "path": "VK_UUID_4b8d8f3a2c1e4f0d9a7b6c5d4e3f2a1b",
            "name": "lavapipe (LLVM 20.1.2, 256 bits)",
        },
        {
            "device_id": 1854200391,
            "path": "VK_UUID_9f1c2d3e4a5b6c7d8e9f0a1b2c3d4e5f",
            "name": "Vulkan Software Rasterizer (Mesa)",
        },
    )


class _DuckDevice:
    """A bare ``kind``/``index`` stand-in.

    ``core.Device`` rejects a negative index at construction, so the negative
    case of the bootstrap's bounds guard is only reachable with a stand-in
    (``_bootstrap_worker_context`` reads just ``.kind``/``.index``).
    """

    __slots__ = ("kind", "index")

    def __init__(self, kind, index):
        self.kind = kind
        self.index = index

    def __repr__(self):
        return f"_DuckDevice({self.kind!r}, {self.index})"


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


def _drive_worker_copy(device, direction="host_to_device", thunk=None):
    """Run ONE real worker copy with a harmless no-op thunk.

    Exercises ``_worker_main``'s non-cpu bootstrap gate via the real
    process-global worker without ever touching a device or a provider (see
    the module docstring for why ``prefetch``/``sink`` cannot be used here).
    ``thunk`` overrides the copy body (used to pin that a failed bootstrap
    never masks the copy's own error).
    """
    handle = io.AsyncCopy(device, direction)
    io._submit(handle, (lambda: None) if thunk is None else thunk)
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


# ---------------------------------------------------------------------------
# 5. Vulkan: the QUERIED per-process info dict, never an ordinal device id
# ---------------------------------------------------------------------------
class TestVulkanQueriedInfoDictMapping:
    """``vulkan`` keeps its own HAL driver name and addresses a device through
    ``query_available_devices()``: the info dict at the ``Device.index``
    position is what ``create_device`` receives (positionally) — Vulkan HAL ids
    are per-process UUID hashes, NOT the ordinal ``index + 1``."""

    def test_the_vulkan_driver_is_queried_and_its_info_dict_created(
        self, fake_iree
    ):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        handle = _drive_worker_copy(Device("vulkan", 0))
        assert handle.wait() is None

        # The mapped driver is "vulkan" itself — NOT "hip" (the rocm/hip
        # branch) and NOT "cuda".
        assert fake_iree.driver_calls == ["vulkan"]
        # The device is DISCOVERED, not assumed: the query runs first.
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_signatures == [("vulkan", (infos[0],), {})]
        assert fake_iree.create_device_calls == [("vulkan", infos[0])]

    def test_no_ordinal_device_id_kwarg_is_used_for_vulkan(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None

        (name, args, kwargs) = fake_iree.create_device_signatures[0]
        assert name == "vulkan"
        # Positional info dict, NO keyword at all — the cuda/rocm contract
        # (``create_device(device_id=index + 1)``) must not leak in here.
        assert kwargs == {}
        assert "device_id" not in kwargs
        assert args == (infos[0],)
        # The hash id INSIDE the info dict identifies the device; the id the
        # bootstrap passed is that dict, never an ordinal int.
        assert args[0]["device_id"] == 857002912
        assert args[0]["name"] == "lavapipe (LLVM 20.1.2, 256 bits)"
        assert fake_iree.device_ids == [infos[0]]
        assert not isinstance(fake_iree.device_ids[0], int)

    def test_cuda_keeps_the_ordinal_kwarg_while_vulkan_uses_the_info_dict(
        self, fake_iree
    ):
        # Same run, two kinds: the call SHAPES differ, which is the whole
        # point of the vulkan branch.
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("cuda", 0)).wait() is None
        assert _drive_worker_copy(Device("vulkan", 1)).wait() is None

        assert fake_iree.driver_calls == ["cuda", "vulkan"]
        # Only the vulkan branch queries; cuda never does.
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_signatures == [
            ("cuda", (), {"device_id": 1}),  # 1-based ORDINAL, keyword
            ("vulkan", (infos[1],), {}),  # queried info dict, positional
        ]

    @pytest.mark.parametrize("index", [0, 1])
    def test_the_queried_info_dict_of_that_index_is_used(self, fake_iree, index):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("vulkan", index)).wait() is None

        assert fake_iree.device_ids == [infos[index]]
        # The throwaway device wraps that exact info dict...
        assert [d.device_id for d in io._bootstrap_devices] == [infos[index]]
        assert [d.driver_name for d in io._bootstrap_devices] == ["vulkan"]
        assert io._bootstrapped == {("vulkan", index)}

    def test_the_second_index_is_a_different_device_not_an_ordinal(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None
        assert _drive_worker_copy(Device("vulkan", 1)).wait() is None

        assert fake_iree.device_ids == [infos[0], infos[1]]
        assert infos[0]["device_id"] != infos[1]["device_id"]
        # ... and neither id equals the ordinal the cuda/rocm path would use.
        assert fake_iree.device_ids != [1, 2]


# ---------------------------------------------------------------------------
# 6. Vulkan: an unresolvable index is a silent no-op (never a validation point)
# ---------------------------------------------------------------------------
class TestVulkanOutOfRangeIndexSkipsTheBootstrap:
    """The bounds guard turns an out-of-range index (or an empty enumeration)
    into a silent skip: the copy still succeeds, no device is created and no
    exception escapes — this is a best-effort context warm-up."""

    @pytest.mark.parametrize(
        ("infos_count", "index"),
        [(1, 5), (2, 2), (2, 9), (0, 0), (0, 3)],
    )
    def test_out_of_range_or_missing_device_creates_nothing(
        self, fake_iree, infos_count, index
    ):
        fake_iree.available_device_infos = list(_vulkan_infos())[:infos_count]

        handle = _drive_worker_copy(Device("vulkan", index))
        assert handle.wait() is None
        assert handle.done is True

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_signatures == []
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []
        # The key is still recorded (added BEFORE the try), so a later copy on
        # the same key re-queries nothing either.
        assert io._bootstrapped == {("vulkan", index)}

    def test_a_negative_index_skips_the_bootstrap_too(self, fake_iree):
        fake_iree.available_device_infos = list(_vulkan_infos())

        # The value model rejects negative indices up front ...
        with pytest.raises(ValueError, match="non-negative"):
            Device("vulkan", -1)
        # ... so the guard's negative branch is driven with a bare stand-in.
        handle = _drive_worker_copy(_DuckDevice("vulkan", -1))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []
        assert io._bootstrapped == {("vulkan", -1)}

    def test_an_out_of_range_copy_bootstraps_once_for_the_same_key(self, fake_iree):
        fake_iree.available_device_infos = list(_vulkan_infos())[:1]

        assert _drive_worker_copy(Device("vulkan", 5)).wait() is None
        assert _drive_worker_copy(Device("vulkan", 5)).wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_calls == []


# ---------------------------------------------------------------------------
# 7. Vulkan: at most one bootstrap per (kind, index)
# ---------------------------------------------------------------------------
class TestVulkanBootstrapOncePerKindAndIndex:
    """``io._bootstrapped`` dedups vulkan devices on the same literal
    ``(kind, index)`` key as every other kind."""

    def test_repeated_copies_for_the_same_vulkan_key_bootstrap_once(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        first = _drive_worker_copy(Device("vulkan", 0))
        assert first.wait() is None
        second = _drive_worker_copy(Device("vulkan", 0))
        assert second.wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.device_ids == [infos[0]]
        assert len(io._bootstrap_devices) == 1
        assert io._bootstrapped == {("vulkan", 0)}

    def test_a_different_vulkan_index_bootstraps_again(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None
        assert _drive_worker_copy(Device("vulkan", 1)).wait() is None

        assert fake_iree.driver_calls == ["vulkan", "vulkan"]
        assert fake_iree.query_calls == ["vulkan", "vulkan"]
        assert fake_iree.device_ids == [infos[0], infos[1]]
        assert [d.device_id for d in io._bootstrap_devices] == [infos[0], infos[1]]
        assert io._bootstrapped == {("vulkan", 0), ("vulkan", 1)}

    def test_the_vulkan_key_is_distinct_from_other_kinds(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None
        assert _drive_worker_copy(Device("cuda", 0)).wait() is None

        assert fake_iree.driver_calls == ["vulkan", "cuda"]
        assert fake_iree.device_ids == [infos[0], 1]
        assert io._bootstrapped == {("vulkan", 0), ("cuda", 0)}


# ---------------------------------------------------------------------------
# 8. Vulkan: a failing bootstrap is swallowed, the copy's own error is not
# ---------------------------------------------------------------------------
class TestVulkanBootstrapFailuresAreSwallowed:
    """Every vulkan bootstrap failure (driver, query, device creation) is
    swallowed — the copy succeeds and no throwaway device is retained; a copy
    that DOES fail still raises its own error at ``wait()``."""

    def test_a_failing_vulkan_driver_lookup_never_fails_the_copy(self, fake_iree):
        fake_iree.raise_on_get_driver = RuntimeError("no Vulkan HAL driver")
        handle = _drive_worker_copy(Device("vulkan", 0))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == []
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []

    def test_a_failing_vulkan_query_never_fails_the_copy(self, fake_iree):
        fake_iree.available_device_infos = list(_vulkan_infos())
        fake_iree.raise_on_query_available_devices = RuntimeError(
            "vkEnumeratePhysicalDevices failed"
        )
        handle = _drive_worker_copy(Device("vulkan", 0))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []
        assert io._bootstrapped == {("vulkan", 0)}

    def test_a_failing_vulkan_device_creation_never_fails_the_copy(self, fake_iree):
        fake_iree.available_device_infos = list(_vulkan_infos())
        fake_iree.raise_on_create_device = RuntimeError("no Vulkan runtime")
        handle = _drive_worker_copy(Device("vulkan", 0))
        assert handle.wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert fake_iree.create_device_calls == []
        assert io._bootstrap_devices == []

    def test_a_failed_vulkan_bootstrap_does_not_mask_the_copy_error(self, fake_iree):
        fake_iree.available_device_infos = list(_vulkan_infos())
        fake_iree.raise_on_create_device = RuntimeError("no Vulkan runtime")
        copy_error = DeviceError("vulkan upload failed at the provider")

        def thunk():
            raise copy_error

        handle = _drive_worker_copy(Device("vulkan", 0), thunk=thunk)
        with pytest.raises(DeviceError, match="provider") as excinfo:
            handle.wait()
        # The copy's own exception surfaces VERBATIM (same object).
        assert excinfo.value is copy_error
        assert fake_iree.query_calls == ["vulkan"]
        assert io._bootstrap_devices == []

    def test_a_failed_vulkan_bootstrap_is_not_retried_for_the_same_key(
        self, fake_iree
    ):
        fake_iree.available_device_infos = list(_vulkan_infos())
        fake_iree.raise_on_create_device = RuntimeError("no Vulkan runtime")

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None
        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None

        assert fake_iree.driver_calls == ["vulkan"]
        assert fake_iree.query_calls == ["vulkan"]
        assert io._bootstrapped == {("vulkan", 0)}
        assert io._bootstrap_devices == []

    def test_a_failed_vulkan_bootstrap_does_not_block_another_key(self, fake_iree):
        infos = _vulkan_infos()
        fake_iree.available_device_infos = list(infos)
        fake_iree.raise_on_create_device = RuntimeError("no Vulkan runtime")

        assert _drive_worker_copy(Device("vulkan", 0)).wait() is None

        fake_iree.raise_on_create_device = None
        assert _drive_worker_copy(Device("vulkan", 1)).wait() is None

        assert fake_iree.query_calls == ["vulkan", "vulkan"]
        assert fake_iree.device_ids == [infos[1]]
        assert [d.device_id for d in io._bootstrap_devices] == [infos[1]]
        assert io._bootstrapped == {("vulkan", 0), ("vulkan", 1)}