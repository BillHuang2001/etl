"""Vulkan HAL device ACQUISITION, STAGING TIER and TRANSFER PROVIDER (unit).

``"vulkan"`` is a first-class etl device kind served by the iree adapter, and
its HAL device-discovery rules differ from cuda/rocm in three ways that this
file pins WITHOUT a device, a Vulkan ICD or a compiler:

1. **Acquisition** (``iree._acquire_vulkan_device``): IREE Vulkan
   ``device_id`` values are QUERIED PER-PROCESS HASHES, not the cuda/rocm
   ``index + 1`` ordinals — ``Device("vulkan", N)`` maps to the N-th
   ENUMERATED device-info dict, handed to ``driver.create_device(info_dict)``
   positionally (never ``device_id=``). Acquisition is cached per
   ``(index, runtime args)`` in the module THREAD-LOCAL HAL device cache, so
   ``load()`` and the placement provider share ONE HAL device on a thread.
   Zero enumerated devices / an out-of-range index / a failing
   ``create_device`` each raise an actionable ``core.BackendError`` (an
   already-``ETLError`` propagates unwrapped, never double-wrapped).
2. **Staging tier** (``iree._staged_input``): the CUDA pinned/DMA and ROCm
   host-local tiers are CUDA/ROCm ONLY — ``"vulkan"`` has NO staging tier and
   returns ``None`` immediately, so a vulkan upload IS the plain
   ``rt.asdevicearray`` copy.
3. **Transfer provider** (``iree.register()``): the adapter registers
   ``upload_tensor`` under the PER-BACKEND slot ``("vulkan", "iree")`` AND —
   the one documented exception to the per-backend-only rule — in the FLAT
   default slot for ``"vulkan"``, because ``etl.backends`` installs lazy
   thunks for cuda/rocm/hip only and no other backend serves vulkan. The flat
   cuda/rocm/hip slots therefore keep their ``etl.backends`` thunks.

Everything here is device-free and compiler-free: the whole ``iree.runtime``
surface these paths touch (``get_driver`` + a recording fake vulkan driver,
``asdevicearray``, ``flags.parse_flags``, ``array_interop``) is a
monkeypatched fake module installed with the same save/restore discipline as
``tests/backends/test_options_override.py::_install_fake_iree_runtime``, so
the file runs identically with and without a real IREE install. Every
monkeypatched module, provider registry and thread-local cache is restored
on teardown so nothing leaks into another suite.

Pinned against ``etl/backends/adapters/iree.py`` (``_acquire_vulkan_device`` /
``_acquire_runtime_device`` / ``_cached_hal_device`` / ``_staged_input`` /
``upload_tensor`` / ``register``) and ``etl/core/tensor.py`` (the dual
device-transfer provider registry + ``Tensor.to`` resolution order).
"""
import sys
import threading
import types

import numpy as np
import pytest

from etl import core
from etl.backends import registry as _backends_registry
from etl.backends.adapters import iree as iree_adapt
from etl.core.tensor import (  # internal cross-module contract (Tensor.to routing)
    _BACKEND_DEVICE_TRANSFER_PROVIDERS as _PER_BACKEND_PROVIDERS,
    _DEVICE_TRANSFER_PROVIDERS as _FLAT_PROVIDERS,
    _PREFERRED_TRANSFER_BACKENDS as _PREFERRED_BACKENDS,
)

VULKAN0 = core.Device("vulkan", 0)
VULKAN1 = core.Device("vulkan", 1)

#: Enumerated vulkan device-info dicts exactly as IREE returns them: the
#: ``device_id`` entries are QUERIED HASHES (e.g. 857002912), never ordinals.
_LAVAPIPE = {
    "device_id": 857002912,
    "path": "8e0e6c2d-6a4e-4b0f-9a2f-1c3d5e7f9a11",
    "name": "lavapipe (LLVM 20.1.2, 256 bits)",
}
_RADV = {
    "device_id": 1184003912,
    "path": "0b7c1f2a-3d4e-4f50-8a6b-2c4d6e8f0a22",
    "name": "AMD Radeon RX 7900 XTX (RADV NAVI31)",
}
_INFOS = [_LAVAPIPE, _RADV]


# ---------------------------------------------------------------------------
# fakes: a recording vulkan HAL driver + a fake ``iree.runtime`` module
# ---------------------------------------------------------------------------


class _FakeHalDevice:
    """Bare HAL-device marker.

    Deliberately carries NO ``allocator`` / ``create_semaphore`` /
    ``queue_copy``: any attempt to run a staging tier against it raises
    ``AttributeError`` instead of silently succeeding, which is what makes
    the "vulkan has no staging tier" pins structural rather than stylistic.
    """

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return f"<_FakeVulkanHalDevice {self.index}>"


class _FakeDeviceArray:
    """Minimal duck device array: ``shape`` / ``dtype`` / ``to_host()``.

    No ``_buffer_view`` / ``_override_dtype``, so ``IreeDevicePayload`` takes
    its documented defensive fallback to the public accessors.
    """

    def __init__(self, host):
        self._host = np.array(host, copy=True)

    @property
    def shape(self):
        return tuple(self._host.shape)

    @property
    def dtype(self):
        return self._host.dtype

    def to_host(self):
        return np.array(self._host, copy=True)


class _FakeVulkanDriver:
    """Recording fake of IREE's vulkan HAL driver."""

    def __init__(self, infos, *, create_error=None):
        self.infos = [dict(info) for info in infos]
        self.create_error = create_error
        self.query_calls = 0
        #: one ``(args, kwargs)`` tuple per ``create_device`` call, in order.
        self.create_calls = []
        self.devices = []

    def query_available_devices(self):
        self.query_calls += 1
        return [dict(info) for info in self.infos]

    def create_device(self, *args, **kwargs):
        self.create_calls.append((args, kwargs))
        if self.create_error is not None:
            raise self.create_error
        device = _FakeHalDevice(len(self.devices))
        self.devices.append(device)
        return device


class _FakeRuntimeRecord:
    """Calls recorded on the fake ``iree.runtime`` module."""

    def __init__(self):
        self.get_driver = []
        self.asdevicearray = []
        self.parse_flags = []


def _install_fake_runtime(monkeypatch, drivers, *, record=None):
    """Install a fake ``iree.runtime`` module; return the call record.

    ``drivers`` maps a driver name to the object ``rt.get_driver(name)`` must
    return — an UNEXPECTED driver name fails the test loudly (a vulkan
    acquisition must ask for ``"vulkan"`` only). Same two-layer install as
    ``tests/backends/test_options_override.py::_install_fake_iree_runtime``:
    ``import iree.runtime as rt`` imports the parent ``iree`` package first
    and then resolves the ``runtime`` attribute, so BOTH layers are faked —
    a real ``iree`` already imported gets its ``runtime`` attribute patched
    (restored by monkeypatch), and with ``iree`` absent a minimal fake
    package module (``__path__ = []``) is installed. Submodules
    (``flags``, ``array_interop``) are installed too, because the adapter
    imports them lazily inside its functions.
    """
    record = _FakeRuntimeRecord() if record is None else record
    fake = types.ModuleType("iree.runtime")

    def _get_driver(name):
        record.get_driver.append(name)
        try:
            return drivers[name]
        except KeyError:  # pragma: no cover - a failing test, by design
            raise AssertionError(
                f"unexpected iree driver {name!r}: this test fakes only "
                f"{sorted(drivers)}"
            ) from None

    def _asdevicearray(device, array):
        record.asdevicearray.append((device, array))
        return _FakeDeviceArray(array)

    def _parse_flags(*args):
        record.parse_flags.append(args)

    fake.get_driver = _get_driver
    fake.asdevicearray = _asdevicearray

    flags = types.ModuleType("iree.runtime.flags")
    flags.parse_flags = _parse_flags
    fake.flags = flags

    array_interop = types.ModuleType("iree.runtime.array_interop")

    def _map_dtype_to_element_type(dtype):
        # Opaque non-None element type: enough for ``_staged_input``'s
        # kind gate, which is the only thing under test here.
        return ("element-type", np.dtype(dtype).str)

    array_interop.map_dtype_to_element_type = _map_dtype_to_element_type
    fake.array_interop = array_interop

    iree_pkg = sys.modules.get("iree")
    if iree_pkg is None:
        iree_pkg = types.ModuleType("iree")
        iree_pkg.__path__ = []  # a package, so ``import iree.runtime`` works
        monkeypatch.setitem(sys.modules, "iree", iree_pkg)
    monkeypatch.setattr(iree_pkg, "runtime", fake, raising=False)
    monkeypatch.setitem(sys.modules, "iree.runtime", fake)
    monkeypatch.setitem(sys.modules, "iree.runtime.flags", flags)
    monkeypatch.setitem(sys.modules, "iree.runtime.array_interop", array_interop)
    return record


class _Trap:
    """Any attribute access fails: a structural "this is never touched" pin."""

    def __init__(self, what):
        self._what = what

    def __getattr__(self, name):
        raise AssertionError(
            f"the vulkan path must not touch {self._what} (it accessed "
            f"{name!r})"
        )

    def __repr__(self):
        return f"<_Trap {self._what}>"


# ---------------------------------------------------------------------------
# fixtures: per-test HAL-device cache + provider/backend registry restoration
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_hal_device_cache():
    """Start every test with an EMPTY thread-local HAL device cache.

    The cache is a module-level ``threading.local()`` (``_HAL_DEVICE_CACHE``)
    whose ``devices`` dict holds the acquired HAL devices for the lifetime of
    the thread, so it is process state: an entry cached by another test would
    make a "create_device was called" pin order-dependent. The pre-test dict
    is restored afterwards, so this module leaves no cache behind either.
    """
    sentinel = object()
    saved = getattr(iree_adapt._HAL_DEVICE_CACHE, "devices", sentinel)
    iree_adapt._HAL_DEVICE_CACHE.devices = {}
    try:
        yield
    finally:
        if saved is sentinel:
            try:
                del iree_adapt._HAL_DEVICE_CACHE.devices
            except AttributeError:  # pragma: no cover - already gone
                pass
        else:
            iree_adapt._HAL_DEVICE_CACHE.devices = saved


@pytest.fixture(autouse=True)
def _restore_registries():
    """Snapshot/restore the provider registries and the backend registry.

    ``iree.register()`` writes the per-backend slots, the flat vulkan slot and
    the backend registry entry; restoring all of them keeps this module
    invisible to every other suite (run order insensitive in both the
    compiler-free and the compiler-installed environment).
    """
    flat = dict(_FLAT_PROVIDERS)
    per_backend = dict(_PER_BACKEND_PROVIDERS)
    preferred = dict(_PREFERRED_BACKENDS)
    backends = dict(_backends_registry._registry)
    try:
        yield
    finally:
        for live, saved in (
            (_FLAT_PROVIDERS, flat),
            (_PER_BACKEND_PROVIDERS, per_backend),
            (_PREFERRED_BACKENDS, preferred),
            (_backends_registry._registry, backends),
        ):
            live.clear()
            live.update(saved)


def _register_adapter(monkeypatch):
    """Call ``iree.register()`` with the availability probe bypassed.

    ``register()`` first calls ``IreeBackend.check_available`` (the
    ``pip install etl[iree]`` probe). The provider-registration contract is
    independent of the installed packages, so the probe is patched to a no-op
    at the CLASS level (monkeypatch restores the original classmethod
    descriptor) — that is what lets these pins run with iree absent.
    """
    monkeypatch.setattr(
        iree_adapt.IreeBackend, "check_available", classmethod(lambda cls: None)
    )
    return iree_adapt.register()


# ---------------------------------------------------------------------------
# 1. acquisition: index -> enumerated info dict (never an ordinal)
# ---------------------------------------------------------------------------


def test_index_zero_acquires_the_first_enumerated_info_dict(monkeypatch):
    """``Device("vulkan", 0)`` -> ``create_device(info_dict #0)`` positionally."""
    driver = _FakeVulkanDriver(_INFOS)
    record = _install_fake_runtime(monkeypatch, {"vulkan": driver})

    runtime_device = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])

    assert record.get_driver == ["vulkan"]
    assert record.asdevicearray == []
    assert driver.query_calls == 1
    assert len(driver.create_calls) == 1
    args, kwargs = driver.create_calls[0]
    assert args == (_LAVAPIPE,), "create_device must receive the info dict"
    assert kwargs == {}, "no device_id= kwarg: the info dict IS the selector"
    assert args[0]["device_id"] == 857002912
    assert runtime_device is driver.devices[0]


def test_index_one_acquires_the_second_info_dict_not_an_ordinal(monkeypatch):
    """``Device("vulkan", 1)`` is the 2nd enumerated device — NOT ``device_id=2``.

    The cuda/rocm 1-based ``device_id = index + 1`` mapping does not exist on
    vulkan: the enumerated info dicts carry QUERIED HASHES, so the requested
    index selects an ENTRY, and that entry's hash is what the HAL gets.
    """
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    iree_adapt._acquire_runtime_device("vulkan", VULKAN1, [])

    assert len(driver.create_calls) == 1
    args, kwargs = driver.create_calls[0]
    assert args[0] == _RADV
    assert "device_id" not in kwargs
    assert args[0]["device_id"] == 1184003912
    assert args[0]["device_id"] not in (1, 2), "hash ids are not ordinals"


def test_device_none_defaults_to_index_zero_and_shares_its_cache_entry(
    monkeypatch,
):
    """``device=None`` is index 0 — same HAL device as an explicit index 0."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    implicit = iree_adapt._acquire_runtime_device("vulkan", None, [])
    explicit = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])

    assert implicit is explicit
    assert len(driver.create_calls) == 1
    assert driver.create_calls[0][0] == (_LAVAPIPE,)


# ---------------------------------------------------------------------------
# 2. acquisition: the THREAD-LOCAL (kind, index, runtime args) cache
# ---------------------------------------------------------------------------


def test_same_index_and_args_reuse_one_thread_local_device(monkeypatch):
    """Two acquisitions for one ``(index, args)`` create ONE HAL device."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    first = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    second = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])

    assert first is second
    assert len(driver.create_calls) == 1
    assert driver.query_calls == 1, "a cache hit must not enumerate again"
    assert iree_adapt._HAL_DEVICE_CACHE.devices[("vulkan", 0, ())] is first


def test_equal_content_runtime_args_hit_the_same_cache_entry(monkeypatch):
    """The cache key is the ARGS TUPLE, so equal lists hit (no re-create)."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    first = iree_adapt._acquire_runtime_device(
        "vulkan", VULKAN0, ["--vulkan_use_immediate_data=true"]
    )
    second = iree_adapt._acquire_runtime_device(
        "vulkan", VULKAN0, ["--vulkan_use_immediate_data=true"]  # distinct list
    )

    assert first is second
    assert len(driver.create_calls) == 1
    assert sorted(iree_adapt._HAL_DEVICE_CACHE.devices) == [
        ("vulkan", 0, ("--vulkan_use_immediate_data=true",))
    ]


def test_different_index_acquires_a_second_device(monkeypatch):
    """The index is part of the cache key."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    device0 = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    device1 = iree_adapt._acquire_runtime_device("vulkan", VULKAN1, [])

    assert device0 is not device1
    assert len(driver.create_calls) == 2
    assert driver.create_calls[1][0] == (_RADV,)
    assert set(iree_adapt._HAL_DEVICE_CACHE.devices) == {
        ("vulkan", 0, ()),
        ("vulkan", 1, ()),
    }


def test_different_runtime_args_acquire_a_second_device(monkeypatch):
    """The runtime args are part of the cache key."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    baseline = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    flagged = iree_adapt._acquire_runtime_device(
        "vulkan", VULKAN0, ["--vulkan_use_immediate_data=true"]
    )

    assert baseline is not flagged
    assert len(driver.create_calls) == 2
    assert driver.query_calls == 2


def test_hal_device_cache_is_thread_local(monkeypatch):
    """A second thread gets (and caches) its OWN HAL device.

    THREAD-LOCAL IS REQUIRED, not an optimization: HAL devices are
    thread-affine (the ``etl.io`` worker thread uploads separately and must
    get a device it created itself). A process-global cache would hand that
    thread a device created elsewhere.
    """
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    main_device = iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    seen = {}

    def _worker():
        try:
            seen["first"] = iree_adapt._acquire_runtime_device(
                "vulkan", VULKAN0, []
            )
            seen["second"] = iree_adapt._acquire_runtime_device(
                "vulkan", VULKAN0, []
            )
        except BaseException as exc:  # surfaced below, never swallowed
            seen["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert "error" not in seen, f"worker acquisition failed: {seen.get('error')}"
    assert seen["first"] is seen["second"], "the worker reuses its own entry"
    assert seen["first"] is not main_device, "the cache is per thread"
    assert len(driver.create_calls) == 2
    # The main thread's entry is untouched by the worker's acquisition.
    assert iree_adapt._acquire_runtime_device("vulkan", VULKAN0, []) is main_device


# ---------------------------------------------------------------------------
# 3. acquisition errors: zero devices / out-of-range index
# ---------------------------------------------------------------------------


def test_zero_enumerated_devices_raises_with_the_icd_hint(monkeypatch):
    """No enumerated device -> actionable ``BackendError`` naming the remedy."""
    driver = _FakeVulkanDriver([])
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    with pytest.raises(core.BackendError) as excinfo:
        iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    message = str(excinfo.value)

    assert "enumerated no devices" in message
    assert repr(VULKAN0) in message
    assert "VK_ICD_FILENAMES" in message
    assert "llvmpipe" in message
    assert 'query_available_devices()' in message
    assert driver.create_calls == []
    # A failed acquisition is never cached: the next call re-queries.
    assert iree_adapt._HAL_DEVICE_CACHE.devices == {}
    with pytest.raises(core.BackendError):
        iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    assert driver.query_calls == 2


@pytest.mark.parametrize("index", [2, 7, -1])
def test_out_of_range_index_names_count_names_and_index(monkeypatch, index):
    """Out-of-range (and negative) index -> ``BackendError`` with the facts."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    device = core.Device("vulkan", index) if index >= 0 else _NegativeIndexDevice(index)
    with pytest.raises(core.BackendError) as excinfo:
        iree_adapt._acquire_runtime_device("vulkan", device, [])
    message = str(excinfo.value)

    assert "enumerated 2 device(s)" in message
    assert _LAVAPIPE["name"] in message
    assert _RADV["name"] in message
    assert f"asks for index {index}" in message, message
    assert "QUERIED HASHES, not ordinals" in message
    assert driver.create_calls == []
    assert iree_adapt._HAL_DEVICE_CACHE.devices == {}


class _NegativeIndexDevice:
    """A ``Device``-shaped stand-in with a negative index.

    ``core.Device`` rejects negative indices at construction (``ValueError``),
    so the adapter's own ``index < 0`` guard can only be reached with a
    duck-typed device object — the guard exists for exactly such a caller and
    must still produce the actionable error (never an ``IndexError`` from
    ``infos[index]`` wrapping around to the last device).
    """

    kind = "vulkan"

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return f"Device(kind={self.kind!r}, index={self.index})"


# ---------------------------------------------------------------------------
# 4. acquisition errors: create_device failures
# ---------------------------------------------------------------------------


def test_create_device_failure_is_wrapped_with_the_enumerated_info(monkeypatch):
    """A driver failure -> ``BackendError`` naming the device + info, chained."""
    original = RuntimeError("vkCreateInstance failed: no ICD for this platform")
    driver = _FakeVulkanDriver(_INFOS, create_error=original)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    with pytest.raises(core.BackendError) as excinfo:
        iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    message = str(excinfo.value)

    assert repr(VULKAN0) in message
    assert _LAVAPIPE["name"] in message
    assert str(_LAVAPIPE["device_id"]) in message
    assert _LAVAPIPE["path"] in message
    assert "vkCreateInstance failed: no ICD for this platform" in message
    assert excinfo.value.__cause__ is original, "must be from-chained"
    assert len(driver.create_calls) == 1
    assert iree_adapt._HAL_DEVICE_CACHE.devices == {}


@pytest.mark.parametrize("error_type", [core.BackendError, core.DeviceError])
def test_create_device_etl_error_propagates_unwrapped(monkeypatch, error_type):
    """An ``ETLError`` from ``create_device`` is re-raised as-is (no wrapping)."""
    original = error_type("the vulkan HAL refused this device")
    driver = _FakeVulkanDriver(_INFOS, create_error=original)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    with pytest.raises(error_type) as excinfo:
        iree_adapt._acquire_runtime_device("vulkan", VULKAN0, [])
    assert excinfo.value is original
    assert excinfo.value.__cause__ is None
    assert iree_adapt._HAL_DEVICE_CACHE.devices == {}


# ---------------------------------------------------------------------------
# 5. staging tier: vulkan has NONE (the cuda/rocm tiers are never reached)
# ---------------------------------------------------------------------------


def test_staged_input_returns_none_for_vulkan_without_touching_anything(
    monkeypatch,
):
    """``"vulkan"`` returns None before ANY buffer work.

    The staged-inputs dict, the runtime HAL device and the tensor are all
    access traps — the call can only succeed by not reading them (which also
    pins that nothing is host-materialized and no ``asdevicearray`` upload
    happens inside ``_staged_input``).
    """
    driver = _FakeVulkanDriver(_INFOS)
    record = _install_fake_runtime(monkeypatch, {"vulkan": driver})

    staged = iree_adapt._staged_input(
        _Trap("staged_inputs"),
        _Trap("runtime HAL device"),
        0,
        _Trap("tensor"),
        "vulkan",
    )

    assert staged is None
    assert record.asdevicearray == []
    assert record.get_driver == []
    assert iree_adapt._HAL_DEVICE_CACHE.devices == {}


def test_vulkan_staging_never_reaches_the_cuda_or_rocm_tiers(monkeypatch):
    """The kind dispatch is explicit: no cuda/rocm staging helper is entered."""
    driver = _FakeVulkanDriver(_INFOS)
    _install_fake_runtime(monkeypatch, {"vulkan": driver})

    reached = []

    def _boom(*args, **kwargs):
        reached.append(args)
        raise AssertionError("a cuda/rocm staging helper was entered for vulkan")

    monkeypatch.setattr(iree_adapt, "_rocm_staged_input", _boom)
    monkeypatch.setattr(iree_adapt, "_rocm_source_buffer", _boom)
    monkeypatch.setattr(iree_adapt, "_wait_copy_semaphore", _boom)

    tensor = core.Tensor(np.zeros((4,), np.float32))
    trap_device = _Trap("runtime HAL device")

    assert iree_adapt._staged_input({}, trap_device, 0, tensor, "vulkan") is None
    assert reached == []

    # Positive controls: the SAME call DOES enter a tier for cuda and rocm,
    # which is what makes the vulkan pin above meaningful rather than vacuous.
    with pytest.raises(AssertionError):
        iree_adapt._staged_input({}, trap_device, 0, tensor, "cuda")
    with pytest.raises(AssertionError):
        iree_adapt._staged_input({}, trap_device, 0, tensor, "rocm")
    assert len(reached) == 1, "exactly the rocm helper was entered"


# ---------------------------------------------------------------------------
# 6. provider registration: per-backend slot AND the flat default slot
# ---------------------------------------------------------------------------


def test_register_installs_the_vulkan_provider_in_both_registries(monkeypatch):
    """``("vulkan", "iree")`` AND the flat ``"vulkan"`` slot -> ``upload_tensor``."""
    _register_adapter(monkeypatch)

    assert (
        core._get_backend_device_transfer_provider("vulkan", "iree")
        is iree_adapt.upload_tensor
    )
    assert core._get_device_transfer_provider("vulkan") is iree_adapt.upload_tensor
    # The cuda/rocm/hip kinds get their PER-BACKEND slots from the same call
    # (their flat slots stay the etl.backends thunks — see the pin below).
    for kind in ("cuda", "rocm", "hip"):
        assert (
            core._get_backend_device_transfer_provider(kind, "iree")
            is iree_adapt.upload_tensor
        )


def test_register_is_idempotent(monkeypatch):
    """A second ``register()`` neither raises nor changes the registries."""
    _register_adapter(monkeypatch)
    per_backend = dict(_PER_BACKEND_PROVIDERS)
    flat = dict(_FLAT_PROVIDERS)

    iree_adapt.register()

    assert _PER_BACKEND_PROVIDERS == per_backend
    assert _FLAT_PROVIDERS == flat
    assert (
        core._get_backend_device_transfer_provider("vulkan", "iree")
        is iree_adapt.upload_tensor
    )
    assert core._get_device_transfer_provider("vulkan") is iree_adapt.upload_tensor


def test_tensor_to_resolution_would_dispatch_to_that_provider(monkeypatch):
    """``Tensor.to(Device("vulkan", 0))`` resolves to the registered provider.

    Pinned through the RESOLVED provider identity (never by performing an
    upload): ``core.Tensor.to`` consults the preferred backend's per-backend
    slot first, then the flat default slot — both resolve to
    ``iree.upload_tensor`` here, so a host tensor's ``.to`` lands on this
    provider rather than a ``DeviceError``.
    """
    _register_adapter(monkeypatch)
    target = core.Device("vulkan", 0)
    host = core.Tensor(np.arange(4, dtype=np.float32))
    assert host.device == core.Device("cpu", 0)

    preferred = core._get_preferred_transfer_backend("vulkan")
    provider = None
    if preferred is not None:
        provider = core._get_backend_device_transfer_provider("vulkan", preferred)
    if provider is None:
        provider = core._get_device_transfer_provider("vulkan")

    assert provider is iree_adapt.upload_tensor
    # The target differs from the host tensor's device, so ``.to`` would take
    # the provider branch (never the equal-device fast path).
    assert target != host.device


def test_flat_slots_for_cuda_rocm_hip_keep_the_lazy_backends_thunks(monkeypatch):
    """The asymmetric pin: only ``"vulkan"`` gets a flat slot from ``register()``.

    What is TRUE in this codebase: ``etl.backends`` fills the FLAT default
    slot for cuda / rocm / hip at import time with LAZY THUNKS over the iree
    adapter (``_cuda_transfer_thunk`` / ``_rocm_transfer_thunk``, which look
    ``upload_tensor`` up by name on first call) — never the adapter's direct
    ``upload_tensor``. The adapter's ``register()`` deliberately leaves those
    flat slots alone; it writes its PER-BACKEND slots for all four kinds and
    the flat slot for ``"vulkan"`` ONLY (no etl.backends thunk exists for
    vulkan and no other backend serves it, so that one slot cannot clobber
    another backend's provider).
    """
    import etl.backends as etl_backends

    flat_before = {
        kind: core._get_device_transfer_provider(kind)
        for kind in ("cuda", "rocm", "hip")
    }
    assert flat_before["cuda"] is etl_backends._cuda_transfer_thunk
    assert flat_before["rocm"] is etl_backends._rocm_transfer_thunk
    assert flat_before["hip"] is etl_backends._rocm_transfer_thunk
    for kind, provider in flat_before.items():
        assert provider is not iree_adapt.upload_tensor, kind

    _register_adapter(monkeypatch)

    for kind, before in flat_before.items():
        assert core._get_device_transfer_provider(kind) is before, kind
        assert core._get_device_transfer_provider(kind) is not iree_adapt.upload_tensor
    # vulkan, by contrast, now HAS a flat provider — the adapter's own.
    assert core._get_device_transfer_provider("vulkan") is iree_adapt.upload_tensor


# ---------------------------------------------------------------------------
# 7. the vulkan upload IS the plain ``rt.asdevicearray`` copy
# ---------------------------------------------------------------------------


def test_vulkan_upload_is_the_plain_asdevicearray_copy(monkeypatch):
    """``upload_tensor`` on vulkan: no staging tier, one ``asdevicearray``.

    The fake HAL device is a bare marker with no ``allocator`` /
    ``create_semaphore`` / ``queue_copy``, so a staging attempt would raise
    ``AttributeError`` — the plain copy path is what makes this test pass.
    """
    monkeypatch.delenv("ETL_IREE_RUNTIME_ARGS", raising=False)
    driver = _FakeVulkanDriver(_INFOS)
    record = _install_fake_runtime(monkeypatch, {"vulkan": driver})

    host = core.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    placed = iree_adapt.upload_tensor(host, VULKAN0)

    assert isinstance(placed, core.Tensor)
    assert placed.device == VULKAN0
    assert isinstance(placed.data, iree_adapt.IreeDevicePayload)
    assert placed.shape == (2, 3)
    assert placed.dtype == np.dtype(np.float32)

    assert len(record.asdevicearray) == 1
    runtime_device, uploaded = record.asdevicearray[0]
    assert runtime_device is driver.devices[0]
    assert isinstance(uploaded, np.ndarray)
    np.testing.assert_array_equal(uploaded, host.numpy())
    # The upload ran on a staging-free HAL device handle.
    assert not hasattr(driver.devices[0], "allocator")

    # The explicit download path needs no vulkan-specific code (the payload's
    # host copy) and returns the uploaded values.
    np.testing.assert_array_equal(
        placed.to(core.Device("cpu", 0)).numpy(), host.numpy()
    )

    # A second upload reuses the thread-local HAL device (one create_device).
    iree_adapt.upload_tensor(host, VULKAN0)
    assert len(driver.create_calls) == 1
    assert len(record.asdevicearray) == 2
