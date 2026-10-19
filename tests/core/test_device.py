"""Tests for the device model and explicit multi-device data preparation.

Covers ``etl.core.device``: ``Device``, ``devices()``, ``split_tensor`` and
``replicate_tensor``.  The etl package is READ-ONLY from the tests' point of
view: every assertion follows the implemented contract documented in
``etl/core/CONTEXT.md`` and ``etl/core/device.py`` (e.g. ``split_tensor``
returns *views* and ``replicate_tensor`` shares one buffer — both documented
"no copies" v1 semantics).

**v1 physical truth (binding):** ``split_tensor``/``replicate_tensor`` are
host-data-only helpers — the source must be an ndarray-backed tensor on
``Device('cpu', 0)`` and every target device must be ``Device('cpu', 0)``.
Non-host targets (cuda, or cpu with index != 0) and device-payload sources
raise ``DeviceError`` with the explicit-transfer remedy (never a raw
``AttributeError`` on a payload); axis/divisibility/type errors are unchanged
for valid host sources.

The ``devices()`` **ROCm** pins (hip/rocm alias, rocm presence in the unfiltered
list, the unknown-kind message, and "probe failure → ``[]``") inject FAKE
``iree``/``torch`` modules through ``monkeypatch`` so the probe is deterministic
without any ROCm runtime; a separate real leg skips cleanly when no ROCm runtime
is detectable in this process.

The ``devices("vulkan")`` pins use the same fake-``iree`` mechanism: they cover
the documented index contract (per-kind ENUMERATION index, never the iree HAL
hash ``device_id``), the lenient-probe contract (zero devices / a raising query
/ a raising ``get_driver`` / an unimportable ``iree`` all ⇒ ``[]``, never an
exception and never an invented device), and the request-only rule — the
unfiltered ``devices(None)`` list never includes yet never even queries vulkan.
"""

import ast
import inspect
import sys
import types
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from etl.core import Device, Tensor, devices, replicate_tensor, split_tensor, tensor
from etl.core.errors import DeviceError, ShapeError

# Stable fragments of the canonical v1 host-data-only DeviceError (see
# device._host_only_error): the source and target variants both carry the
# explicit-transfer remedy, so one regex asserts the documented fragments
# "host (numpy)", "Device('cpu', 0)", "no implicit device" and "t.to(".
_HOST_ONLY_FRAGMENTS = (
    r"host \(numpy\).*Device\('cpu', 0\).*no implicit device.*t\.to\("
)


class _DummyDevicePayload:
    """Minimal device-payload duck type: shape/dtype/device only.

    Deliberately has NO ndarray data (and no ``.ndim``) — it mirrors a
    device-resident buffer (e.g. an iree ``DeviceArray``) and pins the
    regression where ``split_tensor``/``replicate_tensor`` probed
    ``.data.ndim`` on payload sources and leaked a raw ``AttributeError``.
    """

    def __init__(self, device: Device):
        self.shape = (2, 3)
        self.dtype = np.float32
        self.device = device


# --- ROCm device-enumeration pins -------------------------------------------
#
# ``devices("rocm")`` and ``devices("hip")`` both delegate to ``_rocm_devices()``,
# whose probes run *inside* the function (``import iree.runtime as rt``) and
# which is documented to return ``[]`` whenever nothing is detectable (every
# probe is wrapped).  The mocked pins therefore install FAKE ``iree`` /
# ``iree.runtime`` (and a host-only ``torch``) module objects via the
# ``monkeypatch`` fixture: the in-function imports resolve through
# ``sys.modules``, so the fake is used whether or not the real, optional
# packages are installed.  ``monkeypatch`` restores ``sys.modules`` afterwards
# — the real modules are never mutated.

_ROCM_FAKE_COUNT = 3


class _FakeIreeHIPDriver:
    """Minimal stand-in for an IREE HAL driver: one info dict per device."""

    def __init__(self, count: int, name: str = "hip"):
        self.count = count
        self.name = name

    def query_available_devices(self):
        # IREE HAL device ids are 1-based; etl only uses the list LENGTH.
        return [{"device_id": i + 1} for i in range(self.count)]


def _inject_fake_iree_hip(monkeypatch, *, count=_ROCM_FAKE_COUNT, raises=False):
    """Install a FAKE ``iree.runtime`` whose HIP driver reports ``count`` devices.

    ``raises=True`` makes ``get_driver`` raise instead — the "probe failure"
    branch.  ``fake_iree.runtime`` is set as an *attribute* as well as a
    ``sys.modules`` entry because ``import iree.runtime as rt`` resolves the
    attribute on the parent package after the import returns.
    """
    fake_rt = types.ModuleType("iree.runtime")

    def get_driver(name):
        if raises:
            raise RuntimeError("fake iree: HIP driver unavailable")
        assert name == "hip", name
        return _FakeIreeHIPDriver(count, name)

    fake_rt.get_driver = get_driver
    fake_iree = types.ModuleType("iree")
    fake_iree.runtime = fake_rt
    monkeypatch.setitem(sys.modules, "iree", fake_iree)
    monkeypatch.setitem(sys.modules, "iree.runtime", fake_rt)


def _inject_fake_host_only_torch(monkeypatch):
    """Install a FAKE ``torch`` reporting NO ROCm build and NO CUDA device.

    ``_rocm_devices()`` falls back to a ROCm-capable torch build only when the
    IREE probe yields nothing, and ``devices("cuda")`` consults ``torch.cuda``
    — this fake neutralizes both, so the mocked pins stay deterministic even on
    a machine with a real ROCm-capable torch build.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.version = types.SimpleNamespace(hip=None, cuda=None)
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, device_count=lambda: 0
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


# --- Device -----------------------------------------------------------------


class TestDevice:
    def test_default_index(self):
        d = Device("cpu")
        assert d.kind == "cpu"
        assert d.index == 0
        assert Device("cuda", 2).index == 2

    @pytest.mark.parametrize("kind", ["", None, 5])
    def test_invalid_kind_raises(self, kind):
        with pytest.raises(ValueError, match="non-empty string"):
            Device(kind)

    @pytest.mark.parametrize("index", [-1, 1.5])
    def test_invalid_index_raises(self, index):
        with pytest.raises(ValueError, match="non-negative int"):
            Device("cpu", index)

    @pytest.mark.parametrize("attr", ["kind", "index", "extra"])
    def test_frozen(self, attr):
        # Frozen dataclass: no reassignment of fields, no new attributes.
        d = Device("cpu")
        with pytest.raises(FrozenInstanceError):
            setattr(d, attr, 1)

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (Device("cpu"), Device("cpu"), True),
            (Device("cpu"), Device("cpu", 0), True),
            (Device("cpu", 1), Device("cpu"), False),
            (Device("cpu", 0), Device("gpu", 0), False),
            (Device("cpu"), "cpu", False),
        ],
    )
    def test_equality(self, a, b, expected):
        assert (a == b) is expected

    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            (Device("cpu"), "Device(kind='cpu', index=0)"),
            (Device("cuda", 2), "Device(kind='cuda', index=2)"),
        ],
    )
    def test_repr(self, device, expected):
        # Exact format is defined by the source's custom __repr__.
        assert repr(device) == expected

    def test_hashable(self):
        # Frozen dataclass with structural __eq__ → hashable.
        assert hash(Device("cpu")) == hash(Device("cpu", 0))
        assert len({Device("cpu"), Device("cpu", 1), Device("gpu", 0)}) == 3
        d = {Device("cpu"): "x"}
        assert d[Device("cpu", 0)] == "x"


# --- devices() --------------------------------------------------------------


class TestDevices:
    def test_cpu_always_present(self):
        # CPU-only policy: never require a GPU, only assert cpu presence.
        assert Device("cpu", 0) in devices()

    def test_cpu_always_first(self):
        # Source ordering: [Device("cpu", 0)] + devices("cuda").
        assert devices()[0] == Device("cpu", 0)

    def test_deterministic(self):
        assert devices() == devices()

    def test_cpu_filter(self):
        result = devices("cpu")
        assert result == [Device("cpu", 0)]
        assert all(d.kind == "cpu" for d in result)

    def test_cuda_filter_wellformed(self):
        # cuda may be absent (no torch/GPU) but never malformed.
        result = devices("cuda")
        assert all(d.kind == "cuda" for d in result)
        assert [d.index for d in result] == list(range(len(result)))

    @pytest.mark.parametrize("kind", ["gpu", "nonexistent"])
    def test_unknown_kind_raises(self, kind):
        with pytest.raises(DeviceError, match="Unknown device kind"):
            devices(kind)

    def test_hip_is_alias_of_rocm(self):
        # "hip" is accepted as an alias/equivalent filter for "rocm" (both
        # call the same probe), so the two lists are ALWAYS equal — with or
        # without a ROCm runtime detectable in this process.
        assert devices("hip") == devices("rocm")

    def test_rocm_devices_all_mocked(self, monkeypatch):
        # Portable form of the "rocm appears in devices(None)" pin: the rocm
        # probe is made deterministic with fake iree/torch modules, so this
        # asserts the same thing on a non-ROCm machine.  devices(None)
        # composes cpu + cuda + rocm; every faked rocm index must appear,
        # asserted order-independently (membership), with cpu still first.
        _inject_fake_iree_hip(monkeypatch, count=_ROCM_FAKE_COUNT)
        _inject_fake_host_only_torch(monkeypatch)
        expected = [Device("rocm", i) for i in range(_ROCM_FAKE_COUNT)]
        assert devices("rocm") == expected
        assert devices("hip") == expected
        all_devices = devices(None)
        assert all_devices[0] == Device("cpu", 0)
        for device in expected:
            assert device in all_devices

    def test_rocm_devices_all_real(self):
        # Real leg: only meaningful where a ROCm runtime is actually
        # detectable (the IREE HIP HAL driver, else a ROCm-capable torch
        # build); skips cleanly on a non-ROCm host.
        rocm = devices("rocm")
        if not rocm:
            pytest.skip("no ROCm (HIP) runtime detectable in this process")
        all_devices = devices(None)
        assert all_devices[0] == Device("cpu", 0)
        for device in rocm:
            assert device in all_devices

    def test_unknown_kind_error_mentions_rocm(self):
        # The canonical unknown-kind error enumerates the recognized kinds,
        # including rocm and its hip alias — assert the fragments, not the
        # whole sentence.
        with pytest.raises(DeviceError) as excinfo:
            devices("bogus-kind")
        message = str(excinfo.value)
        assert "Unknown device kind" in message
        assert "rocm" in message
        assert "hip" in message

    def test_rocm_probe_failure_yields_empty(self, monkeypatch):
        # Documented contract: a failing probe means "not detectable" — no
        # exception leak and no invented device.  The fake host-only torch
        # build neutralizes the ROCm-capable-torch fallback, which is only
        # consulted when the IREE probe yields nothing.
        _inject_fake_iree_hip(monkeypatch, raises=True)
        _inject_fake_host_only_torch(monkeypatch)
        assert devices("rocm") == []
        assert devices("hip") == []
        # With both probes neutralized, the unfiltered list is cpu only.
        assert devices(None) == [Device("cpu", 0)]


# --- Vulkan device-enumeration pins -----------------------------------------
#
# ``devices("vulkan")`` delegates to ``_vulkan_devices()``, whose probe runs
# *inside* the function (``import iree.runtime as rt``) and which is documented
# to return ``[]`` whenever nothing is detectable (the whole probe is wrapped,
# exactly like ``_rocm_devices()``).  Same mechanism as the ROCm pins above: a
# FAKE ``iree`` / ``iree.runtime`` is installed through ``monkeypatch`` (the
# in-function import resolves via ``sys.modules``), so the pins hold whether or
# not the real optional package is installed, and ``monkeypatch`` restores
# ``sys.modules`` afterwards — the real modules are never mutated and every pin
# is order-independent.
#
# Index contract (binding, see ``_vulkan_devices``' docstring): ``Device(
# "vulkan", i)`` carries the PER-KIND ENUMERATION index ``i`` — deliberately
# NOT the IREE HAL ``device_id``, which for Vulkan is a per-process hash/UUID
# rather than the dense 1-based range the rocm HAL uses.  The fake info dicts
# therefore carry realistic, distinct, large hash ids, so "index == device_id"
# and "index == device_id - 1" would both fail loudly.

_VULKAN_FAKE_HASH_IDS = (857002912, 3910271561, 2104832)
_VULKAN_FAKE_UUIDS = (
    "8f2a4c1e-6b3d-4f7a-9c05-1de2b8a71344",
    "c41b9d07-2e58-4a13-bf6c-7d90ae31528b",
    "0a9f31c5-84d2-47e6-9b1a-532c8e7d6f01",
)
_VULKAN_FAKE_NAMES = (
    "lavapipe (LLVM 20.1.2, 256 bits)",
    "AMD Radeon RX 7900 XTX (RADV NAVI31)",
    "llvmpipe (LLVM 20.1.2, 256 bits)",
)

_VULKAN_FAKE_COUNT = len(_VULKAN_FAKE_HASH_IDS)


def _fake_vulkan_info(index: int) -> dict:
    """One ``iree.runtime`` device-info dict: hash id + UUID path + name."""
    return {
        "device_id": _VULKAN_FAKE_HASH_IDS[index],
        "path": _VULKAN_FAKE_UUIDS[index],
        "name": _VULKAN_FAKE_NAMES[index],
    }


class _FakeIreeVulkanDriver:
    """Minimal stand-in for the IREE Vulkan HAL driver.

    Reports ``runtime.vulkan_count`` info dicts, and counts every
    ``query_available_devices()`` call on its owning fake runtime so a pin can
    prove the vulkan probe was (or was NOT) reached.
    """

    def __init__(self, runtime: "_FakeIreeRuntime"):
        self._runtime = runtime

    def query_available_devices(self):
        self._runtime.vulkan_query_calls += 1
        if self._runtime.query_raises:
            raise RuntimeError("fake iree: vulkan device query failed")
        return [_fake_vulkan_info(i) for i in range(self._runtime.vulkan_count)]


class _FakeIreeRuntime:
    """Stand-in for ``iree.runtime``: per-driver fakes + call tracking.

    ``get_driver`` mirrors the real API for the only two names the enumeration
    paths use (``"vulkan"`` and ``"hip"``) and records every request in
    ``driver_requests``; ``vulkan_query_calls`` counts vulkan device queries.
    """

    def __init__(
        self,
        *,
        vulkan_count: int = 0,
        driver_raises: bool = False,
        query_raises: bool = False,
        hip_count: int = 0,
    ):
        self.vulkan_count = vulkan_count
        self.driver_raises = driver_raises
        self.query_raises = query_raises
        self.hip_count = hip_count
        self.driver_requests = []
        self.vulkan_query_calls = 0

    def get_driver(self, name):
        self.driver_requests.append(name)
        if name == "vulkan":
            if self.driver_raises:
                raise RuntimeError("fake iree: vulkan driver unavailable")
            return _FakeIreeVulkanDriver(self)
        if name == "hip":
            return _FakeIreeHIPDriver(self.hip_count, name)
        raise AssertionError(f"unexpected driver name {name!r}")


def _inject_fake_iree_vulkan(
    monkeypatch,
    *,
    count=_VULKAN_FAKE_COUNT,
    driver_raises=False,
    query_raises=False,
    hip_count=0,
):
    """Install a FAKE ``iree.runtime`` whose VULKAN driver reports ``count`` devices.

    ``driver_raises=True`` makes ``get_driver("vulkan")`` raise (driver
    initialization failure); ``query_raises=True`` makes the driver's
    ``query_available_devices()`` raise.  ``hip_count`` sets what the HIP
    driver reports, so the rocm path stays deterministic in the same fake.
    ``fake_iree.runtime`` is set as an *attribute* as well as a ``sys.modules``
    entry because ``import iree.runtime as rt`` resolves the attribute on the
    parent package after the import returns.  Returns the tracker.
    """
    runtime = _FakeIreeRuntime(
        vulkan_count=count,
        driver_raises=driver_raises,
        query_raises=query_raises,
        hip_count=hip_count,
    )
    fake_rt = types.ModuleType("iree.runtime")
    fake_rt.get_driver = runtime.get_driver
    fake_iree = types.ModuleType("iree")
    fake_iree.runtime = fake_rt
    monkeypatch.setitem(sys.modules, "iree", fake_iree)
    monkeypatch.setitem(sys.modules, "iree.runtime", fake_rt)
    return runtime


def _block_iree_imports(monkeypatch):
    """Simulate an absent ``iree``: a ``None`` ``sys.modules`` entry halts imports.

    ``import iree.runtime`` then raises ``ImportError`` (Python's
    ``None``-in-``sys.modules`` mechanism), which is the "optional extra not
    installed" branch of the probe.  ``monkeypatch`` restores both entries.
    """
    monkeypatch.setitem(sys.modules, "iree", None)
    monkeypatch.setitem(sys.modules, "iree.runtime", None)


class TestVulkanDevices:
    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_enumeration_index_is_per_kind_and_dense(self, monkeypatch, count):
        # N info dicts (hash ids!) ⇒ [Device("vulkan", i) for i in range(N)],
        # in order: the index is the per-kind ENUMERATION index, never the
        # iree HAL hash `device_id` and never `device_id - 1`.
        tracker = _inject_fake_iree_vulkan(monkeypatch, count=count)
        result = devices("vulkan")
        assert result == [Device("vulkan", i) for i in range(count)]
        assert all(d.kind == "vulkan" for d in result)
        assert [d.index for d in result] == list(range(count))
        assert all(isinstance(d.index, int) for d in result)
        # No index equals a hash id and none equals `hash id - 1`, whatever
        # the count (vacuously true for an empty enumeration, pinned below).
        assert all(d.index not in _VULKAN_FAKE_HASH_IDS for d in result)
        assert all(d.index + 1 not in _VULKAN_FAKE_HASH_IDS for d in result)
        if count:
            assert [d.index for d in result] != list(_VULKAN_FAKE_HASH_IDS[:count])
            assert [d.index for d in result] != [
                h - 1 for h in _VULKAN_FAKE_HASH_IDS[:count]
            ]
        # One driver request, one query: the count comes from the query alone.
        assert tracker.driver_requests == ["vulkan"]
        assert tracker.vulkan_query_calls == 1

    def test_zero_devices_yields_empty_list(self, monkeypatch):
        # [] is a real ANSWER (empty enumeration), not a swallowed exception:
        # the driver was queried exactly once and answered with no devices.
        tracker = _inject_fake_iree_vulkan(monkeypatch, count=0)
        assert devices("vulkan") == []
        assert tracker.vulkan_query_calls == 1
        assert tracker.driver_requests == ["vulkan"]

    def test_query_failure_yields_empty(self, monkeypatch):
        # A raising query_available_devices() ⇒ [] (best-effort enumeration,
        # lenient exactly like the rocm probe) — never an exception leak.
        tracker = _inject_fake_iree_vulkan(
            monkeypatch, count=_VULKAN_FAKE_COUNT, query_raises=True
        )
        assert devices("vulkan") == []
        assert tracker.vulkan_query_calls == 1

    def test_driver_initialization_failure_yields_empty(self, monkeypatch):
        # get_driver("vulkan") itself raising (driver init failure) ⇒ []; the
        # driver's query is never reached.
        tracker = _inject_fake_iree_vulkan(monkeypatch, driver_raises=True)
        assert devices("vulkan") == []
        assert tracker.driver_requests == ["vulkan"]
        assert tracker.vulkan_query_calls == 0

    def test_missing_iree_yields_empty(self, monkeypatch):
        # Optional-extra branch: an unimportable iree (None in sys.modules ⇒
        # ImportError) means "not detectable" ⇒ [], with no exception leak.
        _block_iree_imports(monkeypatch)
        # Anti-vacuity: the block really does halt the import the probe needs.
        with pytest.raises(ImportError):
            import iree.runtime  # noqa: F401  (deliberately failing import)
        assert devices("vulkan") == []
        # Positive control: the same probe DOES report devices once importable.
        _inject_fake_iree_vulkan(monkeypatch, count=_VULKAN_FAKE_COUNT)
        assert devices("vulkan") == [
            Device("vulkan", i) for i in range(_VULKAN_FAKE_COUNT)
        ]

    def test_iree_is_imported_lazily_not_at_module_scope(self):
        # `import etl` must never pull iree/torch in (optional extras): the
        # module-level import graph of etl.core.device names neither.  A
        # structural AST check is subprocess-free and order-independent —
        # another suite in this process may already have imported iree.
        import etl.core.device as device_module

        source = inspect.getsource(device_module)
        tree = ast.parse(source)
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module is not None
            ):
                top_level.add(node.module.split(".")[0])
        assert "iree" not in top_level
        assert "torch" not in top_level
        # Anti-vacuity: the probe imports are still present, function-scoped.
        assert "import iree.runtime" in source

    def test_none_never_includes_or_queries_vulkan(self, monkeypatch):
        # devices(None) composes cpu + cuda + rocm ONLY: vulkan is queried on
        # explicit request only (its probe force-instantiates the loader and a
        # software host would list a CPU rasterizer as an "accelerator").
        tracker = _inject_fake_iree_vulkan(
            monkeypatch, count=_VULKAN_FAKE_COUNT, hip_count=0
        )
        _inject_fake_host_only_torch(monkeypatch)
        result = devices(None)
        assert result == [Device("cpu", 0)]
        assert result[0] == Device("cpu", 0)
        assert all(d.kind != "vulkan" for d in result)
        assert "vulkan" not in tracker.driver_requests
        assert tracker.vulkan_query_calls == 0
        # Positive control on the SAME fake: an explicit request does see them
        # (devices(None) itself only reached the hip probe, never vulkan).
        assert devices("vulkan") == [
            Device("vulkan", i) for i in range(_VULKAN_FAKE_COUNT)
        ]
        assert tracker.driver_requests == ["hip", "vulkan"]
        assert tracker.vulkan_query_calls == 1

    def test_vulkan_never_leaks_into_rocm_hip_or_the_unfiltered_list(self, monkeypatch):
        # The three kind lists are disjoint: a vulkan-reporting driver never
        # adds devices to the rocm/hip lists nor to devices(None), and the
        # hip/rocm equivalence is unaffected.
        tracker = _inject_fake_iree_vulkan(
            monkeypatch, count=_VULKAN_FAKE_COUNT, hip_count=2
        )
        _inject_fake_host_only_torch(monkeypatch)
        assert devices("vulkan") == [
            Device("vulkan", i) for i in range(_VULKAN_FAKE_COUNT)
        ]
        expected_rocm = [Device("rocm", i) for i in range(2)]
        assert devices("rocm") == expected_rocm
        assert devices("hip") == devices("rocm") == expected_rocm
        unfiltered = devices(None)
        assert unfiltered[0] == Device("cpu", 0)
        assert [d for d in unfiltered if d.kind == "vulkan"] == []
        assert all(
            Device("vulkan", i) not in unfiltered
            for i in range(_VULKAN_FAKE_COUNT)
        )
        # Only the explicit devices("vulkan") reached the vulkan driver.
        assert tracker.driver_requests.count("vulkan") == 1
        assert tracker.vulkan_query_calls == 1

    def test_unknown_kind_error_names_vulkan(self):
        # The canonical unknown-kind error enumerates every recognized kind,
        # vulkan included (assert fragments, not the whole sentence).
        with pytest.raises(DeviceError) as excinfo:
            devices("vulcan")  # typo: not a recognized kind
        message = str(excinfo.value)
        assert "Unknown device kind" in message
        for fragment in ("'cpu'", "'cuda'", "'rocm'", "'hip'", "'vulkan'"):
            assert fragment in message


# --- split_tensor -----------------------------------------------------------


class TestSplitTensor:
    # v1 physical truth: every target must be Device('cpu', 0) — the helper
    # cannot place data on other devices, so multi-piece targets all name
    # the host device.
    TWO = [Device("cpu"), Device("cpu")]

    def test_split_along_axis(self):
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, 0, self.TWO)
        assert len(parts) == 2
        reference = np.arange(12).reshape(4, 3)
        for i, (part, expected) in enumerate(zip(parts, [reference[:2], reference[2:]])):
            assert np.array_equal(part.data, expected)
            assert part.shape == (2, 3)
            assert part.dtype == t.dtype
            assert part.device == self.TWO[i]
        assert parts[0] is not parts[1]

    def test_split_returns_views(self):
        # Documented: "np.split returns views of the input — no copies".
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, 0, self.TWO)
        assert np.shares_memory(parts[0].data, t.data)
        assert np.shares_memory(parts[1].data, t.data)

    def test_split_negative_axis(self):
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, -1, [Device("cpu")] * 3)
        assert [p.shape for p in parts] == [(4, 1)] * 3
        reference = np.arange(12).reshape(4, 3)
        for i, part in enumerate(parts):
            assert np.array_equal(part.data, reference[:, i : i + 1])
            assert part.device == Device("cpu")

    def test_split_single_device(self):
        # A single host target: one piece covering all the data.
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, 0, [Device("cpu")])
        assert len(parts) == 1
        assert np.array_equal(parts[0].data, t.data)
        assert parts[0].device == Device("cpu", 0)

    def test_original_unchanged(self):
        t = tensor(np.arange(12).reshape(4, 3))
        split_tensor(t, 0, self.TWO)
        assert np.array_equal(t.data, np.arange(12).reshape(4, 3))
        assert t.device == Device("cpu", 0)

    def test_empty_devices_raises(self):
        with pytest.raises(DeviceError, match="at least one device"):
            split_tensor(tensor(np.arange(4)), 0, [])

    @pytest.mark.parametrize("axis", [2, -3, 2.5])
    def test_axis_out_of_range_raises(self, axis):
        # 2-D tensor: 2 and -3 are out of range; the non-int 2.5 is out of
        # range as well and is caught by the same range check (DeviceError).
        t = tensor(np.arange(12).reshape(4, 3))
        with pytest.raises(DeviceError, match="out of range"):
            split_tensor(t, axis, self.TWO)

    def test_non_int_axis_in_range_raises(self):
        t = tensor(np.arange(4))
        with pytest.raises(
            DeviceError, match="not a valid axis: expected an integer, got float"
        ):
            split_tensor(t, 0.5, self.TWO)

    @pytest.mark.parametrize(
        ("shape", "n_devices"),
        [
            ((5, 3), 2),
            ((2, 3), 4),
        ],
    )
    def test_non_divisible_raises(self, shape, n_devices):
        # Dimension size not divisible by the chunk count → ShapeError, for a
        # valid host source and all-host targets (non-host targets are
        # covered by test_non_host_target_raises).
        t = tensor(np.zeros(shape))
        with pytest.raises(ShapeError, match="not divisible"):
            split_tensor(t, 0, [Device("cpu")] * n_devices)

    def test_non_tensor_raises(self):
        with pytest.raises(DeviceError, match="split_tensor expects a Tensor, got list"):
            split_tensor([1, 2, 3, 4], 0, self.TWO)

    @pytest.mark.parametrize(
        "target",
        [Device("cuda", 3), Device("cpu", 1)],
        ids=["cuda", "cpu-index-1"],
    )
    def test_non_host_target_raises(self, target):
        # v1 physical truth: only Device('cpu', 0) targets are legal — a cuda
        # (or cpu index != 0) target raises the canonical DeviceError with
        # the explicit-transfer remedy; the helper never places data on
        # other devices and never transfers implicitly.
        t = tensor(np.arange(12).reshape(4, 3))
        with pytest.raises(DeviceError, match=_HOST_ONLY_FRAGMENTS):
            split_tensor(t, 0, [target])

    def test_payload_source_raises_device_error(self):
        # Regression pin: a device-payload source raises the canonical
        # DeviceError (validated BEFORE any payload attribute access) — not a
        # raw AttributeError from the old .data.ndim probe.
        t = Tensor(_DummyDevicePayload(Device("cuda", 0)))
        with pytest.raises(DeviceError, match=_HOST_ONLY_FRAGMENTS):
            split_tensor(t, 0, [Device("cpu")])


# --- replicate_tensor -------------------------------------------------------


class TestReplicateTensor:
    # v1 physical truth: every target must be Device('cpu', 0).
    TWO = [Device("cpu"), Device("cpu")]

    def test_replicate_values_and_devices(self):
        t = tensor(np.arange(6).reshape(2, 3))
        parts = replicate_tensor(t, self.TWO)
        assert len(parts) == 2
        assert parts[0] is not parts[1]
        for part, device in zip(parts, self.TWO):
            assert np.array_equal(part.data, t.data)
            assert part.shape == t.shape
            assert part.device == device

    def test_replicate_shares_single_buffer(self):
        # Documented v1 semantics: "no copies" — replicas share one buffer
        # (not distinct ones).
        t = tensor(np.arange(6))
        parts = replicate_tensor(t, self.TWO)
        assert parts[0].data is parts[1].data
        assert parts[0].data is t.data

    def test_replicate_single_device(self):
        # A single host target: one replica sharing the source data.
        t = tensor(np.arange(6))
        parts = replicate_tensor(t, [Device("cpu")])
        assert len(parts) == 1
        assert parts[0].device == Device("cpu", 0)
        assert np.array_equal(parts[0].data, t.data)

    def test_original_unchanged(self):
        t = tensor(np.arange(6))
        replicate_tensor(t, self.TWO)
        assert np.array_equal(t.data, np.arange(6))
        assert t.device == Device("cpu", 0)

    def test_empty_devices_raises(self):
        with pytest.raises(DeviceError, match="at least one device"):
            replicate_tensor(tensor(np.arange(6)), [])

    def test_non_tensor_raises(self):
        with pytest.raises(
            DeviceError, match="replicate_tensor expects a Tensor, got ndarray"
        ):
            replicate_tensor(np.arange(6), self.TWO)

    @pytest.mark.parametrize(
        "target",
        [Device("cuda", 3), Device("cpu", 1)],
        ids=["cuda", "cpu-index-1"],
    )
    def test_non_host_target_raises(self, target):
        # v1 physical truth: only Device('cpu', 0) targets are legal — see
        # TestSplitTensor.test_non_host_target_raises for the contract.
        t = tensor(np.arange(6))
        with pytest.raises(DeviceError, match=_HOST_ONLY_FRAGMENTS):
            replicate_tensor(t, [target])

    def test_payload_source_raises_device_error(self):
        # Regression pin: same as TestSplitTensor's — a device-payload source
        # raises the canonical DeviceError, never an AttributeError.
        t = Tensor(_DummyDevicePayload(Device("cuda", 0)))
        with pytest.raises(DeviceError, match=_HOST_ONLY_FRAGMENTS):
            replicate_tensor(t, [Device("cpu")])
