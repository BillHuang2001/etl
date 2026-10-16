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
"""

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
