"""Tests for the device model and explicit multi-device data preparation.

Covers ``etl.core.device``: ``Device``, ``devices()``, ``split_tensor`` and
``replicate_tensor``.  The etl package is READ-ONLY from the tests' point of
view: every assertion follows the implemented contract documented in
``etl/core/CONTEXT.md`` and ``etl/core/device.py`` (e.g. ``split_tensor``
returns *views* and ``replicate_tensor`` shares one buffer — both documented
"no copies" v1 semantics).
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from etl.core import Device, devices, replicate_tensor, split_tensor, tensor
from etl.core.errors import DeviceError, ShapeError


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


# --- split_tensor -----------------------------------------------------------


class TestSplitTensor:
    TWO = [Device("cpu"), Device("cpu", 1)]

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

    def test_split_negative_axis(self):
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, -1, [Device("cpu")] * 3)
        assert [p.shape for p in parts] == [(4, 1)] * 3
        reference = np.arange(12).reshape(4, 3)
        for i, part in enumerate(parts):
            assert np.array_equal(part.data, reference[:, i : i + 1])
            assert part.device == Device("cpu")

    def test_split_single_device(self):
        t = tensor(np.arange(12).reshape(4, 3))
        parts = split_tensor(t, 0, [Device("cuda", 3)])
        assert len(parts) == 1
        assert np.array_equal(parts[0].data, t.data)
        assert parts[0].device == Device("cuda", 3)

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
        # Source does not type-validate axis: an in-range non-int axis slips
        # through the range check and numpy raises the TypeError.
        t = tensor(np.arange(4))
        with pytest.raises(TypeError):
            split_tensor(t, 0.5, self.TWO)

    @pytest.mark.parametrize(
        ("shape", "target_devices"),
        [
            ((5, 3), [Device("cpu"), Device("cpu", 1)]),
            ((2, 3), [Device("cpu")] * 4),
        ],
    )
    def test_non_divisible_raises(self, shape, target_devices):
        t = tensor(np.zeros(shape))
        with pytest.raises(ShapeError, match="not divisible"):
            split_tensor(t, 0, target_devices)

    def test_non_tensor_raises(self):
        # Source has no explicit type check: attribute access fails.
        with pytest.raises(AttributeError):
            split_tensor([1, 2, 3, 4], 0, self.TWO)


# --- replicate_tensor -------------------------------------------------------


class TestReplicateTensor:
    TWO = [Device("cpu"), Device("cpu", 1)]

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
        # Documented v1 semantics: "The SAME ndarray object tagged with each
        # device — no copies."  Replicas share one buffer (not distinct ones).
        t = tensor(np.arange(6))
        parts = replicate_tensor(t, self.TWO)
        assert parts[0].data is parts[1].data
        assert parts[0].data is t.data

    def test_replicate_single_device(self):
        t = tensor(np.arange(6))
        parts = replicate_tensor(t, [Device("cuda", 3)])
        assert len(parts) == 1
        assert parts[0].device == Device("cuda", 3)
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
        # An ndarray slips through attribute access (its `.data` is a
        # memoryview) and Tensor() rejects it.
        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            replicate_tensor(np.arange(6), self.TWO)
