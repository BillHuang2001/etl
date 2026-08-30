"""Tests for etl concrete ``Tensor``, the concrete creators, and DLPack interop.

Mirrors ``etl/core/tensor.py``. Contract points under test:

- ``Tensor`` wraps an ndarray *by reference*: ``.numpy()`` and ``.data`` are
  the same array (zero-copy; mutations are visible both ways).
- Creators follow etl's default-dtype rule: ``zeros``/``ones``/``empty``
  default to float32 (the library's documented default dtype — deliberate
  deviation from numpy's float64, matching ``TensorSpec``); ``full`` and
  ``tensor`` infer from the data/fill_value with an inferred float64 coerced
  to float32 for Python data, while existing ndarray inputs keep their own
  dtype (respected as-is); explicit dtype/device honored everywhere.
- Structural equality (dtype + shape + device + elementwise values),
  ``!=`` inverts it, and tensors are unhashable.
- DLPack is zero-copy in both directions. Note (numpy >= 2.0): the
  documented ``np.from_dlpack`` input is an *object exposing* ``__dlpack__``
  (e.g. the tensor itself), not a raw PyCapsule — a capsule has no
  ``__dlpack__`` method. etl's ``from_dlpack`` follows the same contract
  (accepts any ``__dlpack__``-exposing object; rejects raw capsules with
  ``DeviceError``).
"""

import numpy as np
import pytest

from etl.core import (
    Device,
    DeviceError,
    Tensor,
    empty,
    from_dlpack,
    from_numpy,
    full,
    ones,
    tensor,
    zeros,
)

CREATOR_SHAPES = [(), (3,), (2, 3), (4, 4)]


@pytest.fixture
def arr():
    """A small float32 (3, 4) sample array."""
    return np.arange(12, dtype=np.float32).reshape(3, 4)


@pytest.fixture
def t(arr):
    return Tensor(arr)


# ---------------------------------------------------------------------------
# Tensor basics: attributes, storage, zero-copy .numpy()
# ---------------------------------------------------------------------------


class TestTensorBasics:
    def test_attributes(self, arr, t):
        assert t.data is arr  # stored by reference, no copy
        assert isinstance(t.dtype, np.dtype)
        assert t.dtype == arr.dtype
        assert t.shape == arr.shape == (3, 4)
        assert isinstance(t.shape, tuple)
        assert t.device == Device("cpu", 0)  # default device

    def test_device_override(self, arr):
        t = Tensor(arr, device=Device("cpu", 2))
        assert t.device == Device("cpu", 2)

    def test_rejects_non_ndarray(self):
        with pytest.raises(TypeError, match="must be a numpy ndarray"):
            Tensor([[1, 2], [3, 4]])

    def test_numpy_is_the_same_array(self, arr, t):
        out = t.numpy()
        assert isinstance(out, np.ndarray)
        assert out is t.data  # zero-copy: identical object
        assert out is arr

    def test_mutation_visible_both_ways(self, t):
        t.numpy()[0, 0] = 123.0
        assert t.data[0, 0] == 123.0
        t.data[1, 1] = 456.0
        assert t.numpy()[1, 1] == 456.0


# ---------------------------------------------------------------------------
# Concrete creators: tensor / zeros / ones / full / empty / from_numpy
# ---------------------------------------------------------------------------


class TestCreators:
    @pytest.mark.parametrize("shape", CREATOR_SHAPES)
    def test_zeros(self, shape):
        t = zeros(shape)
        assert t.shape == shape
        assert t.device == Device("cpu", 0)
        assert t.dtype == np.dtype("float32")  # etl default dtype (not numpy's float64)
        assert np.all(t.data == 0)

    @pytest.mark.parametrize("shape", CREATOR_SHAPES)
    def test_ones(self, shape):
        t = ones(shape)
        assert t.shape == shape
        assert t.device == Device("cpu", 0)
        assert t.dtype == np.dtype("float32")  # etl default dtype (not numpy's float64)
        assert np.all(t.data == 1)

    @pytest.mark.parametrize(
        "shape,fill_value",
        [((), 3.5), ((4,), -2), ((2, 3), 7)],
    )
    def test_full(self, shape, fill_value):
        t = full(shape, fill_value)
        assert t.shape == shape
        assert t.device == Device("cpu", 0)
        assert t.data.size == int(np.prod(shape, dtype=int))
        assert np.all(t.data == fill_value)

    @pytest.mark.parametrize(
        "fill_value,expected_dtype",
        [(7, np.dtype("int64")), (2.5, np.dtype("float32"))],
    )
    def test_full_default_dtype_inferred_from_fill(self, fill_value, expected_dtype):
        # Contract: defaults to the dtype numpy infers from ``fill_value``,
        # except an inferred float64 becomes float32 (etl default dtype;
        # integer fills keep numpy's int64).
        t = full((2, 2), fill_value)
        assert t.dtype == expected_dtype

    @pytest.mark.parametrize("shape", CREATOR_SHAPES)
    def test_empty(self, shape):
        t = empty(shape)
        assert t.shape == shape
        assert t.device == Device("cpu", 0)
        assert t.dtype == np.dtype("float32")  # etl default dtype (not numpy's float64)
        assert t.data.size == int(np.prod(shape, dtype=int))
        t.data.fill(9)  # a real, writable buffer (contents otherwise undefined)
        assert np.all(t.data == 9)

    def test_tensor_default_dtype_rule(self):
        # Python data: numpy inference with float64 coerced to float32.
        assert tensor([1.0, 2.0]).dtype == np.dtype("float32")
        assert tensor([[1, 2], [3, 4]]).dtype == np.dtype("int64")  # ints keep int64
        assert tensor([True]).dtype == np.dtype("bool")
        assert tensor(1j).dtype == np.dtype("complex128")  # complex keeps complex128
        # Existing ndarrays carry an explicit dtype — respected as-is.
        assert tensor(np.zeros(3, dtype=np.float64)).dtype == np.dtype("float64")
        assert tensor(np.ones(3, dtype=np.float32)).dtype == np.dtype("float32")

    def test_tensor_scalar(self):
        t = tensor(5)
        assert t.shape == ()
        assert t.dtype == np.asarray(5).dtype
        assert t.data == 5

    @pytest.mark.parametrize(
        "factory,args,dtype",
        [
            (tensor, ([[1, 2, 3]],), "float32"),
            (zeros, ((2, 2),), "float32"),
            (ones, ((3,),), np.int8),
            (full, ((2, 2), 5), np.uint8),
            (empty, ((2, 3),), np.float16),
        ],
    )
    def test_creators_dtype_arg_respected(self, factory, args, dtype):
        t = factory(*args, dtype=dtype)
        assert t.dtype == np.dtype(dtype)

    def test_tensor_dtype_conversion(self):
        t = tensor([1, 2, 3], dtype=np.float32)
        assert t.dtype == np.dtype("float32")
        np.testing.assert_array_equal(t.data, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    @pytest.mark.parametrize(
        "factory,args",
        [
            (tensor, ([[1, 2]],)),
            (zeros, ((2, 2),)),
            (ones, ((3,),)),
            (full, ((2,), 1)),
            (empty, ((2,),)),
        ],
    )
    @pytest.mark.parametrize("dev", [Device("cpu", 0), Device("cpu", 3)])
    def test_creators_device_arg_respected(self, factory, args, dev):
        t = factory(*args, device=dev)
        assert t.device == dev

    def test_from_numpy_zero_copy(self):
        arr = np.arange(6, dtype=np.int16).reshape(2, 3)
        t = from_numpy(arr)
        assert t.data is arr  # stronger than shares_memory: same object
        assert t.dtype == arr.dtype
        assert t.shape == arr.shape
        assert t.device == Device("cpu", 0)
        arr[0, 0] = -1
        assert t.data[0, 0] == -1

    def test_from_numpy_rejects_non_ndarray(self):
        with pytest.raises(TypeError, match="expects a numpy ndarray"):
            from_numpy([1, 2, 3])


# ---------------------------------------------------------------------------
# Structural equality and unhashability
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equal_when_values_dtype_shape_device_match(self):
        t1 = Tensor(np.arange(8).reshape(2, 4))
        t2 = Tensor(np.arange(8).reshape(2, 4))
        assert t1 == t2
        assert not (t1 != t2)
        assert t1 == t1

    def test_different_values_not_equal(self, t):
        other = Tensor(np.zeros_like(t.data))
        assert t != other
        assert not (t == other)

    def test_different_dtype_not_equal(self, t):
        other = Tensor(t.data.astype(np.float64))  # same values, float64
        assert t != other
        assert not (t == other)

    def test_different_shape_not_equal(self, t):
        other = Tensor(t.data.reshape(2, 6).copy())
        assert t != other
        assert not (t == other)

    def test_different_device_not_equal(self, t):
        other = Tensor(t.data.copy(), device=Device("cpu", 1))
        assert t != other
        assert not (t == other)

    def test_non_tensor_not_equal(self, t):
        # __eq__ returns NotImplemented for non-tensors → falls back to False.
        assert not (t == "not a tensor")
        assert t != "not a tensor"
        assert not (t == [1, 2, 3])
        assert t != [1, 2, 3]

    def test_unhashable(self, t):
        with pytest.raises(TypeError, match="unhashable"):
            hash(t)


# ---------------------------------------------------------------------------
# DLPack interop (zero-copy both directions)
# ---------------------------------------------------------------------------


class TestDLPack:
    def test_dlpack_returns_fresh_capsule_per_call(self, t):
        cap = t.__dlpack__()
        assert type(cap).__name__ == "PyCapsule"
        cap2 = t.__dlpack__(stream=None)
        assert type(cap2).__name__ == "PyCapsule"
        assert cap2 is not cap  # a fresh capsule per call

    def test_dlpack_numpy_roundtrip(self, arr, t):
        # np.from_dlpack takes the exporter object and consumes the capsule
        # produced by t.__dlpack__() internally (numpy >= 2.0 does not accept
        # a raw PyCapsule — capsules have no __dlpack__ method).
        out1 = np.from_dlpack(t)
        out2 = np.from_dlpack(t)  # repeatable: fresh capsule on every call
        np.testing.assert_array_equal(out1, arr)
        np.testing.assert_array_equal(out2, arr)
        assert out1.dtype == arr.dtype
        assert np.shares_memory(out1, arr)  # zero-copy

    def test_from_dlpack_roundtrip_zero_copy(self, arr, t):
        rt = from_dlpack(t)
        assert isinstance(rt, Tensor)
        assert rt.device == Device("cpu", 0)
        assert rt.dtype == arr.dtype
        assert rt.shape == arr.shape
        np.testing.assert_array_equal(rt.data, arr)
        assert np.shares_memory(rt.data, arr)  # zero-copy

    def test_from_dlpack_accepts_ndarray(self, arr):
        rt = from_dlpack(arr)
        assert isinstance(rt, Tensor)
        assert rt.dtype == arr.dtype
        np.testing.assert_array_equal(rt.data, arr)
        assert np.shares_memory(rt.data, arr)

    def test_from_dlpack_requires_dlpack_object(self):
        # Documented contract: input must expose __dlpack__ (raw PyCapsules
        # are likewise not accepted — they have no __dlpack__ method).
        with pytest.raises(DeviceError, match="__dlpack__"):
            from_dlpack([1, 2, 3])


# ---------------------------------------------------------------------------
# Device-payload-backed Tensors (duck-typed payload protocol)
# ---------------------------------------------------------------------------


class _DummyDevicePayload:
    """Minimal duck-typed device payload (the ``core.Tensor`` payload protocol).

    Mirrors the protocol contract (``etl/core/tensor.py``): ``.shape`` (tuple),
    ``.dtype`` (numpy-normalizable), optional ``.device`` (a ``core.Device``),
    and a ``to_host()`` host-copy path returning a FRESH ndarray per call.
    Counts ``to_host`` calls so tests can prove the lazy-copy semantics.
    """

    def __init__(self, shape, dtype, device=None, values=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = device
        self._values = (
            np.zeros(shape, dtype=self.dtype)
            if values is None
            else np.asarray(values, dtype=self.dtype)
        )
        self.to_host_calls = 0

    def to_host(self):
        self.to_host_calls += 1
        return self._values.copy()  # a fresh host copy per call

    def __array__(self, dtype=None):
        # np.asarray fallback path: a fresh host copy per call, like to_host.
        return np.asarray(self._values, dtype=dtype)


class _AsarrayOnlyPayload:
    """A payload WITHOUT ``to_host`` — ``np.asarray`` is the only host path."""

    def __init__(self, shape, dtype, device=None, values=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = device
        self._values = (
            np.zeros(shape, dtype=self.dtype)
            if values is None
            else np.asarray(values, dtype=self.dtype)
        )

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class TestDevicePayload:
    """Device-payload-backed ``Tensor`` semantics (no GPU / no backend code)."""

    def test_explicit_device_wins_over_payload_device(self):
        payload = _DummyDevicePayload((2, 3), "float32", device=Device("cuda", 1))
        t = Tensor(payload, device=Device("cpu", 2))
        assert t.device == Device("cpu", 2)  # explicit device= argument wins

    def test_device_derived_from_payload_device(self):
        payload = _DummyDevicePayload((2, 3), "float32", device=Device("cuda", 1))
        t = Tensor(payload)
        assert t.device == Device("cuda", 1)

    def test_no_device_anywhere_raises(self):
        payload = _DummyDevicePayload((2, 3), "float32", device=None)
        with pytest.raises(DeviceError, match="requires a device"):
            Tensor(payload)

    def test_numpy_fresh_host_copy_per_call(self):
        payload = _DummyDevicePayload(
            (2, 2), "float32", device=Device("cpu", 0),
            values=[[1.0, 2.0], [3.0, 4.0]],
        )
        t = Tensor(payload)
        assert payload.to_host_calls == 0  # lazy: no host copy at construction
        first = t.numpy()
        second = t.numpy()
        assert payload.to_host_calls == 2  # one fresh host copy per call
        assert first is not second
        first[0, 0] = 999.0
        assert t.numpy()[0, 0] == 1.0  # mutation never leaks into later copies
        assert payload.to_host_calls == 3

    def test_data_is_the_payload_object(self):
        payload = _DummyDevicePayload((3,), "int64", device=Device("cpu", 0))
        t = Tensor(payload)
        assert t.data is payload

    def test_shape_dtype_from_payload(self):
        # dtype via a name string and via an np.dtype payload
        t1 = Tensor(_DummyDevicePayload((2, 3), "float32", device=Device("cpu", 0)))
        assert t1.shape == (2, 3)
        assert t1.dtype == np.dtype("float32")
        t2 = Tensor(_DummyDevicePayload((4,), np.dtype("int64"), device=Device("cpu", 0)))
        assert t2.shape == (4,)
        assert t2.dtype == np.dtype("int64")

    def test_eq_metadata_only_no_host_copy(self):
        a = _DummyDevicePayload(
            (2, 2), "float32", device=Device("cpu", 0),
            values=[[1.0, 2.0], [3.0, 4.0]],
        )
        b = _DummyDevicePayload(
            (2, 2), "float32", device=Device("cpu", 0),
            values=[[99.0, 98.0], [97.0, 96.0]],
        )
        ta, tb = Tensor(a), Tensor(b)
        assert ta == tb  # metadata-only: differing to_host values are EQUAL
        assert not (ta != tb)
        assert a.to_host_calls == 0 and b.to_host_calls == 0  # no hidden copies

    def test_eq_metadata_mismatch_not_equal(self):
        t1 = Tensor(_DummyDevicePayload((2, 2), "float32", device=Device("cpu", 0)))
        t2 = Tensor(_DummyDevicePayload((2, 2), "float64", device=Device("cpu", 0)))
        assert t1 != t2
        t3 = Tensor(_DummyDevicePayload((2, 2), "float32", device=Device("cpu", 1)))
        assert t1 != t3

    def test_dlpack_raises_device_error_mentioning_numpy(self):
        t = Tensor(_DummyDevicePayload((2, 2), "float32", device=Device("cpu", 0)))
        with pytest.raises(DeviceError, match="numpy"):
            t.__dlpack__()
        with pytest.raises(DeviceError, match="numpy"):
            t.__dlpack_device__()

    def test_asarray_fallback_payload(self):
        payload = _AsarrayOnlyPayload(
            (2, 2), "float32", device=Device("cpu", 0),
            values=[[1.0, 2.0], [3.0, 4.0]],
        )
        t = Tensor(payload)
        np.testing.assert_array_equal(
            t.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )

    def test_broken_payload_clear_error(self):
        class _NoShape:
            dtype = np.dtype("float32")

        with pytest.raises(TypeError, match="device payload"):
            Tensor(_NoShape())

    def test_ndarray_backed_behavior_unchanged(self, arr, t):
        # same-reference .numpy() (zero-copy)
        assert t.numpy() is t.data is arr
        # __eq__ is elementwise (array_equal): same values equal, others not
        assert t == Tensor(arr.copy())
        assert t != Tensor(np.zeros_like(arr))
        # dlpack roundtrip still works
        rt = from_dlpack(t)
        np.testing.assert_array_equal(rt.data, arr)
        assert np.shares_memory(rt.data, arr)


# ---------------------------------------------------------------------------
# Torch interop (optional dependency — skipped when torch is missing)
# ---------------------------------------------------------------------------


class TestTorchInterop:
    @pytest.mark.parametrize("np_dtype", [np.float32, np.float64, np.int32])
    def test_dlpack_export_to_torch(self, np_dtype):
        torch = pytest.importorskip("torch")
        arr = np.arange(12, dtype=np_dtype).reshape(3, 4)
        t = Tensor(arr)
        tt = torch.utils.dlpack.from_dlpack(t.__dlpack__())
        assert tt.dtype == torch.from_numpy(arr).dtype
        assert tuple(tt.shape) == arr.shape
        np.testing.assert_array_equal(tt.numpy(), arr)

    @pytest.mark.parametrize("dtype_name", ["float32", "float64", "int32"])
    def test_dlpack_import_from_torch(self, dtype_name):
        torch = pytest.importorskip("torch")
        tt = torch.arange(12, dtype=getattr(torch, dtype_name)).reshape(3, 4)
        # torch tensors expose __dlpack__ (the documented from_dlpack input).
        rt = from_dlpack(tt)
        assert isinstance(rt, Tensor)
        assert rt.device == Device("cpu", 0)
        assert rt.dtype == np.dtype(dtype_name)
        np.testing.assert_array_equal(rt.data, tt.numpy())
        assert np.shares_memory(rt.data, tt.numpy())  # zero-copy import
