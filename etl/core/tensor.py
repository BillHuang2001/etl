"""Concrete materialized tensors and concrete creators.

A :class:`Tensor` wraps either a numpy ``ndarray`` (dtype, shape, device,
data — numpy host memory) or an opaque *device payload* object (duck-typed:
``shape``/``dtype``/``device`` + a ``to_host()`` host-copy path, see
:class:`Tensor`). ndarray-backed tensors are the v1 default: DLPack interop
is zero-copy in both directions.

**Explicit placement (binding).** A tensor is always a *local physical
tensor*: ndarray-backed data IS host memory and therefore always lives on
``Device("cpu", 0)``; a device payload lives on the device it declares.
There are NO implicit device transfers in etl — ``.numpy()``/DLPack on a
non-CPU-kind payload raise :class:`DeviceError`, and placing data on another
device is always the explicit ``Tensor.to(device)`` (which dispatches to the
registered device-transfer provider for the target kind, see
:func:`register_device_transfer_provider`).

``SymbolicTensor`` (see ``symbolic.py``) is the graph-side counterpart and
must never be confused with a concrete ``Tensor``: concrete tensors cannot
enter a graph implicitly (closure capture is a ``TraceError``;
``etl.constant`` is the only explicit way).

The concrete creators in this module (``tensor``, ``zeros``, ``ones``,
``full``, ``empty``, ``from_numpy``, ``from_dlpack``) are part of the only
concrete-computation paths in etl — random generation, ``linspace`` etc. must
go through compiled graphs, never be added here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .device import Device
from .dtypes import dtype as _dtype
from .errors import DeviceError

__all__ = [
    "Tensor",
    "from_numpy",
    "from_dlpack",
    "tensor",
    "zeros",
    "ones",
    "full",
    "empty",
    # device-transfer provider hook (populated by etl.backends at import
    # time; internal getter _get_device_transfer_provider not in __all__)
    "register_device_transfer_provider",
]


def _default_dtype(inferred: Any) -> np.dtype:
    """The concrete-creator default-dtype rule for Python array-like data.

    When ``dtype=None``, creators follow numpy-style inference EXCEPT for the
    default floating-point width: the library's documented default dtype is
    float32 (``TensorSpec`` defaults to float32), so an inferred ``float64``
    becomes ``float32``. Integer inference (``int64``), bool, and complex
    (``complex128``) are unchanged — only the floating-point default deviates
    from numpy. Existing ``ndarray`` inputs are NOT coerced (they carry an
    explicit dtype; see :func:`tensor`).
    """
    dt = _dtype(inferred)
    if dt == np.dtype("float64"):
        return np.dtype("float32")
    return dt


# --- device-transfer provider registry (explicit placement; see Tensor.to) --
# Follows the core registration-hook pattern (operator handlers in
# symbolic.py): core only holds the registry; ``etl.backends`` registers the
# lazy "cuda" thunk at import time, and the iree adapter overwrites it with a
# direct provider when it activates (last registration wins). A provider
# receives a HOST tensor (ndarray-backed, Device("cpu", 0)) plus the target
# ``core.Device`` and returns a ``Tensor`` placed on the target device.

_DEVICE_TRANSFER_PROVIDERS: Dict[str, Callable] = {}


def register_device_transfer_provider(kind: str, provider: Callable) -> None:
    """Register the device-transfer provider for ``Tensor.to`` kind ``kind``.

    Called by ``etl.backends`` at import time (kind ``"cuda"`` → a lazy
    thunk over the iree adapter) and overwritten by the iree adapter when it
    activates — re-registering a kind replaces the previous provider.

    Args:
        kind: Device kind the provider places data on (e.g. ``"cuda"``).
        provider: Callable ``provider(tensor, device) -> Tensor`` receiving a
            host (ndarray-backed, ``Device("cpu", 0)``) tensor and the target
            :class:`Device`, returning a tensor placed on the target.

    Raises:
        TypeError: If ``kind`` is not a non-empty string or ``provider`` is
            not callable.
    """
    if not isinstance(kind, str) or not kind:
        raise TypeError("device-transfer provider kind must be a non-empty string")
    if not callable(provider):
        raise TypeError(
            f"Device-transfer provider for kind {kind!r} must be callable"
        )
    _DEVICE_TRANSFER_PROVIDERS[kind] = provider


def _get_device_transfer_provider(kind: str) -> Callable:
    """Return the registered provider for ``kind`` or raise a clear DeviceError.

    Internal cross-module contract (``Tensor.to`` and tests). A provider
    receives a host (ndarray-backed, ``Device("cpu", 0)``) tensor and a
    target ``core.Device`` and returns a ``Tensor`` placed on the target.
    """
    provider = _DEVICE_TRANSFER_PROVIDERS.get(kind)
    if provider is None:
        raise DeviceError(
            f"no device-transfer provider is registered for device kind "
            f"{kind!r}: Tensor.to cannot place data on devices of this "
            "kind. The etl iree backend registers a 'cuda' placement "
            "provider when activated — see "
            "core.register_device_transfer_provider."
        )
    return provider


class Tensor:
    """A materialized runtime tensor.

    Two payload kinds are supported:

    - a numpy ``ndarray`` (the v1 default): ``.data`` is the array itself,
      ``.numpy()`` returns the same reference (zero-copy), DLPack export is
      zero-copy, and ``__eq__`` compares metadata + elementwise values.
      ndarray-backed data IS host memory, so the tensor always lives on
      ``Device("cpu", 0)`` (physical truth — see :meth:`__init__`).
    - an opaque *device payload* object (duck-typed protocol — core never
      imports backend code): a device-resident buffer exposing ``.shape``
      (tuple of ints), ``.dtype`` (anything :func:`etl.dtype` normalizes),
      optionally ``.device`` (a :class:`Device` or ``None``), and a way to
      materialize a host copy — a ``to_host()`` method returning an
      ``ndarray`` (tried first) or convertibility via ``np.asarray(payload)``
      (fallback). For these tensors ``.data`` is the payload itself and
      ``.device`` is the payload's device (an explicit ``device=`` argument
      must agree with it). Host access is EXPLICIT: a CPU-kind payload's
      ``.numpy()`` performs a LAZY fresh host copy on demand (never cached,
      so device-memory updates are never stale and no host memory is
      retained), while a non-CPU-kind payload's ``.numpy()``/DLPack raise
      :class:`DeviceError` — no implicit device-to-host transfer; move the
      data with :meth:`Tensor.to`. ``__eq__`` compares metadata only
      (dtype/shape/device — never a hidden host copy).

    Concrete tensors are *local physical tensors* — multi-device layouts are
    prepared explicitly via ``split_tensor``/``replicate_tensor``, and moving
    a tensor to another device is the explicit :meth:`to` method.

    Attributes:
        data: The underlying :class:`numpy.ndarray`, or the device payload
            object for device-resident tensors. Mutable ``data`` access is
            documented as unsupported for graph constants (``etl.constant``
            snapshots — do not mutate embedded data).
        dtype: The dtype as a :class:`numpy.dtype` (derived from ``data``).
        shape: The concrete shape tuple (derived from ``data``).
        device: The :class:`Device` this tensor lives on. For ndarray-backed
            tensors this is always ``Device("cpu", 0)`` (numpy host memory).
            A device payload must know its device: an explicit ``device=``
            argument must equal ``payload.device`` when the payload declares
            one, else ``payload.device`` is used, else a :class:`DeviceError`
            is raised.
    """

    def __init__(self, data: Any, device: Optional[Device] = None):
        if isinstance(data, np.ndarray):
            # R1 physical truth: ndarray-backed data IS host memory, so the
            # tensor can only live on Device('cpu', 0).
            if device is not None and device != Device("cpu", 0):
                raise DeviceError(
                    f"Tensor data is host (numpy) memory and physically "
                    f"lives on Device('cpu', 0), not on {device!r}: a "
                    "numpy-backed tensor cannot be constructed on another "
                    "device. To place data on another device, transfer "
                    "explicitly after construction, e.g. "
                    "t.to(core.Device('cuda', 0)) — there is no implicit "
                    "transfer."
                )
            self.data = data
            self.device = Device("cpu", 0)
            return
        # Device payload (duck-typed protocol): must at least describe
        # itself (shape/dtype). Host materialization (to_host or
        # np.asarray) is checked lazily in .numpy().
        if not (hasattr(data, "shape") and hasattr(data, "dtype")):
            raise TypeError(
                "Tensor data must be a numpy ndarray or a device payload "
                "object exposing shape/dtype (plus to_host() or "
                f"__array__), got {type(data).__name__}"
            )
        payload_device = getattr(data, "device", None)
        if device is None:
            if payload_device is None:
                raise DeviceError(
                    "Tensor wrapping a device payload requires a device: "
                    "pass device=... or expose payload.device (a "
                    f"core.Device); payload of type {type(data).__name__} "
                    "provides neither"
                )
            if not isinstance(payload_device, Device):
                raise TypeError(
                    "payload.device must be a core.Device or None, got "
                    f"{type(payload_device).__name__} from "
                    f"{type(data).__name__}"
                )
            device = payload_device
        elif payload_device is not None:
            if not isinstance(payload_device, Device):
                raise TypeError(
                    "payload.device must be a core.Device or None, got "
                    f"{type(payload_device).__name__} from "
                    f"{type(data).__name__}"
                )
            if device != payload_device:
                raise DeviceError(
                    f"Tensor device mismatch: explicit device={device!r} "
                    f"conflicts with the payload's own device "
                    f"{payload_device!r}. A device payload physically "
                    "lives on its declared device and cannot be relabeled: "
                    "construct the Tensor without device= to use the "
                    "payload's device, or move the data explicitly with "
                    "t.to(...) between devices."
                )
        self.data = data
        self.device = device

    @property
    def dtype(self) -> np.dtype:
        """The tensor dtype (a :class:`numpy.dtype`)."""
        return _dtype(self.data.dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The concrete shape tuple."""
        return tuple(self.data.shape)

    def numpy(self) -> np.ndarray:
        """Return a numpy array view of the tensor's data.

        ndarray-backed tensors: the SAME array reference, zero-copy —
        mutating the returned array mutates the tensor's data (see the
        ``data`` attribute caveat about graph constants).

        Device-payload-backed tensors: host access is explicit. A CPU-kind
        payload (``Device('cpu', 0)`` — e.g. iree llvm-cpu buffers) gets a
        LAZY host copy materialized on demand — a FRESH copy per call,
        never cached (device memory may be updated in place by the backend,
        and caching would retain host memory for the tensor's lifetime). A
        NON-CPU-kind payload (e.g. cuda) raises :class:`DeviceError` — no
        implicit device-to-host transfer; call :meth:`to` first.
        """
        if isinstance(self.data, np.ndarray):
            return self.data
        if self.device.kind != "cpu":
            raise DeviceError(
                f"Tensor.numpy() needs host memory, but this tensor lives "
                f"on {self.device!r}: there is no implicit device-to-host "
                "transfer. Transfer explicitly first, e.g. "
                "t.to(core.Device('cpu', 0)).numpy()."
            )
        return self._host_copy()

    def to(self, device: Device) -> "Tensor":
        """Explicitly transfer/place this tensor on ``device``.

        The ONLY way to move a tensor across devices (no implicit transfers
        anywhere in etl). Semantics:

        - ``device == self.device`` → returns ``self`` (no copy).
        - ``device`` kind ``"cpu"`` (only ``Device('cpu', 0)`` exists):
          ndarray-backed sources are already on cpu:0; a device-payload
          source is copied to fresh host memory (the explicit D2H transfer).
        - any other kind (e.g. ``"cuda"``): dispatches to the registered
          device-transfer provider for that kind (see
          :func:`register_device_transfer_provider`). The provider receives
          a HOST tensor (ndarray-backed, cpu:0) in v1; a payload-backed
          source raises :class:`DeviceError` suggesting the explicit
          two-hop ``t.to(cpu).to(target)`` — cross-device copies are not
          implemented in v1.

        Args:
            device: The target :class:`Device`.

        Returns:
            A tensor on ``device`` (``self`` when already there).

        Raises:
            TypeError: If ``device`` is not a :class:`Device`.
            DeviceError: If the target does not exist (cpu index != 0), the
                transfer is not implemented in v1 (payload source to a
                non-cpu target), or no provider is registered for the
                target kind.
        """
        if not isinstance(device, Device):
            raise TypeError(
                f"Tensor.to expects a core.Device, got {type(device).__name__}"
            )
        if device == self.device:
            return self
        if device.kind == "cpu":
            if device.index != 0:
                raise DeviceError(
                    "Tensor.to: only Device('cpu', 0) exists for kind "
                    f"'cpu' (got {device!r}) — CPU tensors always live on "
                    "the host"
                )
            # An ndarray-backed source would already be on cpu:0 (equal →
            # returned above); reaching here means a device-payload source:
            # this is the EXPLICIT device-to-host transfer (fresh copy).
            return Tensor(self._host_copy())
        if not isinstance(self.data, np.ndarray):
            raise DeviceError(
                f"Tensor.to cannot copy {self.device!r} data directly to "
                f"{device!r}: cross-device copies are not implemented in "
                "v1. Transfer in two explicit hops instead: "
                "t.to(core.Device('cpu', 0)) first, then .to(target)."
            )
        provider = _DEVICE_TRANSFER_PROVIDERS.get(device.kind)
        if provider is None:
            raise DeviceError(
                f"Tensor.to cannot place data on {device!r}: no "
                f"device-transfer provider is registered for device kind "
                f"{device.kind!r}. The etl iree backend provides cuda "
                "placement — activate it (import etl.backends or the iree "
                "adapter) or register a provider via "
                "core.register_device_transfer_provider."
            )
        return provider(self, device)

    def _host_copy(self) -> np.ndarray:
        """Materialize a fresh host numpy copy from a device payload.

        Payload protocol: ``payload.to_host() -> np.ndarray`` is tried
        first; ``np.asarray(payload)`` is the fallback. A clear
        ``TypeError`` is raised when neither yields a real host array
        (never silently wrong data).
        """
        to_host = getattr(self.data, "to_host", None)
        if callable(to_host):
            arr = to_host()
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    "payload.to_host() must return a numpy ndarray, got "
                    f"{type(arr).__name__} from {type(self.data).__name__}"
                )
            return arr
        try:
            arr = np.asarray(self.data)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Tensor device payload cannot materialize a host copy: it "
                "must provide to_host() -> np.ndarray or support "
                f"np.asarray; {type(self.data).__name__} provides neither"
            ) from exc
        if arr.dtype == object:
            raise TypeError(
                "Tensor device payload cannot materialize a host copy: "
                "np.asarray produced an object array; it must provide "
                f"to_host() -> np.ndarray or a numeric __array__; "
                f"{type(self.data).__name__} provides neither"
            )
        return arr

    def _dlpack_unavailable(self) -> DeviceError:
        """The error raised when DLPack is requested on a device payload.

        CPU-kind payloads (host-mappable memory): DLPack is unavailable —
        the caller materializes a host copy via ``.numpy()`` first.
        Non-CPU-kind payloads (e.g. cuda): the memory is genuinely not host
        memory — the caller must transfer explicitly (``t.to(...)``) first.
        """
        if self.device.kind == "cpu":
            return DeviceError(
                "Tensor.__dlpack__ is unavailable for a device payload "
                f"(device={self.device!r}): call .numpy() first to "
                "materialize a host copy"
            )
        return DeviceError(
            f"Tensor.__dlpack__ is unavailable for a device payload "
            f"(device={self.device!r}): the memory is not host memory, and "
            "there is no implicit device-to-host transfer. Transfer "
            "explicitly first, e.g. t.to(core.Device('cpu', 0)), then call "
            ".numpy() and export the host array."
        )

    def __dlpack__(self, stream: Any = None):
        """Export the tensor via DLPack (zero-copy capsule).

        ndarray-backed tensors: delegates to the underlying array's
        ``__dlpack__``; the consumer shares memory with this tensor (no
        implicit copies on interop).

        Device-payload-backed tensors: DLPack is unavailable (exporting a
        device buffer requires a device-aware consumer) — raises
        :class:`DeviceError`. CPU-kind payloads are told to materialize a
        host copy via ``.numpy()`` first; non-CPU-kind payloads (e.g. cuda)
        are told to transfer explicitly via ``t.to(...)`` first (no
        implicit device-to-host transfer).

        Args:
            stream: Optional stream hint for the consumer (ignored for CPU
                host memory).

        Returns:
            A DLPack capsule (``PyCapsule``).
        """
        if not isinstance(self.data, np.ndarray):
            raise self._dlpack_unavailable()
        try:
            return self.data.__dlpack__(stream=stream)
        except TypeError:
            # numpy >= 2.0 requires the keyword-only ``max_version`` argument
            # (numpy < 2.0 rejects it — hence the plain call above first).
            return self.data.__dlpack__(stream=stream, max_version=(1, 0))

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Report the DLPack device where this tensor's memory lives.

        The DLPack Python protocol requires ``__dlpack_device__`` alongside
        ``__dlpack__``; consumers (e.g. ``torch.utils.dlpack.from_dlpack``)
        call it to learn the exporter's device before consuming the capsule.

        ndarray-backed tensors always wrap numpy host memory, so the memory
        physically lives on the CPU — we report the device the memory
        ACTUALLY lives on, even if a ``device`` label (core.Device) is
        attached. Per dlpack.h: kDLCPU = 1, kDLCUDA = 2, ...; device id 0 is
        the sole CPU.

        Device-payload-backed tensors: DLPack is unavailable — raises
        :class:`DeviceError` (see :meth:`__dlpack__`).

        Returns:
            A ``(device_type, device_id)`` tuple: ``(1, 0)`` for CPU host
            memory.
        """
        if not isinstance(self.data, np.ndarray):
            raise self._dlpack_unavailable()
        return (1, 0)

    def __eq__(self, other: Any) -> bool:
        """Structural equality: identity of metadata (+ value for host data).

        Two ndarray-backed tensors are equal iff they have the same dtype,
        shape, device and elementwise-equal data (``numpy.array_equal``) —
        unchanged behavior.

        If EITHER tensor is device-payload-backed, ``__eq__`` compares
        metadata ONLY (dtype/shape/device) — never a hidden host copy
        (``.numpy()``/``to()`` are the explicit ways to access or move the
        data).
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        metadata_equal = (
            self.dtype == other.dtype
            and self.shape == other.shape
            and self.device == other.device
        )
        if not metadata_equal:
            return False
        if isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            return np.array_equal(self.data, other.data)
        return True

    # Like ndarray, tensors are unhashable (value equality is not identity).
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device!r})"


def from_numpy(array: np.ndarray) -> Tensor:
    """Wrap a numpy array as a :class:`Tensor` without copying.

    The tensor shares memory with ``array`` (no implicit copy); the device is
    the default CPU device. This is the dtype-preserving, zero-copy wrap
    path — unlike :func:`tensor`, the array's dtype is kept exactly as-is
    (no float64 → float32 default-dtype coercion).

    Args:
        array: The numpy array to wrap.

    Returns:
        The new :class:`Tensor`.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"from_numpy expects a numpy ndarray, got {type(array).__name__}"
        )
    return Tensor(array)


def from_dlpack(obj: Any) -> Tensor:
    """Import a tensor from any object exposing ``__dlpack__`` (zero-copy).

    Accepts any object with a ``__dlpack__`` method (numpy arrays, torch
    tensors via the ``interop`` extra, other tensor libraries). No implicit
    copies on interop — the tensor shares memory with the exporter.

    Args:
        obj: An object exposing ``__dlpack__(stream=None)``.

    Returns:
        The imported :class:`Tensor` (CPU host memory, v1).

    Raises:
        DeviceError: If ``obj`` has no ``__dlpack__`` or the capsule cannot
            be consumed.
    """
    if not callable(getattr(obj, "__dlpack__", None)):
        raise DeviceError(
            "from_dlpack requires an object exposing a __dlpack__ method, "
            f"got {type(obj).__name__}"
        )
    try:
        array = np.from_dlpack(obj)
    except (TypeError, RuntimeError, BufferError, ValueError) as exc:
        raise DeviceError(
            f"Failed to consume the DLPack capsule from {type(obj).__name__}: {exc}"
        ) from exc
    return Tensor(array, device=Device("cpu", 0))


def _check_creator_device(creator: str, device: Optional[Device]) -> None:
    """R1 physical truth for the concrete creators: host-only targets.

    Concrete creators (``tensor``/``zeros``/``ones``/``full``/``empty``)
    produce ndarray-backed (host) tensors, so ``device`` may only be
    ``None`` or ``Device("cpu", 0)``. Anything else raises a
    creator-specific :class:`DeviceError` naming the creator — never letting
    ``Tensor.__init__``'s message leak through. Placing data on another
    device is an explicit post-creation ``Tensor.to(...)`` transfer.
    """
    if device is None or device == Device("cpu", 0):
        return
    raise DeviceError(
        f"etl.{creator}(...) creates host (numpy) memory, which can only "
        f"live on Device('cpu', 0), not on {device!r}: concrete creators "
        "cannot place data on other devices. Create the tensor on the "
        "host and transfer explicitly with t.to(...) — there is no "
        "implicit transfer."
    )


def tensor(data: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor from array-like data.

    Converts ``data`` (numpy array, nested sequences, scalars) to a
    :class:`Tensor` on the given device (default CPU). If ``dtype`` is given,
    the data is converted to it (a copy is made only when necessary, per
    numpy conversion semantics).

    Default dtype rule (``dtype=None``): an existing ``ndarray`` keeps its
    own dtype (an ndarray always carries an explicit dtype — respected
    as-is, no copy); Python array-likes (lists, tuples, scalars) follow
    numpy inference EXCEPT an inferred ``float64`` becomes ``float32`` — the
    library's documented default dtype (see :func:`_default_dtype`; integer
    data stays ``int64``, bool stays bool, complex stays ``complex128``).
    For a zero-copy, dtype-preserving wrap of an existing array use
    :func:`from_numpy` instead.

    Args:
        data: Array-like data (ndarray, list, scalar, ...).
        dtype: Optional target dtype (anything ``etl.dtype`` accepts).
        device: Optional target device — only ``None`` (default) or
            ``Device("cpu", 0)``: creators produce host (numpy) memory
            (place on another device via ``Tensor.to`` afterwards).

    Returns:
        The new concrete :class:`Tensor`.

    Raises:
        DeviceError: If ``device`` is anything but ``Device("cpu", 0)``.
    """
    _check_creator_device("tensor", device)
    if dtype is None:
        inferred = np.asarray(data).dtype
        # ndarrays carry an explicit dtype (respected as-is); Python data
        # goes through the default-dtype rule (float64 → float32).
        if not isinstance(data, np.ndarray):
            inferred = _default_dtype(inferred)
        array = np.asarray(data, dtype=inferred)
    else:
        array = np.asarray(data, dtype=_dtype(dtype))
    return Tensor(array, device=device)


def zeros(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor filled with zeros.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float32`` — the library's
            documented default dtype (numpy defaults to ``float64``; etl
            deliberately deviates, matching ``TensorSpec``).
        device: Optional target device — only ``None`` (default) or
            ``Device("cpu", 0)`` (host memory; other devices need an
            explicit ``Tensor.to`` transfer).

    Returns:
        The new concrete :class:`Tensor`.

    Raises:
        DeviceError: If ``device`` is anything but ``Device("cpu", 0)``.
    """
    _check_creator_device("zeros", device)
    if dtype is None:
        dtype = np.float32
    array = np.zeros(shape, dtype=_dtype(dtype))
    return Tensor(array, device=device)


def ones(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor filled with ones.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float32`` — the library's
            documented default dtype (numpy defaults to ``float64``; etl
            deliberately deviates, matching ``TensorSpec``).
        device: Optional target device — only ``None`` (default) or
            ``Device("cpu", 0)`` (host memory; other devices need an
            explicit ``Tensor.to`` transfer).

    Returns:
        The new concrete :class:`Tensor`.

    Raises:
        DeviceError: If ``device`` is anything but ``Device("cpu", 0)``.
    """
    _check_creator_device("ones", device)
    if dtype is None:
        dtype = np.float32
    array = np.ones(shape, dtype=_dtype(dtype))
    return Tensor(array, device=device)


def full(
    shape: Any, fill_value: Any, dtype: Optional[Any] = None, device: Optional[Device] = None
) -> Tensor:
    """Create a concrete tensor filled with ``fill_value``.

    Args:
        shape: The concrete shape (tuple/list of ints).
        fill_value: Scalar value to fill with.
        dtype: Element dtype; defaults to numpy inference from
            ``fill_value``, except an inferred ``float64`` becomes ``float32``
            (the library's documented default dtype; integer fills keep
            ``int64``).
        device: Optional target device — only ``None`` (default) or
            ``Device("cpu", 0)`` (host memory; other devices need an
            explicit ``Tensor.to`` transfer).

    Returns:
        The new concrete :class:`Tensor`.

    Raises:
        DeviceError: If ``device`` is anything but ``Device("cpu", 0)``.
    """
    _check_creator_device("full", device)
    if dtype is None:
        dtype = _default_dtype(np.asarray(fill_value).dtype)
    array = np.full(shape, fill_value, dtype=_dtype(dtype))
    return Tensor(array, device=device)


def empty(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create an uninitialized concrete tensor.

    Caveat: contents are undefined (numpy ``empty`` semantics) — intended for
    buffers the caller fully overwrites.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float32`` — the library's
            documented default dtype (numpy defaults to ``float64``; etl
            deliberately deviates, matching ``TensorSpec``).
        device: Optional target device — only ``None`` (default) or
            ``Device("cpu", 0)`` (host memory; other devices need an
            explicit ``Tensor.to`` transfer).

    Returns:
        The new (uninitialized) concrete :class:`Tensor`.

    Raises:
        DeviceError: If ``device`` is anything but ``Device("cpu", 0)``.
    """
    _check_creator_device("empty", device)
    if dtype is None:
        dtype = np.float32
    array = np.empty(shape, dtype=_dtype(dtype))
    return Tensor(array, device=device)
