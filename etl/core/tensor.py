"""Concrete materialized tensors and concrete creators.

In v1 a :class:`Tensor` wraps a numpy ``ndarray`` (dtype, shape, device,
data) — numpy host memory is the only concrete-computation path. DLPack
interop is zero-copy in both directions. ``SymbolicTensor`` (see
``symbolic.py``) is the graph-side counterpart and must never be confused
with a concrete ``Tensor``: concrete tensors cannot enter a graph implicitly
(closure capture is a ``TraceError``; ``etl.constant`` is the only explicit
way).

The concrete creators in this module (``tensor``, ``zeros``, ``ones``,
``full``, ``empty``, ``from_numpy``, ``from_dlpack``) are part of the only
concrete-computation paths in etl — random generation, ``linspace`` etc. must
go through compiled graphs, never be added here.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .device import Device
from .dtypes import dtype

__all__ = ["Tensor", "from_numpy", "from_dlpack", "tensor", "zeros", "ones", "full", "empty"]


class Tensor:
    """A materialized runtime tensor.

    v1: wraps a numpy ``ndarray``. Concrete tensors are *local physical
    tensors* — multi-device layouts are prepared explicitly via
    ``split_tensor``/``replicate_tensor``.

    Attributes:
        data: The underlying :class:`numpy.ndarray`. Mutable ``data`` access
            is documented as unsupported for graph constants (``etl.constant``
            snapshots — do not mutate embedded data).
        dtype: The dtype as a :class:`numpy.dtype` (derived from ``data``).
        shape: The concrete shape tuple (derived from ``data``).
        device: The :class:`Device` this tensor lives on (default
            ``Device("cpu", 0)``).
    """

    def __init__(self, data: np.ndarray, device: Optional[Device] = None):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Tensor data must be a numpy ndarray, got {type(data).__name__}")
        self.data = data
        self.device = device if device is not None else Device("cpu", 0)

    @property
    def dtype(self) -> np.dtype:
        """The tensor dtype (a :class:`numpy.dtype`)."""
        return dtype(self.data.dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The concrete shape tuple."""
        return tuple(self.data.shape)

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array — the same reference, no copy.

        Interop guarantee: this is zero-copy; mutating the returned array
        mutates the tensor's data (see the ``data`` attribute caveat about
        graph constants).
        """
        return self.data

    def __dlpack__(self, stream: Any = None):
        """Export the tensor via DLPack (zero-copy capsule).

        Delegates to the underlying array's ``__dlpack__``; the consumer
        shares memory with this tensor (no implicit copies on interop).

        Args:
            stream: Optional stream hint for the consumer (ignored for CPU
                host memory).

        Returns:
            A DLPack capsule (``PyCapsule``).
        """
        return self.data.__dlpack__(stream=stream)

    def __eq__(self, other: Any) -> bool:
        """Structural equality: identity of metadata + value.

        Two tensors are equal iff they have the same dtype, shape, device and
        elementwise-equal data (``numpy.array_equal``).
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        return (
            self.dtype == other.dtype
            and self.shape == other.shape
            and self.device == other.device
            and np.array_equal(self.data, other.data)
        )

    # Like ndarray, tensors are unhashable (value equality is not identity).
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device!r})"


def from_numpy(array: np.ndarray) -> Tensor:
    """Wrap a numpy array as a :class:`Tensor` without copying.

    The tensor shares memory with ``array`` (no implicit copy); the device is
    the default CPU device.

    Args:
        array: The numpy array to wrap.

    Returns:
        The new :class:`Tensor`.
    """
    raise NotImplementedError(
        "from_numpy is not implemented yet (architecture phase); "
        "it will wrap the array in a Tensor with no copy."
    )


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
    raise NotImplementedError(
        "from_dlpack is not implemented yet (architecture phase); "
        "it will consume the DLPack capsule with numpy (zero-copy)."
    )


def tensor(data: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor from array-like data.

    Converts ``data`` (numpy array, nested sequences, scalars) to a
    :class:`Tensor` on the given device (default CPU). If ``dtype`` is given,
    the data is converted to it (a copy is made only when necessary, per
    numpy conversion semantics).

    Args:
        data: Array-like data (ndarray, list, scalar, ...).
        dtype: Optional target dtype (anything ``etl.dtype`` accepts).
        device: Optional target device (default ``Device("cpu", 0)``).

    Returns:
        The new concrete :class:`Tensor`.
    """
    raise NotImplementedError(
        "tensor() is not implemented yet (architecture phase); "
        "it will wrap array-like data into a Tensor."
    )


def zeros(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor filled with zeros.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float64`` (numpy convention).
        device: Optional target device (default CPU).

    Returns:
        The new concrete :class:`Tensor`.
    """
    raise NotImplementedError(
        "zeros() is not implemented yet (architecture phase); "
        "it will create a zero-filled Tensor (default dtype float64, numpy convention)."
    )


def ones(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create a concrete tensor filled with ones.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float64`` (numpy convention).
        device: Optional target device (default CPU).

    Returns:
        The new concrete :class:`Tensor`.
    """
    raise NotImplementedError(
        "ones() is not implemented yet (architecture phase); "
        "it will create a one-filled Tensor (default dtype float64, numpy convention)."
    )


def full(
    shape: Any, fill_value: Any, dtype: Optional[Any] = None, device: Optional[Device] = None
) -> Tensor:
    """Create a concrete tensor filled with ``fill_value``.

    Args:
        shape: The concrete shape (tuple/list of ints).
        fill_value: Scalar value to fill with.
        dtype: Element dtype; defaults to the dtype numpy would infer from
            ``fill_value``.
        device: Optional target device (default CPU).

    Returns:
        The new concrete :class:`Tensor`.
    """
    raise NotImplementedError(
        "full() is not implemented yet (architecture phase); "
        "it will create a fill_value-filled Tensor (dtype inferred from the value, numpy convention)."
    )


def empty(shape: Any, dtype: Optional[Any] = None, device: Optional[Device] = None) -> Tensor:
    """Create an uninitialized concrete tensor.

    Caveat: contents are undefined (numpy ``empty`` semantics) — intended for
    buffers the caller fully overwrites.

    Args:
        shape: The concrete shape (tuple/list of ints).
        dtype: Element dtype; defaults to ``float64`` (numpy convention).
        device: Optional target device (default CPU).

    Returns:
        The new (uninitialized) concrete :class:`Tensor`.
    """
    raise NotImplementedError(
        "empty() is not implemented yet (architecture phase); "
        "it will create an uninitialized Tensor (default dtype float64, numpy convention)."
    )
