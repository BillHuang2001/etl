"""Device model and explicit multi-device data-preparation helpers.

A tensor is always a *local* physical tensor. ``Device`` identifies a
physical device; ``devices()`` enumerates what is actually available;
``split_tensor``/``replicate_tensor`` are *explicit* data-preparation
utilities that produce ordinary local ``Tensor``\\s — they do no graph
rewriting, no implicit communication, and never shard inside the graph
(collectives are the explicit in-graph communication mechanism, owned by
``etl.dist``).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from .errors import DeviceError, ShapeError

if TYPE_CHECKING:  # avoid a runtime import cycle: tensor.py imports device.py
    from .tensor import Tensor

__all__ = ["Device", "devices", "split_tensor", "replicate_tensor"]


@dataclass(frozen=True)
class Device:
    """A physical device: a kind plus an optional index.

    v1 kinds: ``"cpu"`` (always available); ``"cuda"`` may be listed by
    ``devices()`` when a CUDA runtime is detectable (see ``devices``).

    Attributes:
        kind: Device kind string, e.g. ``"cpu"`` or ``"cuda"``.
        index: Device index within the kind (default 0).
    """

    kind: str
    index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"Device.kind must be a non-empty string, got {self.kind!r}")
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"Device.index must be a non-negative int, got {self.index!r}")

    def __repr__(self) -> str:
        return f"Device(kind={self.kind!r}, index={self.index})"


def _tensor(data: np.ndarray, device: Device) -> "Tensor":
    """Build a :class:`Tensor` from data + device (deferred import).

    ``tensor.py`` imports ``device.py`` at module level, so ``Tensor`` is
    imported lazily inside this helper to keep the import graph acyclic.
    """
    from .tensor import Tensor  # deferred: avoids a runtime import cycle

    return Tensor(data, device=device)


def devices(kind: Optional[str] = None) -> List[Device]:
    """Enumerate the devices actually available to this process.

    v1 semantics (honest, minimal): ``kind=None`` lists all detectable
    devices; the CPU device (``Device("cpu", 0)``) is *always* present;
    ``"cuda"`` devices are listed only if a CUDA runtime is detectable in
    this process (e.g. torch with CUDA) — otherwise the cuda list is empty.
    No devices are invented, and etl never requires a GPU.

    Args:
        kind: Optional filter; only devices of this kind are returned.

    Returns:
        The list of :class:`Device`\\s.

    Raises:
        DeviceError: If ``kind`` is not a recognized device kind.
    """
    if kind == "cpu":
        return [Device("cpu", 0)]
    if kind == "cuda":
        try:
            import torch  # lazy: torch is an optional interop extra
        except ImportError:
            return []
        if torch.cuda.is_available():
            return [Device("cuda", i) for i in range(torch.cuda.device_count())]
        return []
    if kind is None:
        return [Device("cpu", 0)] + devices("cuda")
    raise DeviceError(
        f"Unknown device kind {kind!r}; recognized kinds are 'cpu' and 'cuda'."
    )


def _invalid_axis_error(axis: Any) -> DeviceError:
    """Canonical invalid-axis error: name the bad value and its type."""
    return DeviceError(
        f"split_tensor axis {axis!r} is not a valid axis: expected an integer, "
        f"got {type(axis).__name__}"
    )


def split_tensor(tensor: "Tensor", axis: int, devices: List[Device]) -> List["Tensor"]:
    """Split a local tensor along ``axis`` into one tensor per device.

    Pure explicit data preparation: slices ``tensor`` into ``len(devices)``
    contiguous chunks along ``axis`` (the dimension must divide evenly) and
    associates each chunk with the corresponding device. No graph rewriting,
    no collectives, no implicit communication — the caller is responsible for
    moving/using the resulting local tensors.

    Args:
        tensor: The local concrete tensor to split.
        axis: Axis along which to split (normalized, may be negative; must
            be an integer — bools are rejected).
        devices: Target devices (one result tensor each).

    Returns:
        A list of :class:`Tensor`\\s, one per device, each a contiguous chunk.

    Raises:
        DeviceError: If ``devices`` is empty, if ``tensor`` is not a
            :class:`Tensor`, if ``axis`` is not an integer (bools included),
            if ``axis`` is out of range, or if the split fails.
        ShapeError: If ``tensor.shape[axis]`` is not divisible by
            ``len(devices)``.
    """
    if not devices:
        raise DeviceError("split_tensor requires at least one device")
    from .tensor import Tensor  # deferred: avoids a runtime import cycle

    if not isinstance(tensor, Tensor):
        raise DeviceError(f"split_tensor expects a Tensor, got {type(tensor).__name__}")
    ndim = tensor.data.ndim
    # bools are rejected up-front (they are ints, but never a valid axis).
    if isinstance(axis, bool):
        raise _invalid_axis_error(axis)
    try:
        # Non-numeric axes (str, None, complex, multi-element arrays) raise
        # here; numpy arrays can also raise ValueError from ambiguous truth.
        normalized = axis + ndim if axis < 0 else axis
    except (TypeError, ValueError):
        raise _invalid_axis_error(axis) from None
    # Range check comes before the integrality check so a numeric out-of-range
    # axis (e.g. 2.5 on a 2-D tensor) keeps the "out of range" diagnostic.
    if normalized < 0 or normalized >= ndim:
        raise DeviceError(
            f"split_tensor axis {axis} is out of range for a tensor with "
            f"{ndim} dimension(s)"
        )
    # In-range non-integral axes (e.g. 0.5 on a 1-D tensor). numpy integer
    # scalars (np.int32, ...) are numbers.Integral and pass.
    if not isinstance(axis, numbers.Integral):
        raise _invalid_axis_error(axis)
    dim_size = tensor.shape[normalized]
    if dim_size % len(devices) != 0:
        raise ShapeError(
            f"Cannot split tensor of shape {tensor.shape} along axis {axis} "
            f"into {len(devices)} chunks: dimension size {dim_size} is not "
            f"divisible by {len(devices)}"
        )
    # np.split returns views of the input — no copies, just device tagging.
    try:
        chunks = np.split(tensor.data, len(devices), axis=normalized)
    except (TypeError, ValueError, IndexError) as exc:
        raise DeviceError(f"split_tensor failed to split along axis {axis}: {exc}") from exc
    return [_tensor(chunk, device) for chunk, device in zip(chunks, devices)]


def replicate_tensor(tensor: "Tensor", devices: List[Device]) -> List["Tensor"]:
    """Replicate a local tensor onto each device (one tensor per device).

    Pure explicit data preparation: the same underlying data is associated
    with every device (v1: views over the same numpy buffer). No graph
    rewriting, no collectives, no implicit communication.

    Args:
        tensor: The local concrete tensor to replicate.
        devices: Target devices (one result tensor each).

    Returns:
        A list of :class:`Tensor`\\s, one per device, all sharing the data.

    Raises:
        DeviceError: If ``devices`` is empty or ``tensor`` is not a
            :class:`Tensor`.
    """
    if not devices:
        raise DeviceError("replicate_tensor requires at least one device")
    from .tensor import Tensor  # deferred: avoids a runtime import cycle

    if not isinstance(tensor, Tensor):
        raise DeviceError(
            f"replicate_tensor expects a Tensor, got {type(tensor).__name__}"
        )
    # The SAME ndarray object tagged with each device — no copies.
    return [_tensor(tensor.data, device) for device in devices]
