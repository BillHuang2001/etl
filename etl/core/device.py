"""Device model and explicit multi-device data-preparation helpers.

A tensor is always a *local* physical tensor. ``Device`` identifies a
physical device; ``devices()`` enumerates what is actually available;
``split_tensor``/``replicate_tensor`` are *explicit* data-preparation
utilities that produce ordinary local ``Tensor``\s — they do no graph
rewriting, no implicit communication, and never shard inside the graph
(collectives are the explicit in-graph communication mechanism, owned by
``etl.dist``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from .errors import DeviceError

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
        The list of :class:`Device`\s.

    Raises:
        DeviceError: If ``kind`` is not a recognized device kind.
    """
    raise NotImplementedError(
        "devices() is not implemented yet (architecture phase); "
        "it will enumerate cpu always + cuda only when a CUDA runtime is detectable."
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
        axis: Axis along which to split (normalized, may be negative).
        devices: Target devices (one result tensor each).

    Returns:
        A list of :class:`Tensor`\s, one per device, each a contiguous chunk.

    Raises:
        DeviceError: If ``devices`` is empty or ``axis`` is out of range.
        ShapeError: If ``tensor.shape[axis]`` is not divisible by
            ``len(devices)``.
    """
    raise NotImplementedError(
        "split_tensor is not implemented yet (architecture phase); "
        "it will be pure data slicing with device tagging (no communication)."
    )


def replicate_tensor(tensor: "Tensor", devices: List[Device]) -> List["Tensor"]:
    """Replicate a local tensor onto each device (one tensor per device).

    Pure explicit data preparation: the same underlying data is associated
    with every device (v1: views over the same numpy buffer). No graph
    rewriting, no collectives, no implicit communication.

    Args:
        tensor: The local concrete tensor to replicate.
        devices: Target devices (one result tensor each).

    Returns:
        A list of :class:`Tensor`\s, one per device, all sharing the data.

    Raises:
        DeviceError: If ``devices`` is empty.
    """
    raise NotImplementedError(
        "replicate_tensor is not implemented yet (architecture phase); "
        "it will tag the same buffer with each device (no copies, no communication)."
    )
