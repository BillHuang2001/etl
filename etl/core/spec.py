"""TensorSpec — describes a future runtime tensor.

Lifecycle of a spec: ``TensorSpec → trace → SymbolicTensor → run → Tensor``.
A spec has no storage; it carries a shape (symbolic dims allowed), a dtype,
and optionally a device and a name. ``None`` entries in ``shape`` mean
runtime-dynamic (unchecked at trace time); rank is always known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

from .device import Device
from .dim import Dim, DimExpr
from .dtypes import dtype

__all__ = ["TensorSpec"]

# A single shape entry: symbolic dim, sub-expression, concrete int, or None
# (runtime-dynamic).
_ShapeEntry = Union[Dim, DimExpr, int, None]


@dataclass(frozen=True)
class TensorSpec:
    """Describes a future runtime tensor (no storage).

    Attributes:
        shape: Tuple of ``Dim | DimExpr | int | None``. ``None`` = runtime
            dynamic (unchecked). Rank is always known. Normalized to a tuple
            on construction.
        dtype: The tensor dtype; normalized to a :class:`numpy.dtype` on
            construction (accepts anything ``etl.dtype`` accepts).
        device: Optional :class:`Device` the tensor is expected on
            (``None`` = unspecified/any).
        name: Optional name for the tensor (debugging, signatures, binding).
    """

    shape: Tuple[_ShapeEntry, ...]
    dtype: np.dtype
    device: Optional[Device] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        for entry in shape:
            if not (isinstance(entry, (Dim, DimExpr, int)) or entry is None):
                raise TypeError(
                    "TensorSpec.shape entries must be Dim | DimExpr | int | None, "
                    f"got {entry!r} (rank must be a tuple)"
                )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype(self.dtype))

    @property
    def rank(self) -> int:
        """The (always known) rank of the described tensor."""
        return len(self.shape)

    def __repr__(self) -> str:
        parts = [f"shape={self.shape}", f"dtype={self.dtype}"]
        if self.device is not None:
            parts.append(f"device={self.device!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"TensorSpec({', '.join(parts)})"
