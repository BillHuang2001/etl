"""Value types: dtype + (symbolic) shape for every SSA value.

``ValueType`` is the IR-level counterpart of the trace-level ``TensorSpec``
(owned by ``etl.core``): a ``TensorSpec`` describes a *future* runtime tensor
(with optional device and name); a ``ValueType`` types an SSA *value* inside a
function and has no device, name, or storage. Shapes are tuples of:

* ``int`` — statically known dimension,
* ``Dim`` / ``DimExpr`` (from ``etl.core``) — symbolic dimension (e.g. batch
  size ``B``, ``B * 2``),
* ``None`` — runtime-dynamic dimension (unchecked; only allowed where the
  backend declares ``dynamic_shapes`` capability).

Rank is always known at trace time. Shape arithmetic over symbolic dims uses
``core.DimExpr`` (owned by ``etl.core``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from etl.core import DimExpr  # symbolic shape components (owned by core)

#: One dimension of a shape: static int, symbolic DimExpr, or None (dynamic).
ShapeDim = Union[int, DimExpr, None]

#: A shape: tuple of dims; length (rank) is always known at trace time.
Shape = tuple[ShapeDim, ...]

_DTYPE_ABBREV = {
    "bool": "i1",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "complex64": "c64",
    "complex128": "c128",
}


@dataclass(frozen=True)
class ValueType:
    """The type of an SSA value: a dtype and a symbolic shape.

    Attributes:
        dtype: A numpy dtype (etl dtypes ARE numpy dtypes; see ``etl.core``).
        shape: Tuple of ``int | DimExpr | None`` dims (see module docstring).
    """

    dtype: np.dtype
    shape: Shape

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    def rank(self) -> int:
        """Number of dimensions (always known at trace time)."""
        return len(self.shape)

    def __str__(self) -> str:
        dims = "x".join("?" if d is None else str(d) for d in self.shape)
        return f"tensor<{dims}x{_dtype_str(self.dtype)}>"

    def __repr__(self) -> str:
        return f"ValueType(dtype=np.dtype('{self.dtype.name}'), shape={self.shape!r})"


def _dtype_str(dtype: np.dtype) -> str:
    """Compact dtype spelling for printing, e.g. float32 -> f32."""
    return _DTYPE_ABBREV.get(dtype.name, dtype.name)
