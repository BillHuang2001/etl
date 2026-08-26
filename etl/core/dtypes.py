"""DType constants and the ``dtype`` normalizer.

etl uses numpy dtype objects directly (numpy is the only hard dependency).
The constants below are ``np.dtype`` instances — they are the canonical
representation of a dtype anywhere in the library (``TensorSpec.dtype``,
``SymbolicTensor.dtype``, ``Tensor.dtype`` are always ``np.dtype``).

Naming: ``bool_`` is used instead of ``bool`` to avoid shadowing the builtin
(see also the operator-handler registry in ``symbolic.py`` for the same
pattern).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import DTypeError

__all__ = [
    "dtype",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
    "complex64",
    "complex128",
]

# --- dtype constants (numpy dtype objects) ---

float16: np.dtype = np.dtype("float16")
float32: np.dtype = np.dtype("float32")
float64: np.dtype = np.dtype("float64")
int8: np.dtype = np.dtype("int8")
int16: np.dtype = np.dtype("int16")
int32: np.dtype = np.dtype("int32")
int64: np.dtype = np.dtype("int64")
uint8: np.dtype = np.dtype("uint8")
uint16: np.dtype = np.dtype("uint16")
uint32: np.dtype = np.dtype("uint32")
uint64: np.dtype = np.dtype("uint64")
# `bool_` (not `bool`) — avoids shadowing the builtin `bool`.
bool_: np.dtype = np.dtype("bool")
complex64: np.dtype = np.dtype("complex64")
complex128: np.dtype = np.dtype("complex128")


def dtype(obj: Any) -> np.dtype:
    """Normalize any dtype-like object to a :class:`numpy.dtype`.

    Accepts: ``np.dtype`` (passthrough), dtype name strings (e.g.
    ``"float32"``), scalar types (``np.float32``, ``float``, ``int``,
    ``bool``), and any object exposing a ``dtype`` attribute (numpy arrays,
    ``Tensor``, ``SymbolicTensor``, ``TensorSpec``).

    Args:
        obj: The object to normalize.

    Returns:
        The normalized :class:`numpy.dtype`.

    Raises:
        DTypeError: If ``obj`` cannot be normalized to a numpy dtype.
    """
    if isinstance(obj, np.dtype):
        return obj
    if isinstance(obj, str):
        try:
            return np.dtype(obj)
        except TypeError as exc:
            raise DTypeError(f"Unknown dtype name: {obj!r}") from exc
    if isinstance(obj, type):
        try:
            return np.dtype(obj)
        except TypeError as exc:
            raise DTypeError(f"Cannot interpret type {obj!r} as a dtype") from exc
    if hasattr(obj, "dtype"):
        # Covers numpy arrays, Tensor, SymbolicTensor, TensorSpec (duck-typed —
        # core must not import etl modules).
        return dtype(obj.dtype)
    raise DTypeError(
        f"Cannot convert object of type {type(obj).__name__} ({obj!r}) to a dtype"
    )
