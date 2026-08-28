"""etl.core — the value-model foundation of the EvoX Tensor Library.

This package sits at the bottom of the etl import DAG
(``core ← ir ← ops ← trace ← ...``): it imports **nothing** from etl — only
numpy and the stdlib. Everything here is a pure Python/numpy value-model
type; no IR, ops, tracing, or backend logic lives in core.

Contents:

- **Errors** — ``ETLError`` and all public error subclasses.
- **Dtypes** — ``dtype()`` normalizer + dtype constants (numpy dtype
  objects; ``bool_`` instead of ``bool``).
- **Symbolic shapes** — ``Dim``, ``DimExpr`` (``+ - * // % min max``),
  ``dim()``.
- **Value model** — ``TensorSpec`` (describes a future tensor),
  ``SymbolicTensor`` (SSA graph value, no storage), ``Tensor`` (materialized
  concrete tensor wrapping a numpy ndarray in v1) + concrete creators
  (``tensor``, ``zeros``, ``ones``, ``full``, ``empty``, ``from_numpy``,
  ``from_dlpack``) + ``constant`` (explicit graph embedding).
- **Devices** — ``Device``, ``devices()``, explicit multi-device preparation
  ``split_tensor`` / ``replicate_tensor``.
- **Pytrees** — ``TreeSpec``, ``flatten`` / ``unflatten`` /
  ``register_pytree_node``.

``SymbolicTensor`` operator overloads dispatch through a handler registry
(``register_operator_handlers``) that ``etl.ops`` populates at import time —
this is what keeps the import DAG acyclic.

Status: implementation phase complete — all value-model behaviors are
implemented (no stubs remain).
"""

from .dtypes import (
    bool_,
    complex128,
    complex64,
    dtype,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .dim import Dim, DimExpr, dim
from .device import Device, devices, replicate_tensor, split_tensor
from .errors import (
    BackendError,
    DTypeError,
    DeviceError,
    ETLError,
    PersistenceError,
    ShapeError,
    TraceError,
    TransformError,
    VerificationError,
)
from .spec import TensorSpec
from .symbolic import (
    SymbolicTensor,
    _get_constant_builder,  # internal cross-module contract for ops
    _get_operator_handler,  # internal cross-module contract for ops
    constant,
    register_constant_builder,  # internal cross-module contract for ops
    register_operator_handlers,
)
from .tensor import (
    Tensor,
    empty,
    from_dlpack,
    from_numpy,
    full,
    ones,
    tensor,
    zeros,
)
from .tree import (
    TreeSpec,
    first_mismatch_path,  # internal cross-module contract (trace/pipeline/transforms)
    flatten,
    format_path,  # internal cross-module contract (trace/pipeline/transforms)
    register_pytree_node,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
    unflatten,
)

__all__ = [
    # errors
    "ETLError",
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
    # dtypes
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
    # symbolic shapes
    "Dim",
    "DimExpr",
    "dim",
    # value model
    "TensorSpec",
    "Tensor",
    "SymbolicTensor",
    "constant",
    # concrete creators
    "tensor",
    "zeros",
    "ones",
    "full",
    "empty",
    "from_numpy",
    "from_dlpack",
    # operator-handler hook (populated by etl.ops)
    "register_operator_handlers",
    # devices
    "Device",
    "devices",
    "split_tensor",
    "replicate_tensor",
    # pytrees
    "TreeSpec",
    "flatten",
    "unflatten",
    "register_pytree_node",
]
