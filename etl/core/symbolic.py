"""SymbolicTensor — SSA values inside a graph — and operator dispatch.

A ``SymbolicTensor`` is a *graph value*: it has NO storage. Its SSA identity
is ``value.id`` (the underlying ``ir.Value``). It must NOT define ``numpy``,
``data_ptr``, ``__dlpack__`` or ``__array__`` — a symbolic tensor can never
be confused with (or implicitly converted to) a concrete ``Tensor`` or numpy
array.

Operator dispatch (key design): ``core`` sits at the bottom of the import DAG
(``core ← ir ← ops ← ...``) and must not import ``ops``. Therefore the
operator dunders of ``SymbolicTensor`` dispatch through a module-level
handler registry that ``etl.ops`` populates at import time
(``core.register_operator_handlers``). If a handler is missing at call time
(e.g. ``etl.ops`` was never imported), a ``TraceError`` is raised with a
clear message. The same hook pattern is used by ``etl.constant``: ``ops``
registers a constant builder so ``core.constant`` can construct a Constant op
without importing ``ops``.

Handler calling convention (binding for ``etl.ops``):
- binary kinds (``add``, ``sub``, ``mul``, ``matmul``, ``truediv``, ``pow``,
  ``lt``, ``gt``, ``le``, ``ge``, ``eq``): ``handler(left, right)``
- unary kind ``neg``: ``handler(operand)``
- ``getitem``: ``handler(obj, key)``
Each returns a ``SymbolicTensor`` (or a valid static value for indexing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from .dim import Dim, DimExpr
from .dtypes import dtype
from .errors import TraceError
from .tensor import Tensor

__all__ = [
    "SymbolicTensor",
    "constant",
    "register_operator_handlers",
    # internal (not in __all__): _get_operator_handler, register_constant_builder
]

# --- operator handler registry (populated by etl.ops at import time) ---

_OPERATOR_HANDLERS: Dict[str, Callable] = {}

# The full set of handler kinds SymbolicTensor may dispatch to. etl.ops must
# register every kind it supports; unsupported kinds stay unregistered so the
# missing-handler TraceError fires honestly.
_OPERATOR_KINDS = (
    "add", "sub", "mul", "matmul", "truediv", "pow",
    "neg", "lt", "gt", "le", "ge", "eq", "getitem",
)

# Constant-op builder hook (registered by etl.ops; see constant() below).
_CONSTANT_BUILDER: Optional[Callable] = None


def register_operator_handlers(kind: str, handler: Callable) -> None:
    """Register the operator handler for ``SymbolicTensor`` dispatch kind ``kind``.

    Called by ``etl.ops`` at import time so ``SymbolicTensor.__add__`` etc.
    build IR ops without ``core`` importing ``ops`` (import acyclicity).
    Re-registering a kind replaces the previous handler.

    Args:
        kind: Dispatch kind — one of ``"add"``, ``"sub"``, ``"mul"``,
            ``"matmul"``, ``"truediv"``, ``"pow"``, ``"neg"``, ``"lt"``,
            ``"gt"``, ``"le"``, ``"ge"``, ``"eq"``, ``"getitem"``.
        handler: Callable following the calling convention documented in the
            module docstring.

    Raises:
        TypeError: If ``handler`` is not callable.
    """
    if not callable(handler):
        raise TypeError(f"Operator handler for kind {kind!r} must be callable")
    _OPERATOR_HANDLERS[kind] = handler


def _get_operator_handler(kind: str) -> Callable:
    """Return the registered handler for ``kind`` or raise a clear TraceError.

    Internal cross-module contract for ``etl.ops`` (and tests).
    """
    handler = _OPERATOR_HANDLERS.get(kind)
    if handler is None:
        raise TraceError(
            f"SymbolicTensor operator kind {kind!r} has no registered handler. "
            "etl.ops registers SymbolicTensor operator handlers at import time — "
            "import etl.ops (e.g. `import etl` which imports everything, or "
            "`import etl.ops`) before operating on symbolic tensors."
        )
    return handler


def register_constant_builder(fn: Callable[[Tensor], SymbolicTensor]) -> None:
    """Register the builder used by ``core.constant`` to construct Constant ops.

    Internal cross-module contract: ``etl.ops`` (which owns op construction)
    registers its constant-op builder here at import time, so
    ``etl.constant`` can live in ``core`` without an import cycle.

    Args:
        fn: A callable taking a concrete :class:`Tensor` and returning a
            :class:`SymbolicTensor` for the Constant op (snapshotting data,
            warning above ``ETL_LARGE_CONSTANT_BYTES``).

    Raises:
        TypeError: If ``fn`` is not callable.
    """
    global _CONSTANT_BUILDER
    if not callable(fn):
        raise TypeError("constant builder must be callable")
    _CONSTANT_BUILDER = fn


def _get_constant_builder() -> Callable[[Tensor], SymbolicTensor]:
    """Return the registered constant builder or raise a clear TraceError."""
    if _CONSTANT_BUILDER is None:
        raise TraceError(
            "etl.constant requires the constant-op builder that etl.ops registers "
            "at import time. Import etl.ops before using etl.constant."
        )
    return _CONSTANT_BUILDER


@dataclass(frozen=True, eq=False)
class SymbolicTensor:
    """An SSA value inside a graph — no storage, graph semantics only.

    Attributes:
        value: The underlying ``ir.Value`` (duck-typed protocol: must expose
            an ``id`` attribute; ``SymbolicTensor`` must not import ``ir``).
        dtype: The element dtype (normalized to :class:`numpy.dtype`).
        shape: Tuple of ``DimExpr``/``int`` entries (``Dim`` entries are
            accepted and should be resolved to ``DimExpr`` by callers);
            symbolic shape of the value.
        location: Optional source location (duck-typed ``ir.Location`` or a
            string); used in error messages.

    Deliberately NOT defined: ``numpy``, ``data_ptr``, ``__dlpack__``,
    ``__array__`` — a symbolic tensor must never be mistaken for concrete
    data. ``__bool__`` raises :class:`TraceError` (runtime control flow must
    use ``etl.cond``/``etl.while_loop``/``etl.scan``). Unhashable
    (``__hash__ = None``): ``__eq__`` builds an IR ``equal`` op, not a Python
    bool, so symbolic tensors cannot safely be dict keys.
    """

    value: Any
    dtype: np.dtype
    shape: Tuple[Union[DimExpr, int], ...]
    location: Any = None

    # Unhashable: __eq__ returns a SymbolicTensor (an IR op), not a bool.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not hasattr(self.value, "id"):
            raise TypeError(
                "SymbolicTensor.value must expose an 'id' attribute "
                "(ir.Value protocol), got "
                f"{type(self.value).__name__ if self.value is not None else None}"
            )
        object.__setattr__(self, "dtype", dtype(self.dtype))
        shape = tuple(self.shape)
        for entry in shape:
            if entry is None:
                continue  # runtime-dynamic dim (matches TensorSpec/ValueType)
            if not isinstance(entry, (Dim, DimExpr, int)):
                raise TypeError(
                    "SymbolicTensor.shape entries must be Dim | DimExpr | int "
                    f"| None, got {entry!r}"
                )
        object.__setattr__(self, "shape", shape)

    @property
    def id(self) -> Any:
        """The SSA identity of this value (``value.id``)."""
        return self.value.id

    # --- operator dispatch (all route through the ops-registered handlers) ---

    def __add__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("add")(self, other)

    def __radd__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("add")(other, self)

    def __sub__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("sub")(self, other)

    def __rsub__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("sub")(other, self)

    def __mul__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("mul")(self, other)

    def __rmul__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("mul")(other, self)

    def __matmul__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("matmul")(self, other)

    def __rmatmul__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("matmul")(other, self)

    def __truediv__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("truediv")(self, other)

    def __rtruediv__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("truediv")(other, self)

    def __pow__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("pow")(self, other)

    def __rpow__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("pow")(other, self)

    def __neg__(self) -> "SymbolicTensor":
        return _get_operator_handler("neg")(self)

    def __lt__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("lt")(self, other)

    def __gt__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("gt")(self, other)

    def __le__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("le")(self, other)

    def __ge__(self, other: Any) -> "SymbolicTensor":
        return _get_operator_handler("ge")(self, other)

    def __eq__(self, other: Any) -> "SymbolicTensor":  # type: ignore[override]
        # Graph semantics: builds an elementwise `equal` op (a boolean
        # symbolic tensor), NOT a Python bool. Python-level identity checks
        # must use `is`.
        return _get_operator_handler("eq")(self, other)

    def __getitem__(self, key: Any) -> "SymbolicTensor":
        return _get_operator_handler("getitem")(self, key)

    def __bool__(self) -> bool:
        # Principle 4: `if etl.sum(x) > 0:` must fail clearly.
        where = f" (at {self.location})" if self.location is not None else ""
        raise TraceError(
            f"SymbolicTensor cannot be used as a Python boolean{where}. "
            "Runtime tensor control flow must use etl.cond / etl.while_loop / "
            "etl.scan; static Python control flow is only valid over Python values."
        )

    def __repr__(self) -> str:
        return f"SymbolicTensor(id={self.id!r}, dtype={self.dtype}, shape={self.shape})"


def constant(tensor: Tensor) -> SymbolicTensor:
    """Embed a concrete tensor's data into the graph as a Constant op.

    Graph-time only; the *only* (explicit) way to embed tensor data into a
    graph (closure capture is an error). The concrete-computation path is
    delegated to the constant-op builder registered by ``etl.ops``, which
    snapshots the data and warns above the ``ETL_LARGE_CONSTANT_BYTES``
    threshold.

    Args:
        tensor: The concrete tensor whose data is snapshotted.

    Returns:
        A :class:`SymbolicTensor` for the Constant op.

    Raises:
        TraceError: If the constant builder is not registered (``etl.ops``
            not imported) or no trace is active.
    """
    return _get_constant_builder()(tensor)
