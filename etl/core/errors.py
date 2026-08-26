"""Error hierarchy for etl.

All public errors raised by etl derive from :class:`ETLError` so callers can
catch a single base class. Each subclass identifies a distinct failure domain
(see the root ``CONTEXT.md`` error strategy — binding for all modules).

Convention (binding): error messages include a source location (e.g.
``model.py:83``) whenever a graph location exists. No etl API silently
swallows or works around errors.
"""

from __future__ import annotations

__all__ = [
    "ETLError",
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
]


class ETLError(Exception):
    """Base class for all etl errors."""


class TraceError(ETLError):
    """Raised when a graph-building operation is used outside a valid trace.

    Examples: calling an op function outside ``etl.trace``/``@etl.defn``,
    passing a concrete :class:`~etl.core.Tensor` to an op function (no eager
    mode), calling a ``Defn`` with concrete tensors, or using a
    ``SymbolicTensor`` as a Python boolean / with unregistered operator
    handlers.
    """


class ShapeError(ETLError):
    """Raised on shape/rank mismatches and failed shape inference.

    Includes symbolic-dimension unification failures reported by
    ``Dim``/``DimExpr`` evaluation.
    """


class TransformError(ETLError):
    """Raised by graph→graph transforms (vectorize/vmap/grad/jvp/vjp).

    E.g. axis mismatch, unsupported batching rule, missing derivative rule.
    Transforms must never silently fall back when a rule is missing.
    """


class BackendError(ETLError):
    """Raised when a backend cannot lower/compile/load/execute a program.

    Backend capability limitations (dynamic shapes, collectives, runtime
    calls, custom blocks, unsupported dtypes) fail explicitly.
    """


class PersistenceError(ETLError):
    """Raised on save/load failures: format mismatch, integrity failure,
    backend/device/ABI incompatibility. Loading never silently re-traces or
    recompiles."""


class DeviceError(ETLError):
    """Raised on device errors: unknown device kind/index, device mismatch,
    interop (DLPack) failures, multi-device preparation errors."""


class DTypeError(ETLError):
    """Raised when an object cannot be normalized to a supported dtype, or a
    dtype is unsupported by an operation/backend."""


class VerificationError(ETLError):
    """Raised when IR verification fails (malformed module, op operand/result
    mismatches, violated invariants)."""
