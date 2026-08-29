"""Internal helpers shared by the sparse frontend (``etl.sparse.ops``) and
the sparse transform rules (``etl.sparse.rules``).

NOT part of the public API — nothing outside ``etl.sparse`` may import from
this module. The helpers mirror the ``etl.dist`` frontend discipline
(``etl/dist/_op_utils.py``): location capture with etl-package frame skip,
operand normalization with the mandated three-option ``TraceError``, and
result wrapping that reads dtype/shape from the IR ``ValueType``.

Import layering (binding, root CONTEXT.md): this module imports ``etl.core``,
``etl.ir``, ``etl.ops``, and ``etl.trace`` (active-builder hook) only — never
``etl.backends`` / ``etl.pipeline`` / ``etl.persist`` (``import etl`` must
stay clean). ``etl.ops`` / ``etl.trace`` are loaded before ``etl.sparse`` in
``etl/__init__.py`` and import nothing from ``etl.sparse``, so the module
level imports below cannot cycle.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

import numpy as np

from etl import core
from etl import ir
from etl import ops
from etl.trace import current_builder

from etl.sparse.value import is_sparse

__all__ = [
    "_get_location",
    "_require_symbolic_sparse",
    "_require_symbolic_dense",
    "_wrap_dense",
    "_raw_reshape",
    "_row_lookup",
]

#: Absolute path of the ``etl`` package directory (frames inside it are
#: skipped during location capture).
_ETL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_location(depth: int = 1) -> "ir.Location":
    """Capture the Python call site of the *user* sparse op call.

    Walks ``inspect.stack()`` starting ``depth`` frames up the call stack,
    skipping every frame whose filename contains the ``etl`` package
    directory (so internal helper frames never pollute user locations), and
    returns ``ir.Location(file, line, 0)`` for the first external frame.

    Returns:
        An ``ir.Location`` carrying the external call site, or
        ``ir.Location.unknown()`` when capture is impossible.

    Raises:
        Never raises — location capture failure degrades to
        ``ir.Location.unknown()`` (a missing location must not break
        tracing).
    """
    try:
        for frame in inspect.stack()[depth:]:
            if _ETL_DIR in frame.filename:
                continue
            return ir.Location(frame.filename, frame.lineno, 0)
    except Exception:  # pragma: no cover - capture must never break tracing
        pass
    return ir.Location.unknown()


def _concrete_shape(x: Any):
    """Shape description of a concrete value: ``dense_shape`` for a sparse
    tensor, ``shape`` for a dense tensor/array."""
    if is_sparse(x):
        return x.dense_shape
    return getattr(x, "shape", None)


def _require_symbolic_sparse(x: Any):
    """Normalize a sparse operand into a SYMBOLIC sparse tensor.

    - A symbolic sparse tensor (``is_sparse(x)`` with a
      ``core.SymbolicTensor`` values leaf) is returned unchanged.
    - A CONCRETE ``SparseTensor``, ``core.Tensor``, or numpy ``ndarray``
      raises ``core.TraceError`` with the mandated three-option message
      (mirroring ``etl.dist``): no eager mode — make the tensor an explicit
      input, or embed it explicitly via ``etl.constant(...)``, or use
      ``etl.evaluate(...)``.
    - Anything else (including a spec-phase ``SparseTensorSpec``) raises
      ``TypeError`` naming the accepted operand kinds.

    Raises:
        core.TraceError: ``x`` is a concrete value (no eager mode).
        TypeError: ``x`` is not a sparse tensor in any phase.
    """
    if is_sparse(x) and isinstance(x.values, core.SymbolicTensor):
        return x
    if is_sparse(x):
        if isinstance(x.values, (core.Tensor, np.ndarray)):
            raise core.TraceError(
                f"sparse ops require SymbolicTensor graph operands, got a "
                f"concrete {type(x).__name__} (shape={_concrete_shape(x)}, "
                f"dtype={getattr(x, 'dtype', None)}). There is no eager mode: "
                "make the tensor an explicit input (trace with a TensorSpec), "
                "or embed it explicitly via etl.constant(...) (snapshot "
                "semantics), or use etl.evaluate(...) to build and run a "
                "graph."
            )
        raise TypeError(
            "sparse operands must be symbolic SparseTensor graph values "
            "(COO/CSR/CSC with SymbolicTensor leaves) or concrete "
            f"SparseTensor instances, got {type(x).__name__}"
        )
    if isinstance(x, (core.Tensor, np.ndarray)):
        raise core.TraceError(
            f"sparse ops require SymbolicTensor graph operands, got a "
            f"concrete {type(x).__name__} (shape={_concrete_shape(x)}, "
            f"dtype={getattr(x, 'dtype', None)}). There is no eager mode: "
            "make the tensor an explicit input (trace with a TensorSpec), or "
            "embed it explicitly via etl.constant(...) (snapshot semantics), "
            "or use etl.evaluate(...) to build and run a graph."
        )
    raise TypeError(
        "sparse operands must be symbolic SparseTensor graph values (COO/"
        f"CSR/CSC with SymbolicTensor leaves), got {type(x).__name__}"
    )


def _require_symbolic_dense(x: Any) -> "core.SymbolicTensor":
    """Normalize a DENSE operand (the dense side of the sparse ops) into a
    ``core.SymbolicTensor``.

    - ``core.SymbolicTensor``: returned unchanged.
    - ``core.Tensor`` / numpy ``ndarray``: raises ``core.TraceError`` with
      the mandated three-option message (no eager mode).
    - Anything else: raises ``TypeError`` naming the accepted operand kinds.

    Raises:
        core.TraceError: ``x`` is a concrete ``Tensor`` (no eager mode).
        TypeError: ``x`` is not a SymbolicTensor / Tensor.
    """
    if isinstance(x, core.SymbolicTensor):
        return x
    if isinstance(x, (core.Tensor, np.ndarray)):
        raise core.TraceError(
            f"sparse ops require SymbolicTensor graph operands, got a "
            f"concrete {type(x).__name__} (shape={getattr(x, 'shape', None)}, "
            f"dtype={getattr(x, 'dtype', None)}). There is no eager mode: "
            "make the tensor an explicit input (trace with a TensorSpec), or "
            "embed it explicitly via etl.constant(...) (snapshot semantics), "
            "or use etl.evaluate(...) to build and run a graph."
        )
    raise TypeError(
        f"dense operands must be SymbolicTensor graph values, got "
        f"{type(x).__name__}"
    )


def _wrap_dense(value: Any, location: Any) -> "core.SymbolicTensor":
    """Wrap an op's result ``ir.Value`` in a ``SymbolicTensor`` facade.

    Dtype and shape are read from ``value.type`` (the op's registered
    ``shape_fn`` already applied the sparse shape rules — this module never
    recomputes them). ``None`` shape entries (runtime-dynamic ``nnz`` dims)
    are sanitized to 0 for construction and then restored directly, so
    ``result.shape`` always matches the result's ``ValueType`` (exactly like
    ``etl.dist._op_utils._wrap_result``).
    """
    value_type = value.type
    shape = value_type.shape
    sanitized = tuple(0 if dim is None else dim for dim in shape)
    result = core.SymbolicTensor(
        value=value,
        dtype=value_type.dtype,
        shape=sanitized,
        location=location,
    )
    if any(dim is None for dim in shape):
        object.__setattr__(result, "shape", tuple(shape))
    return result


def _raw_reshape(value: Any, shape: tuple, location: Any = None) -> "core.SymbolicTensor":
    """Build a raw ``reshape`` op into the active builder (dynamic-dim safe).

    The frontend ``ops.reshape`` rejects target shapes that resolve to a
    runtime-dynamic dim — a ``-1`` wildcard over a dynamic element count
    (``SymbolicTensor`` cannot carry the inferred ``None`` dim) — but the IR
    ``reshape`` op and the numpy kernel handle ``-1`` natively (inferred from
    the element count at run time; the inferred dim stays ``None`` in the
    result type). The sparse transform rules need exactly this to expand /
    contract the runtime-dynamic ``nnz`` dim (e.g. ``(nnz, ndim)`` →
    ``(nnz, 1, ndim)`` or ``(prod,)``).

    Args:
        value: An ``ir.Value`` (not a ``SymbolicTensor``).
        shape: Target shape; may contain a single ``-1`` wildcard and/or
            ``None`` entries (the latter just pass through the inferred
            result type).

    Returns:
        A ``SymbolicTensor`` wrapping the reshape result (``None`` dims
        preserved in ``.shape``, like :func:`_wrap_dense`).
    """
    op = current_builder().create(
        "reshape", operands=(value,), attributes={"shape": tuple(shape)},
        location=location,
    )
    return _wrap_dense(op.result, location)


def _row_lookup(
    input_indices: Any,
    merged_indices: Any,
    ndim: int,
    location: Any = None,
) -> tuple:
    """For each row of ``input_indices``, its row position in
    ``merged_indices``, plus a validity mask.

    Used by the sparse vjp rules to pull a merged-values cotangent back to
    each merge operand's values: ``sparse_add`` (union merge) and
    ``sparse_multiply`` (intersection merge) produce merged indices that
    contain every *surviving* input row exactly once, and
    ``sparse_transpose`` reorders rows 1:1 — in all three cases each input
    row's contribution to the merged cotangent is the merged cotangent entry
    at the input row's position in the merged indices (zero where the row is
    absent from the merge, i.e. ``sparse_multiply``).

    The lookup is built from ordinary ``etl.ops`` (all dynamic-dim safe):

    - expand both index tensors to rank 3 via raw ``-1``-wildcard reshapes:
      ``(nnz_in, 1, ndim)`` vs ``(1, nnz_m, ndim)``;
    - elementwise ``equal`` (broadcast) -> ``(nnz_in, nnz_m, ndim)`` bool;
    - ``reduce_all`` over the coord axis = ``reduce_sum(cast(bool, int64),
      axes=(-1,)) == ndim`` -> ``(nnz_in, nnz_m)`` bool row-match matrix;
    - ``argmax`` over the merged axis -> the position (0 when absent);
    - the mask = ``reduce_sum(match, axes=(1,)) > 0`` (the ``where`` arm).

    O(nnz_in * nnz_m) in space — acceptable for the vjp rules (documented).

    Args:
        input_indices: The input sparse indices ``ir.Value`` ``(nnz, ndim)``.
        merged_indices: The merged sparse indices ``ir.Value``
            ``(nnz_m, ndim)``.
        ndim: The (static) coordinate count.
        location: Optional ``ir.Location`` for the emitted ops.

    Returns:
        ``(positions, mask)`` — two ``SymbolicTensor``s of shape ``(nnz,)``:
        ``positions`` is int64 (the merged row of each input row, 0 when
        absent), ``mask`` is bool (True where the input row exists in the
        merged indices).
    """
    a = _raw_reshape(input_indices, (-1, 1, ndim), location)
    b = _raw_reshape(merged_indices, (1, -1, ndim), location)
    eq = ops.equal(a, b)  # (nnz_in, nnz_m, ndim) bool
    match_int = ops.reduce_sum(ops.cast(eq, np.dtype("int64")), axes=(-1,))
    match = ops.equal(match_int, ndim)  # (nnz_in, nnz_m) bool
    positions = ops.argmax(match, axis=1)  # (nnz_in,) int64
    mask = ops.greater(ops.reduce_sum(match_int, axes=(1,)), 0)  # (nnz_in,) bool
    return positions, mask
