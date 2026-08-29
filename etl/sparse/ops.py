"""Sparse frontend ops — graph-time sparse tensor operations.

A sparse value in IR is an ``(indices, values)`` pair: ``indices`` int64
``(B..., nnz, ndim)``, ``values`` ``(B..., nnz)``; the unbatched dense shape
lives in every sparse op's ``dense_shape`` attribute and the value dtype in
``dtype`` (both declared on ALL 16 sparse op defs, see
``etl/ir/op_defs/sparse.py``). Batch dims propagate from the operand types;
``nnz`` is runtime-dynamic.

**COO is the computation format (binding):** every CSR/CSC SYMBOLIC input to
a computation op (and to ``to_dense``) is first converted by emitting
``sparse_csr_to_coo`` / ``sparse_csc_to_coo`` — the converted COO feeds the
computation. The csr/csc conversions are RANK-2 ONLY (``core.ShapeError`` at
trace time for rank != 2). COO inputs pass through untouched.

Every op follows the frontend discipline (mirroring ``etl.dist``):
``current_builder()`` first (``TraceError`` outside a trace), location
capture, operand normalization (symbolic sparse -> COO via auto-conversion;
concrete sparse/dense -> the three-option ``TraceError``), static checks
first (``dense_shape`` equality -> ``core.ShapeError``; dtype equality ->
``core.DTypeError``), ``builder.create(op_name, operands=...,
attributes=..., location=...)``, then result wrapping: dtype/shape read from
the result ``ValueType`` (sparse results via ``SparseTensor.from_parts`` —
the result ``dense_shape`` is known at trace time).

The creators (``coo`` / ``csr`` / ``csc`` / ``from_dense``) are POLYMORPHIC:
concrete components (numpy arrays / ``core.Tensor``) build the validated
eager value; symbolic components (``core.SymbolicTensor``) assemble the
in-graph sparse value — ``from_dense`` emits the ``sparse_from_dense`` op,
``coo``/``csr``/``csc`` wrap the symbolic leaves via ``from_parts`` with NO
validation (the numpy kernels validate canonical form at run time). The
converters (``to_dense`` / ``to_csr`` / ``to_csc`` / ``to_coo``) are
polymorphic too: a concrete instance dispatches to its eager ``value.py``
method; a symbolic instance builds the graph op.

Import contract (binding): this module imports ``etl.core``, ``etl.trace``
(active-builder hook), and the sibling ``etl.sparse`` modules only — never
``etl.backends`` / ``etl.pipeline`` / ``etl.persist``. ``etl.ops`` is
imported lazily inside ``matmul`` (dense @ dense path) to keep the import
graph acyclic.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from etl import core
from etl.trace import current_builder

from etl.sparse._utils import (
    _get_location,
    _require_symbolic_dense,
    _require_symbolic_sparse,
    _wrap_dense,
)
from etl.sparse.value import CSCTensor, CSRTensor, SparseTensor, is_sparse

__all__ = [
    "coo",
    "csr",
    "csc",
    "from_dense",
    "to_dense",
    "to_csr",
    "to_csc",
    "to_coo",
    "add",
    "subtract",
    "multiply",
    "multiply_dense",
    "negate",
    "sum",
    "reduce_sum",
    "transpose",
    "reshape",
    "concatenate",
    "matmul",
]


# --- internal helpers --------------------------------------------------------


def _sparse_attrs(x: Any) -> dict:
    """The canonical sparse attribute pair: dense_shape + dtype (name)."""
    return {"dense_shape": tuple(x.dense_shape), "dtype": x.dtype.name}


def _coo_result(op: Any, dense_shape: Tuple[Any, ...], location: Any) -> "SparseTensor":
    """Wrap a 2-result sparse op (indices, values) into a symbolic COO."""
    indices, values = op.results
    return SparseTensor.from_parts(
        _wrap_dense(indices, location),
        _wrap_dense(values, location),
        dense_shape=dense_shape,
        format="coo",
    )


def _csr_result(op: Any, dense_shape: Tuple[Any, ...], location: Any) -> "CSRTensor":
    """Wrap a 3-result coo->csr op (indptr, indices, values) into a CSR."""
    indptr, indices, values = op.results
    return CSRTensor.from_parts(
        _wrap_dense(indptr, location),
        _wrap_dense(indices, location),
        _wrap_dense(values, location),
        dense_shape=dense_shape,
        format="csr",
    )


def _csc_result(op: Any, dense_shape: Tuple[Any, ...], location: Any) -> "CSCTensor":
    """Wrap a 3-result coo->csc op (indptr, indices, values) into a CSC."""
    indptr, indices, values = op.results
    return CSCTensor.from_parts(
        _wrap_dense(indptr, location),
        _wrap_dense(indices, location),
        _wrap_dense(values, location),
        dense_shape=dense_shape,
        format="csc",
    )


def _as_coo(x: Any):
    """Auto-convert a symbolic sparse operand to COO — the computation format.

    ``"coo"`` -> ``x`` unchanged; ``"csr"`` -> emit ``sparse_csr_to_coo``;
    ``"csc"`` -> emit ``sparse_csc_to_coo``. Both conversions are RANK-2 ONLY
    (``core.ShapeError`` at trace time for rank != 2). The result dense_shape
    equals ``x.dense_shape``.
    """
    fmt = x.format
    if fmt == "coo":
        return x
    if x.ndim != 2:
        raise core.ShapeError(
            f"{fmt.upper()} to COO conversion requires a rank-2 sparse tensor, "
            f"got rank {x.ndim} (dense_shape={x.dense_shape})"
        )
    builder = current_builder()
    loc = _get_location()
    op_name = "sparse_csr_to_coo" if fmt == "csr" else "sparse_csc_to_coo"
    op = builder.create(
        op_name,
        operands=(x.indptr.value, x.indices.value, x.values.value),
        attributes=_sparse_attrs(x),
        location=loc,
    )
    return _coo_result(op, x.dense_shape, loc)


def _normalize_axes(seq: Sequence[int], rank: int, what: str) -> Tuple[int, ...]:
    """Normalize axes into a sorted, deduplicated tuple of non-negative ints.

    Negative values wrap Python-style (add ``rank``).

    Raises:
        core.ShapeError: an axis is out of ``range(-rank, rank)``.
    """
    normalized = []
    for axis in seq:
        original = axis
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            raise core.ShapeError(
                f"{what}: axis {original} out of range for rank {rank}"
            )
        normalized.append(int(axis))
    return tuple(sorted(set(normalized)))


# --- creators (polymorphic: concrete -> eager value; symbolic -> graph) ------


def _creator_phase(leaves: Tuple[Any, ...], what: str) -> str:
    """Classify creator components into a single phase.

    Returns:
        ``"symbolic"`` when every leaf is a ``core.SymbolicTensor``
        (graph-time assembly via ``from_parts`` — no validation), or
        ``"concrete"`` when every leaf is a numpy ``ndarray`` /
        ``core.Tensor`` (eager validated construction).

    Raises:
        TypeError: Mixed symbolic/concrete leaves, or ``core.TensorSpec``
            components (directing to ``SparseTensorSpec`` for trace inputs).
    """
    if all(isinstance(leaf, core.SymbolicTensor) for leaf in leaves):
        return "symbolic"
    if all(isinstance(leaf, (np.ndarray, core.Tensor)) for leaf in leaves):
        return "concrete"
    for leaf in leaves:
        if isinstance(leaf, core.TensorSpec):
            raise TypeError(
                f"etl.sparse.{what}: creator components must be concrete "
                f"(numpy arrays / core.Tensor) or symbolic (SymbolicTensor), "
                f"got a core.TensorSpec — for trace inputs use a "
                f"SparseTensorSpec"
            )
    kinds = ", ".join(type(leaf).__name__ for leaf in leaves)
    raise TypeError(
        f"etl.sparse.{what}: creator components must be ALL concrete (numpy "
        f"arrays / core.Tensor) or ALL symbolic (SymbolicTensor), got mixed "
        f"kinds: {kinds}"
    )


def coo(indices, values, shape):
    """Construct a COO sparse tensor from raw components.

    Concrete components (numpy arrays / ``core.Tensor``) -> eager
    construction with canonical-form validation (lex-sorted unique in-range
    rows). Symbolic components (``core.SymbolicTensor``) -> in-graph assembly
    via ``SparseTensor.from_parts`` (no validation — the numpy kernels
    validate canonical form at run time).
    """
    if _creator_phase((indices, values), "coo") == "symbolic":
        return SparseTensor.from_parts(
            indices, values, dense_shape=tuple(shape), format="coo"
        )
    return SparseTensor(indices, values, tuple(shape), format="coo")


def csr(indptr, indices, values, shape):
    """Construct a CSR sparse tensor from raw components.

    Concrete components (numpy arrays / ``core.Tensor``) -> eager
    construction (rank-2) with canonical CSR validation. Symbolic components
    (``core.SymbolicTensor``) -> in-graph assembly via
    ``CSRTensor.from_parts`` (no validation — the numpy kernels validate
    canonical form at run time).
    """
    if _creator_phase((indptr, indices, values), "csr") == "symbolic":
        return CSRTensor.from_parts(
            indptr, indices, values, dense_shape=tuple(shape), format="csr"
        )
    return CSRTensor(indptr, indices, values, tuple(shape))


def csc(indptr, indices, values, shape):
    """Construct a CSC sparse tensor from raw components.

    Concrete components (numpy arrays / ``core.Tensor``) -> eager
    construction (rank-2) with canonical CSC validation. Symbolic components
    (``core.SymbolicTensor``) -> in-graph assembly via
    ``CSCTensor.from_parts`` (no validation — the numpy kernels validate
    canonical form at run time).
    """
    if _creator_phase((indptr, indices, values), "csc") == "symbolic":
        return CSCTensor.from_parts(
            indptr, indices, values, dense_shape=tuple(shape), format="csc"
        )
    return CSCTensor(indptr, indices, values, tuple(shape))


def from_dense(dense, format: str = "coo"):
    """Extract the exact non-zero entries of a dense tensor into a COO sparse.

    ``dense`` may be a numpy ``ndarray`` or a concrete ``core.Tensor`` (its
    ``.numpy()`` is used) -> eager extraction: int64 ``indices (nnz, ndim)``
    from ``np.nonzero`` and ``values`` in the dense array's dtype. A
    ``core.SymbolicTensor`` -> the ``sparse_from_dense`` graph op (the
    in-graph creator; ``nnz`` is runtime-dynamic, batch dims propagate from
    the dense operand). v1 supports ``format="coo"`` only.

    Raises:
        ValueError: ``format`` is not ``"coo"`` (v1 restriction).
        TypeError: ``dense`` is neither an array / ``core.Tensor`` nor a
            ``core.SymbolicTensor``.
    """
    if format != "coo":
        raise ValueError(
            f"etl.sparse.from_dense supports only format='coo' in v1, got "
            f"{format!r}"
        )
    if isinstance(dense, core.SymbolicTensor):
        builder = current_builder()
        loc = _get_location()
        op = builder.create(
            "sparse_from_dense",
            operands=(dense.value,),
            attributes={
                "dense_shape": tuple(dense.shape),
                "dtype": dense.dtype.name,
            },
            location=loc,
        )
        return _coo_result(op, tuple(dense.shape), loc)
    arr = dense.numpy() if isinstance(dense, core.Tensor) else dense
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"etl.sparse.from_dense expects a numpy array or core.Tensor, got "
            f"{type(dense).__name__}"
        )
    nonzero = np.nonzero(arr)
    indices = np.stack(nonzero, axis=1).astype(np.int64)
    values = arr[nonzero]
    return SparseTensor(indices, values, tuple(arr.shape), format="coo")


# --- converters (polymorphic: concrete -> eager method; symbolic -> op) ------


def _require_sparse(x: Any, what: str):
    """Require ``x`` to be a sparse tensor in any phase; return it."""
    if not is_sparse(x):
        raise TypeError(
            f"etl.sparse.{what} expects a sparse tensor (COO/CSR/CSC), got "
            f"{type(x).__name__}"
        )
    return x


def _require_value_phase(x: Any, what: str):
    """Reject spec-phase sparse tensors with a clear message.

    Returns True when ``x`` is symbolic (graph op path), False when concrete
    (eager method path).
    """
    if isinstance(x.values, core.SymbolicTensor):
        return True
    if isinstance(x.values, core.TensorSpec):
        raise TypeError(
            f"etl.sparse.{what}: a SparseTensorSpec describes a future runtime "
            "input and cannot be materialized eagerly — trace a graph with "
            "this spec (etl.trace) and run it to obtain the converted value"
        )
    return False


def to_dense(x):
    """Materialize the dense tensor described by a sparse tensor.

    Concrete instance -> eager ``x.to_dense()`` (numpy array); symbolic
    instance -> convert to COO (``sparse_csr_to_coo``/``sparse_csc_to_coo``
    when needed) and emit ``sparse_to_dense``, returning a dense
    ``SymbolicTensor`` (zeros elsewhere).
    """
    x = _require_sparse(x, "to_dense")
    if _require_value_phase(x, "to_dense"):
        builder = current_builder()
        loc = _get_location()
        coo_x = _as_coo(x)
        op = builder.create(
            "sparse_to_dense",
            operands=(coo_x.indices.value, coo_x.values.value),
            attributes=_sparse_attrs(coo_x),
            location=loc,
        )
        return _wrap_dense(op.results[0], loc)
    return x.to_dense()


def to_csr(x):
    """Convert a sparse tensor to CSR.

    Concrete instance -> eager ``x.to_csr()``; symbolic instance: ``"csr"``
    is returned unchanged, ``"coo"`` emits ``sparse_coo_to_csr`` (rank-2),
    ``"csc"`` goes through COO first.
    """
    x = _require_sparse(x, "to_csr")
    if not _require_value_phase(x, "to_csr"):
        return x.to_csr()
    if x.format == "csr":
        return x
    builder = current_builder()
    loc = _get_location()
    coo_x = _as_coo(x)
    if coo_x.ndim != 2:
        raise core.ShapeError(
            f"sparse to_csr: CSR conversion requires a rank-2 sparse tensor, "
            f"got rank {coo_x.ndim} (dense_shape={coo_x.dense_shape})"
        )
    op = builder.create(
        "sparse_coo_to_csr",
        operands=(coo_x.indices.value, coo_x.values.value),
        attributes=_sparse_attrs(coo_x),
        location=loc,
    )
    return _csr_result(op, coo_x.dense_shape, loc)


def to_csc(x):
    """Convert a sparse tensor to CSC.

    Concrete instance -> eager ``x.to_csc()``; symbolic instance: ``"csc"``
    is returned unchanged, ``"coo"`` emits ``sparse_coo_to_csc`` (rank-2),
    ``"csr"`` goes through COO first.
    """
    x = _require_sparse(x, "to_csc")
    if not _require_value_phase(x, "to_csc"):
        return x.to_csc()
    if x.format == "csc":
        return x
    builder = current_builder()
    loc = _get_location()
    coo_x = _as_coo(x)
    if coo_x.ndim != 2:
        raise core.ShapeError(
            f"sparse to_csc: CSC conversion requires a rank-2 sparse tensor, "
            f"got rank {coo_x.ndim} (dense_shape={coo_x.dense_shape})"
        )
    op = builder.create(
        "sparse_coo_to_csc",
        operands=(coo_x.indices.value, coo_x.values.value),
        attributes=_sparse_attrs(coo_x),
        location=loc,
    )
    return _csc_result(op, coo_x.dense_shape, loc)


def to_coo(x):
    """Convert a sparse tensor to canonical COO.

    Concrete instance -> eager ``x.to_coo()``; symbolic instance: ``"coo"``
    is returned unchanged, ``"csr"`` emits ``sparse_csr_to_coo``, ``"csc"``
    emits ``sparse_csc_to_coo`` (both rank-2 only).
    """
    x = _require_sparse(x, "to_coo")
    if _require_value_phase(x, "to_coo"):
        return _as_coo(x)
    return x.to_coo()


# --- computation ops (symbolic sparse in -> symbolic sparse/dense out) -------


def negate(a):
    """Negate the sparse values; the sparsity structure is preserved.

    Result is a COO sparse tensor with the same dense_shape and dtype.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    a = _as_coo(a)
    op = builder.create(
        "sparse_negate",
        operands=(a.indices.value, a.values.value),
        attributes=_sparse_attrs(a),
        location=loc,
    )
    return _coo_result(op, a.dense_shape, loc)


def add(a, b):
    """Sparse + sparse with a union merge (overlapping entries summed).

    Both operands are auto-converted to COO. Static checks first:
    ``dense_shape`` equality -> ``core.ShapeError`` (naming both); dtype
    equality -> ``core.DTypeError``.

    Raises:
        core.ShapeError: ``a.dense_shape != b.dense_shape``.
        core.DTypeError: ``a.dtype != b.dtype``.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    b = _require_symbolic_sparse(b)
    if a.dense_shape != b.dense_shape:
        raise core.ShapeError(
            f"sparse add: dense shapes must match, got {a.dense_shape} and "
            f"{b.dense_shape}"
        )
    if a.dtype != b.dtype:
        raise core.DTypeError(
            f"sparse add: dtypes must match, got {a.dtype} and {b.dtype}"
        )
    a = _as_coo(a)
    b = _as_coo(b)
    op = builder.create(
        "sparse_add",
        operands=(
            a.indices.value,
            a.values.value,
            b.indices.value,
            b.values.value,
        ),
        attributes=_sparse_attrs(a),
        location=loc,
    )
    return _coo_result(op, a.dense_shape, loc)


def subtract(a, b):
    """Sparse - sparse: composed as ``add(a, negate(b))`` (no dedicated op)."""
    return add(a, negate(b))


def multiply(a, b):
    """Sparse * sparse with an intersection merge (overlapping entries
    multiplied); the union structure is dropped.

    Static checks first, exactly like :func:`add`.

    Raises:
        core.ShapeError: ``a.dense_shape != b.dense_shape``.
        core.DTypeError: ``a.dtype != b.dtype``.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    b = _require_symbolic_sparse(b)
    if a.dense_shape != b.dense_shape:
        raise core.ShapeError(
            f"sparse multiply: dense shapes must match, got {a.dense_shape} "
            f"and {b.dense_shape}"
        )
    if a.dtype != b.dtype:
        raise core.DTypeError(
            f"sparse multiply: dtypes must match, got {a.dtype} and {b.dtype}"
        )
    a = _as_coo(a)
    b = _as_coo(b)
    op = builder.create(
        "sparse_multiply",
        operands=(
            a.indices.value,
            a.values.value,
            b.indices.value,
            b.values.value,
        ),
        attributes=_sparse_attrs(a),
        location=loc,
    )
    return _coo_result(op, a.dense_shape, loc)


def multiply_dense(a, dense):
    """Sparse * dense elementwise: structure preserved, values scaled by the
    dense tensor.

    ``dense`` must be a symbolic dense tensor (three-option ``TraceError``
    for concrete values). Static shape check: ``dense.shape`` must equal
    ``a.dense_shape`` or match its trailing ``ndim`` dims (batched-dense
    tolerance).

    Raises:
        core.ShapeError: incompatible static shapes.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    dense = _require_symbolic_dense(dense)
    ndim = a.ndim
    if dense.shape != a.dense_shape and (
        len(dense.shape) < ndim or dense.shape[-ndim:] != a.dense_shape
    ):
        raise core.ShapeError(
            f"sparse multiply_dense: dense shape {dense.shape} must equal the "
            f"sparse dense_shape {a.dense_shape} (or match its trailing "
            f"{ndim} dims for batched dense)"
        )
    a = _as_coo(a)
    op = builder.create(
        "sparse_multiply_dense",
        operands=(a.indices.value, a.values.value, dense.value),
        attributes=_sparse_attrs(a),
        location=loc,
    )
    return _coo_result(op, a.dense_shape, loc)


def sum(a, axes: Optional[Union[int, Tuple[int, ...], list]] = None, keepdims: bool = False):
    """Sum the sparse values over the given (unbatched) sparse axes -> DENSE.

    ``axes=None`` reduces ALL axes (an empty ``axes`` tuple is a ``ShapeError``
    — the kernel treats an empty tuple as NO reduction, so the frontend never
    emits it). Negatives wrap; out-of-range axes raise ``core.ShapeError``.
    The result is a dense ``SymbolicTensor``.

    Alias: ``reduce_sum = sum``.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    rank = a.ndim
    if axes is None:
        normalized: Tuple[int, ...] = tuple(range(rank))
    elif isinstance(axes, Integral) and not isinstance(axes, bool):
        normalized = _normalize_axes((int(axes),), rank, "sparse sum")
    elif isinstance(axes, (tuple, list)):
        if not all(
            isinstance(ax, Integral) and not isinstance(ax, bool) for ax in axes
        ):
            raise core.ShapeError(
                f"sparse sum: axes must be ints, got {axes!r}"
            )
        normalized = _normalize_axes(tuple(int(ax) for ax in axes), rank, "sparse sum")
    else:
        raise TypeError(
            "sparse sum: axes must be None, an int, or a tuple/list of ints, "
            f"got {type(axes).__name__}"
        )
    if not normalized:
        raise core.ShapeError(
            "sparse sum: reduce over no axes (empty axes) — pass axes=None to "
            "reduce all axes"
        )
    a = _as_coo(a)
    op = builder.create(
        "sparse_reduce_sum",
        operands=(a.indices.value, a.values.value),
        attributes={
            **_sparse_attrs(a),
            "axes": normalized,
            "keepdims": bool(keepdims),
        },
        location=loc,
    )
    return _wrap_dense(op.results[0], loc)


#: Alias of :func:`sum` (numpy-style spelling).
reduce_sum = sum


def transpose(a, perm: Optional[Sequence[int]] = None):
    """Permute the sparse axes; the result is COO with the permuted shape.

    ``perm=None`` reverses the axes. ``perm`` must be a permutation of
    ``range(rank)`` (non-bool ints, length == rank, ``sorted(perm) ==
    list(range(rank))``) — ``core.ShapeError`` otherwise.

    Raises:
        core.ShapeError: invalid ``perm``.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    rank = a.ndim
    if perm is None:
        perm_tuple: Tuple[int, ...] = tuple(reversed(range(rank)))
    else:
        if (
            not isinstance(perm, (tuple, list))
            or len(perm) != rank
            or not all(isinstance(p, int) and not isinstance(p, bool) for p in perm)
        ):
            raise core.ShapeError(
                f"sparse transpose: perm must be a permutation of {rank} axes, "
                f"got {perm!r}"
            )
        if sorted(perm) != list(range(rank)):
            raise core.ShapeError(
                f"sparse transpose: {perm!r} is not a permutation of {rank} axes"
            )
        perm_tuple = tuple(int(p) for p in perm)
    result_shape = tuple(a.dense_shape[p] for p in perm_tuple)
    a = _as_coo(a)
    op = builder.create(
        "sparse_transpose",
        operands=(a.indices.value, a.values.value),
        attributes={
            "dense_shape": result_shape,
            "dtype": a.dtype.name,
            "perm": perm_tuple,
        },
        location=loc,
    )
    return _coo_result(op, result_shape, loc)


def reshape(a, new_shape: Union[int, Sequence[int]]):
    """Reshape the sparse tensor to a new dense shape (element count equal).

    ``new_shape`` is a tuple of positive non-bool ints (``ShapeError``
    otherwise); its element count must equal ``math.prod(a.dense_shape)``
    (``ShapeError`` naming both). Static shapes only in v1: a ``core.Dim`` in
    ``a.dense_shape`` raises ``ShapeError``. The result is COO with the new
    shape; the op records the old shape in ``old_shape``.

    Raises:
        core.ShapeError: invalid ``new_shape``, symbolic dims, or element-count
            mismatch.
    """
    builder = current_builder()
    loc = _get_location()
    a = _require_symbolic_sparse(a)
    if isinstance(new_shape, Integral) and not isinstance(new_shape, bool):
        new_shape = (int(new_shape),)
    if not isinstance(new_shape, (tuple, list)) or not new_shape:
        raise core.ShapeError(
            f"sparse reshape: new_shape must be a non-empty tuple of positive "
            f"ints, got {new_shape!r}"
        )
    for entry in new_shape:
        if not isinstance(entry, Integral) or isinstance(entry, bool) or entry <= 0:
            raise core.ShapeError(
                f"sparse reshape: new_shape entries must be positive non-bool "
                f"ints, got {new_shape!r}"
            )
    new_tuple = tuple(int(entry) for entry in new_shape)
    if any(isinstance(d, (core.Dim, core.DimExpr)) for d in a.dense_shape):
        raise core.ShapeError(
            "sparse reshape: static shapes only in v1 (dense_shape contains "
            f"a symbolic dim: {a.dense_shape})"
        )
    old_count = math.prod(a.dense_shape)
    new_count = math.prod(new_tuple)
    if old_count != new_count:
        raise core.ShapeError(
            f"sparse reshape: element count {old_count} (dense_shape "
            f"{a.dense_shape}) does not match new_shape {new_tuple} "
            f"({new_count} elements)"
        )
    a = _as_coo(a)
    op = builder.create(
        "sparse_reshape",
        operands=(a.indices.value, a.values.value),
        attributes={
            "dense_shape": new_tuple,
            "dtype": a.dtype.name,
            "old_shape": tuple(a.dense_shape),
        },
        location=loc,
    )
    return _coo_result(op, new_tuple, loc)


def concatenate(operands: Sequence[Any], axis: int = 0):
    """Concatenate >= 2 sparse tensors along one sparse axis (variadic).

    All operands are auto-converted to COO; they must share one dtype
    (``core.DTypeError``) and match in dense shape except along ``axis``
    (``core.ShapeError``). v1 supports ``axis`` in ``{0, 1}`` only
    (``ShapeError`` otherwise). The result is COO with the axis summed; the
    op records per-operand extents in ``operand_extents``.

    Raises:
        TypeError: ``operands`` is not a sequence of >= 2 sparse tensors.
        core.ShapeError / core.DTypeError: static shape/dtype violations.
    """
    builder = current_builder()
    loc = _get_location()
    if not isinstance(operands, (tuple, list)):
        raise TypeError(
            "sparse concatenate: operands must be a sequence (tuple/list) of "
            f"sparse tensors, got {type(operands).__name__}"
        )
    if len(operands) < 2:
        raise TypeError(
            f"sparse concatenate: at least 2 operands required, got "
            f"{len(operands)}"
        )
    normalized = [_require_symbolic_sparse(op) for op in operands]
    first = normalized[0]
    rank = first.ndim
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in (0, 1):
        raise core.ShapeError(
            f"sparse concatenate: v1 supports axis 0 or 1, got {axis!r}"
        )
    if rank <= axis:
        raise core.ShapeError(
            f"sparse concatenate: axis {axis} out of range for rank {rank}"
        )
    for op in normalized[1:]:
        if op.dtype != first.dtype:
            raise core.DTypeError(
                f"sparse concatenate: operand dtypes must match, got "
                f"{first.dtype} and {op.dtype}"
            )
    result_shape = list(first.dense_shape)
    extents = []
    for op in normalized:
        ds = op.dense_shape
        if len(ds) != rank:
            raise core.ShapeError(
                f"sparse concatenate: operand ranks must match, got {ds} and "
                f"{first.dense_shape}"
            )
        for d in range(rank):
            if d != axis and ds[d] != first.dense_shape[d]:
                raise core.ShapeError(
                    f"sparse concatenate: dense shapes must match except along "
                    f"axis {axis}, got {first.dense_shape} and {ds} (mismatch "
                    f"at axis {d})"
                )
        extents.append(ds[axis])
    extents_total = 0
    for e in extents:
        extents_total += e
    result_shape[axis] = extents_total
    result_shape = tuple(result_shape)
    coos = [_as_coo(op) for op in normalized]
    flat_operands = []
    for c in coos:
        flat_operands.extend([c.indices.value, c.values.value])
    op = builder.create(
        "sparse_concatenate",
        operands=tuple(flat_operands),
        attributes={
            **_sparse_attrs(first),
            "dense_shape": result_shape,
            "axis": axis,
            "operand_extents": tuple(extents),
        },
        location=loc,
    )
    return _coo_result(op, result_shape, loc)


def matmul(a, b):
    """Sparse/dense matmul, dispatching on the operand kinds.

    - sparse @ dense -> ``sparse_dot_dense`` (rank-2 sparse (M, K) x dense
      (..., K, N)); result DENSE. Static check: ``b.shape[-2] ==
      a.dense_shape[1]`` when ``b.shape[-2]`` is a static int.
    - dense @ sparse -> ``dense_dot_sparse`` (dense (..., M, K) x rank-2
      sparse (K, N)); result DENSE. Static check: ``a.shape[-1] ==
      b.dense_shape[0]`` when ``a.shape[-1]`` is a static int.
    - sparse @ sparse -> ``core.TraceError`` (v1 deferral: densify one
      operand with ``etl.sparse.to_dense``).
    - dense @ dense -> ``etl.ops.dot`` (lazy import — no cycle).
    - concrete values anywhere -> the three-option ``TraceError``.

    Raises:
        core.ShapeError: rank/inner-dim static violations.
        core.TraceError: sparse @ sparse (v1 deferral) or concrete operands.
    """
    a_sparse = is_sparse(a)
    b_sparse = is_sparse(b)
    a_sym = a_sparse and isinstance(a.values, core.SymbolicTensor)
    b_sym = b_sparse and isinstance(b.values, core.SymbolicTensor)
    if a_sym and b_sym:
        raise core.TraceError(
            "v1 deferral: sparse @ sparse matmul is not supported in v1 "
            "(densify one operand with etl.sparse.to_dense)"
        )
    if a_sparse or b_sparse:
        # At this point not BOTH operands are symbolic-sparse (handled
        # above). Any sparse operand that is NOT symbolic is concrete (or a
        # spec): normalize it — the three-option TraceError / TypeError.
        if a_sparse and not a_sym:
            _require_symbolic_sparse(a)
        if b_sparse and not b_sym:
            _require_symbolic_sparse(b)
        builder = current_builder()
        loc = _get_location()
        if a_sym:
            # sparse @ dense
            if a.ndim != 2:
                raise core.ShapeError(
                    f"sparse matmul: sparse operand must be rank-2, got rank "
                    f"{a.ndim} (dense_shape={a.dense_shape})"
                )
            b = _require_symbolic_dense(b)
            if len(b.shape) < 2:
                raise core.ShapeError(
                    f"sparse matmul: dense operand must have rank >= 2, got "
                    f"rank {len(b.shape)}"
                )
            k = a.dense_shape[1]
            if (
                isinstance(b.shape[-2], int)
                and not isinstance(b.shape[-2], bool)
                and b.shape[-2] != k
            ):
                raise core.ShapeError(
                    f"sparse matmul: inner dims must match — sparse (M, K) "
                    f"with K={k}, dense shape {b.shape} has K={b.shape[-2]}"
                )
            a = _as_coo(a)
            op = builder.create(
                "sparse_dot_dense",
                operands=(a.indices.value, a.values.value, b.value),
                attributes=_sparse_attrs(a),
                location=loc,
            )
            return _wrap_dense(op.results[0], loc)
        # dense @ sparse
        if b.ndim != 2:
            raise core.ShapeError(
                f"sparse matmul: sparse operand must be rank-2, got rank "
                f"{b.ndim} (dense_shape={b.dense_shape})"
            )
        a = _require_symbolic_dense(a)
        if len(a.shape) < 2:
            raise core.ShapeError(
                f"sparse matmul: dense operand must have rank >= 2, got rank "
                f"{len(a.shape)}"
            )
        k = b.dense_shape[0]
        if (
            isinstance(a.shape[-1], int)
            and not isinstance(a.shape[-1], bool)
            and a.shape[-1] != k
        ):
            raise core.ShapeError(
                f"sparse matmul: inner dims must match — dense shape {a.shape} "
                f"has K={a.shape[-1]}, sparse (K, N) with K={k}"
            )
        b = _as_coo(b)
        op = builder.create(
            "dense_dot_sparse",
            operands=(a.value, b.indices.value, b.values.value),
            attributes=_sparse_attrs(b),
            location=loc,
        )
        return _wrap_dense(op.results[0], loc)
    # dense @ dense: no sparse involved — delegate to the dense frontend.
    a = _require_symbolic_dense(a)
    b = _require_symbolic_dense(b)
    from etl.ops import dot  # lazy: keeps the import graph acyclic

    return dot(a, b)
