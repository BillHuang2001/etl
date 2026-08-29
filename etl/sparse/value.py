"""Sparse value model — the three-phase sparse tensor.

A sparse tensor is an explicit (structure, values) pair with a known dense
shape:

- ``indices`` — COO: int64 coordinates ``(nnz, ndim)`` (one row per stored
  entry); CSR/CSC: the per-row/per-column column/row index list ``(nnz,)``.
- ``values`` — the stored values ``(nnz,)`` (any numeric dtype).
- ``indptr`` — CSR/CSC only: the row/column pointer array ``(rows+1,)`` /
  ``(cols+1,)``.
- ``dense_shape`` — the dense shape of the materialized tensor.

Three phases share ONE class hierarchy rooted at :class:`SparseTensor`
(``is_sparse`` catches every phase and variant):

1. **Spec phase** — :class:`SparseTensorSpec`: the leaves are
   ``core.TensorSpec``; describes a future runtime sparse tensor (trace
   inputs).
2. **Symbolic phase** — ``SparseTensor`` / ``CSRTensor`` / ``CSCTensor``
   instances whose leaves are ``core.SymbolicTensor`` (graph values).
   Assembled via :meth:`SparseTensor.from_parts` — **no canonical
   validation** (there is no data to validate).
3. **Concrete phase** — instances whose leaves are numpy arrays or
   ``core.Tensor``; built through the variant constructors, which apply
   canonical-form validation (never silently).

Canonical forms (binding; violations raise ``core.ShapeError`` /
``core.DTypeError``, never silent):

- **COO** — ``indices`` is 2-D ``(nnz, ndim)`` with ``ndim ==
  len(dense_shape)``; rows are lex-sorted (leftmost column first), strictly
  unique, and in-range ``0 <= idx[d] < dense_shape[d]``.
- **CSR** — rank-2 ``(rows, cols)``; ``indptr`` is ``(rows+1,)`` int64,
  monotone non-decreasing, ``indptr[0] == 0``, ``indptr[-1] == nnz``;
  per-row column indices are sorted and strictly increasing, in-range.
- **CSC** — symmetric: indptr over columns, per-column sorted rows.

``dense_shape`` entries are positive ints; a ``core.Dim`` is allowed only at
position 0 (batched/vmap outputs). For a ``Dim`` at position 0 the concrete
in-range bound is ``indices.shape[0]`` (its extent).

**Pytree contract (binding for trace/pipeline/transforms):** the whole
hierarchy is ONE registered pytree node (``core.register_pytree_node(
SparseTensor, ...)`` — the MRO lookup makes ``TreeSpec.type == SparseTensor``
for every variant). Children layout (context is always ``None``):

- COO: ``[indices, values, *dense_shape, dtype, "coo"]``
- CSR: ``[indptr, indices, values, *dense_shape, dtype, "csr"]``
- CSC: ``[indptr, indices, values, *dense_shape, dtype, "csc"]``

``dense_shape`` contributes ONE leaf per dim (plain ints, or the ``Dim``
object itself), then the values dtype as an ``np.dtype`` leaf, then the
format string leaf. Unflatten dispatches on the first child: ``TensorSpec``
→ spec phase, ``SymbolicTensor`` → symbolic (``from_parts``), ``Tensor`` /
``ndarray`` → concrete (canonical validation).

The layout helpers (``to_dense`` / ``to_coo`` / ``to_csr`` / ``to_csc``) are
pure numpy materialization utilities for the concrete phase — the GRAPH-side
conversions live in ``etl.sparse.ops`` (another module).

Import contract: this module imports ``etl.core`` and ``numpy`` only.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from etl import core

__all__ = [
    "SparseTensor",
    "CSRTensor",
    "CSCTensor",
    "SparseTensorSpec",
    "is_sparse",
]

_FORMATS = ("coo", "csr", "csc")
_INT64 = np.dtype("int64")


# --- validation helpers (concrete + spec phases; never silent) ---------------


def _validate_dense_shape(dense_shape, class_name):
    """Validate a dense shape: non-empty, positive ints, ``Dim`` at position 0.

    Returns the normalized tuple (numpy integers converted to plain ints; a
    ``Dim`` entry preserved as the ``Dim`` object itself).
    """
    try:
        dense_shape = tuple(dense_shape)
    except TypeError as exc:
        raise core.ShapeError(
            f"{class_name}: dense_shape must be a tuple of positive ints, "
            f"got {dense_shape!r}"
        ) from exc
    if len(dense_shape) == 0:
        raise core.ShapeError(f"{class_name}: dense_shape must be non-empty, got ()")
    normalized = []
    for axis, entry in enumerate(dense_shape):
        if isinstance(entry, core.Dim):
            if axis != 0:
                raise core.ShapeError(
                    f"{class_name}: only dense_shape[0] may be a core.Dim, "
                    f"got {entry!r} at position {axis}"
                )
            normalized.append(entry)
        elif isinstance(entry, Integral) and not isinstance(entry, bool):
            if entry <= 0:
                raise core.ShapeError(
                    f"{class_name}: dense_shape entries must be positive ints, "
                    f"got {entry!r} at position {axis}"
                )
            normalized.append(int(entry))
        else:
            raise core.ShapeError(
                f"{class_name}: dense_shape entries must be positive ints "
                f"(a core.Dim allowed at position 0), got {entry!r} at position {axis}"
            )
    return tuple(normalized)


def _normalize_index_leaf(leaf, name, class_name):
    """Normalize an indices/indptr leaf to int64 (concrete path).

    Accepts numpy arrays and ``core.Tensor`` wrappers. Any integer dtype is
    normalized to int64 via ``astype`` (a copy is made only when needed); a
    non-integer dtype (float/bool/object/...) raises ``core.DTypeError``.
    """
    if isinstance(leaf, core.Tensor):
        arr = leaf.numpy()
        is_tensor = True
    elif isinstance(leaf, np.ndarray):
        arr = leaf
        is_tensor = False
    else:
        raise core.DTypeError(
            f"{class_name}: {name} must be an integer numpy array or core.Tensor, "
            f"got {type(leaf).__name__} (symbolic leaves: assemble via "
            "SparseTensor.from_parts)"
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise core.DTypeError(
            f"{class_name}: {name} must be integer-typed (normalized to int64), "
            f"got dtype {arr.dtype}"
        )
    if arr.dtype != _INT64:
        normalized = arr.astype(np.int64)
        if is_tensor:
            return core.Tensor(normalized, device=leaf.device)
        return normalized
    return leaf


def _validate_values_leaf(values, class_name):
    """Validate a values leaf: ndarray/``core.Tensor`` with a numeric dtype."""
    if isinstance(values, (np.ndarray, core.Tensor)):
        dt = values.dtype
    else:
        raise core.DTypeError(
            f"{class_name}: values must be a numpy array or core.Tensor, got "
            f"{type(values).__name__} (symbolic leaves: assemble via "
            "SparseTensor.from_parts)"
        )
    if not (
        np.issubdtype(dt, np.integer)
        or np.issubdtype(dt, np.floating)
        or np.issubdtype(dt, np.bool_)
        or np.issubdtype(dt, np.complexfloating)
    ):
        raise core.DTypeError(
            f"{class_name}: values must have a numeric dtype (int/float/bool/"
            f"complex), got {dt}"
        )
    return values


def _leaf_numpy(leaf, what):
    """Unwrap a concrete leaf to its numpy array (layout helpers only)."""
    if isinstance(leaf, np.ndarray):
        return leaf
    if isinstance(leaf, core.Tensor):
        return leaf.numpy()
    raise TypeError(
        f"{what} requires concrete leaves (numpy arrays or core.Tensor), got "
        f"{type(leaf).__name__} — symbolic/spec sparse tensors have no data; "
        "use the graph-side conversions in etl.sparse.ops"
    )


def _axis_bound(shape_entry, nnz, axis, class_name):
    """The in-range bound for one dense axis.

    A ``Dim`` is only legal at position 0 (validated by
    ``_validate_dense_shape``); its extent is ``nnz`` (``indices.shape[0]``),
    per the value-model contract for batched/vmap outputs.
    """
    if isinstance(shape_entry, core.Dim):
        if axis != 0:
            raise core.ShapeError(
                f"{class_name}: only dense_shape[0] may be a core.Dim, "
                f"got {shape_entry!r} at position {axis}"
            )
        return nnz
    return shape_entry


def _validate_coo(indices, values, dense_shape, class_name):
    """Canonical COO validation: ``(nnz, ndim)`` int64 lex-sorted unique
    in-range rows and ``(nnz,)`` values."""
    indices = _normalize_index_leaf(indices, "indices", class_name)
    values = _validate_values_leaf(values, class_name)
    indices_arr = indices.numpy() if isinstance(indices, core.Tensor) else indices
    values_arr = values.numpy() if isinstance(values, core.Tensor) else values
    if indices_arr.ndim != 2:
        raise core.ShapeError(
            f"{class_name}: COO indices must be 2-D (nnz, ndim), got rank "
            f"{indices_arr.ndim}"
        )
    ndim = len(dense_shape)
    if indices_arr.shape[1] != ndim:
        raise core.ShapeError(
            f"{class_name}: COO indices must have shape (nnz, {ndim}) matching "
            f"dense_shape rank, got {indices_arr.shape}"
        )
    nnz = indices_arr.shape[0]  # ndarray shapes are never negative (nnz >= 0)
    if values_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: COO values must be 1-D (nnz,), got rank {values_arr.ndim}"
        )
    if values_arr.shape[0] != nnz:
        raise core.ShapeError(
            f"{class_name}: COO values length must equal nnz ({nnz}), got "
            f"{values_arr.shape[0]}"
        )
    for d, entry in enumerate(dense_shape):
        bound = _axis_bound(entry, nnz, d, class_name)
        if np.any((indices_arr[:, d] < 0) | (indices_arr[:, d] >= bound)):
            raise core.ShapeError(
                f"{class_name}: COO indices out of range for axis {d}: entries "
                f"must satisfy 0 <= idx < {bound}"
            )
    # Lex-sorted row-major (leftmost column first): np.lexsort sorts by its
    # LAST key first, so the column-coordinate rows must be reversed.
    if not np.array_equal(np.lexsort(indices_arr.T[::-1]), np.arange(nnz)):
        raise core.ShapeError(
            f"{class_name}: COO indices must be lex-sorted in row-major order "
            "(leftmost column first)"
        )
    # Strictly unique rows (adjacent comparison is valid after the sort check).
    if nnz > 1 and np.any(np.all(indices_arr[1:] == indices_arr[:-1], axis=1)):
        raise core.ShapeError(
            f"{class_name}: COO indices must be unique (no duplicate rows)"
        )
    return indices, values


def _validate_csr(indptr, indices, values, dense_shape, class_name):
    """Canonical CSR validation: rank-2, ``(rows+1,)`` monotone indptr with
    ``indptr[0] == 0`` / ``indptr[-1] == nnz``, per-row strictly increasing
    in-range columns."""
    if len(dense_shape) != 2:
        raise core.ShapeError(
            f"{class_name}: CSR format requires a rank-2 dense_shape (rows, "
            f"cols), got rank {len(dense_shape)}"
        )
    indptr = _normalize_index_leaf(indptr, "indptr", class_name)
    indices = _normalize_index_leaf(indices, "indices", class_name)
    values = _validate_values_leaf(values, class_name)
    indptr_arr = indptr.numpy() if isinstance(indptr, core.Tensor) else indptr
    indices_arr = indices.numpy() if isinstance(indices, core.Tensor) else indices
    values_arr = values.numpy() if isinstance(values, core.Tensor) else values
    rows, cols = dense_shape
    if indptr_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: indptr must be 1-D ({rows}+1,), got rank "
            f"{indptr_arr.ndim}"
        )
    if isinstance(rows, int) and indptr_arr.shape[0] != rows + 1:
        raise core.ShapeError(
            f"{class_name}: indptr must have shape ({rows}+1,), got "
            f"{indptr_arr.shape}"
        )
    if indptr_arr.shape[0] == 0:
        raise core.ShapeError(
            f"{class_name}: indptr must be non-empty ({rows}+1 entries)"
        )
    if np.any(np.diff(indptr_arr) < 0):
        raise core.ShapeError(
            f"{class_name}: indptr must be monotone non-decreasing"
        )
    if indptr_arr[0] != 0:
        raise core.ShapeError(f"{class_name}: indptr[0] must be 0, got {indptr_arr[0]}")
    if indices_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: indices must be 1-D (nnz,), got rank {indices_arr.ndim}"
        )
    if values_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: values must be 1-D (nnz,), got rank {values_arr.ndim}"
        )
    nnz = indices_arr.shape[0]
    if values_arr.shape[0] != nnz:
        raise core.ShapeError(
            f"{class_name}: values length must equal nnz ({nnz}), got "
            f"{values_arr.shape[0]}"
        )
    if indptr_arr[-1] != nnz:
        raise core.ShapeError(
            f"{class_name}: indptr[-1] must equal nnz ({nnz}), got {indptr_arr[-1]}"
        )
    if np.any((indices_arr < 0) | (indices_arr >= cols)):
        raise core.ShapeError(
            f"{class_name}: CSR indices must be in range 0 <= idx < {cols} (cols)"
        )
    # Per-row sorted and strictly increasing (cross-row diffs are unchecked).
    if nnz > 1:
        row_ids = np.repeat(np.arange(indptr_arr.shape[0] - 1), np.diff(indptr_arr))
        same_row = np.diff(row_ids) == 0
        if np.any((np.diff(indices_arr) <= 0) & same_row):
            raise core.ShapeError(
                f"{class_name}: per-row indices must be sorted and strictly "
                "increasing"
            )
    return indptr, indices, values


def _validate_csc(indptr, indices, values, dense_shape, class_name):
    """Canonical CSC validation: symmetric to CSR (indptr over columns,
    per-column sorted rows)."""
    if len(dense_shape) != 2:
        raise core.ShapeError(
            f"{class_name}: CSC format requires a rank-2 dense_shape (rows, "
            f"cols), got rank {len(dense_shape)}"
        )
    indptr = _normalize_index_leaf(indptr, "indptr", class_name)
    indices = _normalize_index_leaf(indices, "indices", class_name)
    values = _validate_values_leaf(values, class_name)
    indptr_arr = indptr.numpy() if isinstance(indptr, core.Tensor) else indptr
    indices_arr = indices.numpy() if isinstance(indices, core.Tensor) else indices
    values_arr = values.numpy() if isinstance(values, core.Tensor) else values
    rows, cols = dense_shape
    if indptr_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: indptr must be 1-D ({cols}+1,), got rank "
            f"{indptr_arr.ndim}"
        )
    if indptr_arr.shape[0] != cols + 1:
        raise core.ShapeError(
            f"{class_name}: indptr must have shape ({cols}+1,), got "
            f"{indptr_arr.shape}"
        )
    if np.any(np.diff(indptr_arr) < 0):
        raise core.ShapeError(
            f"{class_name}: indptr must be monotone non-decreasing"
        )
    if indptr_arr[0] != 0:
        raise core.ShapeError(f"{class_name}: indptr[0] must be 0, got {indptr_arr[0]}")
    if indices_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: indices must be 1-D (nnz,), got rank {indices_arr.ndim}"
        )
    if values_arr.ndim != 1:
        raise core.ShapeError(
            f"{class_name}: values must be 1-D (nnz,), got rank {values_arr.ndim}"
        )
    nnz = indices_arr.shape[0]
    if values_arr.shape[0] != nnz:
        raise core.ShapeError(
            f"{class_name}: values length must equal nnz ({nnz}), got "
            f"{values_arr.shape[0]}"
        )
    if indptr_arr[-1] != nnz:
        raise core.ShapeError(
            f"{class_name}: indptr[-1] must equal nnz ({nnz}), got {indptr_arr[-1]}"
        )
    row_bound = _axis_bound(rows, nnz, 0, class_name)
    if np.any((indices_arr < 0) | (indices_arr >= row_bound)):
        raise core.ShapeError(
            f"{class_name}: CSC indices must be in range 0 <= idx < {row_bound} (rows)"
        )
    # Per-column sorted and strictly increasing.
    if nnz > 1:
        col_ids = np.repeat(np.arange(indptr_arr.shape[0] - 1), np.diff(indptr_arr))
        same_col = np.diff(col_ids) == 0
        if np.any((np.diff(indices_arr) <= 0) & same_col):
            raise core.ShapeError(
                f"{class_name}: per-column indices must be sorted and strictly "
                "increasing"
            )
    return indptr, indices, values


def _require_tensorspec(spec, class_name, leaf_name):
    """Require a leaf spec to be a ``core.TensorSpec``."""
    if not isinstance(spec, core.TensorSpec):
        raise core.ShapeError(
            f"{class_name}: {leaf_name} leaf spec must be a core.TensorSpec, "
            f"got {type(spec).__name__}"
        )


# --- the value classes -------------------------------------------------------


class SparseTensor:
    """Sparse tensor — base class AND the concrete COO representation.

    Attributes:
        format: ``"coo"`` | ``"csr"`` | ``"csc"``.
        dense_shape: Tuple of positive ints (a ``core.Dim`` allowed only at
            position 0, for batched/vmap outputs).
        _leaves: Tuple of the tensor leaves — ``(indices, values)`` for COO,
            ``(indptr, indices, values)`` for CSR/CSC — followed by the
            ``dense_shape`` / ``format`` attributes stored separately.

    The concrete constructor (this one) applies canonical-form validation.
    Symbolic assembly must use :meth:`from_parts` (no validation — symbolic
    leaves have no data to check).
    """

    def __init__(self, indices, values, dense_shape, format="coo"):
        """Construct a concrete sparse tensor with canonical validation.

        The base class is the COO representation — only ``format="coo"`` is
        constructible here; CSR/CSC have dedicated constructors
        (:class:`CSRTensor` / :class:`CSCTensor`), and
        :meth:`from_parts` assembles without validation (symbolic leaves).

        Raises:
            core.ShapeError: Shape/order/range violations of the canonical
                COO form.
            core.DTypeError: Non-integer indices or non-numeric values.
            ValueError: Unknown ``format`` (or a non-COO format).
        """
        class_name = type(self).__name__
        if format != "coo":
            if format in _FORMATS:
                raise ValueError(
                    f"{class_name}: format {format!r} is not constructible "
                    "through the base class — use CSRTensor/CSCTensor (or "
                    "SparseTensor.from_parts for validation-free assembly)"
                )
            raise ValueError(
                f"{class_name}: unknown sparse format {format!r} — expected "
                f"one of {_FORMATS}"
            )
        dense_shape = _validate_dense_shape(dense_shape, class_name)
        leaves = _validate_coo(indices, values, dense_shape, class_name)
        self._leaves = tuple(leaves)
        self.dense_shape = dense_shape
        self.format = format

    @classmethod
    def from_parts(cls, *leaves, dense_shape, format):
        """Assemble a sparse tensor from raw leaves WITHOUT validation.

        Used for SYMBOLIC assembly (leaves are ``core.SymbolicTensor``) and
        by the pytree ``unflatten`` path. No canonical checks are applied —
        there is no data to validate. Returns an instance of the variant
        class selected by ``format``: ``"coo"`` → :class:`SparseTensor`,
        ``"csr"`` → :class:`CSRTensor`, ``"csc"`` → :class:`CSCTensor`.
        Leaves are stored exactly as given.

        Args:
            *leaves: The tensor leaves — ``(indices, values)`` for COO,
                ``(indptr, indices, values)`` for CSR/CSC.
            dense_shape: The dense shape (stored as a tuple).
            format: The sparse format string.

        Raises:
            ValueError: Unknown ``format``.
        """
        if format not in _FORMATS:
            raise ValueError(
                f"from_parts: unknown sparse format {format!r} — expected one "
                f"of {_FORMATS}"
            )
        variant = {"coo": SparseTensor, "csr": CSRTensor, "csc": CSCTensor}[format]
        obj = variant.__new__(variant)
        obj._leaves = tuple(leaves)
        obj.dense_shape = tuple(dense_shape)
        obj.format = format
        return obj

    # --- attributes / properties --------------------------------------------

    @property
    def ndim(self) -> int:
        """Rank of the dense tensor (``len(dense_shape)``)."""
        return len(self.dense_shape)

    @property
    def indices(self):
        """The coordinate/index leaf: ``(nnz, ndim)`` for COO, the
        per-row/per-column index list for CSR/CSC."""
        return self._leaves[0] if self.format == "coo" else self._leaves[1]

    @property
    def values(self):
        """The values leaf (always the LAST leaf)."""
        return self._leaves[-1]

    @property
    def indptr(self):
        """The pointer leaf (CSR/CSC only). COO has no ``indptr``."""
        if self.format == "coo":
            raise AttributeError(
                f"{type(self).__name__} (COO) has no indptr — only CSR/CSC "
                "sparse tensors carry a pointer array"
            )
        return self._leaves[0]

    @property
    def dtype(self) -> np.dtype:
        """The values leaf's dtype as a :class:`numpy.dtype`."""
        return core.dtype(self._leaves[-1])

    def __repr__(self) -> str:
        values_leaf = self._leaves[-1]
        shape = getattr(values_leaf, "shape", None)
        return (
            f"{type(self).__name__}(format={self.format!r}, "
            f"dense_shape={self.dense_shape!r}, dtype={self.dtype}, "
            f"values_shape={shape!r})"
        )

    # --- concrete layout helpers (numpy materialization; concrete phase) -----

    def to_dense(self) -> np.ndarray:
        """Materialize the dense tensor (zeros + scatter-add of the values).

        Duplicate rows accumulate (``np.add.at``). The result dtype is the
        sparse values dtype. A ``core.Dim`` at ``dense_shape[0]`` is
        substituted with ``indices.shape[0]`` (its extent) — the BATCHED
        case: the tensor leaves carry leading batch dims (COO indices
        ``(B, nnz, ndim)``; CSR/CSC indices/values ``(B, nnz)`` with indptr
        ``(B, rows+1)``), and the batch id is scatter-added as the leading
        coordinate axis.

        Returns:
            The dense :class:`numpy.ndarray` of shape ``dense_shape``.
        """
        indices = _leaf_numpy(self.indices, "to_dense")
        values = _leaf_numpy(self.values, "to_dense")
        dense_shape = list(self.dense_shape)
        if isinstance(dense_shape[0], core.Dim):
            dense_shape[0] = indices.shape[0]
        if self.format == "coo":
            out = np.zeros(dense_shape, dtype=self.dtype)
            if indices.ndim == 3:
                # Batched: (B, nnz, ndim) — batch id + per-axis coordinates.
                batch_ids = np.broadcast_to(
                    np.arange(indices.shape[0])[:, None], indices.shape[:2]
                )
                np.add.at(
                    out,
                    (batch_ids,)
                    + tuple(indices[..., d] for d in range(indices.shape[2])),
                    values,
                )
            else:
                np.add.at(out, tuple(indices.T), values)
            return out
        indptr = _leaf_numpy(self.indptr, "to_dense")
        if indptr.ndim == 2:
            # Batched: one (rows+1,) pointer row per batch element; the
            # per-batch row/col ids repeat over a COMMON nnz (batched sparse
            # tensors are padded to one nnz per batch).
            nnz = indices.shape[1]
            batch_ids = np.broadcast_to(
                np.arange(indptr.shape[0])[:, None], (indptr.shape[0], nnz)
            )
            ids = np.stack(
                [
                    np.repeat(np.arange(indptr.shape[1] - 1), np.diff(ip))
                    for ip in indptr
                ],
                axis=0,
            )
        else:
            ids = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
        out = np.zeros(dense_shape, dtype=self.dtype)
        if self.format == "csr":
            if indptr.ndim == 2:
                np.add.at(out, (batch_ids, ids, indices), values)
            else:
                np.add.at(out, (ids, indices), values)
            return out
        # csc
        if indptr.ndim == 2:
            np.add.at(out, (batch_ids, indices, ids), values)
        else:
            np.add.at(out, (indices, ids), values)
        return out

    def to_coo(self):
        """Convert to canonical COO (row-major). Identity when already COO.

        CSR expands indptr to row ids; CSC expands indptr to column ids and
        re-sorts the rows lexicographically (``np.lexsort``) so the result is
        canonical row-major COO.
        """
        if self.format == "coo":
            return self
        indptr = _leaf_numpy(self.indptr, "to_coo")
        indices = _leaf_numpy(self.indices, "to_coo")
        values = _leaf_numpy(self.values, "to_coo")
        ids = np.repeat(np.arange(indptr.shape[0] - 1), np.diff(indptr))
        if self.format == "csr":
            # Row-major already: CSR segments are rows in order, columns
            # strictly increasing within each row.
            new_indices = np.stack([ids, indices], axis=1)
            return SparseTensor(new_indices, values, self.dense_shape, format="coo")
        # csc: ids are column ids; re-sort to row-major (row primary key).
        order = np.lexsort((ids, indices))
        new_indices = np.stack([indices[order], ids[order]], axis=1)
        return SparseTensor(new_indices, values[order], self.dense_shape, format="coo")

    def to_csr(self):
        """Convert to CSR. Identity when already CSR; CSC goes through COO.

        From canonical COO the indptr is built with ``np.bincount`` over the
        row indices — no re-sort is needed (canonical COO is row-major).
        """
        if self.format == "csr":
            return self
        if self.format == "csc":
            return self.to_coo().to_csr()
        indices = _leaf_numpy(self.indices, "to_csr")
        values = _leaf_numpy(self.values, "to_csr")
        rows = self.dense_shape[0]
        if isinstance(rows, core.Dim):
            rows = indices.shape[0]
        counts = np.bincount(indices[:, 0], minlength=rows)
        indptr = np.zeros(rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts)
        return CSRTensor(indptr, indices[:, 1], values, self.dense_shape)

    def to_csc(self):
        """Convert to CSC. Identity when already CSC; CSR goes through COO.

        From canonical COO the rows are re-sorted column-major
        (``np.lexsort``) and the indptr is built over the columns.
        """
        if self.format == "csc":
            return self
        if self.format == "csr":
            return self.to_coo().to_csc()
        indices = _leaf_numpy(self.indices, "to_csc")
        values = _leaf_numpy(self.values, "to_csc")
        cols = self.dense_shape[1]
        if isinstance(cols, core.Dim):
            cols = indices.shape[0]
        order = np.lexsort((indices[:, 0], indices[:, 1]))  # column primary
        sorted_indices = indices[order]
        counts = np.bincount(sorted_indices[:, 1], minlength=cols)
        indptr = np.zeros(cols + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(counts)
        return CSCTensor(indptr, sorted_indices[:, 0], values[order], self.dense_shape)


class CSRTensor(SparseTensor):
    """Concrete CSR sparse tensor: ``(indptr, indices, values)`` leaves.

    Rank-2 ``dense_shape (rows, cols)`` enforced; CSR canonical validation
    applies on construction (see :class:`SparseTensor`).
    """

    def __init__(self, indptr, indices, values, dense_shape):
        """Construct a concrete CSR sparse tensor with canonical validation.

        Args:
            indptr: ``(rows+1,)`` integer pointer array (normalized to int64).
            indices: ``(nnz,)`` per-row column indices (normalized to int64).
            values: ``(nnz,)`` stored values (any numeric dtype).
            dense_shape: Rank-2 ``(rows, cols)``.

        Raises:
            core.ShapeError: Canonical CSR violations.
            core.DTypeError: Non-integer indptr/indices or non-numeric values.
        """
        class_name = type(self).__name__
        dense_shape = _validate_dense_shape(dense_shape, class_name)
        leaves = _validate_csr(indptr, indices, values, dense_shape, class_name)
        self._leaves = tuple(leaves)
        self.dense_shape = dense_shape
        self.format = "csr"


class CSCTensor(SparseTensor):
    """Concrete CSC sparse tensor: ``(indptr, indices, values)`` leaves.

    Rank-2 ``dense_shape (rows, cols)`` enforced; CSC canonical validation
    applies on construction (see :class:`SparseTensor`).
    """

    def __init__(self, indptr, indices, values, dense_shape):
        """Construct a concrete CSC sparse tensor with canonical validation.

        Args:
            indptr: ``(cols+1,)`` integer pointer array (normalized to int64).
            indices: ``(nnz,)`` per-column row indices (normalized to int64).
            values: ``(nnz,)`` stored values (any numeric dtype).
            dense_shape: Rank-2 ``(rows, cols)``.

        Raises:
            core.ShapeError: Canonical CSC violations.
            core.DTypeError: Non-integer indptr/indices or non-numeric values.
        """
        class_name = type(self).__name__
        dense_shape = _validate_dense_shape(dense_shape, class_name)
        leaves = _validate_csc(indptr, indices, values, dense_shape, class_name)
        self._leaves = tuple(leaves)
        self.dense_shape = dense_shape
        self.format = "csc"


class SparseTensorSpec(SparseTensor):
    """Spec-phase sparse tensor: the leaves are ``core.TensorSpec``.

    Describes a future runtime sparse tensor (trace inputs). COO carries 2
    leaf specs ``(indices, values)``; CSR/CSC carry 3 ``(indptr, indices,
    values)``. ``dense_shape`` entries are positive ints (a ``core.Dim``
    allowed at position 0); the indptr spec shape must be STATIC
    ``(rows+1,)`` / ``(cols+1,)`` i64.

    Attributes:
        dtype: The values spec's dtype.
    """

    def __init__(self, *leaf_specs, dense_shape, format="coo"):
        """Construct a sparse spec with structural validation.

        Args:
            *leaf_specs: ``(indices_spec, values_spec)`` for COO;
                ``(indptr_spec, indices_spec, values_spec)`` for CSR/CSC —
                all ``core.TensorSpec``.
            dense_shape: The dense shape (positive ints; a ``core.Dim``
                allowed at position 0).
            format: ``"coo"`` | ``"csr"`` | ``"csc"``.

        Raises:
            core.ShapeError: Wrong leaf count/type or shape mismatches.
            core.DTypeError: Indices/indptr spec dtype not int64.
            ValueError: Unknown ``format``.
        """
        class_name = type(self).__name__
        if format not in _FORMATS:
            raise ValueError(
                f"{class_name}: unknown sparse format {format!r} — expected "
                f"one of {_FORMATS}"
            )
        dense_shape = _validate_dense_shape(dense_shape, class_name)
        if format == "coo":
            if len(leaf_specs) != 2:
                raise core.ShapeError(
                    f"{class_name}: COO requires 2 leaf specs (indices, values), "
                    f"got {len(leaf_specs)}"
                )
            indices_spec, values_spec = leaf_specs
            _require_tensorspec(indices_spec, class_name, "indices")
            _require_tensorspec(values_spec, class_name, "values")
            if indices_spec.shape != (None, len(dense_shape)):
                raise core.ShapeError(
                    f"{class_name}: COO indices spec must have shape "
                    f"(None, {len(dense_shape)}), got {indices_spec.shape}"
                )
            if indices_spec.dtype != _INT64:
                raise core.DTypeError(
                    f"{class_name}: COO indices spec dtype must be int64, got "
                    f"{indices_spec.dtype}"
                )
            if values_spec.shape != (None,):
                raise core.ShapeError(
                    f"{class_name}: COO values spec must have shape (None,), "
                    f"got {values_spec.shape}"
                )
            leaves = (indices_spec, values_spec)
        else:
            if len(dense_shape) != 2:
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} requires a rank-2 "
                    f"dense_shape (rows, cols), got rank {len(dense_shape)}"
                )
            if len(leaf_specs) != 3:
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} requires 3 leaf specs "
                    f"(indptr, indices, values), got {len(leaf_specs)}"
                )
            indptr_spec, indices_spec, values_spec = leaf_specs
            _require_tensorspec(indptr_spec, class_name, "indptr")
            _require_tensorspec(indices_spec, class_name, "indices")
            _require_tensorspec(values_spec, class_name, "values")
            ptr_len = dense_shape[0] if format == "csr" else dense_shape[1]
            if isinstance(ptr_len, core.Dim):
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} indptr spec shape must be "
                    f"STATIC ({ptr_len}+1,) — dense_shape[0] must be a concrete "
                    f"int for {format.upper()} specs"
                )
            if indptr_spec.shape != (ptr_len + 1,):
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} indptr spec must have "
                    f"shape ({ptr_len + 1},), got {indptr_spec.shape}"
                )
            if indptr_spec.dtype != _INT64:
                raise core.DTypeError(
                    f"{class_name}: {format.upper()} indptr spec dtype must be "
                    f"int64, got {indptr_spec.dtype}"
                )
            if indices_spec.shape != (None,):
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} indices spec must have "
                    f"shape (None,), got {indices_spec.shape}"
                )
            if indices_spec.dtype != _INT64:
                raise core.DTypeError(
                    f"{class_name}: {format.upper()} indices spec dtype must be "
                    f"int64, got {indices_spec.dtype}"
                )
            if values_spec.shape != (None,):
                raise core.ShapeError(
                    f"{class_name}: {format.upper()} values spec must have "
                    f"shape (None,), got {values_spec.shape}"
                )
            leaves = (indptr_spec, indices_spec, values_spec)
        self._leaves = leaves
        self.dense_shape = dense_shape
        self.format = format

    @classmethod
    def from_concrete(cls, sparse) -> "SparseTensorSpec":
        """Derive the canonical spec from a CONCRETE sparse instance.

        The pending ``etl.evaluate`` pipeline fix calls this to derive trace
        inputs from concrete sparse arguments. The concrete instance is
        flattened via ``core.flatten`` and each tensor leaf becomes a
        ``core.TensorSpec``: the nnz dim (dim 0 of the indices/values
        leaves) becomes runtime-dynamic (``None``), while the indptr leaf
        keeps its STATIC ``(rows+1,)`` / ``(cols+1,)`` shape. The static
        leaves (``dense_shape`` ints/Dims, the values ``np.dtype``, the
        format string) pass through unchanged, and ``core.unflatten``
        rebuilds the spec through the polymorphic pytree ``unflatten_fn``
        (so the result is always a validated :class:`SparseTensorSpec`).

        The result flattens EXACTLY back to the concrete instance's children
        layout — COO: ``[indices_spec, values_spec, *dense_shape, dtype,
        "coo"]``; CSR/CSC: ``[indptr_spec, indices_spec, values_spec,
        *dense_shape, dtype, format]`` — so the spec structure-matches the
        concrete instance leaf-for-leaf (static leaves
        ``[*dense_shape, dtype, format]`` identical).

        Args:
            sparse: A CONCRETE sparse tensor (COO/CSR/CSC with numpy or
                ``core.Tensor`` leaves).

        Returns:
            The derived :class:`SparseTensorSpec`.

        Raises:
            TypeError: ``sparse`` is not a concrete sparse tensor (a spec, a
                symbolic instance, or a non-sparse value).
            core.ShapeError: Batched concrete leaves (leaf shapes with more
                dims than the unbatched spec layout admits — v1 specs are
                unbatched).
        """
        if not is_sparse(sparse):
            raise TypeError(
                "SparseTensorSpec.from_concrete expects a concrete sparse "
                f"tensor (COO/CSR/CSC), got {type(sparse).__name__}"
            )
        if isinstance(sparse, cls):
            raise TypeError(
                "SparseTensorSpec.from_concrete expects a CONCRETE sparse "
                "tensor — a SparseTensorSpec is already a spec; pass a "
                "concrete COO/CSR/CSC instance instead"
            )
        if isinstance(sparse.values, core.SymbolicTensor):
            raise TypeError(
                "SparseTensorSpec.from_concrete cannot derive a spec from a "
                "symbolic sparse tensor (no concrete data) — pass a concrete "
                "COO/CSR/CSC instance"
            )
        children, tree = core.flatten(sparse)
        format = children[-1]
        n_leaves = 2 if format == "coo" else 3
        spec_children = []
        for index, child in enumerate(children):
            if index >= n_leaves:
                # Static leaves (dense_shape ints/Dims, values dtype, format)
                # pass through unchanged — snapshotted and run-validated by
                # the trace/pipeline machinery like any other static value.
                spec_children.append(child)
                continue
            arr = child.numpy() if isinstance(child, core.Tensor) else child
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    "SparseTensorSpec.from_concrete: sparse leaf "
                    f"{index} must be a numpy array or core.Tensor, got "
                    f"{type(child).__name__}"
                )
            if index == 0 and format != "coo":
                # indptr: STATIC (rows+1,)/(cols+1,) — nnz is not its dim.
                shape = tuple(arr.shape)
            else:
                # indices/values: the nnz dim (dim 0) becomes dynamic.
                shape = tuple(
                    None if dim_index == 0 else dim
                    for dim_index, dim in enumerate(arr.shape)
                )
            spec_children.append(core.TensorSpec(shape=shape, dtype=arr.dtype))
        result = core.unflatten(spec_children, tree)
        if not isinstance(result, cls):  # pragma: no cover - type-locked
            raise TypeError(
                "SparseTensorSpec.from_concrete: internal error — unflatten "
                f"produced {type(result).__name__}, expected {cls.__name__}"
            )
        return result


def is_sparse(x) -> bool:
    """True if ``x`` is a sparse tensor in ANY phase or variant.

    ``isinstance(x, SparseTensor)`` — catches the spec phase
    (:class:`SparseTensorSpec`), the symbolic phase, and the concrete phase
    (COO/CSR/CSC) via the shared base class.
    """
    return isinstance(x, SparseTensor)


# --- pytree registration (binding children layout, context always None) -----


def _flatten_sparse(x):
    """Pytree ``flatten_fn`` for the whole sparse hierarchy.

    Children layout (binding): ``[tensor leaves..., *dense_shape, dtype,
    format]`` — COO: ``[indices, values, *dense_shape, dtype, "coo"]``; CSR:
    ``[indptr, indices, values, *dense_shape, dtype, "csr"]``; CSC:
    ``[indptr, indices, values, *dense_shape, dtype, "csc"]``. One leaf per
    dense dim (plain ints, or the ``Dim`` object itself), then the values
    dtype as an ``np.dtype`` leaf, then the format string leaf. Context is
    always ``None`` (shape data are static leaves — snapshotted and
    run-validated by the pipeline machinery).
    """
    children = list(x._leaves)
    children.extend(x.dense_shape)
    children.append(x.dtype)
    children.append(x.format)
    return children, None


def _unflatten_sparse(context, children):
    """Pytree ``unflatten_fn`` — polymorphic on the first child's kind:

    ``core.TensorSpec`` → rebuild a :class:`SparseTensorSpec` (validated);
    ``core.SymbolicTensor`` → rebuild a symbolic instance via
    :meth:`SparseTensor.from_parts` (no validation); ``core.Tensor`` /
    ``np.ndarray`` → rebuild a concrete instance (canonical validation).
    The format leaf selects the variant class; the dtype leaf is checked
    against the values leaf when cheap.
    """
    if context is not None:
        raise ValueError(
            f"unflatten: sparse pytree context must be None, got {context!r}"
        )
    if not children:
        raise ValueError("unflatten: sparse pytree node has no children")
    format = children[-1]
    if format not in _FORMATS:
        raise ValueError(f"unflatten: unknown sparse format leaf {format!r}")
    n_leaves = 2 if format == "coo" else 3
    if len(children) < n_leaves + 3:
        raise ValueError(
            f"unflatten: sparse pytree node needs {n_leaves} leaves + "
            f"dense_shape + dtype + format children, got {len(children)}"
        )
    leaves = list(children[:n_leaves])
    rest = children[n_leaves:]
    dense_shape = tuple(rest[:-2])
    dtype_leaf = rest[-2]
    first = children[0]
    if isinstance(first, core.TensorSpec):
        result = SparseTensorSpec(*leaves, dense_shape=dense_shape, format=format)
    elif isinstance(first, core.SymbolicTensor):
        result = SparseTensor.from_parts(*leaves, dense_shape=dense_shape, format=format)
    elif isinstance(first, (core.Tensor, np.ndarray)):
        if any(isinstance(d, core.Dim) for d in dense_shape):
            # Batched/vmap output: dense_shape[0] is the batch Dim (its
            # extent is indices.shape[0]); the tensor leaves carry leading
            # batch dims ((B..., nnz, ndim) / (B..., nnz)), which the
            # per-element validating constructors reject. The interpreter
            # already applied canonical validation per batch element, so
            # rebuild validation-free via from_parts (same variant dispatch).
            result = SparseTensor.from_parts(
                *leaves, dense_shape=dense_shape, format=format
            )
        elif format == "coo":
            result = SparseTensor(*leaves, dense_shape=dense_shape, format="coo")
        elif format == "csr":
            result = CSRTensor(*leaves, dense_shape=dense_shape)
        else:
            result = CSCTensor(*leaves, dense_shape=dense_shape)
    else:
        raise TypeError(
            f"unflatten: cannot rebuild a sparse tensor from a "
            f"{type(first).__name__} leaf — expected core.TensorSpec (spec), "
            "core.SymbolicTensor (symbolic), or core.Tensor/numpy.ndarray "
            "(concrete)"
        )
    if result.dtype != dtype_leaf:
        raise core.DTypeError(
            f"unflatten: sparse dtype leaf {dtype_leaf!r} does not match the "
            f"values leaf dtype {result.dtype!r}"
        )
    return result


core.register_pytree_node(SparseTensor, _flatten_sparse, _unflatten_sparse)
