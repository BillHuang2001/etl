"""Indexing and shape-manipulation ops.

NOTE: this module shadows the builtin ``slice`` with its op function; use
``builtins.slice`` when a Python slice object is needed inside this module.

All functions follow the unified semantics documented in this node's
``CONTEXT.md`` (operand normalization via ``_utils.as_operand``, active
builder, call-site ``Location``). Category-specific rules:

- ``slice``/``gather``/``scatter``/``concatenate``/``pad``/``transpose``/
  ``reshape``/``broadcast`` produce NEW SSA values — etl has no in-place
  tensor mutation.
- All index/shape parameters are STATIC Python values (ints, slices, tuples
  of ints, ``Dim``/``DimExpr`` where a symbolic extent is meaningful —
  e.g. ``reshape`` output dims, ``broadcast`` target dims, ``slice``
  ``lengths``). A ``SymbolicTensor`` index anywhere raises ``TraceError``.
- Output shapes are computed statically via ``DimExpr`` arithmetic; the
  numpy backend enforces exact runtime semantics.
"""
from __future__ import annotations

import builtins
from typing import Tuple

import numpy as np

from etl import core

from . import _utils

__all__ = [
    "broadcast", "reshape", "transpose", "slice", "gather", "scatter",
    "concatenate", "pad", "getitem",
]


def _is_int(value) -> bool:
    """True for plain ints (bools excluded — numpy-style tag checks)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _wrap(op, location) -> "core.SymbolicTensor":
    """Wrap an op's single IR result value in a ``SymbolicTensor``.

    The result dtype/shape are READ BACK from the IR value type (the opdef
    shape_fn is the single source of inference truth).
    """
    result_type = op.result.type
    return core.SymbolicTensor(
        value=op.result,
        dtype=result_type.dtype,
        shape=result_type.shape,
        location=location,
    )


def _per_axis(value, rank: int, name: str, *, symbolic: bool) -> Tuple:
    """Normalize a scalar-or-per-axis-tuple parameter to a length-``rank`` tuple.

    A bare int (or ``Dim``/``DimExpr`` when ``symbolic``) is replicated over
    all axes; a tuple must have exactly ``rank`` entries.

    Raises:
        core.ShapeError: tuple arity mismatch.
        TypeError: ``value`` is neither a scalar nor a tuple of the accepted
            kinds.
    """
    if _is_int(value) or (symbolic and isinstance(value, (core.Dim, core.DimExpr))):
        return (value,) * rank
    if isinstance(value, tuple):
        if len(value) != rank:
            raise core.ShapeError(
                f"{name}: expected {rank} entries, got {len(value)}"
            )
        return value
    kinds = "int/Dim/DimExpr" if symbolic else "int"
    raise TypeError(
        f"{name} must be an {kinds} or a tuple of {rank} entries, got "
        f"{type(value).__name__}"
    )


def _check_shape(shape, op_name: str, *, allow_wildcard: bool) -> Tuple:
    """Validate a target-shape tuple: ``int``/``Dim``/``DimExpr`` entries.

    A ``-1`` wildcard is accepted only when ``allow_wildcard``; negative ints
    other than ``-1`` are always rejected.

    Raises:
        core.ShapeError: negative int dim (other than an allowed ``-1``).
        TypeError: ``shape`` is not a tuple, or an entry is not a shape
            element.
    """
    if not isinstance(shape, tuple):
        raise TypeError(
            f"{op_name}: shape must be a tuple of shape elements, got "
            f"{type(shape).__name__}"
        )
    for d in shape:
        if _is_int(d):
            if d < -1 or (not allow_wildcard and d == -1):
                raise core.ShapeError(f"{op_name}: invalid shape dim {d!r}")
        elif not isinstance(d, (core.Dim, core.DimExpr)):
            raise TypeError(
                f"{op_name}: shape entries must be int/Dim/DimExpr, got {d!r}"
            )
    return shape


def broadcast(x, shape) -> "core.SymbolicTensor":
    """Broadcast to the target shape (numpy ``broadcast_to`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        shape: Tuple of ``int``/``Dim``/``DimExpr``. Target rank may exceed
            ``x``'s rank (new dims prepended). Per aligned dim: ``x`` dim of
            ``1`` expands to the target; equal dims stay; otherwise
            ``core.ShapeError`` (static) — symbolic conflicts defer to
            ``DimExpr`` equality and runtime enforcement.

    Returns:
        ``SymbolicTensor`` of the target shape; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: static incompatibility with the target shape.
        TypeError: ``shape`` is not a tuple of shape elements.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    shape = _check_shape(shape, "broadcast", allow_wildcard=False)
    # Static target-rank/broadcast compatibility is enforced by the IR
    # shape_fn (infer_broadcast_to): target rank < input rank and static
    # int-vs-int incompatibilities raise ShapeError at creation time.
    op = builder.create(
        "broadcast", operands=(x.value,), attributes={"shape": shape},
        location=loc,
    )
    return _wrap(op, loc)


def reshape(x, shape) -> "core.SymbolicTensor":
    """Reshape to the given shape.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        shape: Tuple of ``int``/``Dim``/``DimExpr``; at most ONE element may
            be ``-1`` (inferred from the element count via ``DimExpr``
            arithmetic).

    Returns:
        ``SymbolicTensor`` with the given (or inferred) shape; dtype
        preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: more than one ``-1``; static element-count
            mismatch; negative dims other than ``-1``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    shape = _check_shape(shape, "reshape", allow_wildcard=True)
    # Wildcard count / element-count agreement are enforced by the IR
    # shape_fn (infer_reshape); a -1 wildcard that cannot be decided
    # statically (dynamic or non-dividing symbolic counts) yields a
    # runtime-dynamic dim, which SymbolicTensor cannot carry — fail clearly.
    op = builder.create(
        "reshape", operands=(x.value,), attributes={"shape": shape},
        location=loc,
    )
    result = _wrap(op, loc)
    if any(d is None for d in result.shape):
        raise core.ShapeError(
            "reshape: cannot infer the -1 dim statically (runtime-dynamic or "
            "non-dividing symbolic element counts); give an explicit shape"
        )
    return result


def transpose(x, axes=None) -> "core.SymbolicTensor":
    """Permute tensor axes (numpy ``transpose`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axes: ``None`` (reverse all axes) or a permutation tuple of
            ``len(x.shape)`` ints.

    Returns:
        ``SymbolicTensor`` with axes permuted; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: ``axes`` is not a valid permutation.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if axes is None:
        pass  # None = full reversal (the IR default)
    elif isinstance(axes, tuple):
        if any(not _is_int(a) for a in axes):
            raise TypeError(
                f"transpose: axes entries must be ints, got {axes!r}"
            )
    else:
        raise TypeError(
            "transpose: axes must be None or a tuple of ints, got "
            f"{type(axes).__name__}"
        )
    # Permutation validity (arity + one-of-each) is enforced by the IR
    # shape_fn (infer_transpose) with ShapeError.
    op = builder.create(
        "transpose", operands=(x.value,), attributes={"permutation": axes},
        location=loc,
    )
    return _wrap(op, loc)


def slice(x, start, lengths, strides=1) -> "core.SymbolicTensor":
    """Static strided slice (Nx ``slice`` semantics).

    Args:
        x: ``SymbolicTensor``.
        start: int or per-axis tuple of ints (scalar broadcast to all axes).
        lengths: int/``Dim``/``DimExpr`` or per-axis tuple thereof.
        strides: int or per-axis tuple of ints; must be positive.

    Returns:
        ``SymbolicTensor`` of shape ``lengths`` (dims as given); dtype
        preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: static out-of-bounds; negative or zero strides;
            arity mismatch.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    rank = len(x.shape)
    starts = _per_axis(start, rank, "slice start", symbolic=False)
    length_entries = _per_axis(lengths, rank, "slice lengths", symbolic=True)
    stride_entries = _per_axis(strides, rank, "slice strides", symbolic=False)

    start_indices = []
    limit_indices = []
    for s, length, stride, dim in zip(
        starts, length_entries, stride_entries, x.shape
    ):
        if not _is_int(s):
            raise TypeError(f"slice: start entries must be ints, got {s!r}")
        if isinstance(length, (core.Dim, core.DimExpr)):
            raise core.ShapeError(
                f"slice: symbolic length {length!r} is not expressible — the "
                "IR slice op requires static integer limit_indices; use a "
                "static int length or gather with explicit index arrays"
            )
        if not _is_int(length) or length < 0:
            raise core.ShapeError(
                f"slice: lengths must be non-negative ints, got {length!r}"
            )
        if not _is_int(stride) or stride < 1:
            raise core.ShapeError(
                f"slice: strides must be positive ints, got {stride!r}"
            )
        if s < 0:
            # Nx semantics: negative starts count from the end of the axis.
            if not _is_int(dim):
                raise core.ShapeError(
                    f"slice: negative start {s} over symbolic dim {dim!r} is "
                    "not expressible in the IR slice op; use a non-negative "
                    "start or gather with explicit index arrays"
                )
            s = dim + s
            if s < 0:
                raise core.ShapeError(
                    f"slice: start index out of bounds for dim {dim}"
                )
        if _is_int(dim):
            if s > dim:
                raise core.ShapeError(f"slice: start {s} exceeds dim {dim}")
            limit = s + length
            if limit > dim:
                raise core.ShapeError(f"slice: limit {limit} exceeds dim {dim}")
        else:
            # Symbolic dim with static int start/length: the limit is a plain
            # int (runtime enforces bounds). limit_indices must never be None.
            limit = s + length
        start_indices.append(s)
        limit_indices.append(limit)

    op = builder.create(
        "slice",
        operands=(x.value,),
        attributes={
            "start_indices": tuple(start_indices),
            "limit_indices": tuple(limit_indices),
            "strides": stride_entries,
        },
        location=loc,
    )
    return _wrap(op, loc)


def gather(x, indices, axis=0) -> "core.SymbolicTensor":
    """Gather entries along an axis (numpy ``take`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        indices: ``SymbolicTensor`` of integer dtype (int32/int64), arbitrary
            shape.
        axis: int; the axis of ``x`` being indexed.

    Returns:
        ``SymbolicTensor`` with shape ``x.shape[:axis] + indices.shape +
        x.shape[axis + 1:]``; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand;
            symbolic indices.
        core.DTypeError: ``indices`` is not an integer dtype.
        core.ShapeError: ``axis`` out of range.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    indices = _utils.as_operand(indices, location=loc)
    if np.dtype(indices.dtype).kind not in "iu":
        raise core.DTypeError(
            "gather: indices must be an integer dtype (int32/int64), got "
            f"{indices.dtype}"
        )
    rank = len(x.shape)
    if not _is_int(axis):
        raise TypeError(f"gather: axis must be an int, got {axis!r}")
    original = axis
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        raise core.ShapeError(
            f"gather: axis {original} out of range for rank {rank}"
        )
    op = builder.create(
        "gather", operands=(x.value, indices.value),
        attributes={"axes": (axis,)}, location=loc,
    )
    return _wrap(op, loc)


def scatter(x, indices, updates, axis=0) -> "core.SymbolicTensor":
    """Scatter ``updates`` into a COPY of ``x`` at ``indices`` along an axis
    (numpy ``put``-along-axis / JAX ``scatter``-update semantics; ``x`` is
    never mutated).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        indices: ``SymbolicTensor`` of integer dtype.
        updates: ``SymbolicTensor`` (or Python scalar), cast to ``x.dtype``.
        axis: int; the axis being scattered along.

    Returns:
        ``SymbolicTensor`` with ``x``'s exact shape and dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.DTypeError: ``indices`` is not an integer dtype.
        core.ShapeError: ``axis`` out of range; static shape incompatibility
            between ``indices`` and ``updates``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    indices = _utils.as_operand(indices, location=loc)
    if np.dtype(indices.dtype).kind not in "iu":
        raise core.DTypeError(
            "scatter: indices must be an integer dtype (int32/int64), got "
            f"{indices.dtype}"
        )
    updates = _utils.as_operand(updates, location=loc)
    rank = len(x.shape)
    if not _is_int(axis):
        raise TypeError(f"scatter: axis must be an int, got {axis!r}")
    original = axis
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        raise core.ShapeError(
            f"scatter: axis {original} out of range for rank {rank}"
        )
    if updates.dtype != x.dtype:
        # Explicit composition: cast updates to the target dtype.
        cast_op = builder.create(
            "cast", operands=(updates.value,),
            attributes={"dtype": x.dtype}, location=loc,
        )
        updates = _wrap(cast_op, loc)
    expected = x.shape[:axis] + indices.shape + x.shape[axis + 1:]
    if len(updates.shape) != len(expected):
        raise core.ShapeError(
            f"scatter: updates rank {len(updates.shape)} does not match the "
            f"expected rank {len(expected)} "
            "(x.shape[:axis] + indices.shape + x.shape[axis + 1:])"
        )
    for actual, wanted in zip(updates.shape, expected):
        if actual == wanted or actual is None or wanted is None:
            continue
        if _is_int(actual) and _is_int(wanted):
            raise core.ShapeError(
                f"scatter: updates dim {actual} does not match the expected "
                f"dim {wanted}"
            )
        # Symbolic mismatch defers to the runtime backend.
    op = builder.create(
        "scatter",
        operands=(x.value, indices.value, updates.value),
        attributes={"axis": axis},
        location=loc,
    )
    return _wrap(op, loc)


def concatenate(tensors, axis=0) -> "core.SymbolicTensor":
    """Concatenate tensors along an axis (numpy ``concatenate`` semantics).

    Args:
        tensors: Non-empty list/tuple of ``SymbolicTensor`` (or Python
            scalars, treated as 0-d) of equal rank.
        axis: int; the axis along which to join.

    Returns:
        ``SymbolicTensor`` with the ``axis`` dim equal to the ``DimExpr`` SUM
        of input axis dims; other dims unchanged. Dtype = ``promote_dtypes``
        of all inputs.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: empty input; rank mismatch; static mismatch on a
            non-axis dim; ``axis`` out of range.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(
            "concatenate: tensors must be a non-empty list/tuple, got "
            f"{type(tensors).__name__}"
        )
    if not tensors:
        raise core.ShapeError("concatenate: expected at least one tensor")
    if not _is_int(axis):
        raise TypeError(f"concatenate: axis must be an int, got {axis!r}")
    operands = [_utils.as_operand(t, location=loc) for t in tensors]
    # Rank/axis/dim agreement and dtype promotion are enforced by the IR
    # shape_fn (infer_concatenate) with ShapeError.
    op = builder.create(
        "concatenate",
        operands=tuple(t.value for t in operands),
        attributes={"axis": axis},
        location=loc,
    )
    return _wrap(op, loc)


def pad(x, config, value=0) -> "core.SymbolicTensor":
    """Pad a tensor with a constant value (Nx ``pad`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        config: Per-axis padding spec: a tuple of length ``rank`` whose
            entries are either ``int`` (symmetric pad) or ``(before, after)``
            pairs of non-negative ints.
        value: Python scalar (cast to ``x.dtype``); the fill value.

    Returns:
        ``SymbolicTensor`` with each dim ``d + before + after`` (``DimExpr``
        arithmetic); dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: malformed config; negative padding; arity mismatch.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    rank = len(x.shape)
    if not isinstance(config, (tuple, list)):
        raise core.ShapeError(
            f"pad: config must be a tuple of {rank} entries, got "
            f"{type(config).__name__}"
        )
    if len(config) != rank:
        raise core.ShapeError(
            f"pad: config must have {rank} entries (one per axis), got "
            f"{len(config)}"
        )
    pairs = []
    for entry in config:
        if _is_int(entry):
            lo = hi = entry  # symmetric pad, normalized to a pair
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and all(_is_int(v) for v in entry)
        ):
            lo, hi = entry
        else:
            raise core.ShapeError(
                f"pad: invalid padding entry {entry!r} "
                "(expected int or (lo, hi) pair)"
            )
        if lo < 0 or hi < 0:
            raise core.ShapeError(f"pad: negative padding ({lo}, {hi})")
        pairs.append((lo, hi))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"pad: value must be a Python int/float scalar, got {value!r}"
        )
    op = builder.create(
        "pad", operands=(x.value,),
        attributes={"padding_config": tuple(pairs), "value": value},
        location=loc,
    )
    return _wrap(op, loc)


def _index_constant(builder, array, location) -> "core.SymbolicTensor":
    """Build a Constant op holding ``array`` (int64 gather indices) and wrap it."""
    # NOTE: np.ascontiguousarray would promote 0-d arrays to shape (1,) —
    # np.asarray preserves 0-d (a scalar index must drop the gathered axis).
    payload = np.asarray(array, dtype=np.int64)
    op = builder.create(
        "constant", attributes={"value": payload}, location=location
    )
    return _wrap(op, location)


def getitem(x, key) -> "core.SymbolicTensor":
    """``x[key]`` — static indexing entry point (operator handler kind
    ``getitem``, registered by ``_registration``).

    Strictly STATIC indexing, mapped onto ``slice``/``gather`` ops:

    - ``int`` index → ``gather`` op that drops the axis.
    - ``slice`` object (contiguous) → ``slice`` op.
    - tuple of ints/slices → per-axis combination of the above.
    - strided slices → ``gather`` with explicit index arrays.

    NOT supported (raises ``core.TraceError``): ``SymbolicTensor`` indices
    (runtime control flow on indexing is graph semantics — use ``etl.cond``
    explicitly), boolean masks, ``None``/newaxis, ellipsis.

    Args:
        x: ``SymbolicTensor``.
        key: int, ``builtins.slice``, or tuple of ints/slices.

    Returns:
        ``SymbolicTensor``; dtype preserved.

    Raises:
        core.TraceError: no active trace; symbolic index; unsupported key.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    rank = len(x.shape)

    if isinstance(key, tuple):
        entries = key
    elif isinstance(key, builtins.slice) or _is_int(key):
        entries = (key,)
    else:
        raise core.TraceError(
            f"getitem: unsupported index key {key!r}; supported: static int, "
            "builtins.slice, or a tuple of ints/slices (no boolean masks, "
            "None/newaxis, ellipsis, or symbolic indices)"
        )
    if len(entries) > rank:
        raise core.ShapeError(
            f"getitem: too many indices ({len(entries)}) for rank {rank}"
        )

    result = x
    contiguous = {}  # original axis -> (start, length) for the slice op
    # Int indices and strided slices become gather ops, processed in
    # DESCENDING axis order: a 0-d index gather removes one axis and shifts
    # every higher axis down (positions below are untouched), so gathering
    # from the top keeps all unprocessed positions stable; a 1-d index gather
    # replaces one axis and shifts nothing.
    for axis in range(len(entries) - 1, -1, -1):
        entry = entries[axis]
        dim = x.shape[axis]
        if isinstance(entry, bool):
            raise core.TraceError(
                "getitem: boolean masks are not supported (static indexing "
                "only)"
            )
        if entry is None or entry is Ellipsis:
            raise core.TraceError(
                f"getitem: {entry!r} in the index key is not supported "
                "(None/newaxis and ellipsis are not static indexing)"
            )
        if isinstance(entry, core.SymbolicTensor):
            raise core.TraceError(
                "getitem: symbolic (runtime) indices are not supported — "
                "runtime control flow on indexing is graph semantics; use "
                "etl.cond explicitly"
            )
        if _is_int(entry):
            index = entry
            if index < 0:
                if not _is_int(dim):
                    raise core.ShapeError(
                        f"getitem: negative index {index} over symbolic dim "
                        f"{dim!r} is not expressible"
                    )
                index += dim
            if _is_int(dim) and not 0 <= index < dim:
                raise core.ShapeError(
                    f"getitem: index {entry} out of range for dim {dim}"
                )
            result = gather(
                result,
                _index_constant(builder, np.asarray(index), loc),
                axis=axis,
            )
        elif isinstance(entry, builtins.slice):
            step = entry.step
            if step is not None and not _is_int(step):
                raise core.TraceError(
                    f"getitem: slice step must be an int or None, got {step!r}"
                )
            if step in (None, 1):
                start = 0 if entry.start is None else entry.start
                stop = entry.stop
                if not _is_int(start) or (stop is not None and not _is_int(stop)):
                    raise core.TraceError(
                        "getitem: slice bounds must be static ints (or None), "
                        f"got {entry!r}"
                    )
                if stop is None and start == 0:
                    continue  # full-axis no-op — leave the axis untouched
                if _is_int(dim):
                    # numpy slicing semantics: out-of-bounds bounds clamp
                    # (x[1:10] == x[1:]) — the slice op below is in-bounds.
                    if start < 0:
                        start += dim
                    if start < 0:
                        start = 0
                    if stop is not None and stop < 0:
                        stop += dim
                    if stop is not None and stop < 0:
                        stop = 0
                    if start > dim:
                        start = dim
                    stop = dim if stop is None else stop
                    if stop > dim:
                        stop = dim
                else:
                    # Symbolic dim: bounds must be static non-negative ints
                    # (negative bounds need dim + bound, which is not
                    # expressible; the runtime clamps out-of-bounds bounds).
                    if start < 0 or (stop is not None and stop < 0):
                        raise core.TraceError(
                            f"getitem: negative slice bound over symbolic dim "
                            f"{dim!r} is not expressible"
                        )
                    if stop is None:
                        raise core.TraceError(
                            f"getitem: full-axis slice (stop=None) over "
                            f"symbolic dim {dim!r} is not expressible in the "
                            "IR slice op"
                        )
                length = stop - start
                if length < 0:
                    length = 0  # numpy: reversed bounds yield an empty slice
                contiguous[axis] = (start, length)
            else:
                # Strided slice → gather with numpy's exact index array.
                if not _is_int(dim):
                    raise core.TraceError(
                        f"getitem: strided slice over symbolic dim {dim!r} is "
                        "not expressible (cannot build a static index array)"
                    )
                result = gather(
                    result,
                    _index_constant(builder, np.arange(dim)[entry], loc),
                    axis=axis,
                )
        else:
            raise core.TraceError(
                f"getitem: unsupported index key element {entry!r}; supported: "
                "static int / builtins.slice"
            )

    if contiguous:
        # One combined slice op over the post-gather tensor: contiguous
        # entries slice their axes; every other axis is a full slice over its
        # current dim (strided-gathered axes hold static index dims).
        starts = []
        lengths = []
        shift = 0  # axes removed by 0-d index gathers below the current axis
        for axis in range(rank):
            entry = entries[axis] if axis < len(entries) else None
            if _is_int(entry):
                shift += 1
                continue  # axis was dropped by an index gather
            if axis in contiguous:
                start, length = contiguous[axis]
                starts.append(start)
                lengths.append(length)
                continue
            dim = result.shape[axis - shift]
            if not _is_int(dim):
                raise core.TraceError(
                    f"getitem: cannot express a full-axis slice over symbolic "
                    f"dim {dim!r} in the IR slice op (limit_indices must be "
                    "static ints); use gather with explicit index arrays"
                )
            starts.append(0)
            lengths.append(dim)
        result = slice(result, tuple(starts), tuple(lengths), strides=1)
    return result
