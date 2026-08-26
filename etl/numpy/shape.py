"""Shape/manipulation sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` where an op exists; documented compositions where
numpy's semantics are richer than one op (`stack`, `split`, `expand_dims`,
`squeeze`, `transpose(axes=None)`). All shape computation happens at trace
time via `DimExpr` arithmetic (rank is always known). Concrete `Tensor` args
raise `TraceError`.

Implemented: `reshape`/`transpose`/`broadcast_to`/`concatenate`/`pad`
(constant mode)/`tril`/`triu` are 1:1 forwards to ops; `expand_dims`/
`squeeze`/`stack` compose `ops.reshape` + `ops.concatenate`; `split` composes
`ops.slice` with sections resolved statically at trace time.
"""

from __future__ import annotations

from .. import core  # ShapeError/TraceError — lower layer, allowed import
from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "reshape", "transpose", "broadcast_to", "expand_dims", "squeeze",
    "concatenate", "stack", "split", "pad", "tril", "triu",
]


def reshape(a, shape):
    """numpy.reshape → ops.reshape(a, shape).

    One `-1` entry allowed (numpy semantics), resolved at trace time via
    DimExpr arithmetic over the known rank. `order` kwarg unsupported in v1.
    """
    return ops.reshape(a, shape)


def transpose(a, axes=None):
    """numpy.transpose → ops.transpose(a, axes).

    axes=None → reversed axes (numpy default; rank known at trace time).
    """
    return ops.transpose(a, axes)


def broadcast_to(a, shape):
    """numpy.broadcast_to → ops.broadcast(a, shape)."""
    return ops.broadcast(a, shape)


def expand_dims(a, axis):
    """numpy.expand_dims → ops.reshape with size-1 dims inserted at axis.

    axis may be an int (negative normalized at trace time) or a tuple of
    ints; each tuple entry is normalized against the FINAL ndim
    (rank + len(tuple)), sorted, and inserted in ascending order.
    Repeated (duplicate) axes raise ShapeError (numpy parity).
    """
    rank = len(a.shape)
    if isinstance(axis, tuple):
        out_ndim = rank + len(axis)
        axes = sorted(ax + out_ndim if ax < 0 else ax for ax in axis)
    else:
        out_ndim = rank + 1
        axes = [axis + out_ndim if axis < 0 else axis]
    for ax in axes:
        if ax < 0 or ax >= out_ndim:
            raise core.ShapeError(
                f"enp.expand_dims: axis {ax} out of range for final ndim {out_ndim}"
            )
    for prev, ax in zip(axes, axes[1:]):
        if prev == ax:
            raise core.ShapeError(
                f"enp.expand_dims: repeated axis {ax} — each axis may be "
                "expanded only once"
            )
    new_shape = list(a.shape)
    for ax in axes:
        new_shape.insert(ax, 1)
    return ops.reshape(a, tuple(new_shape))


def squeeze(a, axis=None):
    """numpy.squeeze → ops.reshape with size-1 dims dropped.

    Dropped dims must be statically size-1 at trace time; an unknown-size
    (None/dynamic) dim → TraceError. axis=None drops all statically-size-1
    dims; an explicit axis that is not statically 1 → TraceError.
    """
    shape = tuple(a.shape)
    if axis is None:
        new_shape = tuple(d for d in shape if not (isinstance(d, int) and d == 1))
    else:
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            raise core.ShapeError(
                f"enp.squeeze: axis {axis} out of range for rank {len(shape)}"
            )
        if not (isinstance(shape[axis], int) and shape[axis] == 1):
            raise core.TraceError(
                f"enp.squeeze: dim {shape[axis]!r} on axis {axis} is not "
                "statically 1 (symbolic/unknown dims cannot be squeezed) — "
                "squeeze requires a statically-known size-1 dim at trace time"
            )
        new_shape = shape[:axis] + shape[axis + 1:]
    return ops.reshape(a, new_shape)


def concatenate(arrays, axis=0):
    """numpy.concatenate → ops.concatenate(arrays, axis=axis).

    arrays is a list/tuple of SymbolicTensors.
    """
    return ops.concatenate(arrays, axis=axis)


def stack(arrays, axis=0):
    """numpy.stack → expand_dims(each array, axis) + concatenate(axis)."""
    arrays = list(arrays)
    if not arrays:
        raise core.ShapeError("enp.stack: arrays must be a non-empty list/tuple")
    rank = len(arrays[0].shape)
    if axis < 0:
        axis += rank + 1
    if axis < 0 or axis > rank:
        raise core.ShapeError(f"enp.stack: axis {axis} out of range for rank {rank}")
    expanded = []
    for x in arrays:
        new_shape = x.shape[:axis] + (1,) + x.shape[axis:]
        expanded.append(ops.reshape(x, new_shape))
    return ops.concatenate(expanded, axis=axis)


def split(a, indices_or_sections, axis=0):
    """numpy.split → composition of ops.slice along axis.

    indices_or_sections may be an int (equal sections; count/divisibility
    resolved statically at trace time) or an explicit list of split indices.
    """
    shape = tuple(a.shape)
    rank = len(shape)
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise core.ShapeError(f"enp.split: axis {axis} out of range for rank {rank}")
    size = shape[axis]
    if not isinstance(size, int):
        raise core.TraceError(
            f"enp.split: dim {size!r} on axis {axis} is not a static int — "
            "split sections must be resolved at trace time"
        )
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            raise core.ShapeError(
                f"enp.split: number of sections must be positive, got {n}"
            )
        if size % n != 0:
            raise core.ShapeError(
                f"enp.split: axis {axis} has size {size}, which is not "
                f"divisible into {n} equal sections"
            )
        boundaries = [i * (size // n) for i in range(n + 1)]
    else:
        indices = list(indices_or_sections)
        prev = -1
        for idx in indices:
            if not isinstance(idx, int) or idx <= prev or idx < 0 or idx > size:
                raise core.ShapeError(
                    "enp.split: split indices must be strictly increasing "
                    f"ints within [0, {size}], got {indices!r}"
                )
            prev = idx
        boundaries = [0] + indices + [size]
    result = []
    for i in range(len(boundaries) - 1):
        start = tuple(boundaries[i] if j == axis else 0 for j in range(rank))
        lengths = tuple(
            (boundaries[i + 1] - boundaries[i]) if j == axis else d
            for j, d in enumerate(shape)
        )
        result.append(ops.slice(a, start, lengths))
    return result


def pad(a, pad_width, mode="constant", constant_values=0):
    """numpy.pad → ops.pad(...).

    v1: only mode="constant" (with constant_values); other modes raise
    NotImplementedError (deferred to v2 — see CONTEXT.md).

    pad_width forms: int (symmetric on all axes); per-axis sequence of ints
    or (before, after) pairs; length-1 (before, after) pair (broadcast to
    all axes); bare (before, after) pair on a rank-1 array (pads axis 0,
    numpy parity).
    """
    if mode != "constant":
        raise NotImplementedError(
            "enp.pad: only mode='constant' is implemented in v1; "
            f"mode={mode!r} is deferred to v2"
        )
    rank = len(a.shape)
    if isinstance(pad_width, int):
        config = (pad_width,) * rank
    else:
        try:
            entries = tuple(pad_width)
        except TypeError:
            raise core.ShapeError(
                "enp.pad: pad_width must be an int or a sequence of ints / "
                f"(before, after) pairs, got {pad_width!r}"
            )
        if len(entries) == 1 and _is_pair(entries[0]):
            config = (entries[0],) * rank
        elif len(entries) == rank:
            config = entries
        elif rank == 1 and _is_pair(entries):
            # numpy parity: a bare (before, after) pair on a rank-1 array
            # pads the sole axis.
            config = (entries,)
        else:
            raise core.ShapeError(
                f"enp.pad: pad_width has {len(entries)} entries but rank is "
                f"{rank} (a length-1 (before, after) pair broadcasts to all axes)"
            )
    # Validate and canonicalize entries: int (symmetric) or (before, after)
    # pair of non-negative ints.
    normalized = []
    for entry in config:
        if isinstance(entry, int):
            if entry < 0:
                raise core.ShapeError(f"enp.pad: negative padding {entry}")
            normalized.append(entry)
        elif _is_pair(entry):
            if entry[0] < 0 or entry[1] < 0:
                raise core.ShapeError(f"enp.pad: negative padding {entry}")
            normalized.append((entry[0], entry[1]))
        else:
            raise core.ShapeError(
                f"enp.pad: malformed pad entry {entry!r} — expected an int or "
                "a (before, after) pair of non-negative ints"
            )
    return ops.pad(a, tuple(normalized), value=constant_values)


def tril(a, k=0):
    """numpy.tril → ops.tril(a, k=k)."""
    return ops.tril(a, k=k)


def triu(a, k=0):
    """numpy.triu → ops.triu(a, k=k)."""
    return ops.triu(a, k=k)


def _is_pair(entry):
    """True if entry is a (before, after) pair of two ints."""
    return (
        isinstance(entry, (tuple, list))
        and len(entry) == 2
        and isinstance(entry[0], int)
        and isinstance(entry[1], int)
    )
