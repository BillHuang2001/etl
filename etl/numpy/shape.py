"""Shape/manipulation sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` where an op exists; documented compositions where
numpy's semantics are richer than one op (`stack`, `split`, `expand_dims`,
`squeeze`, `transpose(axes=None)`). All shape computation happens at trace
time via `DimExpr` arithmetic (rank is always known). Concrete `Tensor` args
raise `TraceError`. Architecture phase: stub bodies raise NotImplementedError.
"""

from __future__ import annotations

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
    raise NotImplementedError("enp.reshape: architecture stub — maps to ops.reshape")


def transpose(a, axes=None):
    """numpy.transpose → ops.transpose(a, axes).

    axes=None → reversed axes (numpy default; rank known at trace time).
    """
    raise NotImplementedError(
        "enp.transpose: architecture stub — maps to ops.transpose "
        "(axes=None reverses all axes)"
    )


def broadcast_to(a, shape):
    """numpy.broadcast_to → ops.broadcast(a, shape)."""
    raise NotImplementedError(
        "enp.broadcast_to: architecture stub — maps to ops.broadcast"
    )


def expand_dims(a, axis):
    """numpy.expand_dims → ops.reshape with size-1 dims inserted at axis.

    axis may be an int (negative normalized at trace time) or a tuple of
    ints (repeated expansion in ascending order).
    """
    raise NotImplementedError(
        "enp.expand_dims: architecture stub — maps to ops.reshape "
        "with inserted 1-dims"
    )


def squeeze(a, axis=None):
    """numpy.squeeze → ops.reshape with size-1 dims dropped.

    Dropped dims must be statically size-1 at trace time; an unknown-size
    (None/dynamic) dim → TraceError. axis=None drops all statically-size-1
    dims; an explicit axis that is not statically 1 → TraceError.
    """
    raise NotImplementedError(
        "enp.squeeze: architecture stub — maps to ops.reshape "
        "with size-1 dims dropped (static check at trace time)"
    )


def concatenate(arrays, axis=0):
    """numpy.concatenate → ops.concatenate(arrays, axis=axis).

    arrays is a list/tuple of SymbolicTensors.
    """
    raise NotImplementedError(
        "enp.concatenate: architecture stub — maps to ops.concatenate"
    )


def stack(arrays, axis=0):
    """numpy.stack → expand_dims(each array, axis) + concatenate(axis)."""
    raise NotImplementedError(
        "enp.stack: architecture stub — maps to "
        "expand_dims(each, axis) + concatenate(axis)"
    )


def split(a, indices_or_sections, axis=0):
    """numpy.split → composition of ops.slice along axis.

    indices_or_sections may be an int (equal sections; count/divisibility
    resolved statically at trace time) or an explicit list of split indices.
    """
    raise NotImplementedError(
        "enp.split: architecture stub — maps to a composition of ops.slice "
        "(sections resolved at trace time)"
    )


def pad(a, pad_width, mode="constant", constant_values=0):
    """numpy.pad → ops.pad(...).

    v1: only mode="constant" (with constant_values); other modes raise
    NotImplementedError (deferred to v2 — see CONTEXT.md).
    """
    raise NotImplementedError(
        "enp.pad: architecture stub — maps to ops.pad "
        "(v1: constant mode only)"
    )


def tril(a, k=0):
    """numpy.tril → ops.tril(a, k=k).

    NOTE: ops.tril is not yet in the etl/ops public contract (see CONTEXT.md
    conflicts) — stays NotImplementedError until ops publishes it.
    """
    raise NotImplementedError("enp.tril: architecture stub — maps to ops.tril")


def triu(a, k=0):
    """numpy.triu → ops.triu(a, k=k).

    NOTE: ops.triu is not yet in the etl/ops public contract (see CONTEXT.md
    conflicts) — stays NotImplementedError until ops publishes it.
    """
    raise NotImplementedError("enp.triu: architecture stub — maps to ops.triu")
