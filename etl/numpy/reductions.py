"""Reduction sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` reduce ops. numpy semantics mapped at trace time:
`axis=None` reduces ALL axes (expanded explicitly — rank is always known);
`dtype≠None` (sum/mean/prod/cumsum, per numpy) composes `ops.cast`;
`cumsum(axis=None)` flattens via reshape first. Concrete `Tensor` args raise
`TraceError`. Architecture phase: stub bodies raise NotImplementedError.
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "sum", "mean", "prod", "max", "min", "argmax", "argmin", "cumsum",
]


def sum(a, axis=None, dtype=None, keepdims=False):  # noqa: A001 — numpy.sum
    """numpy.sum → ops.sum (reduce_sum).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    raise NotImplementedError(
        "enp.sum: architecture stub — maps to ops.sum "
        "(axis=None → all axes; dtype≠None → + ops.cast)"
    )


def mean(a, axis=None, dtype=None, keepdims=False):
    """numpy.mean → ops.mean (reduce_mean).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    raise NotImplementedError(
        "enp.mean: architecture stub — maps to ops.mean "
        "(axis=None → all axes; dtype≠None → + ops.cast)"
    )


def prod(a, axis=None, dtype=None, keepdims=False):
    """numpy.prod → ops.prod (reduce_prod).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    raise NotImplementedError(
        "enp.prod: architecture stub — maps to ops.prod "
        "(axis=None → all axes; dtype≠None → + ops.cast)"
    )


def max(a, axis=None, keepdims=False):  # noqa: A001 — shadows builtin by design
    """numpy.max → ops.max (reduce_max). axis=None reduces all axes."""
    raise NotImplementedError(
        "enp.max: architecture stub — maps to ops.max (axis=None → all axes)"
    )


def min(a, axis=None, keepdims=False):  # noqa: A001 — shadows builtin by design
    """numpy.min → ops.min (reduce_min). axis=None reduces all axes."""
    raise NotImplementedError(
        "enp.min: architecture stub — maps to ops.min (axis=None → all axes)"
    )


def argmax(a, axis=None, keepdims=False):
    """numpy.argmax → ops.argmax(a, axis=axis, keepdims=keepdims)."""
    raise NotImplementedError("enp.argmax: architecture stub — maps to ops.argmax")


def argmin(a, axis=None, keepdims=False):
    """numpy.argmin → ops.argmin(a, axis=axis, keepdims=keepdims)."""
    raise NotImplementedError("enp.argmin: architecture stub — maps to ops.argmin")


def cumsum(a, axis=None, dtype=None):
    """numpy.cumsum → ops.cumsum.

    axis=None flattens first (ops.reshape to 1-D), then ops.cumsum(axis=0);
    dtype≠None composes ops.cast. NOTE: ops.cumsum is not yet in the etl/ops
    public contract (see CONTEXT.md conflicts) — stays NotImplementedError
    until ops publishes it.
    """
    raise NotImplementedError(
        "enp.cumsum: architecture stub — maps to ops.cumsum "
        "(axis=None → flatten via reshape first)"
    )
