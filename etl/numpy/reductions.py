"""Reduction sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` reduce ops. numpy semantics mapped at trace time:
`axis=None` reduces ALL axes (expanded explicitly — rank is always known);
`dtype≠None` (sum/mean/prod/cumsum, per numpy) composes `ops.cast`;
`cumsum(axis=None)` flattens via reshape first. Concrete `Tensor` args raise
`TraceError`. Implemented: all functions forward to the frozen ops contract.
"""

from __future__ import annotations

import functools
import operator

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "sum", "mean", "prod", "max", "min", "argmax", "argmin", "cumsum",
]


def sum(a, axis=None, dtype=None, keepdims=False):  # noqa: A001 — numpy.sum
    """numpy.sum → ops.sum (reduce_sum).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    r = ops.sum(a, axes=axis, keepdims=keepdims)
    if dtype is not None:
        return ops.cast(r, dtype)
    return r


def mean(a, axis=None, dtype=None, keepdims=False):
    """numpy.mean → ops.mean (reduce_mean).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    r = ops.mean(a, axes=axis, keepdims=keepdims)
    if dtype is not None:
        return ops.cast(r, dtype)
    return r


def prod(a, axis=None, dtype=None, keepdims=False):
    """numpy.prod → ops.prod (reduce_prod).

    axis=None reduces all axes; dtype≠None composes ops.cast.
    """
    r = ops.prod(a, axes=axis, keepdims=keepdims)
    if dtype is not None:
        return ops.cast(r, dtype)
    return r


def max(a, axis=None, keepdims=False):  # noqa: A001 — shadows builtin by design
    """numpy.max → ops.max (reduce_max). axis=None reduces all axes."""
    return ops.max(a, axes=axis, keepdims=keepdims)


def min(a, axis=None, keepdims=False):  # noqa: A001 — shadows builtin by design
    """numpy.min → ops.min (reduce_min). axis=None reduces all axes."""
    return ops.min(a, axes=axis, keepdims=keepdims)


def argmax(a, axis=None, keepdims=False):
    """numpy.argmax → ops.argmax(a, axis=axis, keepdims=keepdims)."""
    return ops.argmax(a, axis=axis, keepdims=keepdims)


def argmin(a, axis=None, keepdims=False):
    """numpy.argmin → ops.argmin(a, axis=axis, keepdims=keepdims)."""
    return ops.argmin(a, axis=axis, keepdims=keepdims)


def cumsum(a, axis=None, dtype=None):
    """numpy.cumsum → ops.cumsum.

    axis=None flattens first (ops.reshape to 1-D), then ops.cumsum(axis=0);
    dtype≠None composes ops.cast.
    """
    if axis is None:
        numel = functools.reduce(operator.mul, a.shape, 1)
        x = ops.reshape(a, (numel,))
        r = ops.cumsum(x, axis=0)
    else:
        r = ops.cumsum(a, axis=axis)
    if dtype is not None:
        return ops.cast(r, dtype)
    return r
