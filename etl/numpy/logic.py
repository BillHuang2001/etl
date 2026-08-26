"""Comparison and logical sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` (same IR as the mapped ops calls; concrete `Tensor`
args raise `TraceError`). `where` is a documented rename/alias of
`ops.select`. Architecture phase: stub bodies raise NotImplementedError.
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "where",
]


def equal(a, b):
    """numpy.equal → ops.equal(a, b)."""
    raise NotImplementedError("enp.equal: architecture stub — maps to ops.equal")


def not_equal(a, b):
    """numpy.not_equal → ops.not_equal(a, b)."""
    raise NotImplementedError(
        "enp.not_equal: architecture stub — maps to ops.not_equal"
    )


def less(a, b):
    """numpy.less → ops.less(a, b)."""
    raise NotImplementedError("enp.less: architecture stub — maps to ops.less")


def less_equal(a, b):
    """numpy.less_equal → ops.less_equal(a, b)."""
    raise NotImplementedError(
        "enp.less_equal: architecture stub — maps to ops.less_equal"
    )


def greater(a, b):
    """numpy.greater → ops.greater(a, b)."""
    raise NotImplementedError("enp.greater: architecture stub — maps to ops.greater")


def greater_equal(a, b):
    """numpy.greater_equal → ops.greater_equal(a, b)."""
    raise NotImplementedError(
        "enp.greater_equal: architecture stub — maps to ops.greater_equal"
    )


def logical_and(a, b):
    """numpy.logical_and → ops.logical_and(a, b)."""
    raise NotImplementedError(
        "enp.logical_and: architecture stub — maps to ops.logical_and"
    )


def logical_or(a, b):
    """numpy.logical_or → ops.logical_or(a, b)."""
    raise NotImplementedError(
        "enp.logical_or: architecture stub — maps to ops.logical_or"
    )


def logical_not(a):
    """numpy.logical_not → ops.logical_not(a)."""
    raise NotImplementedError(
        "enp.logical_not: architecture stub — maps to ops.logical_not"
    )


def where(cond, x, y):
    """numpy.where(cond, x, y) → ops.select(cond, x, y).

    numpy's single-arg index-array form is NOT provided in v1.
    """
    raise NotImplementedError(
        "enp.where: architecture stub — maps to ops.select(cond, x, y)"
    )
