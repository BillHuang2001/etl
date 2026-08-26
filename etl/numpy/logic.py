"""Comparison and logical sugar of `etl.numpy` (enp).

1:1 sugar over `etl.ops` (same IR as the mapped ops calls; concrete `Tensor`
args raise `TraceError`). `where` is a documented rename/alias of
`ops.select`. All functions are implemented pure-sugar forwards: no extra
validation, dtype logic, or error handling — ops' errors surface unchanged.
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not", "where",
]


def equal(a, b):
    """numpy.equal → ops.equal(a, b)."""
    return ops.equal(a, b)


def not_equal(a, b):
    """numpy.not_equal → ops.not_equal(a, b)."""
    return ops.not_equal(a, b)


def less(a, b):
    """numpy.less → ops.less(a, b)."""
    return ops.less(a, b)


def less_equal(a, b):
    """numpy.less_equal → ops.less_equal(a, b)."""
    return ops.less_equal(a, b)


def greater(a, b):
    """numpy.greater → ops.greater(a, b)."""
    return ops.greater(a, b)


def greater_equal(a, b):
    """numpy.greater_equal → ops.greater_equal(a, b)."""
    return ops.greater_equal(a, b)


def logical_and(a, b):
    """numpy.logical_and → ops.logical_and(a, b)."""
    return ops.logical_and(a, b)


def logical_or(a, b):
    """numpy.logical_or → ops.logical_or(a, b)."""
    return ops.logical_or(a, b)


def logical_not(a):
    """numpy.logical_not → ops.logical_not(a)."""
    return ops.logical_not(a)


def where(cond, x, y):
    """numpy.where(cond, x, y) → ops.select(cond, x, y).

    numpy's single-arg index-array form is NOT provided in v1.
    """
    return ops.select(cond, x, y)
