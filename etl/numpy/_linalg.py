"""Linear-algebra sugar shared by the top-level namespace and `enp.linalg`.

`matmul`/`dot` are re-exported at top level (numpy: np.matmul/np.dot are NOT
in np.linalg); `linalg.py` re-exports `solve` into the public
`etl.numpy.linalg` submodule. Implemented: all functions forward to the frozen
ops contract (ops.dot / ops.solve).
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = ["matmul", "dot", "solve"]


def matmul(a, b):
    """numpy.matmul → ops.dot(a, b)."""
    return ops.dot(a, b)


def dot(a, b):
    """numpy.dot → ops.dot(a, b) (v1 alias of matmul).

    Deviation: numpy's 1-D vector inner-product semantics for dot are not
    special-cased in v1 — dot and matmul are identical here.
    """
    return ops.dot(a, b)


def solve(a, b):
    """numpy.linalg.solve → ops.solve(a, b)."""
    return ops.solve(a, b)
