"""Linear-algebra sugar shared by the top-level namespace and `enp.linalg`.

`matmul`/`dot` are re-exported at top level (numpy: np.matmul/np.dot are NOT
in np.linalg); `linalg.py` re-exports `solve` into the public
`etl.numpy.linalg` submodule. Implemented: all functions forward to the frozen
ops contract (ops.dot / ops.solve). ``matrix_exp`` is a documented deviation —
numpy has no ``linalg.matrix_exp``; etl defines it (scipy/torch semantics).
"""

from __future__ import annotations

from .. import ops  # etl.ops — lower layer, allowed import

__all__ = [
    "matmul", "dot", "solve", "eigh", "cholesky", "qr", "matrix_rank",
    "svd", "matrix_exp",
]


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


def eigh(x):
    """numpy.linalg.eigh → ops.eigh(x) (returns (w, v))."""
    return ops.eigh(x)


def cholesky(x):
    """numpy.linalg.cholesky → ops.cholesky(x)."""
    return ops.cholesky(x)


def qr(x):
    """numpy.linalg.qr (reduced mode) → ops.qr(x) (returns (q, r))."""
    return ops.qr(x)


def matrix_rank(x, tol=None):
    """numpy.linalg.matrix_rank → ops.matrix_rank(x, tol)."""
    return ops.matrix_rank(x, tol=tol)


def svd(x):
    """numpy.linalg.svd (full_matrices=False) → ops.svd(x) (returns
    (u, s, vh))."""
    return ops.svd(x)


def matrix_exp(x):
    """numpy.linalg.matrix_exp (deviation: numpy has none — scipy/torch
    semantics) → ops.matrix_exp(x)."""
    return ops.matrix_exp(x)
