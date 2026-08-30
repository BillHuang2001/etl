"""Public `etl.numpy.linalg` submodule (mirrors numpy.linalg).

Exposes `solve`, `eigh`, `cholesky`, `qr`, `matrix_rank`, `svd` (numpy
semantics) and `matrix_exp` (scipy/torch semantics — numpy has no
`linalg.matrix_exp`; documented deviation). `inv`/`norm`/`det` are deferred
to v2 (they need new IR ops not present in the etl/ops contract — see
CONTEXT.md).
"""

from __future__ import annotations

from ._linalg import cholesky, eigh, matrix_exp, matrix_rank, qr, solve, svd

__all__ = [
    "solve", "eigh", "cholesky", "qr", "matrix_rank", "svd", "matrix_exp",
]
