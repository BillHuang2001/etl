"""Public `etl.numpy.linalg` submodule (mirrors numpy.linalg).

v1 exposes `solve` only; `inv`/`norm`/`det` are deferred to v2 (they need
new IR ops not present in the etl/ops contract — see CONTEXT.md).
"""

from __future__ import annotations

from ._linalg import solve

__all__ = ["solve"]
