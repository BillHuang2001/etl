"""Block-call inlining support (thin re-export of the shared machinery).

The shared block-inlining machinery moved to ``etl.backends.inline``
(``..inline``) so compiler backends (``CompilerBackend``, see
``../compiler.py``) reuse the exact same expansion logic as the numpy
reference interpreter — see that module for the full docs and bookkeeping
invariants. This module keeps the numpy backend's historical import path
working (``from .inline import clone_ops_into, drop_op_uses``).
"""
from __future__ import annotations

from ..inline import (  # noqa: F401
    clone_ops_into,
    drop_op_uses,
    inline_portables,
    iter_block_ops,
    iter_ops,
)

__all__ = ["clone_ops_into"]
