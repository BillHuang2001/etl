"""etl.block — custom operations for the EvoX tensor library.

Declares user-defined operations (`block_call` ops), registers backend
implementations and portable decompositions, and bridges batching and
derivative rules into etl.transforms under namespaced keys (`block:<name>`).

See `CONTEXT.md` in this directory for the full design.
"""

from __future__ import annotations

from .decl import AttributeField, BatchingPolicy, StaticValue, block
from .errors import BlockError
from .op import BlockOp
from .registry import get_block

__all__ = [
    "AttributeField",
    "BatchingPolicy",
    "BlockError",
    "BlockOp",
    "StaticValue",
    "block",
    "get_block",
]
