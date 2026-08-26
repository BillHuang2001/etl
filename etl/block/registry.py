"""Block registries: declared BlockOps, portable impls, backend impls.

All persistent state of the block subsystem lives here. `etl.block(...)`
declarations register a BlockOp; `BlockOp.portable` / `BlockOp.impl` register
implementations; `get_block(name)` is the public accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from .errors import BlockError

if TYPE_CHECKING:  # pragma: no cover
    from .op import BlockOp

__all__ = [
    "get_block",
    "get_impl",
    "get_portable",
    "register",
    "register_impl",
    "register_portable",
    "validate_portable",
]

_BLOCKS: Dict[str, "BlockOp"] = {}          # name -> declared BlockOp
_IMPLS: Dict[Tuple[str, str], Callable] = {}  # (block_name, backend_name) -> impl
_PORTABLES: Dict[str, Callable] = {}        # block_name -> etl.defn function


def register(op: "BlockOp") -> None:
    """Register a declared BlockOp (called by decl.block at declaration time)."""
    if op.name in _BLOCKS:
        raise BlockError(
            f"block '{op.name}' is already registered — reuse "
            f"get_block('{op.name}') to add impls/rules instead of redeclaring"
        )
    _BLOCKS[op.name] = op


def get_block(name: str) -> "BlockOp":
    """Return the BlockOp registered under `name`."""
    if not isinstance(name, str) or name not in _BLOCKS:
        raise BlockError(
            f"unknown block {name!r}: declare it first with etl.block(...)"
        )
    return _BLOCKS[name]


def register_impl(name: str, backend_name: str, fn: Callable) -> None:
    """Register a backend-specific implementation for `name`."""
    _IMPLS[(name, backend_name)] = fn


def get_impl(name: str, backend_name: str) -> Optional[Callable]:
    """The registered implementation for (name, backend_name), or None."""
    return _IMPLS.get((name, backend_name))


def validate_portable(name: str, fn: Any) -> None:
    """Pure validation of a portable implementation (no mutation).

    Portable implementations must be `@etl.defn` functions: they are traced
    lazily into ordinary graphs at lower/transform time.
    """
    if not callable(fn) or getattr(fn, "__etl_defn__", None) is None:
        raise BlockError(
            f"portable implementation for block '{name}' must be an etl.defn "
            f"function, got {fn!r}"
        )


def register_portable(name: str, fn: Callable) -> None:
    """Register the portable (etl.defn) implementation for `name`."""
    validate_portable(name, fn)
    _PORTABLES[name] = fn


def get_portable(name: str) -> Optional[Callable]:
    """The registered portable implementation for `name`, or None."""
    return _PORTABLES.get(name)
