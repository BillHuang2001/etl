"""Error types owned by `etl.block`."""

from __future__ import annotations

from etl.core import ETLError

__all__ = ["BlockError"]


class BlockError(ETLError):
    """Invalid block declaration or static misuse of a `BlockOp`.

    Declaration-time problems: duplicate or unknown block names, malformed
    input/output specs, undeclared or wrongly typed static attributes,
    invalid effect kinds or batching policies, and structurally invalid
    declarations (e.g. a portable implementation that is not an `etl.defn`
    function, or a missing name).

    Graph-time problems use the standard core error classes instead:
    `TraceError` (call outside a trace / concrete `Tensor` operand),
    `ShapeError` / `DTypeError` (operand mismatch against `input_specs`),
    and `TransformError` (missing batching/derivative rule — raised by
    `etl.transforms`, never by block itself).
    """
