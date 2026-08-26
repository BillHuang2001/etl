"""StableHLO export mapping tables (pure data + trivial lookup helpers).

These tables are the single source of truth for the v1 StableHLO mapping
defined in `../../CONTEXT.md` ("StableHLO exporter — v1 scope", binding).
The MLIR text emission logic lives in `./writer.py` and consumes these
tables; this module contains no emission logic.

NOTE: the exact StableHLO mnemonics below must be verified against the
StableHLO spec at implementation time (the writer implementer is
responsible for this check; do not trust the mnemonics blindly).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Direct 1:1 mappings: etl op name -> StableHLO mnemonic (with `stablehlo.`
# dialect prefix, as emitted in MLIR text).
# ---------------------------------------------------------------------------

ELEMENTWISE_MAP: dict[str, str] = {
    "add": "stablehlo.add",
    "subtract": "stablehlo.subtract",
    "multiply": "stablehlo.multiply",
    "divide": "stablehlo.divide",
    "power": "stablehlo.power",
    "remainder": "stablehlo.remainder",
    "maximum": "stablehlo.maximum",
    "minimum": "stablehlo.minimum",
    "abs": "stablehlo.abs",
    "negate": "stablehlo.negate",
    "sqrt": "stablehlo.sqrt",
    "sign": "stablehlo.sign",
    "exp": "stablehlo.exponential",
    "log": "stablehlo.log",
    "log1p": "stablehlo.log_plus_one",
    "sin": "stablehlo.sine",
    "cos": "stablehlo.cosine",
    "tan": "stablehlo.tan",
    "tanh": "stablehlo.tanh",
    "sigmoid": "stablehlo.logistic",
    "erf": "stablehlo.erf",
    "bitwise_and": "stablehlo.and",
    "bitwise_or": "stablehlo.or",
    "bitwise_xor": "stablehlo.xor",
    "logical_and": "stablehlo.and",
    "logical_or": "stablehlo.or",
    "logical_not": "stablehlo.not",
    "cast": "stablehlo.convert",
}

# Ops emitted by expansion into other ops rather than a direct mnemonic.
# Values are human-readable expansion notes; the writer implements them as
# ordinary sub-op emissions (never as a single mnemonic).
DECOMPOSITIONS: dict[str, str] = {
    "square": "multiply(x, x)",
    "relu": "maximum(x, 0)",
    "gelu": "erf-based: 0.5*x*(1+erf(x/sqrt(2)))",
    "stop_gradient": "identity passthrough (emit operand directly)",
    "reduce_mean": "reduce-sum then divide by element count",
}

# Comparisons emitted as `stablehlo.compare` plus a `comparison_direction`
# attribute; the value is the direction string (EQ/NE/LT/LE/GT/GE).
COMPARISON_MAP: dict[str, str] = {
    "equal": "EQ",
    "not_equal": "NE",
    "less": "LT",
    "less_equal": "LE",
    "greater": "GT",
    "greater_equal": "GE",
}

# Shape/data-movement, reduction, and linear-algebra ops with direct
# mnemonics. Reduce ops all use `stablehlo.reduce` (their body carries the
# reduction kind; the writer must inspect the op's attributes).
SHAPE_MAP: dict[str, str] = {
    "select": "stablehlo.select",
    "broadcast": "stablehlo.broadcast_in_dim",
    "reshape": "stablehlo.reshape",
    "transpose": "stablehlo.transpose",
    "slice": "stablehlo.slice",
    "concatenate": "stablehlo.concatenate",
    "pad": "stablehlo.pad",
    "reduce_sum": "stablehlo.reduce",
    "reduce_max": "stablehlo.reduce",
    "reduce_min": "stablehlo.reduce",
    "reduce_prod": "stablehlo.reduce",
    "argmax": "stablehlo.argmax",
    "argmin": "stablehlo.argmin",
    "dot": "stablehlo.dot_general",
    "conv": "stablehlo.convolution",
}

# `etl.constant` embeds data as `stablehlo.constant` with a dense elements
# attribute (see `Writer.render_constant`). Kept separate from the other
# direct tables because its emission path is attribute-heavy, not a plain
# mnemonic call.
CONSTANT_MAP: dict[str, str] = {
    "constant": "stablehlo.constant",
}

CONTROL_FLOW_MAP: dict[str, str] = {
    "cond": "stablehlo.if",
    "while_loop": "stablehlo.while",
}

COLLECTIVE_MAP: dict[str, str] = {
    "all_reduce": "stablehlo.all_reduce",
    "all_gather": "stablehlo.all_gather",
    "reduce_scatter": "stablehlo.reduce_scatter",
    "all_to_all": "stablehlo.all_to_all",
    "broadcast": "stablehlo.collective_broadcast",
    "collective_permute": "stablehlo.collective_permute",
}

# Deferred in v1: the writer raises core.BackendError naming the op, with a
# message suggesting decomposition or a future adapter. `rank`/`world_size`
# are the dist graph scalars; complex-number elementwise beyond cast is
# additionally deferred (not an op name — enforced by dtype checks in the
# writer).
DEFERRED_OPS: frozenset[str] = frozenset(
    {"gather", "scatter", "scan", "runtime_call", "block_call", "rank", "world_size"}
)

# numpy dtype -> MLIR type string. Keys are numpy dtype objects; lookups
# must normalize with `numpy.dtype(value)` so that e.g. `np.float32`,
# `np.dtype("float32")` and `"float32"` all resolve.
DTYPE_MAP: dict = {
    np.dtype("float16"): "f16",
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int8"): "i8",
    np.dtype("int16"): "i16",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
    np.dtype("uint8"): "ui8",
    np.dtype("uint16"): "ui16",
    np.dtype("uint32"): "ui32",
    np.dtype("uint64"): "ui64",
    np.dtype("bool"): "i1",
    np.dtype("complex64"): "complex<f32>",
    np.dtype("complex128"): "complex<f64>",
}

# Deterministic search order for lookup_mapping/status. ELEMENTWISE and
# SHAPE tables are checked before COLLECTIVE so that the etl data-movement
# op `broadcast` resolves to `stablehlo.broadcast_in_dim` (see the note on
# the name collision below).
_SEARCH_ORDER: tuple[dict[str, str], ...] = (
    ELEMENTWISE_MAP,
    SHAPE_MAP,
    CONSTANT_MAP,
    CONTROL_FLOW_MAP,
    COLLECTIVE_MAP,
    COMPARISON_MAP,
    DECOMPOSITIONS,
)

_DIRECT_TABLES: tuple[dict[str, str], ...] = (
    ELEMENTWISE_MAP,
    SHAPE_MAP,
    CONSTANT_MAP,
    CONTROL_FLOW_MAP,
    COLLECTIVE_MAP,
    COMPARISON_MAP,
)


def lookup_mapping(op_name: str) -> str:
    """Return the first matching entry across the mapping tables.

    Search order: ELEMENTWISE_MAP, SHAPE_MAP, CONSTANT_MAP, CONTROL_FLOW_MAP,
    COLLECTIVE_MAP, COMPARISON_MAP, DECOMPOSITIONS.

    For comparisons the value is the `comparison_direction` (e.g. "LT");
    for decompositions it is a human-readable expansion note; otherwise it
    is the full StableHLO mnemonic (`stablehlo.*`).

    Raises KeyError if `op_name` appears in no table.

    Note: exact StableHLO mnemonics must be verified against the StableHLO
    spec at implementation time.
    """
    for table in _SEARCH_ORDER:
        if op_name in table:
            return table[op_name]
    raise KeyError(op_name)


def status(op_name: str) -> Literal["v1", "decompose", "deferred"]:
    """Export status of an etl op name.

    - ``"v1"``: has a direct mnemonic (or comparison direction).
    - ``"decompose"``: emitted as an expansion of ordinary ops.
    - ``"deferred"``: in DEFERRED_OPS or unmapped anywhere; the writer
      raises core.BackendError naming the op either way.
    """
    if op_name in DECOMPOSITIONS:
        return "decompose"
    if op_name in DEFERRED_OPS:
        return "deferred"
    for table in _DIRECT_TABLES:
        if op_name in table:
            return "v1"
    return "deferred"


def is_supported(op_name: str) -> bool:
    """True iff the writer can emit this op in v1 (direct or decomposed)."""
    return status(op_name) in ("v1", "decompose")


def mlir_dtype(dtype) -> str:
    """MLIR type string for a numpy dtype (normalized via ``numpy.dtype``).

    Raises KeyError for dtypes outside DTYPE_MAP (complex-number elementwise
    beyond cast, and any other unsupported dtype, is enforced here).
    """
    return DTYPE_MAP[np.dtype(dtype)]


# ---------------------------------------------------------------------------
# Known name collision (do not "fix" without updating ../../CONTEXT.md):
# the etl op name `broadcast` appears in BOTH SHAPE_MAP
# (`stablehlo.broadcast_in_dim` — the ops.broadcast data-movement op) and
# COLLECTIVE_MAP (`stablehlo.collective_broadcast` — the dist.broadcast
# collective). lookup_mapping resolves it to the SHAPE_MAP entry; the writer
# must disambiguate by the op's effect kind (collective effect ⇒ collective
# mnemonic) at emission time. Confirm the IR op name used by `dist.broadcast`
# with the `dist` owner during implementation.
# ---------------------------------------------------------------------------
