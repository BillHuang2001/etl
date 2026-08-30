"""StableHLO export mapping tables (pure data + trivial lookup helpers).

These tables are the single source of truth for the v1 StableHLO mapping
defined in `../../CONTEXT.md` ("StableHLO exporter — v1 scope", binding).
The MLIR text emission logic lives in `./writer.py` and consumes these
tables; this module contains no emission logic.

MNEMONIC VERIFICATION (done at implementation time against the StableHLO
spec, https://openxla.org/stablehlo/spec, the interpreter status table
https://openxla.org/stablehlo/interpreter_status, and
stablehlo/dialect/StablehloOps.td):

* Confirmed as specced StableHLO ops: add/subtract/multiply/divide/power/
  remainder/maximum/minimum, abs/negate/sqrt/sign, exponential
  (``exp``), log/log_plus_one, sine/cosine/tan/tanh, logistic (sigmoid),
  and/or/xor/not, convert (cast), compare (+ ``comparison_direction``),
  select, broadcast_in_dim, reshape, transpose, slice, concatenate, pad,
  reduce (with a reducer region + init value), dot_general, convolution,
  constant, if/while, shift_left/shift_right (arithmetic+logical),
  gather, scatter, sort (with a comparator region + ``dimension``),
  reverse, iota (multi-rank ``dim`` form), all_gather/all_reduce/all_to_all/
  collective_broadcast/collective_permute/reduce_scatter.
* Emitted via dedicated writer routines (``SPECIAL_EMITTERS``), not a
  single mnemonic:
  - ``gather``/``scatter`` → ``stablehlo.gather``/``stablehlo.scatter``
    with the exact numpy ``take``/``put_along_axis`` semantics (single-axis
    gather; full-rank index reshaping + numpy broadcast for scatter).
  - ``sort``/``argsort`` → ``stablehlo.sort`` with an LT comparator region
    (argsort sorts a (value, iota) pair; iota tie-break gives numpy stable
    semantics; descending = ``stablehlo.reverse`` after — the numpy
    composition).
  - ``argmax``/``argmin`` → a two-operand ``stablehlo.reduce`` over
    (value, iota) with an index tie-break comparator (first occurrence on
    ties, matching ``np.argmax``/``np.argmin``).
  - ``tile`` → reshape + ``broadcast_in_dim`` + reshape decomposition.
  - the ``random_*`` ops → inline SplitMix64 subgraph expansion (see
    ``./random_export.py``): bit-identical to the numpy kernels by
    two's-complement equivalence; NOT ``stablehlo.rng`` (implementation-
    defined algorithm would break the same-key⇒same-values determinism
    contract).
* NOT in StableHLO (moved to DEFERRED_OPS here):
  - ``erf`` — only ``chlo.erf`` exists (CHLO: "an intermediate value in
    decompositions, never constructed directly"), and there is no trivial
    StableHLO decomposition of the error function. Emitting
    ``stablehlo.erf`` would produce invalid MLIR.
  - ``gelu`` — the binding decomposition (0.5*x*(1+erf(x/sqrt(2)))) needs
    erf, so it is deferred together with it (no silent approximation).
  - sparse ops — the ``sparse_*``/``dense_dot_sparse`` family
    (``etl.sparse``) is numpy-backend-only in v1; densify via
    ``etl.sparse.to_dense`` to export.
* ``collective_broadcast`` IS a specced StableHLO op (present in the
  spec's collective-op list and the status table).
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
    "acos": "stablehlo.acos",
    "tanh": "stablehlo.tanh",
    "floor": "stablehlo.floor",
    "ceil": "stablehlo.ceil",
    "round": "stablehlo.round_nearest_even",
    "sigmoid": "stablehlo.logistic",
    "bitwise_and": "stablehlo.and",
    "bitwise_or": "stablehlo.or",
    "bitwise_xor": "stablehlo.xor",
    "logical_and": "stablehlo.and",
    "logical_or": "stablehlo.or",
    "logical_not": "stablehlo.not",
    "cast": "stablehlo.convert",
    "bitwise_left_shift": "stablehlo.shift_left",
    # bitwise_right_shift maps to shift_right_arithmetic (signed operands)
    # or shift_right_logical (unsigned) — the writer picks by result dtype
    # at emission time (the entry documents the signed default).
    "bitwise_right_shift": "stablehlo.shift_right_arithmetic",
}

# Ops emitted by dedicated writer routines (gather/scatter/sort/argsort/
# argmax/argmin/tile use multi-op StableHLO compositions; the random_* ops
# are expanded as inline SplitMix64 i64 subgraphs in `./random_export.py` —
# never `stablehlo.rng`, whose implementation-defined algorithm would break
# the same-key⇒same-values determinism contract). The values are emitter
# family keys; `status()` reports these as supported ("v1").
SPECIAL_EMITTERS: dict[str, str] = {
    "gather": "gather",
    "scatter": "scatter",
    "sort": "sort",
    "argsort": "argsort",
    "argmax": "arg_reduce",
    "argmin": "arg_reduce",
    "tile": "tile",
    "random_key_mix": "random",
    "random_uniform": "random",
    "random_normal": "random",
    "random_randint": "random",
    "random_permutation": "random",
    # random_multinomial stays deferred (cumulative-search decomposition is
    # not wired in v1; not needed by the benchmark suite).
}

# Ops emitted by expansion into other ops rather than a direct mnemonic.
# Values are human-readable expansion notes; the writer implements them as
# ordinary sub-op emissions (never as a single mnemonic).
DECOMPOSITIONS: dict[str, str] = {
    "square": "multiply(x, x)",
    "relu": "maximum(x, 0)",
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

# The real IR op names for the trace-level cond/while_loop constructs are
# `if` and `while` (etl.trace lowers cond/while_loop to if/while ops); the
# frontend names never appear in IR.
CONTROL_FLOW_MAP: dict[str, str] = {
    "if": "stablehlo.if",
    "while": "stablehlo.while",
}

COLLECTIVE_MAP: dict[str, str] = {
    "all_reduce": "stablehlo.all_reduce",
    "all_gather": "stablehlo.all_gather",
    "reduce_scatter": "stablehlo.reduce_scatter",
    "all_to_all": "stablehlo.all_to_all",
    "broadcast_collective": "stablehlo.collective_broadcast",
    "collective_permute": "stablehlo.collective_permute",
}

# Deferred in v1: the writer raises core.BackendError naming the op, with a
# message suggesting decomposition or a future adapter. `rank`/`world_size`
# are the dist graph scalars; complex-number elementwise beyond cast is
# additionally deferred (not an op name — enforced by dtype checks in the
# writer). `erf`/`gelu` are deferred because StableHLO has no erf op (chlo
# only); `diagonal` because its StableHLO emission is not wired in v1
# (defer explicitly, never silently). The linalg factorizations
# (`eigh`/`cholesky`/`qr`/`svd`) have StableHLO counterparts but are not
# wired in v1; `matrix_rank`/`matrix_exp` need decomposition (SVD cutoff /
# Padé) — all six defer explicitly. `random_multinomial` defers because its
# cumulative-search decomposition is not wired in v1. The 16
# `sparse_*`/`dense_dot_sparse` ops (etl.sparse family) are numpy-backend-only
# in v1 — densify via `etl.sparse.to_dense` to export.
DEFERRED_OPS: frozenset[str] = frozenset(
    {
        "scan",
        "runtime_call",
        "block_call",
        "rank",
        "world_size",
        "erf",
        "gelu",
        "diagonal",
        "eigh",
        "cholesky",
        "qr",
        "matrix_rank",
        "svd",
        "matrix_exp",
        "random_multinomial",
        "sparse_from_dense",
        "sparse_to_dense",
        "sparse_coo_to_csr",
        "sparse_csr_to_coo",
        "sparse_coo_to_csc",
        "sparse_csc_to_coo",
        "sparse_negate",
        "sparse_add",
        "sparse_multiply",
        "sparse_multiply_dense",
        "sparse_reduce_sum",
        "sparse_transpose",
        "sparse_reshape",
        "sparse_concatenate",
        "sparse_dot_dense",
        "dense_dot_sparse",
    }
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

    Note: the mnemonics were verified against the StableHLO spec at
    implementation time (see the module docstring).
    """
    for table in _SEARCH_ORDER:
        if op_name in table:
            return table[op_name]
    raise KeyError(op_name)


def status(op_name: str) -> Literal["v1", "decompose", "deferred"]:
    """Export status of an etl op name.

    - ``"v1"``: has a direct mnemonic (or comparison direction) or a
      dedicated emitter routine (``SPECIAL_EMITTERS``).
    - ``"decompose"``: emitted as an expansion of ordinary ops.
    - ``"deferred"``: in DEFERRED_OPS or unmapped anywhere; the writer
      raises core.BackendError naming the op either way.
    """
    if op_name in DECOMPOSITIONS:
        return "decompose"
    if op_name in DEFERRED_OPS:
        return "deferred"
    if op_name in SPECIAL_EMITTERS:
        return "v1"
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
# Naming note: the IR op names do NOT collide — the shape op is `broadcast`
# (SHAPE_MAP → `stablehlo.broadcast_in_dim`) and the dist collective is
# `broadcast_collective` (COLLECTIVE_MAP → `stablehlo.collective_broadcast`),
# matching the collective op defs in `etl/ir/op_defs/collective.py`. No
# effect-kind disambiguation is needed at emission time.
# ---------------------------------------------------------------------------
