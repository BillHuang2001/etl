"""Key-based functional RNG — the ``etl.random`` frontend (graph ops).

Design (binding — see ``etl/CONTEXT.md``, section "etl.random"):

- **Algorithms & key representation**: three canonical algorithms
  (``ALGORITHMS``, imported from ``etl.ir.op_defs.random`` — the SINGLE
  source of truth). ``splitmix64`` (the DEFAULT — v1 behavior unchanged): a
  rank-0 int64 tensor holding a 64-bit state (two's complement).
  ``threefry2x32``: a shape ``(2,)`` int32 tensor (2 counter words).
  ``philox4x32_10``: a shape ``(4,)`` int32 tensor (4 counter words).
  ``key(seed, algorithm=...)`` accepts canonical strings or ``Algorithm``
  members and derives the key words deterministically (see below).
- **Key-word derivation** (STABLE for threefry2x32/philox4x32_10): the 2/4
  key words are the lower 32 bits (interpreted as signed int32) of the first
  2/4 canonical SplitMix64 words from the per-algorithm 64-bit state
  ``seed ^ SALT_ALG mod 2^64`` (``SALT_ALG`` a fixed DISTINCT constant per
  algorithm — hex digits of pi, matching the per-op salt philosophy):
  ``word_i = low32(mix64((seed ^ SALT_ALG) + (i + 1) * GOLDEN mod 2^64))``,
  where ``mix64`` is the standard SplitMix64 finalizer
  ``((z ^ (z >> 30)) * B ^ ...) mod 2^64`` (the same generator the numpy
  kernels use — see ``etl/backends/numpy/kernels/random.py``). The SAME
  derivation is used inside a trace (Constant op) and outside (concrete
  ``core.Tensor``). splitmix64 keeps the v1 behavior: the seed itself,
  masked to 64 bits.
- **Algorithm inference**: every op OTHER than ``key`` takes NO algorithm
  parameter — the algorithm is inferred from the key operand's STATIC
  shape/dtype at trace time (rank-0 int64 → splitmix64; shape (2,) int32 →
  threefry2x32; shape (4,) int32 → philox4x32_10) and stamped into the op's
  ``algorithm`` attribute. A key matching no form raises ``ShapeError``
  naming the three accepted forms.
- **Determinism**: every random op is a PURE, deterministic function of its
  key operand — the same key + same operands yield BIT-IDENTICAL values,
  repeatable across separate evaluate calls. Keys are never mutated or
  consumed by an op call; users call ``split``/``split_n`` to derive
  independent keys (two ops sharing one key are correlated by construction).
- **Stream derivation (SplitMix64)**: each op kind draws its stream from the
  SplitMix64 generator seeded with ``key ^ SALT``, where ``SALT`` is a fixed
  per-op constant (below) — different op kinds with the same key therefore
  draw decorrelated streams. ``split``/``split_n`` are sugar over the
  ``random_key_mix`` op: ``mix(key ^ salt)`` with salt ``0`` / the golden
  gamma / ``i * golden``.
- **Polymorphism**: ``key`` is a deterministic CREATOR — outside a trace it
  returns a concrete ``core.Tensor`` (usable as an explicit graph input);
  inside a trace it builds a Constant op. ALL other functions are graph ops:
  outside a trace they raise ``TraceError``; a concrete ``Tensor`` operand
  raises ``TraceError`` (etl has no eager mode; randomness goes through
  compiled graphs — root CONTEXT.md principle 9).
- **dtype rules**: ``uniform``/``normal`` require a floating dtype (default
  float32); ``randint``/``permutation`` require an integer dtype (default
  int32); ``multinomial`` output is int32.
- **shapes**: the ``shape`` argument accepts Python ints and symbolic
  ``Dim``/``DimExpr`` entries (evaluated at run time via dim bindings);
  runtime-dynamic ``None`` entries are rejected. ``low``/``high``/``mean``/
  ``std`` may be Python scalars or SYMBOLIC tensors broadcastable against
  ``shape`` (the result shape is the broadcast — matching the kernels).
- **Backends**: the numpy interpreter is the reference backend for all 6
  ops. 5 of the 6 ops (``random_key_mix``, ``random_uniform``,
  ``random_normal``, ``random_randint``, ``random_permutation``) ALSO export
  as v1 StableHLO — inline SplitMix64 i64 subgraph expansion in
  ``etl/backends/stablehlo/random_export.py`` (never ``stablehlo.rng``),
  bit-exact vs the numpy kernels (uniform/randint/permutation EXACT;
  ``random_normal`` with f32 output uses a documented f32 Box–Muller fast
  path, maxdiff ~1e-6 vs the f64 numpy kernel, same-key bit-identical across
  runs). Only ``random_multinomial`` defers on compiler backends with an
  explicit ``BackendError`` — never silent fallback. Transforms
  (``vmap``/``grad``/...) have no rules for random ops → ``TransformError``.
"""
from __future__ import annotations

import enum

import numpy as np

from etl import core
from etl.ir.op_defs.random import (
    ALGORITHMS,
    DEFAULT_ALGORITHM,
    algorithm_key_type,
    validate_algorithm,
)

from . import _utils
from . import constant as _constant

__all__ = [
    "key",
    "split",
    "split_n",
    "uniform",
    "normal",
    "randint",
    "permutation",
    "multinomial",
    "Algorithm",
    "ALGORITHMS",
]


class Algorithm(enum.Enum):
    """Canonical RNG algorithms (values are the canonical names).

    Members: ``SPLITMIX64``, ``THREEFRY2X32``, ``PHILOX4X32_10``. Accepted
    wherever ``etl.random.key`` takes an ``algorithm`` argument — alongside
    the plain canonical strings. Values are exactly
    ``etl.ir.op_defs.random.ALGORITHMS`` (validated via
    ``validate_algorithm`` on the member's value).
    """

    SPLITMIX64 = "splitmix64"
    THREEFRY2X32 = "threefry2x32"
    PHILOX4X32_10 = "philox4x32_10"


#: SplitMix64 golden gamma — also the ``split`` second salt.
_GOLDEN = 0x9E3779B97F4A7C15
_MASK64 = 0xFFFFFFFFFFFFFFFF
#: SplitMix64 finalizer multipliers (standard constants).
_B_MUL = 0xBF58476D1CE4E5B9
_C_MUL = 0x94D049BB133111EB
#: Per-algorithm key-derivation salts (hex digits of pi): the counter-based
#: key words are derived from the SplitMix64 stream of ``seed ^ SALT_ALG``
#: so threefry2x32 and philox4x32_10 keys are decorrelated for the same
#: seed (splitmix64 keeps the v1 raw masked seed — no salt).
_KEY_SALTS = {
    "threefry2x32": 0x243F6A8885A308D3,
    "philox4x32_10": 0x13198A2E03707344,
}

_FLOAT_KINDS = frozenset("f")  # float16/32/64
_INT_KINDS = frozenset("iu")  # signed + unsigned integers


def _wrap(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result, reading dtype/shape back from the IR."""
    result = op.result
    return core.SymbolicTensor(
        value=result,
        dtype=result.type.dtype,
        shape=result.type.shape,
        location=loc,
    )


def _check_key(key, name: str):
    """Validate a key operand and infer its algorithm.

    Returns ``(key_t, algorithm)``: the validated ``SymbolicTensor`` plus the
    algorithm inferred from its STATIC shape/dtype — rank-0 int64 →
    splitmix64, shape (2,) int32 → threefry2x32, shape (4,) int32 →
    philox4x32_10. A key matching no form raises ``ShapeError`` naming the
    three accepted forms (mirroring ``etl.ir.inference``'s wording).
    """
    if not isinstance(key, core.SymbolicTensor):
        if isinstance(key, core.Tensor):
            raise core.TraceError(
                f"{name}: got a concrete Tensor key — etl has no eager mode; "
                "create the key inside the traced function with "
                "etl.random.key(seed) or pass it as an explicit graph input"
            )
        raise TypeError(
            f"{name}: key must be a SymbolicTensor (a graph value), got "
            f"{type(key).__name__}"
        )
    for alg in ALGORITHMS:
        shape, dtype = algorithm_key_type(alg)
        if key.shape == shape and key.dtype == dtype:
            return key, alg
    raise core.ShapeError(
        f"{name}: key must be {_key_forms()}, got "
        f"dtype={key.dtype} shape={key.shape}"
    )


def _key_forms() -> str:
    """Human-readable list of the three accepted key forms.

    e.g. "a rank-0 int64 tensor (splitmix64), a shape (2,) int32 tensor
    (threefry2x32), or a shape (4,) int32 tensor (philox4x32_10)" — the
    exact wording mirrored from ``etl.ir.inference``'s validation.
    """
    parts = []
    for alg in ALGORITHMS:
        shape, dtype = algorithm_key_type(alg)
        if shape == ():
            parts.append(f"a rank-0 {dtype} tensor ({alg})")
        else:
            parts.append(f"a shape {shape} {dtype} tensor ({alg})")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", or " + parts[-1]


def _algorithm_arg(algorithm) -> str:
    """Normalize an algorithm argument: ``Algorithm`` member or canonical
    string (``validate_algorithm`` raises for anything else)."""
    if isinstance(algorithm, Algorithm):
        algorithm = algorithm.value
    return validate_algorithm(algorithm)


def _mix64(z):
    """The standard SplitMix64 finalizer (vectorized uint64, wrapped mod
    2^64): ``z = (z ^ (z >> 30)) * B; z = (z ^ (z >> 27)) * C; return
    z ^ (z >> 31)``."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_B_MUL) & np.uint64(_MASK64)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_C_MUL) & np.uint64(_MASK64)
    return (z ^ (z >> np.uint64(31))) & np.uint64(_MASK64)


def _derive_key_words(seed: int, count: int, salt: int) -> np.ndarray:
    """``count`` int32 key words for threefry2x32/philox4x32_10 keys.

    STABLE derivation (documented in the module docstring): word i is the
    lower 32 bits of the canonical SplitMix64 word ``mix64((seed ^ salt) +
    (i + 1) * GOLDEN mod 2^64)`` (the same generator the numpy kernels use),
    interpreted as signed int32; ``salt`` is the per-algorithm derivation
    salt. Identical inside a trace (Constant) and outside (concrete Tensor)
    — ``key()`` calls this in both paths.
    """
    state = np.uint64(seed & _MASK64) ^ np.uint64(salt)
    states = (
        state + np.arange(count, dtype=np.uint64) * np.uint64(_GOLDEN)
    ) & np.uint64(_MASK64)
    z = _mix64((states + np.uint64(_GOLDEN)) & np.uint64(_MASK64))
    return (z & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)


def _shape(shape, name: str) -> tuple:
    """Normalize a random-op shape: int or tuple/list of int/Dim/DimExpr.

    Runtime-dynamic (``None``) entries are rejected: the numpy kernels must
    evaluate the shape at run time via dim bindings.
    """
    if isinstance(shape, int) and not isinstance(shape, bool):
        return (shape,)
    if isinstance(shape, (tuple, list)):
        out = []
        for d in shape:
            if isinstance(d, bool) or not isinstance(
                d, (int, core.Dim, core.DimExpr)
            ):
                raise core.TraceError(
                    f"{name}: shape entries must be int | Dim | DimExpr "
                    f"(runtime-dynamic None is not supported), got {d!r}"
                )
            out.append(d)
        return tuple(out)
    raise core.TraceError(
        f"{name}: shape must be an int or a tuple/list of int | Dim | "
        f"DimExpr, got {shape!r}"
    )


def _out_dtype(dtype, name: str, kinds) -> np.dtype:
    """Normalize and kind-check the output dtype."""
    try:
        dt = core.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise core.DTypeError(
            f"{name}: invalid dtype {dtype!r}"
        ) from exc
    if dt.kind not in kinds:
        kind_desc = "a floating" if kinds == _FLOAT_KINDS else "an integer"
        raise core.DTypeError(
            f"{name}: dtype must be {kind_desc} dtype, got {dt}"
        )
    return dt


def _param(value, name: str, scalar_kinds, loc):
    """Promote one broadcastable parameter (low/high/mean/std/n) to an operand.

    ``SymbolicTensor`` passes through; an exact Python scalar of
    ``scalar_kinds`` becomes a 0-d Constant with its natural dtype (float64
    for floats, int64 for ints); concrete Tensors and other kinds raise.
    """
    if isinstance(value, core.SymbolicTensor):
        return value
    if type(value) in scalar_kinds:
        return _utils.as_operand(value, location=loc)
    if isinstance(value, core.Tensor):
        raise core.TraceError(
            f"{name}: got a concrete Tensor operand — etl has no eager mode; "
            "use etl.constant or pass it as an explicit graph input"
        )
    raise TypeError(
        f"{name}: expected a Python scalar or a SymbolicTensor, got "
        f"{type(value).__name__}"
    )


def _key_mix(builder, key_t, salt: int, algorithm: str, loc) -> "core.SymbolicTensor":
    """Build one ``random_key_mix`` op (the split/split_n building block)."""
    op = builder.create(
        "random_key_mix",
        operands=(key_t.value,),
        attributes={"salt": salt, "algorithm": algorithm},
        location=loc,
    )
    return _wrap(op, loc)


# --- public API ---------------------------------------------------------------


def key(seed: int, algorithm=DEFAULT_ALGORITHM):
    """Create a key from an int seed (polymorphic creator).

    Outside a trace: returns a concrete ``core.Tensor`` (usable as an
    explicit graph input). Inside a trace: builds a Constant op and returns
    the ``SymbolicTensor``.

    ``algorithm`` selects the key representation: ``splitmix64`` (default —
    v1 behavior unchanged: a rank-0 int64 tensor with the seed masked to 64
    bits, bit-identical to today), ``threefry2x32`` (shape ``(2,)`` int32),
    or ``philox4x32_10`` (shape ``(4,)`` int32). Accepts the canonical
    strings or ``Algorithm`` members. The 2/4 words of the counter-based
    keys are derived deterministically from the masked seed via SplitMix64
    (see the module docstring — the SAME derivation inside and outside a
    trace).

    Args:
        seed: A Python int (any value; masked to 64 bits).
        algorithm: Canonical algorithm name (str) or ``Algorithm`` member.
            Default: ``splitmix64``.

    Returns:
        ``core.Tensor`` (outside a trace) or ``core.SymbolicTensor`` (inside).
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(
            f"etl.random.key: seed must be a Python int, got "
            f"{type(seed).__name__}"
        )
    alg = _algorithm_arg(algorithm)
    shape, _ = algorithm_key_type(alg)
    if alg == DEFAULT_ALGORITHM:
        # splitmix64 (v1, unchanged): the 64-bit masked seed itself.
        value = np.array(seed & _MASK64, dtype=np.uint64).view(np.int64).reshape(())
    else:
        value = _derive_key_words(seed, shape[0], _KEY_SALTS[alg]).reshape(shape)
    tensor = core.tensor(value)
    try:
        _utils.check_in_trace()
    except core.TraceError:
        return tensor  # no active trace → concrete key (usable as graph input)
    return _constant.constant(tensor)


def split(key):
    """Deterministically split one key into two independent keys.

    Graph op (outside a trace → ``TraceError``). The two derived keys are
    decorrelated states (the algorithm is inferred from the key operand's
    shape/dtype); the same input key always yields the same pair. Returns
    ``(key_a, key_b)`` of ``SymbolicTensor``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.split")
    return (_key_mix(builder, key_t, 0, alg, loc),
            _key_mix(builder, key_t, _GOLDEN, alg, loc))


def split_n(key, n: int):
    """Split one key into ``n`` independent keys (helper).

    Graph op. ``n`` must be a non-negative Python int (static). Returns a
    tuple of ``n`` ``SymbolicTensor`` keys, derived as
    ``mix(key ^ (i * GOLDEN))``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.split_n")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(
            f"random.split_n: n must be a Python int, got {type(n).__name__}"
        )
    if n < 0:
        raise ValueError(f"random.split_n: n must be >= 0, got {n}")
    return tuple(
        _key_mix(builder, key_t, i * _GOLDEN, alg, loc) for i in range(n)
    )


def uniform(key, shape, low=0.0, high=1.0, dtype=core.float32):
    """Draw uniform values in ``[low, high)`` from the key's stream.

    Graph op. ``shape``: int or tuple/list of int/Dim/DimExpr. ``low``/``high``
    may be Python floats or SYMBOLIC float tensors broadcastable against
    ``shape`` (e.g. a runtime-computed bound of shape ``(m,)`` with
    ``shape=(num_sample, m)``). The result shape is the broadcast of
    ``shape``, ``low`` and ``high``. ``dtype``: floating (default float32).
    Deterministic: the same key + operands ⇒ bit-identical values.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.uniform")
    shape_t = _shape(shape, "random.uniform")
    out_dtype = _out_dtype(dtype, "random.uniform", _FLOAT_KINDS)
    low_t = _param(low, "random.uniform low", (int, float), loc)
    high_t = _param(high, "random.uniform high", (int, float), loc)
    op = builder.create(
        "random_uniform",
        operands=(key_t.value, low_t.value, high_t.value),
        attributes={"shape": shape_t, "dtype": out_dtype, "algorithm": alg},
        location=loc,
    )
    return _wrap(op, loc)


def normal(key, shape, mean=0.0, std=1.0, dtype=core.float32):
    """Draw normal values ``mean + std * N(0,1)`` from the key's stream.

    Graph op. ``shape``: int or tuple/list of int/Dim/DimExpr.
    ``mean``/``std``: Python floats or symbolic float tensors broadcastable
    against ``shape``. ``dtype``: floating (default float32). Deterministic.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.normal")
    shape_t = _shape(shape, "random.normal")
    out_dtype = _out_dtype(dtype, "random.normal", _FLOAT_KINDS)
    mean_t = _param(mean, "random.normal mean", (int, float), loc)
    std_t = _param(std, "random.normal std", (int, float), loc)
    op = builder.create(
        "random_normal",
        operands=(key_t.value, mean_t.value, std_t.value),
        attributes={"shape": shape_t, "dtype": out_dtype, "algorithm": alg},
        location=loc,
    )
    return _wrap(op, loc)


def randint(key, shape, low, high, dtype=core.int32):
    """Draw integer values in ``[low, high)`` (high exclusive).

    Graph op. ``low``/``high`` are REQUIRED: Python ints or symbolic integer
    tensors broadcastable against ``shape``; static ``low >= high`` is a
    trace-time ``ValueError`` (runtime validation for tensor operands).
    ``dtype``: integer (default int32). Deterministic.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.randint")
    shape_t = _shape(shape, "random.randint")
    out_dtype = _out_dtype(dtype, "random.randint", _INT_KINDS)
    low_t = _param(low, "random.randint low", (int,), loc)
    high_t = _param(high, "random.randint high", (int,), loc)
    if type(low) is int and type(high) is int and low >= high:
        raise ValueError(
            f"random.randint: high must be greater than low, got low={low}, "
            f"high={high}"
        )
    op = builder.create(
        "random_randint",
        operands=(key_t.value, low_t.value, high_t.value),
        attributes={"shape": shape_t, "dtype": out_dtype, "algorithm": alg},
        location=loc,
    )
    return _wrap(op, loc)


def permutation(key, n, dtype=core.int32):
    """Draw a uniformly random permutation of ``0..n-1``.

    Graph op. ``n`` may be a static Python int (>= 0) or a SYMBOLIC rank-0
    integer tensor scalar (the result length is then runtime-dynamic). The
    result is a 1-D tensor whose values are exactly ``0..n-1`` shuffled.
    A static Python int traces to a STATIC ``(n,)`` result shape (so
    ``etl.cond``/``etl.while_loop`` branch unification works for
    static-size populations); a symbolic rank-0 ``n`` traces to ``(None,)``.
    ``dtype``: integer (default int32). Deterministic.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.permutation")
    out_dtype = _out_dtype(dtype, "random.permutation", _INT_KINDS)
    if isinstance(n, core.SymbolicTensor):
        if n.dtype.kind not in _INT_KINDS or n.shape != ():
            raise core.ShapeError(
                "random.permutation: symbolic n must be a rank-0 integer "
                f"tensor, got dtype={n.dtype} shape={n.shape}"
            )
        n_t = n
    elif type(n) is int:
        if n < 0:
            raise ValueError(f"random.permutation: n must be >= 0, got {n}")
        n_t = _utils.as_operand(n, location=loc)
    else:
        raise TypeError(
            f"random.permutation: n must be a Python int or a rank-0 "
            f"integer SymbolicTensor, got {type(n).__name__}"
        )
    attributes = {"dtype": out_dtype, "algorithm": alg}
    if type(n) is int:
        # Record the static population size so the traced result shape is
        # (n,) — keeps cond/while branch unification working for static-size
        # populations. A symbolic n keeps the shape runtime-dynamic (None,).
        attributes["n"] = n
    op = builder.create(
        "random_permutation",
        operands=(key_t.value, n_t.value),
        attributes=attributes,
        location=loc,
    )
    return _wrap(op, loc)


def multinomial(key, input, num_samples: int):
    """Draw ``num_samples`` indices from a 1-D probability distribution.

    Graph op. ``input``: a 1-D floating ``SymbolicTensor`` of non-negative
    probabilities summing to 1 (np.random.choice(a=input.shape[0], p=input)
    semantics; validated at run time). ``num_samples``: static non-negative
    Python int. Result: 1-D int32 tensor of drawn indices. Deterministic.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t, alg = _check_key(key, "random.multinomial")
    if not isinstance(input, core.SymbolicTensor):
        if isinstance(input, core.Tensor):
            raise core.TraceError(
                "random.multinomial: got a concrete Tensor input — etl has "
                "no eager mode; pass the probabilities as an explicit graph "
                "input"
            )
        raise TypeError(
            f"random.multinomial: input must be a SymbolicTensor (1-D "
            f"probabilities), got {type(input).__name__}"
        )
    if input.dtype.kind not in _FLOAT_KINDS:
        raise core.DTypeError(
            f"random.multinomial: input must have a floating dtype, got "
            f"{input.dtype}"
        )
    if len(input.shape) != 1:
        raise core.ShapeError(
            f"random.multinomial: input must be 1-D, got shape "
            f"{input.shape}"
        )
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise TypeError(
            f"random.multinomial: num_samples must be a Python int, got "
            f"{type(num_samples).__name__}"
        )
    if num_samples < 0:
        raise ValueError(
            f"random.multinomial: num_samples must be >= 0, got {num_samples}"
        )
    op = builder.create(
        "random_multinomial",
        operands=(key_t.value, input.value),
        attributes={
            "num_samples": num_samples,
            "dtype": np.dtype("int32"),
            "algorithm": alg,
        },
        location=loc,
    )
    return _wrap(op, loc)
