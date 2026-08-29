"""Key-based functional RNG — the ``etl.random`` frontend (graph ops).

Design (binding — see ``etl/CONTEXT.md``, section "etl.random"):

- **Key representation**: a rank-0 int64 tensor holding a 64-bit state
  (two's complement). ``key(seed)`` masks the seed to 64 bits.
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
- **Backends**: numpy interpreter only in v1. Compiler backends
  (stablehlo/iree/xla/tvm) reject every random op with an explicit
  ``BackendError`` (no stablehlo writer) — never silent fallback. Transforms
  (``vmap``/``grad``/...) have no rules for random ops → ``TransformError``.
"""
from __future__ import annotations

import numpy as np

from etl import core

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
]

#: SplitMix64 golden gamma — also the ``split`` second salt.
_GOLDEN = 0x9E3779B97F4A7C15
_MASK64 = 0xFFFFFFFFFFFFFFFF

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


def _check_key(key, name: str) -> "core.SymbolicTensor":
    """Validate a key operand: a rank-0 int64 SymbolicTensor."""
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
    if key.dtype != np.dtype("int64") or key.shape != ():
        raise core.ShapeError(
            f"{name}: key must be a rank-0 int64 tensor, got "
            f"dtype={key.dtype} shape={key.shape}"
        )
    return key


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


def _key_mix(builder, key_t, salt: int, loc) -> "core.SymbolicTensor":
    """Build one ``random_key_mix`` op (the split/split_n building block)."""
    op = builder.create(
        "random_key_mix",
        operands=(key_t.value,),
        attributes={"salt": salt},
        location=loc,
    )
    return _wrap(op, loc)


# --- public API ---------------------------------------------------------------


def key(seed: int):
    """Create a key from an int seed (polymorphic creator).

    Outside a trace: returns a concrete rank-0 int64 ``core.Tensor`` (usable
    as an explicit graph input). Inside a trace: builds a Constant op and
    returns the ``SymbolicTensor``. The seed is masked to 64 bits.

    Args:
        seed: A Python int (any value; masked to 64 bits).

    Returns:
        ``core.Tensor`` (outside a trace) or ``core.SymbolicTensor`` (inside).
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(
            f"etl.random.key: seed must be a Python int, got "
            f"{type(seed).__name__}"
        )
    value = np.array(seed & _MASK64, dtype=np.uint64).view(np.int64).reshape(())
    tensor = core.tensor(value)
    try:
        _utils.check_in_trace()
    except core.TraceError:
        return tensor  # no active trace → concrete key (usable as graph input)
    return _constant.constant(tensor)


def split(key):
    """Deterministically split one key into two independent keys.

    Graph op (outside a trace → ``TraceError``). The two derived keys are
    decorrelated 64-bit states; the same input key always yields the same
    pair. Returns ``(key_a, key_b)`` of ``SymbolicTensor``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t = _check_key(key, "random.split")
    return (_key_mix(builder, key_t, 0, loc),
            _key_mix(builder, key_t, _GOLDEN, loc))


def split_n(key, n: int):
    """Split one key into ``n`` independent keys (helper).

    Graph op. ``n`` must be a non-negative Python int (static). Returns a
    tuple of ``n`` ``SymbolicTensor`` keys, derived as
    ``mix(key ^ (i * GOLDEN))``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t = _check_key(key, "random.split_n")
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(
            f"random.split_n: n must be a Python int, got {type(n).__name__}"
        )
    if n < 0:
        raise ValueError(f"random.split_n: n must be >= 0, got {n}")
    return tuple(
        _key_mix(builder, key_t, i * _GOLDEN, loc) for i in range(n)
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
    key_t = _check_key(key, "random.uniform")
    shape_t = _shape(shape, "random.uniform")
    out_dtype = _out_dtype(dtype, "random.uniform", _FLOAT_KINDS)
    low_t = _param(low, "random.uniform low", (int, float), loc)
    high_t = _param(high, "random.uniform high", (int, float), loc)
    op = builder.create(
        "random_uniform",
        operands=(key_t.value, low_t.value, high_t.value),
        attributes={"shape": shape_t, "dtype": out_dtype},
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
    key_t = _check_key(key, "random.normal")
    shape_t = _shape(shape, "random.normal")
    out_dtype = _out_dtype(dtype, "random.normal", _FLOAT_KINDS)
    mean_t = _param(mean, "random.normal mean", (int, float), loc)
    std_t = _param(std, "random.normal std", (int, float), loc)
    op = builder.create(
        "random_normal",
        operands=(key_t.value, mean_t.value, std_t.value),
        attributes={"shape": shape_t, "dtype": out_dtype},
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
    key_t = _check_key(key, "random.randint")
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
        attributes={"shape": shape_t, "dtype": out_dtype},
        location=loc,
    )
    return _wrap(op, loc)


def permutation(key, n, dtype=core.int32):
    """Draw a uniformly random permutation of ``0..n-1``.

    Graph op. ``n`` may be a static Python int (>= 0) or a SYMBOLIC rank-0
    integer tensor scalar (the result length is then runtime-dynamic). The
    result is a 1-D tensor whose values are exactly ``0..n-1`` shuffled.
    ``dtype``: integer (default int32). Deterministic.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    key_t = _check_key(key, "random.permutation")
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
    op = builder.create(
        "random_permutation",
        operands=(key_t.value, n_t.value),
        attributes={"dtype": out_dtype},
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
    key_t = _check_key(key, "random.multinomial")
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
        attributes={"num_samples": num_samples, "dtype": np.dtype("int32")},
        location=loc,
    )
    return _wrap(op, loc)
