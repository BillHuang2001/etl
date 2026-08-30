"""SplitMix64 subgraph expansion for the StableHLO exporter (random ops).

The numpy kernels (`etl/backends/numpy/kernels/random.py`) are the reference
implementation: every random op is a deterministic function of its rank-0
int64 key. This module re-derives those kernels as INLINE i64 elementwise
subgraphs so compiler backends can run the same streams — deliberately NOT
``stablehlo.rng``, whose algorithm is implementation-defined and would break
the same-key ⇒ same-values determinism contract.

Bit-identity argument: the kernels compute in uint64 with wrapping
arithmetic. Two's-complement i64 xor/mul/add/and are bit-identical to the
uint64 versions (mod-2^64 wrap is the same), and a LOGICAL right shift
equals ``shift_right_arithmetic`` + ``and`` with a low-bits mask — so the
i64 graph below produces exactly the numpy kernel's uint64 words. The
f64 stages (uniform scaling, Box–Muller, randint floor-scaling) use the
same IEEE ops (add/sub/mul/floor are exact per-op; sqrt/log/cos may differ
in the last ulp across libm implementations — uniform/randint/permutation/
key_mix are bit-identical by construction, normal is expected identical or
within 1 ulp, reported separately).

The writer calls ``emit_random_op(writer, op)``; all emission goes through
the Writer's SSA/type helpers (this module is writer-private, not part of
the public API). All expansions require STATIC shapes (the v1 compiler
backends reject symbolic dims); dynamic cases raise ``core.BackendError``.
"""

from __future__ import annotations

import numpy as np

from etl.core import BackendError

# SplitMix64 constants (uint64 hex → emitted as two's-complement i64).
_GOLDEN = 0x9E3779B97F4A7C15
_B_MUL = 0xBF58476D1CE4E5B9
_C_MUL = 0x94D049BB133111EB
_SIGN_FLIP = 0x8000000000000000
# Logical-shift masks: (1 << (64 - shift)) - 1 (all < 2^63 → positive i64).
_MASK30 = 0x3FFFFFFFF  # 2^34 - 1 (shift 30)
_MASK27 = 0x1FFFFFFFFF  # 2^37 - 1 (shift 27)
_MASK31 = 0x1FFFFFFFF  # 2^33 - 1 (shift 31)
_MASK11 = 0x1FFFFFFFFFFFFF  # 2^53 - 1
_INV_2P53 = 1.0 / 2.0**53
_TWO_PI = 2.0 * np.pi
# Per-op stream salts (must match kernels/random.py `_SALTS` EXACTLY).
_SALTS = {
    "uniform": 0x243F6A8885A308D3,
    "normal": 0x13198A2E03707344,
    "randint": 0xA4093822299F31D0,
    "permutation": 0x082EFA98EC4E6C89,
}

_I64 = np.dtype("int64")
_F64 = np.dtype("float64")


def _i64(w, value, shape):
    """Emit a scalar/static i64 constant broadcast to ``shape`` (numpy
    int → two's-complement i64) → ``(name, lines)``.

    The value is wrapped mod 2^64 first (bit-preserving uint64 → i64
    view): the SplitMix64 constants exceed 2^63−1, the frontend's
    ``split_n`` salts are ``i * GOLDEN`` computed UNWRAPPED (> 2^64 for
    i ≥ 2), and numpy 2.x raises OverflowError on any of them when
    converted through ``np.asarray(v, dtype=np.int64)``. Wrapping matches
    the kernels' uint64 arithmetic exactly (the kernels consume salts and
    constants mod 2^64)."""
    return w._scalar_constant_for(
        _I64, int(np.asarray(value % (1 << 64), dtype=np.uint64).view(np.int64)), shape
    )


def _static_shape(w, op, attr_shape) -> tuple:
    """The op's declared ``shape`` attribute as a tuple of static ints
    (symbolic dims → BackendError, matching the v1 static-shape scope)."""
    shape = tuple(attr_shape)
    for d in shape:
        if not isinstance(d, (int, np.integer)) or isinstance(d, bool):
            raise BackendError(
                f"stablehlo export: op '{op.name}'{w._loc(op)} declares the "
                f"symbolic shape {shape!r} — random-op expansion requires "
                "static shapes (the v1 compiler backends reject symbolic "
                "dims; use the numpy backend for symbolic shapes)"
            )
    return tuple(int(d) for d in shape)


def _count_of(shape: tuple) -> int:
    return int(np.prod(shape)) if shape else 1


def _emit_mix3(w, z_name, z_shape, lines) -> str:
    """The three SplitMix64 mixing steps over an i64 tensor ``z_name`` of
    static shape ``z_shape``; appends lines, returns the result name.

    Each step ``z ^ ((z >> s) & mask)`` computes a LOGICAL right shift in
    two's-complement (arithmetic shift + low-bits mask) — bit-identical to
    the numpy kernel's uint64 ``>>``.
    """
    for shift, mul in ((30, _B_MUL), (27, _C_MUL)):
        amt, extra = _i64(w, shift, z_shape)
        lines.extend(extra)
        s = w._new_name()
        lines.append(
            f"{s} = stablehlo.shift_right_arithmetic {z_name}, {amt} : "
            f"{w._type_str(_I64, z_shape)}"
        )
        mask, extra = _i64(w, {30: _MASK30, 27: _MASK27}[shift], z_shape)
        lines.extend(extra)
        m = w._new_name()
        lines.append(f"{m} = stablehlo.and {s}, {mask} : {w._type_str(_I64, z_shape)}")
        x = w._new_name()
        lines.append(f"{x} = stablehlo.xor {z_name}, {m} : {w._type_str(_I64, z_shape)}")
        c, extra = _i64(w, mul, z_shape)
        lines.extend(extra)
        z_name = w._new_name()
        lines.append(
            f"{z_name} = stablehlo.multiply {x}, {c} : {w._type_str(_I64, z_shape)}"
        )
    amt, extra = _i64(w, 31, z_shape)
    lines.extend(extra)
    s = w._new_name()
    lines.append(
        f"{s} = stablehlo.shift_right_arithmetic {z_name}, {amt} : "
        f"{w._type_str(_I64, z_shape)}"
    )
    mask, extra = _i64(w, _MASK31, z_shape)
    lines.extend(extra)
    m = w._new_name()
    lines.append(f"{m} = stablehlo.and {s}, {mask} : {w._type_str(_I64, z_shape)}")
    x = w._new_name()
    lines.append(f"{x} = stablehlo.xor {z_name}, {m} : {w._type_str(_I64, z_shape)}")
    return x


def _emit_words(w, seed_name, count, lines) -> str:
    """``count`` SplitMix64 words as a ``(count,)`` i64 tensor from the
    scalar i64 ``seed_name``: word_i = mix3(seed + (i + 1) * GOLDEN)
    (matches ``kernels/random.py`` ``_words`` exactly)."""
    if count == 0:
        name = w._new_name()
        lines.append(f"{name} = stablehlo.constant dense<> : tensor<0xi64>")
        return name
    base, extra = _i64(w, _GOLDEN, ())
    lines.extend(extra)
    src = base
    base = w._new_name()
    lines.append(f"{base} = stablehlo.add {seed_name}, {src} : tensor<i64>")
    iota = w._new_name()
    lines.append(f"{iota} = stablehlo.iota dim = 0 : tensor<{count}xi64>")
    gold, extra = _i64(w, _GOLDEN, (count,))
    lines.extend(extra)
    prod = w._new_name()
    lines.append(
        f"{prod} = stablehlo.multiply {iota}, {gold} : tensor<{count}xi64>"
    )
    b = w._new_name()
    lines.append(
        f'{b} = "stablehlo.broadcast_in_dim"({base}) '
        f"{{broadcast_dimensions = {w._i64_array(())}}} : "
        f"(tensor<i64>) -> tensor<{count}xi64>"
    )
    states = w._new_name()
    lines.append(
        f"{states} = stablehlo.add {prod}, {b} : tensor<{count}xi64>"
    )
    return _emit_mix3(w, states, (count,), lines)


def _emit_uniforms(w, seed_name, shape, lines) -> str:
    """float64 uniform draws ``u_i = (words_i >> 11) * 2^-53`` reshaped to
    ``shape`` (matches ``kernels/random.py`` ``_uniforms``)."""
    count = _count_of(shape)
    if count == 0:
        name = w._new_name()
        lines.append(f"{name} = stablehlo.constant dense<> : tensor<0xf64>")
        cur = name
    else:
        words = _emit_words(w, seed_name, count, lines)
        amt, extra = _i64(w, 11, (count,))
        lines.extend(extra)
        s = w._new_name()
        lines.append(
            f"{s} = stablehlo.shift_right_arithmetic {words}, {amt} : "
            f"tensor<{count}xi64>"
        )
        mask, extra = _i64(w, _MASK11, (count,))
        lines.extend(extra)
        m = w._new_name()
        lines.append(f"{m} = stablehlo.and {s}, {mask} : tensor<{count}xi64>")
        c = w._new_name()
        lines.append(
            f"{c} = stablehlo.convert {m} : (tensor<{count}xi64>) -> "
            f"tensor<{count}xf64>"
        )
        inv, extra = _scalar_f64(w, _INV_2P53, (count,))
        lines.extend(extra)
        u = w._new_name()
        lines.append(
            f"{u} = stablehlo.multiply {c}, {inv} : tensor<{count}xf64>"
        )
        cur = u
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {cur} : (tensor<{count}xf64>) -> "
            f"{w._type_str(_F64, shape)}"
        )
        cur = r
    return cur


def _scalar_f64(w, value, shape):
    return w._scalar_constant_for(_F64, value, shape)


def _emit_to_dtype_broadcast(w, v, target_dtype, target_shape, lines, op) -> str:
    """Convert operand ``v`` to ``target_dtype`` and numpy-broadcast it to
    the static ``target_shape`` (trailing alignment); returns the SSA name.
    Raises BackendError when numpy broadcasting would fail."""
    src_shape = tuple(v.type.shape)
    src_dtype = np.dtype(v.type.dtype)
    if len(src_shape) > len(target_shape):
        raise BackendError(
            f"stablehlo export: op '{op.name}'{w._loc(op)} cannot broadcast "
            f"an operand of shape {src_shape!r} to the target shape "
            f"{target_shape!r} (numpy broadcasting would fail)"
        )
    pad = len(target_shape) - len(src_shape)
    for sd, td in zip(src_shape, target_shape[pad:]):
        if sd != td and sd != 1:
            raise BackendError(
                f"stablehlo export: op '{op.name}'{w._loc(op)} cannot "
                f"broadcast shapes {src_shape!r} and {target_shape!r} "
                "(numpy broadcasting would fail)"
            )
    name = w._name(v)
    if src_dtype != target_dtype:
        t = w._new_name()
        lines.append(
            f"{t} = stablehlo.convert {name} : ({w._vt(v.type)}) -> "
            f"{w._type_str(target_dtype, src_shape)}"
        )
        name = t
    if src_shape == target_shape:
        return name
    b = w._new_name()
    dims = list(range(pad, len(target_shape)))
    lines.append(
        f'{b} = "stablehlo.broadcast_in_dim"({name}) '
        f"{{broadcast_dimensions = {w._i64_array(dims)}}} : "
        f"({w._type_str(target_dtype, src_shape)}) -> "
        f"{w._type_str(target_dtype, target_shape)}"
    )
    return b


def _emit_seed(w, op, key, salt, lines) -> str:
    """seed = key ^ salt (scalar i64; matches the kernels' per-op salt)."""
    salt_c, extra = _i64(w, salt, ())
    lines.extend(extra)
    seed = w._new_name()
    lines.append(f"{seed} = stablehlo.xor {w._name(key)}, {salt_c} : tensor<i64>")
    return seed


def _emit_random_key_mix(w, op) -> str:
    """``random_key_mix`` → mix3(key ^ salt): the split/split_n building
    block (matches ``kernels/random.py`` ``_key_mix_kernel`` + ``_mix_scalar``:
    ``_mix_scalar(x) = mix3(x + GOLDEN)``)."""
    lines = []
    seed = _emit_seed(w, op, op.operands[0], op.attributes["salt"], lines)
    z, extra = _i64(w, _GOLDEN, ())
    lines.extend(extra)
    src = z
    z = w._new_name()
    lines.append(f"{z} = stablehlo.add {seed}, {src} : tensor<i64>")
    out = _emit_mix3(w, z, (), lines)
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_uniform(w, op) -> str:
    """``random_uniform`` → u = uniforms(seed, shape); vals =
    (low + u * (high - low)).astype(dtype) — all f64, then converted."""
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    lines = []
    seed = _emit_seed(w, op, op.operands[0], _SALTS["uniform"], lines)
    u = _emit_uniforms(w, seed, shape, lines)
    lo = _emit_to_dtype_broadcast(w, op.operands[1], _F64, shape, lines, op)
    hi = _emit_to_dtype_broadcast(w, op.operands[2], _F64, shape, lines, op)
    diff = w._new_name()
    lines.append(f"{diff} = stablehlo.subtract {hi}, {lo} : {w._type_str(_F64, shape)}")
    prod = w._new_name()
    lines.append(f"{prod} = stablehlo.multiply {u}, {diff} : {w._type_str(_F64, shape)}")
    s = w._new_name()
    lines.append(f"{s} = stablehlo.add {lo}, {prod} : {w._type_str(_F64, shape)}")
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {s} : ({w._type_str(_F64, shape)}) -> "
        f"{w._type_str(out_dtype, shape)}"
    )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_normal(w, op) -> str:
    """``random_normal`` → Box–Muller over 2*count words (matches the
    kernel: u1 = max(word>>11 * 2^-53, 2^-53), u2 = ..., z =
    sqrt(-2 log u1) cos(2π u2); vals = (mean + z * std).astype(dtype))."""
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    count = _count_of(shape)
    lines = []
    seed = _emit_seed(w, op, op.operands[0], _SALTS["normal"], lines)
    if count == 0:
        z = w._new_name()
        lines.append(f"{z} = stablehlo.constant dense<> : tensor<0xf64>")
        cur = z
    else:
        words = _emit_words(w, seed, 2 * count, lines)
        wr = w._new_name()
        lines.append(
            f"{wr} = stablehlo.reshape {words} : (tensor<{2 * count}xi64>) -> "
            f"tensor<{count}x2xi64>"
        )
        # u1 = words[:, 0] >> 11, u2 = words[:, 1] >> 11 (logical), × 2^-53.
        u1, u2 = [], []
        for col, out in ((0, u1), (1, u2)):
            sl = w._new_name()
            lines.append(
                f'{sl} = "stablehlo.slice"({wr}) {{start_indices = '
                f"array<i64: 0, {col}>, limit_indices = "
                f"array<i64: {count}, {col + 1}>, strides = array<i64: 1, 1>}}"
                f" : (tensor<{count}x2xi64>) -> tensor<{count}x1xi64>"
            )
            sv = w._new_name()
            lines.append(
                f"{sv} = stablehlo.reshape {sl} : (tensor<{count}x1xi64>) -> "
                f"tensor<{count}xi64>"
            )
            amt, extra = _i64(w, 11, (count,))
            lines.extend(extra)
            sh = w._new_name()
            lines.append(
                f"{sh} = stablehlo.shift_right_arithmetic {sv}, {amt} : "
                f"tensor<{count}xi64>"
            )
            mask, extra = _i64(w, _MASK11, (count,))
            lines.extend(extra)
            m = w._new_name()
            lines.append(f"{m} = stablehlo.and {sh}, {mask} : tensor<{count}xi64>")
            cv = w._new_name()
            lines.append(
                f"{cv} = stablehlo.convert {m} : (tensor<{count}xi64>) -> "
                f"tensor<{count}xf64>"
            )
            inv, extra = _scalar_f64(w, _INV_2P53, (count,))
            lines.extend(extra)
            u = w._new_name()
            lines.append(
                f"{u} = stablehlo.multiply {cv}, {inv} : tensor<{count}xf64>"
            )
            out.append(u)
        # u1 = maximum(u1, 2^-53) (kernel's log(0) guard).
        eps, extra = _scalar_f64(w, _INV_2P53, (count,))
        lines.extend(extra)
        u1m = w._new_name()
        lines.append(
            f"{u1m} = stablehlo.maximum {u1[0]}, {eps} : tensor<{count}xf64>"
        )
        # z = sqrt(-2 log u1) * cos(2π u2)
        two, extra = _scalar_f64(w, -2.0, (count,))
        lines.extend(extra)
        lg = w._new_name()
        lines.append(f"{lg} = stablehlo.log {u1m} : tensor<{count}xf64>")
        prod = w._new_name()
        lines.append(f"{prod} = stablehlo.multiply {two}, {lg} : tensor<{count}xf64>")
        rt = w._new_name()
        lines.append(f"{rt} = stablehlo.sqrt {prod} : tensor<{count}xf64>")
        twopi, extra = _scalar_f64(w, _TWO_PI, (count,))
        lines.extend(extra)
        ang = w._new_name()
        lines.append(
            f"{ang} = stablehlo.multiply {twopi}, {u2[0]} : tensor<{count}xf64>"
        )
        cs = w._new_name()
        lines.append(f"{cs} = stablehlo.cosine {ang} : tensor<{count}xf64>")
        z = w._new_name()
        lines.append(f"{z} = stablehlo.multiply {rt}, {cs} : tensor<{count}xf64>")
        cur = z
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {cur} : (tensor<{count}xf64>) -> "
            f"{w._type_str(_F64, shape)}"
        )
        cur = r
    mean = _emit_to_dtype_broadcast(w, op.operands[1], _F64, shape, lines, op)
    std = _emit_to_dtype_broadcast(w, op.operands[2], _F64, shape, lines, op)
    zp = w._new_name()
    lines.append(f"{zp} = stablehlo.multiply {cur}, {std} : {w._type_str(_F64, shape)}")
    s = w._new_name()
    lines.append(f"{s} = stablehlo.add {mean}, {zp} : {w._type_str(_F64, shape)}")
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {s} : ({w._type_str(_F64, shape)}) -> "
        f"{w._type_str(out_dtype, shape)}"
    )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_randint(w, op) -> str:
    """``random_randint`` → low/high cast to int64, span = high - low
    (wrapping int64 — matches the kernel's numpy int64 subtraction);
    vals = (low + floor(u * span)).astype(dtype) in f64."""
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    lines = []
    seed = _emit_seed(w, op, op.operands[0], _SALTS["randint"], lines)
    u = _emit_uniforms(w, seed, shape, lines)
    lo_i = _emit_to_dtype_broadcast(w, op.operands[1], _I64, shape, lines, op)
    hi_i = _emit_to_dtype_broadcast(w, op.operands[2], _I64, shape, lines, op)
    span_i = w._new_name()
    lines.append(f"{span_i} = stablehlo.subtract {hi_i}, {lo_i} : {w._type_str(_I64, shape)}")
    span_f = w._new_name()
    lines.append(
        f"{span_f} = stablehlo.convert {span_i} : ({w._type_str(_I64, shape)}) -> "
        f"{w._type_str(_F64, shape)}"
    )
    prod = w._new_name()
    lines.append(f"{prod} = stablehlo.multiply {u}, {span_f} : {w._type_str(_F64, shape)}")
    fl = w._new_name()
    lines.append(f"{fl} = stablehlo.floor {prod} : {w._type_str(_F64, shape)}")
    lo_f = w._new_name()
    lines.append(
        f"{lo_f} = stablehlo.convert {lo_i} : ({w._type_str(_I64, shape)}) -> "
        f"{w._type_str(_F64, shape)}"
    )
    s = w._new_name()
    lines.append(f"{s} = stablehlo.add {lo_f}, {fl} : {w._type_str(_F64, shape)}")
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {s} : ({w._type_str(_F64, shape)}) -> "
        f"{w._type_str(out_dtype, shape)}"
    )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_permutation(w, op) -> str:
    """``random_permutation`` → words = _words(seed, n); stable argsort of
    the UNSIGNED words (sign-bit flip in i64 is monotonic and order-
    preserving), ties (astronomically rare) resolve by index order —
    exactly ``np.argsort(words, kind="stable")``."""
    n = op.attributes.get("n")
    out_dtype = np.dtype(op.attributes["dtype"])
    if n is None:
        raise BackendError(
            f"stablehlo export: op 'random_permutation'{w._loc(op)} has a "
            "runtime population size operand (attr n=None) — the SplitMix64 "
            "expansion requires a static n in v1 (the compiler backends "
            "reject dynamic result lengths; use the numpy backend)"
        )
    n = int(n)
    if n < 0:
        raise BackendError(
            f"stablehlo export: op 'random_permutation'{w._loc(op)} has "
            f"negative n={n} (the numpy kernel raises ValueError)"
        )
    lines = []
    if n == 0:
        out = w._new_name()
        lines.append(f"{out} = stablehlo.constant dense<> : {w._type_str(out_dtype, (0,))}")
        w._names[id(op.result)] = out
        return "\n".join(lines)
    seed = _emit_seed(w, op, op.operands[0], _SALTS["permutation"], lines)
    words = _emit_words(w, seed, n, lines)
    flip, extra = _i64(w, _SIGN_FLIP, (n,))
    lines.extend(extra)
    wf = w._new_name()
    lines.append(f"{wf} = stablehlo.xor {words}, {flip} : tensor<{n}xi64>")
    si = w._emit_stable_argsort(wf, _I64, (n,), 0, lines)
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {si} : (tensor<{n}xi64>) -> "
        f"{w._type_str(out_dtype, (n,))}"
    )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def emit_random_op(w, op) -> str:
    """Dispatch one random-op emission (called by ``Writer._emit_op`` for
    the ``SPECIAL_EMITTERS`` family "random")."""
    if op.name == "random_key_mix":
        return _emit_random_key_mix(w, op)
    if op.name == "random_uniform":
        return _emit_random_uniform(w, op)
    if op.name == "random_normal":
        return _emit_random_normal(w, op)
    if op.name == "random_randint":
        return _emit_random_randint(w, op)
    if op.name == "random_permutation":
        return _emit_random_permutation(w, op)
    raise BackendError(
        f"stablehlo export: op '{op.name}'{w._loc(op)} has no SplitMix64 "
        "expansion — random ops not in the expansion family stay deferred"
    )
