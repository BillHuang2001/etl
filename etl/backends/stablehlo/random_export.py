"""SplitMix64 subgraph expansion for the StableHLO exporter (random ops).

The numpy kernels (`etl/backends/numpy/kernels/random.py`) are the reference
implementation: every random op is a deterministic function of its rank-0
int64 key. This module re-derives those kernels as INLINE i64 elementwise
subgraphs so compiler backends can run the same streams — deliberately NOT
``stablehlo.rng``, whose algorithm is implementation-defined and would break
the same-key ⇒ same-values determinism contract.

Multi-algorithm lowering (the ``algorithm`` attribute on random ops; absent
⇒ ``splitmix64``, the pre-multi-algorithm behavior, so existing graphs are
unchanged): besides the splitmix64 inline i64 expansion, the
``threefry2x32`` and ``philox4x32_10`` algorithms lower as INLINE bit-exact
i32/ui32 elementwise expansions of the pinned Random123/JAX ciphers (the
default — no target support required), or — when the writer's
``rng_bit_generator`` option (fed from the backend's capability flag) is
set — as a native ``stablehlo.rng_bit_generator`` call with the verified
state layout ``[key0, key1, ctr...]`` (key words FIRST, counter words
zero). Key layouts: splitmix64 → rank-0 int64 (unchanged); threefry2x32 →
``(2,)`` int32; philox4x32_10 → ``(4,)`` int32 (the pinned key'
definition folds the salt into all four words; the INLINE cipher consumes
them CYCLICALLY — round r uses the pair ``(k'_{2r mod 4},
k'_{(2r+1) mod 4})`` plus the per-round Weyl bumps, mirroring the numpy
kernel — while the NATIVE ``rng_bit_generator`` state carries only words
0–1, a documented v1 caveat, see ``_emit_rng_bit_generator``). The
word-stream contract (per-op salt folding, per-block counters, element→word
mapping, post-processing) is pinned in the module body below and matches
the numpy kernels bit-for-bit (32-bit words scale as ``u = w * 2^-32`` —
the full word — per ``_word_uniforms``; the ``(w >> 11) * 2^-53`` form is
the splitmix64 64-bit-word scaling only).

Bit-identity argument: the kernels compute in uint64 with wrapping
arithmetic. Two's-complement i64 xor/mul/add/and are bit-identical to the
uint64 versions (mod-2^64 wrap is the same), and the uint64 LOGICAL right
shift is emitted directly as ``stablehlo.shift_right_logical`` (bit-exact
on the i64 bit pattern) — so the i64 graph below produces exactly the
numpy kernel's uint64 words. The threefry inline path argues the same way
in i32 (add/xor bit-identical; the rotl uses ``shift_right_logical`` so the
u32 rotation is exact); the philox inline path computes in ui32 with the
i64 mulhilo chain (ui32→i64 zero-extend multiply + logical shift + truncate
= the exact 64-bit high half). The f64 stages (uniform scaling, Box–Muller
for non-f32 outputs, randint floor-scaling) use the same IEEE ops as the
kernels: uniform/randint/permutation/key_mix are bit-identical by
construction, f64-output normal is identical or within 1 ulp (libm noise).

``random_normal`` f32 fast path — DOCUMENTED v1 deviation (compiler
backends only): for float32 outputs (the default dtype and the hot path —
e.g. the new-evox DE crossover masks), the Box–Muller tail runs in
float32: ``(word >> 11)`` converts i64→f32, the ``2^-53`` scaling and the
``log``/``sqrt``/``cos`` are f32 ops. The numpy backend and the
``etl.random`` Python API are UNCHANGED (they keep the float64 math).
Same key + operands ⇒ still BIT-IDENTICAL values across compiler runs
(the same-key determinism contract is the hard guarantee); the values
differ from the numpy backend by ~1e-6 relative (f32 mantissa rounding of
the 53-bit fraction + f32 transcendental ulps) instead of being
bit-identical/1-ulp. Rationale: GPUs like the A6000 run f64
transcendentals in software at ~1/64 the f32 SFU rate — the f64
Box–Muller kernel was the dominant random cost (measured ≈ +0.75–1.24 ms
device per (4096, 50) normal draw vs ≈ +0.05–0.15 ms for the f32 path).
f64 and f16 outputs keep the exact f64 math.

The writer calls ``emit_random_op(writer, op)``; all emission goes through
the Writer's SSA/type helpers (this module is writer-private, not part of
the public API). All expansions require STATIC shapes (the v1 compiler
backends reject symbolic dims); dynamic cases raise ``core.BackendError``.
"""

from __future__ import annotations

import numpy as np

from etl.core import BackendError

# Canonical algorithm-name constants — owned by `etl.ir.op_defs.random`
# (the frontend side of the multi-algorithm RNG work). Guarded import:
# until that side lands, the pinned fallback strings keep `import etl`
# green and match the frontend's eventual values exactly.
try:
    from etl.ir.op_defs import random as _random_opdefs

    ALGORITHM_SPLITMIX64 = _random_opdefs.ALGORITHM_SPLITMIX64
    ALGORITHM_THREEFRY2X32 = _random_opdefs.ALGORITHM_THREEFRY2X32
    ALGORITHM_PHILOX4X32_10 = _random_opdefs.ALGORITHM_PHILOX4X32_10
except (ImportError, AttributeError):  # workstream A not merged yet — pinned fallback
    ALGORITHM_SPLITMIX64 = "splitmix64"
    ALGORITHM_THREEFRY2X32 = "threefry2x32"
    ALGORITHM_PHILOX4X32_10 = "philox4x32_10"

# SplitMix64 constants (uint64 hex → emitted as two's-complement i64).
_GOLDEN = 0x9E3779B97F4A7C15
_B_MUL = 0xBF58476D1CE4E5B9
_C_MUL = 0x94D049BB133111EB
_SIGN_FLIP = 0x8000000000000000
_INV_2P53 = 1.0 / 2.0**53
_INV_2P32 = 1.0 / 2.0**32  # exact float64; full-word 32-bit uniform scaling
_TWO_PI = 2.0 * np.pi
# Per-op stream salts (must match kernels/random.py `_SALTS` EXACTLY).
_SALTS = {
    "uniform": 0x243F6A8885A308D3,
    "normal": 0x13198A2E03707344,
    "randint": 0xA4093822299F31D0,
    "permutation": 0x082EFA98EC4E6C89,
}

# threefry2x32-20 constants (pinned Random123/JAX cipher; u32 semantics,
# emitted bit-identically in i32).
_THREEFRY_KS_XOR = 0x1BD11BDA
_THREEFRY_ROTS_EVEN = (13, 15, 26, 6)
_THREEFRY_ROTS_ODD = (17, 29, 16, 24)

# philox4x32-10 constants (pinned Random123/JAX cipher; u32 semantics,
# emitted on a ui32 core with an i64 mulhilo chain).
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85

_I64 = np.dtype("int64")
_I32 = np.dtype("int32")
_UI32 = np.dtype("uint32")
_F32 = np.dtype("float32")
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


def _i32(w, value, shape):
    """Emit a scalar/static i32 constant broadcast to ``shape`` (numpy
    int → two's-complement i32) → ``(name, lines)`` — the i32 mirror of
    ``_i64`` (wraps mod 2^32 first so any 32-bit pattern, e.g. the
    permutation sign-flip ``0x80000000``, emits as a valid i32 literal)."""
    return w._scalar_constant_for(
        _I32, int(np.asarray(value % (1 << 32), dtype=np.uint32).view(np.int32)), shape
    )


def _ui32(w, value, shape):
    """Emit a scalar/static ui32 constant broadcast to ``shape`` (numpy
    int wrapped mod 2^32, kept UNSIGNED — the philox core's element type)
    → ``(name, lines)``."""
    return w._scalar_constant_for(
        _UI32, int(np.asarray(value % (1 << 32), dtype=np.uint32)), shape
    )


def _i32_vals(values) -> np.ndarray:
    """Python ints → two's-complement int32 dense-constant payload (for
    salt-fold vectors like ``[salt_lo, salt_hi]``)."""
    return np.asarray(
        [int(v) % (1 << 32) for v in values], dtype=np.uint32
    ).view(np.int32)


def _salt_words(salt: int) -> tuple:
    """The 64-bit per-op salt split into its low/high 32-bit words (the
    threefry/philox key-fold words; matches the pinned contract). The salt
    is masked to 64 bits FIRST — ``split_n`` computes ``i * GOLDEN``
    UNWRAPPED (≥ 2^64 for i ≥ 2) and the kernels consume salts mod 2^64,
    so bit 64+ must not leak into word 32."""
    salt &= (1 << 64) - 1
    return (salt & 0xFFFFFFFF, (salt >> 32) & 0xFFFFFFFF)


def _algorithm_of(w, op) -> str:
    """The op's random algorithm: the ``algorithm`` attribute, defaulting to
    ``splitmix64`` when absent (the pre-multi-algorithm IR). Unknown values
    → BackendError naming the op and the value."""
    alg = op.attributes.get("algorithm", ALGORITHM_SPLITMIX64)
    if alg not in (ALGORITHM_SPLITMIX64, ALGORITHM_THREEFRY2X32, ALGORITHM_PHILOX4X32_10):
        raise BackendError(
            f"stablehlo export: op '{op.name}'{w._loc(op)} declares the "
            f"unknown random algorithm {alg!r} (expected one of "
            f"{sorted((ALGORITHM_SPLITMIX64, ALGORITHM_THREEFRY2X32, ALGORITHM_PHILOX4X32_10))})"
        )
    return alg


def _validate_key(w, op, alg) -> None:
    """The op's key operand must match the algorithm's key layout (dtype +
    shape): splitmix64 → rank-0 int64; threefry2x32 → ``(2,)`` int32;
    philox4x32_10 → ``(4,)`` int32. Mismatch → BackendError naming the op
    and the expected layout."""
    key = op.operands[0]
    dtype = np.dtype(key.type.dtype)
    shape = tuple(key.type.shape)
    if alg == ALGORITHM_SPLITMIX64:
        expected_dtype, expected_shape = np.dtype("int64"), ()
    elif alg == ALGORITHM_THREEFRY2X32:
        expected_dtype, expected_shape = np.dtype("int32"), (2,)
    else:  # ALGORITHM_PHILOX4X32_10
        expected_dtype, expected_shape = np.dtype("int32"), (4,)
    if dtype != expected_dtype or shape != expected_shape:
        raise BackendError(
            f"stablehlo export: op '{op.name}'{w._loc(op)} uses the "
            f"{alg!r} algorithm, which requires a key of dtype "
            f"{np.dtype(expected_dtype).name} and shape {expected_shape!r} "
            f"(got {np.dtype(dtype).name} {shape!r})"
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

    Each step ``z ^ (z >> s)`` emits ``stablehlo.shift_right_logical`` —
    the uint64 logical right shift, bit-exact on the two's-complement i64
    bit pattern (the former arithmetic-shift + low-bits-mask form computed
    exactly the same bits; the direct op drops the mask constants).
    """
    for shift, mul in ((30, _B_MUL), (27, _C_MUL)):
        amt, extra = _i64(w, shift, z_shape)
        lines.extend(extra)
        s = w._new_name()
        lines.append(
            f"{s} = stablehlo.shift_right_logical {z_name}, {amt} : "
            f"{w._type_str(_I64, z_shape)}"
        )
        x = w._new_name()
        lines.append(f"{x} = stablehlo.xor {z_name}, {s} : {w._type_str(_I64, z_shape)}")
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
        f"{s} = stablehlo.shift_right_logical {z_name}, {amt} : "
        f"{w._type_str(_I64, z_shape)}"
    )
    x = w._new_name()
    lines.append(f"{x} = stablehlo.xor {z_name}, {s} : {w._type_str(_I64, z_shape)}")
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


def _emit_word_pairs(w, seed_name, count, lines) -> tuple:
    """The two SplitMix64 word chains behind one Box–Muller pair: chain A
    element i = mix3(seed + (2i + 1) * GOLDEN), chain B element i =
    mix3(seed + (2i + 2) * GOLDEN). These are exactly the even/odd words of
    ``_emit_words(seed, 2 * count)`` (each element mixes its own derived
    state — no sequential chain), so the VALUES are bit-identical to the
    kernel's ``w[0::2]`` / ``w[1::2]``, while the emission stays pure
    elementwise (no ``(2*count,)`` tensor + reshape + slice column
    extraction — the whole normal expansion fuses into ONE dispatch).
    Returns ``(chain_a, chain_b)`` names.
    """
    if count == 0:
        z = w._new_name()
        lines.append(f"{z} = stablehlo.constant dense<> : tensor<0xi64>")
        return z, z
    two_gold = (2 * _GOLDEN) % (1 << 64)  # mod 2^64 (matches uint64 wrap)
    # prod = iota * 2G (shared by both chains).
    iota = w._new_name()
    lines.append(f"{iota} = stablehlo.iota dim = 0 : tensor<{count}xi64>")
    gold2, extra = _i64(w, two_gold, (count,))
    lines.extend(extra)
    prod = w._new_name()
    lines.append(
        f"{prod} = stablehlo.multiply {iota}, {gold2} : tensor<{count}xi64>"
    )
    # base_a = seed + G, base_b = seed + 2G (scalar adds, mod-2^64 wrap).
    g, extra = _i64(w, _GOLDEN, ())
    lines.extend(extra)
    base_a = w._new_name()
    lines.append(f"{base_a} = stablehlo.add {seed_name}, {g} : tensor<i64>")
    g2, extra = _i64(w, two_gold, ())
    lines.extend(extra)
    base_b = w._new_name()
    lines.append(f"{base_b} = stablehlo.add {seed_name}, {g2} : tensor<i64>")
    out = []
    for base in (base_a, base_b):
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
        out.append(_emit_mix3(w, states, (count,), lines))
    return tuple(out)


def _emit_u_scaled(w, words_name, count, dtype, lines) -> str:
    """``u = (word >> 11) * 2^-53`` in ``dtype`` (f64: bit-exact vs the
    kernel; f32: the documented f32 fast-path rounding — the 53-bit
    fraction converts to f32's 24-bit mantissa)."""
    amt, extra = _i64(w, 11, (count,))
    lines.extend(extra)
    s = w._new_name()
    lines.append(
        f"{s} = stablehlo.shift_right_logical {words_name}, {amt} : "
        f"tensor<{count}xi64>"
    )
    cv = w._new_name()
    lines.append(
        f"{cv} = stablehlo.convert {s} : (tensor<{count}xi64>) -> "
        f"{w._type_str(dtype, (count,))}"
    )
    if dtype == _F32:
        inv, extra = _scalar_f32(w, _INV_2P53, (count,))
    else:
        inv, extra = _scalar_f64(w, _INV_2P53, (count,))
    lines.extend(extra)
    u = w._new_name()
    lines.append(
        f"{u} = stablehlo.multiply {cv}, {inv} : {w._type_str(dtype, (count,))}"
    )
    return u


# ---------------------------------------------------------------------------
# threefry2x32 / philox4x32_10 — algorithm-aware lowering
#
# Word-stream contract (PINNED, bit-exactness reference — see the module
# docstring): the key operand is the algorithm's key layout; per-op salt
# folds into the key words (threefry: k0^salt_lo, k1^salt_hi; philox:
# k0..k3 with the [lo, hi, lo, hi] pattern — the inline cipher's round-r
# key pair is (k'_{2r mod 4} + r*W0, k'_{(2r+1) mod 4} + r*W1), matching
# the numpy kernel); cipher block p (0-based) uses counter (p, 0) /
# (p, 0, 0, 0) (low word = call index, high words 0); element i uses flat
# word i (uniform/randint/permutation) or flat words 2i/2i+1 (normal's
# u1/u2). u-scaling by word width: 32-bit words (threefry/philox) use the
# FULL word u = w * 2^-32 (mirrors ``_word_uniforms`` — no shift); the
# splitmix64 (w >> 11) * 2^-53 form applies to 64-bit words only.
# ---------------------------------------------------------------------------


def _scalar_from_vec(w, vec_name, index, vec_len, dtype, lines) -> str:
    """Extract scalar element ``index`` of a 1-D tensor via slice+reshape
    (the codebase's generic slice form)."""
    s = w._new_name()
    lines.append(
        f'{s} = "stablehlo.slice"({vec_name}) '
        f"{{start_indices = {w._i64_array([index])}, "
        f"limit_indices = {w._i64_array([index + 1])}, "
        f"strides = {w._i64_array([1])}}} : "
        f"({w._type_str(dtype, (vec_len,))}) -> {w._type_str(dtype, (1,))}"
    )
    r = w._new_name()
    lines.append(
        f"{r} = stablehlo.reshape {s} : ({w._type_str(dtype, (1,))}) -> "
        f"{w._type_str(dtype, ())}"
    )
    return r


def _bcast_scalar(w, name, dtype, shape, lines) -> str:
    """Broadcast a scalar SSA value to the static ``shape`` (no-op for
    scalars; the writer's elementwise shape-equalization pattern)."""
    if tuple(shape) == ():
        return name
    b = w._new_name()
    lines.append(
        f'{b} = "stablehlo.broadcast_in_dim"({name}) '
        f"{{broadcast_dimensions = {w._i64_array(())}}} : "
        f"({w._elem_type(dtype)}) -> {w._type_str(dtype, shape)}"
    )
    return b


def _slice_flat(w, src_name, start, limit, src_len, out_len, dtype, lines) -> str:
    """``stablehlo.slice`` of a flat 1-D tensor in the codebase's generic
    form (unit strides; the element type is ``dtype`` — the threefry stream
    is i32, the philox flat is ui32 before its final convert)."""
    s = w._new_name()
    lines.append(
        f'{s} = "stablehlo.slice"({src_name}) '
        f"{{start_indices = {w._i64_array([start])}, "
        f"limit_indices = {w._i64_array([limit])}, "
        f"strides = {w._i64_array([1])}}} : "
        f"({w._type_str(dtype, (src_len,))}) -> {w._type_str(dtype, (out_len,))}"
    )
    return s


def _strided_slice(w, src_name, start, limit, stride, src_len, out_len, lines) -> str:
    """Strided ``stablehlo.slice`` of a flat 1-D tensor (the normal
    u1/u2 word extraction)."""
    s = w._new_name()
    lines.append(
        f'{s} = "stablehlo.slice"({src_name}) '
        f"{{start_indices = {w._i64_array([start])}, "
        f"limit_indices = {w._i64_array([limit])}, "
        f"strides = {w._i64_array([stride])}}} : "
        f"(tensor<{src_len}xi32>) -> tensor<{out_len}xi32>"
    )
    return s


def _emit_threefry_words_inline(w, key, salt, n_words, lines) -> str:
    """Flat ``(n_words,)`` i32 word stream for threefry2x32-20 as an
    INLINE i32 elementwise expansion: the ``(2,)`` i32 key (salt-folded)
    with per-block counters ``(p, 0)`` runs the 5×4-round cipher with the
    ``ks = [k0, k1, k0^k1^0x1BD11BDA]`` schedule — bit-exact vs the numpy
    reference (i32 add/xor and the logical-shift rotl are bit-identical to
    the u32 semantics; the ``stablehlo.shift_right_logical`` in the rotl
    keeps the rotation exact)."""
    M = -(-n_words // 2)
    if n_words == 0:
        name = w._new_name()
        lines.append(f"{name} = stablehlo.constant dense<> : tensor<0xi32>")
        return name
    key_name = w._name(key)
    slo, shi = _salt_words(salt)
    k0 = _scalar_from_vec(w, key_name, 0, 2, _I32, lines)
    k1 = _scalar_from_vec(w, key_name, 1, 2, _I32, lines)
    sl, extra = _i32(w, slo, ())
    lines.extend(extra)
    k0f = w._new_name()
    lines.append(f"{k0f} = stablehlo.xor {k0}, {sl} : tensor<i32>")
    sh, extra = _i32(w, shi, ())
    lines.extend(extra)
    k1f = w._new_name()
    lines.append(f"{k1f} = stablehlo.xor {k1}, {sh} : tensor<i32>")
    ksxor, extra = _i32(w, _THREEFRY_KS_XOR, ())
    lines.extend(extra)
    kx = w._new_name()
    lines.append(f"{kx} = stablehlo.xor {k0f}, {k1f} : tensor<i32>")
    ks2 = w._new_name()
    lines.append(f"{ks2} = stablehlo.xor {kx}, {ksxor} : tensor<i32>")
    # x0 = iota(M) (counter low word = block index), x1 = zeros.
    iota = w._new_name()
    lines.append(f"{iota} = stablehlo.iota dim = 0 : tensor<{M}xi32>")
    z, extra = _i32(w, 0, (M,))
    lines.extend(extra)
    x0, x1 = iota, z
    # Pre-injection: x0 += ks[0]; x1 += ks[1].
    ks0b = _bcast_scalar(w, k0f, _I32, (M,), lines)
    a = w._new_name()
    lines.append(f"{a} = stablehlo.add {x0}, {ks0b} : tensor<{M}xi32>")
    x0 = a
    ks1b = _bcast_scalar(w, k1f, _I32, (M,), lines)
    a = w._new_name()
    lines.append(f"{a} = stablehlo.add {x1}, {ks1b} : tensor<{M}xi32>")
    x1 = a
    ks = (k0f, k1f, ks2)
    for g in range(5):
        rots = _THREEFRY_ROTS_EVEN if g % 2 == 0 else _THREEFRY_ROTS_ODD
        for r in rots:
            # x0 = x0 + x1; x1 = rotl(x1, r); x1 ^= x0  (rotl via shl |
            # shift_right_logical — exact on the u32 bit pattern).
            a = w._new_name()
            lines.append(f"{a} = stablehlo.add {x0}, {x1} : tensor<{M}xi32>")
            amt, extra = _i32(w, r, (M,))
            lines.extend(extra)
            l = w._new_name()
            lines.append(f"{l} = stablehlo.shift_left {x1}, {amt} : tensor<{M}xi32>")
            amtr, extra = _i32(w, 32 - r, (M,))
            lines.extend(extra)
            rr = w._new_name()
            lines.append(
                f"{rr} = stablehlo.shift_right_logical {x1}, {amtr} : "
                f"tensor<{M}xi32>"
            )
            o = w._new_name()
            lines.append(f"{o} = stablehlo.or {l}, {rr} : tensor<{M}xi32>")
            nx1 = w._new_name()
            lines.append(f"{nx1} = stablehlo.xor {o}, {a} : tensor<{M}xi32>")
            x0, x1 = a, nx1
        # Injection after group g: x0 += ks[(g+1)%3];
        # x1 += ks[(g+2)%3] + (g + 1).
        kb = _bcast_scalar(w, ks[(g + 1) % 3], _I32, (M,), lines)
        a = w._new_name()
        lines.append(f"{a} = stablehlo.add {x0}, {kb} : tensor<{M}xi32>")
        x0 = a
        kb = _bcast_scalar(w, ks[(g + 2) % 3], _I32, (M,), lines)
        gc, extra = _i32(w, g + 1, (M,))
        lines.extend(extra)
        ksg = w._new_name()
        lines.append(f"{ksg} = stablehlo.add {kb}, {gc} : tensor<{M}xi32>")
        a = w._new_name()
        lines.append(f"{a} = stablehlo.add {x1}, {ksg} : tensor<{M}xi32>")
        x1 = a
    # Interleaved flat stream: (M,1)+(M,1) → (M,2) → (2M,) → slice
    # [0:n_words] (C-order: block p's (x0_p, x1_p) land at words 2p, 2p+1).
    x0r = w._new_name()
    lines.append(
        f"{x0r} = stablehlo.reshape {x0} : (tensor<{M}xi32>) -> "
        f"tensor<{M}x1xi32>"
    )
    x1r = w._new_name()
    lines.append(
        f"{x1r} = stablehlo.reshape {x1} : (tensor<{M}xi32>) -> "
        f"tensor<{M}x1xi32>"
    )
    cat = w._new_name()
    lines.append(
        f'{cat} = "stablehlo.concatenate"({x0r}, {x1r}) '
        f"{{dimension = 1 : i64}} : (tensor<{M}x1xi32>, tensor<{M}x1xi32>) "
        f"-> tensor<{M}x2xi32>"
    )
    flat = w._new_name()
    lines.append(
        f"{flat} = stablehlo.reshape {cat} : (tensor<{M}x2xi32>) -> "
        f"tensor<{2 * M}xi32>"
    )
    return _slice_flat(w, flat, 0, n_words, 2 * M, n_words, _I32, lines)


def _emit_philox_words_inline(w, key, salt, n_words, lines) -> str:
    """Flat ``(n_words,)`` i32 word stream for philox4x32-10 as an INLINE
    ui32 elementwise expansion: the ``(4,)`` i32 key (salt-folded per the
    pinned key') with per-block counters ``(p, 0, 0, 0)`` runs the 10
    rounds of the pinned cipher — M0/M1 multiplies with the i64 mulhilo
    chain (ui32→i64 zero-extend multiply + logical shift 32 + truncate =
    the exact high half) — and the ROUND-KEY SCHEDULE mirrors
    ``kernels/random.py`` ``_philox4x32`` EXACTLY: round r (0-based) uses
    the cyclically permuted pair ``(k'_{2r mod 4} + r*W0, k'_{(2r+1) mod 4}
    + r*W1)`` mod 2^32 — (k0', k1') on even rounds, (k2', k3') on odd
    rounds, each with the per-round Weyl bump. The final ui32 stream
    converts to i32 (bit-preserving)."""
    M = -(-n_words // 4)
    if n_words == 0:
        name = w._new_name()
        lines.append(f"{name} = stablehlo.constant dense<> : tensor<0xi32>")
        return name
    key_name = w._name(key)
    ku = w._new_name()
    lines.append(
        f"{ku} = stablehlo.convert {key_name} : (tensor<4xi32>) -> "
        f"tensor<4xui32>"
    )
    slo, shi = _salt_words(salt)
    k0 = _scalar_from_vec(w, ku, 0, 4, _UI32, lines)
    k1 = _scalar_from_vec(w, ku, 1, 4, _UI32, lines)
    k2 = _scalar_from_vec(w, ku, 2, 4, _UI32, lines)
    k3 = _scalar_from_vec(w, ku, 3, 4, _UI32, lines)
    sl, extra = _ui32(w, slo, ())
    lines.extend(extra)
    k0f = w._new_name()
    lines.append(f"{k0f} = stablehlo.xor {k0}, {sl} : tensor<ui32>")
    k2f = w._new_name()
    lines.append(f"{k2f} = stablehlo.xor {k2}, {sl} : tensor<ui32>")
    sh, extra = _ui32(w, shi, ())
    lines.extend(extra)
    k1f = w._new_name()
    lines.append(f"{k1f} = stablehlo.xor {k1}, {sh} : tensor<ui32>")
    k3f = w._new_name()
    lines.append(f"{k3f} = stablehlo.xor {k3}, {sh} : tensor<ui32>")
    # x0 = iota(M) (counter low word = block index; the i32 iota converts
    # bit-preservingly to ui32), x1 = x2 = x3 = zeros.
    iota = w._new_name()
    lines.append(f"{iota} = stablehlo.iota dim = 0 : tensor<{M}xi32>")
    x0 = w._new_name()
    lines.append(f"{x0} = stablehlo.convert {iota} : (tensor<{M}xi32>) -> "
                 f"tensor<{M}xui32>")
    z, extra = _ui32(w, 0, (M,))
    lines.extend(extra)
    x1, x2, x3 = z, z, z
    # Round constants (once, outside the loop).
    m0_32, extra = _ui32(w, _PHILOX_M0, (M,))
    lines.extend(extra)
    m1_32, extra = _ui32(w, _PHILOX_M1, (M,))
    lines.extend(extra)
    m0_64, extra = _i64(w, _PHILOX_M0, (M,))
    lines.extend(extra)
    m1_64, extra = _i64(w, _PHILOX_M1, (M,))
    lines.extend(extra)
    sh32, extra = _i64(w, 32, (M,))
    lines.extend(extra)
    w0s, extra = _ui32(w, _PHILOX_W0, ())
    lines.extend(extra)
    w1s, extra = _ui32(w, _PHILOX_W1, ())
    lines.extend(extra)
    # Round-key schedule (mirrors ``kernels/random.py`` ``_philox4x32``
    # EXACTLY): round r (0-based) uses the cyclically permuted pair
    # (k'_{2r mod 4} + r*W0, k'_{(2r+1) mod 4} + r*W1) mod 2^32 — the
    # (k0', k1') pair on even rounds, (k2', k3') on odd rounds. All four
    # accumulators bump by W0/W1 once per round, so after r bumps word i
    # holds k_i' + r*W — the kernel's r*W term for round r.
    k0, k1, k2, k3 = k0f, k1f, k2f, k3f
    for rnd in range(10):
        if rnd > 0:
            nk0 = w._new_name()
            lines.append(f"{nk0} = stablehlo.add {k0}, {w0s} : tensor<ui32>")
            k0 = nk0
            nk1 = w._new_name()
            lines.append(f"{nk1} = stablehlo.add {k1}, {w1s} : tensor<ui32>")
            k1 = nk1
            nk2 = w._new_name()
            lines.append(f"{nk2} = stablehlo.add {k2}, {w0s} : tensor<ui32>")
            k2 = nk2
            nk3 = w._new_name()
            lines.append(f"{nk3} = stablehlo.add {k3}, {w1s} : tensor<ui32>")
            k3 = nk3
        rk0 = k0 if rnd % 2 == 0 else k2
        rk1 = k1 if rnd % 2 == 0 else k3
        # lo0 = M0 * x0 mod 2^32; hi0 = mulhi32(M0, x0).
        lo0 = w._new_name()
        lines.append(f"{lo0} = stablehlo.multiply {x0}, {m0_32} : tensor<{M}xui32>")
        x0_64 = w._new_name()
        lines.append(
            f"{x0_64} = stablehlo.convert {x0} : (tensor<{M}xui32>) -> "
            f"tensor<{M}xi64>"
        )
        p0 = w._new_name()
        lines.append(f"{p0} = stablehlo.multiply {x0_64}, {m0_64} : tensor<{M}xi64>")
        h0 = w._new_name()
        lines.append(
            f"{h0} = stablehlo.shift_right_logical {p0}, {sh32} : tensor<{M}xi64>"
        )
        hi0 = w._new_name()
        lines.append(
            f"{hi0} = stablehlo.convert {h0} : (tensor<{M}xi64>) -> tensor<{M}xui32>"
        )
        # lo1 = M1 * x2 mod 2^32; hi1 = mulhi32(M1, x2).
        lo1 = w._new_name()
        lines.append(f"{lo1} = stablehlo.multiply {x2}, {m1_32} : tensor<{M}xui32>")
        x2_64 = w._new_name()
        lines.append(
            f"{x2_64} = stablehlo.convert {x2} : (tensor<{M}xui32>) -> "
            f"tensor<{M}xi64>"
        )
        p1 = w._new_name()
        lines.append(f"{p1} = stablehlo.multiply {x2_64}, {m1_64} : tensor<{M}xi64>")
        h1 = w._new_name()
        lines.append(
            f"{h1} = stablehlo.shift_right_logical {p1}, {sh32} : tensor<{M}xi64>"
        )
        hi1 = w._new_name()
        lines.append(
            f"{hi1} = stablehlo.convert {h1} : (tensor<{M}xi64>) -> tensor<{M}xui32>"
        )
        # x0' = hi1 ^ x1 ^ rk0; x1' = lo1; x2' = hi0 ^ x3 ^ rk1; x3' = lo0.
        kb0 = _bcast_scalar(w, rk0, _UI32, (M,), lines)
        t = w._new_name()
        lines.append(f"{t} = stablehlo.xor {hi1}, {x1} : tensor<{M}xui32>")
        nx0 = w._new_name()
        lines.append(f"{nx0} = stablehlo.xor {t}, {kb0} : tensor<{M}xui32>")
        kb1 = _bcast_scalar(w, rk1, _UI32, (M,), lines)
        t = w._new_name()
        lines.append(f"{t} = stablehlo.xor {hi0}, {x3} : tensor<{M}xui32>")
        nx2 = w._new_name()
        lines.append(f"{nx2} = stablehlo.xor {t}, {kb1} : tensor<{M}xui32>")
        x0, x1, x2, x3 = nx0, lo1, nx2, lo0
    # Interleaved flat stream: (M,1)x4 → (M,4) → (4M,) → slice [0:n_words]
    # (C-order: block p's (x0..x3) land at words 4p..4p+3) → i32.
    parts = []
    for v in (x0, x1, x2, x3):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {v} : (tensor<{M}xui32>) -> "
            f"tensor<{M}x1xui32>"
        )
        parts.append(r)
    cat = w._new_name()
    op_types = ", ".join(f"tensor<{M}x1xui32>" for _ in parts)
    lines.append(
        f'{cat} = "stablehlo.concatenate"({", ".join(parts)}) '
        f"{{dimension = 1 : i64}} : ({op_types}) -> tensor<{M}x4xui32>"
    )
    flat = w._new_name()
    lines.append(
        f"{flat} = stablehlo.reshape {cat} : (tensor<{M}x4xui32>) -> "
        f"tensor<{4 * M}xui32>"
    )
    cut = _slice_flat(w, flat, 0, n_words, 4 * M, n_words, _UI32, lines)
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {cut} : (tensor<{n_words}xui32>) -> "
        f"tensor<{n_words}xi32>"
    )
    return out


def _emit_rng_bit_generator(w, op, alg, key, salt, n_words, lines) -> str:
    """Flat ``(n_words,)`` i32 word stream via the NATIVE
    ``stablehlo.rng_bit_generator`` (used when the writer's
    ``rng_bit_generator`` option is set): one call emits ``n_words`` words
    from the salt-folded state — key words FIRST, then the counter words
    (zero, so block p advances the counter internally per the spec).
    State layouts: threefry ``[k0', k1', ctr0, ctr1]`` (tensor<4xui32>),
    philox ``[k0', k1', ctr0..ctr3]`` (tensor<6xui32> — the native philox
    state carries a 2-WORD key, so it takes words 0–1 of the salt-folded
    4-word key; the k2'/k3' fold words are dropped). ``%state_out`` is
    unused. The ui32 output converts to i32 (bit-preserving).

    V1 CAVEAT (native PHILOX — documented, export-only): native
    ``stablehlo.rng_bit_generator`` with ``algorithm = PHILOX`` is NOT
    validated against the numpy reference anywhere — iree 3.11 fails to
    legalize it (``rng_bit_generator=False`` on iree), tvm has no
    rng_bit_generator support, and the xla adapter's ``True`` flag is
    gate-skipped without a user-provided PJRT plugin. The StableHLO philox
    cipher consumes only the 2-word key (Random123 semantics), while the
    etl reference uses the 4-word CYCLIC round-key schedule — so native
    philox words may differ from the numpy reference for nonzero keys.
    Bit-exactness MUST be re-validated against a real XLA plugin before
    any adapter enables native philox; if XLA's philox semantics differ,
    philox stays on the bit-exact INLINE path for xla."""
    key_name = w._name(key)
    if alg == ALGORITHM_THREEFRY2X32:
        key_words, counter_words, alg_enum = 2, 2, "THREE_FRY"
    else:  # ALGORITHM_PHILOX4X32_10
        key_words, counter_words, alg_enum = 4, 4, "PHILOX"
    slo, shi = _salt_words(salt)
    pat = [slo, shi] if key_words == 2 else [slo, shi, slo, shi]
    saltc = w._new_name()
    lines.append(
        f"{saltc} = stablehlo.constant {w._constant_text(_i32_vals(pat))} : "
        f"{w._type_str(_I32, (key_words,))}"
    )
    kf = w._new_name()
    lines.append(f"{kf} = stablehlo.xor {key_name}, {saltc} : "
                 f"tensor<{key_words}xi32>")
    if key_words == 4:
        # The 6-word philox state carries a 2-word key — words 0–1 of the
        # folded key (the pinned state layout [k0', k1', 0, 0, 0, 0]).
        state_key = _slice_flat(w, kf, 0, 2, 4, 2, _I32, lines)
    else:
        state_key = kf
    ku = w._new_name()
    lines.append(
        f"{ku} = stablehlo.convert {state_key} : (tensor<2xi32>) -> "
        f"tensor<2xui32>"
    )
    z, extra = _ui32(w, 0, (counter_words,))
    lines.extend(extra)
    state = w._new_name()
    lines.append(
        f'{state} = "stablehlo.concatenate"({ku}, {z}) '
        f"{{dimension = 0 : i64}} : (tensor<2xui32>, "
        f"tensor<{counter_words}xui32>) -> tensor<{2 + counter_words}xui32>"
    )
    so, out = w._new_name(), w._new_name()
    lines.append(
        f"{so}, {out} = stablehlo.rng_bit_generator {state}, "
        f"algorithm = {alg_enum} : (tensor<{2 + counter_words}xui32>) -> "
        f"(tensor<{2 + counter_words}xui32>, tensor<{n_words}xui32>)"
    )
    oi = w._new_name()
    lines.append(
        f"{oi} = stablehlo.convert {out} : (tensor<{n_words}xui32>) -> "
        f"tensor<{n_words}xi32>"
    )
    return oi


def _emit_words_for(w, op, alg, salt, n_words, lines) -> str:
    """The flat ``(n_words,)`` i32 word stream for the algorithm: the
    native ``rng_bit_generator`` when the writer's flag is set, else the
    inline bit-exact expansion (the default — no target support needed)."""
    if n_words == 0:
        name = w._new_name()
        lines.append(f"{name} = stablehlo.constant dense<> : tensor<0xi32>")
        return name
    if w._rng_bit_generator:
        return _emit_rng_bit_generator(w, op, alg, op.operands[0], salt, n_words, lines)
    if alg == ALGORITHM_THREEFRY2X32:
        return _emit_threefry_words_inline(w, op.operands[0], salt, n_words, lines)
    # ALGORITHM_PHILOX4X32_10
    return _emit_philox_words_inline(w, op.operands[0], salt, n_words, lines)


def _emit_u_scaled32(w, words_name, count, dtype, lines) -> str:
    """``u = w * 2^-32`` over i32 words — the u-scaling for the
    threefry/philox streams (mirrors ``kernels/random.py``
    ``_word_uniforms`` EXACTLY: the FULL 32-bit unsigned word scaled to
    [0, 1) via /2^32 — NO shift; the ``(w >> 11) * 2^-53`` form is the
    splitmix64 64-bit-word scaling only). The i32 words zero-extend to
    i64 (sign-extending convert + 0xFFFFFFFF mask — the i32 SSA values
    are the two's-complement word bit patterns), so the i64→dtype convert
    sees the UNSIGNED value (f64: bit-exact vs the kernel; f32: the
    documented f32 fast-path rounding, mirroring ``_emit_u_scaled``).
    (An i32→ui32 intermediate would be equivalent but would put ui32 into
    the threefry INLINE expansion, which is pinned as a pure-i32
    emission.)"""
    e = w._new_name()
    lines.append(
        f"{e} = stablehlo.convert {words_name} : (tensor<{count}xi32>) -> "
        f"tensor<{count}xi64>"
    )
    mask, extra = _i64(w, 0xFFFFFFFF, (count,))
    lines.extend(extra)
    a = w._new_name()
    lines.append(f"{a} = stablehlo.and {e}, {mask} : tensor<{count}xi64>")
    cv = w._new_name()
    lines.append(
        f"{cv} = stablehlo.convert {a} : (tensor<{count}xi64>) -> "
        f"{w._type_str(dtype, (count,))}"
    )
    if dtype == _F32:
        inv, extra = _scalar_f32(w, _INV_2P32, (count,))
    else:
        inv, extra = _scalar_f64(w, _INV_2P32, (count,))
    lines.extend(extra)
    u = w._new_name()
    lines.append(
        f"{u} = stablehlo.multiply {cv}, {inv} : {w._type_str(dtype, (count,))}"
    )
    return u


def _emit_bm_tail(w, u1, u2, count, comp, fast, lines) -> str:
    """The Box–Muller tail shared by the threefry/philox normal paths:
    ``u1 = max(u1, 2^-32)`` (the 32-bit word-width grid-step guard —
    ``kernels/random.py`` ``_normal_kernel`` uses ``min_u1 = _INV_2P32``
    for 32-bit words; the splitmix64 64-bit path guards with 2^-53 in its
    own body), ``z = sqrt(-2 log u1) * cos(2π u2)`` in ``comp`` (f32 fast
    path / f64); returns the ``(count,)`` name."""
    if fast:
        eps, extra = _scalar_f32(w, _INV_2P32, (count,))
    else:
        eps, extra = _scalar_f64(w, _INV_2P32, (count,))
    lines.extend(extra)
    u1m = w._new_name()
    lines.append(
        f"{u1m} = stablehlo.maximum {u1}, {eps} : {w._type_str(comp, (count,))}"
    )
    if fast:
        two, extra = _scalar_f32(w, -2.0, (count,))
    else:
        two, extra = _scalar_f64(w, -2.0, (count,))
    lines.extend(extra)
    lg = w._new_name()
    lines.append(f"{lg} = stablehlo.log {u1m} : {w._type_str(comp, (count,))}")
    prod = w._new_name()
    lines.append(
        f"{prod} = stablehlo.multiply {two}, {lg} : {w._type_str(comp, (count,))}"
    )
    rt = w._new_name()
    lines.append(f"{rt} = stablehlo.sqrt {prod} : {w._type_str(comp, (count,))}")
    if fast:
        twopi, extra = _scalar_f32(w, _TWO_PI, (count,))
    else:
        twopi, extra = _scalar_f64(w, _TWO_PI, (count,))
    lines.extend(extra)
    ang = w._new_name()
    lines.append(
        f"{ang} = stablehlo.multiply {twopi}, {u2} : {w._type_str(comp, (count,))}"
    )
    cs = w._new_name()
    lines.append(f"{cs} = stablehlo.cosine {ang} : {w._type_str(comp, (count,))}")
    z = w._new_name()
    lines.append(f"{z} = stablehlo.multiply {rt}, {cs} : {w._type_str(comp, (count,))}")
    return z


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
        cur = _emit_u_scaled(w, words, count, _F64, lines)
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {cur} : (tensor<{count}xf64>) -> "
            f"{w._type_str(_F64, shape)}"
        )
        cur = r
    return cur


def _scalar_f32(w, value, shape):
    return w._scalar_constant_for(_F32, value, shape)


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


def _emit_random_key_mix_splitmix64(w, op) -> str:
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


def _emit_random_uniform_splitmix64(w, op) -> str:
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


def _emit_random_normal_splitmix64(w, op) -> str:
    """``random_normal`` → Box–Muller over count pairs (matches the kernel:
    u1 = max(word>>11 * 2^-53, 2^-53), u2 = ..., z =
    sqrt(-2 log u1) cos(2π u2); vals = (mean + z * std).astype(dtype)).

    float32 outputs (the default dtype — the hot path, e.g. the new-evox DE
    crossover masks) take the f32 fast path (DOCUMENTED deviation, see the
    module docstring): the u1/u2 extraction and the log/sqrt/cos tail run
    in f32, so the compiler backends avoid the f64 transcendental software
    routines (the dominant random cost on GPU and CPU). f64/f16 outputs
    keep the exact f64 math. Both paths draw their two words from two
    parallel ``(count,)`` elementwise chains (``_emit_word_pairs``) so the
    whole expansion fuses into one dispatch (no reshape/slice column
    extraction)."""
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    count = _count_of(shape)
    fast = out_dtype == _F32
    comp = _F32 if fast else _F64
    lines = []
    seed = _emit_seed(w, op, op.operands[0], _SALTS["normal"], lines)
    if count == 0:
        z = w._new_name()
        lines.append(f"{z} = stablehlo.constant dense<> : {w._type_str(comp, (0,))}")
        cur = z
    else:
        words_a, words_b = _emit_word_pairs(w, seed, count, lines)
        # u1 = maximum(u1, 2^-53) (kernel's log(0) guard); u2 unscaled.
        u1 = _emit_u_scaled(w, words_a, count, comp, lines)
        u2 = _emit_u_scaled(w, words_b, count, comp, lines)
        if fast:
            eps, extra = _scalar_f32(w, _INV_2P53, (count,))
        else:
            eps, extra = _scalar_f64(w, _INV_2P53, (count,))
        lines.extend(extra)
        u1m = w._new_name()
        lines.append(
            f"{u1m} = stablehlo.maximum {u1}, {eps} : {w._type_str(comp, (count,))}"
        )
        # z = sqrt(-2 log u1) * cos(2π u2)
        if fast:
            two, extra = _scalar_f32(w, -2.0, (count,))
        else:
            two, extra = _scalar_f64(w, -2.0, (count,))
        lines.extend(extra)
        lg = w._new_name()
        lines.append(f"{lg} = stablehlo.log {u1m} : {w._type_str(comp, (count,))}")
        prod = w._new_name()
        lines.append(
            f"{prod} = stablehlo.multiply {two}, {lg} : {w._type_str(comp, (count,))}"
        )
        rt = w._new_name()
        lines.append(f"{rt} = stablehlo.sqrt {prod} : {w._type_str(comp, (count,))}")
        if fast:
            twopi, extra = _scalar_f32(w, _TWO_PI, (count,))
        else:
            twopi, extra = _scalar_f64(w, _TWO_PI, (count,))
        lines.extend(extra)
        ang = w._new_name()
        lines.append(
            f"{ang} = stablehlo.multiply {twopi}, {u2} : {w._type_str(comp, (count,))}"
        )
        cs = w._new_name()
        lines.append(f"{cs} = stablehlo.cosine {ang} : {w._type_str(comp, (count,))}")
        z = w._new_name()
        lines.append(f"{z} = stablehlo.multiply {rt}, {cs} : {w._type_str(comp, (count,))}")
        cur = z
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {cur} : ({w._type_str(comp, (count,))}) -> "
            f"{w._type_str(comp, shape)}"
        )
        cur = r
    mean = _emit_to_dtype_broadcast(w, op.operands[1], comp, shape, lines, op)
    std = _emit_to_dtype_broadcast(w, op.operands[2], comp, shape, lines, op)
    zp = w._new_name()
    lines.append(f"{zp} = stablehlo.multiply {cur}, {std} : {w._type_str(comp, shape)}")
    s = w._new_name()
    lines.append(f"{s} = stablehlo.add {mean}, {zp} : {w._type_str(comp, shape)}")
    if fast:
        out = s
    else:
        out = w._new_name()
        lines.append(
            f"{out} = stablehlo.convert {s} : ({w._type_str(comp, shape)}) -> "
            f"{w._type_str(out_dtype, shape)}"
        )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_randint_splitmix64(w, op) -> str:
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


def _emit_random_permutation_splitmix64(w, op) -> str:
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
    _sv, si = w._emit_stable_argsort(wf, _I64, (n,), 0, lines)
    out = w._new_name()
    lines.append(
        f"{out} = stablehlo.convert {si} : (tensor<{n}xi64>) -> "
        f"{w._type_str(out_dtype, (n,))}"
    )
    w._names[id(op.result)] = out
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Algorithm dispatch wrappers — each random op checks its ``algorithm``
# attribute (absent ⇒ splitmix64) and routes: splitmix64 → the untouched
# legacy bodies above; threefry2x32/philox4x32_10 → the word-stream
# emitters (inline by default, native ``rng_bit_generator`` when the
# writer's flag is set) + the shared post-processing tails.
# ---------------------------------------------------------------------------


def _emit_random_key_mix(w, op) -> str:
    """``random_key_mix`` → the algorithm's key derivation: splitmix64 →
    mix3(key ^ salt); threefry2x32 → one cipher call at counter (0, 0)
    reshaped ``(2,)``; philox4x32_10 → one call reshaped ``(4,)``."""
    alg = _algorithm_of(w, op)
    if alg == ALGORITHM_SPLITMIX64:
        return _emit_random_key_mix_splitmix64(w, op)
    _validate_key(w, op, alg)
    lines = []
    salt = op.attributes["salt"]
    n_words = 2 if alg == ALGORITHM_THREEFRY2X32 else 4
    words = _emit_words_for(w, op, alg, salt, n_words, lines)
    w._names[id(op.result)] = words
    return "\n".join(lines)


def _emit_random_uniform(w, op) -> str:
    """``random_uniform`` → the algorithm's word stream, then the shared
    tail: u = (w >> 11) * 2^-53 (splitmix64 64-bit words) or u = w * 2^-32
    (threefry/philox 32-bit words — ``_emit_u_scaled32``, mirroring the
    kernel's ``_word_uniforms``) in f64; vals =
    (low + u * (high - low)).astype(dtype) — all f64, then converted."""
    alg = _algorithm_of(w, op)
    if alg == ALGORITHM_SPLITMIX64:
        return _emit_random_uniform_splitmix64(w, op)
    _validate_key(w, op, alg)
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    count = _count_of(shape)
    lines = []
    if count == 0:
        u = w._new_name()
        lines.append(f"{u} = stablehlo.constant dense<> : tensor<0xf64>")
    else:
        words = _emit_words_for(w, op, alg, _SALTS["uniform"], count, lines)
        u = _emit_u_scaled32(w, words, count, _F64, lines)
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {u} : (tensor<{count}xf64>) -> "
            f"{w._type_str(_F64, shape)}"
        )
        u = r
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
    """``random_normal`` → the algorithm's 2*count word stream with u1 =
    words[0::2], u2 = words[1::2] (strided slices), then the shared
    Box–Muller tail (f32 fast path / f64 math; the u1 guard is the
    word-width grid step — 2^-32 for the 32-bit streams) + mean/std
    affine — the kernel's post-processing."""
    alg = _algorithm_of(w, op)
    if alg == ALGORITHM_SPLITMIX64:
        return _emit_random_normal_splitmix64(w, op)
    _validate_key(w, op, alg)
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    count = _count_of(shape)
    fast = out_dtype == _F32
    comp = _F32 if fast else _F64
    lines = []
    if count == 0:
        z = w._new_name()
        lines.append(f"{z} = stablehlo.constant dense<> : {w._type_str(comp, (0,))}")
        cur = z
    else:
        words = _emit_words_for(w, op, alg, _SALTS["normal"], 2 * count, lines)
        u1w = _strided_slice(w, words, 0, 2 * count, 2, 2 * count, count, lines)
        u2w = _strided_slice(w, words, 1, 2 * count, 2, 2 * count, count, lines)
        u1 = _emit_u_scaled32(w, u1w, count, comp, lines)
        u2 = _emit_u_scaled32(w, u2w, count, comp, lines)
        cur = _emit_bm_tail(w, u1, u2, count, comp, fast, lines)
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {cur} : ({w._type_str(comp, (count,))}) -> "
            f"{w._type_str(comp, shape)}"
        )
        cur = r
    mean = _emit_to_dtype_broadcast(w, op.operands[1], comp, shape, lines, op)
    std = _emit_to_dtype_broadcast(w, op.operands[2], comp, shape, lines, op)
    zp = w._new_name()
    lines.append(f"{zp} = stablehlo.multiply {cur}, {std} : {w._type_str(comp, shape)}")
    s = w._new_name()
    lines.append(f"{s} = stablehlo.add {mean}, {zp} : {w._type_str(comp, shape)}")
    if fast:
        out = s
    else:
        out = w._new_name()
        lines.append(
            f"{out} = stablehlo.convert {s} : ({w._type_str(comp, shape)}) -> "
            f"{w._type_str(out_dtype, shape)}"
        )
    w._names[id(op.result)] = out
    return "\n".join(lines)


def _emit_random_randint(w, op) -> str:
    """``random_randint`` → the algorithm's word stream, then the shared
    tail (u per word width — 2^-53 for 64-bit words, 2^-32 for 32-bit):
    u in f64; low/high cast to int64, span = high - low (wrapping int64);
    vals = (low + floor(u * span)).astype(dtype)."""
    alg = _algorithm_of(w, op)
    if alg == ALGORITHM_SPLITMIX64:
        return _emit_random_randint_splitmix64(w, op)
    _validate_key(w, op, alg)
    shape = _static_shape(w, op, op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    count = _count_of(shape)
    lines = []
    words = _emit_words_for(w, op, alg, _SALTS["randint"], count, lines)
    u = _emit_u_scaled32(w, words, count, _F64, lines)
    if tuple(shape) != (count,):
        r = w._new_name()
        lines.append(
            f"{r} = stablehlo.reshape {u} : (tensor<{count}xf64>) -> "
            f"{w._type_str(_F64, shape)}"
        )
        u = r
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
    """``random_permutation`` → the algorithm's n-word stream, then the
    shared splitmix64 tail: stable argsort of the UNSIGNED words (the i32
    sign-flip trick — xor the i32 words with 0x80000000 is monotonic and
    order-preserving on the unsigned order), ties resolve by index order —
    exactly ``np.argsort(words, kind="stable")``."""
    alg = _algorithm_of(w, op)
    if alg == ALGORITHM_SPLITMIX64:
        return _emit_random_permutation_splitmix64(w, op)
    _validate_key(w, op, alg)
    n = op.attributes.get("n")
    out_dtype = np.dtype(op.attributes["dtype"])
    if n is None:
        raise BackendError(
            f"stablehlo export: op 'random_permutation'{w._loc(op)} has a "
            "runtime population size operand (attr n=None) — the random "
            "expansions require a static n in v1 (the compiler backends "
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
    words = _emit_words_for(w, op, alg, _SALTS["permutation"], n, lines)
    flip, extra = _i32(w, 0x80000000, (n,))
    lines.extend(extra)
    wf = w._new_name()
    lines.append(f"{wf} = stablehlo.xor {words}, {flip} : tensor<{n}xi32>")
    _sv, si = w._emit_stable_argsort(wf, _I32, (n,), 0, lines)
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
        f"stablehlo export: op '{op.name}'{w._loc(op)} has no expansion for "
        "any random algorithm — random ops not in the expansion family "
        "stay deferred"
    )
