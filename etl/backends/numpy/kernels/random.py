"""Numpy kernels for the key-based RNG ops (etl.random) — 3 algorithms.

Reference implementation — deterministic, pure, no state, no ``np.random``.

Every random op is a deterministic function of its key operand and the op's
``algorithm`` attribute (canonical names from ``etl/ir/op_defs/random.py``):
``splitmix64`` (the v1 default — a rank-0 int64 64-bit state),
``threefry2x32`` (2 x uint32 key, 2 x uint32 counter, 20 rounds), and
``philox4x32_10`` (4 x uint32 key, 2 x uint32 counter, 10 rounds). The same
key + operands ALWAYS produce bit-identical values (repeatable across
evaluate calls).

WORD-STREAM CONTRACT (pinned — the bit-exactness reference for the StableHLO
export and all other backends; never re-derive it elsewhere):

- **Per-op salt**: every op kind uses the fixed 64-bit salt in ``_SALTS``
  (hex digits of pi) so different op kinds sharing one key draw decorrelated
  streams. The salt is folded into the KEY (never the counter):
  ``salt_lo = salt & 0xFFFFFFFF``, ``salt_hi = (salt >> 32) & 0xFFFFFFFF``;
  threefry: ``key' = (k0 ^ salt_lo, k1 ^ salt_hi)``; philox: ``key' =
  (k0 ^ salt_lo, k1 ^ salt_hi, k2 ^ salt_lo, k3 ^ salt_hi)``.
- **Counter**: cipher call/block p (0-based) uses counter ``(p, 0)`` — LOW
  word = call index, high word = 0. Threefry outputs 2 words per call,
  philox outputs 4 words per block.
- **Element -> flat-word mapping (C-order)**: ``uniform``/``randint``/
  ``permutation``: element i uses flat word i. ``normal``: element i uses
  flat words 2i (u1) and 2i+1 (u2) — the ``w[0::2]``/``w[1::2]`` convention.
- **Post-processing (parametrized by word width)**:
  - 64-bit words (splitmix64): uniform u = ``(w >> 11) * 2^-53`` (top 53
    bits; the v1 scaling — UNCHANGED).
  - 32-bit words (threefry/philox): uniform u = ``w * 2^-32`` (the FULL
    32-bit word scaled to [0, 1); this choice is pinned — the StableHLO
    export must mirror it exactly.
  - Box-Muller, ``floor(u * span)`` randint, stable-argsort permutation, and
    inverse-CDF multinomial use identical formulas given u1/u2.
- Cipher math runs on uint32 (int32 key words convert to uint32 for the
  cipher, and back).

CIPHERS:

- **Threefry2x32-20**: ported from the published Random123 reference
  (``threefry.h``; identical to the JAX ``threefry2x32`` reference): 20
  rounds = 5 iterations of (4 unrolled rounds + key injection); rotation
  constants 13/15/26/6 then 17/29/16/24 alternating per iteration; key
  schedule k0, k1, k2 = k0 ^ k1 ^ 0x1BD11BDA (SKEIN parity), injection
  ``(i+1)``-th uses (k_{(i+1) mod 3}, k_{(i+2) mod 3}) plus the counter
  increment ``X1 += (i+1)``. Random123 KAT vectors pin it exactly.
- **Philox4x32-10**: Random123 round function (two 32x32 -> 64 multiplications
  by M0=0xD2511F53 / M1=0xCD9E8D57, high words xored with the counter and the
  round key, Weyl constants W0=0x9E3779B9 / W1=0xBB67AE85). The etl key type
  is FOUR words (vs Random123's two), so the reference definition uses a
  cyclically permuted round-key PAIR: round r (0-based) uses
  ``(k_{2r mod 4} + r*W0, k_{(2r+1) mod 4} + r*W1)`` (mod 2^32) — pair
  (k0, k1), (k2, k3), (k0, k1), ... With a ZERO key this reduces exactly to
  the Random123 Philox4x32-10 definition (round key = r*(W0, W1)), so the
  standard Random123 KAT vector (key 0, ctr 0 -> 0x6627e8d5, 0xe169c58d,
  0xbc57ac4c, 0x9b00dbd8) pins this variant at key=0.

``random_key_mix`` (the split building block) derives one child key per
algorithm: splitmix64 -> ``mix(key ^ salt)`` (rank-0 int64); threefry ->
threefry2x32(key' = key ^ salt folded as above, counter (0, 0)) -> 2 words;
philox -> philox4x32_10(key' = key ^ salt folded as above, counter (0, 0))
-> 4 words. Salts unchanged (0 / golden / i*golden).

All arithmetic is exact integer math (numpy uint32/uint64 wrap mod 2^32/2^64;
explicit masks keep it version-stable). Uniform values are float64.
"""
from __future__ import annotations

import numpy as np

from etl import core
from etl.ir.op_defs.random import (
    DEFAULT_ALGORITHM,
    algorithm_key_type,
    validate_algorithm,
)

__all__ = ["register_kernels"]

_GOLDEN = 0x9E3779B97F4A7C15
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_SHIFT30 = np.uint64(30)
_SHIFT27 = np.uint64(27)
_SHIFT31 = np.uint64(31)
_SHIFT11 = np.uint64(11)
_B_MUL = np.uint64(0xBF58476D1CE4E5B9)
_C_MUL = np.uint64(0x94D049BB133111EB)
_INV_2P53 = 1.0 / 2.0**53  # exact float64

#: Fixed per-op salts: sampling ops seed their stream with the salt folded
#: into the key so different op kinds sharing one key draw decorrelated
#: streams. (Hex digits of pi; any distinct 64-bit constants work.)
_SALTS = {
    "uniform": 0x243F6A8885A308D3,
    "normal": 0x13198A2E03707344,
    "randint": 0xA4093822299F31D0,
    "permutation": 0x082EFA98EC4E6C89,
    "multinomial": 0x452821E638D01377,
}

# --- 32-bit constants (threefry2x32 / philox4x32_10) -------------------------

_INV_2P32 = 1.0 / 2.0**32  # exact float64; full-word 32-bit uniform scaling

#: SKEIN parity constant for Threefry2x32 (ks2 = k0 ^ k1 ^ parity).
_PARITY32 = np.uint32(0x1BD11BDA)

#: Threefry2x32 rotation constants, alternating per 4-round iteration
#: (Random123 threefry.h R_32x2_*; matches the JAX reference exactly).
_ROT_A = (13, 15, 26, 6)
_ROT_B = (17, 29, 16, 24)

#: Philox4x32-10 multipliers and Weyl constants (Random123 philox.h).
_PHILOX_M0 = np.uint32(0xD2511F53)
_PHILOX_M1 = np.uint32(0xCD9E8D57)
_PHILOX_W0 = np.uint32(0x9E3779B9)
_PHILOX_W1 = np.uint32(0xBB67AE85)


def _key_u64(key_tensor: core.Tensor) -> int:
    """The key operand's 64-bit state as a Python int in [0, 2^64)."""
    return int(key_tensor.numpy().item()) & 0xFFFFFFFFFFFFFFFF


def _mix3(z):
    """The three SplitMix64 mixing steps (vectorized, uint64, wrapped)."""
    z = (z ^ (z >> _SHIFT30)) * _B_MUL & _MASK64
    z = (z ^ (z >> _SHIFT27)) * _C_MUL & _MASK64
    return (z ^ (z >> _SHIFT31)) & _MASK64


def _mix_scalar(x: int) -> int:
    """Canonical SplitMix64 mix of one 64-bit value (Python int in/out)."""
    z = (x + _GOLDEN) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _words(seed: int, count: int) -> np.ndarray:
    """``count`` SplitMix64 words (uint64 array) from a 64-bit seed.

    Canonical SplitMix64 generator: word_i = mix(seed + (i+1) * GOLDEN)
    (state advances by GOLDEN between outputs; mix adds GOLDEN internally).
    Vectorized in exact uint64 arithmetic — deterministic and stable.
    """
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    states = (
        np.uint64(seed)
        + np.arange(count, dtype=np.uint64) * np.uint64(_GOLDEN)
    ) & _MASK64
    return _mix3((states + np.uint64(_GOLDEN)) & _MASK64)


def _algorithm(op) -> str:
    """The op's canonical ``algorithm`` attribute (validated; defaults to
    splitmix64 when the attribute is absent)."""
    return validate_algorithm(op.attributes.get("algorithm", DEFAULT_ALGORITHM))


def _extract_key(key_tensor: core.Tensor, algorithm: str):
    """Normalize a key operand for its algorithm.

    splitmix64 -> the 64-bit state as a Python int (existing behavior);
    threefry2x32 -> the 2 key words as a uint32 array; philox4x32_10 -> the
    4 key words as a uint32 array. Counter-based key shapes/dtypes are
    validated (the interpreter already validates against the traced spec; the
    check is a safety net for direct kernel invocation).
    """
    if algorithm == "splitmix64":
        return _key_u64(key_tensor)
    shape, dtype = algorithm_key_type(algorithm)
    arr = key_tensor.numpy()
    if arr.shape != tuple(shape) or arr.dtype != np.dtype(dtype):
        raise ValueError(
            f"random op with algorithm {algorithm!r}: key must be a "
            f"{tuple(shape)} {np.dtype(dtype)} tensor, got shape {arr.shape} "
            f"dtype {arr.dtype}"
        )
    return arr.astype(np.uint32)  # bit-preserving (int32 -> uint32)


def _threefry2x32(c0, c1, k0, k1):
    """Vectorized Threefry2x32-20 over cipher calls.

    Args:
        c0, c1: counter words (uint32 scalars or equal-shaped uint32 arrays).
        k0, k1: key words (uint32 scalars).

    Returns:
        (x0, x1): the two output words (same shape as the counters, uint32).

    Ported from Random123 ``threefry.h`` (identical to the JAX
    ``threefry2x32`` reference): 20 rounds = 5 iterations of (4 unrolled
    rounds + key injection); each round is ``x0 += x1; x1 = rotl(x1, r) ^ x0``
    with rotations 13/15/26/6 (even iterations) or 17/29/16/24 (odd);
    injection ``i+1`` (0-based i) adds ``(ks[(i+1) % 3], ks[(i+2) % 3])`` to
    ``(x0, x1)`` plus the counter increment ``x1 += (i+1)``, where
    ``ks = (k0, k1, k0 ^ k1 ^ 0x1BD11BDA)``. Pinned by the Random123 KAT:
    (key 0, ctr 0) -> (0x6b200159, 0x99ba4efe).
    """
    # Work in uint64 lanes (values in [0, 2^32)): adds of two 32-bit values
    # never overflow uint64, so no wrap-around is relied on (and numpy emits
    # no overflow warnings, incl. for scalar cipher calls).
    M = np.uint64(0xFFFFFFFF)
    k0 = np.uint64(k0)
    k1 = np.uint64(k1)
    x0 = ((np.asarray(c0, dtype=np.uint64) & M) + k0) & M
    x1 = ((np.asarray(c1, dtype=np.uint64) & M) + k1) & M
    ks = (k0, k1, k0 ^ k1 ^ np.uint64(_PARITY32))
    for i in range(5):
        rots = _ROT_A if i % 2 == 0 else _ROT_B
        for r in rots:
            x0 = (x0 + x1) & M
            x1 = (((x1 << np.uint64(r)) | (x1 >> np.uint64(32 - r))) ^ x0) & M
        x0 = (x0 + ks[(i + 1) % 3]) & M
        x1 = (x1 + ks[(i + 2) % 3] + np.uint64(i + 1)) & M
    return x0.astype(np.uint32), x1.astype(np.uint32)


def _philox4x32(c0, c1, c2, c3, key):
    """Vectorized Philox4x32-10 over cipher blocks.

    Args:
        c0..c3: counter words (uint32 scalars or equal-shaped uint32 arrays).
        key: the 4 key words (uint32 scalar-array of shape (4,)).

    Returns:
        (w0, w1, w2, w3): the four output words (same shape as the counter
        arrays, uint32).

    Round function from Random123 ``philox.h``: one round maps the state
    (c0, c1, c2, c3) to
    ``(hi1 ^ c1 ^ rk0, lo1, hi0 ^ c3 ^ rk1, lo0)`` where ``(hi0, lo0)`` /
    ``(hi1, lo1)`` are the 64-bit products ``M0 * c0`` / ``M1 * c2``.
    ROUND-KEY SCHEDULE (the etl reference definition for the 4-word key —
    pinned; the StableHLO export must mirror it exactly): round r (0-based)
    uses the cyclically permuted word pair
    ``rk0 = k_{2r mod 4} + r*W0``, ``rk1 = k_{(2r+1) mod 4} + r*W1``
    (mod 2^32) — pair (k0, k1), (k2, k3), (k0, k1), ... With a zero key this
    reduces exactly to the Random123 schedule (round key = r*(W0, W1)), so
    the standard KAT vector (key 0, ctr 0 -> 0x6627e8d5, 0xe169c58d,
    0xbc57ac4c, 0x9b00dbd8) pins this variant at key=0.
    """
    k = np.asarray(key, dtype=np.uint32).reshape(4)
    # Work in uint64 lanes (values in [0, 2^32)) so the 32x32 -> 64 multiply
    # is exact and never overflows.
    x0 = np.asarray(c0, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    x1 = np.asarray(c1, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    x2 = np.asarray(c2, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    x3 = np.asarray(c3, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    for r in range(10):
        p0 = x0 * np.uint64(_PHILOX_M0)  # < 2^64: exact 64-bit product
        p2 = x2 * np.uint64(_PHILOX_M1)
        if r % 2 == 0:
            rk0 = (np.uint64(k[0]) + np.uint64(r) * np.uint64(_PHILOX_W0)) & np.uint64(0xFFFFFFFF)
            rk1 = (np.uint64(k[1]) + np.uint64(r) * np.uint64(_PHILOX_W1)) & np.uint64(0xFFFFFFFF)
        else:
            rk0 = (np.uint64(k[2]) + np.uint64(r) * np.uint64(_PHILOX_W0)) & np.uint64(0xFFFFFFFF)
            rk1 = (np.uint64(k[3]) + np.uint64(r) * np.uint64(_PHILOX_W1)) & np.uint64(0xFFFFFFFF)
        x0 = (p2 >> 32) ^ x1 ^ rk0
        x1 = p2 & np.uint64(0xFFFFFFFF)
        x2 = (p0 >> 32) ^ x3 ^ rk1
        x3 = p0 & np.uint64(0xFFFFFFFF)
    return (
        x0.astype(np.uint32),
        x1.astype(np.uint32),
        x2.astype(np.uint32),
        x3.astype(np.uint32),
    )


def _words_for(algorithm: str, key, salt: int, count: int) -> np.ndarray:
    """``count`` flat stream words: uint64 for splitmix64, uint32 for the
    counter-based algorithms (see the module docstring WORD-STREAM CONTRACT).

    Flat word i is the i-th C-order output word: threefry call p outputs
    flat words (2p, 2p+1); philox block p outputs flat words (4p..4p+3).
    """
    if count == 0:
        return np.empty(0, dtype=np.uint64 if algorithm == "splitmix64" else np.uint32)
    if algorithm == "splitmix64":
        return _words(key ^ salt, count)
    salt_lo = np.uint32(salt & 0xFFFFFFFF)
    salt_hi = np.uint32((salt >> 32) & 0xFFFFFFFF)
    if algorithm == "threefry2x32":
        k0 = key[0] ^ salt_lo
        k1 = key[1] ^ salt_hi
        n_calls = (count + 1) // 2
        p = np.arange(n_calls, dtype=np.uint32)
        x0, x1 = _threefry2x32(p, np.uint32(0), k0, k1)
        words = np.empty(2 * n_calls, dtype=np.uint32)
        words[0::2] = x0
        words[1::2] = x1
        return words[:count]
    # philox4x32_10
    key4 = np.array(
        [
            key[0] ^ salt_lo,
            key[1] ^ salt_hi,
            key[2] ^ salt_lo,
            key[3] ^ salt_hi,
        ],
        dtype=np.uint32,
    )
    n_blocks = (count + 3) // 4
    p = np.arange(n_blocks, dtype=np.uint32)
    w0, w1, w2, w3 = _philox4x32(p, np.uint32(0), np.uint32(0), np.uint32(0), key4)
    words = np.empty(4 * n_blocks, dtype=np.uint32)
    words[0::4] = w0
    words[1::4] = w1
    words[2::4] = w2
    words[3::4] = w3
    return words[:count]


def _word_uniforms(words: np.ndarray) -> np.ndarray:
    """float64 uniforms in [0, 1) from stream words, by word width:

    - 64-bit words (splitmix64): top-53-bit scaling ``(w >> 11) * 2^-53`` —
      the v1 scaling, UNCHANGED.
    - 32-bit words (threefry/philox): full-word scaling ``w * 2^-32`` —
      PINNED (the StableHLO export must mirror it exactly).
    """
    if words.dtype == np.uint64:
        return (words >> _SHIFT11) * _INV_2P53
    return words.astype(np.float64) * _INV_2P32


def _uniforms_for(algorithm: str, key, salt: int, shape: tuple) -> np.ndarray:
    """float64 uniform draws in [0, 1) with the given (evaluated) shape."""
    count = int(np.prod(shape)) if shape else 1
    words = _words_for(algorithm, key, salt, count)
    return _word_uniforms(words).reshape(shape)


def _child_words(algorithm: str, key_words: np.ndarray, salt: int) -> np.ndarray:
    """One derived key (the algorithm's key type) from a 64-bit salt: cipher
    call 0 (counter (0, 0)) with the salt folded into the key words per the
    word-stream contract."""
    salt_lo = np.uint32(salt & 0xFFFFFFFF)
    salt_hi = np.uint32((salt >> 32) & 0xFFFFFFFF)
    if algorithm == "threefry2x32":
        x0, x1 = _threefry2x32(
            np.uint32(0),
            np.uint32(0),
            key_words[0] ^ salt_lo,
            key_words[1] ^ salt_hi,
        )
        return np.array([x0, x1], dtype=np.uint32)
    key4 = np.array(
        [
            key_words[0] ^ salt_lo,
            key_words[1] ^ salt_hi,
            key_words[2] ^ salt_lo,
            key_words[3] ^ salt_hi,
        ],
        dtype=np.uint32,
    )
    w0, w1, w2, w3 = _philox4x32(
        np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0), key4
    )
    return np.array([w0, w1, w2, w3], dtype=np.uint32)


# --- kernels ------------------------------------------------------------------


def _key_mix_kernel(ctx, op, operands):
    salt = op.attributes["salt"] & 0xFFFFFFFFFFFFFFFF
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    if algorithm == "splitmix64":
        out = _mix_scalar(key ^ salt)
        arr = np.array(out, dtype=np.uint64).view(np.int64).reshape(())
        return core.Tensor(arr)
    words = _child_words(algorithm, key, salt)
    return core.Tensor(words.astype(np.int32))  # bit-preserving (uint32)


def _uniform_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    low = operands[1].numpy()
    high = operands[2].numpy()
    u = _uniforms_for(algorithm, key, _SALTS["uniform"], shape)
    vals = (low + u * (high - low)).astype(out_dtype)
    return core.Tensor(vals)


def _normal_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    mean = operands[1].numpy()
    std = operands[2].numpy()
    count = int(np.prod(shape)) if shape else 1
    # Box-Muller: each (u1, u2) pair yields one normal value — count pairs.
    words = _words_for(algorithm, key, _SALTS["normal"], 2 * count)
    u1 = _word_uniforms(words[0::2])
    u2 = _word_uniforms(words[1::2])
    # Clamp u1 away from 0 to the smallest positive uniform of the word width
    # (the u1 grid step), so log(u1) is never -inf.
    min_u1 = _INV_2P53 if words.dtype == np.uint64 else _INV_2P32
    u1 = np.maximum(u1, min_u1)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z = z.reshape(shape)
    vals = (mean + z * std).astype(out_dtype)
    return core.Tensor(vals)


def _randint_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    low = operands[1].numpy().astype(np.int64)
    high = operands[2].numpy().astype(np.int64)
    span = high - low
    if np.any(span <= 0):
        raise ValueError(
            "random.randint: high must be greater than low (runtime check "
            f"got span <= 0 for low={low}, high={high})"
        )
    u = _uniforms_for(algorithm, key, _SALTS["randint"], shape)
    # Float scaling (exact for spans < 2^53; no modulo bias).
    vals = (low + np.floor(u * span)).astype(out_dtype)
    return core.Tensor(vals)


def _permutation_kernel(ctx, op, operands):
    out_dtype = np.dtype(op.attributes["dtype"])
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    n = int(operands[1].numpy().item())
    if n < 0:
        raise ValueError(f"random.permutation: n must be >= 0, got {n}")
    words = _words_for(algorithm, key, _SALTS["permutation"], n)
    # Stable argsort: deterministic and version-stable; a valid permutation
    # even when 32-bit words collide (ties resolve by index order).
    perm = np.argsort(words, kind="stable").astype(out_dtype)
    return core.Tensor(perm)


def _multinomial_kernel(ctx, op, operands):
    out_dtype = np.dtype(op.attributes["dtype"])
    algorithm = _algorithm(op)
    key = _extract_key(operands[0], algorithm)
    num_samples = op.attributes["num_samples"]
    p = operands[1].numpy()
    if p.ndim != 1:
        raise ValueError(
            f"random.multinomial: input must be 1-D, got shape {p.shape}"
        )
    m = p.shape[0]
    if m == 0:
        raise ValueError("random.multinomial: input must be non-empty")
    if np.any(p < 0):
        raise ValueError(
            "random.multinomial: input probabilities must be non-negative"
        )
    if not np.isclose(p.sum(dtype=np.float64), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(
            "random.multinomial: input probabilities must sum to 1 "
            f"(got {p.sum(dtype=np.float64)})"
        )
    u = _uniforms_for(algorithm, key, _SALTS["multinomial"], (num_samples,))
    # Inverse-CDF sampling via cumulative search (np.random.choice semantics).
    cum = np.cumsum(p.astype(np.float64))
    idx = np.searchsorted(cum, u)
    idx = np.minimum(idx, m - 1)  # defensive clip for u == 1 edge (2^-53 headroom)
    return core.Tensor(idx.astype(out_dtype))


def register_kernels(table: dict) -> None:
    """Register the random-op kernels into ``table`` (see kernels/__init__)."""
    table["random_key_mix"] = _key_mix_kernel
    table["random_uniform"] = _uniform_kernel
    table["random_normal"] = _normal_kernel
    table["random_randint"] = _randint_kernel
    table["random_permutation"] = _permutation_kernel
    table["random_multinomial"] = _multinomial_kernel
