"""Numpy kernels for the key-based RNG ops (etl.random).

Reference implementation — deterministic, pure, no state, no ``np.random``.

Every random op is a deterministic function of its key operand (a rank-0
int64 tensor). The stream is derived with the SplitMix64 generator seeded
with ``key ^ SALT``, where ``SALT`` is a fixed per-op constant (below):
different op kinds sharing one key draw decorrelated streams; the same key +
operands ALWAYS produce bit-identical values (repeatable across evaluate
calls). ``random_key_mix`` implements ``mix(key ^ salt)`` — the building
block behind ``split``/``split_n``.

All arithmetic is exact 64-bit integer math (numpy uint64 wraps mod 2^64;
explicit masks keep it version-stable). Uniform values use the top 53 bits
of each word (float64 in [0, 1)), so results are reproducible bit-for-bit.
"""
from __future__ import annotations

import numpy as np

from etl import core

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

#: Fixed per-op salts: sampling ops seed their SplitMix64 stream with
#: ``key ^ SALT`` so different op kinds sharing one key draw decorrelated
#: streams. (Hex digits of pi; any distinct 64-bit constants work.)
_SALTS = {
    "uniform": 0x243F6A8885A308D3,
    "normal": 0x13198A2E03707344,
    "randint": 0xA4093822299F31D0,
    "permutation": 0x082EFA98EC4E6C89,
    "multinomial": 0x452821E638D01377,
}


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


def _uniforms(seed: int, shape: tuple) -> np.ndarray:
    """float64 uniform draws in [0, 1) with the given (evaluated) shape."""
    count = int(np.prod(shape)) if shape else 1
    u = (_words(seed, count) >> _SHIFT11) * _INV_2P53
    return u.reshape(shape)


# --- kernels ------------------------------------------------------------------


def _key_mix_kernel(ctx, op, operands):
    salt = op.attributes["salt"] & 0xFFFFFFFFFFFFFFFF
    out = _mix_scalar(_key_u64(operands[0]) ^ salt)
    arr = np.array(out, dtype=np.uint64).view(np.int64).reshape(())
    return core.Tensor(arr)


def _uniform_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    seed = _key_u64(operands[0]) ^ _SALTS["uniform"]
    low = operands[1].numpy()
    high = operands[2].numpy()
    u = _uniforms(seed, shape)
    vals = (low + u * (high - low)).astype(out_dtype)
    return core.Tensor(vals)


def _normal_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    seed = _key_u64(operands[0]) ^ _SALTS["normal"]
    mean = operands[1].numpy()
    std = operands[2].numpy()
    count = int(np.prod(shape)) if shape else 1
    # Box-Muller: each (u1, u2) pair yields one normal value — count pairs.
    w = _words(seed, 2 * count)
    u1 = np.maximum((w[0::2] >> _SHIFT11) * _INV_2P53, _INV_2P53)
    u2 = (w[1::2] >> _SHIFT11) * _INV_2P53
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z = z.reshape(shape)
    vals = (mean + z * std).astype(out_dtype)
    return core.Tensor(vals)


def _randint_kernel(ctx, op, operands):
    shape = ctx.evaluate_shape(op.attributes["shape"])
    out_dtype = np.dtype(op.attributes["dtype"])
    seed = _key_u64(operands[0]) ^ _SALTS["randint"]
    low = operands[1].numpy().astype(np.int64)
    high = operands[2].numpy().astype(np.int64)
    span = high - low
    if np.any(span <= 0):
        raise ValueError(
            "random.randint: high must be greater than low (runtime check "
            f"got span <= 0 for low={low}, high={high})"
        )
    u = _uniforms(seed, shape)
    # Float scaling (53-bit exact for spans < 2^53; no modulo bias).
    vals = (low + np.floor(u * span)).astype(out_dtype)
    return core.Tensor(vals)


def _permutation_kernel(ctx, op, operands):
    out_dtype = np.dtype(op.attributes["dtype"])
    seed = _key_u64(operands[0]) ^ _SALTS["permutation"]
    n = int(operands[1].numpy().item())
    if n < 0:
        raise ValueError(f"random.permutation: n must be >= 0, got {n}")
    words = _words(seed, n)
    # Stable argsort: deterministic and version-stable; ties (astronomically
    # unlikely for 64-bit words) resolve by index order.
    perm = np.argsort(words, kind="stable").astype(out_dtype)
    return core.Tensor(perm)


def _multinomial_kernel(ctx, op, operands):
    out_dtype = np.dtype(op.attributes["dtype"])
    seed = _key_u64(operands[0]) ^ _SALTS["multinomial"]
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
    u = _uniforms(seed, (num_samples,))
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
