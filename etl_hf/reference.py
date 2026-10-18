"""Independent pure-numpy parity oracle for :mod:`etl_hf`.

This module is the *reference implementation* the compiled ``etl`` graphs in
:mod:`etl_hf.models` are validated against.  It is deliberately written twice:
the graphs are built from ``etl`` ops, while everything here is plain numpy
written in a **different formulation** (explicit per-head Python loops,
hand-written mean/variance formulas, ``math.erf``), so that a parity failure
points at exactly one of the two implementations and a shared bug would have to
be introduced twice, in two different shapes of code, to slip through.

Hard rules for this module:

* **numpy + stdlib only.**  Nothing here imports ``etl``, ``etl_hf.ops``,
  ``etl_hf.config`` or a family module — the one allowed ``etl_hf`` import is
  :class:`etl_hf.errors.HFUnsupportedError` (for :func:`resolve_activation`).
  Importing the graph code would make this oracle an echo of the thing it
  validates, which defeats its purpose.
* **float32 faithful.**  Every numeric input is normalised to
  ``numpy.float32`` and every numeric return value is ``numpy.float32``,
  mirroring the fp32 HuggingFace modules these primitives reproduce.  ``float64``
  inputs are therefore *downcast* (each docstring says so).  The only internal
  float64 pass is the :func:`gelu_erf` ``erf`` evaluation, which is rounded back
  to float32 before returning.
* **Primitives, not models.**  This module holds the shared numeric pieces
  (softmax / norms / activations / RoPE / attention) plus :func:`param_count`.
  The per-family forward passes live in the family modules and compose these
  helpers.
* **Explicit errors only.**  Malformed shapes, dtypes and static values raise
  ``ValueError`` / ``TypeError`` with a message naming the function, the
  argument and what was expected; unsupported activations raise
  :class:`etl_hf.errors.HFUnsupportedError`.  No silent coercion, no fallback.

Layout conventions used throughout (HF's ``LlamaForCausalLM`` /
``GPT2LMHeadModel`` layouts):

===============  ============================================================
``x``            ``(..., d)`` hidden states; ``d`` is the normalised/feature axis
``weight``       ``(d,)`` per-channel scale (HF ``nn.LayerNorm`` / RMSNorm shape)
``q/k/v``        ``(batch, heads, seq, head_dim)`` (attention, RoPE)
``cos/sin``      ``(seq, head_dim)`` RoPE tables
``mask``         ``(seq, seq)`` additive float mask (broadcastable)
===============  ============================================================
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable

import numpy as np

from etl_hf.errors import HFUnsupportedError

__all__ = [
    "ACTIVATIONS",
    "MASK_NEG",
    "additive_causal_mask",
    "apply_rope",
    "attention",
    "gelu_erf",
    "gelu_tanh",
    "layer_norm",
    "param_count",
    "relu",
    "resolve_activation",
    "rms_norm",
    "rope_tables",
    "silu",
    "softmax",
]


MASK_NEG: float = -1e4
"""Additive causal-mask constant (HuggingFace GPT-2's ``attn.masked_bias``).

Any value ``<= -104`` is small enough that ``exp(score + MASK_NEG - max)``
underflows to *exactly* ``0.0`` in float32, so adding this constant to the
attention scores gives masked-out keys exactly zero weight (HF's Llama uses
``finfo.min`` for the same reason).  ``-1e4`` is the value GPT-2's checkpoint
buffer actually carries, and it is comfortably inert in the f32 score range.
"""


# ---------------------------------------------------------------------------
# internal helpers: dtype normalisation + static-value validation
# ---------------------------------------------------------------------------
def _to_f32(value: object, label: str) -> np.ndarray:
    """Return ``value`` as a ``float32`` ndarray (the oracle's dtype policy).

    Args:
        value: Anything :func:`numpy.asarray` accepts (array, nested sequence,
            Python/numpy scalar).
        label: Function + argument name used in the error message.

    Returns:
        ``numpy.float32`` array.  No copy is made when the input already is a
        float32 array.

    Raises:
        TypeError: If ``value`` is complex-valued or not numeric.
    """
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise TypeError(f"{label} must be real-valued; got complex dtype {arr.dtype}")
    if arr.dtype == np.float32:
        return arr
    try:
        return arr.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - exotic dtypes only
        raise TypeError(f"{label} must be a real numeric array; got dtype {arr.dtype}") from exc


def _check_non_negative_int(value: object, label: str) -> int:
    """Return ``value`` as a Python ``int``, requiring ``value >= 0``.

    Raises:
        TypeError: If ``value`` is not integer-like (bools and numpy integers are
            accepted; floats are not).
        ValueError: If ``value`` is negative.
    """
    try:
        out = operator.index(value)  # accepts int / numpy integers, rejects floats
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer; got {value!r}") from exc
    if out < 0:
        raise ValueError(f"{label} must be >= 0; got {out!r}")
    return out


def _check_scalar(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Return ``value`` as a finite Python ``float``, optionally range-checked.

    Raises:
        TypeError: If ``value`` is not a real scalar.
        ValueError: If ``value`` is NaN/inf, ``<= 0`` when ``strictly_positive``,
            or below ``minimum`` when given.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real scalar; got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite; got {out!r}")
    if strictly_positive and not out > 0.0:
        raise ValueError(f"{label} must be > 0; got {out!r}")
    if minimum is not None and out < minimum:
        raise ValueError(f"{label} must be >= {minimum!r}; got {out!r}")
    return out


def _check_last_axis_vector(value: np.ndarray, channels: int, label: str) -> None:
    """Require ``value`` to be a ``(channels,)`` 1-D per-channel vector.

    Raises:
        ValueError: If the shape is not exactly ``(channels,)``.  This catches the
            classic ``(1, d)`` / transposed-weight mistake instead of silently
            broadcasting it.
    """
    if value.shape != (channels,):
        raise ValueError(
            f"{label} must have shape ({channels},) (one entry per normalised "
            f"feature); got shape {value.shape}"
        )


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``.

    Computes ``exp(x - max(x, axis)) / sum(exp(x - max(x, axis)), axis)``, i.e.
    the maximum is subtracted first so no ``exp`` term can overflow.

    Args:
        x: Real array of any rank with at least one dimension.
        axis: Axis to normalise over (default: last).

    Returns:
        ``float32`` array with ``x.shape``; every slice along ``axis`` sums to
        ``1.0`` (up to f32 rounding).  ``float64`` input is downcast.

    Raises:
        ValueError: If ``x`` is a 0-d scalar (nothing to normalise over) or
            ``axis`` is out of range.
        TypeError: If ``x`` is not real-valued.
    """
    arr = _to_f32(x, "softmax: x")
    if arr.ndim == 0:
        raise ValueError("softmax: x must have at least one dimension; got a 0-d scalar")
    peak = np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr - peak)
    total = np.sum(exp, axis=axis, keepdims=True)
    return np.asarray(exp / total, dtype=np.float32)


# ---------------------------------------------------------------------------
# normalisation layers
# ---------------------------------------------------------------------------
def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """HuggingFace ``nn.LayerNorm`` over the **last** axis (biased variance).

    ``((x - mean) / sqrt(var + eps)) * weight + bias`` where ``mean``/``var`` are
    taken over the last axis with ``keepdims`` and ``var`` is the **biased**
    (divide-by-N) estimator, exactly as ``torch.nn.LayerNorm`` computes it.

    Args:
        x: ``(..., d)`` float array; any leading batch/sequence axes.
        weight: ``(d,)`` per-channel scale (f32).
        bias: ``(d,)`` per-channel shift (f32).
        eps: Numerical floor added to the variance (>= 0, finite).

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` inputs are downcast.

    Raises:
        ValueError: If ``x`` has rank 0, ``weight``/``bias`` are not ``(d,)``, or
            ``eps`` is negative / non-finite.
        TypeError: If any input is non-real or ``eps`` is not a scalar.
    """
    arr = _to_f32(x, "layer_norm: x")
    scale = _to_f32(weight, "layer_norm: weight")
    shift = _to_f32(bias, "layer_norm: bias")
    eps_f = _check_scalar(eps, "layer_norm: eps", minimum=0.0)
    if arr.ndim < 1:
        raise ValueError("layer_norm: x must have at least one dimension; got a 0-d scalar")
    channels = arr.shape[-1]
    _check_last_axis_vector(scale, channels, "layer_norm: weight")
    _check_last_axis_vector(shift, channels, "layer_norm: bias")

    mean = np.mean(arr, axis=-1, keepdims=True, dtype=np.float32)
    centered = arr - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normed = centered / np.sqrt(var + np.float32(eps_f))
    return np.asarray(normed * scale + shift, dtype=np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """HuggingFace ``LlamaRMSNorm`` over the **last** axis (no mean subtraction).

    ``weight * x * rsqrt(mean(x**2, -1, keepdims) + eps)`` — the exact
    ``LlamaRMSNorm.forward`` order (scale ``x``, then multiply by ``weight``).

    Args:
        x: ``(..., d)`` float array.
        weight: ``(d,)`` per-channel scale (f32).
        eps: Numerical floor added to the mean square (>= 0, finite).

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` inputs are downcast.

    Raises:
        ValueError: If ``x`` has rank 0, ``weight`` is not ``(d,)``, or ``eps`` is
            negative / non-finite.
        TypeError: If any input is non-real or ``eps`` is not a scalar.
    """
    arr = _to_f32(x, "rms_norm: x")
    scale = _to_f32(weight, "rms_norm: weight")
    eps_f = _check_scalar(eps, "rms_norm: eps", minimum=0.0)
    if arr.ndim < 1:
        raise ValueError("rms_norm: x must have at least one dimension; got a 0-d scalar")
    channels = arr.shape[-1]
    _check_last_axis_vector(scale, channels, "rms_norm: weight")

    mean_square = np.mean(arr * arr, axis=-1, keepdims=True, dtype=np.float32)
    inv_rms = 1.0 / np.sqrt(mean_square + np.float32(eps_f))
    return np.asarray((arr * inv_rms) * scale, dtype=np.float32)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: ``x * sigmoid(x)``, evaluated in a numerically stable form.

    ``sigmoid(x)`` is computed from ``exp(-|x|)`` so large negative inputs do not
    overflow ``exp`` (the result underflows to ``0.0`` instead, which is what
    ``torch.nn.functional.silu`` produces).

    Args:
        x: Real array of any shape.

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` input is downcast.
    """
    arr = _to_f32(x, "silu: x")
    exp_neg_abs = np.exp(-np.abs(arr))
    one = np.float32(1.0)
    sigmoid = np.where(arr >= 0, one / (one + exp_neg_abs), exp_neg_abs / (one + exp_neg_abs))
    return np.asarray(arr * sigmoid, dtype=np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit: ``max(x, 0)``.

    Args:
        x: Real array of any shape.

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` input is downcast.
    """
    arr = _to_f32(x, "relu: x")
    return np.asarray(np.maximum(arr, np.float32(0.0)), dtype=np.float32)


_GELU_TANH_COEFF = np.float32(0.044715)
_GELU_TANH_SCALE = np.float32(math.sqrt(2.0 / math.pi))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh-approximated GELU — HF ``gelu_new`` / ``gelu_pytorch_tanh``.

    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``, the formula in
    ``transformers.activations.NewGELUActivation`` (which GPT-2 uses).

    Args:
        x: Real array of any shape.

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` input is downcast.
    """
    arr = _to_f32(x, "gelu_tanh: x")
    inner = _GELU_TANH_SCALE * (arr + _GELU_TANH_COEFF * arr * arr * arr)
    return np.asarray(np.float32(0.5) * arr * (np.float32(1.0) + np.tanh(inner)), dtype=np.float32)


_ERF = np.frompyfunc(math.erf, 1, 1)


def _erf_f64(values: np.ndarray) -> np.ndarray:
    """Elementwise ``math.erf`` over a float64 array (stdlib only, no scipy)."""
    return np.asarray(_ERF(values), dtype=np.float64)


_GELU_ERF_INV_SQRT2 = math.sqrt(0.5)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    """Exact (erf-form) GELU: ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    This is the f32 ``torch.nn.functional.gelu`` reference (HF's
    ``hidden_act: "gelu"``).  ``erf`` has no numpy ufunc and scipy is not a
    dependency, so the erf is evaluated with the stdlib ``math.erf`` over a
    **float64** pass and the product is rounded back to **float32**.  A float64
    intermediate is acceptable (and desirable) for the oracle: it keeps the
    reference's own rounding error far below the f32 budget the graph is
    compared against.

    Args:
        x: Real array of any shape.

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` input is downcast.

    Note:
        ``erf`` / ``gelu`` are *v1 StableHLO export deferrals* in etl, so an
        exact-erf config can be validated against this oracle on the numpy
        backend but cannot be compiled — the graph path must raise explicitly
        (see :mod:`etl_hf.models`).
    """
    arr = _to_f32(x, "gelu_erf: x")
    exact = arr.astype(np.float64)
    cdf = 0.5 * (1.0 + _erf_f64(exact * _GELU_ERF_INV_SQRT2))
    return np.asarray(exact * cdf, dtype=np.float32)


ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "silu": silu,
    "swish": silu,
    "gelu_new": gelu_tanh,
    "gelu_pytorch_tanh": gelu_tanh,
    "gelu": gelu_erf,
    "erf": gelu_erf,
    "relu": relu,
}
"""Map of HF ``config.hidden_act`` / ``activation_function`` names to callables.

``"swish"`` is the synonym of ``"silu"``; ``"gelu_new"`` and
``"gelu_pytorch_tanh"`` are the tanh approximation (GPT-2's activation);
``"gelu"`` is the exact erf form (numpy-backend only, see :func:`gelu_erf`);
``"erf"`` is treated as that same exact-erf GELU variant, the only erf-based
transformer activation.
"""


def resolve_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the activation callable registered under ``name``.

    Args:
        name: HF activation name, matched **case-sensitively** (HF's ``ACT2FN``
            is case-sensitive too).

    Returns:
        One of the callables in :data:`ACTIVATIONS`.

    Raises:
        HFUnsupportedError: If ``name`` is not a supported activation; the message
            names it and lists ``sorted(ACTIVATIONS)``.
    """
    if not isinstance(name, str) or name not in ACTIVATIONS:
        raise HFUnsupportedError(
            f"unsupported hidden activation {name!r}; supported activations are "
            f"{sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# causal mask
# ---------------------------------------------------------------------------
def additive_causal_mask(seq_len: int) -> np.ndarray:
    """Causal additive mask of shape ``(seq_len, seq_len)`` in float32.

    Entry ``(i, j)`` is ``0.0`` when key position ``j <= i`` (the query may
    attend to itself and the past) and :data:`MASK_NEG` when ``j > i`` (strictly
    future).  Adding it to the attention scores is the HF convention; the
    resulting softmax weight for a future key is *exactly* ``0.0`` in f32.

    Args:
        seq_len: Sequence length (integer >= 0).

    Returns:
        ``(seq_len, seq_len)`` float32 array; a ``(0, 0)`` array for
        ``seq_len == 0``.

    Raises:
        TypeError: If ``seq_len`` is not integer-like.
        ValueError: If ``seq_len`` is negative.
    """
    size = _check_non_negative_int(seq_len, "additive_causal_mask: seq_len")
    idx = np.arange(size)
    keep = idx[None, :] <= idx[:, None]  # key position j <= query position i
    return np.asarray(np.where(keep, np.float32(0.0), np.float32(MASK_NEG)), dtype=np.float32)


# ---------------------------------------------------------------------------
# rotary position embeddings (HF rotate_half convention)
# ---------------------------------------------------------------------------
def rope_tables(seq_len: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotary-embedding cos/sin tables, bit-for-bit the HF computation.

    Reproduces ``LlamaRotaryEmbedding`` exactly::

        inv_freq = theta ** (arange(0, head_dim, 2, float32) * (-1 / head_dim))
        freqs    = outer(arange(seq_len, float32), inv_freq)
        emb      = concat([freqs, freqs], axis=-1)
        cos, sin = cos(emb), sin(emb)

    all in float32, so the duplicated first/second half matches the
    ``rotate_half`` layout used by :func:`apply_rope`.

    Args:
        seq_len: Number of positions (integer >= 0).
        head_dim: Attention head width; must be a positive **even** integer (the
            ``rotate_half`` layout splits it in two halves).
        theta: RoPE base (``rope_theta``, e.g. ``10000.0``); finite and > 0.

    Returns:
        ``(cos, sin)``, both ``(seq_len, head_dim)`` float32.

    Raises:
        TypeError: If ``seq_len``/``head_dim`` are not integer-like or ``theta``
            is not a scalar.
        ValueError: If ``head_dim`` is not positive and even, ``seq_len`` is
            negative, or ``theta`` is not finite and > 0.
    """
    seq = _check_non_negative_int(seq_len, "rope_tables: seq_len")
    width = _check_non_negative_int(head_dim, "rope_tables: head_dim")
    if width == 0 or width % 2:
        raise ValueError(
            f"rope_tables: head_dim must be a positive even integer (the rotate_half "
            f"layout needs two equal halves); got {width}"
        )
    theta_f = _check_scalar(theta, "rope_tables: theta", strictly_positive=True)

    pair_index = np.arange(0, width, 2, dtype=np.float32)
    inv_freq = np.power(np.float32(theta_f), pair_index * np.float32(-1.0 / width))
    freqs = np.arange(seq, dtype=np.float32)[:, None] * inv_freq.astype(np.float32)[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.asarray(np.cos(emb), dtype=np.float32), np.asarray(np.sin(emb), dtype=np.float32)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate ``x`` with the HF ``rotate_half`` RoPE convention.

    ``rot = concat([-x2, x1])`` (where ``x1``/``x2`` are the two head halves),
    then ``x * cos + rot * sin``, with ``cos``/``sin`` broadcast over the leading
    batch/head axes.  At position ``0`` (``cos = 1``, ``sin = 0``) this is the
    identity; for every other position it is a per-pair 2-D rotation, so the
    per-position norm is preserved.

    Args:
        x: ``(heads, seq, head_dim)`` or ``(batch, heads, seq, head_dim)`` float
            array (the 3-D form is the 4-D layout with the batch axis dropped).
        cos: ``(seq, head_dim)`` float32 cos table from :func:`rope_tables`.
        sin: ``(seq, head_dim)`` float32 sin table from :func:`rope_tables`.

    Returns:
        ``float32`` array with ``x.shape``.  ``float64`` inputs are downcast.

    Raises:
        ValueError: If ``x`` is not 3-D/4-D, ``cos``/``sin`` are not equal 2-D
            tables, the trailing ``(seq, head_dim)`` axes of ``x`` disagree with
            the table shape, or ``head_dim`` is odd.
        TypeError: If any input is non-real.
    """
    arr = _to_f32(x, "apply_rope: x")
    cos_t = _to_f32(cos, "apply_rope: cos")
    sin_t = _to_f32(sin, "apply_rope: sin")
    if arr.ndim not in (3, 4):
        raise ValueError(
            "apply_rope: x must be (heads, seq, head_dim) or "
            f"(batch, heads, seq, head_dim); got shape {arr.shape}"
        )
    if cos_t.ndim != 2 or cos_t.shape != sin_t.shape:
        raise ValueError(
            "apply_rope: cos and sin must be equal-shaped 2-D (seq_len, head_dim) "
            f"tables; got cos {cos_t.shape} and sin {sin_t.shape}"
        )
    seq, width = cos_t.shape
    if arr.shape[-2] != seq or arr.shape[-1] != width:
        raise ValueError(
            f"apply_rope: x trailing (seq, head_dim) = {(arr.shape[-2], arr.shape[-1])} "
            f"does not match the cos/sin table shape {cos_t.shape}"
        )
    if width % 2:
        raise ValueError(
            f"apply_rope: head_dim must be even for the rotate_half convention; got {width}"
        )

    half = width // 2
    rotated = np.concatenate([-arr[..., half:], arr[..., :half]], axis=-1)
    broadcast = (1,) * (arr.ndim - 2) + (seq, width)
    cos_b = cos_t.reshape(broadcast)
    sin_b = sin_t.reshape(broadcast)
    return np.asarray(arr * cos_b + rotated * sin_b, dtype=np.float32)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Causal multi-head attention, one explicit Python loop per (batch, head).

    For each batch element and head::

        scores  = (q_h @ k_h.T) * scale + mask
        context = softmax(scores) @ v_h

    filled into a ``(batch, seq, heads * head_dim)`` output (heads merged in
    order, i.e. the layout a ``o_proj`` / ``c_proj`` linear consumes).  The
    per-head loop is intentional: it is a *different* formulation from the graph
    version (which is one batched ``dot`` chain), so a shared indexing bug is
    impossible.

    Grouped-query attention is supported here: with ``heads_q > heads_kv`` each
    K/V head is repeated ``heads_q // heads_kv`` times inside this function.

    Args:
        q: ``(batch, heads_q, seq, head_dim)`` float array.
        k: ``(batch, heads_kv, seq, head_dim)`` float array.
        v: ``(batch, heads_kv, seq, head_dim)`` float array (same shape as ``k``).
        mask: Additive mask broadcastable to ``(seq, seq)``, optionally with
            singleton leading axes (HF's ``(1, 1, seq, seq)``); use
            :func:`additive_causal_mask` for causal attention.
        scale: Score scale (``head_dim ** -0.5`` for the HF models).

    Returns:
        ``(batch, seq, heads_q * head_dim)`` float32.  ``float64`` inputs are
        downcast.

    Raises:
        ValueError: If ``q``/``k``/``v`` are not 4-D, their batch/head_dim/seq
            spans disagree, ``heads_q % heads_kv != 0``, or ``mask`` is not
            broadcastable to ``(seq, seq)`` (or carries a non-singleton leading
            axis).
        TypeError: If any input is non-real or ``scale`` is not a scalar.
    """
    q32 = _to_f32(q, "attention: q")
    k32 = _to_f32(k, "attention: k")
    v32 = _to_f32(v, "attention: v")
    for name, arr in (("q", q32), ("k", k32), ("v", v32)):
        if arr.ndim != 4:
            raise ValueError(
                f"attention: {name} must be (batch, heads, seq, head_dim); "
                f"got shape {arr.shape}"
            )
    batch, n_q, seq, head_dim = q32.shape
    n_kv = k32.shape[1]
    if n_kv == 0:
        raise ValueError("attention: k/v must have at least one head; got 0")
    if n_q % n_kv:
        raise ValueError(
            f"attention: number of query heads ({n_q}) must be a multiple of the number "
            f"of key/value heads ({n_kv}) for grouped-query attention"
        )
    expected_kv = (batch, n_kv, seq, head_dim)
    if k32.shape != expected_kv:
        raise ValueError(f"attention: k must have shape {expected_kv}; got {k32.shape}")
    if v32.shape != expected_kv:
        raise ValueError(f"attention: v must have shape {expected_kv}; got {v32.shape}")

    mask_arr = _to_f32(mask, "attention: mask")
    if mask_arr.ndim < 2:
        raise ValueError(
            f"attention: mask must be at least 2-D (seq, seq); got shape {mask_arr.shape}"
        )
    leading = mask_arr.shape[:-2]
    if any(dim != 1 for dim in leading):
        raise ValueError(
            f"attention: mask shape {mask_arr.shape} may carry only singleton leading axes "
            f"(e.g. HF's (1, 1, seq, seq)); got leading axes {leading}"
        )
    mask_2d = mask_arr.reshape(mask_arr.shape[-2:])
    try:
        mask32 = np.broadcast_to(mask_2d, (seq, seq))
    except ValueError as exc:
        raise ValueError(
            f"attention: mask of shape {mask_2d.shape} is not broadcastable to "
            f"({seq}, {seq})"
        ) from exc
    scale_f = _check_scalar(scale, "attention: scale")
    scale32 = np.float32(scale_f)

    repeat = n_q // n_kv
    if repeat != 1:
        k32 = np.repeat(k32, repeat, axis=1)
        v32 = np.repeat(v32, repeat, axis=1)

    out = np.zeros((batch, seq, n_q * head_dim), dtype=np.float32)
    for batch_index in range(batch):
        for head in range(n_q):
            # 2-D matmul per head: no batched-matmul shortcut, so this cannot
            # mirror an indexing mistake in the graph's batched dot chain.
            scores = (
                q32[batch_index, head] @ k32[batch_index, head].T
            ).astype(np.float32) * scale32 + mask32
            weights = softmax(scores)
            out[batch_index, :, head * head_dim : (head + 1) * head_dim] = (
                weights @ v32[batch_index, head]
            )
    return out


# ---------------------------------------------------------------------------
# parameter counting
# ---------------------------------------------------------------------------
def param_count(tree: object) -> int:
    """Total number of scalar elements in a nested parameter tree.

    Args:
        tree: A nested structure of ``dict`` / ``list`` / ``tuple`` nodes whose
            leaves are ``numpy.ndarray`` (0-d arrays and numpy scalars count as
            one element each).

    Returns:
        The element count as a Python ``int``.

    Raises:
        TypeError: If a leaf is not a numpy array/scalar (the message names the
            offending path, e.g. ``params['layers'][0]['w']``).
    """
    return _count(tree, "params")


def _count(node: object, path: str) -> int:
    """Recursive worker for :func:`param_count` (``path`` locates error nodes)."""
    if isinstance(node, np.ndarray):
        return int(node.size)
    if isinstance(node, np.generic):
        return 1
    if isinstance(node, dict):
        return sum(_count(value, f"{path}[{key!r}]") for key, value in node.items())
    if isinstance(node, (list, tuple)):
        return sum(_count(value, f"{path}[{index}]") for index, value in enumerate(node))
    raise TypeError(
        f"param_count: unsupported node {type(node).__name__} at {path}; expected a "
        "nested dict/list/tuple whose leaves are numpy arrays"
    )
