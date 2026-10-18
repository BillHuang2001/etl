"""The transformer idioms ``etl`` does not ship, as graph→graph compositions.

``etl`` deliberately owns no NN layers: it has no ``softmax``, ``layer_norm``,
``rms_norm``, ``silu``, top-level ``where``, ``rsqrt`` or ``iota``. Everything in
this module is therefore a *composition* of public ``etl`` ops built on
``SymbolicTensor`` values — there is no eager numerical path here, and no new IR
op. Calling these functions with concrete tensors is the same user error as
calling ``etl.add`` on a concrete tensor (an ``etl.TraceError`` from the
underlying op), which is exactly the intended explicitness.

Shape/dtype policy
------------------
Every helper is **static-shape only**: where a helper needs an extent to build a
``reshape``/``slice``, it reads it from the operand's static shape and raises
:class:`etl_hf.errors.HFUnsupportedError` naming the offending axis if that dim
is symbolic (``Dim``/``DimExpr``). Nothing is ever guessed from a dynamic dim,
and no helper silently falls back to a Python loop. Cost: one graph per concrete
shape (the package-wide static-shape-first contract). Dtypes are whatever the
operands are — every helper preserves the input floating dtype; there is no
implicit cast.

RoPE tables: in-graph, not host inputs
-------------------------------------
:func:`rope_cos_sin` computes the ``(seq_len, head_dim)`` cos/sin tables **inside
the graph** from ``arange`` + ``power`` + ``cos``/``sin`` (all StableHLO-exportable
in v1) rather than taking them as extra graph inputs. This deliberately differs
from a host-precomputed-tables design and was chosen because it:

* keeps the graph signature at ``(params, input_ids)`` — no extra positional
  inputs the caller could forget to supply, and nothing to keep in sync when the
  sequence length changes;
* keeps the emitted program's op count independent of ``seq_len`` (the tables are
  a rank-1 iota expansion, not a payload);
* avoids an ``etl.constant`` payload, which is the one explicit-but-warned way to
  embed data (``etl.constant`` inlines the bytes into MLIR text);

and was measured bit-comparable to HuggingFace's own f32 tables. The host
equivalent :func:`rope_tables` is exposed for the pure-numpy reference oracle and
for reporting; it produces the same numbers via the same formula.

Export safety
-------------
Exact erf-GELU (:func:`gelu_erf`, i.e. ``hidden_act`` ``"gelu"`` / ``"erf"``)
uses ``etl.erf``, a documented **v1 StableHLO export deferral**: such a graph
still traces, compiles and runs on the numpy interpreter backend, but any
compiler backend rejects it with an explicit ``etl.BackendError``. Configs that
select it are flagged through :data:`EXPORT_UNSAFE_ACTIVATIONS` /
:func:`activation_is_export_safe` so the model builders can refuse the compiled
path up front instead of failing deep inside lowering.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import etl

from etl_hf.errors import HFUnsupportedError

__all__ = [
    "ACTIVATIONS",
    "EXPORT_UNSAFE_ACTIVATIONS",
    "MASK_NEG",
    "activation_is_export_safe",
    "apply_rope",
    "causal_mask",
    "gelu_erf",
    "gelu_tanh",
    "layer_norm",
    "merge_heads",
    "repeat_kv",
    "resolve_activation",
    "rms_norm",
    "rope_cos_sin",
    "rope_tables",
    "silu",
    "softmax",
    "split_heads",
]

#: Additive attention-mask constant for masked-out positions.
#:
#: Any value ``<= -104`` makes ``exp(v - max)`` underflow to *exactly* 0 in f32,
#: so the exact constant is numerically irrelevant; ``-1e4`` is what HF's GPT-2
#: checkpoint uses (buffer ``attn.masked_bias``). Exposed so references/tests can
#: reproduce the same mask without duplicating the number.
MASK_NEG: float = -1e4

#: 1/sqrt(2), the erf-GELU argument scale (``erf(x / sqrt(2))``).
_INV_SQRT2 = 0.7071067811865476

#: sqrt(2/pi), the tanh-GELU argument scale.
_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))

#: gelu_new / gelu_pytorch_tanh cubic coefficient.
_GELU_TANH_CUBIC = 0.044715


# ---------------------------------------------------------------------------
# small internal helpers
# ---------------------------------------------------------------------------
def _rsqrt(x):
    """``1 / sqrt(x)`` — ``etl`` has no ``rsqrt`` op, so compose it.

    Args:
        x: ``SymbolicTensor`` with strictly positive entries.

    Returns:
        ``SymbolicTensor``: same shape/dtype as ``x``.
    """
    return etl.divide(1.0, etl.sqrt(x))


def _static_dim(x, axis, what):
    """Return ``x.shape[axis]`` as a plain ``int``, or raise ``HFUnsupportedError``.

    ``etl`` shape entries are either plain ``int`` (static) or ``Dim``/``DimExpr``
    (symbolic). A ``Dim`` with a known ``size`` counts as static.

    Args:
        x: ``SymbolicTensor`` (or anything with a ``shape`` tuple).
        axis: int axis index (negative indices allowed).
        what: name of the calling public helper, used in the error message.

    Returns:
        int: the concrete extent at ``axis``.

    Raises:
        HFUnsupportedError: the axis is out of range, or its extent is symbolic.

    Note:
        Private helper: this is the single place where "static shapes only" is
        enforced, so every public helper reports the same actionable message.
    """
    shape = x.shape
    try:
        entry = shape[axis]
    except IndexError as exc:  # explicit: never let a bad axis look like a shape bug
        raise HFUnsupportedError(
            f"{what}: axis {axis} is out of range for rank {len(shape)} "
            f"(shape {tuple(shape)})."
        ) from exc
    if isinstance(entry, int):
        return entry
    size = getattr(entry, "size", None)
    if size is None:
        raise HFUnsupportedError(
            f"{what}: axis {axis} of the operand has the symbolic dimension {entry!r}; "
            "etl_hf builds static-shape graphs only — bake the concrete size into the "
            "graph (one graph per shape; see etl_hf/CONTEXT.md 'Static-shape-first')."
        )
    return int(size)


def _positive_int(value, what):
    """Validate a static integer parameter, raising ``ValueError`` when invalid.

    Args:
        value: the caller-supplied value.
        what: parameter name for the error message.

    Returns:
        int: ``value`` as an ``int``.

    Raises:
        ValueError: ``value`` is not a positive integer (a pure host-side argument
            error, so it is not an etl error).
    """
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{what} must be a positive integer, got {value!r}.")
    return int(value)


# ---------------------------------------------------------------------------
# normalisation + activations
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    """Numerically stable softmax over ``axis``: ``exp(x - max) / sum(exp(x - max))``.

    ``etl`` has no ``softmax`` op; this is the plain three-op composition. The
    max-subtraction makes it safe for large-magnitude inputs (a row of ``1e30``
    values yields exact one-hots rather than ``inf/inf`` → ``nan``). Both
    reductions use ``keepdims=True`` and the subtraction/division broadcast, so
    no ``reshape`` is involved — this keeps it clear of the exporter's
    dynamic-reshape restriction and works for any ``axis`` value or rank.

    Args:
        x: ``SymbolicTensor`` of floating dtype, any rank.
        axis: int axis to normalise over; ``-1`` (last axis, the usual attention
            axis) and ``0`` are both supported, as is any other axis.

    Returns:
        ``SymbolicTensor`` of the same shape and dtype as ``x``; each 1-D slice
        along ``axis`` sums to ~1.

    Raises:
        etl.core.TraceError: no active trace, or a concrete ``Tensor`` operand.
        etl.core.ShapeError: ``axis`` out of range.
    """
    shifted = etl.subtract(x, etl.reduce_max(x, axes=axis, keepdims=True))
    exp_shifted = etl.exp(shifted)
    return etl.divide(exp_shifted, etl.reduce_sum(exp_shifted, axes=axis, keepdims=True))


def layer_norm(x, weight, bias, eps):
    """HF ``nn.LayerNorm``: biased variance over the **last** axis.

    ``y = (x - mean) / sqrt(var + eps) * weight + bias`` where ``var`` is the
    *biased* (divide-by-N) variance, matching PyTorch/HF semantics — do not
    substitute the sample variance.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(..., features)``; the
            normalisation is over the last axis only (any rank ≥ 1).
        weight: per-feature scale, shape ``(features,)`` (broadcasts over leading
            axes).
        bias: per-feature offset, shape ``(features,)``.
        eps: float epsilon added to the variance before the square root (HF:
            ``layer_norm_eps`` / ``layer_norm_epsilon``).

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    mean = etl.reduce_mean(x, axes=-1, keepdims=True)
    centered = etl.subtract(x, mean)
    var = etl.reduce_mean(etl.multiply(centered, centered), axes=-1, keepdims=True)
    inv_std = _rsqrt(etl.add(var, eps))
    return etl.add(etl.multiply(etl.multiply(centered, inv_std), weight), bias)


def rms_norm(x, weight, eps):
    """HF ``LlamaRMSNorm``: ``weight * x * (1 / sqrt(mean(x^2, -1, keepdims) + eps))``.

    No mean subtraction and no bias, unlike :func:`layer_norm`.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(..., features)``.
        weight: per-feature scale, shape ``(features,)``.
        eps: float epsilon added to the mean square (HF: ``rms_norm_eps``).

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    mean_square = etl.reduce_mean(etl.multiply(x, x), axes=-1, keepdims=True)
    return etl.multiply(etl.multiply(x, _rsqrt(etl.add(mean_square, eps))), weight)


def silu(x):
    """SiLU / swish: ``x * sigmoid(x)``.

    Args:
        x: ``SymbolicTensor`` of floating dtype, any shape.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    return etl.multiply(x, etl.sigmoid(x))


def gelu_tanh(x):
    """``gelu_new`` / ``gelu_pytorch_tanh``: ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))``.

    This is the tanh approximation GPT-2 and most small checkpoints actually use
    (HF names it ``gelu_new`` / ``gelu_pytorch_tanh``); it is fully
    StableHLO-exportable in v1 — no ``erf`` involved.

    Args:
        x: ``SymbolicTensor`` of floating dtype, any shape.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    cubic = etl.multiply(x, etl.multiply(x, x))
    inner = etl.multiply(
        etl.add(x, etl.multiply(cubic, _GELU_TANH_CUBIC)),
        _SQRT_2_OVER_PI,
    )
    return etl.multiply(etl.multiply(x, 0.5), etl.add(1.0, etl.tanh(inner)))


def gelu_erf(x):
    """Exact erf-GELU: ``0.5x(1 + erf(x / sqrt(2)))``.

    **Numpy-interpreter backend only.** ``etl.erf`` is a documented v1 StableHLO
    *export deferral*, so any compiler backend (stablehlo / iree / tvm / xla)
    rejects a graph built with this function with an explicit
    ``etl.BackendError``. It is never silently replaced by the tanh
    approximation — if a config selects an erf-GELU, callers must either use the
    numpy backend or explicitly map the activation themselves. See
    :data:`EXPORT_UNSAFE_ACTIVATIONS` / :func:`activation_is_export_safe`.

    Args:
        x: ``SymbolicTensor`` of floating dtype, any shape.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    return etl.multiply(
        etl.multiply(x, 0.5),
        etl.add(1.0, etl.erf(etl.multiply(x, _INV_SQRT2))),
    )


def _relu(x):
    """ReLU: ``max(x, 0)`` (private — reachable through :data:`ACTIVATIONS`).

    Args:
        x: ``SymbolicTensor`` of floating dtype, any shape.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x``.
    """
    return etl.relu(x)


#: HF ``hidden_act`` / ``activation_function`` name → graph-building activation.
#:
#: HF spells the same function several ways across families (``silu``/``swish``,
#: ``gelu_new``/``gelu_pytorch_tanh``, ``gelu``/``erf``); every alias that means
#: the same math maps to the same builder.
ACTIVATIONS: dict[str, Callable] = {
    "silu": silu,
    "swish": silu,
    "gelu_new": gelu_tanh,
    "gelu_pytorch_tanh": gelu_tanh,
    "gelu": gelu_erf,
    "erf": gelu_erf,
    "relu": _relu,
}

#: Activation names whose graph uses ``etl.erf`` — unusable on compiler backends.
#:
#: Kept as an explicit set (rather than inferred) so the deferral is visible at
#: the API level: model builders check :func:`activation_is_export_safe` before
#: promising a compiled path.
EXPORT_UNSAFE_ACTIVATIONS: frozenset[str] = frozenset({"gelu", "erf", "gelu_erf"})


def resolve_activation(name):
    """Map an HF activation name to its graph-building activation function.

    Args:
        name: HF config value, e.g. ``"silu"``, ``"gelu_new"``, ``"gelu"``.

    Returns:
        Callable[[SymbolicTensor], SymbolicTensor]: the activation composition.

    Raises:
        HFUnsupportedError: ``name`` is not supported; the message lists every
            supported name in sorted order.
    """
    try:
        return ACTIVATIONS[name]
    except (KeyError, TypeError) as exc:
        raise HFUnsupportedError(
            f"unsupported activation {name!r}; supported activations: "
            f"{sorted(ACTIVATIONS)} (add a composition in etl_hf/ops.py for "
            "anything else)."
        ) from exc


def activation_is_export_safe(name):
    """Whether a graph built with this activation can be lowered to StableHLO.

    Args:
        name: HF activation name (must be supported — see
            :func:`resolve_activation`).

    Returns:
        bool: ``False`` exactly for the erf-based names
        (:data:`EXPORT_UNSAFE_ACTIVATIONS`), ``True`` otherwise.

    Raises:
        HFUnsupportedError: ``name`` is not a supported activation.
    """
    resolve_activation(name)  # validates the name (and raises for unknown ones)
    return name not in EXPORT_UNSAFE_ACTIVATIONS


# ---------------------------------------------------------------------------
# causal mask + rotary position embeddings
# ---------------------------------------------------------------------------
def causal_mask(seq_len):
    """In-graph ``(seq_len, seq_len)`` f32 additive causal mask.

    Entry ``(i, j)`` is ``0.0`` when ``j <= i`` (position ``i`` may attend to
    ``j``) and :data:`MASK_NEG` above the diagonal. Composed from
    ``arange`` + ``less_equal`` + ``cast`` + arithmetic — **not** an
    ``etl.constant`` payload — so the mask costs four cheap ops per graph and
    nothing is inlined into the emitted program.

    Args:
        seq_len: int sequence length (must be positive).

    Returns:
        ``SymbolicTensor`` f32 of shape ``(seq_len, seq_len)``; add it to raw
        attention scores *before* the softmax.

    Raises:
        ValueError: ``seq_len`` is not a positive integer.
    """
    seq_len = _positive_int(seq_len, "causal_mask(seq_len)")
    idx = etl.arange(0, seq_len, dtype=etl.int32)
    # keep[i, j] = (j <= i): query i may attend to keys up to and including itself.
    keep = etl.less_equal(
        etl.reshape(idx, (1, seq_len)),
        etl.reshape(idx, (seq_len, 1)),
    )
    above = etl.subtract(etl.cast(keep, etl.float32), 1.0)  # 0.0 on/below, -1.0 above
    return etl.multiply(above, -MASK_NEG)


def _rope_frequencies(seq_len, head_dim, theta):
    """Validate RoPE parameters and return ``(seq_len, head_dim // 2)`` angles.

    In-graph counterpart of the host-side validation in :func:`rope_tables`;
    factored out so both entry points reject the same bad parameters with the
    same messages.

    Args:
        seq_len: int sequence length.
        head_dim: int per-head width (must be even).
        theta: float RoPE base (HF: ``rope_theta``, 10000.0 by default).

    Returns:
        tuple: ``(seq_len, head_dim, half)`` as ints.

    Raises:
        ValueError: a non-positive/invalid integer parameter.
        HFUnsupportedError: ``head_dim`` is odd (``rotate_half`` pairs the two
            halves, so an odd width has no HF meaning).
    """
    seq_len = _positive_int(seq_len, "seq_len")
    head_dim = _positive_int(head_dim, "head_dim")
    if not isinstance(theta, (int, float, np.floating, np.integer)) or isinstance(theta, bool):
        raise ValueError(f"theta must be a positive real number, got {theta!r}.")
    if float(theta) <= 0.0:
        raise ValueError(f"theta must be a positive real number, got {theta!r}.")
    if head_dim % 2:
        raise HFUnsupportedError(
            f"head_dim must be even for RoPE (HF rotate_half pairs the two halves), "
            f"got {head_dim}."
        )
    return seq_len, head_dim, head_dim // 2


def rope_cos_sin(seq_len, head_dim, theta):
    """In-graph ``(seq_len, head_dim)`` f32 RoPE ``(cos, sin)`` tables.

    HF ``rotate_half`` layout: the ``head_dim // 2`` inverse frequencies
    ``theta ** (-2i / head_dim)`` are laid out by ``arange(0, head_dim, 2)`` and
    then duplicated, so ``cos = concat([cos(freqs), cos(freqs)], axis=1)`` and
    likewise for ``sin``. Built from ``arange`` + ``power`` + ``cos``/``sin`` (all
    StableHLO-exportable in v1), which is why the forward graph needs no extra
    positional inputs and no constant payload — see the module docstring.

    Args:
        seq_len: int sequence length.
        head_dim: int per-head width (must be even).
        theta: float RoPE base (HF: ``rope_theta``).

    Returns:
        tuple: ``(cos, sin)``, each a ``SymbolicTensor`` f32 of shape
        ``(seq_len, head_dim)``.

    Raises:
        ValueError: a non-positive/invalid integer parameter or non-positive
            ``theta``.
        HFUnsupportedError: ``head_dim`` is odd.
    """
    seq_len, head_dim, half = _rope_frequencies(seq_len, head_dim, theta)
    idx = etl.cast(etl.arange(0, head_dim, 2, dtype=etl.int32), etl.float32)
    inv_freq = etl.power(theta, etl.multiply(idx, -1.0 / head_dim))
    pos = etl.cast(etl.arange(0, seq_len, dtype=etl.int32), etl.float32)
    freqs = etl.multiply(
        etl.reshape(pos, (seq_len, 1)),
        etl.reshape(inv_freq, (1, half)),
    )
    emb = etl.concatenate([freqs, freqs], axis=1)
    return etl.cos(emb), etl.sin(emb)


def rope_tables(seq_len, head_dim, theta):
    """Host-side numpy f32 RoPE ``(cos, sin)`` tables — the reference oracle.

    Same formula, same dtype and same layout as :func:`rope_cos_sin` (HF
    ``rotate_half``): meant for the pure-numpy reference forward, for reporting,
    and for users who want the numbers without building a graph. Not used by the
    forward graph itself (see the module docstring).

    Args:
        seq_len: int sequence length.
        head_dim: int per-head width (must be even).
        theta: float RoPE base (HF: ``rope_theta``).

    Returns:
        tuple: ``(cos, sin)``, each ``np.ndarray`` f32 of shape
        ``(seq_len, head_dim)``.

    Raises:
        ValueError: a non-positive/invalid integer parameter or non-positive
            ``theta``.
        HFUnsupportedError: ``head_dim`` is odd.
    """
    seq_len, head_dim, _half = _rope_frequencies(seq_len, head_dim, theta)
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = np.power(
        np.float32(theta), idx * np.float32(-1.0 / head_dim)
    ).astype(np.float32)
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = (pos[:, None] * inv_freq[None, :]).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """Rotate ``(batch, heads, seq, head_dim)`` queries/keys with HF ``rotate_half``.

    ``y = x * cos + rotate_half(x) * sin`` where
    ``rotate_half(x) = concat([-x2, x1], -1)`` splits ``x`` into its two halves
    ``x1 = x[..., :hd/2]``, ``x2 = x[..., hd/2:]``. ``cos``/``sin`` (shape
    ``(seq, head_dim)``) are broadcast over the batch and head axes.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(b, n, s, hd)`` — static
            dims only.
        cos: ``(seq_len, head_dim)`` cos table, e.g. from :func:`rope_cos_sin`.
        sin: ``(seq_len, head_dim)`` sin table, same shape as ``cos``.

    Returns:
        ``SymbolicTensor`` with the same shape and dtype as ``x`` (a rotation, so
        per-position norms are preserved up to fp rounding).

    Raises:
        HFUnsupportedError: any dim of ``x`` is symbolic, ``head_dim`` is odd, or
            the table shape does not match ``(seq, head_dim)``.
    """
    b = _static_dim(x, 0, "apply_rope")
    n = _static_dim(x, 1, "apply_rope")
    s = _static_dim(x, 2, "apply_rope")
    hd = _static_dim(x, 3, "apply_rope")
    if hd % 2:
        raise HFUnsupportedError(
            f"apply_rope: head_dim must be even (HF rotate_half pairs the two halves), got {hd}."
        )
    half = hd // 2
    for name, table in (("cos", cos), ("sin", sin)):
        if tuple(table.shape) != (s, hd):
            raise HFUnsupportedError(
                f"apply_rope: {name} has shape {tuple(table.shape)}, expected "
                f"({s}, {hd}) to match x of shape {(b, n, s, hd)}."
            )
    x1 = etl.slice(x, (0, 0, 0, 0), (b, n, s, half))
    x2 = etl.slice(x, (0, 0, 0, half), (b, n, s, half))
    rotated = etl.concatenate([etl.negate(x2), x1], axis=3)
    cos4 = etl.reshape(cos, (1, 1, s, hd))
    sin4 = etl.reshape(sin, (1, 1, s, hd))
    return etl.add(etl.multiply(x, cos4), etl.multiply(rotated, sin4))


# ---------------------------------------------------------------------------
# head layout helpers
# ---------------------------------------------------------------------------
def split_heads(x, n_heads):
    """``(b, s, n_heads * head_dim) -> (b, n_heads, s, head_dim)``.

    The projection output's last axis is split into ``n_heads`` head blocks and
    moved in front of the sequence axis — the layout :func:`apply_rope` and
    :func:`_common.attention` expect.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(b, s, fused)`` — static
            dims only.
        n_heads: int number of heads; ``fused`` must be divisible by it.

    Returns:
        ``SymbolicTensor`` of shape ``(b, n_heads, s, fused // n_heads)``, dtype
        preserved.

    Raises:
        ValueError: ``n_heads`` is not a positive integer.
        HFUnsupportedError: a dim of ``x`` is symbolic, or ``fused`` is not
            divisible by ``n_heads``.
    """
    n_heads = _positive_int(n_heads, "split_heads(n_heads)")
    b = _static_dim(x, 0, "split_heads")
    s = _static_dim(x, 1, "split_heads")
    fused = _static_dim(x, 2, "split_heads")
    if fused % n_heads:
        raise HFUnsupportedError(
            f"split_heads: the fused feature dim {fused} is not divisible by "
            f"n_heads={n_heads} (check the config's num_attention_heads against the "
            "projection weight shape)."
        )
    head_dim = fused // n_heads
    return etl.transpose(etl.reshape(x, (b, s, n_heads, head_dim)), (0, 2, 1, 3))


def merge_heads(x):
    """``(b, n_heads, s, head_dim) -> (b, s, n_heads * head_dim)``.

    Exact inverse of :func:`split_heads`; the fused output is what the output
    projection (``attn.o`` / ``attn.proj``) expects.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(b, n, s, hd)`` — static
            dims only.

    Returns:
        ``SymbolicTensor`` of shape ``(b, s, n * hd)``, dtype preserved.

    Raises:
        HFUnsupportedError: a dim of ``x`` is symbolic.
    """
    b = _static_dim(x, 0, "merge_heads")
    n = _static_dim(x, 1, "merge_heads")
    s = _static_dim(x, 2, "merge_heads")
    hd = _static_dim(x, 3, "merge_heads")
    return etl.reshape(etl.transpose(x, (0, 2, 1, 3)), (b, s, n * hd))


def repeat_kv(x, n_rep):
    """GQA/MQA expansion: ``(b, n_kv, s, hd) -> (b, n_kv * n_rep, s, hd)``.

    Mirrors HF's ``repeat_kv``: key/value head ``i`` is made available to the
    ``n_rep`` consecutive query heads ``i * n_rep ... i * n_rep + n_rep - 1``
    (i.e. ``np.repeat(x, n_rep, axis=1)``), never interleaved. Implemented as an
    insert-axis + ``broadcast`` + ``reshape`` view, so no op copies per repetition.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(b, n_kv, s, hd)`` —
            static dims only.
        n_rep: int repetitions; ``1`` is the no-GQA identity (returns ``x``).

    Returns:
        ``SymbolicTensor`` of shape ``(b, n_kv * n_rep, s, hd)``, dtype preserved.

    Raises:
        ValueError: ``n_rep`` is not a positive integer.
        HFUnsupportedError: a dim of ``x`` is symbolic.
    """
    if not isinstance(n_rep, (int, np.integer)) or isinstance(n_rep, bool) or int(n_rep) <= 0:
        raise ValueError(
            f"repeat_kv: n_rep must be a positive integer, got {n_rep!r} "
            "(num_attention_heads // num_key_value_heads must be >= 1)."
        )
    n_rep = int(n_rep)
    b = _static_dim(x, 0, "repeat_kv")
    n_kv = _static_dim(x, 1, "repeat_kv")
    s = _static_dim(x, 2, "repeat_kv")
    hd = _static_dim(x, 3, "repeat_kv")
    if n_rep == 1:
        return x
    expanded = etl.reshape(x, (b, n_kv, 1, s, hd))
    expanded = etl.broadcast(expanded, (b, n_kv, n_rep, s, hd))
    return etl.reshape(expanded, (b, n_kv * n_rep, s, hd))
