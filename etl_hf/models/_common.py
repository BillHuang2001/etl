"""Shared transformer block assembly for the ``etl_hf`` model families.

This module is *assembly*, not new math: four helpers built from
``etl_hf.ops`` compositions and public ``etl`` ops, used verbatim by both family
graphs (``llama`` / ``gpt2``) so neither family module duplicates attention or
MLP wiring. Anything numerical lives in :mod:`etl_hf.ops`.

All helpers are ``SymbolicTensor`` graph builders (no eager path): concrete
tensors are a user error and surface as the underlying ``etl.TraceError``.
Shapes are static-only, matching the package contract; the helpers that need an
extent validate it through :mod:`etl_hf.ops` internals and raise
``etl_hf.errors.HFUnsupportedError`` naming the offending axis rather than
guessing a dimension.
"""

from __future__ import annotations

import etl

from etl_hf.errors import HFUnsupportedError
from etl_hf.ops import merge_heads, softmax

__all__ = ["attention", "gated_mlp", "linear", "two_layer_mlp"]


def linear(x, weight, bias=None):
    """Apply a linear projection: ``dot(x, weight) [+ bias]``.

    Weights are stored ``(in_features, out_features)`` throughout ``etl_hf`` (the
    loader transposes HF's ``(out, in)`` linears, and GPT-2's ``Conv1D`` weights
    are already ``(in, out)``), so the weight is the second ``etl.dot`` operand
    with no transpose here. ``etl.dot`` does not auto-cast mixed dtypes — pass
    weights of the same dtype as the activations.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(..., in_features)``.
        weight: graph input of shape ``(in_features, out_features)``.
        bias: optional graph input of shape ``(out_features,)``; ``None`` means a
            bias-free projection (Llama-style).

    Returns:
        ``SymbolicTensor`` of shape ``(..., out_features)``, dtype of ``x``.
    """
    projected = etl.dot(x, weight)
    if bias is None:
        return projected
    return etl.add(projected, bias)


def attention(q, k, v, mask, scale):
    """Multi-head scaled dot-product attention (causal by default via ``mask``).

    ``scores = dot(q, transpose(k, (0, 1, 3, 2))) * scale + mask`` →
    ``softmax(scores)`` → ``dot(probs, v)`` → :func:`etl_hf.ops.merge_heads`.

    Grouped-query attention is handled *outside* this helper: call
    :func:`etl_hf.ops.repeat_kv` on ``k``/``v`` first so all three operands have
    the same head count. The mask is a plain additive array (``(s, s)`` from
    :func:`etl_hf.ops.causal_mask`, broadcasting over batch/head axes), or
    ``None`` for unmasked (bidirectional) attention.

    Args:
        q: ``SymbolicTensor``, shape ``(b, n_heads, s, head_dim)``.
        k: ``SymbolicTensor``, shape ``(b, n_heads, s_kv, head_dim)`` — already
            expanded by ``repeat_kv`` for GQA/MQA.
        v: ``SymbolicTensor``, shape ``(b, n_heads, s_kv, head_dim)``.
        mask: additive mask broadcastable to ``(b, n_heads, s, s_kv)`` — normally
            ``(s, s)`` from :func:`etl_hf.ops.causal_mask` — or ``None``.
        scale: float score scale, conventionally ``head_dim ** -0.5``.

    Returns:
        ``SymbolicTensor`` of shape ``(b, s, n_heads * head_dim)`` — the fused
        layout the output projection consumes.

    Raises:
        HFUnsupportedError: ``q``/``k``/``v`` are not rank-4 head tensors, the
            head counts differ (forget ``repeat_kv``), ``head_dim`` differs, or
            the value sequence length differs from the key sequence length.
    """
    for name, operand in (("q", q), ("k", k), ("v", v)):
        if len(operand.shape) != 4:
            raise HFUnsupportedError(
                f"attention: {name} must be a rank-4 head tensor "
                f"(batch, heads, seq, head_dim), got shape {tuple(operand.shape)}; "
                "use etl_hf.ops.split_heads on the projection output first."
            )
    if tuple(k.shape) != tuple(v.shape):
        raise HFUnsupportedError(
            f"attention: k has shape {tuple(k.shape)} but v has shape "
            f"{tuple(v.shape)}; they must be identical (repeat_kv both)."
        )
    if tuple(q.shape[:2]) != tuple(k.shape[:2]) or q.shape[3] != k.shape[3]:
        raise HFUnsupportedError(
            f"attention: incompatible q {tuple(q.shape)} / k {tuple(k.shape)}; expected "
            "the same (batch, heads) and head_dim — repeating kv heads is the "
            "caller's explicit step (etl_hf.ops.repeat_kv)."
        )
    scores = etl.multiply(etl.dot(q, etl.transpose(k, (0, 1, 3, 2))), scale)
    if mask is not None:
        scores = etl.add(scores, mask)
    return merge_heads(etl.dot(softmax(scores), v))


def gated_mlp(x, w_gate, w_up, w_down, act):
    """SwiGLU-style gated MLP (Llama/Mistral): ``dot(act(dot(x, w_gate)) * dot(x, w_up), w_down)``.

    The gate branch passes through ``act``; the up branch is linear; the elementwise
    product then projects back down. Bias-free (these families have no MLP bias).

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(..., hidden)``.
        w_gate: ``(hidden, intermediate)`` — gate projection weight.
        w_up: ``(hidden, intermediate)`` — up projection weight.
        w_down: ``(intermediate, hidden)`` — down projection weight.
        act: activation builder from :func:`etl_hf.ops.resolve_activation`.

    Returns:
        ``SymbolicTensor`` of shape ``(..., hidden)``, dtype of ``x``.
    """
    gated = etl.multiply(act(etl.dot(x, w_gate)), etl.dot(x, w_up))
    return etl.dot(gated, w_down)


def two_layer_mlp(x, w1, b1, w2, b2, act):
    """Classic two-layer MLP (GPT-2): ``dot(act(dot(x, w1) + b1), w2) + b2``.

    Args:
        x: ``SymbolicTensor`` of floating dtype, shape ``(..., hidden)``.
        w1: ``(hidden, intermediate)`` — first projection weight.
        b1: ``(intermediate,)`` bias for the first projection, or ``None``.
        w2: ``(intermediate, hidden)`` — second projection weight.
        b2: ``(hidden,)`` bias for the second projection, or ``None``.
        act: activation builder from :func:`etl_hf.ops.resolve_activation`.

    Returns:
        ``SymbolicTensor`` of shape ``(..., hidden)``, dtype of ``x``.
    """
    return linear(act(linear(x, w1, b1)), w2, b2)
