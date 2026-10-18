"""Graph-vs-numpy parity tests for :mod:`etl_hf.ops` and :mod:`etl_hf.models._common`.

Every case builds the real graph (``SymbolicTensor`` compositions) and runs it on
the numpy interpreter backend through ``etl.evaluate``, then compares against an
**independently written numpy expectation computed here** — no etl_hf reference
module is imported, so these tests pin the graph semantics rather than pin a
reference implementation against itself.

Float comparisons use float64 expectations cast against float32 graph output
(``atol``/``rtol`` per case). Bit-exactness is not asserted for the transcendental
compositions (exp/tanh/erf/sin/cos) — the point is the *semantics* (which axis, which
variance convention, which rotate_half layout), and the export test pins that the
idiom set is StableHLO-expressible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import etl
from etl.backends import stablehlo
from etl_hf.errors import HFUnsupportedError
from etl_hf.models._common import attention, gated_mlp, linear, two_layer_mlp
from etl_hf.ops import (
    ACTIVATIONS,
    EXPORT_UNSAFE_ACTIVATIONS,
    MASK_NEG,
    activation_is_export_safe,
    apply_rope,
    causal_mask,
    gelu_erf,
    gelu_tanh,
    layer_norm,
    merge_heads,
    repeat_kv,
    resolve_activation,
    rms_norm,
    rope_cos_sin,
    rope_tables,
    silu,
    softmax,
    split_heads,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# helpers: run a graph builder on concrete arrays, plus self-contained references
# ---------------------------------------------------------------------------
def _eval(builder, *arrays):
    """Trace ``builder`` over ``arrays`` and run it on the numpy backend; return numpy."""
    result = etl.evaluate(builder, *arrays)
    if isinstance(result, tuple):
        return tuple(leaf.numpy() for leaf in result)
    return result.numpy()


def _f32(x):
    """Return ``x`` as a fresh float32 array (the dtype every helper preserves)."""
    return np.asarray(x, dtype=np.float32)


def _ref_softmax(x, axis=-1):
    """Max-shifted softmax in float64, the definition both numpy and the graph follow."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _ref_layer_norm(x, weight, bias, eps):
    """HF ``nn.LayerNorm`` with the *biased* (divide-by-N) variance."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(-1, keepdims=True)
    return centered / np.sqrt(var + eps) * np.asarray(weight, np.float64) + np.asarray(bias, np.float64)


def _ref_rms_norm(x, weight, eps):
    """HF ``LlamaRMSNorm``: no mean subtraction, no bias."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(weight, np.float64) * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _ref_silu(x):
    """``x * sigmoid(x)`` in float64."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def _ref_gelu_tanh(x):
    """``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`` in float64."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _ref_gelu_erf(x):
    """Exact erf-GELU in float64, using ``math.erf`` elementwise."""
    x = np.asarray(x, dtype=np.float64)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _ref_rope_tables(seq_len, head_dim, theta):
    """HF ``rotate_half`` cos/sin tables computed independently in float64."""
    inv_freq = np.asarray(
        [theta ** (-2.0 * i / head_dim) for i in range(head_dim // 2)], dtype=np.float64
    )
    pos = np.arange(seq_len, dtype=np.float64)
    freqs = pos[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=1)
    return np.cos(emb), np.sin(emb)


def _ref_apply_rope(x, cos, sin):
    """HF ``rotate_half`` rotation: ``x*cos + concat([-x2, x1])*sin``."""
    x = np.asarray(x, dtype=np.float64)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * np.asarray(cos, np.float64)[None, None] + rotated * np.asarray(sin, np.float64)[None, None]


def _ref_attention(q, k, v, mask, scale):
    """Per-head python-loop attention: ``softmax(q kᵀ scale + mask) v``, heads concatenated."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    mask = None if mask is None else np.asarray(mask, np.float64)
    heads = []
    for h in range(q.shape[1]):
        scores = q[:, h] @ k[:, h].transpose(0, 2, 1) * scale
        if mask is not None:
            # a (s, s) mask broadcasts over batch/heads; a (b, heads, s, s) mask is sliced
            scores = scores + (mask if mask.ndim == 2 else mask[:, h])
        heads.append(_ref_softmax(scores) @ v[:, h])
    return np.concatenate(heads, axis=-1)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def test_softmax_normalises_last_axis():
    """``softmax`` (axis=-1) matches the numpy definition row-wise and preserves shape/dtype."""
    x = _f32(RNG.normal(size=(3, 5)) * 4.0)
    out = _eval(lambda a: softmax(a), x)
    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, _ref_softmax(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.sum(-1), np.ones(3), rtol=1e-6)


def test_softmax_axis_zero_normalises_columns():
    """``axis=0`` normalises down the first axis (columns sum to one)."""
    x = _f32(RNG.normal(size=(4, 3)))
    out = _eval(lambda a: softmax(a, axis=0), x)
    np.testing.assert_allclose(out, _ref_softmax(x, axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.sum(0), np.ones(3), rtol=1e-6)


def test_softmax_three_dimensional_last_axis():
    """Rank-3 inputs normalise over the last axis only (the attention case)."""
    x = _f32(RNG.normal(size=(2, 3, 6)))
    out = _eval(lambda a: softmax(a), x)
    np.testing.assert_allclose(out, _ref_softmax(x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "row",
    [
        [1e30, -1e30, 0.0],  # max-subtraction keeps exp() finite
        [1e20, 1e20, 1e20],  # all-equal huge values: uniform, not inf/inf = nan
        [-1e30, -1e30, -1e30],
    ],
)
def test_softmax_is_stable_for_large_magnitudes(row):
    """The max shift makes huge/equal-huge rows exact instead of nan."""
    x = _f32([row])
    out = _eval(lambda a: softmax(a), x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, _ref_softmax(x), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# layer_norm / rms_norm
# ---------------------------------------------------------------------------
def test_layer_norm_two_dimensional():
    """``layer_norm`` matches HF nn.LayerNorm on 2-D input (biased variance, affine)."""
    x = _f32(RNG.normal(size=(4, 6)) * 3.0 + 1.0)
    w = _f32(RNG.normal(size=(6,)))
    b = _f32(RNG.normal(size=(6,)))
    out = _eval(lambda a, ww, bb: layer_norm(a, ww, bb, 1e-5), x, w, b)
    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, _ref_layer_norm(x, w, b, 1e-5), rtol=1e-5, atol=1e-5)


def test_layer_norm_three_dimensional_uses_biased_variance():
    """3-D input normalises per last-axis row, with N (not N-1) in the variance."""
    x = _f32(RNG.normal(size=(2, 3, 8)))
    w = _f32(RNG.normal(size=(8,)))
    b = _f32(RNG.normal(size=(8,)))
    out = _eval(lambda a, ww, bb: layer_norm(a, ww, bb, 1e-5), x, w, b)
    np.testing.assert_allclose(out, _ref_layer_norm(x, w, b, 1e-5), rtol=1e-5, atol=1e-5)
    # Guard the variance convention: the normalised (pre-affine) rows must have unit
    # biased variance, and must NOT match a ddof=1 (sample) variance normalisation.
    normalised = (out - b) / w
    np.testing.assert_allclose(normalised.var(-1), np.ones((2, 3)), rtol=1e-4, atol=1e-4)
    ddof1 = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, ddof=1, keepdims=True) + 1e-5)
    assert np.max(np.abs(normalised - ddof1)) > 1e-3


def test_rms_norm_two_dimensional():
    """``rms_norm`` matches LlamaRMSNorm on 2-D input (mean square, no mean subtraction)."""
    x = _f32(RNG.normal(size=(5, 4)) * 2.0)
    w = _f32(RNG.normal(size=(4,)))
    out = _eval(lambda a, ww: rms_norm(a, ww, 1e-6), x, w)
    assert out.dtype == np.float32 and out.shape == x.shape
    np.testing.assert_allclose(out, _ref_rms_norm(x, w, 1e-6), rtol=1e-5, atol=1e-6)


def test_rms_norm_three_dimensional():
    """``rms_norm`` on 3-D input scales each last-axis row to unit RMS (times weight)."""
    x = _f32(RNG.normal(size=(2, 4, 6)))
    w = np.ones((6,), np.float32)
    out = _eval(lambda a, ww: rms_norm(a, ww, 1e-6), x, w)
    np.testing.assert_allclose(out, _ref_rms_norm(x, w, 1e-6), rtol=1e-5, atol=1e-6)
    rms = np.sqrt((out.astype(np.float64) ** 2).mean(-1))
    np.testing.assert_allclose(rms, np.ones((2, 4)), rtol=1e-5)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def test_silu_matches_numpy():
    """``silu`` is ``x * sigmoid(x)``."""
    x = _f32(RNG.normal(size=(3, 7)) * 3.0)
    out = _eval(silu, x)
    assert out.dtype == np.float32 and out.shape == x.shape
    np.testing.assert_allclose(out, _ref_silu(x), rtol=1e-6, atol=1e-6)


def test_gelu_tanh_matches_numpy():
    """``gelu_tanh`` is the ``gelu_new`` tanh approximation (the GPT-2 activation)."""
    x = _f32(RNG.normal(size=(3, 7)) * 2.0)
    out = _eval(gelu_tanh, x)
    np.testing.assert_allclose(out, _ref_gelu_tanh(x), rtol=1e-6, atol=1e-6)


def test_gelu_erf_matches_float64_math_erf():
    """``gelu_erf`` reproduces exact erf-GELU (numpy backend; erf is an export deferral)."""
    x = _f32(np.linspace(-4.0, 4.0, 21))
    out = _eval(gelu_erf, x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, _ref_gelu_erf(x), rtol=1e-6, atol=1e-6)
    # at x=0 both GELUs are exactly 0; far out they track x / 0
    assert abs(float(out[x.tolist().index(0.0)])) < 1e-7


def test_activations_registry_contents():
    """``ACTIVATIONS`` exposes every HF spelling, with aliases sharing one builder."""
    assert set(ACTIVATIONS) == {
        "silu",
        "swish",
        "gelu_new",
        "gelu_pytorch_tanh",
        "gelu",
        "erf",
        "relu",
    }
    assert ACTIVATIONS["silu"] is ACTIVATIONS["swish"] is silu
    assert ACTIVATIONS["gelu_new"] is ACTIVATIONS["gelu_pytorch_tanh"] is gelu_tanh
    assert ACTIVATIONS["gelu"] is ACTIVATIONS["erf"] is gelu_erf
    assert EXPORT_UNSAFE_ACTIVATIONS == frozenset({"gelu", "erf", "gelu_erf"})


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_resolve_activation_every_supported_name(name):
    """``resolve_activation`` returns the registered builder for every supported name."""
    assert resolve_activation(name) is ACTIVATIONS[name]


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_activation_builder_matches_numpy_for_every_name(name):
    """Each registered activation's graph agrees with its numpy formula (sanity per alias)."""
    x = _f32([[-2.5, -0.5, 0.0, 0.5, 2.5]])
    builder = resolve_activation(name)
    out = _eval(builder, x)
    if builder is silu:
        expected = _ref_silu(x)
    elif builder is gelu_tanh:
        expected = _ref_gelu_tanh(x)
    elif builder is gelu_erf:
        expected = _ref_gelu_erf(x)
    else:  # relu
        expected = np.maximum(x.astype(np.float64), 0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_resolve_activation_unknown_name_lists_supported_names():
    """An unknown activation raises ``HFUnsupportedError`` naming the value and the options."""
    with pytest.raises(HFUnsupportedError) as excinfo:
        resolve_activation("gelu_fast")
    message = str(excinfo.value)
    assert "gelu_fast" in message
    for supported in sorted(ACTIVATIONS):
        assert supported in message
    assert isinstance(excinfo.value, etl.core.ETLError)


def test_activation_is_export_safe_flags_only_erf_names():
    """Only the erf-based names are export-unsafe; aliases map accordingly."""
    assert activation_is_export_safe("gelu_new") is True
    assert activation_is_export_safe("gelu_pytorch_tanh") is True
    assert activation_is_export_safe("silu") is True
    assert activation_is_export_safe("relu") is True
    assert activation_is_export_safe("gelu") is False
    assert activation_is_export_safe("erf") is False
    with pytest.raises(HFUnsupportedError):
        activation_is_export_safe("not-an-activation")


# ---------------------------------------------------------------------------
# causal mask
# ---------------------------------------------------------------------------
def test_causal_mask_exact_pattern():
    """The 4x4 mask is exactly 0 on/below the diagonal and MASK_NEG above it."""
    out = _eval(lambda: causal_mask(4))
    expected = np.array(
        [
            [0.0, MASK_NEG, MASK_NEG, MASK_NEG],
            [0.0, 0.0, MASK_NEG, MASK_NEG],
            [0.0, 0.0, 0.0, MASK_NEG],
            [0.0, 0.0, 0.0, 0.0],
        ],
        np.float32,
    )
    assert out.shape == (4, 4) and out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)
    # the kept entries are exactly +0.0 (no signed-zero surprise in emitted programs)
    assert not np.signbit(out[np.tril_indices(4)]).any()


@pytest.mark.parametrize("seq_len", [1, 2, 7])
def test_causal_mask_shape_and_layout(seq_len):
    """Any positive ``seq_len`` yields an ``(s, s)`` mask with the strict upper triangle masked."""
    out = _eval(lambda: causal_mask(seq_len))
    assert out.shape == (seq_len, seq_len)
    keep = np.tril(np.ones((seq_len, seq_len), bool))
    np.testing.assert_array_equal(out[keep], np.zeros(int(keep.sum()), np.float32))
    np.testing.assert_array_equal(out[~keep], np.full(int((~keep).sum()), MASK_NEG, np.float32))


def test_causal_mask_masks_softmax_to_exact_zero():
    """Adding the mask before softmax zeroes attention to future positions exactly."""
    scores = _f32(np.zeros((4, 4)))
    masked = _eval(lambda s: etl.add(s, causal_mask(4)), scores)
    probs = _ref_softmax(masked)
    np.testing.assert_array_equal(probs[np.triu_indices(4, 1)], np.zeros(6))


@pytest.mark.parametrize("seq_len", [0, -3])
def test_causal_mask_rejects_non_positive_seq_len(seq_len):
    """``causal_mask`` validates its length (host-side argument error => ValueError)."""
    with pytest.raises(ValueError):
        causal_mask(seq_len)


# ---------------------------------------------------------------------------
# rope tables + apply_rope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len,head_dim,theta", [(5, 8, 10000.0), (1, 4, 10000.0), (7, 2, 500000.0)])
def test_rope_cos_sin_matches_host_tables(seq_len, head_dim, theta):
    """In-graph ``rope_cos_sin`` and host ``rope_tables`` produce the same numbers."""
    cos_g, sin_g = _eval(lambda: rope_cos_sin(seq_len, head_dim, theta))
    cos_h, sin_h = rope_tables(seq_len, head_dim, theta)
    assert cos_g.shape == cos_h.shape == (seq_len, head_dim)
    assert cos_g.dtype == cos_h.dtype == np.float32
    np.testing.assert_allclose(cos_g, cos_h, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(sin_g, sin_h, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("seq_len,head_dim,theta", [(4, 6, 10000.0), (6, 16, 500000.0)])
def test_rope_tables_match_float64_definition(seq_len, head_dim, theta):
    """Host tables follow ``theta ** (-2i/head_dim)`` duplicated over the rotate_half halves."""
    cos_h, sin_h = rope_tables(seq_len, head_dim, theta)
    cos_ref, sin_ref = _ref_rope_tables(seq_len, head_dim, theta)
    np.testing.assert_allclose(cos_h, cos_ref, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(sin_h, sin_ref, rtol=0.0, atol=1e-6)
    # the two halves of the table are identical (HF rotate_half layout), and position 0
    # is the identity rotation (cos=1, sin=0).
    half = head_dim // 2
    np.testing.assert_array_equal(cos_h[:, :half], cos_h[:, half:])
    np.testing.assert_array_equal(sin_h[:, :half], sin_h[:, half:])
    np.testing.assert_allclose(cos_h[0], np.ones(head_dim), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(sin_h[0], np.zeros(head_dim), rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("seq_len,head_dim,theta", [(0, 4, 10000.0), (4, 0, 10000.0), (4, -2, 10000.0)])
def test_rope_rejects_non_positive_sizes(seq_len, head_dim, theta):
    """Non-positive lengths are rejected before any IR is built."""
    with pytest.raises(ValueError):
        rope_cos_sin(seq_len, head_dim, theta)
    with pytest.raises(ValueError):
        rope_tables(seq_len, head_dim, theta)


def test_rope_rejects_odd_head_dim():
    """An odd ``head_dim`` has no HF rotate_half meaning — explicit rejection."""
    with pytest.raises(HFUnsupportedError):
        rope_cos_sin(4, 5, 10000.0)
    with pytest.raises(HFUnsupportedError):
        rope_tables(4, 5, 10000.0)


@pytest.mark.parametrize("theta", [0.0, -1.0])
def test_rope_rejects_non_positive_theta(theta):
    """``theta`` must be a positive base (a zero/negative base is not a rotation)."""
    with pytest.raises(ValueError):
        rope_tables(4, 4, theta)


def test_apply_rope_position_zero_is_identity():
    """At position 0 the rotation is the identity (cos=1, sin=0)."""
    s, hd = 4, 6
    x = _f32(RNG.normal(size=(2, 3, s, hd)))
    cos, sin = rope_tables(s, hd, 10000.0)
    out = _eval(lambda a, c, sn: apply_rope(a, c, sn), x, cos, sin)
    assert out.dtype == np.float32 and out.shape == x.shape
    np.testing.assert_allclose(out[:, :, 0, :], x[:, :, 0, :], rtol=0.0, atol=1e-6)


def test_apply_rope_matches_rotate_half_reference():
    """``apply_rope`` implements HF ``x*cos + rotate_half(x)*sin`` exactly."""
    s, hd = 5, 8
    x = _f32(RNG.normal(size=(2, 3, s, hd)))
    cos, sin = rope_tables(s, hd, 10000.0)
    out = _eval(lambda a, c, sn: apply_rope(a, c, sn), x, cos, sin)
    np.testing.assert_allclose(out, _ref_apply_rope(x, cos, sin), rtol=1e-6, atol=1e-6)


def test_apply_rope_preserves_head_norms():
    """A rotation preserves each (position, head) vector's L2 norm."""
    s, hd = 4, 8
    x = _f32(RNG.normal(size=(1, 2, s, hd)))
    cos, sin = rope_tables(s, hd, 10000.0)
    out = _eval(lambda a, c, sn: apply_rope(a, c, sn), x, cos, sin)
    np.testing.assert_allclose(
        np.linalg.norm(out.astype(np.float64), axis=-1),
        np.linalg.norm(x.astype(np.float64), axis=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_apply_rope_rejects_mismatched_tables():
    """A cos/sin table whose shape is not ``(seq, head_dim)`` is an explicit error."""
    x = _f32(RNG.normal(size=(1, 2, 4, 8)))
    cos, sin = rope_tables(4, 4, 10000.0)  # wrong head_dim
    with pytest.raises(HFUnsupportedError):
        _eval(lambda a, c, sn: apply_rope(a, c, sn), x, cos, sin)


def test_apply_rope_rejects_odd_head_dim():
    """``apply_rope`` needs paired halves; an odd head_dim is rejected."""
    cos, sin = rope_tables(3, 4, 10000.0)
    with pytest.raises(HFUnsupportedError):
        _eval(lambda a, c, sn: apply_rope(a, c, sn), _f32(RNG.normal(size=(1, 2, 3, 5))), cos, sin)


# ---------------------------------------------------------------------------
# head layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch,seq,n_heads,head_dim", [(2, 5, 3, 4), (1, 3, 2, 1), (2, 4, 6, 2)])
def test_split_heads_matches_reshape_transpose(batch, seq, n_heads, head_dim):
    """``split_heads`` equals ``reshape(b, s, n, hd).transpose(0, 2, 1, 3)``."""
    x = _f32(RNG.normal(size=(batch, seq, n_heads * head_dim)))
    out = _eval(lambda a: split_heads(a, n_heads), x)
    assert out.shape == (batch, n_heads, seq, head_dim) and out.dtype == np.float32
    expected = x.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
    np.testing.assert_array_equal(out, np.ascontiguousarray(expected))


@pytest.mark.parametrize("batch,seq,n_heads,head_dim", [(2, 5, 3, 4), (1, 2, 1, 8)])
def test_merge_heads_inverts_split_heads(batch, seq, n_heads, head_dim):
    """``merge_heads(split_heads(x)) == x`` (both directions round-trip)."""
    x = _f32(RNG.normal(size=(batch, seq, n_heads * head_dim)))
    out = _eval(lambda a: merge_heads(split_heads(a, n_heads)), x)
    assert out.shape == x.shape
    np.testing.assert_array_equal(out, x)
    heads = _f32(RNG.normal(size=(batch, n_heads, seq, head_dim)))
    back = _eval(lambda a: split_heads(merge_heads(a), n_heads), heads)
    np.testing.assert_array_equal(back, heads)


def test_merge_heads_matches_numpy():
    """``merge_heads`` equals ``transpose(0, 2, 1, 3).reshape(b, s, n*hd)``."""
    batch, n_heads, seq, head_dim = 2, 3, 4, 2
    x = _f32(RNG.normal(size=(batch, n_heads, seq, head_dim)))
    out = _eval(merge_heads, x)
    expected = x.transpose(0, 2, 1, 3).reshape(batch, seq, n_heads * head_dim)
    np.testing.assert_array_equal(out, expected)


def test_split_heads_rejects_indivisible_features():
    """A fused feature count that n_heads does not divide is an explicit error."""
    with pytest.raises(HFUnsupportedError):
        _eval(lambda a: split_heads(a, 3), _f32(RNG.normal(size=(2, 4, 8))))


@pytest.mark.parametrize("n_heads", [0, -2])
def test_split_heads_rejects_non_positive_head_count(n_heads):
    """``n_heads`` must be a positive integer."""
    with pytest.raises(ValueError):
        split_heads(_f32(RNG.normal(size=(2, 4, 8))), n_heads)


# ---------------------------------------------------------------------------
# repeat_kv (GQA / MQA)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_rep", [1, 2, 3])
def test_repeat_kv_matches_np_repeat(n_rep):
    """``repeat_kv`` repeats each kv head consecutively, exactly like ``np.repeat``."""
    x = _f32(RNG.normal(size=(2, 3, 4, 5)))
    out = _eval(lambda a: repeat_kv(a, n_rep), x)
    assert out.shape == (2, 3 * n_rep, 4, 5) and out.dtype == np.float32
    np.testing.assert_array_equal(out, np.repeat(x, n_rep, axis=1))


@pytest.mark.parametrize("n_rep", [0, -1])
def test_repeat_kv_rejects_non_positive_repetition(n_rep):
    """``n_rep <= 0`` is a host-side argument error (ValueError), never a silent identity."""
    x = _f32(RNG.normal(size=(1, 2, 3, 4)))
    with pytest.raises(ValueError):
        repeat_kv(x, n_rep)


# ---------------------------------------------------------------------------
# static-shape guards
# ---------------------------------------------------------------------------
def test_apply_rope_rejects_symbolic_batch_dim():
    """A symbolic dim cannot supply a reshape/slice extent — explicit HFUnsupportedError."""
    spec = etl.TensorSpec((etl.dim("b"), 2, 4, 4), etl.float32)

    def graph(x):
        cos, sin = rope_cos_sin(4, 4, 10000.0)
        return apply_rope(x, cos, sin)

    with pytest.raises(HFUnsupportedError) as excinfo:
        etl.trace(graph, spec)
    assert "axis 0" in str(excinfo.value)
    assert "symbolic" in str(excinfo.value)


def test_split_heads_rejects_symbolic_batch_dim():
    """``split_heads`` enforces the same static-shape policy on the batch axis."""
    spec = etl.TensorSpec((etl.dim("b"), 4, 8), etl.float32)
    with pytest.raises(HFUnsupportedError) as excinfo:
        etl.trace(lambda x: split_heads(x, 2), spec)
    assert "axis 0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# etl_hf.models._common
# ---------------------------------------------------------------------------
def test_linear_without_bias():
    """``linear`` is ``dot(x, weight)`` with ``(in, out)`` weights."""
    x = _f32(RNG.normal(size=(2, 3, 6)))
    w = _f32(RNG.normal(size=(6, 9)))
    out = _eval(lambda a, ww: linear(a, ww), x, w)
    assert out.shape == (2, 3, 9) and out.dtype == np.float32
    np.testing.assert_allclose(out, x @ w, rtol=1e-5, atol=1e-6)


def test_linear_with_bias():
    """``linear`` adds the per-output bias after the projection."""
    x = _f32(RNG.normal(size=(4, 6)))
    w = _f32(RNG.normal(size=(6, 5)))
    b = _f32(RNG.normal(size=(5,)))
    out = _eval(lambda a, ww, bb: linear(a, ww, bb), x, w, b)
    np.testing.assert_allclose(out, x @ w + b, rtol=1e-5, atol=1e-6)


def test_attention_matches_per_head_reference():
    """``attention`` fuses heads to ``(b, s, n*hd)`` and matches a per-head numpy loop."""
    batch, heads, seq, hd = 2, 3, 4, 5
    q = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    k = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    v = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    mask = _eval(lambda: causal_mask(seq))
    scale = hd**-0.5
    out = _eval(lambda qq, kk, vv, mm: attention(qq, kk, vv, mm, scale), q, k, v, mask)
    assert out.shape == (batch, seq, heads * hd) and out.dtype == np.float32
    np.testing.assert_allclose(out, _ref_attention(q, k, v, mask, scale), rtol=1e-5, atol=1e-6)


def test_attention_without_mask_is_bidirectional():
    """``mask=None`` attends everywhere (no causal bias)."""
    batch, heads, seq, hd = 1, 2, 3, 4
    q = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    k = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    v = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    out = _eval(lambda qq, kk, vv: attention(qq, kk, vv, None, 1.0), q, k, v)
    np.testing.assert_allclose(out, _ref_attention(q, k, v, None, 1.0), rtol=1e-5, atol=1e-6)


def test_attention_applies_scale():
    """The ``scale`` argument multiplies the raw scores (head_dim ** -0.5 by convention)."""
    batch, heads, seq, hd = 1, 1, 2, 2
    q = _f32(np.ones((batch, heads, seq, hd)))
    k = _f32(np.ones((batch, heads, seq, hd)))
    v = _f32(RNG.normal(size=(batch, heads, seq, hd)))
    equal = _eval(lambda qq, kk, vv: attention(qq, kk, vv, None, 0.0), q, k, v)
    # scale 0 => uniform attention => the mean of v over positions.
    np.testing.assert_allclose(equal, np.broadcast_to(v.mean(axis=2), equal.shape), rtol=1e-6, atol=1e-6)


def test_attention_rejects_mismatched_kv_heads():
    """GQAs must be expanded explicitly — a head-count mismatch is an explicit error."""
    q = _f32(RNG.normal(size=(1, 2, 3, 4)))
    kv = _f32(RNG.normal(size=(1, 4, 3, 4)))
    with pytest.raises(HFUnsupportedError) as excinfo:
        _eval(lambda qq, kk: attention(qq, kk, kk, None, 1.0), q, kv)
    assert "repeat_kv" in str(excinfo.value)


def test_attention_rejects_non_rank_four_operands():
    """Projections must be split into heads first (rank-4 operands only)."""
    x = _f32(RNG.normal(size=(2, 3, 8)))
    with pytest.raises(HFUnsupportedError):
        _eval(lambda a: attention(a, a, a, None, 1.0), x)


def test_gated_mlp_matches_numpy():
    """``gated_mlp`` is SwiGLU: ``dot(act(dot(x, w_gate)) * dot(x, w_up), w_down)``."""
    x = _f32(RNG.normal(size=(2, 3, 6)))
    w_gate = _f32(RNG.normal(size=(6, 10)))
    w_up = _f32(RNG.normal(size=(6, 10)))
    w_down = _f32(RNG.normal(size=(10, 6)))
    out = _eval(lambda a, g, u, d: gated_mlp(a, g, u, d, silu), x, w_gate, w_up, w_down)
    expected = (_ref_silu(x @ w_gate) * (x @ w_up)) @ w_down
    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_two_layer_mlp_matches_numpy():
    """``two_layer_mlp`` is GPT-2's ``dot(act(dot(x,w1)+b1),w2)+b2``."""
    x = _f32(RNG.normal(size=(2, 3, 6)))
    w1 = _f32(RNG.normal(size=(6, 10)))
    b1 = _f32(RNG.normal(size=(10,)))
    w2 = _f32(RNG.normal(size=(10, 6)))
    b2 = _f32(RNG.normal(size=(6,)))
    out = _eval(lambda a, p, q, r, s: two_layer_mlp(a, p, q, r, s, gelu_tanh), x, w1, b1, w2, b2)
    expected = _ref_gelu_tanh(x @ w1 + b1) @ w2 + b2
    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_two_layer_mlp_allows_missing_biases():
    """Both biases are optional (``None`` => bias-free projections)."""
    x = _f32(RNG.normal(size=(4, 6)))
    w1 = _f32(RNG.normal(size=(6, 10)))
    w2 = _f32(RNG.normal(size=(10, 6)))
    out = _eval(lambda a, p, r: two_layer_mlp(a, p, None, r, None, gelu_tanh), x, w1, w2)
    np.testing.assert_allclose(out, _ref_gelu_tanh(x @ w1) @ w2, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# whole-block assembly (proves both families compose these helpers)
# ---------------------------------------------------------------------------
def test_llama_style_block_assembles_from_common_helpers():
    """A Llama-shaped block (RMSNorm + RoPE + GQA attention + SwiGLU) built only from helpers."""
    batch, seq, hidden, n_heads, n_kv, inter, hd = 1, 4, 8, 2, 1, 6, 4
    scale, eps = hd**-0.5, 1e-6
    x = _f32(RNG.normal(size=(batch, seq, hidden)))
    ln = _f32(np.abs(RNG.normal(size=(hidden,))) + 0.5)
    wq = _f32(RNG.normal(size=(hidden, n_heads * hd)))
    wk = _f32(RNG.normal(size=(hidden, n_kv * hd)))
    wv = _f32(RNG.normal(size=(hidden, n_kv * hd)))
    wo = _f32(RNG.normal(size=(n_heads * hd, hidden)))
    w_gate = _f32(RNG.normal(size=(hidden, inter)))
    w_up = _f32(RNG.normal(size=(hidden, inter)))
    w_down = _f32(RNG.normal(size=(inter, hidden)))
    n_rep = n_heads // n_kv

    def block(a, ln_w, q_w, k_w, v_w, o_w, gate_w, up_w, down_w):
        h = rms_norm(a, ln_w, eps)
        cos, sin = rope_cos_sin(seq, hd, 10000.0)
        q = apply_rope(split_heads(linear(h, q_w), n_heads), cos, sin)
        k = apply_rope(repeat_kv(split_heads(linear(h, k_w), n_kv), n_rep), cos, sin)
        v = repeat_kv(split_heads(linear(h, v_w), n_kv), n_rep)
        attended = etl.add(a, linear(attention(q, k, v, causal_mask(seq), scale), o_w))
        return etl.add(attended, gated_mlp(attended, gate_w, up_w, down_w, silu))

    out = _eval(block, x, ln, wq, wk, wv, wo, w_gate, w_up, w_down)

    cos_h, sin_h = _ref_rope_tables(seq, hd, 10000.0)
    h = _ref_rms_norm(x, ln, eps)
    q = _ref_apply_rope((h @ wq).reshape(batch, seq, n_heads, hd).transpose(0, 2, 1, 3), cos_h, sin_h)

    def kv_heads(w):
        heads = (h @ w).reshape(batch, seq, n_kv, hd).transpose(0, 2, 1, 3)
        return np.repeat(heads, n_rep, axis=1)

    k = _ref_apply_rope(kv_heads(wk), cos_h, sin_h)
    v = kv_heads(wv)
    mask = np.where(np.tril(np.ones((seq, seq), bool)), 0.0, MASK_NEG)
    attended = x + _ref_attention(q, k, v, mask, scale) @ wo
    expected = attended + (_ref_silu(attended @ w_gate) * (attended @ w_up)) @ w_down

    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def test_gpt2_style_block_assembles_from_common_helpers():
    """A GPT-2-shaped block (LayerNorm + fused qkv + attention + two-layer MLP) from helpers."""
    batch, seq, hidden, n_head, inter, hd = 1, 4, 8, 2, 6, 4
    scale, eps = hd**-0.5, 1e-5
    x = _f32(RNG.normal(size=(batch, seq, hidden)))
    ln1_w, ln1_b = _f32(np.abs(RNG.normal(size=(hidden,))) + 0.5), _f32(RNG.normal(size=(hidden,)))
    w_qkv = _f32(RNG.normal(size=(hidden, 3 * hidden)))
    b_qkv = _f32(RNG.normal(size=(3 * hidden,)))
    w_proj = _f32(RNG.normal(size=(hidden, hidden)))
    b_proj = _f32(RNG.normal(size=(hidden,)))
    ln2_w, ln2_b = _f32(np.abs(RNG.normal(size=(hidden,))) + 0.5), _f32(RNG.normal(size=(hidden,)))
    w1 = _f32(RNG.normal(size=(hidden, inter)))
    b1 = _f32(RNG.normal(size=(inter,)))
    w2 = _f32(RNG.normal(size=(inter, hidden)))
    b2 = _f32(RNG.normal(size=(hidden,)))

    def qkv_part(qkv, offset, n_heads):
        part = etl.slice(qkv, (0, 0, offset), (batch, seq, hidden))
        return split_heads(part, n_heads)

    def block(a, l1w, l1b, qkv_w, qkv_b, proj_w, proj_b, l2w, l2b, f1_w, f1_b, f2_w, f2_b):
        h = layer_norm(a, l1w, l1b, eps)
        qkv = linear(h, qkv_w, qkv_b)
        q = qkv_part(qkv, 0, n_head)
        k = qkv_part(qkv, hidden, n_head)
        v = qkv_part(qkv, 2 * hidden, n_head)
        attended = etl.add(a, linear(attention(q, k, v, causal_mask(seq), scale), proj_w, proj_b))
        return etl.add(
            attended,
            two_layer_mlp(layer_norm(attended, l2w, l2b, eps), f1_w, f1_b, f2_w, f2_b, gelu_tanh),
        )

    out = _eval(block, x, ln1_w, ln1_b, w_qkv, b_qkv, w_proj, b_proj, ln2_w, ln2_b, w1, b1, w2, b2)

    h = _ref_layer_norm(x, ln1_w, ln1_b, eps)
    qkv = h @ w_qkv + b_qkv
    heads = [
        qkv[:, :, offset : offset + hidden].reshape(batch, seq, n_head, hd).transpose(0, 2, 1, 3)
        for offset in (0, hidden, 2 * hidden)
    ]
    mask = np.where(np.tril(np.ones((seq, seq), bool)), 0.0, MASK_NEG)
    attended = x + _ref_attention(*heads, mask, scale) @ w_proj + b_proj
    hidden_state = _ref_layer_norm(attended, ln2_w, ln2_b, eps)
    expected = attended + _ref_gelu_tanh(hidden_state @ w1 + b1) @ w2 + b2

    assert out.shape == x.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# StableHLO exportability (the reason these idioms live here)
# ---------------------------------------------------------------------------
def test_transformer_idioms_export_to_stablehlo():
    """A graph combining every non-erf idiom exports to StableHLO text."""
    specs = (
        etl.TensorSpec((2, 4, 6), etl.float32),  # activations
        etl.TensorSpec((6,), etl.float32),  # norm weight
        etl.TensorSpec((6,), etl.float32),  # norm bias
        etl.TensorSpec((6, 8), etl.float32),  # qkv-ish projection
        etl.TensorSpec((4, 4), etl.float32),  # extra additive mask operand
    )

    def graph(a, ln_w, ln_b, proj_w, extra_mask):
        y = rms_norm(layer_norm(a, ln_w, ln_b, 1e-5), ln_w, 1e-6)
        y = etl.multiply(y, etl.add(silu(y), gelu_tanh(y)))
        cos, sin = rope_cos_sin(4, 4, 10000.0)
        heads = split_heads(etl.dot(y, proj_w), 2)
        kv = repeat_kv(heads, 2)
        mask = etl.add(etl.add(extra_mask, causal_mask(4)), etl.multiply(softmax(extra_mask), 0.0))
        attended = attention(kv, kv, kv, mask, 0.5)
        rotated = merge_heads(apply_rope(kv, cos, sin))
        return etl.add(attended, rotated)

    text = stablehlo.export(etl.trace(graph, *specs))
    for marker in (
        "stablehlo.cosine",
        "stablehlo.sine",
        "stablehlo.logistic",  # silu
        "stablehlo.tanh",  # gelu_tanh
        "stablehlo.power",  # rope inverse frequencies
        "stablehlo.compare",  # causal mask
        "stablehlo.slice",  # rotate_half
        "stablehlo.exponential",  # softmax
    ):
        assert marker in text, f"expected {marker} in the exported StableHLO"
    assert "erf" not in text


def test_gelu_erf_export_defers_explicitly():
    """Exact erf-GELU is a documented v1 export deferral: explicit BackendError, never a fallback."""
    spec = etl.TensorSpec((2, 4), etl.float32)
    graph = etl.trace(gelu_erf, spec)
    with pytest.raises(etl.BackendError) as excinfo:
        stablehlo.export(graph)
    assert "erf" in str(excinfo.value)
    # ... while the numpy interpreter backend still runs it (documented numpy-only path).
    out = _eval(gelu_erf, _f32([[-1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(out, _ref_gelu_erf(np.array([[-1.0, 0.0, 1.0]])), rtol=1e-6, atol=1e-6)
