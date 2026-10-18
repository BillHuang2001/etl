"""Brute-force cross-checks for :mod:`etl_hf.reference` (the parity oracle).

Every expectation here is written out **independently** of the helper under
test: explicit mean/variance formulas, the closed-form activation expressions,
hand-computed rotations and — for attention — a third, triple-loop
implementation with no matmul and no shared softmax helper.  The tolerances are
tight (f32 rounding noise only): this file proves the *oracle* is right, so the
graph-vs-oracle parity tests downstream inherit that trust.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import etl_hf.reference as ref
from etl_hf.errors import HFUnsupportedError

# f32 rounding budget for a handful of intermediate roundings.
F32_ATOL = 1e-6
F32_RTOL = 1e-6


# ---------------------------------------------------------------------------
# independent expectation helpers (float64 / loop formulations)
# ---------------------------------------------------------------------------
def _mean_last(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1, keepdims=True)


def _layer_norm_f64(x, weight, bias, eps):
    """Explicit HF nn.LayerNorm formula in float64 (biased variance)."""
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(weight, dtype=np.float64)
    b64 = np.asarray(bias, dtype=np.float64)
    mean = _mean_last(x64)
    centered = x64 - mean
    var = _mean_last(centered * centered)
    return ((centered / np.sqrt(var + np.float64(eps))) * w64) + b64


def _rms_norm_f64(x, weight, eps):
    """Explicit HF LlamaRMSNorm formula in float64."""
    x64 = np.asarray(x, dtype=np.float64)
    w64 = np.asarray(weight, dtype=np.float64)
    mean_square = _mean_last(x64 * x64)
    return (x64 * (1.0 / np.sqrt(mean_square + np.float64(eps)))) * w64


def _erf_f64(x):
    return np.vectorize(math.erf, otypes=[np.float64])(np.asarray(x, dtype=np.float64))


def _gelu_erf_f64(x):
    x64 = np.asarray(x, dtype=np.float64)
    return 0.5 * x64 * (1.0 + _erf_f64(x64 / math.sqrt(2.0)))


def _gelu_tanh_f64(x):
    x64 = np.asarray(x, dtype=np.float64)
    return 0.5 * x64 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x64 + 0.044715 * x64**3)))


def _silu_f64(x):
    x64 = np.asarray(x, dtype=np.float64)
    return x64 / (1.0 + np.exp(-x64))


def _brute_force_attention(q, k, v, mask, scale):
    """Triple-loop attention in float64: scalar products only, no matmul.

    A genuinely different formulation from both the graph (one batched ``dot``
    chain) and :func:`etl_hf.reference.attention` (per-head 2-D matmuls), so an
    indexing/layout bug cannot be shared.
    """
    batch, n_q, seq, head_dim = q.shape
    n_kv = k.shape[1]
    repeat = n_q // n_kv
    out = np.zeros((batch, seq, n_q * head_dim), dtype=np.float64)
    for b in range(batch):
        for h in range(n_q):
            kv_head = h // repeat
            for i in range(seq):
                scores = [
                    sum(
                        float(q[b, h, i, d]) * float(k[b, kv_head, j, d])
                        for d in range(head_dim)
                    )
                    for j in range(seq)
                ]
                logits = [score * scale + float(mask[i, j]) for j, score in enumerate(scores)]
                peak = max(logits)
                exps = [math.exp(value - peak) for value in logits]
                total = sum(exps)
                for d in range(head_dim):
                    out[b, i, h * head_dim + d] = sum(
                        exps[j] / total * float(v[b, kv_head, j, d]) for j in range(seq)
                    )
    return out


# ---------------------------------------------------------------------------
# module-level contract
# ---------------------------------------------------------------------------
def test_mask_neg_constant_underflows_exp():
    assert ref.MASK_NEG == -1e4
    # The documented invariant: the masked additive term underflows to exactly 0.
    assert np.exp(np.float32(ref.MASK_NEG - 1.0)) == 0.0
    assert np.exp(np.float32(-104.0)) == 0.0


def test_reference_does_not_import_etl_graph_code():
    """The oracle must not share code with the implementation it validates."""
    source = ref.__file__
    with open(source, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    import_lines = [
        line.strip()
        for line in lines
        if line.lstrip().startswith(("import etl", "from etl"))
    ]
    assert import_lines == ["from etl_hf.errors import HFUnsupportedError"]
    assert not hasattr(ref, "etl")


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def test_softmax_matches_explicit_formula():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, 5)) * 2.0).astype(np.float32)
    e = np.exp(np.asarray(x, dtype=np.float64))
    expected = e / e.sum(axis=-1, keepdims=True)
    actual = ref.softmax(x)
    assert actual.dtype == np.float32
    assert_allclose(actual, expected.astype(np.float32), rtol=F32_RTOL, atol=F32_ATOL)
    # Every row sums to one.
    assert_allclose(actual.sum(axis=-1), np.ones(3, np.float32), rtol=F32_RTOL)


def test_softmax_1d_input():
    x = np.array([1.0, 2.0, 3.0], np.float32)
    expected = np.exp(x.astype(np.float64)) / np.exp(x.astype(np.float64)).sum()
    assert_allclose(ref.softmax(x), expected.astype(np.float32), rtol=F32_RTOL, atol=F32_ATOL)


def test_softmax_axis0_matches_transposed_expectation():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    along_axis0 = ref.softmax(x, axis=0)
    assert along_axis0.shape == x.shape
    assert_allclose(along_axis0, ref.softmax(x.T, axis=-1).T, rtol=F32_RTOL, atol=F32_ATOL)
    # Column sums are one (axis=0 normalisation).
    assert_allclose(along_axis0.sum(axis=0), np.ones(3, np.float32), rtol=F32_RTOL)


def test_softmax_large_magnitudes_do_not_overflow():
    x = np.array([[1000.0, 1000.0, 1000.0, -1000.0], [500.0, 0.0, -500.0, 100.0]], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any numpy overflow would fail here
        actual = ref.softmax(x)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual.sum(axis=-1), np.ones(2, np.float32), rtol=F32_RTOL)
    assert_allclose(actual[0, :3], np.full(3, 1.0 / 3.0, np.float32), rtol=F32_RTOL)
    assert actual[0, 3] == 0.0


def test_softmax_rejects_0d_scalar():
    with pytest.raises(ValueError, match="at least one dimension"):
        ref.softmax(np.float32(1.0))


# ---------------------------------------------------------------------------
# layer_norm / rms_norm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(7, 16), (2, 5, 16), (2, 3, 4, 16)])
def test_layer_norm_matches_float64_formula(shape):
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(shape) * 3.0).astype(np.float32)
    channels = shape[-1]
    weight = rng.standard_normal(channels).astype(np.float32)
    bias = rng.standard_normal(channels).astype(np.float32)
    eps = 1e-5
    actual = ref.layer_norm(x, weight, bias, eps)
    expected = _layer_norm_f64(x, weight, bias, eps).astype(np.float32)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    assert_allclose(actual, expected, rtol=F32_RTOL, atol=1e-5)


def test_layer_norm_constant_row_has_zero_mean_and_unit_variance():
    x = np.full((2, 8), 5.0, np.float32)
    weight = np.zeros(8, np.float32)
    bias = np.ones(8, np.float32)
    # Centered row is exactly 0 -> normalised is 0 -> output is the bias.
    assert_array_equal(ref.layer_norm(x, weight, bias, 1e-5), np.ones((2, 8), np.float32))


def test_layer_norm_weight_shape_is_validated():
    x = np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError, match=r"layer_norm: weight must have shape \(4,\)"):
        ref.layer_norm(x, np.zeros((1, 4), np.float32), np.zeros(4, np.float32), 1e-5)
    with pytest.raises(ValueError, match=r"layer_norm: bias must have shape \(4,\)"):
        ref.layer_norm(x, np.zeros(4, np.float32), np.zeros(5, np.float32), 1e-5)


def test_layer_norm_rejects_negative_eps():
    x = np.zeros((2, 4), np.float32)
    with pytest.raises(ValueError, match="eps must be >= 0"):
        ref.layer_norm(x, np.zeros(4, np.float32), np.zeros(4, np.float32), -1e-5)


@pytest.mark.parametrize("shape", [(7, 16), (2, 5, 16), (2, 3, 4, 16)])
def test_rms_norm_matches_float64_formula(shape):
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    weight = rng.standard_normal(shape[-1]).astype(np.float32)
    actual = ref.rms_norm(x, weight, 1e-6)
    expected = _rms_norm_f64(x, weight, 1e-6).astype(np.float32)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    assert_allclose(actual, expected, rtol=F32_RTOL, atol=1e-5)


def test_rms_norm_does_not_subtract_the_mean():
    # A constant positive row: rms == the constant, so x * (1/rms) == 1 -> weight.
    x = np.full((3, 4), 2.0, np.float32)
    weight = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    assert_allclose(ref.rms_norm(x, weight, 0.0), np.tile(weight, (3, 1)), rtol=F32_RTOL)


def test_rms_norm_weight_shape_is_validated():
    x = np.zeros((2, 4), np.float32)
    with pytest.raises(ValueError, match=r"rms_norm: weight must have shape \(4,\)"):
        ref.rms_norm(x, np.zeros(3, np.float32), 1e-6)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def test_silu_matches_explicit_formula():
    x = np.linspace(-6.0, 6.0, 25).astype(np.float32)
    actual = ref.silu(x)
    assert actual.dtype == np.float32
    assert_allclose(actual, _silu_f64(x).astype(np.float32), rtol=F32_RTOL, atol=F32_ATOL)


def test_silu_is_stable_for_extreme_inputs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # exp(-x) overflow would surface here
        actual = ref.silu(np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], np.float32))
    assert np.all(np.isfinite(actual))
    assert actual[0] == 0.0  # -0.0 compares equal to 0.0
    assert actual[2] == 0.0
    assert actual[4] == 1000.0


def test_gelu_tanh_matches_explicit_formula():
    x = np.linspace(-4.0, 4.0, 33).astype(np.float32)
    actual = ref.gelu_tanh(x)
    assert actual.dtype == np.float32
    assert_allclose(actual, _gelu_tanh_f64(x).astype(np.float32), rtol=1e-5, atol=F32_ATOL)


def test_gelu_erf_matches_float64_erf_formula():
    x = np.linspace(-3.0, 3.0, 25).astype(np.float32)
    actual = ref.gelu_erf(x)
    expected = _gelu_erf_f64(x).astype(np.float32)
    assert actual.dtype == np.float32
    assert_allclose(actual, expected, rtol=F32_RTOL, atol=F32_ATOL)


def test_gelu_erf_known_values():
    assert float(ref.gelu_erf(0.0)) == 0.0
    assert abs(float(ref.gelu_erf(1.0)) - 0.8413447) < 1e-6
    assert abs(float(ref.gelu_erf(-1.0)) - (-0.1586553)) < 1e-6
    # Symmetry reference: gelu(x) - gelu(-x) == x for the exact erf form.
    x = np.linspace(-2.0, 2.0, 9).astype(np.float32)
    assert_allclose(ref.gelu_erf(x) - ref.gelu_erf(-x), x, rtol=F32_RTOL, atol=F32_ATOL)


def test_gelu_erf_and_gelu_tanh_agree_within_1e3():
    x = np.linspace(-3.0, 3.0, 2001).astype(np.float32)
    max_diff = float(np.max(np.abs(ref.gelu_erf(x) - ref.gelu_tanh(x))))
    assert max_diff < 1e-3, f"tanh-GELU strays {max_diff} from exact GELU on [-3, 3]"


def test_relu():
    x = np.array([-2.0, -0.0, 0.0, 3.0], np.float32)
    assert_array_equal(ref.relu(x), np.array([0.0, 0.0, 0.0, 3.0], np.float32))


def test_activations_table_contents():
    assert set(ref.ACTIVATIONS) == {
        "silu",
        "swish",
        "gelu_new",
        "gelu_pytorch_tanh",
        "gelu",
        "erf",
        "relu",
    }
    assert ref.ACTIVATIONS["silu"] is ref.silu
    assert ref.ACTIVATIONS["swish"] is ref.silu
    assert ref.ACTIVATIONS["gelu_new"] is ref.gelu_tanh
    assert ref.ACTIVATIONS["gelu_pytorch_tanh"] is ref.gelu_tanh
    assert ref.ACTIVATIONS["gelu"] is ref.gelu_erf
    assert ref.ACTIVATIONS["erf"] is ref.gelu_erf
    assert ref.ACTIVATIONS["relu"] is ref.relu
    x = np.array([-1.5, 0.0, 0.5, 2.0], np.float32)
    for name, fn in ref.ACTIVATIONS.items():
        out = fn(x)
        assert out.shape == x.shape
        assert out.dtype == np.float32, f"{name} did not return float32"


def test_resolve_activation_returns_registered_callables():
    for name in ref.ACTIVATIONS:
        assert ref.resolve_activation(name) is ref.ACTIVATIONS[name]


@pytest.mark.parametrize("name", ["gelu_fast", "SILU", "silu ", None, 3])
def test_resolve_activation_rejects_unknown_names(name):
    with pytest.raises(HFUnsupportedError) as info:
        ref.resolve_activation(name)
    message = str(info.value)
    assert repr(name) in message
    assert str(sorted(ref.ACTIVATIONS)) in message


# ---------------------------------------------------------------------------
# causal mask
# ---------------------------------------------------------------------------
def test_additive_causal_mask_exact_pattern():
    mask = ref.additive_causal_mask(4)
    expected = np.array(
        [
            [0.0, ref.MASK_NEG, ref.MASK_NEG, ref.MASK_NEG],
            [0.0, 0.0, ref.MASK_NEG, ref.MASK_NEG],
            [0.0, 0.0, 0.0, ref.MASK_NEG],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert mask.dtype == np.float32
    assert_array_equal(mask, expected)
    assert_set = {np.float32(0.0), np.float32(ref.MASK_NEG)}
    assert set(np.unique(mask)) <= assert_set


def test_additive_causal_mask_row0_attends_only_to_itself():
    mask = ref.additive_causal_mask(4)
    assert int(np.count_nonzero(mask[0] == 0.0)) == 1
    assert mask[0, 0] == 0.0
    # The last row may attend to the whole prefix.
    assert int(np.count_nonzero(mask[-1] == 0.0)) == 4


def test_additive_causal_mask_empty_and_validation():
    empty = ref.additive_causal_mask(0)
    assert empty.shape == (0, 0) and empty.dtype == np.float32
    with pytest.raises(ValueError, match="must be >= 0"):
        ref.additive_causal_mask(-1)
    with pytest.raises(TypeError, match="must be an integer"):
        ref.additive_causal_mask(2.5)


def test_additive_causal_mask_makes_future_weights_exactly_zero():
    scores = np.array([[3.0, 999.0, 999.0], [3.0, 4.0, 999.0], [3.0, 4.0, 5.0]], np.float32)
    weights = ref.softmax(scores + ref.additive_causal_mask(3))
    assert weights[0, 1] == 0.0 and weights[0, 2] == 0.0
    assert weights[1, 2] == 0.0
    assert np.all(weights[np.tril_indices(3)] > 0.0)


# ---------------------------------------------------------------------------
# RoPE tables
# ---------------------------------------------------------------------------
def test_rope_tables_known_small_case():
    seq, head_dim, theta = 6, 4, 10000.0
    cos, sin = ref.rope_tables(seq, head_dim, theta)
    assert cos.shape == (seq, head_dim) and sin.shape == (seq, head_dim)
    assert cos.dtype == np.float32 and sin.dtype == np.float32
    # Position 0: cos == 1, sin == 0 (exactly).
    assert_array_equal(cos[0], np.ones(head_dim, np.float32))
    assert_array_equal(sin[0], np.zeros(head_dim, np.float32))
    # cos**2 + sin**2 == 1 elementwise.
    assert_allclose(cos**2 + sin**2, np.ones_like(cos), rtol=F32_RTOL, atol=1e-6)
    # The second half of each row duplicates the first half.
    assert_array_equal(cos[:, : head_dim // 2], cos[:, head_dim // 2 :])
    assert_array_equal(sin[:, : head_dim // 2], sin[:, head_dim // 2 :])


def test_rope_tables_inv_freq_values():
    seq, head_dim, theta = 4, 4, 10000.0
    cos, _ = ref.rope_tables(seq, head_dim, theta)
    positions = np.arange(seq, dtype=np.float32)
    # Pair 0 has inv_freq 1.0; pair 1 has theta ** (-1/head_dim) = 0.01.
    assert_array_equal(cos[:, 0], np.cos(positions))
    assert_allclose(cos[:, 1], np.cos(positions * np.float32(10000.0 ** -0.5)), rtol=F32_RTOL, atol=1e-6)


def test_rope_tables_depend_on_theta():
    cos_default, _ = ref.rope_tables(3, 8, theta=10000.0)
    cos_wide, _ = ref.rope_tables(3, 8, theta=500000.0)
    assert not np.allclose(cos_default, cos_wide)


def test_rope_tables_validation():
    with pytest.raises(ValueError, match="positive even integer"):
        ref.rope_tables(3, 5, 10000.0)
    with pytest.raises(ValueError, match="positive even integer"):
        ref.rope_tables(3, 0, 10000.0)
    with pytest.raises(ValueError, match="seq_len must be >= 0"):
        ref.rope_tables(-1, 4, 10000.0)
    with pytest.raises(ValueError, match="theta must be > 0"):
        ref.rope_tables(3, 4, 0.0)


# ---------------------------------------------------------------------------
# apply_rope
# ---------------------------------------------------------------------------
def test_apply_rope_is_identity_at_position_zero():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    cos, sin = ref.rope_tables(5, 6, 10000.0)
    out = ref.apply_rope(x, cos, sin)
    assert out.dtype == np.float32 and out.shape == x.shape
    assert_array_equal(out[..., 0, :], x[..., 0, :])
    assert not np.array_equal(out[..., 3, :], x[..., 3, :])


def test_apply_rope_identity_when_tables_are_trivial():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    cos = np.ones((4, 4), np.float32)
    sin = np.zeros((4, 4), np.float32)
    assert_array_equal(ref.apply_rope(x, cos, sin), x)


def test_apply_rope_hand_computed_rotation():
    x = np.array([[[3.0, 4.0]]], np.float32)  # (batch=1, heads=1, seq=1, head_dim=2)
    angle = 0.5
    cos = np.full((1, 2), math.cos(angle), np.float32)
    sin = np.full((1, 2), math.sin(angle), np.float32)
    actual = ref.apply_rope(x, cos, sin)
    expected = np.array(
        [[[3.0 * math.cos(angle) - 4.0 * math.sin(angle),
           4.0 * math.cos(angle) + 3.0 * math.sin(angle)]]],
        np.float32,
    )
    assert_allclose(actual, expected, rtol=F32_RTOL, atol=F32_ATOL)


def test_apply_rope_preserves_per_position_norms():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 3, 7, 8)).astype(np.float32)
    cos, sin = ref.rope_tables(7, 8, 12345.0)
    out = ref.apply_rope(x, cos, sin)
    before = np.linalg.norm(x.astype(np.float64), axis=-1)
    after = np.linalg.norm(out.astype(np.float64), axis=-1)
    assert_allclose(after, before, rtol=1e-6, atol=1e-6)
    # The first half / second half split does not mix pairs across halves.
    assert_array_equal(cos[:, :4], cos[:, 4:])


def test_apply_rope_accepts_3d_heads_first_and_matches_4d_batch_of_one():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 5, 4)).astype(np.float32)  # (heads, seq, head_dim)
    cos, sin = ref.rope_tables(5, 4, 10000.0)
    assert_allclose(
        ref.apply_rope(x, cos, sin), ref.apply_rope(x[None], cos, sin)[0], rtol=F32_RTOL
    )


def test_apply_rope_shape_validation():
    x = np.zeros((1, 2, 3, 4), np.float32)
    cos, sin = ref.rope_tables(3, 4, 10000.0)
    with pytest.raises(ValueError, match=r"apply_rope: x must be \(heads, seq, head_dim\)"):
        ref.apply_rope(np.zeros((3, 4), np.float32), cos, sin)
    with pytest.raises(ValueError, match="does not match the cos/sin table shape"):
        ref.apply_rope(np.zeros((1, 2, 4, 4), np.float32), cos, sin)
    with pytest.raises(ValueError, match="does not match the cos/sin table shape"):
        ref.apply_rope(np.zeros((1, 2, 3, 6), np.float32), cos, sin)
    with pytest.raises(ValueError, match="equal-shaped 2-D"):
        ref.apply_rope(x, cos, sin[:, :2])
    odd_cos = np.ones((3, 3), np.float32)
    with pytest.raises(ValueError, match="head_dim must be even"):
        ref.apply_rope(np.zeros((1, 2, 3, 3), np.float32), odd_cos, odd_cos)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def _tiny_attention_inputs(seed=0, batch=2, n_heads=2, seq=3, head_dim=2):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float32)
    k = rng.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float32)
    v = rng.standard_normal((batch, n_heads, seq, head_dim)).astype(np.float32)
    return q, k, v


def test_attention_matches_brute_force_triple_loop():
    q, k, v = _tiny_attention_inputs(batch=2, n_heads=2, seq=3, head_dim=2)
    mask = ref.additive_causal_mask(3)
    scale = 2 ** -0.5
    actual = ref.attention(q, k, v, mask, scale)
    expected = _brute_force_attention(
        q.astype(np.float64), k.astype(np.float64), v.astype(np.float64),
        mask.astype(np.float64), scale,
    )
    assert actual.shape == (2, 3, 4)
    assert actual.dtype == np.float32
    assert_allclose(actual, expected.astype(np.float32), rtol=0.0, atol=1e-6)


def test_attention_matches_softmax_expectation():
    q, k, v = _tiny_attention_inputs(batch=1, n_heads=1, seq=4, head_dim=2)
    mask = ref.additive_causal_mask(4)
    scale = 2 ** -0.5
    actual = ref.attention(q, k, v, mask, scale)
    # Independent expectation via the helper's own (already-tested) softmax.
    scores = (q[0, 0].astype(np.float64) @ k[0, 0].astype(np.float64).T) * scale + mask
    expected = ref.softmax(scores.astype(np.float32)) @ v[0, 0]
    assert_allclose(actual[0], expected, rtol=F32_RTOL, atol=F32_ATOL)


def test_attention_masked_positions_contribute_exactly_zero():
    q, k, v = _tiny_attention_inputs(batch=2, n_heads=2, seq=4, head_dim=2)
    mask = ref.additive_causal_mask(4)
    scale = 2 ** -0.5
    baseline = ref.attention(q, k, v, mask, scale)

    perturbed_v = v.copy()
    perturbed_v[:, :, -1, :] += 5.0  # only the strictly-future key of rows 0..2
    perturbed = ref.attention(q, k, perturbed_v, mask, scale)

    # Rows 0..2 cannot see the last key: their context is bit-identical.
    assert_array_equal(baseline[:, :-1, :], perturbed[:, :-1, :])
    # The last row does see it, so it must change.
    assert not np.array_equal(baseline[:, -1, :], perturbed[:, -1, :])


def test_attention_accepts_kv_gqa_by_repeating_heads():
    rng = np.random.default_rng(8)
    q = rng.standard_normal((1, 4, 3, 2)).astype(np.float32)
    k = rng.standard_normal((1, 2, 3, 2)).astype(np.float32)
    v = rng.standard_normal((1, 2, 3, 2)).astype(np.float32)
    mask = ref.additive_causal_mask(3)
    scale = 2 ** -0.5
    gqa = ref.attention(q, k, v, mask, scale)
    repeated = ref.attention(q, np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1), mask, scale)
    assert_array_equal(gqa, repeated)
    # And it matches the brute-force GQA formulation.
    expected = _brute_force_attention(
        q.astype(np.float64), k.astype(np.float64), v.astype(np.float64),
        mask.astype(np.float64), scale,
    )
    assert_allclose(gqa, expected.astype(np.float32), rtol=0.0, atol=1e-6)


def test_attention_rejects_non_divisible_head_counts():
    q = np.zeros((1, 3, 2, 2), np.float32)
    k = np.zeros((1, 2, 2, 2), np.float32)
    with pytest.raises(ValueError, match="must be a multiple of the number of key/value heads"):
        ref.attention(q, k, k, ref.additive_causal_mask(2), 1.0)


def test_attention_validates_shapes():
    q, k, v = _tiny_attention_inputs()
    mask = ref.additive_causal_mask(3)
    with pytest.raises(ValueError, match="attention: q must be"):
        ref.attention(q[0], k[0], v[0], mask, 1.0)
    with pytest.raises(ValueError, match="attention: k must have shape"):
        ref.attention(q, k[:, :, :2, :], v[:, :, :2, :], mask, 1.0)
    with pytest.raises(ValueError, match="attention: v must have shape"):
        ref.attention(q, k, v[:, :, :, :1], mask, 1.0)
    with pytest.raises(ValueError, match="not broadcastable"):
        ref.attention(q, k, v, ref.additive_causal_mask(5), 1.0)


def test_attention_mask_broadcasts_from_4d_hf_layout():
    q, k, v = _tiny_attention_inputs(batch=2)
    base = ref.additive_causal_mask(3)
    from_4d = ref.attention(q, k, v, base[None, None, :, :], 2 ** -0.5)
    assert_array_equal(from_4d, ref.attention(q, k, v, base, 2 ** -0.5))


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------
def test_param_count_over_nested_tree():
    tree = {
        "embed": np.zeros((10, 4), np.float32),
        "layers": [
            {"w": np.zeros((4, 4), np.float32), "b": np.zeros(4, np.float32)},
            (np.zeros(2, np.float32), np.zeros((3, 3, 3), np.float32)),
        ],
        "scale": np.float32(1.5),  # numpy scalar leaf counts as one element
        "empty": {},
    }
    assert ref.param_count(tree) == 40 + 16 + 4 + 2 + 27 + 1 + 0
    assert ref.param_count([]) == 0
    assert ref.param_count(np.zeros((), np.float32)) == 1


def test_param_count_rejects_unsupported_leaves():
    with pytest.raises(TypeError, match=r"params\['layers'\]\[0\]\['w'\]"):
        ref.param_count({"layers": [{"w": 1.0}]})


# ---------------------------------------------------------------------------
# dtype policy
# ---------------------------------------------------------------------------
def test_float64_inputs_are_downcast_to_float32():
    rng = np.random.default_rng(9)
    x64 = rng.standard_normal((2, 3, 4))
    weight64 = rng.standard_normal(4)
    pairs = [
        (ref.softmax(x64), ref.softmax(x64.astype(np.float32))),
        (
            ref.layer_norm(x64, weight64, weight64, 1e-5),
            ref.layer_norm(x64.astype(np.float32), weight64.astype(np.float32), weight64.astype(np.float32), 1e-5),
        ),
        (
            ref.rms_norm(x64, weight64, 1e-6),
            ref.rms_norm(x64.astype(np.float32), weight64.astype(np.float32), 1e-6),
        ),
        (ref.silu(x64), ref.silu(x64.astype(np.float32))),
        (ref.gelu_tanh(x64), ref.gelu_tanh(x64.astype(np.float32))),
        (ref.gelu_erf(x64), ref.gelu_erf(x64.astype(np.float32))),
        (ref.relu(x64), ref.relu(x64.astype(np.float32))),
    ]
    for actual, expected in pairs:
        assert actual.dtype == np.float32
        assert_array_equal(actual, expected)


def test_public_outputs_are_float32():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    weight = rng.standard_normal(4).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    cos, sin = ref.rope_tables(3, 4, 10000.0)
    q = rng.standard_normal((1, 2, 3, 4)).astype(np.float32)
    outputs = {
        "softmax": ref.softmax(x),
        "layer_norm": ref.layer_norm(x, weight, bias, 1e-5),
        "rms_norm": ref.rms_norm(x, weight, 1e-6),
        "silu": ref.silu(x),
        "gelu_tanh": ref.gelu_tanh(x),
        "gelu_erf": ref.gelu_erf(x),
        "relu": ref.relu(x),
        "additive_causal_mask": ref.additive_causal_mask(3),
        "rope_cos": cos,
        "rope_sin": sin,
        "apply_rope": ref.apply_rope(q, cos, sin),
        "attention": ref.attention(q, q, q, ref.additive_causal_mask(3), 2 ** -0.5),
    }
    for name, value in outputs.items():
        assert isinstance(value, np.ndarray), name
        assert value.dtype == np.float32, f"{name} returned {value.dtype}"


def test_zero_dimensional_scalar_helpers_return_float32():
    for fn in (ref.silu, ref.gelu_tanh, ref.gelu_erf, ref.relu):
        out = fn(0.5)
        assert out.shape == ()
        assert out.dtype == np.float32


def test_empty_inputs_are_supported():
    empty = np.zeros((0, 4), np.float32)
    assert ref.softmax(empty).shape == (0, 4)
    assert ref.gelu_erf(empty).shape == (0, 4)
    assert ref.gelu_tanh(empty).shape == (0, 4)
    weight = np.ones(4, np.float32)
    assert ref.rms_norm(empty, weight, 1e-6).shape == (0, 4)
    assert ref.layer_norm(empty, weight, weight, 1e-5).shape == (0, 4)
