"""End-to-end TRAINING examples (category "e2e", tag "train") — driven through
the optional ``Example.runner`` factory (see the runner contract in
``etl.bench._util``'s docstring).

Why a runner (binding rationale):
  v1 has NO VJP rule for control flow — ``etl.grad`` through ``cond``/
  ``while_loop``/``scan`` raises ``TransformError`` (verified at dev time), so
  a training loop with gradient descent CANNOT live inside the graph. The
  runner moves the loop to Python: per iteration it runs the etl ``grad``
  graph (the graph itself contains only forward+backward ops — no control
  flow), updates the weights in numpy (SGD), and repeats. Each example's
  ``numpy_ref`` mirrors the FULL procedure in pure numpy — the same iteration
  count, lr, update rule, and input data — with a hand-written analytic
  float64 backprop loop. The etl grads are fp32 while the reference grads are
  fp64, so the fp32 noise ACCUMULATES over the iterations; every example's
  ``tolerance`` (max-abs over ALL outputs — the final loss plus the final
  weights) is set from the measured final error with margin (numbers below).

Runner mechanics:
  ``runner(backend, device, opts)`` builds its executables ONCE (the forward
  loss ``@etl.defn`` via ``etl.build``; the ``etl.grad`` TransformCallable via
  ``grad_fn(*specs)`` → ``etl.lower`` → ``etl.compile`` → ``etl.load``) and
  returns a run-callable ``(inputs) -> outputs`` performing the whole training
  loop. It NEVER calls ``stage_example`` (infinite recursion). All initial
  weights are EXPLICIT graph inputs (so the runner and the reference start
  from identical state), and the run-callable never mutates its input arrays
  (updates allocate fresh arrays — benchmark reuses the same inputs list).

The three examples (all f32, static shapes; sizes are module-level constants
so users can bump them for real benchmarking):

1. e2e_train_mlp — SGD (lr=0.05, 10 iters) of a 2-layer MLP
   x[16,8] → relu(x@w1+b1) [16,16] → h@w2+b2 [16,2], MSE loss; trains
   w1[8,16], b1[16], w2[16,2], b2[2] (x and y are data, not trained).
   Outputs: (final loss, w1, b1, w2, b2). numpy_ref = hand-written analytic
   MLP backprop (chain rule) in fp64. Measured final max-abs error vs the
   fp64 ref: ≈4.4e-7 (see the comment at the registry entry) — tolerance=1e-3
   (~2000x margin).

2. e2e_train_convnet — SGD (lr=0.005, 10 iters) of a small conv net
   x[2,2,8,8] → conv(3x3 → 4ch) → relu → conv(3x3 → 8ch) → relu → flatten
   [2,128] → linear [2,2], MSE loss. THE CONV LAYERS ARE FROZEN (fixed
   inputs, never updated; v1 deferral — see below): the etl forward graph
   still contains the real conv ops, run once per invocation (the features
   are constant because the conv weights and x are fixed), and only the
   linear head (wl[128,2], bl[2]) trains via etl.grad. Outputs: (final loss,
   wl, bl). numpy_ref = the full loop in fp64 with the analytic head
   backprop (fd-verified at dev time). The raw conv2 features are O(10)
   (max |f| ≈ 39 at seed 0), which makes the head loss ill-conditioned
   (2/lambda_max of the Hessian ≈ 2e-4 — measured: stable only at
   lr ≲ 1e-4, diverging at lr ≥ 3e-4); a constant 0.1 feature scale in BOTH
   the etl graph and the ref (an input normalization) widens the stable band
   ~100x (2/lambda_max ≈ 2e-2), so lr=0.005 trains comfortably (measured
   loss 14.8 → 0.21 over the 10 iterations). Measured final max-abs ≈2.8e-7
   — tolerance=1e-3 (~3500x margin).

   WHY FROZEN (verified at dev time with a live ``etl.grad`` probe): v1 has
   NO VJP rule for ``conv`` — ``TransformError: grad/vjp: no VJP rule for op
   'conv' — the IR has no transposed-convolution op (v1 gap)``. The rule
   raises unconditionally (the reverse sweep invokes every op's rule even
   with a zero cotangent, ``etl/transforms/autodiff.py``), so ANY conv op in
   the differentiated graph fails ``etl.grad``; a trainable conv is a core
   v1 deferral (documented in ``etl/transforms/CONTEXT.md``), not a
   bench-level choice. This example uses the objective-sanctioned frozen-conv
   variant: the conv is fixed, only the linear part trains. The etl forward
   graph exercises the real conv kernels on every invocation.

3. e2e_train_transformer — THE FLAGSHIP: SGD (lr=0.02, 10 iters) trains TWO
   full simplified transformer blocks (MHA with 2 heads of dim 16 +
   layernorm + gelu FFN + residuals, no biases) plus the final projection
   head against a fixed dummy training batch x[2,16,32], y[2,16,2]
   (BATCH=2, SEQ=16, D_MODEL=32, FFN=64). Budget-aware sizing: one runner
   invocation ≈ 0.1 s on the numpy backend (measured; the block_transformer
   precedent at [1,512,768] runs ≈1 s for ONE forward — this config is ~30x
   smaller and runs 10 forward+backward iterations). 9 trainable tensors:
   per block w_qkv[32,96], w_out[32,32], w1[32,64], w2[64,32], plus the head
   wh[32,2]. Outputs: (final loss, all 9 final weights). The weights are
   initialized with fan-in scaling (std 1/sqrt(fan_in), xavier-style) via
   ``inputs_fn`` — REQUIRED for stability (measured, see the dev-time
   section): with unit-variance standard-normal weights the attention
   saturates (logits std ~30), the initial loss is ~67 with per-grad
   magnitudes up to ~34, and the steep-descent trajectory AMPLIFIES the
   per-step fp32 grad noise ~10000x over the 10 iterations (final max-abs
   ≈1.6e-1 — a 15% error on the final loss); with the scaled init the
   forward is O(1) end-to-end (loss ~2.6, grads ~0.2-0.9) and the loop
   tracks the fp64 reference to ≈1.5e-7. numpy_ref = the full training loop
   with hand-written numpy backward for the simplified transformer (softmax
   grad, layernorm grad, gelu grad, matmul grads — each piece verified
   against central finite differences at dev time BEFORE wiring the loop;
   numbers below). Measured final max-abs ≈1.5e-7 — tolerance=1e-3
   (~6500x margin).

Dev-time measurements (this module's tolerances are set from these, with
margin; all on the numpy backend, seed 0):
  - fd vs analytic one-step grads (fp64, h=1e-4): layernorm backward
    ≈4.3e-9; gelu derivative ≈3.0e-9; softmax backward ≈8.8e-10; MLP
    backprop ≈6.1e-11 (64-elem loss); convnet head backprop ≈2.1e-10;
    transformer block backward (small config: batch 1, seq 4, d=8, 2 heads
    of dim 4, ffn 16) ≈2.3e-12 over all five grads (x, w_qkv, w_out, w1,
    w2) — the softmax / layernorm / gelu backward pieces are exact closed
    forms.
  - one-step etl fp32 grads vs the fp64 analytic ref (real configs, before
    any update): mlp ≈4.1e-6, convnet head ≈2.4e-6, transformer ≈1.7e-7
    (relative to grad magnitudes ~0.2-34 — plain fp32 accumulation noise).
  - FULL loop (10 iters), final-output max-abs (loss + weights): mlp
    ≈4.4e-7, convnet ≈2.8e-7, transformer ≈1.5e-7 — the per-step fp32 grad
    noise accumulates roughly linearly over the iterations, as documented.

Conformance status: numpy backend PASS for all three (the default harness
path). The e2e_train_mlp/convnet graphs are StableHLO-export-clean (static
shapes; the convnet graph uses conv/reshape, the mlp graph is plain
dot/relu/add), so compiler backends (iree/xla/tvm) should build them; the
transformer graph is also export-clean (static shapes, no dynamic-dim
keepdims reshapes), but its fp32 accumulation noise on compiled backends may
exceed the numpy-measured tolerance — the documented per-example tolerance
is measured on the numpy backend. All three need the harness ``runner``
path, which builds/stages the executables itself.

NOTE: ``Example.graph`` is INFORMATIONAL for runner examples (the harness
skips the single stage+run path when ``runner`` is set — ``stage_example``
routes straight to the runner factory). For each example below it is the
loss ``@etl.defn`` (the convnet example's graph is the linear-head loss —
its specs are the head specs, not the full 6-spec input list; the full
forward lives inside the runner).
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .base import (
    Example,
    _F32,
    conv2d_im2col_numpy,
    gelu_numpy,
    layernorm_numpy,
    register_all,
)

# ---------------------------------------------------------------------------
# Shared numpy reference helpers (fp64 analytic backward pieces, fd-verified)
# ---------------------------------------------------------------------------


def _layernorm_backward(g, x, dim, eps=1e-5):
    """Backward of the sum-primitive layernorm (the graph's ``_tf_layernorm``
    / base ``layernorm_numpy`` form): mean = sum(x)/dim; y = x - mean;
    s = sqrt(mean(y²) + eps); out = y / s. Closed form (verified against
    central finite differences at dev time):
        g_y = g / s
        g_x = g_y - mean(g_y, -1) - y * mean(g * y, -1) / s³
    """
    mean = x.mean(axis=-1, keepdims=True)
    y = x - mean
    s = np.sqrt((y * y).mean(axis=-1, keepdims=True) + eps)
    g_y = g / s
    return g_y - g_y.mean(axis=-1, keepdims=True) - y * (
        (g * y).mean(axis=-1, keepdims=True) / (s * s * s)
    )


def _gelu_deriv(x):
    """d/dx of gelu(x) = 0.5·x·(1 + erf(x/√2)) (the exact etl gelu form, see
    the numpy backend kernel):
        0.5·(1 + erf(x/√2)) + x·exp(−x²/2)/√(2π).
    ``math.erf`` via ``np.frompyfunc`` — the same technique as the etl
    kernels and base's ``gelu_numpy``."""
    x = np.asarray(x)
    erf = np.frompyfunc(math.erf, 1, 1)(x / math.sqrt(2.0)).astype(x.dtype)
    return 0.5 * (1.0 + erf) + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _softmax_backward(g, probs):
    """Backward of the max-subtract softmax (probs = e / Σe, e = exp(s − m)):
    g_s = probs * (g − Σ(g·probs, axis=−1, keepdims)). The max-subtraction
    constant m has zero gradient except on a measure-zero argmax set.
    Verified against central finite differences at dev time."""
    return probs * (g - (g * probs).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# e2e_train_mlp — SGD training of a 2-layer MLP (etl.grad + numpy updates)
# ---------------------------------------------------------------------------

_MLP_ITERS = 10
_MLP_LR = 0.05
_MLP_SPECS = tuple(
    TensorSpec(s, _F32) for s in [(16, 8), (16, 2), (8, 16), (16,), (16, 2), (2,)]
)  # x, y, w1, b1, w2, b2
_MLP_ARGS = (2, 3, 4, 5)  # trained: w1, b1, w2, b2 (x and y are data)


@defn
def _mlp_loss(x, y, w1, b1, w2, b2):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    out = etl.add(etl.dot(h, w2), b2)
    diff = etl.subtract(out, y)
    return etl.mean(etl.multiply(diff, diff))


_mlp_grad = etl.grad(_mlp_loss, argnums=_MLP_ARGS)


def _mlp_forward_np(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2


def _mlp_backward_np(x, y, w1, b1, w2, b2):
    """Analytic MLP backprop (chain rule) — the fp64 reference grads.
    loss = mean((out − y)²) over N = out.size elements; d loss/d out =
    2·(out − y)/N; relu derivative = the pre-activation mask. Verified
    against central finite differences at dev time (max-abs ≈6e-11)."""
    h = np.maximum(x @ w1 + b1, 0.0)
    out = h @ w2 + b2
    g_out = 2.0 * (out - y) / out.size
    g_w2 = h.T @ g_out
    g_b2 = g_out.sum(axis=0)
    g_h = g_out @ w2.T
    g_pre = g_h * (h > 0)
    g_w1 = x.T @ g_pre
    g_b1 = g_pre.sum(axis=0)
    return g_w1, g_b1, g_w2, g_b2


def _mlp_train_numpy(inputs):
    """The FULL training procedure in pure numpy (fp64 analytic backprop):
    the same iteration count, lr, update rule, and input data as the
    runner."""
    x, y, w1, b1, w2, b2 = (a.astype(np.float64) for a in inputs)
    loss = None
    for _ in range(_MLP_ITERS):
        g_w1, g_b1, g_w2, g_b2 = _mlp_backward_np(x, y, w1, b1, w2, b2)
        w1 = w1 - _MLP_LR * g_w1
        b1 = b1 - _MLP_LR * g_b1
        w2 = w2 - _MLP_LR * g_w2
        b2 = b2 - _MLP_LR * g_b2
        diff = _mlp_forward_np(x, w1, b1, w2, b2) - y
        loss = np.mean(diff * diff)
    return (np.asarray(loss), w1, b1, w2, b2)


def _mlp_runner(backend, device, opts):
    """Runner factory: stages the loss ``@etl.defn`` (etl.build) and the
    ``etl.grad`` TransformCallable (lower/compile/load) ONCE, then returns a
    run-callable performing the 10-iteration SGD loop in Python: per
    iteration an etl grad run, an fp32 numpy SGD update, and an etl loss
    run on the updated weights (the final loss is the loss of the trained
    model — the reference mirrors this order).

    Explicit device placement (non-cpu devices): the data is placed once and
    stays device-resident; the per-step grads are read back explicitly (the
    numpy SGD update runs on the host, so they are a hard per-step data
    dependency) and the updated host weights are re-placed per step; the
    per-step loss runs do not affect the trajectory (the loss is a pure
    function of the weights), so on non-cpu devices the loss is run and read
    back ONCE after the loop — the timed loop performs no per-step host
    readback of device outputs beyond the required grads."""
    from .._util import place_input, to_host  # lazy: _util imports .examples

    loss_exe = etl.build(
        _mlp_loss, *_MLP_SPECS, backend=backend, device=device, **opts
    )
    grad_graph = _mlp_grad(*_MLP_SPECS)
    lp = etl.lower(grad_graph, backend=backend, **opts)
    ca = etl.compile(lp, **opts)
    grad_exe = etl.load(ca, device=device)

    def run(inputs):
        x, y = inputs[0], inputs[1]
        ws = list(inputs[2:])  # current weights (fresh list; inputs untouched)
        loss = None
        if device.kind == "cpu":
            # cpu/host tensors throughout — the current per-step behavior
            # (.numpy() readbacks are zero-copy host views).
            for _ in range(_MLP_ITERS):
                grads = [t.numpy() for t in etl.run(grad_exe, x, y, *ws)]
                ws = [w - _MLP_LR * g for w, g in zip(ws, grads)]
                loss = etl.run(loss_exe, x, y, *ws).numpy()
            return (loss, *ws)
        # Non-cpu device: explicit placement/readback (no implicit
        # host<->device transfers) — see the factory docstring.
        x_d, y_d = place_input(x, device), place_input(y, device)
        for _ in range(_MLP_ITERS):
            ws_d = [place_input(w, device) for w in ws]
            grads = [to_host(t).numpy() for t in etl.run(grad_exe, x_d, y_d, *ws_d)]
            ws = [w - _MLP_LR * g for w, g in zip(ws, grads)]
        loss = to_host(
            etl.run(loss_exe, x_d, y_d, *(place_input(w, device) for w in ws))
        ).numpy()
        return (loss, *ws)

    return run


# ---------------------------------------------------------------------------
# e2e_train_convnet — SGD training of a small conv net (frozen convs, v1
# conv-VJP deferral — see the module docstring): the etl forward graph
# contains the REAL conv ops (conv → relu → conv → relu → flatten, run once
# per invocation — the features are constant because the conv weights are
# fixed inputs), and only the linear head trains via etl.grad.
# ---------------------------------------------------------------------------

_CONV_BATCH = 2
_CONV_ITERS = 10
_CONV_LR = 0.005
# The raw conv2 features are O(10) (max |f| ≈ 39 at seed 0 — two chained
# 3x3 VALID convs of standard-normal data), which makes the head loss
# ill-conditioned (2/lambda_max of the Hessian ≈ 2e-4 — only lr ≲ 1e-4 is
# stable). Scaling the features by 0.1 in BOTH the etl graph and the numpy
# ref (a constant input normalization) widens the stable band ~100x
# (2/lambda_max ≈ 2e-2), so lr=0.005 trains comfortably.
_CONV_FEAT_SCALE = 0.1
_CONV_FLAT = 8 * 4 * 4  # flattened conv2 output: 8ch × 4×4 (VALID 3×3 ×2)
_CONV_SPECS = tuple(
    TensorSpec(s, _F32)
    for s in [
        (2, 2, 8, 8),  # x
        (4, 2, 3, 3),  # wc1 (frozen conv 1)
        (8, 4, 3, 3),  # wc2 (frozen conv 2)
        (_CONV_FLAT, 2),  # wl (trained)
        (2,),  # bl (trained)
        (2, 2),  # y
    ]
)
_CONV_HEAD_SPECS = (
    TensorSpec((_CONV_BATCH, _CONV_FLAT), _F32),  # features
    TensorSpec((_CONV_FLAT, 2), _F32),  # wl
    TensorSpec((2,), _F32),  # bl
    TensorSpec((2, 2), _F32),  # y
)
_CONV_ARGS = (1, 2)  # trained: wl, bl


@defn
def _convnet_features(x, wc1, wc2):
    h1 = etl.relu(etl.conv(x, wc1, strides=1, padding="VALID"))
    h2 = etl.relu(etl.conv(h1, wc2, strides=1, padding="VALID"))
    feats = etl.reshape(h2, (_CONV_BATCH, _CONV_FLAT))
    return etl.multiply(feats, _CONV_FEAT_SCALE)


@defn
def _convnet_head_loss(feats, wl, bl, y):
    out = etl.add(etl.dot(feats, wl), bl)
    diff = etl.subtract(out, y)
    return etl.mean(etl.multiply(diff, diff))


_convnet_grad = etl.grad(_convnet_head_loss, argnums=_CONV_ARGS)


def _convnet_features_np(x, wc1, wc2):
    h1 = np.maximum(
        conv2d_im2col_numpy(x, wc1, strides=(1, 1), padding="VALID"), 0.0
    )
    h2 = np.maximum(
        conv2d_im2col_numpy(h1, wc2, strides=(1, 1), padding="VALID"), 0.0
    )
    return h2.reshape(_CONV_BATCH, _CONV_FLAT) * _CONV_FEAT_SCALE


def _convnet_train_numpy(inputs):
    """The FULL training procedure in pure numpy (fp64): the fixed conv
    features (same conv semantics as the etl graph via the shared base
    helper) followed by the analytic linear-head backprop loop — the same
    iteration count, lr, update rule, and input data as the runner. The
    head grads were verified against central finite differences at dev time
    (max-abs ≈2e-10)."""
    x, wc1, wc2, wl, bl, y = (a.astype(np.float64) for a in inputs)
    feats = _convnet_features_np(x, wc1, wc2)
    loss = None
    for _ in range(_CONV_ITERS):
        out = feats @ wl + bl
        g_out = 2.0 * (out - y) / out.size
        g_wl = feats.T @ g_out
        g_bl = g_out.sum(axis=0)
        wl = wl - _CONV_LR * g_wl
        bl = bl - _CONV_LR * g_bl
        out = feats @ wl + bl
        loss = np.mean((out - y) ** 2)
    return (np.asarray(loss), wl, bl)


def _convnet_runner(backend, device, opts):
    """Runner factory: stages the features forward (etl.build), the head
    loss (etl.build), and the head ``etl.grad`` TransformCallable
    (lower/compile/load) ONCE, then returns a run-callable performing the
    10-iteration SGD loop: features computed once (frozen convs — constant
    across iterations), then per iteration an etl grad run, an fp32 numpy
    SGD update of the head weights, and an etl loss run on the updated
    weights.

    Explicit device placement (non-cpu devices): the conv inputs are placed
    once and the features stay DEVICE-RESIDENT (never read back — they are
    not part of the returned outputs and are fed into every step's
    executable as device inputs); per-step head grads are read back
    explicitly (the numpy SGD update runs on the host) and the updated host
    head weights are re-placed per step; the final loss is run and read back
    ONCE after the loop (per-step loss runs do not affect the trajectory —
    the loss is a pure function of the head weights)."""
    from .._util import place_input, to_host  # lazy: _util imports .examples

    feats_exe = etl.build(
        _convnet_features, *_CONV_SPECS[:3],
        backend=backend, device=device, **opts
    )
    loss_exe = etl.build(
        _convnet_head_loss, *_CONV_HEAD_SPECS,
        backend=backend, device=device, **opts
    )
    grad_graph = _convnet_grad(*_CONV_HEAD_SPECS)
    lp = etl.lower(grad_graph, backend=backend, **opts)
    ca = etl.compile(lp, **opts)
    grad_exe = etl.load(ca, device=device)

    def run(inputs):
        x, wc1, wc2 = inputs[0], inputs[1], inputs[2]
        y = inputs[5]
        ws = list(inputs[3:5])  # wl, bl (fresh list; inputs untouched)
        loss = None
        if device.kind == "cpu":
            # cpu/host tensors throughout — the current per-step behavior
            # (.numpy() readbacks are zero-copy host views).
            feats = etl.run(feats_exe, x, wc1, wc2).numpy()
            for _ in range(_CONV_ITERS):
                grads = [t.numpy() for t in etl.run(grad_exe, feats, *ws, y)]
                ws = [w - _CONV_LR * g for w, g in zip(ws, grads)]
                loss = etl.run(loss_exe, feats, *ws, y).numpy()
            return (loss, *ws)
        # Non-cpu device: explicit placement/readback (no implicit
        # host<->device transfers) — see the factory docstring.
        x_d, wc1_d, wc2_d = (
            place_input(x, device), place_input(wc1, device),
            place_input(wc2, device),
        )
        y_d = place_input(y, device)
        # Device-resident features: computed once, fed back into every
        # step's executables as a device input — zero host round-trips.
        feats_d = etl.run(feats_exe, x_d, wc1_d, wc2_d)
        for _ in range(_CONV_ITERS):
            ws_d = [place_input(w, device) for w in ws]
            grads = [to_host(t).numpy()
                     for t in etl.run(grad_exe, feats_d, *ws_d, y_d)]
            ws = [w - _CONV_LR * g for w, g in zip(ws, grads)]
        loss = to_host(
            etl.run(loss_exe, feats_d, *(place_input(w, device) for w in ws),
                    y_d)
        ).numpy()
        return (loss, *ws)

    return run


# ---------------------------------------------------------------------------
# e2e_train_transformer — THE FLAGSHIP: SGD training of TWO full simplified
# transformer blocks + projection head (etl.grad + numpy updates).
# Sizes are module-level constants — bump D_MODEL/SEQ/BATCH/ITERS/LR for
# real benchmarking; the defaults keep one runner invocation ≈0.1 s on the
# numpy backend (measured).
# ---------------------------------------------------------------------------

_TF_D_MODEL = 32
_TF_SEQ = 16
_TF_BATCH = 2
_TF_HEADS = 2
_TF_HEAD_DIM = 16
_TF_FFN = 64
_TF_BLOCKS = 2
_TF_ITERS = 10
_TF_LR = 0.02
_TF_EPS = 1e-5

_TF_SPECS = tuple(
    TensorSpec(s, _F32)
    for s in [
        (2, 16, 32),  # x (dummy training data)
        (2, 16, 2),  # y (dummy targets)
        (32, 96), (32, 32), (32, 64), (64, 32),  # block 1: w_qkv, w_out, w1, w2
        (32, 96), (32, 32), (32, 64), (64, 32),  # block 2
        (32, 2),  # wh (head projection)
    ]
)
_TF_ARGS = tuple(range(2, 11))  # trained: all 9 weight tensors (x/y are data)


def _tf_inputs_fn(seed):
    """Custom input generator: x/y stay standard-normal (data), but the 9
    weight tensors are initialized with fan-in scaling (xavier-style —
    standard deviation 1/sqrt(fan_in)) instead of unit variance.

    WHY (measured at dev time): with unit-variance standard-normal weights
    the 2-block transformer starts in a degenerate regime — attention
    logits of std ~30 (saturated softmax, fp32 denormal gradients through
    the non-argmax rows), loss ≈ 67, per-grad magnitudes up to ~34 — and
    the resulting steep-descent trajectory AMPLIFIES the per-step fp32
    grad noise (~1e-3 abs on the deep grads) ~10000x over 10 iterations:
    the final max-abs vs the fp64 reference was ~1.6e-1 (a 15% error on
    the final loss) instead of the ~1e-4 noise-accumulation band. With
    fan-in scaling the forward is O(1) end-to-end (attention logits std
    ~1, loss ~1, grads ~1) and the loop matches the reference to
    ~1e-5..1e-4 — a proper init is part of the training procedure the
    example validates. All weights remain EXPLICIT graph inputs, so the
    runner and the reference still start from identical state."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((_TF_BATCH, _TF_SEQ, _TF_D_MODEL))
    y = rng.standard_normal((_TF_BATCH, _TF_SEQ, 2))
    s_d = 1.0 / math.sqrt(_TF_D_MODEL)
    s_f = 1.0 / math.sqrt(_TF_FFN)
    scales = [s_d, s_d, s_d, s_f, s_d, s_d, s_d, s_f, s_d]
    weights = [
        rng.standard_normal(s) * sc for s, sc in zip(
            [
                (32, 96), (32, 32), (32, 64), (64, 32),  # block 1
                (32, 96), (32, 32), (32, 64), (64, 32),  # block 2
                (32, 2),  # head (fan_in = d)
            ],
            scales,
        )
    ]
    return [a.astype(np.float32) for a in [x, y] + weights]


def _tf_layernorm(x):
    """Layer norm over the last axis from sum primitives (eps 1e-5) — the
    exact forward mirror of base's ``layernorm_numpy`` reference helper."""
    dim = float(_TF_D_MODEL)
    mean = etl.divide(etl.sum(x, axes=-1, keepdims=True), dim)
    diff = etl.subtract(x, mean)
    var = etl.divide(etl.sum(etl.multiply(diff, diff), axes=-1, keepdims=True), dim)
    return etl.divide(diff, etl.sqrt(etl.add(var, _TF_EPS)))


def _tf_block(x, w_qkv, w_out, w1, w2):
    """One simplified transformer block (traced at defn time): MHA with
    ``_TF_HEADS`` heads of dim ``_TF_HEAD_DIM`` + layernorm + gelu FFN +
    residuals (no biases). Static shapes throughout — the op-for-op mirror
    of the numpy ``_tf_block_np`` forward."""
    b, s, d = _TF_BATCH, _TF_SEQ, _TF_D_MODEL
    h, hd = _TF_HEADS, _TF_HEAD_DIM
    qkv = etl.reshape(etl.dot(x, w_qkv), (b, s, 3, h, hd))
    q = etl.reshape(
        etl.slice(qkv, (0, 0, 0, 0, 0), (b, s, 1, h, hd)), (b, s, h, hd)
    )
    k = etl.reshape(
        etl.slice(qkv, (0, 0, 1, 0, 0), (b, s, 1, h, hd)), (b, s, h, hd)
    )
    v = etl.reshape(
        etl.slice(qkv, (0, 0, 2, 0, 0), (b, s, 1, h, hd)), (b, s, h, hd)
    )
    scale = 1.0 / math.sqrt(hd)
    q = etl.transpose(q, (0, 2, 1, 3))  # [b,h,s,hd]
    kt = etl.transpose(k, (0, 2, 3, 1))  # [b,h,hd,s]
    v = etl.transpose(v, (0, 2, 1, 3))  # [b,h,s,hd]
    scores = etl.multiply(etl.dot(q, kt), scale)  # [b,h,s,s]
    m = etl.max(scores, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(scores, m))
    probs = etl.divide(e, etl.sum(e, axes=-1, keepdims=True))
    attn = etl.dot(probs, v)  # [b,h,s,hd]
    attn = etl.transpose(attn, (0, 2, 1, 3))  # [b,s,h,hd]
    heads = [
        etl.slice(attn, (0, 0, hi, 0), (b, s, 1, hd)) for hi in range(h)
    ]
    attn = etl.reshape(etl.concatenate(heads, axis=3), (b, s, d))
    out = etl.add(etl.dot(attn, w_out), x)
    out = _tf_layernorm(out)
    out = etl.add(etl.dot(etl.gelu(etl.dot(out, w1)), w2), out)
    return _tf_layernorm(out)


@defn
def _transformer_loss(x, y, wqkv1, wout1, w11, w21, wqkv2, wout2, w12, w22, wh):
    h = _tf_block(x, wqkv1, wout1, w11, w21)
    h = _tf_block(h, wqkv2, wout2, w12, w22)
    logits = etl.dot(h, wh)
    diff = etl.subtract(logits, y)
    return etl.mean(etl.multiply(diff, diff))


_transformer_grad = etl.grad(_transformer_loss, argnums=_TF_ARGS)


def _tf_block_np(x, w_qkv, w_out, w1, w2, heads=_TF_HEADS, head_dim=_TF_HEAD_DIM):
    """Forward of one transformer block in numpy (fp64) — op-for-op mirror
    of the etl ``_tf_block`` via the shared base helpers (layernorm_numpy /
    gelu_numpy). Shape-parameterized (heads/head_dim) so the backward can
    be fd-verified on a small config at dev time."""
    b, s, d = x.shape
    qkv = (x @ w_qkv).reshape(b, s, 3, heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q.transpose(0, 2, 1, 3) @ k.transpose(0, 2, 3, 1)) * scale
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    attn = (probs @ v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    attn = np.concatenate(
        [attn[:, :, hi : hi + 1, :] for hi in range(heads)], axis=3
    ).reshape(b, s, d)
    out = attn @ w_out + x
    out = layernorm_numpy(out)
    out = gelu_numpy(out @ w1) @ w2 + out
    return layernorm_numpy(out)


def _tf_block_backward_np(
    x, w_qkv, w_out, w1, w2, g_out, heads=_TF_HEADS, head_dim=_TF_HEAD_DIM
):
    """Analytic backward of one transformer block (fp64). ``g_out`` is the
    cotangent of the block OUTPUT (post-layernorm2). Returns
    (g_x, g_w_qkv, g_w_out, g_w1, g_w2). Every piece (layernorm, softmax,
    gelu, matmuls) was verified against central finite differences at dev
    time on a small config (max-abs ≈2e-12 over all five grads)."""
    b, s, d = x.shape
    h, hd = heads, head_dim
    scale = 1.0 / math.sqrt(hd)

    # --- forward (saved intermediates) ---
    qkv = (x @ w_qkv).reshape(b, s, 3, h, hd)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    qt = q.transpose(0, 2, 1, 3)  # [b,h,s,hd]
    kt = k.transpose(0, 2, 3, 1)  # [b,h,hd,s]
    vt = v.transpose(0, 2, 1, 3)  # [b,h,s,hd]
    scores = (qt @ kt) * scale  # [b,h,s,s]
    m = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / e.sum(axis=-1, keepdims=True)
    attn_h = probs @ vt  # [b,h,s,hd]
    attn_t = attn_h.transpose(0, 2, 1, 3)  # [b,s,h,hd]
    attn_c = np.concatenate(
        [attn_t[:, :, hi : hi + 1, :] for hi in range(h)], axis=3
    ).reshape(b, s, d)
    z1 = attn_c @ w_out + x
    n1 = layernorm_numpy(z1)
    ff_in = n1 @ w1
    ff1 = gelu_numpy(ff_in)
    ff2 = ff1 @ w2
    z2 = ff2 + n1
    n2 = layernorm_numpy(z2)

    # --- backward (verify against n2: layernorm2 -> residual split) ---
    g_z2 = _layernorm_backward(g_out, z2, d)
    g_ff2 = g_z2
    g_n1 = g_z2.copy()
    g_w2 = ff1.reshape(-1, w2.shape[0]).T @ g_ff2.reshape(-1, w2.shape[1])
    g_ff1 = g_ff2 @ w2.T
    g_ff_in = g_ff1 * _gelu_deriv(ff_in)
    g_w1 = n1.reshape(-1, w1.shape[0]).T @ g_ff_in.reshape(-1, w1.shape[1])
    g_n1 = g_n1 + g_ff_in @ w1.T
    g_z1 = _layernorm_backward(g_n1, z1, d)
    g_x = g_z1.copy()
    g_w_out = attn_c.reshape(-1, d).T @ g_z1.reshape(-1, d)
    g_attn_c = (g_z1 @ w_out.T).reshape(b, s, h, hd)
    g_attn_h = g_attn_c.transpose(0, 2, 1, 3)  # [b,h,s,hd]
    g_probs = g_attn_h @ vt.transpose(0, 1, 3, 2)  # [b,h,s,s]
    g_vt = probs.transpose(0, 1, 3, 2) @ g_attn_h  # [b,h,hd,s]
    g_scores = _softmax_backward(g_probs, probs) * scale
    g_qt = g_scores @ kt.transpose(0, 1, 3, 2)  # [b,h,s,hd]
    g_kt = qt.transpose(0, 1, 3, 2) @ g_scores  # [b,h,hd,s]
    g_q = g_qt.transpose(0, 2, 1, 3)  # [b,s,h,hd]
    g_k = g_kt.transpose(0, 3, 1, 2)
    g_v = g_vt.transpose(0, 2, 1, 3)
    g_qkv = np.concatenate(
        [g_q[:, :, None, :, :], g_k[:, :, None, :, :], g_v[:, :, None, :, :]],
        axis=2,
    ).reshape(b, s, 3 * h * hd)
    g_w_qkv = x.reshape(-1, d).T @ g_qkv.reshape(-1, 3 * h * hd)
    g_x = g_x + g_qkv @ w_qkv.T
    return g_x, g_w_qkv, g_w_out, g_w1, g_w2


def _transformer_train_numpy(inputs):
    """The FULL training procedure in pure numpy (fp64): the analytic
    transformer backward loop — the same iteration count, lr, update rule,
    and input data as the runner. Verified end-to-end against central
    finite differences on a small config at dev time."""
    x = inputs[0].astype(np.float64)
    y = inputs[1].astype(np.float64)
    wqkv1, wout1, w11, w21, wqkv2, wout2, w12, w22, wh = (
        a.astype(np.float64) for a in inputs[2:]
    )
    loss = None
    for _ in range(_TF_ITERS):
        h1 = _tf_block_np(x, wqkv1, wout1, w11, w21)
        h2 = _tf_block_np(h1, wqkv2, wout2, w12, w22)
        logits = h2 @ wh
        diff = logits - y
        g_logits = 2.0 * diff / diff.size
        g_wh = h2.reshape(-1, _TF_D_MODEL).T @ g_logits.reshape(-1, 2)
        g_h2 = g_logits @ wh.T
        g_h1, g_wqkv2, g_wout2, g_w12, g_w22 = _tf_block_backward_np(
            h1, wqkv2, wout2, w12, w22, g_h2
        )
        _, g_wqkv1, g_wout1, g_w11, g_w21 = _tf_block_backward_np(
            x, wqkv1, wout1, w11, w21, g_h1
        )
        wqkv1 = wqkv1 - _TF_LR * g_wqkv1
        wout1 = wout1 - _TF_LR * g_wout1
        w11 = w11 - _TF_LR * g_w11
        w21 = w21 - _TF_LR * g_w21
        wqkv2 = wqkv2 - _TF_LR * g_wqkv2
        wout2 = wout2 - _TF_LR * g_wout2
        w12 = w12 - _TF_LR * g_w12
        w22 = w22 - _TF_LR * g_w22
        wh = wh - _TF_LR * g_wh
    # Final loss on the TRAINED weights (the runner's order: the etl loss
    # executable runs on the updated weights at the end of every iteration).
    h1 = _tf_block_np(x, wqkv1, wout1, w11, w21)
    h2 = _tf_block_np(h1, wqkv2, wout2, w12, w22)
    loss = np.mean((h2 @ wh - y) ** 2)
    return (np.asarray(loss), wqkv1, wout1, w11, w21, wqkv2, wout2, w12, w22, wh)


def _transformer_runner(backend, device, opts):
    """Runner factory: stages the loss ``@etl.defn`` (etl.build) and the
    ``etl.grad`` TransformCallable over all 9 weight tensors
    (lower/compile/load) ONCE, then returns a run-callable performing the
    10-iteration SGD loop in Python: per iteration an etl grad run over the
    full 11-input batch, an fp32 numpy SGD update of all 9 weights, and an
    etl loss run on the updated weights.

    Explicit device placement (non-cpu devices): the data is placed once and
    stays device-resident; the per-step grads are read back explicitly (the
    numpy SGD update runs on the host, so they are a hard per-step data
    dependency) and the updated host weights are re-placed per step; the
    per-step loss runs do not affect the trajectory (the loss is a pure
    function of the weights), so on non-cpu devices the loss is run and read
    back ONCE after the loop — the timed loop performs no per-step host
    readback of device outputs beyond the required grads."""
    from .._util import place_input, to_host  # lazy: _util imports .examples

    loss_exe = etl.build(
        _transformer_loss, *_TF_SPECS, backend=backend, device=device, **opts
    )
    grad_graph = _transformer_grad(*_TF_SPECS)
    lp = etl.lower(grad_graph, backend=backend, **opts)
    ca = etl.compile(lp, **opts)
    grad_exe = etl.load(ca, device=device)

    def run(inputs):
        x, y = inputs[0], inputs[1]
        ws = list(inputs[2:])  # 9 current weights (fresh list; inputs untouched)
        loss = None
        if device.kind == "cpu":
            # cpu/host tensors throughout — the current per-step behavior
            # (.numpy() readbacks are zero-copy host views).
            for _ in range(_TF_ITERS):
                grads = [t.numpy() for t in etl.run(grad_exe, x, y, *ws)]
                ws = [w - _TF_LR * g for w, g in zip(ws, grads)]
                loss = etl.run(loss_exe, x, y, *ws).numpy()
            return (loss, *ws)
        # Non-cpu device: explicit placement/readback (no implicit
        # host<->device transfers) — see the factory docstring.
        x_d, y_d = place_input(x, device), place_input(y, device)
        for _ in range(_TF_ITERS):
            ws_d = [place_input(w, device) for w in ws]
            grads = [to_host(t).numpy() for t in etl.run(grad_exe, x_d, y_d, *ws_d)]
            ws = [w - _TF_LR * g for w, g in zip(ws, grads)]
        loss = to_host(
            etl.run(loss_exe, x_d, y_d, *(place_input(w, device) for w in ws))
        ).numpy()
        return (loss, *ws)

    return run


# ---------------------------------------------------------------------------
# Registry (category "e2e", tag "train"). Tolerances: max-abs over ALL
# outputs (final loss + final weights), set from the MEASURED final error of
# the fp32-etl training loop vs the fp64 analytic reference (seed 0, numpy
# backend — see the module docstring): the per-step fp32 grad noise
# accumulates over the 10 iterations.
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="e2e_train_mlp",
        description=(
            "10-iter SGD training of a 2-layer MLP via etl.grad (runner: "
            "Python loop; outputs final loss + weights)"
        ),
        specs=_MLP_SPECS,
        graph=_mlp_loss,
        numpy_ref=_mlp_train_numpy,
        # Measured final max-abs (loss + w1/b1/w2/b2) = 4.4e-07 vs the fp64
        # analytic loop — tolerance=1e-3 with ~2000x margin (fp32
        # accumulation noise; the numpy backend itself is exact).
        tolerance=1e-3,
        category="e2e",
        tags=("train",),
        runner=_mlp_runner,
    ),
    Example(
        name="e2e_train_convnet",
        description=(
            "10-iter SGD training of a small conv net's linear head, convs "
            "frozen (v1 conv-VJP deferral; runner: Python loop)"
        ),
        specs=_CONV_SPECS,
        # Informational only (runner path stages its own executables): the
        # head loss defn — its specs are the 4 head specs, not the full
        # 6-spec input list.
        graph=_convnet_head_loss,
        numpy_ref=_convnet_train_numpy,
        # Measured final max-abs (loss + wl/bl) = 2.8e-07 vs the fp64
        # analytic loop — tolerance=1e-3 with ~3500x margin.
        tolerance=1e-3,
        category="e2e",
        tags=("train",),
        runner=_convnet_runner,
    ),
    Example(
        name="e2e_train_transformer",
        description=(
            "10-iter SGD training of TWO simplified transformer blocks "
            "(2 heads x 16, layernorm, gelu FFN) + head via etl.grad "
            "(runner: Python loop; d=32/seq=16/batch=2)"
        ),
        specs=_TF_SPECS,
        inputs_fn=_tf_inputs_fn,
        graph=_transformer_loss,
        numpy_ref=_transformer_train_numpy,
        # Measured final max-abs (loss + all 9 weights) = 1.5e-07 vs the
        # fp64 analytic loop (xavier init via inputs_fn, see the module
        # docstring) — tolerance=1e-3 with ~6500x margin.
        tolerance=1e-3,
        category="e2e",
        tags=("train",),
        runner=_transformer_runner,
    ),
])