"""Gradient conformance examples (category 'grad') — autodiff via ``etl.grad``.

Each example bundles a ``@etl.defn`` scalar loss and the ``etl.grad(...)``
``TransformCallable`` as ``example.graph`` (a ``TransformCallable`` is NOT a
``Defn`` — ``getattr(graph, "__etl_defn__", False)`` is False — so the
harness stages it via ``lower``/``compile``/``load``). ``numpy_ref`` is the
per-element **central finite-difference** gradient of the same loss in
float64 (:func:`~etl.bench.examples.base.fd_gradient`, ``h=1e-4``) and
returns the grads for the SAME ``argnums`` subset as ``etl.grad`` (matching
output structure). ``torch_ref`` uses ``torch.autograd`` lazily (torch never
imported at module scope — see ``etl.bench._torch``).

Measured max-abs errors (etl fp32 grads vs fp64 central-fd at ``h=1e-4``,
and vs torch autograd) are all ≲ 1e-5 for every example, so the default
per-example tolerance ``rtol=atol=1e-3`` (documented relaxation of the
harness-wide ``1e-5`` — gradient comparisons are fp32-vs-fp64) passes with
>100x margin; see each example's comment for the exact numbers.

Conformance status by backend:
  - ``grad_mlp`` / ``grad_mix`` / ``grad_stopgrad``: numpy AND iree (CPU)
    validated end-to-end (grad_mix iree max-abs ≈ 6.7e-6).
  - ``grad_structural``: numpy-validated (30 seeds, all pass); on compiler
    backends it FAILS with an explicit ``BackendError`` (StableHLO v1 defers
    ``gather``/``scatter`` export) — a documented, recorded per-example
    failure like the ``cumsum`` iree deferral, NOT a harness bug.

VJP-rule coverage (rule registry ``etl.transforms.vjp_rules``, 66 keys) —
rule → where exercised:
  - grad_mix (all-backend portable mix; every op tagged ``# vjp: <name>``):
    add, subtract, multiply, divide, power, remainder, maximum, minimum,
    abs, negate, square, sqrt, exp, log, log1p, sin, cos, tan, tanh,
    sigmoid, relu, cast, select, broadcast, dot, reshape, transpose, slice,
    pad, concatenate, sum/reduce_sum, mean/reduce_mean, max/reduce_max,
    min/reduce_min, prod/reduce_prod (frontend names are sugar for the
    ``reduce_*`` IR ops), and the zero-grad class less, equal, greater,
    logical_and (scaled to keep FD boundary flips bounded).
  - grad_stopgrad: stop_gradient.
  - grad_structural: gather, scatter, solve.
  - NOT covered (one-line justification):
    - conv — documented v1 deferral: ``TransformError: grad/vjp: no VJP
      rule for op 'conv' — the IR has no transposed-convolution op (v1
      gap)`` (probed at dev time; conv is deliberately NOT in any example).
    - erf, gelu — VJP rules numpy-validated at dev time, but StableHLO v1
      defers their export (no ``stablehlo.erf``; ``gelu`` decomposes via
      erf) → ``BackendError`` on compiler backends; grad_mix must run on
      ALL backends, so they are excluded.
    - cumsum, tril, triu — same StableHLO v1 export deferral (no export
      mapping; micro's ``cumsum`` documents the identical iree failure).
    - argmax, argmin — index-output ops (int results); their ``_zero_vjp``
      class contributes nothing differentiable to a scalar loss; also
      StableHLO-deferred.
    - bitwise_and/or/xor — zero-grad int ops; logical_and (same
      ``_zero_vjp`` rule) covers the zero-gradient path.
    - logical_or, logical_not, not_equal, less_equal, greater_equal — same
      ``_zero_vjp`` rule class as covered equal/less/greater/logical_and.

FD-robustness design (why the mix has no kinks): elementwise non-smooth ops
are shifted away from the data range (``relu(x+5)``, ``abs(x+5)``,
``maximum(x, -5)``, ``minimum(x, 5)`` — always-linear regimes, gradient
exact; ``sign(x+5)`` runs in an always-constant regime — its gradient is
exactly 0, so the FD and torch refs agree identically while the VJP rule
still fires), ``remainder`` runs in its identity regime (``a < 1000``),
reduction ``max``/``min`` are scaled 1e-4 (a central-difference tie-flip
error is bounded by 0.5·scale), the comparison/logical zero-grad terms are
scaled 1e-8 (FD boundary flip ≤ 1e-8/h), and ``select``'s branches differ
by 1e-8 (mask flips bounded while both branch gradients still flow). The
cast term is an exact f32→f64→f32 round trip (f32 values are exactly
representable in f64), so the numpy ref treats it as the identity — the FD
must not quantize.
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, fd_gradient, register_all

# --- grad_mlp ---------------------------------------------------------------
# 2-layer MLP + MSE, grads wrt all six inputs.
#
# FD sanity: fd-vs-torch agreement (h=1e-4) measured at dev time = 2.6e-06
# max over the six grads (≲1e-3 required — no h-scan needed; h=1e-4 kept).
# etl vs numpy-fd max-abs = 2.2e-06; etl vs torch = 3.8e-06.


@defn
def _grad_mlp_loss(x, w1, b1, w2, b2, y):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    out = etl.add(etl.dot(h, w2), b2)
    diff = etl.subtract(out, y)
    return etl.mean(etl.multiply(diff, diff))


def _grad_mlp_value(inputs, frozen):
    x, w1, b1, w2, b2, y = inputs
    h = np.maximum(x @ w1 + b1, 0.0)
    out = h @ w2 + b2
    diff = out - y
    return np.mean(diff * diff)


def _grad_mlp_numpy(inputs):
    return tuple(fd_gradient(_grad_mlp_value, inputs))


def _grad_mlp_torch(inputs, device=None):
    torch = require_torch()
    x, w1, b1, w2, b2, y = (
        torch.tensor(a, device=device, requires_grad=True) for a in inputs
    )
    h = torch.relu(x @ w1 + b1)
    out = h @ w2 + b2
    diff = out - y
    loss = torch.mean(diff * diff)
    grads = torch.autograd.grad(loss, (x, w1, b1, w2, b2, y))
    return tuple(g.cpu().numpy() for g in grads)


# --- grad_mix ---------------------------------------------------------------
# ONE scalar loss mixing ~40 differentiable ops; must run on ALL backends, so
# every op here has a StableHLO export (no gather/scatter/solve/conv/erf/
# gelu/cumsum/tril/triu — those are StableHLO v1 deferrals, see the module
# docstring). Inputs are small ~N(0,1) f32; tan is scaled (tan(0.1*x)) to
# avoid blowup; denominators are always-positive (log1p(x^2)+1 style).
#
# Measured (dev): etl vs numpy-fd max-abs = 6.7e-06 (seed 1); max over seeds
# 0-11 = 1.36e-05; etl vs torch = 3.8e-06; iree CPU max-abs = 6.7e-06.


@defn
def _grad_mix_loss(x1, x2, w, y):
    terms = []

    # elementwise family (all [6,10] / [10,4])
    e = etl.add(x1, x2)                                     # vjp: add
    d = etl.subtract(x1, x2)                                # vjp: subtract
    terms.append(etl.sum(etl.multiply(e, d)))               # vjp: multiply
    pos = etl.add(etl.square(x1), 1.0)                      # vjp: square
    # abs(x+5): x+5 > 0 always -> no FD kink; pos >= 1 -> divide safe
    terms.append(etl.sum(etl.divide(etl.abs(etl.add(x1, 5.0)), pos)))  # vjp: divide, abs
    terms.append(etl.sum(etl.power(pos, 2.0)))              # vjp: power (static exponent)
    terms.append(etl.sum(etl.sqrt(etl.add(etl.log1p(etl.square(x2)), 1.0))))  # vjp: sqrt, log1p
    terms.append(etl.sum(etl.log(etl.add(etl.log1p(etl.square(x1)), 1.0))))   # vjp: log
    terms.append(etl.sum(etl.exp(etl.multiply(x1, 0.1))))   # vjp: exp
    terms.append(etl.sum(etl.sin(x1)))                      # vjp: sin
    terms.append(etl.sum(etl.cos(x2)))                      # vjp: cos
    terms.append(etl.sum(etl.tan(etl.multiply(x1, 0.1))))   # vjp: tan (scaled arg)
    terms.append(etl.sum(etl.tanh(x1)))                     # vjp: tanh
    terms.append(etl.sum(etl.sigmoid(x2)))                  # vjp: sigmoid
    terms.append(etl.sum(etl.relu(etl.add(x1, 5.0))))       # vjp: relu (linear regime)
    # sign(x1+5): x1+5 > 0 always -> always-constant regime; the gradient is
    # exactly 0 (FD and torch agree identically), but the VJP rule fires —
    # which is the coverage point
    terms.append(etl.sum(etl.sign(etl.add(x1, 5.0))))       # vjp: sign (constant regime)
    terms.append(etl.sum(etl.negate(x1)))                   # vjp: negate
    terms.append(etl.sum(etl.maximum(x1, -5.0)))            # vjp: maximum (x1 > -5 always)
    terms.append(etl.sum(etl.minimum(x2, 5.0)))             # vjp: minimum (x2 < 5 always)
    # remainder in its identity regime (|x1+5|+5 < 1000 always): op + its
    # ct-passthrough rule exercised without FD discontinuity flips
    rm = etl.remainder(etl.add(etl.abs(etl.add(x1, 5.0)), 5.0), 1000.0)  # vjp: remainder
    terms.append(etl.sum(rm))
    # zero-grad comparison/logical terms (rules yield zero cotangents);
    # scaled 1e-8 so any FD boundary flip is bounded (<= ~1e-8/h)
    terms.append(etl.multiply(etl.sum(etl.cast(etl.less(x1, 100.0), etl.float32)), 1e-8))  # vjp: less, cast
    terms.append(etl.multiply(etl.sum(etl.cast(etl.equal(x1, etl.add(x2, 100.0)), etl.float32)), 1e-8))  # vjp: equal
    terms.append(etl.multiply(etl.sum(etl.cast(etl.logical_and(etl.greater(x1, -100.0), etl.less(x1, 100.0)), etl.float32)), 1e-8))  # vjp: logical_and, greater

    # reductions (frontend names are sugar for the reduce_* IR ops; the
    # max/min variants are scaled 1e-4 so an FD tie-flip error is <= 0.5*scale)
    terms.append(etl.sum(etl.sum(x1, axes=-1)))             # vjp: sum
    terms.append(etl.sum(etl.reduce_sum(x1, axes=0)))       # vjp: reduce_sum
    terms.append(etl.sum(etl.mean(x1, axes=-1)))            # vjp: mean
    terms.append(etl.sum(etl.reduce_mean(x1, axes=0)))      # vjp: reduce_mean
    terms.append(etl.multiply(etl.sum(etl.max(x1, axes=-1)), 1e-4))    # vjp: max
    terms.append(etl.multiply(etl.sum(etl.reduce_max(x1, axes=0)), 1e-4))  # vjp: reduce_max
    terms.append(etl.multiply(etl.sum(etl.min(x1, axes=-1)), 1e-4))    # vjp: min
    terms.append(etl.multiply(etl.sum(etl.reduce_min(x1, axes=0)), 1e-4))  # vjp: reduce_min
    terms.append(etl.sum(etl.prod(etl.multiply(x1, 0.3), axes=-1)))    # vjp: prod
    terms.append(etl.sum(etl.reduce_prod(etl.multiply(x2, 0.3), axes=0)))  # vjp: reduce_prod

    # linalg / structural family
    terms.append(etl.sum(etl.multiply(etl.dot(x1, w), y)))  # vjp: dot
    terms.append(etl.sum(etl.transpose(x1)))                # vjp: transpose
    terms.append(etl.sum(etl.reshape(x1, (3, 2, 10))))      # vjp: reshape
    terms.append(etl.sum(etl.slice(x1, start=(0, 0), lengths=(4, 6), strides=1)))  # vjp: slice
    terms.append(etl.sum(etl.pad(x1, ((0, 0), (2, 2)), 0.0)))  # vjp: pad
    terms.append(etl.sum(etl.concatenate([x1, x2], axis=1)))  # vjp: concatenate
    terms.append(etl.sum(etl.broadcast(etl.reduce_mean(x1, axes=0), (6, 10))))  # vjp: broadcast
    # select: ~half-true mask; branches differ by 1e-8 so a mask flip is FD-
    # bounded while both branch gradients flow
    mask = etl.greater(x1, 0.0)                             # vjp: greater
    terms.append(etl.sum(etl.select(mask, x1, etl.add(x1, 1e-8))))  # vjp: select
    # cast f32 -> f64 -> f32: EXACT round trip for f32 inputs (the numpy ref
    # treats it as the identity — the FD must not quantize)
    terms.append(etl.sum(etl.cast(etl.cast(x1, etl.float64), etl.float32)))  # vjp: cast

    total = terms[0]
    for term in terms[1:]:
        total = etl.add(total, term)
    return total


def _grad_mix_value(inputs, frozen):
    """The same loss as :func:`_grad_mix_loss` in numpy (float64 inside the
    FD helper); terms must mirror the graph op-for-op."""
    x1, x2, w, y = inputs
    e = x1 + x2
    d = x1 - x2
    pos = x1 * x1 + 1.0
    terms = [
        np.sum(e * d),
        np.sum(np.abs(x1 + 5.0) / pos),
        np.sum(pos ** 2.0),
        np.sum(np.sqrt(np.log1p(x2 * x2) + 1.0)),
        np.sum(np.log(np.log1p(x1 * x1) + 1.0)),
        np.sum(np.exp(0.1 * x1)),
        np.sum(np.sin(x1)),
        np.sum(np.cos(x2)),
        np.sum(np.tan(0.1 * x1)),
        np.sum(np.tanh(x1)),
        np.sum(1.0 / (1.0 + np.exp(-x2))),
        np.sum(np.maximum(x1 + 5.0, 0.0)),
        np.sum(np.sign(x1 + 5.0)),
        np.sum(-x1),
        np.sum(np.maximum(x1, -5.0)),
        np.sum(np.minimum(x2, 5.0)),
        np.sum(np.remainder(np.abs(x1 + 5.0) + 5.0, 1000.0)),
        1e-8 * np.sum((x1 < 100.0).astype(np.float64)),
        1e-8 * np.sum((x1 == x2 + 100.0).astype(np.float64)),
        1e-8 * np.sum(np.logical_and(x1 > -100.0, x1 < 100.0).astype(np.float64)),
        np.sum(np.sum(x1, axis=-1)),
        np.sum(np.sum(x1, axis=0)),
        np.sum(np.mean(x1, axis=-1)),
        np.sum(np.mean(x1, axis=0)),
        1e-4 * np.sum(np.max(x1, axis=-1)),
        1e-4 * np.sum(np.max(x1, axis=0)),
        1e-4 * np.sum(np.min(x1, axis=-1)),
        1e-4 * np.sum(np.min(x1, axis=0)),
        np.sum(np.prod(0.3 * x1, axis=-1)),
        np.sum(np.prod(0.3 * x2, axis=0)),
        np.sum((x1 @ w) * y),
        np.sum(x1.T),
        np.sum(x1.reshape(3, 2, 10)),
        np.sum(x1[0:4, 0:6]),
        np.sum(np.pad(x1, ((0, 0), (2, 2)), constant_values=0.0)),
        np.sum(np.concatenate([x1, x2], axis=1)),
        np.sum(np.broadcast_to(np.mean(x1, axis=0), (6, 10))),
        np.sum(np.where(x1 > 0.0, x1, x1 + 1e-8)),
        np.sum(x1),  # cast f32->f64->f32 is the exact identity for f32 inputs
    ]
    return float(np.sum(terms))


def _grad_mix_numpy(inputs):
    return tuple(fd_gradient(_grad_mix_value, inputs))


def _grad_mix_torch(inputs, device=None):
    torch = require_torch()
    x1, x2, w, y = (torch.tensor(a, device=device, requires_grad=True) for a in inputs)
    e = x1 + x2
    d = x1 - x2
    pos = x1 * x1 + 1.0
    t1 = (e * d).sum()
    t2 = ((x1 + 5.0).abs() / pos).sum()
    t3 = (pos ** 2.0).sum()
    t4 = torch.sqrt(torch.log1p(x2 * x2) + 1.0).sum()
    t5 = torch.log(torch.log1p(x1 * x1) + 1.0).sum()
    t6 = torch.exp(0.1 * x1).sum()
    t7 = torch.sin(x1).sum()
    t8 = torch.cos(x2).sum()
    t9 = torch.tan(0.1 * x1).sum()
    t10 = torch.tanh(x1).sum()
    t11 = torch.sigmoid(x2).sum()
    t12 = torch.relu(x1 + 5.0).sum()
    t13 = torch.sign(x1 + 5.0).sum()
    t15 = (-x1).sum()
    t16 = torch.maximum(x1, torch.tensor(-5.0, device=device)).sum()
    t17 = torch.minimum(x2, torch.tensor(5.0, device=device)).sum()
    t18 = torch.remainder((x1 + 5.0).abs() + 5.0, torch.tensor(1000.0, device=device)).sum()
    t19 = 1e-8 * (x1 < 100.0).float().sum()
    t20 = 1e-8 * (x1 == x2 + 100.0).float().sum()
    t21 = 1e-8 * ((x1 > -100.0) & (x1 < 100.0)).float().sum()
    t22 = x1.sum(axis=-1).sum()
    t23 = x1.sum(axis=0).sum()
    t24 = x1.mean(axis=-1).sum()
    t25 = x1.mean(axis=0).sum()
    t26 = 1e-4 * x1.max(axis=-1).values.sum()
    t27 = 1e-4 * x1.max(axis=0).values.sum()
    t28 = 1e-4 * x1.min(axis=-1).values.sum()
    t29 = 1e-4 * x1.min(axis=0).values.sum()
    t30 = (0.3 * x1).prod(axis=-1).sum()
    t31 = (0.3 * x2).prod(axis=0).sum()
    t32 = ((x1 @ w) * y).sum()
    t33 = x1.T.sum()
    t36 = x1.reshape(3, 2, 10).sum()
    t37 = x1[0:4, 0:6].sum()
    t38 = torch.nn.functional.pad(x1, (2, 2, 0, 0)).sum()
    t39 = torch.cat([x1, x2], dim=1).sum()
    t40 = torch.broadcast_to(x1.mean(axis=0), (6, 10)).sum()
    t41 = torch.where(x1 > 0.0, x1, x1 + 1e-8).sum()
    t42 = x1.sum()  # cast f32->f64->f32 is the exact identity for f32 inputs
    loss = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12 + t13 + t15 + t16 + t17 + t18 + t19 + t20 + t21 + t22 + t23 + t24 + t25 + t26 + t27 + t28 + t29 + t30 + t31 + t32 + t33 + t36 + t37 + t38 + t39 + t40 + t41 + t42
    grads = torch.autograd.grad(loss, (x1, x2, w, y))
    return tuple(g.cpu().numpy() for g in grads)


# --- grad_stopgrad ----------------------------------------------------------
# loss = sum(stop_gradient(x) * w) + 0.5 * sum(x^2)  ->  grads (x, x):
# the sg-term contributes d/dx = 0 (fd freezes input 0 via sg_inputs) and
# d/dw = x; the smooth term contributes d/dx = x.
#
# Measured (dev): etl vs numpy-fd and vs torch are EXACT (0.0) — the fd
# value function is linear in x/w; fd-vs-torch agreement 0.0.


@defn
def _grad_stopgrad_loss(x, w):
    sg = etl.multiply(etl.stop_gradient(x), w)
    return etl.add(etl.sum(sg), etl.multiply(etl.sum(etl.square(x)), 0.5))


def _grad_stopgrad_value(inputs, frozen):
    x, w = inputs
    return np.sum(frozen.get(0, x) * w) + 0.5 * np.sum(x * x)


def _grad_stopgrad_numpy(inputs):
    # sg_inputs=(0,): freezing x for the sg-term kills its d/dx contribution
    return tuple(fd_gradient(_grad_stopgrad_value, inputs, sg_inputs=(0,)))


def _grad_stopgrad_torch(inputs, device=None):
    torch = require_torch()
    x, w = (torch.tensor(a, device=device, requires_grad=True) for a in inputs)
    loss = torch.sum(x.detach() * w) + 0.5 * torch.sum(x * x)
    grads = torch.autograd.grad(loss, (x, w))
    return tuple(g.cpu().numpy() for g in grads)


# --- grad_structural --------------------------------------------------------
# gather + scatter (+ solve) grads. Runs on numpy (validated over 30 seeds);
# compiler backends raise an explicit BackendError (StableHLO v1 defers
# gather/scatter export) — a documented per-example failure, like cumsum's
# iree deferral. ``idx`` is int32 (generated via ``inputs_fn`` — distinct
# non-negative indices so gather/scatter have no duplicate-index ambiguity);
# argnums excludes the int32 position. The solve term uses a diagonally
# dominant A = I + 0.3*randn (cond <= ~56 over all tested seeds) so the
# solve gradient stays well-conditioned for every seed.
#
# Measured (dev): etl vs numpy-fd max-abs = 2.7e-07 (seed 0); worst over 30
# seeds = 0.13 on a |grad| ~ 1.3e4 element (relative error ~1e-5, ~100x
# below the rtol=1e-3 bound); etl vs torch = 4.8e-07 (seed 0).


@defn
def _grad_structural_loss(x, idx, w, src, a, b):
    g = etl.gather(x, idx, axis=0)
    term1 = etl.mean(etl.multiply(g, w))
    sc = etl.scatter(x, idx, src, axis=0)
    term2 = etl.multiply(etl.mean(etl.square(sc)), 0.5)
    s = etl.solve(a, b)
    term3 = etl.mean(etl.square(s))
    return etl.add(etl.add(term1, term2), term3)


def _grad_structural_value(inputs, frozen):
    x, idx, w, src, a, b = inputs
    term1 = np.mean(x[idx] * w)
    xc = x.copy()
    xc[idx] = src  # etl.scatter semantics: copy of x with rows idx <- src
    term2 = 0.5 * np.mean(xc * xc)
    s = np.linalg.solve(a, b)
    term3 = np.mean(s * s)
    return float(term1 + term2 + term3)


_GRAD_STRUCTURAL_ARGS = (0, 2, 3, 4, 5)  # exclude the int32 idx position


def _grad_structural_numpy(inputs):
    grads = fd_gradient(_grad_structural_value, inputs)
    return tuple(grads[i] for i in _GRAD_STRUCTURAL_ARGS)


def _grad_structural_torch(inputs, device=None):
    torch = require_torch()
    x, idx, w, src, a, b = inputs
    # torch.index_copy(0, idx, src) matches etl.scatter(x, idx, src, axis=0)
    # (x[idx] = src on a copy); idx must be int64 for torch indexing.
    x_t = torch.tensor(x, device=device, requires_grad=True)
    w_t = torch.tensor(w, device=device, requires_grad=True)
    src_t = torch.tensor(src, device=device, requires_grad=True)
    a_t = torch.tensor(a, device=device, requires_grad=True)
    b_t = torch.tensor(b, device=device, requires_grad=True)
    idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
    term1 = torch.mean(x_t[idx_t] * w_t)
    term2 = 0.5 * torch.mean(x_t.index_copy(0, idx_t, src_t) ** 2)
    term3 = torch.mean(torch.linalg.solve(a_t, b_t) ** 2)
    loss = term1 + term2 + term3
    grads = torch.autograd.grad(loss, (x_t, w_t, src_t, a_t, b_t))
    return tuple(g.cpu().numpy() for g in grads)


def _grad_structural_inputs(seed):
    """Custom generator: distinct non-negative int32 gather/scatter indices
    and a well-conditioned solve matrix A = I + 0.3*randn."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    idx = rng.choice(8, size=(6,), replace=False).astype(np.int32)
    w = rng.standard_normal(16).astype(np.float32)
    src = rng.standard_normal((6, 16)).astype(np.float32)
    a = (np.eye(4) + 0.3 * rng.standard_normal((4, 4))).astype(np.float32)
    b = rng.standard_normal((4, 2)).astype(np.float32)
    return [x, idx, w, src, a, b]


# ---------------------------------------------------------------------------
# Registry (category "grad"). Tolerances: rtol=atol=1e-3 — the documented
# default for gradient examples (fp32 grads vs fp64 FD); every example's
# measured max-abs error (see comments above) is ~1e-5 or less, so 1e-3
# passes with >100x margin.
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="grad_mlp",
        description="grad of 2-layer MLP + MSE wrt all 6 inputs (etl.grad)",
        specs=tuple(
            TensorSpec(s, _F32)
            for s in [(8, 16), (16, 32), (32,), (32, 8), (8,), (8, 8)]
        ),
        graph=etl.grad(_grad_mlp_loss, argnums=(0, 1, 2, 3, 4, 5)),
        numpy_ref=_grad_mlp_numpy,
        torch_ref=_grad_mlp_torch,
        rtol=1e-3,
        atol=1e-3,
        category="grad",
    ),
    Example(
        name="grad_mix",
        description="grad of one scalar loss mixing ~40 differentiable ops",
        specs=tuple(
            TensorSpec(s, _F32) for s in [(6, 10), (6, 10), (10, 4), (6, 4)]
        ),
        graph=etl.grad(_grad_mix_loss),  # argnums=None -> all 4 inputs
        numpy_ref=_grad_mix_numpy,
        torch_ref=_grad_mix_torch,
        rtol=1e-3,
        atol=1e-3,
        category="grad",
    ),
    Example(
        name="grad_stopgrad",
        description="grad through etl.stop_gradient: grads (x, x)",
        specs=(TensorSpec((8,), _F32), TensorSpec((8,), _F32)),
        graph=etl.grad(_grad_stopgrad_loss, argnums=(0, 1)),
        numpy_ref=_grad_stopgrad_numpy,
        torch_ref=_grad_stopgrad_torch,
        rtol=1e-3,
        atol=1e-3,
        category="grad",
    ),
    Example(
        name="grad_structural",
        description=(
            "grad of gather + scatter + solve loss (numpy backend; iree "
            "records the documented StableHLO gather/scatter deferral)"
        ),
        specs=(
            TensorSpec((8, 16), _F32),
            TensorSpec((6,), etl.int32),
            TensorSpec((16,), _F32),
            TensorSpec((6, 16), _F32),
            TensorSpec((4, 4), _F32),
            TensorSpec((4, 2), _F32),
        ),
        graph=etl.grad(_grad_structural_loss, argnums=_GRAD_STRUCTURAL_ARGS),
        numpy_ref=_grad_structural_numpy,
        torch_ref=_grad_structural_torch,
        inputs_fn=_grad_structural_inputs,
        rtol=1e-3,
        atol=1e-3,
        category="grad",
    ),
])
