"""Second gradient-batch op conformance examples (category "op", tag "grad").

Each example bundles a ``@etl.defn`` scalar loss and the ``etl.grad(...)``
``TransformCallable`` as ``example.graph`` (staged by the harness through the
explicit ``lower``/``compile``/``load`` pipeline). ``numpy_ref`` is the
per-element central finite-difference gradient in float64
(:func:`~etl.bench.examples.base.fd_gradient`, ``h=1e-4``) of the same loss,
returning the same structure as ``etl.grad`` (a 1-tuple — every loss here has
exactly one float32 input). Measured max-abs errors (etl fp32 grads vs fp64
central-fd, seed 0) are all <= 1.5e-09 — the default gradient tolerance
``rtol=atol=1e-3`` passes with >6 orders of magnitude margin.

VJP-rule coverage added by this module (extending grad_mix / grad_stopgrad /
grad_structural — see :mod:`etl.bench.examples.grad` for the full coverage
map):

=====================  =======================================================
rule                   example
=====================  =======================================================
erf                    grad_erf        (smooth everywhere — FD-safe)
gelu                   grad_gelu       (exact-erf form)
cumsum                 grad_cumsum     (reverse-scan VJP; axis=1)
tril / triu            grad_tril / grad_triu   (mask-multiply VJPs; triu k=1)
power (FRACTIONAL      grad_power_frac (static exponent 2.5, positive base
  static exponent)                      x+5 > 0 — a new regime vs grad_mix's
                                        power(pos, 2.0))
bitwise_and (zero-vjp grad_bitwise_zero (int-only op reached through
  class)                                cast f32->i32 -> bitwise_and -> cast
                                        back; the zero cotangent proves the
                                        _zero_vjp rule fires, FD flips bounded
                                        by the 1e-8 scale)
=====================  =======================================================

Deferral notes (documented, never silent — see root ``etl/bench/CONTEXT.md``):
  - StableHLO v1 defers export of ``cumsum``, ``tril``, ``triu``, ``erf``
    (no ``stablehlo.erf``; ``gelu`` decomposes via erf). These five examples
    are therefore NUMPY-BACKEND-ONLY: on compiler backends they fail with an
    explicit ``BackendError`` at ``lower()`` time, recorded per-example in
    the report (the same documented deferral class as micro's ``cumsum``).
  - ``conv`` VJP remains a v1 deferral — probed at dev time:
    ``TransformError: grad/vjp: no VJP rule for op 'conv' — the IR has no
    transposed-convolution op (v1 gap)``. Conv is deliberately NOT registered
    here.
  - ``remainder`` in its identity regime is already covered by grad_mix
    (same ct-passthrough rule) — not duplicated here.
  - No ``torch_ref``: the float64 central-FD gradient IS the reference of
    record for these single-op losses (torch autograd would re-derive the
    same mathematics); torch is not installed in the dev environment, and
    ``torch_ref`` is optional by the :class:`Example` contract. The
    torch-optionality binding holds — nothing here imports torch.
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .base import Example, _F32, fd_gradient, gelu_numpy, register_all

# --- grad_erf ----------------------------------------------------------------
# loss = mean(erf(x)); d/dx = (2/sqrt(pi)) * exp(-x^2) — smooth everywhere,
# fully FD-safe. VJP rule: erf' = 2/sqrt(pi) * exp(-x^2).
# Measured (dev, seed 0): etl vs numpy-fd max-abs = 1.4e-09.


@defn
def _grad_erf_loss(x):
    return etl.mean(etl.erf(x))


def _grad_erf_value(inputs, frozen):
    (x,) = inputs
    erf = np.frompyfunc(math.erf, 1, 1)(x).astype(np.float64)
    return float(np.mean(erf))


def _grad_erf_numpy(inputs):
    return tuple(fd_gradient(_grad_erf_value, inputs))


# --- grad_gelu ---------------------------------------------------------------
# loss = mean(gelu(x)) with the exact-erf form (0.5*x*(1+erf(x/sqrt(2))) —
# the etl gelu definition; gelu_numpy mirrors it via math.erf, matching the
# numpy backend kernel). FD-safe everywhere (smooth erf-based form).
# Measured (dev, seed 0): etl vs numpy-fd max-abs = 8.9e-10.


@defn
def _grad_gelu_loss(x):
    return etl.mean(etl.gelu(x))


def _grad_gelu_value(inputs, frozen):
    (x,) = inputs
    return float(np.mean(gelu_numpy(x)))


def _grad_gelu_numpy(inputs):
    return tuple(fd_gradient(_grad_gelu_value, inputs))


# --- grad_cumsum -------------------------------------------------------------
# loss = mean(cumsum(x, axis=1)) — the VJP is a reverse cumulative sum along
# the same axis; linear in x, so FD is exact. StableHLO v1 defers cumsum
# export (documented) — numpy backend only.
# Measured (dev, seed 0): etl vs numpy-fd max-abs = 1.1e-12.


@defn
def _grad_cumsum_loss(x):
    return etl.mean(etl.cumsum(x, axis=1))


def _grad_cumsum_value(inputs, frozen):
    (x,) = inputs
    return float(np.mean(np.cumsum(x, axis=1)))


def _grad_cumsum_numpy(inputs):
    return tuple(fd_gradient(_grad_cumsum_value, inputs))


# --- grad_tril / grad_triu ---------------------------------------------------
# loss = sum(tril(x)) (resp. sum(triu(x, k=1))) — the VJP is a mask multiply
# (strictly-upper triangle for k=1: the diagonal gets gradient 0, exercised
# identically by etl, FD, and the numpy ref). Linear in x, so FD is exact.
# StableHLO v1 defers tril/triu export (documented) — numpy backend only.
# Measured (dev, seed 0): etl vs numpy-fd max-abs = 2.3e-12 (both).


@defn
def _grad_tril_loss(x):
    return etl.sum(etl.tril(x))


def _grad_tril_value(inputs, frozen):
    (x,) = inputs
    return float(np.sum(np.tril(x)))


def _grad_tril_numpy(inputs):
    return tuple(fd_gradient(_grad_tril_value, inputs))


@defn
def _grad_triu_loss(x):
    return etl.sum(etl.triu(x, k=1))


def _grad_triu_value(inputs, frozen):
    (x,) = inputs
    return float(np.sum(np.triu(x, k=1)))


def _grad_triu_numpy(inputs):
    return tuple(fd_gradient(_grad_triu_value, inputs))


# --- grad_power_frac ---------------------------------------------------------
# loss = mean((x+5)**2.5) — power with a FRACTIONAL static exponent (a new
# regime vs grad_mix's power(pos, 2.0)). FD-safe: x ~ N(0,1) keeps the base
# x+5 > 0 always (min over 128 draws is ~2, far from 0), so the derivative
# 2.5*(x+5)**1.5 is finite everywhere.
# Measured (dev, seed 0): etl vs numpy-fd max-abs = 4.2e-08.


@defn
def _grad_power_frac_loss(x):
    return etl.mean(etl.power(etl.add(x, 5.0), 2.5))


def _grad_power_frac_value(inputs, frozen):
    (x,) = inputs
    return float(np.mean((x + 5.0) ** 2.5))


def _grad_power_frac_numpy(inputs):
    return tuple(fd_gradient(_grad_power_frac_value, inputs))


# --- grad_bitwise_zero -------------------------------------------------------
# loss = mean(x*x) + 1e-8 * sum(cast(bitwise_and(cast(x, int32), 3), float32)).
# The bitwise term is an int-only op reached through casts: its _zero_vjp
# rule fires at trace time and yields a zero cotangent that propagates back
# through BOTH casts to x — the gradient of the term is exactly 0, so
# d/dx(loss) = 2*mean(x). The 1e-8 scale (grad_mix's comparison-term recipe)
# bounds any FD boundary flip at the int-cast truncation points to
# <= 1e-8/(2h) = 5e-5 — far below rtol=atol=1e-3 (seed 0 measured 5.7e-13).
# int32 values are exactly representable in float64, so the FD value function
# reproduces the graph's truncation bit-for-bit.


@defn
def _grad_bitwise_zero_loss(x):
    xi = etl.cast(x, etl.int32)
    z = etl.cast(etl.bitwise_and(xi, 3), etl.float32)
    return etl.add(etl.mean(etl.square(x)), etl.multiply(etl.sum(z), 1e-8))


def _grad_bitwise_zero_value(inputs, frozen):
    (x,) = inputs
    z = (x.astype(np.int32) & 3).astype(np.float64)
    return float(np.mean(x * x) + 1e-8 * np.sum(z))


def _grad_bitwise_zero_numpy(inputs):
    return tuple(fd_gradient(_grad_bitwise_zero_value, inputs))


# ---------------------------------------------------------------------------
# Registry (category "op", tag "grad"). Tolerances: rtol=atol=1e-3 — the
# documented default for gradient examples (fp32 grads vs fp64 FD); every
# example's measured max-abs error is <= 1.5e-09 (see comments above), so
# 1e-3 passes with >6 orders of magnitude margin.
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="grad_erf",
        description="grad of mean(erf(x)) (smooth; numpy backend — StableHLO defers erf export)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_erf_loss),
        numpy_ref=_grad_erf_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_gelu",
        description="grad of mean(gelu(x)) (exact-erf form; numpy backend — StableHLO defers gelu export)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_gelu_loss),
        numpy_ref=_grad_gelu_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_cumsum",
        description="grad of mean(cumsum(x, axis=1)) (numpy backend — StableHLO defers cumsum export)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_cumsum_loss),
        numpy_ref=_grad_cumsum_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_tril",
        description="grad of sum(tril(x)) (numpy backend — StableHLO defers tril export)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_tril_loss),
        numpy_ref=_grad_tril_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_triu",
        description="grad of sum(triu(x, k=1)) (numpy backend — StableHLO defers triu export)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_triu_loss),
        numpy_ref=_grad_triu_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_power_frac",
        description="grad of mean((x+5)**2.5) — fractional static exponent, positive base",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_power_frac_loss),
        numpy_ref=_grad_power_frac_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
    Example(
        name="grad_bitwise_zero",
        description="zero-grad bitwise_and term (via int casts, scaled 1e-8): grad = 2*mean(x)",
        specs=(TensorSpec((8, 16), _F32),),
        graph=etl.grad(_grad_bitwise_zero_loss),
        numpy_ref=_grad_bitwise_zero_numpy,
        rtol=1e-3,
        atol=1e-3,
        category="op",
        tags=("grad",),
    ),
])
