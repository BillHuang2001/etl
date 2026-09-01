"""Regression tests for the fixed block portable-decomposition VJP fallback.

Context: commit 1a88219 fixed ``etl/block/rules.py::_portable_vjp_rule`` —
the fallback used to seed its local reverse sweep on the block_call's
PRE-inline result ids (which are never inserted into the builder), so it
returned all-``ZeroTangent`` whenever a real cotangent existed. The
``_repair_portable_vjp`` workaround in ``etl/transforms/autodiff.py`` that
compensated was deleted. The fixed rule seeds on the POST-inline
decomposition outputs and guards output-count mismatches with
``TransformError`` (never guessing).

These tests pin the fixed contract for blocks declared with ONLY a portable
decomposition (decorator form, no explicit batching/jvp/vjp rules, default
policy resolution ``batching_rule``):

* correct NONZERO input cotangents through ``etl.grad`` / ``etl.vjp`` when a
  real cotangent flows, including non-scalar-output cases (elementwise
  ``y = x * w`` and a small matmul, grads wrt ALL inputs checked against a
  numpy reference) and a nested ordinary-op chain (``tanh(x^2) + sigmoid(x)``
  — no constants);
* the all-``ZeroTangent`` short-circuit (a path that does not influence the
  loss → zeros) and the per-entry zero-skip for multi-output blocks;
* ``etl.jvp`` derived from the fixed vjp fallback;
* the output-count guard (portable returning a different number of outputs
  than the declared results → ``TransformError``).

Notes:

* Same eager-annotation caveat as ``test_portable.py``: no
  ``from __future__ import annotations`` in this module.
* The registries are process-wide: every block declared here uses the
  ``reg_`` prefix to stay unique.
* All blocks are decorator-form portable-only declarations — the fallback
  rules registered at declaration time are exactly what the transforms
  resolve to (nothing explicit overrides them).
* ``# BUG(etl)`` on ``test_grad_portable_containing_constant_op``: the
  fixed local sweep still does not skip ``constant`` ops (see the marker).
"""

import numpy as np
import pytest

import etl

DIM = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(graph, *args):
    """Explicit pipeline: lower -> compile -> load -> run (returns structure)."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


# ---------------------------------------------------------------------------
# Block declarations (module scope, unique ``reg_`` names — portable-only)
# ---------------------------------------------------------------------------


# Non-scalar (vector) output, two inputs: y = x * w.
@etl.block
@etl.defn
def reg_mul(
    x: etl.TensorSpec((DIM,), etl.float64),
    w: etl.TensorSpec((DIM,), etl.float64),
) -> etl.TensorSpec((DIM,), etl.float64):
    return etl.multiply(x, w)


@etl.defn
def reg_mul_loss(x, w):
    return etl.sum(reg_mul(x, w))


@etl.defn
def reg_mul_fn(x, w):
    return reg_mul(x, w)


# Non-scalar (matrix) output, two inputs: y = a @ b.
@etl.block
@etl.defn
def reg_mm(
    a: etl.TensorSpec((DIM, DIM), etl.float64),
    b: etl.TensorSpec((DIM, DIM), etl.float64),
) -> etl.TensorSpec((DIM, DIM), etl.float64):
    return etl.dot(a, b)


@etl.defn
def reg_mm_loss(a, b):
    return etl.sum(reg_mm(a, b))


# Nested ordinary-op chain, no constants: y = tanh(x^2) + sigmoid(x).
@etl.block
@etl.defn
def reg_chain(x: etl.TensorSpec((DIM,), etl.float64)) -> etl.TensorSpec((DIM,), etl.float64):
    return etl.tanh(etl.square(x)) + etl.sigmoid(x)


@etl.defn
def reg_chain_loss(x):
    return etl.sum(reg_chain(x))


@etl.defn
def reg_chain_fn(x):
    return reg_chain(x)


# Multi-output portable: grad through only the first output.
@etl.block
@etl.defn
def reg_pair(
    x: etl.TensorSpec((DIM,), etl.float64),
) -> (etl.TensorSpec((DIM,), etl.float64), etl.TensorSpec((DIM,), etl.float64)):
    return etl.square(x), etl.tanh(x)


@etl.defn
def reg_pair_loss(x):
    return etl.sum(reg_pair(x)[0])


# A portable whose body contains a constant op (2.0 traces to `constant`).
@etl.block
@etl.defn
def reg_const(x: etl.TensorSpec((DIM,), etl.float64)) -> etl.TensorSpec((DIM,), etl.float64):
    return etl.multiply(x, 2.0)


@etl.defn
def reg_const_loss(x):
    return etl.sum(reg_const(x))


# Portable returning MORE outputs than the declared results -> the fallback's
# output-count guard must raise TransformError (never guess).
@etl.defn
def reg_two_out(x):
    return x, etl.multiply(x, 2.0)


reg_mismatch = etl.block(
    "reg_mismatch",
    inputs=[etl.TensorSpec((DIM,), etl.float64)],
    outputs=[etl.TensorSpec((DIM,), etl.float64)],  # declares 1; portable returns 2
    portable=reg_two_out,
)


@etl.defn
def reg_mismatch_loss(x):
    return etl.sum(reg_mismatch(x))


# ---------------------------------------------------------------------------
# Correct NONZERO input cotangents (post-inline seeding)
# ---------------------------------------------------------------------------


def test_blocks_are_portable_only_declarations():
    """Default policy resolution: portable-only blocks with no explicit rules.

    The decorator form registers the portable decomposition and the
    decomposition fallbacks (derivative fallback always) under ``block:<name>``
    — exactly what the transforms resolve to, since nothing explicit overrides.
    """
    assert reg_mul.has_portable is True
    assert reg_mul.batching_policy == "batching_rule"
    assert reg_chain.has_portable is True
    assert reg_chain.batching_policy == "batching_rule"
    # The fallback vjp rule was registered at declaration time.
    assert callable(etl.transforms.vjp_rules["block:reg_mul"])
    assert callable(etl.transforms.vjp_rules["block:reg_chain"])


def test_grad_nonzero_cotangents_two_input_portable():
    """d/d(x,w) sum(x*w) == (w, x) — the all-ZeroTangent bug returned zeros."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    w = np.array([2.0, -1.0, 0.5, 4.0])

    tf = etl.grad(reg_mul_loss, argnums=(0, 1))
    graph = tf(
        etl.TensorSpec((DIM,), etl.float64), etl.TensorSpec((DIM,), etl.float64)
    )
    grad_x, grad_w = _run(graph, x, w)
    assert np.any(grad_x.numpy() != 0.0) and np.any(grad_w.numpy() != 0.0)
    np.testing.assert_allclose(grad_x.numpy(), w, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_w.numpy(), x, rtol=1e-10, atol=1e-10)


def test_vjp_two_input_portable_nontrivial_cotangent():
    """vjp of y = x*w with a NON-ONES cotangent: (ct*w, ct*x)."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    w = np.array([2.0, -1.0, 0.5, 4.0])
    ct = np.array([0.1, -0.2, 0.3, 0.4])

    tf = etl.vjp(reg_mul_fn, etl.TensorSpec((DIM,), etl.float64))
    graph = tf(
        etl.TensorSpec((DIM,), etl.float64), etl.TensorSpec((DIM,), etl.float64)
    )
    primal, (in_ct_x, in_ct_w) = _run(graph, (x, w), (ct,))
    np.testing.assert_allclose(primal.numpy(), x * w, rtol=1e-10, atol=1e-10)
    assert np.any(in_ct_x.numpy() != 0.0) and np.any(in_ct_w.numpy() != 0.0)
    np.testing.assert_allclose(in_ct_x.numpy(), ct * w, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(in_ct_w.numpy(), ct * x, rtol=1e-10, atol=1e-10)


def test_grad_matmul_portable_both_operands():
    """d/d(a,b) sum(a@b) == (ones@b^T, a^T@ones) — matrix (non-scalar) output."""
    rng = np.random.RandomState(0)
    a = rng.randn(DIM, DIM)
    b = rng.randn(DIM, DIM)

    tf = etl.grad(reg_mm_loss, argnums=(0, 1))
    graph = tf(
        etl.TensorSpec((DIM, DIM), etl.float64),
        etl.TensorSpec((DIM, DIM), etl.float64),
    )
    grad_a, grad_b = _run(graph, a, b)
    ones = np.ones((DIM, DIM))
    assert np.any(grad_a.numpy() != 0.0) and np.any(grad_b.numpy() != 0.0)
    np.testing.assert_allclose(grad_a.numpy(), ones @ b.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_b.numpy(), a.T @ ones, rtol=1e-10, atol=1e-10)


def test_grad_nested_chain_portable():
    """d/dx sum(tanh(x^2) + sigmoid(x)) == 2x(1-tanh^2(x^2)) + s(1-s)."""
    x = np.array([0.5, 1.0, 2.0, 3.0])

    graph = etl.grad(reg_chain_loss)(etl.TensorSpec((DIM,), etl.float64))
    (grad_out,) = _run(graph, x)
    s = 1.0 / (1.0 + np.exp(-x))
    expected = 2.0 * x * (1.0 - np.tanh(x**2) ** 2) + s * (1.0 - s)
    assert np.any(grad_out.numpy() != 0.0)
    np.testing.assert_allclose(grad_out.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_vjp_nested_chain_portable_nontrivial_cotangent():
    """vjp of the chain with a non-ones cotangent == ct * analytic gradient."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    ct = np.array([0.1, -0.2, 0.3, 0.4])

    tf = etl.vjp(reg_chain_fn, etl.TensorSpec((DIM,), etl.float64))
    graph = tf(etl.TensorSpec((DIM,), etl.float64))
    primal, (in_ct,) = _run(graph, (x,), (ct,))
    s = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(
        primal.numpy(), np.tanh(x**2) + s, rtol=1e-10, atol=1e-10
    )
    expected = ct * (2.0 * x * (1.0 - np.tanh(x**2) ** 2) + s * (1.0 - s))
    assert np.any(in_ct.numpy() != 0.0)
    np.testing.assert_allclose(in_ct.numpy(), expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Zero-cotangent short-circuits
# ---------------------------------------------------------------------------


def test_grad_zero_cotangent_path_short_circuits_to_zeros():
    """A path that does not influence the loss -> exact zeros, no inlining."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    w = np.array([2.0, -1.0, 0.5, 4.0])

    @etl.defn
    def reg_zero_loss(x, w):
        return etl.sum(etl.stop_gradient(reg_mul(x, w)))

    tf = etl.grad(reg_zero_loss, argnums=(0, 1))
    graph = tf(
        etl.TensorSpec((DIM,), etl.float64), etl.TensorSpec((DIM,), etl.float64)
    )
    grad_x, grad_w = _run(graph, x, w)
    np.testing.assert_array_equal(grad_x.numpy(), np.zeros(DIM))
    np.testing.assert_array_equal(grad_w.numpy(), np.zeros(DIM))


def test_grad_multi_output_portable_mixed_cotangents():
    """Only the loss-feeding output's cotangent is real; the other is zero.

    The rule must inline and sweep for the real cotangent while skipping the
    zero entry: d/dx sum(pair(x)[0]) == 2x (the tanh output contributes 0).
    """
    x = np.array([0.5, 1.0, 2.0, 3.0])

    graph = etl.grad(reg_pair_loss)(etl.TensorSpec((DIM,), etl.float64))
    (grad_out,) = _run(graph, x)
    assert np.any(grad_out.numpy() != 0.0)
    np.testing.assert_allclose(grad_out.numpy(), 2.0 * x, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# jvp derived from the fixed vjp fallback
# ---------------------------------------------------------------------------


def test_jvp_derived_from_vjp_fallback_two_input_portable():
    """jvp of y = x*w (no explicit jvp rule): tangent == dx*w + x*dw."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    w = np.array([2.0, -1.0, 0.5, 4.0])
    dx = np.array([1.0, 1.0, 1.0, 1.0])
    dw = np.array([0.0, 1.0, 0.0, 1.0])

    spec = etl.TensorSpec((DIM,), etl.float64)
    tf = etl.jvp(reg_mul_fn, (spec, spec))
    graph = tf(spec, spec)
    primal, (tangent,) = _run(graph, (x, w), (dx, dw))
    np.testing.assert_allclose(primal.numpy(), x * w, rtol=1e-10, atol=1e-10)
    assert np.any(tangent.numpy() != 0.0)
    np.testing.assert_allclose(
        tangent.numpy(), dx * w + x * dw, rtol=1e-10, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Output-count guard
# ---------------------------------------------------------------------------


def test_portable_output_count_mismatch_raises_transform_error():
    """Portable returning 2 outputs for 1 declared result -> TransformError."""
    tf = etl.grad(reg_mismatch_loss)
    with pytest.raises(
        etl.TransformError, match="returned 2 outputs for 1 declared results"
    ):
        tf(etl.TensorSpec((DIM,), etl.float64))


# ---------------------------------------------------------------------------
# Constants inside the portable (see module docstring)
# ---------------------------------------------------------------------------


# BUG(etl): a portable decomposition whose body contains a `constant` op (any
# literal or etl.constant value feeding an inlined op) fails grad/vjp with
# "grad/vjp: no VJP rule for op 'constant'": the local reverse sweep in
# etl/block/rules.py::_portable_vjp_rule looks up a rule for EVERY inlined op
# whose result carries a cotangent, but `constant` has NO registry entry by
# design — the main sweep in etl/transforms/autodiff.py skips
# `proxy.name == "constant"` (zero operands, nothing to propagate backward).
# The decomposition is a graph of ordinary ops; constants in it must
# differentiate fine (post-inline seeding), like the batching fallback's
# "constant ops are unmapped" handling. Minimal repro: block with portable
# `etl.multiply(x, 2.0)` — the 2.0 traces to a `constant` op — and
# `grad(sum(block(x)))` raises TransformError instead of returning 2.0.
# Do NOT skip/xfail/weaken it.
def test_grad_portable_containing_constant_op():
    """A constant in the portable must not break grad: d/dx sum(2x) == 2."""
    x = np.array([0.5, 1.0, 2.0, 3.0])

    graph = etl.grad(reg_const_loss)(etl.TensorSpec((DIM,), etl.float64))
    (grad_out,) = _run(graph, x)
    assert np.any(grad_out.numpy() != 0.0)
    np.testing.assert_allclose(grad_out.numpy(), np.full(DIM, 2.0), rtol=1e-10, atol=1e-10)
