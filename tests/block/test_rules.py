"""Batching / jvp / vjp rule registration and fallbacks for ``etl.block``.

Covers the rule-bridge contract from ``etl/block/CONTEXT.md`` and the frozen
rule signatures from ``etl/transforms/CONTEXT.md``:

* ``BlockOp.batching_rule / jvp_rule / vjp_rule(fn)`` register under the
  ``block:<name>`` keys of the PUBLIC ``etl.transforms`` registries and return
  ``fn``; non-callable arguments raise ``BlockError`` (match "callable").
* batching rule: ``rule(op, operands, axes) -> (new_values, new_axes)``;
  vjp rule: ``rule(op, cotangents, primals) -> input_cotangents``; jvp rule:
  ``rule(op, tangents) -> output_tangents``. Rules build replacement ops over
  the values the machinery passes (wrapped as ``etl.SymbolicTensor``).
* Without an explicit rule: the portable decomposition is the fallback for
  vmap/grad/jvp/vjp; with NO rule and NO portable the transforms raise
  ``TransformError`` naming the block (never a silent fallback). An explicit
  rule always wins over the decomposition fallback.

Notes:

* Same eager-annotation caveat as ``test_portable.py``: no
  ``from __future__ import annotations`` in this module.
* The registries are process-wide: every block declared here uses the
  ``rule_`` prefix to stay unique.
* vectorize/vmap cannot map axis 0 of a rank-0 input (v1), so vmap tests use
  rank-1 blocks with a batch of rows, e.g. spec ``(4,)`` batched to ``(8, 4)``.
* A block whose graph is RUN must be executable: rule-only factory blocks get
  a numpy impl via ``@blk.impl("numpy")`` (execution-side only; it plays no
  role in batching/derivative resolution).
"""

import numpy as np
import pytest

import etl
from etl import transforms
from etl.block.errors import BlockError
from etl.transforms.autodiff import ZeroTangent

BATCH = 8
DIM = 4


def _sym(value):
    """Wrap an ir.Value as a SymbolicTensor so etl.ops can build on it."""
    return etl.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def _run(graph, *args):
    """Explicit pipeline: lower -> compile -> load -> run (returns structure)."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *args)


def _swish(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * x


def _swish_deriv(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


# ---------------------------------------------------------------------------
# Block + rule declarations (module scope, unique ``rule_`` names)
# ---------------------------------------------------------------------------

# 1. Portable-only block (decomposition fallback pre-registered at decl time).
@etl.block
@etl.defn
def rule_swish(x: etl.TensorSpec((DIM,), etl.float32)) -> etl.TensorSpec((DIM,), etl.float32):
    return etl.sigmoid(x) * x


@etl.defn
def rule_swish_fn(x):
    return rule_swish(x)


# 2. Factory block, no portable -> policy "unsupported"; an EXPLICIT batching
#    rule must still win (a registered rule always beats the policy).
rule_explicit = etl.block(
    "rule_explicit",
    inputs=[etl.TensorSpec((DIM,), etl.float32)],
    outputs=[etl.TensorSpec((DIM,), etl.float32)],
)


@rule_explicit.batching_rule
def rule_explicit_batch(op, operands, axes):
    x = operands[0]
    y = etl.multiply(_sym(x), 2)
    return (y.value,), (axes[0],)


@etl.defn
def rule_explicit_fn(x):
    return rule_explicit(x)


# 3. Portable computes 2*x; an explicit rule computes 3*x and must win.
@etl.block
@etl.defn
def rule_override(x: etl.TensorSpec((DIM,), etl.float32)) -> etl.TensorSpec((DIM,), etl.float32):
    return etl.multiply(x, 2)


@rule_override.batching_rule
def rule_override_batch(op, operands, axes):
    x = operands[0]
    y = etl.multiply(_sym(x), 3)
    return (y.value,), (axes[0],)


@etl.defn
def rule_override_fn(x):
    return rule_override(x)


# 4. "double" block: explicit jvp rule (+ numpy impl so the primal runs).
rule_jvp = etl.block(
    "rule_jvp",
    inputs=[etl.TensorSpec((DIM,), etl.float32)],
    outputs=[etl.TensorSpec((DIM,), etl.float32)],
)


@rule_jvp.impl("numpy")
def _np_rule_jvp(x):
    return 2 * x


@rule_jvp.jvp_rule
def rule_jvp_forward(op, tangents):
    (t,) = tangents
    if t is None or isinstance(t, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(_sym(t), 2).value,)


@etl.defn
def rule_jvp_fn(x):
    return rule_jvp(x)


# 5. "double" block: explicit vjp rule (+ numpy impl so the primal runs).
rule_vjp = etl.block(
    "rule_vjp",
    inputs=[etl.TensorSpec((DIM,), etl.float32)],
    outputs=[etl.TensorSpec((DIM,), etl.float32)],
)


@rule_vjp.impl("numpy")
def _np_rule_vjp(x):
    return 2 * x


@rule_vjp.vjp_rule
def rule_vjp_backward(op, cotangents, primals):
    (ct,) = cotangents
    if ct is None or isinstance(ct, ZeroTangent):
        return (ZeroTangent(),)
    return (etl.multiply(_sym(ct), 2).value,)


@etl.defn
def rule_vjp_loss(x):
    return etl.sum(rule_vjp(x))


# 6. Portable-only swish (diff fallback pre-registered; no explicit rules).
@etl.block
@etl.defn
def rule_portgrad(x: etl.TensorSpec((DIM,), etl.float32)) -> etl.TensorSpec((DIM,), etl.float32):
    return etl.sigmoid(x) * x


@etl.defn
def rule_portgrad_loss(x):
    return etl.sum(rule_portgrad(x))


# 7. No portable, no rules -> transforms must raise TransformError.
rule_none = etl.block(
    "rule_none",
    inputs=[etl.TensorSpec((DIM,), etl.float32)],
    outputs=[etl.TensorSpec((DIM,), etl.float32)],
)


@etl.defn
def rule_none_fn(x):
    return rule_none(x)


@etl.defn
def rule_none_loss(x):
    return etl.sum(rule_none(x))


# 9. elementwise policy: batch dims pass through — safe WITHOUT a rule.
rule_ew = etl.block(
    "rule_ew",
    inputs=[etl.TensorSpec((DIM,), etl.float32)],
    outputs=[etl.TensorSpec((DIM,), etl.float32)],
    batching="elementwise",
)


@rule_ew.impl("numpy")
def _np_rule_ew(x):
    return x


@etl.defn
def rule_ew_fn(x):
    return rule_ew(x)


# ---------------------------------------------------------------------------
# vmap: portable fallback + explicit rules
# ---------------------------------------------------------------------------


def _batched_x():
    return np.linspace(-1, 1, BATCH * DIM, dtype=np.float32).reshape(BATCH, DIM)


def test_vmap_via_portable_fallback():
    """vmap of a block with only a portable == per-row unvectorized run."""
    # BUG(etl): vectorize looks batching rules up under the RAW op name
    # ("block_call") instead of the "block:<name>" namespace where etl.block
    # registers them (batching_rules["block:rule_swish"] IS present, as is
    # the pre-registered decomposition fallback) — so vmap on ANY block_call
    # raises TransformError("no batching rule for op 'block_call'").
    tf = etl.vmap(rule_swish_fn)
    graph = tf(etl.TensorSpec((BATCH, DIM), etl.float32))
    out = _run(graph, _batched_x())

    # Reference: the unvectorized computation, row by row (build once, run
    # per row — the documented explicit form of etl.evaluate).
    exe = etl.build(rule_swish_fn, etl.TensorSpec((DIM,), etl.float32))
    reference = np.stack([etl.run(exe, row).numpy() for row in _batched_x()])
    assert np.allclose(out.numpy(), reference, rtol=1e-6)
    assert np.allclose(out.numpy(), _swish(_batched_x()), rtol=1e-6)


def test_explicit_batching_rule_registers_and_wins_over_unsupported_policy():
    """Rule lands in the public registry; explicit rule beats the policy."""
    assert rule_explicit.batching_policy == "unsupported"
    assert transforms.batching_rules["block:rule_explicit"] is rule_explicit_batch

    x = _batched_x()
    # BUG(etl): same block_call dispatch gap as test_vmap_via_portable_fallback
    # — the explicit rule exists but vectorize never looks it up.
    graph = etl.vmap(rule_explicit_fn)(etl.TensorSpec((BATCH, DIM), etl.float32))
    out = _run(graph, x)
    assert np.allclose(out.numpy(), 2 * x)


def test_explicit_rule_overrides_portable_decomposition():
    """A registered rule always wins over the portable fallback."""
    # The portable computes 2*x (verifiable through the normal lower path).
    x4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(etl.evaluate(rule_override_fn, x4).numpy(), 2 * x4)

    x = _batched_x()
    # BUG(etl): same block_call dispatch gap — the explicit rule computing
    # 3*x exists but vectorize raises before consulting any block:<name> rule.
    graph = etl.vmap(rule_override_fn)(etl.TensorSpec((BATCH, DIM), etl.float32))
    out = _run(graph, x)
    assert np.allclose(out.numpy(), 3 * x)


def test_vmap_without_rule_raises_transform_error_naming_the_block():
    """No rule + no portable -> TransformError at the tf(spec) call."""
    tf = etl.vmap(rule_none_fn)
    # The rule lookup happens during vectorize, i.e. when calling tf(...).
    # BUG(etl): the raised TransformError names the raw op 'block_call'
    # instead of the block ('rule_none') because vectorize never maps
    # block_call ops to their block:<name> registry key.
    with pytest.raises(etl.TransformError, match="rule_none"):
        tf(etl.TensorSpec((BATCH, DIM), etl.float32))


def test_elementwise_policy_passes_batch_dims_through():
    """elementwise policy: batch dims pass through safely without a rule."""
    x = _batched_x()
    # BUG(etl): transforms never honor the batching policy — vectorize raises
    # TransformError("no batching rule for op 'block_call'") before any
    # policy handling (there is none in transforms). The policy contract says
    # an elementwise block is safe to vmap without a rule.
    graph = etl.vmap(rule_ew_fn)(etl.TensorSpec((BATCH, DIM), etl.float32))
    out = _run(graph, x)
    assert np.allclose(out.numpy(), x)


# ---------------------------------------------------------------------------
# AD rules: jvp / vjp / grad
# ---------------------------------------------------------------------------


def test_jvp_rule_forward_mode():
    """Explicit jvp rule: primal == block semantics, tangent == 2*tangent."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dx = np.array([0.5, -0.25, 1.0, 2.0], dtype=np.float32)

    tf = etl.jvp(rule_jvp_fn, etl.TensorSpec((DIM,), etl.float32))
    graph = tf(etl.TensorSpec((DIM,), etl.float32))
    # Result graph inputs: primal inputs followed by tangent inputs;
    # outputs: (primal_outputs, tangent_outputs).
    out = _run(graph, (x,), (dx,))
    primal, (tangent,) = out
    assert np.allclose(primal.numpy(), 2 * x)
    assert np.allclose(tangent.numpy(), 2 * dx)


def test_vjp_rule_reverse_mode_grad():
    """Explicit vjp rule drives grad: d/dx sum(2x) == 2 everywhere."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    # The primal runs through the numpy impl: sum(2x) == 20.
    assert np.allclose(etl.evaluate(rule_vjp_loss, x).numpy(), np.float32(20.0))

    tf = etl.grad(rule_vjp_loss)
    graph = tf(etl.TensorSpec((DIM,), etl.float32))
    (grad_out,) = _run(graph, x)
    assert np.allclose(grad_out.numpy(), np.full(DIM, 2.0))


def test_grad_via_portable_decomposition():
    """No explicit rule: grad inlines the portable and differentiates it."""
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    expected = _swish_deriv(x)

    graph = etl.grad(rule_portgrad_loss)(etl.TensorSpec((DIM,), etl.float32))
    # BUG(etl): the portable vjp fallback (block/rules.py) seeds the incoming
    # cotangent on the block_call RESULT values but the inlined decomposition
    # produces NEW values — the local reverse sweep never sees the seed, so
    # grad-via-portable returns all zeros instead of the analytic derivative.
    (grad_out,) = _run(graph, x)
    assert np.allclose(grad_out.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_jvp_derived_from_portable_vjp_fallback():
    """No explicit jvp rule: jvp derives from the portable vjp fallback."""
    x = np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    dx = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    expected = _swish_deriv(x) * dx

    tf = etl.jvp(rule_portgrad, etl.TensorSpec((DIM,), etl.float32))
    # BUG(etl): transforms never derives jvp from the vjp rule (block/op.py
    # documents "when absent, transforms derives jvp from the vjp rule" but
    # autodiff.require_jvp_rule consults only jvp_rules) — jvp of a
    # portable-only block raises TransformError("no JVP rule for op
    # 'block:rule_portgrad'").
    graph = tf(etl.TensorSpec((DIM,), etl.float32))
    out = _run(graph, (x,), (dx,))
    primal, (tangent,) = out
    assert np.allclose(primal.numpy(), _swish(x), rtol=1e-6)
    assert np.allclose(tangent.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_grad_without_rule_raises_transform_error_naming_the_block():
    """No rule + no portable -> grad raises TransformError at tf(spec)."""
    tf = etl.grad(rule_none_loss)
    with pytest.raises(etl.TransformError, match="rule_none"):
        tf(etl.TensorSpec((DIM,), etl.float32))


# ---------------------------------------------------------------------------
# Registration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["batching_rule", "jvp_rule", "vjp_rule"])
def test_non_callable_rule_registration_raises(method):
    """Registering a non-callable rule raises BlockError (match "callable")."""
    with pytest.raises(BlockError, match="callable"):
        getattr(rule_none, method)(42)
