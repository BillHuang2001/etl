"""Portable-decomposition inlining tests for ``etl.block`` (decorator form).

Covers the binding contract from ``etl/block/CONTEXT.md``:

* ``@etl.block @etl.defn`` registers a portable implementation (name = function
  name; ``has_portable`` True; resolved batching policy ``batching_rule``;
  the decomposition fallbacks land in the transforms registries).
* The numpy backend INLINES the portable decomposition at ``etl.lower`` time
  (graph->graph splice), so the interpreter never sees the ``block_call`` and
  numerical results equal the decomposition's computation.
* Errors: non-defn portables fail at declaration (``BlockError``); a portable
  returning a non-tensor or a shape that mismatches the declared outputs
  fails loudly at lower time (``BackendError``).

Notes:

* Block annotations are eagerly evaluated (``etl.defn``/``etl.block`` do not
  evaluate string annotations), so this module deliberately has NO
  ``from __future__ import annotations`` and multi-output return annotations
  are spelled as literal tuples of ``TensorSpec``.
* The block registry and the transforms rule registries are process-wide:
  every block declared here uses the ``port_`` prefix to stay unique.
* A rank-0 spec cannot accept a rank-1 input (``ShapeError`` at call time),
  so vector-numerics tests declare rank-1 specs with a runtime-dynamic
  ``None`` dim (``TensorSpec((None,), ...)``) — the objective's scalar-spec +
  linspace pairing is unsatisfiable as written.
"""

import numpy as np
import pytest

import etl
from etl.block.errors import BlockError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_blocks(block):
    """Yield every op of a block, nested-region ops first (bottom-up)."""
    for op in block.ops:
        for region in op.regions:
            for nested in region.blocks:
                yield from _walk_blocks(nested)
        yield op


def _count_block_calls(module):
    """Number of ``block_call`` ops anywhere in a (de)serialized ir.Module."""
    count = 0
    for fn in module.functions:
        for block in fn.region.blocks:
            for op in _walk_blocks(block):
                if op.name == "block_call":
                    count += 1
    return count


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Block declarations (module scope, unique ``port_`` names)
# ---------------------------------------------------------------------------


@etl.block
@etl.defn
def port_swish(x: etl.TensorSpec((None,), etl.float32)) -> etl.TensorSpec((None,), etl.float32):
    return etl.sigmoid(x) * x


@etl.defn
def port_swish_fn(x):
    return port_swish(x)


@etl.block
@etl.defn
def port_pair(
    x: etl.TensorSpec((None,), etl.float32),
) -> (etl.TensorSpec((None,), etl.float32), etl.TensorSpec((None,), etl.float32)):
    return etl.sigmoid(x) * x, etl.tanh(x)


@etl.defn
def port_pair_fn(x):
    return port_pair(x)


# Portable returning a non-tensor: the return annotation is omitted and the
# output spec is declared explicitly, so the declaration succeeds and the
# failure surfaces at LOWER time (where the decomposition is inlined).
@etl.block(outputs=[etl.TensorSpec((), etl.float32)])
@etl.defn
def port_badtensor(x: etl.TensorSpec((), etl.float32)):
    return 2.0


@etl.defn
def port_badtensor_fn(x):
    return port_badtensor(x)


# Declared output is a scalar, the portable returns rank 1 (a rank-vs-dim
# variant of the declared-vs-portable mismatch).
@etl.block(outputs=[etl.TensorSpec((), etl.float32)])
@etl.defn
def port_shape_mismatch(x: etl.TensorSpec((2,), etl.float32)):
    return etl.reshape(x, (2,))


@etl.defn
def port_shape_mismatch_fn(x):
    return port_shape_mismatch(x)


# ---------------------------------------------------------------------------
# Inlining + numerics
# ---------------------------------------------------------------------------


def test_decorator_block_inlined_with_correct_numerics():
    """The portable decomposition replaces the block_call and computes it."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    y = etl.evaluate(port_swish_fn, x)
    assert isinstance(y, etl.Tensor)
    assert np.allclose(y.numpy(), _sigmoid_np(x) * x, rtol=1e-6)


def test_decorator_block_attributes():
    """Name = fn name; portable registered; default policy = batching_rule."""
    assert port_swish.name == "port_swish"
    assert port_swish.has_portable is True
    assert port_swish.batching_policy == "batching_rule"


def test_output_specs_derived_from_return_annotation():
    """No declared outputs: output_specs derive from the TensorSpec annotation."""
    (spec,) = port_swish.output_specs
    assert isinstance(spec, etl.TensorSpec)
    assert spec.shape == (None,)
    assert spec.dtype == np.dtype(np.float32)


def test_lowering_inlines_the_block_call():
    """After lower() the serialized payload contains no block_call ops."""
    graph = etl.trace(port_swish_fn, etl.TensorSpec((None,), etl.float32))
    assert _count_block_calls(graph.module) == 1  # the traced graph has it

    lowered = etl.lower(graph)
    module = etl.ir.deserialize_module(lowered.payload)
    assert _count_block_calls(module) == 0


def test_multi_output_portable():
    """A tuple-returning portable inlines into two numerically-correct results."""
    x = np.linspace(-2, 2, 32, dtype=np.float32)
    out = etl.evaluate(port_pair_fn, x)
    assert isinstance(out, tuple) and len(out) == 2
    a, b = out
    assert np.allclose(a.numpy(), _sigmoid_np(x) * x, rtol=1e-6)
    assert np.allclose(b.numpy(), np.tanh(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_factory_portable_must_be_a_defn():
    """A non-defn portable (lambda) fails the declaration with BlockError."""
    with pytest.raises(BlockError, match="must be an etl.defn"):
        etl.block(
            "port_factory_bad",
            inputs=[etl.TensorSpec((), etl.float32)],
            outputs=[etl.TensorSpec((), etl.float32)],
            portable=lambda v: v,
        )


def test_decorator_over_non_defn_raises():
    """Decorating a plain (non-defn) function with @etl.block raises."""
    with pytest.raises(BlockError, match="must be an etl.defn"):

        @etl.block
        def port_decorator_bad(x: etl.TensorSpec((), etl.float32)) -> etl.TensorSpec((), etl.float32):
            return x


def test_portable_returning_non_tensor_fails_at_lower():
    """Trace succeeds; lower inlines the portable and fails loudly.

    The portable returns a Python float: the traced decomposition has zero
    tensor results, so the numpy backend reports a BackendError at lower time
    (the objective's "must return symbolic" BlockError is the transforms-side
    message; the lower-time contract is "produces 0 result(s)").
    """
    graph = etl.trace(port_badtensor_fn, etl.TensorSpec((), etl.float32))
    with pytest.raises(etl.BackendError, match="produces 0 result"):
        etl.lower(graph)


def test_declared_vs_portable_shape_mismatch():
    """Declared scalar output vs rank-1 portable output -> BackendError."""
    graph = etl.trace(port_shape_mismatch_fn, etl.TensorSpec((2,), etl.float32))
    with pytest.raises(etl.BackendError, match="but the block_call declares"):
        etl.lower(graph)
