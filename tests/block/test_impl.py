"""Backend impl registration + execution for `etl.block` custom ops.

Covers the finalized v1 numpy impl call convention
(`etl/backends/numpy/kernels/custom.py`)::

    impl(*numpy_arrays, **static_args) -> ndarray | tuple/list of ndarrays

- Operands arrive as numpy arrays.
- `static_args` is the op's JSON-safe attribute payload — a dict of
  name -> ``{"kind": ..., "value": ...}`` (NOT decoded Python values).
- The impl return is validated against the block_call's declared
  `result_specs` (count, dtype exactly, shape) before wrapping as Tensors.
- Blocks with a registered numpy impl are KEPT at lower time and dispatched
  by the interpreter kernel at run time (portables are inlined instead).

Block names are prefixed ``impl_`` because the block registry is global and
process-wide — a name can only ever be declared once per process.
"""

import numpy as np
import pytest

import etl
from etl.block import block

# ---------------------------------------------------------------------------
# Declarations (declaration IS registration — module level, unique names)
# ---------------------------------------------------------------------------

impl_double = block(
    "impl_double",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
)


@impl_double.impl("numpy")
def double_impl(x):
    """The registered numpy impl — must return a numpy array."""
    return x * 2.0


def test_basic_impl_execution():
    @etl.defn
    def f(x):
        return impl_double(x)

    x = np.arange(4, dtype=np.float32)
    y = etl.evaluate(f, x)

    assert isinstance(y, etl.Tensor)
    assert y.shape == (4,)
    assert y.dtype == etl.float32
    assert np.allclose(y.numpy(), x * 2.0)


def test_impl_registry_lookup():
    assert impl_double.get_impl("numpy") is double_impl
    assert impl_double.get_impl("iree") is None


# ---------------------------------------------------------------------------
# Static attributes arrive as payload kwargs
# ---------------------------------------------------------------------------

impl_scaled = block(
    "impl_scaled",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
    attributes={"scale": float},
)

_scaled_kwargs = {}


@impl_scaled.impl("numpy")
def scaled_impl(x, **kwargs):
    """Static attributes arrive as JSON-safe payload dicts (NOT decoded
    Python values): ``{"scale": {"kind": "float", "value": 2.5}}``."""
    _scaled_kwargs["seen"] = kwargs
    scale = kwargs["scale"]["value"]
    return x * scale


def test_static_attributes_flow_as_payload_kwargs():
    @etl.defn
    def f(x):
        return impl_scaled(x, 2.5)

    x = np.arange(4, dtype=np.float32)
    y = etl.evaluate(f, x)

    assert _scaled_kwargs["seen"] == {"scale": {"kind": "float", "value": 2.5}}
    assert np.allclose(y.numpy(), x * 2.5)


# ---------------------------------------------------------------------------
# Multi-output impls
# ---------------------------------------------------------------------------

impl_multi_list = block(
    "impl_multi_list",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32), etl.TensorSpec((4,), etl.float32)],
)


@impl_multi_list.impl("numpy")
def multi_list_impl(x):
    return [x * 2.0, x + 1.0]


impl_multi_tuple = block(
    "impl_multi_tuple",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32), etl.TensorSpec((4,), etl.float32)],
)


@impl_multi_tuple.impl("numpy")
def multi_tuple_impl(x):
    return (x * 2.0, x + 1.0)


@pytest.mark.parametrize(
    "blk", [impl_multi_list, impl_multi_tuple], ids=["list", "tuple"]
)
def test_multi_output_impl(blk):
    @etl.defn
    def f(x):
        return blk(x)

    x = np.arange(4, dtype=np.float32)
    y = etl.evaluate(f, x)

    assert isinstance(y, tuple)
    assert len(y) == 2
    assert all(isinstance(out, etl.Tensor) for out in y)
    assert np.allclose(y[0].numpy(), x * 2.0)
    assert np.allclose(y[1].numpy(), x + 1.0)


# ---------------------------------------------------------------------------
# Missing impl: trace succeeds, lowering fails explicitly
# ---------------------------------------------------------------------------

impl_none = block(
    "impl_none",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
)  # deliberately NO portable and NO numpy impl


def test_missing_impl_fails_at_lower_not_trace():
    @etl.defn
    def f(x):
        return impl_none(x)

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))  # tracing succeeds

    with pytest.raises(
        etl.BackendError,
        match="impl_none.*has neither a portable decomposition nor a registered numpy impl",
    ):
        etl.lower(graph)


# ---------------------------------------------------------------------------
# The impl executes only at run time, never at trace/lower time
# ---------------------------------------------------------------------------

_call_counter = {"n": 0}

impl_counted = block(
    "impl_counted",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
)


@impl_counted.impl("numpy")
def counted_impl(x):
    _call_counter["n"] += 1
    return x + 1.0


def test_impl_not_called_at_trace_time_only_at_run_time():
    @etl.defn
    def f(x):
        return impl_counted(x)

    x = np.arange(4, dtype=np.float32)

    _call_counter["n"] = 0
    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    assert _call_counter["n"] == 0  # tracing never executes the impl

    etl.lower(graph)
    assert _call_counter["n"] == 0  # lowering keeps the op (impl registered)

    y = etl.evaluate(f, x)
    assert _call_counter["n"] == 1  # exactly once, at run time
    assert np.allclose(y.numpy(), x + 1.0)
