"""BlockError hierarchy + lower-time verification errors for `etl.block`.

Covers:

- the error hierarchy (`BlockError` ⊂ `ETLError`, sibling error classes),
- declaration-time misuse (malformed specs, invalid effects, non-defn
  portable, invalid impl backend name),
- lower-time failures: unknown (unregistered) block, portable output
  type/shape/dtype guards (`etl/backends/numpy/inline.py`), and portables
  returning non-symbolic values.

Block names are prefixed ``err_`` because the block registry is global and
process-wide — a name can only ever be declared once per process.
"""

import numpy as np
import pytest

import etl
from etl.block import BlockError, block

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_type",
    [
        BlockError,
        etl.TraceError,
        etl.ShapeError,
        etl.DTypeError,
        etl.BackendError,
        etl.TransformError,
        etl.PersistenceError,
        etl.DeviceError,
        etl.VerificationError,
    ],
)
def test_error_hierarchy_under_etlerror(error_type):
    assert issubclass(error_type, etl.ETLError)


# ---------------------------------------------------------------------------
# Declaration-time BlockError cases
# ---------------------------------------------------------------------------


def test_declaration_malformed_input_specs():
    with pytest.raises(BlockError, match="inputs must be a TensorSpec"):
        block(
            "err_badinput",
            inputs=[3],
            outputs=[etl.TensorSpec((4,), etl.float32)],
        )


def test_declaration_invalid_effects():
    with pytest.raises(BlockError, match="effects must be one of"):
        block(
            "err_badeffects",
            inputs=[etl.TensorSpec((4,), etl.float32)],
            outputs=[etl.TensorSpec((4,), etl.float32)],
            effects="teleports",
        )


def test_declaration_non_defn_portable():
    with pytest.raises(BlockError, match="must be an etl.defn"):
        block(
            "err_badportable",
            inputs=[etl.TensorSpec((4,), etl.float32)],
            outputs=[etl.TensorSpec((4,), etl.float32)],
            portable=lambda x: x,
        )


err_implname = block(
    "err_implname",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
)


@pytest.mark.parametrize("bad_name", [123, "", None, 1.5], ids=repr)
def test_impl_requires_nonempty_string_backend_name(bad_name):
    with pytest.raises(
        BlockError, match="backend name must be a non-empty string"
    ):
        err_implname.impl(bad_name)


# ---------------------------------------------------------------------------
# Unregistered (unknown) block at lower time
# ---------------------------------------------------------------------------

# Constructed DIRECTLY — never registered in the block registry.
err_ghost = etl.BlockOp(
    name="err_ghost",
    input_specs=(etl.TensorSpec((4,), etl.float32),),
    output_specs=(etl.TensorSpec((4,), etl.float32),),
    attribute_schema={},
    effects="pure",
    batching_policy="unsupported",
)


def test_unregistered_block_instance_fails_at_lower():
    @etl.defn
    def f(x):
        return err_ghost(x)

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))  # trace succeeds

    with pytest.raises(
        etl.BackendError, match="cannot lower block_call: unknown block"
    ) as excinfo:
        etl.lower(graph)
    assert "err_ghost" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Portable output-type guards (etl/backends/numpy/inline.py)
# ---------------------------------------------------------------------------


@etl.defn
def _err_mismatch_shape_portable(x):
    return x * 2.0  # traces to shape (4,) — declared outputs say ()


err_mismatch_shape = block(
    "err_mismatch_shape",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((), etl.float32)],
    portable=_err_mismatch_shape_portable,
)


@etl.defn
def _err_mismatch_dtype_portable(x):
    return etl.cast(x, etl.int32)  # int32 — declared outputs say float32


err_mismatch_dtype = block(
    "err_mismatch_dtype",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((4,), etl.float32)],
    portable=_err_mismatch_dtype_portable,
)


@pytest.mark.parametrize(
    ("blk", "message"),
    [
        (err_mismatch_shape, "portable decomposition produces shape"),
        (err_mismatch_dtype, "portable decomposition produces dtype"),
    ],
    ids=["shape", "dtype"],
)
def test_portable_output_type_guard_at_lower(blk, message):
    @etl.defn
    def f(x):
        return blk(x)

    # Trace succeeds — the block_call result is typed from the DECLARED
    # output specs. The guard fires when the portable decomposition is
    # inlined at lower time.
    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    with pytest.raises(etl.BackendError, match=message):
        etl.lower(graph)


# ---------------------------------------------------------------------------
# Portable returning a non-symbolic value
# ---------------------------------------------------------------------------


@etl.defn
def _err_floatret_portable(x):
    return 1.5  # static Python float — NOT a symbolic tensor


err_floatret = block(
    "err_floatret",
    inputs=[etl.TensorSpec((4,), etl.float32)],
    outputs=[etl.TensorSpec((), etl.float32)],
    portable=_err_floatret_portable,
)


def test_portable_returning_python_float_fails_at_lower():
    @etl.defn
    def f(x):
        return err_floatret(x)

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    # The static float lands in the trace's output_static_values, so the
    # decomposition's return terminator has zero tensor operands; the
    # inline.py arity guard catches the mismatch at lower time.
    with pytest.raises(
        etl.BackendError, match="portable decomposition produces 0 result"
    ):
        etl.lower(graph)


def test_portable_returning_python_float_fails_in_vjp_fallback():
    @etl.defn
    def f(x):
        return etl.sum(err_floatret(x))

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    # The pre-registered vjp fallback traces the portable directly and
    # rejects the non-symbolic return via _flatten_outputs (rules.py).
    with pytest.raises(BlockError, match="must return symbolic tensors"):
        etl.grad(graph)
