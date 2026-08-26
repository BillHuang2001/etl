"""Deferred / failing staging paths of the numpy reference backend.

The numpy kernel table registers ALL 74 non-return IR ops, so the deferred-op
surface is minimal — the only lower-time rejection is a `block_call` whose
block has NEITHER a registered numpy impl NOR a portable decomposition (the
sanctioned paths are `BlockOp.portable(...)` / `BlockOp.impl('numpy')`; there
is no silent fallback). Everything else fails at explicit staging boundaries:

- `compile` with a backend whose name != the lowered program's backend →
  `core.BackendError` (never cross-backend compilation).
- `load` on a non-CPU device → `core.BackendError` naming the kind (a
  non-`Device` object → `core.DeviceError`).
- `etl.backends.get("stablehlo")` → `core.BackendError` (the exporter is an
  export-only utility, deliberately NOT registered as a backend).

StableHLO export deferred-op tests (gather/scatter/scan/runtime_call/...) live
in `test_stablehlo.py` — this file deliberately does not duplicate them.
"""

import numpy as np
import pytest

import etl
from etl.backends import Backend, Capabilities, register


def _simple_graph():
    def f(x):
        return etl.add(x, x)

    return etl.trace(f, etl.TensorSpec((2,), etl.float32))


def _stage():
    """Trace → lower → compile a trivial graph through the numpy backend."""
    lowered = etl.lower(_simple_graph())
    artifact = etl.compile(lowered)
    return lowered, artifact


# --- 1. block_call without impl/portable fails at LOWER time -----------------

def test_block_call_without_impl_fails_at_lower_time():
    bare = etl.block(
        "deferred_errors_bare_block",
        inputs=[etl.TensorSpec((), etl.float32)],
        outputs=[etl.TensorSpec((), etl.float32)],
    )

    def f(x):
        return bare(x)

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    # Negative control: the graph itself is valid IR — the rejection is
    # genuinely lower-time (block resolution), not a verification failure.
    graph.verify()
    with pytest.raises(etl.BackendError) as excinfo:
        etl.lower(graph)
    msg = str(excinfo.value)
    assert "deferred_errors_bare_block" in msg
    assert "portable" in msg  # message points at BlockOp.portable / BlockOp.impl


def test_block_call_with_portable_decomposition_lowers():
    # The sanctioned path: a portable decomposition inlines at lower time.

    @etl.block(outputs=[etl.TensorSpec((), etl.float32)])
    @etl.defn
    def deferred_errors_portable_block(x: etl.TensorSpec((), etl.float32)):
        return etl.add(x, x)

    def f(x):
        return deferred_errors_portable_block(x)

    graph = etl.trace(f, etl.TensorSpec((), etl.float32))
    lowered = etl.lower(graph)  # must NOT raise
    assert lowered.backend == "numpy"


# --- 2. compile with a mismatched backend --------------------------------------

def test_numpy_compile_rejects_lowered_program_from_another_backend():
    fake = etl.backends.LoweredProgram(backend="torch", payload="mlir text")
    with pytest.raises(
        etl.BackendError, match="produced by backend 'torch'"
    ):
        etl.backends.numpy_backend.compile(fake)


class _FakeBackend(Backend):
    """Minimal registered backend used to exercise the pipeline-level guard."""

    name = "etl_test_fake_backend"
    capabilities = Capabilities()

    def lower(self, graph, options=None):  # pragma: no cover - never called
        raise NotImplementedError

    def compile(self, lowered, options=None):  # pragma: no cover - never called
        raise NotImplementedError

    def load(self, artifact, device=None):  # pragma: no cover - never called
        raise NotImplementedError


_FAKE_BACKEND = _FakeBackend()
register(_FAKE_BACKEND)


def test_pipeline_compile_guard_rejects_backend_mismatch():
    lowered, _ = _stage()
    with pytest.raises(
        etl.BackendError, match="cannot compile a LoweredProgram"
    ):
        etl.compile(lowered, backend=_FAKE_BACKEND)


# --- 3. load with an unsupported device -----------------------------------------

def test_load_rejects_non_cpu_device():
    _, artifact = _stage()
    with pytest.raises(etl.BackendError) as excinfo:
        etl.load(artifact, device=etl.Device("gpu", 0))
    msg = str(excinfo.value)
    assert "CPU devices only" in msg
    assert "'gpu'" in msg


def test_load_rejects_non_device_object():
    _, artifact = _stage()
    with pytest.raises(etl.DeviceError):
        etl.load(artifact, device=42)


# --- 4. stablehlo is not a registered backend -------------------------------------

def test_stablehlo_is_not_a_registered_backend():
    with pytest.raises(etl.BackendError, match="unknown backend 'stablehlo'"):
        etl.backends.get("stablehlo")
    # The default reference backend IS registered.
    assert etl.backends.get("numpy").name == "numpy"


# --- 5. capability probe: numpy supports the whole standard surface ----------------

def test_numpy_capabilities_cover_the_standard_surface():
    caps = etl.backends.numpy_backend.capabilities
    # All four standard-op flags are True: the numpy backend defers nothing
    # among standard ops — its only lower-time rejection is an unresolvable
    # block_call (tested above).
    assert caps.dynamic_shapes is True
    assert caps.collectives is True
    assert caps.runtime_calls is True
    assert caps.custom_blocks is True
    for dtype in (
        etl.bool_,
        etl.float16,
        etl.float32,
        etl.float64,
        etl.int32,
        etl.int64,
        etl.uint8,
        etl.complex64,
    ):
        assert caps.supports_dtype(np.dtype(dtype))
