"""Contract tests for ``etl.constant`` — the only explicit data-embedding path.

Assertions on the contract documented in ``etl/ops/CONTEXT.md`` (constant.py
section) and ``etl/ops/constant.py``:

- ``etl.constant(tensor)`` builds a ``constant`` IR op (effect ``pure``)
  whose result dtype/shape equal the source Tensor's (SymbolicTensor fields
  AND the IR ``Value.type``), and the embedded payload is the source data.
- Data SNAPSHOT: the payload is copied at trace time — mutating the source
  Tensor after tracing does not change what the graph evaluates to.
- A ``UserWarning`` is issued when the payload exceeds
  ``ETL_LARGE_CONSTANT_BYTES`` (default 1 MiB = 1048576, read ONCE at import
  time from the ``ETL_LARGE_CONSTANT_BYTES`` env var — patch the module
  attribute directly, since ``monkeypatch.setenv`` would have no effect).
- ``core.TraceError`` outside a trace / for a SymbolicTensor input / for any
  non-Tensor input (ndarray, list, scalar); ``etl.constant`` is the public
  entry point registered via ``core.register_constant_builder``.
- Explicit device placement: a NON-cpu tensor (``device.kind != "cpu"``) is
  rejected with ``core.DeviceError`` — the snapshot would be an implicit
  device-to-host transfer, which never happens. The documented preparation
  path is the explicit two-step ``tensor.to(core.Device('cpu', 0))`` then
  ``etl.constant``; cpu-kind PAYLOAD tensors (host-reachable via
  ``to_host()``, not ndarray-backed) pass the gate and snapshot normally.
"""
import importlib
import warnings

import numpy as np
import pytest

import etl
from etl import core

from tests.ops.conftest import ops_of, run_numpy

# NOTE: ``etl.ops.constant`` as an ATTRIBUTE is the re-exported function
# (``etl.ops.__init__`` re-exports the op), so the module itself must be
# resolved through the import system to reach the module-level threshold
# attribute that ``constant()`` reads at call time.
constant_module = importlib.import_module("etl.ops.constant")

DEFAULT_THRESHOLD = 1 * 1024 * 1024  # documented default: 1 MiB


# ---------------------------------------------------------------------------
# IR construction: op name, effect, result types, SymbolicTensor fields
# ---------------------------------------------------------------------------


def test_constant_builds_pure_constant_op_with_dtype_and_shape():
    source = etl.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    captured = {}

    def f():
        out = etl.constant(source)
        captured["out"] = out
        return out

    graph = etl.trace(f)
    ops = ops_of(graph)
    assert [op.name for op in ops] == ["constant", "return"]
    const_op = ops[0]
    return_op = ops[1]

    # A pure Constant op with no operands, carrying the embedded payload.
    assert const_op.effect == "pure"
    assert const_op.operands == ()
    payload = const_op.attributes["value"]
    assert isinstance(payload, np.ndarray)
    np.testing.assert_array_equal(payload, source.numpy())

    # The IR result type preserves the source dtype/shape.
    (result,) = const_op.results
    assert result.type.dtype == np.dtype("float32")
    assert result.type.shape == (2, 3)

    # The SymbolicTensor wrapper agrees with the IR value type.
    sym = captured["out"]
    assert isinstance(sym, etl.SymbolicTensor)
    assert sym.dtype == source.dtype == np.dtype("float32")
    assert sym.shape == source.shape == (2, 3)

    # The graph result is the constant value.
    assert return_op.name == "return"
    assert return_op.operands == (result,)


def test_constant_preserves_int64_scalar_dtype_and_shape():
    source = etl.tensor(np.array(7, dtype=np.int64))
    captured = {}

    def f():
        out = etl.constant(source)
        captured["out"] = out
        return out

    etl.trace(f)
    sym = captured["out"]
    assert sym.dtype == np.dtype("int64")
    assert sym.shape == ()  # 0-d payload
    assert sym.value.type.dtype == np.dtype("int64")
    assert sym.value.type.shape == ()


# ---------------------------------------------------------------------------
# Numerics through the numpy backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int64, np.float32], ids=["int64", "float32"])
def test_constant_used_in_arithmetic_matches_numpy(dtype):
    source = etl.tensor(np.arange(12, dtype=dtype).reshape(3, 4))

    def f(x):
        return x * 2 + etl.constant(source)

    x = np.ones((3, 4), dtype=dtype)
    out = run_numpy(f, x)
    expected = x * 2 + source.numpy()
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# Data snapshot semantics
# ---------------------------------------------------------------------------


def test_constant_snapshots_data_at_trace_time():
    source = etl.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f():
        return etl.constant(source)

    # Trace ONCE — the snapshot happens here; execution below must never
    # re-trace.
    graph = etl.trace(f)
    const_op = ops_of(graph, "constant")[0]
    executable = etl.load(etl.compile(etl.lower(graph)))
    np.testing.assert_array_equal(etl.run(executable).numpy(), expected)

    # Mutating the source tensor AFTER tracing must not affect the graph.
    source.data[:] = 99.0
    np.testing.assert_array_equal(etl.run(executable).numpy(), expected)

    # The IR payload is a copy of the trace-time buffer, not a view of it,
    # and keeps the pre-mutation values.
    assert not np.shares_memory(const_op.attributes["value"], source.data)
    np.testing.assert_array_equal(const_op.attributes["value"], expected)


# ---------------------------------------------------------------------------
# Large-constant warning threshold
# ---------------------------------------------------------------------------


def test_default_threshold_is_one_mib():
    assert constant_module.ETL_LARGE_CONSTANT_BYTES == DEFAULT_THRESHOLD
    assert DEFAULT_THRESHOLD == 1048576


def test_large_constant_warns_above_threshold(monkeypatch):
    monkeypatch.setattr(constant_module, "ETL_LARGE_CONSTANT_BYTES", 8)
    source = etl.tensor(np.arange(10, dtype=np.int64))  # 80 bytes > 8

    def f():
        return etl.constant(source)

    with pytest.warns(UserWarning, match="Embedding a tensor of 80 bytes"):
        graph = etl.trace(f)
    # The warning does not suppress the embedding.
    const_op = ops_of(graph, "constant")[0]
    np.testing.assert_array_equal(const_op.attributes["value"], source.numpy())


def test_small_constant_does_not_warn(monkeypatch, recwarn):
    monkeypatch.setattr(constant_module, "ETL_LARGE_CONSTANT_BYTES", 1024)
    source = etl.tensor(np.arange(10, dtype=np.int64))  # 80 bytes < 1024

    def f():
        return etl.constant(source)

    graph = etl.trace(f)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
    const_op = ops_of(graph, "constant")[0]
    np.testing.assert_array_equal(const_op.attributes["value"], source.numpy())


def test_constant_exactly_at_threshold_does_not_warn(monkeypatch, recwarn):
    # The check is strictly `payload.nbytes > threshold`.
    monkeypatch.setattr(constant_module, "ETL_LARGE_CONSTANT_BYTES", 16)
    source = etl.tensor(np.arange(2, dtype=np.int64))  # exactly 16 bytes

    def f():
        return etl.constant(source)

    etl.trace(f)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_constant_warning_disabled_with_always_filter(monkeypatch):
    # warnings.catch_warnings + simplefilter("error"): any UserWarning during
    # the trace would blow up the test (strictest below-threshold check).
    monkeypatch.setattr(constant_module, "ETL_LARGE_CONSTANT_BYTES", 1024)
    source = etl.tensor(np.arange(10, dtype=np.int64))

    def f():
        return etl.constant(source)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        etl.trace(f)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_constant_outside_trace_raises_traceerror():
    source = etl.tensor(np.ones(3, dtype=np.float32))
    with pytest.raises(etl.TraceError, match="No active trace"):
        etl.constant(source)


@pytest.mark.parametrize(
    "bad",
    [np.array([1, 2, 3]), [1, 2, 3], 5, 1.5, True, "abc"],
    ids=["ndarray", "list", "int", "float", "bool", "str"],
)
def test_constant_of_non_tensor_raises_traceerror(bad):
    def f(bad=bad):
        return etl.constant(bad)

    with pytest.raises(etl.TraceError, match="expects a concrete core.Tensor"):
        etl.trace(f)


def test_constant_of_symbolic_tensor_raises_traceerror():
    def f(x):
        return etl.constant(x)

    with pytest.raises(etl.TraceError, match="already a graph value"):
        etl.trace(f, etl.TensorSpec((2,), etl.float32))


# ---------------------------------------------------------------------------
# Explicit device placement: non-cpu tensors are rejected; the explicit
# .to(cpu) two-step is the documented preparation path
# ---------------------------------------------------------------------------


class _FakeDevicePayload:
    """Minimal duck-typed device payload (protocol documented on
    ``core.Tensor``: ``.shape`` / ``.dtype``, optional ``.device`` — a
    ``core.Device`` — and ``to_host() -> ndarray``). Local stand-in only;
    tests never import from ``tests/core``.
    """

    def __init__(self, array, device):
        self._array = np.asarray(array)
        self.shape = self._array.shape
        self.dtype = self._array.dtype
        self.device = device

    def to_host(self):
        """A FRESH host copy (a real device buffer would copy on demand)."""
        return self._array.copy()


def test_constant_of_non_cpu_tensor_raises_deviceerror():
    """A device-resident (non-cpu) tensor cannot be embedded: the snapshot
    would be an implicit device-to-host transfer, which never happens at
    constant snapshot time."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    cuda_tensor = core.Tensor(_FakeDevicePayload(data, core.Device("cuda", 0)))
    assert cuda_tensor.device == core.Device("cuda", 0)

    def f():
        return etl.constant(cuda_tensor)

    with pytest.raises(etl.DeviceError) as excinfo:
        etl.trace(f)
    message = str(excinfo.value)
    assert "etl.constant requires host data" in message
    assert "no implicit device-to-host transfer" in message
    assert "t.to(" in message


def test_constant_embeds_after_explicit_to_cpu_transfer():
    """The explicit two-step workflow (R2): ``Tensor.to(core.Device('cpu',
    0))`` on a non-cpu payload tensor returns a fresh HOST tensor — the legal
    way to prepare device data for constant embedding."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    cuda_tensor = core.Tensor(_FakeDevicePayload(data, core.Device("cuda", 0)))

    host_tensor = cuda_tensor.to(core.Device("cpu", 0))
    # The explicit transfer yields a fresh ndarray-backed tensor on cpu:0;
    # the source stays untouched on its device.
    assert host_tensor.device == core.Device("cpu", 0)
    assert isinstance(host_tensor.data, np.ndarray)
    assert host_tensor is not cuda_tensor
    assert cuda_tensor.device == core.Device("cuda", 0)

    def f():
        return etl.constant(host_tensor)

    graph = etl.trace(f)
    const_op = ops_of(graph, "constant")[0]
    np.testing.assert_array_equal(const_op.attributes["value"], data)

    # End-to-end: the graph evaluates to the embedded values (numpy backend).
    executable = etl.load(etl.compile(etl.lower(graph)))
    np.testing.assert_array_equal(etl.run(executable).numpy(), data)


def test_constant_of_cpu_kind_device_payload_embeds_fine():
    """The snapshot gate is kind-based (``device.kind != 'cpu'``), NOT
    ndarray-backedness: a cpu-kind PAYLOAD tensor (host-reachable via
    ``to_host()``) passes the gate and snapshots normally."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    cpu_tensor = core.Tensor(_FakeDevicePayload(data, core.Device("cpu", 0)))
    assert cpu_tensor.device == core.Device("cpu", 0)
    assert not isinstance(cpu_tensor.data, np.ndarray)  # payload-backed

    def f():
        return etl.constant(cpu_tensor)

    graph = etl.trace(f)
    const_op = ops_of(graph, "constant")[0]
    np.testing.assert_array_equal(const_op.attributes["value"], data)

    executable = etl.load(etl.compile(etl.lower(graph)))
    np.testing.assert_array_equal(etl.run(executable).numpy(), data)


# ---------------------------------------------------------------------------
# Public entry point / registration hook
# ---------------------------------------------------------------------------


def test_public_constant_is_the_registered_builder():
    """``etl.constant`` is the hook installed via ``core.register_constant_builder``."""
    from etl.core import symbolic

    assert etl.constant is constant_module.constant
    assert symbolic._get_constant_builder() is etl.constant
