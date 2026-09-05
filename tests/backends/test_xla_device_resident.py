"""xla adapter device-resident staging semantics (regression suite).

Pins the EXPLICIT device-placement execution contract for the xla adapter
(``etl/backends/adapters/xla.py`` + ``xla_util.py``), exercised against the
TEST-ONLY fake PJRT plugin built from C at test time (the same fake as
``test_pjrt_ctypes_plugin.py``; zero-filled outputs, ONE addressable
device, no computation):

* ``load(artifact, device=Device("cuda", 0))`` produces a device-resident
  executable; run outputs are ``XlaDevicePayload``-backed ``core.Tensor``\\ s
  that STAY on the device: ``.numpy()`` raises the standard
  ``core.DeviceError`` (no implicit device-to-host transfer); the explicit
  ``t.to(cpu).numpy()`` hop round-trips.
* same-device pass-through: payload inputs are handed to
  ``PJRT_LoadedExecutable_Execute`` DIRECTLY — ZERO
  ``BufferFromHostBuffer`` / ``ToHostBuffer`` calls after the seed
  placement (spy-counter pinned): the device-resident loop (each step's
  outputs fed back as the next step's inputs — the evox StdWorkflow
  pattern) never touches host memory, so the per-call ~0.75-0.8 ms staging
  fixed cost is paid exactly ONCE (the seed ``t.to(...)``).
* host inputs are ILLEGAL at the run boundary: the pipeline raises
  ``core.DeviceError`` naming the input path + the ``t.to(...)`` remedy;
  the backend level raises ``core.DeviceError`` (never stages host inputs)
  for direct ``raw.run([...])`` calls.
* load-device validation: ``Device("cpu", 1)`` -> ``DeviceError``, unknown
  kind -> ``BackendError`` naming {"cpu","cuda"}, out-of-range cuda index
  -> ``BackendError`` naming the device count.
* payload lifecycle: payloads keep the shared client alive after the
  executable's ``close()`` (usable ``.to(cpu)`` round-trips); the client is
  destroyed at the LAST payload GC; a stale payload finalizer can never
  decrement a re-created client (cache-identity guard in
  ``xla_util.release_payload``).

SEMANTICS ONLY — the fake plugin performs no computation (zero-filled
outputs), so numerical parity is out of scope here (real-plugin parity
lives in ``test_adapter_xla.py``).
"""

import gc

import numpy as np
import pytest

import etl
from etl.backends.adapters import xla
from etl.backends.adapters import xla_util

from tests.backends import _adapter_utils as u
from tests.backends.test_pjrt_ctypes_plugin import _build_plugin, _compile_fake

NAME = "xla"
CPU = etl.core.Device("cpu", 0)
CUDA0 = etl.core.Device("cuda", 0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _activate(fake_plugin, monkeypatch):
    """Point plugin discovery at the fake .so and (re)activate the adapter."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_plugin))
    xla.register()  # idempotent; also (re)installs the "cuda" placement provider


def _load_raw(artifact, device=CUDA0):
    """Backend-level load on ``device`` (the raw ``XlaExecutable``)."""
    return etl.backends.get(NAME).load(artifact, device=device)


def _assert_device_resident(out, device):
    """A run-out tensor is payload-backed, on ``device``, host-copy explicit.

    Public-class assertions (``XlaDevicePayload`` is public, mirroring
    ``IreeDevicePayload``): not an ndarray, knows shape/dtype/device, and
    exposes the explicit ``to_host`` path.
    """
    assert isinstance(out, etl.Tensor)
    assert isinstance(out.data, xla.XlaDevicePayload)
    assert not isinstance(out.data, np.ndarray)  # NOT numpy-backed
    assert out.device == device
    assert out.data.device == device
    assert out.shape == out.data.shape
    assert out.dtype == out.data.dtype


# ---------------------------------------------------------------------------
# 1. device-resident run-out semantics + explicit host access
# ---------------------------------------------------------------------------


def test_cuda_runout_device_resident(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_runout.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    raw = _load_raw(artifact)
    assert raw.device == CUDA0  # the load device is the executable's device

    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)  # numpy reference (shape/dtype only)
    xa, xb = etl.core.Tensor(a).to(CUDA0), etl.core.Tensor(b).to(CUDA0)
    (out,) = raw.run([xa, xb])
    _assert_device_resident(out, CUDA0)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype

    # Semantic pin (explicit-placement model): a non-cpu run-out payload
    # NEVER transfers implicitly — .numpy() raises DeviceError; the
    # explicit .to(cpu) hop is the only host path.
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    host = out.to(CPU)
    assert isinstance(host.data, np.ndarray)  # explicit D2H -> host memory
    assert host.device == CPU
    # The fake performs no computation: outputs are zero-filled.
    np.testing.assert_array_equal(host.numpy(), np.zeros_like(expected.numpy()))

    # The pipeline-level run reaches the same device-resident path.
    exe = etl.load(artifact, device=CUDA0)
    pipeline_out = etl.run(exe, xa, xb)
    _assert_device_resident(pipeline_out, CUDA0)
    np.testing.assert_array_equal(
        pipeline_out.to(CPU).numpy(), np.zeros_like(expected.numpy())
    )


def test_rank0_staging_scalar_shape(tmp_path, monkeypatch):
    """rank-0 host arrays stage as shape-() buffers.

    Regression pin for the xla_util.buffer_from_host rank-0 fix: the driver
    stages rank-0 host arrays with ``dims=NULL`` + ``num_dims=0`` (the
    header-contract form), so shape () round-trips via
    ``PJRT_Buffer_Dimensions`` — verified on the real jax_cuda12_pjrt
    0.10.2 plugin (both NULL and non-NULL dims report () there when
    num_dims=0; the historical (1,) symptom was traced to
    ``np.ascontiguousarray``'s hardcoded ndmin=1 promoting 0-d to (1,)
    BEFORE dims handling). The fake plugin emulates the hypothesized
    non-NULL-dims quirk as a driver-regression guard: evox states carry
    rank-0 leaves (``key`` () i64, ``generation`` () i32), and any driver
    regression re-staging rank-0 as a non-NULL-dims (1,) array fails here.
    """
    plug = _build_plugin(tmp_path, "fake_devres_rank0.so")
    _activate(plug, monkeypatch)
    for value, dtype in ((42, np.int64), (7, np.int32), (2.5, np.float32)):
        t = etl.core.Tensor(np.array(value, dtype=dtype)).to(CUDA0)
        _assert_device_resident(t, CUDA0)
        assert t.shape == (), f"rank-0 staged as {t.shape!r}, not ()"
        assert t.dtype == np.dtype(dtype)
        host = t.to(CPU)
        np.testing.assert_array_equal(
            host.numpy(), np.asarray(value, dtype=dtype)
        )


# ---------------------------------------------------------------------------
# 2. same-device loop: zero host staging after the seed placement
# ---------------------------------------------------------------------------


def _step(x):
    """An iterative step graph: output shape == input shape, so each step's
    output feeds back as the next step's input (the evox loop pattern)."""
    return etl.add(x, 1.0)


STEP_SPECS = (etl.TensorSpec((4, 8), etl.float32),)


def test_same_device_loop_zero_host_staging(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_loop.so")
    _activate(plug, monkeypatch)
    artifact = _compile_fake(_step, STEP_SPECS, plug)
    raw = _load_raw(artifact)

    # Spy counters: the ONLY host<->device events are BufferFromHostBuffer
    # (staging) and ToHostBuffer (copy-back).
    stages: list = []
    copies: list = []
    real_stage = xla_util._Client.buffer_from_host
    real_copy = xla_util._Buffer.to_host

    def _count_stage(self, array, device_index=None):
        stages.append((tuple(array.shape), device_index))
        return real_stage(self, array, device_index=device_index)

    def _count_copy(self):
        copies.append(1)
        return real_copy(self)

    monkeypatch.setattr(xla_util._Client, "buffer_from_host", _count_stage)
    monkeypatch.setattr(xla_util._Buffer, "to_host", _count_copy)

    # Seed: the explicit one-time placement (the bootstrap host->device
    # upload) — the ONLY staging event of the whole workload.
    x0 = u.standard_normal((4, 8))
    expected = etl.evaluate(_step, x0)
    state = etl.core.Tensor(x0).to(CUDA0)
    assert len(stages) == 1  # one staged seed
    assert stages[0][1] == 0  # staged on cuda device 0
    stages.clear()

    # The device-resident loop: outputs feed back as next-step inputs.
    # ZERO host staging and ZERO copy-backs from the FIRST call on.
    for _ in range(5):
        state = raw.run([state])[0]
    assert stages == []  # no BufferFromHostBuffer after the seed
    assert copies == []  # no ToHostBuffer during the loop
    _assert_device_resident(state, CUDA0)
    assert state.shape == expected.shape

    # The explicit host read-back is the FIRST (and only) ToHostBuffer.
    host = state.to(CPU)
    assert len(copies) == 1
    np.testing.assert_array_equal(host.numpy(), np.zeros_like(expected.numpy()))


# ---------------------------------------------------------------------------
# 3. host inputs are ILLEGAL at the run boundary (never staged)
# ---------------------------------------------------------------------------


def test_host_inputs_require_explicit_placement(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_boundary.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    a, b = u.matmul_relu_sum_args()

    # (a) Pipeline run boundary: feeding host cpu:0 tensors (or raw numpy
    # arrays, auto-wrapped as cpu:0) to a cuda executable raises
    # DeviceError naming the input path and the t.to(...) remedy, BEFORE
    # the backend is dispatched.
    exe = etl.load(artifact, device=CUDA0)
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl.run(exe, etl.core.Tensor(a), etl.core.Tensor(b))
    msg = str(excinfo.value)
    assert "input at path [0]" in msg
    assert (
        "no implicit device-to-host or host-to-device transfer happens at "
        "the run boundary" in msg
    )
    assert "t.to(" in msg
    with pytest.raises(
        etl.core.DeviceError,
        match="no implicit device-to-host or host-to-device transfer happens "
        "at the run boundary",
    ):
        etl.run(exe, a, b)

    # (b) Backend-level defense: a direct raw.run([...]) call never stages
    # host inputs for a device-resident executable.
    raw = _load_raw(artifact)
    with pytest.raises(etl.core.DeviceError, match="never stages host inputs"):
        raw.run([etl.core.Tensor(a), etl.core.Tensor(b)])

    # (c) Inputs placed explicitly via .to(cuda) run fine.
    (out,) = raw.run(
        [etl.core.Tensor(a).to(CUDA0), etl.core.Tensor(b).to(CUDA0)]
    )
    _assert_device_resident(out, CUDA0)


def test_cross_device_input_rejected(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_cross.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    raw = _load_raw(artifact)
    a, b = u.matmul_relu_sum_args()
    # A payload on the right device from a DIFFERENT client (different
    # plugin path) is never silently mixed into the execution.
    other_plug = _build_plugin(tmp_path, "fake_devres_other.so")
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(other_plug))
    foreign = etl.core.Tensor(a).to(CUDA0)
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(plug))
    with pytest.raises(etl.core.DeviceError, match="different PJRT client"):
        raw.run([foreign, etl.core.Tensor(b).to(CUDA0)])


# ---------------------------------------------------------------------------
# 4. load-device validation
# ---------------------------------------------------------------------------


def test_load_device_validation(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_loaddev.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    backend = etl.backends.get(NAME)

    # Out-of-range cuda index -> BackendError naming the device count.
    with pytest.raises(etl.BackendError, match="addressable device"):
        backend.load(artifact, device=etl.core.Device("cuda", 999))
    # Unknown kind -> BackendError naming the supported set.
    with pytest.raises(etl.BackendError, match="supports CPU"):
        backend.load(artifact, device=etl.core.Device("gpu", 0))
    # Non-zero cpu index -> DeviceError (only Device('cpu', 0) exists).
    with pytest.raises(etl.core.DeviceError, match="only Device\\('cpu', 0\\)"):
        backend.load(artifact, device=etl.core.Device("cpu", 1))
    # Non-Device -> DeviceError.
    with pytest.raises(etl.core.DeviceError, match="core.Device"):
        backend.load(artifact, device="cuda")  # type: ignore[arg-type]

    # Default (None) and explicit cpu:0 loads stay on the historical CPU
    # staging path (host in -> host out).
    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)
    for dev in (None, etl.core.Device("cpu", 0)):
        raw = backend.load(artifact, device=dev)
        assert raw.device == etl.core.Device("cpu", 0)
        (out,) = raw.run([etl.core.Tensor(a), etl.core.Tensor(b)])
        assert isinstance(out.data, np.ndarray)  # cpu path copies back
        np.testing.assert_array_equal(out.numpy(), np.zeros_like(expected.numpy()))


# ---------------------------------------------------------------------------
# 5. payload lifecycle: client refcounts, close() ordering, stale finalizers
# ---------------------------------------------------------------------------


def test_payload_keeps_client_alive_and_client_dies_at_last_payload(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_lifecycle.so")
    _activate(plug, monkeypatch)
    fn, specs = u.matmul_relu_sum()
    artifact = _compile_fake(fn, specs, plug)
    raw = _load_raw(artifact)
    a, b = u.matmul_relu_sum_args()
    expected = etl.evaluate(fn, a, b)
    xa, xb = etl.core.Tensor(a).to(CUDA0), etl.core.Tensor(b).to(CUDA0)
    (out,) = raw.run([xa, xb])

    # References: executable (1) + xa payload (1) + xb payload (1) + out
    # payload (1) — the shared client is refcounted process-wide.
    path = str(plug)
    client = xla_util._shared_clients[path][0]
    assert xla_util._shared_clients[path][1] == 4

    # Closing the executable releases its reference only; the payloads keep
    # the client alive and stay fully usable (explicit D2H round-trip).
    raw.close()
    assert raw._client is None
    assert xla_util._shared_clients[path][1] == 3
    assert not client.closed
    np.testing.assert_array_equal(
        out.to(CPU).numpy(), np.zeros_like(expected.numpy())
    )

    # The LAST payload GC releases the final reference: buffer destroyed
    # first, client destroyed second, cache entry removed.
    del out, xa, xb
    gc.collect()
    assert path not in xla_util._shared_clients
    assert client.closed


def test_stale_payload_finalizer_never_touches_recreated_client(tmp_path):
    """The cache-identity guard: a payload owned by a DEAD client can never
    decrement (or destroy) a LATER re-created client for the same plugin
    path — ``release_payload`` early-returns when the cache no longer holds
    that exact client."""
    plug = _build_plugin(tmp_path, "fake_stale_payload.so")
    plugin = xla_util._load_plugin({"plugin_path": str(plug)})
    client_a = xla_util.acquire_client(plugin)
    buffer_a = client_a.buffer_from_host(np.zeros((2, 3), np.float32))
    xla_util.release_client(plugin)  # client A dies with its buffers
    assert client_a.closed

    # A new client for the same path (fresh reference count = 1).
    client_b = xla_util.acquire_client(plugin)
    assert client_b is not client_a
    assert xla_util._shared_clients[plugin.path][1] == 1

    # The stale finalizer for (client_a, buffer_a) is a NO-OP: it must not
    # decrement client_b's refcount nor close the buffer (which died with A).
    xla_util.release_payload(plugin, client_a, buffer_a)
    assert xla_util._shared_clients[plugin.path][1] == 1
    assert not client_b.closed

    xla_util.release_client(plugin)
    assert plugin.path not in xla_util._shared_clients
    assert client_b.closed


# ---------------------------------------------------------------------------
# 6. placement-provider contract (upload_tensor)
# ---------------------------------------------------------------------------


def test_placement_provider_contract(tmp_path, monkeypatch):
    plug = _build_plugin(tmp_path, "fake_devres_upload.so")
    _activate(plug, monkeypatch)
    a = u.standard_normal((4, 8))

    # The provider is the xla adapter's upload_tensor (registered last-wins
    # for kind "cuda" — the iree-mirroring pattern).
    from etl.core.tensor import _get_device_transfer_provider

    assert _get_device_transfer_provider("cuda") is xla.upload_tensor

    # Bootstrap placement: host cpu:0 -> cuda 0 via BufferFromHostBuffer.
    xa = etl.core.Tensor(a).to(CUDA0)
    _assert_device_resident(xa, CUDA0)
    # Same-device .to() is a no-op (returns self, zero staging).
    assert xa.to(CUDA0) is xa

    # Out-of-range cuda index on a HOST source -> the provider raises
    # BackendError naming the device count.
    with pytest.raises(etl.BackendError, match="addressable device"):
        etl.core.Tensor(a).to(etl.core.Device("cuda", 5))
    # A payload-backed source to a DIFFERENT non-cpu target raises the core
    # cross-device DeviceError BEFORE the provider is consulted (the
    # explicit two-hop .to(cpu).to(target) is the remedy).
    with pytest.raises(etl.core.DeviceError, match="cross-device copies"):
        xa.to(etl.core.Device("cuda", 1))
