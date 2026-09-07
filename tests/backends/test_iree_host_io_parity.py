"""etl.io host<->device async-copy parity suite (Workstream B).

Parity/negative coverage for the NEW ``etl.io`` module (``etl/io.py``,
public API frozen): ``prefetch(host_tensor, device)`` / ``sink(device_tensor)``
schedule explicit ``Tensor.to`` copies on ONE process-global FIFO daemon
worker; schedule-time validation (tensor-ness, device-ness, host-data
source rules) is synchronous on the calling thread; runtime failures (a
missing device-transfer provider, ...) surface at ``wait()``/``result()``
— never swallowed; a wait timeout never cancels a copy.

* Group A (llvm-cpu, NO GPU — device-free semantics on real cpu-kind
  payloads): schedule-time validation errors pinned VERBATIM against (a) a
  payload-backed run output from an iree-llvm-cpu executable
  (``IreeDevicePayload`` on cpu:0 — the ``prefetch``/``sink`` host-data
  rule errors) and (b) plain ndarray-backed host tensors; the
  unregistered-kind copy (``Device("nonexistent_kind", 0)``) pins that
  runtime failures surface at ``result()``/``wait()`` — ``.done`` goes
  True and every ``wait()`` re-raises.
* Group B (GPU-guarded, nvidia-smi most-free scan EXCLUDING the known
  ECC-broken device index 7 on this host; HAL device_id = index + 1):
  prefetch == explicit ``Tensor.to`` upload (bit-exact, device-resident
  result), sink round-trips to an ndarray-backed cpu:0 host tensor
  (bit-exact, source stays device-resident), a big prefetch overlaps
  same-device run calls (run outputs stay device-resident — ``.numpy()``
  raises ``DeviceError`` until the explicit ``.to(cpu)`` hop), and three
  queued prefetches complete bound to their own sources while main-thread
  runs proceed (all bit-exact).

Explicit-placement model (binding, see ``test_iree_device_resident.py``):
a cuda executable rejects host inputs at the run boundary, so every run
input is placed on the executable's device first (``Tensor.to(cuda_device)``
or an ``etl.io.prefetch`` result) and outputs are read through the explicit
``to(Device('cpu', 0))`` hop — a bare ``.numpy()`` on a cuda payload raises
``core.DeviceError``. Device-active time is kept well under a minute.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl
import etl.io as etl_io

# ---------------------------------------------------------------------------
# data + graph (exactly representable in fp32 → bit-exact elementwise)
# ---------------------------------------------------------------------------

XA = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25
XB = np.arange(24, dtype=np.float32).reshape(4, 6) / 2.0 + 1.0
XC = (np.arange(24, dtype=np.float32) * 2.0).reshape(4, 6)  # distinct from XA/XB
ADD_SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)
# DE-scale payload for the overlap/FIFO tests (~800 KB upload on the worker).
BIG = (np.random.default_rng(1).standard_normal((4096, 50)) * 0.01
       ).astype(np.float32)


@etl.defn
def _add(x, y):
    return etl.add(x, y)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor → ndarray via the explicit host hop (device-resident safe)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(etl.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    """Bit-exact tree comparison (host-materializing)."""
    for gp, wp in zip(etl.tree_leaves(etl.tree_map(_np, got)),
                      etl.tree_leaves(etl.tree_map(_np, want))):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _assert_payload_tensor(t, device, shape, dtype):
    """A tensor is payload-backed (never ndarray) and resident on ``device``."""
    assert isinstance(t, etl.Tensor)
    assert not isinstance(t.data, np.ndarray)
    assert t.device == device
    assert t.shape == shape
    assert t.dtype == dtype


def _pick_cuda_device_index():
    """Most-free GPU via nvidia-smi; pytest.skip when unavailable.

    The known ECC-broken device index 7 on this host is EXCLUDED up front
    (never scanned, never returned).
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not found — no CUDA device to test")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"nvidia-smi failed: {exc}")
    if proc.returncode != 0:
        pytest.skip(f"nvidia-smi failed: {proc.stderr.strip()}")
    gpus = []
    for line in proc.stdout.strip().splitlines():
        try:
            idx, free_mib = (part.strip() for part in line.split(","))
        except ValueError:
            continue  # malformed line — ignore
        if int(idx) == 7:
            continue  # known ECC-broken device on this host — never touch it
        gpus.append((int(free_mib), int(idx)))
    if not gpus:
        pytest.skip("nvidia-smi reported no healthy free GPU "
                    "(known ECC-broken device index 7 excluded)")
    gpus.sort(reverse=True)
    return gpus[0][1]


@pytest.fixture(scope="module")
def llvm_cpu_payload_out():
    """A payload-backed run output on cpu:0 (iree llvm-cpu; no GPU needed).

    Established idiom (test_iree_device_resident.py): an iree executable's
    run outputs are payload-backed ``core.Tensor``\\ s whose cpu-kind payload
    reads directly from host memory.
    """
    exe = etl.build(_add, *ADD_SPECS, backend="iree")
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_payload_tensor(out, etl.core.Device("cpu", 0), XA.shape,
                           np.dtype("float32"))
    assert hasattr(out.data, "to_host")  # duck-typed payload protocol
    np.testing.assert_array_equal(out.numpy(), XA + XB)  # lazy host copy
    return out


@pytest.fixture(scope="module")
def cuda_device():
    """A free CUDA device (most-free GPU via nvidia-smi); skip when unavailable."""
    idx = _pick_cuda_device_index()
    import iree.runtime as rt

    try:
        # etl Device("cuda", idx) maps to iree device_id idx + 1 (1-based ids).
        rt.get_driver("cuda").create_device(device_id=idx + 1)
    except Exception as exc:  # noqa: BLE001 — any driver/device failure skips
        pytest.skip(f"IREE cuda HAL driver or GPU {idx} unavailable: {exc}")
    return etl.core.Device("cuda", idx)


@pytest.fixture(scope="module")
def cuda_exe(cuda_device):
    """One shared cuda add executable for all Group B tests (one compile)."""
    return etl.build(_add, *ADD_SPECS, backend="iree", device=cuda_device,
                     target_backends=["cuda"])


# ---------------------------------------------------------------------------
# Group A — llvm-cpu negatives (no GPU): schedule-time validation + the
# unregistered-kind runtime failure path, all pinned verbatim
# ---------------------------------------------------------------------------


def test_prefetch_cpu_kind_payload_source_raises_with_to_cpu_hint(
        llvm_cpu_payload_out):
    # prefetch() is host→device sugar and requires ndarray-backed HOST data;
    # a cpu-kind payload tensor (llvm-cpu run output) is rejected at
    # schedule time with the explicit materialization remedy. The payload
    # class name is interpolated (it is the live tensor's own payload type).
    out = llvm_cpu_payload_out
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl_io.prefetch(out, etl.core.Device("cuda", 0))
    assert str(excinfo.value) == (
        "prefetch() copies host → device and needs ndarray-backed host "
        "data, but the source tensor is backed by "
        f"{type(out.data).__name__} on Device(kind='cpu', index=0). "
        "Materialize host data explicitly first: "
        "t.to(core.Device('cpu', 0)) (t.to(cpu))"
    )


def test_sink_cpu_kind_payload_source_raises_with_numpy_hint(
        llvm_cpu_payload_out):
    # sink() downloads device → host data; a cpu-kind payload tensor is
    # ALREADY host-readable, so the error directs to the synchronous
    # t.numpy() instead of an async download.
    out = llvm_cpu_payload_out
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl_io.sink(out)
    assert str(excinfo.value) == (
        "sink() downloads device → host data, but the source tensor sits "
        "on Device('cpu', 0) with a cpu-kind payload whose data reads "
        "directly from host memory — use t.numpy() instead of an async "
        "download"
    )


def test_prefetch_host_to_cpu_target_no_copy_needed():
    # prefetch(host_tensor, Device('cpu', 0)): host data is already at the
    # target — no copy needed (raised synchronously at schedule time).
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl_io.prefetch(etl.core.Tensor(XA), etl.core.Device("cpu", 0))
    assert str(excinfo.value) == (
        "prefetch() from Device(kind='cpu', index=0) to "
        "Device(kind='cpu', index=0): no copy needed — host data is "
        "already at the target"
    )


def test_sink_host_ndarray_source_already_host():
    # sink() of a plain ndarray-backed host tensor: already host data — no
    # download needed (synchronous schedule-time error).
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl_io.sink(etl.core.Tensor(XA))
    assert str(excinfo.value) == (
        "sink() downloads device → host data, but the source tensor is "
        "already host data (ndarray-backed on Device('cpu', 0)) — no "
        "download needed"
    )


def test_raw_ndarray_args_raise_typeerror():
    # Raw numpy ndarrays are NEVER accepted: wrap with etl.tensor(...) /
    # core.Tensor(...) first (both directions).
    with pytest.raises(TypeError) as excinfo:
        etl_io.prefetch(XA, etl.core.Device("cuda", 0))
    assert str(excinfo.value) == (
        "prefetch() expects a core.Tensor host tensor, got a raw numpy "
        "ndarray — wrap it with etl.tensor(...) (or core.Tensor(...)) first"
    )
    with pytest.raises(TypeError) as excinfo:
        etl_io.sink(XA)
    assert str(excinfo.value) == (
        "sink() expects a core.Tensor device tensor, got a raw numpy "
        "ndarray — wrap it with etl.tensor(...) (or core.Tensor(...)) first"
    )


def test_non_tensor_and_non_device_args_raise_typeerror():
    # Non-Tensor, non-ndarray args name the offending type; a non-core.Device
    # target names the type and points at etl.Device(kind, index).
    with pytest.raises(TypeError) as excinfo:
        etl_io.prefetch(5, etl.core.Device("cuda", 0))
    assert str(excinfo.value) == (
        "prefetch() expects a core.Tensor host tensor, got int"
    )
    with pytest.raises(TypeError) as excinfo:
        etl_io.sink([1, 2])
    assert str(excinfo.value) == (
        "sink() expects a core.Tensor device tensor, got list"
    )
    with pytest.raises(TypeError) as excinfo:
        etl_io.prefetch(etl.core.Tensor(XA), "cuda")
    assert str(excinfo.value) == (
        "prefetch() expects a core.Device target, got str — construct one "
        "with etl.Device(kind, index)"
    )


def test_unregistered_kind_failure_surfaces_at_result_never_swallowed():
    # Device("nonexistent_kind", 0): schedule-time validation passes (the
    # host tensor and device ARE well-formed) — the missing provider is a
    # RUNTIME failure that surfaces at result()/wait() (never swallowed):
    # .done goes True and EVERY wait() re-raises the same DeviceError.
    host = etl.core.Tensor(XA)
    target = etl.core.Device("nonexistent_kind", 0)
    handle = etl_io.prefetch(host, target)
    assert handle.device == target
    assert handle.direction == "host_to_device"
    with pytest.raises(etl.core.DeviceError) as excinfo:
        handle.result(timeout=60)
    expected = (
        "Tensor.to cannot place data on Device(kind='nonexistent_kind', "
        "index=0): no device-transfer provider is registered for device "
        "kind 'nonexistent_kind'. The etl iree backend provides cuda "
        "placement — activate it (etl.backends.get('iree')) or register a "
        "provider via core.register_device_transfer_provider."
    )
    assert str(excinfo.value) == expected
    assert handle.done  # failure is terminal on the handle, like success
    with pytest.raises(etl.core.DeviceError) as excinfo:
        handle.wait(timeout=5)  # re-raised on every call — never swallowed
    assert str(excinfo.value) == expected
    # The failed copy drained from the outstanding set: wait_all stays clean.
    assert etl_io.wait_all(timeout=30) is None


# ---------------------------------------------------------------------------
# Group B — GPU-guarded positives (real host→device / device→host copies)
# ---------------------------------------------------------------------------


def test_cuda_prefetch_equals_explicit_upload(cuda_device, cuda_exe):
    # prefetch() schedules the SAME explicit Tensor.to upload on the worker;
    # the handle's result must be bit-exact vs the synchronous .to() upload
    # and behave like any device-resident input at the run boundary.
    handle = etl_io.prefetch(etl.core.Tensor(XA), cuda_device)
    assert handle.device == cuda_device
    assert handle.direction == "host_to_device"
    dev_a = handle.result(timeout=60)
    assert handle.done
    _assert_payload_tensor(dev_a, cuda_device, XA.shape, np.dtype("float32"))
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        dev_a.numpy()  # device-resident: no implicit D2H, .to(cpu) first
    ref = etl.core.Tensor(XA).to(cuda_device)  # the synchronous upload
    np.testing.assert_array_equal(_np(dev_a), _np(ref))  # bit-exact
    # A prefetched tensor feeds a same-device run like any placed input.
    out = etl.run(cuda_exe, dev_a, etl.core.Tensor(XB).to(cuda_device))
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_cuda_sink_roundtrip_bit_exact(cuda_device, cuda_exe):
    # sink() downloads a device-resident run output to an ndarray-backed
    # cpu:0 host tensor; the source stays device-resident and untouched.
    out = etl.run(
        cuda_exe,
        etl.core.Tensor(XA).to(cuda_device),
        etl.core.Tensor(XB).to(cuda_device),
    )
    _assert_payload_tensor(out, cuda_device, XA.shape, np.dtype("float32"))
    handle = etl_io.sink(out)
    assert handle.device == cuda_device  # the SOURCE device
    assert handle.direction == "device_to_host"
    host = handle.result(timeout=60)
    assert handle.done
    assert isinstance(host, etl.Tensor)
    assert isinstance(host.data, np.ndarray)  # ndarray-backed host result
    assert host.device == etl.core.Device("cpu", 0)
    np.testing.assert_array_equal(host.numpy(), XA + XB)  # bit-exact
    # The source was not consumed or auto-copied: still device-resident.
    assert out.device == cuda_device
    assert not isinstance(out.data, np.ndarray)
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    # Sinking the same tensor again is legal and bit-exact (no state consumed).
    np.testing.assert_array_equal(etl_io.sink(out).result(timeout=60).numpy(),
                                  XA + XB)


def test_cuda_prefetch_overlaps_same_device_runs(cuda_device, cuda_exe):
    # A big prefetch (~800 KB) is issued WITHOUT waiting; same-device runs on
    # the main thread keep executing while the worker copy is queued/in
    # flight (no deadlock — the worker is a separate daemon thread). Every
    # run output stays device-resident (.numpy() raises DeviceError) until
    # the explicit .to(cpu) hop, and the uploaded payload is only
    # materialized via the explicit handle.result().
    xa = etl.core.Tensor(XA).to(cuda_device)
    xb = etl.core.Tensor(XB).to(cuda_device)
    handle = etl_io.prefetch(etl.core.Tensor(BIG), cuda_device)
    assert handle.direction == "host_to_device"
    for _ in range(40):
        out = etl.run(cuda_exe, xa, xb)
        _assert_payload_tensor(out, cuda_device, XA.shape, np.dtype("float32"))
        with pytest.raises(etl.core.DeviceError,
                           match="no implicit device-to-host"):
            out.numpy()  # outputs stay device-resident until the explicit hop
    dev_big = handle.result(timeout=60)  # the only way to obtain the upload
    _assert_payload_tensor(dev_big, cuda_device, BIG.shape, np.dtype("float32"))
    np.testing.assert_array_equal(_np(dev_big), BIG)  # bit-exact round trip
    np.testing.assert_array_equal(_np(out), XA + XB)  # runs unaffected


def test_cuda_fifo_three_prefetches_with_concurrent_runs(cuda_device,
                                                         cuda_exe):
    # Three prefetches queue on the single FIFO worker in issue order; each
    # handle stays bound to ITS OWN source (results never cross), main-thread
    # runs proceed while the copies are pending, and the prefetched results
    # are ordinary device-resident run inputs. wait_all() drains everything.
    xa = etl.core.Tensor(XA).to(cuda_device)
    xb = etl.core.Tensor(XB).to(cuda_device)
    sources = (XA, XB, XC)
    handles = [etl_io.prefetch(etl.core.Tensor(src), cuda_device)
               for src in sources]
    for _ in range(20):  # concurrent main-thread same-device runs
        out = etl.run(cuda_exe, xa, xb)
        with pytest.raises(etl.core.DeviceError,
                           match="no implicit device-to-host"):
            out.numpy()
    devs = [handle.result(timeout=60) for handle in handles]
    for dev, src in zip(devs, sources):  # issue order ↔ own source, bit-exact
        _assert_payload_tensor(dev, cuda_device, src.shape,
                               np.dtype("float32"))
        np.testing.assert_array_equal(_np(dev), src)
    # Prefetched results feed the executable like any placed inputs.
    np.testing.assert_array_equal(
        _np(etl.run(cuda_exe, devs[0], devs[1])), XA + XB
    )
    assert etl_io.wait_all(timeout=60) is None  # everything drained
