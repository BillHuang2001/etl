"""iree adapter device-resident tensor semantics (regression suite).

Pins the EXPLICIT device-placement execution contract
(``etl/backends/adapters/iree.py`` + ``core.Tensor.to``/``.numpy`` + the
pipeline run-boundary device check in ``etl/pipeline.py``):

* run outputs on BOTH llvm-cpu and cuda are payload-backed ``core.Tensor``\\ s:
  ``.data`` is the duck-typed device payload (the iree DeviceArray), never a
  numpy ndarray; ``.device`` is the executable's ``core.Device``.
* host access is EXPLICIT: a CPU-kind payload's (llvm-cpu) ``.numpy()``
  materializes a LAZY fresh host copy on demand; a NON-CPU-kind payload's
  (cuda) ``.numpy()`` RAISES ``core.DeviceError`` — no implicit
  device-to-host transfer — the explicit
  ``t.to(core.Device('cpu', 0)).numpy()`` hop is the only host path.
* run-boundary placement (R5): every tensor input to ``etl.run`` must ALREADY
  live on the executable's device. Host numpy inputs to a cuda executable are
  ILLEGAL — feeding a cpu:0 tensor (or a raw numpy array, auto-wrapped as
  cpu:0) raises ``core.DeviceError`` naming the input path; inputs must be
  placed explicitly via ``t.to(core.Device('cuda', N))`` before ``run``.
* same-device input pass-through: a device-resident tensor fed into an iree
  executable's ``run()`` goes device-to-device — ``iree.runtime.asdevicearray``
  is NEVER called for it (monkeypatched to raise). Under the
  explicit-placement model EVERY cuda call with device-resident inputs is such
  a pass-through: ZERO ``asdevicearray`` calls from the very first call.
* llvm-cpu executables still accept host cpu:0 ndarray inputs via
  ``asdevicearray`` (host→host representation copy, unchanged).
* the numpy backend is untouched: ndarray-backed outputs, zero-copy
  same-reference ``.numpy()``.

SEMANTICS ONLY: nothing here depends on the internal host-input upload
mechanism (persistent pinned buffers vs ``rt.asdevicearray``). The llvm-cpu
host-input test asserts ``asdevicearray`` IS used on that target (the current
observable contract there); the cuda host-input test pins the run-boundary
``DeviceError`` (host inputs never reach the executable). GPU tests
additionally guard on a free device scanned via nvidia-smi (most-free GPU;
etl ``Device("cuda", idx)`` maps to iree device_id ``idx + 1``, 1-based).

IREE runs the real MLIR compiler (seconds per compile), so executables are
cached per (fn, specs, device, target_backends) — one compile per distinct
graph, shared across the tests in this module.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (deterministic; exactly representable in fp32 → bit-exact elementwise)
# ---------------------------------------------------------------------------

XA = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25
XB = np.arange(24, dtype=np.float32).reshape(4, 6) / 2.0 + 1.0
XR = np.random.default_rng(7).standard_normal((4, 8)).astype(np.float32)

ADD_SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)
SUM_SPECS = (etl.TensorSpec((4, 8), etl.float32),)


def _add(x, y):
    return etl.add(x, y)


def _row_sum(x):
    return etl.sum(x, axes=(1,))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays.

    Host access is explicit: payload-backed Tensors are host-materialized via
    the explicit transfer ``v.to(core.Device('cpu', 0)).numpy()`` — a
    no-op (returns self) for cpu-kind payloads whose lazy ``.numpy()`` then
    applies, and the explicit D2H hop for non-cpu-kind (cuda) payloads whose
    ``.numpy()`` would raise ``DeviceError``. ndarray-backed Tensors are
    already host memory (cpu:0 → ``to`` returns self → same zero-copy
    reference). The numpy-backend zero-copy assertion
    (``test_numpy_backend_untouched``) calls ``.numpy()`` directly and is
    unaffected.
    """
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(etl.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _assert_close(got, want, rtol=1e-5, atol=1e-5):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.allclose(gp, wp, rtol=rtol, atol=atol), f"{gp} != {wp}"


def _assert_device_resident(out, device):
    """A run-out tensor is payload-backed, on ``device``, with a lazy host copy.

    Duck-typed payload assertions only (the payload class is private): not an
    ndarray, knows its shape/dtype/device, and exposes the ``to_host`` path.
    """
    assert isinstance(out, etl.Tensor)
    assert not isinstance(out.data, np.ndarray)  # NOT a numpy-backed tensor
    assert out.device == device  # the executable's core.Device label
    assert hasattr(out.data, "to_host")  # duck-typed payload protocol
    assert out.data.device == device  # the payload knows its device
    assert out.shape == out.data.shape
    assert out.dtype == out.data.dtype


_EXECUTABLES: dict = {}


def _build_exe(fn, specs, device=None, target_backends=None):
    """One compile per (fn, specs, device, target_backends) — shared module-wide."""
    key = (
        fn,
        specs,
        device,
        None if target_backends is None else tuple(target_backends),
    )
    exe = _EXECUTABLES.get(key)
    if exe is None:
        kwargs = {"backend": "iree", "device": device}
        if target_backends is not None:
            kwargs["target_backends"] = target_backends
        exe = etl.build(fn, *specs, **kwargs)
        _EXECUTABLES[key] = exe
    return exe


def _pick_cuda_device_index():
    """Most-free GPU index via nvidia-smi; ``pytest.skip`` when unavailable."""
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
            gpus.append((int(free_mib), int(idx)))
        except ValueError:
            continue  # malformed line — ignore
    if not gpus:
        pytest.skip("nvidia-smi reported no GPUs")
    gpus.sort(reverse=True)
    return gpus[0][1]


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


# ---------------------------------------------------------------------------
# llvm-cpu: device-resident run-out semantics (no GPU needed)
# ---------------------------------------------------------------------------


def test_llvm_cpu_runout_device_resident():
    # Full explicit pipeline: trace -> lower -> compile -> load -> run.
    graph = etl.trace(_add, *ADD_SPECS)
    lowered = etl.lower(graph, backend="iree")
    artifact = etl.compile(lowered)
    exe = etl.load(artifact, backend="iree")
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_device_resident(out, etl.core.Device("cpu", 0))
    assert out.shape == (4, 6)
    assert out.dtype == np.dtype("float32")
    # .numpy() materializes the correct host array — bit-exact vs the numpy
    # backend (the elementwise add of exactly-representable fp32 values).
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_llvm_cpu_passthrough_zero_asdevicearray_calls(monkeypatch):
    exe = _build_exe(_add, ADD_SPECS)
    out1 = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    out2 = etl.run(exe, etl.core.Tensor(XB), etl.core.Tensor(XA))

    def _forbid(*args, **kwargs):
        raise AssertionError(
            "asdevicearray must not be called for a same-device "
            "pass-through input"
        )

    import iree.runtime as rt

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    # BOTH run-out device tensors feed back into the SAME executable with no
    # host round-trip (asdevicearray would raise if the path regressed).
    out3 = etl.run(exe, out1, out2)
    _assert_exact(out3, etl.evaluate(_add, XA + XB, XB + XA))


def test_llvm_cpu_host_input_uses_asdevicearray(monkeypatch):
    exe = _build_exe(_add, ADD_SPECS)
    import iree.runtime as rt

    real = rt.asdevicearray
    calls = []

    def _counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(rt, "asdevicearray", _counting)
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    assert len(calls) >= 1  # numpy-backed host inputs upload via asdevicearray
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_numpy_backend_untouched():
    # Belt-and-braces for the payload refactor: the numpy backend still
    # returns ndarray-backed Tensors with a zero-copy same-reference .numpy().
    exe = etl.build(_add, *ADD_SPECS)  # default numpy backend
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    assert isinstance(out.data, np.ndarray)
    assert out.numpy() is out.data  # zero-copy same-reference
    np.testing.assert_array_equal(out.data, XA + XB)


# ---------------------------------------------------------------------------
# iree-cuda (GPU-guarded): same run-out semantics on a real device
# ---------------------------------------------------------------------------


def test_cuda_runout_device_resident(cuda_device):
    exe = _build_exe(_add, ADD_SPECS, device=cuda_device, target_backends=["cuda"])
    out = etl.run(
        exe,
        etl.core.Tensor(XA).to(cuda_device),
        etl.core.Tensor(XB).to(cuda_device),
    )
    _assert_device_resident(out, cuda_device)
    assert out.dtype == np.dtype("float32")
    # Semantic pin (explicit-placement model): a cuda run-out payload tensor
    # never transfers implicitly — .numpy() raises DeviceError (no implicit
    # device-to-host transfer); the explicit .to(cpu) hop is the only host
    # path and round-trips bit-exact vs the numpy backend.
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    host = out.to(etl.core.Device("cpu", 0))
    assert isinstance(host.data, np.ndarray)  # explicit D2H → host memory
    assert host.device == etl.core.Device("cpu", 0)
    _assert_exact(host, etl.evaluate(_add, XA, XB))


def test_cuda_reduce_runout_dtype_shape(cuda_device):
    exe = _build_exe(_row_sum, SUM_SPECS, device=cuda_device, target_backends=["cuda"])
    out = etl.run(exe, etl.core.Tensor(XR).to(cuda_device))
    _assert_device_resident(out, cuda_device)
    assert out.shape == (4,)  # reduce over axis 1 of (4, 8)
    assert out.dtype == np.dtype("float32")
    # fp32 reduction accumulation-order noise: within tolerance of numpy.
    # _np host-materializes via .to(cpu) — a cuda output's .numpy() would raise.
    _assert_close(out, etl.evaluate(_row_sum, XR), rtol=1e-5, atol=1e-5)


def test_cuda_host_inputs_require_explicit_placement(cuda_device):
    exe = _build_exe(_add, ADD_SPECS, device=cuda_device, target_backends=["cuda"])
    # (a) Run-boundary contract (R5): a cuda executable NEVER stages host
    # inputs — feeding host cpu:0 tensors raises DeviceError naming the input
    # path and the explicit placement remedy, before the backend dispatches.
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    msg = str(excinfo.value)
    assert "input at path [0]" in msg  # the offending input tree path
    assert (
        "no implicit device-to-host or host-to-device transfer happens at "
        "the run boundary" in msg
    )
    assert "t.to(" in msg  # the explicit placement remedy
    # Raw numpy arrays auto-wrap as cpu:0 host tensors — same boundary error.
    with pytest.raises(
        etl.core.DeviceError,
        match="no implicit device-to-host or host-to-device transfer happens "
        "at the run boundary",
    ):
        etl.run(exe, XA, XB)
    # (b) Inputs placed explicitly via .to(cuda_device) run bit-exact.
    out = etl.run(
        exe,
        etl.core.Tensor(XA).to(cuda_device),
        etl.core.Tensor(XB).to(cuda_device),
    )
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_cuda_passthrough_zero_asdevicearray_calls(cuda_device, monkeypatch):
    exe = _build_exe(_add, ADD_SPECS, device=cuda_device, target_backends=["cuda"])
    # Place BOTH inputs explicitly BEFORE the monkeypatch: under the
    # explicit-placement model every cuda call with device-resident inputs is
    # a same-device pass-through — ZERO asdevicearray calls from the very
    # FIRST call (the monkeypatch raises if the path ever regresses).
    xa = etl.core.Tensor(XA).to(cuda_device)
    xb = etl.core.Tensor(XB).to(cuda_device)

    def _forbid(*args, **kwargs):
        raise AssertionError(
            "asdevicearray must never be called: a cuda run with "
            "device-resident inputs is a same-device pass-through (zero "
            "host staging from the very first call)"
        )

    import iree.runtime as rt

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    out1 = etl.run(exe, xa, xb)
    # A run-out device tensor fed straight back in is likewise a same-device
    # pass-through with no host staging.
    out2 = etl.run(exe, out1, xa)
    # Host comparison goes through the explicit .to(cpu) hop (_np).
    _assert_exact(out1, etl.evaluate(_add, XA, XB))
    _assert_exact(out2, etl.evaluate(_add, XA + XB, XA))
