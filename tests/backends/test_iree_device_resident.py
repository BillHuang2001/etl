"""iree adapter device-resident tensor semantics (regression suite).

Pins the device-resident execution contract (``etl/backends/adapters/iree.py``):

* run outputs on BOTH llvm-cpu and cuda are payload-backed ``core.Tensor``\\ s:
  ``.data`` is the duck-typed device payload (the iree DeviceArray), never a
  numpy ndarray; ``.device`` is the executable's ``core.Device``; the host
  copy happens LAZILY in ``.numpy()``.
* same-device input pass-through: a run-out tensor fed back into an iree
  executable's ``run()`` goes device-to-device — ``iree.runtime.asdevicearray``
  is NEVER called for it (monkeypatched to raise).
* host numpy inputs still work on both targets.
* the numpy backend is untouched: ndarray-backed outputs, zero-copy
  same-reference ``.numpy()``.

SEMANTICS ONLY: nothing here depends on the internal host-input upload
mechanism (persistent pinned buffers vs ``rt.asdevicearray``). The llvm-cpu
host-input test asserts ``asdevicearray`` IS used on that target (the current
observable contract there); the cuda host-input test asserts correctness
without pinning the upload path. GPU tests additionally guard on a free
device scanned via nvidia-smi (most-free GPU; etl ``Device("cuda", idx)``
maps to iree device_id ``idx + 1``, 1-based).

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
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
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
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_device_resident(out, cuda_device)
    assert out.dtype == np.dtype("float32")
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_cuda_reduce_runout_dtype_shape(cuda_device):
    exe = _build_exe(_row_sum, SUM_SPECS, device=cuda_device, target_backends=["cuda"])
    out = etl.run(exe, etl.core.Tensor(XR))
    _assert_device_resident(out, cuda_device)
    assert out.shape == (4,)  # reduce over axis 1 of (4, 8)
    assert out.dtype == np.dtype("float32")
    # fp32 reduction accumulation-order noise: within tolerance of numpy.
    _assert_close(out, etl.evaluate(_row_sum, XR), rtol=1e-5, atol=1e-5)


def test_cuda_host_input_works(cuda_device):
    exe = _build_exe(_add, ADD_SPECS, device=cuda_device, target_backends=["cuda"])
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_cuda_passthrough_zero_asdevicearray_calls(cuda_device, monkeypatch):
    exe = _build_exe(_add, ADD_SPECS, device=cuda_device, target_backends=["cuda"])
    out1 = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    out2 = etl.run(exe, etl.core.Tensor(XB), etl.core.Tensor(XA))

    def _forbid(*args, **kwargs):
        raise AssertionError(
            "asdevicearray must not be called for a same-device "
            "pass-through input"
        )

    import iree.runtime as rt

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    out3 = etl.run(exe, out1, out2)
    _assert_exact(out3, etl.evaluate(_add, XA + XB, XB + XA))
