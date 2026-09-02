"""iree llvm-cpu host-input path under explicit device placement (regression
suite for the R6.2 device-placement rules — ``etl/core/tensor.py``,
``etl/core/device.py``, ``etl/pipeline.py`` ``_check_run_device``, and the
iree adapter's buffer-building path):

Explicit placement is now binding: ``Tensor.to(device)`` is the ONLY transfer
path, ``.numpy()`` on a NON-cpu-kind payload tensor (cuda) raises
``core.DeviceError``, and the run/bind boundary requires inputs to live on
the executable's device — raw host numpy arrays auto-wrap as cpu:0, so they
are ILLEGAL for cuda executables (never staged).

The llvm-cpu path is deliberately UNCHANGED — this file pins that
no-regression contract:

* plain llvm-cpu executables still accept RAW host ndarray inputs (no
  ``Tensor`` wrap): the pipeline auto-wraps them as cpu:0 and the adapter
  uploads them via ``iree.runtime.asdevicearray`` (a host->host ABI
  representation copy — monkeypatch-pinned here with a counting wrapper,
  one call per host input leaf);
* cpu:0 ``core.Tensor`` inputs are equally accepted (same-device);
* run outputs are payload-backed ``core.Tensor``\\s on
  ``core.Device("cpu", 0)`` whose host copy materializes LAZILY and
  PER-CALL: zero ``IreeDevicePayload.to_host`` invocations after ``run()``,
  one fresh materialization per ``.numpy()`` call — the Tensor never caches
  the host array (the etl-level freshness contract, pinned exactly like
  ``tests/core/test_tensor.py::test_numpy_fresh_host_copy_per_call`` counts
  a dummy payload's ``to_host``). NOTE on object identity: for MAPPABLE
  llvm-cpu memory iree's own ``DeviceArray.to_host()`` maps the buffer ONCE
  and returns the same mapped ndarray on later calls (zero-copy host
  mapping), so this suite pins the per-call re-materialization, not
  ``a is not b`` — the fresh-array-per-call guarantee at the payload level
  is iree-version-dependent by design;
* ``Tensor.to`` on the same device returns ``self`` (no copy), for
  payload-backed run outputs AND ndarray-backed cpu:0 tensors;
* even llvm-cpu executables REJECT an input tensor living on another device
  with the run-boundary ``core.DeviceError`` (no implicit device-to-host
  transfer ever happens at the run boundary).

No GPU anywhere in this file — pure llvm-cpu (one shared compile via the
per-(fn, specs) executable cache). IREE runs the real MLIR compiler (seconds
per compile), so executables are cached module-wide like the sibling suites.
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
ADD_SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)


def _add(x, y):
    return etl.add(x, y)


def _np(v):
    """Tensor / ndarray → ndarray (host copy for payload-backed tensors)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


_EXECUTABLES: dict = {}


def _build_exe(fn, specs):
    """One compile per (fn, specs) — shared module-wide (llvm-cpu default)."""
    key = (fn, specs)
    exe = _EXECUTABLES.get(key)
    if exe is None:
        exe = etl.build(fn, *specs, backend="iree")  # default llvm-cpu device
        _EXECUTABLES[key] = exe
    return exe


def _count_asdevicearray_calls(monkeypatch):
    """Wrap ``iree.runtime.asdevicearray`` with a counting recorder.

    Returns the call list; the real function is still invoked (a host->host
    representation copy must actually happen for the run to succeed).
    """
    import iree.runtime as rt

    real = rt.asdevicearray
    calls = []

    def _counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(rt, "asdevicearray", _counting)
    return calls


# ---------------------------------------------------------------------------
# raw host ndarray inputs (no Tensor wrap) — still accepted on llvm-cpu
# ---------------------------------------------------------------------------


def test_host_ndarray_inputs_accepted():
    # Raw numpy arrays (never wrapped in a core.Tensor) are auto-wrapped as
    # cpu:0 at the run boundary and accepted by the llvm-cpu executable.
    exe = _build_exe(_add, ADD_SPECS)
    out = etl.run(exe, XA, XB)
    _assert_exact(out, etl.evaluate(_add, XA, XB))  # bit-exact fp32 add


def test_host_ndarray_inputs_upload_via_asdevicearray(monkeypatch):
    # Pins the representation path: each raw host ndarray input is uploaded
    # through iree.runtime.asdevicearray (host->host ABI representation) —
    # one call per input leaf — never silently dropped or pass-through'd.
    exe = _build_exe(_add, ADD_SPECS)
    calls = _count_asdevicearray_calls(monkeypatch)
    out = etl.run(exe, XA, XB)
    assert len(calls) == 2, (
        f"expected one asdevicearray call per host input, got {len(calls)}"
    )
    _assert_exact(out, etl.evaluate(_add, XA, XB))


# ---------------------------------------------------------------------------
# cpu:0 core.Tensor inputs — same-device, accepted unchanged
# ---------------------------------------------------------------------------


def test_cpu0_tensor_inputs_accepted():
    # cpu:0 ndarray-backed core.Tensor inputs are same-device for an llvm-cpu
    # executable — accepted and bit-exact (the ndarray-backed path is
    # unchanged: still asdevicearray-hosted on llvm-cpu, never rejected).
    exe = _build_exe(_add, ADD_SPECS)
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_exact(out, etl.evaluate(_add, XA, XB))


# ---------------------------------------------------------------------------
# run outputs: payload-backed cpu:0 tensors with LAZY, per-call .numpy()
# ---------------------------------------------------------------------------


def test_cpu0_payload_output_lazy_numpy(monkeypatch):
    exe = _build_exe(_add, ADD_SPECS)
    # The run output is a DEVICE-payload-backed tensor on the executable's
    # cpu:0 device — NOT an ndarray-backed tensor.
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    assert isinstance(out, etl.Tensor)
    assert not isinstance(out.data, np.ndarray)
    assert out.device == etl.core.Device("cpu", 0)
    assert out.data.device == etl.core.Device("cpu", 0)
    assert out.shape == (4, 6)
    assert out.dtype == np.dtype("float32")

    # Lazy + per-call host materialization: count IreeDevicePayload.to_host
    # invocations (the same counting strategy the core suite uses for its
    # dummy payload). run() itself materializes NOTHING; every .numpy() call
    # re-invokes the payload's host-copy path — the Tensor never caches the
    # host array. (For mappable llvm-cpu memory iree's DeviceArray.to_host()
    # maps the buffer once and returns the same zero-copy view afterwards,
    # so we pin the per-call re-materialization — the etl-owned half of the
    # freshness contract — not ndarray object identity.)
    from etl.backends.adapters.iree import IreeDevicePayload

    real_to_host = IreeDevicePayload.to_host
    calls = []

    def _counting(self):
        calls.append(self)
        return real_to_host(self)

    monkeypatch.setattr(IreeDevicePayload, "to_host", _counting)
    out2 = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    assert calls == []  # run() never materializes the host copy
    a1 = out2.numpy()
    assert len(calls) == 1
    a2 = out2.numpy()
    assert len(calls) == 2  # one fresh materialization per .numpy() call
    # Both materializations carry the bit-exact host array.
    np.testing.assert_array_equal(a1, XA + XB)
    np.testing.assert_array_equal(a2, XA + XB)
    np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# Tensor.to on the same device returns self (no copy)
# ---------------------------------------------------------------------------


def test_to_same_device_returns_self():
    exe = _build_exe(_add, ADD_SPECS)
    cpu0 = etl.core.Device("cpu", 0)
    # Payload-backed run output: to(cpu:0) is a no-op returning the SAME
    # tensor object — never a host copy.
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    assert out.to(cpu0) is out
    # ndarray-backed cpu:0 tensors ("on the host") behave identically.
    t = etl.core.Tensor(XA)
    assert t.device == cpu0
    assert t.to(cpu0) is t


# ---------------------------------------------------------------------------
# boundary enforcement: no implicit transfer, even on llvm-cpu
# ---------------------------------------------------------------------------


def test_foreign_device_tensor_rejected_at_run_boundary():
    # A tensor living on ANOTHER device (here a synthetic cuda-kind payload —
    # no GPU needed: the pipeline run-boundary check rejects it BEFORE any
    # backend work) must raise the boundary DeviceError even for an llvm-cpu
    # executable: the explicit-placement rules allow host inputs, never a
    # silent device-to-host transfer of a foreign-device tensor.
    class _FakeCudaPayload:
        """Minimal duck-typed payload on Device('cuda', 0) (shape/dtype/device)."""

        shape = (4, 6)
        dtype = np.dtype("float32")
        device = etl.core.Device("cuda", 0)

    foreign = etl.core.Tensor(_FakeCudaPayload())
    assert foreign.device == etl.core.Device("cuda", 0)
    exe = _build_exe(_add, ADD_SPECS)
    with pytest.raises(
        etl.DeviceError,
        match=r"input at path \[0\] is on device Device\(kind='cuda', index=0\), "
              r"but the executable runs on device Device\(kind='cpu', index=0\): "
              r"no implicit .* run boundary .* t\.to\(",
    ):
        etl.run(exe, foreign, etl.core.Tensor(XB))
