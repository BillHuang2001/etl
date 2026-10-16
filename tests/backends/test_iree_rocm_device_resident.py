"""iree adapter ROCm/HIP device-resident semantics (regression suite).

The ROCm counterpart of ``test_iree_device_resident.py`` (host-input /
pass-through placement contract) plus the structural half of
``test_iree_same_device_loop.py`` (the recommended same-device iteration
pattern). Those suites guard their GPU legs on CUDA hardware via
``nvidia-smi`` and therefore skip on an AMD-only host; this one guards on a
real ROCm/HIP device through IREE's ``hip`` HAL driver
(``tests/backends/_rocm_guard.py``) and RUNS on the AMD box.

Pins the explicit-placement execution contract on the ROCm target
(``etl/backends/adapters/iree.py``: ``Device("rocm", idx)`` -> IREE ``hip``
driver ``device_id = idx + 1``; ``"hip"`` is the accepted etl alias of the
``"rocm"`` kind, same path, label preserved):

* a ``target_backends=["rocm"]`` executable runs the compiled graph on the
  AMD GPU and is bit-exact / fp32-tolerance-equal to the numpy reference;
* run outputs are payload-backed ``core.Tensor``\\ s whose ``.device`` is the
  executable's ``core.Device``; ``.numpy()`` on them RAISES
  ``core.DeviceError`` (no implicit device-to-host transfer) — the explicit
  ``t.to(core.Device("cpu", 0)).numpy()`` hop is the only host path;
* ``etl.zeros(...).to(Device("rocm", 0))`` stages host data onto the device
  and yields a rocm-resident tensor (the same explicit-readback rules apply);
* run-boundary placement (R5): a rocm executable NEVER stages host inputs —
  a cpu:0 tensor (or a raw numpy array, auto-wrapped as cpu:0) raises
  ``core.DeviceError`` naming the input path + the ``t.to(...)`` remedy;
* same-device pass-through: inputs placed once via ``.to(rocm_device)`` and
  fed back after each run go device-to-device — ``iree.runtime.asdevicearray``
  is NEVER called from the FIRST call on (monkeypatched to raise), which is
  the documented fast iteration pattern for stateful workloads.

Deliberately NO timing / benchmarking here: only short correctness checks, so
the suite stays cheap on a shared GPU box (another worker may be on it).
IREE runs the real MLIR compiler (seconds per compile), so executables are
cached per (fn, specs, device, target_backends) — one compile per distinct
graph, shared across this module.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

from tests.backends import _rocm_guard as guard

# ---------------------------------------------------------------------------
# data (deterministic; small shapes — GPU use stays short)
# ---------------------------------------------------------------------------

# Exactly representable in fp32 -> the elementwise add is bit-exact.
XA = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25
XB = np.arange(24, dtype=np.float32).reshape(4, 6) / 2.0 + 1.0
XR = np.random.default_rng(7).standard_normal((4, 6)).astype(np.float32)
XW = np.random.default_rng(11).standard_normal((6, 5)).astype(np.float32)

ADD_SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)
MLP_SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((6, 5), etl.float32),
)
STEP_SPECS = MLP_SPECS


def _add(x, y):
    return etl.add(x, y)


def _mlp_sum(x, w):
    """matmul + relu + reduce — the fp32-accumulation-order case (1e-5)."""
    return etl.sum(etl.relu(etl.dot(x, w)), axes=(1,))


def _step(x, w):
    """One loop step: a matmul/relu branch + a carried state branch."""
    return etl.relu(etl.dot(x, w)), x + 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors -> ndarray / tuple-of-ndarrays.

    Host access is EXPLICIT: payload-backed Tensors are materialized via the
    explicit transfer ``v.to(core.Device('cpu', 0)).numpy()`` — a same-device
    no-op for cpu-kind payloads, and the D2H hop for rocm payloads whose
    bare ``.numpy()`` raises ``DeviceError``.
    """
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(etl.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _place(host, device):
    """Place host numpy data on ``device`` via the explicit transfer API."""
    return etl.core.Tensor(host).to(device)


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


def _build_exe(fn, specs, device, target_backends=("rocm",)):
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
            kwargs["target_backends"] = list(target_backends)
        exe = etl.build(fn, *specs, **kwargs)
        _EXECUTABLES[key] = exe
    return exe


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rocm():
    """The ROCm etl device; ``pytest.skip`` when no HIP device is usable."""
    return guard.require_rocm()


# ---------------------------------------------------------------------------
# 1. build / run / parity / device-resident run-outs
# ---------------------------------------------------------------------------


def test_rocm_runout_device_resident_and_numpy_parity(rocm):
    exe = _build_exe(_mlp_sum, MLP_SPECS, rocm)
    out = etl.run(exe, _place(XR, rocm), _place(XW, rocm))
    # Payload-backed (never ndarray-backed), labelled with the executable's
    # rocm device, declaring the graph's output shape/dtype.
    _assert_device_resident(out, rocm)
    assert out.shape == (4,)
    assert out.dtype == np.dtype("float32")
    # fp32 reduction accumulation-order noise: within the documented budget.
    # _np host-materializes through the explicit .to(cpu) hop.
    _assert_close(
        out, etl.evaluate(_mlp_sum, XR, XW, backend="numpy"), rtol=1e-5, atol=1e-5
    )


def test_rocm_explicit_pipeline_and_readback(rocm):
    # Full explicit pipeline: trace -> lower -> compile -> load -> run, then
    # the explicit D2H readback (a bare .numpy() on a rocm payload RAISES).
    graph = etl.trace(_add, *ADD_SPECS)
    lowered = etl.lower(graph, backend="iree", target_backends=["rocm"])
    artifact = etl.compile(lowered, target_backends=["rocm"])
    exe = etl.load(artifact, backend="iree", device=rocm)
    out = etl.run(exe, _place(XA, rocm), _place(XB, rocm))
    _assert_device_resident(out, rocm)
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    host = out.to(etl.core.Device("cpu", 0))
    assert isinstance(host.data, np.ndarray)  # explicit D2H -> host memory
    assert host.device == etl.core.Device("cpu", 0)
    _assert_exact(host, etl.evaluate(_add, XA, XB, backend="numpy"))


# ---------------------------------------------------------------------------
# 2. explicit host placement of host data
# ---------------------------------------------------------------------------


def test_zeros_to_rocm_and_explicit_round_trip(rocm):
    z = etl.zeros((4, 6), etl.float32).to(rocm)
    _assert_device_resident(z, rocm)
    assert z.shape == (4, 6)
    assert z.dtype == np.dtype("float32")
    # No implicit device-to-host transfer, ever.
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        z.numpy()
    host = z.to(etl.core.Device("cpu", 0))
    assert isinstance(host.data, np.ndarray)
    assert host.device == etl.core.Device("cpu", 0)
    np.testing.assert_array_equal(host.numpy(), np.zeros((4, 6), np.float32))


# ---------------------------------------------------------------------------
# 3. run-boundary placement (R5): host inputs are NEVER staged
# ---------------------------------------------------------------------------


def test_host_inputs_rejected_at_run_boundary(rocm):
    exe = _build_exe(_add, ADD_SPECS, rocm)
    # (a) A rocm executable never stages host inputs: feeding host cpu:0
    # tensors raises DeviceError naming the input path + the remedy.
    with pytest.raises(etl.core.DeviceError) as excinfo:
        etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    msg = str(excinfo.value)
    assert "input at path [0]" in msg  # the offending input tree path
    assert (
        "no implicit device-to-host or host-to-device transfer happens at "
        "the run boundary" in msg
    )
    assert "t.to(" in msg  # the explicit placement remedy
    # (b) Raw numpy arrays auto-wrap as cpu:0 host tensors — same error.
    with pytest.raises(
        etl.core.DeviceError,
        match="no implicit device-to-host or host-to-device transfer happens "
        "at the run boundary",
    ):
        etl.run(exe, XA, XB)
    # (c) Inputs explicitly placed on the device run bit-exact.
    out = etl.run(exe, _place(XA, rocm), _place(XB, rocm))
    _assert_device_resident(out, rocm)
    _assert_exact(out, etl.evaluate(_add, XA, XB, backend="numpy"))


# ---------------------------------------------------------------------------
# 4. same-device loop: zero host staging from the FIRST call on
# ---------------------------------------------------------------------------


def test_same_device_loop_zero_asdevicearray_calls(rocm, monkeypatch):
    exe = _build_exe(_step, STEP_SPECS, rocm)
    # Placement is an explicit one-shot upload (it legitimately uses
    # asdevicearray on the rocm staging path) — do it BEFORE the counter goes
    # up, so every run below is a pure same-device pass-through.
    x = _place(XR, rocm)
    w = _place(XW, rocm)

    import iree.runtime as rt

    calls = []

    def _forbid(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "iree.runtime.asdevicearray must never be called: a rocm run with "
            "device-resident inputs is a same-device pass-through (zero host "
            "staging from the very first call)"
        )

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    first = None
    for i in range(5):  # steps 1..5: pure pass-through, first included
        branch, x = etl.run(exe, x, w)
        _assert_device_resident(branch, rocm)
        _assert_device_resident(x, rocm)
        assert x.shape == (4, 6) and x.dtype == np.dtype("float32")
        if i == 0:
            first = (branch, x)
    assert calls == []  # ZERO asdevicearray on EVERY step incl. the first

    # First-step parity vs the numpy interpreter (the reference).
    ref_branch, ref_state = etl.evaluate(_step, XR, XW, backend="numpy")
    _assert_close(first[0], ref_branch, rtol=1e-5, atol=1e-5)
    _assert_close(first[1], ref_state, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. the guard's detection half (no compile, no GPU work)
# ---------------------------------------------------------------------------


def test_guard_advertises_usable_rocm_device(rocm):
    # The authoritative capability probe (IREE's 'hip' driver) and the
    # device-index mapping the suite builds on. Pinned so a regression that
    # turns this suite into a silent skip is caught (the fixture is a skip,
    # not a failure — this test only runs when the guard did NOT skip).
    assert guard.rocm_available() is True
    assert guard.rocm_device_count() >= 1
    indices = guard.available_rocm_indices()
    assert indices, "hip driver reported devices with unreadable device_ids"
    assert guard.pick_rocm_device_index() == indices[0]
    assert 0 in indices  # iree device_id 1 -> etl Device("rocm", 0)
    assert rocm == guard.rocm_device()
    assert rocm.kind == "rocm" and rocm.index == 0
    assert guard.skip_unless_rocm is guard.require_rocm  # documented alias


# ---------------------------------------------------------------------------
# 6. the "hip" kind alias resolves to the same execution path
# ---------------------------------------------------------------------------

def test_hip_kind_alias_same_execution_path(rocm):
    guard.require_rocm()  # the module gate (also covered by the fixture)
    hip = guard.rocm_device(kind="hip")
    assert hip.kind == "hip" and etl.Device("rocm", 0).index == hip.index
    exe = _build_exe(_mlp_sum, MLP_SPECS, hip)
    out = etl.run(exe, _place(XR, hip), _place(XW, hip))
    # The alias preserves its label end to end — the payload/executable are
    # on the hip-kind device, the same code path as "rocm".
    assert out.device == hip
    assert out.device.kind == "hip"
    _assert_device_resident(out, hip)
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    _assert_close(
        out, etl.evaluate(_mlp_sum, XR, XW, backend="numpy"), rtol=1e-5, atol=1e-5
    )
    # The same placement contract applies on the alias spelling.
    with pytest.raises(
        etl.core.DeviceError,
        match="no implicit device-to-host or host-to-device transfer happens "
        "at the run boundary",
    ):
        etl.run(exe, XR, XW)
