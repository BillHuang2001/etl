"""iree adapter Vulkan/SPIR-V device-resident semantics (regression suite).

The Vulkan counterpart of ``test_iree_rocm_device_resident.py`` (explicit
placement contract: host-input rejection / pass-through / the same-device
iteration pattern) for the FOURTH first-class device kind, ``"vulkan"``
(``etl/backends/adapters/iree.py`` + ``core.Tensor.to``/``.numpy`` + the
pipeline run-boundary device check in ``etl/pipeline.py``):

* ``Device("vulkan", N)`` selects the N-th entry of IREE's enumerated vulkan
  device list — the ids are QUERIED PER-PROCESS HASHES, not the cuda/rocm
  1-based ordinals (``_acquire_vulkan_device``);
* a ``vulkan`` device infers the ``"vulkan-spirv"`` compile target
  (``etl.pipeline`` / ``etl.bench._util``), so ``etl.build(..., backend=
  "iree", device=Device("vulkan", 0))`` produces a ``target == "vulkan"``
  artifact that ``load()`` accepts on that device;
* run outputs are payload-backed ``core.Tensor``\\ s on the executable's
  ``core.Device``; ``.numpy()`` on them RAISES ``core.DeviceError`` (no
  implicit device-to-host transfer) — the explicit
  ``t.to(core.Device("cpu", 0)).numpy()`` hop is the only host path;
* run-boundary placement: a vulkan executable NEVER stages host inputs — a
  cpu:0 tensor (or a raw numpy array, auto-wrapped as cpu:0) raises
  ``core.DeviceError`` naming the input path + the ``t.to(...)`` remedy, while
  explicitly placed inputs run fine;
* same-device pass-through: inputs placed once via ``.to(vulkan_device)`` and
  fed back after each run go device-to-device — ``iree.runtime.asdevicearray``
  is NEVER called from the FIRST call on (monkeypatched to raise), the
  documented fast iteration pattern for stateful workloads;
* an out-of-range vulkan index raises the explicit ``core.BackendError``
  naming the enumerated COUNT, the enumerated device names and the requested
  index, and stating that vulkan ids are queried hashes rather than ordinals.

ENVIRONMENT (the guard skips cleanly without it): IREE 3.11's vulkan driver
HIDES software-rasterizer devices whose name begins with ``llvmpipe`` (a
hard-coded device-name denylist in ``iree_hal_vulkan_is_device_visible()``,
no flag/env override), so a host whose only Vulkan device is Mesa lavapipe
must point ``VK_ICD_FILENAMES`` at an ICD whose device name IREE accepts (a
renamed lavapipe ICD under a machine-local temp dir — never a repo path).
Validated here against such an ICD, whose enumerated device is
``lavapipe (LLVM 20.1.2, 256 bits)``. Pinning the variable also keeps the
loader away from a wedged vendor ICD: with it unset, this host's loader
dlopens a broken amdgpu ICD and ``get_driver("vulkan")`` HANGS instead of
failing. A hang is neither a skip nor a failure, so the guard enumerates in a
CHILD process under a timeout (``_probe_vulkan_devices``, following Known
Issue 16b in ``etl/backends/adapters/CONTEXT.md``: "wrap any Vulkan/DRM probe
in ``timeout``") and skips with that reason when it fires.

Deliberately NO timing / benchmarking here: only short correctness checks on
a SHARED software device. Graphs stay TINY on purpose — IREE's default SPIR-V
target profile (``vp_android_baseline_2022``) allows only 16384 bytes of
workgroup shared memory, so large reductions / transposes / convs do not
compile (Known Issue 16d). IREE runs the real MLIR compiler, so executables
are cached per ``(fn, specs, device, target_backends)`` — one compile per
distinct graph, shared across this module.
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (deterministic; small shapes — the software device stays cheap)
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
    """matmul + relu + reduce — the fp32-accumulation-order case (1e-5).

    Measured vs the numpy backend on the software vulkan device: max_abs
    ~5e-07 (well inside the fp32 budget), never bit-exact — accumulation
    order is the compiler's.
    """
    return etl.sum(etl.relu(etl.dot(x, w)), axes=(1,))


def _step(x, w):
    """One loop step: a matmul/relu branch + a carried state branch."""
    return etl.relu(etl.dot(x, w)), x + 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CPU = etl.core.Device("cpu", 0)


def _np(v):
    """Tensor / tuple-of-Tensors -> ndarray / tuple-of-ndarrays.

    Host access is EXPLICIT: payload-backed Tensors are materialized via the
    explicit transfer ``v.to(core.Device('cpu', 0)).numpy()`` — a same-device
    no-op for cpu-kind payloads, and the D2H hop for vulkan payloads whose
    bare ``.numpy()`` raises ``DeviceError``.
    """
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(_CPU).numpy())
    return np.asarray(v)


def _place(host, device):
    """Place host numpy data on ``device`` via the explicit transfer API."""
    return etl.core.Tensor(host).to(device)


def _numpy_ref(fn, *args):
    """The numpy interpreter's result as plain ndarrays (the reference)."""
    return etl.tree_map(_np, etl.evaluate(fn, *args, backend="numpy"))


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


def _build_exe(fn, specs, device, target_backends=None):
    """One compile per (fn, specs, device, target_backends) — shared module-wide.

    ``target_backends=None`` means: let the pipeline INFER the target from the
    vulkan device (``"vulkan-spirv"``) — the inference is itself pinned by the
    ``artifact.target == "vulkan"`` assertions.
    """
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
# guard
# ---------------------------------------------------------------------------

#: Why the suite skips. Names BOTH the IREE 3.11 ``llvmpipe`` device-name
#: denylist AND the ``VK_ICD_FILENAMES`` remedy (the ``_acquire_vulkan_device``
#: contract in ``etl/backends/adapters/iree.py``).
NO_VULKAN_DEVICE = (
    "no usable Vulkan device for IREE: iree.runtime.get_driver('vulkan') "
    "enumerated no devices (or could not even create a driver instance). "
    "IREE 3.11's vulkan driver HIDES software-rasterizer devices: "
    "iree_hal_vulkan_is_device_visible() drops every device whose name begins "
    "with 'llvmpipe' (a hard-coded device-name denylist with no flag or env "
    "override), so a host whose only Vulkan device is Mesa lavapipe must "
    "point the VK_ICD_FILENAMES environment variable at an ICD whose device "
    "name IREE accepts (a renamed lavapipe ICD — machine-local, never a repo "
    "path). Diagnose with "
    "iree.runtime.get_driver('vulkan').query_available_devices()"
)


def _enumerated_vulkan_devices():
    """IREE's enumerated vulkan device-info dicts (the adapter's own source).

    ``etl`` maps ``Device("vulkan", N)`` to the N-th ENUMERATED entry, so this
    is both the ground truth the error messages must name and the source the
    adapter's ``_acquire_vulkan_device`` reads. IN-PROCESS ONLY AFTER THE
    GUARD PASSED — see ``_probe_vulkan_devices`` for why the guard itself must
    probe in a child process.
    """
    import iree.runtime as rt

    return list(rt.get_driver("vulkan").query_available_devices())


#: How long the child probe may take before the guard treats the loader as
#: hung. A healthy probe finishes in well under a second (one ``iree.runtime``
#: import); the bound only has to catch the hang described below.
_PROBE_TIMEOUT_S = 30.0

_PROBE_SCRIPT = """
import json

try:
    import iree.runtime as rt

    infos = rt.get_driver("vulkan").query_available_devices()
except Exception as exc:  # driver/loader failure -> report, do not raise
    print(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}))
else:
    print(json.dumps({"devices": [
        {key: info.get(key) for key in ("name", "device_id", "path")}
        for info in infos
    ]}))
"""


def _probe_vulkan_devices():
    """Enumerate vulkan devices in a CHILD process, hard-timed.

    IN-PROCESS IS NOT SAFE ON EVERY HOST: with ``VK_ICD_FILENAMES`` unset, a
    host whose vendor ICD is wedged makes the Vulkan loader dlopen that ICD
    and ``query_available_devices()`` HANG — it neither returns nor raises
    (Known Issue 16b in ``etl/backends/adapters/CONTEXT.md``: "wrap any
    Vulkan/DRM probe in ``timeout``"). A hung probe is neither a skip nor a
    failure, so the guard runs it in a subprocess and skips when the timeout
    fires. Once the probe HAS succeeded the loader is proven healthy and the
    in-process calls the rest of this module (and the adapter itself) makes
    are safe.

    Returns ``(infos, reason)``: ``infos`` is the enumerated device-info list
    (empty when no device is usable) and ``reason`` describes a failure or a
    timeout (``""`` for a healthy enumeration).
    """
    import json
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return [], (
            f"the probe did not finish within {_PROBE_TIMEOUT_S:g}s — the "
            "Vulkan loader is HUNG, which is what this host does when "
            "VK_ICD_FILENAMES lets it dlopen a wedged vendor ICD"
        )
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        break
    if payload is None:
        detail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        return [], (
            f"the probe produced no parsable result (exit {proc.returncode}): "
            f"{detail[0][:400]}"
        )
    if "error" in payload:
        return [], f"the probe raised {payload['error']}"
    return list(payload.get("devices") or []), ""


@pytest.fixture(scope="module")
def vulkan():
    """The etl vulkan device; ``pytest.skip`` when no device is usable."""
    infos, reason = _probe_vulkan_devices()
    if not infos:
        pytest.skip(f"{NO_VULKAN_DEVICE} ({reason})" if reason else NO_VULKAN_DEVICE)
    # Activate the iree adapter for its registration side effect: ``register()``
    # fills the per-backend AND the flat-DEFAULT vulkan device-transfer slots
    # (the iree adapter is the only backend that serves a vulkan device), which
    # is what makes ``Tensor.to(Device("vulkan", N))`` resolve.
    etl.backends.get("iree")
    return etl.core.Device("vulkan", 0)


# ---------------------------------------------------------------------------
# 1. build (target inferred) / run / parity / device-resident run-outs
# ---------------------------------------------------------------------------


def test_vulkan_runout_device_resident_and_numpy_parity(vulkan):
    # NOTE: no ``target_backends`` — the vulkan device infers "vulkan-spirv".
    exe = _build_exe(_mlp_sum, MLP_SPECS, vulkan)
    assert exe.backend == "iree"
    assert exe.device == vulkan
    # The inference is real, not a default llvm-cpu artifact that happened to
    # load: load()'s artifact↔device gate accepts only a "vulkan-spirv" target.
    assert exe.backend_executable.artifact.target == "vulkan"
    out = etl.run(exe, _place(XR, vulkan), _place(XW, vulkan))
    # Payload-backed (never ndarray-backed), labelled with the executable's
    # vulkan device, declaring the graph's output shape/dtype.
    _assert_device_resident(out, vulkan)
    assert out.device.kind == "vulkan" and out.device.index == 0
    assert out.shape == (4,)
    assert out.dtype == np.dtype("float32")
    # fp32 reduction accumulation-order noise: inside the documented budget
    # (measured max_abs ~5e-07 vs the numpy backend).
    # _np host-materializes through the explicit .to(cpu) hop.
    _assert_close(out, _numpy_ref(_mlp_sum, XR, XW), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. full explicit pipeline + explicit readback
# ---------------------------------------------------------------------------


def test_vulkan_explicit_pipeline_and_readback(vulkan):
    # Trace -> lower -> compile -> load -> run, with the vulkan target passed
    # EXPLICITLY (the sugar-free spelling of the same compile).
    graph = etl.trace(_add, *ADD_SPECS)
    lowered = etl.lower(graph, backend="iree", target_backends=["vulkan-spirv"])
    artifact = etl.compile(lowered, target_backends=["vulkan-spirv"])
    assert artifact.target == "vulkan"
    exe = etl.load(artifact, backend="iree", device=vulkan)
    assert exe.device == vulkan
    out = etl.run(exe, _place(XA, vulkan), _place(XB, vulkan))
    _assert_device_resident(out, vulkan)
    # No implicit device-to-host transfer, ever.
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        out.numpy()
    # The explicit D2H hop is the only host path — and it is BIT-EXACT here:
    # the elementwise add of exactly-representable fp32 values.
    host = out.to(_CPU)
    assert isinstance(host.data, np.ndarray)  # explicit D2H -> host memory
    assert host.device == _CPU
    _assert_exact(host, _numpy_ref(_add, XA, XB))


# ---------------------------------------------------------------------------
# 3. explicit host placement of host data
# ---------------------------------------------------------------------------


def test_zeros_to_vulkan_and_explicit_round_trip(vulkan):
    # etl.zeros is a concrete creator (cpu:0 host memory); .to(vulkan) is the
    # explicit upload and yields a vulkan-resident tensor with the REAL label.
    z = etl.zeros((4, 6), etl.float32).to(vulkan)
    _assert_device_resident(z, vulkan)
    assert z.shape == (4, 6)
    assert z.dtype == np.dtype("float32")
    # A host tensor upload takes the same provider (the same one-shot copy).
    placed = _place(XA, vulkan)
    _assert_device_resident(placed, vulkan)
    assert placed.shape == XA.shape
    # No implicit device-to-host transfer, ever.
    with pytest.raises(etl.core.DeviceError, match="no implicit device-to-host"):
        z.numpy()
    # The explicit download is correct in both directions.
    host = z.to(_CPU)
    assert isinstance(host.data, np.ndarray)
    assert host.device == _CPU
    np.testing.assert_array_equal(host.numpy(), np.zeros((4, 6), np.float32))
    downloaded = placed.to(_CPU)
    assert isinstance(downloaded.data, np.ndarray)
    np.testing.assert_array_equal(downloaded.numpy(), XA)


# ---------------------------------------------------------------------------
# 4. run-boundary placement: host inputs are NEVER staged
# ---------------------------------------------------------------------------


def test_host_inputs_rejected_at_run_boundary(vulkan):
    exe = _build_exe(_add, ADD_SPECS, vulkan)
    # (a) A vulkan executable never stages host inputs: feeding host cpu:0
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
    out = etl.run(exe, _place(XA, vulkan), _place(XB, vulkan))
    _assert_device_resident(out, vulkan)
    _assert_exact(out, _numpy_ref(_add, XA, XB))


# ---------------------------------------------------------------------------
# 5. same-device loop: zero host staging from the FIRST call on
# ---------------------------------------------------------------------------


def test_same_device_loop_zero_asdevicearray_calls(vulkan, monkeypatch):
    exe = _build_exe(_step, STEP_SPECS, vulkan)
    # Placement is an explicit one-shot upload — do it BEFORE the counter goes
    # up, so every run below is a pure same-device pass-through.
    x = _place(XR, vulkan)
    w = _place(XW, vulkan)

    import iree.runtime as rt

    calls = []

    def _forbid(*args, **kwargs):
        calls.append(args)
        raise AssertionError(
            "iree.runtime.asdevicearray must never be called: a vulkan run "
            "with device-resident inputs is a same-device pass-through (zero "
            "host staging from the very first call)"
        )

    monkeypatch.setattr(rt, "asdevicearray", _forbid)
    first = None
    for i in range(5):  # steps 1..5: pure pass-through, first included
        branch, x = etl.run(exe, x, w)
        _assert_device_resident(branch, vulkan)
        _assert_device_resident(x, vulkan)
        assert branch.shape == (4, 5) and branch.dtype == np.dtype("float32")
        assert x.shape == (4, 6) and x.dtype == np.dtype("float32")
        if i == 0:
            first = (branch, x)
    assert calls == []  # ZERO asdevicearray on EVERY step incl. the first

    # First-step parity vs the numpy interpreter (the reference).
    ref_branch, ref_state = _numpy_ref(_step, XR, XW)
    _assert_close(first[0], ref_branch, rtol=1e-5, atol=1e-5)
    _assert_close(first[1], ref_state, rtol=1e-5, atol=1e-5)
    # The carried state really threaded through all five steps (a loop that
    # silently dropped the carry would still be device-resident).
    _assert_close(x, XR + 5.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. out-of-range index: the _acquire_vulkan_device enumeration contract
# ---------------------------------------------------------------------------


def test_out_of_range_vulkan_index_names_enumeration(vulkan):
    infos = _enumerated_vulkan_devices()
    assert infos, "the module fixture would have skipped"
    names = [info.get("name") for info in infos]
    count = len(infos)

    # The requested index must not exist for this pin to mean anything.
    bad_index = count + 3
    bad_device = etl.core.Device("vulkan", bad_index)

    # The (cached) vulkan artifact of a previous test — the load-time gate is
    # reached before any device work, so the failure is purely the index.
    artifact = _build_exe(_add, ADD_SPECS, vulkan).backend_executable.artifact
    assert artifact.target == "vulkan"
    # (a) load() on the out-of-range device: an explicit BackendError naming
    # the enumerated COUNT, the enumerated device NAMES and the requested
    # index — and stating that vulkan ids are queried hashes, not ordinals.
    with pytest.raises(etl.core.BackendError) as excinfo:
        etl.load(artifact, backend="iree", device=bad_device)
    msg = str(excinfo.value)
    assert f"enumerated {count} device(s)" in msg
    for name in names:
        assert name in msg  # the enumerated device names
    assert f"asks for index {bad_index}" in msg  # the requested index
    assert repr(bad_device) in msg
    assert "QUERIED HASHES" in msg and "not ordinals" in msg

    # (b) The placement provider resolves the SAME device path, so an upload to
    # the out-of-range device raises the same contract-bearing BackendError.
    with pytest.raises(etl.core.BackendError) as excinfo2:
        _place(XA, bad_device)
    msg2 = str(excinfo2.value)
    assert f"enumerated {count} device(s)" in msg2
    assert f"asks for index {bad_index}" in msg2
    assert "QUERIED HASHES" in msg2
