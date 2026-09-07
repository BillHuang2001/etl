"""iree adapter + DEVICE-RESIDENT external kernels composed with the
batched-variant sugar (vmap/vectorize) — Workstream-A follow-up suite.

Pins the extended, MODE-AWARE result-dim contract of the iree adapter's
external-call host-dispatch path (``etl/backends/external_split.py``,
round 4+): the lower-time STATIC-dims gate now applies to HOST-mode
boundaries only — host staging genuinely needs concrete shapes — while
DEVICE-RESIDENT boundaries (``device_resident=True`` registrations)
ALLOW symbolic / runtime-dynamic result dims (e.g. the batch ``Dim``\ s
that vmap/vectorize introduce into a mapped ``external_call``'s rebuilt
result specs). Their results are validated METADATA-ONLY at run time with
the ACTUAL runtime shapes, and the stablehlo exporter's per-op
symbolic-shape support decides what compiles — never silent.

The gap this closes: triton-style device kernels vmapped over batch (the
vectorized-variant use case) previously could not reach the HAL device at
all — ``lower()`` rejected EVERY external_call with symbolic result dims
regardless of boundary mode ("adapter host-dispatch requires STATIC
(integer) result dims…"). The numpy path was always green; the iree path
now is too, for device-resident kernels:

* Gap-2 pin (``batch_variant``): the DERIVED batched kernel
  (``<name>__etl_batched``, registered ``backend="iree",
  device_resident=True``) runs through ``etl.vmap`` AND ``etl.vectorize``
  on iree llvm-cpu — bit-exact vs the numpy reference; the kernel receives
  the FULL batched stack directly as ``core.Tensor``\\ s wrapping the
  public ``IreeDevicePayload`` (never host-staged; asserted inside every
  kernel); it is invoked EXACTLY ONCE per run; ZERO host-dispatch staging
  warnings (fully device-resident boundary).
* ``batch_invariant`` declaration (original kernel name preserved in the
  rebuilt op) with a device-resident iree-slot kernel: runs bit-exact on
  llvm-cpu, single invocation.
* Gap-3 pin: the device-mode vmapped boundary is a CLEAN
  zero-host-round-trip audit — no staging UserWarning; payload-in /
  payload-out (the run output is payload-backed, never a host ndarray);
  a second run feeding the first run's device output back in performs
  ZERO ``iree.runtime.asdevicearray`` calls (monkeypatched to raise);
  bit-exact.
* Host-mode deferral (unchanged, never silent): a HOST-mode batched
  variant through vmap still raises the explicit ``BackendError``
  (``STATIC``) at ``lower()`` — v1 host-mode staging needs concrete
  shapes.
* One GPU-guarded iree-cuda smoke: a device-resident batched variant
  through vmap on a real GPU — bit-exact vs the numpy reference under the
  explicit-placement model (run input placed via ``Tensor.to(cuda)``;
  device-resident output read back via the explicit ``.to(cpu)`` hop — a
  bare ``out.numpy()`` raises ``core.DeviceError``).

The companion runtime rule (``external_split.dispatch_external_kernel``):
device-mode result validation skips SYMBOLIC dims by design (rank, dtype,
and concrete dims stay exact) — the kernel is the only source of the
runtime extent.

CUDA fixture policy: the most-free HEALTHY GPU is picked via nvidia-smi,
EXCLUDING device index 7 (the ECC-broken device on this host — never
used); iree HAL device_id = idx + 1 (1-based ids); skip when no free GPU
is available. Stand-in kernels are plain Python/numpy functions + one
tiny hand-compiled second vmfb — NO triton anywhere.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl
from etl import transforms
from etl.backends.adapters.iree import IreeDevicePayload

# ---------------------------------------------------------------------------
# data (exactly representable in fp32 -> bit-exact comparisons)
# ---------------------------------------------------------------------------

B, H, W = 4, 2, 3
X = np.arange(B * H * W, dtype=np.float32).reshape(B, H, W)  # 0..23
# The graph's row function multiplies the kernel result by 2.0 after the
# call, so the final expected values stay small integers (<= 94) — every
# intermediate is exactly representable in fp32 on every backend.
EXPECTED = (X * 2.0 + 1.0) * 2.0

_BATCH_SUFFIX = "__etl_batched"


def _derived(name):
    return f"{name}{_BATCH_SUFFIX}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / ndarray -> ndarray. Host access is explicit: payload-backed
    Tensors are host-materialized via the explicit ``to(cpu)`` transfer (a
    no-op for cpu-kind payloads whose lazy ``.numpy()`` then applies; the
    D2H hop for cuda payloads whose bare ``.numpy()`` raises DeviceError)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(etl.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


def _assert_bit_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _staging_warnings(records):
    return [
        r for r in records
        if r.category is UserWarning and "host-dispatch" in str(r.message)
    ]


def _ext_names(graph):
    """The ``external_call`` op names of a graph's main function."""
    block = graph.module.functions[0].entry_block
    return [
        op.attributes["name"] for op in block.ops if op.name == "external_call"
    ]


def _cleanup(name):
    """Remove base + derived kernel slots (``unregister_external_kernel``
    removes both) AND every transform rule the test may have installed —
    batching rules are graph-level and survive unregister by design; tests
    must clean them up so later tests start from a blank registry."""
    try:
        etl.unregister_external_kernel(name)
    except KeyError:
        pass  # the test may already have unregistered mid-test
    transforms.batching_rules.pop(f"external:{name}", None)
    transforms.batching_rules.pop(f"external:{_derived(name)}", None)


_EXECUTABLES: dict = {}


def _build_exe(key, graph, device=None, target_backends=None):
    """One iree compile per (key, device, target_backends), shared
    module-wide. Uses the EXPLICIT lower/compile/load pipeline: a vmapped
    Graph is already traced — ``etl.build`` traces callables only."""
    cache_key = (
        key,
        device,
        None if target_backends is None else tuple(target_backends),
    )
    exe = _EXECUTABLES.get(cache_key)
    if exe is None:
        lowered = etl.lower(graph, backend="iree")
        if target_backends is None:
            artifact = etl.compile(lowered)
        else:
            artifact = etl.compile(lowered, target_backends=list(target_backends))
        exe = etl.load(artifact, backend="iree", device=device)
        _EXECUTABLES[cache_key] = exe
    return exe


def _run_numpy(graph, *arrays):
    """Reference run of a vmapped Graph on the numpy backend (resolves the
    default-slot kernels — the batched-variant sugar's host registrations)."""
    return etl.run(etl.load(etl.compile(etl.lower(graph))), *arrays)


# ---------------------------------------------------------------------------
# kernels + graphs
# ---------------------------------------------------------------------------


def _row_fn(name):
    """A one-call row function: external kernel ``name`` over a (H, W)
    row, plus a trailing elementwise op so the vmapped segments carry the
    symbolic batch dims through a real stablehlo op (per-op exporter
    symbolic-shape support decides compilability)."""

    @etl.defn
    def f(x):
        y = etl.external_call(
            name, x, result=etl.TensorSpec((H, W), etl.float32)
        )
        return y * 2.0

    return f


def _vmapped_graph(name):
    """vmap (function-side sugar) of the row function over a (B, H, W)
    batch spec."""
    return etl.vmap(_row_fn(name))(etl.TensorSpec((B, H, W), etl.float32))


def _vectorized_graph(name):
    """vectorize (the graph->graph primitive) of the traced row function."""
    return etl.vectorize(
        etl.trace(_row_fn(name), etl.TensorSpec((H, W), etl.float32)), 0
    )


def _assert_batched_device_operand(x, device):
    """The device-mode operand contract for the BATCHED stack: each operand
    arrives DIRECTLY as a ``core.Tensor`` whose ``.data`` is the public
    ``IreeDevicePayload`` holding the FULL (B, H, W) stack on ``device`` —
    never a host ndarray, never host-staged at the boundary."""
    assert isinstance(x, etl.Tensor), type(x)
    assert isinstance(x.data, IreeDevicePayload), type(x.data)
    assert not isinstance(x.data, np.ndarray)
    assert tuple(x.data.shape) == (B, H, W)
    assert x.data.dtype == np.dtype(np.float32)
    assert x.data.device == device


def _make_host_kernel(calls, tag):
    """Host-mode batched kernel: records the invocation + full stack shape
    and computes ``x * 2 + 1`` (exactly representable)."""

    def kernel(x):
        calls.append((tag, tuple(np.shape(x))))
        return x * 2.0 + 1.0

    return kernel


# A tiny hand-compiled second vmfb (x * 2 + 1) on the operand's HAL device
# — the device-kernel interop pattern from test_external_call_iree.py: the
# kernel stays fully device-resident (no to_host) and returns a raw
# DeviceArray that the adapter wraps metadata-only.
_SECOND_VMFB_MLIR = f"""
module {{
  func.func public @main(%arg: tensor<{B}x{H}x{W}xf32>) -> tensor<{B}x{H}x{W}xf32> {{
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %b2 = stablehlo.broadcast_in_dim %c2, dims = [] : (tensor<f32>) -> tensor<{B}x{H}x{W}xf32>
    %m = stablehlo.multiply %arg, %b2 : tensor<{B}x{H}x{W}xf32>
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %b1 = stablehlo.broadcast_in_dim %c1, dims = [] : (tensor<f32>) -> tensor<{B}x{H}x{W}xf32>
    %r = stablehlo.add %m, %b1 : tensor<{B}x{H}x{W}xf32>
    return %r : tensor<{B}x{H}x{W}xf32>
  }}
}}
"""

_SECOND_VMFB_CACHE: dict = {}  # compiled flatbuffer bytes (lazy, once)


def _run_second_vmfb(payload):
    """Compile (once, cached) + run the tiny hand-written vmfb on the SAME
    HAL device as ``payload``; return its raw DeviceArray (x * 2 + 1)."""
    import iree.compiler as ic
    import iree.runtime as rt

    fb = _SECOND_VMFB_CACHE.get("llvm-cpu")
    if fb is None:
        fb = ic.compile_str(
            _SECOND_VMFB_MLIR,
            target_backends=["llvm-cpu"],
            input_type="stablehlo",
            extra_args=["--iree-llvmcpu-target-cpu=generic"],
        )
        _SECOND_VMFB_CACHE["llvm-cpu"] = fb
    hal_device = payload.device_array._device
    config = rt.Config(device=hal_device)
    ctx = rt.SystemContext(config=config)
    ctx.add_vm_module(rt.VmModule.copy_buffer(config.vm_instance, fb))
    return ctx.modules.module["main"](payload.device_array)


def _make_device_vmfb_kernel(calls, device):
    """DEVICE-RESIDENT batched kernel (llvm-cpu): asserts the payload
    operand contract, records the invocation, computes x * 2 + 1 via the
    second vmfb (no host round-trip inside the kernel), returns the raw
    DeviceArray."""

    def kernel(x):
        _assert_batched_device_operand(x, device)
        calls.append(("device", tuple(x.data.shape)))
        return _run_second_vmfb(x.data)

    return kernel


def _make_cuda_relay_kernel(calls, device):
    """DEVICE-RESIDENT batched kernel for iree-cuda: operands must live on
    ``device`` (asserted); the kernel computes on the operand's device and
    returns a device DeviceArray (host math + asdevicearray upload INSIDE
    the kernel — the adapter itself never stages the boundary). Matches
    the cuda relay idiom of test_external_call_iree.py."""

    def kernel(x):
        import iree.runtime as rt

        _assert_batched_device_operand(x, device)
        calls.append(("device", tuple(x.data.shape)))
        return rt.asdevicearray(
            x.data.device_array._device, x.data.to_host() * 2.0 + 1.0
        )

    return kernel


# ---------------------------------------------------------------------------
# Gap-2 pin: device-resident batch_variant through vmap/vectorize on llvm-cpu
# ---------------------------------------------------------------------------


def test_llvm_cpu_device_resident_batch_variant_through_vmap_and_vectorize():
    """Gap-2 pin. A DEVICE-RESIDENT batched variant (derived kernel
    ``<name>__etl_batched`` in the exact "iree" slot with
    ``device_resident=True``) composes with vmap AND vectorize on iree
    llvm-cpu: the vmapped graphs LOWER (symbolic batch result dims are
    allowed on device-resident boundaries — the mode-aware gate), the
    kernel receives the full batched stack directly as core.Tensors
    wrapping IreeDevicePayloads (never host-staged — asserted inside), is
    invoked exactly ONCE per run, the runs are bit-exact vs the numpy
    reference, and no host-dispatch staging UserWarning fires (fully
    device-resident boundary)."""
    name = "eb_iree_bv_gap2"
    calls = []
    handle = etl.register_external_kernel(name, _make_host_kernel(calls, "base"))
    # Default-slot HOST batched variant: the numpy-reference kernel.
    handle.batch_variant(_make_host_kernel(calls, "host"))
    # Exact-"iree"-slot DEVICE-RESIDENT batched variant: the iree kernel.
    handle.batch_variant(
        _make_device_vmfb_kernel(calls, etl.core.Device("cpu", 0)),
        backend="iree",
        device_resident=True,
    )
    try:
        # vmap path.
        graph = _vmapped_graph(name)
        assert _ext_names(graph) == [_derived(name)]
        _assert_bit_exact(_run_numpy(graph, X), EXPECTED)
        assert calls == [("host", (B, H, W))]  # full stack, ONE call

        calls.clear()
        exe = _build_exe("bv_vmap", graph)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = etl.run(exe, etl.Tensor(X))
        assert not _staging_warnings(records)  # fully device-resident boundary
        _assert_bit_exact(out, EXPECTED)
        assert calls == [("device", (B, H, W))]  # ONE full-stack device call

        # vectorize path: identical semantics through the same registry.
        calls.clear()
        vgraph = _vectorized_graph(name)
        assert _ext_names(vgraph) == [_derived(name)]
        _assert_bit_exact(_run_numpy(vgraph, X), EXPECTED)
        assert calls == [("host", (B, H, W))]

        calls.clear()
        exe_v = _build_exe("bv_vectorize", vgraph)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out_v = etl.run(exe_v, etl.Tensor(X))
        assert not _staging_warnings(records)
        _assert_bit_exact(out_v, EXPECTED)
        assert calls == [("device", (B, H, W))]
    finally:
        _cleanup(name)


def test_llvm_cpu_device_resident_batch_invariant_through_vmap():
    """``batch_invariant()`` (no callable; the rebuilt vmapped op keeps the
    ORIGINAL kernel name) with a DEVICE-RESIDENT iree-slot kernel under
    that base name runs bit-exact on llvm-cpu — the kernel handles the
    batched stack in place, invoked once per run, payload operands
    received directly."""
    name = "eb_iree_bi"
    calls = []
    handle = etl.register_external_kernel(
        name, _make_host_kernel(calls, "host")
    )
    etl.register_external_kernel(
        name,
        _make_device_vmfb_kernel(calls, etl.core.Device("cpu", 0)),
        backend="iree",
        device_resident=True,
    )
    handle.batch_invariant()  # rule under external:<name> only; name unchanged
    try:
        graph = _vmapped_graph(name)
        assert _ext_names(graph) == [name]  # base name PRESERVED
        _assert_bit_exact(_run_numpy(graph, X), EXPECTED)
        assert calls == [("host", (B, H, W))]

        calls.clear()
        exe = _build_exe("bi", graph)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = etl.run(exe, etl.Tensor(X))
        assert not _staging_warnings(records)
        _assert_bit_exact(out, EXPECTED)
        assert calls == [("device", (B, H, W))]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Gap-3 pin: device-mode vmapped boundary is a clean zero-host-round-trip
# ---------------------------------------------------------------------------


def test_llvm_cpu_vmapped_device_boundary_zero_host_round_trip(monkeypatch):
    """Gap-3 pin. The device-mode vmapped boundary is a clean audit:
    (a) no host-dispatch staging UserWarning on a fresh executable;
    (b) payload-in / payload-out — the kernel receives IreeDevicePayloads
    and the run output is itself payload-backed (never a host ndarray);
    (c) feeding the first run's device output back into the SAME
    executable performs ZERO ``iree.runtime.asdevicearray`` calls
    (monkeypatched to raise) — the device-to-device carry path and the
    kernel boundary never touch host memory; (d) bit-exact vs the numpy
    reference. The only host involvement anywhere is the FIRST run's
    llvm-cpu host-input upload (the executable-boundary representation
    copy — not a kernel-boundary stage)."""
    name = "eb_iree_nowarn"
    calls = []
    handle = etl.register_external_kernel(name, _make_host_kernel(calls, "host"))
    handle.batch_variant(_make_host_kernel(calls, "host"))
    handle.batch_variant(
        _make_device_vmfb_kernel(calls, etl.core.Device("cpu", 0)),
        backend="iree",
        device_resident=True,
    )
    try:
        graph = _vmapped_graph(name)
        exe = _build_exe("nowarn", graph)

        # (a) + (b) + (d): first run (host input upload is legal on
        # llvm-cpu), zero warnings, payload-backed output.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out1 = etl.run(exe, etl.Tensor(X))
        assert not _staging_warnings(records)
        assert isinstance(out1, etl.Tensor)
        assert isinstance(out1.data, IreeDevicePayload)  # payload-out
        assert not isinstance(out1.data, np.ndarray)
        _assert_bit_exact(out1, EXPECTED)
        assert calls == [("device", (B, H, W))]

        # (c): the second run feeds the device output straight back in —
        # asdevicearray would raise if ANY host representation copy crept
        # into the device-to-device carry path or the kernel boundary.
        import iree.runtime as rt

        def _forbid(*args, **kwargs):
            raise AssertionError(
                "asdevicearray must not be called for a device-to-device "
                "carry into a device-resident vmapped boundary"
            )

        monkeypatch.setattr(rt, "asdevicearray", _forbid)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out2 = etl.run(exe, out1)
        assert not _staging_warnings(records)
        assert isinstance(out2.data, IreeDevicePayload)
        # out2 = f(out1): the full graph f(x) = (x * 2 + 1) * 2 applied to
        # the first run's device output (still exact integers <= 378).
        _assert_bit_exact(out2, (EXPECTED * 2.0 + 1.0) * 2.0)
        assert calls == [("device", (B, H, W)), ("device", (B, H, W))]
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# Host-mode deferral: v1 host staging needs STATIC dims — never silent
# ---------------------------------------------------------------------------


def test_lower_rejects_host_mode_batched_variant_symbolic_result_dims():
    """A HOST-mode batched variant (derived kernel registered in the
    default slot only — no iree device slot) through vmap still defers at
    ``lower()`` with the explicit ``BackendError`` (uppercase ``STATIC`` +
    ``HOST-mode`` wording): v1 host-mode boundaries stage host numpy at
    the boundary and genuinely need concrete shapes — never a silent
    fallback. (Pins the pre-existing host-mode vmap'd-kernel STATIC
    deferral under the batched-variant sugar.)"""
    name = "eb_iree_host_defer"
    handle = etl.register_external_kernel(name, _make_host_kernel([], "base"))
    handle.batch_variant(_make_host_kernel([], "host"))  # default slot ONLY
    try:
        graph = _vmapped_graph(name)
        assert _ext_names(graph) == [_derived(name)]
        with pytest.raises(etl.BackendError) as exc:
            etl.lower(graph, backend="iree")
        message = str(exc.value)
        assert "STATIC" in message
        assert "HOST-mode" in message
        assert _derived(name) in message  # names the batched kernel
    finally:
        _cleanup(name)


# ---------------------------------------------------------------------------
# iree-cuda (GPU-guarded smoke): device-resident batched variant on a GPU
# ---------------------------------------------------------------------------


def _pick_cuda_device_index():
    """Most-free HEALTHY GPU index via nvidia-smi, EXCLUDING device index
    7 (the ECC-broken device on this host — never used); ``pytest.skip``
    when unavailable."""
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
            idx = int(idx)
            if idx == 7:
                continue  # ECC-broken device on this host — never used
            gpus.append((int(free_mib), idx))
        except ValueError:
            continue  # malformed line — ignore
    if not gpus:
        pytest.skip("nvidia-smi reported no usable GPUs")
    gpus.sort(reverse=True)
    return gpus[0][1]


@pytest.fixture(scope="module")
def cuda_device():
    """A free CUDA device (most-free HEALTHY GPU via nvidia-smi, index 7
    excluded); skip when unavailable."""
    idx = _pick_cuda_device_index()
    import iree.runtime as rt

    try:
        # etl Device("cuda", idx) maps to iree device_id idx + 1 (1-based ids).
        rt.get_driver("cuda").create_device(device_id=idx + 1)
    except Exception as exc:  # noqa: BLE001 — any driver/device failure skips
        pytest.skip(f"IREE cuda HAL driver or GPU {idx} unavailable: {exc}")
    return etl.core.Device("cuda", idx)


def test_cuda_device_resident_batch_variant_through_vmap_bit_exact(cuda_device):
    """GPU smoke: a DEVICE-RESIDENT batched variant through vmap on a real
    cuda device — bit-exact vs the numpy reference under the
    explicit-placement model. The run input is placed on the executable's
    device first (``Tensor.to(cuda_device)`` — a host array at the run
    boundary raises ``core.DeviceError``); the kernel asserts every
    operand arrives as a device payload ON the cuda device (never
    host-staged); the device-resident run output stays on the cuda device
    (a bare ``out.numpy()`` raises ``core.DeviceError``) and is read back
    via the explicit ``out.to(core.Device('cpu', 0))`` transfer."""
    name = "eb_iree_cuda"
    calls = []
    handle = etl.register_external_kernel(name, _make_host_kernel(calls, "host"))
    handle.batch_variant(_make_host_kernel(calls, "host"))
    handle.batch_variant(
        _make_cuda_relay_kernel(calls, cuda_device),
        backend="iree",
        device_resident=True,
    )
    try:
        graph = _vmapped_graph(name)
        assert _ext_names(graph) == [_derived(name)]
        ref = _run_numpy(graph, X)
        _assert_bit_exact(ref, EXPECTED)
        assert calls == [("host", (B, H, W))]

        calls.clear()
        exe = _build_exe(
            "cuda", graph, device=cuda_device, target_backends=["cuda"]
        )
        x_in = etl.core.Tensor(X).to(cuda_device)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = etl.run(exe, x_in)
        assert not _staging_warnings(records)
        assert isinstance(out, etl.Tensor)
        assert out.data.device == cuda_device  # stays on the GPU
        with pytest.raises(etl.core.DeviceError):
            out.numpy()  # no implicit device-to-host transfer
        _assert_bit_exact(out.to(etl.core.Device("cpu", 0)), ref)
        assert calls == [("device", (B, H, W))]  # ONE full-stack device call
    finally:
        _cleanup(name)
