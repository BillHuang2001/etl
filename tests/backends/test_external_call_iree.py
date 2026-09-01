"""iree adapter host-dispatch for external kernels (round 2) — end-to-end suite.

Pins the round-2 semantics documented in ``etl/CONTEXT.md`` ("External
kernels") and ``etl/backends/adapters/CONTEXT.md``:

* ``lower()`` SPLITS an external-call graph into segment modules + a
  kernel-call plan (payload ``stablehlo-segments``); ``compile()`` produces
  per-segment vmfbs (artifact ``iree-vmfb-segments``); the executable runs
  the segments in plan order with host staging at each kernel boundary —
  bit-exact vs the pure-graph numpy run (llvm-cpu AND iree-cuda).
* Regression: TWO SEQUENTIAL calls in one graph (the round-2 executor bug —
  ``decode_plan`` dropped ``result_outputs``, so the multi-call path raised
  ``KeyError`` at run time; single-call graphs were unaffected).
* Placement coverage: kernel mid-graph, at graph start (no leading segment),
  at graph end (kernel outputs are graph outputs), and multi-output kernels
  (incl. with a plain-op carry in between).
* Errors (all explicit, never silent): unregistered kernel at RUN time
  (``BackendError`` naming the kernel + ``register_external_kernel``);
  external_call inside a while/cond body at lower (``BackendError``
  "not splittable"); symbolic/None result dims at lower (``BackendError``
  "STATIC"); zero-operand calls at lower; xla/tvm still reject at lower
  with the round-1 message ("not yet wired").
* Persistence: the split artifact round-trips through save/load and still
  runs (the JSON-safe plan carries the kernel-call records).

Stand-in kernels are plain Python/numpy functions — NO triton anywhere.

Round 3 (per-backend dispatch + staging warning): the run path resolves
kernels through ``etl.external.get_external_kernel(name, "iree")`` — the
exact ``"iree"`` slot wins over the default slot (pinned by differentiated
outputs), and the host-dispatch staging ``UserWarning`` fires at the FIRST
external-call boundary of an executable and never again for that executable
(pinned by record counts; plain graphs without external calls never warn).

Round 4 (device-resident kernels): a kernel registered under the exact
``"iree"`` slot with ``device_resident=True`` receives the boundary
operands DIRECTLY — ``core.Tensor`` s whose ``.data`` is the public
``etl.backends.adapters.iree.IreeDevicePayload`` (never host-staged numpy)
— and may return device tensors / duck-typed payloads / raw
``DeviceArray`` s / numpy arrays / tuple-list mixes; results are validated
METADATA-ONLY (never a forced host copy), a device kernel may compile +
run a SECOND tiny vmfb on the operand's HAL device, fully device-resident
boundaries emit ZERO staging warnings (host-array results stage back and
warn once per executable), and on iree-cuda the operands stay on the GPU.
The numpy backend is unchanged: it resolves the default slot only, so each
device-mode test also registers a default-slot HOST kernel (identical math)
as the ``_run_numpy`` reference — pinning that device mode is consumed by
iree alone.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl
from etl.backends.adapters.iree import IreeDevicePayload

# ---------------------------------------------------------------------------
# kernels (plain numpy stand-ins; deterministic + exactly representable fp32)
# ---------------------------------------------------------------------------

KERNELS = {
    # (x * 2) — exactly representable
    "t_kernel_double": lambda x: x * 2.0,
    # (y + 1) — exactly representable
    "t_kernel_inc": lambda y: y + 1.0,
    # (x * 3) — exactly representable
    "t_kernel_triple": lambda x: x * 3.0,
    # two inputs: (x + w) — exactly representable
    "t_kernel_add": lambda x, w: x + w,
    # two outputs: (x * 2, x + 1)
    "t_kernel_multi": lambda x: (x * 2.0, x + 1.0),
}


@pytest.fixture(scope="module", autouse=True)
def _register_kernels():
    for name, kernel in KERNELS.items():
        etl.register_external_kernel(name, kernel)
    try:
        yield
    finally:
        for name in KERNELS:
            etl.unregister_external_kernel(name)


# ---------------------------------------------------------------------------
# data (exactly representable in fp32 -> bit-exact comparisons)
# ---------------------------------------------------------------------------

X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
X2 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
SPEC = etl.TensorSpec((3,), etl.float32)


def _specs(*specs):
    return specs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_bit_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


_EXECUTABLES: dict = {}


def _build_exe(builder, specs, device=None, target_backends=None):
    """One compile per (builder, specs, device, target_backends)."""
    key = (
        builder,
        specs,
        device,
        None if target_backends is None else tuple(target_backends),
    )
    exe = _EXECUTABLES.get(key)
    if exe is None:
        kwargs = {"backend": "iree", "device": device}
        if target_backends is not None:
            kwargs["target_backends"] = target_backends
        exe = etl.build(builder, *specs, **kwargs)
        _EXECUTABLES[key] = exe
    return exe


def _run_numpy(builder, *arrays):
    """The pure-graph numpy run (reference for the iree results)."""
    return etl.evaluate(builder, *arrays)


@contextmanager
def _device_kernel(name, host_fn, device_fn):
    """Register a DEVICE kernel under the exact "iree" slot plus an
    optional default-slot HOST kernel (the numpy-backend reference — the
    numpy backend resolves the default slot only, so device mode is
    consumed by iree alone); unregister everything on exit."""
    if host_fn is not None:
        etl.register_external_kernel(name, host_fn)
    etl.register_external_kernel(
        name, device_fn, backend="iree", device_resident=True
    )
    try:
        yield
    finally:
        etl.unregister_external_kernel(name)


def _assert_device_operand(x, device):
    """The device-mode operand contract: a ``core.Tensor`` wrapping the
    public ``IreeDevicePayload`` (never a host ndarray), on ``device``."""
    assert isinstance(x, etl.Tensor), type(x)
    assert isinstance(x.data, IreeDevicePayload), type(x.data)
    assert not isinstance(x, np.ndarray)
    assert x.data.shape == (3,)
    assert x.data.dtype == np.dtype(np.float32)
    assert x.data.device == device


# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------


def _mid_graph(x):
    y = etl.external_call(
        "t_kernel_double", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def _sequential(x):
    y1 = etl.external_call(
        "t_kernel_double", x, result=etl.TensorSpec((3,), etl.float32)
    )
    y2 = etl.external_call(
        "t_kernel_inc", y1, result=etl.TensorSpec((3,), etl.float32)
    )
    return y2 * 2.0


def _kernel_at_start(x):
    y = etl.external_call(
        "t_kernel_triple", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def _kernel_at_end(x):
    return etl.external_call(
        "t_kernel_triple", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _multi_output(x):
    a, b = etl.external_call(
        "t_kernel_multi",
        x,
        result=[
            etl.TensorSpec((3,), etl.float32),
            etl.TensorSpec((3,), etl.float32),
        ],
    )
    return a + b


def _multi_output_with_carry(x):
    a, b = etl.external_call(
        "t_kernel_multi",
        x,
        result=[
            etl.TensorSpec((3,), etl.float32),
            etl.TensorSpec((3,), etl.float32),
        ],
    )
    return etl.sum(a) + b


def _two_inputs(x, w):
    y = etl.external_call(
        "t_kernel_add", x, w, result=etl.TensorSpec((3,), etl.float32)
    )
    return y * 2.0


def _slot_graph(x):
    """Single call to a kernel registered per-backend (round 3): the run
    path must prefer the exact "iree" slot over the default slot."""
    return etl.external_call(
        "t_slot_kernel", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _staging_warn_graph(x):
    """One-call graph used ONLY by the staging-warning test (fresh
    executable — the once-per-executable warning must not be consumed by
    another test's run of the same builder)."""
    y = etl.external_call(
        "t_kernel_double", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def _plain_graph(x):
    """Plain graph (no external calls) — the negative control for the
    staging warning."""
    return x + 1.0


def _ghost(x):
    return etl.external_call(
        "t_ghost_kernel", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _while_body(c, i):
    def cond(state):
        return etl.less(
            state[1], etl.constant(etl.tensor(3, dtype=etl.int32))
        )

    def body(state):
        y = etl.external_call(
            "t_kernel_double", state[0], result=etl.TensorSpec((3,), etl.float32)
        )
        return (
            y,
            etl.add(state[1], etl.constant(etl.tensor(1, dtype=etl.int32))),
        )

    return etl.while_loop(cond, body, (c, i))[0]


def _cond_body(x):
    def true_fn(x):
        return etl.external_call(
            "t_kernel_double", x, result=etl.TensorSpec((3,), etl.float32)
        )

    return etl.cond(
        etl.constant(etl.tensor(True)), true_fn, lambda x: x, x
    )


def _symbolic_result(x):
    n = x.shape[0]
    return etl.external_call(
        "t_kernel_double", x, result=etl.TensorSpec((n, 2), etl.float32)
    )


def _none_result_dim(x):
    n = etl.Dim("n")  # pragma: no cover — never traced
    del n
    return etl.external_call(
        "t_kernel_double", x, result=etl.TensorSpec((None,), etl.float32)
    )


def _zero_operand(x):
    del x
    return etl.external_call(
        "t_zero_kernel", result=etl.TensorSpec((3,), etl.float32)
    )


# ---------------------------------------------------------------------------
# llvm-cpu end-to-end: lower -> compile -> load -> run, bit-exact vs numpy
# ---------------------------------------------------------------------------


def test_lower_produces_segments_payload_and_segments_artifact():
    graph = etl.trace(_mid_graph, SPEC)
    lowered = etl.lower(graph, backend="iree")
    assert lowered.backend == "iree"
    assert lowered.payload["format"] == "stablehlo-segments"
    segments = lowered.payload["segments"]
    plan = lowered.payload["plan"]
    assert len(segments) == 2  # [pre-call ops] + [post-call ops incl. return]
    assert plan["format_version"] == 1
    calls = [
        seg["call"] for seg in plan["segments"] if "call" in seg
    ]
    assert len(calls) == 1
    assert calls[0]["name"] == "t_kernel_double"
    assert calls[0]["result_outputs"] == [1]  # slot after the module outputs
    for segment in segments:
        assert isinstance(segment["mlir_text"], str)
        assert segment["entry_functions"] == ["main"]

    artifact = etl.compile(lowered)
    assert artifact.backend == "iree"
    assert artifact.payload["format"] == "iree-vmfb-segments"
    assert len(artifact.payload["segments"]) == 2
    exe = etl.load(artifact)
    out = etl.run(exe, etl.Tensor(X))
    _assert_bit_exact(out, _run_numpy(_mid_graph, X))


def test_llvm_cpu_mid_graph_call_bit_exact():
    exe = _build_exe(_mid_graph, _specs(SPEC))
    _assert_bit_exact(etl.run(exe, etl.Tensor(X)), _run_numpy(_mid_graph, X))


def test_llvm_cpu_sequential_calls():
    """Regression: two external_calls in one graph (kernel-result
    consumption across segments). Round-2 executor bug: ``decode_plan``
    dropped ``result_outputs`` -> ``KeyError`` at the first boundary."""
    exe = _build_exe(_sequential, _specs(SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_sequential, X)
    )


def test_llvm_cpu_kernel_at_start():
    exe = _build_exe(_kernel_at_start, _specs(SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_kernel_at_start, X)
    )


def test_llvm_cpu_kernel_at_end():
    """Kernel outputs are the graph outputs — the final segment is the
    call's operand-staging segment with no trailing plain ops."""
    exe = _build_exe(_kernel_at_end, _specs(SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_kernel_at_end, X)
    )


def test_llvm_cpu_kernel_at_start_and_end():
    @etl.defn
    def f(x):
        y = etl.external_call(
            "t_kernel_inc", x, result=etl.TensorSpec((3,), etl.float32)
        )
        return etl.external_call(
            "t_kernel_double", y, result=etl.TensorSpec((3,), etl.float32)
        )

    exe = _build_exe(f, _specs(SPEC))
    _assert_bit_exact(etl.run(exe, etl.Tensor(X)), _run_numpy(f, X))


def test_llvm_cpu_multi_output_kernel():
    exe = _build_exe(_multi_output, _specs(SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_multi_output, X)
    )


def test_llvm_cpu_multi_output_kernel_with_carry():
    exe = _build_exe(_multi_output_with_carry, _specs(SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_multi_output_with_carry, X)
    )


def test_llvm_cpu_two_graph_inputs_one_operand_each():
    """A call whose operands span multiple graph inputs (input-slot
    plumbing across the staging boundary)."""
    exe = _build_exe(_two_inputs, _specs(SPEC, SPEC))
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X), etl.Tensor(X2)),
        _run_numpy(_two_inputs, X, X2),
    )


def test_llvm_cpu_repeat_run_deterministic():
    """Same inputs + same registered kernels -> identical results across
    runs (purity contract)."""
    exe = _build_exe(_sequential, _specs(SPEC))
    first = etl.run(exe, etl.Tensor(X))
    second = etl.run(exe, etl.Tensor(X))
    _assert_bit_exact(first, second)
    _assert_bit_exact(first, _run_numpy(_sequential, X))


def test_split_artifact_save_load_roundtrip(tmp_path):
    """The segments artifact (vmfbs + JSON-safe plan) persists and re-runs."""
    graph = etl.trace(_sequential, SPEC)
    artifact = etl.compile(etl.lower(graph, backend="iree"))
    path = tmp_path / "ext_seq.etl"
    artifact.save(str(path))
    restored = etl.backends.CompiledArtifact.load(str(path))
    assert restored.payload["format"] == "iree-vmfb-segments"
    exe = etl.load(restored)
    _assert_bit_exact(etl.run(exe, etl.Tensor(X)), _run_numpy(_sequential, X))


# ---------------------------------------------------------------------------
# errors (explicit BackendError, never silent)
# ---------------------------------------------------------------------------


def test_run_unregistered_kernel_raises_naming_registry():
    exe = _build_exe(_ghost, _specs(SPEC))
    with pytest.raises(etl.BackendError, match="t_ghost_kernel"):
        etl.run(exe, etl.Tensor(X))
    # the message points at the registration API
    try:
        etl.run(exe, etl.Tensor(X))
    except etl.BackendError as exc:
        assert "register_external_kernel" in str(exc)


def test_lower_rejects_call_inside_while_body():
    graph = etl.trace(
        _while_body, SPEC, etl.TensorSpec((), etl.int32)
    )
    with pytest.raises(etl.BackendError, match="not splittable"):
        etl.lower(graph, backend="iree")


def test_lower_rejects_call_inside_cond_body():
    graph = etl.trace(_cond_body, SPEC)
    with pytest.raises(etl.BackendError, match="not splittable"):
        etl.lower(graph, backend="iree")


def test_lower_rejects_symbolic_result_dims():
    graph = etl.trace(_symbolic_result, etl.TensorSpec((etl.Dim("n"),), etl.float32))
    with pytest.raises(etl.BackendError, match="STATIC"):
        etl.lower(graph, backend="iree")


def test_lower_rejects_none_result_dim():
    graph = etl.trace(_none_result_dim, SPEC)
    with pytest.raises(etl.BackendError, match="STATIC"):
        etl.lower(graph, backend="iree")


def test_lower_rejects_zero_operand_call():
    graph = etl.trace(_zero_operand, SPEC)
    with pytest.raises(etl.BackendError, match="at least one tensor operand"):
        etl.lower(graph, backend="iree")


@pytest.mark.parametrize("backend", ["xla", "tvm"])
def test_xla_tvm_still_reject_with_round1_message(backend):
    """The round-2 host-dispatch is iree-only in v1: xla and tvm still
    reject external-call graphs at lower() with the round-1 message
    (adapter host-dispatch not yet wired). Env-dependent adapter-missing
    errors skip, mirroring the round-1 test pattern."""
    graph = etl.trace(_mid_graph, SPEC)
    with pytest.raises(etl.BackendError) as exc:
        etl.lower(graph, backend=backend)
    message = str(exc.value)
    if "external_call" not in message:
        pytest.skip(f"{backend} adapter not available in this env: {message}")
    assert "not yet wired" in message
    assert backend in message


# ---------------------------------------------------------------------------
# per-backend kernel resolution + staging warning (round 3)
# ---------------------------------------------------------------------------


def test_run_prefers_exact_iree_slot_over_default():
    """Per-backend resolution: a kernel registered under backend="iree" wins
    over the default slot at iree run time (differentiated by output)."""
    name = "t_slot_kernel"
    etl.register_external_kernel(name, lambda x: x * 2.0)  # default slot
    etl.register_external_kernel(name, lambda x: x * 5.0, backend="iree")
    try:
        exe = _build_exe(_slot_graph, _specs(SPEC))
        _assert_bit_exact(etl.run(exe, etl.Tensor(X)), X * 5.0)
    finally:
        etl.unregister_external_kernel(name)


def test_run_default_slot_only_kernel_still_works():
    """Without an exact "iree" slot, the default-slot kernel is used (the
    shared executable resolves kernels at run time, so the cached build
    serves both tests)."""
    name = "t_slot_kernel"
    etl.register_external_kernel(name, lambda x: x * 2.0)  # default slot only
    try:
        exe = _build_exe(_slot_graph, _specs(SPEC))
        _assert_bit_exact(etl.run(exe, etl.Tensor(X)), X * 2.0)
    finally:
        etl.unregister_external_kernel(name)


def test_staging_warning_once_per_executable():
    """The host-dispatch staging warning fires at the FIRST external-call
    boundary of an executable (pytest.warns) and never again for that
    executable (record count across a second run is zero)."""
    exe = _build_exe(_staging_warn_graph, _specs(SPEC))
    with pytest.warns(UserWarning, match="host-dispatch"):
        first = etl.run(exe, etl.Tensor(X))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        second = etl.run(exe, etl.Tensor(X))
    assert not [
        r for r in records
        if r.category is UserWarning and "host-dispatch" in str(r.message)
    ]
    _assert_bit_exact(first, _run_numpy(_staging_warn_graph, X))
    _assert_bit_exact(second, first)


def test_no_staging_warning_without_external_calls():
    """A plain iree graph (no external calls) never emits the staging
    warning — it exists only for the split host-dispatch path."""
    exe = _build_exe(_plain_graph, _specs(SPEC))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        out = etl.run(exe, etl.Tensor(X))
    assert not [
        r for r in records
        if r.category is UserWarning and "host-dispatch" in str(r.message)
    ]
    _assert_bit_exact(out, X + 1.0)


# ---------------------------------------------------------------------------
# device-resident kernels (round 4): device tensors in, device tensors out
# ---------------------------------------------------------------------------


_DEVICE_SECOND_VMFB_MLIR = """
module {
  func.func public @main(%arg: tensor<3xf32>) -> tensor<3xf32> {
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    %b = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %0 = stablehlo.add %arg, %b : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""

_SECOND_VMFB_CACHE: dict = {}  # compiled flatbuffer bytes (lazy, once)


def _run_second_vmfb(payload):
    """Compile (once, cached) + run a tiny hand-written vmfb on the SAME
    HAL device as ``payload`` — the device-kernel interop pattern: a
    device kernel may run further iree modules on the operand's device and
    return their raw ``DeviceArray`` (x + 2)."""
    import iree.compiler as ic
    import iree.runtime as rt

    fb = _SECOND_VMFB_CACHE.get("llvm-cpu")
    if fb is None:
        fb = ic.compile_str(
            _DEVICE_SECOND_VMFB_MLIR,
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


def _t_dev_second_vmfb(x):
    """Device kernel: run a SECOND vmfb on the operand's HAL device and
    return its raw DeviceArray (x + 2)."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return _run_second_vmfb(x.data)


def _t_dev_numpy(x):
    """Device kernel returning a HOST numpy array (explicit to_host inside
    the kernel; the boundary stages it back — the adapter never forces)."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return x.data.to_host() * 2.0


def _t_dev_mixed(x):
    """Device kernel returning a (device_array, host_array) mix."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return (_run_second_vmfb(x.data), x.data.to_host() + 1.0)


def _t_dev_core_tensor(x):
    """Device kernel returning a ``core.Tensor`` wrapping a device payload."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return etl.core.Tensor(
        IreeDevicePayload(_run_second_vmfb(x.data), x.data.device)
    )


def _t_dev_duck_payload(x):
    """Device kernel returning a duck-typed payload (a raw
    ``IreeDevicePayload`` — shape + dtype) the adapter wraps itself."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return IreeDevicePayload(_run_second_vmfb(x.data), x.data.device)


def _t_dev_double(x):
    """Device kernel for the placement tests (x + 2 via the second vmfb)."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return _run_second_vmfb(x.data)


def _t_dev_inc(x):
    """Second sequential device kernel: the operand is the FIRST kernel's
    device result (device-to-device carry, asserted inside); returns a
    DeviceArray via asdevicearray (x + 1)."""
    import iree.runtime as rt

    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return rt.asdevicearray(
        x.data.device_array._device, x.data.to_host() + 1.0
    )


def _t_dev_multi(x):
    """Device kernel with two device outputs (both via the second vmfb)."""
    _assert_device_operand(x, etl.core.Device("cpu", 0))
    return (_run_second_vmfb(x.data), _run_second_vmfb(x.data))


def _t_dev_wrong_shape(x):
    import iree.runtime as rt

    return rt.asdevicearray(
        x.data.device_array._device, np.zeros(2, np.float32)
    )


def _t_dev_wrong_dtype(x):
    import iree.runtime as rt

    return rt.asdevicearray(
        x.data.device_array._device, np.zeros(3, np.float64)
    )


def _t_dev_wrong_count(x):
    import iree.runtime as rt

    device = x.data.device_array._device
    return (
        rt.asdevicearray(device, np.zeros(3, np.float32)),
        rt.asdevicearray(device, np.zeros(3, np.float32)),
    )


def _t_dev_garbage(x):
    del x
    return "not a tensor"


def _dev_second_vmfb_graph(x):
    y = etl.external_call(
        "t_dev_second_vmfb", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def _dev_numpy_warn_graph(x):
    y = etl.external_call(
        "t_dev_numpy", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y * 2.0


def _dev_mixed_graph(x):
    a, b = etl.external_call(
        "t_dev_mixed",
        x,
        result=[
            etl.TensorSpec((3,), etl.float32),
            etl.TensorSpec((3,), etl.float32),
        ],
    )
    return a + b


def _dev_tensor_duck_graph(x):
    a = etl.external_call(
        "t_dev_core_tensor", x, result=etl.TensorSpec((3,), etl.float32)
    )
    b = etl.external_call(
        "t_dev_duck_payload", a, result=etl.TensorSpec((3,), etl.float32)
    )
    return b + 1.0


def _dev_multi_graph(x):
    a, b = etl.external_call(
        "t_dev_multi",
        x,
        result=[
            etl.TensorSpec((3,), etl.float32),
            etl.TensorSpec((3,), etl.float32),
        ],
    )
    return a + b


def _dev_sequential_graph(x):
    a = etl.external_call(
        "t_dev_double", x, result=etl.TensorSpec((3,), etl.float32)
    )
    b = etl.external_call(
        "t_dev_inc", a, result=etl.TensorSpec((3,), etl.float32)
    )
    return b + 1.0


def _dev_start_graph(x):
    y = etl.external_call(
        "t_dev_double", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def _dev_end_graph(x):
    return etl.external_call(
        "t_dev_double", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _dev_no_warn_graph(x):
    """One-call graph used ONLY by the fully-device-resident no-warning
    test (fresh executable — the once-per-executable warning state must
    not be consumed by another test's run of the same builder)."""
    y = etl.external_call(
        "t_dev_no_warn", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y * 2.0


def _dev_garbage_graph(x):
    return etl.external_call(
        "t_dev_garbage", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _dev_unregistered_graph(x):
    return etl.external_call(
        "t_dev_unregistered", x, result=etl.TensorSpec((3,), etl.float32)
    )


def _dev_cuda_graph(x):
    y = etl.external_call(
        "t_dev_cuda_relay", x, result=etl.TensorSpec((3,), etl.float32)
    )
    return y + 1.0


def test_llvm_cpu_device_kernel_receives_device_payload_and_runs_second_vmfb():
    """A device kernel receives each operand DIRECTLY — a core.Tensor
    whose .data is the public IreeDevicePayload (never a host ndarray) —
    and may compile + run a SECOND vmfb on the operand's HAL device,
    returning its raw DeviceArray (wrapped by the adapter)."""
    with _device_kernel(
        "t_dev_second_vmfb", lambda x: x + 2.0, _t_dev_second_vmfb
    ):
        exe = _build_exe(_dev_second_vmfb_graph, _specs(SPEC))
        _assert_bit_exact(
            etl.run(exe, etl.Tensor(X)), _run_numpy(_dev_second_vmfb_graph, X)
        )


def test_llvm_cpu_device_kernel_returning_numpy_and_mixed():
    """(a) A device kernel returning HOST numpy arrays gives correct
    results and the staging warning fires exactly once per executable;
    (b) a 2-output device kernel returning a (device_array, np_array) mix
    gives correct results and the warning fires."""
    with _device_kernel("t_dev_numpy", lambda x: x * 2.0, _t_dev_numpy):
        exe = _build_exe(_dev_numpy_warn_graph, _specs(SPEC))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            first = etl.run(exe, etl.Tensor(X))
        staged = [
            r for r in records
            if r.category is UserWarning and "host-dispatch" in str(r.message)
        ]
        assert len(staged) == 1
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            second = etl.run(exe, etl.Tensor(X))
        assert not [
            r for r in records
            if r.category is UserWarning and "host-dispatch" in str(r.message)
        ]
        _assert_bit_exact(first, _run_numpy(_dev_numpy_warn_graph, X))
        _assert_bit_exact(second, first)

    with _device_kernel(
        "t_dev_mixed", lambda x: (x + 2.0, x + 1.0), _t_dev_mixed
    ):
        exe = _build_exe(_dev_mixed_graph, _specs(SPEC))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = etl.run(exe, etl.Tensor(X))
        staged = [
            r for r in records
            if r.category is UserWarning and "host-dispatch" in str(r.message)
        ]
        assert len(staged) == 1
        _assert_bit_exact(out, _run_numpy(_dev_mixed_graph, X))


def test_llvm_cpu_device_kernel_returning_core_tensor_and_duck_payload():
    """A device kernel may return a core.Tensor wrapping a device payload,
    and another may return a duck-typed payload (shape + dtype, e.g. a raw
    IreeDevicePayload instance) the adapter wraps itself — both bit-exact
    (the second call's operand is the first call's device result)."""
    with _device_kernel(
        "t_dev_core_tensor", lambda x: x + 2.0, _t_dev_core_tensor
    ):
        with _device_kernel(
            "t_dev_duck_payload", lambda x: x + 2.0, _t_dev_duck_payload
        ):
            exe = _build_exe(_dev_tensor_duck_graph, _specs(SPEC))
            _assert_bit_exact(
                etl.run(exe, etl.Tensor(X)),
                _run_numpy(_dev_tensor_duck_graph, X),
            )


def test_llvm_cpu_device_multi_output_kernel():
    """Device-mode variant of the multi-output placement coverage: two
    device results occupy the plan's result slots and recombine."""
    with _device_kernel(
        "t_dev_multi", lambda x: (x + 2.0, x + 2.0), _t_dev_multi
    ):
        exe = _build_exe(_dev_multi_graph, _specs(SPEC))
        _assert_bit_exact(
            etl.run(exe, etl.Tensor(X)), _run_numpy(_dev_multi_graph, X)
        )


def test_llvm_cpu_device_sequential_calls():
    """Two sequential DEVICE calls: the first kernel's device result is
    passed device-to-device into the second kernel (asserted inside)."""
    with _device_kernel("t_dev_double", lambda x: x + 2.0, _t_dev_double):
        with _device_kernel("t_dev_inc", lambda x: x + 1.0, _t_dev_inc):
            exe = _build_exe(_dev_sequential_graph, _specs(SPEC))
            _assert_bit_exact(
                etl.run(exe, etl.Tensor(X)),
                _run_numpy(_dev_sequential_graph, X),
            )


def test_llvm_cpu_device_kernel_at_start():
    """Device-mode variant of the graph-start placement coverage."""
    with _device_kernel("t_dev_double", lambda x: x + 2.0, _t_dev_double):
        exe = _build_exe(_dev_start_graph, _specs(SPEC))
        _assert_bit_exact(
            etl.run(exe, etl.Tensor(X)), _run_numpy(_dev_start_graph, X)
        )


def test_llvm_cpu_device_kernel_at_end():
    """Device-mode variant of the graph-end placement coverage: the kernel
    outputs are the graph outputs (device results occupy the final result
    slots directly)."""
    with _device_kernel("t_dev_double", lambda x: x + 2.0, _t_dev_double):
        exe = _build_exe(_dev_end_graph, _specs(SPEC))
        _assert_bit_exact(
            etl.run(exe, etl.Tensor(X)), _run_numpy(_dev_end_graph, X)
        )


def test_no_staging_warning_for_fully_device_resident_boundary():
    """A FULLY device-resident boundary (device kernel, device results)
    never stages tensors host<->device — ZERO host-dispatch warnings."""
    with _device_kernel("t_dev_no_warn", lambda x: x + 2.0, _t_dev_second_vmfb):
        exe = _build_exe(_dev_no_warn_graph, _specs(SPEC))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            out = etl.run(exe, etl.Tensor(X))
        assert not [
            r for r in records
            if r.category is UserWarning and "host-dispatch" in str(r.message)
        ]
        _assert_bit_exact(out, _run_numpy(_dev_no_warn_graph, X))


def test_device_wrong_shape_result_raises_shape_error():
    """Metadata-only validation still checks shape: a device array of
    shape (2,) when (3,) is declared -> ShapeError."""
    with _device_kernel("t_dev_garbage", None, _t_dev_wrong_shape):
        exe = _build_exe(_dev_garbage_graph, _specs(SPEC))
        with pytest.raises(etl.ShapeError, match="callback returned shape"):
            etl.run(exe, etl.Tensor(X))


def test_device_wrong_dtype_result_raises_backend_error():
    """Metadata-only validation still checks dtype: an f64 device array
    when f32 is declared -> BackendError (never silent coercion)."""
    with _device_kernel("t_dev_garbage", None, _t_dev_wrong_dtype):
        exe = _build_exe(_dev_garbage_graph, _specs(SPEC))
        with pytest.raises(etl.BackendError, match="no silent dtype coercion"):
            etl.run(exe, etl.Tensor(X))


def test_device_wrong_count_result_raises_backend_error():
    """Two device results against one declared spec -> BackendError with
    the canonical count wording."""
    with _device_kernel("t_dev_garbage", None, _t_dev_wrong_count):
        exe = _build_exe(_dev_garbage_graph, _specs(SPEC))
        with pytest.raises(
            etl.BackendError, match=r"produced 2 output\(s\), expected 1"
        ):
            etl.run(exe, etl.Tensor(X))


def test_device_garbage_return_raises_backend_error():
    """A garbage non-payload return (a string) -> BackendError naming the
    call — the dispatch path never guesses."""
    with _device_kernel("t_dev_garbage", None, _t_dev_garbage):
        exe = _build_exe(_dev_garbage_graph, _specs(SPEC))
        with pytest.raises(etl.BackendError, match="callback returned str"):
            etl.run(exe, etl.Tensor(X))


def test_device_unregistered_kernel_raises_naming_registry():
    """Device-mode flavor of the run-time registry error: a name that WAS
    registered (device kernel) and then unregistered raises BackendError
    naming the kernel + register_external_kernel."""
    name = "t_dev_unregistered"
    etl.register_external_kernel(
        name, _t_dev_double, backend="iree", device_resident=True
    )
    etl.unregister_external_kernel(name)
    exe = _build_exe(_dev_unregistered_graph, _specs(SPEC))
    with pytest.raises(etl.BackendError, match="t_dev_unregistered"):
        etl.run(exe, etl.Tensor(X))
    try:
        etl.run(exe, etl.Tensor(X))
    except etl.BackendError as exc:
        assert "register_external_kernel" in str(exc)


# ---------------------------------------------------------------------------
# iree-cuda (GPU-guarded): same host-dispatch semantics on a real device
# ---------------------------------------------------------------------------


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


def test_cuda_mid_graph_call_bit_exact(cuda_device):
    exe = _build_exe(
        _mid_graph, _specs(SPEC), device=cuda_device, target_backends=["cuda"]
    )
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_mid_graph, X)
    )


def test_cuda_sequential_calls_bit_exact(cuda_device):
    exe = _build_exe(
        _sequential, _specs(SPEC), device=cuda_device, target_backends=["cuda"]
    )
    _assert_bit_exact(
        etl.run(exe, etl.Tensor(X)), _run_numpy(_sequential, X)
    )


def _make_cuda_relay_kernel(device):
    """Device kernel for the cuda test: operands must live on ``device``
    (asserted); the kernel computes on the operand's device and returns a
    device DeviceArray (host math + asdevicearray upload inside the
    kernel — the adapter itself never stages the boundary)."""

    def kernel(x):
        import iree.runtime as rt

        _assert_device_operand(x, device)
        return rt.asdevicearray(
            x.data.device_array._device, x.data.to_host() * 2.0
        )

    return kernel


def test_cuda_device_kernel_bit_exact_and_operands_device_resident(cuda_device):
    """Device-resident mode on a real GPU: operands arrive as device
    payloads living on the cuda device (asserted inside the kernel), a
    device result comes back, and the graph output — still on the cuda
    device — is bit-exact vs the numpy reference."""
    name = "t_dev_cuda_relay"
    with _device_kernel(
        name, lambda x: x * 2.0, _make_cuda_relay_kernel(cuda_device)
    ):
        exe = _build_exe(
            _dev_cuda_graph,
            _specs(SPEC),
            device=cuda_device,
            target_backends=["cuda"],
        )
        out = etl.run(exe, etl.Tensor(X))
        assert isinstance(out, etl.Tensor)
        assert out.data.device == cuda_device
        _assert_bit_exact(out, _run_numpy(_dev_cuda_graph, X))
