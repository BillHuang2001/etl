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
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

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
