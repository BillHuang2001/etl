"""iree coverage for the ``build_cached``/``CompileCache`` memoized-build
sugar (Workstream C "True JIT" — ``etl/build_cache.py``; import from
``etl.build_cache``, the top-level ``etl`` namespace does not export it
yet). Public API under test:

``build_cached(fn, *specs, backend=None, device=None, cache=None, **options)``
is documented shorthand for a cache-guarded ``etl.build`` whose exact
``trace -> lower -> compile -> load`` composition runs AT MOST ONCE per
distinct cache key — (fn identity, the full ``*specs`` tree incl. static
content, the resolved backend name, the resolved device, canonicalized
resolved options) — later calls return the SAME fully loaded Executable
object with zero pipeline work. ``CompileCache`` is the LRU table
(maxsize 128, thread-safe, ``hits``/``misses`` counters, ``__len__``).

Coverage here (every test uses a FRESH ``CompileCache()`` — never the
process-global default; real iree compiles run, so the <2s-per-file rule
does not apply):

1. compile-once on iree-llvm-cpu — two identical ``build_cached`` calls
   return the SAME object (is-identity) with misses==1/hits==1, and the
   run matches a pure-numpy fp64 reference (rtol/atol 1e-5);
2. ``opt_level`` participates in the key — "O0" vs "O3" are two distinct
   entries (misses==2) that BOTH run with parity;
3. semantic-equality hit — the DEFAULT device resolution (cpu:0) vs the
   EXPLICIT spelling ``etl.core.Device("cpu", 0)`` (and the plain ``"cpu"``
   kind string, which normalizes to the same Device) key identically: one
   entry, is-identity. NOTE on the device spelling: a bare ``"cpu:0"``
   string is NOT a device kind in etl — ``Device("cpu:0")`` would construct
   the bogus kind ``"cpu:0"`` and the load stage would reject it with
   ``BackendError`` (never cached); the explicit form of the default cpu:0
   device is the ``Device`` object (or the ``"cpu"`` kind string);
4. ONE GPU-guarded same-device inner-loop test over a CACHED cuda
   executable: nvidia-smi most-free HEALTHY-device scan (the known
   ECC-broken index 7 on this host is excluded; HAL device_id = index + 1;
   pytest.skip when no free GPU; never occupies the GPU long). The step
   graph is built ONCE via ``build_cached`` (misses==1), the initial host
   key/state are placed on the device via the explicit ``Tensor.to``
   upload, ~5 iterations feed each step's device-resident outputs back as
   inputs (the same-device loop pattern — zero host round-trips), outputs
   are read back through the explicit ``.to(cpu)`` hop, and each step is
   compared to an iterated numpy-backend reference (f32 state within the
   documented normal fast-path budget; splitmix64 keys bit-exact).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl
from etl.build_cache import CompileCache, build_cached

# ---------------------------------------------------------------------------
# llvm-cpu graph (deterministic elementwise fp32; trivial numpy reference)
# ---------------------------------------------------------------------------
SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)

XA = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25
XB = np.arange(24, dtype=np.float32).reshape(4, 6) / 2.0 + 1.0


@etl.defn
def mix_fn(x, y):
    """Elementwise fp32 graph: x + 0.5 * y."""
    return etl.add(x, etl.multiply(y, 0.5))


def _mix_reference():
    """Pure-numpy fp64 reference for mix_fn (same formula as the kernel)."""
    return XA.astype(np.float64) + 0.5 * XB.astype(np.float64)


def _np(v):
    """Tensor / ndarray -> ndarray (device-resident -> explicit .to(cpu) hop)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.to(etl.core.Device("cpu", 0)).numpy())
    return np.asarray(v)


# ---------------------------------------------------------------------------
# 1. compile-once on iree-llvm-cpu
# ---------------------------------------------------------------------------
def test_compile_once_llvm_cpu_same_object_and_parity():
    cache = CompileCache()
    exe = build_cached(mix_fn, *SPECS, backend="iree", cache=cache)
    again = build_cached(mix_fn, *SPECS, backend="iree", cache=cache)
    assert again is exe  # memoized: the SAME fully loaded executable object
    assert cache.misses == 1 and cache.hits == 1
    assert len(cache) == 1
    got = _np(etl.run(exe, XA, XB))
    assert np.allclose(got, _mix_reference(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. opt_level participates in the key
# ---------------------------------------------------------------------------
def test_opt_level_participates_in_key():
    cache = CompileCache()
    exe_o0 = build_cached(mix_fn, *SPECS, backend="iree", cache=cache,
                          opt_level="O0")
    exe_o3 = build_cached(mix_fn, *SPECS, backend="iree", cache=cache,
                          opt_level="O3")
    assert exe_o0 is not exe_o3  # distinct entries: opt_level is in the key
    assert cache.misses == 2 and cache.hits == 0
    assert len(cache) == 2
    again = build_cached(mix_fn, *SPECS, backend="iree", cache=cache,
                         opt_level="O3")
    assert again is exe_o3  # same opt_level -> hit
    assert cache.misses == 2 and cache.hits == 1
    ref = _mix_reference()
    for exe in (exe_o0, exe_o3):
        assert np.allclose(_np(etl.run(exe, XA, XB)), ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. semantic-equality hit: default device resolution vs explicit cpu:0
# ---------------------------------------------------------------------------
def test_default_and_explicit_cpu_device_key_identically():
    cache = CompileCache()
    exe_default = build_cached(mix_fn, *SPECS, backend="iree", cache=cache)
    # The explicit spelling of the resolved default device (cpu:0) — the
    # bare "cpu:0" STRING is not a device kind in etl (Device("cpu:0")
    # would be the bogus kind "cpu:0", rejected at load); the Device
    # object is the explicit form, and the plain "cpu" kind string
    # normalizes to the same Device("cpu", 0).
    exe_explicit = build_cached(mix_fn, *SPECS, backend="iree",
                                device=etl.core.Device("cpu", 0), cache=cache)
    assert exe_explicit is exe_default  # same key -> hit, no second compile
    exe_str = build_cached(mix_fn, *SPECS, backend="iree", device="cpu",
                           cache=cache)
    assert exe_str is exe_default
    assert cache.misses == 1 and cache.hits == 2
    assert len(cache) == 1
    assert np.allclose(_np(etl.run(exe_default, XA, XB)), _mix_reference(),
                       rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. GPU-guarded same-device inner loop over a cached cuda executable
# ---------------------------------------------------------------------------
N, D = 128, 16  # small step (this is a compile-once smoke, not a perf test)
STEP_SPECS = (
    etl.TensorSpec((), etl.int64),        # splitmix64 key (rank-0 i64)
    etl.TensorSpec((N, D), etl.float32),  # state
)


@etl.defn
def step_fn(key, state):
    """One DE-style step: split the key, add mutation noise, decay state."""
    k_noise, k_next = etl.random.split(key)
    noise = etl.random.normal(k_noise, (N, D), mean=0.0, std=0.1)
    return etl.add(etl.multiply(state, 0.9), noise), k_next


def _new_key(seed=12345):
    return np.array(seed, dtype=np.int64)  # rank-0 i64


def _new_state(seed=0):
    return (np.random.default_rng(seed).standard_normal((N, D)) * 0.01
            ).astype(np.float32)


def _pick_cuda_device_index():
    """Most-free HEALTHY GPU via nvidia-smi; pytest.skip when unavailable.

    Index 7 is the known ECC-broken device on this host and is never
    chosen; a machine whose remaining GPUs are all unavailable skips.
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
            gpus.append((int(free_mib), int(idx)))
        except ValueError:
            continue
    healthy = [(free_mib, idx) for free_mib, idx in gpus if idx != 7]
    if not healthy:
        pytest.skip("nvidia-smi reported no healthy GPUs "
                    "(known-broken index 7 excluded)")
    healthy.sort(reverse=True)
    return healthy[0][1]


def test_cached_cuda_same_device_inner_loop():
    idx = _pick_cuda_device_index()
    import iree.runtime as rt
    try:
        rt.get_driver("cuda").create_device(device_id=idx + 1)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"IREE cuda HAL driver or GPU {idx} unavailable: {exc}")
    cuda_device = etl.core.Device("cuda", idx)

    # Compile ONCE via the memoized build (explicit placement: the run
    # boundary rejects host inputs, so the executable lives on the cuda
    # device and every input must already be placed there).
    cache = CompileCache()
    exe = build_cached(step_fn, *STEP_SPECS, backend="iree",
                       device=cuda_device, target_backends=["cuda"],
                       opt_level="O3", cache=cache)
    assert cache.misses == 1 and cache.hits == 0

    # Iterated numpy-backend reference (same initial key/state as below).
    key_ref, state_ref = _new_key(), _new_state()
    refs = []
    for _ in range(5):
        state_ref, key_ref = etl.evaluate(step_fn, key_ref, state_ref,
                                          backend="numpy")
        refs.append((_np(state_ref), _np(key_ref)))

    # Same-device inner loop over the CACHED executable: the initial host
    # values are placed on the device ONCE via the explicit .to() upload;
    # every later call feeds the device-resident outputs (new state + split
    # key) back — zero host round-trips across the whole loop.
    key = etl.core.Tensor(_new_key()).to(cuda_device)
    state = etl.core.Tensor(_new_state()).to(cuda_device)
    for step in range(5):
        state, key = etl.run(exe, key, state)
        assert isinstance(state, etl.Tensor) and state.device == cuda_device
        assert isinstance(key, etl.Tensor) and key.device == cuda_device
        assert state.shape == (N, D) and state.dtype == np.dtype("float32")
        state_h, key_h = _np(state), _np(key)
        want_state, want_key = refs[step]
        assert state_h.shape == want_state.shape
        # f32 state: the documented normal fast-path budget (~4.8e-7 at the
        # normal level; values stay O(0.1) under the 0.9 decay).
        assert np.allclose(state_h, want_state, rtol=1e-4, atol=1e-5), (
            f"step {step}: cuda state vs numpy reference diverged")
        assert np.array_equal(key_h, want_key)  # splitmix64 keys: bit-exact

    # Compile-once across the whole loop: a re-call is a pure hit — the
    # SAME executable object, zero extra trace/lower/compile/load work.
    again = build_cached(step_fn, *STEP_SPECS, backend="iree",
                         device=cuda_device, target_backends=["cuda"],
                         opt_level="O3", cache=cache)
    assert again is exe
    assert cache.misses == 1 and cache.hits == 1
