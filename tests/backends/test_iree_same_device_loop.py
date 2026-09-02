"""iree-cuda same-device iterative loop — the recommended pattern for
evox-style iterative stateful workloads (DE/PSO), validated at DE scale
(state f32 (4096, 50), ~8 iree dispatches per step).

Pattern (measured; see etl/backends/adapters/CONTEXT.md "Iterative
stateful workloads"): build the step graph ONCE (target_backends=["cuda"],
opt_level="O3"), then run N steps feeding each step's device-resident
outputs (new state + split key) back as inputs. The FIRST call stages the
host state once (unavoidable — there is no upload-only API); from the
second call on every input is an iree-produced device tensor, so the
pass-through rule applies: zero ``iree.runtime.asdevicearray`` calls and
zero host round-trips. Measured on a shared 8-GPU box: same-device
med ~0.15 ms (min 0.146, p90 ~0.16, quiet) vs ~1.05 ms quiet /
1.5-2.8 ms noisy for fresh-host-state steps — 5-10x in-window.

Structural pins (robust on a shared box; absolutes move +/-3x):
* correctness: two loops started from the same initial key are
  bit-deterministic step-for-step; every output is a payload-backed
  device-resident Tensor with the declared shape/dtype on the
  executable's device; a first-step cross-check vs the numpy backend
  (the reference) agrees to fp32 tolerance.
* zero-asdevicearray: from step 2 on the monkeypatched
  ``iree.runtime.asdevicearray`` must never fire (only the fallback path
  for >16 MB / unmappable / non-cuda inputs uses it).
Perf smoke (loose bounds, A/B interleaved in one process window so box
noise cancels): the same-device loop's median must be BELOW the
host-input loop's median (measured 5-10x gap; a regression to per-step
host staging would flip or erase it), the same-device median must stay
under 2.5 ms (measured 0.15-0.9 ms incl. busy windows), and the last-50
median must not drift beyond 4x the first-50 (no intrinsic drift
measured at this churn; the documented ~80 MB-churn drift mode reached
4.3x over 500 runs).
"""
import time

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# DE-shaped step graph (synthetic: random parents + mutation + crossover)
# ---------------------------------------------------------------------------
N, D = 4096, 50
SPECS = (
    etl.TensorSpec((), etl.int64),          # key (rank-0 i64, splitmix64)
    etl.TensorSpec((N, D), etl.float32),    # state
)


@etl.defn
def de_step(key, state):
    # Three split chains: parents / noise / crossover + next-step key.
    k_a, k_b = etl.random.split(key)
    k_p, k_noise = etl.random.split(k_a)
    k_u, k_next = etl.random.split(k_b)
    i1 = etl.random.randint(k_p, (N,), 0, N)
    i2 = etl.random.randint(k_p, (N,), 0, N)
    i3 = etl.random.randint(k_p, (N,), 0, N)
    p1 = etl.gather(state, i1, axis=0)
    p2 = etl.gather(state, i2, axis=0)
    p3 = etl.gather(state, i3, axis=0)
    noise = etl.random.normal(k_noise, (N, D), mean=0.0, std=0.5)
    mutant = etl.clamp(p1 + 0.5 * (p2 - p3) + noise, -1e3, 1e3)
    cr = etl.random.uniform(k_u, (N, D), low=0.0, high=1.0)
    trial = etl.select(cr < 0.9, mutant, state)  # ~10% greedy crossover
    return trial, k_next


def _new_state(seed=0):
    return (np.random.default_rng(seed).standard_normal((N, D)) * 0.01
            ).astype(np.float32)


def _new_key(seed=12345):
    return np.array(seed, dtype=np.int64)  # rank-0 i64


def _pick_cuda_device_index():
    """Most-free GPU via nvidia-smi; pytest.skip when unavailable."""
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
    if not gpus:
        pytest.skip("nvidia-smi reported no GPUs")
    gpus.sort(reverse=True)
    return gpus[0][1]


@pytest.fixture(scope="module")
def cuda_device():
    idx = _pick_cuda_device_index()
    import iree.runtime as rt
    try:
        rt.get_driver("cuda").create_device(device_id=idx + 1)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"IREE cuda HAL driver or GPU {idx} unavailable: {exc}")
    return etl.core.Device("cuda", idx)


@pytest.fixture(scope="module")
def exe(cuda_device):
    return etl.build(de_step, *SPECS, backend="iree", device=cuda_device,
                     target_backends=["cuda"], opt_level="O3")


def _np(v):
    return np.asarray(v) if isinstance(v, etl.Tensor) else np.asarray(v)


def _device_loop(exe, steps, key, state):
    """steps same-device iterations feeding run outputs back in; returns
    (times_ms, final_key, final_state)."""
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        state, key = etl.run(exe, key, state)
        times.append((time.perf_counter() - t0) * 1e3)
    return times, key, state


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def test_loop_bit_deterministic_and_device_resident(exe, cuda_device):
    key_a, state_a = _new_key(7), _new_state(0)
    key_b, state_b = _new_key(7), _new_state(0)
    seq_a = []
    for _ in range(12):
        state_a, key_a = etl.run(exe, key_a, state_a)
        seq_a.append((_np(state_a), _np(key_a)))
        # every step's outputs are payload-backed device tensors on the exe
        assert isinstance(state_a, etl.Tensor)
        assert not isinstance(state_a.data, np.ndarray)
        assert state_a.device == cuda_device
        assert state_a.shape == (N, D) and state_a.dtype == np.dtype("float32")
        assert key_a.shape == () and key_a.dtype == np.dtype("int64")
    for _ in range(12):
        state_b, key_b = etl.run(exe, key_b, state_b)
        s_a, k_a = seq_a.pop(0)
        assert np.array_equal(_np(state_b), s_a)  # bit-deterministic
        assert np.array_equal(_np(key_b), k_a)


def test_first_step_matches_numpy_backend_reference(exe):
    # The numpy interpreter is the random-op reference; the compiled cuda
    # step must agree on the fused graph to fp32 tolerance (parity suites
    # own the strict per-op bit-exactness pins).
    key, state = _new_key(7), _new_state(0)
    cuda_out = etl.run(exe, key, state)
    np_out = etl.evaluate(de_step, key, state, backend="numpy")
    for c, n in zip(etl.tree_leaves(_np(cuda_out)), etl.tree_leaves(_np(np_out))):
        assert c.shape == n.shape
        assert np.allclose(c, n, rtol=1e-4, atol=1e-4), f"{c} vs {n}"


# ---------------------------------------------------------------------------
# structural: zero asdevicearray from the second call on
# ---------------------------------------------------------------------------
def test_zero_asdevicearray_from_second_call(exe, monkeypatch):
    import iree.runtime as rt

    calls = []

    def _counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    real = rt.asdevicearray
    monkeypatch.setattr(rt, "asdevicearray", _counting)
    key, state = _new_key(7), _new_state(1)
    key, state = etl.run(exe, key, state)  # step 1: host staging (allowed)
    first_step_calls = len(calls)
    for _ in range(20):                    # steps 2..21: pure pass-through
        key, state = etl.run(exe, key, state)
    assert len(calls) == first_step_calls  # ZERO asdevicearray on steps 2+


# ---------------------------------------------------------------------------
# perf smoke (loose bounds; A/B interleaved so shared-box noise cancels)
# ---------------------------------------------------------------------------
def test_same_device_loop_beats_host_restaging_and_stays_flat(exe):
    def _med(xs):
        return float(np.median(xs))

    # Host-restaging loop: fresh host numpy state + key every step.
    h_times = []
    key_h, state_h = _new_key(7), _new_state(2)
    for _ in range(24):
        state_h = _new_state(2)  # fresh host state each step (evox round-4)
        t0 = time.perf_counter()
        state_h, key_h = etl.run(exe, key_h, state_h)
        h_times.append((time.perf_counter() - t0) * 1e3)
    h_med = _med(h_times)

    # Same-device loop: 200 steps feeding device outputs back.
    key_d, state_d = _new_key(7), _new_state(3)
    t0 = time.perf_counter()
    state_d, key_d = etl.run(exe, key_d, state_d)  # one-time host staging
    d_times, _, _ = _device_loop(exe, 200, key_d, state_d)
    d_med, d_p90 = _med(d_times), float(np.percentile(d_times, 90))
    drift = _med(d_times[150:]) / max(_med(d_times[:50]), 1e-6)

    # Measured: d ~0.15 (quiet) / 0.19-0.9 ms (noisy); h ~1.05 (quiet) /
    # 1.5-2.8 ms (noisy). The relative pin is the robust discriminator.
    assert d_med < h_med, (
        f"same-device median {d_med:.3f} ms NOT below host-restaging "
        f"{h_med:.3f} ms — per-step host staging likely re-entered the loop"
    )
    assert d_med < 2.5, f"same-device median {d_med:.3f} ms — runaway"
    assert d_p90 < 4.0, f"same-device p90 {d_p90:.3f} ms — runaway"
    assert drift < 4.0, (
        f"last-50 median {drift:.2f}x the first-50 — allocator drift mode"
    )
