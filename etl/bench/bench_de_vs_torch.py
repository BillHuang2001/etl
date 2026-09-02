"""etl.bench.bench_de_vs_torch — DE-step parity benchmark: etl vs torch.

Standalone measurement tool (NOT part of the ``etl.bench`` public API — run
as a script: ``python3 etl/bench/bench_de_vs_torch.py``) that demonstrates
etl-vs-torch parity on the DE/PSO-class iterative workload the evox-refactor
benchmark measured (etl 1.37-1.48x behind torch on GPU with the FINAL-5
per-step host-restaging pattern; CPU DE 2.55x behind). It answers: does the
recommended SAME-DEVICE loop (state stays on the device across steps) reach
or beat torch eager parity, how much of the FINAL-5 gap remains with per-step
host restaging, and what is the numpy-backend CPU story.

The step under test is a synthetic-but-faithful fused DE/rand/1/bin
optimization step mirroring the new-evox fused DE graph (ask + evaluate +
tell in ONE graph), shaped after ``tests/backends/test_iree_same_device_loop.py``
(its ``de_step`` defn + ``N, D = 4096, 50`` + ``SPECS``), extended with the
sphere-fitness reduction so it is a real optimization step:

- **ask** — parent picks: three ``(P,)`` randint streams + three row gathers
  of the population (evox ``DE_differential_sum`` with ``diff_padding_num=3``),
  mutant = ``p1 + 0.5 * (p2 - p3)`` (rand/1 with F=0.5); one ``(P, D)`` f32
  normal noise draw whose ``noise < 0.9`` mask is the crossover bit mask
  (evox ``DE_binary_crossover``'s ``randn < CR`` trick — true with
  probability Phi(0.9) ~= 0.816), one ``(P,)`` jind randint with the forced
  jind column via an ``arange(D) == jind[:, None]`` broadcast (evox's
  "at least one dimension from the mutant" rule); trial = clamp(select(mask
  | forced, mutant, state), lb, ub).
- **evaluate** — sphere fitness: ``fitness = sum(x * x, axis=1)`` for the
  trial AND the incumbent population (two ``(P,)`` reductions).
- **tell** — greedy 1:1 selection (``trial_fit < fit``) into the new
  population (evox greedy DE), with the running population best tracked as
  ``min(new_fit)`` (a rank-0 reduce_min output).

At ``(4096, 50)`` the exported step is ~8 iree dispatches (~380-400
stablehlo ops — the round-5 measured census of the new-evox fused DE step,
see ``etl/backends/adapters/CONTEXT.md`` "Iterative stateful workloads").
The step is pure: the key is split inside the graph (three split chains:
parents / noise / jind + the next-step key) so every step consumes a rank-0
int64 key and a ``(P, D)`` f32 state and produces ``(new_state, best, key)``.

Three timing sides with IDENTICAL step semantics per cell
----------------------------------------------------------
1. **etl iree-cuda SAME-DEVICE loop** — build once (``target_backends=
   ["cuda"]``, ``opt_level="O3"``, free GPU); the first step stages the host
   state once (unavoidable — there is no upload-only API); steps 2..N feed
   the previous step's device-resident outputs (state + split key) back.
   ``.numpy()`` is NEVER called in the timed loop (the invoke is fully
   synchronous with the device queue — see the adapter notes — so the
   per-call wall time is the honest per-step latency).
2. **etl iree-cuda HOST-RESTAGING loop** — the FINAL-5 evox pattern: fresh
   host numpy state (and key) are passed in every step and the outputs are
   read back (``state.numpy()``) every step — an 800 KB D2H readback + an
   800 KB H2D staging per step at ``(4096, 50)``.
3. **torch eager CUDA loop** — the same math in torch on cuda (randint
   parents, randn ``(P, D)``, advanced-indexing gathers, ``where``/clamp,
   sum-square fitness, greedy best update); state tensors stay on cuda
   (what old torch evox does); per-step ``torch.cuda.synchronize()`` inside
   the timed window. RNG differs (torch CUDA Philox vs etl splitmix64) —
   the step STRUCTURE is identical, the trajectories are not (apples-vs-
   oranges caveat applies to the RNG only, on GPU as well as CPU).

Cells: GPU ``(4096, 50)`` and ``(32, 10)`` (the evox DE benchmark cells);
CPU ``(4096, 50)`` and ``(1024, 50)``. Timing: 1 warm-up step, then 50
timed steps per pass, 3 passes per side, with the sides INTERLEAVED per
pass in rotating order (pass 1: 1,2,3; pass 2: 2,3,1; pass 3: 3,1,2) so
shared-box drift cancels. Reported per side: per-pass medians, the overall
median and min ms/step, steps/s (= 1000 / median ms), and the ms ratio vs
torch (1.0 = parity, <1 = etl faster, >1 = etl slower).

Sanity checks (loud failures, never silently skipped):
- etl same-key determinism: two 50-step trajectories from the same initial
  key/state are BIT-IDENTICAL in best_fit (and final state) — checked on
  every etl side (iree-cuda same-device, iree-cuda host-restaging, numpy
  CPU). The two GPU patterns must also agree with each other bitwise (host
  round-trips are exact f32 bit copies).
- best_fit decreases over the 50-step run on ALL sides (strictly monotone
  non-increasing by construction of the greedy selection; asserted exact
  plus a strict final < initial check).
- nvidia-smi snapshot printed at start and end (box-noise documentation).

The CPU section documents the numpy-BACKEND story (the reference
interpreter — the compiled-CPU 2.55x figure of the evox benchmark was iree
llvm-cpu): etl ``de_step`` on the numpy backend vs torch CPU eager at
``(4096, 50)`` and ``(1024, 50)``. The etl numpy kernel runs the round's f32
normal fast path, but the RNG comparison is apples-to-oranges: etl's
deterministic SplitMix64 word chain is ~8 ms of the step vs torch's
multithreaded MT19937 (different algorithm AND different threading model) —
documented in ``de_vs_torch_results.md``, never presented as a kernel-level
comparison.

Usage::

    python3 etl/bench/bench_de_vs_torch.py [--steps N] [--passes N]
        [--seed N] [--device auto|cuda:N] [--gpu-only] [--cpu-only]
        [--quick] [--out FILE]

Only stdlib + numpy + etl are imported at module scope; the iree adapter and
torch are imported LAZILY inside the staging helpers (torch-optionality
discipline: ``import etl.bench`` must never import torch — this module
follows the same rule even when run as a script). Exit codes: 0 = success,
1 = a sanity check failed, 2 = usage/environment error.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time

import numpy as np

import etl
from etl import core

__all__ = [
    "main", "run_benchmark", "de_step", "GPU_CELLS", "CPU_CELLS",
    "DEFAULT_STEPS", "DEFAULT_PASSES", "PASS_ORDER",
]

#: GPU cells: (population, dims) — the evox DE benchmark cells.
GPU_CELLS = ((4096, 50), (32, 10))
#: CPU cells (numpy-backend vs torch CPU eager).
CPU_CELLS = ((4096, 50), (1024, 50))
DEFAULT_STEPS = 50
DEFAULT_PASSES = 3
DEFAULT_WARMUP = 1
QUICK_STEPS = 20
QUICK_PASSES = 2
#: Rotating interleave order per pass (side indexes: 0 = same-device,
#: 1 = host-restaging, 2 = torch) so shared-box drift cancels.
PASS_ORDER = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
KEY_SEED = 12345
INIT_STD = 0.01  # initial-population scale: N(0, 0.01) per element
LB, UB = -100.0, 100.0  # clamp bounds (never bind at this scale — evox semantics)
CR_NOISE_THRESHOLD = 0.9  # randn < CR: true w.p. Phi(0.9) ~= 0.816


# ---------------------------------------------------------------------------
# The step graph under test (P, D come from the traced state spec — the SAME
# defn serves every cell; shapes are baked statically per trace).
# ---------------------------------------------------------------------------


@etl.defn
def de_step(key, state):
    """One fused DE/rand/1/bin ask+evaluate+tell step on the sphere.

    Args:
        key: rank-0 int64 splitmix64 key (split inside the graph).
        state: the ``(P, D)`` f32 population.

    Returns:
        ``(new_state, best, key)`` — the greedily-selected population, the
        running population best (rank-0 f32 ``reduce_min`` over the new
        fitnesses), and the next-step key.
    """
    p, d = state.shape
    # --- ask: parents (DE_differential_sum, diff_padding_num=3) ------------
    # NOTE: etl's stateless RNG contract makes two same-kind ops sharing one
    # key CORRELATED by construction (per-op-kind salts) — independent draws
    # require split keys. The three parent streams therefore come from three
    # split sub-keys (mirroring evox's single (P,3) randint draw, whose
    # columns are independent words).
    k_a, k_b = etl.random.split(key)
    k_p, k_noise = etl.random.split(k_a)
    k_u, k_next = etl.random.split(k_b)
    k_p1, k_p23 = etl.random.split(k_p)
    k_p2, k_p3 = etl.random.split(k_p23)
    i1 = etl.random.randint(k_p1, (p,), 0, p)
    i2 = etl.random.randint(k_p2, (p,), 0, p)
    i3 = etl.random.randint(k_p3, (p,), 0, p)
    p1 = etl.gather(state, i1, axis=0)
    p2 = etl.gather(state, i2, axis=0)
    p3 = etl.gather(state, i3, axis=0)
    mutant = p1 + 0.5 * (p2 - p3)  # rand/1, F = 0.5
    # --- ask: crossover (DE_binary_crossover) -------------------------------
    noise = etl.random.normal(k_noise, (p, d), mean=0.0, std=1.0)
    mask = noise < CR_NOISE_THRESHOLD  # the randn<CR bit-mask trick
    jind = etl.random.randint(k_u, (p,), 0, d)
    forced = etl.equal(
        etl.enp.arange(d, dtype=etl.int32), etl.reshape(jind, (p, 1))
    )
    trial = etl.clamp(
        etl.select(etl.logical_or(mask, forced), mutant, state), LB, UB
    )
    # --- evaluate: sphere fitness sum(x^2, axis=1) --------------------------
    fit = etl.sum(state * state, axes=(1,))
    tfit = etl.sum(trial * trial, axes=(1,))
    # --- tell: greedy 1:1 selection + running best --------------------------
    better = tfit < fit
    new_state = etl.select(etl.reshape(better, (p, 1)), trial, state)
    new_fit = etl.select(better, tfit, fit)
    best = etl.min(new_fit)
    return new_state, best, k_next


def _specs(p, d):
    return (
        etl.TensorSpec((), etl.int64),  # key (rank-0 i64, splitmix64)
        etl.TensorSpec((p, d), etl.float32),  # state
    )


def _new_state(p, d, seed=0):
    return (np.random.default_rng(seed).standard_normal((p, d)) * INIT_STD
            ).astype(np.float32)


def _new_key(seed=KEY_SEED):
    return np.array(seed, dtype=np.int64)  # rank-0 i64


# ---------------------------------------------------------------------------
# torch step (identical semantics; lazy torch import)
# ---------------------------------------------------------------------------


def _torch_de_step(state, gen, torch_mod):
    """One DE step in torch eager — mirrors ``de_step`` op-for-op.

    ``state`` is the f32 ``(P, D)`` population tensor (already on the target
    device); ``gen`` is a seeded ``torch.Generator`` on the same device;
    ``torch_mod`` is the lazily-imported torch module. Returns
    ``(new_state, best)`` — best is a 0-dim tensor.
    """
    torch = torch_mod
    p, d = state.shape[0], state.shape[1]
    i1 = torch.randint(0, p, (p,), device=state.device, generator=gen)
    i2 = torch.randint(0, p, (p,), device=state.device, generator=gen)
    i3 = torch.randint(0, p, (p,), device=state.device, generator=gen)
    p1 = state[i1]
    p2 = state[i2]
    p3 = state[i3]
    mutant = p1 + 0.5 * (p2 - p3)
    noise = torch.randn((p, d), device=state.device, generator=gen)
    mask = noise < CR_NOISE_THRESHOLD
    jind = torch.randint(0, d, (p,), device=state.device, generator=gen)
    forced = torch.arange(d, device=state.device) == jind.unsqueeze(1)
    trial = torch.clamp(
        torch.where(mask | forced, mutant, state), LB, UB
    )
    fit = (state * state).sum(dim=1)
    tfit = (trial * trial).sum(dim=1)
    better = tfit < fit
    new_state = torch.where(better.unsqueeze(1), trial, state)
    new_fit = torch.where(better, tfit, fit)
    return new_state, new_fit.min()


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------


def _med(xs):
    return float(np.median(xs))


def _fmt_ms(ms):
    return f"{ms:.3f}"


def _nvidia_smi_snapshot():
    """``index, memory.total, memory.free`` per GPU — for the box-noise note."""
    if shutil.which("nvidia-smi") is None:
        return "(nvidia-smi not found)"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"(nvidia-smi failed: {exc})"
    if proc.returncode != 0:
        return f"(nvidia-smi failed: {proc.stderr.strip()})"
    return " | ".join(line.strip() for line in proc.stdout.strip().splitlines())


def _pick_free_gpu(min_free_mib=2048):
    """Most-free GPU index via nvidia-smi (repo GPU policy). Prefers the big
    free GPUs (3/5, ~48 GB); refuses to pick a nearly-full one."""
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("bench_de_vs_torch: nvidia-smi not found — no CUDA "
                         "device to benchmark")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"bench_de_vs_torch: nvidia-smi failed: {exc}") from None
    if proc.returncode != 0:
        raise SystemExit(
            f"bench_de_vs_torch: nvidia-smi failed: {proc.stderr.strip()}"
        ) from None
    gpus = []
    for line in proc.stdout.strip().splitlines():
        try:
            idx, free_mib = (part.strip() for part in line.split(","))
            gpus.append((int(free_mib), int(idx)))
        except ValueError:
            continue
    if not gpus:
        raise SystemExit("bench_de_vs_torch: nvidia-smi reported no GPUs")
    gpus.sort(reverse=True)
    free_mib, idx = gpus[0]
    if free_mib < min_free_mib:
        raise SystemExit(
            f"bench_de_vs_torch: most-free GPU {idx} has only {free_mib} MiB "
            f"free (need >= {min_free_mib}) — box is busy, retry later"
        )
    return idx


# ---------------------------------------------------------------------------
# per-side runners (pattern-carrying step loops)
# ---------------------------------------------------------------------------


def _etl_device_trace(exe, steps, key0, state0):
    """steps same-device iterations feeding device outputs back in; returns
    (bests, final_state, final_key) with per-step best read back (untimed)."""
    key = np.array(key0)
    state = np.asarray(state0)
    bests = []
    for _ in range(steps):
        state, best, key = etl.run(exe, key, state)
        bests.append(float(np.asarray(best.numpy())))
    return np.array(bests), np.asarray(state.numpy()), np.asarray(key.numpy())


def _etl_host_trace(exe, steps, key0, state0):
    """steps HOST-RESTAGING iterations (fresh host numpy inputs, outputs read
    back via .numpy() every step); returns (bests, final_state, final_key)."""
    key = np.array(key0)
    state = np.array(state0, copy=True)
    bests = []
    for _ in range(steps):
        state_t, best, key_t = etl.run(exe, key, state)
        bests.append(float(np.asarray(best.numpy())))
        state = np.asarray(state_t.numpy())
        key = np.asarray(key_t.numpy())
    return np.array(bests), state, key


def _torch_trace(state0_np, gen_seed, steps):
    """steps torch eager iterations on the target device; returns bests."""
    torch = _require_torch()
    state = torch.from_numpy(state0_np).to(_TORCH_DEVICE)
    gen = torch.Generator(device=_TORCH_DEVICE)
    gen.manual_seed(gen_seed)
    bests = []
    for _ in range(steps):
        state, best = _torch_de_step(state, gen, torch)
        bests.append(float(best.item()))
    return np.array(bests)


def _etl_device_pass(exe, key0, state0, warmup, timed):
    """One timed pass of the same-device pattern; returns per-step ms."""
    key = np.array(key0)
    state = np.asarray(state0)
    for _ in range(warmup):  # untimed (step 1 stages the host state once)
        state, _best, key = etl.run(exe, key, state)
    times = []
    for _ in range(timed):  # steps 2..N: device pass-through, NO .numpy()
        t0 = time.perf_counter()
        state, _best, key = etl.run(exe, key, state)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _etl_host_pass(exe, key0, state0, warmup, timed):
    """One timed pass of the FINAL-5 host-restaging pattern; per-step ms."""
    key = np.array(key0)
    state = np.array(state0, copy=True)
    for _ in range(warmup):
        state_t, _best, key_t = etl.run(exe, key, state)
        state = np.asarray(state_t.numpy())
        key = np.asarray(key_t.numpy())
    times = []
    for _ in range(timed):  # every step: host in, .numpy() readback out
        t0 = time.perf_counter()
        state_t, _best, key_t = etl.run(exe, key, state)
        state = np.asarray(state_t.numpy())
        key = np.asarray(key_t.numpy())
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _torch_pass(state0_np, gen_seed, warmup, timed, sync_each_step):
    """One timed pass of the torch eager pattern; per-step ms. On cuda every
    step is followed by ``torch.cuda.synchronize()`` inside the timed window
    (the honest per-step latency — the etl iree invoke is likewise fully
    synchronous with the device queue)."""
    torch = _require_torch()
    state = torch.from_numpy(state0_np).to(_TORCH_DEVICE)
    gen = torch.Generator(device=_TORCH_DEVICE)
    gen.manual_seed(gen_seed)
    for _ in range(warmup):
        state, _best = _torch_de_step(state, gen, torch)
        if sync_each_step:
            torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        state, _best = _torch_de_step(state, gen, torch)
        if sync_each_step:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


# ---------------------------------------------------------------------------
# sanity
# ---------------------------------------------------------------------------


def _check_monotone_decreasing(bests, label):
    """Greedy selection makes the population best exactly non-increasing;
    the 50-step run must also strictly improve (checked by the caller)."""
    bad = np.where(bests[1:] > bests[:-1])[0]
    if bad.size:
        raise AssertionError(
            f"{label}: best_fit INCREASED at step {int(bad[0]) + 1}: "
            f"{bests[bad[0]]:.6e} -> {bests[bad[0] + 1]:.6e}"
        )
    if not bests[-1] < bests[0]:
        raise AssertionError(
            f"{label}: best_fit did not strictly decrease over the run: "
            f"{bests[0]:.6e} -> {bests[-1]:.6e}"
        )


@dataclasses.dataclass
class SideResult:
    """Aggregated timing + sanity result for one side of one cell."""

    name: str
    per_pass_medians: list
    all_times: list
    best_first: float = float("nan")
    best_last: float = float("nan")

    @property
    def median_ms(self):
        return _med(self.all_times)

    @property
    def min_ms(self):
        return float(min(self.all_times))

    @property
    def steps_per_s(self):
        return 1000.0 / self.median_ms


def _trace_etl(exe, pattern, steps, key0, state0):
    """One untimed 50-step trace of an etl pattern; returns
    ``(bests, final_state)`` with per-step best read back."""
    trace = _etl_device_trace if pattern == "device" else _etl_host_trace
    bests, state, _key = trace(exe, steps, key0, state0)
    return bests, state


def _sanity_etl(exe, pattern, p, d, key0, state0, steps, label):
    """etl same-key determinism + monotone-decrease sanity for one pattern:
    two trajectories from the same initial key/state must be BIT-IDENTICAL in
    best_fit and final state; best_fit must be monotone non-increasing and
    strictly improve over the run. Returns the first trajectory's bests."""
    bests1, state1 = _trace_etl(exe, pattern, steps, key0, state0)
    bests2, state2 = _trace_etl(exe, pattern, steps, key0, state0)
    _check_monotone_decreasing(bests1, label)
    if not np.array_equal(bests1, bests2):
        raise AssertionError(
            f"{label}: SAME-KEY DETERMINISM FAILED — two trajectories from "
            "the same initial key/state differ in best_fit"
        )
    if not np.array_equal(state1, state2):
        raise AssertionError(
            f"{label}: SAME-KEY DETERMINISM FAILED — final states differ"
        )
    return bests1


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------


def _gpu_cell(p, d, device, args):
    """One GPU cell: build once, sanity + interleaved timed passes per side."""
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise SystemExit("bench_de_vs_torch: torch has no CUDA — cannot run "
                         "the GPU section (torch.cuda.is_available() False)")
    global _TORCH_DEVICE
    _TORCH_DEVICE = f"cuda:{device.index}"
    specs = _specs(p, d)
    key0 = _new_key()
    state0 = _new_state(p, d, args.seed)
    print(f"[gpu {p}x{d}] building iree-cuda executable "
          f"(target_backends=[\"cuda\"], opt_level=\"O3\") ...",
          file=sys.stderr)
    t0 = time.perf_counter()
    exe = etl.build(de_step, *specs, backend="iree", device=device,
                    target_backends=["cuda"], opt_level="O3")
    print(f"[gpu {p}x{d}] build+load {time.perf_counter() - t0:.1f} s",
          file=sys.stderr)

    # sanity + determinism (untimed; also warms the device pools)
    dev_bests = _sanity_etl(exe, "device", p, d, key0, state0, args.steps,
                            f"same-device ({p}x{d})")
    host_bests = _sanity_etl(exe, "host", p, d, key0, state0, args.steps,
                             f"host-restaging ({p}x{d})")
    # the two GPU patterns run the SAME compiled exe from the SAME initial
    # state — host round-trips are exact f32 bit copies, so the trajectories
    # must agree bitwise
    if not np.array_equal(dev_bests, host_bests):
        raise AssertionError(
            f"gpu {p}x{d}: same-device and host-restaging trajectories DIFFER "
            "— host round-trip is not bit-preserving"
        )
    # torch sanity (monotone decrease; no bit-determinism pin — different RNG)
    torch_gen_seed = KEY_SEED + args.seed
    torch_bests = _torch_trace(state0, torch_gen_seed, args.steps)
    _check_monotone_decreasing(
        torch_bests, f"torch-cuda ({p}x{d})"
    )

    # timed passes, sides interleaved per pass in rotating order
    side_times = [[], [], []]  # 0 same-device, 1 host-restaging, 2 torch
    for pass_i in range(args.passes):
        for side_i in PASS_ORDER[pass_i]:
            if side_i == 0:
                t = _etl_device_pass(exe, key0, state0, args.warmup, args.steps)
            elif side_i == 1:
                t = _etl_host_pass(exe, key0, state0, args.warmup, args.steps)
            else:
                t = _torch_pass(state0, torch_gen_seed, args.warmup,
                                args.steps, sync_each_step=True)
            side_times[side_i].append(t)
    results = {
        "same-device": SideResult(
            name="etl iree-cuda same-device",
            per_pass_medians=[_med(t) for t in side_times[0]],
            all_times=[ms for t in side_times[0] for ms in t],
            best_first=float(dev_bests[0]), best_last=float(dev_bests[-1]),
        ),
        "host-restaging": SideResult(
            name="etl iree-cuda host-restaging",
            per_pass_medians=[_med(t) for t in side_times[1]],
            all_times=[ms for t in side_times[1] for ms in t],
            best_first=float(host_bests[0]), best_last=float(host_bests[-1]),
        ),
        "torch": SideResult(
            name="torch eager cuda",
            per_pass_medians=[_med(t) for t in side_times[2]],
            all_times=[ms for t in side_times[2] for ms in t],
            best_first=float(torch_bests[0]),
            best_last=float(torch_bests[-1]),
        ),
    }
    return results


#: CPU-side interleave order per pass (0 = etl-numpy, 1 = torch) — balanced
#: over the 3 passes.
CPU_ORDER = ((0, 1), (1, 0), (0, 1))


def _cpu_cell(p, d, args):
    """One CPU cell: etl numpy-backend vs torch CPU eager."""
    _require_torch()  # loud early error when torch is missing
    global _TORCH_DEVICE
    _TORCH_DEVICE = "cpu"
    specs = _specs(p, d)
    key0 = _new_key()
    state0 = _new_state(p, d, args.seed)
    print(f"[cpu {p}x{d}] building numpy-backend executable ...",
          file=sys.stderr)
    exe = etl.build(de_step, *specs, backend="numpy")
    # etl numpy sanity + determinism
    bests = _sanity_etl(exe, "host", p, d, key0, state0, args.steps,
                        f"etl-numpy ({p}x{d})")
    # torch CPU sanity
    torch_gen_seed = KEY_SEED + args.seed
    torch_bests = _torch_trace(state0, torch_gen_seed, args.steps)
    _check_monotone_decreasing(torch_bests, f"torch-cpu ({p}x{d})")

    # timed passes, sides interleaved (balanced over the passes)
    etl_times, torch_times = [], []
    for pass_i in range(args.passes):
        for side_i in CPU_ORDER[pass_i]:
            if side_i == 0:
                t = _etl_host_pass(exe, key0, state0, args.warmup, args.steps)
                etl_times.append(t)
            else:
                t = _torch_pass(state0, torch_gen_seed, args.warmup,
                                args.steps, sync_each_step=False)
                torch_times.append(t)
    return {
        "etl-numpy": SideResult(
            name="etl numpy-backend",
            per_pass_medians=[_med(t) for t in etl_times],
            all_times=[ms for t in etl_times for ms in t],
            best_first=float(bests[0]), best_last=float(bests[-1]),
        ),
        "torch": SideResult(
            name="torch eager cpu",
            per_pass_medians=[_med(t) for t in torch_times],
            all_times=[ms for t in torch_times for ms in t],
            best_first=float(torch_bests[0]),
            best_last=float(torch_bests[-1]),
        ),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _ratio_vs(ref_ms, ms):
    """etl/torch ms ratio (1.0 = parity, <1 = etl faster, >1 = etl slower)."""
    return ms / ref_ms if ref_ms > 0 else float("nan")


def _print_cell_summary(section, cell_label, results, order, out=sys.stderr):
    """Human-readable per-cell table (the md report is written by hand from
    the JSON dump)."""
    print(f"== {section} {cell_label} ==", file=out)
    torch_ms = results["torch"].median_ms
    for name in order:
        res = results[name]
        per_pass = ", ".join(_fmt_ms(v) for v in res.per_pass_medians)
        if name == "torch":
            ratio = "ref"
        else:
            ratio = f"{_ratio_vs(torch_ms, res.median_ms):.2f}x of torch"
        print(
            f"  {res.name:<28} med {res.median_ms:8.3f} ms  "
            f"min {res.min_ms:8.3f} ms  {res.steps_per_s:9.0f} steps/s  "
            f"({ratio})  best {res.best_first:.4e} -> {res.best_last:.4e}  "
            f"per-pass med: [{per_pass}]",
            file=out,
        )


def _print_summary(result, out=sys.stderr):
    if "gpu" in result:
        for cell, results in result["gpu"].items():
            _print_cell_summary("GPU", cell, results,
                                ("same-device", "host-restaging", "torch"))
    if "cpu" in result:
        for cell, results in result["cpu"].items():
            _print_cell_summary("CPU", cell, results, ("etl-numpy", "torch"))


# ---------------------------------------------------------------------------
# torch probing (lazy)
# ---------------------------------------------------------------------------

_TORCH_DEVICE = "cpu"


def _require_torch():
    try:
        import torch  # noqa: PLC0415 — lazy by design (torch-optionality)
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise SystemExit(
            "bench_de_vs_torch: torch is required for the torch comparison "
            "sides (pip install etl[bench]); the numpy-CPU section still "
            "runs with --cpu-only"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="bench_de_vs_torch",
        description="DE-step parity benchmark: etl iree-cuda (same-device "
                    "and host-restaging) vs torch eager CUDA + CPU section.",
    )
    parser.add_argument("--steps", type=int, default=None,
                        help=f"timed steps per pass (default: {DEFAULT_STEPS}, "
                        f"--quick: {QUICK_STEPS})")
    parser.add_argument("--passes", type=int, default=None,
                        help=f"timed passes per side (default: {DEFAULT_PASSES}, "
                        f"--quick: {QUICK_PASSES})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"untimed warm-up steps per pass (default: "
                        f"{DEFAULT_WARMUP})")
    parser.add_argument("--seed", type=int, default=0,
                        help="numpy seed for the initial population (default: 0)")
    parser.add_argument("--device", default="auto",
                        help="cuda device: 'auto' (most-free GPU via "
                        "nvidia-smi) or 'cuda:N' (default: auto)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="run only the GPU section")
    parser.add_argument("--cpu-only", action="store_true",
                        help="run only the CPU section")
    parser.add_argument("--quick", action="store_true",
                        help="smoke mode: 20 steps, 2 passes per side")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="write the JSON result dump to FILE")
    args = parser.parse_args(argv)
    if args.gpu_only and args.cpu_only:
        raise SystemExit("bench_de_vs_torch: --gpu-only and --cpu-only are "
                         "mutually exclusive")
    if args.steps is None:
        args.steps = QUICK_STEPS if args.quick else DEFAULT_STEPS
    if args.steps < 1:
        raise SystemExit(f"bench_de_vs_torch: --steps must be >= 1, got "
                         f"{args.steps}")
    if args.passes is None:
        args.passes = QUICK_PASSES if args.quick else DEFAULT_PASSES
    if args.passes < 1 or args.passes > len(PASS_ORDER):
        raise SystemExit(
            f"bench_de_vs_torch: --passes must be 1..{len(PASS_ORDER)}, got "
            f"{args.passes}"
        )
    if args.warmup < 0:
        raise SystemExit(f"bench_de_vs_torch: --warmup must be >= 0, got "
                         f"{args.warmup}")
    return args


def _env_banner(args):
    lines = ["bench_de_vs_torch — DE-step parity benchmark (etl vs torch)"]
    lines.append(f"  python: {sys.version.split()[0]}")
    lines.append(f"  numpy: {np.__version__}")
    lines.append(f"  etl: {etl.__file__}")
    try:
        import torch  # noqa: PLC0415
        lines.append(f"  torch: {torch.__version__} "
                     f"(cuda {torch.version.cuda}, "
                     f"available={torch.cuda.is_available()})")
    except ImportError:
        lines.append("  torch: NOT INSTALLED")
    try:
        import iree.compiler  # noqa: PLC0415
        import iree.runtime  # noqa: PLC0415
        lines.append(f"  iree: compiler {getattr(iree.compiler, '__version__', '?')} "
                     f"/ runtime {getattr(iree.runtime, '__version__', '?')}")
    except ImportError:
        lines.append("  iree: NOT INSTALLED")
    lines.append(f"  steps/pass: {args.steps}  passes: {args.passes}  "
                 f"warmup: {args.warmup}  seed: {args.seed}")
    return "\n".join(lines)


def run_benchmark(args):
    """Run the benchmark; returns the result dict (JSON-serializable)."""
    result = {"env": _env_banner(args)}
    start_smi = _nvidia_smi_snapshot()
    result["nvidia_smi_start"] = start_smi
    print(_env_banner(args), file=sys.stderr)
    print(f"nvidia-smi at start: {start_smi}", file=sys.stderr)

    gpu = not args.cpu_only
    cpu = not args.gpu_only
    if gpu:
        result["gpu"] = {}
        if args.device == "auto":
            idx = _pick_free_gpu()
        elif args.device.startswith("cuda:"):
            idx = int(args.device.split(":")[1])
        else:
            raise SystemExit(
                f"bench_de_vs_torch: --device must be 'auto' or 'cuda:N', "
                f"got {args.device!r}"
            )
        try:
            etl.backends.get("iree")
        except core.BackendError as exc:
            raise SystemExit(
                f"bench_de_vs_torch: {exc} (needed for the GPU section)"
            ) from None
        device = core.Device("cuda", idx)
        print(f"GPU section on {device} ...", file=sys.stderr)
        for p, d in GPU_CELLS:
            result["gpu"][f"({p},{d})"] = _gpu_cell(p, d, device, args)
    if cpu:
        result["cpu"] = {}
        print("CPU section (numpy backend vs torch cpu) ...", file=sys.stderr)
        for p, d in CPU_CELLS:
            result["cpu"][f"({p},{d})"] = _cpu_cell(p, d, args)
    end_smi = _nvidia_smi_snapshot()
    result["nvidia_smi_end"] = end_smi
    print(f"nvidia-smi at end: {end_smi}", file=sys.stderr)
    _print_summary(result)
    return result


def _jsonify(result):
    """Convert the result dict (SideResult values) to plain JSON data."""

    def _side(res):
        return {
            "name": res.name,
            "per_pass_median_ms": res.per_pass_medians,
            "median_ms": res.median_ms,
            "min_ms": res.min_ms,
            "steps_per_s": res.steps_per_s,
            "best_first": res.best_first,
            "best_last": res.best_last,
        }

    out = {"env": result["env"]}
    for key in ("nvidia_smi_start", "nvidia_smi_end"):
        if key in result:
            out[key] = result[key]
    for section in ("gpu", "cpu"):
        if section in result:
            out[section] = {
                cell: {name: _side(res) for name, res in sides.items()}
                for cell, sides in result[section].items()
            }
    return out


def main(argv=None) -> int:
    """CLI entry point. Returns 0 on success, 1 when a sanity check failed,
    2 on usage/environment errors."""
    try:
        args = _parse_args(argv)
        result = run_benchmark(args)
    except AssertionError as exc:
        print(f"bench_de_vs_torch: SANITY FAILURE: {exc}", file=sys.stderr)
        return 1
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — loud, never silent
        print(f"bench_de_vs_torch: FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    data = _jsonify(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print(f"bench_de_vs_torch: JSON dump written to {args.out}",
              file=sys.stderr)
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
