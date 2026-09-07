"""iree SO-path CUDA crash-repro + device-resident timing probe (Defect A).

Repro class (from the T1 sessions): iree-cuda 3.9.0 SIGSEGV'd on fresh
processes running 2+ CONSECUTIVE device-resident etl.run sequences at big
scale (state retention across runs/steps; error-path use-after-free/race in
the 3.9.0 runtime — upstream, not an etl graph defect). This probe runs the
SAME class of work on the INSTALLED iree version:

  - PSO-shaped step (split + uniform + elementwise move + sum + argmin +
    gather) at (10000, 100): build cuda, explicit placement, same-device
    loop of 20+ steps (device outputs fed back as next inputs — the state
    retention pattern).
  - DE-shaped step (split_n + per-row 3-parent randint/gather + CR mask +
    select + eval + argmin) at (10000, 100): SECOND build + placement +
    loop in the SAME process.
  - Both again at (1000, 50), plus one fresh placement of the big PSO graph
    (a "3rd sequence") to stress teardown/init churn.

Healthy expectation: no segfault in any sequence, process exits cleanly,
per-step times in the DEVICE-RESIDENT range (~0.5-3 ms/step at 10000x100;
seconds/step means a host-staging regression). Run the whole probe several
times in FRESH processes (the 3.9.0 crash was ~9/10 / deterministic-step-1).

Usage:
  CUDA_VISIBLE_DEVICES=<free gpu> <venv>/bin/python bench_logs/probe_iree_so_cuda.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import etl  # noqa: E402
from etl import core  # noqa: E402

CUDA0 = core.Device("cuda", 0)
CPU0 = core.Device("cpu", 0)
RESULTS = {"pass": [], "fail": []}
STEPS = 20  # timed steps per sequence (after 2 warmups)


def report(name, ok, detail=""):
    RESULTS["pass" if ok else "fail"].append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")


def make_pso_step(N, D):
    @etl.defn
    def pso_step(key, x, v, pbest, gbest):
        k1, k2 = etl.random.split(key)
        r1 = etl.random.uniform(k1, shape=(N, D))
        r2 = etl.random.uniform(k2, shape=(N, D))
        v_new = 0.7 * v + 1.5 * r1 * (pbest - x) + 1.5 * r2 * (gbest - x)
        x_new = x + v_new
        fitness = etl.sum(x_new * x_new, axes=1)
        fit_p = etl.sum(pbest * pbest, axes=1)  # re-eval of stored pbest
        better = fitness < fit_p
        pbest_new = etl.select(better[:, None], x_new, pbest)
        best_idx = etl.argmin(fitness, axis=None)
        gbest_new = etl.gather(x_new, best_idx, axis=0)
        # Every output keeps its input's shape so the same-device loop can
        # feed run outputs back as the next step's inputs.
        return k2, x_new, v_new, pbest_new, gbest_new

    return pso_step


def make_de_step(N, D):
    @etl.defn
    def de_step(key, pop, fit, gbest):
        # DE/rand/1/bin one generation, per-row 3-parent gather + eval.
        # Carry state (key, pop, fit, gbest) maps 1:1 onto the outputs so the
        # same-device loop can feed run outputs back as next inputs.
        k1, k2, k3, k4, k5 = etl.random.split_n(key, 5)
        i0 = etl.random.randint(k1, shape=(N, 1), low=0, high=N)
        i1 = etl.random.randint(k2, shape=(N, 1), low=0, high=N)
        i2 = etl.random.randint(k3, shape=(N, 1), low=0, high=N)
        a = etl.reshape(etl.gather(pop, i0, axis=0), (N, D))
        b = etl.reshape(etl.gather(pop, i1, axis=0), (N, D))
        c = etl.reshape(etl.gather(pop, i2, axis=0), (N, D))
        mutant = a + 0.5 * (b - c)
        cr = etl.random.uniform(k4, shape=(N, D)) < 0.9
        trial = etl.select(cr, mutant, pop)
        fit_new = etl.sum(trial * trial, axes=1)
        better = fit_new < fit
        pop_new = etl.select(better[:, None], trial, pop)
        best_idx = etl.argmin(fit_new, axis=None)
        gbest_new = etl.gather(trial, best_idx, axis=0)
        return k5, pop_new, fit_new, gbest_new

    return de_step


def place(arrays):
    return [core.Tensor(a).to(CUDA0) for a in arrays]


def run_sequence(name, fn, host_state, steps=STEPS):
    """Build (once), place, warmup, then a same-device loop of `steps` runs."""
    t0 = time.time()
    specs = tuple(etl.TensorSpec(np.asarray(a).shape, str(np.asarray(a).dtype))
                  for a in host_state)
    exe = etl.build(fn, *specs, backend="iree", target_backends=["cuda"],
                    device=CUDA0)
    t_build = time.time() - t0
    state = place(host_state)
    ok_resident = all(s.device.kind == "cuda" for s in state)
    report(f"{name}.placed_on_cuda", ok_resident)
    for _ in range(2):  # warmup (untimed)
        state = list(etl.run(exe, *state))
    t0 = time.time()
    for _ in range(steps):
        state = list(etl.run(exe, *state))
    ms = (time.time() - t0) / steps * 1000.0
    ok_out = all(s.device.kind == "cuda" for s in state)
    report(f"{name}.outputs_device_resident", ok_out,
           f"{ms:.3f} ms/step over {steps} runs")
    # Light sanity: finite host round-trip of the largest output (explicit).
    probe = max(state, key=lambda s: s.data.shape[0] * s.data.shape[1] if
                len(s.data.shape) >= 2 else 0)
    arr = probe.to(CPU0).numpy()
    finite = bool(np.isfinite(arr).all())
    report(f"{name}.host_roundtrip_finite", finite,
           f"shape {arr.shape} dtype {arr.dtype} "
           f"min {arr.min():.4g} max {arr.max():.4g}")
    print(f"  [{name}] build {t_build:.1f}s, {ms:.3f} ms/step")
    return ms


def main():
    import etl.backends  # noqa: F401  (exercise the import path pre-activation)
    from importlib.metadata import version as _pkg_version
    rng = np.random.default_rng(0)
    print(f"iree {_pkg_version('iree-base-compiler')} / "
          f"{_pkg_version('iree-base-runtime')} | etl {etl.__file__} | "
          f"GPU visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # Sequence 1: PSO 10000x100 (build 1).
    N, D = 10000, 100
    pso = make_pso_step(N, D)
    key = np.array(42, dtype=np.int64)  # splitmix64 rank-0 i64 key
    x = rng.uniform(-5.12, 5.12, (N, D)).astype(np.float32)
    v = rng.standard_normal((N, D)).astype(np.float32)
    pbest = x.copy()
    gbest = rng.standard_normal((D,)).astype(np.float32)
    m1 = run_sequence("s1.pso_10000x100", pso, [key, x, v, pbest, gbest])
    # Sequence 2: DE 10000x100 (build 2, same process).
    de = make_de_step(N, D)
    pop = rng.uniform(-5.12, 5.12, (N, D)).astype(np.float32)
    fit0 = np.sum(pop * pop, axis=1).astype(np.float32)
    gbest0 = pop[int(np.argmin(fit0))].copy()
    m2 = run_sequence("s2.de_10000x100", de,
                      [np.array(7, dtype=np.int64), pop, fit0, gbest0])
    # Sequence 3: fresh placement + loop on the SAME pso executable (state
    # retention churn: new upload after teardown-free runs).
    m3 = run_sequence("s3.pso_10000x100_replacement", pso,
                      [np.array(1, dtype=np.int64), x, v, pbest, gbest])
    # Sequences 4+5: smaller scale, second graphs.
    N2, D2 = 1000, 50
    pso2 = make_pso_step(N2, D2)
    de2 = make_de_step(N2, D2)
    x2 = rng.uniform(-5.12, 5.12, (N2, D2)).astype(np.float32)
    v2 = rng.standard_normal((N2, D2)).astype(np.float32)
    pop2 = rng.uniform(-5.12, 5.12, (N2, D2)).astype(np.float32)
    m4 = run_sequence("s4.pso_1000x50", pso2,
                      [np.array(3, dtype=np.int64), x2, v2, x2.copy(),
                       rng.standard_normal((D2,)).astype(np.float32)])
    pop2 = rng.uniform(-5.12, 5.12, (N2, D2)).astype(np.float32)
    fit2 = np.sum(pop2 * pop2, axis=1).astype(np.float32)
    gbest2 = pop2[int(np.argmin(fit2))].copy()
    m5 = run_sequence("s5.de_1000x50", de2,
                      [np.array(9, dtype=np.int64), pop2, fit2, gbest2])
    # Device-resident range check at the big scale (0.5-3 ms/step healthy;
    # seconds/step = host staging regression).
    ok = (0.3 <= m1 <= 6.0) and (0.3 <= m2 <= 6.0) and (0.3 <= m3 <= 6.0)
    report("big_scale_device_resident_ms_per_step", ok,
           f"s1={m1:.2f} s2={m2:.2f} s3={m3:.2f} s4={m4:.2f} s5={m5:.2f}")
    print("RESULT:", "ALL PASS" if not RESULTS["fail"] else
          f"{len(RESULTS['fail'])} FAILURES")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
