"""MI300X rocm:0 host<->device staging microbenchmark (post ec58c62).

(a) etl.tensor(np.zeros(n, f32)).to(rocm:0)         for n=1000 (4 KB) / n=1_000_000 (4 MB)
(b) full harness per-call pattern on the `matmul` example:
    place both inputs on rocm:0 (explicit Tensor.to) -> etl.run -> .to(cpu) readback
(c) run-only, device-resident: same executable, inputs already a device Tensor,
    outputs left on the device (no staging inside the timed loop)

Best-of-N (N=25) wall times via perf_counter; the first warm-up calls are untimed.
Run from the repo root:  .venv/bin/python etl/bench/_matrix2/probe_rocm_staging2.py
"""
from __future__ import annotations

import time

import numpy as np

import etl
from etl.core import Device

from etl.bench._util import resolve_backend_options, stage_example
from etl.bench.examples import get_example

ROCM = Device("rocm", 0)
CPU = Device("cpu", 0)
N = 25
WARM = 3
results: list[tuple[str, float, float]] = []


def best_of(fn, n=N, warm=WARM):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def bench(label, fn):
    b, med = best_of(fn)
    results.append((label, b, med))
    print(f"  {label:<52s} best={b:8.4f} ms  median={med:8.4f} ms", flush=True)


# ---------------------------------------------------------------- (a) uploads
print("=== (a) host->device upload: etl.tensor(np.zeros(n, f32)).to(rocm:0) ===", flush=True)
for n in (1000, 1_000_000):
    arr = np.zeros(n, np.float32)
    t = etl.tensor(arr)
    bench(f"full expr etl.tensor+to  n={n} ({arr.nbytes/1024:.0f} KB)", lambda a=arr: etl.tensor(a).to(ROCM))
    bench(f".to(rocm) alone        n={n} ({arr.nbytes/1024:.0f} KB)", lambda tt=t: tt.to(ROCM))

up_small = etl.tensor(np.arange(1000, dtype=np.float32)).to(ROCM)
up_large = etl.tensor(np.arange(1_000_000, dtype=np.float32)).to(ROCM)
print("  4 KB  bit-exact round-trip:", bool(np.array_equal(up_small.to(CPU).numpy(),
                                                           np.arange(1000, dtype=np.float32))), flush=True)
print("  4 MB  bit-exact round-trip:", bool(np.array_equal(up_large.to(CPU).numpy(),
                                                           np.arange(1_000_000, dtype=np.float32))), flush=True)

# ------------------------------------------- (b)/(c) matmul example patterns
print("\n=== (b)/(c) `matmul` example on iree/rocm:0 ===", flush=True)
backend = "iree"
opts = resolve_backend_options(backend, ROCM, {})
print("  resolved options:", {k: repr(v) for k, v in opts.items()}, flush=True)

ex = get_example("matmul")
inputs = ex.generate_inputs(0)
run_fn = stage_example(ex, backend, ROCM, opts)  # harness callable: place + etl.run

ref = ex.numpy_ref(inputs)
out = run_fn(inputs)
got = out[0] if isinstance(out, (tuple, list)) else out
got = got.to(CPU).numpy()
max_abs = float(np.max(np.abs(got - ref)))
print(f"  correctness: max_abs={max_abs:.3g} vs numpy ref", flush=True)

# (b) full harness pattern: place (explicit Tensor.to) + run
host = [np.ascontiguousarray(v) for v in inputs]
bench("(b) harness pattern place+run (run_fn)", lambda: run_fn(host))


# (b') same + explicit .to(cpu) readback of the output
def place_run_readback():
    out = run_fn(host)
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o.to(CPU).numpy()


bench("(b') place+run+readback (.to(cpu))", place_run_readback)

# component decomposition
bench("   placement only (etl.tensor+to x2)", lambda: [etl.tensor(v).to(ROCM) for v in host])
dev_ins = [etl.tensor(v).to(ROCM) for v in host]
exe = etl.build(ex.graph, *ex.specs, backend=backend, device=ROCM, **opts)
out_dev = etl.run(exe, *dev_ins)
out_dev = out_dev[0] if isinstance(out_dev, (tuple, list)) else out_dev
bench("   readback only (D2H x1, .to(cpu))", lambda: out_dev.to(CPU))

# (c) run-only, fully device-resident
bench("(c) run only, device-resident in+out", lambda: etl.run(exe, *dev_ins))

print("\n=== compact table ===", flush=True)
print(f"  {'case':<46s} {'best ms':>10s} {'median ms':>10s}", flush=True)
for label, b, med in results:
    print(f"  {label:<46s} {b:10.4f} {med:10.4f}", flush=True)

print("\nPrior (pre-fix) reference numbers: 4 KB ~0.05 ms, 4 MB ~0.43 ms, (b) ~0.28 ms", flush=True)
print("PROBE_DONE", flush=True)
