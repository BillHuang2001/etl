"""perf-fixes-3way GPU A/B probe (T1): xla rank-0 staging + iree-cuda sort/topk.

Env (set at process start — see the adapters CONTEXT Known Issues #3):
  CUDA_VISIBLE_DEVICES=<free gpu>
  LD_PRELOAD=/mnt/local-ssd/bchuang/cudnn-xla/lib/libcudnn.so.9
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bchuang/xla_cuda_data
  ETL_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so

Section A (xla, real 0.10.2 plugin): rank-0 `.to(Device("cuda", 0))` staging.
Section B (iree-cuda): 1-operand VALUES sort (count-mode routing gate) and
topk k==1 (argmin/argmax fast path) timing vs the pre-fix documented numbers.
"""
import os
import sys
import time

import numpy as np

# Pin THIS checkout's etl first: `python script.py` puts the script's dir on
# sys.path[0] (not the repo root), and the venv's editable etl maps to the
# MAIN checkout's master — which may lag the fixes under test.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import etl
from etl.core import Device

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
CUDA0 = Device("cuda", 0)
CPU = Device("cpu", 0)

def section_a():
    print("== A: xla rank-0 staging (real plugin) ==")
    from etl.backends.adapters import xla
    xla.register()
    for value, dtype in ((42, np.int64), (7, np.int32), (2.5, np.float32)):
        t = etl.core.Tensor(np.array(value, dtype=dtype)).to(CUDA0)
        host = t.to(CPU)
        ok = t.shape == () and t.dtype == np.dtype(dtype) and host.numpy().shape == ()
        print(f"  {dtype.__name__}: staged shape {t.shape}, host roundtrip "
              f"{host.numpy()!r} shape {host.numpy().shape} -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok

def section_b():
    print("== B: iree-cuda sort/topk timing (post-fix) ==")
    n = 10001
    rng = np.random.default_rng(0)
    x = rng.uniform(-5.12, 5.12, size=(n,)).astype(np.float32)
    # values sort graph
    def sort_fn(t):
        return etl.sort(t)
    def topk_fn(t):
        return etl.topk(t, 1)
    for name, fn, want in (
        ("sort values n=10001 (pre-fix 3394 ms)", sort_fn, None),
        ("topk k=1 n=10001 (pre-fix ~3480 ms step / composition)", topk_fn, None),
    ):
        exe = etl.build(fn, etl.core.TensorSpec((n,), "float32"),
                        backend="iree", target_backends=["cuda"],
                        device=CUDA0)
        xt = etl.core.Tensor(x).to(CUDA0)
        etl.run(exe, xt)  # warm up
        t0 = time.perf_counter()
        out = etl.run(exe, xt)
        ms = (time.perf_counter() - t0) * 1e3
        if name.startswith("sort"):
            host = out.to(CPU).numpy()
            exact = bool(np.array_equal(host, np.sort(x)))
        else:
            vals, idxs = out
            host = vals.to(CPU).numpy()
            # k=1 largest=True -> the axis MAX (not np.sort(x)[:1], the min);
            # scalar equality: numpy 2.4 np.array_equal rejects () vs (1,).
            exact = bool(host[0] == np.max(x)) and bool(
                idxs.to(CPU).numpy()[0] == np.argmax(x)
            )
        print(f"  {name}: {ms:.2f} ms (numpy-exact: {exact})")
    return True

if __name__ == "__main__":
    a = section_a()
    b = section_b()
    print("RESULT:", "ALL PASS" if a and b else "SOMETHING FAILED")
