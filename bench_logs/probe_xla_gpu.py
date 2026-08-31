#!/usr/bin/env python3
"""XLA adapter E2E probe on the fused DE/PSO 4096x50 graphs (real PJRT GPU
plugin). Compiles the phase-A StableHLO dumps (with the rank-0-input
surgery, see _common.py) through the etl xla adapter, runs parity inputs,
and times: per-call etl-run-equivalent (host staging + execute + host
output copies) and execute-only (pre-staged buffers, no host copies).

Env (set in the shell before launching this script):
  CUDA_VISIBLE_DEVICES=<free gpu>   (adapter uses addressable_devices()[0])
  ETL_PJRT_PLUGIN=<plugin .so>
  PATH=<ptxas bin>:...              (nvidia-cuda-nvcc-cu12 12.3.52)
  LD_LIBRARY_PATH=<12 nvidia lib dirs of the venv's torch wheels>
  ulimit -n 65536

Usage: python bench_logs/probe_xla_gpu.py [--steps 300] [--warmup 50]
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
CLONE = "/mnt/hdd_pool/bchuang/tmp_pjrt_probe/etl_763a415"
sys.path.insert(0, CLONE)

import numpy as np  # noqa: E402
import etl  # noqa: E402
from etl import core  # noqa: E402
from etl.backends.program import LoweredProgram, Signature  # noqa: E402

from _common import (  # noqa: E402
    GRAPHS, PLUGIN, XLA_OUT, load_surg_mlir, make_inputs, stats_ms,
)

_DT = {"f32": core.float32, "i64": core.int64}


def build_specs(graph):
    ins = tuple(
        core.TensorSpec(shape, _DT[d]) for shape, d in GRAPHS[graph]["ins"]
    )
    outs = tuple(
        core.TensorSpec(shape, _DT[d]) for shape, d in GRAPHS[graph]["outs"]
    )
    return ins, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(XLA_OUT, exist_ok=True)
    print(f"etl from: {etl.__file__}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"LD_LIBRARY_PATH set: {len(os.environ.get('LD_LIBRARY_PATH', '')) > 0}", flush=True)

    results = {"env": {"cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                       "etl": etl.__file__, "plugin": PLUGIN},
               "graphs": {}}

    for graph in ("de", "pso"):
        print(f"\n===== {graph} =====", flush=True)
        orig, surg = load_surg_mlir(graph)
        ins_specs, outs_specs = build_specs(graph)
        lp = LoweredProgram(
            backend="xla",
            signature=Signature(input_specs=ins_specs, output_specs=outs_specs),
            payload={"format": "stablehlo", "format_version": 1, "mlir_text": surg},
        )
        t0 = time.perf_counter()
        art = etl.compile(lp, backend="xla", plugin_path=PLUGIN)
        compile_s = time.perf_counter() - t0
        print(f"compile OK in {compile_s:.2f}s", flush=True)
        exe = etl.load(art)
        be = exe.backend_executable
        platform = art.runtime_dependencies.get("plugin", "?")
        print("plugin:", platform, flush=True)

        # --- rank-0 staging diagnostic (evidence for the plugin quirk) ---
        try:
            b = be._client.buffer_from_host(np.array(42, np.int64))
            r0_shape = tuple(b.to_host().shape)
            b.close()
        except Exception as exc:  # noqa: BLE001
            r0_shape = f"error: {exc}"
        print("rank-0 i64 staged buffer shape (plugin view):", r0_shape, flush=True)

        # --- parity runs ---
        run_records = {}
        for label, zeros in (("real", False), ("zeros", True)):
            ins = [core.Tensor(a) for a in make_inputs(graph, zeros=zeros)]
            outs = be.run(ins)
            run_records[label] = {
                "out_shapes": [list(o.shape) for o in outs],
                "finite": all(np.isfinite(o.numpy()).all() for o in outs),
            }
            np.savez(
                os.path.join(XLA_OUT, f"{graph}_{label}_ins.npz"),
                **{f"a{i}": a for i, a in enumerate(make_inputs(graph, zeros=zeros))},
            )
            np.savez(
                os.path.join(XLA_OUT, f"{graph}_{label}_outs.npz"),
                **{f"o{i}": o.numpy() for i, o in enumerate(outs)},
            )
            print(f"parity run ({label}): {len(outs)} outs, "
                  f"finite={run_records[label]['finite']}", flush=True)

        # --- timing: full per-call (host staging + execute + host copies) ---
        ins_arrays = make_inputs(graph, zeros=False)
        ins_tensors = [core.Tensor(a) for a in ins_arrays]
        for _ in range(args.warmup):
            be.run(ins_tensors)
        batches = []
        for b_i in range(2):
            ms = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                be.run(ins_tensors)
                ms.append((time.perf_counter() - t0) * 1e3)
            batches.append(stats_ms(ms))
            print(f"run batch {b_i + 1} (n={args.steps}): {batches[-1]}", flush=True)

        # --- timing: execute-only (pre-staged buffers, no host copies) ---
        staged = [be._client.buffer_from_host(a) for a in ins_arrays]
        try:
            def exec_once():
                for ob in be.native_module.execute(staged):
                    ob.close()
            for _ in range(20):
                exec_once()
            ms = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                exec_once()
                ms.append((time.perf_counter() - t0) * 1e3)
            exec_stats = stats_ms(ms)
            print(f"execute-only (n={args.steps}): {exec_stats}", flush=True)
        finally:
            for b in staged:
                b.close()

        results["graphs"][graph] = {
            "compile_s": round(compile_s, 2),
            "platform": platform,
            "rank0_buffer_shape": r0_shape,
            "parity_runs": run_records,
            "run_batches_ms": batches,
            "execute_only_ms": exec_stats,
        }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_xla.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults_xla.json written", flush=True)


if __name__ == "__main__":
    main()
