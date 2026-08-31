#!/usr/bin/env python3
"""IREE side of the same-mlir comparison: recompiles the SURG'D mlir text
(the exact text the xla adapter compiles) with the venv's iree-compile
(cuda target) and times vm.main with host numpy inputs (auto H2D staging
per call + D2H output copies) — the closest iree analog of the xla
adapter's per-call run. Also runs the ORIGINAL vmfb (rank-0 key) as a
cross-check (iree handles rank-0 inputs natively).

Env: no CUDA_VISIBLE_DEVICES (iree cuda device ids are 1-based;
device_id = gpu+1). ulimit -n 65536.
Usage: python bench_logs/probe_iree_same_mlir.py [--gpu 3] [--steps 300]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from _common import GRAPHS, IREE_OUT, HERE, load_surg_mlir, make_inputs, stats_ms

VENV_BIN = "/mnt/hdd_pool/bchuang/venvs/etl-bench/bin"
COMPILE = os.path.join(VENV_BIN, "iree-compile")
MLIR_SRC = (
    "/mnt/hdd_pool/bchuang/evox-refactor/.genesis/workers/worker_T1_A1/"
    "benchmarks/results"
)


def compile_vmfb(mlir_path: str, out_vmfb: str) -> float:
    if os.path.exists(out_vmfb):
        return 0.0
    t0 = time.time()
    proc = subprocess.run(
        [COMPILE, mlir_path,
         "--iree-input-demote-f64-to-f32=false",
         "--iree-hal-target-backends=cuda", "-o", out_vmfb],
        capture_output=True, text=True, timeout=1200,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"iree-compile failed rc={proc.returncode}: {proc.stderr[-3000:]}"
        )
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(IREE_OUT, exist_ok=True)

    import iree.runtime as rt
    dev = rt.get_driver("cuda").create_device(device_id=args.gpu + 1)
    print(f"iree device: cuda device_id={args.gpu + 1} (gpu {args.gpu})", flush=True)

    results = {"gpu": args.gpu, "graphs": {}}

    for graph in ("de", "pso"):
        print(f"\n===== {graph} =====", flush=True)
        orig, surg = load_surg_mlir(graph)
        mlir_path = os.path.join(HERE, GRAPHS[graph]["mlir"].replace(".mlir", "_key1.mlir"))
        vmfb_path = os.path.join(HERE, GRAPHS[graph]["mlir"].replace(".mlir", "_key1.vmfb"))
        ctime = compile_vmfb(mlir_path, vmfb_path)
        print(f"iree-compile (surg mlir): {ctime:.1f}s" if ctime else
              "iree-compile: cached", flush=True)

        mod = rt.VmModule.from_flatbuffer(rt.VmInstance(), open(vmfb_path, "rb").read())
        vm = rt.load_vm_module(mod, config=rt.Config(device=dev))
        f = vm.main

        def norm(res):
            res = res if isinstance(res, tuple) else (res,)
            return [np.asarray(o) for o in res]

        # parity runs
        run_records = {}
        for label, zeros in (("real", False), ("zeros", True)):
            ins = make_inputs(graph, zeros=zeros)
            outs = norm(f(*ins))
            run_records[label] = {
                "out_shapes": [list(o.shape) for o in outs],
                "finite": all(np.isfinite(o).all() for o in outs),
            }
            np.savez(os.path.join(IREE_OUT, f"{graph}_{label}_ins.npz"),
                     **{f"a{i}": a for i, a in enumerate(ins)})
            np.savez(os.path.join(IREE_OUT, f"{graph}_{label}_outs.npz"),
                     **{f"o{i}": o for i, o in enumerate(outs)})
            print(f"parity run ({label}): {len(outs)} outs, "
                  f"finite={run_records[label]['finite']}", flush=True)

        # timing: full per-call (H2D staging + execute + D2H copies)
        ins = make_inputs(graph, zeros=False)
        for _ in range(args.warmup):
            norm(f(*ins))
        batches = []
        for b_i in range(2):
            ms = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                norm(f(*ins))
                ms.append((time.perf_counter() - t0) * 1e3)
            batches.append(stats_ms(ms))
            print(f"run batch {b_i + 1} (n={args.steps}): {batches[-1]}", flush=True)

        # original vmfb cross-check (rank-0 key; value-identical graph)
        orig_vmfb = os.path.join(MLIR_SRC, GRAPHS[graph]["mlir"].replace(".mlir", ".vmfb"))
        orig_res = None
        orig_stats = None
        if os.path.exists(orig_vmfb):
            mod0 = rt.VmModule.from_flatbuffer(rt.VmInstance(), open(orig_vmfb, "rb").read())
            vm0 = rt.load_vm_module(mod0, config=rt.Config(device=dev))
            f0 = vm0.main
            ins0 = make_inputs(graph, zeros=False)
            # restore rank-0 scalars for the ORIGINAL signature
            ins0 = [np.asarray(a).reshape(()) if (a.ndim == 1 and a.shape[0] == 1 and a.dtype.kind in "fi") else a for a in ins0]
            try:
                orig_res = norm(f0(*ins0))
                for _ in range(20):
                    norm(f0(*ins0))
                ms = []
                for _ in range(args.steps):
                    t0 = time.perf_counter()
                    norm(f0(*ins0))
                    ms.append((time.perf_counter() - t0) * 1e3)
                orig_stats = stats_ms(ms)
                print(f"original vmfb (rank-0 key) run: {orig_stats}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"original vmfb run failed: {exc}", flush=True)
                orig_stats = f"error: {exc}"

        results["graphs"][graph] = {
            "iree_compile_s": round(ctime, 1),
            "parity_runs": run_records,
            "run_batches_ms": batches,
            "original_vmfb_ms": orig_stats,
        }

    with open(os.path.join(HERE, "results_iree.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults_iree.json written", flush=True)


if __name__ == "__main__":
    main()
