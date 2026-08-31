#!/usr/bin/env python3
"""Parity analysis for the xla/iree GPU probe.

1. xla vs iree on the IDENTICAL surg'd mlir text (saved npz outputs):
   per-output bit-exactness + max-abs/max-rel.
2. xla vs numpy backend reference: re-trace the same fused algorithm
   graphs via evox (worker_T1_A1/src) with ETL_BACKEND=numpy, run one
   step, feed the pre-step state tensors into the xla executable, and
   compare the outputs (state leaves + extras).

Env for the xla re-run (same as probe_xla_gpu.py): CUDA_VISIBLE_DEVICES,
ETL_PJRT_PLUGIN, PATH (ptxas), LD_LIBRARY_PATH. ETL_BACKEND=numpy is set
here for the evox side.
Usage: python bench_logs/probe_parity.py [--gpu 3]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ["ETL_BACKEND"] = "numpy"

CLONE = "/mnt/hdd_pool/bchuang/tmp_pjrt_probe/etl_763a415"
EVOX_SRC = "/mnt/hdd_pool/bchuang/evox-refactor/.genesis/workers/worker_T1_A1/src"
sys.path.insert(0, CLONE)
sys.path.insert(0, EVOX_SRC)

import numpy as np  # noqa: E402
import etl  # noqa: E402
from etl import core  # noqa: E402
from etl.backends.program import LoweredProgram, Signature  # noqa: E402

from _common import GRAPHS, PLUGIN, XLA_OUT, IREE_OUT, HERE, load_surg_mlir  # noqa: E402

_DT = {"f32": core.float32, "i64": core.int64}


def cmp_pair(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    exact = bool(np.array_equal(a, b))
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    denom = np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64)))
    rel = d / np.maximum(denom, 1e-30)
    return {
        "bit_exact": exact,
        "max_abs": float(np.max(d)) if d.size else 0.0,
        "max_rel": float(np.max(rel)) if rel.size else 0.0,
    }


def compare_npz(graph: str, label: str) -> list[dict]:
    xo = np.load(os.path.join(XLA_OUT, f"{graph}_{label}_outs.npz"))
    io = np.load(os.path.join(IREE_OUT, f"{graph}_{label}_outs.npz"))
    n = len(xo.files)
    per = []
    for i in range(n):
        per.append(cmp_pair(xo[f"o{i}"], io[f"o{i}"]))
    return per


def xla_compile(graph: str, surg: str):
    ins = tuple(core.TensorSpec(shape, _DT[d]) for shape, d in GRAPHS[graph]["ins"])
    outs = tuple(core.TensorSpec(shape, _DT[d]) for shape, d in GRAPHS[graph]["outs"])
    lp = LoweredProgram(
        backend="xla",
        signature=Signature(input_specs=ins, output_specs=outs),
        payload={"format": "stablehlo", "format_version": 1, "mlir_text": surg},
    )
    art = etl.compile(lp, backend="xla", plugin_path=PLUGIN)
    return etl.load(art).backend_executable


def numpy_parity() -> dict:
    """Trace the fused graphs via evox on the numpy backend; compare with xla."""
    from evox.algorithms.so.de_variants.de import DE  # noqa: E402
    from evox.algorithms.so.pso_variants.pso import PSO  # noqa: E402
    from evox.problems.numerical.basic import Sphere  # noqa: E402
    from evox.workflows import StdWorkflow  # noqa: E402

    lb = etl.tensor(np.full((1, 50), -5.12, np.float32))
    ub = etl.tensor(np.full((1, 50), 5.12, np.float32))
    out = {}
    for graph in ("de", "pso"):
        print(f"\n--- numpy parity: {graph} ---", flush=True)
        if graph == "de":
            algo = DE(pop_size=4096, dim=50, lb=lb, ub=ub)
            state0 = StdWorkflow(algo, Sphere(), fused=True).init_step(
                etl.random.key(41))
            wf = StdWorkflow(algo, Sphere(), fused=True)
            state1 = wf.step(etl.random.key(42))
            ins = [
                state0.pop, state0.fit, state0.trial_pop, np.array([42], np.int64)
            ]
        else:
            algo = PSO(lb=lb, ub=ub, pop_size=4096, dim=50)
            wf = StdWorkflow(algo, Sphere(), fused=True)
            state0 = wf.init_step(etl.random.key(41))
            state1 = wf.step(etl.random.key(42))
            ins = [
                state0.pop, state0.fit, state0.velocity,
                state0.local_best_location, state0.local_best_fit,
                state0.global_best_location, state0.global_best_fit,
                np.array([42], np.int64),
            ]
        leaves, treedef = core.flatten(state1)
        print("state1 leaves:", len(leaves), "treedef:", type(treedef).__name__,
              flush=True)
        _, surg = load_surg_mlir(graph)
        be = xla_compile(graph, surg)
        xouts = be.run([core.Tensor(np.asarray(a)) for a in ins])
        per = []
        for i, xo in enumerate(xouts):
            ref = leaves[i].numpy() if i < len(leaves) else None
            if ref is not None:
                per.append({"output": i, "vs_state_leaf": cmp_pair(xo.numpy(), ref)})
            else:
                per.append({"output": i, "vs_state_leaf": "no matching state leaf"})
        out[graph] = {
            "n_leaves": len(leaves),
            "per_output": per,
            "all_finite": all(np.isfinite(o.numpy()).all() for o in xouts),
        }
        for p in per:
            print(p, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    report = {"graphs": {}}
    for graph in ("de", "pso"):
        per_real = compare_npz(graph, "real")
        per_zero = compare_npz(graph, "zeros")
        report["graphs"][graph] = {
            "xla_vs_iree_real": per_real,
            "xla_vs_iree_zeros": per_zero,
        }
        exact = sum(1 for p in per_real if p.get("bit_exact"))
        print(f"\n{graph}: xla-vs-iree (real inputs): {exact}/{len(per_real)} "
              f"outputs bit-exact; max_abs per output: "
              f"{[round(p.get('max_abs', -1), 9) for p in per_real]}", flush=True)

    report["numpy"] = numpy_parity()
    with open(os.path.join(HERE, "results_parity.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nresults_parity.json written", flush=True)


if __name__ == "__main__":
    main()
