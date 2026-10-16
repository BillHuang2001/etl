#!/usr/bin/env python3
"""Crunch the raw `etl.bench` matrix JSONs into the tables/aggregates used by
`rocm_mi300x_results.md`.

This is evidence/tooling, not part of the `etl.bench` public API: it reads the
committed raw JSONs in this directory with the stdlib only and prints the
per-backend conformance summary, the timing matrix, the build-time table and the
HEADLINE table. Run from anywhere:

    python3 etl/bench/rocm_mi300x_data/gen_fragments.py [section ...]

Sections: conf, matrix, matrix_highest, ratios, build, headline, errors, all
(default: all).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "numpy_cpu": "A_numpy_cpu.json",
    "iree_cpu": "B_iree_cpu.json",
    "iree_rocm": "C_iree_rocm.json",
    "xla_conf": "D_xla_conf_highest.json",
    "xla_tf32": "D_xla_bench_tf32.json",
    "xla_highest": "D_xla_bench_highest.json",
}

COLS = [
    ("numpy/cpu", "numpy_cpu"),
    ("iree/llvm-cpu", "iree_cpu"),
    ("iree/rocm:0", "iree_rocm"),
    ("xla/rocm:0", "xla_tf32"),
]


def load(key: str) -> dict:
    with open(os.path.join(HERE, FILES[key])) as fh:
        return json.load(fh)


def fmt(x, sig=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000 or ax < 1e-3:
        return f"{x:.2e}"
    return f"{x:.{sig}g}"


def conf_stats(d: dict) -> dict:
    """n rows, numeric pass/fail, error rows, crashes + errors grouped by text."""
    conf = d.get("conf") or {}
    npass = nfail = nerr = 0
    groups: dict[str, list[str]] = {}
    for name, r in conf.items():
        if r.get("error"):
            nerr += 1
            groups.setdefault(r["error"], []).append(name)
        elif r.get("numpy_pass"):
            npass += 1
        else:
            nfail += 1
    return {
        "rows": len(conf),
        "pass": npass,
        "fail": nfail,
        "err": nerr,
        "crash": len(d.get("crashes") or []),
        "groups": groups,
    }


def torch_stats(d: dict) -> tuple[int, int, int]:
    """(torch refs present+pass, torch refs present+fail, no torch ref)."""
    ok = bad = nref = 0
    for r in (d.get("conf") or {}).values():
        tp = r.get("torch_pass")
        if tp is True:
            ok += 1
        elif tp is False:
            bad += 1
        else:
            nref += 1
    return ok, bad, nref


def build_stats(key: str) -> dict:
    d = load(key)
    ms = sorted(v for v in (d.get("build_ms") or {}).values())
    if not ms:
        return {"n": 0}
    total = sum(ms) / 1000.0

    def pct(p):
        i = min(len(ms) - 1, int(round(p * (len(ms) - 1))))
        return ms[i]

    return {
        "n": len(ms),
        "min": ms[0],
        "median": statistics.median(ms),
        "p90": pct(0.9),
        "max": ms[-1],
        "total_s": total,
        "slow": sorted(
            ((v, k) for k, v in (d.get("build_ms") or {}).items()), reverse=True
        )[:8],
    }


def wall(key: str) -> tuple[float, float]:
    d = load(key)
    conf = sum((d.get("conf_wall_s") or {}).values())
    bench = sum((d.get("bench_wall_s") or {}).values())
    return conf, bench


def bench_ms(key: str, name: str):
    return ((load(key).get("bench") or {}).get(name) or {}).get("etl_ms")


def conf_ms(key: str, name: str):
    return ((load(key).get("conf") or {}).get(name) or {}).get("etl_ms")


def conf_err(key: str, name: str):
    return ((load(key).get("conf") or {}).get(name) or {}).get("error")


def max_abs(key: str, name: str):
    return ((load(key).get("conf") or {}).get(name) or {}).get("max_abs_error")


# --------------------------------------------------------------------------- #
def section_conf():
    print("## Per-backend conformance summary (generated)\n")
    print("| backend | rows | numeric PASS | numeric FAIL | deferral/error | crash | conf wall | bench wall |")
    print("|---|---|---|---|---|---|---|---|")
    for label, key in [
        ("numpy/cpu", "numpy_cpu"),
        ("iree/llvm-cpu", "iree_cpu"),
        ("iree/rocm:0", "iree_rocm"),
        ("xla/rocm:0", "xla_conf"),
    ]:
        s = conf_stats(load(key))
        cw, bw = wall(key)
        print(
            f"| {label} | {s['rows']} | {s['pass']} | {s['fail']} | {s['err']} | "
            f"{s['crash']} | {cw:.1f} s | {bw:.1f} s |"
        )
    print()
    for label, key in [("numpy/cpu", "numpy_cpu"), ("iree/llvm-cpu", "iree_cpu"),
                       ("iree/rocm:0", "iree_rocm"),
                       ("xla/rocm:0 (HIGHEST)", "xla_conf")]:
        ok, bad, nref = torch_stats(load(key))
        print(f"- torch {label}: pass={ok} fail={bad} no-ref={nref}")


def section_errors():
    print("\n## Deferrals/errors (generated)\n")
    for label, key in [("numpy/cpu", "numpy_cpu"), ("iree/llvm-cpu", "iree_cpu"),
                       ("iree/rocm:0", "iree_rocm"), ("xla/rocm:0", "xla_conf")]:
        s = conf_stats(load(key))
        print(f"**{label}** — {len(s['groups'])} distinct error(s), {s['err']} rows, {s['crash']} crash")
        for err, names in sorted(s["groups"].items(), key=lambda kv: -len(kv[1])):
            first = err.splitlines()[0][:110]
            print(f"  - [{len(names):>2}] {first} :: {', '.join(sorted(names))}")
        print()


def section_matrix():
    print("\n## Timing matrix (generated)\n")
    d = load("numpy_cpu")
    names = list(d["conf"].keys())
    tf32 = load("xla_tf32")
    print("| example | numpy/cpu | iree/llvm-cpu | iree/rocm:0 | xla/rocm:0(TF32) | torch-cpu |")
    print("|---|---|---|---|---|---|")
    for n in names:
        cells = []
        for key in ["numpy_cpu", "iree_cpu", "iree_rocm"]:
            v = bench_ms(key, n)
            er = conf_err(key, n) or ((load(key).get("bench") or {}).get(n) or {}).get("error")
            cells.append(fmt(v) if v is not None else ("ERR"))
        tv = bench_ms("xla_tf32", n)
        ter = ((tf32.get("bench") or {}).get(n) or {}).get("error")
        cells.append(fmt(tv) if tv is not None else "ERR")
        tc = ((load("numpy_cpu").get("bench") or {}).get(n) or {}).get("torch_ms")
        print(f"| {n} | " + " | ".join(cells) + f" | {fmt(tc)} |")


def section_matrix_highest():
    print("\n## xla dot_precision=HIGHEST timings (generated)\n")
    hi = load("xla_highest")
    tf = load("xla_tf32")
    print("| example | TF32 ms | HIGHEST ms | ratio H/TF32 |")
    print("|---|---|---|---|")
    for n in hi["bench"]:
        t = ((tf.get("bench") or {}).get(n) or {}).get("etl_ms")
        h = (hi["bench"][n] or {}).get("etl_ms")
        r = (h / t) if (t and h) else None
        print(f"| {n} | {fmt(t)} | {fmt(h)} | {r:.2f} |" if r else f"| {n} | {fmt(t)} | {fmt(h)} | - |")


def section_ratios():
    print("\n## Cross-backend ratios (generated)\n")
    names = list(load("numpy_cpu")["bench"].keys())
    print("| example | numpy/iree-cpu | iree-cpu/iree-rocm | iree-rocm/xla-TF32 |")
    print("|---|---|---|---|")
    for n in names:
        a = bench_ms("numpy_cpu", n)
        b = bench_ms("iree_cpu", n)
        c = bench_ms("iree_rocm", n)
        x = bench_ms("xla_tf32", n)
        print(
            f"| {n} | "
            + (f"{a / b:.3g}×" if a and b else "-")
            + " | "
            + (f"{b / c:.3g}×" if b and c else "-")
            + " | "
            + (f"{c / x:.3g}×" if c and x else "-")
            + " |"
        )


def section_build():
    print("\n## Build wall-times (generated)\n")
    print("| backend | builds | min | median | p90 | max | total |")
    print("|---|---|---|---|---|---|---|")
    for label, key in [("numpy/cpu", "numpy_cpu"), ("iree/llvm-cpu", "iree_cpu"),
                       ("iree/rocm:0", "iree_rocm"), ("xla/rocm:0", "xla_tf32")]:
        s = build_stats(key)
        print(
            f"| {label} | {s['n']} | {fmt(s.get('min'))} | {fmt(s.get('median'))} | "
            f"{fmt(s.get('p90'))} | {fmt(s.get('max'))} | {s['total_s']:.1f} s |"
        )
    print()
    for label, key in [("iree/llvm-cpu", "iree_cpu"), ("iree/rocm:0", "iree_rocm"),
                       ("xla/rocm:0", "xla_tf32")]:
        s = build_stats(key)
        slow = ", ".join(f"`{k}` {v:.0f}" for v, k in s["slow"])
        print(f"- {label} slowest: {slow}")


HL = [
    "matmul_1024", "matmul_square", "transformer", "attention", "conv2d_large",
    "resnet_conv_block", "nbody", "layernorm_large",
    "e2e_train_mlp", "e2e_train_convnet", "e2e_train_transformer",
    "e2e_infer_transformer", "e2e_pso_optimize", "e2e_kmeans",
    "e2e_power_iteration", "mha_block",
]


def cell(key: str, name: str) -> str:
    """bench ms, or 'defer'/'crash'."""
    d = load(key)
    cr = {c.get("name") for c in (d.get("crashes") or [])}
    if name in cr:
        return "crash"
    b = (d.get("bench") or {}).get(name)
    if b is not None and b.get("etl_ms") is not None and not b.get("error"):
        return fmt(b["etl_ms"])
    return "defer"


def build_cell(key: str, name: str) -> str:
    b = (load(key).get("build_ms") or {}).get(name)
    return fmt(b) if b is not None else "-"


def section_headline():
    print("\n## HEADLINE table (generated)\n")
    print("| example | numpy bench | numpy build | iree-cpu bench | iree-cpu build | iree-rocm bench | iree-rocm build | xla-TF32 bench | xla build |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n in HL:
        print(
            "| " + n
            + f" | {cell('numpy_cpu', n)} | {build_cell('numpy_cpu', n)}"
            + f" | {cell('iree_cpu', n)} | {build_cell('iree_cpu', n)}"
            + f" | {cell('iree_rocm', n)} | {build_cell('iree_rocm', n)}"
            + f" | {cell('xla_tf32', n)} | {build_cell('xla_tf32', n)} |"
        )
    print()
    print("ratio = reference etl_ms / backend etl_ms (>1 = accelerator faster)")
    print()
    print("| example | numpy-cpu/iree-cpu | iree-cpu/iree-rocm | iree-rocm/xla-TF32 |")
    print("|---|---|---|---|")
    for n in HL:
        a = bench_ms("numpy_cpu", n)
        b = bench_ms("iree_cpu", n)
        c = bench_ms("iree_rocm", n)
        x = bench_ms("xla_tf32", n)
        print(
            f"| {n} | "
            + (f"{a / b:.3g}×" if a and b else "-")
            + " | "
            + (f"{b / c:.3g}×" if b and c else "-")
            + " | "
            + (f"{c / x:.3g}×" if c and x else "-")
            + " |"
        )


SECTIONS = {
    "conf": section_conf,
    "errors": section_errors,
    "matrix": section_matrix,
    "matrix_highest": section_matrix_highest,
    "ratios": section_ratios,
    "build": section_build,
    "headline": section_headline,
}


def main(argv):
    want = argv or ["all"]
    if "all" in want:
        want = list(SECTIONS)
    for s in want:
        SECTIONS[s]()


if __name__ == "__main__":
    main(sys.argv[1:])
