"""Shared helpers for the xla-vs-iree GPU probe (bench_logs/).

- MLIR surgery: promote rank-0 @main inputs to rank-1 + a leading reshape.
  Workaround for a jax_cuda12_pjrt 0.4.38 plugin quirk: PJRT_Client_
  BufferFromHostBuffer with num_dims=0 (even dims=NULL) creates a rank-1
  [1] buffer, so executables with rank-0 inputs fail at execute
  ("Executable expected shape s64[] for argument 3 but got incompatible
  shape s64[1]{0}"). Promoting the input to [1] and reshaping inside the
  graph is value-identical (verified).
- Graph/input/spec tables shared by the xla, iree and parity probes.
"""
from __future__ import annotations
import os
import re

import numpy as np

# etl checkout with the xla fixes (commit 763a415 + a local rank-0
# NULL-dims patch in xla_util.buffer_from_host that did NOT fix the
# plugin quirk; scripts import etl via sys.path.insert(0, CLONE)).
CLONE = "/mnt/hdd_pool/bchuang/tmp_pjrt_probe/etl_763a415"
# Real PJRT GPU plugin (jax_cuda12_pjrt 0.4.38 wheel).
PLUGIN = "/mnt/hdd_pool/bchuang/tmp_pjrt_probe/x/jax_plugins/xla_cuda12/xla_cuda_plugin.so"
# Read-only source of the phase-A StableHLO dumps.
MLIR_SRC = (
    "/mnt/hdd_pool/bchuang/evox-refactor/.genesis/workers/worker_T1_A1/"
    "benchmarks/results"
)
HERE = os.path.dirname(os.path.abspath(__file__))
XLA_OUT = os.path.join(HERE, "results_xla")
IREE_OUT = os.path.join(HERE, "results_iree")

# graph -> (mlir dump name, input [(shape, dtype)] POST-SURGERY, output shapes).
# Note: rank-0 inputs (de arg3 key; pso arg6 gbest_fit scalar, arg7 key)
# become (1,) after surgery.
GRAPHS = {
    "de": {
        "mlir": "de_4096x50_fused.mlir",
        "ins": [
            ([4096, 50], "f32"),  # pop
            ([4096], "f32"),      # fit
            ([4096, 50], "f32"),  # trial_pop
            ([1], "i64"),         # key (was rank-0)
        ],
        "outs": [
            ([4096, 50], "f32"), ([4096], "f32"), ([4096, 50], "f32"),
            ([4096, 50], "f32"), ([4096], "f32"), ([4096], "f32"),
        ],
    },
    "pso": {
        "mlir": "pso_4096x50_fused.mlir",
        "ins": [
            ([4096, 50], "f32"),  # pop
            ([4096], "f32"),      # fit
            ([4096, 50], "f32"),  # velocity
            ([4096, 50], "f32"),  # local_best_location
            ([4096], "f32"),      # local_best_fit
            ([50], "f32"),        # global_best_location
            ([1], "f32"),         # global_best_fit (was rank-0)
            ([1], "i64"),         # key (was rank-0)
        ],
        "outs": [
            ([4096, 50], "f32"), ([4096], "f32"), ([4096, 50], "f32"),
            ([4096, 50], "f32"), ([4096], "f32"), ([50], "f32"), ([] , "f32"),
            ([4096, 50], "f32"), ([4096], "f32"), ([4096], "f32"),
        ],
    },
}

_DT = {"f32": np.float32, "f64": np.float64, "i64": np.int64, "i32": np.int32}


def surg_rank0_inputs(mlir_text: str) -> str:
    """Rewrite @main's rank-0 scalar inputs to rank-1 + leading reshape.

    Value-identical: the reshape of a [1] tensor to a scalar is an
    identity in value space. Only the func.func header args section is
    touched; outputs and body ops are untouched (except %argN operand
    rewrites).
    """
    lines = mlir_text.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith("func.func @main"):
            header = ln
            break
    else:
        raise RuntimeError("no func.func @main found")
    m = re.search(r"\((.*?)\) ->", header)
    if not m:
        raise RuntimeError("cannot parse @main args section: " + header)
    args_section = m.group(1)
    scalars = re.findall(
        r"%arg(\d+): tensor<(f32|f64|i8|i16|i32|i64|u8|u16|u32|u64)>",
        args_section,
    )
    new_args = args_section
    for idx, typ in scalars:
        new_args = new_args.replace(
            f"%arg{idx}: tensor<{typ}>", f"%arg{idx}: tensor<1x{typ}>"
        )
    lines[i] = header.replace(args_section, new_args)
    reshapes = [
        f"    %arg{idx}_r = stablehlo.reshape %arg{idx} : "
        f"(tensor<1x{typ}>) -> tensor<{typ}>"
        for idx, typ in scalars
    ]
    out = lines[: i + 1] + reshapes
    for ln in lines[i + 1:]:
        for idx, _typ in scalars:
            ln = re.sub(rf"%arg{idx}(?![0-9])", f"%arg{idx}_r", ln)
        out.append(ln)
    return "\n".join(out)


def load_surg_mlir(graph: str) -> tuple[str, str]:
    """Load the dump and produce the surg'd text; return (orig, surg)."""
    src = os.path.join(MLIR_SRC, GRAPHS[graph]["mlir"])
    orig = open(src).read()
    surg = surg_rank0_inputs(orig)
    out = os.path.join(HERE, GRAPHS[graph]["mlir"].replace(".mlir", "_key1.mlir"))
    if not os.path.exists(out) or open(out).read() != surg:
        with open(out, "w") as f:
            f.write(surg)
    return orig, surg


def make_inputs(graph: str, zeros: bool = False, key: int = 42) -> list[np.ndarray]:
    """Build input arrays matching the POST-SURGERY signature."""
    rng = np.random.default_rng(0)
    out = []
    for shape, dtype in GRAPHS[graph]["ins"]:
        if zeros:
            arr = np.zeros(shape, _DT[dtype])
        else:
            if dtype == "f32":
                arr = rng.uniform(-5.12, 5.12, shape).astype(np.float32)
            else:  # i64 key
                arr = np.full(shape, key, np.int64)
        out.append(arr)
    return out


def output_shapes(graph: str) -> list[tuple]:
    return [tuple(s) for s, _d in GRAPHS[graph]["outs"]]


def stats_ms(ms: list[float]) -> dict:
    import statistics
    s = sorted(ms)
    return {
        "median": round(statistics.median(s), 4),
        "mean": round(statistics.mean(s), 4),
        "p90": round(s[int(len(s) * 0.9)], 4),
        "min": round(s[0], 4),
    }
