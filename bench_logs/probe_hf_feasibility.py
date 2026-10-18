"""probe_hf_feasibility — can etl run a FULL HuggingFace transformer forward
pass as ONE compiled graph? Scratch probe (bench_logs is throwaway).

For each model it:
  1. loads a REAL tiny HF checkpoint (config.json + safetensors / torch .bin,
     numpy only — no torch, no transformers, no safetensors lib, no hub libs),
  2. hand-builds the complete forward pass as ONE ``@etl.defn`` graph with the
     weights as explicit graph INPUTS (a nested dict/list pytree of specs),
     static shapes only (batch 1, fixed seq len),
  3. validates the numpy interpreter backend against an independent pure-numpy
     reference forward (same weights),
  4. compiles the SAME graph with ``backend="iree"`` (``llvm-cpu``) — reporting
     whether the whole graph compiles, compile wall time, run time, and parity
     vs both the numpy backend and the reference,
  5. reports graph size (op histogram, StableHLO text size, input count).

Usage:
    uv run python bench_logs/probe_hf_feasibility.py \
        [--models tiny_llama,tiny_mistral,tiny_gpt2] [--seq-len 8] \
        [--models-dir $TMPDIR/hf_models] [--json out.json] [--no-iree] \
        [--repeat 5]

Partial results are written to --json after EVERY model, so a wall-clock
timeout (``timeout 600 uv run python ...``) still leaves a report behind.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import etl
from hf_probe_lib import (
    count_ops,
    gpt2_params,
    gpt2_reference,
    graph_stats,
    llama_params,
    llama_reference,
    load_checkpoint,
    make_gpt2_graph,
    make_llama_graph,
    param_count,
    spec_tree,
)

MODELS = {
    # name: (family, subdir, default seq len)
    "tiny_llama": ("llama", "tiny_llama", 8),
    "tiny_mistral": ("llama", "tiny_mistral", 8),
    "tiny_gpt2": ("gpt2", "tiny_gpt2", 8),
    "distilgpt2": ("gpt2", "distilgpt2", 32),
    "gpt2": ("gpt2", "gpt2", 32),
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def best_of(fn, n: int) -> float:
    ts = []
    for _ in range(max(1, n)):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def rel_err(got: np.ndarray, want: np.ndarray) -> dict:
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    scale = max(float(np.abs(want).max()), 1e-30)
    return {
        "ref_abs_max": float(np.abs(want).max()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel_vs_max_abs": float(diff.max() / scale),
        "argmax_rel": float((diff / (np.abs(want.astype(np.float64)) + 1e-8)).max()),
    }

def build_model(name: str, models_dir: Path, seq_len: int | None):
    family, subdir, default_seq = MODELS[name]
    cfg, sd, fmt = load_checkpoint(models_dir / subdir)
    params = llama_params(cfg, sd) if family == "llama" else gpt2_params(cfg, sd)
    if family == "llama":
        maxpos = int(cfg.get("max_position_embeddings", 2048))
    else:
        maxpos = int(cfg.get("n_positions", 1024))
    s = seq_len or default_seq
    if s > maxpos:
        raise ValueError(f"{name}: seq_len {s} > max positions {maxpos}")
    return {
        "family": family,
        "cfg": cfg,
        "fmt": fmt,
        "params": params,
        "seq_len": s,
        "model_type": cfg.get("model_type"),
        "arch": (cfg.get("architectures") or ["?"])[0],
        "default_act": cfg.get("hidden_act") or cfg.get("activation_function") or "silu",
    }


def make_forward(m: dict, act: str):
    """Build the @etl.defn forward for this model with a given hidden activation."""
    if m["family"] == "llama":
        return make_llama_graph(m["cfg"], m["seq_len"], act)
    return make_gpt2_graph(m["cfg"], m["seq_len"])


def run_reference(m: dict, params: dict, ids: np.ndarray, act: str) -> np.ndarray:
    if m["family"] == "llama":
        return llama_reference(m["cfg"], params, ids, act)
    return gpt2_reference(m["cfg"], params, ids)


def probe(name: str, models_dir: Path, seq_len: int | None, do_iree: bool, repeat: int,
          hidden_act: str | None = None, iree_compile_args: list[str] | None = None) -> dict:
    r: dict = {"model": name, "status": "running"}
    m = build_model(name, models_dir, seq_len)
    cfg, params, s = m["cfg"], m["params"], m["seq_len"]
    VOCAB = int(cfg.get("vocab_size", cfg.get("n_vocab", params["lm_head"].shape[1])))
    act = hidden_act or m["default_act"]
    r.update(
        {
            "family": m["family"],
            "arch": m["arch"],
            "model_type": m["model_type"],
            "weights_format": m["fmt"],
            "seq_len": s,
            "params": param_count(params),
            "vocab": VOCAB,
            "hidden_act": act,
            "config": {
                k: cfg[k]
                for k in (
                    "hidden_size",
                    "n_embd",
                    "num_hidden_layers",
                    "n_layer",
                    "num_attention_heads",
                    "n_head",
                    "num_key_value_heads",
                    "intermediate_size",
                    "hidden_act",
                    "activation_function",
                    "tie_word_embeddings",
                )
                if k in cfg
            },
        }
    )

    rng = np.random.default_rng(0)
    ids = rng.integers(0, VOCAB, size=(1, s)).astype(np.int32)

    # -- trace -------------------------------------------------------------
    specs = spec_tree(params)
    ispec = etl.TensorSpec((1, s), etl.int32)
    forward = make_forward(m, act)
    t0 = time.perf_counter()
    graph = etl.trace(forward, specs, ispec)
    r["trace_s"] = time.perf_counter() - t0
    graph.verify()
    try:
        export_text = etl.backends.stablehlo.export(graph)
        r["export"] = {"ok": True}
    except etl.BackendError as e:
        export_text = None
        r["export"] = {"ok": False, "error": str(e), "blocking_ops": _blocking_op(str(e), graph)}

    # -- numpy interpreter backend vs independent numpy reference ----------
    ex_np, r["build_numpy_s"] = timed(etl.build, forward, specs, ispec)
    out_np = etl.run(ex_np, params, ids).numpy()
    ref = run_reference(m, params, ids, act)
    assert out_np.shape == ref.shape, (out_np.shape, ref.shape)
    r["trace_s"] = r["trace_s"]
    r["graph"] = graph_stats(graph, export_text)
    r["numpy_backend"] = {
        "logits_shape": list(out_np.shape),
        "logits_abs_max": float(np.abs(ref).max()),
        "vs_reference": rel_err(out_np, ref),
        "run_ms_best": best_of(lambda: etl.run(ex_np, params, ids), repeat) * 1e3,
    }
    r["finite"] = bool(np.isfinite(out_np).all())

    # -- if export deferred on the hidden activation, retry with the
    #    approximation the model family actually ships (GPT-2 gelu_new) ------
    if export_text is None and do_iree:
        alt = "gelu_new" if act in ("gelu",) else None
        if alt is not None:
            alt_ref = run_reference(m, params, ids, alt)
            forward_alt = make_forward(m, alt)
            graph = etl.trace(forward_alt, specs, ispec)
            graph.verify()
            export_text = etl.backends.stablehlo.export(graph)
            ex_np_alt = etl.build(forward_alt, specs, ispec)
            out_np_alt = etl.run(ex_np_alt, params, ids).numpy()
            r["activation_fallback"] = {
                "from": act,
                "to": alt,
                "approx_delta_vs_exact": rel_err(alt_ref, ref),
                "numpy_backend_vs_alt_reference": rel_err(out_np_alt, alt_ref),
                "graph": graph_stats(graph, export_text),
            }
            forward, ref, out_np = forward_alt, alt_ref, out_np_alt
            r["hidden_act_compiled"] = alt

    if not do_iree:
        r["status"] = "ok (numpy only)"
        return r
    if export_text is None:
        r["status"] = "IR-EXPORT DEFERRAL"
        return r

    # -- iree / llvm-cpu: does the WHOLE graph compile? ---------------------
    r["iree_compile_args"] = list(iree_compile_args or [])
    try:
        low, r["lower_s"] = timed(etl.lower, graph, backend="iree", target_backends=["llvm-cpu"])
        ckwargs = {"iree_compile_args": list(iree_compile_args)} if iree_compile_args else {}
        art, r["compile_s"] = timed(etl.compile, low, backend="iree", **ckwargs)
        ex, r["load_s"] = timed(etl.load, art, backend="iree")
    except etl.BackendError as e:
        r["status"] = "IREE DEFERRAL"
        r["blocked_by"] = str(e)
        r["blocked_op"] = _blocking_op(str(e), graph)
        return r
    out_iree = etl.run(ex, params, ids).numpy()
    r["iree"] = {
        "target": "llvm-cpu",
        "whole_graph_compiled": True,
        "vs_reference": rel_err(out_iree, ref),
        "vs_numpy_backend": rel_err(out_iree, out_np),
        "run_ms_best": best_of(lambda: etl.run(ex, params, ids), repeat) * 1e3,
        "peak_rss_mb": peak_rss_mb(),
    }
    r["iree"]["speedup_vs_numpy_backend"] = (
        r["numpy_backend"]["run_ms_best"] / r["iree"]["run_ms_best"]
    )
    r["status"] = "ok"
    return r

def _blocking_op(msg: str, graph) -> list[str]:
    """Best-effort: which op names appearing in the graph are named in the error."""
    ops = count_ops(graph)
    return sorted(n for n in ops if n in msg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny_llama,tiny_mistral,tiny_gpt2")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--models-dir", default=os.path.join(os.environ.get("TMPDIR", "."), "hf_models"))
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-iree", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--hidden-act",
        default=None,
        help="override the config's hidden activation (e.g. gelu_new to sidestep the "
        "erf/GELU v1 export deferral)",
    )
    ap.add_argument(
        "--iree-compile-args",
        default=None,
        help="comma-separated iree-compile flags passed at COMPILE time (e.g. "
        "'--iree-llvmcpu-target-cpu=host,--iree-opt-data-tiling')",
    )
    args = ap.parse_args(argv)
    cargs = [a.strip() for a in args.iree_compile_args.split(",")] if args.iree_compile_args else None
    models_dir = Path(args.models_dir)
    results = []
    out_path = Path(args.json) if args.json else None
    for name in [n.strip() for n in args.models.split(",") if n.strip()]:
        print(f"\n===== {name} =====", flush=True)
        t0 = time.perf_counter()
        try:
            r = probe(name, models_dir, args.seq_len, not args.no_iree, args.repeat,
                      args.hidden_act, cargs)
        except Exception as e:  # noqa: BLE001 - probe: record, never swallow silently
            r = {"model": name, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        r["wall_s"] = time.perf_counter() - t0
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)
        if out_path:
            out_path.write_text(json.dumps(results, indent=2))

    if out_path:
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
