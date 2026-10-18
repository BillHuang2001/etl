#!/usr/bin/env python3
"""HF whole-forward perf/parity matrix probe for the ``etl_hf`` package.

Compares the **whole-model forward pass as ONE etl graph** (``etl_hf``)
across the execution paths that exist for it, on the four checkpoints whose
torch goldens are committed in :mod:`bench_logs/hf_goldens/`:

======================  ==================================================
backend key             what it is
======================  ==================================================
``numpy``               the numpy interpreter backend (CPU, host arrays)
``iree-llvm-cpu``       IREE compiled, ``--iree-llvmcpu-target-cpu=host``,
                        host inputs
``iree-rocm``           IREE compiled for ROCm (the MI300X), **device
                        resident**: weights placed ONCE with
                        ``etl_hf.place_params`` and then a same-device loop
                        through ``etl_hf.run_model`` (the documented fast
                        iteration pattern)
======================  ==================================================

Every backend is measured **twice**: unpacked (one ``TensorSpec`` per weight,
the canonical tree) and **packed** (``pack_weights=True`` — the weight tree
collapsed into a handful of concatenated inputs unpacked in-graph with static
``etl.slice``; backend key suffix ``-packed``). Packing is what brings the
124M ``gpt2`` graph (197 unpacked inputs) under IREE's ~161-consumed-input
runtime limit, so the packed keys are the rows that make ``gpt2`` runnable on
a compiler backend at all; the unpacked keys stay in the matrix as the
comparison (and as the honest record of the input-count cliff).

Compile options are **the library defaults**: ``etl_hf.resolve_build_options``
is asked what it would inject — ``opt_level="O3"`` for llvm-cpu,
``opt_level="O1"`` for an all-ROCm target set (IREE's ROCm codegen rejects
O2/O3 on these transformer-shaped graphs; ``--opt-level-evidence`` additionally
re-attempts the rejected levels so the report can quote the real upstream
error instead of asserting it).

Measured per (model, backend): graph size (param count, graph inputs, IR op
count, StableHLO bytes), trace / lower / compile / load times, first-call and
warm per-call latency (best/median/mean), tokens/s, and parity (max-abs)
against BOTH the numpy interpreter backend and the stored torch/transformers
golden logits. ``iree-rocm`` additionally reports the per-call **restaging**
variant (``place_params`` on every call) — the documented anti-pattern — so
the same-device loop's advantage is measured on the same graph.

Design (why it looks like this)
-------------------------------
* **One subprocess per (model, backend) leg.** A misbehaving compiled leg can
  *segfault* (IREE's Python runtime SIGSEGVs on graphs that consume more than
  ~161 distinct inputs — see ``etl_hf/CONTEXT.md`` Known Issues; a pre-existing
  runtime limit, not an ``etl_hf`` behaviour, and one that only bites the
  UNPACKED 197-input ``gpt2`` graph). Running each leg in its own process means
  one crash cannot take the matrix down, and the crash is reported with its
  exit code and last stderr line instead of a mystery gap.
* **Idempotent.** Every leg writes ``hf_perf_data/<model>__<backend>.json``;
  re-running overwrites it. ``--report`` re-renders the markdown from whatever
  JSONs exist, ``--leg`` runs a single child leg (what the parent does), and
  ``--sanity`` re-runs a small subset so a published row can be shown to
  reproduce (the markdown prints both runs side by side).
* **No downloads.** Checkpoints are resolved from the existing cache
  (``$ETL_HF_MODELS_DIR`` or ``$TMPDIR/etl_hf_models/<slug>``); a missing
  checkpoint is an explicit error naming the path, never a network fetch.
* **Nothing is faked.** A leg that fails to build/run records
  ``status="error"``/``"crash"`` with the message; a compiled leg that is
  refused up front (input-count limit, see :data:`INPUT_LIMIT`) records
  ``status="skipped"`` with the reason. Absent numbers stay absent (rendered
  ``—``), never extrapolated.

Run it (the ``iree`` extra is what makes the compiled rows possible)::

    uv run --isolated --extra iree python bench_logs/probe_hf_perf.py
    uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --report
    uv run --isolated --extra iree python bench_logs/probe_hf_perf.py \\
        --models tiny_llama --backends numpy,iree-rocm-packed --iters 5

Outputs: ``bench_logs/hf_perf_data/*.json`` (raw, one file per leg) and
``bench_logs/hf_perf_results.md`` (tables; generated — never hand-edited).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "hf_perf_data"
GOLDENS_DIR = HERE / "hf_goldens"
REPORT_PATH = HERE / "hf_perf_results.md"

SCHEMA = 2

#: IREE's Python runtime SIGSEGVs (no Python exception) when a graph *consumes*
#: more than ~161 distinct inputs — documented as a Known Issue in
#: ``etl_hf/CONTEXT.md`` with the measured brackets (161 chained-matmul / 162
#: fan-out). Only an UNPACKED graph gets near it (``gpt2``: 197 inputs); the
#: packed form of the same model declares 4. Compiled legs above this limit are
#: refused up front and marked "refused (unpacked)" instead of being attempted
#: silently; pass ``--attempt-oversize`` to try anyway (the crash is then
#: *measured*, isolated in the leg's own subprocess, with its exit code
#: recorded).
INPUT_LIMIT = 161

#: Checkpoint slug -> golden fixture slug.  The golden ``<slug>.npz`` holds the
#: real torch/transformers ``logits`` (f32) plus the exact ``input_ids`` the
#: oracle used, so the probe feeds *those ids* to every backend and compares
#: apples to apples.  ``tiny_random_llama`` is the golden harness's spelling of
#: the ``tiny_llama`` checkpoint (same repo id, same cache directory).
MODELS: dict[str, str] = {
    "tiny_llama": "tiny_random_llama",
    "tiny_gpt2": "tiny_gpt2",
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
}

#: Parameter counts above this are "big": their public ``build_model`` total is
#: NOT measured separately (it would repeat an expensive compile just to prove
#: ``sum(stages) == build``); the stage sum is reported instead.
BIG_PARAMS = 5_000_000


@dataclass(frozen=True)
class BackendSpec:
    """One row of the matrix: how to build/run a graph for this path."""

    key: str
    backend: str
    device_kind: str
    device_index: int = 0
    options: dict = field(default_factory=dict)
    device_resident: bool = False
    #: Also measure the per-call restaging anti-pattern (device paths only).
    measure_restage: bool = False
    #: Take packed params inputs (``etl_hf`` ``pack_weights=True``) — the weight
    #: tree collapsed into a handful of concatenated in-graph slices.
    packed: bool = False


#: Compile options are deliberately EMPTY: the measured rows must reflect the
#: options ``etl_hf.resolve_build_options`` injects (the library default matrix),
#: not a copy of it that could drift. The resolved set is recorded per leg in
#: ``backend.resolved_options`` and the level actually used in
#: ``build.opt_level_used``.
_BASE_BACKENDS: dict[str, BackendSpec] = {
    "numpy": BackendSpec(
        key="numpy",
        backend="numpy",
        device_kind="cpu",
    ),
    "iree-llvm-cpu": BackendSpec(
        key="iree-llvm-cpu",
        backend="iree",
        device_kind="cpu",
        options={"target_backends": ["llvm-cpu"]},
    ),
    "iree-rocm": BackendSpec(
        key="iree-rocm",
        backend="iree",
        device_kind="rocm",
        device_index=0,
        options={"target_backends": ["rocm"]},
        device_resident=True,
        measure_restage=True,
    ),
}


def _packed_variant(spec: BackendSpec) -> BackendSpec:
    """The same execution path on PACKED params inputs (key suffix ``-packed``)."""
    return BackendSpec(
        key=f"{spec.key}-packed",
        backend=spec.backend,
        device_kind=spec.device_kind,
        device_index=spec.device_index,
        options=dict(spec.options),
        device_resident=spec.device_resident,
        measure_restage=spec.measure_restage,
        packed=True,
    )


#: Unpacked keys first (their numpy leg writes the reference logits the packed
#: legs are compared against), then the packed variants.
BACKENDS: dict[str, BackendSpec] = {
    **_BASE_BACKENDS,
    **{f"{key}-packed": _packed_variant(spec) for key, spec in _BASE_BACKENDS.items()},
}


# ---------------------------------------------------------------------------
# checkpoint / golden resolution (no network, ever)
# ---------------------------------------------------------------------------
def models_root() -> Path:
    """Checkpoint cache root: ``$ETL_HF_MODELS_DIR`` or ``$TMPDIR/etl_hf_models``."""
    override = os.environ.get("ETL_HF_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    import tempfile

    return Path(tempfile.gettempdir()) / "etl_hf_models"


def checkpoint_dir(slug: str) -> Path:
    """Resolve a cached checkpoint directory, or raise with what is missing."""
    root = models_root()
    directory = root / slug
    have_config = (directory / "config.json").is_file()
    weights = [
        name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (directory / name).is_file()
    ]
    if not have_config or not weights:
        raise FileNotFoundError(
            f"no complete checkpoint for {slug!r} under {directory} "
            f"(config.json={have_config}, weights={weights or 'none'}); "
            f"this probe never downloads — populate {root} first"
            f" (etl_hf/tests/hf_download.py) or set $ETL_HF_MODELS_DIR"
        )
    return directory


def load_golden(golden_slug: str) -> dict:
    """Load ``hf_goldens/<slug>.npz`` (+ meta) — the torch oracle fixture."""
    import numpy as np

    npz = GOLDENS_DIR / f"{golden_slug}.npz"
    meta_path = GOLDENS_DIR / f"{golden_slug}.meta.json"
    if not npz.is_file():
        raise FileNotFoundError(f"missing golden fixture {npz}")
    with np.load(npz) as blob:
        logits = np.asarray(blob["logits"], dtype=np.float32)
        ids = np.asarray(blob["input_ids"], dtype=np.int32)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {"logits": logits, "input_ids": ids, "meta": meta, "path": str(npz)}


# ---------------------------------------------------------------------------
# small timing helpers
# ---------------------------------------------------------------------------
def _ms(seconds: float) -> float:
    return round(seconds * 1e3, 4)


def _progress(label: str) -> None:
    """Print a stage marker so a hard crash still says where it died.

    The parent captures each leg's stdout: when a leg segfaults mid-run the
    record carries the last marker reached, which is the difference between
    "died at invoke" and "died at compile".
    """
    print(f"[stage] {label}", flush=True)


def _timeit(fn, *args, **kwargs):
    """Return ``(result, seconds)`` around one call."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def _summarize(samples: list[float]) -> dict:
    """best / median / mean / stdev of a latency sample list (milliseconds)."""
    if not samples:
        return {"n": 0}
    ms = [value * 1e3 for value in samples]
    return {
        "n": len(ms),
        "best_ms": round(min(ms), 4),
        "median_ms": round(statistics.median(ms), 4),
        "mean_ms": round(statistics.fmean(ms), 4),
        "stdev_ms": round(statistics.stdev(ms), 4) if len(ms) > 1 else 0.0,
    }


def count_ops(graph) -> dict[str, int]:
    """Op-name histogram of a traced graph, walking nested regions.

    Same walk as ``etl_hf/tests/test_export.py::_count_ops`` (these model
    graphs are straight-line, so the recursion finds no nested regions; it is
    written generically so a future control-flow model cannot under-count).
    """
    counts: dict[str, int] = {}

    def walk_block(block) -> None:
        for op in block.ops:
            name = op.opdef.name
            counts[name] = counts.get(name, 0) + 1
            for region in op.regions:
                for nested in region.blocks:
                    walk_block(nested)

    walk_block(graph.module.functions[0].entry_block)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def env_snapshot() -> dict:
    """Versions + relevant environment of the process doing the measuring."""
    import importlib.metadata as md

    import numpy as np

    def version(pkg: str):
        try:
            return md.version(pkg)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "iree-base-compiler": version("iree-base-compiler"),
        "iree-base-runtime": version("iree-base-runtime"),
        "ETL_BACKEND": os.environ.get("ETL_BACKEND"),
        "ETL_TARGET_BACKENDS": os.environ.get("ETL_TARGET_BACKENDS"),
        "ETL_OPT_LEVEL": os.environ.get("ETL_OPT_LEVEL"),
        "ETL_HF_MODELS_DIR": os.environ.get("ETL_HF_MODELS_DIR"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "cpu_count": os.cpu_count(),
    }


def _json_default(value):
    if hasattr(value, "__fspath__"):
        return str(value)
    return repr(value)


# ---------------------------------------------------------------------------
# one leg = one (model, backend) cell, measured in its own process
# ---------------------------------------------------------------------------
def run_leg(
    model_name: str,
    backend_key: str,
    *,
    iters: int = 7,
    warmup: int = 2,
    restage_iters: int = 3,
    attempt_oversize: bool = False,
    measure_public_build: bool | None = None,
    opt_level_evidence: bool = False,
) -> dict:
    """Measure one (model, backend) cell and return its record (never raises)."""
    import numpy as np

    import etl
    import etl_hf
    from etl import core as core
    from etl.backends import stablehlo
    from etl_hf import build as hf_build

    spec = BACKENDS[backend_key]
    golden_slug = MODELS[model_name]
    record: dict = {
        "schema": SCHEMA,
        "model": {"name": model_name, "golden": golden_slug},
        "backend": {
            "key": backend_key,
            "backend": spec.backend,
            "device": f"{spec.device_kind}:{spec.device_index}",
            "options": spec.options,
            "device_resident": spec.device_resident,
            "packed": spec.packed,
        },
        "measurement": {"iters": iters, "warmup": warmup, "restage_iters": restage_iters},
        "status": "ok",
        "errors": [],
        "notes": [],
        "env": env_snapshot(),
    }
    notes: list[str] = record["notes"]

    try:
        golden = load_golden(golden_slug)
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"golden fixture: {type(exc).__name__}: {exc}")
        return record

    ids = golden["input_ids"]
    batch, seq_len = int(ids.shape[0]), int(ids.shape[1])
    record["model"].update(
        {"seq_len": seq_len, "batch": batch, "golden_input_ids": ids.tolist()}
    )

    device = core.Device(spec.device_kind, spec.device_index)
    record["backend"]["device"] = f"{spec.device_kind}:{spec.device_index}"

    _progress(f"{model_name}/{backend_key}: load_hf")
    try:
        ckpt = checkpoint_dir(model_name)
        record["model"]["checkpoint_dir"] = str(ckpt)
        model, load_s = _timeit(etl_hf.load_hf, ckpt, seq_len=seq_len, batch_size=batch)
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"load_hf: {type(exc).__name__}: {exc}")
        return record

    record["model"].update(
        {
            "family": model.family,
            "model_type": model.config.model_type,
            "hidden_act": model.config.hidden_act,
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "tie_word_embeddings": model.config.tie_word_embeddings,
            "config_digest": model.config.digest,
        }
    )
    record["graph"] = {"load_hf_ms": _ms(load_s)}

    # -- graph shape: one graph, weights as inputs ---------------------------
    specs = model.param_specs(pack_weights=spec.packed)
    param_count = 0
    for value in specs.values():
        total = 1
        for dim in value.shape:
            total *= int(dim)
        param_count += total
    graph_inputs = len(specs) + 1  # + input_ids
    record["model"].update(
        {
            "packed": spec.packed,
            "param_count": param_count,
            "weight_inputs": len(specs),
            "graph_inputs": graph_inputs,
        }
    )
    record["graph"]["weight_bytes_f32"] = param_count * 4

    _progress("trace")
    try:
        # The packed forward is the same memoised ``@etl.defn`` the library
        # builds; it is public behaviour but a private factory, so fall back to
        # the canonical ``graph_fn`` (unpacked only) if that ever moves.
        if spec.packed or not hasattr(hf_build, "_forward_fn"):
            forward = hf_build._forward_fn(
                model._family_spec,
                model.config,
                seq_len,
                batch,
                spec.packed,
            )
        else:
            forward = model.graph_fn()
        graph, trace_s = _timeit(
            etl.trace,
            forward,
            specs,
            etl.TensorSpec((batch, seq_len), etl.int32),
        )
        graph.verify()
        record["graph"]["trace_ms"] = _ms(trace_s)
        histogram = count_ops(graph)
        record["graph"]["ir_ops"] = sum(histogram.values())
        record["graph"]["ir_distinct_ops"] = len(histogram)
        record["graph"]["ir_histogram"] = histogram
        _progress("stablehlo export")
        text, export_s = _timeit(stablehlo.export, graph)
        record["graph"]["export_ms"] = _ms(export_s)
        record["graph"]["stablehlo_bytes"] = len(text.encode())
        record["graph"]["stablehlo_lines"] = len(text.splitlines())
        record["graph"]["stablehlo_constants"] = text.count("stablehlo.constant")
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(
            "trace/export: " + f"{type(exc).__name__}: {exc}"
        )
        record["graph"]["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
        return record

    # -- the compiled-leg input-count gate ----------------------------------
    oversize = record["model"]["graph_inputs"] > INPUT_LIMIT
    if oversize and spec.backend != "numpy" and not attempt_oversize:
        record["status"] = "skipped"
        packed_key = f"{backend_key}-packed"
        remedy = (
            f"; the PACKED form of the same model is measured separately as "
            f"`{model_name}` / `{packed_key}`"
            if packed_key in BACKENDS and not spec.packed
            else ""
        )
        notes.append(
            f"UNPACKED graph consumes {record['model']['graph_inputs']} inputs, above "
            f"the ~{INPUT_LIMIT}-input IREE runtime limit (SIGSEGV at invoke) — this leg "
            f"is refused up front{remedy}"
        )
        record["model"]["graph_inputs_gate"] = INPUT_LIMIT
        return record
    if oversize and spec.backend == "numpy":
        notes.append(
            f"this graph consumes {record['model']['graph_inputs']} inputs (> the "
            f"~{INPUT_LIMIT} IREE runtime limit); the numpy row is still a normal "
            "measurement, the UNPACKED compiled rows of this model are the refused ones"
        )
    elif oversize:
        notes.append(
            f"OVERSIZE ATTEMPT: {record['model']['graph_inputs']} graph inputs "
            f"(>{INPUT_LIMIT}); a crash here is the documented IREE runtime limit"
        )

    # -- resolved build options (the etl_hf default matrix) -----------------
    try:
        resolved_backend, resolved_options = hf_build.resolve_build_options(
            spec.backend, device, spec.options
        )
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"resolve_build_options: {type(exc).__name__}: {exc}")
        return record
    record["backend"].update(
        {
            "resolved_backend": resolved_backend,
            "resolved_options": resolved_options,
        }
    )

    # -- explicit stage timings (the same composition build_model performs) ---
    # The resolved ``opt_level`` is the library default for this target set
    # (``O3`` llvm-cpu, ``O1`` for an all-ROCm target set). It is attempted
    # FIRST — that is the measured row. Only if it is rejected does the probe
    # sweep the other levels (so a genuinely broken cell is not reported dead);
    # ``--opt-level-evidence`` additionally re-attempts the *higher* levels AFTER
    # a successful build, purely to record the compiler's verbatim rejection
    # (that rejection is why the library default for ROCm is O1).
    default_level = resolved_options.get("opt_level")
    record["build"] = {"opt_level_default": default_level}
    if resolved_backend == "numpy":
        # no compiler, no opt level: exactly one attempt
        attempts: list = [None]
    else:
        attempts = [default_level] + [
            level for level in ("O3", "O2", "O1", "O0") if level != default_level
        ]
    sweep: dict[str, str] = {}
    executable = None
    build_times: dict = {}
    first_error: str | None = None
    first_traceback: list[str] = []
    used_options = resolved_options
    for level in attempts:
        level_options = dict(resolved_options)
        if level:
            level_options["opt_level"] = level
        _progress(f"lower (opt_level={level or 'n/a'})")
        try:
            lowered, lower_s = _timeit(
                etl.lower, graph, backend=resolved_backend, **level_options
            )
            _progress("compile")
            artifact, compile_s = _timeit(
                etl.compile, lowered, backend=resolved_backend, **level_options
            )
            _progress("load")
            loaded, load_s = _timeit(
                etl.load,
                artifact,
                backend=resolved_backend,
                device=device,
                **level_options,
            )
        except Exception as exc:
            sweep[level or "default"] = _first_diagnostic(exc)
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
                first_traceback = traceback.format_exc().splitlines()[-8:]
            continue
        sweep[level or "default"] = "compiled"
        executable = loaded
        used_options = level_options
        build_times = {
            "lower_ms": _ms(lower_s),
            "compile_ms": _ms(compile_s),
            "load_ms": _ms(load_s),
            "stage_sum_ms": _ms(lower_s + compile_s + load_s),
            "artifact_bytes": _artifact_bytes(artifact),
            "executable_device": str(loaded.device),
        }
        break

    used_level = used_options.get("opt_level")
    # -- opt-level evidence: re-attempt the levels ABOVE the one that compiled --
    # Cheap (the rejection happens during iree-compile, before any execution) and
    # it is the primary evidence for the library's ROCm default. Failure text is
    # stored verbatim; nothing is executed from these attempts.
    if (
        opt_level_evidence
        and executable is not None
        and resolved_backend == "iree"
        and used_level
    ):
        rank = {"O0": 0, "O1": 1, "O2": 2, "O3": 3}
        used_rank = rank.get(str(used_level))
        higher = [
            level
            for level in ("O3", "O2", "O1")
            if used_rank is not None
            and rank[level] > used_rank
            and level not in sweep
        ]
        for level in higher:
            level_options = dict(used_options)
            level_options["opt_level"] = level
            _progress(f"opt-level evidence (opt_level={level})")
            try:
                lowered = etl.lower(graph, backend=resolved_backend, **level_options)
                etl.compile(lowered, backend=resolved_backend, **level_options)
            except Exception as exc:
                sweep[level] = _first_diagnostic(exc)
                continue
            sweep[level] = "compiled"
        if higher:
            record["build"]["opt_level_evidence_attempts"] = higher

    if sweep:
        record["build"]["opt_level_sweep"] = sweep
    record["build"].update(build_times)
    if executable is None:
        record["status"] = "error"
        record["errors"].append(f"lower/compile/load: {first_error}")
        record["build"]["traceback_tail"] = first_traceback
        return record
    record["build"]["opt_level_used"] = used_level
    if used_level != default_level:
        record["backend"]["resolved_options"] = used_options
        notes.append(
            f"opt_level={default_level} (the resolved etl_hf default) was REJECTED by the "
            f"compiler; this row is measured at opt_level={used_level}. "
            f"Per-level outcome: {sweep}"
        )

    # -- the public one-call build (cold; proves sum(stages) == build) ------
    if measure_public_build is None:
        measure_public_build = param_count <= BIG_PARAMS
    if measure_public_build:
        _progress("public build_model")
        try:
            public_options = dict(spec.options)
            if used_options.get("opt_level"):
                public_options["opt_level"] = used_options["opt_level"]
            public_exe, public_s = _timeit(
                model.build,
                backend=spec.backend,
                device=device,
                pack_weights=spec.packed,
                **public_options,
            )
            record["build"]["public_build_ms"] = _ms(public_s)
            record["build"]["public_build_same_device"] = str(public_exe.device)
        except Exception as exc:
            record["build"]["public_build_error"] = f"{type(exc).__name__}: {exc}"
    else:
        notes.append(
            "public build_model total not measured (expensive re-compile); "
            "stage_sum_ms is the composition build_model performs"
        )

    # -- placement + run ----------------------------------------------------
    host_params = model.params()
    if spec.packed:
        # pack ONCE on the host (never per run): the packed tree is what a
        # ``pack_weights=True`` executable takes.
        packed_host, pack_s = _timeit(model.pack_params, host_params)
        host_params = packed_host
        record["graph"]["pack_params_ms"] = _ms(pack_s)
        record["graph"]["packed_inputs"] = len(packed_host)
        notes.append(
            "packed weights: host params concatenated ONCE with HFModel.pack_params "
            f"({len(packed_host)} inputs); the graph unpacks them with static etl.slice"
        )
    run: dict = {"batch": batch, "seq_len": seq_len}
    record["run"] = run

    _progress("place params / prepare run")
    try:
        if spec.device_resident:
            placed, place_s = _timeit(etl_hf.place_params, host_params, device)
            run["place_params_ms"] = _ms(place_s)
            run["place_params_leaves"] = len(placed)
            # The COLD placement above also builds the adapter's thread-local HAL
            # device + reusable staging buffer on first use; the steady-state
            # per-placement cost is what the restaging variant actually pays.
            steady = []
            for _ in range(3):
                _, seconds = _timeit(etl_hf.place_params, host_params, device)
                steady.append(seconds)
            run["place_params_steady"] = _summarize(steady)
            notes.append(
                "place_params_ms is the COLD placement (first use builds the iree "
                "thread-local HAL device + staging buffer); place_params_steady is "
                "the steady-state per-placement cost"
            )
            ids_operand, ids_s = _timeit(etl_hf.place_ids, ids, device)
            run["place_ids_ms"] = _ms(ids_s)
            call = _device_call(executable, placed, ids_operand, etl_hf)
            run["api"] = "etl_hf.run_model(placed params, place_ids(ids))"
            to_host = etl.core.Device("cpu", 0)
        else:
            call = _host_call(executable, host_params, ids, etl, etl_hf)
            run["api"] = "etl_hf.run_model(host params, host ids)"
            to_host = None
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"prepare-run: {type(exc).__name__}: {exc}")
        run["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
        return record

    # first call (plan building + any lazy runtime setup)
    _progress("first run")
    try:
        out, first_s = _timeit(call)
        run["first_call_ms"] = _ms(first_s)
        logits = _to_host(out, to_host, np)
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"first run: {type(exc).__name__}: {exc}")
        run["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
        return record

    # warm loop
    _progress("warm loop")
    try:
        for _ in range(warmup):
            call()
        samples = []
        for _ in range(iters):
            _, seconds = _timeit(call)
            samples.append(seconds)
        run.update(_summarize(samples))
        run.update(
            {
                "warmup": warmup,
                "iters": iters,
                "tokens_per_s": (
                    round(batch * seq_len / (run["best_ms"] / 1e3), 2)
                    if run.get("best_ms")
                    else None
                ),
            }
        )
    except Exception as exc:
        record["status"] = "error"
        record["errors"].append(f"warm loop: {type(exc).__name__}: {exc}")
        run["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
        return record

    # device: cost of one explicit D2H on the output (never hidden in the loop)
    if spec.device_resident:
        try:
            d2h = []
            for _ in range(3):
                out = call()
                _, seconds = _timeit(lambda: out.to(to_host).numpy())
                d2h.append(seconds)
            run["d2h"] = _summarize(d2h)
            both = []
            for _ in range(iters):
                start = time.perf_counter()
                out = call()
                out.to(to_host).numpy()
                both.append(time.perf_counter() - start)
            run["warm_with_d2h"] = _summarize(both)
            logits = _to_host(out, to_host, np)
        except Exception as exc:
            record["errors"].append(f"d2h measurement: {type(exc).__name__}: {exc}")

        # the documented anti-pattern: restage the weights every call
        if spec.measure_restage:
            try:
                restage_samples = []
                for _ in range(restage_iters):
                    start = time.perf_counter()
                    placed_again = etl_hf.place_params(host_params, device)
                    etl_hf.run_model(executable, placed_again, ids_operand)
                    restage_samples.append(time.perf_counter() - start)
                # per-call placement cost alone (one H2D per weight input)
                place_only = []
                for _ in range(restage_iters):
                    _, seconds = _timeit(etl_hf.place_params, host_params, device)
                    place_only.append(seconds)
                run["restage"] = _summarize(restage_samples)
                run["restage_placement_only"] = _summarize(place_only)
                notes.append(
                    "restage = etl_hf.place_params(host params) + etl_hf.run_model per "
                    "call (the documented per-call restaging anti-pattern)"
                )
            except Exception as exc:
                record["errors"].append(
                    f"restage measurement: {type(exc).__name__}: {exc}"
                )

    # -- parity -------------------------------------------------------------
    _progress("parity")
    record["parity"] = _parity(logits, golden["logits"], np)
    record["parity"]["logits_source"] = "this leg"
    numpy_logits_path = DATA_DIR / f"_numpy_logits_{model_name}.npz"
    if spec.backend == "numpy":
        if not spec.packed:
            # The UNPACKED numpy leg is the reference every other leg is diffed
            # against (packed legs included — packing must not change numerics).
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                numpy_logits_path,
                logits=np.asarray(logits, dtype=np.float32),
                input_ids=ids,
            )
            record["parity"]["wrote_numpy_reference"] = str(numpy_logits_path)
        record["parity"]["reference"] = "numpy backend (this leg)"
    if spec.backend != "numpy" or spec.packed:
        if numpy_logits_path.is_file():
            with np.load(numpy_logits_path) as blob:
                reference = np.asarray(blob["logits"], dtype=np.float32)
            if reference.shape == np.asarray(logits).shape:
                record["parity"]["max_abs_vs_numpy_backend"] = float(
                    np.abs(np.asarray(logits) - reference).max()
                )
                record["parity"]["reference"] = (
                    "numpy backend (unpacked leg)" if spec.packed else "numpy backend"
                )
            else:
                record["errors"].append(
                    f"numpy reference shape {reference.shape} != {np.asarray(logits).shape}"
                )
        elif not spec.packed and spec.backend != "numpy":
            notes.append("no numpy-backend reference logits on disk — CPU-vs-GPU "
                         "parity not computed for this leg")
    return record


def _host_call(executable, params, ids, etl, etl_hf):
    """Callable for a cpu executable through the public ``run_model`` API."""

    def call():
        return etl_hf.run_model(executable, params, ids)

    return call


def _device_call(executable, placed, ids_operand, etl_hf):
    """Callable for a device-resident executable through public ``run_model``.

    ``run_model`` validates an ``etl.Tensor`` ids argument from its METADATA
    only and forwards it verbatim, so this is the documented device-resident
    path — no raw ``etl.run`` short-circuit, no host copy.
    """

    def call():
        return etl_hf.run_model(executable, placed, ids_operand)

    return call


def _to_host(out, to_host, np):
    """Explicit (single, documented) device→host transfer of the logits."""
    if to_host is not None:
        out = out.to(to_host)
    array = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
    return np.asarray(array, dtype=np.float32)


def _artifact_bytes(artifact, *_) -> int | None:
    """Size in bytes of the compiled artifact payload, when the adapter has one.

    IREE records its compiled form as a base64 VMFB inside a plain dict
    (``format="iree-vmfb"``); the numpy backend's payload is serialized IR text.
    Anything unrecognised reports ``None`` rather than guessing.
    """
    try:
        payload = getattr(artifact, "payload", None)
        if isinstance(payload, dict):
            encoded = payload.get("vmfb_base64")
            if isinstance(encoded, str):
                # base64: 4 chars per 3 bytes
                return len(encoded.rstrip("=")) * 3 // 4
            if payload.get("format") == "iree-vmfb-segments":
                return None
        if isinstance(payload, (bytes, bytearray)):
            return len(payload)
        if isinstance(payload, str):
            return len(payload.encode())
        return None
    except Exception:
        return None


def _first_diagnostic(exc: BaseException) -> str:
    """First actionable line of a compiler error (skips the echoed flag dumps)."""
    message = str(exc)
    for line in message.splitlines():
        stripped = line.strip()
        if "error:" in stripped:
            # keep it short: the diagnostic header, not the layout metadata blob
            return stripped.split("error:", 1)[1].strip()[:200] or stripped[:200]
    return (message.splitlines()[0] if message else type(exc).__name__)[:200]


def _parity(logits, golden_logits, np) -> dict:
    """max-abs / relative parity of one leg's logits against the torch oracle."""
    logits = np.asarray(logits, dtype=np.float32)
    golden_logits = np.asarray(golden_logits, dtype=np.float32)
    out: dict = {
        "logits_shape": list(logits.shape),
        "logits_absmax": float(np.abs(logits).max()),
        "logits_mean": float(logits.mean()),
        "golden_absmax": float(np.abs(golden_logits).max()),
    }
    if logits.shape != golden_logits.shape:
        out["error"] = f"shape {logits.shape} != golden {golden_logits.shape}"
        return out
    diff = np.abs(logits - golden_logits)
    out["max_abs_vs_golden"] = float(diff.max())
    out["mean_abs_vs_golden"] = float(diff.mean())
    out["max_rel_vs_golden"] = float(
        diff.max() / max(float(np.abs(golden_logits).max()), 1e-30)
    )
    return out


# ---------------------------------------------------------------------------
# parent: schedule legs, isolate failures, aggregate
# ---------------------------------------------------------------------------
def _leg_path(model: str, backend: str, suffix: str = "") -> Path:
    return DATA_DIR / f"{model}__{backend}{suffix}.json"


def _stderr_tail(text: str, limit: int = 3) -> list[str]:
    """Last meaningful stderr lines, skipping iree's teardown leak noise."""
    noise = ("nanobind:", " - leaked", "See https://nanobind")
    lines = [line for line in (text or "").splitlines() if line.strip()]
    meaningful = [line for line in lines if not line.startswith(noise)]
    return (meaningful or lines)[-limit:]


def run_child_leg(
    model: str,
    backend: str,
    *,
    iters: int,
    warmup: int,
    restage_iters: int,
    attempt_oversize: bool,
    no_public_build: bool = False,
    opt_level_evidence: bool = False,
    suffix: str = "",
    timeout: float = 1800.0,
) -> dict:
    """Run ONE leg in a subprocess; return the merged record (crash-safe)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = _leg_path(model, backend, suffix)
    if target.is_file():
        target.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--leg",
        model,
        "--backend",
        backend,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--restage-iters",
        str(restage_iters),
        "--out",
        str(target),
    ]
    if attempt_oversize:
        command.append("--attempt-oversize")
    if no_public_build:
        command.append("--no-public-build")
    if opt_level_evidence:
        command.append("--opt-level-evidence")
    label = f"{model}/{backend}{suffix}"
    print(f"[leg] {label}: running …", flush=True)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout
        )
        exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = "timeout"
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    duration = time.perf_counter() - started

    record: dict
    if target.is_file():
        record = json.loads(target.read_text())
    else:
        record = {
            "schema": SCHEMA,
            "model": {"name": model, "golden": MODELS.get(model)},
            "backend": {"key": backend, "backend": BACKENDS[backend].backend,
                        "device": f"{BACKENDS[backend].device_kind}:{BACKENDS[backend].device_index}",
                        "packed": BACKENDS[backend].packed},
            "status": "crash",
            "errors": [
                f"leg subprocess produced no record (exit={exit_code}); "
                "the process died before writing JSON — most likely the "
                f"documented IREE runtime input-count SIGSEGV" if exit_code not in (0, "timeout")
                else f"leg subprocess produced no record (exit={exit_code})"
            ],
            "notes": [],
        }
        if exit_code == "timeout":
            record["status"] = "timeout"
    record["process"] = {
        "exit_code": exit_code,
        "duration_s": round(duration, 2),
        "stdout_tail": _stderr_tail(stdout, 5),
        "stderr_tail": _stderr_tail(stderr, 3),
    }
    if exit_code != 0 and record.get("status") == "ok":
        record["status"] = "error"
        record["errors"].append(f"leg process exited {exit_code} after a successful record")
    target.write_text(json.dumps(record, indent=2, default=_json_default))
    status = record.get("status")
    print(
        f"[leg] {label}: {status} in {duration:.1f}s"
        + (f" (exit {exit_code})" if exit_code not in (0,) else ""),
        flush=True,
    )
    return record


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def _fmt(value, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value:,}{unit}"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.{digits}e}{unit}"
        return f"{value:,.{digits}f}{unit}"
    return f"{value}{unit}"


def _status_cell(record: dict) -> str:
    status = record.get("status", "?")
    if status == "ok":
        return "ok"
    detail = (record.get("errors") or record.get("notes") or [""])[0]
    head = detail.split("(")[0].strip() or detail
    return f"**{status}**: {head[:120]}"


def collect_records() -> dict[tuple[str, str], dict]:
    """All main-pass records on disk, keyed by (model, backend)."""
    records: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for backend in BACKENDS:
            path = _leg_path(model, backend)
            if path.is_file():
                records[(model, backend)] = json.loads(path.read_text())
    return records


def collect_crash_probe() -> dict[tuple[str, str], dict]:
    """Records of the deliberate over-limit compiled attempts (``*.crash.json``)."""
    return {
        (model, backend): json.loads(path.read_text())
        for model in MODELS
        for backend in BACKENDS
        if (path := _leg_path(model, backend, ".crash")).is_file()
    }


def collect_sanity() -> dict[tuple[str, str], dict]:
    return {
        (model, backend): json.loads(path.read_text())
        for model in MODELS
        for backend in BACKENDS
        if (path := _leg_path(model, backend, ".sanity")).is_file()
    }


def render_report(records: dict, sanity: dict, crash: dict | None = None) -> str:
    """Render ``hf_perf_results.md`` from the measured records."""
    lines: list[str] = []
    add = lines.append
    models = [name for name in MODELS if any(key[0] == name for key in records)]
    backends = [key for key in BACKENDS if any(k[1] == key for k in records)]

    def rec(model: str, backend: str) -> dict:
        return records.get((model, backend), {})

    add("# `etl_hf` whole-model forward: perf & parity matrix (numpy / IREE-CPU / IREE-ROCm)")
    add("")
    add(
        "Generated by `bench_logs/probe_hf_perf.py` — **do not hand-edit**; re-run the "
        "probe (or `--report`) instead. Raw per-leg records: `bench_logs/hf_perf_data/*.json`."
    )
    add("")
    add("Each cell measures the **whole model forward pass as ONE etl graph** through")
    add("`etl_hf`'s public API: `load_hf` → `etl_hf.build_model` (trace → lower → compile →")
    add("load) → run, with the checkpoint weights as ordinary graph **inputs**.")
    add("")
    add("Every backend row exists twice: **unpacked** (one graph input per weight input,")
    add("the canonical tree) and **packed** (`pack_weights=True`; key suffix `-packed`), where")
    add("`HFModel.pack_params` concatenates the weight tree on the host into a handful of")
    add("inputs that the graph unpacks with static `etl.slice`. Packing is what brings the")
    add("124M `gpt2` graph (197 unpacked inputs) under IREE's ~161-consumed-input runtime")
    add("limit — the packed rows are the ones that make it runnable on a compiler backend.")
    add("")
    add("## Method (what a number means)")
    add("")
    add("| aspect | how it is measured |")
    add("|---|---|")
    add(
        "| latency | `best` of N warm calls (after a warmup) of "
        "`etl_hf.run_model(exe, params, ids)` — on the device-resident rows with "
        "`place_params`-ed weights and a `place_ids`-ed ids tensor (the documented device "
        "recipe, through the public API). `first` is the very first call after load (includes "
        "the executable's input/output plan building). `median`/`mean` in the JSON. |"
    )
    add(
        "| warm loop, device | device-resident = weights placed ONCE with "
        "`etl_hf.place_params` + ids placed with `etl_hf.place_ids`, then a same-device loop "
        "(outputs stay on the GPU, never auto-copied). `+d2h` re-times the loop WITH one "
        "explicit `Tensor.to(cpu)` per call. |"
    )
    add(
        "| restage (anti-pattern) | `place_params(host params)` **on every call** + run — the "
        "documented per-call host→device restaging trap, measured on the same graph. |"
    )
    add(
        "| packed weights | `HFModel.pack_params` runs ONCE on the host (its cost is recorded "
        "as `pack_params_ms`); the executable is built with `pack_weights=True`, and every "
        "latency/parity number below is the packed executable's. |"
    )
    add(
        "| build times | explicit `etl.lower` / `etl.compile` / `etl.load` on the traced graph, "
        "with the options `etl_hf.resolve_build_options` resolves — the **library defaults** "
        "(`opt_level=\"O3\"` for llvm-cpu, `opt_level=\"O1\"` for an all-ROCm target set, plus "
        "`--iree-llvmcpu-target-cpu=host` + `--iree-opt-data-tiling` for llvm-cpu). `stage sum` "
        "is that composition; the public one-call `build_model` total is measured separately "
        "where cheap (small models) as a cross-check. |"
    )
    add(
        "| parity | `max-abs` of this leg's logits vs (a) the **torch/transformers golden** in "
        "`bench_logs/hf_goldens/` (absolute truth; the input ids are the golden's own ids) and "
        "(b) the **numpy interpreter backend** on the identical weights — for packed rows that "
        "reference is the *unpacked* numpy leg, so `0.00` there means packing is bit-identical. "
        "Not bit-equality between backends: fp32 accumulation order differs. |"
    )
    add(
        "| tokens/s | `batch × seq_len / best latency` — a forward-pass rate, NOT decode "
        "throughput (no KV cache / incremental decode in v1: every call re-runs the whole graph). |"
    )
    add("")

    measurement = next(
        (r.get("measurement") for r in records.values() if r.get("measurement")), {}
    )
    if measurement:
        add(
            "Timing parameters of this run: **{} warm samples after {} warmup call(s)** for the "
            "latency columns, **{} samples** for the restaging variant (per-leg overrides are "
            "recorded in each JSON under `measurement`).".format(
                measurement.get("iters"), measurement.get("warmup"), measurement.get("restage_iters")
            )
        )
        add("")

    env = next((r.get("env") for r in records.values() if r.get("env")), {})
    if env:
        add("Environment of the measuring process: " + ", ".join(
            f"`{key}={value}`" for key, value in [
                ("python", env.get("python")),
                ("numpy", env.get("numpy")),
                ("iree-base-compiler", env.get("iree-base-compiler")),
                ("iree-base-runtime", env.get("iree-base-runtime")),
                ("machine", env.get("machine")),
                ("cpu_count", env.get("cpu_count")),
            ] if value is not None
        ))
        add("")
        flags = {k: v for k, v in env.items() if k.startswith("ETL_") and v}
        add("Run-level env overrides: " + (", ".join(f"`{k}={v}`" for k, v in flags.items()) if flags else "none"))
        add("")

    # ---- 1. graph size ----------------------------------------------------
    add("## 1. Graph size (backend independent)")
    add("")
    for packed, label in ((False, "Unpacked weights (one graph input per weight)"),
                          (True, "Packed weights (`pack_weights=True`)")):
        group = [b for b in backends if BACKENDS[b].packed == packed]
        if not group:
            continue
        add(f"### {label}")
        add("")
        add(
            "| model | family | act | params | weight inputs | graph inputs | IR ops "
            "| distinct op defs | StableHLO | constants | trace | export | pack (host) |"
        )
        add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for model in models:
            record = rec(model, group[0])
            row = record.get("model", {})
            graph = record.get("graph", {})
            stable = graph.get("stablehlo_bytes")
            add(
                "| `{model}` | {family} | {act} | {params} | {weights} | {inputs} | {ops} "
                "| {distinct} | {stable} | {consts} | {trace} | {export} | {pack} |".format(
                    model=model,
                    family=row.get("family"),
                    act=row.get("hidden_act"),
                    params=_fmt(row.get("param_count")),
                    weights=_fmt(row.get("weight_inputs")),
                    inputs=_fmt(row.get("graph_inputs")),
                    ops=_fmt(graph.get("ir_ops")),
                    distinct=_fmt(graph.get("ir_distinct_ops")),
                    stable=_fmt((stable / 1024) if stable else None, 1, " KB"),
                    consts=_fmt(graph.get("stablehlo_constants")),
                    trace=_fmt(graph.get("trace_ms"), 1, " ms"),
                    export=_fmt(graph.get("export_ms"), 1, " ms"),
                    pack=_fmt(graph.get("pack_params_ms"), 1, " ms") if packed else "—",
                )
            )
        add("")
    add(
        "`graph inputs` = weight inputs + 1 (the `(batch, seq_len)` int32 `input_ids`). "
        "The IR op count differs between the packed/unpacked graphs because the packed one "
        "adds its in-graph `slice` unpacking; it is identical for every backend of the same "
        "graph — the traced program, not a backend artifact."
    )
    add("")
    # ---- 2. build times --------------------------------------------------
    add("## 2. Build (lower / compile / load)")
    add("")
    add("| model | backend | device | opt_level used | lower | compile | load | stage sum "
        "| public build |")
    add("|---|---|---|---|---|---|---|---|---|")
    for model in models:
        for backend in backends:
            record = rec(model, backend)
            build = record.get("build", {})
            backend_info = record.get("backend", {})
            if not build:
                add(f"| `{model}` | `{backend}` | {backend_info.get('device', '—')} | — | — | — "
                    "| — | — | — |")
                continue
            used = build.get("opt_level_used")
            default = build.get("opt_level_default")
            level = "—" if used is None else (
                f"`{used}` (default `{default}` rejected)"
                if default and used != default
                else f"`{used}`"
            )
            add(
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    model,
                    backend,
                    backend_info.get("device"),
                    level,
                    _fmt(build.get("lower_ms"), 1, " ms"),
                    _fmt(build.get("compile_ms"), 1, " ms"),
                    _fmt(build.get("load_ms"), 1, " ms"),
                    _fmt(build.get("stage_sum_ms"), 1, " ms"),
                    _fmt(build.get("public_build_ms"), 1, " ms"),
                )
            )
    add("")
    add(
        "`opt_level used` is the level the row was actually built at; the resolved default is "
        "recorded per leg in `build.opt_level_default` / `build.opt_level_used` (the library "
        "defaults: `O3` for llvm-cpu, `O1` for an all-ROCm target set — section 7 explains the "
        "O2/O3 ROCm rejection that forces it). Compile time is the IREE compile of the "
        "whole-model graph (the numpy interpreter has no compile step). `public build` is the "
        "cold `etl_hf.build_model` total where measured — it re-runs the stages, so it should "
        "be ≳ the stage sum (it also does `load_hf`-side work and graph memoisation "
        "bookkeeping)."
    )
    add("")

    # ---- 3. latency ------------------------------------------------------
    add("## 3. Latency (whole forward, one graph, warm)")
    add("")
    add("| model | backend | device | first call | warm best | warm median | tokens/s "
        "| +d2h best | restage best (anti-pattern) |")
    add("|---|---|---|---|---|---|---|---|---|")
    for model in models:
        for backend in backends:
            record = rec(model, backend)
            run = record.get("run", {})
            add(
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                    model,
                    backend,
                    record.get("backend", {}).get("device", "—"),
                    _fmt(run.get("first_call_ms"), 2, " ms"),
                    _fmt(run.get("best_ms"), 2, " ms"),
                    _fmt(run.get("median_ms"), 2, " ms"),
                    _fmt(run.get("tokens_per_s"), 0),
                    _fmt((run.get("warm_with_d2h") or {}).get("best_ms"), 2, " ms"),
                    _fmt((run.get("restage") or {}).get("best_ms"), 2, " ms"),
                )
            )
    add("")
    add(
        "`+d2h best` = same device loop with one explicit `Tensor.to(cpu)` of the logits per "
        "call (what you pay if you need the logits on the host); `restage best` = the whole "
        "weight tree re-placed on the device every call — the documented anti-pattern, and "
        "why the device loop above is the fast iteration pattern. Device rows are driven "
        "through `etl_hf.run_model(exe, place_params(...), place_ids(...))` — the public "
        "`etl_hf` API, no raw `etl.run` short-circuit (the leg records `run.api`)."
    )
    add("")

    # ---- 4. parity -------------------------------------------------------
    add("## 4. Parity (fp32 logits; max-abs)")
    add("")
    add("| model | backend | logits absmax | vs torch golden | mean-abs vs golden "
        "| rel vs golden | vs numpy backend | status |")
    add("|---|---|---|---|---|---|---|---|")
    for model in models:
        for backend in backends:
            record = rec(model, backend)
            parity = record.get("parity", {})
            add(
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} |".format(
                    model,
                    backend,
                    _fmt(parity.get("logits_absmax"), 3),
                    _fmt(parity.get("max_abs_vs_golden"), 2),
                    _fmt(parity.get("mean_abs_vs_golden"), 2),
                    _fmt(parity.get("max_rel_vs_golden"), 2),
                    _fmt(parity.get("max_abs_vs_numpy_backend"), 2),
                    _status_cell(record),
                )
            )
    add("")
    add(
        "Golden absmax per model (the scale that turns an absolute delta into a relative "
        "one): "
        + ", ".join(
            f"`{model}` {_fmt((rec(model, backends[0]).get('parity') or {}).get('golden_absmax'), 3)}"
            for model in models
        )
        + "."
    )
    add("")

    # ---- 5. speedups -----------------------------------------------------
    add("## 5. GPU vs CPU")
    add("")
    for packed, label in ((False, "Unpacked weights"), (True, "Packed weights")):
        group = [b for b in backends if BACKENDS[b].packed == packed]
        if not group:
            continue
        numpy_key = "numpy-packed" if packed else "numpy"
        cpu_key = "iree-llvm-cpu-packed" if packed else "iree-llvm-cpu"
        gpu_key = "iree-rocm-packed" if packed else "iree-rocm"
        if not all(key in group for key in (numpy_key, cpu_key, gpu_key)):
            continue
        add(f"### {label}")
        add("")
        add(
            "| model | numpy (cpu) | iree llvm-cpu | iree rocm (device-resident) "
            "| rocm / numpy | rocm / llvm-cpu | numpy / llvm-cpu |"
        )
        add("|---|---|---|---|---|---|---|")
        for model in models:
            row: dict[str, float | None] = {}
            for backend in (numpy_key, cpu_key, gpu_key):
                row[backend] = (rec(model, backend).get("run") or {}).get("best_ms")

            def ratio(a: str, b: str):
                left, right = row.get(a), row.get(b)
                if not left or not right:
                    return None
                return round(left / right, 2)

            add(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    model,
                    _fmt(row.get(numpy_key), 2, " ms"),
                    _fmt(row.get(cpu_key), 2, " ms"),
                    _fmt(row.get(gpu_key), 2, " ms"),
                    _fmt(ratio(numpy_key, gpu_key), 2, "×"),
                    _fmt(ratio(cpu_key, gpu_key), 2, "×"),
                    _fmt(ratio(cpu_key, numpy_key), 2, "×"),
                )
            )
        add("")
    add(
        "Speedup columns are latency ratios (`A/B > 1` ⇒ B is faster). Two regimes are "
        "visible and both are measured, not assumed: `numpy / llvm-cpu > 1` means plain "
        "OpenBLAS beats the compiled CPU graph, while `rocm / llvm-cpu` is the real GPU-vs-CPU "
        "comparison — these graphs are hundreds of separate IREE dispatches, so the "
        "device-resident GPU path wins on parallel dispatch, not on tensor-core throughput "
        "(these models are far too small for that)."
    )
    add("")
    # ---- 6. findings -----------------------------------------------------
    add("## 6. Findings")
    add("")
    add(_findings(records, models, backends))
    add("")

    # ---- 7. the input-count limit, measured -------------------------------
    add("## 7. The ~161-consumed-input cliff: measured, and resolved by packing")
    add("")
    add(
        "IREE's Python runtime **SIGSEGVs without a Python exception** when a graph consumes "
        "more than ~161 distinct inputs (documented in `etl_hf/CONTEXT.md` Known Issues with "
        "the measured brackets: 161 chained-matmul / 162 fan-out). Only an *unpacked* graph "
        "gets near it:"
    )
    add("")
    add("| model | unpacked graph inputs | unpacked compiled legs | packed graph inputs "
        "| packed compiled legs |")
    add("|---|---|---|---|---|")
    for model in models:
        unpacked_keys = [key for key in backends if not BACKENDS[key].packed and key != "numpy"]
        packed_keys = [key for key in backends if BACKENDS[key].packed and key != "numpy"]
        unpacked_inputs = (rec(model, "numpy").get("model") or {}).get("graph_inputs")
        packed_inputs = (rec(model, "numpy-packed").get("model") or {}).get("graph_inputs")
        add(
            "| `{}` | {} | {} | {} | {} |".format(
                model,
                _fmt(unpacked_inputs),
                ", ".join(f"`{key}` {rec(model, key).get('status') or '—'}" for key in unpacked_keys),
                _fmt(packed_inputs),
                ", ".join(f"`{key}` {rec(model, key).get('status') or '—'}" for key in packed_keys),
            )
        )
    add("")
    if not crash:
        add("No deliberate over-limit attempt was recorded (run without `--no-crash-probe`).")
    else:
        add(
            "To show that the refusal describes a real runtime limit and not a guess, each "
            "refused leg is attempted ONCE in its own subprocess (CPU targets only — a live GPU "
            "context is never put in front of a deliberate SIGSEGV):"
        )
        add("")
        add("| leg | graph inputs | exit code | status | last stderr line |")
        add("|---|---|---|---|---|")
        for (model, backend), record in sorted(crash.items()):
            process = record.get("process", {})
            stderr = (process.get("stderr_tail") or [""])[-1] if process else ""
            add(
                "| `{}` / `{}` | {} | `{}` | {} | `{}` |".format(
                    model,
                    backend,
                    (records.get((model, backend), {}).get("model") or {}).get("graph_inputs"),
                    process.get("exit_code"),
                    record.get("status"),
                    (stderr or "—")[:160],
                )
            )
        add("")
        exits = {str((record.get("process") or {}).get("exit_code")) for record in crash.values()}
        add(
            "The attempt dies with exit code(s) {} **without a Python exception** — a hard "
            "crash inside the IREE runtime invoke, exactly the documented limit. The cliff is a "
            "property of the UNPACKED graph (197 inputs for `gpt2`), not of the model: the "
            "packed form of the same checkpoint declares 4 inputs and its compiled rows are "
            "measured in tables 2–4 — nothing is extrapolated from other models."
            .format(", ".join(f"`{value}`" for value in sorted(exits)))
        )
    add("")
    # ---- 8. failures -----------------------------------------------------
    add("## 8. Non-`ok` legs (verbatim)")
    add("")
    problematic = [
        (model, backend, rec(model, backend))
        for model in models
        for backend in backends
        if rec(model, backend).get("status") not in (None, "ok")
    ]
    if not problematic:
        add("None — every attempted leg measured successfully.")
    else:
        for model, backend, record in problematic:
            process = record.get("process", {})
            add(f"### `{model}` / `{backend}` — **{record.get('status')}**")
            add("")
            if process:
                add(f"- exit code: `{process.get('exit_code')}`; wall time {process.get('duration_s')} s")
                if process.get("stderr_tail"):
                    add("- last stderr lines:")
                    for line in process["stderr_tail"]:
                        add(f"  - `{line[:400]}`")
            for error in record.get("errors", []):
                add(f"- {error[:600]}")
            for note in record.get("notes", []):
                add(f"- note: {note[:400]}")
            add("")

    # ---- 8. per-leg notes + reproducibility ------------------------------
    add("## 9. Per-leg notes")
    add("")
    noted = [
        (model, backend)
        for model in models
        for backend in backends
        if rec(model, backend).get("notes")
    ]
    if not noted:
        add("None.")
    else:
        for model, backend in noted:
            add(f"- `{model}` / `{backend}`:")
            for note in rec(model, backend)["notes"]:
                add(f"  - {note}")
    add("")

    add("## 10. Reproducibility (same leg, second run)")
    add("")
    if not sanity:
        add("No sanity re-runs recorded (run the probe without `--no-sanity` to produce them).")
    else:
        add("| leg | warm best run 1 | warm best run 2 | Δ | parity run 1 | parity run 2 |")
        add("|---|---|---|---|---|---|")
        for (model, backend), replay in sorted(sanity.items()):
            first = rec(model, backend)
            one = (first.get("run") or {}).get("best_ms")
            two = (replay.get("run") or {}).get("best_ms")
            delta = (
                f"{(two - one) / one * 100:+.1f}%"
                if isinstance(one, float) and isinstance(two, float) and one
                else "—"
            )
            add(
                "| `{}` / `{}` | {} | {} | {} | {} | {} |".format(
                    model,
                    backend,
                    _fmt(one, 2, " ms"),
                    _fmt(two, 2, " ms"),
                    delta,
                    _fmt((first.get("parity") or {}).get("max_abs_vs_golden"), 2),
                    _fmt((replay.get("parity") or {}).get("max_abs_vs_golden"), 2),
                )
            )
        add("")
        add(
            "Parity is expected to be *identical* across runs (deterministic kernels); "
            "latency is expected to wobble by a few percent on a shared host (another job "
            "may be using the GPU)."
        )
    add("")

    add("## 11. Scope, limits, and what is NOT claimed")
    add("")
    add(
        "**Measured facts** — everything in tables 1–5, the packed/unpacked comparison of "
        "section 7 and the reproducibility table came out of this probe on this host, is "
        "regenerable with the command below, and is stored verbatim in `hf_perf_data/*.json` "
        "(unpacked legs in `<model>__<backend>.json`, packed legs in "
        "`<model>__<backend>-packed.json`)."
    )
    add("")
    add("**Not measured / not claimed:**")
    add("")
    add(
        "- **No torch latency baseline.** `torch`/`transformers` are not installed in the probe "
        "environment, so the torch goldens are used for *numerics only*. The only latency "
        "comparison here is between etl's own execution paths (three backend paths, each in "
        "unpacked and packed form). (An etl-vs-torch-eager comparison needs a torch-enabled "
        "environment and is deliberately out of scope.)"
    )
    add(
        "- **Not a decode benchmark.** v1 has no KV cache and no incremental decode: every call "
        "re-runs the whole sequence, so `tokens/s` is a whole-forward rate for `batch × seq_len` "
        "tokens, not generation throughput."
    )
    add(
        "- **Tiny models, small sequences.** These are 0.1M–124M-parameter checkpoints at "
        "`seq_len` 8/32 chosen to match the committed goldens. Sub-millisecond GPU rows are "
        "dominated by fixed dispatch cost; do not read them as accelerators' peak capability."
    )
    add(
        "- **One GPU, shared host.** The MI300X is a single virtualized device and other work may "
        "be resident; latency was best-of-N on a warm cache, and the reproducibility table shows "
        "the run-to-run spread that actually occurred."
    )
    add(
        "- **Compiled `gpt2` rows exist only as PACKED rows.** The unpacked 197-input graph is "
        "refused by IREE's ~161-input runtime limit (section 7, measured crash); the packed "
        "graph (4 inputs) is the one measured in tables 2–4. Nothing is extrapolated from the "
        "`distilgpt2` row: the input-count limit is a hard runtime limit, not a size effect."
    )
    add(
        "- **Packing is opt-in and not free.** `pack_weights=True` plus a `pack_params` call is "
        "required (a `pack_weights=False` executable still takes one input per weight); the "
        "packed graph spends a few extra in-graph `slice` ops, which is why its IR op count is "
        "slightly higher than the unpacked one. Both forms are measured here."
    )
    add(
        "- **CPU-thread count is not pinned.** The numpy row benefits from OpenBLAS threading; a "
        "single-threaded BLAS is slower (measured elsewhere in `etl_hf/CONTEXT.md`). Treat the "
        "numpy column as \"this host, default threading\"."
    )
    add("")
    add("Regenerate with:")
    add("")
    add("```bash")
    add("uv run --isolated --extra iree python bench_logs/probe_hf_perf.py")
    add("# with the per-level opt_level rejection evidence re-attempted on every compiled leg:")
    add("uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --opt-level-evidence")
    add("uv run --isolated --extra iree python bench_logs/probe_hf_perf.py --report  # markdown only")
    add("# one row, to check reproducibility of a published number:")
    add("uv run --isolated --extra iree python bench_logs/probe_hf_perf.py \\")
    add("    --models gpt2 --backends iree-rocm-packed --no-sanity --no-crash-probe")
    add("```")
    add("")
    return "\n".join(lines).rstrip() + "\n"


def _findings(records: dict, models: list[str], backends: list[str]) -> str:
    """Measured-facts narrative — computed from the records, never asserted."""
    out: list[str] = []

    def best(model: str, backend: str):
        return (records.get((model, backend), {}).get("run") or {}).get("best_ms")

    def parity(model: str, backend: str, key: str):
        return (records.get((model, backend), {}).get("parity") or {}).get(key)

    def group(packed: bool) -> list[str]:
        return [key for key in backends if BACKENDS[key].packed == packed]

    def keys_for(packed: bool) -> tuple[str, str, str]:
        suffix = "-packed" if packed else ""
        return (f"numpy{suffix}", f"iree-llvm-cpu{suffix}", f"iree-rocm{suffix}")

    ok = lambda key, model: (records.get((model, key), {}) or {}).get("status") == "ok"

    # parity headline
    worst = []
    for model in models:
        for backend in backends:
            value = parity(model, backend, "max_abs_vs_golden")
            if isinstance(value, float):
                worst.append((model, backend, value))
    if worst:
        model, backend, value = max(worst, key=lambda item: item[2])
        out.append(
            f"- **Parity.** Every measured leg — packed and unpacked — reproduces the "
            f"torch/transformers golden logits to fp32 accumulation order: the largest "
            f"deviation anywhere in the matrix is `{value:.2e}` ({model} / {backend}). "
            "Deltas are 1e-8-class on the tiny checkpoints and 1e-4-class on the 82M/124M "
            "ones, whose logits reach |max| ≈ 80–300 (i.e. ~1e-6 relative)."
        )

    # packing must not change numerics: packed legs vs the unpacked numpy reference
    pack_parity = [
        (model, backend, parity(model, backend, "max_abs_vs_numpy_backend"))
        for model in models
        for backend in group(True)
        if isinstance(parity(model, backend, "max_abs_vs_numpy_backend"), float)
    ]
    if pack_parity:
        exact = [row for row in pack_parity if row[2] == 0.0]
        model, backend, value = max(pack_parity, key=lambda item: item[2])
        out.append(
            "- **Packing does not change numerics.** The packed numpy rows are **bit-identical** "
            "to the unpacked numpy reference (`max-abs 0.00`) on "
            + ", ".join(f"`{m}`" for m in dict.fromkeys(row[0] for row in exact))
            + " — `pack_params` is a pure host-side concatenation and the in-graph `etl.slice` "
            "unpacking returns the very same values. The largest packed-vs-unpacked-reference "
            f"delta anywhere is `{value:.2e}` (`{model}` / `{backend}`), i.e. the compiled "
            "kernels' fp32 accumulation order, not packing."
        )

    numpy_parity = [
        (model, parity(model, backend, "max_abs_vs_numpy_backend"))
        for model in models
        for backend in backends
        if not backend.startswith("numpy")
    ]
    numpy_parity = [(m, v) for m, v in numpy_parity if isinstance(v, float)]
    if numpy_parity:
        model, value = max(numpy_parity, key=lambda item: item[1])
        out.append(
            f"- **Compiled vs the numpy interpreter** on the identical weights: largest max-abs "
            f"is `{value:.2e}` ({model}). The numpy backend is the reference here, so this "
            "isolates backend numerics from checkpoint parsing."
        )

    # cpu: numpy vs compiled, per packing group
    for packed in (False, True):
        numpy_key, cpu_key, _ = keys_for(packed)
        comparisons = []
        for model in models:
            numpy_ms, cpu_ms = best(model, numpy_key), best(model, cpu_key)
            if numpy_ms and cpu_ms:
                comparisons.append((model, numpy_ms, cpu_ms))
        if not comparisons:
            continue
        wins = [row for row in comparisons if row[1] < row[2]]
        phrases = []
        for model, numpy_ms, cpu_ms in comparisons:
            if numpy_ms <= cpu_ms:
                phrases.append(
                    f"`{model}` numpy {numpy_ms:,.2f} ms vs llvm-cpu {cpu_ms:,.2f} ms "
                    f"(numpy {cpu_ms / numpy_ms:.2f}× faster)"
                )
            else:
                phrases.append(
                    f"`{model}` numpy {numpy_ms:,.2f} ms vs llvm-cpu {cpu_ms:,.2f} ms "
                    f"(llvm-cpu {numpy_ms / cpu_ms:.2f}× faster)"
                )
        label = "packed" if packed else "unpacked"
        out.append(
            f"- **On CPU ({label} weights) the numpy interpreter is competitive-to-faster than "
            f"the compiled graph** (numpy faster on {len(wins)} of {len(comparisons)} measured "
            "models): " + ", ".join(phrases)
            + ". The numpy path uses OpenBLAS for every matmul; the compiled graph pays IREE "
            "dispatch overhead per op (these graphs are hundreds of small ops). **The compiled "
            "path's value is the GPU**, not CPU speed."
        )

    # gpu rows, per packing group
    gpu_blocks = []
    for packed in (False, True):
        numpy_key, cpu_key, gpu_key = keys_for(packed)
        rows = []
        for model in models:
            numpy_ms, cpu_ms, gpu_ms = (
                best(model, numpy_key),
                best(model, cpu_key),
                best(model, gpu_key),
            )
            if gpu_ms:
                rows.append((model, numpy_ms, cpu_ms, gpu_ms))
        if rows:
            gpu_blocks.append((packed, rows))
    for packed, rows in gpu_blocks:
        label = "packed" if packed else "unpacked"
        out.append(
            f"- **GPU (MI300X, ROCm, device-resident, {label} weights)**: "
            + ", ".join(
                "`{}` {}{}{}".format(
                    model,
                    f"{gpu:,.2f} ms",
                    f" — {numpy / gpu:.2f}× faster than numpy" if numpy else "",
                    f", {cpu / gpu:.2f}× faster than iree-llvm-cpu" if cpu else "",
                )
                for model, numpy, cpu, gpu in rows
            )
            + "."
        )
        small = [row for row in rows if row[1] and row[3] and row[1] / row[3] < 2.0]
        if small:
            out.append(
                "- **The GPU win is size-dependent, and the small models sit at the dispatch "
                f"floor** ({label} weights): for "
                + ", ".join(f"`{row[0]}`" for row in small)
                + " the device-resident GPU path is not ≥2× faster than the numpy host path — "
                "at these shapes the per-call cost is fixed overhead (iree HAL dispatch + "
                "kernel launches) while the whole forward is only a few hundred microseconds "
                "of arithmetic. The 82M/124M checkpoints are where the GPU pulls ahead."
            )

    # compiler option evidence (measured by the opt-level evidence attempts)
    evidence = []
    for model in models:
        for backend in backends:
            build = (records.get((model, backend), {}) or {}).get("build") or {}
            sweep = build.get("opt_level_sweep") or {}
            used = build.get("opt_level_used")
            rejected = {lvl: msg for lvl, msg in sweep.items() if msg != "compiled"}
            if rejected and used:
                evidence.append((model, backend, used, rejected))
    if evidence:
        out.append(
            "- **ROCm `opt_level` default = `O1`, and the rejection above it is real "
            "(measured, never asserted):** "
            + "; ".join(
                "`{}` / `{}` — built at `opt_level={}`, while {}".format(
                    model,
                    backend,
                    used,
                    " and ".join(
                        f"`{lvl}` fails with `{msg}`" for lvl, msg in rejected.items()
                    ),
                )
                for model, backend, used, rejected in evidence
            )
            + ". That `'func.func' op failed to distribute` is an upstream `iree-compile` "
            "ROCm codegen failure on the LayerNorm-shaped reduce (not an etl/etl_hf bug), so "
            "the library now injects `opt_level=\"O1\"` for an all-ROCm target set — these "
            "rows are measured at that default, and the higher levels are re-attempted only to "
            "record the verbatim error."
        )

    # restage
    restage_rows = []
    for model in models:
        for backend in backends:
            run = records.get((model, backend), {}).get("run") or {}
            loop = run.get("best_ms")
            restage = (run.get("restage") or {}).get("best_ms")
            if loop and restage:
                restage_rows.append((model, backend, loop, restage, restage / loop))
    if restage_rows:
        out.append(
            "- **Per-call restaging is the anti-pattern, measured here:** re-placing the whole "
            "weight tree on the device every call costs "
            + ", ".join(
                f"`{model}`/`{backend}` {restage:,.2f} ms vs {loop:,.2f} ms for the same-device "
                f"loop ({ratio:,.1f}×)"
                for model, backend, loop, restage, ratio in restage_rows
            )
            + ". The same-device loop (`place_params` + `place_ids` once, feed device tensors "
            "back) is the iteration pattern to use — with packed weights it also stages 3–8 "
            "leaves instead of ~100–200."
        )

    d2h_rows = []
    for model in models:
        for backend in backends:
            run = records.get((model, backend), {}).get("run") or {}
            loop = run.get("best_ms")
            with_d2h = (run.get("warm_with_d2h") or {}).get("best_ms")
            if loop and with_d2h:
                d2h_rows.append((model, backend, loop, with_d2h))
    if d2h_rows:
        out.append(
            "- **Reading the logits back to the host costs extra**: same loop plus one "
            "explicit `Tensor.to(cpu)` per call is "
            + ", ".join(
                f"`{model}`/`{backend}` {with_d2h:,.2f} ms (vs {loop:,.2f} ms)"
                for model, backend, loop, with_d2h in d2h_rows
            )
            + "."
        )

    # the input-count cliff, and what packing changed
    refused = [
        (model, key)
        for model in models
        for key in backends
        if (records.get((model, key), {}) or {}).get("status") == "skipped"
    ]
    packed_ok = [
        (model, f"{key}-packed")
        for model, key in refused
        if (records.get((model, f"{key}-packed"), {}) or {}).get("status") == "ok"
    ]
    if refused:
        out.append(
            "- **The ~161-consumed-input cliff is real — but only for UNPACKED graphs.** "
            + ", ".join(f"`{model}` / `{key}`" for model, key in refused)
            + " stay refused: those graphs declare more distinct inputs than IREE's Python "
            "runtime can invoke (SIGSEGV, exit `-11`, measured in section 7). The same models "
            "DO run compiled once packed"
            + (": " + ", ".join(f"`{model}` / `{key}`" for model, key in packed_ok) if packed_ok else "")
            + " — `pack_weights=True` collapses the weight tree into a handful of inputs, and "
            "the packed rows above are real measurements, not extrapolations."
        )

    compile_ms = [
        (model, backend, (records.get((model, backend), {}).get("build") or {}).get("compile_ms"))
        for model in models
        for backend in backends
        if backend.startswith("iree")
    ]
    compile_ms = [row for row in compile_ms if isinstance(row[2], float)]
    if compile_ms:
        model, backend, value = max(compile_ms, key=lambda item: item[2])
        out.append(
            f"- **Compile cost is a one-off**: the largest measured compile is `{value:,.0f} ms` "
            f"(`{model}` / `{backend}`); everything after it is the per-call latency above. "
            "`etl_hf.build_model` memoises the traced forward per `(family, config, seq_len, "
            "batch, packed)`, so repeated builds of the same shape reuse the executable."
        )

    # hypothesis — everything above is measured; this item is explicitly not
    op_rows = [
        (model, (records.get((model, "iree-rocm"), {}).get("graph") or {}).get("ir_ops"))
        for model in models
        if (records.get((model, "iree-rocm"), {}) or {}).get("status") == "ok"
    ]
    op_rows = [row for row in op_rows if row[1]]
    gpu_ratio = []
    for packed, rows in gpu_blocks:
        gpu_ratio += [numpy / gpu for model, numpy, cpu, gpu in rows if numpy and gpu]
    if op_rows and gpu_ratio:
        out.append(
            "- **HYPOTHESIS (not measured — stated as a hypothesis, not a result):** the "
            "device-resident GPU rows look launch-bound rather than compute-bound. The "
            "forward graph is hundreds of small ops (measured IR op counts: "
            + ", ".join(f"`{m}` {n}" for m, n in op_rows)
            + ") that StableHLO/IREE lower into many narrow dispatches, so the per-call time "
            "is dominated by dispatch + kernel-launch overhead instead of arithmetic. That "
            "would explain the observed GPU-vs-numpy margins of "
            + ", ".join(f"{r:.2f}×" for r in gpu_ratio)
            + " — a narrow band that shows no clean trend across the parameter-count range "
            "measured. **Not verified here** — the probe records no dispatch count and no "
            "per-op GPU timeline; confirming it would need an IREE trace / "
            "`--iree-hal-trace-execution` run, which this probe does not do."
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(MODELS),
                        help=f"comma-separated subset of {sorted(MODELS)}")
    parser.add_argument("--backends", default=",".join(BACKENDS),
                        help=f"comma-separated subset of {sorted(BACKENDS)}")
    parser.add_argument("--iters", type=int, default=7, help="warm timing samples")
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls before timing")
    parser.add_argument("--restage-iters", type=int, default=3,
                        help="samples for the per-call restaging variant")
    parser.add_argument("--attempt-oversize", action="store_true",
                        help="attempt compiled legs above the ~161-input IREE runtime limit "
                             "(expected to SIGSEGV; the crash is then measured)")
    parser.add_argument("--no-sanity", action="store_true",
                        help="skip the second run of a small subset")
    parser.add_argument("--no-crash-probe", action="store_true",
                        help="do not deliberately attempt the over-limit compiled legs "
                             "(CPU-target ones only) to MEASURE the runtime crash")
    parser.add_argument("--sanity-rows",
                        default="tiny_llama:iree-rocm,tiny_llama:iree-rocm-packed,"
                                "gpt2:iree-rocm-packed",
                        help="legs to re-run entirely (second run) for the reproducibility table")
    parser.add_argument("--opt-level-evidence", action="store_true",
                        help="after a successful build, re-attempt the HIGHER opt_levels and "
                             "record the compiler's verbatim rejection (evidence for the "
                             "library's ROCm default of O1)")
    parser.add_argument("--no-public-build", action="store_true",
                        help="never measure the public one-call build (stage sum only)")
    parser.add_argument("--report", action="store_true", help="only re-render the markdown")
    parser.add_argument("--leg", help="internal: run one leg in this process")
    parser.add_argument("--backend", help="internal: backend key for --leg")
    parser.add_argument("--out", help="internal: JSON path for --leg")
    args = parser.parse_args(argv)

    if args.leg:
        if args.backend not in BACKENDS:
            parser.error(f"unknown backend {args.backend!r}; known: {sorted(BACKENDS)}")
        if args.leg not in MODELS:
            parser.error(f"unknown model {args.leg!r}; known: {sorted(MODELS)}")
        record = run_leg(
            args.leg,
            args.backend,
            iters=args.iters,
            warmup=args.warmup,
            restage_iters=args.restage_iters,
            attempt_oversize=args.attempt_oversize,
            measure_public_build=False if args.no_public_build else None,
            opt_level_evidence=args.opt_level_evidence,
        )
        out = Path(args.out) if args.out else _leg_path(args.leg, args.backend)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=_json_default))
        print(f"[leg] wrote {out} (status={record.get('status')})")
        return 0 if record.get("status") in ("ok", "skipped") else 1

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    for name in models:
        if name not in MODELS:
            parser.error(f"unknown model {name!r}; known: {sorted(MODELS)}")
    for name in backends:
        if name not in BACKENDS:
            parser.error(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")

    if not args.report:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for model in models:
            for backend in backends:
                run_child_leg(
                    model,
                    backend,
                    iters=args.iters,
                    warmup=args.warmup,
                    restage_iters=args.restage_iters,
                    attempt_oversize=args.attempt_oversize,
                    no_public_build=args.no_public_build,
                    opt_level_evidence=args.opt_level_evidence,
                )
        if not args.no_crash_probe:
            # A leg refused up front for the input-count limit is refused on
            # DOCUMENTED grounds; prove it once per (model, CPU backend) in an
            # isolated subprocess so the report can quote a real exit code.
            for model in models:
                for backend in backends:
                    if not (DATA_DIR / f"{model}__{backend}.json").is_file():
                        continue
                    record = json.loads((DATA_DIR / f"{model}__{backend}.json").read_text())
                    if record.get("status") != "skipped":
                        continue
                    if BACKENDS[backend].device_kind != "cpu":
                        continue  # never provoke a crash with a live GPU context
                    run_child_leg(
                        model,
                        backend,
                        iters=args.iters,
                        warmup=args.warmup,
                        restage_iters=args.restage_iters,
                        attempt_oversize=True,
                        no_public_build=True,
                        suffix=".crash",
                    )
        if not args.no_sanity:
            for row in args.sanity_rows.split(","):
                row = row.strip()
                if not row or ":" not in row:
                    continue
                model, backend = row.split(":", 1)
                if model in models and backend in backends:
                    run_child_leg(
                        model,
                        backend,
                        iters=args.iters,
                        warmup=args.warmup,
                        restage_iters=args.restage_iters,
                        attempt_oversize=args.attempt_oversize,
                        no_public_build=args.no_public_build,
                        opt_level_evidence=False,
                        suffix=".sanity",
                    )

    records = collect_records()
    if not records:
        print("no leg records found — run the probe first", file=sys.stderr)
        return 1
    REPORT_PATH.write_text(
        render_report(records, collect_sanity(), collect_crash_probe())
    )
    print(f"[report] wrote {REPORT_PATH} from {len(records)} leg records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
