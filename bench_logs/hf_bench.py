#!/usr/bin/env python3
"""Reproducible **HuggingFace model benchmark** for ``etl_hf``: iree / xla / numpy vs PyTorch.

Measures a real HuggingFace checkpoint's **whole forward pass as ONE etl graph**
(``etl_hf``: ``load_hf`` → ``forward_fn`` → ``etl_hf.build`` → ``etl_hf.run``, every
weight an ordinary graph **input**) on every execution path that exists for it, plus
the PyTorch baseline the same checkpoint runs at.

Legs (``--legs``) — one leg = one execution path, and the file/record key below::

    ==================  ==========================================================
    leg                 what it is
    ==================  ==========================================================
    numpy-cpu           the numpy interpreter backend (host arrays)
    iree-llvmcpu        IREE compiled for llvm-cpu, host-resident inputs
    iree-rocm           IREE compiled for ROCm (the MI300X), **device resident**
    xla-rocm            PJRT/XLA on ROCm (needs ``ETL_PJRT_PLUGIN``), device resident
    torch-cpu           ``transformers.AutoModelForCausalLM`` on CPU, fp32, no_grad
    torch-rocm          the same on ``cuda:0`` (the ROCm torch wheel says "cuda")
    torchcompile-cpu    ``torch.compile(model)`` on CPU
    torchcompile-rocm   ``torch.compile(model)`` on ``cuda:0``
    ==================  ==========================================================

``--phase`` expands to two default leg sets — ``cpu`` = ``numpy-cpu,iree-llvmcpu,
torch-cpu`` and ``gpu`` = ``iree-rocm,xla-rocm,torch-rocm``.  The two phases are run
**separately and sequentially, never concurrently**, because the GPU is shared.

Running it
----------
The full matrix runs from the throwaway bench venv that
``bench_logs/hf_bench_env.py`` provisions (torch 2.13.0+rocm7.1 + transformers +
iree + ``jax-rocm7-pjrt``) — **never** the repo ``.venv``::

    export PYTHONPATH=<repo root>                 # makes etl + etl_hf importable
    $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase cpu
    $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase gpu
    $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --report

Without a torch install only the ``etl``/torch-less legs run (``uv run --extra iree
python bench_logs/hf_bench.py --phase cpu --models tiny_llama`` is the CPU smoke
test); a torch leg then records an explicit ``status="error"`` naming the missing
module — a leg is never silently skipped.

Design (why it looks like this)
-------------------------------
* **One subprocess per (model, seq_len, leg).** A compiled leg can *segfault* (IREE's
  Python runtime dies above ~161 consumed inputs) and the xla/ROCm PJRT plugin
  **aborts at interpreter exit** (``malloc_consolidate(): invalid chunk size``) after
  all work succeeds. Each leg therefore runs in its own process, writes its JSON and
  ``.npz`` **before** any risky teardown and ends with ``os._exit(0)``; the driver
  enforces ``--leg-timeout`` and, on a non-zero exit / signal / timeout / no-record,
  writes an explicit ``<cell>.error.json`` with the returncode, signal, last
  ``[stage]`` marker and stderr tail. A leg that cannot run is **recorded**, never
  omitted.
* **Device legs use the documented device-resident loop**: ``place_params`` +
  ``place_ids`` ONCE, then run with those device tensors (no per-call restaging, the
  output is never auto-copied to the host). ``asdevicearray`` calls during the warm
  loop are counted (the technique from
  ``tests/backends/test_iree_same_device_loop.py``) and must be **zero**; the
  per-call restage anti-pattern is measured once (``tiny_llama`` / ``iree-rocm``) for
  reference and labelled as such.
* **Library defaults, never hand-picked flags.** Compile options come from
  ``etl_hf.pipeline.resolve_build_options`` (all-ROCm → ``opt_level="O1"``,
  llvm-cpu → ``opt_level="O3"`` + ``--iree-llvmcpu-target-cpu=host``). The resolved
  set is recorded per leg.
* **No downloads.** Checkpoints come from ``$ETL_HF_MODELS_DIR`` or
  ``$TMPDIR/etl_hf_models/<slug>``; a missing one is an explicit error naming the path.
* **``--report`` reads only JSON/``.npz``** (no torch, no iree, no measuring) and can
  be re-run at any time; cross-leg parity is recomputed from the saved
  ``*.logits.npz`` files, so the report never depends on leg execution order.

Files written under ``--out`` (default ``bench_logs/hf_bench_data``)
--------------------------------------------------------------------
``<model>__s<seq_len>__<leg>.json``         one record per measured cell (see below)
``<model>__s<seq_len>__<leg>.logits.npz``   that cell's ``logits`` (f32) + ``input_ids``
``<model>__s<seq_len>__<leg>.error.json``   a cell whose subprocess never wrote a record

Record schema (JSON, ``schema: 1``)
-----------------------------------
``status`` (``ok``/``error``/``timeout``), ``kind`` (``etl``/``torch``), plus::

    provenance  model, golden_slug, seq_len, leg, timestamp, hostname, git_commit,
                python, executable, command, env{python, packages{...}, ETL_*,
                XLA_FLAGS, LD_LIBRARY_PATH, visible devices, TMPDIR}
    leg         key, kind, backend, device, device_resident, options_requested,
                options_resolved, resolved_backend, pack{mode, packed, reason}
    model       checkpoint_dir, family, model_type, hidden_act, config_digest,
                vocab_size, hidden_size, num_hidden_layers, param_count,
                param_bytes, weight_keys, batch
    graph       unpacked_graph_inputs, packed_graph_inputs, graph_inputs,
                weight_inputs, ir_ops, ir_distinct_ops, ir_histogram,
                stablehlo_bytes, trace_ms, param_specs_ms, pack_plan_inputs,
                pack_params_ms, load_hf_ms
    build       stage_timing ("build_total" | "stages"), build_total_ms,
                build_total_cached_ms, trace_ms, lower_ms, compile_ms, load_ms,
                stage_sum_ms, artifact_bytes, executable_device
    staging     place_params_ms, place_params_leaves, place_params_steady,
                place_ids_ms                                  (device legs only)
    run         api, first_call_ms, best_ms, median_ms, mean_ms, stdev_ms, n,
                tokens_per_s, warm_with_d2h, d2h, peak_rss_mb
    checks      zero_asdevicearray_after_warmup, asdevicearray_calls,
                checked, reason
    parity      ids_source, input_ids, vs_golden, vs_numpy, vs_torch_cpu,
                vs_torch_rocm  (each: comparable, max_abs, rel_to_absmax,
                argmax_agreement, ref_absmax, reason/reference)
    logits      path, shape, absmax, mean, finite
    anti_pattern (device legs, ``tiny_llama``/``iree-rocm`` only)
    notes       human-readable remarks (never load-bearing numbers)
    errors      the verbatim failure messages, when status != "ok"
    process     (added by the driver) exit_code, signal, duration_s,
                stdout_tail, stderr_tail, forced_exit

Nothing is faked: an absent number is absent (the report renders ``—`` **with the
reason**), and a leg that could not run is listed with its exact error/returncode.
"""
from __future__ import annotations

import argparse
import json
import os
import signal as signal_module
import subprocess
import sys
import time
import traceback
from pathlib import Path

import hf_bench_lib as lib

#: Models whose GPU legs get the per-call restaging anti-pattern measurement.
ANTI_PATTERN_MODELS: tuple[str, ...] = ("tiny_llama",)
#: Packing is "beneficial" (``--pack auto``) for a device leg staging more weight
#: leaves than this once — the host→device placement cost is per leaf.
PACK_BENEFICIAL_LEAVES = 50
#: Skip the (backend-independent) StableHLO export above this many IR ops.
EXPORT_OP_LIMIT = 4000


def _progress(label: str) -> None:
    """Print a stage marker so a hard crash still says where it died.

    The driver captures each leg's stdout and records the last marker reached, which
    is the difference between "died in compile" and "died at invoke".
    """
    print(f"[stage] {label}", flush=True)


# ---------------------------------------------------------------------------
# pack policy
# ---------------------------------------------------------------------------
def decide_pack(leg: lib.LegSpec, unpacked_inputs: int, weight_inputs: int, mode: str):
    """Resolve ``--pack`` for one leg: ``(packed, reason)``.

    ``auto`` (the default) packs only when it is **needed or measured-beneficial**:

    * needed — a compiled (non-numpy) leg whose unpacked graph consumes more than
      :data:`lib.INPUT_LIMIT` inputs, i.e. the documented IREE runtime SIGSEGV at
      ``invoke`` (only the 124M ``gpt2`` gets there: 197 unpacked inputs);
    * beneficial — a device-resident leg with more than
      :data:`PACK_BENEFICIAL_LEAVES` weight leaves, because placement is one
      host→device transfer per leaf.

    Everything else stays unpacked (the library default), so a leg's row is the
    canonical graph unless there is a reason for it not to be.
    """
    if mode == "on":
        return True, "forced by --pack on"
    if mode == "off":
        return False, "forced by --pack off"
    if leg.backend not in (None, "numpy") and unpacked_inputs > lib.INPUT_LIMIT:
        return True, (
            f"auto: {unpacked_inputs} unpacked graph inputs exceed the ~"
            f"{lib.INPUT_LIMIT}-input IREE runtime limit (SIGSEGV at invoke)"
        )
    if leg.device_resident and weight_inputs > PACK_BENEFICIAL_LEAVES:
        return True, (
            f"auto: device-resident leg would stage {weight_inputs} weight leaves; "
            f"packing collapses them to a handful of host->device transfers"
        )
    if leg.backend == "numpy" and unpacked_inputs > lib.INPUT_LIMIT:
        return False, (
            f"auto: unpacked — the numpy interpreter has no input-count limit "
            f"({unpacked_inputs} inputs)"
        )
    return False, "auto: unpacked (below the input limit, no staging benefit)"


# ---------------------------------------------------------------------------
# referencing helpers shared by the measuring legs
# ---------------------------------------------------------------------------
def parity_block(leg_key, model, seq_len, logits, out_dir, ids_source) -> dict:
    """Parity of one leg's logits against every reference available on disk."""
    block: dict = {"ids_source": ids_source}
    golden, golden_reason = lib.golden_reference(model, seq_len)
    if golden is None:
        block["vs_golden"] = {"comparable": False, "reason": golden_reason}
    else:
        block["vs_golden"] = {
            **lib.compare_logits(logits, golden["logits"]),
            "reference": golden["path"],
            "reference_note": golden_reason,
        }
    references = (
        ("numpy-cpu", "vs_numpy"),
        ("torch-cpu", "vs_torch_cpu"),
        ("torch-rocm", "vs_torch_rocm"),
    )
    for other_key, field in references:
        if other_key == leg_key:
            block[field] = {"comparable": False, "reason": "same leg"}
            continue
        path = lib.leg_paths(out_dir, model, seq_len, other_key)["logits"]
        array = lib.read_logits(path)
        if array is None:
            block[field] = {
                "comparable": False,
                "reason": f"no logits on disk for the {other_key} leg",
            }
            continue
        block[field] = {
            **lib.compare_logits(logits, array),
            "reference": str(path),
            "reference_leg": other_key,
        }
    return block


def _emit_record(record: dict, out_dir, model: str, seq_len: int, leg_key: str) -> Path:
    """Write a leg's record where the driver and ``--report`` look for it."""
    path = lib.leg_paths(out_dir, model, seq_len, leg_key)["json"]
    lib.write_json(path, record)
    print(f"[record] {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# the etl leg: one model forward as ONE etl graph on one execution path
# ---------------------------------------------------------------------------
def measure_etl_leg(
    model: str,
    seq_len: int,
    leg_key: str,
    *,
    out_dir,
    warmup: int,
    repeat: int,
    save_logits: bool,
    stage_timings: bool,
    pack_mode: str,
    anti_pattern: bool,
    batch: int = 1,
) -> dict:
    """Measure one ``etl_hf`` leg cell.  Returns its record (never raises)."""
    import etl
    import etl_hf
    import numpy as np
    from etl.backends import stablehlo

    spec = lib.leg_spec(leg_key)
    record: dict = {
        "schema": lib.SCHEMA,
        "status": "ok",
        "kind": "etl",
        "provenance": lib.provenance(model, seq_len, leg_key),
        "leg": {
            "key": leg_key,
            "kind": spec.kind,
            "backend": spec.backend,
            "device": spec.device_string,
            "device_resident": spec.device_resident,
            "options_requested": dict(spec.options),
        },
        "measurement": {"warmup": warmup, "repeat": repeat, "batch": batch},
        "notes": [],
        "errors": [],
    }
    notes: list[str] = record["notes"]
    errors: list[str] = record["errors"]

    def fail(stage: str, exc: BaseException) -> dict:
        record["status"] = "error"
        record["failed_stage"] = stage
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        record["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
        return record

    # -- checkpoint + config + ids -----------------------------------------
    _progress("load checkpoint")
    try:
        ckpt = lib.checkpoint_dir(model)
        record["model"] = {"checkpoint_dir": str(ckpt), "batch": batch}
        loaded, load_s = lib.timeit(etl_hf.load_hf, ckpt)
        cfg, params = loaded.config, loaded.params
        record["model"].update(
            {
                "family": loaded.family,
                "model_type": cfg.model_type,
                "hidden_act": cfg.hidden_act,
                "config_digest": cfg.digest,
                "vocab_size": cfg.vocab_size,
                "hidden_size": cfg.hidden_size,
                "num_hidden_layers": cfg.num_hidden_layers,
                "tie_word_embeddings": bool(cfg.tie_word_embeddings),
                "param_count": int(sum(int(a.size) for a in params.values())),
                "param_bytes": int(sum(int(a.nbytes) for a in params.values())),
                "weight_keys": sorted(params),
            }
        )
        record["graph"] = {"load_hf_ms": lib.ms(load_s)}
    except Exception as exc:
        return fail("load_hf", exc)

    # ids: the golden fixture's own ids when they match this cell's seq_len
    try:
        golden, golden_reason = lib.golden_reference(model, seq_len)
        if golden is not None:
            ids, ids_source = np.asarray(golden["input_ids"], np.int32), "hf_goldens input_ids"
        else:
            ids, ids_source = lib.reference_ids(model, seq_len, cfg.vocab_size)
            ids_source += f" [{golden_reason}]"
        record["parity"] = {"ids_source": ids_source, "input_ids": ids.tolist()}
    except Exception as exc:
        return fail("input ids", exc)

    # -- graph size: specs + pack decision ---------------------------------
    _progress("param specs")
    try:
        specs_t, specs_s = lib.timeit(etl_hf.param_specs, cfg)
        unpacked = lib.spec_tree_summary(specs_t)
        packed_specs_t = etl_hf.param_specs(cfg, pack_weights=True)
        packed_summary = lib.spec_tree_summary(packed_specs_t)
        plan = etl_hf.pack_plan(specs_t)
        packed, pack_reason = decide_pack(
            spec, unpacked["graph_inputs"], unpacked["weight_inputs"], pack_mode
        )
        record["leg"]["pack"] = {"mode": pack_mode, "packed": packed, "reason": pack_reason}
        record["graph"].update(
            {
                "unpacked_graph_inputs": unpacked["graph_inputs"],
                "packed_graph_inputs": packed_summary["graph_inputs"],
                "pack_plan_inputs": len(plan.specs),
                "param_specs_ms": lib.ms(specs_s),
                "weight_inputs": (
                    packed_summary["weight_inputs"] if packed else unpacked["weight_inputs"]
                ),
                "graph_inputs": (
                    packed_summary["graph_inputs"] if packed else unpacked["graph_inputs"]
                ),
                "packed_input_names": sorted(packed_specs_t),
            }
        )
        notes.append(f"pack: {pack_reason}")
    except Exception as exc:
        return fail("param_specs", exc)

    # -- the traced graph: IR op histogram + StableHLO size ----------------
    _progress("trace")
    try:
        fwd = etl_hf.forward_fn(cfg, seq_len=seq_len, batch_size=batch, pack_weights=packed)
        specs = etl_hf.param_specs(cfg, pack_weights=packed)
        graph, trace_s = lib.timeit(
            etl.trace, fwd, specs, etl.TensorSpec((batch, seq_len), etl.int32)
        )
        graph.verify()
        histogram = lib.count_ops(graph)
        record["graph"].update(
            {
                "trace_ms": lib.ms(trace_s),
                "ir_ops": int(sum(histogram.values())),
                "ir_distinct_ops": len(histogram),
                "ir_histogram": histogram,
            }
        )
    except Exception as exc:
        return fail("trace/verify", exc)

    _progress("stablehlo export")
    try:
        if record["graph"]["ir_ops"] > EXPORT_OP_LIMIT:
            record["graph"]["stablehlo_bytes"] = None
            notes.append(
                f"StableHLO export skipped: {record['graph']['ir_ops']} IR ops exceed the "
                f"{EXPORT_OP_LIMIT}-op limit of this harness"
            )
        else:
            text, export_s = lib.timeit(stablehlo.export, graph)
            record["graph"]["stablehlo_bytes"] = len(text.encode())
            record["graph"]["stablehlo_export_ms"] = lib.ms(export_s)
    except Exception as exc:
        record["graph"]["stablehlo_bytes"] = None
        notes.append(f"StableHLO export failed: {type(exc).__name__}: {exc}")

    # -- build (the library default matrix, no hand-picked flags) ----------
    _progress("resolve build options")
    try:
        device = spec.etl_device()
        backend_name, options = etl_hf.pipeline.resolve_build_options(
            spec.backend, device, spec.options
        )
        record["leg"].update(
            {
                "resolved_backend": backend_name,
                "options_resolved": options,
            }
        )
    except Exception as exc:
        return fail("resolve_build_options", exc)

    _progress(f"build (backend={backend_name})")
    try:
        exe, build_s = lib.timeit(
            etl_hf.build,
            fwd,
            cfg,
            backend=backend_name,
            device=device,
            pack_weights=packed,
            **options,
        )
        build: dict = {
            "stage_timing": "build_total",
            "build_total_ms": lib.ms(build_s),
            "executable_device": str(exe.device),
            "resolved_options": options,
        }
        # memoisation cross-check: an identical build must not recompile
        try:
            _, cached_s = lib.timeit(
                etl_hf.build,
                fwd,
                cfg,
                backend=backend_name,
                device=device,
                pack_weights=packed,
                **options,
            )
            build["build_total_cached_ms"] = lib.ms(cached_s)
        except Exception as exc:
            build["build_total_cached_error"] = f"{type(exc).__name__}: {exc}"
        record["build"] = build
        notes.append(
            "build stage times are reported as build_total: etl_hf.build runs "
            "trace -> lower -> compile -> load through etl.build_cached, so the inner "
            "stages are not separately observable (pass --stage-timings for an explicit, "
            "additionally-compiled lower/compile/load breakdown)"
        )
    except Exception as exc:
        return fail("build", exc)

    # -- optional explicit stage timings (a SECOND compilation) -------------
    if stage_timings:
        _progress("stage timings (lower/compile/load)")
        try:
            lowered, lower_s = lib.timeit(etl.lower, graph, backend=backend_name, **options)
            artifact, compile_s = lib.timeit(
                etl.compile, lowered, backend=backend_name, **options
            )
            _, load_s = lib.timeit(
                etl.load, artifact, backend=backend_name, device=device, **options
            )
            record["build"].update(
                {
                    "stage_timing": "stages",
                    "lower_ms": lib.ms(lower_s),
                    "compile_ms": lib.ms(compile_s),
                    "load_ms": lib.ms(load_s),
                    "stage_sum_ms": lib.ms(lower_s + compile_s + load_s),
                    "artifact_bytes": lib.artifact_bytes(artifact),
                }
            )
        except Exception as exc:
            notes.append(
                "explicit stage timings failed: " f"{type(exc).__name__}: {exc}"
            )

    # -- placement + the run loop ------------------------------------------
    _progress("place params / prepare run")
    run: dict = {"batch": batch, "seq_len": seq_len}
    record["run"] = run
    try:
        host_params = params
        if packed:
            # pack ONCE on the host (never per call), with the plan from the specs stage
            host_params, pack_s = lib.timeit(etl_hf.pack_params, plan, params)
            record["graph"]["pack_params_ms"] = lib.ms(pack_s)
            notes.append(
                f"packed weights: the canonical tree is concatenated ONCE on the host "
                f"into {len(host_params)} inputs (the graph unpacks them with static slices)"
            )
        if spec.device_resident:
            placed, place_s = lib.timeit(etl_hf.place_params, host_params, device)
            ids_arg, ids_s = lib.timeit(etl_hf.place_ids, ids, device)
            staging: dict = {
                "place_params_ms": lib.ms(place_s),
                "place_params_leaves": len(placed),
                "place_ids_ms": lib.ms(ids_s),
            }
            try:
                steady = []
                for _ in range(3):
                    _, seconds = lib.timeit(etl_hf.place_params, host_params, device)
                    steady.append(seconds)
                staging["place_params_steady"] = lib.summarize(steady)
            except Exception as exc:
                staging["place_params_steady_error"] = f"{type(exc).__name__}: {exc}"
            record["staging"] = staging
            params_arg = placed
            run["api"] = "etl_hf.run(exe, place_params(params, device), place_ids(ids, device))"
            notes.append(
                "device-resident: weights and ids are placed ONCE; the measured loop is "
                "a same-device loop (nothing is restaged, the output never auto-copies "
                "to the host)"
            )
        else:
            params_arg, ids_arg = host_params, ids
            run["api"] = "etl_hf.run(exe, host params, host ids)"
    except Exception as exc:
        return fail("placement", exc)

    def to_host(tensor):
        """Explicit (single, documented) device→host transfer of the logits."""
        if spec.device_resident:
            tensor = tensor.to(etl.core.Device("cpu", 0))
        return np.asarray(tensor.numpy(), dtype=np.float32)

    _progress("first run")
    try:
        out, first_s = lib.timeit(etl_hf.run, exe, params_arg, ids_arg)
        run["first_call_ms"] = lib.ms(first_s)
        logits = to_host(out)
        run["logits_device"] = str(getattr(out, "device", "unknown"))
    except Exception as exc:
        return fail("first run", exc)

    # -- the warm loop, with the zero-host-round-trip probe ----------------
    _progress("warm loop")
    counter = lib.AsDeviceArrayCounter()
    try:
        with counter:
            for _ in range(max(0, warmup)):
                etl_hf.run(exe, params_arg, ids_arg)
            counter.reset()  # only the measured samples count
            samples = []
            for _ in range(max(1, repeat)):
                _, seconds = lib.timeit(etl_hf.run, exe, params_arg, ids_arg)
                samples.append(seconds)
    except Exception as exc:
        return fail("warm loop", exc)
    run.update(lib.summarize(samples))
    run["warmup"] = warmup
    run["tokens_per_s"] = lib.tokens_per_s(batch, seq_len, run.get("best_ms"))
    run["peak_rss_mb"] = lib.peak_rss_mb()
    checks: dict = {
        "checked": counter.checked,
        "asdevicearray_calls": counter.calls,
        "reason": counter.reason,
    }
    if spec.device_resident:
        checks["zero_asdevicearray_after_warmup"] = (
            counter.calls == 0 if counter.checked else None
        )
        if counter.checked and counter.calls:
            notes.append(
                f"VIOLATION: {counter.calls} asdevicearray call(s) during the measured "
                f"warm loop — the device-resident loop must perform ZERO host uploads"
            )
    record["checks"] = checks

    # -- +d2h: one explicit device→host transfer per call -------------------
    if spec.device_resident:
        _progress("d2h")
        try:
            cpu = etl.core.Device("cpu", 0)
            transfer = []
            for _ in range(3):
                out = etl_hf.run(exe, params_arg, ids_arg)
                _, seconds = lib.timeit(lambda: out.to(cpu).numpy())
                transfer.append(seconds)
            run["d2h"] = lib.summarize(transfer)
            combined = []
            for _ in range(max(1, repeat)):
                start = time.perf_counter()
                out = etl_hf.run(exe, params_arg, ids_arg)
                out.to(cpu).numpy()
                combined.append(time.perf_counter() - start)
            run["warm_with_d2h"] = lib.summarize(combined)
        except Exception as exc:
            errors.append(f"d2h measurement: {type(exc).__name__}: {exc}")
    else:
        run["d2h"] = {
            "applicable": False,
            "reason": "the output is already host-resident (cpu executable)",
        }

    # -- the per-call restaging anti-pattern (once, labelled) --------------
    if spec.device_resident and anti_pattern and model in ANTI_PATTERN_MODELS:
        _progress("anti-pattern restage")
        try:
            restaged = []
            for _ in range(max(2, min(repeat, 5))):
                start = time.perf_counter()
                again = etl_hf.place_params(host_params, device)
                etl_hf.run(exe, again, ids_arg)
                restaged.append(time.perf_counter() - start)
            record["anti_pattern"] = {
                "label": "ANTI-PATTERN: per-call host->device restage of the params tree",
                "detail": "etl_hf.place_params(host params, device) + etl_hf.run per call",
                **lib.summarize(restaged),
            }
        except Exception as exc:
            errors.append(f"anti-pattern restage: {type(exc).__name__}: {exc}")

    # -- parity + logits ---------------------------------------------------
    _progress("parity")
    try:
        record["parity"].update(
            parity_block(leg_key, model, seq_len, logits, out_dir, ids_source)
        )
    except Exception as exc:
        errors.append(f"parity: {type(exc).__name__}: {exc}")
    summary = lib.logits_summary(logits)
    paths = lib.leg_paths(out_dir, model, seq_len, leg_key)
    if save_logits:
        lib.save_logits(paths["logits"], logits, ids)
        summary["path"] = str(paths["logits"])
    else:
        summary["path"] = None
        summary["reason"] = "--no-logits"
    record["logits"] = summary
    return record


# ---------------------------------------------------------------------------
# the torch leg: the PyTorch/transformers baseline
# ---------------------------------------------------------------------------
def _load_torch_model(ckpt, torch):
    """``transformers.AutoModelForCausalLM`` from the checkpoint dir, fp32."""
    from transformers import AutoModelForCausalLM

    base = {"local_files_only": True}
    attempts = [
        dict(base, dtype=torch.float32, attn_implementation="eager"),
        dict(base, torch_dtype=torch.float32, attn_implementation="eager"),
        dict(base, attn_implementation="eager"),
        dict(base),
    ]
    last: Exception | None = None
    for kwargs in attempts:
        try:
            return AutoModelForCausalLM.from_pretrained(str(ckpt), **kwargs), kwargs
        except TypeError as exc:  # unsupported keyword for this transformers version
            last = exc
    raise last if last is not None else RuntimeError("from_pretrained failed")


def measure_torch_leg(
    model: str,
    seq_len: int,
    leg_key: str,
    *,
    out_dir,
    warmup: int,
    repeat: int,
    save_logits: bool,
    batch: int = 1,
) -> dict:
    """Measure one PyTorch baseline leg cell (torch imported HERE, lazily)."""
    import numpy as np

    spec = lib.leg_spec(leg_key)
    record: dict = {
        "schema": lib.SCHEMA,
        "status": "ok",
        "kind": "torch",
        "provenance": lib.provenance(model, seq_len, leg_key),
        "leg": {
            "key": leg_key,
            "kind": spec.kind,
            "backend": None,
            "device": spec.device_string,
            "device_resident": str(spec.torch_device).startswith("cuda"),
            "torch_compile": spec.torch_compile,
        },
        "measurement": {"warmup": warmup, "repeat": repeat, "batch": batch},
        "notes": [],
        "errors": [],
    }
    notes: list[str] = record["notes"]
    errors: list[str] = record["errors"]

    def fail(stage: str, exc: BaseException) -> dict:
        record["status"] = "error"
        record["failed_stage"] = stage
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")
        record["traceback_last"] = traceback.format_exc().splitlines()[-12:]
        return record

    _progress("import torch")
    try:
        import torch
    except Exception as exc:
        return fail("import torch", exc)

    _progress("load checkpoint (transformers)")
    try:
        ckpt = lib.checkpoint_dir(model)
        record["model"] = {"checkpoint_dir": str(ckpt), "batch": batch}
        if spec.torch_device and str(spec.torch_device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "torch reports no CUDA/ROCm device (torch.cuda.is_available() is False)"
                )
            record["model"]["torch_device_name"] = torch.cuda.get_device_name(0)
        torch_model, kwargs = _load_torch_model(ckpt, torch)
        record["model"]["from_pretrained_kwargs"] = {
            key: str(value) for key, value in kwargs.items()
        }
        torch_model.eval()
        device = torch.device(spec.torch_device)
        torch_model.to(device)
        record["model"].update(
            {
                "family": "torch",
                "model_class": type(torch_model).__name__,
                "config_model_type": getattr(torch_model.config, "model_type", None),
                "param_count": int(sum(p.numel() for p in torch_model.parameters())),
                "param_bytes": int(
                    sum(p.numel() * p.element_size() for p in torch_model.parameters())
                ),
                "dtype": str(next(torch_model.parameters()).dtype),
            }
        )
        if spec.torch_compile:
            _progress("torch.compile")
            compiled, compile_s = lib.timeit(torch.compile, torch_model)
            torch_model = compiled
            record["build"] = {
                "stage_timing": "torch",
                "torch_compile_call_ms": lib.ms(compile_s),
            }
            notes.append(
                "torch.compile compiles lazily: the first measured call includes the "
                "Inductor compilation"
            )
    except Exception as exc:
        return fail("load checkpoint", exc)

    # ids: exactly the ones the etl legs use (golden ids when they match)
    try:
        golden, golden_reason = lib.golden_reference(model, seq_len)
        vocab = int(getattr(torch_model.config, "vocab_size", 0) or 0)
        if golden is not None:
            ids, ids_source = np.asarray(golden["input_ids"], np.int32), "hf_goldens input_ids"
        else:
            ids, ids_source = lib.reference_ids(model, seq_len, vocab)
            ids_source += f" [{golden_reason}]"
        record["parity"] = {"ids_source": ids_source, "input_ids": ids.tolist()}
    except Exception as exc:
        return fail("input ids", exc)

    sync = str(spec.torch_device).startswith("cuda")
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)

    def call():
        if sync:
            torch.cuda.synchronize()
        logits = torch_model(input_ids=ids_t).logits
        if sync:
            torch.cuda.synchronize()
        return logits

    _progress("first run")
    try:
        with torch.no_grad():
            logits_t, first_s = lib.timeit(call)
        run: dict = {"batch": batch, "seq_len": seq_len, "api": "model(input_ids=ids).logits"}
        run["first_call_ms"] = lib.ms(first_s)
        array = logits_t.detach().float().cpu().numpy().astype(np.float32)
    except Exception as exc:
        return fail(
            "torch.compile" if spec.torch_compile else "first run", exc
        )
    record["run"] = run
    notes.append(
        "timed region includes torch.cuda.synchronize() before and after each call "
        "(GPU) — a forward-pass wall time, no decode loop"
        if sync
        else "CPU wall time per forward call"
    )

    _progress("warm loop")
    try:
        samples = []
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                call()
            for _ in range(max(1, repeat)):
                _, seconds = lib.timeit(call)
                samples.append(seconds)
    except Exception as exc:
        return fail("warm loop", exc)
    run.update(lib.summarize(samples))
    run["warmup"] = warmup
    run["tokens_per_s"] = lib.tokens_per_s(batch, seq_len, run.get("best_ms"))
    run["peak_rss_mb"] = lib.peak_rss_mb()
    run["d2h"] = {
        "applicable": False,
        "reason": "the logits are moved to the host once per call inside the timed region"
        if sync
        else "the output is already host-resident",
    }

    _progress("parity")
    try:
        record["parity"].update(
            parity_block(leg_key, model, seq_len, array, out_dir, ids_source)
        )
    except Exception as exc:
        errors.append(f"parity: {type(exc).__name__}: {exc}")
    summary = lib.logits_summary(array)
    paths = lib.leg_paths(out_dir, model, seq_len, leg_key)
    if save_logits:
        lib.save_logits(paths["logits"], array, ids)
        summary["path"] = str(paths["logits"])
    else:
        summary["path"] = None
        summary["reason"] = "--no-logits"
    record["logits"] = summary
    return record


# ---------------------------------------------------------------------------
# child mode: measure one (model, seq_len, leg) cell in THIS process
# ---------------------------------------------------------------------------
def run_child(args) -> int:
    """``--leg LEG`` child: measure the cell(s) and write the record(s)."""
    leg_key = lib.leg_spec(args.leg).key
    out_dir = args.out[0]
    for model in args.models:
        for seq_len in args.seq_lens:
            _progress(f"{model}/s{seq_len}/{leg_key}: start")
            spec = lib.leg_spec(leg_key)
            if spec.kind == "etl":
                record = measure_etl_leg(
                    model,
                    seq_len,
                    leg_key,
                    out_dir=out_dir,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    save_logits=args.logits,
                    stage_timings=args.stage_timings,
                    pack_mode=args.pack,
                    anti_pattern=args.anti_pattern,
                )
            else:
                record = measure_torch_leg(
                    model,
                    seq_len,
                    leg_key,
                    out_dir=out_dir,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    save_logits=args.logits,
                )
            path = _emit_record(record, out_dir, model, seq_len, leg_key)
            print(
                f"[leg] {model}/s{seq_len}/{leg_key}: {record['status']} -> {path}",
                flush=True,
            )
    sys.stdout.flush()
    sys.stderr.flush()
    # A PJRT plugin (xla/ROCm) corrupts the heap at interpreter exit AFTER all work
    # succeeds; every record/id is already on disk, so skip teardown entirely.
    os._exit(0)


# ---------------------------------------------------------------------------
# driver: schedule legs as isolated subprocesses
# ---------------------------------------------------------------------------
def child_env(leg_key: str) -> dict:
    """Environment for a leg subprocess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(lib.REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # never reach the network from a leg
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if leg_key.startswith("xla"):
        # REQUIRED on this host: HIP-graph command buffers SIGSEGV inside
        # PJRT_Executable_Execute (see bench_logs/CONTEXT.md).
        required = "--xla_gpu_enable_command_buffer="
        flags = (env.get("XLA_FLAGS") or "").strip()
        if required not in flags:
            env["XLA_FLAGS"] = f"{flags} {required}".strip()
    return env


def leg_command(args, model: str, seq_len: int, leg_key: str) -> list[str]:
    """The exact argv of one leg subprocess."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--leg",
        leg_key,
        "--models",
        model,
        "--seq-lens",
        str(int(seq_len)),
        "--out",
        str(args.out[0]),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--pack",
        args.pack,
    ]
    if not args.logits:
        command.append("--no-logits")
    if args.stage_timings:
        command.append("--stage-timings")
    if not args.anti_pattern:
        command.append("--no-anti-pattern")
    return command


def _synthesize_error_record(
    args, model: str, seq_len: int, leg_key: str, command, *, exit_code, stdout, stderr,
    duration_s, timed_out: bool,
) -> dict:
    """The explicit record of a leg that died before it could write one."""
    spec = lib.leg_spec(leg_key)
    signo = -exit_code if isinstance(exit_code, int) and exit_code < 0 else None
    signame = None
    if signo:
        try:
            signame = signal_module.Signals(signo).name
        except Exception:
            signame = f"SIG{signo}"
    stage = lib.last_stage(stdout) or "process start"
    tail = lib.stderr_tail(stderr, 5)
    if timed_out:
        message = (
            f"leg timed out after {args.leg_timeout:g}s at stage {stage!r} "
            f"(--leg-timeout); the process was killed"
        )
    elif signo:
        message = (
            f"leg process died with {signame} (signal {signo}) at stage {stage!r} "
            f"before writing a record"
        )
    else:
        message = (
            f"leg process exited {exit_code} at stage {stage!r} before writing a record"
        )
    record = {
        "schema": lib.SCHEMA,
        "status": "timeout" if timed_out else "error",
        "kind": spec.kind,
        "provenance": lib.provenance(model, seq_len, leg_key, command),
        "leg": {
            "key": leg_key,
            "kind": spec.kind,
            "backend": spec.backend,
            "device": spec.device_string,
            "device_resident": spec.device_resident,
            "options_requested": dict(spec.options),
        },
        "measurement": {"warmup": args.warmup, "repeat": args.repeat},
        "process": {
            "returncode": exit_code,
            "signal": signo,
            "signal_name": signame,
            "stage": stage,
            "duration_s": round(duration_s, 2),
            "timed_out": bool(timed_out),
            "stderr_tail": tail,
            "stdout_tail": lib.stderr_tail(stdout, 6),
            "command": [str(part) for part in command],
        },
        "notes": [],
        "errors": [message, *tail],
    }
    return record


def run_cell(args, model: str, seq_len: int, leg_key: str) -> dict:
    """Run ONE leg in its own subprocess and return the merged record."""
    out_dir = Path(args.out[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = lib.leg_paths(out_dir, model, seq_len, leg_key)
    for key in ("json", "error"):
        if paths[key].is_file():
            paths[key].unlink()  # a re-run must never read a stale record
    command = leg_command(args, model, seq_len, leg_key)
    label = f"{model}/s{seq_len}/{leg_key}"
    print(f"[leg] {label}: running …", flush=True)
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.leg_timeout,
            env=child_env(leg_key),
            cwd=str(lib.REPO_ROOT),
        )
        exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None

        def _text(value):
            return value.decode() if isinstance(value, bytes) else (value or "")

        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
    duration = time.perf_counter() - started

    if paths["json"].is_file():
        record = json.loads(paths["json"].read_text())
        record.setdefault("process", {})
        record["process"].update(
            {
                "returncode": exit_code,
                "signal": (-exit_code if isinstance(exit_code, int) and exit_code < 0 else None),
                "duration_s": round(duration, 2),
                "timed_out": timed_out,
                "stderr_tail": lib.stderr_tail(stderr, 5),
                "stdout_tail": lib.stderr_tail(stdout, 6),
                "command": [str(part) for part in command],
                "forced_exit": "os._exit(0) after the record was written"
                if exit_code == 0
                else None,
            }
        )
        if timed_out:
            record["status"] = "timeout"
            record.setdefault("errors", []).append(
                f"leg timed out after {args.leg_timeout:g}s (--leg-timeout); "
                "the record above is what it had written"
            )
        lib.write_json(paths["json"], record)
    else:
        record = _synthesize_error_record(
            args,
            model,
            seq_len,
            leg_key,
            command,
            exit_code=exit_code,            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            timed_out=timed_out,
        )
        lib.write_json(paths["error"], record)
    print(
        f"[leg] {label}: {record.get('status')} in {duration:.1f}s"
        + (f" (exit {exit_code})" if exit_code not in (0, None) else ""),
        flush=True,
    )
    return record


def resolve_legs(args) -> list[str]:
    """Legs to run, from ``--legs``, ``--phase``, or the documented default set."""
    if args.legs:
        requested = [part.strip() for part in args.legs.split(",") if part.strip()]
        unknown = [key for key in requested if key not in lib.LEGS]
        if unknown:
            raise SystemExit(
                f"unknown leg(s) {unknown}; known legs: {', '.join(lib.LEG_ORDER)}"
            )
        return lib.sort_legs(requested)
    phases = (
        [part.strip() for part in args.phase.split(",") if part.strip()]
        if args.phase
        else []
    )
    if phases:
        legs: list[str] = []
        for phase in phases:
            if phase not in lib.PHASES:
                raise SystemExit(
                    f"unknown phase {phase!r}; known phases: {', '.join(sorted(lib.PHASES))}"
                )
            legs.extend(lib.PHASES[phase])
        return lib.sort_legs(legs)
    # default: the six primary legs (no torch.compile variants)
    return lib.sort_legs([key for key in lib.LEG_ORDER if not key.startswith("torchcompile")])


def plan_cells(args) -> list[tuple[str, int, str]]:
    """Deterministic (model, seq_len, leg) schedule."""
    legs = resolve_legs(args)
    cells: list[tuple[str, int, str]] = []
    for model in args.models:
        for seq_len in args.seq_lens:
            for leg_key in legs:
                cells.append((model, int(seq_len), leg_key))
    return cells


def drive(args) -> int:
    """Run the plan (all legs already present are re-run: the run is idempotent)."""
    cells = plan_cells(args)
    if args.dry_run:
        print(f"[plan] {len(cells)} leg cell(s) -> {args.out[0]}")
        for model, seq_len, leg_key in cells:
            print("  " + " ".join(leg_command(args, model, seq_len, leg_key)))
        return 0
    print(f"[plan] {len(cells)} leg cell(s) -> {args.out[0]}", flush=True)
    results: dict = {}
    for model, seq_len, leg_key in cells:
        results[(model, seq_len, leg_key)] = run_cell(args, model, seq_len, leg_key)
    bad = [
        f"{model}/s{seq_len}/{leg}"
        for (model, seq_len, leg), record in results.items()
        if record.get("status") != "ok"
    ]
    print(
        f"[done] {len(cells) - len(bad)}/{len(cells)} leg(s) ok"
        + (f"; not ok: {', '.join(bad)}" if bad else ""),
        flush=True,
    )
    print(
        f"[done] records in {args.out[0]}; generate the markdown with "
        f"`--report` (or `--out {args.out[0]} --report`)",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# report: markdown from the JSON/npz files ONLY (no torch, no iree, no measure)
# ---------------------------------------------------------------------------
def collect_records(out_dirs) -> dict:
    """Every leg record on disk, keyed by ``(model, seq_len, leg)``."""
    records: dict = {}
    for out_dir in out_dirs:
        directory = Path(out_dir)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            name = path.name
            if name.endswith(".error.json"):
                base = name[: -len(".error.json")]
            elif name.endswith(".json"):
                base = name[: -len(".json")]
            else:
                continue
            parts = base.split("__")
            if len(parts) != 3 or not parts[1].startswith("s"):
                continue
            model, seq_part, leg_key = parts
            try:
                seq_len = int(seq_part[1:])
            except ValueError:
                continue
            if leg_key not in lib.LEGS:
                continue
            key = (model, seq_len, leg_key)
            if key in records:  # earlier --out dirs win
                continue
            try:
                record = json.loads(path.read_text())
                if not isinstance(record, dict):
                    raise ValueError(f"record is a {type(record).__name__}, not an object")
                if not isinstance(record.get("leg"), dict):
                    # Hand-written / partial records may carry the leg key as a
                    # plain string; normalise so every renderer can read it.
                    record["leg"] = {"key": leg_key, "kind": lib.leg_spec(leg_key).kind}
                records[key] = (record, path)
            except Exception as exc:
                records[key] = (
                    {
                        "status": "error",
                        "errors": [f"unreadable record {path}: {exc}"],
                        "provenance": {"model": model, "seq_len": seq_len, "leg": leg_key},
                    },
                    path,
                )
    return records


def _cell_order(records) -> list[tuple[str, int, str]]:
    """Deterministic ordering: model (``MODELS`` order first), seq, leg."""
    model_rank = {model: index for index, model in enumerate(lib.MODELS)}
    leg_rank = {leg: index for index, leg in enumerate(lib.LEG_ORDER)}
    return sorted(
        records,
        key=lambda key: (
            model_rank.get(key[0], len(model_rank)),
            key[0],
            key[1],
            leg_rank.get(key[2], len(leg_rank)),
            key[2],
        ),
    )


def _record_at(records, model, seq_len, leg_key):
    entry = records.get((model, seq_len, leg_key))
    return entry[0] if entry else None


def _best_ms(record) -> float | None:
    if not record or record.get("status") != "ok":
        return None
    return (record.get("run") or {}).get("best_ms")


def _speedup(best_ms, base_ms) -> float | None:
    if not best_ms or not base_ms:
        return None
    return float(base_ms) / float(best_ms)


def _status_reason(record) -> str:
    if record is None:
        return "no record on disk for this cell"
    if record.get("status") != "ok":
        detail = (record.get("errors") or [""])[0]
        return f"{record.get('status')}: {str(detail)[:110]}"
    notes = record.get("notes") or []
    pack = (record.get("leg") or {}).get("pack") or {}
    reason = pack.get("reason")
    return reason or (notes[0][:110] if notes else "ok")


def _reference_from_disk(out_dirs, model, seq_len, leg_key):
    """Saved logits of one leg (first ``--out`` dir that has them)."""
    for out_dir in out_dirs:
        path = lib.leg_paths(out_dir, model, seq_len, leg_key)["logits"]
        array = lib.read_logits(path)
        if array is not None:
            return array, str(path)
    return None, None


def _parity_table_lines(records, out_dirs) -> list[str]:
    """Parity recomputed from the saved ``.logits.npz`` files (leg-order agnostic)."""
    rows = []
    for model, seq_len, leg_key in _cell_order(records):
        record = _record_at(records, model, seq_len, leg_key)
        cand, _cand_path = _reference_from_disk(out_dirs, model, seq_len, leg_key)
        if cand is None:
            reason = (
                "saved logits disabled (--no-logits)"
                if record and record.get("status") == "ok"
                else f"leg did not produce logits ({lib.status_cell(record)})"
            )
            rows.append([model, seq_len, leg_key, "—", "—", "—", "—", reason])
            continue
        golden, golden_reason = lib.golden_reference(model, seq_len)
        references = [
            ("golden (torch/transformers)", golden["logits"] if golden is not None else None,
             golden_reason),
        ]
        for other_key in ("torch-cpu", "torch-rocm", "numpy-cpu"):
            if other_key == leg_key:
                continue
            array, path = _reference_from_disk(out_dirs, model, seq_len, other_key)
            references.append(
                (
                    other_key,
                    array,
                    f"no saved logits for the {other_key} leg" if array is None else path,
                )
            )
        for label, reference, note in references:
            if reference is None:
                rows.append([model, seq_len, leg_key, label, "—", "—", "—", note])
                continue
            stats = lib.compare_logits(cand, reference)
            if not stats.get("comparable"):
                rows.append(
                    [model, seq_len, leg_key, label, "—", "—", "—", stats.get("reason", "")]
                )
                continue
            rows.append(
                [
                    model,
                    seq_len,
                    leg_key,
                    label,
                    lib.fmt(stats["max_abs"]),
                    lib.fmt(stats["rel_to_absmax"]),
                    lib.fmt(stats["argmax_agreement"]),
                    "",
                ]
            )
    return rows


def render_report(records, out_dirs) -> str:
    """Render the benchmark markdown from the records on disk."""
    lines: list[str] = []
    add = lines.append
    ordered = _cell_order(records)
    models = [model for model in lib.MODELS if any(key[0] == model for key in records)]
    models += sorted({key[0] for key in records} - set(models))
    seq_lens = sorted({key[1] for key in records})
    legs = lib.sort_legs([key[2] for key in records])
    ok_cells = [record for record, _path in records.values() if record.get("status") == "ok"]

    add("# `etl_hf` benchmark — HuggingFace whole-model forward: iree / xla / numpy vs PyTorch")
    add("")
    add(
        "Generated by `bench_logs/hf_bench.py --report` from the JSON/`.npz` files under "
        "`" + "`, `".join(str(Path(d)) for d in out_dirs) + "` — **do not hand-edit**; re-run "
        "`--report` instead."
    )
    add("")
    git = next(
        (record.get("provenance", {}).get("git_commit") for record in ok_cells
         if record.get("provenance", {}).get("git_commit")),
        None,
    )
    add(
        f"Cells on disk: **{len(records)}** ({len(ok_cells)} ok, "
        f"{len(records) - len(ok_cells)} not ok) — models: "
        f"{', '.join(f'`{model}`' for model in models) or '—'}; seq lens: "
        f"{', '.join(str(seq) for seq in seq_lens) or '—'}; legs: "
        f"{', '.join(f'`{leg}`' for leg in legs) or '—'}. "
        f"Harness git commit: `{git or 'unknown'}`."
    )
    add("")
    add("## Method (what a number means)")
    add("")
    add("| aspect | how it is measured |")
    add("|---|---|")
    add(
        "| graph | the whole model forward is **one etl graph** (`etl_hf`): every weight an "
        "ordinary graph **input**, `input_ids` a `(1, seq_len)` int32 input, all shapes static. "
        "Packed rows take `pack_weights=True` (the weight tree concatenated on the host once "
        "and unpacked in-graph with static slices). |"
    )
    add(
        "| latency | best/median/mean/stdev over `--repeat` warm calls after `--warmup` "
        "warmups, via `etl_hf.run`. `first` is the very first call (builds the run plans). |"
    )
    add(
        "| device legs | the documented **device-resident same-device loop**: `place_params` + "
        "`place_ids` happen ONCE and the measured loop passes the same device tensors, so "
        "nothing is staged per call and the output never auto-copies to the host; "
        "`asdevicearray` calls during the measured loop must be **0** (see the checks column). |"
    )
    add(
        "| `+d2h` | a *separate* column: the same loop with one explicit "
        "`Tensor.to(Device('cpu', 0))` per call, i.e. what a host-facing caller pays. |"
    )
    add(
        "| restage (anti-pattern) | `place_params(host params)` **on every call** + run — the "
        "documented per-call host→device restaging trap, measured once for reference "
        "(device legs of `" + "`, `".join(ANTI_PATTERN_MODELS) + "` only). |"
    )
    add(
        "| tokens/s | `batch × seq_len / best latency` — a forward-pass rate, **not** decode "
        "throughput (no KV cache / incremental decode in v1: every call re-runs the graph). |"
    )
    add(
        "| speedups | `×torch-cpu` = `torch-cpu best ms / this row's best ms` — **>1 means "
        "this row is faster** than the PyTorch baseline on the same checkpoint and ids. |"
    )
    add(
        "| build | `build_total` is the public `etl_hf.build` (trace → lower → compile → load, "
        "the library defaults from `etl_hf.pipeline.resolve_build_options`). A `stages` mode "
        "row has the explicit `lower`/`compile`/`load` breakdown from `--stage-timings` (a "
        "second compilation). `cached` re-times an identical `build` call (memoised). |"
    )
    add(
        "| parity | `max_abs` / `rel_to_absmax` (= `max_abs / absmax(reference)`) of the "
        "saved logits vs a reference: the torch/transformers golden (only when its "
        "`input_ids` shape matches the cell), the torch legs, and the numpy interpreter leg. "
        "Recomputed by `--report` from the `*.logits.npz` files, so leg order does not "
        "matter. `argmax` is the per-row next-token argmax agreement rate. |"
    )
    add("")
    measurement = next(
        (record.get("measurement") for record, _ in records.values() if record.get("measurement")),
        {},
    )
    if measurement:
        add(
            f"Timing parameters of this run: **{measurement.get('repeat')} warm samples after "
            f"{measurement.get('warmup')} warmup call(s)**, batch `{measurement.get('batch')}` "
            "(per-cell overrides are recorded in each JSON under `measurement`)."
        )
        add("")

    # ---- environment -----------------------------------------------------
    add("## 1. Environment")
    add("")
    env_json = lib.load_env_json(out_dirs)
    if env_json is not None:
        add(
            f"Provisioned bench environment (`{env_json['_source']}`, produced by "
            "`bench_logs/hf_bench_env.py`):"
        )
        add("")
        flat = _flatten(env_json)
        rows = [[key, lib.fmt(value)] for key, value in list(flat.items()) if key != "_source"]
        if len(rows) > 120:
            rows = rows[:120] + [["…", f"{len(flat) - 120} more entries in {env_json['_source']}"]]
        lines.extend(lib.md_table(["key", "value"], rows))
    else:
        add(
            f"No `bench_logs/{lib.ENV_JSON_NAME}` found (the env-provisioning artifact is "
            "produced by `bench_logs/hf_bench_env.py`); versions below are what the measuring "
            "legs recorded themselves."
        )
    add("")
    versions: dict[str, set] = {}
    sources: dict[str, str] = {}
    for record, path in records.values():
        packages = ((record.get("provenance") or {}).get("env") or {}).get("packages") or {}
        for name, value in packages.items():
            if value is None:
                continue
            versions.setdefault(name, set()).add(str(value))
            sources.setdefault(name, Path(path).name)
    for record, _path in records.values():
        prov = record.get("provenance") or {}
        for name, value in (prov.get("env") or {}).get("packages", {}).items():
            if value is not None:
                versions.setdefault(f"leg:{name}", set()).add(str(value))
    rows = [
        [name, ", ".join(sorted(values))]
        for name, values in sorted(versions.items())
        if not name.startswith("leg:")
    ]
    if rows:
        lines.extend(lib.md_table(["package / variable", "version(s) seen in the legs"], rows))
        add("")
    flags: dict[str, set] = {}
    for record, _path in records.values():
        env = (record.get("provenance") or {}).get("env") or {}
        for name, value in env.items():
            if name.startswith(("ETL_", "XLA_FLAGS", "LD_LIBRARY_PATH", "CUDA_VISIBLE",
                                "HIP_VISIBLE", "ROCR_VISIBLE")) and value:
                flags.setdefault(name, set()).add(str(value))
    add(
        "Run-level environment of the measuring processes: "
        + (
            "; ".join(
                f"`{name}={', '.join(sorted(values))}`" for name, values in sorted(flags.items())
            )
            or "none set"
        )
    )
    add("")

    # ---- leg status ------------------------------------------------------
    add("## 2. Legs measured")
    add("")
    rows = []
    for model, seq_len, leg_key in ordered:
        record = _record_at(records, model, seq_len, leg_key)
        pack = ((record or {}).get("leg") or {}).get("pack") or {}
        rows.append(
            [
                model,
                seq_len,
                leg_key,
                lib.status_cell(record),
                ("packed" if pack.get("packed") else "unpacked") if pack else "—",
                _status_reason(record),
            ]
        )
    lines.extend(lib.md_table(
        ["model", "seq", "leg", "status", "weights", "note / reason"], rows
    ))
    add("")

    # ---- graph size (backend independent) --------------------------------
    add("## 3. Graph size (independent of the execution path)")
    add("")
    rows = []
    graph_reasons: list[str] = []
    for model in models:
        for seq_len in seq_lens:
            source = next(
                (
                    record
                    for leg_key in legs
                    if (record := _record_at(records, model, seq_len, leg_key)) is not None
                    and record.get("graph", {}).get("ir_ops")
                ),
                None,
            )
            if source is None:
                graph_reasons.append(f"`{model}` s{seq_len}: no leg traced this graph")
                rows.append([model, seq_len, "—", "—", "—", "—", "—", "—", "—", "—"])
                continue
            graph = source["graph"]
            modelinfo = source.get("model") or {}
            rows.append(
                [
                    model,
                    seq_len,
                    lib.fmt(modelinfo.get("param_count")),
                    lib.fmt(modelinfo.get("param_bytes")),
                    lib.fmt(graph.get("weight_inputs")),
                    lib.fmt(graph.get("unpacked_graph_inputs")),
                    lib.fmt(graph.get("packed_graph_inputs")),
                    lib.fmt(graph.get("ir_ops")),
                    lib.fmt(graph.get("ir_distinct_ops")),
                    lib.fmt(graph.get("stablehlo_bytes")),
                ]
            )
    lines.extend(lib.md_table(
        [
            "model",
            "seq",
            "params",
            "param bytes",
            "weight inputs",
            "graph inputs (unpacked)",
            "graph inputs (packed)",
            "IR ops",
            "distinct ops",
            "StableHLO bytes",
        ],
        rows,
    ))
    add("")
    add(
        "Counts come from `etl_hf.param_specs` / `etl_hf.pack_plan` and from the traced graph "
        "(`etl.trace` → `graph.verify()` → op walk), never from a constant. Packed graph input "
        "counts are `len(pack_plan(...).specs)`."
    )
    add("")

    # ---- build times -----------------------------------------------------
    add("## 4. Build times")
    add("")
    rows = []
    for model, seq_len, leg_key in ordered:
        record = _record_at(records, model, seq_len, leg_key)
        if record is None:
            rows.append([model, seq_len, leg_key, "—", "—", "—", "—", "—", "—", "—", "—", "no record"])
            continue
        pack = (record.get("leg") or {}).get("pack") or {}
        build = record.get("build") or {}
        if record.get("status") != "ok" or not build:
            detail = (record.get("errors") or [""])[0] if record.get("status") != "ok" else ""
            rows.append(
                [
                    model,
                    seq_len,
                    leg_key,
                    "packed" if pack.get("packed") else ("unpacked" if pack else "—"),
                    build.get("stage_timing") or "—",
                    "—", "—", "—", "—", "—", "—",
                    str(detail)[:120] or "no build measured",
                ]
            )
            continue
        stage_timing = build.get("stage_timing") or "—"
        rows.append(
            [
                model,
                seq_len,
                leg_key,
                "packed" if pack.get("packed") else ("unpacked" if pack else "—"),
                stage_timing,
                lib.fmt(build.get("trace_ms")),
                lib.fmt(build.get("lower_ms")),
                lib.fmt(build.get("compile_ms")),
                lib.fmt(build.get("load_ms")),
                lib.fmt(build.get("build_total_ms")),
                lib.fmt(build.get("build_total_cached_ms")),
                lib.fmt(build.get("artifact_bytes")),
            ]
        )
    lines.extend(lib.md_table(
        [
            "model",
            "seq",
            "leg",
            "weights",
            "stage timing",
            "trace ms",
            "lower ms",
            "compile ms",
            "load ms",
            "build total ms",
            "cached build ms",
            "artifact bytes",
        ],
        rows,
    ))
    add("")
    add(
        "`build total` is `etl_hf.build` (trace → lower → compile → load in one memoised call); "
        "`lower`/`compile`/`load` are only filled for `--stage-timings` rows, where the "
        "breakdown comes from an explicit `etl.lower`/`etl.compile`/`etl.load` on the same "
        "traced graph — a second compilation, which is why it is opt-in. `cached build` "
        "re-times an identical `build` call and shows the memoisation (weights are not part of "
        "the cache key: they are graph inputs)."
    )
    add("")

    # ---- latency ---------------------------------------------------------
    add("## 5. Warm latency, throughput, and speedups")
    add("")
    rows = []
    for model, seq_len, leg_key in ordered:
        record = _record_at(records, model, seq_len, leg_key)
        pack = ((record or {}).get("leg") or {}).get("pack") or {}
        pack_cell = ("packed" if pack.get("packed") else "unpacked") if pack else "—"
        run = (record or {}).get("run") or {}
        best = _best_ms(record)
        torch_cpu = _best_ms(_record_at(records, model, seq_len, "torch-cpu"))
        torch_rocm = _best_ms(_record_at(records, model, seq_len, "torch-rocm"))
        d2h = run.get("d2h") or {}
        if d2h.get("best_ms"):
            d2h_cell = lib.fmt(d2h.get("best_ms"))
        elif d2h.get("applicable") is False:
            d2h_cell = "n/a"
        else:
            d2h_cell = "—"
        checks = (record or {}).get("checks") or {}
        zero = checks.get("zero_asdevicearray_after_warmup")
        if zero is True:
            uploads = "0"
        elif zero is False:
            uploads = f"{checks.get('asdevicearray_calls')} (VIOLATION)"
        elif checks.get("checked"):
            uploads = f"{checks.get('asdevicearray_calls')} (host inputs, not asserted)"
        else:
            uploads = "n/a"
        reason = _status_reason(record)
        if best is not None and zero is False:
            reason = f"{reason}; VIOLATION: {checks.get('asdevicearray_calls')} asdevicearray call(s)"
        rows.append(
            [
                model,
                seq_len,
                leg_key,
                pack_cell,
                lib.fmt(run.get("first_call_ms")),
                lib.fmt(best),
                lib.fmt(run.get("median_ms")),
                f"{lib.fmt(run.get('mean_ms'))} ± {lib.fmt(run.get('stdev_ms'))}",
                lib.fmt(run.get("tokens_per_s")),
                lib.fmt_ratio(_speedup(best, torch_cpu)),
                lib.fmt_ratio(_speedup(best, torch_rocm)),
                d2h_cell if best is not None else "—",
                uploads,
                reason,
            ]
        )
    lines.extend(lib.md_table(
        [
            "model",
            "seq",
            "leg",
            "weights",
            "first ms",
            "best ms",
            "median ms",
            "mean ± stdev ms",
            "tokens/s",
            "×torch-cpu",
            "×torch-rocm",
            "+d2h best ms",
            "asdevicearray",
            "note / reason",
        ],
        rows,
    ))
    add("")
    add(
        "A `—` cell means the number was **not measured** (the reason is in the last column), "
        "never that it was zero. `×torch-cpu`/`×torch-rocm` are `torch best ms / this row's best "
        "ms`, so **>1× is faster than PyTorch**; when that torch baseline is absent for the "
        "cell the cell stays `—`. The `asdevicearray` column is the host→device upload count "
        "during the measured warm loop (device-resident legs must be `0`; `n/a` means the "
        "backend has no such entry point to probe)."
    )
    add("")

    # ---- anti-pattern ----------------------------------------------------
    anti = [
        (key, record["anti_pattern"])
        for key in ordered
        if (record := _record_at(records, *key)) is not None and record.get("anti_pattern")
    ]
    if anti:
        add("## 5b. The per-call restaging anti-pattern (reference only)")
        add("")
        rows = []
        for (model, seq_len, leg_key), block in anti:
            record = _record_at(records, model, seq_len, leg_key)
            best = _best_ms(record)
            rows.append(
                [
                    model,
                    seq_len,
                    leg_key,
                    lib.fmt(block.get("best_ms")),
                    lib.fmt(block.get("median_ms")),
                    lib.fmt(best),
                    lib.fmt_ratio(_speedup(block.get("best_ms"), best)),
                    block.get("detail"),
                ]
            )
        lines.extend(lib.md_table(
            [
                "model",
                "seq",
                "leg",
                "restage best ms",
                "restage median ms",
                "same-device loop best ms",
                "slowdown",
                "what the anti-pattern is",
            ],
            rows,
        ))
        add("")
    # ---- parity ----------------------------------------------------------
    add("## 6. Parity")
    add("")
    lines.extend(lib.md_table(
        [
            "model",
            "seq",
            "leg",
            "reference",
            "max_abs",
            "rel_to_absmax",
            "argmax agree",
            "note",
        ],
        _parity_table_lines(records, out_dirs),
    ))
    add("")
    add(
        "`max_abs` is an absolute logit difference, `rel_to_absmax` normalises it by the "
        "reference's own absmax. A reference whose saved ids shape does not match the cell, "
        "or that was never run, renders `—` with the reason. Bad parity is a **result**, "
        "reported as measured — nothing here is tuned away."
    )
    add("")

    # ---- failures --------------------------------------------------------
    add("## 7. Legs that could not run, and why")
    add("")
    broken = [
        (key, _record_at(records, *key))
        for key in ordered
        if (_record_at(records, *key) or {}).get("status") != "ok"
    ]
    entries = [
        (key, record) for key, record in broken if record is not None
    ]
    entries += [
        (key, {"status": "missing", "errors": ["no record on disk for this cell"]})
        for key in ((model, seq_len, leg_key) for model in models for seq_len in seq_lens for leg_key in legs)
        if key not in records
    ]
    if not entries:
        add("None — every planned leg produced an `ok` record.")
    else:
        rows = []
        for (model, seq_len, leg_key), record in entries:
            process = record.get("process") or {}
            message = (record.get("errors") or record.get("notes") or [""])[0]
            rows.append(
                [
                    model,
                    seq_len,
                    leg_key,
                    record.get("status"),
                    lib.fmt(process.get("returncode")),
                    lib.fmt(process.get("signal_name") or process.get("signal")),
                    lib.fmt(process.get("stage") or record.get("failed_stage")),
                    str(message)[:240],
                ]
            )
        lines.extend(lib.md_table(
            [
                "model",
                "seq",
                "leg",
                "status",
                "returncode",
                "signal",
                "stage",
                "exact error / reason",
            ],
            rows,
        ))
        add("")
        details = []
        for (model, seq_len, leg_key), record in entries:
            process = record.get("process") or {}
            tail = process.get("stderr_tail") or []
            if not tail:
                continue
            details.append(f"**`{model}` s{seq_len} `{leg_key}`** stderr tail:")
            details.append("")
            details.append("```")
            details.extend(str(line) for line in tail)
            details.append("```")
            details.append("")
        if details:
            add("Verbatim stdio of the failing legs:")
            add("")
            lines.extend(details)
    add("")

    # ---- notes -----------------------------------------------------------
    add("## 8. Notes")
    add("")
    add(
        "* One subprocess per `(model, seq_len, leg)`. A leg that segfaults, times out or "
        "writes no record is still reported (section 7 quotes its returncode, signal, last "
        "`[stage]` marker and stderr tail). The xla/ROCm PJRT plugin aborts at interpreter "
        "exit **after** the work succeeds, so legs end with `os._exit(0)` once their record "
        "is on disk."
    )
    add(
        "* `--pack auto` packs only when it is needed (a compiled leg above the ~161-consumed-"
        "input IREE runtime limit — the 124M `gpt2`) or measured-beneficial (a device leg "
        "staging many weight leaves). The per-row reason is in section 2."
    )
    add(
        "* Compile options are the `etl_hf.pipeline.resolve_build_options` defaults (all-ROCm → "
        "`opt_level=\"O1\"`, llvm-cpu → `opt_level=\"O3\"` + `--iree-llvmcpu-target-cpu=host` + "
        "`--iree-opt-data-tiling`); the resolved set is recorded in every leg JSON."
    )
    add(
        "* The torch legs are the baseline, not the reference: parity versus the stored "
        "torch/transformers golden is the absolute check (only where the golden's ids shape "
        "matches the cell)."
    )
    graph_note = [reason for reason in graph_reasons]
    if graph_note:
        add("* Graph sizes missing for: " + "; ".join(graph_note) + ".")
    add("")
    return "\n".join(lines).rstrip() + "\n"


def _flatten(payload: dict, prefix: str = "") -> dict:
    """Flatten a nested JSON object into dotted ``key -> scalar`` pairs."""
    out: dict = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            out[name] = ", ".join(str(item) for item in value)
        else:
            out[name] = value
    return out


def report_main(args) -> int:
    """``--report``: regenerate the markdown from the records (no measuring)."""
    records = collect_records(args.out)
    if not records:
        print(
            f"[report] no leg records found under "
            f"{', '.join(str(Path(d)) for d in args.out)} — run the legs first",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        print(f"[report] would read {len(records)} record(s) and write {args.report_out}")
        return 0
    markdown = render_report(records, args.out)
    target = Path(args.report_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown)
    print(f"[report] wrote {target} from {len(records)} record(s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The CLI. ``--leg`` is the child mode; everything else is driver/report."""
    parser = argparse.ArgumentParser(
        prog="hf_bench.py",
        description=(
            "Benchmark HuggingFace checkpoints through etl_hf (iree / xla / numpy) against "
            "PyTorch. One subprocess per (model, seq_len, leg); --report renders the "
            "markdown from the JSON/.npz files only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # CPU smoke test, no torch needed\n"
            "  PYTHONPATH=$(git rev-parse --show-toplevel) uv run --extra iree python \\\n"
            "      bench_logs/hf_bench.py --leg numpy-cpu --models tiny_llama --seq-lens 32\n"
            "  # full matrix from the provisioned bench venv (see bench_logs/hf_bench_env.py)\n"
            "  PYTHONPATH=$(git rev-parse --show-toplevel) \\\n"
            "      $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase cpu\n"
            "  PYTHONPATH=$(git rev-parse --show-toplevel) \\\n"
            "      $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --phase gpu\n"
            "  $TMPDIR/hfbench-venv/bin/python bench_logs/hf_bench.py --report\n"
        ),
    )
    parser.add_argument(
        "--models",
        default=",".join(lib.MODELS),
        help=f"comma-separated checkpoint slugs (default: {','.join(lib.MODELS)})",
    )
    parser.add_argument(
        "--seq-lens", default="32", help="comma-separated sequence lengths (default: 32)"
    )
    parser.add_argument(
        "--legs", default=None, help=f"comma-separated leg keys ({', '.join(lib.LEG_ORDER)})"
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="phase leg set(s): cpu = numpy-cpu,iree-llvmcpu,torch-cpu; "
        "gpu = iree-rocm,xla-rocm,torch-rocm (run the two phases separately, never "
        "concurrently — the GPU is shared)",
    )
    parser.add_argument(
        "--out",
        action="append",
        default=None,
        metavar="DIR",
        help=f"record directory; repeatable (default: {lib.DEFAULT_DATA_DIR}). The first is "
        "the write target; --report reads (and merges) all of them, earlier ones winning",
    )
    parser.add_argument(
        "--leg",
        default=None,
        metavar="LEG",
        help="CHILD MODE: measure this single leg in this process (what the driver spawns)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="warmup calls (default: 2)")
    parser.add_argument("--repeat", type=int, default=5, help="measured calls (default: 5)")
    parser.add_argument(
        "--pack",
        choices=("auto", "on", "off"),
        default="auto",
        help="weight packing: auto = packed only where required/beneficial (documented in "
        "each record's leg.pack.reason), on, off (default: auto)",
    )
    parser.add_argument(
        "--logits", dest="logits", action="store_true", default=True,
        help="save each leg's logits as *.logits.npz (default)",
    )
    parser.add_argument(
        "--no-logits", dest="logits", action="store_false",
        help="do not save logits (cross-leg parity then cannot be recomputed)",
    )
    parser.add_argument(
        "--stage-timings",
        action="store_true",
        help="additionally time explicit etl.lower/etl.compile/etl.load (a SECOND "
        "compilation) instead of only build_total",
    )
    parser.add_argument(
        "--anti-pattern",
        dest="anti_pattern",
        action="store_true",
        default=True,
        help="measure the per-call restage anti-pattern on the labelled device cells (default)",
    )
    parser.add_argument(
        "--no-anti-pattern",
        dest="anti_pattern",
        action="store_false",
        help="skip the per-call restage anti-pattern measurement",
    )
    parser.add_argument(
        "--leg-timeout",
        type=float,
        default=900.0,
        metavar="SECONDS",
        help="per-leg subprocess timeout (default: 900)",
    )
    parser.add_argument(
        "--report", action="store_true", help="render the markdown from the records only"
    )
    parser.add_argument(
        "--report-out",
        default=str(lib.DEFAULT_REPORT_PATH),
        help=f"markdown target of --report (default: {lib.DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned leg commands (or the report inputs) and exit",
    )
    return parser


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def main(argv=None) -> int:
    """Parse the CLI and dispatch (child leg / report / driver)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out = list(args.out) if args.out else [str(lib.DEFAULT_DATA_DIR)]
    args.models = _split_csv(args.models)
    unknown = [model for model in args.models if model not in lib.GOLDEN_SLUGS]
    if unknown:
        parser.error(
            f"unknown model(s) {unknown}; known: {', '.join(lib.MODELS)} "
            "(add the checkpoint under $ETL_HF_MODELS_DIR to benchmark a new one)"
        )
    try:
        args.seq_lens = [int(part) for part in _split_csv(args.seq_lens)]
    except ValueError:
        parser.error(f"--seq-lens must be integers, got {args.seq_lens!r}")
    if not args.seq_lens:
        parser.error("--seq-lens must contain at least one length")
    if args.leg:
        return run_child(args)
    if args.report:
        return report_main(args)
    if args.dry_run:
        return drive(args)
    return drive(args)


if __name__ == "__main__":
    sys.exit(main())
