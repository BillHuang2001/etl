#!/usr/bin/env python3
"""Shared helpers for :mod:`bench_logs/hf_bench.py` — the ``etl_hf`` benchmark harness.

This module is **stdlib only at import time**: ``numpy`` / ``etl`` / ``etl_hf`` /
``iree`` / ``torch`` / ``transformers`` are imported lazily inside the functions that
need them, so ``--report`` (which needs nothing but JSON + ``numpy`` to read the
``.npz`` logits) never pays for a backend import.

What lives here (and therefore *not* in the CLI module):

* the **leg table** (:class:`LegSpec`, :data:`LEGS`, :data:`PHASES`) — one entry per
  execution path, plus the model → golden-fixture-slug map, the human labels of
  :data:`LEG_DISPLAY` and the ``ETL_VULKAN_ICD`` lookup (:func:`vulkan_icd`) the
  ``iree-vulkan`` leg needs;
* **checkpoint / golden resolution** (never downloads);
* the **record file naming convention** (:func:`leg_paths`, :func:`read_record`) —
  one JSON per ``(model, seq_len, leg)`` cell, so cross-leg parity can be recomputed
  later from the ``.logits.npz`` files regardless of leg execution order;
* **provenance helpers** (hostname / timestamp / git commit / package versions / env
  snapshot) so every record can explain itself;
* **timing + parity metrics** (best/median/mean/stdev, ``max_abs``,
  ``rel_to_absmax``, argmax agreement) shared by the measuring legs and the report;
* the **``asdevicearray`` zero-copy probe** used to prove the device-resident loop
  performs no host round-trips, and
* the **markdown formatting primitives** used by ``--report``.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Directory holding this file (``bench_logs/``) and the repository root above it.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
#: Where legs write their records by default (``--out``), and where ``--report`` reads.
DEFAULT_DATA_DIR = HERE / "hf_bench_data"
#: Default markdown target of ``--report``.
DEFAULT_REPORT_PATH = HERE / "hf_bench_results.md"
#: Torch-free golden logits produced by ``bench_logs/probe_hf_goldens.py``.
GOLDENS_DIR = HERE / "hf_goldens"
#: Version table written by ``bench_logs/hf_bench_env.py`` (provisioning artifact).
ENV_JSON_NAME = "hf_bench_env.json"
#: Environment variable naming the **Vulkan ICD json** the ``iree-vulkan`` leg needs.
#: The harness never hard-codes a machine path: the runner points this at the renamed
#: lavapipe ICD (see ``bench_logs/vulkan_findings.md``) and ``hf_bench.child_env``
#: forwards it to the leg subprocess as the ``VK_ICD_FILENAMES`` the Vulkan loader
#: reads. A vulkan leg requested without it fails with an explicit error naming this
#: variable (it is a RECORD, never a silent omission) — the alternative would be
#: silently enumerating 0 devices or, worse, letting the loader dlopen the wedged
#: radeon ICD.
VULKAN_ICD_ENV = "ETL_VULKAN_ICD"
#: Human labels for legs whose bare key would misrepresent what was measured. Used by
#: ``--report`` wherever a leg is named in prose or in a section heading; tables keep
#: the key itself so every row stays machine-mappable to a record file.
LEG_DISPLAY: dict[str, str] = {
    "iree-vulkan": "iree-vulkan (lavapipe, software Vulkan on CPU)",
    "iree-vulkan-o1": (
        "iree-vulkan-o1 (lavapipe, software Vulkan on CPU; hand-picked "
        '`opt_level="O1"`)'
    ),
}

#: Bumped whenever the record schema below changes in a way a reader cares about.
SCHEMA = 1

#: IREE's Python runtime SIGSEGVs (no Python exception) when a graph *consumes* more
#: than ~161 distinct inputs — documented as a Known Issue in ``etl_hf/CONTEXT.md``.
#: Only an UNPACKED graph gets near it: the 124M ``gpt2`` declares 197 inputs while
#: its packed form declares 4. A compiled leg whose graph exceeds this is packed
#: (``--pack auto``) instead of being crashed into.
INPUT_LIMIT = 161

#: Checkpoint slug (``$TMPDIR/etl_hf_models/<slug>``) -> golden fixture slug
#: (``bench_logs/hf_goldens/<slug>.npz``).  The golden ``.npz`` holds the real
#: torch/transformers ``logits`` (f32) **and** the exact ``input_ids`` the oracle used.
GOLDEN_SLUGS: dict[str, str] = {
    "tiny_llama": "tiny_random_llama",
    "tiny_mistral": "tiny_random_mistral",
    "tiny_gpt2": "tiny_gpt2",
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
}
#: Default model order (deterministic everywhere: CLI defaults, report ordering).
MODELS: tuple[str, ...] = tuple(GOLDEN_SLUGS)


# ---------------------------------------------------------------------------
# the leg table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LegSpec:
    """One row of the matrix: *how* a leg runs one model forward.

    Attributes:
        key: the leg key used on the CLI and in every file/record name.
        kind: ``"etl"`` (an ``etl_hf`` build + run) or ``"torch"`` (a
            ``transformers`` baseline).
        backend: ``etl`` backend name (``"numpy"`` / ``"iree"`` / ``"xla"``).
        device_kind: ``etl`` device kind, or ``None`` for the host (cpu) executable.
        device_index: device index for ``device_kind``.
        options: options handed to :func:`etl_hf.pipeline.resolve_build_options`.
            Deliberately minimal — the *library defaults* (opt_level, the llvm-cpu
            compile flags) are what gets measured, never a hand-picked copy.
        torch_device: torch device string for a ``torch`` leg.
        torch_compile: wrap the model in ``torch.compile``.
    """

    key: str
    kind: str
    backend: str | None = None
    device_kind: str | None = None
    device_index: int = 0
    options: dict = field(default_factory=dict)
    torch_device: str | None = None
    torch_compile: bool = False

    @property
    def device_resident(self) -> bool:
        """True when the executable lives on an accelerator (non-cpu) device."""
        return self.kind == "etl" and self.device_kind not in (None, "cpu")

    @property
    def device_string(self) -> str:
        """Human/record spelling of the device this leg runs on."""
        if self.kind == "torch":
            return str(self.torch_device)
        if self.device_kind is None:
            return "cpu (host)"
        return f"{self.device_kind}:{self.device_index}"

    def etl_device(self):
        """The :class:`etl.core.Device` for this leg, or ``None`` for a host one."""
        if self.kind != "etl" or self.device_kind is None:
            return None
        import etl

        return etl.core.Device(self.device_kind, self.device_index)


LEGS: dict[str, LegSpec] = {
    "numpy-cpu": LegSpec(key="numpy-cpu", kind="etl", backend="numpy"),
    "iree-llvmcpu": LegSpec(
        key="iree-llvmcpu",
        kind="etl",
        backend="iree",
        options={"target_backends": ["llvm-cpu"]},
    ),
    "iree-rocm": LegSpec(
        key="iree-rocm",
        kind="etl",
        backend="iree",
        device_kind="rocm",
        options={"target_backends": ["rocm"]},
    ),
    "xla-rocm": LegSpec(
        key="xla-rocm",
        kind="etl",
        backend="xla",
        device_kind="rocm",
    ),
    # IREE's **Vulkan** HAL.  The only Vulkan device on this host is **lavapipe** (Mesa's
    # software Vulkan implementation running on the CPU — a correctness/perf datapoint,
    # never a GPU number), reached through a renamed ICD pinned by ``VK_ICD_FILENAMES``,
    # which the driver forwards from :data:`VULKAN_ICD_ENV`.
    # ``"vulkan"`` is a **first-class ``etl`` device kind** (``etl.core.Device``), so the
    # leg needs no shim and no hand-declared compile target: ``etl.pipeline`` infers
    # ``target_backends=["vulkan-spirv"]`` from the device for the ``iree`` backend, so
    # options stay EMPTY and the library defaults apply (``etl_hf.pipeline`` injects
    # ``opt_level="O3"`` for a non-rocm target set, which compiles fine for
    # vulkan-spirv).  The SPIR-V *profile* is a separate knob — the ``vulkan_target``
    # compile option (``--iree-vulkan-target=<profile>``) — left unset here, i.e. IREE's
    # default ``vp_android_baseline_2022``.
    "iree-vulkan": LegSpec(
        key="iree-vulkan",
        kind="etl",
        backend="iree",
        device_kind="vulkan",
        device_index=0,
    ),
    # The SAME Vulkan device and path, with the ONE hand-picked compile option in this
    # matrix: ``opt_level="O1"``.  The library default for a non-ROCm target set is
    # ``O3``, and ``--iree-opt-level=O3`` cannot translate the 124M-class graphs for
    # ``vulkan-spirv`` (IREE's default Vulkan target profile,
    # ``vp_android_baseline_2022``, allows 16384 B of workgroup memory):
    # ``distilgpt2``/``gpt2`` at ``seq_len`` 32 fail with ``function
    # 'main_dispatch_0_reduction_32x768_f32' uses 196864 bytes of shared memory;
    # exceeded the limit of 16384 bytes`` (the fused LayerNorm-mean/variance
    # reduction) and at ``seq_len`` 128 with ``failed to legalize operation
    # 'vector.from_elements'`` (the packed in-graph gather).  ``O1`` (and ``O0``,
    # whose vmfb is byte-identical) compiles all four cells and they run bit-plausibly
    # on lavapipe — see ``bench_logs/vulkan_findings.md``.  ``vulkan_target`` is NOT
    # used (no other flag is): the SPIR-V profile stays at IREE's default, so this leg
    # differs from ``iree-vulkan`` in exactly one compile option, and every record says
    # so (``record["vulkan"]["compile_options_hand_picked"]``) — which is why it is a
    # separate key from ``iree-vulkan`` (whose O3 cells stay recorded as failures,
    # unweakened).
    "iree-vulkan-o1": LegSpec(
        key="iree-vulkan-o1",
        kind="etl",
        backend="iree",
        device_kind="vulkan",
        device_index=0,
        options={"opt_level": "O1"},
    ),
    # Additive variant of ``xla-rocm``: the ONE place the harness passes a
    # precision option, because XLA/ROCm's default fp32 ``dot`` runs at
    # TF32-class precision (10-bit mantissa) while the explicit
    # ``dot_precision="HIGHEST"`` restores true fp32.  It flows through the
    # ordinary options plumbing (``resolve_build_options`` copies it, the xla
    # adapter declares it a reserved lower option) — nothing else differs from
    # ``xla-rocm``, so the pair isolates the precision decision.
    "xla-rocm-fp32": LegSpec(
        key="xla-rocm-fp32",
        kind="etl",
        backend="xla",
        device_kind="rocm",
        options={"dot_precision": "HIGHEST"},
    ),
    "torch-cpu": LegSpec(key="torch-cpu", kind="torch", torch_device="cpu"),
    "torch-rocm": LegSpec(key="torch-rocm", kind="torch", torch_device="cuda:0"),
    "torchcompile-cpu": LegSpec(
        key="torchcompile-cpu", kind="torch", torch_device="cpu", torch_compile=True
    ),
    "torchcompile-rocm": LegSpec(
        key="torchcompile-rocm", kind="torch", torch_device="cuda:0", torch_compile=True
    ),
}
#: Deterministic leg ordering: the ``etl`` paths first (``numpy-cpu`` before the
#: compiled ones so it is normally the first reference on disk), then torch.
LEG_ORDER: tuple[str, ...] = (
    "numpy-cpu",
    "iree-llvmcpu",
    "iree-rocm",
    "iree-vulkan",
    "iree-vulkan-o1",
    "xla-rocm",
    "xla-rocm-fp32",
    "torch-cpu",
    "torch-rocm",
    "torchcompile-cpu",
    "torchcompile-rocm",
)
#: ``--phase`` expansions.  ``cpu`` and ``gpu`` are run **separately and
#: sequentially** because the GPU is shared with other work on this host.  ``vulkan`` is
#: its own phase for the same reason plus one more: lavapipe is a CPU software device,
#: so a vulkan leg must never overlap a leg that is meant to be measured on the CPU
#: (``cpu``) — it would steal the cores both measurements assume they have.
PHASES: dict[str, tuple[str, ...]] = {
    "cpu": ("numpy-cpu", "iree-llvmcpu", "torch-cpu"),
    "gpu": ("iree-rocm", "xla-rocm", "torch-rocm"),
    "vulkan": ("iree-vulkan", "iree-vulkan-o1"),
}
#: Legs that exist for a documented, additive *comparison* and are therefore never
#: part of a phase or of the default leg set — they run only when named explicitly
#: with ``--legs`` (``xla-rocm-fp32`` isolates the TF32-vs-true-fp32 ``dot_precision``
#: decision against ``xla-rocm``; nothing else about it differs).
OPT_IN_LEGS: frozenset[str] = frozenset({"xla-rocm-fp32"})


def leg_spec(key: str) -> LegSpec:
    """Look up a leg by key, with a message that lists the valid keys."""
    try:
        return LEGS[key]
    except KeyError:
        raise SystemExit(
            f"unknown leg {key!r}; known legs: {', '.join(LEG_ORDER)} "
            f"(phases: {', '.join(sorted(PHASES))})"
        ) from None


def sort_legs(keys) -> list[str]:
    """Order leg keys deterministically (``LEG_ORDER``, unknown keys last)."""
    rank = {key: index for index, key in enumerate(LEG_ORDER)}
    return sorted(set(keys), key=lambda key: (rank.get(key, len(rank)), key))


def leg_display(key: str) -> str:
    """A leg's human label — the bare key unless :data:`LEG_DISPLAY` qualifies it."""
    return LEG_DISPLAY.get(key, key)


def vulkan_icd(var: str = VULKAN_ICD_ENV) -> str:
    """The Vulkan ICD json path the runner exported, or an explicit error naming it.

    The path is **machine-local** (a renamed lavapipe ICD under a scratch dir), so it
    is never hard-coded here: the harness reads :data:`VULKAN_ICD_ENV` in the driver
    and re-exports it to the leg subprocess as ``VK_ICD_FILENAMES``.

    Raises:
        RuntimeError: the variable is unset/blank or points at a missing file — the
            message names the variable and what it must contain.
    """
    value = (os.environ.get(var) or "").strip()
    if not value:
        raise RuntimeError(
            f"the iree-vulkan leg needs {var} set to the Vulkan ICD json to use "
            "(the renamed lavapipe ICD of bench_logs/vulkan_findings.md), and it is "
            f"empty/unset; export {var}=/path/to/lavapipe_icd.json — the driver forwards "
            "it to the leg as VK_ICD_FILENAMES"
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise RuntimeError(
            f"{var} points at {value!r}, which is not a readable file; it must be the "
            "Vulkan ICD json of the device the leg should enumerate "
            "(bench_logs/vulkan_findings.md builds one for the renamed lavapipe ICD)"
        )
    return str(path)


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


def checkpoint_dir(model: str) -> Path:
    """Resolve a cached checkpoint directory, or raise naming what is missing.

    This harness never downloads: a missing checkpoint is an explicit error that
    says which path was expected (the runner provisions the cache).
    """
    directory = models_root() / model
    have_config = (directory / "config.json").is_file()
    weights = [
        name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (directory / name).is_file()
    ]
    if not have_config or not weights:
        raise FileNotFoundError(
            f"no complete checkpoint for {model!r} under {directory} "
            f"(config.json={have_config}, weights={weights or 'none'}); this harness "
            f"never downloads — populate {models_root()} or set $ETL_HF_MODELS_DIR"
        )
    return directory


def golden_path(model: str) -> Path:
    """Path of the torch/transformers golden fixture for a checkpoint slug."""
    return GOLDENS_DIR / f"{GOLDEN_SLUGS[model]}.npz"


def load_golden(model: str) -> dict:
    """Load ``hf_goldens/<slug>.npz`` (+ its meta) as the parity oracle."""
    import numpy as np

    path = golden_path(model)
    if not path.is_file():
        raise FileNotFoundError(f"missing golden fixture {path}")
    meta_path = path.with_name(path.name.replace(".npz", ".meta.json"))
    with np.load(path) as blob:
        logits = np.asarray(blob["logits"], dtype=np.float32)
        ids = np.asarray(blob["input_ids"], dtype=np.int32)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {"logits": logits, "input_ids": ids, "meta": meta, "path": str(path)}


def golden_reference(model: str, seq_len: int) -> tuple[dict | None, str]:
    """The golden fixture **only if its ``input_ids`` shape matches this cell**.

    The tiny goldens were produced at ``seq_len=8`` and ``distilgpt2``/``gpt2`` at
    ``seq_len=32``, so a golden is usable for parity only when its ``input_ids`` shape
    equals ``(1, seq_len)``.  Returns ``(fixture_or_None, reason)``; the reason is
    always populated (it explains a ``—`` cell in the report).
    """
    try:
        fixture = load_golden(model)
    except Exception as exc:  # missing / unreadable fixture
        return None, f"no golden fixture ({type(exc).__name__}: {exc})"
    shape = tuple(int(dim) for dim in fixture["input_ids"].shape)
    if shape != (1, int(seq_len)):
        return None, (
            f"golden fixture is {(1, int(seq_len))[0]}x{shape[1]} (input_ids {shape}) "
            f"but this cell is 1x{int(seq_len)} — compare against the torch / numpy legs"
        )
    return fixture, "golden fixture (torch/transformers oracle)"


def reference_ids(model: str, seq_len: int, vocab_size: int) -> tuple[object, str]:
    """The token ids every leg of one (model, seq_len) cell must use.

    Reuses the golden fixture's own ids when the golden matches this cell (so a
    golden parity comparison is apples-to-apples), else synthesizes the documented
    recipe ``input_ids[0, i] = (i * 7 + 3) % vocab_size``.

    Returns:
        ``(ids, source)`` where ``ids`` is an ``int32`` ``(1, seq_len)`` array.
    """
    import numpy as np

    fixture, _reason = golden_reference(model, seq_len)
    if fixture is not None:
        return np.asarray(fixture["input_ids"], dtype=np.int32), "hf_goldens input_ids"
    ids = np.asarray(
        [[(i * 7 + 3) % int(vocab_size) for i in range(int(seq_len))]], dtype=np.int32
    )
    return ids, f"synthesized (i*7+3) % vocab_size={int(vocab_size)}"


# ---------------------------------------------------------------------------
# record naming / IO
# ---------------------------------------------------------------------------
def leg_base(model: str, seq_len: int, leg: str) -> str:
    """File stem of one leg cell: ``<model>__s<seq_len>__<leg>``."""
    return f"{model}__s{int(seq_len)}__{leg}"


def leg_paths(out_dir, model: str, seq_len: int, leg: str) -> dict[str, Path]:
    """Every file one leg cell may write (``json`` / ``error`` / ``logits``)."""
    directory = Path(out_dir)
    base = leg_base(model, seq_len, leg)
    return {
        "json": directory / f"{base}.json",
        "error": directory / f"{base}.error.json",
        "logits": directory / f"{base}.logits.npz",
    }


def read_record(out_dirs, model: str, seq_len: int, leg: str):
    """Read one leg's record from the first directory that has it.

    A record is ``<base>.json`` (written by the leg itself, whatever its status) or —
    when the leg subprocess died before it could write anything — the explicit
    ``<base>.error.json`` the driver synthesizes.

    Returns:
        ``(record_or_None, path_or_None)``.
    """
    for out_dir in out_dirs:
        paths = leg_paths(out_dir, model, seq_len, leg)
        for key in ("json", "error"):
            if paths[key].is_file():
                return json.loads(paths[key].read_text()), paths[key]
    return None, None


def write_json(path: Path, payload: dict) -> Path:
    """Write one record atomically-ish (parents created, deterministic layout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


def _json_default(value):
    """JSON fallback for numpy scalars / paths / etl objects in records."""
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (set, frozenset, tuple)):
        return sorted(str(item) for item in value)
    return repr(value)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp of when a record was produced."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str | None:
    """``git rev-parse HEAD`` of the checkout the harness runs from, if available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def package_version(name: str) -> str | None:
    """Installed distribution version, or ``None`` when the package is absent."""
    import importlib.metadata as md

    try:
        return md.version(name)
    except Exception:
        return None


def adapter_versions() -> dict:
    """Versions of everything that can influence a compiled leg."""
    return {
        "numpy": package_version("numpy"),
        "etl": package_version("etl"),
        "etl_hf": package_version("etl_hf"),
        "iree-base-compiler": package_version("iree-base-compiler"),
        "iree-base-runtime": package_version("iree-base-runtime"),
        "jaxlib": package_version("jaxlib"),
        "jax-rocm7-pjrt": package_version("jax-rocm7-pjrt"),
        "jax-cuda12-pjrt": package_version("jax-cuda12-pjrt"),
        "apache-tvm": package_version("apache-tvm"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "safetensors": package_version("safetensors"),
        "ETL_PJRT_PLUGIN": os.environ.get("ETL_PJRT_PLUGIN"),
    }


def env_snapshot() -> dict:
    """Environment + versions of the process doing the measuring (recorded per leg)."""
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": adapter_versions(),
        "ETL_BACKEND": os.environ.get("ETL_BACKEND"),
        "ETL_TARGET_BACKENDS": os.environ.get("ETL_TARGET_BACKENDS"),
        "ETL_OPT_LEVEL": os.environ.get("ETL_OPT_LEVEL"),
        "ETL_DEVICE": os.environ.get("ETL_DEVICE"),
        "ETL_HF_MODELS_DIR": os.environ.get("ETL_HF_MODELS_DIR"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "TMPDIR": os.environ.get("TMPDIR"),
    }


def provenance(model: str, seq_len: int, leg: str, command=None) -> dict:
    """The provenance block every record starts with (including the env snapshot)."""
    return {
        "model": model,
        "golden_slug": GOLDEN_SLUGS.get(model),
        "seq_len": int(seq_len),
        "leg": leg,
        "timestamp": utc_timestamp(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "command": list(command) if command else list(sys.argv),
        "env": env_snapshot(),
    }


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def ms(seconds: float) -> float:
    """Seconds → milliseconds, rounded for stable JSON."""
    return round(float(seconds) * 1e3, 4)


def timeit(fn, *args, **kwargs):
    """Return ``(result, seconds)`` around one call."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def summarize(samples) -> dict:
    """best / median / mean / stdev of a latency sample list (milliseconds)."""
    if not samples:
        return {"n": 0}
    values = [float(value) * 1e3 for value in samples]
    return {
        "n": len(values),
        "best_ms": round(min(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "mean_ms": round(statistics.fmean(values), 4),
        "stdev_ms": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def tokens_per_s(batch: int, seq_len: int, best_ms: float | None) -> float | None:
    """``batch * seq_len / best latency`` — a forward-pass rate, not decode throughput."""
    if not best_ms:
        return None
    return round(int(batch) * int(seq_len) / (float(best_ms) / 1e3), 2)


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process (Linux ``ru_maxrss`` is in KiB)."""
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# graph metrics
# ---------------------------------------------------------------------------
def count_ops(graph) -> dict[str, int]:
    """Op-name histogram of a traced graph, walking nested regions.

    Same walk as ``etl_hf/tests/test_export.py::_count_ops``; written generically so a
    control-flow model cannot under-count.
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


def spec_size(spec) -> int:
    """Element count of a (static, int-dim) ``TensorSpec``."""
    total = 1
    for dim in spec.shape:
        total *= int(dim)
    return total


def spec_tree_summary(specs) -> dict:
    """Weight-input count / element count / bytes of a params ``TensorSpec`` tree."""
    import numpy as np

    elements = 0
    nbytes = 0
    for spec in specs.values():
        size = spec_size(spec)
        elements += size
        nbytes += size * np.dtype(str(spec.dtype)).itemsize
    return {
        "weight_inputs": len(specs),
        "graph_inputs": len(specs) + 1,  # + input_ids
        "param_elements": elements,
        "param_bytes": nbytes,
        "keys": sorted(specs),
    }


def artifact_bytes(artifact) -> int | None:
    """Byte size of a compiled artifact payload, when the adapter exposes one.

    IREE records its compiled form as a base64 VMFB inside a plain dict; the numpy
    backend's payload is serialized IR text.  Anything unrecognised reports ``None``
    rather than guessing.
    """
    payload = getattr(artifact, "payload", None)
    if isinstance(payload, dict):
        encoded = payload.get("vmfb_base64")
        if isinstance(encoded, str):
            return len(encoded.rstrip("=")) * 3 // 4
        if payload.get("format") == "iree-vmfb-segments":
            return None
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    if isinstance(payload, str):
        return len(payload.encode())
    return None


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------
def compare_logits(candidate, reference) -> dict:
    """Parity of one leg's logits against a reference array.

    Returns ``max_abs`` (largest absolute difference), ``rel_to_absmax``
    (``max_abs / absmax(reference)``), the mean absolute difference and the argmax
    agreement rate (per-row next-token argmax), plus both absmax values.  A shape
    mismatch is reported as ``comparable: False`` with the reason — never a number.
    """
    import numpy as np

    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    out: dict = {"comparable": False, "shape": list(cand.shape), "ref_shape": list(ref.shape)}
    if cand.shape != ref.shape:
        out["reason"] = f"shape {tuple(cand.shape)} != reference {tuple(ref.shape)}"
        return out
    diff = np.abs(cand - ref)
    ref_absmax = float(np.abs(ref).max())
    out.update(
        {
            "comparable": True,
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rel_to_absmax": float(diff.max() / max(ref_absmax, 1e-30)),
            "ref_absmax": ref_absmax,
            "cand_absmax": float(np.abs(cand).max()),
            "argmax_agreement": float((cand.argmax(-1) == ref.argmax(-1)).mean()),
        }
    )
    return out


def logits_summary(array) -> dict:
    """Shape / absmax / mean of a leg's logits (saved next to the record)."""
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "absmax": float(np.abs(arr).max()),
        "mean": float(arr.mean()),
        "finite": bool(np.isfinite(arr).all()),
    }


def read_logits(path) -> object | None:
    """Load a saved ``.logits.npz`` (``None`` when absent)."""
    import numpy as np

    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as blob:
        return np.asarray(blob["logits"], dtype=np.float32)


def save_logits(path: Path, logits, ids) -> Path:
    """Persist a leg's logits + the ids they were produced from."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        logits=np.asarray(logits, dtype=np.float32),
        input_ids=np.asarray(ids, dtype=np.int32),
    )
    return path


# ---------------------------------------------------------------------------
# the zero-host-round-trip probe (device legs)
# ---------------------------------------------------------------------------
class AsDeviceArrayCounter:
    """Count ``iree.runtime.asdevicearray`` calls (the host→device upload path).

    The documented device-resident loop must perform **zero** of them once the
    inputs are placed, so a leg wraps its warm loop in this counter and records the
    result.  The technique is the one ``tests/backends/test_iree_same_device_loop.py``
    uses (a counting wrapper around the real function, restored on exit).

    Only the ``iree`` adapter has this entry point; any other backend reports
    ``checked: False`` with the reason instead of pretending to be zero.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.checked = False
        self.reason: str | None = None
        self._real = None
        self._module = None

    def __enter__(self) -> "AsDeviceArrayCounter":
        try:
            import iree.runtime as rt
        except Exception as exc:
            self.reason = f"iree.runtime not importable ({type(exc).__name__}: {exc})"
            return self
        real = getattr(rt, "asdevicearray", None)
        if real is None:
            self.reason = "iree.runtime exposes no asdevicearray"
            return self
        counter = self

        def counting(*args, **kwargs):
            counter.calls += 1
            return real(*args, **kwargs)

        self._module = rt
        self._real = real
        rt.asdevicearray = counting
        self.checked = True
        return self

    def reset(self) -> None:
        """Forget calls made so far (used after the warmup calls)."""
        self.calls = 0

    def __exit__(self, *exc_info) -> None:
        if self._module is not None and self._real is not None:
            self._module.asdevicearray = self._real
        return None


# ---------------------------------------------------------------------------
# versions table / report formatting
# ---------------------------------------------------------------------------
def env_source_display(path) -> str:
    """Report form of the environment artifact path (never fabricated).

    A recorded path that ends in ``bench_logs/hf_bench_env.json`` — i.e. the file the
    measuring run found inside *some* checkout of this repo, typically a worker
    worktree that exists for nobody else — is rendered as that repo-relative path, the
    artifact that actually ships. Any other path is returned exactly as recorded.
    """
    text = str(path)
    suffix = f"bench_logs/{ENV_JSON_NAME}"
    if text.replace(os.sep, "/").endswith(suffix):
        return suffix
    return text


def load_env_json(out_dirs) -> dict | None:
    """Read the provisioning artifact ``hf_bench_env.json`` if the runner produced it.

    Optional by design: the report degrades to the versions each leg recorded when
    the file is absent (the env-provisioning script is a separate artifact).
    ``_source`` is the *report* form of the path (:func:`env_source_display`), not the
    machine-specific one the measuring process happened to use.
    """
    for out_dir in [*out_dirs, HERE]:
        path = Path(out_dir) / ENV_JSON_NAME
        if path.is_file():
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            payload["_source"] = env_source_display(path)
            return payload
    return None


def fmt(value, digits: int = 2) -> str:
    """Format one table cell; ``None`` renders as an explicit ``—`` (never a guess)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.{digits}e}"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.{max(digits, 3)}f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def fmt_ratio(value) -> str:
    """Format a speedup ratio (``>1`` means the row is faster than the baseline)."""
    if value is None:
        return "—"
    return f"{value:.2f}×"


def status_cell(record: dict | None, reason: str | None = None) -> str:
    """One-line status of a leg, quoting the first error/note when it is not ``ok``."""
    if record is None:
        return f"**not run**: {(reason or 'no record on disk')[:160]}"
    status = record.get("status", "?")
    if status == "ok":
        return "ok"
    detail = (record.get("errors") or record.get("notes") or [""])[0]
    head = str(detail).split("(")[0].strip() or str(detail)
    return f"**{status}**: {head[:160]}"


def stderr_tail(text: str, limit: int = 3) -> list[str]:
    """Last meaningful stderr lines, skipping iree's nanobind teardown leak noise."""
    noise = ("nanobind:", " - leaked", "See https://nanobind", " - leaked type")
    lines = [line for line in (text or "").splitlines() if line.strip()]
    meaningful = [line for line in lines if not line.lstrip().startswith(noise)]
    return (meaningful or lines)[-limit:]


def last_stage(stdout: str) -> str | None:
    """The last ``[stage] ...`` marker a leg child printed before it died."""
    marker = None
    for line in (stdout or "").splitlines():
        if line.startswith("[stage] "):
            marker = line[len("[stage] "):].strip()
    return marker


def first_diagnostic(exc: BaseException) -> str:
    """First actionable line of a (compiler) error, skipping echoed flag dumps."""
    message = str(exc)
    for line in message.splitlines():
        stripped = line.strip()
        if "error:" in stripped:
            return stripped.split("error:", 1)[1].strip()[:300] or stripped[:300]
    return (message.splitlines()[0] if message else type(exc).__name__)[:300]


def md_table(headers, rows) -> list[str]:
    """Render a markdown table (pipe-escaped, newline-collapsed, deterministic)."""
    def cell(value) -> str:
        # A newline inside a cell would end the table row and corrupt the table,
        # so multi-line text (e.g. a compiler diagnostic) is collapsed to one line.
        text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        return text.replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return lines
