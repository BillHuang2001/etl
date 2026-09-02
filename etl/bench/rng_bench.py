"""etl.bench.rng_bench — benchmark the etl.random algorithms across targets.

Standalone measurement tool (NOT part of the etl.bench public API — run via
``python3 -m etl.bench.rng_bench``) that benchmarks the three canonical
``etl.random`` algorithms (``splitmix64`` / ``threefry2x32`` /
``philox4x32_10``) across the lowering paths and targets, producing the data
needed to choose the DEFAULT algorithm for ``etl.random.key``.

Matrix
------
- ops: ``uniform``, ``normal``, ``randint`` (element counts from ``--sizes``)
  and ``split`` (fixed key count from ``--split-n`` — one row per
  algorithm×target, not per size). The ``split`` graph derives ``n`` keys via
  ``split_n`` and ACCUMULATES them (elementwise add over all ``n`` keys) into
  a single output tensor: this consumes every derived key (no dead-code risk)
  while avoiding both a 4096-result pytree and the iree HAL binding-count
  limits that any n-way stack/concat formulation hits.
- algorithms: the three canonical names (``--algorithms``); the key type
  (shape/dtype) is the static algorithm tag: splitmix64 → rank-0 int64,
  threefry2x32 → ``(2,)`` int32, philox4x32_10 → ``(4,)`` int32.
- targets: ``numpy`` (reference interpreter), ``llvm-cpu`` and ``cuda``
  (both via the iree adapter). Every graph is built with an EXPLICIT key
  input (``etl.random.key(seed, algorithm)`` — a concrete tensor, never a
  graph Constant).
- paths (compiler targets only): ``inline`` (the exporter's bit-exact inline
  cipher expansions — the capability forced to the EMPTY set) and ``native``
  (``stablehlo.rng_bit_generator`` — the capability forced to the full
  native-capable set ``frozenset({"threefry2x32", "philox4x32_10"})``).
  ``Capabilities.rng_bit_generator`` is a PER-ALGORITHM frozenset of algorithm
  names (not a bool); the per-config path is toggled by monkeypatching
  ``IreeBackend.capabilities`` for the duration of the config and RESTORED
  afterwards. On iree, native THREE_FRY is now the ADAPTER DEFAULT for
  threefry2x32 — the inline/native rows still measure BOTH paths via the
  monkeypatch. splitmix64 has no native form → inline only. numpy is the
  reference → path ``-``.

Methodology (per config)
------------------------
- ``compile_ms`` = wall time of lower + compile + load (once). Graphs are
  traced once per (op, size, algorithm) and cached; tracing is excluded from
  the timing.
- ``best_ms`` = best-of-``repeats`` of ``run()`` after ``warmup`` untimed
  runs, via ``perf_counter`` around ``run()`` ONLY. On cuda the run returns a
  device-resident ``Tensor``; the host copy (``.numpy()``) is NEVER part of
  the timed loop.
- Bit-exactness: every successful compiler-target config is verified against
  the numpy-backend run of the SAME graph + key (bitwise ``array_equal``;
  ``normal`` f32 additionally falls back to the documented
  Box–Muller fast-path budget ``rtol=1e-4`` with ``atol = 1e-4 ·
  max(1, size/2**20)`` plus a mean-abs < 1e-6 guard — see
  ``_normal_fast_path_ok`` — since the compiler backends run the f32 fast
  path, so bitwise vs numpy is NOT expected there, and the deviation tail
  grows with the sample size). Two consecutive runs are also compared
  bitwise (same-key determinism).
  Any unexpected bitwise failure is reported loudly.

Known exclusions (documented in the report header)
--------------------------------------------------
- ``random_permutation``: excluded — a deterministic cuda argsort failure
  (upstream iree stablehlo.sort bufferization bug).
- ``random_multinomial``: excluded — v1 deferral on ALL compiler backends
  (explicit ``BackendError``); numpy-only.
- native PHILOX on iree (llvm-cpu AND cuda): EXPECTED compile failure —
  iree 3.11 cannot legalize ``stablehlo.rng_bit_generator`` with
  RNG_ALG_PHILOX. Recorded as ``FAIL (iree legalization)`` in the table;
  never crashes the run.
- ``split`` on cuda: excluded — the iree cuda HAL limits each export to 16
  bindings, so ANY ``split_n`` graph (n outputs, or the n-way stack/concat/
  accumulation chain) fails at executable create. On llvm-cpu the
  accumulation formulation runs at the default n (512) for all three
  algorithms (bit-exact); n=4096 still hits the 32-binding limit at run
  ("binding count 4097 > 32").
- ``--split-n`` default 512 (NOT the originally-spec'd 4096): 4096 is
  impossible on iree (binding limits above) and compiles ~5-25 min per
  config (the inline expansions are huge); 512 keeps the worst-case
  per-config compile ≈ 2 min while remaining a representative key-derivation
  workload. Override with ``--split-n``.

Results
-------
Benchmark results and the default-algorithm decision are recorded in
``etl/bench/rng_bench_results.md`` (sibling of this module); reproduce them
with the usage line below.

Usage::

    python3 -m etl.bench.rng_bench [--ops OPS] [--sizes N,N,N] [--targets T,T]
        [--algorithms A,A,A] [--split-n N] [--repeats N] [--warmup N]
        [--seed N] [--device auto|KIND[:INDEX]] [--out FILE] [--quick]

Only stdlib + numpy + etl are imported at module scope (the iree adapter is
imported lazily inside the staging helpers — numpy-only runs never import
iree). Torch is irrelevant here (no torch references).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
import subprocess
import sys
import time

import numpy as np

import etl
from etl import core
from etl.bench._util import best_time_ms, place_input, to_host
from etl.ir.op_defs.random import algorithm_key_type

__all__ = ["main", "run_benchmark", "ALGORITHMS", "OPS", "TARGETS"]

#: Canonical algorithm names (order = canonical order).
ALGORITHMS = ("splitmix64", "threefry2x32", "philox4x32_10")
#: Ops measured by the matrix. ``permutation``/``multinomial`` are excluded
#: (see the module docstring — iree-cuda sort bug / compiler-backend deferral).
OPS = ("uniform", "normal", "randint", "split")
#: Targets: numpy reference interpreter, iree llvm-cpu, iree cuda.
TARGETS = ("numpy", "llvm-cpu", "cuda")

DEFAULT_SIZES = (2**20, 2**22, 2**24)  # 1048576 / 4194304 / 16777216
QUICK_SIZES = (2**20, 2**22)
DEFAULT_SPLIT_N = 512  # not 4096 — see the module docstring (iree binding limits)
QUICK_SPLIT_N = 256
DEFAULT_REPEATS = 7
NUMPY_REPEATS = 3  # numpy is slow (reference interpreter)
DEFAULT_WARMUP = 2
MAX_NOTE_LEN = 80
#: Documented f32-normal Box–Muller fast-path budget (mirrors
#: ``tests/backends/test_iree_random_algorithms_parity.py`` RANDOM_NORMAL_TOL).
NORMAL_F32_TOL = {"rtol": 1e-4, "atol": 1e-5}
#: Algorithms with a native ``stablehlo.rng_bit_generator`` form.
_NATIVE_CAPABLE = ("threefry2x32", "philox4x32_10")


# ---------------------------------------------------------------------------
# Graphs under test (module-level defns; the size / split-n is a STATIC
# Python int trace argument — never a graph input, never a closure tensor).
# ---------------------------------------------------------------------------


@etl.defn
def _uniform(key, shape):
    return etl.random.uniform(key, shape, low=0.0, high=1.0)


@etl.defn
def _normal(key, shape):
    return etl.random.normal(key, shape, mean=0.0, std=1.0)


@etl.defn
def _randint(key, shape):
    return etl.random.randint(key, shape, 0, 2**31 - 1)


@etl.defn
def _split(key, n):
    # Derive n keys and ACCUMULATE them (elementwise add over all n keys)
    # into a single output tensor. This consumes every derived key — no
    # dead-code risk — and avoids the iree HAL binding-count limits that any
    # n-way stack/concat hits (each concat operand is a dispatch binding; the
    # add chain is folded into one n+1-operand reduce at worst, which fits
    # llvm-cpu's 32-binding limit at n=512 but not cuda's 16).
    ks = etl.random.split_n(key, n)
    acc = ks[0]
    for k in ks[1:]:
        acc = etl.add(acc, k)
    return acc


_OP_DEFNS = {
    "uniform": _uniform,
    "normal": _normal,
    "randint": _randint,
    "split": _split,
}


# ---------------------------------------------------------------------------
# Caches (per-process; keyed by (op, size, algorithm) / + seed)
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict = {}
_NUMPY_REF_OUT: dict = {}


def _get_graph(op: str, size: int, algorithm: str):
    """Trace once per (op, size, algorithm); reused by every target/path."""
    key = (op, size, algorithm)
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        shape, dtype = algorithm_key_type(algorithm)
        graph = etl.trace(_OP_DEFNS[op], core.TensorSpec(shape, dtype), size)
        _GRAPH_CACHE[key] = graph
    return graph


def _numpy_reference_out(op: str, size: int, algorithm: str, seed: int) -> np.ndarray:
    """The numpy-backend output array for (op, size, algorithm, seed) — the
    bit-exactness reference for every compiler-target config."""
    key = (op, size, algorithm, seed)
    arr = _NUMPY_REF_OUT.get(key)
    if arr is None:
        graph = _get_graph(op, size, algorithm)
        lowered = etl.lower(graph, backend="numpy")
        artifact = etl.compile(lowered)
        exe = etl.load(artifact)
        out = etl.run(exe, etl.random.key(seed, algorithm), size)
        arr = _asarray(out)
        _NUMPY_REF_OUT[key] = arr
    return arr


def _asarray(out):
    """Normalize a single-tensor run output to a numpy ndarray, with an
    EXPLICIT host readback of device-resident outputs (``.to(cpu)`` first —
    the explicit device-placement contract: ``.numpy()`` on a non-cpu device
    tensor raises; on cpu tensors ``to_host`` is a no-op, so the cpu paths
    are byte-identical to a plain ``.numpy()``)."""
    if isinstance(out, core.Tensor):
        return to_host(out).numpy()
    if isinstance(out, np.ndarray):
        return out
    raise TypeError(
        f"unexpected run output {type(out).__name__} (expected a single "
        "etl.Tensor / ndarray)"
    )


# ---------------------------------------------------------------------------
# iree native-toggle monkeypatch (restored after every config)
# ---------------------------------------------------------------------------


def _set_iree_native(path: str):
    """Patch ``IreeBackend.capabilities.rng_bit_generator`` (a PER-ALGORITHM
    frozenset of algorithm names, not a bool) for the given lowering
    ``path``: ``"native"`` → ``frozenset(_NATIVE_CAPABLE)`` (both
    native-capable ciphers); ``"inline"`` → ``frozenset()`` (forces the
    exporter's bit-exact inline expansions for every algorithm). Returns the
    ORIGINAL capabilities for restoration."""
    from etl.backends.adapters.iree import IreeBackend

    orig = IreeBackend.capabilities
    caps = frozenset(_NATIVE_CAPABLE) if path == "native" else frozenset()
    if orig.rng_bit_generator != caps:
        IreeBackend.capabilities = dataclasses.replace(
            orig, rng_bit_generator=caps
        )
    return orig


def _restore_iree(orig) -> None:
    from etl.backends.adapters.iree import IreeBackend

    IreeBackend.capabilities = orig


def _normal_fast_path_ok(a1: np.ndarray, ref: np.ndarray, size: int) -> bool:
    """Is the f32-normal deviation vs the numpy reference within the
    DOCUMENTED Box–Muller fast-path budget?

    The compiler backends run the f32 fast path, so bitwise equality with the
    numpy run is NOT expected; the deviation is dominated by rare
    near-zero-cancellation events whose extreme tail grows with the sample
    size (measured max-abs: ~2.3e-5 at 2^20 — philox; threefry ~8.9e-6 at
    2^20, splitmix ~4.8e-7; at 2^24: philox ~2.2e-4, splitmix ~4.6e-5).
    Budget: rtol=1e-4 with atol = 1e-4 · max(1, size/2^20) — a 4x+ margin
    over the largest measured tail at every size — PLUS a mean-abs guard
    (noise mean stays ~1e-7; a systematic deviation — e.g. a wrong constant
    or missing salt — would push the mean far above 1e-6 even when the
    extreme tail is unremarkable). A real bug fails loudly either way.
    """
    if not np.allclose(
        a1, ref, rtol=NORMAL_F32_TOL["rtol"], atol=1e-4 * max(1.0, size / 2**20)
    ):
        return False
    diff = np.abs(a1.astype(np.float64) - ref.astype(np.float64))
    return bool(np.mean(diff) < 1e-6)


# ---------------------------------------------------------------------------
# CUDA device resolution (free-GPU scan; prefer cuda:5)
# ---------------------------------------------------------------------------


def _scan_free_cuda() -> core.Device:
    """Most-free GPU via nvidia-smi (>= 2 GiB free), preferring index 5.

    Raises SystemExit with a clear message when nvidia-smi is unavailable,
    reports no GPUs, or every GPU is busy.
    """
    if shutil.which("nvidia-smi") is None:
        raise SystemExit(
            "rng_bench: --targets cuda requires nvidia-smi (not found) to "
            "pick a free GPU — pass --device cuda:N to select one explicitly"
        )
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"rng_bench: nvidia-smi failed: {exc}") from None
    if proc.returncode != 0:
        raise SystemExit(
            f"rng_bench: nvidia-smi failed: {proc.stderr.strip()}"
        )
    gpus = []
    for line in proc.stdout.strip().splitlines():
        try:
            idx, free_mib = (part.strip() for part in line.split(","))
            gpus.append((int(free_mib), int(idx)))
        except ValueError:
            continue  # malformed line — ignore
    if not gpus:
        raise SystemExit("rng_bench: nvidia-smi reported no GPUs")
    free = [(free_mib, idx) for free_mib, idx in gpus if free_mib >= 2048]
    if not free:
        raise SystemExit(
            "rng_bench: no GPU with >= 2 GiB free memory (nvidia-smi): "
            + ", ".join(f"gpu{idx} {free_mib}MiB" for free_mib, idx in gpus)
        )
    free.sort(reverse=True)  # most free first
    if 5 in (idx for _, idx in free):
        return core.Device("cuda", 5)
    return core.Device("cuda", free[0][1])


def _resolve_cuda_device(device_arg, targets) -> core.Device | None:
    """Resolve the cuda device for the run (``None`` when cuda is not among
    the requested targets). ``auto`` → free-GPU scan (prefer cuda:5). A
    ``KIND[:INDEX]`` string is validated via ``etl.bench._util.resolve_device``
    (contradictions raise SystemExit with a clear message)."""
    if "cuda" not in targets:
        if device_arg is not None and device_arg != "auto" \
                and device_arg.startswith("cuda"):
            print(
                f"rng_bench: note: --device {device_arg} given but the cuda "
                "target is not in --targets; the device is unused",
                file=sys.stderr,
            )
        return None
    if device_arg in (None, "auto"):
        return _scan_free_cuda()
    from etl.bench._util import resolve_device

    device = resolve_device(device_arg)
    if device.kind == "cpu":
        raise SystemExit(
            f"rng_bench: --device {device_arg} is a cpu device but "
            "--targets includes cuda — pick a cuda device or drop the cuda "
            "target"
        )
    return device


def _probe_cuda(device: core.Device) -> None:
    """Acquire the iree cuda HAL device once, up front — a broken driver/
    GPU fails cleanly BEFORE the (long) measurement phase instead of as a
    wall of per-config errors. Mirrors the tests' device fixture."""
    try:
        import iree.runtime as rt

        driver = rt.get_driver("cuda")
        driver.create_device(device_id=device.index + 1)  # 1-based ids
    except Exception as exc:  # noqa: BLE001 — surface any driver failure
        raise SystemExit(
            f"rng_bench: IREE cuda HAL driver or GPU {device} unavailable: "
            f"{exc}"
        ) from None


# ---------------------------------------------------------------------------
# Config enumeration + measurement
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Config:
    op: str
    size: int  # element count (uniform/normal/randint) or split key count
    algorithm: str
    target: str
    path: str  # "inline" | "native" | "-" (numpy)


def _iter_configs(args, targets):
    """Yield the config matrix in stable order (op, size, algorithm, target,
    path). ``split`` uses ``--split-n`` instead of the sizes list and is
    excluded on cuda (iree cuda HAL 16-binding/export limit — documented)."""
    for op in args.ops:
        sizes = [args.split_n] if op == "split" else args.sizes
        for size in sizes:
            for algorithm in args.algorithms:
                for target in targets:
                    if op == "split" and target == "cuda":
                        continue  # documented exclusion (binding limit)
                    if target == "numpy":
                        yield _Config(op, size, algorithm, target, "-")
                        continue
                    paths = (
                        ("inline",)
                        if algorithm == "splitmix64"
                        else ("inline", "native")
                    )
                    for path in paths:
                        yield _Config(op, size, algorithm, target, path)


def _measure_config(cfg: _Config, args, cuda_device, repeats) -> dict:
    """Measure one config. NEVER raises: every failure is recorded in the
    returned row (error text truncated to ~80 chars in ``notes``) so the
    script always completes and prints the table."""
    row = {
        "op": cfg.op,
        "size": cfg.size,
        "algorithm": cfg.algorithm,
        "target": cfg.target,
        "path": cfg.path,
        "compile_ms": None,
        "best_ms": None,
        "bit_exact": None,
        "notes": "",
        "error": None,
    }
    key_tensor = etl.random.key(args.seed, cfg.algorithm)
    try:
        graph = _get_graph(cfg.op, cfg.size, cfg.algorithm)
        t0 = time.perf_counter()
        if cfg.target == "numpy":
            lowered = etl.lower(graph, backend="numpy")
            artifact = etl.compile(lowered)
            exe = etl.load(artifact)
        else:
            orig = _set_iree_native(cfg.path)
            try:
                lowered = etl.lower(graph, backend="iree")
                target_backends = (
                    ["cuda"] if cfg.target == "cuda" else ["llvm-cpu"]
                )
                artifact = etl.compile(lowered, target_backends=target_backends)
                device = cuda_device if cfg.target == "cuda" else None
                exe = etl.load(artifact, device=device)
            finally:
                _restore_iree(orig)
        row["compile_ms"] = (time.perf_counter() - t0) * 1000.0

        # Explicit device placement: the (immutable, reused) host key tensor
        # is placed ONCE on the run device, before the timed loop — no
        # implicit host->device staging inside the loop (cpu targets — numpy
        # and iree llvm-cpu — keep the host key as-is; the run returns
        # device-resident tensors on cuda, read back explicitly in the
        # verification step below via _asarray, never in the timed loop).
        run_key = (
            place_input(key_tensor, cuda_device)
            if cfg.target == "cuda"
            else key_tensor
        )
        run_fn = lambda: etl.run(exe, run_key, cfg.size)  # noqa: E731
        row["best_ms"] = best_time_ms(run_fn, args.warmup, repeats)

        if cfg.target != "numpy":
            # Bit-exactness vs the numpy reference + same-key determinism
            # across runs (both documented contracts). The host copy is only
            # ever taken in this verification step, never in the timed loop.
            a1 = _asarray(run_fn())
            a2 = _asarray(run_fn())
            if not np.array_equal(a1, a2):
                row["notes"] += "DETERMINISM FAIL (two runs differ); "
            ref = _numpy_reference_out(
                cfg.op, cfg.size, cfg.algorithm, args.seed
            )
            if np.array_equal(a1, ref):
                row["bit_exact"] = "yes"
            elif cfg.op == "normal" and _normal_fast_path_ok(a1, ref, cfg.size):
                row["bit_exact"] = "no*"
                row["notes"] += (
                    "doc f32 normal fast-path (allclose ok, ~1e-6 rel)"
                )
            else:
                row["bit_exact"] = "NO"
                row["notes"] += "BIT-EXACTNESS FAIL vs numpy reference"
        else:
            row["bit_exact"] = "-"  # numpy IS the reference
    except Exception as exc:  # noqa: BLE001 — record, never crash
        message = f"{type(exc).__name__}: {exc}"
        row["error"] = message
        if "rng_bit_generator" in message:
            row["notes"] = "FAIL (iree legalization)"
        else:
            row["notes"] = message[:MAX_NOTE_LEN]
    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ms(ms) -> str:
    return f"{ms:.2f}" if ms is not None else "-"


def _fmt_compile_ms(ms) -> str:
    return f"{ms:.0f}" if ms is not None else "-"


def _fmt_vs_numpy(ratio) -> str:
    if ratio is None:
        return "-"
    return f"{ratio:.2f}×"


def _versions() -> str:
    import importlib.metadata as metadata

    parts = [f"python {sys.version.split()[0]}", f"numpy {np.__version__}"]
    for pkg in (
        "iree-base-compiler", "iree-base-runtime",
        "iree-compiler", "iree-runtime",
    ):
        try:
            parts.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            continue
    return ", ".join(parts)


def _render(args, rows, numpy_best, targets, cuda_device, quick) -> str:
    lines = []
    add = lines.append
    add("# etl.random RNG benchmark — algorithm × lowering path × target")
    add("")
    add(f"- environment: {_versions()}")
    add(f"- seed={args.seed}, warmup={args.warmup}, repeats={args.repeats}"
        f" (numpy target: {args.numpy_repeats}), split-n={args.split_n}"
        + (f" (--quick: sizes {', '.join(map(str, args.sizes))})" if quick else ""))
    add(f"- targets: {', '.join(targets)}"
        + (f", cuda device {core.Device('cuda', cuda_device.index)}"
           if "cuda" in targets else "")
        + f", algorithms: {', '.join(args.algorithms)}")
    add("")
    add("## Exclusions & notes")
    add("- `random_permutation` and `random_multinomial` are excluded from the"
        " matrix (iree-cuda sort bufferization failure; compiler-backend v1"
        " deferral).")
    add("- native PHILOX on iree (llvm-cpu AND cuda) is an EXPECTED compile"
        " failure (iree cannot legalize `stablehlo.rng_bit_generator` with"
        " RNG_ALG_PHILOX) — recorded as `FAIL (iree legalization)`, never a"
        " crash.")
    add("- `split` is excluded on cuda: the iree cuda HAL limits exports to 16"
        " bindings, so any split_n graph (n outputs / n-way stack / n-way"
        " accumulation) fails at executable create. On llvm-cpu the"
        " accumulation formulation runs at the default n; n=4096 hits the"
        " 32-binding limit at run.")
    add(f"- `--split-n` default {DEFAULT_SPLIT_N} (the originally-planned 4096"
        " is impossible on iree — binding limits + 5-25 min compiles of the"
        " inline expansions).")
    add("- `normal` f32 on compiler targets uses the DOCUMENTED f32 Box–Muller"
        " fast path: bitwise vs numpy differs (deviation tail grows with the"
        " sample size; budget rtol=1e-4, atol = 1e-4 · max(1, size/2^20),"
        " mean-abs < 1e-6). `bit_exact` = `no*` marks exactly that (budget"
        " passes); a bare `NO` means a real bug.")
    add("- `bit_exact` = `yes` means bitwise-identical to the numpy-backend run"
        " of the same graph + key (also verified: two consecutive runs on the"
        " same target are bitwise identical).")
    add("")
    add("## Results (op × size × algorithm × target × path)")
    add("")
    add("| op | size | algorithm | target | path | compile_ms | best_ms |"
        " vs_numpy | bit_exact | notes |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        cell = (row["op"], row["size"], row["algorithm"])
        numpy_ms = numpy_best.get(cell)
        ratio = (
            None
            if numpy_ms is None or row["best_ms"] is None
            else numpy_ms / row["best_ms"]
        )
        add(
            f"| {row['op']} | {row['size']} | {row['algorithm']} | "
            f"{row['target']} | {row['path']} | {_fmt_compile_ms(row['compile_ms'])} | "
            f"{_fmt_ms(row['best_ms'])} | {_fmt_vs_numpy(ratio)} | "
            f"{row['bit_exact'] or '-'} | {row['notes']} |"
        )
    add("")

    # Per-target fastest (algorithm, path) per (op, size).
    for target in targets:
        add(f"## Fastest (algorithm, path) per (op, size) — target: {target}")
        add("")
        add("| op | size | fastest algorithm | path | best_ms |")
        add("|---|---|---|---|---|")
        for op in args.ops:
            sizes = [args.split_n] if op == "split" else args.sizes
            for size in sizes:
                best = None
                for algorithm in args.algorithms:
                    for row in rows:
                        if (
                            row["op"] == op
                            and row["size"] == size
                            and row["algorithm"] == algorithm
                            and row["target"] == target
                            and row["best_ms"] is not None
                            and (best is None or row["best_ms"] < best["best_ms"])
                        ):
                            best = row
                if best is None:
                    add(f"| {op} | {size} | - | - | - |")
                else:
                    add(
                        f"| {best['op']} | {best['size']} | "
                        f"{best['algorithm']} | {best['path']} | "
                        f"{_fmt_ms(best['best_ms'])} |"
                    )
        add("")

    # Final summary: geometric-mean best_ms per algorithm (best legal path
    # per cell) — the data needed to choose the default algorithm.
    add("## Geometric-mean best_ms per algorithm (best legal path per cell)")
    add("")
    add("| target | cells | splitmix64 | threefry2x32 | philox4x32_10 |"
        " fastest |")
    add("|---|---|---|---|---|---|")
    for target in targets:
        # Total (op, size) cells for this target (split cells don't exist on
        # cuda — the documented exclusion).
        total_cells = sum(
            1 if op == "split" else len(args.sizes)
            for op in args.ops
            if not (op == "split" and target == "cuda")
        )
        cell_alg_best = {}  # (op, size) -> {algorithm: best_ms}
        for row in rows:
            if row["target"] != target or row["best_ms"] is None:
                continue
            cell = (row["op"], row["size"])
            prev = cell_alg_best.setdefault(cell, {})
            if (
                row["algorithm"] not in prev
                or row["best_ms"] < prev[row["algorithm"]]
            ):
                prev[row["algorithm"]] = row["best_ms"]
        cells_ok = len(cell_alg_best)
        geo = {}
        for algorithm in args.algorithms:
            vals = [
                per_alg[algorithm]
                for per_alg in cell_alg_best.values()
                if algorithm in per_alg
            ]
            if vals:
                geo[algorithm] = math.exp(
                    sum(math.log(v) for v in vals) / len(vals)
                )
        cells_out = [target, f"{cells_ok}/{total_cells}"]
        fastest = None
        for algorithm in ALGORITHMS:
            if algorithm in geo:
                cells_out.append(f"{geo[algorithm]:.2f}")
                if fastest is None or geo[algorithm] < fastest[1]:
                    fastest = (algorithm, geo[algorithm])
            else:
                cells_out.append("-")
        cells_out.append(fastest[0] if fastest else "-")
        add("| " + " | ".join(cells_out) + " |")
    add("")
    add("Per cell, each algorithm contributes its best LEGAL path (splitmix64:"
        " inline — it has no native form; threefry2x32/philox4x32_10:"
        " min(inline, native); native philox fails, so its cells use inline)."
        " Cells with no successful measurement are excluded from the"
        " geometric mean (see the `cells` column).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m etl.bench.rng_bench",
        description="Benchmark the etl.random algorithms (splitmix64 / "
        "threefry2x32 / philox4x32_10) × lowering path (native "
        "stablehlo.rng_bit_generator vs bit-exact inline expansions) × "
        "target (numpy / iree-llvm-cpu / iree-cuda).",
    )
    parser.add_argument(
        "--ops", default=",".join(OPS),
        help=f"comma list of ops (default: {', '.join(OPS)}); 'split' uses "
        "--split-n instead of --sizes",
    )
    parser.add_argument(
        "--sizes", default=None,
        help="comma list of element counts for uniform/normal/randint "
        f"(default: {','.join(map(str, DEFAULT_SIZES))}, --quick: "
        f"{','.join(map(str, QUICK_SIZES))})",
    )
    parser.add_argument(
        "--targets", default=",".join(TARGETS),
        help=f"comma list of targets (default: {', '.join(TARGETS)})",
    )
    parser.add_argument(
        "--algorithms", default=",".join(ALGORITHMS),
        help=f"comma list of algorithms (default: {', '.join(ALGORITHMS)})",
    )
    parser.add_argument(
        "--split-n", type=int, default=None,
        help="key count for the split op (default: "
        f"{DEFAULT_SPLIT_N}, --quick: {QUICK_SPLIT_N}; 4096 is impossible on "
        "iree — HAL binding limits, see the module docstring)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help=f"timed runs per config (default: {DEFAULT_REPEATS}, "
        "--quick: halved; the numpy target defaults to "
        f"{NUMPY_REPEATS} — it is slow)",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"untimed runs before timing (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for key derivation (default: 0)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="cuda device: 'auto' (free-GPU scan, prefers cuda:5), "
        "'cuda:N', or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="write the markdown report to FILE instead of stdout",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="smoke/CI mode: sizes 2^20,2^22, repeats halved, split-n halved",
    )
    args = parser.parse_args(argv)

    def _csv(value, choices, name):
        items = [item.strip() for item in value.split(",") if item.strip()]
        for item in items:
            if item not in choices:
                raise SystemExit(
                    f"rng_bench: unknown {name} {item!r}; expected one of "
                    f"{', '.join(choices)}"
                )
        return tuple(items)

    args.ops = _csv(args.ops, OPS, "op")
    args.algorithms = _csv(args.algorithms, ALGORITHMS, "algorithm")
    args.targets = _csv(args.targets, TARGETS, "target")

    if args.sizes is None:
        args.sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    else:
        try:
            sizes = tuple(
                int(item.strip()) for item in args.sizes.split(",")
                if item.strip()
            )
        except ValueError:
            raise SystemExit(
                f"rng_bench: --sizes must be comma-separated integers, got "
                f"{args.sizes!r}"
            ) from None
        if not sizes or any(size <= 0 for size in sizes):
            raise SystemExit(
                f"rng_bench: --sizes must be positive integers, got {sizes}"
            )
        args.sizes = sizes
    if args.split_n is None:
        args.split_n = QUICK_SPLIT_N if args.quick else DEFAULT_SPLIT_N
    if args.split_n <= 0:
        raise SystemExit(f"rng_bench: --split-n must be positive, got {args.split_n}")
    if args.repeats is None:
        args.repeats = (
            max(1, DEFAULT_REPEATS // 2) if args.quick else DEFAULT_REPEATS
        )
        args.numpy_repeats = 1 if args.quick else NUMPY_REPEATS
    else:
        if args.repeats < 1:
            raise SystemExit(f"rng_bench: --repeats must be >= 1, got {args.repeats}")
        args.numpy_repeats = args.repeats
    if args.warmup < 0:
        raise SystemExit(f"rng_bench: --warmup must be >= 0, got {args.warmup}")
    return args


def run_benchmark(args, targets, cuda_device) -> tuple[list, dict]:
    """Run the config matrix; returns (rows, numpy_best). Raises nothing —
    per-config failures are recorded in the rows."""
    repeats_map = {
        "numpy": args.numpy_repeats,
        "llvm-cpu": args.repeats,
        "cuda": args.repeats,
    }
    configs = list(_iter_configs(args, targets))
    rows = []
    total = len(configs)
    for index, cfg in enumerate(configs, start=1):
        row = _measure_config(cfg, args, cuda_device, repeats_map[cfg.target])
        rows.append(row)
        if row["error"] is not None:
            print(
                f"[{index}/{total}] {cfg.op} {cfg.size} {cfg.algorithm} "
                f"{cfg.target} {cfg.path}: FAILED "
                f"({row['notes'][:MAX_NOTE_LEN]})",
                file=sys.stderr,
            )
        else:
            print(
                f"[{index}/{total}] {cfg.op} {cfg.size} {cfg.algorithm} "
                f"{cfg.target} {cfg.path}: compile={row['compile_ms']:.0f}ms "
                f"best={row['best_ms']:.3f}ms bit={row['bit_exact']}",
                file=sys.stderr,
            )
    numpy_best = {}
    for row in rows:
        if row["target"] == "numpy" and row["best_ms"] is not None:
            numpy_best[(row["op"], row["size"], row["algorithm"])] = row["best_ms"]
    return rows, numpy_best


def main(argv=None) -> int:
    """CLI entry point (``python3 -m etl.bench.rng_bench``). Returns 0 on
    completion (even with per-config FAILs recorded in the report); exits 2
    on usage/environment errors (bad arguments, missing iree install, no
    free GPU)."""
    args = _parse_args(argv)
    cuda_device = _resolve_cuda_device(args.device, args.targets)
    if "cuda" in args.targets:
        # The iree adapter must be installed AND the cuda HAL device
        # acquirable — fail cleanly up front (mirrors the bench harness's
        # up-front backend validation).
        try:
            etl.backends.get("iree")
        except core.BackendError as exc:
            raise SystemExit(
                f"rng_bench: {exc} (needed for --targets cuda)"
            ) from None
        _probe_cuda(cuda_device)
    elif any(target != "numpy" for target in args.targets):
        try:
            etl.backends.get("iree")
        except core.BackendError as exc:
            raise SystemExit(
                f"rng_bench: {exc} (needed for --targets "
                f"{', '.join(t for t in args.targets if t != 'numpy')})"
            ) from None

    rows, numpy_best = run_benchmark(args, args.targets, cuda_device)
    report = _render(
        args, rows, numpy_best, args.targets, cuda_device, args.quick
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")
        print(f"rng_bench: report written to {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
