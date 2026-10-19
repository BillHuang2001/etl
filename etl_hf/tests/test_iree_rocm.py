"""ROCm parity suite: the WHOLE ``etl_hf`` forward **device-resident** on iree/rocm.

The package runs the whole forward as ONE etl graph whose weights are ordinary
graph **inputs**, and etl's placement rule is that a non-cpu executable rejects
host inputs at its run boundary. This suite pins the documented device-resident
recipe end to end on the real GPU of this host (AMD Instinct MI300X VF, gfx942,
ROCm 7.1)::

    device = etl.core.Device("rocm", 0)                   # HAL device_id = index+1
    loaded = load_hf(dir)                                 # config + host params
    fwd    = forward_fn(loaded.config, seq_len=…)         # the pure @etl.defn
    exe    = build(fwd, loaded.config, backend="iree", device=device)   # ONE program
    out    = etl_hf.run(
                 exe,
                 etl_hf.place_params(loaded.params, device),    # explicit H2D/leaf
                 etl_hf.place_ids(input_ids, device))           # explicit H2D
    logits = out.to(etl.core.Device("cpu", 0)).numpy()    # exactly ONE explicit D2H
Pinned here (nowhere else — it needs the hardware): the **ROCm opt-level default**
(``"O1"`` for an all-ROCm IREE target set; O2/O3 hit an upstream ``iree-compile``
``'func.func' op failed to distribute`` on transformer LayerNorm-reduce shapes,
while ``llvm-cpu`` keeps ``"O3"`` — options-only, so that test needs neither a
compiler nor a GPU); ``Device("rocm", 0)`` ↔ HAL ``device_id = 1``; the
**device-resident forward** (outputs stay on ``Device('rocm', 0)``, ``.numpy()``
raises :class:`etl.DeviceError`, one explicit ``.to(cpu)`` yields logits matching
the numpy interpreter and the stored HF golden within a *relative-to-absmax* fp32
budget); **host inputs rejected** at the run boundary (params *and* ids); the
**same-device loop** with **zero** ``iree.runtime.asdevicearray`` calls; **packed +
device-resident** (3–5 staged leaves instead of ~21/36, bit-identical to the
unpacked device run); true-fp32 ``dot`` (``dot_precision="HIGHEST"``, the knob
xla/rocm needs, is a no-op); and, under ``ETL_HF_BIG=1``, ``distilgpt2`` / the
124M ``gpt2`` as 4-input packed programs with 5 / 10 consecutive bit-identical
runs (ROCm has no ~161-consumed-input runtime cliff, so the shape that SIGSEGVs
on llvm-cpu unpacked runs here).

Gating: every compiling test is marked ``@requires_rocm`` (a module-level probe
using ``importlib.util.find_spec("iree")`` plus an ``etl.devices()`` rocm/hip
lookup — never a bare ``import iree`` at module scope), so the file skips cleanly
— never errors — without the ``iree`` extra or without a GPU. The options-only
test runs unconditionally. **No timing is ever asserted** (the GPU is shared):
wall times land in :data:`MEASURED` and in the failure-message text.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

import etl
from etl_hf import run
from etl_hf.pipeline import (
    DEFAULT_LLVM_CPU_COMPILE_ARGS,
    DEFAULT_OPT_LEVEL,
    DEFAULT_ROCM_OPT_LEVEL,
    build,
    place_ids,
    place_params,
    resolve_build_options,
)
from etl_hf.load import LoadedModel, load_hf
from etl_hf.pack import pack_params, pack_plan
from etl_hf.program import forward_fn, param_specs
from etl_hf.tests import hf_download
# --- module guards: clean skips, never errors (no iree / no GPU / not opted in) ---
def _iree_available() -> bool:
    """Probe for the IREE bindings **without importing them** (metadata lookup)."""
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("iree", "iree.compiler", "iree.runtime")
        )
    except Exception:  # noqa: BLE001 - a partial install must skip, not error
        return False


IREE_AVAILABLE = _iree_available()


def _rocm_available() -> bool:
    """Whether ``etl.devices()`` reports a rocm/hip device (probed only with iree).

    ``etl.devices()`` probes ROCm through the iree HIP driver, hence the
    short-circuit: without the bindings nothing is probed or imported."""
    if not IREE_AVAILABLE:
        return False
    try:
        return any(device.kind in ("rocm", "hip") for device in etl.devices())
    except Exception:  # noqa: BLE001 - an unusable HAL must skip, not error
        return False


ROCM_AVAILABLE = _rocm_available()

#: Distinguish the two skip causes so an operator knows what to install/enable.
_ROCM_SKIP_REASON = (
    "no rocm/hip device is visible through etl.devices() (iree HAL); this suite needs a real AMD GPU"
    if IREE_AVAILABLE
    else "iree (iree.compiler / iree.runtime) is not installed; run with `uv run --extra iree`"
)
requires_rocm = pytest.mark.skipif(not ROCM_AVAILABLE, reason=_ROCM_SKIP_REASON)

#: Real-size checkpoints are opt-in (``gpt2`` is ≈0.52 GB of weights, and a
#: device-resident run holds the host tree *and* its device copy).
_BIG = pytest.mark.skipif(
    os.environ.get("ETL_HF_BIG") != "1",
    reason="real-size checkpoints are opt-in: ETL_HF_BIG=1 loads distilgpt2 ≈ 0.33 GB / gpt2 ≈ 0.52 GB",
)

#: The device under test. ``Device("rocm", 0)`` ↔ IREE HAL ``device_id = 1``.
DEVICE = etl.core.Device("rocm", 0)
CPU = etl.core.Device("cpu", 0)

#: fp32 parity budget **relative to the reference's absmax**: measured 2.6e-7 for
#: tiny_llama, 0.0 for tiny_gpt2, 2.4e-6 / 2.8e-6 for the 124M models — one budget
#: for all vocabularies, and accumulation-order noise cannot flake an exact zero.
REL_TOL = 1e-5

#: Packed-vs-unpacked on the SAME device is the same program up to a static slice
#: (measured bit-identical for both tiny models); the budget still allows a
#: compiler to reorder around the fused concatenation.
PACKED_REL_TOL = 1e-6

#: Golden sequence lengths: the ``tiny_*`` fixtures are 8, the real ones 32.
SEQ_LEN = 8
BIG_SEQ_LEN = 32

#: slug → golden fixture name (``tiny_llama`` is spelled after its HF repo there).
GOLDEN_FOR = {"tiny_llama": "tiny_random_llama", "tiny_gpt2": "tiny_gpt2",
              "distilgpt2": "distilgpt2", "gpt2": "gpt2"}

#: ``bench_logs/hf_goldens/`` resolved from the repository root (read-only area).
GOLDEN_DIR = Path(__file__).resolve().parents[2] / "bench_logs" / "hf_goldens"

#: Free-memory floors before a real-size checkpoint is loaded/placed (host tree +
#: device copy + numpy's workspace; measured peak RSS over both: ≈3.5 GiB).
MIN_FREE_BYTES = {"distilgpt2": 4 * 1024**3, "gpt2": 8 * 1024**3}

#: label → measured dict (build_s, place_s, run_first_s, run_best_s, parity…).
MEASURED: dict[str, dict[str, object]] = {}


# ---------------------------------------------------------------- helpers
def _label(slug: str, pack: bool) -> str:
    """The :data:`MEASURED` label of one ``(slug, packed?)`` device run."""
    return f"{slug}/{'packed' if pack else 'unpacked'}/rocm"


def _golden(slug: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read ``<repo>/bench_logs/hf_goldens/<slug>.npz`` as ``(input_ids, logits)``.

    ``None`` (a skipped *assertion*, never a failure) when the read-only fixture
    area is absent or carries no ``input_ids``/``logits`` pair (``allow_pickle``
    off)."""
    path = GOLDEN_DIR / f"{GOLDEN_FOR[slug]}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if "input_ids" not in archive.files or "logits" not in archive.files:
            return None
        return np.asarray(archive["input_ids"]), np.asarray(archive["logits"])


def _absmax(a: np.ndarray) -> float:
    """Largest absolute value of ``a`` (in fp64, so the scale never overflows)."""
    return float(np.max(np.abs(np.asarray(a, np.float64))))


def _parity(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """``(max-abs, max-abs relative to want's absmax)`` of two logit arrays."""
    delta = float(np.max(np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64))))
    return delta, delta / max(_absmax(want), 1e-45)


def _argmax_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two logit arrays agree on the argmax token at every position."""
    return bool(np.array_equal(np.asarray(a).argmax(-1), np.asarray(b).argmax(-1)))


def _readback(tensor: "etl.Tensor") -> np.ndarray:
    """One **explicit** device→host transfer, then host logits."""
    return tensor.to(CPU).numpy()


def _timings(fn, n: int) -> dict[str, float]:
    """``first`` / ``best-of-n`` / ``median`` wall time of ``fn`` in seconds.

    The first call is timed separately (it pays laziness: plan build, the HAL
    invoke path, allocator warm-up) then excluded from the sample. Nothing is
    asserted on these numbers."""
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    samples: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return {"first_s": first, "best_s": min(samples), "median_s": statistics.median(samples)}


def _record(label: str, **values: object) -> None:
    """Merge ``values`` into :data:`MEASURED` under ``label`` (later wins)."""
    MEASURED.setdefault(label, {}).update(values)


def _summary(label: str) -> str:
    """Render one :data:`MEASURED` entry as a compact message fragment."""
    measured = MEASURED.get(label)
    if measured is None:
        return f"[{label}] <not measured>"
    parts = [
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in measured.items()
    ]
    return f"[{label}] " + " ".join(parts)


def _free_bytes() -> int:
    """Host ``MemAvailable`` in bytes (``0`` when unreadable)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:  # pragma: no cover - non-Linux host: report unknown
        return 0
    return 0


def _require_memory(slug: str) -> None:
    """Skip (never fail) when too little host memory is free for ``slug``."""
    need = MIN_FREE_BYTES[slug]
    free = _free_bytes()
    if free and free < need:
        pytest.skip(
            f"{slug} needs ≥{need / 1024**3:.0f} GiB free host memory (host tree + "
            f"device copy); this host reports {free / 1024**3:.1f} GiB available"
        )


# --- memoised loaded checkpoints / forwards / executables / placements (one
# --- placement reused by several tests IS the documented pattern, so it is
# --- shared on purpose)
_LOADED: dict[str, LoadedModel] = {}
_FORWARDS: dict[tuple[str, int, bool], object] = {}
_EXECUTABLES: dict[tuple, tuple[etl.Executable, float]] = {}
_CACHES: dict[tuple, etl.CompileCache] = {}
_PLACED: dict[tuple[str, int, bool], dict[str, "etl.Tensor"]] = {}
_PLACED_IDS: dict[tuple[str, int], "etl.Tensor"] = {}


def loaded_for(slug: str) -> LoadedModel:
    """``load_hf`` the (cached) checkpoint for ``slug`` once per process.

    Loading answers only *what is this checkpoint?* — a static config plus the
    canonical host params — so the sequence length is **not** part of the key:
    the length specialises the *graph*, not the checkpoint (see
    :func:`forward_for`)."""
    loaded = _LOADED.get(slug)
    if loaded is not None:
        return loaded
    directory = hf_download.require_model(slug)  # pytest.skip when unavailable
    start = time.perf_counter()
    loaded = load_hf(directory)
    _LOADED[slug] = loaded
    _record(f"{slug}/host", load_s=time.perf_counter() - start,
            canonical_keys=len(loaded.params))
    return loaded


def forward_for(slug: str, seq_len: int, *, pack: bool = False) -> object:
    """The memoised ``@etl.defn`` forward for ``(slug, seq_len, packed?)``.

    ``(config, seq_len, batch_size, pack_weights)`` is the static specialisation
    one graph corresponds to, and :func:`etl_hf.forward_fn` returns the *same*
    object for the same specialisation — which is what makes two identical
    :func:`etl_hf.build` calls return the identical executable. ``build`` reads
    the length/packing back off the forward."""
    key = (slug, seq_len, bool(pack))
    fwd = _FORWARDS.get(key)
    if fwd is None:
        loaded = loaded_for(slug)
        fwd = forward_fn(loaded.config, seq_len=seq_len, pack_weights=bool(pack))
        _FORWARDS[key] = fwd
    return fwd


def executable_for(
    slug: str, seq_len: int, *, pack: bool = False, label: str = "", **options: object
) -> tuple[etl.Executable, float]:
    """Build (once) the iree executable for ``slug``; returns ``(exe, build_s)``.

    A *private* :class:`etl.CompileCache` per ``(slug, pack, label)`` keeps builds
    isolated from the process-wide default, and memoising here means the module
    pays each compile exactly once."""
    key = (slug, seq_len, bool(pack), label)
    cached = _EXECUTABLES.get(key)
    if cached is not None:
        return cached
    cache = _CACHES.setdefault(key, etl.CompileCache())
    start = time.perf_counter()
    exe = build(
        forward_for(slug, seq_len, pack=pack),
        loaded_for(slug).config,
        backend="iree",
        device=DEVICE,
        cache=cache,
        pack_weights=pack,
        **options,
    )
    build_s = time.perf_counter() - start
    _EXECUTABLES[key] = (exe, build_s)
    return exe, build_s


def numpy_executable_for(slug: str, seq_len: int, *, pack: bool = False) -> etl.Executable:
    """The numpy-interpreter executable for ``slug`` (the parity reference)."""
    key = (slug, seq_len, bool(pack), "numpy")
    cached = _EXECUTABLES.get(key)
    if cached is not None:
        return cached[0]
    exe = build(
        forward_for(slug, seq_len, pack=pack),
        loaded_for(slug).config,
        backend="numpy",
        cache=etl.CompileCache(),
        pack_weights=pack,
    )
    _EXECUTABLES[key] = (exe, 0.0)
    return exe


def host_tree(slug: str, seq_len: int, *, pack: bool = False) -> dict[str, np.ndarray]:
    """The host params tree an executable for ``(slug, pack)`` takes.

    ``pack=True`` packs **on the host first** (the supported order), so the
    explicit H2D step transfers 3–5 leaves instead of one per weight."""
    loaded = loaded_for(slug)
    if not pack:
        return loaded.params
    return pack_params(pack_plan(param_specs(loaded.config)), loaded.params)


def placed_tree(slug: str, seq_len: int, *, pack: bool = False) -> dict[str, "etl.Tensor"]:
    """Place (once) the host tree of ``(slug, pack)`` on :data:`DEVICE`, recording
    total and per-leaf cost (the first leaf pays the one-time HIP driver init)."""
    key = (slug, seq_len, bool(pack))
    placed = _PLACED.get(key)
    if placed is not None:
        return placed
    host = host_tree(slug, seq_len, pack=pack)
    per_leaf: list[float] = []
    placed = {}
    start = time.perf_counter()
    for name, array in host.items():
        leaf_start = time.perf_counter()
        placed.update(place_params({name: array}, DEVICE))
        per_leaf.append(time.perf_counter() - leaf_start)
    total = time.perf_counter() - start
    assert all(t.device == DEVICE for t in placed.values()), (
        f"{slug}: place_params must leave every leaf on {DEVICE!r}"
    )
    _PLACED[key] = placed
    _record(_label(slug, pack), staged_leaves=len(placed), place_total_s=total,
            canonical_leaves=len(loaded_for(slug).params),
            place_first_leaf_s=per_leaf[0], place_median_leaf_s=statistics.median(per_leaf))
    return placed

def placed_ids_for(slug: str, seq_len: int) -> "etl.Tensor":
    """Place (once) the fixture's ``(1, seq_len)`` ids on :data:`DEVICE`.

    The golden's own ``input_ids`` are used when the fixture exists (so one run
    answers both the numpy-parity and the golden-parity question); otherwise
    all-zero ids of the model's static shape."""
    key = (slug, seq_len)
    ids = _PLACED_IDS.get(key)
    if ids is not None:
        return ids
    golden = _golden(slug)
    golden_ids = np.zeros((1, seq_len), np.int32) if golden is None else golden[0]
    host = np.ascontiguousarray(golden_ids, dtype=np.int32)
    assert host.shape == (1, seq_len), f"{slug}: fixture ids must be (1, {seq_len}), got {host.shape}"
    ids = place_ids(host, DEVICE)
    _PLACED_IDS[key] = ids
    assert ids.device == DEVICE, f"place_ids must leave the ids on {DEVICE!r}, got {ids.device!r}"
    return ids


def device_run(slug: str, seq_len: int, *, pack: bool = False) -> "etl.Tensor":
    """One device-resident :func:`etl_hf.run` on the placed ``(params, ids)`` trees."""
    exe, _ = executable_for(slug, seq_len, pack=pack)
    return run(exe, placed_tree(slug, seq_len, pack=pack), placed_ids_for(slug, seq_len))

def assert_device_resident(out: "etl.Tensor", slug: str) -> None:
    """Assert an output stays on :data:`DEVICE` and refuses an implicit readback."""
    assert out.device == DEVICE, (
        f"{slug}: run outputs must STAY on the executable's device ({DEVICE!r}); got "
        f"{out.device!r} — an output copied to the host would be a placement violation"
    )
    with pytest.raises(etl.DeviceError) as excinfo:
        out.numpy()
    message = str(excinfo.value)
    assert "no implicit" in message and "rocm" in message, (
        f"{slug}: .numpy() on a rocm output must raise DeviceError naming the device "
        f"and the rule against implicit transfers; got {message!r}"
    )


def _numpy_logits(slug: str, seq_len: int, *, pack: bool = False) -> np.ndarray:
    """Run the numpy interpreter on the SAME params/ids (the parity reference)."""
    ids = placed_ids_for(slug, seq_len).to(CPU).numpy()
    return run(
        numpy_executable_for(slug, seq_len, pack=pack), host_tree(slug, seq_len, pack=pack), ids
    ).numpy()

def assert_oracle_parity(
    slug: str, seq_len: int, logits: np.ndarray, *, pack: bool = False
) -> None:
    """Assert device logits against the numpy interpreter AND the stored golden.

    Both comparisons use :data:`REL_TOL` relative to the reference's absmax and
    must agree on the argmax token. A missing golden fixture skips the test."""
    label = _label(slug, pack)
    numpy_logits = _numpy_logits(slug, seq_len, pack=pack)
    numpy_abs, numpy_rel = _parity(logits, numpy_logits)
    _record(label, max_abs_vs_numpy=numpy_abs, rel_vs_numpy=numpy_rel)
    assert numpy_rel <= REL_TOL, (
        f"{slug}: iree/rocm vs numpy interpreter relative-to-absmax = {numpy_rel:g} "
        f"exceeds {REL_TOL:g} (max-abs {numpy_abs:g} of {_absmax(numpy_logits):g}); "
        f"{_summary(label)}"
    )
    assert _argmax_equal(logits, numpy_logits), (
        f"{slug}: both backends must agree on the argmax token at every position; "
        f"{_summary(label)}"
    )

    golden = _golden(slug)
    if golden is None:
        pytest.skip(f"no golden fixture at {GOLDEN_DIR / (GOLDEN_FOR[slug] + '.npz')}")
    want = golden[1]
    golden_abs, golden_rel = _parity(logits, want)
    _record(label, max_abs_vs_golden=golden_abs, rel_vs_golden=golden_rel)
    assert golden_rel <= REL_TOL, (
        f"{slug}: iree/rocm vs the stored HF golden relative-to-absmax = {golden_rel:g} "
        f"exceeds {REL_TOL:g} (max-abs {golden_abs:g} of {_absmax(want):g}); "
        f"{_summary(label)}"
    )
    assert _argmax_equal(logits, want), (
        f"{slug}: device logits must reproduce the golden's argmax token at every "
        f"position; {_summary(label)}"
    )


# --- 1. the ROCm opt-level default (options only — no compiler, no GPU needed) ---
def test_rocm_defaults_to_opt_level_o1_llvm_cpu_keeps_o3() -> None:
    """``resolve_build_options`` injects ``O1`` for all-ROCm IREE targets only.

    ``iree-compile`` cannot translate these transformer graphs for the ``rocm``
    HAL at O2/O3 (the LayerNorm mean is a ``stablehlo.reduce`` over the hidden
    axis and the translation aborts with ``'func.func' op failed to distribute``),
    hence :data:`DEFAULT_ROCM_OPT_LEVEL`."""
    assert DEFAULT_ROCM_OPT_LEVEL == "O1", (
        f"the measured ROCm default must stay O1 (O2/O3 fail iree-compile), got "
        f"{DEFAULT_ROCM_OPT_LEVEL!r}"
    )
    assert DEFAULT_OPT_LEVEL == "O3", f"the non-ROCm default must stay O3, got {DEFAULT_OPT_LEVEL!r}"

    name, opts = resolve_build_options("iree", DEVICE, {})
    assert name == "iree"
    assert opts["opt_level"] == "O1", (
        f"an all-ROCm iree build must resolve opt_level='O1' (the codegen-safe "
        f"level); got {opts!r}"
    )
    assert "iree_compile_args" not in opts, (
        f"the llvm-cpu flags are llvm-cpu only and must NOT be injected for rocm; got {opts!r}"
    )
    assert "dot_precision" not in opts, (
        f"iree/rocm runs true fp32 by default — dot_precision is the explicit "
        f"xla/rocm knob and must never be injected; got {opts!r}"
    )

    # An explicit opt_level still wins outright.
    assert resolve_build_options("iree", DEVICE, {"opt_level": "O3"})[1]["opt_level"] == "O3"
    # llvm-cpu is unchanged: O3 plus the measured host-target flags.
    _, cpu_opts = resolve_build_options("iree", None, {})
    assert cpu_opts["opt_level"] == "O3", cpu_opts
    assert cpu_opts["iree_compile_args"] == list(DEFAULT_LLVM_CPU_COMPILE_ARGS), cpu_opts
    # A mixed target set (or cuda) is not "all rocm" — the O3 default stands.
    mixed = {"target_backends": ["rocm", "llvm-cpu"]}
    assert resolve_build_options("iree", DEVICE, mixed)[1]["opt_level"] == "O3"
    assert resolve_build_options("iree", etl.core.Device("cuda", 0), {})[1]["opt_level"] == "O3"
    # An explicit rocm target set keeps the O1 default even without a device.
    assert resolve_build_options("iree", None, {"target_backends": ["rocm"]})[1]["opt_level"] == "O1"
    # The numpy interpreter documents ignoring compile options: nothing injected.
    assert resolve_build_options("numpy", DEVICE, {})[1] == {}


# --- 2. the device + the 1-based HAL id mapping ---
@requires_rocm
def test_rocm_device_is_visible_and_hip_ids_are_one_based() -> None:
    """``etl.devices()`` sees the GPU and ``Device('rocm', 0)`` is HAL id **1**.

    IREE's ROCm/CUDA device ids are 1-based (``device_id = index + 1``); id 0 does
    not exist. Pinned through the HIP driver — the call the adapter's
    ``upload_tensor``/``load`` path makes."""
    import iree.runtime as rt  # local import: this test only runs with iree present

    devices = etl.devices()
    assert DEVICE.kind in [device.kind for device in devices], (
        f"Device('rocm', 0) must be visible through etl.devices(); got {devices}"
    )
    _record("environment/rocm", devices=",".join(str(device) for device in devices))

    driver = rt.get_driver("hip")
    assert driver.create_device(device_id=DEVICE.index + 1) is not None, (
        f"etl Device('rocm', {DEVICE.index}) must map to HAL device_id="
        f"{DEVICE.index + 1} (1-based); the HIP driver returned nothing"
    )
    with pytest.raises(ValueError) as excinfo:
        driver.create_device(device_id=0)
    assert "0" in str(excinfo.value), (
        f"device_id=0 must be rejected by the HIP driver (ids are 1-based); got "
        f"{excinfo.value!r}"
    )


# --- 3. device-resident forward for both v1 families (tiny real checkpoints) ---
@requires_rocm
@pytest.mark.parametrize("slug", ["tiny_llama", "tiny_gpt2"])
def test_device_resident_forward_matches_numpy_and_golden(slug: str) -> None:
    """The whole forward runs ON the GPU and its logits survive both oracles."""
    label = _label(slug, False)
    loaded = loaded_for(slug)
    exe, build_s = executable_for(slug, SEQ_LEN)
    assert len(exe.signature.input_specs) == len(loaded.params) + 1, (
        f"{slug}: the unpacked graph declares {len(loaded.params)} weight inputs + "
        f"ids, got {len(exe.signature.input_specs)}"
    )

    out = device_run(slug, SEQ_LEN)
    assert_device_resident(out, slug)
    logits = _readback(out)
    expected_shape = (1, SEQ_LEN, loaded.config.vocab_size)
    assert logits.shape == expected_shape, f"{slug}: logits shape {logits.shape} != {expected_shape}"
    assert logits.dtype == np.float32, f"{slug}: logits must be float32, got {logits.dtype}"
    assert np.isfinite(logits).all(), f"{slug}: device logits must be finite"

    start = time.perf_counter()
    again = _readback(device_run(slug, SEQ_LEN))
    _record(label, readback_first_s=time.perf_counter() - start)
    assert np.array_equal(logits, again), (
        f"{slug}: two device-resident runs on the same inputs must be bit-identical"
    )
    _record(label, build_s=build_s, **_timings(lambda: device_run(slug, SEQ_LEN), 5))
    assert_oracle_parity(slug, SEQ_LEN, logits)
    print(f"{label}: {_summary(label)}")


# --- 4. host inputs are rejected at the run boundary ---
@requires_rocm
def test_host_inputs_are_rejected_by_a_rocm_executable() -> None:
    """A non-cpu executable refuses host params AND host ids — never a hidden copy.

    The flip side of the device-resident test: placement must be explicit, so
    feeding the host tree (or host ids) raises :class:`etl.DeviceError` whose
    message names ``t.to(`` and the run boundary."""
    exe, _ = executable_for("tiny_llama", SEQ_LEN)
    host = host_tree("tiny_llama", SEQ_LEN)
    placed = placed_tree("tiny_llama", SEQ_LEN)
    device_ids = placed_ids_for("tiny_llama", SEQ_LEN)

    with pytest.raises(etl.DeviceError) as params_exc:
        run(exe, host, device_ids)
    params_message = str(params_exc.value)
    assert "no implicit" in params_message and "t.to(" in params_message, (
        f"a host params tree must be rejected with a DeviceError naming the explicit "
        f"transfer; got {params_message!r}"
    )
    with pytest.raises(etl.DeviceError) as ids_exc:
        run(exe, placed, np.zeros((1, SEQ_LEN), dtype=np.int32))
    ids_message = str(ids_exc.value)
    assert "no implicit" in ids_message and "t.to(" in ids_message, (
        f"host ids must be rejected the same way (both sides need placing); got "
        f"{ids_message!r}"
    )


# --- 5. the same-device loop: zero host round-trips, bit-identical repeats ---
@requires_rocm
def test_same_device_loop_stages_nothing_and_is_bit_identical() -> None:
    """≥5 runs on the SAME placed params/ids: no ``asdevicearray``, all identical.

    This is the documented iteration pattern for stateful workloads: place once,
    then feed each run's device outputs back as the next run's inputs. The spy on
    ``iree.runtime.asdevicearray`` (the adapter's ONLY host-input path) proves
    nothing is re-staged, and bit-identical logits prove the loop is stable."""
    import iree.runtime as rt

    exe, _ = executable_for("tiny_llama", SEQ_LEN)
    placed = placed_tree("tiny_llama", SEQ_LEN)
    device_ids = placed_ids_for("tiny_llama", SEQ_LEN)
    calls: list[int] = []
    original = rt.asdevicearray

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    rt.asdevicearray = spy  # type: ignore[assignment]
    try:
        outputs = [run(exe, placed, device_ids) for _ in range(6)]
        staged = len(calls)
    finally:
        rt.asdevicearray = original  # type: ignore[assignment]

    assert staged == 0, (
        f"the same-device loop must not stage any host input; "
        f"iree.runtime.asdevicearray was called {staged} time(s)"
    )
    assert all(out.device == DEVICE for out in outputs), (
        f"every loop iteration must stay device-resident; got "
        f"{[str(out.device) for out in outputs]}"
    )
    first = _readback(outputs[0])
    assert all(np.array_equal(_readback(out), first) for out in outputs[1:]), (
        "6 device-resident runs on the same inputs must be bit-identical"
    )
    _record("tiny_llama/same_device_loop", runs=len(outputs), asdevicearray_calls=staged)


# --- 6. packed + device-resident (pack on the host FIRST, then place) ---
@requires_rocm
@pytest.mark.parametrize("slug", ["tiny_llama", "tiny_gpt2"])
def test_packed_device_resident_matches_unpacked(slug: str) -> None:
    """Packing changes the SIGNATURE, not the numbers — on the GPU too.

    The packed tree is built on the host (``pack_params(pack_plan(
    param_specs(config)), params)``) and *then* placed, so ``place_params``
    stages 3–5 leaves instead of one per weight; the device logits are
    bit-identical (≤ ``PACKED_REL_TOL``) to the unpacked device run and within
    ``REL_TOL`` of the numpy interpreter."""
    label = _label(slug, True)
    loaded = loaded_for(slug)
    plan = pack_plan(param_specs(loaded.config))
    exe, build_s = executable_for(slug, SEQ_LEN, pack=True)
    assert len(exe.signature.input_specs) == len(plan.specs) + 1, (
        f"{slug}: the packed graph declares {len(plan.specs)} packed inputs + ids, "
        f"got {len(exe.signature.input_specs)}"
    )
    assert set(host_tree(slug, SEQ_LEN, pack=True)) == set(plan.specs), (
        f"{slug}: the host packed tree must use exactly the plan's inputs"
    )

    packed_device = placed_tree(slug, SEQ_LEN, pack=True)
    canonical = len(loaded.params)
    assert 3 <= len(packed_device) <= 5 < canonical, (
        f"{slug}: the packed placement must stage 3–5 leaves (the measured count), "
        f"fewer than the {canonical} canonical weights; got {len(packed_device)}"
    )

    out = device_run(slug, SEQ_LEN, pack=True)
    assert_device_resident(out, slug)
    packed_logits = _readback(out)
    packed_abs, packed_rel = _parity(packed_logits, _readback(device_run(slug, SEQ_LEN)))
    _record(label, build_s=build_s, max_abs_vs_unpacked_device=packed_abs,
            rel_vs_unpacked_device=packed_rel)
    assert packed_rel <= PACKED_REL_TOL, (
        f"{slug}: packed vs unpacked device logits relative-to-absmax = {packed_rel:g} "
        f"exceeds {PACKED_REL_TOL:g} (max-abs {packed_abs:g}); packing must be a static "
        f"slice, not a different program; {_summary(label)}"
    )
    assert_oracle_parity(slug, SEQ_LEN, packed_logits, pack=True)
    print(f"{label}: {_summary(label)}")


# --- 7. rocm fp32 dot is TRUE fp32 (no dot_precision needed, unlike xla/rocm) ---
@requires_rocm
def test_rocm_dot_is_true_fp32_without_dot_precision() -> None:
    """``dot_precision="HIGHEST"`` is a no-op on iree/rocm — the default is fp32.

    On xla/rocm an fp32 ``dot`` runs at TF32-class (10-bit mantissa) precision by
    default and needs the explicit knob; iree's ROCm codegen has no such
    reduction, so the flag changes nothing. Pinned by bit-identical logits (a
    TF32-class difference would be ~1e-3, orders above ``REL_TOL``)."""
    default_logits = _readback(device_run("tiny_llama", SEQ_LEN))
    highest_exe, highest_build_s = executable_for(
        "tiny_llama", SEQ_LEN, label="dot_precision=HIGHEST", dot_precision="HIGHEST"
    )
    highest_logits = _readback(
        run(highest_exe, placed_tree("tiny_llama", SEQ_LEN),
            placed_ids_for("tiny_llama", SEQ_LEN))
    )
    delta, relative = _parity(highest_logits, default_logits)
    _record("tiny_llama/dot_precision", build_s=highest_build_s, max_abs_vs_default=delta,
            rel_vs_default=relative,
            bit_identical=bool(np.array_equal(highest_logits, default_logits)))
    assert relative <= REL_TOL, (
        f"dot_precision='HIGHEST' must be a no-op on iree/rocm (the default is already "
        f"fp32); logits differ by max-abs {delta:g} (relative {relative:g}); "
        f"{_summary('tiny_llama/dot_precision')}"
    )
    print(f"tiny_llama/dot_precision: {_summary('tiny_llama/dot_precision')}")


# --- 8. real checkpoints, device-resident + packed (opt-in, best-of-3) ---
def _assert_real_checkpoint_device_resident(slug: str, runs: int) -> None:
    """Shared body of the two real-checkpoint cases (``distilgpt2`` / ``gpt2``).

    Builds the **packed** executable, stages only the packed leaves, runs once
    device-resident, then asserts ``runs`` consecutive bit-identical iterations
    and parity against the numpy interpreter and the stored golden."""
    _require_memory(slug)
    label = _label(slug, True)
    loaded = loaded_for(slug)
    params = loaded.params
    plan = pack_plan(param_specs(loaded.config))
    assert len(plan.specs) == 3, f"{slug} must pack to 3 inputs, got {len(plan.specs)}"
    exe, build_s = executable_for(slug, BIG_SEQ_LEN, pack=True)
    declared = len(exe.signature.input_specs)
    assert declared == 4, (
        f"{slug} packed must declare 3 weight inputs + ids = 4, got {declared} (the "
        f"unpacked graph declares {len(params) + 1} — the shape that dies on llvm-cpu "
        f"above ~161 consumed inputs)"
    )
    packed_device = placed_tree(slug, BIG_SEQ_LEN, pack=True)
    assert len(packed_device) == 3, (
        f"{slug}: place_params must stage only the 3 packed leaves, got "
        f"{len(packed_device)} (vs {len(params)} canonical weights)"
    )

    out = device_run(slug, BIG_SEQ_LEN, pack=True)
    assert_device_resident(out, slug)
    logits = _readback(out)
    expected_shape = (1, BIG_SEQ_LEN, loaded.config.vocab_size)
    assert logits.shape == expected_shape, f"{slug}: logits shape {logits.shape} != {expected_shape}"
    assert np.isfinite(logits).all(), f"{slug}: device logits must be finite"

    _record(label, build_s=build_s, declared_inputs=declared, canonical_keys=len(params))
    _record(label, **_timings(lambda: device_run(slug, BIG_SEQ_LEN, pack=True), 3))
    iterations = [_readback(device_run(slug, BIG_SEQ_LEN, pack=True)) for _ in range(runs)]
    assert all(np.array_equal(run, iterations[0]) for run in iterations[1:]), (
        f"{slug}: {runs} consecutive device-resident runs at seq_len={BIG_SEQ_LEN} must "
        f"be bit-identical (a crash here would be the llvm-cpu input-count cliff; ROCm "
        f"has no such cap)"
    )
    _record(label, stable_runs=len(iterations))
    assert_oracle_parity(slug, BIG_SEQ_LEN, iterations[0], pack=True)
    print(f"{label}: {_summary(label)}")


@_BIG
@requires_rocm
def test_distilgpt2_device_resident_packed() -> None:
    """82M-param ``distilgpt2`` at ``seq_len=32``: 3 packed inputs, on the GPU.

    The LayerNorm-reduce shape that fails ``iree-compile`` at O2/O3 — exactly why
    the ROCm default is O1 — runs device-resident here."""
    _assert_real_checkpoint_device_resident("distilgpt2", runs=5)


@_BIG
@requires_rocm
def test_real_gpt2_device_resident_packed_is_stable() -> None:
    """124M ``gpt2`` at ``seq_len=32`` device-resident: 4 inputs, 10 stable runs.

    On llvm-cpu the *unpacked* 197-input graph SIGSEGVs on the first run (IREE's
    Python runtime cliff above ~161 consumed inputs). ROCm has no such cap, so the
    packed 4-input program runs repeated same-device iterations without a crash
    *and* reproduces the numpy interpreter and the HF golden."""
    _assert_real_checkpoint_device_resident("gpt2", runs=10)
