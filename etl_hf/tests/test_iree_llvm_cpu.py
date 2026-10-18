"""Compiled-backend parity suite: the WHOLE ``etl_hf`` forward on ``iree``/llvm-cpu.

The package contract is "**the whole model forward pass is ONE etl graph**"
(``./etl_hf/CONTEXT.md``). :mod:`etl_hf.tests.test_export` proves that at the
StableHLO-text level with numpy only; this suite proves it *behaviourally* — the
traced graph is lowered, compiled and executed as one IREE program on the
``llvm-cpu`` target, and its logits are compared against

* each family's independent pure-numpy oracle (``FamilySpec.reference``), and
* the numpy interpreter backend's logits for the identical params/ids,

for both v1 families (``llama`` with GQA+RoPE+SwiGLU, ``gpt2`` with
LayerNorm+``gelu_new``+tied head) and for a **real** HuggingFace checkpoint
(``tiny_llama``) loaded through :func:`etl_hf.build.load_hf`.

It also pins the two *measured* pitfalls recorded in
``./bench_logs/hf_feasibility_findings.md`` and documented in
:mod:`etl_hf.build`:

* **pitfall #1 — ``iree_compile_args`` are a COMPILE-stage option.** Passing
  them to ``etl.lower`` alone is silently ineffective; they must reach
  ``etl.compile``. This suite pins that by **spying on ``etl.pipeline.compile``**
  (:func:`test_iree_compile_args_reach_the_compile_stage`) and asserting the
  injected ``--iree-llvmcpu-target-cpu=host`` / ``--iree-opt-data-tiling`` flags
  are in the kwargs the compile stage actually receives — a deterministic
  plumbing check, never a flaky timing assertion.
* **pitfall #2 — llvm-cpu needs ``--iree-llvmcpu-target-cpu=host``.** The IREE
  default ``generic`` target is 12–19× slower for these graphs. The pure-options
  resolution assertion (:func:`test_default_iree_llvm_cpu_compile_options`) is
  intentionally **not** gated behind the iree import, so the default matrix is
  pinned even on a machine without a compiler installed.

The suite also pins **determinism** (same executable + same inputs ⇒
bit-identical logits) and the **one-program structure** (one compiled entry
function, one exported IR function, zero ``external_call``/``runtime_call`` ops —
so no external-kernel splitting is involved).

Gating: every test that actually compiles is marked ``@requires_iree`` (a
module-level import probe), so the file skips cleanly — never errors — under a
plain ``uv run --with pytest`` without the ``iree`` extra. The options-only test
runs unconditionally.

Measured numbers (compile wall time, best-of-N run time, parity) land in
:data:`MEASURED` and in the failure-message text, so an operator can read them
without re-running anything.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import etl
import etl.pipeline as etl_pipeline
from etl_hf.build import (
    DEFAULT_LLVM_CPU_COMPILE_ARGS,
    build_model,
    load_hf,
    resolve_build_options,
    run_model,
)
from etl_hf.config import ModelConfig
from etl_hf.models import gpt2 as gpt2_module
from etl_hf.models import llama as llama_module

# ``hf_download`` lives next to this file (stdlib only; importing it is
# network-free — only ``download_model``/``require_model`` touch the network).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
try:  # the module is always present next to this file; guard only for safety
    import hf_download
except Exception:  # noqa: BLE001 - a missing helper must degrade to a skip, not fail collection
    hf_download = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# module-level iree guard (a clean skip, never an error, when iree is absent)
# ---------------------------------------------------------------------------
def _iree_available() -> bool:
    """Probe for the IREE Python bindings (compiler + runtime) without importing
    them into ``etl_hf`` (this is a test module; ``import etl_hf`` stays
    numpy-only — that property is asserted elsewhere in a subprocess)."""
    try:
        import iree.compiler
        import iree.runtime  # noqa: F401 - presence probe only (last import carries the F401)
    except Exception:  # noqa: BLE001 - an unusable/partial iree install must skip, not error
        return False
    return True


IREE_AVAILABLE = _iree_available()
requires_iree = pytest.mark.skipif(
    not IREE_AVAILABLE,
    reason="iree (iree.compiler / iree.runtime) is not installed; "
    "run with `uv run --extra iree --with pytest`",
)


# ---------------------------------------------------------------------------
# static configs (small, so the compile is fast — one config = one graph)
# ---------------------------------------------------------------------------
_SEQ = 8
_BATCH = 1

#: 1-layer grouped-query Llama: 16 hidden / 4 heads / 2 kv heads, SwiGLU.
LLAMA_TINY: dict[str, object] = {
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 32,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}

#: 2-layer GPT-2 (gelu_new, learned positions, tied lm_head).
GPT2_TINY: dict[str, object] = {
    "model_type": "gpt2",
    "n_embd": 16,
    "n_layer": 2,
    "n_head": 4,
    "n_inner": 32,
    "n_vocab": 32,
    "n_positions": 16,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": True,
}

_CONFIGS = {"llama": LLAMA_TINY, "gpt2": GPT2_TINY}
_FAMILY_MODULES = {"llama": llama_module, "gpt2": gpt2_module}

#: fp32 parity budget for iree-vs-numpy / iree-vs-reference. The measured
#: max-abs for these graphs is ~1e-8 (see :data:`MEASURED`); 1e-4 is a
#: deliberately loose ceiling that still catches a real divergence.
PARITY_ATOL = 1e-4

#: label -> measured dict (compile_s, run_best_s, max_abs_vs_reference,
#: max_abs_vs_numpy, ...). Filled while the tests run, read in the messages.
MEASURED: dict[str, dict[str, float]] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def config_for(family: str) -> ModelConfig:
    """Build the tiny :class:`ModelConfig` for ``family`` (``"llama"``/``"gpt2"``)."""
    return ModelConfig.from_dict(_CONFIGS[family])


def make_params(cfg: ModelConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded canonical params matching ``cfg``'s own family specs.

    Norm scales (1-D specs) are ``1.0`` so activations stay well-conditioned;
    matrices get small seeded gaussians. Exact values do not matter — graph and
    reference consume the same tree, so parity is the claim.
    """
    specs = _FAMILY_MODULES[cfg.family].param_specs(cfg)
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        shape = tuple(int(dim) for dim in spec.shape)
        if len(shape) == 1:
            params[name] = np.ones(shape, dtype=np.float32)
        else:
            params[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return params


def make_ids(cfg: ModelConfig, batch: int = _BATCH, seq: int = _SEQ, seed: int = 1) -> np.ndarray:
    """Seeded ``(batch, seq)`` int32 token ids inside the vocabulary."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference (the number reported in parity assertions)."""
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def best_of(fn, n: int = 5) -> float:
    """Best-of-``n`` wall time of ``fn`` in seconds (one warm-up call first)."""
    fn()  # warm up: exclude first-call laziness (host map, plan build, ...)
    best = math.inf
    for _ in range(n):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def count_ops(graph: object) -> dict[str, int]:
    """Op-name histogram of a traced graph, walking nested regions.

    Mirrors ``etl_hf.tests.test_export._count_ops`` (kept local so this module
    does not import another test module). These model graphs are straight-line,
    so the walk finds no nested regions — written generically regardless.
    """
    counts: dict[str, int] = {}

    def walk(block: object) -> None:
        for op in block.ops:  # type: ignore[attr-defined]
            name = op.opdef.name
            counts[name] = counts.get(name, 0) + 1
            for region in op.regions:
                for nested in region.blocks:
                    walk(nested)

    walk(graph.module.functions[0].entry_block)  # type: ignore[attr-defined]
    return counts


def summary(label: str) -> str:
    """Render one :data:`MEASURED` entry as a compact message fragment."""
    m = MEASURED.get(label)
    if m is None:
        return f"[{label}] <not measured>"
    parts = [f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}" for key, value in m.items()]
    return f"[{label}] " + " ".join(parts)


# ---------------------------------------------------------------------------
# executable memoisation (share the ~1 s compiles across tests; a per-family
# private CompileCache keeps the builds isolated from the process-wide default)
# ---------------------------------------------------------------------------
_COMPILE_CACHES: dict[tuple[str, str], etl.CompileCache] = {}
_EXECUTABLES: dict[tuple[str, str], etl.Executable] = {}


def iree_executable(family: str) -> etl.Executable:
    """The IREE/llvm-cpu executable for ``family`` (compiled once, cached).

    The compile wall time is recorded in :data:`MEASURED` on the first build.
    """
    key = (family, "iree")
    exe = _EXECUTABLES.get(key)
    if exe is not None:
        return exe
    cache = _COMPILE_CACHES.setdefault(key, etl.CompileCache())
    start = time.perf_counter()
    exe = build_model(config_for(family), _SEQ, backend="iree", cache=cache)
    elapsed = time.perf_counter() - start
    _EXECUTABLES[key] = exe
    MEASURED.setdefault(family, {})["compile_s"] = elapsed
    return exe


def numpy_executable(family: str) -> etl.Executable:
    """The numpy-interpreter executable for ``family`` (built once, cached)."""
    key = (family, "numpy")
    exe = _EXECUTABLES.get(key)
    if exe is not None:
        return exe
    cache = _COMPILE_CACHES.setdefault(key, etl.CompileCache())
    exe = build_model(config_for(family), _SEQ, backend="numpy", cache=cache)
    _EXECUTABLES[key] = exe
    return exe


# ---------------------------------------------------------------------------
# 1. whole-graph iree compile + run, llama
# ---------------------------------------------------------------------------
@requires_iree
def test_llama_whole_graph_compiles_and_runs_on_iree() -> None:
    """Tiny Llama forward: ONE iree program, parity vs reference and numpy."""
    cfg = config_for("llama")
    params, ids = make_params(cfg), make_ids(cfg)

    exe = iree_executable("llama")
    iree_logits = run_model(exe, params, ids).numpy()

    assert iree_logits.shape == (_BATCH, _SEQ, cfg.vocab_size), (
        f"iree logits shape {iree_logits.shape} != expected "
        f"{(_BATCH, _SEQ, cfg.vocab_size)}"
    )
    assert np.isfinite(iree_logits).all(), "iree logits must be finite"

    reference = _FAMILY_MODULES["llama"].reference(cfg, params, ids)
    numpy_logits = run_model(numpy_executable("llama"), params, ids).numpy()

    d_ref = max_abs(iree_logits, reference)
    d_numpy = max_abs(iree_logits, numpy_logits)
    run_best = best_of(lambda: run_model(exe, params, ids))
    MEASURED["llama"].update(
        run_best_s=run_best,
        max_abs_vs_reference=d_ref,
        max_abs_vs_numpy=d_numpy,
    )

    assert d_ref <= PARITY_ATOL, (
        f"llama iree vs pure-numpy reference max-abs = {d_ref:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('llama')}"
    )
    assert d_numpy <= PARITY_ATOL, (
        f"llama iree vs numpy interpreter max-abs = {d_numpy:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('llama')}"
    )
    print(f"llama: {summary('llama')}")


# ---------------------------------------------------------------------------
# 2. whole-graph iree compile + run, gpt2
# ---------------------------------------------------------------------------
@requires_iree
def test_gpt2_whole_graph_compiles_and_runs_on_iree() -> None:
    """Tiny GPT-2 forward: ONE iree program, parity vs reference and numpy."""
    cfg = config_for("gpt2")
    params, ids = make_params(cfg), make_ids(cfg)

    exe = iree_executable("gpt2")
    iree_logits = run_model(exe, params, ids).numpy()

    assert iree_logits.shape == (_BATCH, _SEQ, cfg.vocab_size), (
        f"iree logits shape {iree_logits.shape} != expected "
        f"{(_BATCH, _SEQ, cfg.vocab_size)}"
    )
    assert np.isfinite(iree_logits).all(), "iree logits must be finite"

    reference = _FAMILY_MODULES["gpt2"].reference(cfg, params, ids)
    numpy_logits = run_model(numpy_executable("gpt2"), params, ids).numpy()

    d_ref = max_abs(iree_logits, reference)
    d_numpy = max_abs(iree_logits, numpy_logits)
    run_best = best_of(lambda: run_model(exe, params, ids))
    MEASURED["gpt2"].update(
        run_best_s=run_best,
        max_abs_vs_reference=d_ref,
        max_abs_vs_numpy=d_numpy,
    )

    assert d_ref <= PARITY_ATOL, (
        f"gpt2 iree vs pure-numpy reference max-abs = {d_ref:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('gpt2')}"
    )
    assert d_numpy <= PARITY_ATOL, (
        f"gpt2 iree vs numpy interpreter max-abs = {d_numpy:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('gpt2')}"
    )
    print(f"gpt2: {summary('gpt2')}")


# ---------------------------------------------------------------------------
# 3. pitfall #2 (pure options — NOT gated behind the iree import)
# ---------------------------------------------------------------------------
def test_default_iree_llvm_cpu_compile_options() -> None:
    """The measured llvm-cpu default flags + ``opt_level`` are injected, and an
    explicit override wins outright — asserted on the resolved options only, so
    it needs no compiler installed."""
    assert DEFAULT_LLVM_CPU_COMPILE_ARGS == (
        "--iree-llvmcpu-target-cpu=host",
        "--iree-opt-data-tiling",
    ), "the measured llvm-cpu flags must stay the documented pair"

    name, opts = resolve_build_options("iree", None, {})
    assert name == "iree"
    assert opts["opt_level"] == "O3", opts
    assert opts["iree_compile_args"] == list(DEFAULT_LLVM_CPU_COMPILE_ARGS), opts
    for flag in DEFAULT_LLVM_CPU_COMPILE_ARGS:
        assert flag in opts["iree_compile_args"], (
            f"default {flag!r} must be injected for an iree/llvm-cpu build; got "
            f"{opts['iree_compile_args']!r}"
        )

    # An explicit ``iree_compile_args`` overrides the defaults entirely: only
    # the caller's flags survive (the ``generic`` target is the slow default
    # pitfall #2 fixes, so overriding to it must remove the host flags).
    explicit = ["--iree-llvmcpu-target-cpu=generic"]
    name2, opts2 = resolve_build_options("iree", None, {"iree_compile_args": explicit})
    assert name2 == "iree"
    assert opts2["iree_compile_args"] == explicit, opts2
    assert opts2["iree_compile_args"] is explicit, "explicit list is returned verbatim, never copied"
    assert "--iree-llvmcpu-target-cpu=host" not in opts2["iree_compile_args"]
    assert "--iree-opt-data-tiling" not in opts2["iree_compile_args"]


# ---------------------------------------------------------------------------
# 4. pitfall #1 (behavioural plumbing pin — the spy)
# ---------------------------------------------------------------------------
@requires_iree
def test_iree_compile_args_reach_the_compile_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """``iree_compile_args`` must reach the COMPILE stage, not only ``lower``.

    Mechanism: ``etl.build_cached`` binds ``compile``/``lower``/``load`` from
    ``etl.pipeline`` *inside* its compute function (``from etl.pipeline import
    ...``) at call time, so monkeypatching ``etl.pipeline.compile`` intercepts
    every compile the build performs. The spy records the actual ``**kwargs``
    and delegates, giving a deterministic assertion on the plumbing that does not
    depend on any (machine-dependent) timing difference. A *fresh* private
    ``CompileCache`` forces a real compile so the spy always observes a call.

    It additionally checks the behavioural consequence — a ``generic``-target
    build produces matching logits — and records (but never asserts on) the
    host-vs-generic run times.
    """
    cfg = config_for("llama")
    params, ids = make_params(cfg), make_ids(cfg)

    records: list[dict[str, object]] = []
    original = etl_pipeline.compile

    def spy(lowered, backend=None, **options):
        records.append(dict(options))
        return original(lowered, backend=backend, **options)

    monkeypatch.setattr(etl_pipeline, "compile", spy)

    # (a) the etl_hf defaults (host llvm-cpu target) — fresh cache => real compile.
    host_exe = build_model(cfg, _SEQ, backend="iree", cache=etl.CompileCache())
    assert len(records) == 1, (
        f"expected exactly one compile-stage call for a fresh build, saw "
        f"{len(records)}; kwargs forwarded were {records}"
    )
    fired = records[0]
    assert fired.get("iree_compile_args") == list(DEFAULT_LLVM_CPU_COMPILE_ARGS), (
        "pitfall #1: the injected llvm-cpu flags did NOT reach etl.pipeline.compile; "
        f"compile-stage kwargs were {sorted(fired)} with "
        f"iree_compile_args={fired.get('iree_compile_args')!r}"
    )
    assert fired.get("opt_level") == "O3", fired
    assert "target_backends" in fired, (
        f"build_cached resolved target_backends before compile; got {sorted(fired)}"
    )

    # (b) an explicit override also reaches compile verbatim (only that flag).
    generic_args = ["--iree-llvmcpu-target-cpu=generic"]
    generic_exe = build_model(
        cfg, _SEQ, backend="iree", cache=etl.CompileCache(), iree_compile_args=generic_args
    )
    assert len(records) == 2, f"expected a second compile-stage call, saw {len(records)}"
    assert records[1].get("iree_compile_args") == generic_args, (
        f"an explicit iree_compile_args must reach compile verbatim; got "
        f"{records[1].get('iree_compile_args')!r}"
    )

    # Behavioural consequence: host and generic targets agree (plumbing intact).
    host_logits = run_model(host_exe, params, ids).numpy()
    generic_logits = run_model(generic_exe, params, ids).numpy()
    d = max_abs(host_logits, generic_logits)
    t_host = best_of(lambda: run_model(host_exe, params, ids))
    t_generic = best_of(lambda: run_model(generic_exe, params, ids))
    MEASURED["pitfall1"] = {
        "max_abs_host_vs_generic": d,
        "t_host_best_s": t_host,
        "t_generic_best_s": t_generic,
    }
    # NOTHING is asserted about timing (machine-dependent); the numbers ride
    # along in the message for the operator. Only parity is asserted.
    assert d <= PARITY_ATOL, (
        f"host-target vs generic-target iree logits differ by max-abs {d:g} > "
        f"{PARITY_ATOL:g} (plumbing changed semantics); {summary('pitfall1')} "
        f"— for reference, t_host={t_host:.3g}s vs t_generic={t_generic:.3g}s "
        f"(report only, not asserted)"
    )
    print(f"pitfall#1: {summary('pitfall1')}")


# ---------------------------------------------------------------------------
# 5. determinism
# ---------------------------------------------------------------------------
@requires_iree
def test_iree_executable_is_deterministic() -> None:
    """One iree executable + identical inputs ⇒ BIT-IDENTICAL logits across runs."""
    cfg = config_for("llama")
    params, ids = make_params(cfg), make_ids(cfg)
    exe = iree_executable("llama")

    first = run_model(exe, params, ids).numpy()
    second = run_model(exe, params, ids).numpy()

    d = max_abs(first, second)
    assert np.array_equal(first, second), (
        f"the same iree executable run twice on the same inputs must be "
        f"bit-identical; max-abs difference = {d:g}"
    )


# ---------------------------------------------------------------------------
# 6. a real checkpoint through iree (guarded: skips cleanly when offline)
# ---------------------------------------------------------------------------
@requires_iree
@pytest.mark.skipif(hf_download is None, reason="hf_download helper module unavailable")
def test_real_tiny_llama_checkpoint_through_iree() -> None:
    """``load_hf`` of a real ``tiny_llama`` checkpoint runs on iree at parity.

    ``require_model`` downloads (or reuses a cached) checkpoint and ``pytest.skip``s
    on any network/filesystem failure, so an offline machine skips cleanly.
    """
    model_dir = hf_download.require_model("tiny_llama")
    model = load_hf(model_dir, seq_len=_SEQ)
    params = model.params()
    ids = np.zeros((model.batch_size, model.seq_len), dtype=np.int32)

    iree_cache = etl.CompileCache()
    start = time.perf_counter()
    model.build(backend="iree", cache=iree_cache)
    compile_s = time.perf_counter() - start

    iree_logits = model.forward(input_ids=ids, backend="iree", cache=iree_cache)
    assert iree_logits.shape == (model.batch_size, model.seq_len, model.config.vocab_size), (
        f"real-checkpoint iree logits shape {iree_logits.shape} is wrong"
    )
    assert np.isfinite(iree_logits).all(), "real-checkpoint iree logits must be finite"

    reference = model.reference(input_ids=ids)
    numpy_logits = model.forward(input_ids=ids, backend="numpy", cache=etl.CompileCache())

    d_ref = max_abs(iree_logits, reference)
    d_numpy = max_abs(iree_logits, numpy_logits)
    run_best = best_of(lambda: run_model(
        model.build(backend="iree", cache=iree_cache), params, ids
    ))
    MEASURED["tiny_llama"] = {
        "compile_s": compile_s,
        "run_best_s": run_best,
        "max_abs_vs_reference": d_ref,
        "max_abs_vs_numpy": d_numpy,
    }

    assert d_ref <= PARITY_ATOL, (
        f"real tiny_llama iree vs reference max-abs = {d_ref:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('tiny_llama')}"
    )
    assert d_numpy <= PARITY_ATOL, (
        f"real tiny_llama iree vs numpy interpreter max-abs = {d_numpy:g} exceeds "
        f"{PARITY_ATOL:g}; {summary('tiny_llama')}"
    )
    print(f"tiny_llama: {summary('tiny_llama')}")


# ---------------------------------------------------------------------------
# 7. the graph is ONE program (structural pin)
# ---------------------------------------------------------------------------
@requires_iree
def test_model_graph_is_a_single_program() -> None:
    """The whole forward is one compiled program — no external-call splitting.

    Pins, on the llama executable:

    * one flat tensor input per canonical param + the ``input_ids`` leaf;
    * exactly ONE compiled entry function (``("main",)``);
    * the backend executable is the plain ``IreeExecutable`` — an
      ``external_call`` graph would have been split into
      ``IreeExternalExecutable`` + a segment plan;
    * the traced IR is a single function containing zero ``external_call`` /
      ``runtime_call`` ops.
    """
    cfg = config_for("llama")
    params = make_params(cfg)
    exe = iree_executable("llama")

    assert len(exe.signature.input_specs) == len(params) + 1, (
        f"expected {len(params)} param tensor inputs + 1 input_ids leaf, got "
        f"{len(exe.signature.input_specs)}"
    )
    assert exe.functions == ("main",), (
        f"the whole forward must compile to exactly one entry function; got "
        f"{exe.functions!r} (more than one means the graph was split)"
    )
    backend_exec_name = type(exe.backend_executable).__name__
    assert backend_exec_name == "IreeExecutable", (
        f"expected the non-split IreeExecutable, got {backend_exec_name!r} "
        f"(an external-call graph is split into IreeExternalExecutable)"
    )

    specs = llama_module.param_specs(cfg)
    graph = etl.trace(
        llama_module.graph(cfg, _SEQ, _BATCH),
        specs,
        etl.TensorSpec((_BATCH, _SEQ), etl.int32),
    )
    assert len(graph.module.functions) == 1, (
        f"the traced model graph must be a single IR function; got "
        f"{len(graph.module.functions)}"
    )
    counts = count_ops(graph)
    assert counts.get("external_call", 0) == 0, (
        f"the model graph must contain zero external_call ops; histogram "
        f"top: {sorted(counts.items(), key=lambda kv: -kv[1])[:8]}"
    )
    assert counts.get("runtime_call", 0) == 0, (
        f"the model graph must contain zero runtime_call ops; histogram "
        f"top: {sorted(counts.items(), key=lambda kv: -kv[1])[:8]}"
    )
