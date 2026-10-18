"""Packed-weights suite: ``pack_weights=True`` is a *transparent* re-bucketing.

The package contract is "the whole model forward pass is ONE etl graph whose
weights are ordinary graph **inputs**" (``./etl_hf/CONTEXT.md``). IREE's Python
runtime SIGSEGVs on a program that *consumes* more than ~161 distinct inputs, and
the 124M ``gpt2`` declares 197 (196 weights + ids) — so the last mile needed a way
to feed the same weights as a handful of inputs. :mod:`etl_hf.pack` does that
generically (concatenate along axis 0 per ``(dtype, rank, shape[1:])`` bucket,
slice back in-graph with **static** ``etl.slice``).

The property this suite pins is *equivalence*, not a new numerics claim:

* the plan covers every canonical key exactly once, with deterministic names and
  contiguous static offsets (``test_every_canonical_key_is_covered_exactly_once``,
  ``test_plan_is_deterministic``);
* host packing/unpacking is **bit-identical** per key (``numpy`` mirror of the
  in-graph slices: ``test_host_round_trip_is_bit_identical``);
* a packed executable's logits are **bit-identical** to the unpacked one on the
  numpy interpreter (concatenation + static slice is exact) and within fp32
  tolerance on ``iree``/llvm-cpu (``test_numpy_backend_*``,
  ``test_iree_*``);
* unpacked stays the DEFAULT (no silent behaviour change), and mixing the two
  tree shapes gets an actionable ``pack_weights`` error rather than a bare etl
  pytree message (``test_unpacked_is_the_default``,
  ``test_run_model_reports_the_packing_mismatch``);
* nothing is baked: the packed graph is still all graph inputs, so the
  StableHLO export keeps the same ``stablehlo.constant`` ceiling
  (``test_packed_graph_bakes_no_weights``);
* the real 124M ``gpt2`` — the case packing exists for — builds a 3-input packed
  executable whose numpy forward reproduces the stored HuggingFace golden
  (``test_real_size_gpt2_packed_matches_golden``, ``ETL_HF_BIG=1``).

Gating: the compiled pin is marked ``@requires_iree`` (module-level import probe),
so it skips cleanly without the ``iree`` extra; the real-size pin needs
``ETL_HF_BIG=1`` plus a readable golden fixture and a downloadable checkpoint
(``hf_download.require_model`` turns "cannot fetch" into a skip). Everything else
is pure numpy and always runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import etl
from etl.backends import stablehlo
from etl_hf import pack as pack_module
from etl_hf.build import HFModel, build_model, load_hf, place_params, run_model
from etl_hf.config import ModelConfig
from etl_hf.errors import HFUnsupportedError, HFWeightError
from etl_hf.models import gpt2 as gpt2_module
from etl_hf.models import llama as llama_module
from etl_hf.tests import hf_download

# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def _iree_available() -> bool:
    """Probe for the IREE Python bindings without importing them into ``etl_hf``."""
    try:
        import iree.compiler
        import iree.runtime  # noqa: F401 - presence probe only
    except Exception:  # noqa: BLE001 - a partial install must skip, not error
        return False
    return True


IREE_AVAILABLE = _iree_available()
requires_iree = pytest.mark.skipif(
    not IREE_AVAILABLE,
    reason="iree (iree.compiler / iree.runtime) is not installed; "
    "run with `uv run --extra iree --with pytest`",
)

#: Real-size checkpoints are opt-in (``gpt2`` is ~0.52 GB of weights).
_BIG = pytest.mark.skipif(
    os.environ.get("ETL_HF_BIG") != "1",
    reason="real-size checkpoints are opt-in: set ETL_HF_BIG=1 "
    "(downloads gpt2 ≈ 0.52 GB of weights)",
)

#: The IREE runtime limit packing exists for: a program that CONSUMES more than
#: this many distinct inputs SIGSEGVs on the first run (measured bracket 161/162).
IREE_CONSUMED_INPUT_CLIFF = 161

#: fp32 parity budget for iree-vs-numpy interpreter on the same graph.
IREE_ATOL = 1e-6

#: Reference-free float32 logit tolerance against the stored HF golden.
GOLDEN_ATOL = 1e-3
GOLDEN_RTOL = 1e-5

_SEQ = 8
_BATCH = 1

# ---------------------------------------------------------------------------
# static configs — vocab_size is deliberately NOT equal to intermediate_size, so
# every bucketing class is distinguishable (a collision would merge two buckets
# and hide an offset bug).
# ---------------------------------------------------------------------------
LLAMA_UNTIED: dict[str, object] = {
    "model_type": "llama",
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "vocab_size": 48,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}
LLAMA_TIED: dict[str, object] = dict(LLAMA_UNTIED, tie_word_embeddings=True)

GPT2_UNTIED: dict[str, object] = {
    "model_type": "gpt2",
    "n_embd": 16,
    "n_layer": 2,
    "n_head": 4,
    "n_inner": 32,
    "n_vocab": 48,
    "n_positions": 8,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": False,
}
GPT2_TIED: dict[str, object] = dict(GPT2_UNTIED, tie_word_embeddings=True)

#: The 124M GPT-2 shape (tied head) — config only: no checkpoint, no weights, so
#: the packed input count can be pinned for free.
REAL_GPT2: dict[str, object] = {
    "model_type": "gpt2",
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,
    "n_vocab": 50257,
    "n_positions": 1024,
    "layer_norm_epsilon": 1e-5,
    "activation_function": "gelu_new",
    "tie_word_embeddings": True,
}

_FAMILIES = {"llama": llama_module, "gpt2": gpt2_module}

#: ``(label, raw config, expected packed input count)`` — the counts this host
#: measured with the ``(dtype, rank, shape[1:])`` bucketing rule; ``gpt2`` tied is
#: one bucket fewer because a tied config has no ``lm_head.weight`` at all.
_COUNT_CASES = [
    ("llama_untied", LLAMA_UNTIED, 5),
    ("llama_tied", LLAMA_TIED, 4),
    ("gpt2_untied", GPT2_UNTIED, 4),
    ("gpt2_tied", GPT2_TIED, 3),
]
_COUNT_CASES_BY_LABEL = {label: (raw, want) for label, raw, want in _COUNT_CASES}

#: ``(label, raw config)`` pairs used by the numeric-equivalence tests.
_PARITY_CASES = list(_COUNT_CASES)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _config(raw: dict[str, object]) -> ModelConfig:
    """Build a :class:`ModelConfig` from a raw HF-style config dict."""
    return ModelConfig.from_dict(raw)


def _specs(cfg: ModelConfig) -> dict[str, etl.TensorSpec]:
    """The canonical ``param_specs`` tree of ``cfg``'s own family."""
    return _FAMILIES[cfg.family].param_specs(cfg)


def _plan(cfg: ModelConfig) -> pack_module.PackPlan:
    """The packed-input plan of ``cfg``'s canonical specs."""
    return pack_module.pack_plan(_specs(cfg))


def _params(specs: dict[str, etl.TensorSpec], seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded synthetic canonical params (1-D scales are 1.0, matrices small)."""
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        shape = tuple(int(dim) for dim in spec.shape)
        if len(shape) == 1:
            params[name] = np.ones(shape, dtype=np.float32)
        else:
            params[name] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    return params


def _ids(cfg: ModelConfig, batch: int = _BATCH, seq: int = _SEQ, seed: int = 1) -> np.ndarray:
    """Seeded ``(batch, seq)`` int32 token ids inside the vocabulary."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)


def _host_unpack(
    plan: pack_module.PackPlan, packed: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Numpy mirror of :meth:`PackPlan.unpack` — the host side of the same slices.

    Deliberately written against :class:`PackEntry`'s ``start``/``shape`` (not by
    re-deriving offsets), so a wrong plan is detected by comparing this to the
    original canonical tree.
    """
    canonical: dict[str, np.ndarray] = {}
    for name, entry in plan.entries.items():
        block = packed[entry.input]
        if len(entry.shape) == 0:
            canonical[name] = block[entry.start[0]].reshape(())
        else:
            index = tuple(
                slice(start, start + length)
                for start, length in zip(entry.start, entry.shape)
            )
            canonical[name] = block[index]
    return canonical


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute difference, computed in float64."""
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _built(cfg: ModelConfig, pack_weights: bool, seq: int = _SEQ, batch: int = _BATCH):
    """Build a numpy-backend executable (a private cache keeps builds isolated)."""
    return build_model(
        cfg,
        seq,
        batch_size=batch,
        backend="numpy",
        cache=etl.CompileCache(),
        pack_weights=pack_weights,
    )


def _count_ops(graph: object) -> dict[str, int]:
    """Op-name histogram of a traced graph, walking nested regions."""
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


# ---------------------------------------------------------------------------
# 1. exact packed input counts (the whole point of the exercise)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,raw,want", _COUNT_CASES, ids=[c[0] for c in _COUNT_CASES])
def test_packed_input_counts(label: str, raw: dict, want: int) -> None:
    """Packing turns one input per weight into a handful of inputs."""
    cfg = _config(raw)
    specs = _specs(cfg)
    plan = _plan(cfg)
    assert len(plan.specs) == want, (
        f"{label}: expected {want} packed inputs, got {len(plan.specs)} "
        f"({sorted(plan.specs)}); canonical keys = {len(specs)}"
    )
    assert len(plan.specs) <= len(specs) // 2 and len(plan.specs) <= 8, (
        f"{label}: packing must collapse the tree well below the ~161-input IREE "
        f"cliff (keys={len(specs)}, inputs={len(plan.specs)})"
    )
    assert [n for n in plan.specs] == [f"packed.{i}" for i in range(want)], (
        f"{label}: packed inputs must be named packed.0..packed.{want - 1} in "
        f"bucket order, got {list(plan.specs)}"
    )
    assert plan.canonical_keys == tuple(sorted(specs)), (
        f"{label}: plan must cover exactly the canonical keys"
    )


def test_real_size_gpt2_plan_clears_the_iree_input_cliff() -> None:
    """The 124M GPT-2's plan fits far below IREE's ~161-input runtime cliff.

    Config-only (no checkpoint): 196 canonical weights → 3 packed inputs, so the
    one-graph forward stops being a SIGSEGV waiting to happen.
    """
    cfg = _config(REAL_GPT2)
    specs = _specs(cfg)
    plan = _plan(cfg)
    assert len(specs) == 196, f"unexpected canonical key count {len(specs)}"
    assert len(plan.specs) == 3, (
        f"the 124M gpt2 must pack to 3 inputs, got {len(plan.specs)}: "
        f"{ {name: tuple(spec.shape) for name, spec in plan.specs.items()} }"
    )
    # The executable declares the packed inputs + one ids leaf.
    assert len(specs) + 1 > IREE_CONSUMED_INPUT_CLIFF, (
        "this test is only meaningful while the UNPACKED graph is over the cliff "
        f"(it declares {len(specs) + 1} inputs)"
    )
    assert len(plan.specs) + 1 < IREE_CONSUMED_INPUT_CLIFF / 10


# ---------------------------------------------------------------------------
# 2. the plan is complete, contiguous and deterministic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,raw,want", _COUNT_CASES, ids=[c[0] for c in _COUNT_CASES]
)
def test_every_canonical_key_is_covered_exactly_once(
    label: str, raw: dict, want: int
) -> None:
    """Each key appears once, buckets are contiguous, rows add up, dtypes survive."""
    cfg = _config(raw)
    specs = _specs(cfg)
    plan = _plan(cfg)
    assert set(plan.entries) == set(specs), f"{label}: entries must match the specs"

    used: set[str] = set()
    for name, entry in plan.entries.items():
        assert isinstance(entry, pack_module.PackEntry)
        assert entry.shape == tuple(int(dim) for dim in specs[name].shape), (
            f"{label}/{name}: entry shape {entry.shape} != spec shape "
            f"{tuple(specs[name].shape)}"
        )
        assert len(entry.start) == max(len(entry.shape), 1), (
            f"{label}/{name}: start {entry.start} must have one entry per axis "
            f"(or one for a rank-0 scalar)"
        )
        assert entry.start[0] >= 0 and set(entry.start[1:]) <= {0}, (
            f"{label}/{name}: only axis 0 may be offset, got start {entry.start}"
        )
        assert entry.input in plan.specs, (
            f"{label}/{name}: entry points at unknown input {entry.input!r}"
        )
        used.add(entry.input)
    assert used == set(plan.specs), (
        f"{label}: every packed input must be used (unused={sorted(set(plan.specs) - used)})"
    )

    # Contiguous, gapless, in-bounds offsets within each bucket.
    for input_name, names in plan.names_by_input().items():
        spec = plan.specs[input_name]
        assert names, f"{label}: packed input {input_name} has no keys"
        assert spec.shape[0] is not None, f"{label}: packed dim must be static"
        offset = 0
        first = None
        for name in names:
            entry = plan.entries[name]
            assert entry.start[0] == offset, (
                f"{label}/{name}: start {entry.start[0]} != expected {offset} "
                f"(buckets must be prefix-sum positive and contiguous)"
            )
            rows = entry.shape[0] if entry.shape else 1
            offset += rows
            if first is None:
                first = entry.shape
            else:
                assert entry.shape[1:] == first[1:], (
                    f"{label}: {name} shares {input_name} but has trailing dims "
                    f"{entry.shape[1:]} != {first[1:]}"
                )
        assert offset == int(spec.shape[0]), (
            f"{label}: {input_name} packed dim {int(spec.shape[0])} != summed rows "
            f"{offset}"
        )
        assert tuple(spec.shape[1:]) == first[1:], (
            f"{label}: {input_name} trailing dims {tuple(spec.shape[1:])} != "
            f"{first[1:]}"
        )
        assert spec.device is None, f"{label}: packed specs are device-agnostic"
        assert spec.dtype == specs[names[0]].dtype, (
            f"{label}: {input_name} dtype {spec.dtype} != {specs[names[0]].dtype}"
        )


def test_bucket_count_is_shape_driven_not_family_driven() -> None:
    """Two families whose shapes collide pack into FEWER inputs (one rule, no special cases)."""
    # LLAMA_UNTIED has vocab_size == intermediate_size == 32 (as in the export
    # suite's config), so the lm_head/MLP buckets merge.
    collided = dict(LLAMA_UNTIED, vocab_size=32)
    plan = _plan(_config(collided))
    assert len(plan.specs) == 4, (
        f"a vocab==intermediate llama must merge two buckets (4 inputs), got "
        f"{len(plan.specs)}: {sorted(plan.specs)}"
    )
    assert len(plan.specs) == len(_plan(_config(LLAMA_TIED)).specs), (
        "with colliding shapes the tied/untied counts coincide — that is the "
        "evidence the rule is shape-driven only"
    )


@pytest.mark.parametrize("label,raw,want", _COUNT_CASES, ids=[c[0] for c in _COUNT_CASES])
def test_plan_is_deterministic(label: str, raw: dict, want: int) -> None:
    """Equal (but distinct) spec trees produce equal plans: same names, offsets, order."""
    cfg = _config(raw)
    first = _plan(cfg)
    # A fresh dict with reversed insertion order: bucket order must not depend on it.
    second = pack_module.pack_plan(dict(reversed(list(_specs(cfg).items()))))
    assert list(first.specs) == list(second.specs), (
        f"{label}: bucket order must be deterministic, got {list(first.specs)} vs "
        f"{list(second.specs)}"
    )
    assert {k: tuple(v.shape) for k, v in first.specs.items()} == {
        k: tuple(v.shape) for k, v in second.specs.items()
    }, f"{label}: packed shapes drifted between equal plans"
    assert first.entries == second.entries, f"{label}: entry offsets drifted"
    assert first.canonical_keys == second.canonical_keys
    assert first.input_names == tuple(f"packed.{i}" for i in range(want))


def test_plan_rejects_unpackable_inputs() -> None:
    """Symbolic dims and non-spec values are explicit errors, never a guess."""
    symbolic = {
        "a": etl.TensorSpec((etl.Dim("batch"), 4), etl.float32),
    }
    with pytest.raises(HFUnsupportedError) as excinfo:
        pack_module.pack_plan(symbolic)
    message = str(excinfo.value)
    assert "'a'" in message and "axis 0" in message and "static" in message, message

    dynamic = {"a": etl.TensorSpec((None, 4), etl.float32)}
    with pytest.raises(HFUnsupportedError):
        pack_module.pack_plan(dynamic)

    with pytest.raises(TypeError):
        pack_module.pack_plan([("a", etl.TensorSpec((2, 2), etl.float32))])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pack_module.pack_plan({"a": (2, 2)})  # not a TensorSpec
    with pytest.raises(TypeError):
        pack_module.pack_plan({1: etl.TensorSpec((2, 2), etl.float32)})
    with pytest.raises(TypeError):
        pack_module.pack_plan({"a": etl.TensorSpec((2, 2), etl.float32)}, prefix="")


# ---------------------------------------------------------------------------
# 3. host pack/unpack round trip is bit-identical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,raw,want", _COUNT_CASES, ids=[c[0] for c in _COUNT_CASES])
def test_host_round_trip_is_bit_identical(label: str, raw: dict, want: int) -> None:
    """Pack → slice back reproduces every canonical array exactly (no dtype change)."""
    cfg = _config(raw)
    specs = _specs(cfg)
    plan = _plan(cfg)
    params = _params(specs)
    packed = plan.pack(params)
    assert set(packed) == set(plan.specs), f"{label}: packed keys must be the plan's"
    for name, array in packed.items():
        assert isinstance(array, np.ndarray), f"{label}/{name} must be a host ndarray"
        assert array.flags.c_contiguous, f"{label}/{name} must be C-contiguous"
        assert array.dtype == np.dtype(etl.float32), f"{label}/{name} dtype drifted"
        want_shape = tuple(int(dim) for dim in plan.specs[name].shape)
        assert array.shape == want_shape, (
            f"{label}/{name}: packed shape {array.shape} != plan {want_shape}"
        )
    restored = _host_unpack(plan, packed)
    assert set(restored) == set(params)
    for name, array in params.items():
        got = restored[name]
        assert got.shape == array.shape, f"{label}/{name}: shape drifted"
        if not np.array_equal(got, array):
            raise AssertionError(
                f"{label}/{name}: packed round trip is not bit-identical "
                f"(max-abs {_max_abs(got, array):g}, entry "
                f"{plan.entries[name]})"
            )
    # The cached-class API must agree with the module-level shorthand.
    assert all(
        np.array_equal(plan_module_value, packed[name])
        for name, plan_module_value in pack_module.pack_params(
            plan, params
        ).items()
    ), f"{label}: etl_hf.pack_params must equal PackPlan.pack"


@pytest.mark.parametrize(
    "label,raw,want", _COUNT_CASES[:2], ids=[c[0] for c in _COUNT_CASES[:2]]
)
def test_pack_validates_every_key_shape_and_dtype(
    label: str, raw: dict, want: int
) -> None:
    """Missing/extra keys, wrong shape, wrong dtype and non-arrays name the key."""
    cfg = _config(raw)
    specs = _specs(cfg)
    plan = _plan(cfg)
    params = _params(specs)
    first = plan.canonical_keys[0]

    missing = {k: v for k, v in params.items() if k != first}
    with pytest.raises(HFWeightError) as excinfo:
        plan.pack(missing)
    assert first in str(excinfo.value) and "missing" in str(excinfo.value)

    extra = dict(params, **{"not.a.weight": np.zeros((2, 2), np.float32)})
    with pytest.raises(HFWeightError) as excinfo:
        plan.pack(extra)
    assert "not.a.weight" in str(excinfo.value) and "extra" in str(excinfo.value)

    wrong_shape = dict(params, **{first: np.zeros((1, 1), np.float32)})
    with pytest.raises(HFWeightError) as excinfo:
        plan.pack(wrong_shape)
    message = str(excinfo.value)
    assert first in message and "shape" in message, message

    wrong_dtype = dict(params, **{first: np.asarray(params[first], np.float64)})
    with pytest.raises(HFWeightError) as excinfo:
        plan.pack(wrong_dtype)
    message = str(excinfo.value)
    assert first in message and "dtype" in message and "float64" in message, message

    with pytest.raises(HFWeightError) as excinfo:
        plan.pack(dict(params, **{first: object()}))
    assert first in str(excinfo.value)

    with pytest.raises(TypeError):
        plan.pack([first])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pack_module.pack_params(params, params)  # type: ignore[arg-type]


def test_unpack_validates_the_packed_tree() -> None:
    """Wrong keys/shapes/types on the in-graph side are explicit, never silent."""
    cfg = _config(LLAMA_UNTIED)
    plan = _plan(cfg)
    with pytest.raises(TypeError):
        plan.unpack([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(HFWeightError) as excinfo:
        plan.unpack({"packed.0": np.zeros((4, 4), np.float32)})
    message = str(excinfo.value)
    assert "packed.1" in message and "missing" in message, message
    # Host arrays are not in-graph values: the unpack needs a trace.
    with pytest.raises(TypeError) as excinfo:
        plan.unpack({name: np.zeros((1,), np.float32) for name in plan.specs})
    assert "SymbolicTensor" in str(excinfo.value)
    # A packed tree whose shapes disagree with the plan is a ShapeError at trace time.
    wrong = {name: etl.TensorSpec((4, 4), etl.float32) for name in plan.specs}
    with pytest.raises(etl.ShapeError):
        etl.trace(lambda p: plan.unpack(p)["embed.weight"], wrong)

    # A plan built for ANOTHER config rejects this config's packed shapes.
    other = pack_module.pack_plan(_specs(_config(dict(LLAMA_UNTIED, hidden_size=8))))
    with pytest.raises(etl.ShapeError):
        etl.trace(
            lambda p, x: other.unpack(p)["embed.weight"],
            plan.specs,
            etl.TensorSpec((_BATCH, _SEQ), etl.int32),
        )


# ---------------------------------------------------------------------------
# 4. numeric equivalence: packed vs unpacked executables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,raw,want", _PARITY_CASES, ids=[c[0] for c in _PARITY_CASES])
def test_numpy_backend_packed_matches_unpacked_bit_identically(
    label: str, raw: dict, want: int
) -> None:
    """Concatenation + static slice is exact: identical logits on the interpreter."""
    cfg = _config(raw)
    params = _params(_specs(cfg))
    packed = _plan(cfg).pack(params)
    ids = _ids(cfg)
    unpacked_logits = run_model(_built(cfg, pack_weights=False), params, ids).numpy()
    packed_logits = run_model(_built(cfg, pack_weights=True), packed, ids).numpy()
    assert packed_logits.shape == unpacked_logits.shape
    assert np.array_equal(packed_logits, unpacked_logits), (
        f"{label}: packed logits must be bit-identical to unpacked on the numpy "
        f"backend (max-abs {_max_abs(packed_logits, unpacked_logits):g})"
    )
    reference = _FAMILIES[cfg.family].reference(cfg, params, ids)
    assert _max_abs(packed_logits, reference) <= 1e-4, (
        f"{label}: packed forward drifted from the pure-numpy oracle "
        f"({_max_abs(packed_logits, reference):g})"
    )


def test_packed_matches_unpacked_across_shapes() -> None:
    """Packing is shape-agnostic: (seq, batch) combinations agree bit-for-bit."""
    cfg = _config(GPT2_TIED)
    params = _params(_specs(cfg))
    packed = _plan(cfg).pack(params)
    for seq, batch in ((4, 1), (8, 2)):
        ids = _ids(cfg, batch=batch, seq=seq)
        unpacked = run_model(
            _built(cfg, pack_weights=False, seq=seq, batch=batch), params, ids
        ).numpy()
        packed_logits = run_model(
            _built(cfg, pack_weights=True, seq=seq, batch=batch), packed, ids
        ).numpy()
        assert packed_logits.shape == (batch, seq, cfg.vocab_size)
        assert np.array_equal(packed_logits, unpacked), (
            f"seq={seq} batch={batch}: packed vs unpacked logits differ "
            f"(max-abs {_max_abs(packed_logits, unpacked):g})"
        )


def test_unpacked_is_the_default() -> None:
    """``build_model`` without the flag keeps one graph input per canonical weight."""
    cfg = _config(LLAMA_UNTIED)
    specs = _specs(cfg)
    params = _params(specs)
    ids = _ids(cfg)
    default = build_model(cfg, _SEQ, backend="numpy", cache=etl.CompileCache())
    explicit = _built(cfg, pack_weights=False)
    assert len(default.signature.input_specs) == len(specs) + 1, (
        f"the default must declare {len(specs)} weight inputs + ids, got "
        f"{len(default.signature.input_specs)}"
    )
    assert np.array_equal(
        run_model(default, params, ids).numpy(), run_model(explicit, params, ids).numpy()
    ), "pack_weights=False must be byte-identical to the default build"
    # And a packed build really is a different graph (its own cache entry).
    packed_exe = _built(cfg, pack_weights=True)
    assert packed_exe is not default
    assert len(packed_exe.signature.input_specs) == len(_plan(cfg).specs) + 1


def test_run_model_reports_the_packing_mismatch() -> None:
    """Mixing the two tree shapes names ``pack_weights`` and the remedy."""
    cfg = _config(GPT2_TIED)
    params = _params(_specs(cfg))
    packed = _plan(cfg).pack(params)
    ids = _ids(cfg)
    packed_exe = _built(cfg, pack_weights=True)
    unpacked_exe = _built(cfg, pack_weights=False)

    with pytest.raises(HFWeightError) as excinfo:
        run_model(packed_exe, params, ids)
    message = str(excinfo.value)
    assert "pack_weights=True" in message and "packed.0" in message, message
    assert "pack_params" in message and "pack_weights=False" in message, message

    with pytest.raises(HFWeightError) as excinfo:
        run_model(unpacked_exe, packed, ids)
    message = str(excinfo.value)
    assert "pack_weights=False" in message and "packed.0" in message, message
    assert "pack_weights=True" in message, message

    # The legitimate combinations still run.
    assert run_model(packed_exe, packed, ids).shape == (1, _SEQ, cfg.vocab_size)
    assert run_model(unpacked_exe, params, ids).shape == (1, _SEQ, cfg.vocab_size)


# ---------------------------------------------------------------------------
# 5. nothing is baked: the packed graph is still all graph inputs
# ---------------------------------------------------------------------------
def test_packed_graph_bakes_no_weights() -> None:
    """Packed inputs, not ``etl.constant``: the export keeps the literal ceiling.

    Mirrors ``test_export.test_no_weights_baked_into_ir``: the input count is the
    plan's, every canonical key is recovered by a static ``slice``, and no weight
    payload appears as a dense literal.
    """
    cfg = _config(LLAMA_UNTIED)
    specs = _specs(cfg)
    plan = _plan(cfg)
    graph = etl.trace(
        llama_module.graph(cfg, _SEQ, _BATCH, pack_plan=plan),
        plan.specs,
        etl.TensorSpec((_BATCH, _SEQ), etl.int32),
    )
    graph.verify()
    assert len(graph.tensor_specs) == len(plan.specs) + 1, (
        f"packed graph must take {len(plan.specs)} packed inputs + ids, got "
        f"{len(graph.tensor_specs)}"
    )
    counts = _count_ops(graph)
    # The unpack adds exactly one static slice per canonical key on top of the
    # family body's own slices (llama's RoPE rotate_half slices).
    body = etl.trace(
        llama_module.graph(cfg, _SEQ, _BATCH),
        specs,
        etl.TensorSpec((_BATCH, _SEQ), etl.int32),
    )
    baseline = _count_ops(body).get("slice", 0)
    assert counts.get("slice", 0) == baseline + len(specs), (
        f"every canonical key must be recovered by exactly one static slice, got "
        f"{counts.get('slice', 0)} slices for {len(specs)} keys (+{baseline} in the "
        f"family body); top: "
        f"{sorted(counts.items(), key=lambda kv: -kv[1])[:8]}"
    )
    assert counts.get("external_call", 0) == 0
    assert counts.get("runtime_call", 0) == 0

    text = stablehlo.export(graph)
    constants = text.count("stablehlo.constant")
    assert constants < 40, (
        f"too many stablehlo.constant ops ({constants}) — packed weights must stay "
        f"graph inputs"
    )
    import re

    literals = re.findall(r"dense<\[([^\]]*)\]>", text)
    biggest = max((len(lit.split(",")) for lit in literals), default=0)
    assert biggest <= 4 * _SEQ, (
        f"largest dense literal has {biggest} elements — only O(seq_len) structural "
        f"constants are allowed (no weight payloads)"
    )
    assert "stablehlo.slice" in text, (
        "the packed graph must export the static slices (stablehlo.slice)"
    )


# ---------------------------------------------------------------------------
# 6. HFModel: pack_params + build/forward with pack_weights
# ---------------------------------------------------------------------------
def _model(raw: dict, seq: int = _SEQ, batch: int = _BATCH) -> HFModel:
    """An in-memory :class:`HFModel` with synthetic canonical weights."""
    cfg = _config(raw)
    return HFModel(cfg, _params(_specs(cfg)), seq_len=seq, batch_size=batch)


def test_hfmodel_pack_weights_end_to_end() -> None:
    """``HFModel`` packs once and forwards through the packed executable."""
    model = _model(GPT2_TIED)
    canonical_specs = model.param_specs()
    packed_specs = model.param_specs(pack_weights=True)
    plan = model.pack_plan()
    assert set(packed_specs) == set(plan.specs) and len(packed_specs) < len(canonical_specs)
    assert model.pack_plan() is plan, "the plan must be memoised per model"

    packed = model.pack_params()
    assert set(packed) == set(plan.specs)
    for name, array in packed.items():
        assert np.array_equal(array, plan.pack(model.params())[name])

    ids = _ids(model.config)
    logits = model.forward(input_ids=ids, pack_weights=True)
    reference = model.reference(input_ids=ids)
    assert logits.shape == (model.batch_size, model.seq_len, model.config.vocab_size)
    assert _max_abs(logits, reference) <= 1e-4, (
        f"packed HFModel forward drifted from the oracle "
        f"({_max_abs(logits, reference):g})"
    )
    # Passing the already-packed tree skips the host re-pack and agrees exactly.
    assert np.array_equal(
        model.forward(params=packed, input_ids=ids, pack_weights=True), logits
    ), "a pre-packed tree must produce the same logits as the implicit pack"
    # ... and the unpacked default still works and agrees bit-for-bit on numpy.
    assert np.array_equal(model.forward(input_ids=ids), logits), (
        "packed and unpacked HFModel forwards must be bit-identical on the numpy "
        "interpreter"
    )


def test_hfmodel_pack_params_error_paths() -> None:
    """``pack_params`` names the offending key for every kind of mismatch."""
    model = _model(LLAMA_TIED)
    params = model.params()
    first = min(params)

    with pytest.raises(HFWeightError) as excinfo:
        model.pack_params({k: v for k, v in params.items() if k != first})
    assert first in str(excinfo.value), str(excinfo.value)

    with pytest.raises(HFWeightError) as excinfo:
        model.pack_params(dict(params, **{"layer.0.bogus": np.zeros(3, np.float32)}))
    assert "layer.0.bogus" in str(excinfo.value)

    with pytest.raises(HFWeightError) as excinfo:
        model.pack_params(dict(params, **{first: np.zeros(3, np.float32)}))
    assert first in str(excinfo.value) and "shape" in str(excinfo.value)

    with pytest.raises(HFWeightError) as excinfo:
        model.pack_params(dict(params, **{first: np.zeros_like(params[first], np.float16)}))
    assert first in str(excinfo.value) and "dtype" in str(excinfo.value)

    with pytest.raises(TypeError):
        model.pack_params([1, 2])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.param_specs(pack_weights=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.forward(pack_weights="yes")  # type: ignore[arg-type]

    # A device tensor is host-only data for packing: explicit, actionable error.
    placed = place_params({first: params[first]}, etl.core.Device("cpu", 0))[first]
    with pytest.raises(HFWeightError) as excinfo:
        model.pack_params(dict(params, **{first: placed}))
    message = str(excinfo.value)
    assert first in message and "host" in message, message


def test_packed_params_place_into_fewer_leaves() -> None:
    """``pack → place`` is the supported order and transfers far fewer leaves."""
    model = _model(GPT2_UNTIED, seq=4)
    device = etl.core.Device("cpu", 0)
    packed = model.pack_params()
    placed_packed = place_params(packed, device)
    placed_canonical = place_params(model.params(), device)
    assert set(placed_packed) == set(packed)
    assert len(placed_packed) < len(placed_canonical), (
        "packed placement must stage fewer leaves than the canonical tree"
    )
    ids = _ids(model.config, seq=4)
    assert np.array_equal(
        run_model(model.build(pack_weights=True), placed_packed, ids).numpy(),
        run_model(model.build(pack_weights=False), placed_canonical, ids).numpy(),
    ), "placed packed params must run the same graph to the same logits"


# ---------------------------------------------------------------------------
# 7. compiled pin: iree/llvm-cpu runs the packed graph
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,raw,want", _PARITY_CASES, ids=[c[0] for c in _PARITY_CASES])
@requires_iree
def test_iree_llvm_cpu_runs_the_packed_graph(label: str, raw: dict, want: int) -> None:
    """Packed graph compiles + runs on iree/llvm-cpu with fp32 parity vs numpy."""
    cfg = _config(raw)
    params = _params(_specs(cfg))
    packed = _plan(cfg).pack(params)
    ids = _ids(cfg)
    numpy_logits = run_model(_built(cfg, pack_weights=True), packed, ids).numpy()
    iree_exe = build_model(
        cfg,
        _SEQ,
        backend="iree",
        cache=etl.CompileCache(),
        pack_weights=True,
    )
    assert len(iree_exe.signature.input_specs) == len(_plan(cfg).specs) + 1, (
        f"{label}: the compiled executable must declare the packed inputs only, got "
        f"{len(iree_exe.signature.input_specs)}"
    )
    iree_logits = run_model(iree_exe, packed, ids).numpy()
    assert iree_logits.shape == numpy_logits.shape
    assert np.isfinite(iree_logits).all(), f"{label}: iree logits must be finite"
    delta = _max_abs(iree_logits, numpy_logits)
    assert delta <= IREE_ATOL, (
        f"{label}: iree packed vs numpy packed max-abs = {delta:g} exceeds "
        f"{IREE_ATOL:g}"
    )


# ---------------------------------------------------------------------------
# 8. the real 124M GPT-2: packed build + stored HF golden
# ---------------------------------------------------------------------------
_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "bench_logs" / "hf_goldens"


def _golden(slug: str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``<repo>/bench_logs/hf_goldens/<slug>.npz`` (``input_ids``, ``logits``)."""
    path = _GOLDEN_DIR / f"{slug}.npz"
    if not path.is_file():
        pytest.skip(f"no golden fixture at {path} (read-only bench_logs/ area)")
    with np.load(path, allow_pickle=False) as archive:
        if "input_ids" not in archive.files or "logits" not in archive.files:
            pytest.skip(f"{path} carries no input_ids/logits arrays")
        return np.asarray(archive["input_ids"]), np.asarray(archive["logits"])


@_BIG
def test_real_size_gpt2_packed_matches_golden() -> None:
    """The case packing exists for: 124M GPT-2, 197 inputs → 4, logits unchanged.

    ``gpt2`` at ``seq_len=32`` is the checkpoint IREE's Python runtime cannot run
    unpacked (196 weights + ids = 197 consumed inputs > the ~161 cliff). Here it
    builds a packed executable with 3 weight inputs and its **numpy-backend**
    forward must reproduce the stored HuggingFace golden — i.e. packing changed
    the program's signature and nothing else.
    """
    ids, want = _golden("gpt2")
    if ids.shape != (1, 32):
        pytest.skip(f"golden gpt2 ids shape {ids.shape} is not (1, 32)")
    directory = hf_download.require_model("gpt2")
    model = load_hf(directory, seq_len=32)
    params = model.params()
    assert len(params) == 196, f"expected 196 canonical keys, got {len(params)}"

    plan = model.pack_plan()
    assert len(plan.specs) == 3, (
        f"the 124M gpt2 must pack to 3 inputs, got {len(plan.specs)}: "
        f"{ {name: tuple(spec.shape) for name, spec in plan.specs.items()} }"
    )
    packed = model.pack_params(params)
    assert set(packed) == set(plan.specs)

    executable = model.build(backend="numpy", pack_weights=True)
    assert len(executable.signature.input_specs) == len(plan.specs) + 1, (
        f"the packed executable must declare {len(plan.specs)} weight inputs + ids, "
        f"got {len(executable.signature.input_specs)} (the unpacked graph declares "
        f"{len(params) + 1}, over the IREE runtime cliff of "
        f"{IREE_CONSUMED_INPUT_CLIFF})"
    )
    got = run_model(executable, packed, ids).numpy()
    assert got.shape == want.shape == (1, 32, 50257)
    delta = _max_abs(got, want)
    scale = max(1e-45, float(np.max(np.abs(want.astype(np.float64)))))
    assert delta <= GOLDEN_ATOL and delta / scale <= GOLDEN_RTOL, (
        f"packed gpt2 logits drifted from the HF golden: max-abs {delta:g} "
        f"(relative {delta / scale:g}) — measured ≈2.1e-4 vs an |logits| max of "
        f"≈{scale:g}"
    )
    # Packing is transparent: the unpacked numpy forward is bit-identical.
    unpacked = run_model(model.build(backend="numpy"), params, ids).numpy()
    assert np.array_equal(got, unpacked), (
        f"packed vs unpacked numpy logits differ (max-abs {_max_abs(got, unpacked):g})"
    )
