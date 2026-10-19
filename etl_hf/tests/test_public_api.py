"""Tests for the public surface of the ``etl_hf`` package (``etl_hf/__init__.py``).

The package-level contract is what users see first, so it is pinned here. Since
the functional refactor the surface is deliberately **not** object oriented —
there is a loading half (:mod:`etl_hf.load`), a compute-function half
(:mod:`etl_hf.program`) and a staging half (:mod:`etl_hf.pipeline`), joined only
by ordinary values (a :class:`ModelConfig`, a ``dict`` of host params, an
``@etl.defn`` forward, an ``Executable``). Concretely:

* ``import etl_hf`` works and pulls in **no** optional dependency (torch /
  transformers / safetensors / iree) — checked in a subprocess, so the importing
  interpreter's own state cannot hide a leak;
* ``__all__`` is complete, duplicate-free, and every entry is reachable from the
  documented defining module — a star-import yields exactly the documented names;
* every exported function/class carries a docstring, and every parameter of every
  exported function is named in it (the package is API documentation as much as
  code);
* there is no OOP surface: ``HFModel`` / ``build_model`` / ``run_model`` are gone
  from the namespace *and* from disk (``etl_hf.build`` is a **function** of
  :mod:`etl_hf.pipeline`, not a module);
* :func:`etl_hf.load_hf` is a pure data record — a
  :class:`etl_hf.load.LoadedModel` tuple with no methods, no ``seq_len`` and no
  compilation;
* loading splits cleanly: ``load_config`` + ``load_params`` reproduce
  ``load_hf`` exactly (same digest, same keys, array-for-array identical values);
* :func:`etl_hf.forward_fn` is the memoised ``@etl.defn`` compute function — it
  captures **no** parameters, so it traces from ``param_specs`` alone and its
  graph has exactly ``len(param_specs) + 1`` inputs (the weights are graph
  inputs, never ``etl.constant`` payloads);
* the staging stages are distinct public functions (``build`` / ``run`` /
  ``predict`` / ``reference_logits``), ``build`` is memoised to one
  ``Executable`` per specialisation, and ``reference_logits`` agrees with a
  numpy-backend ``predict``;
* the submodules (``config``, ``families``, ``load``, ``ops``, ``pack``,
  ``pipeline``, ``program``, ``reference``, ``weights``) are reachable as
  attributes;
* the error hierarchy derives from :class:`etl.core.ETLError` as documented.

Everything here runs offline and without an optional dependency: the checkpoint
the data-record pins use is written by hand (``numpy`` + ``struct`` + ``json``)
into a temporary directory.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import etl
import etl_hf
from etl_hf import errors as errors_module

#: Every documented public name, and the ``(module, attribute)`` it re-exports.
_EXPECTED_EXPORTS = {
    # -- loading (checkpoint -> static config + host params) -----------------
    "LoadedModel": ("etl_hf.load", "LoadedModel"),
    "ModelConfig": ("etl_hf.config", "ModelConfig"),
    "load_config": ("etl_hf.load", "load_config"),
    "load_hf": ("etl_hf.load", "load_hf"),
    "load_params": ("etl_hf.load", "load_params"),
    "validate_params": ("etl_hf.load", "validate_params"),
    # -- the compute function ------------------------------------------------
    "forward_fn": ("etl_hf.program", "forward_fn"),
    "param_specs": ("etl_hf.program", "param_specs"),
    "reference_logits": ("etl_hf.program", "reference_logits"),
    # -- staging (build / run / predict / explicit placement) ----------------
    "build": ("etl_hf.pipeline", "build"),
    "place_ids": ("etl_hf.pipeline", "place_ids"),
    "place_params": ("etl_hf.pipeline", "place_params"),
    "predict": ("etl_hf.pipeline", "predict"),
    "run": ("etl_hf.pipeline", "run"),
    # -- packed weights (opt-in) --------------------------------------------
    "PackPlan": ("etl_hf.pack", "PackPlan"),
    "pack_plan": ("etl_hf.pack", "pack_plan"),
    "pack_params": ("etl_hf.pack", "pack_params"),
    # -- family registry ----------------------------------------------------
    "FAMILIES": ("etl_hf.families", "FAMILIES"),
    "FamilySpec": ("etl_hf.families", "FamilySpec"),
    "family_for_model_type": ("etl_hf.families", "family_for_model_type"),
    "get_family": ("etl_hf.families", "get_family"),
    # -- errors -------------------------------------------------------------
    "HFError": ("etl_hf.errors", "HFError"),
    "HFConfigError": ("etl_hf.errors", "HFConfigError"),
    "HFWeightError": ("etl_hf.errors", "HFWeightError"),
    "HFUnsupportedError": ("etl_hf.errors", "HFUnsupportedError"),
}

#: Submodules the package exposes as attributes (``etl_hf.<name>``). ``build`` is
#: deliberately absent: it is a **function** re-exported from
#: :mod:`etl_hf.pipeline`, not a module (see ``test_build_is_a_function``).
_SUBMODULES = (
    "config",
    "families",
    "load",
    "ops",
    "pack",
    "pipeline",
    "program",
    "reference",
    "weights",
)

#: The OOP surface that the functional refactor deleted — no shim, no alias.
_REMOVED_OOP_NAMES = ("HFModel", "build_model", "run_model")

#: Every exported function whose parameters must all be named in its docstring.
_PUBLIC_FUNCTIONS = (
    "build",
    "family_for_model_type",
    "forward_fn",
    "get_family",
    "load_config",
    "load_hf",
    "load_params",
    "pack_params",
    "pack_plan",
    "param_specs",
    "place_ids",
    "place_params",
    "predict",
    "reference_logits",
    "run",
    "validate_params",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Clear the env knobs :mod:`etl_hf.pipeline` / ``etl`` consult.

    ``ETL_BACKEND`` / ``ETL_DEVICE`` / ``ETL_OPT_LEVEL`` /
    ``ETL_TARGET_BACKENDS`` are read at build time (and inherited by the
    subprocess checks), so a developer shell that exports them must not change
    which backend this file's pins actually exercise.
    """
    for variable in (
        "ETL_BACKEND",
        "ETL_DEVICE",
        "ETL_OPT_LEVEL",
        "ETL_TARGET_BACKENDS",
    ):
        monkeypatch.delenv(variable, raising=False)


# ---------------------------------------------------------------------------
# basic importability
# ---------------------------------------------------------------------------
def test_import_etl_hf_works_and_has_a_version():
    assert isinstance(etl_hf.__version__, str)
    assert etl_hf.__version__ == "0.1.0"
    assert etl_hf.__doc__, "the package needs a docstring (it is the entry point)"


def test_package_docstring_states_the_explicit_pipeline_philosophy():
    doc = etl_hf.__doc__ or ""
    assert "HuggingFace" in doc
    for phrase in ("explicit", "trace", "compile", "device", "weights"):
        assert phrase in doc, phrase


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
def test_all_is_complete_and_duplicate_free():
    assert isinstance(etl_hf.__all__, list)
    assert len(etl_hf.__all__) == len(set(etl_hf.__all__)), "duplicate names in __all__"
    expected = set(_EXPECTED_EXPORTS) | set(_SUBMODULES) | {"__version__"}
    assert set(etl_hf.__all__) == expected


@pytest.mark.parametrize("name", sorted(_EXPECTED_EXPORTS))
def test_every_public_name_is_reachable(name):
    assert hasattr(etl_hf, name)
    module_path, attribute = _EXPECTED_EXPORTS[name]
    module = importlib.import_module(module_path)
    assert getattr(etl_hf, name) is getattr(module, attribute), (
        f"etl_hf.{name} is not {module_path}.{attribute} — the re-export drifted"
    )


def test_every_public_name_has_a_docstring():
    """Functions and classes carry real docstrings (arguments / returns / errors)."""
    for name in etl_hf.__all__:
        value = getattr(etl_hf, name)
        if isinstance(value, types.ModuleType) or name == "__version__":
            continue
        if inspect.isclass(value) or inspect.isfunction(value):
            doc = inspect.getdoc(value)
            assert doc, f"{name} has no docstring"
            assert len(doc) > 60, f"{name}'s docstring is too thin: {doc!r}"


@pytest.mark.parametrize("name", _PUBLIC_FUNCTIONS)
def test_public_functions_document_their_arguments(name):
    """Every parameter of a public function is mentioned in its docstring."""
    function = getattr(etl_hf, name)
    assert inspect.isfunction(function), f"{name} is not a function"
    doc = inspect.getdoc(function) or ""
    for parameter in inspect.signature(function).parameters:
        if parameter.startswith("_"):
            continue
        assert parameter in doc, f"{name} does not document {parameter!r}"


def test_star_import_yields_exactly_all():
    namespace: dict = {}
    exec("from etl_hf import *", namespace)  # noqa: S102 - deliberate star-import test
    imported = {key for key in namespace if not key.startswith("__")}
    assert imported == {name for name in etl_hf.__all__ if not name.startswith("__")}


# ---------------------------------------------------------------------------
# no OOP surface: the object-oriented API is gone, not shimmed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _REMOVED_OOP_NAMES)
def test_removed_oop_names_are_gone(name):
    """``HFModel`` / ``build_model`` / ``run_model`` exist nowhere in the package."""
    assert not hasattr(etl_hf, name), f"etl_hf.{name} still exists (no shim by design)"
    assert name not in etl_hf.__all__
    for module_name in _SUBMODULES:
        module = importlib.import_module(f"etl_hf.{module_name}")
        assert not hasattr(module, name), f"etl_hf.{module_name}.{name} still exists"


def test_build_is_a_function():
    """``etl_hf.build`` is the staging function of :mod:`etl_hf.pipeline`."""
    assert inspect.isfunction(etl_hf.build)
    assert not isinstance(etl_hf.build, types.ModuleType)
    assert etl_hf.build.__module__ == "etl_hf.pipeline"
    assert etl_hf.build is importlib.import_module("etl_hf.pipeline").build


def test_the_build_module_is_deleted():
    """The old ``etl_hf.build`` module is gone — not even importable."""
    assert importlib.util.find_spec("etl_hf.build") is None
    with pytest.raises(ImportError):
        importlib.import_module("etl_hf.build")
    assert "etl_hf.build" not in sys.modules
    assert not (Path(etl_hf.__file__).parent / "build.py").exists()


# ---------------------------------------------------------------------------
# submodules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodules_are_reachable_attributes(name):
    value = getattr(etl_hf, name)
    assert isinstance(value, types.ModuleType)
    assert value.__name__ == f"etl_hf.{name}"


@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodules_are_importable_by_path(name):
    assert getattr(etl_hf, name) is importlib.import_module(f"etl_hf.{name}")


def test_error_classes_form_a_documented_hierarchy():
    assert issubclass(etl_hf.HFError, etl.ETLError)
    for subclass in (
        etl_hf.HFConfigError,
        etl_hf.HFWeightError,
        etl_hf.HFUnsupportedError,
    ):
        assert issubclass(subclass, etl_hf.HFError)
        assert issubclass(subclass, etl.ETLError)
    assert set(errors_module.__all__) == {
        "HFError",
        "HFConfigError",
        "HFWeightError",
        "HFUnsupportedError",
    }


def test_registry_is_reachable_from_the_package():
    assert set(etl_hf.FAMILIES) == {"llama", "gpt2"}
    assert etl_hf.get_family("llama") is etl_hf.FAMILIES["llama"]
    assert etl_hf.family_for_model_type("mistral") is etl_hf.FAMILIES["llama"]


# ---------------------------------------------------------------------------
# fixture: a tiny, hand-written HF checkpoint (numpy + struct + json only)
# ---------------------------------------------------------------------------
#: Static sequence length used by the graph pins (short: the fixture is cheap).
SEQ_LEN = 4

#: Weight scale small enough that RMSNorm/softmax stay well-conditioned in fp32.
_SCALE = 0.1

#: A tiny untied Llama checkpoint (GQA: 4 query heads, 2 KV heads, 1 layer).
_CHECKPOINT_JSON: dict = {
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "intermediate_size": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 16,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def _safetensors_bytes(header: dict, payload: bytes) -> bytes:
    """Assemble a safetensors blob: 8-byte LE header length + JSON header + data."""
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)  # the real writer pads the header to 8 bytes
    return struct.pack("<Q", len(blob)) + blob + payload


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write ``tensors`` as a float32 safetensors file (no ``safetensors`` dep)."""
    header: dict = {}
    payload = bytearray()
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        begin = len(payload)
        payload += data.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(data.shape),
            "data_offsets": [begin, len(payload)],
        }
    path.write_bytes(_safetensors_bytes(header, bytes(payload)))
    return path


def _llama_state_dict(raw: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """An HF ``LlamaForCausalLM``-shaped state dict (linears stored ``(out, in)``)."""
    hidden = raw["hidden_size"]
    heads = raw["num_attention_heads"]
    kv_heads = raw["num_key_value_heads"]
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim
    intermediate, vocab = raw["intermediate_size"], raw["vocab_size"]

    def weigh(*shape: int) -> np.ndarray:
        return np.asarray(rng.standard_normal(shape) * _SCALE, dtype=np.float32)

    def scale(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(1.0 + 0.05 * rng.standard_normal(shape), dtype=np.float32)

    state: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weigh(vocab, hidden),
        "model.norm.weight": scale((hidden,)),
    }
    for index in range(raw["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        state[prefix + "input_layernorm.weight"] = scale((hidden,))
        state[prefix + "self_attn.q_proj.weight"] = weigh(heads * head_dim, hidden)
        state[prefix + "self_attn.k_proj.weight"] = weigh(kv_dim, hidden)
        state[prefix + "self_attn.v_proj.weight"] = weigh(kv_dim, hidden)
        state[prefix + "self_attn.o_proj.weight"] = weigh(hidden, heads * head_dim)
        state[prefix + "post_attention_layernorm.weight"] = scale((hidden,))
        state[prefix + "mlp.gate_proj.weight"] = weigh(intermediate, hidden)
        state[prefix + "mlp.up_proj.weight"] = weigh(intermediate, hidden)
        state[prefix + "mlp.down_proj.weight"] = weigh(hidden, intermediate)
    if not raw["tie_word_embeddings"]:
        state["lm_head.weight"] = weigh(vocab, hidden)
    return state


def _write_checkpoint_dir(directory: Path, raw: dict, *, seed: int = 7) -> Path:
    """Write a minimal-but-real HF checkpoint directory (config + safetensors)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    _write_safetensors(
        directory / "model.safetensors", _llama_state_dict(raw, np.random.default_rng(seed))
    )
    return directory


@pytest.fixture(scope="module")
def checkpoint_dir(tmp_path_factory) -> Path:
    """A tiny synthetic checkpoint directory (offline, no optional dependency)."""
    base = tmp_path_factory.mktemp("etl_hf_public_api")
    return _write_checkpoint_dir(base / "tiny_llama", _CHECKPOINT_JSON)


@pytest.fixture(scope="module")
def loaded(checkpoint_dir) -> object:
    """The loaded checkpoint — computed once for every data-record pin."""
    return etl_hf.load_hf(checkpoint_dir)


@pytest.fixture(scope="module")
def ids() -> np.ndarray:
    """A seeded ``(1, SEQ_LEN)`` int32 id batch inside the vocabulary."""
    rng = np.random.default_rng(3)
    return rng.integers(0, _CHECKPOINT_JSON["vocab_size"], size=(1, SEQ_LEN)).astype(
        np.int32
    )


# ---------------------------------------------------------------------------
# load: ``load_hf`` answers plain data, and the two halves compose to it
# ---------------------------------------------------------------------------
def test_load_hf_returns_a_plain_data_record(loaded, checkpoint_dir):
    """``LoadedModel`` is a 4-field tuple of data — no methods, no ``seq_len``."""
    assert isinstance(loaded, tuple), "LoadedModel must be a plain tuple of data"
    assert type(loaded) is etl_hf.load.LoadedModel
    assert etl_hf.LoadedModel is etl_hf.load.LoadedModel
    assert etl_hf.LoadedModel._fields == ("config", "params", "family", "model_dir")

    public = [name for name in dir(loaded) if not name.startswith("_")]
    assert set(public) == {"config", "params", "family", "model_dir", "count", "index"}
    # The only callables left are tuple's inherited protocol (``count``/``index``):
    # there is no ``params()`` to call, no ``build``/``forward`` to drive.
    callables = [name for name in public if callable(getattr(loaded, name))]
    assert set(callables) <= {"count", "index"}, (
        f"LoadedModel grew callable attributes {callables} — it must stay plain data"
    )
    for absent in ("seq_len", "batch_size", "build", "forward", "predict", "run"):
        assert not hasattr(loaded, absent), f"LoadedModel must not carry {absent!r}"

    # ``params`` is a FIELD holding a dict, not a method.
    assert isinstance(loaded.params, dict)
    assert not callable(loaded.params)
    assert isinstance(loaded.config, etl_hf.ModelConfig)
    assert loaded.family in etl_hf.FAMILIES
    assert loaded.family == loaded.config.family
    assert isinstance(loaded.model_dir, Path)
    assert loaded.model_dir == checkpoint_dir

    # Immutable: a record, not a mutable handle on the checkpoint.
    with pytest.raises(AttributeError):
        loaded.family = "gpt2"  # type: ignore[misc]


def test_load_split_equals_load_whole(checkpoint_dir, loaded):
    """``load_config`` + ``load_params`` are the documented split of ``load_hf``."""
    assert etl_hf.load_config.__module__ == "etl_hf.load"
    assert etl_hf.load_params.__module__ == "etl_hf.load"

    config = etl_hf.load_config(checkpoint_dir)
    params = etl_hf.load_params(checkpoint_dir, config)

    # Same static config: one config = one graph, so the digest is the identity.
    assert config.digest == loaded.config.digest
    assert loaded.config == config

    # Same tree: exactly the param_specs keys, array-for-array identical values.
    specs = etl_hf.param_specs(config)
    assert set(params) == set(loaded.params) == set(specs)
    assert params is not loaded.params, "load_params must return a fresh dict"
    for key in specs:
        assert params[key].shape == tuple(specs[key].shape), key
        assert params[key].dtype == np.float32, key
        assert np.array_equal(params[key], loaded.params[key]), key

    # The one-call composition really is those two calls, in that order.
    # (A literal ``load_hf(dir) == (load_config(dir), load_params(dir, cfg))``
    # cannot be asserted: ``load_hf`` returns the 4-field record, and ``dict``
    # ``==`` on ndarray values raises ``ValueError`` — so it is pinned
    # field-by-field instead, which is strictly stronger than a bool.)
    whole = etl_hf.load_hf(checkpoint_dir)
    assert whole.config == config
    assert whole.family == loaded.family
    assert whole.model_dir == checkpoint_dir
    assert set(whole.params) == set(params)
    assert whole.params is not loaded.params
    for key in specs:
        assert np.array_equal(whole.params[key], params[key]), key


def test_load_params_rejects_a_tree_that_does_not_match(loaded):
    """The split exposes the validator: a wrong tree fails at the loader."""
    broken = dict(loaded.params)
    broken.pop(next(iter(broken)))
    with pytest.raises(etl_hf.HFWeightError):
        etl_hf.validate_params(loaded.config, broken)
    assert set(etl_hf.validate_params(loaded.config, loaded.params)) == set(loaded.params)


# ---------------------------------------------------------------------------
# program: the compute function is pure, memoised, and captures no parameters
# ---------------------------------------------------------------------------
def _closure_values(function) -> list:
    """Every filled closure cell of ``function`` (an unfilled cell holds nothing)."""
    values = []
    for cell in getattr(function, "__closure__", None) or ():
        try:
            values.append(cell.cell_contents)
        except ValueError:  # pragma: no cover - an as-yet-unfilled cell
            continue
    return values


def _contains_ndarray(value) -> bool:
    """Whether ``value`` is — or (through containers) contains — a numpy array."""
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_ndarray(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_ndarray(item) for item in value)
    return False


def test_forward_fn_is_the_compute_function(loaded):
    """The forward is a memoised ``@etl.defn`` — per static specialisation."""
    config = loaded.config
    fwd = etl_hf.forward_fn(config, seq_len=SEQ_LEN)
    assert etl_hf.forward_fn.__module__ == "etl_hf.program"
    assert getattr(fwd, "__etl_defn__", False) is True
    assert callable(fwd)

    # Memoised: the identity component of etl's build-cache key stays stable.
    assert etl_hf.forward_fn(config, seq_len=SEQ_LEN) is fwd
    # ...but every static specialisation is its own graph.
    assert etl_hf.forward_fn(config, seq_len=SEQ_LEN + 1) is not fwd
    assert etl_hf.forward_fn(config, seq_len=SEQ_LEN, batch_size=2) is not fwd
    assert etl_hf.forward_fn(config, seq_len=SEQ_LEN, pack_weights=True) is not fwd

    # A defn is never executed eagerly — calling it is an explicit error.
    with pytest.raises(etl.TraceError):
        fwd(loaded.params, np.zeros((1, SEQ_LEN), dtype=np.int32))


def test_forward_fn_captures_no_parameters(loaded):
    """The weights are graph INPUTS: tracing needs ``param_specs`` and nothing else."""
    config = loaded.config
    fwd = etl_hf.forward_fn(config, seq_len=SEQ_LEN)
    function = fwd.fn

    captured = (
        _closure_values(function)
        + list(getattr(function, "__defaults__", None) or ())
        + list((getattr(function, "__kwdefaults__", None) or {}).values())
    )
    for value in captured:
        assert not _contains_ndarray(value), (
            f"the forward closed over an ndarray ({type(value).__name__}) — "
            "weights must be graph inputs, never captured or `etl.constant`"
        )

    # The real proof: it traces from the specs alone, and the traced graph has
    # exactly one input per weight plus the ids tensor.
    specs = etl_hf.param_specs(config)
    assert isinstance(specs, dict) and specs
    ids_spec = etl.TensorSpec((1, SEQ_LEN), etl.int32)
    graph = etl.trace(fwd, specs, ids_spec)
    graph.verify()
    assert len(graph.tensor_specs) == len(specs) + 1
    assert graph.tensor_specs[-1] == ids_spec


# ---------------------------------------------------------------------------
# pipeline: distinct, documented staging stages (build -> run / predict)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("build", "etl_hf.pipeline"),
        ("run", "etl_hf.pipeline"),
        ("predict", "etl_hf.pipeline"),
        ("reference_logits", "etl_hf.program"),
        ("param_specs", "etl_hf.program"),
        ("forward_fn", "etl_hf.program"),
    ],
)
def test_staging_stages_are_distinct_public_functions(name, module):
    """Each stage is its own documented function of its own module."""
    value = getattr(etl_hf, name)
    assert inspect.isfunction(value), f"etl_hf.{name} is not a function"
    assert value.__module__ == module
    assert inspect.getdoc(value), f"etl_hf.{name} has no docstring"


def test_build_is_memoised_per_specialisation(loaded):
    """``build`` answers one ``Executable`` per (config, seq_len, batch, packing)."""
    config = loaded.config
    fwd = etl_hf.forward_fn(config, seq_len=SEQ_LEN)
    executable = etl_hf.build(fwd, config)
    assert isinstance(executable, etl.Executable)
    assert etl_hf.build(fwd, config) is executable, "no second compile per equal call"
    # The forward already carries the specialisation, so an agreeing explicit
    # value is the same build; a different graph is a different artifact.
    assert etl_hf.build(fwd, config, seq_len=SEQ_LEN) is executable
    other = etl_hf.forward_fn(config, seq_len=SEQ_LEN + 1)
    assert etl_hf.build(other, config) is not executable


def test_run_returns_a_tensor_on_the_executables_device(loaded, ids):
    """``run`` is the explicit runtime boundary: an ``etl.Tensor`` comes back."""
    config = loaded.config
    executable = etl_hf.build(etl_hf.forward_fn(config, seq_len=SEQ_LEN), config)
    logits = etl_hf.run(executable, loaded.params, ids)
    assert isinstance(logits, etl.Tensor)
    assert logits.shape == (1, SEQ_LEN, config.vocab_size)
    assert logits.dtype == etl.float32
    assert logits.device == etl.core.Device("cpu", 0)


def test_reference_logits_is_the_independent_oracle(loaded, ids):
    """The pure-numpy oracle agrees with a numpy-backend ``predict``."""
    config = loaded.config
    oracle = etl_hf.reference_logits(config, loaded.params, ids)
    predicted = etl_hf.predict(loaded, ids, seq_len=SEQ_LEN)
    assert isinstance(oracle, np.ndarray) and isinstance(predicted, np.ndarray)
    assert oracle.shape == predicted.shape == (1, SEQ_LEN, config.vocab_size)
    assert oracle.dtype == predicted.dtype == np.float32
    np.testing.assert_allclose(predicted, oracle, rtol=1e-5, atol=1e-5)
    assert float(np.max(np.abs(predicted - oracle))) <= 1e-5

    # The oracle is a *separate implementation*: etl_hf.reference imports no etl.
    imported = {
        value.__name__
        for value in vars(etl_hf.reference).values()
        if isinstance(value, types.ModuleType)
    }
    assert not {name for name in imported if name == "etl" or name.startswith("etl.")}


# ---------------------------------------------------------------------------
# import hygiene (subprocess: no state from this interpreter)
# ---------------------------------------------------------------------------
def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return completed.stdout


def test_import_etl_hf_pulls_in_no_optional_dependency():
    """torch / transformers / safetensors / iree must never be imported."""
    output = _run(
        "import sys\n"
        "import etl_hf\n"
        "forbidden = ('torch', 'transformers', 'safetensors')\n"
        "leaked = [name for name in forbidden if name in sys.modules]\n"
        "leaked += [name for name in ('iree', 'iree.compiler', 'iree.runtime')\n"
        "           if name in sys.modules]\n"
        "missing = [name for name in ('etl_hf.ops', 'etl_hf.pipeline', 'etl_hf.load',\n"
        "                             'etl_hf.program', 'etl_hf.weights')\n"
        "           if name not in sys.modules]\n"
        "print('LEAKED=' + ','.join(leaked))\n"
        "print('MISSING=' + ','.join(missing))\n"
        "print('BUILDMODULE=' + str('etl_hf.build' in sys.modules))\n"
    )
    reported = dict(line.split("=", 1) for line in output.strip().splitlines())
    assert reported["LEAKED"] == "", f"optional dependencies imported: {reported['LEAKED']}"
    assert reported["MISSING"] == "", f"submodules not imported: {reported['MISSING']}"
    assert reported["BUILDMODULE"] == "False", "the deleted etl_hf.build module reappeared"


def test_import_etl_hf_only_requires_numpy_and_etl():
    """The only new non-stdlib top-level imports of ``import etl_hf`` are ours."""
    output = _run(
        "import sys\n"
        "before = set(sys.modules)\n"
        "import etl_hf\n"
        "new = {name.split('.')[0] for name in set(sys.modules) - before}\n"
        "third_party = new - set(sys.stdlib_module_names)\n"
        "print(','.join(sorted(third_party)))\n"
    )
    allowed = {"etl", "etl_hf", "numpy", "__editable___etl_0_1_0_finder"}
    unexpected = [name for name in output.strip().split(",") if name not in allowed]
    assert unexpected == [], f"import etl_hf pulled in {unexpected}"


def test_public_api_is_usable_from_scratch():
    """Smoke-test the documented entry points in a fresh subprocess."""
    output = _run(
        "import etl_hf\n"
        "print(etl_hf.load_hf.__name__, etl_hf.__version__, sorted(etl_hf.FAMILIES),\n"
        "      'build:', type(etl_hf.build).__name__,\n"
        "      'oop:', hasattr(etl_hf, 'HFModel'))\n"
    )
    assert (
        output.strip()
        == "load_hf 0.1.0 ['gpt2', 'llama'] build: function oop: False"
    )
