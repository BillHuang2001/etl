"""Per-backend option env-var resolution tests (the options-override contract,
env half — ``etl.pipeline_options``).

Validates ``apply_env_options`` / ``ENV_OPTION_TABLE``: every per-backend
compiler flag/option must be settable at every pipeline stage — explicitly as
a kwarg to ``lower``/``compile``/``load``/``run`` (or the ``build``/
``evaluate`` sugar), or, when not passed explicitly, from an environment
variable. Precedence (binding): **explicit kwarg/option > env var > etl
default**.

Pinned here:

* no env set -> options unchanged; env vars read LAZILY at call time (every
  test drives them with ``monkeypatch`` after ``import etl``).
* stage-scoped application: ``ETL_IREE_COMPILE_ARGS`` applies at the compile
  stage only, ``ETL_IREE_RUNTIME_ARGS`` at the load stage only.
* explicit-wins-per-key; empty/whitespace env = unset; the caller's dict is
  never mutated.
* ``ETL_TVM_TARGET`` / ``ETL_TVM_PASS_CONFIGS`` (JSON object) /
  ``ETL_XLA_COMPILE_OPTIONS`` (base64 -> bytes) parse into the documented
  option keys.
* malformed env values raise ``core.BackendError`` naming the variable and
  the value — never a raw shlex/base64/json exception.
* an unknown backend / a stage with no table entry is a no-op.
* full-path stub tests (a functional ``"iree"`` stub registered under the
  adapter name, exactly like ``pipeline_env_defaults_test.py``): the env
  compile args reach the compile stage, the env runtime args reach the load
  stage (``etl.load`` applies env internally even when the ``build`` sugar
  passes no explicit load options), an explicit kwarg beats the env end to
  end, run options are forwarded, the numpy backend documents ignoring
  options (``rank_context`` still flows through), and
  ``BoundExecutable.backend`` reports the stub's backend name.

No test imports (or triggers the import of) a real compiler adapter: the
iree-family paths use stub ``Backend`` instances (helper-level) or a
functional stub registered under the name ``"iree"`` (full-path). The
backend half of the contract (per-adapter ``KNOWN_OPTIONS`` validation) is
pinned by ``tests/backends/test_options_override.py``.
"""

from __future__ import annotations

import base64
import sys

import numpy as np
import pytest

import etl
from etl import core
from etl.backends.options import STAGES
from etl.pipeline_options import ENV_OPTION_TABLE, apply_env_options

#: The env vars this module drives (cleared by the autouse fixture so no test
#: ever sees a stray value from the host environment).
OPTION_ENV_VARS = (
    "ETL_IREE_COMPILE_ARGS",
    "ETL_IREE_RUNTIME_ARGS",
    "ETL_XLA_COMPILE_OPTIONS",
    "ETL_TVM_TARGET",
    "ETL_TVM_PASS_CONFIGS",
)

#: Was the iree adapter already imported before this module ran (e.g. by an
#: earlier test in the same process)? Our tests must never be the cause.
_IREE_IMPORTED_BEFORE = "etl.backends.adapters.iree" in sys.modules


def _assert_no_new_iree_import():
    """Our resolution/tests must never import the iree adapter."""
    if not _IREE_IMPORTED_BEFORE:
        assert "etl.backends.adapters.iree" not in sys.modules


@pytest.fixture(autouse=True)
def _clean_option_env(monkeypatch):
    """Every test starts with the option env vars unset."""
    for var in OPTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Shared graph definition
# ---------------------------------------------------------------------------


@etl.defn
def _linear(x, w, b):
    """A small linear layer: dot + bias + relu (three positional inputs)."""
    return etl.relu(etl.add(etl.dot(x, w), b))


_LINEAR_SPECS = (
    etl.TensorSpec((2, 3), etl.float32, name="x"),
    etl.TensorSpec((3, 4), etl.float32, name="w"),
    etl.TensorSpec((4,), etl.float32, name="b"),
)


def _linear_inputs():
    rng = np.random.default_rng(0)
    return (
        rng.random((2, 3)).astype("float32"),
        rng.random((3, 4)).astype("float32"),
        rng.random((4,)).astype("float32"),
    )


def _ref(x, w, b):
    return np.maximum(0.0, np.asarray(x) @ np.asarray(w) + np.asarray(b))


# ---------------------------------------------------------------------------
# ENV_OPTION_TABLE shape
# ---------------------------------------------------------------------------


def test_env_option_table_entries():
    """The env table maps exactly the documented (backend, stage) pairs."""
    assert set(ENV_OPTION_TABLE) == {
        ("iree", "compile"),
        ("iree", "load"),
        ("xla", "compile"),
        ("tvm", "compile"),
    }


def test_env_option_table_keys_and_parsers():
    """The documented env var -> option key mapping (parsers are functions)."""
    iree_compile = ENV_OPTION_TABLE[("iree", "compile")]
    assert len(iree_compile) == 1
    var, key, parser = iree_compile[0]
    assert (var, key) == ("ETL_IREE_COMPILE_ARGS", "iree_compile_args")
    assert callable(parser)

    iree_load = ENV_OPTION_TABLE[("iree", "load")]
    assert len(iree_load) == 1
    var, key, parser = iree_load[0]
    assert (var, key) == ("ETL_IREE_RUNTIME_ARGS", "iree_runtime_args")
    assert callable(parser)

    compile_entries = ENV_OPTION_TABLE[("tvm", "compile")]
    assert {entry[1] for entry in compile_entries} == {
        "tvm_target",
        "tvm_pass_configs",
    }
    assert {entry[0] for entry in compile_entries} == {
        "ETL_TVM_TARGET",
        "ETL_TVM_PASS_CONFIGS",
    }
    assert all(callable(entry[2]) for entry in compile_entries)
    assert ENV_OPTION_TABLE[("xla", "compile")][0][:2] == (
        "ETL_XLA_COMPILE_OPTIONS",
        "xla_compile_options",
    )


def test_env_option_table_stages_are_canonical():
    """Every table stage is one of the canonical option stages."""
    for (backend, stage) in ENV_OPTION_TABLE:
        assert backend in ("iree", "xla", "tvm")
        assert stage in STAGES


# ---------------------------------------------------------------------------
# apply_env_options — helper-level contract (no backends involved)
# ---------------------------------------------------------------------------


def test_no_env_options_unchanged():
    """No env vars -> the options dict passes through untouched."""
    for backend, stage in (("iree", "compile"), ("iree", "load"),
                           ("xla", "compile"), ("tvm", "compile")):
        assert apply_env_options(backend, {}, stage) == {}
    assert apply_env_options("iree", {"target_backends": ["llvm-cpu"]},
                             "compile") == {"target_backends": ["llvm-cpu"]}


def test_env_read_lazily_per_call(monkeypatch):
    """Env vars are read at call time — set after import, honored on the next
    call; unset before the next call, no longer applied."""
    assert apply_env_options("iree", {}, "compile") == {}
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": ["--iree-llvmcpu-target-cpu=native"]
    }
    monkeypatch.delenv("ETL_IREE_COMPILE_ARGS")
    assert apply_env_options("iree", {}, "compile") == {}


def test_explicit_wins_per_key(monkeypatch):
    """An explicit option key always wins over the env var; other keys are
    still env-supplied."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--flag-from-env")
    resolved = apply_env_options(
        "iree", {"iree_compile_args": ["--flag-explicit"]}, "compile"
    )
    assert resolved == {"iree_compile_args": ["--flag-explicit"]}
    # Mixed: explicit target_backends + env-supplied iree_compile_args.
    resolved = apply_env_options(
        "iree", {"target_backends": ["cuda"]}, "compile"
    )
    assert resolved == {
        "target_backends": ["cuda"],
        "iree_compile_args": ["--flag-from-env"],
    }


@pytest.mark.parametrize("value", ["", "   ", "\t \n"])
def test_empty_or_whitespace_env_is_unset(monkeypatch, value):
    """An empty/whitespace env value is treated as unset."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", value)
    assert apply_env_options("iree", {}, "compile") == {}


def test_per_stage_application_compile_args(monkeypatch):
    """ETL_IREE_COMPILE_ARGS applies at the iree compile stage only — never
    at lower, load, run, or on other backends."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": ["--iree-llvmcpu-target-cpu=native"]
    }
    for stage in ("lower", "load", "run"):
        assert apply_env_options("iree", {}, stage) == {}
    assert apply_env_options("xla", {}, "compile") == {}
    assert apply_env_options("tvm", {}, "compile") == {}


def test_per_stage_application_runtime_args(monkeypatch):
    """ETL_IREE_RUNTIME_ARGS applies at the iree load stage only."""
    monkeypatch.setenv("ETL_IREE_RUNTIME_ARGS", "--cuda_async_allocations=false")
    assert apply_env_options("iree", {}, "load") == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    for stage in ("lower", "compile", "run"):
        assert apply_env_options("iree", {}, stage) == {}


def test_flag_list_parsing_quotes(monkeypatch):
    """Flag lists use shlex syntax — quoting is honored."""
    monkeypatch.setenv(
        "ETL_IREE_COMPILE_ARGS",
        '--iree-llvmcpu-target-cpu=native '
        '--iree-hal-dump-executable-sources-to="dir with spaces"',
    )
    assert apply_env_options("iree", {}, "compile") == {
        "iree_compile_args": [
            "--iree-llvmcpu-target-cpu=native",
            "--iree-hal-dump-executable-sources-to=dir with spaces",
        ]
    }


def test_tvm_target_and_pass_configs_mapping(monkeypatch):
    """ETL_TVM_TARGET and ETL_TVM_PASS_CONFIGS map onto the tvm compile
    options (target stripped, pass configs parsed as a JSON object)."""
    monkeypatch.setenv("ETL_TVM_TARGET", "  llvm -mcpu=native  ")
    monkeypatch.setenv(
        "ETL_TVM_PASS_CONFIGS",
        '{"tir.disable_assert": false, "relax.transform.": {"a": 1}}',
    )
    resolved = apply_env_options("tvm", {}, "compile")
    assert resolved["tvm_target"] == "llvm -mcpu=native"
    assert resolved["tvm_pass_configs"] == {
        "tir.disable_assert": False,
        "relax.transform.": {"a": 1},
    }
    # tvm has no other stages.
    for stage in ("lower", "load", "run"):
        assert apply_env_options("tvm", {}, stage) == {}


def test_xla_compile_options_base64_decode(monkeypatch):
    """ETL_XLA_COMPILE_OPTIONS is a base64 payload decoded to bytes."""
    payload = b"\x1a\x04\x20\x01\x28\x01"  # a serialized CompileOptionsProto
    monkeypatch.setenv("ETL_XLA_COMPILE_OPTIONS", base64.b64encode(payload).decode("ascii"))
    assert apply_env_options("xla", {}, "compile") == {
        "xla_compile_options": payload
    }


@pytest.mark.parametrize(
    ("var", "value", "fragment"),
    [
        ("ETL_IREE_COMPILE_ARGS", "--flag 'unclosed", "not a parseable"),
        ("ETL_XLA_COMPILE_OPTIONS", "%%%not-base64%%%", "not valid base64"),
        ("ETL_TVM_PASS_CONFIGS", "{bad json", "not valid JSON"),
        ("ETL_TVM_PASS_CONFIGS", "[1, 2]", "expected a JSON object"),
    ],
)
def test_malformed_env_values_raise_backend_error(monkeypatch, var, value, fragment):
    """A malformed env value raises core.BackendError naming the variable and
    the value — never a raw shlex/base64/json exception."""
    backend, stage = {
        "ETL_IREE_COMPILE_ARGS": ("iree", "compile"),
        "ETL_XLA_COMPILE_OPTIONS": ("xla", "compile"),
        "ETL_TVM_PASS_CONFIGS": ("tvm", "compile"),
    }[var]
    monkeypatch.setenv(var, value)
    with pytest.raises(core.BackendError) as excinfo:
        apply_env_options(backend, {}, stage)
    message = str(excinfo.value)
    assert var in message
    assert value in message
    assert fragment in message


def test_unknown_backend_noop(monkeypatch):
    """An unknown backend name (or None) is a no-op — no error, no keys."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--flag")
    assert apply_env_options("no-such-backend", {}, "compile") == {}
    assert apply_env_options("no-such-backend",
                             {"existing": 1}, "compile") == {"existing": 1}
    assert apply_env_options(None, {}, "compile") == {}


def test_caller_dict_never_mutated(monkeypatch):
    """apply_env_options returns a copy; the caller's dict is untouched."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--flag")
    opts = {"target_backends": ["llvm-cpu"]}
    resolved = apply_env_options("iree", opts, "compile")
    assert opts == {"target_backends": ["llvm-cpu"]}
    assert resolved is not opts
    assert resolved == {
        "target_backends": ["llvm-cpu"],
        "iree_compile_args": ["--flag"],
    }


# ---------------------------------------------------------------------------
# Stub backends (no real adapter imports)
# ---------------------------------------------------------------------------


class _IreeStubExecutable:
    """Minimal backend executable wrapper reporting ``backend_name="iree"``
    while delegating everything to the numpy backend executable."""

    backend_name = "iree"

    def __init__(self, inner):
        self._inner = inner

    @property
    def functions(self):
        return self._inner.functions

    @property
    def device(self):
        return self._inner.device

    def run(self, flat_input_tensors, options=None):
        return self._inner.run(flat_input_tensors, options)

    def save(self, path):
        return self._inner.save(path)


class _RecordingIreeStub(etl.backends.Backend):
    """Functional ``"iree"`` stub for full-path tests: delegates everything to
    the numpy backend but stamps its own name into the staged objects, and
    records the compile/load options it receives (per stage)."""

    name = "iree"
    capabilities = etl.backends.numpy_backend.capabilities
    recorded_compile_options: list = []
    recorded_load_options: list = []

    def lower(self, graph, options=None):
        lp = etl.backends.numpy_backend.lower(graph, options)
        return etl.backends.LoweredProgram(
            backend=self.name, signature=lp.signature, payload=lp.payload
        )

    def compile(self, lowered, options=None):
        _RecordingIreeStub.recorded_compile_options.append(dict(options or {}))
        art = etl.backends.numpy_backend.compile(
            etl.backends.LoweredProgram(
                backend="numpy", signature=lowered.signature, payload=lowered.payload
            ),
            options,
        )
        return etl.backends.CompiledArtifact(
            backend=self.name,
            signature=art.signature,
            target=art.target,
            payload=art.payload,
        )

    def load(self, artifact, device=None, options=None):
        _RecordingIreeStub.recorded_load_options.append(dict(options or {}))
        art = etl.backends.CompiledArtifact(
            backend="numpy",
            signature=artifact.signature,
            target=artifact.target,
            payload=artifact.payload,
        )
        return _IreeStubExecutable(etl.backends.numpy_backend.load(art, device))


@pytest.fixture
def iree_stub_registered():
    """Register the functional ``"iree"`` stub for a full-path test and
    restore the registry afterwards.

    The stub is installed by direct registry assignment (save/restore of the
    previous entry) so these tests also run when a real ``"iree"`` backend is
    already registered in this process (e.g. the compiler extra installed):
    the stub replaces it for the duration of the test and is never visible to
    other tests. Tests run sequentially, so no cross-test interference.
    """
    from etl.backends import registry as _registry

    stub = _RecordingIreeStub()
    _RecordingIreeStub.recorded_compile_options = []
    _RecordingIreeStub.recorded_load_options = []
    previous = _registry._registry.get("iree")
    _registry._registry["iree"] = stub
    try:
        yield stub
    finally:
        if previous is None:
            _registry._registry.pop("iree", None)
        else:
            _registry._registry["iree"] = previous
        _RecordingIreeStub.recorded_compile_options = []
        _RecordingIreeStub.recorded_load_options = []


# ---------------------------------------------------------------------------
# Full path through a registered "iree" stub (no real adapter import)
# ---------------------------------------------------------------------------


def test_full_path_env_compile_args_reaches_compile(monkeypatch, iree_stub_registered):
    """ETL_IREE_COMPILE_ARGS reaches the compile stage through ``etl.build``."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--iree-llvmcpu-target-cpu=native")
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="iree")
    recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert recorded["iree_compile_args"] == ["--iree-llvmcpu-target-cpu=native"]
    # The build sugar also inferred the device-based target_backends.
    assert recorded["target_backends"] == ["llvm-cpu"]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_explicit_compile_kwarg_beats_env(monkeypatch, iree_stub_registered):
    """An explicit iree_compile_args kwarg beats ETL_IREE_COMPILE_ARGS."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--flag-from-env")
    exe = etl.build(
        _linear, *_LINEAR_SPECS, backend="iree",
        iree_compile_args=["--flag-explicit"],
    )
    assert _RecordingIreeStub.recorded_compile_options[-1][
        "iree_compile_args"
    ] == ["--flag-explicit"]
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_env_runtime_args_reaches_load(monkeypatch, iree_stub_registered):
    """ETL_IREE_RUNTIME_ARGS reaches the load stage: ``etl.load`` applies the
    env internally, so even the ``build`` sugar (which passes no explicit
    load options) delivers the env-supplied load option."""
    monkeypatch.setenv("ETL_IREE_RUNTIME_ARGS", "--cuda_async_allocations=false")
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="iree")
    assert _RecordingIreeStub.recorded_load_options[-1] == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_load_stage_explicit_runtime_args_beat_env(monkeypatch, iree_stub_registered):
    """At the explicit load stage, an explicit iree_runtime_args kwarg beats
    ETL_IREE_RUNTIME_ARGS."""
    monkeypatch.setenv("ETL_IREE_RUNTIME_ARGS", "--flag-from-env")
    graph = etl.trace(_linear, *_LINEAR_SPECS)
    lowered = etl.lower(graph, backend="iree")
    artifact = etl.compile(lowered, backend="iree")
    exe = etl.load(
        artifact, backend="iree",
        iree_runtime_args=["--cuda_async_allocations=false"],
    )
    assert _RecordingIreeStub.recorded_load_options[-1] == {
        "iree_runtime_args": ["--cuda_async_allocations=false"]
    }
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_no_env_stages_stay_clean(monkeypatch, iree_stub_registered):
    """With no option env vars, the compile/load stages receive no option
    keys (the device-based target_backends inference is separate, documented
    env-defaults behavior)."""
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="iree")
    compile_recorded = _RecordingIreeStub.recorded_compile_options[-1]
    assert "iree_compile_args" not in compile_recorded
    assert compile_recorded == {"target_backends": ["llvm-cpu"]}
    assert _RecordingIreeStub.recorded_load_options[-1] == {}
    assert exe.device == core.Device("cpu", 0)
    _assert_no_new_iree_import()


def test_full_path_run_options_forwarded_and_executes(monkeypatch, iree_stub_registered):
    """The stub executable runs end to end; per-run options are forwarded to
    the backend run (the numpy interpreter reads ``rank_context`` from the
    options dict — the documented numpy behavior, verified through the
    stub)."""
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="iree")
    x, w, b = _linear_inputs()
    out = etl.run(exe, x, w, b)
    np.testing.assert_allclose(out.numpy(), _ref(x, w, b))
    out_rank = etl.run(exe, x, w, b, rank_context=etl.dist.RankContext(0, 1))
    np.testing.assert_allclose(out_rank.numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()


def test_numpy_documented_ignore(monkeypatch, iree_stub_registered):
    """The numpy reference backend documents ignoring options: cross-backend
    option dicts (e.g. target_backends) pass through ``etl.run`` without
    error, and ``rank_context`` still flows into the interpreter."""
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="numpy")
    x, w, b = _linear_inputs()
    out = etl.run(exe, x, w, b)
    np.testing.assert_allclose(
        etl.run(exe, x, w, b, target_backends=["cuda"]).numpy(),
        _ref(x, w, b),
    )
    np.testing.assert_allclose(
        etl.run(exe, x, w, b, rank_context=etl.dist.RankContext(0, 1)).numpy(),
        _ref(x, w, b),
    )
    assert out.dtype == np.dtype("float32")
    _assert_no_new_iree_import()


def test_bound_executable_backend_property(monkeypatch, iree_stub_registered):
    """``BoundExecutable.backend`` delegates to the wrapped executable and
    reports the stub's backend name."""
    exe = etl.build(_linear, *_LINEAR_SPECS, backend="iree")
    assert exe.backend == "iree"
    bound = etl.bind(exe, x=_linear_inputs()[0])
    assert bound.backend == "iree"
    w, b = _linear_inputs()[1:]
    x = _linear_inputs()[0]
    np.testing.assert_allclose(etl.run(bound, w, b).numpy(), _ref(x, w, b))
    _assert_no_new_iree_import()
