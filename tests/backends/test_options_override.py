"""Backend options-override contract tests (the backend half).

Pins the per-backend option machinery of the options-override contract:
``etl/backends/options.py::validate_options`` (option NAMES validated against
the UNION of the per-stage ``KNOWN_OPTIONS`` sets; option VALUES are never
validated by etl — the compiler validates them), every adapter's
``KNOWN_OPTIONS`` declaration, the iree compile-flag policy
(``_resolve_iree_compile_args``: minimal defaults, each default overridable
by name collision, the small documented deny list), the iree runtime-flag
paths (``_configure_cuda_runtime_flags`` suppression + ``_apply_iree_runtime_args``
loud parsing), the tvm ``tvm_target`` / ``tvm_pass_configs`` options
(version-adaptive pass_configs), the xla ``plugin_path`` /
``xla_compile_options`` options (against the fake PJRT plugin — the
``test_pjrt_ctypes_plugin.py`` fixture), and the numpy documented-ignore.

The env half (``etl.pipeline_options``, precedence explicit > env > default,
stub full-path) is pinned by ``tests/test_pipeline_options.py``.

Adapter-test conventions (same as ``test_adapter_{iree,xla,tvm}.py``): real
compilers run, so the 2s-per-file rule does NOT apply — keep compile counts
low (module-scoped fixtures; small graphs). The iree/tvm sections gate with
per-test ``pytest.importorskip`` so the pure-``validate_options`` and
helper-level tests run everywhere; the xla section uses the fake plugin built
from C at test time (skips when no C compiler is available).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

import etl
from etl import core
from etl.backends.options import validate_options
from etl.backends.adapters import iree as iree_adapt
from etl.backends.adapters import tvm as tvm_adapt
from etl.backends.adapters import xla as xla_adapt
from etl.backends.adapters import xla_util

from tests.backends.test_pjrt_ctypes_plugin import _build_plugin


# ---------------------------------------------------------------------------
# Shared graphs
# ---------------------------------------------------------------------------


def _add_fn(x):
    return etl.add(x, 1.0)


_ADD_SPEC = etl.TensorSpec((4,), etl.float32)
_ADD_INPUT = np.ones(4, dtype="float32")
_ADD_EXPECTED = np.full(4, 2.0, dtype="float32")


def _argsort_fn(x):
    return etl.argsort(x)


_ARGSORT_SPEC = etl.TensorSpec((8,), etl.float32)
_ARGSORT_INPUT = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 7.0, 6.0], "float32")


def _while_fn(init):
    return etl.while_loop(
        lambda c: etl.less(c, 4.0), lambda c: etl.add(c, 1.0), init
    )


_WHILE_SPEC = etl.TensorSpec((), etl.float32)


# ---------------------------------------------------------------------------
# 1. validate_options — union semantics + per-adapter declarations
# ---------------------------------------------------------------------------


def test_validate_options_union_semantics_iree():
    """A key valid for another stage of the same backend passes (accepted and
    ignored) at this stage — the build/evaluate sugar forwards one options
    dict to every stage."""
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    validate_options({"target_backends": ["llvm-cpu"]}, known, "iree", "lower")
    validate_options(
        {"iree_compile_args": ["--flag"]}, known, "iree", "lower"
    )
    validate_options({"iree_runtime_args": []}, known, "iree", "compile")
    validate_options({"rng_bit_generator": True}, known, "iree", "run")
    validate_options({"sort_emission": "count"}, known, "iree", "compile")
    validate_options({"while_init_rewrite": False}, known, "iree", "compile")


def test_validate_options_union_semantics_xla_tvm():
    """The xla load-stage key passes at xla compile; the tvm compile keys pass
    at tvm lower."""
    validate_options(
        {"plugin_path": "/tmp/p.so"}, xla_adapt.XlaBackend.KNOWN_OPTIONS,
        "xla", "compile",
    )
    validate_options(
        {"tvm_target": "llvm", "tvm_pass_configs": {}},
        tvm_adapt.TvmBackend.KNOWN_OPTIONS, "tvm", "lower",
    )


def test_validate_options_none_or_empty_passes():
    """None/empty options pass silently at every stage."""
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    validate_options(None, known, "iree", "compile")
    validate_options({}, known, "iree", "load")
    validate_options(None, xla_adapt.XlaBackend.KNOWN_OPTIONS, "xla", "run")


def test_validate_options_unknown_key_lists_known_options_per_stage():
    """A key valid for NO stage raises core.BackendError listing the known
    options per stage (sorted; '(none)' for empty sets)."""
    with pytest.raises(core.BackendError) as excinfo:
        validate_options(
            {"bogus": 1}, iree_adapt.IreeBackend.KNOWN_OPTIONS, "iree", "compile"
        )
    message = str(excinfo.value)
    assert "the iree backend does not recognize the compile option 'bogus'" in message
    assert "lower: rng_bit_generator, sort_emission, while_init_rewrite" in message
    assert "compile: iree_compile_args, opt_level, target_backends" in message
    assert "load: iree_runtime_args" in message
    assert "run: (none)" in message


def test_validate_options_unknown_key_xla_tvm():
    with pytest.raises(core.BackendError) as excinfo:
        validate_options(
            {"bogus": 1}, xla_adapt.XlaBackend.KNOWN_OPTIONS, "xla", "run"
        )
    message = str(excinfo.value)
    assert "does not recognize the run option" in message
    assert "compile: opt_level, plugin_path, xla_compile_options" in message
    assert "load: plugin_path" in message

    with pytest.raises(core.BackendError, match="tvm backend does not recognize"):
        validate_options(
            {"bogus": 1}, tvm_adapt.TvmBackend.KNOWN_OPTIONS, "tvm", "compile"
        )


def test_adapter_known_options_declarations():
    """The merged per-adapter declarations (single source of truth for the
    contract)."""
    assert iree_adapt.IreeBackend.KNOWN_OPTIONS == {
        "lower": frozenset(
            {"rng_bit_generator", "sort_emission", "while_init_rewrite"}
        ),
        "compile": frozenset(
            {"target_backends", "iree_compile_args", "opt_level"}
        ),
        "load": frozenset({"iree_runtime_args"}),
        "run": frozenset(),
    }
    assert xla_adapt.XlaBackend.KNOWN_OPTIONS == {
        "lower": frozenset({"rng_bit_generator"}),
        "compile": frozenset(
            {"plugin_path", "xla_compile_options", "opt_level"}
        ),
        "load": frozenset({"plugin_path"}),
        "run": frozenset(),
    }
    assert tvm_adapt.TvmBackend.KNOWN_OPTIONS == {
        "lower": frozenset({"rng_bit_generator"}),
        "compile": frozenset(
            {"tvm_target", "tvm_pass_configs", "opt_level"}
        ),
        "load": frozenset(),
        "run": frozenset(),
    }


def test_iree_unknown_lower_option_rejected_without_compiler():
    """A lower-stage unknown key raises before any compiler work (the shared
    CompilerBackend validates before exporting) — no IREE needed."""
    graph = etl.trace(_argsort_fn, _ARGSORT_SPEC)
    backend = iree_adapt.IreeBackend()
    with pytest.raises(core.BackendError, match="does not recognize the lower option"):
        backend.lower(graph, {"bogus": 1})


def test_xla_unknown_load_option_rejected_without_plugin():
    """The xla load stage validates options before any plugin work — no
    plugin needed for the rejection."""
    with pytest.raises(core.BackendError, match="does not recognize the load option"):
        xla_adapt.XlaBackend().load(
            types.SimpleNamespace(backend="xla"), options={"bogus": 1}
        )


# ---------------------------------------------------------------------------
# 2. iree compile-flag policy (pure helpers — no IREE dependency)
# ---------------------------------------------------------------------------


def test_iree_resolve_compile_args_defaults_survive():
    """With no user flags the final extra_args are exactly etl's minimal
    defaults (f64 semantics + the two llvm-cpu portability flags)."""
    assert iree_adapt._resolve_iree_compile_args(("llvm-cpu",), None) == [
        "--iree-input-demote-f64-to-f32=false",
        "--iree-llvmcpu-target-cpu=generic",
        "--iree-llvmcpu-link-embedded=false",
    ]


def test_iree_resolve_compile_args_cuda_skips_llvmcpu_defaults():
    """The llvm-cpu-only defaults are added only for llvm-cpu targets."""
    assert iree_adapt._resolve_iree_compile_args(("cuda",), None) == [
        "--iree-input-demote-f64-to-f32=false"
    ]


def test_iree_resolve_compile_args_user_flag_drops_default_by_name():
    """A user flag whose NAME matches a default drops the default — the user
    wins (never both)."""
    args = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), ["--iree-input-demote-f64-to-f32=true"]
    )
    assert "--iree-input-demote-f64-to-f32=true" in args
    assert "--iree-input-demote-f64-to-f32=false" not in args
    assert args[-1] == "--iree-input-demote-f64-to-f32=true"


def test_iree_resolve_compile_args_unknown_flag_passes_through():
    """An arbitrary unknown flag passes through unvalidated (the compiler
    validates flag values; etl never guesses)."""
    args = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",), ["--iree-llvmcpu-target-cpu=native", "--any-flag=whatever"]
    )
    assert "--iree-llvmcpu-target-cpu=native" in args
    assert "--iree-llvmcpu-target-cpu=generic" not in args  # name collision
    assert "--any-flag=whatever" in args


@pytest.mark.parametrize(
    ("flag", "fragment"),
    [
        (
            "--iree-hal-target-backends=cuda",
            "use the 'target_backends' compile option instead",
        ),
        ("--iree-input-type=stablehlo", "etl always feeds StableHLO"),
        (
            "--iree-vm-bytecode-module-output-format=x",
            "would break artifact loading",
        ),
    ],
)
def test_iree_resolve_compile_args_deny_list(flag, fragment):
    """The documented deny list rejects genuinely required flags with the
    documented reason — never silently overwritten, never passed through."""
    with pytest.raises(core.BackendError) as excinfo:
        iree_adapt._resolve_iree_compile_args(("llvm-cpu",), [flag])
    assert "denies the iree-compile flag" in str(excinfo.value)
    assert fragment in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. iree runtime-flag paths (fake iree.runtime — no IREE dependency)
# ---------------------------------------------------------------------------


def _install_fake_iree_runtime(monkeypatch, parse_flags_impl):
    """Install a fake ``iree.runtime`` module (flags.parse_flags recorder) so
    the cuda-runtime-flag helpers are pinned deterministically without the
    real runtime. Returns the recorded ``parse_flags`` call tuples.

    ``import iree.runtime as rt`` resolves the submodule through
    ``getattr(iree, "runtime")`` when the real ``iree`` package is loaded, so
    the package attribute is patched in addition to the ``sys.modules``
    entry (both restored by monkeypatch afterwards).
    """
    import sys

    calls = []

    def _parse_flags(*args):
        calls.append(args)
        return parse_flags_impl(*args)

    fake = types.ModuleType("iree.runtime")
    flags = types.ModuleType("iree.runtime.flags")
    flags.parse_flags = _parse_flags
    fake.flags = flags
    monkeypatch.setitem(sys.modules, "iree.runtime", fake)
    iree_pkg = sys.modules.get("iree")
    if iree_pkg is not None:
        monkeypatch.setattr(iree_pkg, "runtime", fake, raising=False)
    return calls


def test_iree_cuda_flags_default_applied(monkeypatch):
    """Neither the user's runtime args nor IREE_PY_RUNTIME_FLAGS mention
    cuda_async_allocations -> etl's default is parsed."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    monkeypatch.setattr(iree_adapt, "_CUDA_FLAGS_CONFIGURED", False)
    iree_adapt._configure_cuda_runtime_flags()
    assert calls == [("--cuda_async_allocations=false",)]


def test_iree_cuda_flags_suppressed_by_user_args(monkeypatch):
    """The user's explicit iree_runtime_args mentioning cuda_async_allocations
    suppress etl's default (the explicit option wins by last-wins semantics)."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    monkeypatch.setattr(iree_adapt, "_CUDA_FLAGS_CONFIGURED", False)
    iree_adapt._configure_cuda_runtime_flags(("--cuda_async_allocations=true",))
    assert calls == []


def test_iree_cuda_flags_suppressed_by_env(monkeypatch):
    """IREE_PY_RUNTIME_FLAGS mentioning cuda_async_allocations suppresses
    etl's default (env parsed at iree import wins over the later default)."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    monkeypatch.setattr(iree_adapt, "_CUDA_FLAGS_CONFIGURED", False)
    monkeypatch.setenv("IREE_PY_RUNTIME_FLAGS", "--cuda_async_allocations=true")
    iree_adapt._configure_cuda_runtime_flags()
    assert calls == []


def test_iree_cuda_flags_configured_once(monkeypatch):
    """The process-global flag is parsed at most once (idempotent)."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    monkeypatch.setattr(iree_adapt, "_CUDA_FLAGS_CONFIGURED", False)
    iree_adapt._configure_cuda_runtime_flags()
    iree_adapt._configure_cuda_runtime_flags(("--cuda_async_allocations=true",))
    assert calls == [("--cuda_async_allocations=false",)]


def test_iree_apply_runtime_args_valid_flag(monkeypatch):
    """A valid runtime flag is parsed via iree.runtime.flags.parse_flags."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    iree_adapt._apply_iree_runtime_args(["--cuda_async_allocations=false"])
    assert calls == [("--cuda_async_allocations=false",)]


def test_iree_apply_runtime_args_empty_is_noop(monkeypatch):
    """None/empty runtime args are a no-op (no parse_flags call)."""
    calls = _install_fake_iree_runtime(monkeypatch, lambda *a: None)
    iree_adapt._apply_iree_runtime_args(None)
    iree_adapt._apply_iree_runtime_args([])
    assert calls == []


def test_iree_apply_runtime_args_rejected_flag_loud(monkeypatch):
    """A flag the runtime rejects raises core.BackendError naming the option
    and the flags — never silently swallowed."""

    def _reject(*args):
        raise ValueError("unknown flag")

    calls = _install_fake_iree_runtime(monkeypatch, _reject)
    with pytest.raises(core.BackendError) as excinfo:
        iree_adapt._apply_iree_runtime_args(["--definitely-not-a-real-iree-flag"])
    message = str(excinfo.value)
    assert "iree_runtime_args" in message
    assert "--definitely-not-a-real-iree-flag" in message
    assert calls == [("--definitely-not-a-real-iree-flag",)]


# ---------------------------------------------------------------------------
# 4. iree end-to-end (real IREE — llvm-cpu)
# ---------------------------------------------------------------------------


def _require_iree():
    pytest.importorskip("iree.compiler")
    pytest.importorskip("iree.runtime")


@pytest.fixture(scope="module")
def iree_argsort_lowered():
    """One shared trace+lower of the argsort graph (no compile)."""
    _require_iree()
    return etl.lower(etl.trace(_argsort_fn, _ARGSORT_SPEC), backend="iree")


@pytest.fixture(scope="module")
def iree_argsort_exe(iree_argsort_lowered):
    """One shared compiled+loaded executable carrying USER compile and load
    options: iree_compile_args at compile, iree_runtime_args at load."""
    artifact = etl.compile(
        iree_argsort_lowered, iree_compile_args=["--iree-llvmcpu-target-cpu=generic"]
    )
    return etl.load(artifact, iree_runtime_args=["--cuda_async_allocations=false"])


def test_iree_compile_and_load_options_end_to_end(iree_argsort_exe):
    """iree_compile_args (compile) + iree_runtime_args (load) on a real
    llvm-cpu build: parity with the numpy reference."""
    out = etl.run(iree_argsort_exe, _ARGSORT_INPUT)
    np.testing.assert_array_equal(out.numpy(), np.argsort(_ARGSORT_INPUT))


def test_iree_run_stage_unknown_option_rejected(iree_argsort_exe):
    """The iree run stage has NO known options in v1: a non-empty run options
    dict raises core.BackendError."""
    with pytest.raises(core.BackendError, match="does not recognize the run option"):
        etl.run(iree_argsort_exe, _ARGSORT_INPUT, bogus=1)


def test_iree_real_runtime_garbage_flag_rejected(iree_argsort_exe):
    """A garbage runtime flag is rejected by the REAL iree runtime with a
    BackendError naming the option (loud, never swallowed)."""
    artifact = iree_argsort_exe.backend_executable.artifact
    with pytest.raises(core.BackendError) as excinfo:
        etl.load(artifact, iree_runtime_args=["--definitely-not-a-real-iree-flag"])
    assert "iree_runtime_args" in str(excinfo.value)
    assert "--definitely-not-a-real-iree-flag" in str(excinfo.value)


def test_iree_explicit_compile_kwarg_beats_env(monkeypatch, iree_argsort_lowered):
    """End to end: ETL_IREE_COMPILE_ARGS carries a flag iree-compile rejects;
    the explicit iree_compile_args kwarg wins (explicit > env > default)."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--definitely-not-a-real-iree-flag")
    artifact = etl.compile(
        iree_argsort_lowered, iree_compile_args=["--iree-llvmcpu-target-cpu=generic"]
    )
    exe = etl.load(artifact)
    out = etl.run(exe, _ARGSORT_INPUT)
    np.testing.assert_array_equal(out.numpy(), np.argsort(_ARGSORT_INPUT))


def test_iree_env_compile_args_alone_rejected(monkeypatch, iree_argsort_lowered):
    """Without an explicit kwarg, the env flag IS applied — and the compiler's
    rejection surfaces as BackendError (the env path is real, not inert)."""
    monkeypatch.setenv("ETL_IREE_COMPILE_ARGS", "--definitely-not-a-real-iree-flag")
    with pytest.raises(core.BackendError, match="iree-compile"):
        etl.compile(iree_argsort_lowered)


def test_iree_lower_reserved_option_sort_emission():
    """``sort_emission`` is a RESERVED exporter lower key declared in
    KNOWN_OPTIONS: it passes validation and the pair emission compiles and
    runs on llvm-cpu."""
    _require_iree()
    exe = etl.build(
        _argsort_fn, _ARGSORT_SPEC, backend="iree", sort_emission="pair"
    )
    out = etl.run(exe, _ARGSORT_INPUT)
    np.testing.assert_array_equal(out.numpy(), np.argsort(_ARGSORT_INPUT))


def test_iree_lower_reserved_option_while_init_rewrite():
    """``while_init_rewrite`` is a RESERVED exporter lower key declared in
    KNOWN_OPTIONS: it passes validation and a scalar-init while loop compiles
    and runs on llvm-cpu with the rewrite disabled."""
    _require_iree()
    exe = etl.build(
        _while_fn, _WHILE_SPEC, backend="iree", while_init_rewrite=False
    )
    out = etl.run(exe, np.array(0.0, "float32"))
    assert out.numpy() == 4.0


# ---------------------------------------------------------------------------
# 5. tvm (real TVM — version-adaptive pass_configs)
# ---------------------------------------------------------------------------


def _require_tvm():
    pytest.importorskip("tvm")
    pytest.importorskip("jaxlib")


def test_tvm_target_accepted():
    """``tvm_target`` (default "llvm") is accepted and forwarded to the Relax
    VM build; the graph compiles and runs."""
    _require_tvm()
    exe = etl.build(_add_fn, _ADD_SPEC, backend="tvm", tvm_target="llvm")
    np.testing.assert_allclose(etl.run(exe, _ADD_INPUT).numpy(), _ADD_EXPECTED)


def test_tvm_pass_configs_version_adaptive():
    """``tvm_pass_configs`` is forwarded when the installed TVM build accepts
    the parameter; otherwise a loud BackendError naming the option — never
    silently dropped."""
    _require_tvm()
    lowered = etl.lower(etl.trace(_add_fn, _ADD_SPEC), backend="tvm")
    try:
        artifact = etl.compile(
            lowered, tvm_pass_configs={"tir.disable_assert": False}
        )
    except core.BackendError as exc:
        # TVM 0.26's relax.vm_build.build does not accept the keyword.
        assert "pass_configs" in str(exc)
        assert "tvm_pass_configs" in str(exc)
    else:
        exe = etl.load(artifact)
        np.testing.assert_allclose(etl.run(exe, _ADD_INPUT).numpy(), _ADD_EXPECTED)


def test_tvm_unknown_compile_option_rejected():
    _require_tvm()
    lowered = etl.lower(etl.trace(_add_fn, _ADD_SPEC), backend="tvm")
    with pytest.raises(core.BackendError, match="does not recognize the compile option"):
        etl.compile(lowered, bogus=1)


# ---------------------------------------------------------------------------
# 6. xla (fake PJRT plugin — no real XLA build needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_pjrt_plugin(tmp_path_factory):
    """The fake PJRT plugin (same build as test_pjrt_ctypes_plugin.py): ABI-
    valid, ZERO-FILLED outputs — shape/dtype/plumbing are asserted, numerical
    parity is out of scope (the fake performs no computation)."""
    return _build_plugin(
        tmp_path_factory.mktemp("fake_pjrt_plugin"), "fake_pjrt_plugin.so"
    )


def _xla_lowered():
    return etl.lower(etl.trace(_add_fn, _ADD_SPEC), backend="xla")


def test_xla_plugin_discovery_requires_env_or_option(monkeypatch, fake_pjrt_plugin):
    """No plugin configured -> BackendError naming ETL_PJRT_PLUGIN; the
    ``plugin_path`` option is honored for discovery."""
    monkeypatch.delenv("ETL_PJRT_PLUGIN", raising=False)
    monkeypatch.setattr(xla_util, "_DEFAULT_PLUGIN_PATHS", ())
    with pytest.raises(core.BackendError, match="ETL_PJRT_PLUGIN"):
        xla_util._find_plugin_path(None)
    plugin = xla_util._load_plugin({"plugin_path": str(fake_pjrt_plugin)})
    assert plugin.api is not None


def test_xla_compile_load_options_end_to_end(monkeypatch, fake_pjrt_plugin):
    """plugin_path at compile AND load + xla_compile_options bytes: the full
    pipeline runs against the fake (zero-filled outputs — the declared
    shape/dtype plumbing is what the fake exercises)."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_pjrt_plugin))
    lowered = _xla_lowered()
    artifact = etl.compile(
        lowered,
        plugin_path=str(fake_pjrt_plugin),
        xla_compile_options=b"\x1a\x04\x20\x01\x28\x01",  # a CompileOptionsProto
    )
    exe = etl.load(artifact, plugin_path=str(fake_pjrt_plugin))
    out = etl.run(exe, _ADD_INPUT)
    assert out.shape == (4,)
    assert out.dtype == np.dtype("float32")
    np.testing.assert_array_equal(out.numpy(), np.zeros(4, "float32"))


def test_xla_compile_options_non_bytes_rejected(monkeypatch, fake_pjrt_plugin):
    """``xla_compile_options`` must be bytes (a serialized
    CompileOptionsProto) — rejected before any plugin work."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_pjrt_plugin))
    lowered = _xla_lowered()
    with pytest.raises(core.BackendError, match="xla_compile_options"):
        etl.compile(lowered, xla_compile_options="not-bytes")


def test_xla_run_stage_unknown_option_rejected(monkeypatch, fake_pjrt_plugin):
    """The xla run stage has NO known options in v1: a non-empty run options
    dict raises core.BackendError."""
    monkeypatch.setenv("ETL_PJRT_PLUGIN", str(fake_pjrt_plugin))
    lowered = _xla_lowered()
    artifact = etl.compile(lowered, plugin_path=str(fake_pjrt_plugin))
    exe = etl.load(artifact, plugin_path=str(fake_pjrt_plugin))
    with pytest.raises(core.BackendError, match="does not recognize the run option"):
        etl.run(exe, _ADD_INPUT, bogus=1)


# ---------------------------------------------------------------------------
# 7. numpy documented-ignore
# ---------------------------------------------------------------------------


def test_numpy_ignores_unknown_options():
    """The numpy reference backend DELIBERATELY declares no validation:
    cross-backend option dicts pass at every stage without error."""
    exe = etl.build(
        _add_fn, _ADD_SPEC, backend="numpy",
        target_backends=["llvm-cpu"], bogus_option=1,
    )
    np.testing.assert_allclose(
        etl.run(exe, _ADD_INPUT, target_backends=["cuda"], another_bogus=2).numpy(),
        _ADD_EXPECTED,
    )


def test_numpy_rank_context_flows_through_run_options():
    """The documented numpy exception: ``rank_context`` inside run options
    still flows into the interpreter's execution context."""
    exe = etl.build(_add_fn, _ADD_SPEC, backend="numpy")
    np.testing.assert_allclose(
        etl.run(exe, _ADD_INPUT, rank_context=etl.dist.RankContext(0, 1)).numpy(),
        _ADD_EXPECTED,
    )
