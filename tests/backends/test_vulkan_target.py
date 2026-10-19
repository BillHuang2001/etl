"""Unit tests for the FIRST-CLASS ``vulkan`` device kind's iree surface: the
``vulkan_target`` compile option, the ``vulkan`` target LABEL and the vulkan
LOAD GATE.

Modelled on ``tests/backends/test_opt_level.py``: pure unit tests — NO real
compiles, NO GPU, and they RUN in a compiler-free (numpy-only) environment.
Where an iree runtime entry point is unavoidable (``IreeBackend.load`` calls
``check_available`` and, past the gate, the HAL device acquisition) a COMPLETE
fake ``iree`` package is installed through ``monkeypatch.setitem`` — the same
compiler-free mechanism as ``tests/backends/test_options_override.py::
_install_fake_iree_runtime`` — so the gate arithmetic is pinned deterministically
with or without IREE installed.

Covered (see ``etl/backends/adapters/iree.py``):

1. ``_resolve_iree_compile_args`` — the derivation matrix for
   ``vulkan_target``: UNSET emits no ``--iree-vulkan-target`` flag at all
   (byte-identical to the pre-option behavior, where IREE's compiler default
   profile applies), a profile emits exactly one flag, and a user flag of the
   same NAME wins (the derived flag is dropped — never both, the documented
   conflict rule shared with ``opt_level``). Interactions with
   ``opt_level_args``, the ``llvm-cpu``/``rocm`` target-specific defaults and
   the deny list stay unchanged.
2. ``IreeBackend._resolve_compile_options`` — the ``vulkan_target`` VALUE
   validation (non-str / empty => loud ``BackendError`` naming the option, the
   accepted form and the offending value) and the "only applies to
   vulkan-spirv compilation" rule.
3. ``KNOWN_OPTIONS`` — ``vulkan_target`` is declared at the compile stage and
   reachable through the pipeline's option validation (accepted-and-ignored at
   ``lower``), while an unknown key is still rejected at every stage (the
   asymmetric pin).
4. ``_artifact_target`` — the ``"vulkan"`` label plus the pre-existing
   single-target labels and the mixed ``"+".join(...)`` form.
5. ``_expected_target_for_kind`` / ``_EXPECTED_TARGET_BY_KIND`` — ``vulkan``
   => ``vulkan-spirv`` and the other kinds unchanged; unknown kind =>
   ``KeyError``. ``_GPU_DEVICE_KINDS`` carries ``vulkan`` (the device-resident
   branch gate).
6. ``IreeBackend.load``'s artifact/device target compatibility gate for the
   vulkan device, on BOTH load paths (single-program and split segments).

Staging (``_staged_input``) is a separate concern and is NOT tested here.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest

import etl
from etl.core import BackendError, Device
from etl.backends import CompiledArtifact, external_split
from etl.backends import registry as backends_registry
from etl.backends.adapters import iree as iree_adapt
from etl.backends.options import validate_options
from etl.pipeline_options import ENV_OPTION_TABLE, apply_env_options

# ---------------------------------------------------------------------------
# Shared constants + the tiny graph used by the public-path tests
# ---------------------------------------------------------------------------

_OPTION = "vulkan_target"
_TARGET = "vulkan-spirv"
_FLAG_NAME = "--iree-vulkan-target"
_ENV_VAR = "ETL_IREE_VULKAN_TARGET"

_TINY_SPEC = etl.TensorSpec((2, 2), etl.float32)

#: Every env var that could leak into an option dict during these tests (the
#: per-stage env half of ``etl.pipeline_options`` + the staging-level vars).
_ENV_VARS = (
    _ENV_VAR,
    "ETL_IREE_COMPILE_ARGS",
    "ETL_IREE_RUNTIME_ARGS",
    "ETL_OPT_LEVEL",
    "ETL_EIGH_ALGORITHM",
    "ETL_SPLIT_REDUCTION",
    "ETL_BACKEND",
    "ETL_DEVICE",
    "ETL_TARGET_BACKENDS",
)


@pytest.fixture(autouse=True)
def _clean_option_env(monkeypatch):
    """No ambient ``ETL_*`` option/environment leaks into these assertions."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _tiny_fn(x):
    return etl.add(etl.multiply(x, x), 1.0)


def _vulkan_flags(flags):
    """All flags whose NAME is ``--iree-vulkan-target`` (name-based counting,
    mirroring the adapter's own override machinery — values never matter)."""
    return [flag for flag in flags if flag.partition("=")[0] == _FLAG_NAME]


def _derived(profile):
    return f"{_FLAG_NAME}={profile}"


# ---------------------------------------------------------------------------
# 1. _resolve_iree_compile_args — the derivation matrix (pure, no IREE)
# ---------------------------------------------------------------------------


def test_resolve_compile_args_vulkan_unset_emits_no_flag():
    """UNSET ``vulkan_target`` => NO ``--iree-vulkan-target`` flag at all.

    Byte-identical to the pre-option argument set: IREE's compiler default
    profile applies, exactly as before the option existed.
    """
    out = iree_adapt._resolve_iree_compile_args((_TARGET,), None)
    assert _vulkan_flags(out) == []
    assert out == list(iree_adapt._IREE_DEFAULT_COMPILE_ARGS)


def test_resolve_compile_args_vulkan_derived_flag_exactly_once():
    out = iree_adapt._resolve_iree_compile_args(
        (_TARGET,), None, vulkan_target_args=(_derived("rdna3"),)
    )
    assert _vulkan_flags(out) == [_derived("rdna3")]
    # The other etl defaults are untouched by the derived flag.
    for default in iree_adapt._IREE_DEFAULT_COMPILE_ARGS:
        assert default in out


def test_resolve_compile_args_vulkan_user_flag_wins():
    """A user flag of the same NAME replaces the derived one — never both."""
    out = iree_adapt._resolve_iree_compile_args(
        (_TARGET,),
        [_derived("gfx1100")],
        vulkan_target_args=(_derived("rdna3"),),
    )
    assert _vulkan_flags(out) == [_derived("gfx1100")]


def test_resolve_compile_args_vulkan_user_flag_without_derived():
    """A user ``--iree-vulkan-target`` with the option UNSET is passed through
    verbatim (the derived tuple is empty, nothing to collide with)."""
    out = iree_adapt._resolve_iree_compile_args((_TARGET,), [_derived("gfx1100")])
    assert _vulkan_flags(out) == [_derived("gfx1100")]


def test_resolve_compile_args_vulkan_coexists_with_opt_level():
    out = iree_adapt._resolve_iree_compile_args(
        (_TARGET,),
        None,
        opt_level_args=("--iree-opt-level=O2",),
        vulkan_target_args=(_derived("rdna3"),),
    )
    assert _vulkan_flags(out) == [_derived("rdna3")]
    assert [f for f in out if f.partition("=")[0] == "--iree-opt-level"] == [
        "--iree-opt-level=O2"
    ]


def test_resolve_compile_args_llvm_cpu_defaults_unchanged():
    """The ``llvm-cpu`` target-specific defaults are unrelated to (and
    unaffected by) the vulkan derived flag; a vulkan-only target carries NONE
    of them."""
    base = iree_adapt._resolve_iree_compile_args(("llvm-cpu",), None)
    for default in iree_adapt._IREE_DEFAULT_LLVMCPU_ARGS:
        assert default in base

    out = iree_adapt._resolve_iree_compile_args(
        ("llvm-cpu",),
        None,
        opt_level_args=("--iree-opt-level=O3",),
        vulkan_target_args=(_derived("rdna3"),),
    )
    llvmcpu = {_FLAG_NAME, "--iree-opt-level"}
    assert [f for f in out if f.partition("=")[0] not in llvmcpu] == list(base)
    assert _vulkan_flags(out) == [_derived("rdna3")]

    vulkan_only = iree_adapt._resolve_iree_compile_args(
        (_TARGET,), None, vulkan_target_args=(_derived("rdna3"),)
    )
    for default in iree_adapt._IREE_DEFAULT_LLVMCPU_ARGS:
        assert default not in vulkan_only


def test_resolve_compile_args_rocm_default_unchanged(monkeypatch):
    """The detected ``--iree-rocm-target`` default still joins the defaults —
    the vulkan mechanism is additive and target-scoped."""
    monkeypatch.setattr(
        iree_adapt,
        "_rocm_target_default",
        lambda user_args: ("--iree-rocm-target=gfx942",),
    )
    out = iree_adapt._resolve_iree_compile_args(("rocm",), None)
    assert "--iree-rocm-target=gfx942" in out
    assert _vulkan_flags(out) == []

    out = iree_adapt._resolve_iree_compile_args(
        ("rocm",), None, vulkan_target_args=(_derived("rdna3"),)
    )
    assert "--iree-rocm-target=gfx942" in out
    assert _vulkan_flags(out) == [_derived("rdna3")]


def test_resolve_compile_args_deny_list_unchanged():
    """The documented deny list still rejects genuinely required flags — the
    derived vulkan flag does not bypass it."""
    with pytest.raises(BackendError) as excinfo:
        iree_adapt._resolve_iree_compile_args(
            (_TARGET,),
            ["--iree-hal-target-backends=vulkan-spirv"],
            vulkan_target_args=(_derived("rdna3"),),
        )
    message = str(excinfo.value)
    assert "denies the iree-compile flag" in message
    assert "use the 'target_backends' compile option instead" in message


# ---------------------------------------------------------------------------
# 2. IreeBackend._resolve_compile_options — vulkan_target value validation
# ---------------------------------------------------------------------------


def test_resolve_compile_options_derives_vulkan_target_flag():
    targets, extra = iree_adapt.iree_backend._resolve_compile_options(
        {"target_backends": [_TARGET], _OPTION: "rdna3"}
    )
    assert targets == (_TARGET,)
    assert _vulkan_flags(extra) == [_derived("rdna3")]


def test_resolve_compile_options_vulkan_target_unset_default():
    targets, extra = iree_adapt.iree_backend._resolve_compile_options(
        {"target_backends": [_TARGET]}
    )
    assert targets == (_TARGET,)
    assert _vulkan_flags(extra) == []
    # Byte-identical to the pre-option resolution.
    assert extra == iree_adapt._resolve_iree_compile_args((_TARGET,), [])


def test_resolve_compile_options_vulkan_target_user_flag_wins():
    _, extra = iree_adapt.iree_backend._resolve_compile_options(
        {
            "target_backends": [_TARGET],
            _OPTION: "rdna3",
            "iree_compile_args": [_derived("gfx1100")],
        }
    )
    assert _vulkan_flags(extra) == [_derived("gfx1100")]


@pytest.mark.parametrize(
    "bad",
    [
        5,
        0,
        5.0,
        True,
        False,
        b"rdna3",
        ["rdna3"],
        ("rdna3",),
        {"profile": "rdna3"},
        "",
    ],
)
def test_resolve_compile_options_vulkan_target_bad_value(bad):
    """Non-str / empty values raise a loud ``core.BackendError`` naming the
    option, the accepted form and the offending value — never a silent
    fallback, never an obscure AttributeError."""
    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend._resolve_compile_options(
            {"target_backends": [_TARGET], _OPTION: bad}
        )
    message = str(excinfo.value)
    assert "'vulkan_target'" in message
    assert "non-empty string" in message
    assert _FLAG_NAME in message  # the accepted form is spelled out
    assert repr(bad) in message


@pytest.mark.parametrize(
    "opts",
    [
        # the DEFAULT target set (unset => llvm-cpu)
        {},
        {"target_backends": ["llvm-cpu"]},
        {"target_backends": ["cuda"]},
        {"target_backends": ["llvm-cpu", "cuda"]},
    ],
)
def test_resolve_compile_options_vulkan_target_requires_vulkan_spirv(opts):
    """Setting ``vulkan_target`` without requesting ``vulkan-spirv`` is a loud
    ``core.BackendError`` — an explicit option is never silently ignored."""
    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend._resolve_compile_options({**opts, _OPTION: "rdna3"})
    message = str(excinfo.value)
    assert "'vulkan_target'" in message
    assert "'rdna3'" in message
    assert _TARGET in message
    assert "only applies to vulkan-spirv compilation" in message


def test_resolve_compile_options_vulkan_target_accepted_alongside_other_targets():
    """``vulkan-spirv`` among a MIXED target tuple is enough for the option."""
    targets, extra = iree_adapt.iree_backend._resolve_compile_options(
        {"target_backends": ["llvm-cpu", _TARGET], _OPTION: "rdna3"}
    )
    assert targets == ("llvm-cpu", _TARGET)
    assert _vulkan_flags(extra) == [_derived("rdna3")]


# ---------------------------------------------------------------------------
# 3. KNOWN_OPTIONS declaration + the pipeline option-validation asymmetry
# ---------------------------------------------------------------------------


def test_known_options_declares_vulkan_target_at_compile_only():
    known = iree_adapt.IreeBackend.KNOWN_OPTIONS
    assert _OPTION in known["compile"]
    for stage in ("lower", "load", "run"):
        assert _OPTION not in known[stage]


@pytest.mark.parametrize("stage", ["compile", "lower", "load", "run"])
def test_known_options_union_accepts_vulkan_target(stage):
    """A compile-stage key passes validation at EVERY stage (accepted and
    ignored elsewhere — the union semantics the build/evaluate sugar relies
    on)."""
    validate_options(
        {_OPTION: "rdna3"}, iree_adapt.IreeBackend.KNOWN_OPTIONS, "iree", stage
    )


@pytest.mark.parametrize("stage", ["compile", "lower", "load"])
@pytest.mark.parametrize("unknown", ["vulkan_targets", "vulkan_profile", "target"])
def test_known_options_unknown_key_still_rejected(stage, unknown):
    """The asymmetric half: a near-miss / unknown key is rejected loudly at
    every stage (the message lists the per-stage known sets, incl.
    ``vulkan_target``)."""
    with pytest.raises(BackendError) as excinfo:
        validate_options(
            {_OPTION: "rdna3", unknown: "x"},
            iree_adapt.IreeBackend.KNOWN_OPTIONS,
            "iree",
            stage,
        )
    message = str(excinfo.value)
    assert repr(unknown) in message
    assert "vulkan_target" in message  # the known-set listing names the option


# ---------------------------------------------------------------------------
# 4. The public path — etl.lower / etl.compile / etl.build option validation
# ---------------------------------------------------------------------------


@pytest.fixture
def iree_registered():
    """Register the REAL iree adapter singleton in the backend registry.

    ``etl.lower``/``etl.compile`` resolve backends through the registry; the
    normal ``registry.get("iree")`` activation path calls
    ``check_available()`` (which needs the IREE packages). These tests target
    the PIPELINE's option plumbing — ``lower`` never touches IREE (it only
    exports StableHLO) and ``compile`` is driven by the fake compiler below —
    so the singleton is inserted directly (the ``tests/sparse/test_backends.py``
    stub pattern) and the previous registry entry is restored afterwards.
    """
    present = "iree" in backends_registry._registry
    previous = backends_registry._registry.get("iree")
    backends_registry._registry["iree"] = iree_adapt.iree_backend
    try:
        yield iree_adapt.iree_backend
    finally:
        if present:
            backends_registry._registry["iree"] = previous
        else:
            backends_registry._registry.pop("iree", None)


class _FakeRuntimeReached(Exception):
    """Sentinel: the load path got PAST the target gate into the runtime."""


class _FakeHalDevice:
    def __init__(self, driver_name, info):
        self.driver_name = driver_name
        self.info = info


class _FakeDriver:
    """Minimal iree HAL driver: one enumerated (fake) device."""

    def __init__(self, name):
        self.name = name
        self.create_device_calls = []

    def query_available_devices(self):
        return [{"name": "etl-fake-vulkan", "device_id": 4242, "path": "/fake/0"}]

    def create_device(self, info):
        self.create_device_calls.append(dict(info))
        return _FakeHalDevice(self.name, dict(info))


def _install_fake_iree(monkeypatch):
    """Install a COMPLETE fake ``iree`` package (compiler recorder + a runtime
    whose ``Config`` — the first entry point PAST the target gate — raises
    ``_FakeRuntimeReached``).

    Two layers are needed exactly as in
    ``tests/backends/test_options_override.py::_install_fake_iree_runtime``:
    ``import iree.runtime as rt`` first imports the PARENT package and then
    resolves the ``runtime`` attribute (falling back to ``sys.modules``). When
    a REAL ``iree`` is installed its attributes are patched and restored by
    ``monkeypatch``; when it is absent a fake package (``__path__ = []``) is
    installed. Returns the recorder with the ``compile_str`` calls + the fake
    drivers.
    """
    compiled: list[dict] = []
    drivers: dict[str, _FakeDriver] = {}

    def _compile_str(text, target_backends=None, input_type=None, extra_args=None):
        compiled.append(
            {
                "mlir_text": text,
                "target_backends": tuple(target_backends or ()),
                "input_type": input_type,
                "extra_args": tuple(extra_args or ()),
            }
        )
        return b"ETL-FAKE-VMFB"

    compiler = types.ModuleType("iree.compiler")
    compiler.compile_str = _compile_str

    class _CompilerToolError(Exception):
        pass

    compiler.CompilerToolError = _CompilerToolError

    runtime = types.ModuleType("iree.runtime")
    flags = types.ModuleType("iree.runtime.flags")
    flags.parse_flags = lambda *args: None
    runtime.flags = flags

    def _get_driver(name):
        driver = drivers.get(name)
        if driver is None:
            driver = _FakeDriver(name)
            drivers[name] = driver
        return driver

    def _config(*args, **kwargs):
        # The FIRST runtime call on both load paths (``_load_vm_module`` for a
        # GPU-class kind) — reaching it proves the target gate did NOT fire.
        raise _FakeRuntimeReached("Config")

    def _load_vm_module(*args, **kwargs):
        raise _FakeRuntimeReached("load_vm_module")

    runtime.get_driver = _get_driver
    runtime.Config = _config
    runtime.load_vm_module = _load_vm_module

    iree_pkg = sys.modules.get("iree")
    if iree_pkg is None:
        iree_pkg = types.ModuleType("iree")
        iree_pkg.__path__ = []  # a package, so ``import iree.<sub>`` works
        monkeypatch.setitem(sys.modules, "iree", iree_pkg)
    monkeypatch.setattr(iree_pkg, "compiler", compiler, raising=False)
    monkeypatch.setattr(iree_pkg, "runtime", runtime, raising=False)
    monkeypatch.setitem(sys.modules, "iree.compiler", compiler)
    monkeypatch.setitem(sys.modules, "iree.runtime", runtime)
    monkeypatch.setitem(sys.modules, "iree.runtime.flags", flags)
    return {"compiled": compiled, "drivers": drivers}


def test_lower_accepts_vulkan_target_option(iree_registered):
    """``etl.lower`` validates option KEYS only: the compile-stage
    ``vulkan_target`` is accepted (and ignored) at the lower stage — lower
    never touches IREE, so this runs compiler-free."""
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    lowered = etl.lower(
        graph,
        backend="iree",
        target_backends=[_TARGET],
        vulkan_target="rdna3",
    )
    assert lowered.backend == "iree"


def test_lower_rejects_unknown_option_alongside_vulkan_target(iree_registered):
    """The asymmetric pin at the public entry point: a key valid at NO stage is
    still rejected, and the message lists ``vulkan_target`` as known."""
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    with pytest.raises(BackendError) as excinfo:
        etl.lower(
            graph,
            backend="iree",
            target_backends=[_TARGET],
            vulkan_target="rdna3",
            vulkan_profile="rdna3",
        )
    message = str(excinfo.value)
    assert "'vulkan_profile'" in message
    assert "vulkan_target" in message


def test_compile_derives_vulkan_target_flag(monkeypatch, iree_registered):
    """``etl.compile`` forwards ``vulkan_target`` to iree-compile as the
    derived flag AND records the ``"vulkan"`` target label on the artifact
    (the fake compiler stands in for iree-compile — no real compilation)."""
    recorder = _install_fake_iree(monkeypatch)
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    lowered = etl.lower(graph, backend="iree", target_backends=[_TARGET])

    artifact = etl.compile(
        lowered, target_backends=[_TARGET], vulkan_target="rdna3"
    )

    assert len(recorder["compiled"]) == 1
    call = recorder["compiled"][0]
    assert call["target_backends"] == (_TARGET,)
    assert call["input_type"] == "stablehlo"
    assert _vulkan_flags(call["extra_args"]) == [_derived("rdna3")]
    assert artifact.backend == "iree"
    assert artifact.target == "vulkan"
    assert artifact.payload["target_backends"] == (_TARGET,)
    assert _OPTION not in artifact.payload  # the option never leaks into the payload


def test_compile_vulkan_target_without_vulkan_spirv_is_loud(
    monkeypatch, iree_registered
):
    recorder = _install_fake_iree(monkeypatch)
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    lowered = etl.lower(graph, backend="iree", target_backends=["llvm-cpu"])

    with pytest.raises(BackendError) as excinfo:
        etl.compile(lowered, target_backends=["llvm-cpu"], vulkan_target="rdna3")
    message = str(excinfo.value)
    assert "'vulkan_target'" in message
    assert "only applies to vulkan-spirv compilation" in message
    assert "llvm-cpu" in message  # names the requested targets
    assert recorder["compiled"] == []  # never reached the compiler


def test_compile_vulkan_target_bad_value_is_loud(monkeypatch, iree_registered):
    recorder = _install_fake_iree(monkeypatch)
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    lowered = etl.lower(graph, backend="iree", target_backends=[_TARGET])

    with pytest.raises(BackendError) as excinfo:
        etl.compile(lowered, target_backends=[_TARGET], vulkan_target=5)
    message = str(excinfo.value)
    assert "'vulkan_target'" in message
    assert "non-empty string" in message
    assert "5" in message
    assert recorder["compiled"] == []


def test_build_forwards_vulkan_target_to_compile(monkeypatch, iree_registered):
    """``etl.build`` forwards the option to the compile stage (the loud
    wrong-target error fires at compile time, before any load)."""
    recorder = _install_fake_iree(monkeypatch)
    with pytest.raises(BackendError) as excinfo:
        etl.build(
            _tiny_fn,
            _TINY_SPEC,
            backend="iree",
            target_backends=["llvm-cpu"],
            vulkan_target="rdna3",
        )
    assert "only applies to vulkan-spirv compilation" in str(excinfo.value)
    assert recorder["compiled"] == []


def test_compile_env_vulkan_target_reaches_compile(monkeypatch, iree_registered):
    """The env half is part of the same contract: ``ETL_IREE_VULKAN_TARGET``
    reaches compile when not passed explicitly (explicit kwarg wins)."""
    recorder = _install_fake_iree(monkeypatch)
    graph = etl.trace(_tiny_fn, _TINY_SPEC)
    lowered = etl.lower(graph, backend="iree", target_backends=[_TARGET])

    monkeypatch.setenv(_ENV_VAR, "  rdna3  ")
    etl.compile(lowered, target_backends=[_TARGET])
    assert _vulkan_flags(recorder["compiled"][-1]["extra_args"]) == [
        _derived("rdna3")
    ]

    etl.compile(lowered, target_backends=[_TARGET], vulkan_target="gfx1100")
    assert _vulkan_flags(recorder["compiled"][-1]["extra_args"]) == [
        _derived("gfx1100")
    ]


# ---------------------------------------------------------------------------
# 5. The env-option row (etl.pipeline_options.ETL_IREE_VULKAN_TARGET)
# ---------------------------------------------------------------------------


def test_env_row_declared_for_iree_compile():
    entries = ENV_OPTION_TABLE[("iree", "compile")]
    rows = {env_var: option_key for env_var, option_key, _ in entries}
    assert rows[_ENV_VAR] == _OPTION
    # The row is compile-scoped for iree (no other stage / backend carries it).
    for key, table_entries in ENV_OPTION_TABLE.items():
        if key == ("iree", "compile"):
            continue
        assert all(env_var != _ENV_VAR for env_var, _, _ in table_entries), key


def test_env_row_strips_whitespace(monkeypatch):
    assert apply_env_options("iree", {}, "compile") == {}  # unset => nothing

    monkeypatch.setenv(_ENV_VAR, "  rdna3 \n")
    assert apply_env_options("iree", {}, "compile") == {_OPTION: "rdna3"}
    # Explicit kwarg wins over the env var.
    assert apply_env_options("iree", {_OPTION: "gfx1100"}, "compile") == {
        _OPTION: "gfx1100"
    }
    # A blank env value is UNSET (documented) — never an empty profile.
    monkeypatch.setenv(_ENV_VAR, "   ")
    assert apply_env_options("iree", {}, "compile") == {}
    # The numpy backend has no row: the env var is a no-op there.
    monkeypatch.setenv(_ENV_VAR, "rdna3")
    assert apply_env_options("numpy", {}, "compile") == {}


def test_env_row_feeds_the_adapter_derivation(monkeypatch):
    """env -> options -> ``_resolve_compile_options`` -> derived flag."""
    monkeypatch.setenv(_ENV_VAR, "rdna3")
    options = apply_env_options(
        "iree", {"target_backends": [_TARGET]}, "compile"
    )
    _, extra = iree_adapt.iree_backend._resolve_compile_options(options)
    assert _vulkan_flags(extra) == [_derived("rdna3")]


# ---------------------------------------------------------------------------
# 6. _artifact_target — the vulkan label + the pre-existing labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("targets", "label"),
    [
        ((_TARGET,), "vulkan"),
        (("llvm-cpu",), "cpu"),
        (("cuda",), "cuda"),
        (("rocm",), "rocm"),
        # MIXED tuples fall through to the "+".join form.
        (("llvm-cpu", _TARGET), f"llvm-cpu+{_TARGET}"),
        ((_TARGET, "llvm-cpu"), f"{_TARGET}+llvm-cpu"),
        (("cuda", "rocm"), "cuda+rocm"),
    ],
)
def test_artifact_target_labels(targets, label):
    assert iree_adapt._artifact_target(targets) == label


# ---------------------------------------------------------------------------
# 7. _expected_target_for_kind / the kind tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        ("cpu", "llvm-cpu"),
        ("cuda", "cuda"),
        ("rocm", "rocm"),
        ("vulkan", _TARGET),
    ],
)
def test_expected_target_for_kind_table(kind, target):
    assert iree_adapt._expected_target_for_kind(kind) == target
    assert iree_adapt._EXPECTED_TARGET_BY_KIND[kind] == target


def test_expected_target_by_kind_is_the_documented_table():
    assert iree_adapt._EXPECTED_TARGET_BY_KIND == {
        "cpu": "llvm-cpu",
        "cuda": "cuda",
        "rocm": "rocm",
        "vulkan": _TARGET,
    }


@pytest.mark.parametrize("unknown", ["tpu", "vulkan-spirv", "", "hip"])
def test_expected_target_for_kind_unknown_raises_keyerror(unknown):
    """The documented behaviour: an unnormalized/unknown kind raises the plain
    ``KeyError`` (the public entry points validate the kind first)."""
    with pytest.raises(KeyError):
        iree_adapt._expected_target_for_kind(unknown)


def test_vulkan_kind_normalizes_and_dispatches_unchanged():
    assert iree_adapt._normalize_device_kind("vulkan") == "vulkan"
    assert iree_adapt._runtime_driver_name("vulkan") == "vulkan"
    # The hip alias still normalizes to the rocm path / iree "hip" driver.
    assert iree_adapt._normalize_device_kind("hip") == "rocm"
    assert iree_adapt._runtime_driver_name("rocm") == "hip"


def test_vulkan_is_a_gpu_device_kind():
    """``_GPU_DEVICE_KINDS`` is the device-resident branch gate (the
    ``rt.Config(device=...)`` module-load path + the device-resident run
    branch). ``_staged_input``'s CUDA-only tier stays a separate concern."""
    assert "vulkan" in iree_adapt._GPU_DEVICE_KINDS
    assert {"cuda", "rocm"} <= set(iree_adapt._GPU_DEVICE_KINDS)
    assert "cpu" not in iree_adapt._GPU_DEVICE_KINDS
    assert "hip" not in iree_adapt._GPU_DEVICE_KINDS  # normalized before use


# ---------------------------------------------------------------------------
# 8. The LOAD GATE — artifact targets must satisfy the device's target
# ---------------------------------------------------------------------------


def _clear_hal_device_cache():
    """Drop any thread-local cached HAL device (another suite may have cached
    a real one; the fakes below must be the ones exercised)."""
    cache = getattr(iree_adapt._HAL_DEVICE_CACHE, "devices", None)
    if cache is not None:
        cache.clear()


@pytest.fixture
def _isolated_hal_device_cache():
    """No cached HAL device before OR after the test: the fake devices these
    tests create must neither be shadowed by a real cached one nor leak into
    another suite of the same process."""
    _clear_hal_device_cache()
    try:
        yield
    finally:
        _clear_hal_device_cache()


def _artifact(target_backends, entry_functions=("main",)):
    return CompiledArtifact(
        backend="iree",
        signature=None,
        target=iree_adapt._artifact_target(tuple(target_backends)),
        payload={
            "format": "iree-vmfb",
            "format_version": 1,
            "mlir_text": "func.func @main() {}",
            "vmfb_base64": base64.b64encode(b"ETL-FAKE-VMFB").decode("ascii"),
            "entry_functions": tuple(entry_functions),
            "target_backends": tuple(target_backends),
        },
    )


def _segments_artifact(segment_targets, plan=None):
    return CompiledArtifact(
        backend="iree",
        signature=None,
        target=iree_adapt._artifact_target(tuple(segment_targets)),
        payload={
            "format": "iree-vmfb-segments",
            "format_version": 1,
            "segments": [
                {
                    "vmfb_base64": base64.b64encode(b"ETL-FAKE-VMFB").decode(
                        "ascii"
                    ),
                    "entry_functions": ("main",),
                    "target_backends": tuple(segment_targets),
                }
            ],
            "plan": {} if plan is None else plan,
        },
    )


def test_load_gate_vulkan_device_requires_vulkan_spirv(
    monkeypatch, _isolated_hal_device_cache
):
    """A vulkan-device load of an artifact compiled for another target fails
    loudly, naming the expected target AND the artifact's targets — never a
    silent recompile."""
    _install_fake_iree(monkeypatch)

    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend.load(
            _artifact(("llvm-cpu",)), device=Device("vulkan", 0)
        )
    message = str(excinfo.value)
    assert _TARGET in message  # the target this device kind requires
    assert "llvm-cpu" in message  # what the artifact was compiled for
    assert "device kind 'vulkan'" in message
    assert "never silently recompile" in message


def test_load_gate_cpu_device_rejects_vulkan_spirv_artifact(
    monkeypatch, _isolated_hal_device_cache
):
    """The same gate in the other direction: a vulkan-spirv artifact cannot
    run on the default (cpu) device — it requires ``llvm-cpu``."""
    _install_fake_iree(monkeypatch)

    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend.load(_artifact((_TARGET,)), device=None)
    message = str(excinfo.value)
    assert "llvm-cpu" in message
    assert _TARGET in message
    assert "device kind 'cpu'" in message


def test_load_gate_vulkan_matching_target_passes_the_gate(
    monkeypatch, _isolated_hal_device_cache
):
    """A matching-target artifact is NOT rejected by the gate: the load path
    proceeds into the runtime (the fake's ``Config`` sentinel) instead."""
    recorder = _install_fake_iree(monkeypatch)

    with pytest.raises(_FakeRuntimeReached) as excinfo:
        iree_adapt.iree_backend.load(
            _artifact((_TARGET,)), device=Device("vulkan", 0)
        )
    assert excinfo.value.args == ("Config",)
    # ... and it acquired the device through iree's "vulkan" driver.
    assert "vulkan" in recorder["drivers"]
    assert recorder["drivers"]["vulkan"].create_device_calls == [
        {"name": "etl-fake-vulkan", "device_id": 4242, "path": "/fake/0"}
    ]


def test_segments_load_gate_vulkan_device_requires_vulkan_spirv(
    monkeypatch, _isolated_hal_device_cache
):
    """The SPLIT (segments) load path consumes the SAME table
    (``_expected_target_for_kind``) per segment."""
    _install_fake_iree(monkeypatch)
    monkeypatch.setattr(external_split, "decode_plan", lambda plan: [{}])

    with pytest.raises(BackendError) as excinfo:
        iree_adapt.iree_backend.load(
            _segments_artifact(("llvm-cpu",)), device=Device("vulkan", 0)
        )
    message = str(excinfo.value)
    assert "split artifact segment 0" in message
    assert "llvm-cpu" in message
    assert _TARGET in message
    assert "device kind 'vulkan'" in message
    assert "never silently recompile" in message


def test_segments_load_gate_vulkan_matching_target_passes_the_gate(
    monkeypatch, _isolated_hal_device_cache
):
    _install_fake_iree(monkeypatch)
    monkeypatch.setattr(external_split, "decode_plan", lambda plan: [{}])

    with pytest.raises(_FakeRuntimeReached) as excinfo:
        iree_adapt.iree_backend.load(
            _segments_artifact((_TARGET,)), device=Device("vulkan", 0)
        )
    assert excinfo.value.args == ("Config",)
