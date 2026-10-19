"""etl.bench backend-options resolution contract (``_util.resolve_backend_options``).

The harness resolves the backend options dict passed to ``etl.build`` (and to
every ``Example.runner`` factory):

- ``target_backends`` is an **iree-only** lower option (the xla, tvm, and numpy
  backends reject it — it is declared in no stage of theirs), so the
  device-derived default is injected ONLY for the iree backend family, gated on
  the ``backend.startswith("iree")`` name convention — ``["cuda"]`` for a cuda
  device, ``["rocm"]`` for a rocm (or its ``"hip"`` alias) device,
  ``["vulkan-spirv"]`` for a vulkan device (IREE's Vulkan/SPIR-V compile
  target), ``["llvm-cpu"]`` otherwise. xla, tvm, and numpy receive NO
  ``target_backends``.
- the ``opt_level`` harness default ``"O3"`` is injected for EVERY non-numpy
  backend (numpy is the only interpreter backend) — iree, xla AND tvm — unless
  an explicit ``opt_level`` is present or the ``ETL_OPT_LEVEL`` env var is set
  (injection is skipped when the env var is set, so the pipeline's env
  machinery applies it at compile; env unset/blank -> the O3 harness default is
  injected). numpy is the reference interpreter — NEVER affected.

Pure-python unit tests (no compiler, no torch): ``import etl.bench._util`` is
torch-safe (torch is imported lazily inside function bodies only).
"""
from __future__ import annotations

import types

import pytest

from etl import core
from etl.bench._util import (
    format_device,
    resolve_backend_options,
    resolve_device,
    resolve_torch_device,
)
from etl.pipeline_options import apply_env_options

_ALL_DEVICE_KINDS = [
    core.Device("cpu", 0),
    core.Device("cuda", 0),
    core.Device("cuda", 3),
    core.Device("rocm", 0),
    core.Device("hip", 1),
    core.Device("vulkan", 0),
    core.Device("vulkan", 2),
]


@pytest.mark.parametrize("device", _ALL_DEVICE_KINDS)
def test_numpy_backend_never_gets_opt_level(device):
    """numpy is the reference interpreter: no target_backends, no opt_level —
    the options dict passes through untouched (even when user options exist),
    for every device kind (cpu / cuda / rocm / hip / vulkan)."""
    assert resolve_backend_options("numpy", device, {}) == {}
    given = {"opt_level": "O1", "target_backends": ["cuda"]}
    assert resolve_backend_options("numpy", device, given) == given


def test_iree_backend_gets_O3_and_llvm_cpu_default():
    """The literal iree backend gets the O3 harness default plus the
    device-derived target_backends default (llvm-cpu for a cpu device)."""
    opts = resolve_backend_options("iree", core.Device("cpu", 0), {})
    assert opts["opt_level"] == "O3"
    assert opts["target_backends"] == ["llvm-cpu"]


@pytest.mark.parametrize(
    "backend",
    ["iree", "iree-cuda", "iree_llvm_cpu", "iree_rocm"],
)
def test_iree_family_name_convention_gets_target_backends(backend):
    """target_backends injection is gated on the ``backend.startswith("iree")``
    name convention (the pipeline's own env-default convention) — iree-family
    names beyond the literal "iree" also receive the device-derived default."""
    opts = resolve_backend_options(backend, core.Device("cuda", 0), {})
    assert opts["target_backends"] == ["cuda"]
    assert opts["opt_level"] == "O3"


@pytest.mark.parametrize(
    "device",
    _ALL_DEVICE_KINDS,
)
@pytest.mark.parametrize("backend", ["xla", "tvm"])
def test_non_iree_compiler_backend_gets_O3_without_target_backends(
    backend, device
):
    """Every non-numpy compiler backend gets the O3 harness default, but the
    iree-only target_backends default is NOT injected for xla/tvm — on ANY
    device (cpu / cuda / rocm / hip / vulkan)."""
    opts = resolve_backend_options(backend, device, {})
    assert opts["opt_level"] == "O3"
    assert "target_backends" not in opts


def test_explicit_opt_level_wins_never_overridden():
    """An explicit opt_level is never overridden by the harness default."""
    for value in ("O1", "O0", 2, 0):
        opts = resolve_backend_options(
            "iree", core.Device("cpu", 0), {"opt_level": value}
        )
        assert opts["opt_level"] == value


def test_env_set_skips_injection(monkeypatch):
    """ETL_OPT_LEVEL set -> the harness default is NOT injected; the env value
    flows through the pipeline's env machinery instead."""
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    opts = resolve_backend_options("iree", core.Device("cpu", 0), {})
    assert "opt_level" not in opts
    # The pipeline env half then applies the env value at compile.
    resolved = apply_env_options("iree", opts, "compile")
    assert resolved["opt_level"] == 2
    assert resolved["target_backends"] == ["llvm-cpu"]


def test_env_blank_treated_as_unset(monkeypatch):
    """A blank ETL_OPT_LEVEL counts as unset -> O3 injected."""
    for value in ("", "   ", "\t\n"):
        monkeypatch.setenv("ETL_OPT_LEVEL", value)
        opts = resolve_backend_options("iree", core.Device("cpu", 0), {})
        assert opts["opt_level"] == "O3"


def test_env_unset_injects_O3(monkeypatch):
    """ETL_OPT_LEVEL unset -> the O3 harness default is injected."""
    monkeypatch.delenv("ETL_OPT_LEVEL", raising=False)
    opts = resolve_backend_options("iree", core.Device("cpu", 0), {})
    assert opts["opt_level"] == "O3"


def test_explicit_opt_level_beats_env(monkeypatch):
    """An explicit opt_level wins even when ETL_OPT_LEVEL is set."""
    monkeypatch.setenv("ETL_OPT_LEVEL", "O2")
    opts = resolve_backend_options(
        "iree", core.Device("cpu", 0), {"opt_level": "O0"}
    )
    assert opts["opt_level"] == "O0"


def test_device_derived_target_backends_default():
    """The device-derived target_backends default: cuda device -> cuda."""
    opts = resolve_backend_options("iree", core.Device("cuda", 0), {})
    assert opts["target_backends"] == ["cuda"]
    assert opts["opt_level"] == "O3"


def test_explicit_target_backends_wins():
    """An explicit target_backends is never overridden by the device default."""
    opts = resolve_backend_options(
        "iree", core.Device("cuda", 0), {"target_backends": ["llvm-cpu"]}
    )
    assert opts["target_backends"] == ["llvm-cpu"]


def test_caller_dict_never_mutated():
    """resolve_backend_options returns a NEW dict; the caller's is untouched."""
    given = {"opt_level": "O1"}
    opts = resolve_backend_options("iree", core.Device("cpu", 0), given)
    assert given == {"opt_level": "O1"}
    assert opts is not given
    given_empty = {}
    resolved = resolve_backend_options("iree", core.Device("cpu", 0), given_empty)
    assert given_empty == {}
    assert resolved is not given_empty


def test_none_backend_options_treated_as_empty():
    """None backend_options is treated as an empty dict."""
    opts = resolve_backend_options("iree", core.Device("cpu", 0), None)
    assert opts["opt_level"] == "O3"
    assert opts["target_backends"] == ["llvm-cpu"]
    assert resolve_backend_options("numpy", core.Device("cpu", 0), None) == {}


# --------------------------------------------------------------------------
# ROCm / HIP device plumbing (device-free pins: NO GPU needed).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    [core.Device("rocm", 0), core.Device("rocm", 3), core.Device("hip", 1)],
)
def test_iree_family_rocm_device_derives_rocm_target_backends(device):
    """A rocm device (and its equivalent 'hip' alias) derives
    target_backends == ["rocm"] for the iree backend family — the AMD GPU
    target counterpart of the cuda / llvm-cpu device default. The O3 harness
    default is injected exactly as for any other device."""
    opts = resolve_backend_options("iree", device, {})
    assert opts["target_backends"] == ["rocm"]
    assert opts["opt_level"] == "O3"


@pytest.mark.parametrize(
    "device",
    [core.Device("rocm", 0), core.Device("rocm", 3), core.Device("hip", 1)],
)
@pytest.mark.parametrize("backend", ["xla", "tvm"])
def test_rocm_device_does_not_add_target_backends_for_xla_tvm(backend, device):
    """The rocm target_backends derivation is iree-only: xla/tvm get the O3
    harness default but NO target_backends key, even on a rocm/hip device."""
    opts = resolve_backend_options(backend, device, {})
    assert "target_backends" not in opts
    assert opts["opt_level"] == "O3"


def test_rocm_explicit_target_backends_wins():
    """An explicit target_backends is never overridden on a rocm device."""
    opts = resolve_backend_options(
        "iree", core.Device("rocm", 0), {"target_backends": ["llvm-cpu"]}
    )
    assert opts["target_backends"] == ["llvm-cpu"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("rocm", core.Device("rocm", 0)),
        ("rocm:3", core.Device("rocm", 3)),
        # The "hip" alias is an equivalent spelling: the kind is PRESERVED as
        # given (never rewritten to "rocm") — only the derived
        # target_backends normalizes to ["rocm"].
        ("hip", core.Device("hip", 0)),
        ("hip:0", core.Device("hip", 0)),
    ],
)
def test_resolve_device_accepts_rocm_and_hip(text, expected):
    """ROCm devices parse as ``"KIND[:INDEX]"`` with the kind preserved."""
    assert resolve_device(text) == expected


def test_resolve_device_documented_failure_messages_mention_rocm_hip():
    """The unsupported-kind help text lists the AMD kinds ('rocm' + the
    'hip' alias) — both for a string device and for a pre-built Device."""
    with pytest.raises(ValueError, match=r"rocm") as string_case:
        resolve_device("tpu")
    assert "hip" in str(string_case.value)

    with pytest.raises(ValueError, match=r"rocm") as device_case:
        resolve_device(core.Device("tpu", 0))
    assert "hip" in str(device_case.value)


# --------------------------------------------------------------------------
# Vulkan device plumbing (device-free pins: NO GPU / Vulkan device needed).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("vulkan", core.Device("vulkan", 0)),
        ("vulkan:0", core.Device("vulkan", 0)),
        ("vulkan:3", core.Device("vulkan", 3)),
    ],
)
def test_resolve_device_accepts_vulkan(text, expected):
    """``"vulkan"`` / ``"vulkan:INDEX"`` parse with the kind and index
    preserved (bare "vulkan" -> index 0), and the parsed device is a
    first-class supported kind (``Device("vulkan", N)`` instances pass the
    same validation)."""
    assert resolve_device(text) == expected
    assert resolve_device(text).kind == "vulkan"
    assert resolve_device(expected) == expected


def test_format_device_reports_vulkan_as_given():
    """The report string names the etl device as given — "vulkan:0" is NOT
    collapsed to "cpu" (only ``Device("cpu", 0)`` is) and never relabelled as
    a GPU kind: the kind names the etl device, not the silicon (a lavapipe
    software Vulkan device is not GPU hardware)."""
    assert format_device(core.Device("vulkan", 0)) == "vulkan:0"
    assert format_device(core.Device("vulkan", 3)) == "vulkan:3"
    assert format_device(core.Device("cpu", 0)) == "cpu"


def test_resolve_device_documented_failure_message_lists_vulkan():
    """The unsupported-kind help text lists 'vulkan' among the accepted kinds
    — both for a string device and for a pre-built Device."""
    with pytest.raises(ValueError, match=r"vulkan") as string_case:
        resolve_device("tpu")
    assert "'vulkan'" in str(string_case.value)

    with pytest.raises(ValueError, match=r"vulkan") as device_case:
        resolve_device(core.Device("tpu", 0))
    assert "'vulkan'" in str(device_case.value)


@pytest.mark.parametrize(
    "device",
    [core.Device("vulkan", 0), core.Device("vulkan", 3)],
)
def test_iree_family_vulkan_device_derives_vulkan_spirv_target_backends(device):
    """A vulkan device derives target_backends == ["vulkan-spirv"] for the
    iree backend family — the IREE Vulkan/SPIR-V compile-target spelling, NOT
    ["vulkan"] and not the ``--iree-vulkan-target`` profile flag (that is the
    separate ``vulkan_target`` option). The O3 harness default is injected
    exactly as for any other device."""
    opts = resolve_backend_options("iree", device, {})
    assert opts["target_backends"] == ["vulkan-spirv"]
    assert isinstance(opts["target_backends"], list)
    assert opts["target_backends"] != ["vulkan"]
    (target,) = opts["target_backends"]
    assert target == "vulkan-spirv" and not target.startswith("--")
    assert opts["opt_level"] == "O3"
    # The SPIR-V profile knob is never injected by the harness default.
    assert "vulkan_target" not in opts


@pytest.mark.parametrize("backend", ["iree", "iree-vulkan", "iree_vulkan"])
def test_iree_family_name_convention_derives_vulkan_target_backends(backend):
    """The vulkan derivation is gated on the same ``startswith("iree")`` name
    convention as the cuda/rocm ones."""
    opts = resolve_backend_options(backend, core.Device("vulkan", 0), {})
    assert opts["target_backends"] == ["vulkan-spirv"]
    assert opts["opt_level"] == "O3"


@pytest.mark.parametrize(
    "device",
    [core.Device("vulkan", 0), core.Device("vulkan", 3)],
)
@pytest.mark.parametrize("backend", ["xla", "tvm"])
def test_vulkan_device_does_not_add_target_backends_for_xla_tvm(backend, device):
    """The vulkan target_backends derivation is iree-only: xla/tvm get the O3
    harness default but NO target_backends key, even on a vulkan device."""
    opts = resolve_backend_options(backend, device, {})
    assert "target_backends" not in opts
    assert opts["opt_level"] == "O3"


def test_vulkan_explicit_options_win_never_overridden():
    """Explicit options always win on a vulkan device: an explicit
    target_backends is never replaced by the derived default, and an explicit
    opt_level is never replaced by the harness default."""
    opts = resolve_backend_options(
        "iree", core.Device("vulkan", 0), {"target_backends": ["llvm-cpu"]}
    )
    assert opts["target_backends"] == ["llvm-cpu"]
    opts = resolve_backend_options(
        "iree",
        core.Device("vulkan", 0),
        {"target_backends": ["vulkan-spirv"], "opt_level": "O1"},
    )
    assert opts == {"target_backends": ["vulkan-spirv"], "opt_level": "O1"}


def test_numpy_backend_untouched_by_vulkan_device():
    """numpy is the reference interpreter: a vulkan device injects nothing."""
    assert resolve_backend_options("numpy", core.Device("vulkan", 0), {}) == {}


# --------------------------------------------------------------------------
# resolve_torch_device: a vulkan device is NEVER a torch cuda device.
# --------------------------------------------------------------------------


class _FakeCuda:
    """Minimal ``torch.cuda`` stand-in that records what was asked of it."""

    def __init__(self, available=True, count=4):
        self.available = available
        self.count = count
        self.queried: list = []

    def is_available(self):
        self.queried.append("is_available")
        return self.available

    def device_count(self):
        self.queried.append("device_count")
        return self.count


class _FakeTorch:
    """Minimal torch module stand-in: claims CUDA support with N devices.

    No torch import (the file is torch-optional; ``resolve_torch_device``
    receives an already-imported module and never imports torch itself).
    """

    def __init__(self, available=True, count=4, hip=None):
        self.cuda = _FakeCuda(available, count)
        self.version = types.SimpleNamespace(hip=hip)

    def device(self, name):
        return ("torch.device", name)


def test_resolve_torch_device_never_maps_vulkan_onto_torch_cuda():
    """torch ships NO Vulkan backend, so a vulkan device must never be aliased
    onto a torch cuda device: even for a fake torch that claims
    ``cuda.is_available() is True`` with ``device_count() >= 1`` (and would
    happily accept ``cuda:0``), a vulkan device resolves to ``None`` — torch
    references degrade to CPU. The torch cuda API is not even consulted."""
    fake = _FakeTorch(available=True, count=4)
    assert resolve_torch_device(core.Device("vulkan", 0), fake) is None
    # Explicit: the fake WOULD serve cuda (positive control, same module).
    assert resolve_torch_device(core.Device("cuda", 0), fake) == (
        "torch.device",
        "cuda:0",
    )
    # ... but the vulkan lookup above never touched the torch cuda API.
    assert fake.cuda.queried == ["is_available", "device_count"]


def test_resolve_torch_device_vulkan_index_beyond_count_is_none():
    """A vulkan index beyond any torch device count is still ``None`` — the
    kind alone rules it out, so the index is irrelevant (never remapped onto
    a cuda ordinal either)."""
    fake = _FakeTorch(available=True, count=2)
    assert resolve_torch_device(core.Device("vulkan", 9), fake) is None
    assert fake.cuda.queried == []
    # The cuda analogue shows the fake really is index-aware (cuda:9 -> None
    # is an index decision, vulkan:9 -> None is a kind decision).
    assert resolve_torch_device(core.Device("cuda", 9), fake) is None
    assert resolve_torch_device(core.Device("cuda", 1), fake) == (
        "torch.device",
        "cuda:1",
    )
