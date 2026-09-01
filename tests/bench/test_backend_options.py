"""etl.bench backend-options resolution contract (``_util.resolve_backend_options``).

The harness resolves the backend options dict passed to ``etl.build`` (and to
every ``Example.runner`` factory): the device-derived ``target_backends``
default for every non-numpy backend, and the ``opt_level`` harness default
``"O3"`` — injected for every non-numpy backend unless an explicit
``opt_level`` is present or the ``ETL_OPT_LEVEL`` env var is set/blank-checked
(injection is skipped when the env var is set, so the pipeline's env
machinery applies it at compile; env unset/blank -> the O3 harness default is
injected). numpy is the reference interpreter — NEVER affected.

Pure-python unit tests (no compiler, no torch): ``import etl.bench._util`` is
torch-safe (torch is imported lazily inside function bodies only).
"""
from __future__ import annotations

import pytest

from etl import core
from etl.bench._util import resolve_backend_options
from etl.pipeline_options import apply_env_options


def test_numpy_backend_never_gets_opt_level():
    """numpy is the reference interpreter: no target_backends, no opt_level —
    the options dict passes through untouched (even when user options exist)."""
    assert resolve_backend_options("numpy", core.Device("cpu", 0), {}) == {}
    given = {"opt_level": "O1", "target_backends": ["cuda"]}
    assert resolve_backend_options("numpy", core.Device("cpu", 0), given) == given


@pytest.mark.parametrize("backend", ["iree", "xla", "tvm"])
def test_compiler_backend_gets_O3_by_default(backend):
    """Every non-numpy backend gets the O3 harness default injected."""
    opts = resolve_backend_options(backend, core.Device("cpu", 0), {})
    assert opts["opt_level"] == "O3"
    assert opts["target_backends"] == ["llvm-cpu"]


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
