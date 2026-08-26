"""`etl.numpy` (alias `enp`) namespace surface tests.

Validates the documented public surface of `etl.numpy` — the `enp` alias,
the three documented import styles, the exact `__all__` name set (all
callable except the `linalg` submodule), and the intentional v1 deferrals
(absent names raise `AttributeError` — never silently exist).

The IR-equivalence of every function (enp.f ≡ mapped etl.ops composition)
is covered in `test_equivalence.py` — the defining property of the
namespace (see `etl/numpy/CONTEXT.md`).
"""

from __future__ import annotations

import importlib
import types

import pytest

import etl
import etl.numpy as enp

# --- The documented v1 surface ----------------------------------------------
# etl/numpy/CONTEXT.md "API surface" + etl/numpy/__init__.py __all__:
# 55 callables + the `linalg` submodule = 56 names.
EXPECTED_SURFACE = frozenset(
    {
        # elementwise
        "abs", "add", "subtract", "multiply", "divide", "power", "maximum",
        "minimum", "negative", "square", "sqrt", "exp", "log", "sin", "cos",
        "tanh", "sign", "clip", "astype",
        # logic
        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "logical_and", "logical_or", "logical_not", "where",
        # shape
        "reshape", "transpose", "broadcast_to", "expand_dims", "squeeze",
        "concatenate", "stack", "split", "pad", "tril", "triu",
        # reductions
        "sum", "mean", "prod", "max", "min", "argmax", "argmin", "cumsum",
        # creation
        "zeros", "ones", "full", "empty", "arange",
        # linalg
        "matmul", "dot", "linalg",
    }
)

CALLABLE_NAMES = sorted(EXPECTED_SURFACE - {"linalg"})

# Documented v1 deferrals (etl/numpy/CONTEXT.md "Deferrals (v1)").
DEFERRED_NAMES = ["linspace", "absolute", "var", "std", "einsum"]


def test_enp_is_numpy_alias():
    """`enp` is the documented alias of `etl.numpy` (registered at the
    package level in etl/__init__.py)."""
    assert etl.enp is etl.numpy


def test_import_styles():
    """All documented import forms resolve to the same module object."""
    # import etl.numpy as enp
    assert importlib.import_module("etl.numpy") is etl.numpy
    # from etl import numpy as enp
    assert importlib.import_module("etl").numpy is etl.numpy
    # the etl-level alias
    assert importlib.import_module("etl").enp is etl.numpy


def test_surface_exact():
    """`enp.__all__` is exactly the documented surface: no missing names,
    no extras, no duplicates — and every advertised name resolves."""
    assert set(enp.__all__) == EXPECTED_SURFACE
    assert len(enp.__all__) == len(EXPECTED_SURFACE)
    for name in EXPECTED_SURFACE:
        assert getattr(enp, name, None) is not None, name


@pytest.mark.parametrize("name", CALLABLE_NAMES)
def test_surface_members_are_callable(name):
    """Every documented name except `linalg` is a callable function."""
    assert callable(getattr(enp, name))


def test_linalg_submodule():
    """`linalg` is a module whose v1 surface is exactly `solve`."""
    assert isinstance(enp.linalg, types.ModuleType)
    assert enp.linalg.__all__ == ["solve"]
    assert callable(enp.linalg.solve)


@pytest.mark.parametrize("name", DEFERRED_NAMES)
def test_deferred_names_absent(name):
    """Documented v1 deferrals are absent: attribute access raises
    AttributeError — never a silent workaround."""
    with pytest.raises(AttributeError, match=name):
        getattr(enp, name)
