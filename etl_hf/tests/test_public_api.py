"""Tests for the public surface of the ``etl_hf`` package (``etl_hf/__init__.py``).

The package-level contract is what users see first, so it is pinned here:

* ``import etl_hf`` works and pulls in **no** optional dependency (torch /
  transformers / safetensors / iree) — checked in a subprocess, so the importing
  interpreter's own state cannot hide a leak;
* ``__all__`` is complete, duplicate-free, and every entry is reachable — a
  star-import yields exactly the documented names;
* every exported function/class carries a docstring (the package is API
  documentation as much as code);
* the re-exports are the *same objects* as their defining modules' attributes
  (no shadowing wrapper that could drift);
* the submodules (``ops``, ``reference``, ``weights``, ``config``, ``families``,
  ``build``, ``pack``) are reachable as attributes;
* the error hierarchy derives from :class:`etl.core.ETLError` as documented.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import types

import pytest

import etl
import etl_hf
from etl_hf import errors as errors_module

#: Every documented public name, and the ``(module, attribute)`` it re-exports.
_EXPECTED_EXPORTS = {
    "HFModel": ("etl_hf.build", "HFModel"),
    "ModelConfig": ("etl_hf.config", "ModelConfig"),
    "build_model": ("etl_hf.build", "build_model"),
    "load_hf": ("etl_hf.build", "load_hf"),
    "place_params": ("etl_hf.build", "place_params"),
    "run_model": ("etl_hf.build", "run_model"),
    "PackPlan": ("etl_hf.pack", "PackPlan"),
    "pack_plan": ("etl_hf.pack", "pack_plan"),
    "pack_params": ("etl_hf.pack", "pack_params"),
    "FAMILIES": ("etl_hf.families", "FAMILIES"),
    "FamilySpec": ("etl_hf.families", "FamilySpec"),
    "family_for_model_type": ("etl_hf.families", "family_for_model_type"),
    "get_family": ("etl_hf.families", "get_family"),
    "HFError": ("etl_hf.errors", "HFError"),
    "HFConfigError": ("etl_hf.errors", "HFConfigError"),
    "HFWeightError": ("etl_hf.errors", "HFWeightError"),
    "HFUnsupportedError": ("etl_hf.errors", "HFUnsupportedError"),
}

#: Submodules the package exposes as attributes (``etl_hf.<name>``).
_SUBMODULES = ("config", "families", "build", "ops", "pack", "reference", "weights")


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
    module = sys.modules[module_path]
    assert getattr(etl_hf, name) is getattr(module, attribute)


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


def test_public_functions_document_their_arguments():
    """Every parameter of a public function is mentioned in its docstring."""
    for name in ("build_model", "run_model", "place_params", "load_hf", "get_family"):
        function = getattr(etl_hf, name)
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
        "missing = [name for name in ('etl_hf.ops', 'etl_hf.build', 'etl_hf.weights')\n"
        "           if name not in sys.modules]\n"
        "print('LEAKED=' + ','.join(leaked))\n"
        "print('MISSING=' + ','.join(missing))\n"
    )
    reported = dict(line.split("=", 1) for line in output.strip().splitlines())
    assert reported["LEAKED"] == "", f"optional dependencies imported: {reported['LEAKED']}"
    assert reported["MISSING"] == "", f"submodules not imported: {reported['MISSING']}"


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
        "print(etl_hf.load_hf.__name__, etl_hf.__version__, sorted(etl_hf.FAMILIES))\n"
    )
    assert output.strip() == "load_hf 0.1.0 ['gpt2', 'llama']"
