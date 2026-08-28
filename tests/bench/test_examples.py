"""etl.bench example registry contract.

``list_examples()`` returns exactly the 26 documented example names in
registry order (micro → grad → vectorize → large); ``get_example(name)``
returns a frozen ``Example`` dataclass with the 11 documented fields
(``name``, ``description``, ``specs`` tuple of ``etl.TensorSpec``, ``graph``
and ``numpy_ref`` callables, optional ``torch_ref``, per-example ``rtol``/
``atol``/``tolerance`` overrides, ``category``, optional ``inputs_fn``) and a
``generate_inputs(seed)`` method returning one numpy array per spec.
``list_categories()`` returns the four categories in first-appearance order.
Unknown names raise ``UnknownExampleError`` — a ``ValueError`` subclass
importable from both ``etl.bench`` and ``etl.bench.examples`` — whose message
lists the available names; ``conformance`` raises the same error for unknown
names.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import etl
from etl.bench import (
    UnknownExampleError,
    conformance,
    get_example,
    list_categories,
    list_examples,
)
from etl.bench.examples import UnknownExampleError as ExamplesUnknownExampleError

# Authoritative registry (registry order = module import order: micro, grad,
# vectorize, large) — documented in etl/bench/CONTEXT.md.
EXPECTED_NAMES = [
    # micro (10)
    "matmul",
    "conv2d",
    "conv2d_same",
    "conv2d_stride2",
    "elementwise_fusion",
    "softmax",
    "layernorm",
    "mlp",
    "cumsum",
    "attention",
    # grad (4)
    "grad_mlp",
    "grad_mix",
    "grad_stopgrad",
    "grad_structural",
    # vectorize (6)
    "vmap_softmax",
    "vmap_layernorm",
    "vmap_linear",
    "vmap_mlp",
    "vmap_attention",
    "vmap_grad_mlp",
    # large (6)
    "transformer",
    "nbody",
    "matmul_1024",
    "conv2d_large",
    "layernorm_large",
    "vmap_mlp_large",
]

EXPECTED_CATEGORIES = ["micro", "grad", "vectorize", "large"]


def test_list_examples_returns_exact_documented_set():
    assert list_examples() == EXPECTED_NAMES
    # Every registered example carries one of the four documented categories,
    # and all four categories are present in the registry.
    categories = {get_example(name).category for name in EXPECTED_NAMES}
    assert categories == set(EXPECTED_CATEGORIES)


def test_list_categories_returns_documented_categories():
    assert list_categories() == EXPECTED_CATEGORIES


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_get_example_returns_valid_example(name):
    example = get_example(name)
    assert example.name == name
    assert example.description
    assert isinstance(example.specs, tuple) and len(example.specs) >= 1
    for spec in example.specs:
        assert isinstance(spec, etl.TensorSpec)
        assert all(isinstance(dim, int) for dim in spec.shape)
    assert callable(example.graph)
    assert callable(example.numpy_ref)
    inputs = example.generate_inputs(seed=0)
    assert isinstance(inputs, list)
    assert len(inputs) == len(example.specs)
    for spec, array in zip(example.specs, inputs):
        assert isinstance(array, np.ndarray)
        assert array.shape == tuple(spec.shape)
        assert array.dtype == np.dtype(spec.dtype)


def test_example_is_a_frozen_dataclass():
    example = get_example("matmul")
    assert dataclasses.is_dataclass(example)
    assert {field.name for field in dataclasses.fields(example)} == {
        "name",
        "description",
        "specs",
        "graph",
        "numpy_ref",
        "torch_ref",
        "rtol",
        "atol",
        "tolerance",
        "category",
        "inputs_fn",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.name = "other"


def test_example_per_example_tolerance_overrides():
    # mlp carries a max-abs-error override (iree fp32 noise — documented);
    # matmul falls back to the global defaults (all overrides None).
    assert get_example("mlp").tolerance == 1e-4
    assert get_example("mlp").rtol is None
    assert get_example("mlp").atol is None
    assert get_example("matmul").tolerance is None
    # grad examples carry rtol=atol=1e-3 (fd-reference noise — documented);
    # vectorize examples stay at the strict defaults.
    for name in ("grad_mlp", "grad_mix", "grad_stopgrad", "grad_structural"):
        example = get_example(name)
        assert example.rtol == 1e-3
        assert example.atol == 1e-3
        assert example.tolerance is None
    for name in ("vmap_softmax", "vmap_mlp", "vmap_grad_mlp"):
        example = get_example(name)
        assert example.rtol is None
        assert example.atol is None
        assert example.tolerance is None


def test_get_example_unknown_name_error_lists_available_names():
    with pytest.raises(UnknownExampleError) as excinfo:
        get_example("does_not_exist")
    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert "matmul" in message
    assert "attention" in message


def test_unknown_example_error_importable_from_examples_module():
    assert ExamplesUnknownExampleError is UnknownExampleError


def test_conformance_unknown_name_raises_same_error():
    with pytest.raises(UnknownExampleError) as excinfo:
        conformance(["nope"])
    assert isinstance(excinfo.value, ValueError)
    assert "nope" in str(excinfo.value)
