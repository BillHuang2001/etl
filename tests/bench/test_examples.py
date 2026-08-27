"""etl.bench example registry contract.

``list_examples()`` returns exactly the 10 documented example names in
registry order; ``get_example(name)`` returns a frozen ``Example`` dataclass
(``name``, ``description``, ``specs`` tuple of ``etl.TensorSpec``, ``graph``
and ``numpy_ref`` callables, optional ``torch_ref``) with a
``generate_inputs(seed)`` method returning one numpy array per spec. Unknown
names raise ``UnknownExampleError`` — a ``ValueError`` subclass importable
from both ``etl.bench`` and ``etl.bench.examples`` — whose message lists the
available names; ``conformance`` raises the same error for unknown names.
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
    list_examples,
)
from etl.bench.examples import UnknownExampleError as ExamplesUnknownExampleError

EXPECTED_NAMES = [
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
]


def test_list_examples_returns_exact_documented_set():
    assert list_examples() == EXPECTED_NAMES


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
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.name = "other"


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
