"""Tests for etl dtype constants and the ``dtype`` normalizer.

Covers the contract in ``etl/core/CONTEXT.md``: ``dtype(obj) -> np.dtype``
normalizer (passthrough / type / string / ``.dtype``-attribute / errors) and
the 14 dtype constants (numpy dtype objects, ``bool_`` instead of ``bool``).
"""

import numpy as np
import pytest

import etl
import etl.core

# (identifier, np.dtype.name, itemsize in bytes, kind)
# Note: the etl identifier is ``bool_``, but the underlying numpy dtype's
# canonical ``.name`` is "bool" (``bool_ = np.dtype("bool")``).
DTYPE_CONSTANTS = [
    ("float16", "float16", 2, "f"),
    ("float32", "float32", 4, "f"),
    ("float64", "float64", 8, "f"),
    ("int8", "int8", 1, "i"),
    ("int16", "int16", 2, "i"),
    ("int32", "int32", 4, "i"),
    ("int64", "int64", 8, "i"),
    ("uint8", "uint8", 1, "u"),
    ("uint16", "uint16", 2, "u"),
    ("uint32", "uint32", 4, "u"),
    ("uint64", "uint64", 8, "u"),
    ("bool_", "bool", 1, "b"),
    ("complex64", "complex64", 8, "c"),
    ("complex128", "complex128", 16, "c"),
]

ALL_NAMES = [name for name, _, _, _ in DTYPE_CONSTANTS]


# --- dtype constants --------------------------------------------------------


@pytest.mark.parametrize("name,dtype_name,itemsize,kind", DTYPE_CONSTANTS)
def test_constants_are_np_dtypes(name, dtype_name, itemsize, kind):
    dt = getattr(etl.core, name)
    assert isinstance(dt, np.dtype)
    assert dt.name == dtype_name
    assert dt.itemsize == itemsize
    assert dt.kind == kind


@pytest.mark.parametrize("name,dtype_name,itemsize,kind", DTYPE_CONSTANTS)
def test_constants_exported_from_etl_and_etl_core(name, dtype_name, itemsize, kind):
    assert hasattr(etl.core, name)
    assert hasattr(etl, name)
    # same object re-exported, not a copy
    assert getattr(etl, name) is getattr(etl.core, name)


def test_bool_constant_is_named_bool_():
    assert etl.bool_ == np.dtype(bool)
    assert etl.bool_.name == "bool"  # numpy's canonical name for the dtype
    # the constant must not be exported as the bare builtin name `bool`
    assert not isinstance(getattr(etl, "bool", None), np.dtype)


# --- dtype() normalizer -----------------------------------------------------


def test_dtype_np_dtype_passthrough_is_same_object():
    d = np.dtype("float32")
    assert etl.dtype(d) is d


@pytest.mark.parametrize(
    ("py_type", "expected_name"),
    [
        (float, "float64"),
        (int, "int64"),
        (bool, "bool"),
        (complex, "complex128"),
    ],
)
def test_dtype_python_types(py_type, expected_name):
    assert etl.dtype(py_type) == np.dtype(expected_name)


@pytest.mark.parametrize(
    ("np_type", "expected_name"),
    [
        (np.float16, "float16"),
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int8, "int8"),
        (np.int64, "int64"),
        (np.bool_, "bool"),
        (np.complex64, "complex64"),
    ],
)
def test_dtype_numpy_scalar_types(np_type, expected_name):
    assert etl.dtype(np_type) == np.dtype(expected_name)


@pytest.mark.parametrize(
    "name",
    ["float16", "float32", "float64", "int8", "int64", "uint8", "complex128", "bool"],
)
def test_dtype_strings(name):
    assert etl.dtype(name) == np.dtype(name)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_dtype_constants_pass_through(name):
    c = getattr(etl.core, name)
    assert etl.dtype(c) is c


class _HasDType:
    """Minimal duck-typed object exposing a ``dtype`` attribute."""

    def __init__(self, dt):
        self.dtype = dt


def test_dtype_objects_with_dtype_attribute():
    assert etl.dtype(_HasDType(np.dtype("float32"))) == etl.float32
    assert etl.dtype(_HasDType(etl.int64)) is etl.int64
    assert etl.dtype(np.zeros(2)) == etl.float64


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "notadtype",
        "float33",
        [1, 2, 3],
        {"float32": 1},
        object(),
    ],
)
def test_dtype_invalid_input_raises_dtypeerror(bad):
    with pytest.raises(etl.DTypeError):
        etl.dtype(bad)
