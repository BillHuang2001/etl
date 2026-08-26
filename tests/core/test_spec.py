"""Tests for ``TensorSpec`` validation/normalization.

``TensorSpec`` describes a future runtime tensor (no storage). These tests
pin the documented contract in ``etl/core/spec.py``: shape tuple-ification
and entry validation, dtype normalization to ``np.dtype``, the ``rank``
property, frozen immutability, optional device/name fields, equality
semantics, and import paths.
"""

import dataclasses

import numpy as np
import pytest

from etl import TensorSpec
from etl.core import DTypeError, Device, Dim, TensorSpec as CoreTensorSpec, float32

# --- imports -----------------------------------------------------------------


def test_importable_from_etl_and_etl_core():
    assert TensorSpec is CoreTensorSpec


# --- construction / normalization -------------------------------------------


def test_shape_list_is_tupleified():
    spec = TensorSpec([2, 3], float32)
    assert spec.shape == (2, 3)
    assert isinstance(spec.shape, tuple)


def test_shape_entries_preserved():
    dim = Dim("batch")
    expr = dim * 2 + 1
    spec = TensorSpec((dim, expr, 3, None), float32)
    assert spec.shape == (dim, expr, 3, None)
    assert spec.shape[0] is dim
    assert spec.shape[1] is expr
    assert spec.rank == 4


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (float, np.dtype("float64")),
        (np.float32, np.dtype("float32")),
        ("int32", np.dtype("int32")),
        (np.dtype("float16"), np.dtype("float16")),  # passthrough
        (int, np.dtype("int64")),
        (np.int16, np.dtype("int16")),
    ],
)
def test_dtype_normalization(given, expected):
    spec = TensorSpec((2,), given)
    assert spec.dtype == expected
    assert isinstance(spec.dtype, np.dtype)


def test_dtype_passthrough_preserves_instance():
    dt = np.dtype("float16")
    assert TensorSpec((2,), dt).dtype is dt


@pytest.mark.parametrize(
    ("shape", "expected_rank"),
    [
        ((), 0),
        ((2,), 1),
        ((2, 3), 2),
        ((2, None, Dim("n")), 3),
        ([4, 5], 2),  # lists normalize too, rank is always known
    ],
)
def test_rank_equals_shape_length(shape, expected_rank):
    assert TensorSpec(shape, float32).rank == expected_rank


# --- immutability -----------------------------------------------------------


def test_frozen_immutable():
    spec = TensorSpec((2, 3), float32)
    assert isinstance(spec.shape, tuple)  # stored shape is immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.shape = (1,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dtype = np.dtype("int8")


# --- device / name ----------------------------------------------------------


def test_device_and_name_default_to_none():
    spec = TensorSpec((2,), float32)
    assert spec.device is None
    assert spec.name is None


def test_device_and_name_are_stored():
    spec = TensorSpec((2,), float32, device=Device("cpu", 1), name="weights")
    assert spec.device == Device("cpu", 1)
    assert spec.name == "weights"


def test_none_dims_allowed():
    spec = TensorSpec((None, 3, None), float32)
    assert spec.shape == (None, 3, None)
    assert spec.rank == 3


# --- validation errors -------------------------------------------------------


@pytest.mark.parametrize("bad", [2.5, "batch", {"a": 1}])
def test_invalid_shape_entries_raise_type_error(bad):
    with pytest.raises(
        TypeError,
        match=r"TensorSpec\.shape entries must be Dim \| DimExpr \| int \| None, got",
    ):
        TensorSpec((2, bad), float32)


@pytest.mark.parametrize(
    ("bad", "msg"),
    [
        ("not_a_real_dtype", r"Unknown dtype name: 'not_a_real_dtype'"),
        (3.14, r"Cannot convert object of type float"),
        (object(), r"Cannot convert object of type object"),
    ],
)
def test_invalid_dtype_raises_dtype_error(bad, msg):
    with pytest.raises(DTypeError, match=msg):
        TensorSpec((2,), bad)


def test_invalid_dtype_type_raises_dtype_error(monkeypatch):
    # On numpy >= 2, np.dtype(<type>) coerces unknown types to `object`
    # instead of raising, so the documented "Cannot interpret type" branch of
    # etl's normalizer is exercised by rejecting the np.dtype call directly.
    class _RejectingDtype:
        # np.dtype cannot be subclassed (numpy >= 2); a plain class works as
        # the isinstance second arg and raises on instantiation.
        def __init__(self, obj, *args, **kwargs):
            raise TypeError(f"Cannot interpret type {obj!r} as a dtype")

    monkeypatch.setattr(np, "dtype", _RejectingDtype)
    with pytest.raises(
        DTypeError, match=r"Cannot interpret type <class 'dict'> as a dtype"
    ):
        TensorSpec((2,), dict)


# --- equality ---------------------------------------------------------------


def test_specs_equal_after_normalization():
    # List shape + string dtype normalize to the same spec as the tuple-shape
    # + np.dtype form: frozen-dataclass eq compares the normalized fields.
    assert TensorSpec((2, 3), float32) == TensorSpec([2, 3], "float32")
    assert not (TensorSpec((2, 3), float32) != TensorSpec([2, 3], "float32"))


def test_spec_equality_includes_all_fields():
    base = TensorSpec((2, 3), float32)
    assert base != TensorSpec((2, 4), float32)  # shape differs
    assert base != TensorSpec((2, 3), np.dtype("int32"))  # dtype differs
    assert base != TensorSpec((2, 3), float32, device=Device("cpu"))  # device
    assert base != TensorSpec((2, 3), float32, name="x")  # name
    # identical fields (normalized) are equal, device/name included
    assert base == TensorSpec([2, 3], float32)
    named = TensorSpec((2, 3), float32, device=Device("cpu", 0), name="x")
    assert named == TensorSpec((2, 3), "float32", device=Device("cpu", 0), name="x")
