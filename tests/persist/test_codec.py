"""pytest suite for the etl persistence value codec (``encode_value``/``decode_value``).

Covers the envelope contract, round-trips for every built-in type,
JSON-serializability, error behavior, and the ``register_codec`` extension.
"""

import base64
import copy
import io
import json
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest

from etl.core import (
    Device,
    Dim,
    DimExpr,
    PersistenceError,
    TensorSpec,
    TreeSpec,
    flatten,
)
from etl.persist import codec as codec_mod
from etl.persist import decode_value, encode_value, register_codec

# --- deep-equality helpers (NaN-aware; TreeSpec needs a recursive comparator) ---


def assert_treespec_equal(a, b):
    """Compare two TreeSpecs field-by-field (they have no usable ``==``)."""
    assert isinstance(a, TreeSpec) and isinstance(b, TreeSpec), (a, b)
    assert a.type == b.type
    assert_deep_equal(a.context, b.context)
    assert_deep_equal(a.node_data, b.node_data)
    assert len(a.children) == len(b.children)
    for child_a, child_b in zip(a.children, b.children):
        assert_treespec_equal(child_a, child_b)


def assert_deep_equal(a, b):
    """NaN-aware structural equality for every codec-supported value."""
    if isinstance(a, TreeSpec):
        assert_treespec_equal(a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=True)
    elif isinstance(a, np.generic):
        assert type(a) is type(b) and a == b
    elif isinstance(a, float):
        assert isinstance(b, float)
        assert (math.isnan(a) and math.isnan(b)) or a == b
    elif isinstance(a, complex):
        assert isinstance(b, complex)
        for x, y in ((a.real, b.real), (a.imag, b.imag)):
            assert (math.isnan(x) and math.isnan(y)) or x == y
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_deep_equal(x, y)
    elif isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_deep_equal(x, y)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for k in a:
            assert_deep_equal(a[k], b[k])
    else:
        assert type(a) is type(b) and a == b


def roundtrip(value):
    """Encode → (assert the envelope is JSON-serializable) → decode."""
    envelope = encode_value(value)
    json.dumps(envelope)  # every envelope must be JSON-serializable
    return decode_value(envelope)


# --- scalars and envelope format ---

SCALAR_VALUES = [
    None, True, False, 0, -7, 2**63 + 123,  # arbitrary-precision ints survive JSON
    0.0, -1.25, 1e300, "", "hello", "snowman \u2603", 1 + 2j, -0.5j,
]


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_scalar_roundtrip(value):
    decoded = roundtrip(value)
    assert type(decoded) is type(value)
    assert_deep_equal(decoded, value)


@pytest.mark.parametrize(
    ("value", "type_name", "data"),
    [
        (None, "NoneType", None),
        (True, "bool", True),
        (42, "int", 42),
        (1.5, "float", 1.5),
        ("x", "str", "x"),
    ],
)
def test_envelope_format(value, type_name, data):
    assert encode_value(value) == {
        "__etl_encoded__": True,
        "type": type_name,
        "data": data,
    }


def test_bool_dispatches_apart_from_int():
    assert encode_value(True)["type"] == "bool"
    assert encode_value(1)["type"] == "int"
    assert type(roundtrip(True)) is bool
    assert type(roundtrip(1)) is int


@pytest.mark.parametrize(
    ("value", "payload"),
    [
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ],
)
def test_float_special_roundtrip_payload(value, payload):
    envelope = encode_value(value)
    assert envelope["type"] == "float"
    assert envelope["data"] == payload  # NaN/Inf travel as strings
    assert_deep_equal(decode_value(envelope), value)


def test_complex_special_components_roundtrip():
    envelope = encode_value(complex(1.0, float("nan")))
    assert envelope["data"] == {"real": 1.0, "imag": "nan"}
    decoded = decode_value(envelope)
    assert decoded.real == 1.0
    assert math.isnan(decoded.imag)

    envelope = encode_value(complex(float("-inf"), float("inf")))
    assert envelope["data"] == {"real": "-inf", "imag": "inf"}
    decoded = decode_value(envelope)
    assert decoded.real == float("-inf")
    assert decoded.imag == float("inf")


# --- containers ---


@pytest.mark.parametrize(
    "value",
    [
        [],
        (),
        {},
        [1, [2, [3]]],
        (1, (2, (3,))),
        {"a": {"b": {"c": 1}}},
    ],
)
def test_container_roundtrip(value):
    decoded = roundtrip(value)
    assert type(decoded) is type(value)
    assert_deep_equal(decoded, value)


def test_dict_non_string_keys_roundtrip():
    value = {1: "int-key", 2.5: "float-key", (1, 2): "tuple-key", "s": None}
    decoded = roundtrip(value)
    assert list(decoded) == list(value)
    assert_deep_equal(decoded, value)


def test_shared_reference_is_not_cyclic():
    shared = [1, 2]
    decoded = roundtrip([shared, shared])
    assert decoded == [[1, 2], [1, 2]]


# --- numpy arrays, scalars and dtypes ---

ARRAY_VALUES = [
    np.array([1.5, -2.0, 3.25], dtype=np.float32),
    np.array([1.5, -2.0, 3.25], dtype=np.float64),
    np.array([3, -7, 2**31 - 1], dtype=np.int32),
    np.array([3, -7, 2**40], dtype=np.int64),
    np.array([True, False, True], dtype=np.bool_),
    np.array([1 + 2j, -0.5 + 0.25j], dtype=np.complex64),
    np.array([1 + 2j, -0.5 + 0.25j], dtype=np.complex128),
    np.arange(24, dtype=np.float64).reshape(2, 3, 4),
    np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64),
]


@pytest.mark.parametrize("arr", ARRAY_VALUES)
def test_ndarray_roundtrip(arr):
    decoded = roundtrip(arr)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == arr.dtype
    assert decoded.shape == arr.shape
    assert np.array_equal(decoded, arr, equal_nan=True)


@pytest.mark.parametrize(
    "arr", [np.array([], dtype=np.float32), np.empty((0, 3), dtype=np.int64)]
)
def test_ndarray_empty_roundtrip(arr):
    decoded = roundtrip(arr)
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype
    assert decoded.size == 0


def test_ndarray_non_contiguous_roundtrip():
    arr = np.arange(12.0).reshape(3, 4).T
    assert arr.flags.c_contiguous is False  # premise of the test
    decoded = roundtrip(arr)
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype
    assert np.array_equal(decoded, arr, equal_nan=True)


def test_ndarray_envelope_layout():
    arr = np.array([1, 2], dtype=np.int32)
    data = encode_value(arr)["data"]
    assert data["dtype"] == arr.dtype.str
    assert data["shape"] == list(arr.shape)
    assert isinstance(data["data_b64"], str)
    loaded = np.load(io.BytesIO(base64.b64decode(data["data_b64"])), allow_pickle=False)
    np.testing.assert_array_equal(loaded, arr)


def test_ndarray_decode_is_readonly_and_fresh():
    arr = np.arange(6, dtype=np.float64)
    decoded = decode_value(encode_value(arr))
    assert decoded.flags.writeable is False
    assert decoded.flags.owndata
    arr[0] = 999.0  # mutating the source must never alias into the decoded copy
    assert decoded[0] == 0.0
    with pytest.raises(ValueError):
        decoded[0] = 1.0


GENERIC_VALUES = [
    np.float32(1.5),
    np.float64(-2.25),
    np.int32(-7),
    np.int64(3),
    np.bool_(True),
    np.complex128(1 + 2j),
]


@pytest.mark.parametrize("scalar", GENERIC_VALUES)
def test_numpy_generic_roundtrip(scalar):
    decoded = roundtrip(scalar)
    # decodes back to the numpy scalar, not a 0-d array
    assert isinstance(decoded, type(scalar))
    assert decoded == scalar


@pytest.mark.parametrize("name", ["float32", "int64", "complex128", "bool"])
def test_numpy_dtype_roundtrip(name):
    dt = np.dtype(name)
    decoded = roundtrip(dt)
    assert isinstance(decoded, np.dtype)
    assert decoded == dt


# --- etl value-model types ---


@pytest.mark.parametrize(
    "sl", [slice(None), slice(1, 10, 2), slice(1, None, -1), slice(-3, 0)]
)
def test_slice_roundtrip(sl):
    assert roundtrip(sl) == sl


@pytest.mark.parametrize("dim", [Dim("B"), Dim("B", 16), Dim("seq_len", 128)])
def test_dim_roundtrip(dim):
    decoded = roundtrip(dim)
    assert isinstance(decoded, Dim)
    assert decoded == dim
    assert decoded.name == dim.name
    assert decoded.size == dim.size


def test_dimexpr_roundtrip():
    exprs = [
        Dim("B") * 2 + Dim("C"),
        Dim("B") + Dim("C"),
        Dim("B") // 2,
        Dim("B") % 3,
        2 - Dim("B"),
        Dim("B").min(Dim("C")),
        Dim("B").max(16),
    ]
    for expr in exprs:
        decoded = roundtrip(expr)
        assert isinstance(decoded, DimExpr)
        assert decoded == expr  # DimExpr equality is structural


@pytest.mark.parametrize("dev", [Device("cpu"), Device("cuda", 3)])
def test_device_roundtrip(dev):
    decoded = roundtrip(dev)
    assert isinstance(decoded, Device)
    assert decoded == dev
    assert decoded.kind == dev.kind
    assert decoded.index == dev.index


def test_tensorspec_roundtrip_full():
    original = TensorSpec(
        shape=(Dim("B"), 4, None),
        dtype=np.float32,
        device=Device("cpu"),
        name="x",
    )
    decoded = roundtrip(original)
    assert decoded is not original
    assert decoded.shape == original.shape
    assert decoded.dtype == original.dtype
    assert decoded.device == original.device
    assert decoded.name == original.name


def test_tensorspec_roundtrip_minimal_and_dimexpr():
    original = TensorSpec(shape=(2, 3), dtype="float64")
    decoded = roundtrip(original)
    assert decoded.shape == (2, 3)
    assert decoded.dtype == original.dtype == np.dtype("float64")
    assert decoded.device is None
    assert decoded.name is None

    original = TensorSpec(shape=(Dim("B") * 2 + 1, Dim("C", 8)), dtype=np.int32)
    decoded = roundtrip(original)
    assert decoded.shape == original.shape
    assert decoded.dtype == original.dtype


def test_treespec_roundtrip_leaf():
    _, spec = flatten(3)
    assert spec.type is int
    assert_treespec_equal(roundtrip(spec), spec)


def test_treespec_roundtrip_nested_pytree():
    leaves, spec = flatten({"a": [1, 2.5], "b": (None, "x"), "c": {"d": 3}})
    assert leaves == [1, 2.5, None, "x", 3]
    assert_treespec_equal(roundtrip(spec), spec)


@pytest.mark.parametrize("obj", [[], (), {}])
def test_treespec_roundtrip_empty_containers(obj):
    _, spec = flatten(obj)
    assert_treespec_equal(roundtrip(spec), spec)


def test_nested_mixed_container_roundtrip():
    _, spec = flatten({"x": 1})
    expr = Dim("B") * 2 + Dim("C")
    value = {
        "a": [Dim("B"), np.array([1.0, np.nan, np.inf], dtype=np.float64), {"k": (1, 2.5)}],
        "b": (spec, expr),
    }
    assert_deep_equal(roundtrip(value), value)


# --- invariants ---


def test_encode_does_not_mutate_input():
    value = {"a": [1, (2.5, "x")], "b": {"c": None}}
    snapshot = copy.deepcopy(value)
    encode_value(value)
    assert value == snapshot

    arr = np.arange(6.0).reshape(2, 3)
    snapshot = arr.copy()
    encode_value(arr)
    np.testing.assert_array_equal(arr, snapshot)
    assert arr.flags.writeable is True


# --- error cases ---


@pytest.mark.parametrize(
    "payload",
    [
        {},
        "junk",
        [1, 2],
        {"type": "int", "data": 5},  # missing envelope tag
        {"__etl_encoded__": False, "type": "int", "data": 5},
    ],
)
def test_decode_rejects_non_envelope(payload):
    with pytest.raises(PersistenceError, match="not an encoded value"):
        decode_value(payload)


def test_decode_rejects_corrupt_envelopes():
    with pytest.raises(PersistenceError, match="unknown encoded type"):
        decode_value({"__etl_encoded__": True, "type": "no.such.Type", "data": 1})
    with pytest.raises(PersistenceError, match="no 'data'"):
        decode_value({"__etl_encoded__": True, "type": "int"})
    with pytest.raises(PersistenceError, match="corrupt"):
        decode_value({"__etl_encoded__": True, "type": "int", "data": "not-an-int"})
    with pytest.raises(PersistenceError, match="corrupt"):
        decode_value({"__etl_encoded__": True, "type": "NoneType", "data": 1})


class _TestEnum(Enum):
    ALPHA = "a"
    BETA = "b"


@pytest.mark.parametrize(
    "value",
    [b"bytes", {1, 2}, range(3), _TestEnum.ALPHA, object()],
)
def test_encode_unsupported_type_raises(value):
    with pytest.raises(PersistenceError, match="cannot encode"):
        encode_value(value)


def test_cyclic_structures_rejected():
    lst = []
    lst.append(lst)
    with pytest.raises(PersistenceError, match="cyclic"):
        encode_value(lst)

    d = {}
    d["self"] = d
    with pytest.raises(PersistenceError, match="cyclic"):
        encode_value(d)


# --- register_codec extension point ---


@dataclass(frozen=True)
class _CodecPoint:
    """Module-level custom type used by the register_codec tests."""

    x: float
    y: float
    label: str


def test_register_codec_rejects_bad_names():
    with pytest.raises(PersistenceError, match="already registered"):
        register_codec("int", lambda v, path: v, lambda d: d)
    with pytest.raises(PersistenceError, match="non-empty string"):
        register_codec("", lambda v, path: v, lambda d: d)


def test_register_codec_custom_type_dispatch_and_roundtrip():
    name = f"{_CodecPoint.__module__}.{_CodecPoint.__qualname__}"
    register_codec(
        name,
        lambda p, path: {"x": p.x, "y": p.y, "label": p.label},
        lambda d: _CodecPoint(d["x"], d["y"], d["label"]),
    )
    try:
        point = _CodecPoint(1.5, -2.25, "pt")
        envelope = encode_value(point)
        assert envelope["type"] == name
        assert envelope["data"] == {"x": 1.5, "y": -2.25, "label": "pt"}
        json.dumps(envelope)
        assert decode_value(envelope) == point

        # dispatch also applies inside containers
        nested = roundtrip({"points": [point, _CodecPoint(0.0, 0.0, "origin")]})
        assert nested["points"][0] == point
    finally:
        # the global registry must not leak between tests
        codec_mod._CODECS.pop(name, None)


def test_register_codec_registry_has_no_leak():
    name = f"{_CodecPoint.__module__}.{_CodecPoint.__qualname__}"
    assert name not in codec_mod._CODECS
