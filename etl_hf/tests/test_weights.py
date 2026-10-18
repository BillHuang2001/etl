"""Tests for :mod:`etl_hf.weights` — checkpoint readers.

Everything except the single network-guarded real-checkpoint test is offline:
synthetic safetensors files are written with ``struct`` + ``json``, and synthetic
torch checkpoints (zip *and* legacy formats) are written with a ``pickle``
``Pickler`` whose ``persistent_id`` emits real torch storage persistent ids and
whose rebuilt tensors reference real ``torch._utils._rebuild_tensor_v2`` globals.

The torch globals are produced by registering fake ``torch`` / ``torch._utils``
module objects in ``sys.modules`` for the duration of the pickle (see
:func:`_fake_torch_modules`) — no torch installation is needed, and the fake
modules are always removed again.
"""

from __future__ import annotations

import collections
import contextlib
import http.client
import io
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import types
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

from etl_hf.errors import HFWeightError
from etl_hf.weights import (
    _LEGACY_MAGIC,
    _LEGACY_PROTOCOL,
    _RestrictedUnpickler,
    load_checkpoint,
    read_safetensors,
    read_torch_bin,
    state_dict_parameter_count,
    widen_to_float32,
)

# ---------------------------------------------------------------------------
# synthetic safetensors writer
# ---------------------------------------------------------------------------


def _safetensors_bytes(header: dict, payload: bytes) -> bytes:
    """Assemble a safetensors blob from an explicit header dict and payload."""
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)  # the real writer pads the header to 8 bytes
    return struct.pack("<Q", len(blob)) + blob + payload


def write_safetensors_raw(path, header: dict, payload: bytes = b"") -> Path:
    """Write a safetensors file with a caller-supplied (possibly invalid) header."""
    path = Path(path)
    path.write_bytes(_safetensors_bytes(header, payload))
    return path


def write_safetensors(path, entries: dict[str, tuple[np.ndarray, str]], *, metadata=None) -> Path:
    """Write a valid synthetic safetensors file.

    ``entries`` maps tensor name → ``(array, safetensors dtype tag)``; the array's
    raw ``tobytes()`` order is the stored payload (so pass ``uint16`` bit patterns
    for bf16).
    """
    header: dict = {"__metadata__": metadata} if metadata is not None else {}
    payload = bytearray()
    for name, (array, tag) in entries.items():
        data = np.ascontiguousarray(array).tobytes()
        begin = len(payload)
        payload += data
        header[name] = {
            "dtype": tag,
            "shape": list(np.shape(array)),
            "data_offsets": [begin, len(payload)],
        }
    return write_safetensors_raw(path, header, bytes(payload))


# ---------------------------------------------------------------------------
# synthetic torch checkpoint writer (zip + legacy), torch-free
# ---------------------------------------------------------------------------


class _StorageRef:
    """Placeholder for a torch storage; turned into a torch persistent id."""

    def __init__(self, storage_type, key: str, numel: int):
        self.storage_type = storage_type
        self.key = key
        self.numel = int(numel)


class _TensorFixture:
    """Pickles as ``torch._utils._rebuild_tensor_v2(storage, offset, size, stride, …)``."""

    def __init__(self, storage, offset, size, stride, rebuild):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride
        self._rebuild = rebuild

    def __reduce__(self):
        return (
            self._rebuild,
            (self.storage, self.offset, self.size, self.stride, False, collections.OrderedDict()),
        )


class _ParameterFixture:
    """Pickles as ``torch._utils._rebuild_parameter(tensor, requires_grad, hooks)``."""

    def __init__(self, inner, rebuild_parameter):
        self.inner = inner
        self._rebuild = rebuild_parameter

    def __reduce__(self):
        return (self._rebuild, (self.inner, False, collections.OrderedDict()))


class _FixturePickler(pickle.Pickler):
    """Pickler whose ``persistent_id`` emits torch storage persistent ids."""

    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", obj.storage_type, obj.key, "cpu", obj.numel)
        return None


@contextlib.contextmanager
def _fake_torch_modules():
    """Register fake ``torch``/``torch._utils`` modules so pickling emits torch globals.

    ``pickle`` stores callables and classes by reference (``module.qualname``) and
    verifies the reference against ``sys.modules``; these stand-ins make
    ``_rebuild_tensor_v2`` and ``torch.FloatStorage`` picklable without torch.
    The previous ``sys.modules`` entries are restored on exit.
    """
    saved = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch._utils")

    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        raise AssertionError("fixture rebuild helper must never be called")

    def rebuild_tensor(storage, storage_offset, size, stride, *args):
        raise AssertionError("fixture rebuild helper must never be called")

    def rebuild_parameter(data, *args):
        raise AssertionError("fixture rebuild helper must never be called")

    for function, name in (
        (rebuild_tensor_v2, "_rebuild_tensor_v2"),
        (rebuild_tensor, "_rebuild_tensor"),
        (rebuild_parameter, "_rebuild_parameter"),
    ):
        function.__module__ = "torch._utils"
        function.__qualname__ = name
        setattr(utils_module, name, function)

    class Size(tuple):
        """Stand-in for ``torch.Size`` (a tuple subclass, like the real one)."""

    Size.__module__ = "torch"
    Size.__qualname__ = "Size"
    torch_module.Size = Size

    for value_name in ("device", "dtype", "Tensor"):
        value_type = type(value_name, (), {})
        value_type.__module__ = "torch"
        value_type.__qualname__ = value_name
        setattr(torch_module, value_name, value_type)

    storage_types: dict[str, type] = {}

    def storage_type(name: str) -> type:
        """Create (once) and return a fake ``torch.<name>`` storage class."""
        if name not in storage_types:
            cls = type(name, (), {})
            cls.__module__ = "torch"
            cls.__qualname__ = name
            setattr(torch_module, name, cls)
            storage_types[name] = cls
        return storage_types[name]

    sys.modules["torch"] = torch_module
    sys.modules["torch._utils"] = utils_module
    try:
        yield types.SimpleNamespace(
            rebuild=rebuild_tensor_v2,
            rebuild_parameter=rebuild_parameter,
            size_type=Size,
            torch_module=torch_module,
            storage_type=storage_type,
        )
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def tensor_spec(
    name, flat, size=None, stride=None, offset=0, storage="FloatStorage", wrapper="v2",
    storage_key=None,
):
    """Describe one tensor view over a fresh flat storage.

    ``storage_key`` makes the view reuse an existing storage key instead of
    allocating its own (``str(index)``) — i.e. several tensors sharing one
    storage blob, which is what torch produces for a tensor and its transpose.
    """
    flat = np.ascontiguousarray(flat)
    if size is None:
        size = flat.shape
    size = tuple(int(s) for s in size)
    if stride is None:
        stride = contiguous_stride(size)
    return {
        "name": name,
        "flat": flat,
        "size": size,
        "stride": tuple(int(s) for s in stride),
        "offset": int(offset),
        "storage": storage,
        "wrapper": wrapper,
        "storage_key": storage_key,
    }

def contiguous_stride(size) -> tuple[int, ...]:
    """Row-major (C-order) element strides for ``size``."""
    stride = []
    running = 1
    for dim in reversed(size):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def expected_view(spec) -> np.ndarray:
    """The array a reader must produce for ``spec`` (ignoring the dtype policy)."""
    flat = spec["flat"]
    out = np.zeros(spec["size"], dtype=flat.dtype)
    for index in np.ndindex(*spec["size"]):
        offset = spec["offset"] + sum(i * s for i, s in zip(index, spec["stride"]))
        out[index] = flat[offset]
    return out


def _build_state(specs, api, extra_entries=None):
    """Build a picklable state dict + the raw storage blobs it references.

    Returns ``(state, storages, numels)``: ``storages`` maps storage key → raw
    little-endian bytes, ``numels`` the element count of each storage (the
    legacy format's per-blob prefix). Specs sharing a ``storage_key`` share one
    blob.
    """
    state = collections.OrderedDict()
    storages: dict[str, bytes] = {}
    numels: dict[str, int] = {}
    for index, spec in enumerate(specs):
        key = str(spec.get("storage_key") or index)
        flat = np.ascontiguousarray(spec["flat"])
        if key in storages:
            if storages[key] != flat.tobytes():
                raise AssertionError(
                    f"test fixture bug: views over shared storage {key!r} disagree on the "
                    f"flat data"
                )
        else:
            storages[key] = flat.tobytes()
            numels[key] = int(flat.size)
        ref = _StorageRef(api.storage_type(spec["storage"]), key, int(flat.size))
        tensor = _TensorFixture(
            ref, spec["offset"], api.size_type(spec["size"]), api.size_type(spec["stride"]), api.rebuild
        )
        if spec["wrapper"] == "parameter":
            tensor = _ParameterFixture(tensor, api.rebuild_parameter)
        state[spec["name"]] = tensor
    for name, value in (extra_entries or {}).items():
        state[name] = value
    return state, storages, numels

def _write_zip_archive(path, data_pkl: bytes, storages, *, prefix="archive", drop=()) -> Path:
    """Write a torch-1.6-style zip archive around an already-pickled ``data.pkl``."""
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(f"{prefix}/data.pkl", data_pkl)
        for key, raw in storages.items():
            if f"data/{key}" in drop:
                continue
            archive.writestr(f"{prefix}/data/{key}", raw)
        archive.writestr(f"{prefix}/version", "3\n")
    return path


def write_torch_zip(
    path,
    specs,
    *,
    header=True,
    protocol=_LEGACY_PROTOCOL,
    wrap_state_dict=False,
    extra_entries=None,
    prefix="archive",
    drop=(),
) -> Path:
    """Write a synthetic zip-format torch checkpoint (torch ≥ 1.6 layout)."""
    with _fake_torch_modules() as api:
        state, storages, _numels = _build_state(specs, api, extra_entries)
        outer = {"state_dict": state, "epoch": 3} if wrap_state_dict else state
        stream = io.BytesIO()
        pickler = _FixturePickler(stream, protocol=2)
        if header:
            pickler.dump(_LEGACY_MAGIC)
            pickler.dump(protocol)
            pickler.dump({"protocol_version": protocol, "little_endian": True})
        pickler.dump(outer)
    return _write_zip_archive(path, stream.getvalue(), storages, prefix=prefix, drop=drop)


def write_torch_zip_with_pickle(path, obj) -> Path:
    """Write a zip archive whose ``data.pkl`` is ``obj`` pickled with plain pickle."""
    return _write_zip_archive(path, pickle.dumps(obj), {})


def write_torch_legacy(
    path,
    specs,
    *,
    extra_entries=None,
    order=None,
    truncate_last=0,
    trailer_key=None,
    declared_counts=None,
    truncate=None,
    trailing=b"",
) -> Path:
    """Write a synthetic legacy torch checkpoint, mirroring torch's layout exactly.

    The format (torch's ``_legacy_save``) is::

        int64 magic (0x1950A86A20F9469CFC6C)   # pickled
        int 1001                               # pickled protocol version
        dict sys_info                          # pickled
        state dict                             # pickled (tensors via persistent ids)
        list of storage keys                   # pickled
        per key: int64 element count + raw little-endian storage bytes

    The ``int64`` element-count prefix is what
    ``StorageMethods._write_file(..., save_size=True)`` writes with
    ``struct.pack("=q", storage.size())`` — the zip format has no such prefix.

    ``declared_counts`` overrides the written element count for a key (corruption
    tests), ``truncate`` chops bytes off the very end of the file, and
    ``trailing`` appends junk after the last storage.
    """
    path = Path(path)
    with _fake_torch_modules() as api:
        state, storages, numels = _build_state(specs, api, extra_entries)
        keys = list(storages) if order is None else list(order)
        if trailer_key is not None:
            keys = keys + [trailer_key]
        declared = dict(declared_counts or {})
        with open(path, "wb") as handle:
            pickle.dump(_LEGACY_MAGIC, handle, protocol=2)
            pickle.dump(_LEGACY_PROTOCOL, handle, protocol=2)
            pickle.dump({"protocol_version": _LEGACY_PROTOCOL, "little_endian": True}, handle, protocol=2)
            _FixturePickler(handle, protocol=2).dump(state)
            pickle.dump(keys, handle, protocol=2)
            for position, key in enumerate(keys):
                raw = storages.get(key, b"")
                if truncate_last and position == len(keys) - 1:
                    raw = raw[: max(0, len(raw) - truncate_last)]
                handle.write(struct.pack("<q", declared.get(key, numels.get(key, 0))))
                handle.write(raw)
            handle.write(trailing)
    if truncate:
        path.write_bytes(path.read_bytes()[: -truncate])
    return path

# ---------------------------------------------------------------------------
# safetensors: happy paths
# ---------------------------------------------------------------------------


def test_safetensors_roundtrip_preserves_dtypes_and_values(tmp_path):
    f32 = np.array([[1.5, -2.25], [3.0, 4.5]], dtype=np.float32)
    f64 = np.array([1.0 / 3.0, 2.0 / 3.0], dtype=np.float64)
    i32 = np.array([[1, -2], [3, -4]], dtype=np.int32)
    bf16_words = np.array([0x3F80, 0xBF80, 0x4000, 0x0000], dtype=np.uint16)  # 1, -1, 2, 0
    f16 = np.array([0.5, -1.5, 65504.0], dtype=np.float16)
    i64 = np.array([2**40, -(2**40)], dtype=np.int64)
    booleans = np.array([True, False, True])
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {
            "f32": (f32, "F32"),
            "f64": (f64, "F64"),
            "i32": (i32, "I32"),
            "bf16": (bf16_words, "BF16"),
            "f16": (f16, "F16"),
            "i64": (i64, "I64"),
            "bool": (booleans, "BOOL"),
        },
    )
    tensors = read_safetensors(path)

    assert set(tensors) == {"f32", "f64", "i32", "bf16", "f16", "i64", "bool"}
    np.testing.assert_array_equal(tensors["f32"], f32)
    np.testing.assert_array_equal(tensors["f64"], f64)
    np.testing.assert_array_equal(tensors["i32"], i32)
    np.testing.assert_array_equal(tensors["i64"], i64)
    np.testing.assert_array_equal(tensors["bool"], booleans)
    # bf16 0x3F80 is exactly 1.0, 0xBF80 is -1.0, 0x4000 is 2.0.
    assert tensors["bf16"].dtype == np.float32
    np.testing.assert_array_equal(tensors["bf16"], np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32))
    # f16 is widened to f32, exactly.
    assert tensors["f16"].dtype == np.float32
    np.testing.assert_array_equal(tensors["f16"], f16.astype(np.float32))


def test_safetensors_scalar_and_empty_shapes(tmp_path):
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"scalar": (np.array(7.5, dtype=np.float32), "F32"), "empty": (np.zeros((0, 3), np.float32), "F32")},
    )
    tensors = read_safetensors(path)
    assert tensors["scalar"].shape == ()
    assert tensors["scalar"] == np.float32(7.5)
    assert tensors["empty"].shape == (0, 3)


def test_safetensors_skips_metadata_and_returns_writable_arrays(tmp_path):
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {"w": (np.arange(6, dtype=np.float32).reshape(2, 3), "F32")},
        metadata={"format": "pt", "author": "etl_hf"},
    )
    tensors = read_safetensors(path)
    assert set(tensors) == {"w"}
    array = tensors["w"]
    assert array.flags.writeable and array.flags.c_contiguous and array.flags.owndata
    array[0, 0] = 42.0
    assert array[0, 0] == 42.0


def test_safetensors_corrupt_header_cases(tmp_path):
    good = np.arange(4, dtype=np.float32).tobytes()
    # header length far beyond the file
    truncated = tmp_path / "short.safetensors"
    truncated.write_bytes(struct.pack("<Q", 10**6) + b"{}")
    with pytest.raises(HFWeightError) as err:
        read_safetensors(truncated)
    assert "header length" in str(err.value)
    # file shorter than the 8-byte length prefix
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"\x01\x02")
    with pytest.raises(HFWeightError) as err:
        read_safetensors(tiny)
    assert "too short" in str(err.value)
    # header is valid JSON but not an object
    write_safetensors_raw(tmp_path / "list.safetensors", [1, 2], good)
    with pytest.raises(HFWeightError) as err:
        read_safetensors(tmp_path / "list.safetensors")
    assert "JSON object" in str(err.value)
    # header is not JSON at all
    (tmp_path / "garbage.safetensors").write_bytes(struct.pack("<Q", 8) + b"not json" + good)
    with pytest.raises(HFWeightError) as err:
        read_safetensors(tmp_path / "garbage.safetensors")
    assert "JSON header" in str(err.value)


def test_safetensors_unknown_dtype_is_named(tmp_path):
    path = write_safetensors_raw(
        tmp_path / "model.safetensors",
        {"w": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [0, 2]}},
        b"\x00\x01",
    )
    with pytest.raises(HFWeightError) as err:
        read_safetensors(path)
    assert "F8_E4M3" in str(err.value)
    assert "'w'" in str(err.value)


def test_safetensors_offset_and_shape_validation(tmp_path):
    payload = np.arange(4, dtype=np.float32).tobytes()
    # data_offsets beyond the data buffer
    path = write_safetensors_raw(
        tmp_path / "beyond.safetensors",
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4096]}},
        payload,
    )
    with pytest.raises(HFWeightError) as err:
        read_safetensors(path)
    assert "data_offsets" in str(err.value) and "w" in str(err.value)
    # element count does not match the byte span
    path = write_safetensors_raw(
        tmp_path / "mismatch.safetensors",
        {"w": {"dtype": "F32", "shape": [8], "data_offsets": [0, 16]}},
        payload,
    )
    with pytest.raises(HFWeightError) as err:
        read_safetensors(path)
    assert "8 bytes" in str(err.value) or "32 bytes" in str(err.value)
    # inverted offsets
    path = write_safetensors_raw(
        tmp_path / "inverted.safetensors",
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [8, 0]}},
        payload,
    )
    with pytest.raises(HFWeightError) as err:
        read_safetensors(path)
    assert "invalid data offsets" in str(err.value)
    # negative dimension
    path = write_safetensors_raw(
        tmp_path / "negdim.safetensors",
        {"w": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}},
        payload,
    )
    with pytest.raises(HFWeightError) as err:
        read_safetensors(path)
    assert "invalid dimension" in str(err.value)


# ---------------------------------------------------------------------------
# torch zip format
# ---------------------------------------------------------------------------


def test_torch_zip_header_and_plain_tensors(tmp_path):
    """The three-pickle header (magic + protocol + sysinfo) must be skipped."""
    weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    ids = np.array([5, 6, 7, 8], dtype=np.int64)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin",
        [
            tensor_spec("embed.weight", weights),
            tensor_spec("ids", ids, storage="LongStorage"),
        ],
    )
    state = read_torch_bin(path)
    assert set(state) == {"embed.weight", "ids"}
    np.testing.assert_array_equal(state["embed.weight"], weights)
    np.testing.assert_array_equal(state["ids"], ids)
    assert state["ids"].dtype == np.int64


def test_torch_zip_offset_and_stride_view(tmp_path):
    """A transposed/offset view must be materialized through offset + stride."""
    flat = np.arange(12, dtype=np.float32)
    spec = tensor_spec("w", flat, size=(2, 3), stride=(1, 2), offset=1)
    path = write_torch_zip(tmp_path / "pytorch_model.bin", [spec])
    state = read_torch_bin(path)
    # flat[1 + i*1 + j*2] → [[1, 3, 5], [2, 4, 6]]
    np.testing.assert_array_equal(state["w"], np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32))
    np.testing.assert_array_equal(state["w"], expected_view(spec))
    assert state["w"].flags.writeable and state["w"].flags.c_contiguous


def test_torch_zip_negative_stride_view(tmp_path):
    flat = np.arange(6, dtype=np.float32)
    spec = tensor_spec("w", flat, size=(3,), stride=(-1,), offset=5)
    state = read_torch_bin(write_torch_zip(tmp_path / "pytorch_model.bin", [spec]))
    np.testing.assert_array_equal(state["w"], np.array([5, 4, 3], dtype=np.float32))


def test_torch_zip_empty_tensor(tmp_path):
    spec = tensor_spec("w", np.zeros(0, dtype=np.float32), size=(0, 4))
    state = read_torch_bin(write_torch_zip(tmp_path / "pytorch_model.bin", [spec]))
    assert state["w"].shape == (0, 4)


def test_torch_zip_bfloat16_and_float16_storages(tmp_path):
    bf16_words = np.array([0x3F80, 0xBF80, 0x4000], dtype=np.uint16)
    half = np.array([0.5, -2.0], dtype=np.float16)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin",
        [
            tensor_spec("bf16", bf16_words, storage="BFloat16Storage"),
            tensor_spec("half", half, storage="HalfStorage"),
        ],
    )
    state = read_torch_bin(path)
    np.testing.assert_array_equal(state["bf16"], np.array([1.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(state["half"], half.astype(np.float32))


def test_torch_zip_parameter_wrapper_is_unwrapped(tmp_path):
    weights = np.arange(4, dtype=np.float32)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("w", weights, wrapper="parameter")]
    )
    np.testing.assert_array_equal(read_torch_bin(path)["w"], weights)


def test_torch_zip_without_header_is_tolerated(tmp_path):
    weights = np.arange(4, dtype=np.float32)
    path = write_torch_zip(tmp_path / "pytorch_model.bin", [tensor_spec("w", weights)], header=False)
    np.testing.assert_array_equal(read_torch_bin(path)["w"], weights)


def test_torch_zip_state_dict_wrapper_is_unwrapped(tmp_path):
    weights = np.arange(3, dtype=np.float32)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("w", weights)], wrap_state_dict=True
    )
    assert set(read_torch_bin(path)) == {"w"}


def test_torch_zip_bad_protocol_version_is_reported(tmp_path):
    weights = np.arange(2, dtype=np.float32)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("w", weights)], protocol=999
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "999" in str(err.value)
    assert "protocol" in str(err.value)


def test_torch_zip_missing_storage_blob(tmp_path):
    weights = np.arange(2, dtype=np.float32)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("w", weights)], drop=("data/0",)
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "missing storage blob" in str(err.value)


def test_torch_zip_unknown_storage_type(tmp_path):
    weights = np.arange(2, dtype=np.float32)
    path = write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("w", weights, storage="ComplexFloatStorage")]
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "ComplexFloatStorage" in str(err.value)


def test_torch_zip_rejects_arbitrary_pickled_globals(tmp_path):
    """A checkpoint must never execute arbitrary pickled code."""
    path = write_torch_zip_with_pickle(tmp_path / "pytorch_model.bin", {"w": os.system})
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "refusing to unpickle" in str(err.value)
    assert "system" in str(err.value)

    path = write_torch_zip_with_pickle(tmp_path / "pytorch_model2.bin", {"w": Path("/etc/passwd")})
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "refusing to unpickle" in str(err.value)


def test_torch_zip_non_dict_and_non_tensor_entries(tmp_path):
    path = write_torch_zip_with_pickle(tmp_path / "list.bin", [1, 2, 3])
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "expected a state dict" in str(err.value)

    path = write_torch_zip(tmp_path / "scalar.bin", [], extra_entries={"bogus": 5})
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "bogus" in str(err.value)
    assert "expected a tensor" in str(err.value)


def test_torch_zip_inert_torch_values_are_accepted_as_globals(tmp_path):
    """``torch.device``-style globals are inert: no code runs, they just aren't tensors."""
    with _fake_torch_modules() as api:
        device_class = api.torch_module.device
        data_pkl = pickle.dumps({"w": device_class})
    path = _write_zip_archive(tmp_path / "device.bin", data_pkl, {})
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    message = str(err.value)
    assert "expected a tensor" in message
    assert "refusing to unpickle" not in message


def test_inert_torch_size_unpickles_through_copyreg():
    """Protocol-0/1 pickles of ``torch.Size`` go through ``copyreg._reconstructor``."""
    with _fake_torch_modules() as api:
        payload = pickle.dumps(api.size_type((2, 3)), protocol=0)
    assert b"copy_reg" in payload
    unpickler = _RestrictedUnpickler(io.BytesIO(payload), source="<size>")
    assert tuple(unpickler.load()) == (2, 3)


# ---------------------------------------------------------------------------
# torch legacy format
#
# The legacy layout is torch's ``_legacy_save``: pickled magic, pickled protocol
# int ``1001``, pickled ``sys_info`` dict, the pickled state dict, the pickled
# list of storage keys, then per key an 8-byte little-endian int64 element count
# (``StorageMethods._write_file(..., save_size=True)`` →
# ``struct.pack("=q", storage.size())``) followed by that storage's raw bytes.
# ``write_torch_legacy`` above emits exactly that; **no test here requires
# torch** — the ``torch.*`` globals are fake modules registered only while
# pickling (see ``_fake_torch_modules``).
# ---------------------------------------------------------------------------


def test_torch_legacy_roundtrip(tmp_path):
    weights = np.arange(6, dtype=np.float32).reshape(3, 2)
    ids = np.arange(3, dtype=np.int64)
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [
            tensor_spec("w", weights),
            tensor_spec("ids", ids, storage="LongStorage"),
            tensor_spec("view", np.arange(12, dtype=np.float32), size=(2, 3), stride=(1, 2), offset=1),
        ],
    )
    state = read_torch_bin(path)
    assert set(state) == {"w", "ids", "view"}
    np.testing.assert_array_equal(state["w"], weights)
    np.testing.assert_array_equal(state["ids"], ids)
    np.testing.assert_array_equal(state["view"], np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32))


def test_torch_legacy_single_f32_storage_exact_values(tmp_path):
    """One float32 storage: the per-storage int64 prefix must be consumed, not read as data."""
    values = np.array([1.0, -0.0, 0.5, -2.25, 1e-38], dtype=np.float32)
    path = write_torch_legacy(tmp_path / "pytorch_model.bin", [tensor_spec("w", values)])
    state = read_torch_bin(path)
    assert state["w"].dtype == np.float32
    np.testing.assert_array_equal(state["w"], values)
    # Bit-exact, including the negative zero.
    assert state["w"].tobytes() == values.tobytes()


def test_torch_legacy_two_storages_mixed_dtypes_exact_values(tmp_path):
    """Two storages of different dtypes: the second blob must not be shifted by 8 bytes."""
    f32 = np.array([0.25, -1.5, 3.75], dtype=np.float32)
    i64 = np.array([-(2**40), 0, 7, 2**40 + 1], dtype=np.int64)
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [
            tensor_spec("f", f32),
            tensor_spec("i", i64, storage="LongStorage"),
        ],
    )
    state = read_torch_bin(path)
    assert state["f"].dtype == np.float32
    assert state["i"].dtype == np.int64
    np.testing.assert_array_equal(state["f"], f32)
    np.testing.assert_array_equal(state["i"], i64)


def test_torch_legacy_shared_storage_offset_and_negative_stride(tmp_path):
    """Two views over ONE storage: a stored offset plus a transposed and a reversed stride."""
    flat = np.arange(12, dtype=np.float32)
    base = tensor_spec("base", flat, size=(2, 4), stride=(4, 1), offset=2, storage_key="0")
    transposed = tensor_spec("t", flat, size=(4, 2), stride=(1, 4), offset=2, storage_key="0")
    reversed_view = tensor_spec("r", flat, size=(2, 4), stride=(-4, 1), offset=6, storage_key="0")
    path = write_torch_legacy(tmp_path / "pytorch_model.bin", [base, transposed, reversed_view])

    # The storage blob is written exactly once, with one int64 prefix.
    blob = path.read_bytes()
    assert blob.endswith(struct.pack("<q", flat.size) + flat.tobytes())

    state = read_torch_bin(path)
    for spec in (base, transposed, reversed_view):
        np.testing.assert_array_equal(state[spec["name"]], expected_view(spec))
    np.testing.assert_array_equal(state["base"], np.array([[2, 3, 4, 5], [6, 7, 8, 9]], np.float32))
    np.testing.assert_array_equal(state["t"], np.array([[2, 6], [3, 7], [4, 8], [5, 9]], np.float32))
    np.testing.assert_array_equal(state["r"], np.array([[6, 7, 8, 9], [2, 3, 4, 5]], np.float32))


def test_torch_legacy_zero_element_storage(tmp_path):
    """A 0-element storage writes ``declared == 0`` and no data bytes."""
    empty = np.zeros((0,), dtype=np.float32)
    other = np.array([1.5, -2.5], dtype=np.float32)
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("empty", empty, size=(0,)), tensor_spec("w", other)],
    )
    assert struct.pack("<q", 0) + struct.pack("<q", 2) + other.tobytes() in path.read_bytes()
    state = read_torch_bin(path)
    assert state["empty"].shape == (0,)
    assert state["empty"].dtype == np.float32
    np.testing.assert_array_equal(state["w"], other)


def test_torch_legacy_truncated_storage_bytes(tmp_path):
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("w", np.arange(4, dtype=np.float32))],
        truncate_last=4,
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "truncated" in str(err.value)


def test_torch_legacy_truncated_element_count_prefix(tmp_path):
    """A cut that eats into the int64 prefix is an explicit error, not garbage data."""
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("w", np.arange(4, dtype=np.float32))],
        truncate=20,  # 16 data bytes + 4 of the 8-byte prefix
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    message = str(err.value)
    assert "0" in message and "element-count prefix" in message


def test_torch_legacy_wrong_declared_element_count(tmp_path):
    """A prefix disagreeing with the pickle stream is rejected, naming the storage key."""
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("w", np.arange(4, dtype=np.float32))],
        declared_counts={"0": 3},
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    message = str(err.value)
    assert "'0'" in message
    assert "3 elements" in message and "4" in message


def test_torch_legacy_trailing_junk_after_last_storage(tmp_path):
    """Unconsumed bytes after the last blob must fail loudly (a mis-parse is never silent)."""
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("w", np.arange(2, dtype=np.float32))],
        trailing=b"\x00" * 4,
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    message = str(err.value)
    assert "4 unexpected trailing bytes" in message


def test_torch_legacy_unknown_storage_key_in_trailer(tmp_path):
    path = write_torch_legacy(
        tmp_path / "pytorch_model.bin",
        [tensor_spec("w", np.arange(2, dtype=np.float32))],
        trailer_key="99",
    )
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert "99" in str(err.value)


def test_torch_legacy_without_element_count_prefix_is_rejected(tmp_path):
    """The pre-fix (prefix-less) layout must now fail explicitly instead of shifting bytes."""
    values = np.arange(4, dtype=np.float32)
    path = tmp_path / "pytorch_model.bin"
    with _fake_torch_modules() as api:
        state, _storages, numels = _build_state([tensor_spec("w", values)], api)
        with open(path, "wb") as handle:
            pickle.dump(_LEGACY_MAGIC, handle, protocol=2)
            pickle.dump(_LEGACY_PROTOCOL, handle, protocol=2)
            pickle.dump({"protocol_version": _LEGACY_PROTOCOL, "little_endian": True}, handle, protocol=2)
            _FixturePickler(handle, protocol=2).dump(state)
            pickle.dump(list(numels), handle, protocol=2)
            for key in numels:
                handle.write(values.tobytes())  # no int64 element count
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    message = str(err.value)
    assert "declares" in message and "'0'" in message


def test_torch_reader_rejects_non_checkpoint_file(tmp_path):
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(b"definitely not a checkpoint")
    with pytest.raises(HFWeightError) as err:
        read_torch_bin(path)
    assert str(path) in str(err.value)


# ---------------------------------------------------------------------------
# directory / sharded loading
# ---------------------------------------------------------------------------


def test_load_checkpoint_safetensors_dir(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"a": (np.ones(2, np.float32), "F32")})
    state = load_checkpoint(tmp_path)
    assert set(state) == {"a"}


def test_load_checkpoint_prefers_safetensors_over_torch(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"st": (np.ones(2, np.float32), "F32")})
    write_torch_zip(
        tmp_path / "pytorch_model.bin", [tensor_spec("bin", np.zeros(2, np.float32))]
    )
    assert set(load_checkpoint(tmp_path)) == {"st"}


def test_load_checkpoint_legacy_bin_dir(tmp_path):
    weights = np.arange(4, dtype=np.float32)
    write_torch_legacy(tmp_path / "pytorch_model.bin", [tensor_spec("w", weights)])
    np.testing.assert_array_equal(load_checkpoint(tmp_path)["w"], weights)


def test_load_checkpoint_empty_dir_lists_candidates(tmp_path):
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    message = str(err.value)
    assert str(tmp_path) in message
    for candidate in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                      "pytorch_model.bin.index.json"):
        assert candidate in message


def test_load_checkpoint_missing_path(tmp_path):
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path / "nope")
    assert "not an existing checkpoint" in str(err.value)


def test_load_checkpoint_direct_file_paths(tmp_path):
    weights = np.arange(3, dtype=np.float32)
    safetensors_path = write_safetensors(tmp_path / "weights.safetensors", {"w": (weights, "F32")})
    direct = load_checkpoint(safetensors_path)
    np.testing.assert_array_equal(direct["w"], weights)

    torch_path = write_torch_legacy(tmp_path / "checkpoint.bin", [tensor_spec("w", weights)])
    np.testing.assert_array_equal(load_checkpoint(torch_path)["w"], weights)

    # Unknown suffix: sniffed by content (safetensors first, then torch).
    sniffed = tmp_path / "weights.data"
    sniffed.write_bytes(safetensors_path.read_bytes())
    np.testing.assert_array_equal(load_checkpoint(sniffed)["w"], weights)

    torch_sniffed = tmp_path / "weights.blob"
    torch_sniffed.write_bytes(torch_path.read_bytes())
    np.testing.assert_array_equal(load_checkpoint(torch_sniffed)["w"], weights)

    garbage = tmp_path / "garbage.data"
    garbage.write_bytes(b"nothing to see here")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(garbage)
    assert "not a readable checkpoint" in str(err.value)


def test_load_checkpoint_sharded_safetensors(tmp_path):
    shard_a = {"a": (np.arange(3, dtype=np.float32), "F32")}
    shard_b = {"b": (np.arange(4, dtype=np.float32), "F32"), "c": (np.arange(2, dtype=np.float32), "F32")}
    path_a = write_safetensors(tmp_path / "model-00001-of-00002.safetensors", shard_a)
    path_b = write_safetensors(tmp_path / "model-00002-of-00002.safetensors", shard_b)
    index = {
        "metadata": {"total_size": 36},
        "weight_map": {
            "a": path_a.name,
            "b": path_b.name,
            "c": path_b.name,
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    merged = load_checkpoint(tmp_path)
    direct = {**read_safetensors(path_a), **read_safetensors(path_b)}
    assert set(merged) == {"a", "b", "c"} == set(direct)
    for key in merged:
        np.testing.assert_array_equal(merged[key], direct[key])


def test_load_checkpoint_sharded_missing_shard(tmp_path):
    write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": (np.ones(2, np.float32), "F32")})
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    message = str(err.value)
    assert "model-00002-of-00002.safetensors" in message
    assert str(tmp_path) in message


def test_load_checkpoint_sharded_index_errors(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text("[]", encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    assert "JSON object" in str(err.value)

    index_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    assert "weight_map" in str(err.value)

    index_path.write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}), encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    assert "relative path" in str(err.value)

    index_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    assert "invalid JSON" in str(err.value)


def test_load_checkpoint_sharded_tensor_missing_from_shard(tmp_path):
    write_safetensors(tmp_path / "shard.safetensors", {"a": (np.ones(2, np.float32), "F32")})
    index = {"weight_map": {"a": "shard.safetensors", "b": "shard.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(HFWeightError) as err:
        load_checkpoint(tmp_path)
    assert "'b'" in str(err.value)
    assert "absent from the shard" in str(err.value)


def test_load_checkpoint_sharded_torch_legacy(tmp_path):
    shard_a = write_torch_legacy(
        tmp_path / "pytorch_model-00001-of-00002.bin",
        [tensor_spec("a", np.arange(2, dtype=np.float32))],
    )
    shard_b = write_torch_legacy(
        tmp_path / "pytorch_model-00002-of-00002.bin",
        [tensor_spec("b", np.arange(3, dtype=np.float32))],
    )
    index = {"weight_map": {"a": shard_a.name, "b": shard_b.name}}
    (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps(index), encoding="utf-8")
    merged = load_checkpoint(tmp_path)
    assert set(merged) == {"a", "b"}
    np.testing.assert_array_equal(merged["a"], np.arange(2, dtype=np.float32))
    np.testing.assert_array_equal(merged["b"], np.arange(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# dtype policy helpers
# ---------------------------------------------------------------------------


def test_widen_to_float32_policy():
    half = np.array([0.5, -2.0], dtype=np.float16)
    widened = widen_to_float32(half)
    assert widened.dtype == np.float32
    np.testing.assert_array_equal(widened, half.astype(np.float32))

    f32 = np.arange(3, dtype=np.float32)
    assert widen_to_float32(f32) is not None
    np.testing.assert_array_equal(widen_to_float32(f32), f32)

    f64 = np.arange(3, dtype=np.float64)
    assert widen_to_float32(f64).dtype == np.float64
    ints = np.arange(3, dtype=np.int32)
    assert widen_to_float32(ints).dtype == np.int32
    bools = np.array([True, False])
    assert widen_to_float32(bools).dtype == np.bool_


def test_state_dict_parameter_count():
    state = {
        "a": np.zeros((2, 3), np.float32),
        "b": np.zeros((4,), np.int64),
        "scalar": np.zeros((), np.float32),
    }
    assert state_dict_parameter_count(state) == 6 + 4 + 1
    assert state_dict_parameter_count({}) == 0
    with pytest.raises(HFWeightError) as err:
        state_dict_parameter_count({"bad": [1, 2, 3]})
    assert "bad" in str(err.value)


def test_modules_import_without_optional_dependencies():
    """Importing the reader modules must not pull in torch/transformers/iree/safetensors."""
    code = (
        "import sys\n"
        "import etl_hf.config, etl_hf.weights\n"
        "optional = ('torch', 'transformers', 'iree', 'safetensors')\n"
        "leaked = [name for name in optional if name in sys.modules]\n"
        "print(','.join(leaked))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"leaked modules: {result.stdout.strip()}{result.stderr}"


# ---------------------------------------------------------------------------
# network-guarded real checkpoints
# ---------------------------------------------------------------------------

_HF_BASE = "https://huggingface.co"
_NETWORK_TIMEOUT = 60


def _download(relative_url: str, subdir: str, filename: str):
    """Fetch a HF file into ``$TMPDIR/etl_hf_models/<subdir>``; ``None`` on failure."""
    root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "etl_hf_models" / subdir
    dest = root / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    root.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{_HF_BASE}/{relative_url}", headers={"User-Agent": "etl_hf-tests/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_NETWORK_TIMEOUT) as response:
            payload = response.read()
    except (OSError, http.client.HTTPException):
        # Network guard: any download failure means the test skips instead of
        # failing (HTTPError/URLError are OSError subclasses).
        return None
    if not payload:
        return None
    dest.write_bytes(payload)
    return dest


def test_real_tiny_gpt2_legacy_checkpoint():
    """The real ``sshleifer/tiny-gpt2`` ``pytorch_model.bin`` is in the legacy format."""
    path = _download("sshleifer/tiny-gpt2/resolve/main/pytorch_model.bin", "tiny_gpt2", "pytorch_model.bin")
    if path is None:
        pytest.skip("could not download sshleifer/tiny-gpt2/pytorch_model.bin (offline?)")
    state = load_checkpoint(path)
    assert "transformer.wte.weight" in state
    assert "transformer.wpe.weight" in state
    assert "transformer.h.0.attn.c_attn.weight" in state
    assert "transformer.ln_f.bias" in state
    embed = state["transformer.wte.weight"]
    assert embed.ndim == 2 and embed.shape[1] > 0
    positions = state["transformer.wpe.weight"]
    assert positions.shape[1] == embed.shape[1]
    # GPT-2 Conv1D weights are stored (in, out): c_attn maps hidden → 3 * hidden.
    attn = state["transformer.h.0.attn.c_attn.weight"]
    assert attn.shape == (embed.shape[1], 3 * embed.shape[1])
    # All float tensors stay float32; the only non-float entries are GPT-2's
    # uint8 causal-mask buffers (``attn.bias``), which the dtype policy keeps.
    for name, value in state.items():
        if value.dtype == np.uint8:
            assert name.endswith("attn.bias"), name
        else:
            assert value.dtype == np.float32, (name, value.dtype)
    mask = state["transformer.h.0.attn.bias"]
    assert mask.shape == (1, 1, mask.shape[-1], mask.shape[-1])
    assert state_dict_parameter_count(state) > 0


def test_real_tiny_llama_safetensors_checkpoint():
    """The real ``tiny-random-LlamaForCausalLM`` ``model.safetensors`` reads back intact."""
    path = _download(
        "hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/model.safetensors",
        "tiny_llama",
        "model.safetensors",
    )
    if path is None:
        pytest.skip("could not download tiny-random-LlamaForCausalLM/model.safetensors (offline?)")
    state = load_checkpoint(path)
    assert "model.embed_tokens.weight" in state
    assert "model.norm.weight" in state
    assert "lm_head.weight" in state
    embed = state["model.embed_tokens.weight"]
    q_proj = state["model.layers.0.self_attn.q_proj.weight"]
    assert embed.ndim == 2 and q_proj.ndim == 2
    # Llama linears are stored (out_features, in_features) in HF.
    assert q_proj.shape[1] == embed.shape[1]
    assert state["model.layers.1.self_attn.v_proj.weight"].shape == q_proj.shape
    assert all(value.dtype == np.float32 for value in state.values())
    assert state_dict_parameter_count(state) > 0


def test_real_tiny_gpt2_zip_checkpoint_matches_safetensors():
    """A real zip-format ``pytorch_model.bin`` must match the same repo's safetensors."""
    bin_path = _download(
        "hf-internal-testing/tiny-random-gpt2/resolve/main/pytorch_model.bin",
        "tiny_gpt2_zip",
        "pytorch_model.bin",
    )
    st_path = _download(
        "hf-internal-testing/tiny-random-gpt2/resolve/main/model.safetensors",
        "tiny_gpt2_zip",
        "model.safetensors",
    )
    if bin_path is None or st_path is None:
        pytest.skip("could not download hf-internal-testing/tiny-random-gpt2 (offline?)")
    assert zipfile.is_zipfile(bin_path), "expected the modern torch zip format"
    from_bin = read_torch_bin(bin_path)
    from_st = read_safetensors(st_path)
    shared = set(from_bin) & set(from_st)
    assert shared, "the two formats should share tensor names"
    assert set(from_st) <= set(from_bin)
    for key in sorted(shared):
        assert from_bin[key].dtype == from_st[key].dtype, key
        assert from_bin[key].shape == from_st[key].shape, key
        np.testing.assert_array_equal(from_bin[key], from_st[key])
    # The .bin ships extra buffers the (partial) safetensors file omits, among them
    # GPT-2's uint8 causal-mask buffers.
    extra = set(from_bin) - set(from_st)
    assert extra
    assert "h.0.attn.bias" in extra
    assert from_bin["h.0.attn.bias"].dtype == np.uint8
