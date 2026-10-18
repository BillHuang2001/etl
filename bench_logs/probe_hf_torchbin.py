"""Minimal torch-checkpoint reader — numpy ONLY, no torch.

Handles BOTH torch checkpoint formats so HF models published as
``pytorch_model.bin`` can be loaded with numpy alone:

* **zip format** (torch >= 1.6): a ZIP archive with ``<root>/data.pkl``
  (the pickled state dict) + ``<root>/data/<key>`` (raw storage bytes).
* **legacy format** (torch < 1.6, still served by older HF repos such as
  ``sshleifer/tiny-gpt2``): one pickle stream containing the pickled state
  dict, followed by a pickle of the STORAGE-KEY LIST, followed by the raw
  little-endian bytes of each storage in that key order
  (torch.serialization._legacy_load).

In both formats the pickle refers to storages through persistent ids
``('storage', StorageType, key, location, numel)`` and rebuilds tensors via
``torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)``.
Storages are therefore materialized LAZILY: the unpickler yields
``_TensorRef`` records which are resolved once every storage has its bytes.

Scratch probe helper (bench_logs is a throwaway probe area).
"""

from __future__ import annotations

import collections
import pickle
import zipfile
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "DoubleStorage": np.float64,
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "BFloat16Storage": "bf16",  # special-cased: raw u16 holding bf16 bits
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        if name not in _STORAGE_DTYPES:
            raise NotImplementedError(f"unsupported torch storage type {name!r}")
        self.name = name

    @property
    def dtype(self):
        return _STORAGE_DTYPES[self.name]

    @property
    def itemsize(self) -> int:
        return 2 if self.dtype == "bf16" else np.dtype(self.dtype).itemsize


class _Storage:
    """A storage's raw bytes, filled in after the pickle stream is consumed."""

    def __init__(self, stype: _StorageType, key: str, numel: int):
        self.stype, self.key, self.numel = stype, key, int(numel)
        self.raw: bytes | None = None

    def flat(self) -> np.ndarray:
        assert self.raw is not None, f"storage {self.key} was never filled"
        want = self.numel * self.stype.itemsize
        assert len(self.raw) >= want, f"storage {self.key}: {len(self.raw)} < {want} bytes"
        if self.stype.dtype == "bf16":
            u16 = np.frombuffer(self.raw[:want], dtype="<u2")
            return (u16.astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(self.raw[:want], dtype=np.dtype(self.stype.dtype).newbyteorder("<"))


class _TensorRef:
    """A pending tensor view: resolved to an ndarray once storages are loaded."""

    def __init__(self, storage: _Storage, offset: int, size, stride):
        self.storage, self.offset = storage, int(offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)

    def resolve(self) -> np.ndarray:
        flat = self.storage.flat()
        if 0 in self.size:  # empty tensor
            return np.zeros(self.size, dtype=flat.dtype)
        itemsize = flat.dtype.itemsize
        view = np.lib.stride_tricks.as_strided(
            flat[self.offset :],
            shape=self.size,
            strides=tuple(s * itemsize for s in self.stride),
            writeable=False,
        )
        return np.ascontiguousarray(view).copy()


def _rebuild_tensor(storage, storage_offset, size, stride, *args, **kwargs):
    return _TensorRef(storage, storage_offset, size, stride)


_rebuild_tensor_v2 = _rebuild_tensor


def _rebuild_parameter(data, *args, **kwargs):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh):
        super().__init__(fh)
        self.storages: dict[str, _Storage] = {}

    def find_class(self, module, name):  # noqa: ANN001
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and isinstance(name, str) and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch" and name in ("Size", "device", "dtype", "Tensor"):
            return name
        raise pickle.UnpicklingError(f"unsupported pickle class {module}.{name}")

    def persistent_load(self, pid):  # noqa: ANN001
        kind, stype, key, _location, numel = pid[:5]
        assert kind == "storage", f"unsupported persistent id kind {kind!r}"
        key = str(key)
        st = self.storages.get(key)
        if st is None:
            st = self.storages[key] = _Storage(stype, key, numel)
        return st


def _materialize(obj):
    if isinstance(obj, _TensorRef):
        return obj.resolve()
    if isinstance(obj, dict):
        return {k: _materialize(v) for k, v in obj.items()}
    return obj


def _read_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        pkls = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if len(pkls) != 1:
            raise ValueError(f"{path}: expected one */data.pkl, found {pkls}")
        root = pkls[0][: -len("data.pkl")]
        with zf.open(pkls[0]) as fh:
            unp = _Unpickler(fh)
            obj = unp.load()
        for key, st in unp.storages.items():
            try:
                st.raw = zf.read(f"{root}data/{key}")
            except KeyError:
                st.raw = zf.read(f"{root}data/{int(key) if key.isdigit() else key}")
    return _materialize(obj)


_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_LEGACY_PROTOCOL = 1001


def _read_legacy(path: Path) -> dict:
    with open(path, "rb") as fh:
        # torch's legacy header: magic number, protocol version, sys_info dict
        magic = pickle.load(fh)
        if magic != _LEGACY_MAGIC:
            raise ValueError(f"{path}: bad legacy magic number {magic!r} (corrupt file?)")
        proto = pickle.load(fh)
        if proto != _LEGACY_PROTOCOL:
            raise ValueError(f"{path}: unsupported legacy protocol version {proto!r}")
        _sys_info = pickle.load(fh)
        unp = _Unpickler(fh)
        obj = unp.load()
        keys = pickle.load(fh)  # the serialized STORAGE-KEY LIST
        assert isinstance(keys, list), f"legacy trailer is not a key list: {type(keys)}"
        for key in keys:
            st = unp.storages[str(key)]
            nbytes = st.numel * st.stype.itemsize
            raw = fh.read(nbytes)
            assert len(raw) == nbytes, f"legacy {path}: short read for storage {key}"
            st.raw = raw
    return _materialize(obj)


def read_torch_bin(path: str | Path) -> dict[str, np.ndarray]:
    """Read a torch checkpoint (zip or legacy) into ``{name: ndarray}``."""
    path = Path(path)
    try:
        obj = _read_zip(path)
    except zipfile.BadZipFile:
        obj = _read_legacy(path)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: unpickled object is {type(obj)}, expected a dict")
    return {k: v for k, v in obj.items() if isinstance(v, np.ndarray)}


if __name__ == "__main__":
    import os
    import sys

    root = Path(os.environ["TMPDIR"]) / "hf_models"
    for name in sys.argv[1:] or ["tiny_gpt2", "tiny_mistral"]:
        d = root / name
        cands = sorted(d.glob("*.bin")) if d.is_dir() else []
        if not cands:
            print(f"[{name}] no .bin in {d}")
            continue
        sd = read_torch_bin(cands[0])
        total = sum(int(np.prod(v.shape)) for v in sd.values())
        print(f"[{name}] {len(sd)} tensors, {total} params")
        for k in sorted(sd):
            v = sd[k]
            print(f"  {k:60s} {str(tuple(v.shape)):20s} {v.dtype}")
