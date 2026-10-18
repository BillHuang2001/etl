"""Checkpoint readers for ``etl_hf`` — safetensors and both torch formats.

Three readers, all pure ``numpy`` + stdlib (no ``torch``, no ``safetensors``
package, no etl tensor API):

* :func:`read_safetensors` — the safetensors container
  (``u64 LE header length`` + JSON header + raw tensor buffer).
* :func:`read_torch_bin` — ``pytorch_model.bin``, in **both** on-disk formats:
  the zip format (torch ≥ 1.6: ``*/data.pkl`` + ``*/data/<key>`` storage blobs)
  and the legacy format (magic number + protocol 1001 + ``sys_info`` + the
  pickled state dict + the pickled storage-key list + the raw storage bytes in
  that key order).
* :func:`load_checkpoint` — a model directory or file, including sharded
  checkpoints via ``model.safetensors.index.json`` /
  ``pytorch_model.bin.index.json``.

**What this module does NOT do:** it performs *no* key mapping, *no*
transposition and *no* weight-tree assembly. In particular, the
``(in_features, out_features)`` orientation convention that the etl graphs need
for ``etl.dot(x, W)`` is applied **later**, by the per-family weight
canonicalisers (``etl_hf.models.<family>.canonicalize_weights``) — they know
which keys are ``(out, in)`` in HF (llama linears) and which are already
``(in, out)`` (GPT-2 ``Conv1D``). Readers return raw checkpoint tensors keyed
by their on-disk names.

**Dtype policy** (:func:`widen_to_float32`): ``float32`` preserved, ``float16``
and ``bfloat16`` widened to ``float32`` (bfloat16 via the ``u16 -> u32 << 16 ->
f32`` bit trick), ``float64``, integer and bool dtypes kept. Every returned
array is a normal, writable, C-contiguous ``numpy`` array — never a pickled
object, and ``allow_pickle`` is never used.

**Safety:** reading ``pytorch_model.bin`` unpickles data, so the unpickler is
restricted (:class:`_RestrictedUnpickler`): it can rebuild tensors from
storages and nothing else. Any other global reference raises
:class:`~etl_hf.errors.HFWeightError` instead of executing arbitrary code.
"""

from __future__ import annotations

import builtins
import collections
import copyreg
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

from etl_hf.errors import HFWeightError

__all__ = [
    "read_safetensors",
    "read_torch_bin",
    "load_checkpoint",
    "widen_to_float32",
    "state_dict_parameter_count",
]

#: safetensors dtype tag → numpy dtype (``None`` = bfloat16, handled specially:
#: numpy has no bfloat16, so the raw u16 words are widened to float32).
_SAFETENSORS_DTYPES: dict[str, np.dtype | None] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": None,
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "U16": np.dtype(np.uint16),
    "U32": np.dtype(np.uint32),
    "U64": np.dtype(np.uint64),
    "BOOL": np.dtype(np.bool_),
}

#: torch storage class name → numpy dtype (``_BF16`` = raw u16 bfloat16 words).
_BF16 = "bfloat16"
_STORAGE_DTYPES: dict[str, object] = {
    "DoubleStorage": np.dtype(np.float64),
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "BFloat16Storage": _BF16,
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
}

#: torch value types accepted as inert placeholders (never used by etl_hf).
_INERT_TORCH_NAMES = frozenset({"Size", "device", "dtype", "Tensor"})

#: Builtin base classes a ``copyreg._reconstructor`` call may name.
_BUILTIN_BASES = frozenset({"tuple", "list", "dict", "object", "set", "frozenset"})

#: torch's legacy on-disk magic number and protocol version.
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_LEGACY_PROTOCOL = 1001

#: Directory candidate order for :func:`load_checkpoint`.
_DIR_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

#: File suffixes read by the torch reader (anything else is sniffed).
_TORCH_SUFFIXES = frozenset({".bin", ".pt", ".pth", ".ckpt", ".pkl"})


# ---------------------------------------------------------------------------
# dtype policy
# ---------------------------------------------------------------------------


def widen_to_float32(arr: np.ndarray) -> np.ndarray:
    """Apply the ``etl_hf`` checkpoint dtype policy to one tensor.

    ``float16`` is widened to ``float32`` (a copying conversion, exact — every
    float16 value is representable in float32); every other dtype is returned
    unchanged. bfloat16 never exists as a numpy dtype — both readers widen it
    at parse time (see :func:`_bf16_words_to_float32`) — so there is nothing to
    convert here.
    """
    array = np.asarray(arr)
    if array.dtype == np.float16:
        return np.ascontiguousarray(array, dtype=np.float32)
    return array


def _bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    """Widen an array of bfloat16 bit patterns (``uint16``) to ``float32``."""
    wide = words.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(wide.view(np.float32)).copy()


def state_dict_parameter_count(state_dict: dict[str, np.ndarray]) -> int:
    """Total number of scalar parameters in ``state_dict`` (sum of element counts)."""
    total = 0
    for key, value in state_dict.items():
        if not isinstance(value, np.ndarray):
            raise HFWeightError(
                f"state-dict entry {key!r} is {type(value).__name__}, expected a numpy array"
            )
        total += int(np.prod(value.shape, dtype=np.int64)) if value.shape else 1
    return total


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


def read_safetensors(path) -> dict[str, np.ndarray]:
    """Read a ``.safetensors`` file into ``{tensor_name: ndarray}``.

    Layout: 8-byte little-endian header length, a UTF-8 JSON header of
    ``{name: {"dtype", "shape", "data_offsets"}}`` (plus an optional
    ``"__metadata__"`` entry, which is skipped), then the raw tensor buffer.
    Offsets, element counts and dtype tags are validated against the file —
    any inconsistency raises :class:`~etl_hf.errors.HFWeightError` naming the
    tensor. Dtypes follow the module dtype policy (f16/bf16 → f32).
    """
    path = Path(path)
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise HFWeightError(f"{path}: cannot read safetensors file: {exc}") from exc
    if len(blob) < 8:
        raise HFWeightError(
            f"{path}: file is {len(blob)} bytes — too short to hold a safetensors header length"
        )
    (header_len,) = struct.unpack("<Q", blob[:8])
    if header_len == 0 or 8 + header_len > len(blob):
        raise HFWeightError(
            f"{path}: safetensors header length {header_len} does not fit in a "
            f"{len(blob)}-byte file"
        )
    try:
        header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HFWeightError(f"{path}: invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise HFWeightError(
            f"{path}: safetensors header must be a JSON object, got {type(header).__name__}"
        )
    buffer = memoryview(blob)[8 + header_len :]
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        out[name] = _read_safetensors_tensor(path, name, info, buffer)
    return {name: widen_to_float32(array) for name, array in out.items()}


def _read_safetensors_tensor(
    path: Path, name: str, info, buffer: memoryview
) -> np.ndarray:
    """Decode one safetensors header entry into an owning ``ndarray``."""
    if not isinstance(info, dict):
        raise HFWeightError(
            f"{path}: safetensors entry {name!r} must be a JSON object, "
            f"got {type(info).__name__}"
        )
    dtype_tag = info.get("dtype")
    if not isinstance(dtype_tag, str):
        raise HFWeightError(
            f"{path}: safetensors entry {name!r} has no string 'dtype' (got {dtype_tag!r})"
        )
    if dtype_tag not in _SAFETENSORS_DTYPES:
        raise HFWeightError(
            f"{path}: tensor {name!r} uses unsupported safetensors dtype {dtype_tag!r}; "
            f"supported: {sorted(_SAFETENSORS_DTYPES)}"
        )
    shape_raw = info.get("shape")
    if not isinstance(shape_raw, (list, tuple)):
        raise HFWeightError(
            f"{path}: tensor {name!r} has no list 'shape' (got {shape_raw!r})"
        )
    shape = []
    for dim in shape_raw:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise HFWeightError(
                f"{path}: tensor {name!r} has invalid dimension {dim!r} in shape {shape_raw!r}"
            )
        shape.append(dim)
    offsets = info.get("data_offsets")
    if (
        not isinstance(offsets, (list, tuple))
        or len(offsets) != 2
        or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
    ):
        raise HFWeightError(
            f"{path}: tensor {name!r} has invalid 'data_offsets' {offsets!r} "
            f"(expected [begin, end])"
        )
    begin, end = offsets
    if begin < 0 or end < begin:
        raise HFWeightError(
            f"{path}: tensor {name!r} has invalid data offsets [{begin}, {end}]"
        )
    if end > len(buffer):
        raise HFWeightError(
            f"{path}: tensor {name!r} claims data_offsets [{begin}, {end}] but the file's "
            f"data buffer is only {len(buffer)} bytes"
        )
    numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
    np_dtype = _SAFETENSORS_DTYPES[dtype_tag]
    itemsize = 2 if np_dtype is None else np_dtype.itemsize
    expected = numel * itemsize
    if end - begin != expected:
        raise HFWeightError(
            f"{path}: tensor {name!r} has shape {tuple(shape)} × {dtype_tag} "
            f"({expected} bytes) but data_offsets span {end - begin} bytes"
        )
    raw = buffer[begin:end]
    if np_dtype is None:
        words = np.frombuffer(raw, dtype=np.dtype("<u2")).reshape(shape)
        return _bf16_words_to_float32(words)
    values = np.frombuffer(raw, dtype=np_dtype.newbyteorder("<")).reshape(shape)
    return values.astype(np_dtype, copy=True)


# ---------------------------------------------------------------------------
# torch checkpoints — shared machinery
# ---------------------------------------------------------------------------


class _StorageType:
    """The storage flavour named by a torch persistent id (``torch.FloatStorage`` …)."""

    __slots__ = ("name", "dtype", "itemsize")

    def __init__(self, name: str):
        if name not in _STORAGE_DTYPES:
            raise HFWeightError(
                f"unsupported torch storage type torch.{name}; supported: "
                f"{sorted(_STORAGE_DTYPES)}"
            )
        self.name = name
        self.dtype = _STORAGE_DTYPES[name]
        self.itemsize = 2 if self.dtype == _BF16 else self.dtype.itemsize

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"torch.{self.name}"


class _Storage:
    """One storage's raw bytes, filled in after the pickle stream is consumed."""

    __slots__ = ("stype", "key", "numel", "raw")

    def __init__(self, stype: _StorageType, key: str, numel: int):
        if numel < 0:
            raise HFWeightError(f"storage {key!r} declares a negative element count {numel}")
        self.stype = stype
        self.key = key
        self.numel = numel
        self.raw: bytes | None = None

    def flat(self) -> np.ndarray:
        """The storage's elements as a flat (native dtype) array view."""
        if self.raw is None:
            raise HFWeightError(
                f"storage {self.key!r} ({self.stype.name}) was never materialized — "
                f"the checkpoint is corrupt or truncated"
            )
        want = self.numel * self.stype.itemsize
        if len(self.raw) < want:
            raise HFWeightError(
                f"storage {self.key!r} ({self.stype.name}) needs {want} bytes but only "
                f"{len(self.raw)} are available"
            )
        raw = self.raw[:want]
        if self.stype.dtype == _BF16:
            return _bf16_words_to_float32(
                np.frombuffer(raw, dtype=np.dtype("<u2"))
            )
        return np.frombuffer(raw, dtype=self.stype.dtype.newbyteorder("<"))


class _TensorRef:
    """A pending tensor view, resolved to an ``ndarray`` once storages have bytes."""

    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)
        if len(self.size) != len(self.stride):
            raise HFWeightError(
                f"tensor view has size {self.size} but stride {self.stride}"
            )

    def resolve(self) -> np.ndarray:
        """Materialize this view as a writable, C-contiguous copy."""
        flat = self.storage.flat()
        if 0 in self.size:
            return np.zeros(self.size, dtype=flat.dtype)
        # Element-index range covered by this view (negative strides included).
        low = self.offset + sum(min(0, (n - 1) * s) for n, s in zip(self.size, self.stride))
        high = self.offset + sum(max(0, (n - 1) * s) for n, s in zip(self.size, self.stride))
        if low < 0 or high >= flat.size:
            raise HFWeightError(
                f"tensor view (offset {self.offset}, size {self.size}, stride {self.stride}) "
                f"indexes elements [{low}, {high}] of a {flat.size}-element storage"
            )
        window = flat[low : high + 1]
        byte_strides = tuple(abs(s) * flat.itemsize for s in self.stride)
        view = np.lib.stride_tricks.as_strided(
            window, shape=self.size, strides=byte_strides, writeable=False
        )
        # as_strided can only walk forward: a dimension with a negative torch stride
        # is materialized forward from the view's low corner, then reversed.
        reversed_axes = tuple(i for i, s in enumerate(self.stride) if s < 0)
        if reversed_axes:
            view = np.flip(view, axis=reversed_axes)
        return np.array(view, dtype=flat.dtype, copy=True, order="C")


class _InertSize(tuple):
    """Inert stand-in for ``torch.Size`` (a tuple subclass, so it unpickles cleanly)."""


class _InertTorchValue:
    """Inert stand-in for ``torch.device`` / ``torch.dtype`` / ``torch.Tensor``.

    These values are accepted (real checkpoints reference them) but never used;
    holding them as inert objects guarantees no torch code can run.
    """

    __slots__ = ("args", "kwargs")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<inert torch value args={self.args!r}>"


def _rebuild_tensor(storage, storage_offset, size, stride, *args, **kwargs) -> _TensorRef:
    """``torch._utils._rebuild_tensor[_v2]`` replacement (extra args ignored)."""
    return _TensorRef(storage, storage_offset, size, stride)


def _rebuild_parameter(data, *args, **kwargs) -> object:
    """``torch._utils._rebuild_parameter`` replacement (drops the wrapper)."""
    return data


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild tensors — never execute arbitrary code.

    Allowed globals: ``collections.OrderedDict``, the three
    ``torch._utils`` rebuild helpers, ``torch.<X>Storage`` for the storage
    types in :data:`_STORAGE_DTYPES`, the inert torch value types, and the two
    ``copyreg`` reconstruction helpers (needed because ``torch.Size``-style
    values pickle through them; they can only instantiate the inert
    placeholders above). Anything else raises
    :class:`~etl_hf.errors.HFWeightError`.
    """

    def __init__(self, fh, *, source: str):
        super().__init__(fh)
        self._source = source
        self.storages: dict[str, _Storage] = {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and isinstance(name, str) and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch" and name in _INERT_TORCH_NAMES:
            return _InertSize if name == "Size" else _InertTorchValue
        # Protocol 0/1 streams (and some writers) rebuild tuple subclasses such as
        # ``torch.Size`` through ``copyreg``; both only ever instantiate the inert
        # placeholders resolved above, so they are safe to allow. The old Python 2
        # spellings appear verbatim in such streams.
        if module in ("copyreg", "copy_reg") and name in ("_reconstructor", "__newobj__"):
            return getattr(copyreg, name)
        if module in ("builtins", "__builtin__") and name in _BUILTIN_BASES:
            return getattr(builtins, name)
        raise HFWeightError(
            f"{self._source}: refusing to unpickle {module}.{name} — only tensor-rebuilding "
            f"globals are allowed (the file is not a plain torch state dict)"
        )

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise HFWeightError(
                f"{self._source}: unsupported torch persistent id {pid!r}"
            )
        _kind, storage_type, key, _location, numel = pid[:5]
        if not isinstance(storage_type, _StorageType):
            raise HFWeightError(
                f"{self._source}: unsupported storage type marker {storage_type!r} in "
                f"persistent id for key {key!r}"
            )
        if isinstance(numel, bool) or not isinstance(numel, int):
            raise HFWeightError(
                f"{self._source}: storage {key!r} declares a non-integer element count {numel!r}"
            )
        key = str(key)
        storage = self.storages.get(key)
        if storage is None:
            storage = self.storages[key] = _Storage(storage_type, key, numel)
        elif storage.numel != numel or storage.stype.name != storage_type.name:
            raise HFWeightError(
                f"{self._source}: storage key {key!r} is referenced twice with conflicting "
                f"metadata ({storage.stype.name}/{storage.numel} vs "
                f"{storage_type.name}/{numel})"
            )
        return storage


def _load_pickled(unpickler: _RestrictedUnpickler, path: Path):
    """Read one pickle object, converting pickle-level failures into HFWeightError."""
    try:
        return unpickler.load()
    except HFWeightError:
        raise
    except (
        pickle.UnpicklingError,
        EOFError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        struct.error,
    ) as exc:
        raise HFWeightError(
            f"{path}: cannot unpickle torch checkpoint stream: {type(exc).__name__}: {exc}"
        ) from exc


def _load_torch_object(unpickler: _RestrictedUnpickler, path: Path):
    """Read the real object of a torch pickle stream, skipping the 3-object header.

    ``torch.save`` prefixes the stream with the legacy magic number, the
    protocol version and a ``sys_info`` dict before the real object — for the
    legacy format always, and for the zip format's ``data.pkl`` as well when
    written by real torch. A bare pickled state dict (other writers) is used
    as-is.
    """
    first = _load_pickled(unpickler, path)
    if isinstance(first, int) and not isinstance(first, bool) and first == _LEGACY_MAGIC:
        protocol = _load_pickled(unpickler, path)
        if protocol != _LEGACY_PROTOCOL:
            raise HFWeightError(
                f"{path}: unsupported torch checkpoint protocol version {protocol!r} "
                f"(only {_LEGACY_PROTOCOL} is understood)"
            )
        _load_pickled(unpickler, path)  # sys_info dict — informational only
        return _load_pickled(unpickler, path)
    return first


def _unwrap_state_dict(obj, path: Path) -> dict:
    """Return the tensor dict from a ``{"state_dict": ...}`` wrapper or bare dict."""
    if not isinstance(obj, dict):
        raise HFWeightError(
            f"{path}: unpickled object is {type(obj).__name__}, expected a state dict (dict)"
        )
    inner = obj.get("state_dict")
    if isinstance(inner, dict):
        return inner
    return obj


# ---------------------------------------------------------------------------
# torch checkpoints — the two on-disk formats
# ---------------------------------------------------------------------------


def _read_torch_zip(path: Path) -> dict:
    """Read the torch ≥ 1.6 zip format (``*/data.pkl`` + ``*/data/<key>``)."""
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise HFWeightError(f"{path}: not a readable torch zip archive: {exc}") from exc
    with archive:
        pkl_members = [n for n in archive.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(pkl_members) != 1:
            raise HFWeightError(
                f"{path}: expected exactly one 'data.pkl' member in the torch zip archive, "
                f"found {pkl_members}"
            )
        root = pkl_members[0][: -len("data.pkl")]
        members = set(archive.namelist())
        try:
            with archive.open(pkl_members[0]) as handle:
                unpickler = _RestrictedUnpickler(handle, source=str(path))
                obj = _load_torch_object(unpickler, path)
        except HFWeightError:
            raise
        except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
            raise HFWeightError(f"{path}: cannot read {pkl_members[0]}: {exc}") from exc
        for key, storage in unpickler.storages.items():
            member = f"{root}data/{key}"
            if member not in members:
                raise HFWeightError(
                    f"{path}: missing storage blob {member!r} for storage key {key!r}"
                )
            storage.raw = archive.read(member)
    return _unwrap_state_dict(obj, path)


def _read_torch_legacy(path: Path) -> dict:
    """Read the legacy format: headers + state dict + key list + raw storages."""
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise HFWeightError(f"{path}: cannot read torch checkpoint: {exc}") from exc
    stream = io.BytesIO(blob)
    unpickler = _RestrictedUnpickler(stream, source=str(path))
    obj = _load_torch_object(unpickler, path)
    keys = _load_pickled(unpickler, path)
    if not isinstance(keys, list) or not all(isinstance(k, (str, int)) for k in keys):
        raise HFWeightError(
            f"{path}: legacy torch checkpoint trailer is not a list of storage keys "
            f"(got {type(keys).__name__})"
        )
    for raw_key in keys:
        key = str(raw_key)
        storage = unpickler.storages.get(key)
        if storage is None:
            raise HFWeightError(
                f"{path}: legacy trailer lists storage key {key!r} that the pickle stream "
                f"never referenced"
            )
        want = storage.numel * storage.stype.itemsize
        raw = stream.read(want)
        if len(raw) != want:
            raise HFWeightError(
                f"{path}: truncated legacy torch checkpoint — storage {key!r} needs {want} "
                f"bytes, only {len(raw)} remain"
            )
        storage.raw = raw
    return _unwrap_state_dict(obj, path)


def read_torch_bin(path) -> dict[str, np.ndarray]:
    """Read a ``pytorch_model.bin`` checkpoint (zip **or** legacy format).

    The format is decided by content (``zipfile.is_zipfile``), not by name.
    Tensors are rebuilt from their storages through a restricted unpickler
    (:class:`_RestrictedUnpickler`), so tensor views, offsets, strides and
    transposed views are all honoured while arbitrary pickled code is
    rejected. A ``{"state_dict": {...}}`` wrapper is unwrapped; a non-dict
    unpickled object or a non-tensor state-dict entry raises
    :class:`~etl_hf.errors.HFWeightError`. Dtypes follow the module dtype
    policy (f16/bf16 → f32). No key mapping and no transposition happens here —
    that is the per-family canonicaliser's job.
    """
    path = Path(path)
    if not path.is_file():
        raise HFWeightError(f"{path}: no such checkpoint file")
    state = _read_torch_zip(path) if zipfile.is_zipfile(path) else _read_torch_legacy(path)
    out: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise HFWeightError(f"{path}: state-dict key {key!r} is not a string")
        if isinstance(value, _TensorRef):
            value = value.resolve()
        if not isinstance(value, np.ndarray):
            raise HFWeightError(
                f"{path}: state-dict entry {key!r} is {type(value).__name__}, expected a tensor"
            )
        out[key] = widen_to_float32(value)
    return out


# ---------------------------------------------------------------------------
# checkpoint loading (single file, directory, sharded index)
# ---------------------------------------------------------------------------


def _read_checkpoint_file(path: Path) -> dict[str, np.ndarray]:
    """Read one checkpoint file, dispatching on suffix then on content."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return read_safetensors(path)
    if suffix in _TORCH_SUFFIXES:
        return read_torch_bin(path)
    try:
        return read_safetensors(path)
    except HFWeightError as safetensors_exc:
        try:
            return read_torch_bin(path)
        except HFWeightError as torch_exc:
            raise HFWeightError(
                f"{path}: not a readable checkpoint — safetensors: {safetensors_exc}; "
                f"torch: {torch_exc}"
            ) from torch_exc


def _read_sharded(model_dir: Path, index_path: Path) -> dict[str, np.ndarray]:
    """Read a sharded checkpoint through its index JSON ``weight_map``."""
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HFWeightError(f"{index_path}: cannot read shard index: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HFWeightError(f"{index_path}: invalid JSON in shard index: {exc}") from exc
    if not isinstance(raw, dict):
        raise HFWeightError(
            f"{index_path}: shard index must be a JSON object, got {type(raw).__name__}"
        )
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise HFWeightError(
            f"{index_path}: shard index has no non-empty 'weight_map' object"
        )
    keys_by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard, str):
            raise HFWeightError(
                f"{index_path}: 'weight_map' entry {key!r}: {shard!r} must map a string "
                f"tensor name to a string shard file name"
            )
        keys_by_shard.setdefault(shard, []).append(key)

    merged: dict[str, np.ndarray] = {}
    origin: dict[str, str] = {}
    for shard in sorted(keys_by_shard):
        shard_rel = Path(shard)
        if shard_rel.is_absolute() or ".." in shard_rel.parts:
            raise HFWeightError(
                f"{index_path}: shard name {shard!r} must be a relative path inside the "
                f"model directory"
            )
        shard_path = model_dir / shard_rel
        if not shard_path.is_file():
            raise HFWeightError(
                f"{index_path}: shard {shard!r} is missing from {model_dir} "
                f"(it holds {len(keys_by_shard[shard])} tensors, e.g. "
                f"{keys_by_shard[shard][0]!r})"
            )
        tensors = _read_checkpoint_file(shard_path)
        for key in keys_by_shard[shard]:
            if key not in tensors:
                raise HFWeightError(
                    f"{shard_path}: tensor {key!r} is listed in {index_path.name} but is "
                    f"absent from the shard"
                )
            if key in merged:
                raise HFWeightError(
                    f"{index_path}: tensor {key!r} appears in both {origin[key]!r} and "
                    f"{shard!r}"
                )
            merged[key] = tensors[key]
            origin[key] = shard
    return merged


def load_checkpoint(model_dir_or_file) -> dict[str, np.ndarray]:
    """Load a checkpoint from a model directory *or* a single checkpoint file.

    Directory priority: ``model.safetensors`` → ``pytorch_model.bin`` →
    ``model.safetensors.index.json`` → ``pytorch_model.bin.index.json``. When
    none of those exist the error lists every candidate it searched together
    with the directory. A file argument is read directly (suffix first, then a
    content sniff: safetensors, else torch). Sharded checkpoints are merged
    through their index ``weight_map``; a missing shard, a tensor listed in the
    index but absent from its shard, or a tensor present in two shards raises
    :class:`~etl_hf.errors.HFWeightError`.

    Returned names are the **on-disk** checkpoint keys — no key mapping, no
    transposition (see the module docstring).
    """
    path = Path(model_dir_or_file)
    if path.is_dir():
        for name in _DIR_CANDIDATES[:2]:
            candidate = path / name
            if candidate.is_file():
                return _read_checkpoint_file(candidate)
        for name in _DIR_CANDIDATES[2:]:
            candidate = path / name
            if candidate.is_file():
                return _read_sharded(path, candidate)
        raise HFWeightError(
            f"{path}: no checkpoint found — searched for {list(_DIR_CANDIDATES)}"
        )
    if path.is_file():
        return _read_checkpoint_file(path)
    raise HFWeightError(
        f"{path}: not an existing checkpoint file or model directory"
    )
