"""JSON-safe value codec for the etl persistence layer.

`persist.codec` is the single encoding channel between etl's value-model
objects and the JSON representation persisted by `persist.container` and
used for `persist.cache` keys. Every encoded value is self-describing,
so artifacts round-trip without per-caller serialization logic.

Envelope format (binding, versioned with ``ETL_FORMAT_VERSION``)::

    {
        "__etl_encoded__": True,       # ENVELOPE_TAG
        "type": "<type_name>",         # registry key, see table below
        "data": <json-safe value>,     # type-specific payload
    }

``decode_value`` REQUIRES this envelope: input that is not an envelope
dict raises ``PersistenceError`` (never a silent identity fallback).

Built-in registry (``type_name`` -> ``data`` layout):

=========================  ====================================================
type_name                  data layout
=========================  ====================================================
"NoneType"                 null
"bool"                     true / false
"int"                      integer
"float"                    number (NaN/Inf serialized as strings "nan"/"inf"/"-inf")
"str"                      string
"complex"                  {"real": float-component, "imag": float-component}
                           (components use the "float" representation, so
                           complex NaN/Inf round-trips through strings)
"list"                     [encoded element, ...]
"tuple"                    [encoded element, ...]
"dict"                     {"items": [[encoded_key, encoded_value], ...]}
                           (keys encoded too — non-string keys allowed)
"numpy.ndarray"            {"dtype": dtype.str, "shape": [int, ...], "data_b64": base64 npy}
"numpy.generic"            same layout as "numpy.ndarray" (0-d; decode returns
                           the numpy scalar via ``loaded[()]``)
"numpy.dtype"              {"dtype": dtype.str}        (decode: np.dtype(...))
"slice"                    {"start": enc|None, "stop": enc|None, "step": enc|None}
"Dim"                      {"name": str, "size": enc}  (size None -> "NoneType" envelope;
                           core.Dim fields are ``name``/``size``)
"DimExpr"                  {"op": op str, "left": enc, "right": enc}
                           (core.DimExpr fields are ``op``/``left``/``right``)
"Device"                   {"kind": str, "index": int}
"TensorSpec"               {"shape": [enc, ...], "dtype": enc, "device": enc|None, "name": str|None}
"TreeSpec"                 {"type": treetype name, "context": enc, "node_data": enc,
                           "children": [enc, ...]}
                           treetype name: tuple -> "tuple", list -> "list",
                           dict -> "dict", None -> "NoneType"; any other type
                           -> "<module>.<qualname>" (e.g. "builtins.int");
                           decode resolves the name via ``importlib`` +
                           getattr chain (short names first)
=========================  ====================================================

numpy arrays: ``np.save`` into ``io.BytesIO`` with ``allow_pickle=False``,
then base64 (ASCII). Decode: ``np.load`` with ``allow_pickle=False``; the
returned array is made read-only (artifacts must not mutate saved bytes).
Declared dtype and shape are validated against the loaded array.

``ETL_FORMAT_VERSION`` travels with the container, so the codec evolves
with it; any type-name/layout change bumps the version (see CONTEXT.md).

Custom types: call ``register_codec(type_name, encoder, decoder)`` before
any save/load that may encounter the type. A codec registered under the
qualified name of a class (``"<module>.<qualname>"``) is dispatched
automatically by ``encode_value`` — this is the documented route for
trace/backends to persist custom types if ever needed; etl core registers
nothing extra in v1.

Import rule (binding): this module may import ``etl.core`` ONLY (Dim,
DimExpr, Device, TensorSpec, TreeSpec, PersistenceError) plus stdlib and
numpy — no other etl modules.
"""

from __future__ import annotations

import base64
import importlib
import io
import math

import numpy as np

from etl.core import Device, Dim, DimExpr, PersistenceError, TensorSpec, TreeSpec

ENVELOPE_TAG = "__etl_encoded__"
"""Tag field present in every encoded envelope dict."""

# type_name -> (encoder, decoder). Populated exactly once at import time
# by _install_builtin_codecs(); extended by user register_codec() calls.
_CODECS = {}

# exact type -> type_name, populated by _install_builtin_codecs() for the
# built-in codecs; used for fast exact-type dispatch in encode_value.
_TYPE_TO_NAME = {}

# Idempotency flag: _install_builtin_codecs() runs once at import time;
# any later call is a no-op.
_INSTALLED = False


def register_codec(type_name, encoder, decoder) -> None:
    """Register a codec pair in the global registry (trivial registry insert).

    Protocol (binding):
      encoder(value) -> JSON-safe dict  (the ``data`` part of the envelope)
      decoder(data_dict) -> value       (reconstructs the original object)

    Rules:
      * ``type_name`` must be a non-empty string and must NOT already be
        registered — shadowing raises ``PersistenceError`` (no silent
        override; an artifact must always decode with the codec that
        encoded it).
      * Registration is an explicit, programmatic act; persist never
        auto-discovers codecs.

    This is the documented extension point if trace/backends ever need to
    persist custom types: e.g. a backend vendor could
    ``register_codec("mytarget.buffer", enc, dec)`` to persist device
    buffer handles as plain metadata. Registering under the qualified name
    of a class (``"<module>.<qualname>"``) makes ``encode_value`` dispatch
    instances of that class to the codec automatically.
    """
    if not isinstance(type_name, str) or not type_name:
        raise PersistenceError(
            "persist.codec: invalid codec type_name {!r} (must be a non-empty string)".format(
                type_name
            )
        )
    if type_name in _CODECS:
        raise PersistenceError(
            "persist.codec: codec {!r} is already registered (shadowing a built-in "
            "or previously registered codec is not allowed)".format(type_name)
        )
    _CODECS[type_name] = (encoder, decoder)


# ---------------------------------------------------------------------------
# Shared encode/decode helpers
# ---------------------------------------------------------------------------

def _envelope(type_name, data):
    return {ENVELOPE_TAG: True, "type": type_name, "data": data}


def _enc_float_component(value):
    """Encode one float component: NaN/Inf become strings, else the number."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return value


def _dec_float_component(data):
    """Decode one float component: number (int/float, not bool) or one of
    the three special strings. Raises PersistenceError otherwise."""
    if isinstance(data, bool):
        raise PersistenceError(
            "persist.codec: invalid float payload {!r} (bool is not a float)".format(data)
        )
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, str) and data in ("nan", "inf", "-inf"):
        return float(data)
    raise PersistenceError(
        "persist.codec: invalid float payload {!r} (expected a number or "
        "'nan'/'inf'/'-inf')".format(data)
    )


def _require_dict(data, type_name):
    if not isinstance(data, dict):
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: expected a dict, got {!r}".format(
                type_name, data
            )
        )
    return data


def _get_field(data, key, type_name):
    try:
        return data[key]
    except KeyError:
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: missing {!r} field".format(type_name, key)
        ) from None


def _enter_container(path, obj):
    """Cycle detection: add obj's id to the active path; a repeat visit is
    a cyclic structure (raises PersistenceError)."""
    oid = id(obj)
    if oid in path:
        raise PersistenceError("persist.codec: cyclic value cannot be serialized")
    path.add(oid)


def _leave_container(path, obj):
    path.remove(id(obj))


def _decode_npy_payload(data, type_name):
    """Shared decoder body for "numpy.ndarray"/"numpy.generic": validate the
    layout, base64-decode + np.load with allow_pickle=False, and check the
    declared dtype/shape against the loaded array."""
    d = _require_dict(data, type_name)
    dtype_field = _get_field(d, "dtype", type_name)
    try:
        declared_dtype = np.dtype(dtype_field)
    except Exception as e:
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: invalid dtype {!r}: {}".format(
                type_name, dtype_field, e
            )
        ) from e
    shape_field = _get_field(d, "shape", type_name)
    if not isinstance(shape_field, list) or not all(
        isinstance(s, int) and not isinstance(s, bool) for s in shape_field
    ):
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: shape must be a list of ints, "
            "got {!r}".format(type_name, shape_field)
        )
    data_b64 = _get_field(d, "data_b64", type_name)
    if not isinstance(data_b64, str):
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: data_b64 must be a base64 string".format(
                type_name
            )
        )
    try:
        raw = base64.b64decode(data_b64.encode("ascii"))
        loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: {}".format(type_name, e)
        ) from e
    if loaded.dtype != declared_dtype:
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: dtype mismatch (declared {}, "
            "loaded {})".format(type_name, declared_dtype, loaded.dtype)
        )
    if list(loaded.shape) != shape_field:
        raise PersistenceError(
            "persist.codec: corrupt {!r} payload: shape mismatch (declared {}, "
            "loaded {})".format(type_name, shape_field, list(loaded.shape))
        )
    return loaded


# ---------------------------------------------------------------------------
# Built-in codec pairs
# ---------------------------------------------------------------------------

def _enc_none(value, path):
    return None


def _dec_none(data):
    if data is not None:
        raise PersistenceError(
            "persist.codec: corrupt NoneType payload: expected null, got {!r}".format(data)
        )
    return None


def _enc_bool(value, path):
    return value


def _dec_bool(data):
    if not isinstance(data, bool):
        raise PersistenceError(
            "persist.codec: corrupt bool payload: expected true/false, got {!r}".format(data)
        )
    return data


def _enc_int(value, path):
    return value


def _dec_int(data):
    if not isinstance(data, int) or isinstance(data, bool):
        raise PersistenceError(
            "persist.codec: corrupt int payload: expected an integer, got {!r}".format(data)
        )
    return data


def _enc_float(value, path):
    return _enc_float_component(value)


def _dec_float(data):
    return _dec_float_component(data)


def _enc_str(value, path):
    return value


def _dec_str(data):
    if not isinstance(data, str):
        raise PersistenceError(
            "persist.codec: corrupt str payload: expected a string, got {!r}".format(data)
        )
    return data


def _enc_complex(value, path):
    return {"real": _enc_float_component(value.real), "imag": _enc_float_component(value.imag)}


def _dec_complex(data):
    d = _require_dict(data, "complex")
    real = _dec_float_component(_get_field(d, "real", "complex"))
    imag = _dec_float_component(_get_field(d, "imag", "complex"))
    return complex(real, imag)


def _enc_list(value, path):
    _enter_container(path, value)
    try:
        return [_encode(item, path) for item in value]
    finally:
        _leave_container(path, value)


def _dec_list(data):
    if not isinstance(data, list):
        raise PersistenceError(
            "persist.codec: corrupt list payload: expected a list, got {!r}".format(data)
        )
    return [decode_value(item) for item in data]


def _enc_tuple(value, path):
    _enter_container(path, value)
    try:
        return [_encode(item, path) for item in value]
    finally:
        _leave_container(path, value)


def _dec_tuple(data):
    if not isinstance(data, list):
        raise PersistenceError(
            "persist.codec: corrupt tuple payload: expected a list, got {!r}".format(data)
        )
    return tuple(decode_value(item) for item in data)


def _enc_dict(value, path):
    _enter_container(path, value)
    try:
        items = [[_encode(k, path), _encode(v, path)] for k, v in value.items()]
        return {"items": items}
    finally:
        _leave_container(path, value)


def _dec_dict(data):
    d = _require_dict(data, "dict")
    items = _get_field(d, "items", "dict")
    if not isinstance(items, list):
        raise PersistenceError(
            "persist.codec: corrupt dict payload: 'items' must be a list, got {!r}".format(
                items
            )
        )
    result = {}
    for pair in items:
        if not isinstance(pair, list) or len(pair) != 2:
            raise PersistenceError(
                "persist.codec: corrupt dict payload: each item must be a "
                "[key, value] pair, got {!r}".format(pair)
            )
        key = decode_value(pair[0])
        value = decode_value(pair[1])
        result[key] = value
    return result


def _enc_ndarray(value, path):
    buf = io.BytesIO()
    try:
        np.save(buf, value, allow_pickle=False)
    except Exception as e:
        raise PersistenceError(
            "persist.codec: cannot encode numpy array of dtype {}: {}".format(value.dtype, e)
        ) from e
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _dec_ndarray(data):
    loaded = _decode_npy_payload(data, "numpy.ndarray")
    array = np.array(loaded)  # fresh buffer, never aliases the decoded bytes
    array.setflags(write=False)
    return array


def _enc_generic(value, path):
    array = np.asarray(value)
    buf = io.BytesIO()
    try:
        np.save(buf, array, allow_pickle=False)
    except Exception as e:
        raise PersistenceError(
            "persist.codec: cannot encode numpy scalar of dtype {}: {}".format(
                value.dtype, e
            )
        ) from e
    return {
        "dtype": value.dtype.str,
        "shape": [],
        "data_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _dec_generic(data):
    if not isinstance(data, dict) or data.get("shape") != []:
        raise PersistenceError(
            "persist.codec: corrupt numpy.generic payload: expected a 0-d array "
            "layout with shape []"
        )
    loaded = _decode_npy_payload(data, "numpy.generic")
    return loaded[()]  # the numpy scalar (dtype preserved; scalars are immutable)


def _enc_dtype(value, path):
    return {"dtype": value.str}


def _dec_dtype(data):
    d = _require_dict(data, "numpy.dtype")
    dtype_field = _get_field(d, "dtype", "numpy.dtype")
    try:
        return np.dtype(dtype_field)
    except Exception as e:
        raise PersistenceError(
            "persist.codec: corrupt numpy.dtype payload: invalid dtype {!r}: {}".format(
                dtype_field, e
            )
        ) from e


def _enc_slice(value, path):
    return {
        "start": None if value.start is None else _encode(value.start, path),
        "stop": None if value.stop is None else _encode(value.stop, path),
        "step": None if value.step is None else _encode(value.step, path),
    }


def _dec_slice(data):
    d = _require_dict(data, "slice")
    parts = []
    for key in ("start", "stop", "step"):
        field = _get_field(d, key, "slice")
        if field is None:
            parts.append(None)
        else:
            part = decode_value(field)
            if not isinstance(part, int) or isinstance(part, bool):
                raise PersistenceError(
                    "persist.codec: corrupt slice payload: {!r} must be None or an "
                    "int, got {!r}".format(key, part)
                )
            parts.append(part)
    return slice(*parts)


def _enc_dim(value, path):
    return {"name": value.name, "size": _encode(value.size, path)}


def _dec_dim(data):
    d = _require_dict(data, "Dim")
    name = _get_field(d, "name", "Dim")
    if not isinstance(name, str):
        raise PersistenceError(
            "persist.codec: corrupt Dim payload: name must be a string, got {!r}".format(name)
        )
    size = decode_value(_get_field(d, "size", "Dim"))
    return Dim(name, size)


def _enc_dimexpr(value, path):
    return {"op": value.op, "left": _encode(value.left, path), "right": _encode(value.right, path)}


def _dec_dimexpr(data):
    d = _require_dict(data, "DimExpr")
    op = _get_field(d, "op", "DimExpr")
    left = decode_value(_get_field(d, "left", "DimExpr"))
    right = decode_value(_get_field(d, "right", "DimExpr"))
    try:
        return DimExpr(op, left, right)
    except (TypeError, ValueError) as e:
        raise PersistenceError(
            "persist.codec: corrupt DimExpr payload: {}".format(e)
        ) from e


def _enc_device(value, path):
    return {"kind": value.kind, "index": value.index}


def _dec_device(data):
    d = _require_dict(data, "Device")
    kind = _get_field(d, "kind", "Device")
    index = _get_field(d, "index", "Device")
    try:
        return Device(kind, index)
    except (TypeError, ValueError) as e:
        raise PersistenceError(
            "persist.codec: corrupt Device payload: {}".format(e)
        ) from e


def _enc_tensorspec(value, path):
    return {
        "shape": [_encode(entry, path) for entry in value.shape],
        "dtype": _encode(value.dtype, path),
        "device": None if value.device is None else _encode(value.device, path),
        "name": value.name,
    }


def _dec_tensorspec(data):
    d = _require_dict(data, "TensorSpec")
    shape_field = _get_field(d, "shape", "TensorSpec")
    if not isinstance(shape_field, list):
        raise PersistenceError(
            "persist.codec: corrupt TensorSpec payload: 'shape' must be a list, "
            "got {!r}".format(shape_field)
        )
    shape = tuple(decode_value(entry) for entry in shape_field)
    dtype = decode_value(_get_field(d, "dtype", "TensorSpec"))
    device_field = _get_field(d, "device", "TensorSpec")
    device = None if device_field is None else decode_value(device_field)
    name = _get_field(d, "name", "TensorSpec")
    try:
        return TensorSpec(shape=shape, dtype=dtype, device=device, name=name)
    except TypeError as e:
        raise PersistenceError(
            "persist.codec: corrupt TensorSpec payload: {}".format(e)
        ) from e


def _treetype_name(typ):
    """Map a TreeSpec 'type' field to a registry string (see module table)."""
    if typ is None:
        return "NoneType"
    if not isinstance(typ, type):
        raise PersistenceError(
            "persist.codec: cannot encode TreeSpec with a non-type 'type' field "
            "{!r} (expected a type or None)".format(typ)
        )
    if typ is tuple:
        return "tuple"
    if typ is list:
        return "list"
    if typ is dict:
        return "dict"
    return "{}.{}".format(typ.__module__, typ.__qualname__)


def _resolve_treetype(name):
    """Resolve a treetype name back to a type (short names first, then
    importlib + getattr chain over the qualified name)."""
    short_names = {
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "NoneType": None,
        # ``_treetype_name`` writes ``builtins.NoneType`` for TreeSpec leaves
        # whose type field is ``type(None)`` (e.g. static None leaves —
        # core.flatten records the leaf type as NoneType, which is not None);
        # builtins has no NoneType attribute, so resolve it here.
        "builtins.NoneType": type(None),
    }
    if name in short_names:
        return short_names[name]
    if not isinstance(name, str) or "." not in name:
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: cannot resolve treetype "
            "name {!r} (expected a short name or '<module>.<qualname>')".format(name)
        )
    parts = name.split(".")
    module = None
    split_at = None
    # The module part is the longest importable prefix (module names may
    # themselves contain dots); the remainder is the getattr chain.
    for i in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            split_at = i
            break
        except ImportError:
            continue
    if module is None:
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: cannot resolve treetype "
            "name {!r} (no importable module)".format(name)
        )
    obj = module
    try:
        for part in parts[split_at:]:
            obj = getattr(obj, part)
    except AttributeError:
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: cannot resolve treetype "
            "name {!r}".format(name)
        ) from None
    return obj


def _enc_treespec(value, path):
    return {
        "type": _treetype_name(value.type),
        "context": _encode(value.context, path),
        "node_data": _encode(value.node_data, path),
        "children": [_encode(child, path) for child in value.children],
    }


def _dec_treespec(data):
    d = _require_dict(data, "TreeSpec")
    type_name = _get_field(d, "type", "TreeSpec")
    if not isinstance(type_name, str):
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: 'type' must be a string, "
            "got {!r}".format(type_name)
        )
    treetype = _resolve_treetype(type_name)
    children_field = _get_field(d, "children", "TreeSpec")
    if not isinstance(children_field, list):
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: 'children' must be a list, "
            "got {!r}".format(children_field)
        )
    children = tuple(decode_value(child) for child in children_field)
    if not all(isinstance(child, TreeSpec) for child in children):
        raise PersistenceError(
            "persist.codec: corrupt TreeSpec payload: children must decode to "
            "TreeSpec instances"
        )
    context = decode_value(_get_field(d, "context", "TreeSpec"))
    node_data = decode_value(_get_field(d, "node_data", "TreeSpec"))
    return TreeSpec(type=treetype, children=children, context=context, node_data=node_data)


# (type_name, encoder, decoder) for every built-in codec, in registry order.
_BUILTIN_CODECS = [
    ("NoneType", _enc_none, _dec_none),
    ("bool", _enc_bool, _dec_bool),
    ("int", _enc_int, _dec_int),
    ("float", _enc_float, _dec_float),
    ("str", _enc_str, _dec_str),
    ("complex", _enc_complex, _dec_complex),
    ("list", _enc_list, _dec_list),
    ("tuple", _enc_tuple, _dec_tuple),
    ("dict", _enc_dict, _dec_dict),
    ("numpy.ndarray", _enc_ndarray, _dec_ndarray),
    ("numpy.generic", _enc_generic, _dec_generic),
    ("numpy.dtype", _enc_dtype, _dec_dtype),
    ("slice", _enc_slice, _dec_slice),
    ("Dim", _enc_dim, _dec_dim),
    ("DimExpr", _enc_dimexpr, _dec_dimexpr),
    ("Device", _enc_device, _dec_device),
    ("TensorSpec", _enc_tensorspec, _dec_tensorspec),
    ("TreeSpec", _enc_treespec, _dec_treespec),
]

# exact-type dispatch table built alongside the registry.
_BUILTIN_TYPES = {
    type(None): "NoneType",
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
    complex: "complex",
    list: "list",
    tuple: "tuple",
    dict: "dict",
    np.ndarray: "numpy.ndarray",
    np.dtype: "numpy.dtype",
    slice: "slice",
    Dim: "Dim",
    DimExpr: "DimExpr",
    Device: "Device",
    TensorSpec: "TensorSpec",
    TreeSpec: "TreeSpec",
}


def _install_builtin_codecs() -> None:
    """Install every codec listed in the module docstring into ``_CODECS``.

    Runs exactly once at import time (idempotent; a second run is a no-op).
    Each encoder/decoder pair follows the ``register_codec`` protocol and
    the ``data`` layouts from the module docstring. Decoders validate the
    data layout and raise ``PersistenceError`` on any mismatch (including
    ``allow_pickle=False`` violations in corrupt or malicious numpy
    payloads). numpy array decoding returns read-only arrays.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    for type_name, encoder, decoder in _BUILTIN_CODECS:
        register_codec(type_name, encoder, decoder)
    _TYPE_TO_NAME.update(_BUILTIN_TYPES)
    _INSTALLED = True


def _lookup_type_name(value):
    """Find the registry name for a value's type: exact-type dispatch on the
    built-in table, isinstance fallbacks for numpy scalar/array subclasses,
    then the qualified-name fallback for custom codecs. None if unknown."""
    exact = _TYPE_TO_NAME.get(type(value))
    if exact is not None:
        return exact
    if isinstance(value, np.generic):
        return "numpy.generic"
    if isinstance(value, np.ndarray):
        return "numpy.ndarray"
    if isinstance(value, np.dtype):
        return "numpy.dtype"
    qualname = "{}.{}".format(type(value).__module__, type(value).__qualname__)
    if qualname in _CODECS:
        return qualname
    return None


def _encode(value, path):
    """Encode one value into its envelope dict, threading the active
    container-id path for cycle detection."""
    type_name = _lookup_type_name(value)
    if type_name is None:
        raise PersistenceError(
            "persist.codec: cannot encode value of type {!r}; registered codec "
            "names: {}".format(type(value), sorted(_CODECS))
        )
    encoder, _ = _CODECS[type_name]
    return _envelope(type_name, encoder(value, path))


def encode_value(value):
    """Encode ``value`` into a JSON-safe envelope dict (see module docstring).

    Algorithm:
      1. Dispatch on ``type(value)`` via the registry (built-ins installed
         at import time; numpy scalar/array subclasses map to their base
         codec; custom codecs registered under a class qualified name are
         dispatched by ``"<module>.<qualname>"``). Unknown types raise
         ``PersistenceError`` with a message listing the registered type
         names.
      2. Container codecs ("list", "tuple", "dict") recurse; "dict" encodes
         keys as well (keys may be non-string), as
         ``{"items": [[enc_key, enc_value], ...]}``.
      3. Recursion carries a path set of active container ids (added before
         recursing into a container, removed after); a cyclic structure
         raises ``PersistenceError`` ("cyclic value cannot be serialized")
         — never an infinite loop. Shared (non-cyclic) references are
         encoded once per occurrence.
      4. The returned envelope is guaranteed JSON-serializable by
         ``json.dumps`` (numpy payloads are base64 strings; NaN/Inf floats
         are strings per the registry table).

    Returns the envelope dict. Never mutates the input value.
    """
    return _encode(value, set())


def decode_value(encoded):
    """Decode an envelope dict (as produced by ``encode_value``) back to a value.

    Algorithm:
      1. Require a dict with ``ENVELOPE_TAG`` truthy and a ``"type"`` key;
         otherwise raise ``PersistenceError`` ("corrupt: not an encoded
         value") — never a silent pass-through.
      2. Look up ``type`` in the registry; unknown names raise
         ``PersistenceError`` listing the registered type names.
      3. Call the registered decoder on ``data``; decoders recursively call
         ``decode_value`` for nested envelopes.
      4. Return the reconstructed value; numpy arrays come back read-only
         (no aliasing with anything on disk).

    The result is a fresh value tree each call.
    """
    if not isinstance(encoded, dict) or not encoded.get(ENVELOPE_TAG) or "type" not in encoded:
        raise PersistenceError(
            "persist.codec: corrupt: not an encoded value (missing {} tag or "
            "'type' field)".format(ENVELOPE_TAG)
        )
    type_name = encoded["type"]
    entry = _CODECS.get(type_name)
    if entry is None:
        raise PersistenceError(
            "persist.codec: unknown encoded type {!r}; registered codec names: "
            "{}".format(type_name, sorted(_CODECS))
        )
    if "data" not in encoded:
        raise PersistenceError(
            "persist.codec: corrupt: encoded value of type {!r} has no 'data' "
            "field".format(type_name)
        )
    _, decoder = entry
    return decoder(encoded["data"])


# Install the built-in registry exactly once, at import time (idempotent).
_install_builtin_codecs()
