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
"complex"                  {"real": float, "imag": float}
"list"                     [encoded element, ...]
"tuple"                    [encoded element, ...]
"dict"                     {"items": [[encoded_key, encoded_value], ...]}
"numpy.ndarray"            {"dtype": dtype.str, "shape": [int, ...], "data_b64": base64 npy}
"numpy.generic"            same layout as "numpy.ndarray" (0-d; scalar dtype preserved)
"numpy.dtype"              {"dtype": dtype.str}        (decode: np.dtype(...))
"slice"                    {"start": enc|None, "stop": enc|None, "step": enc|None}
"Dim"                      {"name": str|int, "expr": enc(DimExpr)|None}
"DimExpr"                  {"kind": op-kind str, "operands": [enc, ...]}
"Device"                   {"kind": str, "index": int}
"TensorSpec"               {"shape": [enc, ...], "dtype": enc, "device": enc|None, "name": str|None}
"TreeSpec"                 {"type": treetype name, "num_leaves": int, "context": enc, "children": [enc, ...]}
=========================  ====================================================

numpy arrays: ``np.save`` into ``io.BytesIO`` with ``allow_pickle=False``,
then base64 (ASCII). Decode: ``np.load`` with ``allow_pickle=False``; the
returned array is made read-only (artifacts must not mutate saved bytes).

``ETL_FORMAT_VERSION`` travels with the container, so the codec evolves
with it; any type-name/layout change bumps the version (see CONTEXT.md).

Custom types (future): call ``register_codec(type_name, encoder, decoder)``
before any save/load that may encounter the type. This is the documented
route for trace/backends to persist custom types if ever needed; etl core
registers nothing extra in v1.

Import rule (binding): this module may import ``etl.core`` ONLY (Dim,
DimExpr, Device, TensorSpec, TreeSpec, PersistenceError) plus stdlib and
numpy — no other etl modules.
"""

from __future__ import annotations

import base64
import io

import numpy as np

from etl.core import Device, Dim, DimExpr, PersistenceError, TensorSpec, TreeSpec

ENVELOPE_TAG = "__etl_encoded__"
"""Tag field present in every encoded envelope dict."""

# type_name -> (encoder, decoder). Populated exactly once at import time
# by _install_builtin_codecs().
_CODECS = {}


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
    buffer handles as plain metadata.
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


def _install_builtin_codecs() -> None:
    """Install every codec listed in the module docstring into ``_CODECS``.

    Must run exactly once at import time (idempotent; a second run is a
    no-op). Each encoder/decoder pair follows the ``register_codec``
    protocol and the ``data`` layouts from the module docstring. Decoders
    validate the data layout and raise ``PersistenceError`` on any
    mismatch (including ``allow_pickle=False`` violations in corrupt or
    malicious numpy payloads). numpy array decoding returns read-only
    arrays. TreeSpec decoding requires ``core.TreeSpec`` to be
    reconstructible from (treetype name, context, children) — the codec
    calls that core constructor and raises ``PersistenceError`` if core
    does not provide it.
    """
    raise NotImplementedError(
        "persist.codec._install_builtin_codecs: architecture stub — "
        "implementation lands in Phase 2"
    )


def encode_value(value):
    """Encode ``value`` into a JSON-safe envelope dict (see module docstring).

    Algorithm:
      1. Dispatch on ``type(value)`` via the registry (built-ins installed
         at import time). Unknown types raise ``PersistenceError`` with a
         message listing the registered type names.
      2. Container codecs ("list", "tuple", "dict") recurse; "dict" encodes
         keys as well (keys may be non-string), as
         ``{"items": [[enc_key, enc_value], ...]}``.
      3. Recursion carries a memo of already-visited container ids; a
         cyclic structure raises ``PersistenceError`` ("cyclic value cannot
         be serialized") — never an infinite loop.
      4. The returned envelope is guaranteed JSON-serializable by
         ``json.dumps`` (numpy payloads are base64 strings; NaN/Inf floats
         are strings per the registry table).

    Returns the envelope dict. Never mutates the input value.
    """
    raise NotImplementedError(
        "persist.codec.encode_value: architecture stub — "
        "implementation lands in Phase 2"
    )


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
    raise NotImplementedError(
        "persist.codec.decode_value: architecture stub — "
        "implementation lands in Phase 2"
    )


# Install the built-in registry exactly once, at import time.
_install_builtin_codecs()
