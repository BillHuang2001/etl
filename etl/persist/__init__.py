"""etl.persist — versioned/self-describing/integrity-checked save/load container + explicit cache.

This package is the persistence layer of etl: it stores JSON metadata +
encoded payloads + a SHA-256 integrity hash, and provides an explicit,
user-owned cache. It deliberately contains NO trace/backend logic: it can
never silently re-trace or re-compile anything. See this directory's
CONTEXT.md for the container byte layout, the codec registry, and the
cache keying contract.
"""

from .cache import Cache, FileCache, compute_key
from .codec import decode_value, encode_value, register_codec
from .container import ETL_FORMAT_VERSION, LoadedObject, load_object, save_object

__all__ = [
    "ETL_FORMAT_VERSION",
    "Cache",
    "FileCache",
    "LoadedObject",
    "compute_key",
    "decode_value",
    "encode_value",
    "load_object",
    "register_codec",
    "save_object",
]
