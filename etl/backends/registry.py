"""Backend registry: register / get.

The default backend, ``numpy_backend`` (a ``NumpyBackend`` instance), is
registered by ``etl.backends.numpy`` at import time. ``etl.lower``/``compile``/
``load`` default to it (defaulting is pipeline's job, not the registry's).
"""
from __future__ import annotations

from etl.core import BackendError

from .backend import Backend

__all__ = ["register", "get"]

_registry: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Register a backend under ``backend.name``.

    Idempotent for the same instance (imports may re-execute); a DIFFERENT
    instance under an existing name raises ``BackendError``. A missing or
    non-string name raises ``BackendError``.

    Returns the backend (convenience for chaining).
    """
    if not isinstance(backend, Backend):
        raise BackendError(
            f"register() expects a Backend instance, got {type(backend).__name__}"
        )
    name = backend.name
    if not isinstance(name, str) or not name:
        raise BackendError(
            f"backend of type {type(backend).__name__} must declare a non-empty `name`"
        )
    existing = _registry.get(name)
    if existing is not None and existing is not backend:
        raise BackendError(
            f"backend name {name!r} is already registered by {type(existing).__name__}"
        )
    _registry[name] = backend
    return backend


def get(name: str) -> Backend:
    """Look up a backend by name. Unknown name raises ``BackendError``."""
    try:
        return _registry[name]
    except KeyError:
        known = ", ".join(sorted(_registry)) or "(none)"
        raise BackendError(f"unknown backend {name!r}; registered backends: {known}") from None
