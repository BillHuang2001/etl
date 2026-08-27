"""Backend registry: register / get (+ optional-adapter auto-activation).

The default backend, ``numpy_backend`` (a ``NumpyBackend`` instance), is
registered by ``etl.backends.numpy`` at import time. ``etl.lower``/``compile``/
``load`` default to it (defaulting is pipeline's job, not the registry's).

Optional compiler adapters (IREE / XLA / TVM — see ``adapters/``) are NOT
imported here or at ``etl``/``etl.backends`` import time. ``get(name)``
auto-activates them ON FIRST USE: on a registry miss it imports the adapter
module named by ``OPTIONAL_ADAPTERS`` and calls its ``register()`` (which
probes the compiler dependency — a missing dependency raises
``core.BackendError`` with a pip-install hint, propagated unchanged), then
retries the lookup.
"""
from __future__ import annotations

from etl.core import BackendError

from .backend import Backend

__all__ = ["register", "get", "OPTIONAL_ADAPTERS"]

_registry: dict[str, Backend] = {}

#: Optional compiler adapters: backend name -> adapter module path.
#: Modules implement a ``register()`` function that checks the compiler
#: dependency and calls ``registry.register(instance)`` (idempotent). They
#: are imported ONLY inside ``get()`` — never at package import time.
OPTIONAL_ADAPTERS: dict[str, str] = {
    "iree": "etl.backends.adapters.iree",
    "xla": "etl.backends.adapters.xla",
    "tvm": "etl.backends.adapters.tvm",
}


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
    """Look up a backend by name; auto-activate optional adapters on a miss.

    Registered backends resolve directly. On a miss, an optional compiler
    adapter named in ``OPTIONAL_ADAPTERS`` is imported and its
    ``register()`` is called (which probes the compiler dependency and
    registers the instance — a missing dependency raises
    ``core.BackendError`` with a pip-install hint, propagated UNCHANGED);
    the lookup is then retried. Unknown names (and adapters that
    ``register()`` without registering anything) raise ``BackendError``.
    """
    backend = _registry.get(name)
    if backend is not None:
        return backend
    if name in OPTIONAL_ADAPTERS:
        import importlib

        module = importlib.import_module(OPTIONAL_ADAPTERS[name])
        module.register()
        backend = _registry.get(name)
        if backend is not None:
            return backend
    known = ", ".join(sorted(_registry)) or "(none)"
    raise BackendError(f"unknown backend {name!r}; registered backends: {known}")
