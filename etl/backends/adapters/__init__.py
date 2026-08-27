"""etl.backends.adapters — optional compiler backends (pluggable adapters).

This package is the home of the OPTIONAL compiler adapters. Each adapter is
a separate module implemented by the adapter authors (NOT this package's
framework — see ``etl.backends.compiler`` for the shared
``CompilerBackend`` / ``CompilerExecutable`` base classes they subclass):

- ``iree.py`` — IREE adapter (module-level singleton ``iree_backend``).
- ``xla.py`` — XLA-via-PJRT adapter (module-level singleton ``xla_backend``).
- ``tvm.py`` — TVM adapter (module-level singleton ``tvm_backend``).

Adapter-module contract (binding for adapter authors):

- Subclass ``etl.backends.compiler.CompilerBackend`` (name, capabilities,
  ``check_available``, ``compile``, ``load``) and
  ``etl.backends.compiler.CompilerExecutable`` (``backend_name``, ``run``).
- Each module exposes a module-level ``register()`` function that (a)
  checks the compiler dependency's availability — raising
  ``core.BackendError`` with a pip-install hint (e.g. ``pip install
  etl[iree]``) when missing — and (b) calls
  ``etl.backends.registry.register(instance)`` (idempotent).
- Heavy-import rule (binding): imports of the compiler dependency
  (iree/jax/tvm/…) live ONLY inside function bodies, NEVER at module top
  level — ``import etl`` and ``import etl.backends`` must never import an
  adapter or its compiler dependency.
- Activation is register-on-first-use: ``etl.backends.registry.get(name)``
  (via ``registry.OPTIONAL_ADAPTERS``) imports the adapter module and calls
  its ``register()`` on the first registry miss — the same path that
  auto-activates adapters when loading persisted artifacts.

This ``__init__.py`` deliberately exposes NO names and imports nothing
heavy — it is documentation only (plus the package marker).
"""
from __future__ import annotations

__all__: list[str] = []
