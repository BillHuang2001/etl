"""etl.backends — backend abstraction, reference numpy CPU backend, StableHLO exporter.

Public surface (full contract in this directory's CONTEXT.md and ../CONTEXT.md):

- ``Capabilities``, ``Backend``, ``Executable`` (protocol)      — backend.py
- ``Signature``, ``LoweredProgram``, ``CompiledArtifact``       — program.py (owned HERE, not by pipeline)
- ``register``, ``get``                                        — registry.py
- ``numpy_backend`` (default), ``NumpyBackend``, ``NumpyExecutable`` — numpy/
- ``stablehlo`` submodule (``stablehlo.export``; export utility, NOT a backend) — stablehlo/

Import acyclicity: this package imports ``etl.core`` / ``etl.ir`` at top level;
``etl.ops`` only inside function bodies; never ``etl.pipeline``.
"""
from .backend import Backend, Capabilities, Executable
from .program import CompiledArtifact, LoweredProgram, Signature
from .registry import get, register
from . import numpy, stablehlo
from .numpy import NumpyBackend, NumpyExecutable, numpy_backend

__all__ = [
    # backend contract
    "Capabilities",
    "Backend",
    "Executable",
    # staged objects owned by backends
    "Signature",
    "LoweredProgram",
    "CompiledArtifact",
    # registry
    "register",
    "get",
    # numpy reference backend (default)
    "numpy",
    "NumpyBackend",
    "NumpyExecutable",
    "numpy_backend",
    # stablehlo export utility
    "stablehlo",
]
