"""etl.backends — backend abstraction, reference numpy CPU backend, StableHLO exporter.

Public surface (full contract in this directory's CONTEXT.md and ../CONTEXT.md):

- ``Capabilities``, ``Backend``, ``Executable`` (protocol)      — backend.py
- ``Signature``, ``LoweredProgram``, ``CompiledArtifact``       — program.py (owned HERE, not by pipeline)
- ``register``, ``get`` (auto-activates optional adapters)      — registry.py
- ``CompilerBackend``, ``CompilerExecutable``                   — compiler.py (shared pluggable-compiler framework)
- ``numpy_backend`` (default), ``NumpyBackend``, ``NumpyExecutable`` — numpy/
- ``stablehlo`` submodule (``stablehlo.export``; export utility, NOT a backend) — stablehlo/
- shared block-inlining machinery                              — inline.py
- ``adapters`` subpackage (iree/xla/tvm adapter modules — a separate effort) — adapters/

Import acyclicity: this package imports ``etl.core`` / ``etl.ir`` at top level;
``etl.ops`` only inside function bodies; never ``etl.pipeline``. Optional
compiler adapters are imported only on first registry use (``get("iree")``) —
never at ``import etl`` / ``import etl.backends`` time.
"""
from .backend import Backend, Capabilities, Executable
from .compiler import CompilerBackend, CompilerExecutable
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
    # shared pluggable-compiler framework (adapters/ subpackage)
    "CompilerBackend",
    "CompilerExecutable",
    # numpy reference backend (default)
    "numpy",
    "NumpyBackend",
    "NumpyExecutable",
    "numpy_backend",
    # stablehlo export utility
    "stablehlo",
]
