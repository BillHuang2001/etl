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
never at ``import etl`` / ``import etl.backends`` time (the device-transfer
thunk below imports the iree adapter lazily, on first call).
"""
from etl.core import DeviceError, register_device_transfer_provider

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


# --- device-transfer provider registration (explicit placement) --------------
# ``Tensor.to(Device('cuda', N))`` dispatches through the core registry
# (``core.register_device_transfer_provider``); core never imports backends.
# This package registers the "cuda" slot at import time as a LAZY thunk over
# the iree adapter: the adapter module is imported only on first transfer,
# and the adapter's module-level ``upload_tensor(tensor, device)`` is looked
# up by name at call time (implemented by the iree adapter; when the adapter
# activates it may overwrite this thunk with a direct provider under the
# same "cuda" key — registration is last-wins, so the overwrite is safe).

def _cuda_transfer_thunk(tensor, device):
    """Lazy "cuda" placement provider: delegates to the iree adapter.

    Imports ``etl.backends.adapters.iree`` on first call and delegates to
    its module-level ``upload_tensor(tensor, device)``. Any import or
    availability failure surfaces as a clean ``core.DeviceError`` with a
    pip hint — never a raw ``ModuleNotFoundError`` or other exception.
    """
    try:
        from etl.backends.adapters import iree as _iree_adapter
    except Exception as exc:  # noqa: BLE001 — always a clean DeviceError
        raise DeviceError(
            f"Tensor.to cannot place data on {device!r}: the cuda "
            "device-transfer provider requires the etl iree backend (the "
            "iree-base-compiler + iree-base-runtime packages), which is "
            f"unavailable: {exc}. Install it with `pip install etl[iree]`"
        ) from exc
    upload_tensor = getattr(_iree_adapter, "upload_tensor", None)
    if not callable(upload_tensor):
        raise DeviceError(
            f"Tensor.to cannot place data on {device!r}: the iree adapter "
            "does not expose upload_tensor(tensor, device), so the cuda "
            "device-transfer provider is unavailable (the iree adapter "
            "registers it when activated)"
        )
    return upload_tensor(tensor, device)


register_device_transfer_provider("cuda", _cuda_transfer_thunk)
