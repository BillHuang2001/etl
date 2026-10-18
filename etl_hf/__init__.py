"""``etl_hf`` — load a HuggingFace checkpoint and run its forward as ONE etl graph.

``etl_hf`` sits **above** :mod:`etl` (a separate top-level package, never a
submodule): etl core's non-goals exclude NN layers, loaders and model zoos, so
model-level code lives here and only *depends on* the public etl API.

The philosophy is etl's, applied to whole models — **make computation
explicit**:

* a checkpoint's weights are ordinary *graph inputs* (``TensorSpec`` trees),
  never ``etl.constant`` payloads, so nothing is silently inlined into the
  program;
* config values are static specialisation values: one config + one
  ``seq_len``/``batch_size`` = one graph, memoised at the build layer;
* the pipeline is the explicit etl one — ``trace → lower → compile → load →
  run`` — reached through :func:`build_model` (``etl.build_cached``) and
  :func:`run_model` (``etl.run``), with no hidden tracing, compiling, weight
  embedding or device movement;
* moving weights to a device is an explicit, per-leaf ``Tensor.to(device)``
  (:func:`place_params`) — a non-cpu executable rejects host inputs by design;
* every failure is an explicit, actionable error deriving from
  :class:`etl.core.ETLError` — never a silent fallback, dtype swap or positional
  guess.

Typical use::

    import etl, etl_hf

    model = etl_hf.load_hf("path/to/checkpoint", seq_len=32)
    logits = model.forward()                     # numpy backend, host logits
    logits = model.forward(backend="iree")       # compiled path (explicit)
    exe = model.build(backend="iree")            # the Executable itself
    device_params = etl_hf.place_params(model.params(), etl.core.Device("cuda", 0))

Import-time constraint: ``import etl_hf`` requires only ``numpy`` (plus ``etl``).
torch / transformers / safetensors / iree are **never** imported at module scope.
"""

from __future__ import annotations

from etl_hf import build, config, families, ops, pack, reference, weights
from etl_hf.build import HFModel, build_model, load_hf, place_params, run_model
from etl_hf.config import ModelConfig
from etl_hf.errors import (
    HFConfigError,
    HFError,
    HFUnsupportedError,
    HFWeightError,
)
from etl_hf.families import FAMILIES, FamilySpec, family_for_model_type, get_family
from etl_hf.pack import PackPlan, pack_params, pack_plan

__version__ = "0.1.0"

__all__ = [
    # -- model loading / execution -----------------------------------------
    "HFModel",
    "ModelConfig",
    "build_model",
    "load_hf",
    "place_params",
    "run_model",
    # -- packed weights (opt-in) --------------------------------------------
    "PackPlan",
    "pack_params",
    "pack_plan",
    # -- family registry ----------------------------------------------------
    "FAMILIES",
    "FamilySpec",
    "family_for_model_type",
    "get_family",
    # -- errors -------------------------------------------------------------
    "HFConfigError",
    "HFError",
    "HFUnsupportedError",
    "HFWeightError",
    # -- submodules ---------------------------------------------------------
    "build",
    "config",
    "families",
    "ops",
    "pack",
    "reference",
    "weights",
    # -- metadata -----------------------------------------------------------
    "__version__",
]
