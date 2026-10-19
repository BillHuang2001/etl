"""``etl_hf`` — load a HuggingFace checkpoint and run its forward as ONE etl graph.

``etl_hf`` sits **above** :mod:`etl` (a separate top-level package, never a
submodule): etl core's non-goals exclude NN layers, loaders and model zoos, so
model-level code lives here and only *depends on* the public etl API.

The API is **functional**, mirroring etl's own value model, with a hard split
between the two halves of the problem:

* **loading** (:mod:`etl_hf.load`) answers *what is this checkpoint?* — a static
  :class:`ModelConfig` (one config = one graph, see ``config.digest``) plus a
  canonical ``dict[str, np.ndarray]`` of host params (:class:`LoadedModel` is
  plain immutable data — no methods, no ``seq_len``);
* **the compute function** (:mod:`etl_hf.program`) answers *what does it
  compute?* — :func:`forward_fn` is a pure ``@etl.defn`` forward
  ``(params, input_ids) -> logits`` capturing **no** parameters, only the static
  specialisation ``(family, config, seq_len, batch_size, pack_weights)``;
* **staging** (:mod:`etl_hf.pipeline`) answers *where and how does it run?* —
  :func:`build` (``etl.build_cached``) and :func:`run` (``etl.run``), with
  :func:`predict` as the one documented host-facing shorthand.

The philosophy is etl's, applied to whole models — **make computation
explicit**:

* a checkpoint's weights are ordinary *graph inputs* (:func:`param_specs`
  ``TensorSpec`` trees), never ``etl.constant`` payloads, so nothing is silently
  inlined into the program;
* config values are static specialisation values: one config + one
  ``seq_len``/``batch_size``/packing = one traced graph, memoised so two
  identical :func:`build` calls return the identical ``Executable``;
* the pipeline is the explicit etl one — ``trace → lower → compile → load →
  run`` — reached through :func:`build` and :func:`run`, with no hidden tracing,
  compiling, weight embedding or device movement;
* moving weights *and* token ids to a device is an explicit, per-leaf
  ``Tensor.to(device)`` (:func:`place_params` / :func:`place_ids`) — a non-cpu
  executable rejects host inputs by design;
* every failure is an explicit, actionable error deriving from
  :class:`etl.core.ETLError` — never a silent fallback, dtype swap or positional
  guess.

Typical use::

    import etl, etl_hf

    loaded = etl_hf.load_hf("path/to/checkpoint")      # config + host params
    logits = etl_hf.predict(loaded, ids, seq_len=32)   # numpy backend, one call
    oracle = etl_hf.reference_logits(loaded.config, loaded.params, ids)

    # ...or stage the graph yourself:
    fwd = etl_hf.forward_fn(loaded.config, seq_len=32)     # a pure @etl.defn
    exe = etl_hf.build(fwd, loaded.config)                 # numpy interpreter
    exe = etl_hf.build(fwd, loaded.config, backend="iree") # compiled path
    logits = etl_hf.run(exe, loaded.params, ids)           # an etl.Tensor

    device = etl.core.Device("cuda", 0)
    exe = etl_hf.build(fwd, loaded.config, backend="iree", device=device)
    out = etl_hf.run(
        exe,
        etl_hf.place_params(loaded.params, device),
        etl_hf.place_ids(ids, device),
    )

Import-time constraint: ``import etl_hf`` requires only ``numpy`` (plus ``etl``).
torch / transformers / safetensors / iree are **never** imported at module scope.
"""

from __future__ import annotations

from etl_hf import (
    config,
    families,
    load,
    ops,
    pack,
    pipeline,
    program,
    reference,
    weights,
)
from etl_hf.config import ModelConfig
from etl_hf.errors import (
    HFConfigError,
    HFError,
    HFUnsupportedError,
    HFWeightError,
)
from etl_hf.families import FAMILIES, FamilySpec, family_for_model_type, get_family
from etl_hf.load import (
    LoadedModel,
    load_config,
    load_hf,
    load_params,
    validate_params,
)
from etl_hf.pack import PackPlan, pack_params, pack_plan
from etl_hf.pipeline import build, place_ids, place_params, predict, run
from etl_hf.program import forward_fn, param_specs, reference_logits

__version__ = "0.1.0"

__all__ = [
    # -- loading (checkpoint -> static config + host params) ------------------
    "LoadedModel",
    "ModelConfig",
    "load_config",
    "load_hf",
    "load_params",
    "validate_params",
    # -- the compute function ------------------------------------------------
    "forward_fn",
    "param_specs",
    "reference_logits",
    # -- staging (build / run / predict / explicit placement) ----------------
    "build",
    "place_ids",
    "place_params",
    "predict",
    "run",
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
    "config",
    "families",
    "load",
    "ops",
    "pack",
    "pipeline",
    "program",
    "reference",
    "weights",
    # -- metadata -----------------------------------------------------------
    "__version__",
]
