# etl/backends — backend abstraction, numpy reference backend, StableHLO exporter

## Intent

Turn verified frontend Graphs into executable programs. Three responsibilities:

1. **Backend contract** — `Capabilities`, `Backend` ABC, backend `Executable` protocol, registry. This is the plug-in seam for compilers (IREE/XLA/TVM are future integration points, never implemented here).
2. **Reference numpy CPU backend** (`numpy_backend`, default) — a pure-Python numpy interpreter proving the IR semantics; the only concrete-computation path besides `core`'s concrete creators.
3. **StableHLO export utility** (`stablehlo`) — emits StableHLO MLIR text so external compilers (`iree-compile model.mlir -o model.vmfb`) can take over compilation. Export-only in v1, NOT a compilable backend.

**Ownership (binding):** `LoweredProgram` and `CompiledArtifact` are owned by THIS package, not by `pipeline`. `pipeline` orchestrates staging and re-exports them. See `../../CONTEXT.md` (staging pipeline, error strategy) and `../CONTEXT.md` (cross-module contracts) — both binding.

## API Surface

Must-expose names (re-exported from `etl/backends/__init__.py` and from `etl`):

| Name | Where | Notes |
|---|---|---|
| `Capabilities` | `backend.py` | frozen dataclass: `dynamic_shapes: bool`, `dtypes: frozenset`, `collectives: bool`, `runtime_calls: bool`, `custom_blocks: bool`, `async_collectives: bool` |
| `Backend` | `backend.py` | ABC: `name: str` (class attr), `capabilities: Capabilities`; `lower(graph, options=None) -> LoweredProgram`; `compile(lowered, options=None) -> CompiledArtifact`; `load(artifact, device=None) -> executable` |
| `Executable` | `backend.py` | `Protocol` (runtime-checkable): `run(flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]`; attrs `.functions` (tuple of function names), `.device`; optional `.save(path)` / classmethod `.load(path, device=None)` |
| `Signature` | `program.py` | frozen dataclass: `input_tree`/`output_tree` (core.TreeSpec), `input_specs`/`output_specs` (per-leaf core.TensorSpec), `static_values` — passed down from the Graph at `lower()` time |
| `LoweredProgram` | `program.py` | attrs `.backend` (name str), `.signature`, `.payload` (backend-specific, serializable); methods `.text() -> str`, `.save(path)`, classmethod `.load(path)` |
| `CompiledArtifact` | `program.py` | attrs `.backend`, `.signature`, `.target`, `.payload`; self-describing: records required custom ops + runtime dependencies; methods `.save(path)`, classmethod `.load(path)` |
| `register(backend)`, `get(name) -> Backend` | `registry.py` | `get` raises `core.BackendError` for unknown names; duplicate name with a different instance raises `BackendError` |
| `numpy_backend` | `numpy/__init__.py` | `NumpyBackend` instance, registered at import; the DEFAULT backend |
| `NumpyBackend`, `NumpyExecutable` | `numpy/__init__.py` | reference CPU interpreter |
| `stablehlo` | `stablehlo/__init__.py` | submodule; `stablehlo.export(graph_or_module) -> str` (MLIR text) |

## Constraints (binding)

- **Import acyclicity** (from `../CONTEXT.md`): top-level imports restricted to `etl.core` and `etl.ir`. `etl.ops` may be imported ONLY inside function bodies (block-impl registration). `etl.persist` may be imported ONLY inside function bodies (persist sits ABOVE backends in the DAG — a top-level import would be a cycle). NEVER import `etl.pipeline` (pipeline imports backends).
- **Errors**: capability violations and unsupported ops raise `core.BackendError` — always naming the op/feature. Load-time mismatches (backend/device/ABI/custom-op availability) raise `core.PersistenceError` — never silently re-lower/re-compile. `lower()` surfaces `core.VerificationError` from `graph.verify()`. Runtime shape failures raise `core.ShapeError`. No silent fallbacks or partial semantics.
- **No hidden staging**: `load()` never traces/lowers/compiles. Artifacts are self-describing (backend name/version, target, signature, required custom ops, IR format version).
- **Files < ~1000 lines** — kernel modules are pre-split by category (`numpy/kernels/`); stablehlo mapping data lives separately from the writer.
- **CPU only in v1**: the numpy backend never requires or touches GPU.

## Numpy backend design (reference CPU interpreter)

**Capabilities**: name `"numpy"`; `dynamic_shapes=True`; `dtypes=` all numpy dtypes; `collectives=True` (single-process simulation); `runtime_calls=True`; `custom_blocks=True`; `async_collectives=False` (simulation is synchronous).

**Staging flow** (implemented — see `numpy/__init__.py`):
- `lower(graph, options=None)`: (1) `graph.verify()`; (2) capability pre-check (v1 numpy supports everything — the check pattern stays); (3) inline `block_call` portable decompositions as graph→graph expansion (`numpy/inline.py` splices the traced portable defn; block has neither portable decomposition nor registered numpy impl ⇒ `BackendError` naming the block); (4) record `Signature` from the Graph's LIVE attributes (input/output TreeSpec + per-leaf specs + static values — passed down, not re-derived); (5) `payload` = versioned self-describing `ir.Module` serialization (`ir.serialize_module`).
- `compile(lowered, options=None)`: validate `lowered.backend == "numpy"`; wrap the serialized module into a `CompiledArtifact` (`target="cpu"`, records required custom ops + `runtime_dependencies={"numpy": <version>}`). No machine code exists — the artifact IS serialized IR.
- `load(artifact, device=None)`: validate backend/device (None or CPU; else `BackendError`/`DeviceError`); `ir.deserialize_module`; build a `NumpyExecutable`. Never re-compiles.

**Interpreter execution model** (`NumpyExecutable.run(flat_input_tensors)`):
- **Execution order = block op order.** This IS the effect ordering (write/read/collective/callback ops anchor order); pure ops could be reordered but program order is kept for determinism.
- **Shape inference: reuse, don't duplicate.** Runtime shapes are computed from concrete input shapes using the SAME ops-level inference rules with symbolic dims bound to concrete values (`numpy/shapes.py` evaluates `DimExpr` against name→int bindings). The backend carries NO second copy of shape rules. Free symbolic dims at run time ⇒ `ShapeError`.
- **dtypes**: numpy dtype ↔ etl dtype is 1:1; kernels validate support; no promotion beyond what ops define.
- **Control flow**: `cond`/`while_loop`/`scan` region ops are interpreted by recursively running region blocks — genuinely dynamic runtime control flow (the graph is NOT specialized per iteration).
- **`runtime_call`**: executes the Python callback synchronously at the op's position — a documented sync point (no async execution in v1).
- **`block_call` dispatch**: portable decompositions are inlined at `lower()` time, so the interpreter only ever dispatches blocks that have a registered numpy impl (`kernels/custom.py`); otherwise `BackendError`.
- **Collectives**: the interpreter dispatches ALL collective ops through the canonical `CollectiveExecutor` hook in `etl.dist.context` (`dist.context.set_collective_executor`/`get_collective_executor`). The numpy backend installs its default `SingleRankCollectiveExecutor` (identity semantics on a single rank) into that slot at import time; tests simulate multi-rank in-process by installing a custom executor there. `numpy/collectives.py` only defines the default executor class. Group semantics are simulation-only in v1.
- **rank/world_size**: `rank`/`world_size` graph scalars are resolved at RUN time from a per-execution `dist.context.RankContext` (default rank 0 / world_size 1; overridable per run via `numpy/exec_context.py`'s thread-local `set_rank_context` or the `run(..., rank_context=...)` kwarg) — never constant-folded at lower/compile time.
- **`NumpyExecutable` persistence**: `.save(path)` saves the underlying `CompiledArtifact` (documented: the executable is reconstructed explicitly at `load` — device handles are never serialized).

**Kernel split** (`numpy/kernels/`): `elementwise.py` (arith/activations/comparisons/select/cast/broadcast), `reductions.py` (reduce_*, argmax/argmin), `indexing.py` (reshape/transpose/slice/concat/pad/gather/scatter), `linalg.py` (dot/conv), `control_flow.py` (cond/while/scan region execution), `collective.py` (collective op dispatch via the hook), `custom.py` (`runtime_call`, `block_call` dispatch). Dispatch table assembled in `kernels/__init__.py` (op name → kernel).

## StableHLO exporter — v1 scope

Export utility ONLY: `stablehlo.export(graph_or_module) -> str` produces StableHLO MLIR text. Output is compiler input for external tools (`iree-compile model.mlir -o model.vmfb`); IREE/XLA/TVM adapters are FUTURE integration points — documented, not implemented.

**v1 mapping table** (`stablehlo/ops.py`; exact stablehlo op names verified against the StableHLO spec at implementation time):

| etl op(s) | StableHLO |
|---|---|
| add/subtract/multiply/divide/power/remainder/maximum/minimum | `stablehlo.add/subtract/multiply/divide/power/remainder/maximum/minimum` |
| abs/negate/sqrt/sign | `stablehlo.abs/negate/sqrt/sign` |
| exp/log/log1p/sin/cos/tan/tanh/sigmoid/erf | `stablehlo.exponential/log/log_plus_one/sine/cosine/tan/tanh/logistic/erf` |
| square | decompose → `multiply(x, x)` |
| relu | decompose → `maximum(x, 0)` |
| gelu | decompose → erf-based (`0.5*x*(1+erf(x/√2))`) |
| stop_gradient | identity passthrough (emit operand directly) |
| bitwise_and/or/xor; logical_and/or/not | `stablehlo.and/or/xor/not` |
| equal/not_equal/less/less_equal/greater/greater_equal | `stablehlo.compare` + `comparison_direction` attr (EQ/NE/LT/LE/GT/GE) |
| cast | `stablehlo.convert` |
| select / broadcast / reshape / transpose / slice / concatenate / pad | `stablehlo.select/broadcast_in_dim/reshape/transpose/slice/concatenate/pad` |
| reduce_sum/max/min/mean/prod | `stablehlo.reduce` (mean: reduce-sum then divide) |
| argmax / argmin | `stablehlo.argmax/argmin` |
| dot / conv | `stablehlo.dot_general` / `stablehlo.convolution` |
| constant | `stablehlo.constant` |
| cond / while_loop | `stablehlo.if` / `stablehlo.while` |
| collectives (all_reduce/all_gather/reduce_scatter/all_to_all/broadcast/collective_permute) | `stablehlo.all_reduce/all_gather/reduce_scatter/all_to_all/collective_broadcast/collective_permute` |
| symbolic dims | `?` dynamic dims in tensor types, e.g. `tensor<?xNxf32>` |

**Deferred in v1** (⇒ `core.BackendError` naming the op, message suggests decomposition or a future adapter): `gather`, `scatter`, `scan`, `runtime_call`, `block_call` (blocks with no portable decomposition), `dist.rank()`/`dist.world_size()` graph scalars, complex-number elementwise beyond cast.

Type map (dtype → MLIR): float16→f16, float32→f32, float64→f64, int8→i8, int16→i16, int32→i32, int64→i64, uint8→ui8, uint16→ui16, uint32→ui32, uint64→ui64, bool→i1, complex64→`complex<f32>`, complex128→`complex<f64>`.

## Serialization compliance

`LoweredProgram.save`/`CompiledArtifact.save` delegate to the `etl.persist` container format (lazy import): magic header + format version + JSON metadata + payload + SHA-256 integrity. Metadata is self-describing: backend name/version, target, signature (TreeSpecs + specs + static values), required custom ops, IR format version. `load()` validates the recorded backend against the registry (numpy reconstruction handled by `NumpyBackend`) — mismatch or missing custom op ⇒ `PersistenceError`. Never silently re-traces/re-compiles.

## Test strategy

`../../tests/backends/` (sibling — read-only from here; test-related writes escalate to root):
- `registry.py`: register/get/duplicate/unknown-name behavior.
- numpy interpreter: per-op coverage per kernel category; symbolic-dim binding at run time; dynamic control flow; `runtime_call` sync execution; block impl dispatch (impl vs portable decomposition vs missing ⇒ BackendError); collectives — single-rank identity AND multi-rank in-process simulation via the `CollectiveExecutor` hook; persistence round-trips (artifact save/load, backend mismatch ⇒ PersistenceError).
- stablehlo: golden-text exports for the v1 table (elementwise, reduce, dot, if/while, symbolic-dims rendering), deferred ops ⇒ BackendError naming the op.
- CPU only, pytest, numpy-only deps.

## Routing table

| Path | Area |
|---|---|
| `./backend.py` | `Capabilities`, `Backend` ABC, `Executable` protocol |
| `./program.py` | `Signature`, `LoweredProgram`, `CompiledArtifact` (owned by backends) |
| `./registry.py` | `register`/`get` |
| `./numpy/` | Reference numpy CPU interpreter: `NumpyBackend`, `NumpyExecutable`, `kernels/` (per-category kernels), `shapes.py` (DimExpr evaluation), `collectives.py` (CollectiveExecutor hook) |
| `./stablehlo/` | StableHLO MLIR export utility: `export`, `ops.py` (mapping table data), `writer.py` (MLIR text emission) |

## Notes for agents

- **Architecture phase**: all behavioral bodies raise `NotImplementedError`; only trivial pure-type code (dataclasses, protocols, mapping data, registry) is implemented. Implementation is delegated to `subagent_manager` at this node by the parent orchestrator.
- Contract conflicts found while architecting: the cross-module bullet says backends import `etl.ops` (not `etl.block`) for block-impl registration — the registration pathway must be finalized with the `ops`/`block` owners during implementation (`numpy/__init__.py::_register_block_impls` lazy-imports `etl.ops` inside the function body per the letter of the contract).
- `../../tests/` is a sibling: read-only, escalate writes to root.
