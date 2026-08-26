# etl/backends/numpy — reference numpy CPU interpreter backend (default backend)

## Intent

The **reference CPU interpreter backend** (name `"numpy"`, the DEFAULT backend): a pure-Python numpy interpreter proving EvoXIR semantics. It is one of only two concrete-computation paths in etl (the other being `core`'s concrete creators). Compiler-neutral by design: it does NOT optimize, schedule, or generate code — external compilers (IREE/XLA/TVM via the StableHLO export path) own compilation.

Two roles:

1. **Staging** (`NumpyBackend`): verified `Graph` → `LoweredProgram` (payload = versioned self-describing `ir.serialize_module`) → `CompiledArtifact` (target `"cpu"`; the artifact IS serialized IR — there is no machine code) → `NumpyExecutable` at `load`.
2. **Interpretation** (`NumpyExecutable.run`): executes the IR on concrete `core.Tensor`s (numpy arrays in v1) in block op order.

The parent contract (`../CONTEXT.md`, "Numpy backend design" section) is **BINDING** — the design below restates it, never modifies it. Root principles (`../../CONTEXT.md`) and the package contract (`../CONTEXT.md`) are also binding.

## API Surface

Exports (re-exported via `etl/backends/__init__.py` and surfaced at `etl.backends`):

| Name | Where | Notes |
|---|---|---|
| `NumpyBackend` | `__init__.py` | `Backend` ABC impl; `name="numpy"`; `capabilities`: `dynamic_shapes=True`, `dtypes=` all numpy dtypes (numpy.sctypes flattening + `bool_`), `collectives=True` (single-process simulation), `runtime_calls=True`, `custom_blocks=True`, `async_collectives=False` |
| `NumpyExecutable` | `__init__.py` | satisfies the `Executable` protocol: attrs `functions` (module function names), `device`, `signature`, `artifact`; `run(flat_input_tensors) -> flat_outputs`, `save(path)`, classmethod `load(path, device=None)` |
| `numpy_backend` | `__init__.py` | `NumpyBackend()` instance, registered via `..registry.register` at import time |
| `CollectiveExecutor` | `collectives.py` | runtime-checkable Protocol: `all_reduce(tensor, group, op)`, `all_gather(tensor, axis, group)`, `reduce_scatter(tensor, axis, group)`, `all_to_all(tensor, axis, group)`, `broadcast(tensor, src_rank, group)`, `collective_permute(tensor, mapping, group)` — all `-> Tensor` (group/mapping are graph-constant objects) |
| `SingleRankCollectiveExecutor` | `collectives.py` | DEFAULT executor: every collective returns the tensor unchanged (identity semantics on one rank) |
| `set_collective_executor` / `get_collective_executor` | `collectives.py` | module-level hook; default installed unless set (tests simulate multi-rank in-process) |
| `evaluate_dim_expr` / `evaluate_shape` | `shapes.py` | runtime `Dim`/`DimExpr` evaluation against name→int bindings |
| `dispatch(op_name)` / `register_all()` / `KERNEL_TABLE` | `kernels/__init__.py` | op-name → kernel dispatch table |

Not public: `_register_block_impls` (import-time block-impl wiring; see Notes for agents), `_module_function_names` (single touch point for the `ir.Module` accessor, to be finalized with ir owners).

## Interpreter design (binding from parent)

**Staging flow** (`NumpyBackend`; stubs raise `NotImplementedError` in this phase, docstrings encode the design):
- `lower(graph)`: (1) `graph.verify()` (surfaces `core.VerificationError`); (2) capability pre-check (v1 numpy supports everything — the check pattern stays); (3) inline `block_call` portable decompositions as graph→graph expansion at LOWER time (block with neither portable decomposition nor registered numpy impl ⇒ `core.BackendError` naming the block); (4) record `Signature` from the Graph (input/output TreeSpec + per-leaf specs + static values — passed down, never re-derived); (5) payload = `ir.serialize_module(graph.module)`.
- `compile(lowered)`: validate `lowered.backend == "numpy"` (else `BackendError`); record `required_custom_ops` + `runtime_dependencies` (self-describing); wrap serialized module as `CompiledArtifact(target="cpu")`. No machine code.
- `load(artifact, device)`: validate backend/device (None or CPU; else `BackendError`/`DeviceError`), required custom ops availability; `ir.deserialize_module`; build `NumpyExecutable`. **Never re-compiles.**

**Execution model** (`NumpyExecutable.run`):
- **Execution order = block op order.** This IS the effect ordering (write/read/collective/callback ops anchor order); pure ops keep program order for determinism.
- **Shape inference: reuse, don't duplicate.** Runtime shapes = ops-level inference rules with symbolic dims bound to concrete values (`shapes.py` evaluates `DimExpr` against name→int bindings). The backend carries NO second copy of shape rules. Free symbolic dims ⇒ `ShapeError`.
- **dtypes**: numpy dtype ↔ etl dtype is 1:1; kernels validate support; no promotion beyond what ops defines.
- **Control flow**: `cond`/`while_loop`/`scan` region ops are interpreted by recursively running region blocks — genuinely dynamic runtime control flow (the graph is NOT specialized per iteration).
- **`runtime_call`**: executes the Python callback synchronously at the op's position — a documented sync point (no async execution in v1).
- **`block_call` dispatch**: portable decompositions are inlined at `lower()` time, so the interpreter only ever dispatches blocks that have a registered numpy impl (`kernels/custom.py`); otherwise `BackendError` (safety net, never a fallback).
- **Collectives**: ALL collective ops dispatch through the `CollectiveExecutor` hook; default `SingleRankCollectiveExecutor` = identity on one rank; tests install multi-rank simulators via `set_collective_executor`. Group semantics are simulation-only in v1.
- **Persistence**: `NumpyExecutable.save` saves the underlying `CompiledArtifact`; the executable is reconstructed explicitly at `load` — device handles are never serialized.

## Error behavior (binding)

- Unsupported ops/features encountered at run time (capability drift) ⇒ `core.BackendError` **naming the op** — never a silent skip.
- Free symbolic dims at run time ⇒ `core.ShapeError`.
- `lower()` surfaces `core.VerificationError` from `graph.verify()` as-is.
- Load-time mismatches (backend/device/ABI/custom-op availability) ⇒ `core.PersistenceError` — **never** a silent re-lower/re-compile.
- No silent fallbacks or partial semantics anywhere in this package.

## Constraints

- **Import acyclicity (binding)**: top-level imports restricted to `etl.core` and `etl.ir`; `etl.ops` may be imported ONLY inside function bodies (`_register_block_impls` is the sole allowed site — lazy import); NEVER import `etl.pipeline` or `etl.persist` at top level (persist is lazy inside `save`/`load` bodies per `../program.py`).
- **Files < ~1000 lines**: kernels are pre-split by category (`kernels/`); split along the declared category boundaries instead of growing files.
- **CPU only in v1**: never requires or touches GPU.
- **Architecture phase**: behavioral bodies raise `NotImplementedError`; only trivial pure-type code is live (capabilities declaration, `SingleRankCollectiveExecutor` identity bodies, collective hook, `NumpyExecutable` storage constructor).

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | `NumpyBackend`, `NumpyExecutable`, `numpy_backend` + registration, `_register_block_impls`, re-exports |
| `./collectives.py` | `CollectiveExecutor` protocol, `SingleRankCollectiveExecutor` (default), `set/get_collective_executor` hook |
| `./shapes.py` | `evaluate_dim_expr` / `evaluate_shape` — runtime `Dim`/`DimExpr` evaluation (shape-rule reuse) |
| `./kernels/__init__.py` | `KERNEL_TABLE`, `dispatch(op_name)`, `register_all()` — dispatch-table design |
| `./kernels/elementwise.py` | add/subtract/multiply/divide/power/remainder/maximum/minimum/abs/negate/square/sqrt/exp/log/log1p/sin/cos/tan/tanh/sigmoid/relu/gelu/erf/sign/bitwise_*/logical_*/cast/equal/not_equal/less/less_equal/greater/greater_equal/select/broadcast/stop_gradient |
| `./kernels/reductions.py` | reduce_sum/reduce_max/reduce_min/reduce_mean/reduce_prod + sum/max/min/mean/prod + argmax/argmin |
| `./kernels/indexing.py` | reshape/transpose/slice/gather/scatter/concatenate/pad |
| `./kernels/linalg.py` | dot/conv |
| `./kernels/control_flow.py` | cond/while_loop/scan region execution (recursive region runs) |
| `./kernels/collective.py` | dist collective ops dispatched through the `CollectiveExecutor` hook |
| `./kernels/custom.py` | `runtime_call` (sync callback execution), `block_call` (registered numpy impl dispatch) |

Sibling: `../../tests/` → test suite (read-only from here; escalate test-related writes to root). Parent: `../` → Backend ABC, `LoweredProgram`/`CompiledArtifact`/`Signature` (owned there), registry, StableHLO exporter.

## Test strategy

Planned tests live in `../../tests/backends/numpy/` (sibling — read-only from here; test-related writes escalate to root), CPU only, pytest, numpy-only deps:
- **Per-op coverage per kernel category** (elementwise/reductions/indexing/linalg) against the ops-level semantics.
- **Symbolic dims**: runtime binding of `Dim`/`DimExpr` shapes; free symbolic dim ⇒ `ShapeError`.
- **Dynamic control flow**: `cond`/`while_loop`/`scan` genuinely dynamic (per-iteration shapes, early exit).
- **`runtime_call`**: synchronous execution at the op position; output-spec mismatch ⇒ `BackendError`.
- **Block dispatch**: registered impl vs portable decomposition (inlined at lower) vs missing ⇒ `BackendError` naming the block.
- **Collectives**: single-rank identity AND multi-rank in-process simulation via `set_collective_executor` (shared module state — reset in teardown).
- **Persistence round-trips**: `LoweredProgram`/`CompiledArtifact`/`NumpyExecutable` save/load; backend mismatch ⇒ `PersistenceError`; never recompiles.

## Notes for agents

- **Architecture phase now**: all behavioral bodies raise `NotImplementedError` (docstrings encode the binding design); trivial pure-type code is implemented. Implementation is delegated to `subagent_manager` at this node by the parent orchestrator.
- `_register_block_impls()` is defined but NOT called at import time yet — the exact ops-level block-impl registration hook must be finalized with the `etl/ops` and `etl/block` owners during implementation (contract conflict noted in `../CONTEXT.md`); calling it now would break package imports.
- `_module_function_names` is the single touch point for the `ir.Module` accessor API — update it once `etl/ir` lands.
- `Capabilities.dtypes` includes all `numpy.sctypes` entries (+`bool_`) per the binding contract, including non-numeric "others" dtypes — per-op kernels validate concrete dtype support at run time (no silent coercion).
- Kernel call convention (to be finalized with the interpreter loop): `kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]`; `ctx` carries interpreter state (shape-dim bindings, collective executor, runtime-call registry).
