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
| `CollectiveExecutor` | `collectives.py` | runtime-checkable Protocol — CANONICAL home is `etl/dist/context.py` (`dist.context.CollectiveExecutor`); re-exported here as an alias: `all_reduce(tensor, group, op)`, `all_gather(tensor, axis, group)`, `reduce_scatter(tensor, axis, group)`, `all_to_all(tensor, axis, group)`, `broadcast(tensor, src_rank, group)`, `collective_permute(tensor, mapping, group)` — all `-> Tensor` (group/mapping are graph-constant objects) |
| `SingleRankCollectiveExecutor` | `collectives.py` | DEFAULT executor: every collective returns the tensor unchanged (identity semantics on one rank); installed into the `dist.context` slot at `numpy/__init__` import time |
| `set_collective_executor` / `get_collective_executor` | `etl/dist/context.py` (CANONICAL) | process-wide hook; numpy installs `SingleRankCollectiveExecutor()` at import (tests simulate multi-rank in-process) |
| `evaluate_dim_expr` / `evaluate_shape` | `shapes.py` | runtime `Dim`/`DimExpr` evaluation against name→int bindings (free dims / div-by-zero / negative results ⇒ `ShapeError`) |
| `KernelContext` / `Interpreter` / `entry_function` | `interpreter.py` | the execution engine (see "Implemented execution engine" below) |
| `set_rank_context` / `get_rank_context` | `exec_context.py` | thread-local `RankContext` hook (default `RankContext(rank=0, world_size=1)`; per-run override on `NumpyExecutable.run(..., rank_context=...)`) |
| `dispatch(op_name)` / `register_all()` / `KERNEL_TABLE` | `kernels/__init__.py` | op-name → kernel dispatch table; `register_all()` idempotent, duplicate keys across category modules ⇒ `BackendError` |

Not public: `_register_block_impls` (documented no-op — block dispatch resolves via `etl.block.registry` at lower/load/run time), `_module_function_names` (single touch point for the `ir.Module` accessor). The bottom-up op walk (`iter_ops`/`iter_block_ops`), the portable-splicing helpers (`clone_ops_into` / `drop_op_uses`), and the fixpoint driver (`inline_portables`) MOVED to the shared `../inline.py` (also used by compiler backends); `./inline.py` is a thin re-export.

## Implemented execution engine

`interpreter.py` holds the engine the staging flow and every kernel build on:

- **`KernelContext`** — per-execution context handed to kernels. Attributes: `.bindings` (dim name → int), `.rank_context` (from `exec_context.get_rank_context()`), `.module` (ir.Module). Methods: `.run_region(region, arg_tensors)` (bind entry-block arguments to tensors, extend `.bindings` positionally from region-arg types vs concrete shapes — Dim binds by name with conflict checking, DimExpr binds its Dim leaves, int must equal, None unchecked — then run the block ops and return the `return` terminator's operand tensors), `.compute_output_shapes(op, input_shapes, input_dtypes)` (evaluates `op.results[i].type.shape` against `.bindings`; None dims stay None — the IR result types were already inferred by ops-level inference at trace time, so this is the mandated shape-rule reuse, never a second copy of shape rules), `.resolve_callback(callback_id)` (lazy `etl.ops.constant._get_callback`; missing ⇒ `BackendError` naming the id), `.evaluate_shape(shape)` (convenience over `shapes.evaluate_shape`).
- **`Interpreter`** — holds module + signature. `run(flat_input_tensors, rank_context=None)`: resolves the entry function (`module.get_function("main")`, fallback `module.main` when exactly one function); validates input count, per-input dtype (⇒ `DTypeError`) and per-input shape against spec shapes (Dim binds by name with conflict/known-size checks, DimExpr binds leaves then evaluates and compares, int must equal, None unchecked — ⇒ `ShapeError`); wraps the run in `exec_context.set_rank_context(rank_context)` with try/finally restore; executes the entry region; validates output count + dtype vs `signature.output_specs`.
- **Op loop** (`_run_block`, used recursively for nested regions): env keyed by `value.id`; for each op in block order — `return` is special-cased as the terminator; otherwise `kernels.dispatch(op.name)` (unknown ⇒ `BackendError` naming the op), call `kernel(ctx, op, operands)`, normalize single `Tensor` → 1-tuple, validate result count, per-result dtype (exact match ⇒ `BackendError` — kernels never silently coerce) and per-result shape against evaluated `op.results[i].type.shape` with None dims unchecked (mismatch ⇒ `ShapeError`), store into env.

## Interpreter design (binding from parent)

**Staging flow** (`NumpyBackend`; implemented):
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

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | `NumpyBackend` (lower/compile/load — incl. lower-time block_call portable inlining via the shared `../inline.py::inline_portables(keep_backend_impls="numpy")`: `get_impl("numpy")` → keep, `get_portable` → splice, neither → `BackendError`), `NumpyExecutable`, `numpy_backend` + registration, `_register_block_impls` (documented no-op), re-exports, import-time `dist.context.set_collective_executor(SingleRankCollectiveExecutor())` |
| `./interpreter.py` | `KernelContext` (bindings, rank_context, module, run_region, compute_output_shapes, resolve_callback, evaluate_shape), `Interpreter` (run + `_run_block` op loop + entry-function resolution + `_env_stack` outer-value resolution for nested regions), `entry_function` |
| `./exec_context.py` | thread-local `set_rank_context`/`get_rank_context` hook over `dist.context.RankContext` (default rank=0, world_size=1) |
| `./inline.py` | thin re-export of the shared `../inline.py` (`clone_ops_into` — portable block_call splicing with fresh ids + Use bookkeeping + output-type guards; `drop_op_uses` — required before `Block.erase` under strict `ir.verify`; plus `inline_portables`/`iter_ops`/`iter_block_ops`) |
| `./collectives.py` | `CollectiveExecutor` alias of `dist.context.CollectiveExecutor` (CANONICAL home), `SingleRankCollectiveExecutor` (identity default) |
| `./shapes.py` | `evaluate_dim_expr` / `evaluate_shape` — runtime `Dim`/`DimExpr` evaluation (shape-rule reuse) |
| `./kernels/__init__.py` | `KERNEL_TABLE`, `dispatch(op_name)`, `register_all()` (idempotent) — the kernel contract is documented in this module's docstring and is binding for all category modules |
| `./kernels/elementwise.py` | add/subtract/multiply/divide/power/remainder/maximum/minimum/abs/negate/square/sqrt/exp/log/log1p/sin/cos/tan/tanh/sigmoid/relu/gelu/erf/sign/bitwise_*/logical_*/cast/equal/not_equal/less/less_equal/greater/greater_equal/select/broadcast/stop_gradient |
| `./kernels/reductions.py` | reduce_* family (reduce_sum/reduce_max/reduce_min/reduce_mean/reduce_prod), argmax/argmin (cumsum moved out to indexing.py) |
| `./kernels/indexing.py` | reshape/transpose/slice/gather/scatter/concatenate/pad/tril/triu/cumsum |
| `./kernels/linalg.py` | dot/conv/solve |
| `./kernels/control_flow.py` | if/while/call — recursive region execution (the `return` terminator is special-cased by the interpreter loop, never dispatched) |
| `./kernels/collective.py` | dist collective ops dispatched through `dist.context.get_collective_executor()` (rank/world_size resolved from the per-run `RankContext`) |
| `./kernels/custom.py` | `constant`, `runtime_call` (sync callback via `ctx.resolve_callback` — artifacts with `runtime_call` require the same callback registrations at load time), `block_call` (registered numpy impl dispatch) |

Sibling: `../../tests/` → test suite (read-only from here; escalate test-related writes to root). Parent: `../` → Backend ABC, `LoweredProgram`/`CompiledArtifact`/`Signature` (owned there), registry (optional-adapter auto-activation), StableHLO exporter, shared `inline.py` block-inlining machinery, `compiler.py` (`CompilerBackend`/`CompilerExecutable`), `adapters/` (separate parallel effort).

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

- **Kernel dispatch coverage (complete)**: all 74 non-`return` IR op names are registered in `KERNEL_TABLE`; the `return` terminator is special-cased by the interpreter loop and is never dispatched. `register_all()` is idempotent; duplicate keys across category modules ⇒ `BackendError`.
- `_register_block_impls()` is a documented no-op — block dispatch resolves via `etl.block.registry` at lower/load/run time (`get_impl(name, "numpy")` keeps the op; `get_portable(name)` is traced and spliced at lower; neither ⇒ `BackendError` naming the block). User blocks register via `BlockOp.impl("numpy")`/`.portable(...)`.
- **Block-impl call convention (finalized with `etl.block`)**: a registered numpy impl is called as `impl(*numpy_arrays, **static_args) -> ndarray | tuple[ndarray, ...]` — operand tensors arrive as raw numpy arrays (`.numpy()`), the op's JSON-able `static_args` attribute arrives as kwargs, and the return is normalized and validated exactly like `runtime_call` (count/dtype/shape against the declared result specs).
- **Known v1 collective-protocol limitations** (documented, never silently worked around): the `CollectiveExecutor` protocol's `reduce_scatter` has NO `reduce_op` parameter, so the op's `reduce_op` attr is not forwarded to the executor; `all_to_all` takes a single `axis`, so only `split_axis` is forwarded (`concat_axis` is not — v1 protocol limitation).
- **Output validation with runtime-dynamic dims**: op result shapes with `None` (runtime-dynamic) dims are UNCHECKED per-dim by the interpreter (rank is still validated exactly); every non-`None` dim must evaluate to the concrete runtime dim (⇒ `ShapeError` otherwise).
- **Nested-region outer-value resolution**: region ops may legally reference SSA values defined in ENCLOSING blocks (e.g. `etl.scan`'s desugared regions capture the length constant and the `xs` leaves — they are not while-op operands). The interpreter resolves operands against an env STACK (`Interpreter._env_stack`, innermost first); a miss ⇒ `BackendError` naming the value (invalid module, never a silent skip).
- **0-d scalar normalization**: numpy returns bare scalars (`np.bool_`/`np.float64`/…) for 0-d ufunc/matmul/take/slice results; kernels normalize with `np.asarray` to the 0-d ndarray `core.Tensor` requires (no-op for real arrays; dtype never changes — this is shape normalization, not coercion).
- `_module_function_names` is the single touch point for the `ir.Module` accessor API (now finalized).
- `Capabilities.dtypes` includes all `numpy.sctypes` entries (+`bool_`) per the binding contract, including non-numeric "others" dtypes — per-op kernels validate concrete dtype support at run time (no silent coercion).
- Kernel call convention (finalized, binding — see the docstring of `kernels/__init__.py`): `kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]`; `ctx` is `KernelContext` (`.bindings`, `.rank_context`, `.module`, `.run_region`, `.compute_output_shapes`, `.resolve_callback`, `.evaluate_shape`). The interpreter validates outputs against `op.results` types (dtype exact; symbolic dims via `ctx.bindings`; None dims unchecked).
- `runtime_call` callback resolution goes through `etl.ops.constant._get_callback` (via `ctx.resolve_callback`) — artifacts containing `runtime_call` require the same callback registrations at load time.
- Collectives dispatch through `dist.context.get_collective_executor()`; the identity `SingleRankCollectiveExecutor` is installed at `numpy/__init__` import time. rank/world_size resolve from the per-execution `RankContext` (`exec_context.py`; override via `NumpyExecutable.run(..., rank_context=...)`).
- `etl.dist` never imports backends, so `exec_context.py` and `__init__.py` may import `dist.context` at top level (acyclic). `etl.ops`, `etl.block`, `etl.trace`, `etl.persist` stay function-body-lazy.
- `ir.verify` enforces strict use-bookkeeping: erase a spliced `block_call` only after `drop_op_uses` removes its `Use` records (shared `../inline.py`).
