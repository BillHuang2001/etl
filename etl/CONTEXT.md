# etl package — public API contract

## Intent

The `etl` package: an explicit, minimal tensor graph runtime. See root `../CONTEXT.md` for principles, value model, error strategy, and non-goals — all binding here. This file defines the **public API surface** and the **cross-module contracts** every submodule must honor.

## Public API surface (the contract — must exist, exact names)

**Staging & pipeline** (see also `pipeline.py`):
- `@etl.defn` → `Defn` (calling a `Defn` with concrete tensors must raise `TraceError` directing to `trace`/`evaluate`)
- `etl.trace(fn_or_defn, *specs) -> Graph` (accepts nested structures of `TensorSpec` + static values)
- `etl.lower(graph, backend=None, **options) -> LoweredProgram`
- `etl.compile(lowered, backend=None, **options) -> CompiledArtifact`
- `etl.load(artifact, backend=None, device=None) -> Executable`
- `etl.run(executable, *args) -> outputs` (structured inputs/outputs via TreeSpec; validates static fields)
- `etl.bind(obj, **bindings) -> BoundExecutable|callable` (pure argument-supply sugar; validates names/dtypes/shapes/devices; never alters the graph)
- `etl.build(fn, *specs, backend=None, device=None, **options) -> Executable` — documented shorthand for trace→lower→compile→load
- `etl.evaluate(fn, *args, backend=None, device=None, **options)` — documented shorthand: derive specs → build → run. Deriving a spec from a concrete tensor snapshots shape+dtype only.

**Value model** (owned by `core`):
- `etl.TensorSpec(shape, dtype, device=None, name=None)`; `etl.dim(name_or_int)` → `Dim`; `etl.dtype(obj)`; dtype constants `etl.float16/float32/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bool/complex64/complex128` (numpy dtype objects)
- Concrete creators: `etl.tensor(data, dtype=None, device=None)`, `etl.zeros/ones/full/empty(shape, dtype=...)`, `etl.from_numpy(array)`, `etl.from_dlpack(capsule_or_tensor)`
- `etl.constant(tensor) -> SymbolicTensor` (graph-time only; snapshots data; warns for large constants)
- Devices: `etl.devices(kind=None) -> list[Device]`, `etl.split_tensor(tensor, axis, devices) -> list[Tensor]`, `etl.replicate_tensor(tensor, devices) -> list[Tensor]`

**Tensor ops** (owned by `ops`; SymbolicTensor-in/SymbolicTensor-out; Python scalars auto-promote to scalar constants; concrete `Tensor` args raise `TraceError`):
`add, subtract, multiply, divide, power, remainder, maximum, minimum, abs, negate, square, sqrt, exp, log, log1p, sin, cos, tan, tanh, sigmoid, relu, gelu, erf, sign, bitwise_and/or/xor, logical_and/or/not, cast, equal, not_equal, less, less_equal, greater, greater_equal, select, broadcast, reshape, transpose, slice, gather, scatter, concatenate, pad, reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_prod, sum, max, min, mean, prod, argmax, argmin, dot, conv, stop_gradient, runtime_call(callback, *operands, result=TensorSpec...)`

**Control flow** (owned by `trace`): `etl.cond(pred, true_fn, false_fn, *operands)`, `etl.while_loop(cond_fn, body_fn, init)`, `etl.scan(f, init, xs, length=None)` — branch/body functions are traced into IR regions.

**Transforms** (owned by `transforms`, graph→graph): `etl.vectorize(graph_or_fn, axes) -> Graph`, `etl.vmap(fn_or_graph, in_axes=0, out_axes=0) -> callable|Graph`, `etl.grad(fn_or_graph, argnums=None) -> Graph`, `etl.jvp(fn_or_graph, tangents...) -> Graph`, `etl.vjp(fn_or_graph, cotangents...) -> Graph`. Unsupported ops (no rule) raise `TransformError` — never silently fall back.

**Distributed** (owned by `dist`): `etl.dist.group(name, ranks) -> Group`, `etl.dist.all_reduce/all_gather/reduce_scatter/all_to_all/broadcast/collective_permute(tensor, ..., group=Group)`, graph scalars `etl.dist.rank()`, `etl.dist.world_size()`.

**Custom ops** (owned by `block`): `etl.block(name, inputs=None, outputs=None, attributes=None, effects=None, batching=None, portable=None)` — decorator/factory returning a callable `BlockOp`; `BlockOp.impl(backend_name)`; `BlockOp.batching_rule(fn)`; `BlockOp.jvp_rule/vjp_rule(fn)`. No rule + no safe policy + no portable decomposition ⇒ `TransformError`.

**Backends** (owned by `backends`): `etl.backends.Backend` (abstract: `name`, `capabilities`, `lower`, `compile`, `load`), `etl.backends.register/get(name)`, `etl.backends.numpy_backend` (default CPU interpreter), `etl.backends.stablehlo` (`export(graph|module) -> str` MLIR text). `etl.lower/compile/load` default to `numpy_backend`.

**Persistence** (owned by `persist`, used by pipeline types): `Graph.save/load`, `LoweredProgram.save/load/.text()`, `CompiledArtifact.save/load`, `Executable.save/load`, `etl.Cache` (explicit; `FileCache(directory)` in `persist`). Loading never silently re-traces/recompiles; incompatibility fails clearly.

**Namespace**: `etl.numpy` (alias `enp`) + `etl.numpy.linalg` — numpy-style graph API producing the same IR as `ops`.

**Structured I/O**: tuple/list/dict/namedtuple/dataclass supported everywhere (trace inputs, run outputs, bind). Custom containers register via `etl.register_pytree_node`.

## Cross-module contracts (must-expose names)

- **`core`** exports: `Tensor`, `SymbolicTensor`, `TensorSpec`, `Dim`, `DimExpr`, `Device`, `TreeSpec` (`flatten`/`unflatten`/`register_pytree_node`), `dtype(obj)`, dtype constants, `ETLError` + subclasses, concrete creators, device helpers, and `register_operator_handlers(kind, handler)` — a hook dict (`add`/`sub`/`mul`/`matmul`/`getitem`/`truediv`/`pow`/`neg`/`lt`/`eq`/...) that `ops` populates at import time so `SymbolicTensor.__add__` etc. work without import cycles.
- **`ir`** exports: `Module`, `Function`, `Region`, `Block`, `Op`, `Value`, `Builder`, `Location`, `op registry` (`ir.opdef(name)`, op defs declare operands/results/attrs/effects), `verify(module) -> None` (raises `VerificationError`), `serialize_module/deserialize_module` (versioned, self-describing JSON container with integrity hash — format constant `IR_FORMAT_VERSION`), `pretty_print(module)`. Effect kinds: `pure`, `write`, `read`, `collective`, `callback`.
- **`ops`** exports: every function listed above + `runtime_call`; op functions build IR into the **active builder** obtained from `trace.current_builder()`; raise `TraceError` when no trace is active or a concrete `Tensor` operand appears; operator handlers registered into `core`.
- **`trace`** exports: `defn`, `trace`, `Graph` (attrs: `module`, `input_specs`, `output_tree`, `static_values`, `source_locations`; methods: `print()`, `verify()`, `save/load`, `flatten_inputs(args) -> flat_tensors`, `unflatten_outputs(flat) -> structured`, `validate_inputs(args) -> flat_tensors`), `cond`, `while_loop`, `scan`, `current_builder()` / builder-stack context, and `Defn` (stores fn + `__etl_defn__` marker). Graph constructor accepts a prebuilt module so transforms can build new graphs.
- **`block`** exports: `block(...)`, `BlockOp` (attrs: `name`, `input_specs`, `output_specs`, `attributes`, `effects`, `batching_policy`; methods: `impl(backend_name)`, `batching_rule(fn)`, `jvp_rule(fn)`, `vjp_rule(fn)`, `portable(fn)`; `__call__(*symbolic_or_static)` builds a BlockCall op; static args specialize). Registry queryable: `block.get_block(name)`.
- **`transforms`** exports: `vectorize`, `vmap`, `grad`, `jvp`, `vjp`. Batching rules registered per-op via a registry (`transforms.batching_rules`); derivative rules via `transforms.vjp_rules`. Result graphs contain only ordinary ops (the numpy backend needs NO special vectorize/autodiff runtime support).
- **`backends`** exports: `Backend` ABC — `name: str`, `capabilities: Capabilities` (fields: `dynamic_shapes`, `dtypes`, `collectives`, `runtime_calls`, `custom_blocks`, `async_collectives`), `lower(graph, options) -> LoweredProgram`, `compile(lowered, options) -> CompiledArtifact`, `load(artifact, device) -> backend executable`; **backend `Executable` protocol** — `run(flat_input_tensors) -> flat_output_tensors`, `.functions`, `.device`, optional `.save(path)` / `.load(...)` (must save artifact + reconstruct explicitly if handles are device-specific). **`LoweredProgram`** (`.text()`, `.save/.load`, `.backend`, `.signature` — records input/output TreeSpec+specs and static values passed down from the Graph) and **`CompiledArtifact`** (`.save/.load`, `.backend`, `.signature`, `.target`) are **owned by `backends`**. Also `register/get(name)`, `numpy_backend` (default CPU interpreter; its LoweredProgram wraps a serializable program, CompiledArtifact wraps serialized IR), `stablehlo` (v1: export utility module `stablehlo.export(graph_or_module) -> str` MLIR text — export-only, not a compilable backend; IREE/XLA/TVM are documented future integration points).
- **`dist`** exports: `group(name, ranks, backend=None)`, `rank()`, `world_size()`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `collective_permute`. Collectives build `Collective`-effect ops (kind + group attr) with local-tensor shapes (e.g. 4-rank all_gather axis=0: `[256,1024] → [1024,1024]`).
- **`pipeline`** (`etl/pipeline.py`) exports: `lower/compile/load/run/bind/build/evaluate`; `Executable` (user-facing wrapper: backend executable + input/output TreeSpec + signature for validation; `.functions`, `.device`, `.save/.load` delegating to backend), and `BoundExecutable` from `bind`.
- **`persist`** exports: `save_object(obj, path, payload_type, backend_info, signature_info)` / `load_object(path, expected_type)` using a container format: magic header + format version + JSON metadata + payload (JSON/npy-base64/opaque bytes) + SHA-256 integrity; `Cache` interface + `FileCache(directory)` with explicit `get_or_compute(key_components, compute_fn)`.

## Value-model details (binding)

- **Shapes**: `Dim(name)` symbolic; `DimExpr` supports `+ - * // % min max` over dims/ints; `TensorSpec.shape` = tuple of `Dim|DimExpr|int|None` (None = runtime-dynamic, unchecked). Rank is always known at trace time. Shape inference is `ir`/`ops`'s job via `DimExpr` arithmetic.
- **`Tensor`**: wraps numpy `ndarray` in v1 (dtype, shape, device, data); `.numpy()` returns underlying array; `__dlpack__(stream=None)`, `etl.from_dlpack(...)` (accepts objects with `__dlpack__`); equality = identity + metadata. Mutable `data` access documented as unsupported for graph constants.
- **`SymbolicTensor`**: fields `value` (ir.Value), `dtype`, `shape` (tuple of `DimExpr|int`), `location`; SSA identity = `value.id`. Must NOT define `numpy`, `data_ptr`, `__dlpack__`, `__array__`.
- **`Graph` trace semantics**: parameters become `Function` block args; static Python values specialize and are recorded in `static_values` (run validates them); closure-captured `Tensor` used in ops → `TraceError` ("make it an explicit input or use etl.constant"); `etl.constant` creates a Constant op and issues a warning above `ETL_LARGE_CONSTANT_BYTES` (default 1 MiB, env-tunable).
- **Staged sugar rules**: `etl.build`/`etl.evaluate` docstrings must state their exact expansion; no other composition APIs exist in core.

## Serialization contract (binding)

Formats versioned via constants (`ETL_FORMAT_VERSION`). Artifacts are self-describing: record backend name/version, target, input/output signature (TreeSpec + specs), static values, required custom ops, IR format version. `Graph.save` → `.etlgraph` (portable JSON container); `LoweredProgram` numpy → serialized program; stablehlo → `.mlir` text; `CompiledArtifact` → `.etlartifact`; `Executable.save` → backend-dependent (numpy: saves artifact + explicit note it is reconstructed at load). Load with mismatched backend/device/ABI → `PersistenceError`, never silent recompile.

## Routing table

| Path | Area |
|---|---|
| `./core/` | Value model: dtypes, Dim/DimExpr, TensorSpec, Tensor, SymbolicTensor, Device, TreeSpec, device helpers, errors, operator-handler hook |
| `./ir/` | EvoXIR: SSA structures, op defs, builder, shapes integration, verify, serialize, locations, effects |
| `./ops/` | Frontend tensor ops + `runtime_call` (+ op-level shape/dtype inference rules) |
| `./numpy/` | `etl.numpy` (enp) namespace incl. `linalg` — thin sugar over `ops` |
| `./trace/` | `defn`, `trace`, `Graph`, `cond`/`while_loop`/`scan`, builder context |
| `./block/` | Custom blocks: declaration, impl registry, batching/derivative rules |
| `./transforms/` | `vectorize`, `vmap`, `grad`, `jvp`, `vjp` + rule registries |
| `./backends/` | Backend interface, registry, numpy interpreter, stablehlo exporter |
| `./dist/` | Groups + explicit collectives |
| `./pipeline.py` | Orchestration: `lower/compile/load/run/bind/build/evaluate` + user-facing `Executable` wrapper |
| `./persist/` | Save/load container format, explicit cache |

Sibling: `../tests/` → test suite (read-only from here; escalate test-related writes to root).

## Test strategy

pytest; unit tests per module in `../tests/<module>/`; integration tests for the full pipeline and design-principle compliance (`../tests/test_spec_compliance.py`): staging explicitness, closure-capture errors, SymbolicTensor purity, bind-as-sugar, vmap≡vectorize sugar, collectives local-shape semantics, serialization round-trips, DLPack interop (torch via `importorskip`). CPU only.

## Notes for agents

- Keep modules thin and single-purpose; files under ~1000 lines (split along declared boundaries instead).
- Shared capability goes to the lowest common ancestor module per the contracts above — do not duplicate (e.g. numpy kernels live ONLY in `backends/numpy`).
- `etl/__init__.py` re-exports this exact surface; when adding a public name, update `__init__.py` and this contract together.
