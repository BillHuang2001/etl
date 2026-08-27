# etl/backends — backend abstraction, numpy reference backend, StableHLO exporter

## Intent

Turn verified frontend Graphs into executable programs. Four responsibilities:

1. **Backend contract** — `Capabilities`, `Backend` ABC, backend `Executable` protocol, registry. This is the plug-in seam for compilers.
2. **Shared pluggable-compiler framework** (`compiler.py`, `inline.py`, `adapters/`) — `CompilerBackend` / `CompilerExecutable` + the shared block-call inlining machinery, so compiler adapters reuse the exact same lowering logic as the numpy backend. **Three adapters are IMPLEMENTED in `adapters/`: `"iree"`, `"xla"` (via the PJRT C API driven with ctypes — a user-provided plugin `.so`, no jax/jaxlib), `"tvm"`** — each does lower (shared) → compile (native compiler on the StableHLO text) → load → run, satisfying the `Backend`/`Executable` contracts (see `adapters/CONTEXT.md` for validated designs, capabilities, and known issues).
3. **Reference numpy CPU backend** (`numpy_backend`, default) — a pure-Python numpy interpreter proving the IR semantics; the only concrete-computation path besides `core`'s concrete creators.
4. **StableHLO export utility** (`stablehlo`) — emits StableHLO MLIR text so external compilers (`iree-compile model.mlir -o model.vmfb`) can take over compilation. Export-only in v1, NOT a compilable backend.

**Ownership (binding):** `LoweredProgram` and `CompiledArtifact` are owned by THIS package, not by `pipeline`. `pipeline` orchestrates staging and re-exports them. See `../../CONTEXT.md` (staging pipeline, error strategy) and `../CONTEXT.md` (cross-module contracts) — both binding.

## API Surface

Must-expose names (re-exported from `etl/backends/__init__.py` and from `etl`):

| Name | Where | Notes |
|---|---|---|
| `Capabilities` | `backend.py` | frozen dataclass: `dynamic_shapes: bool`, `dtypes: frozenset`, `collectives: bool`, `runtime_calls: bool`, `custom_blocks: bool`, `async_collectives: bool` |
| `Backend` | `backend.py` | ABC: `name: str` (class attr), `capabilities: Capabilities`; `lower(graph, options=None) -> LoweredProgram`; `compile(lowered, options=None) -> CompiledArtifact`; `load(artifact, device=None) -> executable` |
| `Executable` | `backend.py` | `Protocol` (runtime-checkable): `run(flat_input_tensors: list[core.Tensor]) -> list[core.Tensor]`; attrs `.functions` (tuple of function names), `.device`; optional `.save(path)` / classmethod `.load(path, device=None)` |
| `Signature` | `program.py` | frozen dataclass: `input_tree`/`output_tree` (core.TreeSpec), `input_specs`/`output_specs` (per-leaf core.TensorSpec), `static_values` (input static leaves, pre-order), `output_static_values` (static output leaves, pre-order; defaulted `()` — re-inserted by `etl.run`) — passed down from the Graph at `lower()` time |
| `LoweredProgram` | `program.py` | attrs `.backend` (name str), `.signature`, `.payload` (backend-specific, serializable); methods `.text() -> str`, `.save(path)`, classmethod `.load(path)` |
| `CompiledArtifact` | `program.py` | attrs `.backend`, `.signature`, `.target`, `.payload`; self-describing: records required custom ops + runtime dependencies; methods `.save(path)`, classmethod `.load(path)` |
| `register(backend)`, `get(name) -> Backend` | `registry.py` | `get` raises `core.BackendError` for unknown names; duplicate name with a different instance raises `BackendError`. `OPTIONAL_ADAPTERS` (module-level dict: `iree`/`xla`/`tvm` → adapter module path) makes `get` AUTO-ACTIVATE optional adapters on a registry miss (import module → call its `register()` → retry lookup; `register()` errors — the dependency probe with its pip-install hint — propagate unchanged; no adapter is imported at `etl`/`etl.backends` import time) |
| `CompilerBackend` | `compiler.py` | ABC (subclass of `Backend`) — the shared pluggable-compiler base: class attrs `name`/`capabilities`; CONCRETE classmethod `check_available()` (default no-op returning `None` — subclasses optionally override to probe external compiler deps and raise `core.BackendError` with a pip-install hint); `@abstractmethod compile`/`load` (compile never loads; load never re-lowers/re-compiles); SHARED `lower()`: verify → capability pre-check (`runtime_call`/`collective` vs flags via `inline.iter_ops`) → `inline_portables(module, keep_backend_impls=None)` (every block_call needs a portable) → verify → `stablehlo.export` gate → `Signature` recording identical to numpy → JSON-safe payload `{"format": "stablehlo", "format_version": 1, "mlir_text": str, "entry_functions": tuple}`. Class docstring carries the "How to add a new adapter" recipe |
| `CompilerExecutable` | `compiler.py` | ABC base for adapter executables (satisfies the `Executable` protocol): class attr `backend_name`; ctor `(artifact=None, signature=None, device=None, native_module=None, entry_functions=())` (signature falls back to `artifact.signature`); `.functions` property (= `entry_functions`), `.device`; shared `save(path)` (delegates to `artifact.save`; `BackendError` without an artifact) and classmethod `load(path, device=None)` (`CompiledArtifact.load` → backend-name validation (`PersistenceError` naming both) → registry-routed `backend.load` — the registry auto-activates the adapter lazily); `run(flat_input_tensors)` stays abstract |
| `iter_block_ops` / `iter_ops` / `clone_ops_into` / `drop_op_uses` / `inline_portables` | `inline.py` | SHARED block-inlining machinery (extracted from `numpy/`; `numpy/inline.py` is now a thin re-export): regions-first bottom-up op walks, portable-splicing bookkeeping (fresh ids, Use records, output-type guards), and the fixpoint driver `inline_portables(module, keep_backend_impls=None) -> int` (portables may emit block calls; 1000-expansion cap; `keep_backend_impls="numpy"` keeps blocks with a registered numpy impl; `None` = compiler adapters: no portable ⇒ `BackendError` "compiler backends require BlockOp.portable(...)"). `etl.backends.numpy.shapes` is imported lazily inside `_dim_compatible` |
| `adapters` | `adapters/__init__.py` | subpackage marker, docstring-only (no heavy imports): documents the three IMPLEMENTED adapter modules `iree.py`/`xla.py`/`tvm.py` (singletons `iree_backend`/`xla_backend`/`tvm_backend`), each module's `register()` contract, the heavy-import rule, and the register-on-first-use flow via `registry.get` |
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
- `lower(graph, options=None)`: (1) `graph.verify()`; (2) capability pre-check (v1 numpy supports everything — the check pattern stays); (3) inline `block_call` portable decompositions as graph→graph expansion via the SHARED fixpoint `inline.py::inline_portables(module, keep_backend_impls="numpy")` (same machinery compiler backends use; block has neither portable decomposition nor registered numpy impl ⇒ `BackendError` naming the block); (4) record `Signature` from the Graph's LIVE attributes (input/output TreeSpec + per-leaf specs + static values — passed down, not re-derived); (5) `payload` = versioned self-describing `ir.Module` serialization (`ir.serialize_module`).
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

**Kernel split** (`numpy/kernels/`): `elementwise.py` (arith/activations/comparisons/select/cast/broadcast), `reductions.py` (reduce_*, argmax/argmin), `indexing.py` (reshape/transpose/slice/concat/pad/gather/scatter/tril/triu/cumsum), `linalg.py` (dot/conv/solve), `control_flow.py` (if/while/call region execution), `collective.py` (collective op dispatch via the hook + rank/world_size), `custom.py` (`runtime_call`, `block_call`, `constant`). Dispatch table assembled in `kernels/__init__.py` (op name → kernel); coverage = every registered IR op name except `return` (special-cased by the loop). The interpreter loop itself lives in `numpy/interpreter.py` (block-op-order execution, value env, output validation, `KernelContext`); the per-execution rank context in `numpy/exec_context.py`; block_call portable-splicing in the SHARED `../inline.py` (`numpy/inline.py` is a thin re-export).

## StableHLO exporter — v1 scope

Export utility ONLY: `stablehlo.export(graph_or_module) -> str` produces StableHLO MLIR text. Output is compiler input for external tools (`iree-compile model.mlir -o model.vmfb`); the IREE/XLA/TVM adapters in `./adapters/` are IMPLEMENTED and consume this export as their lowering payload.

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
| dot / conv | `stablehlo.dot_general` / `stablehlo.convolution` |
| constant | `stablehlo.constant` |
| cond / while_loop | `stablehlo.if` / `stablehlo.while` |
| collectives (all_reduce/all_gather/reduce_scatter/all_to_all/broadcast_collective/collective_permute) | `stablehlo.all_reduce/all_gather/reduce_scatter/all_to_all/collective_broadcast/collective_permute` |
| symbolic dims | `?` dynamic dims in tensor types, e.g. `tensor<?xNxf32>` |

**Deferred in v1** (⇒ `core.BackendError` naming the op, message suggests decomposition or a future adapter): `gather`, `scatter`, `scan`, `runtime_call`, `block_call` (blocks with no portable decomposition), `dist.rank()`/`dist.world_size()` graph scalars, `erf`, `gelu`, `argmax`, `argmin`, `call`, `tril`/`triu`/`cumsum`/`solve`, complex-number elementwise beyond cast. Mnemonics were verified against the official StableHLO spec at implementation time: `stablehlo.erf` and `stablehlo.argmax/argmin` do NOT exist in the StableHLO opset (erf is a CHLO op; ArgMax/ArgMin are open feature requests), so those ops are deferred rather than emitted with invented mnemonics. The unlisted data-movement ops (tril/triu/cumsum/solve/call) have no v1 mapping and fail explicitly.

Type map (dtype → MLIR): float16→f16, float32→f32, float64→f64, int8→i8, int16→i16, int32→i32, int64→i64, uint8→ui8, uint16→ui16, uint32→ui32, uint64→ui64, bool→i1, complex64→`complex<f32>`, complex128→`complex<f64>`.

## Serialization compliance

`LoweredProgram.save`/`CompiledArtifact.save` delegate to the `etl.persist` container format (lazy import): magic header + format version + JSON metadata + payload + SHA-256 integrity. Metadata is self-describing: backend name/version, target, signature (TreeSpecs + specs + static values), required custom ops, IR format version. `load()` validates the recorded backend against the registry (numpy reconstruction handled by `NumpyBackend`) — mismatch or missing custom op ⇒ `PersistenceError`. Never silently re-traces/re-compiles.

## Compiler adapter framework (design)

`compiler.py` is the pluggability seam: a new StableHLO-consuming compiler backend is a `CompilerBackend` subclass declaring `name`/`capabilities`, implementing `check_available()` (dependency probe with a `pip install etl[...]` hint), `compile()` (invoke the native compiler on the payload's `mlir_text`; JSON-safe artifact payload), and `load()` (rebuild the executable — never recompiling), plus a `CompilerExecutable` subclass implementing `run()`. The shared `lower()` does everything else (verify → capability pre-check → portable block inlining → StableHLO export → Signature → MLIR payload). The "How to add a new adapter" recipe lives in `compiler.py`'s docstring.

**Implemented adapters** (all validated end-to-end with numpy-backend parity on real graphs; details + per-adapter known issues in `adapters/CONTEXT.md`):
- **`"iree"`** — `iree.compiler.compile_str` → VM flatbuffer → `iree.runtime` (local-task driver). `dynamic_shapes=True`; dtypes f16/f32/f64/i8/i16/i32/i64/bool; collectives off.
- **`"xla"`** (XLA via the PJRT C API, driven with ctypes) — loads a user-provided PJRT plugin `.so` exporting `GetPjRtApi` (discovery: `options["plugin_path"]` → `ETL_PJRT_PLUGIN` → well-known paths; distribution is deliberately out of scope — e.g. `bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin` from OpenXLA); StableHLO text compiled directly via `PJRT_Program{code, format="mlir"}` — no MLIR python bindings anywhere; `PJRT_Api` version/`struct_size` ABI gate; persistence via `PJRT_Executable_Serialize`/`DeserializeAndLoad`. `dynamic_shapes=False` (static-shape gate at compile — explicit BackendError for symbolic dims); all 14 etl dtypes; collectives off.
- **`"tvm"`** — `tvm.relax.frontend.stablehlo.from_stablehlo` → `tvm.relax.vm_build.build` (llvm) → `tvm.runtime.vm.VirtualMachine`; true serialize via `export_library` (no load-time rebuild). The `jax` package is NEVER used: the vendored translator's `jax._src.interpreters.mlir` import is satisfied by a `sys.modules` shim; jaxlib is required ONLY for its bundled LLVM MLIR python bindings, accessed behind the `_mlir_bindings.py` seam. `dynamic_shapes=True` (constant-free graphs); 12 dtypes (no complex); collectives off; compile-time op whitelist gate (no control flow/conv/gather/scatter).

All three: `runtime_calls=False` (runtime_call rejected at lower), `custom_blocks=False` (portable-only inlining at lower), `async_collectives=False`; never silently fall back to the numpy backend.

## Known Issues

- **StableHLO writer limits surfaced by adapters** (in this package's `stablehlo/`): (a) mixed-dtype binary ops with Python scalar constants can produce dtype-mismatch parse errors (workaround: explicit `etl.cast`); (b) `stablehlo.add` on i1 is XOR (XLA semantics) vs numpy bool+ = OR. Dynamic-shape broadcasts are RESOLVED — the writer emits `stablehlo.dynamic_broadcast_in_dim` (output_dimensions built via `get_dimension_size` chains) whenever a broadcast's result has dynamic dims, validated end-to-end through iree and tvm at multiple concrete sizes; broadcasts with no full-shape source raise `BackendError` naming "dynamic broadcast" (see `stablehlo/CONTEXT.md`).
- **IREE**: `collective_broadcast` cannot be legalized by iree-compile 20241104 llvm-cpu (upstream) — hence `collectives=False`; u32/u64 excluded (nondeterministic upstream legalization of unsigned reduce). `iree.runtime.system_setup(config=...)` intermittently fails (`TypeError: 'module' object is not callable`, submodule shadowing) — use `get_driver("local-task")` + `create_default_device()`.
- **XLA**: a real CPU PJRT plugin `.so` is REQUIRED and user-provided (no pip package exists — discovery via `plugin_path` option / `ETL_PJRT_PLUGIN` / well-known paths). The adapter is ABI-gated against the vendored `pjrt_c_api.h` translation (recorded header commit; `PJRT_Api.version`/`struct_size` drift ⇒ explicit `BackendError`). ctypes plumbing is validated via a compiled test-plugin (`tests/backends/test_pjrt_ctypes_plugin.py`, gcc); real-XLA numerical parity is pending a real plugin `.so`. collectives off because `collective-broadcast` fails at XLA:CPU run time (5/6 work single-replica — re-probe before flipping).
- **TVM**: requires jaxlib at adapter runtime — ONLY for its bundled LLVM MLIR python bindings (`jaxlib.mlir`, the same bindings any MLIR tooling uses; the `jax` package is never imported — a `sys.modules` shim satisfies the vendored translator's `jax._src.interpreters.mlir` import). A compatibility shim (`tvm_util.ensure_compat()`) patches the 0.26.0 vendored translator against the new mlir python bindings; control flow / conv / gather / scatter / remainder / multi-function / multi-output modules rejected by the compile-time gate.
- **pyproject extras** (repo root — escalated): `iree` extra OK (`>=20240410`); `xla` extra REMOVED — the adapter has NO pip dependency (user-provided PJRT plugin `.so`); `tvm` extra `apache-tvm>=0.26` + `jaxlib>=0.10,<0.11` (`from_stablehlo` exists only in 0.26; jaxlib only for the bundled MLIR bindings).

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
| `./compiler.py` | shared pluggable-compiler framework: `CompilerBackend` (shared `lower` with dtype/shape/op capability pre-check; concrete default `check_available`; abstract `compile`/`load`), `CompilerExecutable` (shared save/load; abstract `run`) |
| `./inline.py` | SHARED block-inlining machinery: `iter_block_ops`/`iter_ops` (regions-first bottom-up walk), `clone_ops_into`/`drop_op_uses` (portable splicing + use bookkeeping), `inline_portables` (fixpoint driver) |
| `./program.py` | `Signature`, `LoweredProgram`, `CompiledArtifact` (owned by backends; `text()` renders str / stablehlo-dict (`mlir_text`) / serialized-module payloads) |
| `./registry.py` | `register`/`get` + `OPTIONAL_ADAPTERS` (first-use auto-activation of optional adapters) |
| `./adapters/` | optional compiler adapter modules — ALL IMPLEMENTED: `iree.py` (iree-base-compiler/runtime, VM flatbuffer), `xla.py` + `xla_util.py` + `_pjrt_c_api.py` (XLA via PJRT C API ctypes — user-provided plugin `.so`, StableHLO text → `PJRT_Client_Compile`), `tvm.py` + `tvm_util.py` + `_mlir_bindings.py` (Relax `from_stablehlo` + VM build/run + vendored-translator compat shim + MLIR-binding seam); `__init__.py` is docstring-only, no heavy imports |
| `./numpy/` | Reference numpy CPU interpreter: `NumpyBackend`, `NumpyExecutable`, `interpreter.py` (execution loop + KernelContext), `exec_context.py` (per-run RankContext), `inline.py` (thin re-export of `../inline.py`), `kernels/` (per-category kernels), `shapes.py` (DimExpr evaluation), `collectives.py` (default executor; canonical hook lives in `etl.dist.context`) |
| `./stablehlo/` | StableHLO MLIR export utility: `export`, `ops.py` (mapping table data), `writer.py` (MLIR text emission) |

## Notes for agents

- **Implementation status: complete.** All behavioral bodies in `backend.py`, `program.py`, `compiler.py`, `inline.py`, `registry.py`, `numpy/`, `stablehlo/`, and `adapters/` (iree/xla/tvm) are implemented; no `NotImplementedError` stubs remain in this node (the abstract `compile`/`load`/`run` methods of `CompilerBackend`/`CompilerExecutable` are adapter contracts, by design; `check_available` is a concrete default no-op). `etl.pipeline` (orchestration above this package) and sibling test suites are handled elsewhere.
- Optional adapters are NEVER imported at `etl`/`etl.backends` import time — `registry.get("iree"|"xla"|"tvm")` imports + registers them on first use (verified with a fresh interpreter: `import etl` leaves `sys.modules` free of iree/jax/jaxlib/tvm). `program.py._require_registered_backend` routes through `get()`, so persisted adapter artifacts auto-activate on load.
- The IR op names for control flow are `if`/`while` (trace lowers `cond`/`while_loop`/`scan` into them) and the dist broadcast collective is `broadcast_collective` — frontend names differ from IR names; kernels and the stablehlo writer dispatch on IR names.
- `runtime_call` carries its callback as a STRING registry id (resolved via `etl.ops.constant._get_callback` at run time) — artifacts with `runtime_call` require the same callback registrations at load time; callbacks are never serialized.
- Block-impl convention (finalized with `etl.block`): the interpreter invokes registered numpy impls as `impl(*numpy_arrays, **static_args) -> ndarray | tuple[ndarray]`; portable decompositions are spliced into the graph at `lower()` time via `etl.block.registry`.
- Collective protocol limitation: `dist.context.CollectiveExecutor` has a single `axis` param for `all_to_all` (kernel forwards `split_axis`) and no `reduce_op` for `reduce_scatter` — documented in `numpy/kernels/collective.py`.
- `../../tests/` is a sibling: read-only, escalate writes to root.
