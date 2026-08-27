# etl/backends/adapters — optional compiler adapters (IREE, XLA via PJRT, TVM)

## Intent

Pluggable compiler backends for etl. Each adapter consumes the SHARED StableHLO lowering produced by `CompilerBackend.lower` (`../compiler.py` — verify, capability pre-check, block-portable inlining, StableHLO export, Signature recording) and implements only the compiler-specific half: optional dependency probe (`check_available` override — the shared base is a concrete no-op), `compile` (invoke the external compiler on the MLIR text), `load` (rebuild the executable from the artifact — never re-compiling). Activation is register-on-first-use: `etl.backends.registry.get(name)` imports the adapter module via `registry.OPTIONAL_ADAPTERS` and calls its `register()`; `import etl` never imports an adapter or its compiler dependency.

Status: **all three adapters implemented and validated** against the installed compiler versions (iree 20241104.1068, jax/jaxlib 0.10.2, apache-tvm 0.26.0 + jaxlib).

## API Surface

Each adapter module (`iree.py`, `xla.py`, `tvm.py`) exposes the same shape:
- `<Name>Backend` (a `CompilerBackend` subclass; names `"iree"`, `"xla"`, `"tvm"`), `<Name>Executable` (a `CompilerExecutable` subclass), singleton `<name>_backend`, module-level `register()` (probes the dependency — `core.BackendError` with a pip hint when missing — then registers idempotently).
- `<name>_util.py` helpers: `xla_util.py` (jaxlib plumbing: StableHLO text parse, compile/execute, buffer staging), `tvm_util.py` (vendored-translator compatibility shim, op whitelist gate, translate/build/run/persist helpers).
- `__init__.py`: documentation only, imports nothing heavy.

## Adapter designs (validated end-to-end, numpy-backend parity)

### IREE (`iree.py`)

- compile: `iree.compiler.compile_str(mlir_text, target_backends=["llvm-cpu"], input_type="stablehlo", extra_args=["--iree-input-demote-f64-to-f32=false", "--iree-llvmcpu-target-cpu=generic"])` → VM flatbuffer, base64'd into the payload (`format="iree-vmfb"`, keeps `mlir_text` + `entry_functions`). Compiler diagnostics re-raised as `core.BackendError` (never silent).
- load/run: device via the RELIABLE path `iree.runtime.get_driver("local-task")` + `driver.create_default_device()` (the `rt.system_setup(config=...)` form intermittently fails with `TypeError: 'module' object is not callable` — submodule shadowing in iree.runtime's package init; do not reintroduce it); `iree.runtime.load_vm_flatbuffer(flat, driver="local-task")`; inputs `iree.runtime.asdevicearray(device, tensor.numpy())`; outputs `np.asarray(result)` → `core.Tensor`.
- Capabilities: `dynamic_shapes=True` (symbolic dims run at multiple concrete sizes); `dtypes` = f16/f32/f64/i8/i16/i32/i64/bool (unsigned ints excluded — iree-compile 20241104 legalizes unsigned `reduce` nondeterministically/not at all; complex excluded — the exporter v1 defers complex computation beyond cast); `collectives=False`, `runtime_calls=False`, `custom_blocks=False`, `async_collectives=False`.

### XLA via PJRT (`xla.py`)

- PJRT acquisition: NO standalone `pjrt_c_api_cpu_plugin.so` exists in jaxlib 0.10.2 — the CPU PJRT client is EMBEDDED in jaxlib's `_xla.so` and acquired via `jaxlib.xla_client.make_cpu_client()`.
- compile: parse StableHLO text with jaxlib's MLIR bindings (dialect registration: `_jax_mlir_ext.register_dialects` + `ctx.append_dialect_registry` + `ctx.load_all_available_dialects()` + explicit `stablehlo`/`chlo` dialect registration), then `client.compile_and_load(module, executable_devices=xc.DeviceList(tuple(client.devices())), compile_options=xc.CompileOptions())` — the same entry point jax's own compiler uses (plain `client.compile` fails with a missing-compiler-factory error in 0.10.2). `client.buffer_from_pyval` DOES NOT exist in 0.10.2 — inputs stage via `xc.batched_device_put(aval, sharding, [arr], [dev], True, enable_x64=True)` (x64 must be ENABLED — the default x64-off state silently truncates float64/int64 to 32-bit).
- Static-shape gate: `compile()` rejects any non-statically-resolvable shape entry in input/output specs (`None` dims, unknown-size `Dim`, free `DimExpr`) with an explicit `core.BackendError` — XLA dynamic shapes are limited; declared honestly (`dynamic_shapes=False`). The gate's decision is recorded in the artifact and enforced exactly at `run()`.
- persist: `exe.serialize() -> bytes` / `client.deserialize_executable(bytes, device_list)` round-trip correctly (true serialize, no load-time recompile).
- Capabilities: `dtypes` = all 14 etl dtypes (incl. complex64/128, validated through the full path); `collectives=False` (5/6 collectives work single-replica but `collective-broadcast` fails at RUN time on XLA:CPU — any failure ⇒ flag off, shared `lower()` rejects all collective graphs explicitly); `dynamic_shapes=False`; `runtime_calls=False`, `custom_blocks=False`, `async_collectives=False`.

### TVM (`tvm.py`)

- translate: `tvm.relax.frontend.stablehlo.from_stablehlo(mlir_text)` (exists in 0.26; `tvm.relax.frontend.from_mlir` does NOT; `tvm.relax.vm.build` does NOT — build lives in `tvm.relax.vm_build`, re-exported as `tvm.relax.build`). The vendored translator parses MLIR with jax's mlir bindings — hence tvm needs jax/jaxlib present too.
- build/run: `tvm.relax.vm_build.build(mod, target=tvm.target.Target("llvm"))` → `VMExecutable`; run via `tvm.runtime.vm.VirtualMachine(ex, tvm.runtime.cpu())["main"](...)`; inputs `tvm.runtime.tensor(np_array, tvm.runtime.cpu())` (the `tvm.nd` namespace no longer exists); outputs `tvm.runtime.Tensor.numpy()` → `core.Tensor`.
- persist: `VMExecutable.export_library(path)` → host .so, base64'd into the payload (`format="tvm-vm-library"`); load decodes to a temp file and `tvm.runtime.load_module(path)` — no recompile at load. `Module.save_to_file` does not exist in 0.26.
- Compatibility shim (REQUIRED): the TVM 0.26.0 vendored StableHLO translator targets the OLD jax mlir python bindings; `tvm_util.ensure_compat()` patches five gaps (type-check classmethods, OpView/Operation key normalization, `broadcast_to` ShapeExpr, `DenseI64ArrayAttr` decoding, float16 np.asarray constant decoding) — each patch restores exactly the semantics the vendored code assumed, no computation changes.
- Op coverage gate: `SUPPORTED_STABLEHLO_OPS` whitelist checked at compile (`core.BackendError` naming the op otherwise): arithmetic, unary math, compare, select, bitwise/logical, convert, broadcast_in_dim, reshape, transpose, concatenate, slice (unit strides), pad (zero interior), reduce (add/max/min/mul), dot_general (matmul), constant. Rejected: control flow (`stablehlo.if`/`while`), gather/scatter, remainder, conv/reduce_window (vendored handlers hardcode NHWC/HWIO — silently wrong for etl's NCHW conv), multi-function modules, multi-output functions.
- Capabilities: `dynamic_shapes=True` (symbolic dims validated at multiple sizes, constant-free graphs); `dtypes` = f16/f32/f64/i8/i16/i32/i64/u8/u16/u32/u64/bool (complex excluded — vendored translator raises); `collectives=False`, `runtime_calls=False`, `custom_blocks=False`, `async_collectives=False`.

## Constraints (binding)

- Heavy-import rule: `iree`/`jax`/`jaxlib`/`tvm` imports live ONLY inside function bodies; top-level imports limited to stdlib, `etl.core`, and the `..compiler`/`..program`/`..registry` siblings. Verified: `import etl` leaves `sys.modules` free of iree/jax/tvm.
- Errors: capability violations and compiler/translator failures raise `core.BackendError` naming the op/feature — never silent fallback to numpy or partial semantics; load-time mismatches raise `core.PersistenceError`; dtype mismatches `core.DTypeError`, shape mismatches `core.ShapeError` (mirrors the numpy interpreter).
- Staging never composes: `load` never re-traces/lowers/compiles; device handles are never serialized.
- CPU only; non-CPU devices raise `core.BackendError`, non-`Device` objects `core.DeviceError`.

## Known Issues

1. **Writer/export gaps surfaced by adapters (sibling-node `stablehlo/` — out of this node's scope):** (a) symbolic-dim graphs mixing scalar-constant broadcasts (e.g. `relu`) export `broadcast_in_dim` to a dynamic-shape result, which the StableHLO verifier forbids — honest compile-time `BackendError` (the spec'd `dynamic_broadcast_in_dim` path exists in iree-compile but the writer does not emit it); (b) mixed-dtype binary ops with Python scalar constants can produce a dtype mismatch parse error (workaround: explicit `etl.cast`); (c) `stablehlo.add` on i1 is XOR vs numpy bool+ = OR (export semantic gap).
2. **IREE:** `collective_broadcast` cannot be legalized by iree-compile 20241104 llvm-cpu (upstream); collectives are off (see capabilities). u32/u64 excluded due to nondeterministic legalization (upstream race). Cosmetic compiler warnings about generic CPU target.
3. **XLA:** jaxlib's CPU client prints 3 cosmetic stderr INFO lines at creation (C++ logging; no Python knob). `collectives=False` is conservative (only collective-broadcast is unimplemented at run time — re-probe before flipping). x64 must stay enabled (silent truncation hazard).
4. **TVM:** control flow, conv, gather/scatter, remainder, multi-function/multi-output modules rejected by the compile-time gate. The vendored translator requires jax/jaxlib at RUNTIME of the adapter (it imports `jax._src.interpreters.mlir`).
5. **Version floors (feed the pyproject extras):** iree validated on the 20241104.1068 release (the extra pins `iree-base-compiler>=2.9.0`/`iree-base-runtime>=2.9.0` — the current IREE distribution names, first released at 2.9.0; the old `iree-compiler`/`iree-runtime` names are deprecated and stale on PyPI, last released 20241104.1068); xla validated on jaxlib 0.10.2 — existing extra `jaxlib>=0.4.23` is TOO LOOSE (the APIs used — `make_cpu_client`, `batched_device_put`, `compile_and_load`, `deserialize_executable` — don't exist in 0.4.x; `check_available()` probes the exact surface and raises on drift) — recommend `jax>=0.10,<0.11`; tvm validated on `apache-tvm 0.26.0` — existing extra `apache-tvm>=0.14` is TOO LOOSE (`from_stablehlo` exists only in 0.26) and the extra must ALSO include jax/jaxlib (the translator imports jax's mlir bindings) — recommend `apache-tvm>=0.26` + `jax>=0.10,<0.11`. pyproject edits live at the repo root — escalate to the root agent.

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | package marker + adapter-module contract (documentation only) |
| `./iree.py` | `IreeBackend` (compile/load/check_available), `IreeExecutable` (run), `iree_backend`, `register()` |
| `./xla.py` | `XlaBackend`, `XlaExecutable` (static-shape gate), `xla_backend`, `register()` |
| `./xla_util.py` | jaxlib plumbing: dialect registration, StableHLO parse, compile/execute, buffer staging, serialization |
| `./tvm.py` | `TvmBackend`, `TvmExecutable`, `tvm_backend`, `register()` |
| `./tvm_util.py` | vendored-translator compat shim (incl. `get_dimension_size`/`dynamic_broadcast_in_dim` handlers), op whitelist gate, translate/build/run/persist helpers |

Sibling: `../compiler.py` → shared `CompilerBackend`/`CompilerExecutable` framework; `../registry.py` → `OPTIONAL_ADAPTERS` auto-activation.

## Test strategy

Validated with throwaway scripts in `$TMPDIR` (not committed): registry auto-activation, numpy-backend parity (matmul+relu+reduce_sum; reshape/broadcast/transpose; symbolic-dim graphs at multiple sizes where supported), artifact/executable save-load round-trips, error paths (unsupported ops, dtype/shape/device mismatches, cross-backend artifacts, collectives at lower, static-shape gate for xla). Committed tests for these adapters would live in `../../../tests/backends/` — a SIBLING directory; test-related writes escalate to the repo root. The full suite (`python3 -m pytest -q`) stays green.
 collectives at lower, static-shape gate for xla). Committed tests for these adapters would live in `../../../tests/backends/` — a SIBLING directory; test-related writes escalate to the repo root. The full suite (`python3 -m pytest -q`) stays green.
