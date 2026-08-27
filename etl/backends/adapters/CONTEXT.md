# etl/backends/adapters — optional compiler backends (pluggable adapters)

## Intent

Optional compiler adapters plugging external compilers into the shared `etl.backends.compiler` framework (`CompilerBackend` with shared `lower()`; `CompilerExecutable` base). Each adapter is ONE module (heavy dependency imports confined to function bodies) activated lazily by `etl.backends.registry.get(name)` on first use — `import etl` never imports an adapter or its compiler dependency. The shared framework is the parent `../compiler.py`; sibling constraints (import acyclicity, error strategy, no hidden staging) live in `../CONTEXT.md` — binding.

**Status:** the XLA-via-PJRT adapter (`xla.py` + `xla_util.py`) is IMPLEMENTED and validated. `iree.py` and `tvm.py` are future parallel efforts (not present; their names are reserved in `registry.OPTIONAL_ADAPTERS`).

## API Surface

| Name | Where | Notes |
|---|---|---|
| `XlaBackend` | `xla.py` | `CompilerBackend` subclass; `name="xla"`, `capabilities`, `check_available()`, `compile(lowered, options)`, `load(artifact, device)`. Inherits the shared `lower()` (verify → capability pre-check → portable inlining → StableHLO export → Signature) |
| `XlaExecutable` | `xla.py` | `CompilerExecutable` subclass (`backend_name="xla"`); `run(flat_input_tensors)`; shared `save`/`.load` (artifact round-trip, explicit reconstruction) |
| `xla_backend` | `xla.py` | module-level singleton |
| `register()` | `xla.py` | probes jaxlib + registers idempotently; raises `core.BackendError` with `pip install etl[xla]` hint |
| jaxlib plumbing | `xla_util.py` | private helpers (`_import_xla_runtime`, `_verify_xla_api_surface`, `_acquire_cpu_client`, `_make_mlir_context`, `_parse_stablehlo_module`, `_make_buffer_putter`, `_resolve_static_shape`, `_Aval`). NOT a public surface |

## XLA adapter design (validated against jax 0.10.2 / jaxlib 0.10.2 / numpy 2.4.6, CPU)

**Acquisition path (record every step when upgrading jaxlib):**

1. **PJRT client discovery.** NO standalone `pjrt_c_api_cpu_plugin.so` exists anywhere in jaxlib 0.10.2 site-packages (searched exhaustively). The CPU PJRT client is EMBEDDED in jaxlib's native `_xla.so`; acquire with `xc.make_cpu_client()` (prints 3 INFO lines from `pjrt_client.cc` to stderr — C++ logging, absl-py not even installed; cosmetic, no Python knob).
2. **StableHLO parsing.** jaxlib MLIR bindings: `_jax_mlir_ext.register_dialects(registry)` + `ctx.append_dialect_registry` + `ctx.load_all_available_dialects()` loads core dialects (func/cf/arith/…); `stablehlo`/`chlo` (.so-separate dialects) need their own `register_dialect(ctx)`. `Context(load_on_create_dialects=[...])` alone does NOT work. No jax frontend needed.
3. **Compilation.** `client.compile_and_load(mlir_module, executable_devices=xc.DeviceList(tuple(client.devices())), compile_options=xc.CompileOptions())` — the same entry point jax's own `_src/compiler.py` uses. The bytecode/`jaxlib._jax.mlir.mlir_module_to_xla_computation` route is NOT needed; plain `client.compile(module)` fails with a missing-compiler-factory error on CPU.
4. **Buffer staging.** `client.buffer_from_pyval` DOES NOT EXIST in 0.10.2. Use `xc.batched_device_put(aval, sharding, [arr], [dev], True, enable_x64=True)` with a duck-typed aval (`shape`/`dtype`/`weak_type`/`named_shape`) and a `xc.SingleDeviceSharding(dev)` instance patched with `_to_xla_hlo_sharding = lambda _: xc.HloSharding.replicate()` (pybind CALLS that method). `enable_x64=True` is MANDATORY — the default x64-off state silently truncates float64/int64 to 32-bit. Returns the `ArrayImpl` directly for a single array.
5. **Serialization.** `exe.serialize() -> bytes` and `client.deserialize_executable(bytes, device_list)` both EXIST and round-trip correctly (verified). Artifact payload is JSON-safe: `{"format": "xla-serialized-executable", "mlir_text", "executable_base64", "entry_functions", "static_input_shapes", "static_output_shapes"}`.

**Capabilities (honestly validated):**
- `dtypes`: ALL etl dtypes (float16/32/64, int8/16/32/64, uint8/16/32/64, bool, complex64/128) validated end-to-end through the full path (elementwise add; dot for float16/32/64/int32/int64; bool in+out; complex64 multiply).
- `dynamic_shapes=False`: `compile()` applies a **static-shape gate** over `signature.input_specs` AND `output_specs` — `None` entries, unknown-size `Dim`, or `DimExpr` with free runtime dims raise `core.BackendError` naming the spec ("the xla adapter requires fully static shapes; got …"). Known-size dims and closed `DimExpr` pass. Gate results are recorded into the artifact for exact run-time validation.
- `collectives=False`: 5/6 etl collectives (all_reduce, all_gather, reduce_scatter, all_to_all, collective_permute) compile AND run single-replica, but `dist.broadcast` (stablehlo `collective-broadcast`) fails AT RUN TIME on XLA:CPU (`UNIMPLEMENTED: HLO opcode collective-broadcast is not supported by XLA:CPU ThunkEmitter`). Per the capability contract (any failure → flag off), collectives are conservatively False so the shared `lower()` rejects collective graphs explicitly.
- `runtime_calls=False`, `custom_blocks=False` (block_calls are portable-inlined before export), `async_collectives=False`.

**Errors:** backend/device/ABI mismatches → `PersistenceError`; unsupported device kind → `BackendError`; compile/conversion failures → `BackendError` with the original message; run-time input dtype/shape mismatches → `DTypeError`/`ShapeError`; capability violations → `BackendError` from the shared `lower()` pre-check. No silent fallbacks, never re-traces/re-compiles.

## Routing Table

| Path | Area |
|---|---|
| `./xla.py` | `XlaBackend`, `XlaExecutable`, `xla_backend`, `register()` (module docstring = acquisition summary) |
| `./xla_util.py` | jaxlib plumbing helpers (private) + the detailed verified acquisition-path doc |
| `./iree.py` / `./tvm.py` | future parallel efforts (not present) |

## Test Strategy

Sibling `../../tests/` is read-only from here (escalate test writes to root). Adapter validation runs as throwaway probe scripts in `$TMPDIR` (never committed): (1) fresh-process `import etl` leaves `sys.modules` free of jax/jaxlib/iree/tvm; (2) `etl.backends.get("xla")` auto-activates via the registry map; (3) parity vs the numpy backend (matmul+relu+reduce_sum, reshape/broadcast/transpose, cond, while_loop, multi-output/0-d, full dtype matrix); (4) NEGATIVE: symbolic/None dims → explicit static-shape `BackendError` at `compile()`; (5) `CompiledArtifact.save/.load` + `XlaExecutable.save/.load` round-trips run identically; (6) `LoweredProgram.text()` renders MLIR; (7) full suite `python3 -m pytest` stays green (4106 passed, 7 torch skips as of the last run).

## Known Issues / Notes for Agents

- **Version pin:** `jaxlib>=0.4.23` (from pyproject extras) is TOO LOOSE — the APIs used (`compile_and_load` with an MLIR module, `batched_device_put`, `deserialize_executable`) were validated against **jaxlib 0.10.2**; `check_available()` probes the exact API surface and raises with a version hint on drift. Bump `VALIDATED_JAXLIB_VERSION` only after re-running the full probe script.
- `client.buffer_from_pyval` does not exist in 0.10.2 — do not "restore" it; the `batched_device_put` staging is the verified path.
- The CPU client prints 3 stderr INFO lines at creation (cosmetic, no absl knob).
- `collectives=False` is conservative: 5/6 collectives actually run single-replica; only `collective-broadcast` is unimplemented in XLA:CPU's ThunkEmitter (run-time failure). Re-probe after jaxlib upgrades before flipping the flag.
- Parent `../CONTEXT.md` (etl/backends) still describes the adapters as "not present yet" — update it when the parent node next touches it.
