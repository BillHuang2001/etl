# etl/backends/adapters — optional compiler adapters (IREE implemented; XLA/TVM future)

## Intent

The pluggable-compiler seam: optional backends that consume the SHARED StableHLO lowering
(`../compiler.py` — `CompilerBackend.lower` does verify → capability pre-check → portable
inlining → StableHLO export → `Signature` recording → MLIR-text payload, for every adapter)
and add only the compiler-specific half (`check_available` / `compile` / `load` /
`CompilerExecutable` subclass). Adapters are NEVER imported at `etl`/`etl.backends` import
time: `registry.get("iree"|"xla"|"tvm")` imports + registers them on first use
(`registry.OPTIONAL_ADAPTERS`) — `import etl` stays light (validated: no iree/jax/tvm in
`sys.modules`).

## API Surface

| Name | Where | Notes |
|---|---|---|
| `iree_backend` | `iree.py` | `IreeBackend` singleton, registered via module-level `register()` |
| `IreeBackend` | `iree.py` | `CompilerBackend` subclass: `name = "iree"`; `capabilities` (see below); `check_available()` probes iree.compiler+iree.runtime (raises `core.BackendError` with hint `pip install etl[iree]` when missing); `compile()` → iree-compile on the MLIR payload; `load()` → `IreeExecutable` |
| `IreeExecutable` | `iree.py` | `CompilerExecutable` subclass (`backend_name = "iree"`): runs the VM flatbuffer on the `local-task` CPU driver; `save`/`load` inherited (artifact round-trip, explicit reconstruction) |
| `register()` | `iree.py` | module-level: availability probe → `registry.register(iree_backend)` (idempotent, same instance) |
| `xla.py`, `tvm.py` | — | FUTURE adapter modules (contract in `__init__.py`); not implemented — `registry.get` raises `BackendError` for them |

**IreeBackend capabilities** (all validated end-to-end against iree-compiler==20241104.1068 /
iree-runtime==20241104.1068, llvm-cpu, CPU only; see `iree.py` docstring for the full data):

- `dtypes`: float16/float32/float64/int8/int16/int32/int64/bool_. NOT declared: uint8/uint16
  (iree-compile cannot legalize unsigned `reduce` — deterministic failure), uint32/uint64
  (iree-compile 20241104 legalizes the same module NON-DETERMINISTICALLY — upstream race,
  so not a reliable capability), complex64/128 (stablehlo exporter v1 defers complex
  computation beyond cast).
- `collectives=False` — see Known Issues (all six `etl.dist` collectives unreachable on the
  local drivers; shared `lower()` rejects them explicitly, never silently).
- `dynamic_shapes=True` — symbolic-dim graphs compile and run with different concrete sizes
  (validated); see Known Issues for the scalar-broadcast exporter limitation.
- `runtime_calls=False`, `custom_blocks=False` (portables inlined by the shared lower),
  `async_collectives=False`.

## Constraints (binding)

- **Heavy-import rule**: `iree`/`iree.compiler`/`iree.runtime`/`numpy` imports live ONLY
  inside function bodies; top level is stdlib + `etl.core` + sibling modules
  (`..compiler`, `..registry`, `..program`, `..backend`). Never import `etl.pipeline`.
- **Honest staging**: `compile` never loads; `load` never re-lowers/re-compiles;
  `iree.compiler.CompilerToolError` → `core.BackendError` carrying the diagnostics.
- **Reliable device acquisition** (validated 12 in-process cycles + 5 fresh processes):
  `rt.get_driver("local-task")` + `driver.create_default_device()` — do NOT use the
  historical `rt.system_setup(config=...)` recipe (`iree.runtime.system_setup` is a MODULE
  in iree-runtime 20241104 → `TypeError: 'module' object is not callable`).

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | docstring-only package marker + adapter-module contract (no heavy imports) |
| `./iree.py` | IREE adapter: `IreeBackend`, `IreeExecutable`, `iree_backend`, `register()` |
| `./xla.py` | FUTURE XLA-via-PJRT adapter (not implemented) |
| `./tvm.py` | FUTURE TVM adapter (not implemented) |

## Test strategy

Validated via throwaway scripts (NOT committed) in `$TMPDIR`: light-import check; registry
auto-activation; end-to-end parity vs the numpy backend (matmul+relu+reduce_sum;
reshape/broadcast/transpose; symbolic dims at two concrete sizes; `cond`/`while_loop`;
structured multi-output; full dtype matrix); explicit rejection of all six collectives and
`runtime_call` at `lower()`; run-time validation errors (`BackendError` for count/dtype,
`ShapeError` for rank/static-shape, symbolic dims pass through); save/load round-trips
(`CompiledArtifact.save/.load`, `IreeExecutable.save/.load`); `LoweredProgram.text()`
rendering MLIR; compile determinism (10/10 on declared-capability graphs); device
acquisition reliability. Full repo suite: 4106 passed, 7 torch skips, 0 failures.

## Known issues

- **`collective_broadcast` cannot be legalized** by iree-compile 20241104 llvm-cpu
  ("failed to legalize operation 'stablehlo.collective_broadcast' that was explicitly
  marked illegal" — upstream). The other five collectives compile but cannot RUN:
  local-task/local-sync HAL raises "UNIMPLEMENTED; collectives not implemented" at
  `hal.channel.create` (the Python wheels ship no communicating channel provider).
  Consequence: `collectives=False`; the shared `lower()` rejects every collective-effect
  op with `BackendError` naming it. Revisit when an MPI-backed channel provider or newer
  IREE lands.
- **StableHLO exporter (Phase A, `../stablehlo/`) emits illegal StableHLO for scalar
  broadcasts to dynamic shapes** (e.g. `relu` on a symbolic-dim tensor):
  `broadcast_in_dim` results MUST be statically shaped per the StableHLO spec; the spec'd
  dynamic form is `dynamic_broadcast_in_dim` (+ `get_dimension_size` shape operand), which
  iree-compile DOES support (validated). Such graphs fail at `compile()` with the
  compiler's diagnostics — explicit, never silent. Fix belongs in the exporter, not here.
- **iree-compile 20241104 nondeterminism on unsigned-int `reduce`** (uint32/uint64):
  identical input+flags flip between success and "failed to legalize unresolved
  materialization" — an upstream race; those dtypes are excluded from capabilities.
- **IREE demotes f64→f32 by default** for StableHLO input; the adapter always passes
  `--iree-input-demote-f64-to-f32=false` (silent dtype coercion would violate etl's
  dtype contract). `--iree-llvmcpu-target-cpu=generic` keeps artifacts portable and
  silences the generic-CPU warning.
- **`stablehlo.add` on i1 is XOR** per the StableHLO spec, while numpy's bool `+` is
  logical-or — an exporter semantic gap for bool arithmetic graphs (bool is validated and
  supported through logical/compare ops, which map unambiguously).
- Compile is a real iree-compile subprocess invocation (~1s per call) — `compile()` is
  not intended for per-call hot loops; cache artifacts explicitly (`etl.Cache`).
