# etl — EvoX Tensor Library

## Intent

A **minimal, explicit, compiler-neutral tensor graph runtime** (Python-first). etl provides an ergonomic tensor language for constructing, transforming, inspecting, caching, loading, and executing computational graphs. It deliberately does **not** own optimization, kernel scheduling, memory planning, or hardware code generation — those belong to external compilers (IREE, XLA via PJRT, TVM).

**Defining principle: Make computation explicit, require large tensors to be explicit inputs, keep `vmap` as transparent function-side sugar over `vectorize`, keep binding as transparent argument-passing sugar, keep communication and multi-device tensor preparation explicit, expose and persist the graph, and let compilers do the compilation.**

etl learns from Nx (explicit numerical language, `block`, `vectorize`, `runtime_call`, backend abstraction), early JAX (NumPy ergonomics, `vmap`/`grad`/`jvp`/`vjp`), PyTorch (interop), and tinygrad (small inspectable IR). The Python API stays semantically close to Elixir Nx so a future Elixir implementation can extend Nx rather than duplicate it.

## Design principles (binding for all submodules)

1. **The graph is the program.** `@etl.defn` marks a numerical graph definition — it is NOT JIT or eager execution. There is no implicit tracing and no eager/graph mode switching.
2. **Explicit staging.** Pipeline: `defn → trace → Graph → transform → Graph → lower → LoweredProgram → compile → CompiledArtifact → load → Executable → run → Tensor`. Every stage has an explicit public function. Convenience APIs (`etl.build`, `etl.evaluate`) are *documented shorthand* compositions only — no additional hidden semantics.
3. **Minimal magic.** No API silently traces, compiles, specializes, moves/reshards tensors, inserts collectives, executes Python callbacks, or switches semantics. Caching/binding are explicit operations. Sugar is allowed only as transparent composition of explicit primitives.
4. **Python value → Python semantics; SymbolicTensor → graph semantics.** `if etl.sum(x) > 0:` must fail clearly; runtime tensor control flow is `etl.cond` / `etl.while_loop` / `etl.scan`.
5. **Local tensors + explicit communication.** A tensor is always a local physical tensor. Collectives (`etl.dist.*`) appear explicitly in the program; the compiler may optimize but never invent communication. Multi-device helpers (`split_tensor`, `replicate_tensor`) are explicit data-preparation utilities, not sharding transformations.
6. **Large tensors must be explicit inputs.** Capturing a concrete `Tensor` from a closure into a graph is an error. `etl.constant(w)` is the only (explicit, warned) way to embed tensors. `etl.bind` is argument-passing sugar only.
7. **Frontend transformations are graph→graph.** `vectorize`/`vmap`/`grad`/`jvp`/`vjp` produce ordinary graphs of ordinary ops. Backends never need to understand `vmap` or Python containers.
8. **Compiler-neutral IR.** EvoXIR (region-based SSA) is the frontend IR. StableHLO is an important export target but not the definition of the IR. Backend limitations fail explicitly.
9. **No eager numerical implementation duplication.** Concrete creators (`etl.zeros`, `etl.tensor`, `etl.from_numpy`, …) and a reference CPU backend are the only concrete-computation paths; random generation, `linspace`, etc. go through compiled graphs.

## Value model (four distinct concepts)

| Concept | Meaning | Has storage? | Key capabilities |
|---|---|---|---|
| Python/static value | `None`, bool, int, float, complex, str, Enum, dtype, slice, config objects | n/a | Evaluated at trace time; specializes the graph. Changing it = new graph. No hidden guards/recompile. |
| `TensorSpec` | Describes a future runtime tensor | No | `shape` (symbolic dims allowed), `dtype`, optional `device`. Lifecycle: spec → trace → SymbolicTensor → run → Tensor. |
| `SymbolicTensor` | SSA value inside a graph | **No** | dtype, symbolic shape, source location, SSA identity. `x.numpy()`, `data_ptr()`, `__dlpack__` must NOT exist on it. |
| `Tensor` | Materialized runtime tensor | Yes | dtype, concrete shape, runtime/device, DLPack interop (`from_dlpack` / `__dlpack__`), `.numpy()`. Wraps numpy host memory (v1) or IREE/PJRT/TVM buffers (future). |

## Architecture overview

```
                         etl  (package)
        tensor language (ops, numpy namespace, trace, block)
        graph transforms (transforms: vectorize/vmap/grad/jvp/vjp)
        explicit collectives (dist)
                         │
                         ▼
                      EvoXIR (ir)
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
          StableHLO     native     (future: Relax/TVM, PJRT/XLA)
          (export)      numpy interpreter backend
────────────────── runtime boundary ──────────────────
                     Tensor  +  DLPack
```

**Pipeline objects (public types):** `Defn`, `Graph`, `LoweredProgram`, `CompiledArtifact`, `Executable`, `Tensor`. Each is a distinct public type; functions never silently consume an earlier-stage object and perform missing steps.

## Repository layout & routing table

| Path | Area |
|---|---|
| `./etl/` | The package. Public API contract lives in `./etl/CONTEXT.md` — read it before touching anything. |
| `./tests/` | pytest suite mirroring `./etl/` module structure + spec-compliance tests (design principles). |
| `./pyproject.toml` | Packaging, deps (numpy only), pytest config. |

Cross-references: none yet (sibling dirs are read-only — escalate writes to parent).

## Cross-module contracts (binding)

**Import acyclicity (strict):** dependency layers, each importing only lower layers:
`core` ← `ir` ← `ops` ← {`trace`, `numpy`(enp)} ← {`block`, `transforms`, `backends`, `dist`, `pipeline`, `persist`}.
Details: `core` imports nothing from etl (numpy only). `ir` imports `core`. `ops` imports `core`, `ir`, `trace` (for the active-builder hook — `trace` must NOT import `ops`). `trace` imports `core`, `ir`. `block` imports `core`, `ir`, `ops`, `trace` (lazily for portable tracing). `transforms` imports `core`, `ir`, `ops`, `trace`. `backends` imports `core`, `ir` (plus lazy `ops`/`block` access for block-impl registration). `dist` imports `core`, `ir`, `trace`. `pipeline` imports `trace`, `backends`, `core`. `persist` imports `core` only.

**Key shared contracts** (defined in `./etl/CONTEXT.md` and owned by the listed module):
- `core`: dtypes, `Dim`/`DimExpr` (symbolic shapes), `TensorSpec`, `Tensor`, `SymbolicTensor`, `Device`, TreeSpec (pytree flatten/unflatten), device helpers (`devices`, `split_tensor`, `replicate_tensor`), plus a **registration hook** so `ops` can install `SymbolicTensor` operator handlers without import cycles.
- `ir`: `Value`, `Op`, `Block`, `Region`, `Function`, `Module`, `Builder`, `Location`, effect annotations, `verify()`, IR serialization.
- `ops`: frontend op functions (`add`, `dot`, `sum`, `reshape`, …) + `runtime_call`. Called with `SymbolicTensor` → builds IR into the active builder; called with concrete `Tensor` → `TraceError` (no eager mode, clear message suggesting `etl.constant` / explicit inputs / `etl.evaluate`); called outside a trace → `TraceError`.
- `trace`: `defn`, `trace`, `Graph`, `cond`, `while_loop`, `scan`, current-builder context, static-value snapshotting.
- `block`: custom ops — declaration, portable impl, per-backend impl registry, batching rules, derivative rules.
- `transforms`: `vectorize`, `vmap`, `grad`, `jvp`, `vjp` — graph→graph.
- `backends`: `Backend` interface (`lower`/`compile`/`load`/capabilities), registry (lazy activation of optional adapters), numpy interpreter backend (default, CPU), stablehlo exporter (MLIR text — also the shared lowering input for compiler backends), and pluggable compiler adapters `iree` / `xla` (PJRT C API via ctypes — user-provided plugin `.so`, no jax/jaxlib) / `tvm` (jaxlib only for its bundled MLIR bindings — the `jax` package is never imported) in `etl.backends.adapters` — optional, activated by `lower(..., backend="iree"|"xla"|"tvm")`, explicit errors when absent (pip hints for iree/tvm; `ETL_PJRT_PLUGIN`/`plugin_path` guidance for xla).
- `dist`: `group`, collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `collective_permute`), `rank`/`world_size` graph scalars.
- `pipeline`: `LoweredProgram`, `CompiledArtifact`, `lower`, `compile`, `load`, `run`, `bind`, `build`, `evaluate`.
- `persist`: versioned/self-describing/integrity-checked save-load container, explicit `Cache`.

## Error strategy (binding)

All public errors derive from `ETLError` (in `core`): `TraceError`, `ShapeError`, `TransformError` (axis mismatch, unsupported batching/diff — never silent fallback), `BackendError`, `PersistenceError`, `DeviceError`, `DTypeError`, `VerificationError`. No API silently swallows or works around errors. Error messages include source location (e.g. `model.py:83`) whenever a graph location exists.

## Dependencies, testing, and hardware policy

- **Deps:** `numpy` is the only hard runtime dependency. Pure Python — no native C/Rust code in the library. Optional extras: torch (DLPack interop tests; `bench` — required only when running `etl.bench` comparisons against PyTorch), IREE/TVM (future adapters).
- **Testing:** pytest, CPU only. Tests mirror package structure under `./tests/`. torch interop tests use `pytest.importorskip("torch")`. Spec-compliance tests live in `./tests/test_spec_compliance.py` (closure capture errors, SymbolicTensor has no `.numpy()`, staging explicitness, bind is sugar, vmap=vectorize sugar, etc.).
- **GPU policy:** v1 never requires GPU — the reference backend is a CPU numpy interpreter. GPU is only considered later for optional adapters; if ever used, scan for an empty GPU first and do not occupy it for long.

## Non-goals (etl core must NOT contain)

NN modules, layers, optimizers, datasets, loaders, training loops, model zoo; automatic global sharding/resharding/implicit collectives; CUDA allocator / GPU runtime / kernel scheduler; compiler optimization passes; Triton compiler; distributed job scheduler, checkpoint manager, model serving. Separate libraries build those on top.

## Status

Implemented and tested. The full explicit pipeline (`trace → lower → compile → load → run` plus `bind`/`build`/`evaluate` sugar), graph transforms (`vectorize`/`vmap`/`grad`/`jvp`/`vjp`), explicit collectives (`dist`), custom blocks (`block`), the `etl.numpy` graph namespace, the numpy interpreter backend (default, CPU), the StableHLO exporter, the pluggable compiler adapters (IREE — CPU and CUDA validated end-to-end: `target_backends=["cuda"]` compile option + `Device("cuda", N)` load; XLA via the PJRT C API — a ctypes driver over a user-provided plugin `.so`, no jax/jaxlib, plumbing validated via a compiled test-plugin, real-XLA parity pending a real plugin; TVM with the `jax` package removed — jaxlib only for its bundled MLIR bindings — `etl.backends.adapters`, lazy optional extras, CPU-validated end-to-end with numpy parity where the compiler is present), and persistence/caching (`persist`, Graph save/load) are in place. A standalone conformance & benchmark harness (`etl.bench`) ships in the package: curated example programs (matmul, conv2d variants, softmax/layernorm, MLP, cumsum, attention, …) run through the explicit pipeline and compared against pure-numpy references (and PyTorch references when torch is available), reporting precision metrics (conformance) and wall-clock ms/run with speedup ratios (benchmark); CLI via `python -m etl.bench`. Torch is strictly optional — imported lazily only inside torch-comparison paths, never at module scope. Test suite (`./tests/`): 4197 tests collected, 0 failures; skips depend on torch presence — 12 without torch (8 torch-interop + 3 `etl.bench` torch-present regression tests + 1 xla-contract module without a PJRT plugin), 3 with torch installed (2 `etl.bench` torch-absent inverse tests + 1 xla-contract module). See `./etl/CONTEXT.md` for the package-level public API contract and `./tests/CONTEXT.md` for the test strategy.

**v1 deferrals (intentional, explicit errors — never silent fallback):** control-flow vectorization; `conv` VJP (no transposed-conv op); symbolic-length `scan`; `linspace` / `linalg.inv` / `norm` / `det`; non-constant pad modes; symbolic-bound `arange`; a few StableHLO op exports. Compiler-adapter limitations (all explicit `BackendError`, never silent fallback — see `etl/backends/CONTEXT.md` Known Issues): collectives and `runtime_call` are not compiled by any adapter in v1 (multi-rank runtime channels not yet wired), a few StableHLO op exports remain deferred (`gather`/`scatter`/`scan`/…), and adapters are validated on CPU only.
