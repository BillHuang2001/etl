# etl/backends/adapters — optional compiler adapters (TVM implemented; IREE/XLA pending)

## Intent

Pluggable compiler backends for etl: each adapter consumes the SHARED StableHLO lowering produced by `CompilerBackend.lower` (`../compiler.py` — capability pre-check, block-portable inlining, StableHLO export, Signature recording) and implements only the compiler-specific half: dependency probe (`check_available`), `compile` (invoke the external compiler), `load` (rebuild the executable — never recompiling). Activation is register-on-first-use: `etl.backends.registry.get("tvm")` imports the module and calls its `register()`; `import etl` never imports an adapter or its compiler dependency.

Status: **`tvm.py` implemented and validated** (TVM 0.26.0). `iree.py` / `xla.py` are documented framework slots, not implemented.

## API Surface

- `tvm.py`: `TvmBackend` (name `"tvm"`, a `CompilerBackend`), `TvmExecutable` (a `CompilerExecutable`), singleton `tvm_backend`, module-level `register()` (probes the dependency — `core.BackendError` with the pip hint when missing — then registers idempotently).
- `tvm_util.py`: the vendored-translator compatibility shim (`ensure_compat()`), the validated-op whitelist (`SUPPORTED_STABLEHLO_OPS`), compile-time gate (`parse_stablehlo`, `precheck_module`), and the translate/build/run/persist helpers (`translate`, `build_vm_executable`, `export_library_base64`, `load_virtual_machine`, `invoke`, `as_tvm_tensor`).
- `__init__.py`: documentation only, no imports.

## TVM adapter — validated design (all end-to-end, numpy-backend parity)

**Pipeline (TVM 0.26.0, tvm_ffi 0.1.13.post3, jax/jaxlib 0.10.2):**
- translate: `tvm.relax.frontend.stablehlo.from_stablehlo(mlir_text)` — parses with `jax._src.interpreters.mlir.make_ir_context()`; NOTE `tvm.relax.vm.build` does NOT exist in 0.26 (`vm` resolves to `tvm.runtime.vm`; build lives in `relax.vm_build`, re-exported as `tvm.relax.build`).
- build: `tvm.relax.vm_build.build(mod, target=tvm.target.Target("llvm"))` → `VMExecutable`; artifact `target="cpu - llvm"`.
- run: `tvm.runtime.vm.VirtualMachine(ex, tvm.runtime.cpu())["main"](...)`; inputs `tvm.runtime.tensor(np_array, tvm.runtime.cpu())` (the `tvm.nd` namespace no longer exists); outputs `tvm.runtime.Tensor.numpy()` → `core.Tensor(np.asarray(...))` (the numpy interpreter's canonical construction).
- persist: `VMExecutable.export_library(path)` writes a host .so; bytes are base64'd into the artifact payload (`format="tvm-vm-library"`, keeps `mlir_text` + `entry_functions`); `load` decodes to a temp file and `tvm.runtime.load_module(path)` — **no recompile at load** (validated: reloaded module runs identically). `Module.save_to_file` does not exist in 0.26. `runtime_dependencies={"numpy": ..., "tvm": ...}`; load enforces the recorded tvm version (mismatch ⇒ `core.PersistenceError`).

**Capabilities (validated):** `dynamic_shapes=True` (symbolic `tensor<?x8xf32>` runs at multiple concrete sizes — with the writer caveat below); `collectives=False` (the vendored translator has no collective handlers — the shared `lower()` rejects collective ops explicitly); `runtime_calls=False`, `custom_blocks=False`, `async_collectives=False`. `dtypes` = float16/float32/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64/bool (each run end-to-end; complex64/128 excluded — the vendored `_convert_data_type` raises `NotImplementedError`).

**Op coverage** (compile-time whitelist gate, `SUPPORTED_STABLEHLO_OPS`): arithmetic (add/subtract/multiply/divide/power/maximum/minimum), unary math (abs/negate/sign/sqrt/rsqrt/exp/log/log_plus_one/sine/cosine/tanh/logistic), compare (EQ/NE/LT/LE/GT/GE), select, bitwise/logical and/or/xor/not, convert, broadcast_in_dim, reshape, transpose, concatenate, slice (unit strides), pad (zero interior), reduce (add/maximum/minimum/multiply reducers), dot_general (matmul), constant. Rejected at compile with `core.BackendError` naming the op: control flow (`stablehlo.if`/`while`), gather/scatter, remainder, convolution and reduce_window (vendored handlers hardcode NHWC/HWIO — silently wrong for etl's NCHW conv), multi-function modules, multi-tensor-output functions (vendored importer keeps only the first output).

## Known Issues

1. **Vendor-compat shim (required):** the TVM 0.26.0 vendored StableHLO translator targets the OLD mlir python bindings (`ShapedType.isinstance` classmethods, `OpView` as an `Operation` subclass) and old FFI conventions; with jaxlib ≥0.9 (new bindings) it crashes on basic programs. `tvm_util.ensure_compat()` patches the five gaps (type-check classmethods, OpView/Operation key normalization, `broadcast_to` ShapeExpr, `DenseI64ArrayAttr` decoding, np.asarray constant decoding for float16) and extends the op map with validated handlers. Each patch restores exactly the semantics the vendored code assumed — no computation changes.
2. **etl StableHLO writer limitation (sibling node — NOT fixed here):** (a) symbolic-dim graphs that mix scalar-constant broadcasts (e.g. `relu`) export `broadcast_in_dim` to a dynamic-shape result, which the StableHLO verifier forbids — the adapter surfaces the parse error as `core.BackendError` at compile; (b) mixed-dtype binary ops with Python scalar constants (e.g. int32 tensor + int literal) export operands of mismatched types (no implicit `convert`) — same honest compile-time parse error. Workaround: cast operands explicitly / avoid scalar constants in those graphs.
3. **Version floor:** the adapter probes the exact APIs it calls at `check_available`; validated against `apache-tvm 0.26.0` + `jax/jaxlib 0.10.2`. pyproject's `apache-tvm>=0.14` extra is TOO LOOSE — 0.26 (from_stablehlo + tvm_ffi API) and a jaxlib matching the translator's mlir bindings are required (the extra must also include jaxlib — the translator imports `jax._src.interpreters.mlir`). Recommend `apache-tvm>=0.26` + `jaxlib` in the `tvm` extra.
4. `iree.py` / `xla.py` are not implemented (documented framework slots).

## Constraints (binding)

- Heavy-import rule: `tvm`/`jaxlib` imports only inside function bodies (`tvm_util` + `tvm.py` both); top-level imports limited to stdlib, `numpy`, `etl.core`, and the `..compiler`/`..program`/`..registry` siblings. Verified: `import etl` leaves `sys.modules` free of tvm/jax/iree.
- Errors: capability violations and translator/build failures raise `core.BackendError` naming the op/feature — never silent fallback; load-time mismatches raise `core.PersistenceError`; input dtype mismatches `core.DTypeError`, shape mismatches `core.ShapeError` (mirrors the numpy interpreter).
- Staging never composes: `load` never re-traces/lowers/compiles.
- CPU only (`llvm` target); non-CPU devices raise `core.BackendError`.

## Routing table

| Path | Area |
|---|---|
| `./__init__.py` | package marker + adapter-module contract (documentation only) |
| `./tvm.py` | `TvmBackend` (compile/load/check_available), `TvmExecutable` (run), `tvm_backend`, `register()` |
| `./tvm_util.py` | vendored-translator compat shim, op whitelist, compile-time gate, translate/build/run/persist helpers |

## Test strategy

Validated with throwaway scripts in `$TMPDIR` (not committed — per the framework's adapter validation pattern): registry auto-activation, numpy-backend parity (matmul+relu+reduce_sum; reshape/broadcast/transpose; symbolic-dim dot at two sizes), artifact/executable/wrapped-executable save-load round-trips, error paths (unsupported ops, dtype/shape/device mismatches, cross-backend artifact, collectives at lower). Committed tests for this adapter would live in `../../../tests/backends/` — a SIBLING directory; test-related writes escalate to the repo root. The full suite (`python3 -m pytest -q`) stays green (4113 collected, exit 0).
