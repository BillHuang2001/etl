# tests/backends — backend-layer test suite

## Intent

pytest suites validating `etl.backends` (sibling: `../../etl/backends` — read its CONTEXT.md and `../../etl/CONTEXT.md` before changing expectations): the backend interface/registry/program contracts, the numpy interpreter backend (the numerical reference), the StableHLO exporter, and error/persistence behavior. Tests are the executable spec of the backend contract: graphs executed by the numpy backend must match direct numpy references computed with the SAME formulas as the kernels in `etl/backends/numpy/kernels/`.

## Files

| File | Area | Tests |
|---|---|---|
| `test_backend_interface.py` | `Capabilities` fields/defaults/frozen-ness, numpy_backend flags (dynamic_shapes/collectives/runtime_calls/custom_blocks True, async_collectives False) + dtype set, Backend ABC abstractness, Executable protocol, registry register/get semantics (idempotent re-register, duplicate-name, unknown-name, non-Backend, empty-name → BackendError), LoweredProgram/CompiledArtifact attrs + save/load smoke | 63 |
| `test_program.py` | numpy LoweredProgram `.text()`/`.payload` (serialized ir.Module dict), Signature fields (input/output TreeSpec, per-leaf TensorSpec, static_values, output_static_values default `()`), structured inputs (dict/dataclass), signature == `graph.signature_info()`, save/load round-trip, frozen Signature | 13 |
| `test_numpy_interpreter.py` | THE big numerical-correctness suite (~300 parametrized cases): elementwise arith/activations (erf via `math.erf`, erf-based gelu, sigmoid), comparisons/logical/bitwise/cast, reductions (axes/keepdims/dtype promotion, argmax/argmin), linalg (dot/solve/conv/tril/triu/cumsum — conv vs a manual cross-correlation reference), structure ops (reshape/transpose/slice/gather/scatter/concatenate/pad/select/broadcast), static getitem, enp sugar, operator overloads, evaluate/build/run ground truth | 303 |
| `test_symbolic_dims.py` | one-build/multi-size symbolic dims (B=3→5), DimExpr reshapes (B*2,), two symbolic dims transpose, None dims, determinism, mixed symbolic+concrete | 8 |
| `test_control_flow.py` | cond both branches + runtime predicate, while_loop iteration counts/0-iteration/early-exit, scan (cumsum), nested cond-in-while and while-in-cond, control flow under symbolic B | 11 |
| `test_runtime_call.py` | runtime_call sync execution (callback receives np.ndarray, called exactly once), scalar operands (arrive as 0-d float64), multi-output, dtype mismatch → BackendError, shape mismatch → ShapeError, wrong count → BackendError, missing callback registration → BackendError naming id | 10 |
| `test_collectives.py` | single-rank identity for all six collectives, rank/world_size defaults (0/1), multi-rank simulation via `set_collective_executor` hook (all_reduce sums, all_gather concatenates, arg forwarding incl. v1 protocol limitations), `rank_context=` kwarg + thread-local `set_rank_context`, hook set/get/reset (None → get raises BackendError, non-conforming → TypeError), RankContext validation | 22 |
| `test_stablehlo.py` | golden mnemonics (add/dot_general/reduce/reshape/transpose/concatenate/broadcast_in_dim/constant/if/while/collectives + module/func.func/return shell), dtype formatting (`tensor<2x3xf32>`, bool→i1), comparison_direction attrs, program-order emission, determinism, symbolic dims → `?`, decomposition emission (square/relu/stop_gradient/reduce_mean), deferred ops → BackendError naming the op (gather/scatter/scan/runtime_call/block_call/rank/world_size/argmax/argmin/erf/gelu + unmapped tril/triu/cumsum/solve), TypeError for non-graph input | 55 |
| `test_deferred_errors.py` | numpy lower-time rejection: block_call with no numpy impl and no portable decomposition → BackendError at `etl.lower` (verify() passes first — genuinely lower-time); compile backend mismatch; load with GPU device → BackendError/DeviceError; stablehlo NOT a registered backend; numpy capability probe (nothing standard is deferred) | 8 |
| `test_artifact_persistence.py` | LoweredProgram/CompiledArtifact/Executable save/load round-trips (incl. re-run after load), corrupt/truncated/wrong-payload-type files → PersistenceError, unregistered recorded backend → PersistenceError, explicit backend-name mismatch at load → PersistenceError, device mismatch (GPU) → BackendError, runtime_dependencies tamper → PersistenceError, resave idempotence | 11 |
| `_adapter_utils.py` | shared helpers for the three adapter test files (graphs, fixed-seed inputs, cross-compiler fp32 tolerance, `stage`/`assert_parity`) — NOT collected (underscore prefix; import package-qualified) | n/a |
| `test_adapter_iree.py` | optional IREE adapter contract via the shared CompilerBackend framework: registration/singleton, name-string resolution through lower/build/evaluate, numerical parity vs numpy backend, symbolic dims iff `dynamic_shapes` else BackendError naming the feature, StableHLO payload contract, artifact save/load round-trips (pipeline + raw `run([flat tensors])`), name mismatch ⇒ PersistenceError, capability rejections naming the feature | 8 |
| `test_adapter_xla.py` | same contract as the IREE file, against the XLA adapter (`jaxlib` CPU PJRT via ctypes, not the `jax` package) | 8 |
| `test_adapter_tvm.py` | same contract as the IREE file, against the TVM adapter (apache-tvm) | 8 |

## Conventions (binding for this directory)

- Run from repo root: `python3 -m pytest -q tests/backends` (root `conftest.py` handles sys.path). CPU only, numpy-only deps, small shapes, <2s per file. Total runtime ~2.4s.
- Execute graphs via `etl.evaluate(fn, *args, backend=etl.backends.numpy_backend)` or `etl.build`+`etl.run`; results are `etl.Tensor` — unwrap with `.numpy()`.
- Numerical references re-implement kernel formulas (never import etl kernels): `math.erf` for erf, `0.5*x*(1+erf(x/sqrt(2)))` for gelu, manual dilate→pad→cross-correlate loops for conv.
- Exact equality for int/bool; float32 rtol=atol=1e-6, float64 rtol=atol=1e-12; result dtype asserted.
- Documented etl quirks encoded as tests (do NOT "fix" etl here): `etl.slice` lengths are spans (limit = start+length, numpy `x[s:s+l:st]` semantics); `solve` requires rhs `(..., n, k)` for batched `a`; getitem supports only static int/slice; `dot` requires rank ≥ 2; etl.scan desugars to while+gather+scatter at trace time (its stablehlo export error names `gather`).
- Shared state hygiene: collective-executor hook and rank context are process/thread global — tests that touch them use fixtures to save/restore.
- **Adapter test conventions** (`test_adapter_{iree,xla,tvm}.py` + `_adapter_utils.py`): these pin the SHARED contract of the `etl/backends/compiler.py` CompilerBackend framework + the three optional adapters (`etl/backends/adapters/`). Each file is structurally identical, guarded by module-level `pytest.importorskip(<dep>)` calls BEFORE the etl imports, uses `NAME = "iree"/"xla"/"tvm"`, and shares all graphs/inputs/tolerances via `_adapter_utils` (import as `from tests.backends import _adapter_utils as u`). Real compilers run, so the <2s-per-file rule does NOT apply — keep compile count low (one shared module-scoped `adapter_artifact` fixture + one compile per distinct graph); fp32 parity tolerance is 1e-5 (cross-compiler), not 1e-6.
- **BUG(etl) policy:** when a test exposes a real etl bug, keep it FAILING with a `# BUG(etl): <description>` comment and report it upward with a minimal repro. Never fix etl and never weaken the test.

## Known Issues

- The three `test_adapter_*.py` files are the forward spec for the IN-FLIGHT `etl/backends/compiler.py` + `etl/backends/adapters/{iree,xla,tvm}.py` implementation (parallel development). Until it lands, each file fails at COLLECTION with `ModuleNotFoundError: No module named 'etl.backends.adapters'` — EXPECTED and correct (NOT a BUG(etl) case; do not add skip guards around the etl-internal imports and do not weaken the tests).

- v1 collective-protocol limitations (asserted as-is): `reduce_scatter`'s `reduce_op` is not forwarded to the executor; `all_to_all` forwards only `split_axis`; `dist.broadcast` records the default `src_rank=0` in IR even when the user passes another value.
