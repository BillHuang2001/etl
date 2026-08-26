# tests/backends — backend-layer test suite

## Intent

pytest suites validating `etl.backends` (sibling: `../../etl/backends`): the backend interface/registry/program contracts, the numpy interpreter backend (numerical reference), the stablehlo exporter, and deferred-error behavior. Tests are the executable spec: graphs executed by a backend must match direct numpy references computed with the SAME formulas as the kernels in `etl/backends/numpy/kernels/`.

## Files

| File | Area |
|---|---|
| `test_numpy_interpreter.py` | The big numerical-correctness suite (~300 parametrized cases): elementwise arith/activations (erf via `math.erf`, erf-based gelu, sigmoid), comparisons/logical/bitwise/cast, reductions (incl. argmax/argmin, keepdims, dtype promotion rules), linalg (dot/solve/conv/tril/triu/cumsum — conv vs a manual cross-correlation reference), structure ops (reshape/transpose/slice/gather/scatter/concatenate/pad/select/broadcast), static getitem, enp sugar, operator overloads, and the evaluate/build/run pipeline ground truth. |
| `test_backend_interface.py` | (to be written) Backend protocol, Capabilities, registry. |
| `test_program.py` | (to be written) LoweredProgram / CompiledArtifact contracts. |
| `test_runtime_call.py` | (to be written) runtime_call / callback semantics. |
| `test_control_flow.py` | (to be written) cond/while_loop/scan execution. |
| `test_collectives.py` | (to be written) dist collective semantics. |
| `test_stablehlo.py` | (to be written) stablehlo exporter. |
| `test_symbolic_dims.py` | (to be written) symbolic-shape execution/binding. |
| `test_artifact_persistence.py` | (to be written) save/load round-trips. |
| `test_deferred_errors.py` | (to be written) backend error surfaces. |

## Conventions (test_numpy_interpreter.py — binding)

- Execution via `etl.evaluate(fn, *args, backend=etl.backends.numpy_backend)`; results are `etl.Tensor` (unwrap with the local `as_np`).
- References re-implement kernel formulas (never import etl kernels): `math.erf` for erf, `0.5*x*(1+erf(x/sqrt(2)))` for gelu, manual dilate→pad→cross-correlate loops for conv, the kernel's index normalization for scatter.
- Exact equality (`assert_array_equal`) for int/bool; float32 rtol=atol=1e-6, float64 rtol=atol=1e-12. Result dtype is asserted against the reference too.
- Documented etl quirks encoded in tests (do not "fix" etl here): `etl.slice` lengths are spans (limit = start + length, numpy `x[s:s+l:st]` semantics); `solve` requires rhs `(..., n, k)` for batched `a` (numpy's bare `(batch, n)` vector form is rejected as ambiguous); getitem supports only static int/slice (ellipsis/mask/int-array raise TraceError); `dot` requires rank >= 2.
- Small shapes, deterministic data (linspace/arange-based), CPU only, <2s per file. When a test exposes a real etl bug: keep it failing with a `# BUG(etl)` comment, never weaken it.
