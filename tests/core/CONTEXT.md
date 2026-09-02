# tests/core — value-model test suite

## Intent

pytest mirror of `../../etl/core/` (the etl value-model foundation: errors, dtypes, Dim/DimExpr, TensorSpec, Tensor, SymbolicTensor, Device, pytrees). These tests are the executable spec of that module's documented contract (see `../../etl/core/CONTEXT.md` — API Surface + Design decisions are binding).

## Test files

| File | Covers |
|---|---|
| `test_errors.py` | `ETLError` hierarchy: base is exactly `Exception`; all 8 subclasses derive from `ETLError`+`Exception`, catchable as `ETLError`, message preserved, re-exported from `etl.core` and `etl` |
| `test_dtypes.py` | `dtype()` normalizer (np.dtype identity passthrough, python/scalar types, strings, duck-typed `.dtype` objects, `DTypeError` on bad input) + 14 dtype constants (np.dtype instances, name/itemsize/kind) |
| `test_dim.py` | `Dim`/`DimExpr` AST construction (`+ - * // %`, `.min()/.max()` methods; op strings are `add/sub/mul/floordiv/mod/min/max`), `DimExpr.evaluate(dim_sizes)` with bindings precedence, structural `==`, ordering comparisons, `bool()` → ShapeError, `dim()` rules (`dim(5)` → `dim_5` deterministic, passthrough, string → symbolic) |
| `test_spec.py` | `TensorSpec` shape tuple-ification, dtype normalization, `.rank`, frozen, device/name fields, None dims, invalid entries → TypeError/DTypeError, normalized equality |
| `test_tensor.py` | `Tensor` attrs, zero-copy `.numpy()` (identity), creators (tensor/zeros/ones/full/empty — zeros/ones/empty default float32, tensor/full infer with float64→float32 coercion), `from_numpy` zero-copy, structural `==`/`!=`, unhashable, `__dlpack__` capsules (numpy 1.x/2.x), `from_dlpack` zero-copy round-trip, torch interop (`pytest.importorskip`); **explicit device placement** pinned with duck-typed FAKE payloads (no GPU, no backend imports): ndarray data is host memory on `Device('cpu', 0)` (`Tensor(ndarray, device=<non-cpu>)` and non-cpu creator `device=` → `DeviceError`), a payload's device is derived from `payload.device` and a conflicting explicit `device=` → `DeviceError` (relabel ban), kind-aware `.numpy()` (cpu-kind payload → lazy FRESH host copy per call, metadata-only `__eq__`; non-cpu-kind → `DeviceError` "no implicit device"), per-kind `__dlpack__`/`__dlpack_device__` `DeviceError`s, and `Tensor.to(device)` (`TestTensorTo`): same-device → self, payload→cpu:0 = fresh host copy per call (never cached), host→non-cpu dispatches to `register_device_transfer_provider` (test-registered kinds only — never "cuda", which has an iree thunk), no provider → `DeviceError`, payload→different non-cpu device → explicit two-hop error |
| `test_symbolic.py` | `SymbolicTensor` purity (no `numpy`/`data_ptr`/`__dlpack__`/`__array__`), `__bool__` → TraceError (mentions etl.cond), `__hash__ is None`, construction validation, operator dispatch mechanics via fake handlers (kinds, arg order, reflected swaps, re-registration), missing-handler TraceError (registry cleared via fixture), real dispatch inside `etl.trace` |
| `test_device.py` | `Device` frozen/eq/repr/hash, `devices()` (cpu always present, deterministic, filtering), `split_tensor`/`replicate_tensor` as HOST-DATA-ONLY data-preparation helpers (source must be ndarray-backed on `Device('cpu', 0)`; every target must be `Device('cpu', 0)` — cuda or cpu-index≠0 targets raise the canonical host-only `DeviceError` carrying the `t.to(...)` remedy, and device-payload sources raise `DeviceError` too, never a raw `AttributeError`); host-side semantics unchanged: split pieces are views, replicas share one buffer, error paths (empty devices, axis, divisibility, non-tensor) |
| `test_tree.py` | `flatten`/`unflatten` round-trips (lists/tuples/dicts/namedtuples/dataclasses/empty/None/scalars/mixed), sorted dict keys, pre-order leaves, `num_leaves`, `TreeSpec` fields/node_data/frozen, etl value types as leaves, `register_pytree_node` (replace-on-re-register, non-type/non-callable → TypeError) |

## Constraints

- CPU only, no network, no GPU (device semantics are pinned with duck-typed fake payloads — see the Notes). Whole suite runs in ~4s (525 tests; budget 10s). Small shapes only.
- `../../etl/` is READ-ONLY — tests never modify the package; if a test exposes a real contract violation, keep the failing test with a `# BUG(etl): <desc>` comment and report it (none found to date).

## Notes for agents

- **Handler registration is global**: importing anything from `etl` runs the package `__init__`, which imports `etl.ops` and populates `etl.core.symbolic._OPERATOR_HANDLERS`. The missing-handler TraceError can therefore only be tested by saving/clearing/restoring the registry dict in a fixture (see `test_symbolic.py`) — a fresh interpreter won't help.
- **Operator dispatch integration** uses the capture-list trick: append op results to a Python list from inside the function passed to `etl.trace`, then assert types after tracing (see `test_symbolic.py::test_operators_build_symbolic_results_inside_trace`).
- **numpy ≥2 DLPack**: `np.from_dlpack` takes an object *exposing* `__dlpack__` (a raw PyCapsule fails); etl's `from_dlpack` documents the same input contract. Tests use the documented paths.
- `Dim` has no `evaluate` (only `DimExpr` does); `bool()` on dims raises ShapeError even for known sizes — both are source behaviors asserted as-is.
- The `bool_` dtype constant's numpy canonical `.name` is `"bool"` (identifier is `bool_`).
- **Explicit-placement pins use duck-typed fake payloads** (`_DummyDevicePayload`, `_AsarrayOnlyPayload` in `test_tensor.py`; `_DummyDevicePayload` in `test_device.py`) — no GPU and no backend imports. `Tensor.to` provider tests register made-up device kinds (never "cuda": `etl.backends` registers a lazy iree thunk for it at import time, which would pull in the iree adapter on first call) and clean the process-global `_DEVICE_TRANSFER_PROVIDERS` registry up in `finally` (see `TestTensorTo`'s docstring).
- `split_tensor`/`replicate_tensor` are host-data-only data-preparation helpers: the source must be an ndarray-backed tensor on `Device('cpu', 0)` and every target must be `Device('cpu', 0)` — non-host targets (cuda, cpu index ≠ 0) and device-payload sources raise the canonical `DeviceError` with the `t.to(...)` remedy (asserted via `_HOST_ONLY_FRAGMENTS`), never a raw `AttributeError` from a payload probe. Pieces still share memory: `split_tensor` returns views of the source and `replicate_tensor` shares ONE buffer across the (all-cpu:0) replicas — don't "fix" the tests to expect distinct buffers or device-tagged replicas.
