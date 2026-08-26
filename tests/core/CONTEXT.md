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
| `test_tensor.py` | `Tensor` attrs, zero-copy `.numpy()` (identity), creators (tensor/zeros/ones/full/empty), `from_numpy` zero-copy, structural `==`/`!=`, unhashable, `__dlpack__` capsules (numpy 1.x/2.x), `from_dlpack` zero-copy round-trip, torch interop (`pytest.importorskip`) |
| `test_symbolic.py` | `SymbolicTensor` purity (no `numpy`/`data_ptr`/`__dlpack__`/`__array__`), `__bool__` → TraceError (mentions etl.cond), `__hash__ is None`, construction validation, operator dispatch mechanics via fake handlers (kinds, arg order, reflected swaps, re-registration), missing-handler TraceError (registry cleared via fixture), real dispatch inside `etl.trace` |
| `test_device.py` | `Device` frozen/eq/repr/hash, `devices()` (cpu always present, deterministic, filtering), `split_tensor` (slices, device tags, view semantics, error paths), `replicate_tensor` (shared buffer, distinct Tensor objects, device tags) |
| `test_tree.py` | `flatten`/`unflatten` round-trips (lists/tuples/dicts/namedtuples/dataclasses/empty/None/scalars/mixed), sorted dict keys, pre-order leaves, `num_leaves`, `TreeSpec` fields/node_data/frozen, etl value types as leaves, `register_pytree_node` (replace-on-re-register, non-type/non-callable → TypeError) |

## Constraints

- CPU only, no network, no GPU. Whole suite runs in ~0.5s (budget 2s). Small shapes only.
- `../../etl/` is READ-ONLY — tests never modify the package; if a test exposes a real contract violation, keep the failing test with a `# BUG(etl): <desc>` comment and report it (none found to date).

## Notes for agents

- **Handler registration is global**: importing anything from `etl` runs the package `__init__`, which imports `etl.ops` and populates `etl.core.symbolic._OPERATOR_HANDLERS`. The missing-handler TraceError can therefore only be tested by saving/clearing/restoring the registry dict in a fixture (see `test_symbolic.py`) — a fresh interpreter won't help.
- **Operator dispatch integration** uses the capture-list trick: append op results to a Python list from inside the function passed to `etl.trace`, then assert types after tracing (see `test_symbolic.py::test_operators_build_symbolic_results_inside_trace`).
- **numpy ≥2 DLPack**: `np.from_dlpack` takes an object *exposing* `__dlpack__` (a raw PyCapsule fails); etl's `from_dlpack` documents the same input contract. Tests use the documented paths.
- `Dim` has no `evaluate` (only `DimExpr` does); `bool()` on dims raises ShapeError even for known sizes — both are source behaviors asserted as-is.
- The `bool_` dtype constant's numpy canonical `.name` is `"bool"` (identifier is `bool_`).
- `replicate_tensor` deliberately tags the SAME ndarray buffer with each device (no copies) — don't "fix" the test to expect distinct buffers.
- Minor undocumented robustness gaps in `etl/core/device.py` (non-int axis leaks numpy TypeError, non-Tensor input leaks AttributeError) are asserted per-source with comments in `test_device.py` — see that file before re-litigating them.
