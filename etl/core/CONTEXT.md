# etl/core — value-model foundation

## Intent

The base of the etl import DAG (`core ← ir ← ops ← trace ← ...`). `core` owns the four value-model concepts (Python/static values, `TensorSpec`, `SymbolicTensor`, `Tensor`), dtypes, symbolic dimensions, devices, pytrees, and the error hierarchy. It imports **nothing from etl** — only numpy and the stdlib — and contains no IR, ops, tracing, backend, or pipeline logic.

## API Surface

All names below are re-exported from `etl/core/__init__.py` (see also `../CONTEXT.md` for the package-level public contract — binding).

- **Errors** (`errors.py`): `ETLError` + subclasses `TraceError`, `ShapeError`, `TransformError`, `BackendError`, `PersistenceError`, `DeviceError`, `DTypeError`, `VerificationError`.
- **Dtypes** (`dtypes.py`): `dtype(obj) -> np.dtype` (normalizer) + constants `float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool_, complex64, complex128` (numpy dtype objects). `bool_` (not `bool`) avoids shadowing the builtin.
- **Symbolic shapes** (`dim.py`): `Dim(name, size=None)`, `DimExpr(op, left, right)` (`+ - * // % min max` construct expression trees; evaluation is `evaluate(dim_sizes)`), `dim(name_or_int) -> Dim`.
- **Specs** (`spec.py`): `TensorSpec(shape, dtype, device=None, name=None)` — shape tuple of `Dim|DimExpr|int|None` (None = runtime-dynamic); dtype normalized to `np.dtype`; frozen dataclass; `.rank` property.
- **Concrete tensors** (`tensor.py`): `Tensor(data, device=None)` (attrs `data`, `dtype`, `shape`, `device`; `.numpy()` zero-copy reference; `__dlpack__(stream=None)`; structural `__eq__`; unhashable), helpers `from_numpy`, `from_dlpack`, creators `tensor/zeros/ones/full/empty`.
- **Symbolic values** (`symbolic.py`): `SymbolicTensor(value, dtype, shape, location=None)` — SSA identity `value.id` (duck-typed ir.Value protocol, must expose `.id`); MUST NOT define `numpy`/`data_ptr`/`__dlpack__`/`__array__`; operator overloads dispatch through the handler registry; `constant(tensor)` via the constant-builder hook.
- **Operator dispatch** (`symbolic.py`): `register_operator_handlers(kind, handler)` (public) + `_get_operator_handler(kind)`, `register_constant_builder(fn)`, `_get_constant_builder()` (internal cross-module contracts for `etl.ops`).
- **Devices** (`device.py`): `Device(kind, index=0)` (frozen), `devices(kind=None)`, `split_tensor(tensor, axis, devices)`, `replicate_tensor(tensor, devices)`.
- **Pytrees** (`tree.py`): `TreeSpec` (frozen; `.type/.children/.context/.node_data`, `.num_leaves`), `flatten(obj) -> (leaves, treespec)`, `unflatten(leaves, treespec)`, `register_pytree_node(type, flatten_fn, unflatten_fn)`.

## Constraints

1. **No etl imports, ever.** Only numpy + stdlib. This is enforced by design (the operator-handler and constant-builder hooks exist precisely to keep `ops` out of `core`).
2. All public errors derive from `ETLError`; messages include source location when a graph location exists.
3. Files under ~1000 lines; one concern per file (see routing table).
4. `SymbolicTensor` must never grow `numpy`/`data_ptr`/`__dlpack__`/`__array__` or become hashable.
5. Concrete creators are the only concrete-computation paths — random generation, `linspace`, etc. must never be added here (they go through compiled graphs).
6. No behavior belongs in `core` beyond value-model data types and trivial plumbing (see Status).

## Routing table

| File | Responsibility |
|---|---|
| `errors.py` | `ETLError` hierarchy (fully implemented) |
| `dtypes.py` | dtype constants + `dtype()` normalizer (implemented) |
| `dim.py` | `Dim`, `DimExpr` expression construction, `dim()`; `evaluate`/comparisons stubbed |
| `spec.py` | `TensorSpec` (frozen dataclass + validation, implemented) |
| `tensor.py` | `Tensor` data holder + DLPack passthrough (implemented); creators/`from_numpy`/`from_dlpack` stubbed |
| `symbolic.py` | `SymbolicTensor` + operator-handler registry + `constant` hook (dispatch implemented; op building lives in `../ops/`) |
| `device.py` | `Device` (implemented); `devices`/`split_tensor`/`replicate_tensor` stubbed |
| `tree.py` | `TreeSpec` + pytree registry (implemented); `flatten`/`unflatten` stubbed |

Cross-references: `../ops/` registers operator handlers + the constant builder into `core` at import time; `../ir/` provides the real `Value`/`Location` types that `SymbolicTensor.value`/`.location` duck-type; `../../tests/core/` is the test mirror (sibling — read-only from here, escalate writes to root).

## Design decisions

- **Operator-handler dispatch**: `SymbolicTensor.__add__` etc. call `_get_operator_handler(kind)(...)`. Kinds: `add, sub, mul, matmul, truediv, pow, neg, lt, gt, le, ge, eq, getitem`. Calling convention: binary `handler(left, right)`; `neg` `handler(operand)`; `getitem` `handler(obj, key)`. Missing handler (e.g. `etl.ops` not imported) → `TraceError` naming the kind and the fix. Reflected dunders (`__radd__`, ..., `__rpow__`) dispatch with swapped operands. `__eq__` returns a symbolic `equal` op (graph semantics), never a Python bool; `__bool__` raises `TraceError` pointing to `etl.cond`/`while_loop`/`scan`; `__hash__ = None` (unhashable, since `__eq__` is not a bool). `!=` is intentionally not defined (Python inverts `__eq__` → hits `__bool__` → clear `TraceError`).
- **`constant` hook**: `etl.constant` is listed under core's value model in the package contract, but building a Constant op requires `ops`. Resolved by the same hook pattern as operators: `ops` registers a builder via `register_constant_builder`; `core.constant` calls it or raises `TraceError`.
- **`bool_` naming**: the dtype constant is `bool_`, not `bool` (the package contract's `etl.bool` would shadow the builtin inside every consumer — see contract conflicts below).
- **`Dim`/`DimExpr` as pure ASTs**: arithmetic dunders only construct nodes (shared `_DimArithmeticMixin`); `evaluate(dim_sizes)` does substitution/arithmetic (stubbed). `==` on dims/exprs is structural; comparisons (`< <= > >=`) evaluate "constraint-free" and raise `ShapeError` when unresolved (stubbed); `bool()` on symbolic dims raises `ShapeError`. `dim(5)` → `Dim(name="dim_5", size=5)` (deterministic name, known size ⇒ exact evaluation).
- **DLPack zero-copy**: `Tensor.__dlpack__` delegates to the ndarray (same memory); `from_dlpack` must consume capsules without copies. `.numpy()` returns the underlying array reference (no copy). `Tensor` is unhashable (like ndarray) and compares structurally (dtype+shape+device+`array_equal`).
- **`TensorSpec` normalization**: shape tuple-ified and element-validated; dtype normalized via `dtype()`; rank always known. `SymbolicTensor.shape` additionally accepts `Dim` entries (callers resolve to `DimExpr`).
- **TreeSpec invariants**: pre-order leaves; `unflatten(flatten(x))` ≡ `x`; dict keys sorted; namedtuple/dataclass metadata in `node_data`; custom types via registry `{type: (flatten_fn, unflatten_fn)}`.
- **`device.py` ↔ `tensor.py`**: `device.py` imports `Tensor` only under `TYPE_CHECKING` to avoid a cycle (`tensor.py` imports `Device`).

## Test strategy

`../../tests/core/` mirrors this package: `test_errors.py` (hierarchy), `test_dtypes.py` (normalizer + constants), `test_dim.py` (expression construction, `dim()` rules), `test_spec.py` (validation/normalization), `test_tensor.py` (zero-copy `.numpy()`, DLPack roundtrip incl. torch via `pytest.importorskip`), `test_symbolic.py` (purity: `SymbolicTensor` has no `numpy`/`data_ptr`/`__dlpack__`/`__array__`; dispatch raises `TraceError` in a fresh interpreter without `etl.ops`; `__bool__` raises), `test_device.py` (cpu always present; cuda only when detectable), `test_tree.py` (roundtrips, sorted dict keys, namedtuple/dataclass, custom registration). Spec-compliance items land in `../../tests/test_spec_compliance.py`.

## Status

**Architecture phase complete** — all public interfaces, docstrings, and pure data types are in place and validated (`py_compile` clean, import acyclicity verified, smoke tests pass). **Stubbed (raise `NotImplementedError`) pending implementation phase:** `DimExpr.evaluate` + comparisons, `devices`, `split_tensor`, `replicate_tensor`, creators (`tensor/zeros/ones/full/empty/from_numpy/from_dlpack`), `flatten`, `unflatten`.

## Known issues (current state)

- Dim/DimExpr comparisons and `evaluate` are stubbed ⇒ comparing `TensorSpec`s whose shapes contain `DimExpr`s raises `NotImplementedError` until implemented.
- Contract conflict noted for the root/parent docs: package contract says dtype constant `etl.bool`; implemented as `bool_` (safer, per this node's architecture directive) — parent docs should be updated to `etl.bool_`. `VerificationError` lives here per the root error strategy (not in the parent's API-surface list).
