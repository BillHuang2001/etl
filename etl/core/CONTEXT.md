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
- **Pytrees** (`tree.py`): `TreeSpec` (frozen; `.type/.children/.context/.node_data`, `.num_leaves`), `flatten(obj) -> (leaves, treespec)`, `unflatten(leaves, treespec)`, `register_pytree_node(type, flatten_fn, unflatten_fn)`; sugared API `tree_map(fn, *trees)`, `tree_leaves(tree)`, `tree_structure(tree)`, `tree_flatten(tree)`, `tree_unflatten(leaves, treespec)` — pure sugar over flatten/unflatten, no new semantics (`tree_flatten`/`tree_unflatten` are identity aliases: `tree_flatten is flatten`, `tree_unflatten is unflatten`); `tree_map` validates N-tree structure pairwise via `first_mismatch_path(..., strict=True)` and raises a plain `TypeError` naming the first mismatch path + described nodes (`describe_node`). Internal cross-module contract names (importable from `etl.core`, NOT top-level `etl` surface): `first_mismatch_path(spec_a, spec_b, *, strict=False, leaf_vs_empty_is_mismatch=False)`, `format_path(path)`, `describe_node(spec) -> str` — consumed by trace (`graph.py`), pipeline, transforms (vectorize's `_first_structure_mismatch`/`_format_path` are superseded by these).

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
| `dim.py` | `Dim`, `DimExpr` expression construction + `evaluate`/comparisons, `dim()` (implemented) |
| `spec.py` | `TensorSpec` (frozen dataclass + validation, implemented) |
| `tensor.py` | `Tensor` data holder + DLPack passthrough (implemented); creators/`from_numpy`/`from_dlpack` implemented |
| `symbolic.py` | `SymbolicTensor` + operator-handler registry + `constant` hook (implemented; op building lives in `../ops/`) |
| `device.py` | `Device` (implemented); `devices`/`split_tensor`/`replicate_tensor` implemented — `split_tensor`/`replicate_tensor` validate inputs up-front and raise `DeviceError` for invalid axis/input kind (no raw numpy/AttributeError leaks) |
| `tree.py` | `TreeSpec` + pytree registry (implemented); `flatten`/`unflatten` implemented; sugared `tree_map`/`tree_leaves`/`tree_structure`/`tree_flatten`/`tree_unflatten` (identity aliases of flatten/unflatten); cross-module mismatch contract `first_mismatch_path`/`format_path`/`describe_node`; container edge handling for `defaultdict`/`Counter` |

Cross-references: `../ops/` registers operator handlers + the constant builder into `core` at import time; `../ir/` provides the real `Value`/`Location` types that `SymbolicTensor.value`/`.location` duck-type; `../../tests/core/` is the test mirror (sibling — read-only from here, escalate writes to root).

## Design decisions

- **Operator-handler dispatch**: `SymbolicTensor.__add__` etc. call `_get_operator_handler(kind)(...)`. Kinds: `add, sub, mul, matmul, truediv, pow, neg, lt, gt, le, ge, eq, getitem`. Calling convention: binary `handler(left, right)`; `neg` `handler(operand)`; `getitem` `handler(obj, key)`. Missing handler (e.g. `etl.ops` not imported) → `TraceError` naming the kind and the fix. Reflected dunders (`__radd__`, ..., `__rpow__`) dispatch with swapped operands. `__eq__` returns a symbolic `equal` op (graph semantics), never a Python bool; `__bool__` raises `TraceError` pointing to `etl.cond`/`while_loop`/`scan`; `__hash__ = None` (unhashable, since `__eq__` is not a bool). `!=` is intentionally not defined (Python inverts `__eq__` → hits `__bool__` → clear `TraceError`).
- **`constant` hook**: `etl.constant` is listed under core's value model in the package contract, but building a Constant op requires `ops`. Resolved by the same hook pattern as operators: `ops` registers a builder via `register_constant_builder`; `core.constant` calls it or raises `TraceError`.
- **`bool_` naming**: the dtype constant is `bool_`, not `bool` (avoiding shadowing the builtin inside every consumer; the package contract uses `bool_`).
- **`Dim`/`DimExpr` as pure ASTs**: arithmetic dunders only construct nodes (shared `_DimArithmeticMixin`); `evaluate(dim_sizes)` does substitution/arithmetic with explicit bindings taking precedence over known sizes. `==` on dims/exprs is structural; comparisons (`< <= > >=`) evaluate "constraint-free" (known sizes only, no bindings) and raise `ShapeError` when unresolved; `bool()` on symbolic dims raises `ShapeError`. `dim(5)` → `Dim(name="dim_5", size=5)` (deterministic name, known size ⇒ exact evaluation).
- **DLPack zero-copy**: `Tensor.__dlpack__` delegates to the ndarray (same memory); `from_dlpack` must consume capsules without copies. `.numpy()` returns the underlying array reference (no copy). `Tensor` is unhashable (like ndarray) and compares structurally (dtype+shape+device+`array_equal`).
- **`TensorSpec` normalization**: shape tuple-ified and element-validated; dtype normalized via `dtype()`; rank always known. `SymbolicTensor.shape` additionally accepts `Dim` entries (callers resolve to `DimExpr`).
- **TreeSpec invariants**: pre-order leaves; `unflatten(flatten(x))` ≡ `x`; dict keys sorted; namedtuple/dataclass metadata in `node_data`; custom types via registry `{type: (flatten_fn, unflatten_fn)}`.
- **Pytree container edge behavior** (`tree.py`): `defaultdict` specs record `(default_factory, sorted_keys)` in `node_data`; `unflatten` rebuilds via `defaultdict(factory, dict(zip(keys, children)))` when the factory is `None` or a class/type, else raises plain `TypeError` naming the unpreservable factory (`!r`) and directing to `register_pytree_node` — `flatten` never fails on unpersistable factories (they are recorded for the error's repr). `Counter` rebuilds via the positional mapping constructor (`Counter(dict(zip(...)))`) so non-str keys work; plain `dict` and other dict subclasses (e.g. `OrderedDict`) keep the legacy `type(zip(keys, children))` rebuild exactly. The raw `TypeError` from `sorted(obj)` on unorderable mixed dict keys is wrapped (dict + all dict subclasses) as `flatten: cannot sort dict keys of mixed types (types: [...])` with distinct type names in first-appearance order — the raw error never leaks. A dataclass rebuild `TypeError` (InitVar without default / `init=False` fields) is wrapped as `unflatten: cannot rebuild dataclass ...` naming the offending fields in declaration order (ClassVar excluded; defaulted InitVars roundtrip fine since `__init__` accepts the recorded fields); namedtuple positional arity errors keep their raw form. `register_pytree_node(object, ...)` raises plain `TypeError` (would hijack the MRO lookup for all types); re-registering list/tuple/dict still replaces. `node_data` stays structurally comparable, so `defaultdict(list)` vs `defaultdict(None)` specs ARE a structure mismatch.
- **Shared structure-mismatch contract** (`tree.py`): `first_mismatch_path`/`format_path`/`describe_node` are the cross-module names trace/pipeline/transforms import. Semantics: container nodes must match on type (`!=`), `node_data`, and child count — a divergence (incl. differing dict keys or defaultdict factories) stops AT that node, so mismatch paths never descend past it; one childless node vs a node with children → mismatch at that prefix; leaf vs leaf OR leaf vs empty container match by default (grad/graph/pipeline semantics) while `leaf_vs_empty_is_mismatch=True` (vectorize's semantics) treats a 1-vs-0 `num_leaves` difference as a mismatch; `strict=True` (tree_map's semantics) additionally treats leaf-vs-empty AND empty-vs-empty containers with differing type/node_data as mismatches — it subsumes `leaf_vs_empty_is_mismatch=True` (which stays for the legacy callers); leaf vs leaf always matches regardless of leaf type. `describe_node(spec)` renders one-line node descriptions for mismatch messages (leaf → type name; containers → `dict with keys [...]`/`tuple of length N`/`namedtuple of length N`/`dataclass with fields [...]`/custom `Name of length N`, with empty containers falling through to the container wording). Descent sources dict-subclass keys from `node_data` (index 1 for defaultdict's `(factory, keys)` record), all other containers descend positionally.
- **`device.py` ↔ `tensor.py`**: `device.py` imports `Tensor` only under `TYPE_CHECKING` to avoid a cycle (`tensor.py` imports `Device`).
- **`split_tensor` axis-validation ordering**: bools are rejected up-front (they are `int`s but never a valid axis); non-numeric axes (str/None/complex/multi-element arrays) are rejected when normalization (`axis < 0` / `axis + ndim`) raises `TypeError`/`ValueError`; the range check stays *first* for numeric axes so an out-of-range non-int like `2.5` keeps "out of range" as the primary diagnostic (a test asserts this); in-range non-integral axes are rejected after the range check via `numbers.Integral` (which accepts numpy integer scalars such as `np.int32`). All invalid-axis cases share one `DeviceError` message naming the value and its type. Residual `np.split` errors are wrapped in `DeviceError` chained via `from exc` — never swallowed.

## Test strategy

`../../tests/core/` mirrors this package: `test_errors.py` (hierarchy), `test_dtypes.py` (normalizer + constants), `test_dim.py` (expression construction, `dim()` rules), `test_spec.py` (validation/normalization), `test_tensor.py` (zero-copy `.numpy()`, DLPack roundtrip incl. torch via `pytest.importorskip`), `test_symbolic.py` (purity: `SymbolicTensor` has no `numpy`/`data_ptr`/`__dlpack__`/`__array__`; dispatch raises `TraceError` in a fresh interpreter without `etl.ops`; `__bool__` raises), `test_device.py` (cpu always present; cuda only when detectable), `test_tree.py` (roundtrips, sorted dict keys, namedtuple/dataclass, custom registration). Spec-compliance items land in `../../tests/test_spec_compliance.py`.

## Status

**Implementation phase complete** — every value-model behavior is implemented and validated (`py_compile` clean, `import etl` clean, inline validation of dim evaluation/comparisons, pytree round-trips, all concrete creators incl. DLPack zero-copy round-trip, device enumeration/split/replicate, symbolic dispatch audit). No `NotImplementedError` stubs remain in `etl/core/*.py`.

## Notes for agents

- **numpy-2.x DLPack gotcha**: numpy >= 2.0 requires the keyword-only `max_version` argument on `ndarray.__dlpack__` (numpy < 2.0 rejects it). `Tensor.__dlpack__` therefore tries the plain `self.data.__dlpack__(stream=stream)` call first and retries with `max_version=(1, 0)` on `TypeError`. Do not "simplify" this back to a single call — that breaks on one numpy generation or the other.
