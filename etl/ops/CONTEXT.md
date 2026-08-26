# etl/ops — frontend tensor operations

## Intent

The frontend tensor-operation namespace: every numerical op that can appear in an EvoXIR graph. Each public function builds one IR op into the active builder (obtained from `trace.current_builder()`) and returns a `SymbolicTensor` wrapping the result value. `ops` is a **builder**, never an executor — no eager mode, no numpy computation of results. Parent contracts (`../../CONTEXT.md`, `../CONTEXT.md`) are binding; this file fixes the op-level contracts.

## Unified semantics (binding for every op function)

1. **Operands.** `SymbolicTensor` or Python scalars (`bool`/`int`/`float`/`complex`), the latter auto-promoted to 0-d Constant ops (transparent sugar) via `_utils.as_operand`. A concrete `Tensor` operand → `core.TraceError` with the mandated three-option message: (1) make it an explicit input, (2) embed explicitly with `etl.constant`, (3) etl has no eager mode — use `etl.evaluate`. Any other operand kind (list, ndarray, str, None) → `TypeError`. At least one operand of a binary op must be a `SymbolicTensor`.
2. **Trace requirement.** Every op calls `_utils.check_in_trace()` FIRST; no active trace → `core.TraceError` mentioning `etl.trace` / `@etl.defn` / `etl.evaluate`.
3. **Construction.** Build via `builder.create(op_name, ...)` on the active builder; op names, arities, attrs, and effects are declared in `_opdefs.py`. Attach a call-site `Location` (`_utils.get_location`) to every op; env `ETL_DISABLE_LOCATIONS=1` disables capture (then `ir.Location.unknown()`).
4. **Results.** Wrap the IR result value in `core.SymbolicTensor(value=..., dtype=..., shape=..., location=...)`. Shape is always statically computed (`DimExpr` arithmetic) at build time; runtime numpy enforces exact semantics.
5. **Static values.** All non-tensor parameters (axes, shapes, padding, `k`, dtype, callback) are static Python values that specialize the graph. Changing one = a different op; no hidden guards.
6. **Sugar rule.** `sum`/`max`/`min`/`mean`/`prod` are documented shorthand for `reduce_*` with the same kwargs — nothing else is sugar.

## Dtype promotion (design decision — binding)

- **Tensor ⊕ tensor:** exactly `numpy.result_type` (via `_utils.promote_dtypes`), including subtleties (`int8+uint8→int16`, `float32+int64→float64`, `float16+float16→float16`).
- **Python scalar ⊕ tensor:** NEP 50 weak promotion implemented EXPLICITLY in `_utils.weak_scalar_dtype` — behavior is identical on numpy 1.x and 2.x (`pyproject` pins `numpy>=1.24`, and numpy < 2 does not apply weak promotion in `result_type`). Rules: int/bool scalars promote DOWN to float16/32/64 and complex64/128 tensors when exactly representable (else float64/complex128); float scalars are weak toward complex64 only (float+float32 → float64); complex scalars are weak toward nothing (complex128). Integer/uint/bool tensors promote via plain `result_type`.
- **Unary math** (`sqrt`, `exp`, `log`, `log1p`, trig, `tanh`, `sigmoid`, `relu`, `gelu`, `erf`): integer/bool input → `float64`; float keeps dtype (numpy semantics). **Unary preserving** (`abs`, `negate`, `square`, `sign`): dtype unchanged.
- **Special:** `divide` → true division (int64/int64 → float64); `reduce_mean` int/bool → float64; `reduce_sum`/`reduce_prod`/`cumsum` bool → int64; `argmax`/`argmin` → int64; comparisons → bool; `logical_*` require bool (`DTypeError`); `bitwise_*` require integer/bool (`DTypeError`); `solve` int → float64; `gather` requires int32/int64 indices; `scatter`/`pad` cast their value operand to the tensor dtype; `cast` is exact.

## Symbolic broadcasting (design decision — binding)

`_utils.broadcast_shapes(*shapes)` implements numpy broadcasting over `int`/`Dim`/`DimExpr` dims:
1. Align right; missing dims are `1`.
2. Per pair: either is `1` → the other; equal → that dim; **both concrete ints and unequal → `ShapeError` at trace time** (statically known failure, never deferred); otherwise (≥1 symbolic dim) → `DimExpr.max(a, b)` — which IS numpy's runtime result shape; the numpy backend enforces the exact compatibility check at run time.
3. Result rank = max input rank.

Reductions: `_utils.normalize_axes` (None=all, negatives shifted, sorted, deduped) + `_utils.reduced_shape` (axes removed, or kept as 1 with `keepdims`; all-axes no-keepdims → scalar `()`). `dot`: batch dims broadcast via `broadcast_shapes`, `k` dims compatible per the broadcast rule (1 broadcasts), ranks ≥ 2. `conv`: out dims = `(d + 2*pad - kdil*(k-1) - 1) // stride + 1` as DimExpr. `reshape`: ≤ one `-1`, inferred via DimExpr product. `concatenate`: axis dim = DimExpr sum; non-axis dims must match (static mismatch → `ShapeError`, symbolic conflicts defer to runtime).

## Operator-handler protocol (binding)

`_registration.py` registers into `core` at import time (kinds from `../CONTEXT.md`): `add, sub, mul, matmul, truediv, pow, neg, lt, le, gt, ge, eq, getitem`.

- Binary handlers: `(left: SymbolicTensor, right: SymbolicTensor|scalar) -> SymbolicTensor`. `core` routes reflected calls (scalar on the left, `__radd__` etc.) through the SAME handler with the tensor as first argument.
- `neg`: `(x) -> SymbolicTensor`. `getitem`: `(x, key) -> SymbolicTensor`; `key` is a STATIC int/`builtins.slice`/tuple thereof → `slice`/`gather` ops. Symbolic indices, boolean masks, `None`/newaxis, ellipsis → `TraceError`. Strided slices → `gather`.
- Mapping: `add→elementwise.add`, `sub→subtract`, `mul→multiply`, `matmul→linalg.dot`, `truediv→divide`, `pow→power`, `neg→negate`, `lt/le/gt/ge/eq→comparison.*`, `getitem→indexing.getitem`.

## IR op definitions (ownership decision — binding)

The generic SSA machinery and the op **registry** live in `etl.ir`; the frontend **dialect** (which ops exist, their arities/attrs/effects) is declared in `_opdefs.py` here in `ops` — `ir` must not import `ops` (layering `core ← ir ← ops`). `_opdefs.register_all()` (stub) will register all entries into `ir`'s registry at ops import time during the implementation phase, once `etl/ir/CONTEXT.md` fixes its registration API. **Effect policy:** all frontend ops are `pure` except `runtime_call` (`callback`). Ops are functional SSA dataflow (`scatter`/`constant`/`pad` produce new values, no `write` effect); `write`/`read`/`collective` kinds are reserved for other layers (e.g. `dist`).

## Error semantics

`TraceError` (no active trace; concrete `Tensor` operand; symbolic index in `getitem`; both operands scalars), `ShapeError` (static broadcast/axis/rank/arity/permutation/padding failures), `DTypeError` (bad `cast` target; non-bool logicals; non-int bitwise; non-int gather indices), `TypeError` (unsupported operand kinds, malformed static params — Python programming errors). Static failures always raise at graph-construction time; symbolic conflicts defer to runtime where numpy enforces them (documented per op). Error messages include the call-site location whenever captured.

## API surface & routing table

| File | Area |
|---|---|
| `__init__.py` | Re-exports the 67 public names (`__all__`); import-time `_registration.register_operator_handlers()` |
| `_utils.py` | Internal: `check_in_trace`, `get_location`, `as_operand`, `weak_scalar_dtype`, `promote_dtypes`, `broadcast_shapes`, `reduced_shape`, `normalize_axes` + `ETL_DISABLE_LOCATIONS_ENV` |
| `elementwise.py` | `add subtract multiply divide power remainder maximum minimum abs negate square sqrt exp log log1p sin cos tan tanh sigmoid relu gelu erf sign cast bitwise_and bitwise_or bitwise_xor` |
| `comparison.py` | `equal not_equal less less_equal greater greater_equal logical_and logical_or logical_not select` |
| `indexing.py` | `broadcast reshape transpose slice gather scatter concatenate pad` (+ `getitem`, the operator-handler entry — not in `__all__`) |
| `reductions.py` | `reduce_sum reduce_max reduce_min reduce_mean reduce_prod sum max min mean prod argmax argmin` |
| `linalg.py` | `dot conv tril triu cumsum solve` |
| `constant.py` | `constant runtime_call stop_gradient` (+ `ETL_LARGE_CONSTANT_BYTES`, `constant_like`) |
| `_opdefs.py` | `OpDef` dataclass, `OPDEFS` dialect table, `register_all` (stub) |
| `_registration.py` | `OPERATOR_HANDLERS` mapping, `register_operator_handlers` (implemented) |

Cross-references: `../core/` (TraceError/ShapeError/DTypeError, SymbolicTensor, TensorSpec, register_operator_handlers hook), `../ir/` (Builder, Location, Value, op registry — sibling, read-only: expectations are listed below), `../trace/` (`current_builder` ONLY — never import anything else), `../../tests/ops/` (test suite — sibling, read-only; escalate test writes to root).

## Cross-module expectations (what ops REQUIRES — coordinate at implementation time)

- `core.register_operator_handlers(kind, handler)` (per-kind, idempotent); `SymbolicTensor` fields `value`/`dtype`/`shape`/`location`; `SymbolicTensor` dunders delegate to the registered hooks and reflected calls pass the tensor as first argument.
- `ir.Builder.create(name, operands=..., attrs=...)` validating against registered opdefs; `ir.Location` constructible with `file=`/`line=` + an `unknown()` sentinel; `ir` registry lookup `ir.opdef(name)`; an ir-side registration API for `_opdefs.register_all()`.
- `trace.current_builder()` returning `None` outside traces. `trace` must NEVER import `ops` (strict one-way dependency).

## Constraints

- Imports: `core`, `ir`, `trace` (current_builder only). NEVER `backends`, `numpy`(enp), `transforms`, `dist`, `pipeline`, `persist`.
- No eager computation of results; no numpy kernels here (they live ONLY in `backends/numpy`); no silent fallback or workaround for any error.
- Files < ~1000 lines; if a module grows, split along the op categories above.
- `slice` shadows the builtin inside `indexing.py` (use `builtins.slice`); `sum/max/min/abs` shadow builtins in their modules — intentional, documented.
- `constant` warns (UserWarning) above `ETL_LARGE_CONSTANT_BYTES` (default 1 MiB, env-tunable) and SNAPSHOTS (copies) data; `runtime_call` carries the callback as an op attr with effect `callback` — backends may reject it (`BackendError`) but never silently drop/reorder it; `stop_gradient` is effect `pure` but transform layers MUST process it before any constant-folding (gradient barrier).

## Test strategy

Mirror in `../../tests/ops/`: `test_elementwise.py`, `test_comparison.py`, `test_indexing.py`, `test_reductions.py`, `test_linalg.py`, `test_constant.py`, `test_opdefs.py`, `test_operator_handlers.py`, `test_utils.py`. Cover: scalar promotion incl. NEP-50 weak cases (int+float32→float32, float+float32→float64, float+complex64→complex64); TraceError for Tensor operands / no-trace / symbolic getitem; symbolic broadcasting (`DimExpr.max`, static int-vs-int ShapeError); per-op dtype rules; opdef table ↔ `__all__` 1:1 coverage; handler registration completeness; location capture + `ETL_DISABLE_LOCATIONS=1`; constant warning threshold; runtime_call effect annotation. CPU only.

## Notes for agents

- Stub bodies currently `raise NotImplementedError` — this node is Phase-1 architecture only; implementation is delegated to a Manager during Phase 2.
- Sibling modules (`core`/`ir`/`trace`) are also mid-architecture; `import etl.ops` fails until they land. Validate with `python3 -m py_compile etl/ops/*.py`.
- When implementing, keep docstrings authoritative — they ARE the per-op contract (dtype rule, shape rule, errors). Update `_opdefs.py` in the same change as any signature change.
- `constant_like` is internal (scalar-promotion helper); `getitem` is internal-but-registered; both intentionally excluded from `__all__`.
