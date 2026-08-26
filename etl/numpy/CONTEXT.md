# etl/numpy — `etl.numpy` (alias `enp`) NumPy-like graph namespace

## Intent

`etl.numpy` (`enp`) is a **pure sugar namespace** over `etl.ops`: NumPy-style function names that build **exactly the same IR** as the equivalent `etl.ops` calls (same op kinds, operands, attrs). It introduces **no new op kinds, no hidden semantics, no eager execution**. It is a graph-building namespace: calling `enp.*` with `SymbolicTensor`s inside `@etl.defn` builds ops into the active builder; calling with concrete `Tensor`s (or outside a trace) raises the same `TraceError` as `ops` — there is no eager numpy fallback and no numerical kernels here (kernels live ONLY in `../backends/`).

Binding parent contracts: root `../../CONTEXT.md` (principles 1–9, especially 4: Python value → Python semantics, SymbolicTensor → graph semantics; 9: concrete computation only via creators + reference backend) and `../CONTEXT.md` (public API + cross-module contracts; the "ops" bullet is the layer this namespace sits on). Import acyclicity: `numpy` layer may import `core`, `ir`, `ops` — never `trace`/`transforms`/`backends`/`pipeline`/`persist`/`dist` and never other frontends.

## API surface

Re-exported from `__init__.py` (exact names; `shape.py` implemented, remaining modules land in Phase 2):

- **Elementwise/arith** (`elementwise.py`): `abs, add, subtract, multiply, divide, power, maximum, minimum, negative, square, sqrt, exp, log, sin, cos, tanh, sign, clip, astype`
- **Comparison/logic** (`logic.py`): `equal, not_equal, less, less_equal, greater, greater_equal, logical_and, logical_or, logical_not, where`
- **Shape/manipulation** (`shape.py`): `reshape, transpose, broadcast_to, expand_dims, squeeze, concatenate, stack, split, pad, tril, triu`
- **Reductions** (`reductions.py`): `sum, mean, prod, max, min, argmax, argmin, cumsum`
- **Creation — graph ops** (`creation.py`): `zeros, ones, full, empty, arange` (`linspace` deferred to v2)
- **Linear algebra**: `matmul`, `dot` re-exported top-level from `_linalg.py` (implementation home); public submodule `etl.numpy.linalg` (`linalg.py`) exposes `solve`

## Mapping table (enp → ops — the whole contract)

| enp function | ops composition | notes |
|---|---|---|
| `abs(x)` | `ops.abs(x)` | |
| `add/subtract/multiply/divide/power(a, b)` | `ops.add/subtract/multiply/divide/power(a, b)` | no numpy ufunc kwargs (`out/where/dtype/casting/order`) in v1 |
| `maximum/minimum(a, b)` | `ops.maximum/minimum(a, b)` | |
| `negative(x)` | `ops.negate(x)` | |
| `square/sqrt/exp/log/sin/cos/tanh/sign(x)` | `ops.square/sqrt/exp/log/sin/cos/tanh/sign(x)` | |
| `clip(a, a_min, a_max)` | `ops.maximum(ops.minimum(a, a_max), a_min)` | `a_min=None`/`a_max=None` skip that side (numpy semantics) |
| `astype(a, dtype)` | `ops.cast(a, dtype)` | numpy has no top-level `astype`; provided for enp ergonomics |
| `equal/not_equal/less/less_equal/greater/greater_equal(a, b)` | `ops.equal/not_equal/less/less_equal/greater/greater_equal(a, b)` | |
| `logical_and/or(a, b)`, `logical_not(a)` | `ops.logical_and/logical_or/logical_not` | |
| `where(cond, x, y)` | `ops.select(cond, x, y)` | numpy's single-arg index-array form is NOT provided in v1 |
| `reshape(a, shape)` | `ops.reshape(a, shape)` | one `-1` allowed, resolved at trace time via `DimExpr` arithmetic; numpy `order` kwarg unsupported |
| `transpose(a, axes=None)` | `ops.transpose(a, axes)` | `axes=None` → reversed axes (numpy default; rank known at trace time) |
| `broadcast_to(a, shape)` | `ops.broadcast(a, shape)` | |
| `expand_dims(a, axis)` | `ops.reshape(a, shape_with_inserted_1s)` | int (negative normalized) or tuple (repeated expansion) |
| `squeeze(a, axis=None)` | `ops.reshape(a, shape_without_1s)` | dropped dims must be statically size-1 at trace time; unknown-size dims → `TraceError` |
| `concatenate(arrays, axis=0)` | `ops.concatenate(arrays, axis=axis)` | arrays: list/tuple of SymbolicTensors |
| `stack(arrays, axis=0)` | `expand_dims(each, axis)` + `concatenate(axis)` | |
| `split(a, indices_or_sections, axis=0)` | composition of `ops.slice` | int sections resolved statically at trace time (divisibility checked) |
| `pad(a, pad_width, mode="constant", constant_values=0)` | `ops.pad(...)` | v1: only `mode="constant"`; other modes raise `NotImplementedError` |
| `tril/triu(a, k=0)` | `ops.tril/triu(a, k=k)` | |
| `sum/mean/prod/max/min(a, axis=None, keepdims=False)` | `ops.sum/mean/prod/max/min` (→ `reduce_*` ops) | `axis=None` → all axes (rank known at trace time); `dtype≠None` composes `ops.cast` (sum/mean/prod/cumsum) |
| `argmax/argmin(a, axis=None, keepdims=False)` | `ops.argmax/argmin` | |
| `cumsum(a, axis=None, dtype=None)` | `ops.cumsum`; `axis=None` flattens first via `reshape` to 1-D, then `axis=0` | |
| `zeros/ones(shape, dtype=float32)` | Constant op (zeros/ones-filled array) | graph ops — same op kind as `etl.constant`; large-constant warning applies |
| `full(shape, fill_value, dtype=None)` | Constant op | `dtype=None` → numpy dtype inference of static `fill_value` at trace time |
| `empty(shape, dtype=float32)` | Constant op (uninitialized array) | values unspecified (numpy semantics); warns like any large constant |
| `arange(start, stop=None, step=1, dtype=None)` | Constant op with `numpy.arange(...)` data | concrete bounds/step required at trace time; symbolic bounds deferred |
| `matmul(a, b)` / `dot(a, b)` | `ops.dot(a, b)` | v1: `dot` is an alias of `matmul`; numpy's 1-D `dot` vector semantics not special-cased |
| `linalg.solve(a, b)` | `ops.solve(a, b)` | |

## Constraints

- **Imports**: only `etl.core`, `etl.ops`, `etl.ir`. Never import `backends` (kernels) — no numerical implementation here.
- **No new op kinds**: if a mapping needs an op that doesn't exist, defer the enp function (documented) — do not build workarounds out of existing ops.
- **TraceError passthrough**: enp functions never catch errors; concrete-`Tensor` args raise `TraceError` exactly like ops.
- **Files < ~1000 lines**; if a module grows, split along the routing-table boundaries.
- **Architecture phase**: stub bodies `raise NotImplementedError` — algorithms land in Phase 2 via `subagent_manager`.

## Design decisions

1. **Same-IR guarantee (defining property)**: a graph built with enp and one built with the mapped ops calls must produce identical op sequences (same op names, operands, attrs). `_map.py` holds the canonical 1:1 name table so implementation and tests share one source of truth; composed functions (`clip`, `stack`, `split`, `where`, `cumsum(axis=None)`) are documented as explicit op sequences above.
2. **Dtype defaults**: creation ops default `float32` (etl convention — deliberate deviation from numpy's `float64`). Arithmetic promotion is entirely `ops`' job; enp does no dtype logic of its own. `full(..., dtype=None)`/`arange(..., dtype=None)` use numpy's own inference over the static value/bounds at trace time.
3. **Reduction `axis=None`** means "all axes" (numpy semantics); enp expands to explicit axes at trace time since rank is always known (root value model).
4. **Creation ops embed full arrays as Constant ops** — the same op kind `etl.constant` builds, so snapshot semantics and the large-constant warning (`ETL_LARGE_CONSTANT_BYTES`, default 1 MiB) apply. `empty` snapshots an uninitialized numpy array (values unspecified — numpy semantics).
5. **`arange` requires concrete bounds** (Python ints/floats): bounds specialize the graph at trace time; symbolic bounds raise `TraceError` and are deferred (need dynamic-length iota/constant machinery in IR).
6. **`dot` ≡ `matmul` in v1** (both → `ops.dot`); numpy's 1-D inner-product semantics for `dot` are a documented deviation.
7. **`enp` alias** is registered at the `etl` package level (`etl/__init__.py`), not here — escalated to root.
8. **Deferrals are recorded, never worked around** — no enp function may implement deferred semantics via eager numpy or extra op compositions.

## Deferrals (v1)

- `linspace` → v2 (same family as symbolic `arange`; needs IR machinery for dynamic-length constants — documented per design decision 5).
- `linalg.inv/norm/det` → v2 (need new IR ops — matrix inverse/norms/determinants — not present in the ops contract).
- `pad` modes other than `"constant"` (`edge/reflect/wrap/linear_ramp/...`) → v2 (need new IR ops; currently `NotImplementedError`).
- Symbolic-bound `arange` → v2 (see design decision 5).
- numpy `where` single-arg (index-array) form, ufunc kwargs (`out/where/dtype/...`), `reshape(..., order=)` → not provided in v1.
- `absolute` alias of `abs` → not provided in v1 (only `abs` per spec).

## Known contract conflicts (escalated to root — see parent report)

1. **Unspecified ops signatures** — implementation phase must confirm against the actual ops module: `ops.select` argument order (assumed `(cond, x, y)`), `ops.reduce_*` kwarg names (assumed `axis=`/`keepdims=`), and the **Constant-op construction path** creators should reuse (assumed: the same path `etl.constant` uses). enp mirrors whatever ops defines. (shape.py's ops signatures are confirmed frozen: `ops.reshape(x, shape)`, `ops.transpose(x, axes=None)`, `ops.broadcast(x, shape)`, `ops.concatenate(tensors, axis=0)`, `ops.pad(x, config, value=0)`, `ops.slice(x, start, lengths, strides=1)`, `ops.tril/triu(x, k=0)`.)
2. **Large creation constants warn by design** (creation embeds full arrays; `ETL_LARGE_CONSTANT_BYTES` applies). The root contract is silent on creators specifically — noted so test authors don't treat the warning as a bug.

## Test strategy

pytest, CPU only, in sibling `../../tests/numpy/` (read-only from here — escalate test writes to root):
- Per-module unit tests mirroring `elementwise/logic/shape/reductions/creation/linalg`.
- **Equivalence tests (defining)**: build each enp function's graph and the documented ops-call graph from identical inputs; assert identical op names/operands/attrs (via IR inspection — `Graph.verify`, `pretty_print`). Composed functions assert the exact documented op sequence.
- Error tests: concrete `Tensor` args → `TraceError`; deferred functions and non-constant `pad` modes → `NotImplementedError`; symbolic `arange` bounds → `TraceError`; `squeeze` of unknown-size dim → `TraceError`.
- Creation: `float32` defaults, `full` dtype inference, large-constant warning behavior.

## Routing table

| Path | Area |
|---|---|
| `./elementwise.py` | arithmetic/math sugar + `clip` + `astype` |
| `./logic.py` | comparisons, logical ops, `where` |
| `./shape.py` | reshape family, concatenate/stack/split/pad, tril/triu |
| `./reductions.py` | reduce family, argmax/argmin, cumsum |
| `./creation.py` | graph creation ops (zeros/ones/full/empty/arange) |
| `./_linalg.py` | `matmul`, `dot`, `solve` implementation home |
| `./linalg.py` | public `etl.numpy.linalg` submodule (v1: `solve`; v2: inv/norm/det) |
| `./_map.py` | canonical enp→ops name mapping (data only, import-safe) |
| `./__init__.py` | re-exports the full public surface + `linalg` submodule |

Siblings: `../ops/`, `../core/`, `../ir/` (read-only from here — ops-contract additions must be escalated to root); `../../tests/` test suite.

## Notes for agents

- If a numpy function is missing here, check the deferral list first — it may be intentionally out of v1 scope.
- When adding a public name: update `__init__.py`, `__all__`, `_map.py` (if 1:1), and the mapping table above together.
- Do not import numpy in this package for anything except static value materialization (`arange`/`full` dtype inference at trace time); all runtime computation stays in `ops`/`backends`.

## Status

Architecture phase complete (stubs raise `NotImplementedError`): CONTEXT.md, `_map.py`, 7 module skeletons + `__init__.py`, all `py_compile`-clean. Phase 2 fills bodies per the mapping table; `solve/tril/triu/cumsum` blocked on ops-contract additions (see conflicts).
