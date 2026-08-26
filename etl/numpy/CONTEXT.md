# etl/numpy — `etl.numpy` (alias `enp`) NumPy-like graph namespace

## Intent

`etl.numpy` (`enp`) is a **pure sugar namespace** over `etl.ops`: NumPy-style function names that build **exactly the same IR** as the equivalent `etl.ops` calls (same op kinds, operands, attrs). It introduces **no new op kinds, no hidden semantics, no eager execution**. It is a graph-building namespace: calling `enp.*` with `SymbolicTensor`s inside `@etl.defn` builds ops into the active builder; calling with concrete `Tensor`s (or outside a trace) raises the same `TraceError` as `ops` — there is no eager numpy fallback and no numerical kernels here (kernels live ONLY in `../backends/`).

Binding parent contracts: root `../../CONTEXT.md` (principles 1–9, especially 4: Python value → Python semantics, SymbolicTensor → graph semantics; 9: concrete computation only via creators + reference backend) and `../CONTEXT.md` (public API + cross-module contracts; the "ops" bullet is the layer this namespace sits on). Import acyclicity: `numpy` layer may import `core`, `ir`, `ops` — never `trace`/`transforms`/`backends`/`pipeline`/`persist`/`dist` and never other frontends.

## API surface

Re-exported from `__init__.py` (exact names; all implemented as sugar over `etl.ops`):

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
| `clip(a, a_min, a_max)` | `ops.maximum(ops.minimum(a, a_max), a_min)` | `a_min=None`/`a_max=None` skip that side (numpy semantics); BOTH None → `ValueError` (numpy parity) |
| `astype(a, dtype)` | `ops.cast(a, dtype)` | numpy has no top-level `astype`; provided for enp ergonomics |
| `equal/not_equal/less/less_equal/greater/greater_equal(a, b)` | `ops.equal/not_equal/less/less_equal/greater/greater_equal(a, b)` | |
| `logical_and/or(a, b)`, `logical_not(a)` | `ops.logical_and/logical_or/logical_not` | |
| `where(cond, x, y)` | `ops.select(cond, x, y)` | numpy's single-arg index-array form is NOT provided in v1 |
| `reshape(a, shape)` | `ops.reshape(a, shape)` | one `-1` allowed (ops resolves it); numpy `order` kwarg unsupported |
| `transpose(a, axes=None)` | `ops.transpose(a, axes)` | `axes=None` → reversed axes (handled by ops) |
| `broadcast_to(a, shape)` | `ops.broadcast(a, shape)` | |
| `expand_dims(a, axis)` | `ops.reshape(a, shape_with_inserted_1s)` | int (negative normalized at trace time) or tuple (each entry normalized against the FINAL ndim = rank + len(tuple), sorted, inserted ascending; repeated axes → `ShapeError`) |
| `squeeze(a, axis=None)` | `ops.reshape(a, shape_without_1s)` | dropped dims must be statically size-1 (plain Python int `== 1`) at trace time; explicit axis on a symbolic/unknown dim → `TraceError`; `axis=None` keeps symbolic dims (only statically-1 dropped) |
| `concatenate(arrays, axis=0)` | `ops.concatenate(arrays, axis=axis)` | arrays: list/tuple of SymbolicTensors |
| `stack(arrays, axis=0)` | `expand_dims(each, axis)` + `concatenate(axis)` | |
| `split(a, indices_or_sections, axis=0)` | composition of `ops.slice` | int sections resolved statically at trace time (divisibility checked → `ShapeError`; symbolic axis dim → `TraceError`); list form requires strictly increasing ints within `[0, size]` |
| `pad(a, pad_width, mode="constant", constant_values=0)` | `ops.pad(a, config, value=constant_values)` | v1: only `mode="constant"`; other modes raise `NotImplementedError`. `pad_width` int (all-axes symmetric) / per-axis seq / length-1 `(before, after)` pair (broadcast) / bare `(before, after)` pair on a rank-1 tensor (pads axis 0, numpy parity) → ops `config` |
| `tril/triu(a, k=0)` | `ops.tril/triu(a, k=k)` | implemented — ops.linalg publishes them |
| `sum/mean/prod/max/min(a, axis=None, keepdims=False)` | `ops.sum/mean/prod/max/min(a, axes=axis, keepdims=keepdims)` | `axis=None` → `axes=None` = all axes (ops normalizes); `dtype≠None` (sum/mean/prod only) composes `ops.cast(result, dtype)` AFTER the reduction |
| `argmax/argmin(a, axis=None, keepdims=False)` | `ops.argmax/argmin(a, axis=axis, keepdims=keepdims)` | `axis` singular (ops signature) |
| `cumsum(a, axis=None, dtype=None)` | `ops.cumsum`; `axis=None` flattens first via `ops.reshape(a, (numel,))` (numel = DimExpr product of `a.shape`, scalar `()` → 1), then `axis=0` | implemented — ops.linalg publishes `cumsum(x, axis=0, reverse=False)`; `dtype≠None` composes `ops.cast` |
| `zeros/ones(shape, dtype=float32)` | Constant op (zeros/ones-filled numpy array) | graph ops — same op kind as `etl.constant`; large-constant warning applies; shape must be concrete ints (see design decision 9) |
| `full(shape, fill_value, dtype=None)` | Constant op | `dtype=None` → `np.result_type(fill_value)` at trace time |
| `empty(shape, dtype=float32)` | Constant op (uninitialized array) | values unspecified (numpy semantics); warns like any large constant |
| `arange(start, stop=None, step=1, dtype=None)` | Constant op with `numpy.arange(...)` data | concrete bounds/step required at trace time (symbolic → `TraceError`); `stop=None` → numpy single-arg semantics |
| `matmul(a, b)` / `dot(a, b)` | `ops.dot(a, b)` | v1: `dot` is an alias of `matmul`; numpy's 1-D `dot` vector semantics not special-cased |
| `linalg.solve(a, b)` | `ops.solve(a, b)` | implemented — ops.linalg publishes `solve` |

## Constraints

- **Imports**: only `etl.core`, `etl.ops` (+ `numpy` in `creation.py` for trace-time static materialization only). Never import `backends` (kernels) — no numerical implementation here.
- **No new op kinds**: if a mapping needs an op that doesn't exist, defer the enp function (documented) — do not build workarounds out of existing ops.
- **TraceError passthrough**: enp functions never catch errors; concrete-`Tensor` args and no-trace calls raise `TraceError` exactly like ops.
- **Files < ~1000 lines**; if a module grows, split along the routing-table boundaries.

## Design decisions

1. **Same-IR guarantee (defining property)**: a graph built with enp and one built with the mapped ops calls must produce identical op sequences (same op names, operands, attrs). `_map.py` holds the canonical 1:1 name table so implementation and tests share one source of truth; composed functions (`clip`, `stack`, `split`, `where`, `cumsum(axis=None)`) are documented as explicit op sequences above.
2. **Dtype defaults**: creation ops default `float32` (etl convention — deliberate deviation from numpy's `float64`). Arithmetic promotion is entirely `ops`' job; enp does no dtype logic of its own. `full(..., dtype=None)`/`arange(..., dtype=None)` use numpy's own inference over the static value/bounds at trace time.
3. **Reduction `axis=None`** means "all axes" (numpy semantics); forwarded as `axes=None` — ops' `normalize_axes` expands it (rank always known). enp's `dtype≠None` (sum/mean/prod/cumsum) is a documented post-reduction `ops.cast` composition, not an accumulator-dtype change.
4. **Creation ops embed full arrays as Constant ops** — the same op kind `etl.constant` builds, so snapshot semantics and the large-constant warning (`ETL_LARGE_CONSTANT_BYTES`, default 1 MiB) apply. `empty` snapshots an uninitialized numpy array (values unspecified — numpy semantics).
5. **`arange` requires concrete bounds** (Python ints/floats): bounds specialize the graph at trace time; symbolic bounds raise `TraceError` and are deferred (need dynamic-length iota/constant machinery in IR).
6. **`dot` ≡ `matmul` in v1** (both → `ops.dot`); numpy's 1-D inner-product semantics for `dot` are a documented deviation.
7. **`enp` alias** is registered at the `etl` package level (`etl/__init__.py`), not here — escalated to root.
8. **Deferrals are recorded, never worked around** — no enp function may implement deferred semantics via eager numpy or extra op compositions.
9. **Creation with symbolic shapes raises `TraceError`** (zeros/ones/full/empty with `Dim`/`DimExpr`/`None` dims; symbolic `arange` bounds): Constant ops embed concrete data, so dynamic-length constants are deferred to v2 (same family as symbolic `arange`). Message directs to concrete ints.
10. **Trace-time static failures raise `ShapeError`** (etl-native): split divisibility/index validation, expand_dims/stack/squeeze axis range, pad config validation, expand_dims negative-tuple entry normalization. Non-resolvable-at-trace-time cases (symbolic split/squeeze axis dim) raise `TraceError`.

## Known deviations from numpy (v1, intentional — fail loudly, never silently)

- `split` list indices must be strictly increasing and within `[0, size]` — numpy clips out-of-range indices and sorts; enp raises `ShapeError`.
- `pad` with a length-1 sequence of a bare int (e.g. `[2]` on rank 2) raises `ShapeError` — numpy pads only the first axis; use the int form (`pad(a, 2)`) for all-axes symmetric padding.
- `clip` with both bounds `None` raises `ValueError` (numpy parity).
- `cumsum(axis=None)` computes numel as a `DimExpr` product; with symbolic dims the flatten is symbolic (ops resolves at runtime).

## Deferrals (v1)

- `linspace` → v2 (same family as symbolic `arange`; needs IR machinery for dynamic-length constants).
- Symbolic-shape creation (`zeros/ones/full/empty` with `Dim`/`DimExpr`/`None` dims) and symbolic-bound `arange` → v2 (design decisions 5/9).
- `linalg.inv/norm/det` → v2 (need new IR ops — matrix inverse/norms/determinants — not present in the ops contract).
- `pad` modes other than `"constant"` (`edge/reflect/wrap/linear_ramp/...`) → v2 (need new IR ops; currently `NotImplementedError`).
- numpy `where` single-arg (index-array) form, ufunc kwargs (`out/where/dtype/...`), `reshape(..., order=)` → not provided in v1.
- `absolute` alias of `abs` → not provided in v1 (only `abs` per spec).

## Resolved contract conflicts (formerly blocking)

`ops.solve`, `ops.tril`, `ops.triu`, `ops.cumsum` are all published by `etl/ops/linalg.py` (frozen signatures: `tril(x, k=0)`, `triu(x, k=0)`, `cumsum(x, axis=0, reverse=False)`, `solve(a, b)`) and re-exported via `etl.ops.__all__`/`etl/__init__.py` — the corresponding enp functions are implemented. Confirmed frozen ops kwarg names: `ops.select(pred, on_true, on_false)`, `ops.concatenate(tensors, axis=)`, `ops.pad(x, config, value=)`, `ops.broadcast(x, shape)`, `ops.reduce_*(x, axes=, keepdims=)`, `ops.argmax/argmin(x, axis=, keepdims=)`, `ops.slice(x, start, lengths, strides=1)`, `ops.constant(core.Tensor)`. Two residual escalations to root: (1) the `ops` bullet in `../CONTEXT.md` does not explicitly name `conv/tril/triu/cumsum/solve`; (2) large creation constants warn by design — test authors should not treat the warning as a bug.

## Test strategy

pytest, CPU only, in sibling `../../tests/numpy/` (read-only from here — escalate test writes to root):
- Per-module unit tests mirroring `elementwise/logic/shape/reductions/creation/linalg`.
- **Equivalence tests (defining)**: build each enp function's graph and the documented ops-call graph from identical inputs; assert identical op names/operands/attrs (via IR inspection — `Graph.verify`, `pretty_print`). Composed functions assert the exact documented op sequence.
- Error tests: concrete `Tensor` args → `TraceError`; non-constant `pad` modes → `NotImplementedError`; symbolic `arange` bounds and symbolic-shape creation → `TraceError`; `squeeze`/`split` of symbolic dim → `TraceError`; split divisibility/pad config → `ShapeError`; `clip(None, None)` → `ValueError`.
- Creation: `float32` defaults, `full` dtype inference, large-constant warning behavior.

## Notes for agents

- **Validation**: enp behavior is validated end-to-end by the equivalence tests in `../../tests/numpy/` (sibling — read-only; implemented and passing) against the fully implemented `etl.ops`; `import etl` must stay clean. Note: running python with a script located in `/tmp` puts `/tmp` on `sys.path` (shadowing stdlib modules) — run via `python3 - < script` from the repo root instead.
- `core.DimExpr` equality is structural: `n * 2 != 2 * n` — build expected values in the same operand order as the implementation (`functools.reduce(operator.mul, shape, 1)` → `2 * n`).
- If a numpy function is missing here, check the deferral list first — it may be intentionally out of v1 scope.
- When adding a public name: update `__init__.py`, `__all__`, `_map.py` (if 1:1), and the mapping table above together.
- Do not import numpy in this package except for static value materialization (`creation.py`); all runtime computation stays in `ops`/`backends`.

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

## Status

Implementation complete: all 44 public functions build IR per the mapping table (validated via mock argument-flow tests against the frozen ops signatures). End-to-end IR equivalence and runtime checks land when `etl.ops` bodies are implemented (parallel track). Deferrals and numpy deviations are documented above.
