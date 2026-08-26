# tests/numpy — `etl.numpy` (enp) test suite

## Intent

pytest suite validating the `etl.numpy` (alias `enp`) sugar namespace against its contract in `../../etl/numpy/CONTEXT.md` (sibling — read-only from here; escalate any enp/ops fix to root). The defining property under test: **every enp function builds exactly the same IR as the mapped `etl.ops` calls** (same op kinds, operands, attrs, result types).

## Structure

| File | Area |
|---|---|
| `test_namespace.py` | enp alias/module identity, exact `__all__` surface (56 names), AttributeError for deferred names |
| `test_equivalence.py` | THE defining property: ~80 parametrized enp≡ops IR-equality cases across elementwise/logic/reductions/shape/creation/linalg (normalized `pretty_print` text comparison) |
| `test_composition.py` | composed-function semantics: clip/stack/split/expand_dims/squeeze/where/pad — numeric vs numpy + IR structure + error paths |
| `test_creation.py` | creation ops are GRAPH Constant ops (never concrete): SymbolicTensor in-trace, numerics vs numpy, float32 defaults, symbolic-shape TraceError |
| `test_deferrals.py` | sugar discipline: concrete-Tensor TraceError, pad-mode NotImplementedError, AttributeError for absent names, no ufunc kwargs |
| `test_linalg_ns.py` | `enp.matmul`/`enp.dot`/`enp.linalg.solve` IR equivalence, numerics vs numpy, singular-solve raises, 1-D dot deviation |
| `_ir_utils.py` | shared `normalize_ir(text)` helper (see Notes for agents) |

## Test strategy

- **Equivalence method**: trace an enp-based defn and an ops-composition defn from identical `TensorSpec`s/static values; `graph.verify()` both; assert normalized `etl.ir.pretty_print` texts are character-identical (pytest shows the diff on failure). Python scalar args become `etl.constant` ops with `ndarray<dtype[shape]>` attr summaries, so identical args print identically.
- Numeric checks go through `etl.evaluate` (all args must be tensors/ndarrays — a Python scalar arg raises TypeError; pass runtime bounds as 0-d arrays or capture them as static closure values) and compare against numpy references.
- Constraints: small shapes (≤ ~16 elements), CPU only, <2s per file, parametrize with ids, `pytest.raises(..., match=...)`.

## Known Issues

**Three real etl bugs (contract violations, all in `etl/numpy/`, kept as deliberately-failing tests marked `# BUG(etl):` — do NOT fix in etl from here, do NOT weaken/xfail/skip the tests; fixes must be escalated to root):**

1. `enp.clip` None-bound branches are inverted (`etl/numpy/elementwise.py`): `clip(a, None, hi)` builds `ops.maximum(a, hi)` but the contract ("None skips that side", numpy parity) requires `minimum(a, hi)`; `clip(a, lo, None)` builds `minimum(a, lo)` but requires `maximum(a, lo)`. Two-bound form is correct. Failing: `test_clip_upper_bound_only_is_minimum`, `test_clip_lower_bound_only_is_maximum`.
2. `enp.expand_dims` validates tuple axes against the ORIGINAL rank (`etl/numpy/shape.py`), but numpy normalizes each tuple entry against the FINAL ndim (rank + len(tuple)): `enp.expand_dims(rank-2, (1, 3))` raises ShapeError while `np.expand_dims` returns shape `(2, 1, 2, 1)`. Failing: `test_expand_dims_tuple_axis_ascending_numeric`.
3. `enp.pad` rejects a bare `(before, after)` pair on a rank-1 array (`enp.pad(a, (1, 2))` → ShapeError; numpy pads 1 before / 2 after the sole axis). Only the length-1 *sequence* form (`((1, 2),)`) broadcasts. Failing: `test_pad_pair_rank1_numeric`.

The suite therefore exits non-zero with exactly these 4 known failures (269 passing) until the bugs are fixed in etl — after fixing each, delete the `# BUG(etl)` comment and this Known Issues entry.

## Notes for agents

- `from tests.numpy._ir_utils import normalize_ir` — test dirs are packages (have `__init__.py`), so helper imports must be package-qualified.
- `normalize_ir` strips per-op ` loc("file":line:col)` tokens: enp callsites live in `etl/numpy/*.py` while ops-composed defns live in the test file, so raw printed IR never matches without this normalization. The entry function is always named `main`.
- `ops.dot` requires rank ≥ 2 (batched matmul) — 1-D `enp.dot`/`enp.matmul` raise ShapeError; that is the documented v1 deviation, not a bug.
- Creation ops emit large-constant warnings by design (`ETL_LARGE_CONSTANT_BYTES`, default 1 MiB) — tests use small shapes and treat warnings as expected, not bugs.
- When a name seems missing from enp, check the deferral list in `../../etl/numpy/CONTEXT.md` before writing an AttributeError test.
