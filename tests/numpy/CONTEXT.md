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

## Notes for agents

- `from tests.numpy._ir_utils import normalize_ir` — test dirs are packages (have `__init__.py`), so helper imports must be package-qualified.
- `normalize_ir` strips per-op ` loc("file":line:col)` tokens: enp callsites live in `etl/numpy/*.py` while ops-composed defns live in the test file, so raw printed IR never matches without this normalization. The entry function is always named `main`.
- `ops.dot` requires rank ≥ 2 (batched matmul) — 1-D `enp.dot`/`enp.matmul` raise ShapeError; that is the documented v1 deviation, not a bug.
- Creation ops emit large-constant warnings by design (`ETL_LARGE_CONSTANT_BYTES`, default 1 MiB) — tests use small shapes and treat warnings as expected, not bugs.
- When a name seems missing from enp, check the deferral list in `../../etl/numpy/CONTEXT.md` before writing an AttributeError test.
