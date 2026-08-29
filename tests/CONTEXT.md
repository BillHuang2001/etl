# tests — etl test suite

## Intent

pytest suite validating the `etl` package (sibling — see `../etl/CONTEXT.md` for the API contract being tested). Tests are the executable spec of design principles: explicitness, no hidden magic, local-tensor semantics, sugar transparency.

## Structure

Mirror the package: `tests/core/`, `tests/ir/`, `tests/ops/`, `tests/numpy/`, `tests/trace/`, `tests/block/`, `tests/transforms/`, `tests/backends/`, `tests/dist/`, `tests/persist/`, `tests/bench/` (`etl.bench` harness — see its own CONTEXT.md), plus:

- `tests/pipeline_test.py` — end-to-end staging pipeline (trace→lower→compile→load→run), bind, build, evaluate
- `tests/test_spec_compliance.py` — design-principle compliance:
  - no implicit tracing/eager mode: ops outside a trace → TraceError; direct `Defn` call raises helpfully
  - closure-captured Tensor in ops → TraceError; `etl.constant` opt-in works (warns above `ETL_LARGE_CONSTANT_BYTES`)
  - `SymbolicTensor` has no `.numpy()` / `__dlpack__`; `bool(symbolic)` → TraceError
  - `etl.build`/`evaluate` = documented shorthand (same result as explicit pipeline); stage types distinct, wrong stage → TypeError
  - `bind` never alters graph/compile; missing/wrong-named binding fails
  - `vmap(f, in_axes, out_axes)` ≡ `vectorize` on the traced function (same IR up to naming)
  - concrete creators (`etl.zeros`) return Tensors with DLPack (+ torch interop via importorskip); `enp.zeros` inside defn produces a graph op
  - serialization round-trips for Graph/LoweredProgram/CompiledArtifact; corrupt file fails
  - Python values → Python semantics (static specialization; `if etl.sum(x) > 0` fails at trace)
  - collectives (`etl.dist.*`) are explicit IR ops only — no implicit insertion
  - error messages include source locations (`file.py:line`)

## Constraints

- CPU only; numpy backend is the reference for correctness.
- torch-dependent tests use `pytest.importorskip("torch")` (interop is optional).
- No GPU usage. No network. Keep tests fast (<2s per file, no large shapes beyond ~256×256 unless needed).

## Routing table

| Path | Area |
|---|---|
| `./core/` | value-model tests |
| `./ir/` | SSA/verify/serialize tests |
| `./ops/` | per-op shape/dtype/error tests |
| `./numpy/` | enp sugar tests |
| `./trace/` | defn/trace/control-flow tests |
| `./block/` | custom op tests |
| `./transforms/` | vmap/grad/jvp/vjp tests |
| `./backends/` | backend interface + stablehlo export tests + compiler-framework & IREE/XLA/TVM adapter tests |
| `./dist/` | collective semantics tests |
| `./sparse/` | sparse-tensor suite (etl.sparse): value model / ops / errors / transforms / control flow / pipeline / backends / deferrals — see `./sparse/CONTEXT.md` |
| `./persist/` | cache + container tests |
| `./bench/` | `etl.bench` harness tests (importability w/o torch, conformance vs numpy/torch refs, benchmark timing, CLI exit codes) — torch-optionality pattern: torch-present tests use `pytest.importorskip("torch")`; torch-absent-only tests guard with `importlib.util.find_spec("torch") is None` and `pytest.mark.skipif` inverted |

## Test strategy

Coverage additions (tree/pytree UX contract — written against the pinned contract of the parallel tree-UX implementation in `../etl/`; until that branch merges, the new tests fail as expected, with no skips/xfails):

- `core/test_tree_utils.py` — `tree_map` (single/multi-tree, empty containers, leaf-type-changing fns, multi-tree mismatch → `TypeError` with first-mismatch pytree path), `tree_leaves`/`tree_structure`/`tree_flatten`/`tree_unflatten` incl. alias identity with `flatten`/`unflatten`.
- `core/test_tree.py` — `defaultdict`/`Counter` roundtrips (factory list/None, nested); structured errors: unpersistable lambda factory, mixed-type dict keys, dataclass InitVar/`init=False` rebuild; `register_pytree_node(object, ...)` → `TypeError` (with try/finally registry cleanup so a missing guard can't hijack the MRO dispatch for the session); plain user class stays a leaf; etl value types (`Device`/`Dim`/`TensorSpec`) flatten to exactly 1 leaf.
- `test_spec_compliance.py` — `tree_map(f, t) == unflatten([f(l) for l in flatten(t)[0]], flatten(t)[1])` composition identity; exact structure preservation.
- `pipeline_test.py` + `trace/test_graph.py` — run/bind/validate_inputs structure-mismatch errors include `first mismatch at pytree path {path}` (old lead-in preserved).
- `trace/test_static_snapshot.py` — `Device`/`Dim` static args snapshot as ONE static value (no field descent); user dataclasses still descend.
- `block/test_portable.py` — portable impls returning namedtuple/dataclass structures of symbolics (incl. vmap through them).

Open coordination points until the parallel implementation lands (see the tree-UX work): path granularity for dict-key mismatches (key-level `[1]['w']` vs node-level `[1]` — tests assert both in different files); bind's unbound-portion lead-in wording vs the pinned unified shape (existing `pipeline_test.py:407` asserts the old wording).

## Notes for agents

- Prefer small focused test files mirroring the module they test; test files may import from `etl` directly.
- When a backend/op behavior is under-specified, encode the intended semantics here and flag it.
- **Package layout:** every test dir (and `tests/` itself) has an `__init__.py`, so pytest imports modules as `tests.<area>.<module>`. Directory-local helpers must be imported package-qualified (`from tests.ops.conftest import ...`, `from tests.numpy._ir_utils import ...`, `from tests.transforms._fd_utils import ...`) — bare `from conftest import ...` breaks collection. Do NOT remove the `__init__.py` files: besides module-name collisions (`test_errors.py` exists in 4 dirs), removing them makes the `tests/numpy` package shadow the real numpy module.
- **`tests/dist` collection:** pytest's default `norecursedirs` includes `dist`, so the root `pyproject.toml` `[tool.pytest.ini_options]` explicitly sets `norecursedirs` WITHOUT `dist` (see the NOTE comment there). Plain `python3 -m pytest` therefore collects `tests/dist/` — never re-add `dist` to `norecursedirs`.
- **BUG(etl) protocol:** tests asserting the documented contract that the implementation currently violates are kept failing with a `# BUG(etl): <description>` comment (and a minimal repro). Do NOT weaken them. After an etl fix, delete the marker comment and the corresponding `Known Issues` entry here and in the area's CONTEXT.md.
- Plain `python3 -m pytest` is green EXCEPT the open BUG(etl) markers listed under Known Issues (deliberately kept failing per protocol); any other failure is a regression in either etl or the tests.

## Known Issues (open BUG(etl) markers — kept failing per the BUG(etl) protocol; after an etl fix, delete the marker comment AND the corresponding entry here and in the area's CONTEXT.md)

- **Concrete COO constructor is not values-aware for stored-zero duplicates** — `tests/sparse/test_value.py::test_stored_zero_duplicate_row_constructs`: the contract (`etl/sparse/CONTEXT.md`: "duplicate NONZERO rows error; duplicate rows with one stored zero are legal") promises the concrete constructor accepts a duplicate row whose pair includes a stored zero; `etl/sparse/value.py:254` rejects ALL duplicate rows ("COO indices must be unique"). The runtime kernels ARE values-aware (pinned passing in `tests/sparse/test_errors.py::test_runtime_stored_zero_duplicate_passes`).
- **vmap callable path fails on sparse registered-node args** — `tests/test_spec_compliance.py::TestSparseExplicitness::test_vmap_callable_bare_axes_on_sparse`: `etl/transforms/vmap.py::_derive_unvectorized_args` strips `shape[1:]` from every mapped tensor leaf when deriving unvectorized specs, which for a sparse node removes the runtime-dynamic nnz dim; `etl.vmap(f, in_axes=0)(SparseTensorSpec(...))` raises ShapeError ("COO indices spec must have shape (None, 2), got (2,)") although transforms CONTEXT.md documents callable-path args for registered pytree nodes. The graph-level path (`etl.vmap(graph, in_axes=0)` / `etl.vectorize`) works and is pinned passing.

