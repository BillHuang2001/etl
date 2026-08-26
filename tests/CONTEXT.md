# tests — etl test suite

## Intent

pytest suite validating the `etl` package (sibling — see `../etl/CONTEXT.md` for the API contract being tested). Tests are the executable spec of design principles: explicitness, no hidden magic, local-tensor semantics, sugar transparency.

## Structure

Mirror the package: `tests/core/`, `tests/ir/`, `tests/ops/`, `tests/numpy/`, `tests/trace/`, `tests/block/`, `tests/transforms/`, `tests/backends/`, `tests/dist/`, `tests/persist/`, plus:

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
| `./backends/` | backend interface + stablehlo export tests |
| `./dist/` | collective semantics tests |
| `./persist/` | cache + container tests |

## Notes for agents

- Prefer small focused test files mirroring the module they test; test files may import from `etl` directly.
- When a backend/op behavior is under-specified, encode the intended semantics here and flag it.
- **Package layout:** every test dir (and `tests/` itself) has an `__init__.py`, so pytest imports modules as `tests.<area>.<module>`. Directory-local helpers must be imported package-qualified (`from tests.ops.conftest import ...`, `from tests.numpy._ir_utils import ...`, `from tests.transforms._fd_utils import ...`) — bare `from conftest import ...` breaks collection. Do NOT remove the `__init__.py` files: besides module-name collisions (`test_errors.py` exists in 4 dirs), removing them makes the `tests/numpy` package shadow the real numpy module.
- **`tests/dist` collection:** pytest's default `norecursedirs` includes `dist`, so the root `pyproject.toml` `[tool.pytest.ini_options]` explicitly sets `norecursedirs` WITHOUT `dist` (see the NOTE comment there). Plain `python3 -m pytest` therefore collects `tests/dist/` — never re-add `dist` to `norecursedirs`.
- **BUG(etl) protocol:** tests asserting the documented contract that the implementation currently violates are kept failing with a `# BUG(etl): <description>` comment (and a minimal repro). Do NOT weaken them. After an etl fix, delete the marker comment and the corresponding `Known Issues` entry here and in the area's CONTEXT.md.
- Plain `python3 -m pytest` is currently fully green (torch-interop tests skip when torch is absent); any failure is a regression in either etl or the tests.
