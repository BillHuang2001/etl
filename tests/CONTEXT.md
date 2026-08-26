# tests — etl test suite

## Intent

pytest suite validating the `etl` package (sibling — see `../etl/CONTEXT.md` for the API contract being tested). Tests are the executable spec of design principles: explicitness, no hidden magic, local-tensor semantics, sugar transparency.

## Structure

Mirror the package: `tests/core/`, `tests/ir/`, `tests/ops/`, `tests/numpy/`, `tests/trace/`, `tests/block/`, `tests/transforms/`, `tests/backends/`, `tests/dist/`, `tests/persist/`, plus:

- `tests/pipeline_test.py` — end-to-end staging pipeline (trace→lower→compile→load→run), bind, build, evaluate
- `tests/test_spec_compliance.py` — design-principle compliance:
  - closure-captured Tensor in ops → TraceError; `etl.constant` opt-in works (with warning)
  - `SymbolicTensor` has no `.numpy()` / `__dlpack__`
  - `etl.build`/`evaluate` = documented shorthand (same result as explicit pipeline)
  - `bind` never alters graph/compile; missing/wrong-named binding fails
  - `vmap(f, in_axes, out_axes)` ≡ `vectorize` on the traced function (same IR up to naming)
  - direct `Defn` call with concrete tensors raises with helpful message
  - serialization round-trips for Graph/LoweredProgram/CompiledArtifact; corrupt file fails
  - concrete creators (`etl.zeros`) return Tensors with DLPack; `enp.zeros` inside defn produces a graph op

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
