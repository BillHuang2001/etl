# tests — etl test suite

## Intent

pytest suite validating the `etl` package (sibling — see `../etl/CONTEXT.md` for the API contract being tested). Tests are the executable spec of design principles: explicitness, no hidden magic, local-tensor semantics, sugar transparency.

## Structure

Mirror the package: `tests/core/`, `tests/ir/`, `tests/ops/`, `tests/numpy/`, `tests/trace/`, `tests/block/`, `tests/transforms/`, `tests/backends/`, `tests/dist/`, `tests/persist/`, `tests/bench/` (`etl.bench` harness — see its own CONTEXT.md), plus:

- `tests/pipeline_test.py` — end-to-end staging pipeline (trace→lower→compile→load→run), bind, build, evaluate
- `tests/pipeline_env_defaults_test.py` — process-wide backend/device defaulting env vars (`ETL_BACKEND`/`ETL_DEVICE`/`ETL_TARGET_BACKENDS`) for `etl.build`/`etl.evaluate`: explicit kwargs win, env read lazily at call time, stub-backed full-path tests (registered `"iree"` stub; never imports a real adapter)
- `tests/test_pipeline_options.py` — per-backend option env vars (the options-override env half, `etl.pipeline_options.apply_env_options`/`ENV_OPTION_TABLE`): per-stage application (`ETL_IREE_COMPILE_ARGS` at compile, `ETL_IREE_RUNTIME_ARGS` at load), explicit > env > default precedence, empty/whitespace = unset, lazy per-call reads, tvm target + pass-configs JSON, xla base64 decode, malformed values → `BackendError` naming var+value, unknown backend no-op; stub full-path tests (compile args reach compile, runtime args reach load, run options forwarded, numpy documented-ignore, `BoundExecutable.backend`); no real adapter imports — the backend half is `tests/backends/test_options_override.py`
- External-kernel suites (the `etl.external_call` custom-kernel path): `ops/test_external_call.py` (registry semantics incl. the device-resident registration contract — `device_resident=True` requires an explicit backend, non-bool → TypeError, `get_external_kernel_entry` `(callable, mode)` tuple semantics with default-slot fallback, re-registration overwrites callable AND mode, `ExternalKernel.impl` direct/decorator device forms, device registration under non-iree backend strings allowed; op building, dispatch, validation errors, TransformError, cond/while bodies, save/load, stablehlo deferral) + `ops/test_external_rules.py` (per-backend registry + default-slot fallback, `ExternalKernel` handle, portable validation, fallback-rule installation, grad/vjp/vmap through portables, unregister semantics), `backends/test_external_call_iree.py` (split + host-dispatch, per-backend slot resolution at run, once-per-executable staging warning; device-resident dispatch round — device tensors in/out via the public `IreeDevicePayload`, second-vmfb-on-the-operand's-HAL-device interop, numpy/mixed/`core.Tensor`/duck-payload returns, metadata-only validation with canonical errors, fully-device-resident boundaries emit zero staging warnings, cuda GPU-guarded bit-exactness + operand residency) + `backends/test_external_portable_fallback.py` (tvm-gated: portable-inline fallback with warning, no-portable BackendError, result-spec validation), `block/test_portable_vjp_regression.py` (fixed post-inline-seeding vjp fallback incl. constant-in-portable)
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
| `./ops/` | per-op shape/dtype/error tests (incl. `test_random.py` — the `etl.random` key-based RNG suite: cross-run determinism, split/split_n semantics, distribution sanity, symbolic operands, dtype rules, explicitness, while_loop key threading, validation errors, TransformError/vmap, compiler-backend deferrals) |
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

Tree/pytree UX coverage (validated green against the tree-UX implementation in `../etl/`, no skips/xfails):

- `core/test_tree_utils.py` — `tree_map` (single/multi-tree, empty containers, leaf-type-changing fns, multi-tree mismatch → `TypeError` with first-mismatch pytree path), `tree_leaves`/`tree_structure`/`tree_flatten`/`tree_unflatten` incl. alias identity with `flatten`/`unflatten`.
- `core/test_tree.py` — `defaultdict`/`Counter` roundtrips (factory list/None, nested); structured errors: unpersistable lambda factory, mixed-type dict keys, dataclass InitVar/`init=False` rebuild; `register_pytree_node(object, ...)` → `TypeError` (with try/finally registry cleanup so a missing guard can't hijack the MRO dispatch for the session); plain user class stays a leaf; etl value types (`Device`/`Dim`/`TensorSpec`) flatten to exactly 1 leaf.
- `test_spec_compliance.py` — `tree_map(f, t) == unflatten([f(l) for l in flatten(t)[0]], flatten(t)[1])` composition identity; exact structure preservation.
- `pipeline_test.py` + `trace/test_graph.py` — run/bind/validate_inputs structure-mismatch errors include `first mismatch at pytree path {path}` (old lead-in preserved).
- `trace/test_static_snapshot.py` — `Device`/`Dim` static args snapshot as ONE static value (no field descent); user dataclasses still descend.
- `block/test_portable.py` — portable impls returning namedtuple/dataclass structures of symbolics (incl. vmap through them).

## Notes for agents

- Prefer small focused test files mirroring the module they test; test files may import from `etl` directly.
- When a backend/op behavior is under-specified, encode the intended semantics here and flag it.
- **Package layout:** every test dir (and `tests/` itself) has an `__init__.py`, so pytest imports modules as `tests.<area>.<module>`. Directory-local helpers must be imported package-qualified (`from tests.ops.conftest import ...`, `from tests.numpy._ir_utils import ...`, `from tests.transforms._fd_utils import ...`) — bare `from conftest import ...` breaks collection. Do NOT remove the `__init__.py` files: besides module-name collisions (`test_errors.py` exists in 4 dirs), removing them makes the `tests/numpy` package shadow the real numpy module.
- **`tests/dist` collection:** pytest's default `norecursedirs` includes `dist`, so the root `pyproject.toml` `[tool.pytest.ini_options]` explicitly sets `norecursedirs` WITHOUT `dist` (see the NOTE comment there). Plain `python3 -m pytest` therefore collects `tests/dist/` — never re-add `dist` to `norecursedirs`.
- **BUG(etl) protocol:** tests asserting the documented contract that the implementation currently violates are kept failing with a `# BUG(etl): <description>` comment (and a minimal repro). Do NOT weaken them. After an etl fix, delete the marker comment and the corresponding `Known Issues` entry here and in the area's CONTEXT.md.
- **Inventory (HEAD bdaf883, `python3 -m pytest --collect-only -q tests/`):** 5914 tests collected across 119 files — backends 817, bench 138, block 127, core 498, dist 242, ir 850, numpy 273, ops 1837, persist 156, pipeline_test 19, pipeline_env_defaults_test 36, test_pipeline_options 28, sparse 358, test_spec_compliance 79, trace 184, transforms 272. `backends/test_adapter_xla.py` (10) skips at collection without a user-provided PJRT plugin. Run-time skips in the default env (torch absent; iree/tvm/jaxlib/gcc present): 7 torch-interop (6 `core/test_tensor.py::TestTorchInterop` + 1 `test_spec_compliance.py`), 4 `etl.bench` torch-present (`test_torch_repeat_calls.py` ×3 + `test_conformance.py` ×1), 6 xla cases of `backends/test_mixed_dtype.py` → 17. With torch installed the 7+4 run and the 2 torch-absent inverse `etl.bench` tests skip instead; with a PJRT plugin the 16 xla tests run.
- Plain `python3 -m pytest` is fully green; any failure is a regression in either etl or the tests.
