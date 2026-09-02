# tests — etl test suite

## Intent

pytest suite validating the `etl` package (sibling — see `../etl/CONTEXT.md` for the API contract being tested). Tests are the executable spec of design principles: explicitness, no hidden magic, local-tensor semantics, sugar transparency.

## Structure

Mirror the package: `tests/core/`, `tests/ir/`, `tests/ops/`, `tests/numpy/`, `tests/trace/`, `tests/block/`, `tests/transforms/`, `tests/backends/`, `tests/dist/`, `tests/persist/`, `tests/bench/` (`etl.bench` harness — see its own CONTEXT.md), plus:

- `tests/pipeline_test.py` — end-to-end staging pipeline (trace→lower→compile→load→run), bind, build, evaluate; run/bind **boundary device enforcement** (explicit placement, R5): every input tensor must already live on the executable's device — a cuda-kind tensor (fake duck-typed payload, no GPU) fed to the cpu executable raises `DeviceError` naming the input pytree path (`input at path [0]['b'] is on device Device(kind='cuda', index=0), but the executable runs on device ...` — nested-path variant included) with the `t.to(...)` remedy; raw host ndarrays auto-wrap as cpu:0 and run fine, and the explicit `t.to(cpu)` transfer unblocks the foreign tensor (positive control)
- `tests/pipeline_env_defaults_test.py` — process-wide backend/device defaulting env vars (`ETL_BACKEND`/`ETL_DEVICE`/`ETL_TARGET_BACKENDS`) for `etl.build`/`etl.evaluate`: explicit kwargs win, env read lazily at call time, stub-backed full-path tests (registered `"iree"` stub; never imports a real adapter)
- `tests/test_pipeline_options.py` — per-backend option env vars (the options-override env half, `etl.pipeline_options.apply_env_options`/`ENV_OPTION_TABLE`): per-stage application (`ETL_IREE_COMPILE_ARGS` at compile, `ETL_IREE_RUNTIME_ARGS` at load), explicit > env > default precedence, empty/whitespace = unset, lazy per-call reads, tvm target + pass-configs JSON, xla base64 decode, malformed values → `BackendError` naming var+value, unknown backend no-op, **`ETL_OPT_LEVEL`** (the first etl-defined+etl-validated option — applied at the compile stage of every compiler backend normalized to int 0..3, explicit kwarg wins, malformed → `BackendError` naming the variable and value, blank = unset, numpy no-op, stage-scoped; full-path through the registered iree stub: env value reaches compile as `opt_level=2`, explicit kwarg beats env); stub full-path tests (compile args reach compile, runtime args reach load, run options forwarded, numpy documented-ignore, `BoundExecutable.backend`); no real adapter imports — the backend half is `tests/backends/test_options_override.py` (+ the 53-test `tests/backends/test_opt_level.py` port)
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
  - explicit device placement (`TestExplicitPlacement`, group 15, 7 tests): placement is explicit — host memory cannot be relabeled (`Tensor`/concrete-creator `device=` non-cpu → `DeviceError`), `.numpy()` on a non-cpu-kind payload raises (no implicit device-to-host transfer), `Tensor.to` is the only transfer API (same-device → self; to-cpu materializes a fresh host copy), and the run/bind/`etl.constant` boundaries reject foreign-device tensors with the explicit-transfer remedy
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

Explicit device placement coverage (validated green; no GPU needed for the core pins — duck-typed fake payloads stand in for device-resident buffers):

- `core/test_tensor.py` — kind-aware `.numpy()` (cpu-kind payload → lazy FRESH host copy per call, metadata-only `__eq__`; non-cpu-kind → `DeviceError`), per-kind `__dlpack__`/`__dlpack_device__` errors, the payload relabel ban (`device=` conflicting with `payload.device` → `DeviceError`), the host-data gates (`Tensor(ndarray, device=<non-cpu>)` and non-cpu creator `device=` → `DeviceError`), and the `Tensor.to` API (`TestTensorTo`, fake payloads): same-device → self, payload→cpu:0 = fresh host copy per call, host→non-cpu dispatches to `register_device_transfer_provider` (test-registered kinds only — never "cuda", which has an iree thunk), no provider → `DeviceError`, payload→different non-cpu device → explicit two-hop error.
- `core/test_device.py` — `split_tensor`/`replicate_tensor` host-data-only pins: ndarray-backed cpu:0 source, every target `Device('cpu', 0)` (cuda / cpu-index≠0 targets → canonical `DeviceError` with the `t.to(` remedy), device-payload sources → `DeviceError` never `AttributeError`; host-side semantics unchanged (views / one shared buffer across the all-cpu:0 targets).
- `pipeline_test.py` — run/bind boundary (R5) device checks: foreign-device input → `DeviceError` naming the input pytree path (incl. the nested `{'a': cpu, 'b': cuda}` first-mismatch variant), raw ndarray inputs accepted on the cpu executable, the explicit `t.to(cpu)` unblocks the cuda-kind tensor.
- `test_spec_compliance.py` — `TestExplicitPlacement` (group 15): the placement rules as design-principle compliance statements (see the Structure section).
- `ops/test_constant.py` — `etl.constant` host-data gate: a non-cpu tensor raises `DeviceError` ("requires host data" + remedy); cpu-kind PAYLOAD tensors embed fine; the explicit `to(cpu)`-then-`constant` preparation path works.
- `backends/test_iree_llvm_cpu_host_inputs.py` (6 tests) — the UNCHANGED llvm-cpu host-input path under explicit placement (no GPU): raw host ndarray inputs still accepted at the run boundary (uploaded via `iree.runtime.asdevicearray`, one call per input leaf, counting-wrapper pinned), cpu:0 `Tensor` inputs accepted (same-device), run outputs are payload-backed cpu:0 tensors with lazy per-call `.numpy()` (zero `IreeDevicePayload.to_host` after `run()`), `Tensor.to` same-device → self, and even llvm-cpu executables REJECT a foreign-device tensor at the run boundary.
- The 6 iree-cuda backend suites (see `./backends/CONTEXT.md` — refreshed this round) place every run input via `Tensor.to(cuda_device)` BEFORE `etl.run` and read outputs through the explicit `to(cpu).numpy()` hop: plain cuda executables never stage host inputs (defensive `DeviceError`) and the same-device loop has ZERO `asdevicearray` calls from the very FIRST call.

## Notes for agents

- Prefer small focused test files mirroring the module they test; test files may import from `etl` directly.
- When a backend/op behavior is under-specified, encode the intended semantics here and flag it.
- **Package layout:** every test dir (and `tests/` itself) has an `__init__.py`, so pytest imports modules as `tests.<area>.<module>`. Directory-local helpers must be imported package-qualified (`from tests.ops.conftest import ...`, `from tests.numpy._ir_utils import ...`, `from tests.transforms._fd_utils import ...`) — bare `from conftest import ...` breaks collection. Do NOT remove the `__init__.py` files: besides module-name collisions (`test_errors.py` exists in 4 dirs), removing them makes the `tests/numpy` package shadow the real numpy module.
- **`tests/dist` collection:** pytest's default `norecursedirs` includes `dist`, so the root `pyproject.toml` `[tool.pytest.ini_options]` explicitly sets `norecursedirs` WITHOUT `dist` (see the NOTE comment there). Plain `python3 -m pytest` therefore collects `tests/dist/` — never re-add `dist` to `norecursedirs`.
- **BUG(etl) protocol:** tests asserting the documented contract that the implementation currently violates are kept failing with a `# BUG(etl): <description>` comment (and a minimal repro). Do NOT weaken them. After an etl fix, delete the marker comment and the corresponding `Known Issues` entry here and in the area's CONTEXT.md.
- **Inventory (current HEAD, `python3 -m pytest --collect-only -q tests/`):** 6084 tests collected across 126 files — backends 904 (31 files, incl. the new `test_iree_llvm_cpu_host_inputs.py` 6), bench 151, block 127, core 525 (9 files — `test_tensor.py` 93, `test_device.py` 49), dist 242, ir 850, numpy 273, ops 1855 (19 files — `test_constant.py` 22), persist 156, pipeline_test 23, pipeline_env_defaults_test 36, test_pipeline_options 42, sparse 358, test_spec_compliance 86, trace 184, transforms 272. The summary line is swallowed by iree nanobind atexit noise in this env — count via the per-file `path: N` lines (126 files, summing to 6084). `backends/test_adapter_xla.py` (10) skips at collection without a user-provided PJRT plugin (reported as one extra `SKIPPED` item in `-rs` listings, no progress char — absent from the counts above). Full run in the default env (torch absent; iree/tvm/jaxlib/gcc present; free GPU so the iree-cuda smoke tests ran on a real GPU): **6064 passed, 20 skipped, 0 failed, exit 0** (passed = collected − 20 skipped; verified green — zero `F` markers in the progress output) — the skips are 11 torch-absent (6 `core/test_tensor.py::TestTorchInterop` + 1 `test_spec_compliance.py` + 4 `etl.bench` torch-present: `test_torch_repeat_calls.py` ×3 + `test_conformance.py` ×1) + 9 xla-plugin-absent (6 `backends/test_mixed_dtype.py` + 2 `backends/test_nsga2_xla_route.py` + the env-dependent xla parametrization of `backends/test_external_call_iree.py::test_xla_tvm_still_reject_with_round1_message`, which passes when the xla adapter was already registered earlier in the process). With torch installed the 7+4 run and the 2 torch-absent inverse `etl.bench` tests skip instead; with a PJRT plugin the xla-plugin-absent skips run. The `opt_level` feature suites: `tests/backends/test_opt_level.py` (53 — ported from the in-package `etl/backends/opt_level_test.py`, which pytest never collected), `tests/bench/test_backend_options.py` (13 — `etl.bench._util.resolve_backend_options` O3-injection contract), and the 14 new `ETL_OPT_LEVEL` items in `test_pipeline_options.py`.
- Plain `python3 -m pytest` is fully green; any failure is a regression in either etl or the tests.
