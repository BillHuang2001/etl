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
  - error messages include source locations (`file.py:line`); shape errors currently lack them (BUG-marked tests)

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
- **`tests/dist` collection:** pytest's default `norecursedirs` includes `dist`; `tests/conftest.py` strips it via `pytest_configure` so plain `python3 -m pytest` collects `tests/dist/`. The proper fix is `norecursedirs` (without `dist`) in the root `pyproject.toml` `[tool.pytest.ini_options]` — escalate to the repo root; once it lands the conftest hook is a no-op and can be removed.
- **BUG(etl) protocol:** tests asserting the documented contract that the implementation currently violates are kept failing with a `# BUG(etl): <description>` comment (and a minimal repro). Do NOT weaken them. After an etl fix, delete the marker comment and the corresponding `Known Issues` entry here and in the area's CONTEXT.md.
- Plain `python3 -m pytest` currently reports `27 failed` — all of them are intentional BUG(etl) tests listed below (not regressions).

## Known Issues (current etl bugs pinned by BUG(etl)-marked failing tests)

Consolidated list; details + minimal repros in each area's CONTEXT.md and the failing tests themselves. Fixes belong to `etl/` (delegate upstream), not here.

| # | Area | Failing test(s) | Bug (one line) |
|---|---|---|---|
| 1 | ops | `test_linalg.py::test_cumsum_reverse_scans_from_the_end` | `cumsum(reverse=True)` computes `flip(cumsum(x))` instead of `flip(cumsum(flip(x)))` (numpy kernel `_cumsum`) |
| 2 | ops | `test_elementwise.py::test_abs_complex_numerics_give_real_magnitude` | `abs` of complex: frontend post-casts result to real dtype but kernel already returns real → BackendError at run |
| 3 | ops/backends | `ops/test_elementwise.py::test_runtime_broadcast_conflict_raises_shape_error`, `backends/test_symbolic_dims.py::test_mixed_symbolic_concrete_size_mismatch_raises_etl_error` | runtime broadcast conflicts leak raw numpy `ValueError` instead of `core.ShapeError` |
| 4 | ops | `test_getitem.py::TestSliceIndex::test_zero_step_slice_malformed_param_error` | `x[::0]` leaks raw numpy `ValueError` instead of `TypeError` |
| 5 | ops | `test_errors.py::test_no_trace_message_mentions_defn` | no-active-trace message omits `@etl.defn` (contract requires naming all three entry points) |
| 6 | ops/spec | `ops/test_errors.py::test_shape_error_message_includes_call_site_location[plain-fn\|defn]`, `test_spec_compliance.py::TestErrorLocations::test_shape_errors_include_source_location[add\|dot]` | shape errors raised during inference carry no call-site location (error-strategy contract) |
| 7 | numpy | `test_composition.py::test_clip_upper_bound_only_is_minimum`, `::test_clip_lower_bound_only_is_maximum` | `enp.clip` None-bound branches inverted (upper bound → `maximum`, lower bound → `minimum`) |
| 8 | numpy | `test_composition.py::test_expand_dims_tuple_axis_ascending_numeric` | `enp.expand_dims` normalizes tuple axes against original rank instead of final ndim |
| 9 | numpy | `test_composition.py::test_pad_pair_rank1_numeric` | `enp.pad` rejects bare `(before, after)` pair on rank-1 (numpy accepts) |
| 10 | trace | `test_defn.py::test_defn_of_defn_is_idempotent` | `etl.defn(existing_defn)` builds a NEW Defn; contract: return unchanged |
| 11 | trace | `test_scan.py::test_scan_length_override_shortens_xs` | `scan` with explicit `length` < static leading dim raises; contract wants prefix-scan (check should be one-sided) |
| 12 | trace | `test_static_snapshot.py::test_dataclass_config_spec_is_rejected` | plain dataclass config as trace spec silently specializes; contract tension (root value-model table lists config objects as static) — needs parent decision |
| 13 | block | `test_rules.py::test_vmap_via_portable_fallback`, `::test_explicit_batching_rule_registers_and_wins_over_unsupported_policy`, `::test_explicit_rule_overrides_portable_decomposition`, `::test_vmap_without_rule_raises_transform_error_naming_the_block` | vmap/vectorize never dispatches `block_call` to the `block:<name>` rule namespace |
| 14 | block | `test_rules.py::test_grad_via_portable_decomposition` | portable VJP fallback seeds cotangents on pre-inline ids → all-zero gradients |
| 15 | block | `test_rules.py::test_jvp_derived_from_portable_vjp_fallback` | jvp never derived from a vjp rule (autodiff consults only jvp_rules) |
| 16 | block | `test_rules.py::test_elementwise_policy_passes_batch_dims_through` | batching policy (elementwise/map_over_batch) not honored by transforms |
| 17 | transforms | `test_autodiff_rules.py::test_grad_through_broadcast` | binary elementwise VJP rules don't reduce implicit broadcast dims back to operand shape |
| 18 | persist | `test_cache.py::test_dict_component_insertion_order_same_key` | dict insertion order changes cache keys (canonical-JSON keying contract; minor) |
| 19 | persist | `test_cache.py::test_distinct_keys_distinct_entries` | bare-string `key_components` collide with single-element tuples (minor) |

Also reported (non-failing): `etl/core/device.py` leaks raw numpy TypeErrors on some invalid inputs (see `tests/core/CONTEXT.md`); stale upstream docs in `etl/dist/CONTEXT.md` (world-group/source-rank "gaps" already resolved).
