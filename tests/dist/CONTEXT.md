# tests/dist — etl.dist collective test suite

## Intent

pytest suite validating the `etl.dist` contract (`../../etl/dist/CONTEXT.md` — sibling, read-only): explicit graph-time collectives under the SPMD local-tensor model, `Group` static values, `rank()`/`world_size()` graph scalars, the pluggable collective-executor hook, and the numpy backend's single-rank identity execution semantics. Tests are the executable spec: collectives are explicit `collective`-effect ops, results are LOCAL shapes only (no global tensor type), no eager mode, hook errors propagate.

## Structure

| File | Covers |
|---|---|
| `test_group.py` | `Group`/`group()`/`WORLD_GROUP`: attrs, validation errors, hashability/immutability, `size()`/membership, world-group defaulting |
| `test_local_shape_semantics.py` | Worked-example local-shape math (4-rank table from the contract), negative-axis normalization, world-group `None` (runtime-dynamic) dims |
| `test_ops.py` | Graph construction: op names/attrs/effects for all six collectives + `rank`/`world_size`, full error matrix (TraceError/TypeError/ShapeError/ValueError) |
| `test_execution.py` | `etl.evaluate` under the default identity executor: world-group identity for all six collectives × dtypes; declared-shape validation failures for explicit-group shape-changing collectives; rank 0 / world_size 1 |
| `test_executor_hook.py` | Hook round-trip (set/get/clear/TypeError), per-collective argument forwarding from the interpreter kernels, multi-rank `SimExecutor` (all_reduce sum, all_gather concat, reduce_scatter slice, broadcast, permute routing, all_to_all), `rank_context` per-run override, error paths (BackendError propagation, dtype/shape validation) |
| `test_context.py` | `RankContext` validation/frozen-ness; thread-local `set_rank_context`/`get_rank_context` overriding `rank()`/`world_size()` inside `etl.evaluate`; per-run `backend_executable.run(..., rank_context=...)` override |

## Constraints

- Siblings (`../../etl/`, other test dirs) are READ-ONLY — never modify etl to make a test pass. A real contract violation stays failing with a `# BUG(etl): <description>` marker and gets reported to the parent.
- CPU only, small shapes (≤ (1024, 1024)), each file < 2s. No network, no GPU.
- The executor hook is process-global and the rank context is thread-local: tests that install custom executors / contexts MUST use autouse fixtures resetting state in a `finally` (patterns live in `test_executor_hook.py` / `test_context.py` — reuse them).

## Notes for agents

- **pytest collection gotcha (IMPORTANT):** `dist` is in pytest's default `norecursedirs`, and the root `pyproject.toml` does not override it — so plain `python3 -m pytest` (or `pytest tests`) silently collects ZERO tests from this directory. Run this suite explicitly: `python3 -m pytest -q tests/dist`. Fixing collection belongs to the parent (root `pyproject.toml`, `[tool.pytest.ini_options] norecursedirs` without `dist`) — outside this node's write scope.
- **Registry wins over prose:** `etl/dist/CONTEXT.md` still documents two stale "known gaps" — (a) `broadcast_collective` declares a `src_rank` attr (default 0, filled by the builder), and (b) world-group tracing is "blocked" by an ATTR_INT `group_size=None` issue. Both are FIXED upstream: `group_size` AttrSpecs carry `default=None` and world-group tracing works. The tests assert current registry behavior; do not "fix" tests to match the stale prose.
- **Documented v1 protocol limitations (not bugs — tested as-is):** the executor protocol has no `reduce_op` for `reduce_scatter` (not forwarded) and a single `axis` for `all_to_all` (only `split_axis` forwarded); `broadcast` forwards `src_rank=0` (dist validates but does not record the attr).
- **Inspection idiom:** after `etl.trace(f, core.TensorSpec(...))`, `graph.module.functions[0].entry_block` is a property (a Block) — `.ops` is the op list; effects compare as plain strings (`op.definition.effect == "collective"` / `"read"`).
- **Execution contract:** the interpreter validates executor results against the op's declared result types (dtype exact → `BackendError`, shape elementwise → `ShapeError`, `None` dims unchecked). The default identity executor therefore FAILS explicit-group shape-changing collectives (declared local shape differs from input) — that is asserted in `test_execution.py`, and shape-changing results are produced by custom simulators in `test_executor_hook.py`.
