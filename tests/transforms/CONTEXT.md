# tests/transforms — graph-transform test suite

## Intent

Tests for `etl.transforms` (sibling contract in `../../etl/transforms/CONTEXT.md`): `vectorize` is the batching primitive (graph→graph), `vmap` is transparent function-side sugar over it. Transforms never execute — all numerical checks use the explicit staging pipeline (`etl.run(etl.load(etl.compile(etl.lower(graph))), *args)`, module-level `run_graph` helper) and per-row references via `etl.evaluate`.

## Files

- `test_vectorize.py` — per-op batching numerics (add/mul/dot/sum, float32/float64 vs stacked per-row runs), dict/tuple `in_axes` forms with unmapped broadcast, no `vectorize` op in the result, input graph not mutated, leading `Dim('batch')` on mapped specs/outputs, static-value handling, argument validation errors (bare callable → TypeError pointing at `etl.vmap`, non-leading/out-of-range axes, pytree mismatch), control-flow deferrals (`cond`/`while_loop`/`scan` → TransformError).
- `test_vmap.py` — vmap⇔vectorize IR equivalence (serialize_module + pretty_print equality, fn and Graph cases), `TransformCallable` kind/Graph return, numerics, default `in_axes=0` + unmapped-arg broadcast, once-only tracing, concrete-Tensor rejection (TraceError), `out_axes` semantics (None-on-mapped → TransformError, size-one insertion for unmapped, non-zero deferred, pytree mismatch), in_axes validation.

## Notes for agents

- The traced input tree wraps EACH argument (root tuple → one child per argument); a dict-taking fn therefore needs axes as a 1-tuple wrapping the dict, e.g. `vectorize(g, ({"a": 0, "b": None},))`.
- `etl.run` returns a bare `Tensor` for single outputs and a tuple for structured outputs; `Tensor` has no `__array__` — convert via `.numpy()` (`to_np` helper).
- `etl.sum` axes must be an int/tuple (lists raise TypeError); `etl.evaluate` rejects static (non-tensor) args — use numpy references there.
- `etl.scan` lowers a step-0 `gather` into the entry block, so vectorizing it raises the gather deferral ("cannot batch op 'gather' …") before the `while` op is reached — still a TransformError per contract.
- Keep files fast (<2s each), CPU only, small shapes (≤ 5×4×4); parametrize fns × dtypes instead of looping.
