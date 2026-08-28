# tests/transforms — graph-transform test suite

## Intent

Tests for `etl.transforms` (sibling binding contract: `../../etl/transforms/CONTEXT.md` — read it before changing anything). Transforms are **graph→graph and never execute**: every numerical check builds the transformed `Graph` from `TensorSpec`s, verifies it (`graph.verify()`), and runs it through the explicit staging pipeline (`lower → compile → load → run`), comparing against numpy finite-difference or per-row references. Error-path tests assert `TransformError`/`ShapeError`/`TraceError`/`TypeError` per the v1 restrictions (mapped axes ∈ {None, 0} with leading-only mapping; control flow not vectorizable; ruleless ops fail loudly — never a silent fallback).

## Structure

- `_fd_utils.py` — shared helpers (leading underscore keeps pytest from collecting it): `run_graph` (explicit staging), `to_np` (Tensor→numpy), `central_directional` / `central_jacobian` / `central_grad` (float64 numpy central differences), and `run_jvp` / `run_vjp` / `run_grad` (transform → verify → run).
- `test_vectorize.py` — the batching primitive: per-op numerics vs stacked per-row runs (add/mul/dot/sum), dict/tuple `in_axes` with unmapped broadcast, result contains NO `vectorize` op, input graph not mutated, ONE shared `Dim('batch')` per pass (object identity across mapped inputs and the mapped output; no `batch_1` dims / DimExprs), vectorize on already-symbolic row specs (`(B, 16)`) preserving the row dim by identity with correct numpy-backend numerics, validation errors (bare callable → TypeError pointing at `etl.vmap`; non-leading/out-of-range axes; pytree mismatch; bare axes with multiple tensor inputs).
- `test_vmap.py` — sugar equivalence: vmap⇔vectorize IR equality (serialize_module + pretty_print, fn and Graph cases), `TransformCallable` kind/Graph return, once-only tracing, numerics, default `in_axes=0` + unmapped-arg broadcast, `out_axes` semantics (None-on-mapped → TransformError, size-one insertion for unmapped, non-zero deferred).
- `test_grad.py` — reverse mode: analytic checks (x²+3x → 2x+3), argnums int/tuple/None (incl. static inputs excluded from numbering), non-scalar/multi-output → `ShapeError`, documented ZeroTangent zeros (argmax/equal), sign's implemented zero-derivative rule (`sum(sign(x))` → all-zeros gradient; `sum(x*sign(x))` = `sum(abs(x))` → gradient `sign(x)` vs analytic + central difference), `runtime_call` → `TransformError`, Graph-form ≡ callable-form, argnums validation errors.
- `test_jvp.py` — forward mode vs central directional differences, tangent normalization (bare spec / 1-tuple / bare None spellings; None = zero tangent), structure/shape/dtype mismatch → `TransformError`, Graph-form ≡ callable-form, concrete Tensor → `TraceError`, `runtime_call` → `TransformError`, zero-tangent rules (argmax/equal, sign).
- `test_vjp.py` — reverse mode: default scalar-one cotangent equals `grad` result, pullback == ct @ finite-difference Jacobian, multi-output cotangent pytrees, cotangent mismatch errors, non-scalar/multi-output with cotangents=None → `ShapeError`, stop_gradient → zero cotangent, sign's zero-derivative rule → all-zeros input cotangent.
- `test_autodiff_rules.py` — chain rule through composites vs finite differences: grad of dot (2D×2D; `etl.dot` requires rank ≥ 2 by design — 1D·1D inner product is composed as multiply+sum), sigmoid, reduce_sum(x²), reshape, transpose, broadcast; jvp AND vjp through dot/sigmoid/reshape/transpose; stop_gradient blocks gradient flow; chained elementwise ops; deferred conv VJP → `TransformError`.
- `test_errors.py` — TransformError paths: axis errors (non-zero/out-of-range/mismatch/mapped-static), ruleless ops (`runtime_call`, collectives like `all_reduce`, control flow) under vmap/grad/jvp/vjp, argnums/tangent/cotangent validation, concrete-Tensor rejection for all four `TransformCallable` kinds, plus the documented zero-gradient behavior of argmax/equal (NOT errors).
- `test_equivalence_meta.py` — transformed graphs are ordinary graphs: `verify()` passes, no banned transform op names anywhere in the module (walk all blocks/regions), every op resolves to a registered opdef, `main` has a `return` terminator; `TransformCallable.kind` values; input-graph non-mutation (serialize before == after) for all five graph-form transforms.
- `test_vmap_control_flow.py` — v1 restriction behavior: vmap/vectorize over `cond`/`while_loop`/`scan` → `TransformError` ("cannot batch op 'if'/'while' … not vectorizable in v1"), grad/jvp/vjp over control flow → "no VJP/JVP rule for op 'if'"; plain `etl.trace` of the same functions still succeeds (restriction is transform-only).

## Conventions

- Finite-difference references are computed in **float64**; float32 tests use a coarser step (a 1e-6 step would be dominated by float32 rounding). Settings: f64 → eps 1e-6, rtol 1e-7; f32 → eps 1e-3, rtol 1e-5; atol 1e-6 both.
- Runtime args mirror the transformed graph's input tree: jvp/vjp graphs take `(primal input tree, flat tangent/cotangent tuple)`; `None` tangent/cotangent entries are passed as `None` leaves at runtime; `grad` graphs keep the original input tree (static leaves included).
- `grad` with an int argnum returns a bare tensor; `argnums=None`/tuple returns a tuple (even a 1-tuple).
- Small shapes (≤ 5×4×4), CPU only, each file <2s; parametrize fns × dtypes instead of looping.

## Notes for agents

- The traced input tree wraps EACH argument (root tuple → one child per argument); a dict-taking fn therefore needs axes as a 1-tuple wrapping the dict, e.g. `vectorize(g, ({"a": 0, "b": None},))`.
- `etl.run` returns a bare `Tensor` for single outputs and a tuple for structured outputs; `Tensor` has no `__array__` protocol — convert via `.numpy()` (the `to_np` helper).
- `etl.sum` axes must be an int/tuple (lists raise TypeError); `etl.evaluate` rejects static (non-tensor) args — use numpy references there.
- `etl.scan` lowers a step-0 `gather` into the entry block, so vectorizing it raises the gather deferral ("cannot batch op 'gather' with mapped data but unmapped indices") before the `while` op is reached — still a TransformError per contract (use `in_axes=(0, None)` on the scan state to reach the `while` path).
- When a test exposes behavior contradicting the etl contract, do NOT fix etl (sibling, read-only) and do NOT weaken the test: mark it with `# BUG(etl): <description>` and keep it failing, with the repro recorded in this file's Known Issues.
