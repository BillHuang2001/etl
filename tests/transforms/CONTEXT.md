# tests/transforms — transform test suite

## Intent

Numerical and semantic validation of the graph→graph transforms (`grad` / `jvp` / `vjp`; `vectorize`/`vmap` tests come later) against the binding contract in `../../etl/transforms/CONTEXT.md`. Transforms never execute at transform time: every numerical check runs the produced graph through the explicit staging pipeline (`lower → compile → load → run`), verifies it, and compares against numpy central finite differences.

## Structure

- `_fd_utils.py` — shared helpers (leading underscore keeps pytest from collecting it): `run_graph` (explicit staging), `to_np` (Tensor→numpy trees), `central_directional` / `central_jacobian` / `central_grad` (numpy central differences), and `run_jvp` / `run_vjp` / `run_grad` (transform → verify → run).
- `test_jvp.py` — forward-mode: jvp vs central directional differences (square/sum-square/sin/cos/exp/power chains, 2-tangent multiply, multi-output, conv, static inputs); tangent normalization (bare spec / 1-tuple / bare None spellings, None = zero tangent); structure/shape/dtype mismatch → `TransformError`; graph-form ≡ callable-form; concrete Tensor → `TraceError`; `runtime_call` → `TransformError`; ZeroTangent ops (equal/argmax) give zero tangents.
- `test_autodiff_rules.py` — chain rule through composites vs finite differences: grad of dot (2D×2D) and 1D·1D inner product (multiply+sum — `etl.dot` requires rank ≥ 2 by design), sigmoid, reduce_sum(x²), reshape, transpose, broadcast (see Known Issues); jvp AND vjp through dot/sigmoid/reshape/transpose (vjp cotangents vs ct·J); stop_gradient blocks flow; chained elementwise ops (tanh∘sigmoid, exp∘sin); argmax zero gradient; deferred conv VJP and runtime_call → `TransformError`; defn + `etl.evaluate` shorthand.

## Conventions

- Finite-difference references are computed in **float64**; float32 tests upcast inputs and use a coarser step (a 1e-6 step would be dominated by float32 rounding). Settings: f64 → eps 1e-6, rtol 1e-7; f32 → eps 1e-3, rtol 1e-5; atol 1e-6 both.
- Runtime args mirror the transformed graph's input tree: jvp/vjp graphs take `(primal input tree, flat tangent/cotangent tuple)`; `None` tangent entries are passed as `None` leaves at runtime; `grad` graphs keep the original input tree (static leaves included).
- `grad` with an int argnum returns a bare tensor; `argnums=None` returns a tuple (even for one input).

## Known Issues

- **etl BUG (kept as an intentionally failing test)**: `test_grad_through_broadcast` — binary elementwise VJP rules (`multiply`, also `add`/`subtract`/`divide`/`power`) do not reduce implicit broadcast dims back to the operand shape. Grad of `sum(x * b)` with `x` shape `(3,1)`, constant `b` shape `(1,4)` comes out shaped `(3,4)` with unreduced values 3 instead of `(3,1)` all-12. The explicit `broadcast` op's VJP reduces correctly; the implicit broadcast inside elementwise ops does not. Minimal repro:

  ```python
  import numpy as np, etl
  spec = etl.TensorSpec((3, 1), etl.float64)
  b = np.full((1, 4), 3.0)
  def f(x):
      return etl.sum(etl.multiply(x, etl.constant(etl.tensor(b))))
  g = etl.grad(f, 0)(spec)
  out = etl.run(etl.load(etl.compile(etl.lower(g))), np.ones((3, 1)))
  out.numpy().shape  # (3, 4) — expected (3, 1), all 12.0
  ```
