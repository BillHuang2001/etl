# tests/backends — backend test suite

## Intent

pytest suites validating the `etl` backend layer: the `Backend` interface,
the numpy interpreter (reference CPU backend, dynamic shapes + runtime
control flow), and the stablehlo exporter.

## Files

- `test_symbolic_dims.py` — symbolic (`Dim`/`DimExpr`) and runtime-dynamic
  (`None`) dim specs: one build runs at multiple concrete sizes; mixed
  symbolic + concrete inputs; rank checking; determinism. Contains one
  intentional failing test `test_mixed_symbolic_concrete_size_mismatch_raises_etl_error`
  (`# BUG(etl)` comment): a symbolic-vs-concrete size mismatch escapes the
  numpy interpreter as a raw `builtins.ValueError` instead of an
  `etl.ShapeError` — do NOT weaken the test; fix the bug in `etl/backends/numpy`
  (wrap kernel shape errors in `core.ShapeError`) instead.
- `test_control_flow.py` — `etl.cond` / `etl.while_loop` / `etl.scan` through
  the numpy interpreter (genuinely dynamic runtime control flow via recursive
  region execution): both branches, dynamic predicates, iteration counting,
  zero-iteration/early-exit loops, scan vs `np.cumsum`, nested
  cond/while (both directions), symbolic-dim graphs containing loops, and the
  documented v1 `TraceError` for symbolic scan lengths.
- `test_backend_interface.py`, `test_program.py`, `test_collectives.py`,
  `test_deferred_errors.py`, `test_runtime_call.py`, `test_artifact_persistence.py`,
  `test_stablehlo.py`, `test_numpy_interpreter.py` — (pending/filled by sibling
  tasks; check file contents before assuming coverage).

## Constraints

- CPU only, small shapes; the numpy backend is the reference for correctness.
- If a test exposes a real etl bug: keep the test failing with a
  `# BUG(etl): <description>` comment — never fix etl from here, never weaken
  the test. Report BUG findings to the parent agent.
- Run: `python3 -m pytest -q tests/backends/` from repo root.
