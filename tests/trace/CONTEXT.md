# tests/trace — tracing / defn / control-flow test suite

## Intent

pytest suite validating the tracing machinery of `etl` (sibling — see
`../../etl/trace/CONTEXT.md` for the binding contract being tested):
`@etl.defn`, `etl.trace`, the `Graph` type, the active-builder context, and
runtime tensor control flow (`cond` / `while_loop` / `scan`). Tests assert
the CONTRACT — explicitness, static-value specialization, region-based IR,
structured pytree I/O — not incidental implementation behavior.

## Structure (flat — one file per module/feature)

| File | Covers |
|---|---|
| `test_defn.py` | `@etl.defn` decorator: Defn/marker, options, idempotence, metadata, always-raises `__call__` |
| `test_trace.py` | `etl.trace` core: symbolic dims, structured inputs, static specialization, closure capture, zero-IO, errors, fresh graphs |
| `test_graph.py` | `Graph` attrs, `print()`, `verify()`, `flatten_inputs`/`validate_inputs`, `unflatten_outputs`, save/load round-trip, `signature_info` |
| `test_builder_context.py` | `current_builder()` / `with_builder` (LIFO stack), ops outside trace, region routing |
| `test_cond.py` | `etl.cond`: runtime branch selection, `if`-op IR structure, pred validation, branch unification errors |
| `test_while_loop.py` | `etl.while_loop`: iteration counts, structured state, `while`-op IR, carried-type/static-leaf errors |
| `test_scan.py` | `etl.scan`: cumsum/max vs numpy, length override, structured xs/init, desugared IR, v1 static-length errors |
| `test_static_snapshot.py` | static dtype/Enum/slice/bool + dataclass-config specialization, trace-time snapshotting, run-time static validation |
| `conftest.py` | shared fixtures (below) |

## Shared fixtures (conftest.py)

- `run_graph(graph, *np_arrays)` — explicit staging pipeline
  `lower → compile → load(device="cpu") → run` (NOTE: `etl.build` accepts a
  defn/callable, NOT an already-traced Graph — this fixture wraps the
  explicit stages).
- `as_numpy(result)` — recursively converts `etl.Tensor` leaves of a
  structured result to numpy arrays for assertions.

## Constraints

- CPU only; numpy backend is the correctness reference. Small shapes, each
  test file < 0.5s in practice (< ~500 lines).
- Only numpy arrays (not numpy scalars) are accepted as run-time inputs —
  pass 0-d `np.array(v, dtype)` for scalar tensors.
- Inside traced functions, constants MUST be `etl.constant(etl.tensor(...))`
  or `etl.constant(etl.zeros(...))` — never capture a concrete `Tensor`.
- `etl.trace` is the FUNCTION (shadows the submodule as attribute); use
  `from etl.trace import ...` for module pieces (`StaticValue`,
  `current_builder`, `with_builder`, `Graph`).

## Notes for agents

- **BUG(etl) protocol**: tests that expose real contract violations in `etl`
  are KEPT FAILING with a `# BUG(etl): <desc>` comment + minimal repro. Never
  fix `etl` from this directory and never weaken such tests — the package
  root owns fixes. The suite is expected to show exactly the failures listed
  in Known Issues; any OTHER failure is a regression in either etl or the
  tests.
- **Tree leaf markers**: trace-time `input_specs`/`output_tree` use private
  `_TensorSpecLeaf`/`_SymbolicLeaf` marker types by design (dataclass leaves
  would be recursed into). Compare tree SKELETONS (type/node_data/children,
  ignoring leaf types) — never full TreeSpec equality against live objects.
- Nested `etl.trace` inside a trace is unspecified by the contract — no
  assertion is made for it (see test_builder_context.py docstring).

## Test strategy

Run `python3 -m pytest -q tests/trace` from the repo root (root conftest.py
puts the repo on sys.path). Spec-compliance cross-cutting tests
(staging explicitness, bind-as-sugar, vmap≡vectorize) live in
`tests/test_spec_compliance.py` (owned by the package root) — do not
duplicate them here.
