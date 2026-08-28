# tests/bench — etl.bench harness tests

## Intent

pytest suite validating the standalone conformance & benchmark harness in the sibling `../../etl/bench/` (read-only from here): torch-optional importability, the example registry, conformance/benchmark reports, and the CLI. Tests are the executable spec of the harness's contract — per-example failures are recorded in reports (never swallowed), so a failing test here means a real regression in etl.bench or its contract.

## Structure

| File | Contract under test |
|---|---|
| `test_import.py` | `import etl` / `import etl.bench` succeed and never pull `torch` into `sys.modules` (lazy-torch binding; no skip guards — valid in both torch-present and torch-absent environments) |
| `test_examples.py` | `list_examples()` exact 26-name registry order (micro → grad → vectorize → large); `list_categories()` == the 4 categories; `get_example(name)` → frozen `Example` dataclass with the 11 documented fields (name/description/specs of `etl.TensorSpec`/graph/numpy_ref/torch_ref/rtol/atol/tolerance/category/inputs_fn) and `generate_inputs(seed)`; per-example tolerance overrides (mlp `tolerance=1e-4`, grad `rtol=atol=1e-3`, vectorize strict defaults); `UnknownExampleError` (ValueError subclass, message lists available names, importable from both `etl.bench` and `etl.bench.examples`); `conformance` raises it for unknown names |
| `test_conformance.py` | `conformance(...)` numpy-only full run (all 26 pass, `torch_pass is None`), per-example tolerance resolution (mlp records `tolerance=1e-4`; matmul falls back to the global `None`; grad records `rtol=atol=1e-3`), grad-example smoke (transform-product staging path), `to_dict()`/`to_json()` round-trip incl. `overall_pass` and all ExampleResult field names (incl. `rtol`/`atol`/`tolerance`), `str`/`print_report` (TypeError for non-reports), single-name + `tolerance` args, torch-absent-only: `use_torch=True` → ImportError with `pip install etl[bench]` hint, auto mode skips torch; torch-present: full `use_torch=True` run passes (skips here) |
| `test_benchmark.py` | `benchmark(...)` best-of-N timing (order kept, positive ms/speedups, torch fields None when disabled), `BenchmarkReport` serialization round-trip, `str` smoke |
| `test_cli.py` | `python -m etl.bench` subprocess + in-process `main(argv)` — exit 0 ok / 2 usage error (`etl.bench: error: ...` on stderr); `--examples` category selection (`grad`, mixed `grad,large`) filters the report |
| `test_torch_repeat_calls.py` | torch-present regression pin: repeated torch-mode `conformance`/`benchmark` calls in one process all succeed (`require_torch` cache); every test gated on `pytest.importorskip("torch")` |

## Constraints

- CPU only; full numpy-only conformance of all 26 examples ≈ 8-9 s per run (large category: conv2d_large im2col ref + matmul_1024 + transformer/nbody etl runs); torch-enabled full run ≈ 13 s (CPU torch refs). The whole file suite is ≈ 46-60 s.
- NEVER `import torch` at module scope — only `pytest.importorskip("torch")` (first line of a function) or `importlib.util.find_spec("torch")` for environment detection (`TORCH_ABSENT`/`requires_torch_absent` pattern).
- Subprocess CLI tests run with `cwd=Path(__file__).resolve().parents[2]` (repo root) so `python -m etl.bench` finds the `etl` package; keep one subprocess per case.
- Plain `python3 -m pytest tests/bench` is green in torch-absent environments with 52 passed / 4 skipped (the torch-present conformance test + the 3 `test_torch_repeat_calls.py` torch-present regression pins); with torch installed: 54 passed / 2 skipped (the 2 torch-absent-only inverse tests). A different skip/failure count means a regression.
