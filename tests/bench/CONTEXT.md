# tests/bench — etl.bench harness tests

## Intent

pytest suite validating the standalone conformance & benchmark harness in the sibling `../../etl/bench/` (read-only from here): torch-optional importability, the example registry, conformance/benchmark reports, and the CLI. Tests are the executable spec of the harness's contract — per-example failures are recorded in reports (never swallowed), so a failing test here means a real regression in etl.bench or its contract.

## Structure

| File | Contract under test |
|---|---|
| `test_import.py` | `import etl` / `import etl.bench` succeed and never pull `torch` into `sys.modules` (lazy-torch binding; no skip guards — valid in both torch-present and torch-absent environments) |
| `test_examples.py` | `list_examples()` exact 10-name registry order; `get_example(name)` → frozen `Example` dataclass (name/description/specs of `etl.TensorSpec`/graph/numpy_ref/generate_inputs); `UnknownExampleError` (ValueError subclass, message lists available names, importable from both `etl.bench` and `etl.bench.examples`); `conformance` raises it for unknown names |
| `test_conformance.py` | `conformance(...)` numpy-only full run (all 10 pass, `torch_pass is None`), `to_dict()`/`to_json()` round-trip incl. `overall_pass` and ExampleResult field names, `str`/`print_report` (TypeError for non-reports), single-name + `tolerance` args, torch-absent-only: `use_torch=True` → ImportError with `pip install etl[bench]` hint, auto mode skips torch; torch-present: full `use_torch=True` run passes (skips here) |
| `test_benchmark.py` | `benchmark(...)` best-of-N timing (order kept, positive ms/speedups, torch fields None when disabled), `BenchmarkReport` serialization round-trip, `str` smoke |
| `test_cli.py` | `python -m etl.bench` subprocess + in-process `main(argv)` — exit 0 ok / 2 usage error (`etl.bench: error: ...` on stderr) |

## Constraints

- CPU only, small fast tests (full conformance ~0.1s; a few calls per file).
- NEVER `import torch` at module scope — only `pytest.importorskip("torch")` (first line of a function) or `importlib.util.find_spec("torch")` for environment detection (`TORCH_ABSENT`/`requires_torch_absent` pattern).
- Subprocess CLI tests run with `cwd=Path(__file__).resolve().parents[2]` (repo root) so `python -m etl.bench` finds the `etl` package; keep one subprocess per case.
- Plain `python3 -m pytest tests/bench` is green in torch-absent environments with exactly 1 skip (the torch-present conformance test) — a different skip/failure count means a regression.
