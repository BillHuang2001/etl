# etl.bench — conformance & benchmark harness

## Intent

A standalone harness subpackage that runs curated example etl programs against pure-numpy references and — optionally — PyTorch references, reporting **precision** (conformance: max-abs/max-rel errors, pass/fail per reference) and **speed** (benchmark: best-of-N wall-clock ms per implementation + speedup ratios). It is a verification/measurement tool for the etl pipeline (`etl.build` + `etl.run` on the default numpy backend), not part of the etl runtime.

## torch-optionality constraint (binding)

- The etl project must NOT depend on torch: `import etl` / `import etl.bench` MUST always succeed without torch installed.
- torch is imported ONLY lazily inside function bodies (see `_torch.py` — `torch_available()` / `require_torch()`; every example's `torch_ref` factory goes through `require_torch()`). NEVER at module top level, never at import time.
- Missing torch → a clear `ImportError` mentioning `pip install etl[bench]` (never a raw `ModuleNotFoundError` escaping). `use_torch=True` / `--torch` require torch; `use_torch=None`/`--torch` default = auto (torch comparisons run iff `import torch` succeeds).

## API surface (public, via `etl.bench`)

- `conformance(examples=None, *, use_torch=None, tolerance=None, rtol=1e-5, atol=1e-5, seed=0) -> ConformanceReport` — per example: generate seeded inputs, `etl.build`+`etl.run`, compare vs numpy ref (always) and torch ref (if enabled) with the documented rule: `tolerance=None` → allclose-style `|a-b| <= atol + rtol*|b|` per element (float64); numeric `tolerance` → `max_abs_error <= tolerance`. NaN fails. Per-example failures are recorded in the report (`error` field, `overall_pass=False`), not swallowed.
- `benchmark(examples=None, *, use_torch=None, repeats=20, warmup=2, seed=0) -> BenchmarkReport` — best-of-`repeats` ms after `warmup` untimed runs for etl/numpy/[torch]; speedup = `reference_ms / etl_ms`.
- `list_examples() -> list[str]`, `get_example(name) -> Example` (unknown name → `UnknownExampleError` listing available names).
- Report types: `ExampleResult`, `ConformanceReport` (has `overall_pass` property + `to_dict()`/`to_json()`), `BenchmarkReport` (`to_dict()`/`to_json()`); `print_report(report)` and `str(report)` render human-readable tables.
- CLI: `python -m etl.bench [--examples NAME[,NAME...]] [--conformance|--no-conformance] [--benchmark|--no-benchmark] [--torch|--no-torch] [--repeats N] [--seed N]`; exit 0 = ok, 1 = conformance failure, 2 = usage error (unknown example, torch missing).

## Structure

| File | Purpose |
|---|---|
| `examples.py` | `Example` dataclass + registry (10 examples: matmul, conv2d, conv2d_same, conv2d_stride2, elementwise_fusion, softmax, layernorm, mlp, cumsum, attention), each with `@etl.defn` graph, static `TensorSpec`s, pure-numpy ref, optional lazy torch ref; `generate_inputs(example, seed)` via `numpy.random.default_rng`. Small static float32 shapes (full conformance < ~0.5s). |
| `conformance.py` | `conformance()` + elementwise `_compare` (documented pass rule). |
| `benchmark.py` | `benchmark()` (best-of-N timing, documented). |
| `report.py` | Dataclasses + table printers (< 300 lines). |
| `_torch.py` | Lazy torch probe/require with the `pip install etl[bench]` hint. |
| `_util.py` | `resolve_examples`, `resolve_torch_mode` (→ `(mode, enabled, available)`), `best_time_ms`, `flatten_outputs`. |
| `__main__.py` | argparse CLI, `main(argv=None) -> int` for testability. |

## Constraints

- Deps: stdlib + numpy + etl only (torch optional/lazy). CPU only. No edits outside `./etl/bench/`.
- Numpy conv reference (`_conv2d_numpy`) mirrors etl's conv semantics exactly (VALID; SAME = TF convention `out=ceil(d/stride)`, pad split `(total//2, total-total//2)` per spatial axis — see `etl/backends/numpy/kernels/linalg.py`).
- Tests live in the sibling `../tests/` node (read-only from here) — this node ships no test files; the objective's verification commands (`python3 -c "import etl.bench"`, `conformance(['matmul'])`, `conformance(use_torch=True)` hint error, full `conformance()`, CLI) are the acceptance checks.
