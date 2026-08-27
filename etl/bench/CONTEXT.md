# etl.bench — conformance & benchmark harness

## Intent

A standalone harness subpackage that runs curated example etl programs against pure-numpy references and — optionally — PyTorch references, reporting **precision** (conformance: max-abs/max-rel errors, pass/fail per reference) and **speed** (benchmark: best-of-N wall-clock ms per implementation + speedup ratios). It is a verification/measurement tool for the etl pipeline (`etl.build` + `etl.run`), usable on **any registered backend and device** (default: numpy backend, CPU). It is not part of the etl runtime.

## torch-optionality constraint (binding)

- The etl project must NOT depend on torch: `import etl` / `import etl.bench` MUST always succeed without torch installed.
- torch is imported ONLY lazily inside function bodies (see `_torch.py` — `torch_available()` / `require_torch()`; every example's `torch_ref` factory goes through `require_torch()`). NEVER at module top level, never at import time.
- Missing torch → a clear `ImportError` mentioning `pip install etl[bench]` (never a raw `ModuleNotFoundError` escaping). `use_torch=True` / `--torch` require torch; `use_torch=None`/`--torch` default = auto (torch comparisons run iff `import torch` succeeds).

## API surface (public, via `etl.bench`)

- `conformance(examples=None, *, use_torch=None, tolerance=None, rtol=1e-5, atol=1e-5, seed=0, backend="numpy", device=None, **backend_options) -> ConformanceReport` — per example: generate seeded inputs, `etl.build(graph, *specs, backend=backend, device=resolved_device, **backend_options)` + `etl.run`, compare vs numpy ref (always) and torch ref (if enabled) with the documented rule: `tolerance=None` → allclose-style `|a-b| <= atol + rtol*|b|` per element (float64); numeric `tolerance` → `max_abs_error <= tolerance`. NaN fails. Per-example failures are recorded in the report (`error` field, `overall_pass=False`), not swallowed.
- `benchmark(examples=None, *, use_torch=None, repeats=20, warmup=2, seed=0, backend="numpy", device=None, **backend_options) -> BenchmarkReport` — best-of-`repeats` ms after `warmup` untimed runs for etl/[numpy]/[torch]; speedup = `reference_ms / etl_ms`.
- **`backend`**: any registered etl backend name (validated up front via `etl.backends.get` — unknown names / missing adapter deps raise `core.BackendError` BEFORE any example runs; the CLI maps that to exit 2).
- **`device`**: `None` → `core.Device("cpu", 0)`; a `core.Device` (kind must be `"cpu"`/`"cuda"`); or a `"KIND[:INDEX]"` string (`"cpu"`, `"cuda"`, `"cuda:3"`). The report records the formatted device string (`"cpu"` or `"cuda:N"`).
- **`**backend_options`**: passthrough to `etl.build` (compile options for compiler backends). Device-derived default: for any non-numpy backend (numpy is the only interpreter; every other registered backend is a compiler backend) WITHOUT an explicit `target_backends` option, the harness injects `target_backends=["cuda"]` on a cuda device, `["llvm-cpu"]` otherwise. An explicit option always wins.
- Torch references run on the same device kind when it is cuda AND torch CUDA is available AND the index is within `torch.cuda.device_count()` (`resolve_torch_device` in `_util.py`); otherwise they run on CPU (`device=None`) — exactly the historical CPU behavior. Every `torch_ref` factory has signature `(inputs, device=None)`.
- `list_examples() -> list[str]`, `get_example(name) -> Example` (unknown name → `UnknownExampleError` listing available names).
- Report types: `ExampleResult`, `ConformanceReport` / `BenchmarkReport` (fields include `backend: str = "numpy"` and `device: str = "cpu"` — the formatted device; `ExampleResult` fields are fixed, see `report.py`); `print_report(report)` and `str(report)` render human-readable tables — the header shows `backend=... device=...` ONLY when non-default, so default runs render byte-identically to before these fields existed.
- CLI: `python -m etl.bench [--examples NAME[,NAME...]] [--conformance|--no-conformance] [--benchmark|--no-benchmark] [--torch|--no-torch] [--repeats N] [--seed N] [--backend NAME] [--device KIND[:INDEX]] [--backend-option KEY=VALUE ...]`; exit 0 = ok, 1 = conformance failure, 2 = usage error (unknown example / torch missing / unknown backend / malformed `--device` / malformed `--backend-option`). `--backend-option` values parse JSON-first with a raw-string fallback (so `target_backends=["cuda"]`, `foo=true`, `foo=bar` all work); later duplicates override earlier ones.

## Structure

| File | Purpose |
|---|---|
| `examples.py` | `Example` dataclass + registry (10 examples: matmul, conv2d, conv2d_same, conv2d_stride2, elementwise_fusion, softmax, layernorm, mlp, cumsum, attention), each with `@etl.defn` graph, static `TensorSpec`s, pure-numpy ref, lazy torch ref `(inputs, device=None) -> ndarray` (torch.as_tensor + `.cpu().numpy()` — CPU behavior identical to `from_numpy`). Small static float32 shapes (full conformance < ~0.5s on numpy; iree adds compile time per example). |
| `conformance.py` | `conformance()` + elementwise `_compare` (documented pass rule). |
| `benchmark.py` | `benchmark()` (best-of-N timing, documented). |
| `report.py` | Dataclasses (`backend`/`device` fields on the reports) + table printers (< 300 lines). |
| `_torch.py` | Lazy torch probe/require with the `pip install etl[bench]` hint. |
| `_util.py` | `resolve_examples`, `resolve_torch_mode`, `best_time_ms`, `flatten_outputs`, `resolve_device` (None/Device/`"KIND[:INDEX]"` → `core.Device`), `format_device`, `resolve_backend` (registry-validated), `resolve_backend_options` (target_backends auto-default), `resolve_torch_device` (torch.device or None — never imports torch). |
| `__main__.py` | argparse CLI (`--backend`, `--device`, repeatable `--backend-option`), `main(argv=None) -> int` for testability. |

## Constraints

- Deps: stdlib + numpy + etl only (torch optional/lazy). CPU only for the reference path; GPU runs go through compiler backends. No edits outside `./etl/bench/`.
- Numpy conv reference (`_conv2d_numpy`) mirrors etl's conv semantics exactly (VALID; SAME = TF convention `out=ceil(d/stride)`, pad split `(total//2, total-total//2)` per spatial axis — see `etl/backends/numpy/kernels/linalg.py`).
- The numpy backend is CPU-only — its `load()` rejects non-CPU devices with an explicit `BackendError`; the harness does NOT special-case around that (the explicit error is the contract, recorded per-example).
- The harness NEVER selects a GPU implicitly — the caller chooses the device explicitly (`--device cuda:N` / `device=`). Operational GPU policy (runners, not the harness): scan for the GPU with the most free memory before any GPU use (`nvidia-smi --query-gpu=index,memory.free --format=csv`) and don't occupy it long.
- Argument errors (bad `backend`/`device`/`tolerance`/`repeats`/`warmup`, unknown backend, unknown example, `use_torch=True` without torch) raise up front; per-example EXECUTION failures (e.g. a backend's `BackendError` during build/run) are recorded in the report — never swallowed, never crash the run.

## Known Issues

- **mlp fails the default tolerance on the iree backend (CPU)**: deterministic `max_abs_error≈3.8e-05` (2 of 4096 elements exceed `atol + rtol*|b|` at the strict defaults). Verified NOT a harness bug: IREE's fp32 output is *closer* to a float64 ground truth (5.3e-05) than the numpy fp32 reference itself (6.5e-05) — the delta is fp32 accumulation-order noise (FMA contraction in llvm-cpu codegen), at a tolerance tuned for the numpy backend (etl and the numpy ref share kernels → 0 error). Recorded per contract (`numpy_pass=False`, `overall_pass=False`, exit 1); users can loosen `rtol`/`atol`/`tolerance` via the API. Do NOT special-case it in the harness.
- **cumsum fails on the iree backend**: explicit `BackendError` (StableHLO v1 defers cumsum export — documented, intentional limitation), recorded per-example.
- **iree/cuda harness runs are blocked by `etl.pipeline.build` dropping compile options** (fix is in `etl/pipeline.py`, outside this node): the harness resolves `target_backends=["cuda"]` correctly and the iree adapter's `compile` DOES consume it (the old "adapter hardcodes llvm-cpu" issue is resolved in the adapter), but `pipeline.build` forwards `**options` only to `lower()`, NOT to `compile()`, so the artifact is always compiled for the default `llvm-cpu` and `load` on a cuda device fails with the explicit `BackendError` "...compiled for target(s) llvm-cpu and cannot run on the requested device... (device kind 'cuda' requires compile target 'cuda')". iree/CPU works only because `llvm-cpu` is the compile default. Until `pipeline.build` is fixed, iree/cuda runs need a runtime shim patching `etl.build` to forward options to `compile` (the full iree/cuda matrix in the T1-A7 report was produced that way — no repo changes).
- iree-base-runtime emits `nanobind: leaked ...` warnings at interpreter shutdown (upstream teardown noise; harmless for exit codes and report output).
- Older torch installs or a torch without CUDA silently fall back to CPU torch references when a cuda device is requested (`resolve_torch_device` → None).

## Test strategy

Sibling `../tests/bench/` (read-only from here) covers the harness contract: CLI exit codes/usage errors, conformance/benchmark reports and serialization, example registry, torch-optional imports, torch-present regressions (skip when torch absent). Verified: 30 passed, 4 skipped (torch-present tests) without torch. The objective's acceptance checks are CLI/API runs (`python3 -m etl.bench --help`, default run, `--backend iree --device cpu`, `conformance(['matmul'], backend='iree', device='cpu')`) — always via `/usr/bin/python3` (plain `python` is not on PATH in the dev environment).
