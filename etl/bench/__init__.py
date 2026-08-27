"""etl.bench — conformance & benchmark harness for example etl programs.

Runs curated example graphs through the explicit etl pipeline
(``etl.build`` + ``etl.run`` on the default numpy backend — or any chosen
backend/device: ``conformance(..., backend="iree", device="cuda:3")``) and
compares the results against pure-numpy references — and, when torch is
installed and requested, against PyTorch references. Reports precision
(conformance) and speed (benchmark) per example.

torch-optionality contract (binding): ``import etl`` and ``import etl.bench``
MUST always succeed without torch installed. torch is imported lazily inside
function bodies only (see ``_torch.py``); missing torch yields a clear
``ImportError`` mentioning ``pip install etl[bench]`` — never a raw
``ModuleNotFoundError`` traceback escaping to the user.

Quick start::

    from etl.bench import conformance, benchmark
    report = conformance(["matmul", "softmax"])   # numpy references
    print(report)                                  # human-readable table
    print(report.overall_pass)
    report = benchmark(use_torch=False)            # numpy-only timing
    report.to_json()                               # machine-readable

    # Chosen backend/device with compile-option passthrough:
    report = conformance(["matmul"], backend="iree", device="cpu",
                         target_backends=["llvm-cpu"])
    # device accepts None | core.Device | "KIND[:INDEX]" (e.g. "cuda:3").

CLI: ``python -m etl.bench --help`` (exit code 1 when any conformance check
fails).
"""
from . import report  # noqa: F401  (submodule, public)
from .benchmark import benchmark
from .conformance import conformance
from .examples import Example, UnknownExampleError, get_example, list_examples
from .report import BenchmarkReport, ConformanceReport, ExampleResult, print_report

__all__ = [
    "Example",
    "UnknownExampleError",
    "ExampleResult",
    "ConformanceReport",
    "BenchmarkReport",
    "conformance",
    "benchmark",
    "list_examples",
    "get_example",
    "print_report",
]
