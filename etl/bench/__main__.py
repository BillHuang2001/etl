"""CLI for etl.bench: ``python -m etl.bench [options]``.

Exit codes: 0 = success; 1 = at least one conformance check failed; 2 =
usage/argument errors (unknown example name, torch requested but missing).
"""
from __future__ import annotations

import argparse
import sys

from .benchmark import benchmark
from .conformance import conformance
from .examples import UnknownExampleError, list_examples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m etl.bench",
        description=(
            "Conformance & benchmark harness for example etl programs "
            "(numpy references always; torch references optional)."
        ),
    )
    parser.add_argument(
        "--examples",
        default=None,
        metavar="NAME[,NAME...]",
        help="comma-separated example names (default: all: "
        + ", ".join(list_examples())
        + ")",
    )
    parser.add_argument(
        "--conformance",
        dest="run_conformance",
        action="store_true",
        default=None,
        help="run conformance checks (default)",
    )
    parser.add_argument(
        "--no-conformance",
        dest="run_conformance",
        action="store_false",
        help="skip conformance checks",
    )
    parser.add_argument(
        "--benchmark",
        dest="run_benchmark",
        action="store_true",
        default=None,
        help="run benchmarks (default)",
    )
    parser.add_argument(
        "--no-benchmark",
        dest="run_benchmark",
        action="store_false",
        help="skip benchmarks",
    )
    parser.add_argument(
        "--torch",
        dest="use_torch",
        action="store_true",
        default=None,
        help="require torch references (clear error if torch is missing)",
    )
    parser.add_argument(
        "--no-torch",
        dest="use_torch",
        action="store_false",
        help="disable torch references (numpy-only)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        metavar="N",
        help="benchmark timed repeats, best-of-N reported (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="RNG seed for generated inputs (default: 0)",
    )
    return parser


def main(argv=None) -> int:
    """Run the CLI; returns the process exit code (see module docstring).

    Args:
        argv: optional argument list (defaults to ``sys.argv[1:]``) — makes
            the CLI testable.
    """
    args = _build_parser().parse_args(argv)
    names = None
    if args.examples is not None:
        names = [part.strip() for part in args.examples.split(",") if part.strip()]
    return_code = 0
    if args.run_conformance is not False:
        try:
            report = conformance(names, use_torch=args.use_torch, seed=args.seed)
        except (UnknownExampleError, ImportError) as exc:
            print(f"etl.bench: error: {exc}", file=sys.stderr)
            return 2
        print(report)
        if not report.overall_pass:
            return_code = 1
    if args.run_benchmark is not False:
        try:
            report = benchmark(
                names, use_torch=args.use_torch, repeats=args.repeats,
                seed=args.seed,
            )
        except (UnknownExampleError, ImportError) as exc:
            print(f"etl.bench: error: {exc}", file=sys.stderr)
            return 2
        print(report)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
