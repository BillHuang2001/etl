"""CLI for etl.bench: ``python -m etl.bench [options]``.

Exit codes: 0 = success; 1 = at least one conformance check failed; 2 =
usage/argument errors (unknown example name, torch requested but missing,
unknown backend, malformed ``--device``/``--backend-option``).
"""
from __future__ import annotations

import argparse
import json
import sys

from etl.core import BackendError

from ._util import resolve_device
from .benchmark import benchmark
from .conformance import conformance
from .examples import UnknownExampleError, list_categories, list_tags


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m etl.bench",
        description=(
            "Conformance & benchmark harness for example etl programs "
            "(numpy references always; torch references optional). Runs the "
            "etl graphs on a chosen backend/device (--backend/--device, "
            "default numpy/cpu) with backend compile options "
            "(--backend-option, repeatable)."
        ),
    )
    parser.add_argument(
        "--examples",
        default=None,
        metavar="NAME[,NAME...]",
        help="comma-separated example names, categories, or tags (default: "
        "all); categories: " + ", ".join(list_categories())
        + "; tags: " + ", ".join(list_tags()),
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
    parser.add_argument(
        "--backend",
        default="numpy",
        metavar="NAME",
        help="etl backend to run the graphs on (default: numpy; e.g. iree, "
        "xla, tvm — registered backends only, validated up front)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="KIND[:INDEX]",
        help="device to run on: 'cpu' or 'cuda[:INDEX]' (default: cpu; e.g. "
        "'cuda:3')",
    )
    parser.add_argument(
        "--backend-option",
        dest="backend_options",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend compile option, repeatable; VALUE is parsed as JSON "
        "with a raw-string fallback, e.g. "
        "--backend-option 'target_backends=[\"cuda\"]'",
    )
    return parser


def _parse_backend_option_value(value: str):
    """Parse one ``--backend-option`` value: JSON first, raw string fallback.

    ``"[\"cuda\"]"`` → ``["cuda"]``, ``"true"`` → ``True``, ``"bar"`` →
    ``"bar"``.
    """
    try:
        return json.loads(value)
    except ValueError:
        return value


def _parse_backend_options(pairs) -> dict:
    """Parse repeated ``KEY=VALUE`` backend-option strings into a dict.

    A missing ``=`` raises ``ValueError`` (usage error). Values parse JSON
    first with a raw-string fallback (see
    :func:`_parse_backend_option_value`). Later duplicates override earlier
    ones.
    """
    options = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(
                f"invalid --backend-option {pair!r}: expected KEY=VALUE"
            )
        key = key.strip()
        if not key:
            raise ValueError(
                f"invalid --backend-option {pair!r}: empty key"
            )
        options[key] = _parse_backend_option_value(value)
    return options


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
    try:
        device = resolve_device(args.device)
        backend_options = _parse_backend_options(args.backend_options)
    except ValueError as exc:
        print(f"etl.bench: error: {exc}", file=sys.stderr)
        return 2
    return_code = 0
    if args.run_conformance is not False:
        try:
            report = conformance(
                names, use_torch=args.use_torch, seed=args.seed,
                backend=args.backend, device=device, **backend_options,
            )
        except (UnknownExampleError, ImportError, BackendError) as exc:
            print(f"etl.bench: error: {exc}", file=sys.stderr)
            return 2
        print(report)
        if not report.overall_pass:
            return_code = 1
    if args.run_benchmark is not False:
        try:
            report = benchmark(
                names, use_torch=args.use_torch, repeats=args.repeats,
                seed=args.seed, backend=args.backend, device=device,
                **backend_options,
            )
        except (UnknownExampleError, ImportError, BackendError) as exc:
            print(f"etl.bench: error: {exc}", file=sys.stderr)
            return 2
        print(report)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
