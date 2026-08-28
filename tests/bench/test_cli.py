"""etl.bench CLI contract (``python -m etl.bench``).

Exit codes: 0 = ok, 1 = at least one conformance check failed, 2 = usage
error (unknown example name / torch requested but missing — printed to
stderr as ``etl.bench: error: ...``). ``main(argv=None) -> int`` is
importable from ``etl.bench.__main__`` for in-process testing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_conformance_ok_subprocess():
    proc = subprocess.run(
        [
            sys.executable, "-m", "etl.bench",
            "--conformance", "--no-benchmark", "--no-torch",
            "--examples", "matmul",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "matmul" in proc.stdout
    assert "PASS" in proc.stdout


def test_cli_unknown_example_usage_error_subprocess():
    proc = subprocess.run(
        [
            sys.executable, "-m", "etl.bench",
            "--conformance", "--no-benchmark", "--no-torch",
            "--examples", "nope",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 2
    assert "nope" in proc.stderr


def test_main_in_process_success(capsys):
    from etl.bench.__main__ import main

    assert (
        main(
            [
                "--conformance", "--no-benchmark", "--no-torch",
                "--examples", "matmul",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "matmul" in captured.out
    assert "PASS" in captured.out


def test_main_in_process_category_selection(capsys):
    from etl.bench.__main__ import main

    # --examples accepts names, categories, AND tags, expanded with precedence
    # category name → exact example name → tag name. "grad" is now a TAG that
    # expands to its 11 grad_* examples only — no micro/large examples in the
    # report.
    assert (
        main(
            [
                "--conformance", "--no-benchmark", "--no-torch",
                "--examples", "grad",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    for name in ("grad_mlp", "grad_mix", "grad_stopgrad", "grad_structural"):
        assert name in captured.out
    for name in ("grad_erf", "grad_cumsum"):
        assert name in captured.out
    assert "matmul" not in captured.out
    assert "transformer" not in captured.out


def test_main_in_process_mixed_names_and_categories(capsys):
    from etl.bench.__main__ import main

    # Names, categories, and tags mix: "grad,large" now expands via TAGS to
    # 17 examples (11 grad_* + 6 large) — the union of both tag sets.
    assert (
        main(
            [
                "--conformance", "--no-benchmark", "--no-torch",
                "--examples", "grad,large",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "grad_mlp" in captured.out
    assert "transformer" in captured.out
    assert "matmul_1024" in captured.out
    assert "conv2d_large" in captured.out
    # micro-only examples are excluded from the grad/large tag sets (assert on
    # absent names — "grad_cumsum" contains the "cumsum" substring, so that
    # name can't be used for an absence check).
    assert "softmax" not in captured.out
    assert "attention" not in captured.out


def test_main_in_process_tag_selection(capsys):
    from etl.bench.__main__ import main

    # "control-flow" is a TAG expanding to its 12 cond/while/e2e examples
    # (none of which is a plain micro example). Measured runtime ≈ 2 s for 12
    # small examples on the numpy backend.
    assert (
        main(
            [
                "--conformance", "--no-benchmark", "--no-torch",
                "--examples", "control-flow",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "cond_basic" in captured.out
    assert "while_fib" in captured.out
    assert "matmul" not in captured.out


def test_main_in_process_usage_error(capsys):
    from etl.bench.__main__ import main

    assert (
        main(
            [
                "--conformance", "--no-benchmark", "--no-torch",
                "--examples", "nope",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "etl.bench: error:" in captured.err
    assert "nope" in captured.err
