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
