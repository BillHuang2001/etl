"""Shared IR-normalization helper for the enp (etl.numpy) test files."""

from __future__ import annotations

import re

# Per-op location tokens are call-site dependent (file:line:col differ
# between the enp package files and the test files) — strip them per line.
_LOC_RE = re.compile(r"\s+loc\(.*?\)\s*$")


def normalize_ir(text: str) -> str:
    """Strip per-op `` loc("file":line:col)`` tokens from pretty-printed IR.

    The regex ``\s+loc\(.*?\)\s*$`` is applied to each line, before removing
    leading/trailing whitespace, because loc tokens differ between enp
    callsites (etl/numpy/*.py) and ops-composed defns (the test files).
    Everything else in the IR must match character-for-character.
    """
    return "\n".join(_LOC_RE.sub("", line) for line in text.splitlines()).strip()
