"""etl.bench torch-optional importability contract.

``import etl`` and ``import etl.bench`` MUST always succeed without torch and
must NOT pull torch into ``sys.modules`` (torch is imported lazily inside
function bodies only, via ``etl.bench._torch`` — never at module top level).
This holds in BOTH torch-present and torch-absent environments, so no skip
guards are needed here.
"""
from __future__ import annotations

import sys


def test_import_etl_bench_does_not_pull_in_torch():
    before = frozenset(sys.modules)
    import etl  # noqa: F401
    import etl.bench  # noqa: F401
    added = frozenset(sys.modules) - before
    assert not any(
        name == "torch" or name.startswith("torch.") for name in added
    ), f"importing etl.bench pulled torch into sys.modules: {sorted(added)}"
    # When torch was not already loaded by an earlier test in this process,
    # it must still be absent after the imports.
    if "torch" not in before:
        assert "torch" not in sys.modules


def test_list_examples_callable_after_import():
    import etl.bench
    assert callable(etl.bench.list_examples)
    assert isinstance(etl.bench.list_examples(), list)
