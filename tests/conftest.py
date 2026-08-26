"""tests/ conftest — complements the root conftest.py (sys.path bootstrap)."""


def pytest_configure(config):
    """Work around pytest's default `norecursedirs` containing 'dist'.

    Without this, a plain `python3 -m pytest` from the repo root silently
    collects 0 tests from `tests/dist/`. The proper fix is adding
    `norecursedirs = ...` (without `dist`) under `[tool.pytest.ini_options]`
    in the root pyproject.toml — once that lands, this hook becomes a no-op
    and can be removed.
    """
    values = list(config.getini("norecursedirs"))
    if "dist" in values:
        config._inicache["norecursedirs"] = [v for v in values if v != "dist"]
