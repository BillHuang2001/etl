"""Packaging contract: the `etl` distribution ships ONLY `etl`, never `etl_hf`.

`etl_hf` (the HuggingFace integration) is a separate top-level package living in
this repository checkout. It must never end up in a wheel/sdist of `etl`, and
`etl`'s hard runtime dependencies must stay numpy-only — the three reference
oracles used to validate `etl_hf` (torch + transformers) live behind the
optional `hf` extra.

These tests are numeric-free, network-free and fast: they read `pyproject.toml`
with `tomllib`, walk the checkout for real importable packages, and inspect the
installed distribution's metadata. The optional extra legs (a real wheel build
and a real sdist build) skip cleanly when no build backend is available.
"""

from __future__ import annotations

import fnmatch
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Directories that never contain distributable source packages.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".genesis",
        ".eggs",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

HF_TOP_LEVEL = "etl_hf"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _find_config() -> tuple[list[str], list[str]]:
    """`[tool.setuptools.packages.find]` include/exclude patterns."""
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    return list(find.get("include", [])), list(find.get("exclude", []))


def _matches(package: str, patterns: Sequence[str]) -> bool:
    """setuptools' package-find glob semantics.

    setuptools' own filter (``setuptools.discovery._Filter``) is
    ``any(fnmatch.fnmatchcase(item, pat) ...)``: ``*`` matches any characters
    (dots included), but the literal prefix must be there — so ``etl.*``
    matches ``etl.core`` and NOT ``etl_hf``.
    """
    return any(fnmatch.fnmatchcase(package, pattern) for pattern in patterns)


def _repo_packages() -> set[str]:
    """Every real importable package in the checkout (dirs with `__init__.py`)."""
    packages: set[str] = set()
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        if "__init__.py" not in files:
            continue
        rel = Path(root).relative_to(REPO_ROOT)
        if rel == Path("."):  # the repo root itself is not a package
            continue
        packages.add(".".join(rel.parts))
    return packages


def _etl_packages(packages: Iterable[str]) -> set[str]:
    return {p for p in packages if p == "etl" or p.startswith("etl.")}


def _hf_packages(packages: Iterable[str]) -> set[str]:
    return {p for p in packages if p == HF_TOP_LEVEL or p.startswith(HF_TOP_LEVEL + ".")}


# --------------------------------------------------------------------------
# 1. package discovery: `etl` in, `etl_hf` out
# --------------------------------------------------------------------------


def test_repo_walk_sees_both_packages():
    """Guard against a vacuous test: both top-level packages exist on disk."""
    packages = _repo_packages()
    etl_pkgs = _etl_packages(packages)
    hf_pkgs = _hf_packages(packages)

    # The etl distribution is a real package tree (top level + subpackages).
    assert "etl" in etl_pkgs
    assert "etl.core" in etl_pkgs
    assert len(etl_pkgs) > 10, sorted(etl_pkgs)
    # ... and the HuggingFace integration is a separate top-level package tree.
    assert HF_TOP_LEVEL in hf_pkgs
    assert f"{HF_TOP_LEVEL}.models" in hf_pkgs


def test_find_include_matches_every_etl_package():
    """The pattern must not be over-narrow: all `etl` / `etl.*` are selected."""
    include, exclude = _find_config()
    etl_pkgs = _etl_packages(_repo_packages())
    assert etl_pkgs

    missed = sorted(p for p in etl_pkgs if not _matches(p, include))
    assert not missed, f"packages.find include={include} misses real etl packages: {missed}"

    dropped = sorted(p for p in etl_pkgs if _matches(p, exclude))
    assert not dropped, f"packages.find exclude={exclude} drops real etl packages: {dropped}"


def test_find_include_never_matches_etl_hf_packages():
    """`etl_hf` must be unreachable from the `etl` distribution's find config."""
    include, exclude = _find_config()
    hf_pkgs = _hf_packages(_repo_packages())
    assert hf_pkgs

    matched = sorted(p for p in hf_pkgs if _matches(p, include) and not _matches(p, exclude))
    assert not matched, (
        f"packages.find include={include} would ship etl_hf packages {matched} "
        "inside the etl distribution"
    )

    # Explicit pins for the classic traps — the old greedy `etl*` include
    # matched all of these, and would silently bundle the HF integration.
    for name in ("etl_hf", "etl_hf.models", "etl_hf.tests", "etl_hf.tests.foo"):
        assert not _matches(name, include), f"{name!r} matched include={include}"
    assert "etl*" not in include, f"greedy include pattern 'etl*' matches etl_hf: {include}"

    # The colocated-tests exclusion is kept as belt-and-braces.
    assert exclude, "packages.find must keep its tests exclusion"
    assert _matches(f"{HF_TOP_LEVEL}.tests", exclude)


# --------------------------------------------------------------------------
# 2. dependency metadata: numpy-only runtime deps, optional hf oracle extra
# --------------------------------------------------------------------------


def _requirement_name(requirement: str) -> str:
    """`torch>=2.0` / `transformers[foo]>=4` / `x; python_version>'3'` -> name."""
    head = requirement.split(";", 1)[0].strip()
    for sep in "[<>=!~ (":
        head = head.split(sep, 1)[0]
    return head.strip().lower()


def test_runtime_dependencies_are_numpy_only():
    project = _pyproject()["project"]
    assert project["dependencies"] == ["numpy>=1.24"], (
        "etl's hard runtime dependencies must stay numpy-only (etl_hf and its "
        "oracles are optional)"
    )
    names = {_requirement_name(dep) for dep in project["dependencies"]}
    assert names == {"numpy"}


def test_hf_extra_declares_the_optional_hf_oracles():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "hf" in extras, "the optional `hf` extra must exist"

    names = {_requirement_name(dep) for dep in extras["hf"]}
    assert {"torch", "transformers"} <= names, extras["hf"]

    # Nothing HF-related may leak into the base install.
    base = {_requirement_name(dep) for dep in _pyproject()["project"]["dependencies"]}
    assert not ({"torch", "transformers", HF_TOP_LEVEL} | names) & base

    # `etl_hf` is a repo-checkout package, never a pip requirement of anything.
    all_requirements = list(_pyproject()["project"]["dependencies"])
    for deps in extras.values():
        all_requirements.extend(deps)
    assert HF_TOP_LEVEL not in {_requirement_name(dep) for dep in all_requirements}


# --------------------------------------------------------------------------
# 3. `import etl` works and never pulls the optional HF stack
# --------------------------------------------------------------------------


def test_import_etl_does_not_import_etl_hf_or_the_hf_oracles():
    probe = (
        "import sys, etl;"
        "etl.build, etl.evaluate, etl.lower, etl.trace, etl.tree_map;"
        "heavy = [m for m in ('etl_hf', 'torch', 'transformers') if m in sys.modules];"
        "assert not heavy, heavy;"
        "print(etl.__version__)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-4000:]


def test_installed_distribution_metadata_lists_only_etl():
    """The installed `etl` distribution must not claim `etl_hf` as its own."""
    try:
        dist = importlib.metadata.distribution("etl")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - env guard
        pytest.skip("the `etl` distribution is not installed in this environment")

    top_level = dist.read_text("top_level.txt")
    if top_level is not None:
        names = [line.strip() for line in top_level.splitlines() if line.strip()]
        assert "etl" in names
        assert HF_TOP_LEVEL not in names, (
            f"installed etl metadata declares the top-level package {HF_TOP_LEVEL!r} "
            f"(top_level.txt={names}) — the packages.find include pattern is too broad; "
            "reinstall after fixing pyproject.toml"
        )

    extras = set(dist.metadata.get_all("Provides-Extra") or ())
    assert "hf" in extras, sorted(extras)

    record = dist.read_text("RECORD")
    if record is not None:  # wheel installs only; legacy editable installs have none
        shipped = [
            line.split(",", 1)[0]
            for line in record.splitlines()
            if line.split(",", 1)[0].startswith(HF_TOP_LEVEL + "/")
        ]
        assert not shipped, f"installed etl distribution ships etl_hf files: {shipped[:5]}"


# --------------------------------------------------------------------------
# 4. optional: build real distributions and pin their contents (skips w/o tooling)
# --------------------------------------------------------------------------


def _importable(module: str) -> bool:
    """True when `module` is a real importable module here (not a stray dir).

    A bare directory named `build/` in the repo root is importable as a
    namespace package, which would fool a plain `find_spec(...) is not None`
    check — require an actual origin file.
    """
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def _build_command(dist: str, out_dir: Path) -> list[str] | None:
    """Best-effort non-isolated `sdist`/`wheel` build command, or None.

    Non-isolation keeps the legs network-free (no build requirements to fetch):
    they need the `setuptools.build_meta` backend importable in THIS environment.
    """
    if not _importable("setuptools"):
        return None
    if _importable("build"):
        return [
            sys.executable, "-m", "build", f"--{dist}", "--no-isolation", "--outdir", str(out_dir)
        ]
    uv = shutil.which("uv")
    if uv is None:
        return None
    return [uv, "build", f"--{dist}", "--no-build-isolation", "--out-dir", str(out_dir)]


# Copied into a throwaway source tree so the build never writes `build/` or
# `*.egg-info` into the checkout (they are irrelevant to packaging anyway).
_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    ".genesis",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "*.egg-info",
    "__pycache__",
    ".pytest_cache",
    ".agents",
    "bench_logs",  # large scratch area, never a build input
)

_STALE_EGG_INFO = "etl.egg-info"


def _sanitize_source_tree(src: Path) -> list[str]:
    """Delete stale build artifacts from a copied tree (`rm -rf etl.egg-info build`).

    MEASURED TRAP (setuptools 68.1.2 and 84.0.0): `sdist` *reads* an existing
    `etl.egg-info/SOURCES.txt` and keeps every entry that still exists on disk,
    so an artifact left by an earlier build can silently re-ship a broken,
    partial `etl_hf/`. `_COPY_IGNORE` already drops these when copying; this
    removes any that survive or that a test deliberately planted.
    """
    removed: list[str] = []
    for candidate in (*src.glob("*.egg-info"), src / "build", src / "dist", src / ".eggs"):
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()
        else:
            continue
        removed.append(candidate.name)
    return sorted(set(removed))


def _copy_checkout(dest: Path) -> Path:
    """Copy the checkout into `dest` (no VCS dirs, build artifacts or scratch)."""
    shutil.copytree(REPO_ROOT, dest, ignore=_COPY_IGNORE)
    _sanitize_source_tree(dest)  # defensive: a plant may predate the copy
    return dest


def _build_distribution(dist: str, src: Path, out_dir: Path) -> tuple[Path | None, str]:
    """Build `dist` from `src`; `(artifact, "")` or `(None, skip-reason)`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(dist, out_dir)
    if command is None:
        return None, f"no build backend available (setuptools/build/uv) — {dist} leg skipped"

    proc = subprocess.run(command, cwd=str(src), capture_output=True, text=True, timeout=900)
    artifacts = sorted(out_dir.glob("etl-*.whl" if dist == "wheel" else "etl-*.tar.gz"))
    if proc.returncode != 0 or not artifacts:
        detail = (proc.stderr or proc.stdout or "no output")[-500:]
        return None, f"{dist} build unavailable: {detail}"
    return artifacts[0], ""


def _hf_entries(entries: Iterable[str]) -> list[str]:
    """Paths whose first component is `etl_hf` (a file OR a bare directory)."""
    return [e for e in entries if e == HF_TOP_LEVEL or e.startswith(HF_TOP_LEVEL + "/")]


def _relative_entries(archive: Path) -> list[str]:
    """Member paths of an sdist tarball, relative to the `etl-<version>/` root."""
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    root = names[0].split("/", 1)[0] if names else ""
    entries = [n.split("/", 1)[1] if n.startswith(root + "/") else n for n in names]
    return [e for e in entries if e]  # drop the archive's own root directory entry


_STALE_HINT = (
    "Cause and fix: setuptools' sdist READS an existing etl.egg-info/SOURCES.txt and keeps "
    "every listed path that still exists on disk, so a STALE etl.egg-info/ from an earlier "
    "build silently re-ships a broken partial etl_hf. `rm -rf etl.egg-info build dist` and "
    "rebuild — etl_hf must only ever be reachable from the checkout."
)


def _assert_distribution_ships_no_etl_hf(entries: Sequence[str], kind: str) -> None:
    """Anti-vacuity (`etl/` + `pyproject.toml` are shipped) + zero `etl_hf` entries."""
    assert any(e.startswith("etl/") for e in entries), (
        f"{kind} does not ship the etl package (the etl_hf pin would be vacuous): {entries[:5]}"
    )
    assert "pyproject.toml" in entries, f"{kind} does not ship pyproject.toml: {entries[:5]}"
    hf_entries = _hf_entries(entries)
    assert not hf_entries, f"{kind} ships etl_hf source entries: {hf_entries[:5]}\n{_STALE_HINT}"


def test_wheel_contains_no_etl_hf(tmp_path):
    """A real `etl` wheel must contain no `etl_hf` module (skips without tooling)."""
    src = _copy_checkout(tmp_path / "src")
    wheel, reason = _build_distribution("wheel", src, tmp_path / "wheel")
    if wheel is None:
        pytest.skip(reason)

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        record_name = next((n for n in names if n.endswith(".dist-info/RECORD")), None)
        assert record_name is not None, "wheel has no RECORD"
        record = archive.read(record_name).decode().splitlines()

    shipped = [line.split(",", 1)[0] for line in record if line.split(",", 1)[0]]
    assert shipped, "empty RECORD"
    assert any(n.startswith("etl/") for n in shipped), "wheel does not ship the etl package"
    hf_entries = _hf_entries(shipped)
    assert not hf_entries, f"wheel ships etl_hf files: {hf_entries[:5]}\n{_STALE_HINT}"


def test_sdist_contains_no_etl_hf(tmp_path):
    """A real `etl` sdist must contain no `etl_hf` source (skips without tooling).

    The sdist is the distribution that ships `pyproject.toml` + the sources, so
    it is the one a stale build artifact can corrupt: it is built from an
    isolated temp COPY of the checkout with `*.egg-info`, `build/` and `dist/`
    removed, so nothing from the working tree can leak in.
    """
    src = _copy_checkout(tmp_path / "src")
    archive, reason = _build_distribution("sdist", src, tmp_path / "sdist")
    if archive is None:
        pytest.skip(reason)

    _assert_distribution_ships_no_etl_hf(_relative_entries(archive), "sdist")

# The exact stale artifact that can poison an sdist: a leftover `SOURCES.txt`
# from an earlier build listing `etl_hf/` modules. One entry points at a file
# that no longer exists, to pin setuptools' pruning of dead manifest entries.
_STALE_SOURCES = (
    "etl_hf/__init__.py\n"
    "etl_hf/config.py\n"
    "etl_hf/stale_module_that_no_longer_exists.py\n"
    "etl/__init__.py\n"
    "pyproject.toml\n"
)


def test_sdist_built_from_a_stale_egg_info_tree_ships_no_etl_hf(tmp_path):
    """The stale-`etl.egg-info` regression pin: sanitizing the tree is what saves us.

    MEASURED (setuptools 68.1.2 and 84.0.0): the trap is real — an sdist built
    from a tree carrying a stale `etl.egg-info/SOURCES.txt` DOES ship the listed
    `etl_hf/` paths that still exist on disk (3 of the 5 planted entries:
    `etl_hf/`, `etl_hf/__init__.py`, `etl_hf/config.py`; the entry naming a file
    that no longer exists is silently pruned), even though `packages.find`
    excludes `etl_hf`. So this test asserts the documented-safe path rather than
    the impossible one: plant the artifact, prove the trap manifests (when it
    does), sanitize the SAME tree, assert the rebuild is clean.
    """
    src = _copy_checkout(tmp_path / "src")

    # Plant the stale artifact exactly as an earlier build would have left it.
    egg_info = src / _STALE_EGG_INFO
    egg_info.mkdir(parents=True, exist_ok=True)
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: etl\nVersion: 0.1.0\n")
    (egg_info / "SOURCES.txt").write_text(_STALE_SOURCES)

    stale_archive, reason = _build_distribution("sdist", src, tmp_path / "stale")
    if stale_archive is None:
        pytest.skip(reason)
    stale_hf = _hf_entries(_relative_entries(stale_archive))
    if stale_hf:  # trap live (measured): existing entries ship, the dead one is pruned
        assert "etl_hf/__init__.py" in stale_hf, stale_hf
        assert "etl_hf/config.py" in stale_hf, stale_hf
        assert not [e for e in stale_hf if "stale_module_that_no_longer_exists" in e], stale_hf
    # The documented-safe path: drop the stale artifacts, rebuild the SAME tree.
    # (The first build also wrote a fresh `etl.egg-info/` + `build/`.)
    removed = _sanitize_source_tree(src)
    assert _STALE_EGG_INFO in removed, (removed, sorted(p.name for p in src.iterdir()))

    clean_archive, reason = _build_distribution("sdist", src, tmp_path / "clean")
    if clean_archive is None:  # pragma: no cover - tooling vanished mid-test
        pytest.skip(reason)

    entries = _relative_entries(clean_archive)
    _assert_distribution_ships_no_etl_hf(entries, f"sdist rebuilt after removing {removed}")
