"""Guarded HuggingFace checkpoint downloader for the ``etl_hf`` test suite.

Fetches a real HF checkpoint (``config.json`` + weights) into a local cache with
plain ``urllib`` — stdlib only, so this module imports without ``numpy``, ``etl``,
``pytest``, ``torch`` or ``huggingface_hub`` present (pytest is imported lazily
inside :func:`require_model`; the CLI runs standalone).

Contract:

* **Guarded, never fatal.** Every network/HTTP/filesystem failure surfaces as
  :class:`DownloadUnavailable` (:class:`RuntimeError`) naming the URL and what was
  tried; :func:`require_model` turns that into ``pytest.skip`` so offline runs
  skip cleanly instead of failing or hanging.
* **Atomic.** Bodies stream to ``<dest>.part`` and are ``os.replace``-d into
  place, so an interrupted transfer can never leave a truncated weights file that
  a later run mistakes for a cache hit.  A cached file is trusted only when its
  size is ``> 0`` — plus, for ``*.safetensors``, when the 8-byte header length
  parses and fits inside the file.
* **Explicit.** No silent fallbacks: :func:`download_model` walks the documented
  weight-candidate order and raises ONE error listing every candidate that
  failed, so a half-downloaded directory is never reported as usable.

CLI::

    python etl_hf/tests/hf_download.py [--slugs tiny_llama,tiny_gpt2] [--root DIR] [--check]

Cache root: ``$ETL_HF_MODELS_DIR`` if set, else ``$TMPDIR/etl_hf_models``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "HF_BASE", "MODELS", "BIG_MODELS", "DownloadUnavailable",
    "models_dir", "fetch", "download_model", "is_available",
    "require_model", "ensure_models", "main",
]  # fmt: skip

HF_BASE = "https://huggingface.co"

#: Model slug -> HuggingFace repo id.  The tiny models are the default test set;
#: the real checkpoints are opt-in and listed in :data:`BIG_MODELS`.
MODELS: dict[str, str] = {
    "tiny_llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "tiny_mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
    "distilgpt2": "distilgpt2",
    "gpt2": "openai-community/gpt2",
}

#: Slugs large enough that tests gate them behind an environment variable.
BIG_MODELS: frozenset[str] = frozenset({"distilgpt2", "gpt2"})

#: Slugs the CLI downloads when ``--slugs`` is not given.
DEFAULT_SLUGS: tuple[str, ...] = ("tiny_llama", "tiny_mistral", "tiny_gpt2")

#: Weight files tried in order; ``*.index.json`` entries expand into their shards.
_WEIGHT_CANDIDATES: tuple[str, ...] = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_INDEX_CANDIDATES: tuple[str, ...] = tuple(n for n in _WEIGHT_CANDIDATES if n.endswith(".index.json"))
_USER_AGENT = "etl_hf-tests/0.1 (python-urllib)"
_CHUNK = 1 << 20
_BACKOFF_CAP = 2.0


class DownloadUnavailable(RuntimeError):
    """A checkpoint could not be fetched (network, HTTP, or on-disk problem).

    The message names the URL / model directory and everything that was tried;
    tests convert this into a skip via :func:`require_model`.
    """


def models_dir() -> Path:
    """Return the checkpoint cache root, creating it if needed.

    ``$ETL_HF_MODELS_DIR`` overrides it; otherwise it is ``$TMPDIR/etl_hf_models``
    (``tempfile.gettempdir()`` honours ``$TMPDIR``).
    """
    root = _root(None)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - exotic filesystem failure
        raise DownloadUnavailable(f"cannot create model cache {root}: {exc}") from None
    return root


def _root(root: str | os.PathLike[str] | None) -> Path:
    """Resolve the cache root without touching the filesystem."""
    if root is not None:
        return Path(root).expanduser()
    override = os.environ.get("ETL_HF_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir()) / "etl_hf_models"


def _size(path: Path) -> int:
    """Size in bytes, or ``0`` when the path is missing/unreadable."""
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _cleanup(path: Path) -> None:
    """Best-effort removal of a partial/temporary download."""
    try:
        path.unlink()
    except OSError:
        pass


def fetch(url: str, dest: str | os.PathLike[str], *, timeout: float = 120.0, retries: int = 2) -> Path:
    """Download ``url`` to ``dest`` atomically and return ``dest``.

    An existing ``dest`` with ``size > 0`` is a cache hit (no network access).
    Otherwise the body streams to ``<dest>.part`` and is ``os.replace``-d into
    place, retrying transient failures ``retries`` times with a short exponential
    backoff; HTTP 4xx (except 429) is permanent and fails at once.  Every failure
    raises :class:`DownloadUnavailable` naming the URL and the status/reason —
    including an empty response body.
    """
    dest = Path(dest)
    if _size(dest) > 0:
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadUnavailable(f"cannot create directory {dest.parent}: {exc}") from None
    tmp = dest.with_name(dest.name + ".part")
    attempts = max(0, int(retries)) + 1
    last = "no attempt made"
    for attempt in range(attempts):
        try:
            written = _download_once(url, tmp, timeout)
        except urllib.error.HTTPError as exc:  # subclass of URLError: must come first
            _cleanup(tmp)
            status, reason = getattr(exc, "code", None), getattr(exc, "reason", "") or ""
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise DownloadUnavailable(
                    f"cannot fetch {url}: HTTP {status} {reason} "
                    "(the repo may not exist, be gated/require auth, or not ship this file)"
                ) from None
            last = f"HTTP {status} {reason}".strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            _cleanup(tmp)
            last = f"network unavailable: {exc}"
        else:
            if written == 0:
                _cleanup(tmp)
                last = "empty response body"
                continue
            try:
                os.replace(tmp, dest)
            except OSError as exc:
                _cleanup(tmp)
                raise DownloadUnavailable(f"cannot write {dest}: {exc}") from None
            return dest
        if attempt + 1 < attempts:
            time.sleep(min(_BACKOFF_CAP, 0.5 * (2**attempt)))
    raise DownloadUnavailable(f"{url}: {last} (after {attempts} attempt(s))")


def _download_once(url: str, tmp: Path, timeout: float) -> int:
    """Stream ``url`` into ``tmp``; return the number of bytes written."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as handle:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                handle.write(chunk)
                written += len(chunk)
    except BaseException:
        _cleanup(tmp)
        raise
    return written


def _repo_and_dir(slug: str, root: str | os.PathLike[str] | None = None) -> tuple[str, Path]:
    """Map a slug (or a raw ``org/name`` repo id) to ``(repo_id, target_dir)``."""
    if not isinstance(slug, str) or not slug.strip():
        raise ValueError(f"model slug must be a non-empty string, got {slug!r}")
    key = slug.strip()
    repo = MODELS.get(key)
    if repo is None:
        parts = key.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"unknown model slug {slug!r}: pass a key of hf_download.MODELS "
                f"({', '.join(sorted(MODELS))}) or a raw 'org/name' HuggingFace repo id"
            )
        repo = key
    return repo, _root(root) / key.replace("/", "__").replace(os.sep, "__").replace("..", "_")


def _safetensors_ok(path: Path) -> bool:
    """Whether ``path`` is a complete safetensors file.

    Validates the 8-byte little-endian header length, that the JSON header it
    announces fits inside the file, and that the header parses — no tensor reads.
    """
    size = _size(path)
    if size < 12:
        return False
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                return False
            header_len = int.from_bytes(head, "little")
            if header_len < 2 or 8 + header_len > size:
                return False
            raw = handle.read(header_len)
        return isinstance(json.loads(raw.decode("utf-8")), dict)
    except (OSError, UnicodeDecodeError, ValueError):
        return False


def _index_shards(index_path: Path) -> list[str]:
    """The weight-file names listed in a shard index's ``weight_map``."""
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DownloadUnavailable(f"cannot read shard index {index_path}: {exc}") from None
    except ValueError as exc:
        raise DownloadUnavailable(f"shard index {index_path} is not valid JSON: {exc}") from None
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise DownloadUnavailable(f"shard index {index_path} carries no usable 'weight_map'")
    return sorted({str(name) for name in weight_map.values()})


def _fetch_file(base: str, target: Path, name: str) -> Path:
    """Fetch one repo file (cached when complete); validate ``*.safetensors``."""
    dest = target / name
    if name.endswith(".safetensors") and _size(dest) > 0 and not _safetensors_ok(dest):
        _cleanup(dest)  # stale/truncated cache entry: re-download below
    fetch(f"{base}/{name}", dest)
    if name.endswith(".safetensors") and not _safetensors_ok(dest):
        _cleanup(dest)
        raise DownloadUnavailable(f"{base}/{name}: downloaded file is not a valid safetensors file")
    return dest


def _fetch_index(base: str, target: Path, name: str) -> list[Path]:
    """Fetch a shard index plus every shard named in its ``weight_map``."""
    index_path = target / name
    shards = None
    if _size(index_path) > 0:
        try:
            shards = _index_shards(index_path)
        except DownloadUnavailable:
            _cleanup(index_path)  # truncated/stale cached index: re-download below
    if shards is None:
        fetch(f"{base}/{name}", index_path)
        shards = _index_shards(index_path)
    return [_fetch_file(base, target, shard) for shard in shards]


def download_model(slug: str, *, root: str | os.PathLike[str] | None = None, revision: str = "main") -> Path:
    """Download one checkpoint into ``root/<slug>`` and return that directory.

    ``slug`` is a key of :data:`MODELS` or a raw ``org/name`` repo id.  Fetches
    ``config.json`` first, then the weight candidates in order —
    ``model.safetensors``, ``model.safetensors.index.json`` (+ its shards),
    ``pytorch_model.bin``, ``pytorch_model.bin.index.json`` (+ its shards) — and
    raises :class:`DownloadUnavailable` when ``config.json`` or every candidate
    fails; a half-downloaded directory is never returned as usable.
    """
    repo, target = _repo_and_dir(slug, root)
    label = repr(slug) if repo == slug else f"{slug!r} ({repo})"
    base = f"{HF_BASE}/{repo}/resolve/{revision}"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadUnavailable(f"cannot create model directory {target}: {exc}") from None
    try:
        fetch(f"{base}/config.json", target / "config.json")
    except DownloadUnavailable as exc:
        raise DownloadUnavailable(f"{label}: config.json unavailable — {exc}") from None
    failures: list[str] = []
    for name in _WEIGHT_CANDIDATES:
        try:
            _fetch_index(base, target, name) if name in _INDEX_CANDIDATES else _fetch_file(base, target, name)
        except DownloadUnavailable as exc:
            failures.append(f"{name}: {exc}")
            continue
        return target
    raise DownloadUnavailable(f"{label}: no usable weights in {target} — " + " | ".join(failures))


def is_available(slug: str, *, root: str | os.PathLike[str] | None = None) -> bool:
    """Whether a complete-enough checkpoint is already cached (no network access).

    True when ``config.json`` is present and either a shard index with all of its
    non-empty shards, or a non-empty single-file weight payload, is on disk.
    """
    try:
        _, target = _repo_and_dir(slug, root)
    except ValueError:
        return False
    if _size(target / "config.json") == 0:
        return False
    for name in _INDEX_CANDIDATES:
        if _size(target / name) == 0:
            continue
        try:
            shards = _index_shards(target / name)
        except DownloadUnavailable:
            continue
        if all(_size(target / shard) > 0 for shard in shards):
            return True
    return any(_size(target / name) > 0 for name in ("model.safetensors", "pytorch_model.bin"))


def require_model(slug: str, *, root: str | os.PathLike[str] | None = None) -> Path:
    """Return the model directory, or ``pytest.skip`` when it cannot be fetched.

    ``pytest`` is imported here — never at module scope — so this module stays
    importable without the test framework (e.g. from the CLI).
    """
    try:
        return download_model(slug, root=root)
    except DownloadUnavailable as exc:
        import pytest

        pytest.skip(f"HF checkpoint {slug!r} unavailable: {exc}")


def ensure_models(slugs, *, root: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Download every slug, returning ``{slug: dir}``.

    Failures are collected and re-raised as ONE :class:`DownloadUnavailable`
    listing both the successes and the failures; a partial dict is never returned.
    """
    done: dict[str, Path] = {}
    failures: list[str] = []
    for slug in dict.fromkeys(slugs):
        try:
            done[slug] = download_model(slug, root=root)
        except (DownloadUnavailable, ValueError) as exc:
            failures.append(f"{slug}: {exc}")
    if failures:
        raise DownloadUnavailable(
            f"failed to fetch {len(failures)} of {len(failures) + len(done)} checkpoint(s) — "
            f"succeeded: {', '.join(sorted(done)) or 'none'}; " + " | ".join(failures)
        )
    return done


def main(argv=None) -> int:
    """CLI entry point: download (or ``--check``) the requested checkpoints.

    Prints one line per model with its directory, weight-file count and total
    size in MB.  ``--check`` reports cached availability only (no network) and
    exits non-zero when any requested model is missing.
    """
    parser = argparse.ArgumentParser(
        prog="hf_download.py", description="Fetch guarded HuggingFace checkpoints for the etl_hf test suite."
    )
    parser.add_argument(
        "--slugs",
        default=",".join(DEFAULT_SLUGS),
        help=f"comma-separated slugs or repo ids (default: {','.join(DEFAULT_SLUGS)})",
    )
    parser.add_argument("--root", default=None, help="cache root (default: $ETL_HF_MODELS_DIR or $TMPDIR/etl_hf_models)")
    parser.add_argument("--check", action="store_true", help="only report what is already cached (no network)")
    args = parser.parse_args(argv)
    slugs = list(dict.fromkeys(part.strip() for part in args.slugs.split(",") if part.strip()))
    if not slugs:
        parser.error("--slugs must list at least one slug")
    root = Path(args.root).expanduser() if args.root else None

    if args.check:
        missing = 0
        for slug in slugs:
            available = is_available(slug, root=root)
            missing += 0 if available else 1
            print(f"{slug:14s} {'available' if available else 'MISSING':9s} {_repo_and_dir(slug, root)[1]}")
        print(f"{len(slugs) - missing}/{len(slugs)} available under {_root(root)}")
        return 0 if missing == 0 else 1

    failures = 0
    for slug in slugs:
        try:
            target = download_model(slug, root=root)
        except (DownloadUnavailable, ValueError) as exc:
            failures += 1
            print(f"{slug:14s} FAILED    {exc}")
            continue
        files = [p for p in target.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin")]
        total_mb = sum(_size(path) for path in files) / (1024 * 1024)
        print(f"{slug:14s} {target}  {len(files)} weight file(s)  {total_mb:.2f} MB")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
