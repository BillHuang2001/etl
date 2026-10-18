"""Optional torch / transformers golden-oracle parity tests for ``etl_hf``.

The rest of this suite validates the compiled forward pass against ``etl_hf``'s
own independent pure-numpy reference (``etl_hf/reference.py``) — which cannot see
a canonicalisation bug *both* implementations share (a transposed weight, a
wrong ``q/k/v`` head split, an off-by-one RoPE offset, a tied head read
backwards).  The oracle here is therefore **external**: the real HuggingFace
implementation (``transformers`` + ``torch``), run on CPU.

``torch`` and ``transformers`` are optional, test-only dependencies — ``etl_hf``
itself never imports them (pinned in ``tests/test_public_api.py`` and by
:func:`test_etl_hf_never_imports_torch` below).  **Nothing optional is imported
at module scope here either**: :func:`_oracle` imports the pair on first use and
caches it, while the ``skipif`` gates use :func:`_oracle_available`, which only
probes ``importlib.util.find_spec``.  So in this repository's default
environment (numpy only) the oracle-dependent cases skip cleanly **and** the
torch-free ones still collect and run: :func:`test_etl_hf_never_imports_torch`,
:func:`test_golden_fixture_format`, :func:`test_golden_slugs_resolve`, and the
fixture parity :func:`test_torch_goldens_parity` (comparing against pre-computed
HF logits needs no torch at all — the goldens are committed under
``bench_logs/hf_goldens/``, so it runs whenever that directory exists and
skips only when a fixture or its checkpoint is genuinely unavailable, never
because torch is missing).

Golden fixture contract
-----------------------
A separate torch-golden harness writes logits produced by the *real* HF model
into ``bench_logs/hf_goldens/`` — a scratch/results directory outside this
package, read-only from here (nothing in this file ever writes it).  Two files
per model, ``<slug>`` naming the model::

    bench_logs/hf_goldens/<slug>.npz
        input_ids   int32    (batch, seq_len)              # ids fed to the HF model
        logits      float32  (batch, seq_len, vocab_size)   # that model's own output
    bench_logs/hf_goldens/<slug>.meta.json
        {"model_id": "<hf repo id>", "revision": "<sha|branch>", "dtype": "<torch dtype>"}
        # a "model": "<hf repo id>" key is accepted in place of "model_id"

* ``input_ids`` and ``logits`` are the two **required** array names, written with
  ``numpy.savez`` and readable with ``allow_pickle=False`` (never pickled
  objects, never torch tensors).
* ``input_ids`` is 2-D ``(batch, seq_len)`` int32; ``logits`` is float32 and
  either 3-D ``(batch, seq_len, vocab_size)`` or 2-D
  ``(batch * seq_len, vocab_size)``.  The leading axes must agree, and the
  dtype must be float32 so a downcast never dominates the comparison.
* The meta's repo id lives under **either** ``model`` **or** ``model_id``
  (:data:`_META_REPO_KEYS`; ``model`` wins when a fixture somehow carries both)
  and must be a non-empty string naming a HuggingFace repo id.  The harness that
  produced today's fixtures writes ``model_id``, so both keys are accepted —
  a producer never has to rename its meta to satisfy this contract.
* ``<slug>`` is resolved to a checkpoint in this order: the fixture's
  ``meta["model"]``, then ``meta["model_id"]``, then
  :data:`hf_download.MODELS`, then the slug itself (assumed to already be an
  ``org/name`` repo id) — see :func:`_repo_for_slug`.  All five current slugs
  are keys of ``hf_download.MODELS``; the harness spells two of them after
  their HF repo (``tiny_random_llama``, ``tiny_random_mistral``), which
  ``hf_download`` carries as :data:`hf_download.ALIASES` of
  ``tiny_llama``/``tiny_mistral`` so both spellings share one cached download.
* Resolution is **strict**: a discovered golden that cannot be mapped onto a
  fetchable checkpoint is a FAILURE, never a silent skip.
  :func:`test_golden_slugs_resolve` fails without any network access, and
  :func:`test_torch_goldens_parity` fails with the offending
  ``<slug>.meta.json`` path.  The only tolerated skip is a genuine download
  failure (:class:`hf_download.DownloadUnavailable`: offline host, missing or
  gated repo) — a missing *fixture directory* skips too, because it means the
  golden harness has not run here yet.
* ``revision``/``dtype`` are informational only (extra keys are ignored) —
  ``dtype`` records the torch dtype the goldens were produced with, and the
  stored ``logits`` must still be float32.

Missing fixtures are never an error: the golden-parity test skips with a message
naming this contract, while :func:`test_golden_fixture_format` validates the
contract strictly whenever the directory *does* exist (that is how the contract
is enforced on the producer).

Tolerances (and why)
--------------------
* tiny random checkpoints ≤ **1e-4 abs** — the feasibility spike measured
  ~1e-7…1e-8 for etl-numpy vs a pure-numpy reference, so 1e-4 leaves room only
  for HF's different accumulation order / kernel choices, not for a real bug.
  The two alias slugs are the same tiny random checkpoints under the harness'
  spelling, whose own logit absmax is 0.35 (``tiny_random_llama``) / 0.56
  (``tiny_random_mistral``), so they get the tiny budget as well —
  :func:`_tolerances` treats ANY ``tiny*`` slug as tiny, and :data:`_TINY_SLUGS`
  lists the tiny fixtures explicitly so the documented set matches the
  fixtures on disk.
* real-size checkpoints (``distilgpt2``/``gpt2``) ≤ **1e-3 abs** *and* ≤ **1e-5
  relative to the logit scale** — their fp32 logits are O(100), where ~1e-3 abs
  is ~1e-5 relative (measured 1.7e-4…9.1e-4 across iree/numpy runs in
  ``bench_logs/hf_feasibility_findings.md``).  Both are asserted, so a
  large-magnitude logit cannot buy tolerance for a wrong one.
* ``sshleifer/tiny-gpt2``'s logits are ordinary (``max |logit| ≈ 0.13``), so it
  is held to the same 1e-4 absolute budget as the other tiny checkpoints — the
  legacy-format checkpoint is *not* a known failure (see below).

Legacy-format checkpoints (why ``tiny_gpt2`` matters)
----------------------------------------------------
``sshleifer/tiny-gpt2`` ships the **legacy** (pre-torch-1.6)
``pytorch_model.bin`` layout, whose trailer prefixes every storage blob with an
int64 element count.  ``etl_hf.weights`` consumes that prefix, so its 33 tensors
agree with ``torch.load`` and the in-process parity case is an ordinary case
(measured at ``seq_len=8``: max-abs 2.2e-8 against a logit scale of 0.128, argmax
identical at every non-tie position).  Keeping it in the in-process set is the
reader's end-to-end check against an **external** oracle: a shifted parse still
yields correctly-shaped, plausible-looking tensors that agree with ``etl_hf``'s
own numpy reference (both sides read the same loaded arrays), so only a
comparison like this one can catch it.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:  # pragma: no cover - pytest already adds it
    sys.path.insert(0, str(_TESTS_DIR))

import hf_download  # noqa: E402  (needs the sys.path entry above)

# ---------------------------------------------------------------------------
# the optional oracle (never imported at module scope — see the docstring)
# ---------------------------------------------------------------------------
#: Reason attached to every ``skipif`` that needs the real HF implementation.
_ORACLE_SKIP_REASON = (
    "torch + transformers are not installed (optional HuggingFace golden oracle)"
)

#: Lazily populated ``(torch, transformers)`` pair — see :func:`_oracle`.
_ORACLE: tuple[object, object] | None = None


def _import_oracle():
    """Import ``(torch, transformers)``, turning a missing oracle into a skip."""
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"{_ORACLE_SKIP_REASON}: {exc}")
    return torch, transformers


def _oracle():
    """The cached ``(torch, transformers)`` pair, imported on first use.

    Call this *inside* a test, never at module scope: the optional oracle must
    not be imported just because this file was collected.
    """
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = _import_oracle()
    return _ORACLE


def _oracle_available() -> bool:
    """Whether both oracle packages are importable — probes, never imports them."""
    for name in ("torch", "transformers"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):  # a broken/partial installation
            return False
    return True


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: Repository root, derived from this file — never from the process' cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the parallel torch-golden harness drops its fixtures (read-only here).
_GOLDENS_SUBPATH = Path("bench_logs") / "hf_goldens"

#: Slugs whose logits are tiny enough for the tight absolute budget: the tiny
#: random checkpoints this file names in both spellings (``tiny_llama`` /
#: ``tiny_mistral`` and the golden harness' ``tiny_random_llama`` /
#: ``tiny_random_mistral`` aliases — see :data:`hf_download.ALIASES`) plus
#: ``tiny_gpt2``.  :func:`_tolerances` ALSO treats any ``tiny*`` slug as tiny, so
#: a future tiny fixture inherits the budget without editing this set; the listed
#: members keep the documented set equal to the fixtures actually on disk.
_TINY_SLUGS = frozenset(
    {"tiny_llama", "tiny_mistral", "tiny_random_llama", "tiny_random_mistral", "tiny_gpt2"}
)

#: Tolerances — see the module docstring for the measurements behind them.
_TINY_ATOL = 1e-4
_REAL_ATOL = 1e-3
_REAL_RTOL = 1e-5

#: Meta keys that may carry the fixture's HuggingFace repo id, in precedence
#: order: ``model`` first, then ``model_id`` (the key the golden harness writes).
_META_REPO_KEYS: tuple[str, ...] = ("model", "model_id")

#: The sequence length used for the in-process (no-fixture) parity test.
_SEQ_LEN = 8

#: A top-1/top-2 logit gap below this is a near-tie: argmax equality is not
#: asserted there (two fp32 implementations may legitimately disagree).
_TOP2_GAP = 1e-6

#: Short restatement of the fixture layout, reused in assertion/skip messages so
#: a producer never has to read this file to learn the contract.
_GOLDENS_CONTRACT = (
    "<repo>/" + str(_GOLDENS_SUBPATH) + ": one <slug>.npz per model holding "
    "`input_ids` (int32, (batch, seq_len)) and `logits` (float32, "
    "(batch, seq_len, vocab_size) or (batch*seq_len, vocab_size)), plus a sibling "
    '<slug>.meta.json carrying either {"model": "<hf repo id>"} or '
    '{"model_id": "<hf repo id>"} (model wins when both exist)'
)


# ---------------------------------------------------------------------------
# golden fixtures (pure numpy: no torch involved, so these stay testable)
# ---------------------------------------------------------------------------
def _goldens_dir() -> Path:
    """Absolute path of the golden-fixture directory (may not exist)."""
    return _REPO_ROOT / _GOLDENS_SUBPATH


def _read_meta(npz_path: Path) -> dict:
    """Read ``<slug>.meta.json`` next to ``npz_path``; ``{}`` when unreadable."""
    meta_path = npz_path.with_suffix(".meta.json")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_goldens(directory: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, dict]]:
    """Load every golden fixture, keyed by slug.

    Args:
        directory: fixture directory (default: :func:`_goldens_dir`).

    Returns:
        dict: ``slug -> (input_ids, logits, meta)``.  Empty when the directory
        is absent; a *malformed* fixture (unreadable zip, missing array) is
        skipped silently here — :func:`test_golden_fixture_format` is the strict
        contract check that reports it.
    """
    root = _goldens_dir() if directory is None else Path(directory)
    goldens: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    if not root.is_dir():
        return goldens
    for path in sorted(root.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                input_ids = np.asarray(archive["input_ids"])
                logits = np.asarray(archive["logits"])
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            continue
        goldens[path.stem] = (input_ids, logits, _read_meta(path))
    return goldens


def _meta_repo_id(meta: dict) -> str | None:
    """The fixture's HuggingFace repo id carried by its meta, else ``None``.

    Reads :data:`_META_REPO_KEYS` in precedence order — ``model`` first, then
    ``model_id`` (the key the golden harness writes) — and returns the first
    non-empty string, stripped.  A missing, empty or non-string value yields
    ``None``, leaving the decision to the caller: :func:`_repo_for_slug` falls
    back to the slug mapping, while :func:`test_golden_fixture_format` fails.
    """
    for key in _META_REPO_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _repo_for_slug(slug: str, meta: dict | None = None) -> str:
    """Resolve the HuggingFace repo id for a fixture slug.

    Precedence (:data:`_META_REPO_KEYS`, then the slug mapping):

    1. the fixture's ``meta["model"]``,
    2. the fixture's ``meta["model_id"]`` — the key the golden harness writes,
    3. :data:`hf_download.MODELS` (which carries the golden-harness aliases
       ``tiny_random_llama``/``tiny_random_mistral``),
    4. the slug itself, assumed to already be an ``org/name`` repo id.

    The result is whatever the fixture/downloader says it is — validating that
    it names a *fetchable* checkpoint is :func:`test_golden_slugs_resolve`'s and
    :func:`_checkpoint_dir`'s job, so this helper never guesses a repo.
    """
    repo_id = _meta_repo_id(meta or {})
    return repo_id if repo_id is not None else hf_download.MODELS.get(slug, slug)


def _checkpoint_dir(slug: str, repo_id: str) -> Path:
    """Local checkpoint directory for golden ``slug``.

    Resolution is STRICT: a golden that cannot be mapped onto a fetchable
    checkpoint is an assertion failure naming its ``.meta.json`` path and
    restating the contract — never a silent skip, which would let an unmappable
    fixture pass unnoticed.

    * A key of :data:`hf_download.MODELS` (including the harness' aliases) goes
      through :func:`hf_download.require_model`, which skips ONLY on a genuine
      :class:`hf_download.DownloadUnavailable` (offline host / missing repo).
    * Any other slug is fetched through its :func:`_repo_for_slug` repo id, with
      the same single tolerated skip.
    * A ``ValueError`` from the downloader means ``repo_id`` is not a fetchable
      repo id at all (not an ``org/name`` pair) — that is a broken fixture, so it
      is raised as an :class:`AssertionError`.
    """
    if slug in hf_download.MODELS:
        return hf_download.require_model(slug)
    try:
        return hf_download.download_model(repo_id)
    except hf_download.DownloadUnavailable as exc:
        pytest.skip(f"checkpoint for golden {slug!r} ({repo_id}) unavailable: {exc}")
    except ValueError as exc:
        raise AssertionError(
            f"golden {slug!r} cannot be mapped onto a fetchable checkpoint: {exc}. "
            f"Either add {slug!r} to hf_download.MODELS, or give "
            f"{_goldens_dir() / (slug + '.meta.json')} a 'model'/'model_id' "
            f"'org/name' HuggingFace repo id — {_GOLDENS_CONTRACT}"
        ) from None


# ---------------------------------------------------------------------------
# the HuggingFace oracle (lazy imports: torch/transformers never at module scope)
# ---------------------------------------------------------------------------
def _torch_dtype(name: object):
    """Map a fixture ``dtype`` name onto a ``torch.dtype`` (explicit, no guessing)."""
    torch, _transformers = _oracle()
    if name is None:
        return torch.float32
    if isinstance(name, torch.dtype):
        return name
    key = str(name).strip().lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(
            f"golden fixture asks for dtype {name!r}, which this test does not map "
            f"onto a torch dtype; supported: {sorted(mapping)}"
        ) from None


def _hf_source(repo_id: str) -> str:
    """Prefer a complete local download of ``repo_id``, else the hub repo id."""
    for slug, repo in hf_download.MODELS.items():
        if repo != repo_id:
            continue
        try:
            return str(hf_download.download_model(slug))
        except hf_download.DownloadUnavailable:
            break  # offline / not cacheable: let transformers resolve it itself
    return repo_id


def _hf_logits(
    repo_id: str, input_ids: object, dtype: object = "float32", *, source: object = None
) -> np.ndarray:
    """Run the REAL HuggingFace causal-LM forward on CPU and return float32 logits.

    Args:
        repo_id: HuggingFace repo id (used in messages and for resolution).
        input_ids: ``(batch, seq_len)`` token ids (cast to ``torch.long``).
        dtype: torch dtype name for the model load (default ``"float32"``).
        source: local checkpoint directory overriding the default resolution
            (a complete local download, else ``repo_id`` via the hub).

    Returns:
        np.ndarray: float32 logits, ``(batch, seq_len, vocab_size)``.

    Raises:
        RuntimeError: the model could not be loaded or is not a causal LM —
            wrapped with the repo id/source so an unsupported architecture
            fails loudly instead of silently substituting a different model.
        ValueError: the requested dtype is not mappable.
    """
    torch, transformers = _oracle()

    torch_dtype = _torch_dtype(dtype)
    location = _hf_source(repo_id) if source is None else str(source)
    try:
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                location, torch_dtype=torch_dtype, trust_remote_code=False
            )
        except TypeError as exc:
            if "torch_dtype" not in str(exc):
                raise
            # transformers >= 4.56 renamed the kwarg to `dtype`; retry explicitly
            # (a silent fallback would hide a real argument error).
            model = transformers.AutoModelForCausalLM.from_pretrained(
                location, dtype=torch_dtype, trust_remote_code=False
            )
    except Exception as exc:
        raise RuntimeError(
            f"cannot run the HuggingFace oracle for {repo_id!r} ({location!r}): "
            f"{type(exc).__name__}: {exc}. This test pre-loads the checkpoint with "
            "transformers' AutoModelForCausalLM and never substitutes another "
            "implementation, so an unsupported architecture/unreadable checkpoint "
            "is an error rather than a silent mismatch"
        ) from exc

    model.eval()
    ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    return logits.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# comparison helpers (pure numpy: exercised even where torch is absent)
# ---------------------------------------------------------------------------
class _ParityReport(NamedTuple):
    """Measured agreement between our logits and the oracle's."""

    max_abs: float
    scale: float
    scale_rel: float
    rtol: float
    atol: float

    @property
    def within_abs(self) -> bool:
        """Whether the absolute budget is met."""
        return self.max_abs <= self.atol

    @property
    def within_rel(self) -> bool:
        """Whether the scale-relative budget is met."""
        return self.scale_rel <= self.rtol

    def describe(self, label: str) -> str:
        """One-line message carrying every measured number and both budgets."""
        return (
            f"{label}: etl_hf vs HuggingFace logits — max-abs {self.max_abs:.3e} "
            f"(budget {self.atol:.0e}); logit scale {self.scale:.3e} → relative "
            f"{self.scale_rel:.3e} (budget {self.rtol:.0e})"
        )


def _compare(got: object, want: object, *, rtol: float, atol: float) -> _ParityReport:
    """Measure the difference between our logits and the oracle's.

    Errors are computed in float64 (so the measurement itself cannot overflow or
    lose the digits being asserted on) and reported both absolutely and relative
    to the oracle's own logit scale — the relative figure is what makes a
    fp32-honest budget meaningful for large-magnitude logits.

    Args:
        got: our logits.
        want: the oracle's logits.
        rtol: relative budget (over the oracle's max |logit|).
        atol: absolute budget.

    Returns:
        _ParityReport: measured max-abs, oracle scale, scale-relative error and
        both budgets.

    Raises:
        ValueError: the two arrays do not have the same shape (assert that
            explicitly in the test: shapes are static here, so a mismatch is a
            graph-shape bug, never a tolerance question).
    """
    ours = np.asarray(got, dtype=np.float64)
    theirs = np.asarray(want, dtype=np.float64)
    if ours.shape != theirs.shape:
        raise ValueError(f"shape mismatch: etl_hf {ours.shape} vs HF {theirs.shape}")
    if ours.size == 0:
        return _ParityReport(0.0, 0.0, 0.0, rtol, atol)
    max_abs = float(np.max(np.abs(ours - theirs)))
    scale = float(np.max(np.abs(theirs)))
    if scale:
        scale_rel = max_abs / scale
    else:
        scale_rel = 0.0 if max_abs == 0.0 else float("inf")
    return _ParityReport(max_abs, scale, scale_rel, rtol, atol)


def _tolerances(slug: str) -> tuple[float, float, bool]:
    """``(rtol, atol, require_relative)`` for a fixture/user model slug.

    A slug is tiny when it is listed in :data:`_TINY_SLUGS` (the tiny fixtures,
    in both spellings) or when it starts with ``tiny`` — the prefix rule keeps
    every future ``tiny*`` fixture on the tight 1e-4 absolute budget without
    editing this function, and the explicit set keeps the documented budget
    readable next to the fixtures it applies to.
    """
    if slug in _TINY_SLUGS or slug.startswith("tiny"):
        return _REAL_RTOL, _TINY_ATOL, False
    return _REAL_RTOL, _REAL_ATOL, True


def _assert_parity(
    got: object, want: object, *, rtol: float, atol: float, require_relative: bool, label: str
) -> _ParityReport:
    """Assert exact shapes plus the absolute (and optionally relative) budget."""
    ours, theirs = np.asarray(got), np.asarray(want)
    assert ours.shape == theirs.shape, (
        f"{label}: shape mismatch — etl_hf produced {ours.shape}, HuggingFace "
        f"{theirs.shape}; shapes are static, so this is a graph-shape bug"
    )
    report = _compare(ours, theirs, rtol=rtol, atol=atol)
    message = report.describe(label)
    assert report.within_abs, message
    if require_relative:
        assert report.within_rel, message
    return report


# ---------------------------------------------------------------------------
# a tiny synthetic checkpoint (for the import-hygiene test; no download needed)
# ---------------------------------------------------------------------------
#: Tiny Llama config: 2 layers, GQA (4 query / 2 KV heads), 16-wide hidden.
_TINY_CONFIG: dict = {
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "intermediate_size": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 16,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
}


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write float32 ``tensors`` as a real safetensors file (no extra dependency)."""
    header: dict = {}
    payload = bytearray()
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32)
        begin = len(payload)
        payload += data.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(data.shape),
            "data_offsets": [begin, len(payload)],
        }
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((-len(blob)) % 8)  # the real writer pads the header to 8 bytes
    path = Path(path)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(payload))
    return path


def _write_tiny_checkpoint(directory: Path) -> Path:
    """Write a tiny synthetic ``LlamaForCausalLM``-shaped checkpoint directory.

    The tensors are generated from the family's own canonical ``param_specs`` and
    written in HF's layout (linears ``(out, in)``, i.e. the canonical ``(in,
    out)`` transposed), so :func:`etl_hf.load_hf` reads them back without a
    download, torch or ``safetensors``.  Every canonical spec must be consumed —
    an unconsumed one is an assertion failure, not a silently smaller checkpoint.
    """
    from etl_hf.config import ModelConfig
    from etl_hf.models import llama

    root = Path(directory) / "tiny_llama_ckpt"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(_TINY_CONFIG), encoding="utf-8")

    cfg = ModelConfig.from_dict(_TINY_CONFIG)
    specs = llama.param_specs(cfg)
    rng = np.random.default_rng(0)

    def weigh(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(0.05 * rng.standard_normal(shape), dtype=np.float32)

    def scale(shape: tuple[int, ...]) -> np.ndarray:
        return np.asarray(1.0 + 0.05 * rng.standard_normal(shape), dtype=np.float32)

    hidden, vocab = cfg.hidden_size, cfg.vocab_size
    assert tuple(specs["embed.weight"].shape) == (vocab, hidden)

    # norm scales (1-D) vs weight matrices (2-D): the only shape split here.
    canonical = {}
    for name, spec in specs.items():
        shape = tuple(int(dim) for dim in spec.shape)
        canonical[name] = scale(shape) if len(shape) == 1 else weigh(shape)

    consumed = {"embed.weight", "final_norm.weight"}
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": canonical["embed.weight"],
        "model.norm.weight": canonical["final_norm.weight"],
    }
    if not cfg.tie_word_embeddings:
        tensors["lm_head.weight"] = np.ascontiguousarray(canonical["lm_head.weight"].T)
        consumed.add("lm_head.weight")
    for index in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{index}."
        layer = f"layers.{index}."
        tensors[prefix + "input_layernorm.weight"] = canonical[layer + "attn_norm.weight"]
        tensors[prefix + "self_attn.q_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "attn.q.weight"].T
        )
        tensors[prefix + "self_attn.k_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "attn.k.weight"].T
        )
        tensors[prefix + "self_attn.v_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "attn.v.weight"].T
        )
        tensors[prefix + "self_attn.o_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "attn.o.weight"].T
        )
        tensors[prefix + "post_attention_layernorm.weight"] = canonical[
            layer + "mlp_norm.weight"
        ]
        tensors[prefix + "mlp.gate_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "mlp.gate.weight"].T
        )
        tensors[prefix + "mlp.up_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "mlp.up.weight"].T
        )
        tensors[prefix + "mlp.down_proj.weight"] = np.ascontiguousarray(
            canonical[layer + "mlp.down.weight"].T
        )
        consumed.update(
            {
                layer + "attn_norm.weight",
                layer + "attn.q.weight",
                layer + "attn.k.weight",
                layer + "attn.v.weight",
                layer + "attn.o.weight",
                layer + "mlp_norm.weight",
                layer + "mlp.gate.weight",
                layer + "mlp.up.weight",
                layer + "mlp.down.weight",
            }
        )
    assert consumed == set(specs), (
        "the synthetic checkpoint does not cover the family's canonical params: "
        f"unused={sorted(set(specs) - consumed)}"
    )
    _write_safetensors(root / "model.safetensors", tensors)
    return root


# ---------------------------------------------------------------------------
# 1. parity against the torch golden fixtures (when the harness produced them)
#    — torch-free: the oracle's logits are already on disk in the fixture
# ---------------------------------------------------------------------------
_GOLDENS = _load_goldens()

_NO_GOLDENS_MESSAGE = (
    f"no torch golden fixtures under {_goldens_dir()} — the golden harness writes "
    f"{_GOLDENS_CONTRACT}"
)


def test_golden_slugs_resolve():
    """Every discovered golden must map to a fetchable checkpoint (no network).

    This is the download-free guarantee behind
    :func:`test_torch_goldens_parity`: that test may only ever skip on a genuine
    :class:`hf_download.DownloadUnavailable`, so an unmappable golden has to be
    reported HERE as a failure instead of disappearing into a skip.

    For each ``<slug>.npz`` on disk it asserts:

    1. :func:`_repo_for_slug` returns a non-empty string — the meta's ``model``,
       its ``model_id``, a :data:`hf_download.MODELS` entry or the slug itself;
    2. the id is at most one ``org/name`` pair (never ``a/b/c``, never a
       half-empty ``a/``), and exactly one when the slug is NOT a
       :data:`hf_download.MODELS` key — HF serves single-namespace ids such as
       ``distilgpt2``, but this downloader's raw-repo-id path needs ``org/name``;
    3. the downloader can build a cache location for it
       (:func:`hf_download._repo_and_dir` — the same pure, I/O-free resolution
       ``download_model`` starts from), taking the MODELS path for a known slug
       and the raw repo-id path otherwise.

    Nothing is downloaded or written, so this runs everywhere — offline hosts
    included — and a golden that cannot be resolved is a red test right now.
    """
    goldens = _load_goldens()
    if not goldens:
        pytest.skip(_NO_GOLDENS_MESSAGE)
    for slug, (_input_ids, _logits, meta) in sorted(goldens.items()):
        meta_path = _goldens_dir() / (slug + ".meta.json")
        repo_id = _repo_for_slug(slug, meta)
        assert isinstance(repo_id, str) and repo_id.strip(), (
            f"golden {slug!r} does not resolve to any HuggingFace repo id (got "
            f"{repo_id!r}); {meta_path.name} must carry one of {_META_REPO_KEYS} "
            f"as a non-empty string — {_GOLDENS_CONTRACT}"
        )
        assert repo_id.count("/") <= 1 and all(repo_id.split("/")), (
            f"golden {slug!r} resolves to {repo_id!r}, which is not a usable "
            f"HuggingFace repo id (at most one 'org/name' pair, no empty part) — "
            f"{_GOLDENS_CONTRACT}"
        )
        known = slug in hf_download.MODELS
        assert known or repo_id.count("/") == 1, (
            f"golden {slug!r} is not a key of hf_download.MODELS "
            f"({', '.join(sorted(hf_download.MODELS))}) and its repo id {repo_id!r} "
            f"is not an 'org/name' pair, so it cannot be downloaded — add the slug "
            f"to hf_download.MODELS or fix 'model'/'model_id' in {meta_path.name} — "
            f"{_GOLDENS_CONTRACT}"
        )
        # The downloader's own resolution, so a mapping this file accepts can
        # never turn into a ValueError deep inside download_model at run time.
        try:
            hf_download._repo_and_dir(slug if known else repo_id)
        except ValueError as exc:
            raise AssertionError(
                f"golden {slug!r} does not resolve to a fetchable checkpoint: {exc}. "
                f"Path checked: {meta_path} — {_GOLDENS_CONTRACT}"
            ) from None


@pytest.mark.parametrize("slug", sorted(_GOLDENS) or ["<no goldens>"])
def test_torch_goldens_parity(slug: str):
    """Compare our logits against the pre-computed HF ones (no torch needed).

    Skips ONLY when no fixture directory exists (the golden harness has not run
    here) or when :func:`_checkpoint_dir` hits a genuine download failure
    (:class:`hf_download.DownloadUnavailable`).  A golden that exists but cannot
    be mapped onto a checkpoint is a FAILURE — that is what
    :func:`_checkpoint_dir` and :func:`test_golden_slugs_resolve` enforce.
    """
    if slug not in _GOLDENS:
        pytest.skip(_NO_GOLDENS_MESSAGE)
    input_ids, want, meta = _GOLDENS[slug]
    repo_id = _repo_for_slug(slug, meta)
    directory = _checkpoint_dir(slug, repo_id)
    batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[-1])

    from etl_hf import load_hf

    model = load_hf(directory, seq_len=seq_len, batch_size=batch)
    got = model.forward(input_ids=input_ids)

    rtol, atol, require_relative = _tolerances(slug)
    _assert_parity(
        got,
        want,
        rtol=rtol,
        atol=atol,
        require_relative=require_relative,
        label=f"golden {slug!r} ({repo_id}, seq_len={seq_len}, batch={batch})",
    )


# ---------------------------------------------------------------------------
# 2. parity against a fresh in-process HF forward (no fixtures needed)
# ---------------------------------------------------------------------------
#: Slugs of the in-process parity test.  ``tiny_gpt2`` is an ordinary case: its
#: checkpoint ships the **legacy** (pre-1.6) ``pytorch_model.bin`` layout, which
#: ``etl_hf.weights`` reads correctly (the trailer's per-storage int64 element
#: count is consumed), so it is compared against HuggingFace like the others —
#: this is the legacy reader's end-to-end check against an external oracle.
_IN_PROCESS_SLUGS = (
    "tiny_llama",
    "tiny_mistral",
    "tiny_gpt2",
)

#: config.json keys carrying the maximum number of positions, in probe order.
_MAX_POSITION_KEYS = ("max_position_embeddings", "n_positions", "n_ctx")


def _max_positions(directory: object) -> int | None:
    """The checkpoint's ``config.json`` position limit (``None`` when unreadable)."""
    try:
        raw = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    for key in _MAX_POSITION_KEYS:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


@pytest.mark.skipif(not _oracle_available(), reason=_ORACLE_SKIP_REASON)
@pytest.mark.parametrize("slug", _IN_PROCESS_SLUGS)
def test_hf_in_process_parity(slug: str):
    """Run HF and ``etl_hf`` on the same ids and compare logits + argmax."""
    directory = hf_download.require_model(slug)
    repo_id = hf_download.MODELS[slug]
    limit = _max_positions(directory)
    seq_len = _SEQ_LEN if limit is None else min(_SEQ_LEN, limit)
    assert seq_len >= 1, f"{slug}: the checkpoint's config.json allows no positions"
    input_ids = np.arange(seq_len, dtype=np.int32)[None]

    want = _hf_logits(repo_id, input_ids, "float32", source=directory)

    from etl_hf import load_hf

    got = load_hf(directory, seq_len=seq_len).forward(input_ids=input_ids)

    rtol, atol, require_relative = _tolerances(slug)
    _assert_parity(
        got,
        want,
        rtol=rtol,
        atol=atol,
        require_relative=require_relative,
        label=f"{slug} ({repo_id}, in-process HF forward, seq_len={seq_len})",
    )

    # The distribution agreement the absolute budget cannot express: identical
    # argmax wherever the oracle's own top-1 margin is not a near-tie.
    want_top = np.argmax(want, axis=-1)
    got_top = np.argmax(got, axis=-1)
    top2 = np.partition(want, -2, axis=-1)[..., -2:]
    margins = top2[..., 1] - top2[..., 0]
    decided = margins >= _TOP2_GAP
    if not decided.any():
        return  # every position is a near-tie: argmax equality is not meaningful
    mismatched = np.argwhere(decided & (got_top != want_top))
    if mismatched.size:
        index = tuple(int(part) for part in mismatched[0])
        raise AssertionError(
            f"{slug}: argmax disagrees at {mismatched.shape[0]} of "
            f"{int(decided.sum())} non-tie position(s) (top-2 margin >= "
            f"{_TOP2_GAP:g}); first at (batch, seq)={index}: etl_hf "
            f"{int(got_top[index])} vs HuggingFace {int(want_top[index])} "
            f"(margin {float(margins[index]):.3e})"
        )


# ---------------------------------------------------------------------------
# 3. the golden fixture contract itself (documents it for the producer)
# ---------------------------------------------------------------------------
def test_golden_fixture_format():
    """Validate every golden fixture against the documented contract."""
    directory = _goldens_dir()
    if not directory.is_dir():
        pytest.skip(_NO_GOLDENS_MESSAGE)
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        pytest.skip(f"{directory} exists but carries no *.npz golden yet — {_GOLDENS_CONTRACT}")

    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            assert {"input_ids", "logits"} <= names, (
                f"{path.name} is missing required arrays (found {sorted(names)}): "
                f"{_GOLDENS_CONTRACT}"
            )
            input_ids = np.asarray(archive["input_ids"])
            logits = np.asarray(archive["logits"])
        assert input_ids.dtype == np.int32, (
            f"{path.name}: `input_ids` must be int32, got {input_ids.dtype} — "
            f"{_GOLDENS_CONTRACT}"
        )
        assert logits.dtype == np.float32, (
            f"{path.name}: `logits` must be float32, got {logits.dtype} — "
            f"{_GOLDENS_CONTRACT}"
        )
        assert input_ids.ndim == 2, (
            f"{path.name}: `input_ids` must be 2-D (batch, seq_len), got "
            f"{input_ids.shape} — {_GOLDENS_CONTRACT}"
        )
        assert logits.ndim in (2, 3), (
            f"{path.name}: `logits` must be 3-D (batch, seq_len, vocab) or 2-D "
            f"(batch*seq_len, vocab), got {logits.shape} — {_GOLDENS_CONTRACT}"
        )
        assert logits.shape[:2] == input_ids.shape, (
            f"{path.name}: `logits` leading axes {logits.shape[:2]} do not match "
            f"`input_ids` {input_ids.shape} — {_GOLDENS_CONTRACT}"
        )

        meta_path = path.with_suffix(".meta.json")
        assert meta_path.is_file(), (
            f"{path.name} has no sibling {meta_path.name} — {_GOLDENS_CONTRACT}"
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert isinstance(meta, dict), (
            f"{meta_path.name}: expected a JSON object, got {type(meta).__name__} — "
            f"{_GOLDENS_CONTRACT}"
        )
        # The repo id lives under `model` or `model_id` (see _META_REPO_KEYS):
        # the harness that produced today's fixtures writes `model_id`, while
        # `model` stays the preferred spelling when a fixture carries both.
        repo_id = _meta_repo_id(meta)
        assert repo_id is not None, (
            f"{meta_path.name}: one of the keys {_META_REPO_KEYS} must carry a "
            f"non-empty HuggingFace repo id, found none (keys present: "
            f"{sorted(meta)}) — {_GOLDENS_CONTRACT}"
        )


# ---------------------------------------------------------------------------
# 4. neither etl_hf nor this module may import the optional oracle
# ---------------------------------------------------------------------------
def _run_probe(code: str) -> dict[str, str]:
    """Run ``code`` in a fresh interpreter and parse its ``KEY=value`` lines.

    A subprocess is essential here: anything this module (or the surrounding
    pytest session) already imported would otherwise pollute the check.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"the probe subprocess failed:\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    return dict(line.split("=", 1) for line in completed.stdout.strip().splitlines())


def test_etl_hf_never_imports_torch(tmp_path):
    """``import etl_hf`` + a full tiny forward must not pull in torch/transformers.

    Two probes, each in a fresh interpreter: the *library* stays numpy-only even
    when the oracle happens to be installed, and this test module's own oracle
    helper stays lazy (collecting the file — ``skipif`` gates included — must not
    import the oracle either).
    """
    directory = _write_tiny_checkpoint(tmp_path)
    library = _run_probe(
        f"""
        import sys
        import etl_hf
        model = etl_hf.load_hf({str(directory)!r}, seq_len=4)
        logits = model.forward()
        print("SHAPE=" + ",".join(str(int(dim)) for dim in logits.shape))
        print("DTYPE=" + str(logits.dtype))
        print("LEAKED=" + ",".join(
            name for name in ("torch", "transformers") if name in sys.modules))
        """
    )
    assert library["LEAKED"] == "", (
        f"etl_hf imported optional oracle dependencies: {library['LEAKED']}"
    )
    assert library["SHAPE"] == f"1,4,{_TINY_CONFIG['vocab_size']}", library["SHAPE"]
    assert library["DTYPE"] == "float32", library["DTYPE"]

    this_file = Path(__file__).resolve()
    gate = _run_probe(
        f"""
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("etl_hf_oracle_gate", {str(this_file)!r})
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print("ORACLE_CACHED=" + str(module._ORACLE is not None))
        print("LEAKED=" + ",".join(
            name for name in ("torch", "transformers") if name in sys.modules))
        """
    )
    assert gate["LEAKED"] == "", (
        f"importing {this_file.name} pulled in optional oracle dependencies: "
        f"{gate['LEAKED']} — the oracle must only be imported by _oracle()"
    )
    assert gate["ORACLE_CACHED"] == "False", (
        "importing this test module already invoked the oracle helper; it must "
        "stay lazy so the torch-free tests can run without torch"
    )
