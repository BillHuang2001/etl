"""Scratch verification (bench_logs probe area): reload all five HF golden fixtures and assert the
``<slug>.meta.json`` contract (``model`` == ``model_id``, required keys present, ordering/formatting
preserved) plus that every ``.npz`` still loads with the expected shapes/dtypes and reproduces the
documented ``input_ids`` recipe.

Run from the repo root:  uv run python bench_logs/verify_hf_goldens_meta.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

G = pathlib.Path(__file__).resolve().parent / "hf_goldens"
SLUGS = ["tiny_random_llama", "tiny_random_mistral", "tiny_gpt2", "distilgpt2", "gpt2"]
REQUIRED = ["model", "model_id", "slug", "seq_len", "vocab_size", "dtype", "input_ids_recipe"]


def main() -> int:
    for slug in SLUGS:
        meta_path = G / f"{slug}.meta.json"
        text = meta_path.read_text()
        meta = json.loads(text)
        assert list(meta)[:2] == ["model", "model_id"], (slug, list(meta)[:3])
        assert meta["model"] == meta["model_id"], (slug, meta["model"], meta["model_id"])
        assert meta["slug"] == slug, (slug, meta["slug"])
        for key in REQUIRED:
            assert key in meta, (slug, "missing required key", key)
        # The files are exactly json.dumps(indent=2) of their own content: key ordering + formatting
        # are byte-preserved (only the inserted ``model`` line differs from the pre-fix files).
        assert text == json.dumps(meta, indent=2) + "\n", f"{slug}: formatting drift"

        seq_len, vocab = meta["seq_len"], meta["vocab_size"]
        with np.load(G / f"{slug}.npz") as z:
            logits, ids = z["logits"], z["input_ids"]
        assert logits.shape == (1, seq_len, vocab), (slug, logits.shape)
        assert logits.dtype == np.float32, (slug, logits.dtype)
        assert ids.shape == (1, seq_len) and ids.dtype == np.int32, (slug, ids.shape, ids.dtype)
        rebuilt = ((np.arange(seq_len) * 7 + 3) % vocab)[None, :].astype(np.int32)
        assert np.array_equal(rebuilt, ids), slug
        assert np.isfinite(logits).all(), f"{slug}: non-finite logits"
        assert float(logits.std()) > 0.0, f"{slug}: constant logits"
        print(
            f"OK {slug:20s} model={meta['model']:52s} logits{tuple(logits.shape)} "
            f"absmax={float(np.abs(logits).max()):.4g}"
        )
    print(f"\n{len(SLUGS)}/{len(SLUGS)} fixtures OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
