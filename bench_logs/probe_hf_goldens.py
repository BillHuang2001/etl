"""probe_hf_goldens — produce ground-truth HuggingFace reference logits (torch/transformers ORACLE)
for the checkpoints used to validate the new ``etl_hf`` package.

Scratch probe (``bench_logs`` is a throwaway probe area, NOT library code and NOT run by pytest).

CRITICAL: this script imports torch + transformers, so it MUST be run with a THROWAWAY venv
created under ``$TMPDIR`` — never the repo ``.venv`` (adding torch there would flip the test
suite's ``pytest.importorskip("torch")`` guards):

    export TMPDIR=${TMPDIR:-/tmp}
    uv venv "$TMPDIR/hforacle-venv" --python 3.12
    uv pip install --python "$TMPDIR/hforacle-venv/bin/python" \\
        --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python "$TMPDIR/hforacle-venv/bin/python" transformers safetensors numpy

    HF_HOME="$TMPDIR/hf_oracle_cache" "$TMPDIR/hforacle-venv/bin/python" bench_logs/probe_hf_goldens.py

What it writes (all under ``bench_logs/hf_goldens/``):

    <slug>.npz            logits (f32, (1,S,V)) + input_ids (i32, (1,S))
    <slug>.meta.json      model id / config summary / versions / recipes / sanity numbers
    <slug>.keys.txt       full on-disk weight tensor listing (name + shape + dtype)
    README.md             how to regenerate + what each file means (written separately)

Every model is run with ``torch.set_grad_enabled(False)``, ``model.eval()``, float32, CPU, and a
plain FORWARD pass (never ``generate``). Input ids are built deterministically with numpy:

    input_ids[0, i] = (i * 7 + 3) % vocab_size          (i = 0 .. S-1)

which needs no RNG and is reproducible outside torch.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent / "hf_goldens"

# (model id, slug, seq len)
MODELS = [
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", "tiny_random_llama", 8),
    ("hf-internal-testing/tiny-random-MistralForCausalLM", "tiny_random_mistral", 8),
    ("sshleifer/tiny-gpt2", "tiny_gpt2", 8),
    ("distilgpt2", "distilgpt2", 32),
    ("gpt2", "gpt2", 32),
]

ID_RECIPE = "input_ids[0, i] = (i * 7 + 3) % vocab_size for i in range(S); int64 for torch, saved int32"


def make_input_ids(seq_len: int, vocab_size: int) -> "np.ndarray":
    import numpy as np

    i = np.arange(seq_len, dtype=np.int64)
    return ((i * 7 + 3) % vocab_size)[None, :]


# ---------------------------------------------------------------------------
# on-disk weight-file / key inspection
# ---------------------------------------------------------------------------
def weight_file_report(model_dir: Path) -> dict:
    """List the weight files actually on disk + first/last tensor keys (name, shape, dtype)."""
    files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    weight_files = [
        f for f in files if f.endswith((".safetensors", ".bin")) and not f.startswith("training_args")
    ]
    index_files = [f for f in files if f.endswith(".index.json")]
    report: dict = {
        "model_dir_files": files,
        "weight_files": weight_files,
        "index_files": index_files,
        "format": None,
        "keys": [],  # [(name, shape, dtype)]
        "shards": {},
    }
    keys: list = []
    if index_files:
        idx = json.loads((model_dir / index_files[0]).read_text())
        report["shards"] = idx.get("weight_map", {})
        keys = [(k, None, None) for k in sorted(idx.get("weight_map", {}))]
        report["format"] = "sharded-index"
    st_files = [f for f in weight_files if f.endswith(".safetensors")]
    if st_files:
        from safetensors import safe_open

        report["format"] = "safetensors" if not index_files else "safetensors-sharded"
        keys = []
        for f in st_files:
            with safe_open(str(model_dir / f), framework="numpy") as fh:
                for k in fh.keys():
                    sl = fh.get_slice(k)
                    get_dtype = getattr(sl, "get_dtype", None)
                    keys.append((k, tuple(sl.get_shape()), get_dtype() if get_dtype else None))
        keys.sort()
    elif not index_files:
        bin_files = [f for f in weight_files if f.endswith(".bin")]
        if bin_files:
            import torch

            sd = torch.load(str(model_dir / bin_files[0]), map_location="cpu", weights_only=True)
            keys = [(k, tuple(v.shape), str(v.dtype)) for k, v in sorted(sd.items())]
            report["format"] = "pytorch_model.bin"
    report["num_keys"] = len(keys)
    report["keys_head"] = keys[:12]
    report["keys_tail"] = keys[-12:]
    report["_all_keys"] = keys
    return report


def write_keys_txt(path: Path, model_id: str, rep: dict) -> None:
    lines = [f"# {model_id}", f"# format: {rep['format']}"]
    lines.append(f"# on-disk files: {', '.join(rep['model_dir_files'])}")
    lines.append(f"# weight files: {', '.join(rep['weight_files']) or '(none)'}")
    if rep["index_files"]:
        lines.append(f"# index: {rep['index_files'][0]}  ({len(rep['shards'])} entries)")
    lines.append(f"# {rep['num_keys']} tensors")
    lines.append("# name\tshape\tdtype")
    for name, shape, dtype in rep["_all_keys"]:
        lines.append(f"{name}\t{shape}\t{dtype}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------
def run_model(model_id: str, slug: str, seq_len: int) -> dict:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.set_grad_enabled(False)
    print(f"\n=== {model_id} (slug={slug}, S={seq_len}) ===", flush=True)

    cfg_pre = AutoConfig.from_pretrained(model_id)
    vocab_size = int(cfg_pre.vocab_size)
    ids_np = make_input_ids(seq_len, vocab_size)
    ids = torch.from_numpy(ids_np).long()

    # torch_dtype= was renamed dtype= in transformers v5; try the modern spelling first.
    model = None
    load_kwargs_used: dict = {}
    for kw in (
        {"dtype": torch.float32, "attn_implementation": "eager"},
        {"torch_dtype": torch.float32, "attn_implementation": "eager"},
        {"dtype": torch.float32},
        {"torch_dtype": torch.float32},
    ):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            load_kwargs_used = {k: str(v) for k, v in kw.items()}
            break
        except TypeError as e:  # unsupported kwarg spelling
            print(f"  from_pretrained{list(kw)} -> TypeError: {e}", flush=True)
            continue
    if model is None:
        raise RuntimeError(f"{model_id}: could not construct the model with any known kwarg spelling")
    model.eval()
    assert next(model.parameters()).dtype == torch.float32, "model is not float32"

    forward_call = "out = model(input_ids=input_ids); logits = out.logits"
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits
    assert tuple(logits.shape) == (1, seq_len, vocab_size), (logits.shape, vocab_size)
    logits_np = logits.detach().to(torch.float32).cpu().numpy()
    assert logits_np.dtype == np.float32
    # determinism: a second identical forward must be bit-identical (the oracle is a pure function)
    with torch.no_grad():
        logits_b = model(input_ids=ids).logits
    second_maxdiff = float((logits_b - logits).abs().max())
    assert second_maxdiff == 0.0, f"non-deterministic forward: {second_maxdiff}"

    cfg = model.config
    # `_name_or_path` is the repo id for hub loads; resolve the local snapshot dir so we can
    # report the exact on-disk weight filenames the etl_hf implementers have to read.
    name_or_path = Path(model.config._name_or_path)
    if name_or_path.is_dir():
        model_dir = name_or_path
    else:
        import huggingface_hub  # transformers dependency

        model_dir = Path(huggingface_hub.snapshot_download(model_id))
    rep = weight_file_report(model_dir)
    write_keys_txt(OUT / f"{slug}.keys.txt", model_id, rep)

    np.savez_compressed(
        OUT / f"{slug}.npz",
        logits=logits_np.astype(np.float32),
        input_ids=ids_np.astype(np.int32),
    )

    def cget(*names, default=None):
        for n in names:
            if hasattr(cfg, n) and getattr(cfg, n) is not None:
                return getattr(cfg, n)
        return default

    hidden_act = cget("hidden_act", "activation_function")
    absmax = float(np.abs(logits_np).max())
    is_finite = bool(np.isfinite(logits_np).all())
    std = float(logits_np.std())

    # ---- tie / untie behaviour, straight from the on-disk keys + the loaded tensors ----
    disk_names = [k[0] for k in rep["_all_keys"]]
    lm_head_keys = [k for k in disk_names if k.endswith("lm_head.weight")]
    embed_keys = [k for k in disk_names if k.endswith(("wte.weight", "embed_tokens.weight"))]
    tie_obs: dict = {
        "lm_head_weight_on_disk": lm_head_keys,
        "embed_weight_on_disk": embed_keys,
        "tie_word_embeddings_config": cget("tie_word_embeddings", default=None),
        "lm_head_present": bool(lm_head_keys),
    }
    sd = model.state_dict()
    for cand_a, cand_b in (
        ("transformer.wte.weight", "lm_head.weight"),
        ("model.embed_tokens.weight", "lm_head.weight"),
        ("wte.weight", "lm_head.weight"),
    ):
        if cand_a in sd and cand_b in sd:
            tie_obs["runtime_pair"] = [cand_a, cand_b]
            tie_obs["runtime_lm_head_equals_embed"] = bool(torch.equal(sd[cand_a], sd[cand_b]))
            tie_obs[f"runtime_{cand_a}_shape"] = list(sd[cand_a].shape)
            tie_obs[f"runtime_{cand_b}_shape"] = list(sd[cand_b].shape)
            break
    if "runtime_lm_head_equals_embed" not in tie_obs:
        tie_obs["runtime_lm_head_equals_embed"] = "lm_head absent from state_dict (implicitly tied)"

    # non-parameter buffers that etl_hf must SKIP when building its weight tree
    # (GPT-2 stores the causal mask as an `attn.bias` buffer and, in older exports,
    # `attn.masked_bias`); identified on disk by name suffix (careful: `c_attn.bias`
    # is the fused-QKV BIAS parameter and must NOT be skipped).
    def _is_mask_buffer(name: str) -> bool:
        parts = name.split(".")
        return parts[-2:] == ["attn", "bias"] or parts[-1] in ("masked_bias",)

    skip_keys = [[n, sh, dt] for (n, sh, dt) in rep["_all_keys"] if _is_mask_buffer(n)]
    param_count = int(sum(p.numel() for p in model.parameters()))
    buffer_count = int(sum(b.numel() for b in model.buffers()))

    # GPT-2 has no `intermediate_size` in config.json: it is hard-coded to 4 * n_embd.
    intermediate = cget("intermediate_size", "n_inner")
    intermediate_source = "config"
    if intermediate is None:
        mlp_key = next((k for k in disk_names if "mlp.c_fc.weight" in k), None)
        n_embd = cget("n_embd")
        if mlp_key is not None and n_embd is not None:
            intermediate = 4 * int(n_embd)
            intermediate_source = "4 * n_embd (GPT-2 hard-codes n_inner = None)"

    # rope_theta lives at the top level for some configs and inside rope_scaling for others
    rope_theta = cget("rope_theta")
    if rope_theta is None:
        rs = cget("rope_scaling") or {}
        if isinstance(rs, dict) and "rope_theta" in rs:
            rope_theta = rs["rope_theta"]

    flags = []
    if not is_finite:
        flags.append("NON-FINITE logits")
    if std == 0.0:
        flags.append("DEGENERATE constant logits")
    if absmax < 1e-20:
        flags.append(f"DENORMAL-SCALE logits (absmax={absmax:.3e}) — checkpoint is degenerate")
    if absmax > 1e6:
        flags.append(f"HUGE logits (absmax={absmax:.3e})")
    if hidden_act in ("gelu", "gelu_new_erf"):
        flags.append(f"hidden_act={hidden_act!r} is EXACT erf-GELU — NOT exportable by etl v1")

    meta = {
        "model_id": model_id,
        "slug": slug,
        "resolved_model_type": cfg.model_type,
        "model_class": type(model).__name__,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "config_summary": {
            "vocab_size": vocab_size,
            "hidden_size": cget("hidden_size", "n_embd", "d_model"),
            "num_hidden_layers": cget("num_hidden_layers", "n_layer", "num_layers"),
            "num_attention_heads": cget("num_attention_heads", "n_head", "num_heads"),
            "num_key_value_heads": cget("num_key_value_heads", default=cget("num_attention_heads", "n_head")),
            "intermediate_size": intermediate,
            "intermediate_size_source": intermediate_source,
            "hidden_act": hidden_act,
            "layer_norm_eps": cget("layer_norm_eps"),
            "rms_norm_eps": cget("rms_norm_eps"),
            "layer_norm_epsilon": cget("layer_norm_epsilon"),
            "max_position_embeddings": cget("max_position_embeddings", "n_positions", "n_ctx"),
            "tie_word_embeddings": cget("tie_word_embeddings", default=None),
            "rope_theta": rope_theta,
            "rope_scaling": cget("rope_scaling"),
            "n_embd": cget("n_embd"),
            "n_head": cget("n_head"),
            "n_layer": cget("n_layer"),
            "n_positions": cget("n_positions"),
            "activation_function": cget("activation_function"),
            "attention_bias": cget("attention_bias"),
            "mlp_bias": cget("mlp_bias"),
            "kv_lora_rank": cget("kv_lora_rank"),
            "head_dim": cget("head_dim"),
        },
        "raw_config": {k: v for k, v in cfg.to_dict().items() if not isinstance(v, (dict, list))},
        "versions": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "dtype": "float32",
        "device": "cpu",
        "grad_enabled": False,
        "eval_mode": True,
        "attn_implementation_effective": getattr(cfg, "_attn_implementation", None),
        "from_pretrained_kwargs": load_kwargs_used,
        "input_ids_recipe": ID_RECIPE,
        "forward_call": forward_call,
        "generation_used": False,
        "logits_shape": list(logits_np.shape),
        "logits_dtype": str(logits_np.dtype),
        "logits_absmax": absmax,
        "logits_mean": float(logits_np.mean()),
        "logits_std": std,
        "logits_finite": is_finite,
        "logits_argmax_last_pos": int(logits_np[0, -1].argmax()),
        "second_forward_maxdiff": second_maxdiff,
        "num_parameters": param_count,
        "num_buffer_elements": buffer_count,
        "flags": flags,
        "tie_observation": tie_obs,
        "skip_keys_non_parameter": skip_keys,
        "weights": {
            "format": rep["format"],
            "weight_files": rep["weight_files"],
            "index_files": rep["index_files"],
            "model_dir_files": rep["model_dir_files"],
            "num_keys": rep["num_keys"],
            "keys_head": rep["keys_head"],
            "keys_tail": rep["keys_tail"],
            "full_key_listing": f"{slug}.keys.txt",
        },
        "npz": f"{slug}.npz",
    }
    (OUT / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  OK logits{list(logits_np.shape)} absmax={absmax:.4g} mean={meta['logits_mean']:.4g} "
        f"std={std:.4g} model_type={cfg.model_type} act={hidden_act} weights={rep['format']} "
        f"keys={rep['num_keys']} flags={flags}",
        flush=True,
    )
    return meta


# ---------------------------------------------------------------------------
# verification pass (reload every npz, assert shape/dtype, non-degenerate)
# ---------------------------------------------------------------------------
def verify() -> list[dict]:
    import numpy as np

    rows = []
    for _, slug, seq_len in MODELS:
        npz_path = OUT / f"{slug}.npz"
        if not npz_path.exists():
            rows.append({"slug": slug, "status": "MISSING"})
            continue
        with np.load(npz_path) as z:
            logits = z["logits"]
            ids = z["input_ids"]
        meta = json.loads((OUT / f"{slug}.meta.json").read_text())
        vocab = meta["vocab_size"]
        assert logits.dtype == np.float32, (slug, logits.dtype)
        assert ids.dtype == np.int32, (slug, ids.dtype)
        assert logits.shape == (1, seq_len, vocab), (slug, logits.shape, vocab)
        assert ids.shape == (1, seq_len), (slug, ids.shape)
        assert int(ids.min()) >= 0 and int(ids.max()) < vocab, (slug, ids.min(), ids.max())
        assert np.isfinite(logits).all(), f"{slug}: non-finite logits"
        assert float(logits.std()) > 0.0, f"{slug}: constant logits"
        absmax = float(np.abs(logits).max())
        assert absmax == pytest_free_close(absmax, meta["logits_absmax"]), (slug, absmax)
        rows.append(
            {
                "slug": slug,
                "status": "PASS",
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "absmax": absmax,
                "mean": float(logits.mean()),
                "std": float(logits.std()),
                "degenerate_flag": absmax < 1e-20,
                "npz_bytes": npz_path.stat().st_size,
            }
        )
        print(f"  VERIFY {slug}: PASS shape={list(logits.shape)} absmax={absmax:.4g}", flush=True)
    return rows


def pytest_free_close(a: float, b: float) -> float:
    return a if abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)) else float("nan")


def main() -> int:
    import numpy as np
    import torch
    import transformers

    os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_oracle_cache"))
    # The HF Xet storage backend 404s on some networks here (`.../xet-read-token/...`); the plain
    # HTTP path works, so pin it for reproducibility.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"transformers {transformers.__version__} / torch {torch.__version__} / numpy {np.__version__}")
    print(f"HF_HOME={os.environ['HF_HOME']}  out={OUT}", flush=True)

    report: dict = {"produced": [], "failed": []}
    for model_id, slug, seq_len in MODELS:
        try:
            meta = run_model(model_id, slug, seq_len)
            report["produced"].append(
                {
                    "model_id": model_id,
                    "slug": slug,
                    "logits_shape": meta["logits_shape"],
                    "weights_format": meta["weights"]["format"],
                    "weight_files": meta["weights"]["weight_files"],
                    "num_keys": meta["weights"]["num_keys"],
                    "absmax": meta["logits_absmax"],
                    "flags": meta["flags"],
                }
            )
        except Exception as e:  # noqa: BLE001 - probe: one model failing must not kill the rest
            print(f"  FAIL {model_id}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            report["failed"].append({"model_id": model_id, "slug": slug, "error": f"{type(e).__name__}: {e}"})

    print("\n=== verification ===", flush=True)
    report["verify"] = verify()
    report["versions"] = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    (OUT / "oracle_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("produced", "failed")}, indent=2)[:4000])
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
