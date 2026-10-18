"""Download tiny HF checkpoints with plain urllib (no hub libs) + parse safetensors with numpy only.

Scratch probe (bench_logs is a throwaway probe area, not part of the library).

Usage:
    uv run python bench_logs/probe_hf_download.py [--out $TMPDIR/hf_models]

Safetensors format (all little-endian):
    [0:8]   u64  header length N
    [8:8+N] utf8 JSON header: {"tensor_name": {"dtype": "F32", "shape": [..],
                                               "data_offsets": [begin, end]}, ...}
            (a "__metadata__" key may be present -> str->str map, skip it)
    [8+N:]  raw tensor data buffer; tensor bytes = buffer[begin:end]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"

ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": None,  # special-cased (no numpy bfloat16); unpack as u16
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest} ({dest.stat().st_size} bytes)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "etl-probe/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        data = r.read()
        f.write(data)
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")
    return dest


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    (n,) = np.frombuffer(raw, dtype="<u8", count=1)
    header = json.loads(raw[8 : 8 + n].decode("utf-8"))
    buf = memoryview(raw)[8 + n :]
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        begin, end = info["data_offsets"]
        nbytes = end - begin
        if dtype == "BF16":
            u16 = np.frombuffer(buf[begin:end], dtype="<u2").reshape(shape)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            out[name] = f32
            continue
        np_dtype = ST_DTYPES[dtype]
        assert np_dtype is not None, f"unknown dtype {dtype} for {name}"
        arr = np.frombuffer(buf[begin:end], dtype=np.dtype(np_dtype).newbyteorder("<"))
        assert arr.size == int(np.prod(shape)) if shape else 1, (
            f"{name}: {arr.size} elems vs shape {shape} ({nbytes} bytes)"
        )
        out[name] = arr.reshape(shape).astype(np_dtype)  # writable copy
    return out


def download_model(repo: str, subdir: str, out_root: Path) -> Path | None:
    print(f"[{repo}]")
    d = out_root / subdir
    fetch(f"{HF}/{repo}/resolve/main/config.json", d / "config.json")
    for weights in ("model.safetensors", "pytorch_model.bin"):
        try:
            fetch(f"{HF}/{repo}/resolve/main/{weights}", d / weights)
            return d / weights
        except Exception as e:  # noqa: BLE001 - probe: report and fall through
            print(f"  no {weights}: {type(e).__name__}: {e}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ["TMPDIR"], "hf_models"))
    args = ap.parse_args()
    out_root = Path(args.out)

    models = [
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "tiny_llama"),
        ("sshleifer/tiny-gpt2", "tiny_gpt2"),
        ("distilgpt2", "distilgpt2"),
    ]
    for repo, sub in models:
        wpath = download_model(repo, sub, out_root)
        if wpath is None:
            print(f"  !! {repo}: no usable weights file; skipping")
            continue
        cfg = json.loads((out_root / sub / "config.json").read_text())
        print(f"  config: {json.dumps(cfg)[:800]}")
        if wpath.suffix != ".safetensors":
            print(f"  !! {repo}: only {wpath.name} available (no safetensors) — skipping load")
            continue
        tensors = read_safetensors(wpath)
        total = sum(int(np.prod(v.shape)) for v in tensors.values())
        print(f"  {len(tensors)} tensors, {total} params")
        for k in sorted(tensors):
            v = tensors[k]
            print(f"    {k:60s} {str(tuple(v.shape)):20s} {v.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
