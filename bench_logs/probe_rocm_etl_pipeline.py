#!/usr/bin/env python3
"""Full etl pipeline (trace -> lower -> compile -> load -> run) on a ROCm PJRT plugin.

Exercises the xla adapter end-to-end against the ``jax-rocm7-pjrt`` plugin on an
AMD Instinct MI300X (gfx942). ``"rocm"`` (alias ``"hip"``) is a first-class
accelerator kind, validated against the plugin's ``platform_name``:

  A. ``etl.build(fn, *specs, backend="xla", plugin_path=<so>)`` with the DEFAULT
     device (None -> Device("cpu", 0) — host-staged execution on the plugin's
     device 0) + ``etl.run`` vs the numpy backend.
  B. the same with ``device=core.Device("rocm", 0)`` / ``Device("hip", 0)`` —
     device-resident: inputs are placed explicitly with ``Tensor.to`` and the
     outputs stay on the device.
  C. the same with ``device=core.Device("cuda", 0)`` — expected to be REJECTED:
     ``"cuda"`` is not a kind the ROCm plugin's platform serves.

Usage:
  ETL_PJRT_PLUGIN=/path/to/xla_rocm_plugin.so python3 probe_rocm_etl_pipeline.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import etl  # noqa: E402
from etl import core  # noqa: E402

PLUGIN = os.environ.get("ETL_PJRT_PLUGIN") or ""
RESULTS: list[tuple[str, str]] = []


def record(step: str, status: str, detail: str = "") -> None:
    RESULTS.append((step, status))
    print(f"[{status}] {step}" + (f"\n       {detail}" if detail else ""))


@etl.defn
def kernel(x, y):
    """A small but non-trivial graph: sin + mul + reduce + add."""
    return etl.sum(etl.sin(x) * y, axes=(1,)) + etl.sum(x, axes=(1,))


def main() -> int:
    if not PLUGIN:
        print("set ETL_PJRT_PLUGIN to the plugin .so")
        return 2
    print(f"python  : {sys.version.split()[0]}")
    print(f"etl     : {etl.__version__} @ {etl.__file__}")
    print(f"plugin  : {PLUGIN!r}")
    print()

    spec = core.TensorSpec((4, 3), core.float32)
    x = np.arange(12, dtype=np.float32).reshape(4, 3) / 7.0
    y = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(4, 3)
    reference = np.sum(np.sin(x) * y, axis=1) + np.sum(x, axis=1)

    # --- A. default device (None -> Device("cpu", 0), host-staged) ---------
    try:
        exe = etl.build(kernel, spec, spec, backend="xla", plugin_path=PLUGIN)
        out = etl.run(exe, x, y)
        out = np.asarray(out.numpy() if hasattr(out, "numpy") else out)
        ok = bool(np.allclose(out, reference, rtol=1e-6, atol=1e-6))
        detail = (f"host-staged executable on the plugin's device 0\n"
                  f"       etl/xla output = {out.tolist()}\n"
                  f"       numpy expected = {reference.tolist()}\n"
                  f"       allclose(rtol=atol=1e-6): {ok}")
        record("A. etl.build(backend='xla') default device (cpu, host-staged)",
               "PASS" if ok else "FAIL", detail)
        try:
            exe.close()
        except Exception:
            pass
    except Exception as exc:
        record("A. etl.build(backend='xla') default device (cpu, host-staged)",
               "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    # --- B/C. explicit device kinds ----------------------------------------
    for kind in ("rocm", "hip", "cuda"):
        try:
            exe = etl.build(kernel, spec, spec, backend="xla",
                            plugin_path=PLUGIN, device=core.Device(kind, 0))
            if kind == "cuda":
                # A "cuda" label must not be accepted by a ROCm plugin.
                record("C. etl.build(device=Device('cuda',0))", "FAIL",
                       "accepted 'cuda' through a ROCm plugin (expected a "
                       "platform-kind rejection)")
            else:
                # device-resident: inputs must be placed explicitly.
                xt = core.Tensor(x).to(core.Device(kind, 0))
                yt = core.Tensor(y).to(core.Device(kind, 0))
                out = etl.run(exe, xt, yt)
                host = out.to(core.Device("cpu", 0)) if hasattr(out, "to") else out
                host = np.asarray(
                    host.numpy() if hasattr(host, "numpy") else host)
                ok = bool(np.allclose(host, reference, rtol=1e-6, atol=1e-6))
                record(f"B. etl.build(device=Device({kind!r},0)) device-resident",
                       "PASS" if ok else "FAIL",
                       f"device-resident output -> host = {host.tolist()}\n"
                       f"       numpy expected = {reference.tolist()}\n"
                       f"       allclose: {ok}")
            try:
                exe.close()
            except Exception:
                pass
        except Exception as exc:
            expected = kind == "cuda"
            record(f"{'C' if expected else 'B'}. "
                   f"etl.build(device=Device({kind!r},0))",
                   "REJECTED" if expected else "FAIL",
                   f"{type(exc).__name__}: {exc}")

    print()
    for step, status in RESULTS:
        print(f"   {status:9s} {step}")
    print()
    print("NOTE: this process may abort AFTER this line — the ROCm PJRT "
          "plugin corrupts the heap at interpreter exit (see "
          "probe_rocm_raw_pjrt.py / probe_rocm_teardown_minimal.py).")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
