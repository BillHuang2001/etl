#!/usr/bin/env python3
"""Can TVM RUN on ROCm here? (apache-tvm 0.26.0 wheel + ROCm 7.1 + gfx942)

Checks, in order, with VERBATIM errors:
  1. the wheel's build config           (tvm.support.libinfo() USE_ROCM / USE_LLVM)
  2. device-runtime modules             (tvm.contrib.rocm / tvm.contrib.hip)
  3. tvm.runtime.device("rocm", 0).exist
  4. target parsing / auto-detection    (tvm.target.Target("rocm") -> mcpu gfx942)
  5. an actual relax build for target="rocm" (etl's tvm_util.build_vm_executable)
  6. an actual relax VM run on the rocm device

Usage:  python3 probe_rocm_tvm.py
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS: list[tuple[str, str]] = []


def record(step: str, status: str, detail: str = "") -> None:
    RESULTS.append((step, status))
    print(f"[{status}] {step}" + (f"\n       {detail}" if detail else ""))


def trivial_relax_module():
    """A trivial relax module (TVMScript — tvm 0.26 has no relax StructInfo)."""
    import tvm
    from tvm.script import relax as R

    @R.function
    def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")):
        with R.dataflow():
            z = R.add(x, y)
            R.output(z)
        return z

    return tvm.IRModule({"main": main})


def main() -> int:
    import tvm
    from tvm import relax

    print(f"python : {sys.version.split()[0]}")
    print(f"tvm    : {tvm.__version__} @ {tvm.__file__}")
    print()

    # --- 1. build config ----------------------------------------------------
    info = tvm.support.libinfo()
    keys = ("USE_ROCM", "USE_LLVM", "USE_CUDA", "ROCM_PATH", "USE_ROCM_RUNTIME")
    detail = "; ".join(f"{k}={info.get(k, '<unset>')}" for k in keys)
    rocm_on = str(info.get("USE_ROCM", "OFF")).upper() not in ("OFF", "0", "FALSE", "")
    record("1. wheel build config (tvm.support.libinfo)", "INFO", detail)

    # --- 2. device runtime modules -----------------------------------------
    import tvm.contrib

    has_rocm = hasattr(tvm.contrib, "rocm")
    has_hip = hasattr(tvm.contrib, "hip")
    record("2. tvm.contrib device-runtime modules", "INFO",
           f"tvm.contrib.rocm = {has_rocm}; tvm.contrib.hip = {has_hip}")

    # --- 3. runtime device existence ---------------------------------------
    try:
        dev = tvm.runtime.device("rocm", 0)
        exists = bool(dev.exist)
        record("3. tvm.runtime.device('rocm', 0)", "PASS" if exists else "FAIL",
               f"device = {dev!r}; .exist = {exists}"
               + ("" if exists else "  <- no ROCm device runtime in this wheel"))
    except Exception as exc:
        record("3. tvm.runtime.device('rocm', 0)", "FAIL",
               f"{type(exc).__name__}: {exc}")
        dev = None

    # --- 4. target parsing --------------------------------------------------
    try:
        tgt = tvm.target.Target("rocm")
        record("4. tvm.target.Target('rocm')", "PASS",
               f"{tgt}  (keys={tgt.keys}, mcpu={tgt.attrs.get('mcpu')})")
    except Exception as exc:
        record("4. tvm.target.Target('rocm')", "FAIL",
               f"{type(exc).__name__}: {exc}")
        tgt = None

    # --- 5. actual build for target="rocm" ----------------------------------
    vm_ex = None
    try:
        from etl.backends.adapters import tvm_util

        mod = trivial_relax_module()
        vm_ex = tvm_util.build_vm_executable(mod, target="rocm")
        record("5. build_vm_executable(target='rocm')", "PASS",
               f"VMExecutable built: {type(vm_ex).__name__} "
               f"(codegen+link for gfx942 succeeded)")
    except Exception as exc:
        record("5. build_vm_executable(target='rocm')", "FAIL",
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    # --- 6. actual run on the rocm device -----------------------------------
    if vm_ex is not None and dev is not None:
        try:
            import numpy as np

            vm = relax.VirtualMachine(vm_ex, dev)
            a = tvm.nd.array(np.array([1, 2, 3, 4], dtype="float32"), dev)
            b = tvm.nd.array(np.array([10, 20, 30, 40], dtype="float32"), dev)
            out = vm["main"](a, b)
            got = out.numpy().tolist()
            ok = got == [11.0, 22.0, 33.0, 44.0]
            record("6. relax VM run on device rocm:0", "PASS" if ok else "FAIL",
                   f"output = {got} (expected [11.0, 22.0, 33.0, 44.0])")
        except Exception as exc:
            record("6. relax VM run on device rocm:0", "FAIL",
                   f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    else:
        record("6. relax VM run on device rocm:0", "SKIPPED",
               "no VMExecutable and/or no rocm device runtime")

    print()
    for step, status in RESULTS:
        print(f"   {status:10s} {step}")
    print()
    print(f"USE_ROCM={info.get('USE_ROCM')!r} -> conclusion: "
          + ("ROCm runtime available" if rocm_on and dev is not None
             and bool(dev.exist) else
             "NO ROCm device runtime in the installed TVM wheel"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
