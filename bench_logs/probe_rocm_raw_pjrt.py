#!/usr/bin/env python3
"""RAW PJRT viability probe of a plugin through etl's OWN ctypes driver.

Exercises, in order, and reports each result (pass/fail + VERBATIM error):
  1. plugin discovery / dlopen            (xla_util._find_plugin_path / _load_plugin)
  2. the version gate                     (_pjrt_c_api.verify_api: ABI major + struct_size)
  3. PJRT_Plugin_Initialize
  4. PJRT_Client_Create
  5. platform name/version + device inventory (kind/debug string via PJRT_Device_*)
  6. PJRT_Client_Compile of a trivial StableHLO module
  7. buffer_from_host + PJRT_LoadedExecutable_Execute + PJRT_Buffer_ToHostBuffer

Usage:
  ETL_PJRT_PLUGIN=/path/to/xla_rocm_plugin.so python3 probe_rocm_raw_pjrt.py
or  python3 probe_rocm_raw_pjrt.py /path/to/xla_rocm_plugin.so
"""
from __future__ import annotations

import ctypes
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import etl  # noqa: E402
from etl.backends.adapters import _pjrt_c_api as pjrt  # noqa: E402
from etl.backends.adapters import xla_util  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def record(step: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((step, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {step}" + (f"\n       {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Extra PJRT_Device_* structs — NOT vendored by _pjrt_c_api.py; defined here
# ONLY for the device inventory (field order from the PJRT C API header).
# ---------------------------------------------------------------------------
ADD_MLIR = """module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""

#: The probe-validated single-replica/single-partition CompileOptionsProto
#: (xla.py's default) — real plugins CHECK-crash on NULL.
DEFAULT_COMPILE_OPTIONS = bytes.fromhex("1a0420012801")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stop_after = None
    hard_exit = False
    for a in sys.argv[1:]:
        if a.startswith("--stop-after="):
            stop_after = a.split("=", 1)[1]
        elif a == "--hard-exit":
            hard_exit = True
    path = (args[0] if args
            else os.environ.get("ETL_PJRT_PLUGIN", ""))

    def stop(stage: str) -> bool:
        if stop_after != stage:
            return False
        print(f"\n[stop-after={stage}]"
              + ("  os._exit(0) — skipping interpreter + exit handlers"
                 if hard_exit else ""), flush=True)
        if hard_exit:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        return True

    print(f"python                : {sys.version.split()[0]}")
    print(f"argv plugin           : {path!r}")
    print(f"env ETL_PJRT_PLUGIN   : {os.environ.get('ETL_PJRT_PLUGIN')!r}")
    print(f"LD_LIBRARY_PATH       : {os.environ.get('LD_LIBRARY_PATH')!r}")
    print(f"stop_after            : {stop_after!r}  hard_exit: {hard_exit}")
    print()

    # --- 1. discovery ------------------------------------------------------
    try:
        found = xla_util._find_plugin_path({"plugin_path": path} if path else None)
        record("1. plugin discovery (_find_plugin_path)", True, repr(found))
    except Exception as exc:
        record("1. plugin discovery (_find_plugin_path)", False,
               f"{type(exc).__name__}: {exc}")
        return 1

    # --- 1b. raw dlopen + GetPjRtApi + struct inventory --------------------
    try:
        library, api = pjrt.load_plugin_library(found)
        v = api.pjrt_api_version
        required = pjrt._LAST_NEEDED_API_OFFSET + pjrt.sizeof(ctypes.c_void_p)
        record("1b. dlopen + GetPjRtApi + verify_api (raw)", True,
               f"plugin abi major/minor  = {v.major_version}/{v.minor_version}\n"
               f"       plugin struct_size      = {api.struct_size} bytes\n"
               f"       driver requires        >= {required} bytes (through "
               f"{pjrt._LAST_NEEDED_API_FIELD})\n"
               f"       driver PJRT_Api sizeof  = {pjrt.sizeof(pjrt.PJRT_Api)}\n"
               f"       vendored header         = major {pjrt.PJRT_API_MAJOR} / "
               f"minor {pjrt.PJRT_API_MINOR} @ {pjrt.HEADER_COMMIT[:12]} "
               f"({pjrt.HEADER_RETRIEVED})")
    except Exception as exc:
        record("1b. dlopen + GetPjRtApi + verify_api (raw)", False,
               f"{type(exc).__name__}: {exc}")
        return 1

    # --- 2. the driver's own ABI gate (wraps verify_api) -------------------
    try:
        plugin = xla_util._load_plugin({"plugin_path": found})
        record("2. xla_util._load_plugin (ABI gate + PJRT_Plugin_Initialize)",
               True, f"PjrtPlugin(path={plugin.path!r}) initialized")
    except Exception as exc:
        record("2. xla_util._load_plugin (ABI gate + PJRT_Plugin_Initialize)",
               False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 1
    if stop("load"):
        print("\n[stop-after=load]")
        return 0

    # --- 3. client create (the driver's own refcounted acquire path) --------
    try:
        client = xla_util.acquire_client(plugin)
        record("3. PJRT_Client_Create (xla_util.acquire_client)", True,
               f"client ptr = {client.ptr}")
    except Exception as exc:
        record("3. PJRT_Client_Create (xla_util.acquire_client)", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 1
    if stop("client"):
        print("\n[stop-after=client]")
        return 0

    # --- 4. platform info + device inventory --------------------------------
    # Only DRIVER-provided queries here (platform_info /
    # addressable_devices — both typed in the vendored PJRT_Api). The
    # untyped PJRT_Device_*/PJRT_DeviceDescription_* slots are exercised by
    # the separate probe_rocm_device_info.py, which hard-exits: the ROCm
    # plugin's device-description path proved FLAKY (it succeeded in
    # isolation, then segfaulted inside the larger probe), so it is kept out
    # of the viability matrix to avoid masking the deterministic results.
    try:
        name, version = client.platform_info()
        devices = client.addressable_devices()
        record("4. platform info + device inventory", True,
               f"platform_name    = {name!r}\n"
               f"       platform_version = {version!r}\n"
               f"       num addressable devices = {len(devices)}\n"
               f"       device pointers  = {tuple(devices)}\n"
               f"       (device kind/debug string -> probe_rocm_device_info.py)")
    except Exception as exc:
        record("4. platform info + device inventory", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    # --- 5. compile ---------------------------------------------------------
    try:
        loaded = client.compile(ADD_MLIR, DEFAULT_COMPILE_OPTIONS)
        record("5. PJRT_Client_Compile (trivial StableHLO add)", True,
               f"PJRT_LoadedExecutable ptr = {loaded.ptr}; "
               f"num_outputs = {loaded.num_outputs()}")
    except Exception as exc:
        record("5. PJRT_Client_Compile (trivial StableHLO add)", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 1

    if stop("compile"):
        print("\n[stop-after=compile]")
        return 0

    # --- 6. execute ---------------------------------------------------------
    try:
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        ba = client.buffer_from_host(a)
        bb = client.buffer_from_host(b)
        outs = loaded.execute([ba, bb])
        got = np.asarray(outs[0].to_host())
        want = a + b
        ok = bool(np.array_equal(got, want))
        record("6. execute (buffer_from_host + Execute + ToHostBuffer)", ok,
               f"a = {a.tolist()}, b = {b.tolist()}\n"
               f"       device output = {got.tolist()} "
               f"(shape {got.shape}, dtype {got.dtype})\n"
               f"       numpy expected = {want.tolist()}\n"
               f"       bit-exact: {ok}")
        for buf in [ba, bb, *outs]:
            try:
                buf.close()
            except Exception:
                pass
    except Exception as exc:
        record("6. execute (buffer_from_host + Execute + ToHostBuffer)", False,
               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    if stop("execute"):
        print("\n[stop-after=execute]")
        return 0

    # --- cleanup ------------------------------------------------------------
    print("[cleanup] closing loaded executable …", flush=True)
    try:
        loaded.close()
        print("[cleanup] loaded executable closed", flush=True)
    except Exception as exc:
        print(f"[warn] close(loaded executable) failed: {exc}")
    print("[cleanup] releasing client …", flush=True)
    try:
        xla_util.release_client(plugin)
        print("[cleanup] client released", flush=True)
    except Exception as exc:
        print(f"[warn] release_client failed: {exc}")

    print()
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"== RAW PJRT PROBE: {npass}/{len(RESULTS)} PASS ==")
    if hard_exit:
        # Skip Python finalization AND glibc exit handlers / C++ global
        # destructors — attributes a teardown-only crash.
        sys.stdout.flush()
        sys.stderr.flush()
        print("[hard-exit] os._exit(0) — skipping interpreter + exit handlers",
              flush=True)
        os._exit(0)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
