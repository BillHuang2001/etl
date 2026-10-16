#!/usr/bin/env python3
"""Device inventory of a PJRT plugin through etl's driver (+ the untyped slots).

The vendored ``_pjrt_c_api.PJRT_Api`` types only the fields etl's driver calls;
``PJRT_Device_*`` / ``PJRT_DeviceDescription_*`` are raw untyped pointers, so
this probe casts them itself (driver untouched) to read:

  * ``PJRT_Client_PlatformName`` / ``PlatformVersion`` (driver API)
  * ``PJRT_Client_AddressableDevices``               (driver API)
  * ``PJRT_Device_GetDescription`` -> ``PJRT_DeviceDescription_Kind`` (e.g.
    "AMD Instinct MI300X VF") and ``..._DebugString`` (e.g. "rocm:0")

It ends with ``os._exit(0)`` on purpose: the ROCm plugin corrupts the heap at
interpreter exit (see probe_rocm_teardown_minimal.py), and this probe is only
about device metadata.

Usage:
  ETL_PJRT_PLUGIN=/path/to/plugin.so python3 probe_rocm_device_info.py
  python3 bench_logs/probe_rocm_device_info.py --repeat 5   # flakiness check
"""
from __future__ import annotations

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.backends.adapters import xla_util  # noqa: E402


class DevGetDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),  # out
    ]


class DevDescKind(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
        ("kind", ctypes.c_void_p),  # out
        ("kind_size", ctypes.c_size_t),  # out
    ]


class DevDescDebugString(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("device_description", ctypes.c_void_p),
        ("debug_string", ctypes.c_void_p),  # out
        ("debug_string_size", ctypes.c_size_t),  # out
    ]


def run_once(path: str, round_no: int) -> None:
    plugin = xla_util._load_plugin({"plugin_path": path})
    client = xla_util.acquire_client(plugin)
    try:
        name, version = client.platform_info()
        devices = client.addressable_devices()
        print(f"[round {round_no}] platform_name={name!r} "
              f"platform_version={version!r} ndev={len(devices)}", flush=True)

        get_desc = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(DevGetDesc))(
                plugin.api.PJRT_Device_GetDescription)
        get_kind = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(DevDescKind))(
                plugin.api.PJRT_DeviceDescription_Kind)
        get_dbg = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(DevDescDebugString))(
                plugin.api.PJRT_DeviceDescription_DebugString)

        for i, d in enumerate(devices):
            dargs = DevGetDesc(struct_size=ctypes.sizeof(DevGetDesc), device=d)
            err = get_desc(ctypes.byref(dargs))
            if err:
                print(f"  device[{i}] PJRT_Device_GetDescription -> err={err}",
                      flush=True)
                plugin._destroy_error(err)
                continue
            desc = dargs.device_description
            kargs = DevDescKind(struct_size=ctypes.sizeof(DevDescKind),
                                device_description=desc)
            err = get_kind(ctypes.byref(kargs))
            kind = (ctypes.string_at(kargs.kind, kargs.kind_size).decode()
                    if not err else f"<err {err}>")
            if err:
                plugin._destroy_error(err)
            sargs = DevDescDebugString(
                struct_size=ctypes.sizeof(DevDescDebugString),
                device_description=desc)
            err = get_dbg(ctypes.byref(sargs))
            dbg = (ctypes.string_at(sargs.debug_string,
                                    sargs.debug_string_size).decode()
                   if not err else f"<err {err}>")
            if err:
                plugin._destroy_error(err)
            print(f"  device[{i}] ptr={d} kind={kind!r} debug_string={dbg!r}",
                  flush=True)
    finally:
        xla_util.release_client(plugin)


def main() -> int:
    path = os.environ.get("ETL_PJRT_PLUGIN") or (
        sys.argv[1] if len(sys.argv) > 1
        and not sys.argv[1].startswith("--") else "")
    repeat = 1
    for a in sys.argv[1:]:
        if a.startswith("--repeat="):
            repeat = int(a.split("=", 1)[1])
    if not path:
        print("set ETL_PJRT_PLUGIN to the plugin .so")
        return 2
    for r in range(1, repeat + 1):
        run_once(path, r)
    sys.stdout.flush()
    os._exit(0)  # skip the plugin's crashing exit path (see module docstring)


if __name__ == "__main__":
    raise SystemExit(main())
