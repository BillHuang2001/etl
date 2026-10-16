#!/usr/bin/env python3
"""Minimal PURE-CTYPES PJRT probe of a plugin — imports NO etl code at all.

Purpose: attribute a teardown crash. The etl-driver probe
(``probe_rocm_raw_pjrt.py``) shows a SIGABRT/SIGSEGV at process exit as soon
as a PJRT client was created, while ``os._exit(0)`` is clean. This script
reproduces the same client creation through a hand-rolled ctypes struct (only
the leading PJRT_Api fields: the metadata + 11 function-pointer slots — enough
for PJRT_Plugin_Initialize at index 3 and PJRT_Client_Create at index 10) so a
crash here cannot be blamed on etl's driver or its handle wrappers.

Usage:
  python3 probe_rocm_teardown_minimal.py <plugin.so> [--destroy] [--hard-exit]
"""
from __future__ import annotations

import ctypes
import os
import sys

SO = (sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
      else os.environ.get("ETL_PJRT_PLUGIN", ""))
DESTROY = "--destroy" in sys.argv
HARD_EXIT = "--hard-exit" in sys.argv


class ApiVersion(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]


#: struct_size + extension_start + version (24 B) + 12 function-pointer slots
#: (indices 0..11 — PJRT_Plugin_Initialize is 3, PJRT_Client_Create is 10,
#: PJRT_Client_Destroy is 11).
class ApiHead(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("version", ApiVersion),
    ] + [(f"fn{i}", ctypes.c_void_p) for i in range(12)]


class PluginInitArgs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.c_void_p)]


class ClientCreateArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("create_options", ctypes.c_void_p),
        ("num_options", ctypes.c_size_t),
        ("kv_get_callback", ctypes.c_void_p),
        ("kv_get_user_arg", ctypes.c_void_p),
        ("kv_put_callback", ctypes.c_void_p),
        ("kv_put_user_arg", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("kv_try_get_callback", ctypes.c_void_p),
        ("kv_try_get_user_arg", ctypes.c_void_p),
    ]


class ClientDestroyArgs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_size_t),
                ("extension_start", ctypes.c_void_p),
                ("client", ctypes.c_void_p)]


def main() -> int:
    print(f"pure-ctypes probe (no etl): plugin = {SO!r}", flush=True)
    lib = ctypes.CDLL(SO)
    for sym in ("GetPjRtApi", "GetPjrtApi"):
        try:
            get_api = getattr(lib, sym)
            print(f"  exported symbol: {sym}()", flush=True)
            break
        except AttributeError:
            continue
    else:
        print("  no GetPjRtApi()/GetPjrtApi() symbol", flush=True)
        return 1
    get_api.restype = ctypes.POINTER(ApiHead)
    api = get_api()
    head = api.contents
    print(f"  PJRT_Api.struct_size = {head.struct_size}, "
          f"abi version = {head.version.major_version}."
          f"{head.version.minor_version}", flush=True)

    init = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(PluginInitArgs))(
        head.fn3)
    err = init(ctypes.byref(PluginInitArgs(
        struct_size=ctypes.sizeof(PluginInitArgs), extension_start=None)))
    print(f"  PJRT_Plugin_Initialize -> err={err}", flush=True)
    if err:
        return 2

    create = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ClientCreateArgs))(
        head.fn10)
    args = ClientCreateArgs(struct_size=ctypes.sizeof(ClientCreateArgs))
    err = create(ctypes.byref(args))
    print(f"  PJRT_Client_Create -> err={err}, client={args.client}", flush=True)
    if err:
        return 3

    if DESTROY:
        destroy = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(ClientDestroyArgs))(head.fn11)
        dargs = ClientDestroyArgs(struct_size=ctypes.sizeof(ClientDestroyArgs),
                                  client=args.client)
        err = destroy(ctypes.byref(dargs))
        print(f"  PJRT_Client_Destroy -> err={err}", flush=True)

    sys.stdout.flush()
    if HARD_EXIT:
        print("  [hard-exit] os._exit(0)", flush=True)
        sys.stdout.flush()
        os._exit(0)
    print("  returning (normal interpreter exit) …", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
