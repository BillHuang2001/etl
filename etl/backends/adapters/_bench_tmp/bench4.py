"""FFI metadata cost + payload ctor details (SCRATCH)."""
import ctypes
import os
import statistics
import time
import weakref

import numpy as np

import etl
from etl.backends.adapters import xla, xla_util
from etl.backends.adapters import _pjrt_c_api as pjrt

DEV = etl.core.Device("rocm", 0)


@etl.defn
def step_small(x, s):
    y = etl.tanh(x)
    return etl.add(etl.multiply(y, 0.5), s), etl.reduce_sum(y)


def timeit(label, fn, n=400):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    med = statistics.median(ts) / 1e3
    print(f"  {label:42s} median {med:7.2f} us")
    return med


def main():
    exe = etl.build(
        step_small,
        etl.TensorSpec((1024, 50), etl.float32),
        etl.TensorSpec((1024, 50), etl.float32),
        backend="xla",
        device=DEV,
    )
    raw = exe.backend_executable
    rng = np.random.default_rng(0)
    xa = etl.core.Tensor(rng.random((1024, 50), dtype=np.float32)).to(DEV)
    xb = etl.core.Tensor(rng.random((1024, 50), dtype=np.float32)).to(DEV)
    bufs = [xa.data.buffer, xb.data.buffer]
    outs = raw.native_module.execute(bufs, execute_device=raw._execute_device)
    out = outs[0]
    plugin, client = raw._plugin, raw._client
    keep = []  # keep every payload alive: no finalizers during measurement

    print("single plugin buffer FFI / buffer wrapper costs:")
    timeit("fresh _Buffer(ptr) only", lambda: xla_util._Buffer(plugin, out.ptr))
    timeit("fresh _Buffer + .shape", lambda: xla_util._Buffer(plugin, out.ptr).shape)
    timeit("fresh _Buffer + .dtype", lambda: xla_util._Buffer(plugin, out.ptr).dtype)
    timeit("fresh _Buffer + .shape + .dtype", lambda: (lambda b: (b.shape, b.dtype))(xla_util._Buffer(plugin, out.ptr)))
    timeit("PJRT_Buffer_Dimensions (already cached)", lambda: out.shape)
    timeit("num_outputs FFI", lambda: raw.native_module.num_outputs())

    p = xla.XlaDevicePayload(plugin, client, out, DEV)
    keep.append(p)
    timeit("core.Tensor(payload) [meta cached]", lambda: etl.core.Tensor(p))
    timeit("XlaDevicePayload ctor [meta cached]", lambda: keep.append(xla.XlaDevicePayload(plugin, client, out, DEV)))

    class _Holder:
        __slots__ = ("a",)
    def mk_fin():
        h = _Holder()
        h.a = out
        return weakref.finalize(h, lambda *a: None, out, client)
    timeit("weakref.finalize(obj, cb, 2 args)", mk_fin)

    print("ctypes scaffolding:")
    lm = raw.native_module
    sz_exec = pjrt.sizeof(pjrt.PJRT_LoadedExecutable_Execute_Args)
    sz_opt = pjrt.sizeof(pjrt.PJRT_ExecuteOptions)

    def build_args():
        n = 2
        num_out = 2
        dp = (ctypes.c_void_p * n)(*[b.ptr for b in bufs])
        al = (ctypes.c_void_p * 1)(ctypes.cast(dp, ctypes.c_void_p))
        oi = (ctypes.c_void_p * num_out)()
        ol = (ctypes.c_void_p * 1)(ctypes.cast(oi, ctypes.c_void_p))
        opt = pjrt.PJRT_ExecuteOptions(struct_size=sz_opt)
        return pjrt.PJRT_LoadedExecutable_Execute_Args(
            struct_size=sz_exec, executable=lm.ptr, options=ctypes.pointer(opt),
            argument_lists=ctypes.cast(al, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            num_devices=1, num_args=n,
            output_lists=ctypes.cast(ol, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            device_complete_events=None, execute_device=raw._execute_device)
    timeit("full ctypes args construction", build_args)

    # cached scaffolding: fill-in only
    args = build_args()
    adapter_keepalive = build_args.__dict__ if False else None
    fill_ts = []
    for _ in range(400):
        t0 = time.perf_counter_ns()
        dp_arr = (ctypes.c_void_p * 2).from_address(
            ctypes.cast(args.argument_lists[0], ctypes.c_void_p).value
        )
        dp_arr[0] = bufs[0].ptr
        dp_arr[1] = bufs[1].ptr
        fill_ts.append(time.perf_counter_ns() - t0)
    print(f"  {'fill cached arg ptrs':42s} median {statistics.median(fill_ts)/1e3:7.2f} us")

    fn = plugin.api.PJRT_LoadedExecutable_Execute
    def raw_call():
        plugin._check(fn(ctypes.byref(args)), "exec")
    timeit("PJRT execute (cached args)", raw_call, n=200)

    # whole lm.execute for reference
    timeit("lm.execute (current helper)", lambda: [b.close() for b in lm.execute(bufs, execute_device=raw._execute_device)], n=200)

    for b in outs:
        b.close()
    os._exit(0)


main()
