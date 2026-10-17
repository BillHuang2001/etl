"""Sub-cost breakdown of payload wrapping + ctypes scaffolding (SCRATCH)."""
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


def timeit(label, fn, n=500):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    print(f"  {label:38s} median {statistics.median(ts)/1e3:7.2f} us")
    return statistics.median(ts) / 1e3


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

    print("payload wrap sub-costs:")
    timeit("acquire_client+release_client pair", lambda: xla_util.release_client(plugin) if xla_util.acquire_client(plugin) is None else xla_util.release_client(plugin))
    timeit("XlaDevicePayload ctor (full)", lambda: xla.XlaDevicePayload(plugin, client, out, DEV))
    timeit("etl.core.Tensor(payload)", lambda: etl.core.Tensor(xla.XlaDevicePayload(plugin, client, out, DEV)))
    p = xla.XlaDevicePayload(plugin, client, out, DEV)
    timeit("core.Tensor(existing payload)", lambda: etl.core.Tensor(p))

    class _Holder:
        __slots__ = ("a", "b")
    def mk_holder():
        h = _Holder()
        h.a = out
        return h
    timeit("plain holder object", mk_holder)
    def mk_fin():
        return weakref.finalize(_Holder(), lambda *a: None, out, client)
    timeit("weakref.finalize(obj, cb, 2 args)", mk_fin)

    # buffer metadata queries (fresh _Buffer per output)
    def mk_buf():
        b = xla_util._Buffer(plugin, out.ptr)
        return b.shape, b.dtype
    timeit("fresh _Buffer + shape + dtype", mk_buf)
    def mk_buf_no_meta():
        return xla_util._Buffer(plugin, out.ptr)
    timeit("fresh _Buffer (lazy meta)", mk_buf_no_meta)
    timeit("buffer.shape (cached)", lambda: out.shape)
    timeit("buffer.dtype (cached)", lambda: out.dtype)
    timeit("buffer.close()", lambda: xla_util._Buffer(plugin, out.ptr).close(), n=200)

    print("ctypes scaffolding:")
    lm = raw.native_module
    timeit("num_outputs() plugin call", lambda: lm.num_outputs())
    timeit("sizeof(Execute_Args)", lambda: pjrt.sizeof(pjrt.PJRT_LoadedExecutable_Execute_Args))
    timeit("sizeof(ExecuteOptions)", lambda: pjrt.sizeof(pjrt.PJRT_ExecuteOptions))

    def build_args():
        n = len(bufs)
        num_out = 2
        dp = (ctypes.c_void_p * n)(*[b.ptr for b in bufs])
        al = (ctypes.c_void_p * 1)(ctypes.cast(dp, ctypes.c_void_p))
        oi = (ctypes.c_void_p * num_out)()
        ol = (ctypes.c_void_p * 1)(ctypes.cast(oi, ctypes.c_void_p))
        opt = pjrt.PJRT_ExecuteOptions(struct_size=pjrt.sizeof(pjrt.PJRT_ExecuteOptions))
        return pjrt.PJRT_LoadedExecutable_Execute_Args(
            struct_size=pjrt.sizeof(pjrt.PJRT_LoadedExecutable_Execute_Args),
            executable=lm.ptr, options=ctypes.pointer(opt),
            argument_lists=ctypes.cast(al, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            num_devices=1, num_args=n,
            output_lists=ctypes.cast(ol, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            device_complete_events=None, execute_device=raw._execute_device)
    timeit("full ctypes args construction", build_args)

    args = build_args()
    fn = plugin.api.PJRT_LoadedExecutable_Execute
    nb = []
    def raw_call():
        plugin._check(fn(ctypes.byref(args)), "exec")
        nb.append(out.ptr)
    timeit("PJRT execute call (args reused)", raw_call, n=200)
    timeit("lm.execute() full helper", lambda: [b.close() for b in lm.execute(bufs, execute_device=raw._execute_device)], n=200)

    for b in outs:
        b.close()
    os._exit(0)


main()
