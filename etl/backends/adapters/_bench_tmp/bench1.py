"""Baseline per-call breakdown of the xla device-resident run path. SCRATCH."""
import os, sys, time, statistics, gc

import numpy as np
import etl
from etl.backends.adapters import xla, xla_util

DEV = etl.core.Device("rocm", 0)


@etl.defn
def step(x, s):
    y = etl.tanh(x)
    z = etl.multiply(y, 0.5)
    return etl.add(z, s), etl.reduce_sum(y)


def specs(shape):
    return (etl.TensorSpec(shape, etl.float32), etl.TensorSpec(shape, etl.float32))


def bench(fn, n=40, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def main():
    shape = (1024, 50)
    t0 = time.perf_counter()
    exe = etl.build(step, *specs(shape), backend="xla", device=DEV)
    print(f"build: {(time.perf_counter()-t0)*1e3:.1f} ms")
    raw = exe.backend_executable
    print("exe type", type(exe), "raw", type(raw))

    x = etl.core.Tensor(np.random.rand(*shape).astype(np.float32)).to(DEV)
    s = etl.core.Tensor(np.random.rand(*shape).astype(np.float32)).to(DEV)

    for i in range(10):
        raw.run([x, s])

    m_raw, mn_raw = bench(lambda: raw.run([x, s]))
    print(f"raw.run (backend only):      median {m_raw:.3f} ms  min {mn_raw:.3f} ms")

    m_p, mn_p = bench(lambda: etl.run(exe, x, s))
    print(f"etl.run (pipeline):          median {m_p:.3f} ms  min {mn_p:.3f} ms")

    # ---- breakdown instrumentation
    import ctypes
    stats = {}

    def timeit(label, fn, n=200):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        stats[label] = (time.perf_counter() - t0) / n * 1e6  # us

    bufs = [x.data.buffer, s.data.buffer]
    timeit("execute(total)", lambda: [b.close() for b in raw.native_module.execute(bufs)])
    timeit("num_outputs", lambda: raw.native_module.num_outputs())
    buf0 = raw.native_module.execute(bufs)[0]
    stats["buffer.shape(cached)"] = -1
    timeit("buffer.shape", lambda: buf0.shape, n=200)
    timeit("buffer.dtype", lambda: buf0.dtype, n=200)

    timeit("payload_ctor", lambda: xla.XlaDevicePayload(raw._plugin, raw._client, buf0, DEV), n=200)

    def mk_finalize():
        import weakref
        o = object()
        return weakref.finalize(o, lambda: None)
    timeit("weakref.finalize", mk_finalize, n=500)

    timeit("_recorded_shapes x2", lambda: (raw._recorded_shapes("static_input_shapes", 2), raw._recorded_shapes("static_output_shapes", 2)), n=500)

    def validate():
        from etl.backends.options import validate_options
        validate_options(None, raw.KNOWN_OPTIONS, "xla", "run")
    timeit("validate_options", validate, n=500)

    def ctor_args():
        n = 2
        dp = (ctypes.c_void_p * n)(*[b.ptr for b in bufs])
        al = (ctypes.c_void_p * 1)(ctypes.cast(dp, ctypes.c_void_p))
        oi = (ctypes.c_void_p * 1)()
        ol = (ctypes.c_void_p * 1)(ctypes.cast(oi, ctypes.c_void_p))
        opt = xla._pjrt.PJRT_ExecuteOptions(struct_size=xla._pjrt.sizeof(xla._pjrt.PJRT_ExecuteOptions))
        a = xla._pjrt.PJRT_LoadedExecutable_Execute_Args(
            struct_size=xla._pjrt.sizeof(xla._pjrt.PJRT_LoadedExecutable_Execute_Args),
            executable=raw.native_module.ptr, options=ctypes.pointer(opt),
            argument_lists=ctypes.cast(al, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            num_devices=1, num_args=n,
            output_lists=ctypes.cast(ol, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
            device_complete_events=None, execute_device=raw._execute_device)
        return a
    timeit("ctypes args ctor", ctor_args, n=500)

    for k, v in stats.items():
        print(f"  {k:24s} {v:8.1f} us")
    buf0.close()
    os._exit(0)


main()
