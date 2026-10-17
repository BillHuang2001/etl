"""Phase breakdown of the xla device-resident run path (SCRATCH probe)."""
import ctypes
import gc
import os
import statistics
import time

import numpy as np

import etl
from etl.backends.adapters import xla, xla_util
from etl.backends.adapters import _pjrt_c_api as pjrt

DEV = etl.core.Device("rocm", 0)


@etl.defn
def step_small(x, s):
    y = etl.tanh(x)
    z = etl.multiply(y, 0.5)
    return etl.add(z, s), etl.reduce_sum(y)


@etl.defn
def step_evo(pop, vel, pbest, pbest_fit, gbest, fit, key, gen):
    # PSO-ish step over 8 leaves: 5 same-shape tensors + 3 small scalars.
    cand = etl.add(pop, vel)
    cand = etl.subtract(cand, etl.multiply(pop, 0.1))
    cand = etl.add(cand, etl.multiply(pbest, 0.2))
    new_vel = etl.multiply(cand, 0.9)
    new_fit = etl.reduce_sum(etl.multiply(cand, cand), axes=1)
    new_pbest_fit = etl.minimum(pbest_fit, new_fit)
    return (
        cand,
        new_vel,
        etl.add(pbest, 0.01),
        new_pbest_fit,
        gbest,
        new_fit,
        etl.add(key, 1),
        etl.add(gen, 1),
    )


def bench(fn, n=60, warmup=15):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(ts), min(ts)


def make_evo_args(pop_size=1000, dim=50):
    rng = np.random.default_rng(0)
    f32 = lambda shp: etl.core.Tensor(rng.random(shp, dtype=np.float32)).to(DEV)
    return [
        f32((pop_size, dim)),
        f32((pop_size, dim)),
        f32((pop_size, dim)),
        f32((pop_size,)),
        f32((dim,)),
        f32((pop_size,)),
        etl.core.Tensor(np.array(7, np.int64)).to(DEV),
        etl.core.Tensor(np.array(3, np.int32)).to(DEV),
    ]


def breakdown(raw, args, label):
    """Reproduce run() with timers around each section (device-resident)."""
    lm = raw.native_module
    plugin = raw._plugin
    client = raw._client
    specs = tuple(raw.signature.input_specs)
    out_specs = tuple(raw.signature.output_specs)
    shapes_in = raw._recorded_shapes("static_input_shapes", len(specs))
    shapes_out = raw._recorded_shapes("static_output_shapes", len(out_specs))

    sec = {}

    def acc(name, t0):
        sec[name] = sec.get(name, 0.0) + (time.perf_counter_ns() - t0) / 1e3

    N = 60
    for _ in range(12):
        raw.run(args)
    for _ in range(N):
        gc.disable()
        t = time.perf_counter_ns()
        from etl.backends.options import validate_options
        validate_options(None, raw.KNOWN_OPTIONS, "xla", "run")
        acc("a_validate_options", t)

        t = time.perf_counter_ns()
        rs = raw._recorded_shapes("static_input_shapes", len(specs))
        ro = raw._recorded_shapes("static_output_shapes", len(out_specs))
        acc("b_recorded_shapes", t)

        t = time.perf_counter_ns()
        bufs = []
        for i, (tensor, spec) in enumerate(zip(args, specs)):
            assert isinstance(tensor, etl.core.Tensor)
            assert tensor.dtype == spec.dtype
            raw._check_device_resident_input(i, tensor)
            bufs.append(tensor.data.buffer)
        acc("c_input_validate", t)

        t = time.perf_counter_ns()
        outs = lm.execute(bufs, execute_device=raw._execute_device)
        acc("d_execute", t)

        t = time.perf_counter_ns()
        tensors = [etl.core.Tensor(xla.XlaDevicePayload(plugin, client, b, raw.device)) for b in outs]
        acc("e_payload_wrap", t)

        t = time.perf_counter_ns()
        for b in outs:
            pass
        acc("g_none", t)

        t = time.perf_counter_ns()
        for i, (tensor, spec) in enumerate(zip(tensors, out_specs)):
            assert tensor.dtype == spec.dtype
            assert tuple(tensor.shape) == tuple(ro[i])
        acc("f_output_validate", t)
        gc.enable()
    print(f"--- breakdown {label} (us/call, {N} calls, {len(args)} in / {N} out)")
    tot = 0.0
    for k in sorted(sec):
        print(f"  {k:22s} {sec[k]/N:8.1f}")
        tot += sec[k] / N
    print(f"  {'TOTAL':22s} {tot:8.1f}")


def pjrt_call_only(raw, bufs):
    """Time ONLY the PJRT_LoadedExecutable_Execute call, args prebuilt."""
    lm = raw.native_module
    n = len(bufs)
    num_out = lm.num_outputs()
    dp = (ctypes.c_void_p * n)(*[b.ptr for b in bufs])
    al = (ctypes.c_void_p * 1)(ctypes.cast(dp, ctypes.c_void_p))
    oi = (ctypes.c_void_p * num_out)()
    ol = (ctypes.c_void_p * 1)(ctypes.cast(oi, ctypes.c_void_p))
    opt = pjrt.PJRT_ExecuteOptions(struct_size=pjrt.sizeof(pjrt.PJRT_ExecuteOptions))
    args = pjrt.PJRT_LoadedExecutable_Execute_Args(
        struct_size=pjrt.sizeof(pjrt.PJRT_LoadedExecutable_Execute_Args),
        executable=lm.ptr,
        options=ctypes.pointer(opt),
        argument_lists=ctypes.cast(al, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
        num_devices=1,
        num_args=n,
        output_lists=ctypes.cast(ol, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))),
        device_complete_events=None,
        execute_device=raw._execute_device,
    )
    fn = raw._plugin.api.PJRT_LoadedExecutable_Execute
    outbufs = []

    def call():
        raw._plugin._check(fn(ctypes.byref(args)), "exec")
        for i in range(num_out):
            outbufs.append(xla_util._Buffer(raw._plugin, oi[i]))

    for _ in range(10):
        call()
    m, mn = bench(call, n=40, warmup=10)
    for b in outbufs:
        b.close()
    print(f"  PJRT execute call only ({n} args): median {m*1000:.1f} us min {mn*1000:.1f} us")


def main():
    small = etl.build(
        step_small,
        etl.TensorSpec((1024, 50), etl.float32),
        etl.TensorSpec((1024, 50), etl.float32),
        backend="xla",
        device=DEV,
    ).backend_executable
    rng = np.random.default_rng(1)
    xa = etl.core.Tensor(rng.random((1024, 50), dtype=np.float32)).to(DEV)
    xb = etl.core.Tensor(rng.random((1024, 50), dtype=np.float32)).to(DEV)
    m, mn = bench(lambda: small.run([xa, xb]))
    print(f"small 2x2 raw.run: median {m*1000:.1f} us min {mn*1000:.1f} us")
    breakdown(small, [xa, xb], "small")
    pjrt_call_only(small, [xa.data.buffer, xb.data.buffer])

    evo = etl.build(
        step_evo,
        *[
            etl.TensorSpec((1000, 50), etl.float32),
            etl.TensorSpec((1000, 50), etl.float32),
            etl.TensorSpec((1000, 50), etl.float32),
            etl.TensorSpec((1000,), etl.float32),
            etl.TensorSpec((50,), etl.float32),
            etl.TensorSpec((1000,), etl.float32),
            etl.TensorSpec((), etl.int64),
            etl.TensorSpec((), etl.int32),
        ],
        backend="xla",
        device=DEV,
    ).backend_executable
    args = make_evo_args()
    m, mn = bench(lambda: evo.run(args))
    print(f"evox-like 8x8 raw.run: median {m*1000:.1f} us min {mn*1000:.1f} us")
    breakdown(evo, args, "evox")
    pjrt_call_only(evo, [a.data.buffer for a in args])

    # how expensive is a per-step D2H read-back (what evox's monitor does)?
    out = evo.run(args)[0]
    m, mn = bench(lambda: out.to(etl.core.Device("cpu", 0)))
    print(f"per-step D2H .to(cpu) of (1000,50) f32: median {m*1000:.1f} us min {mn*1000:.1f} us")
    tiny = evo.run(args)[6]
    m, mn = bench(lambda: tiny.to(etl.core.Device("cpu", 0)))
    print(f"per-step D2H .to(cpu) of rank-0 i64: median {m*1000:.1f} us min {mn*1000:.1f} us")
    del out, tiny
    gc.collect()

    os._exit(0)


main()
