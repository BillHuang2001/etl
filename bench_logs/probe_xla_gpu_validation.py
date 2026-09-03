#!/usr/bin/env python3
"""XLA adapter REAL-plugin GPU validation (jax_cuda12_pjrt 0.10.2 xla_cuda_plugin.so).

Runs the required end-to-end matrix of the xla-adapter real-plugin validation
against the numpy reference backend on CUDA (RTX A6000):

  0. rank-0 input staging probe (the 0.4.38-era blocker — verify on 0.10.2)
  A. PSO-shaped graph (split + uniform + elementwise move + argmin + gather)
  B. DE-shaped graph (split_n normal/randint + gather 3 parents + select)
  C. OpenES-shaped graph (normal + dot + reduce + gradient update)
  D. RNG bit-exactness for threefry2x32 and philox4x32_10 (native
     stablehlo.rng_bit_generator paths on xla)
  E. non-dominate-rank-shaped while_loop ((n,n) bool dominance matrix +
     intra-body reduce + bool->int32 casts)

Tolerances: bit-exact for elementwise/RNG/int ops; fp32 reduce/matmul
accumulation-order tolerance 1e-6 (asserted at 1e-4, actuals reported).

Env (set in the shell BEFORE launching — all must exist at plugin load):
  ETL_PJRT_PLUGIN=.../jax_plugins/xla_cuda12/xla_cuda_plugin.so
  LD_LIBRARY_PATH=<dir with libcudnn.so.9 >= 9.8.0>  (the plugin is compiled
      against cuDNN 9.8.0; with the venv's 9.1.0 every compile fails
      RET_CHECK dnn_support != nullptr)
  CUDA_VISIBLE_DEVICES=<free gpu>  (the adapter executes on
      client.addressable_devices()[0] = the visible device)
  PATH must contain ptxas (the plugin invokes it at compile time).

Usage:
  uv run python bench_logs/probe_xla_gpu_validation.py [--numpy-only] [--skip E]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import etl  # noqa: E402
from etl import core  # noqa: E402

PLUGIN_ENV = "ETL_PJRT_PLUGIN"
RESULTS = {"pass": [], "fail": []}


def check(name, got, expected, mode="bit", tol=1e-4):
    got = np.asarray(got)
    expected = np.asarray(expected)
    ok_shape = got.shape == expected.shape
    ok_dtype = got.dtype == expected.dtype
    if not (ok_shape and ok_dtype):
        RESULTS["fail"].append(name)
        print(f"FAIL {name}: shape {got.shape}/{expected.shape} "
              f"dtype {got.dtype}/{expected.dtype}")
        return
    if mode == "bit":
        ok = (
            np.ascontiguousarray(got).ravel().view(np.uint8).tobytes()
            == np.ascontiguousarray(expected).ravel().view(np.uint8).tobytes()
        )
    else:
        max_abs = float(np.max(np.abs(got.astype(np.float64)
                                       - expected.astype(np.float64))))
        ok = max_abs <= tol
    max_abs = float(np.max(np.abs(got.astype(np.float64)
                                   - expected.astype(np.float64))))
    RESULTS["pass" if ok else "fail"].append(name)
    print(f"{'PASS' if ok else 'FAIL'} {name}: max_abs={max_abs:.3e} "
          f"({mode}, tol={tol})")


def run_both(fn, *args, numpy_only=False):
    """numpy-backend reference + xla-backend run; return (expected, actual)."""
    t0 = time.time()
    expected = etl.evaluate(fn, *args)
    t_np = time.time() - t0
    if numpy_only:
        return expected, None, t_np, 0.0
    t0 = time.time()
    actual = etl.evaluate(fn, *args, backend="xla")
    t_xla = time.time() - t0
    return expected, actual, t_np, t_xla


# ---------------------------------------------------------------------------
# 0. rank-0 input staging probe (the 0.4.38 plugin staged rank-0 as [1])
# ---------------------------------------------------------------------------

def section0(numpy_only):
    print("\n[0] rank-0 input staging probe")

    @etl.defn
    def ident(k):
        return k

    spec = etl.TensorSpec((), etl.int64)
    exe_np = etl.build(ident, spec)
    expected = np.asarray(etl.run(exe_np, np.array(42, np.int64)).numpy())
    if numpy_only:
        check("0.rank0_numpy", expected, np.array(42))
        return
    exe_xla = etl.build(ident, spec, backend="xla")
    actual = np.asarray(etl.run(exe_xla, np.array(42, np.int64)).numpy())
    check("0.rank0_ident_xla", actual, np.array(42))


# ---------------------------------------------------------------------------
# A. PSO-shaped: split + uniform + elementwise move + argmin + gather
# ---------------------------------------------------------------------------

N, D = 32, 10


@etl.defn
def pso_step(key, x, v, pbest, gbest):
    k1, k2 = etl.random.split(key)
    r1 = etl.random.uniform(k1, shape=(N, D))
    r2 = etl.random.uniform(k2, shape=(N, D))
    v_new = 0.7 * v + 1.5 * r1 * (pbest - x) + 1.5 * r2 * (gbest - x)
    x_new = x + v_new
    fitness = etl.sum(x_new * x_new, axes=1)
    best_idx = etl.argmin(fitness, axis=None)
    gbest_new = etl.gather(x_new, best_idx, axis=0)
    return k2, x_new, v_new, fitness, gbest_new


def sectionA(numpy_only):
    print("\n[A] PSO-shaped graph")
    rng = np.random.default_rng(0)
    key = etl.random.key(42)
    x = rng.uniform(-5.12, 5.12, (N, D)).astype(np.float32)
    v = rng.standard_normal((N, D)).astype(np.float32)
    pbest = x.copy()
    gbest = rng.standard_normal((D,)).astype(np.float32)
    expected, actual, t_np, t_xla = run_both(
        pso_step, key, x, v, pbest, gbest, numpy_only=numpy_only)
    if numpy_only:
        print(f"  numpy reference OK ({t_np*1000:.0f} ms); "
              "skipping xla (--numpy-only)")
        return
    print(f"  numpy {t_np*1000:.0f} ms / xla {t_xla*1000:.0f} ms")
    check("A.key", actual[0].numpy(), expected[0].numpy(), mode="bit")
    check("A.x_new", actual[1].numpy(), expected[1].numpy(), mode="bit")
    check("A.v_new", actual[2].numpy(), expected[2].numpy(), mode="bit")
    check("A.fitness", actual[3].numpy(), expected[3].numpy(), mode="tol")
    check("A.gbest_new", actual[4].numpy(), expected[4].numpy(), mode="bit")


# ---------------------------------------------------------------------------
# B. DE-shaped: split_n normal/randint + gather 3 parents + select
# ---------------------------------------------------------------------------

@etl.defn
def de_step(key, pop):
    k1, k2, k3 = etl.random.split_n(key, 3)
    idx = etl.random.randint(k1, shape=(3,), low=0, high=N)
    z = etl.random.normal(k2, shape=(D,))
    a = etl.gather(pop, idx[0], axis=0)
    b = etl.gather(pop, idx[1], axis=0)
    c = etl.gather(pop, idx[2], axis=0)
    mutant = a + 0.5 * (b - c) + 0.1 * z
    mask = etl.random.randint(k3, shape=(D,), low=0, high=100) < 30
    trial = etl.select(mask, mutant, pop[0])
    return k3, trial, mutant


def sectionB(numpy_only):
    print("\n[B] DE-shaped graph")
    rng = np.random.default_rng(1)
    key = etl.random.key(7)
    pop = rng.uniform(-5.12, 5.12, (N, D)).astype(np.float32)
    expected, actual, t_np, t_xla = run_both(
        de_step, key, pop, numpy_only=numpy_only)
    if numpy_only:
        print(f"  numpy reference OK ({t_np*1000:.0f} ms)")
        return
    print(f"  numpy {t_np*1000:.0f} ms / xla {t_xla*1000:.0f} ms")
    check("B.key", actual[0].numpy(), expected[0].numpy(), mode="bit")
    check("B.trial", actual[1].numpy(), expected[1].numpy(), mode="bit")
    check("B.mutant", actual[2].numpy(), expected[2].numpy(), mode="bit")


# ---------------------------------------------------------------------------
# C. OpenES-shaped: normal + dot + reduce + gradient update
# ---------------------------------------------------------------------------

P = 16


@etl.defn
def openes_step(key, theta):
    z = etl.random.normal(key, shape=(P, D))
    eps = z * 0.1
    scores = etl.reshape(etl.dot(eps, etl.reshape(theta, (D, 1))), (P,))
    grad = etl.mean(eps * etl.expand_dims(scores, 1), axes=0)  # (D,)
    theta_new = theta + 0.01 * grad
    return theta_new, scores


def sectionC(numpy_only):
    print("\n[C] OpenES-shaped graph")
    rng = np.random.default_rng(2)
    key = etl.random.key(99)
    theta = rng.standard_normal((D,)).astype(np.float32)
    expected, actual, t_np, t_xla = run_both(
        openes_step, key, theta, numpy_only=numpy_only)
    if numpy_only:
        print(f"  numpy reference OK ({t_np*1000:.0f} ms)")
        return
    print(f"  numpy {t_np*1000:.0f} ms / xla {t_xla*1000:.0f} ms")
    check("C.theta_new", actual[0].numpy(), expected[0].numpy(), mode="tol")
    check("C.scores", actual[1].numpy(), expected[1].numpy(), mode="tol")


# ---------------------------------------------------------------------------
# D. RNG bit-exactness: native rng_bit_generator THREE_FRY / PHILOX paths
# ---------------------------------------------------------------------------

@etl.defn
def rng_suite(k):
    k1, k2 = etl.random.split(k)
    u32 = etl.random.uniform(k2, shape=(1024,))
    u64 = etl.random.uniform(k1, shape=(512,), dtype=etl.float64)
    n64 = etl.random.normal(k1, shape=(1000,), dtype=etl.float64)
    n32 = etl.random.normal(k2, shape=(2048,))
    ri = etl.random.randint(k1, shape=(500,), low=-1000, high=1000)
    perm = etl.random.permutation(k2, 64)
    return k2, u32, u64, n64, n32, ri, perm


def sectionD(numpy_only):
    for alg in ("threefry2x32", "philox4x32_10"):
        print(f"\n[D] RNG bit-exactness — {alg}")
        key = etl.random.key(12345, algorithm=alg)
        expected, actual, t_np, t_xla = run_both(
            rng_suite, key, numpy_only=numpy_only)
        if numpy_only:
            print(f"  numpy reference OK ({t_np*1000:.0f} ms)")
            continue
        print(f"  numpy {t_np*1000:.0f} ms / xla {t_xla*1000:.0f} ms")
        check(f"D.{alg}.key_mix", actual[0].numpy(), expected[0].numpy(),
              mode="bit")
        check(f"D.{alg}.uniform_f32", actual[1].numpy(), expected[1].numpy(),
              mode="bit")
        check(f"D.{alg}.uniform_f64", actual[2].numpy(), expected[2].numpy(),
              mode="bit")
        check(f"D.{alg}.normal_f64", actual[3].numpy(), expected[3].numpy(),
              mode="bit")
        check(f"D.{alg}.normal_f32", actual[4].numpy(), expected[4].numpy(),
              mode="tol", tol=1e-4)  # documented Box-Muller f32 fast path
        check(f"D.{alg}.randint", actual[5].numpy(), expected[5].numpy(),
              mode="bit")
        check(f"D.{alg}.permutation", actual[6].numpy(), expected[6].numpy(),
              mode="bit")


# ---------------------------------------------------------------------------
# E. non-dominate-rank-shaped while_loop: (n,n) bool dominance matrix +
#    intra-body reduce + bool->int32 casts
# ---------------------------------------------------------------------------

@etl.defn
def non_dominate_rank(fit):
    n = fit.shape[0]
    m = fit.shape[1]
    # dominance matrix: dom[i, j] = fit[i] dominates fit[j] (bool, (n, n))
    le = etl.reshape(fit, (n, 1, m)) <= etl.reshape(fit, (1, n, m))
    lt = etl.reshape(fit, (n, 1, m)) < etl.reshape(fit, (1, n, m))
    le_all = etl.min(etl.cast(le, etl.int32), axes=2) == 1  # all(<=)
    lt_any = etl.max(etl.cast(lt, etl.int32), axes=2) == 1  # any(<)
    dom = etl.logical_and(le_all, lt_any)  # (n, n) bool
    i0 = etl.constant(etl.tensor(0, dtype=etl.int32))
    rank0 = etl.constant(etl.zeros((n,), dtype=etl.int32))

    def cond(i, rank):
        return i < n

    def body(i, rank):
        col = etl.gather(dom, i, axis=1)  # gather column i (dynamic index)
        r_i = 1 + etl.sum(etl.cast(col, etl.int32), axes=0)  # bool->int32
        rank = etl.scatter(rank, i, r_i, axis=0)
        return i + 1, rank

    _, rank = etl.while_loop(cond, body, (i0, rank0))
    return rank


def sectionE(numpy_only):
    print("\n[E] non-dominate-rank-shaped while_loop")
    rng = np.random.default_rng(3)
    fit = rng.uniform(-2.0, 2.0, (N, 3)).astype(np.float32)
    expected, actual, t_np, t_xla = run_both(
        non_dominate_rank, fit, numpy_only=numpy_only)
    if numpy_only:
        print(f"  numpy reference OK ({t_np*1000:.0f} ms)")
        return
    print(f"  numpy {t_np*1000:.0f} ms / xla {t_xla*1000:.0f} ms")
    check("E.non_dominate_rank", actual.numpy(), expected.numpy(), mode="bit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy-only", action="store_true",
                    help="run only the numpy-backend reference (no plugin)")
    ap.add_argument("--skip", default="", help="comma-separated sections to skip")
    args = ap.parse_args()
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    if not args.numpy_only and not os.environ.get(PLUGIN_ENV):
        raise SystemExit(
            f"set {PLUGIN_ENV} to the xla_cuda_plugin.so path (or use "
            "--numpy-only for a reference smoke)")
    if not args.numpy_only:
        backend = etl.backends.get("xla")  # activates + probes the plugin
        print(f"xla backend registered; capabilities={backend.capabilities}")
    if "0" not in skip:
        section0(args.numpy_only)
    if "A" not in skip:
        sectionA(args.numpy_only)
    if "B" not in skip:
        sectionB(args.numpy_only)
    if "C" not in skip:
        sectionC(args.numpy_only)
    if "D" not in skip:
        sectionD(args.numpy_only)
    if "E" not in skip:
        sectionE(args.numpy_only)
    print(f"\n=== {len(RESULTS['pass'])} PASS / {len(RESULTS['fail'])} FAIL ===")
    if RESULTS["fail"]:
        print("FAILED:", ", ".join(RESULTS["fail"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
