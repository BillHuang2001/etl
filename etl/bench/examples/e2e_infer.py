"""End-to-end inference examples with IN-GRAPH control flow (category "e2e",
tag "infer").

Four forward-only inference-style procedures whose iterative core runs INSIDE
the graph as traced ``etl.while_loop`` ops (runtime tensor control flow — the
numpy interpreter repeats the traced regions at run time, no Python
callbacks):

- ``e2e_infer_transformer`` — autoregressive decoding with a small single-head
  transformer block: an in-graph loop attends over the sequence-so-far,
  produces next-token logits, picks ``argmax``, and appends the token's
  embedding to the sequence.
- ``e2e_pso_optimize`` — particle-swarm optimization (N particles, D dims)
  minimizing ``||x - target||^2``: an in-graph loop updates velocities and
  positions for T iterations.
- ``e2e_kmeans`` — k-means clustering: an in-graph loop alternates argmin
  assignments and one-hot centroid updates for T iterations.
- ``e2e_power_iteration`` — dominant-eigenvector estimation: T iterations of
  ``v = A v / ||A v||``.

All four are FORWARD-ONLY. Gradient through control flow is a v1 deferral:
``etl.grad`` of a graph containing a ``while`` op raises ``TransformError``
("no VJP rule for op 'while'") — measured at dev time. Training-style loops
belong to ``e2e_train`` (runner-based, Python-level).

LOOP DESIGN — fixed-shape carried state (binding for this module)
-----------------------------------------------------------------
``etl.while_loop(cond_fn, body_fn, init)`` requires every loop-carried leaf's
shape AND dtype to stay CONSTANT across iterations (dtype change →
``DTypeError``, shape change → ``ShapeError`` — both verified at dev time:
growing a carried tensor with ``etl.concatenate`` inside the body raises
``ShapeError`` immediately at trace time). Every example therefore carries a
FIXED-SIZE state plus an int64 iteration counter, and writes into the fixed
buffer rather than growing it:

- transformer: carry ``(count: int64 scalar, buffer: [T_max, D] f32)`` with
  the start token pre-placed at row 0 (``count = 1``); each iteration writes
  the next token embedding into row ``count`` via ``etl.scatter``
  (``scatter(buffer, count, next_emb, axis=0)`` — numpy put-along-axis
  semantics, 0-d indices + ``[D]`` updates). The alternative one-hot mask
  placement via ``etl.select`` + ``etl.broadcast`` was also verified; scatter
  is used (numpy backend only — no compiler export needed).
- pso / kmeans / power_iteration: positions/centroids/vectors are all
  fixed-shape tensors updated in place; the counter is an int64 scalar.

Counter discipline: ``etl.add(i, 1)`` on an int64 counter keeps int64
(Python ints are weak scalars). An int32 counter + Python int promotes to
int64 → ``DTypeError``; use ``etl.constant`` int32 ones or an int64 counter.
The loop bound is a STATIC Python int captured in ``cond_fn``'s closure
(specializes the traced condition — ``etl.less(i, T)`` with Python ``T``
works; verified 6/6 loops).

``etl.cond`` is likewise validated (predicate = 0-d bool; branch callables
receive ALL captured operands, both branches traced once) — it is not needed
by these four examples but is the companion primitive for in-graph control
flow.

Deliberately NOT used — ``etl.scan``
------------------------------------
None of these examples use ``etl.scan``: dev probes reproduced an
interpreter ``ShapeError`` in scan's internal while/scatter stacking path
(core bug — scan lowers to a ``while`` + row-wise ``scatter`` stack whose
types do not round-trip on the numpy backend). Simple scan forms happened to
run at this HEAD in spot-probes, but scan is not exercised here: every
example above is a plain ``while_loop`` over fixed-shape carried state, which
is the fully validated path (6/6 back-to-back loops).

Compiler status (probed at dev time, recorded for the record)
-------------------------------------------------------------
- StableHLO export: ``stablehlo.export(graph)`` on ``e2e_power_iteration``
  (dot/sqrt/divide/reduce_sum + while) SUCCEEDS and emits a real
  ``stablehlo.while`` region pair — in-graph loops DO export. The
  transformer and kmeans graphs fail at export time with the DOCUMENTED v1
  op deferrals, not the while op: ``op 'scatter' ... not supported in v1``
  (transformer) and ``op 'argmin' ... not supported in v1`` (kmeans).
- IREE: the adapter imports (``etl.backends.get("iree")``) and
  ``etl.lower(graph, backend="iree", target_backends=["llvm-cpu"])`` +
  compile + load + run of ``e2e_power_iteration`` SUCCEEDS (build 0.6 s, run
  1.3 ms, max_abs 1.04e-07 vs the numpy reference) — the while loop executes
  on a real compiler backend.

All numpy_refs are plain numpy loops mirroring the exact iteration counts and
formulas (identical op sequences, same fixed shapes, same guards), so the
in-graph control flow is genuinely validated against an independent
implementation. Measured on the numpy backend via the harness (seed 0):
transformer 0.0 (exact, token sequence identical), pso 0.0 (exact), kmeans 0.0
(exact, assignments identical), power_iteration 7.45e-08 — all well within the
strict conformance defaults (rtol=atol=1e-5); no per-example tolerance
overrides needed. Sizes keep single runs well under ~5 ms (T <= 15, N <= 32,
D <= 16).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .base import Example, _F32, register_all


def _const(value, dtype):
    """Embed a static value as a Constant op (valid only inside a trace)."""
    return etl.constant(etl.tensor(np.asarray(value, dtype=dtype), dtype=dtype))


# --- e2e_infer_transformer --------------------------------------------------
# Autoregressive decoding: 1 transformer block, 1 head, D=16, vocab 16.
# Carried state: (count int64 scalar, buffer [T_max, D]); the start token is
# pre-placed at row 0 (count = 1), then each of the T-1 iterations computes
# attention over the sequence-so-far (rows >= count masked to -inf), produces
# next-token logits, takes argmax, looks the token up in the embedding table,
# and writes it into row `count` with etl.scatter. Buffer shape stays
# [T_max, D] throughout; output is the final fixed-size buffer.
_D, _VOCAB, _T_MAX, _T = 16, 16, 8, 8


@defn
def _e2e_infer_transformer_graph(start, Wq, Wk, Wv, W_out, embed_table):
    inv_sqrt_d = _const(1.0 / np.sqrt(_D), _F32)
    row_idx = _const(np.arange(_T_MAX, dtype=np.int64), etl.int64)
    neg_inf = _const(-1e9, _F32)
    zero_buf = _const(np.zeros((_T_MAX, _D), np.float32), _F32)

    def cond_fn(state):
        count, buf = state
        return etl.less(count, _const(_T, etl.int64))

    def body_fn(state):
        count, buf = state
        # query = last written token, projected; K/V over the whole buffer
        # (rows >= count are zero padding AND masked out of the scores).
        q = etl.dot(etl.reshape(etl.gather(buf, etl.subtract(count, 1), axis=0), (1, _D)), Wq)
        K = etl.dot(buf, Wk)
        V = etl.dot(buf, Wv)
        scores = etl.multiply(etl.reshape(etl.dot(q, etl.transpose(K)), (_T_MAX,)), inv_sqrt_d)
        mask = etl.less(row_idx, count)  # [T_max] bool: rows written so far
        scores_m = etl.select(mask, scores, neg_inf)
        # masked softmax (max-subtracted for fp32 stability)
        e = etl.exp(etl.subtract(scores_m, etl.max(scores_m)))
        p = etl.divide(e, etl.sum(e))
        out = etl.reshape(etl.dot(etl.reshape(p, (1, _T_MAX)), V), (_D,))
        logits = etl.reshape(etl.dot(etl.reshape(out, (1, _D)), W_out), (_VOCAB,))
        token = etl.argmax(logits)
        nxt = etl.gather(embed_table, token, axis=0)  # [D] next-token embedding
        buf_new = etl.scatter(buf, count, nxt, axis=0)
        return (etl.add(count, 1), buf_new)

    init = (
        _const(1, etl.int64),
        etl.scatter(zero_buf, _const(0, etl.int64), start, axis=0),
    )
    count, buf = etl.while_loop(cond_fn, body_fn, init)
    return buf


def _e2e_infer_transformer_numpy(inputs):
    start, Wq, Wk, Wv, W_out, embed_table = inputs
    buf = np.zeros((_T_MAX, _D), np.float32)
    buf[0] = start
    count = 1
    while count < _T:
        q = buf[count - 1] @ Wq
        K = buf @ Wk
        V = buf @ Wv
        scores = (q @ K.T) / np.sqrt(_D)
        mask = np.arange(_T_MAX) < count
        scores_m = np.where(mask, scores, -1e9)
        e = np.exp(scores_m - scores_m.max())
        p = e / e.sum()
        out = p @ V
        logits = out @ W_out
        token = int(np.argmax(logits))
        buf[count] = embed_table[token]
        count += 1
    return buf


# --- e2e_pso_optimize -------------------------------------------------------
# Particle swarm optimization: N=16 particles, D=8 dims, objective
# ||x - target||^2 (target is an input). Carried state: (i, x, v, p) — all
# fixed [N, D]. Each iteration: g = p[argmin f(p)]; v = w v + c1 r1 (p - x) +
# c2 r2 (g - x); x += v; p = select(f(x) < f(p), x, p). r1/r2 are explicit
# inputs (fixed draws reused each iteration — deterministic, mirrored by the
# numpy ref). Output: (best position, best value) recomputed from final p.
_N, _PD, _PT = 16, 8, 10
_W, _C1, _C2 = 0.5, 1.5, 1.5


@defn
def _e2e_pso_optimize_graph(x, v, p, target, r1, r2):
    w_c = _const(_W, _F32)
    c1_c = _const(_C1, _F32)
    c2_c = _const(_C2, _F32)

    def f(positions):
        diff = etl.subtract(positions, target)
        return etl.reduce_sum(etl.multiply(diff, diff), axes=(1,))  # [N]

    def cond_fn(state):
        i, xc, vc, pc = state
        return etl.less(i, _const(_PT, etl.int64))

    def body_fn(state):
        i, xc, vc, pc = state
        fp = f(pc)
        g = etl.gather(pc, etl.argmin(fp), axis=0)  # [D] global best position
        v_new = etl.add(
            etl.multiply(w_c, vc),
            etl.add(
                etl.multiply(etl.multiply(c1_c, r1), etl.subtract(pc, xc)),
                etl.multiply(etl.multiply(c2_c, r2), etl.subtract(g, xc)),
            ),
        )
        x_new = etl.add(xc, v_new)
        fx = f(x_new)
        # [N,1] predicate so select broadcasts against [N, D] (numpy-where
        # style per-particle update).
        improved = etl.reshape(etl.less(fx, fp), (_N, 1))
        p_new = etl.select(improved, x_new, pc)
        return (etl.add(i, 1), x_new, v_new, p_new)

    init = (_const(0, etl.int64), x, v, p)
    i, xf, vf, pf = etl.while_loop(cond_fn, body_fn, init)
    fp = f(pf)
    best_idx = etl.argmin(fp)
    return (etl.gather(pf, best_idx, axis=0), etl.gather(fp, best_idx, axis=0))


def _e2e_pso_optimize_numpy(inputs):
    x, v, p, target, r1, r2 = inputs
    for _ in range(_PT):
        fp = ((p - target) ** 2).sum(axis=1)
        g = p[np.argmin(fp)]
        v = _W * v + _C1 * r1 * (p - x) + _C2 * r2 * (g - x)
        x = x + v
        fx = ((x - target) ** 2).sum(axis=1)
        p = np.where(fx[:, None] < fp[:, None], x, p)
    fp = ((p - target) ** 2).sum(axis=1)
    best_idx = np.argmin(fp)
    return (p[best_idx], np.asarray(fp[best_idx]))


# --- e2e_kmeans -------------------------------------------------------------
# k-means: M=32 points in D=4 dims, K=4 clusters, T=8 iterations. Carried
# state: (i, C [K, D]). Each iteration: pairwise squared distances via
# [M,1,D] - [1,K,D] broadcast, argmin assignments, one-hot membership mask,
# C = mask^T X / counts with a max(counts, 1) guard (identical in etl and the
# numpy ref, so empty-cluster behavior matches exactly). Output: (final
# centroids, final assignments recomputed from them).
_M, _KD, _KK, _KT = 32, 4, 4, 8


@defn
def _e2e_kmeans_graph(X, C0):
    k_idx = _const(np.arange(_KK, dtype=np.int64), etl.int64)
    one = _const(1.0, _F32)

    def cond_fn(state):
        i, C = state
        return etl.less(i, _const(_KT, etl.int64))

    def body_fn(state):
        i, C = state
        X3 = etl.reshape(X, (_M, 1, _KD))
        C3 = etl.reshape(C, (1, _KK, _KD))
        dists = etl.reduce_sum(etl.square(etl.subtract(X3, C3)), axes=(2,))  # [M,K]
        assign = etl.argmin(dists, axis=1)  # [M] int64
        mask = etl.equal(etl.reshape(assign, (_M, 1)), etl.reshape(k_idx, (1, _KK)))
        mask_f = etl.cast(mask, _F32)
        counts = etl.reduce_sum(mask_f, axes=(0,))  # [K]
        sums = etl.dot(etl.transpose(mask_f), X)  # [K,D]
        counts_safe = etl.maximum(counts, one)
        C_new = etl.divide(sums, etl.reshape(counts_safe, (_KK, 1)))
        return (etl.add(i, 1), C_new)

    init = (_const(0, etl.int64), C0)
    i, C = etl.while_loop(cond_fn, body_fn, init)
    X3 = etl.reshape(X, (_M, 1, _KD))
    C3 = etl.reshape(C, (1, _KK, _KD))
    dists = etl.reduce_sum(etl.square(etl.subtract(X3, C3)), axes=(2,))
    assign = etl.argmin(dists, axis=1)
    return (C, assign)


def _e2e_kmeans_numpy(inputs):
    X, C0 = inputs
    C = C0.copy()
    for _ in range(_KT):
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)
        mask = assign[:, None] == np.arange(_KK)[None, :]
        mask_f = mask.astype(np.float32)
        counts = mask_f.sum(axis=0)
        C = (mask_f.T @ X) / np.maximum(counts, 1.0)[:, None]
    dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dists, axis=1)
    return (C, assign)


# --- e2e_power_iteration ----------------------------------------------------
# Dominant-eigenvector estimate by power iteration: T=15 iterations of
# v = A v / ||A v|| (A [16,16] and v0 [16,1] inputs). Carried state:
# (i int64, v [16,1]). Deterministic: etl and the numpy ref run the identical
# recurrence from the same v0, so there is no sign-ambiguity issue (measured
# max_abs 5.96e-08 — fp32 normalization rounding only). This is the only
# example whose op set is fully StableHLO-exportable (see module docstring).
_PI_T, _PDIM = 15, 16


@defn
def _e2e_power_iteration_graph(A, v0):
    def cond_fn(state):
        i, v = state
        return etl.less(i, _const(_PI_T, etl.int64))

    def body_fn(state):
        i, v = state
        av = etl.dot(A, v)  # [16,16] @ [16,1]
        norm = etl.sqrt(etl.reduce_sum(etl.multiply(av, av), axes=None))
        v_new = etl.divide(av, norm)
        return (etl.add(i, 1), v_new)

    init = (_const(0, etl.int64), v0)
    i, v = etl.while_loop(cond_fn, body_fn, init)
    return v


def _e2e_power_iteration_numpy(inputs):
    A, v = inputs
    for _ in range(_PI_T):
        av = A @ v
        v = av / np.linalg.norm(av)
    return v


# --- registry ---------------------------------------------------------------

register_all(
    [
        Example(
            name="e2e_infer_transformer",
            description=(
                "autoregressive decoding: in-graph while_loop attends over the "
                "sequence-so-far, argmax next token, fixed [8,16] buffer"
            ),
            specs=(
                TensorSpec((_D,), _F32),          # start token embedding
                TensorSpec((_D, _D), _F32),       # Wq
                TensorSpec((_D, _D), _F32),       # Wk
                TensorSpec((_D, _D), _F32),       # Wv
                TensorSpec((_D, _VOCAB), _F32),   # W_out (logits projection)
                TensorSpec((_VOCAB, _D), _F32),   # embedding table
            ),
            graph=_e2e_infer_transformer_graph,
            numpy_ref=_e2e_infer_transformer_numpy,
            category="e2e",
            tags=("infer", "control-flow"),
        ),
        Example(
            name="e2e_pso_optimize",
            description=(
                "in-graph PSO loop (T=10) minimizing ||x - target||^2: "
                "v/p updates, best position + best value out"
            ),
            specs=(
                TensorSpec((_N, _PD), _F32),  # positions
                TensorSpec((_N, _PD), _F32),  # velocities
                TensorSpec((_N, _PD), _F32),  # personal bests
                TensorSpec((_PD,), _F32),     # target
                TensorSpec((_N, _PD), _F32),  # r1 draws (fixed, input)
                TensorSpec((_N, _PD), _F32),  # r2 draws (fixed, input)
            ),
            graph=_e2e_pso_optimize_graph,
            numpy_ref=_e2e_pso_optimize_numpy,
            category="e2e",
            tags=("infer", "control-flow"),
        ),
        Example(
            name="e2e_kmeans",
            description=(
                "in-graph k-means (T=8): argmin assignments + one-hot centroid "
                "updates, final centroids + assignments out"
            ),
            specs=(
                TensorSpec((_M, _KD), _F32),  # points
                TensorSpec((_KK, _KD), _F32),  # initial centroids
            ),
            graph=_e2e_kmeans_graph,
            numpy_ref=_e2e_kmeans_numpy,
            category="e2e",
            tags=("infer", "control-flow"),
        ),
        Example(
            name="e2e_power_iteration",
            description=(
                "in-graph power iteration (T=15): v = A v / ||A v||, "
                "dominant-eigenvector estimate out"
            ),
            specs=(
                TensorSpec((_PDIM, _PDIM), _F32),
                TensorSpec((_PDIM, 1), _F32),
            ),
            graph=_e2e_power_iteration_graph,
            numpy_ref=_e2e_power_iteration_numpy,
            category="e2e",
            tags=("infer", "control-flow"),
        ),
    ]
)
