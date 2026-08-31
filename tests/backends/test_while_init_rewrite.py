"""iree while-init constant-rewrite parity + SEGV regression suite (fix 71d721e/ecb2e82).

iree 3.11.0 SEGVs (SIGSEGV, ``CompilerToolError "Error code: -11"``) inside
``IREE::Stream::AffinityAnalysis::run()`` on dense rank>=1 ``stablehlo.constant``
values used as while-loop INIT operands — the upstream AffinityAnalysis bug
behind the real evox NSGA2 ``non_dominate_rank`` loop crashing iree-compile
~9/10 on llvm-cpu and ~3/5 on cuda (root-cause + trigger analysis in
``etl/backends/stablehlo/CONTEXT.md`` Known Issues).

The exporter workaround (default ON, opt out with
``lower(..., while_init_rewrite=False)``): every all-zero rank>=1 constant
while-init is replaced with COMPUTED zeros — ``not(x)`` / ``and(x, not(x))``
over a NON-constant pre-loop value + a dtype-changing ``convert`` (+ a
``reduce`` when the source is one axis larger) — the validated-safe "P15"
pattern (never ``x - x`` / broadcast-of-scalar-0: iree's canonicalizer folds
both back into constants). ``_find_while_zero_source`` prefers the REDUCE
path (an input-derived bool/int value ONE axis larger, reduced along the
extra axis — the ONLY class validated safe at real-graph scale).

Pinned here:

* the computed-zeros chain appears in the exported MLIR and the while INIT
  position no longer holds the all-zero constant (MLIR-position assertions),
* the small graph compiles 10/10 on iree-llvm-cpu with zero SEGVs and runs
  bit-exact vs the numpy backend,
* the REAL-graph shape — the evox ``non_dominate_rank`` loop: input-derived
  (n, n) i1 dominance matrix + intra-body reduction + bool→int32 cast chain,
  n=32 — compiles and runs bit-exact on llvm-cpu,
* ``while_init_rewrite=False`` opts out: the while init IS the all-zero
  constant again (MLIR-text assertion only — that shape is the upstream SEGV
  and is deliberately NEVER compiled here).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# graph builders
# ---------------------------------------------------------------------------


def _tiny_ndr_while(dominates):
    """Small-graph proxy of the evox ``non_dominate_rank`` loop: the (n, n) i1
    dominance matrix IS the input, the all-zero (n,) i32 rank constant is the
    while init the rewrite must replace (reduce path: the (4, 4) i1 input is
    one axis larger than the (4,) target)."""
    n = 4
    zero_i32 = etl.constant(etl.tensor(np.int32(0)))
    one_i32 = etl.constant(etl.tensor(np.int32(1)))
    count = etl.sum(etl.enp.astype(dominates, etl.int32), axes=0)
    rank = etl.constant(etl.tensor(np.zeros(n, dtype=np.int32)))

    def cond_fn(carry):
        c, r, cur = carry
        return etl.greater_equal(etl.max(c), zero_i32)

    def body_fn(carry):
        c, r, cur = carry
        front = etl.equal(c, zero_i32)
        r = etl.enp.where(front, cur, r)
        front_dom = etl.sum(
            etl.enp.astype(
                etl.logical_and(dominates, etl.enp.expand_dims(front, 1)),
                etl.int32,
            ),
            axes=0,
        )
        c = etl.subtract(c, front_dom)
        c = etl.subtract(c, etl.enp.astype(front, etl.int32))
        return (c, r, etl.add(cur, one_i32))

    _, rank, _ = etl.while_loop(cond_fn, body_fn, (count, rank, zero_i32))
    return rank


def _non_dominate_rank(x):
    """Verbatim evox ``non_dominate_rank`` (nsga2.py / non_dominate.py) —
    the REAL-graph shape: input-derived (n, n) i1 dominance matrix from
    fitness comparisons, then a while_loop carrying (count, rank,
    current_rank) with intra-body reductions and bool→int32 cast chains."""
    n = x.shape[0]
    x_i = etl.enp.expand_dims(x, 1)
    x_j = etl.enp.expand_dims(x, 0)
    le = etl.less_equal(x_i, x_j)
    lt = etl.less(x_i, x_j)
    le_all = etl.min(le, axes=2)
    lt_any = etl.max(lt, axes=2)
    dominates = etl.logical_and(le_all, lt_any)
    zero_i32 = etl.constant(etl.tensor(np.int32(0)))
    one_i32 = etl.constant(etl.tensor(np.int32(1)))
    count = etl.sum(etl.enp.astype(dominates, etl.int32), axes=0)
    rank = etl.constant(etl.tensor(np.zeros(n, dtype=np.int32)))

    def cond_fn(carry):
        c, r, cur = carry
        return etl.greater_equal(etl.max(c), zero_i32)

    def body_fn(carry):
        c, r, cur = carry
        front = etl.equal(c, zero_i32)
        r = etl.enp.where(front, cur, r)
        front_dom = etl.sum(
            etl.enp.astype(
                etl.logical_and(dominates, etl.enp.expand_dims(front, 1)),
                etl.int32,
            ),
            axes=0,
        )
        c = etl.subtract(c, front_dom)
        c = etl.subtract(c, etl.enp.astype(front, etl.int32))
        return (c, r, etl.add(cur, one_i32))

    _, rank, _ = etl.while_loop(cond_fn, body_fn, (count, rank, zero_i32))
    return rank


def _np_non_dominate_rank(x):
    """Independent numpy front-peeling reference (mirrors the kernels' math,
    never imports etl kernels): ranks = iterative front extraction."""
    le_all = (x[:, None, :] <= x[None, :, :]).all(axis=2)
    lt_any = (x[:, None, :] < x[None, :, :]).any(axis=2)
    a = le_all & lt_any
    count = a.sum(axis=0).astype(np.int32)
    rank = np.zeros(x.shape[0], dtype=np.int32)
    r = 0
    while count.max() >= 0:
        front = count == 0
        rank[front] = r
        count = (
            count - a[front].sum(axis=0).astype(np.int32) - front.astype(np.int32)
        )
        r += 1
    return rank


# ---------------------------------------------------------------------------
# data (fixed seed — same convention as test_iree_emitters_parity.py)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)

_XFIT = np.abs(_RNG.standard_normal((4, 2)).astype(np.float32))
_DOM = (
    (_XFIT[:, None, :] <= _XFIT[None, :, :]).all(axis=2)
    & (_XFIT[:, None, :] < _XFIT[None, :, :]).any(axis=2)
)
_TINY_SPEC = etl.TensorSpec((4, 4), np.bool_)

_N = 32
_X32 = np.abs(_RNG.standard_normal((_N, 2)).astype(np.float32))
_REAL_SPEC = etl.TensorSpec((_N, 2), etl.float32)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _export(graph, **options):
    """Export a traced graph's MLIR with the given exporter options."""
    from etl.backends.stablehlo import export

    return export(graph, options=options)


def _while_operands(mlir):
    """SSA operand list of the (single) ``stablehlo.while`` in ``mlir``."""
    for line in mlir.splitlines():
        if '"stablehlo.while"' in line:
            start = line.index("(") + 1
            end = line.index(")", start)
            return [s.strip() for s in line[start:end].split(",")]
    raise AssertionError("no stablehlo.while in the MLIR")


def _init_def_line(mlir, init_index=1):
    """The defining MLIR line of the while's ``init_index``-th init operand.

    ``init_index=1`` is the rank init in both graphs below (operand 0 is the
    input-derived count, operand 2 the scalar zero)."""
    name = _while_operands(mlir)[init_index]
    for line in mlir.splitlines():
        if line.strip().startswith(name + " ="):
            return line.strip()
    raise AssertionError(f"no defining line for while operand {name}")


# ---------------------------------------------------------------------------
# 1. small graph: MLIR-position assertions + 10/10 compiles + bit-exact run
# ---------------------------------------------------------------------------


def test_small_graph_rewrite_chain_in_mlir():
    graph = etl.trace(_tiny_ndr_while, _TINY_SPEC)
    mlir = _export(graph)  # while_init_rewrite defaults ON
    assert "stablehlo.not" in mlir, "computed-zeros chain (not) missing"
    assert "stablehlo.and" in mlir, "computed-zeros chain (and) missing"
    assert "stablehlo.convert" in mlir
    assert '"stablehlo.reduce"' in _init_def_line(mlir), (
        "while init must be the reduce of the computed zeros, got:\n"
        + _init_def_line(mlir)
    )
    # the all-zero (4,) i32 constant is NOT consumed by the while — its SSA
    # name must not appear among the while operands.
    zero_names = []
    for line in mlir.splitlines():
        if "stablehlo.constant" in line and "dense<[0, 0, 0, 0]>" in line:
            zero_names.append(line.split("=")[0].strip())
    assert zero_names, "expected the all-zero rank-1 constant in the MLIR"
    for name in zero_names:
        assert name not in _while_operands(mlir), (
            f"all-zero constant {name} still consumed by the while"
        )


def test_small_graph_compile_10x_no_segv():
    """The SEGV regression pin: 10 consecutive iree-llvm-cpu compiles of the
    const-init while loop with the rewrite ON must all succeed (the upstream
    AffinityAnalysis SEGV crashed ~9/10 on the un-rewritten shape)."""
    graph = etl.trace(_tiny_ndr_while, _TINY_SPEC)
    for _ in range(10):
        lowered = etl.lower(graph, backend="iree")
        etl.compile(lowered)  # any SEGV surfaces as a raised error here
    # one run for parity (compile once more through the pipeline sugar).
    want = etl.evaluate(_tiny_ndr_while, _DOM)
    exe = etl.build(_tiny_ndr_while, _TINY_SPEC, backend="iree")
    _assert_exact(etl.run(exe, _DOM), want)


# ---------------------------------------------------------------------------
# 2. real-graph style (n=32): compile + run bit-exact on llvm-cpu
# ---------------------------------------------------------------------------


def test_real_graph_style_llvm_cpu_bit_exact():
    graph = etl.trace(_non_dominate_rank, _REAL_SPEC)
    # the rewrite must fire on the REAL shape too (input-derived (n, n) i1
    # le_all/lt_any sources one axis larger than the (n,) i32 init).
    mlir = _export(graph)
    assert "stablehlo.not" in mlir
    assert '"stablehlo.reduce"' in _init_def_line(mlir)

    want = etl.evaluate(_non_dominate_rank, _X32)  # numpy backend reference
    assert np.array_equal(_np(want), _np_non_dominate_rank(_X32)), (
        "numpy-backend reference must match the independent front-peeling ref"
    )
    exe = etl.build(_non_dominate_rank, _REAL_SPEC, backend="iree")
    _assert_exact(etl.run(exe, _X32), want)


def test_opt_out_while_init_rewrite_false():
    """``while_init_rewrite=False`` must be accepted by ``lower()`` and
    restore the constant init — asserted on MLIR text ONLY (the opt-out shape
    is the upstream SEGV trigger and is deliberately never compiled)."""
    graph = etl.trace(_non_dominate_rank, _REAL_SPEC)
    lowered = etl.lower(graph, backend="iree", while_init_rewrite=False)
    mlir = lowered.payload["mlir_text"]
    assert "stablehlo.not" not in mlir, "rewrite must be disabled"
    assert "stablehlo.constant" in _init_def_line(mlir), (
        "opt-out while init must be the all-zero constant, got:\n"
        + _init_def_line(mlir)
    )
    # and the same graph WITHOUT the option keeps the rewrite (default ON).
    lowered_on = etl.lower(graph, backend="iree")
    mlir_on = lowered_on.payload["mlir_text"]
    assert '"stablehlo.reduce"' in _init_def_line(mlir_on)
