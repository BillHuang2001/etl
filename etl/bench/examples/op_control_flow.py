"""Control-flow op conformance examples (category "op", tag "control-flow").

Runtime tensor control flow traced into region-based IR: ``etl.cond``
(``if`` op) and ``etl.while_loop`` (``while`` op), plus their composition.
Each example is an ``@etl.defn`` graph staged through the explicit pipeline
(``etl.build`` + ``etl.run``) and compared against a pure-numpy reference
(``np.where``-style for cond, plain Python loops for while).

Coverage (8 examples, all PASS on the numpy backend with the strict default
tolerances — measured max_abs 0.0):

- ``cond_basic`` — ``etl.cond`` with a 0-d bool predicate, both branches on a
  shared operand (add 1 vs subtract 1).
- ``cond_multi_output`` — ``etl.cond`` returning a tuple ``(x±1, y*2|y/2)``
  (branch trees must match; result is unflattened per the branch output tree).
- ``cond_nested`` — ``etl.cond`` inside a branch (a nested if; the inner
  predicate is captured from the enclosing trace scope).
- ``while_cumsum`` — static-trip ``etl.while_loop`` accumulating ``x`` N=5
  times; carries ``(int64 counter, f32 acc)``; the final counter is returned
  alongside the accumulator (validates 0-d int64 outputs).
- ``while_fib`` — Fibonacci recurrence over 2 carried f32 scalars + an int64
  counter (3 carried leaves, zero graph inputs).
- ``while_cond_combo`` — ``etl.while_loop`` whose body contains an
  ``etl.cond`` (phase-split accumulation: add x for the first 3 steps, 2x for
  the last 2).
- ``while_poly`` — Horner evaluation of a degree-3 polynomial over a static
  trip count, indexing the coefficient vector with a dynamic 0-d gather
  (``etl.gather(c, i, axis=0)``).
- ``cond_while_pipeline`` — ``etl.cond`` selecting between TWO ``while_loop``
  results (an ``if`` op whose branch regions each contain a ``while`` op;
  verified to trace and run).

Loop-carried dtype discipline (binding, per the v1 typed-while contract):
carried dtypes/shapes must stay CONSTANT across iterations. The counters are
int64 (``etl.add(i, 1)`` on int64 stays int64; ``int32 + Python int`` promotes
to int64 and would raise ``DTypeError``). ``etl.cond`` predicates are 0-d
bool graph inputs (specs ``TensorSpec((), etl.bool_)`` — ``generate_inputs``
draws uniform bools). ``N`` (the trip count) is a Python int in the closure —
a static value that specializes the condition region, per the static-value
contract.

Known issues / documented deferrals (measured at HEAD in this worktree):

- **``etl.scan`` — no scan example is registered.** The dev probe found the
  numpy backend's scan path broken: a scan of a ``[5, 3]`` tensor with a
  carry raised ``ShapeError: scatter: shape mismatch: value array of shape
  (1,3) could not be broadcast to indexing result of shape (1,)`` from
  ``etl/backends/numpy/kernels/control_flow.py`` ``_while`` →
  ``indexing._scatter`` (the 0-d loop counter was reshaped to ``(1,)``
  instead of the full-rank all-ones form). AT HEAD THIS IS FIXED: commit
  ``c6fd423`` ("numpy backend: fix 0-d scalar normalization in kernels +
  env-stack resolution for nested-region outer captures") added the 0-d index
  normalization in the scatter kernel, and the same repro (plus tuple
  carries, explicit/prefix lengths, 1-D xs, scalar/vector/rank-2 carries)
  now runs correctly; ``tests/trace/test_scan.py`` is green. Scan still stays
  OUT of this suite (no compiler-backend coverage in v1 and the
  gather/scatter IR it desugars to is a documented StableHLO export deferral)
  — the suite covers the stable ``if``/``while`` surface only.
- **Gradients through control flow are a v1 deferral** — probed, not
  registered: ``etl.grad`` of a scalar-valued graph containing ``etl.cond``
  raises ``TransformError: grad/vjp: no VJP rule for op 'if'`` and through
  ``etl.while_loop`` ``TransformError: ... no VJP rule for op 'while'``
  (``jvp`` likewise: "no JVP rule for op 'while'"). Explicit errors, never
  silent fallback.

Compiler-backend status (measured at HEAD in this worktree):

- **StableHLO exporter** (no compiler needed): ``stablehlo.export`` handles
  both region ops via ``_emit_if_region``/``_emit_while`` — ``cond_basic``
  exports ``stablehlo.if``, ``while_cumsum`` exports ``stablehlo.while``, and
  ``cond_while_pipeline`` exports the nested if→while composition. The only
  deferral in this set is ``while_poly``: its dynamic ``gather`` is not
  exported in v1 (``BackendError: stablehlo export: op 'gather' ... is not
  supported in v1`` — the same documented class as ``grad_structural``).
- **iree (llvm-cpu)**: ``etl.lower(graph, backend='iree')`` succeeds for
  ``cond_basic`` and ``while_cumsum``, and the FULL pipeline
  (``lower → compile → load → run``) executes both correctly (cond selects
  the branch; while loops 5 iterations at run time). The nanobind "leaked
  instances" teardown noise at interpreter exit is the known upstream iree
  runtime message (harmless; see the bench package Known Issues).

All examples use the strict default conformance tolerances (no per-example
``rtol``/``atol``/``tolerance`` overrides): the graphs are exact on the numpy
backend (max_abs 0.0 — the while examples accumulate a fixed small number of
fp32 additions; the reference loops perform the identical operations).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .base import Example, _F32, register_all

# --- cond: basic ---------------------------------------------------------------


@defn
def _cond_basic_graph(pred, x):
    """0-d bool predicate selecting add-1 vs subtract-1 on a shared operand."""
    return etl.cond(
        pred,
        lambda a: etl.add(a, 1.0),
        lambda a: etl.subtract(a, 1.0),
        x,
    )


def _cond_basic_numpy(inputs):
    pred, x = inputs
    return np.where(pred, x + 1.0, x - 1.0)


# --- cond: multi-output --------------------------------------------------------


@defn
def _cond_multi_output_graph(pred, x, y):
    """Tuple-returning cond: (x+1, y*2) vs (x-1, y/2) on two shared operands."""
    return etl.cond(
        pred,
        lambda a, b: (etl.add(a, 1.0), etl.multiply(b, 2.0)),
        lambda a, b: (etl.subtract(a, 1.0), etl.divide(b, 2.0)),
        x,
        y,
    )


def _cond_multi_output_numpy(inputs):
    pred, x, y = inputs
    return (np.where(pred, x + 1.0, x - 1.0), np.where(pred, y * 2.0, y / 2.0))


# --- cond: nested --------------------------------------------------------------


@defn
def _cond_nested_graph(p1, p2, x):
    """Nested if: p1 ? (p2 ? x+2 : x+1) : x-1 — an inner cond inside a branch."""

    def inner(a):
        return etl.cond(
            p2,
            lambda b: etl.add(b, 2.0),
            lambda b: etl.add(b, 1.0),
            a,
        )

    return etl.cond(
        p1,
        inner,
        lambda a: etl.subtract(a, 1.0),
        x,
    )


def _cond_nested_numpy(inputs):
    p1, p2, x = inputs
    return np.where(p1, np.where(p2, x + 2.0, x + 1.0), x - 1.0)


# --- while: static-trip cumsum -------------------------------------------------

_WHILE_N = 5  # static trip count (Python int in the closure — specializes)


@defn
def _while_cumsum_graph(x):
    """Static-trip while: accumulate x into an f32 acc, N=5 times."""

    def cond_fn(state):
        i, acc = state
        return etl.less(i, _WHILE_N)

    def body_fn(state):
        i, acc = state
        return (etl.add(i, 1), etl.add(acc, x))

    init = (
        etl.constant(etl.tensor(np.int64(0), dtype=etl.int64)),
        etl.constant(etl.tensor(np.zeros((5,), dtype=np.float32), dtype=_F32)),
    )
    return etl.while_loop(cond_fn, body_fn, init)


def _while_cumsum_numpy(inputs):
    x = inputs[0]
    return (np.array(_WHILE_N, dtype=np.int64), _WHILE_N * x)


# --- while: Fibonacci recurrence ----------------------------------------------


@defn
def _while_fib_graph():
    """3-leaf carry: (int64 counter, f32 a, f32 b) → (i+1, b, a+b) N=5 times.

    No graph inputs — the whole loop is driven by constants; F_5 = 5 and
    F_6 = 8 after 5 iterations.
    """

    def cond_fn(state):
        i, a, b = state
        return etl.less(i, _WHILE_N)

    def body_fn(state):
        i, a, b = state
        return (etl.add(i, 1), b, etl.add(a, b))

    init = (
        etl.constant(etl.tensor(np.int64(0), dtype=etl.int64)),
        etl.constant(etl.tensor(np.float32(0.0), dtype=_F32)),
        etl.constant(etl.tensor(np.float32(1.0), dtype=_F32)),
    )
    return etl.while_loop(cond_fn, body_fn, init)


def _while_fib_numpy(inputs):
    a, b = 0.0, 1.0
    for _ in range(_WHILE_N):
        a, b = b, a + b
    return (
        np.array(_WHILE_N, dtype=np.int64),
        np.array(a, dtype=np.float32),
        np.array(b, dtype=np.float32),
    )


# --- while with cond inside the body -------------------------------------------

_WHILE_COMBO_N = 5


@defn
def _while_cond_combo_graph(x):
    """While whose body contains a cond: phase-split accumulation.

    Steps 0-2 add ``x``, steps 3-4 add ``2x`` (phase decided by a cond on the
    carried counter) → acc = 3x + 4x = 7x after 5 iterations.
    """

    def cond_fn(state):
        i, acc = state
        return etl.less(i, _WHILE_COMBO_N)

    def body_fn(state):
        i, acc = state
        early = etl.less(i, 3)
        next_acc = etl.cond(
            early,
            lambda a: etl.add(a, x),
            lambda a: etl.add(a, etl.multiply(x, 2.0)),
            acc,
        )
        return (etl.add(i, 1), next_acc)

    init = (
        etl.constant(etl.tensor(np.int64(0), dtype=etl.int64)),
        etl.constant(etl.tensor(np.zeros((4,), dtype=np.float32), dtype=_F32)),
    )
    return etl.while_loop(cond_fn, body_fn, init)


def _while_cond_combo_numpy(inputs):
    x = inputs[0]
    acc = np.zeros_like(x)
    for i in range(_WHILE_COMBO_N):
        acc = acc + (x if i < 3 else 2.0 * x)
    return (np.array(_WHILE_COMBO_N, dtype=np.int64), acc)


# --- while: Horner polynomial via dynamic gather -------------------------------

_WHILE_POLY_N = 4  # degree-3 polynomial: c0 + c1 x + c2 x^2 + c3 x^3


@defn
def _while_poly_graph(x, c):
    """Horner's rule over a static trip count with a dynamic coefficient
    gather: each iteration fetches ``c[i]`` via ``etl.gather(c, i, axis=0)``
    (0-d dynamic index — numpy backend; see the docstring for the StableHLO
    export deferral)."""

    def cond_fn(state):
        i, acc = state
        return etl.less(i, _WHILE_POLY_N)

    def body_fn(state):
        i, acc = state
        coef = etl.gather(c, i, axis=0)
        return (etl.add(i, 1), etl.add(etl.multiply(acc, x), coef))

    init = (
        etl.constant(etl.tensor(np.int64(0), dtype=etl.int64)),
        etl.constant(etl.tensor(np.float32(0.0), dtype=_F32)),
    )
    return etl.while_loop(cond_fn, body_fn, init)


def _while_poly_numpy(inputs):
    x, c = inputs
    acc = 0.0
    for k in range(_WHILE_POLY_N):
        acc = acc * x + c[k]
    # np.asarray: NEP-50 scalar promotion collapses ``0.0 * 0-d array`` to a
    # numpy scalar — the reference must yield ndarrays (flatten_outputs
    # rejects numpy scalars).
    return (
        np.array(_WHILE_POLY_N, dtype=np.int64),
        np.asarray(acc, dtype=np.float32),
    )


# --- cond selecting between two while loops ------------------------------------

_WHILE_PIPE_N = 4


@defn
def _cond_while_pipeline_graph(pred, x):
    """Cond whose branches each run a full while loop: ``pred ? N*x : N*2x``.

    Each branch traces its own ``while`` op inside the branch region (an
    ``if`` op whose regions contain ``while`` ops — verified to trace and to
    run on the numpy backend and to export to StableHLO).
    """

    def run_loop(factor):
        def cond_fn(state):
            i, acc = state
            return etl.less(i, _WHILE_PIPE_N)

        def body_fn(state):
            i, acc = state
            return (etl.add(i, 1), etl.add(acc, etl.multiply(x, factor)))

        init = (
            etl.constant(etl.tensor(np.int64(0), dtype=etl.int64)),
            etl.constant(etl.tensor(np.zeros((3,), dtype=np.float32), dtype=_F32)),
        )
        return etl.while_loop(cond_fn, body_fn, init)

    return etl.cond(
        pred,
        lambda a: run_loop(1.0),
        lambda a: run_loop(2.0),
        x,
    )


def _cond_while_pipeline_numpy(inputs):
    pred, x = inputs
    factor = 1.0 if pred else 2.0
    return (np.array(_WHILE_PIPE_N, dtype=np.int64), _WHILE_PIPE_N * factor * x)


# --- registry ------------------------------------------------------------------

_EXAMPLES = [
    Example(
        name="cond_basic",
        description="etl.cond with a 0-d bool pred: x+1 vs x-1 on a shared operand",
        specs=(TensorSpec((), etl.bool_), TensorSpec((4,), _F32)),
        graph=_cond_basic_graph,
        numpy_ref=_cond_basic_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="cond_multi_output",
        description="etl.cond returning a tuple: (x+1, y*2) vs (x-1, y/2)",
        specs=(
            TensorSpec((), etl.bool_),
            TensorSpec((4,), _F32),
            TensorSpec((3, 2), _F32),
        ),
        graph=_cond_multi_output_graph,
        numpy_ref=_cond_multi_output_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="cond_nested",
        description="nested etl.cond: p1 ? (p2 ? x+2 : x+1) : x-1",
        specs=(
            TensorSpec((), etl.bool_),
            TensorSpec((), etl.bool_),
            TensorSpec((4,), _F32),
        ),
        graph=_cond_nested_graph,
        numpy_ref=_cond_nested_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="while_cumsum",
        description="static-trip etl.while_loop accumulating x N=5 times "
        "(int64 counter + f32 acc carried)",
        specs=(TensorSpec((5,), _F32),),
        graph=_while_cumsum_graph,
        numpy_ref=_while_cumsum_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="while_fib",
        description="3-leaf Fibonacci etl.while_loop: (i, a, b) -> (i+1, b, a+b), "
        "N=5, no graph inputs",
        specs=(),
        graph=_while_fib_graph,
        numpy_ref=_while_fib_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="while_cond_combo",
        description="etl.while_loop whose body contains an etl.cond "
        "(phase-split: add x, then 2x — acc = 7x after 5 steps)",
        specs=(TensorSpec((4,), _F32),),
        graph=_while_cond_combo_graph,
        numpy_ref=_while_cond_combo_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="while_poly",
        description="Horner polynomial via etl.while_loop + dynamic 0-d gather "
        "of the coefficients",
        specs=(TensorSpec((), _F32), TensorSpec((_WHILE_POLY_N,), _F32)),
        graph=_while_poly_graph,
        numpy_ref=_while_poly_numpy,
        category="op",
        tags=("control-flow",),
    ),
    Example(
        name="cond_while_pipeline",
        description="etl.cond selecting between two etl.while_loop results "
        "(pred ? N*x : N*2x)",
        specs=(TensorSpec((), etl.bool_), TensorSpec((3,), _F32)),
        graph=_cond_while_pipeline_graph,
        numpy_ref=_cond_while_pipeline_numpy,
        category="op",
        tags=("control-flow",),
    ),
]

register_all(_EXAMPLES)
