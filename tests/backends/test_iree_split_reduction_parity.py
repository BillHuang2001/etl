"""``split_reduction`` exporter option — iree/llvm-cpu numeric parity.

The StableHLO exporter's ``split_reduction`` option (default ``True``, see
``etl/backends/stablehlo/split_reduce.py``) rewrites a reduction whose RESULT
is rank-0 (or otherwise tiny) and whose reduced extent is large and STATIC
into a two-stage emission: pad to ``R * C``, reshape ``N -> (R, C)``, reduce
the inner axis (``R`` partials — a rank >= 1 intermediate the compiler can
parallelize) and then reduce the tiny partial vector. This file pins that the
rewrite is **numerically transparent** on iree/llvm-cpu: every graph is
compiled through the real adapter and compared against the numpy reference
backend (``etl.evaluate``), which is the ground truth for etl semantics.

Exactness classes (see ``split_reduce.py``) and how they are asserted here:

* **Exact** — ``min`` / ``max`` (incl. the i1 ``any``/``all`` guard shapes),
  ``argmin`` / ``argmax`` (the composite ``(value, index)`` reduce carries the
  original flat index so the first-occurrence tie-break survives both stages)
  and ``sum`` / ``prod`` over INTEGER/bool dtypes (wraparound integer
  arithmetic is associative). These are grouping-invariant, so the assertion
  is ``np.testing.assert_array_equal`` — bit-exact vs the numpy backend.
* **Non-exact** — ``sum`` / ``prod`` over FLOAT dtypes (and therefore
  ``mean``, a ``sum`` composition). Splitting a float accumulation
  reassociates it, so bit-exactness vs numpy is not even a property of the
  *single-stage* emission (the compiler is free to tree-reduce). The
  documented budget is <= 4.5e-5 relative (``NON_EXACT_RTOL``), asserted here
  against the INPUT magnitude — ``|got - expected| <= NON_EXACT_RTOL *
  (|expected| + sum(|x|) over the reduced axes)``, the standard forward-error
  bound for a reassociated accumulation. Measured worst case in this suite:
  2.4e-7 * sum(|x|) for f32 sums and 4.3e-6 relative for float ``prod``.

Coverage: shapes ``(10001,)`` (exact factorization), ``(10000, 100)`` with
``axes=(0,)`` (rank-1 result) and ``axes=(1,)`` (below the extent threshold ->
deliberately NOT split) and with a large result (NOT split), ``(7,)`` (tiny),
``(1024,)`` (exactly at ``split_reduction_threshold``), ``(1025,)`` (just
above) and ``(10007,)`` (prime -> the padded path); dtypes f32/f64/int32/bool;
ops ``min``/``max``/``argmin``/``argmax``/``sum``/``prod``/``mean``; inputs
plain, all-equal (ties), NaN-containing, +-inf, tie-heavy integer and a
``prod``-safe pattern (a random product over thousands of standard normals
underflows to 0, which would make the ``prod`` comparison vacuous).

Budget: real iree compiles run, so the ``<2s per file`` rule does not apply.
Compile count is kept low by computing ALL SEVEN reductions in ONE graph per
(shape, axes, dtype) — the input patterns are parametrized over that single
compiled executable (compile-once cache ``_COMPILED``), and the high-level
shape matrix is covered with f32 while the dtype matrix rides on the three
distinct split classes (exact / rank-1 result / padded).

This file also carries three ``# BUG(etl)`` pin tests (see the bottom of the
module). They are exporter bugs UNRELATED to ``split_reduction`` — each one
reproduces identically with ``split_reduction=False`` and on shapes far below
``split_reduction_threshold`` — and they make some of the dtype/NaN coverage
above silently wrong. Each pin keeps asserting the CORRECT (numpy-backend)
result and is marked ``xfail(strict=True)``: it is expected to fail while the
bug lives, and it FAILS LOUDLY (XPASS under ``strict``) the moment the bug is
fixed, so it can never rot silently into a vacuous pass.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl  # noqa: E402  (must come AFTER the importorskip gates)
from etl.backends import stablehlo  # noqa: E402
from etl.backends.program import LoweredProgram  # noqa: E402
from tests.backends import _adapter_utils as u  # noqa: E402

#: The one-line MLIR comment the split emission puts before its stage-1 reduce
#: (``split_reduce.split_comment``) — its presence is how these tests prove a
#: given reduction was actually split (the marker text names the geometry).
SPLIT_MARKER = "etl split reduction"

#: Documented tolerance for the NON-EXACT (float) class: the error of a
#: reassociated float accumulation, measured RELATIVE TO THE INPUT MAGNITUDE
#: (``sum(|x|)`` over the reduced axes) — see the module docstring.
NON_EXACT_RTOL = 4.5e-5

#: The seven reductions, in the output order of :func:`_graph`.
OPS = ("min", "max", "argmin", "argmax", "sum", "prod", "mean")

#: Float ops whose result depends on ACCUMULATION ORDER (tolerance class).
_REASSOCIATED = frozenset({"sum", "prod", "mean"})

#: Input patterns (see :func:`_pattern`). ``nan_mid``/``pm_inf`` only make
#: sense for floats; bool has no ``ties``-vs-``plain`` distinction but keeps
#: the pattern list symmetric so the ``prod`` assertions stay uniform.
FLOAT_PATTERNS = ("plain", "all_equal", "pm_inf", "nan_mid", "prod_safe")
INT_PATTERNS = ("plain", "all_equal", "ties", "prod_safe")

#: ``(shape, axes)`` matrix. ``axes=None`` reduces ALL axes (rank-0 result);
#: the 2-D cases reduce one axis (a rank-1 result of 100 / 10000 elements).
SHAPE_CASES = (
    ((10001,), None),      # exact factorization 10001 = 73 x 137
    ((10000, 100), (0,)),  # rank-1 result (100,) — exact 100 x 100
    ((10000, 100), (1,)),  # total_reduced = 100 < threshold -> NOT split
    ((7,), None),          # tiny -> NOT split
    ((1024,), None),       # exactly split_reduction_threshold -> split 32x32
    ((1025,), None),       # just above the threshold -> split 25 x 41
    ((10007,), None),      # PRIME -> the padded path 128 x 79 (pad 105)
)

#: The three distinct split classes the dtype matrix rides on.
_DTYPE_SHAPES = (
    ((10001,), None),
    ((10000, 100), (0,)),
    ((10007,), None),
)


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------


def _graph(axes):
    """One graph computing ALL seven reductions over the same input.

    ``axes=None`` reduces every axis (``etl.argmin``/``argmax`` take the
    singular ``axis`` kwarg, the others a tuple ``axes``).
    """

    def fn(x):
        kw_r = {} if axes is None else {"axes": axes}
        kw_a = {} if axes is None else {"axis": axes[0]}
        return (
            etl.min(x, **kw_r),
            etl.max(x, **kw_r),
            etl.argmin(x, **kw_a),
            etl.argmax(x, **kw_a),
            etl.sum(x, **kw_r),
            etl.prod(x, **kw_r),
            etl.mean(x, **kw_r),
        )

    return fn


def _single_op_graph(op, axes):
    """A one-op graph (the export-gate tests do not need the full matrix)."""

    def fn(x):
        if axes is None:
            return _OP_CALLS[op](x, {})
        if op in ("argmin", "argmax"):
            return _OP_CALLS[op](x, {"axis": axes[0]})
        return _OP_CALLS[op](x, {"axes": axes})

    return fn


_OP_CALLS = {
    "min": lambda x, kw: etl.min(x, **kw),
    "max": lambda x, kw: etl.max(x, **kw),
    "argmin": lambda x, kw: etl.argmin(x, **kw),
    "argmax": lambda x, kw: etl.argmax(x, **kw),
    "sum": lambda x, kw: etl.sum(x, **kw),
    "prod": lambda x, kw: etl.prod(x, **kw),
    "mean": lambda x, kw: etl.mean(x, **kw),
}


# ---------------------------------------------------------------------------
# compile-once cache + builders
# ---------------------------------------------------------------------------


#: ``key -> (executable, mlir)`` — the whole file shares one compile per
#: (shape, axes, dtype, option set).
_COMPILED: dict = {}


def _build(fn, spec, *, split=None, exporter_options=None):
    """trace -> lower -> compile -> load for iree; returns ``(exe, mlir)``.

    The default path reuses ``_adapter_utils.stage`` (the shared
    trace/lower/compile helper of the sibling adapter suites).
    ``split`` is the reserved adapter lower option; ``exporter_options`` are
    EXPORTER-ONLY keys (see :func:`_compile_with_exporter_options`).
    """
    if exporter_options is not None:
        return _compile_with_exporter_options(fn, spec, exporter_options)
    if split is None:
        _, lowered, artifact = u.stage("iree", fn, (spec,))
    else:
        graph = etl.trace(fn, spec)
        lowered = etl.lower(graph, backend="iree", split_reduction=split)
        artifact = etl.compile(lowered)
    return etl.load(artifact), lowered.text()


def _compile_with_exporter_options(fn, spec, exporter_options):
    """Compile ``fn`` with EXPORTER-ONLY options forced.

    ``split_reduction_non_exact`` and ``split_reduction_threshold`` are
    exporter-only keys today: ``CompilerBackend._exporter_options`` forwards
    only ``split_reduction`` (and the adapters' ``KNOWN_OPTIONS["lower"]``
    lists only that one), so ``etl.build(..., split_reduction_non_exact=True)``
    raises the adapter's unrecognized-lower-option ``BackendError``. This
    helper produces the exact MLIR the option emits and swaps it into the
    lowered payload (the compile/load/run stages stay the product's).
    """
    graph = etl.trace(fn, spec)
    base = etl.lower(graph, backend="iree")  # for the recorded Signature
    mlir = stablehlo.export(graph, options=dict(exporter_options))
    lowered = LoweredProgram(
        backend=base.backend,
        signature=base.signature,
        payload={**base.payload, "mlir_text": mlir},
    )
    return etl.load(etl.compile(lowered)), mlir


def _executable(shape, axes, dtype, *, split=None, exporter_options=None):
    """The shared (compile-once) executable for a case."""
    key = (
        tuple(shape),
        axes,
        np.dtype(dtype).str,
        split,
        tuple(sorted((exporter_options or {}).items())),
    )
    if key not in _COMPILED:
        _COMPILED[key] = _build(
            _graph(axes),
            etl.TensorSpec(shape, dtype),
            split=split,
            exporter_options=exporter_options,
        )
    return _COMPILED[key]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _pattern(shape, dtype, name):
    """A deterministic input for one (shape, dtype, pattern) triple."""
    dtype = np.dtype(dtype)
    kind = dtype.kind
    rng = np.random.default_rng(7)
    if name == "plain":
        if kind == "f":
            return rng.standard_normal(shape).astype(dtype)
        if kind == "i":
            # Small magnitudes: an int32 sum of 10001 such values stays well
            # inside int32, so this pattern pins the SPLIT semantics without
            # tripping the unrelated integer-reduce dtype bug (BUG(etl) #2).
            return rng.integers(-500, 500, shape).astype(dtype)
        return rng.integers(0, 2, shape).astype(dtype)
    if name == "all_equal":
        value = 3.25 if kind == "f" else (7 if kind == "i" else True)
        return np.full(shape, value, dtype=dtype)
    if name == "pm_inf":
        x = rng.standard_normal(shape).astype(dtype)
        flat = x.reshape(-1)
        flat[3] = np.inf
        flat[-2] = -np.inf
        return x
    if name == "nan_mid":
        x = rng.standard_normal(shape).astype(dtype)
        x.reshape(-1)[5] = np.nan
        return x
    if name == "ties":
        if kind == "i":
            return rng.integers(0, 3, shape).astype(dtype)
        return rng.integers(0, 2, shape).astype(dtype)
    if name == "prod_safe":
        # ``prod`` of thousands of standard normals underflows to 0, which
        # would make the comparison vacuous. Floats stay within ~e^+-10 of 1;
        # int32 uses +-1 (the product never leaves int32); bool is 0/1 (its
        # ``prod`` is exactly AND, so any pattern is safe — the identity pad
        # is exercised by the all-True case here).
        if kind == "f":
            return (1.0 + rng.uniform(-1e-3, 1e-3, shape)).astype(dtype)
        if kind == "i":
            return rng.choice(np.array([-1, 1], dtype=dtype), size=shape)
        return np.ones(shape, dtype=dtype)
    raise AssertionError(f"unknown pattern {name!r}")  # pragma: no cover


def _patterns(dtype):
    return FLOAT_PATTERNS if np.dtype(dtype).kind == "f" else INT_PATTERNS


# ---------------------------------------------------------------------------
# assertion helpers
# ---------------------------------------------------------------------------


def _checked(op, dtype, pattern):
    """Whether ``op`` is asserted for this dtype/pattern.

    ``prod`` is only meaningful on ``prod_safe`` for float/int (see
    :func:`_pattern`); the remaining exclusions are the ``# BUG(etl)`` cases
    pinned separately at the bottom of this module.
    """
    kind = np.dtype(dtype).kind
    if op == "prod" and kind in "fi":
        return pattern == "prod_safe"
    if kind == "b" and op in ("sum", "mean"):
        return False  # BUG(etl) #1: bool sum/mean reduce in i1 (logical OR)
    if kind == "f" and pattern == "nan_mid" and op in ("argmin", "argmax"):
        return False  # BUG(etl) #3: arg-reduce NaN semantics differ from numpy
    return True


def _magnitude(op, expected, x, axes):
    """The input-magnitude scale of the float tolerance (see the docstring).

    ``sum(|x|)`` over the REDUCED axes for ``sum``, divided by the reduced
    extent for ``mean`` (a mean is ``sum / n``), and the magnitude of the
    product itself for ``prod`` (a relative comparison). Non-finite input
    elements contribute nothing to the scale — NaN/inf behaviour is compared
    by ``equal_nan`` instead, and an infinite scale would mask real errors.
    """
    reduced = tuple(range(x.ndim)) if axes is None else tuple(axes)
    finite = np.where(np.isfinite(x), np.abs(x), 0.0)
    l1 = np.sum(finite, axis=reduced)
    if op == "sum":
        return l1
    if op == "mean":
        count = 1
        for axis in reduced:
            count *= x.shape[axis]
        return l1 / count
    return np.abs(expected)


def etl_np(value):
    """Unwrap an etl ``Tensor`` (run outputs are ``Tensor``s, not ndarrays)."""
    return np.asarray(value.numpy()) if hasattr(value, "numpy") else np.asarray(value)


def _assert_close(op, got, expected, magnitude):
    """``|got - expected| <= NON_EXACT_RTOL * (|expected| + scale)`` per element.

    Hand-rolled instead of ``np.testing.assert_allclose`` because that rejects
    a NON-SCALAR ``atol`` (numpy 2.5 formats the array argument into its
    failure message and raises ``TypeError``). NaN positions must match
    structurally (``equal_nan`` semantics: both the numpy backend and the
    compiler produce NaN for a NaN input here); an infinite error still
    exceeds the finite bound, so it fails.
    """
    got = got.astype(np.float64)
    expected = expected.astype(np.float64)
    bound = NON_EXACT_RTOL * (
        np.abs(expected)
        + np.broadcast_to(np.asarray(magnitude, np.float64), expected.shape)
    )
    nan_got, nan_expected = np.isnan(got), np.isnan(expected)
    np.testing.assert_array_equal(
        nan_got, nan_expected, err_msg=f"{op}: NaN positions differ"
    )
    # An INFINITE result has no meaningful error bound — it must be matched
    # exactly (overflow to +-inf on both sides is agreement, +inf vs a finite
    # value is a failure). ``inf - inf`` is NaN, so this cannot be folded into
    # the bound comparison below.
    infinite = np.isinf(expected)
    np.testing.assert_array_equal(
        got[infinite], expected[infinite], err_msg=f"{op}: infinite results differ"
    )
    finite = ~nan_expected & ~infinite
    excess = np.abs(got[finite] - expected[finite]) - bound[finite]
    worst = float(np.max(excess)) if excess.size else 0.0
    assert worst <= 0.0, (
        f"{op}: error exceeds the documented {NON_EXACT_RTOL} budget by "
        f"{worst:.6g} (got={got[finite][int(np.argmax(excess))]!r}, "
        f"expected={expected[finite][int(np.argmax(excess))]!r})"
    )


def _assert_matches(op, got, expected, x, axes):
    """Compare one reduction result against the numpy backend result."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, f"{op}: shape {got.shape} != {expected.shape}"
    assert got.dtype == expected.dtype, f"{op}: dtype {got.dtype} != {expected.dtype}"
    if np.dtype(x.dtype).kind == "f" and op in _REASSOCIATED:
        _assert_close(op, got, expected, _magnitude(op, expected, x, axes))
    else:
        # Grouping-invariant (min/max/argmin/argmax, integer/bool sum/prod):
        # splitting must not change a single bit. ``assert_array_equal``
        # treats NaN in matching positions as equal.
        np.testing.assert_array_equal(got, expected)


def _run_case(shape, axes, dtype, pattern, *, split=None, exporter_options=None):
    """Run one (case, pattern) and assert every checked op against numpy."""
    executable, mlir = _executable(
        shape, axes, dtype, split=split, exporter_options=exporter_options
    )
    expected_split = split is not False and _splits(shape, axes)
    assert (SPLIT_MARKER in mlir) == expected_split, (
        f"split marker mismatch for {shape} axes={axes} {np.dtype(dtype).name}: "
        f"expected split={expected_split}"
    )
    fn = _graph(axes)
    x = _pattern(shape, dtype, pattern)
    # The NaN / +-inf patterns legitimately produce invalid-value floating
    # point flags (inf - inf in a sum, nan comparisons) on BOTH sides — those
    # are the behaviour under test, not a problem to report.
    with np.errstate(invalid="ignore"):
        expected = etl.evaluate(fn, x)  # numpy reference backend
        got = etl.run(executable, x)
        expected = expected if isinstance(expected, tuple) else (expected,)
        got = got if isinstance(got, tuple) else (got,)
        assert len(got) == len(OPS)
        for op, got_value, expected_value in zip(OPS, got, expected):
            if not _checked(op, dtype, pattern):
                continue
            _assert_matches(op, got_value.numpy(), expected_value.numpy(), x, axes)


def _splits(shape, axes):
    """Whether ANY reduction of this case is split (the gate, restated).

    A reduction is split when all dims are static, ``total_reduced >= 1024``
    and the result has <= 1024 elements — true for every op of the eligible
    cases, false for the whole case when the reduced extent is small or the
    result is large (those cases keep the single-stage emission for every
    op, so one flag covers the graph).
    """
    if axes is None:
        reduced = tuple(range(len(shape)))
    else:
        reduced = tuple(axes)
    total = 1
    for axis in reduced:
        total *= shape[axis]
    result_elems = 1
    for axis, extent in enumerate(shape):
        if axis not in reduced:
            result_elems *= extent
    return total >= 1024 and result_elems <= 1024


def _case_id(shape, axes, dtype, extra=""):
    axes_id = "all" if axes is None else "axes" + "".join(str(a) for a in axes)
    shape_id = "x".join(str(d) for d in shape)
    return f"{shape_id}-{axes_id}-{np.dtype(dtype).name}{extra}"


# ---------------------------------------------------------------------------
# export gates (no compile): the split really happens, with the expected
# geometry — otherwise the parity tests below would prove nothing.
# ---------------------------------------------------------------------------

_GEOMETRY_CASES = (
    ((10001,), None, np.float32, "min", "10001 -> 73 x 137"),
    ((10001,), None, np.float32, "max", "10001 -> 73 x 137"),
    ((10001,), None, np.float32, "argmin", "10001 -> 73 x 137"),
    ((10001,), None, np.float32, "argmax", "10001 -> 73 x 137"),
    ((10001,), None, np.float64, "min", "10001 -> 73 x 137"),
    ((1024,), None, np.float32, "min", "1024 -> 32 x 32"),
    ((1025,), None, np.float32, "min", "1025 -> 25 x 41"),
    ((10007,), None, np.float32, "min", "10007 -> 128 x 79"),
    ((10007,), None, np.int32, "min", "10007 -> 127 x 79"),
    ((10007,), None, np.int32, "max", "10007 -> 128 x 79"),
    ((10007,), None, np.int32, "sum", "10007 -> 128 x 79"),
    ((10001,), None, np.int32, "prod", "10001 -> 73 x 137"),
    ((10001,), None, np.bool_, "min", "10001 -> 73 x 137"),
    ((10001,), None, np.bool_, "max", "10001 -> 73 x 137"),
    ((10001,), None, np.bool_, "sum", "10001 -> 73 x 137"),
    ((10001,), None, np.bool_, "prod", "10001 -> 73 x 137"),
    ((10000, 100), (0,), np.float32, "min", "10000 -> 100 x 100"),
    ((10000, 100), (0,), np.float32, "argmin", "10000 -> 100 x 100"),
    ((10000, 100), (0,), np.float64, "max", "10000 -> 100 x 100"),
)


@pytest.mark.parametrize(
    "shape,axes,dtype,op,geometry", _GEOMETRY_CASES,
    ids=[_case_id(s, a, d, f"-{op}") for s, a, d, op, _ in _GEOMETRY_CASES],
)
def test_exact_class_splits_with_expected_geometry(shape, axes, dtype, op, geometry):
    """The exact class splits BY DEFAULT, with the expected ``R x C`` text."""
    graph = etl.trace(_single_op_graph(op, axes), etl.TensorSpec(shape, dtype))
    mlir = stablehlo.export(graph, options={})
    lines = [line for line in mlir.splitlines() if SPLIT_MARKER in line]
    assert len(lines) == 1, f"expected exactly one split marker, got {lines}"
    assert geometry in lines[0], lines[0]
    # Two reduce stages, not one.
    assert mlir.count('"stablehlo.reduce"') == 2


@pytest.mark.parametrize(
    "shape,axes,dtype,op",
    (
        ((7,), None, np.float32, "min"),
        ((10000, 100), (1,), np.float32, "min"),
        ((10000, 100), (1,), np.float64, "argmax"),
        ((10001,), None, np.float32, "sum"),
        ((10001,), None, np.float32, "prod"),
        ((10001,), None, np.float32, "mean"),
        ((10001,), None, np.float64, "sum"),
    ),
    ids=lambda value: str(value),
)
def test_not_split_outside_the_gates(shape, axes, dtype, op):
    """Below the extent threshold / large result / float sum-prod-mean: one stage."""
    graph = etl.trace(_single_op_graph(op, axes), etl.TensorSpec(shape, dtype))
    mlir = stablehlo.export(graph, options={})
    assert SPLIT_MARKER not in mlir
    assert mlir.count('"stablehlo.reduce"') == 1


@pytest.mark.parametrize("op", ("sum", "prod", "mean"))
def test_float_accumulation_splits_only_with_split_reduction_non_exact(op):
    """Opt-in ``split_reduction_non_exact`` splits float sum/prod (hence mean)."""
    graph = etl.trace(
        _single_op_graph(op, None), etl.TensorSpec((10001,), np.float32)
    )
    mlir = stablehlo.export(
        graph, options={"split_reduction": True, "split_reduction_non_exact": True}
    )
    lines = [line for line in mlir.splitlines() if SPLIT_MARKER in line]
    assert len(lines) == 1, lines
    assert "10001 -> 73 x 137" in lines[0]


@pytest.mark.parametrize(
    "shape,axes,dtype,op",
    (
        ((10001,), None, np.float32, "min"),
        ((10001,), None, np.float32, "argmin"),
        ((10001,), None, np.int32, "sum"),
        ((10007,), None, np.float32, "max"),
        ((10000, 100), (0,), np.float32, "argmax"),
    ),
    ids=lambda value: str(value),
)
def test_split_reduction_false_emits_the_single_stage_text(shape, axes, dtype, op):
    """The option is a switch: ``False`` keeps the single-stage emission."""
    graph = etl.trace(_single_op_graph(op, axes), etl.TensorSpec(shape, dtype))
    mlir = stablehlo.export(graph, options={"split_reduction": False})
    assert SPLIT_MARKER not in mlir
    assert mlir.count('"stablehlo.reduce"') == 1


# ---------------------------------------------------------------------------
# numeric parity (compile + run) — the shape x dtype matrix
# ---------------------------------------------------------------------------

_CASES = [(shape, axes, np.float32) for shape, axes in SHAPE_CASES] + [
    (shape, axes, dtype)
    for shape, axes in _DTYPE_SHAPES
    for dtype in (np.float64, np.int32, np.bool_)
]

_PARITY_PARAMS = [
    pytest.param(shape, axes, dtype, pattern, id=_case_id(shape, axes, dtype, f"-{pattern}"))
    for shape, axes, dtype in _CASES
    for pattern in _patterns(dtype)
]


@pytest.mark.parametrize("shape,axes,dtype,pattern", _PARITY_PARAMS)
def test_split_reduction_matches_numpy(shape, axes, dtype, pattern):
    """Split-on parity against the numpy reference backend, per op class."""
    _run_case(shape, axes, dtype, pattern)


# ---------------------------------------------------------------------------
# control: the option is SEMANTICS-PRESERVING IN BOTH DIRECTIONS
# ---------------------------------------------------------------------------

_CONTROL_CASES = (
    ((10001,), None),
    ((10000, 100), (0,)),
    ((1024,), None),
    ((10007,), None),
)

_CONTROL_PARAMS = [
    pytest.param(shape, axes, pattern, id=_case_id(shape, axes, np.float32, f"-{pattern}"))
    for shape, axes in _CONTROL_CASES
    for pattern in ("plain", "all_equal", "prod_safe")
]


@pytest.mark.parametrize("shape,axes,pattern", _CONTROL_PARAMS)
def test_split_reduction_off_matches_numpy(shape, axes, pattern):
    """``split_reduction=False`` (the pre-option text) matches numpy too.

    Together with the group above this proves the option never changes the
    ANSWER — only the emission — for every case where it applies.
    """
    _run_case(shape, axes, np.float32, pattern, split=False)


@pytest.mark.parametrize(
    "shape,axes,dtype",
    (
        ((10001,), None, np.float32),
        ((10007,), None, np.float32),
    ),
    ids=lambda value: str(value),
)
def test_float_accumulation_split_matches_numpy_within_the_documented_budget(
    shape, axes, dtype
):
    """The opt-in float split stays inside the documented tolerance budget.

    The reassociation changes the bits (measured <= 4.3e-6 relative here,
    budget 4.5e-5 — see the module docstring); the exact class in the SAME
    graph stays bit-exact.
    """
    _run_case(
        shape,
        axes,
        dtype,
        "plain",
        exporter_options={"split_reduction": True, "split_reduction_non_exact": True},
    )
    _run_case(
        shape,
        axes,
        dtype,
        "prod_safe",
        exporter_options={"split_reduction": True, "split_reduction_non_exact": True},
    )


# ---------------------------------------------------------------------------
# arg-reduce tie semantics: FIRST OCCURRENCE must survive both stages
# ---------------------------------------------------------------------------


def test_arg_ties_resolve_to_first_occurrence_across_split_stages():
    """Tied extrema resolve to the lowest flat index, as ``np.argmin`` does.

    ``(10001,)`` splits as ``73 x 137``, so rows are 137 long: the two
    minima at 7000/7001 land in the SAME stage-1 row while the third (9000)
    is in a later row, so the tie-break is exercised both WITHIN a stage-1
    row and ACROSS the two partials.
    """
    shape, axes, dtype = (10001,), None, np.float32
    executable, mlir = _executable(shape, axes, dtype)
    assert SPLIT_MARKER in mlir and "10001 -> 73 x 137" in mlir
    x = np.full(shape, 5.0, dtype=dtype)
    x[7000] = x[7001] = 1.0  # tie for the minimum, same stage-1 row
    x[9000] = 1.0            # tie for the minimum, later stage-1 row
    x[2000] = x[8000] = 9.0  # tie for the maximum, different rows
    got = etl.run(executable, x)
    expected = etl.evaluate(_graph(axes), x)
    assert int(got[2].numpy()) == 7000 == int(np.argmin(x))
    assert int(got[3].numpy()) == 2000 == int(np.argmax(x))
    assert int(expected[2].numpy()) == 7000
    assert int(expected[3].numpy()) == 2000


def test_arg_ties_resolve_to_first_occurrence_per_column():
    """2-D, ``axis=0``: per column the first row index wins (a rank-1 result)."""
    shape, axes, dtype = (10000, 100), (0,), np.float32
    executable, mlir = _executable(shape, axes, dtype)
    assert SPLIT_MARKER in mlir and "10000 -> 100 x 100" in mlir
    x = np.full(shape, 0.5, dtype=dtype)
    x[4000, :] = x[7500, :] = -1.0  # global min in rows 4000 and 7500
    x[3000, :] = x[9000, :] = 2.0   # global max in rows 3000 and 9000
    got = etl.run(executable, x)
    expected = etl.evaluate(_graph(axes), x)
    np.testing.assert_array_equal(etl_np(got[2]), np.full(100, 4000, dtype=np.int64))
    np.testing.assert_array_equal(etl_np(got[3]), np.full(100, 3000, dtype=np.int64))
    np.testing.assert_array_equal(etl_np(got[2]), np.argmin(x, axis=0))
    np.testing.assert_array_equal(etl_np(got[3]), np.argmax(x, axis=0))
    np.testing.assert_array_equal(etl_np(expected[2]), np.argmin(x, axis=0))
    np.testing.assert_array_equal(etl_np(expected[3]), np.argmax(x, axis=0))


# ---------------------------------------------------------------------------
# BUG(etl) pins — genuine exporter bugs, NOT caused by ``split_reduction``.
#
# Each one reproduces identically with ``split_reduction=False`` AND on
# shapes far below ``split_reduction_threshold`` (i.e. the single-stage
# emission, which predates the option, is equally wrong), so these tests must
# not be confused with a split-reduction regression. Every pin keeps asserting
# the CORRECT (numpy reference backend) result and is marked
# ``xfail(strict=True)``: expected to fail while the bug lives, and it turns
# into a suite failure (XPASS under ``strict``) the moment etl is fixed — the
# assertion is never weakened. They matter here because they make dtype/NaN
# coverage of the parity suite above silently wrong: the suite has to EXCLUDE
# the affected (op, dtype, pattern) combinations in ``_checked``.
# ---------------------------------------------------------------------------


# BUG(etl): bool sum/mean are reduced in i1 (stablehlo.add on tensor<i1> is
# logical OR) and only then converted to the declared i64 result dtype, so the
# popcount collapses to 0/1. Reproduces with split_reduction=False too.
@pytest.mark.xfail(
    strict=True,
    reason="BUG(etl): bool sum/mean reduce in i1 (stablehlo.add == OR), so the "
    "popcount collapses to 0/1 (also with split_reduction=False)",
)
def test_bug_bool_sum_reduces_in_i1():
    """BUG(etl): bool ``sum``/``mean`` reduce in i1 (``stablehlo.add`` == OR).

    ``etl.sum`` over bool is declared ``int64`` (numpy: the popcount) but the
    exporter emits the reducer in the OPERAND dtype — ``stablehlo.add`` on
    ``tensor<i1>`` is logical OR — and only then ``stablehlo.convert``s the
    single-bit result to i64, so the count collapses to 0/1. ``mean`` (the
    ``sum / n`` composition) is wrong for the same reason.

    Minimal repro::

        import numpy as np, etl
        f = lambda x: etl.sum(x)
        spec = etl.TensorSpec((2000,), np.bool_)
        x = np.zeros(2000, np.bool_); x[:7] = True
        etl.run(etl.build(f, spec, backend="iree"), x).numpy()   # -> 1
        etl.evaluate(f, x).numpy()                                # -> 7
    """
    shape, dtype = (2000,), np.bool_

    def fn(x):
        return etl.sum(x), etl.mean(x)

    x = np.zeros(shape, dtype=dtype)
    x[:7] = True
    spec = etl.TensorSpec(shape, dtype)
    split_on = etl.run(etl.build(fn, spec, backend="iree"), x)
    split_off = etl.run(
        etl.build(fn, spec, backend="iree", split_reduction=False), x
    )
    expected = etl.evaluate(fn, x)
    # Both emission paths produce the SAME (wrong) answer: not a split bug.
    np.testing.assert_array_equal(etl_np(split_on[0]), etl_np(split_off[0]))
    np.testing.assert_array_equal(etl_np(split_on[1]), etl_np(split_off[1]))
    np.testing.assert_array_equal(etl_np(split_on[0]), etl_np(expected[0]))


# BUG(etl): int sum/prod are reduced in the OPERAND dtype and then widened, so
# an int32 reduction wraps at 32 bits before the convert to the declared i64.
# Reproduces with split_reduction=False too (and far below the split gate).
@pytest.mark.xfail(
    strict=True,
    reason="BUG(etl): int sum/prod reduce in the OPERAND dtype and widen "
    "afterwards, so an int32 reduction wraps at 32 bits",
)
def test_bug_integer_reductions_wrap_in_the_operand_dtype():
    """BUG(etl): int ``sum``/``prod`` reduce in the OPERAND dtype, then widen.

    ``etl.prod``/``etl.sum`` over int32 are declared ``int64`` (numpy
    accumulates in int64) but the exporter reduces in i32 and
    ``stablehlo.convert``s the already-wrapped value to i64, so an int32
    reduction silently wraps at 32 bits.

    Minimal repro::

        import numpy as np, etl
        f = lambda x: etl.prod(x)
        spec = etl.TensorSpec((13,), np.int32)
        x = np.arange(1, 14, dtype=np.int32)          # 13! = 6227020800
        etl.run(etl.build(f, spec, backend="iree"), x).numpy()  # -> wrapped i32
        etl.evaluate(f, x).numpy()                              # -> 6227020800
    """
    shape, dtype = (13,), np.int32

    def fn(x):
        return etl.sum(x), etl.prod(x)

    x = np.arange(1, 14, dtype=dtype)  # prod 13! = 6227020800 > int32 max
    spec = etl.TensorSpec(shape, dtype)
    split_on = etl.run(etl.build(fn, spec, backend="iree"), x)
    split_off = etl.run(
        etl.build(fn, spec, backend="iree", split_reduction=False), x
    )
    expected = etl.evaluate(fn, x)
    np.testing.assert_array_equal(etl_np(split_on[1]), etl_np(split_off[1]))
    # numpy accumulates in int64: 13! = 6227020800, no wrap; iree wraps in i32.
    np.testing.assert_array_equal(etl_np(expected[1]), np.asarray(6227020800, np.int64))
    np.testing.assert_array_equal(etl_np(split_on[1]), etl_np(expected[1]))


# BUG(etl): the arg-reduce comparator has no NaN case, so argmin/argmax over an
# array containing NaN return the index of a REAL extremum, while numpy (and the
# numpy reference backend) return the index of the FIRST NaN. min/max themselves
# DO propagate NaN correctly. Reproduces with split_reduction=False too.
@pytest.mark.xfail(
    strict=True,
    reason="BUG(etl): argmin/argmax ignore NaN while numpy returns the first "
    "NaN index (min/max propagate NaN correctly)",
)
def test_bug_arg_reduce_nan_semantics_do_not_match_numpy():
    """BUG(etl): ``argmin``/``argmax`` ignore NaN, numpy returns the first NaN.

    ``np.argmin``/``np.argmax`` on an array containing NaN return the index of
    the FIRST NaN (every comparison against NaN is False, so NaN "wins"). The
    StableHLO arg-reduce comparator is
    ``(value GT|LT rhs) OR (value EQ rhs AND idx LT rhs_idx)``, and both
    terms are false for a NaN operand, so the NaN never wins and the result
    is the index of a REAL extremum (which partial happens to win depends on
    the stage-1 row layout). ``min``/``max`` themselves DO match numpy (NaN
    propagates, pinned green by the parity suite); only the arg-reduces
    diverge — and identically with ``split_reduction=False``.

    Minimal repro::

        import numpy as np, etl
        f = lambda x: etl.argmin(x)
        spec = etl.TensorSpec((10001,), np.float32)
        x = np.random.default_rng(0).standard_normal(10001).astype(np.float32)
        x[5] = np.nan
        etl.run(etl.build(f, spec, backend="iree"), x).numpy()  # -> int, not 5
        etl.evaluate(f, x).numpy()                              # -> 5
    """
    shape, dtype = (10001,), np.float32

    def fn(x):
        return etl.argmin(x), etl.argmax(x)

    x = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    x[5] = np.nan
    spec = etl.TensorSpec(shape, dtype)
    split_on = etl.run(etl.build(fn, spec, backend="iree"), x)
    split_off = etl.run(
        etl.build(fn, spec, backend="iree", split_reduction=False), x
    )
    expected = etl.evaluate(fn, x)
    np.testing.assert_array_equal(etl_np(split_on[0]), etl_np(split_off[0]))
    np.testing.assert_array_equal(etl_np(split_on[1]), etl_np(split_off[1]))
    np.testing.assert_array_equal(etl_np(expected[0]), np.asarray(5, np.int64))
    np.testing.assert_array_equal(etl_np(split_on[0]), etl_np(expected[0]))
