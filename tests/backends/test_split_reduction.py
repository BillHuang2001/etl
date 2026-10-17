"""Emission goldens for the two-stage ("split") reduction exporter option.

``etl/backends/stablehlo/split_reduce.py`` emits an ELIGIBLE reduction as TWO
``stablehlo.reduce`` stages instead of one: the reduced extent ``N`` is
reshaped into ``(R, C)`` (padded to ``R * C`` when no useful factorization
exists, with the reduction IDENTITY as the pad value so slack can never win),
the INNER axis is reduced first (an intermediate with ``R`` elements — a rank
>= 1 result, which IREE parallelizes, unlike a rank-0 reduce that it lowers to
one single-threaded workgroup) and the partials are reduced in a second, tiny
stage. One MLIR comment line (``// etl split reduction: pivot axis P, N -> R x
C``) marks the stage-1 reduce so goldens can assert the split unambiguously.

This suite pins the OPTION SURFACE and the EMITTED TEXT only — ``export()``
(lower) with no compiler invocations, so it stays fast. Binding rules:

- ``split_reduction`` (bool, default True);
- ``split_reduction_threshold`` (int >= 2, default 1024) — the
  ``total_reduced`` gate, inclusive (``min`` over exactly ``(1024,)`` splits);
- ``split_reduction_non_exact`` (bool, default False) — float ``sum``/``prod``
  (and therefore ``mean``/``var``/``std``) reassociate their floating-point
  accumulation, so they split ONLY when this is opted in.

Eligibility gates: fully STATIC operand shape (a ``Dim``/``DimExpr``/``None``
KEEPS the single-stage text — never an error), ``total_reduced >=
split_reduction_threshold``, ``result_elems <= 1024``, and the reduction's
exactness class (``min``/``max`` incl. the i1 ``any``/``all`` guard shapes,
``argmin``/``argmax`` and INTEGER/bool ``sum``/``prod`` split BY DEFAULT).
``split_reduction=False`` is byte-identical to the pre-option text — asserted
here against ``split_reduction_threshold=10**9`` (the same graph forced
ineligible, which is exactly the pre-option behaviour).
"""

from typing import NamedTuple

import pytest

import etl
from etl.core import BackendError

F32 = etl.float32
I32 = etl.int32


# --- helpers ---------------------------------------------------------------

def _export(fn, *specs, options=None):
    """Trace ``fn`` over ``specs`` and export it as StableHLO MLIR text."""
    return etl.backends.stablehlo.export(etl.trace(fn, *specs), options=options)


def _counts(mlir):
    """``(reduce count, pad count, emitted line count)`` of an MLIR text."""
    return (
        mlir.count('"stablehlo.reduce"'),
        mlir.count('"stablehlo.pad"'),
        len(mlir.splitlines()),
    )


#: The single marker line the split emission prepends to the stage-1 reduce.
SPLIT_MARKER = "// etl split reduction:"


def _split_comment(pivot, n, r, c):
    return f"// etl split reduction: pivot axis {pivot}, {n} -> {r} x {c}"


# --- traced functions ------------------------------------------------------

def _fn_min(x):
    return etl.min(x)


def _fn_max(x):
    return etl.max(x)


def _fn_argmin(x):
    return etl.argmin(x)


def _fn_argmax(x):
    return etl.argmax(x)


def _fn_min_axis0(x):
    return etl.min(x, axes=(0,))


def _fn_min_axis1(x):
    return etl.min(x, axes=(1,))


def _fn_max_of_less(a, b):
    """i1 ``any`` guard shape: a bool reduction emitted as i1 ``maximum``."""
    return etl.max(etl.less(a, b))


def _fn_sum(x):
    return etl.sum(x)


def _fn_prod(x):
    return etl.prod(x)


def _fn_mean(x):
    return etl.mean(x)


# --- 1. option defaults and validation -------------------------------------

def test_normalizer_defaults():
    """Absent options resolve to (split ON, threshold 1024, non-exact OFF)."""
    from etl.backends.stablehlo.split_reduce import (
        normalize_split_reduction,
        normalize_split_reduction_non_exact,
        normalize_split_reduction_threshold,
    )

    assert normalize_split_reduction(None) is True
    assert normalize_split_reduction_threshold(None) == 1024
    assert normalize_split_reduction_non_exact(None) is False


def test_default_is_split_on_for_the_exact_class():
    """No options at all: an eligible ``min`` already emits two stages."""
    mlir = _export(_fn_min, etl.TensorSpec((10001,), F32))
    assert mlir.count('"stablehlo.reduce"') == 2
    assert mlir.count(SPLIT_MARKER) == 1


EXACT_CLASS_CASES = [
    # id, fn, specs, expected reduce count with DEFAULT options
    ("min_f32", _fn_min, (etl.TensorSpec((10001,), F32),), 2),
    ("max_f32", _fn_max, (etl.TensorSpec((10001,), F32),), 2),
    ("argmin_f32", _fn_argmin, (etl.TensorSpec((10001,), F32),), 2),
    ("argmax_f32", _fn_argmax, (etl.TensorSpec((10001,), F32),), 2),
    ("sum_i32", _fn_sum, (etl.TensorSpec((10001,), I32),), 2),
    ("prod_i32", _fn_prod, (etl.TensorSpec((10001,), I32),), 2),
    (
        "i1_any_guard_max_less",
        _fn_max_of_less,
        (etl.TensorSpec((10001,), F32), etl.TensorSpec((10001,), F32)),
        2,
    ),
]


@pytest.mark.parametrize(
    "fn,specs,reduces",
    [case[1:] for case in EXACT_CLASS_CASES],
    ids=[case[0] for case in EXACT_CLASS_CASES],
)
def test_exact_class_splits_by_default(fn, specs, reduces):
    """Grouping-invariant reductions split with NO option set (bit-identical)."""
    mlir = _export(fn, *specs)
    assert mlir.count('"stablehlo.reduce"') == reduces
    assert mlir.count(SPLIT_MARKER) == 1


NON_EXACT_CLASS_CASES = [
    ("sum_f32", _fn_sum, (etl.TensorSpec((10001,), F32),), 1, 2),
    ("prod_f32", _fn_prod, (etl.TensorSpec((10001,), F32),), 1, 2),
    ("mean_f32", _fn_mean, (etl.TensorSpec((10001,), F32),), 1, 2),
]


@pytest.mark.parametrize(
    "fn,specs,default_reduces,non_exact_reduces",
    [case[1:] for case in NON_EXACT_CLASS_CASES],
    ids=[case[0] for case in NON_EXACT_CLASS_CASES],
)
def test_float_sum_prod_mean_do_not_split_by_default(
    fn, specs, default_reduces, non_exact_reduces
):
    """FLOAT accumulation is reassociated by a split — opt-in only."""
    default_mlir = _export(fn, *specs)
    assert default_mlir.count('"stablehlo.reduce"') == default_reduces
    assert SPLIT_MARKER not in default_mlir

    opted_in = _export(fn, *specs, options={"split_reduction_non_exact": True})
    assert opted_in.count('"stablehlo.reduce"') == non_exact_reduces
    assert opted_in.count(SPLIT_MARKER) == 1


VALID_OPTION_CASES = [
    ("split_reduction_true", "split_reduction", True, 2),
    ("split_reduction_false", "split_reduction", False, 1),
    ("split_reduction_non_exact_false", "split_reduction_non_exact", False, 2),
    ("split_reduction_non_exact_true", "split_reduction_non_exact", True, 2),
    ("threshold_min_two", "split_reduction_threshold", 2, 2),
    ("threshold_default", "split_reduction_threshold", 1024, 2),
    ("threshold_inclusive_boundary", "split_reduction_threshold", 10001, 2),
    ("threshold_above_extent", "split_reduction_threshold", 10002, 1),
    ("threshold_huge", "split_reduction_threshold", 10**9, 1),
]


@pytest.mark.parametrize(
    "option,value,reduces",
    [case[1:] for case in VALID_OPTION_CASES],
    ids=[case[0] for case in VALID_OPTION_CASES],
)
def test_valid_option_values_are_accepted(option, value, reduces):
    """Every valid value exports (and lands on the documented side of the gate)."""
    mlir = _export(
        _fn_min, etl.TensorSpec((10001,), F32), options={option: value}
    )
    assert mlir.count('"stablehlo.reduce"') == reduces


INVALID_OPTION_CASES = [
    # id, option, value, expected message fragments (the real normalizer wording)
    (
        "split_reduction_string",
        "split_reduction",
        "yes",
        (
            "invalid split_reduction option 'yes'",
            "expected a bool: True enables the two-stage split reduction, "
            "False emits the single-stage reduce text unchanged",
        ),
    ),
    (
        "split_reduction_int",
        "split_reduction",
        1,
        ("invalid split_reduction option 1", "expected a bool"),
    ),
    (
        "split_reduction_zero",
        "split_reduction",
        0,
        ("invalid split_reduction option 0", "expected a bool"),
    ),
    (
        "threshold_one",
        "split_reduction_threshold",
        1,
        (
            "invalid split_reduction_threshold option 1",
            "expected an int >= 2 — the minimum reduced extent that is split "
            "into a two-stage reduction",
        ),
    ),
    (
        "threshold_zero",
        "split_reduction_threshold",
        0,
        ("invalid split_reduction_threshold option 0", "expected an int >= 2"),
    ),
    (
        "threshold_negative",
        "split_reduction_threshold",
        -8,
        ("invalid split_reduction_threshold option -8", "expected an int >= 2"),
    ),
    (
        "threshold_bool",
        "split_reduction_threshold",
        True,
        ("invalid split_reduction_threshold option", "expected an int >= 2"),
    ),
    (
        "threshold_float",
        "split_reduction_threshold",
        2.5,
        ("invalid split_reduction_threshold option 2.5", "expected an int >= 2"),
    ),
    (
        "threshold_string",
        "split_reduction_threshold",
        "1024",
        ("invalid split_reduction_threshold option", "expected an int >= 2"),
    ),
    (
        "non_exact_string",
        "split_reduction_non_exact",
        "no",
        (
            "invalid split_reduction_non_exact option 'no'",
            "expected a bool: True also splits float sum/prod",
        ),
    ),
    (
        "non_exact_int",
        "split_reduction_non_exact",
        0,
        ("invalid split_reduction_non_exact option 0", "expected a bool"),
    ),
]


@pytest.mark.parametrize(
    "option,value,fragments",
    [case[1:] for case in INVALID_OPTION_CASES],
    ids=[case[0] for case in INVALID_OPTION_CASES],
)
def test_invalid_option_values_raise_backend_error(option, value, fragments):
    """Garbage is rejected explicitly, naming the option (never silently)."""
    graph = etl.trace(_fn_min, etl.TensorSpec((10001,), F32))
    with pytest.raises(BackendError) as excinfo:
        etl.backends.stablehlo.export(graph, options={option: value})
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message


def test_threshold_below_two_is_rejected_even_as_a_useful_gate():
    """``split_reduction_threshold`` < 2 is a validation error, not a gate."""
    from etl.backends.stablehlo.split_reduce import (
        normalize_split_reduction_threshold,
    )

    for bad in (1, 0, -1):
        with pytest.raises(BackendError) as excinfo:
            normalize_split_reduction_threshold(bad)
        assert "split_reduction_threshold" in str(excinfo.value)
        assert "expected an int >= 2" in str(excinfo.value)
    assert normalize_split_reduction_threshold(2) == 2


# --- 2. eligibility gates keep the single-stage text ----------------------

DYNAMIC_SPEC_CASES = [
    (
        "symbolic_all_axes",
        _fn_min,
        (etl.TensorSpec((etl.Dim("B"),), F32),),
    ),
    (
        "symbolic_reduced_axis",
        _fn_min_axis0,
        (etl.TensorSpec((etl.Dim("B"), 10000), F32),),
    ),
    (
        "symbolic_reduced_axis_transposed",
        _fn_min_axis0,
        (etl.TensorSpec((10000, etl.Dim("B")), F32),),
    ),
]


@pytest.mark.parametrize(
    "fn,specs",
    [case[1:] for case in DYNAMIC_SPEC_CASES],
    ids=[case[0] for case in DYNAMIC_SPEC_CASES],
)
def test_dynamic_dims_keep_single_stage(fn, specs):
    """A non-static dim keeps today's emission — silently, never an error."""
    mlir = _export(fn, *specs)
    assert mlir.count('"stablehlo.reduce"') == 1
    assert SPLIT_MARKER not in mlir


def test_below_threshold_keeps_single_stage():
    """``total_reduced`` 1023 < the default 1024 threshold: no split."""
    mlir = _export(_fn_min, etl.TensorSpec((1023,), F32))
    assert _counts(mlir) == (1, 0, 11)
    assert SPLIT_MARKER not in mlir


def test_threshold_gate_is_inclusive():
    """``total_reduced`` == threshold splits (``min`` over exactly ``(1024,)``)."""
    mlir = _export(_fn_min, etl.TensorSpec((1024,), F32))
    assert _counts(mlir) == (2, 0, 19)
    assert _split_comment(0, 1024, 32, 32) in mlir


LARGE_RESULT_CASES = [
    # a rank-1 reduction whose RESULT has more than 1024 elements
    ("result_2000", (2000, 2000)),
    ("result_1025", (2000, 1025)),
]


@pytest.mark.parametrize(
    "shape", [case[1] for case in LARGE_RESULT_CASES],
    ids=[case[0] for case in LARGE_RESULT_CASES],
)
def test_large_result_keeps_single_stage(shape):
    """``result_elems > 1024`` is not a rank-0/tiny result: no split."""
    mlir = _export(_fn_min_axis0, etl.TensorSpec(shape, F32))
    assert _counts(mlir) == (1, 0, 11)
    assert SPLIT_MARKER not in mlir


def test_result_at_the_1024_boundary_still_splits():
    """``result_elems == 1024`` is still ``<= 1024`` — the gate is inclusive."""
    mlir = _export(_fn_min_axis0, etl.TensorSpec((2000, 1024), F32))
    assert mlir.count('"stablehlo.reduce"') == 2
    assert mlir.count(SPLIT_MARKER) == 1


def test_reduce_over_no_reduced_dims_keeps_single_stage():
    """A reduction keeping every dim (axes=() equivalent) is never split."""
    # ``min`` over a rank-2 tensor on a NON-reduced-only shape: axes=(1,) of a
    # (10000, 100) tensor reduces 100 (< threshold) and keeps 10000 as result.
    mlir = _export(_fn_min_axis1, etl.TensorSpec((10000, 100), F32))
    assert _counts(mlir) == (1, 0, 11)
    assert SPLIT_MARKER not in mlir


# --- 3. the split marker set ----------------------------------------------

def test_split_case_marker_set():
    """A split emits exactly TWO reduces and exactly ONE marker comment."""
    mlir = _export(_fn_min, etl.TensorSpec((10001,), F32))
    assert mlir.count('"stablehlo.reduce"') == 2
    assert mlir.count(SPLIT_MARKER) == 1


def test_single_stage_case_marker_set():
    """A non-split case keeps ONE reduce and NO marker comment."""
    mlir = _export(_fn_sum, etl.TensorSpec((10001,), F32))
    assert mlir.count('"stablehlo.reduce"') == 1
    assert mlir.count(SPLIT_MARKER) == 0


# --- 4. verified shapes: counts, line totals and split text ---------------


class Case(NamedTuple):
    """One verified emission golden.

    ``split`` / ``single`` are the ``(reduces, pads, lines)`` of the DEFAULT
    export and of the ``split_reduction=False`` export respectively;
    ``splits`` says whether the default export must carry the marker.
    """

    case_id: str
    fn: object
    specs: tuple
    splits: bool
    split: tuple
    single: tuple
    substrings: tuple


SHAPE_CASES = [
    # N = 10001 is PRIME, so no exact (R, C) factorization exists -> padded
    # generic path... except 10001 = 73 * 137 does factor: exact, 0 pads.
    Case(
        "min_10001_f32",
        _fn_min,
        (etl.TensorSpec((10001,), F32),),
        True,
        (2, 0, 19),
        (1, 0, 11),
        (
            _split_comment(0, 10001, 73, 137),
            "(tensor<10001xf32>) -> tensor<73x137xf32>",
        ),
    ),
    Case(
        # ``argmin`` carries many more lines than the plain ``min`` case (52 vs
        # 19 split, 27 vs 11 single-stage): the NaN-aware arg-reduce emits its
        # NaN-selecting compare/select chain (the ``x != x`` / ``acc != acc``
        # test pair feeding the index tie-break) INSIDE the reduce region, which
        # is 14 of the extra split lines and 7 of the extra single-stage ones.
        "argmin_10001_f32",
        _fn_argmin,
        (etl.TensorSpec((10001,), F32),),
        True,
        (2, 0, 52),
        (1, 0, 27),
        (
            _split_comment(0, 10001, 73, 137),
            "(tensor<10001xf32>) -> tensor<73x137xf32>",
            # the composite (value, index) index operand: iota(pivot)*C + iota
            "stablehlo.iota dim = 0 : tensor<73x137xi64>",
            "stablehlo.iota dim = 1 : tensor<73x137xi64>",
        ),
    ),
    Case(
        "min_10000x100_axes0_f32",
        _fn_min_axis0,
        (etl.TensorSpec((10000, 100), F32),),
        True,
        (2, 0, 19),
        (1, 0, 11),
        (
            _split_comment(0, 10000, 100, 100),
            # the reduced axis 0 becomes (R, C); the kept axis stays -> rank 3
            "(tensor<10000x100xf32>) -> tensor<100x100x100xf32>",
            "(tensor<100x100x100xf32>, tensor<f32>) -> tensor<100x100xf32>",
        ),
    ),
    Case(
        # axes=(1,) reduces 100 elements (< threshold) and keeps 10000 result
        # elements (> 1024) — ineligible on TWO gates.
        "min_10000x100_axes1_f32",
        _fn_min_axis1,
        (etl.TensorSpec((10000, 100), F32),),
        False,
        (1, 0, 11),
        (1, 0, 11),
        (),
    ),
    Case(
        # the i1 "any" guard shape: reduce_max over a comparison result
        "max_of_less_i1_10001",
        _fn_max_of_less,
        (etl.TensorSpec((10001,), F32), etl.TensorSpec((10001,), F32)),
        True,
        (2, 0, 20),
        (1, 0, 12),
        (
            _split_comment(0, 10001, 73, 137),
            # the i1 guard keeps i1 through both stages (maximum on i1)
            "(tensor<10001xi1>) -> tensor<73x137xi1>",
            "stablehlo.maximum",
            "(tensor<73x137xi1>, tensor<i1>) -> tensor<73xi1>",
            "(tensor<73xi1>, tensor<i1>) -> tensor<i1>",
        ),
    ),
    Case(
        # 10007 is prime: no exact factorization -> pad 105 to 128 x 79
        "min_10007_f32_padded",
        _fn_min,
        (etl.TensorSpec((10007,), F32),),
        True,
        (2, 1, 21),
        (1, 0, 11),
        (
            "edge_padding_high = array<i64: 105>",
            "(tensor<10007xf32>, tensor<f32>) -> tensor<10112xf32>",
            "(tensor<10112xf32>) -> tensor<128x79xf32>",
            _split_comment(0, 10007, 128, 79),
        ),
    ),
    Case(
        # signed narrow-int min on ROCm: no stage extent may be a multiple of
        # 64, so the rows are re-picked 128 -> 127 (79 stays clean), pad 26.
        "min_10007_i32_rocm_guard",
        _fn_min,
        (etl.TensorSpec((10007,), I32),),
        True,
        (2, 1, 21),
        (1, 0, 11),
        (
            "edge_padding_high = array<i64: 26>",
            "(tensor<10007xi32>, tensor<i32>) -> tensor<10033xi32>",
            "(tensor<10033xi32>) -> tensor<127x79xi32>",
            _split_comment(0, 10007, 127, 79),
        ),
    ),
    Case(
        # FLOAT sum: reassociated by a split -> single-stage by default
        "sum_10001_f32_default",
        _fn_sum,
        (etl.TensorSpec((10001,), F32),),
        False,
        (1, 0, 11),
        (1, 0, 11),
        (),
    ),
    Case(
        # threshold boundary: exactly 1024 reduced elements DO split
        "min_1024_threshold_boundary",
        _fn_min,
        (etl.TensorSpec((1024,), F32),),
        True,
        (2, 0, 19),
        (1, 0, 11),
        (
            _split_comment(0, 1024, 32, 32),
            "(tensor<1024xf32>) -> tensor<32x32xf32>",
        ),
    ),
    Case(
        "min_1023_below_threshold",
        _fn_min,
        (etl.TensorSpec((1023,), F32),),
        False,
        (1, 0, 11),
        (1, 0, 11),
        (),
    ),
]


@pytest.mark.parametrize(
    "case", SHAPE_CASES, ids=[case.case_id for case in SHAPE_CASES]
)
def test_verified_shape_goldens(case):
    """``(reduces, pads, lines)`` plus the split comment/reshape text."""
    mlir = _export(case.fn, *case.specs)
    assert _counts(mlir) == case.split
    assert (mlir.count(SPLIT_MARKER) == 1) is case.splits
    for substring in case.substrings:
        assert substring in mlir


def test_float_sum_splits_when_non_exact_is_opted_in():
    """``split_reduction_non_exact=True``: float ``sum`` gains the split text."""
    mlir = _export(
        _fn_sum,
        etl.TensorSpec((10001,), F32),
        options={"split_reduction_non_exact": True},
    )
    assert _counts(mlir) == (2, 0, 19)
    assert _split_comment(0, 10001, 73, 137) in mlir


def test_non_exact_option_does_not_change_the_exact_class_text():
    """Opting in is a no-op for reductions that already split by default."""
    specs = (etl.TensorSpec((10001,), F32),)
    assert _export(_fn_min, *specs, options={"split_reduction_non_exact": True}) \
        == _export(_fn_min, *specs)


# --- 5. split_reduction=False is the pre-option text ----------------------


@pytest.mark.parametrize(
    "case", SHAPE_CASES, ids=[case.case_id for case in SHAPE_CASES]
)
def test_split_off_is_byte_identical_to_a_huge_threshold(case):
    """Backward-compat contract: off == the same graph forced ineligible.

    ``split_reduction_threshold=10**9`` makes every reduction ineligible, which
    is exactly the pre-option emission — so the two texts must be identical
    byte for byte, and must equal the pinned single-stage counts.
    """
    specs = case.specs
    off = _export(case.fn, *specs, options={"split_reduction": False})
    forced = _export(
        case.fn, *specs, options={"split_reduction_threshold": 10**9}
    )
    assert off == forced
    assert _counts(off) == case.single
    assert SPLIT_MARKER not in off


def test_split_off_suppresses_the_non_exact_split_too():
    """``split_reduction=False`` wins over ``split_reduction_non_exact=True``."""
    specs = (etl.TensorSpec((10001,), F32),)
    mlir = _export(
        _fn_sum,
        *specs,
        options={"split_reduction": False, "split_reduction_non_exact": True},
    )
    assert _counts(mlir) == (1, 0, 11)
    assert mlir == _export(
        _fn_sum, *specs, options={"split_reduction_threshold": 10**9}
    )
