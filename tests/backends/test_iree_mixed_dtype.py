"""IREE mixed-dtype compile+run regression tests (fix 6aec043).

etl IR allows mixed-dtype operands on binary elementwise ops / compare /
select (the result dtype follows numpy promotion — e.g. a Python scalar
like ``3`` becomes a scalar i64 weak constant at trace time); the StableHLO
writer equalizes operand dtypes by inserting ``stablehlo.convert`` after
shape equalization (MLIR goldens pinned in ``tests/backends/
test_stablehlo.py`` section 14). Before the fix these graphs lowered into
invalid StableHLO that iree rejected at ``etl.compile`` with "use of value
... expects different type than prior uses". Pinned here: each blast-radius
graph must build and run on iree (CPU) with results identical to the
default numpy backend.

IREE runs the real MLIR compiler, so the 2s-per-file convention does not
apply; the module keeps the compile count low via one compile per distinct
graph (six total).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

NAME = "iree"

_SHAPE = (8, 16)
_N = 8 * 16

#: Cross-compiler float tolerance (same convention as the adapter tests:
#: real compiler fusion/reordering vs the pure-numpy interpreter reference).
RTOL = 1e-5
ATOL = 1e-5


def _run_iree_and_numpy(fn, specs, args):
    """Build+run ``fn`` on iree (CPU) and evaluate the numpy reference.

    Returns ``(actual, expected)`` as ``etl.Tensor``s; the device defaults
    to CPU for both backends. Callers run the parity assertions.
    """
    expected = etl.evaluate(fn, *args)  # default numpy reference
    exe = etl.build(fn, *specs, backend=NAME)
    actual = etl.run(exe, *args)
    return actual, expected


#: (id, fn, specs, args, is_float) — every graph mixes dtypes on a binary
#: elementwise / compare / select op; args match the specs.
CASES = [
    (
        "add_i32_python_scalar",
        lambda x: etl.add(x, 3),
        (etl.TensorSpec(_SHAPE, etl.int32),),
        (np.arange(_N, dtype=np.int32).reshape(_SHAPE),),
        False,
    ),
    (
        "equal_i32_python_scalar",
        lambda x: etl.equal(x, 3),
        (etl.TensorSpec(_SHAPE, etl.int32),),
        (np.arange(_N, dtype=np.int32).reshape(_SHAPE),),
        False,
    ),
    (
        "select_bool_i32_i64",
        lambda p, a, b: etl.select(p, a, b),
        (etl.TensorSpec(_SHAPE, etl.bool_),
         etl.TensorSpec(_SHAPE, etl.int32),
         etl.TensorSpec(_SHAPE, etl.int64)),
        ((np.arange(_N) % 2).astype(bool).reshape(_SHAPE),
         np.arange(_N).astype(np.int32).reshape(_SHAPE),
         np.arange(_N).astype(np.int64).reshape(_SHAPE)),
        False,
    ),
    (
        "bitwise_and_i32_i64",
        lambda a, b: etl.bitwise_and(a, b),
        (etl.TensorSpec(_SHAPE, etl.int32),
         etl.TensorSpec(_SHAPE, etl.int64)),
        (np.arange(_N).astype(np.int32).reshape(_SHAPE),
         np.arange(_N).astype(np.int64).reshape(_SHAPE)),
        False,
    ),
    (
        "add_i8_i16",
        lambda a, b: etl.add(a, b),
        (etl.TensorSpec(_SHAPE, etl.int8),
         etl.TensorSpec(_SHAPE, etl.int16)),
        ((np.arange(_N) % 100).astype(np.int8).reshape(_SHAPE),
         (np.arange(_N) % 100).astype(np.int16).reshape(_SHAPE)),
        False,
    ),
    (
        "add_f32_f64",
        lambda a, b: etl.add(a, b),
        (etl.TensorSpec(_SHAPE, etl.float32),
         etl.TensorSpec(_SHAPE, etl.float64)),
        ((np.arange(_N) % 7).astype(np.float32).reshape(_SHAPE),
         (np.arange(_N) % 7).astype(np.float64).reshape(_SHAPE)),
        True,
    ),
]


@pytest.mark.parametrize(
    "fn,specs,args,is_float",
    [(fn, specs, args, is_float) for _, fn, specs, args, is_float in CASES],
    ids=[case_id for case_id, _, _, _, _ in CASES],
)
def test_mixed_dtype_compile_run_parity(fn, specs, args, is_float):
    actual, expected = _run_iree_and_numpy(fn, specs, args)
    assert isinstance(actual, etl.Tensor)
    assert actual.dtype == expected.dtype
    if is_float:
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=RTOL, atol=ATOL
        )
    else:
        assert np.array_equal(actual.numpy(), expected.numpy())
