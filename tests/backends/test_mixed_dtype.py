"""Cross-adapter mixed-dtype compile+run regression tests (fix 6aec043).

etl IR allows mixed-dtype operands on binary elementwise ops / compare /
select (the result dtype follows numpy promotion — e.g. a Python scalar
like ``3`` becomes a scalar i64 weak constant at trace time); the StableHLO
writer equalizes operand dtypes by inserting ``stablehlo.convert`` after
shape equalization (MLIR goldens pinned in ``tests/backends/
test_stablehlo.py`` section 14). Before the fix these graphs lowered into
invalid StableHLO that compilers rejected at ``etl.compile`` with "use of
value ... expects different type than prior uses". Pinned here: each
blast-radius graph must build and run on every available compiler adapter —
``iree``, ``tvm`` and ``xla`` (CPU) — with results identical to the default
numpy backend.

Per-backend gates (``_require_backend``), applied per test so one adapter's
absence never affects the others:

* ``iree``: ``pytest.importorskip("iree.compiler")`` +
  ``pytest.importorskip("iree.runtime")``.
* ``tvm``: ``pytest.importorskip("tvm")`` + ``pytest.importorskip("jaxlib")``
  (jaxlib only for its bundled MLIR bindings — the jax frontend is never
  imported, same gate as ``test_adapter_tvm.py``).
* ``xla``: requires a USER-PROVIDED PJRT C API plugin (a ``.so`` exporting
  ``GetPjRtApi``; there is no pip-installable dependency). The gate is a
  ``xla.register()`` probe that skips with an actionable message when no
  plugin is configured — set ``ETL_PJRT_PLUGIN=/path/to/
  pjrt_c_api_cpu_plugin.so`` or pass ``plugin_path=...`` to the compile
  options — and runs for real when a plugin is available (same contract as
  ``test_adapter_xla.py``, but per-test instead of module-level).

Compilers run for real, so the 2s-per-file convention does not apply; the
module keeps the compile count low via one compile per distinct graph (six
per backend, 18 worst case; 12 in this environment where xla skips).
"""

import numpy as np
import pytest

import etl

BACKENDS = ("iree", "tvm", "xla")

_SHAPE = (8, 16)
_N = 8 * 16

#: Cross-compiler float tolerance (same convention as the adapter tests:
#: real compiler fusion/reordering vs the pure-numpy interpreter reference).
RTOL = 1e-5
ATOL = 1e-5


def _require_backend(backend):
    """Skip (per-test) when ``backend``'s dependencies are unavailable."""
    if backend == "iree":
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")
    elif backend == "tvm":
        pytest.importorskip("tvm")
        pytest.importorskip("jaxlib")
    elif backend == "xla":
        from etl.backends.adapters import xla

        try:
            xla.register()
        except etl.BackendError as exc:
            pytest.skip(
                "the xla adapter requires a user-provided PJRT C API plugin "
                "(.so exporting GetPjRtApi); set "
                "ETL_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so or pass "
                f"plugin_path=... to the compile options. Details: {exc}"
            )
    else:
        raise AssertionError(f"unknown backend {backend!r}")


def _run_backend_and_numpy(fn, specs, args, backend):
    """Build+run ``fn`` on ``backend`` (CPU) and evaluate the numpy reference.

    Returns ``(actual, expected)`` as ``etl.Tensor``s; the device defaults
    to CPU for both backends. Callers run the parity assertions.
    """
    expected = etl.evaluate(fn, *args)  # default numpy reference
    exe = etl.build(fn, *specs, backend=backend)
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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn,specs,args,is_float",
    [(fn, specs, args, is_float) for _, fn, specs, args, is_float in CASES],
    ids=[case_id for case_id, _, _, _, _ in CASES],
)
def test_mixed_dtype_compile_run_parity(backend, fn, specs, args, is_float):
    _require_backend(backend)
    actual, expected = _run_backend_and_numpy(fn, specs, args, backend)
    assert isinstance(actual, etl.Tensor)
    assert actual.dtype == expected.dtype
    if is_float:
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=RTOL, atol=ATOL
        )
    else:
        assert np.array_equal(actual.numpy(), expected.numpy())
