"""XLA-route argsort regression (the NSGA2-GPU backend-switch evidence, fix 71d721e).

The recommended NSGA2-on-GPU route is the xla adapter with the real PJRT
plugin — xla handles argsort at sorted-axis extent >= 32 natively: no
``stablehlo.sort`` bufferization bug, no while-init AffinityAnalysis SEGV
(see ``etl/backends/adapters/CONTEXT.md`` Known Issue 3). Pinned here: an
argsort axis >= 32 graph (the count-mode trigger class) must build+run on the
xla adapter bit-exact vs numpy whenever a user-provided PJRT plugin is
configured.

The xla adapter has no pip-installable dependency — it drives the PJRT C API
via ctypes over a user-provided ``.so`` exporting ``GetPjRtApi``. Without one
the tests skip with an actionable message (the same gate as
``tests/backends/test_mixed_dtype.py`` and ``test_adapter_xla.py``): set
``ETL_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so`` or pass
``plugin_path=...`` to the compile options.
"""

import numpy as np
import pytest

import etl


def _require_xla():
    """Skip (per-test) when no user-provided PJRT plugin is configured."""
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


def _argsort_axis1(x):
    return etl.argsort(x, axis=1)


def _argsort_axis1_desc(x):
    return etl.argsort(x, axis=1, descending=True)


_RNG = np.random.default_rng(7)
_X = _RNG.standard_normal((8, 64)).astype(np.float32)
_SPEC = etl.TensorSpec((8, 64), etl.float32)

#: (id, fn, numpy-reference) — the xla pair emission sorts the (value, iota)
#: pair ascending; descending = flip (the numpy composition). fp32 noise has
#: no ties, so the stable reference is exact.
CASES = [
    ("xla_argsort_axis64", _argsort_axis1,
     np.argsort(_X, axis=1, kind="stable")),
    ("xla_argsort_axis64_desc", _argsort_axis1_desc,
     np.flip(np.argsort(_X, axis=1, kind="stable"), axis=1)),
]


@pytest.mark.parametrize(
    "fn,ref", [(fn, ref) for _, fn, ref in CASES], ids=[c[0] for c in CASES]
)
def test_xla_argsort_axis_ge_32(fn, ref):
    """The backend-switch evidence: on xla the axis-64 argsort (the iree
    count-mode trigger class) builds and runs bit-exact vs numpy."""
    _require_xla()
    exe = etl.build(fn, _SPEC, backend="xla")
    got = np.asarray(etl.run(exe, _X).numpy())
    assert np.array_equal(got, ref), f"{got} != {ref}"
