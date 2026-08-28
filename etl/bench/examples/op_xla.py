"""XLA-targeted op conformance examples (category "op", tag "xla").

Operation-level examples deliberately staged on the XLA/PJRT adapter path
(``--backend xla``): small, static-shape graphs built ONLY from ops inside
the StableHLO exporter's coverage (elementwise, cast, select, dot incl.
batched, transpose, concatenate, reductions over static dims) — no
gather/scatter/cumsum/conv/solve/tril/triu/erf/gelu/control-flow, and no
dynamic reshape/slice/pad. Each example passes the strict conformance
defaults on the numpy backend (shared kernels → exact) and is shaped so a
real XLA plugin should compile and run it unchanged.

XLA activation (verified against ``etl/backends/adapters/xla.py`` +
``xla_util.py``, ``tests/backends/test_adapter_xla.py``,
``tests/backends/test_pjrt_ctypes_plugin.py``):

1. **The plugin.** The ``xla`` adapter drives a **user-provided PJRT C API
   plugin** (a ``.so`` exporting ``GetPjRtApi``) through pure-stdlib
   ``ctypes`` — there is NO pip-installable plugin and the adapter never
   imports jax/jaxlib. Build one from OpenXLA with ``bazel build
   //xla/pjrt/c:pjrt_c_api_cpu_plugin`` (or use any XLA build exporting
   ``GetPjRtApi``). ``--backend xla`` resolves through
   ``etl.backends.get("xla")``, which lazily activates the adapter:
   ``xla.register()`` → ``XlaBackend.check_available()`` — bindings
   integrity, plugin discovery + ``GetPjRtApi`` + the ABI version gate
   (PJRT API major 0, ``PJRT_Api.struct_size == 1144``), and a live
   ``PJRT_Client_Create``/``PJRT_Client_Destroy`` round-trip.
2. **Plugin discovery order** (``xla_util._load_plugin``, at compile AND
   load time): (a) ``options["plugin_path"]`` — the backend compile option,
   e.g. ``etl.compile(lowered, backend="xla", plugin_path="/path/to/
   pjrt_c_api_cpu_plugin.so")``; (b) the ``ETL_PJRT_PLUGIN`` environment
   variable; (c) well-known paths: ``/usr/local/lib/
   pjrt_c_api_cpu_plugin.so``, ``/usr/lib/pjrt_c_api_cpu_plugin.so``,
   ``$HOME/.local/lib/pjrt_c_api_cpu_plugin.so``, ``./pjrt_c_api_cpu_plugin.so``.
3. **Error when absent.** No plugin anywhere → ``core.BackendError`` with
   an actionable message: *"the xla adapter requires a PJRT C API plugin
   library (.so exporting GetPjRtApi) and none was found. Provide one via
   the backend compile options (etl.compile(lowered, backend='xla',
   plugin_path='/path/to/pjrt_c_api_cpu_plugin.so')) or the
   ETL_PJRT_PLUGIN environment variable; well-known paths searched: ..."*
   plus the OpenXLA bazel build instructions. The bench CLI raises this
   BEFORE any example runs (backend validation up front) and maps it to
   exit code 2.

   Exact CLI commands for these examples on xla::

       ETL_PJRT_PLUGIN=/path/to/pjrt_c_api_cpu_plugin.so \\
           python3 -m etl.bench --conformance --no-benchmark --no-torch \\
               --backend xla --examples xla

       python3 -m etl.bench --conformance --no-benchmark --no-torch \\
           --backend xla \\
           --backend-option plugin_path=/path/to/pjrt_c_api_cpu_plugin.so \\
           --examples xla

   ``--examples xla`` selects by TAG (every example here carries
   ``tags=("xla",)``; no other example uses that tag). ``--backend-option
   KEY=VALUE`` values parse JSON-first with a raw-string fallback, so
   ``plugin_path=/x/y.so`` stays a raw string. Option flow:
   ``backend_options`` → ``resolve_backend_options`` (injects
   ``target_backends=["llvm-cpu"]`` for non-numpy backends — the xla
   adapter ignores it; its ``compile`` reads only ``plugin_path``) →
   ``etl.build(..., backend=..., device=..., **opts)``, which forwards the
   dict to BOTH ``lower`` and ``compile`` (each stage uses the keys it
   understands); ``compile`` calls ``_load_plugin(options)`` where
   ``options["plugin_path"]`` takes precedence over the env var.

4. **Adapter constraints that shape these examples.** ``compile`` enforces
   a static-shape gate: every input/output spec dim must resolve to a plain
   int (a ``None`` runtime-dynamic dim or a free symbolic dim raises
   ``core.BackendError`` naming the spec). ``load`` is CPU-only (a non-cpu
   device raises ``core.BackendError``). Capabilities are conservative —
   ``dynamic_shapes=False``, ``collectives=False``, ``runtime_calls=False``,
   ``custom_blocks=False``, ``async_collectives=False`` — and the shared
   ``lower`` rejects such graphs explicitly, naming the feature. Hence:
   STATIC shapes only, and only ops within the exporter's coverage listed
   at the top of this docstring.

5. **Fake plugin reference (read-only — do not copy into bench).**
   ``tests/backends/_fake_pjrt_plugin.c`` is a TEST-ONLY minimal PJRT C API
   plugin shim satisfying the same ABI contract (struct_size 1144, API
   major 0, exports ``GetPjRtApi``): it accepts StableHLO MLIR text at
   compile time, parses the entry function's result types, and returns
   ZERO-FILLED buffers of the declared shapes/dtypes — the whole ctypes
   driver (discovery, version gate, compile, serialize, deserialize,
   execute, copy-back, error reporting) runs for real with no numerical
   meaning. Build: ``gcc -shared -fPIC -std=c99 -O1 -o plugin.so
   tests/backends/_fake_pjrt_plugin.c``. ``tests/backends/
   test_pjrt_ctypes_plugin.py`` builds it at test time via a module-scoped
   fixture (``gcc``/``cc``; skips when no C compiler) and drives the REAL
   ``XlaBackend`` lower/compile/load/run against it — including ABI-break
   variants (``-DETL_FAKE_PJRT_STRUCT_SIZE=600``,
   ``-DETL_FAKE_PJRT_MAJOR_VERSION=1``) and injected plugin failures
   (``ETL_FAKE_PJRT_FAIL_STEP`` / ``ETL_FAKE_PJRT_FAIL_MESSAGE`` env vars →
   ``core.BackendError`` carrying the plugin's message).
   ``tests/backends/test_adapter_xla.py`` is the REAL-plugin contract suite
   (module-level skip when no plugin is configured; set ``ETL_PJRT_PLUGIN``
   or pass ``plugin_path=...`` to run it for real).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .base import Example, register_all

# --- xla_elementwise_chain ---------------------------------------------------


@defn
def _xla_elementwise_chain_graph(x, mask):
    y = etl.relu(etl.add(etl.multiply(x, 2.0), 1.0))
    # f32 -> f64 -> f32 cast round-trip (exact: every f32 value is exactly
    # representable in f64), then a select on a bool mask.
    y64 = etl.cast(y, etl.float64)
    y32 = etl.cast(y64, etl.float32)
    return etl.select(mask, y32, etl.negate(y32))


def _xla_elementwise_chain_numpy(inputs):
    x, mask = inputs
    y = np.maximum(x * 2.0 + 1.0, 0.0).astype(np.float64).astype(np.float32)
    return np.where(mask, y, -y)


# --- xla_softmax -------------------------------------------------------------


@defn
def _xla_softmax_graph(x):
    m = etl.max(x, axes=-1, keepdims=True)
    e = etl.exp(etl.subtract(x, m))
    return etl.divide(e, etl.sum(e, axes=-1, keepdims=True))


def _xla_softmax_numpy(inputs):
    (x,) = inputs
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


# --- xla_mlp -----------------------------------------------------------------


@defn
def _xla_mlp_graph(x, w1, b1, w2, b2):
    h = etl.relu(etl.add(etl.dot(x, w1), b1))
    return etl.add(etl.dot(h, w2), b2)


def _xla_mlp_numpy(inputs):
    x, w1, b1, w2, b2 = inputs
    h = np.maximum(x @ w1 + b1, 0.0)
    return h @ w2 + b2


# --- xla_batched_dot ---------------------------------------------------------


@defn
def _xla_batched_dot_graph(a, b, bias):
    return etl.add(etl.dot(a, b), bias)


def _xla_batched_dot_numpy(inputs):
    a, b, bias = inputs
    return a @ b + bias


# --- xla_reductions ----------------------------------------------------------


@defn
def _xla_reductions_graph(x):
    s = etl.reduce_sum(x, axes=1)      # [4, 16]
    m = etl.reduce_mean(x, axes=1)     # [4, 16]
    mx = etl.reduce_max(x, axes=2)     # [4, 8]
    mn = etl.reduce_min(x, axes=2)     # [4, 8]
    cat = etl.concatenate([s, m, mx, mn], axis=1)  # [4, 48]
    return etl.transpose(cat, (1, 0))  # [48, 4]


def _xla_reductions_numpy(inputs):
    (x,) = inputs
    s = x.sum(axis=1)
    m = x.mean(axis=1)
    mx = x.max(axis=2)
    mn = x.min(axis=2)
    return np.concatenate([s, m, mx, mn], axis=1).T


# ---------------------------------------------------------------------------
# Registry (category "op", tag "xla")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="xla_elementwise_chain",
        description="elementwise chain: relu(x*2+1), f32->f64->f32 cast "
        "round-trip, select(mask, y, -y)",
        specs=(
            TensorSpec((8, 16), etl.float32),
            TensorSpec((8, 16), etl.bool_),
        ),
        graph=_xla_elementwise_chain_graph,
        numpy_ref=_xla_elementwise_chain_numpy,
        category="op",
        tags=("xla",),
    ),
    Example(
        name="xla_softmax",
        description="row-wise softmax from max/exp/sum primitives (static "
        "dims)",
        specs=(TensorSpec((8, 128), etl.float32),),
        graph=_xla_softmax_graph,
        numpy_ref=_xla_softmax_numpy,
        category="op",
        tags=("xla",),
    ),
    # xla_mlp tolerance override: mirrors the micro "mlp" precedent —
    # fp32 accumulation-order/FMA noise on compiler backends (the micro mlp
    # measures deterministic max_abs_error ~3.8e-05 on iree; this graph is
    # smaller, K=16/32, so noise is lower, but 1e-4 keeps the headroom
    # documented for the identical op composition). The numpy backend is
    # exact (shared np.matmul kernel) — the override never masks anything
    # on the baseline.
    Example(
        name="xla_mlp",
        description="2-layer MLP: relu(x @ w1 + b1) @ w2 + b2 (static "
        "shapes)",
        specs=(
            TensorSpec((8, 16), etl.float32),
            TensorSpec((16, 32), etl.float32),
            TensorSpec((32,), etl.float32),
            TensorSpec((32, 8), etl.float32),
            TensorSpec((8,), etl.float32),
        ),
        graph=_xla_mlp_graph,
        numpy_ref=_xla_mlp_numpy,
        tolerance=1e-4,
        category="op",
        tags=("xla",),
    ),
    Example(
        name="xla_batched_dot",
        description="batched dot [4,8,16]@[4,16,32] + broadcast bias [32]",
        specs=(
            TensorSpec((4, 8, 16), etl.float32),
            TensorSpec((4, 16, 32), etl.float32),
            TensorSpec((32,), etl.float32),
        ),
        graph=_xla_batched_dot_graph,
        numpy_ref=_xla_batched_dot_numpy,
        category="op",
        tags=("xla",),
    ),
    Example(
        name="xla_reductions",
        description="reduce_sum/mean/max/min over static axes + concatenate "
        "+ transpose",
        specs=(TensorSpec((4, 8, 16), etl.float32),),
        graph=_xla_reductions_graph,
        numpy_ref=_xla_reductions_numpy,
        category="op",
        tags=("xla",),
    ),
])
