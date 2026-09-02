"""iree adapter hot-path validation semantics (regression suite for the
per-call hot-path reductions in ``etl/backends/adapters/iree.py`` — WS-A
round, commits 2622de9/d132b6c):

* ``IreeDevicePayload`` caches ``shape``/``dtype`` at construction: the
  cached metadata must equal the wrapped iree ``DeviceArray``'s metadata and
  stay stable for the payload's lifetime (a DeviceArray's metadata is
  immutable), because the ``IreeExecutable.run`` validation loops read
  tensor shape/dtype several times per call. Pinned here: cache
  correctness, stability across host materialization, and the unchanged
  payload protocol (``device_array`` / ``to_host`` / ``__array__`` /
  ``.device``).
* input-validation fast path: an all-static spec whose tuple EQUALS the
  runtime shape skips the per-dim walk. The observable contract is
  UNCHANGED — matching inputs pass, static-dim mismatches raise
  ``core.ShapeError`` with the canonical message (input + dim), rank
  mismatches raise ``core.ShapeError``, dtype mismatches raise
  ``core.BackendError`` ("never silently coerce"), and declared
  symbolic (``Dim``) / unchecked (``None``) dims still fall through to the
  walker. Pinned here on llvm-cpu: the fast path is observably equivalent
  for every accepting AND rejecting input class.

SEMANTICS ONLY — no timing assertions (perf lives in the GPU-guarded
same-device-loop suite). IREE runs the real MLIR compiler (seconds per
compile), so executables are cached per (fn, specs) — one compile per
distinct graph, shared across this module.
"""
import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# data (exactly representable in fp32 -> bit-exact elementwise)
# ---------------------------------------------------------------------------
XA = np.arange(24, dtype=np.float32).reshape(4, 6) + 0.25
XB = np.arange(24, dtype=np.float32).reshape(4, 6) / 2.0 + 1.0
SPECS = (
    etl.TensorSpec((4, 6), etl.float32),
    etl.TensorSpec((4, 6), etl.float32),
)


def _add(x, y):
    return etl.add(x, y)


def _np(v):
    """Tensor / ndarray -> ndarray (host copy)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    for gp, wp in zip(etl.tree_leaves(etl.tree_map(_np, got)),
                      etl.tree_leaves(etl.tree_map(_np, want))):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


_EXECUTABLES: dict = {}


def _build_exe(fn, specs):
    key = (fn, specs)
    exe = _EXECUTABLES.get(key)
    if exe is None:
        exe = etl.build(fn, *specs, backend="iree")  # llvm-cpu default
        _EXECUTABLES[key] = exe
    return exe


def _payload_of(tensor):
    """The duck-typed device payload behind a run-out tensor."""
    assert isinstance(tensor, etl.Tensor)
    assert not isinstance(tensor.data, np.ndarray)
    return tensor.data


# ---------------------------------------------------------------------------
# IreeDevicePayload metadata cache (run-out tensors on llvm-cpu)
# ---------------------------------------------------------------------------
def test_payload_metadata_cache_matches_device_array():
    exe = _build_exe(_add, SPECS)
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    payload = _payload_of(out)
    # Cached metadata must equal the wrapped DeviceArray's metadata...
    assert payload.shape == tuple(payload.device_array.shape)
    assert np.dtype(payload.dtype) == np.dtype(payload.device_array.dtype)
    # ...and the core.Tensor view is consistent with it.
    assert out.shape == payload.shape == (4, 6)
    assert out.dtype == payload.dtype == np.dtype("float32")
    assert payload.device == etl.core.Device("cpu", 0)


def test_payload_metadata_stable_across_host_materialization():
    out = etl.run(_build_exe(_add, SPECS), etl.core.Tensor(XA), etl.core.Tensor(XB))
    payload = _payload_of(out)
    shape, dtype = payload.shape, payload.dtype
    host = payload.to_host()  # lazy D2H copy must not perturb the cache
    assert payload.shape == shape and payload.dtype == dtype
    assert np.array_equal(np.asarray(payload), XA + XB)
    np.testing.assert_array_equal(host, XA + XB)


def test_payload_protocol_unchanged():
    out = etl.run(_build_exe(_add, SPECS), etl.core.Tensor(XA), etl.core.Tensor(XB))
    payload = _payload_of(out)
    assert payload.device_array is payload.device_array  # stable handle
    # np.asarray fallback materializes a correct host copy.
    np.testing.assert_array_equal(np.asarray(payload), XA + XB)
    # .numpy() path is correct end-to-end (bit-exact vs the numpy backend).
    _assert_exact(out, etl.evaluate(_add, XA, XB))


# ---------------------------------------------------------------------------
# input-validation fast path (static shapes)
# ---------------------------------------------------------------------------
def test_static_shape_fast_path_accepts_matching_inputs():
    # Exact static-shape tuple equality -> the fast path; must pass and be
    # bit-exact (this is the steady-state per-call shape).
    exe = _build_exe(_add, SPECS)
    out = etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(XB))
    _assert_exact(out, etl.evaluate(_add, XA, XB))


def test_static_shape_mismatch_raises_shape_error_naming_dim():
    exe = _build_exe(_add, SPECS)
    bad = np.zeros((4, 7), dtype=np.float32)
    with pytest.raises(etl.ShapeError, match="input 1: shape mismatch at dim 1"):
        etl.run(exe, etl.core.Tensor(XA), etl.core.Tensor(bad))


def test_rank_mismatch_raises_shape_error():
    exe = _build_exe(_add, SPECS)
    flat = np.zeros((4,), dtype=np.float32)
    with pytest.raises(etl.ShapeError, match="input 0: rank mismatch"):
        etl.run(exe, etl.core.Tensor(flat), etl.core.Tensor(XB))


def test_dtype_mismatch_raises_backenderror_never_coerces():
    exe = _build_exe(_add, SPECS)
    f64 = XA.astype(np.float64)
    with pytest.raises(etl.BackendError, match="never silently coerce"):
        etl.run(exe, etl.core.Tensor(f64), etl.core.Tensor(XB))


def test_static_shape_mismatch_raises_on_segment_executable():
    # _IreeSegmentExecutable.run carries the identical fast path; an
    # external-call graph's first segment validates the graph inputs.
    name = "ws_a_identity"

    def _kernel(x):
        return x

    etl.register_external_kernel(name, _kernel)

    @etl.defn
    def fn(x):
        return etl.external_call(name, x, result=etl.TensorSpec((4, 6), etl.float32))

    try:
        exe = _build_exe(fn, SPECS[:1])
        ok = etl.run(exe, etl.core.Tensor(XA))
        _assert_exact(ok, XA)
        bad = np.zeros((4, 7), dtype=np.float32)
        with pytest.raises(etl.ShapeError, match="shape mismatch at dim 1"):
            etl.run(exe, etl.core.Tensor(bad))
    finally:
        etl.unregister_external_kernel(name)


# ---------------------------------------------------------------------------
# symbolic / None declared dims still fall through to the walker
# (iree elementwise dynamic shapes are documented v1-supported per-op; the
# point pinned here is that the fast path never bypasses the walker for
# non-static specs — the tuples are unequal, so validation must run).
# ---------------------------------------------------------------------------
def test_symbolic_declared_dim_binds_at_run_time():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    exe = etl.build(fn, etl.TensorSpec((etl.dim("B"),), etl.float32),
                    backend="iree")
    x3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out3 = etl.run(exe, x3)
    np.testing.assert_array_equal(out3.numpy(), x3 * 2 + 1)
    x5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out5 = etl.run(exe, x5)
    np.testing.assert_array_equal(out5.numpy(), x5 * 2 + 1)


def test_none_declared_dim_unchecked_at_run_time():
    @etl.defn
    def fn(x):
        return x * 2 + 1

    exe = etl.build(fn, etl.TensorSpec((None,), etl.float32), backend="iree")
    for n in (3, 5):
        x = np.arange(n, dtype=np.float32)
        np.testing.assert_array_equal(etl.run(exe, x).numpy(), x * 2 + 1)
