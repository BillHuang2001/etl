"""Cross-cutting explicit-error pins for ``etl.sparse`` v1 deferrals.

Each documented deferral must fail LOUDLY with an explicit error naming the
reason — never a silent fallback. This file pins the CROSS-CUTTING deferrals
only (one focused test each); sibling files own the rest:

- ``test_value.py`` — constructor/spec validation.
- ``test_ops.py`` — op numerics + trace-time static checks + the full 16-op
  compiler-deferral matrix (capability drift + stablehlo naming for ALL 16
  ops).
- ``test_errors.py`` — three-option TraceErrors on concrete operands and
  run-time kernel canonical validation.
- ``test_backends.py`` — the 16-op capability-drift matrix and the StableHLO
  export naming matrix on the numpy/compiler-backend boundary.
- ``test_transforms.py`` — (empty at the time of writing) would own the vjp
  TransformError deferrals in the AD context; this file therefore pins all
  THREE vjp deferrals here (see ``test_vjp_deferrals_*``) so coverage does
  not depend on that file.

Pinned here (the "Known issues / v1 deferrals" list of
``../etl/sparse/CONTEXT.md`` minus what the siblings own):

1. No sparse constant in v1 (``etl.constant`` guard).
2. sparse @ sparse matmul (densify with ``etl.sparse.to_dense``).
3. Whole-sparse ``etl.bind`` (leaf-level bind works — see
   ``test_pipeline.py``).
4. Compiler-backend mechanism: ONE representative capability-drift error +
   ONE representative stablehlo export naming error (the full 16-op matrix
   lives in ``test_backends.py``).
5. The three VJP TransformError deferrals (``sparse_concatenate``,
   ``sparse_coo_to_csc``, ``sparse_csc_to_coo``).
6. ``scan`` over sparse ``xs`` and sparse ``y``-stacking.
"""

import re

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse

from tests.sparse.conftest import coo_example, coo_spec, csc_spec

# ---------------------------------------------------------------------------
# 1. No sparse constant in v1
# ---------------------------------------------------------------------------


def test_no_sparse_constant_in_graph():
    """etl.constant rejects a concrete sparse tensor inside a trace via the
    existing constant guard — only concrete core.Tensor data can be embedded."""

    def f(x):
        w = etl.constant(coo_example())
        return sparse.add(x, w)

    with pytest.raises(etl.TraceError, match=re.escape(
        "etl.constant expects a concrete core.Tensor, got SparseTensor"
    )):
        etl.trace(f, coo_spec())


# ---------------------------------------------------------------------------
# 2. sparse @ sparse matmul
# ---------------------------------------------------------------------------


def test_sparse_sparse_matmul_defers():
    """sparse @ sparse matmul is a v1 deferral: densify one operand with
    etl.sparse.to_dense."""

    def f(a, b):
        return sparse.matmul(a, b)

    with pytest.raises(etl.TraceError) as excinfo:
        etl.trace(f, coo_spec(), coo_spec())
    msg = str(excinfo.value)
    assert "v1 deferral" in msg
    assert "sparse @ sparse" in msg
    assert "etl.sparse.to_dense" in msg


# ---------------------------------------------------------------------------
# 3. Whole-sparse bind (v1 deferral)
# ---------------------------------------------------------------------------
#
# (Full leaf-level bind coverage lives in test_pipeline.py — this is the
# single cross-cutting pin the CONTEXT.md "Known issues" list promises.)


def _bound_all_leaves_executable():
    """An executable whose sparse input's leaf specs all carry names, with
    every tensor leaf bound."""
    spec = sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64, name="idx"),
        etl.TensorSpec((None,), etl.float32, name="val"),
        dense_shape=(3, 4),
    )
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), spec)
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return etl.bind(exe, idx=idx, val=vals)


def test_whole_sparse_bind_fails():
    """Whole-sparse etl.bind is not supported in v1: binding all tensor
    leaves leaves the sparse node's static leaves (dense_shape, dtype,
    format) unbound, so running with no arguments fails the unbound-portion
    structure check — an explicit TraceError, never silent. (The identical
    pin lives in test_pipeline.py — kept here because the CONTEXT.md Known
    issues list promises this failure mode.)"""
    bound = _bound_all_leaves_executable()
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(bound)
    msg = str(excinfo.value)
    assert "run-time input structure does not match the unbound portion" in msg
    assert "first mismatch at pytree path ()" in msg


# ---------------------------------------------------------------------------
# 4. Compiler-backend mechanism pins (ONE representative each — the full
#    16-op matrix lives in test_backends.py)
# ---------------------------------------------------------------------------


def test_compiler_backend_capability_drift_representative():
    """CompilerBackend.lower rejects sparse ops BEFORE any compiler runs:
    the capability-drift BackendError names the op and suggests
    etl.sparse.to_dense (no iree installation needed — the check is a pure
    capability pre-check)."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with pytest.raises(core.BackendError) as excinfo:
        etl.lower(graph, backend="iree")
    msg = str(excinfo.value)
    assert re.search("capability drift", msg)
    assert "sparse_negate" in msg
    assert "etl.sparse.to_dense" in msg


def test_stablehlo_export_names_op_representative():
    """The StableHLO exporter names the offending sparse op in its explicit
    v1 BackendError."""
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    with pytest.raises(core.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph)
    msg = str(excinfo.value)
    assert "stablehlo export: op 'sparse_negate'" in msg
    assert "is not supported in v1" in msg


# ---------------------------------------------------------------------------
# 5. The three VJP TransformError deferrals
# ---------------------------------------------------------------------------
#
# test_transforms.py was empty when this file was written, so ALL THREE
# deferrals are pinned here (grad/vjp share the vjp rules — the errors come
# from the registered rules in etl/sparse/rules.py). Each rule raises
# core.TransformError naming the op, with "v1 deferral".


def test_vjp_defers_sparse_concatenate():
    """sparse_concatenate's vjp rule is an explicit v1 deferral."""
    graph = etl.trace(
        lambda a, b: sparse.to_dense(sparse.concatenate([a, b], axis=0)),
        coo_spec(),
        coo_spec(),
    )
    with pytest.raises(core.TransformError) as excinfo:
        etl.vjp(graph, cotangents=(core.TensorSpec((6, 4), etl.float32),))
    msg = str(excinfo.value)
    assert "v1 deferral" in msg
    assert "sparse_concatenate" in msg


def test_vjp_defers_sparse_coo_to_csc():
    """sparse_coo_to_csc's vjp rule is an explicit v1 deferral (its CSC
    result feeds the graph output directly — the cotangent tree mirrors the
    sparse output node's children layout)."""
    graph = etl.trace(lambda x: sparse.to_csc(x), coo_spec())
    cotangents = sparse.SparseTensorSpec(
        etl.TensorSpec((5,), etl.int64),  # CSC indptr: cols+1 = 4+1
        etl.TensorSpec((None,), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(3, 4),
        format="csc",
    )
    with pytest.raises(core.TransformError) as excinfo:
        etl.vjp(graph, cotangents=cotangents)
    msg = str(excinfo.value)
    assert "v1 deferral" in msg
    assert "sparse_coo_to_csc" in msg


def test_vjp_defers_sparse_csc_to_coo():
    """sparse_csc_to_coo's vjp rule is an explicit v1 deferral (the COO
    result feeds the graph output directly)."""
    graph = etl.trace(lambda x: sparse.to_coo(x), csc_spec())
    cotangents = sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64),
        etl.TensorSpec((None,), etl.float32),
        dense_shape=(3, 4),
        format="coo",
    )
    with pytest.raises(core.TransformError) as excinfo:
        etl.vjp(graph, cotangents=cotangents)
    msg = str(excinfo.value)
    assert "v1 deferral" in msg
    assert "sparse_csc_to_coo" in msg


# ---------------------------------------------------------------------------
# 6. scan over sparse values
# ---------------------------------------------------------------------------


def test_scan_over_sparse_xs_defers():
    """scan's xs leaves must be SymbolicTensors; the sparse node's static
    leaves (dense_shape ints) trip the check with an explicit TraceError."""

    def body(carry, x):
        return carry, x

    with pytest.raises(etl.TraceError) as excinfo:
        etl.trace(lambda x: etl.scan(body, 0, x), coo_spec())
    assert str(excinfo.value) == (
        "etl.scan: xs leaf 2 must be a core.SymbolicTensor, got int"
    )


def test_scan_sparse_y_stacking_defers():
    """scan over sparse y-outputs is also a v1 deferral: the sparse node's
    static leaves make the y-output check fail with an explicit TraceError
    (documented in etl/sparse/CONTEXT.md "scan over sparse xs / sparse
    y-stacking is a v1 deferral (explicit TraceError)")."""

    def body(carry, x):
        return carry, sparse.from_dense(x)

    with pytest.raises(etl.TraceError) as excinfo:
        etl.trace(
            lambda x: etl.scan(body, 0, x), etl.TensorSpec((4,), etl.float32)
        )
    msg = str(excinfo.value)
    assert "etl.scan: f's y outputs must be SymbolicTensors" in msg
    assert "cannot be traced" in msg
