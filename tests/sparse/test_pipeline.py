"""Staging-pipeline tests for ``etl.sparse``.

Covers the explicit pipeline with sparse values (``trace -> lower -> compile
-> load -> run``), the ``build``/``evaluate`` sugar (concrete sparse args
derive ``SparseTensorSpec``s via ``SparseTensorSpec.from_concrete``), leaf-
level ``bind`` (works) vs whole-sparse ``bind`` (v1 deferral), ``Graph``
save/load roundtrips, run-time static-leaf validation (dense_shape / values
dtype / FORMAT leaves), and stage-type discipline.

Contract under test: ``../etl/sparse/CONTEXT.md`` (pytree contract,
``SparseTensorSpec.from_concrete``, ``etl.pipeline.evaluate`` — see
``../etl/pipeline.py`` ~line 894). Sibling files own the deep spec
validation (``test_value.py``) and op numerics (``test_ops.py``); this file
pins the pipeline-level behavior only. Small shapes only.

Per-file helpers: ``named_coo_spec`` / ``named_csr_spec`` (leaf specs carry
``name=`` so ``etl.bind`` can target single leaves).
"""

import numpy as np
import pytest

import etl
from etl import core
from etl import sparse

from tests.sparse.conftest import (
    coo_example,
    coo_spec,
    csr_example,
    csr_spec,
    csc_example,
    csc_spec,
    dense_example,
    materialize,
    run_graph,
)

# ---------------------------------------------------------------------------
# Per-file helpers
# ---------------------------------------------------------------------------


def named_coo_spec():
    """COO spec whose tensor leaves carry names for ``etl.bind``."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((None, 2), etl.int64, name="idx"),
        etl.TensorSpec((None,), etl.float32, name="val"),
        dense_shape=(3, 4),
    )


def named_csr_spec():
    """CSR spec whose tensor leaves carry names for ``etl.bind``."""
    return sparse.SparseTensorSpec(
        etl.TensorSpec((4,), etl.int64, name="indptr"),
        etl.TensorSpec((None,), etl.int64, name="cind"),
        etl.TensorSpec((None,), etl.float32, name="cval"),
        dense_shape=(3, 4),
        format="csr",
    )


# ---------------------------------------------------------------------------
# 1. The explicit staging pipeline
# ---------------------------------------------------------------------------


def test_explicit_pipeline_negate_coo():
    """trace(coo) -> lower(numpy) -> compile -> load -> run: correct result."""
    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    lowered = etl.lower(graph)  # backend=None -> numpy
    artifact = etl.compile(lowered)
    exe = etl.load(artifact)
    out = etl.run(exe, coo_example())
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_explicit_pipeline_to_dense_csr():
    """CSR input through the explicit pipeline (conftest run_graph helper)."""
    graph = etl.trace(lambda x: sparse.to_dense(x), csr_spec())
    out = run_graph(graph, csr_example())
    np.testing.assert_allclose(materialize(out), dense_example(), rtol=1e-6)


def test_explicit_pipeline_to_dense_csc():
    """CSC input through the explicit pipeline."""
    graph = etl.trace(lambda x: sparse.to_dense(x), csc_spec())
    out = run_graph(graph, csc_example())
    np.testing.assert_allclose(materialize(out), dense_example(), rtol=1e-6)


def test_explicit_pipeline_negate_csr():
    """A computation op (negate) on a CSR input auto-converts to COO in-graph."""
    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), csr_spec())
    out = run_graph(graph, csr_example())
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_lower_defaults_to_numpy_backend():
    graph = etl.trace(lambda x: sparse.negate(x), coo_spec())
    lowered = etl.lower(graph)
    assert lowered.backend == "numpy"


# ---------------------------------------------------------------------------
# 2. Sugar: build / evaluate
# ---------------------------------------------------------------------------


def test_build_sugar_runs_with_sparse_input():
    """etl.build = trace -> lower -> compile -> load; run takes the concrete."""
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    out = etl.run(exe, coo_example())
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_evaluate_coo_arg_derives_spec_and_runs():
    """etl.evaluate derives a SparseTensorSpec from a concrete COO arg."""
    out = etl.evaluate(lambda x: sparse.to_dense(sparse.negate(x)), coo_example())
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_evaluate_csr_arg_derives_spec_and_runs():
    """etl.evaluate with a concrete CSR arg (indptr static (4,) in the spec)."""
    out = etl.evaluate(lambda x: sparse.to_dense(x), csr_example())
    np.testing.assert_allclose(materialize(out), dense_example(), rtol=1e-6)


def test_evaluate_mixed_sparse_and_dense_args():
    """One call with a concrete sparse + ndarray + core.Tensor: each arg
    derives its own spec (SparseTensorSpec for the sparse one)."""
    d = dense_example()
    ones = core.from_numpy(np.ones((3, 4), dtype=np.float32))
    out = etl.evaluate(
        lambda x, d1, d2: sparse.to_dense(sparse.multiply_dense(x, d1)) * d2,
        coo_example(),
        d,
        ones,
    )
    np.testing.assert_allclose(materialize(out), d * d * 1.0, rtol=1e-6)


def test_evaluate_sparse_result_comes_back_concrete():
    """A sparse graph result returns a CONCRETE sparse tensor (to_dense works)."""
    out = etl.evaluate(lambda x: sparse.negate(x), coo_example())
    assert sparse.is_sparse(out)
    np.testing.assert_allclose(out.to_dense(), -dense_example(), rtol=1e-6)


def test_evaluate_rejects_non_tensor_args():
    """evaluate accepts only concrete tensors / concrete sparse tensors."""
    with pytest.raises(TypeError, match="arguments that are not concrete tensors"):
        etl.evaluate(lambda x: sparse.to_dense(x), 42)


# ---------------------------------------------------------------------------
# 3. SparseTensorSpec.from_concrete (light — deep validation lives in
#    test_value.py)
# ---------------------------------------------------------------------------


def test_from_concrete_coo_children_align_leaf_for_leaf():
    """The derived spec flattens back to the concrete instance's children
    layout: TensorSpecs for the tensor leaves (nnz dim -> None), identical
    static leaves (dense_shape ints, values dtype, format str)."""
    concrete = coo_example()
    spec = sparse.SparseTensorSpec.from_concrete(concrete)
    spec_children, _ = core.flatten(spec)
    concrete_children, _ = core.flatten(concrete)
    assert len(spec_children) == len(concrete_children) == 6
    indices_spec, values_spec = spec_children[:2]
    assert isinstance(indices_spec, etl.TensorSpec)
    assert indices_spec.shape == (None, 2)
    assert indices_spec.dtype == np.dtype("int64")
    assert isinstance(values_spec, etl.TensorSpec)
    assert values_spec.shape == (None,)
    assert values_spec.dtype == np.dtype("float32")
    # Static leaves pass through unchanged.
    assert tuple(spec_children[2:]) == tuple(concrete_children[2:])


def test_from_concrete_csr_csc_indptr_stays_static():
    """CSR/CSC indptr specs stay STATIC (rows+1,)/(cols+1,) — nnz is not the
    indptr dim."""
    csr_spec_derived = sparse.SparseTensorSpec.from_concrete(csr_example())
    csr_children, _ = core.flatten(csr_spec_derived)
    assert csr_children[0].shape == (4,)  # rows+1 = 3+1
    assert csr_children[0].dtype == np.dtype("int64")
    assert csr_children[1].shape == (None,)  # indices: nnz dynamic
    assert csr_children[2].shape == (None,)  # values: nnz dynamic

    csc_spec_derived = sparse.SparseTensorSpec.from_concrete(csc_example())
    csc_children, _ = core.flatten(csc_spec_derived)
    assert csc_children[0].shape == (5,)  # cols+1 = 4+1
    assert csc_children[0].dtype == np.dtype("int64")


def test_from_concrete_rejects_non_concrete():
    """Only concrete sparse instances are accepted — ndarray and spec phase
    raise TypeError."""
    with pytest.raises(TypeError, match="expects a concrete sparse tensor"):
        sparse.SparseTensorSpec.from_concrete(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="already a spec"):
        sparse.SparseTensorSpec.from_concrete(coo_spec())


# ---------------------------------------------------------------------------
# 4. Leaf-level bind (works) — the unbound portion is the sparse node with
#    the bound leaf removed (rebuild via SparseTensor.from_parts, whose
#    dtype leaf derives from the remaining last leaf)
# ---------------------------------------------------------------------------


def test_bind_indices_leaf_coo():
    """Binding ONE tensor leaf (indices) works: run with the unbound portion
    — the sparse structure minus the bound leaf — and the bound tensor is
    supplied."""
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), named_coo_spec())
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    bound = etl.bind(exe, idx=idx)
    # Unbound portion: the COO node with the indices leaf removed. from_parts
    # computes the dtype leaf from the remaining (values) leaf -> float32,
    # matching the traced static dtype leaf.
    rest = sparse.SparseTensor.from_parts(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dense_shape=(3, 4),
        format="coo",
    )
    out = etl.run(bound, rest)
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_bind_indptr_leaf_csr():
    """The same leaf-level idiom works for a CSR node (bind the static-shaped
    indptr leaf, run with the (indices, values) rest)."""
    exe = etl.build(lambda x: sparse.to_dense(x), named_csr_spec())
    indptr = np.array([0, 1, 2, 4], dtype=np.int64)
    bound = etl.bind(exe, indptr=indptr)
    rest = sparse.SparseTensor.from_parts(
        np.array([1, 2, 0, 3], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dense_shape=(3, 4),
        format="csr",
    )
    out = etl.run(bound, rest)
    np.testing.assert_allclose(materialize(out), dense_example(), rtol=1e-6)


def test_bind_values_leaf_rest_carries_wrong_dtype_leaf():
    """Binding the VALUES leaf cannot be run via the from_parts rest: the
    rest's dtype leaf derives from its LAST (remaining) leaf — the int64
    indices — so the run-time static dtype leaf fails the traced float32.
    This is the documented static-leaf run validation doing its job (the
    dtype leaf is values-owned); the working leaf-level idioms bind a
    non-values leaf (see the two tests above). Not a bug — the contract
    "leaf-level bind works" is satisfied by those idioms."""
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), named_coo_spec())
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bound = etl.bind(exe, val=vals)
    rest = sparse.SparseTensor.from_parts(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        dense_shape=(3, 4),
        format="coo",
    )
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(bound, rest)
    msg = str(excinfo.value)
    assert "graph was specialized on dtype('float32')" in msg
    assert "at path [0][4] does not match" in msg


def test_bind_wrong_dtype_rejected_at_bind_time():
    """bind validates the bound tensor against its leaf spec immediately."""
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), named_coo_spec())
    with pytest.raises(etl.DTypeError, match=r"dtype mismatch for input at path \['val'\]"):
        etl.bind(exe, val=np.zeros(4, dtype=np.float64))


# ---------------------------------------------------------------------------
# 5. Whole-sparse bind (v1 deferral)
# ---------------------------------------------------------------------------


def test_whole_sparse_bind_fails():
    """Binding ALL tensor leaves still leaves the sparse node's STATIC leaves
    (dense_shape, dtype, format) unbound — the reduced input tree keeps the
    sparse node, so running with no arguments is a structure mismatch. The
    failure is explicit (TraceError, never silent)."""
    exe = etl.build(lambda x: sparse.to_dense(sparse.negate(x)), named_coo_spec())
    idx = np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bound = etl.bind(exe, idx=idx, val=vals)
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(bound)
    msg = str(excinfo.value)
    assert "run-time input structure does not match the unbound portion" in msg
    assert "first mismatch at pytree path ()" in msg
    assert "expected tuple of length 1, got tuple of length 0" in msg


# ---------------------------------------------------------------------------
# 6. Graph save/load roundtrip
# ---------------------------------------------------------------------------


def test_graph_save_load_roundtrip(tmp_path):
    """A traced sparse-input graph survives save/load and runs identically."""
    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    path = str(tmp_path / "sparse_negate.etlgraph")
    graph.save(path)
    loaded = etl.Graph.load(path)
    out = run_graph(loaded, coo_example())
    np.testing.assert_allclose(materialize(out), -dense_example(), rtol=1e-6)


def test_graph_save_load_roundtrip_csr(tmp_path):
    """The CSR variant (static indptr leaf) also roundtrips."""
    graph = etl.trace(lambda x: sparse.to_dense(x), csr_spec())
    path = str(tmp_path / "sparse_to_dense_csr.etlgraph")
    graph.save(path)
    loaded = etl.Graph.load(path)
    out = run_graph(loaded, csr_example())
    np.testing.assert_allclose(materialize(out), dense_example(), rtol=1e-6)


# ---------------------------------------------------------------------------
# 7. Run-time static-leaf validation (dense_shape / dtype / FORMAT leaves)
# ---------------------------------------------------------------------------


def test_run_wrong_dense_shape_fails_loudly():
    """dense_shape is a static leaf: a (3,5) concrete vs a (3,4) trace fails
    at run time with the shared static-leaf message."""
    exe = etl.load(etl.compile(etl.lower(etl.trace(
        lambda x: sparse.to_dense(x), coo_spec()
    ))))
    wrong = sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        (3, 5),
    )
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(exe, wrong)
    assert str(excinfo.value) == (
        "graph was specialized on 4 (a int); run-time argument 5 (a int) "
        "at path [0][3] does not match"
    )


def test_run_wrong_values_dtype_fails_loudly():
    """float64 values vs a float32 spec: the values leaf's TensorSpec check
    raises DTypeError (path [0][1] = the values leaf)."""
    exe = etl.load(etl.compile(etl.lower(etl.trace(
        lambda x: sparse.to_dense(x), coo_spec()
    ))))
    wrong = sparse.coo(
        np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=np.int64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        (3, 4),
    )
    with pytest.raises(etl.DTypeError) as excinfo:
        etl.run(exe, wrong)
    assert str(excinfo.value) == (
        "dtype mismatch for input at path [0][1]: spec dtype float32, "
        "got float64"
    )


def test_run_wrong_format_fails_loudly():
    """coo trace + csr run-time input: the FORMAT leaf is a static leaf, but
    the child-count difference (6 vs 7 leaves) already breaks the structure
    match — the mismatch names the sparse node."""
    exe = etl.load(etl.compile(etl.lower(etl.trace(
        lambda x: sparse.to_dense(x), coo_spec()
    ))))
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(exe, csr_example())
    msg = str(excinfo.value)
    assert "run-time input structure does not match the traced signature" in msg
    assert "first mismatch at pytree path [0]" in msg
    assert "expected SparseTensor, got SparseTensor" in msg


def test_run_dense_ndarray_where_sparse_expected():
    """A plain dense ndarray where the sparse node is expected is a
    structure mismatch naming SparseTensor."""
    exe = etl.load(etl.compile(etl.lower(etl.trace(
        lambda x: sparse.to_dense(x), coo_spec()
    ))))
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(exe, dense_example())
    msg = str(excinfo.value)
    assert "first mismatch at pytree path [0]" in msg
    assert "expected SparseTensor, got ndarray" in msg


# ---------------------------------------------------------------------------
# 8. Stage-type discipline
# ---------------------------------------------------------------------------


def test_stage_type_errors():
    """Each stage maps exactly its documented input type to its output type —
    wrong stage objects raise clear TypeErrors (mirrors pipeline_test)."""
    graph = etl.trace(lambda x: sparse.to_dense(sparse.negate(x)), coo_spec())
    lowered = etl.lower(graph)
    with pytest.raises(TypeError, match="lower expects an etl.Graph"):
        etl.lower(coo_example())
    with pytest.raises(TypeError, match="compile expects an etl.backends.LoweredProgram"):
        etl.compile(graph)
    with pytest.raises(TypeError, match="load expects an etl.backends.CompiledArtifact"):
        etl.load(graph)
    with pytest.raises(TypeError, match="run expects an etl.Executable"):
        etl.run(graph, coo_example())
    assert lowered.backend == "numpy"
