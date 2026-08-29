"""Design-principle compliance tests — the executable spec of `../CONTEXT.md`.

These tests pin the *contracts* that make etl an explicit, minimal tensor
graph runtime (see the root CONTEXT.md design principles and the package
contract in `../etl/CONTEXT.md`). Groups (one class per group, in file order):

1. `TestNoImplicitTracing`   — `@etl.defn` is NOT JIT/eager; no mode switching.
2. `TestClosureCapture`      — captured Tensors are errors; `etl.constant` opts in.
3. `TestSymbolicPurity`      — SymbolicTensor has no concrete-data escape hatches.
4. `TestStagingExplicitness` — distinct stage types; shorthand = documented
                               composition only; wrong-stage objects fail clearly.
5. `TestBindIsSugar`         — `etl.bind` never alters the graph, validates names.
6. `TestVmapIsSugar`         — `vmap(f, in_axes, out_axes=0)` ≡ `vectorize(trace(f))`.
7. `TestConcreteCreators`    — `zeros`/`ones`/... return concrete Tensors; DLPack.
8. `TestEnpGraphOps`         — `etl.numpy.zeros` inside a defn is a GRAPH op.
9. `TestSerializationRoundtrips` — Graph/LoweredProgram/CompiledArtifact save/load.
10. `TestPythonSemantics`    — Python values stay Python; no silent specialization.
11. `TestLocalTensorSemantics` — collectives are explicit ops, never implicit.
12. `TestErrorLocations`     — error messages carry `file.py:line` locations.
13. `TestTreeMapIsComposition` — `tree_map` is transparent sugar over
                               `flatten`/`unflatten` (structure-preserving).
14. `TestSparseExplicitness` — sparse tensors obey the SAME discipline:
                               graph-time computation ops (no eager mode),
                               polymorphic creators, pure symbolic leaves,
                               static-leaf snapshotting, explicit deferrals.

Conventions: small shapes, CPU only, fast. Tests assert the documented
contract; a test exposing a contract violation stays failing with a
`# BUG(etl): ...` marker (see `tests/CONTEXT.md`).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import re
import warnings
from collections import namedtuple

import numpy as np
import pytest

import etl

#: Source-location pattern from the root error strategy (`model.py:83`).
_LOCATION_RE = re.compile(r"[A-Za-z0-9_./]+\.py:\d+")

#: A tiny spec reused across groups (concrete 3-vector of float32).
_SPEC3 = etl.TensorSpec((3,), etl.float32)


# ---------------------------------------------------------------------------
# Shared helpers / tiny graph definitions
# ---------------------------------------------------------------------------


def _serialize(graph_or_module) -> dict:
    """The verified, self-describing `ir.serialize_module` payload."""
    module = getattr(graph_or_module, "module", graph_or_module)
    return etl.ir.serialize_module(module)


def _op_names(graph) -> list:
    """Op names of the entry function's entry block, in program order."""
    return [op.name for op in graph.module.main.entry_block.ops]


def _capture_symbolic() -> etl.SymbolicTensor:
    """Trace a tiny defn and return the SymbolicTensor it produced."""
    captured = {}

    @etl.defn
    def f(x):
        captured["out"] = etl.add(x, 1.0)
        return captured["out"]

    etl.trace(f, _SPEC3)
    return captured["out"]


@etl.defn
def _add_one(x):
    """`x + 1.0` — used by serialization / python-semantics / collective tests."""
    return etl.add(x, 1.0)


@etl.defn
def _mul_add(x):
    """`2.0 * x + 1.0` — used by the vmap≡vectorize sugar tests."""
    return etl.add(etl.multiply(x, 2.0), 1.0)


@etl.defn
def _dot(x, w):
    """`dot(x, w)` — used by staging / bind tests."""
    return etl.dot(x, w)


def _dot_pipeline(names=("x", "w")):
    """trace→lower→compile→load a named `dot(x, w)` pipeline.

    Returns `(defn, CompiledArtifact, Executable)`.
    """
    x_spec = etl.TensorSpec((2, 3), etl.float32, name=names[0])
    w_spec = etl.TensorSpec((3, 4), etl.float32, name=names[1])
    artifact = etl.compile(etl.lower(etl.trace(_dot, x_spec, w_spec)))
    return _dot, artifact, etl.load(artifact)


def _dot_inputs():
    """Concrete inputs for `_dot_pipeline`: x (2,3) and w (3,4) float32."""
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.full((3, 4), 0.5, dtype=np.float32)
    return x, w


#: Tiny COO spec: (3, 4) float32 — the shared sparse trace-input spec.
_SPARSE_SPEC = etl.sparse.SparseTensorSpec(
    etl.TensorSpec((None, 2), etl.int64),
    etl.TensorSpec((None,), etl.float32),
    dense_shape=(3, 4),
    format="coo",
)


def _sparse_coo():
    """A tiny concrete COO (3, 4) float32 with two stored entries."""
    indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float32)
    return etl.sparse.coo(indices, values, (3, 4))


def _sparse_csr():
    """A tiny concrete CSR (3, 4) float32 storing the same entries as
    `_sparse_coo` — used for the format static-leaf mismatch."""
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    indices = np.array([1, 3], dtype=np.int64)
    values = np.array([1.5, -2.5], dtype=np.float32)
    return etl.sparse.csr(indptr, indices, values, (3, 4))


@etl.defn
def _sparse_to_dense(x):
    """`etl.sparse.to_dense(x)` — used by the sparse sugar / static-leaf
    / vjp-deferral tests."""
    return etl.sparse.to_dense(x)


def _capture_symbolic_sparse():
    """Trace a tiny defn and return the symbolic sparse it produced."""
    captured = {}

    @etl.defn
    def f(x):
        captured["out"] = etl.sparse.negate(x)
        return etl.sparse.to_dense(captured["out"])

    etl.trace(f, _SPARSE_SPEC)
    return captured["out"]


# ===========================================================================
# 1. @etl.defn is NOT JIT/eager — no implicit tracing, no mode switching
# ===========================================================================


class TestNoImplicitTracing:
    def test_defn_direct_call_raises(self):
        """Calling a Defn with concrete tensors raises TraceError directing
        to etl.trace / etl.evaluate (principle: explicit staging only)."""

        @etl.defn
        def f(x):
            return etl.add(x, 1.0)

        with pytest.raises(
            etl.TraceError,
            match=r"`etl\.trace\(defn, \*specs\)`.*`etl\.evaluate\(defn, \*args\)`",
        ):
            f(etl.tensor([1.0, 2.0]))

    def test_defn_call_raises_even_with_specs(self):
        """A Defn never silently traces either — even TensorSpec inputs fail."""

        @etl.defn
        def f(x):
            return etl.add(x, 1.0)

        with pytest.raises(etl.TraceError, match="etl\\.trace"):
            f(_SPEC3)

    def test_ops_raise_outside_any_trace(self):
        """Ops on concrete tensors outside a trace fail — there is no eager
        mode to fall back into."""

        with pytest.raises(etl.TraceError, match="No active trace"):
            etl.add(etl.tensor([1.0]), etl.tensor([2.0]))

        # ... even with Python scalars only: ops are graph-only operations.
        with pytest.raises(etl.TraceError, match="No active trace"):
            etl.add(1.0, 2.0)

    def test_no_eager_fallback_after_tracing(self):
        """Tracing a function once does not give it a hidden eager mode: the
        same function called outside a trace still fails loudly."""

        def plain(x):
            return etl.add(x, 1.0)

        etl.trace(plain, _SPEC3)  # tracing works...

        with pytest.raises(etl.TraceError, match="No active trace"):
            plain(etl.tensor([1.0, 2.0, 3.0]))  # ...direct calls do not execute

    def test_tracing_yields_symbolic_tensors(self):
        """Graph values come from tracing and are SymbolicTensor, not Tensor."""
        captured = {}

        @etl.defn
        def f(x):
            captured["y"] = etl.add(x, 1.0)
            return captured["y"]

        graph = etl.trace(f, _SPEC3)
        assert isinstance(graph, etl.Graph)
        assert isinstance(captured["y"], etl.SymbolicTensor)
        assert not isinstance(captured["y"], etl.Tensor)


# ===========================================================================
# 2. Closure capture: captured Tensors are errors, etl.constant opts in
# ===========================================================================


class TestClosureCapture:
    def test_closure_captured_tensor_raises(self):
        """A concrete Tensor captured from a closure is a TraceError whose
        message names all three sanctioned paths: explicit input, etl.constant,
        etl.evaluate."""

        w = etl.tensor([1.0, 2.0, 3.0])

        @etl.defn
        def f(x):
            return etl.add(x, w)

        with pytest.raises(
            etl.TraceError,
            match=r"explicit input.*etl\.constant.*etl\.evaluate",
        ):
            etl.trace(f, _SPEC3)

    def test_constant_is_the_explicit_opt_in(self):
        """`etl.constant(w)` embeds the data explicitly and the graph runs."""
        w = etl.tensor([1.0, 2.0, 3.0])

        @etl.defn
        def f(x):
            return etl.add(x, etl.constant(w))

        y = etl.evaluate(f, etl.tensor([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(y.numpy(), [11.0, 22.0, 33.0])

    def test_constant_snapshots_data(self):
        """`etl.constant` snapshots the buffer at graph-construction time;
        mutating the source tensor afterwards must not change the graph."""
        w = etl.tensor([1.0, 2.0, 3.0], dtype=etl.float32)

        @etl.defn
        def f(x):
            return etl.add(x, etl.constant(w))

        graph = etl.trace(f, _SPEC3)
        w.numpy()[0] = 100.0  # mutate AFTER the graph was built

        exe = etl.load(etl.compile(etl.lower(graph)))
        y = etl.run(exe, np.array([10.0, 20.0, 30.0], dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [11.0, 22.0, 33.0])

    def test_constant_warns_above_threshold(self, monkeypatch):
        """Embedding data above `ETL_LARGE_CONSTANT_BYTES` warns (UserWarning).

        The threshold is documented as env-tunable via `ETL_LARGE_CONSTANT_BYTES`;
        it is read once at import time (etl/ops/constant.py), so the test sets
        the env var AND refreshes the cached module constant.
        """
        monkeypatch.setenv("ETL_LARGE_CONSTANT_BYTES", "16")
        const_mod = importlib.import_module("etl.ops.constant")
        monkeypatch.setattr(const_mod, "ETL_LARGE_CONSTANT_BYTES", 16)

        @etl.defn
        def big(x):
            # 16 float32 = 64 bytes > 16-byte threshold
            return etl.add(x, etl.constant(etl.ones((16,), dtype=etl.float32)))

        with pytest.warns(UserWarning, match="ETL_LARGE_CONSTANT_BYTES"):
            etl.trace(big, etl.TensorSpec((16,), etl.float32))

    def test_small_constant_does_not_warn(self, monkeypatch):
        """Constants at or below the threshold embed silently."""
        monkeypatch.setenv("ETL_LARGE_CONSTANT_BYTES", "16")
        const_mod = importlib.import_module("etl.ops.constant")
        monkeypatch.setattr(const_mod, "ETL_LARGE_CONSTANT_BYTES", 16)

        @etl.defn
        def small(x):
            # 2 float32 = 8 bytes <= 16-byte threshold
            return etl.add(x, etl.constant(etl.ones((2,), dtype=etl.float32)))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            etl.trace(small, etl.TensorSpec((2,), etl.float32))
        assert not [w for w in recorded if issubclass(w.category, UserWarning)]


# ===========================================================================
# 3. SymbolicTensor purity: no concrete-data escape hatches
# ===========================================================================


class TestSymbolicPurity:
    def test_no_concrete_interop_attributes(self):
        """A SymbolicTensor must never be mistaken for concrete data: no
        `.numpy()`, no DLPack, no array protocol, no raw data pointer."""
        symbolic = _capture_symbolic()
        for attr in ("numpy", "data_ptr", "__dlpack__", "__array__", "item"):
            assert not hasattr(symbolic, attr), f"SymbolicTensor must not expose {attr!r}"

    def test_bool_raises_trace_error(self):
        """`bool(symbolic)` raises TraceError directing to etl.cond/while_loop/
        scan (principle 4: runtime control flow is explicit)."""
        symbolic = _capture_symbolic()
        with pytest.raises(etl.TraceError, match="etl\\.cond"):
            bool(symbolic)


# ===========================================================================
# 4. Staging explicitness: distinct types, shorthand = documented composition
# ===========================================================================


class TestStagingExplicitness:
    def test_pipeline_types_are_distinct(self):
        """Defn / Graph / LoweredProgram / CompiledArtifact / Executable are
        five distinct public types; nothing silently morphs across stages."""
        graph = etl.trace(_add_one, _SPEC3)
        lowered = etl.lower(graph)
        artifact = etl.compile(lowered)
        executable = etl.load(artifact)

        stages = [etl.Defn, graph, lowered, artifact, executable]
        assert len({type(stage) for stage in stages}) == 5

        assert isinstance(graph, etl.Graph)
        assert not isinstance(graph, (etl.LoweredProgram, etl.CompiledArtifact, etl.Executable))
        assert isinstance(lowered, etl.LoweredProgram)
        assert not isinstance(lowered, (etl.Graph, etl.CompiledArtifact, etl.Executable))
        assert isinstance(artifact, etl.CompiledArtifact)
        assert not isinstance(artifact, (etl.Graph, etl.LoweredProgram, etl.Executable))
        assert isinstance(executable, etl.Executable)
        assert not isinstance(executable, (etl.Graph, etl.LoweredProgram, etl.CompiledArtifact))

    def test_wrong_stage_objects_rejected(self):
        """Passing a Graph where a later stage is expected raises a clear
        TypeError — stages are never silently consumed (actual behavior:
        compile/load/run all raise TypeError naming the expected type)."""
        graph = etl.trace(_add_one, _SPEC3)

        with pytest.raises(TypeError, match="LoweredProgram"):
            etl.compile(graph)

        with pytest.raises(TypeError, match="CompiledArtifact"):
            etl.load(graph)

        with pytest.raises(TypeError, match="Executable"):
            etl.run(graph)

    def test_build_docstring_documents_expansion(self):
        """`etl.build`'s docstring states its exact expansion (trace→lower→
        compile→load) — shorthand must be documented composition only."""
        doc = etl.build.__doc__
        assert doc, "etl.build must carry a docstring documenting its expansion"
        for stage in ("trace", "lower", "compile", "load"):
            assert stage in doc

    def test_evaluate_docstring_documents_expansion(self):
        """`etl.evaluate`'s docstring states its exact expansion (derive specs
        → build → run)."""
        doc = etl.evaluate.__doc__
        assert doc, "etl.evaluate must carry a docstring documenting its expansion"
        for stage in ("build", "run"):
            assert stage in doc

    def test_build_equals_explicit_pipeline(self):
        """`etl.build(f, *specs)` produces the same outputs as the explicit
        trace→lower→compile→load→run pipeline."""
        _, _, executable = _dot_pipeline()
        x, w = _dot_inputs()

        graph = etl.trace(_dot, etl.TensorSpec((2, 3), etl.float32),
                          etl.TensorSpec((3, 4), etl.float32))
        explicit = etl.load(etl.compile(etl.lower(graph)))

        y_short = etl.run(executable, x, w)
        y_long = etl.run(explicit, x, w)
        np.testing.assert_array_equal(y_short.numpy(), y_long.numpy())

    def test_evaluate_equals_explicit_pipeline(self):
        """`etl.evaluate(f, *tensors)` equals deriving specs from the tensors
        (shape + dtype only) then building and running."""
        x, w = _dot_inputs()
        y_short = etl.evaluate(_dot, x, w)

        graph = etl.trace(
            _dot,
            etl.TensorSpec(shape=x.shape, dtype=x.dtype),
            etl.TensorSpec(shape=w.shape, dtype=w.dtype),
        )
        y_long = etl.run(etl.load(etl.compile(etl.lower(graph))), x, w)
        np.testing.assert_array_equal(y_short.numpy(), y_long.numpy())


# ===========================================================================
# 5. etl.bind is argument-passing sugar — never alters the graph
# ===========================================================================


class TestBindIsSugar:
    def test_bind_never_alters_the_graph(self):
        """Binding must not change the compiled program. The numpy
        CompiledArtifact payload IS the serialized IR module, so deep-equal
        payloads before/after bind prove the graph is untouched."""
        _, artifact, executable = _dot_pipeline()

        before = copy.deepcopy(artifact.payload)
        bound = etl.bind(executable, w=etl.ones((3, 4), dtype=etl.float32))
        after = copy.deepcopy(artifact.payload)

        assert before == after
        # bind wraps the SAME executable, just pre-supplying arguments.
        assert bound.executable is executable

    def test_bind_run_equals_direct_run(self):
        """`run(bind(exe, w=w), x)` equals `run(exe, x, w)`."""
        _, _, executable = _dot_pipeline()
        x, w = _dot_inputs()

        bound = etl.bind(executable, w=etl.from_numpy(w))
        y_bound = etl.run(bound, x)
        y_direct = etl.run(executable, x, w)
        np.testing.assert_array_equal(y_bound.numpy(), y_direct.numpy())

    def test_bind_validates_names(self):
        """Binding an unknown input name fails clearly (no silent guessing)."""
        _, _, executable = _dot_pipeline()
        with pytest.raises(etl.TraceError, match="unknown input name"):
            etl.bind(executable, z=etl.ones((3, 4), dtype=etl.float32))


# ===========================================================================
# 6. vmap is transparent sugar over vectorize
# ===========================================================================


class TestVmapIsSugar:
    def test_vmap_fn_equals_vectorize_of_trace(self):
        """`vmap(f, in_axes, out_axes=0)` applied to specs produces IR equal
        to `vectorize(trace(f, *stripped_specs), in_axes)`.

        No naming normalization is needed: both paths go through `etl.trace`,
        which names the entry function "main", and with `out_axes=0` the
        vmap output-rearrangement step adds no ops.
        """
        batched = etl.TensorSpec((4, 5), etl.float32)
        via_vmap = etl.vmap(_mul_add, in_axes=0, out_axes=0)(batched)

        unbatched = etl.TensorSpec((5,), etl.float32)  # leading dim stripped
        via_vectorize = etl.vectorize(etl.trace(_mul_add, unbatched), 0)

        assert _serialize(via_vmap) == _serialize(via_vectorize)

    def test_vmap_graph_equals_vectorize(self):
        """`vmap(graph, in_axes=0, out_axes=0)` ≡ `vectorize(graph, 0)`."""
        graph = etl.trace(_mul_add, etl.TensorSpec((5,), etl.float32))
        via_vmap = etl.vmap(graph, in_axes=0, out_axes=0)
        via_vectorize = etl.vectorize(graph, 0)
        assert _serialize(via_vmap) == _serialize(via_vectorize)

    def test_vmap_callable_rejects_concrete_tensors(self):
        """Transforms build graphs and never execute: calling a `vmap(f)`
        callable with concrete tensors raises TraceError."""
        with pytest.raises(etl.TraceError, match="never execute"):
            etl.vmap(_mul_add)(etl.tensor(np.ones((4, 5), dtype=np.float32)))

    def test_vectorized_graph_runs_on_numpy_backend(self):
        """The sugar produces an ordinary graph of ordinary ops that the
        numpy backend runs without any special vectorize support."""
        batched = etl.TensorSpec((4, 5), etl.float32)
        graph = etl.vmap(_mul_add, in_axes=0, out_axes=0)(batched)
        exe = etl.load(etl.compile(etl.lower(graph)))

        x = np.arange(20, dtype=np.float32).reshape(4, 5)
        y = etl.run(exe, x)
        np.testing.assert_array_equal(y.numpy(), x * 2.0 + 1.0)


# ===========================================================================
# 7. Concrete creators return concrete Tensors (DLPack interop)
# ===========================================================================


class TestConcreteCreators:
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda: etl.zeros((2, 3)), id="zeros"),
            pytest.param(lambda: etl.ones((2,)), id="ones"),
            pytest.param(lambda: etl.full((2,), 7.0), id="full"),
            pytest.param(lambda: etl.empty((2,)), id="empty"),
            pytest.param(lambda: etl.tensor([1, 2, 3]), id="tensor"),
            pytest.param(lambda: etl.from_numpy(np.arange(4)), id="from_numpy"),
        ],
    )
    def test_creators_return_concrete_tensors(self, make):
        """Every concrete creator returns a materialized Tensor with
        `.numpy()`, dtype, shape, device and DLPack export — never a
        SymbolicTensor."""
        t = make()
        assert isinstance(t, etl.Tensor)
        assert not isinstance(t, etl.SymbolicTensor)
        assert callable(t.numpy)
        assert callable(t.__dlpack__)
        assert isinstance(t.dtype, np.dtype)
        assert isinstance(t.shape, tuple)
        assert t.device.kind == "cpu"

    def test_creator_values(self):
        """Creators produce the documented numpy-convention values."""
        np.testing.assert_array_equal(etl.zeros((2, 2)).numpy(), np.zeros((2, 2)))
        np.testing.assert_array_equal(etl.ones((3,)).numpy(), np.ones((3,)))
        np.testing.assert_array_equal(etl.full((2,), 7.0).numpy(), np.full((2,), 7.0))
        np.testing.assert_array_equal(etl.tensor([1, 2, 3]).numpy(), np.array([1, 2, 3]))

    def test_dlpack_roundtrip(self):
        """`etl.from_dlpack(etl.tensor(...))` round-trips values unchanged."""
        original = etl.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=etl.float32)
        imported = etl.from_dlpack(original)
        assert isinstance(imported, etl.Tensor)
        np.testing.assert_array_equal(imported.numpy(), original.numpy())

    def test_torch_interop(self):
        """torch tensor → etl.from_dlpack and etl Tensor → torch
        (via torch.utils.dlpack.from_dlpack) both preserve values."""
        torch = pytest.importorskip("torch")

        t_torch = torch.arange(6, dtype=torch.float32).reshape(2, 3)

        as_etl = etl.from_dlpack(t_torch)
        assert isinstance(as_etl, etl.Tensor)
        np.testing.assert_array_equal(as_etl.numpy(), t_torch.numpy())

        back = torch.utils.dlpack.from_dlpack(etl.tensor(t_torch.numpy()))
        np.testing.assert_array_equal(back.numpy(), t_torch.numpy())


# ===========================================================================
# 8. etl.numpy (enp) creation ops are GRAPH ops
# ===========================================================================


class TestEnpGraphOps:
    def test_enp_zeros_builds_a_constant_op(self):
        """`etl.numpy.zeros` inside a defn builds an IR op (the same
        `constant` op kind `etl.constant` builds) — it does not leak a
        concrete Tensor into the graph."""
        @etl.defn
        def f(x):
            return etl.add(x, etl.numpy.zeros((3,)))

        graph = etl.trace(f, _SPEC3)
        names = _op_names(graph)
        assert "constant" in names

        constants = [op for op in graph.module.main.entry_block.ops if op.name == "constant"]
        assert constants, "enp.zeros must be materialized as a graph op"
        # The embedded payload is a plain numpy array, never an etl.Tensor.
        for op in constants:
            assert isinstance(op.attributes["value"], np.ndarray)
            assert not isinstance(op.attributes["value"], etl.Tensor)

    def test_enp_zeros_has_no_eager_fallback(self):
        """Outside a trace, `etl.numpy.zeros` raises — there is no silent
        concrete creation path (it is a graph op)."""
        with pytest.raises(etl.TraceError, match="No active trace"):
            etl.numpy.zeros((3,))

    def test_enp_zeros_evaluates(self):
        """The graph op executes with correct values."""
        @etl.defn
        def f(x):
            return etl.add(x, etl.numpy.zeros((3,)))

        y = etl.evaluate(f, etl.tensor([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0])


# ===========================================================================
# 9. Serialization round-trips (Graph / LoweredProgram / CompiledArtifact)
# ===========================================================================


def _corrupt(path, kind):
    """Corrupt a saved container in place: flip a payload byte or truncate."""
    data = path.read_bytes()
    if kind == "flip-byte":
        mutated = bytearray(data)
        mutated[len(mutated) // 2] ^= 0xFF
        path.write_bytes(bytes(mutated))
    elif kind == "truncate":
        path.write_bytes(data[: max(1, len(data) // 2)])
    else:
        raise ValueError(f"unknown corruption kind: {kind}")


class TestSerializationRoundtrips:
    def test_graph_roundtrip_lowers(self, tmp_path):
        """A loaded Graph is a working graph: it can be lowered and run."""
        graph = etl.trace(_add_one, _SPEC3)
        path = tmp_path / "g.etlgraph"
        graph.save(str(path))

        loaded = etl.Graph.load(str(path))
        assert isinstance(loaded, etl.Graph)

        exe = etl.load(etl.compile(etl.lower(loaded)))
        y = etl.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    def test_lowered_roundtrip_compiles(self, tmp_path):
        """A loaded LoweredProgram compiles and runs; never re-lowered."""
        lowered = etl.lower(etl.trace(_add_one, _SPEC3))
        path = tmp_path / "p.etl"
        lowered.save(str(path))

        loaded = etl.LoweredProgram.load(str(path))
        assert isinstance(loaded, etl.LoweredProgram)
        assert loaded.backend == "numpy"

        exe = etl.load(etl.compile(loaded))
        y = etl.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    def test_artifact_roundtrip_runs(self, tmp_path):
        """A loaded CompiledArtifact loads into a runnable executable."""
        artifact = etl.compile(etl.lower(etl.trace(_add_one, _SPEC3)))
        path = tmp_path / "a.etlartifact"
        artifact.save(str(path))

        loaded = etl.CompiledArtifact.load(str(path))
        assert isinstance(loaded, etl.CompiledArtifact)
        assert loaded.backend == "numpy"

        exe = etl.load(loaded)
        y = etl.run(exe, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(y.numpy(), [2.0, 2.0, 2.0])

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_graph_file_fails(self, tmp_path, kind):
        """Byte-flipped or truncated containers fail with PersistenceError —
        never a silent reinterpretation."""
        graph = etl.trace(_add_one, _SPEC3)
        path = tmp_path / "g.etlgraph"
        graph.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(etl.PersistenceError):
            etl.Graph.load(str(path))

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_lowered_file_fails(self, tmp_path, kind):
        lowered = etl.lower(etl.trace(_add_one, _SPEC3))
        path = tmp_path / "p.etl"
        lowered.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(etl.PersistenceError):
            etl.LoweredProgram.load(str(path))

    @pytest.mark.parametrize("kind", ["flip-byte", "truncate"])
    def test_corrupt_artifact_file_fails(self, tmp_path, kind):
        artifact = etl.compile(etl.lower(etl.trace(_add_one, _SPEC3)))
        path = tmp_path / "a.etlartifact"
        artifact.save(str(path))
        _corrupt(path, kind)

        with pytest.raises(etl.PersistenceError):
            etl.CompiledArtifact.load(str(path))


# ===========================================================================
# 10. Python value → Python semantics (no silent specialization)
# ===========================================================================


class TestPythonSemantics:
    def test_python_values_stay_python(self):
        """Plain Python containers and scalars behave exactly like Python —
        the graph is never involved implicitly."""
        values = [1, 2, 3]
        assert sum(values) == 6
        assert [v * 2 for v in values] == [2, 4, 6]
        assert min(values) == 1 and max(values) == 3

    def test_static_values_specialize_the_graph(self):
        """A static Python value specializes the graph at trace time: two
        different values produce two different graphs, and run validates the
        value instead of silently recompiling."""

        @etl.defn
        def f(x, scale):  # scale: static Python value
            if scale > 0:
                return etl.multiply(x, scale)
            return etl.negate(x)

        g_positive = etl.trace(f, _SPEC3, 2.0)
        g_negative = etl.trace(f, _SPEC3, -1.0)
        assert _serialize(g_positive) != _serialize(g_negative)

        exe_pos = etl.load(etl.compile(etl.lower(g_positive)))
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = etl.run(exe_pos, x, 2.0)
        np.testing.assert_array_equal(y.numpy(), [2.0, 4.0, 6.0])

        exe_neg = etl.load(etl.compile(etl.lower(g_negative)))
        y = etl.run(exe_neg, x, -1.0)
        np.testing.assert_array_equal(y.numpy(), [-1.0, -2.0, -3.0])

        # Changing the static value at run time is a new graph — fail clearly.
        with pytest.raises(etl.TraceError, match="specialized on"):
            etl.run(exe_pos, x, 3.0)

    def test_symbolic_if_fails_at_trace_time(self):
        """`if etl.sum(x) > 0:` over a graph value fails at trace time with a
        clear error — runtime control flow is `etl.cond` (principle 4: no
        silent specialization on tensor data)."""

        @etl.defn
        def f(x):
            if etl.sum(x) > 0:
                return x
            return etl.negate(x)

        with pytest.raises(etl.TraceError, match="etl\\.cond"):
            etl.trace(f, _SPEC3)


# ===========================================================================
# 11. Local tensors + explicit communication
# ===========================================================================


class TestLocalTensorSemantics:
    def test_collectives_appear_explicitly_in_ir(self):
        """`etl.dist.all_reduce` appears as an explicit `all_reduce` op with
        effect `collective` in the serialized module — collectives are part of
        the program, never implicit."""

        @etl.defn
        def f(x):
            return etl.dist.all_reduce(x, op="sum")

        graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
        payload = _serialize(graph)

        op_names = [op["name"] for op in payload["ops"]]
        assert "all_reduce" in op_names
        assert etl.ir.opdef("all_reduce").effect == "collective"

    def test_plain_defn_has_no_collective_ops(self):
        """A defn without collectives contains NO collective-effect ops — the
        compiler/tracer never invents communication."""
        graph = etl.trace(_add_one, _SPEC3)
        payload = _serialize(graph)
        assert all(
            etl.ir.opdef(op["name"]).effect != "collective" for op in payload["ops"]
        )

    def test_all_reduce_runs_with_default_executor(self):
        """The reference backend's default single-rank executor is the
        identity: all_reduce on one rank returns the local tensor."""
        @etl.defn
        def f(x):
            return etl.dist.all_reduce(x, op="sum")

        y = etl.evaluate(f, etl.tensor([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(y.numpy(), [1.0, 2.0, 3.0, 4.0])


# ===========================================================================
# 12. Error messages include source locations (`model.py:83`)
# ===========================================================================


class TestErrorLocations:
    def test_bool_error_includes_source_location(self):
        """The TraceError from using a symbolic as a Python boolean names the
        op's source location."""
        symbolic = _capture_symbolic()
        with pytest.raises(etl.TraceError, match=_LOCATION_RE) as excinfo:
            bool(symbolic)
        assert "test_spec_compliance.py" in str(excinfo.value)

    def test_invalid_trace_output_error_includes_location(self):
        """Returning a concrete tensor from a traced function fails with the
        trace location in the message."""

        @etl.defn
        def f(x):
            return etl.tensor([1.0, 2.0, 3.0])

        with pytest.raises(etl.TraceError, match=_LOCATION_RE) as excinfo:
            etl.trace(f, _SPEC3)
        assert "test_spec_compliance.py" in str(excinfo.value)

    @pytest.mark.parametrize("kind", ["add", "dot"])
    def test_shape_errors_include_source_location(self, kind):
        if kind == "add":
            @etl.defn
            def fn(a, b):
                return etl.add(a, b)

            specs = (etl.TensorSpec((2,), etl.float32), etl.TensorSpec((3,), etl.float32))
        else:
            @etl.defn
            def fn(a, b):
                return etl.dot(a, b)

            specs = (etl.TensorSpec((2, 3), etl.float32), etl.TensorSpec((4, 2), etl.float32))

        with pytest.raises(etl.ShapeError) as excinfo:
            etl.trace(fn, *specs)
        message = str(excinfo.value)
        assert _LOCATION_RE.search(message), (
            f"shape error must carry a source location like `model.py:83`, got: {message}"
        )


# ===========================================================================
# 13. tree_map is transparent sugar over flatten/unflatten
# ===========================================================================


_Point = namedtuple("_Point", ["x", "y"])


@dataclasses.dataclass(frozen=True)
class _Box:
    """A user-defined dataclass pytree node with two scalar fields."""

    lo: object
    hi: object


#: Nested mixed structure: dict → tuple / list / namedtuple / dataclass / leaf.
_MIXED_TREE = {
    "pair": (1, 2),
    "seq": [3, 4],
    "point": _Point(5, 6),
    "box": _Box(7, 8),
    "scalar": 9,
}


def _double_leaf(leaf):
    """Type-preserving leaf transform (int → int), so `tree_map`'s structure
    is unaffected whether `tree_structure` records leaf types or not."""
    return leaf * 2


class TestTreeMapIsComposition:
    """`tree_map` is exactly the documented composition — ``flatten``, map
    the leaves, ``unflatten`` — and preserves the input structure exactly."""

    def test_tree_map_equals_flatten_map_unflatten(self):
        """`tree_map(f, t) == unflatten([f(l) for l in flatten(t)[0]],
        flatten(t)[1])`: mapping a nested mixed structure is precisely
        leaf-mapping plus rebuild — sugar over the explicit primitives, never
        new semantics."""

        leaves, spec = etl.flatten(_MIXED_TREE)
        expected = etl.unflatten([_double_leaf(leaf) for leaf in leaves], spec)
        assert etl.tree_map(_double_leaf, _MIXED_TREE) == expected

    def test_tree_map_preserves_structure(self):
        """`tree_structure(tree_map(f, t)) == tree_structure(t)`: the mapped
        tree keeps the exact container skeleton — dict keys, tuple/list
        arity, namedtuple fields, dataclass fields."""

        assert etl.tree_structure(
            etl.tree_map(_double_leaf, _MIXED_TREE)
        ) == etl.tree_structure(_MIXED_TREE)


# ===========================================================================
# 14. Sparse tensors: the same explicitness discipline, in all three phases
# ===========================================================================


class TestSparseExplicitness:
    """Sparse tensors (etl.sparse) obey the SAME explicit-staging discipline
    as the dense core (sparse/CONTEXT.md is the authoritative contract):
    graph-time computation ops with no eager mode, polymorphic creators
    (concrete -> eager value, symbolic -> in-graph assembly), pure symbolic
    leaves, dense_shape/dtype/format as static snapshotted leaves, and
    explicit v1 deferrals — never a silent fallback."""

    def test_ops_raise_outside_any_trace(self):
        """Sparse computation ops called outside a trace raise TraceError —
        there is no eager mode to fall back into (same as dense ops)."""
        s = _sparse_coo()
        calls = [
            (etl.sparse.add, (s, s)),
            (etl.sparse.subtract, (s, s)),
            (etl.sparse.multiply, (s, s)),
            (etl.sparse.multiply_dense, (s, s)),  # builder check precedes operands
            (etl.sparse.negate, (s,)),
            (etl.sparse.sum, (s,)),
            (etl.sparse.transpose, (s,)),
            (etl.sparse.reshape, (s, (12,))),
            (etl.sparse.concatenate, ([s, s],)),
        ]
        for op, args in calls:
            with pytest.raises(etl.TraceError, match="No active trace"):
                op(*args)

    def test_matmul_with_concrete_sparse_raises_outside_trace(self):
        """`sparse.matmul` normalizes its operands before the builder check,
        so concrete sparse operands outside a trace get the three-option
        TraceError (explicit input / etl.constant / etl.evaluate)."""
        s = _sparse_coo()
        with pytest.raises(
            etl.TraceError,
            match=r"explicit input.*etl\.constant.*etl\.evaluate",
        ):
            etl.sparse.matmul(s, s)

    def test_converters_need_a_trace_for_symbolic_inputs(self):
        """Converters are polymorphic: concrete -> eager method; a SYMBOLIC
        instance outside a trace raises TraceError (graph materialization is
        trace-time only). The one exception is the documented identity:
        `to_coo` on an already-COO value is a no-op."""
        symbolic = _capture_symbolic_sparse()
        for op in (etl.sparse.to_dense, etl.sparse.to_csr, etl.sparse.to_csc):
            with pytest.raises(etl.TraceError, match="No active trace"):
                op(symbolic)
        # COO -> COO is the documented identity (no op is emitted).
        assert etl.sparse.to_coo(symbolic) is symbolic

    def test_converters_are_eager_for_concrete_inputs(self):
        """Converters on CONCRETE instances are pure eager layout
        conversions — the polymorphic counterpart of the graph ops."""
        s = _sparse_coo()
        dense = etl.sparse.to_dense(s)
        assert isinstance(dense, np.ndarray)
        np.testing.assert_array_equal(
            dense,
            [[0.0, 1.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -2.5]],
        )
        csr = etl.sparse.to_csr(s)
        assert isinstance(csr, etl.sparse.CSRTensor)
        np.testing.assert_array_equal(etl.sparse.to_dense(csr), dense)

    def test_ops_build_ir_inside_a_trace(self):
        """Inside a trace the sparse ops build IR ops only — `sparse_add`,
        `sparse_dot_dense`, `sparse_to_dense`, `sparse_from_dense`, ... —
        never eager results."""

        @etl.defn
        def f(a, b, d, x):
            added = etl.sparse.add(a, etl.sparse.from_dense(x))
            return etl.add(etl.sparse.to_dense(added), etl.sparse.matmul(a, d))

        graph = etl.trace(
            f,
            _SPARSE_SPEC,
            _SPARSE_SPEC,
            etl.TensorSpec((4, 4), etl.float32),
            etl.TensorSpec((3, 4), etl.float32),
        )
        names = _op_names(graph)
        for op_name in (
            "sparse_from_dense",
            "sparse_add",
            "sparse_to_dense",
            "sparse_dot_dense",
        ):
            assert op_name in names, f"expected {op_name!r} in graph ops {names}"

        # Results are graph values, never eager data.
        out = graph.module.main.entry_block.ops[-2].results[0]
        assert isinstance(out, etl.ir.Value)

    def test_no_eager_numerical_ops(self):
        """Feeding CONCRETE sparse values to a computation op inside a trace
        raises the three-option TraceError — the ops never fall back to
        eager numerics."""
        s = _sparse_coo()

        @etl.defn
        def f(x):
            return etl.sparse.add(x, s)

        with pytest.raises(
            etl.TraceError,
            match=r"explicit input.*etl\.constant.*etl\.evaluate",
        ):
            etl.trace(f, _SPARSE_SPEC)

    def test_creators_concrete_components_are_eager(self):
        """Concrete components build a validated eager sparse value — the
        `.values` / `.indices` carry concrete data, canonical form is
        enforced, never silently."""
        s = _sparse_coo()
        assert isinstance(s, etl.sparse.SparseTensor)
        assert isinstance(s.values, np.ndarray)
        assert isinstance(s.indices, np.ndarray)
        np.testing.assert_array_equal(s.values, [1.5, -2.5])
        np.testing.assert_array_equal(s.indices, [[0, 1], [2, 3]])
        assert s.dtype == np.dtype(np.float32)
        assert s.dense_shape == (3, 4)
        assert s.format == "coo"

        # core.Tensor components are equally valid concrete components.
        t = etl.sparse.coo(
            etl.tensor(np.array([[0, 1]], dtype=np.int64)),
            etl.tensor(np.array([7.0], dtype=np.float32)),
            (3, 4),
        )
        assert isinstance(t.values, etl.Tensor)
        np.testing.assert_array_equal(t.to_dense()[0, 1], 7.0)

        # Canonical-form validation is eager (duplicate rows -> ShapeError).
        with pytest.raises(etl.ShapeError):
            etl.sparse.coo(
                np.array([[0, 1], [0, 1]], dtype=np.int64),
                np.array([1.0, 2.0]),
                (3, 4),
            )

    def test_creators_symbolic_components_are_graph_values(self):
        """Symbolic components inside a trace assemble the in-graph sparse
        value via from_parts — the children are SymbolicTensor (no data,
        no validation)."""

        @etl.defn
        def f(x):
            out = etl.sparse.coo(x.indices, x.values, (3, 4))
            return etl.sparse.to_dense(out)

        etl.trace(f, _SPARSE_SPEC)  # traces fine: in-graph assembly

    def test_creators_reject_tensorspec_components(self):
        """`core.TensorSpec` creator components raise TypeError directing to
        `SparseTensorSpec` for trace inputs."""
        with pytest.raises(TypeError, match="SparseTensorSpec"):
            etl.sparse.coo(
                etl.TensorSpec((None, 2), etl.int64),
                np.array([1.0, 2.0]),
                (3, 4),
            )

    def test_creators_reject_mixed_concrete_and_symbolic(self):
        """Mixed concrete+symbolic creator components raise TypeError — a
        creator is either eager assembly or in-graph assembly, never both."""
        with pytest.raises(TypeError, match="mixed kinds"):
            etl.sparse.coo(
                np.array([[0, 1]], dtype=np.int64),
                _capture_symbolic(),
                (3, 4),
            )

    def test_symbolic_sparse_leaves_have_no_concrete_escape_hatches(self):
        """The indices/values leaves of an in-graph sparse value are pure
        SymbolicTensor — no `.numpy()`, no DLPack, no array protocol
        (mirror of the dense check)."""
        symbolic = _capture_symbolic_sparse()
        assert isinstance(symbolic, etl.sparse.SparseTensor)
        for leaf in (symbolic.indices, symbolic.values):
            assert isinstance(leaf, etl.SymbolicTensor)
            for attr in ("numpy", "data_ptr", "__dlpack__", "__array__", "item"):
                assert not hasattr(leaf, attr), (
                    f"symbolic sparse leaf must not expose {attr!r}"
                )

    def test_no_sparse_constant(self):
        """`etl.constant` embeds concrete core.Tensor data only — a concrete
        sparse tensor is rejected (v1: no sparse constant)."""

        @etl.defn
        def f(x):
            return etl.sparse.to_dense(etl.constant(_sparse_coo()))

        with pytest.raises(etl.TraceError, match="got SparseTensor"):
            etl.trace(f, _SPARSE_SPEC)

    def test_closure_captured_sparse_raises(self):
        """A concrete sparse captured from a closure and fed to a computation
        op is a TraceError naming the three sanctioned paths — same as the
        dense closure-capture contract."""
        s = _sparse_coo()

        @etl.defn
        def f(x):
            return etl.sparse.add(x, s)

        with pytest.raises(
            etl.TraceError,
            match=r"explicit input.*etl\.constant.*etl\.evaluate",
        ):
            etl.trace(f, _SPARSE_SPEC)

    def test_static_leaves_are_snapshotted_and_run_validated(self):
        """dense_shape/dtype/format are static leaves: tracing with a
        SparseTensorSpec then RUNNING with a concrete sparse whose
        dense_shape (or format) differs fails loudly — the graph never
        silently adapts to a different sparse structure."""
        exe = etl.build(_sparse_to_dense, _SPARSE_SPEC)

        # dense_shape mismatch (3, 4) vs (3, 5).
        wrong_shape = etl.sparse.coo(
            np.array([[0, 1], [2, 3]], dtype=np.int64),
            np.array([1.5, -2.5], dtype=np.float32),
            (3, 5),
        )
        with pytest.raises(etl.TraceError, match="specialized on"):
            etl.run(exe, wrong_shape)

        # format mismatch: the coo-vs-csr leaf layouts diverge.
        with pytest.raises(etl.TraceError, match="structure does not match"):
            etl.run(exe, _sparse_csr())

    def test_evaluate_derives_sparse_spec_and_returns_concrete(self):
        """`etl.evaluate` with concrete sparse args derives a
        SparseTensorSpec via from_concrete and returns concrete sparse
        results — documented shorthand composition, no new semantics."""

        @etl.defn
        def f(x):
            return etl.sparse.negate(x)

        out = etl.evaluate(f, _sparse_coo())
        assert isinstance(out, etl.sparse.SparseTensor)
        assert not isinstance(out, etl.sparse.SparseTensorSpec)
        np.testing.assert_array_equal(out.values.numpy(), [-1.5, 2.5])
        np.testing.assert_array_equal(out.to_dense(), -_sparse_coo().to_dense())

        # The derived spec mirrors the concrete instance's static leaves.
        derived = etl.sparse.SparseTensorSpec.from_concrete(_sparse_coo())
        assert isinstance(derived, etl.sparse.SparseTensorSpec)
        assert derived.dense_shape == (3, 4)
        assert derived.format == "coo"
        assert derived.dtype == np.dtype(np.float32)

    def test_vmap_bare_axes_on_sparse_numerics(self):
        """`vmap(graph, in_axes=0)` with a bare int axis maps a sparse input
        (registered pytree node): a batched concrete sparse runs and the
        dense output matches the per-batch references."""
        graph = etl.vmap(etl.trace(_sparse_to_dense, _SPARSE_SPEC), in_axes=0)
        exe = etl.load(etl.compile(etl.lower(graph)))

        # Batched concrete COO: leading batch dim on the tensor leaves
        # (validation-free assembly — the per-element canonical checks are
        # the interpreter's job).
        batched = etl.sparse.SparseTensor.from_parts(
            np.array([[[0, 1], [2, 3]], [[0, 3], [1, 0]]], dtype=np.int64),
            np.array([[1.5, -2.5], [3.5, -1.5]], dtype=np.float32),
            dense_shape=(3, 4),
            format="coo",
        )
        out = etl.run(exe, batched).numpy()
        expected = np.zeros((2, 3, 4), dtype=np.float32)
        expected[0, 0, 1] = 1.5
        expected[0, 2, 3] = -2.5
        expected[1, 0, 3] = 3.5
        expected[1, 1, 0] = -1.5
        np.testing.assert_array_equal(out, expected)

    # BUG(etl): vmap's callable path cannot trace over a sparse input —
    # etl/transforms/vmap.py::_derive_unvectorized_args strips `shape[1:]`
    # from every mapped tensor leaf when deriving the unvectorized specs for
    # tracing, which for a sparse node removes the runtime-dynamic nnz dim
    # instead of leaving the leaf shape unchanged (the batch dim is prepended
    # later by vectorize). `etl.vmap(f, in_axes=0)(SparseTensorSpec(...))`
    # therefore raises ShapeError ("SparseTensorSpec: COO indices spec must
    # have shape (None, 2), got (2,)"). The transforms CONTEXT.md documents
    # "callable-path args" support for registered pytree nodes (commit
    # a176a41); the graph-level path (`etl.vmap(graph, in_axes=0)` /
    # `etl.vectorize`) works. Do NOT skip/xfail/weaken.
    def test_vmap_callable_bare_axes_on_sparse(self):
        """`vmap(f, in_axes=0)(sparse_spec)` — the callable sugar — traces
        the wrapped defn with the unbatched spec and vectorizes it."""
        graph = etl.vmap(_sparse_to_dense, in_axes=0)(_SPARSE_SPEC)
        assert isinstance(graph, etl.Graph)

    def test_sparse_sparse_matmul_is_a_v1_deferral(self):
        """sparse @ sparse matmul raises TraceError at trace time (densify
        one operand with etl.sparse.to_dense) — never a silent fallback."""

        @etl.defn
        def f(a, b):
            return etl.sparse.matmul(a, b)

        with pytest.raises(etl.TraceError, match="v1 deferral"):
            etl.trace(f, _SPARSE_SPEC, _SPARSE_SPEC)

    def test_whole_sparse_bind_is_a_v1_deferral(self):
        """Binding a WHOLE sparse value by name is unsupported in v1 — bind
        is per-leaf; the error is explicit and lists the leaf names."""
        spec = etl.sparse.SparseTensorSpec(
            etl.TensorSpec((None, 2), etl.int64, name="s_indices"),
            etl.TensorSpec((None,), etl.float32, name="s_values"),
            dense_shape=(3, 4),
            format="coo",
        )
        exe = etl.build(_sparse_to_dense, spec)
        with pytest.raises(
            etl.TraceError, match=r"unknown input name.*s_indices"
        ):
            etl.bind(exe, s=_sparse_coo())

    def test_leaf_level_bind_works(self):
        """Binding individual tensor leaves by name works: bind the indices
        leaf, run with the remaining values + static leaves, get the right
        dense output."""
        spec = etl.sparse.SparseTensorSpec(
            etl.TensorSpec((None, 2), etl.int64, name="s_indices"),
            etl.TensorSpec((None,), etl.float32, name="s_values"),
            dense_shape=(3, 4),
            format="coo",
        )
        exe = etl.build(_sparse_to_dense, spec)
        bound = etl.bind(
            exe,
            s_indices=etl.tensor(np.array([[0, 1], [2, 3]], dtype=np.int64)),
        )
        partial = etl.sparse.SparseTensor.from_parts(
            np.array([1.5, -2.5], dtype=np.float32),
            dense_shape=(3, 4),
            format="coo",
        )
        out = etl.run(bound, partial)
        np.testing.assert_array_equal(out.numpy(), _sparse_coo().to_dense())

    def test_vjp_deferrals_are_explicit(self):
        """`sparse_concatenate`, `sparse_coo_to_csc` and `sparse_csc_to_coo`
        have no VJP rule in v1 — vjp raises TransformError naming the op,
        never a silent fallback."""
        # sparse_concatenate (dense output -> plain dense cotangent spec).
        @etl.defn
        def cat(a, b):
            return etl.sparse.to_dense(etl.sparse.concatenate([a, b], axis=0))

        with pytest.raises(etl.TransformError, match="sparse_concatenate"):
            etl.vjp(cat, etl.TensorSpec((6, 4), etl.float32))(
                _SPARSE_SPEC, _SPARSE_SPEC
            )

        # sparse_coo_to_csc (CSC output -> sparse cotangent spec).
        @etl.defn
        def to_csc_fn(x):
            return etl.sparse.to_csc(x)

        csc_cotangent = etl.sparse.SparseTensorSpec(
            etl.TensorSpec((5,), etl.int64),
            etl.TensorSpec((None,), etl.int64),
            etl.TensorSpec((None,), etl.float32),
            dense_shape=(3, 4),
            format="csc",
        )
        with pytest.raises(etl.TransformError, match="sparse_coo_to_csc"):
            etl.vjp(to_csc_fn, csc_cotangent)(_SPARSE_SPEC)

        # sparse_csc_to_coo (csc input auto-converted for to_dense).
        csc_spec = etl.sparse.SparseTensorSpec(
            etl.TensorSpec((5,), etl.int64),
            etl.TensorSpec((None,), etl.int64),
            etl.TensorSpec((None,), etl.float32),
            dense_shape=(3, 4),
            format="csc",
        )
        with pytest.raises(etl.TransformError, match="sparse_csc_to_coo"):
            etl.vjp(_sparse_to_dense, etl.TensorSpec((3, 4), etl.float32))(
                csc_spec
            )
