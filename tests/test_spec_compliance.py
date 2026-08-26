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

Conventions: small shapes, CPU only, fast. Tests assert the documented
contract; where the implementation contradicts it, the test stays failing and
is marked with a `# BUG(etl): ...` comment (see group 12).
"""

from __future__ import annotations

import copy
import importlib
import re
import warnings

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

    # BUG(etl): shape errors raised during op shape inference do NOT include
    # the source location, although the failing op carries one. The root
    # error strategy (CONTEXT.md) requires error messages to include a
    # location like `model.py:83` whenever a graph location exists. Minimal
    # repro:
    #     @etl.defn
    #     def f(a, b): return etl.add(a, b)
    #     etl.trace(f, TensorSpec((2,), f32), TensorSpec((3,), f32))
    #     -> ShapeError("cannot broadcast incompatible dims 2 and 3")  # no file:line
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
