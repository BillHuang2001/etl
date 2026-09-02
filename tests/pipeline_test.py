"""End-to-end staging-pipeline tests.

Validates the explicit staging pipeline — ``defn → trace → Graph → lower →
LoweredProgram → compile → CompiledArtifact → load → Executable → run →
Tensor`` — together with the documented shorthands (``etl.build`` /
``etl.evaluate``), ``etl.bind`` as pure argument-supply sugar, symbolic
dimensions, static-value specialization, stage-type errors, persistence
round-trips, and the distinct public pipeline types.

The contract under test is ``etl/CONTEXT.md`` ("Staging & pipeline", "Value
model", "Serialization contract"); the orchestration implementation lives in
``etl/pipeline.py`` and ``etl/backends/program.py``.
"""

from __future__ import annotations

import dataclasses
from collections import namedtuple

import numpy as np
import pytest

import etl
from etl import ir


# ---------------------------------------------------------------------------
# Shared graph definitions and structured-I/O helpers
# ---------------------------------------------------------------------------

_Inputs = namedtuple("_Inputs", ["x", "w", "b"])
_Outputs = namedtuple("_Outputs", ["out", "total"])


@dataclasses.dataclass(frozen=True)
class _ModelInputs:
    x: object
    w: object
    b: object


@dataclasses.dataclass(frozen=True)
class _ModelOutputs:
    out: object
    total: object


@etl.defn
def _linear(x, w, b):
    """A small linear layer: dot + bias + relu (three positional inputs)."""
    return etl.relu(etl.add(etl.dot(x, w), b))


@etl.defn
def _linear_tuple(t):
    out = etl.relu(etl.add(etl.dot(t[0], t[1]), t[2]))
    return (out, etl.sum(out))


@etl.defn
def _linear_list(t):
    out = etl.relu(etl.add(etl.dot(t[0], t[1]), t[2]))
    return [out, etl.sum(out)]


@etl.defn
def _linear_namedtuple(t):
    out = etl.relu(etl.add(etl.dot(t.x, t.w), t.b))
    return _Outputs(out, etl.sum(out))


@etl.defn
def _linear_dataclass(t):
    out = etl.relu(etl.add(etl.dot(t.x, t.w), t.b))
    return _ModelOutputs(out, etl.sum(out))


@etl.defn
def _linear_nested_dict(t):
    out = etl.relu(etl.add(etl.dot(t["x"], t["params"]["w"]), t["params"]["b"]))
    return {"out": out, "stats": {"total": etl.sum(out)}}


@etl.defn
def _linear_pair(x, params):
    """Linear layer over ``(x, {"w": w, "b": b})`` — a mixed tuple+dict input.

    Used by the run/bind structure-mismatch tests: the dict sits at pytree
    path ``[1]`` and its leaves at ``[1]['w']`` / ``[1]['b']``.
    """
    return etl.relu(etl.add(etl.dot(x, params["w"]), params["b"]))


def _pair_specs():
    """Specs for `_linear_pair`: ``(x, {"w": w, "b": b})`` with named leaves."""
    return (
        etl.TensorSpec((2, 3), etl.float32, name="x"),
        {
            "w": etl.TensorSpec((3, 4), etl.float32, name="w"),
            "b": etl.TensorSpec((4,), etl.float32, name="b"),
        },
    )


@etl.defn
def _scaled(x, scale, mode, nothing):
    """Specializes on three static Python values: float, str, None.

    The single ``etl.multiply`` call site keeps op source locations stable
    across specializations, so serialized-IR equality reflects the graph
    semantics alone (not which Python line built an op).
    """
    if mode == "double":
        factor = 2.0
    elif mode == "half":
        factor = 0.5
    else:
        factor = scale
    out = etl.multiply(x, factor)
    if nothing is not None:
        out = etl.add(out, 1.0)
    return out


def _ref(x, w, b):
    """Numpy reference for the linear defns (float64 computation)."""
    return np.maximum(0.0, np.asarray(x) @ np.asarray(w) + np.asarray(b))


def _check_flat_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, etl.Tensor)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)


def _check_pair_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, tuple) and len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[1].numpy(), ref.sum(), rtol=1e-6, atol=1e-6)


def _check_list_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out[1].numpy(), ref.sum(), rtol=1e-6, atol=1e-6)


def _check_namedtuple_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, _Outputs)
    np.testing.assert_allclose(out.out.numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.total.numpy(), ref.sum(), rtol=1e-6, atol=1e-6)


def _check_dataclass_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, _ModelOutputs)
    np.testing.assert_allclose(out.out.numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.total.numpy(), ref.sum(), rtol=1e-6, atol=1e-6)


def _check_dict_out(out, x, w, b):
    ref = _ref(x, w, b)
    assert isinstance(out, dict)
    assert set(out) == {"out", "stats"}
    np.testing.assert_allclose(out["out"].numpy(), ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        out["stats"]["total"].numpy(), ref.sum(), rtol=1e-6, atol=1e-6
    )


def _flat_specs(xs, ws, bs):
    return (xs, ws, bs)


def _flat_args(x, w, b):
    return (x, w, b)


def _tuple_specs(xs, ws, bs):
    return ((xs, ws, bs),)


def _tuple_args(x, w, b):
    return ((x, w, b),)


def _list_specs(xs, ws, bs):
    return ([xs, ws, bs],)


def _list_args(x, w, b):
    return ([x, w, b],)


def _namedtuple_specs(xs, ws, bs):
    return (_Inputs(xs, ws, bs),)


def _namedtuple_args(x, w, b):
    return (_Inputs(x, w, b),)


def _dataclass_specs(xs, ws, bs):
    return (_ModelInputs(xs, ws, bs),)


def _dataclass_args(x, w, b):
    return (_ModelInputs(x, w, b),)


def _dict_specs(xs, ws, bs):
    return ({"x": xs, "params": {"w": ws, "b": bs}},)


def _dict_args(x, w, b):
    return ({"x": x, "params": {"w": w, "b": b}},)


#: (id, defn, specs-builder, args-builder, output-checker) per input structure.
_PIPELINE_CASES = [
    ("flat", _linear, _flat_specs, _flat_args, _check_flat_out),
    ("tuple", _linear_tuple, _tuple_specs, _tuple_args, _check_pair_out),
    ("list", _linear_list, _list_specs, _list_args, _check_list_out),
    ("namedtuple", _linear_namedtuple, _namedtuple_specs, _namedtuple_args, _check_namedtuple_out),
    ("dataclass", _linear_dataclass, _dataclass_specs, _dataclass_args, _check_dataclass_out),
    ("nested_dict", _linear_nested_dict, _dict_specs, _dict_args, _check_dict_out),
]


# ---------------------------------------------------------------------------
# The explicit staging pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "defn, make_specs, make_args, check",
    [case[1:] for case in _PIPELINE_CASES],
    ids=[case[0] for case in _PIPELINE_CASES],
)
def test_full_pipeline(defn, make_specs, make_args, check):
    """The canonical explicit pipeline, with structured inputs and outputs."""
    xs = etl.TensorSpec((2, 3), etl.float32)
    ws = etl.TensorSpec((3, 4), etl.float32)
    bs = etl.TensorSpec((4,), etl.float32)

    # defn → trace → Graph
    graph = etl.trace(defn, *make_specs(xs, ws, bs))
    assert isinstance(graph, etl.Graph)
    graph.verify()  # explicit staging: verification is never implicit

    # → lower → LoweredProgram (default backend: numpy)
    lowered = etl.lower(graph)
    assert isinstance(lowered, etl.LoweredProgram)
    assert lowered.backend == "numpy"
    assert len(lowered.signature.input_specs) == 3
    assert tuple(lowered.signature.output_specs[0].shape) == (2, 4)
    assert lowered.signature.output_specs[0].dtype == etl.float32

    # → compile → CompiledArtifact
    artifact = etl.compile(lowered)
    assert isinstance(artifact, etl.CompiledArtifact)
    assert artifact.backend == "numpy"
    assert artifact.target == "cpu"

    # → load → Executable
    executable = etl.load(artifact)
    assert isinstance(executable, etl.Executable)
    assert executable.functions == ("main",)
    assert executable.device == etl.Device("cpu", 0)

    # → run → Tensor (structured), compared against the numpy reference
    rng = np.random.default_rng(0)
    x = etl.tensor(rng.standard_normal((2, 3)).astype(np.float32))
    w = etl.tensor(rng.standard_normal((3, 4)).astype(np.float32))
    b = etl.tensor(rng.standard_normal((4,)).astype(np.float32))
    out = etl.run(executable, *make_args(x, w, b))
    check(out, x.numpy(), w.numpy(), b.numpy())


def test_symbolic_dims():
    """Symbolic Dim/DimExpr shapes trace once and run at multiple sizes."""
    B = etl.dim("B")
    N = etl.dim("N")

    @etl.defn
    def flatten(x):
        return etl.reshape(x, (B * N,))

    graph = etl.trace(flatten, etl.TensorSpec((B, N), etl.float32))
    graph.verify()
    assert graph.tensor_specs[0].shape == (B, N)

    lowered = etl.lower(graph)
    # the output spec carries the DimExpr arithmetic shape (B * N)
    assert lowered.signature.output_specs[0].shape[0] == B * N

    executable = etl.load(etl.compile(lowered))
    for bsize, nsize in ((2, 3), (5, 2)):
        data = np.arange(bsize * nsize, dtype=np.float32).reshape(bsize, nsize)
        out = etl.run(executable, etl.tensor(data))
        assert out.shape == (bsize * nsize,)
        np.testing.assert_array_equal(out.numpy(), data.reshape(-1))

    # rank mismatches against symbolic specs fail clearly at run time
    with pytest.raises(etl.ShapeError, match="rank mismatch"):
        etl.run(executable, etl.tensor(np.zeros((2, 3, 1), dtype=np.float32)))

    # runtime-dynamic (None) dims are accepted and unchecked per rank
    @etl.defn
    def sum_rows(x):
        return etl.sum(x)

    dyn = etl.load(
        etl.compile(etl.lower(etl.trace(sum_rows, etl.TensorSpec((None, 3), etl.float32))))
    )
    for rows in (2, 4):
        data = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        out = etl.run(dyn, etl.tensor(data))
        np.testing.assert_allclose(out.numpy(), data.sum(), rtol=1e-6, atol=1e-6)


def _serialized(graph):
    return ir.serialize_module(graph.module)


def test_static_specialization():
    """Static Python values snapshot at trace time and specialize the graph."""
    spec = etl.TensorSpec((3,), etl.float32)

    g_custom2 = etl.trace(_scaled, spec, 2.0, "custom", None)
    g_custom3 = etl.trace(_scaled, spec, 3.0, "custom", None)
    g_half = etl.trace(_scaled, spec, 2.0, "half", None)

    # the static values are recorded (incl. None as a valid static value)
    assert [sv.value for sv in g_custom2.static_values] == [2.0, "custom", None]

    # where the value matters, the IR differs (a changed static value
    # IS a new graph — no hidden guards/recompile)
    assert _serialized(g_custom2) != _serialized(g_custom3)
    # static control flow specializes the graph too
    assert _serialized(g_custom2) != _serialized(g_half)

    # where the value does not matter, the IR is identical ...
    g_double3 = etl.trace(_scaled, spec, 3.0, "double", None)
    assert _serialized(g_custom2) == _serialized(g_double3)
    # ... but the recorded static values still differ
    assert [sv.value for sv in g_double3.static_values] == [3.0, "double", None]

    executable = etl.load(etl.compile(etl.lower(g_custom2)))
    x = etl.tensor(np.arange(3, dtype=np.float32))
    expected = np.arange(3, dtype=np.float32) * np.float32(2.0)

    out = etl.run(executable, x, 2.0, "custom", None)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    # run validates static values: mismatched value → clear TraceError
    with pytest.raises(etl.TraceError, match="specialized on 2.0"):
        etl.run(executable, x, 3.0, "custom", None)
    # mismatched type (int where a float was specialized)
    with pytest.raises(etl.TraceError, match="specialized on 2.0"):
        etl.run(executable, x, 2, "custom", None)
    # mismatched mode string
    with pytest.raises(etl.TraceError, match="specialized on 'custom'"):
        etl.run(executable, x, 2.0, "double", None)


# ---------------------------------------------------------------------------
# etl.bind — pure argument-supply sugar
# ---------------------------------------------------------------------------


@etl.defn
def _dot_relu(x, w):
    return etl.relu(etl.dot(x, w))


def _dot_relu_specs():
    return (
        etl.TensorSpec((2, 3), etl.float32, name="x"),
        etl.TensorSpec((3, 4), etl.float32, name="w"),
    )


def test_bind():
    """bind supplies named inputs without altering the graph or recompiling."""
    graph = etl.trace(_dot_relu, *_dot_relu_specs())
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    executable = etl.load(artifact)
    payload_before = artifact.payload

    x = etl.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    w = etl.ones((3, 4), dtype=etl.float32)
    full_out = etl.run(executable, x, w)

    # bind a subset of named inputs → BoundExecutable; run with the rest
    bound = etl.bind(executable, w=w)
    assert isinstance(bound, etl.BoundExecutable)
    assert bound.executable is executable
    assert bound.functions == executable.functions
    assert bound.device == executable.device
    partial_out = etl.run(bound, x)
    np.testing.assert_array_equal(partial_out.numpy(), full_out.numpy())

    # binding ALL inputs → run takes zero remaining arguments
    all_bound = etl.bind(executable, x=x, w=w)
    np.testing.assert_array_equal(etl.run(all_bound).numpy(), full_out.numpy())

    # binding accepts both core.Tensor values and numpy ndarrays (convenience)
    bound_ndarray = etl.bind(executable, w=w.numpy())
    np.testing.assert_array_equal(etl.run(bound_ndarray, x).numpy(), full_out.numpy())

    # bind does NOT alter the graph: same wrapper, same artifact payload,
    # same behavior on the original executable afterwards
    assert artifact.payload == payload_before
    assert lowered.payload == payload_before
    np.testing.assert_array_equal(etl.run(executable, x, w).numpy(), full_out.numpy())

    # unknown binding name → clear error listing the valid names
    with pytest.raises(etl.TraceError, match="unknown input name"):
        etl.bind(executable, z=etl.zeros((2, 3), dtype=etl.float32))

    # the unbound portion of a BoundExecutable is structure-validated
    with pytest.raises(etl.TraceError, match="does not match the unbound portion"):
        etl.run(bound, (x, w))

    # bind accepts only a plain Executable (never a Graph / BoundExecutable)
    with pytest.raises(TypeError, match="bind expects an etl.Executable"):
        etl.bind(graph, w=w)
    with pytest.raises(TypeError, match="bind expects an etl.Executable"):
        etl.bind(bound, x=x)

    # bind also works on etl.build-produced executables
    built = etl.build(_dot_relu, *_dot_relu_specs())
    bound_built = etl.bind(built, w=w)
    np.testing.assert_array_equal(etl.run(bound_built, x).numpy(), full_out.numpy())


def test_bind_validation():
    """bind validates dtype/shape/device/names against the signature."""
    graph = etl.trace(_dot_relu, *_dot_relu_specs())
    executable = etl.load(etl.compile(etl.lower(graph)))

    # wrong dtype → DTypeError
    with pytest.raises(etl.DTypeError, match="dtype mismatch"):
        etl.bind(executable, w=etl.ones((3, 4), dtype=etl.float64))
    # wrong shape → ShapeError
    with pytest.raises(etl.ShapeError, match="shape mismatch"):
        etl.bind(executable, w=etl.ones((4, 3), dtype=etl.float32))
    # wrong Python type (not a Tensor / ndarray) → TraceError
    with pytest.raises(etl.TraceError, match="must be a core.Tensor"):
        etl.bind(executable, w="not-a-tensor")

    # unknown name error is helpful: lists the valid names
    with pytest.raises(etl.TraceError, match="unknown input name"):
        etl.bind(executable, z=etl.zeros((2, 3), dtype=etl.float32))

    # device validation against the spec
    @etl.defn
    def inc(x):
        return etl.add(x, 1.0)

    device_graph = etl.trace(
        inc, etl.TensorSpec((3,), etl.float32, name="x", device=etl.Device("cpu", 1))
    )
    device_exe = etl.load(etl.compile(etl.lower(device_graph)))
    with pytest.raises(etl.DeviceError, match="device mismatch"):
        etl.bind(device_exe, x=etl.zeros((3,), dtype=etl.float32))

    # duplicate input names are ambiguous → TraceError
    amb_graph = etl.trace(
        _dot_relu,
        etl.TensorSpec((2, 3), etl.float32, name="x"),
        etl.TensorSpec((3, 4), etl.float32, name="x"),
    )
    amb_exe = etl.load(etl.compile(etl.lower(amb_graph)))
    with pytest.raises(etl.TraceError, match="ambiguous input name 'x'"):
        etl.bind(amb_exe, x=etl.zeros((2, 3), dtype=etl.float32))

    # a graph with no named inputs explains how to declare names
    unnamed_exe = etl.load(
        etl.compile(etl.lower(etl.trace(inc, etl.TensorSpec((3,), etl.float32))))
    )
    with pytest.raises(etl.TraceError, match="no named inputs"):
        etl.bind(unnamed_exe, x=etl.zeros((3,), dtype=etl.float32))


# ---------------------------------------------------------------------------
# Run/bind boundary device enforcement (explicit placement, R5)
# ---------------------------------------------------------------------------


class _FakeCudaPayload:
    """Duck-typed cuda-kind device payload (.shape/.dtype/.device/.to_host).

    A CPU-only stand-in for a real device-resident buffer: ``device`` is a
    ``core.Device`` of kind ``"cuda"`` and ``to_host()`` materializes a
    FRESH host ndarray — no backend imports, no GPU needed.
    """

    def __init__(self, shape=(3,), dtype=np.float32):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = etl.Device("cuda", 0)
        count = int(np.prod(self._shape))
        self._data = np.arange(count, dtype=dtype).reshape(self._shape)

    @property
    def shape(self):
        return self._shape

    def to_host(self):
        return self._data.copy()


def _cuda_kind_tensor(shape=(3,), dtype=np.float32):
    """A ``core.Tensor`` wrapping the fake cuda-kind payload."""
    return etl.Tensor(_FakeCudaPayload(shape=shape, dtype=dtype))


@etl.defn
def _add_one(x):
    """`x + 1.0` — the trivial single-input graph for boundary checks."""
    return etl.add(x, 1.0)


def test_run_rejects_input_on_another_device():
    """R5 run boundary: every input tensor must ALREADY be at the
    executable's device. A cuda-kind tensor fed to the cpu numpy executable
    raises DeviceError naming the input path, both devices, and the explicit
    ``t.to(...)`` remedy — no implicit device↔host transfer at run time."""
    executable = etl.build(_add_one, etl.TensorSpec((3,), etl.float32))
    assert executable.device == etl.Device("cpu", 0)

    with pytest.raises(etl.DeviceError) as excinfo:
        etl.run(executable, _cuda_kind_tensor())
    message = str(excinfo.value)
    assert "input at path [0] is on device" in message
    assert "no implicit device" in message
    assert "t.to(" in message
    assert "Device(kind='cuda', index=0)" in message
    assert "Device(kind='cpu', index=0)" in message


def test_run_device_mismatch_names_nested_path():
    """A structured input with one cpu leaf and one cuda-kind leaf fails at
    the foreign leaf, naming its pytree path (deterministic first-mismatch
    order; the cpu leaf at 'a' passes the boundary check)."""

    @etl.defn
    def f(d):
        return etl.add(d["a"], d["b"])

    executable = etl.build(
        f,
        {
            "a": etl.TensorSpec((3,), etl.float32, name="a"),
            "b": etl.TensorSpec((3,), etl.float32, name="b"),
        },
    )
    inputs = {"a": etl.ones((3,), dtype=etl.float32), "b": _cuda_kind_tensor()}
    with pytest.raises(etl.DeviceError) as excinfo:
        etl.run(executable, inputs)
    message = str(excinfo.value)
    assert "input at path [0]['b'] is on device" in message
    assert "no implicit device" in message
    assert "t.to(" in message


def test_bind_rejects_tensor_on_another_device():
    """Binding a cuda-kind tensor into a cpu executable fails fast with the
    same run-boundary DeviceError (no implicit transfer at bind time)."""
    executable = etl.build(
        _add_one, etl.TensorSpec((3,), etl.float32, name="x")
    )
    with pytest.raises(etl.DeviceError) as excinfo:
        etl.bind(executable, x=_cuda_kind_tensor())
    message = str(excinfo.value)
    assert "input at path ['x'] is on device" in message
    assert "no implicit device" in message
    assert "t.to(" in message


def test_run_accepts_cpu_inputs_positive_control():
    """Positive control: cpu-kind inputs — a core.Tensor and a raw numpy
    ndarray (auto-wrapped as a cpu:0 tensor) — run fine on the cpu
    executable; the explicit ``t.to(cpu)`` transfer unblocks the cuda-kind
    tensor."""
    executable = etl.build(_add_one, etl.TensorSpec((3,), etl.float32))
    expected = np.arange(3, dtype=np.float32) + 1.0

    out_tensor = etl.run(executable, etl.tensor(np.arange(3, dtype=np.float32)))
    np.testing.assert_array_equal(out_tensor.numpy(), expected)

    out_ndarray = etl.run(executable, np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(out_ndarray.numpy(), expected)

    # the documented remedy: move the tensor explicitly, then run
    moved = _cuda_kind_tensor().to(etl.Device("cpu", 0))
    assert moved.device == etl.Device("cpu", 0)
    out_moved = etl.run(executable, moved)
    np.testing.assert_array_equal(out_moved.numpy(), expected)


# ---------------------------------------------------------------------------
# Run-time structure mismatches name the first diverging pytree path
# ---------------------------------------------------------------------------


def _pair_inputs():
    """Concrete inputs for `_linear_pair`: ``(x, {"w": w, "b": b})``."""
    x = etl.ones((2, 3), dtype=etl.float32)
    w = etl.ones((3, 4), dtype=etl.float32)
    b = etl.ones((4,), dtype=etl.float32)
    return x, w, b


def test_run_structure_mismatch_reports_pytree_path():
    """A wrong dict key at run time keeps the traced-signature lead-in AND
    appends the first-mismatch pytree path (the diverging dict NODE; the
    key difference shows up in the expected/got node descriptions)."""
    executable = etl.build(_linear_pair, *_pair_specs())
    x, w, b = _pair_inputs()

    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(executable, x, {"w2": w, "b": b})  # 'w' replaced by 'w2'
    message = str(excinfo.value)
    assert "run-time input structure does not match the traced signature" in message
    assert "first mismatch at pytree path [1]" in message
    assert "expected dict with keys ['b', 'w']" in message
    assert "got dict with keys ['b', 'w2']" in message


def test_run_structure_mismatch_tuple_arity_reports_pytree_path():
    """A wrong tuple arity at run time keeps the traced-signature lead-in AND
    appends the first-mismatch pytree path (the diverging tuple position)."""
    xs = etl.TensorSpec((2, 3), etl.float32)
    ws = etl.TensorSpec((3, 4), etl.float32)
    bs = etl.TensorSpec((4,), etl.float32)
    executable = etl.build(_linear_tuple, *_tuple_specs(xs, ws, bs))

    x = etl.ones((2, 3), dtype=etl.float32)
    w = etl.ones((3, 4), dtype=etl.float32)
    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(executable, (x, w))  # inner tuple arity 2, traced arity 3
    message = str(excinfo.value)
    assert "run-time input structure does not match the traced signature" in message
    assert "first mismatch at pytree path [0]" in message


def test_bind_structure_mismatch_reports_pytree_path():
    """A BoundExecutable's remaining-argument structure mismatch (wrong dict
    key for the unbound portion) also names the first-mismatch pytree path."""
    executable = etl.build(_linear_pair, *_pair_specs())
    bound = etl.bind(executable, w=etl.ones((3, 4), dtype=etl.float32))
    x, _, b = _pair_inputs()

    with pytest.raises(etl.TraceError) as excinfo:
        etl.run(bound, x, {"bogus": b})  # expected key 'b' at [1], got 'bogus'
    message = str(excinfo.value)
    assert "run-time input structure does not match the unbound portion of the traced signature" in message
    assert "first mismatch at pytree path [1]" in message


# ---------------------------------------------------------------------------
# Documented shorthands: etl.build / etl.evaluate
# ---------------------------------------------------------------------------

_LINEAR_SPECS = (
    etl.TensorSpec((2, 3), etl.float32),
    etl.TensorSpec((3, 4), etl.float32),
    etl.TensorSpec((4,), etl.float32),
)


def _linear_inputs():
    x = etl.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    w = etl.ones((3, 4), dtype=etl.float32)
    b = etl.full((4,), 2.0, dtype=etl.float32)
    return x, w, b


def test_build_equals_pipeline():
    """etl.build ≡ load(compile(lower(trace(...)))) — identical results."""
    graph = etl.trace(_linear, *_LINEAR_SPECS)
    graph.verify()
    explicit = etl.load(etl.compile(etl.lower(graph)))

    built = etl.build(_linear, *_LINEAR_SPECS)
    assert isinstance(built, etl.Executable)

    x, w, b = _linear_inputs()
    explicit_out = etl.run(explicit, x, w, b)
    built_out = etl.run(built, x, w, b)
    np.testing.assert_array_equal(built_out.numpy(), explicit_out.numpy())
    _check_flat_out(built_out, x.numpy(), w.numpy(), b.numpy())

    # backend/device kwargs: by name ...
    by_name = etl.build(_linear, *_LINEAR_SPECS, backend="numpy", device="cpu")
    assert by_name.device == etl.Device("cpu", 0)
    # ... and by instance
    by_instance = etl.build(
        _linear, *_LINEAR_SPECS, backend=etl.backends.numpy_backend, device=etl.Device("cpu", 0)
    )
    np.testing.assert_array_equal(
        etl.run(by_instance, x, w, b).numpy(), built_out.numpy()
    )

    # build expects a callable/Defn — an already-traced Graph must go through
    # the explicit lower/compile/load pipeline
    with pytest.raises(TypeError, match="build expects a callable"):
        etl.build(graph)


def test_evaluate():
    """etl.evaluate ≡ derive specs (shape+dtype only) → build → run."""
    x, w, b = _linear_inputs()

    out = etl.evaluate(_linear, x, w, b)
    assert isinstance(out, etl.Tensor)
    _check_flat_out(out, x.numpy(), w.numpy(), b.numpy())

    # numpy ndarrays are accepted (wrapped via from_numpy)
    out_arrays = etl.evaluate(_linear, x.numpy(), w.numpy(), b.numpy())
    np.testing.assert_array_equal(out_arrays.numpy(), out.numpy())

    # structured args: a dict of tensors in, structured outputs out
    out_dict = etl.evaluate(_linear_nested_dict, {"x": x, "params": {"w": w, "b": b}})
    _check_dict_out(out_dict, x.numpy(), w.numpy(), b.numpy())

    # specs are derived per call: a different shape needs no re-tracing
    x2 = etl.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    w2 = etl.ones((4, 2), dtype=etl.float32)
    out2 = etl.evaluate(_linear, x2, w2, etl.zeros((2,), dtype=etl.float32))
    assert out2.shape == (3, 2)
    _check_flat_out(out2, x2.numpy(), w2.numpy(), np.zeros((2,)))

    # non-tensor (static) args must be traced explicitly — clear TypeError
    with pytest.raises(TypeError, match="arguments that are not concrete tensors"):
        etl.evaluate(_linear, x, w, 2.0)


def test_evaluate_derives_concrete_specs(monkeypatch):
    """evaluate snapshots shape+dtype only: the derived specs are concrete
    (int dims), never symbolic, and never depend on the tensor's data."""
    x = etl.tensor(np.ones((2, 3), dtype=np.float32))
    w = etl.ones((3, 4), dtype=etl.float32)
    b = etl.zeros((4,), dtype=etl.float32)

    captured = {}
    real_build = etl.pipeline.build

    def spy_build(fn, *specs, **kwargs):
        captured["fn"] = fn
        captured["specs"] = specs
        return real_build(fn, *specs, **kwargs)

    monkeypatch.setattr(etl.pipeline, "build", spy_build)
    out = etl.evaluate(_linear, x, w, b)

    assert captured["fn"] is _linear
    specs = captured["specs"]
    assert len(specs) == 3
    for spec in specs:
        assert isinstance(spec, etl.TensorSpec)
        assert all(isinstance(dim, int) for dim in spec.shape)
    assert [tuple(s.shape) for s in specs] == [(2, 3), (3, 4), (4,)]
    _check_flat_out(out, x.numpy(), w.numpy(), b.numpy())


# ---------------------------------------------------------------------------
# Stage-type errors
# ---------------------------------------------------------------------------


def test_stage_type_errors():
    """Each stage maps exactly its documented input type to its output type."""
    graph = etl.trace(_linear, *_LINEAR_SPECS)
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    x = etl.zeros((2, 3), dtype=etl.float32)
    w = etl.zeros((3, 4), dtype=etl.float32)
    b = etl.zeros((4,), dtype=etl.float32)

    # wrong stage objects are rejected with clear TypeErrors
    with pytest.raises(TypeError, match="lower expects an etl.Graph"):
        etl.lower(lowered)
    with pytest.raises(TypeError, match="compile expects an etl.backends.LoweredProgram"):
        etl.compile(graph)
    with pytest.raises(TypeError, match="load expects an etl.backends.CompiledArtifact"):
        etl.load(lowered)
    with pytest.raises(TypeError, match="run expects an etl.Executable"):
        etl.run(graph, x, w, b)
    with pytest.raises(TypeError, match="run expects an etl.Executable"):
        etl.run(lowered, x, w, b)

    # lower is pure: lowering the same graph twice yields equal programs
    again = etl.lower(graph)
    assert again.backend == lowered.backend
    assert again.payload == lowered.payload

    # unknown backend names fail explicitly
    with pytest.raises(etl.BackendError, match="unknown backend 'nope'"):
        etl.lower(graph, backend="nope")
    with pytest.raises(etl.BackendError, match="unknown backend 'nope'"):
        etl.compile(lowered, backend="nope")

    # mismatched backend at load → PersistenceError, never silently recompile
    with pytest.raises(etl.PersistenceError, match="never silently recompile"):
        etl.load(artifact, backend="stablehlo")
    with pytest.raises(TypeError, match="backend must be a registered backend name"):
        etl.load(artifact, backend=42)

    # unsupported/malformed devices fail clearly
    with pytest.raises(etl.BackendError, match="CPU devices only"):
        etl.load(artifact, device="cuda")
    with pytest.raises(etl.DeviceError, match="device must be None"):
        etl.load(artifact, device=42)


# ---------------------------------------------------------------------------
# Persistence round-trips
# ---------------------------------------------------------------------------


def test_executable_save_load(tmp_path):
    """Executable.save/load round-trips; loading never silently recompiles."""
    executable = etl.build(_linear, *_LINEAR_SPECS)
    path = tmp_path / "model.etlexe"
    executable.save(path)
    assert path.exists()

    loaded = etl.Executable.load(path)
    assert isinstance(loaded, etl.Executable)
    assert loaded.functions == executable.functions
    assert loaded.device == executable.device

    x, w, b = _linear_inputs()
    np.testing.assert_array_equal(
        etl.run(loaded, x, w, b).numpy(), etl.run(executable, x, w, b).numpy()
    )

    # a saved executable records its backend; loading with another fails
    with pytest.raises(etl.PersistenceError, match="never silently recompile"):
        etl.Executable.load(path, backend="stablehlo")

    # a saved LoweredProgram is not an executable artifact — compile first
    lowered_path = tmp_path / "lowered.etl"
    etl.lower(etl.trace(_linear, *_LINEAR_SPECS)).save(lowered_path)
    with pytest.raises(etl.PersistenceError, match="compile the lowered program first"):
        etl.Executable.load(lowered_path)


def test_pipeline_objects(tmp_path, capsys):
    """The staging types are distinct public types with documented surfaces."""
    specs = (
        etl.TensorSpec((2, 3), etl.float32, name="x"),
        etl.TensorSpec((3, 4), etl.float32, name="w"),
        etl.TensorSpec((4,), etl.float32, name="b"),
    )

    defn = _linear
    assert isinstance(defn, etl.Defn)
    graph = etl.trace(defn, *specs)
    lowered = etl.lower(graph)
    artifact = etl.compile(lowered)
    executable = etl.load(artifact)
    bound = etl.bind(executable, w=etl.ones((3, 4), dtype=etl.float32))

    stages = [defn, graph, lowered, artifact, executable, bound]
    stage_types = [type(obj) for obj in stages]
    assert len(set(stage_types)) == len(stage_types)

    # --- Graph: module + trees + specs + static values + source locations
    assert isinstance(graph.module, ir.Module)
    assert graph.tensor_specs == specs
    assert graph.static_values == ()
    assert graph.output_static_values == ()
    assert graph.source_locations  # records the trace() call site per input
    graph.verify()
    graph.print()
    assert "func @main" in capsys.readouterr().out

    # Graph save/load round-trip; the loaded graph runs identically
    graph_path = tmp_path / "g.etlgraph"
    graph.save(graph_path)
    loaded_graph = etl.Graph.load(graph_path)
    assert loaded_graph.tensor_specs == graph.tensor_specs
    assert _serialized(loaded_graph) == _serialized(graph)
    reloaded_exe = etl.load(etl.compile(etl.lower(loaded_graph)))
    x, w, b = _linear_inputs()
    np.testing.assert_array_equal(
        etl.run(reloaded_exe, x, w, b).numpy(), etl.run(executable, x, w, b).numpy()
    )

    # --- LoweredProgram: backend + signature + human-readable text + save/load
    assert lowered.backend == "numpy"
    assert lowered.signature.input_tree is graph.input_specs
    text = lowered.text()
    assert isinstance(text, str) and "func @main" in text
    lowered_path = tmp_path / "p.etl"
    lowered.save(lowered_path)
    lowered2 = etl.LoweredProgram.load(lowered_path)
    assert lowered2.backend == lowered.backend
    assert lowered2.payload == lowered.payload
    assert lowered2.text() == text

    # --- CompiledArtifact: backend + target + signature + deps + save/load
    assert artifact.backend == "numpy"
    assert artifact.target == "cpu"
    assert artifact.signature.input_specs == tuple(specs)
    assert "numpy" in artifact.runtime_dependencies
    artifact_path = tmp_path / "a.etlartifact"
    artifact.save(artifact_path)
    artifact2 = etl.CompiledArtifact.load(artifact_path)
    assert artifact2.backend == "numpy"
    assert artifact2.target == "cpu"
    assert artifact2.signature.input_specs == tuple(specs)

    # --- Executable: functions/device/signature + run via etl.run
    assert executable.functions == ("main",)
    assert executable.device == etl.Device("cpu", 0)
    assert executable.signature is artifact.signature
    assert executable.backend_executable is not None
    out = etl.run(executable, x, w, b)
    assert isinstance(out, etl.Tensor)
    _check_flat_out(out, x.numpy(), w.numpy(), b.numpy())

    # --- BoundExecutable: wraps the executable, exposes the same surface
    assert bound.executable is executable
    assert bound.functions == executable.functions
    assert bound.device == executable.device
    np.testing.assert_array_equal(etl.run(bound, x, b).numpy(), out.numpy())
