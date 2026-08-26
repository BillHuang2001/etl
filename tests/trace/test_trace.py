"""Contract tests for the core tracer `etl.trace(fn_or_defn, *specs) -> Graph`.

Covers the 7-step trace algorithm documented in `etl/trace/CONTEXT.md`:
flat tracing, symbolic/`None` dims, structured (pytree) I/O, static-value
snapshotting/specialization, closure-capture rules, plain-fn acceptance,
zero-IO graphs, invalid input/output leaves, output-static re-insertion,
SymbolicTensor purity, and no-caching freshness.

The etl package is the system under test (read-only from here). Only the
container *skeleton* of `Graph.input_specs` is asserted — trace trees record
TensorSpec/SymbolicTensor leaves via private marker types (`_TensorSpecLeaf` /
`_SymbolicLeaf`) by design (see the "Tree leaf markers" note in the trace
CONTEXT.md).
"""

import dataclasses
import enum
from typing import NamedTuple

import numpy as np
import pytest

import etl


# --- module-level pytree containers used as structured inputs -----------------


class PairNT(NamedTuple):
    a: object
    b: object


@dataclasses.dataclass
class PairDC:
    w: object
    z: object


class Mode(enum.Enum):
    A = 1
    B = 2


# --- helpers ------------------------------------------------------------------


def _op_names(graph: etl.Graph) -> list:
    """Op names of the entry function body, in program order."""
    return [op.name for op in graph.module.main.entry_block.ops]


def _skeleton(spec):
    """Reduce a `etl.core.TreeSpec` to its container skeleton.

    Leaf types are deliberately ignored (trace-time leaf types are private
    markers); container nodes keep (type, node_data, children).
    """
    if not spec.children:
        return "leaf"
    node_data = spec.node_data
    if node_data is not None:
        node_data = tuple(node_data)
    return (spec.type, node_data, tuple(_skeleton(child) for child in spec.children))


def _symbolic_tensor() -> etl.SymbolicTensor:
    """A fresh SymbolicTensor wrapping a traced graph's block argument."""
    graph = etl.trace(lambda x: x, etl.TensorSpec((3,), etl.float32))
    arg = graph.module.main.entry_block.arguments[0]
    return etl.SymbolicTensor(value=arg, dtype=etl.float32, shape=(3,))


# --- 1. basic trace ------------------------------------------------------------


def test_basic_trace_flat_fn(run_graph, as_numpy):
    def f(x):
        return etl.add(x, etl.constant(etl.tensor(1.0, dtype=etl.float32)))

    spec = etl.TensorSpec((3,), etl.float32)
    g = etl.trace(f, spec)

    assert isinstance(g, etl.Graph)
    assert g.tensor_specs == (spec,)
    assert isinstance(g.module, etl.ir.Module)
    main = g.module.main
    assert isinstance(main, etl.ir.Function)
    assert main.name == "main"
    assert len(main.entry_block.arguments) == 1
    assert _op_names(g) == ["constant", "add", "return"]

    g.verify()

    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_array_equal(as_numpy(result), np.ones(3, np.float32) + 1)


# --- 2. symbolic and runtime-dynamic dims --------------------------------------


def test_symbolic_dims_preserved_and_accepted_at_run(run_graph, as_numpy):
    b_dim, h_dim = etl.Dim("B"), etl.Dim("H")
    spec = etl.TensorSpec((b_dim, h_dim), etl.float32)

    def f(x):
        return etl.add(x, x)

    g = etl.trace(f, spec)
    traced_shape = g.tensor_specs[0].shape
    assert traced_shape[0] is b_dim
    assert traced_shape[1] is h_dim

    g.verify()
    result = run_graph(g, np.ones((2, 3), np.float32))
    assert as_numpy(result).shape == (2, 3)


def test_none_dims_runtime_dynamic(run_graph, as_numpy):
    def f(x):
        return etl.add(x, x)

    g = etl.trace(f, etl.TensorSpec((None, 3), etl.float32))
    for leading in (1, 5):
        result = run_graph(g, np.zeros((leading, 3), np.float32))
        assert as_numpy(result).shape == (leading, 3)


@pytest.mark.parametrize("bad_shape", [(5,), (5, 3, 2)], ids=["rank-1", "rank-3"])
def test_none_dim_rank_mismatch_raises(run_graph, bad_shape):
    def f(x):
        return etl.add(x, x)

    g = etl.trace(f, etl.TensorSpec((None, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="rank mismatch"):
        run_graph(g, np.zeros(bad_shape, np.float32))


# --- 3. structured (pytree) inputs ---------------------------------------------


def test_structured_inputs_round_trip(run_graph, as_numpy):
    x_spec = etl.TensorSpec((2,), etl.float32)
    y = {
        "a": etl.TensorSpec((2,), etl.float32),
        "dc": PairDC(etl.TensorSpec((3,), etl.float32), etl.TensorSpec((4,), etl.float32)),
        "l": [etl.TensorSpec((8,), etl.float32), etl.TensorSpec((9,), etl.float32)],
        "n": PairNT(etl.TensorSpec((3,), etl.float32), etl.TensorSpec((2,), etl.float32)),
        "t": (
            etl.TensorSpec((5,), etl.float32),
            (etl.TensorSpec((6,), etl.float32), etl.TensorSpec((7,), etl.float32)),
        ),
    }

    def f(x, y):
        return (etl.add(x, y["a"]), {"neg": etl.negate(y["n"].a)})

    g = etl.trace(f, x_spec, y)

    # The recorded input tree matches the container skeleton of the traced
    # structure (leaf types are private markers by design — skeleton only).
    mirror = {
        "a": None,
        "dc": PairDC(None, None),
        "l": [None, None],
        "n": PairNT(None, None),
        "t": (None, (None, None)),
    }
    _, mirror_tree = etl.core.flatten((None, mirror))
    assert g.input_specs.type is tuple  # the specs tuple is the root pytree
    assert _skeleton(g.input_specs) == _skeleton(mirror_tree)

    # tensor_specs = flat pre-order leaf order, dict keys sorted
    # (x, a, dc.w, dc.z, l[0], l[1], n.a, n.b, t[0], t[1][0], t[1][1]).
    assert [s.shape for s in g.tensor_specs] == [
        (2,),
        (2,),
        (3,),
        (4,),
        (8,),
        (9,),
        (3,),
        (2,),
        (5,),
        (6,),
        (7,),
    ]

    # One recorded source location per tensor input.
    assert len(g.source_locations) == 11
    assert all(
        isinstance(location, etl.ir.Location)
        for location in g.source_locations.values()
    )

    g.verify()

    yin = {
        "a": np.ones(2, np.float32),
        "dc": PairDC(np.ones(3, np.float32), np.ones(4, np.float32)),
        "l": [np.ones(8, np.float32), np.ones(9, np.float32)],
        "n": PairNT(np.full(3, 5.0, np.float32), np.ones(2, np.float32)),
        "t": (
            np.ones(5, np.float32),
            (np.ones(6, np.float32), np.ones(7, np.float32)),
        ),
    }
    summed, rest = run_graph(g, np.ones(2, np.float32), yin)
    np.testing.assert_allclose(as_numpy(summed), np.full(2, 2.0, np.float32))
    np.testing.assert_allclose(as_numpy(rest["neg"]), np.full(3, -5.0, np.float32))


# --- 4. static values: snapshot, specialization, validation --------------------


def test_static_int_specializes_graph():
    def f(x, k):
        return etl.add(x, etl.constant(etl.tensor(k, dtype=etl.float32)))

    spec = etl.TensorSpec((2,), etl.float32)
    g1 = etl.trace(f, spec, 1)
    g2 = etl.trace(f, spec, 2)

    assert len(g1.static_values) == 1
    record = g1.static_values[0]
    assert record.index == 1
    assert record.path == (1,)
    assert record.value == 1
    assert record.kind == "int"

    serialized1 = etl.ir.serialize_module(g1.module)
    serialized2 = etl.ir.serialize_module(g2.module)
    assert serialized1 != serialized2
    assert serialized1["constants"] != serialized2["constants"]


@pytest.mark.parametrize(
    ("flag", "present", "absent"),
    [(True, "multiply", "negate"), (False, "negate", "multiply")],
)
def test_static_python_if_specializes(flag, present, absent):
    def f(x, flag):
        if flag:
            return etl.multiply(x, x)
        return etl.negate(x)

    g = etl.trace(f, etl.TensorSpec((2,), etl.float32), flag)
    names = _op_names(g)
    assert present in names
    assert absent not in names
    g.verify()


STATIC_CASES = [
    pytest.param(1.5, "float", id="float"),
    pytest.param("hello", "str", id="str"),
    pytest.param(None, "NoneType", id="none"),
    pytest.param(1 + 2j, "complex", id="complex"),
    pytest.param(True, "bool", id="bool"),
    pytest.param(slice(0, 2), "slice", id="slice"),
    pytest.param(Mode.B, "Mode", id="enum"),
    # np.dtype kind name is numpy-version-dependent (dtype vs Float32DType)
    pytest.param(np.dtype("float32"), None, id="dtype"),
]


@pytest.mark.parametrize(("value", "expected_kind"), STATIC_CASES)
def test_static_value_types_accepted(value, expected_kind):
    def f(x, s):
        return etl.multiply(x, 2.0)

    g = etl.trace(f, etl.TensorSpec((2,), etl.float32), value)
    assert len(g.static_values) == 1
    record = g.static_values[0]
    assert record.index == 1
    assert record.path == (1,)
    assert record.value == value
    if expected_kind is not None:
        assert record.kind == expected_kind
    g.verify()


def test_static_value_run_mismatch_raises(run_graph):
    def f(x, k):
        return etl.add(x, etl.constant(etl.tensor(k, dtype=etl.float32)))

    g = etl.trace(f, etl.TensorSpec((2,), etl.float32), 1)
    with pytest.raises(etl.TraceError, match="graph was specialized on"):
        run_graph(g, np.ones(2, np.float32), 7)
    # A wrong static TYPE at run time is rejected the same way.
    with pytest.raises(etl.TraceError, match="graph was specialized on"):
        run_graph(g, np.ones(2, np.float32), 1.0)


# --- 5. closure capture ---------------------------------------------------------


@pytest.mark.parametrize("closure", [2, 3.5], ids=["int", "float"])
def test_closure_scalar_captured_and_snapshotted(closure, run_graph, as_numpy):
    def f(x):
        return etl.multiply(x, closure)

    g = etl.trace(f, etl.TensorSpec((3,), etl.float32))
    assert "constant" in _op_names(g)
    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_allclose(as_numpy(result), np.ones(3, np.float32) * closure)


def test_closure_captured_tensor_raises():
    w = etl.tensor(np.ones(3, np.float32))

    def f(x):
        return etl.add(x, w)

    with pytest.raises(etl.TraceError, match="Concrete Tensor"):
        etl.trace(f, etl.TensorSpec((3,), etl.float32))


# --- 6. plain functions ----------------------------------------------------------


def test_plain_fn_equivalent_to_defn():
    def body(x):
        return etl.multiply(x, x)

    decorated = etl.defn(body)  # the SAME fn object, wrapped
    spec = etl.TensorSpec((2,), etl.float32)

    g_defn = etl.trace(decorated, spec)
    g_plain = etl.trace(body, spec)
    g_unwrapped = etl.trace(decorated.fn, spec)

    assert isinstance(g_defn, etl.Graph)
    assert isinstance(g_plain, etl.Graph)
    assert etl.ir.serialize_module(g_defn.module) == etl.ir.serialize_module(
        g_plain.module
    )
    assert etl.ir.serialize_module(g_defn.module) == etl.ir.serialize_module(
        g_unwrapped.module
    )


# --- 7. zero-IO graphs -----------------------------------------------------------


def test_zero_input_static_output(run_graph):
    def f():
        return 3

    g = etl.trace(f)
    assert len(g.tensor_specs) == 0
    assert len(g.static_values) == 0
    assert len(g.output_static_values) == 1
    record = g.output_static_values[0]
    assert record.index == 0
    assert record.path == ()
    assert record.value == 3
    assert record.kind == "int"

    g.verify()
    assert run_graph(g) == 3


def test_zero_tensor_output_consuming_input(run_graph):
    def f(x):
        _ = etl.add(x, x)  # builds IR …
        return None  # … but yields no tensor result

    g = etl.trace(f, etl.TensorSpec((3,), etl.float32))
    terminator = g.module.main.entry_block.terminator
    assert terminator.name == "return"
    assert terminator.operands == ()
    g.verify()
    assert run_graph(g, np.ones(3, np.float32)) is None


def test_zero_args_tensor_output(run_graph, as_numpy):
    def f():
        return etl.constant(etl.zeros((2,), etl.float32))

    g = etl.trace(f)
    assert len(g.tensor_specs) == 0
    g.verify()
    result = run_graph(g)
    np.testing.assert_array_equal(as_numpy(result), np.zeros(2, np.float32))


# --- 8. error cases --------------------------------------------------------------

BAD_SPEC_CASES = [
    pytest.param(
        etl.tensor(np.ones(3, np.float32)), id="concrete-tensor"
    ),
    pytest.param(np.ones(3, np.float32), id="ndarray"),
    pytest.param(_symbolic_tensor(), id="symbolic-tensor"),
    pytest.param(object(), id="unknown-object"),
]


@pytest.mark.parametrize("bad_spec", BAD_SPEC_CASES)
def test_invalid_input_spec_raises(bad_spec):
    def f(x):
        return x

    with pytest.raises(etl.TraceError) as excinfo:
        etl.trace(f, bad_spec)
    # every error names the pytree path of the offending leaf
    assert "pytree path [0]" in str(excinfo.value)


BAD_OUTPUT_CASES = [
    pytest.param(
        lambda x: etl.tensor(np.ones(3, np.float32)),
        "Invalid trace output at pytree path",
        id="concrete-tensor",
    ),
    pytest.param(
        lambda x: etl.TensorSpec((3,), etl.float32),
        "TensorSpec cannot be returned",
        id="tensorspec",
    ),
    pytest.param(
        lambda x: object(),
        "Invalid trace output at pytree path",
        id="unknown-object",
    ),
]


@pytest.mark.parametrize(("bad_output_fn", "match"), BAD_OUTPUT_CASES)
def test_invalid_output_raises(bad_output_fn, match):
    with pytest.raises(etl.TraceError, match=match):
        etl.trace(bad_output_fn, etl.TensorSpec((3,), etl.float32))


def test_wrong_arg_count_raises_type_error():
    def f(a, b):
        return etl.add(a, b)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        etl.trace(f, etl.TensorSpec((3,), etl.float32))


def test_non_callable_raises_type_error():
    with pytest.raises(TypeError, match="not callable"):
        etl.trace(42)


# --- 9. output static values re-inserted -----------------------------------------


def test_output_static_values_reinserted(run_graph, as_numpy):
    def f(x):
        return (etl.add(x, x), "meta", 3.5)

    g = etl.trace(f, etl.TensorSpec((2,), etl.float32))
    assert [(r.index, r.path, r.value) for r in g.output_static_values] == [
        (1, (1,), "meta"),
        (2, (2,), 3.5),
    ]

    doubled, meta, half = run_graph(g, np.ones(2, np.float32))
    np.testing.assert_allclose(as_numpy(doubled), np.full(2, 2.0, np.float32))
    assert meta == "meta"
    assert half == 3.5

    # unflatten_outputs re-inserts the statics at the recorded positions too
    rebuilt = g.unflatten_outputs([etl.tensor(np.full(2, 2.0, np.float32))])
    assert isinstance(rebuilt[0], etl.Tensor)
    assert rebuilt[1] == "meta"
    assert rebuilt[2] == 3.5


# --- 10. SymbolicTensor purity ----------------------------------------------------


def test_symbolic_tensor_has_no_concrete_accessors(run_graph, as_numpy):
    def f(x):
        assert not hasattr(x, "numpy")
        assert not hasattr(x, "__dlpack__")
        assert not hasattr(x, "data_ptr")
        return etl.multiply(x, 2.0)

    g = etl.trace(f, etl.TensorSpec((3,), etl.float32))
    result = run_graph(g, np.ones(3, np.float32))
    np.testing.assert_allclose(as_numpy(result), np.full(3, 2.0, np.float32))


# --- 11. fresh graphs, no caching --------------------------------------------------


def test_trace_not_cached():
    def f(x):
        return etl.negate(x)

    spec = etl.TensorSpec((3,), etl.float32)
    g1 = etl.trace(f, spec)
    g2 = etl.trace(f, spec)

    assert g1 is not g2
    assert g1.module is not g2.module
    # Deterministic tracing: structurally identical modules serialize alike.
    assert etl.ir.serialize_module(g1.module) == etl.ir.serialize_module(g2.module)
