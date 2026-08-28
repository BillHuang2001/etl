"""Tests for `etl.Graph`: attributes, printing, verification, I/O trees, persistence.

`Graph` is the public staging object produced by `etl.trace` (see
`etl/trace/CONTEXT.md` for the binding contract). These tests exercise the
contract from the outside: attributes (`module`, `input_specs`, `tensor_specs`,
`output_tree`, `static_values`, `source_locations`), `print`/`verify`,
`flatten_inputs`/`validate_inputs` (structure + static + dtype/shape checks),
`unflatten_outputs` (static leaves re-inserted), save/load round-trips through
the persist container, `PersistenceError` paths, and `signature_info`.

Conventions: constants inside traced functions are embedded with
`etl.constant(etl.tensor(...))` — concrete `Tensor`s are never closure-captured
(nor passed as trace inputs, which are `TensorSpec`s).
"""

import json
import re
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytest

import etl
from etl import ir
from etl.trace import StaticValue

SHAPE = (2, 3)


# --- traced-graph builders (one per shape of I/O used below) -----------------


def _trace_add_graph():
    """Two tensor inputs, one tensor output: f(x, w) = x + w."""

    def f(x, w):
        return etl.add(x, w)

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32), etl.TensorSpec(SHAPE, etl.float32))


def _trace_structured_graph():
    """Structured inputs: x (spec) and y = (list of 2 specs, dict with spec + static int)."""

    def f(x, y):
        ys, cfg = y
        return {"out": etl.add(etl.add(x, ys[0]), etl.add(ys[1], cfg["w"]))}

    specs = (
        etl.TensorSpec(SHAPE, etl.float32),
        (
            [etl.TensorSpec(SHAPE, etl.float32), etl.TensorSpec(SHAPE, etl.float32)],
            {"w": etl.TensorSpec(SHAPE, etl.float32), "b": 5},
        ),
    )
    return etl.trace(f, *specs)


def _trace_wb_graph():
    """f(x, y) with y = {"w": spec, "b": 5} — the flatten/validate workhorse."""

    def f(x, y):
        return etl.add(x, y["w"])

    return etl.trace(
        f,
        etl.TensorSpec(SHAPE, etl.float32),
        {"w": etl.TensorSpec(SHAPE, etl.float32), "b": 5},
    )


def _trace_symbolic_graph():
    """Symbolic dims + static input leaf + structured output — save/load target."""

    def f(x, y):
        return {"r": etl.add(x, y["w"]), "meta": "done"}

    return etl.trace(
        f,
        etl.TensorSpec((etl.Dim("B"), 3), etl.float32),
        {"w": etl.TensorSpec((etl.Dim("B"), 3), etl.float32), "b": 5},
    )


def _trace_zero_input_graph():
    """A legal zero-input graph (constants only, embedded explicitly)."""

    def f():
        c = etl.constant(etl.tensor(2.0, dtype=etl.float32))
        return etl.add(c, c)

    return etl.trace(f)


def _trace_zero_result_graph():
    """A legal zero-result graph (all outputs static)."""

    def f(x):
        _ = etl.add(x, x)
        return {"done": True}

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32))


def _trace_dict_output():
    """Dict output with a static leaf re-inserted by `unflatten_outputs`."""

    def f(x):
        return {"r": etl.add(x, x), "meta": "done"}

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32))


def _trace_nested_tuple_output():
    """Nested tuple output with a static leaf in the middle."""

    def f(x):
        return (etl.add(x, x), ("nested", etl.multiply(x, x)))

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32))


_Point = namedtuple("_Point", ["r", "meta"])


@dataclass
class _Out:
    r: object
    meta: str


def _trace_dataclass_output():
    def f(x):
        return _Out(etl.add(x, x), "hi")

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32))


def _trace_namedtuple_output():
    def f(x):
        return _Point(etl.add(x, x), "hi")

    return etl.trace(f, etl.TensorSpec(SHAPE, etl.float32))


# --- 1. attributes -----------------------------------------------------------


def test_graph_attributes():
    g = _trace_structured_graph()

    assert isinstance(g.module, ir.Module)
    assert g.module.main.name == "main"

    assert isinstance(g.input_specs, etl.TreeSpec)
    assert isinstance(g.output_tree, etl.TreeSpec)

    # tensor_specs: tuple of TensorSpec in flat leaf (== block-arg) order.
    assert isinstance(g.tensor_specs, tuple)
    assert all(isinstance(s, etl.TensorSpec) for s in g.tensor_specs)
    assert len(g.tensor_specs) == 4  # x, ys[0], ys[1], cfg["w"]
    block_args = g.module.main.entry_block.arguments
    assert len(block_args) == 4
    for arg, spec in zip(block_args, g.tensor_specs):
        assert arg.type.dtype == spec.dtype
        assert tuple(arg.type.shape) == tuple(spec.shape)

    # static_values: StaticValue records (index, path, value, kind). Dict keys
    # are sorted ("b" < "w"), so the static "b" leaf sits before cfg["w"].
    assert isinstance(g.static_values, tuple)
    (record,) = g.static_values
    assert isinstance(record, StaticValue)
    assert record.index == 3
    assert record.path == (1, 1, "b")
    assert record.value == 5
    assert record.kind == "int"

    # source_locations: one entry per tensor input (value id → ir.Location).
    assert set(g.source_locations) == {arg.id for arg in block_args}
    assert all(isinstance(loc, ir.Location) for loc in g.source_locations.values())

    # repr mentions tensor-input and static-value counts.
    r = repr(g)
    assert "4 tensor inputs" in r
    assert "1 static values" in r


# --- 2. print() / pretty-printing --------------------------------------------


def test_print_writes_pretty_ir_to_stdout(capsys):
    g = _trace_add_graph()
    g.print()
    out = capsys.readouterr().out
    assert out.strip() == ir.pretty_print(g.module).strip()
    assert "func @main" in out
    assert "etl.return" in out
    # SSA-style result naming (per-function renumbering: %0, %1, ...).
    assert re.search(r"%0 = etl\.(add|constant)", out)


# --- 3. verify() -------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        _trace_add_graph,
        _trace_structured_graph,
        _trace_wb_graph,
        _trace_symbolic_graph,
        _trace_zero_input_graph,
        _trace_zero_result_graph,
        _trace_dict_output,
        _trace_nested_tuple_output,
        _trace_dataclass_output,
        _trace_namedtuple_output,
    ],
)
def test_verify_passes_on_every_traced_graph(builder):
    # The invalid-module case is deliberately skipped: `Graph` accepts only a
    # prebuilt module, and forging one that satisfies the ir.Builder invariants
    # yet fails `ir.verify` would mean duplicating ir internals here. Invalid
    # modules are the ir package's test domain (tests/ir/).
    builder().verify()


# --- 4. flatten_inputs / validate_inputs -------------------------------------


def test_flatten_and_validate_inputs():
    g = _trace_wb_graph()
    x = np.ones(SHAPE, np.float32)
    w = np.zeros(SHAPE, np.float32)

    flat = g.flatten_inputs((x, {"w": w, "b": 5}))
    assert len(flat) == 2
    # ndarrays are wrapped to etl.Tensor via from_numpy ...
    assert all(isinstance(t, etl.Tensor) for t in flat)
    # ... in tree order: x first, then w.
    np.testing.assert_array_equal(flat[0].numpy(), x)
    np.testing.assert_array_equal(flat[1].numpy(), w)

    # Dict keys are sorted — insertion order of the run-time dict is irrelevant.
    flat_permuted = g.flatten_inputs((x, {"b": 5, "w": w}))
    assert flat_permuted == flat

    # validate_inputs is the documented alias: same flat validated list.
    assert g.validate_inputs((x, {"w": w, "b": 5})) == flat

    # core.Tensor inputs are accepted as-is (no copy, no re-wrap).
    tx, tw = etl.Tensor(x), etl.Tensor(w)
    flat_tensors = g.flatten_inputs((tx, {"w": tw, "b": 5}))
    assert flat_tensors[0] is tx
    assert flat_tensors[1] is tw


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        # wrong structure: tuple instead of dict
        (lambda x, w: (x, (w, 5)), etl.TraceError, "does not match"),
        # wrong structure: wrong dict key name
        (lambda x, w: (x, {"z": w, "b": 5}), etl.TraceError, "does not match"),
        # wrong dtype
        (lambda x, w: (x, {"w": w.astype(np.float64), "b": 5}), etl.DTypeError, "dtype mismatch"),
        # wrong static shape
        (lambda x, w: (x, {"w": np.zeros((3, 3), np.float32), "b": 5}), etl.ShapeError, "shape mismatch"),
        # rank mismatch
        (lambda x, w: (x, {"w": np.zeros((2, 3, 1), np.float32), "b": 5}), etl.ShapeError, "rank mismatch"),
        # static value mismatch
        (lambda x, w: (x, {"w": w, "b": 6}), etl.TraceError, "graph was specialized on"),
        # static kind mismatch (bool vs recorded int)
        (lambda x, w: (x, {"w": w, "b": True}), etl.TraceError, "graph was specialized on"),
        # non-tensor leaf at a tensor position
        (lambda x, w: (x, {"w": object(), "b": 5}), etl.TraceError, "must be a core.Tensor"),
    ],
)
def test_flatten_inputs_errors(args, error, match):
    g = _trace_wb_graph()
    x = np.ones(SHAPE, np.float32)
    w = np.zeros(SHAPE, np.float32)
    with pytest.raises(error, match=match):
        g.flatten_inputs(args(x, w))


@pytest.mark.parametrize(
    ("builder", "args", "path"),
    [
        # container-type mismatch at element 1: tuple instead of dict
        (_trace_wb_graph, lambda x, w: (x, (w, 5)), "[1]"),
        # node_data mismatch (wrong dict key) at the dict node itself
        (_trace_wb_graph, lambda x, w: (x, {"z": w, "b": 5}), "[1]"),
        # deep node_data mismatch: wrong dict key at path (1,1)
        (_trace_structured_graph, lambda x, w: (x, ([w, w], {"z": w, "b": 5})), "[1][1]"),
        # deep arity mismatch: list of 1 vs 2 at path (1,0)
        (_trace_structured_graph, lambda x, w: (x, ([w], {"w": w, "b": 5})), "[1][0]"),
        # deep container-type mismatch: tuple instead of list at path (1,0)
        (_trace_structured_graph, lambda x, w: (x, ((w, w), {"w": w, "b": 5})), "[1][0]"),
    ],
    ids=[
        "tuple-instead-of-dict",
        "wrong-dict-key",
        "wrong-dict-key-deep",
        "wrong-list-arity-deep",
        "tuple-instead-of-list-deep",
    ],
)
def test_flatten_inputs_mismatch_reports_first_mismatch_path(builder, args, path):
    g = builder()
    x = np.ones(SHAPE, np.float32)
    w = np.zeros(SHAPE, np.float32)
    with pytest.raises(etl.TraceError) as excinfo:
        g.flatten_inputs(args(x, w))
    msg = str(excinfo.value)
    # the old lead-in is preserved, with the pytree path detail appended
    assert "run-time input structure does not match the traced signature" in msg
    assert f"first mismatch at pytree path {path}:" in msg


def test_flatten_inputs_symbolic_and_static_dims():
    def f(a):
        return etl.add(a, a)

    symbolic = etl.trace(f, etl.TensorSpec((etl.Dim("B"), 3), etl.float32))
    flat = symbolic.flatten_inputs((np.ones((2, 3), np.float32),))
    assert len(flat) == 1
    assert flat[0].shape == (2, 3)
    # The symbolic "B" binds any leading size, but the static dim 3 still holds.
    with pytest.raises(etl.ShapeError, match="shape mismatch"):
        symbolic.flatten_inputs((np.ones((2, 4), np.float32),))

    # A fully static dim in the spec rejects mismatched sizes.
    concrete = etl.trace(f, etl.TensorSpec((2, 3), etl.float32))
    with pytest.raises(etl.ShapeError, match="shape mismatch"):
        concrete.flatten_inputs((np.ones((3, 3), np.float32),))


# --- 5. unflatten_outputs ----------------------------------------------------


def test_unflatten_outputs_dict_with_static_leaf():
    g = _trace_dict_output()
    (record,) = g.output_static_values
    assert record.value == "done"
    assert record.index == 0  # dict keys sorted: "meta" < "r"

    out = g.unflatten_outputs([np.ones(SHAPE, np.float32)])
    assert set(out) == {"r", "meta"}
    assert out["meta"] == "done"  # static leaf re-inserted
    assert isinstance(out["r"], etl.Tensor)
    np.testing.assert_array_equal(out["r"].numpy(), np.ones(SHAPE, np.float32))


def test_unflatten_outputs_nested_tuple():
    g = _trace_nested_tuple_output()
    out = g.unflatten_outputs(
        [np.ones(SHAPE, np.float32), np.full(SHAPE, 2.0, np.float32)]
    )
    assert isinstance(out, tuple) and len(out) == 2
    t, (meta, s) = out
    assert meta == "nested"
    assert isinstance(t, etl.Tensor)
    assert isinstance(s, etl.Tensor)


def test_unflatten_outputs_dataclass_and_namedtuple():
    # Dataclass containers round-trip with the same dataclass type.
    g = _trace_dataclass_output()
    out = g.unflatten_outputs([np.ones(SHAPE, np.float32)])
    assert isinstance(out, _Out)
    assert out.meta == "hi"
    assert isinstance(out.r, etl.Tensor)

    # Same for namedtuples.
    gn = _trace_namedtuple_output()
    outn = gn.unflatten_outputs([np.ones(SHAPE, np.float32)])
    assert isinstance(outn, _Point)
    assert outn.meta == "hi"
    assert isinstance(outn.r, etl.Tensor)


# --- 6. save/load round-trip -------------------------------------------------


def _skeleton(spec):
    """Structural description of a TreeSpec ignoring leaf types.

    Trace trees record tensor leaves via private non-dataclass markers
    (`_TensorSpecLeaf`/`_SymbolicLeaf`) that are compared by skeleton only —
    see the "Tree leaf markers" note in etl/trace/CONTEXT.md.
    """
    if not spec.children:
        return ("leaf",)
    node_data = spec.node_data
    if isinstance(node_data, (tuple, list)):
        node_data = tuple(node_data)
    return (spec.type, node_data, tuple(_skeleton(c) for c in spec.children))


def test_save_load_roundtrip(tmp_path, run_graph, as_numpy):
    g = _trace_symbolic_graph()
    path = tmp_path / "g.etlgraph"
    g.save(str(path))
    g2 = etl.Graph.load(str(path))

    # Symbolic dims survive (Dim equality is by name+size); StaticValue is a
    # frozen dataclass with structural equality.
    assert g2.tensor_specs == g.tensor_specs
    assert g2.static_values == g.static_values
    assert g2.output_static_values == g.output_static_values
    # Input/output tree SKELETONS match (leaf types are private markers).
    assert _skeleton(g2.input_specs) == _skeleton(g.input_specs)
    assert _skeleton(g2.output_tree) == _skeleton(g.output_tree)
    # The IR itself round-trips identically.
    assert ir.serialize_module(g2.module) == ir.serialize_module(g.module)
    g2.verify()

    # Both graphs execute to equal numerical results.
    x = np.ones((2, 3), np.float32)
    w = np.full((2, 3), 3.0, np.float32)
    args = (x, {"w": w, "b": 5})
    r1 = as_numpy(run_graph(g, *args))
    r2 = as_numpy(run_graph(g2, *args))
    assert r1["meta"] == r2["meta"] == "done"
    np.testing.assert_allclose(r1["r"], r2["r"])

    # A loaded graph is itself persistable: save(str) → load(str).
    path3 = tmp_path / "g2.etlgraph"
    g2.save(str(path3))
    g3 = etl.Graph.load(str(path3))
    assert g3.tensor_specs == g.tensor_specs
    g3.verify()

    # pathlib.Path works for both save and load.
    path4 = tmp_path / "g4.etlgraph"
    g.save(path4)
    assert etl.Graph.load(path4).tensor_specs == g.tensor_specs


# --- 7. persistence errors ---------------------------------------------------


def test_load_corrupted_file_raises(tmp_path):
    g = _trace_add_graph()
    path = tmp_path / "g.etlgraph"
    g.save(str(path))
    path.write_bytes(b"garbage" * 64)
    with pytest.raises(etl.PersistenceError, match="corrupt"):
        etl.Graph.load(str(path))


def test_load_wrong_payload_type_raises(tmp_path):
    path = tmp_path / "other.etlgraph"
    etl.persist.save_object(str(path), "tensor", {"v": 1})
    with pytest.raises(etl.PersistenceError, match="payload_type mismatch"):
        etl.Graph.load(str(path))


# --- 8. signature_info -------------------------------------------------------


def test_signature_info_is_json_safe_encoded_metadata():
    g = _trace_symbolic_graph()
    info = g.signature_info()
    assert set(info) == {
        "input_tree",
        "output_tree",
        "input_specs",
        "output_specs",
        "static_values",
        "output_static_values",
    }
    # JSON-safe by contract (all values persist-encoded).
    json.dumps(info)
    # The values are persist-ENCODED metadata (plain dicts), not the live
    # in-process objects — in-process consumers use the Graph's attributes.
    assert not isinstance(info["input_tree"], etl.TreeSpec)
    assert not isinstance(info["input_specs"], etl.TensorSpec)
    assert isinstance(info["input_tree"], dict)
