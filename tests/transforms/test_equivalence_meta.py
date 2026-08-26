"""Meta-level equivalence: transforms are graph→graph.

Every transform (`vectorize`, `vmap`, `grad`, `jvp`, `vjp`) produces an
ORDINARY `Graph` of ORDINARY `etl.ops` ops — no transform-specific op may leak
into the IR, the result verifies, and the input graph is never mutated
(design principle 7; `./etl/transforms/CONTEXT.md`). Transforms never execute:
graphs are built from `TensorSpec`s alone, and the wrapped function only ever
sees `SymbolicTensor`s while tracing.
"""

import pytest

import etl
from etl.ir.serialize import serialize_module

SPEC = etl.TensorSpec((4, 3), etl.float32)
BATCHED_SPEC = etl.TensorSpec((8, 4, 3), etl.float32)
TANGENT = etl.TensorSpec((4, 3), etl.float32)

#: Names that would appear in the IR only if a transform leaked a
#: transform-level op instead of lowering to ordinary ops.
BANNED_OP_NAMES = frozenset(
    {"vmap", "vectorize", "grad", "jvp", "vjp", "batch", "batching", "autodiff"}
)


def f_rich(x):
    """A straight-line computation exercising several rule categories."""
    return etl.sum(etl.relu(x) * etl.tanh(x) + etl.exp(-x))


def all_ops(graph):
    """Every op in every block/region of the graph's module."""
    found = []

    def walk(block):
        for op in block.ops:
            found.append(op)
            for region in op.regions:
                for nested_block in region.blocks:
                    walk(nested_block)

    walk(graph.module.main.entry_block)
    return found


def assert_ordinary_graph(graph):
    """(a) verifies, (b) contains only ordinary registered ops, (c) has a
    `main` function terminated by `return`."""
    assert isinstance(graph, etl.Graph)
    graph.verify()  # raises VerificationError on bad IR
    main = graph.module.main
    assert main is not None
    entry = main.entry_block
    assert entry.terminator is not None
    assert entry.terminator.name == "return"
    for op in all_ops(graph):
        assert op.name not in BANNED_OP_NAMES, (
            f"transform op leaked into the IR: '{op.name}'"
        )
        assert op.opdef is not None, (
            f"op '{op.name}' is not an ordinary registered op"
        )


GRAPH_FORM_CASES = [
    ("vectorize", lambda graph: etl.vectorize(graph, 0)),
    ("vmap", lambda graph: etl.vmap(graph, 0)),
    ("grad", lambda graph: etl.grad(graph)),
    ("jvp", lambda graph: etl.jvp(graph, TANGENT)),
    ("vjp", lambda graph: etl.vjp(graph)),
]

CALLABLE_FORM_CASES = [
    ("vmap", lambda: etl.vmap(f_rich)(BATCHED_SPEC)),
    ("grad", lambda: etl.grad(f_rich)(SPEC)),
    ("jvp", lambda: etl.jvp(f_rich, TANGENT)(SPEC)),
    ("vjp", lambda: etl.vjp(f_rich)(SPEC)),
]


class TestOrdinaryGraphContract:
    @pytest.mark.parametrize(
        "name,build", GRAPH_FORM_CASES, ids=[name for name, _ in GRAPH_FORM_CASES]
    )
    def test_graph_form_produces_ordinary_graph(self, name, build):
        graph = etl.trace(f_rich, SPEC)
        result = build(graph)
        assert_ordinary_graph(result)

    @pytest.mark.parametrize(
        "name,build", CALLABLE_FORM_CASES, ids=[name for name, _ in CALLABLE_FORM_CASES]
    )
    def test_callable_form_produces_ordinary_graph(self, name, build):
        result = build()
        assert_ordinary_graph(result)


class TestTransformCallableKind:
    @pytest.mark.parametrize(
        "tc,kind",
        [
            (etl.vmap(f_rich), "vmap"),
            (etl.grad(f_rich), "grad"),
            (etl.jvp(f_rich, TANGENT), "jvp"),
            (etl.vjp(f_rich), "vjp"),
        ],
        ids=["vmap", "grad", "jvp", "vjp"],
    )
    def test_kind_attribute(self, tc, kind):
        assert tc.kind == kind


class TestTransformsNeverExecute:
    def test_building_requires_specs_only(self):
        seen = []

        def f(x):
            seen.append(type(x).__name__)
            return etl.sum(x * x)

        graph = etl.vmap(f)(BATCHED_SPEC)
        # The wrapped function is traced with SymbolicTensors — never called
        # with concrete Tensors, never executed.
        assert seen == ["SymbolicTensor"]
        assert isinstance(graph, etl.Graph)
        assert all(isinstance(spec, etl.TensorSpec) for spec in graph.tensor_specs)


class TestInputGraphNotMutated:
    @pytest.mark.parametrize(
        "name,build", GRAPH_FORM_CASES, ids=[name for name, _ in GRAPH_FORM_CASES]
    )
    def test_original_graph_unchanged(self, name, build):
        graph = etl.trace(f_rich, SPEC)
        before = serialize_module(graph.module)
        build(graph)
        assert serialize_module(graph.module) == before
