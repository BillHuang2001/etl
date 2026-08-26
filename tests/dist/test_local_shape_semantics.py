"""Local-shape semantics of the six `etl.dist` collectives.

Contract under test: `etl/dist/CONTEXT.md`, section "Local-shape semantics
(worked examples)" + the shape rules in `etl/ir/inference.py`
(``infer_all_gather``, ``infer_reduce_scatter``, ``infer_all_to_all``,
``infer_identity``).

The semantics encoded here: collectives produce LOCAL tensors with explicit
local-shape math.

- Explicit groups: result dims are exact plain ints computed at trace time
  (axis dim × or ÷ group size; shape-preserving collectives keep the
  identical shape tuple). No global tensor type and no symbolic shape
  objects ever appear in a collective result.
- World group (``group=None``): shape-affected dims become ``None``
  (runtime-dynamic; the executor validates them at run time);
  shape-preserving collectives keep their concrete dims.
- Axes normalize Python-style (negative wraps) and the normalized axis is
  what gets recorded in the op attributes.

Both the IR result type (read off the built op's ``ValueType``) and the
``SymbolicTensor`` facade seen inside the defn must agree.
"""

import pytest

import etl
import etl.dist as dist
from etl import core

#: Collective frontend functions, keyed by contract name.
_COLLECTIVES = {
    "all_reduce": dist.all_reduce,
    "all_gather": dist.all_gather,
    "reduce_scatter": dist.reduce_scatter,
    "all_to_all": dist.all_to_all,
    "broadcast": dist.broadcast,
    "collective_permute": dist.collective_permute,
}

#: The explicit 4-rank group used by the contract's worked examples.
G4 = dist.group("data", (0, 1, 2, 3))
#: Explicit 2-rank group for the smaller-group worked examples.
G2 = dist.group("ab", (0, 1))


def _trace_collective(fn, in_shape):
    """Trace ``fn(x)`` with one float32 input and observe the result.

    ``fn`` must build exactly one collective op and return it. The result
    SymbolicTensor is captured by appending it to a list while the traced
    function runs (the objective's capture idiom).

    Returns a dict with:
    - ``ir_shape`` / ``ir_dtype``: the collective op's result ``ValueType``
      (the registry ``shape_fn`` output — the IR ground truth),
    - ``facade_shape`` / ``facade_dtype``: the ``SymbolicTensor`` facade the
      defn saw (``.shape`` entries may be ``None`` for runtime-dynamic dims),
    - ``op``: the built collective op (attribute checks).
    """
    seen = []

    @etl.defn
    def traced(x):
        y = fn(x)
        seen.append(y)
        return y

    graph = etl.trace(traced, core.TensorSpec(in_shape, core.float32))
    fn_ir = graph.module.functions[0]
    ops = fn_ir.entry_block.ops  # entry_block is a property → Block
    assert ops[-1].name == "return"
    assert len(ops) == 2  # one collective op + the return op
    op = ops[0]
    result_type = op.results[0].type
    facade = seen[0]
    return {
        "ir_shape": result_type.shape,
        "ir_dtype": result_type.dtype,
        "facade_shape": facade.shape,
        "facade_dtype": facade.dtype,
        "op": op,
    }


#: The contract table (worked examples, exact numbers): explicit-group cases.
#: (kind, kwargs, in_shape, expected_out_shape, group ranks)
_CONTRACT_CASES = [
    ("all_reduce", {"op": "sum"}, (4, 8), (4, 8), (0, 1, 2, 3)),
    ("all_gather", {"axis": 0}, (256, 1024), (1024, 1024), (0, 1, 2, 3)),
    ("all_gather", {"axis": 1}, (256, 1024), (256, 4096), (0, 1, 2, 3)),
    ("reduce_scatter", {"op": "sum", "axis": 0}, (1024, 1024), (256, 1024), (0, 1, 2, 3)),
    ("reduce_scatter", {"op": "sum", "axis": 1}, (256, 1024), (256, 256), (0, 1, 2, 3)),
    ("all_to_all", {"split_axis": 0, "concat_axis": 1}, (512, 64), (128, 256), (0, 1, 2, 3)),
    # Equal split/concat axes ⇒ shape unchanged (÷ 4 then × 4 cancels).
    ("all_to_all", {"split_axis": 0, "concat_axis": 0}, (512, 64), (512, 64), (0, 1, 2, 3)),
    ("broadcast", {"src_rank": 0}, (256, 1024), (256, 1024), (0, 1, 2, 3)),
    (
        "collective_permute",
        {"source_target_pairs": ((0, 1), (1, 2), (2, 3), (3, 0))},
        (256, 1024),
        (256, 1024),
        (0, 1, 2, 3),
    ),
    # 2-rank group worked examples.
    ("all_gather", {"axis": 0}, (3, 5), (6, 5), (0, 1)),
    ("reduce_scatter", {"op": "sum", "axis": 1}, (2, 6), (2, 3), (0, 1)),
]

_CONTRACT_IDS = [
    "all_reduce-4rank",
    "all_gather-4rank-axis0",
    "all_gather-4rank-axis1",
    "reduce_scatter-4rank-axis0",
    "reduce_scatter-4rank-axis1",
    "all_to_all-4rank-split0-concat1",
    "all_to_all-4rank-equal-axes",
    "broadcast-4rank",
    "collective_permute-4rank",
    "all_gather-2rank-axis0",
    "reduce_scatter-2rank-axis1",
]


@pytest.mark.parametrize(
    ("kind", "kwargs", "in_shape", "expected", "ranks"),
    _CONTRACT_CASES,
    ids=_CONTRACT_IDS,
)
def test_local_shape_contract(kind, kwargs, in_shape, expected, ranks):
    """Every contract-table case yields exactly the expected local shape.

    Both the IR result type and the SymbolicTensor facade must agree, and
    the dtype must be preserved (float32 in, float32 out).
    """
    group = dist.group(f"g{len(ranks)}", ranks)

    def collective(x):
        return _COLLECTIVES[kind](x, **kwargs, group=group)

    result = _trace_collective(collective, in_shape)
    assert result["ir_shape"] == expected
    assert result["facade_shape"] == expected
    assert result["ir_dtype"] == core.float32
    assert result["facade_dtype"] == core.float32
    # The op must record the group by name + size (self-describing attrs).
    assert result["op"].attributes["group"] == f"g{len(ranks)}"
    assert result["op"].attributes["group_size"] == len(ranks)


@pytest.mark.parametrize(
    ("kind", "kwargs", "in_shape", "expected", "attr_checks"),
    [
        # all_gather axis=-1 → normalized axis 1 → dim × 4.
        ("all_gather", {"axis": -1}, (256, 1024), (256, 4096), (("axis", 1),)),
        # all_gather axis=-2 → normalized axis 0 → dim × 4.
        ("all_gather", {"axis": -2}, (256, 1024), (1024, 1024), (("axis", 0),)),
        ("reduce_scatter", {"op": "sum", "axis": -1}, (256, 1024), (256, 256), (("axis", 1),)),
        ("reduce_scatter", {"op": "sum", "axis": -2}, (1024, 1024), (256, 1024), (("axis", 0),)),
        (
            "all_to_all",
            {"split_axis": -2, "concat_axis": -1},
            (512, 64),
            (128, 256),
            (("split_axis", 0), ("concat_axis", 1)),
        ),
    ],
    ids=[
        "all_gather-axis-1",
        "all_gather-axis-2",
        "reduce_scatter-axis-1",
        "reduce_scatter-axis-2",
        "all_to_all-split-2-concat-1",
    ],
)
def test_negative_axes_normalize(kind, kwargs, in_shape, expected, attr_checks):
    """Negative axes wrap Python-style and produce the same shapes as their
    normalized counterparts; the normalized axis is what gets recorded in
    the op attributes (deterministic serialization)."""
    def collective(x):
        return _COLLECTIVES[kind](x, **kwargs, group=G4)

    result = _trace_collective(collective, in_shape)
    assert result["ir_shape"] == expected
    assert result["facade_shape"] == expected
    assert result["ir_dtype"] == core.float32
    assert result["facade_dtype"] == core.float32
    for attr_name, normalized in attr_checks:
        assert result["op"].attributes[attr_name] == normalized


@pytest.mark.parametrize(
    ("kind", "kwargs", "in_shape", "expected"),
    [
        # Shape-preserving collectives keep concrete dims even for the world.
        ("all_reduce", {"op": "sum"}, (4, 8), (4, 8)),
        ("broadcast", {"src_rank": 0}, (256, 1024), (256, 1024)),
        (
            "collective_permute",
            {"source_target_pairs": ((0, 1), (1, 2))},
            (256, 1024),
            (256, 1024),
        ),
        # Shape-affected collectives defer the affected dim to run time.
        ("all_gather", {"axis": 0}, (256, 1024), (None, 1024)),
        ("reduce_scatter", {"op": "sum", "axis": 1}, (2, 6), (2, None)),
        ("all_to_all", {"split_axis": 0, "concat_axis": 1}, (512, 64), (None, None)),
    ],
    ids=[
        "all_reduce",
        "broadcast",
        "collective_permute",
        "all_gather-axis0",
        "reduce_scatter-axis1",
        "all_to_all-split0-concat1",
    ],
)
def test_world_group_runtime_dynamic_dims(kind, kwargs, in_shape, expected):
    """World group (group=None): affected dims are None (runtime-dynamic).

    The IR result type and the SymbolicTensor facade must both carry the
    ``None`` dims (the facade restores them via the _wrap_result
    workaround), and the op must record ``group="world"`` with
    ``group_size=None`` (size only known at run time).
    """
    def collective(x):
        return _COLLECTIVES[kind](x, **kwargs)  # group=None → world group

    result = _trace_collective(collective, in_shape)
    assert result["ir_shape"] == expected
    assert result["facade_shape"] == expected
    assert result["ir_dtype"] == core.float32
    assert result["facade_dtype"] == core.float32
    assert result["op"].attributes["group"] == "world"
    assert result["op"].attributes["group_size"] is None


def test_explicit_group_results_are_exact_plain_ints():
    """Principle: every result dim for explicit groups is an exact plain int.

    Local shapes only — no global tensor type and no symbolic shape objects
    (``Dim``/``DimExpr``/``None``) may appear in a collective result for an
    explicit group, neither in the IR result type nor in the facade.
    """
    for kind, kwargs, in_shape, expected, ranks in _CONTRACT_CASES:
        group = dist.group(f"g{len(ranks)}", ranks)

        def collective(x):
            return _COLLECTIVES[kind](x, **kwargs, group=group)

        result = _trace_collective(collective, in_shape)
        assert result["ir_shape"] == expected
        assert result["facade_shape"] == expected
        for dim in result["ir_shape"]:
            assert type(dim) is int
        for dim in result["facade_shape"]:
            assert type(dim) is int


def test_shape_preserving_collectives_return_identical_tuple():
    """Shape-preserving collectives return the identical local shape tuple."""
    in_shape = (256, 1024)
    cases = [
        ("all_reduce", lambda x: dist.all_reduce(x, "sum", G4)),
        ("broadcast", lambda x: dist.broadcast(x, 0, G4)),
        (
            "collective_permute",
            lambda x: dist.collective_permute(
                x, ((0, 1), (1, 2), (2, 3), (3, 0)), G4
            ),
        ),
        # all_to_all with equal axes also preserves the shape.
        ("all_to_all-equal-axes", lambda x: dist.all_to_all(x, 0, 0, G4)),
    ]
    for name, collective in cases:
        result = _trace_collective(collective, in_shape)
        assert result["ir_shape"] == in_shape, name
        assert result["facade_shape"] == in_shape, name
        assert result["ir_dtype"] == core.float32, name
        assert result["facade_dtype"] == core.float32, name
