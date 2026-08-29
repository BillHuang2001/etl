"""Op registry tests — the canonical v1 OpDef set and the registry API.

The registry (``etl.ir.op_defs``) declares the contract of every EvoXIR op:
name, category, arity/result arity, effect, attribute schema, shape-inference
hook, region structure, and terminator role. These tests pin the canonical
set (91 ops) so any accidental addition/removal/rename/effect change in the
``etl`` package is caught here.
"""

import pytest

from etl import ir
from etl.ir import op_defs as _op_defs  # private: registry cleanup (see below)

# --- canonical op set ---------------------------------------------------------

_ELEMENTWISE_BINARY = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "remainder",
    "maximum",
    "minimum",
    "logical_and",
    "logical_or",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
)

_ELEMENTWISE_UNARY = (
    "abs",
    "negate",
    "square",
    "sqrt",
    "exp",
    "log",
    "log1p",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "erf",
    "sign",
    "logical_not",
)

_COMPARISON = ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")

_STRUCTURE = (
    "select",
    "broadcast",
    "reshape",
    "transpose",
    "slice",
    "gather",
    "scatter",
    "concatenate",
    "pad",
)

_REDUCTION = (
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_mean",
    "reduce_prod",
    "argmax",
    "argmin",
    "cumsum",
)

_LINALG = ("dot", "conv", "tril", "triu", "solve")

_CONTROL = (
    "constant",
    "stop_gradient",
    "if",
    "while",
    "call",
    "runtime_call",
    "block_call",
)

_TERMINATOR = ("return",)

_COLLECTIVE = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast_collective",
    "collective_permute",
    "rank",
    "world_size",
)

_SPARSE = (
    "sparse_from_dense",
    "sparse_to_dense",
    "sparse_coo_to_csr",
    "sparse_csr_to_coo",
    "sparse_coo_to_csc",
    "sparse_csc_to_coo",
    "sparse_negate",
    "sparse_add",
    "sparse_multiply",
    "sparse_multiply_dense",
    "sparse_reduce_sum",
    "sparse_transpose",
    "sparse_reshape",
    "sparse_concatenate",
    "sparse_dot_dense",
    "dense_dot_sparse",
)

CANONICAL_NAMES = (
    _ELEMENTWISE_BINARY
    + ("cast",)
    + _ELEMENTWISE_UNARY
    + _COMPARISON
    + _STRUCTURE
    + _REDUCTION
    + _LINALG
    + _CONTROL
    + _TERMINATOR
    + _COLLECTIVE
    + _SPARSE
)

_CATEGORY_EXPECTED = {
    **{name: "elementwise" for name in _ELEMENTWISE_BINARY},
    "cast": "elementwise",
    **{name: "elementwise" for name in _ELEMENTWISE_UNARY},
    **{name: "comparison" for name in _COMPARISON},
    **{name: "structure" for name in _STRUCTURE},
    **{name: "reduction" for name in _REDUCTION},
    **{name: "linalg" for name in _LINALG},
    **{name: "control" for name in _CONTROL},
    **{name: "terminator" for name in _TERMINATOR},
    **{name: "collective" for name in _COLLECTIVE},
    **{name: "sparse" for name in _SPARSE},
}

_EFFECT_EXPECTED = {name: "pure" for name in CANONICAL_NAMES}
_EFFECT_EXPECTED["runtime_call"] = "callback"
_EFFECT_EXPECTED["block_call"] = "read"
_EFFECT_EXPECTED["rank"] = "read"
_EFFECT_EXPECTED["world_size"] = "read"
_EFFECT_EXPECTED.update(
    {
        name: "collective"
        for name in (
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "broadcast_collective",
            "collective_permute",
        )
    }
)

_CATEGORY_ITEMS = sorted(_CATEGORY_EXPECTED.items())
_EFFECT_ITEMS = sorted(_EFFECT_EXPECTED.items())

#: (name, arity, result_count, regions, is_terminator) spot-checks.
_SIGNATURE_EXPECTED = [
    ("add", 2, 1, 0, False),
    ("constant", 0, 1, 0, False),
    ("rank", 0, 1, 0, False),
    ("concatenate", (1, None), 1, 0, False),
    ("if", (1, None), None, 2, False),
    ("while", (1, None), None, 2, False),
    ("call", (0, None), None, 0, False),
    ("runtime_call", (0, None), None, 0, False),
    ("block_call", (0, None), None, 0, False),
    ("return", (0, None), 0, 0, True),
    ("select", 3, 1, 0, False),
    ("gather", 2, 1, 0, False),
    ("scatter", 3, 1, 0, False),
    ("slice", 1, 1, 0, False),
    ("pad", 1, 1, 0, False),
    ("broadcast", 1, 1, 0, False),
    ("reduce_sum", 1, 1, 0, False),
    ("argmax", 1, 1, 0, False),
    ("dot", 2, 1, 0, False),
    ("conv", 2, 1, 0, False),
    ("solve", 2, 1, 0, False),
    ("tril", 1, 1, 0, False),
    ("sparse_from_dense", 1, 2, 0, False),
    ("sparse_add", 4, 2, 0, False),
    ("sparse_concatenate", (4, None), 2, 0, False),
    ("sparse_dot_dense", 3, 1, 0, False),
    ("dense_dot_sparse", 3, 1, 0, False),
    ("sparse_negate", 2, 2, 0, False),
    ("sparse_reduce_sum", 2, 1, 0, False),
]

#: (name, attr, tag, required) attribute-schema spot-checks.
_ATTR_SCHEMA_EXPECTED = [
    ("cast", "dtype", ir.ATTR_DTYPE, True),
    ("transpose", "permutation", ir.ATTR_INTS, False),
    ("slice", "start_indices", ir.ATTR_INTS, True),
    ("slice", "limit_indices", ir.ATTR_INTS, True),
    ("slice", "strides", ir.ATTR_INTS, False),
    ("pad", "padding_config", ir.ATTR_NESTED_INTS, True),
    ("pad", "value", ir.ATTR_FLOAT, False),
    ("argmax", "axis", ir.ATTR_INT, False),
    ("conv", "padding", ir.ATTR_ANY, False),
    ("conv", "strides", ir.ATTR_INTS, False),
    ("conv", "input_dilation", ir.ATTR_INTS, False),
    ("conv", "kernel_dilation", ir.ATTR_INTS, False),
    ("conv", "feature_group_count", ir.ATTR_INT, False),
    ("conv", "batch_group_count", ir.ATTR_INT, False),
    ("constant", "value", ir.ATTR_NDARRAY, True),
    ("runtime_call", "callback", ir.ATTR_STR, True),
    ("runtime_call", "result_specs", ir.ATTR_ANY, True),
    ("block_call", "block_name", ir.ATTR_STR, True),
    ("block_call", "static_args", ir.ATTR_ANY, False),
    ("collective_permute", "source_target_pairs", ir.ATTR_NESTED_INTS, True),
    ("all_gather", "group", ir.ATTR_STR, True),
    ("all_gather", "group_size", ir.ATTR_INT, False),
]
# The reduce_* family shares one schema.
_ATTR_SCHEMA_EXPECTED += [
    (name, "axes", ir.ATTR_INTS, True)
    for name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod")
]
_ATTR_SCHEMA_EXPECTED += [
    (name, "keepdims", ir.ATTR_BOOL, False)
    for name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod")
]
_ATTR_SCHEMA_EXPECTED += [
    (name, "reduce_op", ir.ATTR_STR, True)
    for name in ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod")
]

#: (name, attr, default) — defaults of non-required attributes.
_ATTR_DEFAULT_EXPECTED = [
    ("transpose", "permutation", None),
    ("slice", "strides", None),
    ("pad", "value", 0.0),
    ("reduce_sum", "keepdims", False),
    ("argmax", "axis", None),
    ("conv", "padding", "VALID"),
    ("conv", "strides", None),
    ("conv", "input_dilation", None),
    ("conv", "kernel_dilation", None),
    ("conv", "feature_group_count", 1),
    ("conv", "batch_group_count", 1),
    ("block_call", "static_args", ()),
    ("all_gather", "group_size", None),
]

#: (name, operand_count, expected) — ``OpDef.check_arity`` behavior.
_ARITY_CHECKS = [
    ("add", 2, True),
    ("add", 1, False),
    ("add", 3, False),
    ("constant", 0, True),
    ("constant", 1, False),
    ("rank", 0, True),
    ("concatenate", 0, False),
    ("concatenate", 1, True),
    ("concatenate", 7, True),
    ("if", 0, False),
    ("if", 1, True),
    ("if", 2, True),
    ("call", 0, True),
    ("call", 1, True),
    ("return", 0, True),
    ("return", 3, True),
    ("select", 3, True),
    ("select", 2, False),
    ("slice", 1, True),
    ("slice", 0, False),
]


def _attrs(name):
    return {attr.name: attr for attr in ir.opdef(name).attributes}


# --- 1. registry size and canonical name set ---------------------------------


def test_registry_size_and_canonical_name_set():
    names = ir.op_names()
    assert len(names) == 91
    assert names == tuple(sorted(CANONICAL_NAMES))


# --- 2. per-op category mapping ----------------------------------------------


@pytest.mark.parametrize(
    ("name", "category"),
    _CATEGORY_ITEMS,
    ids=[f"{name}={category}" for name, category in _CATEGORY_ITEMS],
)
def test_op_category(name, category):
    assert ir.opdef(name).category == category


# --- 3. per-op effect mapping (all 91) ---------------------------------------


@pytest.mark.parametrize(
    ("name", "effect"),
    _EFFECT_ITEMS,
    ids=[f"{name}={effect}" for name, effect in _EFFECT_ITEMS],
)
def test_op_effect(name, effect):
    assert ir.opdef(name).effect == effect


# --- broadcast vs broadcast_collective ---------------------------------------


def test_broadcast_and_broadcast_collective_are_distinct_ops():
    shape_op = ir.opdef("broadcast")
    collective_op = ir.opdef("broadcast_collective")
    assert shape_op is not collective_op
    assert shape_op.category == "structure"
    assert collective_op.category == "collective"
    assert shape_op.effect == "pure"
    assert collective_op.effect == "collective"


# --- 4. arity / result count / regions / terminator spot-checks --------------


@pytest.mark.parametrize(
    ("name", "arity", "result_count", "regions", "is_terminator"),
    _SIGNATURE_EXPECTED,
    ids=[params[0] for params in _SIGNATURE_EXPECTED],
)
def test_op_signature(name, arity, result_count, regions, is_terminator):
    opdef = ir.opdef(name)
    assert opdef.arity == arity
    assert opdef.result_count == result_count
    assert opdef.regions == regions
    assert opdef.is_terminator is is_terminator


# --- 5. attribute schemas -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "attr", "attr_type", "required"),
    _ATTR_SCHEMA_EXPECTED,
    ids=[f"{name}.{attr}" for name, attr, _tag, _req in _ATTR_SCHEMA_EXPECTED],
)
def test_attr_schema(name, attr, attr_type, required):
    spec = _attrs(name)[attr]
    assert spec.type == attr_type
    assert spec.required is required


@pytest.mark.parametrize(
    ("name", "attr", "default"),
    _ATTR_DEFAULT_EXPECTED,
    ids=[f"{name}.{attr}" for name, attr, _default in _ATTR_DEFAULT_EXPECTED],
)
def test_attr_defaults(name, attr, default):
    assert _attrs(name)[attr].default == default


# --- 6. shape-inference hook presence ----------------------------------------

_SHAPE_FN_OPS = (
    _ELEMENTWISE_BINARY
    + ("cast",)
    + _ELEMENTWISE_UNARY
    + _COMPARISON
    + _STRUCTURE
    + _REDUCTION
    + _LINALG
)

_NO_SHAPE_FN_OPS = ("constant", "call", "if", "runtime_call", "block_call")


@pytest.mark.parametrize("name", _SHAPE_FN_OPS)
def test_shape_fn_present(name):
    assert ir.opdef(name).shape_fn is not None


@pytest.mark.parametrize("name", _NO_SHAPE_FN_OPS)
def test_shape_fn_absent(name):
    assert ir.opdef(name).shape_fn is None


def test_return_shape_fn_yields_no_results():
    assert ir.opdef("return").shape_fn([], {}) == ()


# --- 7. registry API behavior -------------------------------------------------


def test_opdef_unknown_name_raises_keyerror():
    with pytest.raises(KeyError, match="nope"):
        ir.opdef("nope")


def test_register_duplicate_name_raises_valueerror_before_mutation():
    original = ir.opdef("add")
    fresh = ir.OpDef(
        name="add",
        category="elementwise",
        description="test-only duplicate",
        arity=0,
        result_count=1,
        effect="pure",
    )
    with pytest.raises(ValueError, match="already registered"):
        ir.register_opdef(fresh)
    assert ir.opdef("add") is original


def test_register_new_name_works_and_is_cleaned_up():
    # Registry is global: use a name no other test depends on and remove it
    # afterwards so the canonical 91-op set is never polluted.
    name = "__etl_test_ephemeral_op__"
    assert not ir.has_opdef(name)
    fresh = ir.OpDef(
        name=name,
        category="control",
        description="test-only ephemeral op",
        arity=0,
        result_count=1,
        effect="pure",
    )
    try:
        returned = ir.register_opdef(fresh)
        assert returned is fresh
        assert ir.has_opdef(name)
        assert name in ir.op_names()
        assert ir.opdef(name) is fresh
        assert fresh in ir.all_opdefs()
    finally:
        _op_defs._REGISTRY.pop(name, None)
    assert not ir.has_opdef(name)


def test_has_opdef():
    assert ir.has_opdef("add")
    assert not ir.has_opdef("nope")


def test_op_names_returns_sorted_tuple():
    names = ir.op_names()
    assert isinstance(names, tuple)
    assert names == tuple(sorted(names))


def test_all_opdefs_sorted_instances():
    opdefs = ir.all_opdefs()
    assert len(opdefs) == 91
    assert [opdef.name for opdef in opdefs] == sorted(
        opdef.name for opdef in opdefs
    )
    assert all(isinstance(opdef, ir.OpDef) for opdef in opdefs)


@pytest.mark.parametrize(
    ("name", "count", "expected"),
    _ARITY_CHECKS,
    ids=[f"{name}({count})" for name, count, _expected in _ARITY_CHECKS],
)
def test_check_arity(name, count, expected):
    assert ir.opdef(name).check_arity(count) is expected


# --- 8. attribute type tags ---------------------------------------------------


def test_attr_type_names_contains_all_tags():
    tags = (
        ir.ATTR_BOOL,
        ir.ATTR_INT,
        ir.ATTR_FLOAT,
        ir.ATTR_STR,
        ir.ATTR_DTYPE,
        ir.ATTR_INTS,
        ir.ATTR_FLOATS,
        ir.ATTR_STRS,
        ir.ATTR_NESTED_INTS,
        ir.ATTR_SHAPE,
        ir.ATTR_NDARRAY,
        ir.ATTR_ANY,
    )
    assert len(set(tags)) == 12  # the tags are distinct
    assert set(tags) <= set(ir.ATTR_TYPE_NAMES)
    assert len(ir.ATTR_TYPE_NAMES) == 12
