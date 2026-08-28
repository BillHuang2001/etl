"""etl.bench example registry contract.

``list_examples()`` returns exactly the 97 documented example names in
registry order (op → block → e2e); ``get_example(name)`` returns a frozen
``Example`` dataclass with the 13 documented fields (``name``, ``description``,
``specs`` tuple of ``etl.TensorSpec``, ``graph`` and ``numpy_ref`` callables,
optional ``torch_ref``, per-example ``rtol``/``atol``/``tolerance`` overrides,
``category``, optional ``inputs_fn``, ``tags`` tuple, optional ``runner``) and a
``generate_inputs(seed)`` method returning one numpy array per spec (zero
arrays for constant-only graphs such as ``while_fib``).
``list_categories()`` returns the three categories (``op``/``block``/``e2e``)
in first-appearance order; ``list_tags()`` returns the 16 tags in
first-appearance order. ``expand_names`` (in ``etl.bench.examples``) resolves
entries with precedence category name → exact example name → tag name.
Unknown names raise ``UnknownExampleError`` — a ``ValueError`` subclass
importable from both ``etl.bench`` and ``etl.bench.examples`` — whose message
lists the available names; ``conformance`` raises the same error for unknown
names.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import etl
from etl.bench import (
    UnknownExampleError,
    conformance,
    get_example,
    list_categories,
    list_examples,
    list_tags,
)
from etl.bench.examples import UnknownExampleError as ExamplesUnknownExampleError
from etl.bench.examples import expand_names

# Authoritative registry (registry order = module import order) — documented in
# etl/bench/CONTEXT.md. Regenerate via `python3 -c "import etl.bench;
# print(etl.bench.list_examples())"` and update this list (and CONTEXT.md) when
# examples are added/removed.
EXPECTED_NAMES = [
    # op (73): legacy micro/grad/vectorize/large (24)
    "matmul",
    "conv2d",
    "conv2d_same",
    "conv2d_stride2",
    "elementwise_fusion",
    "softmax",
    "layernorm",
    "mlp",
    "cumsum",
    "attention",
    "grad_mlp",
    "grad_mix",
    "grad_stopgrad",
    "grad_structural",
    "vmap_softmax",
    "vmap_layernorm",
    "vmap_linear",
    "vmap_mlp",
    "vmap_attention",
    "vmap_grad_mlp",
    "matmul_1024",
    "conv2d_large",
    "layernorm_large",
    "vmap_mlp_large",
    # op: op_* (13)
    "op_erf",
    "op_gelu",
    "op_tril_triu",
    "op_argmax_argmin",
    "op_bitwise",
    "op_logical",
    "op_compare",
    "op_power_remainder",
    "op_gather_scatter",
    "op_pad_slice",
    "op_stop_gradient",
    "op_select",
    "op_broadcast",
    # op: matmul_* variants (6)
    "batched_3d_matmul",
    "matmul_3d_2d_shared",
    "matvec",
    "matmul_transposed",
    "matmul_square",
    "diagonal_matmul",
    # op: cond/while control flow (8)
    "cond_basic",
    "cond_multi_output",
    "cond_nested",
    "while_cumsum",
    "while_fib",
    "while_cond_combo",
    "while_poly",
    "cond_while_pipeline",
    # op: grad_* (7)
    "grad_erf",
    "grad_gelu",
    "grad_cumsum",
    "grad_tril",
    "grad_triu",
    "grad_power_frac",
    "grad_bitwise_zero",
    # op: vmap_* (5)
    "vmap_elementwise",
    "vmap_nested",
    "vmap_shared_weights_dot",
    "vmap_pad",
    "vmap_concat",
    # op: custom_* (5)
    "custom_l2norm",
    "custom_l2norm_grad",
    "custom_l2norm_vmap",
    "custom_relu2",
    "custom_relu2_grad",
    # op: xla_* (5)
    "xla_elementwise_chain",
    "xla_softmax",
    "xla_mlp",
    "xla_batched_dot",
    "xla_reductions",
    # block (17)
    "transformer",
    "nbody",
    "mha_block",
    "ffn_block",
    "mha_ffn_block",
    "lstm_cell",
    "gru_cell",
    "resnet_conv_block",
    "conv_block_stride2",
    "deep_mlp_block",
    "mlp_block_residual",
    "pso_step",
    "kmeans_iter",
    "als_step",
    "cg_step",
    "power_iter",
    "softmax_reg_step",
    # e2e (7)
    "e2e_train_mlp",
    "e2e_train_convnet",
    "e2e_train_transformer",
    "e2e_infer_transformer",
    "e2e_pso_optimize",
    "e2e_kmeans",
    "e2e_power_iteration",
]

EXPECTED_CATEGORIES = ["op", "block", "e2e"]

# Authoritative tag list (first-appearance order) — regenerate via
# `python3 -c "import etl.bench; print(etl.bench.list_tags())"`.
EXPECTED_TAGS = [
    "micro",
    "grad",
    "vectorize",
    "large",
    "basic",
    "control-flow",
    "vmap",
    "custom",
    "xla",
    "transformer_block",
    "rnn",
    "conv",
    "mlp_block",
    "optimization",
    "train",
    "infer",
]

# The 11 grad_* examples carry rtol=atol=1e-3 (fd-reference noise).
GRAD_NAMES = [
    "grad_mlp",
    "grad_mix",
    "grad_stopgrad",
    "grad_structural",
    "grad_erf",
    "grad_gelu",
    "grad_cumsum",
    "grad_tril",
    "grad_triu",
    "grad_power_frac",
    "grad_bitwise_zero",
]

# The 8 vectorize examples stay at the strict default tolerances.
VMAP_NAMES = [
    "vmap_softmax",
    "vmap_layernorm",
    "vmap_linear",
    "vmap_mlp",
    "vmap_attention",
    "vmap_grad_mlp",
    "vmap_elementwise",
    "vmap_nested",
    "vmap_shared_weights_dot",
    "vmap_pad",
    "vmap_concat",
]


def test_list_examples_returns_exact_documented_set():
    assert list_examples() == EXPECTED_NAMES
    # Every registered example carries one of the three documented categories,
    # and all three categories are present in the registry.
    categories = {get_example(name).category for name in EXPECTED_NAMES}
    assert categories == set(EXPECTED_CATEGORIES)


def test_list_categories_returns_documented_categories():
    assert list_categories() == EXPECTED_CATEGORIES


def test_list_tags_returns_documented_tags():
    assert list_tags() == EXPECTED_TAGS
    # Every registered example's tags tuple is a sub-multiset of the documented
    # tags: each tag is known and appears at most once. (Order within the
    # example's tuple is NOT asserted against first-appearance order — the four
    # e2e infer examples carry ("infer", "control-flow"), which reverses it.)
    for name in EXPECTED_NAMES:
        tags = get_example(name).tags
        assert isinstance(tags, tuple)
        assert len(tags) == len(set(tags)), f"{name}: duplicate tags {tags}"
        assert set(tags) <= set(EXPECTED_TAGS), f"{name}: unknown tags {tags}"


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_get_example_returns_valid_example(name):
    example = get_example(name)
    assert example.name == name
    assert example.description
    # specs may be empty for constant-only graphs (e.g. while_fib: 3-leaf
    # fibonacci loop with no graph inputs).
    assert isinstance(example.specs, tuple)
    for spec in example.specs:
        assert isinstance(spec, etl.TensorSpec)
        assert all(isinstance(dim, int) for dim in spec.shape)
    assert callable(example.graph)
    assert callable(example.numpy_ref)
    inputs = example.generate_inputs(seed=0)
    assert isinstance(inputs, list)
    assert len(inputs) == len(example.specs)
    for spec, array in zip(example.specs, inputs):
        assert isinstance(array, np.ndarray)
        assert array.shape == tuple(spec.shape)
        assert array.dtype == np.dtype(spec.dtype)


def test_example_is_a_frozen_dataclass():
    example = get_example("matmul")
    assert dataclasses.is_dataclass(example)
    assert [field.name for field in dataclasses.fields(example)] == [
        "name",
        "description",
        "specs",
        "graph",
        "numpy_ref",
        "torch_ref",
        "rtol",
        "atol",
        "tolerance",
        "category",
        "inputs_fn",
        "tags",
        "runner",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.name = "other"


def test_example_per_example_tolerance_overrides():
    # mlp carries a max-abs-error override (iree fp32 noise — documented);
    # matmul falls back to the global defaults (all overrides None).
    assert get_example("mlp").tolerance == 1e-4
    assert get_example("mlp").rtol is None
    assert get_example("mlp").atol is None
    assert get_example("matmul").tolerance is None
    assert get_example("matmul").rtol is None
    assert get_example("matmul").atol is None
    # All 11 grad examples carry rtol=atol=1e-3 (fd-reference noise).
    for name in GRAD_NAMES:
        example = get_example(name)
        assert example.rtol == 1e-3
        assert example.atol == 1e-3
        assert example.tolerance is None
    # e2e train examples carry tolerance=1e-3; the four e2e infer examples
    # fall back to the global defaults (None) — live registry, documented.
    for name in ("e2e_train_mlp", "e2e_train_convnet", "e2e_train_transformer"):
        example = get_example(name)
        assert example.tolerance == 1e-3
        assert example.rtol is None
        assert example.atol is None
    for name in (
        "e2e_infer_transformer",
        "e2e_pso_optimize",
        "e2e_kmeans",
        "e2e_power_iteration",
    ):
        example = get_example(name)
        assert example.tolerance is None
        assert example.rtol is None
        assert example.atol is None
    # block/large per-example atol overrides.
    assert get_example("transformer").tolerance == 1e-3
    assert get_example("nbody").atol == 2e-5
    assert get_example("matmul_1024").atol == 5e-4
    assert get_example("conv2d_large").atol == 2e-4
    assert get_example("vmap_mlp_large").atol == 1.2e-4
    # xla/custom overrides (fd-reference noise on gradient examples).
    assert get_example("xla_mlp").tolerance == 1e-4
    for name in ("custom_l2norm_grad", "custom_relu2_grad"):
        example = get_example(name)
        assert example.rtol == 1e-3
        assert example.atol == 1e-3
        assert example.tolerance is None
    # Vectorize examples stay at the strict defaults.
    for name in VMAP_NAMES:
        example = get_example(name)
        assert example.rtol is None
        assert example.atol is None
        assert example.tolerance is None


def test_get_example_unknown_name_error_lists_available_names():
    with pytest.raises(UnknownExampleError) as excinfo:
        get_example("does_not_exist")
    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert "matmul" in message
    assert "attention" in message


def test_unknown_example_error_importable_from_examples_module():
    assert ExamplesUnknownExampleError is UnknownExampleError


def test_conformance_unknown_name_raises_same_error():
    with pytest.raises(UnknownExampleError) as excinfo:
        conformance(["nope"])
    assert isinstance(excinfo.value, ValueError)
    assert "nope" in str(excinfo.value)


def test_expand_names_unknown_entry_raises():
    with pytest.raises(UnknownExampleError) as excinfo:
        expand_names(["nope"])
    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert message.startswith("unknown example 'nope'; available examples: matmul,")
    assert "matmul" in message


def test_expand_names_category_expands_in_registry_order():
    # Category name expands to all its examples in registry order.
    assert expand_names(["e2e"]) == [
        "e2e_train_mlp",
        "e2e_train_convnet",
        "e2e_train_transformer",
        "e2e_infer_transformer",
        "e2e_pso_optimize",
        "e2e_kmeans",
        "e2e_power_iteration",
    ]
    assert expand_names(["block"]) == [
        "transformer",
        "nbody",
        "mha_block",
        "ffn_block",
        "mha_ffn_block",
        "lstm_cell",
        "gru_cell",
        "resnet_conv_block",
        "conv_block_stride2",
        "deep_mlp_block",
        "mlp_block_residual",
        "pso_step",
        "kmeans_iter",
        "als_step",
        "cg_step",
        "power_iter",
        "softmax_reg_step",
    ]


def test_expand_names_exact_name_wins_over_tag():
    # Exact example names are matched before tag names ("mlp"/"transformer"
    # are also tags, but resolve to the single example).
    assert expand_names(["mlp"]) == ["mlp"]
    assert expand_names(["transformer"]) == ["transformer"]


def test_expand_names_tag_expands_to_exact_lists():
    assert expand_names(["grad"]) == GRAD_NAMES
    assert expand_names(["control-flow"]) == [
        "cond_basic",
        "cond_multi_output",
        "cond_nested",
        "while_cumsum",
        "while_fib",
        "while_cond_combo",
        "while_poly",
        "cond_while_pipeline",
        "e2e_infer_transformer",
        "e2e_pso_optimize",
        "e2e_kmeans",
        "e2e_power_iteration",
    ]
    assert expand_names(["large"]) == [
        "matmul_1024",
        "conv2d_large",
        "layernorm_large",
        "vmap_mlp_large",
        "transformer",
        "nbody",
    ]
    # The "vectorize" tag covers only the six legacy vectorize examples; the
    # "vmap" tag covers only the five newer vmap_* op examples.
    assert expand_names(["vectorize"]) == [
        "vmap_softmax",
        "vmap_layernorm",
        "vmap_linear",
        "vmap_mlp",
        "vmap_attention",
        "vmap_grad_mlp",
    ]
    assert expand_names(["vmap"]) == [
        "vmap_elementwise",
        "vmap_nested",
        "vmap_shared_weights_dot",
        "vmap_pad",
        "vmap_concat",
    ]


def test_expand_names_tag_selection_resolution():
    # Tag selection (what conformance uses) keeps registry order and covers
    # only the matching examples.
    expanded = expand_names(["control-flow"])
    assert "cond_basic" in expanded
    assert "while_fib" in expanded
    assert "e2e_kmeans" in expanded
    assert "matmul" not in expanded
    assert expanded == sorted(expanded, key=EXPECTED_NAMES.index)


def test_get_example_runner_field():
    # The three e2e_train_* examples carry an executable runner; every other
    # example's runner is None.
    for name in ("e2e_train_mlp", "e2e_train_convnet", "e2e_train_transformer"):
        example = get_example(name)
        assert callable(example.runner)
        assert example.category == "e2e"
        assert example.tags == ("train",)
    assert get_example("matmul").runner is None
    assert get_example("e2e_infer_transformer").category == "e2e"
    assert get_example("e2e_infer_transformer").tags == ("infer", "control-flow")
