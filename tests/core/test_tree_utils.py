"""Tests for the etl tree utility API: ``tree_map`` / ``tree_leaves`` /
``tree_structure`` / ``tree_flatten`` / ``tree_unflatten``.

Authoritative source: ``etl/core/tree.py``. Contract under test:

- ``tree_map(fn, *trees)``: with a single tree, applies ``fn`` to every leaf
  and rebuilds the same structure (same container types). With multiple
  trees, the structures are validated FIRST (``fn`` is never called on a
  mismatch); a structure mismatch raises ``TypeError`` with message
  ``tree_map: trees do not have the same structure — first mismatch at
  pytree path {path}: expected {spec}, got {spec}`` (path rendered like
  ``[0]['weights'][1]``).
- ``tree_leaves(tree) -> list`` of leaves in pre-order; a scalar is a
  single leaf; empty containers yield no leaves.
- ``tree_structure(tree) -> TreeSpec`` (``type`` / ``node_data`` /
  ``num_leaves`` follow ``flatten``).
- ``tree_flatten`` is an alias of ``flatten``; ``tree_unflatten`` is an
  alias of ``unflatten`` (identity aliases with identical results).
- All five names are exported from ``etl``, ``etl.core`` and
  ``etl.core.tree`` — the very same objects.
"""

from collections import namedtuple
from dataclasses import dataclass

import pytest

import etl
import etl.core
from etl.core import flatten, unflatten
from etl.core import tree as tree_module
from etl.core.tree import (
    TreeSpec,
    register_pytree_node,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)

# ---------------------------------------------------------------------------
# Test-local pytree types
# ---------------------------------------------------------------------------

Point = namedtuple("Point", "x y")
EmptyNT = namedtuple("EmptyNT", [])


@dataclass(frozen=True)
class FrozenPoint:
    x: float
    y: float


class Box:
    """Custom pytree node: children in ``.items``, metadata in ``.label``."""

    def __init__(self, label, items):
        self.label = label
        self.items = list(items)

    def __eq__(self, other):
        return (
            isinstance(other, Box)
            and self.label == other.label
            and self.items == other.items
        )


def _box_flatten(obj):
    return list(obj.items), obj.label


def _box_unflatten(label, children):
    return Box(label, children)


# Registered once at import time; used by the custom-type tests below.
register_pytree_node(Box, _box_flatten, _box_unflatten)


def _boom(*args):
    raise AssertionError("fn must not be called")


# ---------------------------------------------------------------------------
# Public export surface
# ---------------------------------------------------------------------------

TREE_UTIL_NAMES = (
    "tree_map",
    "tree_leaves",
    "tree_structure",
    "tree_flatten",
    "tree_unflatten",
)


def test_tree_utils_exported_from_etl_etl_core_and_tree_module():
    # Import the modules and assert the SAME objects are reachable everywhere.
    for name in TREE_UTIL_NAMES:
        assert getattr(etl, name) is getattr(etl.core, name)
        assert getattr(etl.core, name) is getattr(tree_module, name)


# ---------------------------------------------------------------------------
# tree_map: single tree
# ---------------------------------------------------------------------------

TREE_MAP_CASES = [
    pytest.param(
        {"a": 1, "b": [2, (3, 4)]},
        {"a": 2, "b": [3, (4, 5)]},
        id="nested-dict",
    ),
    pytest.param([1, [2, 3], (4, 5)], [2, [3, 4], (5, 6)], id="nested-list"),
    pytest.param(
        Point(1, Point(2, 3)), Point(2, Point(3, 4)), id="namedtuple-nested"
    ),
    pytest.param(FrozenPoint(1.5, 2.5), FrozenPoint(2.5, 3.5), id="dataclass"),
    pytest.param(
        Box("w", [1, [2, 3]]), Box("w", [2, [3, 4]]), id="registered-custom"
    ),
]


@pytest.mark.parametrize("obj, expected", TREE_MAP_CASES)
def test_tree_map_single_tree_rebuilds_same_structure(obj, expected):
    out = tree_map(lambda x: x + 1, obj)
    assert out == expected
    assert type(out) is type(obj)


def test_tree_map_single_tree_applies_fn_to_every_leaf():
    calls = []

    def record(x):
        calls.append(x)
        return x * 2

    out = tree_map(record, {"a": [1, (2, 3)], "b": Point(4, 5)})
    assert calls == [1, 2, 3, 4, 5]
    assert out == {"a": [2, (4, 6)], "b": Point(8, 10)}


def test_tree_map_changes_leaf_types():
    assert tree_map(float, [1, 2]) == [1.0, 2.0]
    assert tree_map(lambda x: (x, x), [1, 2]) == [(1, 1), (2, 2)]
    assert tree_map(lambda x: (x, x), {"a": 1}) == {"a": (1, 1)}


@pytest.mark.parametrize("empty", [{}, (), []], ids=["dict", "tuple", "list"])
def test_tree_map_empty_containers_never_call_fn(empty):
    out = tree_map(_boom, empty)
    assert out == empty
    assert type(out) is type(empty)


def test_tree_map_single_leaf_tree():
    assert tree_map(str, 5) == "5"
    assert tree_map(lambda x: x, None) is None


# ---------------------------------------------------------------------------
# tree_map: multiple trees (zip corresponding leaves)
# ---------------------------------------------------------------------------

def test_tree_map_two_trees_zip_corresponding_leaves():
    assert (
        tree_map(
            lambda x, y: x + y,
            {"a": 1, "b": [2, 3]},
            {"a": 10, "b": [20, 30]},
        )
        == {"a": 11, "b": [22, 33]}
    )
    assert (
        tree_map(lambda x, y: x + y, Point(1, 2), Point(10, 20))
        == Point(11, 22)
    )
    assert tree_map(lambda x, y: (x, y), [1, 2], [3, 4]) == [(1, 3), (2, 4)]


def test_tree_map_three_trees_zip_corresponding_leaves():
    assert (
        tree_map(lambda x, y, z: x + y + z, [1, 2], [3, 4], [5, 6])
        == [9, 12]
    )
    assert (
        tree_map(lambda x, y, z: (x, y, z), {"k": 1}, {"k": 2}, {"k": 3})
        == {"k": (1, 2, 3)}
    )


def test_tree_map_empty_same_type_containers():
    assert tree_map(lambda x, y: x + y, {}, {}) == {}
    assert tree_map(lambda x, y: x + y, [], []) == []
    assert tree_map(lambda x, y: x + y, (), ()) == ()


# ---------------------------------------------------------------------------
# tree_map: multi-tree structure mismatch
# ---------------------------------------------------------------------------

def test_tree_map_mismatched_dict_keys_raises_with_path():
    # Dict-key mismatches report the dict NODE's path: the node types match
    # but the sorted key lists (node_data) diverge, so the first mismatch is
    # at the dict node itself — here the root, path ().
    with pytest.raises(
        TypeError,
        match=r"tree_map: trees do not have the same structure — first mismatch at pytree path \(\): expected dict with keys \['a', 'b'\], got dict with keys \['a', 'c'\]",
    ):
        tree_map(_boom, {"a": 1, "b": 2}, {"a": 1, "c": 2})


def test_tree_map_nested_mismatch_raises_with_path():
    # Leaf vs container at [0]['weights'].
    with pytest.raises(
        TypeError,
        match=r"tree_map: trees do not have the same structure — first mismatch at pytree path \[0\]\['weights'\]: expected .*, got .*",
    ):
        tree_map(_boom, [{"weights": [1, 2]}], [{"weights": 5}])


def test_tree_map_leaf_count_mismatch_raises():
    # The message PREFIX is pinned; the exact path for a child-count
    # mismatch is not (ambiguous rendering), so only the prefix is asserted.
    with pytest.raises(
        TypeError,
        match=r"tree_map: trees do not have the same structure — first mismatch at pytree path ",
    ):
        tree_map(_boom, [1, 2], [1, 2, 3])


def test_tree_map_container_type_mismatch_raises():
    # Empty dict vs empty tuple: zero leaves in both, but the node TYPES
    # differ — the contract says this is a mismatch (TypeError).
    with pytest.raises(TypeError):
        tree_map(_boom, {}, ())


# ---------------------------------------------------------------------------
# tree_leaves / tree_structure
# ---------------------------------------------------------------------------

def test_tree_leaves_preorder():
    assert tree_leaves([1, [2, 3]]) == [1, 2, 3]
    assert tree_leaves({"x": [1, 2], "y": (3, [4])}) == [1, 2, 3, 4]
    assert tree_leaves(Point(1, Point(2, 3))) == [1, 2, 3]


@pytest.mark.parametrize(
    "scalar", [5, None, "abc", b"bytes", 1.5], ids=repr
)
def test_tree_leaves_scalar_is_single_leaf(scalar):
    assert tree_leaves(scalar) == [scalar]


@pytest.mark.parametrize(
    "empty", [{}, (), [], EmptyNT()], ids=["dict", "tuple", "list", "namedtuple"]
)
def test_tree_leaves_empty_container(empty):
    assert tree_leaves(empty) == []


def test_tree_structure_returns_treespec():
    spec = tree_structure({"b": 1, "a": 2})
    assert isinstance(spec, TreeSpec)
    assert spec.type is dict
    assert spec.num_leaves == 2
    assert spec.node_data == ["a", "b"]


def test_tree_structure_num_leaves_matches_flatten():
    for obj in ([1, [2, 3]], {}, (), 5, None, Point(1, 2), Box("t", [1, 2])):
        assert tree_structure(obj).num_leaves == flatten(obj)[1].num_leaves


def test_tree_structure_matches_flatten_spec_fields():
    # Compare fields, not spec ``==`` (TreeSpec equality is not pinned).
    obj = {"a": [1, 2], "b": Point(3, 4)}
    spec = tree_structure(obj)
    flat_spec = flatten(obj)[1]
    assert spec.type is flat_spec.type
    assert spec.num_leaves == flat_spec.num_leaves
    assert spec.node_data == flat_spec.node_data
    assert spec.context == flat_spec.context
    assert len(spec.children) == len(flat_spec.children)


# ---------------------------------------------------------------------------
# tree_flatten / tree_unflatten aliases
# ---------------------------------------------------------------------------

ALIAS_STRUCTURES = [
    pytest.param([1, 2], id="list"),
    pytest.param({"a": 1, "b": (2, 3)}, id="nested-dict"),
    pytest.param(Point(1, 2), id="namedtuple"),
    pytest.param(FrozenPoint(1.5, 2.5), id="dataclass"),
]


@pytest.mark.parametrize("obj", ALIAS_STRUCTURES)
def test_tree_flatten_alias_matches_flatten(obj):
    leaves, spec = tree_flatten(obj)
    flat_leaves, flat_spec = flatten(obj)
    assert leaves == flat_leaves
    assert spec.num_leaves == flat_spec.num_leaves
    assert spec.type is flat_spec.type
    assert spec.node_data == flat_spec.node_data
    assert spec.context == flat_spec.context
    assert len(spec.children) == len(flat_spec.children)


@pytest.mark.parametrize("obj", ALIAS_STRUCTURES)
def test_tree_unflatten_alias_matches_unflatten(obj):
    leaves, spec = flatten(obj)
    assert tree_unflatten(leaves, spec) == unflatten(leaves, spec)


def test_tree_aliases_are_identity_aliases():
    # Literal alias reading: the exported names ARE flatten/unflatten.
    assert tree_flatten is flatten
    assert tree_unflatten is unflatten


# ---------------------------------------------------------------------------
# Round-trip via the tree utility API
# ---------------------------------------------------------------------------

ROUNDTRIP_CASES = [
    pytest.param([1, [2, 3], (4, 5)], id="nested-list-tuple"),
    pytest.param({"a": [1, (2, 3)], "b": {"c": Point(7, 8)}}, id="mixed-nesting"),
    pytest.param(Point(1, Point(2, 3)), id="namedtuple-nested"),
    pytest.param(FrozenPoint(1.5, 2.5), id="dataclass"),
    pytest.param(Box("w", [1, [2, 3]]), id="registered-custom"),
    pytest.param({}, id="empty-dict"),
    pytest.param((), id="empty-tuple"),
    pytest.param(5, id="scalar-int"),
    pytest.param(None, id="none"),
]


@pytest.mark.parametrize("obj", ROUNDTRIP_CASES)
def test_tree_unflatten_roundtrip(obj):
    rebuilt = tree_unflatten(tree_leaves(obj), tree_structure(obj))
    assert rebuilt == obj
    assert type(rebuilt) is type(obj)
