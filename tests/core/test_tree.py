"""Tests for etl pytrees: ``flatten`` / ``unflatten`` / ``TreeSpec`` /
``register_pytree_node``.

Authoritative source: ``etl/core/tree.py``. Contract under test:

- ``flatten(obj) -> (leaves, treespec)`` collects leaves in pre-order;
  ``unflatten(leaves, treespec)`` is the exact inverse (signature order:
  leaves first, treespec second).
- Built-in container nodes: ``tuple``, ``list``, ``dict`` (keys sorted for
  determinism), ``namedtuple`` instances, *user-defined* ``dataclass``
  instances (fields as children). etl's own dataclass value types
  (``TensorSpec``, ``Device``, ``TreeSpec``, ...) are LEAVES; ``Tensor`` is
  a plain class and therefore naturally a leaf too.
- Anything else (``None``, scalars, str/bytes, ndarrays, ...) is a leaf.
- ``TreeSpec`` is frozen (``FrozenInstanceError`` on assignment) with fields
  ``type`` / ``children`` / ``context`` / ``node_data``; ``node_data`` holds
  the sorted dict keys (list) or field names (tuple for namedtuple, list for
  dataclass). ``num_leaves`` property; ``children_specs`` aliases
  ``children``.
- ``register_pytree_node(type, flatten_fn, unflatten_fn)``: re-registration
  *replaces*; non-type ``node_type`` or non-callable fns raise ``TypeError``.
- ``unflatten`` raises ``ValueError`` on a leaf-count mismatch and never
  mutates the caller's leaves list.
"""

import dataclasses
from collections import namedtuple
from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from etl.core import Device, Tensor, TensorSpec, flatten, unflatten
from etl.core.tree import TreeSpec, register_pytree_node

# ---------------------------------------------------------------------------
# Test-local pytree types
# ---------------------------------------------------------------------------

Point = namedtuple("Point", "x y")
EmptyNT = namedtuple("EmptyNT", [])


@dataclass(frozen=True)
class FrozenPoint:
    x: float
    y: float


@dataclass
class MutablePoint:
    x: float
    y: float


class Tagged:
    """Custom pytree node: children in ``.body``, metadata in ``.tag``."""

    def __init__(self, tag, body):
        self.tag = tag
        self.body = list(body)

    def __eq__(self, other):
        return (
            isinstance(other, Tagged)
            and self.tag == other.tag
            and self.body == other.body
        )


def _tagged_flatten(obj):
    return list(obj.body), obj.tag


def _tagged_unflatten(tag, children):
    return Tagged(tag, children)


# Registered once at import time; used by the custom-type tests below.
register_pytree_node(Tagged, _tagged_flatten, _tagged_unflatten)

# ---------------------------------------------------------------------------
# Round-trip cases (pytest params, small and fast)
# ---------------------------------------------------------------------------

ROUNDTRIP_CASES = [
    pytest.param([1, 2, 3], id="list"),
    pytest.param([[1, [2, 3]], (4, 5)], id="nested-list-tuple"),
    pytest.param((), id="empty-tuple"),
    pytest.param([], id="empty-list"),
    pytest.param({}, id="empty-dict"),
    pytest.param(EmptyNT(), id="empty-namedtuple"),
    pytest.param({"a": 1, "b": 2}, id="dict"),
    pytest.param({"outer": {"inner": [1, 2]}, "t": (3, 4)}, id="nested-dict"),
    pytest.param(Point(1, 2), id="namedtuple"),
    pytest.param(FrozenPoint(1.5, 2.5), id="frozen-dataclass"),
    pytest.param(MutablePoint(1.5, 2.5), id="dataclass"),
    pytest.param(
        {"a": [1, (2, 3)], "b": {"c": Point(7, 8)}}, id="mixed-nesting"
    ),
    pytest.param(Tagged("t", [1, [2, 3]]), id="registered-custom-type"),
    pytest.param(
        [Tagged("a", [1]), {"k": Tagged("b", [2, 3])}],
        id="custom-type-nested",
    ),
    # Leaves (note: str — including "" — is a leaf, not a char container).
    pytest.param(0, id="int-0"),
    pytest.param(1, id="int"),
    pytest.param(-3, id="negative-int"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param("abc", id="str"),
    pytest.param("", id="empty-str"),
    pytest.param(b"bytes", id="bytes"),
    pytest.param(np.array(1.0), id="0d-ndarray"),
    pytest.param(None, id="none"),
]


def test_flatten_returns_leaves_and_spec():
    leaves, spec = flatten([1, (2, 3)])
    assert isinstance(leaves, list)
    assert isinstance(spec, TreeSpec)


@pytest.mark.parametrize("obj", ROUNDTRIP_CASES)
def test_roundtrip(obj):
    leaves, spec = flatten(obj)
    assert spec.num_leaves == len(leaves)
    rebuilt = unflatten(leaves, spec)  # signature: unflatten(leaves, treespec)
    assert rebuilt == obj
    assert type(rebuilt) is type(obj)


def test_dict_keys_sorted_deterministically():
    # Insertion order b, a, c — leaves must come out in sorted-key order.
    d = {"b": 1, "a": 2, "c": 3}
    leaves, spec = flatten(d)
    assert leaves == [2, 1, 3]
    assert spec.node_data == ["a", "b", "c"]
    assert unflatten(leaves, spec) == d

    d2 = {2: "two", 1: "one", 3: "three"}
    leaves2, spec2 = flatten(d2)
    assert leaves2 == ["one", "two", "three"]
    assert spec2.node_data == [1, 2, 3]
    assert unflatten(leaves2, spec2) == d2

    # Nested dicts sort at every level.
    d3 = {"b": {"y": 2, "x": 1}, "a": 0}
    leaves3, spec3 = flatten(d3)
    assert leaves3 == [0, 1, 2]
    assert spec3.node_data == ["a", "b"]
    assert spec3.children[1].node_data == ["x", "y"]


def test_preorder_leaf_order():
    assert flatten([1, [2, 3]])[0] == [1, 2, 3]
    assert flatten({"x": [1, 2], "y": (3, [4])})[0] == [1, 2, 3, 4]
    assert flatten(Point(1, Point(2, 3)))[0] == [1, 2, 3]


NUM_LEAVES_CASES = [
    pytest.param(5, 1, id="scalar-int"),
    pytest.param(None, 1, id="none"),
    pytest.param("", 1, id="empty-str"),
    pytest.param("abc", 1, id="str"),
    pytest.param((), 0, id="empty-tuple"),
    pytest.param([], 0, id="empty-list"),
    pytest.param({}, 0, id="empty-dict"),
    pytest.param(EmptyNT(), 0, id="empty-namedtuple"),
    pytest.param([1, [2, 3]], 3, id="nested-list"),
    pytest.param({"a": 1, "b": (2, 3)}, 3, id="dict-of-tuple"),
    pytest.param(Point(1, 2), 2, id="namedtuple"),
    pytest.param(FrozenPoint(1, 2), 2, id="dataclass"),
    pytest.param(
        {"a": [1, (2, 3)], "b": {"c": Point(7, 8)}}, 5, id="mixed-nesting"
    ),
    pytest.param(Tagged("t", [1, [2, 3]]), 3, id="registered-custom-type"),
]


@pytest.mark.parametrize("obj, expected", NUM_LEAVES_CASES)
def test_num_leaves(obj, expected):
    leaves, spec = flatten(obj)
    assert spec.num_leaves == expected
    assert spec.num_leaves == len(leaves)


def test_treespec_fields_for_dict():
    leaves, spec = flatten({"b": 1, "a": 2})
    assert spec.type is dict
    assert spec.context is None
    assert spec.node_data == ["a", "b"]
    assert len(spec.children) == 2
    assert all(isinstance(child, TreeSpec) for child in spec.children)
    assert spec.children_specs == spec.children


def test_treespec_fields_for_namedtuple():
    _, spec = flatten(Point(1, 2))
    assert spec.type is Point
    assert spec.context is None
    assert spec.node_data == ("x", "y")  # namedtuple keeps _fields as tuple
    assert len(spec.children) == 2


def test_treespec_fields_for_dataclass():
    _, spec = flatten(FrozenPoint(1, 2))
    assert spec.type is FrozenPoint
    assert spec.context is None
    assert spec.node_data == ["x", "y"]
    assert len(spec.children) == 2


def test_treespec_is_frozen():
    _, spec = flatten([1, 2])
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.children = ()
    with pytest.raises(FrozenInstanceError):
        spec.node_data = None


def test_etl_value_types_are_leaves():
    # TensorSpec: an etl-module dataclass → leaf, not a container.
    spec_obj = TensorSpec((2, 3), np.float32)
    leaves, spec = flatten(spec_obj)
    assert leaves == [spec_obj] and leaves[0] is spec_obj
    assert spec.num_leaves == 1
    assert unflatten(leaves, spec) == spec_obj

    # Tensor: a plain class → naturally a leaf.
    tensor_obj = Tensor(np.zeros((2, 2)))
    leaves, spec = flatten(tensor_obj)
    assert leaves == [tensor_obj] and leaves[0] is tensor_obj
    assert spec.num_leaves == 1
    assert unflatten(leaves, spec) == tensor_obj

    # Device: an etl-module dataclass → leaf.
    device_obj = Device("cpu", 0)
    leaves, spec = flatten(device_obj)
    assert leaves == [device_obj] and leaves[0] is device_obj
    assert spec.num_leaves == 1

    # TreeSpec itself is an etl-module dataclass → leaf.
    tree_spec_obj = TreeSpec(type=int)
    leaves, spec = flatten(tree_spec_obj)
    assert leaves == [tree_spec_obj] and leaves[0] is tree_spec_obj
    assert spec.num_leaves == 1


def test_etl_value_types_nested_in_pytree():
    spec_obj = TensorSpec((2, 3), np.float32)
    tensor_obj = Tensor(np.zeros((1,)))
    device_obj = Device("cpu", 0)
    obj = {"spec": spec_obj, "t": (tensor_obj, device_obj)}
    leaves, spec = flatten(obj)
    assert spec.num_leaves == 3
    assert leaves[0] is spec_obj
    assert leaves[1] is tensor_obj
    assert leaves[2] is device_obj
    assert unflatten(leaves, spec) == obj


def test_custom_registered_type_roundtrip():
    obj = Tagged("weight=0.5", [1, [2, 3]])
    leaves, spec = flatten(obj)
    assert leaves == [1, 2, 3]
    assert spec.type is Tagged
    assert spec.context == "weight=0.5"
    assert spec.num_leaves == 3
    assert unflatten(leaves, spec) == obj


def test_custom_registered_type_nested():
    obj = [Tagged("a", [1]), {"k": Tagged("b", [2, 3])}]
    leaves, spec = flatten(obj)
    assert leaves == [1, 2, 3]
    assert unflatten(leaves, spec) == obj


class ReTagged:
    """Local type for re-registration tests (replaces previous fns)."""

    def __init__(self, body):
        self.body = list(body)

    def __eq__(self, other):
        return isinstance(other, ReTagged) and self.body == other.body


def _first_flatten(obj):
    return list(obj.body), "first"


def _first_unflatten(context, children):
    assert context == "first"
    return ReTagged(children)


def _second_flatten(obj):
    return list(obj.body), "second"


def _second_unflatten(context, children):
    assert context == "second"
    return ReTagged(children)


def test_reregistration_replaces_previous_fns():
    # Source contract: re-registering a type REPLACES the old fns (no error).
    register_pytree_node(ReTagged, _first_flatten, _first_unflatten)
    leaves, spec = flatten(ReTagged([1, 2]))
    assert spec.context == "first"
    assert unflatten(leaves, spec) == ReTagged([1, 2])

    register_pytree_node(ReTagged, _second_flatten, _second_unflatten)
    leaves, spec = flatten(ReTagged([1, 2]))
    assert spec.context == "second"
    # _second_unflatten asserts the new context, proving the new fns are used.
    assert unflatten(leaves, spec) == ReTagged([1, 2])


class _ErrorProbe:
    pass


@pytest.mark.parametrize(
    "bad_type", ["notatype", 42, None, 3.14], ids=repr
)
def test_register_pytree_node_rejects_non_type(bad_type):
    with pytest.raises(TypeError):
        register_pytree_node(bad_type, _tagged_flatten, _tagged_unflatten)


@pytest.mark.parametrize(
    "flatten_fn, unflatten_fn",
    [
        ("nope", _tagged_unflatten),
        (_tagged_flatten, "nope"),
        (None, None),
    ],
    ids=["bad-flatten-fn", "bad-unflatten-fn", "both-non-callable"],
)
def test_register_pytree_node_rejects_non_callable(flatten_fn, unflatten_fn):
    with pytest.raises(TypeError):
        register_pytree_node(_ErrorProbe, flatten_fn, unflatten_fn)


def test_unflatten_leaf_count_mismatch_raises():
    leaf = TreeSpec(type=int)
    two_leaf_list = TreeSpec(type=list, children=(leaf, leaf))
    with pytest.raises(ValueError):
        unflatten([1], two_leaf_list)  # too few leaves
    with pytest.raises(ValueError):
        unflatten([1, 2, 3], two_leaf_list)  # too many leaves
    assert unflatten([1, 2], two_leaf_list) == [1, 2]


def test_unflatten_does_not_mutate_leaves():
    leaves = [1, 2, 3]
    spec = TreeSpec(
        type=list,
        children=(
            TreeSpec(type=int),
            TreeSpec(type=int),
            TreeSpec(type=int),
        ),
    )
    assert unflatten(leaves, spec) == [1, 2, 3]
    assert leaves == [1, 2, 3]
