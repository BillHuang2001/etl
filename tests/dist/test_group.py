"""Tests for ``etl.dist.Group`` — static communication-group values.

Contract under test: ``etl/dist/CONTEXT.md`` ("Group semantics", "Error
behavior"). A :class:`Group` is a static Python value in etl's value model:
validated at construction, immutable, hashable, and equal over
``(name, ranks, backend)``. ``ranks=None`` ⇔ the world group (all ranks of
the runtime execution context — membership only known at run time).
"""

import pytest

from etl.dist import WORLD_GROUP, Group, group


# ---------------------------------------------------------------------------
# Construction & attributes
# ---------------------------------------------------------------------------

def test_explicit_group_attributes():
    g = group("data", (0, 1, 2, 3), backend="numpy")
    assert isinstance(g, Group)
    assert g.name == "data"
    assert g.ranks == (0, 1, 2, 3)
    assert g.backend == "numpy"
    assert not g.is_world


def test_world_group_attributes():
    assert WORLD_GROUP.name == "world"
    assert WORLD_GROUP.ranks is None
    assert WORLD_GROUP.backend is None
    assert WORLD_GROUP.is_world


def test_group_without_rank_zero_is_legal():
    # No constraint that rank 0 must be a member.
    g = group("g", (1, 2, 3))
    assert g.ranks == (1, 2, 3)
    assert 0 not in g
    assert g.size() == 3


def test_group_rejects_int_world_size_form():
    # Contract: group() takes a ranks *tuple* only; an int world-size is NOT
    # a supported shorthand (use Group(name, None) / WORLD_GROUP instead).
    with pytest.raises(TypeError):
        group("g", 4)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", 5, None, b"g"])
def test_group_rejects_invalid_name(name):
    with pytest.raises(ValueError, match="name"):
        group(name, (0, 1))


@pytest.mark.parametrize(
    "ranks",
    [
        (),  # empty
        (1.5,),  # non-int
        ("a",),  # non-int
        (-1,),  # negative
        (0, -2),  # negative among valid entries
        (True,),  # bool is not a rank
        (False,),  # bool is not a rank
        (0, True),  # bool among valid entries
        (0, 0),  # duplicate
        (1, 2, 1),  # duplicate among valid entries
    ],
)
def test_group_rejects_invalid_ranks(ranks):
    with pytest.raises(ValueError, match="ranks"):
        group("g", ranks)


# ---------------------------------------------------------------------------
# Equality / hash — no global name registry
# ---------------------------------------------------------------------------

def test_no_global_name_registry():
    # Per contract there is no registry: two groups with the same name but
    # different ranks are BOTH constructible and compare unequal.
    a = group("data", (0, 1))
    b = group("data", (2, 3))
    assert a != b
    assert not (a == b)
    assert len({a, b}) == 2


def test_equality_covers_name_ranks_backend():
    base = group("g", (0, 1))
    assert base == group("g", (0, 1))
    assert base != group("h", (0, 1))  # different name
    assert base != group("g", (0, 2))  # different ranks
    assert base != group("g", (0, 1, 2))  # different size
    assert base != group("g", (0, 1), backend="x")  # backend None vs "x"
    assert group("g", (0, 1), backend="x") != group("g", (0, 1), backend="y")
    # Non-Group comparison falls back cleanly.
    assert not (base == "g")
    assert base != "g"
    assert not (base == 5)


def test_group_is_hashable_static_value():
    g = group("g", (0, 1))
    # Equal groups hash equal and serve as interchangeable dict keys.
    assert hash(g) == hash(group("g", (0, 1)))
    d = {g: "value"}
    assert d[group("g", (0, 1))] == "value"
    # Set semantics.
    assert len({g, group("g", (0, 1))}) == 1
    assert len({g, group("g", (0, 2))}) == 2


def test_group_is_immutable():
    g = group("g", (0, 1))
    with pytest.raises(AttributeError):
        g.name = "other"
    with pytest.raises(AttributeError):
        g.ranks = (1, 2)
    with pytest.raises(AttributeError):
        g.extra = 1
    assert not hasattr(g, "__dict__")


# ---------------------------------------------------------------------------
# World-group defaulting
# ---------------------------------------------------------------------------

def test_group_without_backend_has_none_backend():
    assert group("g", (0, 1)).backend is None


def test_world_group_defaulting():
    # Group("world", None) IS the world group (the default group=None of
    # every collective), by equality over (name, ranks, backend).
    w = Group("world", None)
    assert w == WORLD_GROUP
    assert WORLD_GROUP == w
    assert hash(w) == hash(WORLD_GROUP)


# ---------------------------------------------------------------------------
# size() and __contains__ semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ranks,expected", [((0, 1, 2), 3), ((7,), 1)])
def test_explicit_group_size_is_len_ranks(ranks, expected):
    assert group("g", ranks).size() == expected


def test_world_group_size_unresolved_returns_none():
    assert WORLD_GROUP.size() is None


@pytest.mark.parametrize("world_size", [1, 4])
def test_world_group_size_resolves(world_size):
    assert WORLD_GROUP.size(world_size) == world_size


@pytest.mark.parametrize("world_size", [0, -1, 1.5, True])
def test_world_group_size_rejects_invalid(world_size):
    with pytest.raises(ValueError, match="world_size"):
        WORLD_GROUP.size(world_size)


def test_explicit_group_membership():
    g = group("g", (0, 2))
    assert 0 in g
    assert 2 in g
    assert 1 not in g
    assert -1 not in g
    assert 1.5 not in g
    assert True not in g  # bool is not a rank
    assert "0" not in g


@pytest.mark.parametrize("rank", [0, 1, 10**6, 2**63])
def test_world_group_contains_any_non_negative_int(rank):
    assert rank in WORLD_GROUP


@pytest.mark.parametrize("rank", [-1, -10, 1.5, True, False, "0", None])
def test_world_group_rejects_non_ranks(rank):
    assert rank not in WORLD_GROUP


def test_repr_includes_name_ranks_backend():
    r = repr(group("g", (0, 1), backend="x"))
    assert "g" in r
    assert "(0, 1)" in r
    assert "x" in r
