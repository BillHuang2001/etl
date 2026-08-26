"""TreeSpec — pytree flatten/unflatten.

Structured I/O (tuple/list/dict/namedtuple/dataclass) is supported
everywhere in etl: trace inputs, run outputs, bind. ``TreeSpec`` is the
structural description that makes flattening reversible.

Built-in node types: ``tuple``, ``list``, ``dict`` (keys sorted for
determinism), ``namedtuple`` instances, ``dataclass`` instances. Custom
containers register via ``register_pytree_node``.

Invariants (binding):
- ``flatten`` returns ``(leaves, treespec)`` in pre-order; ``unflatten`` is
  its exact inverse: ``unflatten(flatten(x)) == x`` structurally.
- Leaves are anything that is not a recognized container type.
- Dict keys are sorted (``sorted(keys)``) so specs are order-stable and
  hashable-consistent.
- Treespecs compare structurally (frozen dataclass equality).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Tuple

__all__ = ["TreeSpec", "flatten", "unflatten", "register_pytree_node"]

# Registered custom pytree node types → (flatten_fn, unflatten_fn).
# flatten_fn(obj) -> (children, context); unflatten_fn(context, children) -> obj.
_PYTREE_NODE_REGISTRY: Dict[type, Tuple[Callable, Callable]] = {}


@dataclass(frozen=True)
class TreeSpec:
    """Structural description of a pytree (frozen; structural equality).

    Attributes:
        type: The container type: ``tuple``, ``list``, ``dict``, a namedtuple
            type, a dataclass type, or a registered custom type.
        children: Child ``TreeSpec``\s (empty tuple for a leaf).
        context: Optional metadata supplied by a registered type's
            ``flatten_fn`` (used by its ``unflatten_fn``).
        node_data: Optional per-type data needed to rebuild the container:
            sorted dict keys for ``dict``; field names for ``namedtuple`` and
            ``dataclass``.
    """

    type: Any
    children: Tuple["TreeSpec", ...] = ()
    context: Any = None
    node_data: Any = None

    @property
    def num_leaves(self) -> int:
        """Number of leaves described by this spec (1 for a leaf; 0 for an
        empty container node)."""
        if not self.children:
            return 0 if _is_container_spec(self) else 1
        return sum(child.num_leaves for child in self.children)

    @property
    def children_specs(self) -> Tuple["TreeSpec", ...]:
        """The child :class:`TreeSpec`\\s of this node (alias of ``children``)."""
        return self.children

    def __repr__(self) -> str:
        if not self.children:
            return f"TreeSpec({self.type!r}, leaf)"
        return f"TreeSpec({self.type!r}, children={self.children!r})"


def register_pytree_node(
    node_type: type, flatten_fn: Callable, unflatten_fn: Callable
) -> None:
    """Register a custom container type as a pytree node.

    Args:
        node_type: The container class to register.
        flatten_fn: ``fn(obj) -> (children, context)`` — returns the child
            values (any pytrees) and arbitrary context metadata.
        unflatten_fn: ``fn(context, children) -> obj`` — rebuilds the
            container from the context and the (unflattened) children.

    Raises:
        TypeError: If ``node_type`` is not a type or the functions are not
            callable.
    """
    if not isinstance(node_type, type):
        raise TypeError(f"node_type must be a type, got {node_type!r}")
    if not callable(flatten_fn) or not callable(unflatten_fn):
        raise TypeError("flatten_fn and unflatten_fn must both be callable")
    _PYTREE_NODE_REGISTRY[node_type] = (flatten_fn, unflatten_fn)


def _is_container_spec(spec: TreeSpec) -> bool:
    """True if ``spec`` describes a container node (never a leaf).

    ``flatten`` never produces a leaf with one of these types, so a childless
    spec of such a type is an *empty container* (0 leaves) rather than a leaf
    (1 leaf). Needed by :attr:`TreeSpec.num_leaves`, since both have empty
    ``children`` tuples.
    """
    spec_type = spec.type
    if spec_type in _PYTREE_NODE_REGISTRY:
        return True
    if isinstance(spec_type, type):
        if issubclass(spec_type, (tuple, list, dict)):
            return True
        if dataclasses.is_dataclass(spec_type):
            return True
    return False


def flatten(obj: Any) -> Tuple[List[Any], TreeSpec]:
    """Flatten a pytree into ``(leaves, treespec)``.

    Traverses container nodes in pre-order, collecting leaves (non-container
    values). Built-in support: ``tuple``, ``list``, ``dict`` (keys sorted),
    ``namedtuple``, ``dataclass``, plus types registered via
    ``register_pytree_node``.

    Args:
        obj: Any value (pytree or leaf).

    Returns:
        A tuple ``(leaves, treespec)`` where ``leaves`` is the list of leaf
        values in pre-order and ``treespec`` the :class:`TreeSpec`
        describing the structure.
    """
    leaves: List[Any] = []
    return leaves, _flatten_into(obj, leaves)


def _flatten_into(obj: Any, leaves: List[Any]) -> TreeSpec:
    """Recursive pre-order flattening: append leaves, return the spec node."""
    obj_type = type(obj)
    # 1. Registered custom types (walk the MRO so registered base classes
    #    catch subclasses; exact type first).
    for base in obj_type.__mro__:
        registered = _PYTREE_NODE_REGISTRY.get(base)
        if registered is not None:
            flatten_fn, _ = registered
            children, context = flatten_fn(obj)
            child_specs = tuple(_flatten_into(child, leaves) for child in children)
            return TreeSpec(type=base, children=child_specs, context=context)
    # 2. namedtuple instances (checked before plain tuples).
    if isinstance(obj, tuple) and hasattr(obj_type, "_fields"):
        child_specs = tuple(_flatten_into(child, leaves) for child in obj)
        return TreeSpec(type=obj_type, children=child_specs, node_data=obj_type._fields)
    # 3. dataclass instances (never the class itself).
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = dataclasses.fields(obj)
        child_specs = tuple(
            _flatten_into(getattr(obj, field.name), leaves) for field in fields
        )
        return TreeSpec(
            type=obj_type, children=child_specs, node_data=[field.name for field in fields]
        )
    # 4-6. Plain containers: tuple, list, dict (keys sorted for determinism).
    if isinstance(obj, tuple):
        child_specs = tuple(_flatten_into(child, leaves) for child in obj)
        return TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, list):
        child_specs = tuple(_flatten_into(child, leaves) for child in obj)
        return TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, dict):
        keys = sorted(obj)
        child_specs = tuple(_flatten_into(obj[key], leaves) for key in keys)
        return TreeSpec(type=obj_type, children=child_specs, node_data=keys)
    # Leaf: anything else (None, scalars, ndarrays, ...).
    leaves.append(obj)
    return TreeSpec(type=obj_type)


def unflatten(leaves: List[Any], treespec: TreeSpec) -> Any:
    """Rebuild a pytree from leaves and a :class:`TreeSpec`.

    Exact inverse of :func:`flatten`: consumes leaves in pre-order and
    rebuilds containers per ``treespec``.

    Args:
        leaves: The flat list of leaf values (consumed in order).
        treespec: The structural description to rebuild.

    Returns:
        The rebuilt pytree.

    Raises:
        ValueError: If the number of leaves does not match the spec.
    """
    if len(leaves) != treespec.num_leaves:
        raise ValueError(
            f"unflatten: treespec describes {treespec.num_leaves} leaves, "
            f"but {len(leaves)} were given"
        )
    # Iterate (not pop) so the caller's list is never mutated.
    return _unflatten_into(iter(leaves), treespec)


def _unflatten_into(leaves: Iterator[Any], spec: TreeSpec) -> Any:
    """Recursive pre-order rebuilding: consume leaves, return the object."""
    if not spec.children:
        if _is_container_spec(spec):
            # Empty container node: rebuild without consuming leaves.
            return _rebuild_container(spec, [])
        return next(leaves)
    children = [_unflatten_into(leaves, child) for child in spec.children]
    return _rebuild_container(spec, children)


def _rebuild_container(spec: TreeSpec, children: List[Any]) -> Any:
    """Reconstruct a container node from its (rebuilt) children."""
    registered = _PYTREE_NODE_REGISTRY.get(spec.type)
    if registered is not None:
        _, unflatten_fn = registered
        return unflatten_fn(spec.context, children)
    if (
        isinstance(spec.type, type)
        and issubclass(spec.type, tuple)
        and hasattr(spec.type, "_fields")
    ):
        # namedtuple: rebuild positionally in recorded field order.
        return spec.type(*children)
    if isinstance(spec.type, type) and dataclasses.is_dataclass(spec.type):
        # dataclass: rebuild by recorded field name.
        return spec.type(**dict(zip(spec.node_data, children)))
    if isinstance(spec.type, type) and issubclass(spec.type, tuple):
        return spec.type(children)
    if isinstance(spec.type, type) and issubclass(spec.type, list):
        return spec.type(children)
    if isinstance(spec.type, type) and issubclass(spec.type, dict):
        return spec.type(zip(spec.node_data, children))
    raise TypeError(f"unflatten: no reconstruction rule for TreeSpec node type {spec.type!r}")
