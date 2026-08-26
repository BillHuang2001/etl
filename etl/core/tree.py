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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

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
        """Number of leaves described by this spec (1 for a leaf)."""
        if not self.children:
            return 1
        return sum(child.num_leaves for child in self.children)

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
    raise NotImplementedError(
        "flatten is not implemented yet (architecture phase); "
        "it will traverse tuple/list/dict/namedtuple/dataclass/registered types "
        "in pre-order (dict keys sorted)."
    )


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
    raise NotImplementedError(
        "unflatten is not implemented yet (architecture phase); "
        "it will rebuild the container from the spec, consuming leaves in order."
    )
