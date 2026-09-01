"""TreeSpec — pytree flatten/unflatten.

Structured I/O (tuple/list/dict/namedtuple/dataclass) is supported
everywhere in etl: trace inputs, run outputs, bind. ``TreeSpec`` is the
structural description that makes flattening reversible.

Built-in node types: ``tuple``, ``list``, ``dict`` (keys sorted for
determinism), ``defaultdict`` / ``Counter`` (dict subclasses with preserved
rebuild semantics), ``namedtuple`` instances, ``dataclass`` instances. Custom
containers register via ``register_pytree_node``.

Invariants (binding):
- ``flatten`` returns ``(leaves, treespec)`` in pre-order; ``unflatten`` is
  its exact inverse: ``unflatten(flatten(x)) == x`` structurally.
- Leaves are anything that is not a recognized container type.
- Dict keys are sorted (``sorted(keys)``) so specs are order-stable and
  hashable-consistent.
- Treespecs compare structurally (frozen dataclass equality).
- ``tree_map`` / ``tree_leaves`` / ``tree_structure`` / ``tree_flatten`` /
  ``tree_unflatten`` are pure sugar over ``flatten``/``unflatten`` — no new
  semantics.
- ``first_mismatch_path`` / ``format_path`` are the shared structure-mismatch
  contract consumed by trace/pipeline/transforms (NOT top-level ``etl``
  surface; importable from ``etl.core``).
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "TreeSpec",
    "flatten",
    "unflatten",
    "register_pytree_node",
    # pytree sugar (pure flatten/unflatten aliases and compositions)
    "tree_map",
    "tree_leaves",
    "tree_structure",
    "tree_flatten",
    "tree_unflatten",
    # cross-module structure-mismatch contract (trace/pipeline/transforms)
    "first_mismatch_path",
    "format_path",
    "describe_node",
]

# Registered custom pytree node types → (flatten_fn, unflatten_fn).
# flatten_fn(obj) -> (children, context); unflatten_fn(context, children) -> obj.
_PYTREE_NODE_REGISTRY: Dict[type, Tuple[Callable, Callable]] = {}


@dataclass(frozen=True)
class TreeSpec:
    """Structural description of a pytree (frozen; structural equality).

    Attributes:
        type: The container type: ``tuple``, ``list``, ``dict``, a namedtuple
            type, a dataclass type, or a registered custom type.
        children: Child ``TreeSpec``\\s (empty tuple for a leaf).
        context: Optional metadata supplied by a registered type's
            ``flatten_fn`` (used by its ``unflatten_fn``).
        node_data: Optional per-type data needed to rebuild the container:
            sorted dict keys for ``dict`` (a ``(default_factory, keys)`` pair
            for ``defaultdict``); field names for ``namedtuple`` and
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
            callable; if ``node_type`` is ``object`` itself (every type's MRO
            ends in ``object``, so registering it would hijack the lookup for
            all types).
    """
    if node_type is object:
        raise TypeError(
            "register_pytree_node: cannot register 'object' as a pytree node — "
            "it would hijack the MRO lookup for all types"
        )
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
        # etl's own dataclass value types are leaves (see _flatten_into);
        # only user-defined dataclasses act as containers.
        if dataclasses.is_dataclass(spec_type) and not spec_type.__module__.split(".")[0] == "etl":
            return True
    return False


def flatten(obj: Any) -> Tuple[List[Any], TreeSpec]:
    """Flatten a pytree into ``(leaves, treespec)``.

    Traverses container nodes in pre-order, collecting leaves (non-container
    values). Built-in support: ``tuple``, ``list``, ``dict`` (keys sorted),
    ``defaultdict`` / ``Counter``, ``namedtuple``, ``dataclass``, plus types
    registered via ``register_pytree_node``.

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
    # 3. dataclass instances (never the class itself). etl's own value types
    #    (SymbolicTensor, TensorSpec, Tensor, Dim/DimExpr, Device, Group,
    #    Location, ...) are dataclasses too, but they are LEAVES — only
    #    user-defined dataclasses act as pytree containers.
    if (
        dataclasses.is_dataclass(obj)
        and not isinstance(obj, type)
        and not obj_type.__module__.split(".")[0] == "etl"
    ):
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
        try:
            keys = sorted(obj)
        except TypeError:
            # The raw sort TypeError must never leak: wrap it with guidance.
            type_names: List[str] = []
            for key in obj:
                name = type(key).__name__
                if name not in type_names:
                    type_names.append(name)
            raise TypeError(
                f"flatten: cannot sort dict keys of mixed types (types: {type_names}); "
                "dict keys are sorted for deterministic tree structure — use keys of one "
                "type or register_pytree_node"
            ) from None
        child_specs = tuple(_flatten_into(obj[key], leaves) for key in keys)
        if isinstance(obj, collections.defaultdict):
            # Record the factory alongside the sorted keys so unflatten can
            # rebuild. Unpersistable factories (lambdas, partials, ...) are
            # still recorded (for repr in the error) and rejected at
            # unflatten time — flatten never fails on them.
            node_data = (obj.default_factory, keys)
        else:
            node_data = keys
        return TreeSpec(type=obj_type, children=child_specs, node_data=node_data)
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


def _dataclass_init_rejecting_field_names(cls: type) -> List[str]:
    """Field names a ``node_data``-driven ``cls(**...)`` rebuild rejects.

    ``flatten`` records every real field name, but ``__init__`` will not
    accept: (a) ``init=False`` fields (stored in ``node_data``, rejected as
    unexpected keyword arguments) and (b) ``InitVar`` pseudo-fields (required
    by ``__init__`` but never stored). ``ClassVar`` pseudo-fields are neither
    recorded nor required. Declaration order is preserved.
    """
    names: List[str] = []
    real_fields = {field.name: field for field in dataclasses.fields(cls)}
    for name in cls.__dataclass_fields__:
        entry = cls.__dataclass_fields__[name]
        if isinstance(entry.type, dataclasses.InitVar):
            names.append(name)
        elif name in real_fields and not real_fields[name].init:
            names.append(name)
    return names


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
        # dataclass: rebuild by recorded field name. InitVar / init=False
        # fields make __init__ reject the recorded mapping — wrap that
        # TypeError only when such fields exist (namedtuple arity errors and
        # unrelated TypeErrors from a dataclass's own __init__ keep their raw
        # form).
        try:
            return spec.type(**dict(zip(spec.node_data, children)))
        except TypeError:
            rejecting = _dataclass_init_rejecting_field_names(spec.type)
            if rejecting:
                raise TypeError(
                    f"unflatten: cannot rebuild dataclass {spec.type.__qualname__}: its "
                    f"__init__ rejects field(s) {rejecting} "
                    "(InitVar/init=False fields are not stored) — register the type via "
                    "register_pytree_node for custom reconstruction"
                ) from None
            raise
    if isinstance(spec.type, type) and issubclass(spec.type, tuple):
        return spec.type(children)
    if isinstance(spec.type, type) and issubclass(spec.type, list):
        return spec.type(children)
    if isinstance(spec.type, type) and issubclass(spec.type, dict):
        if issubclass(spec.type, collections.defaultdict):
            factory, keys = spec.node_data
            if factory is None or isinstance(factory, type):
                return spec.type(factory, dict(zip(keys, children)))
            raise TypeError(
                f"unflatten: cannot rebuild {spec.type.__qualname__}: "
                f"default_factory {factory!r} cannot be persisted — register the type "
                "via register_pytree_node for custom reconstruction"
            )
        if issubclass(spec.type, collections.Counter):
            # Mapping constructor (positional form) so non-str keys roundtrip.
            return spec.type(dict(zip(spec.node_data, children)))
        return spec.type(zip(spec.node_data, children))
    raise TypeError(f"unflatten: no reconstruction rule for TreeSpec node type {spec.type!r}")


# --- Structure-mismatch helpers (cross-module contract) ----------------------
# Consumed by trace (graph.py), pipeline, and transforms (vectorize/vmap):
# importable as ``from etl.core import first_mismatch_path, format_path,
# describe_node``. These are internal contract names, NOT part of the
# top-level ``etl`` surface.


def _is_defaultdict_spec(spec: TreeSpec) -> bool:
    """True when ``spec`` describes a ``defaultdict`` (or subclass) node."""
    return (
        isinstance(spec.type, type)
        and issubclass(spec.type, collections.defaultdict)
    )


def _dict_keys(spec: TreeSpec) -> Optional[List[Any]]:
    """The recorded sorted key list of a dict-subclass spec, or ``None`` when
    the spec carries no ``node_data`` (hand-built specs).

    ``defaultdict`` specs record ``(default_factory, keys)`` — the keys live
    at index 1 so structural comparison covers the factory too.
    """
    if spec.node_data is None:
        return None
    if _is_defaultdict_spec(spec):
        return spec.node_data[1]
    return spec.node_data


def first_mismatch_path(
    spec_a: TreeSpec,
    spec_b: TreeSpec,
    *,
    strict: bool = False,
    leaf_vs_empty_is_mismatch: bool = False,
) -> Optional[Tuple[Any, ...]]:
    """The pytree path where ``spec_a`` first diverges from ``spec_b`` in
    CONTAINER structure, or ``None`` when the two structures match.

    Cross-module contract (binding; consumed by trace/pipeline/transforms):

    - Container nodes must match exactly: node ``type`` (``!=``),
      ``node_data``, and child count.
    - One childless node vs a node with children → mismatch at that prefix.
    - Both childless (leaf vs leaf OR leaf vs empty container): match by
      default (grad/graph/pipeline semantics). With
      ``leaf_vs_empty_is_mismatch=True`` (vectorize's semantics) or
      ``strict=True``, a ``num_leaves`` difference (1 vs 0) at the same node
      IS a mismatch (``strict=True`` subsumes
      ``leaf_vs_empty_is_mismatch=True``, which stays for the legacy
      callers).
    - ``strict=True`` additionally treats two childless EMPTY containers with
      differing ``type`` or ``node_data`` as a mismatch; leaf vs leaf always
      matches regardless of leaf type.
    - Leaf *types* are deliberately ignored — childless nodes with equal
      ``num_leaves`` always match.
    - While descending, dict-subclass keys are sourced from ``node_data``
      (positional index for all other container types).

    Args:
        spec_a: The first structure (the "expected" one).
        spec_b: The second structure (the "got" one).
        strict: Full structural strictness — leaf (1 leaf) vs empty container
            (0 leaves) and empty-vs-empty containers with differing
            type/node_data are mismatches. Subsumes
            ``leaf_vs_empty_is_mismatch=True``.
        leaf_vs_empty_is_mismatch: Treat a leaf (1 leaf) vs an empty
            container (0 leaves) at the same node as a mismatch (legacy flag;
            superseded by ``strict=True``).

    Returns:
        The path (tuple of int positional indices / dict keys) of the first
        diverging node, or ``None`` when the structures match.
    """

    def walk(a: TreeSpec, b: TreeSpec, prefix: Tuple[Any, ...]):
        if not a.children and not b.children:
            if strict:
                if a.num_leaves != b.num_leaves:
                    return prefix
                if a.num_leaves == 0 and (a.type != b.type or a.node_data != b.node_data):
                    return prefix
                return None
            if leaf_vs_empty_is_mismatch and a.num_leaves != b.num_leaves:
                return prefix
            return None
        if not a.children or not b.children:
            return prefix
        if a.type != b.type:
            return prefix
        if a.node_data != b.node_data:
            return prefix
        if len(a.children) != len(b.children):
            return prefix
        keys = _dict_keys(a)
        for index, (child_a, child_b) in enumerate(zip(a.children, b.children)):
            key = keys[index] if keys is not None else index
            mismatch = walk(child_a, child_b, prefix + (key,))
            if mismatch is not None:
                return mismatch
        return None

    return walk(spec_a, spec_b, ())


def format_path(path) -> str:
    """Render a pytree key path readably, e.g. ``[0]['weights'][1]``."""
    if not path:
        return "()"
    parts = []
    for key in path:
        if isinstance(key, str):
            parts.append(f"[{key!r}]")
        else:
            parts.append(f"[{key}]")
    return "".join(parts)


def _subtree_spec_at(spec: TreeSpec, path: Tuple[Any, ...]) -> TreeSpec:
    """The subtree :class:`TreeSpec` at ``path`` (keys as produced by
    :func:`first_mismatch_path`)."""
    current = spec
    for key in path:
        keys = _dict_keys(current)
        if keys is not None:
            index = keys.index(key)
        else:
            index = key
        current = current.children[index]
    return current


def describe_node(spec: TreeSpec) -> str:
    """One-line human description of the node ``spec`` describes, for
    structure-mismatch messages.

    Canonical node-description contract (consumed by ``tree_map`` here and by
    trace/pipeline/transforms — e.g. ``dict with keys ['a', 'b']``, ``tuple
    of length 2``, ``namedtuple of length 2``, ``dataclass with fields
    ['f1', 'f2']``):

    - Childless leaf → its type name (``int``, ``Tensor``, ``NoneType``, ...).
    - Containers → kind + arity/keys, in this precedence (mirroring
      ``flatten``'s node precedence): namedtuple, dataclass, tuple, list,
      dict, then registered custom containers.
    - Childless empty containers fall through to the container wording with
      an empty list/keys (``dict with keys []``).
    """
    if not spec.children:
        if spec.num_leaves == 0:
            return _container_desc(spec)
        spec_type = spec.type
        return spec_type.__name__ if isinstance(spec_type, type) else str(spec_type)
    return _container_desc(spec)


def _container_desc(spec: TreeSpec) -> str:
    """One-line description of the container node ``spec`` describes.

    Kind + arity/keys, e.g. ``dict with keys ['a', 'b']``, ``tuple of length
    2``, ``namedtuple of length 2``, ``dataclass with fields ['f1', 'f2']``.
    Childless (empty) containers yield the same wording with an empty list
    (``dict with keys []``). Kind detection mirrors ``flatten``'s node
    precedence (namedtuple before tuple; dataclass before plain containers).
    """
    spec_type = spec.type
    if (
        isinstance(spec_type, type)
        and issubclass(spec_type, tuple)
        and hasattr(spec_type, "_fields")
    ):
        return f"namedtuple of length {len(spec.children)}"
    if isinstance(spec_type, type) and dataclasses.is_dataclass(spec_type):
        return f"dataclass with fields {spec.node_data!r}"
    if isinstance(spec_type, type) and issubclass(spec_type, tuple):
        return f"tuple of length {len(spec.children)}"
    if isinstance(spec_type, type) and issubclass(spec_type, list):
        return f"list of length {len(spec.children)}"
    if isinstance(spec_type, type) and issubclass(spec_type, dict):
        return f"dict with keys {_dict_keys(spec)!r}"
    # Registered custom container types (no pinned wording).
    name = spec_type.__name__ if isinstance(spec_type, type) else str(spec_type)
    return f"{name} of length {len(spec.children)}"


# --- Sugared pytree API (pure sugar over flatten/unflatten) -------------------


def tree_map(fn: Callable, *trees) -> Any:
    """Map ``fn`` over the leaves of one or more trees of identical structure.

    Pure sugar over :func:`flatten` / :func:`unflatten` — no new semantics:

    - One tree: ``unflatten([fn(leaf) for leaf in leaves], spec)``.
    - Several trees: structures are validated pairwise against the first tree
      (:func:`first_mismatch_path` with ``strict=True`` — leaf vs empty
      container and empty-vs-empty container mismatches are errors); ``fn``
      is applied per leaf position over the zipped leaf lists; results are
      rebuilt into the first tree's structure.

    Args:
        fn: Called as ``fn(leaf)`` for one tree, ``fn(*leaves)`` for several.
        *trees: One or more pytrees sharing one structure.

    Returns:
        The rebuilt tree (first tree's structure).

    Raises:
        TypeError: If no trees are given, or the trees do not share one
            structure.
    """
    if not trees:
        raise TypeError("tree_map: expected at least one tree")
    if len(trees) == 1:
        leaves, spec = flatten(trees[0])
        return unflatten([fn(leaf) for leaf in leaves], spec)
    first_leaves, first_spec = flatten(trees[0])
    leaf_lists = [first_leaves]
    for tree in trees[1:]:
        leaves, spec = flatten(tree)
        mismatch = first_mismatch_path(first_spec, spec, strict=True)
        if mismatch is not None:
            expected_spec = _subtree_spec_at(first_spec, mismatch)
            got_spec = _subtree_spec_at(spec, mismatch)
            raise TypeError(
                "tree_map: trees do not have the same structure — "
                f"first mismatch at pytree path {format_path(mismatch)}: "
                f"expected {describe_node(expected_spec)}, "
                f"got {describe_node(got_spec)} "
                f"(expected {expected_spec!r}, got {got_spec!r})"
            )
        leaf_lists.append(leaves)
    return unflatten([fn(*zipped) for zipped in zip(*leaf_lists)], first_spec)


def tree_leaves(tree) -> List[Any]:
    """All leaves of ``tree`` in pre-order (alias of ``flatten(tree)[0]``)."""
    return flatten(tree)[0]


def tree_structure(tree) -> TreeSpec:
    """The :class:`TreeSpec` of ``tree`` (alias of ``flatten(tree)[1]``)."""
    return flatten(tree)[1]


# Identity aliases (pure sugar): ``tree_flatten is flatten`` and
# ``tree_unflatten is unflatten`` — the inherited docstrings apply.
tree_flatten = flatten
tree_unflatten = unflatten
