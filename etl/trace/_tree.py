"""Shared internal pytree/static-value helpers for the tracing machinery.

Single home for the helpers that `trace.py`, `graph.py`, and
`control_flow.py` all need: the static-value predicate, the registered-node
MRO lookup, the local pytree walker, the leaf-path iterator, and the
`ir.Value -> SymbolicTensor` wrapper.

Why this module exists (the duplication it kills): static-value
classification used to be copy-pasted between `trace.py` and
`control_flow.py` ("keep the two implementations in sync" — they drifted);
the pytree walker existed in three recursive variants (`core.tree`'s,
`trace.py`'s, `control_flow.py`'s) differing only in which objects count as
leaves; and leaf-path iteration was re-implemented in `trace.py` and
`graph.py`. All of that now lives here.

Ownership rules:

- **The leaf policy is a parameter, not a copy.** The container-descent
  rules (registered pytree nodes, namedtuple, user dataclass, tuple, list,
  dict) are shared; what counts as a *leaf* differs per caller:
  `trace.py` keeps `TensorSpec`/`SymbolicTensor` as typed marker leaves,
  `control_flow.py` keeps its `_LEAF_TYPES` as plain leaves, and
  `core.flatten` (not this module) keeps etl-module dataclasses as leaves.
  Callers pass a ``leaf_spec(obj) -> (recorded_leaf, TreeSpec) | None``
  policy into `_flatten_into`.
- **No imports from the other `trace` submodules** (this module imports
  `etl.core` + `etl.ir` + numpy only), so any trace module may import it
  without creating a cycle.
- Names here are private (`_`-prefixed); `trace.py` re-exports
  `_format_path` / `_iter_leaf_paths` for the one external consumer
  (`etl.transforms.grad` imports them from ``etl.trace.trace``).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from etl import core
from etl import ir
from etl.core import format_path as _format_path  # canonical path renderer
from etl.core import tree as _core_tree

#: The registered-custom-node table of core's pytrees. `register_pytree_node`
#: mutates this dict in place, so the alias stays live. All trace-tree walks
#: honor the same registrations as `core.flatten`.
_PYTREE_NODE_REGISTRY = _core_tree._PYTREE_NODE_REGISTRY


def _is_static_value(obj: Any) -> bool:
    """True iff `obj` is a static Python value that specializes the graph.

    Accepted (per the root value-model contract): `None`, bool, int, float,
    complex, str, `enum.Enum`, numpy `dtype` objects, `slice`, `core.Dim` /
    `core.DimExpr` (symbolic shape expressions — one leaf, snapshotted like
    any other static value), and `core.Device` (a static device spec — one
    leaf, snapshotted like any other static value). Everything else
    (including numpy scalars and other config objects) is NOT static in v1 —
    the tracer raises `TraceError` for it.

    The single canonical copy: `trace.py` classifies trace inputs/outputs
    with it and `control_flow.py` classifies operands/carries with it — there
    must never be a second implementation to keep in sync.
    """
    if obj is None:
        return True
    # bool must be checked before int (True is an int instance).
    if isinstance(obj, (bool, int, float, complex, str, slice, enum.Enum)):
        return True
    if isinstance(obj, np.dtype):
        return True
    if isinstance(obj, (core.Dim, core.DimExpr)):
        return True
    if isinstance(obj, core.Device):
        return True
    return False


def _registered_pytree_base(obj_type: Any) -> Optional[type]:
    """The first type in `obj_type`'s MRO registered in the core pytree
    registry, or None.

    Mirrors the MRO walk `_flatten_into` uses (registered base classes catch
    subclasses); used both for object-level checks (`_is_registered_node`)
    and for TreeSpec node-type checks (`_leaf_registered_flags` in
    control_flow.py). Non-type spec entries (e.g. the `None` leaf type)
    return None.
    """
    if not isinstance(obj_type, type):
        return None
    for base in obj_type.__mro__:
        if base in _PYTREE_NODE_REGISTRY:
            return base
    return None


def _to_symbolic(value: "ir.Value") -> "core.SymbolicTensor":
    """Wrap an `ir.Value` (op result or region block arg) as a SymbolicTensor.

    The trace-level wrapper: dtype/shape come from the value's IR type. (For
    trace INPUTS, `trace.py`'s `_spec_to_symbolic` additionally maps `None`
    spec dims to fresh `Dim` names — a different concern, kept there.)
    """
    return core.SymbolicTensor(
        value=value, dtype=value.type.dtype, shape=value.type.shape
    )


def _flatten(
    obj: Any,
    leaf_spec: Callable[[Any], Optional[Tuple[Any, "core.TreeSpec"]]],
    plain_leaf_type: Callable[[Any], Any] = lambda obj: None,
) -> Tuple[list, "core.TreeSpec"]:
    """``(leaves, treespec)`` — the shared local pytree walk.

    Same container rules as ``core.flatten`` (registered custom nodes via the
    MRO walk, namedtuple before tuple, user dataclasses as containers, etl
    value types as leaves) with two deliberate differences: which objects
    count as leaves is decided by the caller's ``leaf_spec`` policy instead
    of `core`'s etl-module check, and the *fallback* leaf's ``TreeSpec.type``
    is decided by ``plain_leaf_type`` instead of always ``None``.

    - ``leaf_spec(obj)`` returns the pair ``(leaf_to_record, TreeSpec)`` when
      ``obj`` is a leaf (the first element is appended to ``leaves`` — it may
      be a marker standing in for ``obj``, e.g. trace's `_TensorSpecLeaf`) or
      ``None`` to keep descending.
    - ``plain_leaf_type(obj)`` returns the ``TreeSpec.type`` recorded for a
      fallback leaf (an object the policy declined that is not a container).
      `trace.py` passes the identity ``type`` so static leaves record their
      own Python type (pipeline's ``_is_tensor_leaf_spec`` distinguishes
      static positions from tensor markers by leaf type — ``type(None)`` for
      a static ``None``); `control_flow.py` passes a constant ``None`` (its
      trees are transient and plain leaves are all ``TreeSpec(type=None)``).
    """
    leaves: List[Any] = []
    return leaves, _flatten_into(obj, leaves, leaf_spec, plain_leaf_type)


def _flatten_into(
    obj: Any,
    leaves: List[Any],
    leaf_spec: Callable[[Any], Optional[Tuple[Any, "core.TreeSpec"]]],
    plain_leaf_type: Callable[[Any], Any],
) -> "core.TreeSpec":
    """Recursive pre-order flattening: append leaves, return the spec node.

    Order of checks (binding, matches `core.tree._flatten_into`): registered
    custom types first (registered nodes win over everything — even etl value
    types registered via `register_pytree_node` become containers), then the
    caller's leaf policy, then namedtuple / user dataclass / tuple / list /
    dict container descent, then the plain leaf fallback.

    Note: unlike `core.tree._flatten_into`, dict nodes record ONLY the sorted
    keys in ``node_data`` (no `defaultdict`/`Counter` default-factory
    handling) — trace trees deliberately keep the plain convention; the
    factories are a `core.flatten`-only feature.
    """
    obj_type = type(obj)
    # 1. Registered custom types (walk the MRO so registered base classes
    #    catch subclasses; exact type first) — same as core.tree. A
    #    registered node (e.g. a sparse tensor) flattens via its registered
    #    flatten_fn and its children recurse; `TreeSpec.type` records the
    #    registered base type so `core.unflatten` rebuilds via the registered
    #    unflatten_fn.
    for base in obj_type.__mro__:
        registered = _PYTREE_NODE_REGISTRY.get(base)
        if registered is not None:
            flatten_fn, _ = registered
            children, context = flatten_fn(obj)
            child_specs = tuple(
                _flatten_into(child, leaves, leaf_spec, plain_leaf_type)
                for child in children
            )
            return core.TreeSpec(type=base, children=child_specs, context=context)
    # 2. The caller's leaf policy (e.g. marker leaves for TensorSpec/
    #    SymbolicTensor in trace.py, `_LEAF_TYPES` in control_flow.py).
    leaf = leaf_spec(obj)
    if leaf is not None:
        recorded, spec = leaf
        leaves.append(recorded)
        return spec
    # 3. namedtuple instances (checked before plain tuples).
    if isinstance(obj, tuple) and hasattr(obj_type, "_fields"):
        child_specs = tuple(
            _flatten_into(child, leaves, leaf_spec, plain_leaf_type) for child in obj
        )
        return core.TreeSpec(
            type=obj_type, children=child_specs, node_data=obj_type._fields
        )
    # 4. dataclass instances (never the class itself). etl's own value types
    #    (Device, Dim, Group, ...) are LEAVES — the same module check as
    #    core.tree._flatten_into; only user-defined dataclasses act as pytree
    #    containers.
    if (
        dataclasses.is_dataclass(obj)
        and not isinstance(obj, type)
        and not obj_type.__module__.split(".")[0] == "etl"
    ):
        field_names = [field.name for field in dataclasses.fields(obj)]
        child_specs = tuple(
            _flatten_into(getattr(obj, name), leaves, leaf_spec, plain_leaf_type)
            for name in field_names
        )
        return core.TreeSpec(
            type=obj_type, children=child_specs, node_data=field_names
        )
    # 5-7. Plain containers: tuple, list, dict (keys sorted for determinism).
    if isinstance(obj, tuple):
        child_specs = tuple(
            _flatten_into(child, leaves, leaf_spec, plain_leaf_type) for child in obj
        )
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, list):
        child_specs = tuple(
            _flatten_into(child, leaves, leaf_spec, plain_leaf_type) for child in obj
        )
        return core.TreeSpec(type=obj_type, children=child_specs)
    if isinstance(obj, dict):
        keys = sorted(obj)  # core.TreeSpec convention: keys sorted
        child_specs = tuple(
            _flatten_into(obj[key], leaves, leaf_spec, plain_leaf_type)
            for key in keys
        )
        return core.TreeSpec(type=obj_type, children=child_specs, node_data=keys)
    # 8. Plain leaf: record the object itself; `plain_leaf_type(obj)` decides
    #    the recorded TreeSpec.type (trace: the object's own type, so
    #    pipeline's `_is_tensor_leaf_spec` can tell static leaves from tensor
    #    markers; control flow: constant None).
    leaves.append(obj)
    return core.TreeSpec(type=plain_leaf_type(obj))


def _iter_leaf_paths(
    tree_spec: "core.TreeSpec", prefix: Tuple[Any, ...] = ()
) -> Iterator[Tuple[Any, ...]]:
    """Yield the pytree key path of every leaf in pre-order.

    Matches `core.flatten`'s leaf order exactly (one path per leaf). Path
    entries are child indices, except `dict` nodes where the recorded sorted
    key (`node_data`) is used.
    """
    if tree_spec.num_leaves == 0:
        # Empty container (or a container of empty containers): no leaves.
        return
    if not tree_spec.children:
        yield prefix  # a leaf
        return
    for index, child in enumerate(tree_spec.children):
        if isinstance(tree_spec.type, type) and issubclass(tree_spec.type, dict):
            key = tree_spec.node_data[index]
        else:
            key = index
        yield from _iter_leaf_paths(child, prefix + (key,))
