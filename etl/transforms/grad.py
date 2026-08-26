"""Public `grad` — reverse-mode gradients via the shared VJP machinery.

`grad` is graph→graph: the result is a `Graph` of ordinary ops mapping the
same inputs to gradient tensors. It shares VJP rules with `vjp` (and the
backward sweep in `autodiff.py`). Semantics are binding — see `./CONTEXT.md`
"AD semantics".

This module also hosts the small shared helpers used by the sibling AD entry
points (`jvp.py`, `vjp.py`): return-op inspection and tangent/cotangent
pytree normalization.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from etl import core, ir
from etl.trace import Graph, StaticValue, trace
from etl.trace.trace import _format_path, _iter_leaf_paths
from etl.transforms._wrappers import TransformCallable
from etl.transforms.autodiff import reverse_sweep


def grad(fn_or_graph, argnums=None):
    """Reverse-mode gradients: inputs → gradients w.r.t. selected arguments.

    Args:
        fn_or_graph: a `Graph`, or a callable/`Defn` (see Returns).
        argnums: which inputs to differentiate: an `int`, or a tuple/list of
            ints indexing the flattened **tensor** inputs (static values are
            excluded from the numbering); `None` = all tensor inputs. Selected
            inputs must be floating/complex — `core.TransformError` otherwise.

    Graph requirements:
        exactly one output, a scalar tensor — otherwise `core.ShapeError`.

    Returns:
        * `Graph` when given a `Graph`: same inputs → one gradient tensor
          when `argnums` is an int, else a tuple of gradient tensors (one per
          selected input, in order).
        * `TransformCallable` when given a callable/`Defn`: `grad(f)(*specs)`
          traces `f` exactly once and returns that same gradient graph.
          Concrete tensors raise `TraceError` (transforms never execute).

    Ops without a VJP rule (`runtime_call`, collectives, custom blocks without
    a registered rule) raise `core.TransformError` naming the op/block —
    never a silent fallback. `stop_gradient` yields a zero gradient.
    """
    if isinstance(fn_or_graph, Graph):
        return _grad_graph(fn_or_graph, argnums)

    if callable(fn_or_graph) or getattr(fn_or_graph, "__etl_defn__", False):
        return TransformCallable(
            build=lambda *args: _grad_fn(fn_or_graph, args, argnums),
            kind="grad",
        )

    raise TypeError("grad expects an etl.Graph, a callable, or an etl.Defn")


# ---------------------------------------------------------------------------
# Shared helpers (imported by jvp.py / vjp.py — keep in this one file)
# ---------------------------------------------------------------------------


def _return_values(graph: Graph) -> Tuple["ir.Value", ...]:
    """The entry function's `return`-terminator operands — the flattened
    tensor outputs, in output-tree leaf order (static output leaves are not
    `return` operands)."""
    terminator = graph.module.main.entry_block.terminator
    if terminator is None:
        raise core.ShapeError(
            "graph has no tensor outputs: the entry function has no "
            "`return` terminator"
        )
    return tuple(terminator.operands)


def _single_scalar_output(graph: Graph, transform: str) -> "ir.Value":
    """The graph's single tensor output, required to be a scalar.

    Raises:
        core.ShapeError: If the graph has != 1 tensor output or that output's
            shape is not `()`.
    """
    outputs = _return_values(graph)
    if len(outputs) != 1:
        raise core.ShapeError(
            f"{transform} requires exactly one tensor output; the graph "
            f"returns {len(outputs)}"
        )
    value = outputs[0]
    if tuple(value.type.shape) != ():
        raise core.ShapeError(
            f"{transform} requires the output to be a scalar tensor (shape "
            f"()), got shape {tuple(value.type.shape)}"
        )
    return value


def _first_structure_mismatch(
    user_spec: "core.TreeSpec",
    graph_spec: "core.TreeSpec",
    prefix: Tuple[Any, ...] = (),
):
    """Compare a `core.flatten` spec against a trace-time tree spec, ignoring
    leaf *types* (trace trees record marker types at tensor leaves, which can
    never equal the user's `TensorSpec`/`None`/static leaf types).

    Returns `None` when the structures match, else the pytree path of the
    first mismatching node. Container nodes must match exactly (same node
    type, same `node_data` — e.g. sorted dict keys — same child count);
    childless nodes match any childless node (leaf vs. empty-container
    mismatches are caught by the caller's leaf-count check).
    """
    user_children = user_spec.children
    graph_children = graph_spec.children
    if user_children and graph_children:
        if (
            user_spec.type != graph_spec.type
            or user_spec.node_data != graph_spec.node_data
        ):
            return prefix
        if len(user_children) != len(graph_children):
            return prefix
        is_dict = isinstance(graph_spec.type, type) and issubclass(
            graph_spec.type, dict
        )
        for index, (user_child, graph_child) in enumerate(
            zip(user_children, graph_children)
        ):
            key = graph_spec.node_data[index] if is_dict else index
            mismatch = _first_structure_mismatch(
                user_child, graph_child, prefix + (key,)
            )
            if mismatch is not None:
                return mismatch
        return None
    # Both childless: both leaves or both empty containers.
    if bool(user_children) != bool(graph_children):
        return prefix
    return None


def _normalize_extra_pytree(
    pytree: Any,
    tree_spec: "core.TreeSpec",
    static_records: Sequence[StaticValue],
    primal_specs: Sequence["core.TensorSpec"],
    transform: str,
    kind: str,
    none_requires_scalar: bool = False,
) -> Tuple:
    """Validate a tangent/cotangent pytree against a trace-time tree.

    Walks the user's pytree in parallel with `tree_spec` (the graph's input
    tree for jvp tangents, output tree for vjp cotangents) and returns the
    flat tuple of `core.TensorSpec`/`None` entries aligned with the tree's
    tensor leaves (== `primal_specs` order) — exactly what
    `forward_sweep`/`reverse_sweep` expect.

    Rules:

    * Structure must match the tree (ignoring leaf types) — else
      `core.TransformError` naming the first mismatching path.
    * Static positions (recorded in `static_records`) carry no
      tangent/cotangent: the entry must be `None` or mirror the recorded
      static value; anything else → `core.TransformError`.
    * Tensor positions may hold a `core.TensorSpec` (shape AND dtype must
      equal the primal spec exactly, symbolic `Dim` entries included — else
      `core.TransformError`) or `None` (zero tangent; for vjp cotangents a
      `None` entry seeds the sweep's in-graph scalar-one, valid only for
      scalar outputs — `core.ShapeError` otherwise when
      `none_requires_scalar` is set).
    """
    leaves, user_spec = core.flatten(pytree)
    mismatch = _first_structure_mismatch(user_spec, tree_spec)
    if mismatch is not None:
        raise core.TransformError(
            f"{transform}: the {kind} structure does not match the graph's "
            f"tree at path {_format_path(mismatch)}: got {user_spec}, "
            f"expected {tree_spec}"
        )
    if len(leaves) != tree_spec.num_leaves:
        raise core.TransformError(
            f"{transform}: the {kind} tree has {len(leaves)} leaves but the "
            f"graph's tree describes {tree_spec.num_leaves} leaves"
        )
    static_by_index = {record.index: record for record in static_records}
    paths = list(_iter_leaf_paths(tree_spec))
    flat = []
    tensor_pos = 0
    for index, leaf in enumerate(leaves):
        record = static_by_index.get(index)
        if record is not None:
            # Static position: nothing to differentiate here — the user
            # either passes None or mirrors the traced tree with the same
            # static value ("absent").
            if leaf is not None and (
                type(leaf).__qualname__ != record.kind or leaf != record.value
            ):
                raise core.TransformError(
                    f"{transform}: entry at static path "
                    f"{_format_path(paths[index])} must be None (static "
                    f"values carry no {kind}); got {leaf!r}"
                )
            continue
        primal = primal_specs[tensor_pos]
        if leaf is None:
            if none_requires_scalar and tuple(primal.shape) != ():
                raise core.ShapeError(
                    f"{transform}: a None {kind} at output path "
                    f"{_format_path(paths[index])} seeds a scalar-one "
                    f"cotangent, which requires a scalar output; got shape "
                    f"{tuple(primal.shape)}"
                )
            flat.append(None)
        elif isinstance(leaf, core.TensorSpec):
            if tuple(leaf.shape) != tuple(primal.shape):
                raise core.TransformError(
                    f"{transform}: {kind} shape {tuple(leaf.shape)} at path "
                    f"{_format_path(paths[index])} does not match the primal "
                    f"shape {tuple(primal.shape)}"
                )
            if leaf.dtype != primal.dtype:
                raise core.TransformError(
                    f"{transform}: {kind} dtype {leaf.dtype} at path "
                    f"{_format_path(paths[index])} does not match the primal "
                    f"dtype {primal.dtype}"
                )
            flat.append(leaf)
        else:
            raise core.TransformError(
                f"{transform}: {kind} entries must be a core.TensorSpec (an "
                f"explicit input) or None, got {leaf!r} (type "
                f"{type(leaf).__name__}) at path {_format_path(paths[index])}"
            )
        tensor_pos += 1
    return tuple(flat)


def _normalize_argnums(argnums, num_inputs: int) -> Tuple[int, ...]:
    """Normalize `grad`'s `argnums` against the flattened tensor inputs.

    `None` → all indices; an `int` → `(int,)`; a tuple/list → tuple. Entries
    must be plain in-range ints (bools rejected) — else
    `core.TransformError`.
    """
    if argnums is None:
        return tuple(range(num_inputs))
    if isinstance(argnums, int) and not isinstance(argnums, bool):
        selected = (argnums,)
    elif isinstance(argnums, (tuple, list)):
        selected = tuple(argnums)
    else:
        raise core.TransformError(
            f"grad: argnums must be None, an int, or a tuple/list of ints, "
            f"got {argnums!r} (type {type(argnums).__name__})"
        )
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, int):
            raise core.TransformError(
                f"grad: argnums entries must be ints indexing the flattened "
                f"tensor inputs, got {index!r}"
            )
        if not 0 <= index < num_inputs:
            raise core.TransformError(
                f"grad: argnum {index} is out of range — the graph has "
                f"{num_inputs} flattened tensor inputs (static values are "
                f"excluded from the numbering)"
            )
    return selected


# ---------------------------------------------------------------------------
# grad entry points
# ---------------------------------------------------------------------------


def _grad_graph(graph: Graph, argnums) -> Graph:
    """Validate the single-scalar-output requirement (`ShapeError`), normalize
    `argnums` (out-of-range/non-differentiable entries → `TransformError`),
    run the reverse sweep with a scalar-one cotangent of the output dtype, and
    return a graph producing only the selected input gradients."""
    _single_scalar_output(graph, "grad")

    is_single = isinstance(argnums, int) and not isinstance(argnums, bool)
    selected = _normalize_argnums(argnums, len(graph.tensor_specs))
    for index in selected:
        spec = graph.tensor_specs[index]
        if spec.dtype.kind not in "fc":
            raise core.TransformError(
                f"grad: input {index} (dtype {spec.dtype}) cannot be "
                f"differentiated — selected inputs must be floating-point "
                f"or complex"
            )

    # Reverse sweep seeded with the scalar-one cotangent (`None` → in-graph
    # ones constant, so NO cotangent block args). The sweep module's `return`
    # op yields (primal output, one cotangent per tensor input).
    swept = reverse_sweep(graph, (None,))
    module = swept.module
    return_op = module.main.entry_block.terminator
    cotangents = return_op.operands[1:]
    new_operands = tuple(cotangents[index] for index in selected)

    # Re-point the `return` op at only the selected cotangents. `ir.Op` is a
    # mutable dataclass: reassign `operands` and reconcile the `Use` records
    # by hand (`ir.verify` checks them in both directions).
    for value in return_op.operands:
        value.uses[:] = [use for use in value.uses if use.owner is not return_op]
    return_op.operands = new_operands
    for position, value in enumerate(new_operands):
        value.add_use(ir.Use(return_op, position))

    # Output tree: one bare gradient (int argnum) or a tuple of gradients.
    # Leaf specs use `type=None` — the canonical plain-leaf spec (the persist
    # codec round-trips "NoneType" ↔ None; `TreeSpec.num_leaves` counts it as
    # one leaf and `unflatten_outputs` consumes one tensor for it).
    if is_single:
        output_tree = core.TreeSpec(type=None)
    else:
        output_tree = core.TreeSpec(
            type=tuple,
            children=tuple(core.TreeSpec(type=None) for _ in new_operands),
        )

    # Inputs are unchanged: drop the sweep's 2-tuple input tree (and the
    # StaticValue records it appended for the None extra leaves) in favor of
    # the original tree/records. No static outputs.
    return Graph(
        module,
        input_specs=graph.input_specs,
        tensor_specs=graph.tensor_specs,
        output_tree=output_tree,
        static_values=graph.static_values,
        output_static_values=(),
        source_locations=swept.source_locations,
    )


def _grad_fn(fn, args, argnums) -> Graph:
    """`grad(f)(*specs)` — trace `fn` once via `etl.trace(fn, *args)`, then
    apply `_grad_graph`."""
    return _grad_graph(trace(fn, *args), argnums)
