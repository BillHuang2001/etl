"""Explicit collective operations (SPMD, local-tensor semantics).

Every collective builds exactly one IR op — effect kind ``collective`` — into
the active trace builder (``current_builder()``), following the same
discipline as the frontend ops in ``etl/ops``. Operands are *local*
:class:`etl.core.SymbolicTensor` values; results carry *local* shapes
computed by the op's registered ``shape_fn`` (``etl/ir/inference.py``). The
IR stays backend-neutral: no global tensor type ever appears, and the
reference numpy backend needs no multi-rank machinery beyond the
collective-executor hook (see ``context.py``).

Graph-time only: calling a collective with a concrete ``Tensor`` (or outside
a trace) raises ``TraceError`` — there is no eager mode. ``group=None``
selects the world group (``WORLD_GROUP``).

Canonical IR contract (the ``etl/ir`` registry wins over this directory's
CONTEXT.md where they disagree): op names have NO ``dist.`` prefix; all six
communication collectives carry attrs ``group: str`` (required) and
``group_size: int | None`` (required — ``None`` for the world group) plus
per-op params. ``broadcast`` builds the ``broadcast_collective`` op (the
shape op ``broadcast`` owns its name); the registry declares a
``src_rank`` attr (default 0) that the builder fills — dist validates
``src_rank`` itself and does not pass it.
"""

from __future__ import annotations

from typing import Optional, Tuple

from etl import core
from etl.trace import current_builder

from etl.dist._op_utils import (
    REDUCTIONS,  # re-exported here: collectives.REDUCTIONS stays importable
    _check_reduction,
    _check_static_divisibility,
    _get_location,
    _group_attrs,
    _is_rank,
    _normalize_axis,
    _normalize_pairs,
    _require_symbolic_tensor,
    _resolve_group,
    _wrap_result,
)
from etl.dist.group import Group

__all__ = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "collective_permute",
]


def _build_collective(
    op_name: str,
    tensor: "core.SymbolicTensor",
    attrs: dict,
    location,
) -> "core.SymbolicTensor":
    """Build the collective IR op into the active builder and wrap the result.

    Flow (mirrors the ``etl/ops`` discipline):
    1. ``builder = current_builder()`` — ``TraceError`` when no trace
       is active.
    2. ``builder.create(op_name, operands=(tensor.value,),
       attributes=attrs, location=location)`` — arity/attr checks and result
       shape inference (the op's registered ``shape_fn``) all happen here.
    3. Wrap the single result value in ``core.SymbolicTensor`` (dtype/shape
       read from the result's ``ValueType`` — dist never recomputes shapes).

    Raises:
        TraceError: no active trace.
    """
    builder = current_builder()
    op = builder.create(
        op_name,
        operands=(tensor.value,),
        attributes=attrs,
        location=location,
    )
    return _wrap_result(op.results[0], location)


def all_reduce(tensor: "core.SymbolicTensor", op: str = "sum", group: Optional[Group] = None) -> "core.SymbolicTensor":
    """All-reduce: combine ``tensor`` across all ranks of ``group`` with ``op``.

    Every rank supplies its local tensor and every rank receives the same
    combined result. Local shape and dtype are unchanged.

    Args:
        tensor: Local SymbolicTensor operand.
        op: Reduction kind — ``"sum"``, ``"max"``, ``"min"`` or ``"prod"``.
            v1 supports exactly these four; unknown names raise ValueError.
            (A user-reduction registry is a possible future extension.)
        group: Group or ``None`` for the world group. A static Python value:
            it specializes the graph and is recorded in op attributes by
            name + size.

    Returns:
        SymbolicTensor with the same local shape and dtype as ``tensor``.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ValueError: unknown reduction ``op``.

    IR: one ``all_reduce`` op; effect kind ``collective``; attrs
    ``{reduce_op, group, group_size}``; 1 operand, 1 result.
    """
    group = _resolve_group(group)
    _check_reduction(op)
    tensor = _require_symbolic_tensor(tensor)
    location = _get_location()
    return _build_collective(
        "all_reduce",
        tensor,
        {"reduce_op": op, **_group_attrs(group)},
        location,
    )


def all_gather(tensor: "core.SymbolicTensor", axis: int = 0, group: Optional[Group] = None) -> "core.SymbolicTensor":
    """All-gather: concatenate the local tensors of all ranks along ``axis``.

    Every rank receives the full concatenation of the group's local tensors.
    Local-shape semantics: ``result.shape[axis] == tensor.shape[axis] *
    group_size``; all other dims unchanged. Worked example (explicit 4-rank
    group, ``axis=0``): local ``[256, 1024]`` → ``[1024, 1024]``.

    Args:
        tensor: Local SymbolicTensor operand.
        axis: Concatenation axis, normalized Python-style (negative wraps).
        group: Group or ``None`` for the world group (static value).

    Returns:
        SymbolicTensor whose axis dim is input dim × group size. Explicit
        groups: computed by the ir shape_fn via DimExpr multiply. World
        group: the axis dim is ``None`` (runtime-dynamic) — the size is only
        known at execution.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ShapeError: ``axis`` out of range for the operand's rank.

    IR: one ``all_gather`` op; effect kind ``collective``; attrs
    ``{axis, group, group_size}``; 1 operand, 1 result.
    """
    group = _resolve_group(group)
    tensor = _require_symbolic_tensor(tensor)
    axis = _normalize_axis(axis, len(tensor.shape), "axis")
    location = _get_location()
    return _build_collective(
        "all_gather",
        tensor,
        {"axis": axis, **_group_attrs(group)},
        location,
    )


def reduce_scatter(tensor: "core.SymbolicTensor", op: str = "sum", axis: int = 0, group: Optional[Group] = None) -> "core.SymbolicTensor":
    """Reduce-scatter: reduce across ranks with ``op``, then scatter along ``axis``.

    The group's local tensors are first reduced elementwise with ``op``; the
    result is split into ``group_size`` chunks along ``axis`` and each rank
    receives the chunk matching its position in the group. Local-shape
    semantics: ``result.shape[axis] == tensor.shape[axis] // group_size``.

    Args:
        tensor: Local SymbolicTensor operand.
        op: Reduction kind (``"sum"``, ``"max"``, ``"min"``, ``"prod"``).
        axis: Scatter axis, normalized Python-style (negative wraps).
        group: Group or ``None`` for the world group (static value).

    Returns:
        SymbolicTensor whose axis dim is input dim ÷ group size. Explicit
        groups: computed at trace time; the input axis dim must divide
        evenly (``ShapeError`` otherwise). World group: the axis dim is
        ``None`` (runtime-dynamic); divisibility is validated by the
        executor at execution.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ShapeError: ``axis`` out of range; or, for explicit groups,
            ``tensor.shape[axis]`` not divisible by ``len(group.ranks)``.
        ValueError: unknown reduction ``op``.

    IR: one ``reduce_scatter`` op; effect kind ``collective``; attrs
    ``{reduce_op, axis, group, group_size}``; 1 operand, 1 result.
    """
    group = _resolve_group(group)
    _check_reduction(op)
    tensor = _require_symbolic_tensor(tensor)
    axis = _normalize_axis(axis, len(tensor.shape), "axis")
    _check_static_divisibility(tensor.shape, axis, group, "reduce_scatter")
    location = _get_location()
    return _build_collective(
        "reduce_scatter",
        tensor,
        {"reduce_op": op, "axis": axis, **_group_attrs(group)},
        location,
    )


def all_to_all(tensor: "core.SymbolicTensor", split_axis: int, concat_axis: int, group: Optional[Group] = None) -> "core.SymbolicTensor":
    """All-to-all: scatter along ``split_axis``, gather along ``concat_axis``.

    ``tensor`` is split into ``group_size`` chunks along ``split_axis``;
    chunk ``i`` is sent to rank ``i`` of the group; each rank concatenates
    the chunks it received along ``concat_axis``. Local-shape semantics:
    ``result.shape[split_axis] == tensor.shape[split_axis] // group_size``,
    ``result.shape[concat_axis] == tensor.shape[concat_axis] * group_size``
    (equal axes ⇒ shape unchanged); all other dims unchanged.

    Worked example (explicit 4-rank group): local ``[512, 64]`` with
    ``split_axis=0, concat_axis=1`` → four ``[128, 64]`` chunks exchanged →
    concatenated along axis 1 → ``[128, 256]``.

    Args:
        tensor: Local SymbolicTensor operand.
        split_axis: Axis split into ``group_size`` chunks.
        concat_axis: Axis along which received chunks are concatenated.
        group: Group or ``None`` for the world group (static value).

    Returns:
        SymbolicTensor with the split/concat axes reshaped as above.
        Explicit groups: computed at trace time; the split-axis dim must
        divide evenly (``ShapeError`` otherwise). World group: both axes'
        dims are ``None`` (runtime-dynamic).

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ShapeError: ``split_axis``/``concat_axis`` out of range; or, for
            explicit groups, ``tensor.shape[split_axis]`` not divisible by
            ``len(group.ranks)``.

    IR: one ``all_to_all`` op; effect kind ``collective``; attrs
    ``{split_axis, concat_axis, group, group_size}``; 1 operand, 1 result.
    """
    group = _resolve_group(group)
    tensor = _require_symbolic_tensor(tensor)
    split = _normalize_axis(split_axis, len(tensor.shape), "split_axis")
    concat = _normalize_axis(concat_axis, len(tensor.shape), "concat_axis")
    _check_static_divisibility(tensor.shape, split, group, "all_to_all")
    location = _get_location()
    return _build_collective(
        "all_to_all",
        tensor,
        {"split_axis": split, "concat_axis": concat, **_group_attrs(group)},
        location,
    )


def broadcast(tensor: "core.SymbolicTensor", src_rank: int = 0, group: Optional[Group] = None) -> "core.SymbolicTensor":
    """Broadcast: every rank receives the local tensor of ``src_rank``.

    Local shape and dtype are unchanged on every rank.

    Args:
        tensor: Local SymbolicTensor operand (each rank supplies its own;
            only ``src_rank``'s copy is used).
        src_rank: Rank whose tensor is broadcast. Must be a member of the
            group: explicit groups are validated at trace time; for the
            world group only ``src_rank >= 0`` can be checked at trace
            time — membership is validated by the executor at run time.
        group: Group or ``None`` for the world group (static value).

    Returns:
        SymbolicTensor with the same local shape and dtype.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ValueError: ``src_rank`` negative or not in an explicit group.

    IR: one ``broadcast_collective`` op (the shape op ``broadcast`` owns its
    name); effect kind ``collective``; attrs ``{group, group_size}`` plus
    the registry-declared ``src_rank`` (default 0, filled by the builder —
    dist validates ``src_rank`` above but does not record it); 1 operand,
    1 result.
    """
    group = _resolve_group(group)
    if not _is_rank(src_rank) or src_rank < 0:
        raise ValueError(f"src_rank must be a non-negative int, got {src_rank!r}")
    if group.ranks is not None and src_rank not in group.ranks:
        raise ValueError(
            f"src_rank {src_rank} is not in group {group.name!r} ranks "
            f"{group.ranks!r}"
        )
    tensor = _require_symbolic_tensor(tensor)
    location = _get_location()
    return _build_collective(
        "broadcast_collective",
        tensor,
        _group_attrs(group),
        location,
    )


def collective_permute(tensor: "core.SymbolicTensor", source_target_pairs: Tuple[Tuple[int, int], ...], group: Optional[Group] = None) -> "core.SymbolicTensor":
    """Collective permute: pairwise send/recv per ``source_target_pairs``.

    Each ``(src, dst)`` pair means: rank ``src`` sends its local tensor to
    rank ``dst``. Every rank's result is the tensor it received from its
    source; ranks without a source receive a zero tensor of the same shape
    and dtype in the reference executor. Local shape and dtype are
    unchanged.

    Args:
        tensor: Local SymbolicTensor operand (each rank supplies its own).
        source_target_pairs: Non-empty tuple of ``(src, dst)`` int pairs.
            Each rank may appear at most once as ``src`` and at most once
            as ``dst`` (a violation makes the routing ambiguous).
        group: Group or ``None`` for the world group (static value).

    Returns:
        SymbolicTensor with the same local shape and dtype.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ValueError: empty or malformed pairs (not int pairs), duplicate
            ``src`` or ``dst``, or a rank outside an explicit group.

    IR: one ``collective_permute`` op; effect kind ``collective``; attrs
    ``{source_target_pairs, group, group_size}``; 1 operand, 1 result.
    """
    group = _resolve_group(group)
    pairs = _normalize_pairs(source_target_pairs, group)
    tensor = _require_symbolic_tensor(tensor)
    location = _get_location()
    return _build_collective(
        "collective_permute",
        tensor,
        {"source_target_pairs": pairs, **_group_attrs(group)},
        location,
    )
