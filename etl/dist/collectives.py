"""Explicit collective operations (SPMD, local-tensor semantics).

Every collective builds exactly one IR op — effect kind ``collective`` — into
the active trace builder (``trace.current_builder()``), following the same
discipline as the frontend ops in ``etl/ops``. Operands are *local*
:class:`etl.core.SymbolicTensor` values; results carry *local* shapes
computed from the group size. The IR stays backend-neutral: no global tensor
type ever appears, and the reference numpy backend needs no multi-rank
machinery beyond the collective-executor hook (see ``context.py``).

Graph-time only: calling a collective with a concrete ``Tensor`` (or outside
a trace) raises ``TraceError`` — there is no eager mode. ``group=None``
selects the world group (``WORLD_GROUP``).
"""

from __future__ import annotations

from typing import Optional, Tuple

from etl import core, ir, trace

from etl.dist.group import Group, WORLD_GROUP

__all__ = [
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "collective_permute",
]

#: Reduction kinds supported by ``all_reduce`` / ``reduce_scatter`` in v1.
REDUCTIONS: Tuple[str, ...] = ("sum", "max", "min", "prod")


def _resolve_group(group: Optional[Group]) -> Group:
    """Normalize the group argument: ``None`` → ``WORLD_GROUP``.

    Raises:
        TypeError: ``group`` is not a Group.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


def _build_collective(
    op_name: str,
    tensor: "core.SymbolicTensor",
    group: Group,
    attrs: dict,
    result_shape: Tuple[object, ...],
) -> "core.SymbolicTensor":
    """Build the collective IR op into the active builder and wrap the result.

    Flow (mirrors ``etl/ops`` discipline):
    1. ``builder = trace.current_builder()`` — ``TraceError`` when no trace
       is active.
    2. ``opdef = ir.opdef(op_name)``; ``builder.insert(opdef,
       operands=[tensor.value], attrs={**attrs, "group_name": group.name,
       "group_ranks": group.ranks})``.
    3. Wrap the single result value in ``core.SymbolicTensor(value=...,
       dtype=tensor.dtype, shape=result_shape, location=...)``.

    Raises:
        TraceError: no active trace.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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
            name + ranks.

    Returns:
        SymbolicTensor with the same local shape and dtype as ``tensor``.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ValueError: unknown reduction ``op``.

    IR: one ``dist.all_reduce`` op; effect kind ``collective``; attrs
    ``{reduction, group_name, group_ranks}``; 1 operand, 1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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
        groups: computed at trace time via DimExpr multiply. World group:
        the axis dim is ``None`` (runtime-dynamic) — the size is only known
        at execution.

    Raises:
        TraceError: no active trace, or a concrete ``Tensor`` operand.
        ShapeError: ``axis`` out of range for the operand's rank.

    IR: one ``dist.all_gather`` op; effect kind ``collective``; attrs
    ``{axis, group_name, group_ranks}``; 1 operand, 1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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

    IR: one ``dist.reduce_scatter`` op; effect kind ``collective``; attrs
    ``{reduction, axis, group_name, group_ranks}``; 1 operand, 1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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

    IR: one ``dist.all_to_all`` op; effect kind ``collective``; attrs
    ``{split_axis, concat_axis, group_name, group_ranks}``; 1 operand,
    1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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

    IR: one ``dist.broadcast`` op; effect kind ``collective``; attrs
    ``{src_rank, group_name, group_ranks}``; 1 operand, 1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)


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

    IR: one ``dist.collective_permute`` op; effect kind ``collective``;
    attrs ``{source_target_pairs, group_name, group_ranks}``; 1 operand,
    1 result.
    """
    raise NotImplementedError  # implemented in Phase 2 (Manager)
