"""Internal helpers shared by the dist frontend (collectives + context).

NOT part of the public API — nothing outside ``etl.dist`` may import from this
module. The helpers mirror the ``etl/ops`` frontend discipline (operand rules,
location capture, active-builder hook, result wrapping) but are private to
``dist``: ``etl/ops/_utils.py`` is a sibling architecture stub that dist must
not depend on.

Import layering (binding, root CONTEXT.md): this module may import
``etl.core``, ``etl.ir``, ``etl.trace``, and ``etl.dist.group`` — never
``etl.backends`` / ``etl.pipeline`` / ``etl.persist``.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Optional, Tuple

from etl import core, ir

from etl.dist.group import Group, WORLD_GROUP

__all__ = [
    "REDUCTIONS",
    "_require_symbolic_tensor",
    "_get_location",
    "_wrap_result",
    "_resolve_group",
    "_group_attrs",
    "_check_reduction",
    "_is_rank",
    "_normalize_axis",
    "_check_static_divisibility",
    "_normalize_pairs",
]

#: Reduction kinds supported by ``all_reduce`` / ``reduce_scatter`` in v1.
REDUCTIONS: Tuple[str, ...] = ("sum", "max", "min", "prod")

#: Absolute path of the ``etl`` package directory (frames inside it are
#: skipped during location capture).
_ETL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _require_symbolic_tensor(x: Any) -> "core.SymbolicTensor":
    """Normalize a collective operand into a ``SymbolicTensor``.

    - ``SymbolicTensor``: returned unchanged.
    - ``core.Tensor``: raises ``core.TraceError`` with the mandated
      three-option message — make the tensor an explicit input (trace with a
      ``TensorSpec``), or embed it explicitly with ``etl.constant(...)``
      (snapshot semantics), or note that there is no eager mode (use
      ``etl.evaluate(...)`` to build and run a graph).
    - Anything else: raises ``TypeError`` naming the accepted operand kinds.

    Raises:
        core.TraceError: ``x`` is a concrete ``Tensor`` (no eager mode).
        TypeError: ``x`` is not a SymbolicTensor / Tensor.
    """
    if isinstance(x, core.SymbolicTensor):
        return x
    if isinstance(x, core.Tensor):
        raise core.TraceError(
            f"collectives require SymbolicTensor graph operands, got a "
            f"concrete Tensor (shape={x.shape}, dtype={x.dtype}). There is "
            "no eager mode: make the tensor an explicit input (trace with a "
            "TensorSpec), or embed it explicitly via etl.constant(...) "
            "(snapshot semantics), or use etl.evaluate(...) to build and "
            "run a graph."
        )
    raise TypeError(
        "collective operands must be SymbolicTensor graph values, got "
        f"{type(x).__name__}"
    )


def _get_location(depth: int = 1) -> "ir.Location":
    """Capture the Python call site of the *user* collective call.

    Walks ``inspect.stack()`` starting ``depth`` frames up the call stack,
    skipping every frame whose filename contains the ``etl`` package
    directory (so internal helper frames never pollute user locations), and
    returns ``ir.Location(file, line, 0)`` for the first external frame.

    Returns:
        An ``ir.Location`` carrying the external call site, or
        ``ir.Location.unknown()`` when capture is impossible.

    Raises:
        Never raises — location capture failure degrades to
        ``ir.Location.unknown()`` (a missing location must not break
        tracing).
    """
    try:
        for frame in inspect.stack()[depth:]:
            if _ETL_DIR in frame.filename:
                continue
            return ir.Location(frame.filename, frame.lineno, 0)
    except Exception:  # pragma: no cover - capture must never break tracing
        pass
    return ir.Location.unknown()


def _wrap_result(value: Any, location: Any) -> "core.SymbolicTensor":
    """Wrap an op's result ``ir.Value`` in a ``SymbolicTensor`` facade.

    Dtype and shape are read from ``value.type`` (the op's shape_fn already
    applied the collective shape rules — dist never recomputes them).
    """
    value_type = value.type
    shape = value_type.shape
    # TEMPORARY CORE-GAP WORKAROUND: core.SymbolicTensor.__post_init__
    # rejects None shape entries with TypeError (while ir.ValueType and
    # core.TensorSpec allow them). World-group collectives infer None dims
    # (runtime-dynamic), so construct the facade with a sanitized shape
    # (None -> 0) and immediately restore the true IR shape directly —
    # keeping world-group tracing functional once the ir-side group_size
    # attr issue (ATTR_INT rejects None) is fixed upstream.
    sanitized = tuple(0 if dim is None else dim for dim in shape)
    result = core.SymbolicTensor(
        value=value,
        dtype=value_type.dtype,
        shape=sanitized,
        location=location,
    )
    if any(dim is None for dim in shape):
        object.__setattr__(result, "shape", tuple(shape))
    return result


def _resolve_group(group: Optional[Group]) -> Group:
    """Normalize the group argument: ``None`` → ``WORLD_GROUP``.

    Raises:
        TypeError: ``group`` is not a Group.
    """
    if group is None:
        return WORLD_GROUP
    if not isinstance(group, Group):
        raise TypeError(
            f"group must be a Group or None (world group), got "
            f"{type(group).__name__}"
        )
    return group


def _group_attrs(group: Group) -> dict:
    """The canonical group attribute pair: name + size (None for world)."""
    return {"group": group.name, "group_size": group.size()}


def _check_reduction(op: str) -> None:
    """Validate a reduction kind.

    Raises:
        ValueError: ``op`` is not one of ``"sum"``, ``"max"``, ``"min"``,
            ``"prod"``.
    """
    if op not in REDUCTIONS:
        raise ValueError(
            f"unknown reduction op {op!r}; expected one of {REDUCTIONS!r}"
        )


def _is_rank(x: Any) -> bool:
    """True for plain non-negative ints (bools excluded — numpy-style checks)."""
    return isinstance(x, int) and not isinstance(x, bool)


def _normalize_axis(axis: Any, rank: int, what: str) -> int:
    """Normalize a (possibly negative) axis into ``range(rank)``.

    Negative axes wrap Python-style (add ``rank``). The normalized
    non-negative axis is returned (and recorded in op attrs — deterministic
    serialization).

    Raises:
        core.ShapeError: ``axis`` is not an int (bool rejected) or out of
            range for ``rank``.
    """
    if not _is_rank(axis):
        raise core.ShapeError(f"{what}: expected an int axis, got {axis!r}")
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise core.ShapeError(f"{what}: axis {axis} out of range for rank {rank}")
    return normalized


def _check_static_divisibility(
    shape: Tuple[Any, ...], axis: int, group: Group, what: str
) -> None:
    """Explicit-group pre-check: static int dims must divide evenly.

    Symbolic (``Dim``/``DimExpr``) and ``None`` dims are skipped — their
    divisibility is deferred to the runtime executor (world group: the ir
    shape_fn defers via ``None`` dims; explicit symbolic groups: the ir
    shape_fn raises when the symbolic division is impossible).

    Raises:
        core.ShapeError: a static int dim at ``axis`` is not divisible by
            ``len(group.ranks)``.
    """
    if group.ranks is None:
        return
    dim = shape[axis]
    if _is_rank(dim) and dim % len(group.ranks) != 0:
        raise core.ShapeError(
            f"{what}: dimension {dim} at axis {axis} is not divisible by "
            f"group size {len(group.ranks)} (group {group.name!r})"
        )


def _normalize_pairs(
    source_target_pairs: Any, group: Group
) -> Tuple[Tuple[int, int], ...]:
    """Validate/normalize ``source_target_pairs`` into a tuple of int pairs.

    Rules: non-empty tuple of 2-element tuples of non-bool ints >= 0; each
    rank may appear at most once as source and at most once as destination;
    for explicit groups every rank must be a member.

    Raises:
        ValueError: empty/malformed pairs, duplicate src or dst, or a rank
            outside an explicit group.
    """
    if not isinstance(source_target_pairs, tuple):
        raise ValueError(
            "source_target_pairs must be a tuple of (src, dst) pairs, got "
            f"{type(source_target_pairs).__name__}"
        )
    if not source_target_pairs:
        raise ValueError(
            "source_target_pairs must be a non-empty tuple of (src, dst) pairs"
        )
    pairs = []
    for pair in source_target_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not _is_rank(pair[0])
            or not _is_rank(pair[1])
            or pair[0] < 0
            or pair[1] < 0
        ):
            raise ValueError(
                f"source_target_pairs entries must be non-negative (src, dst) "
                f"int pairs, got {pair!r}"
            )
        pairs.append(pair)
    sources = [src for src, _ in pairs]
    targets = [dst for _, dst in pairs]
    if len(set(sources)) != len(sources):
        raise ValueError(
            f"source_target_pairs: duplicate source rank in {pairs!r}"
        )
    if len(set(targets)) != len(targets):
        raise ValueError(
            f"source_target_pairs: duplicate destination rank in {pairs!r}"
        )
    if group.ranks is not None:
        for src in sources:
            if src not in group.ranks:
                raise ValueError(
                    f"source_target_pairs: source rank {src} not in group "
                    f"{group.name!r} ranks {group.ranks!r}"
                )
        for dst in targets:
            if dst not in group.ranks:
                raise ValueError(
                    f"source_target_pairs: destination rank {dst} not in "
                    f"group {group.name!r} ranks {group.ranks!r}"
                )
    return tuple(pairs)
