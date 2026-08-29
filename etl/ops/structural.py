"""Structural / creation frontends: tile, flip, roll, stack, diag, isnan,
nan_to_num, clamp, eye, linspace (graph ops).

Implementation choices (design notes — see this node's CONTEXT.md):

- ``stack``: composition over ``reshape`` (insert a size-1 axis) +
  ``concatenate`` — the two existing IR ops, exactly ``np.stack``'s
  definition. Input shapes must agree statically (``ShapeError``); ``axis``
  normalized to ``[0, rank]``.
- ``flip``/``roll``: dedicated IR ops + numpy kernels (``np.flip`` /
  ``np.roll`` exact). ``roll`` normalizes ``shift`` to a per-axis tuple
  frontend-side; with ``axis=None`` a multi-entry shift folds to its sum
  (numpy's flattened-roll semantics).
- ``clamp``: pure composition over ``maximum``/``minimum`` (no dedicated
  IR op) — the composition inherits their vjp/batching rules and reproduces
  numpy 2.x ``np.clip`` dtype behavior EXACTLY: Python-scalar bounds whose
  natural dtype ``can_cast`` to ``x``'s dtype with ``same_kind`` are
  pre-cast to ``x``'s dtype (int bound on an int tensor stays int; int
  bound on a float tensor stays float), and bounds the rule rejects fall
  back to weak scalar promotion (float bound on an int tensor promotes to
  float64 — numpy 2.x; numpy 1.x raised ``TypeError`` via ``same_kind``
  casts; we match the installed numpy). ``min > max`` yields all-``max``
  (numpy semantics) and NaN bounds propagate.
- ``isnan``: composition over ``not_equal(x, x)`` (complex-safe comparison
  kernel — True iff a real or imaginary part is NaN, matching ``np.isnan``).
- ``nan_to_num``: dedicated IR op + numpy kernel; scalar replacements
  ``nan``/``posinf``/``neginf`` are IR attributes (``None`` = numpy default).
- ``eye``/``linspace``: Constant-op compositions (the ``enp`` creation
  pattern) — no new IR ops; all parameters are static Python values.
  ``linspace`` defaults to float64 (numpy exactness; deliberate deviation
  from the etl float32 creation convention — documented).

Transform coverage: no vjp/batching rules for the dedicated IR ops
(``tile``/``flip``/``roll``/``diag``/``nan_to_num`` → documented
``TransformError``); ``stack``/``clamp``/``isnan``/``eye``/``linspace``
inherit their components' rules.
"""
from __future__ import annotations

import numpy as np

from etl import core

from . import _utils
from . import comparison as _comparison
from . import constant as _constant
from . import elementwise as _elementwise
from . import indexing as _indexing

__all__ = [
    "tile",
    "flip",
    "roll",
    "stack",
    "diag",
    "isnan",
    "nan_to_num",
    "clamp",
    "eye",
    "linspace",
]


def _wrap(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result, reading dtype/shape back from the IR."""
    result = op.result
    return core.SymbolicTensor(
        value=result,
        dtype=result.type.dtype,
        shape=result.type.shape,
        location=loc,
    )


def _plain_int(value, name: str) -> None:
    """Type-check a static plain-int parameter."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int, got {value!r}")


def _int_tuple(value, name: str, *, allow_none=False):
    """Normalize an int-or-tuple-of-ints parameter to a tuple of ints
    (``None`` passes through when ``allow_none``)."""
    if value is None and allow_none:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (tuple, list)):
        seq = tuple(value)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in seq):
            raise TypeError(f"{name}: entries must be ints, got {value!r}")
        return seq
    raise TypeError(
        f"{name}: expected an int or a tuple of ints, got {value!r}"
    )


def tile(x, reps) -> "core.SymbolicTensor":
    """Repeat the tensor per numpy ``tile`` semantics.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        reps: int or tuple of non-negative ints (per-dim repetition counts;
            may be shorter or longer than the operand rank — the operand is
            promoted with leading size-1 dims, numpy tile semantics).

    Returns:
        ``SymbolicTensor`` with the tiled shape; dtype preserved.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``reps`` is not an int or tuple of ints.
        core.ShapeError: negative ``reps`` entries (numpy raises
            ``ValueError``; etl uses ``ShapeError``).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    reps = _int_tuple(reps, "tile: reps")
    if any(r < 0 for r in reps):
        raise core.ShapeError(
            f"tile: reps entries must be non-negative, got {reps!r}"
        )
    op = builder.create(
        "tile", operands=(x.value,), attributes={"reps": reps}, location=loc
    )
    return _wrap(op, loc)


def flip(x, axes) -> "core.SymbolicTensor":
    """Reverse the tensor along the given axes (numpy ``flip`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        axes: ``None`` (all axes), an int, or a tuple of ints (negative
            indices supported).

    Returns:
        ``SymbolicTensor`` with the same shape and dtype, axes reversed.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``axes`` is not ``None``/int/tuple of ints.
        core.ShapeError: axis out of range (numpy ``AxisError``, a
            ``ValueError`` subclass, converted by the kernel).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    axes = _int_tuple(axes, "flip: axes", allow_none=True)
    if axes is None:
        attr = None  # flip all axes
    else:
        attr = axes  # normalized tuple of ints (np.flip accepts a tuple)
    op = builder.create(
        "flip", operands=(x.value,), attributes={"axes": attr}, location=loc
    )
    return _wrap(op, loc)


def roll(x, shift, axis=None) -> "core.SymbolicTensor":
    """Roll the tensor by a shift (numpy ``roll`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        shift: int or tuple of ints (per-axis amounts; a scalar shift
            applies to every axis in ``axis``).
        axis: ``None`` (flattened roll), an int, or a tuple of ints. With
            ``axis=None`` a multi-entry shift folds to its sum (numpy's
            flattened-roll semantics).

    Returns:
        ``SymbolicTensor`` with the same shape and dtype, elements rolled.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``shift``/``axis`` of the wrong kind.
        ValueError: ``shift`` and ``axis`` tuple lengths differ (numpy
            message).
        core.ShapeError: axis out of range (numpy ``AxisError`` converted by
            the kernel).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    shift = _int_tuple(shift, "roll: shift")
    if axis is None:
        if len(shift) > 1:
            shift = (sum(shift),)  # numpy flat-roll: tuple shift sums
    elif isinstance(axis, int) and not isinstance(axis, bool):
        if len(shift) > 1:
            raise ValueError(
                "roll: if shift is a tuple, axis must be a tuple of the "
                "same length"
            )
    else:
        axis_t = _int_tuple(axis, "roll: axis")
        if len(shift) == 1:
            shift = shift * len(axis_t)
        elif len(shift) != len(axis_t):
            raise ValueError(
                "roll: the length of shift must equal the length of axis"
            )
        axis = axis_t
    op = builder.create(
        "roll",
        operands=(x.value,),
        attributes={"shift": shift, "axis": axis},
        location=loc,
    )
    return _wrap(op, loc)


def stack(tensors, axis=0) -> "core.SymbolicTensor":
    """Stack tensors along a new axis (numpy ``stack`` semantics).

    Composition over ``reshape`` (insert a size-1 axis) + ``concatenate`` —
    no dedicated IR op.

    Args:
        tensors: non-empty tuple/list of ``SymbolicTensor``s (or Python
            scalars) with identical shapes.
        axis: int; position of the new axis in ``[0, rank]`` (negative
            values count from the end).

    Returns:
        ``SymbolicTensor`` of shape ``shape[:axis] + (n,) + shape[axis:]``
        where ``n`` is the number of inputs; dtype = the promoted input
        dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        ValueError: empty ``tensors`` (numpy message).
        TypeError: ``tensors`` is not a tuple/list; ``axis`` not an int.
        core.ShapeError: input shapes differ; ``axis`` out of range.
    """
    if not isinstance(tensors, (tuple, list)):
        raise TypeError(
            f"stack: tensors must be a tuple/list, got {type(tensors).__name__}"
        )
    if not tensors:
        raise ValueError("stack: need at least one array to stack")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"stack: axis must be an int, got {axis!r}")
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    syms = [
        t if isinstance(t, core.SymbolicTensor) else _utils.as_operand(t, location=loc)
        for t in tensors
    ]
    first_shape = syms[0].shape
    rank = len(first_shape)
    for i, t in enumerate(syms[1:], start=1):
        if t.shape != first_shape:
            raise core.ShapeError(
                f"stack: all input arrays must have the same shape, got "
                f"shapes {first_shape} and {t.shape} (input {i})"
            )
    original = axis
    if axis < 0:
        axis += rank + 1
    if not 0 <= axis <= rank:
        raise core.ShapeError(
            f"stack: axis {original} out of bounds for rank {rank} inputs"
        )
    expanded = [
        _indexing.reshape(t, first_shape[:axis] + (1,) + first_shape[axis:])
        for t in syms
    ]
    return _indexing.concatenate(expanded, axis=axis)


def diag(x) -> "core.SymbolicTensor":
    """Extract the main diagonal, or build a diagonal matrix (numpy ``diag``
    semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar of rank 1 (→ ``(n, n)``
            diagonal matrix) or rank 2 (→ the main diagonal ``(min(m, n),)``).

    Returns:
        ``SymbolicTensor``; dtype preserved in both directions.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        core.ShapeError: rank not 1 or 2 (numpy raises ``ValueError``; etl
            uses ``ShapeError``).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    if len(x.shape) not in (1, 2):
        raise core.ShapeError(
            f"diag: input must be 1- or 2-d, got rank {len(x.shape)}"
        )
    op = builder.create("diag", operands=(x.value,), location=loc)
    return _wrap(op, loc)


def isnan(x) -> "core.SymbolicTensor":
    """Elementwise NaN test (numpy ``isnan`` semantics) → bool tensor.

    Composition over ``not_equal(x, x)``: true iff any component is NaN
    (integer inputs are never NaN; complex inputs are NaN when the real OR
    imaginary part is NaN). No dedicated IR op.

    Args:
        x: ``SymbolicTensor`` or Python scalar of any numeric dtype.

    Returns:
        ``SymbolicTensor`` of bool with ``x``'s shape.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
    """
    return _comparison.not_equal(x, x)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None) -> "core.SymbolicTensor":
    """Replace NaN / +inf / -inf entries (numpy ``nan_to_num`` semantics).

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        nan: int/float; replacement for NaN entries (default 0.0).
        posinf: int/float or ``None``; replacement for +inf entries
            (``None`` = the numpy default, the dtype's max finite value).
        neginf: int/float or ``None``; replacement for -inf entries
            (``None`` = the numpy default, the dtype's min finite value).
            Only the corresponding infinity is replaced per argument — e.g.
            ``nan_to_num(x, nan=0, posinf=1)`` leaves -inf alone.

    Returns:
        ``SymbolicTensor`` with ``x``'s exact shape and dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: any replacement is not an int/float (or ``None`` for the
            infinities).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x = _utils.as_operand(x, location=loc)
    for name, value in (("nan", nan), ("posinf", posinf), ("neginf", neginf)):
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"nan_to_num: {name} must be an int/float (or None for the "
                f"infinity replacements), got {value!r}"
            )
    op = builder.create(
        "nan_to_num",
        operands=(x.value,),
        attributes={"nan": nan, "posinf": posinf, "neginf": neginf},
        location=loc,
    )
    return _wrap(op, loc)


def clamp(x, min, max) -> "core.SymbolicTensor":
    """Clamp values into ``[min, max]`` (numpy ``clip`` semantics).

    Pure composition over ``maximum``/``minimum`` (no dedicated IR op):
    ``minimum(maximum(x, min), max)``. Python-scalar bounds whose natural
    dtype ``np.can_cast(..., casting="same_kind")`` accepts are pre-cast to
    ``x``'s dtype exactly like ``np.clip``'s scalar handling (int bound on
    an int tensor stays int; int bound on a float tensor stays float; float
    bound on a float tensor stays float). Bounds the rule rejects (a float
    bound on an int tensor) fall back to weak promotion per NEP 50 — the
    result promotes to float64, matching the installed numpy 2.x (numpy 1.x
    raised ``TypeError`` via ``same_kind`` casts); symbolic (tensor) bounds
    broadcast per ``DimExpr`` rules and promote with the tensor.
    ``min > max`` returns the all-``max`` result and NaN bounds propagate,
    matching numpy.

    Args:
        x: ``SymbolicTensor`` or Python scalar.
        min: ``None``, Python scalar, or ``SymbolicTensor`` (lower bound;
            ``None`` = unbounded below).
        max: ``None``, Python scalar, or ``SymbolicTensor`` (upper bound;
            ``None`` = unbounded above).

    Returns:
        ``SymbolicTensor`` with the broadcast shape and the numpy-2.x
        promoted dtype.

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand; both
            bounds are ``None``.
        TypeError: a bound is neither ``None``/scalar/``SymbolicTensor``.
        core.DTypeError: complex input (via ``maximum``/``minimum``, like
            ``np.clip``).
    """
    if min is None and max is None:
        raise TypeError("clamp: at least one of min/max must be given")
    if isinstance(x, core.SymbolicTensor):
        # Symbolic operand: pre-cast scalar bounds per np.clip's same_kind
        # rule, then the usual maximum/minimum composition. Scalar/concrete
        # ``x`` keeps the plain composition path below (identical error
        # semantics — no eager two-scalar mode).
        builder = _utils.check_in_trace()
        loc = _utils.get_location(depth=2)
        lo = x
        if min is not None:
            lo = _elementwise.maximum(x, _clip_bound(builder, x, min, loc))
        if max is not None:
            return _elementwise.minimum(lo, _clip_bound(builder, x, max, loc))
        return lo
    lo = _elementwise.maximum(x, min) if min is not None else x
    return _elementwise.minimum(lo, max) if max is not None else lo


#: Exact Python scalar kinds eligible for the np.clip-style same_kind
#: pre-cast (numpy scalars deliberately excluded — they fall through to the
#: canonical ``maximum``/``minimum`` operand errors).
_CLIP_SCALAR_KINDS = (bool, int, float, complex)


def _clip_bound(builder, x, bound, loc):
    """Normalize one clamp bound against ``x`` per ``np.clip`` scalar rules.

    A Python scalar whose natural dtype ``can_cast`` to ``x.dtype`` with
    ``casting="same_kind"`` (int bound on an int/float tensor, float bound
    on a float tensor) becomes a 0-d Constant of ``x.dtype``. Bounds the
    rule rejects (float bound on an int tensor) and symbolic tensor bounds
    pass through unchanged to the ``maximum``/``minimum`` weak-promotion /
    broadcast path.
    """
    if type(bound) not in _CLIP_SCALAR_KINDS:
        return bound
    if not np.can_cast(np.asarray(bound).dtype, x.dtype, casting="same_kind"):
        return bound
    payload = np.asarray(bound, dtype=x.dtype)
    op = builder.create("constant", attributes={"value": payload}, location=loc)
    return core.SymbolicTensor(
        value=op.result, dtype=np.dtype(x.dtype), shape=(), location=loc
    )


def eye(n, m=None, dtype=core.float32) -> "core.SymbolicTensor":
    """Identity matrix (numpy ``eye`` semantics) as a Constant op.

    Graph creation op: the matrix data is materialized at trace time and
    embedded via ``etl.constant`` (the ``enp`` creation pattern) — no new IR
    op. ``n``/``m`` must be static Python ints; ``dtype`` defaults to
    float32 (the etl creation-op convention).

    Args:
        n: int; number of rows.
        m: int or ``None``; number of columns (``None`` = ``n``).
        dtype: numpy dtype for the output.

    Returns:
        ``SymbolicTensor`` of shape ``(n, m)``.

    Raises:
        core.TraceError: no active trace.
        TypeError: ``n``/``m`` are not Python ints.
        ValueError: negative ``n``/``m`` (numpy parity, raised by numpy).
    """
    _plain_int(n, "eye: n")
    if m is None:
        m = n
    else:
        _plain_int(m, "eye: m")
    return _constant.constant(core.tensor(np.eye(n, m, dtype=dtype)))


def linspace(start, stop, num, dtype=None) -> "core.SymbolicTensor":
    """Evenly spaced numbers (numpy ``linspace`` semantics) as a Constant op.

    Graph creation op: all parameters are static Python values at trace time
    (symbolic ``start``/``stop`` raise ``TraceError`` — deferred to v2, like
    ``enp.arange``). ``dtype=None`` defaults to float64 — numpy exactness,
    a DELIBERATE deviation from the etl float32 creation convention (the
    objective's recommendation; tests pass explicit dtypes where float32 is
    wanted).

    Args:
        start: int/float; interval start.
        stop: int/float; interval end (inclusive).
        num: int; number of samples.
        dtype: numpy dtype or ``None`` (→ float64).

    Returns:
        ``SymbolicTensor`` of shape ``(num,)``.

    Raises:
        core.TraceError: no active trace; symbolic ``start``/``stop``.
        TypeError: ``start``/``stop`` are not Python numbers; ``num`` is not
            a Python int.
        ValueError: ``num < 0`` (numpy parity, raised by numpy).
    """
    builder = _utils.check_in_trace()
    for name, value in (("start", start), ("stop", stop)):
        if isinstance(value, (core.Dim, core.DimExpr)):
            raise core.TraceError(
                f"linspace: symbolic {name}={value!r} is not supported — "
                "bounds must be concrete Python numbers at trace time; "
                "dynamic-length constants (symbolic linspace) are deferred "
                "to v2"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"linspace: {name} must be a Python int or float, got "
                f"{value!r}"
            )
    _plain_int(num, "linspace: num")
    if dtype is None:
        dtype = np.float64  # numpy exactness; deviation from etl float32
    return _constant.constant(core.tensor(np.linspace(start, stop, num, dtype=dtype)))
