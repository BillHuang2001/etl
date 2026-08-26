"""Internal helpers shared by all frontend op modules.

NOT part of the public API — nothing outside ``etl.ops`` may import from this
module. Public behavior contracts (operand rules, dtype promotion, symbolic
broadcasting, location capture) are defined here once and used by every op
function; the rules are documented in this node's ``CONTEXT.md``.

Import layering (binding, see root ``CONTEXT.md``): this module may import
``etl.core``, ``etl.ir``, and ``etl.trace`` (``trace`` ONLY for the
active-builder hook). It must never import ``etl.backends``.
"""
from __future__ import annotations

import inspect
import os
from typing import Iterable, Tuple, Union

import numpy as np

from etl import core
from etl import ir
from etl import trace

__all__ = [
    "check_in_trace",
    "get_location",
    "as_operand",
    "weak_scalar_dtype",
    "promote_dtypes",
    "broadcast_shapes",
    "reduced_shape",
    "normalize_axes",
]

#: Environment variable that disables call-site location capture (any non-empty
#: value equal to "1" disables). See :func:`get_location`.
ETL_DISABLE_LOCATIONS_ENV = "ETL_DISABLE_LOCATIONS"

#: Natural (un-hinted) dtypes for Python scalars promoted to 0-d constant ops.
#: Numpy's python-scalar defaults: int -> int64, float -> float64,
#: complex -> complex128, bool -> bool.
PYTHON_SCALAR_DTYPES = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}

#: Root directory of the ``etl`` package (realpath). Frames whose filename
#: lives under this directory are internal helper frames and never produce
#: user-facing locations (see :func:`get_location`).
_ETL_ROOT = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: dtype NAMES a Python ``bool``/``int`` scalar is weak toward (NEP 50
#: semantics implemented explicitly — see :func:`weak_scalar_dtype`).
_WEAK_FLOAT_DTYPES = frozenset(
    {"float16", "float32", "float64", "complex64", "complex128"}
)

#: Mantissa bits per weak target dtype name: an int scalar is "exactly
#: representable" in the target iff ``abs(value) < 2 ** bits`` (float16 has 11
#: mantissa bits, float32 24, float64 53; complex64/128 share their
#: component precisions).
_WEAK_INT_BITS = {
    "float16": 11,
    "float32": 24,
    "float64": 53,
    "complex64": 24,
    "complex128": 53,
}

Scalar = Union[bool, int, float, complex]
ShapeElement = Union[int, core.Dim, core.DimExpr]
ShapeTuple = Tuple[ShapeElement, ...]


def check_in_trace() -> "ir.Builder":
    """Return the active IR builder, or raise ``TraceError`` if there is none.

    Every op function must call this FIRST, before any promotion/inference
    work, so that "op used outside a trace" fails deterministically with a
    clear message. There is no eager mode in etl.

    Returns:
        The active ``ir.Builder`` obtained from ``trace.current_builder()``.

    Raises:
        core.TraceError: no trace is active (``trace.current_builder()``
            returned ``None``). The message must mention the supported ways to
            obtain a trace (``etl.trace`` / ``@etl.defn`` / ``etl.evaluate``).
    """
    # Delegate entirely to the active-builder hook: `trace.current_builder()`
    # raises the canonical TraceError (with the directing message) when no
    # builder is active — this module must not duplicate that logic.
    return trace.current_builder()


def get_location(depth: int = 1) -> "ir.Location":
    """Capture the Python call site of the *user* op call.

    Walks ``inspect.stack()`` starting ``depth`` frames up the call stack,
    skipping every frame whose filename lives inside the ``etl`` package
    directory (so internal helper frames never pollute user locations), and
    returns ``ir.Location(file=..., line=...)`` for the first external frame.

    If the environment variable ``ETL_DISABLE_LOCATIONS`` is set to ``"1"``,
    returns ``ir.Location.unknown()`` immediately (zero stack-walk cost; useful
    for tracing-heavy benchmarks and for tests asserting location-independent
    graph equality).

    Args:
        depth: Number of frames to skip before the walk begins. Op functions
            call ``get_location(depth=2)`` (skip ``get_location`` itself and
            the op function); helpers inside op modules adjust accordingly.

    Returns:
        An ``ir.Location`` carrying the external call site, or
        ``ir.Location.unknown()`` when capture is disabled or no external
        frame exists.

    Raises:
        Never raises — location capture failure degrades to
        ``ir.Location.unknown()`` (a missing location must not break tracing).
    """
    if os.environ.get(ETL_DISABLE_LOCATIONS_ENV) == "1":
        return ir.Location.unknown()
    try:
        for frame in inspect.stack()[depth:]:
            filename = frame.filename
            if not filename or filename.startswith("<"):
                continue  # no real source path (builtins, frozen modules)
            if _inside_etl_package(filename):
                continue  # internal helper frame
            return ir.Location(file=filename, line=frame.lineno, col=0)
    except Exception:
        # Contract: location capture must never break tracing — degrade.
        pass
    return ir.Location.unknown()


def _inside_etl_package(filename: str) -> bool:
    """True when ``filename`` lives inside the ``etl`` package directory."""
    try:
        return os.path.commonpath([_ETL_ROOT, os.path.realpath(filename)]) == _ETL_ROOT
    except ValueError:
        return False  # different drives (Windows) — never inside


def as_operand(x, *, dtype_hint=None, location=None) -> "core.SymbolicTensor":
    """Normalize one op operand into a ``SymbolicTensor``.

    Unified operand rule for EVERY op function (see CONTEXT.md, "Unified
    semantics"):

    - ``SymbolicTensor``: returned unchanged.
    - Python scalar (``bool``/``int``/``float``/``complex``): transparently
      promoted to a 0-d Constant op built into the active builder. Its dtype
      is :func:`weak_scalar_dtype`-promoted against ``dtype_hint`` (the other
      operand's dtype for binary ops; ``None`` for unary ops, which yields the
      natural scalar dtype from ``PYTHON_SCALAR_DTYPES``). Shape is ``()``.
    - ``core.Tensor``: raises ``core.TraceError`` with the mandated
      three-option message: (1) make it an explicit input
      (``TensorSpec`` → trace parameter), (2) embed it explicitly with
      ``etl.constant`` (snapshot semantics, warning for large tensors), or
      (3) note that etl has no eager mode — use ``etl.evaluate`` to build and
      run a graph.
    - Anything else (``list``, ``np.ndarray``, ``str``, ``None``, ...):
      raises ``TypeError`` (a Python programming error, not an etl semantics
      error) with a message naming the accepted operand kinds.

    Args:
        x: The operand to normalize.
        dtype_hint: numpy dtype (or etl dtype) of the other operand, used for
            NEP-50-style weak scalar promotion; ``None`` for unary contexts.
        location: Optional ``ir.Location`` to attach to a generated Constant
            op; if ``None``, the caller should pass a captured location.

    Returns:
        A ``core.SymbolicTensor`` wrapping either the input or a fresh 0-d
        Constant IR value in the active builder.

    Raises:
        core.TraceError: no active trace, or ``x`` is a concrete ``Tensor``.
        TypeError: ``x`` is an unsupported non-scalar Python value.
    """
    if isinstance(x, core.SymbolicTensor):
        return x
    scalar_kind = type(x)
    if scalar_kind in PYTHON_SCALAR_DTYPES:
        # NOTE: numpy scalars are deliberately NOT Python scalars: `type()`
        # distinguishes np.float64 (a float subclass) / np.int64 / np.bool_
        # / np.complex128 from the exact builtin kinds — they fall through
        # to the TypeError below.
        builder = check_in_trace()
        dtype = PYTHON_SCALAR_DTYPES[scalar_kind]
        if dtype_hint is not None:
            dtype = weak_scalar_dtype(x, dtype_hint)
        if location is None:
            location = get_location()
        payload = np.asarray(x, dtype=dtype)
        op = builder.create(
            "constant", attributes={"value": payload}, location=location
        )
        return core.SymbolicTensor(
            value=op.result, dtype=np.dtype(dtype), shape=(), location=location
        )
    if isinstance(x, core.Tensor):
        raise core.TraceError(
            "Concrete Tensor operands are not allowed in graph ops (etl has "
            "no eager mode): (1) pass the tensor as an explicit input "
            "(TensorSpec at trace time), (2) embed its data explicitly with "
            "etl.constant (snapshots the data and warns for large tensors), "
            "or (3) build and run a graph with etl.evaluate."
        )
    raise TypeError(
        f"unsupported operand type {type(x).__name__}: op operands must be "
        "SymbolicTensor or Python scalars (bool, int, float, complex); "
        "numpy scalars/arrays are not Python scalars"
    )


def weak_scalar_dtype(value: Scalar, array_dtype) -> np.dtype:
    """Dtype of a Python scalar constant promoted against an array dtype.

    Implements NEP 50 weak-scalar semantics EXPLICITLY so behavior is
    identical on numpy 1.x and 2.x (``np.result_type`` only applies weak
    promotion to Python scalars on numpy >= 2.0; this library pins
    ``numpy>=1.24`` and must not depend on that). Rules, by scalar kind:

    - ``bool``: weak toward float/complex — result is ``array_dtype`` when it
      is float16/32/64 or complex64/128; otherwise ``np.result_type(bool,
      array_dtype)`` (i.e. integer/uint tensors promote as usual).
    - ``int``: weak toward float16/32/64 and complex64/128 — result is
      ``array_dtype`` when the value is exactly representable in it, else
      ``float64`` (complex64 with an unrepresentable int → ``complex128``);
      integer/uint/bool tensors promote via ``np.result_type``.
    - ``float``: weak toward complex64 ONLY — ``float + complex64 → complex64``;
      otherwise ``np.result_type(float64, array_dtype)`` (so ``float + float32
      → float64``, matching NEP 50).
    - ``complex``: weak toward nothing — always ``np.result_type(complex128,
      array_dtype)``.

    Args:
        value: The Python scalar.
        array_dtype: numpy dtype of the symbolic/tensor operand.

    Returns:
        The numpy dtype the 0-d constant must be created with.

    Raises:
        TypeError: ``value`` is not a Python scalar.
    """
    scalar_kind = type(value)
    if scalar_kind not in PYTHON_SCALAR_DTYPES:
        raise TypeError(
            "weak_scalar_dtype expects a Python scalar (bool, int, float, "
            f"complex), got {type(value).__name__}"
        )
    array_dtype = core.dtype(array_dtype)
    name = array_dtype.name
    if scalar_kind is bool:
        if name in _WEAK_FLOAT_DTYPES:
            return array_dtype
        return np.result_type(np.dtype("bool"), array_dtype)
    if scalar_kind is int:
        if name in _WEAK_FLOAT_DTYPES:
            if abs(value) < 2 ** _WEAK_INT_BITS[name]:
                return array_dtype
            # Unrepresentable: complex targets fall back to complex128,
            # float targets to float64 (NEP 50 default-scalar promotion).
            if array_dtype.kind == "c":
                return np.dtype("complex128")
            return np.dtype("float64")
        # Integer/uint/bool tensors promote via plain result_type (the
        # scalar's natural dtype is int64).
        return np.result_type(np.dtype("int64"), array_dtype)
    if scalar_kind is float:
        if array_dtype == core.complex64:
            return array_dtype
        return np.result_type(np.dtype("float64"), array_dtype)
    # complex: weak toward nothing — plain result_type with complex128.
    return np.result_type(np.dtype("complex128"), array_dtype)


def promote_dtypes(*dtypes) -> np.dtype:
    """Numpy dtype promotion for mixed symbolic dtypes.

    For tensor⊕tensor promotion this is EXACTLY ``np.result_type(*dtypes)`` —
    including numpy's subtleties (``int8 + uint8 → int16``,
    ``float32 + int64 → float64``, ``float16 + float16 → float16``). Python
    scalars must NOT reach this function; they are pre-promoted by
    :func:`as_operand` via :func:`weak_scalar_dtype`.

    Args:
        *dtypes: One or more numpy dtype objects.

    Returns:
        The promoted numpy dtype.

    Raises:
        TypeError: zero dtypes given, or an argument is not a dtype.
    """
    if not dtypes:
        raise TypeError("promote_dtypes requires at least one dtype")
    normalized = []
    for item in dtypes:
        try:
            normalized.append(np.dtype(item))
        except TypeError as exc:
            raise TypeError(
                f"promote_dtypes: {item!r} is not a dtype"
            ) from exc
    return np.result_type(*normalized)


def broadcast_shapes(*shapes) -> ShapeTuple:
    """Symbolic elementwise broadcasting of shape tuples.

    Follows numpy's exact broadcasting algorithm, extended to symbolic dims:

    1. Align shapes on the right; missing leading dims are treated as ``1``.
    2. Per aligned dim pair, in order:
       - either is ``1`` → result is the other dim;
       - dims are equal → that dim;
       - both dims are concrete ints and unequal → ``core.ShapeError`` NOW
         (statically known incompatibility, never deferred);
       - otherwise (at least one dim is a ``Dim``/``DimExpr``) → result is
         ``DimExpr.max(a, b)``. This is the symbolic statement of numpy's
         runtime rule (the runtime result shape IS ``max(a, b)``); the numpy
         backend enforces the exact compatibility check at run time and
         surfaces a ``ShapeError`` there if the dims disagree.

    Rank of the result is ``max(len(shape) for shape in shapes)``.

    Args:
        *shapes: Shape tuples of ``int``/``Dim``/``DimExpr`` elements.

    Returns:
        The broadcast shape tuple.

    Raises:
        core.ShapeError: static (int vs int, neither 1) incompatibility.
    """
    if not shapes:
        return ()
    rank = max(len(s) for s in shapes)
    out = []
    for i in range(rank):
        dims = tuple(
            1 if i < rank - len(s) else s[i - (rank - len(s))]
            for s in shapes
        )
        out.append(_broadcast_dim(dims))
    return tuple(out)


def _broadcast_dim(dims: Tuple[ShapeElement, ...]) -> ShapeElement:
    """Broadcast one aligned group of dims.

    Kept consistent with ``etl.ir.inference._broadcast_shapes`` (the IR-level
    implementation of the same numpy rule — the numpy backend enforces the
    exact runtime check): ``1`` yields the other side; equal dims pass
    through; two unequal concrete ints raise ``ShapeError`` NOW; ``None``
    (runtime-dynamic) is unchecked and yields ``None``; otherwise the result
    is ``DimExpr("max", a, b)`` — the symbolic statement of numpy's runtime
    rule.
    """
    result: ShapeElement = 1
    for d in dims:
        if d == 1:
            continue
        if result == 1:
            result = d
            continue
        if d is None:
            result = None  # type: ignore[assignment]  # runtime-dynamic
            continue
        if result is None:
            continue
        if d == result:
            continue
        if isinstance(d, int) and isinstance(result, int):
            raise core.ShapeError(
                f"cannot broadcast incompatible dims {result} and {d}"
            )
        # Symbolic mismatch: DimExpr.max defers the equality check to runtime.
        result = core.DimExpr("max", result, d)
    return result


def reduced_shape(shape, axes, keepdims: bool) -> ShapeTuple:
    """Output shape of a reduction.

    Args:
        shape: Input shape tuple.
        axes: Normalized (non-negative, deduplicated, sorted) tuple of axis
            ints; the empty tuple means "no reduction" (identity shape).
        keepdims: If True, reduced axes keep extent ``1`` instead of being
            removed.

    Returns:
        The output shape tuple. Reducing ALL axes without ``keepdims`` yields
        ``()`` (a scalar).

    Raises:
        core.ShapeError: any axis out of range (should not happen if the
            caller used :func:`normalize_axes`).
    """
    rank = len(shape)
    reduced = set()
    for axis in axes:
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or not 0 <= axis < rank
        ):
            raise core.ShapeError(
                f"reduction axis {axis!r} out of range for rank {rank}"
            )
        reduced.add(axis)
    if keepdims:
        return tuple(1 if i in reduced else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i not in reduced)


def normalize_axes(axes, rank: int) -> Tuple[int, ...]:
    """Normalize a user-supplied axes specification.

    Accepts ``None`` (→ ALL axes, i.e. ``tuple(range(rank))``), a single int,
    or a tuple of ints. Negative values are shifted by ``rank``. The result is
    sorted ascending with duplicates removed (reducing an axis twice is a
    no-op, matching numpy).

    Args:
        axes: ``None``, int, or tuple of ints.
        rank: Input tensor rank.

    Returns:
        A sorted, deduplicated tuple of non-negative axis ints.

    Raises:
        core.ShapeError: axis out of ``range(-rank, rank)``.
        TypeError: axes is not an int/tuple of ints.
    """
    if axes is None:
        return tuple(range(rank))
    if isinstance(axes, int) and not isinstance(axes, bool):
        seq: Tuple[int, ...] = (axes,)
    elif isinstance(axes, tuple):
        seq = axes
        for axis in seq:
            if not isinstance(axis, int) or isinstance(axis, bool):
                raise TypeError(
                    "axes must be an int or a tuple of ints, got element "
                    f"{axis!r}"
                )
    else:
        raise TypeError(
            "axes must be None, an int, or a tuple of ints, got "
            f"{type(axes).__name__}"
        )
    normalized = []
    for axis in seq:
        original = axis
        if axis < 0:
            axis += rank
        if not 0 <= axis < rank:
            raise core.ShapeError(
                f"axis {original} out of range for rank {rank}"
            )
        normalized.append(axis)
    return tuple(sorted(set(normalized)))
