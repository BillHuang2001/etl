"""Shape-inference hooks referenced by ``OpDef``s.

Each hook computes the result types of an op from its operand types and
attributes:

    fn(input_types: tuple[ValueType, ...], attributes: dict) -> tuple[ValueType, ...]

All bodies are implemented (Phase 2). Docstrings state the exact expected
semantics.

``shape_fn=None`` on an ``OpDef`` means result types are *op-specific* and are
resolved by the ``Builder`` from the op's attributes, regions, or the
enclosing module (``constant``: from payload; ``call``: from callee signature;
``if``: from region terminators; ``runtime_call``/``block_call``: from declared
specs), or must be passed explicitly via
``Builder.create(..., result_types=...)``. ``verify`` enforces agreement in
both cases.

Conventions shared by all hooks (keep in sync with ``etl.ops._utils`` and the
numpy backend, which evaluate the SAME rules with concrete dim bindings):

- **Broadcasting** (``_broadcast_shapes``): numpy's algorithm extended to
  symbolic dims — ``1`` yields the other side; structurally equal dims pass
  through; two unequal concrete ints raise ``ShapeError`` NOW; ``None``
  (runtime-dynamic) dims are unchecked and yield ``None``; otherwise the
  result is ``DimExpr("max", a, b)`` (left dim first) — the symbolic
  statement of numpy's runtime rule; the backend enforces exact equality.
- **Element counts** (``_count_factors``): mul-chains are flattened and int
  factors folded, so ``(B * 2, 3)`` counts as ``(6, {B: 1})``. Comparisons
  raise ``ShapeError`` only on *definite* mismatch (fully static counts that
  differ, or equal symbolic factor sets with differing static parts);
  ``None`` dims and structurally different symbolic counts defer to runtime.
- **Sum folds** (concatenate axis) are left-associative ``DimExpr("add", ...)``
  chains, static ints folded separately.
- **Dtype promotion** uses ``np.result_type`` exactly (etl dtypes ARE numpy
  dtypes); reduction dtypes follow numpy per op (``np.sum``/``np.prod``/
  ``np.mean`` promotion; ``max``/``min`` preserve).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from etl.core import Dim, DimExpr, ShapeError, dtype

from .types import ValueType


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _one(
    input_types: tuple[ValueType, ...], name: str
) -> ValueType:
    """Single-operand guard: return the only operand type."""
    if len(input_types) != 1:
        raise ShapeError(
            f"{name}: expected exactly one operand, got {len(input_types)}"
        )
    return input_types[0]


def _normalize_axis(axis: Any, rank: int, name: str = "axis") -> int:
    """Normalize a (possibly negative) axis into ``range(rank)``.

    Raises:
        ShapeError: ``axis`` is not an int or is out of range.
    """
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise ShapeError(f"{name}: expected an int axis, got {axis!r}")
    if axis < 0:
        axis += rank
    if not 0 <= axis < rank:
        raise ShapeError(f"{name}: axis {axis} out of range for rank {rank}")
    return axis


def _shape_attr(value: Any, name: str, *, allow_wildcard: bool) -> tuple:
    """Validate/normalize a target-shape attribute into a tuple of dims.

    Accepts ints (``-1`` only when ``allow_wildcard``), ``Dim``/``DimExpr``
    and ``None`` entries.

    Raises:
        ShapeError: malformed shape or disallowed negative dim.
    """
    if not isinstance(value, (tuple, list)):
        raise ShapeError(
            f"{name}: expected a tuple of shape dims, got {type(value).__name__}"
        )
    dims = []
    for d in value:
        if d is None:
            dims.append(None)
        elif isinstance(d, int) and not isinstance(d, bool):
            if d == -1 and not allow_wildcard:
                raise ShapeError(f"{name}: wildcard -1 is not allowed here")
            if d < -1:
                raise ShapeError(f"{name}: invalid negative dim {d}")
            dims.append(d)
        elif isinstance(d, (Dim, DimExpr)):
            dims.append(d)
        else:
            raise ShapeError(f"{name}: invalid shape dim {d!r}")
    return tuple(dims)


def _per_dim_tuple(
    value: Any, rank: int, default: Any, name: str
) -> tuple:
    """Normalize an int-or-per-dim-tuple attribute into a length-``rank`` tuple.

    ``None`` means "all defaults"; a bare int replicates over all dims.

    Raises:
        ShapeError: malformed value or arity mismatch.
    """
    if value is None:
        return tuple(default for _ in range(rank))
    if isinstance(value, int) and not isinstance(value, bool):
        return tuple(value for _ in range(rank))
    if isinstance(value, (tuple, list)):
        seq = tuple(value)
        if len(seq) != rank:
            raise ShapeError(
                f"{name}: expected {rank} entries, got {len(seq)}"
            )
        return seq
    raise ShapeError(
        f"{name}: expected an int or a tuple of {rank} ints, got {value!r}"
    )


def _broadcast_shapes(*shapes: tuple) -> tuple:
    """Symbolic broadcast of shape tuples (numpy rules + symbolic dims).

    Aligns shapes on the right (missing leading dims act as ``1``). Per
    aligned pair: ``1`` yields the other side; equal dims pass through; two
    unequal concrete ints raise ``ShapeError``; ``None`` is unchecked and
    yields ``None``; otherwise the result is ``DimExpr("max", a, b)`` (see
    module docstring). Result rank is the maximum input rank.
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


def _broadcast_dim(dims: tuple) -> Any:
    """Broadcast one aligned group of dims (see ``_broadcast_shapes``)."""
    result = 1
    for d in dims:
        if d == 1:
            continue
        if result == 1:
            result = d
            continue
        if d is None:
            result = None  # runtime-dynamic: unchecked
            continue
        if result is None:
            continue
        if d == result:
            continue
        if isinstance(d, int) and isinstance(result, int):
            raise ShapeError(
                f"cannot broadcast incompatible dims {result} and {d}"
            )
        # Symbolic mismatch: DimExpr.max defers the equality check to runtime.
        result = DimExpr("max", result, d)
    return result


def _count_factors(shape: tuple) -> tuple | None:
    """Fold a shape's element count into ``(static_product, {factor: n})``.

    Mul-chains of ``DimExpr`` are flattened and their int factors folded, so
    e.g. ``(B * 2, 3)`` counts as ``(6, {B: 1})``. Returns ``None`` when any
    dim is ``None`` (runtime-dynamic: count unknown).
    """
    total = 1
    factors: dict = {}
    for d in shape:
        if d is None:
            return None
        if isinstance(d, int):
            total *= d
        else:
            total = _fold_symbol(total, factors, d)
    return total, factors


def _fold_symbol(total: int, factors: dict, expr: Any) -> int:
    """Fold one symbolic factor: flatten mul chains, fold ints."""
    if isinstance(expr, int):
        return total * expr
    if isinstance(expr, DimExpr) and expr.op == "mul":
        total = _fold_symbol(total, factors, expr.left)
        return _fold_symbol(total, factors, expr.right)
    factors[expr] = factors.get(expr, 0) + 1
    return total


def _check_counts_equal(count_a: tuple | None, count_b: tuple | None, what: str) -> None:
    """Element-count agreement: raise ``ShapeError`` on definite mismatch.

    Definite mismatch: both counts fully static and unequal, or equal
    symbolic factor multisets with differing static parts (the symbols
    cancel). Dynamic counts (``None``) and structurally different symbolic
    counts are undecidable statically and defer to the runtime backend.
    """
    if count_a is None or count_b is None:
        return
    int_a, facs_a = count_a
    int_b, facs_b = count_b
    if not facs_a and not facs_b:
        if int_a != int_b:
            raise ShapeError(
                f"{what}: element counts differ: {int_a} vs {int_b}"
            )
        return
    if facs_a == facs_b and int_a != int_b:
        raise ShapeError(
            f"{what}: element counts differ symbolically: "
            f"{int_a}*{facs_a} vs {int_b}*{facs_b}"
        )


def _divide_counts(total: tuple | None, known: tuple | None) -> Any:
    """Symbolic element-count division ``total / known`` (reshape ``-1`` dim).

    Returns an int when fully static, a ``DimExpr`` quotient when the
    symbolic factors of ``known`` divide those of ``total`` (runtime checks
    divisibility), or ``None`` when undecidable statically. Raises
    ``ShapeError`` on definite mismatch (zero known product, or inexact
    integer division with no symbolic remainder).
    """
    if total is None or known is None:
        return None
    int_t, facs_t = total
    int_k, facs_k = known
    if int_k == 0:
        raise ShapeError(
            "reshape: cannot infer the -1 dim against a zero-size known product"
        )
    rem = dict(facs_t)
    for f, n in facs_k.items():
        rem[f] = rem.get(f, 0) - n
        if rem[f] < 0:
            return None  # symbolic factors do not divide: runtime-dynamic
    rem = {f: n for f, n in rem.items() if n > 0}
    if not rem:
        if int_t % int_k != 0:
            raise ShapeError(
                f"reshape: element count {int_t} is not divisible by the "
                f"known target count {int_k}"
            )
        return int_t // int_k
    numerator: Any = int_t
    for f, n in rem.items():
        for _ in range(n):
            numerator = f if numerator == 1 else DimExpr("mul", numerator, f)
    if int_k == 1:
        return numerator
    return DimExpr("floordiv", numerator, int_k)


def _check_contract_dim(a: Any, b: Any, what: str) -> None:
    """Contracted dims must be equal: definite int/int mismatch raises
    ``ShapeError``; symbolic or dynamic (``None``) dims defer to runtime."""
    if a == b or a is None or b is None:
        return
    if isinstance(a, int) and isinstance(b, int):
        raise ShapeError(
            f"{what}: contracting dims {a} and {b} do not match"
        )


def _agree_dim(dims: tuple, what: str) -> Any:
    """Non-axis dims of a concatenate must agree; ``None`` dims are unchecked
    (yield ``None``) and symbolic differences defer to runtime."""
    if any(d is None for d in dims):
        return None
    first = dims[0]
    for d in dims[1:]:
        if d == first:
            continue
        if isinstance(d, int) and isinstance(first, int):
            raise ShapeError(f"{what}: dims must agree, got {first} vs {d}")
    return first


def _sum_dims(dims: tuple) -> Any:
    """Left-associative symbolic sum of shape dims (concatenate axis).

    Static ints fold together; ``None`` makes the sum dynamic (``None``).
    """
    if any(d is None for d in dims):
        return None
    int_sum = 0
    sym: Any = None
    for d in dims:
        if isinstance(d, int):
            int_sum += d
        elif sym is None:
            sym = d
        else:
            sym = DimExpr("add", sym, d)
    if sym is None:
        return int_sum
    return sym + int_sum if int_sum else sym


def _add_dim(d: Any, amount: int) -> Any:
    """``d + amount`` (symbolic addition); ``None`` stays dynamic."""
    if d is None:
        return None
    if amount == 0:
        return d
    return d + amount


def _mul_dim(d: Any, factor: int) -> Any:
    """``d * factor`` (symbolic product); ``None`` stays dynamic."""
    return None if d is None else d * factor


def _div_dim(d: Any, group_size: int, what: str) -> Any:
    """``d // group_size``: int dims must divide exactly (``ShapeError``),
    symbolic dims defer (runtime checks divisibility), ``None`` stays."""
    if d is None:
        return None
    if isinstance(d, int):
        if d % group_size != 0:
            raise ShapeError(
                f"{what}: dim {d} is not divisible by group_size {group_size}"
            )
        return d // group_size
    return d // group_size


def _reduce_dtype(dtype_: np.dtype, op: str) -> np.dtype:
    """Result dtype of a reduce_* op, exactly per numpy.

    ``sum``/``prod`` promote like ``np.sum``/``np.prod`` (bool → int64,
    signed ints → int64, uints → uint64, floats/complex keep); ``mean``
    promotes like ``np.mean`` (integer/bool → float64); ``max``/``min`` keep
    the dtype.
    """
    if op in ("max", "min"):
        return dtype_
    if op not in ("sum", "mean", "prod"):
        raise ValueError(
            f"unknown reduce_op {op!r} "
            "(expected 'sum' | 'max' | 'min' | 'mean' | 'prod')"
        )
    probe = np.zeros((1,), dtype=dtype_)
    if op == "mean":
        return np.mean(probe).dtype
    if op == "prod":
        return np.prod(probe).dtype
    return np.sum(probe).dtype


def _conv_padding(padding: Any, n_spatial: int) -> tuple:
    """Normalize the conv ``padding`` attribute to per-spatial-dim entries
    (``"VALID"`` / ``"SAME"`` / ``(lo, hi)`` int pairs).

    Raises:
        ShapeError: unknown mode, arity mismatch, or negative padding.
    """
    if isinstance(padding, str):
        if padding not in ("VALID", "SAME"):
            raise ShapeError(f"conv: unknown padding mode {padding!r}")
        return (padding,) * n_spatial
    if isinstance(padding, int) and not isinstance(padding, bool):
        if padding < 0:
            raise ShapeError("conv: negative padding")
        return ((padding, padding),) * n_spatial
    if isinstance(padding, (tuple, list)):
        pads = tuple(padding)
        if len(pads) != n_spatial:
            raise ShapeError(
                f"conv: expected {n_spatial} padding entries, got {len(pads)}"
            )
        out = []
        for p in pads:
            if isinstance(p, int) and not isinstance(p, bool):
                if p < 0:
                    raise ShapeError("conv: negative padding")
                out.append((p, p))
            elif (
                isinstance(p, (tuple, list))
                and len(p) == 2
                and all(isinstance(v, int) for v in p)
            ):
                if p[0] < 0 or p[1] < 0:
                    raise ShapeError("conv: negative padding")
                out.append((p[0], p[1]))
            else:
                raise ShapeError(f"conv: invalid padding entry {p!r}")
        return tuple(out)
    raise ShapeError(f"conv: invalid padding {padding!r}")


def _conv_out_dim(d: Any, k: Any, stride: int, in_dil: int, k_dil: int, pad: Any) -> Any:
    """One conv output spatial dim (DimExpr arithmetic where symbolic).

    Effective sizes follow numpy convolution: ``eff = (dim - 1) * dil + 1``.
    ``"VALID"``: ``(eff_d - eff_k) // stride + 1``; ``"SAME"`` (TF
    convention): ``ceil(d / stride)``; ``(lo, hi)``: padding added exactly.
    """
    if d is None:
        return None
    if pad == "SAME":
        out = (d + stride - 1) // stride
    else:
        if k is None:
            return None
        eff_d = (d - 1) * in_dil + 1
        eff_k = (k - 1) * k_dil + 1
        lo, hi = (0, 0) if pad == "VALID" else pad
        out = (eff_d + (lo + hi) - eff_k) // stride + 1
    if isinstance(out, int) and out < 0:
        raise ShapeError(
            f"conv: negative output dim {out} (kernel larger than input?)"
        )
    return out


def _check_conv_channels(c_in: Any, k_c_in: Any, feature_groups: int) -> None:
    """Static grouped-conv channel checks (definite failures only)."""
    if not isinstance(feature_groups, int) or feature_groups <= 0:
        raise ShapeError(
            f"conv: feature_group_count must be a positive int, got {feature_groups!r}"
        )
    if isinstance(c_in, int):
        if c_in % feature_groups != 0:
            raise ShapeError(
                f"conv: in_channels {c_in} not divisible by feature_group_count {feature_groups}"
            )
        if isinstance(k_c_in, int) and k_c_in != c_in // feature_groups:
            raise ShapeError(
                f"conv: kernel in-channels {k_c_in} != in_channels {c_in} // "
                f"feature_group_count {feature_groups}"
            )


# ---------------------------------------------------------------------------
# Public hooks (referenced by name from op_defs/)
# ---------------------------------------------------------------------------

def infer_elementwise_binary(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a broadcasting binary elementwise op.

    Shape: DimExpr-based broadcast of the two operand shapes — equal dims pass
    through, ``1`` broadcasts to the other side, symbolic dims unify by name
    (structurally equal ``Dim``/``DimExpr`` values; unequal symbolic pairs
    defer as ``DimExpr.max``); incompatible concrete dims raise
    ``ShapeError``; ``None`` dims are unchecked and yield ``None``. Dtype:
    numpy promotion (``np.result_type``).
    """
    if len(input_types) != 2:
        raise ShapeError(
            f"elementwise binary: expected 2 operands, got {len(input_types)}"
        )
    a, b = input_types
    shape = _broadcast_shapes(a.shape, b.shape)
    return (ValueType(np.result_type(a.dtype, b.dtype), shape),)


def infer_elementwise_unary(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a unary elementwise op: same shape and dtype as operand."""
    t = _one(input_types, "elementwise unary")
    return (ValueType(t.dtype, t.shape),)


def infer_abs(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``abs``: numpy semantics — the magnitude of a complex
    value is REAL (complex64 → float32, complex128 → float64); every other
    dtype is preserved. Shape preserved."""
    t = _one(input_types, "abs")
    if t.dtype == np.dtype("complex64"):
        out_dtype = np.dtype("float32")
    elif t.dtype == np.dtype("complex128"):
        out_dtype = np.dtype("float64")
    else:
        out_dtype = t.dtype
    return (ValueType(out_dtype, t.shape),)


def infer_cast(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``cast``: operand shape, dtype from ``attributes["dtype"]``."""
    t = _one(input_types, "cast")
    return (ValueType(dtype(attributes["dtype"]), t.shape),)


def infer_compare(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a comparison: broadcast shape, dtype ``bool``."""
    if len(input_types) != 2:
        raise ShapeError(
            f"comparison: expected 2 operands, got {len(input_types)}"
        )
    a, b = input_types
    return (ValueType(np.dtype("bool"), _broadcast_shapes(a.shape, b.shape)),)


def infer_select(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``select``: broadcast of all three operands; dtype is
    the promoted dtype of the two branch operands (pred's dtype is bool)."""
    if len(input_types) != 3:
        raise ShapeError(
            f"select: expected 3 operands, got {len(input_types)}"
        )
    pred, on_true, on_false = input_types
    shape = _broadcast_shapes(pred.shape, on_true.shape, on_false.shape)
    return (ValueType(np.result_type(on_true.dtype, on_false.dtype), shape),)


def infer_broadcast_to(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``broadcast``: shape from ``attributes["shape"]``,
    operand dtype. The operand must be broadcast-compatible with the target."""
    t = _one(input_types, "broadcast")
    target = _shape_attr(
        attributes["shape"], "broadcast.shape", allow_wildcard=False
    )
    if len(target) < t.rank:
        raise ShapeError(
            f"broadcast: target rank {len(target)} < operand rank {t.rank}"
        )
    offset = len(target) - t.rank
    for i, td in enumerate(target):
        sd = 1 if i < offset else t.shape[i - offset]
        if sd == 1 or sd is None or td is None or sd == td:
            continue
        if isinstance(sd, int) and isinstance(td, int):
            raise ShapeError(
                f"broadcast: operand dim {sd} cannot expand to target dim {td}"
            )
        # symbolic mismatch: runtime-checked (result is the target anyway)
    return (ValueType(t.dtype, target),)


def infer_reshape(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``reshape``: shape from ``attributes["shape"]``
    (with exactly one ``-1`` wildcard allowed if the target has dynamic dims),
    operand dtype. Total element counts must agree (symbolically)."""
    t = _one(input_types, "reshape")
    target = _shape_attr(
        attributes["shape"], "reshape.shape", allow_wildcard=True
    )
    wildcards = [
        i for i, d in enumerate(target) if isinstance(d, int) and d == -1
    ]
    if len(wildcards) > 1:
        raise ShapeError(
            f"reshape: at most one -1 wildcard allowed, got {len(wildcards)}"
        )
    count_in = _count_factors(t.shape)
    if not wildcards:
        _check_counts_equal(count_in, _count_factors(target), "reshape")
        return (ValueType(t.dtype, target),)
    i = wildcards[0]
    rest = _count_factors(tuple(d for j, d in enumerate(target) if j != i))
    out = list(target)
    out[i] = _divide_counts(count_in, rest)
    return (ValueType(t.dtype, tuple(out)),)


def infer_transpose(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``transpose``: operand shape permuted by
    ``attributes["permutation"]`` (None = full reversal, numpy convention)."""
    t = _one(input_types, "transpose")
    perm = attributes.get("permutation")
    if perm is None:
        return (ValueType(t.dtype, t.shape[::-1]),)
    perm = tuple(perm)
    if len(perm) != t.rank or not all(isinstance(p, int) for p in perm):
        raise ShapeError(f"transpose: invalid permutation {perm!r}")
    if sorted(perm) != list(range(t.rank)):
        raise ShapeError(
            f"transpose: {perm} is not a permutation of {t.rank} axes"
        )
    return (ValueType(t.dtype, tuple(t.shape[p] for p in perm)),)


def infer_slice(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``slice``: dims computed from ``start_indices``,
    ``limit_indices``, ``strides`` per numpy slice semantics; dims sliced
    completely (stride 1 over the full range) stay symbolic."""
    t = _one(input_types, "slice")
    rank = t.rank
    starts = _per_dim_tuple(
        attributes.get("start_indices"), rank, 0, "slice.start_indices"
    )
    limits_raw = attributes.get("limit_indices")
    limits = (
        t.shape
        if limits_raw is None
        else _per_dim_tuple(limits_raw, rank, None, "slice.limit_indices")
    )
    strides = _per_dim_tuple(
        attributes.get("strides"), rank, 1, "slice.strides"
    )
    out = []
    for d, start, limit, stride in zip(t.shape, starts, limits, strides):
        if not isinstance(stride, int) or stride <= 0:
            raise ShapeError(
                f"slice: strides must be positive ints, got {stride!r}"
            )
        if limit is None:
            out.append(None)  # runtime-dynamic result dim
            continue
        if start == 0 and stride == 1 and limit == d:
            out.append(d)  # complete slice: preserve symbolic dims as-is
            continue
        if isinstance(start, int) and isinstance(limit, int) and isinstance(d, int):
            if start < 0:
                raise ShapeError(f"slice: negative start index {start}")
            if limit > d:
                raise ShapeError(f"slice: limit {limit} exceeds dim {d}")
            if start >= limit:
                out.append(0)  # numpy: empty slice
            else:
                out.append((limit - start + stride - 1) // stride)
        else:
            # symbolic limits: (limit - start + stride - 1) // stride via
            # DimExpr arithmetic (runtime checks bounds)
            out.append((limit - start + stride - 1) // stride)
    return (ValueType(t.dtype, tuple(out)),)


def infer_gather(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``gather``: indices shape spliced into the tensor shape
    at ``attributes["axes"]``; result dtype = tensor dtype.

    Single axis (the frontend ``etl.gather`` case) reproduces numpy ``take``:
    ``x.shape[:a] + indices.shape + x.shape[a + 1:]``. Multiple axes remove
    all of them and insert ``indices.shape`` as a block at the smallest axis.
    """
    if len(input_types) != 2:
        raise ShapeError(
            f"gather: expected 2 operands, got {len(input_types)}"
        )
    tensor, indices = input_types
    axes = attributes.get("axes", (0,))
    if isinstance(axes, int):
        axes = (axes,)
    removed = sorted(
        {_normalize_axis(a, tensor.rank, "gather.axes") for a in axes}
    )
    pos = removed[0]
    kept = tuple(
        d for i, d in enumerate(tensor.shape) if i not in set(removed)
    )
    return (ValueType(tensor.dtype, kept[:pos] + indices.shape + kept[pos:]),)


def infer_scatter(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``scatter``: shape and dtype of the first operand."""
    if len(input_types) != 3:
        raise ShapeError(
            f"scatter: expected 3 operands, got {len(input_types)}"
        )
    t = input_types[0]
    return (ValueType(t.dtype, t.shape),)


def infer_concatenate(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``concatenate``: dims combined along
    ``attributes["axis"]`` (symbolic addition via DimExpr); other dims must
    agree; dtype = promoted dtype of the operands."""
    if not input_types:
        raise ShapeError("concatenate: expected at least one operand")
    rank = input_types[0].rank
    for t in input_types:
        if t.rank != rank:
            raise ShapeError(
                f"concatenate: rank mismatch {rank} vs {t.rank}"
            )
    axis = _normalize_axis(attributes["axis"], rank, "concatenate.axis")
    out = []
    for j in range(rank):
        dims = tuple(t.shape[j] for t in input_types)
        out.append(
            _sum_dims(dims) if j == axis else _agree_dim(dims, "concatenate")
        )
    promoted = np.result_type(*[t.dtype for t in input_types])
    return (ValueType(promoted, tuple(out)),)


def infer_pad(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``pad``: operand dims plus the low/high paddings from
    ``attributes["padding_config"]`` (symbolic addition); operand dtype."""
    t = _one(input_types, "pad")
    config = attributes.get("padding_config")
    if not isinstance(config, (tuple, list)) or len(config) != t.rank:
        raise ShapeError(
            f"pad: padding_config must have {t.rank} entries, got {config!r}"
        )
    out = []
    for d, entry in zip(t.shape, config):
        if isinstance(entry, int) and not isinstance(entry, bool):
            lo = hi = entry
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and all(isinstance(v, int) for v in entry)
        ):
            lo, hi = entry
        else:
            raise ShapeError(
                f"pad: invalid padding entry {entry!r} "
                "(expected int or (lo, hi) pair)"
            )
        if lo < 0 or hi < 0:
            raise ShapeError(f"pad: negative padding ({lo}, {hi})")
        out.append(_add_dim(d, lo + hi))
    return (ValueType(t.dtype, tuple(out)),)


def infer_reduction(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a reduce_* op: operand shape with ``axes`` dims removed
    (or set to 1 with ``keepdims=True``); dtype per the reduction
    (``attributes["reduce_op"]``): sum/prod promote per numpy (bool → int64,
    signed ints → int64, uints → uint64, floats/complex keep), mean promotes
    integer/bool to float64, max/min keep the dtype."""
    t = _one(input_types, "reduction")
    op = attributes["reduce_op"]
    axes = attributes.get("axes", ())
    if isinstance(axes, int):
        axes = (axes,)
    keepdims = bool(attributes.get("keepdims", False))
    reduced = sorted(
        {_normalize_axis(a, t.rank, "reduction.axes") for a in axes}
    )
    if not reduced:
        reduced = list(range(t.rank))  # empty axes = reduce over ALL axes
    reduced_set = set(reduced)
    if keepdims:
        shape = tuple(
            1 if i in reduced_set else d for i, d in enumerate(t.shape)
        )
    else:
        shape = tuple(
            d for i, d in enumerate(t.shape) if i not in reduced_set
        )
    return (ValueType(_reduce_dtype(t.dtype, op), shape),)


def infer_arg_reduction(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``argmax``/``argmin``: operand shape with the axis
    removed (or 1 with ``keepdims``); dtype ``int64``."""
    t = _one(input_types, "argmax/argmin")
    axis = attributes.get("axis")
    keepdims = bool(attributes.get("keepdims", False))
    if axis is None:
        # numpy: flatten-reduce over all axes → scalar (keepdims → all-1 shape)
        shape = (1,) * t.rank if keepdims else ()
    else:
        a = _normalize_axis(axis, t.rank, "argmax/argmin.axis")
        if keepdims:
            shape = tuple(1 if i == a else d for i, d in enumerate(t.shape))
        else:
            shape = tuple(d for i, d in enumerate(t.shape) if i != a)
    return (ValueType(np.dtype("int64"), shape),)


def infer_identity(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of shape-preserving ops (``tril``, ``triu``, ``cumsum``,
    ``stop_gradient``, identity collectives, ``while``): each result mirrors
    the corresponding operand type exactly."""
    return tuple(input_types)


def infer_dot(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``dot``: numpy matmul contract over the last two dims
    (vector/vector -> scalar, matrix/vector -> vector, batched matmul);
    dtype = numpy promotion of the operands."""
    if len(input_types) != 2:
        raise ShapeError(f"dot: expected 2 operands, got {len(input_types)}")
    a, b = input_types
    sa, sb = a.shape, b.shape
    if not sa or not sb:
        raise ShapeError("dot: operands must have rank >= 1")
    a_vec = len(sa) == 1
    b_vec = len(sb) == 1
    ea = [1, sa[0]] if a_vec else list(sa)
    eb = [sb[0], 1] if b_vec else list(sb)
    batch = _broadcast_shapes(tuple(ea[:-2]), tuple(eb[:-2]))
    m, ka = ea[-2], ea[-1]
    kb, n = eb[-2], eb[-1]
    _check_contract_dim(ka, kb, "dot")
    out = list(batch) + [m, n]
    if a_vec:
        out.pop(-2)
    if b_vec:
        out.pop(-1)
    return (ValueType(np.result_type(a.dtype, b.dtype), tuple(out)),)


def infer_conv(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``conv``: spatial dims per stride/dilation/padding from
    attributes (``"VALID"`` drops, ``"SAME"`` preserves via ``ceil(in /
    stride)``, tuple padding adds exactly); output channel dim from the kernel
    (NCHW layout per the frontend contract: kernel shape
    ``(C_out, C_in / feature_group_count, *kernel_spatial)``, so ``C_out`` is
    the kernel's FIRST dim); batch from the input; dtype = promotion of the
    operands."""
    if len(input_types) != 2:
        raise ShapeError(f"conv: expected 2 operands, got {len(input_types)}")
    x, w = input_types
    sx, sw = x.shape, w.shape
    if len(sx) != len(sw):
        raise ShapeError(
            f"conv: input and kernel ranks differ: {len(sx)} vs {len(sw)}"
        )
    if len(sx) < 3:
        raise ShapeError(
            f"conv: rank must be >= 3 (N, C, *spatial), got {len(sx)}"
        )
    n_spatial = len(sx) - 2
    strides = _per_dim_tuple(
        attributes.get("strides"), n_spatial, 1, "conv.strides"
    )
    in_dil = _per_dim_tuple(
        attributes.get("input_dilation"), n_spatial, 1, "conv.input_dilation"
    )
    k_dil = _per_dim_tuple(
        attributes.get("kernel_dilation"), n_spatial, 1, "conv.kernel_dilation"
    )
    for name, vals in (
        ("strides", strides),
        ("input_dilation", in_dil),
        ("kernel_dilation", k_dil),
    ):
        for v in vals:
            if not isinstance(v, int) or v <= 0:
                raise ShapeError(
                    f"conv: {name} entries must be positive ints, got {v!r}"
                )
    pads = _conv_padding(attributes.get("padding", "VALID"), n_spatial)
    feature_groups = attributes.get("feature_group_count", 1)
    batch_groups = attributes.get("batch_group_count", 1)
    _check_conv_channels(sx[1], sw[1], feature_groups)
    if not isinstance(batch_groups, int) or batch_groups <= 0:
        raise ShapeError(
            f"conv: batch_group_count must be a positive int, got {batch_groups!r}"
        )
    if isinstance(sx[0], int) and sx[0] % batch_groups != 0:
        raise ShapeError(
            f"conv: batch dim {sx[0]} not divisible by "
            f"batch_group_count {batch_groups}"
        )
    out_ch = _mul_dim(sw[0], batch_groups)
    out_spatial = tuple(
        _conv_out_dim(sx[2 + i], sw[2 + i], strides[i], in_dil[i], k_dil[i], pads[i])
        for i in range(n_spatial)
    )
    return (ValueType(np.result_type(x.dtype, w.dtype), (sx[0], out_ch) + out_spatial),)


def infer_solve(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``solve``: shape of the right-hand side operand ``b``
    (batch dims of ``a`` broadcast per numpy ``linalg.solve``); dtype =
    promotion of ``a`` and ``b`` — integer/bool inputs promote to ``float64``
    exactly like numpy ``linalg.solve``."""
    if len(input_types) != 2:
        raise ShapeError(
            f"solve: expected 2 operands, got {len(input_types)}"
        )
    a, b = input_types
    sa, sb = a.shape, b.shape
    if len(sa) < 2:
        raise ShapeError("solve: 'a' must have rank >= 2")
    if not sb:
        raise ShapeError("solve: 'b' must have rank >= 1")
    n1, n2 = sa[-2], sa[-1]
    if isinstance(n1, int) and isinstance(n2, int) and n1 != n2:
        raise ShapeError(f"solve: 'a' must be square, got {n1} vs {n2}")
    # b is (..., n, k) — the contracting dim is b[-2] (rank >= 2) or b[-1]
    # (vector b); the result keeps b's trailing dims.
    if len(sb) == 1:
        b_contract = sb[-1]
        b_batch = ()
        b_tail = sb
    else:
        b_contract = sb[-2]
        b_batch = sb[:-2]
        b_tail = sb[-2:]
    _check_contract_dim(n2, b_contract, "solve")
    batch = _broadcast_shapes(sa[:-2], b_batch)
    da, db = a.dtype, b.dtype
    if da.kind in "biu" or db.kind in "biu":
        result_dtype = np.dtype("float64")  # numpy linalg.solve behavior
    else:
        result_dtype = np.result_type(da, db)
    return (ValueType(result_dtype, batch + b_tail),)


def infer_all_gather(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``all_gather``: input shape with the ``axis`` dim
    multiplied by ``attributes["group_size"]`` (symbolic product); input
    dtype."""
    t = _one(input_types, "all_gather")
    gs = attributes["group_size"]
    if gs is None:
        # World group: rank count unknown at trace time — result dim is
        # runtime-dynamic; the backend validates it at run time.
        axis = _normalize_axis(attributes["axis"], t.rank, "all_gather.axis")
        out = list(t.shape)
        out[axis] = None
        return (ValueType(t.dtype, tuple(out)),)
    if not isinstance(gs, int) or gs <= 0:
        raise ShapeError(
            f"all_gather: group_size must be a positive int or None, got {gs!r}"
        )
    axis = _normalize_axis(attributes["axis"], t.rank, "all_gather.axis")
    out = list(t.shape)
    out[axis] = _mul_dim(out[axis], gs)
    return (ValueType(t.dtype, tuple(out)),)


def infer_reduce_scatter(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``reduce_scatter``: input shape with the ``axis`` dim
    divided by ``attributes["group_size"]`` (must divide exactly or be
    symbolic); input dtype."""
    t = _one(input_types, "reduce_scatter")
    gs = attributes["group_size"]
    if gs is None:
        # World group: rank count unknown at trace time — result dim is
        # runtime-dynamic; the backend validates it at run time.
        axis = _normalize_axis(
            attributes["axis"], t.rank, "reduce_scatter.axis"
        )
        out = list(t.shape)
        out[axis] = None
        return (ValueType(t.dtype, tuple(out)),)
    if not isinstance(gs, int) or gs <= 0:
        raise ShapeError(
            f"reduce_scatter: group_size must be a positive int or None, got {gs!r}"
        )
    axis = _normalize_axis(attributes["axis"], t.rank, "reduce_scatter.axis")
    out = list(t.shape)
    out[axis] = _div_dim(out[axis], gs, "reduce_scatter")
    return (ValueType(t.dtype, tuple(out)),)


def infer_all_to_all(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``all_to_all``: dims permuted between ``split_axis``
    and ``concat_axis`` per group size; equal axes preserve the shape;
    input dtype."""
    t = _one(input_types, "all_to_all")
    gs = attributes["group_size"]
    if gs is not None and (not isinstance(gs, int) or gs <= 0):
        raise ShapeError(
            f"all_to_all: group_size must be a positive int or None, got {gs!r}"
        )
    split = _normalize_axis(
        attributes["split_axis"], t.rank, "all_to_all.split_axis"
    )
    concat = _normalize_axis(
        attributes["concat_axis"], t.rank, "all_to_all.concat_axis"
    )
    out = list(t.shape)
    if split != concat:
        # World group (gs None): affected dims become runtime-dynamic.
        out[split] = _div_dim(out[split], gs, "all_to_all") if gs is not None else None
        out[concat] = _mul_dim(out[concat], gs) if gs is not None else None
    return (ValueType(t.dtype, tuple(out)),)


def infer_scalar_int64(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``rank``/``world_size``: scalar (``()`` shape) int64."""
    return (ValueType(np.dtype("int64"), ()),)


# ---------------------------------------------------------------------------
# Sparse hooks (referenced by the sparse.py OpDefs)
#
# A sparse value is an (indices, values) pair: ``indices`` is an int64 tensor
# of shape ``(B..., nnz, ndim)`` (batch dims, runtime-dynamic ``nnz``, one
# coordinate row per non-zero), ``values`` holds the non-zero values with
# shape ``(B..., nnz)``. ``nnz`` is always runtime-dynamic (``None``); the
# unbatched dense shape and the sparse value dtype come from the
# ``dense_shape`` / ``dtype`` attributes (``dense_shape`` entries may carry a
# symbolic first dim after vectorize — hooks only use ``len()`` on it and
# never assume ints). Batch dims always come from INPUT TYPES: for a sparse
# operand they are the leading dims of its indices type minus the trailing
# ``(nnz, ndim)`` pair; for dense operands they are the leading dims minus the
# dense rank. Result indices always end ``(None, ndim)``.
# ---------------------------------------------------------------------------


def _sparse_dense_shape(attributes: dict[str, Any], name: str) -> tuple:
    """The ``dense_shape`` attribute as a tuple (hooks only use ``len()``)."""
    dense_shape = attributes["dense_shape"]
    if not isinstance(dense_shape, (tuple, list)):
        raise ShapeError(
            f"{name}: dense_shape must be a tuple of dims, got {dense_shape!r}"
        )
    return tuple(dense_shape)


def _sparse_rank2_shape(attributes: dict[str, Any], name: str) -> tuple:
    """The ``dense_shape`` attribute validated to rank 2 — the csr/csc
    conversions and the dot ops are rank-2-only."""
    dense_shape = _sparse_dense_shape(attributes, name)
    if len(dense_shape) != 2:
        raise ShapeError(
            f"{name}: sparse operand must be rank-2, got dense_shape "
            f"{dense_shape!r}"
        )
    return dense_shape


def _sparse_batch(indices_t: ValueType, name: str) -> tuple:
    """Batch dims of a COO-form sparse indices operand: the indices type
    shape minus the trailing ``(nnz, ndim)`` pair."""
    if len(indices_t.shape) < 2:
        raise ShapeError(
            f"{name}: sparse indices must have rank >= 2 (trailing "
            f"(nnz, ndim)), got rank {len(indices_t.shape)}"
        )
    return indices_t.shape[:-2]


def _sparse_coo_to_csr_like(
    input_types: tuple[ValueType, ...],
    attributes: dict[str, Any],
    name: str,
) -> tuple[ValueType, ...]:
    """Shared result types of ``sparse_coo_to_csr`` / ``sparse_coo_to_csc``
    (rank-2 only): (indptr, indices, values). indptr = batch + (rows + 1,);
    indices = batch + (None,); values = batch + (None,) with the input values
    dtype."""
    if len(input_types) != 2:
        raise ShapeError(
            f"{name}: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, values_t = input_types
    dense_shape = _sparse_rank2_shape(attributes, name)
    batch = _sparse_batch(indices_t, name)
    return (
        ValueType(np.dtype("int64"), batch + (dense_shape[0] + 1,)),
        ValueType(np.dtype("int64"), batch + (None,)),
        ValueType(values_t.dtype, batch + (None,)),
    )


def _sparse_csr_to_coo_like(
    input_types: tuple[ValueType, ...],
    attributes: dict[str, Any],
    name: str,
) -> tuple[ValueType, ...]:
    """Shared result types of ``sparse_csr_to_coo`` / ``sparse_csc_to_coo``
    (rank-2 only): (indices, values). indices = batch + (None, 2); values =
    batch + (None,) with the input values dtype. Batch dims come from the
    indptr operand (``batch + (rows + 1,)``)."""
    if len(input_types) != 3:
        raise ShapeError(
            f"{name}: expected 3 operands (indptr, indices, values), got "
            f"{len(input_types)}"
        )
    indptr_t, _indices_t, values_t = input_types
    _sparse_rank2_shape(attributes, name)
    if len(indptr_t.shape) < 1:
        raise ShapeError(f"{name}: indptr must have rank >= 1")
    batch = indptr_t.shape[:-1]
    return (
        ValueType(np.dtype("int64"), batch + (None, 2)),
        ValueType(values_t.dtype, batch + (None,)),
    )


def _sparse_binary_merge(
    input_types: tuple[ValueType, ...],
    attributes: dict[str, Any],
    name: str,
) -> tuple[ValueType, ...]:
    """Shared result types of the sparse binary merge ops (``sparse_add``
    union merge, ``sparse_multiply`` intersection merge): (indices, values).
    indices = batch of the first sparse operand + (None, ndim); values =
    batch + (None,) with the first operand's values dtype."""
    if len(input_types) != 4:
        raise ShapeError(
            f"{name}: expected 4 operands (ia, va, ib, vb), got "
            f"{len(input_types)}"
        )
    ia_t, va_t, _ib_t, _vb_t = input_types
    ndim = len(_sparse_dense_shape(attributes, name))
    batch = _sparse_batch(ia_t, name)
    return (
        ValueType(np.dtype("int64"), batch + (None, ndim)),
        ValueType(va_t.dtype, batch + (None,)),
    )


def infer_sparse_from_dense(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_from_dense``: (indices i64, values). indices
    = batch dims of the dense operand + (None, ndim); values = batch +
    (None,) with the dense operand's dtype."""
    t = _one(input_types, "sparse_from_dense")
    rank = len(_sparse_dense_shape(attributes, "sparse_from_dense"))
    if rank > t.rank:
        raise ShapeError(
            f"sparse_from_dense: dense rank {rank} exceeds operand rank "
            f"{t.rank}"
        )
    batch = t.shape[: t.rank - rank]
    return (
        ValueType(np.dtype("int64"), batch + (None, rank)),
        ValueType(t.dtype, batch + (None,)),
    )


def infer_sparse_to_dense(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``sparse_to_dense``: a dense tensor with the batch dims
    of the indices operand + the ``dense_shape`` attribute; dtype from the
    ``dtype`` attribute."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_to_dense: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, _values_t = input_types
    batch = _sparse_batch(indices_t, "sparse_to_dense")
    dense_shape = _sparse_dense_shape(attributes, "sparse_to_dense")
    return (ValueType(dtype(attributes["dtype"]), batch + dense_shape),)


def infer_sparse_coo_to_csr(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_coo_to_csr`` (rank-2 only): (indptr, indices,
    values); the COO is lex-sorted, i.e. already row-major, so no reorder."""
    return _sparse_coo_to_csr_like(input_types, attributes, "sparse_coo_to_csr")


def infer_sparse_csr_to_coo(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_csr_to_coo`` (rank-2 only): (indices, values)
    in COO form."""
    return _sparse_csr_to_coo_like(input_types, attributes, "sparse_csr_to_coo")


def infer_sparse_coo_to_csc(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_coo_to_csc`` (rank-2 only): (indptr, indices,
    values). CSC indptr runs over COLUMNS, so indptr = batch + (cols + 1,);
    indices = batch + (None,); values = batch + (None,) with the input values
    dtype."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_coo_to_csc: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, values_t = input_types
    dense_shape = _sparse_rank2_shape(attributes, "sparse_coo_to_csc")
    batch = _sparse_batch(indices_t, "sparse_coo_to_csc")
    return (
        ValueType(np.dtype("int64"), batch + (dense_shape[1] + 1,)),
        ValueType(np.dtype("int64"), batch + (None,)),
        ValueType(values_t.dtype, batch + (None,)),
    )


def infer_sparse_csc_to_coo(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_csc_to_coo`` (rank-2 only): (indices, values)
    — same shapes as ``sparse_csr_to_coo``; the op reorders back to
    row-major at run time."""
    return _sparse_csr_to_coo_like(input_types, attributes, "sparse_csc_to_coo")


def infer_sparse_negate(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_negate``: identical to the input (indices,
    values) pair — negation preserves the sparsity structure."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_negate: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    return (input_types[0], input_types[1])


def infer_sparse_add(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_add`` (union merge): (indices, values)."""
    return _sparse_binary_merge(input_types, attributes, "sparse_add")


def infer_sparse_multiply(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_multiply`` (intersection merge): (indices,
    values) — same shapes as ``sparse_add``."""
    return _sparse_binary_merge(input_types, attributes, "sparse_multiply")


def infer_sparse_multiply_dense(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_multiply_dense``: (indices, values) with the
    input shapes; values dtype = promotion of the values and dense dtypes."""
    if len(input_types) != 3:
        raise ShapeError(
            f"sparse_multiply_dense: expected 3 operands (indices, values, "
            f"dense), got {len(input_types)}"
        )
    indices_t, values_t, dense_t = input_types
    return (
        ValueType(indices_t.dtype, indices_t.shape),
        ValueType(np.result_type(values_t.dtype, dense_t.dtype), values_t.shape),
    )


def infer_sparse_reduce_sum(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``sparse_reduce_sum``: a dense tensor with the batch
    dims of the indices operand + the ``dense_shape`` dims reduced over
    ``axes`` (dropped, or kept when ``keepdims``); dtype from the ``dtype``
    attribute. ``axes`` refer to the unbatched sparse axes."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_reduce_sum: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, _values_t = input_types
    batch = _sparse_batch(indices_t, "sparse_reduce_sum")
    dense_shape = _sparse_dense_shape(attributes, "sparse_reduce_sum")
    rank = len(dense_shape)
    axes = attributes["axes"]
    if isinstance(axes, int):
        axes = (axes,)
    keepdims = bool(attributes["keepdims"])
    reduced = sorted(
        {_normalize_axis(a, rank, "sparse_reduce_sum.axes") for a in axes}
    )
    reduced_set = set(reduced)
    dense_out = tuple(
        1 if (i in reduced_set and keepdims) else dim
        for i, dim in enumerate(dense_shape)
        if (i not in reduced_set) or keepdims
    )
    return (ValueType(dtype(attributes["dtype"]), batch + dense_out),)


def infer_sparse_transpose(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_transpose``: (indices, values). indices =
    batch + (None, rank) with rank = len(dense_shape); values = batch +
    (None,) with the input values dtype. ``perm`` must be a permutation of
    ``rank`` axes."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_transpose: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, values_t = input_types
    rank = len(_sparse_dense_shape(attributes, "sparse_transpose"))
    perm = attributes["perm"]
    if not isinstance(perm, (tuple, list)) or len(perm) != rank or not all(
        isinstance(p, int) and not isinstance(p, bool) for p in perm
    ):
        raise ShapeError(f"sparse_transpose: invalid perm {perm!r} for rank {rank}")
    if sorted(perm) != list(range(rank)):
        raise ShapeError(
            f"sparse_transpose: {perm} is not a permutation of {rank} axes"
        )
    batch = _sparse_batch(indices_t, "sparse_transpose")
    return (
        ValueType(np.dtype("int64"), batch + (None, rank)),
        ValueType(values_t.dtype, batch + (None,)),
    )


def infer_sparse_reshape(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_reshape``: (indices, values). indices = batch
    + (None, new_rank) with new_rank = len(dense_shape); values = batch +
    (None,) with the input values dtype. ``old_shape`` must have the same
    element count as ``dense_shape`` (checked when fully static)."""
    if len(input_types) != 2:
        raise ShapeError(
            f"sparse_reshape: expected 2 operands (indices, values), got "
            f"{len(input_types)}"
        )
    indices_t, values_t = input_types
    dense_shape = _sparse_dense_shape(attributes, "sparse_reshape")
    old_shape = _shape_attr(
        attributes["old_shape"], "sparse_reshape.old_shape", allow_wildcard=False
    )
    _check_counts_equal(
        _count_factors(old_shape), _count_factors(dense_shape), "sparse_reshape"
    )
    batch = _sparse_batch(indices_t, "sparse_reshape")
    return (
        ValueType(np.dtype("int64"), batch + (None, len(dense_shape))),
        ValueType(values_t.dtype, batch + (None,)),
    )


def infer_sparse_concatenate(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result types of ``sparse_concatenate`` (variadic (indices, values)
    pairs): (indices, values). indices = batch of the first indices operand +
    (None, rank); values = batch + (None,) with the promoted values dtype."""
    if len(input_types) < 4 or len(input_types) % 2 != 0:
        raise ShapeError(
            f"sparse_concatenate: expected an even number of operands >= 4 "
            f"(ia, va, ib, vb, ...), got {len(input_types)}"
        )
    ia_t = input_types[0]
    rank = len(_sparse_dense_shape(attributes, "sparse_concatenate"))
    batch = _sparse_batch(ia_t, "sparse_concatenate")
    values_dtype = np.result_type(*[t.dtype for t in input_types[1::2]])
    return (
        ValueType(np.dtype("int64"), batch + (None, rank)),
        ValueType(values_dtype, batch + (None,)),
    )


def infer_sparse_dot_dense(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``sparse_dot_dense`` (rank-2 sparse (M, K) x dense
    (..., K, N)): a dense tensor with the batch dims of the sparse operand's
    indices + (M, N); dtype = promotion of the values and dense dtypes."""
    if len(input_types) != 3:
        raise ShapeError(
            f"sparse_dot_dense: expected 3 operands (indices, values, dense), "
            f"got {len(input_types)}"
        )
    indices_t, values_t, dense_t = input_types
    m, _k = _sparse_rank2_shape(attributes, "sparse_dot_dense")
    if len(dense_t.shape) < 2:
        raise ShapeError(
            f"sparse_dot_dense: dense operand must have rank >= 2, got rank "
            f"{len(dense_t.shape)}"
        )
    batch = _sparse_batch(indices_t, "sparse_dot_dense")
    n = dense_t.shape[-1]
    return (
        ValueType(np.result_type(values_t.dtype, dense_t.dtype), batch + (m, n)),
    )


def infer_sparse_dense_dot_sparse(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``dense_dot_sparse`` (dense (..., M, K) x rank-2 sparse
    (K, N)): a dense tensor with the batch dims of the sparse operand's
    indices + (M, N); dtype = promotion of the values and dense dtypes."""
    if len(input_types) != 3:
        raise ShapeError(
            f"dense_dot_sparse: expected 3 operands (dense, indices, values), "
            f"got {len(input_types)}"
        )
    dense_t, indices_t, values_t = input_types
    _k, n = _sparse_rank2_shape(attributes, "dense_dot_sparse")
    if len(dense_t.shape) < 2:
        raise ShapeError(
            f"dense_dot_sparse: dense operand must have rank >= 2, got rank "
            f"{len(dense_t.shape)}"
        )
    batch = _sparse_batch(indices_t, "dense_dot_sparse")
    m = dense_t.shape[-2]
    return (
        ValueType(np.result_type(values_t.dtype, dense_t.dtype), batch + (m, n)),
    )
