"""Builtin batching/JVP/VJP rules for the standard op set.

The op-name → rule-category tables below are the design record of how each
builtin op vectorizes and differentiates; registration happens at import
time. Ops deliberately absent from every table — runtime_call, collectives,
control-flow ops — raise `core.TransformError` when transformed (no silent
fallback). `constant` has a batching rule only; the AD sweep handles it
structurally (fixed data ⇒ zero tangent).

Rules build replacement ops with ordinary `etl.ops.*` functions into the
machinery's pushed builder (raw `trace.current_builder()` only for multi-axis
`gather` and batched `scatter`). v1 gaps are explicit `TransformError`s: conv
VJP, mapped-kernel conv batching, mapped-operand slice batching, and
differentiating `mean`/segment-splitting over symbolic dims.
"""
from __future__ import annotations

import numpy as np

from etl import core, ir, ops
from etl.core import TransformError
from etl.trace import current_builder
from etl.transforms.autodiff import (
    ZeroTangent,
    register_jvp_rule,
    register_vjp_rule,
)
from etl.transforms.batching import register_batching_rule
from etl.transforms._metadata import MappedAxes, UNMAPPED

# --- op categories (design record) --------------------------------------
# Elementwise: result axes = union of operand mapped axes (unmapped operands get leading size-one dims inserted, i.e. broadcast).
ELEMENTWISE_OPS = (
    "add", "subtract", "multiply", "divide", "power", "remainder", "maximum",
    "minimum", "abs", "negate", "square", "sqrt", "exp", "log", "log1p",
    "sin", "cos", "tan", "tanh", "sigmoid", "relu", "gelu", "erf", "sign",
    "cast", "select", "broadcast", "stop_gradient",
)
# Batch like elementwise; AD: bool/int outputs cannot backpropagate — JVP/VJP rules yield ZeroTangent (not an error).
NONDIFFERENTIABLE_OUTPUT_OPS = (
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "logical_and", "logical_or", "logical_not",
    "bitwise_and", "bitwise_or", "bitwise_xor",
    "argmax", "argmin",
)
# Reductions: shift the reduced axes by the leading-mapped count; argmax/argmin batching lives here too (axis-based, not pointwise).
REDUCTION_OPS = (
    "reduce_sum", "reduce_max", "reduce_min", "reduce_mean", "reduce_prod",
    "sum", "max", "min", "mean", "prod",
)
# Structural: reshape preserves mapped dims; transpose keeps mapped axes leading; a mapped slice is lowered to per-axis gathers over static index constants (static int limit_indices cannot pass through symbolic batch dims);
# pad inserts leading zero padding; concatenate broadcasts the batch dims and shifts the axis; gather/scatter adjust axis args (mapped indices work; mapped-data/unmapped-indices gathers are a v1 TransformError).
STRUCTURAL_OPS = ("reshape", "transpose", "slice", "pad", "concatenate")
# Dot/conv: batched matmul/convolution with mapped dims broadcast across operands; AD via transposed dot (conv VJP: TransformError — no transposed conv op in the IR).
DOT_OPS = ("dot", "conv")
# tril/triu/cumsum pass mapped axes through (per-row semantics); solve broadcasts batch dims like dot.
LINALG_AUX_OPS = ("tril", "triu", "cumsum", "solve")

# --- shared helpers -------------------------------------------------------

def _sym(value: ir.Value) -> "core.SymbolicTensor":
    """Wrap an ir.Value as a SymbolicTensor (dtype/shape from its type)."""
    return core.SymbolicTensor(value=value, dtype=value.type.dtype, shape=value.type.shape)
def _z(tangent) -> bool:
    """True when a tangent/cotangent is structurally zero (None/ZeroTangent)."""
    return tangent is None or isinstance(tangent, ZeroTangent)
def _ok(tangent):
    """The tangent as an ir.Value, or None when structurally zero."""
    return None if _z(tangent) else tangent
def _zero_of(value: ir.Value) -> ir.Value:
    """All-zeros tensor with `value`'s shape/dtype (0 keeps weak promotion dtype-exact)."""
    zero = ops.multiply(_sym(value), 0)
    if zero.dtype != value.type.dtype:
        return ops.cast(zero, value.type.dtype).value
    return zero.value
def _c(dtype, value: float) -> "core.SymbolicTensor":
    """A 0-d constant of exactly `dtype` (avoids NEP-50 float scalar upcasts)."""
    return ops.constant(core.tensor(np.asarray(value, dtype=dtype)))
def _sum_terms(terms) -> ir.Value:
    """Sum a non-empty list of ir.Values (single term returns as-is)."""
    if len(terms) == 1: return terms[0]
    out = ops.add(_sym(terms[0]), _sym(terms[1])).value
    for term in terms[2:]:
        out = ops.add(_sym(out), _sym(term)).value
    return out
def _reduce_leading(v: ir.Value, primal: ir.Value) -> ir.Value:
    """Sum away broadcast-only leading dims so `v` matches `primal`'s rank."""
    diff = v.type.rank - primal.type.rank
    if diff <= 0: return v
    return ops.reduce_sum(_sym(v), axes=tuple(range(diff))).value
def _fold_product(shape) -> "int | core.DimExpr | None":
    """Fold a shape into one int/DimExpr (None when a dim is dynamic)."""
    out = None
    for d in shape:
        if d is None:
            return None
        out = d if out is None else out * d
    return 1 if out is None else out
def _conv_kwargs(op: ir.Op) -> dict:
    """Validated frontend kwargs for rebuilding a conv op from its attrs."""
    attrs = op.attributes
    if attrs.get("batch_group_count", 1) != 1:
        raise TransformError(
            f"vectorize/jvp: cannot transform 'conv' with batch_group_count {attrs['batch_group_count']!r} (v1 gap: batch-grouped conv)"
        )
    return dict(
        strides=attrs.get("strides") or 1,
        padding=attrs.get("padding", "VALID"),
        input_dilation=attrs.get("input_dilation") or 1,
        kernel_dilation=attrs.get("kernel_dilation") or 1,
        feature_group_size=attrs.get("feature_group_count", 1),
    )
def _gelu_coeff(x: "core.SymbolicTensor") -> "core.SymbolicTensor":
    """d gelu / d x = 0.5·(1 + erf(x/√2)) + (x/√(2π))·exp(−x²/2)."""
    inv_sqrt2 = _c(x.dtype, 0.7071067811865476)
    u = ops.multiply(x, inv_sqrt2)
    term1 = ops.multiply(_c(x.dtype, 0.5), ops.add(_c(x.dtype, 1.0), ops.erf(u)))
    term2 = ops.multiply(
        ops.multiply(x, _c(x.dtype, 0.3989422804014327)), ops.exp(ops.negate(ops.square(u)))
    )
    return ops.add(term1, term2)

# --- batching rules -------------------------------------------------------

def _pointwise_batching(op, operands, axes):
    """Elementwise batching: align unmapped operands with leading size-one dims (broadcast) and rebuild; result axes = union of operand axes."""
    name = op.name
    counts = [ax.count for ax in axes]
    mapped_count = max(counts) if counts else 0
    values = [
        ops.broadcast(_sym(value), (1,) * (mapped_count - count) + value.type.shape).value
        if count < mapped_count else value
        for value, count in zip(operands, counts)
    ]
    if name == "cast":
        out = ops.cast(_sym(values[0]), op.attributes["dtype"])
    elif name == "broadcast":
        # Mapped dims must be preserved: new target = operand batch dims + original target.
        target = tuple(op.attributes["shape"])
        out = ops.broadcast(_sym(values[0]), values[0].type.shape[:mapped_count] + target)
    elif name == "select":
        out = ops.select(_sym(values[0]), _sym(values[1]), _sym(values[2]))
    elif name == "stop_gradient":
        out = ops.stop_gradient(_sym(values[0]))
    else:
        out = getattr(ops, name)(*[_sym(value) for value in values])
    return (out.value,), (MappedAxes(tuple(range(mapped_count))),)
def _reduction_batching(op, operands, axes):
    """Reduction batching: shift the reduced axes by the mapped count (result keeps the operand's mapped axes); also argmax/argmin (axis=None flattens the unvectorized dims via a reshape)."""
    value, mapped = operands[0], axes[0]
    count = mapped.count
    name = op.name
    if name in ("argmax", "argmin"):
        fn = getattr(ops, name)
        axis, keepdims = op.attributes["axis"], op.attributes["keepdims"]
        if axis is None and count > 0:
            # Flatten-over-all per batch row: (batch..., *dims) → (batch..., prod(dims)), argmax last axis.
            old_shape = op.operands[0].type.shape
            flat = _fold_product(old_shape)
            if flat is None:
                raise TransformError(f"vectorize: cannot batch '{name}' with axis=None over dynamic dims (v1 gap)")
            flat_value = ops.reshape(_sym(value), value.type.shape[:count] + (flat,))
            out = fn(flat_value, axis=count, keepdims=False)
            if keepdims:
                out = ops.reshape(out, value.type.shape[:count] + (1,) * len(old_shape))
        else:
            new_axis = axis if axis is None else axis + count
            out = fn(_sym(value), axis=new_axis, keepdims=keepdims)
        return (out.value,), (mapped,)
    fn = getattr(ops, name)
    attrs = op.attributes
    old_rank = op.operands[0].type.rank
    reduced = tuple(attrs["axes"])
    if reduced:
        new_axes = tuple(a + count for a in reduced)
    elif old_rank == 0 and count > 0:
        # Reduce-all over a mapped scalar input: identity per row, with the reduction's dtype rule applied.
        kind = attrs["reduce_op"]
        dtype = value.type.dtype
        if kind == "mean" and dtype.kind in "biu":
            out = ops.cast(_sym(value), np.dtype("float64"))
        elif kind in ("sum", "prod") and dtype == np.dtype("bool"):
            out = ops.cast(_sym(value), np.dtype("int64"))
        else:
            out = _sym(value)
        return (out.value,), (mapped,)
    else:
        # Empty axes = ALL (unvectorized) axes: reduce over the per-row axes (mapped dims excluded).
        new_axes = tuple(range(count, count + old_rank))
    out = fn(_sym(value), axes=new_axes, keepdims=attrs["keepdims"])
    return (out.value,), (mapped,)
def _reshape_batching(op, operands, axes):
    """Reshape batching: mapped dims must be preserved — new target = operand batch dims + the original target shape (the -1 wildcard still resolves)."""
    value = operands[0]
    mapped = axes[0]
    target = tuple(op.attributes["shape"])
    if mapped.count > 0:
        target = value.type.shape[:mapped.count] + target
    out = ops.reshape(_sym(value), target)
    return (out.value,), (mapped,)
def _transpose_batching(op, operands, axes):
    """Transpose batching: mapped leading axes stay leading; the original permutation (None = full reversal) applies to the unvectorized axes."""
    value = operands[0]
    mapped = axes[0]
    count = mapped.count
    if count == 0:
        out = ops.transpose(_sym(value), axes=op.attributes["permutation"])
        return (out.value,), (mapped,)
    old_rank = op.operands[0].type.rank
    perm = op.attributes["permutation"]
    base = tuple(range(old_rank - 1, -1, -1)) if perm is None else tuple(perm)
    new_perm = tuple(range(count)) + tuple(p + count for p in base)
    out = ops.transpose(_sym(value), axes=new_perm)
    return (out.value,), (mapped,)
def _slice_batching(op, operands, axes):
    """Slice batching: static int limit_indices cannot pass through symbolic batch dims, so a mapped operand is lowered to gather-based per-row slices (per-axis static int64 `arange` Constants, unmapped/batch-safe). Batch dims pass through untouched: reshape to one batch axis, gather each sliced axis (shifted by one), reshape back; full-range stride-1 slices over static dims are skipped (gather is the identity); unmapped operands rebuild the slice op unchanged."""
    value = operands[0]
    mapped = axes[0]
    count = mapped.count
    starts = tuple(op.attributes["start_indices"])
    limits = tuple(op.attributes["limit_indices"])
    strides = op.attributes["strides"]
    strides = (1,) * len(starts) if strides is None else strides
    if isinstance(strides, int): strides = (strides,) * len(starts)
    if count == 0:
        lengths = tuple(l - s for s, l in zip(starts, limits))
        out = ops.slice(_sym(value), starts, lengths, strides=strides)
        return (out.value,), (mapped,)
    batch = value.type.shape[:count]
    dims = value.type.shape[count:]
    batch_prod = _fold_product(batch)
    if batch_prod is None:
        raise TransformError(f"vectorize: cannot batch op 'slice' with a dynamic batch dim in {batch!r} (v1 gap)")
    x = ops.reshape(_sym(value), (batch_prod,) + dims)
    sliced = []
    for i, (start, limit, stride) in enumerate(zip(starts, limits, strides)):
        dim = dims[i]
        if start == 0 and limit == dim and stride == 1 and isinstance(dim, int):
            sliced.append(dim)  # full-range stride-1: gather is the identity
            continue
        index_array = np.arange(start, limit, stride, dtype=np.int64)
        x = ops.gather(x, ops.constant(core.tensor(index_array)), axis=i + 1)
        sliced.append(index_array.size)
    out = ops.reshape(x, batch + tuple(sliced))
    return (out.value,), (mapped,)
def _pad_batching(op, operands, axes):
    """Pad batching: insert leading zero-padding entries for the mapped dims."""
    count = axes[0].count
    config = ((0, 0),) * count + tuple(op.attributes["padding_config"])
    out = ops.pad(_sym(operands[0]), config, value=op.attributes["value"])
    return (out.value,), (axes[0],)
def _concatenate_batching(op, operands, axes):
    """Concatenate batching: broadcast operands missing mapped axes up to the batch dims (numpy concat does NOT broadcast), then shift the axis by the mapped count."""
    counts = [ax.count for ax in axes]
    mapped_count = max(counts) if counts else 0
    axis = op.attributes["axis"]
    if mapped_count == 0:
        out = ops.concatenate([_sym(v) for v in operands], axis=axis)
        return (out.value,), (UNMAPPED,)
    batch_dims = None
    for value, count in zip(operands, counts):
        if count == mapped_count:
            batch_dims = value.type.shape[:mapped_count]
            break
    aligned = [
        ops.broadcast(_sym(value), tuple(batch_dims) + value.type.shape).value
        if count < mapped_count else value
        for value, count in zip(operands, counts)
    ]
    out = ops.concatenate([_sym(v) for v in aligned], axis=axis + mapped_count)
    return (out.value,), (MappedAxes(tuple(range(mapped_count))),)
def _structural_batching(op, operands, axes):
    """Dispatcher for slice/pad/concatenate batching."""
    fn = {"slice": _slice_batching, "pad": _pad_batching}.get(op.name, _concatenate_batching)
    return fn(op, operands, axes)
def _gather_batching(op, operands, axes):
    """Gather batching: remove ALL mapped axes plus the data axis with one multi-axis gather (infer_gather splices ``indices.shape`` wholesale at the smallest removed axis — the indices' batch dims double as the removed data batch axes, exactly per-row semantics), then transpose the index dims into the data axis's position and reshape back. Mapped data with unmapped indices is a v1 TransformError."""
    x, indices = operands
    kx, ki = axes[0].count, axes[1].count
    attr_axes = op.attributes["axes"]
    if len(attr_axes) != 1:
        raise TransformError("vectorize: cannot batch op 'gather' with multiple axes (v1 gap)")
    axis = attr_axes[0]
    if kx == 0 and ki == 0:
        out = ops.gather(_sym(x), _sym(indices), axis=axis)
        return (out.value,), (UNMAPPED,)
    if kx > 0 and ki != kx:
        raise TransformError("vectorize: cannot batch op 'gather' with mapped data but unmapped indices (v1 deferral: index broadcasting is not supported)")
    mapped_count = max(kx, ki)
    original_shape = op.operands[0].type.shape
    if kx < mapped_count:
        batch_dims = indices.type.shape[:mapped_count]
        x = ops.broadcast(_sym(x), (1,) * (mapped_count - kx) + original_shape).value
        x = ops.broadcast(_sym(x), tuple(batch_dims) + original_shape).value
    # Merge the index dims (beyond the batch dims) into one axis so the indices rank matches the removed-axes count (mapped + 1); a batch-only index shape (B,) becomes (B, 1).
    index_dims = indices.type.shape[mapped_count:]
    merged_index = _fold_product(index_dims)
    if merged_index is None:
        raise TransformError("vectorize: cannot batch op 'gather' with dynamic index dims (v1 gap)")
    if merged_index != 1 or index_dims == ():
        indices = ops.reshape(_sym(indices), indices.type.shape[:mapped_count] + (merged_index,)).value
    builder = current_builder()
    removed = tuple(range(mapped_count)) + (axis + mapped_count,)
    gathered = builder.create("gather", operands=(x, indices), attributes={"axes": removed}).results[0]
    # gathered = (batch..., J, prefix..., suffix...) — move J into the data axis's position: (batch..., prefix..., J, suffix...).
    n_prefix = axis
    rank = gathered.type.rank
    perm = list(range(mapped_count)) + list(range(mapped_count + 1, mapped_count + 1 + n_prefix)) + [mapped_count]
    perm += list(range(mapped_count + 1 + n_prefix, rank))
    reordered = ops.transpose(_sym(gathered), axes=tuple(perm)).value
    # Restore the original multi-dim index shape (drop the size-one J axis for batch-only indices).
    target = tuple(reordered.type.shape[:mapped_count]) + tuple(original_shape[:axis]) + tuple(index_dims) + tuple(original_shape[axis + 1:])
    if tuple(reordered.type.shape) != target:
        reordered = ops.reshape(_sym(reordered), target).value
    return (reordered,), (MappedAxes(tuple(range(mapped_count))),)
def _scatter_batching(op, operands, axes):
    """Scatter batching: shift the axis by the mapped count; the data operand is broadcast up to the batch dims when unmapped. Batched scatter cannot go through `ops.scatter` (its rank check rejects batch dims), so the op is built on the raw builder."""
    x, indices, updates = operands
    kx, ki, ku = axes[0].count, axes[1].count, axes[2].count
    axis = op.attributes["axis"]
    if kx == 0 and ki == 0 and ku == 0:
        out = ops.scatter(_sym(x), _sym(indices), _sym(updates), axis=axis)
        return (out.value,), (UNMAPPED,)
    mapped_count = max(kx, ki, ku)
    if kx > 0 and (ki != kx or ku != kx):
        raise TransformError("vectorize: cannot batch op 'scatter' when the data operand is mapped but indices/updates are not (v1 deferral)")
    if kx == 0 and ki != ku:
        raise TransformError("vectorize: cannot batch op 'scatter' when indices and updates disagree on mapped axes (v1 deferral)")
    if kx < mapped_count:
        original_shape = x.type.shape
        batch_dims = indices.type.shape[:mapped_count]
        x = ops.broadcast(_sym(x), (1,) * (mapped_count - kx) + original_shape).value
        x = ops.broadcast(_sym(x), tuple(batch_dims) + original_shape).value
    builder = current_builder()
    scattered = builder.create(
        "scatter", operands=(x, indices, updates), attributes={"axis": axis + mapped_count}
    ).results[0]
    return (scattered,), (MappedAxes(tuple(range(mapped_count))),)
def _gather_scatter_batching(op, operands, axes):
    """Dispatcher for gather/scatter batching."""
    fn = _gather_batching if op.name == "gather" else _scatter_batching
    return fn(op, operands, axes)
def _dot_batching(op, operands, axes):
    """Dot batching: matmul already broadcasts batch dims; align unmapped operands with leading size-one dims and rebuild."""
    mapped_count = max(axes[0].count, axes[1].count)
    values = [
        ops.broadcast(_sym(value), (1,) * (mapped_count - ax.count) + value.type.shape).value
        if ax.count < mapped_count else value
        for value, ax in zip(operands, axes)
    ]
    out = ops.dot(_sym(values[0]), _sym(values[1]))
    return (out.value,), (MappedAxes(tuple(range(mapped_count))),)
def _conv_batching(op, operands, axes):
    """Conv batching: the IR conv op has exactly ONE batch dim (sx[0]), so a mapped input's batch dims are folded into it via reshape (per-row convs are independent — numerically exact). A mapped kernel would need a symbolic batch_group_count (static int attr) → v1 TransformError."""
    x, w = operands
    kx, kw = axes[0].count, axes[1].count
    if kw > 0:
        raise TransformError("vectorize: cannot batch op 'conv' with a mapped kernel — batch_group_count must be a static int (v1 gap)")
    kwargs = _conv_kwargs(op)
    if kx == 0:
        out = ops.conv(_sym(x), _sym(w), **kwargs)
        return (out.value,), (UNMAPPED,)
    old_shape = op.operands[0].type.shape  # (N, C, *spatial)
    if old_shape[0] is None:
        raise TransformError("vectorize: cannot batch op 'conv' over a dynamic batch dim (v1 gap)")
    batch_dims = x.type.shape[:kx]
    merged = batch_dims[0]
    for dim in batch_dims[1:]: merged = merged * dim
    merged = merged * old_shape[0]
    flat_x = ops.reshape(_sym(x), (merged,) + tuple(old_shape[1:]))
    out = ops.conv(flat_x, _sym(w), **kwargs)
    back = ops.reshape(out, tuple(batch_dims) + (old_shape[0],) + out.shape[1:])
    return (back.value,), (MappedAxes(tuple(range(kx))),)
def _dot_conv_batching(op, operands, axes):
    """Dot/conv batching dispatcher."""
    fn = _dot_batching if op.name == "dot" else _conv_batching
    return fn(op, operands, axes)
def _triangular_batching(op, operands, axes):
    """tril/triu batching: the mask applies to the last two dims of every batch row — rebuild on the (possibly mapped) operand unchanged."""
    out = getattr(ops, op.name)(_sym(operands[0]), k=op.attributes["k"])
    return (out.value,), (axes[0],)
def _cumsum_batching(op, operands, axes):
    """cumsum batching: shift the scan axis by the mapped count (the scan runs per batch row); shape preserved, mapped axes pass through."""
    out = ops.cumsum(_sym(operands[0]), axis=op.attributes["axis"] + axes[0].count, reverse=op.attributes["reverse"])
    return (out.value,), (axes[0],)
def _solve_batching(op, operands, axes):
    """Solve batching: infer_solve broadcasts batch dims — align unmapped operands with leading size-one dims and rebuild."""
    if op.attributes.get("left_side", True) is not True:
        raise TransformError("vectorize: cannot batch op 'solve' with left_side=False (v1 gap)")
    mapped_count = max(axes[0].count, axes[1].count)
    values = [
        ops.broadcast(_sym(value), (1,) * (mapped_count - ax.count) + value.type.shape).value
        if ax.count < mapped_count else value
        for value, ax in zip(operands, axes)
    ]
    out = ops.solve(_sym(values[0]), _sym(values[1]))
    return (out.value,), (MappedAxes(tuple(range(mapped_count))),)
def _constant_batching(op, operands, axes):
    """Constant batching: rebuild with the same payload; result UNMAPPED (data shared across the batch — no mapped dims)."""
    out = ops.constant(core.tensor(op.attributes["value"]))
    return (out.value,), (UNMAPPED,)

# --- JVP/VJP rules ---------------------------------------------------------

#: Unary pointwise derivative coefficient d y/d x — self-adjoint elementwise ops share one table between JVP and VJP: fn(a, y) -> SymbolicTensor.
_UNARY_DERIV = {
    "abs": lambda a, y: ops.sign(a), "negate": lambda a, y: ops.negate(_c(a.dtype, 1.0)),
    "square": lambda a, y: ops.multiply(_c(a.dtype, 2.0), a), "sqrt": lambda a, y: ops.divide(_c(a.dtype, 0.5), y),
    "exp": lambda a, y: y, "log": lambda a, y: ops.divide(_c(a.dtype, 1.0), a),
    "log1p": lambda a, y: ops.divide(_c(a.dtype, 1.0), ops.add(_c(a.dtype, 1.0), a)), "sin": lambda a, y: ops.cos(a),
    "cos": lambda a, y: ops.negate(ops.sin(a)), "tan": lambda a, y: ops.add(_c(a.dtype, 1.0), ops.square(y)),
    "tanh": lambda a, y: ops.subtract(_c(a.dtype, 1.0), ops.square(y)), "sigmoid": lambda a, y: ops.multiply(y, ops.subtract(_c(a.dtype, 1.0), y)),
    "gelu": lambda a, y: _gelu_coeff(a), "erf": lambda a, y: ops.multiply(_c(a.dtype, 1.1283791670955126), ops.exp(ops.negate(ops.square(a)))),
}
def _unary_jvp(op, tangents):
    """Unary pointwise JVP: out-tangent = ta · deriv(a, y)."""
    ta = _ok(tangents[0])
    if ta is None: return (ZeroTangent(),)
    deriv = _UNARY_DERIV[op.name](_sym(op.operands[0]), _sym(op.results[0]))
    return (ops.multiply(_sym(ta), deriv).value,)
def _unary_vjp(op, ct, primals):
    """Unary pointwise VJP: input cotangent = ct · deriv(a, y)."""
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(),)
    deriv = _UNARY_DERIV[op.name](_sym(op.operands[0]), _sym(op.results[0]))
    return (ops.multiply(_sym(ct), deriv).value,)
def _jvp_add(op, tangents):
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    if ta is None and tb is None: return (ZeroTangent(),)
    if ta is None: return (tb,)
    if tb is None: return (ta,)
    return (ops.add(_sym(ta), _sym(tb)).value,)
def _jvp_subtract(op, tangents):
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    if ta is None and tb is None: return (ZeroTangent(),)
    if ta is None: return (ops.negate(_sym(tb)).value,)
    if tb is None: return (ta,)
    return (ops.subtract(_sym(ta), _sym(tb)).value,)
def _jvp_multiply(op, tangents):
    a, b = op.operands
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    terms = []
    if tb is not None: terms.append(ops.multiply(_sym(a), _sym(tb)).value)
    if ta is not None: terms.append(ops.multiply(_sym(ta), _sym(b)).value)
    if not terms: return (ZeroTangent(),)
    return (_sum_terms(terms),)
def _jvp_divide(op, tangents):
    b = op.operands[1]
    y = op.results[0]
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    terms = []
    if ta is not None: terms.append(ops.divide(_sym(ta), _sym(b)).value)
    if tb is not None: terms.append(ops.negate(ops.divide(ops.multiply(_sym(y), _sym(tb)), _sym(b))).value)
    if not terms: return (ZeroTangent(),)
    return (_sum_terms(terms),)
def _jvp_power(op, tangents):
    a, b = op.operands
    y = op.results[0]
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    terms = []
    if tb is not None: terms.append(ops.multiply(_sym(y), ops.multiply(_sym(tb), ops.log(_sym(a)))).value)
    if ta is not None: terms.append(ops.multiply(_sym(y), ops.divide(ops.multiply(_sym(b), _sym(ta)), _sym(a))).value)
    if not terms: return (ZeroTangent(),)
    return (_sum_terms(terms),)
def _jvp_remainder(op, tangents):
    ta = _ok(tangents[0])
    return ((ta if ta is not None else ZeroTangent()),)
def _jvp_maxmin(op, tangents):
    a, b = op.operands
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    if ta is None and tb is None: return (ZeroTangent(),)
    ta = ta if ta is not None else _zero_of(a)
    tb = tb if tb is not None else _zero_of(b)
    cmp = ops.greater if op.name == "maximum" else ops.less
    return (ops.select(cmp(_sym(a), _sym(b)), _sym(ta), _sym(tb)).value,)
def _jvp_relu(op, tangents):
    a = op.operands[0]
    ta = _ok(tangents[0])
    if ta is None: return (ZeroTangent(),)
    return (ops.select(ops.greater(_sym(a), 0), _sym(ta), _sym(_zero_of(a))).value,)
def _jvp_cast(op, tangents):
    ta = _ok(tangents[0])
    if ta is None: return (ZeroTangent(),)
    return (ops.cast(_sym(ta), op.results[0].type.dtype).value,)
def _jvp_select(op, tangents):
    cond = op.operands[0]
    tt, tf = _ok(tangents[1]), _ok(tangents[2])
    if tt is None and tf is None: return (ZeroTangent(),)
    tt = tt if tt is not None else _zero_of(op.operands[1])
    tf = tf if tf is not None else _zero_of(op.operands[2])
    return (ops.select(_sym(cond), _sym(tt), _sym(tf)).value,)
def _jvp_broadcast(op, tangents):
    ta = _ok(tangents[0])
    if ta is None: return (ZeroTangent(),)
    return (ops.broadcast(_sym(ta), op.attributes["shape"]).value,)

#: Per-op JVP rules for the special-cased elementwise ops.
_POINTWISE_JVP = {
    "add": _jvp_add, "subtract": _jvp_subtract, "multiply": _jvp_multiply, "divide": _jvp_divide,
    "power": _jvp_power, "remainder": _jvp_remainder, "maximum": _jvp_maxmin, "minimum": _jvp_maxmin,
    "relu": _jvp_relu, "cast": _jvp_cast, "select": _jvp_select, "broadcast": _jvp_broadcast,
}
def _pointwise_jvp(op, tangents):
    """Per-elementwise-op linearization: unary ops use the shared derivative table; special cases (product/quotient rules, select on the primal mask, cast, broadcast) use their own rules. Primals read from `op.operands`."""
    name = op.name
    if name in _UNARY_DERIV:
        return _unary_jvp(op, tangents)
    fn = _POINTWISE_JVP.get(name)
    if fn is None:
        raise TransformError(f"jvp: no JVP rule for op '{name}'")
    return fn(op, tangents)
def _vjp_add(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    return (ct, ct)
def _vjp_subtract(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    return (ct, ops.negate(_sym(ct)).value)
def _vjp_multiply(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    return (ops.multiply(_sym(ct), _sym(b)).value, ops.multiply(_sym(ct), _sym(a)).value)
def _vjp_divide(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    y = op.results[0]
    g_a = ops.divide(_sym(ct), _sym(b)).value
    g_b = ops.negate(ops.divide(ops.multiply(_sym(ct), _sym(y)), _sym(b))).value
    return (g_a, g_b)
def _vjp_power(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    y = op.results[0]
    g_a = ops.multiply(ops.multiply(_sym(ct), _sym(y)), ops.divide(_sym(b), _sym(a))).value
    g_b = ops.multiply(ops.multiply(_sym(ct), _sym(y)), ops.log(_sym(a))).value
    return (g_a, g_b)
def _vjp_remainder(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    return (ct, ZeroTangent())
def _vjp_maxmin(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    cmp = ops.greater if op.name == "maximum" else ops.less
    mask = ops.cast(cmp(_sym(a), _sym(b)), ct.type.dtype)
    one = _c(ct.type.dtype, 1.0)
    return (ops.multiply(_sym(ct), mask).value, ops.multiply(_sym(ct), ops.subtract(one, mask)).value)
def _vjp_relu(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(),)
    mask = ops.cast(ops.greater(_sym(primals[0]), 0), ct.type.dtype)
    return (ops.multiply(_sym(ct), mask).value,)
def _vjp_cast(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(),)
    return (ops.cast(_sym(ct), primals[0].type.dtype).value,)
def _vjp_select(op, ct, primals):
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent(), ZeroTangent())
    cond = primals[0]
    mask = ops.cast(_sym(cond), ct.type.dtype)
    one = _c(ct.type.dtype, 1.0)
    return (ZeroTangent(), ops.multiply(_sym(ct), mask).value, ops.multiply(_sym(ct), ops.subtract(one, mask)).value)
def _vjp_broadcast(op, ct, primals):
    """Broadcast VJP: reduce the cotangent over the leading inserted axes and over any size-one expanded dims, then reshape back to the input shape."""
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(),)
    a = primals[0]
    diff = ct.type.rank - a.type.rank
    axes = list(range(diff)) + [diff + i for i, dim in enumerate(a.type.shape) if dim == 1]
    if not axes: return (ct,)
    reduced = ops.reduce_sum(_sym(ct), axes=tuple(axes))
    target = tuple(a.type.shape)
    if tuple(reduced.shape) != target:
        reduced = ops.reshape(reduced, target)
    return (reduced.value,)

#: Per-op VJP rules for the special-cased elementwise ops.
_POINTWISE_VJP = {
    "add": _vjp_add, "subtract": _vjp_subtract, "multiply": _vjp_multiply, "divide": _vjp_divide,
    "power": _vjp_power, "remainder": _vjp_remainder, "maximum": _vjp_maxmin, "minimum": _vjp_maxmin,
    "relu": _vjp_relu, "cast": _vjp_cast, "select": _vjp_select, "broadcast": _vjp_broadcast,
}
def _pointwise_vjp(op, cotangents, primals):
    """Per-elementwise-op pullback: product with the local partials (unary ops share the self-adjoint derivative table; special cases use their own rules); returns a tuple aligned with `op.operands`."""
    name = op.name
    if name in _UNARY_DERIV:
        return _unary_vjp(op, cotangents[0], primals)
    fn = _POINTWISE_VJP.get(name)
    if fn is None:
        raise TransformError(f"grad/vjp: no VJP rule for op '{name}'")
    return fn(op, cotangents[0], primals)
def _reduction_jvp(op, tangents):
    """Reduction JVP: reduce the tangent over the same axes (linear op)."""
    t = _ok(tangents[0])
    if t is None: return (ZeroTangent(),)
    name = op.name
    fn = getattr(ops, name, None)
    if fn is None and name in ("sum", "max", "min", "mean", "prod"):
        fn = getattr(ops, "reduce_" + name)
    attrs = op.attributes
    return (fn(_sym(t), axes=attrs["axes"], keepdims=attrs["keepdims"]).value,)
def _expand_reduced(v: ir.Value, x: ir.Value, axes, keepdims: bool) -> ir.Value:
    """Expand a reduced value back to the input shape: insert size-one dims at the reduced axes (when not kept) and broadcast to the input shape."""
    x_shape = tuple(x.type.shape)
    if v.type.rank != len(x_shape):
        v = ops.reshape(_sym(v), tuple(1 if i in axes else d for i, d in enumerate(x_shape))).value
    if tuple(v.type.shape) != x_shape:
        v = ops.broadcast(_sym(v), x_shape).value
    return v
def _reduction_vjp(op, cotangents, primals):
    """Reduction VJP: broadcast the cotangent back to the input shape, with reduction-specific weighting (mean divides by the static count; prod multiplies by y/x; max/min select on the arg mask)."""
    ct = _ok(cotangents[0])
    if ct is None: return (ZeroTangent(),)
    x = primals[0]
    kind = op.attributes["reduce_op"]
    axes, keepdims = tuple(op.attributes["axes"]), op.attributes["keepdims"]
    rank = x.type.rank
    eff = axes if axes else tuple(range(rank))
    if not eff: return (ct,)  # rank-0 identity
    if kind == "sum":
        return (_expand_reduced(ct, x, eff, keepdims),)
    if kind == "mean":
        count = 1
        for i in eff:
            dim = x.type.shape[i]
            if not isinstance(dim, int):
                raise TransformError(f"grad/vjp: cannot differentiate 'mean' over symbolic dim {dim!r} — the reduction count is unknown (v1 gap)")
            count *= dim
        weighted = ops.divide(_sym(ct), count).value
        return (_expand_reduced(weighted, x, eff, keepdims),)
    if kind == "prod":
        # d(prod)/dx_i = prod/x_i = y/x_i (broadcast to the input shape).
        cb = _expand_reduced(ct, x, eff, keepdims)
        yb = _expand_reduced(op.results[0], x, eff, keepdims)
        g = ops.divide(ops.multiply(_sym(cb), _sym(yb)), _sym(x)).value
        return (g,)
    # max/min: gradient flows only through the arg positions.
    cb = _expand_reduced(ct, x, eff, keepdims)
    yb = _expand_reduced(op.results[0], x, eff, keepdims)
    mask = ops.cast(ops.equal(_sym(x), _sym(yb)), ct.type.dtype)
    return (ops.multiply(_sym(cb), mask).value,)
def _structural_jvp(op, tangents):
    """Structural JVP: apply the same structural op to the tangents (all are linear in their tensor operands)."""
    name = op.name
    if name == "concatenate":
        ts = [_ok(t) for t in tangents]
        if all(t is None for t in ts): return (ZeroTangent(),)
        aligned = [_sym(t if t is not None else _zero_of(p)) for t, p in zip(ts, op.operands)]
        return (ops.concatenate(aligned, axis=op.attributes["axis"]).value,)
    if name == "scatter":
        # scatter: linear in x (identity at untouched positions) and updates.
        tx, tupd = _ok(tangents[0]), _ok(tangents[2])
        if tx is None and tupd is None: return (ZeroTangent(),)
        x = tx if tx is not None else _zero_of(op.operands[0])
        upd = tupd if tupd is not None else _zero_of(op.operands[2])
        return (ops.scatter(_sym(x), _sym(op.operands[1]), _sym(upd), axis=op.attributes["axis"]).value,)
    t = _ok(tangents[0])
    if t is None: return (ZeroTangent(),)
    if name == "reshape":
        return (ops.reshape(_sym(t), op.attributes["shape"]).value,)
    if name == "transpose":
        return (ops.transpose(_sym(t), axes=op.attributes["permutation"]).value,)
    if name == "slice":
        starts = op.attributes["start_indices"]
        limits = op.attributes["limit_indices"]
        return (ops.slice(_sym(t), starts, tuple(l - s for s, l in zip(starts, limits)), strides=op.attributes["strides"] or 1).value,)
    if name == "pad":
        return (ops.pad(_sym(t), op.attributes["padding_config"], value=op.attributes["value"]).value,)
    # gather
    axes = op.attributes["axes"]
    if len(axes) != 1:
        raise TransformError("jvp: cannot differentiate 'gather' with multiple axes (v1 gap)")
    return (ops.gather(_sym(t), _sym(op.operands[1]), axis=axes[0]).value,)
def _slice_vjp(op, ct, x):
    """Slice VJP: stride-1 slices restore via pad ((start, dim-limit) per sliced axis); a single strided axis restores via scatter with a static arange index; anything else is a v1 TransformError."""
    starts = op.attributes["start_indices"]
    limits = op.attributes["limit_indices"]
    strides = op.attributes["strides"]
    strides = tuple(strides if strides is not None else (1,) * len(starts))
    rank = x.type.rank
    partial = [
        i for i in range(rank)
        if not (starts[i] == 0 and strides[i] == 1 and limits[i] == x.type.shape[i])
    ]
    if not partial: return ct  # full slice: gradient is the identity
    strided = [i for i in partial if strides[i] > 1]
    if not strided:
        config = []
        for i in range(rank):
            if i not in partial:
                config.append((0, 0))
                continue
            dim = x.type.shape[i]
            if isinstance(dim, int):
                hi = dim - limits[i]
            elif limits[i] == dim:
                hi = 0
            else:
                raise TransformError(f"grad/vjp: cannot invert a slice over symbolic dim {dim!r} (v1 gap)")
            config.append((starts[i], hi))
        return ops.pad(_sym(ct), tuple(config), value=0).value
    if len(partial) != 1 or len(strided) != 1:
        raise TransformError("grad/vjp: cannot invert a slice with strided entries over multiple axes (v1 gap)")
    axis = partial[0]
    zeros = _zero_of(x)
    indices = ops.constant(core.tensor(np.arange(starts[axis], limits[axis], strides[axis])))
    return ops.scatter(_sym(zeros), indices, _sym(ct), axis=axis).value
def _structural_vjp(op, cotangents, primals):
    """Structural VJP: the inverse structural op applied to the cotangent (reshape back, inverse permutation, pad for stride-1 slices / scatter for one strided axis, inverse slice for pad, per-segment slices for concatenate, scatter for gather, gather for scatter)."""
    ct = _ok(cotangents[0])
    if ct is None: return (ZeroTangent(),) * len(primals)
    name = op.name
    x = primals[0]
    if name == "reshape":
        return (ops.reshape(_sym(ct), x.type.shape).value,)
    if name == "transpose":
        perm = op.attributes["permutation"]
        if perm is None:
            return (ops.transpose(_sym(ct)).value,)
        inverse = [0] * len(perm)
        for i, p in enumerate(perm): inverse[p] = i
        return (ops.transpose(_sym(ct), axes=tuple(inverse)).value,)
    if name == "slice":
        return (_slice_vjp(op, ct, x),)
    if name == "pad":
        config = op.attributes["padding_config"]
        starts = tuple(lo for lo, _ in config)
        lengths = []
        for dim in x.type.shape:
            if not isinstance(dim, int):
                raise TransformError(f"grad/vjp: cannot invert a pad over symbolic dim {dim!r} (the inverse slice needs static lengths, v1 gap)")
            lengths.append(dim)
        return (ops.slice(_sym(ct), starts, tuple(lengths), strides=1).value,)
    if name == "concatenate":
        axis = op.attributes["axis"]
        grads = []
        offset = 0
        for p in primals:
            length = p.type.shape[axis]
            if not isinstance(length, int):
                raise TransformError(f"grad/vjp: cannot split a concatenate over symbolic dim {length!r} (v1 gap)")
            starts = [0] * ct.type.rank
            starts[axis] = offset
            lens = list(ct.type.shape)
            lens[axis] = length
            if any(not isinstance(d, int) for d in lens):
                raise TransformError(f"grad/vjp: cannot split a concatenate with symbolic dims {tuple(ct.type.shape)!r} (v1 gap)")
            grads.append(ops.slice(_sym(ct), tuple(starts), tuple(lens), strides=1).value)
            offset += length
        return tuple(grads)
    if name == "gather":
        axes = op.attributes["axes"]
        if len(axes) != 1:
            raise TransformError("grad/vjp: cannot differentiate 'gather' with multiple axes (v1 gap)")
        g = ops.scatter(_sym(_zero_of(x)), _sym(primals[1]), _sym(ct), axis=axes[0])
        return (g.value, ZeroTangent())
    # scatter: g_x = scatter(ct, idx, zeros-of-updates); g_updates = gather(ct, idx); indices are non-differentiable.
    zeros_upd = _zero_of(primals[2])
    g_x = ops.scatter(_sym(ct), _sym(primals[1]), _sym(zeros_upd), axis=op.attributes["axis"])
    g_upd = ops.gather(_sym(ct), _sym(primals[1]), axis=op.attributes["axis"])
    return (g_x.value, ZeroTangent(), g_upd.value)
def _dot_conv_jvp(op, tangents):
    """Dot/conv JVP: the two product terms (dot(a,tb)+dot(ta,b) / conv(a,tw)+conv(ta,w)), zero tangents skipped."""
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    terms = []
    if op.name == "dot":
        if tb is not None: terms.append(ops.dot(_sym(op.operands[0]), _sym(tb)).value)
        if ta is not None: terms.append(ops.dot(_sym(ta), _sym(op.operands[1])).value)
    else:  # conv
        kwargs = _conv_kwargs(op)
        if tb is not None: terms.append(ops.conv(_sym(op.operands[0]), _sym(tb), **kwargs).value)
        if ta is not None: terms.append(ops.conv(_sym(ta), _sym(op.operands[1]), **kwargs).value)
    if not terms: return (ZeroTangent(),)
    return (_sum_terms(terms),)
def _dot_vjp(op, ct, primals):
    """Dot VJP: g_a = dot(ct, bᵀ), g_b = dot(aᵀ, ct), broadcast-only leading batch dims summed away."""
    ct = _ok(ct)
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    ra, rb = a.type.rank, b.type.rank
    b_t = ops.transpose(_sym(b), axes=tuple(range(rb - 2)) + (rb - 1, rb - 2))
    g_a = ops.dot(_sym(ct), b_t).value
    a_t = ops.transpose(_sym(a), axes=tuple(range(ra - 2)) + (ra - 1, ra - 2))
    g_b = ops.dot(a_t, _sym(ct)).value
    return (_reduce_leading(g_a, a), _reduce_leading(g_b, b))
def _dot_conv_vjp(op, cotangents, primals):
    """Dot/conv VJP: transposed dot of the cotangent with each primal operand; conv has NO transposed form in the IR → TransformError."""
    if op.name == "conv":
        raise TransformError("grad/vjp: no VJP rule for op 'conv' — the IR has no transposed-convolution op (v1 gap)")
    return _dot_vjp(op, cotangents[0], primals)
def _triangular_jvp(op, tangents):
    """tril/triu JVP: the mask is linear — apply it to the tangent."""
    t = _ok(tangents[0])
    if t is None: return (ZeroTangent(),)
    return (getattr(ops, op.name)(_sym(t), k=op.attributes["k"]).value,)
def _triangular_vjp(op, cotangents, primals):
    """tril/triu VJP: the mask is self-adjoint — apply it to the cotangent."""
    ct = _ok(cotangents[0])
    if ct is None: return (ZeroTangent(),)
    return (getattr(ops, op.name)(_sym(ct), k=op.attributes["k"]).value,)
def _cumsum_jvp(op, tangents):
    """cumsum JVP: linear — scan the tangent in the same direction."""
    t = _ok(tangents[0])
    if t is None: return (ZeroTangent(),)
    return (ops.cumsum(_sym(t), axis=op.attributes["axis"], reverse=op.attributes["reverse"]).value,)
def _cumsum_vjp(op, cotangents, primals):
    """cumsum VJP: the adjoint scans in the opposite direction."""
    ct = _ok(cotangents[0])
    if ct is None: return (ZeroTangent(),)
    return (ops.cumsum(_sym(ct), axis=op.attributes["axis"], reverse=not op.attributes["reverse"]).value,)
def _solve_jvp(op, tangents):
    """Solve JVP: a·dx + da·x = db ⇒ dx = solve(a, db − da·x). Vector b (rank 1) goes through a (n, 1) reshape (matvec via dot)."""
    if op.attributes.get("left_side", True) is not True:
        raise TransformError("jvp: cannot differentiate op 'solve' with left_side=False (v1 gap)")
    ta, tb = _ok(tangents[0]), _ok(tangents[1])
    if ta is None and tb is None: return (ZeroTangent(),)
    a, b = op.operands
    x = op.results[0]
    if x.type.rank == 1:
        n = b.type.shape[0]
        x_m = ops.reshape(_sym(x), (n, 1))
        diff = ops.reshape(_sym(tb if tb is not None else _zero_of(b)), (n, 1)).value
        if ta is not None:
            diff = ops.subtract(_sym(diff), ops.dot(_sym(ta), x_m)).value
        dx = ops.solve(_sym(a), _sym(diff)).value
        return (ops.reshape(_sym(dx), (n,)).value,)
    diff = tb if tb is not None else _zero_of(b)
    if ta is not None:
        diff = ops.subtract(_sym(diff), ops.dot(_sym(ta), _sym(x))).value
    return (ops.solve(_sym(a), _sym(diff)).value,)
def _solve_vjp(op, cotangents, primals):
    """Solve VJP: g_b = solve(aᵀ, ct); g_a = −g_b·xᵀ (outer product via reshaped multiply for vector x, dot for matrix x); broadcast-only leading batch dims summed away."""
    if op.attributes.get("left_side", True) is not True:
        raise TransformError("grad/vjp: cannot differentiate op 'solve' with left_side=False (v1 gap)")
    ct = _ok(cotangents[0])
    if ct is None: return (ZeroTangent(), ZeroTangent())
    a, b = primals
    x = op.results[0]
    ra = a.type.rank
    a_t = ops.transpose(_sym(a), axes=tuple(range(ra - 2)) + (ra - 1, ra - 2))
    g_b = ops.solve(a_t, _sym(ct)).value
    if x.type.rank == 1:
        n = x.type.shape[0]
        g_b_m = ops.reshape(_sym(g_b), (n, 1))
        x_m = ops.reshape(_sym(x), (1, n))
        g_a = ops.negate(ops.multiply(g_b_m, x_m)).value
    else:
        rx = x.type.rank
        x_t = ops.transpose(_sym(x), axes=tuple(range(rx - 2)) + (rx - 1, rx - 2))
        g_a = ops.negate(ops.dot(_sym(g_b), x_t)).value
    return (_reduce_leading(g_a, a), _reduce_leading(g_b, b))

# --- trivial zero rules (implemented — no algorithm) ----------------------

def _zero_jvp(op, tangents):
    """JVP rule for ops with non-differentiable outputs: zero tangents."""
    return (ZeroTangent(),) * len(op.results)
def _zero_vjp(op, cotangents, primals):
    """VJP rule for ops with non-differentiable outputs: zero gradients."""
    return (ZeroTangent(),) * len(op.operands)
def _stop_gradient_jvp(op, tangents):
    """`stop_gradient`: tangent = zero (spec: gradient is zero)."""
    return (ZeroTangent(),) * len(op.results)
def _stop_gradient_vjp(op, cotangents, primals):
    """`stop_gradient`: cotangent = zero (spec: gradient is zero)."""
    return (ZeroTangent(),) * len(op.operands)

# --- registration (table-driven; runs at import) --------------------------

def register_builtin_rules() -> None:
    """Install the builtin rules above into the public registries.

    Ops left unregistered (⇒ TransformError when transformed): runtime_call, collectives, control flow (cond/while_loop/scan — v1 defers region vectorization), and block_call (rules arrive as block:<name> from etl/block). constant gets a batching rule only — the AD machinery handles it structurally.
    """
    # batching
    nondiff_pointwise = tuple(n for n in NONDIFFERENTIABLE_OUTPUT_OPS if n not in ("argmax", "argmin"))
    for name in ELEMENTWISE_OPS + nondiff_pointwise:
        register_batching_rule(name, _pointwise_batching)
    for name in REDUCTION_OPS + ("argmax", "argmin"):
        register_batching_rule(name, _reduction_batching)
    for name, fn in (("reshape", _reshape_batching), ("transpose", _transpose_batching)):
        register_batching_rule(name, fn)
    for name in ("slice", "pad", "concatenate"):
        register_batching_rule(name, _structural_batching)
    for name in ("gather", "scatter"):
        register_batching_rule(name, _gather_scatter_batching)
    for name in DOT_OPS:
        register_batching_rule(name, _dot_conv_batching)
    for name, fn in (("tril", _triangular_batching), ("triu", _triangular_batching), ("cumsum", _cumsum_batching), ("solve", _solve_batching)):
        register_batching_rule(name, fn)
    register_batching_rule("constant", _constant_batching)

    # jvp
    for name in ELEMENTWISE_OPS:
        register_jvp_rule(name, _pointwise_jvp)
    for name in REDUCTION_OPS:
        register_jvp_rule(name, _reduction_jvp)
    for name in STRUCTURAL_OPS + ("gather", "scatter"):
        register_jvp_rule(name, _structural_jvp)
    for name in DOT_OPS:
        register_jvp_rule(name, _dot_conv_jvp)
    for name in NONDIFFERENTIABLE_OUTPUT_OPS:
        register_jvp_rule(name, _zero_jvp)
    for name, fn in (("tril", _triangular_jvp), ("triu", _triangular_jvp), ("cumsum", _cumsum_jvp), ("solve", _solve_jvp)):
        register_jvp_rule(name, fn)
    register_jvp_rule("stop_gradient", _stop_gradient_jvp)

    # vjp
    for name in ELEMENTWISE_OPS:
        register_vjp_rule(name, _pointwise_vjp)
    for name in REDUCTION_OPS:
        register_vjp_rule(name, _reduction_vjp)
    for name in STRUCTURAL_OPS + ("gather", "scatter"):
        register_vjp_rule(name, _structural_vjp)
    for name in DOT_OPS:
        register_vjp_rule(name, _dot_conv_vjp)
    for name in NONDIFFERENTIABLE_OUTPUT_OPS:
        register_vjp_rule(name, _zero_vjp)
    for name, fn in (("tril", _triangular_vjp), ("triu", _triangular_vjp), ("cumsum", _cumsum_vjp), ("solve", _solve_vjp)):
        register_vjp_rule(name, fn)
    register_vjp_rule("stop_gradient", _stop_gradient_vjp)

register_builtin_rules()
