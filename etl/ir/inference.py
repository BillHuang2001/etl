"""Shape-inference hooks referenced by ``OpDef``s.

Each hook computes the result types of an op from its operand types and
attributes:

    fn(input_types: tuple[ValueType, ...], attributes: dict) -> tuple[ValueType, ...]

All bodies are ``NotImplementedError`` stubs — this is the architecture phase;
a Manager implements them in Phase 2. Docstrings state the exact expected
semantics so the implementation has no guesswork.

``shape_fn=None`` on an ``OpDef`` means result types are *op-specific* and are
resolved by the ``Builder`` from the op's attributes, regions, or the
enclosing module (``constant``: from payload; ``call``: from callee signature;
``if``: from region terminators; ``runtime_call``/``block_call``: from declared
specs), or must be passed explicitly via
``Builder.create(..., result_types=...)``. ``verify`` enforces agreement in
both cases.
"""

from __future__ import annotations

from typing import Any

from .types import ValueType


def infer_elementwise_binary(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a broadcasting binary elementwise op.

    Shape: DimExpr-based broadcast of the two operand shapes — equal dims pass
    through, ``1`` broadcasts to the other side, symbolic dims unify by name;
    incompatible concrete dims raise ``ShapeError``. Dtype: numpy promotion
    (``np.result_type``) unless ``etl.core`` exposes a promotion API (then
    delegate to it).
    """
    raise NotImplementedError(
        "infer_elementwise_binary: Phase 2 (implementation)"
    )


def infer_elementwise_unary(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a unary elementwise op: same shape and dtype as operand."""
    raise NotImplementedError("infer_elementwise_unary: Phase 2 (implementation)")


def infer_cast(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``cast``: operand shape, dtype from ``attributes["dtype"]``."""
    raise NotImplementedError("infer_cast: Phase 2 (implementation)")


def infer_compare(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a comparison: broadcast shape, dtype ``bool``."""
    raise NotImplementedError("infer_compare: Phase 2 (implementation)")


def infer_select(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``select``: broadcast of all three operands; dtype is
    the promoted dtype of the two branch operands (pred's dtype is bool)."""
    raise NotImplementedError("infer_select: Phase 2 (implementation)")


def infer_broadcast_to(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``broadcast``: shape from ``attributes["shape"]``,
    operand dtype. The operand must be broadcast-compatible with the target."""
    raise NotImplementedError("infer_broadcast_to: Phase 2 (implementation)")


def infer_reshape(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``reshape``: shape from ``attributes["shape"]``
    (with exactly one ``-1`` wildcard allowed if the target has dynamic dims),
    operand dtype. Total element counts must agree (symbolically)."""
    raise NotImplementedError("infer_reshape: Phase 2 (implementation)")


def infer_transpose(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``transpose``: operand shape permuted by
    ``attributes["permutation"]`` (None = full reversal, numpy convention)."""
    raise NotImplementedError("infer_transpose: Phase 2 (implementation)")


def infer_slice(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``slice``: dims computed from ``start_indices``,
    ``limit_indices``, ``strides`` per numpy slice semantics; dims sliced
    completely (stride 1 over the full range) stay symbolic."""
    raise NotImplementedError("infer_slice: Phase 2 (implementation)")


def infer_gather(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``gather``: indices shape spliced into the tensor shape
    at ``attributes["axes"]``; result dtype = tensor dtype."""
    raise NotImplementedError("infer_gather: Phase 2 (implementation)")


def infer_scatter(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``scatter``: shape and dtype of the first operand."""
    raise NotImplementedError("infer_scatter: Phase 2 (implementation)")


def infer_concatenate(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``concatenate``: dims combined along
    ``attributes["axis"]`` (symbolic addition via DimExpr); other dims must
    agree; dtype = promoted dtype of the operands."""
    raise NotImplementedError("infer_concatenate: Phase 2 (implementation)")


def infer_pad(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``pad``: operand dims plus the low/high paddings from
    ``attributes["padding_config"]`` (symbolic addition); operand dtype."""
    raise NotImplementedError("infer_pad: Phase 2 (implementation)")


def infer_reduction(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of a reduce_* op: operand shape with ``axes`` dims removed
    (or set to 1 with ``keepdims=True``); dtype per the reduction (reduce_sum
    of bool/int promotes per numpy; reduce_mean promotes to float)."""
    raise NotImplementedError("infer_reduction: Phase 2 (implementation)")


def infer_arg_reduction(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``argmax``/``argmin``: operand shape with the axis
    removed (or 1 with ``keepdims``); dtype ``int64``."""
    raise NotImplementedError("infer_arg_reduction: Phase 2 (implementation)")


def infer_identity(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of shape-preserving ops (``tril``, ``triu``, ``cumsum``,
    ``stop_gradient``, identity collectives, ``while``): each result mirrors
    the corresponding operand type exactly."""
    raise NotImplementedError("infer_identity: Phase 2 (implementation)")


def infer_dot(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``dot``: numpy matmul contract over the last two dims
    (vector/vector -> scalar, matrix/vector -> vector, batched matmul);
    dtype = numpy promotion of the operands."""
    raise NotImplementedError("infer_dot: Phase 2 (implementation)")


def infer_conv(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``conv``: spatial dims per stride/dilation/padding from
    attributes (``"VALID"`` drops, ``"SAME"`` preserves, tuple padding adds
    exactly); output channel dim from the kernel; batch from the input;
    dtype = promotion of the operands."""
    raise NotImplementedError("infer_conv: Phase 2 (implementation)")


def infer_solve(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``solve``: shape of the right-hand side operand ``b``;
    dtype = promotion of ``a`` and ``b`` (must both be floating)."""
    raise NotImplementedError("infer_solve: Phase 2 (implementation)")


def infer_all_gather(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``all_gather``: input shape with the ``axis`` dim
    multiplied by ``attributes["group_size"]`` (symbolic product); input
    dtype."""
    raise NotImplementedError("infer_all_gather: Phase 2 (implementation)")


def infer_reduce_scatter(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``reduce_scatter``: input shape with the ``axis`` dim
    divided by ``attributes["group_size"]`` (must divide exactly or be
    symbolic); input dtype."""
    raise NotImplementedError("infer_reduce_scatter: Phase 2 (implementation)")


def infer_all_to_all(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``all_to_all``: dims permuted between ``split_axis``
    and ``concat_axis`` per group size; equal axes preserve the shape;
    input dtype."""
    raise NotImplementedError("infer_all_to_all: Phase 2 (implementation)")


def infer_scalar_int64(
    input_types: tuple[ValueType, ...], attributes: dict[str, Any]
) -> tuple[ValueType, ...]:
    """Result type of ``rank``/``world_size``: scalar (``()`` shape) int64."""
    raise NotImplementedError("infer_scalar_int64: Phase 2 (implementation)")
