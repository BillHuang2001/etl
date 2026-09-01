"""External-kernel calls: the graph-side declaration of an opaque named call.

``external_call`` is the graph-side half of the external-kernel mechanism: a
way to declare an opaque, kernel-agnostic call ``name(*inputs)`` with DECLARED
output specs inside an ``@etl.defn`` graph. The actual kernel (a plain Python
callable in v1 — a triton kernel later, written OUTSIDE etl against this
interface) is registered separately under the same NAME in the process that
runs the graph (``etl.external.register_external_kernel``); graph artifacts
carry only the name string.

Contrast with ``runtime_call`` (``etl.ops.constant``): ``runtime_call``
captures an arbitrary Python callback object under an auto-generated,
process-lifetime registry id (``callback_N``) — it is the generic escape
hatch. ``external_call`` is the NAMED, contract-stable variant: the name is
user-chosen and survives graph save/load and process boundaries (the kernel
must be re-registered in any process that runs the graph). Compiler backends
reject it explicitly in v1 (adapter host-dispatch is round 2); the numpy
backend dispatches through ``etl.external``.

Binding rules (see ``etl/CONTEXT.md``, "External kernels"):

- The name is STATIC — a Python str that specializes the graph. Registration
  is a RUN-TIME concern: building a graph does not require the kernel to be
  registered; running it does (``BackendError`` naming the kernel otherwise).
- ``result`` declares the output specs (shapes/dtypes) — the graph's static
  contract. Specs may use symbolic dims (resolved against the runtime input
  bindings, like ``runtime_call``) or ``None`` (runtime-dynamic, unchecked).
- No static (Python) parameters in v1 — inputs are declared tensor operands
  only (static params are a v2 candidate).
"""
from __future__ import annotations

from typing import Tuple, Union

from etl import core
from etl import ir

from . import _utils

__all__ = ["external_call"]


def external_call(name, *operands, result) -> Union[
        "core.SymbolicTensor", Tuple["core.SymbolicTensor", ...]]:
    """Call an externally-registered kernel by NAME at run time.

    Builds an ``external_call`` IR op (effect ``callback``) carrying the
    kernel name and the declared result specs as op attributes. The kernel is
    NOT looked up at trace time — registration is a run-time concern. The
    numpy backend resolves the name through ``etl.external.get_external_kernel``
    (per-backend registry: the exact backend slot, then the default slot) and
    executes the callable with the operand tensors as numpy arrays; the iree
    adapter host-dispatches via segment-split at ``lower()``/``run()``
    (``etl/backends/external_split.py``); xla/tvm reject the op with an
    explicit ``BackendError``.

    Args:
        name: The registered kernel name (non-empty str). Same name at trace
            and run time — graph artifacts carry only this string.
        *operands: ``SymbolicTensor`` or Python scalar inputs to the kernel.
        result: A ``core.TensorSpec`` (single output) or a non-empty
            tuple/list of ``TensorSpec`` (multiple outputs).

    Returns:
        A single ``SymbolicTensor`` when ``result`` is a ``TensorSpec``, else
        a tuple of ``SymbolicTensor`` (one per spec).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``name`` is not a non-empty str; ``result`` is not a
            TensorSpec or a non-empty tuple/list of TensorSpecs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not isinstance(name, str) or not name:
        kind = type(name).__name__ if not isinstance(name, str) else "empty str"
        raise TypeError(
            f"external_call: name must be a non-empty str, got {kind}"
        )
    if isinstance(result, core.TensorSpec):
        single = True
        specs = (result,)
    elif isinstance(result, (tuple, list)):
        if not result or not all(
            isinstance(spec, core.TensorSpec) for spec in result
        ):
            raise TypeError(
                "external_call: result must be a TensorSpec or a non-empty "
                "tuple/list of TensorSpecs"
            )
        single = False
        specs = tuple(result)
    else:
        raise TypeError(
            f"external_call: result must be a TensorSpec or a tuple/list of "
            f"TensorSpecs, got {type(result).__name__}"
        )
    # The IR requires result_specs to be a sequence of ValueType instances
    # EXACTLY equal to the op's result types (ir.verify enforces this).
    result_specs = tuple(
        ir.ValueType(dtype=spec.dtype, shape=tuple(spec.shape))
        for spec in specs
    )
    op_operands = tuple(
        _utils.as_operand(operand, location=loc).value for operand in operands
    )
    op = builder.create(
        "external_call",
        operands=op_operands,
        attributes={
            "name": name,
            "result_specs": result_specs,
        },
        location=loc,
    )
    results = tuple(
        core.SymbolicTensor(
            value=value,
            dtype=value.type.dtype,
            shape=value.type.shape,
            location=loc,
        )
        for value in op.results
    )
    return results[0] if single else results
