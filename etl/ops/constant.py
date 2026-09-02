"""Graph-time constants, runtime callbacks, and gradient barriers.

- ``constant``: the ONLY explicit way to embed concrete tensor data into a
  graph (closure capture is an error — see root CONTEXT.md). Snapshot
  semantics: the data is COPIED into the Constant op at graph-construction
  time; later mutation of the source tensor does not affect the graph.
- ``runtime_call``: an explicit escape hatch that carries an arbitrary Python
  callback as an op attribute (effect kind ``callback``). Backends must
  either implement it or reject it explicitly — the numpy interpreter
  executes it; pure exporters (e.g. stablehlo) fail with ``BackendError``.
- ``stop_gradient``: a unary barrier op (effect ``pure``) whose value passes
  through unchanged but whose gradient is treated as ZERO by all derivative
  transforms.
"""
from __future__ import annotations

import itertools
import os
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from etl import core
from etl import ir

from . import _utils

__all__ = ["constant", "runtime_call", "stop_gradient", "ETL_LARGE_CONSTANT_BYTES"]


def _large_constant_bytes() -> int:
    """Read the size threshold above which :func:`constant` warns.

    Environment variable ``ETL_LARGE_CONSTANT_BYTES`` (integer bytes);
    default 1 MiB (1048576).
    """
    raw = os.environ.get("ETL_LARGE_CONSTANT_BYTES")
    if raw is None:
        return 1 * 1024 * 1024
    return int(raw)


#: Byte threshold above which ``constant`` issues a ``UserWarning``.
#: Read once at import time from ``ETL_LARGE_CONSTANT_BYTES`` (default 1 MiB).
ETL_LARGE_CONSTANT_BYTES: int = _large_constant_bytes()


# --- runtime callback registry (internal) -----------------------------------
#
# IR attributes must serialize, so ``runtime_call`` carries its callback as a
# JSON-safe STRING attribute — a registered identifier. This module-level
# registry maps ids back to the Python callables. Internal contract: the
# numpy interpreter backend resolves the op's ``callback`` attribute through
# ``_get_callback`` at run time; backends that cannot represent callbacks
# reject the op with ``BackendError`` (never silently drop or reorder it).
_CALLBACKS: Dict[str, Callable] = {}
_CALLBACK_IDS: Dict[int, str] = {}  # id(callback) -> registered id (dedupe)
_ID_COUNTER = itertools.count()


def _register_callback(callback: Callable) -> str:
    """Register ``callback`` and return its stable string identifier.

    The same callback object (identity) reuses its id, so repeated
    ``runtime_call``s with one callable share a single registry entry.
    """
    key = id(callback)
    callback_id = _CALLBACK_IDS.get(key)
    if callback_id is None:
        callback_id = f"callback_{next(_ID_COUNTER)}"
        _CALLBACK_IDS[key] = callback_id
        _CALLBACKS[callback_id] = callback
    return callback_id


def _get_callback(callback_id: str) -> Optional[Callable]:
    """The callable registered under ``callback_id`` (``None`` if unknown).

    Internal contract: the numpy interpreter backend uses this lookup to
    resolve the ``runtime_call`` op's string attribute at run time.
    """
    return _CALLBACKS.get(callback_id)


def _wrap_result(op, loc) -> "core.SymbolicTensor":
    """Wrap an op's single result value in a ``SymbolicTensor``.

    The dtype/shape are READ BACK from the IR value's inferred type (never
    recomputed frontend-side) so the wrapper always agrees with ``ir.verify``.
    """
    result = op.result
    return core.SymbolicTensor(
        value=result,
        dtype=result.type.dtype,
        shape=result.type.shape,
        location=loc,
    )


def constant(tensor) -> "core.SymbolicTensor":
    """Embed a concrete ``Tensor`` into the graph as a Constant op.

    The ONLY sanctioned way to embed tensor data (closure capture of a
    ``Tensor`` in traced code raises ``TraceError``). Graph-time only —
    requires an active trace.

    Semantics: the op SNAPSHOTS the data (copies the underlying buffer at
    construction time); later mutation of ``tensor`` does not change the
    graph. Issues a ``UserWarning`` when the data exceeds
    ``ETL_LARGE_CONSTANT_BYTES`` (default 1 MiB, env-tunable), suggesting
    explicit inputs instead.

    Args:
        tensor: A ``core.Tensor``.

    Returns:
        ``SymbolicTensor`` with ``tensor``'s concrete shape and dtype,
        wrapping a ``constant`` IR op (effect ``pure``).

    Raises:
        core.TraceError: no active trace; ``tensor`` is a
            ``SymbolicTensor`` (already a graph value) or not a ``Tensor``.
        core.DeviceError: ``tensor`` is not on a cpu-kind device — the
            snapshot would be an implicit device-to-host transfer (no
            implicit device↔host transfers, ever).
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if isinstance(tensor, core.SymbolicTensor):
        raise core.TraceError(
            "etl.constant expects a concrete Tensor to embed, got a "
            "SymbolicTensor — it is already a graph value: pass it directly "
            "to ops instead of embedding it"
        )
    if not isinstance(tensor, core.Tensor):
        raise core.TraceError(
            f"etl.constant expects a concrete core.Tensor, got "
            f"{type(tensor).__name__} — only concrete tensor data can be "
            "embedded into a graph"
        )
    # Snapshot gate: the data is copied via ``tensor.numpy()`` — for a
    # non-cpu tensor that would be an IMPLICIT device-to-host (D2H)
    # transfer, which is never allowed. cpu-kind tensors (ndarray-backed
    # or cpu:0 payload-backed) hold host data and snapshot normally.
    if tensor.device.kind != "cpu":
        raise core.DeviceError(
            f"etl.constant requires host data: the tensor is on device "
            f"{tensor.device} — no implicit device-to-host transfer happens "
            f"at constant snapshot time; transfer it to the CPU explicitly "
            f"via t.to(core.Device('cpu', 0)) before embedding"
        )
    # SNAPSHOT the data: copy the underlying buffer so later mutation of the
    # source tensor cannot change the graph.
    payload = np.array(tensor.numpy(), copy=True)
    if payload.nbytes > ETL_LARGE_CONSTANT_BYTES:
        warnings.warn(
            f"Embedding a tensor of {payload.nbytes} bytes as a graph "
            f"constant (threshold {ETL_LARGE_CONSTANT_BYTES} bytes, set via "
            "ETL_LARGE_CONSTANT_BYTES); prefer passing it as an explicit "
            "input instead.",
            UserWarning,
            stacklevel=2,
        )
    op = builder.create(
        "constant", attributes={"value": payload}, location=loc
    )
    return _wrap_result(op, loc)


def constant_like(value, tensor) -> "core.SymbolicTensor":
    """Embed a Python scalar as a 0-d Constant op with the dtype of the given
    tensor operand.

    Internal helper used by scalar promotion paths (see
    ``_utils.as_operand``); not part of the public API — prefer
    :func:`constant` for public embedding.

    Args:
        value: Python scalar (bool/int/float/complex).
        tensor: ``SymbolicTensor`` whose dtype guides promotion (weak NEP-50
            semantics via ``_utils.weak_scalar_dtype``).

    Returns:
        0-d ``SymbolicTensor`` constant.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not isinstance(tensor, core.SymbolicTensor):
        raise TypeError(
            f"constant_like expects a SymbolicTensor, got "
            f"{type(tensor).__name__}"
        )
    dtype = _utils.weak_scalar_dtype(value, tensor.dtype)
    payload = np.asarray(value, dtype=dtype)
    op = builder.create(
        "constant", attributes={"value": payload}, location=loc
    )
    return _wrap_result(op, loc)


def runtime_call(callback, *operands, result) -> Union[
        "core.SymbolicTensor", Tuple["core.SymbolicTensor", ...]]:
    """Call an arbitrary Python callback at run time (explicit escape hatch).

    Builds a ``runtime_call`` IR op with effect kind ``callback``, carrying
    ``callback`` and the result spec as op attributes. This is the ONLY op
    that can execute Python at run time; compilers that cannot represent
    callbacks (e.g. the stablehlo exporter) must REJECT it with
    ``BackendError`` — never silently drop or reorder it.

    Args:
        callback: Any Python callable (e.g. a numpy function). Purity is the
            caller's responsibility — etl performs no analysis of it.
        *operands: ``SymbolicTensor`` or Python scalar inputs to the
            callback.
        result: A ``core.TensorSpec`` (single output) or a tuple/list of
            ``TensorSpec`` (multiple outputs).

    Returns:
        A single ``SymbolicTensor`` when ``result`` is a ``TensorSpec``, else
        a tuple of ``SymbolicTensor`` (one per spec).

    Raises:
        core.TraceError: no active trace; concrete ``Tensor`` operand.
        TypeError: ``result`` is not a TensorSpec or tuple of TensorSpecs.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    if not callable(callback):
        raise TypeError(
            f"runtime_call: callback must be callable, got "
            f"{type(callback).__name__}"
        )
    if isinstance(result, core.TensorSpec):
        single = True
        specs = (result,)
    elif isinstance(result, (tuple, list)):
        if not result or not all(
            isinstance(spec, core.TensorSpec) for spec in result
        ):
            raise TypeError(
                "runtime_call: result must be a TensorSpec or a non-empty "
                "tuple/list of TensorSpecs"
            )
        single = False
        specs = tuple(result)
    else:
        raise TypeError(
            f"runtime_call: result must be a TensorSpec or a tuple/list of "
            f"TensorSpecs, got {type(result).__name__}"
        )
    callback_id = _register_callback(callback)
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
        "runtime_call",
        operands=op_operands,
        attributes={
            "callback": callback_id,
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


def stop_gradient(x) -> "core.SymbolicTensor":
    """Barrier op: value passes through unchanged, gradient is zero.

    Used by autodiff transforms (``grad``/``jvp``/``vjp``): when a derivative
    rule encounters a ``stop_gradient`` op it emits a zero tangent/cotangent
    of the matching shape and dtype. The op itself is effect ``pure`` and
    semantically the identity — backends may constant-fold it, but the
    transform layer MUST process it before any folding would erase the
    barrier.

    Args:
        x: ``SymbolicTensor`` or Python scalar.

    Returns:
        ``SymbolicTensor`` identical in shape and dtype to ``x``.
    """
    builder = _utils.check_in_trace()
    loc = _utils.get_location(depth=2)
    x_sym = _utils.as_operand(x, location=loc)
    op = builder.create(
        "stop_gradient", operands=(x_sym.value,), location=loc
    )
    return _wrap_result(op, loc)


# Register the ``etl.constant`` builder hook at import time: ``core.constant``
# (the public ``etl.constant`` entry point) delegates to this module's
# ``constant`` so it can build the Constant op without ``core`` importing
# ``ops`` (import acyclicity; see ``core.symbolic.register_constant_builder``).
core.register_constant_builder(constant)
