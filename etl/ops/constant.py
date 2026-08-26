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

import os
from typing import Tuple, Union

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
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
