"""Reduction kernels: reduce_* family, argmax/argmin.

Covers the IR reduction ops (the sugar names ``sum``/``max``/``min``/
``mean``/``prod`` NEVER reach the IR — ``ops`` expands them onto the
``reduce_*`` ops at build time, and the ``ir`` registry defines no sugar op
names): ``reduce_sum``, ``reduce_max``, ``reduce_min``, ``reduce_mean``,
``reduce_prod``, ``argmax``, ``argmin`` — all ``pure``, one operand, one
result. (``cumsum`` lives in ``indexing.py`` — it honors the ops
dtype-preservation contract with ``dtype=operand.dtype``.)

Semantics (each mirrors the IR inference hook that declared the result type,
so the interpreter's post-kernel validation always agrees):

- ``reduce_*``: attrs ``axes`` (tuple of ints; EMPTY = reduce ALL axes),
  ``keepdims`` (bool), ``reduce_op`` (str — a consistency check only: the OP
  NAME is authoritative for dispatch). Executed as
  ``np.sum/np.max/np.min/np.mean/np.prod(array, axis=axes, keepdims=...)``
  with ``axis=None`` for the empty-axes attr on rank >= 1 (all axes). The
  reducer's numpy dtype rule is EXACTLY what ``ir.inference._reduce_dtype``
  declared (bool sum/prod -> int64, int -> int64/uint64, mean int/bool ->
  float64, max/min preserve) — the kernel never coerces; any pre-casts (e.g.
  ``ops`` mean compensations) arrive as explicit graph ops and execute as-is.
- Rank-0 ``reduce_*``: the empty-axes attr means all axes, but a scalar has
  none — ``axis=()`` (numpy's no-op reduction) applies the reduction's dtype
  promotion while ``keepdims`` keeps nothing, matching the IR's declared
  ``()`` shape for either ``keepdims``.
- ``argmax``/``argmin``: attrs ``axis`` (int | None; None = flatten-reduce),
  ``keepdims`` (bool). ``np.argmax/np.argmin(array, axis=axis)``; the
  declared result dtype is ``int64`` (``infer_arg_reduction``), so the numpy
  ``intp`` index result is cast to ``int64`` when the platform disagrees
  (explicit dtype match, never a silent promotion). ``keepdims`` with an int
  axis -> ``np.expand_dims(result, axis)``; with ``axis=None`` numpy has no
  keepdims, so the scalar index is reshaped to the all-1 shape the IR
  declared (``(1,) * rank``).

Design notes (binding, parent CONTEXT.md):
- Axes and keepdims arrive as op attributes (graph constants); runtime shapes
  come from the IR-declared result types evaluated against the concrete dim
  bindings by the interpreter — the backend carries no second copy of shape
  rules.
- dtype handling is exactly what ops/inference define (no hidden promotion;
  unsupported dtypes raise ``core.BackendError`` naming the op — never
  silently coerced). Object/str/void dtypes are rejected outright for every
  op in this module (numpy would either do Python-level "arithmetic" or
  raise an unrelated error).
- ``axes`` may arrive as a list after IR deserialization (JSON round-trip of
  a tuple) — normalized to ``tuple`` here; numpy accepts both, but the
  empty-tuple test must be reliable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from etl import core

__all__ = ["register_kernels"]

#: numpy reducer per reduce_* op NAME (the op name is authoritative for
#: dispatch; ``reduce_op`` is checked for agreement — see ``_reduce``).
_REDUCE_FUNCS = {
    "reduce_sum": np.sum,
    "reduce_max": np.max,
    "reduce_min": np.min,
    "reduce_mean": np.mean,
    "reduce_prod": np.prod,
}

#: expected ``reduce_op`` attribute value per reduce_* op name (IR contract,
#: mirrors ``etl.ops.reductions._REDUCE_KINDS``).
_REDUCE_KINDS = {
    "reduce_sum": "sum",
    "reduce_max": "max",
    "reduce_min": "min",
    "reduce_mean": "mean",
    "reduce_prod": "prod",
}

#: numpy dtype kinds the reduction family refuses (object/bytes/str/void):
#: reductions on them are either Python-level "arithmetic" or numpy errors —
#: both silent-semantics traps. Rejected with ``core.BackendError``.
_UNSUPPORTED_KINDS = frozenset("OSUV")


def _check_dtype(op_name: str, array: np.ndarray) -> None:
    """Reject object/bytes/str/void dtypes — never silently coerce or promote."""
    if array.dtype.kind in _UNSUPPORTED_KINDS:
        raise core.BackendError(
            f"kernel for op '{op_name}': unsupported dtype {array.dtype} — "
            "object/bytes/str/void tensors cannot be reduced"
        )


def _reduce(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``reduce_sum``/``reduce_max``/``reduce_min``/``reduce_mean``/
    ``reduce_prod``: numpy reduction over the ``axes`` attribute.

    Empty ``axes`` = ALL axes (rank >= 1 -> ``axis=None``; rank 0 -> ``()``,
    which keeps numpy's dtype promotion without ``keepdims`` inventing a
    dim). The result's dtype comes from the reducer itself — numpy's rule is
    exactly what ``ir.inference._reduce_dtype`` declared at trace time, so
    the interpreter's exact-dtype validation agrees by construction.
    """
    del ctx  # reductions need no symbolic-dim state (shapes come from numpy)
    array = operands[0].numpy()
    _check_dtype(op.name, array)
    reduce_op = op.attributes.get("reduce_op")
    expected_kind = _REDUCE_KINDS[op.name]
    if reduce_op is not None and reduce_op != expected_kind:
        raise core.BackendError(
            f"kernel for op '{op.name}': reduce_op attribute {reduce_op!r} "
            f"does not match the op name (expected {expected_kind!r})"
        )
    reducer = _REDUCE_FUNCS[op.name]
    axes = tuple(op.attributes.get("axes", ()))
    keepdims = bool(op.attributes.get("keepdims", False))
    if axes:
        axis = axes
    elif array.ndim == 0:
        # rank-0: no axes exist to reduce; ``()`` applies the dtype rule
        # (bool sum -> int64, mean -> float64, ...) and keeps shape ``()``
        # for either keepdims — the IR declares ``()`` in both cases.
        axis = ()
    else:
        axis = None  # empty axes attr = reduce over ALL axes
    # numpy returns a SCALAR for full reductions (axis=None, keepdims=False)
    # — np.asarray normalizes to the 0-d ndarray core.Tensor requires.
    return core.Tensor(np.asarray(reducer(array, axis=axis, keepdims=keepdims)))


def _arg_reduce(ctx: Any, op: Any, operands: tuple) -> core.Tensor:
    """``argmax``/``argmin``: numpy index reduction (declared dtype int64).

    ``axis=None`` flattens (scalar index; ``keepdims`` -> the all-1 shape the
    IR declared). ``np.argmax/argmin`` return the platform ``intp`` dtype —
    cast to ``int64`` only when it differs (the exact declared result dtype;
    the interpreter validates it afterwards).
    """
    del ctx
    array = operands[0].numpy()
    _check_dtype(op.name, array)
    axis = op.attributes.get("axis")
    keepdims = bool(op.attributes.get("keepdims", False))
    fn = np.argmax if op.name == "argmax" else np.argmin
    result = fn(array, axis=axis)
    if result.dtype != np.int64:
        result = result.astype(np.int64)
    if keepdims:
        if axis is None:
            # numpy has no keepdims for a flattened arg reduction; the IR
            # declares the all-1 shape (infer_arg_reduction).
            result = np.reshape(result, (1,) * array.ndim)
        else:
            result = np.expand_dims(result, axis=axis)
    # a flattened arg reduction returns a numpy scalar — normalize to the
    # 0-d ndarray core.Tensor requires (no copy for real ndarrays).
    return core.Tensor(np.asarray(result))


def register_kernels(table: dict) -> None:
    """Register this module's reduction kernels into the dispatch table.

    Kernel signature convention (see ``kernels/__init__.py``):
    ``kernel(ctx, op, operands) -> Tensor | tuple[Tensor, ...]``.
    """
    for name in _REDUCE_FUNCS:
        table[name] = _reduce
    table["argmax"] = _arg_reduce
    table["argmin"] = _arg_reduce
