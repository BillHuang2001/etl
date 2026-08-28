"""Basic-op conformance examples (category "op", tag "basic").

Thirteen single-family examples covering the operation-level surface of etl
that the curated micro/grad/vectorize examples do not isolate: one example
per op family, each testing ONE or TWO etl ops (the bitwise and comparison
examples cover their full 3-op family — ``and``/``or``/``xor`` and
``not_equal``/``less_equal``/``greater_equal`` — as a single unit, mirroring
how the family is listed in the harness brief).

Covered ops (13 examples):

- ``erf`` (single-op; numpy ref mirrors the numpy-backend kernel exactly:
  vectorized ``math.erf`` cast back to the operand dtype)
- ``gelu`` (single-op, exact erf form — vs :func:`~etl.bench.examples.base.
  gelu_numpy`, the same formula the numpy backend kernel executes)
- ``tril`` / ``triu`` with explicit ``k`` offsets (k=1 / k=-1; numpy refs
  ``np.tril``/``np.triu``)
- ``argmax`` / ``argmin`` along ``axis=-1`` (int64 index outputs, exactly
  like numpy's)
- ``bitwise_and`` / ``bitwise_or`` / ``bitwise_xor`` on int32 inputs
- ``logical_or`` / ``logical_not`` on bool inputs (forward-only; the
  ``logical_and`` counterpart is already exercised in the ``grad_mix`` graph)
- comparisons ``not_equal`` / ``less_equal`` / ``greater_equal`` (bool
  outputs)
- ``power`` with a fractional static exponent (``x ** 2.5``, positive base
  via ``inputs_fn``) and ``remainder`` (``a % b`` with positive divisor)
- ``gather`` (x[8,16], idx[6] int32 distinct non-negative) and ``scatter``
  (functional update of a copy; distinct indices make the reference
  unambiguous)
- ``pad`` constant mode (``np.pad`` semantics) and strided ``slice``
  (``x[::2, 1:5]``; etl ``slice`` takes ``start``/``lengths``/``strides``
  where ``lengths`` is the SOURCE-range length, i.e. output length is
  ``lengths // stride`` per axis)
- ``stop_gradient`` forward passthrough (``etl.stop_gradient(x) == x``)
- ``select`` with a computed bool mask (``x > 0`` — ``greater``/``negate``
  are incidental; the op under test is ``select``/``where``)
- ``broadcast`` (``etl.broadcast(x, (4, 8))`` vs ``np.broadcast_to``)

StableHLO v1 export status (all explicit ``BackendError`` at ``lower()``
time on compiler backends — never silent fallback; the numpy backend handles
everything here, which is what this module validates):

- deferred: ``gather``/``scatter``, ``tril``/``triu``, ``erf``/``gelu``,
  ``argmax``/``argmin`` (the same v1 export gaps as the documented
  ``grad_structural`` gather/scatter/solve deferral);
- exported: ``power``, ``remainder``, ``bitwise_*``, ``logical_*``,
  comparisons, ``pad``, ``slice``, ``stop_gradient``, ``select``,
  ``broadcast``.

Tolerance justification: NONE of the examples needs a per-example override —
the numpy backend executes the same numpy kernels the references use, so
every example is exact (measured max_abs_error 0.0 on the numpy backend,
seed 0). All 13 pass the strict global defaults (rtol=atol=1e-5,
tolerance=None).

Shape note: the brief suggested ``etl.broadcast(x[4], (4, 8))``, but that is
NOT valid numpy broadcasting — the aligned dims 4 vs 8 conflict (etl raises
``ShapeError`` at trace time, numpy raises ``ValueError``). The example uses
the canonical row-broadcast pattern ``x[8] -> (4, 8)`` instead
(``np.broadcast_to(x, (4, 8))``), which exercises the same op.
"""
from __future__ import annotations

import math

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, gelu_numpy, register_all

# --- erf --------------------------------------------------------------------


@defn
def _erf_graph(x):
    return etl.erf(x)


def _erf_numpy_ref(inputs):
    # Mirrors the numpy-backend kernel exactly (math.erf vectorized via
    # frompyfunc, cast back to the operand dtype) — no scipy dependency.
    (x,) = inputs
    return np.frompyfunc(math.erf, 1, 1)(x).astype(x.dtype)


def _erf_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.erf(torch.as_tensor(x, device=device)).cpu().numpy()


# --- gelu (exact erf form) ----------------------------------------------------


@defn
def _gelu_graph(x):
    return etl.gelu(x)


def _gelu_numpy(inputs):
    (x,) = inputs
    return gelu_numpy(x)


def _gelu_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.nn.functional.gelu(torch.as_tensor(x, device=device)).cpu().numpy()


# --- tril / triu (explicit k offsets) -----------------------------------------


@defn
def _tril_triu_graph(x):
    return etl.tril(x, k=1), etl.triu(x, k=-1)


def _tril_triu_numpy(inputs):
    (x,) = inputs
    return np.tril(x, k=1), np.triu(x, k=-1)


def _tril_triu_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    t = torch.as_tensor(x, device=device)
    return torch.tril(t, diagonal=1).cpu().numpy(), torch.triu(t, diagonal=-1).cpu().numpy()


# --- argmax / argmin (axis=-1, int64 outputs) ---------------------------------


@defn
def _argmax_argmin_graph(x):
    return etl.argmax(x, axis=-1), etl.argmin(x, axis=-1)


def _argmax_argmin_numpy(inputs):
    (x,) = inputs
    return np.argmax(x, axis=-1), np.argmin(x, axis=-1)


def _argmax_argmin_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    t = torch.as_tensor(x, device=device)
    return torch.argmax(t, dim=-1).cpu().numpy(), torch.argmin(t, dim=-1).cpu().numpy()


# --- bitwise and / or / xor (int32) -------------------------------------------


@defn
def _bitwise_graph(x, y):
    return etl.bitwise_and(x, y), etl.bitwise_or(x, y), etl.bitwise_xor(x, y)


def _bitwise_numpy(inputs):
    x, y = inputs
    return np.bitwise_and(x, y), np.bitwise_or(x, y), np.bitwise_xor(x, y)


def _bitwise_torch(inputs, device=None):
    torch = require_torch()
    x, y = (torch.as_tensor(a, device=device) for a in inputs)
    return (
        torch.bitwise_and(x, y).cpu().numpy(),
        torch.bitwise_or(x, y).cpu().numpy(),
        torch.bitwise_xor(x, y).cpu().numpy(),
    )


# --- logical or / not (bool) --------------------------------------------------


@defn
def _logical_graph(a, b):
    return etl.logical_or(a, b), etl.logical_not(a)


def _logical_numpy(inputs):
    a, b = inputs
    return np.logical_or(a, b), np.logical_not(a)


def _logical_torch(inputs, device=None):
    torch = require_torch()
    a, b = (torch.as_tensor(x, device=device) for x in inputs)
    return torch.logical_or(a, b).cpu().numpy(), torch.logical_not(a).cpu().numpy()


# --- comparisons: not_equal / less_equal / greater_equal ----------------------


@defn
def _compare_graph(x, y):
    return etl.not_equal(x, y), etl.less_equal(x, y), etl.greater_equal(x, y)


def _compare_numpy(inputs):
    x, y = inputs
    return np.not_equal(x, y), np.less_equal(x, y), np.greater_equal(x, y)


def _compare_torch(inputs, device=None):
    torch = require_torch()
    x, y = (torch.as_tensor(a, device=device) for a in inputs)
    return (
        torch.ne(x, y).cpu().numpy(),
        torch.le(x, y).cpu().numpy(),
        torch.ge(x, y).cpu().numpy(),
    )


# --- power (fractional exponent) + remainder (positive divisor) ---------------


def _positive_pair_inputs(seed):
    """Two strictly positive float32 [6,6] arrays: |normal| + offset.

    ``power(x, 2.5)`` needs a positive base and ``remainder(x, y)`` needs a
    positive divisor (so the result lies in ``[0, y)``, matching
    ``np.remainder`` unambiguously)."""
    rng = np.random.default_rng(seed)
    x = (np.abs(rng.standard_normal((6, 6))) + 0.5).astype(np.float32)
    y = (np.abs(rng.standard_normal((6, 6))) + 1.0).astype(np.float32)
    return [x, y]


@defn
def _power_remainder_graph(x, y):
    return etl.power(x, 2.5), etl.remainder(x, y)


def _power_remainder_numpy(inputs):
    x, y = inputs
    return np.power(x, 2.5), np.remainder(x, y)


def _power_remainder_torch(inputs, device=None):
    torch = require_torch()
    x, y = (torch.as_tensor(a, device=device) for a in inputs)
    return torch.pow(x, 2.5).cpu().numpy(), torch.remainder(x, y).cpu().numpy()


# --- gather + scatter ----------------------------------------------------------


def _gather_scatter_inputs(seed):
    """x[8,16] f32, idx[6] int32 DISTINCT non-negative, src[6,16] f32.

    Distinct indices make the scatter reference unambiguous (functional
    update of a copy: ``out[idx] = src``)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, 16)).astype(np.float32)
    idx = rng.choice(8, size=6, replace=False).astype(np.int32)
    src = rng.standard_normal((6, 16)).astype(np.float32)
    return [x, idx, src]


@defn
def _gather_scatter_graph(x, idx, src):
    return etl.gather(x, idx, axis=0), etl.scatter(x, idx, src, axis=0)


def _gather_scatter_numpy(inputs):
    x, idx, src = inputs
    gathered = x[idx]
    scattered = x.copy()
    scattered[idx] = src
    return gathered, scattered


# --- pad (constant mode) + strided slice --------------------------------------


@defn
def _pad_slice_graph(x):
    return (
        etl.pad(x, ((1, 2), (0, 1)), value=0.0),
        etl.slice(x, (0, 1), (6, 4), strides=(2, 1)),
    )


def _pad_slice_numpy(inputs):
    (x,) = inputs
    padded = np.pad(x, ((1, 2), (0, 1)), mode="constant", constant_values=0.0)
    sliced = x[::2, 1:5]
    return padded, sliced


def _pad_slice_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    t = torch.as_tensor(x, device=device)
    padded = torch.nn.functional.pad(t, (0, 1, 1, 2), mode="constant", value=0.0)
    return padded.cpu().numpy(), t[::2, 1:5].cpu().numpy()


# --- stop_gradient forward passthrough ----------------------------------------


@defn
def _stop_gradient_graph(x):
    return etl.stop_gradient(x)


def _stop_gradient_numpy(inputs):
    (x,) = inputs
    return np.array(x, copy=True)


# --- select (computed bool mask) ----------------------------------------------


@defn
def _select_graph(x):
    return etl.select(x > 0.0, x, etl.negate(x))


def _select_numpy(inputs):
    (x,) = inputs
    return np.where(x > 0.0, x, -x)


def _select_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    t = torch.as_tensor(x, device=device)
    return torch.where(t > 0.0, t, -t).cpu().numpy()


# --- broadcast ----------------------------------------------------------------


@defn
def _broadcast_graph(x):
    return etl.broadcast(x, (4, 8))


def _broadcast_numpy(inputs):
    (x,) = inputs
    return np.broadcast_to(x, (4, 8))


def _broadcast_torch(inputs, device=None):
    torch = require_torch()
    (x,) = inputs
    return torch.broadcast_to(torch.as_tensor(x, device=device), (4, 8)).cpu().numpy()


# --- registry -----------------------------------------------------------------

register_all([
    Example(
        name="op_erf",
        description="elementwise Gauss error function (etl.erf, vectorized math.erf kernel)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_erf_graph,
        numpy_ref=_erf_numpy_ref,
        torch_ref=_erf_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_gelu",
        description="GELU in its exact erf form (etl.gelu vs the shared gelu_numpy reference)",
        specs=(TensorSpec((32, 64), _F32),),
        graph=_gelu_graph,
        numpy_ref=_gelu_numpy,
        torch_ref=_gelu_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_tril_triu",
        description="lower/upper triangle with explicit k offsets (tril k=1, triu k=-1)",
        specs=(TensorSpec((8, 8), _F32),),
        graph=_tril_triu_graph,
        numpy_ref=_tril_triu_numpy,
        torch_ref=_tril_triu_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_argmax_argmin",
        description="index of the max/min along axis=-1 (int64 outputs, numpy argmax/argmin semantics)",
        specs=(TensorSpec((4, 8), _F32),),
        graph=_argmax_argmin_graph,
        numpy_ref=_argmax_argmin_numpy,
        torch_ref=_argmax_argmin_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_bitwise",
        description="bitwise and/or/xor on int32 tensors",
        specs=(
            TensorSpec((6, 6), etl.int32),
            TensorSpec((6, 6), etl.int32),
        ),
        graph=_bitwise_graph,
        numpy_ref=_bitwise_numpy,
        torch_ref=_bitwise_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_logical",
        description="logical or / logical not on bool tensors (forward-only)",
        specs=(
            TensorSpec((5, 5), etl.bool_),
            TensorSpec((5, 5), etl.bool_),
        ),
        graph=_logical_graph,
        numpy_ref=_logical_numpy,
        torch_ref=_logical_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_compare",
        description="comparisons not_equal / less_equal / greater_equal (bool outputs)",
        specs=(
            TensorSpec((4, 4), _F32),
            TensorSpec((4, 4), _F32),
        ),
        graph=_compare_graph,
        numpy_ref=_compare_numpy,
        torch_ref=_compare_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_power_remainder",
        description="power with fractional static exponent (x**2.5) and remainder a%b with positive divisor",
        specs=(
            TensorSpec((6, 6), _F32),
            TensorSpec((6, 6), _F32),
        ),
        graph=_power_remainder_graph,
        numpy_ref=_power_remainder_numpy,
        torch_ref=_power_remainder_torch,
        inputs_fn=_positive_pair_inputs,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_gather_scatter",
        description="gather (take) rows along axis 0 and scatter (functional update) with distinct int32 indices",
        specs=(
            TensorSpec((8, 16), _F32),
            TensorSpec((6,), etl.int32),
            TensorSpec((6, 16), _F32),
        ),
        graph=_gather_scatter_graph,
        numpy_ref=_gather_scatter_numpy,
        inputs_fn=_gather_scatter_inputs,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_pad_slice",
        description="constant-mode pad (np.pad semantics) and strided slice (x[::2, 1:5])",
        specs=(TensorSpec((6, 10), _F32),),
        graph=_pad_slice_graph,
        numpy_ref=_pad_slice_numpy,
        torch_ref=_pad_slice_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_stop_gradient",
        description="stop_gradient forward passthrough (etl.stop_gradient(x) == x)",
        specs=(TensorSpec((4, 8), _F32),),
        graph=_stop_gradient_graph,
        numpy_ref=_stop_gradient_numpy,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_select",
        description="ternary select with a computed bool mask (np.where(x > 0, x, -x))",
        specs=(TensorSpec((5, 6), _F32),),
        graph=_select_graph,
        numpy_ref=_select_numpy,
        torch_ref=_select_torch,
        category="op",
        tags=("basic",),
    ),
    Example(
        name="op_broadcast",
        description="broadcast a length-8 vector to (4, 8) (np.broadcast_to semantics)",
        specs=(TensorSpec((8,), _F32),),
        graph=_broadcast_graph,
        numpy_ref=_broadcast_numpy,
        torch_ref=_broadcast_torch,
        category="op",
        tags=("basic",),
    ),
])
