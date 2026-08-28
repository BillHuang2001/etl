"""RNN-cell block examples (category "block", tag "rnn").

Two single-step recurrent-cell blocks: a full LSTM cell (multi-output
``(h_new, c_new)``) and a GRU cell. Each is a SMALL BLOCK covering several
ops at once — 4 concatenated gate projections + sigmoid/tanh elementwise
fusions + state updates — not an end-to-end program (no ``while_loop`` over
time steps). Static shapes only.

Design notes:

- Gate formulation (LSTM): the four gates i/f/o/g share ONE concatenated
  projection ``gates = x @ w + h @ u + b`` ([2,64] = 4 × [2,16]) and are
  split with ``etl.slice`` — exercises dot + slice + elementwise fusions in
  a single block; the numpy reference mirrors the same formulation. State
  update: ``c_new = f ⊙ c + i ⊙ tanh(g)``, ``h_new = o ⊙ tanh(c_new)``.
  The multi-output ``(h_new, c_new)`` tuple exercises structured graph
  outputs (the same tuple-output shape the numpy/torch references return).
- Gate formulation (GRU): the r/z gates share the concatenated projection
  ``gz = x @ wz + h @ uz + bz`` split by ``etl.slice``; the candidate
  ``h_tilde = tanh(x @ wh + (r ⊙ h) @ uh + bh)``; output
  ``h_new = z ⊙ h + (1 - z) ⊙ h_tilde`` (single-output, the classic
  z-gated interpolation).
- Sigmoid is ``1 / (1 + exp(-x))`` in both the graph and the reference
  (shared :func:`~etl.bench.examples.base.sigmoid_numpy` helper); the
  numpy backend executes the same numpy kernels, so etl-vs-numpy errors are
  ~0 (max_abs 0.0) and the strict conformance defaults hold.
- torch references are exact formula mirrors (``torch.sigmoid`` /
  ``torch.tanh`` / matmul + tensor slicing) — never
  ``torch.nn.LSTMCell``/``torch.nn.GRUCell`` (the nn-module formulations
  differ in weight layout and are not per-op mirrors); fp32
  accumulation-order noise only, the same class as the micro examples
  (strict defaults hold).
"""
from __future__ import annotations

import numpy as np

import etl
from etl import TensorSpec, defn

from .._torch import require_torch
from .base import Example, _F32, register_all, sigmoid_numpy

# --- lstm_cell (one LSTM cell step, multi-output) -----------------------------


@defn
def _lstm_cell_graph(x, h, c, w, u, b):
    """One LSTM cell step. x[2,16], h[2,16], c[2,16], w[16,64], u[16,64],
    b[64] (all f32).

    ``gates = x @ w + h @ u + b`` is the concatenated i/f/o/g projection
    ([2,64]); each gate is a 16-wide slice, then
    ``c_new = f ⊙ c + i ⊙ tanh(g)``, ``h_new = o ⊙ tanh(c_new)`` —
    returned as the tuple ``(h_new, c_new)``.
    """
    gates = etl.add(etl.add(etl.dot(x, w), etl.dot(h, u)), b)  # [2,64]
    i = etl.sigmoid(etl.slice(gates, (0, 0), (2, 16)))
    f = etl.sigmoid(etl.slice(gates, (0, 16), (2, 16)))
    o = etl.sigmoid(etl.slice(gates, (0, 32), (2, 16)))
    g = etl.tanh(etl.slice(gates, (0, 48), (2, 16)))
    c_new = etl.add(etl.multiply(f, c), etl.multiply(i, g))
    h_new = etl.multiply(o, etl.tanh(c_new))
    return (h_new, c_new)


def _lstm_cell_numpy(inputs):
    x, h, c, w, u, b = inputs
    gates = x @ w + h @ u + b
    i = sigmoid_numpy(gates[:, 0:16])
    f = sigmoid_numpy(gates[:, 16:32])
    o = sigmoid_numpy(gates[:, 32:48])
    g = np.tanh(gates[:, 48:64])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return (h_new, c_new)


def _lstm_cell_torch(inputs, device=None):
    torch = require_torch()
    x, h, c, w, u, b = (torch.as_tensor(a, device=device) for a in inputs)
    gates = x @ w + h @ u + b
    i = torch.sigmoid(gates[:, 0:16])
    f = torch.sigmoid(gates[:, 16:32])
    o = torch.sigmoid(gates[:, 32:48])
    g = torch.tanh(gates[:, 48:64])
    c_new = f * c + i * g
    h_new = o * torch.tanh(c_new)
    return (h_new.cpu().numpy(), c_new.cpu().numpy())


# --- gru_cell (one GRU cell step) --------------------------------------------


@defn
def _gru_cell_graph(x, h, wz, uz, bz, wh, uh, bh):
    """One GRU cell step. x[2,16], h[2,16], wz/uz[16,32], bz[32],
    wh/uh[16,16], bh[16] (all f32).

    ``gz = x @ wz + h @ uz + bz`` is the concatenated r/z projection
    ([2,32]); r and z are 16-wide slices, then
    ``h_tilde = tanh(x @ wh + (r ⊙ h) @ uh + bh)`` and
    ``h_new = z ⊙ h + (1 - z) ⊙ h_tilde`` — returned as a single tensor.
    """
    gz = etl.add(etl.add(etl.dot(x, wz), etl.dot(h, uz)), bz)  # [2,32]
    r = etl.sigmoid(etl.slice(gz, (0, 0), (2, 16)))
    z = etl.sigmoid(etl.slice(gz, (0, 16), (2, 16)))
    gh = etl.add(
        etl.add(etl.dot(x, wh), etl.dot(etl.multiply(r, h), uh)), bh
    )
    h_tilde = etl.tanh(gh)
    return etl.add(etl.multiply(z, h), etl.multiply(etl.subtract(1.0, z), h_tilde))


def _gru_cell_numpy(inputs):
    x, h, wz, uz, bz, wh, uh, bh = inputs
    gz = x @ wz + h @ uz + bz
    r = sigmoid_numpy(gz[:, 0:16])
    z = sigmoid_numpy(gz[:, 16:32])
    gh = x @ wh + (r * h) @ uh + bh
    h_tilde = np.tanh(gh)
    return z * h + (1.0 - z) * h_tilde


def _gru_cell_torch(inputs, device=None):
    torch = require_torch()
    x, h, wz, uz, bz, wh, uh, bh = (
        torch.as_tensor(a, device=device) for a in inputs
    )
    gz = x @ wz + h @ uz + bz
    r = torch.sigmoid(gz[:, 0:16])
    z = torch.sigmoid(gz[:, 16:32])
    gh = x @ wh + (r * h) @ uh + bh
    h_tilde = torch.tanh(gh)
    return (z * h + (1.0 - z) * h_tilde).cpu().numpy()


# ---------------------------------------------------------------------------
# Registry (category "block", tag "rnn")
# ---------------------------------------------------------------------------

register_all([
    Example(
        name="lstm_cell",
        description=(
            "one LSTM cell step: 4 concatenated gate projections + "
            "sigmoid/tanh state update, multi-output (h_new, c_new) "
            "(B=2, D=H=16)"
        ),
        specs=(
            TensorSpec((2, 16), _F32),
            TensorSpec((2, 16), _F32),
            TensorSpec((2, 16), _F32),
            TensorSpec((16, 64), _F32),
            TensorSpec((16, 64), _F32),
            TensorSpec((64,), _F32),
        ),
        graph=_lstm_cell_graph,
        numpy_ref=_lstm_cell_numpy,
        torch_ref=_lstm_cell_torch,
        # numpy backend executes the same kernels as the reference (max_abs
        # 0.0); the torch ref is an exact formula mirror — fp32 noise only.
        category="block",
        tags=("rnn",),
    ),
    Example(
        name="gru_cell",
        description=(
            "one GRU cell step: r/z gate projection + tanh candidate + "
            "z-gated output h_new (B=2, D=H=16)"
        ),
        specs=(
            TensorSpec((2, 16), _F32),
            TensorSpec((2, 16), _F32),
            TensorSpec((16, 32), _F32),
            TensorSpec((16, 32), _F32),
            TensorSpec((32,), _F32),
            TensorSpec((16, 16), _F32),
            TensorSpec((16, 16), _F32),
            TensorSpec((16,), _F32),
        ),
        graph=_gru_cell_graph,
        numpy_ref=_gru_cell_numpy,
        torch_ref=_gru_cell_torch,
        # Same noise class as lstm_cell: numpy exact (max_abs 0.0), torch
        # fp32 accumulation-order noise only.
        category="block",
        tags=("rnn",),
    ),
])
