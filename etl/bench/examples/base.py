"""Shared infrastructure for the etl.bench example registry.

Holds the :class:`Example` dataclass, the module-level registry
(``_REGISTRY`` + ``register``/``register_all``), the shared input generator
(:func:`generate_inputs`), the shared numpy references — conv
(:func:`_conv2d_numpy` / :func:`conv2d_im2col_numpy`), activation/norm
helpers (:func:`softmax_numpy` / :func:`layernorm_numpy` /
:func:`sigmoid_numpy` / :func:`gelu_numpy`) — and the finite-difference
gradient helper (:func:`fd_gradient`) used by the gradient examples.
Concrete examples live in sibling modules (``micro``, ``grad``,
``vectorize``, ``op_large``, the ``op_*`` / ``block_*`` / ``e2e_*``
modules) which self-register at import time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from etl import float32

__all__ = [
    "Example",
    "UnknownExampleError",
    "register",
    "register_all",
    "generate_inputs",
    "conv2d_im2col_numpy",
    "softmax_numpy",
    "layernorm_numpy",
    "sigmoid_numpy",
    "gelu_numpy",
    "fd_gradient",
]

_F32 = float32


class UnknownExampleError(ValueError):
    """Raised by :func:`get_example` for unknown example names."""


@dataclass(frozen=True)
class Example:
    """A registered benchmark/conformance example.

    Attributes:
        name: stable registry key.
        description: one-line human-readable description.
        specs: tuple of ``etl.TensorSpec`` (static integer shapes).
        graph: ``@etl.defn`` graph taking one symbolic tensor per spec.
        numpy_ref: ``(inputs) -> ndarray | tuple[ndarray]``; pure numpy.
        torch_ref: optional ``(inputs, device=None) -> ndarray | tuple[ndarray]``
            factory that imports torch inside its body (never at module scope);
            ``device`` is an optional ``torch.device`` (``None`` = CPU).
        rtol: per-example relative-tolerance override (``None`` = fall back
            to the global ``conformance()`` value).
        atol: per-example absolute-tolerance override (``None`` = fall back
            to the global ``conformance()`` value).
        tolerance: per-example max-abs-error override (``None`` = fall back
            to the global ``conformance()`` value). Independent of
            ``rtol``/``atol``.
        category: grouping key — one of the three registry categories
            ``"op"``, ``"block"``, ``"e2e"``; used by the CLI ``--examples``
            category expansion (default ``"op"``).
        inputs_fn: optional custom input generator ``(seed) -> list[np.ndarray]``
            producing one numpy array per spec (e.g. non-negative integer
            indices for gather); :meth:`generate_inputs` uses it when set.
        tags: tuple of subgroup selector strings (e.g. ``"micro"``,
            ``"grad"``, ``"control-flow"``, ``"vmap"``, ``"custom"``,
            ``"xla"``, ``"large"``). Tags are also accepted by the CLI
            ``--examples`` expansion (after categories, before bare names).
        runner: optional runner factory for multi-run procedures (e.g. a
            Python-level training loop): ``runner(backend, device, opts) ->
            callable(inputs) -> outputs`` — builds executables ONCE and
            returns a run-callable taking the same inputs list the single-run
            path uses and returning the same outputs structure as
            ``numpy_ref``. See the runner contract in
            ``etl.bench._util``'s docstring.
    """

    name: str
    description: str
    specs: tuple
    graph: Callable
    numpy_ref: Callable
    torch_ref: Optional[Callable] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    tolerance: Optional[float] = None
    category: str = "op"
    inputs_fn: Optional[Callable[[int], list]] = None
    tags: tuple = ()
    runner: Optional[Callable] = None

    def generate_inputs(self, seed: int = 0):
        """Generate a list of numpy arrays matching ``specs`` (see module
        :func:`generate_inputs`)."""
        return generate_inputs(self, seed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Registered examples in insertion order (name -> Example). Later modules
#: call :func:`register_all` at import time; import order in the package
#: ``__init__`` defines the registry order.
_REGISTRY: dict = {}


def register(example: Example) -> None:
    """Register ``example`` under its name (duplicate names raise ValueError)."""
    if example.name in _REGISTRY:
        raise ValueError(f"duplicate example registration: {example.name!r}")
    _REGISTRY[example.name] = example


def register_all(examples) -> None:
    """Register every example from the iterable (registry order preserved)."""
    for example in examples:
        register(example)


def generate_inputs(example: Example, seed: int = 0):
    """Generate a list of numpy arrays matching ``example.specs``.

    Uses ``numpy.random.default_rng(seed)``: standard-normal draws for
    floating dtypes, small integers for integer dtypes, uniform bools for
    bool specs. Specs must have static integer shapes (``Dim``/``DimExpr``
    shapes are not supported by the harness — explicit error).

    When ``example.inputs_fn`` is set it is called with ``seed`` instead and
    its result is validated (a list of numpy arrays matching the specs in
    length, shape, and dtype — ``ValueError`` otherwise).
    """
    if example.inputs_fn is not None:
        arrays = example.inputs_fn(seed)
        _validate_inputs_fn_result(example, arrays)
        return arrays
    rng = np.random.default_rng(seed)
    arrays = []
    for index, spec in enumerate(example.specs):
        shape = []
        for dim in spec.shape:
            if not isinstance(dim, (int, np.integer)):
                raise ValueError(
                    f"example {example.name!r}: spec {index} has non-static "
                    f"shape dim {dim!r}; bench examples require static "
                    "integer shapes"
                )
            shape.append(int(dim))
        dtype = np.dtype(spec.dtype)
        if np.issubdtype(dtype, np.floating):
            arrays.append(rng.standard_normal(shape).astype(dtype))
        elif np.issubdtype(dtype, np.integer):
            arrays.append(rng.integers(-5, 6, size=shape, dtype=dtype))
        elif dtype == np.dtype("bool"):
            arrays.append(rng.integers(0, 2, size=shape).astype(dtype))
        else:
            raise ValueError(
                f"example {example.name!r}: unsupported spec dtype {dtype} "
                "for input generation"
            )
    return arrays


def _validate_inputs_fn_result(example: Example, arrays) -> None:
    if not isinstance(arrays, list) or not all(
        isinstance(a, np.ndarray) for a in arrays
    ):
        raise ValueError(
            f"example {example.name!r}: inputs_fn must return a list of "
            "numpy arrays"
        )
    if len(arrays) != len(example.specs):
        raise ValueError(
            f"example {example.name!r}: inputs_fn returned {len(arrays)} "
            f"arrays but {len(example.specs)} specs are declared"
        )
    for index, (spec, array) in enumerate(zip(example.specs, arrays)):
        if tuple(array.shape) != tuple(spec.shape):
            raise ValueError(
                f"example {example.name!r}: inputs_fn array {index} has "
                f"shape {tuple(array.shape)} but spec shape "
                f"{tuple(spec.shape)}"
            )
        if array.dtype != np.dtype(spec.dtype):
            raise ValueError(
                f"example {example.name!r}: inputs_fn array {index} has "
                f"dtype {array.dtype} but spec dtype {np.dtype(spec.dtype)}"
            )


# ---------------------------------------------------------------------------
# Shared reference implementations
# ---------------------------------------------------------------------------


def _conv2d_numpy(x, w, strides=(1, 1), padding="VALID"):
    """Loop-based NCHW 2D convolution reference.

    Mirrors etl's conv semantics exactly: ``"VALID"`` → no padding;
    ``"SAME"`` → TF convention — ``out = ceil(d / stride)`` and total pad
    ``(out - 1) * stride + k - d`` split as ``(total // 2, total - total // 2)``
    per spatial axis (matches ``etl/backends/numpy/kernels/linalg.py``).
    A NEGATIVE total (kernel smaller than the stride-1 footprint, e.g. a 1x1
    kernel at stride 2) CROPS the input — ``np.pad`` with a negative pad value
    crops, exactly like the kernel's slicing.
    """
    n, c_in, h, win = x.shape
    c_out, _, kh, kw = w.shape
    sh, sw = strides
    if padding == "SAME":
        out_h = (h + sh - 1) // sh
        out_w = (win + sw - 1) // sw
        total_h = (out_h - 1) * sh + kh - h
        total_w = (out_w - 1) * sw + kw - win
        pad_h = (total_h // 2, total_h - total_h // 2)
        pad_w = (total_w // 2, total_w - total_w // 2)
    elif padding == "VALID":
        out_h = (h - kh) // sh + 1
        out_w = (win - kw) // sw + 1
        pad_h = pad_w = (0, 0)
    else:
        raise ValueError(f"unsupported padding mode {padding!r}")
    # Clamp to non-negative for np.pad, then crop negative totals (kernel
    # parity: etl/backends/numpy/kernels/linalg.py slices
    # slice(-lo if lo<0 else None, hi if hi<0 else None) after clamping).
    xp = np.pad(
        x,
        ((0, 0), (0, 0), (max(pad_h[0], 0), max(pad_h[1], 0)),
         (max(pad_w[0], 0), max(pad_w[1], 0))),
    )
    lo_h, hi_h = pad_h
    lo_w, hi_w = pad_w
    if lo_h < 0 or hi_h < 0 or lo_w < 0 or hi_w < 0:
        xp = xp[:, :, slice(-lo_h if lo_h < 0 else None, hi_h if hi_h < 0 else None),
                slice(-lo_w if lo_w < 0 else None, hi_w if hi_w < 0 else None)]
    out = np.zeros(
        (n, c_out, out_h, out_w), dtype=np.result_type(x.dtype, w.dtype)
    )
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            out[:, :, i, j] = np.einsum("nchw,fchw->nf", patch, w)
    return out


def conv2d_im2col_numpy(x, w, strides=(1, 1), padding="VALID"):
    """Vectorized NCHW 2D convolution reference (im2col + einsum).

    Same semantics as :func:`_conv2d_numpy`: ``"VALID"`` → no padding;
    ``"SAME"`` → TF convention — ``out = ceil(d / stride)`` and total pad
    ``(out - 1) * stride + k - d`` split as ``(total // 2, total - total // 2)``
    per spatial axis (negative totals crop, exactly like the etl kernel).

    Implementation: ``np.lib.stride_tricks.sliding_window_view`` over the
    padded NCHW array's H/W axes yields windows of shape
    ``(N, C, out_h, out_w, kh, kw)`` (subsampled at ``strides``), then a
    single ``np.einsum`` contraction with the weight
    ``(F, C, kh, kw) -> (N, F, out_h, out_w)``. Verified against
    :func:`_conv2d_numpy` on random inputs (max abs error < 1e-12 in
    float64).
    """
    n, c_in, h, win = x.shape
    c_out, _, kh, kw = w.shape
    sh, sw = strides
    if padding == "SAME":
        out_h = (h + sh - 1) // sh
        out_w = (win + sw - 1) // sw
        total_h = (out_h - 1) * sh + kh - h
        total_w = (out_w - 1) * sw + kw - win
        pad_h = (total_h // 2, total_h - total_h // 2)
        pad_w = (total_w // 2, total_w - total_w // 2)
    elif padding == "VALID":
        out_h = (h - kh) // sh + 1
        out_w = (win - kw) // sw + 1
        pad_h = pad_w = (0, 0)
    else:
        raise ValueError(f"unsupported padding mode {padding!r}")
    # Clamp to non-negative for np.pad, then crop negative totals (kernel
    # parity: etl/backends/numpy/kernels/linalg.py slices
    # slice(-lo if lo<0 else None, hi if hi<0 else None) after clamping).
    xp = np.pad(
        x,
        ((0, 0), (0, 0), (max(pad_h[0], 0), max(pad_h[1], 0)),
         (max(pad_w[0], 0), max(pad_w[1], 0))),
    )
    lo_h, hi_h = pad_h
    lo_w, hi_w = pad_w
    if lo_h < 0 or hi_h < 0 or lo_w < 0 or hi_w < 0:
        xp = xp[:, :, slice(-lo_h if lo_h < 0 else None, hi_h if hi_h < 0 else None),
                slice(-lo_w if lo_w < 0 else None, hi_w if hi_w < 0 else None)]
    # (N, C, out_h, out_w, kh, kw): stride-1 windows subsampled at `strides`.
    windows = np.lib.stride_tricks.sliding_window_view(
        xp, (kh, kw), axis=(2, 3)
    )[..., ::sh, ::sw, :, :]
    return np.einsum("ncpqkl,fckl->nfpq", windows, w)


def softmax_numpy(x):
    """Row-wise softmax over the last axis (max-subtracted, stable).

    Mirrors the etl softmax formula used by the examples (``max`` →
    ``exp`` → normalized ``sum`` over the last axis, keepdims).
    """
    x = np.asarray(x)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def layernorm_numpy(x, eps=1e-5):
    """Layer norm over the last axis (mean/var from sum primitives).

    ``mean = x.mean(axis=-1, keepdims=True)``, ``var = mean((x - mean)**2)``,
    output ``(x - mean) / sqrt(var + eps)`` — the same formula the etl
    layernorm graphs build from ``sum`` primitives.
    """
    x = np.asarray(x)
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    return diff / np.sqrt(var + eps)


def sigmoid_numpy(x):
    """Elementwise logistic sigmoid ``1 / (1 + exp(-x))``."""
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def gelu_numpy(x):
    """etl's ``gelu`` — the EXACT erf form ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    Verified against ``etl/ops/elementwise.py`` and the numpy backend kernel
    (``etl/backends/numpy/kernels/elementwise.py``): etl implements gelu as
    the exact erf form, NOT the tanh approximation
    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))`` (the two
    differ by up to ~1e-3 — far beyond the strict conformance defaults, so
    the reference MUST match the erf form). ``math.erf`` is vectorized via
    ``np.frompyfunc`` with the result cast back to ``x``'s dtype — the same
    technique as the numpy backend kernel (no scipy dependency).
    """
    x = np.asarray(x)
    erf = np.frompyfunc(math.erf, 1, 1)(x / np.sqrt(2.0)).astype(x.dtype)
    return 0.5 * x * (1.0 + erf)


# ---------------------------------------------------------------------------
# Finite-difference gradient helper
# ---------------------------------------------------------------------------


def fd_gradient(value_fn, inputs, h=1e-4, sg_inputs=()):
    """Central-difference gradient of a pure-numpy scalar loss.

    ``value_fn(inputs, frozen) -> float`` is a pure-numpy scalar loss that
    MUST compute in float64 (the helper converts every floating input array
    to float64 before calling). Non-floating inputs (int/bool) are passed
    through unchanged and get a zero float64 gradient array of the same
    shape (they are never perturbed).

    Stop-gradient modeling (``sg_inputs``): freezing input ``i`` while
    perturbing it makes the sg-term independent of ``i`` — the sg-term's
    contribution to ``d/di`` vanishes, exactly like ``etl.stop_gradient``.
    Mechanically: for each perturbed floating input ``i in sg_inputs`` the
    call is ``value_fn(inputs_perturbed, {i: original_float64_copy})``, and
    ``value_fn`` models the stop-gradient term(s) by substituting
    ``frozen.get(i, inputs[i])`` for ``inputs[i]`` wherever the sg'd value
    appears (e.g. ``loss = sum(frozen.get(0, x) * w) + 0.5 * sum(x ** 2)``
    with ``sg_inputs=(0,)`` gives ``d/dx = x`` — the ``x * w`` term frozen).
    When perturbing an input ``j not in sg_inputs``, ``frozen`` is ``{}``
    (empty dict) and is passed through to ``value_fn`` as-is, so the full
    gradient flows.

    Args:
        value_fn: ``(inputs, frozen) -> float`` pure-numpy scalar loss.
        inputs: list of numpy arrays.
        h: central-difference step.
        sg_inputs: iterable of input indices whose stop-gradient terms are
            frozen while their own gradient is computed.

    Returns:
        A list with one float64 gradient array per input (same shapes):
        for each floating input, every element is perturbed in turn and the
        central difference ``(f(x + h) - f(x - h)) / (2h)`` fills the
        corresponding gradient element; non-floating inputs get zeros.
    """
    sg = set(sg_inputs)
    f64 = _float64_inputs(inputs)
    grads = []
    for i, arr in enumerate(inputs):
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            grads.append(np.zeros(arr.shape, dtype=np.float64))
            continue
        base = f64[i]
        frozen = {i: base.copy()} if i in sg else {}
        grad = np.zeros(base.shape, dtype=np.float64)
        for j in range(base.size):
            plus = list(f64)
            minus = list(f64)
            plus[i] = base.copy()
            minus[i] = base.copy()
            plus[i].reshape(-1)[j] = base.reshape(-1)[j] + h
            minus[i].reshape(-1)[j] = base.reshape(-1)[j] - h
            f_plus = value_fn(plus, frozen)
            f_minus = value_fn(minus, frozen)
            grad.reshape(-1)[j] = (f_plus - f_minus) / (2.0 * h)
        grads.append(grad)
    return grads


def _float64_inputs(inputs):
    """Floating inputs as float64 copies; non-floating inputs unchanged."""
    out = []
    for arr in inputs:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            out.append(arr.astype(np.float64))
        else:
            out.append(arr)
    return out
