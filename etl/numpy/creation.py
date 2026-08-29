"""Graph creation ops of `etl.numpy` (enp).

These are GRAPH ops: inside `@etl.defn` they build Constant ops (the same
op kind `etl.constant` builds) into the active builder — they do NOT create
concrete Tensors and there is no eager fallback. They embed full arrays as
constants, so the large-constant warning (ETL_LARGE_CONSTANT_BYTES, default
1 MiB) applies. Default dtype is float32 (etl convention; numpy defaults
float64 — deliberate deviation). Implemented: all bodies build Constant ops
via `ops.constant` from concrete, trace-time-materialized numpy data.
"""

from __future__ import annotations

import numpy as np

from .. import core  # dtype constants (float32 default)
from .. import ops  # Constant-op construction path (same as etl.constant)

__all__ = ["zeros", "ones", "full", "empty", "arange"]


def _concrete_shape(shape, fn_name):
    """Normalize a shape spec to a tuple of Python ints.

    Accepts a plain Python int (→ 1-D) or a tuple/list whose elements are
    plain Python ints. Symbolic (`Dim`/`DimExpr`) or runtime-dynamic
    (`None`) elements raise `TraceError`: they require dynamic-length
    constants, which are deferred to v2 — pass concrete Python ints.
    Negative dims are not checked here: numpy raises its own ValueError
    (numpy parity). Other invalid elements/types fall through to numpy,
    which raises its own error (enp never catches).
    """
    if isinstance(shape, int):
        return (shape,)
    if isinstance(shape, (tuple, list)):
        for d in shape:
            if d is None or isinstance(d, (core.Dim, core.DimExpr)):
                raise core.TraceError(
                    f"enp.{fn_name}: symbolic or runtime-dynamic shapes "
                    f"({d!r} in {shape!r}) require dynamic-length constants, "
                    "which are deferred to v2 — pass concrete Python ints"
                )
        return tuple(shape)
    raise core.TraceError(
        f"enp.{fn_name}: shape must be a Python int or a tuple/list of "
        f"Python ints, got {shape!r}"
    )


def zeros(shape, dtype=core.float32):
    """numpy.zeros → Constant op filled with zeros (graph op).

    dtype defaults to float32 (numpy: float64 — documented deviation).
    """
    s = _concrete_shape(shape, "zeros")
    return ops.constant(core.tensor(np.zeros(s, dtype=dtype)))


def ones(shape, dtype=core.float32):
    """numpy.ones → Constant op filled with ones (graph op).

    dtype defaults to float32 (numpy: float64 — documented deviation).
    """
    s = _concrete_shape(shape, "ones")
    return ops.constant(core.tensor(np.ones(s, dtype=dtype)))


def full(shape, fill_value, dtype=None):
    """numpy.full → Constant op filled with fill_value (graph op).

    dtype=None → numpy dtype inference of the static fill_value at trace
    time (fill_value is a Python scalar; it specializes the graph), except an
    inferred float64 becomes float32 (etl default-dtype rule, matching the
    concrete creators in core and TensorSpec; integer fills keep int64).
    """
    s = _concrete_shape(shape, "full")
    if dtype is None:
        inferred = np.result_type(fill_value)
        dtype = np.float32 if inferred == np.dtype("float64") else inferred
    return ops.constant(core.tensor(np.full(s, fill_value, dtype=dtype)))


def empty(shape, dtype=core.float32):
    """numpy.empty → Constant op wrapping an uninitialized array (graph op).

    Values are unspecified (numpy semantics); dtype defaults to float32.
    Warns like any large constant.
    """
    s = _concrete_shape(shape, "empty")
    return ops.constant(core.tensor(np.empty(s, dtype=dtype)))


def arange(start, stop=None, step=1, dtype=None):
    """numpy.arange → Constant op with numpy.arange(start, stop, step) data.

    v1: start/stop/step must be concrete Python numbers at trace time (they
    specialize the graph); symbolic bounds raise TraceError and are deferred
    to v2 (see CONTEXT.md). dtype=None → numpy's own inference over the
    concrete bounds. linspace is deferred to v2.
    """
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if isinstance(value, (core.Dim, core.DimExpr)):
            raise core.TraceError(
                f"enp.arange: symbolic {name}={value!r} is not supported — "
                "bounds must be concrete Python numbers at trace time; "
                "dynamic-length constants (symbolic arange) are deferred to "
                "v2"
            )
    if stop is None:
        data = np.arange(start, dtype=dtype)
    else:
        data = np.arange(start, stop, step, dtype=dtype)
    return ops.constant(core.tensor(data))
