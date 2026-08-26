"""Graph creation ops of `etl.numpy` (enp).

These are GRAPH ops: inside `@etl.defn` they build Constant ops (the same
op kind `etl.constant` builds) into the active builder — they do NOT create
concrete Tensors and there is no eager fallback. They embed full arrays as
constants, so the large-constant warning (ETL_LARGE_CONSTANT_BYTES, default
1 MiB) applies. Default dtype is float32 (etl convention; numpy defaults
float64 — deliberate deviation). Architecture phase: stubs raise
NotImplementedError.
"""

from __future__ import annotations

from .. import core  # dtype constants (float32 default)
from .. import ops  # Constant-op construction path (same as etl.constant)

__all__ = ["zeros", "ones", "full", "empty", "arange"]


def zeros(shape, dtype=core.float32):
    """numpy.zeros → Constant op filled with zeros (graph op).

    dtype defaults to float32 (numpy: float64 — documented deviation).
    """
    raise NotImplementedError(
        "enp.zeros: architecture stub — builds a zeros-filled Constant op"
    )


def ones(shape, dtype=core.float32):
    """numpy.ones → Constant op filled with ones (graph op).

    dtype defaults to float32 (numpy: float64 — documented deviation).
    """
    raise NotImplementedError(
        "enp.ones: architecture stub — builds a ones-filled Constant op"
    )


def full(shape, fill_value, dtype=None):
    """numpy.full → Constant op filled with fill_value (graph op).

    dtype=None → numpy dtype inference of the static fill_value at trace
    time (fill_value is a Python scalar; it specializes the graph).
    """
    raise NotImplementedError(
        "enp.full: architecture stub — builds a Constant op filled with "
        "fill_value (dtype=None infers from fill_value)"
    )


def empty(shape, dtype=core.float32):
    """numpy.empty → Constant op wrapping an uninitialized array (graph op).

    Values are unspecified (numpy semantics); dtype defaults to float32.
    Warns like any large constant.
    """
    raise NotImplementedError(
        "enp.empty: architecture stub — builds a Constant op wrapping an "
        "uninitialized array"
    )


def arange(start, stop=None, step=1, dtype=None):
    """numpy.arange → Constant op with numpy.arange(start, stop, step) data.

    v1: start/stop/step must be concrete Python numbers at trace time (they
    specialize the graph); symbolic bounds raise TraceError and are deferred
    to v2 (see CONTEXT.md). dtype=None → numpy's own inference over the
    concrete bounds. linspace is deferred to v2.
    """
    raise NotImplementedError(
        "enp.arange: architecture stub — builds a Constant op from "
        "numpy.arange with concrete trace-time bounds"
    )
