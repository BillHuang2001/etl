"""etl.random — key-based functional RNG (graph ops).

Pure, deterministic randomness for etl graphs: the same key + same operands
yield BIT-IDENTICAL values, repeatable across evaluate calls. Keys (rank-0
int64 tensors) are created with :func:`key`, split with :func:`split` /
:func:`split_n`, and threaded explicitly through the sampling ops
(:func:`uniform`, :func:`normal`, :func:`randint`, :func:`permutation`,
:func:`multinomial`) — usable INSIDE traced ``@etl.defn`` graphs (and inside
``cond``/``while_loop``/``scan`` regions, since the ops are pure).

Design decisions (binding — see ``etl/CONTEXT.md``, section "etl.random"):
key representation, SplitMix64 stream derivation with per-op salts,
key-consumption semantics (ops never consume/mutate a key — split for
independence), dtype rules, symbolic-shape and symbolic-``high`` handling,
and the numpy-backend-only v1 coverage (compiler backends reject every
random op with an explicit ``BackendError``). ``etl.random.key`` is a
polymorphic creator (concrete tensor outside a trace, Constant op inside);
every other function is a graph op (``TraceError`` outside a trace).

This module is a thin re-export of the frontend in ``etl.ops.random`` (same
pattern as ``etl.numpy`` over ``etl.ops``).
"""

from etl.ops.random import (  # noqa: F401
    key,
    split,
    split_n,
    uniform,
    normal,
    randint,
    permutation,
    multinomial,
)

__all__ = [
    "key",
    "split",
    "split_n",
    "uniform",
    "normal",
    "randint",
    "permutation",
    "multinomial",
]
