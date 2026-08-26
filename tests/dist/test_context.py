"""Tests for the runtime rank context and ``rank()``/``world_size()``.

Covers the ``etl.dist.context`` contract (see ``etl/dist/CONTEXT.md``):

- ``RankContext``: frozen dataclass validating ``0 <= rank < world_size``
  (ints only, ``bool`` rejected as an int subclass), value equality.
- The thread-local rank-context hook of the numpy reference backend
  (``etl.backends.numpy.exec_context``): default single-process context
  (rank 0 / world size 1), set/get/reset, TypeError on non-RankContext.
- Effect on execution: ``etl.dist.rank()`` / ``etl.dist.world_size()`` are
  scalar int64 graph values resolved from the runtime context at run time
  (never constant-folded) — both through the thread-local hook inside
  ``etl.evaluate`` and through the per-``run`` override
  (``NumpyExecutable.run(..., rank_context=...)``).
- Graph-vs-Python semantics: SymbolicTensor inside a trace, TraceError
  outside a trace.

Thread-local state caveat: the hook is thread-local but lives in a
process-global ``threading.local`` slot, so an autouse fixture resets it
around every test.
"""

import dataclasses

import numpy as np
import pytest

import etl
from etl import core
from etl.backends.numpy.exec_context import get_rank_context, set_rank_context
from etl.dist import RankContext, rank, world_size


@pytest.fixture(autouse=True)
def _clean_rank_context():
    """Start from and restore the default single-process rank context."""
    assert get_rank_context() == RankContext(0, 1)
    try:
        yield
    finally:
        set_rank_context(None)


# ---------------------------------------------------------------------------
# 1. RankContext: validation, frozen immutability, equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rank, bad_world",
    [
        (-1, 1),  # rank negative
        (True, 2),  # rank bool
        (1.5, 2),  # rank non-int
        (None, 1),  # rank non-int
        ("0", 2),  # rank non-int
        (0, 0),  # world_size < 1
        (0, -2),  # world_size < 1
        (0, True),  # world_size bool
        (0, 2.5),  # world_size non-int
        (0, "2"),  # world_size non-int
        (2, 2),  # rank >= world_size
        (5, 3),  # rank >= world_size
    ],
)
def test_rank_context_rejects_invalid_values(bad_rank, bad_world):
    with pytest.raises(ValueError):
        RankContext(bad_rank, bad_world)


@pytest.mark.parametrize(
    "rank_, world",
    [(0, 1), (3, 4)],
)
def test_rank_context_accepts_valid_values(rank_, world):
    ctx = RankContext(rank_, world)
    assert ctx.rank == rank_
    assert ctx.world_size == world


def test_rank_context_is_a_frozen_dataclass():
    field_names = {f.name for f in dataclasses.fields(RankContext)}
    assert field_names == {"rank", "world_size"}
    assert dataclasses.is_dataclass(RankContext)

    ctx = RankContext(0, 1)
    # FrozenInstanceError subclasses AttributeError.
    with pytest.raises(AttributeError):
        ctx.rank = 1
    with pytest.raises(AttributeError):
        ctx.world_size = 8
    assert ctx == RankContext(0, 1)


def test_rank_context_equality():
    assert RankContext(0, 1) == RankContext(0, 1)
    assert RankContext(3, 4) == RankContext(3, 4)
    assert RankContext(0, 1) != RankContext(3, 4)
    assert RankContext(0, 1) != RankContext(0, 2)
    assert RankContext(0, 1) != (0, 1)
    # Frozen dataclass: hashable by value.
    assert hash(RankContext(0, 1)) == hash(RankContext(0, 1))


# ---------------------------------------------------------------------------
# 2. Thread-local rank-context hook: default, set/get/reset, TypeError
# ---------------------------------------------------------------------------


def test_default_rank_context_is_single_process():
    ctx = get_rank_context()
    assert ctx == RankContext(0, 1)
    assert ctx.rank == 0
    assert ctx.world_size == 1


def test_set_get_reset_rank_context_round_trip():
    ctx = RankContext(2, 5)
    set_rank_context(ctx)
    assert get_rank_context() == ctx
    assert get_rank_context() is ctx

    set_rank_context(None)
    assert get_rank_context() == RankContext(0, 1)


@pytest.mark.parametrize("bad", ["nope", 42, (0, 1), object()])
def test_set_rank_context_rejects_non_rank_context(bad):
    with pytest.raises(TypeError):
        set_rank_context(bad)
    # The failed set must not disturb the effective context.
    assert get_rank_context() == RankContext(0, 1)


# ---------------------------------------------------------------------------
# 3. rank()/world_size() resolve from the thread-local context at run time
# ---------------------------------------------------------------------------


@etl.defn
def _rank_world():
    return (etl.dist.rank(), etl.dist.world_size())


def _assert_rank_world_tensor(tensor, expected):
    """A rank/world_size result is a 0-d int64 core.Tensor."""
    assert isinstance(tensor, core.Tensor)
    assert tensor.dtype == core.int64
    assert tensor.shape == ()
    np.testing.assert_array_equal(
        tensor.numpy(), np.asarray(expected, dtype=np.int64)
    )


def test_evaluate_defaults_to_single_process_context():
    out_rank, out_world = etl.evaluate(_rank_world)
    _assert_rank_world_tensor(out_rank, 0)
    _assert_rank_world_tensor(out_world, 1)


@pytest.mark.parametrize(
    "rank_, world",
    [(0, 1), (2, 5), (3, 4)],
)
def test_evaluate_resolves_rank_world_from_thread_local_context(rank_, world):
    set_rank_context(RankContext(rank_, world))
    out_rank, out_world = etl.evaluate(_rank_world)
    _assert_rank_world_tensor(out_rank, rank_)
    _assert_rank_world_tensor(out_world, world)


# ---------------------------------------------------------------------------
# 4. Per-run backend-executable override; thread-local hook unaffected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank_, world",
    [(0, 1), (2, 5), (3, 4)],
)
def test_backend_executable_run_rank_context_override(rank_, world):
    exe = etl.build(_rank_world)
    outputs = exe.backend_executable.run(
        [], rank_context=RankContext(rank_, world)
    )
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    _assert_rank_world_tensor(outputs[0], rank_)
    _assert_rank_world_tensor(outputs[1], world)


def test_run_override_does_not_mutate_thread_local_hook():
    exe = etl.build(_rank_world)
    outputs = exe.backend_executable.run(
        [], rank_context=RankContext(3, 4)
    )
    assert [int(t.numpy()) for t in outputs] == [3, 4]
    # The override is per-run and restores the hook afterwards.
    assert get_rank_context() == RankContext(0, 1)


def test_run_override_takes_precedence_and_restores_thread_local_context():
    set_rank_context(RankContext(2, 5))
    exe = etl.build(_rank_world)
    outputs = exe.backend_executable.run(
        [], rank_context=RankContext(3, 4)
    )
    assert [int(t.numpy()) for t in outputs] == [3, 4]
    # The thread-local context installed before the run is restored.
    assert get_rank_context() == RankContext(2, 5)


# ---------------------------------------------------------------------------
# 5. rank()/world_size() are graph values, not Python values
# ---------------------------------------------------------------------------


def test_rank_world_are_symbolic_tensors_inside_a_trace():
    seen = {}

    @etl.defn
    def probe():
        r = rank()
        w = world_size()
        seen["rank"] = r
        seen["world"] = w
        return (r, w)

    etl.trace(probe)
    for value in (seen["rank"], seen["world"]):
        assert isinstance(value, core.SymbolicTensor)
        assert not isinstance(value, int)
        assert value.dtype == core.int64
        assert value.shape == ()


def test_rank_world_raise_trace_error_outside_a_trace():
    with pytest.raises(core.TraceError):
        rank()
    with pytest.raises(core.TraceError):
        world_size()
