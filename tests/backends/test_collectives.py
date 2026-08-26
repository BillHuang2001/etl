"""Tests for dist collectives dispatched through the numpy interpreter.

Contract under test (see ``etl/dist``, ``etl/backends/numpy/kernels/collective.py``,
``etl/backends/numpy/exec_context.py``):

- All six communication collectives funnel through the process-wide
  ``CollectiveExecutor`` hook (``etl.dist.set_collective_executor`` /
  ``get_collective_executor``) at RUN time — the interpreter itself has no
  collective semantics. The numpy backend installs
  ``SingleRankCollectiveExecutor`` (identity on one rank) at import time.
- ``rank()`` / ``world_size()`` are scalar int64 graph values resolved from
  the per-execution ``RankContext`` — never constant-folded at lower/compile
  time. ``NumpyExecutable.run(flat_inputs, rank_context=...)`` overrides the
  context per run (thread-local, restored afterwards); the
  ``etl.backends.numpy.exec_context.set_rank_context`` thread-local hook is
  the public equivalent for plain ``etl.run``.
- Multi-rank semantics are simulated in-process by installing a custom
  executor implementing the six-method protocol (runtime-checkable — a
  non-conforming object raises TypeError; an unset hook raises BackendError).
- Known v1 protocol limitations (recorded, never silently worked around):
  the executor receives only ``(tensor, axis, group)`` for ``reduce_scatter``
  (no reduce op), only ``split_axis`` for ``all_to_all``, and the group NAME
  STRING (not a Group object) for every collective.
"""

import dataclasses

import numpy as np
import pytest

import etl


# ---------------------------------------------------------------------------
# Fixtures: the executor hook is process-wide — always restore on teardown.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_executor_hook():
    previous = etl.dist.get_collective_executor()
    yield
    etl.dist.set_collective_executor(previous)


class _RecordingTwoRankExecutor:
    """Simulate a 2-rank world: the local tensor plus a synthetic peer tensor.

    The peer is ``local + 10``, so expected outputs are deterministic. Every
    method records its call arguments (including unexpected kwargs) so tests
    can assert exactly what the interpreter kernels forward through the hook.
    """

    def __init__(self):
        self.calls = []

    def _peer(self, tensor):
        return tensor.numpy() + 10.0

    def all_reduce(self, tensor, group=None, op=None):
        self.calls.append(("all_reduce", group, op))
        return tensor.numpy() + self._peer(tensor)

    def all_gather(self, tensor, axis=0, group=None):
        self.calls.append(("all_gather", axis, group))
        return np.concatenate([tensor.numpy(), self._peer(tensor)], axis=axis)

    def reduce_scatter(self, tensor, axis=0, group=None, **extra):
        self.calls.append(("reduce_scatter", axis, group, extra))
        return tensor.numpy()  # identity-ish: no reduce op in the protocol

    def all_to_all(self, tensor, axis=0, group=None, **extra):
        self.calls.append(("all_to_all", axis, group, extra))
        return tensor.numpy()  # identity-ish

    def broadcast(self, tensor, src_rank=None, group=None):
        self.calls.append(("broadcast", src_rank, group))
        return tensor.numpy()

    def collective_permute(self, tensor, mapping=None, group=None):
        self.calls.append(("collective_permute", mapping, group))
        return tensor.numpy()


@pytest.fixture
def fake_executor():
    """Install the recording 2-rank simulator; restore the previous executor."""
    fake = _RecordingTwoRankExecutor()
    etl.dist.set_collective_executor(fake)
    return fake


def _arange_2x3():
    return np.arange(6, dtype=np.float32).reshape(2, 3)


# ---------------------------------------------------------------------------
# 1. Default (single-rank) executor: every collective is the identity.
# ---------------------------------------------------------------------------

_IDENTITY_COLLECTIVES = [
    ("all_reduce sum", lambda t: etl.dist.all_reduce(t, op="sum")),
    ("all_reduce max", lambda t: etl.dist.all_reduce(t, op="max")),
    ("all_gather", lambda t: etl.dist.all_gather(t, axis=0)),
    ("reduce_scatter", lambda t: etl.dist.reduce_scatter(t, op="sum", axis=0)),
    ("all_to_all", lambda t: etl.dist.all_to_all(t, split_axis=0, concat_axis=1)),
    ("broadcast", lambda t: etl.dist.broadcast(t, src_rank=0)),
    ("collective_permute", lambda t: etl.dist.collective_permute(t, ((0, 1),))),
]


@pytest.mark.parametrize(
    "name, collective",
    _IDENTITY_COLLECTIVES,
    ids=[name for name, _ in _IDENTITY_COLLECTIVES],
)
def test_default_executor_is_identity(name, collective):
    x = _arange_2x3()
    out = etl.evaluate(collective, etl.from_numpy(x.copy()))
    assert isinstance(out, etl.Tensor)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out.numpy(), x)


# ---------------------------------------------------------------------------
# 2. rank() / world_size() graph scalars (default context: rank 0 of 1).
# ---------------------------------------------------------------------------


def test_rank_and_world_size_defaults():
    def fn():
        return etl.dist.rank(), etl.dist.world_size()

    executable = etl.build(fn)
    rank, world_size = etl.run(executable)
    assert isinstance(rank, etl.Tensor)
    assert isinstance(world_size, etl.Tensor)
    assert rank.dtype == np.int64 and rank.shape == ()
    assert world_size.dtype == np.int64 and world_size.shape == ()
    assert rank.numpy().item() == 0
    assert world_size.numpy().item() == 1


# ---------------------------------------------------------------------------
# 3. Multi-rank simulation via the collective-executor hook.
# ---------------------------------------------------------------------------


def test_multi_rank_all_reduce_sums_across_ranks(fake_executor):
    def fn(x):
        return etl.dist.all_reduce(x, op="sum")

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    # local (rank 0) + synthetic peer (rank 1: x + 10) => 2x + 10
    np.testing.assert_array_equal(out.numpy(), 2 * x + 10)
    assert fake_executor.calls == [("all_reduce", "world", "sum")]


def test_multi_rank_all_gather_concatenates_along_axis(fake_executor):
    def fn(x):
        return etl.dist.all_gather(x, axis=0)

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    expected = np.concatenate([x, x + 10], axis=0)
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out.numpy(), expected)
    assert fake_executor.calls == [("all_gather", 0, "world")]


def test_multi_rank_reduce_scatter_protocol_has_no_reduce_op(fake_executor):
    def fn(x):
        return etl.dist.reduce_scatter(x, op="sum", axis=0)

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    np.testing.assert_array_equal(out.numpy(), x)
    name, axis, group, extra = fake_executor.calls[0]
    assert (name, axis, group) == ("reduce_scatter", 0, "world")
    # v1 protocol limitation: the executor gets (tensor, axis, group) only —
    # the op's reduce_op is NOT forwarded.
    assert extra == {}


def test_multi_rank_all_to_all_forwards_only_split_axis(fake_executor):
    def fn(x):
        return etl.dist.all_to_all(x, split_axis=0, concat_axis=1)

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    np.testing.assert_array_equal(out.numpy(), x)
    name, axis, group, extra = fake_executor.calls[0]
    assert (name, axis, group) == ("all_to_all", 0, "world")
    # v1 protocol limitation: a SINGLE axis is forwarded (the op's
    # split_axis); concat_axis is not.
    assert extra == {}


def test_multi_rank_broadcast_forwards_src_rank(fake_executor):
    def fn(x):
        return etl.dist.broadcast(x, src_rank=1)

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    np.testing.assert_array_equal(out.numpy(), x)
    name, src_rank, group = fake_executor.calls[0]
    assert (name, group) == ("broadcast", "world")
    # The op's declared src_rank attr (builder default 0) is what the kernel
    # forwards: dist validates the user's src_rank but does not record it on
    # the op (documented ir-side gap in etl/dist/collectives.py), so the
    # executor always sees the default 0 for the world group.
    assert src_rank == 0


def test_multi_rank_collective_permute_forwards_mapping(fake_executor):
    def fn(x):
        return etl.dist.collective_permute(x, ((0, 1),))

    x = _arange_2x3()
    out = etl.evaluate(fn, etl.from_numpy(x.copy()))
    np.testing.assert_array_equal(out.numpy(), x)
    name, mapping, group = fake_executor.calls[0]
    assert (name, group) == ("collective_permute", "world")
    assert mapping == ((0, 1),)


# ---------------------------------------------------------------------------
# 4. Per-execution RankContext overrides (rank/world_size graph scalars).
# ---------------------------------------------------------------------------


def test_rank_context_override_on_backend_run():
    def fn(x):
        return etl.dist.rank(), etl.dist.world_size(), x

    executable = etl.build(fn, etl.TensorSpec((2, 3), etl.float32))
    backend_executable = executable.backend_executable

    x = etl.tensor(_arange_2x3())
    rank, world_size, y = etl.run(executable, x)  # baseline
    assert rank.numpy().item() == 0
    assert world_size.numpy().item() == 1

    outputs = backend_executable.run(
        [x], rank_context=etl.dist.RankContext(rank=1, world_size=2)
    )
    assert outputs[0].numpy().item() == 1
    assert outputs[1].numpy().item() == 2
    np.testing.assert_array_equal(outputs[2].numpy(), x.numpy())

    # The override is per-run: the next plain run sees the default again.
    rank, world_size, y = etl.run(executable, x)
    assert rank.numpy().item() == 0
    assert world_size.numpy().item() == 1


def test_thread_local_rank_context():
    from etl.backends.numpy.exec_context import get_rank_context, set_rank_context

    def fn(x):
        return etl.dist.rank(), etl.dist.world_size(), x

    executable = etl.build(fn, etl.TensorSpec((2, 3), etl.float32))
    x = etl.tensor(_arange_2x3())

    set_rank_context(etl.dist.RankContext(rank=1, world_size=2))
    try:
        rank, world_size, y = etl.run(executable, x)
        assert rank.numpy().item() == 1
        assert world_size.numpy().item() == 2
        np.testing.assert_array_equal(y.numpy(), x.numpy())
    finally:
        set_rank_context(None)
    assert get_rank_context() == etl.dist.RankContext(rank=0, world_size=1)


# ---------------------------------------------------------------------------
# 5. The executor hook itself: set / get / reset / protocol check.
# ---------------------------------------------------------------------------


def test_executor_hook_set_get_reset():
    default = etl.dist.get_collective_executor()
    assert etl.dist.get_collective_executor() is default

    fake = _RecordingTwoRankExecutor()
    etl.dist.set_collective_executor(fake)
    assert etl.dist.get_collective_executor() is fake

    etl.dist.set_collective_executor(None)
    with pytest.raises(etl.BackendError):
        etl.dist.get_collective_executor()
    # (the autouse fixture restores the default executor for the next test)


def test_executor_hook_rejects_non_conforming_object():
    class Partial:
        def all_reduce(self, tensor, group=None, op=None):
            return tensor

    with pytest.raises(TypeError):
        etl.dist.set_collective_executor(Partial())


# ---------------------------------------------------------------------------
# 6. RankContext validation (frozen dataclass, 0 <= rank < world_size).
# ---------------------------------------------------------------------------


def test_rank_context_valid_and_frozen():
    ctx = etl.dist.RankContext(rank=1, world_size=2)
    assert ctx.rank == 1
    assert ctx.world_size == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.rank = 0


@pytest.mark.parametrize(
    "rank, world_size", [(-1, 2), (2, 2), (0, 0)]
)
def test_rank_context_validation(rank, world_size):
    with pytest.raises(ValueError):
        etl.dist.RankContext(rank=rank, world_size=world_size)
