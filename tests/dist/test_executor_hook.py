"""Collective executor hook tests (``etl.dist.context``).

Asserts the "Collective executor hook contract" section of
``etl/dist/CONTEXT.md``:

- the process-global hook slot (default install / set / get / clear /
  protocol validation),
- per-collective argument forwarding from the numpy interpreter kernels
  (``etl/backends/numpy/kernels/collective.py``),
- multi-rank in-process simulation via a custom executor,
- ``rank``/``world_size`` resolution from the per-execution ``RankContext``
  (NOT the executor hook),
- interpreter validation of executor results (dtype/shape) and error
  propagation.

All shapes are small ((2,3) / (8,3)), CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest

import etl
from etl import core
from etl.backends.numpy import SingleRankCollectiveExecutor
from etl.dist.context import CollectiveExecutor

#: Explicit 4-rank group used throughout (group NAME attr: "data").
G4 = etl.dist.group("data", (0, 1, 2, 3))
F32 = np.float32


# ---------------------------------------------------------------------------
# Executors used by the tests
# ---------------------------------------------------------------------------


class RecordingExecutor:
    """Records ``(method_name, tensor, kwargs)`` per call and returns a
    per-test-configured ``result`` (default: the input tensor unchanged —
    identity semantics, like the numpy backend's default)."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def _record(self, name, tensor, **kwargs):
        self.calls.append((name, tensor, kwargs))
        if self.result is not None:
            return self.result
        return tensor

    def all_reduce(self, tensor, group, op):
        return self._record("all_reduce", tensor, group=group, op=op)

    def all_gather(self, tensor, axis, group):
        return self._record("all_gather", tensor, axis=axis, group=group)

    def reduce_scatter(self, tensor, axis, group):
        return self._record("reduce_scatter", tensor, axis=axis, group=group)

    def all_to_all(self, tensor, axis, group):
        return self._record("all_to_all", tensor, axis=axis, group=group)

    def broadcast(self, tensor, src_rank, group):
        return self._record("broadcast", tensor, src_rank=src_rank, group=group)

    def collective_permute(self, tensor, mapping, group):
        return self._record("collective_permute", tensor, mapping=mapping, group=group)


class IdentityExecutor:
    """All six methods return the input unchanged (structurally valid
    executor; subclasses override a single method to test error paths)."""

    def all_reduce(self, tensor, group, op):
        return tensor

    def all_gather(self, tensor, axis, group):
        return tensor

    def reduce_scatter(self, tensor, axis, group):
        return tensor

    def all_to_all(self, tensor, axis, group):
        return tensor

    def broadcast(self, tensor, src_rank, group):
        return tensor

    def collective_permute(self, tensor, mapping, group):
        return tensor


class SimExecutor:
    """Stateful in-process 4-rank simulator.

    Holds every simulated rank's local array (``stack``) and computes each
    collective's post-collective result for the simulated rank (default 0).
    ``all_to_all_result`` is pre-configured: the protocol forwards only
    ``split_axis`` (documented v1 limitation in
    ``etl/backends/numpy/kernels/collective.py``), so the simulator cannot
    know ``concat_axis`` from the protocol arguments alone.
    """

    def __init__(self, stack, all_to_all_result=None, rank=0):
        self.stack = [np.asarray(a) for a in stack]
        self.rank = rank
        self.all_to_all_result = all_to_all_result
        self.calls = []

    def all_reduce(self, tensor, group, op):
        self.calls.append(("all_reduce", group, op))
        # The tests use op="sum"; the simulator always sums.
        return np.sum(np.stack(self.stack), axis=0)

    def all_gather(self, tensor, axis, group):
        self.calls.append(("all_gather", axis, group))
        return np.concatenate([a.copy() for a in self.stack], axis=axis)

    def reduce_scatter(self, tensor, axis, group):
        self.calls.append(("reduce_scatter", axis, group))
        # Protocol has no reduce_op parameter: the simulator always sums
        # (matches the tests' op="sum").
        total = np.sum(np.stack(self.stack), axis=0)
        chunk = total.shape[axis] // len(self.stack)
        index = [slice(None)] * total.ndim
        index[axis] = slice(self.rank * chunk, (self.rank + 1) * chunk)
        return total[tuple(index)]

    def all_to_all(self, tensor, axis, group):
        self.calls.append(("all_to_all", axis, group))
        assert self.all_to_all_result is not None, (
            "pre-configure all_to_all_result (concat_axis is not forwarded "
            "by the v1 protocol)"
        )
        return self.all_to_all_result

    def broadcast(self, tensor, src_rank, group):
        self.calls.append(("broadcast", src_rank, group))
        return self.stack[src_rank].copy()

    def collective_permute(self, tensor, mapping, group):
        self.calls.append(("collective_permute", mapping, group))
        source = next((src for src, dst in mapping if dst == self.rank), None)
        if source is None:
            return np.zeros_like(self.stack[self.rank])
        return self.stack[source].copy()


# ---------------------------------------------------------------------------
# Graph definitions (explicit group G4)
# ---------------------------------------------------------------------------


@etl.defn
def f_all_reduce(x):
    return etl.dist.all_reduce(x, op="sum", group=G4)


@etl.defn
def f_broadcast(x):
    return etl.dist.broadcast(x, group=G4)


@etl.defn
def f_permute(x):
    # Rotation: rank i sends to rank (i+1)%4 — rank 0 receives rank 3's tensor.
    return etl.dist.collective_permute(x, ((0, 1), (1, 2), (2, 3), (3, 0)), group=G4)


@etl.defn
def f_rank_world():
    return (etl.dist.rank(), etl.dist.world_size())


def make_all_gather(axis):
    @etl.defn
    def f(x):
        return etl.dist.all_gather(x, axis=axis, group=G4)

    return f


def make_reduce_scatter(op="sum", axis=0):
    @etl.defn
    def f(x):
        return etl.dist.reduce_scatter(x, op=op, axis=axis, group=G4)

    return f


def make_all_to_all(split_axis, concat_axis):
    @etl.defn
    def f(x):
        return etl.dist.all_to_all(
            x, split_axis=split_axis, concat_axis=concat_axis, group=G4
        )

    return f


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_default_executor():
    """The hook is process-global: restore the numpy backend's default
    executor after every test (in ``finally`` — it must run even on failure)."""
    try:
        yield
    finally:
        etl.dist.set_collective_executor(SingleRankCollectiveExecutor())


def _run_once(recorder, fn, x):
    """Evaluate ``fn(x)``, asserting the recorder saw exactly one call."""
    recorder.calls.clear()
    out = etl.evaluate(fn, x)
    assert len(recorder.calls) == 1
    return out, recorder.calls[0]


def _assert_group_name(kwargs):
    """The kernel forwards the group NAME string, never a Group object."""
    group = kwargs["group"]
    assert isinstance(group, str)
    assert not isinstance(group, etl.dist.Group)
    assert group == "data"


# ---------------------------------------------------------------------------
# 1. Default-installed hook: get/set/clear round trip
# ---------------------------------------------------------------------------


def test_default_executor_installed_and_round_trip():
    # The numpy backend installs its default at import time.
    default = etl.dist.get_collective_executor()
    assert isinstance(default, SingleRankCollectiveExecutor)
    assert isinstance(default, CollectiveExecutor)

    custom = RecordingExecutor()
    etl.dist.set_collective_executor(custom)
    assert etl.dist.get_collective_executor() is custom

    etl.dist.set_collective_executor(None)
    with pytest.raises(core.BackendError, match="no collective executor registered"):
        etl.dist.get_collective_executor()

    with pytest.raises(TypeError, match="CollectiveExecutor"):
        etl.dist.set_collective_executor(object())

    class MissingMethods:
        """Only one of the six protocol methods — must be rejected."""

        def all_reduce(self, tensor, group, op):
            return tensor

    with pytest.raises(TypeError, match="CollectiveExecutor"):
        etl.dist.set_collective_executor(MissingMethods())


# ---------------------------------------------------------------------------
# 2. Hook executed with expected arguments (per collective, group g4)
# ---------------------------------------------------------------------------


def test_all_reduce_arguments():
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.arange(6, dtype=F32).reshape(2, 3)

    out, (name, tensor, kwargs) = _run_once(recorder, f_all_reduce, x)

    assert name == "all_reduce"
    assert isinstance(tensor, core.Tensor)
    np.testing.assert_array_equal(tensor.numpy(), x)
    assert kwargs == {"group": "data", "op": "sum"}
    _assert_group_name(kwargs)
    assert out.shape == (2, 3) and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), x)


@pytest.mark.parametrize("axis_arg,expected_axis", [(0, 0), (1, 1), (-1, 1)])
def test_all_gather_arguments(axis_arg, expected_axis):
    # Negative axes are normalized Python-style at trace time; the executor
    # sees the normalized non-negative int.
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.arange(6, dtype=F32).reshape(2, 3)
    recorder.result = np.concatenate([x] * 4, axis=expected_axis)

    out, (name, tensor, kwargs) = _run_once(recorder, make_all_gather(axis_arg), x)

    assert name == "all_gather"
    assert kwargs == {"axis": expected_axis, "group": "data"}
    _assert_group_name(kwargs)
    expected_shape = (8, 3) if expected_axis == 0 else (2, 12)
    assert out.shape == expected_shape and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), recorder.result)


def test_reduce_scatter_arguments():
    # Input (8,3) declares output (2,3) for the 4-rank group. reduce_op is
    # NOT forwarded (protocol has no reduce_op parameter — documented v1
    # limitation): the executor receives exactly {axis, group}.
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.arange(24, dtype=F32).reshape(8, 3)
    recorder.result = np.sum(np.stack([x] * 4), axis=0)[0:2]  # sum-then-slice

    out, (name, tensor, kwargs) = _run_once(recorder, make_reduce_scatter("max", 0), x)

    assert name == "reduce_scatter"
    assert kwargs == {"axis": 0, "group": "data"}
    _assert_group_name(kwargs)
    assert out.shape == (2, 3) and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), recorder.result)


def test_all_to_all_arguments():
    # Input (8,3), split_axis=0, concat_axis=1 declares (2,12). Only
    # split_axis is forwarded (protocol has a single axis param — documented
    # v1 limitation): the executor receives exactly {axis, group}.
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.arange(24, dtype=F32).reshape(8, 3)
    recorder.result = np.full((2, 12), 3.5, dtype=F32)

    out, (name, tensor, kwargs) = _run_once(recorder, make_all_to_all(0, 1), x)

    assert name == "all_to_all"
    assert kwargs == {"axis": 0, "group": "data"}
    _assert_group_name(kwargs)
    assert out.shape == (2, 12) and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), recorder.result)


def test_broadcast_arguments():
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.ones((2, 3), dtype=F32)

    out, (name, tensor, kwargs) = _run_once(recorder, f_broadcast, x)

    assert name == "broadcast"
    assert kwargs == {"src_rank": 0, "group": "data"}  # 0: registry default
    _assert_group_name(kwargs)
    assert out.shape == (2, 3) and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), x)


def test_collective_permute_arguments():
    recorder = RecordingExecutor()
    etl.dist.set_collective_executor(recorder)
    x = np.ones((2, 3), dtype=F32)

    out, (name, tensor, kwargs) = _run_once(recorder, f_permute, x)

    assert name == "collective_permute"
    assert kwargs["mapping"] == ((0, 1), (1, 2), (2, 3), (3, 0))
    assert isinstance(kwargs["mapping"], tuple)
    assert all(isinstance(pair, tuple) for pair in kwargs["mapping"])
    _assert_group_name(kwargs)
    assert out.shape == (2, 3) and out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), x)


# ---------------------------------------------------------------------------
# 3. Multi-rank simulation (stateful in-process executor)
# ---------------------------------------------------------------------------


def test_multi_rank_all_reduce():
    stack = [np.full((2, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(f_all_reduce, stack[0])

    np.testing.assert_array_equal(out.numpy(), np.sum(np.stack(stack), axis=0))
    assert out.shape == (2, 3) and out.dtype == F32
    assert sim.calls == [("all_reduce", "data", "sum")]


def test_multi_rank_all_gather():
    stack = [np.full((2, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(make_all_gather(0), stack[0])

    np.testing.assert_array_equal(out.numpy(), np.concatenate(stack, axis=0))
    assert out.shape == (8, 3) and out.dtype == F32
    assert sim.calls == [("all_gather", 0, "data")]


def test_multi_rank_reduce_scatter():
    stack = [np.full((8, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(make_reduce_scatter("sum", 0), stack[0])

    expected = np.sum(np.stack(stack), axis=0)[0:2]  # rank 0's chunk
    np.testing.assert_array_equal(out.numpy(), expected)
    assert out.shape == (2, 3) and out.dtype == F32
    assert sim.calls == [("reduce_scatter", 0, "data")]


def test_multi_rank_broadcast():
    stack = [np.full((2, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(f_broadcast, stack[0])

    # src_rank 0's copy (all ones), not the other ranks' arrays.
    np.testing.assert_array_equal(out.numpy(), stack[0])
    assert out.shape == (2, 3) and out.dtype == F32
    assert sim.calls == [("broadcast", 0, "data")]


def test_multi_rank_collective_permute():
    stack = [np.full((2, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(f_permute, stack[0])

    # Rotation mapping: rank 0 receives rank 3's tensor (4s).
    np.testing.assert_array_equal(out.numpy(), stack[3])
    assert out.shape == (2, 3) and out.dtype == F32
    assert sim.calls == [("collective_permute", ((0, 1), (1, 2), (2, 3), (3, 0)), "data")]


def test_multi_rank_collective_permute_no_source():
    # A mapping with no source for rank 0 -> zero tensor of the same shape.
    stack = [np.full((2, 3), r + 1, dtype=F32) for r in range(4)]
    sim = SimExecutor(stack)
    etl.dist.set_collective_executor(sim)

    @etl.defn
    def f_partial(x):
        return etl.dist.collective_permute(x, ((0, 1), (1, 2)), group=G4)

    out = etl.evaluate(f_partial, stack[0])

    np.testing.assert_array_equal(out.numpy(), np.zeros_like(stack[0]))
    assert out.shape == (2, 3) and out.dtype == F32


def test_multi_rank_all_to_all():
    # concat_axis is not forwarded (v1 protocol limitation), so the
    # simulator returns a pre-configured declared-shape result.
    stack = [np.full((8, 3), r + 1, dtype=F32) for r in range(4)]
    configured = np.full((2, 12), 3.5, dtype=F32)
    sim = SimExecutor(stack, all_to_all_result=configured)
    etl.dist.set_collective_executor(sim)

    out = etl.evaluate(make_all_to_all(0, 1), stack[0])

    np.testing.assert_array_equal(out.numpy(), configured)
    assert out.shape == (2, 12) and out.dtype == F32
    assert sim.calls == [("all_to_all", 0, "data")]  # split_axis only


# ---------------------------------------------------------------------------
# 4. rank/world_size resolve from the execution-context RankContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank,world_size", [(1, 2), (3, 4), (0, 1)])
def test_rank_world_size_per_simulated_rank(rank, world_size):
    # etl.run/etl.evaluate take no rank_context — the backend executable's
    # run() accepts the per-execution override.
    exe = etl.build(f_rank_world)
    outputs = exe.backend_executable.run(
        [], rank_context=etl.dist.RankContext(rank, world_size)
    )

    assert isinstance(outputs, list) and len(outputs) == 2
    for tensor in outputs:
        assert isinstance(tensor, core.Tensor)
        assert tensor.dtype == np.int64
        assert tensor.shape == ()
    assert outputs[0].numpy() == rank
    assert outputs[1].numpy() == world_size


def test_rank_world_size_default_context():
    exe = etl.build(f_rank_world)
    outputs = exe.backend_executable.run([])

    assert outputs[0].numpy() == 0
    assert outputs[1].numpy() == 1


# ---------------------------------------------------------------------------
# 5. Error paths
# ---------------------------------------------------------------------------


def test_executor_backend_error_propagates():
    class TransportFail(IdentityExecutor):
        def all_reduce(self, tensor, group, op):
            raise core.BackendError("simulated transport failure")

    etl.dist.set_collective_executor(TransportFail())
    with pytest.raises(core.BackendError, match="simulated transport failure"):
        etl.evaluate(f_all_reduce, np.ones((2, 3), dtype=F32))


def test_executor_wrong_dtype_rejected():
    class WrongDtype(IdentityExecutor):
        def all_reduce(self, tensor, group, op):
            return np.zeros(tensor.shape, dtype=np.int32)

    etl.dist.set_collective_executor(WrongDtype())
    with pytest.raises(core.BackendError, match="dtype"):
        etl.evaluate(f_all_reduce, np.ones((2, 3), dtype=F32))


def test_executor_wrong_shape_rejected():
    class WrongShape(IdentityExecutor):
        def all_reduce(self, tensor, group, op):
            return np.ones((3, 3), dtype=F32)

    etl.dist.set_collective_executor(WrongShape())
    with pytest.raises(core.ShapeError, match="shape"):
        etl.evaluate(f_all_reduce, np.ones((2, 3), dtype=F32))


def test_executor_non_tensor_result_rejected():
    # Neither a Tensor nor an ndarray surfaces the interpreter's validation
    # error (the kernel passes anything else through untouched).
    class NonTensor(IdentityExecutor):
        def all_reduce(self, tensor, group, op):
            return 3.14

    etl.dist.set_collective_executor(NonTensor())
    with pytest.raises(TypeError):
        etl.evaluate(f_all_reduce, np.ones((2, 3), dtype=F32))


# ---------------------------------------------------------------------------
# 6. Raw ndarray returns are tolerated (wrapped in core.Tensor)
# ---------------------------------------------------------------------------


def test_raw_ndarray_result_wrapped_as_tensor():
    class RawNumpy(IdentityExecutor):
        def all_reduce(self, tensor, group, op):
            return tensor.numpy() * 2.0

    etl.dist.set_collective_executor(RawNumpy())
    out = etl.evaluate(f_all_reduce, np.ones((2, 3), dtype=F32))

    assert isinstance(out, core.Tensor)
    assert out.dtype == F32
    np.testing.assert_array_equal(out.numpy(), np.full((2, 3), 2.0, dtype=F32))
