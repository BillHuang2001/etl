"""Single-rank execution tests for etl.dist collectives.

End-to-end behavior of the default collective executor (installed by the
numpy backend): `SingleRankCollectiveExecutor` with rank 0 / world size 1,
identity semantics — every collective returns its input tensor unchanged.

These tests document the runtime contract (see `../../etl/dist/CONTEXT.md`):

- World-group collectives declare `None` (runtime-dynamic) dims on the
  collective axes, so the identity result passes result-type validation.
- Explicit-group SHAPE-PRESERVING collectives also pass under the identity
  executor (input shape == declared local shape).
- Explicit-group SHAPE-CHANGING collectives declare a different local shape,
  so the identity result violates the declared result type and the
  interpreter raises `core.ShapeError` ("kernel for op ...") — the executor
  is validated against declared local shapes, and simulating real
  multi-rank semantics requires installing a custom executor (covered by
  the sibling `test_executor_hook.py`).
- `rank()` / `world_size()` resolve to scalar int64 graph values (0 / 1)
  from the runtime execution context.

CPU only, no network, small shapes.
"""

import numpy as np
import pytest

import etl
from etl import core

G4 = etl.dist.group("data4", (0, 1, 2, 3))

# (collective frontend, extra kwargs for the 2-D input shape) — group=None
# (the world group) in all cases.
WORLD_CASES = [
    (etl.dist.all_reduce, {}),
    (etl.dist.all_gather, {}),
    (etl.dist.reduce_scatter, {}),
    (etl.dist.all_to_all, {"split_axis": 0, "concat_axis": 1}),
    (etl.dist.broadcast, {}),
    (etl.dist.collective_permute, {"source_target_pairs": ((0, 1), (1, 0))}),
]

DTYPE_SHAPE_CASES = [
    (np.float32, (2, 3)),
    (np.int32, (2, 3)),
    (np.float64, (4, 8)),
]


def _collective_defn(collective, **kwargs):
    """Wrap a collective frontend call in a one-argument `@etl.defn` graph."""

    @etl.defn
    def f(t):
        return collective(t, **kwargs)

    return f


# ---------------------------------------------------------------------------
# 1. World-group (group=None) identity execution across collectives × dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collective,kwargs",
    WORLD_CASES,
    ids=[c.__name__ for c, _ in WORLD_CASES],
)
@pytest.mark.parametrize(
    "dtype,shape",
    DTYPE_SHAPE_CASES,
    ids=[f"{np.dtype(d).name}-{s[0]}x{s[1]}" for d, s in DTYPE_SHAPE_CASES],
)
def test_world_group_identity_execution(collective, kwargs, dtype, shape):
    """Every world-group collective returns its input unchanged on 1 rank."""
    fn = _collective_defn(collective, **kwargs)
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    out = etl.evaluate(fn, x)

    assert isinstance(out, core.Tensor)
    assert out.dtype == core.dtype(dtype)
    assert tuple(out.shape) == shape
    np.testing.assert_array_equal(out.numpy(), x)
    assert isinstance(out.numpy(), np.ndarray)


# ---------------------------------------------------------------------------
# 2. Explicit-group shape-preserving collectives under the identity executor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "collective,kwargs",
    [
        (etl.dist.all_reduce, {}),
        (etl.dist.broadcast, {"src_rank": 0}),
        (
            etl.dist.collective_permute,
            {"source_target_pairs": ((0, 1), (1, 2), (2, 3), (3, 0))},
        ),
    ],
    ids=["all_reduce", "broadcast", "collective_permute"],
)
def test_explicit_group_shape_preserving_identity(collective, kwargs):
    """Shape-preserving collectives on g4 pass validation: output == input."""
    fn = _collective_defn(collective, group=G4, **kwargs)
    x = np.arange(6, dtype=np.float32).reshape(2, 3)

    out = etl.evaluate(fn, x)

    assert isinstance(out, core.Tensor)
    assert out.dtype == core.float32
    assert tuple(out.shape) == (2, 3)
    np.testing.assert_array_equal(out.numpy(), x)


# ---------------------------------------------------------------------------
# 3. Explicit-group shape-changing collectives: declared-shape validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op_name,collective,kwargs,input_shape",
    [
        ("all_gather", etl.dist.all_gather, {}, (2, 3)),
        ("reduce_scatter", etl.dist.reduce_scatter, {}, (8, 3)),
        (
            "all_to_all",
            etl.dist.all_to_all,
            {"split_axis": 0, "concat_axis": 1},
            (4, 4),
        ),
    ],
    ids=["all_gather", "reduce_scatter", "all_to_all"],
)
def test_explicit_group_shape_changing_identity_raises(
    op_name, collective, kwargs, input_shape
):
    """The identity result violates the declared local shape → ShapeError.

    Contract: executor results are validated against the op's declared
    result type. On a single rank the identity executor cannot produce the
    shape-changing local result (e.g. (8, 3) for all_gather on g4), so the
    interpreter must raise rather than silently pass a wrong shape. Real
    multi-rank semantics need a custom simulator executor.
    """
    fn = _collective_defn(collective, group=G4, **kwargs)
    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

    with pytest.raises(core.ShapeError, match=f"kernel for op '{op_name}'"):
        etl.evaluate(fn, x)


# ---------------------------------------------------------------------------
# 4. rank() / world_size() resolve from the single-rank execution context
# ---------------------------------------------------------------------------

def test_rank_and_world_size_single_rank():
    @etl.defn
    def f():
        return (etl.dist.rank(), etl.dist.world_size())

    rank_t, world_t = etl.evaluate(f)

    assert isinstance(rank_t, core.Tensor)
    assert isinstance(world_t, core.Tensor)
    assert rank_t.dtype == core.int64
    assert world_t.dtype == core.int64
    assert tuple(rank_t.shape) == ()
    assert tuple(world_t.shape) == ()
    assert rank_t.numpy() == np.asarray(0, dtype=np.int64)
    assert world_t.numpy() == np.asarray(1, dtype=np.int64)
