"""etl.random — key-based functional RNG graph-op tests.

The contract under test (binding — see ``etl/ops/random.py`` and
``etl/CONTEXT.md`` section "etl.random"):

- Keys are rank-0 int64 tensors holding a 64-bit state. Same key + same
  operands ⇒ BIT-IDENTICAL values, repeatable across separate evaluate runs.
- Ops NEVER consume/mutate a key; ``split``/``split_n`` derive decorrelated
  keys (SplitMix64 with salts). Different op kinds sharing one key draw
  decorrelated streams (fixed per-op salts).
- dtype rules: uniform/normal → floating (default float32); randint/
  permutation → integer (default int32); multinomial output int32.
- ``shape`` accepts int / Dim / DimExpr (runtime-dynamic None rejected);
  low/high/mean/std may be Python scalars or symbolic tensors broadcastable
  against shape; the result shape is the broadcast.
- ``key()`` is a polymorphic creator (concrete Tensor outside a trace,
  Constant op inside); every other function is a graph op (TraceError
  outside a trace; concrete Tensor operands raise TraceError).
- Backends: numpy interpreter (all ops); StableHLO export is v1 for every
  random op EXCEPT ``multinomial`` (the cumulative-search decomposition is
  not wired in v1 — compiler backends reject it with an explicit
  BackendError naming ``random_multinomial``); transforms have no rules for
  random ops → TransformError.
"""
import numpy as np
import pytest

import etl
from etl import core
import etl.numpy as enp

from tests.ops.conftest import ops_of, run_numpy

# --- helpers -----------------------------------------------------------------


def _i64(v):
    """A static int64 scalar constant (the verified while_loop pattern)."""
    return etl.constant(etl.tensor(np.array(v, dtype=np.int64)))


def _exec(fn, *specs):
    """Explicit pipeline (trace → lower → compile → load → run) for graphs
    with symbolic dims, which ``etl.evaluate`` cannot stage (it derives fully
    static specs from concrete args)."""
    graph = etl.trace(fn, *specs)
    return etl.load(etl.compile(etl.lower(graph)))


# --- 1. determinism: same key ⇒ bit-identical draws -------------------------


@pytest.mark.parametrize(
    "op_name",
    ["uniform", "normal", "randint", "permutation", "multinomial", "split"],
)
def test_cross_evaluate_determinism(op_name):
    """Two separate evaluate calls with the same key give bit-identical
    draws for every op kind (split: identical derived key pair)."""
    seed = np.array(123, dtype=np.int64)

    def build():
        @etl.defn
        def f(key):
            if op_name == "uniform":
                return etl.random.uniform(key, (5, 3))
            if op_name == "normal":
                return etl.random.normal(key, (5, 3))
            if op_name == "randint":
                return etl.random.randint(key, (7,), 0, 10)
            if op_name == "permutation":
                return etl.random.permutation(key, 6)
            if op_name == "multinomial":
                probs = np.array([0.25, 0.25, 0.5], dtype=np.float32)
                return etl.random.multinomial(key, etl.constant(etl.tensor(probs)), 20)
            a, b = etl.random.split(key)
            return a, b

        return f

    f = build()
    r1 = run_numpy(f, seed)
    r2 = run_numpy(f, seed)
    if isinstance(r1, tuple):
        assert all(np.array_equal(a, b) for a, b in zip(r1, r2))
    else:
        assert np.array_equal(r1, r2)


def test_determinism_across_staging_paths():
    """The explicit pipeline and etl.evaluate produce the same draws."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (4, 2))

    seed = np.array(7, dtype=np.int64)
    via_eval = run_numpy(f, seed)
    exe = _exec(f, etl.TensorSpec((), etl.int64))
    via_pipeline = etl.run(exe, etl.random.key(7)).numpy()
    assert np.array_equal(via_eval, via_pipeline)


def test_different_keys_differ():
    """Different keys ⇒ different draws (overwhelmingly likely)."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (8,))

    a = run_numpy(f, np.array(1, dtype=np.int64))
    b = run_numpy(f, np.array(2, dtype=np.int64))
    assert not np.array_equal(a, b)


def test_op_kinds_share_one_key_draw_decorrelated_streams():
    """Two op kinds sharing one key draw from decorrelated streams (per-op
    salts): the uniform and normal outputs for the same key must differ."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (8,)), etl.random.normal(key, (8,))

    u, n = run_numpy(f, np.array(1, dtype=np.int64))
    assert not np.array_equal(u, n)


# --- 2. split / split_n semantics -------------------------------------------


def test_split_reproducible_and_distinct():
    """split(key) is reproducible; the two derived keys differ from each
    other and from the parent key."""

    @etl.defn
    def f(key):
        a, b = etl.random.split(key)
        return a, b

    seed = np.array(99, dtype=np.int64)
    a1, b1 = run_numpy(f, seed)
    a2, b2 = run_numpy(f, seed)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, b1)  # derived keys are decorrelated
    assert a1.item() != 99 and b1.item() != 99  # distinct from the parent


def test_split_n_distinct_and_deterministic():
    """split_n(key, n) yields n pairwise-distinct keys, deterministically."""

    @etl.defn
    def f(key):
        return etl.random.split_n(key, 5)

    keys = run_numpy(f, np.array(99, dtype=np.int64))
    assert len(keys) == 5
    values = [k.item() for k in keys]
    assert len(set(values)) == 5  # pairwise distinct
    keys2 = run_numpy(f, np.array(99, dtype=np.int64))
    assert all(np.array_equal(a, b) for a, b in zip(keys, keys2))


def test_split_n_two_matches_split():
    """split_n(key, 2) ≡ split(key) (both are random_key_mix sugar)."""

    @etl.defn
    def f(key):
        return etl.random.split(key)

    @etl.defn
    def g(key):
        return etl.random.split_n(key, 2)

    seed = np.array(5, dtype=np.int64)
    a, b = run_numpy(f, seed)
    c, d = run_numpy(g, seed)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_split_derived_keys_drive_independent_draws():
    """Samples drawn from split-derived keys are decorrelated: two uniform
    draws from split(key) differ from two draws sharing one key's stream."""

    @etl.defn
    def f(key):
        k1, k2 = etl.random.split(key)
        return etl.random.uniform(k1, (6,)), etl.random.uniform(k2, (6,))

    u1, u2 = run_numpy(f, np.array(42, dtype=np.int64))
    assert not np.array_equal(u1, u2)


# --- 3. distribution sanity (loose tolerances) ------------------------------


def test_uniform_range_and_mean():
    @etl.defn
    def f(key):
        return etl.random.uniform(key, (10000,), 0.0, 1.0)

    u = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.shape == (10000,)
    assert u.dtype == np.float32
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.05
    assert abs(u.std() - 1.0 / np.sqrt(12.0)) < 0.05


def test_uniform_range_custom_low_high():
    @etl.defn
    def f(key):
        return etl.random.uniform(key, (10000,), 2.0, 5.0)

    u = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.min() >= 2.0 and u.max() < 5.0
    assert abs(u.mean() - 3.5) < 0.05


def test_normal_mean_and_std():
    @etl.defn
    def f(key):
        return etl.random.normal(key, (10000,), 0.0, 1.0)

    z = run_numpy(f, np.array(1, dtype=np.int64))
    assert z.dtype == np.float32
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.1


def test_normal_affine_transform():
    @etl.defn
    def f(key):
        return etl.random.normal(key, (10000,), 3.0, 2.0)

    z = run_numpy(f, np.array(1, dtype=np.int64))
    assert abs(z.mean() - 3.0) < 0.1
    assert abs(z.std() - 2.0) < 0.2


def test_randint_range():
    @etl.defn
    def f(key):
        return etl.random.randint(key, (20000,), 0, 10)

    r = run_numpy(f, np.array(1, dtype=np.int64))
    assert r.dtype == np.int32
    assert r.min() >= 0 and r.max() <= 9  # high exclusive
    # all values in the range appear
    assert set(np.unique(r)) == set(range(10))
    assert abs(r.mean() - 4.5) < 0.2


def test_multinomial_indices_in_range():
    @etl.defn
    def f(key, probs):
        return etl.random.multinomial(key, probs, 10000)

    p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    idx = run_numpy(f, np.array(1, dtype=np.int64), p)
    assert idx.dtype == np.int32
    assert set(np.unique(idx)) <= {0, 1, 2}
    # the 0.7 mass dominates the empirical frequency
    assert (idx == 2).mean() > 0.6


def test_multinomial_zero_samples():
    @etl.defn
    def f(key, probs):
        return etl.random.multinomial(key, probs, 0)

    idx = run_numpy(f, np.array(1, dtype=np.int64), np.array([0.5, 0.5], np.float32))
    assert idx.shape == (0,)


# --- 4. permutation ----------------------------------------------------------


def test_permutation_static_n_is_a_shuffle():
    @etl.defn
    def f(key):
        return etl.random.permutation(key, 7)

    perm = run_numpy(f, np.array(1, dtype=np.int64))
    assert perm.dtype == np.int32
    assert perm.shape == (7,)
    assert np.array_equal(np.sort(perm), np.arange(7))


def test_permutation_symbolic_n():
    """n as a symbolic rank-0 tensor input: the result is a shuffle of
    0..n-1 with runtime-dynamic length."""

    @etl.defn
    def f(key, n):
        return etl.random.permutation(key, n)

    perm = run_numpy(f, np.array(1, dtype=np.int64), np.array(6, dtype=np.int64))
    assert perm.shape == (6,)
    assert np.array_equal(np.sort(perm), np.arange(6))


def test_permutation_dtype_and_zero():
    @etl.defn
    def f(key):
        return etl.random.permutation(key, 4, dtype=etl.int64)

    perm = run_numpy(f, np.array(1, dtype=np.int64))
    assert perm.dtype == np.int64
    assert np.array_equal(np.sort(perm), np.arange(4))

    @etl.defn
    def g(key):
        return etl.random.permutation(key, 0)

    empty = run_numpy(g, np.array(1, dtype=np.int64))
    assert empty.shape == (0,)


def test_permutation_static_n_traced_shape_is_static():
    """A static Python int n traces to a STATIC (n,) output shape — so
    etl.cond branch unification works for static-size populations; a
    symbolic rank-0 n input stays runtime-dynamic (None,)."""

    @etl.defn
    def f_static(key):
        return etl.random.permutation(key, 7)

    static_spec = etl.lower(
        etl.trace(f_static, etl.TensorSpec((), "int64"))
    ).signature.output_specs[0]
    assert static_spec.shape == (7,)

    @etl.defn
    def f_symbolic(key, n):
        return etl.random.permutation(key, n)

    dynamic_spec = etl.lower(
        etl.trace(f_symbolic, etl.TensorSpec((), "int64"), etl.TensorSpec((), "int64"))
    ).signature.output_specs[0]
    assert dynamic_spec.shape == (None,)


# --- 5. symbolic operands ----------------------------------------------------


def test_uniform_symbolic_high_columnwise():
    """uniform(key, (num_sample, m), 0.0, bound) with bound an explicit
    graph input of shape (m,): values within [0, bound) column-wise and the
    broadcast result shape (num_sample, m)."""
    m = core.Dim("m")

    @etl.defn
    def f(key, bound):
        return etl.random.uniform(key, (50, m), 0.0, bound)

    exe = _exec(f, etl.TensorSpec((), etl.int64), etl.TensorSpec((m,), etl.float32))
    bound = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = etl.run(exe, etl.random.key(1), bound)
    assert out.shape == (50, 3)
    assert out.dtype == np.float32
    assert (out.numpy() >= 0.0).all()
    assert (out.numpy() < bound[None, :]).all()


def test_uniform_symbolic_low_scalar():
    """A rank-0 symbolic low operand is broadcast against shape."""

    @etl.defn
    def f(key, lo):
        return etl.random.uniform(key, (4,), lo, 1.0)

    out = run_numpy(f, np.array(1, dtype=np.int64), np.array(0.25, dtype=np.float32))
    assert (out >= 0.25).all() and (out < 1.0).all()


def test_randint_symbolic_low_high():
    """randint with symbolic low/high operands: values in [low, high)."""

    @etl.defn
    def f(key, lo, hi):
        return etl.random.randint(key, (1000,), lo, hi)

    out = run_numpy(
        f, np.array(2, dtype=np.int64), np.array(0, dtype=np.int64),
        np.array(10, dtype=np.int64),
    )
    assert out.min() >= 0 and out.max() <= 9


def test_randint_symbolic_high_columnwise():
    """randint(key, (num, m), 0, highs) with highs of shape (m,): values in
    [0, highs) column-wise (broadcast result shape)."""
    m = core.Dim("m")

    @etl.defn
    def f(key, highs):
        return etl.random.randint(key, (1000, m), 0, highs)

    exe = _exec(f, etl.TensorSpec((), etl.int64), etl.TensorSpec((m,), etl.int32))
    highs = np.array([2, 5], dtype=np.int32)
    out = etl.run(exe, etl.random.key(4), highs)
    v = out.numpy()
    assert v.shape == (1000, 2)
    assert (v[:, 0] < 2).all() and (v[:, 1] < 5).all()
    assert (v >= 0).all()


# --- 6. dtype rules ----------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "spec"),
    [
        (lambda key: etl.random.uniform(key, (2,), dtype=etl.int32),
         etl.TensorSpec((), etl.int64)),
        (lambda key: etl.random.normal(key, (2,), dtype=etl.int64),
         etl.TensorSpec((), etl.int64)),
    ],
    ids=["uniform-int", "normal-int"],
)
def test_float_ops_reject_integer_dtype(fn, spec):
    with pytest.raises(core.DTypeError, match="floating dtype"):
        etl.trace(fn, spec)


@pytest.mark.parametrize(
    ("fn", "spec"),
    [
        (lambda key: etl.random.randint(key, (2,), 0, 5, dtype=etl.float32),
         etl.TensorSpec((), etl.int64)),
        (lambda key: etl.random.permutation(key, 4, dtype=etl.float32),
         etl.TensorSpec((), etl.int64)),
    ],
    ids=["randint-float", "permutation-float"],
)
def test_int_ops_reject_float_dtype(fn, spec):
    with pytest.raises(core.DTypeError, match="integer dtype"):
        etl.trace(fn, spec)


def test_multinomial_input_must_be_float():
    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, 3)

    with pytest.raises(core.DTypeError, match="floating dtype"):
        etl.evaluate(f, etl.random.key(1), np.array([1, 1], dtype=np.int32))


def test_default_output_dtypes():
    @etl.defn
    def f(key, probs):
        return (
            etl.random.uniform(key, (2,)),
            etl.random.randint(key, (2,), 0, 3),
            etl.random.permutation(key, 3),
            etl.random.multinomial(key, probs, 3),
        )

    u, r, p, mn = run_numpy(
        f, np.array(1, dtype=np.int64), np.array([0.5, 0.5], np.float32)
    )
    assert u.dtype == np.float32
    assert r.dtype == np.int32
    assert p.dtype == np.int32
    assert mn.dtype == np.int32


def test_explicit_dtypes_respected():
    @etl.defn
    def f(key):
        return (
            etl.random.uniform(key, (2,), dtype=etl.float64),
            etl.random.permutation(key, 4, dtype=etl.int64),
        )

    u, p = run_numpy(f, np.array(1, dtype=np.int64))
    assert u.dtype == np.float64
    assert p.dtype == np.int64


# --- 7. explicitness ---------------------------------------------------------


def test_random_ops_raise_traceerror_outside_trace():
    key = etl.random.key(1)
    calls = [
        lambda: etl.random.uniform(key, (2,)),
        lambda: etl.random.normal(key, (2,)),
        lambda: etl.random.randint(key, (2,), 0, 5),
        lambda: etl.random.permutation(key, 4),
        lambda: etl.random.split(key),
        lambda: etl.random.split_n(key, 2),
    ]
    for call in calls:
        with pytest.raises(core.TraceError):
            call()


def test_multinomial_outside_trace():
    with pytest.raises(core.TraceError):
        etl.random.multinomial(
            etl.random.key(1), np.array([0.5, 0.5], dtype=np.float32), 3
        )


def test_concrete_tensor_key_inside_trace_raises():
    with pytest.raises(core.TraceError, match="concrete Tensor key"):
        etl.trace(
            lambda: etl.random.uniform(etl.tensor(np.array(1, np.int64)), (2,))
        )


def test_concrete_tensor_operand_inside_trace_raises():
    @etl.defn
    def f(key):
        return etl.random.uniform(key, (2,), etl.tensor(np.array(0.0)), 1.0)

    with pytest.raises(core.TraceError, match="concrete Tensor operand"):
        etl.evaluate(f, etl.random.key(1))


def test_key_outside_trace_is_concrete_tensor():
    k = etl.random.key(123)
    assert isinstance(k, core.Tensor)
    assert k.dtype == np.dtype("int64")
    assert k.shape == ()
    assert k.numpy().item() == 123


def test_key_masks_seed_to_64_bits():
    # two's complement 64-bit masking: negative seeds round-trip, values
    # beyond 2^64 wrap, 2^63 becomes the most negative int64.
    assert etl.random.key(-1).numpy().item() == -1
    assert etl.random.key(2**64 + 5).numpy().item() == 5
    assert etl.random.key(2**63).numpy().item() == -(2**63)


def test_key_rejects_non_int_seed():
    with pytest.raises(TypeError, match="seed must be a Python int"):
        etl.random.key(1.5)
    with pytest.raises(TypeError, match="seed must be a Python int"):
        etl.random.key("1")


def test_key_inside_trace_builds_constant():
    @etl.defn
    def f():
        return etl.random.key(42)

    graph = etl.trace(f)
    assert [op.name for op in ops_of(graph)] == ["constant", "return"]
    assert etl.evaluate(f).numpy().item() == 42


def test_concrete_key_usable_as_explicit_graph_input():
    """key(seed) outside a trace returns a concrete Tensor that can be
    passed as an explicit graph input and run through the pipeline."""
    key = etl.random.key(123)

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (3,))

    exe = _exec(f, etl.TensorSpec((), etl.int64))
    out1 = etl.run(exe, key).numpy()
    out2 = etl.run(exe, etl.random.key(123)).numpy()
    assert np.array_equal(out1, out2)


def test_key_spec_must_be_rank0_int64():
    for spec in (etl.TensorSpec((), etl.int32), etl.TensorSpec((2,), etl.int64)):
        with pytest.raises(core.ShapeError, match="rank-0 int64 tensor"):
            etl.trace(lambda key: etl.random.uniform(key, (2,)), spec)


def test_randint_operands_must_be_int():
    with pytest.raises(TypeError, match="random.randint low"):
        etl.trace(
            lambda key: etl.random.randint(key, (2,), 0.5, 2),
            etl.TensorSpec((), etl.int64),
        )


# --- 8. random inside while_loop (key threaded through state) ----------------


def test_random_inside_while_loop():
    """Random ops are pure graph ops: they work inside while_loop bodies
    with the key threaded through the loop state via split, and the whole
    loop is deterministic across evaluate calls."""

    @etl.defn
    def f():
        init = (_i64(0), _i64(7), enp.zeros((2,), etl.float32))

        def cond_fn(state):
            i, _k, _acc = state
            return i < _i64(4)

        def body_fn(state):
            i, k, _acc = state
            k1, k2 = etl.random.split(k)
            return (i + 1, k2, etl.random.uniform(k1, (2,)))

        return etl.while_loop(cond_fn, body_fn, init)

    o1 = etl.evaluate(f)
    o2 = etl.evaluate(f)
    assert o1[0].numpy().item() == 4  # four iterations ran
    acc1, acc2 = o1[2].numpy(), o2[2].numpy()
    assert np.array_equal(acc1, acc2)  # deterministic across runs
    assert (acc1 >= 0.0).all() and (acc1 < 1.0).all()


def test_while_loop_key_threading_changes_stream():
    """Threading a DIFFERENT initial key through the same loop gives a
    different draw (the loop consumes the key stream)."""

    @etl.defn
    def f(seed):
        init = (_i64(0), seed, enp.zeros((2,), etl.float32))

        def cond_fn(state):
            i, _k, _acc = state
            return i < _i64(4)

        def body_fn(state):
            i, k, _acc = state
            k1, k2 = etl.random.split(k)
            return (i + 1, k2, etl.random.uniform(k1, (2,)))

        return etl.while_loop(cond_fn, body_fn, init)

    o1 = etl.evaluate(f, etl.random.key(1))
    o2 = etl.evaluate(f, etl.random.key(2))
    assert not np.array_equal(o1[2].numpy(), o2[2].numpy())


# --- 9. validation errors ----------------------------------------------------


def test_randint_static_low_ge_high_valueerror():
    with pytest.raises(ValueError, match="high must be greater than low"):
        etl.trace(
            lambda key: etl.random.randint(key, (2,), 5, 2),
            etl.TensorSpec((), etl.int64),
        )


def test_randint_symbolic_low_ge_high_runtime_valueerror():
    @etl.defn
    def f(key, lo, hi):
        return etl.random.randint(key, (2,), lo, hi)

    with pytest.raises(ValueError, match="span <= 0"):
        etl.evaluate(
            f, etl.random.key(1), np.array(5, dtype=np.int64),
            np.array(2, dtype=np.int64),
        )


def test_permutation_negative_static_n():
    with pytest.raises(ValueError, match="n must be >= 0"):
        etl.trace(
            lambda key: etl.random.permutation(key, -1),
            etl.TensorSpec((), etl.int64),
        )


def test_permutation_rejects_non_int_n():
    with pytest.raises(TypeError, match="n must be a Python int"):
        etl.trace(
            lambda key: etl.random.permutation(key, 2.5),
            etl.TensorSpec((), etl.int64),
        )


def test_shape_rejects_runtime_dynamic_none():
    with pytest.raises(core.TraceError, match="runtime-dynamic None"):
        etl.trace(
            lambda key: etl.random.uniform(key, (2, None)),
            etl.TensorSpec((), etl.int64),
        )


def test_shape_rejects_non_shape_type():
    with pytest.raises(core.TraceError, match="int or a tuple/list"):
        etl.trace(
            lambda key: etl.random.uniform(key, 2.5),
            etl.TensorSpec((), etl.int64),
        )


def test_split_n_validation():
    with pytest.raises(ValueError, match="n must be >= 0"):
        etl.trace(
            lambda key: etl.random.split_n(key, -1),
            etl.TensorSpec((), etl.int64),
        )
    with pytest.raises(TypeError, match="n must be a Python int"):
        etl.trace(
            lambda key: etl.random.split_n(key, 1.5),
            etl.TensorSpec((), etl.int64),
        )


@pytest.mark.parametrize(
    ("probs", "match"),
    [
        (np.array([-0.5, 1.5], dtype=np.float32), "non-negative"),
        (np.array([0.3, 0.3], dtype=np.float32), "sum to 1"),
        (np.array([], dtype=np.float32), "non-empty"),
    ],
    ids=["negative", "sum-not-1", "empty"],
)
def test_multinomial_input_validation(probs, match):
    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, 10)

    with pytest.raises(ValueError, match=match):
        etl.evaluate(f, etl.random.key(1), probs)


def test_multinomial_input_ndim():
    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, 3)

    with pytest.raises(core.ShapeError, match="1-D"):
        etl.evaluate(f, etl.random.key(1), np.ones((2, 2), np.float32))


def test_multinomial_num_samples_validation():
    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, -1)

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        etl.evaluate(f, etl.random.key(1), np.array([0.5, 0.5], np.float32))

    @etl.defn
    def g(key, p):
        return etl.random.multinomial(key, p, 2.5)

    with pytest.raises(TypeError, match="num_samples must be a Python int"):
        etl.evaluate(g, etl.random.key(1), np.array([0.5, 0.5], np.float32))


def test_multinomial_concrete_input_inside_trace_raises():
    @etl.defn
    def f(key):
        return etl.random.multinomial(
            key, etl.tensor(np.array([0.5, 0.5], np.float32)), 3
        )

    with pytest.raises(core.TraceError, match="concrete Tensor input"):
        etl.evaluate(f, etl.random.key(1))


# --- 10. transforms have no rules for random ops ------------------------------


def test_vmap_of_random_op_raises_transformerror():
    """vmap maps the batch axis into the random op's operands; there is no
    batching rule in v1, so the error is an explicit TransformError (never a
    silent Python-loop fallback)."""

    @etl.defn
    def f(key, bound):
        return etl.random.uniform(key, (3,), 0.0, bound)

    with pytest.raises(core.TransformError, match="no batching rule for op 'random_uniform'"):
        etl.vmap(f, in_axes=(None, 0))(
            etl.TensorSpec((), etl.int64), etl.TensorSpec((2, 3), etl.float32)
        )


# --- 11. backend coverage: numpy only in v1 -----------------------------------


def test_stablehlo_export_rejects_random_op():
    """The StableHLO exporter names the offending random op in its explicit
    v1 BackendError (no compiler installation needed — the check is in the
    exporter). ``random_multinomial`` is the only random op still deferred
    in v1 — the other five export (see tests/backends/
    test_iree_emitters_parity.py for their iree parity)."""

    @etl.defn
    def f(key):
        probs = etl.constant(
            etl.tensor(np.array([0.25, 0.25, 0.5], dtype=np.float32))
        )
        return etl.random.multinomial(key, probs, 5)

    graph = etl.trace(f, etl.TensorSpec((), etl.int64))
    with pytest.raises(core.BackendError) as excinfo:
        etl.backends.stablehlo.export(graph)
    msg = str(excinfo.value)
    assert "op 'random_multinomial'" in msg
    assert "not supported in v1" in msg


def test_compiler_backend_lower_rejects_random_op():
    """Adapter lowering rejects the one still-deferred random op
    (``random_multinomial``) via the shared stablehlo export path before
    any compiler runs (pure capability pre-check)."""

    @etl.defn
    def f(key):
        probs = etl.constant(
            etl.tensor(np.array([0.25, 0.25, 0.5], dtype=np.float32))
        )
        return etl.random.multinomial(key, probs, 5)

    graph = etl.trace(f, etl.TensorSpec((), etl.int64))
    with pytest.raises(core.BackendError, match="random_multinomial"):
        etl.lower(graph, backend="iree")
