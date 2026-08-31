"""etl.random — multi-algorithm key-based RNG tests.

The multi-algorithm framework (see ``etl/ops/random.py`` and
``etl/ir/op_defs/random.py`` — the SINGLE source of truth for the canonical
algorithm names) adds two counter-based algorithms alongside the v1
SplitMix64 default:

- ``splitmix64`` (default, UNCHANGED): a rank-0 int64 key holding a 64-bit
  state — bit-identical to v1 (the existing 64 tests in test_random.py pin
  the back-compat; this file adds no duplicates).
- ``threefry2x32``: a shape ``(2,)`` int32 key (2 counter words), 20 rounds.
- ``philox4x32_10``: a shape ``(4,)`` int32 key (4 counter words), 10 rounds.

``key(seed, algorithm=...)`` accepts canonical strings or ``Algorithm``
enum members; every OTHER function takes no algorithm parameter — the
algorithm is inferred from the key operand's STATIC shape/dtype and stamped
into the op's ``algorithm`` attribute. A key matching no form raises
``ShapeError`` naming the three accepted forms.

All three algorithms are deterministic (same key + operands ⇒ bit-identical
values) and share one word-stream contract: per-op 64-bit pi-hex salts folded
onto the key words (``salt_lo = salt & 0xFFFFFFFF``, ``salt_hi = salt >> 32``),
counter ``(p, 0)`` per cipher call/block p, element i uses flat word i
(uniform/randint/permutation), normal uses words 2i/2i+1; 32-bit words scale
as ``w * 2^-32`` (splitmix64 keeps ``(w >> 11) * 2^-53``). The Random123
published KAT vectors pin the two ciphers exactly (see the known-answer
section below).

Backends: numpy interpreter only (the reference for all 6 ops); this file
asserts nothing about compiler backends.
"""
import re

import numpy as np
import pytest

import etl
from etl import core
import etl.ir as ir

from etl.backends.numpy.kernels.random import _philox4x32, _threefry2x32

from tests.ops.conftest import ops_of, run_numpy

# --- helpers -----------------------------------------------------------------

#: SplitMix64 golden gamma (also the ``split`` second salt) and 64-bit mask —
#: the stream-contract constants re-derived locally for the manual pins.
_GOLDEN = 0x9E3779B97F4A7C15
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix64(z):
    """The standard SplitMix64 finalizer on a Python int (mod 2^64)."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _counter_key_words(seed, salt, count):
    """The documented counter-based key-word derivation: word i is the lower
    32 bits of ``mix64((seed ^ salt) + (i + 1) * GOLDEN mod 2^64)``,
    interpreted as SIGNED int32 (the same generator the numpy kernels use)."""
    state = (seed & _MASK64) ^ salt
    words = []
    for i in range(count):
        w = _mix64((state + (i + 1) * _GOLDEN) & _MASK64) & 0xFFFFFFFF
        words.append(w if w < 2**31 else w - 2**32)
    return words


# --- 1. key(seed, algorithm=...) parameter validation -----------------------


def test_algorithm_accepts_canonical_names_and_enum():
    """Canonical strings and Algorithm enum members produce the SAME key
    words; ``etl.random.ALGORITHMS`` is re-exported."""
    assert etl.random.ALGORITHMS == ("splitmix64", "threefry2x32", "philox4x32_10")
    for alg in etl.random.ALGORITHMS:
        member = etl.random.Algorithm(alg)
        assert np.array_equal(
            etl.random.key(42, algorithm=alg).numpy(),
            etl.random.key(42, algorithm=member).numpy(),
        )


def test_default_algorithm_stays_splitmix64():
    """The default algorithm is splitmix64: key(seed) and
    key(seed, "splitmix64") produce identical words (v1 behavior)."""
    assert np.array_equal(
        etl.random.key(123).numpy(), etl.random.key(123, "splitmix64").numpy()
    )
    assert np.array_equal(
        etl.random.key(-7).numpy(), etl.random.key(-7, algorithm="splitmix64").numpy()
    )


def test_unknown_algorithm_string_raises_valueerror():
    """Unknown canonical names raise ValueError listing the accepted names."""
    with pytest.raises(
        ValueError,
        match="unknown random algorithm 'nope'; expected one of splitmix64, "
        "threefry2x32, philox4x32_10",
    ):
        etl.random.key(1, algorithm="nope")


def test_non_string_algorithm_raises_typeerror():
    with pytest.raises(TypeError, match="random algorithm must be a string"):
        etl.random.key(1, algorithm=3)


# --- 2. key shape/dtype per algorithm ---------------------------------------


@pytest.mark.parametrize(
    ("alg", "shape", "dtype"),
    [
        ("splitmix64", (), np.dtype("int64")),
        ("threefry2x32", (2,), np.dtype("int32")),
        ("philox4x32_10", (4,), np.dtype("int32")),
    ],
    ids=["splitmix64", "threefry2x32", "philox4x32_10"],
)
def test_key_shape_and_dtype_outside_trace(alg, shape, dtype):
    """Outside a trace key() returns a concrete Tensor whose shape/dtype is
    the algorithm's key type."""
    k = etl.random.key(42, alg)
    assert isinstance(k, core.Tensor)
    assert k.shape == shape
    assert k.dtype == dtype


def test_key_words_reproducible_across_calls():
    """The same seed + algorithm ⇒ identical key words across calls."""
    for alg in etl.random.ALGORITHMS:
        assert np.array_equal(
            etl.random.key(42, alg).numpy(), etl.random.key(42, alg).numpy()
        )


@pytest.mark.parametrize(
    ("alg", "shape", "dtype"),
    [
        ("splitmix64", (), np.dtype("int64")),
        ("threefry2x32", (2,), np.dtype("int32")),
        ("philox4x32_10", (4,), np.dtype("int32")),
    ],
    ids=["splitmix64", "threefry2x32", "philox4x32_10"],
)
def test_key_inside_trace_builds_constant(alg, shape, dtype):
    """Inside a trace key() builds a Constant with the SAME shape/dtype and
    the same words as the concrete key."""
    @etl.defn
    def f():
        return etl.random.key(42, alg)

    graph = etl.trace(f)
    assert [op.name for op in ops_of(graph)] == ["constant", "return"]
    out = etl.evaluate(f).numpy()
    assert out.shape == shape
    assert out.dtype == dtype
    assert np.array_equal(out, etl.random.key(42, alg).numpy())


@pytest.mark.parametrize("alg", ["threefry2x32", "philox4x32_10"])
def test_counter_key_usable_as_explicit_graph_input(alg):
    """A concrete counter-based key is usable as an explicit graph input
    (same as the v1 splitmix64 key)."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (4,))

    out1 = run_numpy(f, etl.random.key(7, alg))
    out2 = run_numpy(f, etl.random.key(7, alg))
    assert np.array_equal(out1, out2)


# --- 3. algorithm/key mismatch ----------------------------------------------


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((3,), etl.int32),  # matches no algorithm's key type
        ((2,), etl.int64),  # threefry shape with the wrong dtype
        ((4,), etl.int64),  # philox shape with the wrong dtype
        ((), etl.int32),  # scalar but not int64
    ],
    ids=["3xi32", "2xi64", "4xi64", "scalar-i32"],
)
def test_key_matching_no_algorithm_form_raises_shapeerror(shape, dtype):
    """A key matching none of the three canonical forms fails at trace time
    with a ShapeError naming all three accepted forms (exact wording)."""
    expected = (
        "random.uniform: key must be a rank-0 int64 tensor (splitmix64), "
        "a shape (2,) int32 tensor (threefry2x32), or a shape (4,) int32 "
        f"tensor (philox4x32_10), got dtype={core.dtype(dtype)} shape={shape}"
    )
    with pytest.raises(core.ShapeError, match=re.escape(expected)):
        etl.trace(
            lambda key: etl.random.uniform(key, (2,)), etl.TensorSpec(shape, dtype)
        )


@pytest.mark.parametrize(
    "fn",
    [
        lambda key: etl.random.uniform(key, (2,)),
        lambda key: etl.random.normal(key, (2,)),
        lambda key: etl.random.randint(key, (2,), 0, 5),
        lambda key: etl.random.permutation(key, 4),
        lambda key: etl.random.multinomial(
            key, np.array([0.5, 0.5], dtype=np.float32), 3
        ),
        lambda key: etl.random.split(key),
        lambda key: etl.random.split_n(key, 2),
    ],
    ids=["uniform", "normal", "randint", "permutation", "multinomial", "split", "split_n"],
)
def test_malformed_key_rejected_by_every_op(fn):
    """Every random op (other than key) rejects a (3,) int32 key with the
    three-forms ShapeError — the key check runs before any other validation."""
    with pytest.raises(core.ShapeError, match="key must be"):
        etl.trace(fn, etl.TensorSpec((3,), etl.int32))


def test_ir_level_algorithm_attr_mismatch_raises_shapeerror():
    """IR-level construction with an algorithm attribute inconsistent with
    the key operand's static type fails at shape inference: a key form
    belonging to a DIFFERENT algorithm is called out explicitly, and an
    unknown algorithm name raises ValueError (validate_algorithm)."""
    builder = ir.Builder()
    builder.build_module()
    function = builder.build_function("main", (ir.ValueType(np.int64, ()),))
    key_arg, = function.entry_block.arguments
    with pytest.raises(
        core.ShapeError,
        match="key has the splitmix64 key form \\(dtype=int64 shape=\\(\\)\\) "
        "but the op declares algorithm 'threefry2x32'",
    ):
        builder.emit(
            "random_key_mix",
            (key_arg,),
            attributes={"algorithm": "threefry2x32", "salt": 0},
        )
    with pytest.raises(ValueError, match="unknown random algorithm"):
        builder.emit(
            "random_key_mix", (key_arg,), attributes={"algorithm": "nope", "salt": 0}
        )


def test_ops_stamp_the_algorithm_attribute():
    """Every random op stamps the inferred algorithm into its ``algorithm``
    attribute (the default splitmix64 is stamped explicitly, too)."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (2,)), etl.random.split(key)

    graph = etl.trace(f, etl.TensorSpec((), etl.int64))
    stamped = [
        op.attributes["algorithm"]
        for op in ops_of(graph)
        if op.name in ("random_key_mix", "random_uniform")
    ]
    assert stamped == ["splitmix64", "splitmix64", "splitmix64"]


# --- 4. determinism for all 3 algorithms -------------------------------------


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_cross_evaluate_determinism_per_algorithm(alg):
    """Same key + same operands ⇒ bit-identical draws across separate
    evaluate calls for every algorithm (uniform/normal/randint/permutation)."""

    @etl.defn
    def f(key):
        return (
            etl.random.uniform(key, (5, 3)),
            etl.random.normal(key, (5, 3)),
            etl.random.randint(key, (7,), 0, 10),
            etl.random.permutation(key, 6),
        )

    key = etl.random.key(123, alg)
    r1 = run_numpy(f, key)
    r2 = run_numpy(f, key)
    for a, b in zip(r1, r2):
        assert np.array_equal(a, b)


# --- 5. split / split_n for all 3 algorithms ---------------------------------


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_split_children_shape_distinct_and_deterministic(alg):
    """split() derives two decorrelated keys of the algorithm's key shape;
    the pair is reproducible and each child differs from the parent key."""

    @etl.defn
    def f(key):
        a, b = etl.random.split(key)
        return a, b

    parent = etl.random.key(99, alg)
    a1, b1 = run_numpy(f, parent)
    a2, b2 = run_numpy(f, parent)
    assert a1.shape == b1.shape == parent.shape
    assert a1.dtype == parent.dtype
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, b1)  # derived keys are decorrelated
    assert not np.array_equal(a1, parent.numpy())  # distinct from the parent
    assert not np.array_equal(b1, parent.numpy())


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_split_of_split_works_per_algorithm(alg):
    """Children of children are valid keys of the same shape, pairwise
    distinct from each other and from the first-generation keys."""

    @etl.defn
    def f(key):
        a, b = etl.random.split(key)
        a1, a2 = etl.random.split(a)
        return a, b, a1, a2

    parent = etl.random.key(5, alg)
    a, b, a1, a2 = run_numpy(f, parent)
    assert {x.shape for x in (a, b, a1, a2)} == {parent.shape}
    assert len({x.tobytes() for x in (a, b, a1, a2)}) == 4


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_split_n_distinct_and_deterministic(alg):
    """split_n(key, n) yields n pairwise-distinct keys of the algorithm's
    key shape, deterministically."""

    @etl.defn
    def f(key):
        return etl.random.split_n(key, 5)

    keys = run_numpy(f, etl.random.key(99, alg))
    assert len(keys) == 5
    assert all(k.shape == etl.random.key(1, alg).shape for k in keys)
    assert len({k.tobytes() for k in keys}) == 5  # pairwise distinct
    keys2 = run_numpy(f, etl.random.key(99, alg))
    assert all(np.array_equal(a, b) for a, b in zip(keys, keys2))


# --- 6. known-answer vectors (Random123 ground truth) ------------------------


def test_threefry2x32_kat_vectors():
    """Random123 threefry2x32-20 KAT vectors: (key, ctr) → the two uint32
    output words (the kernel's return convention: ``_threefry2x32(c0, c1,
    k0, k1)``)."""
    x0, x1 = _threefry2x32(0, 0, 0, 0)
    assert int(x0) == 0x6B200159 and int(x1) == 0x99BA4EFE
    x0, x1 = _threefry2x32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    assert int(x0) == 0x1CB996FC and int(x1) == 0xBB002BE7
    x0, x1 = _threefry2x32(0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)
    assert int(x0) == 0xC4923A9C and int(x1) == 0x483DF7A0


def test_philox4x32_kat_vectors():
    """Random123 philox4x32-10 KAT vectors: (key, ctr) → the four uint32
    output words. The kernel takes FOUR counter words (``_philox4x32(c0,
    c1, c2, c3, key)``); the published "ctr=(1,0)" vector is the full
    counter (1, 0, 0, 0)."""
    key0 = np.array([0, 0, 0, 0], dtype=np.uint32)
    w = _philox4x32(0, 0, 0, 0, key0)
    assert [int(v) for v in w] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    w = _philox4x32(0, 0, 0, 0, np.array([1, 2, 3, 4], dtype=np.uint32))
    assert [int(v) for v in w] == [0x962A3161, 0x1CE2572E, 0xB5619CAB, 0xD3996EEE]
    w = _philox4x32(1, 0, 0, 0, key0)
    assert [int(v) for v in w] == [0xF8E4CCA4, 0x5CB200DB, 0xB1A574EB, 0x097EFF67]


def test_threefry_uniform_stream_matches_manual_cipher():
    """End-to-end stream pin: a threefry2x32 uniform graph reproduces a
    manual cipher computation following the documented word-stream contract
    — key words via the SplitMix64 derivation, per-op salt folded onto the
    key (k0 ^ salt_lo, k1 ^ salt_hi), counter (p, 0) per call p, element i
    = flat word i, u = w * 2^-32. Bit-exact (float32)."""
    seed = 7
    salt_alg = 0x243F6A8885A308D3  # threefry key-derivation salt (pi hex digits)
    salt_uniform = 0x243F6A8885A308D3  # the per-op uniform salt
    k0, k1 = _counter_key_words(seed, salt_alg, 2)
    salt_lo = salt_uniform & 0xFFFFFFFF
    salt_hi = salt_uniform >> 32
    k0 = (k0 & 0xFFFFFFFF) ^ salt_lo
    k1 = (k1 & 0xFFFFFFFF) ^ salt_hi
    words = []
    for p in range(2):  # 4 elements -> cipher calls (0,0) and (1,0)
        x0, x1 = _threefry2x32(p, 0, k0, k1)
        words.extend([int(x0), int(x1)])
    manual = np.array([w * 2.0**-32 for w in words], dtype=np.float32)

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (4,))

    out = run_numpy(f, etl.random.key(seed, "threefry2x32"))
    assert np.array_equal(out, manual)


def test_philox_uniform_stream_matches_manual_cipher():
    """Same end-to-end pin for philox4x32_10: 4-word key derivation, salt
    folded as (k0 ^ lo, k1 ^ hi, k2 ^ lo, k3 ^ hi), counter (p, 0, 0, 0)
    per block, 4 words per block, u = w * 2^-32. Bit-exact (float32)."""
    seed = 7
    salt_alg = 0x13198A2E03707344  # philox key-derivation salt (pi hex digits)
    salt_uniform = 0x243F6A8885A308D3  # the per-op uniform salt
    kw = _counter_key_words(seed, salt_alg, 4)
    salt_lo = salt_uniform & 0xFFFFFFFF
    salt_hi = salt_uniform >> 32
    key4 = np.array(
        [
            (kw[0] & 0xFFFFFFFF) ^ salt_lo,
            (kw[1] & 0xFFFFFFFF) ^ salt_hi,
            (kw[2] & 0xFFFFFFFF) ^ salt_lo,
            (kw[3] & 0xFFFFFFFF) ^ salt_hi,
        ],
        dtype=np.uint32,
    )
    words = []
    for p in range(2):  # 8 elements -> blocks (0,0,0,0) and (1,0,0,0)
        w = _philox4x32(p, 0, 0, 0, key4)
        words.extend([int(v) for v in w])
    manual = np.array([w * 2.0**-32 for w in words], dtype=np.float32)

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (8,))

    out = run_numpy(f, etl.random.key(seed, "philox4x32_10"))
    assert np.array_equal(out, manual)


def test_threefry_split_matches_manual_cipher():
    """split() child derivation for threefry2x32: child a is the cipher at
    counter (0, 0) with the key folded by salt 0 (identity); child b folds
    the SplitMix64 golden gamma (low/high words) into the key — the
    documented split semantics, pinned against a manual cipher run."""
    seed = 5
    salt_alg = 0x243F6A8885A308D3
    k0, k1 = _counter_key_words(seed, salt_alg, 2)
    k0 = np.uint32(k0 & 0xFFFFFFFF)
    k1 = np.uint32(k1 & 0xFFFFFFFF)
    x0, x1 = _threefry2x32(0, 0, k0, k1)
    child_a = np.array([x0, x1], dtype=np.uint32).astype(np.int32)
    gl = np.uint32(_GOLDEN & 0xFFFFFFFF)
    gh = np.uint32(_GOLDEN >> 32)
    x0, x1 = _threefry2x32(0, 0, k0 ^ gl, k1 ^ gh)
    child_b = np.array([x0, x1], dtype=np.uint32).astype(np.int32)

    @etl.defn
    def f(key):
        a, b = etl.random.split(key)
        return a, b

    a, b = run_numpy(f, etl.random.key(seed, "threefry2x32"))
    assert np.array_equal(a, child_a)
    assert np.array_equal(b, child_b)


# --- 7. per-algorithm properties ---------------------------------------------


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_uniform_range_and_mean_per_algorithm(alg):
    @etl.defn
    def f(key):
        return etl.random.uniform(key, (20000,), 0.0, 1.0)

    u = run_numpy(f, etl.random.key(1, alg))
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.05


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_uniform_affine_mapping_per_algorithm(alg):
    """The low/high affine mapping holds for every algorithm."""
    @etl.defn
    def f(key):
        return etl.random.uniform(key, (20000,), 2.0, 5.0)

    u = run_numpy(f, etl.random.key(1, alg))
    assert u.min() >= 2.0 and u.max() < 5.0
    assert abs(u.mean() - 3.5) < 0.05


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_randint_range_per_algorithm(alg):
    @etl.defn
    def f(key):
        return etl.random.randint(key, (20000,), 0, 10)

    r = run_numpy(f, etl.random.key(1, alg))
    assert r.dtype == np.int32
    assert r.min() >= 0 and r.max() <= 9  # high exclusive
    assert set(np.unique(r)) == set(range(10))


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_permutation_is_valid_shuffle_per_algorithm(alg):
    @etl.defn
    def f(key):
        return etl.random.permutation(key, 9)

    perm = run_numpy(f, etl.random.key(1, alg))
    assert perm.shape == (9,)
    assert np.array_equal(np.sort(perm), np.arange(9))


@pytest.mark.parametrize("alg", ["splitmix64", "threefry2x32", "philox4x32_10"])
def test_normal_finite_per_algorithm(alg):
    @etl.defn
    def f(key):
        return etl.random.normal(key, (10000,))

    z = run_numpy(f, etl.random.key(1, alg))
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.1


def test_same_seed_draws_differ_across_algorithms():
    """The three algorithms draw decorrelated streams: same seed + same op
    → different values for each algorithm."""

    @etl.defn
    def f(key):
        return etl.random.uniform(key, (8,))

    draws = [
        run_numpy(f, etl.random.key(9, alg)) for alg in etl.random.ALGORITHMS
    ]
    assert len({tuple(d) for d in draws}) == 3


# --- 8. multinomial with the algorithm-aware stream --------------------------


@pytest.mark.parametrize("alg", ["threefry2x32", "philox4x32_10"])
def test_multinomial_algorithm_stream(alg):
    """multinomial with a counter-based key: unchanged semantics — in-range
    indices, (num_samples,) shape, int32 dtype, deterministic."""

    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, 1000)

    p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    idx = run_numpy(f, etl.random.key(1, alg), p)
    assert idx.dtype == np.int32 and idx.shape == (1000,)
    assert set(np.unique(idx)) <= {0, 1, 2}
    assert (idx == 2).mean() > 0.6  # the 0.7 mass dominates empirically
    assert np.array_equal(idx, run_numpy(f, etl.random.key(1, alg), p))


def test_multinomial_zero_samples_with_counter_key():
    for alg in ("threefry2x32", "philox4x32_10"):
        @etl.defn
        def f(key, p):
            return etl.random.multinomial(key, p, 0)

        idx = run_numpy(f, etl.random.key(1, alg), np.array([0.5, 0.5], np.float32))
        assert idx.shape == (0,)


@pytest.mark.parametrize(
    ("probs", "match"),
    [
        (np.array([-0.5, 1.5], dtype=np.float32), "non-negative"),
        (np.array([0.3, 0.3], dtype=np.float32), "sum to 1"),
        (np.array([], dtype=np.float32), "non-empty"),
    ],
    ids=["negative", "sum-not-1", "empty"],
)
def test_multinomial_input_validation_with_counter_key(probs, match):
    """The same explicit runtime validation applies with an algorithm-aware
    key (threefry2x32) — errors, never silent fallback."""

    @etl.defn
    def f(key, p):
        return etl.random.multinomial(key, p, 10)

    with pytest.raises(ValueError, match=match):
        etl.evaluate(f, etl.random.key(1, "threefry2x32"), probs)


# --- 9. Graph save/load preserves the algorithm attribute --------------------


@pytest.mark.parametrize(
    ("alg", "spec"),
    [
        ("threefry2x32", etl.TensorSpec((2,), etl.int32)),
        ("philox4x32_10", etl.TensorSpec((4,), etl.int32)),
    ],
    ids=["threefry2x32", "philox4x32_10"],
)
def test_graph_save_load_preserves_algorithm(tmp_path, alg, spec):
    """A key(seed, alg) + split + uniform graph round-trips through
    save/load with the ``algorithm`` attribute preserved, and the loaded
    graph runs bit-identical to the original."""

    @etl.defn
    def f(key):
        a, b = etl.random.split(key)
        return a, b, etl.random.uniform(key, (3,))

    graph = etl.trace(f, spec)
    path = tmp_path / f"random_{alg}.etlgraph"
    graph.save(path)
    loaded = etl.Graph.load(path)

    # every random op in the loaded graph carries the algorithm attribute
    random_ops = [
        op
        for op in ops_of(loaded)
        if op.name in ("random_key_mix", "random_uniform")
    ]
    assert random_ops
    for op in random_ops:
        assert op.attributes["algorithm"] == alg

    # the loaded graph runs bit-identical to the original
    exe0 = etl.load(etl.compile(etl.lower(graph)))
    exe1 = etl.load(etl.compile(etl.lower(loaded)))
    key = etl.random.key(11, alg)
    for a, b in zip(etl.run(exe0, key), etl.run(exe1, key)):
        assert np.array_equal(a.numpy(), b.numpy())
