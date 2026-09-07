"""tests/test_meta_programming.py — genetic-programming end-to-end coverage
for the ``build_cached`` / ``CompileCache`` memoized-build sugar
(``etl.build_cache``, Workstream C).

A miniature genetic-programming harness: an outer loop generates random
arithmetic expression programs ENCODED AS TENSORS — integer program tensors
produced by a small ``@etl.defn`` over ``etl.random.randint`` (one key
``etl.random.key(base + i * gap)`` per outer iteration, materialized via
``etl.evaluate``), prefix-decoded Python-side into static int-only tuple
trees. Each distinct program is built once through ``build_cached`` (keyed
on the tree's static content BEFORE tracing) and its executable is reused
across an inner evaluation loop over several inputs.

Program-tensor token encoding (int32 tokens, randint low=-5 high=5):

    0            -> the x leaf
    -5..-1       -> constant leaf
    1..4         -> binary op: 1=add 2=subtract 3=multiply 4=maximum

A token stream is a prefix encoding: an op token consumes two subtree
encodings; a leaf stops. Decoding is depth-capped (subtrees below depth 4
decode as the x leaf), so expression trees stay small. Tree node form:
``0`` = the x input, a negative int = constant, ``(op, left, right)``
tuples. Only streams whose first token is an op (a binary-root tree that
uses x) are accepted as programs; op nodes whose children both fold to
Python ints are folded at trace time (an all-static subexpression must not
reach an etl op — ops need at least one SymbolicTensor operand).

Asserted end to end: ``cache.misses`` equals the number of DISTINCT trees
over the whole outer loop; a repeated program is a hit returning the SAME
executable object; a REGENERATED identical program (the same encoded
tensor re-decoded) is a hit too (content-based key); a different
TensorSpec is a miss; and every executable matches a pure-numpy reference
tree evaluator on multiple inputs (rtol/atol 1e-5). Two legs: the numpy
backend (default) and iree-llvm-cpu (function-local ``importorskip``; the
leg compiles at most two programs and re-runs the cached executable).

Determinism: ``etl.random`` is key-based and stateless, so the whole outer
loop is reproducible run to run (same seed -> same program tensors).
"""

import numpy as np
import pytest

import etl
from etl.build_cache import CompileCache, build_cached

SPEC = etl.TensorSpec((4,), etl.float32)
SPEC2 = etl.TensorSpec((2,), etl.float32)  # different-spec leg

# -- outer-loop sizing (small: 10 numpy programs, 2 iree compiles) ---------
NUM_PROGRAMS = 10
SEED_BASE = 1000
SEED_GAP = 500  # each outer iteration scans keys base + i*gap + j
IREE_SEED_BASE = 70000

TOKENS_PER_PROGRAM = 13
DEPTH_CAP = 4

# A spread of eval inputs: positive/negative/zero/mixed magnitudes. The
# programs are small int arithmetic, so float32 stays far from overflow.
EVAL_INPUTS = np.array(
    [
        [0.5, 1.0, -2.0, 4.0],
        [-3.0, 0.25, 2.5, -1.5],
        [1.0, -4.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [7.0, -7.0, 1.5, -0.5],
    ],
    dtype=np.float32,
)

_FOLD = {1: lambda a, b: a + b, 2: lambda a, b: a - b,
         3: lambda a, b: a * b, 4: max}
_ETL_OPS = {1: etl.add, 2: etl.subtract, 3: etl.multiply, 4: etl.maximum}


# -- trace-time composition: static tree -> etl op expression ---------------
def _compose(tree, x):
    """Decode a static tree into an etl op expression over ``x``.

    Runs as plain Python at trace time. Int-only subtrees fold to Python
    ints; an etl op is only emitted when at least one side is symbolic
    (tree programs always contain the x leaf, so the root never folds to a
    bare int). Constant leaves reach etl ops as Python scalars.
    """
    if not isinstance(tree, tuple):
        return x if tree == 0 else tree
    op, left, right = tree
    a = _compose(left, x)
    b = _compose(right, x)
    if isinstance(a, int) and isinstance(b, int):
        return _FOLD[op](a, b)
    return _ETL_OPS[op](a, b)


@etl.defn
def _step(x, tree):
    """Apply the static tree program to x.

    ``tree`` is a static argument: it is decoded at trace time into
    add/subtract/multiply/maximum compositions and specializes the graph.
    """
    return _compose(tree, x)


# -- program generation: random trees encoded as tensors --------------------
@etl.defn
def _gen(k):
    """One random program tensor: TOKENS_PER_PROGRAM int32 tokens."""
    return etl.random.randint(
        k, shape=(TOKENS_PER_PROGRAM,), low=-5, high=5, dtype=etl.int32
    )


def _materialize(seed):
    """Run the generator for ``seed`` -> token list (deterministic)."""
    return etl.evaluate(_gen, etl.random.key(seed)).numpy().tolist()


def _decode(tokens, depth=0):
    """Prefix-decode a token stream into ``(tree, consumed)``.

    Token 0 -> the x leaf; a negative token -> a constant leaf; 1..4 ->
    a binary op consuming two subtree encodings. Streams never contain 5
    (randint high=5 is exclusive). Subtrees below DEPTH_CAP decode as the
    x leaf so programs stay small; untaken tokens are discarded.
    """
    if not tokens or depth > DEPTH_CAP:
        return 0, 0
    tok = tokens[0]
    if tok <= 0:
        return tok, 1
    left, n_left = _decode(tokens[1:], depth + 1)
    right, n_right = _decode(tokens[1 + n_left:], depth + 1)
    return (tok, left, right), 1 + n_left + n_right


def _contains_x(tree):
    if not isinstance(tree, tuple):
        return tree == 0
    return _contains_x(tree[1]) or _contains_x(tree[2])


def _find_program(base):
    """Scan keys ``base + j`` for the first stream whose decode is a
    binary-root tree that uses x; return ``(seed, tokens, tree)``."""
    for j in range(100):
        seed = base + j
        tokens = _materialize(seed)
        tree, _ = _decode(tokens)
        if isinstance(tree, tuple) and _contains_x(tree):
            return seed, tokens, tree
    raise AssertionError(f"no usable program found near seed {base}")


# -- pure-numpy reference evaluator -----------------------------------------
def _np_eval(tree, x):
    """Reference evaluator over float64 ``x``; returns a float64 array."""
    if not isinstance(tree, tuple):
        return x if tree == 0 else np.float64(tree)
    op, left, right = tree
    a = _np_eval(left, x)
    b = _np_eval(right, x)
    return {1: a + b, 2: a - b, 3: a * b, 4: np.maximum(a, b)}[op]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ETL_* defaults leak into per-call backend/device resolution."""
    for var in ("ETL_BACKEND", "ETL_DEVICE", "ETL_TARGET_BACKENDS"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# 1. The genetic-programming outer loop (numpy leg)
# --------------------------------------------------------------------------


class TestMetaLoopNumpy:
    def test_distinct_programs_memoized_with_parity(self):
        cache = CompileCache()
        exes = {}
        seen = set()
        records = []
        for i in range(NUM_PROGRAMS):
            seed, tokens, tree = _find_program(SEED_BASE + i * SEED_GAP)
            records.append((seed, tokens, tree))

            # One build_cached per (specs, static tree) — a repeated tree
            # would be a hit; either way misses == distinct trees so far.
            exe = build_cached(_step, SPEC, tree, cache=cache)
            exes[tree] = exe
            seen.add(tree)
            assert cache.misses == len(seen), (i, tree, cache.misses)

            # Inner evaluation loop over several inputs with numpy parity.
            for x in EVAL_INPUTS:
                got = etl.run(exe, x, tree).numpy().astype(np.float64)
                ref = _np_eval(tree, x.astype(np.float64))
                assert np.allclose(got, ref, rtol=1e-5, atol=1e-5), (tree, x)

        # The whole outer loop memoized exactly one compile per DISTINCT
        # tree — never more, never fewer.
        assert cache.misses == len(seen) == len(exes)
        assert cache.misses + cache.hits == NUM_PROGRAMS
        assert all(exes[t].backend == "numpy" for t in exes)

        # Every generated program is genuinely a binary tree using x, and
        # the deterministic stream produced several distinct programs.
        assert len(seen) >= 2
        for _, tokens, tree in records:
            assert isinstance(tree, tuple) and _contains_x(tree)
            assert _decode(tokens)[0] == tree  # decode is a pure function

    def test_repeated_program_is_identity_hit(self):
        cache = CompileCache()
        records = [
            _find_program(SEED_BASE + i * SEED_GAP) for i in range(NUM_PROGRAMS)
        ]
        exes = {}
        for _, _, tree in records:
            exes[tree] = build_cached(_step, SPEC, tree, cache=cache)
        distinct = len(set(tree for _, _, tree in records))

        # Repeating the LAST program is a hit returning the same object.
        last_seed, last_tokens, last_tree = records[-1]
        again = build_cached(_step, SPEC, last_tree, cache=cache)
        assert again is exes[last_tree]
        assert cache.hits == 1 and cache.misses == distinct
        assert len(cache) == distinct

        # A REGENERATED identical program (same encoded tensor, decoded
        # again from scratch) is also a hit: the key is content-based.
        tokens_regen = _materialize(last_seed)
        assert tokens_regen == last_tokens
        tree_regen, _ = _decode(tokens_regen)
        assert tree_regen == last_tree
        exe_regen = build_cached(_step, SPEC, tree_regen, cache=cache)
        assert exe_regen is exes[last_tree]
        assert cache.hits == 2 and cache.misses == distinct

        # And the cached executable still runs fine on fresh inputs.
        for x in EVAL_INPUTS:
            got = etl.run(exe_regen, x, last_tree).numpy().astype(np.float64)
            assert np.allclose(
                got, _np_eval(last_tree, x.astype(np.float64)),
                rtol=1e-5, atol=1e-5,
            )

    def test_different_spec_misses(self):
        cache = CompileCache()
        _, _, tree = _find_program(SEED_BASE)
        exe = build_cached(_step, SPEC, tree, cache=cache)

        # Same static program, different TensorSpec -> a separate entry.
        exe_small = build_cached(_step, SPEC2, tree, cache=cache)
        assert exe_small is not exe
        assert cache.misses == 2 and len(cache) == 2
        # ... and the spec-2 entry memoizes like any other.
        assert build_cached(_step, SPEC2, tree, cache=cache) is exe_small
        assert cache.hits == 1 and cache.misses == 2

        x2 = np.array([1.0, -3.0], dtype=np.float32)
        got = etl.run(exe_small, x2, tree).numpy().astype(np.float64)
        assert np.allclose(got, _np_eval(tree, x2.astype(np.float64)),
                           rtol=1e-5, atol=1e-5)

    def test_memoized_executable_still_validates_static_program_at_run(self):
        # The build_cached sugar must not weaken the run boundary: the
        # static tree stays in the run signature and is validated per run.
        cache = CompileCache()
        records = [
            _find_program(SEED_BASE + i * SEED_GAP) for i in range(NUM_PROGRAMS)
        ]
        exes = {}
        for _, _, tree in records:
            exes[tree] = build_cached(_step, SPEC, tree, cache=cache)
        trees = [tree for _, _, tree in records]

        wrong = next((t for t in trees if t != trees[-1]), None)
        assert wrong is not None  # several distinct programs were generated
        with pytest.raises(etl.TraceError):
            etl.run(exes[trees[-1]], EVAL_INPUTS[0], wrong)

        # A structurally different static program is rejected as well.
        with pytest.raises(etl.TraceError):
            etl.run(exes[trees[-1]], EVAL_INPUTS[0], (2, 0))
        # And the right program still runs — nothing was corrupted.
        got = etl.run(exes[trees[-1]], EVAL_INPUTS[0], trees[-1]).numpy()
        assert got.shape == (4,)


# --------------------------------------------------------------------------
# 2. The same loop through the iree-llvm-cpu adapter (guarded leg)
# --------------------------------------------------------------------------


class TestMetaLoopIree:
    def test_distinct_programs_memoized_with_parity(self):
        # Guarded leg: skips cleanly when the iree compiler/runtime are
        # absent; compiles at most TWO programs (<= 3 compiles total).
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        device = etl.Device("cpu", 0)
        cache = CompileCache()
        exes = {}
        seen = set()
        last_tree = None
        for i in range(2):
            _, _, tree = _find_program(IREE_SEED_BASE + i * SEED_GAP)
            exe = build_cached(
                _step, SPEC, tree, cache=cache,
                backend="iree", device=device,
            )
            assert exe.backend == "iree" and exe.device == device
            exes[tree] = exe
            seen.add(tree)
            last_tree = tree
            assert cache.misses == len(seen)

            # One inner evaluation loop over several inputs on the cached
            # (device-resident-agnostic llvm-cpu) executable.
            for x in EVAL_INPUTS:
                got = etl.run(exe, x, tree).numpy().astype(np.float64)
                ref = _np_eval(tree, x.astype(np.float64))
                assert np.allclose(got, ref, rtol=1e-5, atol=1e-5), (tree, x)

        assert cache.misses == len(seen) == 2

        # A repeated program hits and returns the SAME executable — the
        # second build_cached call performs no new iree compile.
        again = build_cached(
            _step, SPEC, last_tree, cache=cache,
            backend="iree", device=device,
        )
        assert again is exes[last_tree]
        assert cache.hits == 1 and cache.misses == 2 and len(cache) == 2

        # The cached iree executable keeps running on fresh inputs.
        for x in EVAL_INPUTS:
            got = etl.run(again, x, last_tree).numpy().astype(np.float64)
            assert np.allclose(
                got, _np_eval(last_tree, x.astype(np.float64)),
                rtol=1e-5, atol=1e-5,
            )
