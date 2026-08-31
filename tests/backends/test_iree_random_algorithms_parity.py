"""iree-llvm-cpu round-trip parity for the algorithm-aware RNG lowering
(Threefry2x32 / Philox4x32_10 — merged with commit 119854b).

Covered here (see ``etl/backends/stablehlo/random_export.py``):

* the INLINE threefry2x32/philox4x32_10 word-stream expansions — the
  default on iree, because the adapter declares
  ``Capabilities.rng_bit_generator=False`` (iree 3.11 legalizes
  RNG_ALG_THREE_FRY but fails RNG_ALG_PHILOX, so the exporter uses its
  bit-exact inline expansions for both algorithms), and
* the NATIVE ``stablehlo.rng_bit_generator`` THREE_FRY path (exporter
  option ``{"rng_bit_generator": True}`` + a manual ``LoweredProgram``),
  whose iree-legalized words were empirically verified to match the inline
  cipher.

Bit-exactness contract (the numpy kernels in
``etl/backends/numpy/kernels/random.py`` are the reference — the word-stream
contract is pinned there): per-op pi-hex salts XOR-folded onto the key words;
counter (p, 0); element i <- flat word i; normal pairs words 2i/2i+1;
uniform = ``w * 2^-32`` for 32-bit-word streams (FULL-word scaling —
pinned), ``(w >> 11) * 2^-53`` for 64-bit streams; randint = floor(u*span)
in f64; permutation = stable argsort of the sign-flipped words; key_mix =
one cipher block at counter (0, 0) -> the algorithm's key shape. So
uniform/randint/permutation/key_mix (``split``/``split_n``) are bit-exact vs
``etl.evaluate``; ``random_normal`` with f64 output is bit-exact (or within
1 ulp), and with f32 output takes the DOCUMENTED f32 Box-Muller fast path —
deterministic (same key => bit-identical across runs) but ~1e-6 vs the numpy
f64 math, so the f32 parity checks are tolerance-based
(``RANDOM_NORMAL_TOL``; measured budget ~4.8e-7).

BUG(etl) status at this commit: two distinct exporter defects make the
threefry/philox paths on iree NOT bit-exact vs numpy where the contract pins
bit-exactness — the affected tests below carry ``# BUG(etl):`` markers and
stay FAILING per the repo protocol (this dir is read-only w.r.t. etl):

1. ``_emit_u_scaled32`` applies the 64-bit-word scaling
   ``(word >> 11) * 2^-53`` to the 32-bit words instead of the pinned
   full-word ``word * 2^-32``, so iree's u == numpy's u * 2^-32 —
   uniform/randint/normal wrong by ~2^-32 on BOTH algorithms (inline and
   native, incl. cuda).
2. ``_emit_philox_words_inline`` uses only the (k0', k1') round-key pair for
   all 10 rounds instead of the pinned cyclic (k0,k1)/(k2,k3) alternation
   (numpy ``_philox4x32``: round r uses ``(k_{2r mod 4} + r*W0,
   k_{(2r+1) mod 4} + r*W1)``) — philox key_mix/split/split_n/permutation
   words wrong for keys with distinct words (zero/all-equal keys still
   match).

Green despite the above: threefry permutation + split/split_n, native
threefry permutation (native == inline), and ALL determinism checks.

The xla adapter declares ``Capabilities.rng_bit_generator=True``, but xla
run tests stay gate-skipped without a user-provided PJRT plugin
(``ETL_PJRT_PLUGIN``/``plugin_path``) — there are NO xla tests here.

The iree-cuda smoke set (device 5, ``target_backends=["cuda"]``) skips when
the IREE cuda HAL driver or the GPU is unavailable (same guard style as
``test_iree_emitters_parity.py``; while-loop graphs are not involved here).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl
from etl.backends.program import LoweredProgram, Signature

# ---------------------------------------------------------------------------
# data (fixed seed — same convention as test_iree_emitters_parity.py)
# ---------------------------------------------------------------------------

ALGORITHMS = ("threefry2x32", "philox4x32_10")

#: The algorithm's key layout (dtype + shape) per canonical algorithm:
#: threefry2x32 -> (2,) int32; philox4x32_10 -> (4,) int32.
KEY_SPECS = {
    "threefry2x32": etl.TensorSpec((2,), etl.int32),
    "philox4x32_10": etl.TensorSpec((4,), etl.int32),
}

#: Raw key-word arrays per algorithm (the explicit-input form): (0, 0),
#: (0xFFFFFFFF, 0xFFFFFFFF) as np.int32 -1s, a mixed word, and one
#: seed-derived key via etl.random.key(42, algorithm=...).
KEYS = {
    "threefry2x32": (
        np.array([0, 0], dtype=np.int32),  # (0, 0)
        np.array([-1, -1], dtype=np.int32),  # (0xFFFFFFFF, 0xFFFFFFFF)
        np.array([-1, 0], dtype=np.int32),  # (0xFFFFFFFF, 0)
        etl.random.key(42, algorithm="threefry2x32").numpy(),
    ),
    "philox4x32_10": (
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([-1, -1, -1, -1], dtype=np.int32),
        np.array([-1, 0, -1, 0], dtype=np.int32),
        etl.random.key(42, algorithm="philox4x32_10").numpy(),
    ),
}

#: Documented tolerance for the f32 normal fast path (measured ~4.8e-7;
#: 10x+ headroom over the f32 rounding budget).
RANDOM_NORMAL_TOL = dict(rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / list-or-tuple-of-Tensors -> ndarray / same-shaped ndarray
    container (the native ``exe.run([...])`` path returns a LIST of flat
    Tensors)."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    if isinstance(v, (list, tuple)):
        return type(v)(_np(x) for x in v)
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _assert_close(got, want, rtol=1e-5, atol=1e-5):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.allclose(gp, wp, rtol=rtol, atol=atol), f"{gp} != {wp}"


def _parity(fn, spec, key, exact=True, **iree_kwargs):
    """Build on iree, run, and compare vs ``etl.evaluate`` (numpy reference)."""
    want = etl.evaluate(fn, key)
    exe = etl.build(fn, spec, backend="iree", **iree_kwargs)
    got = etl.run(exe, key)
    if exact:
        _assert_exact(got, want)
    else:
        _assert_close(got, want)


def _require_cuda():
    """Skip when the IREE cuda HAL driver or GPU 5 is unavailable."""
    import iree.runtime as rt

    try:
        # etl Device("cuda", 5) maps to iree device_id 6 (1-based ids).
        rt.get_driver("cuda").create_device(device_id=6)
    except Exception as exc:  # noqa: BLE001 — any driver/device failure skips
        pytest.skip(f"IREE cuda HAL driver or GPU 5 unavailable: {exc}")


def _signature(graph) -> Signature:
    """Rebuild a ``Signature`` from a traced graph (the compiler.py recipe)."""
    from etl.backends.numpy.interpreter import entry_function

    main_fn = entry_function(graph.module)
    return Signature(
        input_tree=graph.input_specs,
        output_tree=graph.output_tree,
        input_specs=tuple(graph.tensor_specs),
        output_specs=tuple(
            etl.core.TensorSpec(shape=value_type.shape, dtype=value_type.dtype)
            for value_type in main_fn.output_types
        ),
        static_values=tuple(record.value for record in graph.static_values),
        output_static_values=tuple(
            record.value for record in graph.output_static_values
        ),
    )


def _lowered_native(graph) -> LoweredProgram:
    """A manual LoweredProgram carrying the exporter's NATIVE
    ``rng_bit_generator`` MLIR (the adapter's capability flag is False, so
    the normal ``etl.build`` path cannot produce this)."""
    mlir = etl.backends.stablehlo.export(
        graph, options={"rng_bit_generator": True}
    )
    return LoweredProgram(
        backend="iree",
        signature=_signature(graph),
        payload={
            "format": "stablehlo",
            "format_version": 1,
            "mlir_text": mlir,
            "entry_functions": ("main",),
        },
    )


# ---------------------------------------------------------------------------
# sampling graphs (all take the algorithm's key as their only input)
# ---------------------------------------------------------------------------


def _uniform5(k):
    return etl.random.uniform(k, (5,), dtype=etl.float32)


def _uniform2d(k):
    return etl.random.uniform(k, (4, 5), dtype=etl.float32)


def _randint8(k):
    return etl.random.randint(k, (8,), low=0, high=100)


def _perm5(k):
    return etl.random.permutation(k, 5)


def _perm8(k):
    return etl.random.permutation(k, 8)


def _normal_f64(k):
    return etl.random.normal(k, (4, 5), dtype=etl.float64)


def _normal_f32(k):
    return etl.random.normal(k, (4, 5), dtype=etl.float32)


def _split(k):
    a, b = etl.random.split(k)
    return a, b


def _split_n3(k):
    return etl.random.split_n(k, 3)


INLINE_CASES = [
    ("uniform_5", _uniform5),
    ("uniform_2d_4x5", _uniform2d),
    ("randint_8", _randint8),
    ("permutation_5", _perm5),
    ("permutation_8", _perm8),
]

INLINE_PARAMS = [
    (f"{case_id}_{algorithm}", fn, KEY_SPECS[algorithm], algorithm)
    for algorithm in ALGORITHMS
    for case_id, fn in INLINE_CASES
]


# ---------------------------------------------------------------------------
# 1. inline uniform/randint/permutation — bit-exact vs numpy, one compile
#    per (case, algorithm), every key looped through the same executable
# ---------------------------------------------------------------------------

# BUG(etl): the uniform/randint AND philox-permutation ids below are NOT
# bit-exact on iree. Two distinct exporter defects (see the module docstring):
# (1) `_emit_u_scaled32` (etl/backends/stablehlo/random_export.py) scales the
# 32-bit words with the SPLITMIX64 64-bit formula `(word >> 11) * 2^-53`
# instead of the pinned 32-bit FULL-word scaling `word * 2^-32`
# (etl/backends/numpy/kernels/random.py `_word_uniforms`), so iree's u ==
# numpy's u * 2^-32: uniform values ~2^-32 too small and randint collapses to
# the low bound (floor(u*span) == 0 for span < 2^32). Repro (threefry2x32,
# key (0, 0), uniform (5,)): iree [1.4111967e-10 3.1785241e-11 2.0577884e-10
# ...] vs numpy [0.60610473 0.136517 0.8838137 ...]; randint (8,) iree
# [0 0 0 0 0 0 0 0] vs numpy [63 1 94 41 45 47 44 11]. (2)
# `_emit_philox_words_inline` never switches the round-key pair to (k2', k3')
# on odd rounds — the philox permutation ids fail on the seed-derived key
# (words wrong; keys with k0==k2 and k1==k3 still match). The threefry
# permutation ids stay bit-exact (order-based — neither defect enters). Do
# NOT skip/xfail/weaken.
@pytest.mark.parametrize(
    "case_id,fn,spec,algorithm",
    [(c[0], c[1], c[2], c[3]) for c in INLINE_PARAMS],
    ids=[c[0] for c in INLINE_PARAMS],
)
def test_inline_bit_exact(case_id, fn, spec, algorithm):
    """One iree compile per (case, algorithm); every key in the algorithm's
    list runs through the same executable. Bit-exact vs ``etl.evaluate``
    (numpy reference) AND bit-identical across repeated runs of the same
    executable (determinism folded into the loop)."""
    exe = etl.build(fn, spec, backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(fn, key)
        got = etl.run(exe, key)
        _assert_exact(got, want)
        _assert_exact(etl.run(exe, key), got)  # same exe, second call
        if "uniform" in case_id:
            assert _np(got).dtype == np.float32
        if "permutation" in case_id:
            perm = _np(got)
            assert sorted(perm.tolist()) == list(range(perm.shape[-1]))


# ---------------------------------------------------------------------------
# 2. normal — f64 bit-exact; f32 within the documented fast-path tolerance
#    AND bit-identical across two separate runs (determinism)
# ---------------------------------------------------------------------------

# BUG(etl): the same `_emit_u_scaled32` scaling violation breaks normal: both
# Box-Muller uniforms are ~2^-32 too small, so z = sqrt(-2 log u1) saturates
# at ~6.66 (log of 2^-32) with cos(2*pi*u2) ~ 1. Repro (threefry2x32, key 42,
# normal (4, 5) f64): iree all-positive [6.68..7.53] vs numpy
# [-1.92..2.21] — NOT bit-exact (f64) and far outside RANDOM_NORMAL_TOL
# (f32). The determinism assertion below still holds (same key => the same
# wrong values, bit-identical). Do NOT skip/xfail/weaken.
@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_normal_f64_bit_exact(algorithm):
    """f64 output normal: bit-exact vs the numpy kernel (both algorithms)."""
    exe = etl.build(_normal_f64, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(_normal_f64, key)
        got = etl.run(exe, key)
        assert _np(got).dtype == np.float64
        _assert_exact(got, want)


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_normal_f32_tolerance_and_determinism(algorithm):
    """f32 output normal: within the documented f32 fast-path budget vs
    numpy AND bit-identical across two separate runs of the same exe."""
    exe = etl.build(_normal_f32, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(_normal_f32, key)
        g1 = etl.run(exe, key)
        g2 = etl.run(exe, key)
        assert _np(g1).dtype == np.float32
        _assert_close(g1, want, **RANDOM_NORMAL_TOL)
        _assert_exact(g1, g2)  # the hard same-key determinism contract


# ---------------------------------------------------------------------------
# 3. key_mix / split — the algorithm's key shape, bit-exact vs numpy
# ---------------------------------------------------------------------------

# BUG(etl): philox key_mix (split/split_n) is NOT bit-exact — the exporter's
# `_emit_philox_words_inline` uses only the (k0', k1') round-key pair for all
# 10 rounds instead of the pinned cyclic (k0,k1)/(k2,k3) alternation (numpy
# `_philox4x32`: round r uses `(k_{2r mod 4} + r*W0, k_{(2r+1) mod 4} + r*W1)`).
# Repro (philox key from seed 42, split's salt-0 first key): iree
# [-1408616261 2096774548 810715712 -217573850] vs numpy
# [-619790300 899531137 425014980 -1284574911]. The threefry ids pass
# (cipher verified); philox keys with k0==k2 and k1==k3 also pass (the
# alternation is a no-op there). Do NOT skip/xfail/weaken.


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_split_bit_exact(algorithm):
    """``split(k)`` -> two keys of the algorithm's key shape ((2,) / (4,)
    int32), bit-exact vs numpy."""
    exe = etl.build(_split, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(_split, key)
        got = etl.run(exe, key)
        _assert_exact(got, want)
        for leaf in etl.tree_leaves(etl.tree_map(_np, got)):
            assert leaf.dtype == np.int32
            assert leaf.shape == KEY_SPECS[algorithm].shape


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_split_n_bit_exact(algorithm):
    """``split_n(k, 3)`` -> a 3-tuple of keys, bit-exact vs numpy."""
    exe = etl.build(_split_n3, KEY_SPECS[algorithm], backend="iree")
    for key in KEYS[algorithm]:
        want = etl.evaluate(_split_n3, key)
        got = etl.run(exe, key)
        assert len(got) == 3
        _assert_exact(got, want)


# ---------------------------------------------------------------------------
# 4. native THREE_FRY path (exporter option rng_bit_generator=True + a
#    manual LoweredProgram) — words must match numpy AND the inline path
# ---------------------------------------------------------------------------

NATIVE_CASES = [
    ("native_uniform", _uniform5),
    ("native_randint", _randint8),
    ("native_permutation", _perm5),
]


# BUG(etl): the native THREE_FRY uniform/randint ids share the same
# `_emit_u_scaled32` scaling bug — their words ARE bit-identical to the
# inline cipher on iree (the "native == inline" assertion below holds) but
# the shared u-scaling tail is wrong, so both differ from numpy by the
# 2^-32 factor (native uniform == inline uniform == numpy * 2^-32; native
# randint all zeros). The permutation id is bit-exact (order-based). Do NOT
# skip/xfail/weaken.
@pytest.mark.parametrize(
    "case_id,fn",
    [(c[0], c[1]) for c in NATIVE_CASES],
    ids=[c[0] for c in NATIVE_CASES],
)
def test_native_threefry_rng_bit_generator(case_id, fn):
    """The manual recipe: trace -> export with ``rng_bit_generator=True`` ->
    LoweredProgram -> iree adapter compile/load -> ``exe.run([key_tensor])``
    (flat input list). Native output must be bit-exact vs ``etl.evaluate``
    AND bit-identical to the inline-path output."""
    spec = KEY_SPECS["threefry2x32"]
    graph = etl.trace(fn, spec)
    mlir = etl.backends.stablehlo.export(
        graph, options={"rng_bit_generator": True}
    )
    assert "rng_bit_generator" in mlir
    adapter = etl.backends.registry.get("iree")
    artifact = adapter.compile(_lowered_native(graph))
    exe = adapter.load(artifact)
    inline_exe = etl.build(fn, spec, backend="iree")  # the inline path
    for key in KEYS["threefry2x32"]:
        want = etl.evaluate(fn, key)
        got = exe.run([etl.core.Tensor(key.copy())])  # list of flat Tensors
        _assert_exact(got, want)
        _assert_exact(got, etl.run(inline_exe, key))  # native == inline
        if "permutation" in case_id:
            perm = _np(got[0])  # the flat-output list's single tensor
            assert sorted(perm.tolist()) == list(range(perm.shape[-1]))


# ---------------------------------------------------------------------------
# 5. determinism across separate run calls / round-trips
# ---------------------------------------------------------------------------


def test_cross_roundtrip_determinism():
    """Two separate etl.build + etl.run round-trips (same key) give
    bit-identical output — the hard same-key determinism contract."""
    for algorithm in ALGORITHMS:
        key = KEYS[algorithm][-1]  # the seed-derived key
        spec = KEY_SPECS[algorithm]
        exe1 = etl.build(_uniform5, spec, backend="iree")
        exe2 = etl.build(_uniform5, spec, backend="iree")
        _assert_exact(etl.run(exe1, key), etl.run(exe2, key))


# ---------------------------------------------------------------------------
# 6. iree-cuda smoke set (device 5; while-loop graphs not involved here)
# ---------------------------------------------------------------------------


# BUG(etl): same `_emit_u_scaled32` scaling violation on cuda — the inline
# uniform paths run (the compile itself is fine) but produce numpy * 2^-32
# (threefry) / the full-word-scaled mismatch (philox), and the native
# THREE_FRY uniform likewise; see the marker on test_inline_bit_exact for
# the repro values. Do NOT skip/xfail/weaken.
def test_iree_cuda_threefry_inline_uniform_smoke():
    _require_cuda()
    _parity(
        _uniform5, KEY_SPECS["threefry2x32"], KEYS["threefry2x32"][-1],
        exact=True, device=etl.core.Device("cuda", 5),
        target_backends=["cuda"],
    )


def test_iree_cuda_philox_inline_uniform_smoke():
    _require_cuda()
    _parity(
        _uniform5, KEY_SPECS["philox4x32_10"], KEYS["philox4x32_10"][-1],
        exact=True, device=etl.core.Device("cuda", 5),
        target_backends=["cuda"],
    )


def test_iree_cuda_threefry_native_uniform_smoke():
    """Native THREE_FRY on cuda: the manual LoweredProgram recipe with
    ``target_backends=["cuda"]`` + ``load(device=cuda 5)``."""
    _require_cuda()
    spec = KEY_SPECS["threefry2x32"]
    key = KEYS["threefry2x32"][-1]
    graph = etl.trace(_uniform5, spec)
    adapter = etl.backends.registry.get("iree")
    artifact = adapter.compile(
        _lowered_native(graph), options={"target_backends": ["cuda"]}
    )
    exe = adapter.load(artifact, device=etl.core.Device("cuda", 5))
    want = etl.evaluate(_uniform5, key)
    got = exe.run([etl.core.Tensor(key.copy())])
    _assert_exact(got, want)
