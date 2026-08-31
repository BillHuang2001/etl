"""iree-llvm-cpu round-trip parity for the algorithm-aware RNG lowering
(Threefry2x32 / Philox4x32_10 — merged with commit 119854b).

Covered here (see ``etl/backends/stablehlo/random_export.py``):

* the INLINE threefry2x32/philox4x32_10 word-stream expansions — on iree
  these are now pinned EXPLICITLY via the per-call lower override
  ``rng_bit_generator=frozenset()``, because the iree adapter declares
  ``Capabilities.rng_bit_generator=frozenset({"threefry2x32"})``: native
  THREE_FRY is the DEFAULT path for threefry graphs (bit-exact on
  llvm-cpu + cuda, ~1.6-2.1x faster — see ``etl/bench/rng_bench.py``),
  while PHILOX stays inline (iree 3.11 fails to legalize RNG_ALG_PHILOX —
  an explicit ``BackendError`` at compile, never a silent fallback);
* the ADAPTER-DEFAULT native ``stablehlo.rng_bit_generator`` THREE_FRY
  path — the exporter option accepts a bool (``True`` → both ciphers,
  backward compat) or a collection of algorithm names, and the native
  emission is selected PER-ALGORITHM iff the name is in the set (the
  per-call override flows through ``etl.lower``/``etl.build``/
  ``etl.evaluate``); and
* the NATIVE path via a manual ``LoweredProgram`` (exporter option
  ``{"rng_bit_generator": True}``), whose iree-legalized words were
  empirically verified to match the inline cipher.

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

The xla adapter declares
``Capabilities.rng_bit_generator=frozenset({"threefry2x32",
"philox4x32_10"})``, but xla run tests stay gate-skipped without a
user-provided PJRT plugin (``ETL_PJRT_PLUGIN``/``plugin_path``) — there
are NO xla tests here.

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
    ``rng_bit_generator`` MLIR (``True`` → both ciphers). For threefry
    this now matches the iree adapter's DEFAULT lowering; the manual
    recipe is kept as the explicit-option form (and lets the cuda native
    smoke test control ``target_backends`` at compile)."""
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
# 1. inline uniform/randint/permutation — pinned via the per-call
#    ``rng_bit_generator=frozenset()`` override (the iree adapter default
#    for threefry is now native THREE_FRY) — bit-exact vs numpy, one
#    compile per (case, algorithm), every key looped through the same exe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_id,fn,spec,algorithm",
    [(c[0], c[1], c[2], c[3]) for c in INLINE_PARAMS],
    ids=[c[0] for c in INLINE_PARAMS],
)
def test_inline_bit_exact(case_id, fn, spec, algorithm):
    """One iree compile per (case, algorithm); every key in the algorithm's
    list runs through the same executable. The INLINE path is pinned via
    the per-call ``rng_bit_generator=frozenset()`` override (for threefry
    the adapter default is now native THREE_FRY). Bit-exact vs
    ``etl.evaluate`` (numpy reference) AND bit-identical across repeated
    runs of the same executable (determinism folded into the loop)."""
    exe = etl.build(fn, spec, backend="iree", rng_bit_generator=frozenset())
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

@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_normal_f64_bit_exact(algorithm):
    """f64 output normal: bit-exact vs the numpy kernel (both algorithms;
    the INLINE path, pinned via the per-call override)."""
    exe = etl.build(
        _normal_f64, KEY_SPECS[algorithm], backend="iree",
        rng_bit_generator=frozenset(),
    )
    for key in KEYS[algorithm]:
        want = etl.evaluate(_normal_f64, key)
        got = etl.run(exe, key)
        assert _np(got).dtype == np.float64
        _assert_exact(got, want)


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_normal_f32_tolerance_and_determinism(algorithm):
    """f32 output normal: within the documented f32 fast-path budget vs
    numpy AND bit-identical across two separate runs of the same exe (the
    INLINE path, pinned via the per-call override)."""
    exe = etl.build(
        _normal_f32, KEY_SPECS[algorithm], backend="iree",
        rng_bit_generator=frozenset(),
    )
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

@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_split_bit_exact(algorithm):
    """``split(k)`` -> two keys of the algorithm's key shape ((2,) / (4,)
    int32), bit-exact vs numpy (the INLINE path, pinned via the per-call
    override)."""
    exe = etl.build(
        _split, KEY_SPECS[algorithm], backend="iree",
        rng_bit_generator=frozenset(),
    )
    for key in KEYS[algorithm]:
        want = etl.evaluate(_split, key)
        got = etl.run(exe, key)
        _assert_exact(got, want)
        for leaf in etl.tree_leaves(etl.tree_map(_np, got)):
            assert leaf.dtype == np.int32
            assert leaf.shape == KEY_SPECS[algorithm].shape


@pytest.mark.parametrize("algorithm", ALGORITHMS, ids=ALGORITHMS)
def test_split_n_bit_exact(algorithm):
    """``split_n(k, 3)`` -> a 3-tuple of keys, bit-exact vs numpy (the
    INLINE path, pinned via the per-call override)."""
    exe = etl.build(
        _split_n3, KEY_SPECS[algorithm], backend="iree",
        rng_bit_generator=frozenset(),
    )
    for key in KEYS[algorithm]:
        want = etl.evaluate(_split_n3, key)
        got = etl.run(exe, key)
        assert len(got) == 3
        _assert_exact(got, want)


# ---------------------------------------------------------------------------
# 4. adapter-default native THREE_FRY (iree capability) — vs the forced
#    inline override, and the native-PHILOX compile failure (never silent)
# ---------------------------------------------------------------------------

def _lowered_mlir(fn, spec, **lower_kwargs) -> str:
    """Lower via the iree adapter and return the recorded StableHLO MLIR."""
    lowered = etl.lower(etl.trace(fn, spec), backend="iree", **lower_kwargs)
    return lowered.payload["mlir_text"]


def test_adapter_default_threefry_native_bit_exact():
    """WITHOUT any override the iree adapter default capability
    (``rng_bit_generator=frozenset({"threefry2x32"})``) lowers a threefry
    graph to a NATIVE ``stablehlo.rng_bit_generator`` call (algorithm
    THREE_FRY) — and that default path runs bit-exact vs
    ``etl.evaluate``."""
    spec = KEY_SPECS["threefry2x32"]
    mlir = _lowered_mlir(_uniform5, spec)
    rng_line = next(
        line for line in mlir.splitlines()
        if "stablehlo.rng_bit_generator" in line
    )
    assert "algorithm = THREE_FRY" in rng_line
    exe = etl.build(_uniform5, spec, backend="iree")
    for key in KEYS["threefry2x32"]:
        _assert_exact(etl.run(exe, key), etl.evaluate(_uniform5, key))


def test_inline_override_flows_to_lowering():
    """The per-call ``rng_bit_generator`` lower override reaches the
    exporter: ``frozenset()`` forces the INLINE path for a threefry graph
    even though the iree adapter default would emit the native call (the
    default MLIR contains ``stablehlo.rng_bit_generator``, the overridden
    MLIR does not)."""
    spec = KEY_SPECS["threefry2x32"]
    assert "stablehlo.rng_bit_generator" in _lowered_mlir(_uniform5, spec)
    forced = _lowered_mlir(_uniform5, spec, rng_bit_generator=frozenset())
    assert "rng_bit_generator" not in forced


def test_adapter_default_philox_inline_bit_exact():
    """The iree capability excludes philox, so a philox graph lowers to
    the INLINE expansion by default (no ``stablehlo.rng_bit_generator`` in
    the MLIR) and runs bit-exact."""
    spec = KEY_SPECS["philox4x32_10"]
    assert "rng_bit_generator" not in _lowered_mlir(_uniform5, spec)
    exe = etl.build(_uniform5, spec, backend="iree")
    for key in KEYS["philox4x32_10"]:
        _assert_exact(etl.run(exe, key), etl.evaluate(_uniform5, key))


def test_adapter_native_philox_override_fails_at_compile():
    """Forcing native PHILOX on iree (``rng_bit_generator=
    {"philox4x32_10"}`` — absent from the capability set) emits the native
    call and fails CLEANLY at compile: iree cannot legalize RNG_ALG_PHILOX,
    surfaced as an explicit ``BackendError`` carrying the compiler
    diagnostics — never a silent fallback."""
    with pytest.raises(etl.BackendError) as excinfo:
        etl.build(
            _uniform5, KEY_SPECS["philox4x32_10"], backend="iree",
            rng_bit_generator={"philox4x32_10"},
        )
    msg = str(excinfo.value)
    assert "legalize" in msg
    assert "rng_bit_generator" in msg


# ---------------------------------------------------------------------------
# 5. native THREE_FRY path (exporter option rng_bit_generator=True + a
#    manual LoweredProgram — for threefry now the adapter default too) —
#    words must match numpy AND the inline path
# ---------------------------------------------------------------------------

NATIVE_CASES = [
    ("native_uniform", _uniform5),
    ("native_randint", _randint8),
    ("native_permutation", _perm5),
]


@pytest.mark.parametrize(
    "case_id,fn",
    [(c[0], c[1]) for c in NATIVE_CASES],
    ids=[c[0] for c in NATIVE_CASES],
)
def test_native_threefry_rng_bit_generator(case_id, fn):
    """The manual recipe: trace -> export with ``rng_bit_generator=True`` ->
    LoweredProgram -> iree adapter compile/load -> ``exe.run([key_tensor])``
    (flat input list). Native output must be bit-exact vs ``etl.evaluate``
    AND bit-identical to the inline-path output (the inline path is pinned
    via the per-call override — the adapter default is now native)."""
    spec = KEY_SPECS["threefry2x32"]
    graph = etl.trace(fn, spec)
    mlir = etl.backends.stablehlo.export(
        graph, options={"rng_bit_generator": True}
    )
    assert "rng_bit_generator" in mlir
    adapter = etl.backends.registry.get("iree")
    artifact = adapter.compile(_lowered_native(graph))
    exe = adapter.load(artifact)
    # the inline path, pinned via the per-call override
    inline_exe = etl.build(fn, spec, backend="iree", rng_bit_generator=frozenset())
    for key in KEYS["threefry2x32"]:
        want = etl.evaluate(fn, key)
        got = exe.run([etl.core.Tensor(key.copy())])  # list of flat Tensors
        _assert_exact(got, want)
        _assert_exact(got, etl.run(inline_exe, key))  # native == inline
        if "permutation" in case_id:
            perm = _np(got[0])  # the flat-output list's single tensor
            assert sorted(perm.tolist()) == list(range(perm.shape[-1]))


# ---------------------------------------------------------------------------
# 6. determinism across separate run calls / round-trips
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
# 7. iree-cuda smoke set (device 5; while-loop graphs not involved here)
# ---------------------------------------------------------------------------


def test_iree_cuda_threefry_inline_uniform_smoke():
    """INLINE threefry on cuda — pinned via the per-call override (the
    adapter default on cuda is native THREE_FRY)."""
    _require_cuda()
    _parity(
        _uniform5, KEY_SPECS["threefry2x32"], KEYS["threefry2x32"][-1],
        exact=True, device=etl.core.Device("cuda", 5),
        target_backends=["cuda"], rng_bit_generator=frozenset(),
    )


def test_iree_cuda_philox_inline_uniform_smoke():
    """PHILOX on cuda stays on the bit-exact INLINE path (pinned via the
    override for explicitness — the iree capability has no native PHILOX)."""
    _require_cuda()
    _parity(
        _uniform5, KEY_SPECS["philox4x32_10"], KEYS["philox4x32_10"][-1],
        exact=True, device=etl.core.Device("cuda", 5),
        target_backends=["cuda"], rng_bit_generator=frozenset(),
    )


def test_iree_cuda_threefry_native_uniform_smoke():
    """Native THREE_FRY on cuda — the manual LoweredProgram recipe with
    ``target_backends=["cuda"]`` + ``load(device=cuda 5)`` (this is also
    the adapter default path on cuda)."""
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
