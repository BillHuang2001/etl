"""Count-mode argsort parity + the cuda 2-operand-sort bufferization
regression (fix 71d721e).

iree 3.11.0 cannot bufferize the two-operand ``stablehlo.sort`` (the pair
argsort emission) on cuda whenever the sorted-axis extent >= 32 — a
deterministic upstream bug (every iota dtype / is_stable / operand-order
variant fails; llvm-cpu is unaffected). The ``sort_emission="count"``
exporter option composes argsort WITHOUT ``stablehlo.sort``: two
broadcast-compare rank sums + an iota ``k < j`` tie-break + ``reduce(add)``
over k; the inverse permutation via a pos-EQ mask + select-sentinel(=n) +
``reduce(min)`` (see ``writer._emit_count_argsort``). ``"auto"`` (the iree
adapter's default) picks count per argsort whenever the sorted-axis extent
>= 32 and keeps the pair form otherwise.

The count composition is INHERENTLY STABLE: it is bit-exact vs
``np.argsort(kind="stable")`` on llvm-cpu AND cuda. (The numpy backend's
default ``stable=False`` uses quicksort, which may order ties differently —
both are valid argsorts; the count-mode contract is pinned to the stable
reference below.)

Pinned here:

* ``sort_emission="count"`` emits NO ``stablehlo.sort`` (iota/compare/
  reduce composition instead); ``"pair"`` still does,
* ``"auto"`` picks count iff sorted-axis extent >= 32 (and the iree adapter
  default is ``"auto"`` — a no-option iree lower of an axis-64 argsort
  carries no sort op),
* llvm-cpu parity: ascending / descending / tie-heavy / axis-0, all
  bit-exact vs the stable numpy references,
* the upstream-bug path: argsort axis 64 bit-exact on iree-cuda
  (GPU-guarded via an nvidia-smi free-device scan). Cuda runs use the
  explicit-device-placement semantics: the host input is uploaded via
  ``Tensor.to`` BEFORE ``etl.run`` (a cuda executable rejects host inputs
  at the run boundary — no implicit device↔host transfer ever happens
  there), and the device-resident output is transferred back to host
  explicitly (``Tensor.numpy()`` on a cuda payload is a ``DeviceError``).
"""

import numpy as np
import pytest

pytest.importorskip("iree.compiler")
pytest.importorskip("iree.runtime")

import etl

# ---------------------------------------------------------------------------
# graphs + data (fixed seed — same convention as test_iree_emitters_parity.py)
# ---------------------------------------------------------------------------


def _argsort_axis1(x):
    return etl.argsort(x, axis=1)


def _argsort_axis1_desc(x):
    return etl.argsort(x, axis=1, descending=True)


def _argsort_axis0(x):
    return etl.argsort(x, axis=0)


def _argsort_axis1_stable(x):
    return etl.argsort(x, axis=1, stable=True)


_RNG = np.random.default_rng(7)

# no ties: fp32 noise — every argsort (stable or not) agrees here.
_XF = _RNG.standard_normal((8, 64)).astype(np.float32)
# tie-heavy: 5 distinct int values over 64 slots.
_XI = _RNG.integers(0, 5, size=(8, 64)).astype(np.int32)

SPEC_F = etl.TensorSpec((8, 64), etl.float32)
SPEC_I = etl.TensorSpec((8, 64), etl.int32)
SPEC_I16 = etl.TensorSpec((8, 16), etl.int32)

#: (id, fn, spec, arg, numpy-reference) — the reference is the STABLE numpy
#: argsort (the count-mode contract); descending = flip of the stable
#: ascending indices (the numpy composition).
PARITY_CASES = [
    ("ascending_fp32_no_ties", _argsort_axis1, SPEC_F, _XF,
     np.argsort(_XF, axis=1, kind="stable")),
    ("descending_fp32_no_ties", _argsort_axis1_desc, SPEC_F, _XF,
     np.flip(np.argsort(_XF, axis=1, kind="stable"), axis=1)),
    ("ascending_tie_heavy_int32", _argsort_axis1, SPEC_I, _XI,
     np.argsort(_XI, axis=1, kind="stable")),
    ("axis0_tie_heavy_int32", _argsort_axis0, SPEC_I, _XI,
     np.argsort(_XI, axis=0, kind="stable")),
    # the etl-contract path: stable=True must match the numpy backend too.
    ("stable_flag_tie_heavy", _argsort_axis1_stable, SPEC_I, _XI,
     np.argsort(_XI, axis=1, kind="stable")),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np(v):
    """Tensor / tuple-of-Tensors → ndarray / tuple-of-ndarrays."""
    if isinstance(v, etl.Tensor):
        return np.asarray(v.numpy())
    return np.asarray(v)


def _assert_exact(got, want):
    g, w = etl.tree_map(_np, got), etl.tree_map(_np, want)
    for gp, wp in zip(etl.tree_leaves(g), etl.tree_leaves(w)):
        assert gp.shape == wp.shape
        assert np.array_equal(gp, wp), f"{gp} != {wp}"


def _export_argsort(fn, spec, **options):
    """Trace + export an argsort graph with the given exporter options."""
    from etl.backends.stablehlo import export

    return export(etl.trace(fn, spec), options=options)


def _pick_cuda_device_index():
    """Most-free GPU index via nvidia-smi; ``pytest.skip`` when unavailable."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not found — no CUDA device to test")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"nvidia-smi failed: {exc}")
    if proc.returncode != 0:
        pytest.skip(f"nvidia-smi failed: {proc.stderr.strip()}")
    gpus = []
    for line in proc.stdout.strip().splitlines():
        try:
            idx, free_mib = (part.strip() for part in line.split(","))
            gpus.append((int(free_mib), int(idx)))
        except ValueError:
            continue  # malformed line — ignore
    if not gpus:
        pytest.skip("nvidia-smi reported no GPUs")
    gpus.sort(reverse=True)
    return gpus[0][1]


@pytest.fixture(scope="module")
def cuda_device():
    """A free CUDA device (most-free GPU via nvidia-smi); skip when unavailable."""
    idx = _pick_cuda_device_index()
    import iree.runtime as rt

    try:
        # etl Device("cuda", idx) maps to iree device_id idx + 1 (1-based ids).
        rt.get_driver("cuda").create_device(device_id=idx + 1)
    except Exception as exc:  # noqa: BLE001 — any driver/device failure skips
        pytest.skip(f"IREE cuda HAL driver or GPU {idx} unavailable: {exc}")
    return etl.core.Device("cuda", idx)


# ---------------------------------------------------------------------------
# 1. emission-shape assertions (no stablehlo.sort in count / auto >= 32)
# ---------------------------------------------------------------------------


def test_count_mode_emits_no_sort_op():
    mlir = _export_argsort(_argsort_axis1, SPEC_I, sort_emission="count")
    assert "stablehlo.sort" not in mlir, "count mode must not emit stablehlo.sort"
    for op in ("stablehlo.iota", "stablehlo.compare", '"stablehlo.reduce"'):
        assert op in mlir, f"count composition missing {op}"
    mlir_pair = _export_argsort(_argsort_axis1, SPEC_I, sort_emission="pair")
    assert "stablehlo.sort" in mlir_pair, "pair mode must emit stablehlo.sort"


def test_auto_mode_picks_count_iff_extent_ge_32():
    big = _export_argsort(_argsort_axis1, SPEC_I, sort_emission="auto")
    assert "stablehlo.sort" not in big, "auto must pick count at axis extent 64"
    small = _export_argsort(_argsort_axis1, SPEC_I16, sort_emission="auto")
    assert "stablehlo.sort" in small, "auto must keep pair at axis extent 16"
    # the iree adapter defaults to "auto": a no-option lower of the axis-64
    # argsort carries the count composition (no sort op) in its payload.
    lowered = etl.lower(etl.trace(_argsort_axis1, SPEC_I), backend="iree")
    assert "stablehlo.sort" not in lowered.payload["mlir_text"]


# ---------------------------------------------------------------------------
# 2. llvm-cpu parity: count mode bit-exact vs the stable numpy references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,spec,arg,ref",
    [(fn, spec, arg, ref) for _, fn, spec, arg, ref in PARITY_CASES],
    ids=[c[0] for c in PARITY_CASES],
)
def test_count_mode_parity_llvm_cpu(fn, spec, arg, ref):
    exe = etl.build(fn, spec, backend="iree", sort_emission="count")
    _assert_exact(etl.run(exe, arg), ref)


# ---------------------------------------------------------------------------
# 3. iree-cuda (GPU-guarded): the upstream-bug path — argsort axis 64
# ---------------------------------------------------------------------------

CUDA_CASES = [
    ("cuda_argsort_axis64_fp32", _argsort_axis1, SPEC_F, _XF,
     np.argsort(_XF, axis=1, kind="stable")),
    ("cuda_argsort_axis64_tie_heavy", _argsort_axis1, SPEC_I, _XI,
     np.argsort(_XI, axis=1, kind="stable")),
]


@pytest.mark.parametrize(
    "fn,spec,arg,ref",
    [(fn, spec, arg, ref) for _, fn, spec, arg, ref in CUDA_CASES],
    ids=[c[0] for c in CUDA_CASES],
)
def test_cuda_argsort_axis64(fn, spec, arg, ref, cuda_device):
    """The 2-operand sort at axis >= 32 fails to bufferize on iree-cuda
    (upstream); the count emission (the iree default "auto" here) must run
    bit-exact.

    Explicit device placement (run-boundary contract): the host input is
    placed on the cuda device via ``Tensor.to`` BEFORE ``etl.run`` — a
    cuda executable never stages host inputs, so raw numpy arrays (cpu:0
    tensors) are rejected at the run boundary. The output is a
    device-resident payload tensor: ``.numpy()`` raises ``DeviceError``,
    and the explicit ``to(Device('cpu', 0))`` transfer is the only host
    path used for the bit-exact comparison.
    """
    exe = etl.build(
        fn, spec, backend="iree", device=cuda_device, target_backends=["cuda"]
    )
    # Place the input on the executable's device first (run never stages
    # host inputs; scalars would need the same placement).
    arg_dev = etl.core.Tensor(arg).to(cuda_device)
    got = etl.run(exe, arg_dev)
    # Device-resident output: no implicit device-to-host transfer — .numpy()
    # on a cuda payload tensor is a DeviceError.
    with pytest.raises(etl.core.DeviceError):
        got.numpy()
    got_host = etl.tree_map(lambda t: t.to(etl.core.Device("cpu", 0)), got)
    _assert_exact(got_host, ref)
