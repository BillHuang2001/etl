"""Compiler-backend fallback for external kernels without host-dispatch.

Pins the v1 fallback semantics documented in ``etl/backends/CONTEXT.md``
("External kernels on compiler backends") and implemented by the shared
``CompilerBackend.lower`` + ``etl.backends.inline.inline_portables``
(``inline_external_portables=True``): a backend WITHOUT host-dispatch (v1:
xla/tvm — ``Capabilities.external_calls=False``) cannot execute
``external_call`` ops, so an op whose kernel name has a registered portable
decomposition (``etl.external.get_portable``) is INLINED at ``lower()``
through the shared portable-splicing machinery — with a ``UserWarning`` —
instead of raising.

Pinned here (tvm is the in-env backend without host-dispatch; the fallback
is backend-neutral shared code):

* portable registered => ``etl.lower(..., backend="tvm")`` succeeds, the
  payload is the ordinary StableHLO contract with the portable's body
  inlined (``stablehlo.multiply`` present, no ``external_call`` text), and a
  ``UserWarning`` naming the kernel fires.
* end-to-end: the inlined graph compiles + runs on tvm with numpy parity
  (the numpy reference dispatches the registered default-slot kernel).
* warning dedupe (the documented once-per-kernel-name behavior — Python's
  default once-per-location filter is the dedupe mechanism):
  two identical-name ops in one graph surface ONE user-visible warning
  (with ``simplefilter("always")`` the raw emission is once per op, all
  carrying the SAME canonical message), and repeated ``lower()`` calls from
  the same call site surface ONE warning.
* op WITHOUT a portable => the upgraded explicit ``BackendError`` naming
  ``external_call``, ``host-dispatch``, the backend and the portable option
  (never a silent fallback).
* portable with a WRONG result spec (count/dtype/shape mismatch vs the op's
  declared ``result_specs``) => ``BackendError`` from the shared output
  validation (never a silently accepted mismatch).
* the numpy backend is unchanged: it still dispatches the registered
  default-slot kernel (no portable inlining, no fallback warning).

TVM compiles for real (a few seconds), so the 2s-per-file convention does
not apply; the module keeps the compile count to ONE (the run-parity test).
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("tvm")
# The tvm adapter also needs jaxlib (ONLY for its bundled LLVM MLIR python
# bindings used by apache-tvm's StableHLO translator — the jax frontend is
# never imported). Skip cleanly when tvm is present but jaxlib is not.
pytest.importorskip("jaxlib")

import etl

# ---------------------------------------------------------------------------
# kernels + portables (plain numpy stand-ins; the portables are @etl.defn
# graph decompositions matching the kernels' outputs)
# ---------------------------------------------------------------------------


@etl.defn
def _portable_double(x):
    return x * 2.0


@etl.defn
def _portable_inc(x):
    return x + 1.0


KERNELS = {
    # (x * 2) — exactly representable fp32
    "p_kernel_double": lambda x: x * 2.0,
    # (x + 1) — exactly representable fp32
    "p_kernel_inc": lambda x: x + 1.0,
    # kernel WITHOUT a portable — the fallback must reject it
    "p_kernel_no_portable": lambda x: x * 3.0,
}

PORTABLES = {
    "p_kernel_double": _portable_double,
    "p_kernel_inc": _portable_inc,
}


@pytest.fixture(scope="module", autouse=True)
def _register_kernels():
    for name, kernel in KERNELS.items():
        etl.register_external_kernel(name, kernel)
    for name, portable in PORTABLES.items():
        etl.register_portable(name, portable)
    try:
        yield
    finally:
        for name in KERNELS:
            etl.unregister_external_kernel(name)


# ---------------------------------------------------------------------------
# data + graph builders
# ---------------------------------------------------------------------------

X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
SPEC = etl.TensorSpec((3,), etl.float32)


def _single_call(x):
    return etl.external_call("p_kernel_double", x, result=SPEC)


def _two_calls_same_name(x):
    y = etl.external_call("p_kernel_double", x, result=SPEC)
    return etl.external_call("p_kernel_double", y, result=SPEC)


def _no_portable_call(x):
    return etl.external_call("p_kernel_no_portable", x, result=SPEC)


def _wrong_count_call(x):
    return etl.external_call("p_kernel_double", x, result=[SPEC, SPEC])


def _wrong_dtype_call(x):
    return etl.external_call(
        "p_kernel_double", x, result=etl.TensorSpec((3,), etl.float64)
    )


def _wrong_shape_call(x):
    return etl.external_call(
        "p_kernel_double", x, result=etl.TensorSpec((4,), etl.float32)
    )


# ---------------------------------------------------------------------------
# portable-backed fallback: lower + end-to-end parity
# ---------------------------------------------------------------------------


def test_tvm_lower_inlines_portable_with_warning():
    """lower() succeeds for a portable-backed external_call: the portable's
    body is inlined into the exported StableHLO and a UserWarning naming the
    kernel explains the fallback (no host-dispatch on the tvm backend)."""
    graph = etl.trace(_single_call, SPEC)
    with pytest.warns(UserWarning, match="p_kernel_double"):
        lowered = etl.lower(graph, backend="tvm")
    assert lowered.backend == "tvm"
    assert lowered.payload["format"] == "stablehlo"
    mlir = lowered.payload["mlir_text"]
    assert "stablehlo.multiply" in mlir  # the inlined portable body (x * 2)
    assert "external_call" not in mlir  # inlined away — never exported


def test_tvm_run_parity_with_numpy():
    """The inlined portable compiles + runs on tvm with numpy parity (the
    numpy reference dispatches the registered default-slot kernel)."""
    exe = etl.build(_single_call, SPEC, backend="tvm")
    got = etl.run(exe, etl.Tensor(X))
    ref = etl.evaluate(_single_call, X)  # numpy backend: kernel dispatch
    np.testing.assert_allclose(
        got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.array_equal(got.numpy(), X * 2.0)  # x*2 is exact in fp32


# ---------------------------------------------------------------------------
# warning dedupe (the documented once-per-kernel-name behavior)
# ---------------------------------------------------------------------------


def test_warning_once_for_two_identical_name_ops():
    """Two identical-name external_calls in one graph surface ONE
    user-visible warning (Python's default once-per-location filter is the
    documented dedupe). With ``simplefilter("always")`` the raw emission is
    once per op — the dedupe is the default filter, never per-op silence —
    and every record carries the SAME canonical message."""
    with pytest.warns(UserWarning, match="p_kernel_double"):
        etl.lower(etl.trace(_two_calls_same_name, SPEC), backend="tvm")

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        etl.lower(etl.trace(_two_calls_same_name, SPEC), backend="tvm")
    matching = [
        r for r in records
        if r.category is UserWarning and "p_kernel_double" in str(r.message)
    ]
    assert len(matching) == 2  # raw emission: once per inlined op
    assert len({str(r.message) for r in matching}) == 1  # ONE canonical message

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("default")  # Python's default per-location dedupe
        etl.lower(etl.trace(_two_calls_same_name, SPEC), backend="tvm")
    matching = [
        r for r in records
        if r.category is UserWarning and "p_kernel_double" in str(r.message)
    ]
    assert len(matching) == 1  # user-visible: once per name


def test_warning_deduped_across_repeated_lower_calls():
    """Repeated lower() calls for the same kernel name surface ONE
    user-visible warning: both emits share the same call-site location, so
    the default filter dedupes them."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("default")
        for _ in range(2):
            etl.lower(etl.trace(_single_call, SPEC), backend="tvm")
    matching = [
        r for r in records
        if r.category is UserWarning and "p_kernel_double" in str(r.message)
    ]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# explicit errors (never a silent fallback)
# ---------------------------------------------------------------------------


def test_tvm_rejects_without_portable():
    """An external_call WITHOUT a registered portable raises the upgraded
    BackendError naming the op, host-dispatch, the backend and the portable
    option — never a silent fallback."""
    graph = etl.trace(_no_portable_call, SPEC)
    with pytest.raises(etl.BackendError) as excinfo:
        etl.lower(graph, backend="tvm")
    msg = str(excinfo.value)
    assert "external_call" in msg
    assert "host-dispatch" in msg
    assert "portable" in msg
    assert "tvm" in msg


@pytest.mark.parametrize(
    ("builder", "needles"),
    [
        # count mismatch: portable produces 1 result, op declares 2 specs
        (_wrong_count_call, ("external_call", "result spec(s)")),
        # dtype mismatch: portable f32 vs declared f64
        (_wrong_dtype_call, ("dtype",)),
        # shape mismatch: portable (3,) vs declared (4,)
        (_wrong_shape_call, ("shape",)),
    ],
    ids=["count", "dtype", "shape"],
)
def test_tvm_wrong_result_spec_validated(builder, needles):
    """A portable whose outputs disagree with the op's declared result specs
    (count/dtype/shape) is rejected at lower() by the shared output
    validation — never a silently accepted mismatch."""
    graph = etl.trace(builder, SPEC)
    with pytest.raises(etl.BackendError) as excinfo:
        etl.lower(graph, backend="tvm")
    msg = str(excinfo.value)
    for needle in needles:
        assert needle in msg


# ---------------------------------------------------------------------------
# numpy backend unchanged (spot check)
# ---------------------------------------------------------------------------


def test_numpy_backend_dispatch_unchanged():
    """The numpy backend still dispatches the registered default-slot kernel:
    no portable inlining, no fallback warning."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        out = etl.evaluate(_single_call, X)
    assert np.array_equal(out.numpy(), X * 2.0)
    assert not [
        r for r in records
        if r.category is UserWarning and "p_kernel_double" in str(r.message)
    ]
