"""Contract tests for the external-kernel mechanism (``etl.external_call`` +
``etl.external`` registry).

Assertions on the contract documented in ``etl/CONTEXT.md`` ("External
kernels") and ``etl/ops/external.py``:

- Trace-time: ``etl.external_call(name, *operands, result=...)`` builds an
  ``external_call`` IR op (effect ``callback``) with attrs ``name`` +
  ``result_specs``; result shapes/dtypes are the declared specs (static
  contract); registration is NOT required at trace time.
- Numpy execution: the kernel is dispatched by name with numpy-array
  operands; its outputs are validated against the declared specs
  (count/dtype/shape — never silent coercion) and flow into downstream ops.
- Registry: register/get/unregister semantics (overwrite allowed,
  ``KeyError`` on unknown unregister, ``TypeError`` on bad arguments).
- Device-resident registration mode: ``register_external_kernel(name, fn,
  backend, device_resident=True)`` stores a ``(callable, True)`` slot entry
  — ``device_resident`` must be an actual bool and requires an explicit
  per-backend slot; ``etl.external.get_external_kernel_entry`` is the
  mode-aware lookup (exact backend slot -> default slot -> ``None``,
  returning ``(callable, mode)`` tuples) while ``get_external_kernel``
  still returns the bare callable; re-registration replaces both the
  callable and the mode; ``ExternalKernel.impl`` takes direct/decorator
  forms; any non-``None`` backend string is accepted (backend-neutral —
  only the iree adapter consumes the mode in v1).
- Errors: unknown name at run time -> ``BackendError`` naming the kernel;
  spec mismatches -> ``BackendError``/``ShapeError``; vmap/grad/jvp/vjp ->
  ``TransformError``; the stablehlo exporter and the xla/tvm adapters ->
  ``BackendError`` (host-dispatch not wired); the iree adapter (round 2)
  SPLITS external-call graphs at lower() into segment programs
  (``stablehlo-segments`` payload) — end-to-end iree coverage lives in
  ``tests/backends/test_external_call_iree.py``.
- Control flow: the op works inside ``cond``/``while_loop`` bodies (numpy).
- Persistence: graphs round-trip through ``Graph.save``/``load``; the kernel
  must be re-registered in the process that runs a loaded graph.
"""
import numpy as np
import pytest

import etl

from tests.ops.conftest import ops_of, run_numpy


# ---------------------------------------------------------------------------
# Registry API (etl.external)
# ---------------------------------------------------------------------------


def test_register_get_roundtrip():
    def k(x):
        return x + 1

    etl.register_external_kernel("reg_roundtrip", k)
    try:
        assert etl.get_external_kernel("reg_roundtrip") is k
    finally:
        etl.unregister_external_kernel("reg_roundtrip")
    assert etl.get_external_kernel("reg_roundtrip") is None


def test_registration_overwrites_last_wins():
    def k1(x):
        return x

    def k2(x):
        return x * 2

    etl.register_external_kernel("reg_overwrite", k1)
    try:
        etl.register_external_kernel("reg_overwrite", k2)  # documented: no error
        assert etl.get_external_kernel("reg_overwrite") is k2
    finally:
        etl.unregister_external_kernel("reg_overwrite")


def test_unregister_unknown_raises_key_error():
    with pytest.raises(KeyError, match="no external kernel registered under 'ghost'"):
        etl.unregister_external_kernel("ghost")


def test_register_rejects_bad_arguments():
    with pytest.raises(TypeError, match="name must be a non-empty str"):
        etl.register_external_kernel(42, lambda x: x)
    with pytest.raises(TypeError, match="name must be a non-empty str"):
        etl.register_external_kernel("", lambda x: x)
    with pytest.raises(TypeError, match="must be callable"):
        etl.register_external_kernel("not_callable", 42)


def test_unregister_rejects_bad_name():
    with pytest.raises(TypeError, match="name must be a non-empty str"):
        etl.unregister_external_kernel(None)


def test_register_device_resident_requires_backend():
    """A device-resident kernel must live in an explicit per-backend slot:
    the default (``backend=None``) slot is also dispatched by the numpy
    backend, which passes host numpy arrays — never device tensors."""

    def k(x):
        return x

    with pytest.raises(TypeError, match="requires an explicit backend"):
        etl.register_external_kernel("dr_no_backend", k, device_resident=True)

    handle = etl.register_external_kernel("dr_no_backend_impl", k)
    try:
        with pytest.raises(TypeError, match="requires an explicit backend"):
            # Validation lives inside register_external_kernel.
            handle.impl(None, k, device_resident=True)
    finally:
        etl.unregister_external_kernel("dr_no_backend_impl")


def test_register_device_resident_rejects_non_bool():
    """``device_resident`` must be an actual bool — ``1``, ``"yes"`` and
    ``None`` are rejected loudly (``1`` is not a bool instance)."""

    def k(x):
        return x

    for bad in (1, "yes", None):
        with pytest.raises(TypeError, match="device_resident must be a bool"):
            etl.register_external_kernel(
                "dr_nonbool", k, backend="iree", device_resident=bad
            )

    # True/False are the only accepted values.
    etl.register_external_kernel("dr_bool_ok", k, backend="iree", device_resident=True)
    try:
        assert etl.external.get_external_kernel_entry("dr_bool_ok", "iree") == (k, True)
    finally:
        etl.unregister_external_kernel("dr_bool_ok")


def test_get_external_kernel_entry_tuple_semantics():
    """``get_external_kernel_entry`` is the mode-aware lookup — same slot
    resolution as ``get_external_kernel`` (exact backend slot, then the
    default slot, then ``None``) — returning ``(callable, device_resident)``
    tuples instead of the bare callable."""

    def host_fn(x):
        return x + 1

    def dev_fn(x):
        return x + 2

    # A plain default registration yields (fn, False) — also when looked up
    # under a backend string (default-slot fallback).
    etl.register_external_kernel("dr_entry_default", host_fn)
    try:
        assert etl.external.get_external_kernel_entry("dr_entry_default") == (host_fn, False)
        assert etl.external.get_external_kernel_entry("dr_entry_default", "iree") == (
            host_fn,
            False,
        )
    finally:
        etl.unregister_external_kernel("dr_entry_default")

    # A device registration yields (fn, True) in its backend slot.
    etl.register_external_kernel(
        "dr_entry_device", dev_fn, backend="iree", device_resident=True
    )
    try:
        assert etl.external.get_external_kernel_entry("dr_entry_device", "iree") == (
            dev_fn,
            True,
        )
    finally:
        etl.unregister_external_kernel("dr_entry_device")

    # An exact backend slot wins over the default slot.
    etl.register_external_kernel("dr_entry_both", host_fn)
    etl.register_external_kernel(
        "dr_entry_both", dev_fn, backend="iree", device_resident=True
    )
    try:
        assert etl.external.get_external_kernel_entry("dr_entry_both") == (host_fn, False)
        assert etl.external.get_external_kernel_entry("dr_entry_both", "iree") == (
            dev_fn,
            True,
        )
    finally:
        etl.unregister_external_kernel("dr_entry_both")

    # Unknown names resolve to None (never an error at lookup time).
    assert etl.external.get_external_kernel_entry("dr_entry_ghost") is None


def test_reregistration_overwrites_callable_and_mode():
    """Re-registering the same backend slot replaces BOTH the callable and
    the device-resident mode (last wins, no error) — in per-backend slots
    and the default slot alike."""

    def fn1(x):
        return x

    def fn2(x):
        return x + 1

    def fn3(x):
        return x + 2

    etl.register_external_kernel("dr_rereg", fn1, backend="iree")
    try:
        assert etl.external.get_external_kernel_entry("dr_rereg", "iree") == (fn1, False)
        etl.register_external_kernel("dr_rereg", fn2, backend="iree", device_resident=True)
        assert etl.external.get_external_kernel_entry("dr_rereg", "iree") == (fn2, True)
        etl.register_external_kernel("dr_rereg", fn3, backend="iree")  # host-mode again
        assert etl.external.get_external_kernel_entry("dr_rereg", "iree") == (fn3, False)
    finally:
        etl.unregister_external_kernel("dr_rereg")

    etl.register_external_kernel("dr_rereg_default", fn1)
    try:
        etl.register_external_kernel("dr_rereg_default", fn2)
        assert etl.external.get_external_kernel_entry("dr_rereg_default") == (fn2, False)
    finally:
        etl.unregister_external_kernel("dr_rereg_default")


def test_impl_direct_and_decorator_device_resident_forms():
    """The ``ExternalKernel.impl`` handle registers device kernels in both
    forms: the direct call returns ``fn``; the decorator registers on
    decoration and returns ``fn``."""

    def default_fn(x):
        return x

    def direct_fn(x):
        return x + 1

    handle = etl.register_external_kernel("dr_impl_forms", default_fn)
    try:
        returned = handle.impl("iree", direct_fn, device_resident=True)
        assert returned is direct_fn
        assert etl.external.get_external_kernel_entry("dr_impl_forms", "iree") == (
            direct_fn,
            True,
        )

        @handle.impl("iree", device_resident=True)
        def decorated_fn(x):
            return x + 2

        assert etl.external.get_external_kernel_entry("dr_impl_forms", "iree") == (
            decorated_fn,
            True,
        )
        assert etl.get_external_kernel("dr_impl_forms", "iree") is decorated_fn
        # The default slot is untouched by per-backend impl registration.
        assert etl.get_external_kernel("dr_impl_forms") is default_fn
    finally:
        etl.unregister_external_kernel("dr_impl_forms")


def test_device_resident_under_non_iree_backend_allowed():
    """Device registration is backend-neutral — any non-None backend string
    is accepted (only the iree adapter consumes the mode in v1)."""

    def k(x):
        return x

    for backend in ("numpy", "xla", "tvm"):
        name = f"dr_neutral_{backend}"
        etl.register_external_kernel(name, k, backend=backend, device_resident=True)
        try:
            assert etl.external.get_external_kernel_entry(name, backend) == (k, True)
        finally:
            etl.unregister_external_kernel(name)


def test_get_external_kernel_unchanged_for_device_registrations():
    """``get_external_kernel`` still returns the bare callable — never the
    entry tuple — with unchanged slot resolution (exact slot -> default
    slot -> ``None``), for device registrations too."""

    def default_fn(x):
        return x

    def dev_fn(x):
        return x + 1

    etl.register_external_kernel("dr_get_default", default_fn)
    etl.register_external_kernel(
        "dr_get_device", dev_fn, backend="iree", device_resident=True
    )
    try:
        assert etl.get_external_kernel("dr_get_default") is default_fn
        assert etl.get_external_kernel("dr_get_default", "iree") is default_fn  # fallback
        assert etl.get_external_kernel("dr_get_device", "iree") is dev_fn  # not a tuple
        assert etl.get_external_kernel("dr_get_device") is None  # no default slot
    finally:
        etl.unregister_external_kernel("dr_get_default")
        etl.unregister_external_kernel("dr_get_device")
    assert etl.get_external_kernel("dr_get_ghost") is None


# ---------------------------------------------------------------------------
# Trace-time contract
# ---------------------------------------------------------------------------


def test_external_call_builds_callback_effect_op_with_declared_specs():
    captured = {}

    def f(x):
        y = etl.external_call(
            "k", x, result=etl.TensorSpec((2, 3), etl.float32)
        )
        captured["y"] = y
        return y

    graph = etl.trace(f, etl.TensorSpec((4,), etl.float32))
    ops = ops_of(graph)
    call_op = ops[0]
    assert call_op.name == "external_call"
    assert call_op.effect == "callback"
    assert call_op.attributes["name"] == "k"
    specs = call_op.attributes["result_specs"]
    assert len(specs) == 1
    assert specs[0].dtype == np.dtype("float32")
    assert specs[0].shape == (2, 3)

    # The SymbolicTensor wrapper agrees with the declared spec.
    sym = captured["y"]
    assert sym.dtype == np.dtype("float32")
    assert sym.shape == (2, 3)


def test_external_call_multi_output_tuple():
    def f(a):
        b, c = etl.external_call(
            "multi",
            a,
            result=(
                etl.TensorSpec((3,), etl.int64),
                etl.TensorSpec((3,), etl.float32),
            ),
        )
        return b, c

    graph = etl.trace(f, etl.TensorSpec((3,), etl.int64))
    call_op = ops_of(graph, "external_call")[0]
    assert len(call_op.results) == 2
    assert call_op.results[0].type.dtype == np.dtype("int64")
    assert call_op.results[1].type.dtype == np.dtype("float32")
    assert call_op.results[1].type.shape == (3,)


def test_registration_not_required_at_trace_time():
    """Building a graph must not require the kernel to be registered (it is a
    run-time concern — the graph may be built in a different process)."""
    def f(a):
        return etl.external_call("never_registered_yet", a, result=etl.TensorSpec((3,), etl.int64))

    graph = etl.trace(f, etl.TensorSpec((3,), etl.int64))
    assert ops_of(graph, "external_call")[0].attributes["name"] == "never_registered_yet"


def test_external_call_outside_trace_raises():
    with pytest.raises(etl.TraceError):
        etl.external_call("k", 1.0, result=etl.TensorSpec((2,), etl.float32))


def test_external_call_rejects_concrete_tensor_operand():
    tensor = etl.tensor(np.array([1.0, 2.0], dtype=np.float32))

    def g():
        return etl.external_call("k", tensor, result=etl.TensorSpec((2,), etl.float32))

    with pytest.raises(etl.TraceError) as exc:
        etl.trace(g)
    message = str(exc.value)
    assert "explicit input" in message
    assert "etl.constant" in message
    assert "etl.evaluate" in message


def test_external_call_rejects_bad_name():
    def f(a):
        return etl.external_call(42, a, result=etl.TensorSpec((3,), etl.int64))

    with pytest.raises(TypeError, match="name must be a non-empty str"):
        etl.trace(f, etl.TensorSpec((3,), etl.int64))


def test_external_call_rejects_bad_result_specs():
    def f(a):
        return etl.external_call("k", a, result=42)

    with pytest.raises(TypeError, match="result must be a TensorSpec"):
        etl.trace(f, etl.TensorSpec((3,), etl.int64))

    def g(a):
        return etl.external_call("k", a, result=())

    with pytest.raises(TypeError, match="non-empty"):
        etl.trace(g, etl.TensorSpec((3,), etl.int64))


def test_external_call_accepts_scalar_operands():
    """Python scalar operands are transparently promoted to 0-d constants."""

    def f(a):
        return etl.external_call("k", a, 2, result=etl.TensorSpec((3,), etl.int64))

    graph = etl.trace(f, etl.TensorSpec((3,), etl.int64))
    call_op = ops_of(graph, "external_call")[0]
    assert len(call_op.operands) == 2


# ---------------------------------------------------------------------------
# Numpy execution end-to-end
# ---------------------------------------------------------------------------


def test_kernel_output_flows_into_downstream_ops_bit_exact():
    def kernel(x, w):
        return np.dot(x, w) * 2.0

    etl.register_external_kernel("e2e_double_dot", kernel)
    try:
        @etl.defn
        def f(x, w):
            y = etl.external_call(
                "e2e_double_dot", x, w,
                result=etl.TensorSpec((2, 3), etl.float32),
            )
            return etl.add(y, 1.0)  # downstream op consumes the kernel output

        x = np.arange(4, dtype=np.float32).reshape(2, 2)
        w = np.ones((2, 3), dtype=np.float32)
        out = run_numpy(f, x, w)
        expected = np.dot(x, w) * 2.0 + 1.0
        np.testing.assert_array_equal(out, expected)  # bit-exact by construction
    finally:
        etl.unregister_external_kernel("e2e_double_dot")


def test_kernel_receives_numpy_arrays_in_operand_order():
    seen = {}

    def kernel(a, b):
        seen["a"] = a
        seen["b"] = b
        return a + b

    etl.register_external_kernel("e2e_arrays", kernel)
    try:
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([10, 20, 30], dtype=np.int64)
        out = run_numpy(
            lambda x, y: etl.external_call(
                "e2e_arrays", x, y, result=etl.TensorSpec((3,), etl.int64)
            ),
            a, b,
        )
        assert isinstance(seen["a"], np.ndarray)
        assert seen["a"].dtype == np.dtype("int64")
        np.testing.assert_array_equal(seen["a"], a)
        np.testing.assert_array_equal(seen["b"], b)
        np.testing.assert_array_equal(out, a + b)
    finally:
        etl.unregister_external_kernel("e2e_arrays")


def test_multi_output_kernel():
    def kernel(a):
        return a + 1, a * 2

    etl.register_external_kernel("e2e_multi", kernel)
    try:
        @etl.defn
        def f(a):
            b, c = etl.external_call(
                "e2e_multi", a,
                result=(
                    etl.TensorSpec((3,), etl.int64),
                    etl.TensorSpec((3,), etl.int64),
                ),
            )
            return b + c

        a = np.array([1, 2, 3], dtype=np.int64)
        out = run_numpy(f, a)
        np.testing.assert_array_equal(out, (a + 1) + (a * 2))
    finally:
        etl.unregister_external_kernel("e2e_multi")


def test_kernel_returning_single_tensor():
    def kernel(a):
        return etl.Tensor(a * 3)  # core.Tensor returns are accepted too

    etl.register_external_kernel("e2e_tensor_ret", kernel)
    try:
        a = np.array([1, 2], dtype=np.int64)
        out = run_numpy(
            lambda x: etl.external_call(
                "e2e_tensor_ret", x, result=etl.TensorSpec((2,), etl.int64)
            ),
            a,
        )
        np.testing.assert_array_equal(out, a * 3)
    finally:
        etl.unregister_external_kernel("e2e_tensor_ret")


def test_evaluate_derives_specs_and_dispatches():
    """etl.evaluate derives TensorSpecs from concrete args and runs the
    external call through the numpy backend."""

    def kernel(x):
        return np.sqrt(x)

    etl.register_external_kernel("e2e_sqrt", kernel)
    try:
        x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        out = etl.evaluate(
            lambda a: etl.external_call(
                "e2e_sqrt", a, result=etl.TensorSpec((3,), etl.float32)
            ),
            x,
        )
        np.testing.assert_array_equal(out.numpy(), np.sqrt(x))
    finally:
        etl.unregister_external_kernel("e2e_sqrt")


# ---------------------------------------------------------------------------
# Loud errors
# ---------------------------------------------------------------------------


def test_unregistered_kernel_raises_backend_error_naming_kernel():
    @etl.defn
    def f(a):
        return etl.external_call("ghost_kernel", a, result=etl.TensorSpec((3,), etl.int64))

    exe = etl.build(f, etl.TensorSpec((3,), etl.int64))
    with pytest.raises(etl.BackendError, match="ghost_kernel") as exc:
        etl.run(exe, np.array([1, 2, 3], dtype=np.int64))
    assert "register_external_kernel" in str(exc.value)


def test_kernel_output_count_mismatch():
    def kernel(a):
        return a, a  # two outputs, one declared

    etl.register_external_kernel("err_count", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.external_call("err_count", a, result=etl.TensorSpec((3,), etl.int64))

        exe = etl.build(f, etl.TensorSpec((3,), etl.int64))
        with pytest.raises(etl.BackendError, match="err_count") as exc:
            etl.run(exe, np.array([1, 2, 3], dtype=np.int64))
        assert "produced 2 output(s), expected 1" in str(exc.value)
    finally:
        etl.unregister_external_kernel("err_count")


def test_kernel_output_dtype_mismatch():
    def kernel(a):
        return a.astype(np.float64)  # declared int64

    etl.register_external_kernel("err_dtype", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.external_call("err_dtype", a, result=etl.TensorSpec((3,), etl.int64))

        exe = etl.build(f, etl.TensorSpec((3,), etl.int64))
        with pytest.raises(etl.BackendError, match="err_dtype") as exc:
            etl.run(exe, np.array([1, 2, 3], dtype=np.int64))
        assert "no silent dtype coercion" in str(exc.value)
    finally:
        etl.unregister_external_kernel("err_dtype")


def test_kernel_output_shape_mismatch():
    def kernel(a):
        return a[:-1]  # (2,) declared (3,)

    etl.register_external_kernel("err_shape", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.external_call("err_shape", a, result=etl.TensorSpec((3,), etl.int64))

        exe = etl.build(f, etl.TensorSpec((3,), etl.int64))
        with pytest.raises(etl.ShapeError, match="err_shape"):
            etl.run(exe, np.array([1, 2, 3], dtype=np.int64))
    finally:
        etl.unregister_external_kernel("err_shape")


def test_kernel_non_array_return():
    def kernel(a):
        return "not an array"

    etl.register_external_kernel("err_ret", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.external_call("err_ret", a, result=etl.TensorSpec((3,), etl.int64))

        exe = etl.build(f, etl.TensorSpec((3,), etl.int64))
        with pytest.raises(etl.BackendError, match="err_ret"):
            etl.run(exe, np.array([1, 2, 3], dtype=np.int64))
    finally:
        etl.unregister_external_kernel("err_ret")


# ---------------------------------------------------------------------------
# Transforms: explicit TransformError (never a silent fallback)
# ---------------------------------------------------------------------------


def _external_graph():
    @etl.defn
    def f(x, w):
        y = etl.external_call(
            "tr_k", x, w, result=etl.TensorSpec((2, 3), etl.float32)
        )
        return etl.sum(y)

    return etl.trace(
        f,
        etl.TensorSpec((2, 2), etl.float32),
        etl.TensorSpec((2, 3), etl.float32),
    )


def test_grad_raises_transform_error():
    with pytest.raises(etl.TransformError, match="external_call"):
        etl.grad(_external_graph())


def test_vjp_raises_transform_error():
    with pytest.raises(etl.TransformError, match="external_call"):
        etl.vjp(_external_graph())


def test_jvp_raises_transform_error():
    with pytest.raises(etl.TransformError, match="external_call"):
        etl.jvp(
            _external_graph(),
            tangents=(
                etl.TensorSpec((2, 2), etl.float32),
                etl.TensorSpec((2, 3), etl.float32),
            ),
        )


def test_vmap_raises_transform_error():
    with pytest.raises(etl.TransformError, match="external_call"):
        etl.vmap(_external_graph(), in_axes=(0, 0))


# ---------------------------------------------------------------------------
# Control flow: works inside cond/while_loop bodies (numpy)
# ---------------------------------------------------------------------------


def test_external_call_inside_cond_body():
    def kernel(a):
        return a * 4.0

    etl.register_external_kernel("cf_kernel", kernel)
    try:
        @etl.defn
        def f(p, a):
            return etl.cond(
                p,
                lambda: etl.external_call(
                    "cf_kernel", a, result=etl.TensorSpec((2, 2), etl.float32)
                ),
                lambda: a * 3.0,
            )

        a = np.arange(4, dtype=np.float32).reshape(2, 2)
        assert np.array_equal(
            run_numpy(f, np.array(True), a), a * 4.0
        )
        assert np.array_equal(
            run_numpy(f, np.array(False), a), a * 3.0
        )
    finally:
        etl.unregister_external_kernel("cf_kernel")


def test_external_call_inside_while_body():
    def kernel(c):
        return c * 2.0

    etl.register_external_kernel("cf_while_kernel", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.while_loop(
                lambda c: etl.sum(c) < 8.0,
                lambda c: etl.external_call(
                    "cf_while_kernel", c, result=etl.TensorSpec((2,), etl.float32)
                ),
                a,
            )

        a = np.array([1.0, 1.0], dtype=np.float32)
        # iterations: sum 2 -> 4 -> 8 (stop): c = [4, 4]
        out = run_numpy(f, a)
        np.testing.assert_array_equal(out, np.array([4.0, 4.0], dtype=np.float32))
    finally:
        etl.unregister_external_kernel("cf_while_kernel")


# ---------------------------------------------------------------------------
# Graph save/load
# ---------------------------------------------------------------------------


def test_graph_save_load_roundtrip(tmp_path):
    def kernel(x, w):
        return np.dot(x, w) * 2.0

    etl.register_external_kernel("persist_kernel", kernel)
    try:
        @etl.defn
        def f(x, w):
            y = etl.external_call(
                "persist_kernel", x, w,
                result=etl.TensorSpec((2, 3), etl.float32),
            )
            return etl.sum(y + 1.0)

        graph = etl.trace(
            f,
            etl.TensorSpec((2, 2), etl.float32),
            etl.TensorSpec((2, 3), etl.float32),
        )
        path = tmp_path / "ext.etl"
        graph.save(str(path))
        loaded = etl.Graph.load(str(path))
        exe = etl.load(etl.compile(etl.lower(loaded)))
        x = np.arange(4, dtype=np.float32).reshape(2, 2)
        w = np.ones((2, 3), dtype=np.float32)
        out = etl.run(exe, x, w)
        np.testing.assert_array_equal(out.numpy(), np.sum(np.dot(x, w) * 2.0 + 1.0))
    finally:
        etl.unregister_external_kernel("persist_kernel")


def test_loaded_graph_requires_reregistered_kernel(tmp_path):
    """The registry is never serialized — a loaded graph run with the kernel
    unregistered fails loudly, and succeeds after re-registration."""

    def kernel(a):
        return a + 1

    etl.register_external_kernel("persist_rereg", kernel)
    try:
        @etl.defn
        def f(a):
            return etl.external_call(
                "persist_rereg", a, result=etl.TensorSpec((3,), etl.int64)
            )

        graph = etl.trace(f, etl.TensorSpec((3,), etl.int64))
        path = tmp_path / "rereg.etl"
        graph.save(str(path))
        loaded = etl.Graph.load(str(path))
    finally:
        etl.unregister_external_kernel("persist_rereg")

    exe = etl.load(etl.compile(etl.lower(loaded)))
    a = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(etl.BackendError, match="persist_rereg"):
        etl.run(exe, a)
    etl.register_external_kernel("persist_rereg", kernel)
    try:
        out = etl.run(exe, a)
        np.testing.assert_array_equal(out.numpy(), a + 1)
    finally:
        etl.unregister_external_kernel("persist_rereg")


# ---------------------------------------------------------------------------
# Compiler backends: explicit BackendError (xla/tvm) vs iree segment split
# ---------------------------------------------------------------------------


def _compiler_graph():
    @etl.defn
    def f(x):
        return etl.external_call(
            "compiler_k", x, result=etl.TensorSpec((3,), etl.float32)
        )

    return etl.trace(f, etl.TensorSpec((3,), etl.float32))


def test_stablehlo_export_deferred_with_backend_error():
    from etl.backends.stablehlo import export

    with pytest.raises(etl.BackendError) as exc:
        export(_compiler_graph())
    message = str(exc.value)
    assert "external_call" in message  # names the op


def test_iree_lower_splits_external_call_graphs_xla_tvm_still_reject():
    """The iree adapter host-dispatches external-call graphs at lower()
    time by SPLITTING them into segment programs (round 2 — payload
    ``stablehlo-segments``); xla and tvm still reject with the round-1
    message. When an adapter is not installed it raises its own explicit
    BackendError (``pip install etl[iree]`` / plugin guidance) — that
    env-dependent error is skipped, the capability contract is what's
    under test."""
    graph = _compiler_graph()

    try:
        lowered = etl.lower(graph, backend="iree")
    except etl.BackendError as exc:
        if "external_call" not in str(exc):
            pytest.skip(f"iree adapter not available in this env: {exc}")
        raise
    assert lowered.backend == "iree"
    assert lowered.payload["format"] == "stablehlo-segments"
    assert lowered.payload["plan"]["segments"]
    assert lowered.payload["segments"]

    for backend in ("xla", "tvm"):
        with pytest.raises(etl.BackendError) as exc:
            etl.lower(graph, backend=backend)
        message = str(exc.value)
        if "external_call" not in message:
            pytest.skip(
                f"{backend} adapter not available in this env: {message}"
            )
        assert "host-dispatch" in message
        assert "not yet wired" in message
